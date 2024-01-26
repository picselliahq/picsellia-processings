import requests
from PIL import Image
from picsellia import Client
from uuid import uuid4
from picsellia.exceptions import (
    ResourceNotFoundError,
    InsufficientResourcesError,
    PicselliaError,
)
from picsellia.types.enums import InferenceType
from picsellia.sdk.asset import Asset
from picsellia.sdk.dataset import DatasetVersion
from picsellia.sdk.annotation import Annotation
from picsellia.sdk.label import Label
from typing import List, Tuple
import tqdm
import os
import cv2
import numpy as np

import logging
import onnxruntime as ort


class YoloxInferenceResult:
    def __init__(self, boxes: np.array, scores: np.array, labels: np.array):
        self.boxes = boxes
        self.scores = scores
        self.labels = labels


class PreAnnotator:
    """ """

    def __init__(
        self,
        client: Client,
        dataset_version_id: uuid4,
        model_version_id: uuid4,
        parameters: dict,
    ) -> None:
        self.client = client
        self.dataset_object: DatasetVersion = self.client.get_dataset_version_by_id(
            dataset_version_id
        )
        self.model_object = self.client.get_model_version_by_id(model_version_id)
        self.parameters = parameters

    # Coherence Checks

    def _type_coherence_check(self) -> bool:
        assert self.dataset_object.type == self.model_object.type, PicselliaError(
            f"Can't run pre-annotation job on a {self.dataset_object.type} with {self.model_object.type} model."
        )

    def _labels_coherence_check(self) -> bool:
        """
        Assert that at least one label from the model labelmap is contained in the dataset version.
        """
        self.model_labels_name = self._get_model_labels_name()
        self.dataset_labels_name = [
            label.name for label in self.dataset_object.list_labels()
        ]

        intersecting_labels = set(self.model_labels_name).intersection(
            self.dataset_labels_name
        )
        logging.info(
            f"Pre-annotation Job will only run on classes: {list(intersecting_labels)}"
        )
        return len(intersecting_labels) > 0

    # Sanity check

    def _check_model_file_sanity(
        self,
    ) -> None:
        try:
            model_file = self.model_object.get_file("model-latest")
            assert model_file.filename.endswith(".onnx")

        except ResourceNotFoundError as e:
            raise ResourceNotFoundError(
                "Can't run a pre-annotation job with this model, expected a 'model-latest' file"
            )

        except AssertionError as e:
            raise ResourceNotFoundError(
                "Can't run a pre-annotation job with this model, expected an onnx file"
            )

    def _check_model_type_sanity(
        self,
    ) -> None:
        if self.model_object.type == InferenceType.NOT_CONFIGURED:
            raise PicselliaError(
                f"Can't run pre-annotation job, {self.model_object.name} type not configured."
            )

    def model_sanity_check(
        self,
    ) -> None:
        self._check_model_file_sanity()
        self._check_model_type_sanity()
        logging.info(f"Model {self.model_object.name} is sane.")

    # Utilities

    def _is_labelmap_starting_at_zero(
        self,
    ) -> bool:
        return "0" in self.model_infos["labels"].keys()

    def _set_dataset_version_type(
        self,
    ) -> None:
        self.dataset_object.set_type(self.model_object.type)
        logging.info(
            f"Setting dataset {self.dataset_object.name}/{self.dataset_object.version} to type {self.model_object.type}"
        )

    def _get_model_labels_name(
        self,
    ) -> List[str]:
        self.model_infos = self.model_object.sync()
        if "labels" not in self.model_infos.keys():
            raise InsufficientResourcesError(
                f"Can't find labelmap for model {self.model_object.name}"
            )
        if not isinstance(self.model_infos["labels"], dict):
            raise InsufficientResourcesError(
                f"Invalid LabelMap type, expected 'dict', got {type(self.model_infos['labels'])}"
            )
        model_labels = list(self.model_infos["labels"].values())
        return model_labels

    def _create_labels(
        self,
    ) -> None:
        if not hasattr(self, "model_labels_name"):
            self.model_labels_name = self._get_model_labels_name()
        for label in tqdm.tqdm(self.model_labels_name):
            self.dataset_object.create_label(name=label)
        self.dataset_labels_name = [
            label.name for label in self.dataset_object.list_labels()
        ]
        logging.info(f"Labels :{self.dataset_labels_name} created.")

    def _download_model_weights(
        self,
    ):
        model_weights = self.model_object.get_file("model-latest")
        model_weights.download()
        cwd = os.getcwd()
        self.onnx_model_weights_path = os.path.join(cwd, model_weights.filename)
        logging.info(
            f"{self.model_object.name}/{self.model_object.version} onnx model downloaded."
        )

    def _load_yolov8_model(
        self,
    ):
        try:
            import onnx

            self.model = onnx.load(self.onnx_model_weights_path)
            onnx.checker.check_model(self.model)
            self.onnx_inference_ort = ort.InferenceSession(self.onnx_model_weights_path)

            logging.info("Model loaded in memory.")
        except Exception as e:
            raise PicselliaError(
                f"Impossible to load saved model located at: {self.onnx_model_weights_path}"
            )

    def setup_preannotation_job(
        self,
    ):
        logging.info(
            f"Setting up the Pre-annotation Job for dataset {self.dataset_object.name}/{self.dataset_object.version} with model {self.model_object.name}/{self.model_object.version}"
        )
        self.model_sanity_check()
        if self.dataset_object.type == InferenceType.NOT_CONFIGURED:
            self._set_dataset_version_type()
            self._create_labels()
        else:
            self._type_coherence_check()
            self._labels_coherence_check()
        self.labels_to_detect = list(
            set(self.model_labels_name).intersection(self.dataset_labels_name)
        )
        self._download_model_weights()
        self._load_yolov8_model()

    def rescale_normalized_segment(
        self, segment: List, width: int, height: int
    ) -> List[int]:
        segment = [
            [
                int(box[0] * height),
                int(box[1] * width),
            ]
            for box in segment
        ]
        return segment

    def _format_picsellia_polygons(
        self, asset: Asset, predictions: np.array
    ) -> Tuple[List, List, List, List]:
        if predictions.masks is None:
            return []
        polygons = predictions.masks.xy
        casted_polygons = list(map(lambda polygon: polygon.astype(int), polygons))
        return list(map(lambda polygon: polygon.tolist(), casted_polygons))

    def _format_and_save_rectangles(
        self,
        asset: Asset,
        yolox_inference_result: YoloxInferenceResult,
        confidence_treshold: float = 0.1,
    ) -> None:
        boxes = yolox_inference_result.boxes
        scores = yolox_inference_result.scores
        labels = yolox_inference_result.labels

        rectangle_list = []
        nb_box_limit = 100
        if len(boxes) < nb_box_limit:
            nb_box_limit = len(boxes)
        if len(boxes) > 0:
            annotation: Annotation = asset.create_annotation(duration=0.0)
        else:
            return
        for i in range(nb_box_limit):
            if scores[i] >= confidence_treshold:
                try:
                    label: Label = self.dataset_object.get_label(
                        name=self.model_labels_name[int(labels[i])]
                    )
                    e = boxes[i].tolist()
                    box = [
                        int(e[0]),
                        int(e[1]),
                        int((e[2] - e[0])),
                        int((e[3] - e[1])),
                        label,
                    ]
                    rectangle_list.append(tuple(box))
                except ResourceNotFoundError as e:
                    print(e)
                    continue
        if len(rectangle_list) > 0:
            annotation.create_multiple_rectangles(rectangle_list)
            logging.info(f"Asset: {asset.filename} pre-annotated.")

    def _format_and_save_polygons(
        self, asset: Asset, predictions: dict, confidence_threshold: float
    ) -> None:
        scores = predictions.boxes.conf.cpu().numpy()
        labels = predictions.boxes.cls.cpu().numpy().astype(np.int16)
        #  Convert predictions to Picsellia format
        masks = self._format_picsellia_polygons(asset=asset, predictions=predictions)
        polygons_list = []
        nb_polygons_limit = 100
        if len(masks) < nb_polygons_limit:
            nb_box_limit = len(masks)
        if len(masks) > 0:
            annotation: Annotation = asset.create_annotation(duration=0.0)
        else:
            return
        for i in range(nb_box_limit):
            if scores[i] >= confidence_threshold:
                try:
                    label: Label = self.dataset_object.get_label(
                        name=predictions.names[labels[i]]
                    )
                    polygons_list.append((masks[i], label))
                except ResourceNotFoundError as e:
                    print(e)
                    continue
        if len(polygons_list) > 0:
            annotation.create_multiple_polygons(polygons_list)
            logging.info(f"Asset: {asset.filename} pre-annotated.")

    def preannotate(self, confidence_threshold: float = 0.5):
        dataset_size = self.dataset_object.sync()["size"]
        if not "batch_size" in self.parameters:
            batch_size = 8
        else:
            batch_size = self.parameters["batch_size"]
        batch_size = batch_size if dataset_size > batch_size else dataset_size
        total_batch_number = self.dataset_object.sync()["size"] // batch_size
        for batch_number in tqdm.tqdm(range(total_batch_number)):
            assets = self.dataset_object.list_assets(
                limit=batch_size, offset=batch_number * batch_size
            )

            for asset in assets:
                image_data = self._preprocess_picsellia_asset(asset=asset)
                yolox_inference_result = self._make_yolox_onnx_inference(
                    image_data=image_data
                )

                if len(asset.list_annotations()) == 0:
                    if len(yolox_inference_result.boxes) > 0:
                        if self.dataset_object.type == InferenceType.OBJECT_DETECTION:
                            self._format_and_save_rectangles(
                                asset, yolox_inference_result, confidence_threshold
                            )

    def _make_yolox_onnx_inference(
        self, image_data: np.array
    ) -> YoloxInferenceResult | None:
        input_shape = (640, 640)
        img, ratio = self._preprocess_yolox_input(image_data, input_shape)

        ort_inputs = {self.onnx_inference_ort.get_inputs()[0].name: img[None, :, :, :]}
        output = self.onnx_inference_ort.run(None, ort_inputs)
        predictions = self._postprocess_yolox_input(output[0], input_shape)[0]

        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
        boxes_xyxy /= ratio
        dets = self._multiclass_nms_class_aware(
            boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1
        )
        if dets is not None:
            final_boxes, final_scores, final_cls_inds = (
                dets[:, :4],
                dets[:, 4],
                dets[:, 5],
            )

            return YoloxInferenceResult(
                boxes=final_boxes, scores=final_scores, labels=final_cls_inds
            )

        return None

    def _preprocess_picsellia_asset(self, asset: Asset) -> np.array:
        image = Image.open(
            requests.get(asset.sync()["data"]["presigned_url"], stream=True).raw
        )
        return np.array(image)

    def _preprocess_yolox_input(self, img, input_size, swap=(2, 0, 1)):
        if len(img.shape) == 3:
            padded_img = (
                np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
            )
        else:
            padded_img = np.ones(input_size, dtype=np.uint8) * 114

        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r

    def _postprocess_yolox_input(self, outputs, img_size, p6=False):
        grids = []
        expanded_strides = []
        strides = [8, 16, 32] if not p6 else [8, 16, 32, 64]

        hsizes = [img_size[0] // stride for stride in strides]
        wsizes = [img_size[1] // stride for stride in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

        return outputs

    def _multiclass_nms_class_aware(self, boxes, scores, nms_thr, score_thr):
        """Multiclass NMS implemented in Numpy. Class-aware version."""
        final_dets = []
        num_classes = scores.shape[1]
        for cls_ind in range(num_classes):
            cls_scores = scores[:, cls_ind]
            valid_score_mask = cls_scores > score_thr
            if valid_score_mask.sum() == 0:
                continue
            else:
                valid_scores = cls_scores[valid_score_mask]
                valid_boxes = boxes[valid_score_mask]
                keep = self._nms(valid_boxes, valid_scores, nms_thr)
                if len(keep) > 0:
                    cls_inds = np.ones((len(keep), 1)) * cls_ind
                    dets = np.concatenate(
                        [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                    )
                    final_dets.append(dets)
        if len(final_dets) == 0:
            return None
        return np.concatenate(final_dets, 0)

    def _nms(self, boxes, scores, nms_thr):
        """Single class NMS implemented in Numpy."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= nms_thr)[0]
            order = order[inds + 1]

        return keep
