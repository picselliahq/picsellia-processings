from picsellia import Client
from uuid import uuid4
from picsellia.exceptions import ResourceNotFoundError, InsufficientResourcesError, PicselliaError
from picsellia.types.enums import InferenceType
from picsellia.sdk.model import ModelVersion
from picsellia.sdk.asset import Asset
from picsellia.sdk.dataset import DatasetVersion
from picsellia.sdk.annotation import Annotation
from picsellia.sdk.label import Label
from typing import List, Tuple
import tqdm
import zipfile
import os
from PIL import Image
import numpy as np 
import requests
import logging



class PreAnnotator:
    """ 
    
    """
    def __init__(self, 
            client: Client,
            dataset_version_id: uuid4,
            model_version_id: uuid4,
            parameters: dict) -> None:
                
        self.client = client
        self.dataset_object: DatasetVersion = self.client.get_dataset_version_by_id(
            dataset_version_id
        )
        self.model_object = self.client.get_model_version_by_id(
            model_version_id
        )
        self.label_list = self.dataset_object.list_labels()
        self.parameters = parameters
    # Coherence Checks 

    def _type_coherence_check(self) -> bool:
        assert self.dataset_object.type == self.model_object.type, PicselliaError(f"Can't run pre-annotation job on a {self.dataset_object.type} with {self.model_object.type} model.")

    def _labels_coherence_check(self) -> bool:
        """ 
        Assert that at least one label from the model labelmap is contained in the dataset version.
        """
        self.model_labels_name = self._get_model_labels_name()
        self.dataset_labels_name = [label.name for label in self.dataset_object.list_labels()]

        intersecting_labels = set(self.model_labels_name).intersection(self.dataset_labels_name)
        logging.info(f"Pre-annotation Job will only run on classes: {list(intersecting_labels)}")
        return len(intersecting_labels) > 0

    # Sanity check 

    def _check_model_file_sanity(self,) -> None:
        try:
            self.model_object.get_file('model-latest')
        except ResourceNotFoundError as e:
            raise ResourceNotFoundError(f"Can't run a pre-annotation job with this model, expected a 'model-latest' file")
        
    def _check_model_type_sanity(self, ) -> None:
        if self.model_object.type == InferenceType.NOT_CONFIGURED:
            raise PicselliaError(f"Can't run pre-annotation job, {self.model_object.name} type not configured.")
        
    def model_sanity_check(self,) -> None:
        self._check_model_file_sanity()
        self._check_model_type_sanity()
        logging.info(f"Model {self.model_object.name} is sane.")


    
    # Utilities 

    def _is_labelmap_starting_at_zero(self,) -> bool:
        return '0' in self.model_infos["labels"].keys()
    
    def _set_dataset_version_type(self, ) -> None:
        self.dataset_object.set_type(
            self.model_object.type
        )
        logging.info(f"Setting dataset {self.dataset_object.name}/{self.dataset_object.version} to type {self.model_object.type}")

    def _get_model_labels_name(self, ) -> List[str]:
        self.model_infos = self.model_object.sync()
        if "labels" not in self.model_infos.keys():
            raise InsufficientResourcesError(f"Can't find labelmap for model {self.model_object.name}")
        if not isinstance(self.model_infos["labels"], dict):
            raise InsufficientResourcesError(f"Invalid LabelMap type, expected 'dict', got {type(self.model_infos['labels'])}")
        model_labels = list(self.model_infos["labels"].values())
        return model_labels
    
    def _create_labels(self, ) -> None:
        if not hasattr(self, 'model_labels_name'):
            self.model_labels_name = self._get_model_labels_name()
        for label in tqdm.tqdm(self.model_labels_name):
            self.dataset_object.create_label(
                name=label
            )
        self.dataset_labels_name = [label.name for label in self.dataset_object.list_labels()]
        logging.info(f"Labels :{self.dataset_labels_name} created.")

    def _download_model_weights(self,):
        model_weights = self.model_object.get_file('checkpoint-index-latest')
        model_weights.download()        
        cwd = os.getcwd()
        self.model_weights_path = os.path.join(cwd,model_weights.filename)
        logging.info(f"{self.model_object.name}/{self.model_object.version} weights downloaded.")

    def _load_yolov8_model(self,):
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_weights_path)
            logging.info("Model loaded in memory.")
        except Exception as e:
            raise PicselliaError(f"Impossible to load saved model located at: {self.model_weights_path}")
        



    def setup_preannotation_job(self,):
        logging.info(f"Setting up the Pre-annotation Job for dataset {self.dataset_object.name}/{self.dataset_object.version} with model {self.model_object.name}/{self.model_object.version}")
        self.model_sanity_check()
        if self.dataset_object.type == InferenceType.NOT_CONFIGURED:
            self._set_dataset_version_type()
            self._create_labels()
        else:
            self._type_coherence_check()
            self._labels_coherence_check()
        self.labels_to_detect = list(set(self.model_labels_name).intersection(self.dataset_labels_name))
        self._download_model_weights()
        self._load_yolov8_model()


    def rescale_normalized_segment(self, segment: List, width: int, height: int) -> List[int]:
        segment = [
            [
                int(box[0] * height),
                int(box[1] * width),
            ]
            for box in segment
        ]
        return segment

    
    def _format_picsellia_polygons(self, asset: Asset, predictions: np.array) -> Tuple[List, List, List, List]:
        if predictions.masks is None:
            return []
        polygons = predictions.masks.xy
        casted_polygons = list(
            map(lambda polygon: polygon.astype(int), polygons))
        return list(map(lambda polygon: polygon.tolist(), casted_polygons))

    
    def _format_and_save_rectangles(self, asset: Asset, prediction: List, confidence_treshold: float = 0.1) -> None:
        boxes = prediction.boxes.xyxyn.cpu().numpy()
        scores = prediction.boxes.conf.cpu().numpy()
        labels = prediction.boxes.cls.cpu().numpy().astype(np.int16)
        #  Convert predictions to Picsellia format

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
                    label = self._get_label_by_name(label_list=self.label_list, label_name=prediction.names[labels[i]])
                    e = boxes[i].tolist()
                    box = [
                        int(e[0] * asset.width),
                        int(e[1] * asset.height),
                        int((e[2] - e[0]) * asset.width),
                        int((e[3] - e[1]) * asset.height),
                    ]
                    box.append(label)
                    rectangle_list.append(tuple(box))
                except ResourceNotFoundError as e:
                    print(e)
                    continue
        if len(rectangle_list) > 0:
            annotation.create_multiple_rectangles(rectangle_list)
            logging.info(f"Asset: {asset.filename} pre-annotated.")

    def _format_and_save_polygons(self, asset: Asset, predictions: dict, confidence_threshold: float) -> None:
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
                    label = self._get_label_by_name(label_list=self.label_list, label_name=predictions.names[labels[i]])
                    polygons_list.append((masks[i], label))
                except ResourceNotFoundError as e:
                    print(e)
                    continue
        if len(polygons_list) > 0:
            annotation.create_multiple_polygons(polygons_list)
            logging.info(f"Asset: {asset.filename} pre-annotated.")

    def _format_and_save_classification(self, asset: Asset, predictions: dict, confidence_threshold: float) -> None:
        if len(self.dataset_object.list_labels()) != len(predictions.names):
            raise ValueError("The labelmaps don't have the same length. Please verify the dataset and pre-annotation model labelmaps.")

        prediction_index = np.argmax(predictions.probs).item()

        if predictions.probs[prediction_index] >= confidence_threshold:
            label_name = predictions.names[prediction_index]
            label = self._get_label_by_name(label_list=self.label_list, label_name=label_name)
            annotation = asset.create_annotation(duration=0.0)
            annotation.create_classification(label)

    def _get_label_by_name(self, label_list: List[Label], label_name: str) -> Label:
        for label in label_list:
            if label.name == label_name:
                return label

        raise ValueError(f"The label {label_name} does not exist in the labelmap.")

    def preannotate(self, confidence_threshold: float = 0.5):
        dataset_size = self.dataset_object.sync()["size"]
        
        if not "batch_size" in self.parameters:
            batch_size = 8
        else:
            batch_size = self.parameters["batch_size"]

        batch_size = batch_size if dataset_size > batch_size else dataset_size
        total_batch_number = self.dataset_object.sync()["size"] // batch_size

        for batch_number in tqdm.tqdm(range(total_batch_number)):
            assets = self.dataset_object.list_assets(limit=batch_size, offset=batch_number*batch_size)
            url_list = [asset.sync()["data"]["presigned_url"] for asset in assets]
            predictions = self.model(url_list)

            for asset, prediction in list(zip(assets, predictions)):
                if len(asset.list_annotations()) == 0:
                    if len(prediction) > 0:
                        if self.dataset_object.type == InferenceType.OBJECT_DETECTION:
                            self._format_and_save_rectangles(asset, prediction, confidence_threshold)
                        elif self.dataset_object.type == InferenceType.SEGMENTATION:
                            self._format_and_save_polygons(asset, prediction, confidence_threshold)
                        elif self.dataset_object.type == InferenceType.CLASSIFICATION:
                            self._format_and_save_classification(asset, prediction, confidence_threshold)
