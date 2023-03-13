from abc import ABC, abstractmethod

import numpy as np
from PIL import Image
import cv2
from picsellia.types.enums import InferenceType

class AbstractFormatter(ABC):
    @abstractmethod
    def format_output(self, raw_output, model_type):
        pass

    @abstractmethod
    def format_object_detection(self, raw_output):
        pass

    @abstractmethod
    def format_segmentation(self, raw_output):
        pass

    # @abstractmethod
    # def format_classification(self, raw_output):
    #     pass


class TensorflowFormatter(AbstractFormatter):
    def __init__(self, image_width, image_height):
        self.image_height = image_height
        self.image_width = image_width

    def format_output(self, raw_output: dict, model_type: InferenceType):
        if model_type == InferenceType.OBJECT_DETECTION:
            return self.format_object_detection(raw_output)
        elif model_type == InferenceType.SEGMENTATION:
            return self.format_segmentation(raw_output)
        # elif model_type == InferenceType.CLASSIFICATION:
        #     return self.format_classification(raw_output)

    def format_object_detection(self, raw_output):
        try:
            scores = (
                raw_output["detection_scores"].numpy()[0].astype(np.float).tolist()
            )
            boxes = self._postprocess_boxes(
                raw_output["detection_boxes"].numpy()[0].astype(np.float).tolist()
            )
            classes = (
                raw_output["detection_classes"].numpy()[0].astype(np.float).tolist()
            )
        except Exception:
            scores = raw_output["output_1"].numpy()[0].astype(np.float).tolist()
            boxes = self._postprocess_boxes(
                raw_output.as_numpy("output_0")[0].astype(np.float).tolist()
            )
            classes = raw_output["output_2"].numpy()[0].astype(np.int16).tolist()
        response = {
            "detection_scores": scores,
            "detection_boxes": boxes,
            "detection_classes": classes,
        }
        return response

    def format_segmentation(self, raw_output):
        scores = (
            raw_output["detection_scores"].numpy()[0].astype(np.float).tolist()[:10]
        )
        boxes = self._postprocess_boxes(
                raw_output["detection_boxes"].numpy()[0].astype(np.float).tolist()
            )
        masks = self._postprocess_masks(
            detection_masks = raw_output["detection_masks"].numpy()[0].astype(np.float).tolist()[:10],
            resized_detection_boxes=boxes,
            mask_threshold=0.4,
        )
        classes = (
            raw_output["detection_classes"].numpy()[0].astype(np.float).tolist()[:10]
        )
        response = {
            "detection_scores": scores,
            "detection_boxes": boxes,
            "detection_masks": masks,
            "detection_classes": classes,
        }

        return response

    def _postprocess_boxes(self, detection_boxes: list) -> list:
        return [
            [
                int(e[1] * self.image_width),
                int(e[0] * self.image_height),
                int((e[3] - e[1]) * self.image_width),
                int((e[2] - e[0]) * self.image_height),
            ]
            for e in detection_boxes
        ]

    def _postprocess_masks(
        self,
        detection_masks: list,
        resized_detection_boxes: list,
        mask_threshold: float = 0.5,
    ) -> list:
        list_mask = []
        for idx, detection_mask in enumerate(detection_masks):

            # background_mask with all black=0
            mask = np.zeros((self.image_height, self.image_width))
            print(resized_detection_boxes[idx])
            # Get normalised bbox coordinates
            xmin, ymin, w, h = resized_detection_boxes[idx]

            xmax = xmin + w 
            ymax = ymin + h 

            # Define bbox height and width
            bbox_height, bbox_width = h, w

            # Resize 'detection_mask' to bbox size
            bbox_mask = np.array(
                Image.fromarray(np.array(detection_mask) * 255).resize(
                    size=(bbox_width, bbox_height), resample=Image.BILINEAR
                )
                # Image.NEAREST is fastest and no weird artefacts
            )
            # Insert detection_mask into image.size np.zeros((height, width)) background_mask
            assert bbox_mask.shape == mask[ymin:ymax, xmin:xmax].shape
            mask[ymin:ymax, xmin:xmax] = bbox_mask
            if (
                mask_threshold > 0
            ):  # np.where(mask != 1, 0, mask)  # in case threshold is used to have other values (0)
                mask = np.where(np.abs(mask) > mask_threshold * 255, 1, mask)
                mask = np.where(mask != 1, 0, mask)

            try:
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_TC89_KCOS,
                )
                to_add = (
                    contours[len(contours) - 1][::1]
                    .reshape(
                        contours[len(contours) - 1][::1].shape[0],
                        contours[len(contours) - 1][::1].shape[2],
                    )
                    .tolist()
                )
                list_mask.append(to_add)
            except Exception:
                pass  # No contours

        return list_mask

