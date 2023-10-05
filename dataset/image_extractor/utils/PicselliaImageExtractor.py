import os

import cv2
from picsellia import Rectangle
from picsellia.types.enums import TagTarget

from abstract_processor.processor import AbstractProcessor


class PicselliaImageExtractor(AbstractProcessor):
    def __init__(self):
        super().__init__()
        context = self.job.sync()["dataset_version_processing_job"]

        self.datalake = self.client.get_datalake(
            name=context["parameters"].get("datalake_name", "default")
        )
        self.dataset_version = self.client.get_dataset_version_by_id(
            id=context["input_dataset_version_id"]
        )
        self.dataset_version_folder = self.dataset_version.version

        self.extracted_dataset_version = self.client.get_dataset_version_by_id(
            id=context["output_dataset_version_id"]
        )
        self.extracted_dataset_version_folder = self.extracted_dataset_version.version

    def process(self):
        self._download_ds()
        self._process_images()
        self._upload_images_to_extracted_ds()

    def _download_ds(self):
        self.dataset_version.download(target_path=self.dataset_version_folder)

    def _process_images(self):
        for image_filename in os.listdir(self.dataset_version_folder):
            self._process_image(image_filename)

    def _process_image(self, image_filename: str):
        image_filepath = os.path.join(self.dataset_version_folder, image_filename)
        image = cv2.imread(image_filepath)
        asset = self.dataset_version.find_asset(filename=image_filename)

        current_bbox = 0
        for annotation in asset.list_annotations():
            for rectangle in annotation.list_rectangles():
                current_bbox += 1
                self._extract(image, rectangle, current_bbox, image_filename)

    def _extract(
        self,
        image: cv2.imread,
        rectangle: Rectangle,
        current_bbox: int,
        image_filename: str,
    ):
        extracted_image = image[
            rectangle.y : rectangle.y + rectangle.h,
            rectangle.x : rectangle.x + rectangle.w,
        ]
        if extracted_image.shape[0] == 0 or extracted_image.shape[1] == 0:
            return
        label_folder = os.path.join(
            self.extracted_dataset_version_folder, rectangle.label.name
        )
        os.makedirs(label_folder, exist_ok=True)

        new_filename = f"{os.path.splitext(image_filename)[0]}_{rectangle.label.name}_{current_bbox}.{image_filename.split('.')[-1]}"
        new_filepath = os.path.join(label_folder, new_filename)

        cv2.imwrite(new_filepath, extracted_image)

    def _upload_images_to_extracted_ds(self):
        for label_folder in os.listdir(self.extracted_dataset_version_folder):
            full_label_folder_path = os.path.join(
                self.extracted_dataset_version_folder, label_folder
            )
            if os.path.isdir(full_label_folder_path):
                filepaths = [
                    os.path.join(full_label_folder_path, file)
                    for file in os.listdir(full_label_folder_path)
                ]
                data = self.datalake.upload_data(
                    filepaths=filepaths, tags=[label_folder]
                )
                adding_job = self.extracted_dataset_version.add_data(
                    data=data, tags=[label_folder]
                )
                adding_job.wait_for_done()
        conversion_job = self.extracted_dataset_version.convert_tags_to_classification(
            tag_type=TagTarget.ASSET,
            tags=self.extracted_dataset_version.list_asset_tags(),
        )
        conversion_job.wait_for_done()
