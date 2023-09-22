import os

import cv2
from picsellia import Client
from picsellia.types.enums import InferenceType, TagTarget


class PicselliaImageExtractor:
    def __init__(self, api_token, dataset_id, dataset_version_id, datalake_name):
        self.client = Client(api_token=api_token)

        self.datalake = self.client.get_datalake(name=datalake_name)

        self.dataset = self.client.get_dataset_by_id(id=dataset_id)

        self.dataset_version = self.client.get_dataset_version_by_id(
            id=dataset_version_id
        )
        self.dataset_version_folder = None

        self.extracted_dataset_version_folder = None
        self.extracted_dataset_version = None

    def download_ds(self):
        self.dataset_version_folder = self.dataset_version.version
        self.dataset_version.download(target_path=self.dataset_version_folder)

    def create_extracted_ds(self):
        self.extracted_dataset_version_folder = (
            f"extracted_{self.dataset_version_folder}"
        )
        os.makedirs(self.extracted_dataset_version_folder, exist_ok=True)
        self.extracted_dataset_version = self.dataset.create_version(
            version=self.extracted_dataset_version_folder,
            type=InferenceType.CLASSIFICATION,
        )

    def process_images(self):
        for image_filename in os.listdir(self.dataset_version_folder):
            self.process_image(image_filename)

    def process_image(self, image_filename):
        image_filepath = os.path.join(self.dataset_version_folder, image_filename)
        image = cv2.imread(image_filepath)
        asset = self.dataset_version.find_asset(filename=image_filename)

        current_bbox = 0
        for annotation in asset.list_annotations():
            for rectangle in annotation.list_rectangles():
                current_bbox += 1
                self.extract(image, rectangle, current_bbox, image_filename)

    def extract(self, image, rectangle, current_bbox, image_filename):
        extracted_image = image[
            rectangle.y : rectangle.y + rectangle.h,
            rectangle.x : rectangle.x + rectangle.w,
        ]

        label_folder = os.path.join(
            self.extracted_dataset_version_folder, rectangle.label.name
        )
        os.makedirs(label_folder, exist_ok=True)

        new_filename = f"{os.path.splitext(image_filename)[0]}_{rectangle.label.name}_{current_bbox}.{image_filename.split('.')[-1]}"
        new_filepath = os.path.join(label_folder, new_filename)

        cv2.imwrite(new_filepath, extracted_image)

    def upload_images_to_extracted_ds(self):
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
