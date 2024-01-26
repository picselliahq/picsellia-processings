from picsellia import Client
from uuid import uuid4
from picsellia.exceptions import ResourceNotFoundError, InsufficientResourcesError, PicselliaError
from picsellia.types.enums import InferenceType
from picsellia.sdk.asset import Asset
from picsellia.sdk.dataset import DatasetVersion
from picsellia.sdk.annotation import Annotation
from picsellia.sdk.label import Label
from typing import List
import tqdm
import os
from PIL import Image
import numpy as np
import logging
import urllib.request
from io import BytesIO
from scipy.special import softmax


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
        self.parameters = parameters

    # Coherence Checks

    def _type_coherence_check(self) -> bool:
        assert self.dataset_object.type == self.model_object.type, PicselliaError(
            f"Can't run pre-annotation job on a {self.dataset_object.type} with {self.model_object.type} model.")

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

    def _check_model_file_sanity(self, ) -> None:
        try:
            self.model_object.get_file('model-latest')
        except ResourceNotFoundError as e:
            raise ResourceNotFoundError(
                f"Can't run a pre-annotation job with this model, expected a 'model-latest' file")

    def _check_model_type_sanity(self, ) -> None:
        if self.model_object.type == InferenceType.NOT_CONFIGURED:
            raise PicselliaError(f"Can't run pre-annotation job, {self.model_object.name} type not configured.")

    def model_sanity_check(self, ) -> None:
        self._check_model_file_sanity()
        self._check_model_type_sanity()
        logging.info(f"Model {self.model_object.name} is sane.")

    # Utilities

    def _is_labelmap_starting_at_zero(self, ) -> bool:
        return '0' in self.model_infos["labels"].keys()

    def _set_dataset_version_type(self, ) -> None:
        self.dataset_object.set_type(
            self.model_object.type
        )
        logging.info(
            f"Setting dataset {self.dataset_object.name}/{self.dataset_object.version} to type {self.model_object.type}")

    def _get_model_labels_name(self, ) -> List[str]:
        self.model_infos = self.model_object.sync()
        if "labels" not in self.model_infos.keys():
            raise InsufficientResourcesError(f"Can't find labelmap for model {self.model_object.name}")
        if not isinstance(self.model_infos["labels"], dict):
            raise InsufficientResourcesError(
                f"Invalid LabelMap type, expected 'dict', got {type(self.model_infos['labels'])}")
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

    def _download_model_weights(self, ):
        model_weights = self.model_object.get_file('model-latest')
        model_weights.download()
        cwd = os.getcwd()
        self.model_weights_path = os.path.join(cwd, model_weights.filename)
        logging.info(f"{self.model_object.name}/{self.model_object.version} weights downloaded.")

    def _load_VIT_ONNX_model(self, ):
        try:
            import onnxruntime as ort
            self.model = ort.InferenceSession(self.model_weights_path)
            logging.info("Inference session loaded.")
        except Exception as e:
            raise PicselliaError(f"Impossible to load onnx model located at: {self.model_weights_path}")

    def setup_preannotation_job(self, ):
        logging.info(
            f"Setting up the Pre-annotation Job for dataset {self.dataset_object.name}/{self.dataset_object.version} with model {self.model_object.name}/{self.model_object.version}")
        self.model_sanity_check()
        if self.dataset_object.type == InferenceType.NOT_CONFIGURED:
            self._set_dataset_version_type()
            self._create_labels()
        else:
            self._type_coherence_check()
            self._labels_coherence_check()
        self.labels_to_detect = list(set(self.model_labels_name).intersection(self.dataset_labels_name))
        self._download_model_weights()
        self._load_VIT_ONNX_model()

    def preprocess(self, url):
        # Open the image from the URL
        with urllib.request.urlopen(url) as url:
            image = Image.open(BytesIO(url.read()))
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = image.resize((224, 224))  # Example size, adjust to your model's input
        image = np.array(image).astype('float32')
        image = np.transpose(image, (2, 0, 1))  # Change HWC to CHW
        image = image / 255  # Normalize
        image = np.expand_dims(image, axis=0)
        return image

    def interpret_output(self, outputs):
        outputs = softmax(outputs)
        # This will vary depending on your model's output
        print(outputs[0])
        class_index = np.argmax(outputs[0])
        return class_index

    def run_onnx(self, image_url):
        try:
            input_image = self.preprocess(image_url)
            inputs = {self.model.get_inputs()[0].name: input_image}
            outputs = self.model.run(None, inputs)
            return self.interpret_output(outputs)
        except Exception as e:
            print(e)
            logging.info(f"Could not perform prediction: {str(e)}")
            return None, None

    def _format_and_save_classification(self, asset: Asset, class_index: int) -> None:

        #  Convert predictions to Picsellia format
        annotation: Annotation = asset.create_annotation(duration=0.0)
        class_name = self.model_labels_name[class_index]
        label: Label = self.dataset_object.get_label(name=class_name)
        print("Label name: ", label.name)
        annotation.create_classification(label)
        logging.info(f"Asset: {asset.filename} pre-annotated.")

    def preannotate(self, ):
        dataset_size = self.dataset_object.sync()["size"]
        if not "batch_size" in self.parameters:
            batch_size = 8
        else:
            batch_size = self.parameters["batch_size"]
        batch_size = batch_size if dataset_size > batch_size else dataset_size
        total_batch_number = self.dataset_object.sync()["size"] // batch_size
        for batch_number in tqdm.tqdm(range(total_batch_number)):
            assets = self.dataset_object.list_assets(limit=batch_size, offset=batch_number * batch_size)
            url_list = [asset.sync()["data"]["presigned_url"] for asset in assets]
            for url, asset in list(zip(url_list, assets)):
                class_index = self.run_onnx(url)
                if class_index is not None:
                    if self.dataset_object.type == InferenceType.CLASSIFICATION:
                        self._format_and_save_classification(asset, class_index)
                else:
                    logging.info(f"Asset: {asset.filename} was not pre-annotated.")