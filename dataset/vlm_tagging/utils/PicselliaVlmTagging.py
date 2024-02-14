import os
import sys

from picsellia import Client
from utils.inference_utils import MiniGPT4Predictor


class PicselliaVlmTagging:
    def __init__(
        self,
        client: Client,
        model_version_id: str,
        dataset_version_id: str,
        list_tags: list=None,
        checkpoint_dir="checkpoints",
        checkpoint_filename="model-latest",
    ):
        self.client = client

        self.dataset_version = self.client.get_dataset_version_by_id(
            id=dataset_version_id
        )
        self.dataset_version_folder = self.dataset_version.version
        self.dataset_version.download(target_path=self.dataset_version_folder)

        self.picsellia_tags_name = self.create_tags(list_tags)
        print(self.picsellia_tags_name)

        self.model_version = self.client.get_model_version_by_id(id=model_version_id)
        checkpoint = self.model_version.get_file(name=checkpoint_filename)
        checkpoint.download(target_path=checkpoint_dir)

        self.predictor = self.load_predictor(checkpoint_dir, checkpoint.filename)

        self.prompt = self.get_prompt()

    def load_predictor(self, checkpoint_dir: str, checkpoint_filename: str):
        script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
        return MiniGPT4Predictor(
            checkpoint_path=f"{script_directory}/{checkpoint_dir}/{checkpoint_filename}"
        )

    def get_prompt(self):
        return (
            f"[vqa] Is it " + " or ".join([f'"{tag}"' for tag in self.picsellia_tags_name.keys()]) + "?" + " Analyze the image and assign the correct tag to this image."
        )

    def create_ds(self, created_dataset_name: str):
        created_dataset_version, job = self.dataset_version.fork(
            version=created_dataset_name,
            description="Tagged dataset",
        )
        job.wait_for_done()
        return created_dataset_version

    def create_tags(self, list_tags: list):
        if list_tags:
            for tag_name in list_tags:
                self.dataset_version.get_or_create_asset_tag(name=tag_name)
        return {k.name: k for k in self.dataset_version.list_asset_tags()}

    def _add_tags_to_assets(self):
        for asset in self.dataset_version.list_assets():
            asset.add_tags(
                tags=self.predictor.get_tags(
                    image_path=os.path.join(
                        self.dataset_version_folder, asset.filename
                    ),
                    prompt=self.prompt,
                    label_matching=self.picsellia_tags_name,
                )
            )
