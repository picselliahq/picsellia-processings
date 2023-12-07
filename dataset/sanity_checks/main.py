import os
from picsellia import Client
from picsellia.types.enums import AnnotationFileType
import logging

from pycocotools.coco import COCO
from utils import (
    get_duplicate_filenames,
    get_duplicate_images,
    add_tags_to_duplicate_images,
    add_nbr_channel_byte_tags,
    get_area_outlier_filenames,
    log_results,
)

os.environ["PICSELLIA_SDK_CUSTOM_LOGGING"] = "True"
os.environ["PICSELLIA_SDK_DOWNLOAD_BAR_MODE"] = "2"
logging.getLogger("picsellia").setLevel(logging.INFO)

api_token = os.environ["api_token"]
organization_id = os.environ["organization_id"]
job_id = os.environ["job_id"]

if "host" not in os.environ:
    host = "https://app.picsellia.com"
else:
    host = os.environ["host"]

client = Client(api_token=api_token, organization_id=organization_id, host=host)
job = client.get_job_by_id(job_id)

context = job.sync()["dataset_version_processing_job"]
input_dataset_version_id = context["input_dataset_version_id"]
parameters = context["parameters"]
area_outlier_threshold = 4
duplicate_image_filenames = {}
dataset_path = "data"
cwd = os.getcwd()

dataset_version = client.get_dataset_version_by_id(input_dataset_version_id)
dataset_version.download(dataset_path)
annotation_file_path = dataset_version.export_annotation_file(
    annotation_file_type=AnnotationFileType.COCO,
    target_path=cwd,
    force_replace=True,
)

coco = COCO(annotation_file_path)

image_duplicates = get_duplicate_images(dataset_path=dataset_path)
if image_duplicates:
    duplicate_image_filenames = add_tags_to_duplicate_images(
        dataset_version=dataset_version, duplicates=image_duplicates
    )

filename_duplicates = get_duplicate_filenames(dataset_version=dataset_version)
if filename_duplicates:
    dup_assets = dataset_version.find_all_assets(filenames=filename_duplicates)
    dup_filename_tag = dataset_version.create_asset_tag(name="dup_filename")
    dup_assets.add_tags(dup_filename_tag)

channel_counts, byte_counts = add_nbr_channel_byte_tags(
    dataset_version=dataset_version, dataset_path=dataset_path
)
area_outlier_filenames = get_area_outlier_filenames(
    coco=coco, area_outlier_threshold=area_outlier_threshold
)

log_results(
    duplicate_image_filenames,
    filename_duplicates,
    byte_counts,
    channel_counts,
    area_outlier_filenames,
)
