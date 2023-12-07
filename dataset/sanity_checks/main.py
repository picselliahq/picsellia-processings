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

dataset_version = client.get_dataset_version_by_id(input_dataset_version_id)
dataset_version.download(dataset_path)
annotation_file_path = dataset_version.export_annotation_file(
    annotation_file_type=AnnotationFileType.COCO,
    target_path=os.path.join("annotations"),
    force_replace=True,
)

coco = COCO(annotation_file_path)

image_duplicates = get_duplicate_images(dataset_path=dataset_path)
if image_duplicates:
    duplicate_image_filenames = add_tags_to_duplicate_images(
        dataset_version=dataset_version, duplicates=image_duplicates
    )

filename_duplicates = get_duplicate_filenames(dataset_version=dataset_version)
channel_counts, byte_counts = add_nbr_channel_byte_tags(
    dataset_version=dataset_version, dataset_path=dataset_path
)
area_outlier_filenames = get_area_outlier_filenames(
    coco=coco, area_outlier_threshold=area_outlier_threshold
)


# log summary of results
if duplicate_image_filenames:
    logging.info(f"duplicate images are: {duplicate_image_filenames}")
if filename_duplicates:
    logging.info(f"duplicate filenames are: {filename_duplicates}")

logging.info(f"Number of images per nbr_bytes:")
for nbr_bytes, count in byte_counts.items():
    logging.info(f"{nbr_bytes}: {count} images")

logging.info("Number of images per nbr_channels:")
for nbr_channels, count in channel_counts.items():
    logging.info(f"{nbr_channels}: {count} images")

len_outlier_files = len(area_outlier_filenames)
logging.info(f"you have {len_outlier_files} image(s) with outlier areas")
if len_outlier_files > 0:
    logging.info(f"filenames with outlier areas: {area_outlier_filenames}")
