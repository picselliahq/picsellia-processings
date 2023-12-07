import os
import numpy as np
import cv2
import logging
from scipy.stats import zscore
from imagededup.methods import PHash
from picsellia.sdk.dataset_version import DatasetVersion
from picsellia import Tag, Asset

import matplotlib.pyplot as plt
from pycocotools.coco import COCO


def get_duplicate_images(dataset_path: str) -> dict:
    phasher = PHash()
    encodings = phasher.encode_images(image_dir=dataset_path)
    duplicates = phasher.find_duplicates(encoding_map=encodings)
    return duplicates


def add_tags_to_duplicate_images(
        dataset_version: DatasetVersion, duplicates: dict
) -> set:
    dup_image_tag = dataset_version.create_asset_tag(name="dup_image")
    duplicate_files = set()
    for filename, duplicate_filenames in duplicates.items():
        for duplicate_filename in duplicate_filenames:
            if duplicate_filename:  # duplicate found
                duplicate_files.add(filename)
                duplicate_files.add(duplicate_filename)

                find_asset_and_add_tag(
                    dataset_version=dataset_version,
                    filename=filename,
                    tag=dup_image_tag,
                )
                find_asset_and_add_tag(
                    dataset_version=dataset_version,
                    filename=duplicate_filename,
                    tag=dup_image_tag,
                )

    return duplicate_files


def find_asset_and_add_tag(dataset_version: DatasetVersion, filename: str, tag: Tag):
    filename_asset = find_asset_by_filename(filename=filename, dataset=dataset_version)
    filename_asset.add_tags(tag)


def find_asset_by_filename(filename: str, dataset: DatasetVersion) -> Asset | None:
    try:
        asset = dataset.find_asset(filename=filename)
        return asset
    except Exception as e:
        logging.info(e)
        return None


def get_duplicate_filenames(dataset_version: DatasetVersion) -> list[str]:
    input_assets = dataset_version.list_assets()
    image_filenames = [asset.filename for asset in input_assets]
    filename_duplicates = find_filename_duplicates(image_filenames)
    return filename_duplicates


def find_filename_duplicates(strings: list[str]) -> list[str]:
    seen = set()
    duplicates = set()

    for string in strings:
        if string in seen:
            duplicates.add(string)
        else:
            seen.add(string)

    return list(duplicates)


def find_filename_by_id(image_id: int, coco: COCO) -> str | None:
    img_info = coco.loadImgs(int(image_id))
    if img_info:
        return img_info[0]["file_name"]
    else:
        return None


def add_nbr_channel_byte_tags(
        dataset_version: DatasetVersion, dataset_path: str
) -> tuple[dict, dict]:
    byte_counts = {}
    channel_counts = {}
    for asset in dataset_version.list_assets():
        existing_tags = dataset_version.list_asset_tags()
        existing_tag_names = [tag.name for tag in existing_tags]

        filename = os.path.join(dataset_path, asset.filename)
        BGR = cv2.imread(filename, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        nbr_bytes = BGR.dtype
        nbr_channels = get_nbr_channels(BGR)
        byte_counts[nbr_bytes] = byte_counts.get(nbr_bytes, 0) + 1
        channel_counts[nbr_channels] = channel_counts.get(nbr_channels, 0) + 1

        tag_name = str(nbr_channels) + "_" + str(nbr_bytes)
        if tag_name not in existing_tag_names:
            image_property_tag = dataset_version.create_asset_tag(name=tag_name)
        else:
            image_property_tag = next(
                (tag for tag in existing_tags if tag.name == tag_name), None
            )
        asset.add_tags(image_property_tag)

    return channel_counts, byte_counts


def get_nbr_channels(image: np.ndarray) -> int:
    shape = len(image.shape)
    nbr_channels = 0
    if shape == 2:
        nbr_channels = 1
    elif shape == 3:
        nbr_channels = image.shape[2]
    else:
        logging.info("Image has an unexpected shape")
    return nbr_channels


def get_area_outlier_filenames(coco: COCO, area_outlier_threshold: int) -> list[str]:
    areas, filename_ids = get_all_areas_filenames(coco=coco)
    z_scores = zscore(areas)
    outlier_indices = np.where(np.abs(z_scores) > area_outlier_threshold)[0]

    outlier_areas = areas[outlier_indices]
    filename_ids = list(set(filename_ids[outlier_indices]))
    ood_filenames = [
        find_filename_by_id(image_id=filename_ids[i], coco=coco)
        for i in range(len(filename_ids))
    ]
    logging.info(f"Outlier Areas: {outlier_areas}")
    # Plot a histogram of area values
    # plt.hist(areas, bins=50, color='blue', edgecolor='black')
    # plt.title('Histogram of Object Areas')
    # plt.xlabel('Area')
    # plt.ylabel('Frequency')
    # plt.show()

    return ood_filenames


def get_all_areas_filenames(coco: COCO) -> tuple[np.ndarray, np.ndarray]:
    area_values = []
    image_ids = []

    for ann_id in coco.getAnnIds():
        annotation = coco.loadAnns(ann_id)[0]
        area = annotation["area"]
        image_id = annotation["image_id"]
        area_values.append(area)
        image_ids.append(image_id)

    area_values_np = np.array(area_values)
    image_ids = np.array(image_ids)
    return area_values_np, image_ids


def log_results(
    duplicate_image_filenames,
    filename_duplicates,
    byte_counts,
    channel_counts,
    area_outlier_filenames,
):
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

# def check_bounding_boxes(coco, asset: Asset):
#     image_width = asset.width
#     image_height = asset.height
#     annotations = coco.loadAnns(coco.getAnnIds())
#
#     for annotation in annotations:
#         x, y, width, height = annotation['bbox']
#         x_max, y_max = x + width, y + height
#
#         if x < 0 or y < 0 or x_max > image_width or y_max > image_height:
#             print(x, y)
#             print(x_max, y_max)
#             print(image_width, image_height)
#             print(
#                 f"Bounding box exceeds image boundaries in annotation id {annotation['id']}, in asset {asset.filename}.")
