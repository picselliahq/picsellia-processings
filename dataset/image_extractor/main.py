import argparse

from utils.PicselliaImageExtractor import PicselliaImageExtractor


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process and transform dataset using Picsellia."
    )
    parser.add_argument(
        "--api_token", required=True, type=str, help="API token for authentication"
    )
    parser.add_argument(
        "--dataset_id", required=True, type=str, help="ID of the dataset"
    )
    parser.add_argument(
        "--dataset_version_id",
        required=True,
        type=str,
        help="ID of the dataset version",
    )
    parser.add_argument(
        "--datalake_name",
        required=True,
        type=str,
        help="Name of the datalake",
        default="default",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    processor = PicselliaImageExtractor(
        args.api_token, args.dataset_id, args.dataset_version_id, args.datalake_name
    )
    processor.download_ds()
    processor.create_extracted_ds()
    processor.process_images()
    processor.upload_images_to_extracted_ds()


if __name__ == "__main__":
    main()
