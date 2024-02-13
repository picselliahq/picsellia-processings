import os

from picsellia import Client

from utils.PicselliaVlmTagging import PicselliaVlmTagging


def get_client():
    if "api_token" not in os.environ:
        raise Exception("You must set an api_token to run this image")
    api_token = os.environ["api_token"]

    if "host" not in os.environ:
        host = "https://app.picsellia.com"
    else:
        host = os.environ["host"]
    if "organization_id" not in os.environ:
        organization_id = None
    else:
        organization_id = os.environ["organization_id"]

    client = Client(api_token=api_token, host=host, organization_id=organization_id)

    return client


if __name__ == "__main__":
    client = get_client()
    job = client.get_job_by_id(os.environ["job_id"])
    context = job.sync()["dataset_version_processing_job"]

    processor = PicselliaVlmTagging(
        client=client,
        model_version_id=context["model_version_id"],
        dataset_version_id=context["input_dataset_version_id"],
    )
    processor._add_tags_to_assets()
