from picsellia import Client
from utils.yolox.annotator import PreAnnotator
import os

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

# job_id = os.environ["job_id"]
# job = client.get_job_by_id(job_id)

# context = job.sync()["dataset_version_processing_job"]
model_version_id = "018d452c-bca0-7f2e-951f-a519e25b6093"  # context["model_version_id"]
dataset_version_id = (
    "018d5604-e89e-7312-b5f3-fc6db1109ec1"  # context["input_dataset_version_id"]
)
parameters = {"batch_size": 8, "confidence_threshold": 0}  # context["parameters"]
confidence_threshold = parameters.get("confidence_threshold", 0.1)


X = PreAnnotator(
    client=client,
    model_version_id=model_version_id,
    dataset_version_id=dataset_version_id,
    parameters=parameters,
)
X.setup_pre_annotation_job()
X.pre_annotate(confidence_threshold)