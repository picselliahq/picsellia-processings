from picsellia import Client
from utils.data import PreAnnotator
import os

if "host" not in os.environ:
    host = "https://app.picsellia.com"
else:
    host = os.environ["host"]
job_id = os.environ["job_id"]

client = Client(
    api_token=api_token,
    host=host
)

job = client.get_job_by_id(job_id)

context = job.sync()["datasetversionprocessingjob"]
model_version_id = context["model_version_id"]
dataset_version_id = context["input_dataset_version_id"]

X = PreAnnotator(
    client=client, 
    model_version_id=model_version_id, # same
    dataset_version_id=dataset_version_id # same
)
X.setup_preannotation_job()
X.preannotate()