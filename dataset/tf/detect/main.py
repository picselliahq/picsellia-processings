from picsellia import Client
api_token = ""
client = Client(api_token=api_token)
from utils.data import PreAnnotator

X = PreAnnotator(
    client=client, 
    model_id="PCB Defects detection", # change to id in PreAnnotator Class when we have the info
    model_version_id=0, # same
    dataset_id="PCB-DEFECTS", #same 
    dataset_version_id="test-preprocessing" # same
)
X.setup_preannotation_job()
X.preannotate()