import os

from abstract_processor.utils import get_client


class AbstractProcessor:
    def __init__(self):
        self.client = get_client()
        self.job = self.client.get_job_by_id(os.environ["job_id"])

    def process(
        self,
    ):
        raise NotImplementedError("You should implement this method in a subclass")
