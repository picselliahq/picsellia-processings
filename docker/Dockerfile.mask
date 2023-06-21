FROM python:3.8-bullseye as base

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ARG DEBIAN_FRONTEND=noninteractive

RUN pip3 install --no-cache-dir picsellia
COPY requirements.txt .
RUN pip3 install -r requirements.txt
COPY . /



CMD ["/picsellia/launch_processing.py"]
ENTRYPOINT ["python3"]
RUN chown -R 42420 /picsellia