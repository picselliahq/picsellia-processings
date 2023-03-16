docker build -f Dockerfile.tensorflow . -t picsellpn/preannotation-tf:capture
docker push picsellpn/preannotation-tf:capture 

docker build -f Dockerfile.yolov8 . -t picsellpn/preannotation-yolov8:capture
docker push picsellpn/preannotation-yolov8:capture 