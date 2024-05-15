yolo detect train data=person_dataset.yaml model=./weights/yolov8s.pt epochs=120 imgsz=800 workers=4 batch=16 project=person_s name=s_100_800
yolo detect train data=person_dataset.yaml model=./weights/yolov8n.pt epochs=120 imgsz=800 workers=4 batch=16 project=person_s name=n_100_800
