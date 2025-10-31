from ultralytics import YOLO
model = YOLO("models/yolov11n-UAV-finetune (1).pt")
print(model.names)