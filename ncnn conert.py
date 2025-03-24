from ultralytics import YOLO

model = YOLO("api_besar_model.pt")

model.export(format="ncnn")