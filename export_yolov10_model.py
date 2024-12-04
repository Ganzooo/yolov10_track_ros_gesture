from ultralytics import YOLO
model = YOLO("./data/weight/detection/yolov10b_keti_tp241120_0_90.pt")
model.imgsz = (544, 960)
model.export(format='onnx', imgsz=(544,960))