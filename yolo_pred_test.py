from ultralytics import YOLOv10

# Load the model
model = YOLOv10('./data/weight/detection/yolov10b_keti_tp241120_0_90.pt')  # or use other sizes: s/m/b/l/x
# Alternative loading method
# model = YOLOv10('yolov10n.pt')

# Run inference
results = model.predict(source='./test/1.jpg', conf=0.5)
boxes = results[0].boxes.data.cpu().numpy()

# Show results
print(boxes)