from ultralytics import YOLO
#model = YOLO("yolov8n.pt")
#
model=YOLO("runs/detect/train4/weights/best.pt")
#model.train(data="burak.yaml", epochs=60, imgsz=640, resume=False)
result=model.predict(source='./GHEK0965_frame_233.jpg', save=True, conf=0.4)