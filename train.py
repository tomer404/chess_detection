from ultralytics import YOLO
model = YOLO("yolo11n.pt")
results = model.train(
    data = "data.yaml",
    epochs = 100,
    imgsz = 640,
    batch = 16,
    device = "cpu"
)
metrics = model.val()
print(metrics.box.map)
print(metrics.box.map50, metrics.box.map75)
