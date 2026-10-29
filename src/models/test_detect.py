from ultralytics import YOLO

#Using detection weights from the trained model
MODEL_PATH = "models/best_det.pt"
SOURCE = "data/detect_v1/test/images"

model = YOLO(MODEL_PATH) 
results = model.predict(
    source=SOURCE,
    conf=0.25, # confidence threshold
    iou=0.45,
    save=True # saves boxed images
)

print("Done. Predictions saved to runs/detect/predict*")
print("Classes:", model.names)