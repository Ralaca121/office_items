from pathlib import Path
from ultralytics import YOLO

# Trained detection weights and the dataset whose test split is evaluated.
MODEL_PATH = Path("models/best_det2.pt")
DATA_YAML = Path("data/detect_v2/data.yaml")
IMG_SIZE = 640
BATCH_SIZE = 16


def main():
    # Fail early with a clear message if the weights or dataset config are missing.
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model weight not found: {MODEL_PATH}")
    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"Dataset config not found: {DATA_YAML}")
    
    # Evaluate on the held-out test split; plots=True saves the confusion matrix and curves under runs/detect/.
    model = YOLO(str(MODEL_PATH)) 
    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        plots=True
    )

    # Per-class precision and recall for the detection boxes.
    box = metrics.box
    precisions = box.p
    recalls = box.r

    if len(precisions) == 0:
        raise RuntimeError("No class metrics returned. Check image and labels.")

    # Average inference time per image in ms, converted to FPS below.
    inference_ms = metrics.speed["inference"]
    if inference_ms <= 0:
        raise RuntimeError(f"Invalid inference time: {inference_ms}")

    # Performance metrics for the detection model.
    print(f"Overall Test Precision: {box.mp:.4f}")
    print(f"mAP50:                  {box.map50:.4f}")
    print(f"Recall:                  {box.mr:.4f}")
    print(f"Inference FPS:          {1000 / inference_ms:.2f}")

if __name__ == "__main__":
    main()

