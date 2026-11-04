from pathlib import Path
from ultralytics import YOLO

#Using detection weights from the trained model
MODEL_PATH = Path("models/best_det2.pt")
DATA_YAML = Path("data/detect_v2/data.yaml")
IMG_SIZE = 640
BATCH_SIZE = 16


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model weight not found: {MODEL_PATH}")
    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"Dataset config not found: {DATA_YAML}")
    
    model = YOLO(str(MODEL_PATH)) 
    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        plots=True
    )

    box = metrics.box
    precisions = box.p
    recalls = box.r

    if len(precisions) == 0:
        raise RuntimeError("No class metrics returned. Check image and labels.")

    class_f1 = [
        2 * p * r / (p + r) if (p + r) > 0 else 0
        for p, r in zip(precisions, recalls)
    ]
    macro_f1 = sum(class_f1) / len(class_f1) 

    inference_ms = metrics.speed["inference"]
    if inference_ms <= 0:
        raise RuntimeError(f"Invalid inference time: {inference_ms}")

    print(f"Overall Test Precision: {box.mp:.4f}")
    print(f"Macro-F1 Score:         {macro_f1:.4f}")
    print(f"mAP50:                  {box.map50:.4f}")
    print(f"Recall:                  {box.mr:.4f}")
    print(f"Inference FPS:          {1000 / inference_ms:.2f}")

if __name__ == "__main__":
    main()

