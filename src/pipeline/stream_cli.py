"""CLI + OpenCV Stream loop for office items classification"""

import argparse
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ]
VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"]
WINDOW_NAME = "Office Items Classification"
DEFAULT_MODELS = {
    "classify": "models/best_cls.pt",
    "detect": "models/best_det.pt"
}
CLS_IMGSZ = 224
DET_IMGSZ = 640


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="OpenCV stream runner (camera / image / video)"
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Camera index or file path.",
    )
    parser.add_argument(
        "--task",
        choices=["classify", "detect"],
        default="classify",
        help="Mode of operation: classify or detect ",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to the trained YOLO model.",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detection model (default: 0.25)",
    )
    parser.add_argument(
        "--infer-every",
        type=int,
        default=2,
        help="Perform inference every N frames for video streams (default: 2)",
    )
    return parser.parse_args(argv)

def resolve_source(source: str):
    # camera source
    if source.isdigit():
        return "camera", int(source)

    # file source
    path = Path(source)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"File not found: {source}")
    
    ext = path.suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return "image", str(path)
    if ext in VIDEO_EXTENSIONS:
        return "video", str(path)

    raise ValueError(
        f" unsupported file extension: {ext}"
        f"Use one of {sorted(IMAGE_EXTENSIONS + VIDEO_EXTENSIONS)}"
    )


def load_model(model_path: str):
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {model_path}")
    return YOLO(str(path))

def predict(model, frame):
    """Return (label, confidence 0-1) for a BGR frame"""
    result = model.predict(frame, imgsz=CLS_IMGSZ, verbose=False)[0]
    idx = int(result.probs.top1)
    return result.names[idx], float(result.probs.top1conf)


def draw_overlay(frame,label, confidence):
    text = f"{label}: {confidence *100:.2f}"
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.rectangle(frame, (5, 5), (15 + w, 15 + h + baseline), (0, 0, 0), -1)
    cv2.putText(frame, text, (10, 10 + h), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
    return frame

def detect_objects(model, frame, conf_thresh=0.25):
    """Detection: return list of label, confidence, and bounding box coordinates"""
    result = model.predict(frame, imgsz=DET_IMGSZ, conf=conf_thresh, verbose=False)[0]
    detections = []
    if result.boxes is None:
        return detections

    names = result.names
    for box in result.boxes:

        cls_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        detections.append((names[cls_id], confidence, (x1, y1, x2, y2)))
    return detections

def draw_detections(frame, detections, conf_thresh=0.25):
    for label, confidence, (x1, y1, x2, y2) in detections:
        if confidence < conf_thresh:
            continue
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label}: {confidence * 100:.1f}"
        cv2.putText(
            frame, text, (x1, max(20, y1 - 10)), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
    return frame

def infer(model, frame, task, conf):
    """Perform inference for the task and return it's raw output."""
    if task == "detect":
        return detect_objects(model, frame, conf)
    return predict(model, frame)
    
def draw_output(frame, output, task, conf):
    """Draw the output of infer() onto the frame."""
    if task == "detect":
        return draw_detections(frame, output, conf)
    label, confidence = output
    return draw_overlay(frame, label, confidence)


def run_image(image_path: str, model, task, conf):
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Could not load image: {image_path}")

    output = infer(model, frame, task, conf)
    cv2.imshow(WINDOW_NAME, draw_output(frame, output, task, conf))
    cv2.waitKey(0)


def run_stream(source_value, model, task, conf, infer_every=2):
    cap = cv2.VideoCapture(source_value)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video stream: {source_value}")

    frame_count = 0
    last_output = [] if task == "detect" else ("Detecting...", 0.0)
    prev_t = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print(f"End of stream or failed to read frame from stream: {source_value}")
                break

            frame_count += 1

            # Inference every n frames for better performance
            if frame_count % infer_every == 0:
                last_output = infer(model, frame, task, conf)

            # Drawing classification overlay
            out = draw_output(frame, last_output, task, conf)

            # FPS overlay
            now = time.time()
            fps = 1.0 / max((now - prev_t), 1e-6)
            prev_t = now
            cv2.putText(
                out, 
                f"FPS: {fps:.2f}", 
                (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 255, 0), 2, 
                cv2.LINE_AA)
            cv2.imshow(WINDOW_NAME, out)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 27 is the ESC key
                break
    finally:
        cap.release()


def main(argv=None):
    args = parse_args(argv)
    try:
        source_type, source_value = resolve_source(args.source)
        model_path = args.model or DEFAULT_MODELS[args.task]
        model = load_model(model_path) # loads once

        if source_type == "image":
            run_image(source_value, model, args.task, args.conf)
        else:
            run_stream(source_value, model, args.task, args.conf, infer_every=args.infer_every)

    except (FileNotFoundError, ValueError, RuntimeError) as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1
    finally:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())