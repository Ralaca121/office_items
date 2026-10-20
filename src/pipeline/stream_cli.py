"""CLI + OpenCV Stream loop for office items classification"""

import argparse
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".zip"]
VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"]
WINDOW_NAME = "Office Items Classification"
DEFAULT_MODEL = "models/best.pt"

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
        "--model",
        default=DEFAULT_MODEL,
        help=f"Path to trained YOLO model (default:{DEFAULT_MODEL}).",
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
    result = model.predict(frame, imgsz=224, verbose=False)[0]
    idx = int(result.probs.top1)
    return result.names[idx], float(result.probs.top1conf)


def draw_overlay(frame,label, confidence):
    text = f"{label}: {confidence *100:.2f}"
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.rectangle(frame, (5, 5), (15 + w, 15 + h + baseline), (0, 0, 0), -1)
    cv2.putText(frame, text, (10, 10 + h), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
    return frame

def process_frame(frame, model):
    label, confidence = predict(model, frame)
    return draw_overlay(frame, label, confidence)

def run_image(image_path: str, model):
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Could not load image: {image_path}")

    cv2.imshow(WINDOW_NAME, process_frame(frame, model))
    cv2.waitKey(0)


def run_stream(source_value, model):
    cap = cv2.VideoCapture(source_value)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video stream: {source_value}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print(f"End of stream or failed to read frame from stream: {source_value}")
                break

            cv2.imshow(WINDOW_NAME, process_frame(frame, model))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 27 is the ESC key
                break
    finally:
        cap.release()


def main(argv=None):
    args = parse_args(argv)
    try:
        source_type, source_value = resolve_source(args.source)
        model = load_model(args.model) # loads once

        if source_type == "image":
            run_image(source_value, model)
        else:
            run_stream(source_value, model)

    except (FileNotFoundError, ValueError, RuntimeError) as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1
    finally:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())