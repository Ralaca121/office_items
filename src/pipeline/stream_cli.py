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
# Default detector and inference image sizes.
DEFAULT_MODELS = {
    "detect": "models/best_det2.pt"
}
CLS_IMGSZ = 224
DET_IMGSZ = 640
# These classes must meet their class-specific minimum as well as the global --conf.
CLASS_MIN_CONF = {"Chair": 0.60, "Book": 0.50}

# Each key is a COCO class; its value lists our classes it can veto (None means all).
COCO_WEIGHTS = "yolo11n.pt"
COCO_CONF = 0.40
COCO_IOU = 0.50
COCO_VETO = {
    "person": None,
    "couch": {"Chair", "Table"},
    "laptop": {"Book"},
    "tv": {"Book"},
    "keyboard": {"Book"},
    "cell phone": {"Book"},
    "remote": {"Book"},
    "mouse": {"Book"},
}
_coco_model = None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="OpenCV stream runner (camera / image / video)"
    )
    # A numeric source selects a camera; otherwise source must be a supported image or video file.
    parser.add_argument(
        "--source",
        required=True,
        help="Camera index or file path.",
    )
    parser.add_argument(
        "--task",
        choices=["detect"],
        default="detect",
        help="Mode of operation: detect ",
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
    parser.add_argument(
        "--min-hits",
        type=int,
        default=3,
        help="Detection must be seen in N inference steps before shown; 1 disables smoothing (default: 3)",
    )
    parser.add_argument(
        "--no-coco-filter",
        action="store_true",
        help="Disable the COCO person/look-alike filter",
    )
    return parser.parse_args(argv)

def resolve_source(source: str):
    # Numeric input is interpreted as an OpenCV camera index.
    if source.isdigit():
        return "camera", int(source)

    # Non-numeric input must point to a supported image or video file.
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

def class_threshold(label, conf_thresh):
    """Effective threshold: the global one, raised for classes that over-trigger."""
    return max(conf_thresh, CLASS_MIN_CONF.get(label, 0.0))


def coco_veto_boxes(frame):
    """Run the COCO model and return [(coco_label, box)] for classes in COCO_VETO."""
    global _coco_model
    # Load the auxiliary model only once; reuse it for subsequent frames.
    if _coco_model is None:
        _coco_model = YOLO(COCO_WEIGHTS)
    result = _coco_model.predict(frame, imgsz=DET_IMGSZ, conf=COCO_CONF, verbose=False)[0]
    boxes = []
    for box in result.boxes:
        name = result.names[int(box.cls[0].item())]
        if name in COCO_VETO:
            boxes.append((name, tuple(map(int, box.xyxy[0].tolist()))))
    return boxes


def filter_distractors(frame, detections):
    """Drop detections that coincide with a person / look-alike object."""
    if not detections:
        return detections
    vetoes = coco_veto_boxes(frame)
    kept = []
    for label, conf, box in detections:
        # A matching COCO box suppresses only the configured office-item classes.
        vetoed = any(
            (COCO_VETO[name] is None or label in COCO_VETO[name]) and iou(box, vbox) >= COCO_IOU
            for name, vbox in vetoes
        )
        if not vetoed:
            kept.append((label, conf, box))
    return kept


def detect_objects(model, frame, conf_thresh=0.25, coco_filter=True):
    """Detection: return list of label, confidence, and bounding box coordinates"""
    # Class-agnostic NMS prevents the same region being returned under multiple labels.
    result = model.predict(
        frame, imgsz=DET_IMGSZ, conf=conf_thresh, agnostic_nms=True, verbose=False
    )[0]
    detections = []
    if result.boxes is None:
        return detections

    names = result.names
    for box in result.boxes:

        cls_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        label = names[cls_id]
        # Apply the stricter class threshold before converting the box to pixel coordinates.
        if confidence < class_threshold(label, conf_thresh):
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        detections.append((label, confidence, (x1, y1, x2, y2)))
    if coco_filter:
        detections = filter_distractors(frame, detections)
    return detections


def iou(a, b):
    """Return intersection-over-union for two xyxy bounding boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class DetectionSmoother:
    """Temporal filter: a detection is shown only after it was seen in
    `min_hits` inference steps, and kept for `max_age` missed steps."""

    def __init__(self, min_hits=3, max_age=2, iou_thresh=0.4, alpha=0.6):
        self.min_hits = min_hits
        self.max_age = max_age
        self.iou_thresh = iou_thresh
        self.alpha = alpha  # weight of the new box when smoothing coordinates
        self.tracks = []

    def update(self, detections):
        # Match each detection to one same-class track with sufficient box overlap.
        for t in self.tracks:
            t["matched"] = False
        for label, conf, box in sorted(detections, key=lambda d: -d[1]):
            best, best_iou = None, self.iou_thresh
            for t in self.tracks:
                if t["matched"] or t["label"] != label:
                    continue
                score = iou(t["box"], box)
                if score >= best_iou:
                    best, best_iou = t, score
            if best is None:
                self.tracks.append({"label": label, "conf": conf, "box": box,
                                    "hits": 1, "misses": 0, "matched": True})
                continue
            a = self.alpha
            best["box"] = tuple(int(a * n + (1 - a) * o) for n, o in zip(box, best["box"]))
            best["conf"] = a * conf + (1 - a) * best["conf"]
            best["hits"] += 1
            best["misses"] = 0
            best["matched"] = True
        # Reduce confidence in tracks not seen this step, then discard stale tracks.
        for t in self.tracks:
            if not t["matched"]:
                t["misses"] += 1
                t["hits"] = max(0, t["hits"] - 1)
        self.tracks = [t for t in self.tracks if t["misses"] <= self.max_age]
        return [(t["label"], t["conf"], t["box"]) for t in self.tracks
                if t["hits"] >= self.min_hits]


def draw_detections(frame, detections, conf_thresh=0.25):
    for label, confidence, (x1, y1, x2, y2) in detections:
        if confidence < class_threshold(label, conf_thresh):
            continue
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label}: {confidence * 100:.0f}"
        cv2.putText(
            frame, text, (x1, max(20, y1 - 10)), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
    return frame

def infer(model, frame, task, conf, coco_filter=True):
    """Perform inference for the task and return it's raw output."""
    if task == "detect":
        return detect_objects(model, frame, conf, coco_filter)
    return predict(model, frame)
    
def draw_output(frame, output, task, conf):
    """Draw the output of infer() onto the frame."""
    if task == "detect":
        return draw_detections(frame, output, conf)
    label, confidence = output
    return draw_overlay(frame, label, confidence)


def run_image(image_path: str, model, task, conf, coco_filter=True):
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Could not load image: {image_path}")

    output = infer(model, frame, task, conf, coco_filter)
    cv2.imshow(WINDOW_NAME, draw_output(frame, output, task, conf))
    cv2.waitKey(0)


def run_stream(source_value, model, task, conf, infer_every=2, min_hits=3, coco_filter=True):
    cap = cv2.VideoCapture(source_value)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video stream: {source_value}")

    frame_count = 0
    last_output = [] if task == "detect" else ("Detecting...", 0.0)
    # Smoothing is optional: min_hits=1 displays detections immediately.
    smoother = DetectionSmoother(min_hits=min_hits) if task == "detect" and min_hits > 1 else None
    prev_t = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print(f"End of stream or failed to read frame from stream: {source_value}")
                break

            frame_count += 1

            # Reuse the last result between inference steps to reduce model workload.
            if frame_count % infer_every == 0:
                last_output = infer(model, frame, task, conf, coco_filter)
                if smoother:
                    last_output = smoother.update(last_output)

            # Draw the latest detection result on the current frame.
            out = draw_output(frame, last_output, task, conf)

            # Display the stream's processing rate.
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
        model = load_model(model_path)  # Load weights once and reuse for every frame.

        if source_type == "image":
            run_image(source_value, model, args.task, args.conf, not args.no_coco_filter)
        else:
            # Pass stream-specific controls only to the video/camera loop.
            run_stream(source_value, model, args.task, args.conf,
                       infer_every=args.infer_every, min_hits=args.min_hits, coco_filter=not args.no_coco_filter)

    except (FileNotFoundError, ValueError, RuntimeError) as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1
    finally:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())