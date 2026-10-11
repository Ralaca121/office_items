"""CLI + OpenCV Stream loop for office items classification"""

import argparse
import sys
from pathlib import Path

import cv2

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".zip"]
VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"]
WINDOW_NAME = "Office Items Classification"

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="OpenCV stream runner (camera / image / video)"
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Camera index or file path.",
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


def process_frame(frame):
    """Placeholder for model inference"""


    return frame

def run_image(image_path: str):
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Could not load image: {image_path}")

    out = process_frame(frame)
    cv2.imshow(WINDOW_NAME, out)
    cv2.waitKey(0)


def run_stream(source_value):
    cap = cv2.VideoCapture(source_value)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video stream: {source_value}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print(f"End of stream or failed to read frame from stream: {source_value}")
                break

            out = process_frame(frame)
            cv2.imshow(WINDOW_NAME, out)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()


def main(argv=None):
    args = parse_args(argv)
    try:
        source_type, source_value = resolve_source(args.source)

        if source_type == "image":
            run_image(source_value)
        else:
            run_stream(source_value)

    except (FileNotFoundError, ValueError, RuntimeError) as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1
    finally:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())