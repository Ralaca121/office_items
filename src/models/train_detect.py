import argparse
from pathlib import Path 

import yaml
from ultralytics import YOLO

BASE_WEIGHTS = "yolo11n.pt"


def parse_args(argv=None):
    parser = argparse.ArgumentParser("Train YOLO11n detector")
    parser.add_argument("--data", required=True, default="data/detect_v2/data.yaml", help="Path to the data.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    # Added patience arg for early stopping (default 15 epochs)
    parser.add_argument("--patience", type=int, default=15, help="Epochs to wait for improvement before early stopping")
    parser.add_argument("--device", default=None, help="e.g. mps, cpu, or 0")
    parser.add_argument("--name", default="detect_v2")
    return parser.parse_args(argv)


def check_dataset(data_yaml: Path):
    """Fails if paths or labels are wrong"""
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Data YAML file not found: {data_yaml}")

    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(cfg.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()

    print (f"Classes ({len(cfg['names'])}): {cfg['names']}")
    for split in ("train", "val", "test"):
        if split not in cfg:
            continue
        img_dir = root / cfg[split]
        if not img_dir.is_dir():
            raise FileNotFoundError(f"{split} directory not found: {img_dir}")
        n_images = sum(1 for f in img_dir.iterdir() if f.is_file())
        label_dir = Path(str(img_dir).replace("images", "labels"))
        n_nonempty = sum(
            1 for f in label_dir.glob("*.txt") if f.stat().st_size > 0
            )if label_dir.is_dir() else 0
        print(f"{split}: {n_images} images, {n_nonempty} non-empty labels")
        if n_nonempty == 0:
            raise ValueError(f"No box labels found for '{split}' in {label_dir}")


def main(argv=None):
    args = parse_args(argv)
    data_yaml = Path(args.data)
    check_dataset(data_yaml)

    model = YOLO(BASE_WEIGHTS)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch,
        project="runs",
        name=args.name,
        exist_ok=True,
        patience=args.patience,
        # data augmentation
        degrees=15.0,
        shear=2.0, 
    )


    best = Path("runs") / "detect_v2" /"weights" / "best.pt"
    print(f"\nBest weights: {best}")

if __name__ == "__main__":
    main()
