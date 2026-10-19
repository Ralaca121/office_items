from pathlib import Path
import random
import tempfile

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from ultralytics import YOLO


NUM_CLASSES = 8
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def images_in(folder):
	return sorted(
		path for path in folder.rglob("*")
		if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
	)


def link_tree(source, destination):
	"""Create class-folder symlinks without copying dataset images."""
	destination.mkdir(parents=True, exist_ok=True)
	for class_dir in sorted(source.iterdir()):
		if class_dir.is_dir() and images_in(class_dir):
			(destination / class_dir.name).symlink_to(class_dir.resolve(), target_is_directory=True)


def make_split(dataset_dir, split_root, validation_fraction=0.2):
	classes = sorted(
		folder for folder in dataset_dir.iterdir()
		if folder.is_dir() and images_in(folder)
	)
	if len(classes) != NUM_CLASSES:
		raise ValueError(f"Expected {NUM_CLASSES} class folders in {dataset_dir}, found {len(classes)}.")

	rng = random.Random(42)
	for class_dir in classes:
		files = images_in(class_dir)
		if len(files) < 2:
			raise ValueError(f"Class {class_dir.name!r} needs at least two images to split.")
		rng.shuffle(files)
		val_count = max(1, round(len(files) * validation_fraction))
		for split, selected in (("train", files[val_count:]), ("val", files[:val_count])):
			class_destination = split_root / split / class_dir.name
			class_destination.mkdir(parents=True, exist_ok=True)
			for image in selected:
				(class_destination / image.name).symlink_to(image.resolve())


def main():
	script_dir = Path(__file__).resolve().parent
	dataset_dir = script_dir / "Dataset"
	if not dataset_dir.is_dir():
		raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

	train_source = dataset_dir / "train"
	test_source = dataset_dir / "test"
	validation_source = dataset_dir / "val"

	with tempfile.TemporaryDirectory(prefix="yolo_classification_") as temp_name:
		temp_root = Path(temp_name)
		if train_source.is_dir():
			train_classes = sorted(folder.name for folder in train_source.iterdir() if folder.is_dir())
			if validation_source.is_dir():
				# Use the supplied train/val splits and reserve test, when present, for final metrics.
				data_root = dataset_dir
				evaluation_dir = test_source if test_source.is_dir() else validation_source
			elif test_source.is_dir():
				# YOLO requires a val split; create a temporary layout with test used for validation.
				data_root = temp_root / "data"
				link_tree(train_source, data_root / "train")
				link_tree(test_source, data_root / "val")
				evaluation_dir = test_source
			else:
				raise ValueError("Dataset/train must be accompanied by Dataset/val or Dataset/test.")
		else:
			# For Dataset/<class>/<image>, reproducibly make an 80/20 train/validation split.
			data_root = temp_root / "data"
			make_split(dataset_dir, data_root)
			train_source = data_root / "train"
			evaluation_dir = data_root / "val"
			train_classes = sorted(folder.name for folder in train_source.iterdir() if folder.is_dir())

		if len(train_classes) != NUM_CLASSES:
			raise ValueError(f"Expected {NUM_CLASSES} training classes, found {len(train_classes)}: {train_classes}")

		model = YOLO("yolo11n-cls.pt")
		model.train(
			data=str(data_root),
			epochs=20,
			imgsz=224,
			project=str(script_dir / "runs"),
			name="classification",
			exist_ok=True,
		)

		best_weights = script_dir / "runs" / "classification" / "weights" / "best.pt"
		model = YOLO(str(best_weights))
		model_names = model.names
		if isinstance(model_names, dict):
			model_names = [model_names[index] for index in range(len(model_names))]

		actual, predicted = [], []
		for class_name in train_classes:
			for image_path in images_in(evaluation_dir / class_name):
				result = model.predict(str(image_path), verbose=False)[0]
				actual.append(class_name)
				predicted.append(model_names[int(result.probs.top1)])

		if not actual:
			raise ValueError(f"No evaluation images found under {evaluation_dir}.")

		print(f"Accuracy: {accuracy_score(actual, predicted):.4f}")
		print("Classification report:")
		print(classification_report(
			actual, predicted, labels=train_classes, target_names=train_classes, zero_division=0
		))
		print("Confusion matrix (rows=true, columns=predicted):")
		print(confusion_matrix(actual, predicted, labels=train_classes))


if __name__ == "__main__":
	main()
