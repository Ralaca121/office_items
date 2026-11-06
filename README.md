# Office Items Object Detection
The following will be steps on how to train the model, obtain the perfomance metrices of the model and how to run the app.

## 1. Set up the Python environment

Run these commands from the repository root having python and pip already installed:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 2. Prepare the dataset
The dataset folder will be found in the folder PDE3802_Assessment1.zip file named data.

The expected structure is:
```text
data/detect_v2/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```
Move the dataset into the directory with the rest of the code.

## 3. Train the detector

With the virtual environment active and the dataset in place, run:

```bash
python -m src.models.train_detect \
  --data data/detect_v2/data.yaml \
  --epochs 100 \
  --img_size 640 \
  --batch 16 \
  --patience 15
```

The script uses pretrained YOLO11n weights and save the best checkpoint to:

```text
runs/detect_v2/weights/best.pt
```

The first training run may need an internet connection to download the pretrained weights.

## 4. Set the weights used for evaluation and the app

The test script and app use `model/best_det2.pt`. Copy the checkpoint from training:

```bash
cp runs/detect_v2/weights/best.pt models/best_det.pt
```

This replaces the existing file at that path.

## 5. Evaluate on the test split

Run this after copying the new weights:

```bash
python -m src.models.test_detect
```

## 6. Run the Streamlit app

With the virtual environment active and `models/best_det2.pt` available, start the app from the repository root:

```bash
streamlit run src/app.py
```

If it does not open automatically open the local URL printed by Streamlit in you browser.
