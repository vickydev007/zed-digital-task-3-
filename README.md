# Applied Computer Vision · Student Assignment
## FEDS Technical Assignment · Problem Statement 3: ₹50 Note Detector

### Goal
Train a model that detects the **new light blue-green ₹50 note** in a live camera feed and draws a box labelled **“₹50”** with its confidence score when the note is in front of the camera. The model must not detect other notes (₹10, ₹20, ₹100, ₹200, ₹500) or objects as ₹50.

---

### Target Banknote Specifications
- **Denomination**: New light blue-green ₹50 banknote
  - **Front**: Mahatma Gandhi portrait
  - **Back**: Hampi stone chariot
- **Exclusions**: Older ₹50 notes, other currency denominations, and non-currency objects.

---

### Quick Start & Run Command

#### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 2. Run Live Webcam Detection
```bash
python test.py --model model.onnx
```

#### Optional Arguments
```bash
# Specify camera index if using an external webcam
python test.py --model model.onnx --camera 1

# Adjust confidence threshold (default: 0.45)
python test.py --model model.onnx --conf 0.50
```

#### Live Webcam Controls:
- **`q` or `ESC`**: Exit the application.
- **`+` / `-`**: Dynamically increase / decrease confidence threshold in real time.

---

### How to Train the Model

To retrain the model and export the updated ONNX weights:
```bash
python train.py
```
This script:
1. Trains a YOLOv8 Nano model on the dataset with positive ₹50 notes and hard negative examples (other notes & background).
2. Evaluates performance on the validation split.
3. Automatically exports the best trained weights to `model.onnx` (`opset=12`) for direct OpenCV DNN compatibility.

---

### Dataset Architecture & Negative Examples

```text
dataset_50rs_yolo/
├── data.yaml                  # YOLO dataset configuration
├── train/
│   ├── images/                # 672 images (112 positive ₹50 notes + 560 negative notes)
│   └── labels/                # 112 positive labels (class 0) + 560 empty labels
├── valid/
│   ├── images/                # 126 images (22 positive ₹50 notes + 104 negative notes)
│   └── labels/                # 22 positive labels (class 0) + 104 empty labels
└── test/
    ├── images/                # 42 images (6 positive ₹50 notes + 36 negative notes)
    └── labels/                # 6 positive labels (class 0) + 36 empty labels
```

#### Why Negative Examples are Included:
In object detection, training **only** on positive ₹50 notes causes the model to confuse other currency notes (₹10, ₹20, ₹100, ₹200, ₹500) for ₹50. By pairing photos of other notes with **empty (0-byte) label files**, the model learns hard negative rejection:
- **When showing a ₹50 note**: Draws a green bounding box labelled **`₹50 (score%)`**.
- **When showing any other note or object**: Leaves the screen **completely blank**.

---

### Performance Metrics

- **Validation mAP50**: **98.0%**
- **Precision**: **86.5%**
- **Recall**: **95.5%**
- **False Positive Rejection on Other Notes**: **< 0.1% confidence** (effectively 0)
- **Webcam Latency**: **~5 ms** inference time on GPU / **~25 ms** on CPU (> 30 FPS real-time).

---

### Repository Structure

| File / Folder | Description |
| :--- | :--- |
| **`model.onnx`** | Trained model in ONNX format, ready for direct loading in OpenCV |
| **`test.py`** | OpenCV webcam detection script (`python test.py --model model.onnx`) |
| **`train.py`** | End-to-end training and ONNX export pipeline |
| **`requirements.txt`** | Python dependencies |
| **`dataset_50rs_yolo/`** | Training, validation, and test photos with YOLO annotations |
| **`50_rupees_photos/`** | Isolated collection of all ₹50 note photos |
| **`review_annotations.py`** | Web tool to visually inspect and refine annotations on `localhost:7860` |
| **`README.md`** | Complete submission documentation |
