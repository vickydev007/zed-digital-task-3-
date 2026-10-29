Problem Statement 3: ₹50 Note Detector

Goal
Train a model that detects the "new light blue-green ₹50 note" in a live camera feed and draws a box labelled “₹50” with its confidence score when the note is in front of the camera. The model must not detect other notes (₹10, ₹20, ₹100, ₹200, ₹500) or objects as ₹50.

Target Banknote Specifications
- Denomination: New light blue-green ₹50 banknote
  - Front: Mahatma Gandhi portrait
  - Back: Hampi stone chariot
- Exclusions: Older ₹50 notes, other currency denominations, and non-currency objects.


Quick Start & Run Command

 1. Install Dependencies
```bash
pip install -r requirements.txt
```

 2. Run Live Webcam Detection
```bash
python test.py --model model.onnx
```

# Adjust confidence threshold (default: 0.45)
python test.py --model model.onnx --conf 0.50


Live Webcam Controls:
- `q` or `ESC`: Exit the application.
- `+` / `-`: Dynamically increase / decrease confidence threshold in real time.

How to Train the Model

To retrain the model and export the updated ONNX weights:
bash
python train.py

This script:
1. Trains a YOLOv8 Nano model on the dataset with positive ₹50 notes and hard negative examples (other notes & background).
2. Evaluates performance on the validation split.
3. Automatically exports the best trained weights to `model.onnx` (`opset=12`) for direct OpenCV DNN compatibility.


Performance Metrics
Validation mAP50: 98.0%
Precision: 86.5%
Recall: 95.5%
False Positive Rejection on Other Notes: < 0.1% confidence (effectively 0)
Webcam Latency: ~5 ms inference time on GPU / ~25 ms on CPU (> 30 FPS real-time).

Dataset:
170 - positive photos
700 - Negative photos 


