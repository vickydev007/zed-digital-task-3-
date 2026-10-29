"""
FEDS Technical Assignment - Problem Statement 3
₹50 Note Detector - Model Training & ONNX Export Script
======================================================
Trains YOLO on ₹50 note images (front & back) with hard negative
examples (₹10, ₹20, ₹100, ₹200, ₹500) and exports to ONNX format.
"""

import shutil
import sys
from pathlib import Path
import torch

def main():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[!] Ultralytics is not installed. Please run:")
        print("    pip install ultralytics")
        sys.exit(1)

    base_dir = Path(__file__).resolve().parent
    data_yaml = base_dir / "dataset_50rs_yolo" / "data.yaml"

    if not data_yaml.exists():
        print(f"[!] Dataset yaml config not found at: {data_yaml}")
        sys.exit(1)

    print("=" * 65)
    print("FEDS Technical Assignment - ₹50 Note Detector Training")
    print("=" * 65)
    print(f"Dataset: {data_yaml}")
    print("Model: YOLOv8 Nano (Fast, lightweight, live webcam friendly)")
    print("-" * 65)

    # Load YOLOv8 pretrained base model
    model = YOLO("yolov8n.pt")

    # Train model
    # With 112 positive and 560 negative images, 35 epochs provides high convergence
    train_results = model.train(
        data=str(data_yaml),
        epochs=30,
        imgsz=640,
        batch=8,
        workers=0,
        patience=8,
        device=0 if torch.cuda.is_available() else "cpu",
        project=str(base_dir / "runs"),
        name="50rs_detector",
        save=True,
        verbose=True
    )

    # Path to best trained weights
    best_pt = base_dir / "runs" / "50rs_detector" / "weights" / "best.pt"
    print("\n" + "=" * 65)
    print(f"Training completed! Best PyTorch weights saved at: {best_pt}")

    # Export to ONNX for direct loading in OpenCV
    print("Exporting model to ONNX format (.onnx) for OpenCV compatibility...")
    best_model = YOLO(str(best_pt))
    exported_path = best_model.export(
        format="onnx",
        imgsz=640,
        opset=12,
        simplify=True
    )

    # Copy exported ONNX model to root as 'model.onnx'
    target_onnx = base_dir / "model.onnx"
    shutil.copy2(exported_path, target_onnx)
    print("=" * 65)
    print(f"SUCCESS: Trained ONNX model saved as: {target_onnx.name}")
    print(f"Location: {target_onnx}")
    print("You can now test live on webcam using:")
    print(f"    python test.py --model {target_onnx.name}")
    print("=" * 65)

if __name__ == "__main__":
    main()
