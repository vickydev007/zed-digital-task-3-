"""
FEDS Technical Assignment - Problem Statement 3
₹50 Note Live Webcam Detector (OpenCV + ONNX)
=============================================
Usage:
    python test.py --model model.onnx

Goal:
    Detects a ₹50 note in a live camera feed and draws a box labelled
    "₹50" with its confidence score when the note is in front of the camera.
    Does not detect other notes or objects (leaves them blank).
"""

import argparse
import sys
import time
from pathlib import Path
import cv2
import numpy as np

def draw_label(img, text, x1, y1, bg_color=(0, 255, 0), text_color=(0, 0, 0)):
    """Draws a label banner with Unicode support (₹ symbol) using PIL when available,
    with automatic fallback to OpenCV putText."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        # Load system font supporting Unicode currency symbols
        font = None
        for f in ["arial.ttf", "segoeui.ttf", "calibri.ttf", "DejaVuSans.ttf"]:
            try:
                font = ImageFont.truetype(f, 20)
                break
            except Exception:
                pass
        if font is None:
            font = ImageFont.load_default()

        # Text bounding box
        bbox = draw.textbbox((x1, y1), text, font=font)
        pad = 6
        banner_x1 = max(0, x1)
        banner_y1 = max(0, y1 - (bbox[3] - bbox[1]) - pad * 2)
        banner_x2 = banner_x1 + (bbox[2] - bbox[1]) + pad * 2 + 10
        banner_y2 = y1

        draw.rectangle([banner_x1, banner_y1, banner_x2, banner_y2], fill=bg_color)
        draw.text((banner_x1 + pad, banner_y1 + pad - 2), text, font=font, fill=text_color)
        
        # Convert back to OpenCV BGR
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        # Fallback to standard OpenCV putText (ASCII)
        ascii_text = text.replace("₹", "Rs ")
        (tw, th), _ = cv2.getTextSize(ascii_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        by1 = max(0, y1 - th - 12)
        cv2.rectangle(img, (x1, by1), (x1 + tw + 12, y1), bg_color, -1)
        cv2.putText(img, ascii_text, (x1 + 6, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.65, text_color, 2, cv2.LINE_AA)
        return img

def main():
    parser = argparse.ArgumentParser(description="Live ₹50 Note Detector using OpenCV and ONNX model")
    parser.add_argument("--model", type=str, default="model.onnx", help="Path to trained ONNX model")
    parser.add_argument("--conf", type=float, default=0.45, help="Confidence threshold (default: 0.45)")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[!] Error: Model file not found at: {model_path}")
        print("Please train the model first with:")
        print("    python train.py")
        sys.exit(1)

    print("=" * 65)
    print("FEDS Technical Assignment - ₹50 Note Live Detector")
    print("=" * 65)
    print(f"Loading ONNX model: {model_path}")
    print(f"Confidence threshold: {args.conf * 100:.0f}%")

    # Load ONNX model directly in OpenCV DNN
    try:
        net = cv2.dnn.readNetFromONNX(str(model_path))
        # Prefer CUDA backend if available, fallback to CPU
        try:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("[*] OpenCV DNN using CUDA acceleration.")
        except Exception:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print("[*] OpenCV DNN using CPU.")
    except Exception as e:
        print(f"[!] OpenCV DNN failed to load ONNX: {e}")
        sys.exit(1)

    # Initialize Webcam
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[!] Error: Could not access webcam at index {args.camera}.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    conf_thresh = args.conf
    nms_thresh = 0.45
    INPUT_SIZE = 640
    GREEN_COLOR = (0, 255, 0)
    BOX_THICKNESS = 3

    print("\nLive camera feed started!")
    print(" - Press 'q' or ESC to Exit")
    print(" - Press '+' to increase confidence threshold")
    print(" - Press '-' to decrease confidence threshold")
    print("-" * 65)

    window_name = "₹50 Note Detector - FEDS Assignment"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Failed to read frame from webcam.")
            break

        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 1e-5))
        prev_time = curr_time

        orig_h, orig_w = frame.shape[:2]

        # Preprocessing: 640x640 blob with 1/255 scale
        blob = cv2.dnn.blobFromImage(frame, 1.0 / 255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True, crop=False)
        net.setInput(blob)
        output = net.forward()

        # Output shape is (1, 4 + num_classes, 8400)
        # Squeeze and transpose to (8400, 5)
        predictions = np.squeeze(output[0])
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T

        boxes = []
        confidences = []

        # Coordinate scale ratios from 640x640 back to original frame dimensions
        x_factor = orig_w / float(INPUT_SIZE)
        y_factor = orig_h / float(INPUT_SIZE)

        for row in predictions:
            # For single-class 50_rupees detector:
            # row[0:4] = cx, cy, w, h
            # row[4] = class 0 confidence score
            score = float(row[4])
            if score >= conf_thresh:
                cx, cy, bw, bh = row[0:4]
                # Convert center coordinates to top-left (x1, y1) in original pixels
                x1 = int((cx - bw / 2.0) * x_factor)
                y1 = int((cy - bh / 2.0) * y_factor)
                w = int(bw * x_factor)
                h = int(bh * y_factor)

                boxes.append([x1, y1, w, h])
                confidences.append(score)

        # Apply Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, nms_thresh)
        detected_count = 0

        if len(indices) > 0:
            for i in indices.flatten():
                detected_count += 1
                bx, by, bw, bh = boxes[i]
                score = confidences[i]

                # Draw vibrant green box
                x2 = bx + bw
                y2 = by + bh
                cv2.rectangle(frame, (bx, by), (x2, y2), GREEN_COLOR, BOX_THICKNESS)

                # Draw label: "₹50 (score%)"
                label_text = f"₹50 ({score * 100:.1f}%)"
                frame = draw_label(frame, label_text, bx, by, bg_color=GREEN_COLOR)

        # Dashboard status header
        status_banner = f"FPS: {fps:.1f} | Conf Min: {int(conf_thresh * 100)}% | ₹50 Detected: {detected_count}"
        cv2.rectangle(frame, (10, 10), (450, 42), (25, 25, 25), -1)
        cv2.rectangle(frame, (10, 10), (450, 42), (90, 90, 90), 1)
        text_color = GREEN_COLOR if detected_count > 0 else (255, 255, 255)
        cv2.putText(frame, status_banner, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.52, text_color, 1, cv2.LINE_AA)

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in [ord("q"), 27]: # 'q' or ESC
            break
        elif key in [ord("+"), ord("=")]:
            conf_thresh = min(0.95, conf_thresh + 0.05)
            print(f"[*] Confidence threshold increased: {conf_thresh:.2f}")
        elif key in [ord("-"), ord("_")]:
            conf_thresh = max(0.20, conf_thresh - 0.05)
            print(f"[*] Confidence threshold decreased: {conf_thresh:.2f}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
