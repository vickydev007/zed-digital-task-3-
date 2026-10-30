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

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

def letterbox(img, target_size=(640, 640)):
    """Resize image to target size while maintaining aspect ratio using padding (114)."""
    h, w = img.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_size[0], target_size[1], 3), 114, dtype=np.uint8)
    dx = (target_size[1] - nw) // 2
    dy = (target_size[0] - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = resized
    return canvas, scale, dx, dy

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

def process_frame(frame, net, conf_thresh, nms_thresh=0.45, input_size=640):
    """Processes a single frame using letterboxed ONNX inference and returns annotated frame and stats."""
    GREEN_COLOR = (0, 255, 0)
    BOX_THICKNESS = 3

    # Preprocessing: letterbox to maintain banknote aspect ratio without distortion
    canvas, scale, dx, dy = letterbox(frame, (input_size, input_size))
    blob = cv2.dnn.blobFromImage(canvas, 1.0 / 255.0, (input_size, input_size), swapRB=True, crop=False)
    net.setInput(blob)
    output = net.forward()

    # Output shape is (1, 4 + num_classes, 8400) -> transpose to (8400, 5)
    predictions = np.squeeze(output[0])
    if predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.T

    boxes = []
    confidences = []
    max_score = 0.0

    for row in predictions:
        score = float(row[4])
        if score > max_score:
            max_score = score
        if score >= conf_thresh:
            cx, cy, bw, bh = row[0:4]
            # Inverse transform from letterboxed canvas coordinates back to original frame coordinates
            orig_cx = (cx - dx) / scale
            orig_cy = (cy - dy) / scale
            orig_w = bw / scale
            orig_h = bh / scale

            x1 = int(orig_cx - orig_w / 2.0)
            y1 = int(orig_cy - orig_h / 2.0)
            w = int(orig_w)
            h = int(orig_h)

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

    return frame, detected_count, max_score

def main():
    parser = argparse.ArgumentParser(description="Live ₹50 Note Detector using OpenCV and ONNX model")
    parser.add_argument("--model", type=str, default="model.onnx", help="Path to trained ONNX model")
    parser.add_argument("--conf", type=float, default=0.30, help="Confidence threshold (default: 0.30)")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--source", type=str, default=None, help="Optional image or video file path")
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
    print(f"Initial confidence threshold: {args.conf * 100:.0f}%")

    # Load ONNX model directly in OpenCV DNN
    try:
        net = cv2.dnn.readNetFromONNX(str(model_path))
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

    conf_thresh = args.conf
    nms_thresh = 0.45
    GREEN_COLOR = (0, 255, 0)
    window_name = "₹50 Note Detector"

    # Single Image Test Mode
    if args.source and Path(args.source).is_file() and Path(args.source).suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        image_path = Path(args.source)
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"[!] Error: Could not read image at {image_path}")
            sys.exit(1)

        result_frame, detected_count, max_score = process_frame(frame, net, conf_thresh, nms_thresh)
        print(f"[*] Processed {image_path.name}: ₹50 detected={detected_count}, max confidence={max_score * 100:.1f}%")

        status_banner = f"Conf Min: {int(conf_thresh * 100)}% | Max Raw: {int(max_score * 100)}% | ₹50 Detected: {detected_count}"
        cv2.rectangle(result_frame, (10, 10), (520, 42), (25, 25, 25), -1)
        cv2.rectangle(result_frame, (10, 10), (520, 42), (90, 90, 90), 1)
        text_color = GREEN_COLOR if detected_count > 0 else (255, 255, 255)
        cv2.putText(result_frame, status_banner, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.50, text_color, 1, cv2.LINE_AA)

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, result_frame)
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # Video or Camera Feed Mode
    cap_source = args.source if args.source else args.camera
    try:
        cap_source = int(cap_source)
    except (ValueError, TypeError):
        pass

    if isinstance(cap_source, int) and sys.platform.startswith("win"):
        # Try DirectShow first on Windows to avoid MSMF frame drop errors
        cap = cv2.VideoCapture(cap_source, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(cap_source)
    else:
        cap = cv2.VideoCapture(cap_source)

    if not cap.isOpened():
        print(f"[!] Error: Could not access video/webcam source: {cap_source}.")
        sys.exit(1)

    if isinstance(cap_source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\nLive camera feed started!")
    print(" - Press 'q' or ESC to Exit")
    print(" - Press '+' to increase confidence threshold")
    print(" - Press '-' to decrease confidence threshold")
    print("-" * 65)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] End of video or failed to read frame.")
            break

        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 1e-5))
        prev_time = curr_time

        result_frame, detected_count, max_score = process_frame(frame, net, conf_thresh, nms_thresh)

        # Dashboard status header showing FPS, Conf Min threshold, Max Raw score, and Detection count
        status_banner = f"FPS: {fps:.1f} | Conf Min: {int(conf_thresh * 100)}% | Max Raw: {int(max_score * 100)}% | ₹50 Detected: {detected_count}"
        cv2.rectangle(result_frame, (10, 10), (540, 42), (25, 25, 25), -1)
        cv2.rectangle(result_frame, (10, 10), (540, 42), (90, 90, 90), 1)
        text_color = GREEN_COLOR if detected_count > 0 else (255, 255, 255)
        cv2.putText(result_frame, status_banner, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.50, text_color, 1, cv2.LINE_AA)

        cv2.imshow(window_name, result_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in [ord("q"), 27]: # 'q' or ESC
            break
        elif key in [ord("+"), ord("=")]:
            conf_thresh = min(0.95, conf_thresh + 0.05)
            print(f"[*] Confidence threshold increased: {conf_thresh:.2f}")
        elif key in [ord("-"), ord("_")]:
            conf_thresh = max(0.15, conf_thresh - 0.05)
            print(f"[*] Confidence threshold decreased: {conf_thresh:.2f}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
