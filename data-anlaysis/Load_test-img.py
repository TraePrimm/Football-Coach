import argparse
import cv2
import os
import sys
from typing import List, Tuple

#!/usr/bin/env python3
"""
Load_test-img.py

Simple utility to visualize YOLO-format bounding boxes on an image.

Usage:
    python Load_test-img.py --image path/to/img.jpg --labels path/to/img.txt
    python Load_test-img.py -i img.jpg -l img.txt -c classes.txt -o out.jpg --no-show

Labels file format (YOLO):
    class_id x_center y_center width height
    all values normalized [0,1], one bbox per line

command example: python data-anlaysis/Load_test-img.py --image E:\data\Football\all-22\valid\images\week01-Cincinnati-Bengals-CIN-PIT-065-623_jpg.rf.1d8853e7e23ea4a189e86a6683745419.jpg --labels E:\data\Football\all-22\valid\labels\week01-Cincinnati-Bengals-CIN-PIT-065-623_jpg.rf.1d8853e7e23ea4a189e86a6683745419.txt

Optional classes file: one class name per line (index matches class_id).
If not provided, numeric class ids are shown.
"""

def parse_args():
    p = argparse.ArgumentParser(description="Draw YOLO bboxes on an image")
    p.add_argument("-i", "--image", required=True, help="Path to image file")
    p.add_argument("-l", "--labels", required=True, help="Path to YOLO label file for the image")
    p.add_argument("-c", "--classes", help="Optional classes names file (one per line)")
    p.add_argument("-o", "--output", help="Optional output image path to save annotated image")
    p.add_argument("--no-show", action="store_true", help="Do not show window (useful on headless)")
    p.add_argument("--thickness", type=int, default=2, help="Bounding box thickness")
    return p.parse_args()

def load_classes(path: str) -> List[str]:
    if not path:
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    except Exception:
        return []

def parse_yolo_labels(label_path: str) -> List[Tuple[int, float, float, float, float]]:
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split()
            if len(parts) < 5:
                continue
            try:
                cls = int(parts[0])
                xc, yc, w, h = map(float, parts[1:5])
                boxes.append((cls, xc, yc, w, h))
            except Exception:
                continue
    return boxes

def class_color(idx: int) -> Tuple[int, int, int]:
    # deterministic color per class id
    palette = [
        (31,119,180), (255,127,14), (44,160,44), (214,39,40),
        (148,103,189), (140,86,75), (227,119,194), (127,127,127),
        (188,189,34), (23,190,207)
    ]
    return palette[idx % len(palette)]

def draw_boxes(img, boxes, classes, thickness=2):
    h, w = img.shape[:2]
    for cls, xc, yc, bw, bh in boxes:
        x1 = int((xc - bw / 2.0) * w)
        y1 = int((yc - bh / 2.0) * h)
        x2 = int((xc + bw / 2.0) * w)
        y2 = int((yc + bh / 2.0) * h)
        color = class_color(cls)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=thickness)
        label = classes[cls] if (classes and 0 <= cls < len(classes)) else str(cls)
        # text background
        (tw, th_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th_text - 6), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img, label, (x1 + 3, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    return img

def main():
    args = parse_args()
    if not os.path.exists(args.image):
        print(f"Image not found: {args.image}", file=sys.stderr)
        sys.exit(2)
    boxes = parse_yolo_labels(args.labels)
    if not boxes:
        print("No boxes found in label file or file missing/empty.", file=sys.stderr)
    classes = load_classes(args.classes)
    img = cv2.imread(args.image)
    if img is None:
        print("Failed to read image.", file=sys.stderr)
        sys.exit(3)
    annotated = draw_boxes(img.copy(), boxes, classes, thickness=args.thickness)
    if args.output:
        cv2.imencode(os.path.splitext(args.output)[1], annotated)[1].tofile(args.output)
        print(f"Saved annotated image to {args.output}")
    if not args.no_show:
        win = "YOLO BBoxes"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.imshow(win, annotated)
        print("Press any key in the image window to exit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()