import os
import cv2
import numpy as np
import ast
import re
from glob import glob
try:
    import yaml
except Exception:
    yaml = None
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
from pathlib import Path
import json

# --- Configuration ---
# Path to your trained model weights
MODEL_WEIGHTS = 'hash_mark_detection/run_v4_data-xl-model/weights/best.pt'

# Root directory of your YOLO dataset (contains train/val/test folders)
DATASET_ROOT = 'E:/data/Football/all-22'

# Which split to process: 'train', 'valid', or 'test'
SPLIT_TO_PROCESS = 'test'  # Change this to 'valid' or 'test' as needed

# Confidence threshold for detections
CONFIDENCE_THRESHOLD = 0.10

# Image extensions to process
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}

# Optional path to a dataset YAML (if you want to force a specific dataset mapping)
DATASET_YAML = "E:/data/Football/all-22/data.yaml"

# Force all new detections to this class index when saving/visualizing.
# Set to None to keep the model's predicted class.
#football dataset has 1:, 2:, 3:, 4: hash marks
FORCE_CLASS = 4
# --- Performance / automation settings ---
# If True, before interactive review the script will run inference for all images and cache results.
PRECOMPUTE_PREDICTIONS = True
# Directory (in cwd) to store per-image prediction caches
PREDICTION_CACHE_DIR = '.pred_cache'
# If a single detection and its confidence >= this, auto-accept and save without interactive review
AUTO_ACCEPT_SINGLE = True
AUTO_ACCEPT_CONFIDENCE = 0.70

# --- Helper Functions ---
def get_split_paths(dataset_root, split):
    """Get the images and labels directories for a specific split."""
    split_dir = os.path.join(dataset_root, split)
    images_dir = os.path.join(split_dir, 'images')
    labels_dir = os.path.join(split_dir, 'labels')
    
    if not os.path.exists(images_dir):
        raise ValueError(f"Images directory not found: {images_dir}")
    
    # Create labels directory if it doesn't exist
    os.makedirs(labels_dir, exist_ok=True)
    
    return images_dir, labels_dir

def create_output_structure(output_dir):
    """Create the output directory structure for YOLO format."""
    images_dir = os.path.join(output_dir, 'images')
    labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    return images_dir, labels_dir

def yolo_to_bbox(yolo_coords, img_width, img_height):
    """Convert YOLO format (normalized) to pixel coordinates."""
    x_center, y_center, width, height = yolo_coords
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)
    
    return x1, y1, x2, y2

def bbox_to_yolo(bbox, img_width, img_height):
    """Convert pixel coordinates to YOLO format (normalized)."""
    x1, y1, x2, y2 = bbox
    
    x_center = ((x1 + x2) / 2) / img_width
    y_center = ((y1 + y2) / 2) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    
    return x_center, y_center, width, height

def load_existing_labels(label_path):
    """Load existing YOLO format labels from a text file."""
    existing_labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_labels.append(line)
    return existing_labels

def save_yolo_annotation(label_path, detections, img_width, img_height, existing_labels):
    """Save detections in YOLO format, appending to existing labels."""
    with open(label_path, 'w') as f:
        # First, write all existing labels
        for label in existing_labels:
            f.write(label + '\n')
        
        # Then append new detections
        for det in detections:
            # Override class id if FORCE_CLASS is set
            class_id = int(det['class']) if FORCE_CLASS is None else int(FORCE_CLASS)
            bbox = det['bbox']
            yolo_coords = bbox_to_yolo(bbox, img_width, img_height)
            f.write(f"{class_id} {yolo_coords[0]:.6f} {yolo_coords[1]:.6f} {yolo_coords[2]:.6f} {yolo_coords[3]:.6f}\n")

def visualize_detections(image, results, class_names, existing_labels=None, img_width=None, img_height=None):
    """Draw bounding boxes on the image for both existing and new detections.

    class_names: model class names (dict or list)
    If dataset_class_names is available it should be passed in place of class_names to display dataset names.
    """
    img_display = image.copy()
    
    # Draw existing labels in blue
    existing_class_names = set()
    if existing_labels and img_width and img_height:
        for label in existing_labels:
            parts = label.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:5])
                x1, y1, x2, y2 = yolo_to_bbox([x_center, y_center, width, height], img_width, img_height)
                # Draw existing annotations in BLUE
                cv2.rectangle(img_display, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # Determine display name
                # Prefer dataset class names if provided via 'class_names' param (can be list or dict)
                try:
                    if isinstance(class_names, dict):
                        name = class_names.get(cls, str(cls))
                    elif isinstance(class_names, (list, tuple)) and cls < len(class_names):
                        name = str(class_names[cls])
                    else:
                        name = str(cls)
                except Exception:
                    name = str(cls)
                existing_class_names.add(name)
                label_text = f"{name} (existing)"
                cv2.putText(img_display, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Draw new detections in green
    new_class_names = set()
    new_count = 0
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get box coordinates, handle both torch tensors and numpy arrays
            xyxy = box.xyxy[0]
            if hasattr(xyxy, 'cpu'):
                coords = xyxy.cpu().numpy()
            else:
                coords = np.array(xyxy)
            x1, y1, x2, y2 = map(int, coords)
            # Safely extract scalar confidence and class values (handle tensors and numpy arrays)
            conf_val = box.conf[0]
            if hasattr(conf_val, 'cpu'):
                try:
                    conf = float(conf_val.cpu().numpy().item())
                except Exception:
                    conf = float(conf_val.cpu().numpy())
            else:
                try:
                    conf = float(np.array(conf_val).item())
                except Exception:
                    conf = float(conf_val)

            cls_val = box.cls[0]
            if hasattr(cls_val, 'cpu'):
                try:
                    cls = int(cls_val.cpu().numpy().item())
                except Exception:
                    cls = int(cls_val.cpu().numpy())
            else:
                try:
                    cls = int(np.array(cls_val).item())
                except Exception:
                    cls = int(cls_val)
            # Determine display name (prefer dataset class names passed in 'class_names' param)
            try:
                if isinstance(class_names, dict):
                    name = class_names.get(cls, str(cls))
                elif isinstance(class_names, (list, tuple)) and cls < len(class_names):
                    name = str(class_names[cls])
                else:
                    name = str(cls)
            except Exception:
                name = str(cls)
            new_class_names.add(name)
            # Draw new predictions in GREEN
            cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw label
            label = f"{name}: {conf:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img_display, (x1, y1 - label_h - 10), (x1 + label_w, y1), (0, 255, 0), -1)
            cv2.putText(img_display, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            new_count += 1
    
    # Legend overlay: show counts and a short list of classes for existing/new
    try:
        exist_count = len(existing_labels) if existing_labels is not None else 0
        exist_names = ",".join(list(existing_class_names)[:3]) if existing_class_names else "-"
        new_names = ",".join(list(new_class_names)[:3]) if new_class_names else "-"
        legend_lines = [f"Existing: {exist_count} ({exist_names})", f"New: {new_count} ({new_names})"]
        pad = 6
        # compute size
        max_w = 0
        total_h = 0
        for i, line in enumerate(legend_lines):
            (w, h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            max_w = max(max_w, w)
            total_h += h + 6
        cv2.rectangle(img_display, (5, 5), (10 + max_w + pad, 10 + total_h + pad), (0, 0, 0), -1)
        y = 10 + 0
        for line in legend_lines:
            (w, h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.putText(img_display, line, (8, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y += h + 6
    except Exception:
        pass

    return img_display

def extract_detections(results, img_width, img_height):
    """Extract detection information from results."""
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            # Safely extract scalar confidence and class values for detection extraction
            conf_val = box.conf[0]
            if hasattr(conf_val, 'cpu'):
                try:
                    conf = float(conf_val.cpu().numpy().item())
                except Exception:
                    conf = float(conf_val.cpu().numpy())
            else:
                try:
                    conf = float(np.array(conf_val).item())
                except Exception:
                    conf = float(conf_val)

            cls_val = box.cls[0]
            if hasattr(cls_val, 'cpu'):
                try:
                    cls = int(cls_val.cpu().numpy().item())
                except Exception:
                    cls = int(cls_val.cpu().numpy())
            else:
                try:
                    cls = int(np.array(cls_val).item())
                except Exception:
                    cls = int(cls_val)
            # If FORCE_CLASS is set, override the predicted class
            if FORCE_CLASS is not None:
                cls = int(FORCE_CLASS)
            detections.append({
                'class': cls,
                'confidence': conf,
                'bbox': (x1, y1, x2, y2)
            })
    return detections

# --- NMS Helper ---
def iou(boxA, boxB):
    """Compute Intersection over Union (IoU) between two bboxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA + 1)
    interH = max(0, yB - yA + 1)
    interArea = interW * interH
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def nms_filter(detections, iou_threshold=0.2):
    """Apply Non-Maximum Suppression to filter overlapping detections."""
    if not detections:
        return []
    # Sort by confidence
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    keep = []
    while detections:
        best = detections.pop(0)
        keep.append(best)
        detections = [d for d in detections if (d['class'] != best['class'] or iou(d['bbox'], best['bbox']) < iou_threshold)]
    return keep


def load_dataset_names(yaml_path=None, dataset_root=None):
    """Load dataset class names from a data.yaml. Returns list or dict or None."""
    # Allow explicit path
    paths_to_try = []
    if yaml_path:
        paths_to_try.append(yaml_path)
    # common locations inside dataset_root
    if dataset_root:
        paths_to_try.extend([
            os.path.join(dataset_root, 'data.yaml'),
            os.path.join(dataset_root, 'dataset.yaml'),
            os.path.join(dataset_root, 'data.yml'),
        ])
    # Also try searching for any yaml in root
    if dataset_root:
        paths_to_try.extend(glob(os.path.join(dataset_root, '*.yaml')))
        paths_to_try.extend(glob(os.path.join(dataset_root, '*.yml')))

    for p in paths_to_try:
        if not p:
            continue
        if os.path.exists(p):
            try:
                if yaml:
                    with open(p, 'r') as f:
                        data = yaml.safe_load(f)
                else:
                    # simple fallback parser for very small yaml: look for names: [...] or names:\n 0: 'a'\n
                    with open(p, 'r') as f:
                        txt = f.read()
                    m = re.search(r'names:\s*(\[.*?\])', txt, flags=re.S)
                    if m:
                        data = {'names': ast.literal_eval(m.group(1))}
                    else:
                        # try to parse lines 'names:\n 0: a' style
                        m2 = re.search(r'names:\s*\n(.*)', txt, flags=re.S)
                        if m2:
                            lines = m2.group(1).strip().splitlines()
                            names = []
                            for ln in lines:
                                ln = ln.strip()
                                if not ln:
                                    continue
                                # accept '- name' or '0: name'
                                if ln.startswith('-'):
                                    names.append(ln.lstrip('-').strip().strip("'\""))
                                else:
                                    parts = ln.split(':', 1)
                                    if len(parts) == 2:
                                        names.append(parts[1].strip().strip("'\""))
                            data = {'names': names}
                        else:
                            data = {}
                names = data.get('names')
                if names is None:
                    continue
                # convert dict to list if needed
                if isinstance(names, dict):
                    # if keys are sequential ints, build list
                    try:
                        maxk = max(int(k) for k in names.keys())
                        lst = [names.get(i) for i in range(maxk+1)]
                        return lst
                    except Exception:
                        return names
                return names
            except Exception:
                continue
    return None


def checkpoint_path_for(labels_dir):
    # Store checkpoint in the current working directory to avoid touching YOLO label folders
    # Include split name to keep per-split checkpoints separate
    cwd = os.getcwd()
    filename = f'.label_checkpoint_{SPLIT_TO_PROCESS}.json'
    return os.path.join(cwd, filename)


def load_checkpoint(labels_dir):
    path = checkpoint_path_for(labels_dir)
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_checkpoint(labels_dir, last_processed=None, last_with_new=None):
    path = checkpoint_path_for(labels_dir)
    data = {}
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except Exception:
            data = {}
    if last_processed is not None:
        data['last_processed'] = last_processed
    if last_with_new is not None:
        data['last_with_new'] = last_with_new
    try:
        with open(path, 'w') as f:
            json.dump(data, f)
    except Exception:
        pass


def progress_path_for():
    cwd = os.getcwd()
    filename = f'.label_progress_{SPLIT_TO_PROCESS}.json'
    return os.path.join(cwd, filename)


def ensure_pred_cache_dir():
    os.makedirs(PREDICTION_CACHE_DIR, exist_ok=True)


def pred_cache_path(img_filename):
    safe = img_filename.replace(os.sep, '_')
    return os.path.join(PREDICTION_CACHE_DIR, safe + '.json')


def load_pred_cache(img_filename):
    p = pred_cache_path(img_filename)
    if os.path.exists(p):
        try:
            with open(p, 'r') as f:
                return json.load(f)
        except Exception:
            return None
    return None


def save_pred_cache(img_filename, data):
    ensure_pred_cache_dir()
    p = pred_cache_path(img_filename)
    try:
        with open(p, 'w') as f:
            json.dump(data, f)
    except Exception:
        pass


def load_progress():
    path = progress_path_for()
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_progress(progress_map):
    path = progress_path_for()
    try:
        with open(path, 'w') as f:
            json.dump(progress_map, f)
    except Exception:
        pass

# --- Main Annotation Pipeline ---
def interactive_annotation_pipeline():
    """
    Load model, process images one by one, show predictions,
    and save annotations only if approved by the user.
    """
    print("--- Interactive YOLO Annotation Tool ---\n")
    
    # 1. Load the trained model
    if not os.path.exists(MODEL_WEIGHTS):
        print(f"❌ Error: Model weights not found at: {MODEL_WEIGHTS}")
        print("Please update the MODEL_WEIGHTS path in the script.")
        return
    
    print(f"Loading model from: {MODEL_WEIGHTS}")
    model = YOLO(MODEL_WEIGHTS)
    class_names = model.names  # Dictionary mapping class IDs to names
    print(f"✅ Model loaded successfully!")
    print(f"Class names: {class_names}\n")

    # Try to load dataset class names (prefer DATASET_YAML or files in DATASET_ROOT)
    dataset_class_names = None
    try:
        dataset_class_names = load_dataset_names(DATASET_YAML, DATASET_ROOT)
        if dataset_class_names:
            print(f"Using dataset class names from YAML: {dataset_class_names}\n")
    except Exception:
        dataset_class_names = None
    
    # 2. Get paths for the specified split
    try:
        images_dir, labels_dir = get_split_paths(DATASET_ROOT, SPLIT_TO_PROCESS)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return
    
    print(f"Processing split: {SPLIT_TO_PROCESS}")
    print(f"Dataset root: {DATASET_ROOT}")
    print(f"  - Images: {images_dir}")
    print(f"  - Labels: {labels_dir}\n")
    
    # 3. Get list of images to process
    image_files = [
        f for f in os.listdir(images_dir)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ]
    
    if not image_files:
        print(f"❌ No images found in: {images_dir}")
        return
    
    print(f"Found {len(image_files)} images to process.\n")
    print("=" * 60)
    print("INSTRUCTIONS:")
    print("  - BLUE boxes = Existing annotations (will be kept)")
    print("  - GREEN boxes = New predictions from your model")
    print("  - Review each image with predicted annotations")
    print("  - Press 'y' or ENTER to APPROVE and ADD new detections")
    print("  - Press 'n' to REJECT and keep only existing labels")
    print("  - Press 'q' to QUIT the annotation process")
    print("=" * 60)
    print()
    
    approved_count = 0
    rejected_count = 0

    # Load per-image progress map to resume
    progress_map = load_progress()
    # progress_map is a dict: {"image_filename": true/false}
    if progress_map:
        print(f"Loaded progress map with {len(progress_map)} entries. Resuming...")
    
    # Optional: precompute predictions for all images (fill cache)
    if PRECOMPUTE_PREDICTIONS:
        print("Precomputing predictions for all images... this may take a while")
        ensure_pred_cache_dir()
        for img_file in image_files:
            cached = load_pred_cache(img_file)
            if cached is not None:
                continue
            img_path = os.path.join(images_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            results = model(img, conf=CONFIDENCE_THRESHOLD, verbose=False)
            # serialize predictions: list of boxes (x1,y1,x2,y2), conf, cls
            preds = []
            for r in results:
                for box in r.boxes:
                    xy = box.xyxy[0]
                    if hasattr(xy, 'cpu'):
                        coords = xy.cpu().numpy().tolist()
                    else:
                        coords = list(xy)
                    confv = float(box.conf[0])
                    clsv = int(box.cls[0])
                    preds.append({'bbox': coords, 'confidence': confv, 'class': clsv})
            save_pred_cache(img_file, preds)
        print("Precompute complete.")

    # 4. Process each image
    for idx, img_file in enumerate(image_files, 1):
        img_path = os.path.join(images_dir, img_file)
        # Skip files already processed according to progress_map
        if img_file in progress_map:
            # already processed (True if had new detections added, False otherwise)
            continue
        print(f"[{idx}/{len(image_files)}] Processing: {img_file}")
        
        # Check if label already exists
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)
        
        # Load existing labels
        existing_labels = load_existing_labels(label_path)
        num_existing = len(existing_labels)
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print(f"  ⚠️ Warning: Could not load image, skipping...")
            continue
        
        img_height, img_width = image.shape[:2]
        
        # Run inference (use cache if available)
        cached = load_pred_cache(img_file)
        if cached is not None:
            # convert cached to detections
            detections = []
            for p in cached:
                x1, y1, x2, y2 = map(int, p['bbox'])
                detections.append({'class': int(p['class']), 'confidence': float(p['confidence']), 'bbox': (x1, y1, x2, y2)})
            num_detections = len(detections)
            print(f"  (cached) New detections: {num_detections}")
        else:
            results = model(image, conf=CONFIDENCE_THRESHOLD, verbose=False)
            num_detections = len(results[0].boxes)
        print(f"  Existing annotations: {num_existing}")
        print(f"  New detections: {num_detections}")
        
        # If no new detections, skip
        if num_detections == 0:
            print(f"  ⚠️ No new detections found, skipping...")
            progress_map[img_file] = False
            save_progress(progress_map)
            continue
        
        # Extract detections and filter with NMS for both visualization and saving
        if cached is not None:
            detections = detections
        else:
            detections = extract_detections(results, img_width, img_height)
        detections_nms = nms_filter(detections, iou_threshold=0.01)

        # Save to prediction cache
        try:
            save_pred_cache(img_file, [{'bbox': list(d['bbox']), 'confidence': d['confidence'], 'class': d['class']} for d in detections])
        except Exception:
            pass

        # Auto-accept rule: single high-confidence detection
        if AUTO_ACCEPT_SINGLE and len(detections_nms) == 1 and detections_nms[0]['confidence'] >= AUTO_ACCEPT_CONFIDENCE:
            print(f"  🔁 Auto-accepting single detection with conf {detections_nms[0]['confidence']:.3f}")
            save_yolo_annotation(label_path, detections_nms, img_width, img_height, existing_labels)
            approved_count += 1
            progress_map[img_file] = True
            save_progress(progress_map)
            print(f"  💾 Updated label file: {label_path} (auto-accepted)")
            continue

        # Prepare a mock results object for visualization with only NMS detections
        class MockBox:
            def __init__(self, bbox, conf, cls):
                self.xyxy = [np.array(bbox)]
                self.conf = [np.array([conf])]
                self.cls = [np.array([cls])]
        class MockResult:
            def __init__(self, boxes):
                self.boxes = boxes
        # Use FORCE_CLASS for mock boxes if set
        mock_boxes = [MockBox(d['bbox'], d['confidence'], (int(d['class']) if FORCE_CLASS is None else int(FORCE_CLASS))) for d in detections_nms]
        mock_results = [MockResult(mock_boxes)]

        # Visualize detections (existing in blue, new in green, but new are NMS-filtered)
        # Pass dataset_class_names if available so preview shows dataset labels
        viz_names = dataset_class_names if dataset_class_names is not None else class_names

        # builder for mock_results from current detections_nms
        def build_mock_results():
            mock_boxes = [MockBox(d['bbox'], d['confidence'], (int(d['class']) if FORCE_CLASS is None else int(FORCE_CLASS))) for d in detections_nms]
            return [MockResult(mock_boxes)]

        mock_results = build_mock_results()

        # Initial draw
        img_display = visualize_detections(image, mock_results, viz_names, existing_labels, img_width, img_height)
        img_display_rgb = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)

        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(img_display_rgb)
        title_text = f"Image {idx}/{len(image_files)}: {img_file}\n"
        title_text += f"BLUE=Existing({num_existing}) | GREEN=New({len(detections_nms)})\n"
        title_text += "Click a GREEN box to delete it. Press 'y' to ADD, 'n' to skip, 'q' to quit"
        ax.set_title(title_text)
        ax.axis('off')
        plt.tight_layout()

        # interactive handlers
        user_input = None

        # undo stack: records ('add', det) or ('remove', det, index)
        undo_stack = []
        # current class to use when drawing (start from FORCE_CLASS or 0)
        current_draw_class = int(FORCE_CLASS) if FORCE_CLASS is not None else 0

        def get_num_classes():
            # prefer dataset_class_names if present
            names = dataset_class_names if dataset_class_names is not None else class_names
            try:
                if isinstance(names, dict):
                    return max(int(k) for k in names.keys()) + 1
                return len(names)
            except Exception:
                return max(1, current_draw_class + 1)

        def on_key(event):
            nonlocal user_input, current_draw_class
            # undo
            if event.key == 'u':
                if not undo_stack:
                    print('  ↩️ Nothing to undo')
                    return
                action = undo_stack.pop()
                if action[0] == 'remove':
                    _, det, idx_old = action
                    try:
                        detections_nms.insert(idx_old, det)
                        print(f"  ↩️ Undo remove: restored detection {det}")
                    except Exception:
                        detections_nms.append(det)
                        print(f"  ↩️ Undo remove: appended detection {det}")
                elif action[0] == 'add':
                    _, det = action
                    found = None
                    for i, d in enumerate(detections_nms):
                        if d['bbox'] == det['bbox'] and d['class'] == det['class']:
                            found = i
                            break
                    if found is not None:
                        detections_nms.pop(found)
                        print(f"  ↩️ Undo add: removed detection {det}")
                # redraw
                new_mock = build_mock_results()
                new_img = visualize_detections(image, new_mock, viz_names, existing_labels, img_width, img_height)
                im.set_data(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))
                ax.set_title(f"Image {idx}/{len(image_files)}: {img_file}\nBLUE=Existing({num_existing}) | GREEN=New({len(detections_nms)})\nClick a GREEN box to delete it. Right-drag to draw a new box. Press 'y' to ADD, 'n' to skip, 'q' to quit | Current class: {current_draw_class}")
                fig.canvas.draw_idle()
                return
            # cycle class down/up
            if event.key == '[':
                numc = get_num_classes()
                current_draw_class = (current_draw_class - 1) % numc
                ax.set_title(f"Image {idx}/{len(image_files)}: {img_file}\nBLUE=Existing({num_existing}) | GREEN=New({len(detections_nms)})\nClick a GREEN box to delete it. Right-drag to draw a new box. Press 'y' to ADD, 'n' to skip, 'q' to quit | Current class: {current_draw_class}")
                fig.canvas.draw_idle()
                return
            if event.key == ']':
                numc = get_num_classes()
                current_draw_class = (current_draw_class + 1) % numc
                ax.set_title(f"Image {idx}/{len(image_files)}: {img_file}\nBLUE=Existing({num_existing}) | GREEN=New({len(detections_nms)})\nClick a GREEN box to delete it. Right-drag to draw a new box. Press 'y' to ADD, 'n' to skip, 'q' to quit | Current class: {current_draw_class}")
                fig.canvas.draw_idle()
                return
            # accept 'return' as enter in some backends
            if event.key in ['y', 'enter', 'n', 'q', 'return']:
                user_input = event.key
                plt.close()

        # state for drawing new box
        draw_state = {'drawing': False, 'x0': None, 'y0': None, 'rect': None}

        def on_click(event):
            # left click: try to delete existing detection
            if event.inaxes != ax:
                return
            # RIGHT click starts drawing (button==3)
            if event.button == 3:
                try:
                    draw_state['drawing'] = True
                    draw_state['x0'] = event.xdata
                    draw_state['y0'] = event.ydata
                    # create rectangle patch
                    if draw_state['rect']:
                        try:
                            draw_state['rect'].remove()
                        except Exception:
                            pass
                    rect = patches.Rectangle((draw_state['x0'], draw_state['y0']), 0, 0, linewidth=2, edgecolor='yellow', facecolor='none')
                    draw_state['rect'] = rect
                    ax.add_patch(rect)
                    fig.canvas.draw_idle()
                except Exception:
                    pass
                return
            if event.button != 1:
                return
            try:
                x = int(event.xdata)
                y = int(event.ydata)
            except Exception:
                return
            # find detections that contain the click point
            candidates = [(i, d) for i, d in enumerate(detections_nms) if d['bbox'][0] <= x <= d['bbox'][2] and d['bbox'][1] <= y <= d['bbox'][3]]
            if not candidates:
                return
            # choose the highest confidence candidate to remove
            candidates.sort(key=lambda tup: tup[1]['confidence'], reverse=True)
            idx_to_remove = candidates[0][0]
            removed = detections_nms.pop(idx_to_remove)
            # push to undo stack with original index
            undo_stack.append(('remove', removed, idx_to_remove))
            print(f"  🗑 Removed detection: class {removed['class']} conf {removed['confidence']:.3f} bbox {removed['bbox']}")
            # rebuild and redraw
            new_mock = build_mock_results()
            new_img = visualize_detections(image, new_mock, viz_names, existing_labels, img_width, img_height)
            im.set_data(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))
            # update title to show new count
            ax.set_title(f"Image {idx}/{len(image_files)}: {img_file}\nBLUE=Existing({num_existing}) | GREEN=New({len(detections_nms)})\nClick a GREEN box to delete it. Right-drag to draw a new box. Press 'y' to ADD, 'n' to skip, 'q' to quit | Current class: {current_draw_class}")
            fig.canvas.draw_idle()

        cid_click = fig.canvas.mpl_connect('button_press_event', on_click)
        cid_key = fig.canvas.mpl_connect('key_press_event', on_key)
        # motion and release events for drawing
        def on_motion(event):
            if not draw_state['drawing'] or draw_state['rect'] is None or event.inaxes != ax:
                return
            try:
                x0 = draw_state['x0']
                y0 = draw_state['y0']
                x1 = event.xdata
                y1 = event.ydata
                xmin = min(x0, x1)
                ymin = min(y0, y1)
                w = abs(x1 - x0)
                h = abs(y1 - y0)
                draw_state['rect'].set_xy((xmin, ymin))
                draw_state['rect'].set_width(w)
                draw_state['rect'].set_height(h)
                fig.canvas.draw_idle()
            except Exception:
                pass

        def on_release(event):
            # finish drawing
            if not draw_state['drawing']:
                return
            draw_state['drawing'] = False
            if draw_state['rect'] is None or event.inaxes != ax:
                return
            try:
                x0 = draw_state['x0']
                y0 = draw_state['y0']
                x1 = event.xdata
                y1 = event.ydata
                xmin = int(max(0, min(x0, x1)))
                ymin = int(max(0, min(y0, y1)))
                xmax = int(min(img_width - 1, max(x0, x1)))
                ymax = int(min(img_height - 1, max(y0, y1)))
                if xmax <= xmin or ymax <= ymin:
                    # invalid box
                    try:
                        draw_state['rect'].remove()
                    except Exception:
                        pass
                    draw_state['rect'] = None
                    fig.canvas.draw_idle()
                    return
                # add new detection with current_draw_class (respect FORCE_CLASS if set)
                new_class = int(FORCE_CLASS) if FORCE_CLASS is not None else int(current_draw_class)
                new_det = {'class': new_class, 'confidence': 1.0, 'bbox': (xmin, ymin, xmax, ymax)}
                detections_nms.append(new_det)
                # push to undo stack
                undo_stack.append(('add', new_det))
                print(f"  ➕ Added detection: class {new_det['class']} bbox {new_det['bbox']}")
                # remove rectangle patch and redraw
                try:
                    draw_state['rect'].remove()
                except Exception:
                    pass
                draw_state['rect'] = None
                new_mock = build_mock_results()
                new_img = visualize_detections(image, new_mock, viz_names, existing_labels, img_width, img_height)
                im.set_data(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))
                ax.set_title(f"Image {idx}/{len(image_files)}: {img_file}\nBLUE=Existing({num_existing}) | GREEN=New({len(detections_nms)})\nClick a GREEN box to delete it. Right-drag to draw a new box. Press 'y' to ADD, 'n' to skip, 'q' to quit | Current class: {current_draw_class}")
                fig.canvas.draw_idle()
            except Exception:
                pass

        cid_motion = fig.canvas.mpl_connect('motion_notify_event', on_motion)
        cid_release = fig.canvas.mpl_connect('button_release_event', on_release)
        plt.show()
        try:
            fig.canvas.mpl_disconnect(cid_click)
        except Exception:
            pass
        try:
            fig.canvas.mpl_disconnect(cid_key)
        except Exception:
            pass
        # Handle user decision
        if user_input == 'q':
            print("\n  🛑 Quitting annotation process...")
            # save progress and quit
            save_progress(progress_map)
            break
        elif user_input in ['y', 'enter']:
            print("  ✅ APPROVED - Adding new annotations to existing labels...")
            # Save labels (append to existing)
            save_yolo_annotation(label_path, detections_nms, img_width, img_height, existing_labels)
            approved_count += 1
            # mark progress: True means new detections were added and saved
            progress_map[img_file] = True
            save_progress(progress_map)
            print(f"  💾 Updated label file: {label_path}")
            print(f"     Total annotations: {num_existing} existing + {len(detections_nms)} new = {num_existing + len(detections_nms)}")
        else:
            print("  ❌ REJECTED - Keeping only existing labels...")
            rejected_count += 1
            # mark progress: False means no new detections were added (either rejected or none found)
            progress_map[img_file] = False
            save_progress(progress_map)
        print()
    
    # 5. Summary
    print("\n" + "=" * 60)
    print("ANNOTATION SUMMARY")
    print("=" * 60)
    print(f"Total images reviewed: {idx}")
    print(f"✅ New detections added: {approved_count}")
    print(f"❌ Rejected (kept existing only): {rejected_count}")
    print(f"\nUpdated labels in: {labels_dir}")
    print("=" * 60)

if __name__ == "__main__":
    interactive_annotation_pipeline()