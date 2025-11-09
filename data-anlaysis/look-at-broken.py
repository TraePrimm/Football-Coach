import os
import cv2
import numpy as np
from pathlib import Path
import re

def parse_anomaly_report(report_path):
    """Parse the anomaly report to extract files with issues."""
    anomaly_files = set()
    
    with open(report_path, 'r') as f:
        content = f.read()
    
    # Find all file paths in the report
    # Look for lines that start with "File: "
    pattern = r'File: (.+\.txt)'
    matches = re.findall(pattern, content)
    
    for match in matches:
        anomaly_files.add(match)
    
    return list(anomaly_files)

def parse_yolo_label(label_path):
    """Parse YOLO format label file and return list of bounding boxes."""
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:5])
                boxes.append({
                    'class': class_id,
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height
                })
    return boxes

def yolo_to_pixels(box, img_width, img_height):
    """Convert YOLO format (normalized) to pixel coordinates."""
    x_center = box['x_center'] * img_width
    y_center = box['y_center'] * img_height
    width = box['width'] * img_width
    height = box['height'] * img_height
    
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)
    
    return x1, y1, x2, y2

def get_color_for_class(class_id, num_classes=20):
    """Generate a consistent color for each class."""
    np.random.seed(class_id)
    color = tuple(int(x) for x in np.random.randint(50, 255, 3))
    return color

def draw_bboxes(image, boxes):
    """Draw bounding boxes on image."""
    img_height, img_width = image.shape[:2]
    
    for box in boxes:
        x1, y1, x2, y2 = yolo_to_pixels(box, img_width, img_height)
        color = get_color_for_class(box['class'])
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        label = f"Class {box['class']}"
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        cv2.rectangle(
            image,
            (x1, y1 - label_height - baseline - 5),
            (x1 + label_width, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            image,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
    
    return image

def find_image_path(label_path, dataset_path):
    """Find corresponding image file for a label file."""
    label_path = Path(label_path)
    dataset_path = Path(dataset_path)
    
    # Common image extensions
    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # Try to find the image in the images folder parallel to labels folder
    label_relative = label_path.relative_to(dataset_path) if label_path.is_relative_to(dataset_path) else label_path
    
    # Replace 'labels' with 'images' in the path
    image_path_str = str(label_relative).replace('/labels/', '/images/').replace('\\labels\\', '\\images\\')
    
    # Try different extensions
    for ext in extensions:
        image_path = dataset_path / Path(image_path_str).with_suffix(ext)
        if image_path.exists():
            return image_path
    
    return None

def visualize_anomalies(dataset_path, report_path, output_dir="anomaly_visualizations", display_mode="save"):
    """
    Visualize images with anomalous bounding boxes.
    
    Args:
        dataset_path: Path to dataset root
        report_path: Path to the anomaly report file
        output_dir: Directory to save visualization images
        display_mode: "save" to save images, "show" to display interactively, "both" for both
    """
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    
    # Test if cv2.imshow is available
    can_display = False
    if display_mode in ["show", "both"]:
        try:
            # Test if imshow works
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imshow("test", test_img)
            cv2.waitKey(1)
            cv2.destroyAllWindows()
            can_display = True
        except:
            print("Warning: OpenCV display not available. Saving images only.")
            print("To enable display, install: pip install opencv-python-headless")
            if display_mode == "show":
                display_mode = "save"
            else:
                display_mode = "save"  # Change "both" to "save"
    
    # Create output directory
    if display_mode in ["save", "both"]:
        output_dir.mkdir(exist_ok=True)
    
    # Parse anomaly report
    print("Parsing anomaly report...")
    anomaly_files = parse_anomaly_report(report_path)
    
    if not anomaly_files:
        print("No anomalies found in report!")
        return
    
    print(f"Found {len(anomaly_files)} files with anomalies")
    
    # Process each file
    for idx, label_file in enumerate(anomaly_files, 1):
        print(f"\nProcessing {idx}/{len(anomaly_files)}: {label_file}")
        
        # Find corresponding image
        image_path = find_image_path(label_file, dataset_path)
        
        if image_path is None or not image_path.exists():
            print(f"  Warning: Could not find image for {label_file}")
            continue
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"  Warning: Could not load image {image_path}")
            continue
        
        # Parse labels
        boxes = parse_yolo_label(label_file)
        
        if not boxes:
            print(f"  Warning: No boxes found in {label_file}")
            continue
        
        # Draw bounding boxes
        image_with_boxes = draw_bboxes(image.copy(), boxes)
        
        # Add title with filename
        title = f"Anomaly: {Path(label_file).name}"
        cv2.putText(
            image_with_boxes,
            title,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # Save or display
        if display_mode in ["save", "both"]:
            output_path = output_dir / f"{Path(label_file).stem}_annotated.jpg"
            cv2.imwrite(str(output_path), image_with_boxes)
            print(f"  Saved to: {output_path}")
        
        if display_mode in ["show", "both"] and can_display:
            # Resize if image is too large
            max_height = 900
            h, w = image_with_boxes.shape[:2]
            if h > max_height:
                scale = max_height / h
                new_w = int(w * scale)
                image_with_boxes = cv2.resize(image_with_boxes, (new_w, max_height))
            
            try:
                cv2.imshow(f"Anomaly Visualization ({idx}/{len(anomaly_files)})", image_with_boxes)
                print("  Press any key to continue, 'q' to quit...")
                key = cv2.waitKey(0)
                
                if key == ord('q') or key == ord('Q'):
                    print("  Quitting visualization...")
                    cv2.destroyAllWindows()
                    break
            except:
                print("  Display error, continuing with save only...")
    
    if display_mode in ["show", "both"] and can_display:
        try:
            cv2.destroyAllWindows()
        except:
            pass
    
    print(f"\n{'='*80}")
    if display_mode in ["save", "both"]:
        print(f"Visualizations saved to: {output_dir}")
    print(f"Processed {len(anomaly_files)} anomaly files")
    print(f"{'='*80}")

if __name__ == "__main__":
    # Configuration
    dataset_path = "E:/data/Football/hash-marks"  # Change this to your dataset path
    report_path = "E:/data/Football/hash-marks/bbox_anomaly_report.txt"  # Path to the report file

    # Choose display mode:
    # "save" - saves images to disk
    # "show" - displays images interactively (press any key to advance)
    # "both" - saves and displays
    display_mode = "show"
    
    visualize_anomalies(dataset_path, report_path, display_mode=display_mode)