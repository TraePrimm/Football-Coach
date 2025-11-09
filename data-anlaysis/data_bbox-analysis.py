import os
import numpy as np
from pathlib import Path
from collections import defaultdict

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
                    'width': width,
                    'height': height,
                    'area': width * height
                })
    return boxes

def analyze_dataset(dataset_path, std_threshold=2.5):
    """
    Analyze YOLO dataset for unusual bounding box sizes.
    
    Args:
        dataset_path: Path to dataset root (should contain train/val/test folders)
        std_threshold: Number of standard deviations to consider as anomaly
    """
    dataset_path = Path(dataset_path)
    splits = ['train', 'val', 'test']
    
    # Collect all box sizes per class
    class_boxes = defaultdict(lambda: {'widths': [], 'heights': [], 'areas': [], 'files': []})
    
    print("Scanning dataset...")
    for split in splits:
        labels_dir = dataset_path / split / 'labels'
        if not labels_dir.exists():
            print(f"Warning: {labels_dir} does not exist, skipping...")
            continue
        
        for label_file in labels_dir.glob('*.txt'):
            boxes = parse_yolo_label(label_file)
            for box in boxes:
                class_id = box['class']
                class_boxes[class_id]['widths'].append(box['width'])
                class_boxes[class_id]['heights'].append(box['height'])
                class_boxes[class_id]['areas'].append(box['area'])
                class_boxes[class_id]['files'].append(str(label_file))
    
    if not class_boxes:
        print("No labels found in dataset!")
        return
    
    print(f"\nFound {len(class_boxes)} unique classes")
    
    # Calculate statistics per class
    anomalies = []
    
    for class_id, data in class_boxes.items():
        widths = np.array(data['widths'])
        heights = np.array(data['heights'])
        areas = np.array(data['areas'])
        
        # Calculate mean and std
        width_mean, width_std = widths.mean(), widths.std()
        height_mean, height_std = heights.mean(), heights.std()
        area_mean, area_std = areas.mean(), areas.std()
        
        print(f"\nClass {class_id} Statistics:")
        print(f"  Total boxes: {len(widths)}")
        print(f"  Width:  mean={width_mean:.4f}, std={width_std:.4f}")
        print(f"  Height: mean={height_mean:.4f}, std={height_std:.4f}")
        print(f"  Area:   mean={area_mean:.4f}, std={area_std:.4f}")
        
        # Find anomalies
        for i, (w, h, a, file) in enumerate(zip(widths, heights, areas, data['files'])):
            reasons = []
            
            if width_std > 0 and abs(w - width_mean) > std_threshold * width_std:
                reasons.append(f"width={w:.4f} (mean={width_mean:.4f}±{width_std:.4f})")
            
            if height_std > 0 and abs(h - height_mean) > std_threshold * height_std:
                reasons.append(f"height={h:.4f} (mean={height_mean:.4f}±{height_std:.4f})")
            
            if area_std > 0 and abs(a - area_mean) > std_threshold * area_std:
                reasons.append(f"area={a:.4f} (mean={area_mean:.4f}±{area_std:.4f})")
            
            if reasons:
                anomalies.append({
                    'file': file,
                    'class': class_id,
                    'width': w,
                    'height': h,
                    'area': a,
                    'reasons': reasons
                })
    
    # Write report
    report_path = dataset_path / 'bbox_anomaly_report.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("YOLO BOUNDING BOX ANOMALY REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Threshold: {std_threshold} standard deviations\n")
        f.write(f"Total anomalies found: {len(anomalies)}\n\n")
        
        if anomalies:
            # Group by file
            file_anomalies = defaultdict(list)
            for anom in anomalies:
                file_anomalies[anom['file']].append(anom)
            
            f.write("=" * 80 + "\n")
            f.write("FILES WITH ANOMALIES\n")
            f.write("=" * 80 + "\n\n")
            
            for file, anoms in sorted(file_anomalies.items()):
                f.write(f"\nFile: {file}\n")
                f.write("-" * 80 + "\n")
                for anom in anoms:
                    f.write(f"  Class {anom['class']}:\n")
                    for reason in anom['reasons']:
                        f.write(f"    - {reason}\n")
                f.write("\n")
            
            # Summary by class
            f.write("\n" + "=" * 80 + "\n")
            f.write("SUMMARY BY CLASS\n")
            f.write("=" * 80 + "\n\n")
            
            class_anomaly_count = defaultdict(int)
            for anom in anomalies:
                class_anomaly_count[anom['class']] += 1
            
            for class_id, count in sorted(class_anomaly_count.items()):
                f.write(f"Class {class_id}: {count} anomalies\n")
        else:
            f.write("No anomalies detected! All bounding boxes are within expected size ranges.\n")
    
    print(f"\n{'=' * 80}")
    print(f"Report saved to: {report_path}")
    print(f"Total anomalies found: {len(anomalies)}")
    print(f"{'=' * 80}")
    
    return anomalies

if __name__ == "__main__":
    # Set your dataset path here
    dataset_path = "E:/data/Football/hash-marks"  # Change this to your dataset path
    
    # Adjust threshold as needed (lower = more sensitive, higher = less sensitive)
    # 2.5 means boxes that are 2.5 standard deviations away from mean are flagged
    threshold = 2.5
    
    analyze_dataset(dataset_path, std_threshold=threshold)