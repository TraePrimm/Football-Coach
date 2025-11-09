import re
from pathlib import Path
from collections import defaultdict

def parse_anomaly_report(report_path):
    """Parse the anomaly report to extract files with issues."""
    anomaly_files = []
    
    with open(report_path, 'r') as f:
        content = f.read()
    
    # Find all file paths in the report
    pattern = r'File: (.+\.txt)'
    matches = re.findall(pattern, content)
    
    return matches

def categorize_by_split(anomaly_files):
    """Categorize anomaly files by train/val/test split."""
    split_counts = {
        'train': [],
        'val': [],
        'test': [],
        'unknown': []
    }
    
    for file_path in anomaly_files:
        file_path_lower = file_path.lower()
        
        if '/train/' in file_path_lower or '\\train\\' in file_path_lower:
            split_counts['train'].append(file_path)
        elif '/val/' in file_path_lower or '\\val\\' in file_path_lower:
            split_counts['val'].append(file_path)
        elif '/test/' in file_path_lower or '\\test\\' in file_path_lower:
            split_counts['test'].append(file_path)
        else:
            split_counts['unknown'].append(file_path)
    
    return split_counts

def count_total_files_in_split(dataset_path, split_name):
    """Count total label files in a split."""
    labels_dir = Path(dataset_path) / split_name / 'labels'
    
    if not labels_dir.exists():
        return 0
    
    return len(list(labels_dir.glob('*.txt')))

def analyze_anomaly_distribution(dataset_path, report_path):
    """
    Analyze how anomalies are distributed across train/val/test splits.
    
    Args:
        dataset_path: Path to dataset root
        report_path: Path to the anomaly report file
    """
    dataset_path = Path(dataset_path)
    
    print("="*80)
    print("ANOMALY DISTRIBUTION ANALYSIS")
    print("="*80)
    
    # Parse anomaly report
    print("\nParsing anomaly report...")
    anomaly_files = parse_anomaly_report(report_path)
    
    if not anomaly_files:
        print("No anomalies found in report!")
        return
    
    print(f"Total files with anomalies: {len(anomaly_files)}")
    
    # Categorize by split
    split_counts = categorize_by_split(anomaly_files)
    
    # Count total files in each split
    print("\nCounting total files in dataset...")
    total_counts = {}
    for split in ['train', 'val', 'test']:
        total_counts[split] = count_total_files_in_split(dataset_path, split)
    
    # Display results
    print("\n" + "="*80)
    print("RESULTS BY SPLIT")
    print("="*80)
    
    for split in ['train', 'val', 'test']:
        anomaly_count = len(split_counts[split])
        total_count = total_counts[split]
        
        if total_count > 0:
            percentage = (anomaly_count / total_count) * 100
            print(f"\n{split.upper()}:")
            print(f"  Files with anomalies: {anomaly_count}")
            print(f"  Total files: {total_count}")
            print(f"  Percentage: {percentage:.2f}%")
        else:
            print(f"\n{split.upper()}:")
            print(f"  No files found in dataset")
    
    if split_counts['unknown']:
        print(f"\nUNKNOWN SPLIT:")
        print(f"  Files with anomalies: {len(split_counts['unknown'])}")
        print("  (Could not determine train/val/test from path)")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total anomalies: {len(anomaly_files)}")
    print(f"  Train: {len(split_counts['train'])}")
    print(f"  Val: {len(split_counts['val'])}")
    print(f"  Test: {len(split_counts['test'])}")
    print(f"  Unknown: {len(split_counts['unknown'])}")
    
    # Save detailed report
    output_path = dataset_path / 'anomaly_split_distribution.txt'
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("ANOMALY DISTRIBUTION BY SPLIT\n")
        f.write("="*80 + "\n\n")
        
        for split in ['train', 'val', 'test']:
            anomaly_count = len(split_counts[split])
            total_count = total_counts[split]
            
            f.write(f"\n{split.upper()}:\n")
            f.write(f"  Files with anomalies: {anomaly_count}\n")
            f.write(f"  Total files: {total_count}\n")
            
            if total_count > 0:
                percentage = (anomaly_count / total_count) * 100
                f.write(f"  Percentage: {percentage:.2f}%\n")
            
            if split_counts[split]:
                f.write(f"\n  Files:\n")
                for file_path in sorted(split_counts[split])[:50]:  # First 50 files
                    f.write(f"    - {Path(file_path).name}\n")
                
                if len(split_counts[split]) > 50:
                    f.write(f"    ... and {len(split_counts[split]) - 50} more\n")
        
        if split_counts['unknown']:
            f.write(f"\nUNKNOWN SPLIT:\n")
            f.write(f"  Files with anomalies: {len(split_counts['unknown'])}\n")
            for file_path in sorted(split_counts['unknown'])[:50]:
                f.write(f"    - {file_path}\n")
    
    print(f"\nDetailed report saved to: {output_path}")
    print("="*80)

if __name__ == "__main__":
    # Configuration
    dataset_path = "E:/data/Football/hash-marks"  # Change this to your dataset path
    report_path = "E:/data/Football/hash-marks/bbox_anomaly_report.txt"  # Path to the report file

    analyze_anomaly_distribution(dataset_path, report_path)