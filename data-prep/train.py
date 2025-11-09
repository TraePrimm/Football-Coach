#TODO:
#hash mark directory: E:/data/Football/hash-marks
#all-22 directory: E:/data/Football/all-22

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO

# --- Configuration ---
# Set the path to the root directory of your annotated hash mark dataset
# This directory must contain your 'data.yaml' file.
DATASET_PATH = 'E:/data/Football/all-22'

# Name of the YOLO model to use. 'yolo11n.pt' (nano) is a great starting
# point as it's lightweight and fast.
MODEL_NAME = 'yolo11m.pt'

# Training Hyperparameters
EPOCHS = 50          # Number of complete passes through the dataset
IMG_SIZE = 640       # All images will be resized to this resolution for training
BATCH_SIZE = 8       # Number of images to process at once; lower if you have VRAM issues
PROJECT_NAME = 'Comprehensive-data-set' # A parent folder for all your experiments
EXPERIMENT_NAME = 'Test-on-all-22-images-v1-M-P2P5'            # A specific name for this training run

# --- Training Code ---
def train_hash_mark_detector():
    """
    Loads a pre-trained YOLOv11 model and fine-tunes it on the custom
    dataset.
    """
    print("--- Starting YOLO Model Training ---")


    if torch.cuda.is_available():
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU is available. Training will run on: {gpu_name}")
    else:
        device = 'cpu'
        print("⚠️ WARNING: No GPU found. Training will run on the CPU, which will be much slower.")
    
    # 1. Define the path to the dataset configuration file
    data_yaml_path = os.path.join(DATASET_PATH, 'data.yaml')

    # Check if the data.yaml file exists before proceeding
    if not os.path.exists(data_yaml_path):
        print(f"Error: Dataset configuration file not found at: {data_yaml_path}")
        print("Please ensure the DATASET_PATH is correct and a 'data.yaml' file exists.")
        return
    
    # 2. Load the YOLO model
    try:
        #add this parameter so we have the custom model config loaded 'yolov11m-p2.yaml'
        model = YOLO('yolov11m-p2-p5.yaml').load(MODEL_NAME)
        print(f"✅ Successfully loaded custom model from 'yolov11m-p2-p5.yaml'")
        print(f"Successfully loaded pre-trained model: {MODEL_NAME}")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return

    # 3. Begin training the model on your dataset
    print(f"Training on dataset: {DATASET_PATH}")
    print(f"The process will run for {EPOCHS} epochs.")
    
    # The 'train' method handles the entire training loop.
    # It will save checkpoints and the final model automatically.
    results = model.train(
        data=data_yaml_path,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=PROJECT_NAME,
        name=EXPERIMENT_NAME,
        device=device,
    )

    # 4. Automatically determine the path to the best saved model weights
    # The 'results.save_dir' attribute holds the path to the experiment folder.
    best_weights_path = os.path.join(results.save_dir, 'weights', 'best.pt')

    print("\n--- Training Complete! ---")
    print(f"✅ Best model weights have been saved to: {best_weights_path}")


def inspect_problem_file():
    """
    Inspect the specific file that was causing issues
    """
    problem_file = "week05-Washington-Commanders-TennesseeTitans-WashingtonCommanders-053-163_jpg.rf.367fb9871a322fb63b9562ebd8de5f60.jpg"
    # Fix the label file name pattern
    label_file = problem_file.replace('.jpg.rf.', '_jpg.rf.').replace('.jpg', '.txt')
    
    label_path = os.path.join(DATASET_PATH, 'train', 'labels', label_file)
    image_path = os.path.join(DATASET_PATH, 'train', 'images', problem_file)
    
    print(f"\n--- Inspecting Problem File ---")
    print(f"Image: {image_path}")
    print(f"Label: {label_path}")
    
    # Also check for alternative label file patterns
    possible_label_paths = [
        label_path,
        os.path.join(DATASET_PATH, 'train', 'labels', problem_file.rsplit('.', 2)[0] + '.txt'),
        os.path.join(DATASET_PATH, 'train', 'labels', problem_file.replace('.jpg', '.txt'))
    ]
    
    label_found = False
    for possible_path in possible_label_paths:
        if os.path.exists(possible_path):
            print(f"\nFound label file at: {possible_path}")
            with open(possible_path, 'r') as f:
                content = f.read()
                print("Label content:")
                print(content)
            label_found = True
            break
            
    if not label_found:
        print("\nWarning: Could not find label file in any of these locations:")
        for path in possible_label_paths:
            print(f"- {path}")
    
    if os.path.exists(image_path):
        img = cv2.imread(image_path)
        if img is not None:
            print(f"Image exists and is readable. Shape: {img.shape}")
        else:
            print("Warning: Image file exists but cannot be read")
    else:
        print(f"Warning: Image file not found at {image_path}")
        
def validate_labels(dataset_path):
    """
    Validate all label files to ensure no class indices exceed 3
    """
    print("\n--- Validating Label Files ---")
    invalid_files = []
    labels_dir = os.path.join(dataset_path, 'train', 'labels')
    
    if not os.path.exists(labels_dir):
        print(f"Error: Labels directory not found at {labels_dir}")
        return
    
    for file in os.listdir(labels_dir):
        if not file.endswith('.txt'):
            continue
            
        filepath = os.path.join(labels_dir, file)
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines, 1):
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        class_idx = int(float(parts[0]))
                        if class_idx > 3:
                            invalid_files.append((filepath, i, class_idx))
                    except ValueError:
                        print(f"Warning: Invalid format in {file} line {i}")
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if invalid_files:
        print("\nFound invalid class indices in these files:")
        for filepath, line_num, class_idx in invalid_files:
            print(f"File: {filepath}")
            print(f"Line {line_num}: class_idx = {class_idx}")
        return False
    
    print("✅ All label files validated - no invalid class indices found")
    return True

if __name__ == "__main__":
    # Execute the training function when the script is run directly
    inspect_problem_file()  # Add this line before training
    if validate_labels(DATASET_PATH):
        train_hash_mark_detector()
    else:
        print("\n⚠️ Please fix invalid label files before training")