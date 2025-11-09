"""
YOLOv11 Football Detection Analysis
Tests trained model on football footage with performance analysis and ball detection tracking
"""

import os
import time
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration
MODEL_PATH = '../Comprehensive-data-set/Test-on-all-22-images-v1-M-P22/weights/best.pt'
FRAMES_DIR = 'E:/data/Football/test_videos/play-1/frames'
SINGLE_IMAGE_PATH = 'E:/data/Football/test_videos/play-1/frames/frame_0004.jpg'

class FootballDetector:
    def __init__(self, model_path):
        """Initialize the football detector with trained YOLO model"""
        self.model = self.load_model(model_path)
        self.class_names = self.model.names if self.model else {}
        self.ball_class_id = self.find_ball_class_id()
        
    def load_model(self, model_path):
        """Load YOLO model with error handling"""
        try:
            model = YOLO(model_path)
            print(f"✅ Successfully loaded model from: {model_path}")
            return model
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None
    
    def find_ball_class_id(self):
        """Find the ball class ID in the model's class names"""
        ball_class_id = None
        for class_id, class_name in self.class_names.items():
            if class_name.lower() in ['ball', '2']:  # Look for 'ball' or class '2'
                ball_class_id = class_id
                break
        
        print(f"📊 Class mappings: {self.class_names}")
        print(f"⚽ Ball class ID: {ball_class_id}")
        return ball_class_id
    
    def filter_detections(self, results, image_shape, min_confidence=0.6, 
                         min_player_area=100, min_ball_area=20):
        """
        Filter out unwanted detections while protecting the ball
        
        Args:
            results: YOLO results object
            image_shape: (height, width) of the image
            min_confidence: minimum confidence threshold
            min_player_area: minimum bounding box area for players/refs
            min_ball_area: minimum bounding box area for ball
            
        Returns:
            filtered_boxes: list of filtered bounding boxes
        """
        if not results or len(results[0].boxes) == 0:
            return []
            
        filtered_boxes = []
        height, width = image_shape[:2]
        
        # Common watermark regions (top/bottom of frame)
        excluded_regions = [
            [0, 0, width//4, height//8],           # Top-left
            [width*3//4, 0, width, height//8],     # Top-right
            [0, height*7//8, width, height],       # Bottom
        ]
        
        for box in results[0].boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            
            # Skip low confidence detections
            if conf < min_confidence:
                continue
                
            # Get bounding box coordinates and area
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            area = (x2 - x1) * (y2 - y1)
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            
            is_ball = (cls == self.ball_class_id)
            
            # Apply object-specific filtering
            if is_ball:
                if area >= min_ball_area:
                    filtered_boxes.append(box)
            else:
                # For non-ball objects, check area and watermark regions
                area_ok = area >= min_player_area
                region_ok = not any(self._point_in_region(center_x, center_y, region) 
                                  for region in excluded_regions)
                
                if area_ok and region_ok:
                    filtered_boxes.append(box)
        
        return filtered_boxes
    
    def _point_in_region(self, x, y, region):
        """Check if point (x,y) is within region [x1,y1,x2,y2]"""
        x1, y1, x2, y2 = region
        return x1 <= x <= x2 and y1 <= y <= y2
    
    def analyze_performance(self, image, num_iterations=10):
        """Analyze model performance and inference speed"""
        if not self.model:
            return None
            
        print(f"🚀 Performance Analysis ({num_iterations} iterations)...")
        
        # Warmup
        _ = self.model.predict(source=image, conf=0.25, verbose=False)
        
        # Timing loop
        times = []
        for i in range(num_iterations):
            start_time = time.time()
            results = self.model.predict(source=image, conf=0.25, verbose=False)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        avg_time = np.mean(times)
        fps = 1000 / avg_time if avg_time > 0 else 0
        
        print(f"📈 Average inference time: {avg_time:.2f} ms")
        print(f"🎯 Estimated FPS: {fps:.2f}")
        print(f"📊 Time range: {np.min(times):.2f} - {np.max(times):.2f} ms")
        
        return avg_time, fps
    
    def scan_folder_for_ball(self, frames_dir, max_frames=None):
        """Scan all frames in folder for ball detections"""
        frames_dir = Path(frames_dir)
        if not frames_dir.exists():
            print(f"❌ Frames directory not found: {frames_dir}")
            return
        
        frame_files = sorted(list(frames_dir.glob("*.jpg")) + 
                           list(frames_dir.glob("*.png")))
        
        if max_frames:
            frame_files = frame_files[:max_frames]
        
        print(f"🔍 Scanning {len(frame_files)} frames for ball detections...")
        
        ball_detections = []
        total_detections = 0
        
        for frame_path in frame_files:
            # Load and process frame
            image = cv2.imread(str(frame_path))
            if image is None:
                continue
                
            # Run inference
            results = self.model.predict(source=image, conf=0.25, verbose=False)
            
            # Count ball detections
            frame_balls = 0
            for box in results[0].boxes:
                if int(box.cls) == self.ball_class_id and float(box.conf) > 0.5:
                    frame_balls += 1
                    total_detections += 1
            
            if frame_balls > 0:
                ball_detections.append({
                    'frame': frame_path.name,
                    'ball_count': frame_balls,
                    'confidence': max([float(box.conf) for box in results[0].boxes 
                                    if int(box.cls) == self.ball_class_id], default=0)
                })
        
        # Print analysis
        print(f"\n📊 Ball Detection Analysis:")
        print(f"   Total frames processed: {len(frame_files)}")
        print(f"   Frames with ball detected: {len(ball_detections)}")
        print(f"   Total ball detections: {total_detections}")
        print(f"   Ball detection rate: {len(ball_detections)/len(frame_files)*100:.1f}%")
        
        if ball_detections:
            print(f"\n🎯 Frames with ball detections:")
            for detection in ball_detections[:10]:  # Show first 10
                print(f"   {detection['frame']}: {detection['ball_count']} balls "
                      f"(max conf: {detection['confidence']:.2f})")
        
        return ball_detections
    
    def visualize_detections(self, image, filtered_boxes, results):
        """Visualize filtered detections with ball highlighting"""
        # Original detections
        original_image = results[0].plot()
        
        # Filtered detections
        filtered_image = image.copy()
        
        for box in filtered_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cls = int(box.cls)
            conf = float(box.conf)
            
            # Highlight ball in red, others in green
            is_ball = (cls == self.ball_class_id)
            color = (0, 0, 255) if is_ball else (0, 255, 0)
            thickness = 3 if is_ball else 2
            
            # Draw bounding box
            cv2.rectangle(filtered_image, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label = f"{self.class_names[cls]}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(filtered_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(filtered_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Display comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        ax1.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        ax1.set_title('Original Detections')
        ax1.axis('off')
        
        ax2.imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
        ax2.set_title('Filtered Detections (Ball Protected)')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return filtered_image

def main():
    """Main execution function"""
    detector = FootballDetector(MODEL_PATH)
    
    if not detector.model:
        return
    
    print("\n" + "="*50)
    print("1. SINGLE IMAGE ANALYSIS")
    print("="*50)
    
    # Single image analysis
    if os.path.exists(SINGLE_IMAGE_PATH):
        image = cv2.imread(SINGLE_IMAGE_PATH)
        if image is not None:
            # Performance analysis
            detector.analyze_performance(image)
            
            # Run detection
            results = detector.model.predict(source=image, conf=0.25, verbose=False)
            print(f"📦 Original detections: {len(results[0].boxes)}")
            
            # Filter detections
            filtered_boxes = detector.filter_detections(
                results, image.shape, 
                min_confidence=0.5,
                min_player_area=150,
                min_ball_area=15
            )
            print(f"✅ Filtered detections: {len(filtered_boxes)}")
            
            # Count objects
            counts = {}
            for box in filtered_boxes:
                cls_name = detector.class_names[int(box.cls)]
                counts[cls_name] = counts.get(cls_name, 0) + 1
            
            print(f"📊 Final counts: {counts}")
            
            # Visualize
            detector.visualize_detections(image, filtered_boxes, results)
            
            # Save result
            cv2.imwrite('filtered_detection_result.jpg', image)
            print("💾 Result saved as 'filtered_detection_result.jpg'")
    
    print("\n" + "="*50)
    print("2. FOLDER SCAN FOR BALL DETECTIONS")
    print("="*50)
    
    # Scan folder for ball detections
    detector.scan_folder_for_ball(FRAMES_DIR)  # Limit for testing

if __name__ == "__main__":
    main()