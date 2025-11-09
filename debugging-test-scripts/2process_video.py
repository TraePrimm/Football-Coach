import cv2
import numpy as np
import json
import os
import torch
from ultralytics import YOLO

# ============================= CONFIGURATION =============================
VIDEO_PATH = "E:\\data\\Football\\test_videos\\play-1\\play-1.mp4"
INITIAL_H_PATH = 'initial_homography.npy'
FIELD_MAP_PATH = 'field_map.json'

# YOLO Model Settings
YOLO_MODEL_PATH = 'Comprehensive-data-set\\Test-on-all-22-images-v1-M-P2P5\\weights\\best.pt'
YOLO_CONFIDENCE = 0.25

# Class IDs (must match your training data.yaml)
CLASS_PLAYER = 0
CLASS_REF = 1
CLASS_BALL = 2
CLASS_HASH = 3

# Colors (BGR)
COLOR_REF = (0, 255, 255)      # Yellow
COLOR_BALL = (255, 255, 255)   # White
COLOR_HOME = (255, 100, 0)     # Blue
COLOR_AWAY = (200, 200, 200)   # Light Gray
COLOR_DEFAULT = (0, 0, 255)    # Red (unassigned)

# ============================= GLOBALS =============================
MAP_WIDTH_PX = 1600
MAP_HEIGHT_PX = 3000
YARD_LINES_Y = {}
HASH_LINES_X = {}
PIXELS_PER_FOOT = 10
H_SMOOTH_ALPHA = 0.5          # 0 = keep old, 1 = take new 
POS_SMOOTH_ALPHA = 0.7        # smoothing for each player’s dot
player_pos_history = {}       # {tracker_id: np.array([x, y])}

yolo_model = None
yolo_device = 'cpu'

# Team color tracking
home_hsv_center = None
away_hsv_center = None
player_team_map = {}  # {tracker_id: 'home' or 'away'}

# ============================= HELPERS =============================
def resize_for_display(img, max_h=900, max_w=1600):
    """Resize image to fit screen while preserving aspect ratio."""
    h, w = img.shape[:2]
    scale = min(max_h / h, max_w / w, 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA), scale
    return img.copy(), 1.0

def ema_matrix(old, new, alpha):
    """Blend two homographies with EMA."""
    if old is None:
        return new
    return (1 - alpha) * old + alpha * new

def smooth_point(old_pt, new_pt, alpha):
    if old_pt is None:
        return new_pt
    return (1 - alpha) * old_pt + alpha * new_pt

# ============================= FIELD & MODEL LOADING =============================
def load_field_map():
    """Load field dimensions and coordinates from JSON."""
    global MAP_WIDTH_PX, MAP_HEIGHT_PX, YARD_LINES_Y, HASH_LINES_X, PIXELS_PER_FOOT
    try:
        with open(FIELD_MAP_PATH, 'r') as f:
            data = json.load(f)
        MAP_WIDTH_PX = data['map_dimensions_pixels']['width']
        MAP_HEIGHT_PX = data['map_dimensions_pixels']['height']
        PIXELS_PER_FOOT = data.get('pixels_per_foot', 10)
        YARD_LINES_Y = data['yard_line_y_coordinates']
        HASH_LINES_X = data['hash_x_coordinates']
        print("Field map loaded.")
        return True
    except Exception as e:
        print(f"Failed to load field map: {e}")
        return False


def load_yolo_model():
    """Load YOLOv8 model with GPU support if available."""
    global yolo_model, yolo_device
    yolo_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {yolo_device.upper()}")
    try:
        yolo_model = YOLO(YOLO_MODEL_PATH)
        yolo_model.to(yolo_device)
        print(f"YOLO model loaded: {YOLO_MODEL_PATH}")
        return True
    except Exception as e:
        print(f"Failed to load YOLO model: {e}")
        return False


# ============================= FIELD DRAWING =============================
def draw_field_on_map(canvas):
    """Draw yard lines and hash marks on the blank canvas."""
    line_color = (255, 255, 255)  # White
    hash_color = (128, 128, 128)  # Gray
    pixels_per_yard = int(PIXELS_PER_FOOT * 3)

    # Draw hash marks
    x_left = int(HASH_LINES_X['left_hash_x'])
    x_right = int(HASH_LINES_X['right_hash_x'])
    hash_len = int(PIXELS_PER_FOOT * 0.5)
    y_coords = sorted(int(y) for y in YARD_LINES_Y.values())

    for i in range(len(y_coords) - 1):
        y0, y1 = y_coords[i], y_coords[i + 1]
        for y_hash in range(y0 + pixels_per_yard, y1, pixels_per_yard):
            cv2.line(canvas, (x_left, y_hash - hash_len), (x_left, y_hash + hash_len), hash_color, 2)
            cv2.line(canvas, (x_right, y_hash - hash_len), (x_right, y_hash + hash_len), hash_color, 2)

    print("Field grid drawn.")


def get_full_destination_map():
    """Generate all hash mark points on the top-down map."""
    x_left = HASH_LINES_X.get('left_hash_x')
    x_right = HASH_LINES_X.get('right_hash_x')
    if x_left is None or x_right is None:
        return np.array([], dtype=np.float32).reshape(-1, 1, 2)

    pixels_per_yard = int(PIXELS_PER_FOOT * 3)
    y_coords = sorted(int(y) for y in YARD_LINES_Y.values())
    points = []

    for i in range(len(y_coords) - 1):
        y0, y1 = y_coords[i], y_coords[i + 1]
        for y in range(y0, y1, pixels_per_yard):
            points.extend([[x_left, y], [x_right, y]])
    points.append([x_left, y_coords[-1]])
    points.append([x_right, y_coords[-1]])

    print(f"Generated {len(points)} hash points.")
    return np.array(points, dtype=np.float32).reshape(-1, 1, 2)


# ============================= YOLO DETECTION =============================
def run_yolo_detector(frame):
    """Run YOLO tracking and return structured detections."""
    if yolo_model is None:
        return {"hash_points": np.array([]), "player_boxes": [], "ball_boxes": [], "ref_boxes": []}

    results = yolo_model.track(
        frame,
        classes=[CLASS_PLAYER, CLASS_BALL, CLASS_HASH, CLASS_REF],
        conf=YOLO_CONFIDENCE,
        device=yolo_device,
        persist=True,
        verbose=False
    )

    hash_points = []
    player_boxes = []
    ball_boxes = []
    ref_boxes = []

    if results and results[0].boxes is not None:
        boxes = results[0].boxes
        ids = boxes.id.cpu().numpy() if boxes.id is not None else None

        for i, cls_id in enumerate(boxes.cls.cpu().numpy()):
            cls_id = int(cls_id)
            xyxy = boxes.xyxy[i].cpu().numpy()

            if cls_id == CLASS_PLAYER and ids is not None:
                player_boxes.append((int(ids[i]), xyxy))
            elif cls_id == CLASS_BALL:
                ball_boxes.append(xyxy)
            elif cls_id == CLASS_REF:
                ref_boxes.append(xyxy)
            elif cls_id == CLASS_HASH:
                cx = (xyxy[0] + xyxy[2]) / 2
                cy = (xyxy[1] + xyxy[3]) / 2
                hash_points.append([cx, cy])

    return {
        "hash_points": np.array(hash_points, dtype=np.float32),
        "player_boxes": player_boxes,
        "ball_boxes": ball_boxes,
        "ref_boxes": ref_boxes
    }


# ============================= HOMOGRAPHY TRACKING =============================
def find_closest_points(predicted, detected, max_dist=25):
    """Match predicted hash points to detected ones."""
    if len(predicted) == 0 or len(detected) == 0:
        return np.array([]), []

    matched_src = []
    matched_idx = []
    for det in detected:
        dists = np.linalg.norm(predicted - det, axis=1)
        idx = np.argmin(dists)
        if dists[idx] < max_dist:
            matched_src.append(det)
            matched_idx.append(idx)
    return np.array(matched_src), matched_idx


def get_player_hotspot(box):
    """Return bottom-center of bounding box as transformable point."""
    x1, y1, x2, y2 = box
    return np.array([[(x1 + x2) / 2, y2]], dtype=np.float32).reshape(1, 1, 2)


# ============================= TEAM COLOR CLASSIFICATION =============================
def get_average_hsv(frame, box):
    """Get average HSV of player's torso (center 50%), ignoring green field."""
    x1, y1, x2, y2 = [int(p) for p in box]
    h, w = y2 - y1, x2 - x1
    if h < 10 or w < 10:
        return np.array([0, 0, 0])

    # Crop to torso (center 50% vertically & horizontally)
    crop = frame[
        int(y1 + 0.25 * h):int(y2 - 0.25 * h),
        int(x1 + 0.25 * w):int(x2 - 0.25 * w)
    ]
    if crop.size == 0:
        return np.array([0, 0, 0])

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    non_green = hsv[cv2.bitwise_not(green_mask) > 0]
    return np.mean(non_green, axis=0) if len(non_green) > 0 else np.mean(hsv.reshape(-1, 3), axis=0)


def find_team_colors(frame, player_boxes):
    """Cluster player colors to find home/away teams."""
    global home_hsv_center, away_hsv_center
    colors = [get_average_hsv(frame, box) for _, box in player_boxes]
    if len(colors) < 2:
        return

    data = np.float32(colors)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(data, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    home_hsv_center, away_hsv_center = centers
    print(f"Team colors set: Home={home_hsv_center}, Away={away_hsv_center}")


def classify_player_team(hsv):
    """Return 'home' or 'away' based on closest cluster."""
    if home_hsv_center is None:
        return "default"
    d1 = np.linalg.norm(hsv - home_hsv_center)
    d2 = np.linalg.norm(hsv - away_hsv_center)
    return "home" if d1 < d2 else "away"


# ============================= MAIN VIDEO LOOP =============================
def run_video_processing():
    global home_hsv_center, player_team_map
    player_team_map.clear()

    # Load resources
    if not load_field_map() or not load_yolo_model():
        return

    try:
        H = np.load(INITIAL_H_PATH)
        print("Initial homography loaded.")
    except FileNotFoundError:
        print("Initial H not found. Run calibration first.")
        return

    dest_map = get_full_destination_map()
    if dest_map.size == 0:
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Failed to open video.")
        return

    # Create static field background
    field_bg = np.zeros((MAP_HEIGHT_PX, MAP_WIDTH_PX, 3), dtype=np.uint8)
    draw_field_on_map(field_bg)

    frame_idx = 0
    print("\nStarting video processing... Press 'q' to quit, 's' to save frame.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Run detection
        det = run_yolo_detector(frame)
        hash_detected = det['hash_points']

        # Find team colors on first frame with players
        if home_hsv_center is None and det['player_boxes']:
            find_team_colors(frame, det['player_boxes'])

        # Predict hash locations
        try:
            H_inv = np.linalg.inv(H)
            predicted = cv2.perspectiveTransform(dest_map, H_inv).reshape(-1, 2)
        except np.linalg.LinAlgError:
            continue

        # Match and update H
        src_matched, idx_matched = find_closest_points(predicted, hash_detected)
        if len(src_matched) >= 4:
            dst_matched = dest_map.reshape(-1, 2)[idx_matched]
            H_new, _ = cv2.findHomography(src_matched, dst_matched, cv2.RANSAC, 5.0)
            if H_new is not None:
                H = ema_matrix(H, H_new, H_SMOOTH_ALPHA)

        # Start with clean field
        top_down = field_bg.copy()

        # Draw players (on top!)
        for pid, box in det['player_boxes']:
            # 1) transform with the *current* (smoothed) H
            raw_pt = cv2.perspectiveTransform(get_player_hotspot(box), H)[0][0]

            # 2) EMA on the point itself
            old = player_pos_history.get(pid)
            smooth_pt = smooth_point(old, raw_pt, POS_SMOOTH_ALPHA)
            player_pos_history[pid] = smooth_pt          # store for next frame
            x, y = int(smooth_pt[0]), int(smooth_pt[1])

            # ---- team colour (unchanged) --------------------------------
            if pid not in player_team_map and home_hsv_center is not None:
                hsv = get_average_hsv(frame, box)
                player_team_map[pid] = classify_player_team(hsv)
            team = player_team_map.get(pid, "default")
            color = COLOR_HOME if team == "home" else COLOR_AWAY if team == "away" else COLOR_DEFAULT

            cv2.circle(top_down, (x, y), 10, color, -1)

        # Draw ball and refs
        for box in det['ball_boxes']:
            pt = cv2.perspectiveTransform(get_player_hotspot(box), H)[0][0]
            cv2.circle(top_down, (int(pt[0]), int(pt[1])), 5, COLOR_BALL, -1)
        for box in det['ref_boxes']:
            pt = cv2.perspectiveTransform(get_player_hotspot(box), H)[0][0]
            cv2.circle(top_down, (int(pt[0]), int(pt[1])), 8, COLOR_REF, -1)

        # Resize for display
        disp_frame = cv2.resize(frame, (960, 540))
        disp_map, scale = resize_for_display(top_down)

        # Draw labeled yard lines on scaled view
        hud = disp_map.copy()
        sorted_lines = sorted(YARD_LINES_Y.items(), key=lambda x: int(x[1]))
        for name, y_src in sorted_lines:
            y_d = int(y_src * scale)
            cv2.line(hud, (0, y_d), (hud.shape[1], y_d), (0, 255, 0), 2)
            label = "50" if name == "50" else f"Opp {name.replace('Other_', '')}" if name.startswith("Other_") else f"Own {name}"
            cv2.putText(hud, label, (10, y_d - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # HUD
        # cv2.putText(hud, f"Scale: {scale:.2f}x | Frame: {frame_idx}", (10, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # Show
        cv2.imshow("Original", disp_frame)
        cv2.imshow("Top-Down View", hud)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            path = f"frame_{frame_idx:04d}_topdown.png"
            cv2.imwrite(path, top_down)
            print(f"Saved: {path}")

    cap.release()
    cv2.destroyAllWindows()
    print("Video processing complete.")


# ============================= ENTRY POINT =============================
if __name__ == "__main__":
    if 'your_all_22_film.mp4' in VIDEO_PATH or 'path/to/your/best.pt' in YOLO_MODEL_PATH:
        print("\nPlease update VIDEO_PATH and YOLO_MODEL_PATH before running!\n")
    else:
        run_video_processing()