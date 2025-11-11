"""
Flask backend for football tracking system.
Integrates YOLO + homography + top-down field mapping with React frontend.
"""

from flask import Flask, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import json
import base64
import logging
import io
from pathlib import Path
from threading import Thread, Lock
from collections import defaultdict
import torch
from ultralytics import YOLO
from moviepy.video.io import VideoFileClip

# ============================= INITIALIZATION =============================
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================= CONFIGURATION =============================
VIDEOS_DIR = r"E:\data\Football\test_videos"
FIELD_MAP_PATH = "field_map.json"
YOLO_MODEL_PATH = r"Comprehensive-data-set\Test-on-all-22-images-v1-M-P2P5\weights\best.pt"
YOLO_CONFIDENCE = 0.25
INITIAL_H_PATH = 'initial_homography.npy' 

# Class IDs
CLASS_PLAYER = 0
CLASS_REF = 1
CLASS_BALL = 2
CLASS_HASH = 3

# Colors (BGR)
COLOR_REF = (0, 255, 255)
COLOR_BALL = (255, 255, 255)
COLOR_HOME = (255, 100, 0)   # Blue-Orange
COLOR_AWAY = (200, 200, 200) # Light Gray
COLOR_FIELD = (34, 139, 34)  # Dark green

# Smoothing
H_SMOOTH_ALPHA = 0.5
POS_SMOOTH_ALPHA = 0.7
PIXELS_PER_FOOT = 10

# ============================= GLOBALS =============================
MAP_WIDTH_PX = 1600
MAP_HEIGHT_PX = 3000
YARD_LINES_Y = {}
HASH_LINES_X = {}

yolo_model = None
yolo_device = "cpu"

home_hsv_center = None
away_hsv_center = None
player_team_votes   = {}   # {play_id: {track_id: {"home":0, "away":0, "locked":False}}}
player_hsv_ema      = {}   # {play_id: {track_id: np.array([H,S,V])}}
VOTE_THRESHOLD      = 0.70 # 70 % of frames must agree
EMA_ALPHA           = 0.25 # how fast a player's own color adapts
MAX_JUMP_PX         = 80   # ignore frames with huge position jumps

processed_plays = {}        # {play_id: {frames, total_frames, fps, duration}}
current_play_cache = {}     # {play_id: {progress, message}}
player_stats_cache = {}     # {play_id: {track_id: stats}}
frame_players_cache = {}    # {play_id: {frame_num: [player_dicts]}}
lock = Lock()


# ============================= HELPERS =============================
def load_field_map() -> bool:
    global MAP_WIDTH_PX, MAP_HEIGHT_PX, YARD_LINES_Y, HASH_LINES_X, PIXELS_PER_FOOT
    try:
        with open(FIELD_MAP_PATH, "r") as f:
            data = json.load(f)
        MAP_WIDTH_PX = data["map_dimensions_pixels"]["width"]
        MAP_HEIGHT_PX = data["map_dimensions_pixels"]["height"]
        PIXELS_PER_FOOT = data.get("pixels_per_foot", 10)
        YARD_LINES_Y = {str(k): int(v) for k, v in data["yard_line_y_coordinates"].items()}
        HASH_LINES_X = data["hash_x_coordinates"]
        return True
    except Exception as e:
        logger.error(f"Failed to load field map: {e}")
        return False


def load_yolo_model() -> bool:
    global yolo_model, yolo_device
    yolo_device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        yolo_model = YOLO(YOLO_MODEL_PATH)
        yolo_model.to(yolo_device)
        logger.info(f"YOLO model loaded on {yolo_device.upper()}")
        return True
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        return False


def ema_matrix(old: np.ndarray, new: np.ndarray, alpha: float) -> np.ndarray:
    return new if old is None else (1 - alpha) * old + alpha * new


def get_player_hotspot(box) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array([[(x1 + x2) / 2, y2]], dtype=np.float32).reshape(1, 1, 2)


def get_average_hsv(frame: np.ndarray, box) -> np.ndarray:
    x1, y1, x2, y2 = [int(p) for p in box]
    h, w = y2 - y1, x2 - x1
    if h < 10 or w < 10:
        return np.array([0, 0, 0])

    crop = frame[
        int(y1 + 0.25 * h): int(y2 - 0.25 * h),
        int(x1 + 0.25 * w): int(x2 - 0.25 * w)
    ]
    if crop.size == 0:
        return np.array([0, 0, 0])

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    non_green = hsv[cv2.bitwise_not(green_mask) > 0]
    return np.mean(non_green, axis=0) if len(non_green) > 0 else np.mean(hsv.reshape(-1, 3), axis=0)


def classify_player_team(hsv: np.ndarray) -> str:
    if home_hsv_center is None or away_hsv_center is None:
        return "unknown"
    d_home = np.linalg.norm(hsv - home_hsv_center)
    d_away = np.linalg.norm(hsv - away_hsv_center)
    return "home" if d_home < d_away else "away"


def run_yolo_detector(frame: np.ndarray) -> dict:
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

    hash_points, player_boxes, ball_boxes, ref_boxes = [], [], [], []

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

def get_brightness(frame: np.ndarray, box) -> float:
    """Calculate average V (brightness) from HSV, ignoring green field and small bright spots."""
    x1, y1, x2, y2 = [int(p) for p in box]
    h, w = y2 - y1, x2 - x1
    if h < 10 or w < 10:
        return 0.0

    crop = frame[
        int(y1 + 0.25 * h): int(y2 - 0.25 * h),
        int(x1 + 0.25 * w): int(x2 - 0.25 * w)
    ]
    if crop.size == 0:
        return 0.0

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    non_green = hsv[cv2.bitwise_not(green_mask) > 0]
    
    if len(non_green) > 0:
        # Get the V (brightness) values
        v_values = non_green[:, 2]

        v_median = np.median(v_values)
        v_std = np.std(v_values)
        
        filtered_v = v_values[np.abs(v_values - v_median) <= 1.5 * v_std]
        
        if len(filtered_v) > 0:
            return float(np.mean(filtered_v))
        return float(np.mean(v_values))
    return 0.0

# --------------------------------------------------------------
# 1. NEW HELPERS (replace the old get_average_hsv / classify_player_team)
# --------------------------------------------------------------
def get_jersey_hsv(frame: np.ndarray, box) -> np.ndarray:
    """
    Return mean HSV of the *upper* 45 % of the player box (jersey area).
    Green field and the lower ~30 % (pants) are masked out.
    """
    x1, y1, x2, y2 = map(int, box)
    h, w = y2 - y1, x2 - x1
    if h < 20 or w < 20:
        return np.array([0, 0, 0])

    # ---- crop to upper jersey (top 45% of the box) ----
    jersey_top    = int(y1 + 0.05 * h)          # avoid helmet
    jersey_bottom = int(y1 + 0.50 * h)          # stop before pants
    crop = frame[jersey_top:jersey_bottom, x1:x2]

    if crop.size == 0:
        return np.array([0, 0, 0])

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # ---- mask green field (very tolerant) ----
    green_lo = np.array([30, 30, 30])
    green_hi = np.array([90, 255, 255])
    green_mask = cv2.inRange(hsv, green_lo, green_hi)

    # ---- mask pants (bottom 30% of *original* box) ----
    pants_top = int(y2 - 0.30 * h)
    pants_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    pants_mask[pants_top - y1: , :] = 255          # relative to crop start

    # combine masks
    field_or_pants = cv2.bitwise_or(green_mask, pants_mask)
    jersey_pixels = hsv[cv2.bitwise_not(field_or_pants) > 0]

    if len(jersey_pixels) == 0:
        return np.array([0, 0, 0])
    return np.mean(jersey_pixels, axis=0)


def update_team_centers(home_center, away_center, hsv_samples_home, hsv_samples_away, alpha=0.3):
    """EMA update of the two team HSV centers."""
    if hsv_samples_home:
        new_home = np.mean(hsv_samples_home, axis=0)
        home_center = new_home if home_center is None else (1-alpha)*home_center + alpha*new_home
    if hsv_samples_away:
        new_away = np.mean(hsv_samples_away, axis=0)
        away_center = new_away if away_center is None else (1-alpha)*away_center + alpha*new_away
    return home_center, away_center


def init_play_state(play_id):
    """Called once at the start of process_video."""
    player_team_votes[play_id] = {}
    player_hsv_ema[play_id]    = {}

def cleanup_play_state(play_id):
    player_team_votes.pop(play_id, None)
    player_hsv_ema.pop(play_id, None)

def update_player_ema(play_id, track_id, hsv):
    """Running EMA of the *jersey* HSV for this player."""
    ema = player_hsv_ema[play_id].get(track_id)
    if ema is None:
        player_hsv_ema[play_id][track_id] = hsv.copy()
    else:
        player_hsv_ema[play_id][track_id] = (1-EMA_ALPHA)*ema + EMA_ALPHA*hsv

def vote_for_team(play_id, track_id, team):
    """Count a vote and lock when we have enough evidence."""
    votes = player_team_votes[play_id].setdefault(track_id, {"home":0, "away":0, "locked":False})
    if votes["locked"]:
        return
    votes[team] += 1
    total = votes["home"] + votes["away"]
    if total >= 5:                              # need at least 5 votes
        ratio = votes["home"] / total if team == "home" else votes["away"] / total
        if max(votes["home"], votes["away"]) / total >= VOTE_THRESHOLD:
            votes["locked"] = True
            votes["final"] = "home" if votes["home"] > votes["away"] else "away"

def find_closest_points(predicted: np.ndarray, detected: np.ndarray, max_dist: float = 150) -> tuple:
    if len(predicted) == 0 or len(detected) == 0:
        return np.array([]), []

    matched_src, matched_idx = [], []
    for det in detected:
        dists = np.linalg.norm(predicted - det, axis=1)
        idx = np.argmin(dists)
        if dists[idx] < max_dist:
            matched_src.append(det)
            matched_idx.append(idx)
    return np.array(matched_src), matched_idx


def get_full_destination_map() -> np.ndarray:
    x_left = HASH_LINES_X.get("left_hash_x")
    x_right = HASH_LINES_X.get("right_hash_x")
    if x_left is None or x_right is None:
        return np.array([]).reshape(-1, 1, 2)

    pixels_per_yard = int(PIXELS_PER_FOOT * 3)
    y_coords = sorted(int(y) for y in YARD_LINES_Y.values())
    points = []

    for i in range(len(y_coords) - 1):
        y0, y1 = y_coords[i], y_coords[i + 1]
        for y in range(y0, y1, pixels_per_yard):
            points.extend([[x_left, y], [x_right, y]])
    points.extend([[x_left, y_coords[-1]], [x_right, y_coords[-1]]])

    return np.array(points, dtype=np.float32).reshape(-1, 1, 2)


def classify_with_centers(hsv, home_center, away_center):
    """Distance-based classification against the two global HSV centres."""
    if home_center is None or away_center is None:
        return "unknown"
    d_home = np.linalg.norm(hsv - home_center)
    d_away = np.linalg.norm(hsv - away_center)
    return "home" if d_home < d_away else "away"


# ============================= VIDEO PROCESSING =============================
def process_video(play_id: str, video_path: str):
    global home_hsv_center, away_hsv_center
    
    try:
        logger.info(f"Starting processing: {play_id}")

        try:
            H = np.load(INITIAL_H_PATH)
            logger.info(f"[{play_id}] Initial homography loaded.")
        except FileNotFoundError:
            logger.error(f"[{play_id}] FATAL: 'initial_homography.npy' not found.")
            current_play_cache[play_id] = {"progress": 100.0, "message": "Error: initial_homography.npy not found"}
            return

        dst_points = get_full_destination_map()
        if dst_points.size == 0:
            logger.error(f"[{play_id}] FATAL: Failed to get destination map points.")
            return

        # ============================= FIRST PASS: COLLECT HSV + BRIGHTNESS SAMPLES =============================
        logger.info(f"[{play_id}] First pass – collecting jersey HSV …")
        cap_temp = cv2.VideoCapture(video_path)
        first_pass_hsv = defaultdict(list)          # track_id → list of HSV vectors
        frame_count_temp = 0
        first_pass_frames = min(60, int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT)))  # a few more frames

        while frame_count_temp < first_pass_frames:
            ret, frame = cap_temp.read()
            if not ret:
                break
            detections = run_yolo_detector(frame)
            for track_id, xyxy in detections["player_boxes"]:
                hsv = get_jersey_hsv(frame, xyxy)
                if np.any(hsv):
                    first_pass_hsv[track_id].append(hsv)
            frame_count_temp += 1
        cap_temp.release()

        # ============================= CLUSTER PLAYERS INTO TWO TEAMS (BRIGHTNESS-BASED) =============================
        all_samples = []
        track_ids   = []
        for tid, samples in first_pass_hsv.items():
            if len(samples) >= 3:                     # need a few observations
                all_samples.extend(samples)
                track_ids.extend([tid] * len(samples))

        if len(all_samples) >= 10:                     # safety
            all_samples = np.array(all_samples, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, centers = cv2.kmeans(all_samples, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

            # decide which cluster is HOME (darker jersey)
            mean_v = [np.mean(c[2]) for c in centers]
            home_idx = np.argmin(mean_v)
            away_idx = 1 - home_idx

            home_center = centers[home_idx]
            away_center = centers[away_idx]

            # build per-player team map from the clustering result
            player_team_map = {}
            label_array = labels.flatten()
            ptr = 0
            for tid, samples in first_pass_hsv.items():
                if len(samples) < 3:
                    continue
                cluster_votes = label_array[ptr:ptr+len(samples)]
                vote_home = np.sum(cluster_votes == home_idx)
                vote_away = np.sum(cluster_votes == away_idx)
                player_team_map[tid] = "home" if vote_home > vote_away else "away"
                ptr += len(samples)
        else:
            logger.warning(f"[{play_id}] Not enough jersey samples for K-means – fallback to brightness")
            home_center = away_center = None
            player_team_map = {}

            

        # ============================= MAIN VIDEO PROCESSING LOOP =============================
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            logger.error(f"[{play_id}] Video file has 0 frames or is unreadable.")
            return

        frame_idx = 0
        frame_cache = {}
        player_tracks = defaultdict(lambda: {"positions": [], "times": [], "team": None, "hsv": None})
        prev_homography = H
        frame_players = {}

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = run_yolo_detector(frame)
            hash_points = detections["hash_points"]
            player_boxes = detections["player_boxes"]
            ball_boxes = detections["ball_boxes"]

            H = prev_homography
            try:
                H_inv = np.linalg.inv(H)
                predicted_video_pts = cv2.perspectiveTransform(dst_points, H_inv).reshape(-1, 2)
                src_m, idx_m = find_closest_points(predicted_video_pts, hash_points, max_dist=50)
                
                if len(src_m) >= 4:
                    dst_m = dst_points[idx_m].squeeze()
                    H_new, _ = cv2.findHomography(src_m, dst_m, cv2.RANSAC, 5.0)
                    if H_new is not None:
                        H = ema_matrix(prev_homography, H_new, H_SMOOTH_ALPHA)
                        prev_homography = H
            except np.linalg.LinAlgError:
                logger.warning(f"[{play_id}] Singular matrix in H_inv, skipping H-update for frame {frame_idx}")

            topdown = np.zeros((MAP_HEIGHT_PX, MAP_WIDTH_PX, 3), dtype=np.uint8)
            topdown[:] = COLOR_FIELD

            for label, y in YARD_LINES_Y.items():
                cv2.line(topdown, (0, int(y)), (MAP_WIDTH_PX, int(y)), (255, 255, 255), 2)
                cv2.putText(topdown, label, (10, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            for x in (HASH_LINES_X["left_hash_x"], HASH_LINES_X["right_hash_x"]):
                cv2.line(topdown, (int(x), 0), (int(x), MAP_HEIGHT_PX), (255, 255, 255), 3)

            cur_time = frame_idx / fps
            current_frame_players = []

            for track_id, xyxy in player_boxes:
                hotspot = get_player_hotspot(xyxy)
                if H is not None:
                    proj = cv2.perspectiveTransform(hotspot, H)[0][0]
                    x, y = int(proj[0]), int(proj[1])
                    if 0 <= x < MAP_WIDTH_PX and 0 <= y < MAP_HEIGHT_PX:
                        track = player_tracks[track_id]

                        if track["team"] is None:
                            # 1. try the first-pass map
                            if track_id in player_team_map:
                                track["team"] = player_team_map[track_id]
                            else:
                                # 2. fallback – classify on-the-fly and also update the EMA centers
                                hsv = get_jersey_hsv(frame, xyxy)
                                if np.any(hsv) and home_center is not None and away_center is not None:
                                    track["team"] = classify_with_centers(hsv, home_center, away_center)

                                    # keep EMA centers alive for the whole play
                                    if track["team"] == "home":
                                        home_center, _ = update_team_centers(home_center, away_center, [hsv], [], alpha=0.2)
                                    else:
                                        _, away_center = update_team_centers(home_center, away_center, [], [hsv], alpha=0.2)
                                else:
                                    track["team"] = "unknown"

                        # colour choice stays the same
                        color = COLOR_HOME if track["team"] == "home" else COLOR_AWAY
                        cv2.circle(topdown, (x, y), 15, color, -1)
                        cv2.putText(topdown, str(track_id), (x - 12, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                        track["positions"].append((x, y))
                        track["times"].append(cur_time)

                        current_frame_players.append({
                            "id": str(track_id),
                            "number": str(track_id),
                            "team": track["team"] or "unknown",
                            "role": "Unknown",
                            "x": x,
                            "y": y,
                        })

            for xyxy in ball_boxes:
                hotspot = get_player_hotspot(xyxy)
                if H is not None:
                    proj = cv2.perspectiveTransform(hotspot, H)[0][0]
                    x, y = int(proj[0]), int(proj[1])
                    if 0 <= x < MAP_WIDTH_PX and 0 <= y < MAP_HEIGHT_PX:
                        cv2.circle(topdown, (x, y), 10, COLOR_BALL, -1)

            # === RESIZE ORIGINAL FRAME (max 960px width, preserve ratio) ===
            h, w = frame.shape[:2]
            max_w = 960
            scale = min(max_w / w, 1.0)
            if scale < 1.0:
                new_w = int(w * scale)
                new_h = int(h * scale)
                frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                frame_resized = frame

            _, orig_buf = cv2.imencode('.jpg', frame_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

            # === RESIZE TOPDOWN (max 800px width, preserve ratio) ===
            max_w_top = 800
            scale_top = min(max_w_top / MAP_WIDTH_PX, 1.0)
            if scale_top < 1.0:
                new_w = int(MAP_WIDTH_PX * scale_top)
                new_h = int(MAP_HEIGHT_PX * scale_top)
                topdown_resized = cv2.resize(topdown, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            else:
                topdown_resized = topdown

            _, top_buf = cv2.imencode('.png', topdown_resized, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
            orig_b64 = base64.b64encode(orig_buf).decode()
            top_b64 = base64.b64encode(top_buf).decode()

            frame_cache[frame_idx] = {
                "original_image": orig_b64,
                "topdown_image": top_b64,
            }

            frame_players[frame_idx] = current_frame_players

            progress_percent = round(frame_idx / total_frames * 100, 1)
            current_play_cache[play_id] = {"progress": progress_percent, "message": "Processing video..."}
            frame_idx += 1

        cap.release()

        logger.info(f"Finished frame processing for {play_id}. Calculating stats...")
        current_play_cache[play_id] = {"progress": 100.0, "message": "Calculating stats..."}

        current_play_stats = {}
        for pid, tr in player_tracks.items():
            pos = np.array(tr["positions"])
            t = np.array(tr["times"])
            if len(pos) > 1:
                diffs = np.diff(pos, axis=0)
                dist_px = np.sqrt(np.sum(diffs ** 2, axis=1)).sum()
                dist_ft = dist_px / PIXELS_PER_FOOT
                duration = t[-1] - t[0]
                speed_mph = (dist_ft / duration) * 0.681818 if duration > 0 else 0.0

                current_play_stats[pid] = {
                    "team": tr["team"],
                    "distance": round(dist_ft, 1),
                    "speed": round(speed_mph, 2),
                    "play_count": len(pos),
                    "role": "Unknown",
                    "name": f"Player {pid}",
                    "number": str(pid)
                }

        logger.info(f"[{play_id}] Stats: {current_play_stats}")

        with lock:
            player_stats_cache[play_id] = current_play_stats
            frame_players_cache[play_id] = frame_players
            processed_plays[play_id] = {
                "frames": frame_cache,
                "total_frames": frame_idx,
                "fps": fps,
                "duration": round(frame_idx / fps, 1)
            }

        current_play_cache.pop(play_id, None)
        logger.info(f"[{play_id}] Processing complete.")

    except Exception as e:
        logger.error(f"[{play_id}] FATAL ERROR: {e}", exc_info=True)
        current_play_cache[play_id] = {"progress": 100.0, "message": f"Error: {e}"}


# ============================= ROUTES =============================
@app.route("/api/plays", methods=["GET"])
def get_plays():
    plays = []
    base = Path(VIDEOS_DIR)
    for folder in sorted(base.iterdir()):
        if not folder.is_dir() or not folder.name.startswith("play-"):
            continue
        mp4_files = list(folder.glob("*.mp4"))
        if not mp4_files:
            continue
        video_path = mp4_files[0]
        try:
            clip = VideoFileClip(str(video_path))
            duration = clip.duration
            clip.close()
        except Exception:
            # Fallback: use OpenCV to read duration if MoviePy fails
            try:
                cap = cv2.VideoCapture(str(video_path))
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
                    duration = float(frames / fps) if fps > 0 else 0.0
                else:
                    duration = 0.0
                cap.release()
            except Exception:
                duration = 0.0
        display_name = video_path.stem.replace('-', ' ')
        plays.append({
            "id": folder.name,
            "name": display_name,
            "path": str(video_path),
            "duration": round(duration, 1) if duration else 0
        })
    return jsonify(plays)


@app.route("/api/play/<play_id>/thumbnail", methods=["GET"])
def get_thumbnail(play_id):
    """Return resized first frame as JPEG thumbnail (160x90)."""
    folder_path = Path(VIDEOS_DIR) / play_id
    if not folder_path.is_dir():
        return jsonify({"error": "Play not found"}), 404

    mp4_files = list(folder_path.glob("*.mp4"))
    if not mp4_files:
        return jsonify({"error": "No video"}), 404

    cap = cv2.VideoCapture(str(mp4_files[0]))
    if not cap.isOpened():
        return jsonify({"error": "Cannot open video"}), 500

    ret, frame = cap.read()
    cap.release()
    if not ret:
        return jsonify({"error": "Cannot read frame"}), 500

    # === RESIZE TO THUMBNAIL SIZE ===
    thumb_width = 160
    thumb_height = 90
    frame_resized = cv2.resize(frame, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA)

    # === ENCODE AS JPEG WITH GOOD QUALITY & SMALL SIZE ===
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]  # 75 is good balance
    _, buffer = cv2.imencode('.jpg', frame_resized, encode_param)

    return send_file(
        io.BytesIO(buffer),
        mimetype='image/jpeg',
        as_attachment=False
    )


@app.route("/api/play/<play_id>/frame/<int:frame_num>", methods=["GET"])
def get_frame(play_id, frame_num):
    """Return original + top-down + players."""
    with lock:
        play = processed_plays.get(play_id)
    if not play:
        return jsonify({"error": "Play not processed"}), 404

    frame_data = play["frames"].get(frame_num)
    if not frame_data:
        return jsonify({"error": "Frame not found"}), 404

    players = frame_players_cache.get(play_id, {}).get(frame_num, [])

    return jsonify({
        "original_image": frame_data["original_image"],
        "topdown_image": frame_data["topdown_image"],
        "total_frames": play["total_frames"],
        "players": players,
        "map_width_px": MAP_WIDTH_PX,
        "map_height_px": MAP_HEIGHT_PX,
    })


@app.route("/api/play/<play_id>/process", methods=["POST"])
def process_play(play_id):
    folder_path = Path(VIDEOS_DIR) / play_id
    if not folder_path.is_dir():
        return jsonify({"error": "Play folder not found"}), 404

    mp4_files = list(folder_path.glob("*.mp4"))
    if not mp4_files:
        return jsonify({"error": "No MP4 in folder"}), 404

    video_path = mp4_files[0]
    thread = Thread(target=process_video, args=(play_id, str(video_path)))
    thread.start()

    return jsonify({"status": "processing", "play_id": play_id})


@app.route("/api/play/<play_id>/status", methods=["GET"])
def get_play_status(play_id):
    if play_id in processed_plays:
        d = processed_plays[play_id]
        return jsonify({
            "status": "ready",
            "total_frames": d["total_frames"],
            "duration": d["duration"]
        })
    elif play_id in current_play_cache:
        cache_data = current_play_cache[play_id]
        return jsonify({
            "status": "processing",
            "progress": cache_data.get("progress", 0),
            "message": cache_data.get("message", "Initializing...")
        })
    else:
        return jsonify({"status": "not_started"})


@app.route("/api/play/<play_id>/player/<int:player_id>", methods=["GET"])
def get_player_stats(play_id, player_id):
    with lock:
        play_stats = player_stats_cache.get(play_id, {})
        stats = play_stats.get(player_id, {})
    if not stats:
        return jsonify({"error": "Stats not found"}), 404
    return jsonify(stats)


@app.route("/api/play/<play_id>/all_stats", methods=["GET"])
def get_all_stats(play_id):
    with lock:
        play_stats = player_stats_cache.get(play_id, {})
    resp = []
    for pid, s in play_stats.items():
        resp.append({
            "id": int(pid),
            "team": s.get("team"),
            "distance": s.get("distance", 0),
            "speed": s.get("speed", 0),
        })
    return jsonify(resp)


@app.route("/api/play/<play_id>/player/<int:player_id>/path", methods=["GET"])
def get_player_path(play_id, player_id):
    with lock:
        play = processed_plays.get(play_id)
        frames_players = frame_players_cache.get(play_id, {})
    if not play:
        return jsonify({"error": "Play not processed"}), 404
    path = []
    for f_idx in sorted(frames_players.keys()):
        for p in frames_players.get(f_idx, []):
            try:
                pid = int(p.get("id"))
            except (TypeError, ValueError):
                continue
            if pid == player_id:
                x = p.get("x")
                y = p.get("y")
                if x is not None and y is not None:
                    path.append({"frame": f_idx, "x": int(x), "y": int(y)})
                break
    return jsonify({"path": path})


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "yolo_loaded": yolo_model is not None,
        "device": yolo_device.upper()
    })


# ============================= MAIN =============================
if __name__ == "__main__":
    import io
    if load_field_map() and load_yolo_model():
        print("Models loaded successfully")
        app.run(host="0.0.0.0", port=5000, debug=False)
    else:
        print("Failed to load models")
        exit(1)