import cv2
import numpy as np
import json
import os

# --- CONFIGURATION ---
TRAINING_FRAME_PATH = "E:\\data\\Football\\all-22\\train\\images\\week01-Arizona-Cardinals-KansasCityChiefs-ArizonaCardinals-037-222_jpg.rf.4de29025ea7fa269ebea4072226b15f5.jpg"
LABEL_FILE_PATH = "E:\\data\\Football\\all-22\\train\\labels\\week01-Arizona-Cardinals-KansasCityChiefs-ArizonaCardinals-037-222_jpg.rf.4de29025ea7fa269ebea4072226b15f5.txt"
FIELD_MAP_PATH = 'field_map.json'

# YOLO class IDs
CLASS_PLAYER = 0
CLASS_REF = 1
CLASS_BALL = 2

# Colors (BGR)
COLOR_PLAYER = (0, 0, 255)      # Red
COLOR_BALL = (255, 255, 255)  # White
COLOR_REF = (0, 255, 255)     # Yellow

# --- GLOBALS ---
source_points = []
frame_copy_for_clicks = None
current_point_index = 0
window_name_clicks = 'Click Points for Homography'
POINTS_TO_COLLECT = []
DESTINATION_MAP = {}
MAP_WIDTH_PX = 1600
MAP_HEIGHT_PX = 3000
YARD_LINES_Y = {}
HASH_LINES_X = {}
PIXELS_PER_FOOT = 10

# --- HELPER: Resize for display ---
def resize_for_display(img, max_h=1400, max_w=1920):
    """Resize image to fit within max_h x max_w, preserve aspect ratio."""
    h, w = img.shape[:2]
    scale = min(max_h / h, max_w / w, 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        resized = img.copy()
    return resized, scale

# --- LOAD FIELD MAP ---
def load_field_map():
    global MAP_WIDTH_PX, MAP_HEIGHT_PX, POINTS_TO_COLLECT, DESTINATION_MAP, YARD_LINES_Y, HASH_LINES_X, PIXELS_PER_FOOT
    try:
        with open(FIELD_MAP_PATH, 'r') as f:
            field_map = json.load(f)

        MAP_WIDTH_PX = field_map['map_dimensions_pixels']['width']
        MAP_HEIGHT_PX = field_map['map_dimensions_pixels']['height']
        PIXELS_PER_FOOT = field_map.get('pixels_per_foot', 10)

        YARD_LINES_Y = field_map['yard_line_y_coordinates']
        HASH_LINES_X = field_map['hash_x_coordinates']

        click_points = field_map['hash_points_for_manual_click']
        for name, coords in click_points.items():
            POINTS_TO_COLLECT.append(name)
            DESTINATION_MAP[name] = coords

        print("Field map loaded successfully.")
        return True
    except Exception as e:
        print(f"Error loading field map: {e}")
        return False

# --- MOUSE CALLBACK ---
def mouse_callback(event, x, y, flags, param):
    global source_points, frame_copy_for_clicks, current_point_index
    if event == cv2.EVENT_LBUTTONDOWN:
        if current_point_index >= len(POINTS_TO_COLLECT):
            return
        point_name = POINTS_TO_COLLECT[current_point_index]
        source_points.append([x, y])
        print(f"Collected: {point_name} at ({x}, {y})")
        cv2.circle(frame_copy_for_clicks, (x, y), 7, (0, 255, 0), -1)
        cv2.putText(frame_copy_for_clicks, str(current_point_index + 1), (x + 10, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        current_point_index += 1
        if current_point_index < len(POINTS_TO_COLLECT):
            print(f"\n---> Click: {POINTS_TO_COLLECT[current_point_index]}\n")
        else:
            print("\nAll points collected! Press 'c' to compute H, 'r' to reset.")

# --- HOMOGRAPHY FROM CLICKS ---
def get_homography_from_clicks(frame):
    global frame_copy_for_clicks, source_points, current_point_index
    frame_copy_for_clicks = frame.copy()
    original_frame = frame.copy()
    cv2.namedWindow(window_name_clicks, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name_clicks, mouse_callback)
    print(f"\n---> Click: {POINTS_TO_COLLECT[0]}\n")

    while True:
        cv2.imshow(window_name_clicks, frame_copy_for_clicks)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            return None
        if key == ord('r'):
            source_points = []
            current_point_index = 0
            frame_copy_for_clicks = original_frame.copy()
            print(f"\n---> Click: {POINTS_TO_COLLECT[0]}\n")
        if key == ord('c') and current_point_index == len(POINTS_TO_COLLECT):
            print("Computing homography...")
            dest_list = [DESTINATION_MAP[n] for n in POINTS_TO_COLLECT]
            src = np.array(source_points, dtype=np.float32)
            dst = np.array(dest_list, dtype=np.float32)
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 10.0)
            print("Homography matrix:\n", H)

            # Warp test
            warped_full = cv2.warpPerspective(original_frame, H, (MAP_WIDTH_PX, MAP_HEIGHT_PX))
            warped_disp, scale = resize_for_display(warped_full)

            # Overlay reference lines on scaled warp
            sorted_lines = sorted(YARD_LINES_Y.items(), key=lambda x: int(x[1]))
            for name, y_src in sorted_lines:
                y_d = int(y_src * scale)
                cv2.line(warped_disp, (0, y_d), (warped_disp.shape[1], y_d), (0, 255, 0), 2)

            cv2.namedWindow("Warped Test", cv2.WINDOW_NORMAL)
            cv2.imshow("Warped Test", warped_disp)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return H

# --- GET BOTTOM-CENTER OF YOLO BOX ---
def get_hotspot(box_x, box_y, box_w, box_h, img_w, img_h):
    x = box_x * img_w
    y = box_y * img_h + (box_h * img_h) / 2
    return np.array([[[x, y]]], dtype=np.float32)

# --- DRAW ORIGINAL YOLO BOXES ---
def draw_original_boxes(frame, label_path):
    if not os.path.exists(label_path):
        return
    h, w = frame.shape[:2]
    with open(label_path, 'r') as f:
        for line in f:
            try:
                cid, xc, yc, bw, bh = map(float, line.strip().split())
                cid = int(cid)
                x1 = int((xc - bw/2) * w)
                y1 = int((yc - bh/2) * h)
                x2 = int((xc + bw/2) * w)
                y2 = int((yc + bh/2) * h)
                color = COLOR_PLAYER if cid == CLASS_PLAYER else \
                        COLOR_BALL if cid == CLASS_BALL else COLOR_REF
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            except:
                continue

# --- MAIN ---
def run_test():
    if not load_field_map():
        return

    frame = cv2.imread(TRAINING_FRAME_PATH)
    if frame is None:
        print("Failed to load image.")
        return

    img_h, img_w = frame.shape[:2]
    H = get_homography_from_clicks(frame.copy())
    if H is None:
        return

    # --- BUILD FULL-RES TOP-DOWN MAP (FIELD LINES ONLY) ---
    top_down_map = np.zeros((MAP_HEIGHT_PX, MAP_WIDTH_PX, 3), dtype=np.uint8)
    line_color = (255, 255, 255)
    hash_color = (128, 128, 128)
    pixels_per_yard = int(PIXELS_PER_FOOT * 3)

    # Yard lines
    sorted_yard_lines = sorted(YARD_LINES_Y.items(), key=lambda x: int(x[1]))
    for name, y in sorted_yard_lines:
        y = int(y)
        thick = 3 if name == "50" else 2
        cv2.line(top_down_map, (0, y), (MAP_WIDTH_PX, y), line_color, thick)

    # Hash marks
    x_left = int(HASH_LINES_X['left_hash_x'])
    x_right = int(HASH_LINES_X['right_hash_x'])
    hash_len = int(PIXELS_PER_FOOT * 0.5)
    y_coords = sorted(int(v) for v in YARD_LINES_Y.values())
    for i in range(len(y_coords)-1):
        y0, y1 = y_coords[i], y_coords[i+1]
        for y_hash in range(y0 + pixels_per_yard, y1, pixels_per_yard):
            cv2.line(top_down_map, (x_left, y_hash - hash_len), (x_left, y_hash + hash_len), hash_color, 2)
            cv2.line(top_down_map, (x_right, y_hash - hash_len), (x_right, y_hash + hash_len), hash_color, 2)

    # Draw original boxes
    draw_original_boxes(frame, LABEL_FILE_PATH)

    # --- DISPLAY (SCALED) ---
    print("\nDisplaying results – 'q' to quit, 's' to save full-res map.")
    cv2.imshow("Original Frame with Labels", frame)

    # Scale full map (which is just the field lines at this point)
    disp_map, scale = resize_for_display(top_down_map)
    hud = disp_map.copy()

    # Draw **reference yard lines** on scaled HUD
    for name, y_src in sorted_yard_lines:
        y_d = int(y_src * scale)
        cv2.line(hud, (0, y_d), (hud.shape[1], y_d), (0, 255, 0), 2)
        label = "50" if name == "50" else \
                f"Opp {name.replace('Other_', '')}" if name.startswith("Other_") else \
                f"Own {name}"
        cv2.putText(hud, label, (10, y_d - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
    # --- Project player hotspots (Draw on full-res map AND scaled HUD) ---
    if os.path.exists(LABEL_FILE_PATH):
        with open(LABEL_FILE_PATH, 'r') as f:
            for line in f:
                try:
                    cid, xc, yc, bw, bh = map(float, line.strip().split())
                    cid = int(cid)
                    hotspot = get_hotspot(xc, yc, bw, bh, img_w, img_h)
                    pt = cv2.perspectiveTransform(hotspot, H)[0][0]
                    
                    # Full-res coordinates
                    x_full, y_full = int(pt[0]), int(pt[1])
                    
                    # Scaled-res coordinates
                    x_scaled, y_scaled = int(pt[0] * scale), int(pt[1] * scale)

                    if cid == CLASS_PLAYER:
                        # Draw on full-res map (for saving)
                        cv2.circle(top_down_map, (x_full, y_full), 10, COLOR_PLAYER, -1)
                        # Draw on scaled HUD (for display) - scale radius but keep a min size
                        cv2.circle(hud, (x_scaled, y_scaled), max(2, int(10 * scale)), COLOR_PLAYER, -1)
                    elif cid == CLASS_BALL:
                        cv2.circle(top_down_map, (x_full, y_full), 5, COLOR_BALL, -1)
                        cv2.circle(hud, (x_scaled, y_scaled), max(1, int(5 * scale)), COLOR_BALL, -1)
                    elif cid == CLASS_REF:
                        cv2.circle(top_down_map, (x_full, y_full), 8, COLOR_REF, -1)
                        cv2.circle(hud, (x_scaled, y_scaled), max(2, int(8 * scale)), COLOR_REF, -1)
                except:
                    continue

    # This line is optional, uncomment to show scale info
    # cv2.putText(hud, f"Scale: {scale:.2f}x  |  Original: {MAP_HEIGHT_PX}×{MAP_WIDTH_PX}",
    #             (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.namedWindow("Top-Down Analytics View", cv2.WINDOW_NORMAL)
    cv2.imshow("Top-Down Analytics View", hud)

    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        if k == ord('s'):
            path = "top_down_full_res.png"
            cv2.imwrite(path, top_down_map)
            print(f"Full-res map saved: {path}")
    cv2.destroyAllWindows()

# --- RUN ---
if __name__ == "__main__":
    run_test()