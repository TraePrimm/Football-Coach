import cv2
import numpy as np
import sys

# --- CONFIGURATION ---

# IMPORTANT: Path to your video file
VIDEO_PATH = 'your_all_22_film.mp4' 

# IMPORTANT: Define your 2D map's "ground truth" coordinates.
# This is your "analytics" view. It NEVER changes.
# We'll map 100 yards (300 feet) to 3000 pixels (10 px/foot)
# We'll map 53.3 yards (160 feet) to 1600 pixels (10 px/foot)

MAP_WIDTH_PX = 3000 # 100 yards @ 10 px/foot
MAP_HEIGHT_PX = 1600 # 53.3 yards @ 10 px/foot

# NFL Hashes are 70' 9" (70.75 ft) from the sideline.
# Sidelines are 160 ft apart.
# Our map's Y=0 is the Top Sideline. Y=1600 is the Bottom Sideline.
# Left Hash Y = 70.75 ft * 10 px/ft = 707.5 (let's use 708)
# Right Hash Y = (160 ft - 70.75 ft) * 10 px/ft = 892.5 (let's use 892)
HASH_Y_LEFT = 708
HASH_Y_RIGHT = 892

# X-coordinates (yard lines)
# 0-yard line (Goalline) = 0
# 50-yard line = 150 ft * 10 px/ft = 1500
# 100-yard line (Other Goalline) = 300 ft * 10 px/ft = 3000
# 40-yard line (closer to 50) = (150 - 30) * 10 = 1200
# 40-yard line (further from 50) = (150 + 30) * 10 = 1800

# This is our static, ground-truth "lookup table"
# We only need 4 points, but more is better for RANSAC. Let's use 6.
DESTINATION_MAP = {
    # (x, y) coordinates on our ideal 2D map
    "40_Yd_Left_Hash": (1200, HASH_Y_LEFT),
    "40_Yd_Right_Hash": (1200, HASH_Y_RIGHT),
    "50_Yd_Left_Hash": (1500, HASH_Y_LEFT),
    "50_Yd_Right_Hash": (1500, HASH_Y_RIGHT),
    "Other_40_Yd_Left_Hash": (1800, HASH_Y_LEFT),
    "Other_40_Yd_Right_Hash": (1800, HASH_Y_RIGHT),
}

# This is the list of points we will ask the user to click, in order.
POINTS_TO_COLLECT = [
    "40_Yd_Left_Hash", "40_Yd_Right_Hash",
    "50_Yd_Left_Hash", "50_Yd_Right_Hash",
    "Other_40_Yd_Left_Hash", "Other_40_Yd_Right_Hash"
]

# --- SCRIPT GLOBALS ---
source_points = []
frame_copy = None
current_point_index = 0
window_name = 'First Frame - Click Points'

def mouse_callback(event, x, y, flags, param):
    """
    OpenCV Mouse Callback Function
    This function is called every time a mouse event happens in the window.
    """
    global source_points, frame_copy, current_point_index

    if event == cv2.EVENT_LBUTTONDOWN:
        if current_point_index >= len(POINTS_TO_COLLECT):
            return # Already collected all points

        # 1. Store the clicked point (the "source")
        point_name = POINTS_TO_COLLECT[current_point_index]
        source_points.append([x, y]) # Use list for numpy compatibility
        
        print(f"Collected: {point_name} at pixel ({x}, {y})")

        # 2. Draw visual feedback on the frame
        cv2.circle(frame_copy, (x, y), 7, (0, 255, 0), -1) # Green dot
        cv2.putText(frame_copy, str(current_point_index + 1), (x + 10, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # 3. Update to ask for the next point
        current_point_index += 1
        
        if current_point_index < len(POINTS_TO_COLLECT):
            next_point_name = POINTS_TO_COLLECT[current_point_index]
            print(f"\n---> NOW, please click on: {next_point_name}\n")
        else:
            print("\nAll points collected! Press 'c' to calculate H.")
            print("Or 'r' to reset.")

def run_bootstrap():
    """
    Main function to run the bootstrap process.
    """
    global frame_copy, source_points, current_point_index

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {VIDEO_PATH}")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        cap.release()
        return
    
    frame_copy = frame.copy()
    original_frame = frame.copy() # Keep a clean copy for resetting

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("--- Initial Homography Setup ---")
    print(f"Window opened. Please click on the first point:")
    print(f"\n---> NOW, please click on: {POINTS_TO_COLLECT[0]}\n")

    while True:
        cv2.imshow(window_name, frame_copy)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('q'): # Quit
            print("Setup cancelled.")
            break
        
        if key == ord('r'): # Reset
            print("Resetting points. Start over.")
            source_points = []
            current_point_index = 0
            frame_copy = original_frame.copy()
            print(f"\n---> NOW, please click on: {POINTS_TO_COLLECT[0]}\n")

        if key == ord('c') and current_point_index == len(POINTS_TO_COLLECT):
            print("Calculating Homography Matrix...")
            
            # Build the destination_points list in the *same order*
            # as the source_points were collected.
            destination_points_list = []
            for point_name in POINTS_TO_COLLECT:
                destination_points_list.append(DESTINATION_MAP[point_name])
            
            # Convert to numpy arrays
            np_source = np.array(source_points, dtype=np.float32)
            np_dest = np.array(destination_points_list, dtype=np.float32)

            # Find the homography matrix
            H, mask = cv2.findHomography(np_source, np_dest, cv2.RANSAC, 5.0)

            print("\nHomography Matrix (H) Found:")
            print(H)
            
            # Save the matrix to a file
            output_file = 'initial_homography.npy'
            np.save(output_file, H)
            print(f"\nSuccessfully saved to {output_file}")
            
            # --- Test the warp ---
            print("Warping image for a test... Close the 'Warped Test' window to exit.")
            warped_image = cv2.warpPerspective(original_frame, H, (MAP_WIDTH_PX, MAP_HEIGHT_PX))
            
            cv2.namedWindow("Warped Test", cv2.WINDOW_NORMAL)
            cv2.imshow("Warped Test", warped_image)
            cv2.waitKey(0) # Wait until user presses a key in the 'Warped Test' window
            break # Exit the main loop

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if VIDEO_PATH == 'your_all_22_film.mp4':
        print("ERROR: Please update the 'VIDEO_PATH' variable in the script.")
        sys.exit(1)
    run_bootstrap()