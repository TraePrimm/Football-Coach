import cv2
import numpy as np
import sys
import json

# --- CONFIGURATION ---

# IMPORTANT: Path to your *single* training frame
TRAINING_FRAME_PATH = "E:\\data\\Football\\test_videos\\play-1\\frames\\frame_0000.jpg"

# IMPORTANT: Path to the field map JSON
FIELD_MAP_PATH = 'field_map.json'

# --- SCRIPT GLOBALS ---
source_points = []
frame_copy = None
current_point_index = 0
window_name = 'First Frame - Click Points'
POINTS_TO_COLLECT = []
DESTINATION_MAP = {}
MAP_WIDTH_PX = 3000
MAP_HEIGHT_PX = 1600

def load_field_map():
    """Loads map coordinates from the JSON file."""
    global MAP_WIDTH_PX, MAP_HEIGHT_PX, POINTS_TO_COLLECT, DESTINATION_MAP
    try:
        with open(FIELD_MAP_PATH, 'r') as f:
            field_map = json.load(f)
        
        MAP_WIDTH_PX = field_map['map_dimensions_pixels']['width']
        MAP_HEIGHT_PX = field_map['map_dimensions_pixels']['height']
        
        # Get the points to click from the JSON
        click_points = field_map['hash_points_for_manual_click']
        
        # Re-structure for the script
        for point_name, coords in click_points.items():
            POINTS_TO_COLLECT.append(point_name)
            DESTINATION_MAP[point_name] = coords
            
        print("Successfully loaded field map from JSON.")
        return True

    except FileNotFoundError:
        print(f"Error: Could not find field map file: {FIELD_MAP_PATH}")
        return False
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {FIELD_MAP_PATH}")
        return False
    except KeyError as e:
        print(f"Error: JSON file is missing expected key: {e}")
        return False

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

    frame = cv2.imread(TRAINING_FRAME_PATH)
    if frame is None:
        print(f"Error: Could not read image file: {TRAINING_FRAME_PATH}")
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

    cv2.destroyAllWindows()

if __name__ == "__main__":
    if TRAINING_FRAME_PATH == 'frame_001.jpg':
        print("WARNING: Please update the 'TRAINING_FRAME_PATH' variable in the script.")
        # We don't exit, as the user might have that file
        
    if not load_field_map():
        sys.exit(1)
        
    run_bootstrap()