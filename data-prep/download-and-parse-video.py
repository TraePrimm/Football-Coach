import os
import cv2


def parse_video_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()

def main():
    video_input_path = "E:/data/Football/test_videos"
    for folder in os.listdir(video_input_path):
        folder_path = os.path.join(video_input_path, folder)
        if folder_path and os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    if file.endswith('.mp4'):
                        video_file_path = os.path.join(folder_path, file)
                        video_output_path = os.path.join(folder_path, "frames")
                        print(f"parsing video: {video_file_path}")
                        parse_video_frames(video_file_path, video_output_path)
    
    
if __name__ == "__main__":
    main()
