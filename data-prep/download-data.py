from roboflow import Roboflow
import os



rf = Roboflow(api_key="15agAqN225u1g9uX20Wn")
project = rf.workspace("test-cm1hp").project("nfl-detection-1500")
version = project.version(4)


download_path = "E:/data/Football/all-22"
print(f"Download path: {download_path}")
print(f"Path exists: {os.path.exists(download_path)}")

# Create directory if it doesn't exist
os.makedirs(download_path, exist_ok=True)

try:
    dataset = version.download("yolov12", location=download_path, overwrite=True)
    print("Download successful!")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    import traceback
    traceback.print_exc()