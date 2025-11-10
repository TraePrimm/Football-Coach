Football Coach — Tracking & Top-down Mapping

A computer vision pipeline and web application to detect players, track movement, and map sideline video to a 2D top-down field.

This project detects players, referees, and the ball in sideline video, maps detections to a top-down field image using homography, and exposes a Flask API consumed by a Vite + React frontend.

🌟 Goals

    Provide an automated pipeline to detect and track players/ball using YOLO.

    Compute and smooth a homography that maps camera video frames to a canonical top-down field image.

    Provide a small web UI to browse plays, trigger processing, and visualize top-down positions and per-player stats.

Repository Layout

    backend/ — Flask REST API and video processing logic (backend/app.py).

    Frontend/ — Vite + React frontend (TypeScript).

    data-analysis/ — Analysis scripts and notebooks.

    data-prep/ — Data download, parsing, and labeling helpers.

    debugging-test-scripts/ — Helper scripts, including generate_H.py.

    environment.yaml — Conda environment for the backend.

    field_map.json — Required field mapping configuration (map size, yard/hash coordinates).

    initial_homography.npy — Required starting homography matrix (see generation steps below).

Prerequisites

    Python: 3.10+ and Conda (or virtualenv).

    Node: 18+ and npm or yarn.

    OpenCV GUI Support: Required for the interactive generate_H.py script.

Configuration & Required Files

Before running, you must provide two key files in the repository root and configure paths in the backend.

1. Required Files

    field_map.json (Already in repo) This JSON file defines the canonical field dimensions and key coordinates. It must include:

        map_dimensions_pixels: {"width": <px>, "height": <px>}

        pixels_per_foot: Conversion factor for speed/distance.

        hash_x_coordinates: Left/right x-values for hash lines.

        yard_line_y_coordinates: Dictionary of yard labels to Y-coordinates.

        hash_points_for_manual_click: Ordered mapping of point names to (x, y) coordinates used by generate_H.py.

    initial_homography.npy (Must be generated) This is a 3x3 NumPy file containing the initial homography matrix. The backend loads this file on startup. Follow the steps below to generate it.

2. Backend Configuration

The backend requires several paths to be set correctly inside backend/app.py.

Update these variables in backend/app.py before running:

    VIDEOS_DIR: Path to your folder containing play subfolders (e.g., play-1/video.mp4, play-2/video.mp4).

    FIELD_MAP_PATH: Path to field_map.json (defaults to repo root).

    INITIAL_H_PATH: Path to initial_homography.npy (defaults to repo root).

    YOLO_MODEL_PATH: Path to your trained YOLO model weights (e.g., model.pt).

Quick Start (End-to-End)

Step 1: Generate Initial Homography

You must create the initial_homography.npy file first.

    Get a sample frame: Extract a clear, representative frame from one of your videos.

    Edit the script: Open debugging-test-scripts/generate_H.py and set TRAINING_FRAME_PATH to the path of your sample frame.

    Check field_map.json: The script will ask you to click points in the exact order they are listed in field_map.json under the hash_points_for_manual_click key.

    Run the script:
    Bash

    # Make sure your conda env is active
    cd debugging-test-scripts
    python generate_H.py

    Click points: An OpenCV window will open. Click the hash points on your frame in the order specified by the terminal prompt.

    Compute & Save: After clicking all points, press c. The script will compute the homography, show a warped test image, and save initial_homography.npy in the same directory.

    Move the file: Move the generated initial_homography.npy to the repository's root directory.

Step 2: Install & Run Backend

    Create Conda environment:
    Bash

conda env create -f environment.yaml -n football-env
conda activate football-env

Verify configuration: Double-check the paths set in backend/app.py (see Configuration section above).

Run Flask server:
Bash

    cd backend
    python app.py

    The server will start on http://localhost:5000.

Step 3: Install & Run Frontend

    Install dependencies:
    Bash

cd Frontend
npm install

Run Vite dev server:
Bash

    npm run dev

    The frontend will be available at http://localhost:5173 (or similar) and will connect to the backend at http://localhost:5000.

    Use the App: Open the frontend URL in your browser. You should see a list of plays. Select one and click "Process" to start the pipeline.

⚙️ Backend API Endpoints

The backend runs at http://localhost:5000.

    GET /api/plays: Returns a list of available play folders.

    GET /api/play/<play_id>/thumbnail: Returns the first-frame thumbnail for the play.

    POST /api/play/<play_id>/process: Starts background processing for the play.

    GET /api/play/<play_id>/status: Returns processing status (not_started, processing, ready).

    GET /api/play/<play_id>/frame/<frame_num>: Returns JSON with base64 images and player data for a frame.

    GET /api/play/<play_id>/all_stats: Returns aggregated stats for all players in the play.

    GET /api/play/<play_id>/player/<player_id>/path: Returns the coordinate path for a single player.

    GET /api/health: Health check (reports YOLO model status).

💡 Roadmap

Short-Term Improvements

    Automated Homography:

        Add a non-interactive homography bootstrap mode (e.g., read matched src/dst points from a CSV) to support headless environments.

        Develop a fully automated homography generation script (e.g., using OpenCV feature detection on field lines and hash marks) to remove the manual-click step entirely.

    Testing: Add unit tests around homography smoothing and player stat calculations.

    Deployment: Add Dockerfiles for consistent deployment of the backend and frontend.

Long-Term Roadmap

    Play Recognition (Run/Pass): Automatically classify the play type (e.g., run vs. pass) from the video and tracking data.

    Formation Recognition (Offense & Defense): Detect and label team formations from player positions and movement patterns.

    Reinforcement Learning (RL) Agent: Once tracking and classification are robust, build an RL agent to explore and evaluate strategic decisions (e.g., given a defensive look, is a run or pass the higher-value option?).