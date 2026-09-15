# AI Face Detect and Recognition WebGPU

A client-side AI face tracking and recognition application leveraging WebGPU and ONNX runtime to run models in the browser.

**Live demo:** https://ai-face.sambhav-surana.online/

> **⚠ WARNING: RESEARCH PROJECT**
> This project is a **hackable, experimental research project** meant to explore capabilities of running AI in the browser via WebGPU. It is **NOT** intended for production security, access control, or sensitive environments. The architecture involves sending cropped images and raw embeddings over HTTP, and running client-side tracking, meaning the data pipeline is fully visible and manipulable by the end-user. Please use responsibly.

## Features

- **Browser-based AI**: Runs YOLOv8 face detection and EdgeFace face embeddings directly in the browser via WebGPU/WASM ONNX Runtime.
- **FastAPI Backend**: A lightweight Python backend that acts purely as an API server for the FAISS session database and JSON metadata.
- **Session Management**: Session persistence and deletion capabilities.
- **Pause/Resume Tracking**: Pause live recognition without unloading the camera, models, or active session.
- **Image Identification Tool**: Upload an image to quickly crop and identify a face using the selected session database.

## Prerequisites

- Python 3.10+
- A modern browser with WebGPU support enabled (Chrome/Edge 113+).

## Setup & Run

### 1. Model Generation

The ONNX models (YOLO and EdgeFace) are compiled locally. The application will automatically execute the `export_onnx.py` script to pull the original weights directly from the `akanametov/yolo-face` and `otroshi/edgeface` GitHub repositories and export them into the `frontend/models/` directory the first time you run the backend server. You do not need to download or host anything manually.

### 2. Running Locally

1. Clone this repository and create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   If `frontend/models/` is missing ONNX models and you need the app to generate them locally, also install:

   ```bash
   pip install -r requirements-models.txt
   ```

   The `archive/` folder has separate legacy dependencies. Install them only when working on archived experiments:

   ```bash
   pip install -r requirements-archive.txt
   ```

3. Run the FastAPI backend server:

   ```bash
   python backend/main.py
   ```

4. Open your browser and navigate to `http://localhost:8000`.

### 3. Running via Docker (incl. Raspberry Pi)

Runtime image is slim: FastAPI + FAISS only. No torch/ultralytics inside
(browser runs WebGPU inference). ONNX files must be baked or mounted.

```bash
docker build -t face-ai-webgpu .
docker run -p 8000:8000 face-ai-webgpu
```

Pi flow (export on PC first, models ~110 MB, git-ignored):

```bash
# On a PC:
pip install -r requirements-models.txt
python export_onnx.py
# Copy frontend/models/*.onnx to the Pi (scp/rsync/USB), then build/run there.
# Or mount at runtime without rebuilding:
docker run -p 8000:8000 \
  -v ./frontend/models:/app/frontend/models:ro \
  -v face-sessions:/app/backend/sessions \
  face-ai-webgpu
```

Notes:
- Container sets `SKIP_MODEL_EXPORT=1`, so startup never runs torch export.
  Missing models = warning only; `/api/health` stays green, browser shows
  which `.onnx` is absent.
- If you built an old image that installed `torch==...+cu124`/ultralytics,
  rebuild with `--no-cache` once; that layer is gone.
- Your old `-p 8100:8000` mapping is fine too, then open `http://<pi>:8100`.

## Note on Repository Structure (Archive Folder)

You may notice an `archive/` folder in the root directory. This directory intentionally contains previous experimental python scripts, legacy data pipelines, and deprecated architectures used early in development. It is kept solely for reference and historical context, and is not utilized by the current WebGPU-based application.
