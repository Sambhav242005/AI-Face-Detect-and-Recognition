# AI Face Detect and Recognition WebGPU

A client-side AI face tracking and recognition application leveraging WebGPU and ONNX runtime to run models in the browser.

> **⚠ WARNING: RESEARCH PROJECT**
> This project is a **hackable, experimental research project** meant to explore capabilities of running AI in the browser via WebGPU. It is **NOT** intended for production security, access control, or sensitive environments. The architecture involves sending cropped images and raw embeddings over HTTP, and running client-side tracking, meaning the data pipeline is fully visible and manipulable by the end-user. Please use responsibly.

## Features

- **Browser-based AI**: Runs YOLOv8 face detection and EdgeFace face embeddings directly in the browser via WebGPU/WASM ONNX Runtime.
- **FastAPI Backend**: A lightweight Python backend that acts purely as an API server for the SQLite/FAISS session database.
- **Session Management**: Session persistence and deletion capabilities.
- **Image Identification Tool**: Upload an image to quickly crop and identify a face using the active session database.

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

3. Run the FastAPI backend server:

   ```bash
   python backend/main.py
   ```

4. Open your browser and navigate to `http://localhost:8000`.

### 3. Running via Docker

You can easily spin up the environment using Docker:

```bash
docker build -t face-ai-webgpu .
docker run -p 8000:8000 face-ai-webgpu
```

## Note on Repository Structure (Archive Folder)

You may notice an `archive/` folder in the root directory. This directory intentionally contains previous experimental python scripts, legacy data pipelines, and deprecated architectures used early in development. It is kept solely for reference and historical context, and is not utilized by the current WebGPU-based application.
