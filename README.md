# AI Face Detect and Recognition

A high-performance, real-time face detection and recognition system leveraging YOLOv11 for detection and ChromaDB for efficient face embedding storage and retrieval. This project supports both a robust desktop application (OpenCV) and a modern web interface (Gradio/WebRTC).

## Features

-   **Real-time Detection**: Utilizes the state-of-the-art YOLOv11 model for fast and accurate face detection.
-   **Robust Tracking**: Implements object tracking to maintain face IDs across frames.
-   **Face Recognition**: Uses face embeddings and ChromaDB to recognize and identify individuals.
-   **Desktop App**: A multi-process, multi-threaded OpenCV application designed for high throughput and low latency.
-   **Performance Optimized**: Features a sophisticated multi-processing architecture separating detection, encoding, and rendering to maximize FPS.
-   **Interactive Renaming**: Easily assign names to recognized faces directly within the application.

## Prerequisites

-   **Python**: 3.8 or higher
-   **CUDA**: Recommended for GPU acceleration (requires NVIDIA GPU).

## Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Sambhav242005/AI-Face-Detect-and-Recognition
    cd AI-Face-Detect-and-Recognition
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Model Weights**
    Ensure the YOLOv11 face model weights are located at `model/yolov11l-face.pt`.

## Usage

### Desktop Application

The desktop application offers the best performance and feature set.

```bash
python face.py
```

**Controls:**
-   `q` or `ESC`: Quit the application.
-   `s`: Enter interactive renaming mode (assign names to detected IDs).
-   `i`: Display detailed performance metrics (FPS, queue sizes, etc.).



## Project Structure

-   `face.py`: Main entry point for the desktop application. Handles multi-processing orchestration.

-   `db.py`: Manages ChromaDB operations for storing and retrieving face embeddings.
-   `face_encoding_worker.py`: Worker script for handling face encoding tasks in separate processes.
-   `process_embedding_result.py`: Alternative implementation containing face saving functionality and embedding processing logic.
-   `model/`: Directory containing model weights.

## Troubleshooting

-   **Camera Issues**: If the camera doesn't open, check the `camera_id` in `face.py` (default is 0).
-   **Performance**: If FPS is low, ensure CUDA is enabled and available. The system automatically falls back to CPU if CUDA is missing, which is significantly slower.
