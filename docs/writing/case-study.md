---
contentKind: case-study
---

## Problem

Traditional face recognition systems require expensive server-side GPU resources for video frame inference, which raises privacy concerns and increases hosting costs.

## Approach

I built a client-side face recognition system that offloads YOLOv8 face detection and EdgeFace embedding extraction directly to the user's browser GPU using ONNX Runtime WebGPU. The backend acts as a lightweight coordinate system, using FastAPI and FAISS to index and search embeddings.

## Technical Decisions

- YOLOv8 face detection compiled to ONNX runs inside a Web Worker via ONNX Runtime WebGPU.
- EdgeFace models generate 512-dimension face embeddings client-side for low latency.
- FastAPI serves backend session routes, saving and loading FAISS database structures locally.
- Built pipeline controls supporting frame capture pausing, real-time bounding box renders, and photo-based lookup overrides.

## Result

This research proof proves that computer vision inference pipelines can run efficiently on client-side GPUs, maintaining user privacy and reducing server load.
