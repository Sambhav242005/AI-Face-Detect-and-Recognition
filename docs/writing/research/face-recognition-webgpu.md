---
contentKind: article
slug: face-recognition-webgpu
title: Real-Time Face Recognition via WebGPU
type: research-note
status: published
date: 2026-05-12
summary: Deploying YOLOv8 face detection and EdgeFace embeddings directly to browser GPUs for zero-latency client-side inference.
tags:
  - WebGPU
  - ONNX
  - Computer Vision
---

Most face recognition systems run expensive, server-side GPU models (like PyTorch on CUDA). This architecture increases hosting costs and poses massive privacy issues since user images must travel over the network. I built a research prototype that runs fully local, client-side inference directly in the browser via WebGPU.

## Client-Side Pipeline: YOLOv8 and EdgeFace

The browser-based AI pipeline relies on two models compiled to ONNX:
- **YOLOv8-face**: For real-time face detection and bounding box coordinate calculations.
- **EdgeFace**: A high-performance embedding model that extracts 512-dimension face representation vectors.

By utilizing ONNX Runtime WebGPU and WebAssembly, these models run directly on the client's local GPU.

## Preventing UI Thread Jitter with Web Workers

Running heavy inference loops on webcam streams causes frame rate drops and rendering delays. To maintain a smooth 60 FPS user experience:
- Frame extraction, canvas cropping, and ONNX model evaluation are offloaded to a background `Web Worker`.
- Bounding boxes and embeddings are sent back to the main thread via structured clones, ensuring the UI remains completely responsive.

## The Coordinate Server (FastAPI + FAISS)

The backend acts purely as a coordinate server:
- It exposes a lightweight FastAPI API that stores and queries session embeddings.
- It leverages a FAISS vector index for near-instant classification of incoming embeddings, mapping them to saved user metadata.
