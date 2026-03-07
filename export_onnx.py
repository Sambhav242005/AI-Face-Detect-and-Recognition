"""
Export YOLO face model to ONNX and download all EdgeFace ONNX models for browser-side WebGPU inference.
Uses akanametov/yolo-face models: https://github.com/akanametov/yolo-face
Uses otroshi/edgeface models: https://github.com/otroshi/edgeface
"""
import os
import ssl
import urllib.request

# Bypass SSL certificate issues on Windows
ssl._create_default_https_context = ssl._create_unverified_context

FRONTEND_MODELS_DIR = os.path.join("frontend", "models")
os.makedirs(FRONTEND_MODELS_DIR, exist_ok=True)

# ─── 1. Download real YOLO face model and export to ONNX ────────────────────
print("=" * 60)
print("Step 1: Download & export YOLO face model to ONNX...")
print("=" * 60)

yolo_pt = os.path.join("backend", "model", "yolov11n-face.pt")
yolo_onnx_dest = os.path.join(FRONTEND_MODELS_DIR, "yolo-face.onnx")

# Download the actual face detection model from akanametov/yolo-face
if not os.path.exists(yolo_pt):
    os.makedirs(os.path.dirname(yolo_pt), exist_ok=True)
    url = "https://github.com/YapaLab/yolo-face/releases/download/1.0.0/yolov11n-face.pt"
    print(f"Downloading yolov11n-face.pt from {url}...")
    try:
        urllib.request.urlretrieve(url, yolo_pt)
        print(f"Downloaded to {yolo_pt}")
    except Exception as e:
        print(f"ERROR: Failed to download face model: {e}")
        exit(1)
else:
    print(f"Face model already exists at {yolo_pt}")

# Export to ONNX (always re-export to ensure correctness)
from ultralytics import YOLO
model = YOLO(yolo_pt)
print(f"Model classes: {model.names}")
exported_path = model.export(format="onnx", imgsz=640, simplify=True, opset=17)

if os.path.exists(exported_path):
    # Move to frontend/models
    if os.path.exists(yolo_onnx_dest):
        os.remove(yolo_onnx_dest)
    os.rename(exported_path, yolo_onnx_dest)
    print(f"Exported YOLO face ONNX to {yolo_onnx_dest}")
else:
    print(f"ERROR: Expected export at {exported_path} not found!")

# ─── 2. Download & export all EdgeFace models to ONNX ───────────────────────
print()
print("=" * 60)
print("Step 2: Downloading & exporting all EdgeFace models to ONNX...")
print("=" * 60)

import torch

EDGEFACE_MODELS = [
    "edgeface_base",
    "edgeface_s_gamma_05",
    "edgeface_xs_gamma_06",
    "edgeface_xxs",
]

for model_name in EDGEFACE_MODELS:
    onnx_dest = os.path.join(FRONTEND_MODELS_DIR, f"{model_name}.onnx")
    
    if os.path.exists(onnx_dest):
        print(f"  [{model_name}] Already exists at {onnx_dest}, skipping.")
        continue
    
    print(f"  [{model_name}] Loading from torch.hub...")
    try:
        model = torch.hub.load(
            'otroshi/edgeface', model_name,
            source='github', pretrained=True
        )
        model.eval()
        
        dummy_input = torch.randn(1, 3, 112, 112)
        
        torch.onnx.export(
            model,
            dummy_input,
            onnx_dest,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        size_mb = os.path.getsize(onnx_dest) / (1024 * 1024)
        print(f"  [{model_name}] Exported to {onnx_dest} ({size_mb:.1f} MB)")
        
    except Exception as e:
        print(f"  [{model_name}] ERROR: {e}")

print()
print("=" * 60)
print("Done! Models are in:", FRONTEND_MODELS_DIR)
print("=" * 60)
