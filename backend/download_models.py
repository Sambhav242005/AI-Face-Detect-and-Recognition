import os
import subprocess
import sys

# Models to ensure are available
MODELS = [
    "edgeface_xs_gamma_06.onnx",
    "yolo-face.onnx",
    "edgeface_xxs.onnx",
    "edgeface_s_gamma_05.onnx",
    "edgeface_base.onnx"
]

def ensure_models_exist():
    # Target directory is frontend/models so WebGPU can load them
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    frontend_models_dir = os.path.join(project_root, "frontend", "models")
    os.makedirs(frontend_models_dir, exist_ok=True)

    print("Checking for required ONNX models in frontend/models/...")

    missing_models = [m for m in MODELS if not os.path.exists(os.path.join(frontend_models_dir, m))]
            
    if missing_models:
        print(f"Missing ONNX models: {missing_models}")
        print("Invoking export_onnx.py to build them locally from original repos...")
        export_script = os.path.join(project_root, "export_onnx.py")
        if os.path.exists(export_script):
            subprocess.run([sys.executable, export_script], cwd=project_root, check=True)
        else:
            print(f"ERROR: Cannot find {export_script} to generate models.")
    else:
        print("All required ONNX models are present.")

if __name__ == "__main__":
    ensure_models_exist()
