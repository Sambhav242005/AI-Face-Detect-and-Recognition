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

SKIP_EXPORT_FLAG = os.environ.get("SKIP_MODEL_EXPORT", "").strip() in {"1", "true", "yes"}
STRICT_CHECK = os.environ.get("STRICT_MODEL_CHECK", "").strip() in {"1", "true", "yes"}


def _warn_missing(missing_models, frontend_models_dir):
    print(f"WARNING: Missing ONNX models: {missing_models}")
    print(f"WARNING: Expected in {frontend_models_dir}/")
    print("WARNING: Server keeps running (API/health OK); browser face pipeline needs those files.")
    print("  PC:  python -m pip install -r requirements-models.txt && python export_onnx.py")
    print("  Pi/Docker: export on a PC first, then COPY or mount frontend/models/*.onnx.")
    print("  Set STRICT_MODEL_CHECK=1 to crash on missing models instead.")


def ensure_models_exist():
    # Target directory is frontend/models so WebGPU can load them
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    frontend_models_dir = os.path.join(project_root, "frontend", "models")
    os.makedirs(frontend_models_dir, exist_ok=True)

    print("Checking for required ONNX models in frontend/models/...")

    missing_models = [m for m in MODELS if not os.path.exists(os.path.join(frontend_models_dir, m))]

    if not missing_models:
        print("All required ONNX models are present.")
        return

    if SKIP_EXPORT_FLAG:
        _warn_missing(missing_models, frontend_models_dir)
        if STRICT_CHECK:
            raise RuntimeError(f"Missing ONNX models: {missing_models}")
        return

    print(f"Missing ONNX models: {missing_models}")
    print("Invoking export_onnx.py to build them locally from original repos...")
    print("Tip: set SKIP_MODEL_EXPORT=1 on Pi/Docker to skip this (needs torch).")
    export_script = os.path.join(project_root, "export_onnx.py")
    if os.path.exists(export_script):
        try:
            subprocess.run([sys.executable, export_script], cwd=project_root, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
            # Never kill the API over model generation: torch/ultralytics are
            # heavyweight, Pi-hostile, and unneeded at runtime (browser runs
            # WebGPU). Warn and keep serving so /api/health stays green.
            print(f"WARNING: Model export failed ({exc}). Server keeps running.")
            print("Install model-generation dependencies with "
                  "`python -m pip install -r requirements-models.txt`, then rerun, "
                  "or copy prebuilt frontend/models/*.onnx into place.")
            if STRICT_CHECK:
                raise RuntimeError(
                    "Model export failed. Install model-generation dependencies with "
                    "`python -m pip install -r requirements-models.txt`, then rerun the server."
                ) from exc
            # Re-check so logs show what is still missing.
            still_missing = [m for m in MODELS if not os.path.exists(os.path.join(frontend_models_dir, m))]
            if still_missing:
                _warn_missing(still_missing, frontend_models_dir)
    else:
        print(f"ERROR: Cannot find {export_script} to generate models.")

if __name__ == "__main__":
    ensure_models_exist()
