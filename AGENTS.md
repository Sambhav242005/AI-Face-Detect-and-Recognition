# AGENTS.md

Guidance for coding agents working in this repository.

## Project Summary

This is an experimental browser-first AI face detection and recognition app.

- The frontend runs YOLO face detection and EdgeFace embeddings in the browser with ONNX Runtime Web.
- The backend is a thin FastAPI API for session management and FAISS-backed face embedding lookup.
- Session data is filesystem-backed and research-oriented.
- This project is not production-ready security software. The client-side pipeline is visible and manipulable, CORS is permissive, and raw embeddings are sent over HTTP.

## Current Source Layout

- `backend/main.py`: FastAPI app, cross-origin isolation middleware, request-scoped session registry, REST routes, startup model check, periodic session cleanup, and static frontend mount.
- `backend/session_db.py`: `SessionDBManager`, FAISS `IndexFlatIP`, 512-dimensional embedding storage, JSON metadata with `model_name`, session save/load/delete, identity update/merge/expand.
- `backend/download_models.py`: Checks for required ONNX model files in `frontend/models/` and invokes `export_onnx.py` if any are missing.
- `frontend/index.html`: Static app shell loaded by the backend.
- `frontend/app.js`: Main webcam, model-loading, detection, tracking, embedding, crop-modal, and API client logic.
- `frontend/style.css`: App styling.
- `export_onnx.py`: Downloads/exports YOLO and EdgeFace models into `frontend/models/`.
- `test_embedding.py`: Offline image test harness that mirrors frontend preprocessing and queries the real `SessionDBManager`.
- `archive/`: Legacy experiments only. Do not modify or depend on this folder for the current WebGPU app unless the user explicitly asks.

## Run Locally

The backend can be launched from the repository root. Static frontend and session paths are resolved relative to `backend/main.py`, not the shell cwd.

```powershell
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python backend\main.py
```

Then open:

```text
http://localhost:8000
```

Notes:

- `frontend/app.js` uses `window.location.origin` for API calls, with `http://localhost:8000` only as a file-origin fallback.
- There is no npm build step and no `package.json`; the frontend is static HTML/CSS/JS.
- `frontend/index.html` loads ONNX Runtime Web from the jsDelivr CDN.
- `requirements.txt` is for the active runtime and offline image harness.
- `requirements-models.txt` is for regenerating ONNX files if `frontend/models/` is missing model artifacts.
- `requirements-archive.txt` is only for legacy code under `archive/`.

## Verification Commands

Syntax checks that do not require the full runtime stack:

```powershell
$files = rg --files -g '*.py' -g '!venv/**' -g '!__pycache__/**' -g '!test/**'
python -m py_compile @files
node --check frontend\app.js
```

Backend import and launch smoke checks after dependencies are installed:

```powershell
python -c "import sys; sys.path.insert(0, 'backend'); import main; print(main.FRONTEND_DIR.exists(), main.SESSION_DIR)"
python backend\main.py
```

Focused session-isolation tests:

```powershell
python -m unittest tests.test_session_scoping
```

Offline image harness:

```powershell
python test_embedding.py
```

Place `.jpg`, `.jpeg`, `.png`, or `.bmp` files in `test/` before using the harness. It uses CPU ONNX Runtime and opens OpenCV windows, so it is not a headless CI test.

## Models And Runtime Data

Generated and runtime paths should stay out of commits:

- `frontend/models/`: Generated ONNX files loaded by the browser.
- `backend/model/`: Downloaded YOLO `.pt` source weights.
- `backend/sessions/`: Runtime FAISS session folders.
- `sessions/`: Old root-level session folder from earlier cwd-dependent launches.
- `venv/`, `__pycache__/`, `.pytest_cache/`: Local environment/cache data.

Startup calls `ensure_models_exist()`. If any required model is missing, it runs `export_onnx.py`, which downloads external weights and can take time.

## Backend API

- `GET /api/health`: Health check.
- `GET /api/session/new?model_name=...`: Create a new FAISS session for one embedding model.
- `GET /api/session/load/{session_id}?model_name=...`: Load an existing session and reject incompatible embedding models.
- `DELETE /api/session/{session_id}`: Delete a session.
- `POST /api/face/query`: Query/add recognition state for a 512D embedding in the requested `session_id`.
- `POST /api/face/update`: Rename a ReID identity in the requested `session_id`.

## Recognition Behavior

- Embeddings are L2-normalized and queried with FAISS inner-product similarity.
- Every face query and name update must include `session_id`; the backend no longer has one global active DB.
- Session metadata stores `model_name`; mismatched model/session requests return HTTP `409`.
- New identities are added only when the caller sets `allow_new_identity` and nearest similarity is below `0.40`.
- Similarity from `0.40` to `0.55` is treated as a grey zone for unknown faces.
- Tracker identity is kept only when the nearest stored identity is still that ReID with similarity at least `0.50`.
- Tracker override can happen above `0.65` when the best database identity conflicts with the tracker's known ReID.
- Profile expansion is opt-in through `allow_profile_expansion`; the frontend currently sends `false` to avoid automatic DB pollution.
- The frontend creates a new session when switching embedding models because each model has a different embedding space.

## Important Cautions

- Do not remove the cross-origin isolation headers in `backend/main.py`; WebGPU and SharedArrayBuffer support depend on them.
- Keep changes scoped to the current app files unless the user asks to revive archive code.
- Be careful with model-generation work; `requirements-models.txt` installs large Torch/Ultralytics dependencies.
- Treat model downloads as networked, heavyweight, and non-deterministic.
- Do not present this app as a secure biometric access-control system.

## Portfolio Cover Asset

Maintain a project-specific SVG at `docs/portfolio-cover.svg`.

Rules:
- The SVG must be hand-authored/static, not a raster screenshot, AI-generated image, base64 image, or external asset.
- Use `width="1200"`, `height="760"`, `viewBox="0 0 1200 760"`.
- It should visually summarize the real current project: architecture, workflow, UI, model pipeline, or system behavior.
- Update this SVG whenever major project functionality, architecture, or branding changes.
- Keep text minimal and readable at thumbnail size.
- No fake product names, unrelated placeholder visuals, or generic charts.
- The portfolio repo may copy this file into `public/project-assets/` as the local backup/rendering copy.

## Last Local Check

Checked on 2026-06-25:

- `node --check frontend\app.js` passed.
- Python `py_compile` over `backend/main.py`, `backend/session_db.py`, `backend/download_models.py`, `export_onnx.py`, `test_embedding.py` passed.
- SVG `docs/portfolio-cover.svg` is well-formed XML, 1200×760, no external assets.

Last full-stack verification (2026-06-09):
- `python -m unittest tests.test_session_scoping` passed.
- Backend imports passed from both repo-root and `backend/` working directories.
- `python backend\main.py` served `/api/health` with `{"status":"ok"}` and served the frontend at `/`.
