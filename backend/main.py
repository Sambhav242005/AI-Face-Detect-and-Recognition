import asyncio
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from download_models import ensure_models_exist
from session_db import SessionDBManager

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
SESSION_DIR = BASE_DIR / "sessions"

DEFAULT_MODEL_NAME = SessionDBManager.DEFAULT_MODEL_NAME
ALLOWED_MODEL_NAMES = {
    "edgeface_xs_gamma_06",
    "edgeface_xxs",
    "edgeface_s_gamma_05",
    "edgeface_base",
}

MATCH_THRESHOLD = 0.55
KNOWN_REID_MIN_SIM = 0.50
TRACKER_OVERRIDE_THRESHOLD = 0.65
NEW_IDENTITY_MAX_SIM = 0.40
PROFILE_EXPANSION_MIN_SIM = 0.70
PROFILE_EXPANSION_MAX_SIM = 0.85

_session_cache: dict[str, SessionDBManager] = {}
_session_cache_lock = threading.Lock()


def _normalize_model_name(model_name: Optional[str]) -> str:
    model_name = model_name or DEFAULT_MODEL_NAME
    if model_name not in ALLOWED_MODEL_NAMES:
        raise HTTPException(status_code=400, detail=f"Unsupported model_name: {model_name}")
    return model_name


def _cache_session(manager: SessionDBManager) -> None:
    if not manager.active_session_id:
        return
    with _session_cache_lock:
        _session_cache[manager.active_session_id] = manager


def clear_session_cache() -> None:
    """Test helper for resetting the in-memory session registry."""
    with _session_cache_lock:
        _session_cache.clear()


def _model_mismatch_detail(session_model: str, request_model: str) -> str:
    return (
        f"Session uses embedding model {session_model}; "
        f"current request uses {request_model}. Switch models or create a new session."
    )


def get_session_manager(session_id: str, model_name: Optional[str] = None) -> SessionDBManager:
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    requested_model = _normalize_model_name(model_name) if model_name else None
    with _session_cache_lock:
        manager = _session_cache.get(session_id)

    if manager is None:
        manager = SessionDBManager(base_dir=str(SESSION_DIR))
        if not manager.load_session(session_id):
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
        _cache_session(manager)

    if requested_model and manager.model_name != requested_model:
        raise HTTPException(
            status_code=409,
            detail=_model_mismatch_detail(manager.model_name, requested_model),
        )

    return manager


def cleanup_old_session_files(days: float = 3.0) -> None:
    with _session_cache_lock:
        protected_session_ids = set(_session_cache.keys())
    cleanup_db = SessionDBManager(base_dir=str(SESSION_DIR))
    cleanup_db.cleanup_old_sessions(days=days, protected_session_ids=protected_session_ids)


def _coerce_reid(value) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _face_response(
    manager: SessionDBManager,
    reid: Optional[int],
    name: Optional[str],
    similarity: Optional[float],
    *,
    matched: bool,
    created: bool = False,
    override: bool = False,
) -> dict:
    return {
        "status": "success",
        "session_id": manager.active_session_id,
        "model_name": manager.model_name,
        "reid": reid,
        "name": name,
        "distance": similarity,
        "matched": matched,
        "created": created,
        "override": override,
    }


async def periodic_cleanup():
    while True:
        cleanup_old_session_files(days=3.0)
        await asyncio.sleep(43200)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_models_exist()
    cleanup_task = asyncio.create_task(periodic_cleanup())
    try:
        yield
    finally:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass


app = FastAPI(lifespan=lifespan)


class CrossOriginIsolationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Embedder-Policy"] = "credentialless"
        return response


app.add_middleware(CrossOriginIsolationMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health_check():
    return {"status": "ok"}


@app.get("/api/session/new")
async def new_session(model_name: str = DEFAULT_MODEL_NAME):
    model_name = _normalize_model_name(model_name)
    manager = SessionDBManager(base_dir=str(SESSION_DIR), model_name=model_name)
    session_id = manager.create_new_session(model_name=model_name)
    _cache_session(manager)
    return {"status": "success", "session_id": session_id, "model_name": model_name}


@app.get("/api/session/load/{session_id}")
async def load_session(session_id: str, model_name: Optional[str] = None):
    manager = get_session_manager(session_id, model_name=model_name)
    return {
        "status": "success",
        "session_id": manager.active_session_id,
        "model_name": manager.model_name,
    }


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    with _session_cache_lock:
        manager = _session_cache.pop(session_id, None)
    if manager is None:
        manager = SessionDBManager(base_dir=str(SESSION_DIR))
    success = manager.delete_session(session_id)
    return {"status": "success" if success else "error"}


@app.post("/api/face/query")
async def query_face(data: dict):
    """Receive a 512D face embedding from the browser, query FAISS, and return match state."""
    embedding = data.get("embedding")
    session_id = data.get("session_id")
    model_name = data.get("model_name")
    track_id = data.get("track_id", -1)
    known_reid = _coerce_reid(data.get("known_reid"))
    allow_new_identity = bool(data.get("allow_new_identity", False))
    allow_profile_expansion = bool(data.get("allow_profile_expansion", False))

    if embedding is None:
        raise HTTPException(status_code=400, detail="embedding is required")

    manager = get_session_manager(session_id, model_name=model_name)

    try:
        nearest_reid, nearest_name, similarity = manager.nearest_face(embedding)
    except ValueError as exc:
        print(f"WARNING: Invalid embedding rejected from track {track_id}: {exc}")
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    best_reid = (
        nearest_reid
        if nearest_reid is not None and similarity is not None and similarity >= MATCH_THRESHOLD
        else None
    )
    best_name = nearest_name if best_reid is not None else None

    if known_reid is not None:
        if manager.has_reid(known_reid):
            known_name = manager.get_name_by_reid(known_reid)

            # Keep tracker identity only when the nearest embedding is still that identity.
            if nearest_reid == known_reid and similarity is not None and similarity >= KNOWN_REID_MIN_SIM:
                if (
                    allow_profile_expansion
                    and PROFILE_EXPANSION_MIN_SIM <= similarity < PROFILE_EXPANSION_MAX_SIM
                ):
                    manager.expand_face(known_reid, embedding, known_name)
                return _face_response(manager, known_reid, known_name, similarity, matched=True)

            # Break tracker lock if the database confidently says it is someone else.
            if best_reid is not None and best_reid != known_reid and similarity >= TRACKER_OVERRIDE_THRESHOLD:
                print(f"TRACKER OVERRIDE! Track {track_id} swapped from {known_reid} to {best_reid}")
                return _face_response(manager, best_reid, best_name, similarity, matched=True, override=True)

            # Ambiguous tracked crops are not trusted and are not merged into the DB.
            return _face_response(manager, None, "Unknown", similarity, matched=False)

        known_reid = None

    if best_reid is None:
        # Add as a new face only when the caller explicitly allows it and nearest similarity is low.
        if allow_new_identity and similarity is not None and similarity < NEW_IDENTITY_MAX_SIM:
            new_reid = manager.add_face(embedding, "Unknown")
            final_name = f"Person_{new_reid}"
            manager.update_name(new_reid, final_name)
            print(f"New face added: reid={new_reid} sim={similarity:.4f} track={track_id}")
            return _face_response(manager, new_reid, final_name, similarity, matched=True, created=True)

        return _face_response(manager, None, "Unknown", similarity, matched=False)

    return _face_response(manager, best_reid, best_name, similarity, matched=True)


@app.post("/api/face/update")
async def update_face(data: dict):
    """Update name for a given ReID inside the requested session."""
    session_id = data.get("session_id")
    model_name = data.get("model_name")
    reid = data.get("reid")
    name = data.get("name")

    if reid is None or name is None:
        raise HTTPException(status_code=400, detail="reid and name are required")

    try:
        reid_num = int(reid)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="reid must be an integer") from exc

    clean_name = str(name).strip()
    if not clean_name:
        raise HTTPException(status_code=400, detail="name is required")

    manager = get_session_manager(session_id, model_name=model_name)
    if manager.update_name(reid_num, clean_name):
        return {
            "status": "success",
            "session_id": manager.active_session_id,
            "model_name": manager.model_name,
        }

    raise HTTPException(status_code=404, detail=f"ReID not found: {reid}")


app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
