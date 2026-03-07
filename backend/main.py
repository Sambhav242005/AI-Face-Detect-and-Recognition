import os
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from session_db import SessionDBManager
import uvicorn
from download_models import ensure_models_exist
import asyncio

# Security configuration
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
ADMIN_TOKEN = "forge-admin-secure-session" # Static token for simplicity
header_scheme = APIKeyHeader(name="X-Admin-Token", auto_error=False)

async def verify_admin(token: str = Depends(header_scheme)):
    if token != ADMIN_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized admin access"
        )
    return True

app = FastAPI()

async def periodic_cleanup():
    while True:
        db.cleanup_old_sessions(days=3.0)
        # Check every 12 hours (43200 seconds)
        await asyncio.sleep(43200)

@app.on_event("startup")
async def startup_event():
    ensure_models_exist()
    asyncio.create_task(periodic_cleanup())

# Cross-Origin Isolation headers required for WebGPU SharedArrayBuffer
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

import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sessions_dir = os.path.join(current_dir, "sessions")
os.makedirs(sessions_dir, exist_ok=True)

db = SessionDBManager(base_dir=sessions_dir)
db.create_new_session()

# ─── Health ─────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health_check():
    return {"status": "ok"}

@app.post("/api/admin/login")
async def admin_login(data: dict):
    password = data.get("password")
    if password == ADMIN_PASSWORD:
        return {"status": "success", "token": ADMIN_TOKEN}
    else:
        raise HTTPException(status_code=401, detail="Invalid password")

# ─── Session ────────────────────────────────────────────────────────────────

@app.get("/api/session/new")
async def new_session():
    session_id = db.create_new_session()
    return {"status": "success", "session_id": session_id}

@app.get("/api/session/load/{session_id}")
async def load_session(session_id: str):
    success = db.load_session(session_id)
    return {"status": "success" if success else "error"}

@app.delete("/api/session/{session_id}", dependencies=[Depends(verify_admin)])
async def delete_session(session_id: str):
    success = db.delete_session(session_id)
    return {"status": "success" if success else "error"}

# ─── Face DB ────────────────────────────────────────────────────────────────

@app.post("/api/face/query")
async def query_face(data: dict):
    """Receive a 512D face embedding from the browser, query FAISS, return or add face."""
    embedding = data.get("embedding")
    track_id = data.get("track_id", -1)
    known_reid = data.get("known_reid")
    image_b64 = data.get("image_b64")
    
    if embedding is None:
        return {"reid": None, "name": None}
    
    # Validate embedding quality — reject if near-zero norm (broken model output)
    embed_norm = sum(v*v for v in embedding) ** 0.5
    if embed_norm < 0.1:
        print(f"WARNING: Degenerate embedding (norm={embed_norm:.4f}) rejected from track {track_id}")
        return {"reid": None, "name": "ModelError", "distance": 0.0}

    # FAISS cosine SIMILARITY: higher = more similar.
    # Query database for absolute closest match regardless of what tracker thinks
    best_reid, best_name, similarity = db.query_face(embedding, threshold=0.55)
    
    if known_reid is not None:
        known_name = db.get_name_by_reid(known_reid)
        
        # TRACKER OVERRIDE:
        # If the embeddings strongly suggest it's actually SOMEONE ELSE who is already registered,
        # we OVERRIDE the tracker and break the identity lock.
        # This prevents Tracker Swaps (e.g. from User to Friend).
        if best_reid is not None and best_reid != known_reid and similarity > 0.65:
            print(f"TRACKER OVERRIDE! Track {track_id} swapped from {known_reid} to {best_reid}")
            
            # Auto-Merge / ReID self-correction
            # If known_reid was an auto-generated identity (Person_X), it's highly likely it was just 
            # a temporarily fragmented track (e.g. due to extreme head pose dropping <0.45 similarity).
            # We merge these bad angles into the true identity to improve future recognition!
            if known_name.startswith("Person_"):
                print(f"AUTO-MERGE: Merging fragmented {known_reid} into true identity {best_reid}")
                db.merge_faces(merge_from_reid=known_reid, merge_to_reid=best_reid)
                
            return {"reid": best_reid, "name": best_name, "distance": similarity}
            
        # PROFILE EXPANSION: Add new angles/lighting of this face to its DB profile.
        # CRITICAL FIX: Only expand if `best_reid` is actually this person, AND similarity is high enough
        # to guarantee it's them. (0.65 -> 0.85). If it's < 0.65, we don't trust the tracker enough to poison the DB.
        if best_reid == known_reid and similarity is not None and 0.65 <= similarity < 0.85:
            db.expand_face(known_reid, embedding, known_name, image_b64=image_b64)
        
        return {"reid": known_reid, "name": known_name, "distance": similarity}
    
    if best_reid is None:
        # Add as a new face only if similarity is very low (clearly different person)
        if similarity is not None and similarity < 0.45:
            new_reid = db.add_face(embedding, "Unknown")
            final_name = f"Person_{new_reid}"
            db.update_name(new_reid, final_name)
            
            if image_b64:
                db.save_face_image(new_reid, image_b64, is_expansion=False)
                
            print(f"New face added: reid={new_reid} sim={similarity:.4f} track={track_id}")
            return {"reid": new_reid, "name": final_name, "distance": similarity}
        
        # Grey Zone: between 0.45 and 0.55. Don't create new person, but don't match.
        return {"reid": None, "name": "Unknown", "distance": similarity}
    
    return {"reid": best_reid, "name": best_name, "distance": similarity}

@app.post("/api/face/update")
async def update_face(data: dict):
    """Update name for a given ReID."""
    reid = data.get("reid")
    name = data.get("name")
    if reid is not None and name is not None:
        db.update_name(int(reid), name)
        return {"status": "success"}
    return {"status": "error", "message": "Invalid parameters"}

# ─── Admin Dashboard ────────────────────────────────────────────────────────

@app.get("/api/admin/sessions", dependencies=[Depends(verify_admin)])
async def get_all_sessions():
    """Returns a list of all folders in the sessions directory with metadata."""
    sessions = []
    base_dir = "./sessions"
    if not os.path.exists(base_dir):
        return {"sessions": []}
        
    for dirname in os.listdir(base_dir):
        session_path = os.path.join(base_dir, dirname)
        if os.path.isdir(session_path):
            stat = os.stat(session_path)
            sessions.append({
                "id": dirname,
                "created_at": stat.st_ctime,
                "updated_at": stat.st_mtime,
                "is_active": dirname == db.active_session_id
            })
            
    # Sort by newest first
    sessions.sort(key=lambda x: x["updated_at"], reverse=True)
    return {"sessions": sessions}

@app.get("/api/admin/session/{session_id}/metrics", dependencies=[Depends(verify_admin)])
async def get_session_metrics(session_id: str):
    """Fetch metrics for a specific session."""
    if session_id != db.active_session_id:
        return {"error": "Session must be loaded first"}
        
    with db._lock:
        return {
            "total_faces": len(db.metadata),
            "total_embeddings": db.index.ntotal if db.index else 0,
            "session_id": session_id
        }

@app.get("/api/admin/session/{session_id}/faces", dependencies=[Depends(verify_admin)])
async def get_session_faces(session_id: str):
    """List all registered identities and available images in a specific session."""
    if session_id != db.active_session_id:
        return {"error": "Session must be loaded first"}
        
    faces = db.get_all_faces_with_images()
    return {"faces": faces}

@app.get("/api/admin/session/{session_id}/map", dependencies=[Depends(verify_admin)])
async def get_session_pca_map(session_id: str):
    """Fetch 2D-projected face embeddings for charting."""
    if session_id != db.active_session_id:
        return {"error": "Session must be loaded first"}
        
    points = db.get_pca_map()
    return {"points": points}

@app.delete("/api/admin/faces/{reid}", dependencies=[Depends(verify_admin)])
async def admin_delete_face(reid: int):
    """Delete a specific identity completely."""
    success = db.delete_face(reid)
    return {"status": "success" if success else "error"}

@app.post("/api/admin/faces/merge", dependencies=[Depends(verify_admin)])
async def admin_merge_faces(data: dict):
    """Manually merge two ReIDs."""
    merge_from = data.get("merge_from")
    merge_to = data.get("merge_to")
    
    if merge_from is None or merge_to is None:
        return {"status": "error", "message": "Missing merge parameters"}
        
    success = db.merge_faces(int(merge_from), int(merge_to))
    return {"status": "success" if success else "error"}

# ─── Static files (frontend & assets) ─────────────────────────────────────

frontend_dir = os.path.join(os.path.dirname(current_dir), "frontend")

app.mount("/sessions", StaticFiles(directory=sessions_dir), name="sessions")
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
