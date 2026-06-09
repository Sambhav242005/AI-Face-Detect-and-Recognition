import os
import shutil
import uuid
import json
import time
import numpy as np
import faiss
import threading
import math
from typing import Optional, Tuple


class SessionDBManager:
    """Face embedding database backed by FAISS (IndexFlatIP for cosine similarity)."""

    EMBED_DIM = 512
    DEFAULT_MODEL_NAME = "edgeface_xs_gamma_06"

    def __init__(self, base_dir="./sessions", model_name: str = DEFAULT_MODEL_NAME):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self._lock = threading.Lock()
        self.active_session_id = None
        self.model_name = model_name
        self.index = None          # faiss.IndexFlatIP
        self.metadata = []         # [{reid_num, name, key}]
        self.reid_name_map = {}    # key -> name  (for fast lookup)
        self._next_reid = 1

    # ─── Sessions ───────────────────────────────────────────────────────────

    def create_new_session(self, model_name: Optional[str] = None) -> str:
        with self._lock:
            if model_name:
                self.model_name = model_name
            session_id = str(uuid.uuid4())
            self.active_session_id = session_id
            self.index = faiss.IndexFlatIP(self.EMBED_DIM)
            self.metadata = []
            self.reid_name_map = {}
            self._next_reid = 1
            self._save_session()
            print(f"Created new session: {session_id}")
            return session_id

    def load_session(self, session_id: str) -> bool:
        with self._lock:
            session_dir = os.path.join(self.base_dir, session_id)
            index_path = os.path.join(session_dir, "faiss.index")
            meta_path = os.path.join(session_dir, "metadata.json")

            if not os.path.exists(index_path) or not os.path.exists(meta_path):
                print(f"Session {session_id} not found on disk.")
                return False

            try:
                self.index = faiss.read_index(index_path)
                with open(meta_path, "r") as f:
                    data = json.load(f)
                self.metadata = data["metadata"]
                self.reid_name_map = {m["key"]: m["name"] for m in self.metadata}
                self._next_reid = data.get("next_reid", 1)
                self.model_name = data.get("model_name", self.DEFAULT_MODEL_NAME)
                self.active_session_id = session_id
                print(f"Loaded session: {session_id} with {len(self.metadata)} embeddings")
                return True
            except Exception as e:
                print(f"Failed to load session {session_id}: {e}")
                return False

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            session_dir = os.path.join(self.base_dir, session_id)
            if not os.path.exists(session_dir):
                return False
            try:
                shutil.rmtree(session_dir)
                if self.active_session_id == session_id:
                    self.active_session_id = None
                    self.index = None
                    self.metadata = []
                    self.reid_name_map = {}
                print(f"Deleted session: {session_id}")
                return True
            except Exception as e:
                print(f"Failed to delete session {session_id}: {e}")
                return False

    def _save_session(self):
        """Persist current session to disk (call while holding lock)."""
        if not self.active_session_id or self.index is None:
            return
        session_dir = os.path.join(self.base_dir, self.active_session_id)
        os.makedirs(session_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(session_dir, "faiss.index"))
        with open(os.path.join(session_dir, "metadata.json"), "w") as f:
            json.dump({
                "metadata": self.metadata,
                "next_reid": self._next_reid,
                "model_name": self.model_name,
            }, f)

    def cleanup_old_sessions(self, days: float = 3.0, protected_session_ids: Optional[set] = None):
        """Delete sessions that haven't been modified in the given number of days."""
        with self._lock:
            if not os.path.exists(self.base_dir):
                return
            protected_session_ids = protected_session_ids or set()
            now = time.time()
            cutoff = now - (days * 86400)
            cleaned = 0
            for dirname in os.listdir(self.base_dir):
                if dirname == self.active_session_id or dirname in protected_session_ids:
                    continue
                session_dir = os.path.join(self.base_dir, dirname)
                if not os.path.isdir(session_dir):
                    continue
                path_to_check = os.path.join(session_dir, "metadata.json")
                if not os.path.exists(path_to_check):
                    path_to_check = session_dir
                try:
                    mtime = os.path.getmtime(path_to_check)
                    if mtime < cutoff:
                        shutil.rmtree(session_dir)
                        print(f"Auto-deleted old session: {dirname}")
                        cleaned += 1
                except Exception as e:
                    print(f"Error cleaning session {dirname}: {e}")
            if cleaned > 0:
                print(f"Auto-deleted {cleaned} old unused sessions.")

    # ─── Face DB ────────────────────────────────────────────────────────────

    def _normalized_embedding(self, embedding) -> np.ndarray:
        emb = np.array(embedding, dtype=np.float32).reshape(1, -1)
        if emb.shape[1] != self.EMBED_DIM:
            raise ValueError(f"Expected {self.EMBED_DIM}D embedding, got {emb.shape[1]}D")
        if not np.isfinite(emb).all():
            raise ValueError("Embedding contains non-finite values")
        norm = float(np.linalg.norm(emb))
        if not math.isfinite(norm) or norm < 0.1:
            raise ValueError(f"Degenerate embedding norm={norm:.4f}")
        faiss.normalize_L2(emb)
        return emb

    def nearest_face(self, embedding) -> Tuple[Optional[int], Optional[str], Optional[float]]:
        """Return the closest stored face regardless of match threshold."""
        with self._lock:
            if self.index is None or self.index.ntotal == 0:
                return None, None, 0.0

            emb = self._normalized_embedding(embedding)
            D, I = self.index.search(emb, 1)
            sim = float(D[0][0])
            idx = int(I[0][0])

            if idx < 0 or idx >= len(self.metadata):
                return None, None, sim

            meta = self.metadata[idx]
            return meta["reid_num"], meta["name"], sim

    def query_face(self, embedding, threshold: float = 0.40) -> Tuple[Optional[int], Optional[str], Optional[float]]:
        """Query the FAISS index for the closest face.

        threshold: cosine SIMILARITY threshold. Values > threshold are considered a match.
        FAISS IndexFlatIP returns inner product (= cosine sim for L2-normalized vectors).
        Typical: >0.40 = same person, <0.30 = different person.
        """
        reid_num, name, sim = self.nearest_face(embedding)
        if reid_num is not None and sim is not None and sim >= threshold:
            print(f"Match: sim={sim:.4f} reid={reid_num} ({name})")
            return reid_num, name, sim

        if sim is not None:
            print(f"No match: sim={sim:.4f} (threshold={threshold})")
        return None, None, sim

    def add_face(self, embedding, name: str) -> int:
        with self._lock:
            if self.index is None:
                return -1

            reid_num = self._next_reid
            self._next_reid += 1
            key = f"reid_{reid_num}"

            emb = self._normalized_embedding(embedding)
            self.index.add(emb)

            self.metadata.append({"reid_num": reid_num, "name": name, "key": key})
            self.reid_name_map[key] = name

            self._save_session()
            return reid_num

    def update_name(self, reid_num: int, new_name: str) -> bool:
        with self._lock:
            key = f"reid_{reid_num}"
            updated = False
            for m in self.metadata:
                if m["reid_num"] == reid_num:
                    m["name"] = new_name
                    updated = True
            if updated:
                self.reid_name_map[key] = new_name
                self._save_session()
            return updated

    def has_reid(self, reid_num: int) -> bool:
        with self._lock:
            return any(m["reid_num"] == reid_num for m in self.metadata)

    def merge_faces(self, merge_from_reid: int, merge_to_reid: int) -> bool:
        """Merge all embeddings of one identity into another.</br>Used to continuously learn from Tracker Overrides."""
        with self._lock:
            # Check if target exists
            target_name = None
            for m in self.metadata:
                if m["reid_num"] == merge_to_reid:
                    target_name = m["name"]
                    break
            
            if not target_name:
                return False
                
            updated = False
            for m in self.metadata:
                if m["reid_num"] == merge_from_reid:
                    m["reid_num"] = merge_to_reid
                    m["name"] = target_name
                    self.reid_name_map[m["key"]] = target_name
                    updated = True
                    
            if updated:
                self._save_session()
            
            return updated


    def get_name_by_reid(self, reid_num: int) -> str:
        with self._lock:
            for m in self.metadata:
                if m["reid_num"] == reid_num:
                    return m["name"]
            return f"Person_{reid_num}"

    def expand_face(self, reid_num: int, embedding, name: str):
        """Add a new embedding variant for an existing identity if it's sufficiently novel."""
        with self._lock:
            if self.index is None or self.index.ntotal == 0:
                return

            try:
                emb = self._normalized_embedding(embedding)

                D, I = self.index.search(emb, 1)
                sim = float(D[0][0])

                # If similarity < 0.70, this is a sufficiently novel expression/angle
                if sim < 0.70:
                    new_key = f"reid_{reid_num}_{str(uuid.uuid4())[:8]}"
                    self.index.add(emb)
                    self.metadata.append({"reid_num": reid_num, "name": name, "key": new_key})
                    self.reid_name_map[new_key] = name
                    self._save_session()
                    print(f"Profile expanded for {name} (reid {reid_num}). Novelty sim: {sim:.4f}")
            except Exception as e:
                print(f"DB expand error: {e}")
