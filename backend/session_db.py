import os
import shutil
import uuid
import json
import time
import numpy as np
import faiss
import threading
import base64
from typing import Optional, Tuple


class SessionDBManager:
    """Face embedding database backed by FAISS (IndexFlatIP for cosine similarity)."""

    EMBED_DIM = 512

    def __init__(self, base_dir="./sessions"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self._lock = threading.Lock()
        self.active_session_id = None
        self.index = None          # faiss.IndexFlatIP
        self.metadata = []         # [{reid_num, name, key}]
        self.reid_name_map = {}    # key -> name  (for fast lookup)
        self._next_reid = 1

    # ─── Sessions ───────────────────────────────────────────────────────────

    def create_new_session(self) -> str:
        with self._lock:
            session_id = str(uuid.uuid4())
            self.active_session_id = session_id
            self.index = faiss.IndexFlatIP(self.EMBED_DIM)
            self.metadata = []
            self.reid_name_map = {}
            self._next_reid = 1
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
            json.dump({"metadata": self.metadata, "next_reid": self._next_reid}, f)

    def cleanup_old_sessions(self, days: float = 3.0):
        """Delete sessions that haven't been modified in the given number of days."""
        with self._lock:
            if not os.path.exists(self.base_dir):
                return
            now = time.time()
            cutoff = now - (days * 86400)
            cleaned = 0
            for dirname in os.listdir(self.base_dir):
                if dirname == self.active_session_id:
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

    def query_face(self, embedding, threshold: float = 0.40) -> Tuple[Optional[int], Optional[str], Optional[float]]:
        """Query the FAISS index for the closest face.
        
        threshold: cosine SIMILARITY threshold. Values > threshold are considered a match.
        FAISS IndexFlatIP returns inner product (= cosine sim for L2-normalized vectors).
        Typical: >0.40 = same person, <0.30 = different person.
        """
        with self._lock:
            if self.index is None or self.index.ntotal == 0:
                return None, None, 0.0  # similarity=0 means no match

            # L2 normalize for cosine similarity
            emb = np.array(embedding, dtype=np.float32).reshape(1, -1)
            faiss.normalize_L2(emb)

            D, I = self.index.search(emb, 1)  # D = similarities, I = indices
            sim = float(D[0][0])
            idx = int(I[0][0])

            if idx < 0 or idx >= len(self.metadata):
                return None, None, sim

            if sim >= threshold:
                meta = self.metadata[idx]
                reid_num = meta["reid_num"]
                name = meta["name"]
                print(f"Match: sim={sim:.4f} reid={reid_num} ({name})")
                return reid_num, name, sim

            print(f"No match: sim={sim:.4f} (threshold={threshold})")
            return None, None, sim

    def add_face(self, embedding, name: str) -> int:
        with self._lock:
            if self.index is None:
                return -1

            reid_num = self._next_reid
            self._next_reid += 1
            key = f"reid_{reid_num}"

            emb = np.array(embedding, dtype=np.float32).reshape(1, -1)
            faiss.normalize_L2(emb)
            self.index.add(emb)

            self.metadata.append({"reid_num": reid_num, "name": name, "key": key})
            self.reid_name_map[key] = name
            
            self._save_session()
            return reid_num

    def save_face_image(self, reid_num: int, image_b64: str, is_expansion: bool = False):
        """Save a cropped face image to disk."""
        if not self.active_session_id or not image_b64:
            return
            
        try:
            # Create directory for this identity
            face_dir = os.path.join(self.base_dir, self.active_session_id, "faces", str(reid_num))
            os.makedirs(face_dir, exist_ok=True)
            
            # Generate filename
            filename = f"{uuid.uuid4().hex[:8]}.jpg"
            if not is_expansion:
                filename = f"primary_{filename}"
                
            file_path = os.path.join(face_dir, filename)
            
            # Decode and save
            # Handle potential Data URI prefix (data:image/jpeg;base64,...)
            if "," in image_b64:
                image_b64 = image_b64.split(",")[1]
                
            image_data = base64.b64decode(image_b64)
            with open(file_path, "wb") as f:
                f.write(image_data)
        except Exception as e:
            print(f"Failed to save face image: {e}")

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

    def expand_face(self, reid_num: int, embedding, name: str, image_b64: Optional[str] = None):
        """Add a new embedding variant for an existing identity if it's sufficiently novel."""
        with self._lock:
            if self.index is None or self.index.ntotal == 0:
                return

            try:
                emb = np.array(embedding, dtype=np.float32).reshape(1, -1)
                faiss.normalize_L2(emb)

                D, I = self.index.search(emb, 1)
                sim = float(D[0][0])

                # If similarity < 0.70, this is a sufficiently novel expression/angle
                if sim < 0.70:
                    new_key = f"reid_{reid_num}_{str(uuid.uuid4())[:8]}"
                    self.index.add(emb)
                    self.metadata.append({"reid_num": reid_num, "name": name, "key": new_key})
                    self.reid_name_map[new_key] = name
                    self._save_session()
                    
                    if image_b64:
                        self.save_face_image(reid_num, image_b64, is_expansion=True)
                        
                    print(f"Profile expanded for {name} (reid {reid_num}). Novelty sim: {sim:.4f}")
            except Exception as e:
                print(f"DB expand error: {e}")
                
    def get_all_faces_with_images(self) -> list:
        """Get all known identities and a list of their saved cropped images."""
        result = []
        with self._lock:
            if not self.active_session_id:
                return result
                
            for m in self.metadata:
                reid = m["reid_num"]
                name = m["name"]
                
                # Look for images
                images = []
                face_dir = os.path.join(self.base_dir, self.active_session_id, "faces", str(reid))
                if os.path.exists(face_dir):
                    images = [f for f in os.listdir(face_dir) if f.endswith('.jpg')]
                    
                result.append({
                    "reid": reid,
                    "name": name,
                    "images": images,
                    "total_embeddings": 1 # we can count duplicates later if needed
                })
        return result

    def get_pca_map(self) -> list:
        """Extract all embeddings from the FAISS index and project them to 2D using PCA."""
        with self._lock:
            if self.index is None or self.index.ntotal == 0:
                return []
                
            try:
                from sklearn.decomposition import PCA
                
                # Extract all vectors from the IndexFlatIP
                n_total = self.index.ntotal
                
                # For IndexFlatIP, we can directly reconstruct vectors
                # reconstruct_n(0, n) gets vectors from 0 to n
                vectors = np.zeros((n_total, self.EMBED_DIM), dtype=np.float32)
                for i in range(n_total):
                    vectors[i] = self.index.reconstruct(i)
                
                # If we have less than 2 vectors, PCA 2D won't work
                if n_total < 2:
                    return [{"x": 0.0, "y": 0.0, "reid": self.metadata[0]["reid_num"], "name": self.metadata[0]["name"]}]
                    
                # Calculate PCA (reduce 512D to 2D)
                pca = PCA(n_components=2)
                coords_2d = pca.fit_transform(vectors)
                
                # Map back to identities
                result = []
                for i in range(n_total):
                    meta = self.metadata[i]
                    result.append({
                        "x": float(coords_2d[i][0]),
                        "y": float(coords_2d[i][1]),
                        "reid": meta["reid_num"],
                        "name": meta["name"]
                    })
                    
                return result
            except Exception as e:
                print(f"PCA generation error: {e}")
                return []

    def delete_face(self, reid_num: int) -> bool:
        """Delete an identity completely (removes from metadata and rebuilds FAISS index)."""
        with self._lock:
            if self.index is None:
                return False
                
            # Filter metadata
            new_metadata = [m for m in self.metadata if m["reid_num"] != reid_num]
            
            if len(new_metadata) == len(self.metadata):
                return False # Nothing removed
                
            # Unfortunately, FAISS IndexFlatIP doesn't easily support targeted deletion by ID.
            # We must rebuild the index from the remaining metadata embeddings.
            
            # 1. Extract keeping embeddings based on index
            keep_indices = [i for i, m in enumerate(self.metadata) if m["reid_num"] != reid_num]
            
            if not keep_indices:
                # We deleted everyone
                self.index = faiss.IndexFlatIP(self.EMBED_DIM)
            else:
                n_total = self.index.ntotal
                new_index = faiss.IndexFlatIP(self.EMBED_DIM)
                
                for i in range(n_total):
                    if i in keep_indices:
                        vec = self.index.reconstruct(i)
                        new_index.add(vec.reshape(1, -1))
                        
                self.index = new_index
                
            # Update state
            self.metadata = new_metadata
            self.reid_name_map = {m["key"]: m["name"] for m in self.metadata}
            
            # Clean up images
            if self.active_session_id:
                face_dir = os.path.join(self.base_dir, self.active_session_id, "faces", str(reid_num))
                if os.path.exists(face_dir):
                    try:
                        import shutil
                        shutil.rmtree(face_dir)
                    except:
                        pass
            
            self._save_session()
            return True
