import threading
from typing import List, Optional, Tuple
from chromadb import logger
import chromadb


class DatabaseManager:
    """Thread-safe database operations"""
    
    def __init__(self, db_path="./face_data_db"):
        self.db_path = db_path
        self.client = chromadb.PersistentClient(path=db_path)
        self.face_db = self.client.get_or_create_collection("face_db")
        self.reid_name_map = {}
        self.lock = threading.Lock()
        self._load_existing_data()
    
    def _load_existing_data(self):
        """Load existing ReID data from database"""
        try:
            all_faces = self.face_db.get(include=["metadatas"])
            ids = all_faces.get("ids", [])
            metadatas = all_faces.get("metadatas", [])
            
            for i, reid_key in enumerate(ids):
                name = metadatas[i].get("name", "unknown")
                self.reid_name_map[reid_key] = name
            
            logger.info(f"Loaded {len(ids)} existing face records")
        except Exception as e:
            logger.error(f"Error loading existing data: {e}")
    
    def query_face(self, embedding: List[float], threshold: float = 0.5) -> Tuple[Optional[int], Optional[str]]:
        """Query database for face match"""
        with self.lock:
            try:
                qr = self.face_db.query(
                    query_embeddings=[embedding],
                    n_results=1
                )
                match_ids = qr.get("ids", [[]])[0]
                match_dists = qr.get("distances", [[]])[0]
                
                if match_ids and match_dists and match_dists[0] < threshold:
                    matched_key = match_ids[0]
                    matched_num = int(matched_key.split("_")[1])
                    name = self.reid_name_map.get(matched_key, f"unknown_{matched_num}")
                    return matched_num, name
                
                return None, None
            except Exception as e:
                logger.error(f"Error querying face: {e}")
                return None, None
    
    def add_face(self, embedding: List[float], reid_num: int, name: str) -> bool:
        """Add new face to database"""
        with self.lock:
            try:
                key = f"reid_{reid_num}"
                self.face_db.add(
                    ids=[key],
                    embeddings=[embedding],
                    metadatas=[{"name": name}]
                )
                self.reid_name_map[key] = name
                return True
            except Exception as e:
                logger.error(f"Error adding face: {e}")
                return False
    
    def update_name(self, reid_key: str, new_name: str) -> bool:
        """Update face name in database"""
        with self.lock:
            try:
                self.face_db.update(ids=[reid_key], metadatas=[{"name": new_name}])
                self.reid_name_map[reid_key] = new_name
                return True
            except Exception as e:
                logger.error(f"Error updating name: {e}")
                return False
    
    def get_next_reid_num(self) -> int:
        """Get next available ReID number"""
        with self.lock:
            existing_nums = [int(k.split("_")[1]) for k in self.reid_name_map.keys()]
            return max(existing_nums + [0]) + 1
