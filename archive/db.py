import os
import threading
import chromadb


class DatabaseManager:
    def __init__(self, db_path: str = "./face_data_db") -> None:
        self.db_path = db_path
        self.client = None
        self.face_db = None       
        self.reid_name_map = {}
        self._lock = threading.Lock()
    
    def _connect(self) -> None:
        """Establishes connection to ChromaDB database."""
        if self.client is not None:
            return
        try:
            os.makedirs(self.db_path, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.db_path)
            self.face_db = self.client.get_or_create_collection("face_db")
            self._load_existing()
            print(f"ChromaDB ready")
        except Exception as e:
            print(f"DB failed ({e}) - using memory")
    
    def _load_existing(self):
        """Load existing face data from database."""
        try:
            if self.face_db:
                print("Check")
                print(self.face_db.count())
                result = self.face_db.get(include=["metadatas"])
                ids = result["ids"]
                metadatas = result["metadatas"]

                for i, key in enumerate(ids):
                    name = "Unknown"
                    if metadatas and i < len(metadatas):
                        name = metadatas[i].get("name", f"unknown_{i}") or "Unknown"
                    self.reid_name_map[key] = name
                print(f"Loaded {len(self.reid_name_map)} existing faces")
        except Exception as e:
            print(f"Loading existing failed: {e}")
    
    def query(self, embedding, threshold: float = 0.4):
        """Query database for matching face."""
        try:
            with self._lock:
                if not self.face_db or self.face_db.count() == 0:
                    return None, None
                
                qr = self.face_db.query(query_embeddings=[embedding], n_results=1)
                ids = qr.get("ids", [[]])[0]
                distances = qr.get("distances", [[]])
                
                if ids and distances and distances[0][0] < threshold:
                    key = ids[0]
                    reid_num = int(key.split("_")[1])
                    name = self.reid_name_map.get(key, f"unknown_{reid_num}")
                    return reid_num, name
        except Exception as e:
            print(f"DB query error: {e}")
        return None, None
    
    def add(self, embedding, reid_num: int, name: str) -> bool:
        """Add new face to database."""
        key = f"reid_{reid_num}"
        try:
            with self._lock:
                if not self.face_db: 
                    return False
                
                self.face_db.add(
                    ids=[key],
                    embeddings=[embedding],
                    metadatas=[{"name": name}]
                )
                self.reid_name_map[key] = name
                return True
        except Exception as e:
            print(f"DB add error: {e}")
            return False
    
    def update_name(self, reid_num: int, new_name: str) -> bool:
        """Update name for existing ReID."""
        key = f"reid_{reid_num}"
        try:
            with self._lock:
                if not self.face_db:
                    print("ERROR: face_db is not initialized")
                    return False
                
                self.face_db.update(
                    ids=[key],
                    metadatas=[{"name": new_name}]
                )
                self.reid_name_map[key] = new_name
                print(f"Updated ReID {reid_num} name to: {new_name}")
                return True
        except Exception as e:
            print(f"DB update error: {e}")
            return False
    
    def next_reid_num(self) -> int:
        """Get next available ReID number."""
        with self._lock:
            if not self.reid_name_map:
                return 1
            
            nums = []
            for key in self.reid_name_map.keys():
                if "_" in key:
                    try:
                        num = int(key.split("_")[1])
                        nums.append(num)
                    except (ValueError, IndexError):
                        continue
            
            return max(nums + [0]) + 1
