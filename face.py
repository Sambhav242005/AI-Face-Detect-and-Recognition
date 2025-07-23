import cv2
import time
import numpy as np
from ultralytics import YOLO
import chromadb
import threading
import queue
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import logging
import torch
import warnings
import os

# Import with error handling - use threading instead of multiprocessing to avoid pickle issues
try:
    import face_encoding_worker
    FACE_ENCODING_AVAILABLE = True
except ImportError:
    print("Warning: face_encoding_worker module not found. Please ensure it exists.")
    face_encoding_worker = None
    FACE_ENCODING_AVAILABLE = False

# Suppress some warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FaceDetection:
    """Data class for face detection results"""
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float
    track_id: int
    face_crop: np.ndarray
    frame_timestamp: float

@dataclass
class FaceEmbedding:
    """Data class for face embeddings"""
    track_id: int
    embedding: List[float]
    reid_num: Optional[int] = None
    name: Optional[str] = None

class DatabaseManager:
    """Thread-safe database operations"""
    
    def __init__(self, db_path="./face_data_db"):
        self.db_path = db_path
        self.client = None
        self.face_db = None
        self.reid_name_map = {}
        self.lock = threading.Lock()
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize database with error handling"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.db_path, exist_ok=True)
            
            self.client = chromadb.PersistentClient(path=self.db_path)
            self.face_db = self.client.get_or_create_collection("face_db")
            self._load_existing_data()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            # Create a fallback in-memory database
            try:
                self.client = chromadb.Client()
                self.face_db = self.client.create_collection("face_db")
                logger.info("Using in-memory database as fallback")
            except Exception as e2:
                logger.error(f"Fallback database creation failed: {e2}")
                self.face_db = None
    
    def _load_existing_data(self):
        """Load existing ReID data from database"""
        if not self.face_db:
            return
        
        try:
            all_faces = self.face_db.get(include=["metadatas"])
            ids = all_faces.get("ids", [])
            metadatas = all_faces.get("metadatas", [])
            
            for i, reid_key in enumerate(ids):
                if i < len(metadatas) and metadatas[i]:
                    name = metadatas[i].get("name", "unknown")
                    self.reid_name_map[reid_key] = name
            
            logger.info(f"Loaded {len(ids)} existing face records")
        except Exception as e:
            logger.error(f"Error loading existing data: {e}")
    
    def query_face(self, embedding: List[float], threshold: float = 0.4) -> Tuple[Optional[int], Optional[str]]:
        """Query database for face match"""
        if not self.face_db:
            return None, None
            
        with self.lock:
            try:
                qr = self.face_db.query(
                    query_embeddings=[embedding],
                    n_results=1
                )
                match_ids = qr.get("ids", [[]])[0]
                match_dists = qr.get("distances", [[]])[0]
                
                if match_ids and match_dists and len(match_dists) > 0 and match_dists[0] < threshold:
                    matched_key = match_ids[0]
                    # Better parsing of reid number
                    try:
                        matched_num = int(matched_key.split("_")[1])
                        name = self.reid_name_map.get(matched_key, f"unknown_{matched_num}")
                        return matched_num, name
                    except (IndexError, ValueError) as e:
                        logger.error(f"Error parsing reid key {matched_key}: {e}")
                        return None, None
                
                return None, None
            except Exception as e:
                logger.error(f"Error querying face: {e}")
                return None, None
    
    def add_face(self, embedding: List[float], reid_num: int, name: str) -> bool:
        """Add new face to database"""
        if not self.face_db:
            return False
            
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
        if not self.face_db:
            return False
            
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
            try:
                existing_nums = []
                for k in self.reid_name_map.keys():
                    try:
                        num = int(k.split("_")[1])
                        existing_nums.append(num)
                    except (IndexError, ValueError):
                        continue
                return max(existing_nums + [0]) + 1
            except Exception as e:
                logger.error(f"Error getting next reid num: {e}")
                return 1


class FaceRecognitionSystem:
    """Main face recognition system with multi-threading"""
    
    def __init__(self, model_path='model/yolov11l-face.pt', camera_id=0):
        # Validate model path
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model with error handling
        try:
            self.model = YOLO(model_path)
            # Try to move to CUDA if available
            if torch.cuda.is_available():
                try:
                    self.model.to('cuda:0')
                    logger.info("Model loaded on CUDA")
                except Exception as e:
                    logger.warning(f"Failed to move model to CUDA: {e}, using CPU")
            else:
                logger.info("CUDA not available, using CPU")
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            raise
        
        # Initialize camera with error handling
        self.cap = None
        self._initialize_camera(camera_id)
        
        self.db_manager = DatabaseManager()
        
        # Threading components
        self.detection_queue = queue.Queue(maxsize=10)
        self.embedding_queue = queue.Queue(maxsize=20)
        self.result_queue = queue.Queue(maxsize=20)
        
        # Tracking data
        self.track_id_to_embedding = {}
        self.track_id_to_reid = {}
        self.processing_tracks = set()  # Tracks currently being processed
        
        # Threading pools - use ThreadPoolExecutor to avoid pickle issues
        max_workers = min(2, mp.cpu_count())
        self.embedding_executor = ThreadPoolExecutor(max_workers=max_workers) if FACE_ENCODING_AVAILABLE else None
        self.db_executor = ThreadPoolExecutor(max_workers=1)
        
        # Control flags
        self.running = True
        self.frame_lock = threading.Lock()
        self.detection_lock = threading.Lock()
        self.current_frame = None
        self.current_detections = []
        self.visible_reids = []
        
        # FPS tracking
        self.prev_time = time.time()
        self.frame_count = 0
        self.fps = 0
    
    def _initialize_camera(self, camera_id):
        """Initialize camera with multiple backends"""
        backends = [cv2.CAP_DSHOW, cv2.CAP_V4L2, cv2.CAP_ANY]
        
        for backend in backends:
            try:
                self.cap = cv2.VideoCapture(camera_id, backend)
                if self.cap.isOpened():
                    # Test read
                    ret, frame = self.cap.read()
                    if ret:
                        logger.info(f"Camera initialized successfully with backend {backend}")
                        # Set camera properties for better performance
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        self.cap.set(cv2.CAP_PROP_FPS, 30)
                        return
                    else:
                        self.cap.release()
                        self.cap = None
            except Exception as e:
                logger.warning(f"Failed to initialize camera with backend {backend}: {e}")
                continue
        
        logger.error("Failed to initialize camera with any backend")
        raise RuntimeError("Could not initialize camera")
    
    def detection_thread(self):
        """Thread for YOLO face detection - runs once per frame"""
        logger.info("Detection thread started")
        
        while self.running:
            try:
                if not self.cap or not self.cap.isOpened():
                    logger.error("Camera not available")
                    time.sleep(1)
                    continue
                
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    time.sleep(0.1)
                    continue
                
                # Validate frame
                if frame is None or frame.size == 0:
                    continue
                
                # Run YOLO detection once per frame
                try:
                    results = self.model.track(frame, stream=True, persist=True, 
                                            tracker="botsort.yaml", verbose=False)
                except Exception as e:
                    logger.error(f"YOLO tracking error: {e}")
                    # Fallback to prediction without tracking
                    try:
                        results = self.model.predict(frame, verbose=False)
                    except Exception as e2:
                        logger.error(f"YOLO prediction error: {e2}")
                        time.sleep(0.1)
                        continue
                
                detections = []
                embedding_candidates = []
                
                for result in results:
                    boxes = getattr(result, 'boxes', None)
                    if boxes is None:
                        continue
                        
                    for box in boxes:
                        try:
                            # Safely extract coordinates
                            coords = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], 'cpu') else box.xyxy[0]
                            x1, y1, x2, y2 = map(int, coords)
                            conf = float(box.conf[0])
                            
                            # Handle tracking ID safely
                            if hasattr(box, 'id') and box.id is not None:
                                track_id = int(box.id[0])
                            else:
                                # Generate a deterministic ID based on box position and size
                                track_id = abs(hash(f"{x1}_{y1}_{x2}_{y2}")) % 100000
                            
                            # Validate bounding box
                            if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0:
                                continue
                            
                            if conf > 0.5:
                                detection = {
                                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                                    'conf': conf, 'track_id': track_id
                                }
                                detections.append(detection)
                                
                                # Check if we need to process this track for embedding
                                if (track_id not in self.track_id_to_embedding and 
                                    track_id not in self.processing_tracks and
                                    FACE_ENCODING_AVAILABLE and self.embedding_executor is not None):
                                    
                                    # Validate crop coordinates
                                    h, w = frame.shape[:2]
                                    x1_crop = max(0, x1)
                                    y1_crop = max(0, y1)
                                    x2_crop = min(w, x2)
                                    y2_crop = min(h, y2)
                                    
                                    if x2_crop > x1_crop and y2_crop > y1_crop:
                                        face_crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]
                                        if face_crop.size > 0:  # Ensure valid crop
                                            embedding_candidates.append(FaceDetection(
                                                x1=x1, y1=y1, x2=x2, y2=y2,
                                                conf=conf, track_id=track_id,
                                                face_crop=face_crop.copy(),
                                                frame_timestamp=time.time()
                                            ))
                        except Exception as e:
                            logger.error(f"Error processing detection box: {e}")
                            continue
                
                # Store frame and detections for rendering
                with self.frame_lock:
                    self.current_frame = frame.copy()
                with self.detection_lock:
                    self.current_detections = detections
                
                # Send only new faces for embedding processing
                if embedding_candidates and self.embedding_executor:
                    try:
                        self.detection_queue.put(embedding_candidates, timeout=0.01)
                    except queue.Full:
                        pass  # Skip if queue is full
                
                # Small delay to prevent CPU overload
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                logger.error(f"Error in detection thread: {e}")
                time.sleep(0.1)  # Prevent rapid error loops
    
    def embedding_thread(self):
        """Thread for processing face embeddings"""
        logger.info("Embedding thread started")
        
        if not FACE_ENCODING_AVAILABLE or not self.embedding_executor:
            logger.warning("Embedding executor not available, embedding thread disabled")
            return
        
        while self.running:
            try:
                detections = self.detection_queue.get(timeout=1.0)
                
                for detection in detections:
                    track_id = detection.track_id
                    
                    # Skip if already processed or currently processing
                    if (track_id in self.track_id_to_embedding or 
                        track_id in self.processing_tracks):
                        continue
                    
                    # Mark as processing
                    self.processing_tracks.add(track_id)
                    
                    # Submit face encoding task using ThreadPoolExecutor (avoids pickle issues)
                    try:
                        # Create a wrapper function to avoid pickle issues
                        def encode_face_wrapper(face_crop):
                                return face_encoding_worker.face_encoding_worker(face_crop)
                        future = self.embedding_executor.submit(
                            encode_face_wrapper, detection.face_crop
                        )
                        
                        # Submit database query task
                        self.db_executor.submit(
                            self._process_embedding_result, 
                            future, detection
                        )
                    except Exception as e:
                        logger.error(f"Error submitting embedding task: {e}")
                        self.processing_tracks.discard(track_id)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in embedding thread: {e}")
    
    def _process_embedding_result(self, future, detection):
        """Process the result of face encoding"""
        track_id = detection.track_id
        
        try:
            embedding = future.result(timeout=10.0)  # Increased timeout
            
            if embedding is None:
                logger.warning(f"No embedding generated for track {track_id}")
                self.processing_tracks.discard(track_id)
                return
            
            # Store embedding
            self.track_id_to_embedding[track_id] = embedding
            
            # Query database for match
            reid_num, name = self.db_manager.query_face(embedding)
            
            if reid_num is not None:
                # Found existing person
                self.track_id_to_reid[track_id] = reid_num
                logger.info(f"Matched track {track_id} to existing person: {name}")
            else:
                # New person
                new_reid_num = self.db_manager.get_next_reid_num()
                new_name = f"unknown_{new_reid_num}"
                
                if self.db_manager.add_face(embedding, new_reid_num, new_name):
                    self.track_id_to_reid[track_id] = new_reid_num
                    logger.info(f"Added new person: {new_name} for track {track_id}")
                else:
                    logger.error(f"Failed to add new person for track {track_id}")
            
        except Exception as e:
            logger.error(f"Error processing embedding result for track {track_id}: {e}")
        finally:
            # Always remove from processing set
            self.processing_tracks.discard(track_id)
    
    def render_frame(self):
        """Render frame with face recognition results - no YOLO inference"""
        with self.frame_lock:
            if self.current_frame is None:
                return None
            frame = self.current_frame.copy()
        
        with self.detection_lock:
            detections = self.current_detections.copy()
        
        self.visible_reids = []
        
        # Process cached detections (no YOLO inference here)
        for detection in detections:
            x1, y1, x2, y2 = detection['x1'], detection['y1'], detection['x2'], detection['y2']
            conf = detection['conf']
            track_id = detection['track_id']
            
            # Determine label and color
            label = "Processing..."
            color = (0, 165, 255)  # Orange for processing
            
            if track_id in self.processing_tracks:
                label = "Processing..."
                color = (0, 165, 255)
            elif track_id in self.track_id_to_reid:
                reid_num = self.track_id_to_reid[track_id]
                key = f"reid_{reid_num}"
                self.visible_reids.append(key)
                name = self.db_manager.reid_name_map.get(key, f"unknown_{reid_num}")
                label = f"{name} ({key})"
                color = (0, 255, 0)  # Green for recognized
            else:
                if self.embedding_executor is None:
                    label = "Embedding disabled"
                    color = (128, 128, 128)  # Gray
                else:
                    label = "Unknown"
                    color = (0, 0, 255)  # Red for unknown
            
            # Validate coordinates before drawing
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(0, min(x2, w-1))
            y2 = max(0, min(y2, h-1))
            
            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Ensure label position is within frame
            label_y = max(y1 - 10, 20)
            cv2.putText(frame, f"{label} | ID {track_id} | {conf:.2f}",
                       (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Display FPS (more accurate calculation)
        curr_time = time.time()
        self.frame_count += 1
        if curr_time - self.prev_time >= 1.0:  # Update FPS every second
            self.fps = self.frame_count / (curr_time - self.prev_time)
            self.prev_time = curr_time
            self.frame_count = 0
        
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        
        # Display processing info
        processing_count = len(self.processing_tracks)
        cv2.putText(frame, f"Processing: {processing_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display detection count
        cv2.putText(frame, f"Faces: {len(detections)}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display status
        embedding_status = "Enabled" if self.embedding_executor else "Disabled"
        cv2.putText(frame, f"Embedding: {embedding_status}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def handle_user_input(self):
        """Handle interactive renaming"""
        print("\nVisible ReIDs:")
        if not self.visible_reids:
            print("  No visible ReIDs")
            return
            
        for rk in self.visible_reids:
            print(f"  {rk} --> {self.db_manager.reid_name_map.get(rk, 'unknown')}")
        
        try:
            sel = input("Enter ReID to assign name (e.g., reid_3): ").strip()
            if sel in self.visible_reids:
                new_name = input(f"Enter new name for {sel}: ").strip()
                if new_name:
                    if self.db_manager.update_name(sel, new_name):
                        print(f"[INFO] Updated {sel} to name: {new_name}")
                    else:
                        print(f"[ERROR] Failed to update {sel}")
                else:
                    print("[WARN] Name cannot be empty")
            else:
                print(f"[WARN] {sel} is not currently visible!")
        except KeyboardInterrupt:
            print("\n[INFO] Input cancelled")
        except Exception as e:
            print(f"[ERROR] Input error: {e}")
    
    def run(self):
        """Main execution loop"""
        # Start threads
        detection_thread = threading.Thread(target=self.detection_thread, daemon=True)
        embedding_thread = threading.Thread(target=self.embedding_thread, daemon=True)
        
        detection_thread.start()
        embedding_thread.start()
        
        logger.info("Face recognition system started - optimized for performance")
        print("\nControls:")
        print("  ESC/Q - Quit")
        print("  S - Interactive renaming")
        print("  I - Performance info")
        
        try:
            while self.running:
                # Render and display frame (no YOLO inference here)
                frame = self.render_frame()
                if frame is not None:
                    cv2.imshow("Multi-threaded Face Recognition", frame)
                else:
                    # Display loading screen
                    loading_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(loading_frame, "Loading...", (250, 240),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    cv2.imshow("Multi-threaded Face Recognition", loading_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC
                    break
                elif key == ord('s') or key == ord('S'):
                    # Interactive renaming
                    self.handle_user_input()
                elif key == ord('q') or key == ord('Q'):
                    break
                elif key == ord('i') or key == ord('I'):
                    # Print performance info
                    print(f"\nPerformance Info:")
                    print(f"Processing tracks: {len(self.processing_tracks)}")
                    print(f"Known tracks: {len(self.track_id_to_reid)}")
                    print(f"Total embeddings: {len(self.track_id_to_embedding)}")
                    print(f"Current FPS: {self.fps:.1f}")
                    print(f"Camera status: {'OK' if self.cap and self.cap.isOpened() else 'ERROR'}")
                    print(f"Database status: {'OK' if self.db_manager.face_db else 'ERROR'}")
                    print(f"Embedding worker: {'OK' if self.embedding_executor else 'DISABLED'}")
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up...")
        self.running = False
        
        # Close video capture
        if self.cap:
            self.cap.release()
        
        # Close windows
        cv2.destroyAllWindows()
        
        # Shutdown thread pools
        if self.embedding_executor:
            self.embedding_executor.shutdown(wait=True)
        if self.db_executor:
            self.db_executor.shutdown(wait=True)
        
        logger.info("Cleanup complete")

def main():
    """Main function"""
    try:
        print("Initializing Face Recognition System...")
        system = FaceRecognitionSystem()
        system.run()
    except FileNotFoundError as e:
        logger.error(f"Required file not found: {e}")
        print("Please ensure all required files are present:")
        print("  - model/yolov11l-face.pt (YOLO model)")
        print("  - face_encoding_worker.py (face encoding module)")
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"System error: {e}")
        raise

if __name__ == "__main__":
    main()