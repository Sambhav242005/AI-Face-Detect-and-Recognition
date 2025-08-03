import cv2
import time
import numpy as np
from ultralytics import YOLO
import chromadb
import threading
import queue
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import logging
import torch
import warnings
import os
import pickle
import base64

# Import with error handling
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

def encode_face_process(face_crop_data):
    """Standalone function for multiprocessing face encoding"""
    try:
        # Deserialize the face crop
        face_crop = pickle.loads(base64.b64decode(face_crop_data))
        
        # Import here to avoid issues with multiprocessing
        try:
            import face_encoding_worker
            return face_encoding_worker.face_encoding_worker(face_crop)
        except ImportError:
            return None
    except Exception as e:
        logger.error(f"Error in face encoding process: {e}")
        return None

def yolo_detection_process(model_path, camera_id, detection_queue, control_queue):
    """Separate process for YOLO detection"""
    try:
        # Load model in this process
        model = YOLO(model_path)
        if torch.cuda.is_available():
            try:
                model.to('cuda:0')
                logger.info("YOLO model loaded on CUDA in detection process")
            except Exception as e:
                logger.warning(f"Failed to move model to CUDA in detection process: {e}")
        
        # Initialize camera
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            logger.error("Failed to open camera in detection process")
            return
        
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        logger.info("YOLO detection process started")
        
        while True:
            # Check for control signals
            try:
                if not control_queue.empty():
                    cmd = control_queue.get_nowait()
                    if cmd == "STOP":
                        break
            except:
                pass
            
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            try:
                # Run YOLO detection
                results = model.track(frame, stream=True, persist=True, 
                                    tracker="botsort.yaml", verbose=False)
                
                detections = []
                embedding_candidates = []
                
                for result in results:
                    boxes = getattr(result, 'boxes', None)
                    if boxes is None:
                        continue
                        
                    for box in boxes:
                        try:
                            coords = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], 'cpu') else box.xyxy[0]
                            x1, y1, x2, y2 = map(int, coords)
                            conf = float(box.conf[0])
                            
                            if hasattr(box, 'id') and box.id is not None:
                                track_id = int(box.id[0])
                            else:
                                track_id = abs(hash(f"{x1}_{y1}_{x2}_{y2}")) % 100000
                            
                            if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0:
                                continue
                            
                            if conf > 0.5:
                                detection = {
                                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                                    'conf': conf, 'track_id': track_id
                                }
                                detections.append(detection)
                                
                                # Extract face crop for embedding
                                h, w = frame.shape[:2]
                                x1_crop = max(0, x1)
                                y1_crop = max(0, y1)
                                x2_crop = min(w, x2)
                                y2_crop = min(h, y2)
                                
                                if x2_crop > x1_crop and y2_crop > y1_crop:
                                    face_crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]
                                    if face_crop.size > 0:
                                        embedding_candidates.append(FaceDetection(
                                            x1=x1, y1=y1, x2=x2, y2=y2,
                                            conf=conf, track_id=track_id,
                                            face_crop=face_crop.copy(),
                                            frame_timestamp=time.time()
                                        ))
                        except Exception as e:
                            logger.error(f"Error processing detection box: {e}")
                            continue
                
                # Send results to main process
                result_data = {
                    'frame': frame,
                    'detections': detections,
                    'embedding_candidates': embedding_candidates,
                    'timestamp': time.time()
                }
                
                try:
                    detection_queue.put(result_data, timeout=0.01)
                except queue.Full:
                    # Skip frame if queue is full
                    pass
                
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                logger.error(f"Error in YOLO detection: {e}")
                time.sleep(0.1)
    
    except Exception as e:
        logger.error(f"Error in detection process: {e}")
    finally:
        if 'cap' in locals():
            cap.release()
        logger.info("YOLO detection process ended")

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
            os.makedirs(self.db_path, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.db_path)
            self.face_db = self.client.get_or_create_collection("face_db")
            self._load_existing_data()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
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
    """Main face recognition system with multi-processing and multi-threading"""
    
    def __init__(self, model_path='model/yolov11l-face.pt', camera_id=0):
        # Validate model path
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model_path = model_path
        self.camera_id = camera_id
        
        # Initialize database manager
        self.db_manager = DatabaseManager()
        
        # Multiprocessing queues and controls
        self.detection_queue = mp.Queue(maxsize=5)
        self.detection_control_queue = mp.Queue()
        
        # Threading components for embedding processing
        self.embedding_queue = queue.Queue(maxsize=20)
        self.result_queue = queue.Queue(maxsize=20)
        
        # Tracking data
        self.track_id_to_embedding = {}
        self.track_id_to_reid = {}
        self.processing_tracks = set()
        
        # Executors - separate processes for face encoding, threads for DB operations
        max_workers = min(4, mp.cpu_count())
        if FACE_ENCODING_AVAILABLE:
            self.embedding_process_executor = ProcessPoolExecutor(max_workers=max_workers)
        else:
            self.embedding_process_executor = None
            
        # Thread pool for database operations (threads work better for I/O)
        self.db_thread_executor = ThreadPoolExecutor(max_workers=2)
        
        # Additional thread pool for face crop preprocessing
        self.preprocessing_thread_executor = ThreadPoolExecutor(max_workers=2)
        
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
        
        # Performance metrics
        self.detection_fps = 0
        self.embedding_fps = 0
        self.detection_frame_count = 0
        self.embedding_count = 0
        self.detection_prev_time = time.time()
        self.embedding_prev_time = time.time()
    
    def detection_consumer_thread(self):
        """Thread to consume detection results from the YOLO process"""
        logger.info("Detection consumer thread started")
        
        while self.running:
            try:
                result_data = self.detection_queue.get(timeout=1.0)
                
                frame = result_data['frame']
                detections = result_data['detections']
                embedding_candidates = result_data['embedding_candidates']
                
                # Update detection FPS
                self.detection_frame_count += 1
                curr_time = time.time()
                if curr_time - self.detection_prev_time >= 1.0:
                    self.detection_fps = self.detection_frame_count / (curr_time - self.detection_prev_time)
                    self.detection_prev_time = curr_time
                    self.detection_frame_count = 0
                
                # Store frame and detections for rendering
                with self.frame_lock:
                    self.current_frame = frame.copy()
                with self.detection_lock:
                    self.current_detections = detections
                
                # Process embedding candidates
                if embedding_candidates and self.embedding_process_executor:
                    # Use thread pool to preprocess and submit to process pool
                    for candidate in embedding_candidates:
                        if (candidate.track_id not in self.track_id_to_embedding and 
                            candidate.track_id not in self.processing_tracks):
                            
                            self.processing_tracks.add(candidate.track_id)
                            
                            # Submit preprocessing task to thread pool
                            self.preprocessing_thread_executor.submit(
                                self._preprocess_and_encode, candidate
                            )
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in detection consumer thread: {e}")
    
    def _preprocess_and_encode(self, detection):
        """Preprocess face crop and submit to process pool for encoding"""
        try:
            track_id = detection.track_id
            
            # Serialize face crop for multiprocessing
            face_crop_data = base64.b64encode(pickle.dumps(detection.face_crop)).decode()
            
            # Submit to process pool
            future = self.embedding_process_executor.submit(encode_face_process, face_crop_data)
            
            # Submit result processing to thread pool (for database operations)
            self.db_thread_executor.submit(self._process_embedding_result, future, detection)
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            self.processing_tracks.discard(detection.track_id)
    
    def _process_embedding_result(self, future, detection):
        """Process the result of face encoding (runs in thread)"""
        track_id = detection.track_id
        
        try:
            embedding = future.result(timeout=15.0)
            
            # Update embedding FPS
            self.embedding_count += 1
            curr_time = time.time()
            if curr_time - self.embedding_prev_time >= 1.0:
                self.embedding_fps = self.embedding_count / (curr_time - self.embedding_prev_time)
                self.embedding_prev_time = curr_time
                self.embedding_count = 0
            
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
        """Render frame with face recognition results"""
        with self.frame_lock:
            if self.current_frame is None:
                return None
            frame = self.current_frame.copy()
        
        with self.detection_lock:
            detections = self.current_detections.copy()
        
        self.visible_reids = []
        
        # Process detections
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
                if self.embedding_process_executor is None:
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
            
            label_y = max(y1 - 10, 20)
            cv2.putText(frame, f"{label} | ID {track_id} | {conf:.2f}",
                       (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Display comprehensive FPS info
        curr_time = time.time()
        self.frame_count += 1
        if curr_time - self.prev_time >= 1.0:
            self.fps = self.frame_count / (curr_time - self.prev_time)
            self.prev_time = curr_time
            self.frame_count = 0
        
        # Display performance metrics
        cv2.putText(frame, f"Render FPS: {self.fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"Detection FPS: {self.detection_fps:.1f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"Embedding FPS: {self.embedding_fps:.1f}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Display processing info
        processing_count = len(self.processing_tracks)
        cv2.putText(frame, f"Processing: {processing_count}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display detection count
        cv2.putText(frame, f"Faces: {len(detections)}", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display system status
        embedding_status = "MP Enabled" if self.embedding_process_executor else "Disabled"
        cv2.putText(frame, f"Embedding: {embedding_status}", (10, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display queue sizes
        try:
            detection_queue_size = self.detection_queue.qsize()
            cv2.putText(frame, f"Det Queue: {detection_queue_size}", (10, 210),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        except:
            pass
        
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
        """Main execution loop with multiprocessing and multithreading"""
        # Start YOLO detection process
        detection_process = mp.Process(
            target=yolo_detection_process,
            args=(self.model_path, self.camera_id, self.detection_queue, self.detection_control_queue),
            daemon=True
        )
        detection_process.start()
        
        # Start detection consumer thread
        detection_consumer = threading.Thread(target=self.detection_consumer_thread, daemon=True)
        detection_consumer.start()
        
        logger.info("Multi-process multi-thread face recognition system started")
        print("\nSystem Architecture:")
        print("  - YOLO Detection: Separate Process")
        print("  - Face Encoding: Process Pool")
        print("  - Database Operations: Thread Pool")
        print("  - Preprocessing: Thread Pool")
        print("  - Main Loop: Single Thread")
        print("\nControls:")
        print("  ESC/Q - Quit")
        print("  S - Interactive renaming")
        print("  I - Performance info")
        
        try:
            while self.running:
                # Render and display frame
                frame = self.render_frame()
                if frame is not None:
                    cv2.imshow("Multi-Process Multi-Thread Face Recognition", frame)
                else:
                    # Display loading screen
                    loading_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(loading_frame, "Initializing...", (200, 240),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    cv2.imshow("Multi-Process Multi-Thread Face Recognition", loading_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC
                    break
                elif key == ord('s') or key == ord('S'):
                    self.handle_user_input()
                elif key == ord('q') or key == ord('Q'):
                    break
                elif key == ord('i') or key == ord('I'):
                    # Print comprehensive performance info
                    print(f"\nPerformance Info:")
                    print(f"Render FPS: {self.fps:.1f}")
                    print(f"Detection FPS: {self.detection_fps:.1f}")
                    print(f"Embedding FPS: {self.embedding_fps:.1f}")
                    print(f"Processing tracks: {len(self.processing_tracks)}")
                    print(f"Known tracks: {len(self.track_id_to_reid)}")
                    print(f"Total embeddings: {len(self.track_id_to_embedding)}")
                    print(f"Detection process alive: {detection_process.is_alive()}")
                    try:
                        print(f"Detection queue size: {self.detection_queue.qsize()}")
                    except:
                        print("Detection queue size: unknown")
                    print(f"Database status: {'OK' if self.db_manager.face_db else 'ERROR'}")
                    print(f"Embedding workers: {'OK' if self.embedding_process_executor else 'DISABLED'}")
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            self.cleanup(detection_process)
    
    def cleanup(self, detection_process=None):
        """Clean up resources"""
        logger.info("Cleaning up...")
        self.running = False
        
        # Stop detection process
        if detection_process and detection_process.is_alive():
            try:
                self.detection_control_queue.put("STOP", timeout=1.0)
                detection_process.join(timeout=5.0)
                if detection_process.is_alive():
                    detection_process.terminate()
                    detection_process.join(timeout=2.0)
                    if detection_process.is_alive():
                        detection_process.kill()
            except Exception as e:
                logger.error(f"Error stopping detection process: {e}")
        
        # Close windows
        cv2.destroyAllWindows()
        
        # Shutdown executors
        if self.embedding_process_executor:
            self.embedding_process_executor.shutdown(wait=True)
        if self.db_thread_executor:
            self.db_thread_executor.shutdown(wait=True)
        if self.preprocessing_thread_executor:
            self.preprocessing_thread_executor.shutdown(wait=True)
        
        logger.info("Cleanup complete")

def main():
    """Main function"""
    # Set multiprocessing start method
    if mp.get_start_method(allow_none=True) != 'spawn':
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass  # Already set
    
    try:
        print("Initializing Multi-Process Multi-Thread Face Recognition System...")
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