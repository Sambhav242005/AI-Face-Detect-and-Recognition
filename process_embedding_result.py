import os
import queue
import time
import logging
import warnings
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import pickle

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Set this before importing chromadb to disable telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"
import chromadb

# ---------------------------------------------------
# Optional external face-encoding worker
# ---------------------------------------------------
try:
    # must return List[float] or None
    from face_encoding import face_encoding_worker
    FACE_ENCODING_AVAILABLE = True
except ImportError:
    print("[WARN] face_encoding_worker.py not found - embeddings disabled")
    FACE_ENCODING_AVAILABLE = False
    face_encoding_worker = None  # type: ignore

# ---------------------------------------------------
# Multiprocessing worker function
# ---------------------------------------------------
def mp_face_encoding_worker(face_crop_data: bytes) -> Optional[List[float]]:
    """
    Multiprocessing-safe face encoding worker.
    Takes serialized image data and returns embedding.
    """
    try:
        # Deserialize the image data
        face_crop = pickle.loads(face_crop_data)
        
        # Import and use the face encoding function
        if FACE_ENCODING_AVAILABLE and face_encoding_worker:
            return face_encoding_worker(face_crop)
        return None
    except Exception as e:
        print(f"MP face encoding error: {e}")
        return None

# ---------------------------------------------------
# Logging & warnings
# ---------------------------------------------------
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------
# Dataclasses
# ---------------------------------------------------
@dataclass
class FaceDetection:
    """Represents a detected face before embedding."""
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float
    track_id: int
    face_crop: np.ndarray
    timestamp: float

@dataclass
class FaceEmbedding:
    """Represents a face after embedding and identification."""
    track_id: int
    embedding: List[float]
    reid_num: Optional[int] = None
    name: Optional[str] = None

def is_blurry(image: np.ndarray, threshold: float = 100.0) -> bool:
    """Returns True if image is blurry based on Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var < threshold
# ---------------------------------------------------
# Face-crop saver with multiprocessing support
# ---------------------------------------------------
class FaceSaver:
    """Saves cropped face images to disk with multiprocessing support."""
    def __init__(self, root: str = "./face_crops", use_multiprocessing: bool = True) -> None:
        self.root = root
        self.use_multiprocessing = use_multiprocessing
        os.makedirs(self.root, exist_ok=True)
        self.lock = threading.Lock()
        
        if use_multiprocessing:
            self.save_pool = ProcessPoolExecutor(max_workers=2)
        else:
            self.save_pool = ThreadPoolExecutor(max_workers=2)
            

    def save(self, crop: np.ndarray, reid_num: int) -> None:
        """Save BGR image to a date-stamped folder."""
        if self.use_multiprocessing:
            # Serialize the image for multiprocessing
            crop_data = pickle.dumps(crop)
            self.save_pool.submit(self._mp_save_worker, crop_data, reid_num, self.root)
        else:
            self.save_pool.submit(self._thread_save_worker, crop, reid_num)

    def _thread_save_worker(self, crop: np.ndarray, reid_num: int) -> None:
        """Thread-based save worker."""
        date_folder = time.strftime("%Y-%m-%d")
        dir_path = os.path.join(self.root, date_folder)
        try:
            os.makedirs(dir_path, exist_ok=True)
            filename = f"img_reid_{reid_num}.jpg"
            path = os.path.join(dir_path, filename)
            cv2.imwrite(path, crop)
        except Exception as e:
            logger.error(f"Failed saving face crop: {e}")

    @staticmethod
    def _mp_save_worker(crop_data: bytes, reid_num: int, root: str) -> None:
        """Multiprocessing-safe save worker."""
        try:
            crop = pickle.loads(crop_data)
            dir_path = os.path.join(root)
            os.makedirs(dir_path, exist_ok=True)
            filename = f"img_reid_{reid_num}.jpg"
            path = os.path.join(dir_path, filename)
            cv2.imwrite(path, crop)
        except Exception as e:
            print(f"MP save error: {e}")

    def cleanup(self) -> None:
        """Shutdown the executor."""
        if hasattr(self, 'save_pool'):
            self.save_pool.shutdown(wait=True)

# ---------------------------------------------------
# ChromaDB wrapper
# ---------------------------------------------------
class DatabaseManager:
    """Manages the face embedding database using ChromaDB."""
    def __init__(self, db_path: str = "./face_data_db"):
        self.lock = threading.Lock()
        self.reid_name_map: Dict[str, str] = {}
        try:
            os.makedirs(db_path, exist_ok=True)
            self.client = chromadb.PersistentClient(path=db_path)
            self.face_db = self.client.get_or_create_collection("face_db")
            self._load_existing()
            logger.info("ChromaDB persistent store ready")
        except Exception as e:
            logger.error(f"Persistent DB failed ({e}) – falling back to memory")
            self.client = chromadb.Client()
            self.face_db = self.client.create_collection("face_db")

    def _load_existing(self) -> None:
        """Loads existing records from the database into the name map."""
        try:
            rows = self.face_db.get(include=["metadatas"])
            metadatas_list = rows.get("metadatas") or []
            for idx, meta in zip(rows.get("ids", []), metadatas_list):
                if meta:
                    self.reid_name_map[idx] = str(meta.get("name", "unknown"))
        except Exception as e:
            logger.error(f"Could not load existing DB data: {e}")
            
    def get_by_reid(self, reid_key: str) -> Optional[Dict[str, Any]]:
        """
        Fetches a face record from the database by its ReID key (e.g., "reid_1").
        Returns None if not found.
        """
        with self.lock:
            try:
                if self.face_db.count() == 0:
                    return None
                result = self.face_db.get(ids=[reid_key], include=["embeddings", "metadatas"])
                if not result.get("ids"):
                    return None
                return {
                    "id": result["ids"][0],
                    "embedding": result["embeddings"][0], # type: ignore
                    "metadata": result["metadatas"][0] # type: ignore
                }
            except Exception as e:
                logger.error(f"Failed to get by reid '{reid_key}': {e}")
                return None


    def query(self, embedding: List[float], thr: float = 0.4) -> Tuple[Optional[int], Optional[str]]:
        """Queries the database to find the closest matching face."""
        with self.lock:
            try:
                start_time = time.time()
                if self.face_db.count() == 0:
                    return None, None
                
                qr = self.face_db.query(query_embeddings=[embedding], n_results=1)
                
                if time.time() - start_time > 1.0:
                    logger.warning(f"Slow DB query: {time.time() - start_time:.2f}s")

                ids = qr.get("ids", [[]])[0]
                dists = qr.get("distances", [[]])

                if ids and dists and dists[0][0] < thr:
                    key = ids[0]
                    reid_num = int(key.split("_")[1])
                    return reid_num, self.reid_name_map.get(key, f"unknown_{reid_num}")
            except Exception as e:
                logger.error(f"DB query error: {e}")
        return None, None

    def add(self, embedding: List[float], reid_num: int, name: str) -> bool:
        """Adds a new face embedding to the database."""
        key = f"reid_{reid_num}"
        with self.lock:
            try:
                self.face_db.add(
                    ids=[key],
                    embeddings=[embedding],
                    metadatas=[{"name": name}]
                )
                self.reid_name_map[key] = name
                return True
            except Exception as e:
                logger.error(f"DB add error: {e}")
                return False

    def update_name(self, key: str, new_name: str) -> bool:
        """Updates the name for an existing ReID."""
        with self.lock:
            try:
                self.face_db.update(ids=[key], metadatas=[{"name": new_name}])
                self.reid_name_map[key] = new_name
                return True
            except Exception as e:
                logger.error(f"DB rename error: {e}")
                return False

    def next_reid_num(self) -> int:
        """Calculates the next available ReID number safely."""
        with self.lock:
            if not self.reid_name_map:
                return 1
    
            nums = []
            for key in self.reid_name_map.keys():
                if "_" in key:
                    try:
                        # Safely attempt to parse the number after the first underscore
                        num = int(key.split("_")[1])
                        nums.append(num)
                    except (ValueError, IndexError):
                        # This will catch errors if the part after '_' isn't a valid number
                        # or if there's nothing after the underscore.
                        logger.warning(f"Ignoring malformed ReID key during next_reid_num calculation: '{key}'")
                        continue
                    
            # Return the next highest number, or 1 if no valid keys were found
            return max(nums + [0]) + 1

# ---------------------------------------------------
# Main system with multiprocessing support
# ---------------------------------------------------
class FaceRecognitionSystem:
    """Handles video capture, face detection, tracking, and recognition with multiprocessing."""
    def __init__(
        self,
        model_path: str = "model/yolov11l-face.pt",
        camera_id: int = 0,
        use_camera: bool = True,
        use_multiprocessing: bool = True,
        max_processes: int = 2
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model missing: {model_path}")

        # Multiprocessing configuration
        self.use_multiprocessing = use_multiprocessing and FACE_ENCODING_AVAILABLE
        if max_processes is None:
            max_processes = max(1, mp.cpu_count() - 1)  # Leave one CPU free
        self.max_processes = max_processes
        
        logger.info(f"Multiprocessing: {'enabled' if self.use_multiprocessing else 'disabled'}")
        logger.info(f"Max processes: {self.max_processes}")

        # Load YOLO model
        self.model = YOLO(model_path)
        if torch.cuda.is_available():
            self.model.to("cuda:0")
            logger.info("YOLO on CUDA")
        else:
            logger.info("YOLO on CPU")

        # Camera setup
        self.use_camera = use_camera
        self.cap = None
        if use_camera:
            self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                raise RuntimeError("Could not open camera")

        # State management
        self.running = True
        self.frame_lock = threading.Lock()
        self.detection_lock = threading.Lock()
        self.current_frame: Optional[np.ndarray] = None
        self.current_detections: List[Dict] = []
        self.processing_tracks: set[int] = set()
        self.reid_to_face_crop: Dict[str, np.ndarray] = {}

        # Queues and threads
        self.detect_to_embed: queue.Queue[List[FaceDetection]] = queue.Queue(maxsize=50)
        self.threads_started = False
        self.detection_thread_obj: Optional[threading.Thread] = None
        self.embedding_thread_obj: Optional[threading.Thread] = None

        # Data maps
        self.track_to_embedding: Dict[int, List[float]] = {}
        self.track_to_reid: Dict[int, int] = {}
        self.visible_reids: List[str] = []

        # Services
        self.db = DatabaseManager()
        self.saver = FaceSaver(use_multiprocessing=self.use_multiprocessing)
        
        # --- FIX ---
        # Initialize pools to None. They will be created in start_threads().
        self.embed_pool = None
        self.db_pool = ThreadPoolExecutor(max_workers=2)

        # FPS calculation
        self.prev_time = time.time()
        self.frame_cnt = 0
        self.fps = 0.0

        # Performance monitoring
        self.embedding_times: List[float] = []
        self.last_perf_report = time.time()

    def start_threads(self) -> None:
        """Starts the main processing threads if they are not already running."""
        if self.threads_started:
            return
        
        # --- FIX ---
        # Create the executor pool here to avoid multiprocessing issues on startup.
        if self.embed_pool is None and FACE_ENCODING_AVAILABLE:
            if self.use_multiprocessing:
                self.embed_pool = ProcessPoolExecutor(max_workers=self.max_processes)
                logger.info(f"Using ProcessPoolExecutor with {self.max_processes} workers")
            else:
                self.embed_pool = ThreadPoolExecutor(max_workers=4)
                logger.info("Using ThreadPoolExecutor with 4 workers")

        if self.use_camera:
            self.detection_thread_obj = threading.Thread(target=self.detection_thread, daemon=True)
            self.detection_thread_obj.start()

        if FACE_ENCODING_AVAILABLE:
            self.embedding_thread_obj = threading.Thread(target=self.embedding_thread, daemon=True)
            self.embedding_thread_obj.start()

        self.threads_started = True

    def get_face_gallery_data(self) -> Dict[str, Dict]:
        """Returns all stored faces from the database, not just visible ones."""
        gallery_data = {}

        for reid_key, name in self.db.reid_name_map.items():
            face_image = self.reid_to_face_crop.get(reid_key)

            # Optionally load face crop from disk if not in memory

            gallery_data[reid_key] = {
                'name': name,
                'face_image': face_image,
                'last_seen': None  # Unknown since we're pulling historical data
            }

        return gallery_data

    def detection_thread(self) -> None:
        """Thread for continuous frame capture and face detection from a camera."""
        if not self.cap:
            logger.error("Detection thread cannot run without a camera.")
            return

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            self._process_frame_logic(frame)
            time.sleep(0.03)  # Aim for ~30 FPS

    def process_single_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Processes a single provided frame and returns the rendered output."""
        if frame is None:
            return None
        
        # This will do detection and queue unknowns for embedding
        self._process_frame_logic(frame) 
        
        # This will render the most recent state onto the frame
        return self.render_frame(frame)

    def _process_frame_logic(self, frame: np.ndarray) -> None:
        """Core logic to process a frame for detections and queue embeddings."""
        try:
            results = self.model.track(
                frame,
                stream=True,
                persist=True,
                tracker="botsort.yaml",
                verbose=False
            )
        except Exception as e:
            logger.error(f"YOLO track failed: {e}")
            return

        detections: List[Dict] = []
        embeds_to_queue: List[FaceDetection] = []

        for r in results:
            for box in getattr(r, "boxes", []):
                coords = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, coords)
                conf = float(box.conf[0])
                
                if conf < 0.5 or x1 >= x2 or y1 >= y2:
                    continue

                # Fallback for untracked boxes
                tid = int(box.id[0]) if getattr(box, "id", None) is not None else \
                      abs(hash(f"{x1}_{y1}_{x2}_{y2}")) % 100000

                detection = {
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "conf": conf, "track_id": tid
                }
                detections.append(detection)

                # Queue for embedding if track is new and not already being processed
                if (FACE_ENCODING_AVAILABLE and
                        tid not in self.track_to_reid and
                        tid not in self.processing_tracks):
                    
                    crop = frame[max(0, y1):y2, max(0, x1):x2]
                    if crop.size > 0:
                        embeds_to_queue.append(FaceDetection(
                            x1, y1, x2, y2, conf, tid, crop.copy(), time.time()
                        ))
                        self.processing_tracks.add(tid)

        with self.frame_lock:
            self.current_frame = frame.copy()
        with self.detection_lock:
            self.current_detections = detections
        
        if embeds_to_queue:
            try:
                self.detect_to_embed.put_nowait(embeds_to_queue)
            except queue.Full:
                # If queue is full, drop the batch to avoid lag
                for det in embeds_to_queue:
                    self.processing_tracks.discard(det.track_id)
                pass

    def embedding_thread(self) -> None:
        """Thread to process face embeddings from the queue."""
        if not (FACE_ENCODING_AVAILABLE and self.embed_pool):
            logger.warning("Embedding disabled or pool not initialized – thread exiting")
            return

        while self.running:
            try:
                embed_batch = self.detect_to_embed.get(timeout=1.0)
            except queue.Empty:
                continue

            for det in embed_batch:
                tid = det.track_id
                if tid not in self.processing_tracks:
                    continue
                try:
                    start_time = time.time()
                    
                    if self.use_multiprocessing:
                        # Serialize the face crop for multiprocessing
                        crop_data = pickle.dumps(det.face_crop)
                        future = self.embed_pool.submit(mp_face_encoding_worker, crop_data)
                    else:
                        # Use the original threading approach
                        if face_encoding_worker:
                            future = self.embed_pool.submit(face_encoding_worker, det.face_crop)
                        else:
                            continue
                    
                    self.db_pool.submit(self._handle_embedding_result, future, det, start_time)
                    
                except Exception as e:
                    logger.error(f"Failed to submit embedding task: {e}")
                    self.processing_tracks.discard(tid)

    def _handle_embedding_result(self, future, det: FaceDetection, start_time: float) -> None:
        """Callback to handle the result from the face encoding worker."""
        tid = det.track_id
        try:
            emb = future.result(timeout=10.0)  # Increased timeout for multiprocessing
            if emb is None:
                return

            # Record performance metrics
            embedding_time = time.time() - start_time
            self.embedding_times.append(embedding_time)
            
            # Keep only recent times for averaging
            if len(self.embedding_times) > 100:
                self.embedding_times = self.embedding_times[-50:]
            
            # Periodic performance reporting
            if time.time() - self.last_perf_report > 30:  # Every 30 seconds
                if self.embedding_times:
                    avg_time = sum(self.embedding_times) / len(self.embedding_times)
                    logger.info(f"Average embedding time: {avg_time:.3f}s (MP: {self.use_multiprocessing})")
                self.last_perf_report = time.time()

            self._process_embedding_async(emb, det)

        except Exception as e:
            logger.error(f"Embedding result error for track {tid}: {e}")
        finally:
            self.processing_tracks.discard(tid)

    def _process_embedding_async(self, emb: List[float], det: FaceDetection) -> None:
        """Processes a new embedding: queries DB, adds if new, and saves crop."""
        tid = det.track_id
        self.track_to_embedding[tid] = emb
    
        face_crop = det.face_crop
        if is_blurry(face_crop):
            print(f"[BLUR] Rejected blurry face for TID {tid}")
            return  # Skip processing blurry images
    
        reid, name = self.db.query(emb)
        if reid is None:  # New face
            reid = self.db.next_reid_num()
            name = f"unknown_{reid}"
            self.db.add(emb, reid, name)
    
        self.track_to_reid[tid] = reid
    
        reid_key = f"reid_{reid}"
        if reid_key not in self.reid_to_face_crop:
            self.reid_to_face_crop[reid_key] = face_crop
    
        # Save a reference crop image
        self.saver.save(face_crop, reid)

    def render_frame(self, frame_to_render: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Renders detections and info onto a frame."""
        with self.frame_lock:
            if frame_to_render is None:
                if self.current_frame is None:
                    return None
                frame_to_render = self.current_frame.copy()

        with self.detection_lock:
            dets = list(self.current_detections)

        frame = frame_to_render
        self.visible_reids.clear()

        for d in dets:
            x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
            tid = d["track_id"]
            
            # Determine label and color
            if tid in self.track_to_reid:
                reid = self.track_to_reid[tid]
                key = f"reid_{reid}"
                self.visible_reids.append(key)
                name = self.db.reid_name_map.get(key, key)
                label = f"{name} ({reid})"
                color = (0, 255, 0)  # Green for recognized
            elif tid in self.processing_tracks:
                label = "Processing..."
                color = (0, 165, 255)  # Orange for processing
            else:
                label = "Unknown"
                color = (0, 0, 255)  # Red for unknown

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Calculate and display FPS
        self.frame_cnt += 1
        now = time.time()
        if now - self.prev_time >= 1.0:
            self.fps = self.frame_cnt / (now - self.prev_time)
            self.prev_time = now
            self.frame_cnt = 0
        
        # Enhanced status display
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        
        mode_text = f"MP: {'ON' if self.use_multiprocessing else 'OFF'}"
        cv2.putText(frame, mode_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        processing_count = len(self.processing_tracks)
        if processing_count > 0:
            cv2.putText(frame, f"Processing: {processing_count}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        return frame
    
    def rename_last_unknown(self, new_name: str) -> str:
        """
        Renames the most recently created, visible 'unknown' face.
        This version is GUI-friendly and does not use console input.
        """
        # --- 1. Basic validation ---
        if not self.visible_reids:
            return "No faces are currently visible to rename."
        if not new_name or not new_name.strip():
            return "Name cannot be empty."
        
        # --- 2. Find the target ReID to rename ---
        # Correctly identify unknown faces by checking the prefix of their assigned name.
        unknown_reids = [
            rk for rk in self.visible_reids
            if self.db.reid_name_map.get(rk, '').startswith('unknown_')
        ]
    
        if not unknown_reids:
            return "No 'unknown' faces are visible to rename. All visible faces already have names."
    
        # --- 3. Identify the 'last' one by the highest number ---
        try:
            target_reid = max(
                unknown_reids,
                key=lambda rk: int(rk.split('_')[1])
            )
        except (IndexError, ValueError):
            return "Could not determine the latest unknown face."
    
        # --- 4. Perform the update and return the result ---
        if self.db.update_name(target_reid, new_name.strip()):
            return f"✅ Success: Renamed {target_reid} to '{new_name}'."
        else:
            return f"❌ Update failed for {target_reid}."

    def interactive_rename(self) -> None:
        """Allows renaming a visible ReID via the console."""
        if not self.visible_reids:
            print("\nNo visible ReIDs to rename.")
            return
        
        print("\n--- Visible ReIDs ---")
        for rk in sorted(list(set(self.visible_reids))):
            name = self.db.reid_name_map.get(rk, 'unknown')
            print(f"  {rk} -> {name}")
        print("--------------------")
        
        try:
            sel = input("Enter ReID to rename (e.g., reid_1): ").strip()
            if not sel or sel not in self.db.reid_name_map:
                print("Invalid or not a known ReID. Aborting.")
                return
            
            new_name = input(f"Enter new name for {sel}: ").strip()
            if not new_name:
                print("Name cannot be empty. Aborting.")
                return
            
            if self.db.update_name(sel, new_name):
                print(f"Success: {sel} renamed to {new_name}")
            else:
                print("Update failed.")
        except (KeyboardInterrupt, EOFError):
            print("\nRename cancelled.")

    def run(self):
        """Main loop for camera-based processing."""
        if not self.use_camera:
            logger.error("Run method is for camera mode. Use process_single_frame for other sources.")
            return

        self.start_threads()
        print("\n--- Live Face Recognition with Multiprocessing ---")
        print("Press 's' to rename a person.")
        print("Press 'm' to toggle multiprocessing mode.")
        print("Press 'p' to show performance stats.")
        print("Press 'q' or ESC to quit.")
        
        try:
            while self.running:
                frame = self.render_frame()
                if frame is not None:
                    cv2.imshow("Live Face Recognition", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q'), ord('Q')):
                    break
                elif key in (ord('s'), ord('S')):
                    self.interactive_rename()
                elif key in (ord('m'), ord('M')):
                    self._toggle_multiprocessing()
                elif key in (ord('p'), ord('P')):
                    self._show_performance_stats()
        finally:
            self.cleanup()

    def _toggle_multiprocessing(self) -> None:
        """Toggle between multiprocessing and threading modes."""
        if not FACE_ENCODING_AVAILABLE:
            print("Cannot toggle: face encoding not available")
            return
            
        print("\nToggling processing mode...")
        
        # Shutdown current pool
        if self.embed_pool:
            self.embed_pool.shutdown(wait=True)
        
        # Toggle mode
        self.use_multiprocessing = not self.use_multiprocessing
        
        # Create new pool
        if self.use_multiprocessing:
            self.embed_pool = ProcessPoolExecutor(max_workers=self.max_processes)
            print(f"Switched to multiprocessing with {self.max_processes} workers")
        else:
            self.embed_pool = ThreadPoolExecutor(max_workers=4)
            print("Switched to threading with 4 workers")

    def _show_performance_stats(self) -> None:
        """Display performance statistics."""
        if not self.embedding_times:
            print("\nNo embedding performance data available yet.")
            return
            
        avg_time = sum(self.embedding_times) / len(self.embedding_times)
        min_time = min(self.embedding_times)
        max_time = max(self.embedding_times)
        
        print(f"\n--- Performance Stats ---")
        print(f"Multiprocessing: {'ON' if self.use_multiprocessing else 'OFF'}")
        print(f"Workers: {self.max_processes if self.use_multiprocessing else 4}")
        print(f"Avg embedding time: {avg_time:.3f}s")
        print(f"Min embedding time: {min_time:.3f}s")
        print(f"Max embedding time: {max_time:.3f}s")
        print(f"Total embeddings: {len(self.embedding_times)}")
        print(f"Currently processing: {len(self.processing_tracks)}")
        print("------------------------")

    def cleanup(self) -> None:
        """Shuts down threads, releases resources, and closes windows."""
        print("Cleaning up resources...")
        self.running = False
        
        if self.detection_thread_obj and self.detection_thread_obj.is_alive():
            self.detection_thread_obj.join(timeout=2)
        if self.embedding_thread_obj and self.embedding_thread_obj.is_alive():
            self.embedding_thread_obj.join(timeout=2)
        
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        if self.embed_pool:
            self.embed_pool.shutdown(wait=True)
        if self.db_pool:
            self.db_pool.shutdown(wait=True)
        self.saver.cleanup()
        logger.info("Clean exit")
