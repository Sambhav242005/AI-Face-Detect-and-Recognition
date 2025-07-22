import os
import traceback
import cv2
import time
from fastrtc import Stream
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
import gradio as gr

from db import DatabaseManager
from face_encoding_worker import face_encoding_worker  # make sure this is a function

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

class FaceRecognitionSystem:
    """Main face recognition system with multi-threading"""
    
    def __init__(self, model_path='model/yolov11l-face.pt'):
        # Load model with error handling
        try:
            self.model = YOLO(model_path)
            self.model.fuse()
            # Try to move to CUDA if available
            if torch.cuda.is_available():
                self.model.to('cuda:0')
                self.model.half()
                logger.info("Model loaded on CUDA")
            else:
                logger.info("Model loaded on CPU")
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            # Try loading without CUDA
            self.model = YOLO(model_path)
            logger.info("Model loaded on CPU (fallback)")
        
        self.db_manager = DatabaseManager()
        
        # Threading components
        self.detection_queue = queue.Queue(maxsize=10)
        self.embedding_queue = queue.Queue(maxsize=20)
        self.result_queue = queue.Queue(maxsize=20)
        
        # Tracking data
        self.track_id_to_embedding = {}
        self.track_id_to_reid = {}
        self.processing_tracks = set()  # Tracks currently being processed
        
        # Threading pools - Use ThreadPoolExecutor instead of ProcessPoolExecutor
        self.embedding_executor = ThreadPoolExecutor(max_workers=2)
        self.db_executor = ThreadPoolExecutor(max_workers=1)
        
        # Control flags
        self.running = True
        self.detection_lock = threading.Lock()
        self.current_detections = []
        self.visible_reids = []
        
        # FPS tracking
        self.prev_time = time.time()
        self.frame_count = 0
        
        # Start background threads
        self.start_background_threads()
    
    def start_background_threads(self):
        """Start background processing threads"""
        self.embedding_thread = threading.Thread(target=self._embedding_worker, daemon=True)
        self.embedding_thread.start()
        logger.info("Background threads started")
    
    def _safe_face_encoding(self, face_crop):
        """Safe wrapper for face encoding that handles errors properly"""
        try:
            return face_encoding_worker(face_crop)
        except Exception as e:
            import traceback
            logger.error(f"Error in face encoding: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _embedding_worker(self):
        """Background thread for processing face embeddings"""
        logger.info("Embedding worker started")
        
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
                    
                    # Submit face encoding task (use threading instead of multiprocessing)
                    future = self.embedding_executor.submit(
                        self._safe_face_encoding, detection.face_crop
                    )
                    
                    # Submit database query task
                    self.db_executor.submit(
                        self._process_embedding_result, 
                        future, detection
                    )
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in embedding worker: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _process_embedding_result(self, future, detection):
        """Process the result of face encoding"""
        try:
            embedding = future.result(timeout=5.0)
            track_id = detection.track_id
            
            if embedding is None:
                logger.warning(f"No embedding returned for track {track_id}")
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
                
                # Save face image
                img_save_dir = "saved_faces"
                os.makedirs(img_save_dir, exist_ok=True)
                img_filename = f"{img_save_dir}/reid_{new_reid_num}.jpg"
                cv2.imwrite(img_filename, detection.face_crop)
                
                if self.db_manager.face_db.add(
                    ids=[f"reid_{new_reid_num}"],
                    embeddings=[embedding],
                    metadatas=[{"name": new_name, "image_path": img_filename}]
                ):
                    self.db_manager.reid_name_map[f"reid_{new_reid_num}"] = new_name
                    self.track_id_to_reid[track_id] = new_reid_num
                    logger.info(f"Added new person: {new_name} with image saved at {img_filename}")
                else:
                    logger.error(f"Failed to add new person for track {track_id}")
            
            # Remove from processing set
            self.processing_tracks.discard(track_id)
            
        except Exception as e:
            import traceback
            logger.error(f"Error processing embedding result for track {detection.track_id}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.processing_tracks.discard(detection.track_id)
    
    def process_frame(self, frame):
        """Process a single frame and return the annotated result"""
        if frame is None:
            return None
        
        try:
            # Run YOLO detection
            results = self.model.track(frame, stream=True, persist=True, 
                                    tracker="botsort.yaml", verbose=False)
        except Exception as e:
            import traceback
            logger.error(f"YOLO tracking error: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Fallback to prediction without tracking
            try:
                results = self.model.predict(frame, verbose=False)
            except Exception as e2:
                import traceback
                logger.error(f"YOLO prediction error: {e2}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                return frame
        
        detections = []
        embedding_candidates = []
        
        for result in results:
            for box in (result.boxes or []):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                # Handle tracking ID safely
                if hasattr(box, 'id') and box.id is not None:
                    track_id = int(box.id[0])
                else:
                    # Generate a temporary ID based on box position
                    track_id = hash(f"{x1}_{y1}_{x2}_{y2}") % 10000
                
                if conf > 0.5:
                    detection = {
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'conf': conf, 'track_id': track_id
                    }
                    detections.append(detection)
                    
                    # Check if we need to process this track for embedding
                    if (track_id not in self.track_id_to_embedding and 
                        track_id not in self.processing_tracks):
                        
                        face_crop = frame[y1:y2, x1:x2]
                        if face_crop.size > 0:  # Ensure valid crop
                            embedding_candidates.append(FaceDetection(
                                x1=x1, y1=y1, x2=x2, y2=y2,
                                conf=conf, track_id=track_id,
                                face_crop=face_crop,
                                frame_timestamp=time.time()
                            ))
        
        # Store detections
        with self.detection_lock:
            self.current_detections = detections
        
        # Send new faces for embedding processing
        if embedding_candidates:
            try:
                self.detection_queue.put(embedding_candidates, timeout=0.01)
            except queue.Full:
                pass  # Skip if queue is full
        
        # Render the frame with annotations
        return self.render_frame(frame, detections)
    
    def render_frame(self, frame, detections):
        """Render frame with face recognition results"""
        if frame is None:
            return None
        
        frame_copy = frame.copy()
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
                label = "Unknown"
                color = (0, 0, 255)  # Red for unknown
            
            # Draw bounding box and label
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_copy, f"{label} | ID {track_id} | {conf:.2f}",
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Display FPS
        curr_time = time.time()
        self.frame_count += 1
        if curr_time - self.prev_time >= 1.0:  # Update FPS every second
            fps = self.frame_count / (curr_time - self.prev_time)
            self.prev_time = curr_time
            self.frame_count = 0
        else:
            fps = self.frame_count / (curr_time - self.prev_time) if curr_time > self.prev_time else 0
        
        cv2.putText(frame_copy, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        
        # Display processing info
        processing_count = len(self.processing_tracks)
        cv2.putText(frame_copy, f"Processing: {processing_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display detection count
        cv2.putText(frame_copy, f"Faces: {len(detections)}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame_copy
    
    def handle_user_input(self):
        """Handle interactive renaming"""
        print("\nVisible ReIDs:")
        for rk in self.visible_reids:
            print(f"  {rk} --> {self.db_manager.reid_name_map.get(rk)}")
        
        sel = input("Enter ReID to assign name (e.g., reid_3): ").strip()
        if sel in self.visible_reids:
            new_name = input(f"Enter new name for {sel}: ").strip()
            if new_name:
                if self.db_manager.update_name(sel, new_name):
                    print(f"[INFO] Updated {sel} to name: {new_name}")
                else:
                    print(f"[ERROR] Failed to update {sel}")
        else:
            print(f"[WARN] {sel} is not currently visible!")
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up...")
        self.running = False
        
        # Shutdown thread pools
        self.embedding_executor.shutdown(wait=True)
        self.db_executor.shutdown(wait=True)
        
        logger.info("Cleanup complete")

# Global system instance with proper initialization
_face_recognition_system = None
_system_lock = threading.Lock()

def get_face_recognition_system():
    """Get or create the face recognition system (thread-safe singleton)"""
    global _face_recognition_system
    
    if _face_recognition_system is None:
        with _system_lock:
            if _face_recognition_system is None:
                _face_recognition_system = FaceRecognitionSystem()
                logger.info("Face recognition system initialized")
    
    return _face_recognition_system

def main(frame):
    """Main function for processing frames"""
    try:
        # Get the system instance
        system = get_face_recognition_system()
        
        # Process the frame
        processed_frame = system.process_frame(frame)
        return processed_frame
        
    except Exception as e:
        import traceback
        logger.error(f"Error in main: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return frame  # Return original frame on error

stream = Stream(
    handler=main,
    modality="video",
    mode="send-receive",
)

        
stream.ui.launch()