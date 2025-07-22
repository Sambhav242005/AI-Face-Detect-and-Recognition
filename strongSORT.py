import os
import traceback
import cv2
import time
from fastrtc import Stream, WebRTC, ReplyOnPause
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
    """Optimized face recognition system with reduced lag"""
    
    def __init__(self, model_path='model/yolov11l-face.pt', target_width=1280, target_height=720):
        # Frame processing settings
        self.target_width = target_width
        self.target_height = target_height
        self.frame_scale_factor = 1.0
        
        # Frame skipping for performance
        self.frame_counter = 0
        self.process_every_n_frames = 2  # Process every 2 frames
        
        # Load model with optimizations
        try:
            self.model = YOLO(model_path)
            self.model.fuse()
            if torch.cuda.is_available():
                self.model.to('cuda:0')
                self.model.half()
                logger.info("Model loaded on CUDA with half precision")
            else:
                logger.info("Model loaded on CPU")
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            self.model = YOLO(model_path)
            logger.info("Model loaded on CPU (fallback)")
        
        self.db_manager = DatabaseManager()
        
        # Optimized threading components with smaller, faster queues
        self.embedding_queue = queue.Queue(maxsize=5)    # Reduced queue size
        self.db_query_queue = queue.Queue(maxsize=10)    # Reduced queue size
        
        # Tracking data with thread-safe access
        self.track_id_to_embedding = {}
        self.track_id_to_reid = {}
        self.processing_tracks = set()
        self.embedding_cache = {}
        
        # Track processing cooldowns to avoid reprocessing
        self.track_last_processed = {}
        self.track_cooldown_time = 5.0  # 5 seconds cooldown
        
        # Thread-safe locks
        self.embedding_lock = threading.Lock()
        self.reid_lock = threading.Lock()
        self.processing_lock = threading.Lock()
        
        # Reduced thread pools for better performance
        self.embedding_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="embedding")
        self.db_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="database")
        
        # Control flags
        self.running = True
        self.detection_lock = threading.Lock()
        self.current_detections = []
        self.visible_reids = []
        
        # FPS tracking
        self.prev_time = time.time()
        self.frame_count = 0
        self.fps = 0
        
        # Performance monitoring
        self.embedding_processing_time = []
        self.detection_processing_time = []
        
        # Cache for recent detections to reduce redundant processing
        self.detection_cache = {}
        self.cache_timeout = 2.0  # Cache timeout in seconds
        
        # Start background threads
        self.start_background_threads()
    
    def start_background_threads(self):
        """Start optimized background processing threads"""
        # Single embedding worker for better control
        self.embedding_thread = threading.Thread(
            target=self._embedding_worker, 
            daemon=True, 
            name="embedding_worker"
        )
        self.embedding_thread.start()
        
        # Single database worker thread
        self.db_thread = threading.Thread(
            target=self._database_worker, 
            daemon=True, 
            name="database_worker"
        )
        self.db_thread.start()
        
        logger.info("Optimized background threads started")
    
    def _preprocess_frame(self, frame):
        """Optimized frame preprocessing"""
        if frame is None:
            return None, 1.0
        
        # Get original dimensions
        original_height, original_width = frame.shape[:2]
        
        # Calculate scaling to fit target resolution
        scale_width = self.target_width / original_width
        scale_height = self.target_height / original_height
        scale_factor = min(scale_width, scale_height)
        
        # Only resize if necessary
        if scale_factor < 0.9 or scale_factor > 1.1:
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        else:
            scale_factor = 1.0
        
        self.frame_scale_factor = scale_factor
        return frame, scale_factor
    
    def _should_process_track(self, track_id):
        """Check if track should be processed based on cooldown"""
        current_time = time.time()
        
        # Check if already processed and within cooldown
        if track_id in self.track_last_processed:
            if current_time - self.track_last_processed[track_id] < self.track_cooldown_time:
                return False
        
        # Check if already known or being processed
        with self.processing_lock:
            if (track_id in self.track_id_to_embedding or 
                track_id in self.processing_tracks or
                track_id in self.track_id_to_reid):
                return False
        
        return True
    
    def _safe_face_encoding(self, face_crop, track_id):
        """Optimized face encoding with quality checks"""
        start_time = time.time()
        try:
            # Quick quality assessment
            if face_crop is None or face_crop.size == 0:
                return None, track_id
            
            if face_crop.shape[0] < 80 or face_crop.shape[1] < 80:  # Increased minimum size
                return None, track_id
            
            # Basic quality checks
            is_good, reason = self._assess_face_quality(face_crop)
            if not is_good:
                logger.debug(f"Poor face quality for track {track_id}: {reason}")
                return None, track_id
            
            # Optimize face crop for encoding
            face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            
            # Resize to optimal size (smaller for speed)
            target_size = 160  # Reduced from 150+ for speed
            if face_crop_rgb.shape[0] != target_size or face_crop_rgb.shape[1] != target_size:
                face_crop_rgb = cv2.resize(face_crop_rgb, (target_size, target_size), 
                                        interpolation=cv2.INTER_LINEAR)  # Faster interpolation
            
            # Get embedding
            embedding = face_encoding_worker(face_crop_rgb)
            
            # Track processing time
            processing_time = time.time() - start_time
            self.embedding_processing_time.append(processing_time)
            if len(self.embedding_processing_time) > 50:  # Reduced history
                self.embedding_processing_time.pop(0)
            
            return embedding, track_id
            
        except Exception as e:
            logger.error(f"Error in face encoding for track {track_id}: {e}")
            return None, track_id
    
    def _embedding_worker(self):
        """Optimized single embedding worker"""
        logger.info("Embedding worker started")
        
        while self.running:
            try:
                # Get detection with shorter timeout
                detection = self.embedding_queue.get(timeout=0.5)
                
                track_id = detection.track_id
                
                # Double-check if still needed
                if not self._should_process_track(track_id):
                    continue
                
                # Mark as processing
                with self.processing_lock:
                    self.processing_tracks.add(track_id)
                
                # Update last processed time
                self.track_last_processed[track_id] = time.time()
                
                # Process face encoding
                embedding, processed_track_id = self._safe_face_encoding(
                    detection.face_crop, track_id
                )
                
                if embedding is not None:
                    # Store embedding
                    with self.embedding_lock:
                        self.track_id_to_embedding[track_id] = embedding
                    
                    # Queue for database processing (with timeout)
                    try:
                        self.db_query_queue.put({
                            'track_id': track_id,
                            'embedding': embedding,
                            'detection': detection
                        }, timeout=0.1)
                    except queue.Full:
                        logger.warning(f"Database queue full, skipping track {track_id}")
                        with self.processing_lock:
                            self.processing_tracks.discard(track_id)
                else:
                    # Remove from processing if failed
                    with self.processing_lock:
                        self.processing_tracks.discard(track_id)
                
                # Mark task as done
                self.embedding_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in embedding worker: {e}")
    
    def _database_worker(self):
        """Optimized database worker"""
        logger.info("Database worker started")
        
        while self.running:
            try:
                # Get embedding data with shorter timeout
                data = self.db_query_queue.get(timeout=0.5)
                
                track_id = data['track_id']
                embedding = data['embedding']
                detection = data['detection']
                
                # Query database for match
                reid_num, name = self.db_manager.query_face(embedding)
                
                if reid_num is not None:
                    # Found existing person
                    with self.reid_lock:
                        self.track_id_to_reid[track_id] = reid_num
                    logger.info(f"Matched track {track_id} to existing person: {name}")
                else:
                    # New person - process in background to avoid blocking
                    self._process_new_person(track_id, embedding, detection)
                
                # Remove from processing set
                with self.processing_lock:
                    self.processing_tracks.discard(track_id)
                
                # Mark task as done
                self.db_query_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in database worker: {e}")
    
    def _process_new_person(self, track_id, embedding, detection):
        """Process new person in background"""
        try:
            new_reid_num = self.db_manager.get_next_reid_num()
            new_name = f"unknown_{new_reid_num}"
            
            # Save face image
            img_save_dir = "saved_faces"
            os.makedirs(img_save_dir, exist_ok=True)
            img_filename = f"{img_save_dir}/reid_{new_reid_num}.jpg"
            
            # Use original face crop
            face_crop = detection.face_crop
            cv2.imwrite(img_filename, face_crop)
            
            # Add to database
            if self.db_manager.face_db.add(
                ids=[f"reid_{new_reid_num}"],
                embeddings=[embedding],
                metadatas=[{"name": new_name, "image_path": img_filename}]
            ):
                self.db_manager.reid_name_map[f"reid_{new_reid_num}"] = new_name
                with self.reid_lock:
                    self.track_id_to_reid[track_id] = new_reid_num
                logger.info(f"Added new person: {new_name}")
            else:
                logger.error(f"Failed to add new person for track {track_id}")
                
        except Exception as e:
            logger.error(f"Error processing new person: {e}")
    
    def _assess_face_quality(self, face_crop):
        """Fast face quality assessment"""
        if face_crop is None or face_crop.size == 0:
            return False, "Empty crop"

        # Check dimensions
        if face_crop.shape[0] < 80 or face_crop.shape[1] < 80:
            return False, "Too small"

        # Quick brightness check
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if len(face_crop.shape) == 3 else face_crop
        mean_intensity = np.mean(gray)

        if mean_intensity < 40:
            return False, "Too dark"
        if mean_intensity > 215:
            return False, "Too bright"

        # Quick blur check (simplified)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 80:  # Lowered threshold for speed
            return False, "Too blurry"

        return True, "Good quality"
    
    def process_frame(self, frame):
        """Optimized frame processing with frame skipping"""
        if frame is None:
            return None
        
        # Frame skipping for performance
        self.frame_counter += 1
        should_detect = (self.frame_counter % self.process_every_n_frames == 0)
        
        detection_start = time.time()
        
        # Preprocess frame
        processed_frame, scale_factor = self._preprocess_frame(frame)
        
        detections = []
        
        if should_detect:
            try:
                # Run YOLO detection with optimized parameters
                results = self.model.track(
                    processed_frame, 
                    stream=True, 
                    persist=True, 
                    tracker="botsort.yaml",
                    # verbose=False,
                    conf=0.7,      # Increased confidence for better quality
                    iou=0.45,      # Slightly lower IoU for better detection
                    max_det=10,    # Reduced max detections
                    imgsz=640      # Smaller input size for speed
                )
            except Exception as e:
                logger.error(f"YOLO tracking error: {e}")
                return processed_frame
            
            for result in results:
                if result.boxes is None:
                    continue
                    
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    # Handle tracking ID
                    if hasattr(box, 'id') and box.id is not None:
                        track_id = int(box.id[0])
                    else:
                        track_id = hash(f"{x1}_{y1}_{x2}_{y2}") % 10000
                    
                    if conf > 0.75:  # Higher confidence threshold
                        detection = {
                            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                            'conf': conf, 'track_id': track_id
                        }
                        detections.append(detection)
                        
                        # Check if we should process this track
                        if self._should_process_track(track_id):
                            # Extract face crop with minimal padding
                            padding = 10  # Reduced padding
                            h, w = processed_frame.shape[:2]
                            x1_pad = max(0, x1 - padding)
                            y1_pad = max(0, y1 - padding)
                            x2_pad = min(w, x2 + padding)
                            y2_pad = min(h, y2 + padding)
                            
                            face_crop = processed_frame[y1_pad:y2_pad, x1_pad:x2_pad]
                            
                            if (face_crop.size > 0 and 
                                face_crop.shape[0] > 80 and 
                                face_crop.shape[1] > 80):
                                
                                face_detection = FaceDetection(
                                    x1=x1, y1=y1, x2=x2, y2=y2,
                                    conf=conf, track_id=track_id,
                                    face_crop=face_crop.copy(),
                                    frame_timestamp=time.time()
                                )
                                
                                # Add to embedding queue (non-blocking)
                                try:
                                    self.embedding_queue.put_nowait(face_detection)
                                except queue.Full:
                                    # Queue is full, skip this detection
                                    logger.debug(f"Embedding queue full, skipping track {track_id}")
        
        # Store detections
        with self.detection_lock:
            if detections:  # Only update if we have new detections
                self.current_detections = detections
            else:
                # Use cached detections if no new ones
                detections = self.current_detections
        
        # Record detection processing time
        if should_detect:
            detection_time = time.time() - detection_start
            self.detection_processing_time.append(detection_time)
            if len(self.detection_processing_time) > 50:
                self.detection_processing_time.pop(0)
        
        # Render the frame
        return self.render_frame(processed_frame, detections)
    
    def render_frame(self, frame, detections):
        """Optimized frame rendering"""
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
            label = "Unknown"
            color = (0, 0, 255)  # Red for unknown
            
            # Check processing status
            with self.processing_lock:
                is_processing = track_id in self.processing_tracks
            
            if is_processing:
                label = "Processing..."
                color = (0, 165, 255)  # Orange for processing
            elif track_id in self.track_id_to_reid:
                reid_num = self.track_id_to_reid[track_id]
                key = f"reid_{reid_num}"
                self.visible_reids.append(key)
                name = self.db_manager.reid_name_map.get(key, f"unknown_{reid_num}")
                label = f"{name}"
                color = (0, 255, 0)  # Green for recognized
            
            # Draw bounding box and label
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_copy, f"{label} ({conf:.2f})",
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Calculate FPS
        curr_time = time.time()
        self.frame_count += 1
        if curr_time - self.prev_time >= 1.0:
            self.fps = self.frame_count / (curr_time - self.prev_time)
            self.prev_time = curr_time
            self.frame_count = 0
        
        # Display minimal info
        cv2.putText(frame_copy, f"FPS: {self.fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.putText(frame_copy, f"Faces: {len(detections)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame_copy
    
    def get_all_faces(self):
        """Get all faces from database"""
        try:
            all_faces = []
            for reid_id, name in self.db_manager.reid_name_map.items():
                try:
                    results = self.db_manager.face_db.get(ids=[reid_id])
                    if results['metadatas'] and len(results['metadatas']) > 0:
                        metadata = results['metadatas'][0]
                        image_path = metadata.get('image_path', '')
                        all_faces.append({
                            'reid_id': reid_id,
                            'name': name,
                            'image_path': image_path,
                            'visible': reid_id in self.visible_reids
                        })
                except Exception as e:
                    logger.error(f"Error getting metadata for {reid_id}: {e}")
                    all_faces.append({
                        'reid_id': reid_id,
                        'name': name,
                        'image_path': '',
                        'visible': reid_id in self.visible_reids
                    })
            return all_faces
        except Exception as e:
            logger.error(f"Error getting all faces: {e}")
            return []
    
    def update_face_name(self, reid_id, new_name):
        """Update face name"""
        try:
            if self.db_manager.update_name(reid_id, new_name):
                logger.info(f"Updated {reid_id} to name: {new_name}")
                return True, f"Successfully updated {reid_id} to '{new_name}'"
            else:
                logger.error(f"Failed to update {reid_id}")
                return False, f"Failed to update {reid_id}"
        except Exception as e:
            logger.error(f"Error updating face name: {e}")
            return False, f"Error: {e}"
    
    def delete_face(self, reid_id):
        """Delete a face from database"""
        try:
            # Remove from ChromaDB
            self.db_manager.face_db.delete(ids=[reid_id])
            
            # Remove from name mapping
            if reid_id in self.db_manager.reid_name_map:
                del self.db_manager.reid_name_map[reid_id]
            
            # Remove from tracking
            tracks_to_remove = []
            with self.reid_lock:
                for track_id, reid_num in self.track_id_to_reid.items():
                    if f"reid_{reid_num}" == reid_id:
                        tracks_to_remove.append(track_id)
            
            for track_id in tracks_to_remove:
                with self.reid_lock:
                    if track_id in self.track_id_to_reid:
                        del self.track_id_to_reid[track_id]
                with self.embedding_lock:
                    if track_id in self.track_id_to_embedding:
                        del self.track_id_to_embedding[track_id]
            
            logger.info(f"Deleted face {reid_id}")
            return True, f"Successfully deleted {reid_id}"
        except Exception as e:
            logger.error(f"Error deleting face: {e}")
            return False, f"Error deleting face: {e}"
    
    def get_performance_stats(self):
        """Get performance statistics"""
        stats = {
            'fps': self.fps,
            'processing_tracks': len(self.processing_tracks),
            'embedding_queue_size': self.embedding_queue.qsize(),
            'db_queue_size': self.db_query_queue.qsize(),
            'avg_embedding_time': 0,
            'avg_detection_time': 0
        }
        
        if self.embedding_processing_time:
            stats['avg_embedding_time'] = sum(self.embedding_processing_time) / len(self.embedding_processing_time)
        
        if self.detection_processing_time:
            stats['avg_detection_time'] = sum(self.detection_processing_time) / len(self.detection_processing_time)
        
        return stats
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up...")
        self.running = False
        
        # Wait for queues to finish (with timeout)
        try:
            self.embedding_queue.join()
            self.db_query_queue.join()
        except:
            pass
        
        # Shutdown thread pools
        self.embedding_executor.shutdown(wait=True)
        self.db_executor.shutdown(wait=True)
        
        logger.info("Cleanup complete")

# Global system instance
_face_recognition_system = None
_system_lock = threading.Lock()

def get_face_recognition_system():
    global _face_recognition_system
    if _face_recognition_system is not None:
        return _face_recognition_system

    with _system_lock:
        if _face_recognition_system is None:
            _face_recognition_system = FaceRecognitionSystem()
            logger.info("Face recognition system initialized")
    return _face_recognition_system

def get_face_list():
    """Get list of all faces for the interface"""
    system = get_face_recognition_system()
    faces = system.get_all_faces()
    
    face_data = []
    for face in faces:
        status = "🟢 Visible" if face['visible'] else "⚫ Not visible"
        face_data.append([
            face['reid_id'],
            face['name'],
            status,
            face['image_path']
        ])
    
    return face_data

def get_face_gallery(face_data=None):
    """Get face gallery items"""
    if face_data is None:
        face_data = get_face_list()
    
    gallery_items = []
    for row in face_data:
        if len(row) >= 4:
            reid, name, status, path = row
            if path and os.path.exists(path):
                gallery_items.append((path, f"{reid} | {name} ({status})"))
    
    return gallery_items

def update_face_name_handler(reid_id, new_name):
    """Handler for updating face names"""
    if not reid_id or not new_name:
        return "❌ Please provide both ReID and new name", get_face_gallery()
    
    system = get_face_recognition_system()
    success, message = system.update_face_name(reid_id, new_name)
    
    status = "✅ " if success else "❌ "
    return status + message, get_face_gallery()

def delete_face_handler(reid_id):
    """Handler for deleting faces"""
    if not reid_id:
        return "❌ Please provide ReID to delete", get_face_gallery()
    
    system = get_face_recognition_system()
    success, message = system.delete_face(reid_id)
    
    status = "✅ " if success else "❌ "
    return status + message, get_face_gallery()

def refresh_face_list():
    """Refresh the face list"""
    return get_face_gallery()

def main(frame):
    """Optimized main function for processing frames"""
    try:
        # Get the system instance
        system = get_face_recognition_system()
        
        # Process the frame
        processed_frame = system.process_frame(frame)
        return processed_frame
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        return frame  # Return original frame on error

def get_reid_on_click(evt: gr.SelectData, face_data):
    try:
        index = evt.index
        if index < len(face_data):
            return face_data[index][0]  # reid
        return ""
    except Exception as e:
        logger.error(f"Error in get_reid_on_click: {e}")
        return ""

def update_name(selected_reid, new_name):
    """Update name handler"""
    if not selected_reid or not new_name:
        return "❌ ReID or name missing", get_face_gallery()
    
    success, msg = update_face_name_handler(selected_reid, new_name)
    return msg, get_face_gallery()

# WebRTC configuration
rtc_config = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}
# Create the Gradio interface
with gr.Blocks(title="Face Recognition System") as demo:
    selected_reid = gr.State()
    
    with gr.Row():
        # Left column - Video stream
        with gr.Column(scale=1):
            gr.HTML("<h2>📹 Live Video Feed</h2>")
            webrtc = WebRTC(
                modality="video",
                mode="send-receive",
                rtc_configuration=rtc_config
            )
            webrtc.stream(
                fn=main,
                inputs=[webrtc],
                outputs=[webrtc],
                concurrency_limit=1,
                
            )
            
        # Right column - Face management
        with gr.Column(scale=1):
            gr.HTML("<h2>👥 Face Management</h2>")
            
            # Status display
            status_display = gr.HTML(
                value="<div class='status-message'>Ready to manage faces</div>",
                elem_classes=["status-message"]
            )
            
            # Face gallery
            gallery = gr.Gallery(
                label="Face Gallery", 
                columns=3, 
                object_fit="cover", 
                height="auto", 
                show_label=True,
                value=get_face_gallery()
            )
            
            # Refresh button
            refresh_btn = gr.Button("🔄 Refresh Face List", variant="secondary")
            
            # Update name section
            with gr.Row():
                name_input = gr.Textbox(
                    label="New Name", 
                    placeholder="Enter new name",
                    scale=2
                )
                update_btn = gr.Button("✅ Update Name", variant="primary")
            
            # Delete face section
            gr.HTML("<h3>🗑️ Delete Face</h3>")
            delete_btn = gr.Button("❌ Delete Face", variant="stop")
    
    # Event handlers
    gallery.select(fn=get_reid_on_click, outputs=[selected_reid])
    
    refresh_btn.click(
        fn=refresh_face_list,
        outputs=[gallery]
    )
    
    update_btn.click(
        fn=update_name,
        inputs=[selected_reid, name_input],
        outputs=[status_display, gallery]  # This order matches the function return
    )
    
    delete_btn.click(
        fn=delete_face_handler,
        inputs=[selected_reid],
        outputs=[status_display, gallery]
    )

demo.launch(share=False)