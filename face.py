import os
import sys
import traceback
import cv2
import time
import numpy as np
from ultralytics import YOLO
import chromadb
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Dict, Optional
import logging
import torch

# --- PyQt6 Imports ---
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QLineEdit, QScrollArea
)
from PyQt6.QtGui import QPixmap, QImage, QFont
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot

# --- Local Imports ---
# Make sure these files are in the same directory.
from db import DatabaseManager
from face_encoding_worker import face_encoding_worker

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Data Classes ---
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

# --- Main System Class ---
class FaceRecognitionSystem:
    """Optimized face recognition system with robust processing pipelines."""

    def __init__(self, model_path='model/yolov11l-face.pt', target_width=1280, target_height=720):
        self.target_width = target_width
        self.target_height = target_height
        self.frame_counter = 0
        self.process_every_n_frames = 2

        # Load YOLO model
        try:
            self.model = YOLO(model_path)
            self.model.fuse()
            if torch.cuda.is_available():
                self.model.to('cuda:0').half()
                logger.info("Model loaded on CUDA with half precision")
            else:
                logger.info("Model loaded on CPU")
        except Exception as e:
            logger.error(f"Fatal error loading YOLO model: {e}")
            sys.exit(1)

        self.db_manager = DatabaseManager()

        # Threading and Queues
        self.embedding_queue = queue.Queue(maxsize=5)
        self.db_query_queue = queue.Queue(maxsize=10)
        self.reid_lock = threading.Lock()
        self.processing_lock = threading.Lock()
        self.detection_lock = threading.Lock()
        
        # Tracking Data
        self.track_id_to_reid = {}
        self.processing_tracks = set()
        self.track_last_processed = {}
        self.track_cooldown_time = 5.0
        self.current_detections = []
        self.visible_reids = []

        # Control & Stats
        self.running = True
        self.prev_time = time.time()
        self.frame_count = 0
        self.fps = 0

        self.start_background_threads()

    def start_background_threads(self):
        """Initializes and starts all background worker threads."""
        self.embedding_thread = threading.Thread(target=self._embedding_worker, daemon=True)
        self.db_thread = threading.Thread(target=self._database_worker, daemon=True)
        self.embedding_thread.start()
        self.db_thread.start()
        logger.info("Background threads started")

    def _preprocess_frame(self, frame):
        """Resizes frame for consistent processing."""
        return cv2.resize(frame, (self.target_width, self.target_height), interpolation=cv2.INTER_LINEAR)

    def _assess_face_quality(self, face_crop):
        """Performs fast checks on a face crop for size, lighting, and blur."""
        if face_crop is None or face_crop.size == 0:
            return False, "Empty crop"
        if face_crop.shape[0] < 60 or face_crop.shape[1] < 60:
            return False, "Too small"
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        mean_intensity = np.mean(gray)
        if not 30 < mean_intensity < 225:
            return False, "Bad lighting"
        if cv2.Laplacian(gray, cv2.CV_64F).var() < 50:
            return False, "Too blurry"
        return True, "Good quality"

    def _initiate_processing(self, track_id):
        """Atomically checks if a track is new and ready for processing."""
        with self.processing_lock:
            if track_id in self.track_id_to_reid or track_id in self.processing_tracks:
                return False
            if time.time() - self.track_last_processed.get(track_id, 0) < self.track_cooldown_time:
                return False
            self.processing_tracks.add(track_id)
            return True

    def _safe_face_encoding(self, face_crop, track_id):
        """Generates an embedding if the face crop is of sufficient quality."""
        is_good, reason = self._assess_face_quality(face_crop)
        if not is_good:
            logger.debug(f"Skipping track {track_id} (quality: {reason})")
            return None
        try:
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (160, 160), interpolation=cv2.INTER_LINEAR)
            return face_encoding_worker(face_resized)
        except Exception as e:
            logger.error(f"Encoding error for track {track_id}: {e}")
            return None

    def _embedding_worker(self):
        """Thread worker that produces face embeddings from detections."""
        while self.running:
            try:
                detection = self.embedding_queue.get(timeout=1)
                track_id = detection.track_id
                self.track_last_processed[track_id] = time.time()
                embedding = self._safe_face_encoding(detection.face_crop, track_id)

                if embedding is not None:
                    self.db_query_queue.put({'track_id': track_id, 'embedding': embedding, 'detection': detection})
                else:
                    with self.processing_lock:
                        self.processing_tracks.discard(track_id)
                self.embedding_queue.task_done()
            except queue.Empty:
                continue

    def _database_worker(self):
        """Thread worker that handles database queries and new registrations."""
        while self.running:
            try:
                data = self.db_query_queue.get(timeout=1)
                track_id, embedding = data['track_id'], data['embedding']
                reid_num, name = self.db_manager.query_face(embedding)

                if reid_num is not None:
                    with self.reid_lock:
                        self.track_id_to_reid[track_id] = reid_num
                else:
                    self._process_new_person(track_id, embedding, data['detection'])
                
                with self.processing_lock:
                    self.processing_tracks.discard(track_id)
                self.db_query_queue.task_done()
            except queue.Empty:
                continue

    def _process_new_person(self, track_id, embedding, detection):
        """Saves a new person's data to the database and memory maps."""
        try:
            new_reid_num = self.db_manager.get_next_reid_num()
            new_name = f"unknown_{new_reid_num}"
            img_filename = os.path.join("saved_faces", f"reid_{new_reid_num}.jpg")
            os.makedirs("saved_faces", exist_ok=True)
            
            if not cv2.imwrite(img_filename, detection.face_crop):
                logger.error(f"CRITICAL: Failed to save image {img_filename}")
                return
            
            self.db_manager.face_db.add(
                ids=[f"reid_{new_reid_num}"],
                embeddings=[embedding],
                metadatas=[{"name": new_name, "image_path": img_filename}]
            )

            self.db_manager.reid_name_map[f"reid_{new_reid_num}"] = new_name
            with self.reid_lock:
                self.track_id_to_reid[track_id] = new_reid_num
            logger.info(f"✅ Successfully registered {new_name} (reid_{new_reid_num}).")
        except Exception as e:
            logger.error(f"EXCEPTION in _process_new_person for track {track_id}: {e}")
            traceback.print_exc()

    def process_frame(self, frame):
        """Main processing entry point for each new frame from the camera."""
        if frame is None:
            return None
        processed_frame = self._preprocess_frame(frame)
        if processed_frame is None:
            return None # Added check to prevent error
            
        self.frame_counter += 1
        
        if self.frame_counter % self.process_every_n_frames == 0:
            detections = []
            try:
                results = self.model.track(processed_frame, stream=False, persist=True, tracker="botsort.yaml", conf=0.7, verbose=False)
                if results and results[0].boxes:
                    for box in results[0].boxes:
                        if box.id is None: continue
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        track_id = int(box.id[0])
                        
                        detections.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'track_id': track_id})

                        if self._initiate_processing(track_id):
                            face_crop = processed_frame[y1:y2, x1:x2]
                            face_detection = FaceDetection(x1, y1, x2, y2, float(box.conf[0]), track_id, face_crop, time.time())
                            try:
                                self.embedding_queue.put_nowait(face_detection)
                            except queue.Full:
                                with self.processing_lock:
                                    self.processing_tracks.discard(track_id)
            except Exception as e:
                logger.error(f"YOLO tracking error: {e}")

            with self.detection_lock:
                self.current_detections = detections
        else:
             with self.detection_lock:
                detections = self.current_detections

        return self.render_frame(processed_frame, detections)

    def render_frame(self, frame, detections):
        """Draws bounding boxes and labels on the frame."""
        self.visible_reids.clear()
        for det in detections:
            track_id = det['track_id']
            label, color = "Unknown", (0, 0, 255)
            
            if track_id in self.processing_tracks:
                label, color = "Processing...", (0, 165, 255)
            elif track_id in self.track_id_to_reid:
                reid_num = self.track_id_to_reid[track_id]
                key = f"reid_{reid_num}"
                self.visible_reids.append(key)
                name = self.db_manager.reid_name_map.get(key, key)
                label, color = name, (0, 255, 0)
            
            x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        if time.time() - self.prev_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.prev_time = time.time()
        self.frame_count += 1
        
        cv2.putText(frame, f"FPS: {self.fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return frame

    def get_all_faces(self):
        """Retrieves all registered faces for display in the gallery."""
        all_faces = []
        for reid_id, name in self.db_manager.reid_name_map.items():
            try:
                meta = self.db_manager.face_db.get(ids=[reid_id], include=["metadatas"])['metadatas'][0]
                all_faces.append({'reid_id': reid_id, 'name': name, 'image_path': meta.get('image_path', ''), 'visible': reid_id in self.visible_reids})
            except Exception as e:
                logger.error(f"Error getting metadata for {reid_id}: {e}")
        return all_faces

    def update_face_name(self, reid_id, new_name):
        """Updates a face's name in the database and memory."""
        if self.db_manager.update_name(reid_id, new_name):
            return True, f"Updated to '{new_name}'"
        return False, "Failed to update."

    def delete_face(self, reid_id):
        """Deletes a face from the database and all internal tracking."""
        try:
            self.db_manager.face_db.delete(ids=[reid_id])
            if reid_id in self.db_manager.reid_name_map:
                del self.db_manager.reid_name_map[reid_id]
            with self.reid_lock:
                for tid, rid in list(self.track_id_to_reid.items()):
                    if f"reid_{rid}" == reid_id:
                        del self.track_id_to_reid[tid]
            return True, f"Deleted {reid_id}"
        except Exception as e:
            logger.error(f"Error deleting face {reid_id}: {e}")
            return False, "Error deleting face."

    def cleanup(self):
        """Shuts down all background threads."""
        logger.info("Cleaning up resources...")
        self.running = False
        if self.embedding_thread.is_alive(): self.embedding_thread.join()
        if self.db_thread.is_alive(): self.db_thread.join()
        logger.info("Cleanup complete.")

# --- GUI Classes ---
class VideoThread(QThread):
    """Runs the video processing loop in a background Qt thread."""
    change_pixmap_signal = pyqtSignal(np.ndarray)
    update_gallery_signal = pyqtSignal(list)

    def __init__(self, system: FaceRecognitionSystem):
        super().__init__()
        self.system = system
        self._run_flag = True

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Could not open video stream.")
            return

        gallery_update_counter = 0
        while self._run_flag:
            ret, frame = cap.read()
            if ret:
                processed_frame = self.system.process_frame(frame)
                if processed_frame is not None:
                    self.change_pixmap_signal.emit(processed_frame)

                gallery_update_counter += 1
                if gallery_update_counter % 30 == 0:
                    self.update_gallery_signal.emit(self.system.get_all_faces())
            time.sleep(0.01)
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

class FaceRecognitionApp(QMainWindow):
    """The main application window (GUI)."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition System")
        self.show()
        self.system = FaceRecognitionSystem()
        self.selected_reid = None
        self.initUI()
        self.start_video_thread()

    def initUI(self):
        """Sets up the entire user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Video Feed Panel
        left_layout = QVBoxLayout()
        self.video_label = QLabel("Starting camera...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("border: 2px solid #3498db; background-color: #000;")
        left_layout.addWidget(self.video_label)
        main_layout.addLayout(left_layout, 3)

        self.video_label.setFixedSize(1280, 720)
        # Management Panel
        right_layout = QVBoxLayout()
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.gallery_widget = QWidget()
        self.gallery_layout = QGridLayout(self.gallery_widget)
        self.gallery_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll_area.setWidget(self.gallery_widget)
        self.status_label = QLabel("System Ready")
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Select face, enter new name...")
        self.update_btn = QPushButton("Update Name")
        self.delete_btn = QPushButton("Delete Face")
        
        right_layout.addWidget(QLabel("<h2>Face Management</h2>"))
        right_layout.addWidget(self.status_label)
        right_layout.addWidget(self.scroll_area, 1)
        right_layout.addWidget(self.name_input)
        right_layout.addWidget(self.update_btn)
        right_layout.addWidget(self.delete_btn)
        main_layout.addLayout(right_layout, 1)

        # Connect signals
        self.update_btn.clicked.connect(self.update_face_name)
        self.delete_btn.clicked.connect(self.delete_face)

    def start_video_thread(self):
        """Creates and starts the video processing thread."""
        self.thread = VideoThread(self.system)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.update_gallery_signal.connect(self.update_gallery_display)
        self.thread.start()
        self.update_gallery_display(self.system.get_all_faces())

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the video label with a new frame from the VideoThread."""
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        # The fix is to remove .rgbSwapped() from the end of this line.
        # QImage.Format_BGR888 correctly interprets the OpenCV image format.
        qt_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(qt_img)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio))
    @pyqtSlot(list)
    def update_gallery_display(self, faces: list):
        """Clears and repopulates the face gallery."""
        # Safely clear existing widgets
        for i in reversed(range(self.gallery_layout.count())):
            item = self.gallery_layout.itemAt(i)
            if item and item.widget():
                item.widget().setParent(None)

        for i, face in enumerate(faces):
            if not face.get('image_path') or not os.path.exists(face['image_path']): continue
            
            face_widget = QWidget()
            face_layout = QVBoxLayout(face_widget)
            img_label = QLabel()
            img_label.setPixmap(QPixmap(face['image_path']).scaled(128, 128, Qt.AspectRatioMode.KeepAspectRatio))
            info_label = QLabel(f"<b>{face['name']}</b><br><i>{face['reid_id']}</i>")
            face_layout.addWidget(img_label)
            face_layout.addWidget(info_label)

            style = "border: 3px solid {}; border-radius: 5px;".format("#2ecc71" if face['visible'] else "#7f8c8d")
            if face['reid_id'] == self.selected_reid:
                style = "border: 3px solid #3498db; border-radius: 5px;"
            face_widget.setStyleSheet(style)
            
            face_widget.mousePressEvent = lambda e, r=face['reid_id']: self.select_face(r)
            self.gallery_layout.addWidget(face_widget, i // 2, i % 2)

    def select_face(self, reid_id):
        self.selected_reid = reid_id
        self.name_input.setText(self.system.db_manager.reid_name_map.get(reid_id, ""))
        self.status_label.setText(f"Selected: {reid_id}")
        self.update_gallery_display(self.system.get_all_faces())

    def update_face_name(self):
        if not self.selected_reid or not self.name_input.text().strip():
            self.status_label.setText("❌ Select a face and enter a name.")
            return
        success, msg = self.system.update_face_name(self.selected_reid, self.name_input.text().strip())
        self.status_label.setText(("✅ " if success else "❌ ") + msg)
        self.update_gallery_display(self.system.get_all_faces())

    def delete_face(self):
        if not self.selected_reid:
            self.status_label.setText("❌ Select a face to delete.")
            return
        success, msg = self.system.delete_face(self.selected_reid)
        self.status_label.setText(("✅ " if success else "❌ ") + msg)
        self.selected_reid = None
        self.name_input.clear()
        self.update_gallery_display(self.system.get_all_faces())

    def closeEvent(self, event):
        """Ensures threads are stopped cleanly when the window is closed."""
        self.thread.stop()
        self.system.cleanup()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec())