import os
import time
import multiprocessing as mp
import threading
from concurrent.futures import ThreadPoolExecutor
import chromadb
import numpy as np
from ultralytics import YOLO
import cv2
import face_recognition
import queue
import torch

from db import DatabaseManager




def face_encoding_worker(face_crop_queue, result_queue):
    """Worker process for face encoding."""
    while True:
        try:
            item = face_crop_queue.get(timeout=1)
            if item is None:  # Stop signal
                break
            
            task_id, face_crop = item
            print(f"Processing task {task_id}, face_crop shape: {face_crop.shape}")
            
            if face_crop is None or face_crop.size == 0:
                result_queue.put((task_id, None))
                continue
                
            # Convert to RGB and encode
            face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(face_crop_rgb, model="cnn")
            
            if not face_locations:
                result_queue.put((task_id, None))
                continue
                
            encodings = face_recognition.face_encodings(face_crop_rgb, face_locations)
            if encodings:
                result_queue.put((task_id, encodings[0].tolist()))
            else:
                result_queue.put((task_id, None))
                
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Encoding worker error: {e}")
            result_queue.put((task_id, None))


class FaceRecognitionSystem:
    def __init__(self, model_path='model/yolov11l-face.pt', face_img_path="saved_faces", camera_id=0):
        
        # Validate inputs
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        os.makedirs(face_img_path, exist_ok=True)
        
        self.model_path = model_path
        self.camera_id = camera_id
        self.face_img_path = face_img_path
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # Load YOLO model once
        print("Loading YOLO model...")
        self.model = YOLO(self.model_path)
        if self.device != "cpu":
            self.model.to(self.device)
        print(f"Model loaded on {self.device}")
        
        # Use exactly 4 cores as requested
        self.num_processes = 4
        self.face_crop_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.encoding_processes = []
        
        # Threading setup
        self.thread_executor = ThreadPoolExecutor(max_workers=2)
        
        # Simple data structures
        self.pending_results = {}
        self.result_lock = threading.Lock()
        
        # Database
        self.db_manager = DatabaseManager()
        self.db_manager._connect()
        
        # FIXED: Simple track_id to identity mapping - this prevents fluctuation
        self.track_to_reid = {}  # track_id -> reid_num
        self.track_to_name = {}  # track_id -> name
        self.track_last_seen = {}  # track_id -> timestamp
        self.processing_tracks = set()  # tracks currently being processed
        
        # FIXED: Add task_id to track_id mapping to handle failed results
        self.task_to_track = {}  # task_id -> track_id
        
        # Clean old tracks every 30 seconds
        self.track_timeout = 30.0
        
        self.active_futures = []
        self.task_counter = 0
        
        # Performance metrics
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
        
        # Start processes and threads
        self._start_processes()
        self.result_thread = threading.Thread(target=self._process_results, daemon=True)
        self.result_thread.start()

    def _start_processes(self):
        """Start exactly 4 encoding processes."""
        print(f"Starting {self.num_processes} encoding processes...")
        for i in range(self.num_processes):
            try:
                process = mp.Process(
                    target=face_encoding_worker,
                    args=(self.face_crop_queue, self.result_queue)
                )
                process.daemon = True
                process.start()
                self.encoding_processes.append(process)
                print(f"Started process {i+1}/{self.num_processes}")
                print(f"Process {i+1} PID: {process.pid}")
            except Exception as e:
                print(f"Failed to start process {i}: {e}")

    def _process_results(self):
        """Thread to collect encoding results."""
        while True:
            try:
                task_id, embedding = self.result_queue.get()
                with self.result_lock:
                    self.pending_results[task_id] = embedding
                    
                    # FIXED: If encoding failed, remove track from processing immediately
                    if embedding is None and task_id in self.task_to_track:
                        track_id = self.task_to_track[task_id]
                        self.processing_tracks.discard(track_id)
                        print(f"Encoding failed for task {task_id}, track {track_id} - removed from processing")
                        
                print(f"Got result for task {task_id}: {embedding is not None}")
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Result processing error: {e}")

    def _clean_old_tracks(self):
        """Remove old track entries to prevent memory buildup."""
        current_time = time.time()
        expired_tracks = []
        
        for track_id, last_seen in self.track_last_seen.items():
            if current_time - last_seen > self.track_timeout:
                expired_tracks.append(track_id)
        
        for track_id in expired_tracks:
            self.track_to_reid.pop(track_id, None)
            self.track_to_name.pop(track_id, None)
            self.track_last_seen.pop(track_id, None)
            self.processing_tracks.discard(track_id)
            print(f"Cleaned expired track {track_id}")

    def detect_face(self, frame):
        """Face detection."""
        if frame is None:
            return []
        
        try:
            results = self.model.track(frame, persist=True, device=self.device, verbose=False)
            detections = []
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        try:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])
                            track_id = int(box.id[0]) if hasattr(box, 'id') and box.id is not None else None
                            
                            if conf > 0.5 and x2 > x1 and y2 > y1 and track_id is not None:
                                face_crop = frame[y1:y2, x1:x2].copy()
                                detections.append({
                                    'bbox': (x1, y1, x2, y2),
                                    'conf': conf,
                                    'face_crop': face_crop,
                                    'track_id': track_id
                                })
                                # Update last seen time
                                self.track_last_seen[track_id] = time.time()
                                
                        except Exception as e:
                            print(f"Detection error: {e}")
                            continue
            
            return detections
            
        except Exception as e:
            print(f"Face detection error: {e}")
            return []

    def submit_encoding_task(self, face_crop, track_id):
        """Submit face encoding task - FIXED: Now tracks task_id to track_id mapping."""
        try:
            if face_crop is None or face_crop.size == 0:
                return None
            
            self.task_counter += 1
            task_id = self.task_counter
            
            # FIXED: Store task_id to track_id mapping
            self.task_to_track[task_id] = track_id
            
            try:
                self.face_crop_queue.put_nowait((task_id, face_crop))
                return task_id
            except queue.Full:
                # FIXED: Clean up mapping if queue is full
                self.task_to_track.pop(task_id, None)
                return None
                
        except Exception as e:
            print(f"Encoding submission error: {e}")
            return None

    def get_encoding_result(self, task_id):
        """Get encoding result."""
        with self.result_lock:
            if task_id in self.pending_results:
                return self.pending_results.pop(task_id)
        return "pending"

    def process_face_async(self, face_crop, track_id):
        """Process face asynchronously."""
        task_id = self.submit_encoding_task(face_crop, track_id)
        if task_id is None:
            return None
        
        future = self.thread_executor.submit(self._process_face_task, task_id, face_crop, track_id)
        return future

    def _process_face_task(self, task_id, face_crop, track_id):
        """Process single face task."""
        # Wait for encoding
        max_wait_time = 5
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            emb = self.get_encoding_result(task_id)
            if emb == "pending":
                time.sleep(0.01)
                continue
            elif emb is None:
                # FIXED: Encoding failed - clean up and return
                print(f"Encoding failed for track {track_id}")
                self.processing_tracks.discard(track_id)
                self.task_to_track.pop(task_id, None)
                return None
            else:
                break
        else:
            # FIXED: Timeout - clean up and return
            print(f"Encoding timeout for track {track_id}")
            self.processing_tracks.discard(track_id)
            self.task_to_track.pop(task_id, None)
            return None

        try:
            # Clean up task mapping
            self.task_to_track.pop(task_id, None)
            
            # Database query
            reid_num, name = self.db_manager.query(embedding=emb)

            if reid_num is None:
                # New face
                reid_num = self.db_manager.next_reid_num()
                name = f"Unknown_{reid_num}"
                cv2.imwrite(f"{self.face_img_path}/reid_{reid_num}.jpg", face_crop)
                self.db_manager.add(embedding=emb, reid_num=reid_num, name=name)

            # FIXED: Store result by track_id for stable display
            self.track_to_reid[track_id] = reid_num
            self.track_to_name[track_id] = name
            
            return {"reid_num": reid_num, "name": name, "track_id": track_id}
            
        except Exception as e:
            print(f"Face processing error: {e}")
            # FIXED: Clean up on any error
            self.processing_tracks.discard(track_id)
            return None

    def cleanup_futures(self):
        """Clean up completed futures."""
        results = []
        remaining = []
        
        for future in self.active_futures:
            if future.done():
                try:
                    result = future.result(timeout=0.1)
                    if result:
                        results.append(result)
                except Exception as e:
                    print(f"Future error: {e}")
            else:
                remaining.append(future)
        
        self.active_futures = remaining
        return results

    def draw_detections(self, frame, detections):
        """Draw detections with stable labels - FIXED to prevent fluctuation."""
        display_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            conf = detection['conf']
            track_id = detection['track_id']
            
            # Draw bounding box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # FIXED: Use stable track_id mapping instead of position-based cache
            if track_id in self.track_to_name:
                # We know this track - show stable name
                name = self.track_to_name[track_id]
                reid_num = self.track_to_reid[track_id]
                label = f"{name} (ID:{reid_num})"
                color = (0, 255, 0)  # Green
            elif track_id in self.processing_tracks:
                # Currently processing this track
                label = f"Processing... {conf:.2f}"
                color = (0, 165, 255)  # Orange
            else:
                # Unknown track
                label = f"Unknown {conf:.2f}"
                color = (0, 0, 255)  # Red
            
            # Draw label with background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(display_frame, (x1, y1-30), (x1 + label_size[0], y1), color, -1)
            cv2.putText(display_frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return display_frame

    def rename_person(self, reid_num: int, new_name: str):
        """Rename person by ReID number."""
        success = self.db_manager.update_name(reid_num, new_name)
        if success:
            # FIXED: Update all track mappings with the new name
            for track_id, stored_reid in self.track_to_reid.items():
                if stored_reid == reid_num:
                    self.track_to_name[track_id] = new_name
        return success

    def process_frame(self, frame):
        """Process single frame."""
        # Calculate FPS
        self.frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - self.last_time
    
        if elapsed_time >= 1.0:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.last_time = current_time
        
        # Clean old tracks periodically
        if self.frame_count % 900 == 0:  # Every ~30 seconds at 30fps
            self._clean_old_tracks()
        
        # Detect faces
        detections = self.detect_face(frame)
        
        # Clean up futures
        completed_results = self.cleanup_futures()
        
        # Process completed results
        for result in completed_results:
            if result and 'track_id' in result:
                track_id = result['track_id']
                self.processing_tracks.discard(track_id)
        
        # FIXED: Clean up stale processing tracks (tracks that have been processing too long)
        current_time = time.time()
        stale_tracks = []
        for track_id in self.processing_tracks.copy():
            if track_id in self.track_last_seen:
                time_since_seen = current_time - self.track_last_seen[track_id]
                if time_since_seen > 10.0:  # 10 seconds timeout
                    stale_tracks.append(track_id)
        
        for track_id in stale_tracks:
            self.processing_tracks.discard(track_id)
            print(f"Removed stale processing track {track_id}")
        
        # Submit new tasks for unknown tracks only
        for detection in detections:
            track_id = detection['track_id']
            
            # FIXED: Only process if we don't know this track and it's not being processed
            if (track_id not in self.track_to_name and 
                track_id not in self.processing_tracks and 
                len(self.active_futures) < 3):  # Limit concurrent tasks
                
                self.processing_tracks.add(track_id)
                future = self.process_face_async(detection["face_crop"], track_id)
                if future:
                    self.active_futures.append(future)
                else:
                    # FIXED: If future creation failed, remove from processing
                    self.processing_tracks.discard(track_id)
        
        # Draw frame
        display_frame = self.draw_detections(frame, detections)
        
        # Add info
        info_text = f"FPS: {self.fps:.1f} | Tasks: {len(self.active_futures)} | Known: {len(self.track_to_name)} | Processing: {len(self.processing_tracks)}"
        cv2.putText(display_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Check worker processes
        alive_processes = sum(1 for p in self.encoding_processes if p.is_alive())
        if alive_processes < self.num_processes:
            print(f"WARNING: Only {alive_processes}/{self.num_processes} workers alive!")
        
        return display_frame

    def run_generator(self, camera_id=None):
        """Generator for external integration."""
        if camera_id is not None:
            self.camera_id = camera_id
            
        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_id}")
            return
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                # FIXED: Process every frame for stable tracking
                processed_frame = self.process_frame(frame)
                yield processed_frame
                
        except Exception as e:
            print(f"Generator error: {e}")
        finally:
            if hasattr(self, 'cap'):
                self.cap.release()

    def run(self):
        """Main run function with UI."""
        print("Starting Face Recognition System...")
        print("Controls:")
        print("  'q' - Quit")
        print("  'r' - Rename person (enter ReID and new name)")
        print("Using 4 CPU cores for processing")
        print("FIXED: Stable tracking - no more fluctuation or stuck processing!")
        
        for processed_frame in self.run_generator():
            cv2.imshow('Face Recognition System', processed_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("\nEnter: ReID_number new_name (e.g., '1 John')")
                try:
                    user_input = input("Rename: ").strip().split(' ', 1)
                    if len(user_input) == 2:
                        reid_num = int(user_input[0])
                        new_name = user_input[1]
                        if self.rename_person(reid_num, new_name):
                            print(f"Successfully renamed ReID {reid_num} to {new_name}")
                        else:
                            print(f"Failed to rename ReID {reid_num}")
                    else:
                        print("Invalid format. Use: ReID_number new_name")
                except ValueError:
                    print("Invalid ReID number")
                except Exception as e:
                    print(f"Rename error: {e}")

        self.cleanup()

    def cleanup(self):
        """Cleanup all resources."""
        print("Cleaning up...")
        
        try:
            # Stop processes
            for _ in range(self.num_processes):
                try:
                    self.face_crop_queue.put_nowait(None)
                except:
                    pass
            
            # Wait for processes
            for process in self.encoding_processes:
                process.join(timeout=1)
                if process.is_alive():
                    process.terminate()
            
            # Shutdown executor
            self.thread_executor.shutdown(wait=False)
            
            # Close camera
            if hasattr(self, 'cap'):
                self.cap.release()
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"Cleanup error: {e}")
        
        print("Cleanup completed")


def main():
    """Main function."""
    face_system = None
    try:
        face_system = FaceRecognitionSystem()
        face_system.run()
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if face_system:
            face_system.cleanup()


if __name__ == "__main__":
    # Set multiprocessing method
    if hasattr(mp, 'set_start_method'):
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
    
    main()