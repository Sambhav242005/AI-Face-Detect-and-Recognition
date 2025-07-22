import os
import queue
import cv2
import time

import face_encoding_worker

logger = logging.getLogger(__name__)

def _process_embedding_result(self, embedding, detection):
    """Process the result of face encoding"""
    try:
        track_id = detection.track_id

        if embedding is None:
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

            # Save face image to disk
            save_dir = os.path.join("saved_faces", new_name)
            os.makedirs(save_dir, exist_ok=True)
            image_path = os.path.join(save_dir, f"{int(time.time())}_{track_id}.jpg")
            cv2.imwrite(image_path, detection.face_crop)

            if self.db_manager.add_face(embedding, new_reid_num, new_name, image_path):
                self.track_id_to_reid[track_id] = new_reid_num
                logger.info(f"Added new person: {new_name} for track {track_id}")
            else:
                logger.error(f"Failed to add new person for track {track_id}")

        # Remove from processing set
        self.processing_tracks.discard(track_id)

    except Exception as e:
        logger.error(f"Error processing embedding result: {e}")
        self.processing_tracks.discard(detection.track_id)


def embedding_thread(self):
    """Thread for processing face embeddings"""
    logger.info("Embedding thread started")

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

                # Process encoding inline to avoid pickling issues
                try:
                    embedding = face_encoding_worker(detection.face_crop)
                    self.db_executor.submit(
                        self._process_embedding_result, embedding, detection
                    )
                except Exception as e:
                    logger.error(f"Encoding failed for track {track_id}: {e}")
                    self.processing_tracks.discard(track_id)

        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Error in embedding thread: {e}")
