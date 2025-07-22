import os
import cv2
import numpy as np
import face_recognition
import logging
import time
# import os  # Uncomment if using image saving

logger = logging.getLogger("face_encoding_worker")

def face_encoding_worker(face_crop):
    """
    Worker function to generate face encodings from face crops.
    Designed to be run in a separate process.
    
    Args:
        face_crop: numpy array of the cropped face image
        
    Returns:
        List of face encoding values or None if encoding fails
    """
    try:
        if face_crop is None or face_crop.size == 0:
            logger.warning("Empty or invalid face crop")
            return None

        # Convert to RGB
        if len(face_crop.shape) == 3 and face_crop.shape[2] == 3:
            face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        else:
            face_crop_rgb = face_crop

        # Resize if too small
        min_size = 64
        if face_crop_rgb.shape[0] < min_size or face_crop_rgb.shape[1] < min_size:
            face_crop_rgb = cv2.resize(face_crop_rgb, (min_size, min_size))

        # Detect face locations
        face_locations = face_recognition.face_locations(face_crop_rgb,model="cnn")  # or 'cnn'
        if not face_locations:
            logger.warning("No face detected in the crop")
            
            # Optional: Save failed crop for debugging
            # timestamp = int(time.time())
            # os.makedirs("debug_failed_faces", exist_ok=True)
            # cv2.imwrite(f"debug_failed_faces/failed_{timestamp}.jpg", face_crop)
            
            return None

        # Get encodings from detected face(s)
        encodings = face_recognition.face_encodings(face_crop_rgb, face_locations)
        if encodings:
            return np.asarray(encodings[0], dtype=np.float32).tolist()
        else:
            logger.warning("Face found but encoding failed")
            return None

    except Exception as e:
        logger.error(f"Error in face encoding worker: {e}")
        return None
