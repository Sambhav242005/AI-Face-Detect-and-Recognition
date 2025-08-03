import cv2
import numpy as np
import face_recognition
import logging

logger = logging.getLogger("face_encoding_worker")

def face_encoding_worker(face_crop):
    try:
        if face_crop is None or face_crop.size == 0:
            return None
        face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(face_crop_rgb, model="cnn")
        if not face_locations:
            return None
        encodings = face_recognition.face_encodings(face_crop_rgb, face_locations)
        if encodings:
            return encodings[0].tolist()
        return None
    except Exception as e:
        logger.error(f"Error: {e}")
        return None
