
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import logging
import os
import pickle
import threading
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)


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
