# from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QSizePolicy
# from PyQt6.QtCore import QThread, pyqtSignal, Qt
# from PyQt5.QtGui import QActionEvent

from db import DatabaseManager

db = DatabaseManager()

db._connect()
# class CameraWorker(QThread):
#     frame_ready = pyqtSignal(np.ndarray)

#     def __init__(self,  camera_id=0):
#         super().__init__()
#         self.camera_id = camera_id
#         self.running = False
        
#         self.model = YOLO("model/yolov11l-face.pt")
        

#     def run(self):
#         self.running = True
#         cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
#         if not cap.isOpened():
#             print(f"Error: Could not open camera {self.camera_id}.")
#             self.running = False
#             return

#         while self.running:
#             ret, frame = cap.read()
#             if ret:
#                 self.frame_ready.emit(frame)
#                 time.sleep(0.1)

#         cap.release()
#         print("Camera thread stopped and resource released.")

#     def stop(self):
#         self.running = False
#         self.wait()

# class MainWindow(QMainWindow):
#     def __init__(self,):
#         super().__init__()
        
#         self.setWindowTitle("PyQt Face Recognition")
#         self.setGeometry(100, 100, 1280, 720)

#         self.central_widget = QWidget()
#         self.setCentralWidget(self.central_widget)
#         layout = QVBoxLayout(self.central_widget)

#         self.video_label = QLabel("Press 'Start Camera' to begin.")
#         self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
#         self.video_label.setStyleSheet("border: 2px solid black; background-color: #333;")
#         self.video_label.setFixedSize(1280, 720)  # or use setMaximumSize
#         layout.addWidget(self.video_label)

#         self.start_button = QPushButton("Start Camera")
#         self.start_button.clicked.connect(self.start_camera)
#         layout.addWidget(self.start_button)

#         self.camera_worker = None

#     def start_camera(self):
#         if self.camera_worker is None or not self.camera_worker.isRunning():
#             self.camera_worker = CameraWorker(camera_id=0)
#             self.camera_worker.frame_ready.connect(self.update_image)
#             self.camera_worker.start()
#             print("Camera worker started.")
#             self.start_button.setText("Stop Camera")
#         else:
#             self.stop_camera()
#             self.start_button.setText("Start Camera")

#     def update_image(self, frame: np.ndarray):
#         if frame is None:
#             return
#         try:
#             rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             h, w, ch = rgb_image.shape
#             bytes_per_line = ch * w
#             qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
#             pixmap = QPixmap.fromImage(qt_image).scaled(
#                 self.video_label.width(), self.video_label.height(),
#                 Qt.AspectRatioMode.KeepAspectRatio,
#                 Qt.TransformationMode.SmoothTransformation
#             )
#             self.video_label.setPixmap(pixmap)
#         except Exception as e:
#             print(f"Error updating image: {e}")

#     def stop_camera(self):
#         if self.camera_worker and self.camera_worker.isRunning():
#             self.camera_worker.stop()
#             self.camera_worker = None
#             self.video_label.setText("Camera stopped. Press 'Start Camera' to begin.")

#     def closeEvent(self, event):
#         print("Closing window...")
#         self.stop_camera()
#         event.accept()

# if __name__ == "__main__":
#     import multiprocessing
#     multiprocessing.freeze_support()
#     app = QApplication(sys.argv)
#     window = MainWindow()
#     window.show()
#     sys.exit(app.exec())

