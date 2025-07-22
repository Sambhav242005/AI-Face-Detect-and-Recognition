import gradio as gr
import numpy as np
import cv2
import time
from fastrtc import WebRTC
from ultralytics import YOLO
import threading

# Globals
prev_time = time.time()
fps = 0.0
alpha = 0.9  # smoothing factor

model = YOLO("model/yolov11l-face.pt")
model.to("cuda:0")
model.fuse()

# Shared detection results (protected by a lock)
latest_results = None
results_lock = threading.Lock()
frame_num = 0  # Track frame count globally

# Tracking thread class
class FaceTracking(threading.Thread):
    def __init__(self, frame):
        super().__init__()
        self.frame = frame

    def run(self):
        global latest_results
        with results_lock:
            results = model.track(source=self.frame, persist=True, verbose=False)
            latest_results = results

def main(frame: np.ndarray) -> np.ndarray:
    global prev_time, fps, latest_results, frame_num

    # Flip and copy frame
    frame = np.flip(frame, axis=1).copy()

    # Start tracking every 2nd frame (reduce load)
    tracking_thread = FaceTracking(frame.copy())
    tracking_thread.start()

    # Draw previous tracking results
    with results_lock:
        if latest_results:
            for result in latest_results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        track_id = int(box.id[0]) if box.id is not None else -1
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # Calculate FPS
    current_time = time.time()
    instantaneous_fps = 1 / (current_time - prev_time + 1e-8)
    fps = alpha * fps + (1 - alpha) * instantaneous_fps
    prev_time = current_time

    # Draw FPS
    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)

    frame_num = (frame_num + 1) % 100000  # Avoid overflow
    return frame

rtc_config = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}

with gr.Blocks() as demo:
    gr.Markdown("### Live Webcam with Threaded YOLOv11 Face Tracking & FPS")
    webrtc = WebRTC(mode="send-receive", modality="video", rtc_configuration=rtc_config)
    webrtc.stream(fn=main, inputs=[webrtc], outputs=[webrtc])

demo.launch(share=False)
