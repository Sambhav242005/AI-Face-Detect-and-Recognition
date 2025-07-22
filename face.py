import streamlit as st
import cv2
import numpy as np

# Set page configuration
st.set_page_config(page_title="Webcam Stream", layout="centered")

# Title
st.title("🔴 Live Webcam Frame Processing")

# Start webcam
run = st.checkbox("Start Webcam")

# Placeholder for the image
frame_placeholder = st.empty()

if run:
    cap = cv2.VideoCapture(0)  # 0 is the default webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to read from webcam.")
            break

        # Process the frame (e.g., convert to grayscale)
        processed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

        # Resize for display (optional)
        resized = cv2.resize(processed, (640, 480))

        # Convert BGR (OpenCV) to RGB (Streamlit expects)
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Show in Streamlit
        frame_placeholder.image(rgb_frame, channels="RGB")

        # Exit loop if checkbox is turned off
        if not st.session_state.get("Start Webcam", True):
            break

    cap.release()
