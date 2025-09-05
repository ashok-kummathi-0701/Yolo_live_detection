import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import math

# Load YOLOv8 model
model = YOLO("yolo-Weights/yolov8n.pt")  # Ensure this path is correct and the file exists

# Object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Streamlit UI
st.title("🔍 Real-Time Object Detection with YOLOv8 + Streamlit")
st.write("Upload a video or use webcam for live detection.")

# File upload option
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

# Webcam toggle
use_webcam = st.checkbox("Use Webcam")

# Function to process video frames
def detect_objects(frame):
    results = model(frame, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Extract coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

            # Confidence
            confidence = math.ceil((box.conf[0] * 100)) / 100

            # Class name
            cls = int(box.cls[0])
            if 0 <= cls < len(classNames):
                label = f"{classNames[cls]} {confidence}"
            else:
                label = f"Unknown {confidence}"

            # Put label on frame
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    return frame

# If webcam is selected
if use_webcam:
    stframe = st.empty()
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_objects(frame)

        # Convert BGR → RGB for Streamlit
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

# If a file is uploaded and webcam is not selected
elif uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    tfile.close()
    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_objects(frame)

        # Show frame
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
