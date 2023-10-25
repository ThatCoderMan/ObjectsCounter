from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('../models/best.pt')
model.conf=0.05
# Open the video file
video_path = "../data/videos/Seno1.mp4"
cap = cv2.VideoCapture(video_path)


# Store the track history
counter = set()

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if success:
        results = model.track(frame, persist=True, imgsz=(1920, 1080))

        track_ids = results[0].boxes.id.int().cpu().tolist()
        boxes = results[0].boxes.xywh.cpu()
        # annotated_frame = results[0].plot()
        annotated_frame = frame.copy()
        for box in boxes:
            x, y, w, h = box
            cv2.rectangle(
                annotated_frame,
                (int(x - w / 2), int(y - h / 2)),
                (int(x + w / 2), int(y + h / 2)),
                (0, 255, 0),
                2
            )

        counter.update(track_ids)
        print(len(counter))
        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

print("Total objects detected:", len(counter))