import cv2
import mediapipe as mp

from mediapipe.tasks.python import holistic

mp_holistic = holistic

holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture("video.mp4")

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = holistic_model.process(rgb)

    print(results)

cap.release()