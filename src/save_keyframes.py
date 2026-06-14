# capturar vários momentos do comportamento
from ultralytics import YOLO
import cv2
from pathlib import Path

VIDEO_PATH = "data/videos/v_HeadBanging_18.mp4"

OUTPUT_DIR = Path("data/results/keyframes")
OUTPUT_DIR.mkdir(
  parents=True,
  exist_ok=True
)

SAVE_EVERY = 100

model = YOLO("models/yolo11n-pose.pt")

cap = cv2.VideoCapture(VIDEO_PATH)

frame_id = 0

while cap.isOpened():

  ret, frame = cap.read()

  if not ret:
    break

  results = model(frame, verbose=False)

  annotated = results[0].plot()

  if frame_id % SAVE_EVERY == 0:

    filename = (
      OUTPUT_DIR /
      f"frame_{frame_id:05d}.png"
    )

    cv2.imwrite(
      str(filename),
      annotated
    )

    print(
      f"Saved {filename}"
    )

  frame_id += 1

cap.release()

print("Finished")