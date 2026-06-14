# demonstrar visualmente todo o processamento do vídeo
from ultralytics import YOLO
import cv2

VIDEO_PATH = "data/videos/v_HeadBanging_18.mp4"
OUTPUT_VIDEO = "data/results/annotated_video.mp4"

model = YOLO("models/yolo11n-pose.pt")

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
  OUTPUT_VIDEO,
  cv2.VideoWriter_fourcc(*"mp4v"),
  fps,
  (width, height)
)

while cap.isOpened():

  ret, frame = cap.read()

  if not ret:
    break

  results = model(frame, verbose=False)

  annotated = results[0].plot()

  writer.write(annotated)

cap.release()
writer.release()

print(f"Saved: {OUTPUT_VIDEO}")