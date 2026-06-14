# ver os keypoints do COCO sendo desenhados em tempo real enquanto o vídeo está rodando
import cv2
from ultralytics import YOLO

VIDEO_PATH = "data/videos/v_HeadBanging_18.mp4"
MODEL_PATH = "models/yolo11n-pose.pt"

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

while cap.isOpened():

  ret, frame = cap.read()

  if not ret:
    break

  results = model(
    frame,
    verbose=False
  )

  annotated_frame = results[0].plot()

  cv2.imshow(
    "YOLO Pose - COCO Keypoints",
    annotated_frame
  )

  key = cv2.waitKey(1)

  if key == 27:  # ESC
    break

cap.release()
cv2.destroyAllWindows()