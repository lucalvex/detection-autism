# ============================================================
# ARQUIVO: realtime_pose.py
#
# O QUE FAZ: reproduz um vídeo frame a frame numa janela, desenhando
# ao vivo os 17 keypoints COCO + esqueleto detectados pela
# YOLO11n-Pose (sem nenhuma classificação de comportamento — é só
# visualização da pose bruta). Para ver a CLASSIFICAÇÃO do
# comportamento (ArmFlapping, HeadBanging etc.) em vídeo, use
# predict_video.py.
# ============================================================

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

  # desenha os keypoints + esqueleto por cima do frame original
  annotated_frame = results[0].plot()

  cv2.imshow(
    "YOLO Pose - COCO Keypoints",
    annotated_frame
  )

  key = cv2.waitKey(1)

  if key == 27:  # ESC fecha a janela e encerra o loop
    break

cap.release()
cv2.destroyAllWindows()
