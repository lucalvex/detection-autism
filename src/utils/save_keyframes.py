# ============================================================
# ARQUIVO: save_keyframes.py
#
# O QUE FAZ: percorre um vídeo e salva como PNG, a cada N frames
# (SAVE_EVERY), o frame já anotado com os keypoints/esqueleto da
# YOLO11n-Pose. Serve para capturar "fotos" representativas de um
# comportamento ao longo do vídeo (ex.: para ilustrar no TCC), sem
# precisar salvar o vídeo inteiro. Não alimenta o treino do modelo.
# ============================================================

from ultralytics import YOLO
import cv2
from pathlib import Path

VIDEO_PATH = "data/videos/v_ArmFlapping_01.mp4"

OUTPUT_DIR = Path("data/results/keyframes")
OUTPUT_DIR.mkdir(
  parents=True,
  exist_ok=True
)

SAVE_EVERY = 100  # salva 1 imagem a cada 100 frames processados

model = YOLO("models/yolo11n-pose.pt")

cap = cv2.VideoCapture(VIDEO_PATH)

frame_id = 0

while cap.isOpened():

  ret, frame = cap.read()

  if not ret:
    break

  results = model(frame, verbose=False)

  # desenha os keypoints + esqueleto por cima do frame original
  annotated = results[0].plot()

  # só salva em disco quando o índice do frame é múltiplo de SAVE_EVERY
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
