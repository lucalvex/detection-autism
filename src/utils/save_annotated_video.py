# ============================================================
# ARQUIVO: save_annotated_video.py
#
# O QUE FAZ: roda a YOLO11n-Pose em um vídeo inteiro e salva um novo
# vídeo (.mp4) com os keypoints/esqueleto desenhados em cada frame.
# É só para visualização/demonstração do processamento (ex.: mostrar
# na defesa do TCC como a extração de pose enxerga o vídeo) — não
# gera nenhum dado usado no treino do classificador.
# ============================================================

from ultralytics import YOLO
import cv2

VIDEO_PATH = "data/videos/v_HeadBanging_18.mp4"
OUTPUT_VIDEO = "data/results/annotated_video.mp4"

model = YOLO("models/yolo11n-pose.pt")

cap = cv2.VideoCapture(VIDEO_PATH)

# lê fps/resolução do vídeo de entrada para o vídeo de saída ficar com as mesmas propriedades
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
  OUTPUT_VIDEO,
  cv2.VideoWriter_fourcc(*"mp4v"),  # codec mp4
  fps,
  (width, height)
)

while cap.isOpened():

  ret, frame = cap.read()

  if not ret:
    break

  results = model(frame, verbose=False)

  # desenha os keypoints + esqueleto por cima do frame original
  annotated = results[0].plot()

  # grava o frame anotado no vídeo de saída
  writer.write(annotated)

cap.release()
writer.release()

print(f"Saved: {OUTPUT_VIDEO}")
