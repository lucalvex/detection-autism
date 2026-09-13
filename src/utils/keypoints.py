# ============================================================
# ARQUIVO: keypoints.py
#
# O QUE FAZ: script de teste rápido — roda a YOLO11n-Pose em um único
# vídeo e imprime no console as coordenadas (x, y) dos 17 keypoints
# COCO detectados por frame. Não gera nenhum arquivo de saída; serve
# só para inspecionar rapidamente o formato bruto que o modelo de
# pose devolve, antes de qualquer normalização.
# ============================================================

# Os keypoints são pontos anatômicos que o modelo de pose estimation identifica no corpo de pessoa
from ultralytics import YOLO

# carrega os pesos da YOLO11n-Pose (modelo pré-treinado em COCO, 17 keypoints por pessoa)
model = YOLO("yolo11n-pose.pt")

# roda a inferência de pose no vídeo inteiro; 'results' é 1 objeto por frame
results = model("v_HeadBanging_18.mp4")

for result in results:
  # .keypoints.xy = coordenadas (x, y) em pixels de cada keypoint, para cada pessoa detectada no frame
  print(result.keypoints.xy)
