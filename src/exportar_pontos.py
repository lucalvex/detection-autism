# ============================================================
# ARQUIVO: exportar_pontos.py
#
# O QUE FAZ: gera <id>_pontos.json (coordenadas de pose em PIXELS
# BRUTOS do vídeo original, formato de tcc-demo-handoff/HANDOFF.md),
# para o overlay de esqueleto da interface. Roda o YOLO11n-Pose de
# novo em cada vídeo -- NÃO TREINA NADA, só extrai keypoints; é uma
# etapa separada de exportar_sessao.py porque data/poses/*.csv só tem
# as coordenadas já normalizadas (centralizadas no quadril, escaladas
# pelos ombros), usadas para classificação -- os pixels brutos nunca
# foram salvos.
#
# MESMA REGRA DE ESCOLHA DE PESSOA de extract_pose.py: assume 1 pessoa
# por vídeo, usa a PRIMEIRA detectada pelo YOLO em cada quadro
# (`people[0]`), sem rastreamento entre quadros. Quadro sem pessoa
# detectada -> null (sem interpolar, sem repetir o quadro anterior --
# a interface deve mostrar esqueleto ausente nesses quadros, não
# inventar um).
#
# Uso: python src/exportar_pontos.py <video_id> --saida saidas/
# ============================================================

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

VIDEOS_DIR = Path("data/videos")
POSE_MODEL = "models/yolo11n-pose.pt"


def exportar_pontos(video_id, saida_dir):

  model = YOLO(POSE_MODEL)

  cap = cv2.VideoCapture(str(VIDEOS_DIR / f"{video_id}.mp4"))

  if not cap.isOpened():
    raise RuntimeError(f"Não consegui abrir {video_id}.mp4")

  quadros = []
  frame_id = 0
  n_detectados = 0

  while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
      break

    results = model(frame, verbose=False)

    ponto_frame = None

    for result in results:

      if result.keypoints is None:
        continue

      people_xy = result.keypoints.xy.cpu().numpy()

      if len(people_xy) == 0:
        continue

      # mesma regra de extract_pose.py: assume 1 pessoa, usa a primeira detectada
      kp_xy = people_xy[0]

      people_conf = result.keypoints.conf
      kp_conf = people_conf.cpu().numpy()[0] if people_conf is not None else [None] * len(kp_xy)

      ponto_frame = [
        [float(x), float(y), (float(c) if c is not None else None)]
        for (x, y), c in zip(kp_xy, kp_conf)
      ]

    if ponto_frame is not None:
      n_detectados += 1

    quadros.append(ponto_frame)  # None se este quadro nao teve deteccao
    frame_id += 1

  cap.release()

  saida = {
    "versao": 1,
    "formato": "coco17",
    "ordem": "[x_px, y_px, confianca] por ponto",
    "quadros": quadros,
  }

  saida_dir = Path(saida_dir)
  saida_dir.mkdir(parents=True, exist_ok=True)

  saida_path = saida_dir / f"{video_id}_pontos.json"

  with open(saida_path, "w", encoding="utf-8") as f:
    json.dump(saida, f, ensure_ascii=False)

  print(f"{video_id}: {n_detectados} quadros detectados de {frame_id} -> {saida_path}")

  return saida_path, frame_id, n_detectados


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("video_id", help="ex.: v_ArmFlapping_12 (sem extensão)")
  parser.add_argument("--saida", default="saidas", help="pasta de saída (default: saidas/)")

  args = parser.parse_args()

  exportar_pontos(args.video_id, args.saida)
