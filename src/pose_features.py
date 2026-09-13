# ============================================================
# ARQUIVO: pose_features.py
#
# O QUE FAZ: função compartilhada de normalização de pose, usada por
# extract_pose.py (geração do dataset), predict_video.py e
# episode_analysis.py (inferência) -- para garantir que o mesmo
# cálculo (centralização no quadril + escala pelos ombros + features
# de movimento dx/dy) seja aplicado igual em treino e em inferência.
# Não é um script executável; só define a função abaixo.
# ============================================================

import numpy as np


def normalize_frame_keypoints(kp, previous_kp):
  """Normaliza os 17 keypoints COCO de 1 frame e calcula o movimento
  em relação ao frame anterior.

  kp: array (17, 2) com as coordenadas (x, y) brutas em pixels.
  previous_kp: lista de 17 tuplas (x, y) já normalizadas do frame
    anterior, ou None se este for o primeiro frame do vídeo.

  Retorna (features, current_kp):
    features: lista de 17 tuplas (x, y, dx, dy) normalizadas, uma por
      keypoint, na ordem usada em todo o pipeline (x, y, dx, dy por
      keypoint, keypoints 0 a 16).
    current_kp: lista de 17 tuplas (x, y) normalizadas deste frame --
      passe como `previous_kp` na chamada do próximo frame.
  """

  # Centro do quadril: ponto de referência para centralizar a pose
  # (kp[11] = quadril esquerdo, kp[12] = quadril direito, índices COCO)
  hip_x = (kp[11][0] + kp[12][0]) / 2
  hip_y = (kp[11][1] + kp[12][1]) / 2

  left_shoulder = kp[5]
  right_shoulder = kp[6]

  # distância entre ombros = "régua" para normalizar a escala
  # (pessoa mais perto/longe da câmera tem tamanhos de pixel diferentes,
  # mas a pose relativa ao próprio corpo deve ser comparável)
  body_size = np.sqrt(
    (right_shoulder[0] - left_shoulder[0]) ** 2 +
    (right_shoulder[1] - left_shoulder[1]) ** 2
  )

  if body_size < 1:
    body_size = 1  # evita divisão por zero se os ombros forem mal detectados

  current_kp = []
  features = []

  for i, (x, y) in enumerate(kp):

    # normalização: centraliza no quadril e escala pela distância entre ombros
    x = (x - hip_x) / body_size
    y = (y - hip_y) / body_size

    current_kp.append((x, y))

    if previous_kp is None:
      # primeiro frame do vídeo: não há frame anterior para calcular movimento
      dx = 0.0
      dy = 0.0

    else:
      # feature de movimento: quanto esse keypoint se deslocou desde o frame anterior
      dx = x - previous_kp[i][0]
      dy = y - previous_kp[i][1]

    features.append((float(x), float(y), float(dx), float(dy)))

  return features, current_kp


def flatten_features(features):
  """Achata a lista de 17 tuplas (x, y, dx, dy) em uma lista plana de
  68 floats, na mesma ordem usada para montar as sequências de
  treino (create_sequences.py)."""

  flat = []

  for x, y, dx, dy in features:
    flat.extend([x, y, dx, dy])

  return flat
