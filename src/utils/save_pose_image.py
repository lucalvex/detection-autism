# ============================================================
# ARQUIVO: save_pose_image.py
#
# O QUE FAZ: pega uma imagem única (por padrão, um keyframe já salvo
# por save_keyframes.py) e desenha em cima dela os 17 keypoints COCO +
# o esqueleto detectado pela YOLO11n-Pose, salvando o resultado como
# PNG. Também imprime no console as coordenadas cruas de cada pessoa
# detectada. É uma ferramenta de inspeção visual, usada para
# ilustrações no TCC — não faz parte do pipeline de treino.
# ============================================================

from pathlib import Path

import cv2
from ultralytics import YOLO

# ==========================
# CONFIG
# ==========================

IMAGE_PATH = "data/results/keyframes/frame_00000.png"  # imagem de entrada (1 frame já extraído)
MODEL_PATH = "models/yolo11n-pose.pt"

OUTPUT_DIR = Path("data/results/images")
OUTPUT_DIR.mkdir(
  parents=True,
  exist_ok=True
)

OUTPUT_IMAGE = OUTPUT_DIR / "pose_image.png"

# ==========================
# LOAD MODEL
# ==========================

model = YOLO(MODEL_PATH)

# ==========================
# POSE DETECTION
# ==========================

# roda a pose estimation numa única imagem (não é vídeo, então results tem 1 elemento só)
results = model(
  IMAGE_PATH,
  verbose=False
)

result = results[0]

# .plot() é um utilitário da ultralytics que desenha os keypoints + as
# conexões do esqueleto COCO por cima da imagem original
annotated_image = result.plot()

# ==========================
# SAVE IMAGE
# ==========================

cv2.imwrite(
  str(OUTPUT_IMAGE),
  annotated_image
)

# ==========================
# INFO
# ==========================

if result.keypoints is not None:

  # coordenadas (x, y) em pixels, sem nenhuma normalização — 1 array de 17 pontos por pessoa detectada
  keypoints = result.keypoints.xy.cpu().numpy()

  print(f"People detected: {len(keypoints)}")

  for person_id, person in enumerate(keypoints):

    print(
      f"\nPerson {person_id + 1}"
    )

    print(
      f"Keypoints shape: {person.shape}"
    )

    print(person)

print(
  f"\nImage saved to: {OUTPUT_IMAGE}"
)
