# mostrar um frame com os 17 keypoints do COCO.
from pathlib import Path

import cv2
from ultralytics import YOLO

# ==========================
# CONFIG
# ==========================

IMAGE_PATH = "data/results/keyframes/frame_00000.png"
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

results = model(
  IMAGE_PATH,
  verbose=False
)

result = results[0]

# Desenha keypoints + esqueleto COCO
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