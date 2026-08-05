# gera X.npy e y.npy

import json
from pathlib import Path
import numpy as np
import pandas as pd

POSES_DIR = Path("data/poses")
OUTPUT_DIR = Path("data/datasets")

SEQUENCE_LENGTH = 30

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Classes do seu dataset
LABELS = {
  "ArmFlapping": 0,
  "HeadBanging": 1,
  "Rocking": 2,
  "HandMovement": 3,
  "Spinning": 4,
  "Normal": 5
}

X = []
y = []
groups = []  # video_id de origem de cada sequência (para split por vídeo)

for csv_file in POSES_DIR.glob("*.csv"):

  print(f"Processing {csv_file.name}")

  label = None

  for class_name, class_id in LABELS.items():

    if class_name.lower() in csv_file.stem.lower():
      label = class_id
      break

  if label is None:
    print(
      f"Skipping {csv_file.name} "
      f"(label not found)"
    )
    continue

  df = pd.read_csv(csv_file)

  # Remove colunas que não serão usadas
  columns_to_remove = []

  if "frame" in df.columns:
    columns_to_remove.append("frame")

  if "person" in df.columns:
    columns_to_remove.append("person")

  df = df.drop(columns=columns_to_remove)

  features = df.values

  if len(features) < SEQUENCE_LENGTH:
    print(
      f"Skipping {csv_file.name} "
      f"(not enough frames)"
    )
    continue

  video_id = csv_file.stem

  for start in range(
    len(features) - SEQUENCE_LENGTH + 1
  ):

    end = start + SEQUENCE_LENGTH

    sequence = features[start:end]

    X.append(sequence)
    y.append(label)
    groups.append(video_id)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)
groups = np.array(groups)

np.save(OUTPUT_DIR / "X.npy", X)
np.save(OUTPUT_DIR / "y.npy", y)
np.save(OUTPUT_DIR / "groups.npy", groups)

with open(
  OUTPUT_DIR / "labels.json",
  "w",
  encoding="utf-8"
) as f:
  json.dump(
    LABELS,
    f,
    indent=4
  )

print("\nDataset created successfully")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"groups shape: {groups.shape}")
print(f"Unique videos: {len(np.unique(groups))}")