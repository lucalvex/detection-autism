# classificação em tempo real

# saber qual é o tamanho de entrada para o classificador?
# qual o tempo de treinamento? usar GPU
import json
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

# ==========================
# CONFIG
# ==========================

VIDEO_PATH = "data/videos/v_HeadBanging_18.mp4"
POSE_MODEL = "models/yolo11n-pose.pt"
CLASSIFIER_MODEL = "models/classifier.pt"
LABELS_FILE = "data/datasets/labels.json"

SEQUENCE_LENGTH = 30

# ==========================
# DEVICE
# ==========================

device = torch.device(
  "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

# ==========================
# LOAD LABELS
# ==========================

with open(LABELS_FILE, "r", encoding="utf-8") as f:
  labels = json.load(f)

id_to_label = {
  value: key
  for key, value in labels.items()
}

# ==========================
# LSTM MODEL
# ==========================

class LSTMClassifier(nn.Module):

  def __init__(
    self,
    input_size,
    hidden_size,
    num_layers,
    num_classes
  ):
    super().__init__()

    self.lstm = nn.LSTM(
      input_size=input_size,
      hidden_size=hidden_size,
      num_layers=num_layers,
      batch_first=True
    )

    self.fc = nn.Linear(
      hidden_size,
      num_classes
    )

  def forward(self, x):

    _, (hidden, _) = self.lstm(x)

    hidden = hidden[-1]

    return self.fc(hidden)

# ==========================
# LOAD CLASSIFIER
# ==========================

classifier = LSTMClassifier(
  input_size=34,     # 17 keypoints * 2 coordenadas
  hidden_size=128,
  num_layers=2,
  num_classes=len(labels)
)

classifier.load_state_dict(
  torch.load(
    CLASSIFIER_MODEL,
    map_location=device
  )
)

classifier.to(device)
classifier.eval()

print("Classifier loaded successfully")

# ==========================
# LOAD YOLO POSE
# ==========================

pose_model = YOLO(POSE_MODEL)

print("YOLO Pose loaded successfully")

# ==========================
# VIDEO
# ==========================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
  raise Exception(
    f"Unable to open video: {VIDEO_PATH}"
  )

buffer = deque(maxlen=SEQUENCE_LENGTH)

while cap.isOpened():

  ret, frame = cap.read()

  if not ret:
    break

  prediction_text = "Collecting frames..."

  results = pose_model(
    frame,
    verbose=False
  )

  for result in results:

    if result.keypoints is None:
      continue

    people = result.keypoints.xy.cpu().numpy()

    if len(people) == 0:
      continue

    # primeira pessoa detectada
    kp = people[0]

    frame_features = []

    for x, y in kp:

      frame_features.extend([
        float(x),
        float(y)
      ])

    buffer.append(frame_features)

    if len(buffer) == SEQUENCE_LENGTH:

      sequence = np.array(
        buffer,
        dtype=np.float32
      )

      sequence = torch.tensor(
        sequence,
        dtype=torch.float32
      )

      sequence = sequence.unsqueeze(0)

      sequence = sequence.to(device)

      with torch.no_grad():

        output = classifier(sequence)

        probabilities = torch.softmax(
          output,
          dim=1
        )

        confidence = torch.max(
          probabilities
        ).item()

        prediction = torch.argmax(
          probabilities,
          dim=1
        ).item()

        label = id_to_label[prediction]

        prediction_text = (
          f"{label} "
          f"({confidence*100:.1f}%)"
        )

  cv2.putText(
    frame,
    prediction_text,
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2
  )

  cv2.imshow(
    "Autism Behavior Detection",
    frame
  )

  key = cv2.waitKey(1)

  if key == 27:
    break

cap.release()
cv2.destroyAllWindows()