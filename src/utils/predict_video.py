# ============================================================
# ARQUIVO: predict_video.py
#
# O QUE FAZ: aplica o pipeline completo de inferência em um vídeo:
# 1) YOLO11n-Pose extrai os keypoints de cada frame ao vivo;
# 2) os keypoints dos últimos 30 frames são acumulados num buffer
#    (janela deslizante, mesmo tamanho usado no treino);
# 3) assim que o buffer enche, a sequência é passada para a LSTM
#    já treinada (models/classifier.pt), que prevê a classe de
#    comportamento (ArmFlapping, HeadBanging, etc.);
# 4) o rótulo previsto + confiança são desenhados em cima do vídeo,
#    exibido ao vivo numa janela.
#
# As features por frame (normalização + dx/dy) replicam exatamente o
# que extract_pose.py faz, para que a entrada da LSTM em inferência
# tenha a mesma distribuição/escala dos dados usados no treino.
# ============================================================

import json
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

# permite importar pose_features.py de src/ mesmo rodando este script
# de dentro de src/utils/ (python src/utils/predict_video.py)
sys.path.append(str(Path(__file__).resolve().parent.parent))
from pose_features import flatten_features, normalize_frame_keypoints

# ==========================
# CONFIG
# ==========================

VIDEO_PATH = "data/videos/v_HeadBanging_18.mp4"
POSE_MODEL = "models/yolo11n-pose.pt"
CLASSIFIER_MODEL = "models/classifier.pt"
LABELS_FILE = "data/datasets/labels.json"

SEQUENCE_LENGTH = 30  # precisa bater com o SEQUENCE_LENGTH usado em create_sequences.py

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

# inverte {classe: id} para {id: classe}, usado para traduzir a predição (um número) de volta pro nome da classe
id_to_label = {
  value: key
  for key, value in labels.items()
}

# ==========================
# LSTM MODEL
# ==========================
# Precisa ser exatamente a mesma arquitetura usada em train_classifier.py,
# senão os pesos salvos em classifier.pt não batem com o modelo aqui.

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

    # pega o estado oculto da última camada da LSTM (resumo da sequência inteira)
    hidden = hidden[-1]

    return self.fc(hidden)

# ==========================
# LOAD CLASSIFIER
# ==========================

classifier = LSTMClassifier(
  input_size=68,     # 17 keypoints * (x, y, dx, dy) -- mesmo formato gerado por extract_pose.py
  hidden_size=128,
  num_layers=2,
  num_classes=len(labels)
)

# carrega os pesos treinados por train_classifier.py
classifier.load_state_dict(
  torch.load(
    CLASSIFIER_MODEL,
    map_location=device
  )
)

classifier.to(device)
classifier.eval()  # desliga dropout/batchnorm (se existissem) — modo de inferência

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

# buffer circular: guarda só os últimos SEQUENCE_LENGTH frames de keypoints (janela deslizante ao vivo)
buffer = deque(maxlen=SEQUENCE_LENGTH)

previous_kp = None  # keypoints normalizados do frame anterior, para calcular dx/dy

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

    # normalização (centro do quadril + escala pelos ombros) e
    # features de movimento (dx/dy) -- mesma função usada por
    # extract_pose.py, para a entrada da LSTM ficar na mesma escala
    # dos dados de treino
    features, previous_kp = normalize_frame_keypoints(kp, previous_kp)

    buffer.append(flatten_features(features))

    # só classifica quando já temos os 30 frames necessários para formar uma sequência
    if len(buffer) == SEQUENCE_LENGTH:

      sequence = np.array(
        buffer,
        dtype=np.float32
      )

      sequence = torch.tensor(
        sequence,
        dtype=torch.float32
      )

      # adiciona a dimensão de batch (LSTM espera [batch, seq_len, features])
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

  # desenha a predição atual em cima do frame exibido
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

  if key == 27:  # ESC encerra
    break

cap.release()
cv2.destroyAllWindows()
