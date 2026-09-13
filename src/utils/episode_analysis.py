# ============================================================
# ARQUIVO: episode_analysis.py
#
# O QUE FAZ: roda o mesmo pipeline de inferência de predict_video.py
# (pose + LSTM, janela deslizante de 30 frames) sobre um vídeo
# inteiro, mas em vez de exibir a predição ao vivo numa janela,
# guarda 1 predição de classe por frame e faz uma análise de
# EPISÓDIOS sobre essa sequência de predições:
#
# 1) Suavização temporal: a predição da LSTM frame a frame costuma
#    "piscar" entre classes por ruído momentâneo do classificador
#    (ex.: 1-2 frames isolados previstos como Normal no meio de um
#    episódio de ArmFlapping). Um filtro de moda (classe mais
#    frequente numa janela de tempo) suaviza essas oscilações antes
#    de segmentar os episódios.
# 2) Segmentação: agrupa frames consecutivos com a mesma classe
#    (após suavização) em um único episódio, com frame/tempo de
#    início e fim.
# 3) Filtro de duração mínima: descarta episódios mais curtos que
#    MIN_EPISODE_DURATION_SEC (ainda tratados como ruído residual).
# 4) Frequência e duração: conta quantos episódios de cada classe
#    ocorreram no vídeo (frequência) e calcula duração total, média,
#    mínima e máxima por classe.
#
# Saída em data/results/:
#   - episodes.csv         -> 1 linha por episódio detectado
#   - episode_summary.txt  -> frequência e duração agregadas por classe
#
# LIMITAÇÃO: como a predição em cada frame depende de uma janela de
# 30 frames anteriores, há uma latência inerente de até ~30 frames
# entre o início real de um comportamento e o momento em que a LSTM
# passa a prevê-lo — os tempos de início/fim dos episódios refletem
# isso.
# ============================================================

import json
import sys
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from ultralytics import YOLO

# permite importar pose_features.py de src/ mesmo rodando este script
# de dentro de src/utils/ (python src/utils/episode_analysis.py)
sys.path.append(str(Path(__file__).resolve().parent.parent))
from pose_features import flatten_features, normalize_frame_keypoints

# ==========================
# CONFIG
# ==========================

VIDEO_PATH = "data/videos/v_HeadBanging_18.mp4"
POSE_MODEL = "models/yolo11n-pose.pt"
CLASSIFIER_MODEL = "models/classifier.pt"
LABELS_FILE = "data/datasets/labels.json"
RESULTS_DIR = Path("data/results")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEQUENCE_LENGTH = 30          # precisa bater com o SEQUENCE_LENGTH usado em create_sequences.py
SMOOTHING_WINDOW = 15         # tamanho (em frames) da janela do filtro de moda -- ~0.5s a 30fps
MIN_EPISODE_DURATION_SEC = 0.5  # episódios mais curtos que isso são descartados como ruído

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

id_to_label = {v: k for k, v in labels.items()}
class_names = [id_to_label[i] for i in range(len(labels))]

# ==========================
# LSTM MODEL
# ==========================
# Mesma arquitetura de train_classifier.py/predict_video.py -- precisa
# bater exatamente com o que foi salvo em classifier.pt.

class LSTMClassifier(nn.Module):

  def __init__(self, input_size, hidden_size, num_layers, num_classes):
    super().__init__()

    self.lstm = nn.LSTM(
      input_size=input_size,
      hidden_size=hidden_size,
      num_layers=num_layers,
      batch_first=True
    )

    self.fc = nn.Linear(hidden_size, num_classes)

  def forward(self, x):

    _, (hidden, _) = self.lstm(x)

    return self.fc(hidden[-1])


classifier = LSTMClassifier(
  input_size=68,     # 17 keypoints * (x, y, dx, dy)
  hidden_size=128,
  num_layers=2,
  num_classes=len(labels)
)

classifier.load_state_dict(
  torch.load(CLASSIFIER_MODEL, map_location=device)
)

classifier.to(device)
classifier.eval()

print("Classifier loaded successfully")

pose_model = YOLO(POSE_MODEL)

print("YOLO Pose loaded successfully")

# ==========================
# 1. RODA POSE + LSTM SOBRE O VÍDEO INTEIRO
# ==========================
# Mesma lógica de predict_video.py (normalização + dx/dy + janela
# deslizante), mas sem exibir janela -- só acumula 1 predição por
# frame em raw_frame_ids / raw_predictions / raw_confidences.

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
  raise Exception(f"Unable to open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # fallback caso o vídeo não informe fps

buffer = deque(maxlen=SEQUENCE_LENGTH)
previous_kp = None

raw_frame_ids = []
raw_predictions = []
raw_confidences = []

frame_id = 0

while cap.isOpened():

  ret, frame = cap.read()

  if not ret:
    break

  results = pose_model(frame, verbose=False)

  for result in results:

    if result.keypoints is None:
      continue

    people = result.keypoints.xy.cpu().numpy()

    if len(people) == 0:
      continue

    kp = people[0]

    # normalização (centro do quadril + escala pelos ombros) e
    # features de movimento (dx/dy) -- mesma função usada por
    # extract_pose.py e predict_video.py
    features, previous_kp = normalize_frame_keypoints(kp, previous_kp)

    buffer.append(flatten_features(features))

    if len(buffer) == SEQUENCE_LENGTH:

      sequence = torch.tensor(
        np.array(buffer, dtype=np.float32),
        dtype=torch.float32
      ).unsqueeze(0).to(device)

      with torch.no_grad():

        output = classifier(sequence)
        probabilities = torch.softmax(output, dim=1)

        confidence = torch.max(probabilities).item()
        prediction = torch.argmax(probabilities, dim=1).item()

      # a predição corresponde ao frame atual (última posição da janela)
      raw_frame_ids.append(frame_id)
      raw_predictions.append(prediction)
      raw_confidences.append(confidence)

  frame_id += 1

cap.release()

print(f"Processed {frame_id} frames, {len(raw_predictions)} predictions collected")

if not raw_predictions:
  raise RuntimeError(
    "No predictions collected -- vídeo muito curto (menor que "
    f"SEQUENCE_LENGTH={SEQUENCE_LENGTH} frames) ou nenhuma pessoa detectada."
  )

# ==========================
# 2. SUAVIZAÇÃO TEMPORAL (filtro de moda)
# ==========================

def smooth_predictions(predictions, window):
  """Substitui cada predição pela classe mais frequente numa janela
  centrada ao seu redor, reduzindo oscilações de 1-2 frames causadas
  por ruído momentâneo do classificador."""

  n = len(predictions)
  half = window // 2
  smoothed = []

  for i in range(n):
    lo = max(0, i - half)
    hi = min(n, i + half + 1)
    most_common = Counter(predictions[lo:hi]).most_common(1)[0][0]
    smoothed.append(most_common)

  return smoothed


smoothed_predictions = smooth_predictions(raw_predictions, SMOOTHING_WINDOW)

# ==========================
# 3. SEGMENTAÇÃO EM EPISÓDIOS
# ==========================
# Agrupa frames consecutivos (na sequência já suavizada) com a mesma
# classe prevista em um único episódio.

episodes = []
start_idx = 0

for i in range(1, len(smoothed_predictions) + 1):

  is_last = i == len(smoothed_predictions)

  if is_last or smoothed_predictions[i] != smoothed_predictions[start_idx]:

    start_frame = raw_frame_ids[start_idx]
    end_frame = raw_frame_ids[i - 1]

    episodes.append({
      "class_id": smoothed_predictions[start_idx],
      "start_frame": start_frame,
      "end_frame": end_frame,
      "start_time_sec": start_frame / fps,
      "end_time_sec": (end_frame + 1) / fps,
      "duration_sec": (end_frame + 1 - start_frame) / fps,
    })

    start_idx = i

# descarta episódios mais curtos que o limiar mínimo (tratados como ruído residual)
episodes = [e for e in episodes if e["duration_sec"] >= MIN_EPISODE_DURATION_SEC]

if not episodes:
  raise RuntimeError(
    "Nenhum episódio sobrou após o filtro de duração mínima "
    f"({MIN_EPISODE_DURATION_SEC}s) -- tente reduzir MIN_EPISODE_DURATION_SEC."
  )

episodes_df = pd.DataFrame(episodes)
episodes_df["class_name"] = episodes_df["class_id"].map(id_to_label)
episodes_df.insert(0, "episode_id", range(1, len(episodes_df) + 1))

episodes_df = episodes_df[[
  "episode_id", "class_name", "start_frame", "end_frame",
  "start_time_sec", "end_time_sec", "duration_sec"
]]

episodes_df.to_csv(RESULTS_DIR / "episodes.csv", index=False)

print(f"\n{len(episodes_df)} episodes saved to {RESULTS_DIR / 'episodes.csv'}")

# ==========================
# 4. FREQUÊNCIA E DURAÇÃO AGREGADAS POR CLASSE
# ==========================

video_duration_min = (frame_id / fps) / 60

summary = (
  episodes_df
  .groupby("class_name")["duration_sec"]
  .agg(episodes="count", total_duration_sec="sum", mean_duration_sec="mean",
       min_duration_sec="min", max_duration_sec="max")
  .reindex(class_names)   # garante 1 linha por classe, mesmo sem episódios (fica com NaN/0)
  .fillna(0)
)

summary["episodes_per_min"] = summary["episodes"] / video_duration_min

summary_text_lines = [
  f"Episode analysis -- {VIDEO_PATH}",
  f"Video duration: {frame_id / fps:.1f}s ({video_duration_min:.2f} min)",
  f"Smoothing window: {SMOOTHING_WINDOW} frames | Min episode duration: {MIN_EPISODE_DURATION_SEC}s",
  "",
  summary.round(3).to_string(),
]

summary_text = "\n".join(summary_text_lines)

print("\n" + summary_text)

with open(RESULTS_DIR / "episode_summary.txt", "w", encoding="utf-8") as f:
  f.write(summary_text + "\n")

print(f"\nSummary saved to {RESULTS_DIR / 'episode_summary.txt'}")

# versão em JSON (metadados + resumo agregado), para consumo programático
# (usado por generate_report.py)

episode_summary = {
  "video_path": VIDEO_PATH,
  "video_duration_sec": frame_id / fps,
  "fps": fps,
  "smoothing_window": SMOOTHING_WINDOW,
  "min_episode_duration_sec": MIN_EPISODE_DURATION_SEC,
  "per_class": {
    class_name: {
      "episodes": int(row["episodes"]),
      "total_duration_sec": float(row["total_duration_sec"]),
      "mean_duration_sec": float(row["mean_duration_sec"]),
      "min_duration_sec": float(row["min_duration_sec"]),
      "max_duration_sec": float(row["max_duration_sec"]),
      "episodes_per_min": float(row["episodes_per_min"]),
    }
    for class_name, row in summary.iterrows()
  },
}

with open(RESULTS_DIR / "episode_summary.json", "w", encoding="utf-8") as f:
  json.dump(episode_summary, f, indent=2, ensure_ascii=False)
