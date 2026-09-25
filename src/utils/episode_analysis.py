# ============================================================
# ARQUIVO: episode_analysis.py
#
# O QUE FAZ: roda o mesmo pipeline de inferência de predict_video.py
# (pose + LSTM, janela deslizante de 30 frames) sobre um vídeo
# inteiro, mas em vez de exibir a predição ao vivo numa janela,
# guarda a pontuação softmax de TODAS as classes em cada janela e
# constrói "trechos" (segmentos de tempo) usando a mesma regra do
# design da interface (HANDOFF.md / episodes() de revisao.dc.html),
# implementada em src/timeline.py:
#
#   Um trecho de uma classe é uma sequência de janelas consecutivas
#   com pontuação >= limiar naquela classe. Vai do quadro_inicio da
#   primeira janela ao quadro_fim da última, e só entra na lista se
#   durar pelo menos MIN_DURATION_SEC. O pico é a maior pontuação
#   entre as janelas do trecho.
#
# Frequência e duração: conta quantos trechos de cada classe
# ocorreram no vídeo (frequência) e calcula duração total, média,
# mínima e máxima por classe.
#
# Saída em data/results/:
#   - episodes.csv         -> 1 linha por trecho detectado
#   - episode_summary.txt  -> frequência e duração agregadas por classe
#
# LIMITAÇÃO: como a predição em cada janela depende de 30 frames, há
# uma latência inerente entre o início real de um comportamento e o
# momento em que a LSTM passa a prevê-lo -- mas como o trecho é
# medido do quadro_inicio da 1ª janela ativa (não do frame em que a
# predição "aconteceu"), essa latência já fica parcialmente
# compensada na própria definição do trecho.
#
# SCORE_THRESHOLD e MIN_DURATION_SEC abaixo são PROVISÓRIOS (0.5 e
# 0.5s) -- os valores finais serão escolhidos por uma varredura
# limiar x duração mínima contra F1 por evento nas predições
# out-of-fold (ver train_oof.py / evaluate_temporal.py), não por
# este arquivo.
# ============================================================

import json
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from ultralytics import YOLO

# permite importar pose_features.py/timeline.py de src/ mesmo rodando
# este script de dentro de src/utils/ (python src/utils/episode_analysis.py)
sys.path.append(str(Path(__file__).resolve().parent.parent))
from pose_features import flatten_features, normalize_frame_keypoints
from timeline import build_segments

# ==========================
# CONFIG
# ==========================

VIDEO_PATH = "data/videos/v_HeadBanging_18.mp4"
POSE_MODEL = "models/yolo11n-pose.pt"
CLASSIFIER_MODEL = "models/classifier.pt"
LABELS_FILE = "data/datasets/labels.json"
RESULTS_DIR = Path("data/results")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEQUENCE_LENGTH = 30       # precisa bater com o SEQUENCE_LENGTH usado em create_sequences.py
SCORE_THRESHOLD = 0.5      # provisório -- ver aviso no cabeçalho
MIN_DURATION_SEC = 0.5     # provisório -- ver aviso no cabeçalho (é um no-op em fps <= 60,
                           # já que a menor janela possível mede SEQUENCE_LENGTH/fps segundos)

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

windows = []  # 1 dict por janela: {"quadro_inicio", "quadro_fim", "pontuacoes"}

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
        probabilities = torch.softmax(output, dim=1).cpu().numpy()[0].tolist()

      # a janela cobre [quadro_inicio, quadro_fim) -- mesma convenção de
      # create_sequences.py (end = start + SEQUENCE_LENGTH, slice exclusivo)
      quadro_fim = frame_id + 1
      quadro_inicio = quadro_fim - SEQUENCE_LENGTH

      windows.append({
        "quadro_inicio": quadro_inicio,
        "quadro_fim": quadro_fim,
        "pontuacoes": probabilities,
      })

  frame_id += 1

cap.release()

print(f"Processed {frame_id} frames, {len(windows)} windows collected")

if not windows:
  raise RuntimeError(
    "No windows collected -- vídeo muito curto (menor que "
    f"SEQUENCE_LENGTH={SEQUENCE_LENGTH} frames) ou nenhuma pessoa detectada."
  )

# ==========================
# 2. SEGMENTAÇÃO EM TRECHOS (regra única -- ver src/timeline.py)
# ==========================

segments = build_segments(
  windows,
  class_names,
  threshold=SCORE_THRESHOLD,
  min_duration_sec=MIN_DURATION_SEC,
  fps=fps
)

if not segments:
  raise RuntimeError(
    f"Nenhum trecho encontrado com limiar={SCORE_THRESHOLD} e "
    f"duração mínima={MIN_DURATION_SEC}s -- tente reduzir SCORE_THRESHOLD."
  )

episodes_df = pd.DataFrame(segments)
episodes_df.insert(0, "episode_id", range(1, len(episodes_df) + 1))

episodes_df = episodes_df.rename(columns={
  "classe": "class_name",
  "inicio_s": "start_time_sec",
  "fim_s": "end_time_sec",
  "duracao_s": "duration_sec",
})

episodes_df = episodes_df[[
  "episode_id", "class_name", "quadro_inicio", "quadro_fim",
  "start_time_sec", "end_time_sec", "duration_sec", "pico"
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
  f"Score threshold: {SCORE_THRESHOLD} | Min segment duration: {MIN_DURATION_SEC}s (PROVISORIO)",
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
  "score_threshold": SCORE_THRESHOLD,
  "min_episode_duration_sec": MIN_DURATION_SEC,
  "valores_provisorios": True,
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
