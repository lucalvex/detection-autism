# ============================================================
# ARQUIVO: create_sequences_annotated.py
#
# O QUE FAZ (RODADA A): gera um dataset ALTERNATIVO para X/y/groups
# usando os intervalos anotados no XML do SSBD como rótulo de cada
# janela, em vez do rótulo por vídeo inteiro (nome do arquivo) que
# create_sequences.py usa. NÃO altera create_sequences.py nem seu
# dataset (data/datasets/) -- grava em data/datasets_annotated/.
#
# REGRA DE RÓTULO POR JANELA (30 quadros, passo 1 -- mesmo tamanho e
# fatiamento de create_sequences.py):
#   - recebe a classe de um intervalo anotado se >= 2/3 dos quadros
#     da janela (>= 20 de 30) caem dentro desse intervalo
#   - recebe "Background" se NENHUM quadro da janela cai em nenhum
#     intervalo anotado (0 de 30)
#   - é DESCARTADA em todos os outros casos (mistura ambígua: nem
#     20+ quadros de uma classe, nem os 30 fora de qualquer intervalo)
#
# Usa o NÚMERO REAL do quadro (coluna "frame" do CSV de pose, que tem
# lacunas onde extract_pose.py não detectou pessoa -- 48 dos 56
# vídeos têm isso) para decidir cobertura, não a posição da linha no
# CSV -- mesma correção já aplicada em evaluate_temporal.py. O
# FATIAMENTO das features em si (para a LSTM) continua por posição de
# linha, igual create_sequences.py -- é assim que o modelo é
# treinado, lacuna ou não.
#
# Saída em data/datasets_annotated/:
#   - X.npy, y.npy, groups.npy -- mesmo formato de create_sequences.py,
#     mas com 4 classes: ArmFlapping, HeadBanging, Spinning, Background
#   - labels.json -- mapeamento das 4 classes
#   - window_stats.json -- quantas janelas foram descartadas pela
#     regra de borda e a contagem final por classe, no total e por
#     vídeo (o cruzamento com o fold de cada vídeo, para o relatório
#     por fold, é feito por train_oof_seeded.py na hora do treino,
#     usando models/oof/fold_assignments.json)
# ============================================================

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ssbd_annotations import load_annotations

POSES_DIR = Path("data/poses")
VIDEOS_DIR = Path("data/videos")
XMLS_DIR = Path("../download-yt/xmls")
OUTPUT_DIR = Path("data/datasets_annotated")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEQUENCE_LENGTH = 30
MAJORITY_FRACTION = 2 / 3
MIN_FRAMES_FOR_CLASS = int(np.ceil(MAJORITY_FRACTION * SEQUENCE_LENGTH))  # 20 de 30

CATEGORY_TO_LABEL = {
  "armflapping": "ArmFlapping",
  "headbanging": "HeadBanging",
  "spinning": "Spinning",
}

LABELS = {"ArmFlapping": 0, "HeadBanging": 1, "Spinning": 2, "Background": 3}


def load_video_fps(video_id):

  cap = cv2.VideoCapture(str(VIDEOS_DIR / f"{video_id}.mp4"))
  fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
  cap.release()

  return fps


def frame_label_array(video_id, n_frames, fps):
  """1 rótulo por quadro real (uma das 3 classes anotadas, ou
  "Background" se nenhum intervalo cobre aquele quadro)."""

  xml_path = XMLS_DIR / f"{video_id}.xml"

  labels_per_frame = ["Background"] * n_frames

  if not xml_path.exists():
    return labels_per_frame

  ann = load_annotations(xml_path)

  for b in ann["behaviours"]:

    label = CATEGORY_TO_LABEL.get(b["category"])

    if label is None:
      continue

    start_f = max(0, round(b["start_s"] * fps))
    end_f = min(n_frames - 1, round(b["end_s"] * fps))

    for i in range(start_f, end_f + 1):
      labels_per_frame[i] = label

  return labels_per_frame


def process_video(video_id):

  csv_path = POSES_DIR / f"{video_id}.csv"

  if not csv_path.exists():
    return None

  df = pd.read_csv(csv_path)

  if "frame" not in df.columns:
    print(f"[AVISO] {video_id}: CSV sem coluna 'frame', pulando (regra de borda precisa do quadro real)")
    return None

  real_frame_index = df["frame"].values

  cols_to_drop = [c for c in ["frame", "person"] if c in df.columns]
  features = df.drop(columns=cols_to_drop).values.astype(np.float32)

  n_rows = len(features)

  if n_rows < SEQUENCE_LENGTH:
    return None

  fps = load_video_fps(video_id)
  n_frames = int(real_frame_index[-1]) + 1

  frame_labels = frame_label_array(video_id, n_frames, fps)

  X_video, y_video = [], []
  n_discarded = 0
  n_behavior = 0
  n_background = 0

  for start in range(n_rows - SEQUENCE_LENGTH + 1):

    frame_ids_in_window = real_frame_index[start:start + SEQUENCE_LENGTH]

    counts = {c: 0 for c in CATEGORY_TO_LABEL.values()}
    background_count = 0

    for f in frame_ids_in_window:

      label = frame_labels[f]

      if label == "Background":
        background_count += 1
      else:
        counts[label] += 1

    window_label = None

    for c, n in counts.items():
      if n >= MIN_FRAMES_FOR_CLASS:
        window_label = c
        break

    if window_label is None and background_count == SEQUENCE_LENGTH:
      window_label = "Background"

    if window_label is None:
      n_discarded += 1
      continue

    if window_label == "Background":
      n_background += 1
    else:
      n_behavior += 1

    X_video.append(features[start:start + SEQUENCE_LENGTH])
    y_video.append(LABELS[window_label])

  return {
    "X": X_video,
    "y": y_video,
    "n_discarded": n_discarded,
    "n_behavior": n_behavior,
    "n_background": n_background,
    "n_windows_total": n_rows - SEQUENCE_LENGTH + 1,
  }


if __name__ == "__main__":

  X, y, groups = [], [], []
  stats_per_video = {}

  total_discarded = 0
  total_behavior = 0
  total_background = 0

  for csv_path in sorted(POSES_DIR.glob("*.csv")):

    video_id = csv_path.stem
    result = process_video(video_id)

    if result is None:
      continue

    X.extend(result["X"])
    y.extend(result["y"])
    groups.extend([video_id] * len(result["y"]))

    stats_per_video[video_id] = {
      "n_windows_total": result["n_windows_total"],
      "n_behavior": result["n_behavior"],
      "n_background": result["n_background"],
      "n_discarded": result["n_discarded"],
    }

    total_discarded += result["n_discarded"]
    total_behavior += result["n_behavior"]
    total_background += result["n_background"]

    print(
      f"{video_id}: {result['n_behavior']} comportamento, "
      f"{result['n_background']} fundo, {result['n_discarded']} descartadas "
      f"(de {result['n_windows_total']} janelas)"
    )

  X = np.array(X, dtype=np.float32)
  y = np.array(y, dtype=np.int64)
  groups = np.array(groups)

  np.save(OUTPUT_DIR / "X.npy", X)
  np.save(OUTPUT_DIR / "y.npy", y)
  np.save(OUTPUT_DIR / "groups.npy", groups)

  with open(OUTPUT_DIR / "labels.json", "w", encoding="utf-8") as f:
    json.dump(LABELS, f, indent=2)

  class_counts = {name: int((y == cid).sum()) for name, cid in LABELS.items()}

  window_stats = {
    "regra": (
      "janela de 30 quadros recebe a classe do intervalo se >=20/30 "
      "quadros estao dentro dele; recebe Background se 0/30 estao em "
      "qualquer intervalo; e descartada nos demais casos"
    ),
    "total_janelas_avaliadas": total_discarded + total_behavior + total_background,
    "total_descartadas": total_discarded,
    "total_comportamento": total_behavior,
    "total_background": total_background,
    "contagem_final_por_classe": class_counts,
    "por_video": stats_per_video,
  }

  with open(OUTPUT_DIR / "window_stats.json", "w", encoding="utf-8") as f:
    json.dump(window_stats, f, indent=2, ensure_ascii=False)

  print(f"\nTotal: {total_behavior} comportamento, {total_background} fundo, {total_discarded} descartadas")
  print(f"X shape: {X.shape}")
  print("Contagem final por classe:", class_counts)
