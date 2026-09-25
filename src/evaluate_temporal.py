# ============================================================
# ARQUIVO: evaluate_temporal.py
#
# O QUE FAZ: avaliação temporal do modelo usando SÓ os vídeos em que
# cada predição é out-of-fold (nunca viu esse vídeo no treino),
# aproveitando os checkpoints de models/oof/fold_{k}.pt e o
# mapeamento models/oof/fold_assignments.json gerados por
# train_oof.py. Duas medidas:
#
#   1) Por quadro: precisão/recall/F1 por classe, comparando a classe
#      prevista (argmax das 6 saídas do modelo, colapsando
#      Rocking/HandMovement/Normal em "Background" -- ver nota
#      abaixo) contra a classe anotada no XML do SSBD naquele frame
#      (ou "Background" se nenhum intervalo anotado cobre o frame).
#
#   2) Por evento: constrói os trechos detectados com
#      src/timeline.py (mesma regra da interface) e faz pareamento
#      guloso por IoU temporal contra os intervalos anotados da MESMA
#      classe, para IoU >= 0.1/0.3/0.5 -- reporta acertos (TP), falsos
#      positivos (FP), perdidos (FN), precisão, recall, F1, e o erro
#      médio de início/fim (em segundos) dos trechos que casaram.
#
# NÃO roda o YOLO de novo: reaproveita os CSVs de pose já extraídos
# em data/poses/*.csv (mesmos usados no treino) para montar as
# janelas -- só a inferência da LSTM (rápida) precisa rodar de novo,
# uma vez por vídeo, com o checkpoint do fold em que aquele vídeo foi
# teste.
#
# NOTA SOBRE "Background" no nível de quadro: o modelo não tem uma
# classe de repouso treinada (Rocking/HandMovement/Normal têm 0
# vídeos reais -- ver decisão registrada nesta conversa). Para a
# métrica por quadro precisar de 4 rótulos (3 classes reais +
# ausência de comportamento), uso a saída de 6 vias do modelo e
# colapso as 3 classes sem dado de treino em "Background". Isso é
# uma escolha de avaliação, não algo calculado -- documentada aqui,
# não escondida.
#
# Este script grava data/results/metricas.json com:
#   - a REGRA PRINCIPAL (limiar=0.5, duração mínima = 1 janela, i.e.
#     sem filtro adicional além do piso natural do algoritmo)
#   - a GRADE COMPLETA (limiar x duração mínima, macro F1 por evento
#     IoU=0.3)
#   - o MELHOR PAR da grade, marcado como "selecionado a posteriori
#     nos dados de avaliação" (não é uma escolha a priori -- ver
#     ressalva de metodologia no bloco correspondente do JSON)
# Tudo isso com média ± desvio-padrão entre os 5 folds, além do
# número agregado sobre os 56 vídeos.
# ============================================================

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

from ssbd_annotations import load_annotations
from timeline import build_segments

# ==========================
# CONFIG
# ==========================

DATASET_DIR = Path("data/datasets")
POSES_DIR = Path("data/poses")
VIDEOS_DIR = Path("data/videos")
OOF_DIR = Path("models/oof")
RESULTS_DIR = Path("data/results")

# As anotações XML do SSBD vivem fora deste repositório -- ajuste se
# você mover os arquivos.
XMLS_DIR = Path("../download-yt/xmls")

SEQUENCE_LENGTH = 30  # precisa bater com create_sequences.py

CATEGORY_TO_LABEL = {
  "armflapping": "ArmFlapping",
  "headbanging": "HeadBanging",
  "spinning": "Spinning",
}

REAL_CLASSES = ["ArmFlapping", "HeadBanging", "Spinning"]  # únicas com dado real de treino

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(DATASET_DIR / "labels.json", "r", encoding="utf-8") as f:
  labels = json.load(f)

num_classes = len(labels)
id_to_label = {v: k for k, v in labels.items()}
class_names = [id_to_label[i] for i in range(num_classes)]

with open(OOF_DIR / "fold_assignments.json", "r", encoding="utf-8") as f:
  fold_assignments = json.load(f)["video_to_fold"]

# ==========================
# MODELO -- cópia literal de train_classifier.py/train_oof.py (ver
# aviso em train_oof.py sobre por que isso não é importado)
# ==========================

class LSTMClassifier(nn.Module):

  def __init__(self, input_size, hidden_size, num_layers, num_classes):
    super().__init__()
    self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_size, num_classes)

  def forward(self, x):
    output, (hidden, _) = self.lstm(x)
    return self.fc(hidden[-1])


_model_cache = {}


def get_fold_model(fold):

  if fold not in _model_cache:

    model = LSTMClassifier(input_size=68, hidden_size=128, num_layers=2, num_classes=num_classes).to(device)

    model.load_state_dict(
      torch.load(OOF_DIR / f"fold_{fold}.pt", map_location=device)
    )

    model.eval()
    _model_cache[fold] = model

  return _model_cache[fold]


# ==========================
# POR VÍDEO: janelas + pontuações (out-of-fold) + gabarito
# ==========================

def load_video_fps(video_id):

  cap = cv2.VideoCapture(str(VIDEOS_DIR / f"{video_id}.mp4"))
  fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
  cap.release()

  return fps


def compute_windows(video_id):
  """Reaproveita o CSV de pose já extraído (mesmas features do
  treino) e roda só a LSTM do fold em que este vídeo foi teste.

  IMPORTANTE: extract_pose.py PULA frames sem pessoa detectada (não
  grava linha nenhuma) -- 48 dos 56 vídeos têm gaps assim, totalizando
  11493 frames faltantes no dataset inteiro (ex.: v_ArmFlapping_08
  sozinho tem 245 frames faltando no meio do vídeo). Isso significa
  que a POSIÇÃO da linha no CSV (0, 1, 2, ...) NÃO é o índice real do
  frame no vídeo sempre que houve um gap antes dela. Para as janelas
  ficarem alinhadas no tempo com as anotações XML (que são medidas
  contra o vídeo real), quadro_inicio/quadro_fim usam a coluna "frame"
  do CSV (índice real, com gaps), não a posição da linha. O
  FATIAMENTO das features em si (para a LSTM) continua por posição de
  linha, igual create_sequences.py -- é assim que o modelo foi
  treinado, gap ou não."""

  df = pd.read_csv(POSES_DIR / f"{video_id}.csv")

  real_frame_index = df["frame"].values if "frame" in df.columns else np.arange(len(df))

  cols_to_drop = [c for c in ["frame", "person"] if c in df.columns]
  features = df.drop(columns=cols_to_drop).values.astype(np.float32)

  n_rows = len(features)

  if n_rows < SEQUENCE_LENGTH:
    return []

  fold = fold_assignments[video_id]
  model = get_fold_model(fold)

  starts = list(range(n_rows - SEQUENCE_LENGTH + 1))
  sequences = np.stack([features[s:s + SEQUENCE_LENGTH] for s in starts])

  with torch.no_grad():

    batch = torch.tensor(sequences, dtype=torch.float32).to(device)
    outputs = model(batch)
    probabilities = torch.softmax(outputs, dim=1).cpu().numpy()

  windows = [
    {
      "quadro_inicio": int(real_frame_index[start]),
      "quadro_fim": int(real_frame_index[start + SEQUENCE_LENGTH - 1]) + 1,
      "pontuacoes": probabilities[i].tolist(),
    }
    for i, start in enumerate(starts)
  ]

  return windows


def ground_truth_frames(video_id, n_frames, fps):
  """1 rótulo por frame (uma das 3 classes reais, ou "Background"),
  a partir dos intervalos anotados no XML."""

  xml_path = XMLS_DIR / f"{video_id}.xml"

  gt = ["Background"] * n_frames

  if not xml_path.exists():
    return gt, []

  ann = load_annotations(xml_path)
  intervals = []  # (classe, start_s, end_s), só das 3 classes reais

  for b in ann["behaviours"]:

    label = CATEGORY_TO_LABEL.get(b["category"])

    if label is None:
      continue

    intervals.append((label, b["start_s"], b["end_s"]))

    start_f = max(0, int(round(b["start_s"] * fps)))
    end_f = min(n_frames - 1, int(round(b["end_s"] * fps)))

    for i in range(start_f, end_f + 1):
      gt[i] = label

  return gt, intervals


def predicted_frames(windows):
  """1 rótulo por quadro processado (argmax das 6 saídas, colapsando
  as 3 classes sem dado real em "Background" -- ver nota no
  cabeçalho do arquivo)."""

  pred = {}

  for w in windows:

    frame_idx = w["quadro_fim"] - 1  # último quadro coberto pela janela
    class_id = int(np.argmax(w["pontuacoes"]))
    class_name = id_to_label[class_id]

    pred[frame_idx] = class_name if class_name in REAL_CLASSES else "Background"

  return pred


# ==========================
# CARREGA TUDO (1 vez só -- caro é a LSTM, não o XML)
# ==========================

def load_all_videos():

  video_ids = sorted(fold_assignments.keys())
  data = {}

  for video_id in video_ids:

    windows = compute_windows(video_id)

    if not windows:
      print(f"[AVISO] {video_id}: sem janelas suficientes, pulando")
      continue

    fps = load_video_fps(video_id)
    n_frames = windows[-1]["quadro_fim"]

    gt_frames, gt_intervals = ground_truth_frames(video_id, n_frames, fps)

    data[video_id] = {
      "fold": fold_assignments[video_id],
      "fps": fps,
      "n_frames": n_frames,
      "windows": windows,
      "gt_frames": gt_frames,
      "gt_intervals": gt_intervals,
    }

  return data


# ==========================
# MÉTRICA POR QUADRO
# ==========================

def frame_level_report(video_data, video_ids, restrict_to_annotated=False):
  """restrict_to_annotated=True: descarta da avaliação todo quadro
  cujo gabarito seja "Background" (fora de qualquer intervalo
  anotado) -- ou seja, só olha quadros que o SSBD realmente anotou
  como um dos 3 comportamentos. O modelo ainda PODE prever
  "Background" nesses quadros (se o argmax cair numa classe sem
  dado de treino); só o lado do gabarito é restrito."""

  y_true_all = []
  y_pred_all = []

  for video_id in video_ids:

    d = video_data[video_id]
    pred = predicted_frames(d["windows"])

    for frame_idx, pred_label in pred.items():

      true_label = d["gt_frames"][frame_idx]

      if restrict_to_annotated and true_label == "Background":
        continue

      y_true_all.append(true_label)
      y_pred_all.append(pred_label)

  frame_labels = REAL_CLASSES + ["Background"]

  report = classification_report(
    y_true_all, y_pred_all,
    labels=frame_labels,
    output_dict=True,
    zero_division=0
  )

  cm = confusion_matrix(y_true_all, y_pred_all, labels=frame_labels)

  macro_f1_com_background = float(np.mean([report[c]["f1-score"] for c in frame_labels]))
  macro_f1_sem_background = float(np.mean([report[c]["f1-score"] for c in REAL_CLASSES]))

  return {
    "report": report,
    "confusion_matrix": cm.tolist(),
    "labels": frame_labels,
    "macro_f1_com_background": macro_f1_com_background,
    "macro_f1_sem_background": macro_f1_sem_background,
    "n_frames": len(y_true_all),
  }


# ==========================
# MÉTRICA POR EVENTO (IoU temporal, pareamento guloso)
# ==========================

def temporal_iou(a_start, a_end, b_start, b_end):

  inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
  union = max(a_end, b_end) - min(a_start, b_start)

  return inter / union if union > 0 else 0.0


def match_events(detected, ground_truth, iou_threshold):
  """detected/ground_truth: listas de (inicio_s, fim_s) da MESMA
  classe e do MESMO vídeo. Pareamento guloso: casa sempre o par de
  maior IoU disponível primeiro."""

  candidates = []

  for i, d in enumerate(detected):
    for j, g in enumerate(ground_truth):
      score = temporal_iou(d[0], d[1], g[0], g[1])
      if score > 0:
        candidates.append((score, i, j))

  candidates.sort(reverse=True)

  used_d, used_g = set(), set()
  matches = []

  for score, i, j in candidates:

    if i in used_d or j in used_g:
      continue

    if score >= iou_threshold:
      used_d.add(i)
      used_g.add(j)
      matches.append((i, j, score))

  false_positives = len(detected) - len(used_d)
  misses = len(ground_truth) - len(used_g)

  return matches, false_positives, misses


def event_level_report(video_data, video_ids, threshold, min_duration_sec, iou_thresholds):
  """Constrói os trechos detectados (limiar + duração mínima) e faz o
  pareamento por IoU contra o gabarito, agregado sobre video_ids."""

  results = {}

  for iou_thresh in iou_thresholds:

    per_class = {c: {"tp": 0, "fp": 0, "fn": 0} for c in REAL_CLASSES}
    start_errors = []
    end_errors = []

    for video_id in video_ids:

      d = video_data[video_id]

      segments = build_segments(
        d["windows"], class_names, threshold=threshold,
        min_duration_sec=min_duration_sec, fps=d["fps"]
      )

      for class_name in REAL_CLASSES:

        detected = [
          (s["inicio_s"], s["fim_s"]) for s in segments if s["classe"] == class_name
        ]
        ground_truth = [
          (start_s, end_s) for (label, start_s, end_s) in d["gt_intervals"] if label == class_name
        ]

        matches, fp, fn = match_events(detected, ground_truth, iou_thresh)

        per_class[class_name]["tp"] += len(matches)
        per_class[class_name]["fp"] += fp
        per_class[class_name]["fn"] += fn

        for i, j, score in matches:
          start_errors.append(abs(detected[i][0] - ground_truth[j][0]))
          end_errors.append(abs(detected[i][1] - ground_truth[j][1]))

    for class_name in REAL_CLASSES:

      tp = per_class[class_name]["tp"]
      fp = per_class[class_name]["fp"]
      fn = per_class[class_name]["fn"]

      precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
      recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
      f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

      per_class[class_name].update({"precisao": precision, "revocacao": recall, "f1": f1})

    macro_f1 = float(np.mean([per_class[c]["f1"] for c in REAL_CLASSES]))

    results[iou_thresh] = {
      "por_classe": per_class,
      "macro_f1": macro_f1,
      "erro_medio_inicio_s": float(np.mean(start_errors)) if start_errors else None,
      "erro_medio_fim_s": float(np.mean(end_errors)) if end_errors else None,
      "n_pares_casados": len(start_errors),
    }

  return results


# ==========================
# AGREGAÇÃO POR FOLD (mesmo padrão de train_classifier.py: pooled + média±desvio entre folds)
# ==========================

def group_by_fold(video_ids):

  by_fold = {}

  for v in video_ids:
    by_fold.setdefault(fold_assignments[v], []).append(v)

  return by_fold


def mean_std(values):
  return float(np.mean(values)), float(np.std(values))


def frame_level_with_folds(video_data, video_ids, restrict_to_annotated=False):

  pooled = frame_level_report(video_data, video_ids, restrict_to_annotated)

  by_fold = group_by_fold(video_ids)
  per_fold = {
    fold: frame_level_report(video_data, fold_video_ids, restrict_to_annotated)
    for fold, fold_video_ids in sorted(by_fold.items())
  }

  all_labels = REAL_CLASSES + ["Background"]

  per_class_agg = {}
  for c in all_labels:
    p_mean, p_std = mean_std([r["report"][c]["precision"] for r in per_fold.values()])
    r_mean, r_std = mean_std([r["report"][c]["recall"] for r in per_fold.values()])
    f_mean, f_std = mean_std([r["report"][c]["f1-score"] for r in per_fold.values()])
    per_class_agg[c] = {
      "precisao_media": p_mean, "precisao_desvio": p_std,
      "revocacao_media": r_mean, "revocacao_desvio": r_std,
      "f1_media": f_mean, "f1_desvio": f_std,
    }

  macro_bg_mean, macro_bg_std = mean_std([r["macro_f1_com_background"] for r in per_fold.values()])
  macro_no_bg_mean, macro_no_bg_std = mean_std([r["macro_f1_sem_background"] for r in per_fold.values()])

  return {
    "pooled_56_videos": pooled,
    "por_fold": {str(f): r for f, r in per_fold.items()},
    "macro_f1_com_background_media": macro_bg_mean,
    "macro_f1_com_background_desvio": macro_bg_std,
    "macro_f1_sem_background_media": macro_no_bg_mean,
    "macro_f1_sem_background_desvio": macro_no_bg_std,
    "por_classe_media_desvio_entre_folds": per_class_agg,
  }


def event_level_with_folds(video_data, video_ids, threshold, min_duration_sec, iou_thresholds):

  pooled = event_level_report(video_data, video_ids, threshold, min_duration_sec, iou_thresholds)

  by_fold = group_by_fold(video_ids)
  per_fold = {
    fold: event_level_report(video_data, fold_video_ids, threshold, min_duration_sec, iou_thresholds)
    for fold, fold_video_ids in sorted(by_fold.items())
  }

  result = {}

  for iou in iou_thresholds:

    macro_mean, macro_std = mean_std([per_fold[f][iou]["macro_f1"] for f in per_fold])

    per_class_agg = {}
    for c in REAL_CLASSES:
      p_mean, p_std = mean_std([per_fold[f][iou]["por_classe"][c]["precisao"] for f in per_fold])
      r_mean, r_std = mean_std([per_fold[f][iou]["por_classe"][c]["revocacao"] for f in per_fold])
      f_mean, f_std = mean_std([per_fold[f][iou]["por_classe"][c]["f1"] for f in per_fold])
      per_class_agg[c] = {
        "precisao_media": p_mean, "precisao_desvio": p_std,
        "revocacao_media": r_mean, "revocacao_desvio": r_std,
        "f1_media": f_mean, "f1_desvio": f_std,
      }

    result[iou] = {
      "pooled_56_videos": pooled[iou],
      "por_fold": {str(f): per_fold[f][iou] for f in per_fold},
      "macro_f1_media_entre_folds": macro_mean,
      "macro_f1_desvio_entre_folds": macro_std,
      "por_classe_media_desvio_entre_folds": per_class_agg,
    }

  return result


# ==========================
# VARREDURA limiar x duração mínima (objetivo: macro F1 por evento, IoU=0.3)
# ==========================

def threshold_duration_grid_search(video_data, video_ids):

  thresholds = np.round(np.arange(0.30, 0.90 + 1e-9, 0.05), 2)
  min_durations = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0]

  grid = np.zeros((len(min_durations), len(thresholds)))
  hits_grid = np.zeros((len(min_durations), len(thresholds)), dtype=int)

  for di, min_dur in enumerate(min_durations):
    for ti, thresh in enumerate(thresholds):

      report = event_level_report(video_data, video_ids, thresh, min_dur, iou_thresholds=[0.3])
      grid[di, ti] = report[0.3]["macro_f1"]
      hits_grid[di, ti] = sum(report[0.3]["por_classe"][c]["tp"] for c in REAL_CLASSES)

  best_idx = np.unravel_index(np.argmax(grid), grid.shape)
  best_min_dur = min_durations[best_idx[0]]
  best_thresh = thresholds[best_idx[1]]
  best_f1 = grid[best_idx]
  best_hits = hits_grid[best_idx]

  return thresholds, min_durations, grid, hits_grid, best_thresh, best_min_dur, best_f1, best_hits


if __name__ == "__main__":

  print("Carregando janelas/pontuações out-of-fold + gabarito XML para todos os vídeos...")

  video_data = load_all_videos()
  video_ids = sorted(video_data.keys())

  print(f"Vídeos carregados: {len(video_ids)}/{len(fold_assignments)}")

  # ---------- 1. MÉTRICA POR QUADRO: todos os quadros ----------

  print("\n" + "=" * 70)
  print("MÉTRICA POR QUADRO -- TODOS OS QUADROS (out-of-fold, 56 vídeos)")
  print("=" * 70)

  frame_all = frame_level_with_folds(video_data, video_ids, restrict_to_annotated=False)
  pooled = frame_all["pooled_56_videos"]

  print(f"\n{'Classe':<15}{'Precisão':<12}{'Recall':<12}{'F1':<12}{'Suporte':<10}")
  for label in pooled["labels"]:
    m = pooled["report"][label]
    print(f"{label:<15}{m['precision']:<12.4f}{m['recall']:<12.4f}{m['f1-score']:<12.4f}{m['support']:<10.0f}")

  print(f"\nMacro F1 COM Background: {frame_all['macro_f1_com_background_media']:.4f} (+/- {frame_all['macro_f1_com_background_desvio']:.4f} entre folds)")
  print(f"Macro F1 SEM Background: {frame_all['macro_f1_sem_background_media']:.4f} (+/- {frame_all['macro_f1_sem_background_desvio']:.4f} entre folds)")
  print(f"\nMatriz de confusão (linhas=real, colunas=previsto), ordem {pooled['labels']}:")
  print(np.array(pooled["confusion_matrix"]))

  # ---------- 2. MÉTRICA POR QUADRO: só quadros dentro de intervalo anotado ----------

  print("\n" + "=" * 70)
  print("MÉTRICA POR QUADRO -- SÓ QUADROS ANOTADOS (exclui Background do gabarito)")
  print("=" * 70)

  frame_restricted = frame_level_with_folds(video_data, video_ids, restrict_to_annotated=True)
  pooled_r = frame_restricted["pooled_56_videos"]

  print(f"\n{'Classe':<15}{'Precisão':<12}{'Recall':<12}{'F1':<12}{'Suporte':<10}")
  for label in pooled_r["labels"]:
    m = pooled_r["report"][label]
    print(f"{label:<15}{m['precision']:<12.4f}{m['recall']:<12.4f}{m['f1-score']:<12.4f}{m['support']:<10.0f}")

  print(f"\nMacro F1 COM Background: {frame_restricted['macro_f1_com_background_media']:.4f} (+/- {frame_restricted['macro_f1_com_background_desvio']:.4f} entre folds)")
  print(f"Macro F1 SEM Background: {frame_restricted['macro_f1_sem_background_media']:.4f} (+/- {frame_restricted['macro_f1_sem_background_desvio']:.4f} entre folds)")
  print(f"\nMatriz de confusão (linhas=real, colunas=previsto), ordem {pooled_r['labels']}:")
  print(np.array(pooled_r["confusion_matrix"]))

  with open(RESULTS_DIR / "frame_level_report.json", "w", encoding="utf-8") as f:
    json.dump({
      "todos_os_quadros": frame_all,
      "somente_quadros_anotados": frame_restricted,
    }, f, indent=2, ensure_ascii=False)

  # ---------- 3. VARREDURA limiar x duração mínima (estendida) ----------

  print("\n" + "=" * 70)
  print("VARREDURA limiar x duração mínima (macro F1 por evento, IoU=0.3)")
  print("=" * 70)

  thresholds, min_durations, grid, hits_grid, best_thresh, best_min_dur, best_f1, best_hits = threshold_duration_grid_search(
    video_data, video_ids
  )

  print(
    f"\nMelhor par encontrado: limiar={best_thresh}, duração_mínima={best_min_dur}s -> "
    f"macro F1@IoU=0.3 = {best_f1:.4f} ({best_hits} pares casados no total, entre as 3 classes)"
  )

  table_lines = ["Macro F1 por evento (IoU=0.3) -- linhas=duração mínima (s), colunas=limiar"]
  header = "dur\\lim  " + "  ".join(f"{t:.2f}" for t in thresholds)
  table_lines.append(header)
  for di, min_dur in enumerate(min_durations):
    row = f"{min_dur:>7.1f}  " + "  ".join(f"{grid[di, ti]:.3f}" for ti in range(len(thresholds)))
    table_lines.append(row)
  table_lines.append("")
  table_lines.append("Pares casados (TP, somado nas 3 classes) -- mesma grade:")
  for di, min_dur in enumerate(min_durations):
    row = f"{min_dur:>7.1f}  " + "  ".join(f"{hits_grid[di, ti]:>4d}" for ti in range(len(thresholds)))
    table_lines.append(row)
  table_text = "\n".join(table_lines)
  print("\n" + table_text)

  with open(RESULTS_DIR / "threshold_duration_grid.txt", "w", encoding="utf-8") as f:
    f.write(table_text + "\n")
    f.write(f"\nMelhor par: limiar={best_thresh}, duracao_minima={best_min_dur}s, macro_f1={best_f1:.4f}, pares_casados={best_hits}\n")

  plt.figure(figsize=(11, 6))
  plt.imshow(grid, aspect="auto", cmap="viridis", origin="lower")
  plt.colorbar(label="Macro F1 por evento (IoU=0.3)")
  plt.xticks(range(len(thresholds)), [f"{t:.2f}" for t in thresholds], rotation=45)
  plt.yticks(range(len(min_durations)), [f"{d:.1f}" for d in min_durations])
  plt.xlabel("Limiar de pontuação")
  plt.ylabel("Duração mínima (s)")
  plt.title("Varredura limiar x duração mínima -- Macro F1 por evento (IoU=0.3)")
  plt.scatter([list(thresholds).index(best_thresh)], [min_durations.index(best_min_dur)],
              color="red", marker="*", s=300, label=f"melhor ({best_thresh}, {best_min_dur}s, {best_hits} pares)")
  plt.legend()
  plt.tight_layout()
  plt.savefig(RESULTS_DIR / "threshold_duration_grid.png", dpi=150)
  plt.close()

  print(f"\nGrid salvo em {RESULTS_DIR / 'threshold_duration_grid.txt'} e {RESULTS_DIR / 'threshold_duration_grid.png'}")

  # ---------- 4. metricas.json: regra principal + melhor par + grade completa ----------

  print("\n" + "=" * 70)
  print("REGRA PRINCIPAL (limiar=0.5, duração mínima = 1 janela)")
  print("=" * 70)

  IOU_THRESHOLDS = [0.1, 0.3, 0.5]

  regra_principal = event_level_with_folds(
    video_data, video_ids, threshold=0.5, min_duration_sec=0.0, iou_thresholds=IOU_THRESHOLDS
  )

  for iou in IOU_THRESHOLDS:
    r = regra_principal[iou]["pooled_56_videos"]
    print(
      f"IoU>={iou}: macro F1={r['macro_f1']:.4f} | "
      f"erro médio início={r['erro_medio_inicio_s']} s | erro médio fim={r['erro_medio_fim_s']} s | "
      f"pares casados={r['n_pares_casados']}"
    )

  print("\n" + "=" * 70)
  print(f"MELHOR PAR DA GRADE (selecionado a posteriori): limiar={best_thresh}, duração_mínima={best_min_dur}s")
  print("=" * 70)

  melhor_par = event_level_with_folds(
    video_data, video_ids, threshold=float(best_thresh), min_duration_sec=float(best_min_dur), iou_thresholds=IOU_THRESHOLDS
  )

  for iou in IOU_THRESHOLDS:
    r = melhor_par[iou]["pooled_56_videos"]
    print(
      f"IoU>={iou}: macro F1={r['macro_f1']:.4f} | "
      f"erro médio início={r['erro_medio_inicio_s']} s | erro médio fim={r['erro_medio_fim_s']} s | "
      f"pares casados={r['n_pares_casados']}"
    )

  with open(RESULTS_DIR / "oof_verification.txt", "r", encoding="utf-8") as f:
    oof_verification_text = f.read()

  metricas = {
    "divisao": "por vídeo, out-of-fold (models/oof/fold_{1..5}.pt + models/oof/fold_assignments.json; groups.npy)",
    "n_folds": 5,
    "verificacao_out_of_fold": {
      "descricao": "acurácia/precisão/recall/F1 por fold, reproduzindo o split de train_classifier.py, comparadas com data/results/cv_metrics.json antes de usar os checkpoints para qualquer avaliação.",
      "texto_completo": oof_verification_text,
    },
    "por_quadro": {
      "todos_os_quadros": frame_all,
      "somente_quadros_anotados": frame_restricted,
    },
    "por_evento": {
      "regra_principal": {
        "descricao": "limiar=0.5, duração mínima = 1 janela (piso natural do algoritmo de segmentação -- nenhum filtro de duração adicional aplicado)",
        "limiar": 0.5,
        "duracao_minima_config_s": 0.0,
        "por_iou": {str(iou): regra_principal[iou] for iou in IOU_THRESHOLDS},
      },
      "melhor_par_da_grade": {
        "descricao": (
          "AVISO METODOLÓGICO: este par (limiar, duração mínima) foi "
          "selecionado a posteriori, escolhendo o ponto de maior macro F1 "
          "por evento (IoU=0.3) NOS MESMOS DADOS usados para relatar o "
          "resultado abaixo. Não é uma escolha a priori, e o F1 aqui tende "
          "a ser otimista em relação ao que se veria em vídeos novos, por "
          "causa desse ajuste posterior. Reportado para referência, não "
          "como o número principal do trabalho -- esse é o da regra_principal."
        ),
        "limiar": float(best_thresh),
        "duracao_minima_config_s": float(best_min_dur),
        "por_iou": {str(iou): melhor_par[iou] for iou in IOU_THRESHOLDS},
      },
      "grade_completa": {
        "objetivo": "macro F1 por evento, IoU=0.3",
        "limiares": [float(t) for t in thresholds],
        "duracoes_minimas_s": [float(d) for d in min_durations],
        "macro_f1": grid.tolist(),
        "pares_casados": hits_grid.tolist(),
      },
    },
  }

  with open(RESULTS_DIR / "metricas.json", "w", encoding="utf-8") as f:
    json.dump(metricas, f, indent=2, ensure_ascii=False)

  print(f"\nmetricas.json salvo em {RESULTS_DIR / 'metricas.json'}")
