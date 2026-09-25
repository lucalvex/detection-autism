# ============================================================
# ARQUIVO: generate_report.py
#
# O QUE FAZ: última etapa do pipeline. Lê os resultados já gerados
# pelas etapas anteriores (dataset em data/datasets/, métricas de
# validação cruzada e matriz de confusão em data/results/, e -- se
# existir -- a análise de episódios de src/utils/episode_analysis.py)
# e monta um único PDF consolidado, em data/results/report.pdf, com
# 1 seção por página:
#
#   1) Capa
#   2) Resumo do dataset (vídeos/sequências por classe)
#   3) Validação cruzada (acurácia por fold + métricas por classe)
#   4) Matriz de confusão
#   5) Frequência/duração de episódios por classe
#   6) Timeline gráfica dos episódios detectados
#
# Se algum arquivo de entrada não existir (ex.: você não rodou
# episode_analysis.py ainda), a seção correspondente aparece no PDF
# com um aviso, em vez de quebrar a geração do relatório inteiro.
# ============================================================

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

DATASET_DIR = Path("data/datasets")
RESULTS_DIR = Path("data/results")

REPORT_PATH = RESULTS_DIR / "report.pdf"

PAGE_SIZE = (8.27, 11.69)  # A4 em polegadas (retrato)


def new_page():
  fig = plt.figure(figsize=PAGE_SIZE)
  ax = fig.add_axes([0.06, 0.06, 0.88, 0.88])
  ax.axis("off")
  return fig, ax


def missing_section_page(pdf, title, missing_file, how_to_generate):
  fig, ax = new_page()
  ax.text(0.5, 0.6, title, ha="center", va="center", fontsize=18, fontweight="bold")
  ax.text(
    0.5, 0.45,
    f"Arquivo não encontrado: {missing_file}\n\nGere-o rodando:\n{how_to_generate}",
    ha="center", va="center", fontsize=11, color="firebrick"
  )
  pdf.savefig(fig)
  plt.close(fig)


def render_table(ax, col_labels, rows, col_widths=None, fontsize=9):
  table = ax.table(
    cellText=rows,
    colLabels=col_labels,
    cellLoc="center",
    loc="upper center",
    colWidths=col_widths
  )
  table.auto_set_font_size(False)
  table.set_fontsize(fontsize)
  table.scale(1, 1.6)
  return table


# ==========================
# 1. CAPA
# ==========================

def build_cover_page(pdf):

  fig, ax = new_page()

  ax.text(0.5, 0.75, "Relatório de Detecção de Comportamentos\nMotores Estereotipados em TEA",
          ha="center", va="center", fontsize=20, fontweight="bold")

  ax.text(0.5, 0.6, "YOLO11n-Pose + LSTM -- Dataset SSBD",
          ha="center", va="center", fontsize=13)

  ax.text(
    0.5, 0.45,
    f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}",
    ha="center", va="center", fontsize=10, color="gray"
  )

  ax.text(
    0.5, 0.3,
    "Seções:\n1. Resumo do dataset\n2. Validação cruzada (k-fold)\n"
    "3. Matriz de confusão\n4. Frequência/duração de episódios\n5. Timeline de episódios",
    ha="center", va="center", fontsize=10
  )

  pdf.savefig(fig)
  plt.close(fig)


# ==========================
# 2. RESUMO DO DATASET
# ==========================

def build_dataset_summary_page(pdf):

  required = ["X.npy", "y.npy", "groups.npy", "labels.json"]
  missing = [f for f in required if not (DATASET_DIR / f).exists()]

  if missing:
    missing_section_page(
      pdf, "1. Resumo do Dataset",
      f"data/datasets/{missing[0]}",
      "python src/create_sequences.py"
    )
    return

  X = np.load(DATASET_DIR / "X.npy")
  y = np.load(DATASET_DIR / "y.npy")
  groups = np.load(DATASET_DIR / "groups.npy", allow_pickle=True)

  with open(DATASET_DIR / "labels.json", "r", encoding="utf-8") as f:
    labels = json.load(f)

  id_to_label = {v: k for k, v in labels.items()}
  class_names = [id_to_label[i] for i in range(len(labels))]

  unique_videos = np.unique(groups)
  video_labels = np.array([y[groups == v][0] for v in unique_videos])

  rows = []
  for class_id, class_name in enumerate(class_names):
    n_videos = int((video_labels == class_id).sum())
    n_sequences = int((y == class_id).sum())
    rows.append([class_name, n_videos, n_sequences])

  fig, ax = new_page()

  ax.text(0.5, 0.95, "1. Resumo do Dataset", ha="center", va="top", fontsize=16, fontweight="bold")

  ax.text(
    0.5, 0.86,
    f"Total de vídeos: {len(unique_videos)}   |   Total de sequências: {len(y)}   |   "
    f"Shape de X: {X.shape}",
    ha="center", va="top", fontsize=10
  )

  table_ax = ax.inset_axes([0.05, 0.35, 0.9, 0.45])
  table_ax.axis("off")

  render_table(
    table_ax,
    col_labels=["Classe", "Vídeos", "Sequências"],
    rows=rows,
    col_widths=[0.4, 0.3, 0.3]
  )

  pdf.savefig(fig)
  plt.close(fig)


# ==========================
# 3. VALIDAÇÃO CRUZADA (K-FOLD)
# ==========================

def build_cv_metrics_page(pdf):

  cv_path = RESULTS_DIR / "cv_metrics.json"

  if not cv_path.exists():
    missing_section_page(
      pdf, "2. Validação Cruzada (k-fold)",
      "data/results/cv_metrics.json",
      "python src/train_classifier.py"
    )
    return

  with open(cv_path, "r", encoding="utf-8") as f:
    cv = json.load(f)

  fig, ax = new_page()

  ax.text(0.5, 0.95, "2. Validação Cruzada (k-fold)", ha="center", va="top", fontsize=16, fontweight="bold")

  fold_acc_str = ", ".join(f"{a:.2f}%" for a in cv["fold_accuracies"])

  ax.text(
    0.5, 0.87,
    f"StratifiedGroupKFold, k={cv['n_splits']}\n"
    f"Acurácia por fold: {fold_acc_str}\n"
    f"Acurácia média: {cv['mean_accuracy']:.2f}% (+/- {cv['std_accuracy']:.2f}%)",
    ha="center", va="top", fontsize=10
  )

  rows = []
  for class_name, m in cv["per_class"].items():
    rows.append([
      class_name,
      f"{m['precision_mean']:.3f} ± {m['precision_std']:.3f}",
      f"{m['recall_mean']:.3f} ± {m['recall_std']:.3f}",
      f"{m['f1_mean']:.3f} ± {m['f1_std']:.3f}",
      f"{m['support']:.0f}",
    ])

  rows.append([""] * 5)

  for avg_key, label in [("macro_avg", "macro avg"), ("weighted_avg", "weighted avg")]:
    m = cv[avg_key]
    rows.append([
      label,
      f"{m['precision_mean']:.3f} ± {m['precision_std']:.3f}",
      f"{m['recall_mean']:.3f} ± {m['recall_std']:.3f}",
      f"{m['f1_mean']:.3f} ± {m['f1_std']:.3f}",
      f"{m['support']:.0f}",
    ])

  table_ax = ax.inset_axes([0.02, 0.15, 0.96, 0.6])
  table_ax.axis("off")

  render_table(
    table_ax,
    col_labels=["Classe", "Precisão", "Recall", "F1-score", "Suporte"],
    rows=rows,
    col_widths=[0.24, 0.24, 0.24, 0.24, 0.14],
    fontsize=8
  )

  pdf.savefig(fig)
  plt.close(fig)


# ==========================
# 4. MATRIZ DE CONFUSÃO
# ==========================

def build_confusion_matrix_page(pdf):

  cm_path = RESULTS_DIR / "confusion_matrix.png"

  if not cm_path.exists():
    missing_section_page(
      pdf, "3. Matriz de Confusão",
      "data/results/confusion_matrix.png",
      "python src/train_classifier.py"
    )
    return

  image = plt.imread(cm_path)

  fig, ax = new_page()

  ax.text(0.5, 0.97, "3. Matriz de Confusão", ha="center", va="top", fontsize=16, fontweight="bold")

  img_ax = ax.inset_axes([0.05, 0.05, 0.9, 0.85])
  img_ax.axis("off")
  img_ax.imshow(image)

  pdf.savefig(fig)
  plt.close(fig)


# ==========================
# 5. FREQUÊNCIA/DURAÇÃO DE EPISÓDIOS
# ==========================

def build_episode_summary_page(pdf):

  summary_path = RESULTS_DIR / "episode_summary.json"

  if not summary_path.exists():
    missing_section_page(
      pdf, "4. Frequência/Duração de Episódios",
      "data/results/episode_summary.json",
      "python src/utils/episode_analysis.py"
    )
    return

  with open(summary_path, "r", encoding="utf-8") as f:
    ep = json.load(f)

  fig, ax = new_page()

  ax.text(0.5, 0.95, "4. Frequência/Duração de Episódios", ha="center", va="top", fontsize=16, fontweight="bold")

  provisorio_txt = " (PROVISÓRIOS)" if ep.get("valores_provisorios") else ""

  ax.text(
    0.5, 0.88,
    f"Vídeo: {ep['video_path']}\n"
    f"Duração: {ep['video_duration_sec']:.1f}s   |   "
    f"Limiar de pontuação: {ep['score_threshold']}   |   "
    f"Duração mínima do trecho: {ep['min_episode_duration_sec']}s{provisorio_txt}",
    ha="center", va="top", fontsize=9
  )

  rows = []
  for class_name, m in ep["per_class"].items():
    rows.append([
      class_name,
      f"{m['episodes']:.0f}",
      f"{m['episodes_per_min']:.2f}",
      f"{m['total_duration_sec']:.1f}",
      f"{m['mean_duration_sec']:.1f}",
      f"{m['min_duration_sec']:.1f}",
      f"{m['max_duration_sec']:.1f}",
    ])

  table_ax = ax.inset_axes([0.0, 0.35, 1.0, 0.45])
  table_ax.axis("off")

  render_table(
    table_ax,
    col_labels=["Classe", "Episódios", "Ep./min", "Dur. total (s)", "Dur. média (s)", "Dur. mín (s)", "Dur. máx (s)"],
    rows=rows,
    col_widths=[0.18, 0.12, 0.12, 0.16, 0.16, 0.13, 0.13],
    fontsize=8
  )

  pdf.savefig(fig)
  plt.close(fig)


# ==========================
# 6. TIMELINE GRÁFICA DOS EPISÓDIOS
# ==========================

def build_episode_timeline_page(pdf):

  episodes_path = RESULTS_DIR / "episodes.csv"

  if not episodes_path.exists():
    missing_section_page(
      pdf, "5. Timeline de Episódios",
      "data/results/episodes.csv",
      "python src/utils/episode_analysis.py"
    )
    return

  import pandas as pd
  episodes_df = pd.read_csv(episodes_path)

  class_names = sorted(episodes_df["class_name"].unique())
  cmap = plt.get_cmap("tab10")
  color_by_class = {name: cmap(i % 10) for i, name in enumerate(class_names)}

  fig = plt.figure(figsize=PAGE_SIZE)
  ax = fig.add_axes([0.15, 0.08, 0.8, 0.8])

  y_positions = {name: i for i, name in enumerate(class_names)}

  for _, ep in episodes_df.iterrows():
    ax.broken_barh(
      [(ep["start_time_sec"], ep["duration_sec"])],
      (y_positions[ep["class_name"]] - 0.4, 0.8),
      facecolors=color_by_class[ep["class_name"]]
    )

  ax.set_yticks(list(y_positions.values()))
  ax.set_yticklabels(list(y_positions.keys()))
  ax.set_xlabel("Tempo (s)")
  ax.set_title("5. Timeline de Episódios Detectados", fontsize=14, fontweight="bold")
  ax.grid(axis="x", linestyle="--", alpha=0.4)

  pdf.savefig(fig)
  plt.close(fig)


# ==========================
# MONTAGEM DO PDF
# ==========================

def main():

  RESULTS_DIR.mkdir(parents=True, exist_ok=True)

  with PdfPages(REPORT_PATH) as pdf:

    build_cover_page(pdf)
    build_dataset_summary_page(pdf)
    build_cv_metrics_page(pdf)
    build_confusion_matrix_page(pdf)
    build_episode_summary_page(pdf)
    build_episode_timeline_page(pdf)

  print(f"Report saved to {REPORT_PATH}")


if __name__ == "__main__":
  main()
