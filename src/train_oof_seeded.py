# ============================================================
# ARQUIVO: train_oof_seeded.py
#
# O QUE FAZ: treina 5 modelos out-of-fold (mesma arquitetura,
# hiperparâmetros e épocas de train_classifier.py/train_oof.py),
# parametrizado por --dataset-dir, para poder rodar tanto no dataset
# original (data/datasets/, rótulo por vídeo inteiro) quanto no da
# rodada A (data/datasets_annotated/, rótulo por janela via XML +
# classe de fundo) com o código de treino idêntico -- isolando o
# efeito da mudança de rótulo.
#
# REAPROVEITA O PARTICIONAMENTO DE VÍDEO -> FOLD JÁ CALCULADO em
# models/oof/fold_assignments.json (StratifiedGroupKFold, seed=42,
# sobre o dataset ORIGINAL) em vez de recalcular. Isso é essencial:
# recalcular o split diretamente no dataset da rodada A daria um
# particionamento DIFERENTE, porque lá um mesmo vídeo (um "group")
# tem janelas de mais de uma classe (comportamento + fundo), e o
# StratifiedGroupKFold usa a distribuição de classes por grupo para
# decidir o split -- o particionamento deixaria de ser "os mesmos 5
# folds". Como o particionamento already existente é por VÍDEO (não
# por janela/sequência), ele pode ser aplicado a qualquer dataset que
# tenha os mesmos vídeos em groups.npy.
#
# DIFERENÇAS EM RELAÇÃO A train_oof.py:
#   - fixa TODAS as sementes (Python random, NumPy, PyTorch, e a
#     ordem dos lotes via torch.Generator por fold) e grava os
#     valores usados em <output-dir>/seeds.json
#   - opcionalmente subamostra a classe "Background" no conjunto de
#     TREINO de cada fold até igualar o total de janelas de
#     comportamento daquele fold, sorteando de forma equilibrada
#     entre os vídeos (--subsample-background) -- só o treino é
#     subamostrado; o teste de cada fold usa todas as janelas, sem
#     subamostragem, para a avaliação continuar refletindo a
#     distribuição real
#   - grava em --output-dir (nunca em models/oof/, que é o baseline
#     já treinado por train_oof.py)
#
# Uso:
#   python src/train_oof_seeded.py --dataset-dir data/datasets --output-dir models/oof_seeded_base
#   python src/train_oof_seeded.py --dataset-dir data/datasets_annotated --output-dir models/oof_seeded_annotated --subsample-background
# ============================================================

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader

SEED = 42
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
N_SPLITS = 5

FOLD_ASSIGNMENTS_PATH = Path("models/oof/fold_assignments.json")


def set_all_seeds(seed):

  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)


# ==========================
# DATASET / MODELO / TREINO / AVALIAÇÃO
# -- cópia literal de train_classifier.py/train_oof.py (mesmo motivo:
# esses arquivos rodam o treino inteiro ao serem importados)
# ==========================

class AutismDataset(Dataset):

  def __init__(self, X, y):
    self.X = torch.tensor(X, dtype=torch.float32)
    self.y = torch.tensor(y, dtype=torch.long)

  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx):
    return self.X[idx], self.y[idx]


class LSTMClassifier(nn.Module):

  def __init__(self, input_size, hidden_size, num_layers, num_classes):
    super().__init__()
    self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_size, num_classes)

  def forward(self, x):
    output, (hidden, _) = self.lstm(x)
    return self.fc(hidden[-1])


def build_model(input_size, num_classes, device):
  return LSTMClassifier(input_size=input_size, hidden_size=128, num_layers=2, num_classes=num_classes).to(device)


def train_model(model, loader, device, log_prefix=""):

  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

  model.train()

  for epoch in range(EPOCHS):

    running_loss = 0

    for X_batch, y_batch in loader:

      X_batch = X_batch.to(device)
      y_batch = y_batch.to(device)

      optimizer.zero_grad()
      outputs = model(X_batch)
      loss = criterion(outputs, y_batch)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    print(f"{log_prefix}Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}")


def evaluate_model(model, loader, device):

  model.eval()

  all_predictions = []
  all_targets = []

  with torch.no_grad():

    for X_batch, y_batch in loader:

      X_batch = X_batch.to(device)
      outputs = model(X_batch)
      predictions = torch.argmax(outputs, dim=1)

      all_predictions.extend(predictions.cpu().numpy())
      all_targets.extend(y_batch.numpy())

  return np.array(all_targets), np.array(all_predictions)


# ==========================
# SUBAMOSTRAGEM DE FUNDO BALANCEADA ENTRE VÍDEOS
# ==========================

def balanced_background_sample(background_indices_by_video, target_total, rng):
  """Dá uma cota igual de janelas de fundo a cada vídeo do fold de
  treino, redistribuindo o que sobra dos vídeos que têm menos que a
  cota entre os demais, até atingir target_total (ou esgotar o que
  existe). Sorteia dentro de cada vídeo com `rng` (seed fixa)."""

  videos = list(background_indices_by_video.keys())
  available = {v: len(background_indices_by_video[v]) for v in videos}
  allocated = {v: 0 for v in videos}

  remaining_target = min(target_total, sum(available.values()))
  remaining_videos = {v for v in videos if available[v] > 0}

  while remaining_target > 0 and remaining_videos:

    share = max(1, remaining_target // len(remaining_videos))

    for v in list(remaining_videos):

      can_take = min(share, available[v] - allocated[v], remaining_target)

      if can_take <= 0:
        remaining_videos.discard(v)
        continue

      allocated[v] += can_take
      remaining_target -= can_take

      if allocated[v] >= available[v]:
        remaining_videos.discard(v)

      if remaining_target <= 0:
        break

  selected = []

  for v in videos:
    if allocated[v] > 0:
      chosen = rng.choice(background_indices_by_video[v], size=allocated[v], replace=False)
      selected.extend(chosen.tolist())

  return selected, allocated


# ==========================
# MAIN
# ==========================

def main(args):

  set_all_seeds(SEED)

  dataset_dir = Path(args.dataset_dir)
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")
  print(f"Dataset: {dataset_dir} | Saída: {output_dir} | Subamostrar fundo: {args.subsample_background}")

  X = np.load(dataset_dir / "X.npy")
  y = np.load(dataset_dir / "y.npy")
  groups = np.load(dataset_dir / "groups.npy", allow_pickle=True)

  with open(dataset_dir / "labels.json", "r", encoding="utf-8") as f:
    labels = json.load(f)

  num_classes = len(labels)
  id_to_label = {v: k for k, v in labels.items()}
  class_names = [id_to_label[i] for i in range(num_classes)]

  background_id = labels.get("Background")

  print("X shape:", X.shape, "| classes:", class_names)

  with open(FOLD_ASSIGNMENTS_PATH, "r", encoding="utf-8") as f:
    fold_assignments = json.load(f)["video_to_fold"]

  fold_accuracies = []
  fold_reports = []
  fold_report_texts = []
  fold_composition = {}

  for fold in range(1, N_SPLITS + 1):

    test_videos = [v for v, f in fold_assignments.items() if f == fold]
    train_videos = [v for v, f in fold_assignments.items() if f != fold]

    train_mask = np.isin(groups, train_videos)
    test_mask = np.isin(groups, test_videos)

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]

    subsample_info = None

    if args.subsample_background and background_id is not None:

      behavior_idx = [int(i) for i in train_idx if y[i] != background_id]
      background_idx = [int(i) for i in train_idx if y[i] == background_id]

      background_by_video = defaultdict(list)
      for i in background_idx:
        background_by_video[str(groups[i])].append(i)
      background_by_video = {v: np.array(idxs) for v, idxs in background_by_video.items()}

      rng = np.random.default_rng(SEED + fold)

      selected_background, allocation = balanced_background_sample(
        background_by_video, target_total=len(behavior_idx), rng=rng
      )

      print(
        f"[Fold {fold}] Subamostragem de fundo: {len(background_idx)} -> "
        f"{len(selected_background)} janelas (alvo = {len(behavior_idx)} de comportamento)"
      )

      train_idx = np.array(sorted(behavior_idx + selected_background))

      subsample_info = {
        "fundo_antes": len(background_idx),
        "fundo_depois": len(selected_background),
        "alvo_comportamento": len(behavior_idx),
        "alocacao_por_video": {v: int(n) for v, n in allocation.items()},
      }

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    fold_composition[fold] = {
      "n_train_videos": len(train_videos),
      "n_test_videos": len(test_videos),
      "n_train_janelas": int(len(train_idx)),
      "n_test_janelas": int(len(test_idx)),
      "contagem_treino_por_classe": {c: int((y_train == cid).sum()) for c, cid in labels.items()},
      "contagem_teste_por_classe": {c: int((y_test == cid).sum()) for c, cid in labels.items()},
      "subamostragem_de_fundo": subsample_info,
    }

    print(f"\n===== Fold {fold}/{N_SPLITS} =====")
    print(f"Train janelas: {len(train_idx)} | Test janelas: {len(test_idx)}")

    train_generator = torch.Generator().manual_seed(SEED + fold)

    train_loader = DataLoader(
      AutismDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True, generator=train_generator
    )
    test_loader = DataLoader(AutismDataset(X_test, y_test), batch_size=BATCH_SIZE)

    model = build_model(X.shape[2], num_classes, device)

    train_model(model, train_loader, device, log_prefix=f"[Fold {fold}] ")

    targets, predictions = evaluate_model(model, test_loader, device)

    accuracy = 100 * (predictions == targets).mean()
    fold_accuracies.append(accuracy)

    print(f"[Fold {fold}] Test Accuracy: {accuracy:.2f}%")

    fold_reports.append(
      classification_report(
        targets, predictions, labels=list(range(num_classes)),
        target_names=class_names, output_dict=True, zero_division=0
      )
    )

    fold_report_texts.append(
      classification_report(
        targets, predictions, labels=list(range(num_classes)),
        target_names=class_names, digits=4, zero_division=0
      )
    )

    torch.save(model.state_dict(), output_dir / f"fold_{fold}.pt")
    print(f"Saved {output_dir / f'fold_{fold}.pt'}")

  # ---------- salva metadados de reprodutibilidade ----------

  with open(output_dir / "fold_assignments.json", "w", encoding="utf-8") as f:
    json.dump({
      "n_splits": N_SPLITS,
      "random_state": 42,
      "video_to_fold": fold_assignments,
      "nota": "reaproveitado de models/oof/fold_assignments.json -- mesmo particionamento por vídeo do baseline, não recalculado",
    }, f, indent=2, ensure_ascii=False)

  with open(output_dir / "fold_composition.json", "w", encoding="utf-8") as f:
    json.dump(fold_composition, f, indent=2, ensure_ascii=False)

  with open(output_dir / "seeds.json", "w", encoding="utf-8") as f:
    json.dump({
      "seed_base": SEED,
      "python_random_seed": SEED,
      "numpy_seed": SEED,
      "torch_seed": SEED,
      "dataloader_generator_seed_por_fold": {fold: SEED + fold for fold in range(1, N_SPLITS + 1)},
    }, f, indent=2)

  # ---------- agregação (mesma lógica de train_classifier.py/train_oof.py) ----------

  def aggregate(row_key, metric_key):
    values = [report[row_key][metric_key] for report in fold_reports]
    return float(np.mean(values)), float(np.std(values))

  def metric_block(row_key):
    p_mean, p_std = aggregate(row_key, "precision")
    r_mean, r_std = aggregate(row_key, "recall")
    f_mean, f_std = aggregate(row_key, "f1-score")
    support_total = sum(report[row_key]["support"] for report in fold_reports)
    return {
      "precision_mean": p_mean, "precision_std": p_std,
      "recall_mean": r_mean, "recall_std": r_std,
      "f1_mean": f_mean, "f1_std": f_std,
      "support": support_total,
    }

  cv_metrics = {
    "n_splits": N_SPLITS,
    "fold_accuracies": fold_accuracies,
    "mean_accuracy": float(np.mean(fold_accuracies)),
    "std_accuracy": float(np.std(fold_accuracies)),
    "per_class": {class_name: metric_block(class_name) for class_name in class_names},
    "macro_avg": metric_block("macro avg"),
    "weighted_avg": metric_block("weighted avg"),
  }

  with open(output_dir / "cv_metrics_oof.json", "w", encoding="utf-8") as f:
    json.dump(cv_metrics, f, indent=2, ensure_ascii=False)

  with open(output_dir / "classification_report_per_fold.txt", "w", encoding="utf-8") as f:
    for fold, report_text in enumerate(fold_report_texts, start=1):
      f.write(f"===== Fold {fold}/{N_SPLITS} (accuracy: {fold_accuracies[fold-1]:.2f}%) =====\n")
      f.write(report_text)
      f.write("\n")

  print(f"\nConcluído. Saída em {output_dir}")
  print(f"Acurácia média: {cv_metrics['mean_accuracy']:.2f}% (+/- {cv_metrics['std_accuracy']:.2f}%)")


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset-dir", required=True, help="ex.: data/datasets ou data/datasets_annotated")
  parser.add_argument("--output-dir", required=True, help="ex.: models/oof_seeded_base")
  parser.add_argument("--subsample-background", action="store_true", help="subamostra a classe Background no treino de cada fold")

  main(parser.parse_args())
