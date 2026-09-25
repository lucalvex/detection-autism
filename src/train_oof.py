# ============================================================
# ARQUIVO: train_oof.py
#
# O QUE FAZ: reproduz os 5 folds de validação cruzada de
# train_classifier.py (mesmo StratifiedGroupKFold, seed=42, mesma
# arquitetura, hiperparâmetros e número de épocas), mas em vez de
# descartar o modelo de cada fold ao final (como train_classifier.py
# faz), salva cada um em models/oof/fold_{k}.pt e grava qual vídeo
# ficou de teste em qual fold (models/oof/fold_assignments.json).
#
# Isso permite avaliar cada vídeo do dataset pelo modelo do fold em
# que ele NUNCA entrou no treino ("out-of-fold") -- uma avaliação sem
# vazamento, coisa que o classifier.pt final (treinado com os 56
# vídeos inteiros) não permite mais.
#
# POR QUE AS DEFINIÇÕES ABAIXO SÃO UMA CÓPIA LITERAL DE
# train_classifier.py, EM VEZ DE IMPORTADAS: esse arquivo não tem
# guarda "if __name__ == '__main__'" -- importá-lo executaria o
# treino inteiro e sobrescreveria models/classifier.pt na hora.
# Por instrução explícita, train_classifier.py não pode ser alterado
# e models/classifier.pt não pode ser tocado por este script. A cópia
# é mantida IDÊNTICA de propósito, para a reprodução ser fiel.
#
# VERIFICAÇÃO: ao final, compara a acurácia e as métricas por classe
# de cada fold recém-treinado com as já salvas em
# data/results/cv_metrics.json. train_classifier.py fixa a seed só do
# StratifiedGroupKFold (o split de vídeos por fold) -- os pesos
# iniciais da LSTM e a ordem dos batches (DataLoader shuffle=True)
# não são fixados por nenhuma seed lá, então mesmo uma reprodução
# fiel do treino tende a divergir um pouco dessas métricas por causa
# disso, não por bug. O script imprime e explica a diferença, não a
# esconde.
# ============================================================

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset, DataLoader

# ==========================
# CONFIG -- idêntico a train_classifier.py
# ==========================

DATASET_DIR = Path("data/datasets")
RESULTS_DIR = Path("data/results")
OOF_DIR = Path("models/oof")

OOF_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
N_SPLITS = 5

# Seed do PRÓPRIO script, para ELE (train_oof.py) ser reprodutível em
# re-execuções futuras. Não faz este script bater com a execução
# passada de train_classifier.py, que não fixava isso -- ver nota de
# verificação no fim do arquivo.
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================
# LOAD DATA -- idêntico a train_classifier.py
# ==========================

X = np.load(DATASET_DIR / "X.npy")
y = np.load(DATASET_DIR / "y.npy")
groups = np.load(DATASET_DIR / "groups.npy", allow_pickle=True)

with open(DATASET_DIR / "labels.json", "r", encoding="utf-8") as f:
  labels = json.load(f)

num_classes = len(labels)
id_to_label = {v: k for k, v in labels.items()}
class_names = [id_to_label[i] for i in range(num_classes)]

print("X shape:", X.shape)
print("Unique videos:", len(np.unique(groups)))

# ==========================
# DATASET / MODELO / TREINO / AVALIAÇÃO
# -- cópia literal de train_classifier.py (ver aviso no cabeçalho)
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

    self.lstm = nn.LSTM(
      input_size=input_size,
      hidden_size=hidden_size,
      num_layers=num_layers,
      batch_first=True
    )

    self.fc = nn.Linear(hidden_size, num_classes)

  def forward(self, x):
    output, (hidden, _) = self.lstm(x)
    hidden = hidden[-1]
    return self.fc(hidden)


def build_model():
  return LSTMClassifier(
    input_size=X.shape[2],
    hidden_size=128,
    num_layers=2,
    num_classes=num_classes
  ).to(device)


def train_model(model, loader, log_prefix=""):

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


def evaluate_model(model, loader):

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
# K-FOLD -- mesmo split de train_classifier.py (StratifiedGroupKFold, seed=42)
# ==========================

sgkf = StratifiedGroupKFold(
  n_splits=N_SPLITS,
  shuffle=True,
  random_state=42
)

fold_assignment = {}  # video_id -> fold em que ficou de TESTE (out-of-fold)
fold_accuracies = []
fold_reports = []

for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups), start=1):

  test_videos_fold = sorted(set(groups[test_idx]))

  for video in test_videos_fold:
    fold_assignment[video] = fold

  print(f"\n===== Fold {fold}/{N_SPLITS} =====")
  print(f"Train sequences: {len(train_idx)} | Test sequences: {len(test_idx)}")
  print(f"Test videos ({len(test_videos_fold)}): {test_videos_fold}")

  X_train, y_train = X[train_idx], y[train_idx]
  X_test, y_test = X[test_idx], y[test_idx]

  train_loader = DataLoader(
    AutismDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True
  )

  test_loader = DataLoader(
    AutismDataset(X_test, y_test),
    batch_size=BATCH_SIZE
  )

  model = build_model()

  train_model(model, train_loader, log_prefix=f"[Fold {fold}] ")

  targets, predictions = evaluate_model(model, test_loader)

  accuracy = 100 * (predictions == targets).mean()
  fold_accuracies.append(accuracy)

  print(f"[Fold {fold}] Test Accuracy: {accuracy:.2f}%")

  fold_reports.append(
    classification_report(
      targets,
      predictions,
      labels=list(range(num_classes)),
      target_names=class_names,
      output_dict=True,
      zero_division=0
    )
  )

  checkpoint_path = OOF_DIR / f"fold_{fold}.pt"
  torch.save(model.state_dict(), checkpoint_path)
  print(f"Saved {checkpoint_path}")

# vídeo -> fold em que foi teste (usado depois para escolher qual
# checkpoint usar na avaliação/demo de cada vídeo do dataset)
with open(OOF_DIR / "fold_assignments.json", "w", encoding="utf-8") as f:
  json.dump({
    "n_splits": N_SPLITS,
    "random_state": 42,
    "video_to_fold": fold_assignment,
  }, f, indent=2, ensure_ascii=False)

print(f"\nSaved fold assignments to {OOF_DIR / 'fold_assignments.json'}")

# ==========================
# AGREGAÇÃO (mesma lógica de train_classifier.py) + SALVAMENTO
# ==========================

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


new_cv_metrics = {
  "n_splits": N_SPLITS,
  "fold_accuracies": fold_accuracies,
  "mean_accuracy": float(np.mean(fold_accuracies)),
  "std_accuracy": float(np.std(fold_accuracies)),
  "per_class": {class_name: metric_block(class_name) for class_name in class_names},
  "macro_avg": metric_block("macro avg"),
  "weighted_avg": metric_block("weighted avg"),
}

with open(OOF_DIR / "cv_metrics_oof.json", "w", encoding="utf-8") as f:
  json.dump(new_cv_metrics, f, indent=2, ensure_ascii=False)

print(f"Saved {OOF_DIR / 'cv_metrics_oof.json'}")

# ==========================
# VERIFICAÇÃO CONTRA data/results/cv_metrics.json
# ==========================

verification_lines = []
verification_lines.append("Verificação: train_oof.py vs data/results/cv_metrics.json")
verification_lines.append("=" * 70)

cv_metrics_path = RESULTS_DIR / "cv_metrics.json"

if not cv_metrics_path.exists():

  verification_lines.append(
    f"[AVISO] {cv_metrics_path} não existe -- não é possível comparar. "
    "Rode train_classifier.py primeiro."
  )

else:

  with open(cv_metrics_path, "r", encoding="utf-8") as f:
    existing = json.load(f)

  verification_lines.append("\nAcurácia por fold:")
  verification_lines.append(f"{'Fold':<6}{'cv_metrics.json':<18}{'train_oof.py':<18}{'diferença':<12}")

  for i in range(N_SPLITS):
    old_acc = existing["fold_accuracies"][i]
    new_acc = fold_accuracies[i]
    verification_lines.append(
      f"{i+1:<6}{old_acc:<18.2f}{new_acc:<18.2f}{new_acc - old_acc:+.2f}"
    )

  verification_lines.append(
    f"\nMédia: {existing['mean_accuracy']:.2f}% (+/- {existing['std_accuracy']:.2f}%) "
    f"-> {new_cv_metrics['mean_accuracy']:.2f}% (+/- {new_cv_metrics['std_accuracy']:.2f}%)"
  )

  verification_lines.append("\nPrecisão/Recall/F1 médios por classe (cv_metrics.json -> train_oof.py):")

  for class_name in class_names:
    old_m = existing["per_class"][class_name]
    new_m = new_cv_metrics["per_class"][class_name]
    verification_lines.append(
      f"  {class_name:<15} "
      f"precision {old_m['precision_mean']:.3f} -> {new_m['precision_mean']:.3f} | "
      f"recall {old_m['recall_mean']:.3f} -> {new_m['recall_mean']:.3f} | "
      f"f1 {old_m['f1_mean']:.3f} -> {new_m['f1_mean']:.3f} | "
      f"support {old_m['support']:.0f} -> {new_m['support']:.0f}"
    )

  verification_lines.append(
    "\nNOTA: train_classifier.py fixa a seed só do StratifiedGroupKFold "
    "(random_state=42) -- por isso os vídeos de teste de cada fold batem "
    "exatamente (ver 'Test videos' impresso acima para cada fold). Mas os "
    "pesos iniciais da LSTM e a ordem dos batches (DataLoader shuffle=True) "
    "não são fixados por nenhuma seed em train_classifier.py, então mesmo "
    "esta reprodução fiel do treino tende a divergir um pouco dessas "
    "acurácias/métricas -- isso é esperado (variância normal de treino), não "
    "indica que este script reproduziu o treino errado. Se a diferença for "
    "grande (dezenas de pontos percentuais) ou o 'support' por classe não "
    "bater, aí sim indica um problema real na reprodução."
  )

verification_text = "\n".join(verification_lines)

print("\n" + verification_text)

with open(RESULTS_DIR / "oof_verification.txt", "w", encoding="utf-8") as f:
  f.write(verification_text + "\n")

print(f"\nVerification saved to {RESULTS_DIR / 'oof_verification.txt'}")
