# ============================================================
# ARQUIVO: train_classifier.py
#
# O QUE FAZ: terceira e última etapa do pipeline. Carrega o dataset
# gerado por create_sequences.py (X.npy, y.npy, groups.npy) e faz
# validação cruzada k-fold (StratifiedGroupKFold, k=5): 5 vezes,
# treina um LSTMClassifier do zero em 4/5 dos vídeos e avalia no 1/5
# restante, garantindo que nenhum vídeo apareça ao mesmo tempo em
# treino e teste em nenhum fold (evita vazamento por sobreposição das
# janelas deslizantes). No final:
#   - agrega acurácia, precisão/recall/F1 por classe (média ± desvio
#     padrão entre os 5 folds) -> data/results/classification_report.txt
#   - soma as 5 matrizes de confusão (equivale à matriz do dataset
#     inteiro sob validação cruzada) -> data/results/confusion_matrix.png
#   - treina um modelo final com TODOS os dados (sem held-out) e salva
#     em models/classifier.pt, para uso em predict_video.py /
#     realtime_pose.py
# ============================================================

# treina a LSTM com k-fold cross-validation

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset, DataLoader

# ==========================
# CONFIG
# ==========================

DATASET_DIR = Path("data/datasets")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("data/results")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 30            # épocas de treino por fold (e também do modelo final)
LEARNING_RATE = 0.001
N_SPLITS = 5           # número de folds da validação cruzada

# ==========================
# DEVICE
# ==========================

device = torch.device(
  "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

# ==========================
# LOAD DATA
# ==========================

X = np.load(DATASET_DIR / "X.npy")        # shape (n_sequencias, 30, 68)
y = np.load(DATASET_DIR / "y.npy")        # label (0-5) de cada sequência
groups = np.load(DATASET_DIR / "groups.npy", allow_pickle=True)  # video_id de cada sequência

with open(
  DATASET_DIR / "labels.json",
  "r",
  encoding="utf-8"
) as f:
  labels = json.load(f)

num_classes = len(labels)

# nomes das classes na ordem dos ids (0..num_classes-1), para relatórios/plots
id_to_label = {v: k for k, v in labels.items()}
class_names = [id_to_label[i] for i in range(num_classes)]

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Unique videos:", len(np.unique(groups)))

# ==========================
# DATASET
# ==========================
# Wrapper simples que converte os arrays numpy em tensores PyTorch e
# permite iterar em batches via DataLoader.

class AutismDataset(Dataset):

  def __init__(self, X, y):

    self.X = torch.tensor(
      X,
      dtype=torch.float32
    )

    self.y = torch.tensor(
      y,
      dtype=torch.long
    )

  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx):
    return self.X[idx], self.y[idx]

# ==========================
# MODEL
# ==========================
# LSTM de 2 camadas: recebe a sequência (30 frames x 68 features) e
# usa o estado oculto final (resumo da sequência inteira) para prever
# a classe via uma camada linear.

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

    output, (hidden, _) = self.lstm(x)

    # hidden[-1] = estado oculto da última camada da LSTM no último frame da sequência
    hidden = hidden[-1]

    return self.fc(hidden)


def build_model():
  # cria um modelo NOVO (pesos aleatórios reiniciados) — chamado 1x por fold,
  # para nenhum fold aproveitar conhecimento treinado no fold anterior
  return LSTMClassifier(
    input_size=X.shape[2],
    hidden_size=128,
    num_layers=2,
    num_classes=num_classes
  ).to(device)


def train_model(model, loader, log_prefix=""):

  criterion = nn.CrossEntropyLoss()

  optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
  )

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

  model.eval()  # desliga comportamento de treino (aqui não muda nada, mas é a convenção correta)

  all_predictions = []
  all_targets = []

  with torch.no_grad():  # não precisa calcular gradiente durante avaliação

    for X_batch, y_batch in loader:

      X_batch = X_batch.to(device)

      outputs = model(X_batch)

      predictions = torch.argmax(outputs, dim=1)

      all_predictions.extend(predictions.cpu().numpy())
      all_targets.extend(y_batch.numpy())

  return np.array(all_targets), np.array(all_predictions)


# ==========================
# K-FOLD CROSS-VALIDATION (StratifiedGroupKFold)
# ==========================
# Mesmo princípio do split video-wise: StratifiedGroupKFold garante que
# todas as sequências de um vídeo (mesmo "group") fiquem inteiramente em
# um único lado de cada fold, evitando o vazamento por sobreposição de
# janelas. Repetir o treino/avaliação em 5 folds dá uma estimativa mais
# robusta do desempenho do que um único split 80/20 fixo, além de reduzir
# a chance de tirar conclusões de um split "sortudo" ou "azarado".

sgkf = StratifiedGroupKFold(
  n_splits=N_SPLITS,
  shuffle=True,
  random_state=42  # fixa a aleatoriedade -> os mesmos 5 folds toda vez que o script rodar
)

fold_accuracies = []
fold_reports = []       # 1 dict de métricas (classification_report) por fold, para agregar depois
fold_report_texts = []  # 1 relatório em texto por fold, para o arquivo de detalhe por fold
cumulative_cm = np.zeros((num_classes, num_classes), dtype=int)  # soma das matrizes de confusão

# sgkf.split devolve, para cada um dos 5 folds, os índices de X/y/groups
# que vão para treino (train_idx) e para teste (test_idx) NAQUELE fold
for fold, (train_idx, test_idx) in enumerate(
  sgkf.split(X, y, groups), start=1
):

  print(f"\n===== Fold {fold}/{N_SPLITS} =====")
  print(f"Train sequences: {len(train_idx)} | Test sequences: {len(test_idx)}")

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

  model = build_model()  # LSTM nova para este fold

  train_model(model, train_loader, log_prefix=f"[Fold {fold}] ")

  targets, predictions = evaluate_model(model, test_loader)

  accuracy = 100 * (predictions == targets).mean()
  fold_accuracies.append(accuracy)

  print(f"[Fold {fold}] Test Accuracy: {accuracy:.2f}%")

  # classification_report como dict (output_dict=True) para poder calcular média/desvio depois
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

  # a mesma info, mas como texto formatado, para o relatório detalhado por fold
  fold_report_texts.append(
    classification_report(
      targets,
      predictions,
      labels=list(range(num_classes)),
      target_names=class_names,
      digits=4,
      zero_division=0
    )
  )

  # acumula a matriz de confusão deste fold na matriz total
  cumulative_cm += confusion_matrix(
    targets,
    predictions,
    labels=list(range(num_classes))
  )

# ==========================
# AGREGAÇÃO DAS MÉTRICAS ENTRE FOLDS (média ± desvio padrão)
# ==========================

def cell(mean, std, width=18):
  # formata "média±desvio" alinhado em colunas de largura fixa, para o arquivo de texto
  return f"{mean:.4f}±{std:.4f}".ljust(width)


def aggregate(row_key, metric_key):
  # pega o valor de 1 métrica (ex.: precision da classe "Rocking") nos 5 folds e calcula média/desvio
  values = [report[row_key][metric_key] for report in fold_reports]
  return np.mean(values), np.std(values)


report_lines = []

report_lines.append(
  f"K-Fold Cross-Validation (StratifiedGroupKFold, k={N_SPLITS})"
)
report_lines.append(
  "Fold accuracies: " +
  ", ".join(f"{a:.2f}%" for a in fold_accuracies)
)
report_lines.append(
  f"Mean accuracy: {np.mean(fold_accuracies):.2f}% "
  f"(+/- {np.std(fold_accuracies):.2f}%)"
)
report_lines.append("")
report_lines.append(
  f"{'':<15}{'precision':<18}{'recall':<18}{'f1-score':<18}{'support':>10}"
)

# 1 linha por classe: precisão/recall/F1 agregados (média ± desvio) entre os 5 folds
for class_name in class_names:

  p_mean, p_std = aggregate(class_name, "precision")
  r_mean, r_std = aggregate(class_name, "recall")
  f_mean, f_std = aggregate(class_name, "f1-score")
  support_total = sum(
    report[class_name]["support"] for report in fold_reports
  )

  report_lines.append(
    f"{class_name:<15}"
    f"{cell(p_mean, p_std)}"
    f"{cell(r_mean, r_std)}"
    f"{cell(f_mean, f_std)}"
    f"{support_total:>10.0f}"
  )

report_lines.append("")

# médias agregadas: macro (todas as classes pesam igual) e weighted (pondera pelo nº de amostras)
for avg_key in ["macro avg", "weighted avg"]:

  p_mean, p_std = aggregate(avg_key, "precision")
  r_mean, r_std = aggregate(avg_key, "recall")
  f_mean, f_std = aggregate(avg_key, "f1-score")
  support_total = sum(
    report[avg_key]["support"] for report in fold_reports
  )

  report_lines.append(
    f"{avg_key:<15}"
    f"{cell(p_mean, p_std)}"
    f"{cell(r_mean, r_std)}"
    f"{cell(f_mean, f_std)}"
    f"{support_total:>10.0f}"
  )

aggregated_report = "\n".join(report_lines)

print("\n" + aggregated_report)

with open(
  RESULTS_DIR / "classification_report.txt",
  "w",
  encoding="utf-8"
) as f:
  f.write(aggregated_report + "\n")

# versão em JSON dos mesmos números agregados, para consumo programático
# (usado por generate_report.py, sem precisar reprocessar o texto acima)

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

with open(
  RESULTS_DIR / "cv_metrics.json",
  "w",
  encoding="utf-8"
) as f:
  json.dump(cv_metrics, f, indent=2, ensure_ascii=False)

# relatório detalhado por fold, para apêndice/consulta
with open(
  RESULTS_DIR / "classification_report_per_fold.txt",
  "w",
  encoding="utf-8"
) as f:

  for fold, report_text in enumerate(fold_report_texts, start=1):

    f.write(f"===== Fold {fold}/{N_SPLITS} (accuracy: {fold_accuracies[fold-1]:.2f}%) =====\n")
    f.write(report_text)
    f.write("\n")

# ==========================
# MATRIZ DE CONFUSÃO (agregada sobre os 5 folds)
# ==========================
# Como os folds particionam o dataset inteiro (cada sequência é teste
# em exatamente 1 fold), a soma das matrizes de confusão dos 5 folds
# equivale à matriz de confusão do dataset inteiro sob validação cruzada.

plt.figure(figsize=(8, 6))

sns.heatmap(
  cumulative_cm,
  annot=True,
  fmt="d",  # anota os valores como inteiros
  cmap="Blues",
  xticklabels=class_names,
  yticklabels=class_names
)

plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title(f"Confusion Matrix ({N_SPLITS}-Fold Cross-Validation, cumulative)")
plt.tight_layout()

plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=200)
plt.close()

print(f"\nConfusion matrix saved to {RESULTS_DIR / 'confusion_matrix.png'}")

# ==========================
# MODELO FINAL (treinado em todos os dados, para inferência/deploy)
# ==========================
# A validação cruzada acima estima o desempenho esperado em vídeos não
# vistos. Para uso em predict_video.py/realtime_pose.py, o modelo final
# é treinado com todo o dataset disponível (sem held-out), já que sua
# capacidade de generalização já foi estimada pelo k-fold.

print("\nTraining final model on the full dataset...")

final_loader = DataLoader(
  AutismDataset(X, y),
  batch_size=BATCH_SIZE,
  shuffle=True
)

final_model = build_model()

train_model(final_model, final_loader, log_prefix="[Final model] ")

torch.save(
  final_model.state_dict(),
  MODEL_DIR / "classifier.pt"
)

print("Model saved to models/classifier.pt")
