# treina a LSTM

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# ==========================
# CONFIG
# ==========================

DATASET_DIR = Path("data/datasets")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("data/results")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001

MODEL_DIR.mkdir(parents=True, exist_ok=True)

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

X = np.load(DATASET_DIR / "X.npy")
y = np.load(DATASET_DIR / "y.npy")
groups = np.load(DATASET_DIR / "groups.npy", allow_pickle=True)

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
# TRAIN / TEST SPLIT (por vídeo, não por sequência)
# ==========================
# A janela deslizante gera sequências fortemente sobrepostas do mesmo
# vídeo. Se o split fosse feito nas sequências, sequências quase
# idênticas do mesmo vídeo poderiam cair uma no treino e outra no
# teste, vazando informação e inflando a acurácia. Por isso o split
# é feito nos vídeos (grupos) inteiros, ANTES de decidir quais
# sequências vão para treino/teste — todas as sequências de um vídeo
# ficam sempre no mesmo lado.

unique_videos = np.unique(groups)

# label de cada vídeo (todas as sequências de um vídeo têm o mesmo y)
video_labels = np.array([
  y[groups == video][0]
  for video in unique_videos
])

train_videos, test_videos = train_test_split(
  unique_videos,
  test_size=0.2,
  random_state=42,
  stratify=video_labels
)

train_mask = np.isin(groups, train_videos)
test_mask = np.isin(groups, test_videos)

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]

print(f"Train videos: {len(train_videos)} ({train_mask.sum()} sequences)")
print(f"Test videos: {len(test_videos)} ({test_mask.sum()} sequences)")

# ==========================
# DATASET
# ==========================

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


train_dataset = AutismDataset(
  X_train,
  y_train
)

test_dataset = AutismDataset(
  X_test,
  y_test
)

train_loader = DataLoader(
  train_dataset,
  batch_size=BATCH_SIZE,
  shuffle=True
)

test_loader = DataLoader(
  test_dataset,
  batch_size=BATCH_SIZE
)

# ==========================
# MODEL
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

    output, (hidden, _) = self.lstm(x)

    hidden = hidden[-1]

    return self.fc(hidden)


input_size = X.shape[2]

model = LSTMClassifier(
  input_size=input_size,
  hidden_size=128,
  num_layers=2,
  num_classes=num_classes
).to(device)

# ==========================
# LOSS / OPTIMIZER
# ==========================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
  model.parameters(),
  lr=LEARNING_RATE
)

# ==========================
# TRAINING
# ==========================

for epoch in range(EPOCHS):

  model.train()

  running_loss = 0

  for X_batch, y_batch in train_loader:

    X_batch = X_batch.to(device)
    y_batch = y_batch.to(device)

    optimizer.zero_grad()

    outputs = model(X_batch)

    loss = criterion(
        outputs,
        y_batch
    )

    loss.backward()

    optimizer.step()

    running_loss += loss.item()

  avg_loss = running_loss / len(train_loader)

  print(
    f"Epoch [{epoch+1}/{EPOCHS}] "
    f"Loss: {avg_loss:.4f}"
  )

# ==========================
# EVALUATION
# ==========================

model.eval()

all_predictions = []
all_targets = []

with torch.no_grad():

  for X_batch, y_batch in test_loader:

    X_batch = X_batch.to(device)
    y_batch = y_batch.to(device)

    outputs = model(X_batch)

    predictions = torch.argmax(
      outputs,
      dim=1
    )

    all_predictions.extend(predictions.cpu().numpy())
    all_targets.extend(y_batch.cpu().numpy())

all_predictions = np.array(all_predictions)
all_targets = np.array(all_targets)

accuracy = 100 * (all_predictions == all_targets).mean()

print(
  f"\nTest Accuracy: "
  f"{accuracy:.2f}%"
)

# ==========================
# PRECISION / RECALL / F1 (por classe + macro/weighted)
# ==========================

report = classification_report(
  all_targets,
  all_predictions,
  labels=list(range(num_classes)),
  target_names=class_names,
  digits=4,
  zero_division=0
)

print("\nClassification Report:")
print(report)

with open(RESULTS_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
  f.write(report)

# ==========================
# MATRIZ DE CONFUSÃO
# ==========================

cm = confusion_matrix(
  all_targets,
  all_predictions,
  labels=list(range(num_classes))
)

plt.figure(figsize=(8, 6))

sns.heatmap(
  cm,
  annot=True,
  fmt="d",
  cmap="Blues",
  xticklabels=class_names,
  yticklabels=class_names
)

plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix")
plt.tight_layout()

plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=200)
plt.close()

print(f"\nConfusion matrix saved to {RESULTS_DIR / 'confusion_matrix.png'}")

# ==========================
# SAVE MODEL
# ==========================

torch.save(
  model.state_dict(),
  MODEL_DIR / "classifier.pt"
)

print(
  "Model saved to "
  "models/classifier.pt"
)