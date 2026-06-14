# treina a LSTM

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# ==========================
# CONFIG
# ==========================

DATASET_DIR = Path("data/datasets")
MODEL_DIR = Path("models")

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

with open(
  DATASET_DIR / "labels.json",
  "r",
  encoding="utf-8"
) as f:
  labels = json.load(f)

num_classes = len(labels)

print("X shape:", X.shape)
print("y shape:", y.shape)

# ==========================
# TRAIN / TEST SPLIT
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
  X,
  y,
  test_size=0.2,
  random_state=42,
  stratify=y
)

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

correct = 0
total = 0

with torch.no_grad():

  for X_batch, y_batch in test_loader:

    X_batch = X_batch.to(device)
    y_batch = y_batch.to(device)

    outputs = model(X_batch)

    predictions = torch.argmax(
      outputs,
      dim=1
    )

    total += y_batch.size(0)

    correct += (
      predictions == y_batch
    ).sum().item()

accuracy = 100 * correct / total

print(
  f"\nTest Accuracy: "
  f"{accuracy:.2f}%"
)

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