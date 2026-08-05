# Detecção de Comportamentos Motores Estereotipados em TEA

Pipeline de detecção de comportamentos motores estereotipados associados ao
Transtorno do Espectro Autista (TEA) a partir de vídeo, usando estimativa de
pose (YOLO11n-Pose) + classificação temporal (LSTM).

Dataset: SSBD, 75 vídeos, 6 classes — `ArmFlapping`, `HeadBanging`, `Rocking`,
`HandMovement`, `Spinning`, `Normal`.

## Requisitos

```bash
python -m venv .venv
.venv\Scripts\activate
pip install ultralytics opencv-python pandas numpy torch scikit-learn matplotlib seaborn
```

Coloque o modelo de pose em `models/yolo11n-pose.pt` e os vídeos em
`data/videos/`.

## Ordem de execução

Execute os scripts abaixo sempre nessa ordem, a partir da raiz do projeto.

### 1. Extração de pose

```bash
python src/extract_pose.py
```

Processa todos os `.mp4` em `data/videos/` e gera um CSV de keypoints por
vídeo em `data/poses/` (nomeado pelo `video_id`, ex. `v_HeadBanging_18.csv`).
É o passo mais lento, pois roda o YOLO frame a frame em cada vídeo.

### 2. Criação das sequências

```bash
python src/create_sequences.py
```

Lê todos os CSVs de `data/poses/`, aplica a janela deslizante (30 frames) e
gera em `data/datasets/`:
- `X.npy`, `y.npy` — sequências e labels
- `groups.npy` — vídeo de origem de cada sequência (usado no split treino/teste)
- `labels.json` — mapeamento classe → id

### 3. Treinamento e avaliação

```bash
python src/train_classifier.py
```

Faz o split treino/teste **por vídeo** (todas as sequências de um mesmo
vídeo ficam sempre do mesmo lado, evitando vazamento de dados entre janelas
sobrepostas), treina a LSTM e gera:
- `models/classifier.pt` — modelo treinado
- `data/results/classification_report.txt` — precisão, recall e F1 por
  classe, com médias macro e weighted
- `data/results/confusion_matrix.png` — matriz de confusão 6x6 (heatmap)

## Regras práticas

- Ao adicionar ou trocar vídeos em `data/videos/`, rode os 3 passos do zero
  nessa ordem — cada passo sobrescreve totalmente a saída do anterior.
- Rodar só o `train_classifier.py` de novo (sem repetir os dois primeiros)
  é seguro apenas se você quiser re-treinar com os mesmos dados (ex. mudar
  hiperparâmetros).

## Predição em vídeo / tempo real

```bash
python src/predict_video.py     # roda sobre um arquivo de vídeo
python src/realtime_pose.py     # roda sobre webcam
```
