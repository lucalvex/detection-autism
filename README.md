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

## Hook de pre-commit (bloqueia vídeos/CSVs/arrays grandes)

O repositório versiona um hook de pre-commit em `.githooks/pre-commit`
que bloqueia o commit se algum arquivo de vídeo (`.mp4`/`.avi`/`.mov`/
`.mkv`), CSV, array de dados (`.npy`/`.npz`) ou qualquer arquivo acima
de 5MB estiver sendo adicionado — esses arquivos são grandes e
regeneráveis pelo pipeline, e não devem ir para o GitHub. Git não ativa
hooks versionados sozinho; rode isto uma vez depois de clonar:

```bash
git config core.hooksPath .githooks
```

Para pular a checagem em um commit específico (uso consciente):
`git commit --no-verify`.

## Rodar tudo de uma vez

```bash
python src/run_pipeline.py
```

Roda as 5 etapas abaixo em sequência (cada uma como o mesmo comando que você
rodaria manualmente) e termina gerando o relatório final consolidado em
`data/results/report.pdf`. Use isso sempre que quiser regenerar tudo do zero
(ex. depois de adicionar vídeos novos). A etapa de análise de episódios é
opcional — se falhar (ex. nenhuma pessoa detectada no vídeo configurado), o
pipeline continua e o relatório só marca essa seção como indisponível.

## Ordem de execução

Se preferir rodar (ou depurar) uma etapa por vez, execute os scripts abaixo
nessa ordem, a partir da raiz do projeto.

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

### 3. Treinamento e avaliação (k-fold cross-validation)

```bash
python src/train_classifier.py
```

Faz validação cruzada **`StratifiedGroupKFold` (k=5)**: em cada um dos 5
folds, todas as sequências de um mesmo vídeo ficam sempre do mesmo lado
(evitando vazamento de dados entre janelas sobrepostas), treinando uma LSTM
nova por fold. Ao final, treina um modelo final com todos os dados e gera:
- `models/classifier.pt` — modelo final treinado (usado pelas ferramentas de
  predição)
- `data/results/classification_report.txt` — precisão, recall e F1 por
  classe (média ± desvio padrão entre os 5 folds), com médias macro e
  weighted
- `data/results/classification_report_per_fold.txt` — relatório detalhado de
  cada fold individualmente
- `data/results/cv_metrics.json` — os mesmos números agregados, em JSON
  (usado por `generate_report.py`)
- `data/results/confusion_matrix.png` — matriz de confusão 6x6 (heatmap),
  acumulada sobre os 5 folds

### 4. Análise de episódios (opcional)

```bash
python src/utils/episode_analysis.py
```

Roda pose + LSTM sobre um vídeo (configurado em `VIDEO_PATH` no topo do
arquivo) e agrupa a sequência de predições em episódios (frequência e
duração por classe). Veja `src/utils/episode_analysis.py` para detalhes do
método (suavização temporal + filtro de duração mínima). Gera:
- `data/results/episodes.csv` — 1 linha por episódio detectado
- `data/results/episode_summary.txt` / `episode_summary.json` — frequência e
  duração agregadas por classe

### 5. Relatório final

```bash
python src/generate_report.py
```

Consolida os resultados das etapas acima em `data/results/report.pdf`, com
1 seção por página: capa, resumo do dataset, validação cruzada, matriz de
confusão, frequência/duração de episódios e timeline gráfica dos episódios.
Se alguma etapa anterior não tiver rodado ainda, a seção correspondente
aparece no PDF avisando qual comando rodar, em vez de quebrar o relatório.

## Regras práticas

- Ao adicionar ou trocar vídeos em `data/videos/`, rode `python
  src/run_pipeline.py` (ou os passos 1-3 manualmente) do zero — cada etapa
  sobrescreve totalmente a saída da anterior.
- Rodar só o `train_classifier.py` de novo (sem repetir os passos 1-2)
  é seguro apenas se você quiser re-treinar com os mesmos dados (ex. mudar
  hiperparâmetros).
- Rodar só o `generate_report.py` de novo é sempre seguro — ele só lê os
  arquivos já gerados, não recalcula nada.

## Ferramentas auxiliares (`src/utils/`)

Scripts que **não** fazem parte do pipeline principal de treino — servem
para inspeção visual, ilustrações para o TCC ou para usar o modelo já
treinado. Rode só quando precisar de algo específico:

```bash
python src/utils/predict_video.py        # classifica o comportamento em um vídeo, exibindo o rótulo previsto ao vivo
python src/utils/episode_analysis.py     # agrupa as predições em episódios (frequência/duração) -- veja passo 4 acima
python src/utils/realtime_pose.py        # mostra os keypoints/esqueleto da YOLO desenhados ao vivo em um vídeo (sem classificação)
python src/utils/save_annotated_video.py # salva um vídeo com os keypoints/esqueleto desenhados em cada frame
python src/utils/save_keyframes.py       # salva PNGs de frames das 3 classes (ArmFlapping/HeadBanging/Spinning) com só o esqueleto, sem a pessoa do vídeo
python src/utils/save_pose_image.py      # salva um PNG de uma imagem única, anotada com os keypoints
python src/utils/keypoints.py            # imprime no console os keypoints brutos de um vídeo (script de teste rápido)
```
