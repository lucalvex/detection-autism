# Detecção de comportamentos estereotipados em vídeo

Trabalho de Conclusão de Curso (UEM) sobre a detecção automática de
comportamentos motores estereotipados associados ao Transtorno do Espectro
Autista (TEA) em vídeo: *arm flapping* (balançar os braços), *head banging*
(bater a cabeça) e *spinning* (girar o corpo).

O método tem duas etapas:

1. **Pose**: o YOLO11n-Pose extrai, quadro a quadro, 17 pontos do corpo da
   pessoa no vídeo.
2. **Classificação temporal**: uma LSTM recebe janelas de 4 s desses pontos
   (normalizados e reamostrados a 15 quadros/s) e dá uma pontuação para cada
   classe e para o fundo (sem estereotipia). Os trechos detectados passam por
   um pós-processamento (mediana de 2 s e junção de trechos separados por
   menos de 1 s).

Os dados vêm do SSBD
(*Self-Stimulatory Behaviours Dataset*). São usados 56 vídeos, com rótulos por
intervalo de tempo, e a avaliação é feita com validação cruzada por vídeo em
5 partições. A escolha do modelo final e os números estão em
[`experimentos/modelo-final.md`](experimentos/modelo-final.md).

> **Privacidade.** Os vídeos do SSBD mostram crianças. Nenhum vídeo, quadro
> de vídeo, imagem, dado de pose ou modelo treinado é versionado: o
> `.gitignore` e o hook de pre-commit (`.githooks/`) bloqueiam esses
> arquivos. A demo só atende em `127.0.0.1`, e o vídeo aberto nela fica só na
> memória do navegador.

## Estrutura

```
src/                     código base: extração de pose, atributos, anotações do SSBD,
                         segmentação em trechos, avaliação temporal e exportação para a demo
experimentos/            rodadas de experimentos, protocolos, resumos e métricas de cada um
  rodar_tudo.py          rodada 1 (etapas 01 a 10) e funções de preparo e treino usadas por todas
  rodar_rodada2.py ...   rodadas 2, 3, critério Q, 4, 4b e 4c (cada uma importa a anterior)
  treinar_final_completo.py   treina o modelo da câmera ao vivo com os 56 vídeos
  <id>/                  um experimento: config.json, métricas, perdas e curva de erro
  protocolo*.md, resumo*.md   o que foi decidido antes de rodar e o resultado de cada rodada
demo/                    interface local (modos Vídeos e Câmera ao vivo)
tests/                   testes
data/results/            métricas do modelo de partida (rodada A e base)
models/oof*/             partições por vídeo e métricas dos modelos de partida
```

## Instalação

Python 3.11 e, para um dos testes, Node.js.

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
git config core.hooksPath .githooks   # ativa o hook que bloqueia vídeos, imagens e dados
```

Os dados não estão no repositório:

- **Vídeos do SSBD** em `data/videos/<id>.mp4` (ex.: `v_ArmFlapping_12.mp4`).
- **Anotações do SSBD** (XML) em `../download-yt/xmls/`, uma pasta ao lado
  deste repositório.
- **Pesos do YOLO11n-Pose** em `models/yolo11n-pose.pt`, baixados da
  [Ultralytics](https://docs.ultralytics.com/tasks/pose/).

## Como rodar

Todos os comandos são rodados a partir da raiz do repositório. Os
experimentos são retomáveis: um experimento que já tem
`metricas-validacao.json` é pulado.

**1. Modelo de partida** (gera `data/poses/`, `data/datasets_annotated/` e os
checkpoints que a avaliação temporal usa):

```bash
python src/extract_pose.py
python src/create_sequences_annotated.py
python src/train_oof_seeded.py --dataset-dir data/datasets_annotated --output-dir models/oof-seeded-annotated --subsample-background
```

**2. Experimentos**, em ordem (cada rodada parte da anterior):

```bash
python experimentos/rodar_tudo.py
python experimentos/rodar_rodada2.py
python experimentos/rodar_rodada3.py
python experimentos/rodar_q.py
python experimentos/rodar_rodada4.py
python experimentos/rodar_rodada4b.py
python experimentos/rodar_rodada4c.py
```

A rodada 4 lê `experimentos/qualidade-dataset/tabela-videos.csv`, gerada
por `experimentos/qualidade-dataset/qualidade_dataset.py` (precisa do `ffprobe`).

**3. Modelo da câmera ao vivo** (`models/final_completo.pt`, o modelo final
`r4b-C1-janela-4s` treinado com os 56 vídeos):

```bash
python experimentos/treinar_final_completo.py
```

**4. Demo**:

```bash
python src/exportar_sessoes_final.py   # pontuações e pontos de cada vídeo em saidas/final/
python demo/servir.py                  # http://127.0.0.1:8765
```

- No modo **Vídeos**, você escolhe um vídeo do SSBD e abre o arquivo dele
  com o seletor da página. A interface mostra o esqueleto, as pontuações e
  os trechos detectados, comparados com a anotação.
- No modo **Câmera ao vivo**, a webcam manda um quadro por vez ao servidor,
  que roda o YOLO e o modelo completo. Nada é gravado.

## Testes

```bash
python tests/test_pose_features.py
python tests/test_ssbd_annotations.py
python tests/test_timeline.py
python tests/test_timeline_js.py   # a segmentação em JavaScript tem de dar o mesmo que a em Python (requer Node.js)
python tests/test_camera.py        # o preparo ao vivo da câmera tem de dar o mesmo que o do treino
```

Os dois últimos usam os dados locais (poses, vídeos, `saidas/` e modelos).
