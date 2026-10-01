# ============================================================
# ARQUIVO: exportar_sessao.py
#
# O QUE FAZ: gera <id>_sessao.json (formato de tcc-demo-handoff/HANDOFF.md,
# com os campos novos combinados depois do handoff -- ver conversa)
# para um vídeo do SSBD, usando o checkpoint da Rodada A
# (models/oof_seeded_annotated/) do fold em que esse vídeo é TESTE,
# conforme models/oof/fold_assignments.json (o mesmo particionamento
# copiado em models/oof_seeded_annotated/fold_assignments.json --
# conferido que os dois batem antes de escrever este script). NÃO
# TREINA NADA: só carrega os checkpoints já existentes.
#
# Reaproveita o pré-processamento do treino em vez de reescrever --
# reusa diretamente evaluate_temporal.compute_windows() (mesmo CSV de
# pose já extraído em data/poses/*.csv, mesma regra de quadro real via
# a coluna "frame", mesmo fatiamento de 30 quadros por posição de
# linha, mesmo checkpoint por fold). Qualquer mudança nessa lógica deve
# ser feita lá, não duplicada aqui.
#
# Uso:
#   python src/exportar_sessao.py <video_id> --saida <pasta>
#   ex.: python src/exportar_sessao.py v_ArmFlapping_08 --saida saidas/
# ============================================================

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import evaluate_temporal as et
from ssbd_annotations import find_annotation_file, load_annotations

SEQUENCE_LENGTH = 30
SPAN_LIMIT_QUADROS = 60  # janela "longa" -- mais que o dobro da duração nominal (ver conversa)

DATASET_DIR = "data/datasets_annotated"
OOF_DIR = "models/oof_seeded_annotated"
VIDEOS_DIR = Path("data/videos")
XMLS_DIR = Path("../download-yt/xmls")

CATEGORY_TO_LABEL = {
  "armflapping": "ArmFlapping",
  "headbanging": "HeadBanging",
  "spinning": "Spinning",
}

NOMES_CLASSE = {
  "ArmFlapping": "Balançar os braços",
  "HeadBanging": "Bater a cabeça",
  "Spinning": "Girar o corpo",
  "Background": "Sem estereotipia",
}


def video_metadata(video_id):
  """fps/total de quadros/dimensões reais, lidos do arquivo de vídeo
  (não do <frames> do XML, que em alguns vídeos diverge da contagem
  real -- ver diagnóstico anterior)."""

  cap = cv2.VideoCapture(str(VIDEOS_DIR / f"{video_id}.mp4"))

  fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
  total_quadros = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

  cap.release()

  return fps, total_quadros, largura, altura


def quadros_sem_deteccao(video_id, total_quadros):
  """Índices reais de quadro (0-based) sem nenhuma pessoa detectada,
  em todo o vídeo (não só no trecho usado para montar janelas) --
  para marcar em cinza a linha do tempo inteira."""

  import pandas as pd

  df = pd.read_csv(et.POSES_DIR / f"{video_id}.csv")
  detectados = set(int(x) for x in df["frame"].values)

  return [i for i in range(total_quadros) if i not in detectados]


def anotacoes_xml(video_id, fps):
  """Intervalos anotados do XML do SSBD, na MESMA lógica de
  ground_truth_frames() de evaluate_temporal.py (mesmo cálculo de
  segundos a partir de <time>)."""

  xml_path = find_annotation_file(video_id, XMLS_DIR)

  if xml_path is None:
    return []

  ann = load_annotations(xml_path)

  return [
    {"classe": b["category"], "inicio_s": b["start_s"], "fim_s": b["end_s"]}
    for b in ann["behaviours"]
    if b["category"] in CATEGORY_TO_LABEL
  ]


def exportar(video_id, saida_dir):

  et.load_config(DATASET_DIR, OOF_DIR)

  if video_id not in et.fold_assignments:
    raise ValueError(
      f"{video_id} não está em {OOF_DIR}/fold_assignments.json -- "
      "só vídeos do SSBD com fold conhecido são suportados por enquanto."
    )

  fold = et.fold_assignments[video_id]
  checkpoint_path = f"{OOF_DIR}/fold_{fold}.pt"

  windows_raw = et.compute_windows(video_id)

  if not windows_raw:
    raise RuntimeError(f"{video_id}: sem janelas suficientes (menos de {SEQUENCE_LENGTH} quadros detectados).")

  fps, total_quadros, largura, altura = video_metadata(video_id)

  janelas = []
  for w in windows_raw:
    span = w["quadro_fim"] - w["quadro_inicio"]
    janelas.append({
      "quadro_inicio": w["quadro_inicio"],
      "quadro_fim": w["quadro_fim"],
      "pontuacoes": w["pontuacoes"],
      "janela_longa": span > SPAN_LIMIT_QUADROS,
    })

  duracao_minima_s = round(SEQUENCE_LENGTH / fps, 4)

  sessao = {
    "versao": 1,
    "sintetico": False,
    "sessao": {
      "id": video_id,
      "video": f"{video_id}.mp4",
      "fps": fps,
      "total_quadros": total_quadros,
      "largura": largura,
      "altura": altura,
    },
    "modelo": {
      "pose": "yolo11n-pose",
      "formato_pontos": "coco17",
      "classificador": "lstm",
      "lstm_camadas": 2,
      "lstm_unidades": 128,
      "n_caracteristicas": 68,
      "ativacao_saida": "softmax",
      "n_saidas": et.num_classes,
      "janela_quadros": SEQUENCE_LENGTH,
      "passo_quadros": 1,
      "particao": fold,
      "checkpoint": checkpoint_path,
      "predicao_fora_da_particao": True,
    },
    "classes": [
      {"id": name.lower(), "nome": NOMES_CLASSE[name]}
      for name in et.class_names
    ],
    "limiar_padrao": 0.5,
    "duracao_minima_s": duracao_minima_s,
    "limite_janela_longa_quadros": SPAN_LIMIT_QUADROS,
    "janelas": janelas,
    "quadros_sem_deteccao": quadros_sem_deteccao(video_id, total_quadros),
    "anotacoes": anotacoes_xml(video_id, fps),
  }

  saida_dir = Path(saida_dir)
  saida_dir.mkdir(parents=True, exist_ok=True)

  saida_path = saida_dir / f"{video_id}_sessao.json"

  with open(saida_path, "w", encoding="utf-8") as f:
    json.dump(sessao, f, indent=2, ensure_ascii=False)

  n_longas = sum(1 for j in janelas if j["janela_longa"])

  print(
    f"{video_id}: fold={fold}, {len(janelas)} janelas ({n_longas} longas), "
    f"{len(sessao['quadros_sem_deteccao'])} quadros sem detecção de {total_quadros}, "
    f"{len(sessao['anotacoes'])} intervalos anotados -> {saida_path}"
  )

  return saida_path


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("video_id", help="ex.: v_ArmFlapping_08 (sem extensão)")
  parser.add_argument("--saida", default="saidas", help="pasta de saída (default: saidas/)")

  args = parser.parse_args()

  exportar(args.video_id, args.saida)
