# ============================================================
# ARQUIVO: selecionar_videos_mediana.py
#
# O QUE FAZ: escolhe os vídeos do modo apresentação da demo -- um por
# classe (ArmFlapping, HeadBanging, Spinning), o de F1 por quadro
# MEDIANO entre os vídeos daquela classe -- e grava a regra e a escolha
# em data/results/videos_desempenho_mediano.json. Não treina nada.
#
# REGRA:
#   - modelo: Rodada A (models/oof_seeded_annotated/), cada vídeo
#     avaliado pelo checkpoint do fold em que ele é teste;
#   - predição por quadro: argmax das 4 saídas (o que a interface
#     mostra), via evaluate_temporal.compute_windows/predicted_frames;
#   - quadros considerados: só os anotados no XML (gabarito != fundo)
#     e com predição;
#   - F1 do vídeo: macro F1 sobre as classes presentes no gabarito
#     anotado DAQUELE vídeo (uma predição de fundo ou de outra classe
#     conta como erro da classe anotada);
#   - classe do vídeo: a do nome do arquivo;
#   - mediana: vídeos da classe ordenados por F1 (empate: video_id);
#     escolhe o índice (n-1)//2 -- a mediana inferior quando n é par.
#
# Uso: python src/selecionar_videos_mediana.py
# ============================================================

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

import evaluate_temporal as et

DATASET_DIR = "data/datasets_annotated"
OOF_DIR = "models/oof_seeded_annotated"
SAIDA = Path("data/results/videos_desempenho_mediano.json")

CLASSES = et.REAL_CLASSES


def classe_do_nome(video_id):
  for c in CLASSES:
    if c.lower() in video_id.lower():
      return c
  return None


def f1_do_video(d):
  pred = et.predicted_frames(d["windows"])
  y_true, y_pred = [], []
  for frame_idx, p in pred.items():
    t = d["gt_frames"][frame_idx]
    if t == "Background":
      continue
    y_true.append(t)
    y_pred.append(p)
  if not y_true:
    return None, 0, []
  presentes = sorted(set(y_true))
  f1 = f1_score(y_true, y_pred, labels=presentes, average="macro", zero_division=0)
  return float(f1), len(y_true), presentes


JUSTIFICATIVA_METRICA = (
  "Escolhida por coerência com o protocolo, em que a referência é sempre a anotação e não o "
  "nome do arquivo; a ambiguidade da regra original foi resolvida depois de conhecer o efeito "
  "de cada leitura, que foi registrado ao lado."
)


def f1_so_classe_do_nome(d, classe):
  pred = et.predicted_frames(d["windows"])
  y_true, y_pred = [], []
  for frame_idx, p in pred.items():
    t = d["gt_frames"][frame_idx]
    if t == "Background":
      continue
    y_true.append(t)
    y_pred.append(p)
  return float(f1_score(y_true, y_pred, labels=[classe], average="macro", zero_division=0))


def mediano(por_video, classe, chave):
  vids = [r for r in por_video if r["classe_do_video"] == classe and r[chave] is not None]
  vids.sort(key=lambda r: (r[chave], r["video_id"]))
  idx = (len(vids) - 1) // 2
  return vids, idx


def main():

  et.load_config(DATASET_DIR, OOF_DIR)
  video_data = et.load_all_videos()

  por_video = []
  for vid in sorted(video_data):
    f1, n, presentes = f1_do_video(video_data[vid])
    por_video.append({
      "video_id": vid,
      "classe_do_video": classe_do_nome(vid),
      "particao": video_data[vid]["fold"],
      "f1_quadro_anotados": f1,
      "n_quadros_anotados_com_predicao": n,
      "classes_no_gabarito": presentes,
    })

  for r in por_video:
    r["f1_so_classe_do_nome"] = f1_so_classe_do_nome(video_data[r["video_id"]], r["classe_do_video"])

  escolhidos = {}
  ranking = {}
  empates = {}
  for c in CLASSES:
    vids, idx = mediano(por_video, c, "f1_quadro_anotados")
    escolhidos[c] = dict(vids[idx], posicao_na_ordem=idx, n_videos_da_classe=len(vids))
    ranking[c] = [{"posicao": i, "video_id": r["video_id"], "f1": r["f1_quadro_anotados"]} for i, r in enumerate(vids)]
    f1_mediano = vids[idx]["f1_quadro_anotados"]
    empates[c] = [r["video_id"] for r in vids if r["f1_quadro_anotados"] == f1_mediano]

  multiclasse = [r for r in por_video if len(r["classes_no_gabarito"]) > 1]
  efeito_por_classe = {}
  for c in CLASSES:
    vids_alt, idx_alt = mediano(por_video, c, "f1_so_classe_do_nome")
    efeito_por_classe[c] = {
      "mediano_macro_classes_presentes": escolhidos[c]["video_id"],
      "mediano_so_classe_do_nome": vids_alt[idx_alt]["video_id"],
    }

  comparacao_leituras = {
    "descricao": (
      "Vídeos com mais de uma classe no gabarito anotado: única situação em que as duas "
      "leituras da regra diferem. Leitura escolhida: macro F1 sobre as classes presentes. "
      "Leitura alternativa: F1 só da classe do nome do arquivo."
    ),
    "videos": [
      {
        "video_id": r["video_id"],
        "classe_do_video": r["classe_do_video"],
        "classes_no_gabarito": r["classes_no_gabarito"],
        "f1_macro_classes_presentes": r["f1_quadro_anotados"],
        "f1_so_classe_do_nome": r["f1_so_classe_do_nome"],
      }
      for r in multiclasse
    ],
    "efeito_na_escolha": efeito_por_classe,
  }

  saida = {
    "titulo": "vídeos de desempenho mediano",
    "regra": {
      "um_video_por_classe": CLASSES,
      "classe_do_video": "a do nome do arquivo",
      "modelo": "Rodada A (" + OOF_DIR + "), checkpoint da partição em que o vídeo é teste (predição fora da partição)",
      "predicao": "natural: argmax das 4 saídas, incluindo fundo (uma predição de fundo num quadro anotado conta como erro)",
      "quadros": "só os anotados no XML do SSBD que têm predição",
      "metrica": "F1 por quadro do vídeo: macro F1 sobre as classes presentes no gabarito anotado daquele vídeo",
      "justificativa_da_metrica": JUSTIFICATIVA_METRICA,
      "ordem": "F1 crescente",
      "posicao_mediana": "(n-1)//2 -- a mediana inferior quando n é par",
      "desempate": "ordem alfabética do video_id (aprovado)",
    },
    "modelo": OOF_DIR,
    "escolhidos": escolhidos,
    "empatados_com_o_escolhido": empates,
    "comparacao_das_leituras_da_regra": comparacao_leituras,
    "ranking_por_classe": ranking,
    "por_video": por_video,
  }

  SAIDA.parent.mkdir(parents=True, exist_ok=True)
  with open(SAIDA, "w", encoding="utf-8") as f:
    json.dump(saida, f, indent=2, ensure_ascii=False)

  for c in CLASSES:
    e = escolhidos[c]
    print(f"{c}: {e['video_id']} (F1={e['f1_quadro_anotados']:.4f}, posição {e['posicao_na_ordem']} de {e['n_videos_da_classe']}, partição {e['particao']})")
  print(f"Gravado em {SAIDA}")


if __name__ == "__main__":
  main()
