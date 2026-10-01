# ============================================================
# ARQUIVO: resumo_dados.py
#
# O QUE FAZ: conta, a partir dos próprios dados, os números de
# contexto que a tela de apresentação mostra (quantas anotações XML do
# SSBD existem, quantos vídeos entraram na avaliação, quantas
# partições) e grava em data/results/resumo_dados.json -- para que
# nenhum desses números seja escrito à mão no HTML.
#
# Uso: python src/resumo_dados.py
# ============================================================

import json
from pathlib import Path

from ssbd_annotations import load_annotations

XMLS_DIR = Path("../download-yt/xmls")
FOLD_ASSIGNMENTS = Path("models/oof/fold_assignments.json")
SAIDA = Path("data/results/resumo_dados.json")


def main():

  xmls = sorted(XMLS_DIR.glob("*.xml"))
  categorias = {}
  n_comportamentos = 0
  for x in xmls:
    for b in load_annotations(x)["behaviours"]:
      categorias[b["category"]] = categorias.get(b["category"], 0) + 1
      n_comportamentos += 1

  folds = json.load(open(FOLD_ASSIGNMENTS, encoding="utf-8"))
  video_to_fold = folds["video_to_fold"]

  saida = {
    "n_anotacoes_xml": len(xmls),
    "n_comportamentos_anotados": n_comportamentos,
    "comportamentos_por_categoria": categorias,
    "n_videos_usados": len(video_to_fold),
    "n_particoes": folds["n_splits"],
    "divisao": "por vídeo (StratifiedGroupKFold, random_state=%d); cada vídeo é teste em exatamente uma partição" % folds["random_state"],
    "fontes": {"anotacoes": str(XMLS_DIR), "particoes": str(FOLD_ASSIGNMENTS)},
  }

  with open(SAIDA, "w", encoding="utf-8") as f:
    json.dump(saida, f, indent=2, ensure_ascii=False)

  print(json.dumps(saida, indent=2, ensure_ascii=False))


if __name__ == "__main__":
  main()
