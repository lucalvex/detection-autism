# ============================================================
# ARQUIVO: comparar_modelos.py
#
# O QUE FAZ: aplica o critério de data/results/criterio_comparacao.md
# (registrado antes da avaliação) aos metricas.json da Rodada A e do
# base re-seedado, e grava o veredito em
# data/results/comparacao_final.json. Não recalcula nenhuma métrica:
# só lê os dois metricas.json já gravados.
#
# Critério:
#   - métrica 1: macro F1 por quadro restrito aos quadros anotados
#     (por_quadro.somente_quadros_anotados, macro_f1_sem_background);
#   - métrica 2: F1 por evento IoU 0,3 na regra principal
#     (por_evento.regra_principal.por_iou["0.3"], macro_f1);
#   - ambas agregadas sobre os 56 vídeos (pooled_56_videos);
#   - Rodada A "melhor" só se melhorar as duas métricas agregadas E a
#     melhora ocorrer em pelo menos 4 dos 5 folds (em cada métrica);
#     caso contrário, "sem diferença distinguível".
#
# Uso: python src/comparar_modelos.py
# ============================================================

import json
from pathlib import Path

RODADA_A = Path("data/results/comparacao_rodada_a/metricas.json")
BASE = Path("data/results/comparacao_base_reseeded/metricas.json")
SAIDA = Path("data/results/comparacao_final.json")

MIN_FOLDS = 4
VEREDITO_MELHOR = "Rodada A melhor"
VEREDITO_EMPATE = "sem diferença distinguível"


def metrica_1(m):
  bloco = m["por_quadro"]["somente_quadros_anotados"]
  return (
    bloco["pooled_56_videos"]["macro_f1_sem_background"],
    {int(k): v["macro_f1_sem_background"] for k, v in bloco["por_fold"].items()},
  )


def metrica_2(m):
  bloco = m["por_evento"]["regra_principal"]["por_iou"]["0.3"]
  return (
    bloco["pooled_56_videos"]["macro_f1"],
    {int(k): v["macro_f1"] for k, v in bloco["por_fold"].items()},
  )


def comparar(nome, fn, a, b):
  pooled_a, folds_a = fn(a)
  pooled_b, folds_b = fn(b)
  por_fold = [
    {"fold": k, "rodada_a": folds_a[k], "base_re_seedado": folds_b[k], "rodada_a_maior": folds_a[k] > folds_b[k]}
    for k in sorted(folds_a)
  ]
  n_folds_a_maior = sum(1 for r in por_fold if r["rodada_a_maior"])
  return {
    "metrica": nome,
    "pooled_56_videos": {"rodada_a": pooled_a, "base_re_seedado": pooled_b},
    "rodada_a_melhora_agregado": pooled_a > pooled_b,
    "por_fold": por_fold,
    "n_folds_rodada_a_maior": n_folds_a_maior,
    "n_folds": len(por_fold),
    "rodada_a_melhora_em_folds_suficientes": n_folds_a_maior >= MIN_FOLDS,
  }


def main():

  a = json.load(open(RODADA_A, encoding="utf-8"))
  b = json.load(open(BASE, encoding="utf-8"))

  m1 = comparar("macro F1 por quadro, restrito aos quadros anotados", metrica_1, a, b)
  m2 = comparar("F1 por evento, IoU 0,3, regra principal", metrica_2, a, b)

  condicao_1 = m1["rodada_a_melhora_agregado"] and m2["rodada_a_melhora_agregado"]
  condicao_2 = m1["rodada_a_melhora_em_folds_suficientes"] and m2["rodada_a_melhora_em_folds_suficientes"]
  veredito = VEREDITO_MELHOR if (condicao_1 and condicao_2) else VEREDITO_EMPATE

  saida = {
    "criterio": "data/results/criterio_comparacao.md",
    "modelos": {"rodada_a": str(RODADA_A), "base_re_seedado": str(BASE)},
    "metricas_principais": [m1, m2],
    "condicao_1_melhora_as_duas_metricas_agregadas": condicao_1,
    "condicao_2_melhora_em_pelo_menos_4_de_5_folds": condicao_2,
    "veredito": veredito,
  }

  SAIDA.parent.mkdir(parents=True, exist_ok=True)
  with open(SAIDA, "w", encoding="utf-8") as f:
    json.dump(saida, f, indent=2, ensure_ascii=False)

  for m in (m1, m2):
    p = m["pooled_56_videos"]
    print(
      f"{m['metrica']}: Rodada A={p['rodada_a']:.4f} | base={p['base_re_seedado']:.4f} | "
      f"Rodada A maior em {m['n_folds_rodada_a_maior']}/{m['n_folds']} folds"
    )
  print(f"Condição 1 (melhora as duas agregadas): {condicao_1}")
  print(f"Condição 2 (>= {MIN_FOLDS}/5 folds nas duas): {condicao_2}")
  print(f"Veredito: {veredito}")
  print(f"Gravado em {SAIDA}")


if __name__ == "__main__":
  main()
