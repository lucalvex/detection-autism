# ============================================================
# ARQUIVO: experimentos/avaliar_versao2.py
#
# O QUE FAZ: avalia, sem treinar nada, o modelo final da rodada 1
# (06-15fps-janela-2s, a "Versão 2" do TCC: normalização robusta,
# 15 quadros/s, janela de 2 s, parada antecipada, sem aumento de dados)
# nas 3 sementes (42, 1000 e 2000), no mesmo conjunto de quadros e com o
# mesmo pós-processamento da avaliação final da rodada 4b
# (r4b-avaliacao-final/avaliacao-final.json, a tab_modelo_final):
#   - quadros válidos por Q comuns ao modelo do TCC, ao final da rodada 2
#     e ao modelo final (semente 42); confere que são os 90.722 de lá;
#   - as seis medidas: F1 por evento, acurácia balanceada, macro F1 por
#     quadro, macro F1 nos quadros anotados, F1 por evento com o
#     pós-processamento escolhido e dependência do movimento (janelas
#     congeladas, com os modelos salvos das partições).
# Antes, refaz as medidas do modelo do TCC e confere que batem com as
# gravadas, para garantir que o ambiente é o mesmo.
# Grava r4b-avaliacao-final/versao2.json; não altera nenhum resultado.
#
# Uso (da raiz do repositório):
#   .venv/Scripts/python.exe experimentos/avaliar_versao2.py
# ============================================================

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(RAIZ / "experimentos"), str(RAIZ / "src")]
import rodar_rodada4b  # noqa: E402,F401  (o mesmo ambiente da avaliação final da rodada 4b)
import rodar_rodada4 as r4  # noqa: E402

rq, q, r = r4.rq, r4.q, r4.rq.r
EXP1 = rq.EXP1
FINAL = "r4b-C1-janela-4s"
SEMENTES = {42: "06-15fps-janela-2s", 1000: "09-final-semente-1000", 2000: "09-final-semente-2000"}
SAIDA = EXP1 / "r4b-avaliacao-final" / "versao2.json"


def cfg_de(d):
  return {**q.NOVOS, **rq.cfg_de(EXP1 / d)}


def main():
  a = r.ler_json(EXP1 / "r4b-avaliacao-final" / "avaliacao-final.json")
  g = a["grade_pos"][a["pos"]]
  tcc = rq.tcc_vd()
  comum = rq.conjunto_comum(tcc, rq.dados(EXP1 / rq.REF, "teste", cfg_de(rq.REF)), rq.dados(EXP1 / FINAL, "teste", cfg_de(FINAL)))
  n = rq.n_quadros(rq.restringir(tcc, comum))
  assert n == a["n_quadros_teste"], (n, a["n_quadros_teste"])
  t = rq.metricas_teste_restritas(rq.restringir(tcc, comum), g)
  for m, v in a["TCC"]["agregado_56_videos"].items():
    assert abs(t["agregado_56_videos"][m] - v) < 1e-12, m
  assert abs(t["f1_evento_com_pos"]["agregado"] - a["TCC"]["f1_evento_com_pos"]["agregado"]) < 1e-12

  out = {"modelo": SEMENTES[42], "papel": "Versão 2 do TCC (final da rodada 1)", "pos": a["pos"], "n_quadros_teste": n,
         "sementes": list(SEMENTES), "teste": [], "dependencia_teste": []}
  for s, d in SEMENTES.items():
    cfg = cfg_de(d)
    assert cfg["semente"] == s and cfg["janela"] == 30 and cfg["fps_alvo"] == 15 and not cfg["aumento"], (d, cfg)
    vd = rq.dados(EXP1 / d, "teste", cfg)
    vr = rq.restringir(vd, comum)
    assert rq.n_quadros(vr) == n, (d, rq.n_quadros(vr))  # cobre todos os quadros comuns
    out["teste"].append(rq.metricas_teste_restritas(vr, g))
    out["dependencia_teste"].append(r4.dependencia(vd, r4.vd_congelado(EXP1 / d, cfg, "teste"), comum))
    print(f"{d}: {out['teste'][-1]['agregado_56_videos']}, pós {out['teste'][-1]['f1_evento_com_pos']['agregado']:.3f}, "
          f"dependência {out['dependencia_teste'][-1]['dependencia']:.3f}", flush=True)
  r.gravar_json(SAIDA, out)
  print(f"gravado {SAIDA.relative_to(RAIZ)}")


if __name__ == "__main__":
  main()
