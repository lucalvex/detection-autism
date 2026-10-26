# ============================================================
# ARQUIVO: experimentos/gerar_tabelas_tcc.py
#
# O QUE FAZ: tabelas e figuras do TCC para o modelo final
# (r4b_C1_janela_4s), no formato de src/gerar_tabelas.py. Só lê os
# resultados gravados das rodadas; não treina nem altera nada. Grava em
# tabelas/ e figuras/:
#   tab_qualidade_videos.tex, tab_qualidade_classes.tex, fig_qualidade_f1.pdf
#     (medidas de qualidade_dataset/ com o F1 do modelo final);
#   tab_rodadas.tex, tab_modelo_final.tex, tab_final_classes.tex,
#   fig_dependencia_desempenho.pdf.
# Imprime as correlações de Spearman por classe com o F1 do modelo final.
#
# Uso (da raiz do repositório):
#   .venv/Scripts/python.exe experimentos/gerar_tabelas_tcc.py
# ============================================================

import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

RAIZ = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(RAIZ / "experimentos"), str(RAIZ / "experimentos" / "qualidade_dataset"), str(RAIZ / "src")]
import rodar_q as rq  # noqa: E402  (faz chdir para a raiz)
import qualidade_dataset as qd  # noqa: E402
from gerar_tabelas import mc, nome_classe, num, pm, tabela  # noqa: E402

r, q = rq.r, rq.q
E = RAIZ / "experimentos"
TAB, FIG = RAIZ / "tabelas", RAIZ / "figuras"
FINAL = {42: "r4b_C1_janela_4s", 1000: "r4b_C1_janela_4s_semente_1000", 2000: "r4b_C1_janela_4s_semente_2000"}
M3 = ["f1_evento_iou_0.3", "acuracia_balanceada_presenca", "macro_f1_quadro_todos"]
ler = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))
val = lambda exp: ler(E / exp / "metricas_validacao.json")["agregado_validacao"]


# ---------- 1. qualidade dos vídeos com o F1 do modelo final ----------

def qualidade():
  qd.FINAL = FINAL
  f1 = qd.f1_por_video()
  linhas = []
  for x in csv.DictReader(open(E / "qualidade_dataset" / "tabela_videos.csv", encoding="utf-8")):
    y = {k: (v if k in ("video", "abrev", "classe") else float(v)) for k, v in x.items() if k != "f1"}
    for k in ("largura", "altura", "quadros"):
      y[k] = int(y[k])
    y["f1"] = f1.get(x["video"])
    linhas.append(y)
  with tempfile.TemporaryDirectory() as tmp:
    qd.PASTA = Path(tmp)
    qd.gravar(linhas)
    qd.figura(linhas)
    for nome in ("tab_qualidade_videos.tex", "tab_qualidade_classes.tex"):
      shutil.copy(Path(tmp) / nome, TAB / nome)
    shutil.copy(Path(tmp) / "fig_qualidade_f1.pdf", FIG / "fig_qualidade_f1.pdf")
  medidas = ["fps", "bpp", "nitidez", "movimento", "brusco_pct", "pose_0_0_pct", "pose_1_4_pct", "pose_5_10_pct", "pose_11_17_pct", "q_pct"]
  print("Spearman entre o F1 do modelo final (média de 3 sementes, quadros anotados) e cada medida")
  for c in ["ArmFlapping", "HeadBanging", "Spinning", None]:
    v = [x for x in linhas if (c is None or x["classe"] == c) and x["f1"] is not None]
    print(f"== {c or 'todas as classes'}: n={len(v)}")
    for m in medidas:
      rho, p = spearmanr([x[m] for x in v], [x["f1"] for x in v])
      print(f"  {m:15s} rho={rho:+.2f} p={p:.3f}")


# ---------- 2. rodadas ----------

def linha(rodada, ajuste, met, ref, decisao):
  cel = lambda m: f"{num(met[m])} ({num(ref[m])})"
  return [rodada, ajuste, *[cel(m) for m in M3], decisao]


def rodadas():
  corpo = []
  # rodada 1: cada etapa contra o melhor até então (resumo.md)
  r1 = [("02_yolo11m", "YOLO11m-Pose"), ("03_normalizacao_robusta", "normalização robusta"), ("04_interpolacao_suavizacao", "interpolação e suavização"),
        ("05_15fps_janela_1s", "15 quadros/s, janela de 1 s"), ("06_15fps_janela_2s", "15 quadros/s, janela de 2 s"),
        ("07_sem_sobreposicao", "janelas sem sobreposição"), ("08a_lr_0.0003", "taxa de aprendizado 0,0003"), ("08b_lr_0.003", "taxa de aprendizado 0,003")]
  melhor = val("01_ponto_de_partida")
  for exp, nome in r1:
    m = val(exp)
    ok, _ = r.supera(m, melhor)
    corpo.append(linha("1", nome, m, melhor, "mantido" if ok and not exp.startswith("08") else "não"))
    if ok and not exp.startswith("08"):
      melhor = m
  corpo.append("\\midrule")
  # rodada 2: referência R (06); C acumulativo; B como linha de base
  melhor = val("06_15fps_janela_2s")
  r2 = [("r2_A1_dxdy", "só deslocamentos"), ("r2_A2_dxdy_angulos", "deslocamentos e ângulos"),
        ("r2_B_2s_logreg", "periodicidade, 2 s, regressão logística"), ("r2_B_2s_gb", "periodicidade, 2 s, \\textit{gradient boosting}"),
        ("r2_B_4s_logreg", "periodicidade, 4 s, regressão logística"), ("r2_B_4s_gb", "periodicidade, 4 s, \\textit{gradient boosting}"),
        ("r2_C1_dropout_0.3", "\\textit{dropout} 0,3"), ("r2_C1_dropout_0.5", "\\textit{dropout} 0,5"), ("r2_C2_l2_1e-4", "penalização L2 $10^{-4}$"),
        ("r2_C2_l2_1e-3", "penalização L2 $10^{-3}$"), ("r2_C3_1x64", "1 camada de 64 unidades"), ("r2_C4_aumento", "aumento de dados")]
  for exp, nome in r2:
    m = val(exp)
    ok, _ = r.supera(m, melhor)
    dec = "linha de base" if "_B_" in exp else ("mantido" if ok else "não")
    corpo.append(linha("2", nome, m, melhor, dec))
    if ok and "_B_" not in exp:
      melhor = m
  corpo.append("\\midrule")
  # rodada 3: cobertura total e critério Q, contra o r2_C4_aumento
  cob = ler(E / "r3_cobertura_decisao.json")
  nomes3 = {"r3_I1_interpolacao_limitada": "interpolação limitada a 0,5 s", "r3_I2_limitada_confianca": "interpolação limitada e confiança",
            "r3_I3_treino_a": "descarte por qualidade no treino (a)", "r3_I3_treino_b": "descarte por qualidade no treino (b)"}
  for exp, x in cob["variantes"].items():
    corpo.append(linha("3", nomes3[exp], x["cobertura_total"]["agregado_validacao"], cob["metricas_referencia"], "não"))
  d = ler(E / "r3_q_resultados.json")["decisao"]
  corpo.append(linha("3", "QF: só quadros válidos por Q", d["QF"]["agregado_validacao"], d["r2_C4_aumento"]["agregado_validacao"], "não"))
  corpo.append("\\midrule")
  # rodada 4: contra o r2_C4_aumento nos quadros válidos por Q
  ref4 = ler(E / "r4b_C3_sem_problematicos" / "rodada4b.json")["met_ref"]["agregado_validacao"]
  for exp, nome in [("r4_P1a_trajetoria", "trajetória centralizada"), ("r4_P1b_trajetoria_postura", "trajetória e postura média"),
                    ("r4_P2_periodicidade", "periodicidade após o LSTM"), ("r4_P3_aumento_postura", "aumento de postura e ângulo")]:
    corpo.append(linha("4", nome, ler(E / exp / "rodada4.json")["metricas_validacao_q"]["agregado_validacao"], ref4, "não"))
  for exp, nome in [("r4b_C1_janela_4s", "janela de 4 s"), ("r4b_C2_periodicidade_4s", "periodicidade, janela de 4 s"),
                    ("r4b_C3_sem_problematicos", "treino sem 5 vídeos problemáticos")]:
    x = ler(E / exp / "rodada4b.json")
    corpo.append(linha("4b", nome, x["met"]["agregado_validacao"], x["met_ref"]["agregado_validacao"],
                       "\\textbf{modelo final}" if x["situacao"] == "passa" else "não"))
  x = ler(E / "r4c_P2c_periodicidade_corrigida" / "rodada4c.json")
  corpo.append(linha("4c", "periodicidade corrigida", x["met"]["agregado_validacao"], x["met_ref"]["agregado_validacao"], "não"))
  cab = [["Rodada", "Ajuste", "F1 evento", "Acurácia bal.", "Macro F1 quadros", "Decisão"]]
  tex = tabela("tab:rodadas", "Ajustes testados por rodada, com as métricas de validação", "llrrrl", cab, corpo,
               "Cada célula: valor do ajuste e, entre parênteses, o da referência da etapa (o melhor modelo até então), no agregado da validação. "
               "Mantém-se um ajuste que melhora pelo menos 2 das 3 métricas em 0,01 ou mais. Rodadas 1 e 2: todos os quadros com pessoa; "
               "rodada 3: mesmos quadros do modelo de referência (cobertura total) e, no QF, quadros válidos pelo critério Q; rodadas 4 a 4c: quadros válidos pelo critério Q. "
               "Na rodada 2, a periodicidade com classificador simples foi linha de base e não concorria a modelo final.",
               tamanho="\\scriptsize")
  (TAB / "tab_rodadas.tex").write_text(tex + "\n", encoding="utf-8")


# ---------- 3 e 4. modelo final ----------

def modelo_final():
  a = ler(E / "r4b_avaliacao_final" / "avaliacao_final.json")
  M4 = M3 + ["macro_f1_quadro_anotados"]
  dep = a["dependencia_teste"]
  agg = lambda ts, m: pm(np.mean([t["agregado_56_videos"][m] for t in ts]), np.std([t["agregado_56_videos"][m] for t in ts]))
  pos = lambda ts: pm(np.mean([t["f1_evento_com_pos"]["agregado"] for t in ts]), np.std([t["f1_evento_com_pos"]["agregado"] for t in ts]))
  dp = lambda xs: pm(np.mean([x["dependencia"] for x in xs]), np.std([x["dependencia"] for x in xs]))
  corpo = [["Modelo do TCC (rótulos por intervalo)", *[num(a["TCC"]["agregado_56_videos"][m]) for m in M4], num(a["TCC"]["f1_evento_com_pos"]["agregado"]),
            num(dep["TCC"][0]["dependencia"])],
           ["Final da rodada 2 (janela de 2 s)", *[agg(a["ref"], m) for m in M4], pos(a["ref"]), dp(dep["r2_C4_aumento"])],
           ["\\textbf{Modelo final} (janela de 4 s)", *[agg(a["final"], m) for m in M4], pos(a["final"]), dp(dep["r4b_C1_janela_4s"])]]
  n_quadros = f"{a['n_quadros_teste']:,}".replace(",", ".")
  cab = [["Modelo", "F1 evento", "Acurácia bal.", "Macro F1 quadros", "Macro F1 anotados", "F1 evento com pós", "Dependência"]]
  tex = tabela("tab:modelo-final", "Teste do modelo final nos quadros válidos pelo critério Q", "lrrrrrr", cab, corpo,
               f"Predições fora da partição, {n_quadros} quadros válidos pelo critério Q comuns aos três modelos; "
               "agregado sobre os vídeos; média $\\pm$ desvio entre 3 sementes (o modelo do TCC é uma execução). F1 evento: IoU 0,3, regra principal. "
               "Pós-processamento: mediana de 2 s e junção de trechos separados por menos de 1 s, escolhidos na validação. "
               "Dependência: macro F1 nos quadros anotados com as janelas reais menos com as janelas congeladas (postura média, deslocamentos zerados).",
               tamanho="\\footnotesize")
  (TAB / "tab_modelo_final.tex").write_text(tex + "\n", encoding="utf-8")

  d = a["detalhes"]["final"]
  pt = d["por_classe"]["todos_os_quadros"]["agregado"]
  pa = d["por_classe"]["quadros_anotados"]["agregado"]
  mz = d["confusao_quadros_anotados"]["matriz"]
  classes = ["ArmFlapping", "HeadBanging", "Spinning", "Background"]
  corpo = []
  for i, c in enumerate(classes):
    conf = [f"{v:,}".replace(",", ".") for v in mz[i]] if c != "Background" else ["--"] * 4
    corpo.append([nome_classe(c), num(pt[c]["precision"]), num(pt[c]["recall"]), num(pt[c]["f1-score"]),
                  num(pa[c]["f1-score"]) if c != "Background" else "--", *conf])
  cab = [["", mc(4, "Todos os quadros (F1 anotados)"), mc(4, "Matriz nos quadros anotados: prevista")],
         ["Classe", "Precisão", "Revocação", "F1", "F1 anot.", *[nome_classe(c) for c in classes]]]
  tex = tabela("tab:final-classes", "Desempenho por classe e matriz de confusão do modelo final", "lrrrr|rrrr", cab, corpo,
               "Teste, semente 42, quadros válidos pelo critério Q comuns ao modelo do TCC e ao final da rodada 2. Precisão, revocação e F1 em todos os quadros; "
               "F1 anot.: F1 nos quadros anotados. Matriz: linhas, classe anotada; colunas, classe prevista; número de quadros.", tamanho="\\footnotesize")
  (TAB / "tab_final_classes.tex").write_text(tex + "\n", encoding="utf-8")


# ---------- 5. dependência contra desempenho ----------

def dependencia_desempenho():
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  from matplotlib.ticker import FuncFormatter
  from gerar_figura_linha_tempo import DPI, LARGURA_POL, TEXTO_PT, cores_da_interface, virgula
  _, var = cores_da_interface()
  plt.rcParams.update({"font.size": TEXTO_PT, "axes.labelsize": TEXTO_PT, "xtick.labelsize": TEXTO_PT, "ytick.labelsize": TEXTO_PT,
                       "legend.fontsize": TEXTO_PT, "text.color": var["texto"], "axes.labelcolor": var["texto"],
                       "xtick.color": var["texto-2"], "ytick.color": var["texto-2"], "axes.edgecolor": var["linhas"], "pdf.fonttype": 42})
  pontos = []
  x = ler(E / "r4b_C3_sem_problematicos" / "rodada4b.json")
  pontos.append(("final da rodada 2", x["dep_ref"]["dependencia"], x["met_ref"]["agregado_validacao"]["macro_f1_quadro_todos"], "val"))
  for exp, nome in [("r4_P1a_trajetoria", "P1a"), ("r4_P1b_trajetoria_postura", "P1b"), ("r4_P2_periodicidade", "P2"), ("r4_P3_aumento_postura", "P3")]:
    y = ler(E / exp / "rodada4.json")
    pontos.append((nome, y["dependencia"]["dependencia"], y["metricas_validacao_q"]["agregado_validacao"]["macro_f1_quadro_todos"], "val"))
  for exp, nome, arq in [("r4b_C1_janela_4s", "modelo final (C1)", "rodada4b.json"), ("r4c_P2c_periodicidade_corrigida", "P2c", "rodada4c.json")]:
    y = ler(E / exp / arq)
    pontos.append((nome, y["dep"]["dependencia"], y["met"]["agregado_validacao"]["macro_f1_quadro_todos"], "val"))
  a = ler(E / "r4b_avaliacao_final" / "avaliacao_final.json")
  pontos.append(("modelo do TCC (teste)", a["dependencia_teste"]["TCC"][0]["dependencia"], a["TCC"]["agregado_56_videos"]["macro_f1_quadro_todos"], "teste"))
  fig, ax = plt.subplots(figsize=(LARGURA_POL, 4.4), dpi=DPI)
  for nome, xx, yy, tipo in pontos:
    final = nome.startswith("modelo final")
    ax.scatter([xx], [yy], s=90 if final else 55, zorder=3, color="#C7522A" if final else var["texto"],
               facecolors="none" if tipo == "teste" else None, linewidth=1.5)
    ax.annotate(nome, (xx, yy), textcoords="offset points", xytext=(7, -16) if nome == "P2" else (7, 5), fontsize=TEXTO_PT - 1, color=var["texto"])
  ax.set_xlabel("dependência do movimento (macro F1 anotados: real menos congelada)")
  ax.set_ylabel("macro F1 por quadro (validação)")
  for eixo in (ax.xaxis, ax.yaxis):
    eixo.set_major_formatter(FuncFormatter(virgula))
  ax.grid(color=var["linhas"], linewidth=0.8, zorder=0)
  for lado in ("top", "right"):
    ax.spines[lado].set_visible(False)
  ax.margins(x=0.25, y=0.12)
  fig.tight_layout()
  fig.savefig(FIG / "fig_dependencia_desempenho.pdf")
  fig.savefig(FIG / "fig_dependencia_desempenho.png", dpi=DPI)
  plt.close(fig)


if __name__ == "__main__":
  r.OPC.update({"saida": str(E), "teste": False})
  qualidade()
  rodadas()
  modelo_final()
  dependencia_desempenho()
