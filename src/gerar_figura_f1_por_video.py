# ============================================================
# ARQUIVO: gerar_figura_f1_por_video.py
#
# O QUE FAZ: gera figuras/fig_f1_por_video.pdf com a distribuição do F1
# por quadro de cada vídeo (quadros anotados, predição natural, macro F1
# sobre as classes do gabarito do vídeo) para os dois modelos da
# comparação, uma faixa por classe e por modelo, com a mediana marcada.
# Lê data/results/distribuicoes_por_janela_e_video.json; nenhum número
# é escrito à mão. Cores da interface, nomes das classes do labels.json
# (mesmas funções de gerar_figura_linha_tempo.py).
#
# Uso: python src/gerar_figura_f1_por_video.py [--saida figuras/fig_f1_por_video.pdf]
# ============================================================

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from gerar_figura_linha_tempo import DPI, LARGURA_POL, TEXTO_PT, cores_da_interface, nomes_legiveis, virgula  # noqa: E402

DADOS = Path("data/results/distribuicoes_por_janela_e_video.json")
MEDIANOS = Path("data/results/videos_desempenho_mediano.json")
# (chave no JSON, rótulo da faixa, rótulo da legenda, marcador cheio)
MODELOS = (
  ("base_sementes_fixas", "base", "base (sementes fixas)", False),
  ("rotulos_por_intervalo", "intervalos", "intervalos (rótulos por intervalo)", True),
)
ESPALHAMENTO = 0.12  # deslocamento vertical entre vídeos com o mesmo F1 (só apresentação)
MEIA_FAIXA = 0.36


def deslocamentos(valores):
  """Empilha na vertical os vídeos com o mesmo F1 (arredondado a 2 casas),
  centrados na faixa, para nenhum ponto esconder outro."""
  grupos = {}
  for i, v in enumerate(valores):
    grupos.setdefault(round(v, 2), []).append(i)
  dy = [0.0] * len(valores)
  for indices in grupos.values():
    n = len(indices)
    passo = min(ESPALHAMENTO, 2 * MEIA_FAIXA / max(1, n - 1))
    for k, i in enumerate(indices):
      dy[i] = (k - (n - 1) / 2) * passo
  return dy


def gerar(saida):
  dados = json.load(open(DADOS, encoding="utf-8"))
  classes = json.load(open(MEDIANOS, encoding="utf-8"))["regra"]["um_video_por_classe"]
  cores, var = cores_da_interface()
  nomes = nomes_legiveis()

  plt.rcParams.update({
    "font.size": TEXTO_PT, "axes.labelsize": TEXTO_PT, "xtick.labelsize": TEXTO_PT, "ytick.labelsize": TEXTO_PT,
    "legend.fontsize": TEXTO_PT, "text.color": var["texto"], "axes.labelcolor": var["texto"],
    "xtick.color": var["texto-2"], "ytick.color": var["texto-2"], "axes.edgecolor": var["linhas"], "pdf.fonttype": 42,
  })

  fig, ax = plt.subplots(figsize=(LARGURA_POL, 0.75 * len(classes) * len(MODELOS) + 1.4), dpi=DPI)
  fig.subplots_adjust(left=0.30, right=0.97, top=0.86, bottom=0.17)

  y = 0
  rotulos_y, posicoes_y, centros_classe = [], [], []
  for c in classes:
    cor = cores[c.lower()]
    inicio = y
    for chave, rotulo, _legenda, cheio in MODELOS:
      f1s = [r["f1_quadro_anotados"] for r in dados["modelos"][chave]["f1_por_video"]
             if r["classe_do_video"] == c and r["f1_quadro_anotados"] is not None]
      dy = deslocamentos(f1s)
      ax.scatter(f1s, [y + d for d in dy], s=46, facecolors=cor if cheio else "none", edgecolors=cor, linewidths=1.6, zorder=3)
      med = statistics.median(f1s)
      ax.plot([med, med], [y - MEIA_FAIXA, y + MEIA_FAIXA], color=var["texto"], linewidth=2.2, zorder=4)
      rotulos_y.append(rotulo)
      posicoes_y.append(y)
      y += 1
    centros_classe.append(((inicio + y - 1) / 2, c))
    y += 0.6

  ax.set_yticks(posicoes_y)
  ax.set_yticklabels(rotulos_y)
  ax.set_ylim(y - 0.6 + 0.2 - 0.5, -0.7)
  for yc, c in centros_classe:
    ax.text(-0.20, yc, nomes[c.lower()], transform=ax.get_yaxis_transform(), ha="right", va="center", style="italic")
  ax.set_xlim(-0.03, 1.03)
  ax.xaxis.set_major_formatter(FuncFormatter(virgula))
  ax.set_xlabel("F1 por quadro do vídeo, nos quadros anotados")
  ax.grid(axis="x", color=var["linhas"], linewidth=0.8)
  ax.set_axisbelow(True)
  for lado in ("top", "right", "left"):
    ax.spines[lado].set_visible(False)
  ax.tick_params(axis="y", length=0)

  neutro = var["texto-2"]
  legenda = [
    Line2D([], [], marker="o", linestyle="", markerfacecolor="none", markeredgecolor=neutro, markeredgewidth=1.6, markersize=8, label=MODELOS[0][2]),
    Line2D([], [], marker="o", linestyle="", markerfacecolor=neutro, markeredgecolor=neutro, markersize=8, label=MODELOS[1][2]),
    Line2D([], [], color=var["texto"], linewidth=2.2, label="mediana"),
  ]
  ax.legend(handles=legenda, loc="lower left", bbox_to_anchor=(-0.40, 1.02), ncol=3, frameon=False, handletextpad=0.4, columnspacing=1.0)

  saida = Path(saida)
  saida.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(saida)
  fig.savefig(saida.with_suffix(".png"), dpi=DPI)
  plt.close(fig)
  print(f"figura gravada em {saida} (e {saida.with_suffix('.png').name})")


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--saida", default="figuras/fig_f1_por_video.pdf")
  gerar(ap.parse_args().saida)
