# ============================================================
# ARQUIVO: gerar_figura_linha_tempo.py
#
# O QUE FAZ: gera a figura da linha do tempo de uma sessão para o
# texto (PDF e PNG), a partir só do <id>_sessao.json -- sem o vídeo e
# sem nenhuma imagem de quadro. De cima para baixo, com o mesmo eixo de
# tempo em segundos:
#   1. os intervalos anotados no XML;
#   2. os quadros sem detecção, em cinza;
#   3. para cada estereotipia, a pontuação por janela no centro real da
#      janela, a linha tracejada do limiar padrão e os trechos da regra
#      principal (src/timeline.py) como barras sob a curva.
#
# Nenhum número de dado é escrito aqui: tempos, limiar, duração mínima,
# pontuações e trechos vêm do JSON; as cores vêm da interface
# (demo/js/comum.js e demo/css/estilo.css); os nomes das classes vêm
# de data/datasets_annotated/labels.json.
#
# Uso:
#   python src/gerar_figura_linha_tempo.py v_HeadBanging_24 --saida figuras/fig_linha_do_tempo
# ============================================================

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from timeline import build_segments  # noqa: E402

CORES_JS = Path("demo/js/comum.js")
ESTILO_CSS = Path("demo/css/estilo.css")
LABELS = Path("data/datasets_annotated/labels.json")

# apresentação: ~820 px de largura no PNG e texto de 12 pt = 16,7 px a 100 dpi
DPI = 100
LARGURA_POL = 8.2
TEXTO_PT = 12


def cores_da_interface():
  js = CORES_JS.read_text(encoding="utf-8")
  bloco = re.search(r"export const CORES = \{(.*?)\};", js, re.S).group(1)
  cores = dict(re.findall(r"(\w+):\s*\"(#[0-9A-Fa-f]{6})\"", bloco))
  css = ESTILO_CSS.read_text(encoding="utf-8")
  variaveis = dict(re.findall(r"--([\w-]+):\s*(#[0-9A-Fa-f]{6})", css))
  return cores, variaveis


def nomes_legiveis():
  """id da classe (minúsculo, como no _sessao.json) -> nome em minúsculas
  com espaço, a partir dos nomes CamelCase do labels.json:
  ArmFlapping -> arm flapping."""
  labels = json.load(open(LABELS, encoding="utf-8"))
  return {nome.lower(): re.sub(r"(?<!^)(?=[A-Z])", " ", nome).lower() for nome in labels}


def sequencias(quadros):
  out = []
  for q in quadros:
    if out and q == out[-1][1] + 1:
      out[-1][1] = q
    else:
      out.append([q, q])
  return out


def virgula(x, _pos=None):
  texto = f"{x:g}"
  return texto.replace(".", ",")


def gerar(video_id, saida):
  sessao = json.load(open(f"saidas/{video_id}_sessao.json", encoding="utf-8"))
  fps = sessao["sessao"]["fps"]
  duracao = sessao["sessao"]["total_quadros"] / fps
  limiar = sessao["limiar_padrao"]
  janelas = sessao["janelas"]
  ids = [c["id"] for c in sessao["classes"]]
  estereotipias = [c for c in ids if c != "background"]

  cores, var = cores_da_interface()
  nomes = nomes_legiveis()
  cor_texto, cor_sec, cor_cinza, cor_linhas = var["texto"], var["texto-2"], var["sem-deteccao"], var["linhas"]

  trechos = build_segments(janelas, ids, limiar, sessao["duracao_minima_s"], fps)
  centros = [(j["quadro_inicio"] + j["quadro_fim"]) / 2 / fps for j in janelas]

  plt.rcParams.update({
    "font.size": TEXTO_PT, "axes.labelsize": TEXTO_PT, "xtick.labelsize": TEXTO_PT, "ytick.labelsize": TEXTO_PT,
    "text.color": cor_texto, "axes.labelcolor": cor_texto, "xtick.color": cor_sec, "ytick.color": cor_sec,
    "axes.edgecolor": cor_linhas, "pdf.fonttype": 42,
  })

  alturas = [0.7, 0.7] + [1.6] * len(estereotipias)
  fig, eixos = plt.subplots(
    len(alturas), 1, sharex=True, figsize=(LARGURA_POL, 0.95 * sum(alturas) + 0.9), dpi=DPI,
    gridspec_kw={"height_ratios": alturas, "hspace": 0.35},
  )
  fig.subplots_adjust(left=0.22, right=0.98, top=0.98, bottom=0.12)

  def rotulo(ax, texto, italico=False):
    ax.set_ylabel(texto, rotation=0, ha="right", va="center", labelpad=10, style="italic" if italico else "normal")

  def faixa_vazia(ax):
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    for lado in ("top", "right", "left"):
      ax.spines[lado].set_visible(False)

  # 1. intervalos anotados no XML
  ax = eixos[0]
  faixa_vazia(ax)
  rotulo(ax, "anotação (XML)")
  for a in sessao["anotacoes"]:
    ax.barh(0.5, a["fim_s"] - a["inicio_s"], left=a["inicio_s"], height=0.7, color=cores[a["classe"]])
    ax.text((a["inicio_s"] + a["fim_s"]) / 2, 0.5, nomes[a["classe"]], ha="center", va="center", style="italic", color=cor_texto)

  # 2. quadros sem detecção
  ax = eixos[1]
  faixa_vazia(ax)
  rotulo(ax, "sem detecção")
  for a, b in sequencias(sessao["quadros_sem_deteccao"]):
    ax.barh(0.5, (b + 1 - a) / fps, left=a / fps, height=0.7, color=cor_cinza, linewidth=0)

  # 3. uma faixa por estereotipia
  base = -0.18
  for ax, classe in zip(eixos[2:], estereotipias):
    i = ids.index(classe)
    cor = cores[classe]
    pontos = [j["pontuacoes"][i] for j in janelas]
    ax.fill_between(centros, 0, pontos, color=cor, alpha=0.16, linewidth=0)
    ax.plot(centros, pontos, color=cor, linewidth=1.5)
    ax.axhline(limiar, color=cor_sec, linestyle=(0, (4, 4)), linewidth=1)
    for t in trechos:
      if t["classe"] == classe:
        ax.barh(base / 2, t["fim_s"] - t["inicio_s"], left=t["inicio_s"], height=-base * 0.6, color=cor)
    ax.set_ylim(base, 1)
    ax.set_yticks([0, limiar, 1])
    ax.yaxis.set_major_formatter(FuncFormatter(virgula))
    for lado in ("top", "right"):
      ax.spines[lado].set_visible(False)
    rotulo(ax, nomes[classe], italico=True)

  eixos[-1].set_xlim(0, duracao)
  eixos[-1].xaxis.set_major_formatter(FuncFormatter(virgula))
  eixos[-1].set_xlabel("tempo (s)")

  saida = Path(saida)
  saida.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(saida.with_suffix(".png"), dpi=DPI)
  fig.savefig(saida.with_suffix(".pdf"))
  plt.close(fig)

  n_trechos = {c: sum(1 for t in trechos if t["classe"] == c) for c in estereotipias}
  print(f"{video_id}: {len(janelas)} janelas, limiar {limiar}, trechos por classe {n_trechos} -> {saida}.png/.pdf")


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("video_id")
  ap.add_argument("--saida", default="figuras/fig_linha_do_tempo")
  args = ap.parse_args()
  gerar(args.video_id, args.saida)
