# ============================================================
# ARQUIVO: timeline.py
#
# O QUE FAZ: implementa a ÚNICA regra usada no projeto inteiro para
# transformar pontuações por janela em "trechos" (segmentos de tempo)
# -- a mesma regra descrita em HANDOFF.md / episodes() de
# revisao.dc.html, reproduzida aqui em Python para ser usada por
# episode_analysis.py e pela avaliação temporal (evaluate_temporal.py).
# A interface web (JS) implementa a MESMA fórmula -- ver comentário
# de sincronização abaixo.
#
# REGRA (igual para os três lugares que constroem "trechos"):
#   Um trecho de uma classe é uma sequência de janelas CONSECUTIVAS
#   (mesmo passo usado no treino/exportação) cuja pontuação daquela
#   classe é >= limiar. O trecho vai do quadro_inicio da PRIMEIRA
#   janela da sequência ao quadro_fim da ÚLTIMA. Só entra na lista
#   se durar pelo menos duracao_minima_s. O pico é a maior pontuação
#   entre as janelas do trecho.
#
# POR QUE ISSO IMPORTA (achado real, não hipotético): cada janela
# cobre SEQUENCE_LENGTH quadros. Medir do quadro_inicio da primeira
# janela ao quadro_fim da última (em vez de só do frame onde a
# primeira/última predição "aconteceu") faz TODO trecho, mesmo de 1
# janela isolada, durar pelo menos SEQUENCE_LENGTH quadros -- é
# SEQUENCE_LENGTH-1 quadros a mais do que medir só a distância entre
# os frames-índice das predições. Em fps=20 (ex.: v_HeadBanging_18.mp4)
# isso é 1.45s a mais em cada trecho. Um filtro de duração mínima de
# 0.5s ou 1.0s não filtra nada sob esta regra, porque o trecho mais
# curto possível já mede SEQUENCE_LENGTH/fps segundos.
# ============================================================


def build_segments(windows, class_names, threshold, min_duration_sec, fps):
  """Constrói os trechos de cada classe a partir das pontuações por janela.

  windows: lista de dicts, 1 por janela, na ordem em que foram
    processadas (quadro_inicio crescente), cada um com:
      - "quadro_inicio": int (primeiro quadro coberto pela janela)
      - "quadro_fim": int (quadro seguinte ao último coberto -- ou
        seja, a janela cobre [quadro_inicio, quadro_fim))
      - "pontuacoes": lista de floats, na mesma ordem de class_names
  class_names: nomes das classes, na mesma ordem de "pontuacoes".
  threshold: pontuação mínima para uma janela "ativar" uma classe.
  min_duration_sec: trechos mais curtos que isso são descartados.
  fps: usado para converter quadros em segundos.

  Retorna uma lista de trechos (um dict por trecho), cada um:
    {"classe": nome, "quadro_inicio": int, "quadro_fim": int,
     "inicio_s": float, "fim_s": float, "duracao_s": float, "pico": float}

  Cada classe é avaliada de forma independente (como em
  revisao.dc.html): trechos de classes diferentes podem se sobrepor
  no tempo se o limiar for baixo o suficiente para mais de uma classe
  passar na mesma janela.
  """

  segments = []

  for class_idx, class_name in enumerate(class_names):

    run_start = None  # índice (na lista `windows`) do início do trecho atual

    for i, window in enumerate(windows):

      active = window["pontuacoes"][class_idx] >= threshold

      if active and run_start is None:
        run_start = i

      run_ends_here = run_start is not None and (
        not active or i == len(windows) - 1
      )

      if run_ends_here:

        # se a janela atual ainda está ativa (é a última da lista),
        # ela faz parte do trecho; senão, o trecho parou na anterior
        run_end = i if active else i - 1

        quadro_inicio = windows[run_start]["quadro_inicio"]
        quadro_fim = windows[run_end]["quadro_fim"]
        duracao_s = (quadro_fim - quadro_inicio) / fps

        if duracao_s >= min_duration_sec:

          pico = max(
            windows[j]["pontuacoes"][class_idx]
            for j in range(run_start, run_end + 1)
          )

          segments.append({
            "classe": class_name,
            "quadro_inicio": quadro_inicio,
            "quadro_fim": quadro_fim,
            "inicio_s": quadro_inicio / fps,
            "fim_s": quadro_fim / fps,
            "duracao_s": duracao_s,
            "pico": pico,
          })

        run_start = None

  return segments


# ---------- pós-processamento do modelo final (r4b_C1_janela_4s) ----------
# Escolhido na validação: mediana de 2 s sobre as pontuações e junção de
# trechos da mesma classe separados por menos de 1 s. Mesmas fórmulas de
# f1_evento_pos() e juntar() em experimentos/rodar_rodada2.py. A interface
# (demo/js/timeline.js) repete as duas; tests/test_timeline_js.py exige
# igualdade exata.

MEDIANA_S = 2.0
JUNTAR_S = 1.0


def median_scores(windows, fps, median_sec):
  """Mediana móvel centrada de median_sec segundos (tamanho ímpar,
  bordas repetidas) sobre as pontuações de cada classe, na ordem das janelas."""

  if not median_sec or not windows:
    return windows

  import numpy as np
  from scipy.ndimage import median_filter

  k = int(round(median_sec * fps)) // 2 * 2 + 1
  P = median_filter(np.array([w["pontuacoes"] for w in windows]), size=(k, 1), mode="nearest")
  return [{**w, "pontuacoes": p.tolist()} for w, p in zip(windows, P)]


def merge_segments(segments, gap_sec):
  """Junta trechos da mesma classe separados por menos de gap_sec segundos."""

  out = []
  for c in dict.fromkeys(s["classe"] for s in segments):
    atual = None
    for s in sorted((s for s in segments if s["classe"] == c), key=lambda s: s["inicio_s"]):
      if atual and s["inicio_s"] - atual["fim_s"] < gap_sec:
        atual.update(fim_s=max(atual["fim_s"], s["fim_s"]), quadro_fim=max(atual["quadro_fim"], s["quadro_fim"]),
                     pico=max(atual["pico"], s["pico"]))
        atual["duracao_s"] = atual["fim_s"] - atual["inicio_s"]
      else:
        atual = dict(s)
        out.append(atual)
  return out


def final_segments(windows, class_names, threshold, fps, median_sec=MEDIANA_S, gap_sec=JUNTAR_S):
  """Trechos do modelo final: mediana, build_segments sem duração mínima e junção."""

  return merge_segments(build_segments(median_scores(windows, fps, median_sec), class_names, threshold, 0.0, fps), gap_sec)
