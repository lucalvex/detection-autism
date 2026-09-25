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
