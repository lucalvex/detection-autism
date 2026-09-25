# ============================================================
# ARQUIVO: tests/test_timeline.py
#
# O QUE FAZ: testa build_segments() (src/timeline.py) -- a regra
# única de segmentação de trechos usada por episode_analysis.py e
# evaluate_temporal.py. Rode com: python tests/test_timeline.py
# ============================================================

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from timeline import build_segments

FPS = 20.0
SEQUENCE_LENGTH = 30
CLASS_NAMES = ["A", "B"]


def window(start, scores):
  return {"quadro_inicio": start, "quadro_fim": start + SEQUENCE_LENGTH, "pontuacoes": scores}


def test_janela_isolada_dura_sequence_length_sobre_fps():

  windows = [window(0, [0.9, 0.1])]
  segs = build_segments(windows, CLASS_NAMES, threshold=0.5, min_duration_sec=0.0, fps=FPS)

  assert len(segs) == 1
  s = segs[0]
  assert s["classe"] == "A"
  assert s["quadro_inicio"] == 0
  assert s["quadro_fim"] == 30
  assert abs(s["duracao_s"] - 30 / FPS) < 1e-9


def test_corrida_de_janelas_consecutivas():

  windows = [window(i, [0.9, 0.1]) for i in range(3)]
  segs = build_segments(windows, CLASS_NAMES, threshold=0.5, min_duration_sec=0.0, fps=FPS)

  assert len(segs) == 1
  s = segs[0]
  assert s["quadro_inicio"] == 0
  assert s["quadro_fim"] == 2 + SEQUENCE_LENGTH
  assert abs(s["duracao_s"] - (32 / FPS)) < 1e-9


def test_gap_quebra_em_dois_trechos():

  windows = [
    window(0, [0.9, 0.1]),
    window(1, [0.9, 0.1]),
    window(2, [0.1, 0.9]),
    window(3, [0.9, 0.1]),
  ]
  segs = build_segments(windows, CLASS_NAMES, threshold=0.5, min_duration_sec=0.0, fps=FPS)

  classes_found = [s["classe"] for s in segs]
  assert classes_found.count("A") == 2
  assert classes_found.count("B") == 1


def test_filtro_duracao_minima():

  windows = [window(0, [0.9, 0.1])]  # 1 janela isolada = 1.5s de duração

  segs_low = build_segments(windows, CLASS_NAMES, threshold=0.5, min_duration_sec=1.0, fps=FPS)
  segs_high = build_segments(windows, CLASS_NAMES, threshold=0.5, min_duration_sec=2.0, fps=FPS)

  assert len(segs_low) == 1
  assert len(segs_high) == 0


def test_pico_e_a_maior_pontuacao_da_corrida():

  windows = [window(0, [0.6, 0.0]), window(1, [0.95, 0.0]), window(2, [0.7, 0.0])]
  segs = build_segments(windows, CLASS_NAMES, threshold=0.5, min_duration_sec=0.0, fps=FPS)

  assert len(segs) == 1
  assert abs(segs[0]["pico"] - 0.95) < 1e-9


if __name__ == "__main__":

  test_janela_isolada_dura_sequence_length_sobre_fps()
  print("OK: janela isolada dura SEQUENCE_LENGTH/fps segundos")

  test_corrida_de_janelas_consecutivas()
  print("OK: corrida de janelas consecutivas mede do inicio da 1a ao fim da ultima")

  test_gap_quebra_em_dois_trechos()
  print("OK: gap no meio quebra a corrida em trechos separados")

  test_filtro_duracao_minima()
  print("OK: filtro de duracao minima descarta/mantem corretamente")

  test_pico_e_a_maior_pontuacao_da_corrida()
  print("OK: pico e a maior pontuacao entre as janelas do trecho")

  print("\nALL TIMELINE TESTS PASSED")
