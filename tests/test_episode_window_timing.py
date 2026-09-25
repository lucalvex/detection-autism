# ============================================================
# ARQUIVO: tests/test_episode_window_timing.py
#
# O QUE FAZ: testa a correção do bug em episode_analysis.py onde
# quadro_inicio/quadro_fim de cada janela eram calculados como
# `frame_id_atual + 1 - SEQUENCE_LENGTH`, assumindo (errado) que as
# SEQUENCE_LENGTH detecções mais recentes do buffer são sempre quadros
# reais consecutivos. Quando há um gap de detecção (YOLO não encontra
# pessoa em 1+ quadros) dentro dessa janela, essa suposição é falsa --
# o buffer ainda tem 30 detecções, mas elas vêm de mais de 30 quadros
# reais.
#
# A correção guarda o frame_id real junto de cada vetor de features no
# buffer e usa o frame_id da detecção mais antiga/mais recente do
# buffer para quadro_inicio/quadro_fim -- mesmo princípio já usado em
# evaluate_temporal.py (coluna "frame" do CSV em vez de posição de
# linha).
#
# Aqui simulamos só a mecânica de buffer/janela (a parte com o bug),
# sem YOLO/LSTM -- é a única coisa que precisa ser testada em
# isolamento; `episode_analysis.py` roda como script (sem função
# separada), então replicamos a lógica exata dos dois lados (antes e
# depois da correção) para comparar.
#
# Rode com: python tests/test_episode_window_timing.py
# ============================================================

from collections import deque

SEQUENCE_LENGTH = 30


def simulate_buffer_windows(detected_frame_ids):
  """Réplica exata da lógica de buffer/janela de episode_analysis.py
  DEPOIS da correção: cada item do buffer é (frame_id, features) --
  aqui usamos None no lugar do vetor de 68 floats, já que só a lógica
  de indexação está sendo testada. Retorna 1 (quadro_inicio,
  quadro_fim) por janela completa."""

  buffer = deque(maxlen=SEQUENCE_LENGTH)
  windows = []

  for frame_id in detected_frame_ids:

    buffer.append((frame_id, None))

    if len(buffer) == SEQUENCE_LENGTH:
      quadro_inicio = buffer[0][0]
      quadro_fim = buffer[-1][0] + 1
      windows.append((quadro_inicio, quadro_fim))

  return windows


def old_buggy_windows(detected_frame_ids):
  """Réplica da lógica ANTES da correção: assume que o buffer sempre
  contém os últimos SEQUENCE_LENGTH quadros reais consecutivos
  (quadro_fim = frame_id_atual + 1, quadro_inicio = quadro_fim -
  SEQUENCE_LENGTH), ignorando qualquer gap de detecção."""

  windows = []
  count = 0

  for frame_id in detected_frame_ids:

    count += 1

    if count >= SEQUENCE_LENGTH:
      quadro_fim = frame_id + 1
      quadro_inicio = quadro_fim - SEQUENCE_LENGTH
      windows.append((quadro_inicio, quadro_fim))

  return windows


def test_sem_gap_bate_com_a_logica_antiga():
  """Sem nenhum gap de detecção (frame_ids 0..59, todos consecutivos),
  a correção não deveria mudar nenhum resultado -- serve de regressão."""

  frame_ids = list(range(60))

  new_windows = simulate_buffer_windows(frame_ids)
  old_windows = old_buggy_windows(frame_ids)

  assert new_windows == old_windows
  assert new_windows[0] == (0, 30)
  assert new_windows[-1] == (30, 60)


def test_gap_no_meio_de_uma_janela_desloca_quadro_inicio():
  """Vídeo sintético de 45 quadros reais (0..44) onde os quadros 10 a
  14 (5 quadros, caindo dentro do que seria a 1a janela) não tiveram
  pessoa detectada -- esses frame_ids nunca chegam à função, porque o
  loop real de episode_analysis.py já os pulou com "continue" antes de
  anexar ao buffer."""

  todos_os_quadros = list(range(45))
  quadros_sem_deteccao = set(range(10, 15))  # 5 quadros de gap
  frame_ids_detectados = [f for f in todos_os_quadros if f not in quadros_sem_deteccao]

  assert len(frame_ids_detectados) == 40

  # a 30a deteccao da lista (indice 29) e' o frame_id real 34, nao 29,
  # porque 5 quadros foram pulados antes dele
  assert frame_ids_detectados[29] == 34

  new_windows = simulate_buffer_windows(frame_ids_detectados)
  old_windows = old_buggy_windows(frame_ids_detectados)

  # ANTES (bug): quadro_fim = 34+1 = 35; quadro_inicio = 35-30 = 5 --
  # assume os quadros 5..34 consecutivos, mas 5 deles (10-14) nao
  # existem no buffer (o buffer tem so 30 deteccoes, nao 30 quadros).
  assert old_windows[0] == (5, 35)

  # DEPOIS (corrigido): quadro_inicio vem do frame_id real da deteccao
  # mais antiga no buffer nesse momento (frame_id=0), quadro_fim da
  # mais recente (frame_id=34) + 1.
  assert new_windows[0] == (0, 35)

  # o deslocamento no quadro_inicio errado e' exatamente do tamanho do
  # gap (5 quadros) que caiu dentro da janela
  deslocamento = old_windows[0][0] - new_windows[0][0]
  assert deslocamento == 5

  # quadro_fim nao e' afetado neste caso -- o gap fica no meio da
  # janela, nao no fim
  assert old_windows[0][1] == new_windows[0][1] == 35


if __name__ == "__main__":

  test_sem_gap_bate_com_a_logica_antiga()
  print("OK: sem gap, a janela nova bate com a lógica antiga (regressão)")

  test_gap_no_meio_de_uma_janela_desloca_quadro_inicio()
  print("OK: gap de 5 quadros no meio da janela -- quadro_inicio corrigido, deslocamento de 5 quadros confirmado")

  print("\nALL EPISODE WINDOW TIMING TESTS PASSED")
