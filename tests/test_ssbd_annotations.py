# ============================================================
# ARQUIVO: tests/test_ssbd_annotations.py
#
# O QUE FAZ: testa o leitor de anotações XML do SSBD
# (src/ssbd_annotations.py) -- a regra de tempo (minutos+segundos por
# token, separador ':' ou '-') foi validada contra os 75 arquivos
# reais antes de escrever o leitor; este teste reproduz essa
# validação para não regredir. Rode com:
#   python tests/test_ssbd_annotations.py
#
# Espera as anotações reais em ../download-yt/xmls/ (fora deste
# repositório) -- se não encontrar, pula a validação contra os
# arquivos reais e roda só os testes unitários da regra de tempo.
# ============================================================

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ssbd_annotations import find_annotation_file, load_annotations, parse_interval, parse_time_token

XMLS_DIR = Path(__file__).resolve().parent.parent.parent / "download-yt" / "xmls"


def test_parse_time_token():

  assert parse_time_token("18") == 18
  assert parse_time_token("0338") == 218  # 3 min 38 s
  assert parse_time_token("0002") == 2


def test_parse_interval_aceita_dois_delimitadores():

  assert parse_interval("18:24") == (18, 24)
  assert parse_interval("0006-0018") == (6, 18)


def test_arquivos_reais():

  if not XMLS_DIR.exists():
    print(f"[AVISO] {XMLS_DIR} não encontrado -- pulando validação contra arquivos reais")
    return

  files = sorted(XMLS_DIR.glob("*.xml"))
  assert len(files) == 75, f"esperava 75 arquivos, achei {len(files)}"

  all_categories = set()
  total_behaviours = 0

  for f in files:

    ann = load_annotations(f)

    assert ann["video_id"] == f.stem
    assert ann["duration_s"] > 0

    for b in ann["behaviours"]:

      total_behaviours += 1
      all_categories.add(b["category"])

      assert b["end_s"] <= ann["duration_s"], (
        f"{f.name}: behaviour {b['id']} end_s={b['end_s']} > duration_s={ann['duration_s']}"
      )
      assert b["start_s"] <= b["end_s"]

  assert total_behaviours == 133
  assert all_categories == {"armflapping", "headbanging", "spinning"}

  print(f"OK: {len(files)}/75 arquivos, {total_behaviours} comportamentos, categorias={all_categories}")


if __name__ == "__main__":

  test_parse_time_token()
  print("OK: parse_time_token (regra minutos+segundos por token)")

  test_parse_interval_aceita_dois_delimitadores()
  print("OK: parse_interval aceita ':' e '-'")

  test_arquivos_reais()

  print("\nALL SSBD ANNOTATION TESTS PASSED")
