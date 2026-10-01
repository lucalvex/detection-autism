# ============================================================
# ARQUIVO: tests/test_timeline_js.py
#
# O QUE FAZ: exige que a buildSegments() em JavaScript da interface
# (demo/js/timeline.js) dê exatamente os mesmos trechos que a
# build_segments() em Python (src/timeline.py), para vários vídeos,
# limiares e durações mínimas. Os dois lados leem o MESMO arquivo JSON
# de entrada, então os números de ponto flutuante são idênticos; a
# comparação é de igualdade exata, campo a campo, sem tolerância.
#
# Casos:
#   - os 56 vídeos do SSBD, com as pontuações da Rodada A fora da
#     partição (evaluate_temporal.compute_windows), as 4 classes;
#   - os _sessao.json exportados em saidas/ (o que a interface lê);
#   - casos sintéticos de borda.
#   Limiares: os da grade da avaliação (0,30 a 0,90, passo 0,05).
#   Durações mínimas: uma janela (regra principal), 0, 1 s e 3 s.
#
# Requer Node.js. Rode da raiz do repositório:
#   python tests/test_timeline_js.py
# ============================================================

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

import evaluate_temporal as et  # noqa: E402
from timeline import build_segments  # noqa: E402

SEQUENCE_LENGTH = 30
LIMIARES = [float(t) for t in np.round(np.arange(0.30, 0.90 + 1e-9, 0.05), 2)]


def janelas_sinteticas():
  def w(i, scores):
    return {"quadro_inicio": i, "quadro_fim": i + SEQUENCE_LENGTH, "pontuacoes": scores}

  return [
    ("vazio", []),
    ("uma_janela_ativa", [w(0, [0.9, 0.1])]),
    ("uma_janela_inativa", [w(0, [0.1, 0.9])]),
    ("igual_ao_limiar", [w(0, [0.5, 0.5]), w(1, [0.5, 0.5]), w(2, [0.4999999, 0.5000001])]),
    ("ativa_ate_o_fim", [w(0, [0.2, 0.8]), w(1, [0.7, 0.3]), w(2, [0.8, 0.2])]),
    ("tudo_ativo", [w(i, [0.95, 0.95]) for i in range(5)]),
    ("alternando", [w(i, [0.9 if i % 2 else 0.1, 0.5]) for i in range(9)]),
    ("lacuna_de_quadros", [
      {"quadro_inicio": 0, "quadro_fim": 30, "pontuacoes": [0.9, 0.1]},
      {"quadro_inicio": 1, "quadro_fim": 95, "pontuacoes": [0.9, 0.1]},
      {"quadro_inicio": 70, "quadro_fim": 100, "pontuacoes": [0.2, 0.8]},
    ]),
  ]


def montar_casos():
  videos = []

  et.load_config("data/datasets_annotated", "models/oof_seeded_annotated")
  dados = et.load_all_videos()
  for vid in sorted(dados):
    videos.append({
      "nome": vid, "fps": dados[vid]["fps"], "class_names": et.class_names,
      "windows": dados[vid]["windows"],
    })

  for path in sorted((RAIZ / "saidas").glob("*_sessao.json")):
    s = json.load(open(path, encoding="utf-8"))
    videos.append({
      "nome": path.name, "fps": s["sessao"]["fps"],
      "class_names": [c["id"] for c in s["classes"]],
      "windows": [{k: j[k] for k in ("quadro_inicio", "quadro_fim", "pontuacoes")} for j in s["janelas"]],
    })

  for nome, janelas in janelas_sinteticas():
    videos.append({"nome": "sintetico_" + nome, "fps": 30.0, "class_names": ["a", "b"], "windows": janelas})

  casos = []
  for i, v in enumerate(videos):
    duracoes = [SEQUENCE_LENGTH / v["fps"], 0.0, 1.0, 3.0]
    for limiar in LIMIARES:
      for dur in duracoes:
        casos.append({"video": i, "threshold": limiar, "min_duration_sec": dur})

  return videos, casos


def test_js_igual_ao_python():

  videos, casos = montar_casos()

  with tempfile.TemporaryDirectory() as tmp:
    entrada = Path(tmp) / "casos.json"
    saida = Path(tmp) / "saida_js.json"

    with open(entrada, "w", encoding="utf-8") as f:
      json.dump({"videos": videos, "casos": casos}, f)

    # os dois lados usam os dados relidos do mesmo arquivo
    dados = json.load(open(entrada, encoding="utf-8"))

    subprocess.run(
      ["node", str(RAIZ / "tests" / "js" / "segmentos_js.mjs"), str(entrada), str(saida)],
      check=True,
    )
    resultados_js = json.load(open(saida, encoding="utf-8"))

  assert len(resultados_js) == len(dados["casos"])

  n_trechos = 0
  for caso, js in zip(dados["casos"], resultados_js):
    v = dados["videos"][caso["video"]]
    py = build_segments(v["windows"], v["class_names"], caso["threshold"], caso["min_duration_sec"], v["fps"])
    assert py == js, (
      f"divergência em {v['nome']}, limiar={caso['threshold']}, duração mínima={caso['min_duration_sec']}:\n"
      f"  python={py[:3]}\n  js={js[:3]}"
    )
    n_trechos += len(py)

  n_videos_reais = sum(1 for v in dados["videos"] if not v["nome"].startswith("sintetico_"))
  return len(dados["casos"]), n_videos_reais, len(dados["videos"]) - n_videos_reais, n_trechos


if __name__ == "__main__":

  n_casos, n_reais, n_sint, n_trechos = test_js_igual_ao_python()
  print(
    f"OK: {n_casos} casos ({n_reais} vídeos reais/exportados + {n_sint} sintéticos, "
    f"{len(LIMIARES)} limiares, 4 durações mínimas), {n_trechos} trechos, "
    "JavaScript idêntico ao Python em todos"
  )
  print("\nALL TIMELINE JS TESTS PASSED")
