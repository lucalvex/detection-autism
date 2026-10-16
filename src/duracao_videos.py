# ============================================================
# ARQUIVO: src/duracao_videos.py
#
# O QUE FAZ: calcula a duração dos vídeos usados no trabalho (os de
# models/oof/fold_assignments.json) a partir do próprio arquivo de
# vídeo, com o ffprobe, e não das detecções de pose.
#   - quadros: contagem por decodificação (ffprobe -count_frames),
#     conferida com o cabeçalho do contêiner e com o OpenCV;
#   - taxa de quadros: avg_frame_rate do fluxo de vídeo, conferida com
#     a do OpenCV (a que o pipeline usa para converter os tempos do XML);
#   - duração em segundos = quadros / taxa de quadros.
#
# Registra também a faixa avaliada no TCC: do primeiro quadro até o
# último com pessoa detectada (coluna "frame" do CSV de pose + 1), que é
# o n_frames de evaluate_temporal.py. A soma dela é o total de quadros
# citado no TCC; a diferença para o arquivo são quadros finais sem
# pessoa detectada. Confere que nenhum intervalo anotado no XML cai
# nesses quadros finais (mesma conversão de evaluate_temporal.py:
# round(t * fps)) e para sem gravar se cair.
#
# Só lê vídeos, CSVs de pose e XMLs; grava apenas
# data/results/duracao_videos.json.
#
# Uso (da raiz do repositório):
#   python src/duracao_videos.py [--ffprobe CAMINHO] [--xmls PASTA]
# ============================================================

import argparse
import csv
import json
import shutil
import statistics
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ssbd_annotations import load_annotations  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
DIVISAO = RAIZ / "models" / "oof" / "fold_assignments.json"
VIDEOS = RAIZ / "data" / "videos"
POSES = RAIZ / "data" / "poses"
XMLS_PADRAO = RAIZ.parent / "download-yt" / "xmls"
SAIDA = RAIZ / "data" / "results" / "duracao_videos.json"
TOTAL_QUADROS_TCC = 125345   # total de quadros citado no TCC (faixa avaliada)
CATEGORIAS = {"armflapping": "ArmFlapping", "headbanging": "HeadBanging", "spinning": "Spinning"}


def sondar(ffprobe, arquivo):
  cmd = [
    ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames",
    "-show_entries", "stream=nb_frames,nb_read_frames,avg_frame_rate:format=duration",
    "-of", "json", str(arquivo),
  ]
  d = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
  s = d["streams"][0]
  cap = cv2.VideoCapture(str(arquivo))
  fps_cv = cap.get(cv2.CAP_PROP_FPS)
  quadros_cv = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  cap.release()
  x = {
    "quadros": int(s["nb_read_frames"]),
    "fps": float(Fraction(s["avg_frame_rate"])),
    "fps_fracao": s["avg_frame_rate"],
    "duracao_conteiner_s": float(d["format"]["duration"]),
  }
  if int(s["nb_frames"]) != x["quadros"] or quadros_cv != x["quadros"]:
    raise SystemExit(f"{arquivo.name}: contagens de quadros divergentes "
                     f"(decodificados {x['quadros']}, cabeçalho {s['nb_frames']}, OpenCV {quadros_cv}).")
  if abs(fps_cv - x["fps"]) > 1e-3:
    raise SystemExit(f"{arquivo.name}: taxa de quadros do ffprobe ({x['fps']}) difere da do OpenCV ({fps_cv}).")
  return x


def faixa_avaliada(video_id):
  with open(POSES / f"{video_id}.csv", newline="") as fh:
    return max(int(float(r["frame"])) for r in csv.DictReader(fh)) + 1


def anotacoes_nos_quadros_finais(xml, fps, n_avaliados):
  casos = []
  for b in load_annotations(xml)["behaviours"]:
    classe = CATEGORIAS.get(b["category"])
    if classe is None:
      continue
    ini = max(0, int(round(b["start_s"] * fps)))
    fim = int(round(b["end_s"] * fps))
    if fim >= n_avaliados:
      casos.append({"classe": classe, "inicio_s": b["start_s"], "fim_s": b["end_s"],
                    "quadros_fora": fim - max(ini, n_avaliados) + 1})
  return casos


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--ffprobe", default=shutil.which("ffprobe") or "ffprobe")
  ap.add_argument("--xmls", type=Path, default=XMLS_PADRAO)
  args = ap.parse_args()

  ids = sorted(json.loads(DIVISAO.read_text(encoding="utf-8"))["video_to_fold"])
  por_video = {}
  problemas = []
  for vid in ids:
    arquivo = VIDEOS / f"{vid}.mp4"
    if not arquivo.exists():
      raise SystemExit(f"Vídeo não encontrado: {arquivo}")
    x = sondar(args.ffprobe, arquivo)
    x["duracao_s"] = x["quadros"] / x["fps"]
    x["quadros_avaliados"] = faixa_avaliada(vid)
    x["quadros_finais_sem_deteccao"] = x["quadros"] - x["quadros_avaliados"]
    if x["quadros_finais_sem_deteccao"] > 0:
      casos = anotacoes_nos_quadros_finais(args.xmls / f"{vid}.xml", x["fps"], x["quadros_avaliados"])
      problemas += [(vid, c) for c in casos]
    por_video[vid] = x
    print(f"{vid}: {x['quadros']} quadros, {x['fps']:.3f} fps, {x['duracao_s']:.2f} s, "
          f"{x['quadros_finais_sem_deteccao']} finais sem detecção", flush=True)

  if problemas:
    for vid, c in problemas:
      print(f"  {vid}: {c}")
    raise SystemExit("Há intervalos anotados nos quadros finais sem detecção. Nada foi gravado.")

  soma = sum(x["quadros"] for x in por_video.values())
  soma_avaliada = sum(x["quadros_avaliados"] for x in por_video.values())
  print(f"\nQuadros nos arquivos: {soma}; faixa avaliada: {soma_avaliada}; TCC: {TOTAL_QUADROS_TCC}")
  if soma_avaliada != TOTAL_QUADROS_TCC:
    raise SystemExit(f"A faixa avaliada soma {soma_avaliada}, não {TOTAL_QUADROS_TCC}. Nada foi gravado.")

  dur = [x["duracao_s"] for x in por_video.values()]
  fps = [x["fps"] for x in por_video.values()]
  finais = {v: x["quadros_finais_sem_deteccao"] for v, x in por_video.items() if x["quadros_finais_sem_deteccao"]}
  resumo = {
    "n_videos": len(por_video),
    "quadros_nos_arquivos": soma,
    "duracao_media_s": statistics.mean(dur),
    "duracao_mediana_s": statistics.median(dur),
    "duracao_minima_s": min(dur),
    "video_mais_curto": min(por_video, key=lambda v: por_video[v]["duracao_s"]),
    "duracao_maxima_s": max(dur),
    "video_mais_longo": max(por_video, key=lambda v: por_video[v]["duracao_s"]),
    "duracao_total_s": sum(dur),
    "duracao_total_min": sum(dur) / 60,
    "fps_minimo": min(fps),
    "video_fps_minimo": min(por_video, key=lambda v: por_video[v]["fps"]),
    "fps_maximo": max(fps),
    "n_videos_fps_maximo": sum(1 for f in fps if f == max(fps)),
  }
  faixa = {
    "descricao": (f"Os {TOTAL_QUADROS_TCC} quadros citados no TCC são a faixa avaliada: em cada vídeo, do primeiro "
                  "quadro ao último com pessoa detectada (n_frames de evaluate_temporal.py, que é o fim da última "
                  "janela). Os quadros depois do último com pessoa detectada não entram na avaliação; nenhum "
                  "intervalo anotado no XML cai neles."),
    "quadros_faixa_avaliada": soma_avaliada,
    "confere_com_total_do_tcc": soma_avaliada == TOTAL_QUADROS_TCC,
    "quadros_finais_sem_deteccao_total": soma - soma_avaliada,
    "quadros_finais_sem_deteccao_por_video": finais,
    "intervalos_anotados_nos_quadros_finais": len(problemas),
  }
  saida = {
    "descricao": ("Duração dos vídeos usados no trabalho (models/oof/fold_assignments.json), lida do próprio arquivo "
                  "de vídeo, não das detecções de pose. quadros = contagem por decodificação (ffprobe -count_frames), "
                  "igual ao cabeçalho do contêiner e ao OpenCV; fps = avg_frame_rate do fluxo de vídeo, igual ao do "
                  "OpenCV; duracao_s = quadros / fps. duracao_conteiner_s é a duração informada pelo contêiner, só "
                  "para conferência. quadros_avaliados = último quadro com pessoa detectada + 1."),
    "gerado_por": "src/duracao_videos.py",
    "resumo": resumo,
    "faixa_avaliada_no_tcc": faixa,
    "por_video": por_video,
  }
  SAIDA.write_text(json.dumps(saida, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
  print(json.dumps({"resumo": resumo, "faixa_avaliada_no_tcc": faixa}, ensure_ascii=False, indent=2))
  print(f"Gravado em {SAIDA.relative_to(RAIZ)}")


if __name__ == "__main__":
  main()
