# ============================================================
# ARQUIVO: experimentos/qualidade-dataset/qualidade_dataset.py
#
# O QUE FAZ: medidas simples de qualidade dos 56 vídeos, sem treinar nada
# e sem gravar nenhum quadro de vídeo:
#   - imagem: resolução, taxa de quadros, bits por pixel por quadro
#     (bit_rate do fluxo de vídeo / (largura x altura x fps), via ffprobe)
#     e nitidez média (variância do Laplaciano em cinza, 1 quadro por s);
#   - movimento: fluxo óptico de Farnebäck entre quadros consecutivos
#     reduzidos a 320 px de largura, em 2 pares por segundo; tamanho
#     médio dos vetores por par e % de pares com mudança brusca (média
#     acima de 3x a mediana do próprio vídeo);
#   - pose (YOLO11n, experimentos/poses): % de quadros do vídeo com 0,
#     1-4, 5-10 e 11-17 pontos com confiança >= 0,5 (quadro sem pessoa
#     conta como 0) e % de quadros válidos pelo critério Q;
#   - F1 por quadro do r2-C4-aumento nos quadros anotados (regra de
#     selecionar_videos_mediana.py), média das 3 sementes, no teste.
# Grava tabela-videos.csv nesta pasta (lida pelo passo 0 da rodada 4).
#
# Uso (da raiz do repositório):
#   .venv/Scripts/python.exe experimentos/qualidade-dataset/qualidade_dataset.py [--ffprobe CAMINHO]
# ============================================================

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(RAIZ / "experimentos"), str(RAIZ / "src")]
import rodar_tudo as r  # noqa: E402  (faz chdir para a raiz)
import rodar_q as rq  # noqa: E402

PASTA = Path(__file__).resolve().parent
FINAL = {42: "r2-C4-aumento", 1000: "r2-final-semente-1000", 2000: "r2-final-semente-2000"}
SIGLA = {"ArmFlapping": "AF", "HeadBanging": "HB", "Spinning": "SP"}
FAIXAS = [(0, 0), (1, 4), (5, 10), (11, 17)]


def medidas_imagem_e_movimento(vid, ffprobe):
  info = json.loads(subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                                    "stream=width,height,bit_rate,avg_frame_rate:format=bit_rate", "-of", "json",
                                    str(RAIZ / "data/videos" / f"{vid}.mp4")], capture_output=True, text=True, check=True).stdout)
  s = info["streams"][0]
  w, h = int(s["width"]), int(s["height"])
  a, b = s["avg_frame_rate"].split("/")
  fps = int(a) / int(b)
  bits = float(s.get("bit_rate") or info["format"]["bit_rate"])
  cap = cv2.VideoCapture(str(RAIZ / "data/videos" / f"{vid}.mp4"))
  passo_nit, passo_mov = max(1, round(fps)), max(1, round(fps / 2))
  nitidez, movimento, anterior, i = [], [], None, 0
  while True:
    ok, img = cap.read()
    if not ok:
      break
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if i % passo_nit == 0:
      nitidez.append(cv2.Laplacian(cinza, cv2.CV_64F).var())
    pequeno = cv2.resize(cinza, (320, max(1, round(h * 320 / w))), interpolation=cv2.INTER_AREA)
    if anterior is not None and (i - 1) % passo_mov == 0:  # par (i-1, i): quadros consecutivos
      fluxo = cv2.calcOpticalFlowFarneback(anterior, pequeno, None, 0.5, 3, 15, 3, 5, 1.2, 0)
      movimento.append(float(np.linalg.norm(fluxo, axis=2).mean()))
    anterior = pequeno
    i += 1
  cap.release()
  mov = np.array(movimento)
  return {"largura": w, "altura": h, "fps": fps, "bpp": bits / (w * h * fps), "nitidez": float(np.mean(nitidez)),
          "movimento": float(mov.mean()), "brusco_pct": float(100 * (mov > 3 * np.median(mov)).mean()), "quadros": i}


def medidas_pose(vid, n_quadros):
  d = np.load(r.caminho("poses", "yolo11n", f"{vid}.npz"))
  n_conf = np.zeros(n_quadros, int)  # quadro sem pessoa: 0 pontos
  n_conf[d["frames"].astype(int)] = (d["conf"] >= r.LIMIAR_CONF).sum(1)
  out = {f"pose_{a}_{b}_pct": float(100 * ((n_conf >= a) & (n_conf <= b)).mean()) for a, b in FAIXAS}
  out["q_pct"] = 100 * rq.valido_q(d["conf"]).sum() / n_quadros
  return out


def f1_por_video():
  from selecionar_videos_mediana import f1_do_video
  cfg = {**r.ler_json(RAIZ / "experimentos/r2-C4-aumento/config.json")}
  cfg = {k: v for k, v in cfg.items() if k in r.BASE}
  por = {}
  for s, d in FINAL.items():
    vd = r.dados_avaliacao(RAIZ / "experimentos" / d, "teste", {**cfg, "semente": s})
    for vid, x in vd.items():
      f = f1_do_video(x)[0]
      if f is not None:
        por.setdefault(vid, []).append(f)
  return {v: float(np.mean(f)) for v, f in por.items()}


def abreviar(vid):
  return f"{SIGLA[r.classe_do_video(vid)]}{vid.rsplit('_', 1)[1]}"


def gravar(linhas):
  with open(PASTA / "tabela-videos.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(linhas[0]))
    w.writeheader()
    w.writerows(linhas)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--ffprobe", default=shutil.which("ffprobe") or r"C:\projetos\tcc\ffmpeg-8.1.1\ffmpeg-8.1.1\bin\ffprobe.exe")
  args = ap.parse_args()
  r.OPC.update({"saida": str(RAIZ / "experimentos"), "teste": False})
  f1 = f1_por_video()
  linhas = []
  for vid in r.videos():
    m = medidas_imagem_e_movimento(vid, args.ffprobe)
    linhas.append({"video": vid, "abrev": abreviar(vid), "classe": r.classe_do_video(vid), **m, **medidas_pose(vid, m["quadros"]),
                   "f1": f1.get(vid)})
    print(f"{vid}: nitidez {m['nitidez']:.0f}, movimento {m['movimento']:.2f}, bpp {m['bpp']:.3f}", flush=True)
  assert len(linhas) == 56 and len({x["abrev"] for x in linhas}) == 56, "abreviações repetidas ou vídeos faltando"
  gravar(linhas)


if __name__ == "__main__":
  main()
