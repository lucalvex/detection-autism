# ============================================================
# ARQUIVO: experimentos/frequencia_r4b.py
#
# O QUE FAZ: parte B da rodada 4b refeita (adendo em protocolo-rodada4b.md),
# sem treino: frequência dominante dos movimentos dentro dos intervalos
# anotados, com remoção da tendência linear, faixa de 0,5 a 7 Hz e janelas
# de 8 s e de 4 s. Mesmos sinais e posições da parte B original
# (rodar_rodada4b.sinal). Grava r4b-B2-frequencia/frequencia.json e .md.
#
# Uso (da raiz do repositório):
#   .venv/Scripts/python.exe experimentos/frequencia_r4b.py
# ============================================================

import sys
from pathlib import Path

import numpy as np
from scipy.signal import detrend

RAIZ = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(RAIZ / "experimentos"), str(RAIZ / "src")]
import rodar_rodada4b as r4b  # noqa: E402

r, q, rq = r4b.r, r4b.q, r4b.rq
FS, BANDA, JANELAS = r4b.FS, (0.5, 7.0), (8, 4)
SAIDA = RAIZ / "experimentos" / "r4b-B2-frequencia"


def dominante(W):
  Z = detrend(W, axis=0, type="linear") * np.hanning(len(W))[:, None]
  pot = (np.abs(np.fft.rfft(Z, n=512, axis=0)) ** 2).sum(1)
  f = np.fft.rfftfreq(512, 1 / FS)
  banda = (f >= BANDA[0]) & (f <= BANDA[1])
  return float(f[banda][pot[banda].argmax()])


def autoverificacao():
  t = np.arange(8 * FS) / FS
  seno = (np.sin(2 * np.pi * 2.0 * t) + 3 * t)[:, None]  # 2 Hz com deriva linear forte
  assert abs(dominante(seno) - 2.0) < 0.1, "a tendência linear deveria ser removida"


def main():
  autoverificacao()
  import evaluate_temporal as et
  cfg = {**q.NOVOS, **rq.cfg_de(RAIZ / "experimentos" / r4b.REF)}
  dados = {(c, j): {"freq": [], "episodios_com_janela": 0, "videos": set()} for c in r.CLASSES[:3] for j in JANELAS}
  episodios = {c: 0 for c in r.CLASSES[:3]}
  for vid in r.videos():
    v = r.preparar_video(vid, cfg)
    if v is None:
      continue
    t = v["quadros"] / v["fps"]
    _, iv = et.ground_truth_frames(vid, int(v["quadros"][-1]) + 1, v["fps"])
    for lab, a, b in iv:
      episodios[lab] += 1
      dentro = np.where((t >= a) & (t <= b))[0]
      if len(dentro) < 2:
        continue
      S, tt = r4b.sinal(v["P"][dentro].astype(np.float64), lab), t[dentro]
      teve = {j: False for j in JANELAS}
      for seg in np.split(np.arange(len(tt)), np.where(np.diff(tt) > 0.5 + 1 / v["fps"])[0] + 1):
        grade = np.arange(tt[seg[0]], tt[seg[-1]] + 1e-9, 1 / FS)
        Sg = np.stack([np.interp(grade, tt[seg], S[seg, k]) for k in range(S.shape[1])], 1)
        for j in JANELAS:
          for ini in range(0, len(grade) - j * FS + 1, FS):
            dados[(lab, j)]["freq"].append(dominante(Sg[ini:ini + j * FS]))
            teve[j] = True
      for j in JANELAS:
        if teve[j]:
          dados[(lab, j)]["episodios_com_janela"] += 1
          dados[(lab, j)]["videos"].add(vid)
  est = lambda x: {"mediana": float(np.median(x)), "p10": float(np.percentile(x, 10)), "p90": float(np.percentile(x, 90))} if len(x) else None
  out = {"banda_hz": list(BANDA), "tendencia": "linear removida em cada janela", "passo_s": 1, "classes": {}}
  for (c, j), d in dados.items():
    f = np.array(d["freq"])
    x = {"n_janelas": len(f), "episodios_com_janela": d["episodios_com_janela"], "episodios": episodios[c], "videos": len(d["videos"]),
         "frequencia_hz": est(f), "ciclos_em_2s": est(np.floor(f * 2)), "ciclos_em_4s": est(np.floor(f * 4))}
    if c == "Spinning":  # a largura dos ombros oscila duas vezes por volta
      x["rotacao_hz"] = est(f / 2)
      x["voltas_em_2s"], x["voltas_em_4s"] = est(np.floor(f / 2 * 2)), est(np.floor(f / 2 * 4))
    out["classes"].setdefault(c, {})[f"janela_{j}s"] = x
  SAIDA.mkdir(exist_ok=True)
  r.gravar_json(SAIDA / "frequencia.json", out)
  f3 = q.f3
  e = lambda x: f"{f3(x['mediana'])} [{f3(x['p10'])}; {f3(x['p90'])}]" if x else "--"
  linhas = ["# Frequência dominante nos intervalos anotados (parte B refeita)", "",
            "Tendência linear removida em cada janela; faixa de 0,5 a 7 Hz; janelas de 8 s e de 4 s com passo de 1 s; mediana [p10; p90].", "",
            "| classe | janela | frequência (Hz) | ciclos em 2 s | ciclos em 4 s | janelas | episódios com janela | vídeos |", "|---|---|---|---|---|---|---|---|"]
  for c, js in out["classes"].items():
    for jn, x in js.items():
      linhas.append(f"| *{c}* | {jn.split('_')[1]} | {e(x['frequencia_hz'])} | {e(x['ciclos_em_2s'])} | {e(x['ciclos_em_4s'])} | {x['n_janelas']} | "
                    f"{x['episodios_com_janela']} de {x['episodios']} | {x['videos']} |")
  for jn, x in out["classes"]["Spinning"].items():
    linhas.append(f"\n*Spinning*, janela de {jn.split('_')[1]}: rotação {e(x['rotacao_hz'])} Hz; voltas em 2 s {e(x['voltas_em_2s'])}, em 4 s {e(x['voltas_em_4s'])}.")
  (SAIDA / "frequencia.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")
  print("\n".join(linhas))


if __name__ == "__main__":
  main()
