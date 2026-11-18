# ============================================================
# ARQUIVO: experimentos/treinar_final_completo.py
#
# O QUE FAZ: treina, só para a demonstração ao vivo da câmera, uma versão
# do modelo final (r4b-C1-janela-4s) com os 56 vídeos: mesma configuração,
# semente 42, número de épocas igual à mediana das épocas escolhidas na
# validação das 5 partições do modelo final. Não entra em nenhuma métrica.
# Grava models/final_completo.pt (fora do git) e
# experimentos/modelo-final-completo.json (registro).
#
# Uso (da raiz do repositório):
#   .venv/Scripts/python.exe experimentos/treinar_final_completo.py
# ============================================================

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(RAIZ / "experimentos"), str(RAIZ / "src")]
import rodar_q as rq  # noqa: E402  (faz chdir para a raiz; aplica as mesmas substituições usadas no treino)

r, q = rq.r, rq.q
FINAL = RAIZ / "experimentos" / "r4b-C1-janela-4s"
SAIDA_PT = RAIZ / "models" / "final_completo.pt"


def main():
  import torch
  import torch.nn as nn
  torch.set_num_threads(4)
  r.OPC.update({"saida": str(RAIZ / "experimentos"), "teste": False})
  cfg = {**q.NOVOS, **rq.cfg_de(FINAL)}
  epocas = [r.ler_json(FINAL / f"particao_{k}.json")["epoca_escolhida"] for k in range(1, 6)]
  n_epocas = int(round(statistics.median(epocas)))
  s = cfg["semente"]
  t0 = time.time()
  videos = r.videos()
  cache = {v: r.preparar_video(v, cfg) for v in videos}
  A, ini, rot, donos = r.conjunto(videos, cfg, 4, cache)
  ini, rot = r.subamostrar(ini, rot, donos, s)
  A = torch.from_numpy(A)
  off = torch.arange(cfg["janela"])
  torch.manual_seed(s)
  modelo = q.rede(cfg)
  opt = torch.optim.Adam(modelo.parameters(), lr=cfg["lr"], weight_decay=cfg["peso_l2"])
  crit = nn.CrossEntropyLoss()
  gerador, g_aum = torch.Generator().manual_seed(s), torch.Generator().manual_seed(s + 2000)
  y = torch.from_numpy(rot)
  perdas = []
  for ep in range(1, n_epocas + 1):
    modelo.train()
    ordem = torch.randperm(len(ini), generator=gerador).numpy()
    soma, nb = 0.0, 0
    for b in range(0, len(ordem), cfg["lote"]):
      j = ordem[b:b + cfg["lote"]]
      Xb = A[torch.from_numpy(ini[j])[:, None] + off]
      if cfg["aumento"]:
        Xb = q.aumentar(Xb, g_aum, cfg)
      opt.zero_grad()
      perda = crit(modelo(q.transformar(Xb, cfg)), y[j])
      perda.backward()
      opt.step()
      soma += perda.item()
      nb += 1
    perdas.append(soma / nb)
    print(f"época {ep}: perda de treino {soma / nb:.4f}", flush=True)
  torch.save(modelo.state_dict(), SAIDA_PT)
  r.gravar_json(RAIZ / "experimentos" / "modelo-final-completo.json", {
    "uso": "só a demonstração ao vivo da câmera; não entra em nenhuma métrica",
    "base": "r4b-C1-janela-4s", "configuracao": cfg, "videos": len(videos), "janelas_treino": int(len(ini)),
    "janelas_por_classe": {r.CLASSES[c]: int((rot == c).sum()) for c in range(4)},
    "epocas_escolhidas_na_validacao": epocas, "epocas": n_epocas, "perdas_treino": perdas,
    "sementes": {"inicializacao": s, "ordem_dos_lotes": s, "subamostragem_fundo": s, "aumento": s + 2000},
    "arquivo": "models/final_completo.pt (fora do git)", "tempo_s": time.time() - t0})
  print(f"gravado {SAIDA_PT} em {time.time() - t0:.0f} s")


if __name__ == "__main__":
  main()
