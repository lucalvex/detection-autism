# ============================================================
# ARQUIVO: experimentos/rodar_rodada2.py
#
# O QUE FAZ: rodada 2 dos experimentos (experimentos/protocolo_rodada2.md),
# sozinha e em sequência: A (entrada só com movimento), B (periodicidade
# com classificador simples), C (sobreajuste), D (pós-processamento), a
# avaliação final e E (curva de quantidade de dados). Retomável: pula o
# que já tem metricas_validacao.json; erro de um experimento vai para
# r2_<id>/erro.txt e o script segue. Um commit por experimento, sem push.
# Reaproveita as funções de rodar_tudo.py (importado, não alterado) e só
# lê os arquivos da rodada 1.
#
# Uso (da raiz do repositório):
#   .venv/Scripts/python.exe experimentos/rodar_rodada2.py
#   .venv/Scripts/python.exe experimentos/rodar_rodada2.py --teste --saida <pasta>
# ============================================================

import argparse
import copy
import json
import math
import multiprocessing as mp
import os
import pickle
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(RAIZ / "experimentos"), str(RAIZ / "src")]
import rodar_tudo as r  # noqa: E402  (faz chdir para a raiz)

EXP1 = RAIZ / "experimentos"
r.OPC.update({"saida": str(EXP1), "teste": False})  # poses e divisão de validação da rodada 1
O = {"saida": str(EXP1), "teste": False}
METRICAS = r.METRICAS
R_ID = "06_15fps_janela_2s"
R_SEMENTES = {42: R_ID, 1000: "09_final_semente_1000", 2000: "09_final_semente_2000"}
NOVOS = {"modelo": "lstm", "entrada": "xy_dxdy", "dropout": 0.0, "peso_l2": 0.0, "aumento": False}
PARES = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
ANGULOS = [(5, 7, 9), (6, 8, 10), (7, 5, 11), (8, 6, 12), (11, 13, 15), (12, 14, 16)]
GRADE_B = {"logreg": [{"C": c} for c in (0.01, 0.1, 1, 10)],
           "gb": [{"learning_rate": a, "max_leaf_nodes": f} for a in (0.05, 0.1) for f in (15, 63)]}


def pasta(i):
  return (EXP1 if not i.startswith("r2_") else Path(O["saida"])) / i


def log(msg):
  linha = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
  print(linha, flush=True)
  with open(Path(O["saida"]) / "progresso_rodada2.log", "a", encoding="utf-8") as f:
    f.write(linha + "\n")


def commit(msg, arquivos):
  if O["teste"]:
    return
  arquivos = [str(Path(a).relative_to(RAIZ)) for a in arquivos if Path(a).exists()]
  if not arquivos:
    return
  subprocess.run(["git", "add", "-f", "--", *arquivos], check=True, capture_output=True)
  res = subprocess.run(["git", "commit", "-m", f"{msg}\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>", "--", *arquivos],
                       capture_output=True, text=True)
  log(f"commit: {msg}" if res.returncode == 0 else f"COMMIT FALHOU: {msg}: {(res.stdout + res.stderr).strip()[-300:]}")


def opcoes():
  return dict(O), dict(r.OPC)


def aplicar_opcoes(o, ro):
  O.update(o)
  r.OPC.update(ro)


def validacao():
  div = r.divisao_validacao()
  return {k: [v for v in div[k] if v in r.videos()] for k in r.particoes()}


# ---------- entrada do LSTM ----------

def n_entrada(cfg):
  return {"xy_dxdy": 68, "dxdy": 34, "dxdy_ang": 48}[cfg["entrada"]]


def angulos(p):
  import torch
  def ang(u, v):
    return torch.atan2(u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0], (u * v).sum(-1))
  out = [ang(p[..., a, :] - p[..., b, :], p[..., c, :] - p[..., b, :]) for a, b, c in ANGULOS]
  ombros, quadris = (p[..., 5, :] + p[..., 6, :]) / 2, (p[..., 11, :] + p[..., 12, :]) / 2
  out.append(ang(p[..., 0, :] - ombros, ombros - quadris))
  return torch.stack(out, -1)


def transformar(X, cfg):
  """X: (lote, W, 68) com x, y, dx, dy por ponto -> entrada do experimento."""
  import torch
  if cfg["entrada"] == "xy_dxdy":
    return X
  B, W = X.shape[:2]
  Y = X.reshape(B, W, 17, 4)
  d = Y[..., 2:].reshape(B, W, 34)
  if cfg["entrada"] == "dxdy":
    return d
  a = angulos(Y[..., :2])
  vel = torch.zeros_like(a)
  vel[:, 1:] = torch.remainder(a[:, 1:] - a[:, :-1] + math.pi, 2 * math.pi) - math.pi
  acel = torch.zeros_like(a)
  acel[:, 2:] = vel[:, 2:] - vel[:, 1:-1]
  return torch.cat([d, vel, acel], 2)


def aumentar(X, g, cfg):
  """Espelhamento, escala, rotação e velocidade, sorteados por janela."""
  import torch
  B, W = X.shape[:2]
  Y = X.reshape(B, W, 17, 4)
  p, d = Y[..., :2], Y[..., 2:]
  s = 0.8 + 0.4 * torch.rand(B, generator=g)
  c = (W - 1) / 2
  tau = (c + (torch.arange(W)[None, :] - c) * s[:, None]).clamp(0, W - 1)
  i0 = tau.floor().long()
  i1 = (i0 + 1).clamp(max=W - 1)
  w = (tau - i0)[..., None, None]
  ix = lambda i: i[:, :, None, None].expand(B, W, 17, 2)
  p = torch.gather(p, 1, ix(i0)) * (1 - w) + torch.gather(p, 1, ix(i1)) * w
  d2 = torch.zeros_like(d)
  d2[:, 1:] = p[:, 1:] - p[:, :-1]
  d2[:, 0] = d[:, 0] * s[:, None, None]
  d = d2
  perm = list(range(17))
  for a, b in PARES:
    perm[a], perm[b] = b, a
  sinal = torch.tensor([-1.0, 1.0])
  esp = (torch.rand(B, generator=g) < 0.5)[:, None, None, None]
  p = torch.where(esp, p[:, :, perm] * sinal, p)
  d = torch.where(esp, d[:, :, perm] * sinal, d)
  e = (0.9 + 0.2 * torch.rand(B, generator=g))[:, None, None]
  th = (torch.rand(B, generator=g) * 20 - 10) * math.pi / 180
  cs, sn = torch.cos(th)[:, None, None], torch.sin(th)[:, None, None]
  rot = lambda q: torch.stack([q[..., 0] * cs - q[..., 1] * sn, q[..., 0] * sn + q[..., 1] * cs], -1)
  p, d = rot(p) * e[..., None], rot(d) * e[..., None]
  if cfg["normalizacao"] == "robusta":
    p = p.clamp(-r.LIMITE_NORM, r.LIMITE_NORM)
  return torch.cat([p, d], -1).reshape(B, W, 68)


def rede(cfg, n_classes=4):
  import torch.nn as nn

  class Rede(nn.Module):
    # mesmos nomes de LSTMClassifier (lstm, fc): carrega os modelos da rodada 1
    def __init__(self):
      super().__init__()
      self.lstm = nn.LSTM(n_entrada(cfg), cfg["oculto"], cfg["camadas"], batch_first=True,
                          dropout=cfg["dropout"] if cfg["camadas"] > 1 else 0.0)
      self.drop = nn.Dropout(cfg["dropout"])
      self.fc = nn.Linear(cfg["oculto"], n_classes)

    def forward(self, x):
      _, (h, _) = self.lstm(x)
      return self.fc(self.drop(h[-1]))
  return Rede()


def ablar(X, ablacao):
  if ablacao == "ordem":
    perm = np.argsort(np.random.default_rng(42).random(X.shape[:2]), axis=1)
    return np.take_along_axis(X, perm[..., None], axis=1)
  if ablacao == "dxdy":
    X = X.copy().reshape(*X.shape[:2], 17, 4)
    X[..., 2:] = 0
    return X.reshape(*X.shape[:2], 68)
  return X


def inferir_lstm(modelo, cache, vids, cfg, ablacao=None):
  import torch
  out = {}
  for vid in vids:
    v = cache.get(vid) or r.preparar_video(vid, cfg)
    j = r.janelas_avaliacao(v, cfg) if v else None
    if j is None:
      continue
    X, ini, fim = j
    X = ablar(X, ablacao).astype(np.float32)
    with torch.no_grad():
      p = np.concatenate([torch.softmax(modelo(transformar(torch.from_numpy(X[b:b + 2048]), cfg)), 1).numpy()
                          for b in range(0, len(X), 2048)])
    out[vid] = {"fps": v["fps"], "quadro_inicio": ini, "quadro_fim": fim, "pontuacoes": p}
  return out


def treinar_lstm(args):
  exp_dir, cfg, k, val_videos, com_teste, treino, opc = args
  aplicar_opcoes(*opc)
  try:
    import torch
    import torch.nn as nn
    torch.set_num_threads(1)
    exp_dir = Path(exp_dir)
    feito = exp_dir / f"particao_{k}.json"
    if feito.exists():
      return k, None
    t0 = time.time()
    treino = treino or [v for v in r.videos() if r.PARTICOES[v] != k and v not in val_videos]
    teste = [v for v in r.videos() if r.PARTICOES[v] == k]
    cache = {v: r.preparar_video(v, cfg) for v in treino + val_videos + (teste if com_teste else [])}
    W, s = cfg["janela"], cfg["semente"]
    A_tr, i_tr, r_tr, d_tr = r.conjunto(treino, cfg, 4, cache)
    A_va, i_va, r_va, d_va = r.conjunto(val_videos, cfg, 4, cache)
    i_tr, r_tr = r.subamostrar(i_tr, r_tr, d_tr, s + k)
    i_va, r_va = r.subamostrar(i_va, r_va, d_va, s + k + 1000)
    if len(i_tr) == 0 or len(i_va) == 0:
      raise RuntimeError(f"partição {k}: {len(i_tr)} janelas de treino e {len(i_va)} de validação")
    A_tr, A_va = torch.from_numpy(A_tr), torch.from_numpy(A_va)
    off = torch.arange(W)
    lote = lambda A, ini: A[torch.from_numpy(ini)[:, None] + off]
    torch.manual_seed(s + k)
    modelo = rede(cfg)
    opt = torch.optim.Adam(modelo.parameters(), lr=cfg["lr"], weight_decay=cfg["peso_l2"])
    crit = nn.CrossEntropyLoss()
    gerador = torch.Generator().manual_seed(s + k)
    g_aum = torch.Generator().manual_seed(s + k + 2000)
    y_tr, y_va = torch.from_numpy(r_tr), torch.from_numpy(r_va)
    perdas, melhor, melhor_ep, estado, parado = [], float("inf"), 0, None, 0
    for ep in range(1, cfg["max_epocas"] + 1):
      modelo.train()
      ordem = torch.randperm(len(i_tr), generator=gerador).numpy()
      soma, nb = 0.0, 0
      for b in range(0, len(ordem), cfg["lote"]):
        j = ordem[b:b + cfg["lote"]]
        Xb = lote(A_tr, i_tr[j])
        if cfg["aumento"]:
          Xb = aumentar(Xb, g_aum, cfg)
        opt.zero_grad()
        perda = crit(modelo(transformar(Xb, cfg)), y_tr[j])
        perda.backward()
        opt.step()
        soma += perda.item()
        nb += 1
      modelo.eval()
      with torch.no_grad():
        pv = sum(crit(modelo(transformar(lote(A_va, i_va[b:b + 2048]), cfg)), y_va[b:b + 2048]).item() * len(i_va[b:b + 2048])
                 for b in range(0, len(i_va), 2048)) / len(i_va)
      perdas.append([ep, soma / nb, pv])
      if pv < melhor:
        melhor, melhor_ep, estado, parado = pv, ep, copy.deepcopy(modelo.state_dict()), 0
      else:
        parado += 1
      if parado >= cfg["paciencia"]:
        break
    modelo.load_state_dict(estado)
    modelo.eval()
    (exp_dir / "modelos").mkdir(exist_ok=True)
    torch.save(estado, exp_dir / "modelos" / f"particao_{k}.pt")
    saida = {"validacao": inferir_lstm(modelo, cache, val_videos, cfg)}
    if com_teste:
      saida["teste"] = inferir_lstm(modelo, cache, teste, cfg)
    with open(exp_dir / f"janelas_{k}.pkl", "wb") as f:
      pickle.dump(saida, f)
    r.gravar_json(feito, {"particao": k, "epoca_escolhida": melhor_ep, "epocas_rodadas": len(perdas), "perda_validacao_minima": melhor,
                          "janelas_treino": int(len(i_tr)), "janelas_validacao": int(len(i_va)), "videos_treino": treino,
                          "videos_validacao": val_videos, "perdas": perdas, "tempo_s": time.time() - t0})
    return k, None
  except Exception:
    return k, traceback.format_exc()


# ---------- B: periodicidade ----------

def sinais(Xw):
  n, W = Xw.shape[:2]
  P = Xw.reshape(n, W, 17, 4)[..., :2]
  s = [P[..., 9, 0] - P[..., 5, 0], P[..., 9, 1] - P[..., 5, 1], P[..., 10, 0] - P[..., 6, 0], P[..., 10, 1] - P[..., 6, 1],
       P[..., 0, 1] - (P[..., 5, 1] + P[..., 6, 1]) / 2, np.linalg.norm(P[..., 6, :] - P[..., 5, :], axis=-1)]
  return np.stack(s, 1).astype(np.float64)  # (n, 6, W)


def atributos_periodicidade(Xw, fs):
  S = sinais(Xw)
  W = S.shape[-1]
  Z = S - S.mean(-1, keepdims=True)
  esp = np.abs(np.fft.rfft(Z * np.hanning(W), axis=-1)) ** 2
  f = np.fft.rfftfreq(W, 1 / fs)
  banda = (f >= 0.5) & (f <= 6)
  Pb, fb = esp[..., banda], f[banda]
  j = Pb.argmax(-1)
  pot = np.take_along_axis(Pb, j[..., None], -1)[..., 0]
  q = Pb / np.maximum(Pb.sum(-1, keepdims=True), 1e-12)
  ent = -(q * np.log(np.maximum(q, 1e-12))).sum(-1) / np.log(Pb.shape[-1])
  ac = np.fft.irfft(np.abs(np.fft.rfft(Z, n=2 * W, axis=-1)) ** 2, axis=-1)[..., :W]
  ac = ac / np.maximum(ac[..., :1], 1e-12)
  pico = ac[..., int(np.ceil(fs / 6)):min(W - 1, int(fs / 0.5)) + 1].max(-1)
  return np.concatenate([fb[j], pot, ent, pico, S.std(-1), np.abs(np.diff(S, axis=-1)).mean(-1)], 1).astype(np.float32)


def atributos_das_janelas(A, inicios, W, fs):
  return np.concatenate([atributos_periodicidade(A[inicios[b:b + 8192, None] + np.arange(W)], fs)
                         for b in range(0, len(inicios), 8192)]) if len(inicios) else np.zeros((0, 36), np.float32)


def classificador(cfg, h):
  from sklearn.ensemble import HistGradientBoostingClassifier
  from sklearn.linear_model import LogisticRegression
  from sklearn.pipeline import make_pipeline
  from sklearn.preprocessing import StandardScaler
  if cfg["classificador"] == "logreg":
    return make_pipeline(StandardScaler(), LogisticRegression(C=h["C"], class_weight="balanced", max_iter=2000))
  return HistGradientBoostingClassifier(max_iter=300, class_weight="balanced", random_state=cfg["semente"], **h)


def inferir_b(clf, cache, vids, cfg, ablacao=None):
  out = {}
  for vid in vids:
    v = cache.get(vid) or r.preparar_video(vid, cfg)
    j = r.janelas_avaliacao(v, cfg) if v else None
    if j is None:
      continue
    X, ini, fim = j
    X = ablar(X, ablacao)
    F = np.concatenate([atributos_periodicidade(X[b:b + 8192], cfg["fps_alvo"]) for b in range(0, len(X), 8192)])
    p = np.zeros((len(F), 4), np.float32)
    p[:, clf.classes_] = clf.predict_proba(F)
    out[vid] = {"fps": v["fps"], "quadro_inicio": ini, "quadro_fim": fim, "pontuacoes": p}
  return out


def treinar_b(args):
  exp_dir, cfg, k, val_videos, com_teste, treino, opc = args
  aplicar_opcoes(*opc)
  try:
    exp_dir = Path(exp_dir)
    feito = exp_dir / f"particao_{k}.json"
    if feito.exists():
      return k, None
    t0 = time.time()
    treino = treino or [v for v in r.videos() if r.PARTICOES[v] != k and v not in val_videos]
    cache = {v: r.preparar_video(v, cfg) for v in treino + val_videos}
    A, ini, rot, _ = r.conjunto(treino, cfg, 4, cache)
    F = atributos_das_janelas(A, ini, cfg["janela"], cfg["fps_alvo"])
    grade = [cfg["hiper"]] if cfg.get("hiper") else GRADE_B[cfg["classificador"]]
    saida, modelos = {}, {}
    for h in grade:
      clf = classificador(cfg, h).fit(F, rot)
      nome = json.dumps(h, sort_keys=True)
      saida[nome] = {"validacao": inferir_b(clf, cache, val_videos, cfg)}
      modelos[nome] = clf
    (exp_dir / "modelos").mkdir(exist_ok=True)
    with open(exp_dir / "modelos" / f"particao_{k}.pkl", "wb") as f:
      pickle.dump(modelos, f)
    with open(exp_dir / f"janelas_grade_{k}.pkl", "wb") as f:
      pickle.dump(saida, f)
    r.gravar_json(feito, {"particao": k, "janelas_treino": int(len(ini)), "videos_treino": treino, "tempo_s": time.time() - t0})
    return k, None
  except Exception:
    return k, traceback.format_exc()


def carregar_modelo(exp_dir, cfg, k):
  if cfg["modelo"] == "periodicidade":
    with open(Path(exp_dir) / "modelos" / f"particao_{k}.pkl", "rb") as f:
      return pickle.load(f)[json.dumps(cfg["hiper"], sort_keys=True)]
  import torch
  m = rede(cfg)
  m.load_state_dict(torch.load(Path(exp_dir) / "modelos" / f"particao_{k}.pt"))
  m.eval()
  return m


def prever(args):
  """Predições de teste de uma partição a partir do modelo salvo, com ou sem ablação."""
  exp_dir, cfg, k, ablacao, opc = args
  aplicar_opcoes(*opc)
  try:
    import torch
    torch.set_num_threads(1)
    teste = [v for v in r.videos() if r.PARTICOES[v] == k]
    m = carregar_modelo(exp_dir, cfg, k)
    f = inferir_b if cfg["modelo"] == "periodicidade" else inferir_lstm
    return k, f(m, {}, teste, cfg, ablacao), None
  except Exception:
    return k, None, traceback.format_exc()


# ---------- métricas nas duas convenções ----------

def met_val(vd):
  ids = sorted(vd)
  pp = {k: r.tres_metricas(vd, [i for i in ids if vd[i]["fold"] == k]) for k in r.particoes()}
  return {"agregado_validacao": r.tres_metricas(vd, ids), "por_particao": pp,
          "media_entre_particoes": {m: float(np.mean([x[m] for x in pp.values()])) for m in METRICAS},
          "desvio_entre_particoes": {m: float(np.std([x[m] for x in pp.values()])) for m in METRICAS}}


def met_val_lidas(exp_id):
  m = r.ler_json(pasta(exp_id) / "metricas_validacao.json")
  if "media_entre_particoes" not in m:  # rodada 1: só agregado e por partição
    pp = {int(k): v for k, v in m["por_particao"].items() if int(k) in r.particoes()}
    m["media_entre_particoes"] = {x: float(np.mean([v[x] for v in pp.values()])) for x in METRICAS}
    m["desvio_entre_particoes"] = {x: float(np.std([v[x] for v in pp.values()])) for x in METRICAS}
  return m


def vd_de_resultados(res, cfg):
  """Mesma estrutura de rodar_tudo.dados_avaliacao, a partir de {partição: {vídeo: ...}}."""
  import evaluate_temporal as et
  et.class_names, et.id_to_label, et.fold_assignments = r.CLASSES, dict(enumerate(r.CLASSES)), r.PARTICOES
  vd = {}
  for k, por_video in res.items():
    for vid, x in por_video.items():
      jan = [{"quadro_inicio": int(a), "quadro_fim": int(b), "pontuacoes": p.tolist()}
             for a, b, p in zip(x["quadro_inicio"], x["quadro_fim"], x["pontuacoes"])]
      gt, iv = et.ground_truth_frames(vid, jan[-1]["quadro_fim"], x["fps"])
      vd[vid] = {"fold": k, "fps": x["fps"], "n_frames": jan[-1]["quadro_fim"], "windows": jan, "gt_frames": gt, "gt_intervals": iv}
  return vd


def f1_anotados_3_classes(vd):
  import evaluate_temporal as et
  forc = {v: {**x, "windows": [{**w, "pontuacoes": w["pontuacoes"][:3] + [-1.0]} for w in x["windows"]]} for v, x in vd.items()}
  res = et.frame_level_with_folds(forc, sorted(forc), True)
  return {"agregado": res["pooled_56_videos"]["macro_f1_sem_background"], "media": res["macro_f1_sem_background_media"],
          "desvio": res["macro_f1_sem_background_desvio"]}


def fracao_classe_do_video(vd):
  import evaluate_temporal as et
  out = {}
  for c in r.CLASSES[:3]:
    dentro, fora = [], []
    for vid, x in vd.items():
      if r.classe_do_video(vid) != c:
        continue
      pred = et.predicted_frames(x["windows"])
      a = [p == c for q, p in pred.items() if x["gt_frames"][q] != "Background"]
      b = [p == c for q, p in pred.items() if x["gt_frames"][q] == "Background"]
      dentro.append(np.mean(a) if a else np.nan)
      fora.append(np.mean(b) if b else np.nan)
    out[c] = {"n_videos": len(dentro), "dentro": float(np.nanmean(dentro)), "fora": float(np.nanmean(fora))}
  return out


def f3(x):
  return "--" if x is None else f"{x:.3f}".replace(".", ",")


def duas(m, x):
  return f"{f3(m['agregado_validacao' if 'agregado_validacao' in m else 'agregado_56_videos'][x])} / " \
         f"{f3(m['media_entre_particoes'][x])} ± {f3(m['desvio_entre_particoes'][x])}"


# ---------- execução de um experimento ----------

def arquivos(exp_dir):
  return [exp_dir / n for n in ("config.json", "seeds.json", "metricas_validacao.json", "metricas_teste.json", "perdas.csv",
                                "curva_erro.pdf", "hiperparametros.json", "erro.txt")]


def rodar(exp_id, cfg, descricao, com_teste=False, treino_por_particao=None):
  """Treina (ou retoma) e devolve as métricas de validação, ou None se falhar."""
  exp_dir = pasta(exp_id)
  feito = exp_dir / "metricas_validacao.json"
  if feito.exists():
    return r.ler_json(feito)
  exp_dir.mkdir(parents=True, exist_ok=True)
  (exp_dir / "erro.txt").unlink(missing_ok=True)
  log(f"{exp_id}: início ({descricao})")
  r.gravar_json(exp_dir / "config.json", {"descricao": descricao, **cfg, "teste_calculado": com_teste})
  s = cfg["semente"]
  r.gravar_json(exp_dir / "seeds.json", {
    "semente": s, "inicializacao_e_ordem_dos_lotes_por_particao": {k: s + k for k in r.particoes()},
    "subamostragem_fundo_treino_por_particao": {k: s + k for k in r.particoes()},
    "subamostragem_fundo_validacao_por_particao": {k: s + k + 1000 for k in r.particoes()},
    "aumento_de_dados_por_particao": {k: s + k + 2000 for k in r.particoes()},
    "divisao_validacao": "experimentos/divisao_validacao.json (a mesma da rodada 1)"})
  t0 = time.time()
  try:
    val = validacao()
    lstm = cfg["modelo"] == "lstm"
    tarefas = [(str(exp_dir), cfg, k, val[k], com_teste and lstm, (treino_por_particao or {}).get(k), opcoes()) for k in r.particoes()]
    with mp.get_context("spawn").Pool(len(tarefas)) as pool:
      erros = [e for _, e in pool.map(treinar_lstm if lstm else treinar_b, tarefas) if e]
    if erros:
      raise RuntimeError("\n\n".join(erros))
    if not lstm:
      escolher_hiper(exp_dir, cfg)
      cfg = r.ler_json(exp_dir / "config.json")
    met = met_val(r.dados_avaliacao(exp_dir, "validacao", cfg))
    if lstm:
      info = [r.ler_json(exp_dir / f"particao_{k}.json") for k in r.particoes()]
      met["epoca_escolhida_por_particao"] = {i["particao"]: i["epoca_escolhida"] for i in info}
      r.gravar_perdas(exp_dir, info)
      r.curva_erro(exp_dir, info)
    if com_teste and lstm:
      r.gravar_json(exp_dir / "metricas_teste.json", r.metricas_teste(r.dados_avaliacao(exp_dir, "teste", cfg)))
    elif com_teste:
      testar(exp_id, cfg)
    met["tempo_s"] = time.time() - t0
    r.gravar_json(feito, met)
    log(f"{exp_id}: fim em {met['tempo_s'] / 60:.0f} min | " + " ".join(f"{m}={met['agregado_validacao'][m]:.3f}" for m in METRICAS))
    return met
  except Exception:
    (exp_dir / "erro.txt").write_text(traceback.format_exc(), encoding="utf-8")
    log(f"{exp_id}: ERRO (ver {exp_id}/erro.txt)")
    return None


def escolher_hiper(exp_dir, cfg):
  """B: escolhe os hiperparâmetros pela média das 3 métricas de validação (agregado)."""
  grades = {}
  for k in r.particoes():
    with open(exp_dir / f"janelas_grade_{k}.pkl", "rb") as f:
      for nome, x in pickle.load(f).items():
        grades.setdefault(nome, {})[k] = x["validacao"]
  tab = {}
  for nome, res in grades.items():
    m = met_val(vd_de_resultados(res, cfg))
    tab[nome] = {**m, "media_das_3": float(np.mean([m["agregado_validacao"][x] for x in METRICAS]))}
  melhor = max(tab, key=lambda n: tab[n]["media_das_3"])
  r.gravar_json(exp_dir / "hiperparametros.json", {"escolhido": json.loads(melhor), "criterio": "maior média das 3 métricas de validação (agregado)",
                                                   "grade": {n: {"media_das_3": t["media_das_3"], "agregado_validacao": t["agregado_validacao"]} for n, t in tab.items()}})
  for k in r.particoes():
    with open(exp_dir / f"janelas_{k}.pkl", "wb") as f:
      pickle.dump({"validacao": grades[melhor][k]}, f)
  c = r.ler_json(exp_dir / "config.json")
  c["hiper"] = json.loads(melhor)
  r.gravar_json(exp_dir / "config.json", c)


def testar(exp_id, cfg):
  """Predições de teste a partir dos modelos salvos; grava metricas_teste.json."""
  exp_dir = pasta(exp_id)
  if (exp_dir / "metricas_teste.json").exists():
    return r.ler_json(exp_dir / "metricas_teste.json")
  with mp.get_context("spawn").Pool(len(r.particoes())) as pool:
    res = pool.map(prever, [(str(exp_dir), cfg, k, None, opcoes()) for k in r.particoes()])
  erros = [e for _, _, e in res if e]
  if erros:
    raise RuntimeError("\n\n".join(erros))
  for k, x, _ in res:
    p = exp_dir / f"janelas_{k}.pkl"
    with open(p, "rb") as f:
      saida = pickle.load(f)
    saida["teste"] = x
    with open(p, "wb") as f:
      pickle.dump(saida, f)
  t = r.metricas_teste(r.dados_avaliacao(exp_dir, "teste", cfg))
  r.gravar_json(exp_dir / "metricas_teste.json", t)
  return t


def candidatas(hist, base_id, base_cfg, base_met, opcoes_etapa):
  """Roda as variantes de uma etapa contra o melhor atual; devolve o novo melhor."""
  passou = []
  for exp_id, mudanca, desc in opcoes_etapa:
    cfg = {**base_cfg, **mudanca}
    met = rodar(exp_id, cfg, desc)
    if met is None:
      hist.append({"id": exp_id, "descricao": desc, "situacao": "erro"})
      continue
    ok, soma = r.supera(met["agregado_validacao"], base_met["agregado_validacao"])
    hist.append({"id": exp_id, "descricao": desc, "met": met, "situacao": "candidata", "ok": ok, "soma": soma})
    if ok:
      passou.append((soma, exp_id, cfg, met))
  escolhido = max(passou, key=lambda x: x[0]) if passou else None
  for h in hist:
    if h["situacao"] == "candidata":
      h["situacao"] = "sim" if escolhido and h["id"] == escolhido[1] else "não"
  for exp_id, _, _ in opcoes_etapa:
    escrever_resumo(hist)
    commit(f"experimentos: {exp_id}", arquivos(pasta(exp_id)) + [Path(O["saida"]) / "resumo_rodada2.md", Path(__file__)])
  if escolhido:
    return escolhido[1], escolhido[2], escolhido[3]
  return base_id, base_cfg, base_met


# ---------- D: pós-processamento ----------

def juntar(segs, intervalo_s):
  out = []
  for c in {s["classe"] for s in segs}:
    atual = None
    for s in sorted((s for s in segs if s["classe"] == c), key=lambda s: s["inicio_s"]):
      if atual and s["inicio_s"] - atual["fim_s"] < intervalo_s:
        atual.update(fim_s=max(atual["fim_s"], s["fim_s"]), quadro_fim=max(atual["quadro_fim"], s["quadro_fim"]),
                     pico=max(atual["pico"], s["pico"]))
        atual["duracao_s"] = atual["fim_s"] - atual["inicio_s"]
      else:
        atual = dict(s)
        out.append(atual)
  return out


def f1_evento_pos(vd, mediana_s, juntar_s):
  import evaluate_temporal as et
  from scipy.ndimage import median_filter
  if mediana_s:
    vd = {v: {**x, "windows": [{**w, "pontuacoes": p.tolist()} for w, p in zip(
      x["windows"], median_filter(np.array([w["pontuacoes"] for w in x["windows"]]), size=(int(round(mediana_s * x["fps"])) // 2 * 2 + 1, 1),
                                  mode="nearest"))]} for v, x in vd.items()}
  original = et.build_segments
  if juntar_s:
    et.build_segments = lambda *a, **kw: juntar(original(*a, **kw), juntar_s)
  try:
    ids = sorted(vd)
    ev = et.event_level_with_folds(vd, ids, 0.5, 0.0, [0.3])[0.3]
    return {"agregado": ev["pooled_56_videos"]["macro_f1"], "media": ev["macro_f1_media_entre_folds"], "desvio": ev["macro_f1_desvio_entre_folds"]}
  finally:
    et.build_segments = original


def posprocessamento(final_id, final_cfg):
  p = pasta("r2_D_posprocessamento") / "posprocessamento.json"
  if p.exists():
    return r.ler_json(p)
  log("r2_D_posprocessamento: início")
  t0 = time.time()
  vd = r.dados_avaliacao(pasta(final_id), "validacao", final_cfg)
  import evaluate_temporal as et
  # validação: chaves "partição|vídeo"; o agrupamento por partição precisa delas
  et.fold_assignments = {**r.PARTICOES, **{v: x["fold"] for v, x in vd.items()}}
  grade = {}
  for med in (0, 0.5, 1.0, 2.0):
    for jun in (0, 0.5, 1.0):
      grade[f"mediana_{med}_juncao_{jun}"] = {"mediana_s": med, "juncao_s": jun, "validacao": f1_evento_pos(vd, med, jun)}
  melhor = max(grade, key=lambda n: (grade[n]["validacao"]["agregado"], -grade[n]["mediana_s"] - grade[n]["juncao_s"]))
  out = {"modelo": final_id, "criterio": "maior F1 por evento IoU 0,3 na validação (agregado); empate: o mais simples",
         "nota": "só a construção dos trechos muda; as métricas por quadro usam as pontuações sem filtro",
         "escolhido": melhor, "grade": grade, "tempo_s": time.time() - t0}
  r.gravar_json(p, out)
  log(f"r2_D_posprocessamento: {melhor}, F1 evento validação {grade[melhor]['validacao']['agregado']:.3f}")
  return out


# ---------- avaliação final ----------

def avaliacao_final(final_id, final_cfg, pos, dirs_sementes):
  p = pasta("r2_avaliacao_final") / "avaliacao_final.json"
  if p.exists():
    return r.ler_json(p)
  log("r2_avaliacao_final: início")
  t0 = time.time()
  import evaluate_temporal as et
  from selecionar_videos_mediana import f1_do_video
  ablacoes = [None, "ordem"] + (["dxdy"] if final_cfg["modelo"] == "lstm" else [])  # toda entrada do LSTM tem dx, dy
  tarefas = [(s, a, (str(pasta(d)), final_cfg if s == 42 else {**final_cfg, "semente": s}, k, a, opcoes()))
             for s, d in dirs_sementes.items() for a in ablacoes for k in r.particoes()]
  with mp.get_context("spawn").Pool(5) as pool:
    res = pool.map(prever, [t[2] for t in tarefas])
  erros = [e for _, _, e in res if e]
  if erros:
    raise RuntimeError("\n\n".join(erros))
  por = {}
  for (s, a, _), (k, x, _) in zip(tarefas, res):
    por.setdefault((s, a), {})[k] = x
  out = {"modelo_final": final_id, "configuracao": final_cfg, "sementes": {}, "pos_processamento": pos["escolhido"]}
  for s in dirs_sementes:
    sem = {}
    for a in ablacoes:
      vd = vd_de_resultados(por[(s, a)], final_cfg)
      t = r.metricas_teste(vd)
      sem[a or "sem_ablacao"] = {"metricas": t, "macro_f1_anotados_3_classes": f1_anotados_3_classes(vd),
                                 "fracao_classe_do_video": fracao_classe_do_video(vd)}
      if a is None:
        hb = [f1_do_video(vd[v])[0] for v in vd if r.classe_do_video(v) == "HeadBanging"]
        sem["head_banging_f1_zero"] = {"n_zero": sum(1 for f in hb if f == 0), "n_videos": len(hb)}
        if pos["escolhido"]:
          g = pos["grade"][pos["escolhido"]]
          sem["f1_evento_com_pos_processamento"] = f1_evento_pos(vd, g["mediana_s"], g["juncao_s"])
        if s == 42:
          todos = et.frame_level_with_folds(vd, sorted(vd), False)
          anot = et.frame_level_with_folds(vd, sorted(vd), True)
          tira = lambda rep: {c: {m: rep[c][m] for m in ("precision", "recall", "f1-score", "support")} for c in r.CLASSES}
          out["por_classe"] = {"todos_os_quadros": {"agregado": tira(todos["pooled_56_videos"]["report"]), "media_desvio": todos["por_classe_media_desvio_entre_folds"]},
                               "quadros_anotados": {"agregado": tira(anot["pooled_56_videos"]["report"]), "media_desvio": anot["por_classe_media_desvio_entre_folds"]}}
          out["confusao_quadros_anotados"] = {"labels": anot["pooled_56_videos"]["labels"], "matriz": anot["pooled_56_videos"]["confusion_matrix"]}
    out["sementes"][s] = sem
  ags = [out["sementes"][s]["sem_ablacao"]["metricas"] for s in dirs_sementes]
  out["entre_sementes"] = {conv: {m: {"media": float(np.mean([a[conv][m] for a in ags])), "desvio": float(np.std([a[conv][m] for a in ags]))}
                                  for m in METRICAS + ["macro_f1_quadro_anotados"]} for conv in ("agregado_56_videos", "media_entre_particoes")}
  out["tempo_s"] = time.time() - t0
  r.gravar_json(p, out)
  return out


# ---------- E: curva de dados ----------

def curva_dados(final_id, final_cfg):
  pdir = pasta("r2_E_curva_dados")
  p = pdir / "curva_dados.json"
  if p.exists():
    return r.ler_json(p)
  val = validacao()
  pontos = []
  for frac in (0.25, 0.5, 0.75, 1.0):
    for d in range(3):
      treino = {}
      for k in r.particoes():
        disp = [v for v in r.videos() if r.PARTICOES[v] != k and v not in val[k]]
        rng = np.random.default_rng(1000 * d + k)
        treino[k] = []
        for c in r.CLASSES[:3]:
          vs = sorted(v for v in disp if r.classe_do_video(v) == c)
          if vs:
            treino[k] += [vs[i] for i in rng.permutation(len(vs))[:max(1, round(frac * len(vs)))]]
        treino[k] = sorted(treino[k])
      cfg = {**final_cfg, "semente": final_cfg["semente"] + 1000 * d}
      exp_id = f"r2_E_{int(frac * 100)}pct_sorteio{d}"
      met = rodar(exp_id, cfg, f"curva de dados: {int(frac * 100)}% dos vídeos de treino, sorteio {d}", treino_por_particao=treino)
      commit(f"experimentos: {exp_id}", arquivos(pasta(exp_id)) + [Path(__file__)])
      if met:
        pontos.append({"fracao": frac, "sorteio": d, "n_videos_treino_medio": float(np.mean([len(t) for t in treino.values()])),
                       "agregado_validacao": met["agregado_validacao"], "media_entre_particoes": met["media_entre_particoes"]})
  out = {"modelo": final_id, "pontos": pontos}
  pdir.mkdir(parents=True, exist_ok=True)
  r.gravar_json(p, out)
  figura_curva_dados(pdir, pontos)
  return out


def figura_curva_dados(pdir, pontos):
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  from matplotlib.ticker import FuncFormatter
  from gerar_figura_linha_tempo import DPI, LARGURA_POL, TEXTO_PT, cores_da_interface, virgula
  _, var = cores_da_interface()
  plt.rcParams.update({"font.size": TEXTO_PT, "axes.labelsize": TEXTO_PT, "xtick.labelsize": TEXTO_PT, "ytick.labelsize": TEXTO_PT,
                       "legend.fontsize": TEXTO_PT, "text.color": var["texto"], "axes.labelcolor": var["texto"],
                       "xtick.color": var["texto-2"], "ytick.color": var["texto-2"], "axes.edgecolor": var["linhas"], "pdf.fonttype": 42})
  nomes = {"f1_evento_iou_0.3": "F1 por evento (IoU 0,3)", "acuracia_balanceada_presenca": "acurácia balanceada",
           "macro_f1_quadro_todos": "macro F1 por quadro"}
  fig, eixos = plt.subplots(1, 3, figsize=(LARGURA_POL, 3.6), dpi=DPI)
  fracs = sorted({x["fracao"] for x in pontos})
  for ax, m in zip(eixos, METRICAS):
    xs = [np.mean([x["n_videos_treino_medio"] for x in pontos if x["fracao"] == f]) for f in fracs]
    vals = [[x["agregado_validacao"][m] for x in pontos if x["fracao"] == f] for f in fracs]
    media, desvio = np.array([np.mean(v) for v in vals]), np.array([np.std(v) for v in vals])
    ax.fill_between(xs, media - desvio, media + desvio, color="#C7522A", alpha=0.15, linewidth=0)
    ax.plot(xs, media, color="#C7522A", linewidth=2, marker="o")
    ax.set_title(nomes[m], fontsize=TEXTO_PT)
    ax.set_xlabel("vídeos de treino")
    ax.yaxis.set_major_formatter(FuncFormatter(virgula))
    ax.xaxis.set_major_formatter(FuncFormatter(virgula))
    ax.grid(axis="y", color=var["linhas"], linewidth=0.8)
    for lado in ("top", "right"):
      ax.spines[lado].set_visible(False)
  eixos[0].set_ylabel("validação (agregado)")
  fig.tight_layout()
  fig.savefig(pdir / "curva_dados.pdf")
  plt.close(fig)


# ---------- resumo ----------

def escrever_resumo(hist, final=None):
  linhas = ["# Resumo da rodada 2", "",
            "Protocolo: `protocolo_rodada2.md`. Cada célula: agregado / média ± desvio entre partições (validação). "
            "Decisão pelo agregado: mantém se melhora ≥ 0,01 em pelo menos 2 das 3 métricas.", "",
            "| etapa | mudança | F1 evento IoU 0,3 | acurácia bal. | macro F1 quadros | mantida | tempo |", "|---|---|---|---|---|---|---|"]
  for h in hist:
    m = h.get("met")
    cel = [duas(m, x) if m else "--" for x in METRICAS]
    t = m.get("tempo_s") if m else None
    linhas.append(f"| {h['id']} | {h['descricao']} | {' | '.join(cel)} | {h['situacao']} | {f'{t / 60:.0f} min' if t else '--'} |")
  if final:
    linhas += ["", *final]
  (Path(O["saida"]) / "resumo_rodada2.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")


def bloco_final(final_id, final_cfg, pos, av, curva, nao_rodou):
  cfg_txt = {k: v for k, v in final_cfg.items() if k not in ("descricao", "teste_calculado")}
  tR = r.ler_json(pasta(R_ID) / "metricas_teste.json")
  tcc_a = r.ler_json(RAIZ / "data/results/comparacao_rodada_a/metricas.json")
  bal = r.ler_json(RAIZ / "data/results/analises_posteriores.json")["analise_3_binario_estereotipia_vs_fundo"]["acuracia_balanceada_e_referencia"]["rodada_a"]
  ev = tcc_a["por_evento"]["regra_principal"]["por_iou"]["0.3"]
  todos, anot = tcc_a["por_quadro"]["todos_os_quadros"], tcc_a["por_quadro"]["somente_quadros_anotados"]
  tcc = {"agregado_56_videos": {"f1_evento_iou_0.3": ev["pooled_56_videos"]["macro_f1"], "acuracia_balanceada_presenca": bal["acuracia_balanceada_pooled_56_videos"],
                                "macro_f1_quadro_todos": todos["pooled_56_videos"]["macro_f1_com_background"], "macro_f1_quadro_anotados": anot["pooled_56_videos"]["macro_f1_sem_background"]},
         "media_entre_particoes": {"f1_evento_iou_0.3": ev["macro_f1_media_entre_folds"], "acuracia_balanceada_presenca": bal["acuracia_balanceada_media_entre_folds"],
                                   "macro_f1_quadro_todos": todos["macro_f1_com_background_media"], "macro_f1_quadro_anotados": anot["macro_f1_sem_background_media"]},
         "desvio_entre_particoes": {"f1_evento_iou_0.3": ev["macro_f1_desvio_entre_folds"], "acuracia_balanceada_presenca": bal["acuracia_balanceada_desvio_entre_folds"],
                                    "macro_f1_quadro_todos": todos["macro_f1_com_background_desvio"], "macro_f1_quadro_anotados": anot["macro_f1_sem_background_desvio"]}}
  M4 = METRICAS + ["macro_f1_quadro_anotados"]
  cel = lambda t, m: f"{f3(t['agregado_56_videos'][m])} / {f3(t['media_entre_particoes'][m])} ± {f3(t['desvio_entre_particoes'][m])}"
  s42 = av["sementes"][42] if 42 in av["sementes"] else av["sementes"]["42"]
  linhas = [f"## Modelo final: {final_id}", "", f"Configuração: `{json.dumps(cfg_txt, ensure_ascii=False)}`", "",
            "Teste (56 vídeos), agregado / média ± desvio entre partições:", "",
            "| | F1 evento IoU 0,3 | acurácia bal. | macro F1 quadros | macro F1 anotados |", "|---|---|---|---|---|",
            f"| TCC, rótulos por intervalo | {' | '.join(cel(tcc, m) for m in M4)} |",
            f"| R (rodada 1) | {' | '.join(cel(tR, m) for m in M4)} |",
            f"| modelo final, semente 42 | {' | '.join(cel(s42['sem_ablacao']['metricas'], m) for m in M4)} |", "",
            "Três sementes (média ± desvio entre sementes; agregado | média entre partições): " + "; ".join(
              f"{m} {f3(av['entre_sementes']['agregado_56_videos'][m]['media'])} ± {f3(av['entre_sementes']['agregado_56_videos'][m]['desvio'])} | "
              f"{f3(av['entre_sementes']['media_entre_particoes'][m]['media'])} ± {f3(av['entre_sementes']['media_entre_particoes'][m]['desvio'])}" for m in M4) + ".", ""]
  if pos.get("escolhido"):
    g = pos["grade"][pos["escolhido"]]
    linhas.append(f"Pós-processamento escolhido: mediana {str(g['mediana_s']).replace('.', ',')} s, junção {str(g['juncao_s']).replace('.', ',')} s; F1 por evento no teste (semente 42): "
                  f"{f3(s42['f1_evento_com_pos_processamento']['agregado'])} / {f3(s42['f1_evento_com_pos_processamento']['media'])} ± "
                  f"{f3(s42['f1_evento_com_pos_processamento']['desvio'])}.")
  sem = av["sementes"]
  abl = [k for k in s42 if k in ("sem_ablacao", "ordem", "dxdy")]
  linhas.append("Ablações, macro F1 nos quadros anotados com 3 classes (agregado / média entre partições), sementes 42, 1000 e 2000: " + "; ".join(
    f"{a}: " + ", ".join(f"{f3(sem[s][a]['macro_f1_anotados_3_classes']['agregado'])} / {f3(sem[s][a]['macro_f1_anotados_3_classes']['media'])}" for s in sem)
    for a in abl) + ".")
  linhas.append("Fração da classe do vídeo, dentro / fora dos episódios (semente 42, sem ablação): " + "; ".join(
    f"*{c}* {f3(x['dentro'])} / {f3(x['fora'])}" for c, x in s42["sem_ablacao"]["fracao_classe_do_video"].items()) + ".")
  linhas.append("*Head banging* com F1 zero: " + ", ".join(f"{sem[s]['head_banging_f1_zero']['n_zero']} de {sem[s]['head_banging_f1_zero']['n_videos']}" for s in sem)
                + " (sementes 42, 1000, 2000). Por classe, matriz de confusão e curva de erro: `r2_avaliacao_final/avaliacao_final.json` e "
                + f"`{final_id}/curva_erro.pdf`.")
  if curva and curva.get("pontos"):
    linhas.append("Curva de dados: `r2_E_curva_dados/curva_dados.pdf`.")
  if nao_rodou:
    linhas.append("Não rodou: " + "; ".join(nao_rodou) + ".")
  return linhas


# ---------- sequência ----------

def ja_rodando():
  p = Path(O["saida"]) / "rodando_rodada2.pid"
  if p.exists():
    pid = p.read_text().strip()
    res = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True)
    if pid and pid in res.stdout and "python" in res.stdout.lower():
      return True
  p.write_text(str(os.getpid()))
  return False


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--teste", action="store_true", help="rodada rápida: 1 partição, 2 épocas, 9 vídeos, sem commit")
  ap.add_argument("--saida", default=str(EXP1))
  args = ap.parse_args()
  O.update({"teste": args.teste, "saida": str(Path(args.saida).resolve())})
  Path(O["saida"]).mkdir(parents=True, exist_ok=True)
  if args.teste:
    div1 = r.divisao_validacao()[1]
    esc = []
    for c in r.CLASSES[:3]:
      esc += [next(v for v in div1 if r.classe_do_video(v) == c),
              next(v for v in sorted(r.PARTICOES) if r.classe_do_video(v) == c and r.PARTICOES[v] != 1 and v not in div1),
              next(v for v in sorted(r.PARTICOES) if r.classe_do_video(v) == c and r.PARTICOES[v] == 1)]
    r.OPC.update({"videos": esc, "particoes": [1]})
  if ja_rodando():
    print("Já há uma execução da rodada 2 em andamento.")
    return
  log(f"início (pid {os.getpid()}){' -- TESTE' if args.teste else ''}")
  hist, nao_rodou = [], []
  try:
    cfg_R = {**NOVOS, **{k: v for k, v in r.ler_json(pasta(R_ID) / "config.json").items() if k in r.BASE}}
    if args.teste:
      cfg_R["max_epocas"] = 2
    met_R = met_val_lidas(R_ID)
    hist.append({"id": "R", "descricao": f"referência: {R_ID} (rodada 1)", "met": met_R, "situacao": "referência"})
    melhor_id, melhor_cfg, melhor_met = R_ID, cfg_R, met_R

    # A: entrada só com movimento
    for exp_id, mud, desc in [("r2_A1_dxdy", {"entrada": "dxdy"}, "só dx, dy"),
                              ("r2_A2_dxdy_angulos", {"entrada": "dxdy_ang"}, "dx, dy + velocidade e aceleração angulares")]:
      melhor_id, melhor_cfg, melhor_met = candidatas(hist, melhor_id, melhor_cfg, melhor_met, [(exp_id, mud, desc)])

    # B: periodicidade (linha de base)
    melhor_b = None
    for jan in (30, 60):
      for clf in ("logreg", "gb"):
        exp_id = f"r2_B_{jan // 15}s_{clf}"
        cfg = {**cfg_R, "modelo": "periodicidade", "janela": jan, "classificador": clf}
        met = rodar(exp_id, cfg, f"periodicidade, janela de {jan // 15} s, {'regressão logística' if clf == 'logreg' else 'gradient boosting'}")
        if met is None:
          hist.append({"id": exp_id, "descricao": "periodicidade", "situacao": "erro"})
        else:
          ok, _ = r.supera(met["agregado_validacao"], melhor_met["agregado_validacao"])
          hist.append({"id": exp_id, "descricao": f"periodicidade, {jan // 15} s, {clf}", "met": met,
                       "situacao": f"linha de base ({'superaria' if ok else 'não supera'} o melhor LSTM)"})
          media3 = np.mean([met["agregado_validacao"][m] for m in METRICAS])
          if melhor_b is None or media3 > melhor_b[0]:
            melhor_b = (media3, exp_id, r.ler_json(pasta(exp_id) / "config.json"), met)
        escrever_resumo(hist)
        commit(f"experimentos: {exp_id}", arquivos(pasta(exp_id)) + [Path(O["saida"]) / "resumo_rodada2.md", Path(__file__)])

    # C: sobreajuste, acumulativo
    for etapa in ([("r2_C1_dropout_0.3", {"dropout": 0.3}, "dropout 0,3"), ("r2_C1_dropout_0.5", {"dropout": 0.5}, "dropout 0,5")],
                  [("r2_C2_l2_1e-4", {"peso_l2": 1e-4}, "penalização L2 1e-4"), ("r2_C2_l2_1e-3", {"peso_l2": 1e-3}, "penalização L2 1e-3")],
                  [("r2_C3_1x64", {"camadas": 1, "oculto": 64}, "1 camada de 64 unidades")],
                  [("r2_C4_aumento", {"aumento": True}, "aumento de dados")]):
      melhor_id, melhor_cfg, melhor_met = candidatas(hist, melhor_id, melhor_cfg, melhor_met, etapa)

    final_id, final_cfg = melhor_id, melhor_cfg
    if melhor_b and r.supera(melhor_b[3]["agregado_validacao"], melhor_met["agregado_validacao"])[0]:
      final_id, final_cfg = melhor_b[1], melhor_b[2]
    final_cfg = {k: v for k, v in final_cfg.items() if k not in ("descricao", "teste_calculado")}
    log(f"modelo final: {final_id}")

    # D: pós-processamento
    pos = posprocessamento(final_id, final_cfg)
    commit("experimentos: r2_D_posprocessamento", [pasta("r2_D_posprocessamento") / "posprocessamento.json", Path(__file__)])

    # avaliação final: teste e três sementes
    if final_id == R_ID:
      dirs = dict(R_SEMENTES)
    else:
      testar(final_id, final_cfg)
      dirs = {42: final_id}
      for s in (1000, 2000):
        exp_id = f"r2_final_semente_{s}"
        if rodar(exp_id, {**final_cfg, "semente": s}, f"modelo final, semente {s}", com_teste=True) is not None:
          dirs[s] = exp_id
        else:
          nao_rodou.append(f"{exp_id}: erro")
        commit(f"experimentos: {exp_id}", arquivos(pasta(exp_id)) + [Path(__file__)])
      commit(f"experimentos: teste de {final_id}", [pasta(final_id) / "metricas_teste.json"])
    av = avaliacao_final(final_id, final_cfg, pos, dirs)
    escrever_resumo(hist, bloco_final(final_id, final_cfg, pos, av, None, nao_rodou))
    commit("experimentos: r2_avaliacao_final", [pasta("r2_avaliacao_final") / "avaliacao_final.json", Path(O["saida"]) / "resumo_rodada2.md", Path(__file__)])

    # E: curva de dados
    curva = None
    try:
      curva = curva_dados(final_id, final_cfg)
    except Exception:
      log("r2_E_curva_dados: ERRO\n" + traceback.format_exc())
      nao_rodou.append("curva de dados: erro (ver progresso_rodada2.log)")
    escrever_resumo(hist, bloco_final(final_id, final_cfg, pos, av, curva, nao_rodou))
    commit("experimentos: r2_E_curva_dados e resumo final", [pasta("r2_E_curva_dados") / "curva_dados.json", pasta("r2_E_curva_dados") / "curva_dados.pdf",
                                                             Path(O["saida"]) / "resumo_rodada2.md", Path(__file__)])
    log("fim")
  except Exception:
    log("ERRO GERAL:\n" + traceback.format_exc())
    raise
  finally:
    (Path(O["saida"]) / "rodando_rodada2.pid").unlink(missing_ok=True)


if __name__ == "__main__":
  main()
