# ============================================================
# ARQUIVO: experimentos/rodar_tudo.py
#
# O QUE FAZ: roda, sozinho e em sequência, a construção acumulativa do
# modelo com rótulos por intervalo descrita em experimentos/protocolo.md:
# passo 0 (análise dos pontos ausentes), etapas 1 a 8, estabilidade
# entre sementes e o experimento separado de 3 classes. Retomável: pula
# todo experimento que já tem metricas_validacao.json. Erro num
# experimento vai para experimentos/<id>/erro.txt e o script segue.
# Commit (sem push) do código e dos resultados ao fim de cada
# experimento; modelos, vídeos e dados de pose nunca são versionados.
# Não altera o modelo base nem data/results/.
#
# Uso (da raiz do repositório):
#   .venv/Scripts/python.exe experimentos/rodar_tudo.py
#   .venv/Scripts/python.exe experimentos/rodar_tudo.py --teste --saida <pasta>   (rodada rápida)
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
os.chdir(RAIZ)  # evaluate_temporal.py usa caminhos relativos à raiz
sys.path.insert(0, str(RAIZ / "src"))

PARTICOES = json.loads((RAIZ / "models/oof/fold_assignments.json").read_text(encoding="utf-8"))["video_to_fold"]
CLASSES = ["ArmFlapping", "HeadBanging", "Spinning", "Background"]
LIMIAR_CONF = 0.5
LIMITE_NORM = 4.0          # alturas de tronco (etapa 3)
MAX_LACUNA_S = 0.5         # etapas 4 e 5
METRICAS = ["f1_evento_iou_0.3", "acuracia_balanceada_presenca", "macro_f1_quadro_todos"]
GANHO_MINIMO = 0.01
SEMENTE = 42
SEMENTES_EXTRAS = [1000, 2000]
MODELOS_POSE = {"yolo11n": "models/yolo11n-pose.pt", "yolo11m": "models/yolo11m-pose.pt"}

BASE = {
  "pose": "yolo11n", "normalizacao": "tcc", "interp_lacuna_s": 0.0, "suavizacao_s": 0.0,
  "fps_alvo": None, "janela": 30, "passo_treino": 1, "lr": 0.001, "lote": 32, "oculto": 128, "camadas": 2,
  "max_epocas": 500, "paciencia": 20, "rotulos": "intervalos_4_classes", "subamostrar_fundo": True, "semente": SEMENTE,
}
ETAPAS = [
  ("01_ponto_de_partida", {}, "configuração atual do modelo com rótulos por intervalo"),
  ("02_yolo11m", {"pose": "yolo11m"}, "pose com YOLO11m-Pose"),
  ("03_normalizacao_robusta", {"normalizacao": "robusta"}, "escala pelo tronco, confiança < 0,5 interpolada, limite ±4"),
  ("04_interpolacao_suavizacao", {"interp_lacuna_s": MAX_LACUNA_S, "suavizacao_s": 0.2}, "lacunas até 0,5 s interpoladas, média móvel de 0,2 s"),
  ("05_15fps_janela_1s", {"fps_alvo": 15, "janela": 15}, "15 quadros/s, janela de 1 s"),
  ("06_15fps_janela_2s", {"fps_alvo": 15, "janela": 30}, "15 quadros/s, janela de 2 s"),
  ("07_sem_sobreposicao", {"passo_treino": "janela"}, "janelas sem sobreposição no treino"),
]
ETAPA_LR = [("08a_lr_0.0003", {"lr": 0.0003}, "taxa de aprendizado 0,0003"),
            ("08b_lr_0.003", {"lr": 0.003}, "taxa de aprendizado 0,003")]

# preenchidos em main()
OPC = {}


def caminho(*partes):
  return Path(OPC["saida"]).joinpath(*partes)


def log(msg):
  linha = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
  print(linha, flush=True)
  with open(caminho("progresso.log"), "a", encoding="utf-8") as f:
    f.write(linha + "\n")


def gravar_json(p, obj):
  p.parent.mkdir(parents=True, exist_ok=True)
  p.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def ler_json(p):
  return json.loads(Path(p).read_text(encoding="utf-8"))


def commit(msg, arquivos):
  if OPC["teste"]:
    return
  arquivos = [str(Path(a).relative_to(RAIZ)) for a in arquivos if Path(a).exists()]
  if not arquivos:
    return
  subprocess.run(["git", "add", "-f", "--", *arquivos], check=True, capture_output=True)
  r = subprocess.run(["git", "commit", "-m", f"{msg}\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>", "--", *arquivos],
                     capture_output=True, text=True)
  log(f"commit: {msg}" if r.returncode == 0 else f"commit sem mudanças ou falhou: {r.stdout.strip()[-200:]} {r.stderr.strip()[-200:]}")


def classe_do_video(v):
  return v.split("_")[1]


def videos():
  return OPC.get("videos") or sorted(PARTICOES)


def particoes():
  return OPC.get("particoes") or [1, 2, 3, 4, 5]


# ---------- divisão de validação ----------

def divisao_validacao():
  p = caminho("divisao_validacao.json")
  if p.exists():
    return {int(k): v for k, v in ler_json(p)["validacao_por_particao"].items()}
  div = {}
  for k in particoes():
    treino = [v for v in videos() if PARTICOES[v] != k]
    val = []
    rng = np.random.default_rng(SEMENTE + k)
    for c in sorted({classe_do_video(v) for v in treino}):
      vs = sorted(v for v in treino if classe_do_video(v) == c)
      n = max(1, round(0.2 * len(vs)))
      val += [vs[i] for i in sorted(rng.permutation(len(vs))[:n])]
    div[k] = sorted(val)
  gravar_json(p, {"regra": "por partição, round(0,2 × n) vídeos de treino de cada classe (mínimo 1), numpy.random.default_rng(42 + partição)",
                  "validacao_por_particao": {str(k): v for k, v in div.items()}})
  return div


# ---------- extração de pose ----------

_yolo = {}


def extrair_um(args):
  modelo, vid, destino, limite = args
  # a ultralytics precisa vir antes do torch: ela fixa OMP_NUM_THREADS=1 ao ser
  # importada, e com essa ordem a pose reproduz data/poses/*.csv sem diferença
  import cv2
  from ultralytics import YOLO
  import torch
  torch.set_num_threads(2)
  if modelo not in _yolo:
    _yolo[modelo] = YOLO(str(RAIZ / MODELOS_POSE[modelo]))
  t0 = time.time()
  cap = cv2.VideoCapture(str(RAIZ / "data/videos" / f"{vid}.mp4"))
  w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  quadros, xy, conf, f = [], [], [], 0
  while limite is None or f < limite:
    ok, img = cap.read()
    if not ok:
      break
    for r in _yolo[modelo](img, verbose=False):  # mesma regra de extract_pose.py: primeira pessoa
      if r.keypoints is None:
        continue
      p = r.keypoints.xy.cpu().numpy()
      if len(p) == 0:
        continue
      quadros.append(f)
      xy.append(p[0])
      conf.append(r.keypoints.conf.cpu().numpy()[0])
    f += 1
  cap.release()
  parcial = str(destino) + ".parcial.npz"
  np.savez(parcial, frames=np.array(quadros, dtype=np.int64), xy=np.array(xy, dtype=np.float32).reshape(-1, 17, 2),
           conf=np.array(conf, dtype=np.float32).reshape(-1, 17), w=w, h=h, quadros_lidos=f)
  os.replace(parcial, destino)
  return vid, len(quadros), f, time.time() - t0


def extrair(modelo):
  pasta = caminho("poses", modelo)
  pasta.mkdir(parents=True, exist_ok=True)
  faltam = [v for v in videos() if not (pasta / f"{v}.npz").exists()]
  if faltam:
    log(f"extração {modelo}: {len(faltam)} vídeos, 4 processos")
    # um processo novo por vídeo: no mesmo processo, o segundo vídeo em diante
    # pode sair com diferenças de ~1e-3 em relação a data/poses/*.csv
    with mp.get_context("spawn").Pool(4, maxtasksperchild=1) as pool:
      for vid, n, f, dt in pool.imap_unordered(extrair_um, [(modelo, v, pasta / f"{v}.npz", OPC.get("limite_quadros")) for v in faltam]):
        log(f"  {modelo} {vid}: {n} de {f} quadros com pessoa, {dt:.0f} s")
  if modelo == "yolo11n":
    conferir_com_csv(pasta)


def conferir_com_csv(pasta):
  """O YOLO11n refeito tem de reproduzir data/poses/*.csv (o ponto de partida do TCC)."""
  p = pasta / "conferencia_csv.json"
  feito = ler_json(p) if p.exists() else {}
  from pose_features import flatten_features, normalize_frame_keypoints
  for v in videos():
    if v in feito:
      continue
    d = np.load(pasta / f"{v}.npz")
    csv = np.loadtxt(RAIZ / "data/poses" / f"{v}.csv", delimiter=",", skiprows=1, ndmin=2)[:len(d["frames"])]
    prev, linhas = None, []
    for q in d["xy"]:
      feats, prev = normalize_frame_keypoints(q, prev)
      linhas.append(flatten_features(feats))
    igual = len(csv) == len(linhas) and np.array_equal(csv[:, 0].astype(int), d["frames"])
    dif = float(np.abs(csv[:, 1:] - np.array(linhas)).max()) if igual and len(linhas) else None
    feito[v] = {"mesmos_quadros": bool(igual), "diferenca_maxima": dif}
    if not igual or dif is None or dif > 1e-6:
      log(f"AVISO: {v} não reproduz data/poses/{v}.csv ({feito[v]})")
  gravar_json(p, feito)


# ---------- atributos por vídeo ----------

def media_movel(P, meia):
  """Média centrada de 2·meia+1 linhas; nas bordas, só sobre as linhas disponíveis."""
  n = len(P)
  c = np.concatenate([np.zeros((1,) + P.shape[1:]), np.cumsum(P, axis=0)])
  i = np.arange(n)
  a, b = np.clip(i - meia, 0, n), np.clip(i + meia + 1, 0, n)
  return (c[b] - c[a]) / (b - a)[:, None, None]


def preparar_video(vid, cfg):
  """Linhas (quadros originais) com posições normalizadas e atributos
  x, y, dx, dy por ponto, e o rótulo de cada quadro original."""
  import evaluate_temporal as et
  from pose_features import normalize_frame_keypoints
  d = np.load(caminho("poses", cfg["pose"], f"{vid}.npz"))
  quadros, xy, conf = d["frames"].astype(np.int64), d["xy"].astype(np.float32), d["conf"].astype(np.float32)
  if len(quadros) == 0:
    return None
  fps = et.load_video_fps(vid)
  n_quadros = int(quadros[-1]) + 1

  if cfg["normalizacao"] == "tcc":
    P = np.empty((len(quadros), 17, 2), np.float32)
    D = np.empty_like(P)
    prev = None
    for i in range(len(quadros)):
      feats, prev = normalize_frame_keypoints(xy[i], prev)
      a = np.asarray(feats, np.float32)
      P[i], D[i] = a[:, :2], a[:, 2:]
  else:
    xy = xy.astype(np.float64)
    for k in range(17):
      ok = conf[:, k] >= LIMIAR_CONF
      if ok.any() and not ok.all():
        for c in (0, 1):
          xy[~ok, k, c] = np.interp(quadros[~ok], quadros[ok], xy[ok, k, c])
    quadril = (xy[:, 11] + xy[:, 12]) / 2
    escala = np.linalg.norm((xy[:, 5] + xy[:, 6]) / 2 - quadril, axis=1)
    ruim = escala < 1
    if ruim.all():
      escala[:] = 1
    elif ruim.any():  # escala degenerada: repete a última válida (ou a primeira, no começo)
      idx = np.maximum.accumulate(np.where(ruim, -1, np.arange(len(escala))))
      escala = escala[np.where(idx < 0, np.argmax(~ruim), idx)]
    P = np.clip((xy - quadril[:, None]) / escala[:, None, None], -LIMITE_NORM, LIMITE_NORM).astype(np.float32)
    D = None

  if cfg["interp_lacuna_s"] > 0:
    falta = np.diff(quadros) - 1
    novos = [np.arange(quadros[i] + 1, quadros[i + 1]) for i in np.where((falta > 0) & (falta / fps <= cfg["interp_lacuna_s"]))[0]]
    if novos:
      novos = np.concatenate(novos)
      Pn = np.empty((len(novos), 17, 2), np.float32)
      for k in range(17):
        for c in (0, 1):
          Pn[:, k, c] = np.interp(novos, quadros, P[:, k, c])
      ordem = np.argsort(np.concatenate([quadros, novos]), kind="stable")
      quadros = np.concatenate([quadros, novos])[ordem]
      P = np.concatenate([P, Pn])[ordem]
    D = None
  if cfg["suavizacao_s"] > 0:
    meia = int(round(cfg["suavizacao_s"] * fps / 2))
    cortes = np.where(np.diff(quadros) > 1)[0] + 1
    P = np.concatenate([media_movel(s, meia) for s in np.split(P.astype(np.float64), cortes)]).astype(np.float32)
    D = None
  if D is None:
    D = np.concatenate([np.zeros((1, 17, 2), np.float32), np.diff(P, axis=0)]).astype(np.float32)

  gt, _ = et.ground_truth_frames(vid, n_quadros, fps)
  rotulo = np.array([CLASSES.index(g) for g in gt], np.int64)
  return {"quadros": quadros, "P": P, "F": np.concatenate([P, D], axis=2).reshape(len(quadros), 68), "fps": fps, "rotulo": rotulo}


def rotular(rotulos_amostras, W, n_classes):
  """Regra de create_sequences_annotated.py sobre cada janela (linhas = janelas)."""
  minimo = int(math.ceil(2 / 3 * W))
  out = np.full(len(rotulos_amostras), -1, np.int64)
  for c in range(3):  # mesma ordem de prioridade do script original
    out[(out < 0) & ((rotulos_amostras == c).sum(1) >= minimo)] = c
  if n_classes == 4:
    out[(out < 0) & ((rotulos_amostras == 3).sum(1) == W)] = 3
  return out


def segmentos(quadros, fps):
  return np.split(np.arange(len(quadros)), np.where((np.diff(quadros) - 1) / fps > MAX_LACUNA_S)[0] + 1)


def amostras_e_janelas_treino(v, cfg, n_classes):
  """Matriz de amostras do vídeo e as janelas de treino (início, rótulo).
  Linhas originais, ou grade reamostrada por trecho contínuo."""
  W = cfg["janela"]
  passo = W if cfg["passo_treino"] == "janela" else int(cfg["passo_treino"])
  if not cfg["fps_alvo"]:
    A, quadro_da_amostra, inicios = v["F"], v["quadros"], np.arange(0, len(v["quadros"]) - W + 1, passo)
  else:
    blocos, qs, inicios, base = [], [], [], 0
    for seg in segmentos(v["quadros"], v["fps"]):
      t = v["quadros"][seg] / v["fps"]
      g = t[0] + np.arange(int(np.floor((t[-1] - t[0]) * cfg["fps_alvo"] + 1e-9)) + 1) / cfg["fps_alvo"]
      Pg = interpolar(g, t, v["P"][seg])
      Dg = np.concatenate([np.zeros((1, 17, 2), np.float32), np.diff(Pg, axis=0)])
      blocos.append(np.concatenate([Pg, Dg], axis=2).reshape(len(g), 68))
      qs.append(np.clip(np.round(g * v["fps"]).astype(np.int64), 0, len(v["rotulo"]) - 1))
      inicios.append(base + np.arange(0, len(g) - W + 1, passo))
      base += len(g)
    if not blocos:
      return np.zeros((0, 68), np.float32), np.zeros(0, np.int64), np.zeros(0, np.int64)
    A, quadro_da_amostra, inicios = np.concatenate(blocos).astype(np.float32), np.concatenate(qs), np.concatenate(inicios)
  if len(inicios) == 0:
    return A, inicios, inicios
  rot = rotular(v["rotulo"][quadro_da_amostra[inicios[:, None] + np.arange(W)]], W, n_classes)
  return A, inicios[rot >= 0], rot[rot >= 0]


def interpolar(tempos, t, P):
  out = np.empty((len(tempos), 17, 2), np.float32)
  for k in range(17):
    for c in (0, 1):
      out[:, k, c] = np.interp(tempos, t, P[:, k, c])
  return out


def janelas_avaliacao(v, cfg):
  """Uma janela por quadro original (passo 1), terminando nele. Devolve
  (X, quadro_inicio, quadro_fim) no formato de evaluate_temporal.py."""
  W = cfg["janela"]
  q = v["quadros"]
  if not cfg["fps_alvo"]:
    s = np.arange(len(q) - W + 1)
    if len(s) == 0:
      return None
    return v["F"][s[:, None] + np.arange(W)], q[s], q[s + W - 1] + 1
  Xs, ini, fim = [], [], []
  for seg in segmentos(q, v["fps"]):
    t = q[seg] / v["fps"]
    T = t[:, None] - (W - np.arange(W + 1))[None, :] / cfg["fps_alvo"]  # W+1 amostras: a primeira só serve para dx
    ok = T[:, 1] >= t[0] - 1e-9
    if not ok.any():
      continue
    T = T[ok]
    Pg = interpolar(T.ravel(), t, v["P"][seg]).reshape(len(T), W + 1, 17, 2)
    Dg = np.diff(Pg, axis=1)
    Dg[T[:, 0] < t[0] - 1e-9, 0] = 0  # antes do início do trecho: como na grade de treino
    Xs.append(np.concatenate([Pg[:, 1:], Dg], axis=3).reshape(len(T), W, 68))
    ini.append(np.maximum(q[seg][0], np.round(T[:, 1] * v["fps"]).astype(np.int64)))
    fim.append(q[seg][ok] + 1)
  if not Xs:
    return None
  return np.concatenate(Xs).astype(np.float32), np.concatenate(ini), np.concatenate(fim)


# ---------- treino de uma partição (processo separado) ----------

def conjunto(vids, cfg, n_classes, cache):
  As, ins, rots, donos, base = [], [], [], [], 0
  for vid in vids:
    v = cache[vid]
    if v is None:
      continue
    A, i, r = amostras_e_janelas_treino(v, cfg, n_classes)
    As.append(A)
    ins.append(i + base)
    rots.append(r)
    donos += [vid] * len(i)
    base += len(A)
  A = np.concatenate(As) if As else np.zeros((0, 68), np.float32)
  return A, np.concatenate(ins) if ins else np.zeros(0, np.int64), np.concatenate(rots) if rots else np.zeros(0, np.int64), np.array(donos)


def subamostrar(inicios, rotulos, donos, semente):
  from collections import defaultdict
  from train_oof_seeded import balanced_background_sample
  comp = [i for i in range(len(rotulos)) if rotulos[i] != 3]
  por_video = defaultdict(list)
  for i in range(len(rotulos)):
    if rotulos[i] == 3:
      por_video[str(donos[i])].append(i)
  escolhidos, _ = balanced_background_sample({v: np.array(x) for v, x in por_video.items()}, len(comp), np.random.default_rng(semente))
  sel = np.array(sorted(comp + escolhidos), np.int64)
  return inicios[sel], rotulos[sel]


def treinar_particao(args):
  exp_dir, cfg, k, val_videos, com_teste, opc = args
  OPC.update(opc)
  try:
    import torch
    import torch.nn as nn
    torch.set_num_threads(1)
    from train_oof_seeded import LSTMClassifier
    exp_dir = Path(exp_dir)
    feito = exp_dir / f"particao_{k}.json"
    if feito.exists():
      return k, None
    t0 = time.time()
    n_classes = 3 if cfg["rotulos"] == "anotados_3_classes" else 4
    treino = [v for v in videos() if PARTICOES[v] != k and v not in val_videos]
    teste = [v for v in videos() if PARTICOES[v] == k]
    cache = {v: preparar_video(v, cfg) for v in treino + val_videos + (teste if com_teste else [])}
    W, s = cfg["janela"], cfg["semente"]
    A_tr, i_tr, r_tr, d_tr = conjunto(treino, cfg, n_classes, cache)
    A_va, i_va, r_va, d_va = conjunto(val_videos, cfg, n_classes, cache)
    if n_classes == 4 and cfg["subamostrar_fundo"]:
      i_tr, r_tr = subamostrar(i_tr, r_tr, d_tr, s + k)
      i_va, r_va = subamostrar(i_va, r_va, d_va, s + k + 1000)
    if len(i_tr) == 0 or len(i_va) == 0:
      raise RuntimeError(f"partição {k}: {len(i_tr)} janelas de treino e {len(i_va)} de validação")
    A_tr, A_va = torch.from_numpy(A_tr), torch.from_numpy(A_va)
    off = torch.arange(W)
    lote = lambda A, ini: A[torch.from_numpy(ini)[:, None] + off]

    torch.manual_seed(s + k)
    modelo = LSTMClassifier(68, cfg["oculto"], cfg["camadas"], n_classes)
    opt = torch.optim.Adam(modelo.parameters(), lr=cfg["lr"])
    crit = nn.CrossEntropyLoss()
    gerador = torch.Generator().manual_seed(s + k)
    y_tr, y_va = torch.from_numpy(r_tr), torch.from_numpy(r_va)
    perdas, melhor, melhor_ep, estado, parado = [], float("inf"), 0, None, 0
    for ep in range(1, cfg["max_epocas"] + 1):
      modelo.train()
      ordem = torch.randperm(len(i_tr), generator=gerador).numpy()
      soma, nb = 0.0, 0
      for b in range(0, len(ordem), cfg["lote"]):
        j = ordem[b:b + cfg["lote"]]
        opt.zero_grad()
        perda = crit(modelo(lote(A_tr, i_tr[j])), y_tr[j])
        perda.backward()
        opt.step()
        soma += perda.item()
        nb += 1
      modelo.eval()
      with torch.no_grad():
        pv = sum(crit(modelo(lote(A_va, i_va[b:b + 2048])), y_va[b:b + 2048]).item() * len(i_va[b:b + 2048])
                 for b in range(0, len(i_va), 2048)) / len(i_va)
      perdas.append([ep, soma / nb, pv])
      if pv < melhor:
        melhor, melhor_ep, estado, parado = pv, ep, copy.deepcopy(modelo.state_dict()), 0
      else:
        parado += 1
      if ep % 10 == 0 or parado >= cfg["paciencia"]:
        with open(exp_dir / f"andamento_{k}.txt", "w", encoding="utf-8") as f:
          f.write(f"época {ep}, treino {soma / nb:.4f}, validação {pv:.4f}, melhor {melhor:.4f} na {melhor_ep}\n")
      if parado >= cfg["paciencia"]:
        break
    modelo.load_state_dict(estado)
    (exp_dir / "modelos").mkdir(exist_ok=True)
    torch.save(estado, exp_dir / "modelos" / f"particao_{k}.pt")
    saida = {"validacao": inferir(modelo, cache, val_videos, cfg)}
    if com_teste:
      saida["teste"] = inferir(modelo, cache, teste, cfg)
    with open(exp_dir / f"janelas_{k}.pkl", "wb") as f:
      pickle.dump(saida, f)
    gravar_json(feito, {"particao": k, "epoca_escolhida": melhor_ep, "epocas_rodadas": len(perdas), "perda_validacao_minima": melhor,
                        "janelas_treino": int(len(i_tr)), "janelas_validacao": int(len(i_va)),
                        "contagem_treino_por_classe": {CLASSES[c]: int((r_tr == c).sum()) for c in range(n_classes)},
                        "videos_treino": treino, "videos_validacao": val_videos, "perdas": perdas, "tempo_s": time.time() - t0})
    return k, None
  except Exception:
    return k, traceback.format_exc()


def inferir(modelo, cache, vids, cfg):
  import torch
  out = {}
  for vid in vids:
    v = cache.get(vid) or preparar_video(vid, cfg)
    j = janelas_avaliacao(v, cfg) if v else None
    if j is None:
      continue
    X, ini, fim = j
    with torch.no_grad():
      p = np.concatenate([torch.softmax(modelo(torch.from_numpy(X[b:b + 2048])), 1).numpy() for b in range(0, len(X), 2048)])
    out[vid] = {"fps": v["fps"], "quadro_inicio": ini, "quadro_fim": fim, "pontuacoes": p}
  return out


def avaliar_teste_particao(args):
  """Teste de um experimento já treinado, a partir dos modelos salvos."""
  exp_dir, cfg, k, opc = args
  OPC.update(opc)
  try:
    import torch
    torch.set_num_threads(1)
    from train_oof_seeded import LSTMClassifier
    exp_dir = Path(exp_dir)
    n_classes = 3 if cfg["rotulos"] == "anotados_3_classes" else 4
    modelo = LSTMClassifier(68, cfg["oculto"], cfg["camadas"], n_classes)
    modelo.load_state_dict(torch.load(exp_dir / "modelos" / f"particao_{k}.pt"))
    modelo.eval()
    teste = [v for v in videos() if PARTICOES[v] == k]
    res = inferir(modelo, {}, teste, cfg)
    p = exp_dir / f"janelas_{k}.pkl"
    with open(p, "rb") as f:
      saida = pickle.load(f)
    saida["teste"] = res
    with open(p, "wb") as f:
      pickle.dump(saida, f)
    return k, None
  except Exception:
    return k, traceback.format_exc()


# ---------- métricas ----------

def dados_avaliacao(exp_dir, parte, cfg):
  import evaluate_temporal as et
  nomes = CLASSES[:3] if cfg["rotulos"] == "anotados_3_classes" else CLASSES
  et.class_names, et.id_to_label, et.fold_assignments = nomes, dict(enumerate(nomes)), PARTICOES
  vd = {}
  for k in particoes():
    with open(exp_dir / f"janelas_{k}.pkl", "rb") as f:
      res = pickle.load(f)[parte]
    for vid, r in res.items():
      janelas = [{"quadro_inicio": int(a), "quadro_fim": int(b), "pontuacoes": p.tolist()}
                 for a, b, p in zip(r["quadro_inicio"], r["quadro_fim"], r["pontuacoes"])]
      n = janelas[-1]["quadro_fim"]
      gt, intervalos = et.ground_truth_frames(vid, n, r["fps"])
      chave = vid if parte == "teste" else f"{k}|{vid}"
      vd[chave] = {"fold": k, "fps": r["fps"], "n_frames": n, "windows": janelas, "gt_frames": gt, "gt_intervals": intervalos}
  return vd


def acuracia_balanceada(vd, ids):
  import evaluate_temporal as et
  from sklearn.metrics import balanced_accuracy_score
  yt, yp = [], []
  for v in ids:
    d = vd[v]
    for q, p in et.predicted_frames(d["windows"]).items():
      yt.append(d["gt_frames"][q] != "Background")
      yp.append(p != "Background")
  return float(balanced_accuracy_score(yt, yp))


def tres_metricas(vd, ids):
  import evaluate_temporal as et
  return {
    "f1_evento_iou_0.3": et.event_level_report(vd, ids, 0.5, 0.0, [0.3])[0.3]["macro_f1"],
    "acuracia_balanceada_presenca": acuracia_balanceada(vd, ids),
    "macro_f1_quadro_todos": et.frame_level_report(vd, ids)["macro_f1_com_background"],
  }


def metricas_teste(vd):
  import evaluate_temporal as et
  ids = sorted(vd)
  todos = et.frame_level_with_folds(vd, ids, False)
  anot = et.frame_level_with_folds(vd, ids, True)
  ev = et.event_level_with_folds(vd, ids, 0.5, 0.0, [0.3])[0.3]
  bal = [acuracia_balanceada(vd, [v for v in ids if vd[v]["fold"] == k]) for k in particoes()]
  return {
    "agregado_56_videos": {**tres_metricas(vd, ids), "macro_f1_quadro_anotados": anot["pooled_56_videos"]["macro_f1_sem_background"]},
    "media_entre_particoes": {"f1_evento_iou_0.3": ev["macro_f1_media_entre_folds"], "acuracia_balanceada_presenca": float(np.mean(bal)),
                              "macro_f1_quadro_todos": todos["macro_f1_com_background_media"], "macro_f1_quadro_anotados": anot["macro_f1_sem_background_media"]},
    "desvio_entre_particoes": {"f1_evento_iou_0.3": ev["macro_f1_desvio_entre_folds"], "acuracia_balanceada_presenca": float(np.std(bal)),
                               "macro_f1_quadro_todos": todos["macro_f1_com_background_desvio"], "macro_f1_quadro_anotados": anot["macro_f1_sem_background_desvio"]},
    "n_videos": len(ids),
  }


# ---------- figura e arquivos do experimento ----------

def curva_erro(exp_dir, info):
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  from matplotlib.ticker import FuncFormatter
  from gerar_figura_linha_tempo import DPI, LARGURA_POL, TEXTO_PT, cores_da_interface, virgula
  _, var = cores_da_interface()
  plt.rcParams.update({"font.size": TEXTO_PT, "axes.labelsize": TEXTO_PT, "xtick.labelsize": TEXTO_PT, "ytick.labelsize": TEXTO_PT,
                       "legend.fontsize": TEXTO_PT, "text.color": var["texto"], "axes.labelcolor": var["texto"],
                       "xtick.color": var["texto-2"], "ytick.color": var["texto-2"], "axes.edgecolor": var["linhas"], "pdf.fonttype": 42})
  n_ep = max(len(i["perdas"]) for i in info)
  fig, ax = plt.subplots(figsize=(LARGURA_POL, 5.2), dpi=DPI)
  for col, nome, cor in [(1, "treino", var["texto"]), (2, "validação", "#C7522A")]:
    m = np.full((len(info), n_ep), np.nan)
    for a, i in enumerate(info):
      m[a, :len(i["perdas"])] = [p[col] for p in i["perdas"]]
    ep = np.arange(1, n_ep + 1)
    media, desvio = np.nanmean(m, 0), np.nanstd(m, 0)
    ax.fill_between(ep, media - desvio, media + desvio, color=cor, alpha=0.15, linewidth=0)
    ax.plot(ep, media, color=cor, linewidth=2, label=nome)
  ax.fill_between([], [], [], color=var["texto-2"], alpha=0.15, linewidth=0, label="± 1 desvio entre partições")
  escolhidas = [i["epoca_escolhida"] for i in info]
  e = float(np.median(escolhidas))
  ax.axvline(e, color=var["texto-2"], linestyle="--", linewidth=1.5,
             label=f"época escolhida: mediana {virgula(e)} (de {min(escolhidas)} a {max(escolhidas)})")
  ax.set_xlabel("época")
  ax.set_ylabel("perda média entre partições")
  ax.yaxis.set_major_formatter(FuncFormatter(virgula))
  ax.grid(axis="y", color=var["linhas"], linewidth=0.8)
  for lado in ("top", "right"):
    ax.spines[lado].set_visible(False)
  ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
  fig.tight_layout()
  fig.savefig(exp_dir / "curva_erro.pdf")
  plt.close(fig)


def gravar_perdas(exp_dir, info):
  linhas = ["particao,epoca,perda_treino,perda_validacao,epoca_escolhida"]
  for i in info:
    for ep, tr, va in i["perdas"]:
      linhas.append(f"{i['particao']},{ep},{tr:.6f},{va:.6f},{int(ep == i['epoca_escolhida'])}")
  (exp_dir / "perdas.csv").write_text("\n".join(linhas) + "\n", encoding="utf-8")


def arquivos_do_experimento(exp_dir):
  return [exp_dir / n for n in ("config.json", "seeds.json", "metricas_validacao.json", "metricas_teste.json", "perdas.csv",
                                "curva_erro.pdf", "comparacao_pose.json", "erro.txt")]


def rodar_experimento(exp_id, cfg, com_teste, descricao):
  """Treina (ou retoma) o experimento e devolve as métricas de validação, ou None se falhar."""
  exp_dir = caminho(exp_id)
  feito = exp_dir / "metricas_validacao.json"
  if feito.exists():
    return ler_json(feito)
  exp_dir.mkdir(parents=True, exist_ok=True)
  (exp_dir / "erro.txt").unlink(missing_ok=True)
  log(f"{exp_id}: início ({descricao})")
  gravar_json(exp_dir / "config.json", {"descricao": descricao, **cfg, "teste_calculado": com_teste})
  s = cfg["semente"]
  gravar_json(exp_dir / "seeds.json", {
    "semente": s, "inicializacao_pytorch_por_particao": {k: s + k for k in particoes()},
    "ordem_dos_lotes_por_particao": {k: s + k for k in particoes()},
    "subamostragem_fundo_treino_por_particao": {k: s + k for k in particoes()},
    "subamostragem_fundo_validacao_por_particao": {k: s + k + 1000 for k in particoes()},
    "divisao_validacao": "divisao_validacao.json (semente 42 + partição, a mesma em todos os experimentos)"})
  t0 = time.time()
  try:
    div = divisao_validacao()
    tarefas = [(str(exp_dir), cfg, k, div[k], com_teste, dict(OPC)) for k in particoes()]
    with mp.get_context("spawn").Pool(len(tarefas)) as pool:
      erros = [e for _, e in pool.map(treinar_particao, tarefas) if e]
    if erros:
      raise RuntimeError("\n\n".join(erros))
    info = [ler_json(exp_dir / f"particao_{k}.json") for k in particoes()]
    t_aval = time.time()
    vd = dados_avaliacao(exp_dir, "validacao", cfg)
    ids = sorted(vd)
    met = {"agregado_validacao": tres_metricas(vd, ids),
           "por_particao": {k: tres_metricas(vd, [i for i in ids if vd[i]["fold"] == k]) for k in particoes()},
           "epoca_escolhida_por_particao": {i["particao"]: i["epoca_escolhida"] for i in info},
           "tempo_s": None}
    if cfg["rotulos"] == "anotados_3_classes":
      import evaluate_temporal as et
      met["macro_f1_quadro_anotados_validacao"] = et.frame_level_report(vd, ids, True)["macro_f1_sem_background"]
    if com_teste:
      gravar_json(exp_dir / "metricas_teste.json", metricas_teste(dados_avaliacao(exp_dir, "teste", cfg)))
    gravar_perdas(exp_dir, info)
    curva_erro(exp_dir, info)
    # partições em paralelo: tempo = a partição mais lenta + a avaliação
    met["tempo_s"] = max(i["tempo_s"] for i in info) + time.time() - t_aval
    gravar_json(feito, met)
    a = met["agregado_validacao"]
    log(f"{exp_id}: fim em {met['tempo_s'] / 60:.0f} min | " + " ".join(f"{m}={a[m]:.3f}" for m in METRICAS))
    return met
  except Exception:
    (exp_dir / "erro.txt").write_text(traceback.format_exc(), encoding="utf-8")
    log(f"{exp_id}: ERRO (ver {exp_id}/erro.txt)")
    return None


def fechar(exp_id):
  commit(f"experimentos: {exp_id}", arquivos_do_experimento(caminho(exp_id)) + [caminho("resumo.md"), Path(__file__)])


def testar_depois(exp_id, cfg):
  """Teste de um experimento já treinado sem teste (o modelo final)."""
  exp_dir = caminho(exp_id)
  if (exp_dir / "metricas_teste.json").exists():
    return
  log(f"{exp_id}: teste do modelo final")
  with mp.get_context("spawn").Pool(len(particoes())) as pool:
    erros = [e for _, e in pool.map(avaliar_teste_particao, [(str(exp_dir), cfg, k, dict(OPC)) for k in particoes()]) if e]
  if erros:
    raise RuntimeError("\n\n".join(erros))
  gravar_json(exp_dir / "metricas_teste.json", metricas_teste(dados_avaliacao(exp_dir, "teste", cfg)))
  commit(f"experimentos: teste de {exp_id}", [exp_dir / "metricas_teste.json"])


def supera(cand, melhor):
  ganhos = [cand[m] - melhor[m] for m in METRICAS]
  return sum(g >= GANHO_MINIMO for g in ganhos) >= 2, sum(ganhos)


# ---------- passo 0 ----------

def passo0():
  pasta = caminho("00_analise_pontos")
  if (pasta / "decisao.json").exists():
    return ler_json(pasta / "decisao.json")
  log("passo 0: análise dos pontos ausentes")
  pasta.mkdir(parents=True, exist_ok=True)
  import evaluate_temporal as et
  from ssbd_annotations import load_annotations
  cfg = dict(BASE)
  linhas_cls, xy, cf, cs, n_por_classe = [], [], [], [], {}
  picos_spin = {"quadros": 0, "anotados": 0, "picos": 0, "picos_dentro": 0}
  for vid in videos():
    d = np.load(caminho("poses", "yolo11n", f"{vid}.npz"))
    v = preparar_video(vid, cfg)
    c = classe_do_video(vid)
    linhas_cls += [c] * len(d["frames"])
    xy.append(d["xy"]); cf.append(d["conf"]); cs.append(v["F"].reshape(-1, 17, 4))
    if c == "Spinning":
      gt = v["rotulo"][d["frames"]] == 2
      pico = np.abs(v["F"].reshape(-1, 17, 4)[..., :2]).max(axis=(1, 2)) > 5
      for chave, val in (("quadros", len(gt)), ("anotados", gt.sum()), ("picos", pico.sum()), ("picos_dentro", (pico & gt).sum())):
        picos_spin[chave] += int(val)
  cl = np.array(linhas_cls)
  xy, cf, cs = np.concatenate(xy), np.concatenate(cf), np.concatenate(cs)
  baixa = cf < LIMIAR_CONF
  norm = baixa[:, [5, 6, 11, 12]].any(1)
  pico = np.abs(cs[..., :2]).max(axis=(1, 2)) > 5
  zero = ((xy[..., 0] == 0) & (xy[..., 1] == 0)).any(1)
  tronco = np.linalg.norm((xy[:, 5] + xy[:, 6]) / 2 - (xy[:, 11] + xy[:, 12]) / 2, axis=1)
  ombros = np.linalg.norm(xy[:, 6] - xy[:, 5], axis=1)
  por_classe = {c: {"quadros": int((cl == c).sum()), "fracao_normalizacao_ausente": float(norm[cl == c].mean()),
                    "fracao_acima_de_5": float(pico[cl == c].mean())} for c in sorted(set(cl))}
  dec = {
    "quadros_com_pessoa": int(len(cl)),
    "fracao_algum_ponto_conf_baixa": float(baixa.any(1).mean()),
    "fracao_conf_baixa_por_ponto": [float(x) for x in baixa.mean(0)],
    "fracao_algum_ponto_em_0_0": float(zero.mean()),
    "fracao_normalizacao_ausente": float(norm.mean()),
    "fracao_acima_de_5": float(pico.mean()),
    "fracao_dos_picos_com_normalizacao_ausente": float(norm[pico].mean()) if pico.any() else None,
    "distancia_ombros_menor_que_1px": int((ombros < 1).sum()),
    "distancia_ombros_p1_p50_px": [float(x) for x in np.percentile(ombros, [1, 50])],
    "altura_tronco_p1_p50_px": [float(x) for x in np.percentile(tronco, [1, 50])],
    "por_classe": por_classe,
    "spinning": {**picos_spin, "fracao_anotada": picos_spin["anotados"] / max(1, picos_spin["quadros"]),
                 "fracao_dos_picos_dentro": picos_spin["picos_dentro"] / max(1, picos_spin["picos"]),
                 "taxa_de_picos_dentro": picos_spin["picos_dentro"] / max(1, picos_spin["anotados"]),
                 "taxa_de_picos_fora": (picos_spin["picos"] - picos_spin["picos_dentro"]) / max(1, picos_spin["quadros"] - picos_spin["anotados"])},
  }
  dec["problema_confirmado"] = dec["fracao_normalizacao_ausente"] >= 0.05 and dec["fracao_acima_de_5"] >= 0.01
  gravar_json(pasta / "decisao.json", dec)
  f3 = lambda x: f"{x:.3f}".replace(".", ",")
  sp = dec["spinning"]
  cc = ler_json(caminho("poses", "yolo11n", "conferencia_csv.json"))
  conf = {"n": len(cc), "iguais": sum(1 for x in cc.values() if x["mesmos_quadros"] and x["diferenca_maxima"] == 0),
          "dif": max((x["diferenca_maxima"] or 0) for x in cc.values())}
  dec["conferencia_csv"] = conf
  gravar_json(pasta / "decisao.json", dec)
  linhas = [
    f"# Passo 0: pontos ausentes na pose (YOLO11n-Pose, {len(videos())} vídeos)", "",
    "Gerado por `experimentos/rodar_tudo.py` a partir da pose refeita com o mesmo modelo e a mesma regra de `extract_pose.py`.",
    f"Conferência com `data/poses/*.csv`: {conf['iguais']} de {conf['n']} vídeos com os mesmos quadros e os mesmos valores "
    f"normalizados; maior diferença em qualquer vídeo: {conf['dif']:.2g}.", "",
    "## 1. O que `keypoints.xy` devolve",
    "Na Ultralytics instalada (8.4.66 no `.venv`; 8.4.60 no Python do sistema, com a mesma classe `Keypoints`), "
    "`Keypoints.xy` é só `self.data[..., :2]` (`ultralytics/engine/results.py`). Não há máscara por confiança: um ponto pouco "
    "confiável sai com a coordenada estimada, nunca zerado. A única alteração é `clip_coords` (`ultralytics/utils/ops.py`), "
    "chamada por `scale_coords` no pós-processamento da pose, que prende x em [0, largura] e y em [0, altura].",
    f"Nos dados: {f3(dec['fracao_algum_ponto_em_0_0'])} dos quadros têm algum ponto em (0, 0).", "",
    "## 2. Pontos com confiança baixa",
    f"Quadros com pessoa: {dec['quadros_com_pessoa']}. Com algum ponto de confiança < 0,5: {f3(dec['fracao_algum_ponto_conf_baixa'])}.",
    "Fração de quadros com confiança < 0,5, por ponto (0 a 16): " + "; ".join(f"{i}: {f3(x)}" for i, x in enumerate(dec["fracao_conf_baixa_por_ponto"])) + ".", "",
    "## 3. Pontos da normalização (5, 6, 11, 12)",
    f"Quadros com algum deles com confiança < 0,5: {f3(dec['fracao_normalizacao_ausente'])}.",
    "`normalize_frame_keypoints` não olha a confiança: centraliza no meio dos quadris e divide pela distância entre ombros, "
    "e a única proteção é trocar a escala por 1 quando ela fica abaixo de 1 pixel (não há divisão por zero, mas um ombro mal "
    f"estimado encolhe a escala). Distância entre ombros abaixo de 1 px: {dec['distancia_ombros_menor_que_1px']} quadros; "
    f"percentis 1 e 50: {f3(dec['distancia_ombros_p1_p50_px'][0])} e {f3(dec['distancia_ombros_p1_p50_px'][1])} px.", "",
    "## 4. Valores normalizados acima de 5 larguras de ombro",
    f"Fração de quadros: {f3(dec['fracao_acima_de_5'])}. Desses, com algum ponto da normalização ausente: "
    f"{f3(dec['fracao_dos_picos_com_normalizacao_ausente'] or 0)}.", "",
    "| classe | quadros | algum de 5, 6, 11, 12 com conf. < 0,5 | algum valor > 5 |", "|---|---|---|---|",
    *[f"| *{c}* | {x['quadros']} | {f3(x['fracao_normalizacao_ausente'])} | {f3(x['fracao_acima_de_5'])} |" for c, x in por_classe.items()],
    f"| total | {dec['quadros_com_pessoa']} | {f3(dec['fracao_normalizacao_ausente'])} | {f3(dec['fracao_acima_de_5'])} |", "",
    "## 5. Picos nos vídeos de *spinning*",
    f"Quadros anotados como giro: {f3(sp['fracao_anotada'])} dos quadros desses vídeos; picos dentro dos intervalos de giro: "
    f"{f3(sp['fracao_dos_picos_dentro'])} dos picos. Taxa de picos dentro dos intervalos: {f3(sp['taxa_de_picos_dentro'])}; "
    f"fora: {f3(sp['taxa_de_picos_fora'])}.", "",
    "## Decisão",
    ("Problema confirmado pelo critério do protocolo (≥ 0,05 dos quadros com ponto da normalização ausente e ≥ 0,01 com valor > 5): "
     "a etapa 3 roda.") if dec["problema_confirmado"] else "Problema não confirmado pelo critério do protocolo: a etapa 3 não roda.",
  ]
  (pasta / "relatorio.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")
  commit("experimentos: passo 0, análise dos pontos ausentes", [pasta / "relatorio.md", pasta / "decisao.json", Path(__file__)])
  return dec


def comparar_poses():
  p = caminho("02_yolo11m", "comparacao_pose.json")
  if p.exists():
    return
  out = {}
  for m in MODELOS_POSE:
    det, lidos, conf, por_classe = 0, 0, [], {}
    for v in videos():
      d = np.load(caminho("poses", m, f"{v}.npz"))
      lidos_v = int(d["quadros_lidos"]) if "quadros_lidos" in d.files else None
      if lidos_v is None:
        import cv2
        cap = cv2.VideoCapture(str(RAIZ / "data/videos" / f"{v}.mp4"))
        lidos_v = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
      det += len(d["frames"]); lidos += lidos_v; conf.append(d["conf"].ravel())
      c = por_classe.setdefault(classe_do_video(v), [0, 0])
      c[0] += len(d["frames"]); c[1] += lidos_v
    out[m] = {"fracao_quadros_com_pessoa": det / lidos, "confianca_media_dos_pontos": float(np.concatenate(conf).mean()),
              "fracao_por_classe": {c: a / b for c, (a, b) in por_classe.items()}}
  gravar_json(p, out)


# ---------- resumo ----------

def escrever_resumo(hist, final=None):
  f3 = lambda x: "--" if x is None else f"{x:.3f}".replace(".", ",")
  linhas = ["# Resumo dos experimentos", "",
            "Decisão na validação (protocolo em `protocolo.md`): mantém o ajuste que supera o melhor modelo em pelo menos 2 das 3 métricas, com ganho ≥ 0,01.", "",
            "| etapa | mudança | F1 evento IoU 0,3 | acurácia bal. presença | macro F1 quadros | mantida | tempo |", "|---|---|---|---|---|---|---|"]
  for h in hist:
    a = h.get("metricas") or {}
    linhas.append(f"| {h['id']} | {h['descricao']} | {f3(a.get(METRICAS[0]))} | {f3(a.get(METRICAS[1]))} | {f3(a.get(METRICAS[2]))} | "
                  f"{h['situacao']} | {h.get('tempo', '--')} |")
  if final:
    linhas += ["", "## Modelo final", "", f"Configuração: `{json.dumps(final['cfg'], ensure_ascii=False)}`", "",
               "| teste (agregado, 56 vídeos) | F1 evento IoU 0,3 | acurácia bal. presença | macro F1 quadros | macro F1 quadros anotados |",
               "|---|---|---|---|---|"]
    for nome, t in final["testes"]:
      linhas.append(f"| {nome} | " + " | ".join(f3(t.get(m)) for m in METRICAS + ["macro_f1_quadro_anotados"]) + " |")
    if final.get("estabilidade"):
      e = final["estabilidade"]
      linhas += ["", "Estabilidade entre as sementes (média ± desvio de 3 execuções):", ""]
      for parte in ("validacao", "teste"):
        linhas.append(f"- {parte}: " + "; ".join(f"{m} {f3(e[parte][m]['media'])} ± {f3(e[parte][m]['desvio'])}" for m in METRICAS))
    if final.get("anotados"):
      linhas += ["", final["anotados"]]
    if final.get("nao_rodou"):
      linhas += ["", "Não rodou: " + "; ".join(final["nao_rodou"]) + "."]
  caminho("resumo.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")


def tempo(met):
  if not met or not met.get("tempo_s"):
    return "--"
  t = met["tempo_s"]
  return f"{t / 60:.0f} min" if t < 3600 else f"{t / 3600:.1f} h".replace(".", ",")


# ---------- sequência ----------

def ja_rodando():
  p = caminho("rodando.pid")
  if p.exists():
    pid = p.read_text().strip()
    r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True)
    if pid and pid in r.stdout and "python" in r.stdout.lower():
      return True
  p.write_text(str(os.getpid()))
  return False


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--teste", action="store_true", help="rodada rápida: 1 partição, 2 épocas, poucos vídeos, sem commit")
  ap.add_argument("--saida", default=str(RAIZ / "experimentos"))
  ap.add_argument("--videos", nargs="*")
  ap.add_argument("--limite-quadros", type=int)
  args = ap.parse_args()
  OPC.update({"teste": args.teste, "saida": str(Path(args.saida).resolve()), "videos": args.videos,
              "limite_quadros": args.limite_quadros, "particoes": [1] if args.teste else None})
  if args.teste:
    BASE.update({"max_epocas": 2})
  Path(OPC["saida"]).mkdir(parents=True, exist_ok=True)
  if ja_rodando():
    print("Já há uma execução em andamento.")
    return
  log(f"início (pid {os.getpid()}){' -- TESTE' if args.teste else ''}")
  hist = []
  try:
    divisao_validacao()
    extrair("yolo11n")
    dec = passo0()

    melhor_cfg, melhor_id = dict(BASE), None
    melhor = None
    for exp_id, mudanca, descricao in ETAPAS:
      if exp_id.startswith("03") and not dec["problema_confirmado"]:
        hist.append({"id": exp_id, "descricao": descricao, "situacao": "não rodou (passo 0 não confirmou)"})
        continue
      if exp_id.startswith("02"):
        try:
          extrair("yolo11m")
          comparar_poses()
        except Exception:
          caminho(exp_id).mkdir(parents=True, exist_ok=True)
          (caminho(exp_id) / "erro.txt").write_text(traceback.format_exc(), encoding="utf-8")
          log(f"{exp_id}: ERRO na extração")
          hist.append({"id": exp_id, "descricao": descricao, "situacao": "erro na extração"})
          escrever_resumo(hist)
          fechar(exp_id)
          continue
      cfg = {**melhor_cfg, **mudanca}
      met = rodar_experimento(exp_id, cfg, com_teste=exp_id.startswith("01"), descricao=descricao)
      if met is None:
        hist.append({"id": exp_id, "descricao": descricao, "situacao": "erro"})
        escrever_resumo(hist)
        fechar(exp_id)
        if melhor is None:
          raise RuntimeError("o ponto de partida falhou; nada a comparar")
        continue
      a = met["agregado_validacao"]
      if melhor is None:
        situacao, (melhor, melhor_cfg, melhor_id) = "ponto de partida", (a, cfg, exp_id)
      else:
        ok, _ = supera(a, melhor)
        situacao = "sim" if ok else "não"
        if ok:
          melhor, melhor_cfg, melhor_id = a, cfg, exp_id
      hist.append({"id": exp_id, "descricao": descricao, "metricas": a, "situacao": situacao, "tempo": tempo(met)})
      escrever_resumo(hist)
      fechar(exp_id)

    candidatos = []
    for exp_id, mudanca, descricao in ETAPA_LR:
      cfg = {**melhor_cfg, **mudanca}
      met = rodar_experimento(exp_id, cfg, False, descricao)
      if met is None:
        hist.append({"id": exp_id, "descricao": descricao, "situacao": "erro"})
        continue
      ok, soma = supera(met["agregado_validacao"], melhor)
      candidatos.append((ok, soma, exp_id, cfg, met))
      hist.append({"id": exp_id, "descricao": descricao, "metricas": met["agregado_validacao"], "situacao": "candidata", "tempo": tempo(met)})
    passaram = sorted([c for c in candidatos if c[0]], key=lambda c: -c[1])
    for h in hist:
      if h["situacao"] == "candidata":
        h["situacao"] = "sim" if passaram and h["id"] == passaram[0][2] else "não"
    if passaram:
      _, _, melhor_id, melhor_cfg, met = passaram[0]
      melhor = met["agregado_validacao"]
    escrever_resumo(hist)
    for exp_id, _, _ in ETAPA_LR:
      fechar(exp_id)

    # modelo final: teste, sementes, experimento de 3 classes
    log(f"modelo final: {melhor_id}")
    testar_depois(melhor_id, melhor_cfg)
    execucoes = [(melhor_id, melhor_cfg)]
    for s in SEMENTES_EXTRAS:
      cfg = {**melhor_cfg, "semente": s}
      exp_id = f"09_final_semente_{s}"
      if rodar_experimento(exp_id, cfg, True, f"modelo final, semente {s}") is not None:
        execucoes.append((exp_id, cfg))
      fechar(exp_id)
    estab = {}
    for parte, arq in (("validacao", "metricas_validacao.json"), ("teste", "metricas_teste.json")):
      vals = [ler_json(caminho(e, arq)) for e, _ in execucoes]
      vals = [v["agregado_validacao"] if parte == "validacao" else v["agregado_56_videos"] for v in vals]
      estab[parte] = {m: {"media": float(np.mean([v[m] for v in vals])), "desvio": float(np.std([v[m] for v in vals]))} for m in METRICAS}
    estab["execucoes"] = [e for e, _ in execucoes]
    gravar_json(caminho("estabilidade.json"), estab)

    cfg3 = {**melhor_cfg, "rotulos": "anotados_3_classes", "subamostrar_fundo": False}
    met3 = rodar_experimento("10_anotados_3_classes", cfg3, True, "modelo final só com janelas anotadas, 3 classes, sem fundo")
    fechar("10_anotados_3_classes")
    anot = None
    if met3 is not None:
      t3 = ler_json(caminho("10_anotados_3_classes", "metricas_teste.json"))
      tcc = ler_json(RAIZ / "data/results/analises_posteriores.json")["analise_4_ablacoes_de_inferencia"]["ablacoes"]["shuffle"][
        "macro_f1_quadro_anotados_media_entre_folds"]["rodada_a_forcado_3_classes"]["sem_ablacao"]
      v3 = lambda x: f"{x:.3f}".replace(".", ",")
      anot = (f"Só janelas anotadas, 3 classes (teste): macro F1 por quadro nos quadros anotados "
              f"{v3(t3['media_entre_particoes']['macro_f1_quadro_anotados'])} (média entre partições) e "
              f"{v3(t3['agregado_56_videos']['macro_f1_quadro_anotados'])} (agregado), contra {v3(tcc)} no TCC "
              "(média entre partições, predição restrita às 3 classes).")

    tcc_a = ler_json(RAIZ / "data/results/comparacao_rodada_a/metricas.json")
    bal_tcc = ler_json(RAIZ / "data/results/analises_posteriores.json")["analise_3_binario_estereotipia_vs_fundo"][
      "acuracia_balanceada_e_referencia"]["rodada_a"]["acuracia_balanceada_pooled_56_videos"]
    tcc_teste = {"f1_evento_iou_0.3": tcc_a["por_evento"]["regra_principal"]["por_iou"]["0.3"]["pooled_56_videos"]["macro_f1"],
                 "acuracia_balanceada_presenca": bal_tcc,
                 "macro_f1_quadro_todos": tcc_a["por_quadro"]["todos_os_quadros"]["pooled_56_videos"]["macro_f1_com_background"],
                 "macro_f1_quadro_anotados": tcc_a["por_quadro"]["somente_quadros_anotados"]["pooled_56_videos"]["macro_f1_sem_background"]}
    testes = [("TCC, rótulos por intervalo", tcc_teste),
              ("ponto de partida", ler_json(caminho("01_ponto_de_partida", "metricas_teste.json"))["agregado_56_videos"]),
              (f"modelo final ({melhor_id})", ler_json(caminho(melhor_id, "metricas_teste.json"))["agregado_56_videos"])]
    nao = [f"{h['id']}: {h['situacao']}" for h in hist if h["situacao"].startswith(("não rodou", "erro"))]
    escrever_resumo(hist, {"cfg": melhor_cfg, "testes": testes, "estabilidade": estab, "anotados": anot, "nao_rodou": nao})
    commit("experimentos: resumo final", [caminho("resumo.md"), caminho("estabilidade.json"), Path(__file__)])
    log("fim")
  except Exception:
    log("ERRO GERAL:\n" + traceback.format_exc())
    raise
  finally:
    caminho("rodando.pid").unlink(missing_ok=True)


if __name__ == "__main__":
  main()
