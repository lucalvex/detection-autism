# ============================================================
# ARQUIVO: experimentos/rodar_rodada3.py
#
# O QUE FAZ: rodada 3 (experimentos/protocolo_rodada3.md): interpolação
# com limite de alcance (I1) e com a confiança na entrada (I2), contra
# o modelo final da rodada 2 (r2_C4_aumento). Se alguma variante passar
# no critério, treina as sementes 1000 e 2000, escolhe o pós-processamento
# e faz a avaliação final. Em qualquer caso, grava a tabela por ponto.
#
# Reaproveita o treino, as métricas e a avaliação de rodar_rodada2.py e
# rodar_tudo.py (importados, não alterados): as funções que mudam são
# substituídas no módulo importado ao carregar este arquivo, o que vale
# também nos processos filhos (o spawn reexecuta este arquivo).
# Correções da auditoria: commit que falha tira os arquivos do índice;
# resultado gravado só é reaproveitado se for do mesmo modelo.
#
# Uso (da raiz do repositório):
#   .venv/Scripts/python.exe experimentos/rodar_rodada3.py
#   .venv/Scripts/python.exe experimentos/rodar_rodada3.py --teste --saida <pasta>
# ============================================================

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(RAIZ / "experimentos"), str(RAIZ / "src")]
import rodar_tudo as r  # noqa: E402
import rodar_rodada2 as q  # noqa: E402

EXP1 = RAIZ / "experimentos"
O3 = {"saida": str(EXP1), "teste": False}
REF = "r2_C4_aumento"
REF_SEMENTES = {42: "r2_C4_aumento", 1000: "r2_final_semente_1000", 2000: "r2_final_semente_2000"}
ALCANCE_S = 0.5
NORMALIZACAO_PONTOS = [5, 6, 11, 12]
# o pós-processamento e a avaliação da rodada 2 gravam em pastas de nome fixo
RENOMEIA = {"r2_D_posprocessamento": "r3_D_posprocessamento", "r2_avaliacao_final": "r3_avaliacao_final"}


# ---------- caminhos, log e commit ----------

def pasta(i):
  i = RENOMEIA.get(i, i)
  return Path(O3["saida"]) / i if i.startswith("r3_") else EXP1 / i


def log(msg):
  linha = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
  print(linha, flush=True)
  with open(Path(O3["saida"]) / "progresso_rodada3.log", "a", encoding="utf-8") as f:
    f.write(linha + "\n")


def commit(msg, arquivos):
  """Commit só dos arquivos pedidos; se falhar, nada fica preparado no índice."""
  if O3["teste"]:
    return
  arquivos = [str(Path(a).relative_to(RAIZ)) for a in arquivos if Path(a).exists()]
  if not arquivos:
    return
  for tentativa in range(3):  # um comando git seu no mesmo instante trava o índice por pouco tempo
    a = subprocess.run(["git", "add", "-f", "--", *arquivos], capture_output=True, text=True)
    c = a.returncode == 0 and subprocess.run(
      ["git", "commit", "-m", f"{msg}\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>", "--", *arquivos],
      capture_output=True, text=True)
    if c and c.returncode == 0:
      log(f"commit: {msg}")
      return
    subprocess.run(["git", "reset", "-q", "--", *arquivos], capture_output=True)
    erro = (a.stderr if a.returncode else (c.stdout + c.stderr)).strip()
    if "nothing to commit" in erro or "nada a submeter" in erro or "no changes added" in erro:
      return
    time.sleep(10)
  log(f"COMMIT FALHOU (índice limpo): {msg}: {erro[-300:]}")


q.pasta, q.log, q.commit = pasta, log, commit


# ---------- I1/I2: pose com interpolação limitada e marcas de presença ----------

_preparar_original, _amostras_original, _janelas_original = r.preparar_video, r.amostras_e_janelas_treino, r.janelas_avaliacao
_aumentar_original, _ablar_original, _n_entrada_original, _transformar_original = q.aumentar, q.ablar, q.n_entrada, q.transformar


def interpolacao_limitada(quadros, xy, conf, fps):
  """Interpola cada ponto de confiança < 0,5 só se houver quadro confiável a até 0,5 s dos dois lados.
  Devolve xy, a máscara de presença (n, 17) e a de interpolados."""
  xy = xy.astype(np.float64).copy()
  presente = conf >= r.LIMIAR_CONF
  interp = np.zeros_like(presente)
  for k in range(17):
    ok = presente[:, k]
    if ok.all() or not ok.any():
      continue
    qv, faltam = quadros[ok], np.where(~ok)[0]
    pos = np.searchsorted(qv, quadros[faltam])
    tem = (pos > 0) & (pos < len(qv))
    antes = qv[np.clip(pos - 1, 0, len(qv) - 1)]
    depois = qv[np.clip(pos, 0, len(qv) - 1)]
    alcanca = tem & ((quadros[faltam] - antes) / fps <= ALCANCE_S) & ((depois - quadros[faltam]) / fps <= ALCANCE_S)
    idx = faltam[alcanca]
    for c in (0, 1):
      xy[idx, k, c] = np.interp(quadros[idx], qv, xy[ok, k, c])
    interp[idx, k] = True
  return xy, presente | interp, interp


def preparar_video(vid, cfg):
  if cfg["normalizacao"] != "robusta_limitada":
    return _preparar_original(vid, cfg)
  import evaluate_temporal as et
  d = np.load(r.caminho("poses", cfg["pose"], f"{vid}.npz"))
  quadros, xy, conf = d["frames"].astype(np.int64), d["xy"].astype(np.float32), d["conf"].astype(np.float32)
  if len(quadros) == 0:
    return None
  fps = et.load_video_fps(vid)
  n_quadros = int(quadros[-1]) + 1
  xy, m, _ = interpolacao_limitada(quadros, xy, conf, fps)
  if cfg.get("descarte") and (cfg.get("descarte_em") == "todos" or vid in _PAPEL_TREINO):
    confiavel = conf[:, NORMALIZACAO_PONTOS] >= r.LIMIAR_CONF
    fica = mantem_quadro(confiavel, cfg["descarte"])
    m[:, NORMALIZACAO_PONTOS] = confiavel  # I3: ombros e quadris sem interpolação
  else:
    fica = m[:, NORMALIZACAO_PONTOS].all(1)  # I1: sem ombros e quadris recuperáveis, quadro sem pessoa
  quadros, xy, m, conf = quadros[fica], xy[fica], m[fica], conf[fica]
  if len(quadros) == 0:
    return None
  centro = lambda a, b: (xy[:, a] * m[:, a, None] + xy[:, b] * m[:, b, None]) / (m[:, a, None].astype(float) + m[:, b, None])
  quadril = centro(11, 12)  # com os dois pontos presentes, é a média simples, como na I1
  escala = np.linalg.norm(centro(5, 6) - quadril, axis=1)
  ruim = escala < 1
  if ruim.all():
    escala[:] = 1
  elif ruim.any():
    idx = np.maximum.accumulate(np.where(ruim, -1, np.arange(len(escala))))
    escala = escala[np.where(idx < 0, np.argmax(~ruim), idx)]
  P = np.clip((xy - quadril[:, None]) / escala[:, None, None], -r.LIMITE_NORM, r.LIMITE_NORM).astype(np.float32)
  P[~m] = 0
  gt, _ = et.ground_truth_frames(vid, n_quadros, fps)
  return {"quadros": quadros, "P": P, "M": m, "C": conf, "fps": fps,
          "rotulo": np.array([r.CLASSES.index(g) for g in gt], np.int64), "com_conf": cfg["entrada"].endswith("_conf")}


_PAPEL_TREINO = set()  # vídeos de treino da partição deste processo (I3-treino)


def mantem_quadro(confiavel, criterio):
  """confiavel: (n, 4) para os pontos 5, 6, 11, 12. (a) os quatro; (b) um ombro e um quadril."""
  return confiavel.all(1) if criterio == "a" else confiavel[:, :2].any(1) & confiavel[:, 2:].any(1)


_treinar_original = q.treinar_lstm


def treinar_lstm(args):
  """Marca os vídeos de treino da partição antes de treinar: a I3-treino descarta quadros só neles."""
  exp_dir, cfg, k, val_videos, com_teste, treino, opc = args
  q.aplicar_opcoes(*opc)
  _PAPEL_TREINO.clear()
  _PAPEL_TREINO.update(treino or [v for v in r.videos() if r.PARTICOES[v] != k and v not in val_videos])
  return _treinar_original(args)


def amostrar(tempos, t, v, seg):
  """Posições, presença e confiança nos tempos pedidos: linear entre as duas linhas vizinhas;
  o ponto só está presente se estiver nas duas."""
  P, M, C = v["P"][seg], v["M"][seg], v["C"][seg]
  forma = tempos.shape
  tt = tempos.ravel()
  i0 = np.clip(np.searchsorted(t, tt, side="right") - 1, 0, len(t) - 1)
  i1 = np.minimum(i0 + 1, len(t) - 1)
  dt = t[i1] - t[i0]
  w = np.clip(np.where(dt > 0, (tt - t[i0]) / np.where(dt > 0, dt, 1), 0), 0, 1)
  pres = M[i0] & (M[i1] | (w[:, None] < 1e-9))
  Pg = P[i0] * (1 - w)[:, None, None] + P[i1] * w[:, None, None]
  Pg[~pres] = 0
  Cg = C[i0] * (1 - w)[:, None] + C[i1] * w[:, None]
  return Pg.reshape(*forma, 17, 2), pres.reshape(*forma, 17), Cg.reshape(*forma, 17)


def montar(Pg, Mg, Cg, primeiro_valido, com_conf):
  """Atributos por amostra: x, y, dx, dy por ponto (68), as 17 marcas e, na I2, as 17 confianças.
  Eixo do tempo: o penúltimo de Pg sem os pontos (..., L, 17, 2); devolve L-1 amostras (a primeira só serve para dx)."""
  D = np.diff(Pg, axis=-3) * (Mg[..., 1:, :] & Mg[..., :-1, :])[..., None]
  D[..., 0, :, :] *= primeiro_valido[..., None, None]
  partes = [np.concatenate([Pg[..., 1:, :, :], D], -1).reshape(*Pg.shape[:-3], Pg.shape[-3] - 1, 68), Mg[..., 1:, :].astype(np.float32)]
  if com_conf:
    partes.append(Cg[..., 1:, :])
  return np.concatenate(partes, -1).astype(np.float32)


def amostras_e_janelas_treino(v, cfg, n_classes):
  if "M" not in v:
    return _amostras_original(v, cfg, n_classes)
  W = cfg["janela"]
  passo = W if cfg["passo_treino"] == "janela" else int(cfg["passo_treino"])
  blocos, qs, inicios, base = [], [], [], 0
  for seg in r.segmentos(v["quadros"], v["fps"]):
    t = v["quadros"][seg] / v["fps"]
    g = t[0] + np.arange(int(np.floor((t[-1] - t[0]) * cfg["fps_alvo"] + 1e-9)) + 1) / cfg["fps_alvo"]
    g_ant = np.concatenate([[g[0]], g])  # a primeira amostra repete o início: dx = 0, como na rodada 2
    Pg, Mg, Cg = amostrar(g_ant, t, v, seg)
    blocos.append(montar(Pg, Mg, Cg, np.zeros(()), v["com_conf"]))
    qs.append(np.clip(np.round(g * v["fps"]).astype(np.int64), 0, len(v["rotulo"]) - 1))
    inicios.append(base + np.arange(0, len(g) - W + 1, passo))
    base += len(g)
  largura = 68 + 17 * (2 if v["com_conf"] else 1)
  if not blocos:
    return np.zeros((0, largura), np.float32), np.zeros(0, np.int64), np.zeros(0, np.int64)
  A, quadro_da_amostra, inicios = np.concatenate(blocos), np.concatenate(qs), np.concatenate(inicios)
  if len(inicios) == 0:
    return A, inicios, inicios
  rot = r.rotular(v["rotulo"][quadro_da_amostra[inicios[:, None] + np.arange(W)]], W, n_classes)
  return A, inicios[rot >= 0], rot[rot >= 0]


def janelas_avaliacao(v, cfg):
  if "M" not in v:
    return _janelas_original(v, cfg)
  W, q_ = cfg["janela"], v["quadros"]
  Xs, ini, fim = [], [], []
  for seg in r.segmentos(q_, v["fps"]):
    t = q_[seg] / v["fps"]
    T = t[:, None] - (W - np.arange(W + 1))[None, :] / cfg["fps_alvo"]
    ok = T[:, 1] >= t[0] - 1e-9
    if not ok.any():
      continue
    T = T[ok]
    Pg, Mg, Cg = amostrar(T, t, v, seg)
    Xs.append(montar(Pg, Mg, Cg, (T[:, 0] >= t[0] - 1e-9).astype(np.float64), v["com_conf"]))
    ini.append(np.maximum(q_[seg][0], np.round(T[:, 1] * v["fps"]).astype(np.int64)))
    fim.append(q_[seg][ok] + 1)
  if not Xs:
    return None
  return np.concatenate(Xs), np.concatenate(ini), np.concatenate(fim)


def aumentar(X, g, cfg):
  """O aumento da rodada 2, estendido às marcas (e confianças): mesma troca esquerda/direita,
  mesma reamostragem no tempo, e pontos ausentes mantidos em 0."""
  if X.shape[-1] == 68:
    return _aumentar_original(X, g, cfg)
  import math
  import torch
  B, W, F = X.shape
  E = (F - 68) // 17
  Y = X[..., :68].reshape(B, W, 17, 4)
  p, d = Y[..., :2], Y[..., 2:]
  ext = X[..., 68:].reshape(B, W, E, 17)
  s = 0.8 + 0.4 * torch.rand(B, generator=g)
  c = (W - 1) / 2
  # shortcut: acelerar prende as bordas (até 10% das amostras de cada lado repetidas); para corrigir, buscar janelas 20% maiores no treino
  tau = (c + (torch.arange(W)[None, :] - c) * s[:, None]).clamp(0, W - 1)
  i0 = tau.floor().long()
  i1 = (i0 + 1).clamp(max=W - 1)
  w = tau - i0
  pega = lambda a, i: torch.gather(a, 1, i.view(B, W, *([1] * (a.dim() - 2))).expand(B, W, *a.shape[2:]))
  m = ext[:, :, 0] > 0.5
  m2 = pega(m, i0) & (pega(m, i1) | (w < 1e-9)[..., None])
  p = pega(p, i0) * (1 - w)[..., None, None] + pega(p, i1) * w[..., None, None]
  p = p * m2[..., None]
  d2 = torch.zeros_like(d)
  d2[:, 1:] = (p[:, 1:] - p[:, :-1]) * (m2[:, 1:] & m2[:, :-1])[..., None]
  d2[:, 0] = d[:, 0] * s[:, None, None] * m2[:, 0, :, None]
  d = d2
  ext = pega(ext, i0) * (1 - w)[..., None, None] + pega(ext, i1) * w[..., None, None]
  ext[:, :, 0] = m2.float()
  perm = list(range(17))
  for a, b in q.PARES:
    perm[a], perm[b] = b, a
  sinal = torch.tensor([-1.0, 1.0])
  esp = (torch.rand(B, generator=g) < 0.5)
  p = torch.where(esp[:, None, None, None], p[:, :, perm] * sinal, p)
  d = torch.where(esp[:, None, None, None], d[:, :, perm] * sinal, d)
  ext = torch.where(esp[:, None, None, None], ext[..., perm], ext)
  e = (0.9 + 0.2 * torch.rand(B, generator=g))[:, None, None, None]
  th = (torch.rand(B, generator=g) * 20 - 10) * math.pi / 180
  cs, sn = torch.cos(th)[:, None, None], torch.sin(th)[:, None, None]
  rot = lambda v_: torch.stack([v_[..., 0] * cs - v_[..., 1] * sn, v_[..., 0] * sn + v_[..., 1] * cs], -1)
  p, d = (rot(p) * e).clamp(-r.LIMITE_NORM, r.LIMITE_NORM), rot(d) * e
  return torch.cat([torch.cat([p, d], -1).reshape(B, W, 68), ext.reshape(B, W, E * 17)], -1)


def ablar(X, ablacao):
  if ablacao == "dxdy" and X.shape[-1] > 68:  # zera dx, dy; mantém marcas e confianças
    X = X.copy()
    X[..., :68] = _ablar_original(np.ascontiguousarray(X[..., :68]), "dxdy")
    return X
  return _ablar_original(X, ablacao)


def n_entrada(cfg):
  return {"xy_dxdy_marcas": 85, "xy_dxdy_marcas_conf": 102}.get(cfg["entrada"]) or _n_entrada_original(cfg)


def transformar(X, cfg):
  return X if cfg["entrada"].startswith("xy_dxdy_marcas") else _transformar_original(X, cfg)


r.preparar_video, r.amostras_e_janelas_treino, r.janelas_avaliacao = preparar_video, amostras_e_janelas_treino, janelas_avaliacao
q.aumentar, q.ablar, q.n_entrada, q.transformar, q.treinar_lstm = aumentar, ablar, n_entrada, transformar, treinar_lstm


def autoverificacao():
  """Checagens rápidas das funções novas (falham alto se alguma regra quebrar)."""
  import torch
  quadros = np.arange(10)
  conf = np.ones((10, 17), np.float32)
  conf[3:5, 0] = 0.1        # lacuna curta: interpolada
  conf[:, 1] = 0.1          # nunca confiável: ausente
  conf[8:, 2] = 0.1         # ponta do vídeo: não repete
  xy = np.tile(np.arange(10, dtype=np.float32)[:, None, None], (1, 17, 2))
  xy2, m, interp = interpolacao_limitada(quadros, xy, conf, fps=10)
  assert interp[3:5, 0].all() and np.allclose(xy2[3:5, 0, 0], [3, 4]), "lacuna curta deveria ser interpolada"
  assert not m[:, 1].any() and not m[8:, 2].any(), "ponto nunca confiável e ponta do vídeo deveriam ficar ausentes"
  q15 = np.arange(15)
  conf2 = np.ones((15, 17), np.float32)
  conf2[1:14, 0] = 0.1      # confiáveis só em 0 e 14 (1,4 s a 10 quadros/s): nenhum quadro fica a até 0,5 s dos dois lados
  xy15 = np.zeros((15, 17, 2), np.float32)
  assert not interpolacao_limitada(q15, xy15, conf2, fps=10)[1][1:14, 0].any(), "lacuna longa não deveria ser interpolada"
  X = torch.randn(4, 30, 85)
  X[..., 68:] = (torch.rand(4, 30, 17) > 0.3).float()
  Xa = aumentar(X.clone(), torch.Generator().manual_seed(0), {"normalizacao": "robusta"})
  pres = Xa[..., 68:] > 0.5
  assert (Xa[..., :68].reshape(4, 30, 17, 4)[..., :2].abs().sum(-1)[~pres] == 0).all(), "ponto ausente deveria ficar em 0 no aumento"


# ---------- execução ----------

def cfg_limpo(c):
  return {k: v for k, v in c.items() if k not in ("descricao", "teste_calculado", "hiper")}


def rodar3(exp_id, cfg, descricao, com_teste=False):
  """q.rodar, mas só reaproveita resultado gravado com a mesma configuração."""
  p = pasta(exp_id) / "config.json"
  if (pasta(exp_id) / "metricas_validacao.json").exists() and cfg_limpo(r.ler_json(p)) != cfg_limpo(cfg):
    log(f"{exp_id}: ERRO: há resultado gravado com outra configuração; não reaproveitado nem sobrescrito")
    return None
  return q.rodar(exp_id, cfg, descricao, com_teste=com_teste)


def gravado_para(caminho_json, chave, modelo):
  """Resultado gravado de outro modelo final: erro (não reaproveita)."""
  if caminho_json.exists() and r.ler_json(caminho_json).get(chave) != modelo:
    raise RuntimeError(f"{caminho_json} foi calculado para outro modelo final; apague-o para recalcular")


def tabela_pontos():
  pdir = pasta("r3_tabela_pontos")
  if (pdir / "tabela_por_ponto.json").exists():
    return r.ler_json(pdir / "tabela_por_ponto.json")
  import evaluate_temporal as et
  n = 0
  cont = {k: np.zeros(17) for k in ("atual_interp", "atual_bruto", "i1_interp", "i1_ausente", "i1_remove")}
  por_classe = {}
  for vid in r.videos():
    d = np.load(r.caminho("poses", "yolo11n", f"{vid}.npz"))
    quadros, xy, conf = d["frames"].astype(np.int64), d["xy"], d["conf"]
    baixa = conf < r.LIMIAR_CONF
    nunca = ~(~baixa).any(0)
    cont["atual_interp"] += (baixa & ~nunca).sum(0)
    cont["atual_bruto"] += (baixa & nunca).sum(0)
    _, m, interp = interpolacao_limitada(quadros, xy, conf, et.load_video_fps(vid))
    cont["i1_interp"] += interp.sum(0)
    cont["i1_ausente"] += (~m).sum(0)
    cont["i1_remove"][NORMALIZACAO_PONTOS] += (~m[:, NORMALIZACAO_PONTOS]).sum(0)
    rem = (~m[:, NORMALIZACAO_PONTOS].all(1)).sum()
    c = por_classe.setdefault(r.classe_do_video(vid), [0, 0])
    c[0] += int(rem)
    c[1] += len(quadros)
    n += len(quadros)
  frac = {k: [float(x / n) for x in v] for k, v in cont.items()}
  total_rem = sum(c[0] for c in por_classe.values())
  out = {"quadros_com_pessoa": n, "por_ponto": frac,
         "quadros_removidos": {"total": total_rem, "fracao": total_rem / n,
                               "por_classe": {c: {"removidos": a, "quadros": b, "fracao": a / b} for c, (a, b) in por_classe.items()}},
         "definicoes": {"atual_interp": "conf < 0,5 e o ponto tem algum quadro confiável no vídeo (interpolado sem limite ou repetido na ponta)",
                        "atual_bruto": "conf < 0,5 e o ponto nunca é confiável no vídeo (fica a estimativa bruta)",
                        "i1_interp": "conf < 0,5 e quadro confiável a até 0,5 s dos dois lados (interpolado)",
                        "i1_ausente": "conf < 0,5 sem quadro confiável a até 0,5 s dos dois lados (0 + marca de ausente)",
                        "i1_remove": "quadro removido por este ponto estar ausente (só 5, 6, 11, 12; um quadro pode ter mais de um)",
                        "denominador": "quadros com pessoa detectada pela YOLO11n, antes da remoção"}}
  r.gravar_json(pdir / "tabela_por_ponto.json", out)
  f3 = q.f3
  nomes = ["nariz", "olho esq.", "olho dir.", "orelha esq.", "orelha dir.", "ombro esq.", "ombro dir.", "cotovelo esq.",
           "cotovelo dir.", "punho esq.", "punho dir.", "quadril esq.", "quadril dir.", "joelho esq.", "joelho dir.",
           "tornozelo esq.", "tornozelo dir."]
  linhas = ["# Pontos interpolados, ausentes e quadros removidos (YOLO11n, frações dos quadros com pessoa)", "",
            "| ponto | atual: interpolado | atual: bruto | I1: interpolado | I1: ausente | I1: quadros removidos |", "|---|---|---|---|---|---|"]
  for k in range(17):
    linhas.append(f"| {k} {nomes[k]} | {f3(frac['atual_interp'][k])} | {f3(frac['atual_bruto'][k])} | {f3(frac['i1_interp'][k])} | "
                  f"{f3(frac['i1_ausente'][k])} | {f3(frac['i1_remove'][k]) if k in NORMALIZACAO_PONTOS else '--'} |")
  linhas += ["", f"Quadros removidos na I1 (algum de 5, 6, 11, 12 ausente): {total_rem} de {n} ({f3(total_rem / n)}); "
             + "; ".join(f"*{c}* {a} de {b} ({f3(a / b)})" for c, (a, b) in por_classe.items()) + ".", "",
             *[f"- {k}: {v}" for k, v in out["definicoes"].items()]]
  (pdir / "tabela_por_ponto.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")
  return out


def relatorio_descarte():
  pdir = pasta("r3_I3_descarte")
  if (pdir / "descarte.json").exists():
    return r.ler_json(pdir / "descarte.json")
  import evaluate_temporal as et
  por_video = {}
  for vid in r.videos():
    d = np.load(r.caminho("poses", "yolo11n", f"{vid}.npz"))
    quadros, conf = d["frames"].astype(np.int64), d["conf"]
    gt, _ = et.ground_truth_frames(vid, int(quadros[-1]) + 1, et.load_video_fps(vid))
    anot = np.array([gt[f] != "Background" for f in quadros])
    confiavel = conf[:, NORMALIZACAO_PONTOS] >= r.LIMIAR_CONF
    por_video[vid] = {}
    for c in ("a", "b"):
      sai = ~mantem_quadro(confiavel, c)
      por_video[vid][c] = np.array([len(quadros), sai.sum(), anot.sum(), (sai & anot).sum()])
  val = q.validacao()
  out = {"criterios": {}, "definicao": "quadros com pessoa (YOLO11n); anotados = dentro de intervalo do XML"}
  for c in ("a", "b"):
    out["criterios"][c] = {}
    for parte, listas in (("treino", [[v for v in r.videos() if r.PARTICOES[v] != k and v not in val[k]] for k in r.particoes()]),
                          ("teste", [r.videos()])):
      soma = {}
      for vids in listas:
        for v in vids:
          soma[r.classe_do_video(v)] = soma.get(r.classe_do_video(v), 0) + por_video[v][c]
      out["criterios"][c][parte] = {cl: {"quadros": int(x[0]), "descartados": int(x[1]), "fracao_quadros": float(x[1] / x[0]),
                                         "anotados": int(x[2]), "anotados_descartados": int(x[3]), "fracao_anotados": float(x[3] / max(1, x[2]))}
                                    for cl, x in sorted(soma.items())}
  r.gravar_json(pdir / "descarte.json", out)
  linhas = ["# Descarte por qualidade da pose (I3)", "", "(a) os dois ombros e os dois quadris com confiança ≥ 0,5; (b) pelo menos um ombro e um quadril.", "",
            "| critério | parte | classe | quadros descartados | anotados descartados |", "|---|---|---|---|---|"]
  for c, x in out["criterios"].items():
    for parte, y in x.items():
      for cl, z in y.items():
        linhas.append(f"| ({c}) | {parte} | *{cl}* | {z['descartados']} de {z['quadros']} ({q.f3(z['fracao_quadros'])}) | "
                      f"{z['anotados_descartados']} de {z['anotados']} ({q.f3(z['fracao_anotados'])}) |")
  (pdir / "descarte.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")
  return out


def tcc_vd():
  import evaluate_temporal as et
  et.load_config("data/datasets_annotated", "models/oof_seeded_annotated")
  return et.load_all_videos()


def comparar_pos(pos, final_dirs, final_cfg):
  """F1 por evento com e sem o pós-processamento escolhido: modelo final, r2_C4_aumento (3 sementes) e TCC."""
  p = pasta("r3_avaliacao_final") / "pos_processamento_comparado.json"
  gravado_para(p, "modelo_final", pos["modelo"])
  if p.exists():
    return r.ler_json(p)
  g = pos["grade"][pos["escolhido"]]
  out = {"modelo_final": pos["modelo"], "escolhido": pos["escolhido"], "modelos": {}}
  cfg_ref = cfg_limpo(r.ler_json(pasta(REF) / "config.json"))
  for nome, dirs, cfg in (("modelo_final", final_dirs, final_cfg), (REF, REF_SEMENTES, cfg_ref)):
    for s, d in dirs.items():
      vd = r.dados_avaliacao(pasta(d), "teste", cfg if s == 42 else {**cfg, "semente": s})
      out["modelos"][f"{nome}_semente_{s}"] = {"sem": q.f1_evento_pos(vd, 0, 0), "com": q.f1_evento_pos(vd, g["mediana_s"], g["juncao_s"])}
  vd = tcc_vd()
  out["modelos"]["TCC"] = {"sem": q.f1_evento_pos(vd, 0, 0), "com": q.f1_evento_pos(vd, g["mediana_s"], g["juncao_s"])}
  r.gravar_json(p, out)
  return out


def escrever_resumo(linhas_val, decisao, final=None, tabela=None, descarte=None):
  linhas = ["# Resumo da rodada 3", "",
            "Protocolo: `protocolo_rodada3.md`. Cada célula: agregado / média ± desvio entre partições (validação). "
            "Decisão pelo agregado: mantém se melhora ≥ 0,01 em pelo menos 2 das 3 métricas.", "",
            "| variante | mudança | F1 evento IoU 0,3 | acurácia bal. | macro F1 quadros | passa | tempo |", "|---|---|---|---|---|---|---|",
            *linhas_val, "", f"**Decisão:** {decisao}"]
  if tabela:
    t = tabela["quadros_removidos"]
    linhas += ["", f"Quadros removidos na I1 (ombro ou quadril ausente): {t['total']} de {tabela['quadros_com_pessoa']} ({q.f3(t['fracao'])}); "
               + "; ".join(f"*{c}* {q.f3(x['fracao'])}" for c, x in t["por_classe"].items()) + ". Tabela por ponto: `r3_tabela_pontos/tabela_por_ponto.md`."]
  if descarte:
    linhas += ["", "Descarte da I3 (fração dos quadros com pessoa / dos quadros anotados; treino = vídeos de treino das 5 partições, teste = os 56 vídeos):"]
    for c, x in descarte["criterios"].items():
      linhas.append(f"- ({c}) " + "; ".join(f"{parte}: " + ", ".join(f"*{cl}* {q.f3(y['fracao_quadros'])} / {q.f3(y['fracao_anotados'])}" for cl, y in x[parte].items())
                                            for parte in ("treino", "teste")))
    linhas.append("Detalhes: `r3_I3_descarte/descarte.md`.")
  if final:
    linhas += ["", *final]
  (Path(O3["saida"]) / "resumo_rodada3.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")


def bloco_final(final_id, av, comp):
  M4 = q.METRICAS + ["macro_f1_quadro_anotados"]
  cel = lambda t, m: f"{q.f3(t['agregado_56_videos'][m])} / {q.f3(t['media_entre_particoes'][m])} ± {q.f3(t['desvio_entre_particoes'][m])}"
  ref = r.ler_json(EXP1 / "r2_avaliacao_final" / "avaliacao_final.json")
  linhas = [f"## Novo modelo final: {final_id}", "", "Teste (56 vídeos), agregado / média ± desvio entre partições; as métricas por quadro não dependem do pós-processamento.", "",
            "| | F1 evento | acurácia bal. | macro F1 quadros | macro F1 anotados |", "|---|---|---|---|---|"]
  for nome, a in ((REF, ref), (final_id, av)):
    s42 = a["sementes"].get("42") or a["sementes"].get(42)
    linhas.append(f"| {nome}, semente 42 | {' | '.join(cel(s42['sem_ablacao']['metricas'], m) for m in M4)} |")
    linhas.append(f"| {nome}, 3 sementes (agregado) | " + " | ".join(
      f"{q.f3(a['entre_sementes']['agregado_56_videos'][m]['media'])} ± {q.f3(a['entre_sementes']['agregado_56_videos'][m]['desvio'])}" for m in M4) + " |")
  g = comp["escolhido"]
  ev = lambda pref: [comp["modelos"][k] for k in comp["modelos"] if k.startswith(pref)]
  linhas += ["", f"F1 por evento com o pós-processamento escolhido ({g}), agregado (média ± desvio entre sementes) | média entre partições:"]
  for nome, pref in ((final_id, "modelo_final"), (REF, REF), ("TCC", "TCC")):
    xs = ev(pref)
    linhas.append(f"- {nome}: sem {q.f3(np.mean([x['sem']['agregado'] for x in xs]))} ± {q.f3(np.std([x['sem']['agregado'] for x in xs]))}, "
                  f"com {q.f3(np.mean([x['com']['agregado'] for x in xs]))} ± {q.f3(np.std([x['com']['agregado'] for x in xs]))} | "
                  f"com {q.f3(np.mean([x['com']['media'] for x in xs]))}")
  sem = av["sementes"]
  linhas.append("Ablações, macro F1 anotados com 3 classes (agregado / média entre partições), por semente: " + "; ".join(
    f"{a}: " + ", ".join(f"{q.f3(sem[s][a]['macro_f1_anotados_3_classes']['agregado'])} / {q.f3(sem[s][a]['macro_f1_anotados_3_classes']['media'])}" for s in sem)
    for a in ("sem_ablacao", "ordem", "dxdy")) + ".")
  s42 = sem.get("42") or sem.get(42)
  linhas.append("Fração da classe do vídeo, dentro / fora (semente 42): " + "; ".join(
    f"*{c}* {q.f3(x['dentro'])} / {q.f3(x['fora'])}" for c, x in s42["sem_ablacao"]["fracao_classe_do_video"].items()) + ".")
  linhas.append("*Head banging* com F1 zero: " + ", ".join(f"{sem[s]['head_banging_f1_zero']['n_zero']} de {sem[s]['head_banging_f1_zero']['n_videos']}" for s in sem)
                + f". Por classe e matriz de confusão: `r3_avaliacao_final/avaliacao_final.json`; curva de erro: `{final_id}/curva_erro.pdf`.")
  return linhas


def ja_rodando():
  p = Path(O3["saida"]) / "rodando_rodada3.pid"
  if p.exists():
    pid = p.read_text().strip()
    res = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True)
    if pid and pid in res.stdout and "python" in res.stdout.lower():
      return True
  p.write_text(str(os.getpid()))
  return False


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--teste", action="store_true", help="rodada rápida: 1 partição, 2 épocas, 9 vídeos, sem commit; força o caminho da avaliação")
  ap.add_argument("--saida", default=str(EXP1))
  args = ap.parse_args()
  O3.update({"teste": args.teste, "saida": str(Path(args.saida).resolve())})
  Path(O3["saida"]).mkdir(parents=True, exist_ok=True)
  if args.teste:
    div1 = r.divisao_validacao()[1]
    esc = []
    for c in r.CLASSES[:3]:
      esc += [next(v for v in div1 if r.classe_do_video(v) == c),
              next(v for v in sorted(r.PARTICOES) if r.classe_do_video(v) == c and r.PARTICOES[v] != 1 and v not in div1),
              next(v for v in sorted(r.PARTICOES) if r.classe_do_video(v) == c and r.PARTICOES[v] == 1)]
    r.OPC.update({"videos": esc, "particoes": [1]})
  if ja_rodando():
    print("Já há uma execução da rodada 3 em andamento.")
    return
  log(f"início (pid {os.getpid()}){' -- TESTE' if args.teste else ''}")
  try:
    autoverificacao()
    tabela = tabela_pontos()
    commit("experimentos: r3_tabela_pontos", [pasta("r3_tabela_pontos") / "tabela_por_ponto.json", pasta("r3_tabela_pontos") / "tabela_por_ponto.md", Path(__file__)])
    descarte = relatorio_descarte()
    commit("experimentos: r3_I3_descarte", [pasta("r3_I3_descarte") / "descarte.json", pasta("r3_I3_descarte") / "descarte.md", Path(__file__)])

    cfg_ref = {**q.NOVOS, **cfg_limpo(r.ler_json(pasta(REF) / "config.json"))}
    if args.teste:
      cfg_ref["max_epocas"] = 2
    met_ref = q.met_val_lidas(REF)
    linhas = [f"| {REF} | referência (rodada 2) | {' | '.join(q.duas(met_ref, m) for m in q.METRICAS)} | -- | -- |"]
    passou = []
    for exp_id, mud, desc in [("r3_I1_interpolacao_limitada", {"normalizacao": "robusta_limitada", "entrada": "xy_dxdy_marcas"},
                               "interpolação até 0,5 s, ausentes = 0 + marca, sem ombro/quadril = sem pessoa"),
                              ("r3_I2_limitada_confianca", {"normalizacao": "robusta_limitada", "entrada": "xy_dxdy_marcas_conf"},
                               "I1 + confiança de cada ponto")] + [
        (f"r3_I3_{onde}_{c}", {"normalizacao": "robusta_limitada", "entrada": "xy_dxdy_marcas", "descarte": c, "descarte_em": em},
         f"I3-{onde}, critério ({c}): descarta quadro sem {'os dois ombros e os dois quadris' if c == 'a' else 'um ombro e um quadril'} confiáveis"
         + (" só no treino" if em == "treino" else " no treino, na validação e no teste (não concorre)"))
        for onde, em in (("treino", "treino"), ("teste", "todos")) for c in ("a", "b")]:
      cfg = {**cfg_ref, **mud}
      concorre = cfg.get("descarte_em") != "todos"
      met = rodar3(exp_id, cfg, desc)
      if met is None:
        linhas.append(f"| {exp_id} | {desc} | -- | -- | -- | erro | -- |")
      else:
        ok, soma = r.supera(met["agregado_validacao"], met_ref["agregado_validacao"])
        linhas.append(f"| {exp_id} | {desc} | {' | '.join(q.duas(met, m) for m in q.METRICAS)} | "
                      f"{('sim' if ok else 'não') if concorre else 'não concorre'} | {met['tempo_s'] / 60:.0f} min |")
        if concorre and (ok or args.teste):
          passou.append((soma, exp_id, cfg))
      escrever_resumo(linhas, "em andamento", tabela=tabela, descarte=descarte)
      commit(f"experimentos: {exp_id}", q.arquivos(pasta(exp_id)) + [Path(O3["saida"]) / "resumo_rodada3.md", Path(__file__)])

    if not passou:
      escrever_resumo(linhas, f"nenhuma variante (I1, I2, I3-treino a e b) melhora {REF} pelo critério; o modelo final continua {REF}.",
                      tabela=tabela, descarte=descarte)
      commit("experimentos: resumo da rodada 3", [Path(O3["saida"]) / "resumo_rodada3.md", Path(__file__)])
      log("fim: modelo final continua " + REF)
      return

    _, final_id, final_cfg = max(passou)
    log(f"novo modelo final: {final_id}{' (forçado no teste)' if args.teste else ''}")
    q.testar(final_id, final_cfg)
    dirs = {42: final_id}
    for s in (1000, 2000):
      exp_id = f"r3_final_semente_{s}"
      if rodar3(exp_id, {**final_cfg, "semente": s}, f"modelo final, semente {s}", com_teste=True) is not None:
        dirs[s] = exp_id
      commit(f"experimentos: {exp_id}", q.arquivos(pasta(exp_id)) + [Path(__file__)])
    commit(f"experimentos: teste de {final_id}", [pasta(final_id) / "metricas_teste.json"])
    gravado_para(pasta("r3_D_posprocessamento") / "posprocessamento.json", "modelo", final_id)
    pos = q.posprocessamento(final_id, final_cfg)
    commit("experimentos: r3_D_posprocessamento", [pasta("r3_D_posprocessamento") / "posprocessamento.json"])
    gravado_para(pasta("r3_avaliacao_final") / "avaliacao_final.json", "modelo_final", final_id)
    av = q.avaliacao_final(final_id, final_cfg, pos, dirs)
    comp = comparar_pos(pos, dirs, final_cfg)
    escrever_resumo(linhas, f"{final_id} melhora {REF} pelo critério e passa a ser o modelo final.", bloco_final(final_id, av, comp), tabela, descarte)
    commit("experimentos: r3_avaliacao_final e resumo", [pasta("r3_avaliacao_final") / "avaliacao_final.json",
                                                         pasta("r3_avaliacao_final") / "pos_processamento_comparado.json",
                                                         Path(O3["saida"]) / "resumo_rodada3.md", Path(__file__)])
    log("fim")
  except Exception:
    log("ERRO GERAL:\n" + traceback.format_exc())
    raise
  finally:
    (Path(O3["saida"]) / "rodando_rodada3.pid").unlink(missing_ok=True)


if __name__ == "__main__":
  main()
