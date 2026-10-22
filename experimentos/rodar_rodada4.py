# ============================================================
# ARQUIVO: experimentos/rodar_rodada4.py
#
# O QUE FAZ: rodada 4 (experimentos/protocolo_rodada4.md), para reduzir a
# dependência da postura: passo 0 (diagnóstico), P1 (trajetória
# centralizada, com e sem a postura média), P2 (periodicidade junto com o
# LSTM), P3 (aumento de postura e ângulo), a medida de dependência do
# movimento (janelas reais contra congeladas) e a avaliação final, sempre
# nos quadros válidos pelo critério Q e com todos os modelos nos mesmos
# quadros. Reaproveita as rodadas 1 a 3 (importadas, não alteradas).
# Retomável; um commit por etapa; commit que falha não deixa nada no índice.
#
# Uso (da raiz do repositório):
#   .venv/Scripts/python.exe experimentos/rodar_rodada4.py
#   .venv/Scripts/python.exe experimentos/rodar_rodada4.py --teste --saida <pasta>
# ============================================================

import argparse
import csv
import math
import multiprocessing as mp
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(RAIZ / "experimentos"), str(RAIZ / "src")]
import rodar_q as rq  # noqa: E402  (aplica as substituições das rodadas 2 e 3 e as funções do critério Q)

t3, r, q = rq.t3, rq.r, rq.q
EXP1 = rq.EXP1
REF, REF_SEMENTES = rq.REF, rq.REF_SEMENTES
METRICAS = q.METRICAS
CLASSES3 = r.CLASSES[:3]


# ---------- caminhos e log (commit seguro de rodar_rodada3) ----------

def pasta(i):
  return Path(t3.O3["saida"]) / i if i.startswith("r4_") else EXP1 / i


def log(msg):
  linha = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
  print(linha, flush=True)
  with open(Path(t3.O3["saida"]) / "progresso_rodada4.log", "a", encoding="utf-8") as f:
    f.write(linha + "\n")


t3.pasta = q.pasta = pasta
t3.log = q.log = log
commit = t3.commit


# ---------- modelo: trajetória centralizada, postura média, periodicidade ----------

_rede_anterior = q.rede


def rede(cfg, n_classes=4):
  if cfg.get("arquitetura") != "p4":
    return _rede_anterior(cfg, n_classes)
  import torch
  import torch.nn as nn

  class Rede4(nn.Module):
    def __init__(self):
      super().__init__()
      self.lstm = nn.LSTM(68, cfg["oculto"], cfg["camadas"], batch_first=True, dropout=cfg["dropout"] if cfg["camadas"] > 1 else 0.0)
      self.drop = nn.Dropout(cfg["dropout"])
      extra = (34 if cfg.get("concat_postura") else 0) + (36 if cfg.get("concat_periodicidade") else 0)
      self.norm_per = nn.BatchNorm1d(36) if cfg.get("concat_periodicidade") else None
      self.fc = nn.Linear(cfg["oculto"] + extra, n_classes)

    def forward(self, X):  # X: (lote, W, 68) com x, y, dx, dy por ponto
      B, W, _ = X.shape
      Y = X.reshape(B, W, 17, 4)
      media = Y[..., :2].mean(1)  # postura média da janela
      if cfg.get("trajetoria"):
        Y = torch.cat([Y[..., :2] - media[:, None], Y[..., 2:]], -1)
      _, (h, _) = self.lstm(Y.reshape(B, W, 68))
      partes = [self.drop(h[-1])]
      if cfg.get("concat_postura"):
        partes.append(media.reshape(B, 34))
      if cfg.get("concat_periodicidade"):
        f = torch.from_numpy(q.atributos_periodicidade(X.detach().cpu().numpy(), cfg["fps_alvo"]))
        if self.training and B == 1:  # lote de uma janela: a normalização em lote não tem variância; usa a acumulada
          n = self.norm_per
          partes.append(torch.nn.functional.batch_norm(f, n.running_mean, n.running_var, n.weight, n.bias, False, 0.0, n.eps))
        else:
          partes.append(self.norm_per(f))
      return self.fc(torch.cat(partes, 1))
  return Rede4()


q.rede = rede


# ---------- P3: aumento de postura e ângulo ----------

_aumentar_anterior = q.aumentar


def aumentar(X, g, cfg):
  if not cfg.get("aum_rot_graus"):
    return _aumentar_anterior(X, g, cfg)
  import torch
  B, W = X.shape[:2]
  Y = X.reshape(B, W, 17, 4)
  p, d = Y[..., :2], Y[..., 2:]
  s = 0.8 + 0.4 * torch.rand(B, generator=g)
  c = (W - 1) / 2
  # shortcut: acelerar prende as bordas (até 10% das amostras de cada lado repetidas); para corrigir, buscar janelas 20% maiores no treino
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
  for a, b in q.PARES:
    perm[a], perm[b] = b, a
  sinal = torch.tensor([-1.0, 1.0])
  esp = (torch.rand(B, generator=g) < 0.5)[:, None, None, None]
  p = torch.where(esp, p[:, :, perm] * sinal, p)
  d = torch.where(esp, d[:, :, perm] * sinal, d)
  k = cfg["aum_escala_xy"]
  esc = torch.stack([1 + k * (2 * torch.rand(B, generator=g) - 1), 1 + k * (2 * torch.rand(B, generator=g) - 1)], -1)[:, None, None, :]
  th = (2 * torch.rand(B, generator=g) - 1) * cfg["aum_rot_graus"] * math.pi / 180
  cs, sn = torch.cos(th)[:, None, None], torch.sin(th)[:, None, None]
  rot = lambda v: torch.stack([v[..., 0] * cs - v[..., 1] * sn, v[..., 0] * sn + v[..., 1] * cs], -1)
  p, d = rot(p) * esc, rot(d) * esc
  ruido = torch.randn(B, W, 17, 2, generator=g) * cfg["aum_ruido"]
  p = p + ruido
  d = d.clone()
  d[:, 1:] += ruido[:, 1:] - ruido[:, :-1]
  d[:, 0] += ruido[:, 0]
  p = p.clamp(-r.LIMITE_NORM, r.LIMITE_NORM)
  return torch.cat([p, d], -1).reshape(B, W, 68)


q.aumentar = aumentar


# ---------- janelas congeladas ----------

_ablar_anterior = q.ablar


def congelar(X):
  Y = X.reshape(*X.shape[:2], 17, 4).copy()
  Y[..., :2] = Y[..., :2].mean(1, keepdims=True)
  Y[..., 2:] = 0
  return Y.reshape(X.shape)


def ablar(X, ablacao):
  return congelar(X) if ablacao == "congelada" else _ablar_anterior(X, ablacao)


q.ablar = ablar


def autoverificacao():
  import torch
  X = np.random.default_rng(0).normal(size=(3, 30, 68)).astype(np.float32)
  C = congelar(X).reshape(3, 30, 17, 4)
  assert np.allclose(C[..., 2:], 0) and np.allclose(C[:, 0, :, :2], C[:, -1, :, :2]), "janela congelada errada"
  cfg = {"arquitetura": "p4", "oculto": 8, "camadas": 2, "dropout": 0.0, "trajetoria": True, "concat_postura": True,
         "concat_periodicidade": True, "fps_alvo": 15}
  m = rede(cfg)
  m.eval()
  Xt = torch.from_numpy(X)
  base = m(Xt)
  desloc = Xt.clone().reshape(3, 30, 17, 4)
  desloc[..., :2] += 0.5  # mesma trajetória, postura deslocada
  cfg_a = {**cfg, "concat_postura": False, "concat_periodicidade": False}
  m_a = rede(cfg_a)
  m_a.eval()
  assert torch.allclose(m_a(Xt), m_a(desloc.reshape(3, 30, 68)), atol=1e-5), "P1a não deveria ver a postura"
  assert base.shape == (3, 4)
  Xa = aumentar(Xt.clone(), torch.Generator().manual_seed(0), {"aum_rot_graus": 30, "aum_escala_xy": 0.2, "aum_ruido": 0.02})
  assert Xa.shape == Xt.shape and torch.isfinite(Xa).all()


# ---------- predições com janela congelada ----------

def prever_val(args):
  exp_dir, cfg, k, vids, ablacao, opc = args
  q.aplicar_opcoes(*opc)
  try:
    import torch
    torch.set_num_threads(1)
    return k, q.inferir_lstm(q.carregar_modelo(exp_dir, cfg, k), {}, vids, cfg, ablacao), None
  except Exception:
    return k, None, traceback.format_exc()


def vd_congelado(exp_dir, cfg, parte):
  """Predições com janelas congeladas, em validação (chave partição|vídeo) ou teste (chave vídeo)."""
  val = q.validacao()
  vids = {k: (val[k] if parte == "validacao" else [v for v in r.videos() if r.PARTICOES[v] == k]) for k in r.particoes()}
  with mp.get_context("spawn").Pool(5) as pool:
    res = pool.map(prever_val, [(str(exp_dir), cfg, k, vids[k], "congelada", q.opcoes()) for k in r.particoes()])
  erros = [e for _, _, e in res if e]
  if erros:
    raise RuntimeError("\n\n".join(erros))
  vd = {}
  for k, x, _ in res:
    for vid, y in q.vd_de_resultados({k: x}, cfg).items():
      vd[f"{k}|{vid}" if parte == "validacao" else vid] = y
  return vd


def f1_anotados(vd):
  import evaluate_temporal as et
  rq.agrupar_por_particao(vd)
  res = et.frame_level_with_folds(vd, sorted(vd), True)
  rep = res["pooled_56_videos"]["report"]
  return {"agregado": res["pooled_56_videos"]["macro_f1_sem_background"], "media": res["macro_f1_sem_background_media"],
          "desvio": res["macro_f1_sem_background_desvio"], "por_classe": {c: rep[c]["f1-score"] for c in CLASSES3}}


def dependencia(vd_real, vd_cong, comum):
  a, b = f1_anotados(rq.restringir(vd_real, comum)), f1_anotados(rq.restringir(vd_cong, comum))
  return {"real": a, "congelada": b, "dependencia": a["agregado"] - b["agregado"], "dependencia_media": a["media"] - b["media"],
          "dependencia_por_classe": {c: a["por_classe"][c] - b["por_classe"][c] for c in CLASSES3}}


def tcc_congelado():
  return rq.tcc_vd("congelada")


# ---------- passo 0 ----------

def passo0(ref_teste):
  p = pasta("r4_passo0") / "diagnostico.json"
  if p.exists():
    return r.ler_json(p)
  tab = {x["video"]: x for x in csv.DictReader(open(EXP1 / "qualidade_dataset/tabela_videos.csv", encoding="utf-8"))}
  f1 = lambda v: float(tab[v]["f1"]) if tab[v]["f1"] not in ("", "None") else None
  ruins = sorted(v for v in tab if float(tab[v]["q_pct"]) > 90 and f1(v) is not None and f1(v) < 0.1 and v in r.videos())
  melhores = []
  for c in r.CLASSES[:3]:
    vs = sorted((v for v in tab if r.classe_do_video(v) == c and f1(v) is not None and v in r.videos()), key=f1, reverse=True)
    melhores += vs[:3]
  comum = rq.conjunto_comum(*ref_teste.values())
  import evaluate_temporal as et
  out = {"criterio": "Q > 90% e F1 < 0,1 (tabela_videos.csv); 3 de maior F1 por classe", "videos": {}}
  for v in ruins + melhores:
    fr = {"dentro": [], "fora": []}
    for vd in ref_teste.values():
      x = rq.restringir({v: vd[v]}, comum).get(v) if v in vd else None
      if not x:
        continue
      pred = et.predicted_frames(x["windows"])
      for parte, cond in (("dentro", lambda g: g != "Background"), ("fora", lambda g: g == "Background")):
        ps = [pv for qd, pv in pred.items() if cond(x["gt_frames"][qd])]
        fr[parte].append({c: (sum(1 for pv in ps if pv == c) / len(ps)) if ps else None for c in r.CLASSES})
    media = {parte: {c: (float(np.mean([s[c] for s in xs if s[c] is not None])) if any(s[c] is not None for s in xs) else None) for c in r.CLASSES}
             for parte, xs in fr.items() if xs}
    out["videos"][v] = {"grupo": "Q>90%, F1<0,1" if v in ruins else "maior F1 da classe", "classe": r.classe_do_video(v), "f1": f1(v),
                        "q_pct": float(tab[v]["q_pct"]), "fracoes": media}
  r.gravar_json(p, out)
  f3 = q.f3
  linhas = ["# Passo 0: o que o `r2_C4_aumento` prevê (teste, quadros válidos por Q, média das 3 sementes)", "",
            "| vídeo | grupo | F1 | anotados: AF / HB / SP / fundo | fora dos episódios: AF / HB / SP / fundo |", "|---|---|---|---|---|"]
  for v, x in out["videos"].items():
    cel = lambda parte: " / ".join(f3(x["fracoes"][parte][c]) if parte in x["fracoes"] and x["fracoes"][parte][c] is not None else "--" for c in r.CLASSES)
    linhas.append(f"| {v} | {x['grupo']} | {f3(x['f1'])} | {cel('dentro')} | {cel('fora')} |")
  (p.parent / "diagnostico.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")
  return out


# ---------- resumo ----------

def escrever_resumo(hist, dep_ref, n_val, final_txt=None):
  f3 = q.f3
  cel = lambda m, x: f"{f3(m['agregado_validacao'][x])} / {f3(m['media_entre_particoes'][x])} ± {f3(m['desvio_entre_particoes'][x])}"
  linhas = ["# Resumo da rodada 4", "",
            f"Protocolo: `protocolo_rodada4.md`. Validação nos {n_val} quadros válidos por Q em que todos os modelos têm predição "
            "(agregado / média ± desvio entre partições). Dependência do movimento: macro F1 nos anotados com janelas reais menos com janelas congeladas.", "",
            "| etapa | mudança | F1 evento | acurácia bal. | macro F1 quadros | dependência (AF / HB / SP) | situação | tempo |", "|---|---|---|---|---|---|---|---|",
            f"| `{REF}` | referência | " + " | ".join(cel(dep_ref["met"], x) for x in METRICAS)
            + f" | {f3(dep_ref['dep']['dependencia'])} ({' / '.join(f3(dep_ref['dep']['dependencia_por_classe'][c]) for c in CLASSES3)}) | referência | -- |"]
  for h in hist:
    if "met" not in h:
      linhas.append(f"| {h['id']} | {h['descricao']} | -- | -- | -- | -- | {h['situacao']} | -- |")
      continue
    d = h["dep"]
    linhas.append(f"| {h['id']} | {h['descricao']} | " + " | ".join(cel(h["met"], x) for x in METRICAS)
                  + f" | {f3(d['dependencia'])} ({' / '.join(f3(d['dependencia_por_classe'][c]) for c in CLASSES3)}) | {h['situacao']} | {h['tempo']} |")
  linhas += ["", "Passo 0 (diagnóstico do `r2_C4_aumento`): `r4_passo0/diagnostico.md`."]
  if final_txt:
    linhas += ["", *final_txt]
  (Path(t3.O3["saida"]) / "resumo_rodada4.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")


def texto_final(fin):
  f3 = q.f3
  M4 = METRICAS + ["macro_f1_quadro_anotados"]
  cel_t = lambda t, x: f"{f3(t['agregado_56_videos'][x])} / {f3(t['media_entre_particoes'][x])} ± {f3(t['desvio_entre_particoes'][x])}"
  med = lambda ts, x, conv: f"{f3(np.mean([t[conv][x] for t in ts]))} ± {f3(np.std([t[conv][x] for t in ts]))}"
  nome = fin["modelo"]
  linhas = [f"## Avaliação final: {nome} ({fin['papel']})", "",
            f"Teste nos {fin['n_quadros_teste']} quadros válidos por Q comuns a TCC, `{REF}` e `{nome}`. Pós-processamento escolhido na validação: "
            f"{fin['pos']}. As métricas por quadro não dependem dele.", "",
            "| | F1 evento | acurácia bal. | macro F1 quadros | macro F1 anotados | F1 evento com pós |", "|---|---|---|---|---|---|",
            f"| TCC | {' | '.join(cel_t(fin['TCC'], x) for x in M4)} | {f3(fin['TCC']['f1_evento_com_pos']['agregado'])} / {f3(fin['TCC']['f1_evento_com_pos']['media'])} |"]
  for rot, chave in ((f"`{REF}`", "ref"), (f"`{nome}`", "final")):
    ts = fin[chave]
    for conv, nome_conv, pc in (("agregado_56_videos", "agregado", "agregado"), ("media_entre_particoes", "média entre partições", "media")):
      linhas.append(f"| {rot}, 3 sementes, {nome_conv} | {' | '.join(med(ts, x, conv) for x in M4)} | "
                    f"{f3(np.mean([t['f1_evento_com_pos'][pc] for t in ts]))} ± {f3(np.std([t['f1_evento_com_pos'][pc] for t in ts]))} |")
  linhas += ["", "F1 por classe (semente 42, agregado; todos os quadros | anotados) e mudança em relação ao `r2_C4_aumento`:"]
  for c in r.CLASSES:
    a = fin["detalhes"]["final"]["por_classe"]
    b = fin["detalhes"]["ref"]["por_classe"]
    t_, n_ = a["todos_os_quadros"]["agregado"][c]["f1-score"], a["quadros_anotados"]["agregado"][c]["f1-score"]
    tr, nr = b["todos_os_quadros"]["agregado"][c]["f1-score"], b["quadros_anotados"]["agregado"][c]["f1-score"]
    linhas.append(f"- *{c}*: {f3(t_)} ({'+' if t_ >= tr else ''}{f3(t_ - tr)}) | " + (f"{f3(n_)} ({'+' if n_ >= nr else ''}{f3(n_ - nr)})" if c != "Background" else "--"))
  dep = fin["dependencia_teste"]
  linhas += ["", "Dependência do movimento no teste (agregado; por classe AF / HB / SP): " + "; ".join(
    f"{k} {f3(np.mean([x['dependencia'] for x in v]))} ({' / '.join(f3(np.mean([x['dependencia_por_classe'][c] for x in v])) for c in CLASSES3)})"
    for k, v in dep.items()) + " (média das sementes quando há 3).",
    "Ablações, macro F1 anotados com 3 classes (agregado; sem / ordem / dxdy): " + "; ".join(
      f"{k}: " + ", ".join(" / ".join(f3(s[a]["macro_f1_anotados_3_classes"]["agregado"]) for a in ("sem_ablacao", "ordem", "dxdy")) for s in v)
      for k, v in fin["ablacoes"].items()) + ".",
    "Fração da classe do vídeo, dentro / fora (semente 42 e TCC): " + "; ".join(
      f"{k}: " + ", ".join(f"{c} {f3(x['dentro'])}/{f3(x['fora'])}" for c, x in v[0]["sem_ablacao"]["fracao_classe_do_video"].items())
      for k, v in fin["ablacoes"].items()) + ".",
    "*Head banging* com F1 zero: " + "; ".join(f"{k}: " + ", ".join(f"{h['n_zero']} de {h['n_videos_com_quadros_validos']}" for h in v)
                                              for k, v in fin["hb"].items()) + ".",
    f"Matriz de confusão e por classe completos: `r4_avaliacao_final/avaliacao_final.json`; curva de erro: `{nome}/curva_erro.pdf`."]
  if fin.get("nao_rodou"):
    linhas.append("Não rodou: " + "; ".join(fin["nao_rodou"]) + ".")
  return linhas


# ---------- sequência ----------

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--teste", action="store_true", help="rodada rápida: 1 partição, 2 épocas, 9 vídeos, sem commit; força a avaliação final")
  ap.add_argument("--saida", default=str(EXP1))
  args = ap.parse_args()
  t3.O3.update({"teste": args.teste, "saida": str(Path(args.saida).resolve())})
  saida = Path(t3.O3["saida"])
  saida.mkdir(parents=True, exist_ok=True)
  if args.teste:
    div1 = r.divisao_validacao()[1]
    esc = []
    for c in r.CLASSES[:3]:
      esc += [next(v for v in div1 if r.classe_do_video(v) == c),
              next(v for v in sorted(r.PARTICOES) if r.classe_do_video(v) == c and r.PARTICOES[v] != 1 and v not in div1),
              next(v for v in sorted(r.PARTICOES) if r.classe_do_video(v) == c and r.PARTICOES[v] == 1)]
    r.OPC.update({"videos": esc, "particoes": [1]})
  log(f"início (pid {os.getpid()}){' -- TESTE' if args.teste else ''}")
  try:
    autoverificacao()
    cfg_ref = {**q.NOVOS, **rq.cfg_de(EXP1 / REF)}
    # só os vídeos em uso (no teste rápido, 9); na rodada completa não muda nada
    usa = lambda vd: {k: v for k, v in vd.items() if k.split("|")[-1] in r.videos()}
    ref_teste = {s: usa(rq.dados(EXP1 / d, "teste", cfg_ref if s == 42 else {**cfg_ref, "semente": s})) for s, d in REF_SEMENTES.items()}
    passo0(ref_teste)
    commit("experimentos: r4_passo0", [pasta("r4_passo0") / "diagnostico.json", pasta("r4_passo0") / "diagnostico.md", Path(__file__)])

    ref_val = usa(rq.dados(EXP1 / REF, "validacao", cfg_ref))
    ref_cong_val = vd_congelado(EXP1 / REF, cfg_ref, "validacao")
    melhor_id, melhor_cfg, melhor_val = REF, cfg_ref, ref_val
    hist, alternativas = [], []
    comum_ref = rq.conjunto_comum(ref_val)
    dep_ref = {"met": q.met_val(rq.restringir(ref_val, comum_ref)), "dep": dependencia(ref_val, ref_cong_val, comum_ref)}
    n_val = rq.n_quadros(rq.restringir(ref_val, comum_ref))
    log(f"referência: {n_val} quadros de validação válidos por Q; dependência do movimento {dep_ref['dep']['dependencia']:.3f}")
    escrever_resumo(hist, dep_ref, n_val)

    etapas = [
      [("r4_P1a_trajetoria", {"arquitetura": "p4", "trajetoria": True, "concat_postura": False}, "P1a: trajetória centralizada, sem postura"),
       ("r4_P1b_trajetoria_postura", {"arquitetura": "p4", "trajetoria": True, "concat_postura": True}, "P1b: trajetória centralizada + postura média após o LSTM")],
      [("r4_P2_periodicidade", {"arquitetura": "p4", "concat_periodicidade": True}, "P2: + periodicidade após o LSTM")],
      [("r4_P3_aumento_postura", {"aum_rot_graus": 30, "aum_escala_xy": 0.2, "aum_ruido": 0.02}, "P3: + rotação ±30°, escala x/y ±20%, ruído")],
    ]
    for etapa in etapas:
      base_met = q.met_val(rq.restringir(melhor_val, rq.conjunto_comum(melhor_val, ref_val)))
      passou = []
      for exp_id, mud, desc in etapa:
        cfg = {**melhor_cfg, **mud}
        if cfg.get("arquitetura") == "p4":
          cfg = {"trajetoria": False, "concat_postura": False, "concat_periodicidade": False, **cfg}
        if args.teste:
          cfg["max_epocas"] = 2
        t0 = time.time()
        met = t3.rodar3(exp_id, cfg, desc)
        if met is None:
          hist.append({"id": exp_id, "descricao": desc, "situacao": "erro"})
        else:
          vd = rq.dados(pasta(exp_id), "validacao", cfg)
          comum = rq.conjunto_comum(ref_val, melhor_val, vd)
          n = rq.n_quadros(rq.restringir(vd, comum))
          if n != n_val:
            log(f"AVISO: {exp_id} avaliada em {n} quadros, a referência em {n_val}")
          m = q.met_val(rq.restringir(vd, comum))
          dep = dependencia(vd, vd_congelado(pasta(exp_id), cfg, "validacao"), comum)
          ok, soma = r.supera(m["agregado_validacao"], base_met["agregado_validacao"])
          nao_piora = all(m["agregado_validacao"][x] >= base_met["agregado_validacao"][x] - 0.01 for x in METRICAS)
          alt = (not ok) and nao_piora and dep["dependencia"] >= 2 * dep_ref["dep"]["dependencia"]
          situacao = "candidata" if ok else ("alternativa" if alt else "não")
          if ok:
            passou.append((soma, exp_id, cfg, vd))
          if alt:
            alternativas.append((dep["dependencia"], exp_id, cfg, vd))
          met_tempo = r.ler_json(pasta(exp_id) / "metricas_validacao.json").get("tempo_s") or (time.time() - t0)
          hist.append({"id": exp_id, "descricao": desc, "met": m, "dep": dep, "situacao": situacao, "tempo": f"{met_tempo / 60:.0f} min", "n": n})
          r.gravar_json(pasta(exp_id) / "rodada4.json", {"metricas_validacao_q": m, "dependencia": dep, "passa": ok, "alternativa": alt, "n_quadros": n})
          log(f"{exp_id}: passa={ok} alternativa={alt} dependência {dep['dependencia']:.3f} | "
              + " ".join(f"{x}={m['agregado_validacao'][x]:.3f}" for x in METRICAS))
      escolhido = max(passou, key=lambda x: x[0]) if passou else None
      for h in hist:
        if h["situacao"] == "candidata":
          h["situacao"] = "mantida" if escolhido and h["id"] == escolhido[1] else "não"
      if escolhido:
        _, melhor_id, melhor_cfg, melhor_val = escolhido
      escrever_resumo(hist, dep_ref, n_val)
      for exp_id, _, _ in etapa:
        commit(f"experimentos: {exp_id}", q.arquivos(pasta(exp_id)) + [pasta(exp_id) / "rodada4.json", saida / "resumo_rodada4.md", Path(__file__)])

    # avaliação final
    alvo, papel = (melhor_id, "modelo mantido") if melhor_id != REF else (None, None)
    if alvo is None and alternativas:
      _, alvo, cfg_alt, _ = max(alternativas, key=lambda x: x[0])
      melhor_cfg, papel = cfg_alt, "alternativa"
    if alvo is None and args.teste:
      alvo, melhor_cfg, papel = hist[-1]["id"], r.ler_json(pasta(hist[-1]["id"]) / "config.json"), "forçado no teste"
      melhor_cfg = t3.cfg_limpo(melhor_cfg)
    fin = avaliacao_final(alvo, melhor_cfg, papel, cfg_ref, ref_teste, args)
    escrever_resumo(hist, dep_ref, n_val, texto_final(fin) if alvo else fin["texto"])
    commit("experimentos: r4_avaliacao_final e resumo", [pasta("r4_avaliacao_final") / "avaliacao_final.json", saida / "resumo_rodada4.md", Path(__file__)])
    log("fim")
  except Exception:
    log("ERRO GERAL:\n" + traceback.format_exc())
    raise


def avaliacao_final(alvo, cfg, papel, cfg_ref, ref_teste, args):
  p = pasta("r4_avaliacao_final") / "avaliacao_final.json"
  if p.exists() and r.ler_json(p).get("modelo") == alvo:
    return r.ler_json(p)
  tcc = rq.tcc_vd()
  tcc_cong = tcc_congelado()
  if alvo is None:
    comum = rq.conjunto_comum(tcc, ref_teste[42])
    ref_cong = vd_congelado(EXP1 / REF, cfg_ref, "teste")
    fin = {"modelo": None, "dependencia_teste": {"TCC": [dependencia(tcc, tcc_cong, comum)], REF: [dependencia(ref_teste[42], ref_cong, comum)]}}
    fin["texto"] = [f"Nenhuma variante foi mantida e nenhuma alternativa apareceu: o modelo final continua `{REF}`. Dependência do movimento no teste "
                    f"(agregado): TCC {q.f3(fin['dependencia_teste']['TCC'][0]['dependencia'])}, `{REF}` {q.f3(fin['dependencia_teste'][REF][0]['dependencia'])}."]
    r.gravar_json(p, fin)
    return fin
  dirs = {42: pasta(alvo)}
  nao_rodou = []
  for s in (1000, 2000):
    e = f"{alvo}_semente_{s}"
    if t3.rodar3(e, {**cfg, "semente": s}, f"{alvo}, semente {s}", com_teste=True) is not None:
      dirs[s] = pasta(e)
    else:
      nao_rodou.append(f"{e}: erro")
    commit(f"experimentos: {e}", q.arquivos(pasta(e)) + [Path(__file__)])
  q.testar(alvo, cfg)
  fin_teste = {s: rq.dados(d, "teste", cfg if s == 42 else {**cfg, "semente": s}) for s, d in dirs.items()}
  comum = rq.conjunto_comum(tcc, ref_teste[42], fin_teste[42])
  ref_val = rq.dados(EXP1 / REF, "validacao", cfg_ref)
  alvo_val = rq.dados(pasta(alvo), "validacao", cfg)
  esc, grade = rq.escolher_pos(rq.restringir(alvo_val, rq.conjunto_comum(ref_val, alvo_val)))
  g = grade[esc]
  fin = {"modelo": alvo, "papel": papel, "pos": esc, "grade_pos": grade, "n_quadros_teste": rq.n_quadros(rq.restringir(tcc, comum)),
         "TCC": rq.metricas_teste_restritas(rq.restringir(tcc, comum), g),
         "ref": [rq.metricas_teste_restritas(rq.restringir(v, comum), g) for v in ref_teste.values()],
         "final": [rq.metricas_teste_restritas(rq.restringir(v, comum), g) for v in fin_teste.values()],
         "detalhes": {"TCC": rq.detalhes(rq.restringir(tcc, comum)), "ref": rq.detalhes(rq.restringir(ref_teste[42], comum)),
                      "final": rq.detalhes(rq.restringir(fin_teste[42], comum))}, "nao_rodou": nao_rodou}
  fin["dependencia_teste"] = {
    "TCC": [dependencia(tcc, tcc_cong, comum)],
    REF: [dependencia(ref_teste[s], vd_congelado(EXP1 / d, cfg_ref if s == 42 else {**cfg_ref, "semente": s}, "teste"), comum) for s, d in REF_SEMENTES.items()],
    alvo: [dependencia(fin_teste[s], vd_congelado(dirs[s], cfg if s == 42 else {**cfg, "semente": s}, "teste"), comum) for s in dirs]}
  tarefas = [(s, a, k) for s in dirs for a in ("ordem", "dxdy") for k in r.particoes()]
  with mp.get_context("spawn").Pool(5) as pool:
    saidas = pool.map(q.prever, [(str(dirs[s]), cfg if s == 42 else {**cfg, "semente": s}, k, a, q.opcoes()) for s, a, k in tarefas])
  erros = [e for _, _, e in saidas if e]
  if erros:
    raise RuntimeError("\n\n".join(erros))
  por = {}
  for (s, a, _), (k, x, _) in zip(tarefas, saidas):
    por.setdefault(s, {}).setdefault(a, {})[k] = x
  fin["ablacoes"] = {alvo: [rq.ablacoes_de(rq.restringir(fin_teste[s], comum), {a: q.vd_de_resultados(x, cfg) for a, x in por[s].items()}, comum)
                            for s in dirs],
                     "TCC": [rq.ablacoes_de(rq.restringir(tcc, comum), {a: rq.tcc_vd(a) for a in ("ordem", "dxdy")}, comum)]}
  fin["hb"] = {alvo: [rq.detalhes(rq.restringir(fin_teste[s], comum))["head_banging_f1_zero"] for s in dirs],
               REF: [rq.detalhes(rq.restringir(v, comum))["head_banging_f1_zero"] for v in ref_teste.values()],
               "TCC": [fin["detalhes"]["TCC"]["head_banging_f1_zero"]]}
  r.gravar_json(p, fin)
  return fin


if __name__ == "__main__":
  main()
