# ============================================================
# ARQUIVO: experimentos/rodar_q.py
#
# O QUE FAZ: critério Q de qualidade da pose (adendo de 2026-10-21 em
# protocolo-rodada3.md). Um quadro com pessoa é válido se os dois ombros
# e pelo menos um ponto da cabeça têm confiança >= 0,5; os inválidos saem
# de todo o processo. Calcula a cobertura de Q, compara sem treino novo o
# modelo do TCC e o r2-C4-aumento restritos a Q, treina o QF (configuração
# do r2-C4-aumento, só com quadros válidos, normalização pela parte de cima
# do corpo) e decide pelo critério na validação. Se o QF passar, treina as
# sementes 1000 e 2000 e faz a avaliação de teste completa; se não, reporta
# o r2-C4-aumento nos quadros válidos. Retomável, um commit por etapa.
#
# Uso (da raiz do repositório):
#   .venv/Scripts/python.exe experimentos/rodar_q.py
#   .venv/Scripts/python.exe experimentos/rodar_q.py --teste --saida <pasta>
# ============================================================

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(RAIZ / "experimentos"), str(RAIZ / "src")]
import rodar_rodada3 as t3  # noqa: E402  (aplica as substituições das rodadas 2 e 3)

r, q = t3.r, t3.q
EXP1 = t3.EXP1
REF, REF_SEMENTES = t3.REF, t3.REF_SEMENTES
CABECA = [0, 1, 2, 3, 4]
LIMITE_QF = 10.0
VARIANTES = ["r3-I1-interpolacao-limitada", "r3-I2-limitada-confianca", "r3-I3-treino-a", "r3-I3-treino-b"]
QF = "r3-q-QF"
_aumentar_anterior = q.aumentar


def aumentar(X, g, cfg):
  """O limite de ±10 só vale no aumento de dados do QF; os outros modelos continuam com ±4."""
  if cfg.get("normalizacao") != "q_superior":
    return _aumentar_anterior(X, g, cfg)
  antes, r.LIMITE_NORM = r.LIMITE_NORM, LIMITE_QF
  try:
    return _aumentar_anterior(X, g, cfg)
  finally:
    r.LIMITE_NORM = antes


q.aumentar = aumentar


def valido_q(conf):
  return (conf[:, 5] >= r.LIMIAR_CONF) & (conf[:, 6] >= r.LIMIAR_CONF) & (conf[:, CABECA] >= r.LIMIAR_CONF).any(1)


_preparar_anterior = r.preparar_video


def preparar_video(vid, cfg):
  """QF: só quadros válidos por Q; centro nos ombros, escala até a média da cabeça confiável."""
  if cfg.get("normalizacao") != "q_superior":
    return _preparar_anterior(vid, cfg)
  import evaluate_temporal as et
  d = np.load(r.caminho("poses", cfg["pose"], f"{vid}.npz"))
  quadros, xy, conf = d["frames"].astype(np.int64), d["xy"].astype(np.float32), d["conf"].astype(np.float32)
  if len(quadros) == 0:
    return None
  n_quadros = int(quadros[-1]) + 1
  ok = valido_q(conf)
  quadros, xy, conf = quadros[ok], xy[ok], conf[ok]
  if len(quadros) == 0:
    return None
  fps = et.load_video_fps(vid)
  xy_l, m, _ = t3.interpolacao_limitada(quadros, xy, conf, fps)
  centro = (xy[:, 5].astype(np.float64) + xy[:, 6]) / 2
  cab_ok = conf[:, CABECA] >= r.LIMIAR_CONF
  cabeca = (xy[:, CABECA] * cab_ok[..., None]).sum(1) / cab_ok.sum(1, keepdims=True)
  escala = np.linalg.norm(cabeca - centro, axis=1)
  ruim = escala < 1
  if ruim.all():
    escala[:] = 1
  elif ruim.any():  # escala degenerada: repete a última válida (ou a primeira, no começo)
    idx = np.maximum.accumulate(np.where(ruim, -1, np.arange(len(escala))))
    escala = escala[np.where(idx < 0, np.argmax(~ruim), idx)]
  P = np.clip((xy_l - centro[:, None]) / escala[:, None, None], -LIMITE_QF, LIMITE_QF).astype(np.float32)
  P[~m] = 0
  gt, _ = et.ground_truth_frames(vid, n_quadros, fps)
  return {"quadros": quadros, "P": P, "M": m, "C": conf, "fps": fps,
          "rotulo": np.array([r.CLASSES.index(g) for g in gt], np.int64), "com_conf": False}


r.preparar_video = preparar_video


# ---------- quadros válidos e restrição ----------

_VALIDOS = {}


def validos(vid):
  if vid not in _VALIDOS:
    d = np.load(r.caminho("poses", "yolo11n", f"{vid}.npz"))
    _VALIDOS[vid] = set(d["frames"][valido_q(d["conf"])].astype(int).tolist())
  return _VALIDOS[vid]


def preditos(vd):
  return {c: {w["quadro_fim"] - 1 for w in x["windows"]} for c, x in vd.items()}


def conjunto_comum(*vds):
  """Por vídeo (ou partição|vídeo): quadros válidos por Q em que todos os modelos têm predição."""
  chaves = set.intersection(*[set(v) for v in vds])
  ps = [preditos(v) for v in vds]
  return {c: set.intersection(validos(c.split("|")[-1]), *[p[c] for p in ps]) for c in chaves}


def restringir(vd, comum):
  """Mantém só as janelas que terminam em quadros do conjunto comum e tira do gabarito de eventos
  os episódios sem nenhum quadro válido por Q."""
  out = {}
  for c, x in vd.items():
    if c not in comum or not comum[c]:
      continue
    vid = c.split("|")[-1]
    fps = x["fps"]
    jan = [w for w in x["windows"] if (w["quadro_fim"] - 1) in comum[c]]
    if not jan:
      continue
    ok = validos(vid)
    iv = [(lab, a, b) for lab, a, b in x["gt_intervals"]
          if any(f in ok for f in range(max(0, int(round(a * fps))), int(round(b * fps)) + 1))]
    out[c] = {**x, "windows": jan, "gt_intervals": iv}
  return out


def n_quadros(vd):
  return sum(len(x["windows"]) for x in vd.values())


def agrupar_por_particao(vd):
  import evaluate_temporal as et
  et.fold_assignments = {**r.PARTICOES, **{c: x["fold"] for c, x in vd.items()}}


def dados(exp_dir, parte, cfg):
  vd = r.dados_avaliacao(Path(exp_dir), parte, cfg)
  return vd


def cfg_de(exp_dir):
  return t3.cfg_limpo(r.ler_json(Path(exp_dir) / "config.json"))


# ---------- cobertura de Q ----------

def cobertura():
  pdir = t3.pasta("r3-q-cobertura")
  if (pdir / "cobertura.json").exists():
    return r.ler_json(pdir / "cobertura.json")
  import evaluate_temporal as et
  soma, sem_validos, episodios_sem_deteccao = {}, [], {}
  for vid in r.videos():
    d = np.load(r.caminho("poses", "yolo11n", f"{vid}.npz"))
    quadros = d["frames"].astype(int)
    ok = valido_q(d["conf"])
    fps = et.load_video_fps(vid)
    gt, iv = et.ground_truth_frames(vid, int(quadros[-1]) + 1, fps)
    anot = np.array([gt[f] != "Background" for f in quadros])
    det, val = set(quadros.tolist()), set(quadros[ok].tolist())
    s = soma.setdefault(r.classe_do_video(vid), np.zeros(6, int))
    ep_fora = 0
    for lab, a, b in iv:
      faixa = range(max(0, int(round(a * fps))), int(round(b * fps)) + 1)
      if not any(f in val for f in faixa):
        ep_fora += 1
        if not any(f in det for f in faixa):
          episodios_sem_deteccao[r.classe_do_video(vid)] = episodios_sem_deteccao.get(r.classe_do_video(vid), 0) + 1
    s += [len(quadros), (~ok).sum(), anot.sum(), (~ok & anot).sum(), len(iv), ep_fora]
    if not ok.any():
      sem_validos.append(vid)
  por_classe = {c: {"quadros": int(x[0]), "excluidos": int(x[1]), "fracao_quadros": float(x[1] / x[0]),
                    "anotados": int(x[2]), "anotados_excluidos": int(x[3]), "fracao_anotados": float(x[3] / max(1, x[2])),
                    "episodios": int(x[4]), "episodios_excluidos": int(x[5]), "fracao_episodios": float(x[5] / max(1, x[4])),
                    "episodios_ja_sem_deteccao": int(episodios_sem_deteccao.get(c, 0))} for c, x in sorted(soma.items())}
  t = np.sum(list(soma.values()), axis=0)
  out = {"criterio": "ombros 5 e 6 e pelo menos um de 0-4 com confiança >= 0,5", "por_classe": por_classe,
         "total": {"fracao_quadros": float(t[1] / t[0]), "fracao_anotados": float(t[3] / t[2]), "episodios_excluidos": int(t[5]), "episodios": int(t[4])},
         "videos_sem_quadro_valido": sem_validos}
  r.gravar_json(pdir / "cobertura.json", out)
  f3 = q.f3
  linhas = ["# Cobertura do critério Q", "", "Quadro válido: os dois ombros e pelo menos um ponto da cabeça com confiança ≥ 0,5.", "",
            "| classe | quadros excluídos | quadros anotados excluídos | episódios excluídos do gabarito |", "|---|---|---|---|"]
  for c, x in por_classe.items():
    linhas.append(f"| *{c}* | {x['excluidos']} de {x['quadros']} ({f3(x['fracao_quadros'])}) | {x['anotados_excluidos']} de {x['anotados']} "
                  f"({f3(x['fracao_anotados'])}) | {x['episodios_excluidos']} de {x['episodios']} ({f3(x['fracao_episodios'])}; "
                  f"{x['episodios_ja_sem_deteccao']} já sem pessoa detectada) |")
  linhas += ["", f"Total: {f3(out['total']['fracao_quadros'])} dos quadros e {f3(out['total']['fracao_anotados'])} dos anotados; "
             f"{out['total']['episodios_excluidos']} de {out['total']['episodios']} episódios saem do gabarito. "
             f"Vídeos sem nenhum quadro válido: {len(sem_validos)}" + (f" ({', '.join(sem_validos)})." if sem_validos else ".")]
  (pdir / "cobertura.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")
  return out


# ---------- avaliações restritas ----------

def metricas_teste_restritas(vd, g=None):
  import evaluate_temporal as et
  et.fold_assignments = r.PARTICOES
  t = r.metricas_teste(vd)
  t["n_quadros"] = n_quadros(vd)
  t["n_episodios_gabarito"] = sum(len(x["gt_intervals"]) for x in vd.values())
  if g:
    t["f1_evento_com_pos"] = q.f1_evento_pos(vd, g["mediana_s"], g["juncao_s"])
  return t


def escolher_pos(vd_val):
  agrupar_por_particao(vd_val)
  grade = {}
  for med in (0, 0.5, 1.0, 2.0):
    for jun in (0, 0.5, 1.0):
      grade[f"mediana_{med}_juncao_{jun}"] = {"mediana_s": med, "juncao_s": jun, "validacao": q.f1_evento_pos(vd_val, med, jun)}
  esc = max(grade, key=lambda n: (grade[n]["validacao"]["agregado"], -grade[n]["mediana_s"] - grade[n]["juncao_s"]))
  return esc, grade


def tcc_vd(ablacao=None):
  """Predições do modelo do TCC (evaluate_temporal.py), opcionalmente com a ablação aplicada às janelas."""
  import evaluate_temporal as et
  import pandas as pd
  import torch
  et.load_config("data/datasets_annotated", "models/oof-seeded-annotated")
  if ablacao is None:
    return et.load_all_videos()
  original = et.compute_windows

  def com_ablacao(video_id):  # mesmo corpo de evaluate_temporal.compute_windows, com a ablação nas janelas
    df = pd.read_csv(et.POSES_DIR / f"{video_id}.csv")
    quadro = df["frame"].values
    feat = df.drop(columns=[c for c in ["frame", "person"] if c in df.columns]).values.astype(np.float32)
    if len(feat) < et.SEQUENCE_LENGTH:
      return []
    ini = np.arange(len(feat) - et.SEQUENCE_LENGTH + 1)
    seq = q.ablar(np.stack([feat[s:s + et.SEQUENCE_LENGTH] for s in ini]), ablacao).astype(np.float32)
    with torch.no_grad():
      p = torch.softmax(et.get_fold_model(et.fold_assignments[video_id])(torch.from_numpy(seq).to(et.device)), 1).cpu().numpy()
    return [{"quadro_inicio": int(quadro[s]), "quadro_fim": int(quadro[s + et.SEQUENCE_LENGTH - 1]) + 1, "pontuacoes": p[i].tolist()}
            for i, s in enumerate(ini)]
  et.compute_windows = com_ablacao
  try:
    return et.load_all_videos()
  finally:
    et.compute_windows = original


def detalhes(vd):
  """Por classe, matriz de confusão e head banging com F1 zero, num conjunto já restrito."""
  import evaluate_temporal as et
  from selecionar_videos_mediana import f1_do_video
  et.fold_assignments = r.PARTICOES
  todos, anot = et.frame_level_with_folds(vd, sorted(vd), False), et.frame_level_with_folds(vd, sorted(vd), True)
  tira = lambda rep: {c: {k: rep[c][k] for k in ("precision", "recall", "f1-score", "support")} for c in r.CLASSES}
  hb = [f1_do_video(x)[0] for c, x in vd.items() if r.classe_do_video(c) == "HeadBanging"]
  hb = [f for f in hb if f is not None]
  return {"por_classe": {"todos_os_quadros": {"agregado": tira(todos["pooled_56_videos"]["report"]), "media_desvio": todos["por_classe_media_desvio_entre_folds"]},
                         "quadros_anotados": {"agregado": tira(anot["pooled_56_videos"]["report"]), "media_desvio": anot["por_classe_media_desvio_entre_folds"]}},
          "confusao_quadros_anotados": {"labels": anot["pooled_56_videos"]["labels"], "matriz": anot["pooled_56_videos"]["confusion_matrix"]},
          "head_banging_f1_zero": {"n_zero": sum(1 for f in hb if f == 0), "n_videos_com_quadros_validos": len(hb),
                                   "n_videos_head_banging": sum(1 for v in r.videos() if r.classe_do_video(v) == "HeadBanging")}}


def ablacoes_de(vd_sem, por_ablacao, comum):
  out = {"sem_ablacao": {"macro_f1_anotados_3_classes": q.f1_anotados_3_classes(vd_sem), "fracao_classe_do_video": q.fracao_classe_do_video(vd_sem)}}
  for a, vd in por_ablacao.items():
    vd = restringir(vd, comum)
    out[a] = {"metricas": metricas_teste_restritas(vd), "macro_f1_anotados_3_classes": q.f1_anotados_3_classes(vd),
              "fracao_classe_do_video": q.fracao_classe_do_video(vd)}
  return out


def avaliacao_completa(nome, dirs, cfg, val_final, comum_val, tcc, comum, teste_por_modelo):
  """Avaliação de teste completa do modelo final (3 sementes), ao lado do TCC e do r2-C4-aumento, no conjunto comum."""
  esc, grade = escolher_pos(restringir(val_final, comum_val))
  g = grade[esc]
  out = {"modelo_final": nome, "pos_processamento": {"escolhido": esc, "grade": grade},
         "n_quadros_teste": n_quadros(restringir(tcc, comum)), "TCC": metricas_teste_restritas(restringir(tcc, comum), g)}
  for rotulo, vds in teste_por_modelo.items():
    for s, vd in vds.items():
      out[f"{rotulo}-semente-{s}"] = metricas_teste_restritas(restringir(vd, comum), g)
  out["detalhes"] = {"TCC": detalhes(restringir(tcc, comum)),
                     nome: detalhes(restringir(teste_por_modelo["final"][42], comum))}
  # ablações: modelo final nas 3 sementes (a partir dos modelos salvos) e TCC
  tarefas = [(s, a, k) for s in dirs for a in ("ordem", "dxdy") for k in r.particoes()]
  with mp.get_context("spawn").Pool(5) as pool:
    saidas = pool.map(q.prever, [(str(dirs[s]), cfg if s == 42 else {**cfg, "semente": s}, k, a, q.opcoes()) for s, a, k in tarefas])
  erros = [e for _, _, e in saidas if e]
  if erros:
    raise RuntimeError("\n\n".join(erros))
  por = {}
  for (s, a, _), (k, x, _) in zip(tarefas, saidas):
    por.setdefault(s, {}).setdefault(a, {})[k] = x
  out["ablacoes"] = {"final": {}, "TCC": None}
  for s in dirs:
    vd_sem = restringir(teste_por_modelo["final"][s], comum)
    out["ablacoes"]["final"][s] = ablacoes_de(vd_sem, {a: q.vd_de_resultados(x, cfg) for a, x in por[s].items()}, comum)
    out["ablacoes"]["final"][s]["head_banging_f1_zero"] = detalhes(vd_sem)["head_banging_f1_zero"]
  out["ablacoes"]["TCC"] = ablacoes_de(restringir(tcc, comum), {a: tcc_vd(a) for a in ("ordem", "dxdy")}, comum)
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--teste", action="store_true", help="rodada rápida: 1 partição, 2 épocas, 9 vídeos, sem commit; força o caminho do QF")
  ap.add_argument("--saida", default=str(EXP1))
  ap.add_argument("--forcar-ref", action="store_true", help="só com --teste: segue o caminho em que o r2-C4-aumento continua final")
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
  t0 = time.time()
  t3.log(f"critério Q: início (pid {os.getpid()}){' -- TESTE' if args.teste else ''}")
  res_p = saida / "r3-q-resultados.json"
  res = {}
  try:
    cob = cobertura()
    t3.commit("experimentos: r3-q-cobertura", [t3.pasta("r3-q-cobertura") / "cobertura.json", t3.pasta("r3-q-cobertura") / "cobertura.md", Path(__file__)])

    cfg_ref = {**q.NOVOS, **cfg_de(EXP1 / REF)}
    tcc = tcc_vd()
    ref_teste = {s: dados(EXP1 / d, "teste", cfg_ref if s == 42 else {**cfg_ref, "semente": s}) for s, d in REF_SEMENTES.items()}
    ref_val = dados(EXP1 / REF, "validacao", cfg_ref)

    # comparação sem treino novo (teste): TCC e r2-C4-aumento restritos a Q, no conjunto comum
    comum0 = conjunto_comum(tcc, ref_teste[42])
    res["sem_treino_teste"] = {"n_quadros": n_quadros(restringir(ref_teste[42], comum0)),
                               "TCC": metricas_teste_restritas(restringir(tcc, comum0)),
                               **{f"{REF}-semente-{s}": metricas_teste_restritas(restringir(v, comum0)) for s, v in ref_teste.items()}}
    r.gravar_json(res_p, res)

    # QF
    cfg_qf = {**cfg_ref, "normalizacao": "q_superior", "entrada": "xy_dxdy_marcas"}
    if args.teste:
      cfg_qf["max_epocas"] = 2
    met = t3.rodar3(QF, cfg_qf, "QF: só quadros válidos por Q, centro nos ombros, escala ombros-cabeça, interpolação até 0,5 s com marcas")
    t3.commit(f"experimentos: {QF}", q.arquivos(t3.pasta(QF)) + [Path(__file__)])
    if met is None:
      raise RuntimeError("o treino do QF falhou (ver r3-q-QF/erro.txt)")
    qf_val = dados(t3.pasta(QF), "validacao", cfg_qf)
    comum_val = conjunto_comum(ref_val, qf_val)
    m_qf, m_ref = q.met_val(restringir(qf_val, comum_val)), q.met_val(restringir(ref_val, comum_val))
    ok, soma = r.supera(m_qf["agregado_validacao"], m_ref["agregado_validacao"])
    res["decisao"] = {"n_quadros_validacao": n_quadros(restringir(ref_val, comum_val)),
                      "n_quadros_validacao_ref_validos_q": n_quadros(restringir(ref_val, conjunto_comum(ref_val))),
                      "QF": m_qf, REF: m_ref, "passa": ok, "soma_dos_ganhos": soma}
    r.gravar_json(res_p, res)
    t3.log(f"{QF}: validação restrita a Q ({res['decisao']['n_quadros_validacao']} quadros) passa={ok} | "
           + " ".join(f"{m}: {m_qf['agregado_validacao'][m]:.3f} vs {m_ref['agregado_validacao'][m]:.3f}" for m in q.METRICAS))

    if (ok or args.teste) and not (args.teste and args.forcar_ref):
      dirs = {42: t3.pasta(QF)}
      for s in (1000, 2000):
        e = f"r3-q-QF-semente-{s}"
        if t3.rodar3(e, {**cfg_qf, "semente": s}, f"QF, semente {s}", com_teste=True) is not None:
          dirs[s] = t3.pasta(e)
        t3.commit(f"experimentos: {e}", q.arquivos(t3.pasta(e)) + [Path(__file__)])
      q.testar(QF, cfg_qf)
      qf_teste = {s: dados(d, "teste", cfg_qf if s == 42 else {**cfg_qf, "semente": s}) for s, d in dirs.items()}
      comum1 = conjunto_comum(tcc, ref_teste[42], qf_teste[42])
      res["final"] = avaliacao_completa(QF, dirs, cfg_qf, qf_val, comum_val, tcc, comum1, {"final": qf_teste, REF: ref_teste})
    else:
      dirs = {s: EXP1 / d for s, d in REF_SEMENTES.items()}
      res["final"] = avaliacao_completa(REF, dirs, cfg_ref, ref_val, conjunto_comum(ref_val), tcc, comum0, {"final": ref_teste})
    res["tempo_s"] = time.time() - t0
    r.gravar_json(res_p, res)
    escrever_resumo(res, cob)
    t3.commit("experimentos: critério Q (decisão, avaliação e resumo)", [res_p, saida / "resumo-rodada3.md", Path(__file__)])
    t3.log(f"critério Q: fim; modelo final {res['final']['modelo_final']}")
  except Exception:
    t3.log("ERRO GERAL (critério Q):\n" + traceback.format_exc())
    raise


# ---------- resumo ----------

def escrever_resumo(res, cob):
  f3, M = q.f3, q.METRICAS
  M4 = M + ["macro_f1_quadro_anotados"]
  cel_v = lambda m, x: f"{f3(m['agregado_validacao'][x])} / {f3(m['media_entre_particoes'][x])} ± {f3(m['desvio_entre_particoes'][x])}"
  cel_t = lambda t, x: f"{f3(t['agregado_56_videos'][x])} / {f3(t['media_entre_particoes'][x])} ± {f3(t['desvio_entre_particoes'][x])}"
  fin = res["final"]
  sementes = lambda pref: [fin[k] for k in fin if k.startswith(pref) and "-semente-" in k]
  med = lambda ts, x, conv="agregado_56_videos": f"{f3(np.mean([t[conv][x] for t in ts]))} ± {f3(np.std([t[conv][x] for t in ts]))}"
  d, nome = res["decisao"], fin["modelo_final"]
  linhas = ["# Resumo da rodada 3", "",
            "**A primeira avaliação da I3 foi substituída.** A regra da I1 tirava os quadros sem ombro ou quadril recuperável só da avaliação dos modelos novos "
            "(cerca de 44% dos quadros), e a comparação com o `r2-C4-aumento` misturava conjuntos de quadros diferentes. Ela foi corrigida de duas formas, que valem juntas: "
            "a cobertura total (todos os modelos nos mesmos quadros) e o critério Q, decidido pelo orientador, que aplica a qualidade da pose como critério de dados "
            "a todos os modelos antes da separação entre treino, validação e teste (adendos de 2026-10-20 e 2026-10-21 em `protocolo-rodada3.md`).", ""]
  cob_dec = Path(t3.O3["saida"]) / "r3-cobertura-decisao.json"
  if cob_dec.exists():
    c = r.ler_json(cob_dec)
    sub = list(c["variantes"].values())[0]["subconjunto_I1"]["quadros_validacao"]
    linhas += ["## Cobertura total (todos os quadros com pessoa; validação, agregado: F1 evento / acurácia bal. / macro F1 quadros)", "",
               f"`{REF}` " + " / ".join(f3(c["metricas_referencia"][m]) for m in M) + "; " + "; ".join(
                 f"{v}: " + " / ".join(f3(x["cobertura_total"]["agregado_validacao"][m]) for m in M) for v, x in c["variantes"].items())
               + ". Nenhuma variante passa.",
               f"No subconjunto de {sub} quadros que a I1 mantém (resultado já calculado, citado): `{REF}` "
               + " / ".join(f3(list(c["variantes"].values())[0]["subconjunto_I1"]["referencia_no_subconjunto"][m]) for m in M) + "; " + "; ".join(
                 f"{v}: " + " / ".join(f3(x["subconjunto_I1"]["variante"][m]) for m in M) for v, x in c["variantes"].items()) + ".", ""]
  linhas += ["## Critério Q", "",
             "Quadro válido: os dois ombros e pelo menos um ponto da cabeça com confiança ≥ 0,5. Excluídos (quadros / anotados / episódios): " + "; ".join(
               f"*{c}* {f3(x['fracao_quadros'])} / {f3(x['fracao_anotados'])} / {x['episodios_excluidos']} de {x['episodios']}" for c, x in cob["por_classe"].items())
             + f". {cob['total']['episodios_excluidos']} episódios saem do gabarito; vídeos sem quadro válido: {len(cob['videos_sem_quadro_valido'])}. "
             "Detalhes: `r3-q-cobertura/cobertura.md`.", "",
             f"**Decisão (validação, {d['n_quadros_validacao']} quadros comuns; agregado / média ± desvio):**", "",
             "| | F1 evento | acurácia bal. | macro F1 quadros |", "|---|---|---|---|",
             f"| `{REF}` restrito a Q | {' | '.join(cel_v(d[REF], x) for x in M)} |",
             f"| QF | {' | '.join(cel_v(d['QF'], x) for x in M)} |", "",
             f"QF {'passa' if d['passa'] else 'não passa'} no critério; modelo final: `{nome}`.", "",
             f"**Teste do modelo final nos quadros válidos por Q ({fin['n_quadros_teste']} quadros comuns; agregado / média ± desvio entre partições).** "
             f"Pós-processamento escolhido na validação restrita: {fin['pos_processamento']['escolhido']}; ele só muda o F1 por evento.", "",
             "| | F1 evento | acurácia bal. | macro F1 quadros | macro F1 anotados | F1 evento com pós |", "|---|---|---|---|---|---|",
             f"| TCC | {' | '.join(cel_t(fin['TCC'], x) for x in M4)} | {f3(fin['TCC']['f1_evento_com_pos']['agregado'])} / {f3(fin['TCC']['f1_evento_com_pos']['media'])} |"]
  for rot, pref in ((f"`{REF}`" if nome != REF else f"`{REF}` (final)", REF if nome != REF else "final"), (f"`{nome}` (final)" if nome != REF else None, "final" if nome != REF else None)):
    if not rot:
      continue
    ts = sementes(pref)
    linhas.append(f"| {rot}, 3 sementes, agregado (média ± desvio entre sementes) | {' | '.join(med(ts, x) for x in M4)} | "
                  f"{f3(np.mean([t['f1_evento_com_pos']['agregado'] for t in ts]))} ± {f3(np.std([t['f1_evento_com_pos']['agregado'] for t in ts]))} |")
    linhas.append(f"| {rot}, 3 sementes, média entre partições | {' | '.join(med(ts, x, 'media_entre_particoes') for x in M4)} | "
                  f"{f3(np.mean([t['f1_evento_com_pos']['media'] for t in ts]))} |")
  dt = fin["detalhes"]
  for mod in ("TCC", nome):
    pc = dt[mod]["por_classe"]["quadros_anotados"]["agregado"]
    pt = dt[mod]["por_classe"]["todos_os_quadros"]["agregado"]
    mz = dt[mod]["confusao_quadros_anotados"]["matriz"]
    hb = dt[mod]["head_banging_f1_zero"]
    linhas += ["", f"*{mod}*, semente 42: F1 por classe em todos os quadros (P / R / F1): " + "; ".join(
      f"{c} {f3(pt[c]['precision'])} / {f3(pt[c]['recall'])} / {f3(pt[c]['f1-score'])}" for c in r.CLASSES)
      + ". Nos anotados: " + "; ".join(f"{c} {f3(pc[c]['precision'])} / {f3(pc[c]['recall'])} / {f3(pc[c]['f1-score'])}" for c in r.CLASSES[:3])
      + f". Matriz (linhas anotadas AF, HB, SP; colunas AF, HB, SP, fundo): {mz[:3]}. *Head banging* com F1 zero: {hb['n_zero']} de {hb['n_videos_com_quadros_validos']} "
      f"vídeos com quadros válidos (de {hb['n_videos_head_banging']})."]
  ab = fin["ablacoes"]
  linhas += ["", f"Ablações, macro F1 nos anotados com 3 classes (agregado / média): *{nome}* por semente: " + "; ".join(
    f"{a}: " + ", ".join(f"{f3(ab['final'][s][a]['macro_f1_anotados_3_classes']['agregado'])} / {f3(ab['final'][s][a]['macro_f1_anotados_3_classes']['media'])}"
                         for s in ab["final"]) for a in ("sem_ablacao", "ordem", "dxdy"))
    + "; *TCC*: " + "; ".join(f"{a}: {f3(ab['TCC'][a]['macro_f1_anotados_3_classes']['agregado'])} / {f3(ab['TCC'][a]['macro_f1_anotados_3_classes']['media'])}"
                              for a in ("sem_ablacao", "ordem", "dxdy")) + ".",
    "Fração da classe do vídeo, dentro / fora, sem ablação: " + "; ".join(
      f"*{nome}* semente {s}: " + ", ".join(f"{c} {f3(x['dentro'])}/{f3(x['fora'])}" for c, x in ab["final"][s]["sem_ablacao"]["fracao_classe_do_video"].items())
      for s in ab["final"]) + "; *TCC*: " + ", ".join(f"{c} {f3(x['dentro'])}/{f3(x['fora'])}" for c, x in ab["TCC"]["sem_ablacao"]["fracao_classe_do_video"].items()) + ".",
    f"*Head banging* com F1 zero por semente (*{nome}*): " + ", ".join(
      f"{ab['final'][s]['head_banging_f1_zero']['n_zero']} de {ab['final'][s]['head_banging_f1_zero']['n_videos_com_quadros_validos']}" for s in ab["final"])
    + ". Tudo em `r3-q-resultados.json`" + (f"; curva de erro: `{QF}/curva-erro.pdf`." if nome == QF else ".")]
  primeira = Path(t3.O3["saida"]) / "r3-primeira-avaliacao-resumo.md"
  if primeira.exists():
    ant = primeira.read_text(encoding="utf-8").splitlines()
    k = next(j for j, l in enumerate(ant) if l.startswith("| variante"))
    linhas += ["", "## Primeira avaliação (substituída: subconjunto de quadros só nos modelos novos)", "",
               *ant[k:k + next(j for j, l in enumerate(ant[k:]) if not l.strip())], "", "Detalhes da primeira avaliação: `r3-avaliacao-final/` (não usar)."]
  (Path(t3.O3["saida"]) / "resumo-rodada3.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")


if __name__ == "__main__":
  main()
