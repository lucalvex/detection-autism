# ============================================================
# ARQUIVO: experimentos/corrigir_cobertura_r3.py
#
# O QUE FAZ: corrige a comparação da rodada 3 (adendo "cobertura total"
# em protocolo_rodada3.md), sem treinar nada. Os modelos já treinados de
# I1, I2, I3-treino (a) e (b) geram predições em todos os quadros que o
# r2_C4_aumento avalia: nos quadros em que ombros ou quadris não podem ser
# recuperados em 0,5 s, centro e escala vêm da normalização do
# r2_C4_aumento e esses pontos ficam marcados como ausentes. Confere que
# os quadros avaliados são exatamente os mesmos, refaz a decisão na
# validação e, se alguma variante passar, refaz a avaliação de teste.
# Retomável: pula o que já foi gravado para o mesmo modelo.
#
# Uso (da raiz do repositório):
#   .venv/Scripts/python.exe experimentos/corrigir_cobertura_r3.py
# ============================================================

import json
import multiprocessing as mp
import pickle
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(RAIZ / "experimentos"), str(RAIZ / "src")]
import rodar_rodada3 as t3  # noqa: E402  (aplica as substituições das rodadas 2 e 3)

r, q = t3.r, t3.q
REF = t3.REF
VARIANTES = ["r3_I1_interpolacao_limitada", "r3_I2_limitada_confianca", "r3_I3_treino_a", "r3_I3_treino_b"]
SEMENTES_I3B = {1000: "r3_final_semente_1000", 2000: "r3_final_semente_2000"}
t3.RENOMEIA.update({"r2_D_posprocessamento": "r3_cobertura_posprocessamento", "r2_avaliacao_final": "r3_cobertura_avaliacao_final"})
_preparar_r3 = r.preparar_video


def preparar_video(vid, cfg):
  """Cobertura total: todos os quadros com pessoa; ombros e quadris não recuperáveis em 0,5 s
  usam a normalização do r2_C4_aumento (interpolação sem limite) e ficam ausentes."""
  if not cfg.get("cobertura_total"):
    return _preparar_r3(vid, cfg)
  import evaluate_temporal as et
  d = np.load(r.caminho("poses", cfg["pose"], f"{vid}.npz"))
  quadros, xy, conf = d["frames"].astype(np.int64), d["xy"].astype(np.float32), d["conf"].astype(np.float32)
  if len(quadros) == 0:
    return None
  fps = et.load_video_fps(vid)
  xy_l, m, _ = t3.interpolacao_limitada(quadros, xy, conf, fps)
  xy_u = xy.astype(np.float64).copy()  # mesma interpolação sem limite de rodar_tudo.preparar_video (normalização robusta)
  for k in t3.NORMALIZACAO_PONTOS:
    ok = conf[:, k] >= r.LIMIAR_CONF
    if ok.any() and not ok.all():
      for c in (0, 1):
        xy_u[~ok, k, c] = np.interp(quadros[~ok], quadros[ok], xy_u[ok, k, c])
  quadril = (xy_u[:, 11] + xy_u[:, 12]) / 2
  escala = np.linalg.norm((xy_u[:, 5] + xy_u[:, 6]) / 2 - quadril, axis=1)
  ruim = escala < 1
  if ruim.all():
    escala[:] = 1
  elif ruim.any():
    idx = np.maximum.accumulate(np.where(ruim, -1, np.arange(len(escala))))
    escala = escala[np.where(idx < 0, np.argmax(~ruim), idx)]
  P = np.clip((xy_l - quadril[:, None]) / escala[:, None, None], -r.LIMITE_NORM, r.LIMITE_NORM).astype(np.float32)
  P[~m] = 0
  gt, _ = et.ground_truth_frames(vid, int(quadros[-1]) + 1, fps)
  return {"quadros": quadros, "P": P, "M": m, "C": conf, "fps": fps,
          "rotulo": np.array([r.CLASSES.index(g) for g in gt], np.int64), "com_conf": cfg["entrada"].endswith("_conf")}


r.preparar_video = preparar_video


def cfg_de(exp_id):
  return {**t3.cfg_limpo(r.ler_json(t3.pasta(exp_id) / "config.json")), "cobertura_total": True}


def prever_particao(args):
  origem, destino, cfg, k, val_videos, partes, opc = args
  q.aplicar_opcoes(*opc)
  try:
    import torch
    torch.set_num_threads(1)
    m = q.carregar_modelo(origem, cfg, k)
    (Path(destino) / "modelos").mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(origem) / "modelos" / f"particao_{k}.pt", Path(destino) / "modelos" / f"particao_{k}.pt")
    teste = [v for v in r.videos() if r.PARTICOES[v] == k]
    p = Path(destino) / f"janelas_{k}.pkl"
    saida = pickle.loads(p.read_bytes()) if p.exists() else {}
    for parte in partes:
      if parte not in saida:
        saida[parte] = q.inferir_lstm(m, {}, val_videos if parte == "validacao" else teste, cfg)
    p.write_bytes(pickle.dumps(saida))
    return k, None
  except Exception:
    return k, traceback.format_exc()


def predizer(origem_id, destino_id, cfg, partes):
  destino = t3.pasta(destino_id)
  destino.mkdir(parents=True, exist_ok=True)
  r.gravar_json(destino / "config.json", {"descricao": f"cobertura total, modelos de {origem_id}", "origem": origem_id, **cfg})
  val = q.validacao()
  feitas = all((destino / f"janelas_{k}.pkl").exists() and all(p in pickle.loads((destino / f"janelas_{k}.pkl").read_bytes()) for p in partes)
               for k in r.particoes())
  if feitas:
    return
  with mp.get_context("spawn").Pool(5) as pool:
    erros = [e for _, e in pool.map(prever_particao, [(str(t3.pasta(origem_id)), str(destino), cfg, k, val[k], partes, q.opcoes())
                                                      for k in r.particoes()]) if e]
  if erros:
    raise RuntimeError("\n\n".join(erros))


def quadros_avaliados(exp_dir, parte):
  out = {}
  for k in r.particoes():
    for vid, x in pickle.loads((Path(exp_dir) / f"janelas_{k}.pkl").read_bytes())[parte].items():
      out[(k, vid)] = np.asarray(x["quadro_fim"])
  return out


def conferir_mesmos_quadros(exp_id, parte):
  ref, var = quadros_avaliados(t3.pasta(REF), parte), quadros_avaliados(t3.pasta(exp_id), parte)
  iguais = ref.keys() == var.keys() and all(np.array_equal(ref[c], var[c]) for c in ref)
  n_ref, n_var = sum(len(x) for x in ref.values()), sum(len(x) for x in var.values())
  if not iguais:
    raise RuntimeError(f"{exp_id} ({parte}): quadros diferentes do {REF} ({n_var} contra {n_ref})")
  return n_ref


def subconjunto_ref(var_original):
  """Só informação: o r2_C4_aumento avaliado nos quadros que a I1 manteve (validação)."""
  mantidos = quadros_avaliados(t3.pasta(var_original), "validacao")
  vd = r.dados_avaliacao(t3.pasta(REF), "validacao", t3.cfg_limpo(r.ler_json(t3.pasta(REF) / "config.json")))
  for chave, x in vd.items():
    k, vid = chave.split("|")
    fica = set(mantidos.get((int(k), vid), np.array([], int)).tolist())
    x["windows"] = [w for w in x["windows"] if w["quadro_fim"] in fica]
  vd = {c: x for c, x in vd.items() if x["windows"]}
  return q.met_val(vd), sum(len(x["windows"]) for x in vd.values())


def comparar_pos(pos, dirs, cfg):
  p = t3.pasta("r3_cobertura_avaliacao_final") / "pos_processamento_comparado.json"
  t3.gravado_para(p, "modelo_final", pos["modelo"])
  if p.exists():
    return r.ler_json(p)
  g = pos["grade"][pos["escolhido"]]
  out = {"modelo_final": pos["modelo"], "escolhido": pos["escolhido"], "modelos": {}}
  cfg_ref = t3.cfg_limpo(r.ler_json(t3.pasta(REF) / "config.json"))
  for nome, ds, c in (("modelo_final", dirs, cfg), (REF, t3.REF_SEMENTES, cfg_ref)):
    for s, d in ds.items():
      vd = r.dados_avaliacao(t3.pasta(d), "teste", c if s == 42 else {**c, "semente": s})
      out["modelos"][f"{nome}_semente_{s}"] = {"sem": q.f1_evento_pos(vd, 0, 0), "com": q.f1_evento_pos(vd, g["mediana_s"], g["juncao_s"])}
  vd = t3.tcc_vd()
  out["modelos"]["TCC"] = {"sem": q.f1_evento_pos(vd, 0, 0), "com": q.f1_evento_pos(vd, g["mediana_s"], g["juncao_s"])}
  r.gravar_json(p, out)
  return out


def main():
  t0 = time.time()
  saida = Path(t3.O3["saida"])
  t3.log("cobertura total: início")
  met_ref = q.met_val_lidas(REF)
  decisao = {"referencia": REF, "metricas_referencia": met_ref["agregado_validacao"], "variantes": {}}
  passou = []
  for exp_id in VARIANTES:
    destino_id = f"r3_cobertura_{exp_id[3:]}"
    cfg = cfg_de(exp_id)
    predizer(exp_id, destino_id, cfg, ["validacao"])
    n = conferir_mesmos_quadros(destino_id, "validacao")
    p = t3.pasta(destino_id) / "metricas_validacao.json"
    met = r.ler_json(p) if p.exists() else q.met_val(r.dados_avaliacao(t3.pasta(destino_id), "validacao", cfg))
    r.gravar_json(p, met)
    ok, soma = r.supera(met["agregado_validacao"], met_ref["agregado_validacao"])
    sub, n_sub = subconjunto_ref(exp_id)
    original = r.ler_json(t3.pasta(exp_id) / "metricas_validacao.json")
    decisao["variantes"][exp_id] = {"quadros_validacao": n, "passa": ok, "soma_dos_ganhos": soma, "cobertura_total": met,
                                    "subconjunto_I1": {"quadros_validacao": n_sub, "variante": original["agregado_validacao"],
                                                       "referencia_no_subconjunto": sub["agregado_validacao"]}}
    t3.log(f"{destino_id}: {n} quadros de validação, iguais aos do {REF} | passa={ok} | "
           + " ".join(f"{m}={met['agregado_validacao'][m]:.3f}" for m in q.METRICAS))
    if ok:
      passou.append((soma, exp_id, destino_id, cfg))
    t3.commit(f"experimentos: {destino_id}", [t3.pasta(destino_id) / "config.json", p, Path(__file__)])

  decisao["vencedora"] = max(passou)[1] if passou else None
  r.gravar_json(saida / "r3_cobertura_decisao.json", decisao)
  bloco = None
  if passou:
    _, exp_id, final_id, cfg = max(passou)
    if exp_id != "r3_I3_treino_b":
      decisao["observacao"] = f"{exp_id} não tem as sementes 1000 e 2000 treinadas; treiná-las exige nova autorização"
      r.gravar_json(saida / "r3_cobertura_decisao.json", decisao)
      dirs = {42: final_id}
    else:
      dirs = {42: final_id}
      for s, d in SEMENTES_I3B.items():
        dirs[s] = f"r3_cobertura_final_semente_{s}"
        predizer(d, dirs[s], {**cfg, "semente": s}, ["validacao", "teste"])
    predizer(exp_id, final_id, cfg, ["validacao", "teste"])
    n_teste = {s: conferir_mesmos_quadros(d, "teste") for s, d in dirs.items()}
    for s, d in dirs.items():
      pt = t3.pasta(d) / "metricas_teste.json"
      if not pt.exists():
        r.gravar_json(pt, r.metricas_teste(r.dados_avaliacao(t3.pasta(d), "teste", cfg if s == 42 else {**cfg, "semente": s})))
    t3.gravado_para(t3.pasta("r3_cobertura_posprocessamento") / "posprocessamento.json", "modelo", final_id)
    pos = q.posprocessamento(final_id, cfg)
    t3.gravado_para(t3.pasta("r3_cobertura_avaliacao_final") / "avaliacao_final.json", "modelo_final", final_id)
    av = q.avaliacao_final(final_id, cfg, pos, dirs)
    comp = comparar_pos(pos, dirs, cfg)
    decisao["quadros_teste"] = n_teste
    r.gravar_json(saida / "r3_cobertura_decisao.json", decisao)
    bloco = t3.bloco_final(final_id, av, comp)
  escrever_resumo(decisao, bloco)
  arquivos = [saida / "r3_cobertura_decisao.json", saida / "resumo_rodada3.md", saida / "r3_primeira_avaliacao_resumo.md", Path(__file__)]
  for d in set(["r3_cobertura_posprocessamento", "r3_cobertura_avaliacao_final"] + [f"r3_cobertura_final_semente_{s}" for s in SEMENTES_I3B]
               + [f"r3_cobertura_{v[3:]}" for v in VARIANTES]):
    for nome in ("config.json", "metricas_validacao.json", "metricas_teste.json", "posprocessamento.json", "avaliacao_final.json",
                 "pos_processamento_comparado.json"):
      arquivos.append(t3.pasta(d) / nome)
  t3.commit("experimentos: rodada 3 com cobertura total e resumo", arquivos)
  t3.log(f"cobertura total: fim em {(time.time() - t0) / 60:.0f} min; vencedora: {decisao['vencedora']}")


def escrever_resumo(dec, bloco):
  f3, duas = q.f3, q.duas
  ref = q.met_val_lidas(REF)
  # o resumo da primeira avaliação é guardado uma vez, antes de ser reescrito
  primeira = Path(t3.O3["saida"]) / "r3_primeira_avaliacao_resumo.md"
  if not primeira.exists():
    shutil.copy(Path(t3.O3["saida"]) / "resumo_rodada3.md", primeira)
  antigo = primeira.read_text(encoding="utf-8").splitlines()
  i = next(j for j, l in enumerate(antigo) if l.startswith("| variante"))
  tabela_antiga = antigo[i:i + next(j for j, l in enumerate(antigo[i:]) if not l.strip())]
  extras = [l for l in antigo if l.startswith(("Quadros removidos", "Descarte da I3", "- (", "Detalhes:"))]
  linhas = ["# Resumo da rodada 3", "",
            "**A primeira avaliação foi substituída.** A regra da I1 tirava da validação e do teste os quadros sem ombro ou quadril recuperável "
            "(cerca de 44% dos quadros), e I1, I2 e I3-treino eram comparadas ao `r2_C4_aumento` em conjuntos de quadros diferentes. "
            "A decisão agora vale pela cobertura total: os mesmos modelos, sem novo treino, avaliados exatamente nos mesmos quadros do "
            "`r2_C4_aumento` (adendo de cobertura total em `protocolo_rodada3.md`).", "",
            "## Cobertura total (validação; agregado / média ± desvio entre partições)", "",
            "| variante | quadros de validação | F1 evento IoU 0,3 | acurácia bal. | macro F1 quadros | passa |", "|---|---|---|---|---|---|",
            f"| {REF} (referência) | {dec['variantes'][VARIANTES[0]]['quadros_validacao']} | {' | '.join(duas(ref, m) for m in q.METRICAS)} | -- |"]
  for v, x in dec["variantes"].items():
    linhas.append(f"| {v} | {x['quadros_validacao']} | {' | '.join(duas(x['cobertura_total'], m) for m in q.METRICAS)} | {'sim' if x['passa'] else 'não'} |")
  linhas += ["", f"**Decisão:** " + (f"{dec['vencedora']} passa no critério com cobertura total e é o modelo final." if dec["vencedora"]
                                    else f"nenhuma variante passa no critério com cobertura total; o modelo final continua `{REF}`.")]
  if dec.get("observacao"):
    linhas.append(dec["observacao"] + ".")
  linhas += ["", "Só como informação, no subconjunto de quadros que a I1 mantém (validação, agregado): " + "; ".join(
    f"{v}: " + ", ".join(f"{f3(x['subconjunto_I1']['variante'][m])} contra {f3(x['subconjunto_I1']['referencia_no_subconjunto'][m])}" for m in q.METRICAS)
    for v, x in dec["variantes"].items()) + f" (variante contra `{REF}` nos mesmos {list(dec['variantes'].values())[0]['subconjunto_I1']['quadros_validacao']} quadros; "
    "métricas na ordem F1 evento, acurácia bal., macro F1 quadros).", "", *extras]
  if bloco:
    linhas += ["", *[l.replace("## Novo modelo final", "## Modelo final, avaliação com cobertura total") for l in bloco]]
  linhas += ["", "## Primeira avaliação (substituída: subconjunto de quadros)", "", *tabela_antiga, "",
             "Detalhes da primeira avaliação: `r3_avaliacao_final/` (não usar)."]
  (Path(t3.O3["saida"]) / "resumo_rodada3.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")


if __name__ == "__main__":
  main()
