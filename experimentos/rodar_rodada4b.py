# ============================================================
# ARQUIVO: experimentos/rodar_rodada4b.py
#
# O QUE FAZ: rodada 4b (experimentos/protocolo-rodada4b.md):
#   A. teste do r2-C4-aumento e do TCC nos quadros válidos por Q, com e
#      sem os 5 vídeos problemáticos (sem treino);
#   B. frequência dominante dos movimentos dentro dos intervalos anotados,
#      com janelas de 8 s (sem treino);
#   C. C1 (janela de 4 s), C2 (P2 com janela de 4 s) e C3 (treino sem os
#      5 vídeos problemáticos), cada uma contra o r2-C4-aumento, com o
#      protocolo da rodada 4; avaliação final completa se alguma passar.
# Reaproveita rodar_rodada4.py e anteriores (importados, não alterados).
# Retomável; um commit por etapa; commit que falha não deixa nada no índice.
#
# Uso (da raiz do repositório):
#   .venv/Scripts/python.exe experimentos/rodar_rodada4b.py
#   .venv/Scripts/python.exe experimentos/rodar_rodada4b.py --teste --saida <pasta>
# ============================================================

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(RAIZ / "experimentos"), str(RAIZ / "src")]
import rodar_rodada4 as r4  # noqa: E402  (modelo P2, janelas congeladas, dependência do movimento)

rq, t3, r, q = r4.rq, r4.t3, r4.r, r4.q
EXP1, REF, REF_SEMENTES, METRICAS = r4.EXP1, r4.REF, r4.REF_SEMENTES, r4.METRICAS
PROBLEMATICOS = ["v_ArmFlapping_03", "v_HeadBanging_24", "v_Spinning_09", "v_Spinning_12", "v_Spinning_16"]
FS = 15
RENOMEIA = {"r4-avaliacao-final": "r4b-avaliacao-final"}


def pasta(i):
  i = RENOMEIA.get(i, i)
  return Path(t3.O3["saida"]) / i if i.startswith("r4b-") else EXP1 / i


def log(msg):
  linha = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
  print(linha, flush=True)
  with open(Path(t3.O3["saida"]) / "progresso_rodada4b.log", "a", encoding="utf-8") as f:
    f.write(linha + "\n")


r4.pasta = t3.pasta = q.pasta = pasta
r4.log = t3.log = q.log = log
commit = t3.commit

_treinar_anterior = q.treinar_lstm


def treinar_lstm(args):
  """C3: tira do treino de cada partição os vídeos de cfg['excluir_treino']; validação e teste não mudam."""
  exp_dir, cfg, k, val, com_teste, treino, opc = args
  if cfg.get("excluir_treino"):
    q.aplicar_opcoes(*opc)
    base = treino or [v for v in r.videos() if r.PARTICOES[v] != k and v not in val]
    args = (exp_dir, cfg, k, val, com_teste, [v for v in base if v not in cfg["excluir_treino"]], opc)
  return _treinar_anterior(args)


q.treinar_lstm = treinar_lstm


# ---------- A ----------

def parte_a(cfg_ref, ref_teste, tcc):
  p = pasta("r4b-A-problematicos") / "resultado.json"
  if p.exists():
    return r.ler_json(p)
  pos = r.ler_json(EXP1 / "r3-q-resultados.json")["final"]["pos_processamento"]
  g = pos["grade"][pos["escolhido"]]
  comum = rq.conjunto_comum(tcc, ref_teste[42])
  sem = {k: v for k, v in comum.items() if k.split("|")[-1] not in PROBLEMATICOS}
  out = {"pos_processamento": pos["escolhido"], "videos_excluidos": PROBLEMATICOS}
  for nome, cj in (("todos", comum), ("sem_problematicos", sem)):
    out[nome] = {"n_quadros": rq.n_quadros(rq.restringir(ref_teste[42], cj)),
                 "n_episodios": sum(len(x["gt_intervals"]) for x in rq.restringir(ref_teste[42], cj).values()),
                 "TCC": rq.metricas_teste_restritas(rq.restringir(tcc, cj), g),
                 "ref": [rq.metricas_teste_restritas(rq.restringir(v, cj), g) for v in ref_teste.values()]}
  saem = {}
  for vid in PROBLEMATICOS:
    x = rq.restringir({vid: ref_teste[42][vid]}, comum).get(vid) if vid in ref_teste[42] else None
    c = saem.setdefault(r.classe_do_video(vid), {"videos": 0, "quadros": 0, "episodios": 0})
    c["videos"] += 1
    if x:
      c["quadros"] += len(x["windows"])
      c["episodios"] += len(x["gt_intervals"])
  out["saem_por_classe"] = saem
  r.gravar_json(p, out)
  return out


# ---------- B ----------

def sinal(P, classe):
  if classe == "ArmFlapping":
    return np.stack([P[:, 9, 0] - P[:, 5, 0], P[:, 9, 1] - P[:, 5, 1], P[:, 10, 0] - P[:, 6, 0], P[:, 10, 1] - P[:, 6, 1]], 1)
  if classe == "HeadBanging":
    return (P[:, 0, 1] - (P[:, 5, 1] + P[:, 6, 1]) / 2)[:, None]
  return np.linalg.norm(P[:, 6] - P[:, 5], axis=1)[:, None]


def dominante(W):
  Z = (W - W.mean(0)) * np.hanning(len(W))[:, None]
  pot = (np.abs(np.fft.rfft(Z, n=512, axis=0)) ** 2).sum(1)
  f = np.fft.rfftfreq(512, 1 / FS)
  banda = (f >= 0.25) & (f <= 7)
  return float(f[banda][pot[banda].argmax()])


def parte_b(cfg_ref):
  p = pasta("r4b-B-frequencia") / "frequencia.json"
  if p.exists():
    return r.ler_json(p)
  import evaluate_temporal as et
  dados = {c: {"freq": [], "fps_baixo": [], "episodios": 0, "episodios_com_janela": 0, "videos": set()} for c in r.CLASSES[:3]}
  for vid in r.videos():
    v = r.preparar_video(vid, cfg_ref)
    if v is None:
      continue
    t = v["quadros"] / v["fps"]
    _, iv = et.ground_truth_frames(vid, int(v["quadros"][-1]) + 1, v["fps"])
    for lab, a, b in iv:
      d = dados[lab]
      d["episodios"] += 1
      dentro = np.where((t >= a) & (t <= b))[0]
      if len(dentro) < 2:
        continue
      S = sinal(v["P"][dentro].astype(np.float64), lab)
      tt = t[dentro]
      teve = False
      for seg in np.split(np.arange(len(tt)), np.where(np.diff(tt) > 0.5 + 1 / v["fps"])[0] + 1):
        if tt[seg[-1]] - tt[seg[0]] < 8:
          continue
        grade = np.arange(tt[seg[0]], tt[seg[-1]] + 1e-9, 1 / FS)
        Sg = np.stack([np.interp(grade, tt[seg], S[seg, j]) for j in range(S.shape[1])], 1)
        for ini in range(0, len(grade) - 8 * FS + 1, FS):
          d["freq"].append(dominante(Sg[ini:ini + 8 * FS]))
          d["fps_baixo"].append(v["fps"] < 15)
          teve = True
      if teve:
        d["episodios_com_janela"] += 1
        d["videos"].add(vid)
  est = lambda x: {"mediana": float(np.median(x)), "p10": float(np.percentile(x, 10)), "p90": float(np.percentile(x, 90)), "n_janelas": len(x)} if len(x) else None
  out = {"janela_s": 8, "passo_s": 1, "banda_hz": [0.25, 7], "classes": {}}
  for c, d in dados.items():
    f = np.array(d["freq"])
    f_ok = f[~np.array(d["fps_baixo"], bool)] if len(f) else f
    x = {"frequencia_hz": est(f), "frequencia_hz_sem_videos_abaixo_de_15_fps": est(f_ok),
         "ciclos_em_2s": est(np.floor(f * 2)), "ciclos_em_4s": est(np.floor(f * 4)),
         "episodios": d["episodios"], "episodios_com_janela": d["episodios_com_janela"], "videos": len(d["videos"])}
    if c == "Spinning":  # a largura dos ombros oscila duas vezes por volta
      x["rotacao_hz"] = est(f / 2)
      x["ciclos_de_rotacao_em_2s"], x["ciclos_de_rotacao_em_4s"] = est(np.floor(f / 2 * 2)), est(np.floor(f / 2 * 4))
    out["classes"][c] = x
  r.gravar_json(p, out)
  return out


# ---------- C ----------

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
    esc[-1] = "v_Spinning_16" if r.PARTICOES.get("v_Spinning_16") == 1 else esc[-1]  # um problemático no teste, se couber
    r.OPC.update({"videos": sorted(set(esc)), "particoes": [1]})
  log(f"início (pid {os.getpid()}){' -- TESTE' if args.teste else ''}")
  try:
    r4.autoverificacao()
    cfg_ref = {**q.NOVOS, **rq.cfg_de(EXP1 / REF)}
    usa = lambda vd: {k: v for k, v in vd.items() if k.split("|")[-1] in r.videos()}
    ref_teste = {s: usa(rq.dados(EXP1 / d, "teste", cfg_ref if s == 42 else {**cfg_ref, "semente": s})) for s, d in REF_SEMENTES.items()}
    tcc = rq.tcc_vd()
    a = parte_a(cfg_ref, ref_teste, tcc)
    commit("experimentos: r4b-A-problematicos", [pasta("r4b-A-problematicos") / "resultado.json", Path(__file__)])
    b = parte_b(cfg_ref)
    commit("experimentos: r4b-B-frequencia", [pasta("r4b-B-frequencia") / "frequencia.json", Path(__file__)])
    escrever_resumo(a, b, [], None)

    ref_val = usa(rq.dados(EXP1 / REF, "validacao", cfg_ref))
    ref_cong = r4.vd_congelado(EXP1 / REF, cfg_ref, "validacao")
    hist, passou, alternativas = [], [], []
    variantes = [("r4b-C1-janela-4s", {"janela": 60}, "C1: janela de 4 s"),
                 ("r4b-C2-periodicidade-4s", {"janela": 60, "arquitetura": "p4", "trajetoria": False, "concat_postura": False,
                                              "concat_periodicidade": True}, "C2: P2 (periodicidade) com janela de 4 s"),
                 ("r4b-C3-sem-problematicos", {"excluir_treino": PROBLEMATICOS}, "C3: janela de 2 s, treino sem os 5 vídeos problemáticos")]
    for exp_id, mud, desc in variantes:
      cfg = {**cfg_ref, **mud}
      if args.teste:
        cfg["max_epocas"] = 2
      met = t3.rodar3(exp_id, cfg, desc)
      if met is None:
        hist.append({"id": exp_id, "descricao": desc, "situacao": "erro"})
      else:
        vd = rq.dados(pasta(exp_id), "validacao", cfg)
        comum = rq.conjunto_comum(ref_val, vd)
        m, m_ref = q.met_val(rq.restringir(vd, comum)), q.met_val(rq.restringir(ref_val, comum))
        dep, dep_ref = r4.dependencia(vd, r4.vd_congelado(pasta(exp_id), cfg, "validacao"), comum), r4.dependencia(ref_val, ref_cong, comum)
        ok, soma = r.supera(m["agregado_validacao"], m_ref["agregado_validacao"])
        nao_piora = all(m["agregado_validacao"][x] >= m_ref["agregado_validacao"][x] - 0.01 for x in METRICAS)
        alt = (not ok) and nao_piora and dep["dependencia"] >= 2 * dep_ref["dependencia"]
        if ok:
          passou.append((soma, exp_id, cfg))
        if alt:
          alternativas.append((dep["dependencia"], exp_id, cfg))
        hist.append({"id": exp_id, "descricao": desc, "met": m, "met_ref": m_ref, "dep": dep, "dep_ref": dep_ref,
                     "n": rq.n_quadros(rq.restringir(vd, comum)), "situacao": "passa" if ok else ("alternativa" if alt else "não"),
                     "tempo": f"{(met.get('tempo_s') or 0) / 60:.0f} min"})
        r.gravar_json(pasta(exp_id) / "rodada4b.json", hist[-1])
        log(f"{exp_id}: {hist[-1]['n']} quadros, passa={ok}, alternativa={alt}, dependência {dep['dependencia']:.3f} (ref {dep_ref['dependencia']:.3f}) | "
            + " ".join(f"{x}: {m['agregado_validacao'][x]:.3f} vs {m_ref['agregado_validacao'][x]:.3f}" for x in METRICAS))
      escrever_resumo(a, b, hist, None)
      commit(f"experimentos: {exp_id}", q.arquivos(pasta(exp_id)) + [pasta(exp_id) / "rodada4b.json", saida / "resumo-rodada4b.md", Path(__file__)])

    alvo = max(passou, key=lambda x: x[0]) if passou else (max(alternativas, key=lambda x: x[0]) if alternativas else None)
    papel = "variante que passou" if passou else ("alternativa" if alternativas else None)
    if alvo is None and args.teste:
      alvo, papel = (0, hist[-1]["id"], t3.cfg_limpo(r.ler_json(pasta(hist[-1]["id"]) / "config.json"))), "forçado no teste"
    final_txt = None
    if alvo:
      fin = r4.avaliacao_final(alvo[1], alvo[2], papel, cfg_ref, ref_teste, args)
      final_txt = [l.replace("r4-avaliacao-final", "r4b-avaliacao-final") for l in r4.texto_final(fin)]
      commit("experimentos: r4b-avaliacao-final", [pasta("r4b-avaliacao-final") / "avaliacao-final.json", Path(__file__)])
    escrever_resumo(a, b, hist, final_txt)
    commit("experimentos: resumo da rodada 4b", [saida / "resumo-rodada4b.md", Path(__file__)])
    log("fim")
  except Exception:
    log("ERRO GERAL:\n" + traceback.format_exc())
    raise


# ---------- resumo ----------

def escrever_resumo(a, b, hist, final_txt):
  f3 = q.f3
  M4 = METRICAS + ["macro_f1_quadro_anotados"]
  med = lambda ts, x, conv: f"{f3(np.mean([t[conv][x] for t in ts]))} ± {f3(np.std([t[conv][x] for t in ts]))}"
  cel_t = lambda t, x: f"{f3(t['agregado_56_videos'][x])} / {f3(t['media_entre_particoes'][x])}"
  linhas = ["# Resumo da rodada 4b", "", "Protocolo: `protocolo-rodada4b.md`. Quadros válidos por Q; agregado / média entre partições.", "",
            f"## A. Sem os 5 vídeos problemáticos ({', '.join(v.replace('v_', '') for v in PROBLEMATICOS)}), teste",
            f"Saem {a['todos']['n_quadros'] - a['sem_problematicos']['n_quadros']} de {a['todos']['n_quadros']} quadros e "
            f"{a['todos']['n_episodios'] - a['sem_problematicos']['n_episodios']} de {a['todos']['n_episodios']} episódios do gabarito ("
            + "; ".join(f"*{c}*: {x['quadros']} quadros, {x['episodios']} episódios" for c, x in a["saem_por_classe"].items())
            + f"). Pós-processamento: {a['pos_processamento']}.", "",
            "| | conjunto | F1 evento | acurácia bal. | macro F1 quadros | macro F1 anotados | F1 evento com pós |", "|---|---|---|---|---|---|---|"]
  for nome in ("todos", "sem_problematicos"):
    x = a[nome]
    linhas.append(f"| TCC | {nome.replace('_', ' ')} | {' | '.join(cel_t(x['TCC'], m) for m in M4)} | "
                  f"{f3(x['TCC']['f1_evento_com_pos']['agregado'])} / {f3(x['TCC']['f1_evento_com_pos']['media'])} |")
    linhas.append(f"| `{REF}` (3 sementes, agregado; média entre partições) | {nome.replace('_', ' ')} | "
                  + " | ".join(f"{med(x['ref'], m, 'agregado_56_videos')}; {f3(np.mean([t['media_entre_particoes'][m] for t in x['ref']]))}" for m in M4)
                  + f" | {f3(np.mean([t['f1_evento_com_pos']['agregado'] for t in x['ref']]))}; {f3(np.mean([t['f1_evento_com_pos']['media'] for t in x['ref']]))} |")
  linhas += ["", "## B. Frequência dominante nos intervalos anotados (janelas de 8 s)", "",
             "| classe | frequência (Hz): mediana [p10; p90] | ciclos em 2 s | ciclos em 4 s | janelas / episódios com janela / vídeos |", "|---|---|---|---|---|"]
  for c, x in b["classes"].items():
    if not x["frequencia_hz"]:
      linhas.append(f"| *{c}* | sem janela de 8 s | -- | -- | 0 / 0 de {x['episodios']} / 0 |")
      continue
    e = lambda k: f"{f3(x[k]['mediana'])} [{f3(x[k]['p10'])}; {f3(x[k]['p90'])}]"
    linhas.append(f"| *{c}* | {e('frequencia_hz')} | {e('ciclos_em_2s')} | {e('ciclos_em_4s')} | {x['frequencia_hz']['n_janelas']} / "
                  f"{x['episodios_com_janela']} de {x['episodios']} / {x['videos']} |")
  sp = b["classes"].get("Spinning", {})
  if sp.get("rotacao_hz"):
    linhas.append(f"\n*Spinning*: a largura dos ombros oscila duas vezes por volta; rotação {f3(sp['rotacao_hz']['mediana'])} Hz "
                  f"[{f3(sp['rotacao_hz']['p10'])}; {f3(sp['rotacao_hz']['p90'])}], {f3(sp['ciclos_de_rotacao_em_2s']['mediana'])} voltas em 2 s e "
                  f"{f3(sp['ciclos_de_rotacao_em_4s']['mediana'])} em 4 s (medianas).")
  if hist:
    linhas += ["", "## C. Variantes contra o `r2-C4-aumento` (validação nos mesmos quadros válidos por Q)", "",
               "| variante | quadros | F1 evento | acurácia bal. | macro F1 quadros | dependência (AF / HB / SP) | situação | tempo |", "|---|---|---|---|---|---|---|---|"]
    for h in hist:
      if "met" not in h:
        linhas.append(f"| {h['id']} | -- | -- | -- | -- | -- | {h['situacao']} | -- |")
        continue
      cel = lambda x: f"{f3(h['met']['agregado_validacao'][x])} vs {f3(h['met_ref']['agregado_validacao'][x])}"
      d, dr = h["dep"], h["dep_ref"]
      linhas.append(f"| {h['descricao']} | {h['n']} | {' | '.join(cel(x) for x in METRICAS)} | {f3(d['dependencia'])} vs {f3(dr['dependencia'])} "
                    f"({' / '.join(f3(d['dependencia_por_classe'][c]) for c in r4.CLASSES3)}) | {h['situacao']} | {h['tempo']} |")
    linhas.append("Cada célula: variante vs `r2-C4-aumento` nos mesmos quadros (agregado).")
  if final_txt:
    linhas += ["", *final_txt]
  elif hist and len(hist) == 3:
    linhas += ["", "Nenhuma variante passou no critério nem se qualificou como alternativa; o modelo final continua `r2-C4-aumento`."]
  (Path(t3.O3["saida"]) / "resumo-rodada4b.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")


if __name__ == "__main__":
  main()
