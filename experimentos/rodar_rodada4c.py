# ============================================================
# ARQUIVO: experimentos/rodar_rodada4c.py
#
# O QUE FAZ: rodada 4c (experimentos/protocolo_rodada4c.md): P2c, o modelo
# final atual (r4b_C1_janela_4s) com atributos de periodicidade corrigidos
# (tendência linear removida, amplitude de 1,5 a 5 Hz, frequência e
# potência de 1,5 a 7 Hz, e a largura dos ombros de 0,5 a 2 Hz para
# spinning) concatenados à saída do LSTM. Compara com a referência nos
# quadros válidos por Q e, se passar (ou for alternativa), faz a avaliação
# final completa. Reaproveita as rodadas anteriores (importadas, não
# alteradas). Retomável; um commit por etapa; índice limpo em caso de falha.
#
# Uso (da raiz do repositório):
#   .venv/Scripts/python.exe experimentos/rodar_rodada4c.py
#   .venv/Scripts/python.exe experimentos/rodar_rodada4c.py --teste --saida <pasta>
# ============================================================

import argparse
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.signal import butter, detrend, filtfilt

RAIZ = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(RAIZ / "experimentos"), str(RAIZ / "src")]
import rodar_rodada4b as r4b  # noqa: E402

r4, rq, t3, r, q = r4b.r4, r4b.rq, r4b.t3, r4b.r, r4b.q
EXP1, METRICAS = r4.EXP1, r4.METRICAS
FS = 15
REF = "r4b_C1_janela_4s"
REF_SEMENTES = {42: REF, 1000: f"{REF}_semente_1000", 2000: f"{REF}_semente_2000"}
r4.REF, r4.REF_SEMENTES = REF, REF_SEMENTES  # a avaliação final da rodada 4 compara com a referência destes nomes
RENOMEIA = {"r4_avaliacao_final": "r4c_avaliacao_final"}
FILTRO = butter(4, [1.5, 5], btype="bandpass", fs=FS)
FILTRO_GIRO = butter(4, [0.5, 2], btype="bandpass", fs=FS)


def pasta(i):
  i = RENOMEIA.get(i, i)
  return Path(t3.O3["saida"]) / i if i.startswith("r4c_") else EXP1 / i


def log(msg):
  linha = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
  print(linha, flush=True)
  with open(Path(t3.O3["saida"]) / "progresso_rodada4c.log", "a", encoding="utf-8") as f:
    f.write(linha + "\n")


r4.pasta = t3.pasta = q.pasta = pasta
r4.log = t3.log = q.log = log
commit = t3.commit


# ---------- atributos de periodicidade corrigidos ----------

def tres_atributos(S, filtro, f_min, f_max):
  """S: (n, W, componentes). Amplitude filtrada, frequência dominante e log(1 + potência)."""
  b, a = filtro
  amp = np.sqrt(filtfilt(b, a, S, axis=1).var(1).sum(-1))
  Z = detrend(S, axis=1, type="linear") * np.hanning(S.shape[1])[None, :, None]
  pot = (np.abs(np.fft.rfft(Z, n=512, axis=1)) ** 2).sum(-1)
  f = np.fft.rfftfreq(512, 1 / FS)
  banda = (f >= f_min) & (f <= f_max)
  j = pot[:, banda].argmax(1)
  return [amp, f[banda][j], np.log1p(pot[:, banda][np.arange(len(j)), j])]


def atributos_p2c(X):
  P = X.reshape(*X.shape[:2], 17, 4)[..., :2].astype(np.float64)
  ombros = (P[:, :, 5] + P[:, :, 6]) / 2
  largura = np.linalg.norm(P[:, :, 6] - P[:, :, 5], axis=-1)[..., None]
  sinais = [P[:, :, 9] - P[:, :, 5], P[:, :, 10] - P[:, :, 6], P[:, :, 7] - P[:, :, 5], P[:, :, 8] - P[:, :, 6], P[:, :, 0] - ombros, largura]
  partes = [x for s in sinais for x in tres_atributos(s, FILTRO, 1.5, 7)] + tres_atributos(largura, FILTRO_GIRO, 0.5, 2)
  return np.stack(partes, 1).astype(np.float32)  # (n, 21)


_rede_anterior = q.rede


def rede(cfg, n_classes=4):
  if cfg.get("arquitetura") != "p2c":
    return _rede_anterior(cfg, n_classes)
  import torch
  import torch.nn as nn

  class RedeP2c(nn.Module):
    def __init__(self):
      super().__init__()
      self.lstm = nn.LSTM(68, cfg["oculto"], cfg["camadas"], batch_first=True, dropout=cfg["dropout"] if cfg["camadas"] > 1 else 0.0)
      self.drop = nn.Dropout(cfg["dropout"])
      self.norm_per = nn.BatchNorm1d(21)
      self.fc = nn.Linear(cfg["oculto"] + 21, n_classes)

    def forward(self, X):
      _, (h, _) = self.lstm(X)
      f = torch.from_numpy(atributos_p2c(X.detach().cpu().numpy()))
      n = self.norm_per
      if self.training and len(X) == 1:  # lote de uma janela: usa a variância acumulada
        f = torch.nn.functional.batch_norm(f, n.running_mean, n.running_var, n.weight, n.bias, False, 0.0, n.eps)
      else:
        f = n(f)
      return self.fc(torch.cat([self.drop(h[-1]), f], 1))
  return RedeP2c()


q.rede = rede


def autoverificacao():
  t = np.arange(60) / FS
  X = np.zeros((2, 60, 17, 4), np.float32)
  X[1, :, 9, 1] = 0.3 * np.sin(2 * np.pi * 3.0 * t) + 0.5 * t  # punho esquerdo: 3 Hz com deriva linear
  A = atributos_p2c(X.reshape(2, 60, 68))
  assert A.shape == (2, 21)
  assert abs(A[1, 1] - 3.0) < 0.2, "frequência dominante do punho deveria ser 3 Hz"
  assert A[1, 0] > 10 * (A[0, 0] + 1e-6), "amplitude filtrada do punho deveria crescer com a oscilação"


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
    usa = lambda vd: {k: v for k, v in vd.items() if k.split("|")[-1] in r.videos()}
    ref_teste = {s: usa(rq.dados(EXP1 / d, "teste", cfg_ref if s == 42 else {**cfg_ref, "semente": s})) for s, d in REF_SEMENTES.items()}
    ref_val = usa(rq.dados(EXP1 / REF, "validacao", cfg_ref))
    ref_cong = r4.vd_congelado(EXP1 / REF, cfg_ref, "validacao")

    exp_id, desc = "r4c_P2c_periodicidade_corrigida", "P2c: C1 + periodicidade corrigida (1,5-5 Hz; 1,5-7 Hz; giro 0,5-2 Hz) após o LSTM"
    cfg = {**cfg_ref, "arquitetura": "p2c"}
    if args.teste:
      cfg["max_epocas"] = 2
    met = t3.rodar3(exp_id, cfg, desc)
    h = {"id": exp_id, "descricao": desc, "situacao": "erro"}
    alvo = None
    if met is not None:
      vd = rq.dados(pasta(exp_id), "validacao", cfg)
      comum = rq.conjunto_comum(ref_val, vd)
      m, m_ref = q.met_val(rq.restringir(vd, comum)), q.met_val(rq.restringir(ref_val, comum))
      dep, dep_ref = r4.dependencia(vd, r4.vd_congelado(pasta(exp_id), cfg, "validacao"), comum), r4.dependencia(ref_val, ref_cong, comum)
      ok, _ = r.supera(m["agregado_validacao"], m_ref["agregado_validacao"])
      nao_piora = all(m["agregado_validacao"][x] >= m_ref["agregado_validacao"][x] - 0.01 for x in METRICAS)
      alt = (not ok) and nao_piora and dep["dependencia"] >= 2 * dep_ref["dependencia"]
      h = {"id": exp_id, "descricao": desc, "met": m, "met_ref": m_ref, "dep": dep, "dep_ref": dep_ref, "n": rq.n_quadros(rq.restringir(vd, comum)),
           "situacao": "passa" if ok else ("alternativa" if alt else "não"), "tempo": f"{(met.get('tempo_s') or 0) / 60:.0f} min"}
      r.gravar_json(pasta(exp_id) / "rodada4c.json", h)
      log(f"{exp_id}: {h['n']} quadros, passa={ok}, alternativa={alt}, dependência {dep['dependencia']:.3f} (ref {dep_ref['dependencia']:.3f}) | "
          + " ".join(f"{x}: {m['agregado_validacao'][x]:.3f} vs {m_ref['agregado_validacao'][x]:.3f}" for x in METRICAS))
      if ok or alt or args.teste:
        alvo = (exp_id, cfg, "variante que passou" if ok else ("alternativa" if alt else "forçado no teste"))
    escrever_resumo(h, None)
    commit(f"experimentos: {exp_id}", q.arquivos(pasta(exp_id)) + [pasta(exp_id) / "rodada4c.json", saida / "resumo_rodada4c.md", Path(__file__)])

    final_txt = None
    if alvo:
      fin = r4.avaliacao_final(alvo[0], alvo[1], alvo[2], cfg_ref, ref_teste, args)
      final_txt = [l.replace("r4_avaliacao_final", "r4c_avaliacao_final") for l in r4.texto_final(fin)]
      commit("experimentos: r4c_avaliacao_final", [pasta("r4c_avaliacao_final") / "avaliacao_final.json", Path(__file__)])
    escrever_resumo(h, final_txt)
    commit("experimentos: resumo da rodada 4c", [saida / "resumo_rodada4c.md", Path(__file__)])
    log("fim")
  except Exception:
    log("ERRO GERAL:\n" + traceback.format_exc())
    raise


def escrever_resumo(h, final_txt):
  f3 = q.f3
  linhas = ["# Resumo da rodada 4c", "", f"Protocolo: `protocolo_rodada4c.md`. Referência: `{REF}` (modelo final atual). Validação nos quadros válidos por Q.", ""]
  if "met" in h:
    cel = lambda x: (f"{f3(h['met']['agregado_validacao'][x])} vs {f3(h['met_ref']['agregado_validacao'][x])} "
                     f"(média {f3(h['met']['media_entre_particoes'][x])} vs {f3(h['met_ref']['media_entre_particoes'][x])})")
    d, dr = h["dep"], h["dep_ref"]
    linhas += [f"| variante | quadros | F1 evento | acurácia bal. | macro F1 quadros | dependência (AF / HB / SP) | situação | tempo |",
               "|---|---|---|---|---|---|---|---|",
               f"| {h['descricao']} | {h['n']} | {' | '.join(cel(x) for x in METRICAS)} | {f3(d['dependencia'])} vs {f3(dr['dependencia'])} "
               f"({' / '.join(f3(d['dependencia_por_classe'][c]) for c in r4.CLASSES3)} vs {' / '.join(f3(dr['dependencia_por_classe'][c]) for c in r4.CLASSES3)}) "
               f"| {h['situacao']} | {h['tempo']} |", "", "Cada célula: variante vs referência nos mesmos quadros (agregado; média entre partições)."]
  else:
    linhas.append(f"{h['id']}: {h['situacao']}.")
  if final_txt:
    linhas += ["", *final_txt]
  elif h.get("situacao") == "não":
    linhas += ["", f"A P2c não passou no critério nem se qualificou como alternativa; o modelo final continua `{REF}`."]
  (Path(t3.O3["saida"]) / "resumo_rodada4c.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")


if __name__ == "__main__":
  main()
