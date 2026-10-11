# ============================================================
# ARQUIVO: gerar_tabelas.py
#
# O QUE FAZ: gera resultados_tabelas.tex (raiz do repositório) com as
# tabelas de resultado do TCC no formato do abntex2 (\caption, \label,
# \legend), lendo tudo dos arquivos de resultado -- nenhum número é
# digitado à mão. Vírgula decimal. Cada legenda diz se os valores são
# agregados sobre os vídeos ou média e desvio-padrão entre partições.
#
# Requer \usepackage{booktabs} no preâmbulo do documento.
#
# Fontes:
#   data/results/cv_metrics.json                          (tabela 1)
#   data/results/metricas.json                            (tabelas 2 e 3, modelo base, models/oof)
#   data/results/comparacao_final.json                    (tabela 4)
#   data/results/comparacao_{base_reseeded,rodada_a}/metricas.json (tabela 5)
#   data/results/videos_desempenho_mediano.json           (tabela 6)
#   data/results/analises_posteriores.json                (tabela 7)
#   data/results/resumo_dados.json                        (número de vídeos e de partições)
#
# Uso: python src/gerar_tabelas.py [--saida resultados_tabelas.tex]
# ============================================================

import argparse
import json
import re
import statistics
from pathlib import Path

R = Path("data/results")
CLASSES_REAIS = ["ArmFlapping", "HeadBanging", "Spinning"]
NOME_FUNDO = "Fundo"
FONTE = "Fonte: o autor."
BASE = "Base"
INTERVALOS = "Intervalos"
SIGLAS_MODELOS = (f"{BASE}: base re-seedado, com rótulo por vídeo; "
                  f"{INTERVALOS}: modelo com rótulos por intervalo anotado e classe de fundo.")


def ler(caminho):
  return json.load(open(caminho, encoding="utf-8"))


def num(x, casas=3):
  return f"{x:.{casas}f}".replace(".", ",")


def pm(media, desvio, casas=3):
  return f"{num(media, casas)} $\\pm$ {num(desvio, casas)}"


def inteiro(x):
  return f"{int(round(x)):,}".replace(",", ".")


def seta(antes, depois, casas=3):
  return f"{num(antes, casas)} $\\rightarrow$ {num(depois, casas)}"


def esc(texto):
  return (str(texto).replace("\\", "\\textbackslash{}").replace("_", "\\_")
          .replace("&", "\\&").replace("%", "\\%").replace("#", "\\#"))


def nome_classe(c):
  return NOME_FUNDO if c == "Background" else c


def colunas(spec):
  # conta colunas l/c/r e p{...} de uma especificação de tabular
  return len(re.findall(r"[lcr]|p\{[^}]*\}", spec))


def celulas_ocupadas(linha):
  total = 0
  for cel in linha:
    m = re.match(r"\\multicolumn\{(\d+)\}", cel)
    total += int(m.group(1)) if m else 1
  return total


def tabela(rotulo, titulo, spec, cabecalho, corpo, nota, tamanho=None):
  """cabecalho e corpo: listas de linhas; cada linha é uma lista de
  células, ou a string "\\midrule"/"\\cmidrule..." inserida como está."""

  n = colunas(spec)
  partes = ["\\begin{table}[htbp]", "\\centering"]
  if tamanho:
    partes.append(tamanho)
  partes += [f"\\caption{{{titulo}}}", f"\\label{{{rotulo}}}", f"\\begin{{tabular}}{{{spec}}}", "\\toprule"]

  def emitir(linhas):
    for linha in linhas:
      if isinstance(linha, str):
        partes.append(linha)
        continue
      ocupadas = celulas_ocupadas(linha)
      assert ocupadas == n, f"{rotulo}: linha com {ocupadas} células para {n} colunas: {linha}"
      partes.append(" & ".join(linha) + " \\\\")

  emitir(cabecalho)
  partes.append("\\midrule")
  emitir(corpo)
  partes += ["\\bottomrule", "\\end{tabular}", f"\\legend{{{nota} {FONTE}}}", "\\end{table}"]
  return "\n".join(partes)


def mc(n, texto, alinhamento="c"):
  return f"\\multicolumn{{{n}}}{{{alinhamento}}}{{{texto}}}"


def maiuscula_inicial(texto):
  return texto[:1].upper() + texto[1:]


def sim_nao(valor):
  return "sim" if valor else "não"


def f1_acaso(suportes):
  """F1 por classe de um classificador que sorteia uniformemente entre
  as k classes: revocação = 1/k, precisão = proporção real p da
  classe, F1 = 2p(1/k)/(p+1/k). Mesma fórmula de
  evaluate_temporal.chance_level_macro_f1."""

  total = sum(suportes.values())
  k = len(suportes)
  out = {}
  for c, s in suportes.items():
    p = s / total if total else 0.0
    out[c] = (2 * p * (1 / k) / (p + 1 / k)) if (p + 1 / k) > 0 else 0.0
  return out


# ---------------------------------------------------------------
# 1. validação cruzada por janela, modelo base
# ---------------------------------------------------------------

def t1_cv_base(cv):
  classes = list(cv["per_class"].keys())
  corpo = []
  for c in classes:
    m = cv["per_class"][c]
    corpo.append([c, pm(m["precision_mean"], m["precision_std"]), pm(m["recall_mean"], m["recall_std"]),
                  pm(m["f1_mean"], m["f1_std"]), inteiro(m["support"])])
  corpo.append("\\midrule")
  for rotulo, chave in (("Média macro", "macro_avg"), ("Média ponderada", "weighted_avg")):
    m = cv[chave]
    corpo.append([rotulo, pm(m["precision_mean"], m["precision_std"]), pm(m["recall_mean"], m["recall_std"]),
                  pm(m["f1_mean"], m["f1_std"]), inteiro(m["support"])])
  corpo.append("\\midrule")
  corpo.append(["Acurácia (\\%)", mc(3, pm(cv["mean_accuracy"], cv["std_accuracy"], 2)), ""])
  sem_dado = [c for c in classes if cv["per_class"][c]["support"] == 0]
  nota = (f"Média $\\pm$ desvio-padrão entre as {cv['n_splits']} partições da validação cruzada por vídeo; "
          f"suporte = total de janelas de teste somado nas partições. "
          f"{', '.join(sem_dado)} não têm nenhuma janela no conjunto e entram na média macro com valor nulo.")
  return tabela(
    "tab:cv-base",
    "Métricas por janela na validação cruzada do modelo base (rótulo por vídeo)",
    "lrrrr",
    [["Classe", "Precisão", "Revocação", "F1", "Suporte"]],
    corpo, nota, "\\small",
  )


# ---------------------------------------------------------------
# 2. por quadro, modelo base
# ---------------------------------------------------------------

def bloco_quadro(rep, classes):
  suportes = {c: rep[c]["support"] for c in classes}
  acaso = f1_acaso(suportes)
  linhas = [[nome_classe(c), num(rep[c]["precision"]), num(rep[c]["recall"]), num(rep[c]["f1-score"]),
             num(acaso[c]), inteiro(rep[c]["support"])] for c in classes]
  return linhas, acaso


def media(vals):
  return sum(vals) / len(vals)


def t2_quadro_base(m, n_videos):
  todos = m["por_quadro"]["todos_os_quadros"]["pooled_56_videos"]
  anot_bloco = m["por_quadro"]["somente_quadros_anotados"]
  anot = anot_bloco["pooled_56_videos"]
  rep_t, rep_a = todos["report"], anot["report"]
  classes_t = todos["labels"]
  corpo = [[mc(6, f"Todos os quadros ({len(classes_t)} classes)", "l")]]
  linhas_t, acaso_t = bloco_quadro(rep_t, classes_t)
  corpo += linhas_t
  for rotulo, cls, guardado in (
    (f"Média macro ({len(CLASSES_REAIS)} estereotipias)", CLASSES_REAIS, todos["macro_f1_sem_background"]),
    (f"Média macro ({len(classes_t)} classes, com fundo)", classes_t, todos["macro_f1_com_background"]),
  ):
    f1 = media([rep_t[c]["f1-score"] for c in cls])
    assert abs(f1 - guardado) < 1e-12, rotulo
    corpo.append([rotulo, num(media([rep_t[c]["precision"] for c in cls])), num(media([rep_t[c]["recall"] for c in cls])),
                  num(f1), num(media([acaso_t[c] for c in cls])), inteiro(sum(rep_t[c]["support"] for c in cls))])

  corpo.append("\\midrule")
  corpo.append([mc(6, f"Só quadros anotados ({len(CLASSES_REAIS)} classes)", "l")])
  linhas_a, acaso_a = bloco_quadro(rep_a, CLASSES_REAIS)
  corpo += linhas_a
  f1 = media([rep_a[c]["f1-score"] for c in CLASSES_REAIS])
  assert abs(f1 - anot["macro_f1_sem_background"]) < 1e-12
  acaso_macro = media([acaso_a[c] for c in CLASSES_REAIS])
  assert abs(acaso_macro - anot_bloco["f1_chance_3_classes"]) < 1e-12
  corpo.append(["Média macro", num(media([rep_a[c]["precision"] for c in CLASSES_REAIS])),
                num(media([rep_a[c]["recall"] for c in CLASSES_REAIS])), num(f1), num(acaso_macro),
                inteiro(sum(rep_a[c]["support"] for c in CLASSES_REAIS))])

  nota = (f"Valores agregados sobre os {n_videos} vídeos (predições fora da partição). "
          "F1 ao acaso: classificador que sorteia uniformemente entre as $k$ classes do bloco, "
          "$F_1 = 2p(1/k)/(p+1/k)$, com $p$ a proporção real da classe. "
          "Nos quadros anotados, uma predição de fundo conta como erro da classe anotada.")
  return tabela(
    "tab:quadro-base",
    "Métricas por quadro do modelo base (rótulo por vídeo)",
    "lrrrrr",
    [["Classe", "Precisão", "Revocação", "F1", "F1 ao acaso", "Quadros"]],
    corpo, nota, "\\small",
  )


# ---------------------------------------------------------------
# 3. por evento, modelo base, regra principal
# ---------------------------------------------------------------

def descricao_regra(rp):
  dur = "uma janela, o piso natural da regra" if rp["duracao_minima_config_s"] == 0 else f"{num(rp['duracao_minima_config_s'], 1)} s"
  return f"limiar {num(rp['limiar'], 2)}, duração mínima de {dur}"


def t3_evento_base(m, n_videos):
  rp = m["por_evento"]["regra_principal"]
  ious = sorted(rp["por_iou"], key=float)
  resumo = []
  detalhe = []
  for i, iou in enumerate(ious):
    p = rp["por_iou"][iou]["pooled_56_videos"]
    resumo.append([num(float(iou), 1), num(p["macro_f1"]), inteiro(p["n_pares_casados"]),
                   num(p["erro_medio_inicio_s"], 2), num(p["erro_medio_fim_s"], 2)])
    if i:
      detalhe.append("\\midrule")
    for c in CLASSES_REAIS:
      k = p["por_classe"][c]
      detalhe.append([num(float(iou), 1), c, inteiro(k["tp"]), inteiro(k["fp"]), inteiro(k["fn"]),
                      num(k["precisao"]), num(k["revocacao"]), num(k["f1"])])

  nota = (f"Valores agregados sobre os {n_videos} vídeos (predições fora da partição); regra principal: "
          f"{descricao_regra(rp)}. Erros de início e de fim: média em segundos, só sobre os pares casados.")
  t_resumo = tabela(
    "tab:evento-base",
    "Métricas por evento do modelo base na regra principal",
    "rrrrr",
    [["IoU", "F1 macro", "Pares casados", "Erro de início (s)", "Erro de fim (s)"]],
    resumo, nota, "\\small",
  )
  t_detalhe = tabela(
    "tab:evento-base-classe",
    "Métricas por evento do modelo base na regra principal, por classe",
    "rlrrrrrr",
    [["IoU", "Classe", "VP", "FP", "FN", "Precisão", "Revocação", "F1"]],
    detalhe, nota, "\\small",
  )
  return t_resumo + "\n\n" + t_detalhe


# ---------------------------------------------------------------
# 4. comparação registrada
# ---------------------------------------------------------------

def t4_comparacao(cf, n_videos):
  m1, m2 = cf["metricas_principais"]
  cab = [
    ["", mc(2, "Métrica 1"), mc(2, "Métrica 2")],
    "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
    ["", BASE, INTERVALOS, BASE, INTERVALOS],
  ]
  corpo = [[f"Agregado ({n_videos} vídeos)", num(m1["pooled_56_videos"]["base_re_seedado"]), num(m1["pooled_56_videos"]["rodada_a"]),
            num(m2["pooled_56_videos"]["base_re_seedado"]), num(m2["pooled_56_videos"]["rodada_a"])], "\\midrule"]
  for a, b in zip(m1["por_fold"], m2["por_fold"]):
    assert a["fold"] == b["fold"]
    corpo.append([f"Partição {a['fold']}", num(a["base_re_seedado"]), num(a["rodada_a"]),
                  num(b["base_re_seedado"]), num(b["rodada_a"])])
  corpo.append([f"Partições com {INTERVALOS} maior",
                mc(2, f"{m1['n_folds_rodada_a_maior']} de {m1['n_folds']}"),
                mc(2, f"{m2['n_folds_rodada_a_maior']} de {m2['n_folds']}")])
  corpo.append("\\midrule")
  corpo.append([mc(4, "Melhora nas duas métricas agregadas", "l"), sim_nao(cf["condicao_1_melhora_as_duas_metricas_agregadas"])])
  corpo.append([mc(4, "Melhora nas duas métricas em partições suficientes, pelo critério", "l"),
                sim_nao(cf["condicao_2_melhora_em_pelo_menos_4_de_5_folds"])])
  corpo.append([mc(5, f"Veredito: {esc(cf['veredito'])}", "l")])
  nota = (f"Linha ``Agregado'': valores agregados sobre os {n_videos} vídeos; demais linhas: valor de cada partição. "
          f"Métrica 1: {esc(m1['metrica'])}. Métrica 2: {esc(m2['metrica'])}. {SIGLAS_MODELOS} "
          "Critério registrado antes da avaliação (criterio\\_comparacao.md): os rótulos por intervalo só seriam "
          "considerados melhores se as duas condições acima fossem atendidas.")
  return tabela(
    "tab:comparacao",
    "Comparação registrada entre o base re-seedado e o modelo com rótulos por intervalo",
    "lrrrr",
    cab, corpo, nota, "\\small",
  )


# ---------------------------------------------------------------
# 5. precisão, revocação e F1 por classe, dois modelos, todos os quadros
# ---------------------------------------------------------------

def t5_dois_modelos(base, rodada_a, n_videos):
  rb = base["por_quadro"]["todos_os_quadros"]["pooled_56_videos"]
  ra = rodada_a["por_quadro"]["todos_os_quadros"]["pooled_56_videos"]
  assert rb["labels"] == ra["labels"]
  cab = [
    ["", mc(3, BASE), mc(3, INTERVALOS), ""],
    "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
    ["Classe", "Precisão", "Revocação", "F1", "Precisão", "Revocação", "F1", "Quadros"],
  ]
  corpo = []
  for c in rb["labels"]:
    b, a = rb["report"][c], ra["report"][c]
    assert b["support"] == a["support"]
    corpo.append([nome_classe(c), num(b["precision"]), num(b["recall"]), num(b["f1-score"]),
                  num(a["precision"]), num(a["recall"]), num(a["f1-score"]), inteiro(b["support"])])
  nota = (f"Valores agregados sobre os {n_videos} vídeos (predições fora da partição), todos os quadros com predição. "
          f"{SIGLAS_MODELOS} No {BASE}, ``fundo'' é a predição de qualquer uma das classes sem nenhum exemplo de treino, "
          f"rara: daí a precisão alta com revocação quase nula; no {INTERVALOS}, é uma saída treinada.")
  return tabela(
    "tab:classes-dois-modelos",
    "Precisão, revocação e F1 por classe nos dois modelos, em todos os quadros",
    "lrrrrrrr",
    cab, corpo, nota, "\\small",
  )


# ---------------------------------------------------------------
# 6. distribuição do F1 por vídeo
# ---------------------------------------------------------------

def t6_f1_por_video(vm):
  regra = vm["regra"]
  corpo = []
  for c in regra["um_video_por_classe"]:
    f1s = [r["f1_quadro_anotados"] for r in vm["por_video"] if r["classe_do_video"] == c and r["f1_quadro_anotados"] is not None]
    e = vm["escolhidos"][c]
    corpo.append([c, inteiro(len(f1s)), num(min(f1s)), num(statistics.median(f1s)), num(max(f1s)),
                  inteiro(sum(1 for v in f1s if v == 0)), f"{esc(e['video_id'])} ({num(e['f1_quadro_anotados'])})"])
  nota = ("Valores por vídeo, não agregados: F1 por quadro de cada vídeo nos quadros anotados, com o modelo de rótulos "
          "por intervalo fora da partição e predição natural (fundo conta como erro); macro F1 sobre as classes presentes "
          f"no gabarito do vídeo. ``F1 zero'': número de vídeos com F1 nulo. Mediana estatística; o vídeo de desempenho "
          f"mediano é o da posição {esc(regra['posicao_mediana'].split(' ')[0])} "
          f"na ordem crescente, com desempate pela {esc(regra['desempate'].replace(' (aprovado)', ''))}.")
  return tabela(
    "tab:f1-por-video",
    "Distribuição do F1 por quadro por vídeo, em cada classe",
    "lrrrrrl",
    [["Classe", "Vídeos", "Mínimo", "Mediana", "Máximo", "F1 zero", "Vídeo mediano (F1)"]],
    corpo, nota, "\\small",
  )


# ---------------------------------------------------------------
# 7. análises posteriores
# ---------------------------------------------------------------

def t7_posteriores(ap):
  a1 = ap["analise_1_argmax_forcado_3_classes"]
  forc = a1["rodada_a_argmax_forcado_3_classes"]
  base1 = a1["base_re_seedado_metrica_1_original"]
  comp = a1["comparacao_por_fold"]
  bal = ap["analise_3_binario_estereotipia_vs_fundo"]["acuracia_balanceada_e_referencia"]
  ab = ap["analise_4_ablacoes_de_inferencia"]["ablacoes"]
  fr = ap["analise_2_fracao_classe_do_video"]["media_por_classe_de_video"]

  ENTRE = "Média $\\pm$ desvio entre partições"
  MEDIA = "Média entre partições"
  VIDEOS = "Média entre os vídeos da classe"

  corpo = [
    ["Discriminação entre as estereotipias, sem o fundo: macro F1 por quadro (anotados)$^{a}$",
     pm(forc["media_entre_folds"], forc["desvio_entre_folds"]), pm(base1["media_entre_folds"], base1["desvio_entre_folds"]), ENTRE],
    [f"Partições com {INTERVALOS} maior", mc(2, f"{comp['n_folds_rodada_a_maior']} de {comp['n_folds_total']}"), "Por partição"],
    "\\midrule",
    ["Acurácia balanceada binária (estereotipia contra fundo)",
     pm(bal["rodada_a"]["acuracia_balanceada_media_entre_folds"], bal["rodada_a"]["acuracia_balanceada_desvio_entre_folds"]),
     pm(bal["base_re_seedado"]["acuracia_balanceada_media_entre_folds"], bal["base_re_seedado"]["acuracia_balanceada_desvio_entre_folds"]), ENTRE],
    "\\midrule",
  ]
  for chave, rotulo in (("shuffle", "ordem dos quadros embaralhada"), ("zero_dxdy", "dx e dy zerados")):
    f = ab[chave]["macro_f1_quadro_anotados_media_entre_folds"]
    b = ab[chave]["acuracia_balanceada_binaria_rodada_a_media_entre_folds"]
    corpo.append([f"Ablação, {rotulo}: macro F1 por quadro (anotados)$^{{a}}$",
                  seta(f["rodada_a_forcado_3_classes"]["sem_ablacao"], f["rodada_a_forcado_3_classes"]["com_ablacao"]),
                  seta(f["base_re_seedado_natural"]["sem_ablacao"], f["base_re_seedado_natural"]["com_ablacao"]), MEDIA])
    corpo.append([f"Ablação, {rotulo}: acurácia balanceada binária",
                  seta(b["sem_ablacao"], b["com_ablacao"]), "---", MEDIA])
  corpo.append("\\midrule")
  for onde, chave in (("dentro", "media_fracao_anotados"), ("fora", "media_fracao_fundo")):
    for c in CLASSES_REAIS:
      corpo.append([f"Fração prevista como a classe do vídeo, {onde} dos episódios: {c}",
                    num(fr["rodada_a"][c][chave]), num(fr["base_re_seedado"][c][chave]), VIDEOS])

  nota = ("Análises definidas depois de conhecer o resultado da comparação registrada; não fazem parte do critério e "
          f"não mudam o veredito. A agregação de cada linha está na última coluna. {SIGLAS_MODELOS} "
          f"$^{{a}}$No {INTERVALOS}, argmax só entre as saídas das estereotipias, ignorando a de fundo; "
          f"no {BASE}, predição natural. Nas ablações, valores sem $\\rightarrow$ com a ablação.")
  return tabela(
    "tab:posteriores",
    "Análises posteriores, definidas depois do resultado da comparação",
    "p{5.4cm}rrp{2.8cm}",
    [["Análise", INTERVALOS, BASE, "Agregação"]],
    corpo, nota, "\\footnotesize",
  )


# ---------------------------------------------------------------

def validar(tex):
  assert tex.count("{") == tex.count("}"), "chaves desbalanceadas"
  for amb in ("table", "tabular"):
    assert tex.count(f"\\begin{{{amb}}}") == tex.count(f"\\end{{{amb}}}"), f"ambiente {amb} desbalanceado"
  assert tex.count("$") % 2 == 0, "modo matemático desbalanceado"
  rotulos = re.findall(r"\\label\{([^}]*)\}", tex)
  assert len(rotulos) == len(set(rotulos)), "rótulos repetidos"
  assert tex.count("\\legend{") == tex.count("\\begin{table}")
  return rotulos


def main():
  ap_ = argparse.ArgumentParser()
  ap_.add_argument("--saida", default="resultados_tabelas.tex")
  args = ap_.parse_args()

  resumo = ler(R / "resumo_dados.json")
  n_videos = resumo["n_videos_usados"]
  vm = ler(R / "videos_desempenho_mediano.json")
  assert len(vm["por_video"]) == n_videos

  blocos = [
    t1_cv_base(ler(R / "cv_metrics.json")),
    t2_quadro_base(ler(R / "metricas.json"), n_videos),
    t3_evento_base(ler(R / "metricas.json"), n_videos),
    t4_comparacao(ler(R / "comparacao_final.json"), n_videos),
    t5_dois_modelos(ler(R / "comparacao_base_reseeded" / "metricas.json"), ler(R / "comparacao_rodada_a" / "metricas.json"), n_videos),
    t6_f1_por_video(vm),
    t7_posteriores(ler(R / "analises_posteriores.json")),
  ]

  cabecalho = (
    "% Gerado por src/gerar_tabelas.py a partir de data/results/*.json -- não edite à mão.\n"
    "% Requer \\usepackage{booktabs}.\n"
    "% A matriz de confusão agregada da validação cruzada do modelo base não foi gravada em nenhum\n"
    "% arquivo de resultado (train_classifier.py só gerou data/results/confusion_matrix.png), por isso\n"
    "% não aparece na tabela de validação cruzada.\n"
  )
  tex = cabecalho + "\n" + "\n\n".join(blocos) + "\n"
  rotulos = validar(tex)

  Path(args.saida).write_text(tex, encoding="utf-8")
  print(f"{len(rotulos)} tabelas gravadas em {args.saida}: {', '.join(rotulos)}")


if __name__ == "__main__":
  main()
