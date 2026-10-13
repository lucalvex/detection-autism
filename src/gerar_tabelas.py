# ============================================================
# ARQUIVO: gerar_tabelas.py
#
# O QUE FAZ: gera resultados_tabelas.tex (raiz do repositório) com as
# tabelas de resultado do TCC no formato do abntex2 (\caption, \label,
# \legend), lendo tudo dos arquivos de resultado -- nenhum número é
# digitado à mão. Vírgula decimal. Cada legenda diz qual modelo base é
# usado e se os valores são agregados sobre os vídeos, por vídeo ou
# média e desvio-padrão entre partições.
#
# Requer \usepackage{booktabs} no preâmbulo do documento.
#
# Fontes (todas versionadas em data/results/):
#   cv_metrics.json, classification_report_per_fold.txt  validação cruzada do modelo base (treino original)
#   metricas.json                                        avaliação temporal do modelo base (treino original)
#   comparacao_final.json                                comparação registrada
#   comparacao_{base_reseeded,rodada_a}/metricas.json    avaliação temporal dos dois modelos da comparação
#   distribuicoes_por_janela_e_video.json                maior pontuação por janela e F1 por vídeo
#   videos_desempenho_mediano.json                       regra dos vídeos de desempenho mediano
#   analises_posteriores.json                            análises posteriores
#   resumo_dados.json                                    número de vídeos
#   ../datasets_annotated/labels.json                    nomes das classes
#
# Uso: python src/gerar_tabelas.py [--saida resultados_tabelas.tex]
# ============================================================

import argparse
import json
import re
import statistics
from pathlib import Path

import numpy as np

R = Path("data/results")
LABELS = Path("data/datasets_annotated/labels.json")
CLASSES_REAIS = ["ArmFlapping", "HeadBanging", "Spinning"]
FONTE = "Fonte: Elaborado pelo autor."

BASE_ORIGINAL = "modelo base (treino original)"
BASE_SEMENTES = "modelo base treinado novamente com sementes fixas"
BASE = "Base"
INTERVALOS = "Intervalos"
SIGLAS_MODELOS = (f"{BASE}: {BASE_SEMENTES}, com rótulo por vídeo; "
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


def _nomes_das_classes():
  """id em minúsculas -> nome legível em minúsculas, a partir dos nomes
  CamelCase do labels.json (ArmFlapping -> arm flapping)."""
  return {nome.lower(): re.sub(r"(?<!^)(?=[A-Z])", " ", nome).lower() for nome in ler(LABELS)}


NOMES = _nomes_das_classes()


def nome_classe(c):
  if c.lower() == "background":
    return "fundo"
  return f"\\textit{{{NOMES[c.lower()]}}}"


def colunas(spec):
  return len(re.findall(r"[lcr]|p\{[^}]*\}", spec))


def celulas_ocupadas(linha):
  total = 0
  for cel in linha:
    m = re.match(r"\\multicolumn\{(\d+)\}", cel)
    total += int(m.group(1)) if m else 1
  return total


def tabela(rotulo, titulo, spec, cabecalho, corpo, nota, tamanho=None):
  """cabecalho e corpo: listas de linhas; cada linha é uma lista de
  células, ou uma string (\\midrule, \\cmidrule...) inserida como está."""

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


def media(vals):
  return sum(vals) / len(vals)


def f1_acaso(suportes):
  """F1 por classe de um classificador que sorteia uniformemente entre
  as k classes: revocação = 1/k, precisão = proporção real p da classe,
  F1 = 2p(1/k)/(p+1/k). Mesma fórmula de
  evaluate_temporal.chance_level_macro_f1."""

  total = sum(suportes.values())
  k = len(suportes)
  out = {}
  for c, s in suportes.items():
    p = s / total if total else 0.0
    out[c] = (2 * p * (1 / k) / (p + 1 / k)) if (p + 1 / k) > 0 else 0.0
  return out


# ---------------------------------------------------------------
# validação cruzada por janela, modelo base (treino original)
# ---------------------------------------------------------------

def ler_relatorio_por_particao(caminho, classes):
  """Lê o classification_report de cada partição gravado pelo
  train_classifier.py na mesma rodada do cv_metrics.json."""
  particoes = []
  for linha in open(caminho, encoding="utf-8"):
    if linha.startswith("====="):
      particoes.append({})
      continue
    partes = linha.split()
    if particoes and len(partes) == 5 and partes[0] in classes:
      particoes[-1][partes[0]] = {"precision": float(partes[1]), "recall": float(partes[2]), "f1": float(partes[3])}
  return particoes


def t_cv_base(cv, por_particao):
  assert len(por_particao) == cv["n_splits"]
  casas_gravadas = 4  # o relatório por partição é impresso com 4 casas; tolerância de arredondamento
  tol = 0.5 * 10 ** -casas_gravadas
  for c in CLASSES_REAIS:
    for chave, campo in (("precision", "precision"), ("recall", "recall"), ("f1", "f1")):
      vals = [p[c][campo] for p in por_particao]
      assert abs(np.mean(vals) - cv["per_class"][c][f"{chave}_mean"]) <= tol, (c, chave)
      assert abs(np.std(vals) - cv["per_class"][c][f"{chave}_std"]) <= 2 * tol, (c, chave)

  corpo = []
  for c in CLASSES_REAIS:
    m = cv["per_class"][c]
    corpo.append([nome_classe(c), pm(m["precision_mean"], m["precision_std"]), pm(m["recall_mean"], m["recall_std"]),
                  pm(m["f1_mean"], m["f1_std"]), inteiro(m["support"])])
  corpo.append("\\midrule")
  macro = {campo: [media([p[c][campo] for c in CLASSES_REAIS]) for p in por_particao] for campo in ("precision", "recall", "f1")}
  suporte_reais = sum(cv["per_class"][c]["support"] for c in CLASSES_REAIS)
  corpo.append([f"Média macro ({len(CLASSES_REAIS)} estereotipias)",
                pm(np.mean(macro["precision"]), np.std(macro["precision"])), pm(np.mean(macro["recall"]), np.std(macro["recall"])),
                pm(np.mean(macro["f1"]), np.std(macro["f1"])), inteiro(suporte_reais)])
  w = cv["weighted_avg"]
  assert w["support"] == suporte_reais
  corpo.append(["Média ponderada", pm(w["precision_mean"], w["precision_std"]), pm(w["recall_mean"], w["recall_std"]),
                pm(w["f1_mean"], w["f1_std"]), inteiro(w["support"])])
  corpo.append("\\midrule")
  corpo.append(["Acurácia (\\%)", mc(3, pm(cv["mean_accuracy"], cv["std_accuracy"], 2)), ""])

  sem_dado = [c for c in cv["per_class"] if cv["per_class"][c]["support"] == 0]
  nota = (f"Modelo base (treino original). Média $\\pm$ desvio-padrão entre as {cv['n_splits']} partições da validação "
          "cruzada por vídeo; suporte = total de janelas de teste somado nas partições. A média macro é calculada "
          "partição a partição sobre as estereotipias, a partir dos valores por partição gravados em "
          f"classification\\_report\\_per\\_fold.txt; as {len(sem_dado)} classes sem nenhuma janela no conjunto ficam "
          "de fora, e a média ponderada não muda, porque o peso delas é zero.")
  return tabela(
    "tab:cv-base",
    "Métricas por janela na validação cruzada do modelo base (treino original)",
    "lrrrr",
    [["Classe", "Precisão", "Revocação", "F1", "Suporte"]],
    corpo, nota, "\\small",
  )


# ---------------------------------------------------------------
# por quadro, modelo base (treino original)
# ---------------------------------------------------------------

def bloco_quadro(rep, classes):
  acaso = f1_acaso({c: rep[c]["support"] for c in classes})
  linhas = [[nome_classe(c), num(rep[c]["precision"]), num(rep[c]["recall"]), num(rep[c]["f1-score"]),
             num(acaso[c]), inteiro(rep[c]["support"])] for c in classes]
  return linhas, acaso


def t_quadro_base(m, n_videos):
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

  nota = (f"Modelo base (treino original), com predições fora da partição; valores agregados sobre os {n_videos} vídeos. "
          "F1 ao acaso: classificador que sorteia uniformemente entre as $k$ classes do bloco, "
          "$F_1 = 2p(1/k)/(p+1/k)$, com $p$ a proporção real da classe. No modelo base, fundo é a predição de "
          "qualquer uma das classes sem nenhum exemplo de treino. Nos quadros anotados, uma predição de fundo conta "
          "como erro da classe anotada.")
  return tabela(
    "tab:quadro-base",
    "Métricas por quadro do modelo base (treino original)",
    "lrrrrr",
    [["Classe", "Precisão", "Revocação", "F1", "F1 ao acaso", "Quadros"]],
    corpo, nota, "\\small",
  )


# ---------------------------------------------------------------
# por evento, modelo base (treino original), regra principal
# ---------------------------------------------------------------

def descricao_regra(rp):
  dur = "uma janela, o piso natural da regra" if rp["duracao_minima_config_s"] == 0 else f"{num(rp['duracao_minima_config_s'], 1)} s"
  return f"limiar {num(rp['limiar'], 2)}, duração mínima de {dur}"


def t_evento_base(m, n_videos):
  rp = m["por_evento"]["regra_principal"]
  ious = sorted(rp["por_iou"], key=float)
  resumo, detalhe = [], []
  for i, iou in enumerate(ious):
    p = rp["por_iou"][iou]["pooled_56_videos"]
    resumo.append([num(float(iou), 1), num(p["macro_f1"]), inteiro(p["n_pares_casados"]),
                   num(p["erro_medio_inicio_s"], 2), num(p["erro_medio_fim_s"], 2)])
    if i:
      detalhe.append("\\midrule")
    for c in CLASSES_REAIS:
      k = p["por_classe"][c]
      detalhe.append([num(float(iou), 1), nome_classe(c), inteiro(k["tp"]), inteiro(k["fp"]), inteiro(k["fn"]),
                      num(k["precisao"]), num(k["revocacao"]), num(k["f1"])])

  comum = (f"Modelo base (treino original), com predições fora da partição; valores agregados sobre os {n_videos} vídeos; "
           f"regra principal: {descricao_regra(rp)}.")
  t_resumo = tabela(
    "tab:evento-base",
    "Métricas por evento do modelo base (treino original) na regra principal",
    "rrrrr",
    [["IoU", "F1 macro", "Pares casados", "Erro de início (s)", "Erro de fim (s)"]],
    resumo, comum + " Erros de início e de fim: média em segundos, só sobre os pares casados.", "\\small",
  )
  t_detalhe = tabela(
    "tab:evento-base-classe",
    "Métricas por evento do modelo base (treino original) na regra principal, por classe",
    "rlrrrrrr",
    [["IoU", "Classe", "VP", "FP", "FN", "Precisão", "Revocação", "F1"]],
    detalhe, comum + " VP: trechos casados com um intervalo anotado da mesma classe; FP: trechos sem par; "
    "FN: intervalos anotados sem par.", "\\small",
  )
  return t_resumo + "\n\n" + t_detalhe


# ---------------------------------------------------------------
# comparação registrada
# ---------------------------------------------------------------

def t_comparacao(cf, n_videos):
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
    f"Comparação registrada entre o {BASE_SEMENTES} e o modelo com rótulos por intervalo",
    "lrrrr",
    cab, corpo, nota, "\\small",
  )


# ---------------------------------------------------------------
# precisão, revocação e F1 por classe, dois modelos, todos os quadros
# ---------------------------------------------------------------

def t_dois_modelos(base, rodada_a, n_videos):
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
  nota = (f"Predições fora da partição; valores agregados sobre os {n_videos} vídeos, todos os quadros com predição. "
          f"{SIGLAS_MODELOS} No {BASE}, fundo é a predição de qualquer uma das classes sem nenhum exemplo de treino, "
          f"rara: daí a precisão alta com revocação quase nula; no {INTERVALOS}, é uma saída treinada.")
  return tabela(
    "tab:classes-dois-modelos",
    "Precisão, revocação e F1 por classe nos dois modelos, em todos os quadros",
    "lrrrrrrr",
    cab, corpo, nota, "\\small",
  )


# ---------------------------------------------------------------
# matriz de confusão por quadro, só quadros anotados
# ---------------------------------------------------------------

def t_confusao(base_original, rodada_a, n_videos):
  blocos = (
    (maiuscula_inicial(BASE_ORIGINAL), base_original["por_quadro"]["somente_quadros_anotados"]["pooled_56_videos"]),
    ("Modelo com rótulos por intervalo", rodada_a["por_quadro"]["somente_quadros_anotados"]["pooled_56_videos"]),
  )
  labels = blocos[0][1]["labels"]
  assert all(b["labels"] == labels for _, b in blocos)
  previstas = labels  # as 3 estereotipias e o fundo
  idx_fundo = labels.index("Background")
  cab = [
    ["", mc(len(previstas), "Classe prevista"), ""],
    f"\\cmidrule(lr){{2-{len(previstas) + 1}}}",
    ["Classe anotada"] + [nome_classe(c) for c in previstas] + ["Total"],
  ]
  corpo = []
  for i, (titulo, b) in enumerate(blocos):
    if i:
      corpo.append("\\midrule")
    corpo.append([mc(len(previstas) + 2, titulo, "l")])
    cm = b["confusion_matrix"]
    for c in CLASSES_REAIS:
      linha = cm[labels.index(c)]
      assert sum(linha) == b["report"][c]["support"], (titulo, c)
      corpo.append([nome_classe(c)] + [inteiro(v) for v in linha] + [inteiro(sum(linha))])
  base_cm = blocos[0][1]["confusion_matrix"]
  base_sem_fundo = all(base_cm[labels.index(c)][idx_fundo] == 0 for c in CLASSES_REAIS)
  nota = (f"Predições fora da partição; contagens de quadros agregadas sobre os {n_videos} vídeos, só nos quadros "
          "anotados no XML (linhas: classe anotada; colunas: classe prevista). No modelo base, fundo é a predição "
          "de qualquer uma das classes sem nenhum exemplo de treino"
          + ("; ele não previu fundo em nenhum quadro anotado." if base_sem_fundo else ".")
          + " No modelo com rótulos por intervalo, fundo é uma saída treinada.")
  return tabela(
    "tab:confusao-anotados",
    "Matriz de confusão por quadro nos quadros anotados",
    "l" + "r" * (len(previstas) + 1),
    cab, corpo, nota, "\\small",
  )


# ---------------------------------------------------------------
# distribuição da maior pontuação por janela
# ---------------------------------------------------------------

def num_corte(texto):
  """Escreve um corte gravado como texto ("0.999") com as mesmas casas,
  e pelo menos duas."""
  casas = max(2, len(texto.split(".")[1]))
  return num(float(texto), casas)


def t_pontuacao_maxima(dist, n_videos):
  modelos = ((BASE, dist["modelos"]["base_sementes_fixas"]), (INTERVALOS, dist["modelos"]["rotulos_por_intervalo"]))
  percentis = ["10", "25"]
  cortes = sorted(modelos[0][1]["pontuacao_maxima_por_janela"]["fracao_janelas_igual_ou_acima"], key=float)
  assert all(sorted(m["pontuacao_maxima_por_janela"]["fracao_janelas_igual_ou_acima"], key=float) == cortes for _, m in modelos)
  cab = [
    ["", "", "", "", "", "", mc(len(cortes), "Fração de janelas")],
    f"\\cmidrule(lr){{7-{6 + len(cortes)}}}",
    ["Modelo", "Saídas", "Janelas", "Média"] + [f"P{p}" for p in percentis] + [f"$\\geq$ {num_corte(c)}" for c in cortes],
  ]
  corpo = []
  for nome, m in modelos:
    p = m["pontuacao_maxima_por_janela"]
    assert p["n_videos"] == n_videos
    corpo.append([nome, inteiro(m["n_saidas"]), inteiro(p["n_janelas"]), num(p["media"])]
                 + [num(p["percentis"][q]) for q in percentis]
                 + [num(p["fracao_janelas_igual_ou_acima"][c]) for c in cortes])
  nota = (f"Maior pontuação de cada janela: máximo da softmax sobre todas as saídas do modelo, com predições fora da "
          f"partição, em todas as janelas dos {n_videos} vídeos (valores agregados sobre as janelas). P$n$: percentil $n$. "
          f"Fração de janelas: fração das janelas com maior pontuação igual ou acima do valor da coluna. {SIGLAS_MODELOS}")
  return tabela(
    "tab:pontuacao-maxima",
    "Distribuição da maior pontuação por janela nos dois modelos",
    "l" + "r" * (len(cab[-1]) - 1),
    cab, corpo, nota, "\\small",
  )


# ---------------------------------------------------------------
# grade limiar x duração mínima, por evento, modelo base (treino original)
# ---------------------------------------------------------------

def t_grade_evento(m, n_videos):
  ev = m["por_evento"]
  g = ev["grade_completa"]
  iou = re.search(r"IoU=([0-9.]+)", g["objetivo"]).group(1)
  limiar = ev["regra_principal"]["limiar"]
  j = g["limiares"].index(limiar)
  tetos = {t["duracao_minima_s"]: t for t in g["teto_revocacao_por_duracao_minima"]}
  total = g["total_eventos_anotados"]

  corpo = []
  for i, d in enumerate(g["duracoes_minimas_s"]):
    t = tetos[d]
    assert abs(t["teto_revocacao_aproximado"] - (total - t["eventos_mais_curtos_que_duracao_minima"]) / total) < 1e-12
    corpo.append([num(d, 1), num(g["macro_f1"][i][j]), inteiro(g["pares_casados"][i][j]),
                  f"{inteiro(total - t['eventos_mais_curtos_que_duracao_minima'])} de {inteiro(total)}",
                  num(t["teto_revocacao_aproximado"])])

  # o melhor par gravado tem que ser o máximo da grade inteira
  melhor = ev["melhor_par_da_grade"]
  pares = [(g["macro_f1"][i][k], g["limiares"][k], g["duracoes_minimas_s"][i])
           for i in range(len(g["duracoes_minimas_s"])) for k in range(len(g["limiares"]))]
  f1_max, lim_max, dur_max = max(pares)
  assert (lim_max, dur_max) == (melhor["limiar"], melhor["duracao_minima_config_s"])
  f1_melhor = melhor["por_iou"][iou]["pooled_56_videos"]["macro_f1"]
  assert abs(f1_melhor - f1_max) < 1e-12
  corpo.append("\\midrule")
  corpo.append([mc(5, f"Melhor par da grade inteira$^{{a}}$: limiar {num(lim_max, 2)}, duração mínima "
                   f"{num(dur_max, 1)} s, F1 macro {num(f1_melhor)}", "l")])

  nota = (f"Modelo base (treino original), com predições fora da partição; valores agregados sobre os {n_videos} vídeos; "
          f"limiar {num(limiar, 2)} e IoU {num(float(iou), 1)}. Teto de revocação: fração dos {inteiro(total)} "
          "comportamentos anotados nesses vídeos com duração maior ou igual à duração mínima; os mais curtos não podem "
          "ser casados. $^{a}$Par limiar e duração mínima de maior F1 macro em toda a grade, para o mesmo IoU, "
          "selecionado a posteriori nos mesmos dados em que é avaliado: não é uma escolha prévia e tende a "
          "superestimar o desempenho em vídeos novos.")
  return tabela(
    "tab:grade-evento",
    "Métricas por evento do modelo base (treino original) por duração mínima do trecho",
    "rrrrr",
    [["Duração mínima (s)", "F1 macro", "Pares casados", "Comportamentos alcançáveis", "Teto de revocação"]],
    corpo, nota, "\\small",
  )


# ---------------------------------------------------------------
# distribuição do F1 por vídeo, dois modelos
# ---------------------------------------------------------------

def mediano_por_regra(linhas):
  ordem = sorted(linhas, key=lambda r: (r["f1_quadro_anotados"], r["video_id"]))
  return ordem[(len(ordem) - 1) // 2]


def t_f1_por_video(dist, vm):
  regra = vm["regra"]
  modelos = ((BASE, dist["modelos"]["base_sementes_fixas"]), (INTERVALOS, dist["modelos"]["rotulos_por_intervalo"]))
  corpo = []
  for i, c in enumerate(regra["um_video_por_classe"]):
    if i:
      corpo.append("\\midrule")
    for nome, m in modelos:
      linhas = [r for r in m["f1_por_video"] if r["classe_do_video"] == c and r["f1_quadro_anotados"] is not None]
      f1s = [r["f1_quadro_anotados"] for r in linhas]
      e = mediano_por_regra(linhas)
      if nome == INTERVALOS:
        assert e["video_id"] == vm["escolhidos"][c]["video_id"]
      corpo.append([nome_classe(c), nome, inteiro(len(f1s)), num(min(f1s)), num(statistics.median(f1s)), num(max(f1s)),
                    inteiro(sum(1 for v in f1s if v == 0)), f"{esc(e['video_id'])} ({num(e['f1_quadro_anotados'])})"])
  nota = ("Valores por vídeo, não agregados: F1 por quadro de cada vídeo nos quadros anotados, com predições fora da "
          "partição e predição natural (fundo conta como erro); macro F1 sobre as classes presentes no gabarito do "
          f"vídeo. {SIGLAS_MODELOS} ``F1 zero'': número de vídeos com F1 nulo. Mediana estatística; o vídeo mediano é o "
          f"da posição {esc(regra['posicao_mediana'].split(' ')[0])} na ordem crescente, com desempate pela "
          f"{esc(regra['desempate'].replace(' (aprovado)', ''))}; os vídeos de desempenho mediano usados na interface "
          "são os do modelo com rótulos por intervalo.")
  return tabela(
    "tab:f1-por-video",
    "Distribuição do F1 por quadro por vídeo, em cada classe, nos dois modelos",
    "llrrrrrl",
    [["Classe", "Modelo", "Vídeos", "Mínimo", "Mediana", "Máximo", "F1 zero", "Vídeo mediano (F1)"]],
    corpo, nota, "\\small",
  )


# ---------------------------------------------------------------
# análises posteriores
# ---------------------------------------------------------------

def t_posteriores(ap):
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
      corpo.append([f"Fração prevista como a classe do vídeo, {onde} dos episódios: {nome_classe(c)}",
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
  assert "re-seed" not in tex.lower()
  return rotulos


def nome_do_arquivo(tabela_tex):
  rotulo = re.search(r"\\label\{([^}]*)\}", tabela_tex).group(1)
  return re.sub(r"[^A-Za-z0-9]", "_", rotulo) + ".tex"


def main():
  ap_ = argparse.ArgumentParser()
  ap_.add_argument("--saida", default="resultados_tabelas.tex")
  ap_.add_argument("--pasta", default="tabelas", help="um arquivo por tabela, com o nome do rótulo")
  args = ap_.parse_args()

  n_videos = ler(R / "resumo_dados.json")["n_videos_usados"]
  vm = ler(R / "videos_desempenho_mediano.json")
  assert len(vm["por_video"]) == n_videos
  cv = ler(R / "cv_metrics.json")
  base_original = ler(R / "metricas.json")
  base_sementes = ler(R / "comparacao_base_reseeded" / "metricas.json")
  rodada_a = ler(R / "comparacao_rodada_a" / "metricas.json")
  dist = ler(R / "distribuicoes_por_janela_e_video.json")

  blocos = [
    t_cv_base(cv, ler_relatorio_por_particao(R / "classification_report_per_fold.txt", list(cv["per_class"]))),
    t_quadro_base(base_original, n_videos),
    t_evento_base(base_original, n_videos),
    t_grade_evento(base_original, n_videos),
    t_comparacao(ler(R / "comparacao_final.json"), n_videos),
    t_dois_modelos(base_sementes, rodada_a, n_videos),
    t_confusao(base_original, rodada_a, n_videos),
    t_pontuacao_maxima(dist, n_videos),
    t_f1_por_video(dist, vm),
    t_posteriores(ler(R / "analises_posteriores.json")),
  ]

  cabecalho = (
    "% Gerado por src/gerar_tabelas.py a partir de data/results/ -- não edite à mão.\n"
    "% Requer \\usepackage{booktabs}.\n"
    "% A matriz de confusão por janela da validação cruzada do modelo base (treino original) não foi gravada em\n"
    "% nenhum arquivo de resultado (train_classifier.py só gerou data/results/confusion_matrix.png).\n"
  )
  tex = cabecalho + "\n" + "\n\n".join(blocos) + "\n"
  rotulos = validar(tex)

  # um arquivo por tabela; as tabelas não têm linhas em branco dentro
  tabelas = [t for bloco in blocos for t in bloco.split("\n\n")]
  assert len(tabelas) == len(rotulos)
  pasta = Path(args.pasta)
  pasta.mkdir(exist_ok=True)
  nomes = []
  for t in tabelas:
    nome = nome_do_arquivo(t)
    validar(t)
    (pasta / nome).write_text(cabecalho + "\n" + t + "\n", encoding="utf-8")
    nomes.append(nome)
  # remove arquivos de tabelas que deixaram de existir (só os gerados por este script)
  for antigo in pasta.glob("tab_*.tex"):
    if antigo.name not in nomes:
      antigo.unlink()

  Path(args.saida).write_text(tex, encoding="utf-8")
  print(f"{len(rotulos)} tabelas gravadas em {args.saida} e em {pasta}/: {', '.join(nomes)}")


if __name__ == "__main__":
  main()
