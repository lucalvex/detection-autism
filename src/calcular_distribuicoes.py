# ============================================================
# ARQUIVO: calcular_distribuicoes.py
#
# O QUE FAZ: para os dois modelos da comparação -- o modelo base
# treinado novamente com sementes fixas (models/oof_seeded_base) e o
# modelo com rótulos por intervalo (models/oof_seeded_annotated) --
# calcula, com os checkpoints fora da partição já existentes (nenhum
# treino):
#   - a distribuição da maior pontuação por janela (máximo da softmax
#     sobre todas as saídas do modelo), em todas as janelas dos vídeos;
#   - o F1 por quadro de cada vídeo, pela mesma definição usada na
#     escolha dos vídeos de desempenho mediano
#     (selecionar_videos_mediana.f1_do_video).
# Grava data/results/distribuicoes_por_janela_e_video.json.
#
# Uso: python src/calcular_distribuicoes.py
# ============================================================

import json
from pathlib import Path

import numpy as np

import evaluate_temporal as et
from selecionar_videos_mediana import classe_do_nome, f1_do_video

SAIDA = Path("data/results/distribuicoes_por_janela_e_video.json")
PERCENTIS = [0, 10, 25, 50, 75, 90, 100]
# frações de janelas com maior pontuação >= cada valor (parâmetros da análise)
CORTES_FRACAO = ["0.90", "0.99", "0.999"]

MODELOS = {
  "base_sementes_fixas": {
    "descricao": "modelo base treinado novamente com sementes fixas (rótulo por vídeo)",
    "dataset": "data/datasets", "oof": "models/oof_seeded_base",
    "metricas": "data/results/comparacao_base_reseeded/metricas.json",
  },
  "rotulos_por_intervalo": {
    "descricao": "modelo com rótulos por intervalo anotado e classe de fundo",
    "dataset": "data/datasets_annotated", "oof": "models/oof_seeded_annotated",
    "metricas": "data/results/comparacao_rodada_a/metricas.json",
  },
}


def avaliar(chave, cfg):
  # tudo deste modelo é calculado antes de trocar de configuração:
  # et.predicted_frames lê o mapeamento de classes global do módulo
  et.load_config(cfg["dataset"], cfg["oof"])
  et._model_cache.clear()
  dados = et.load_all_videos()

  maximos = np.array([max(j["pontuacoes"]) for v in dados.values() for j in v["windows"]])
  limiar = json.load(open(cfg["metricas"], encoding="utf-8"))["por_evento"]["regra_principal"]["limiar"]

  por_video = []
  for vid in sorted(dados):
    f1, n, presentes = f1_do_video(dados[vid])
    por_video.append({
      "video_id": vid, "classe_do_video": classe_do_nome(vid), "particao": dados[vid]["fold"],
      "f1_quadro_anotados": f1, "n_quadros_anotados_com_predicao": n, "classes_no_gabarito": presentes,
    })

  return {
    "descricao": cfg["descricao"],
    "checkpoints": cfg["oof"],
    "n_saidas": et.num_classes,
    "pontuacao_maxima_por_janela": {
      "n_janelas": int(maximos.size),
      "n_videos": len(dados),
      "media": float(maximos.mean()),
      "percentis": {str(p): float(np.percentile(maximos, p)) for p in PERCENTIS},
      "limiar_padrao": limiar,
      "fracao_janelas_acima_do_limiar": float((maximos >= limiar).mean()),
      "fracao_janelas_igual_ou_acima": {c: float((maximos >= float(c)).mean()) for c in CORTES_FRACAO},
    },
    "f1_por_video": por_video,
  }


def main():
  saida = {
    "descricao": (
      "Distribuição da maior pontuação por janela (máximo da softmax sobre todas as saídas) e F1 por quadro "
      "de cada vídeo nos quadros anotados (predição natural; macro F1 sobre as classes presentes no gabarito do "
      "vídeo), com os checkpoints fora da partição de cada modelo. Nenhum treino."
    ),
    "modelos": {chave: avaliar(chave, cfg) for chave, cfg in MODELOS.items()},
  }

  # o F1 por vídeo do modelo com rótulos por intervalo tem que bater com o
  # já gravado na escolha dos vídeos de desempenho mediano
  gravado = {r["video_id"]: r["f1_quadro_anotados"] for r in
             json.load(open("data/results/videos_desempenho_mediano.json", encoding="utf-8"))["por_video"]}
  for r in saida["modelos"]["rotulos_por_intervalo"]["f1_por_video"]:
    assert r["f1_quadro_anotados"] == gravado[r["video_id"]], r["video_id"]

  with open(SAIDA, "w", encoding="utf-8") as f:
    json.dump(saida, f, indent=2, ensure_ascii=False)

  for chave, m in saida["modelos"].items():
    p = m["pontuacao_maxima_por_janela"]
    print(f"{chave}: {p['n_janelas']} janelas, maior pontuação mediana {p['percentis']['50']:.4f}, "
          f"P10 {p['percentis']['10']:.4f}, acima do limiar {p['fracao_janelas_acima_do_limiar']:.4f}")
  print(f"Gravado em {SAIDA}")


if __name__ == "__main__":
  main()
