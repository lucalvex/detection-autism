# ============================================================
# ARQUIVO: run_pipeline.py
#
# O QUE FAZ: roda o pipeline inteiro do zero, na ordem certa, cada
# etapa como um processo separado (mesmo comando que você rodaria
# manualmente):
#
#   1) extract_pose.py         -> data/poses/*.csv
#   2) create_sequences.py     -> data/datasets/{X,y,groups}.npy
#   3) train_classifier.py     -> models/classifier.pt +
#                                  data/results/{classification_report.txt,
#                                  cv_metrics.json, confusion_matrix.png}
#   4) episode_analysis.py     -> data/results/{episodes.csv,
#                                  episode_summary.json}  (opcional --
#                                  se falhar, não trava o restante)
#   5) generate_report.py      -> data/results/report.pdf (relatório
#                                  final, consolidando os resultados
#                                  das etapas acima em um PDF único)
#
# Use isso quando quiser regenerar TUDO (ex.: depois de adicionar
# vídeos novos em data/videos/). Para rodar só uma etapa específica,
# chame o script dela diretamente (veja o README).
# ============================================================

import subprocess
import sys

# (título exibido, caminho do script, obrigatório?)
# etapa não-obrigatória: se falhar, o pipeline avisa e continua para
# as próximas etapas em vez de abortar tudo
STAGES = [
  ("1/5 Extração de pose", "src/extract_pose.py", True),
  ("2/5 Criação das sequências", "src/create_sequences.py", True),
  ("3/5 Treinamento + validação cruzada", "src/train_classifier.py", True),
  ("4/5 Análise de episódios", "src/utils/episode_analysis.py", False),
  ("5/5 Geração do relatório final", "src/generate_report.py", True),
]


def run_stage(title, script, required):

  print(f"\n{'=' * 70}")
  print(title)
  print(script)
  print("=" * 70)

  result = subprocess.run([sys.executable, script])

  if result.returncode != 0:

    if required:
      print(f"\n[ERRO] '{script}' falhou (código {result.returncode}). Abortando pipeline.")
      sys.exit(result.returncode)

    print(
      f"\n[AVISO] '{script}' falhou (código {result.returncode}) -- "
      "etapa opcional, o relatório vai marcar essa seção como indisponível."
    )


def main():

  for title, script, required in STAGES:
    run_stage(title, script, required)

  print("\nPipeline completo. Relatório final em data/results/report.pdf")


if __name__ == "__main__":
  main()
