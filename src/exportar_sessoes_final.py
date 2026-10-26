# ============================================================
# ARQUIVO: src/exportar_sessoes_final.py
#
# O QUE FAZ: exporta, para os 56 vídeos do SSBD, o que a interface de
# demonstração mostra no modo "Vídeos", com o modelo final
# r4b_C1_janela_4s (semente 42):
#   - saidas/final/<id>_sessao.json: as pontuações por janela fora da
#     partição (as de teste gravadas em janelas_<k>.pkl, da partição em
#     que o vídeo é teste), a anotação do SSBD e o pós-processamento
#     escolhido (mediana de 2 s e junção abaixo de 1 s), que a interface
#     aplica com demo/js/timeline.js;
#   - saidas/final/<id>_pontos.json: os pontos do corpo em coordenadas
#     da imagem, da mesma extração de pose usada pelo modelo
#     (experimentos/poses/yolo11n/<id>.npz);
#   - saidas/final/indice.json: a lista dos vídeos, agrupada por classe.
# Não treina nem roda modelo. Não copia vídeo: fps, tamanho e número de
# quadros vêm das predições e da extração de pose (quando faltam lá,
# são lidos do arquivo em data/videos, sem copiá-lo).
# saidas/ fica fora do git.
#
# Uso (da raiz do repositório):
#   .venv/Scripts/python.exe src/exportar_sessoes_final.py
# ============================================================

import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
from ssbd_annotations import find_annotation_file, load_annotations  # noqa: E402
from timeline import JUNTAR_S, MEDIANA_S  # noqa: E402

FINAL = RAIZ / "experimentos" / "r4b_C1_janela_4s"
POSES = RAIZ / "experimentos" / "poses" / "yolo11n"
XMLS = RAIZ.parent / "download-yt" / "xmls"
SAIDA = RAIZ / "saidas" / "final"

# id, nome no seletor (em inglês, como no TCC) e nome em português (apoio)
CLASSES = [("armflapping", "Arm flapping", "Balançar os braços"), ("headbanging", "Head banging", "Bater a cabeça"),
           ("spinning", "Spinning", "Girar o corpo"), ("background", "Sem estereotipia", "Sem estereotipia")]
PREFIXO = {"ArmFlapping": "armflapping", "HeadBanging": "headbanging", "Spinning": "spinning"}


def gravar(caminho, obj):
  with open(caminho, "w", encoding="utf-8") as f:
    json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))


def anotacoes(vid):
  xml = find_annotation_file(vid, XMLS)
  if xml is None:
    return []
  return [{"classe": b["category"], "inicio_s": b["start_s"], "fim_s": b["end_s"]}
          for b in load_annotations(xml)["behaviours"] if b["category"] in PREFIXO.values()]


def metadados_do_video(vid):
  """(número de quadros, fps) lidos do arquivo em data/videos, como load_video_fps()."""
  import cv2
  cap = cv2.VideoCapture(str(RAIZ / "data" / "videos" / f"{vid}.mp4"))
  n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
  cap.release()
  if n <= 0 or not fps:
    raise SystemExit(f"{vid}: não consegui ler o número de quadros e o fps de data/videos/{vid}.mp4.")
  return n, fps


def main():
  SAIDA.mkdir(parents=True, exist_ok=True)
  indice = {c[0]: [] for c in CLASSES[:3]}
  rotulos = {c[0]: c[1] for c in CLASSES}
  particoes = json.loads((RAIZ / "models" / "oof" / "fold_assignments.json").read_text(encoding="utf-8"))["video_to_fold"]
  for k in range(1, 6):
    with open(FINAL / f"janelas_{k}.pkl", "rb") as f:
      teste = pickle.load(f)["teste"]
    # v_Spinning_12 não tem nenhum trecho contínuo de 4 s com pessoa e fica sem janelas
    vazio = {"quadro_inicio": [], "quadro_fim": [], "pontuacoes": []}
    for vid in sorted(v for v, p in particoes.items() if p == k):
      x = teste.get(vid, vazio)
      m = re.fullmatch(r"v_(ArmFlapping|HeadBanging|Spinning)_(\d+)", vid)
      classe, numero = PREFIXO[m.group(1)], int(m.group(2))
      d = np.load(POSES / f"{vid}.npz")
      # as extrações conferidas contra o CSV antigo não gravaram quadros_lidos
      total, fps = metadados_do_video(vid) if "quadros_lidos" not in d.files or "fps" not in x else (None, None)
      total = int(d["quadros_lidos"]) if "quadros_lidos" in d.files else total
      fps = float(x["fps"]) if "fps" in x else fps
      detectados = set(d["frames"].tolist())
      gravar(SAIDA / f"{vid}_sessao.json", {
        "versao": 2,
        "sessao": {"id": vid, "classe": classe, "numero": numero, "fps": fps, "total_quadros": total,
                   "largura": int(d["w"]), "altura": int(d["h"])},
        "modelo": {"nome": "r4b_C1_janela_4s", "semente": 42, "particao": k, "janela_s": 4, "amostras_por_s": 15,
                   "pose": "yolo11n-pose", "predicao_fora_da_particao": True},
        "classes": [{"id": c, "nome": pt} for c, _, pt in CLASSES],
        "limiar_padrao": 0.5,
        "posprocessamento": {"mediana_s": MEDIANA_S, "juntar_s": JUNTAR_S},
        "janelas": [{"quadro_inicio": int(a), "quadro_fim": int(b), "pontuacoes": [round(float(v), 6) for v in p]}
                    for a, b, p in zip(x["quadro_inicio"], x["quadro_fim"], x["pontuacoes"])],
        "quadros_sem_deteccao": [i for i in range(total) if i not in detectados],
        "anotacoes": anotacoes(vid),
      })
      quadros = [None] * total
      for f_, xy, cf in zip(d["frames"], d["xy"], d["conf"]):
        quadros[int(f_)] = [[round(float(a), 1), round(float(b), 1), round(float(c), 3)] for (a, b), c in zip(xy, cf)]
      gravar(SAIDA / f"{vid}_pontos.json", {"versao": 1, "formato": "coco17", "ordem": "[x_px, y_px, confianca] por ponto",
                                            "quadros": quadros})
      indice[classe].append({"id": vid, "rotulo": f"{rotulos[classe]} - {numero:02d}"})
      print(f"{vid}: partição {k}, {len(x['quadro_inicio'])} janelas, {total - len(detectados)} quadros sem pessoa")
  for c in indice:
    indice[c].sort(key=lambda v: v["rotulo"])
  gravar(SAIDA / "indice.json", {"classes": [{"id": c, "nome": n, "videos": indice[c]} for c, n, _ in CLASSES[:3]]})
  n = sum(len(v) for v in indice.values())
  assert n == 56, n
  print(f"{n} vídeos exportados em {SAIDA.relative_to(RAIZ)}")


if __name__ == "__main__":
  main()
