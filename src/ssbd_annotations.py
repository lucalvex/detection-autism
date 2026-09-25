# ============================================================
# ARQUIVO: ssbd_annotations.py
#
# O QUE FAZ: lê as anotações XML do SSBD (intervalos de tempo de
# comportamentos, uma por vídeo) e devolve os dados em uma estrutura
# Python simples. Usado pela avaliação temporal (evaluate_temporal.py)
# como gabarito, e pelo exportador de sessão para preencher o campo
# opcional `anotacoes` do JSON.
#
# FORMATO REAL (inspecionado nos 75 arquivos de
# C:\projetos\tcc\download-yt\xmls\*.xml antes de escrever este
# leitor -- não é o schema oficial do SSBD, é o que os arquivos
# realmente têm):
#
#   <video id="v_ArmFlapping_08" keyword="...">
#     <url>...</url>
#     <height>240</height>
#     <width>320</width>
#     <frames>6084</frames>
#     <persons>2</persons>
#     <duration>253s</duration>
#     <conversation>yes</conversation>
#     <behaviours count="7" id="b_Set_01">
#       <behaviour id="b_01">
#         <time>0018:0021</time>
#         <bodypart>hand</bodypart>
#         <category>armflapping</category>
#         <intensity>high</intensity>
#         <modality>video</modality>
#       </behaviour>
#       ...
#     </behaviours>
#   </video>
#
# ACHADOS (verificados nos 75 arquivos, não presumidos):
#   - <duration> é sempre "<inteiro>s" (segundos).
#   - <time> usa ':' OU '-' como separador entre início e fim (120
#     arquivos usam ':', 13 usam '-' -- ex. v_Spinning_07.xml).
#   - cada lado do separador NÃO é segundos puros: é minutos+segundos
#     concatenados, onde os últimos 2 dígitos são segundos e o
#     restante (se houver) são minutos -- ex. "0338" = 3 min 38 s =
#     218 s, "18" = 18 s. Confirmado testando as duas hipóteses contra
#     a duração real do vídeo nas 133 anotações: a leitura como
#     segundos puros excede a duração do vídeo em 26 casos (impossível);
#     a leitura minutos+segundos nunca excede, em nenhuma das 133.
#   - só 3 categorias aparecem nos 75 arquivos: armflapping (61),
#     spinning (38), headbanging (34). rocking/handmovement/normal
#     não têm nenhuma anotação -- consistente com o dataset de treino
#     não ter vídeo nenhum dessas 3 classes.
#   - um vídeo pode ter mais de uma categoria anotada (o rótulo do
#     nome do arquivo é só uma aproximação do comportamento
#     predominante, não o gabarito completo).
# ============================================================

import re
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_time_token(token):
  """Converte um lado do intervalo <time> em segundos.

  Os últimos 2 dígitos do token são segundos; o que sobrar (se algo
  sobrar) são minutos. "0338" -> 3*60+38 = 218. "18" -> 18.
  """

  token = token.strip()

  if len(token) <= 2:
    return int(token)

  minutes = int(token[:-2])
  seconds = int(token[-2:])

  return minutes * 60 + seconds


def parse_interval(time_str):
  """Separa um <time> em (inicio_s, fim_s). Aceita ':' ou '-' como
  separador -- os dois aparecem nos arquivos reais."""

  time_str = time_str.strip()

  if ":" in time_str:
    start_tok, end_tok = time_str.split(":", 1)
  elif "-" in time_str:
    start_tok, end_tok = time_str.split("-", 1)
  else:
    raise ValueError(f"Formato de intervalo <time> desconhecido: {time_str!r}")

  return parse_time_token(start_tok), parse_time_token(end_tok)


def load_annotations(xml_path):
  """Lê um arquivo de anotação e devolve um dict com os metadados do
  vídeo e a lista de comportamentos anotados (em segundos)."""

  tree = ET.parse(xml_path)
  root = tree.getroot()

  duration_str = root.findtext("duration") or "0s"
  duration_s = int(re.sub(r"[^0-9]", "", duration_str) or 0)

  behaviours = []
  behaviours_el = root.find("behaviours")

  if behaviours_el is not None:

    for b in behaviours_el:

      time_str = b.findtext("time")

      if time_str is None:
        continue

      start_s, end_s = parse_interval(time_str)

      behaviours.append({
        "id": b.get("id"),
        "category": (b.findtext("category") or "").strip().lower(),
        "bodypart": (b.findtext("bodypart") or "").strip(),
        "intensity": (b.findtext("intensity") or "").strip(),
        "modality": (b.findtext("modality") or "").strip(),
        "start_s": start_s,
        "end_s": end_s,
      })

  return {
    "video_id": root.get("id"),
    "frames": int(root.findtext("frames") or 0),
    "persons": int(root.findtext("persons") or 0),
    "duration_s": duration_s,
    "width": int(root.findtext("width") or 0),
    "height": int(root.findtext("height") or 0),
    "behaviours": behaviours,
  }


def find_annotation_file(video_id, xmls_dir):
  """Procura o XML de anotação de um video_id num diretório. Devolve
  None se não existir (nem todo vídeo do dataset necessariamente tem
  uma anotação correspondente disponível)."""

  candidate = Path(xmls_dir) / f"{video_id}.xml"

  return candidate if candidate.exists() else None
