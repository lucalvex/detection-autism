# ============================================================
# ARQUIVO: src/capturar_interface_videos.py
#
# O QUE FAZ: figura do modo "Vídeos" da interface simplificada para o
# TCC (figuras/fig_interface_videos.png): "Arm flapping - 01"
# selecionado, sem abrir o arquivo de vídeo, janela de 900 px de
# largura, fator de escala 3, recorte da linha do seletor de vídeo até a
# legenda abaixo da linha do tempo. Reaproveita o servidor, a
# conferência (nenhum <video>, <img>, <canvas> nem imagem de fundo no
# recorte) e o recorte de capturar_interface.py. O <video> vazio da
# página é removido antes da captura, para que a conferência valha para
# todo o recorte.
#
# Uso (da raiz do repositório):
#   .venv/Scripts/python.exe src/capturar_interface_videos.py
# ============================================================

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
import capturar_interface as ci  # noqa: E402

VIDEO = "v_ArmFlapping_01"
LARGURA = 900
RECORTE = ["#modo-videos .barra", "#palco-video", "#linha-tempo", "#modo-videos .legenda"]


def main():
  base = f"http://127.0.0.1:{ci.porta_do_servidor()}/demo/"
  servidor = ci.subir_servidor(base)
  try:
    with sync_playwright() as p:
      navegador = p.chromium.launch(channel="msedge", headless=True)
      pagina = navegador.new_page(viewport={"width": LARGURA, "height": ci.ALTURA_JANELA},
                                  device_scale_factor=ci.ESCALA, color_scheme="light")
      ci.abrir(pagina, base + f"index.html?modo=videos&video={VIDEO}", "#linha-tempo .trilha")
      if pagina.evaluate("document.getElementById('sel-video').value") != VIDEO:
        raise SystemExit(f"O modo Vídeos não abriu o {VIDEO}.")
      if pagina.evaluate("document.getElementById('video').hasAttribute('src')"):
        raise SystemExit("Há um arquivo de vídeo aberto.")
      pagina.evaluate("document.getElementById('video').remove()")
      pagina.mouse.move(0, 0)
      img = ci.recortar(pagina, RECORTE)
      saida = ci.RAIZ / "figuras" / "fig_interface_videos.png"
      img.save(saida, optimize=True)
      print(f"{saida.relative_to(ci.RAIZ)}: {img.width} x {img.height} px")
      navegador.close()
  finally:
    if servidor:
      servidor.terminate()


if __name__ == "__main__":
  main()
