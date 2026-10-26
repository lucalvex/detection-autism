# ============================================================
# ARQUIVO: src/capturar_interface.py
#
# O QUE FAZ: refaz as figuras da interface de demonstração usadas no
# TCC, com um navegador sem interface gráfica (Playwright, usando o
# Edge instalado), janela de 800 px de largura, fator de escala 3 e
# tema claro. Cada figura é recortada pelos próprios elementos da
# página, não pela tela inteira:
#   - fig_interface_trechos.png: modo apresentação, etapa "Trechos",
#     v_Spinning_22 (anotação do XML e trechos detectados por classe);
#   - fig_interface_revisao.png: modo revisão, v_ArmFlapping_12, sem
#     abrir arquivo de vídeo (linha do tempo por classe e lista de
#     trechos para revisão, um recorte abaixo do outro);
#   - fig_interface_confusao.png: modo apresentação, as duas matrizes
#     de confusão;
#   - fig_interface_achados*.png: modo apresentação, achados posteriores
#     com os quatro painéis e a legenda dos modelos no topo de cada parte
#     (o aviso de que são posteriores fica para a legenda do texto).
# Quando uma figura com vários recortes fica alta demais para uma página
# (altura maior que PROPORCAO_MAXIMA vezes a largura), ela é dividida em
# partes numeradas (_1, _2, ...), sem cortar nenhum recorte ao meio.
#
# Nenhum quadro de vídeo nem imagem de criança pode aparecer: antes de
# cada recorte, confere na página que não há <img>, <video>, <canvas>,
# <image> de SVG nem imagem de fundo dentro dos elementos recortados, e
# que nenhum vídeo foi carregado. Se houver, para sem gravar. O
# esqueleto desenhado a partir dos pontos (SVG) é permitido.
#
# Usa o servidor da interface (demo/servir.py); se ele não estiver no
# ar, sobe um só durante as capturas.
#
# Uso (da raiz do repositório):
#   .venv/Scripts/python.exe src/capturar_interface.py [--saida figuras] [--navegador msedge]
# ============================================================

import argparse
import importlib.util
import io
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

RAIZ = Path(__file__).resolve().parent.parent
LARGURA_JANELA = 800
ALTURA_JANELA = 900
ESCALA = 3
VAO = 16                 # px de página entre recortes empilhados (o mesmo vão dos painéis)
PROPORCAO_MAXIMA = 1.5   # altura/largura; a área útil de uma página A4 com margens ABNT é 16 x 24,7 cm
FONTE = "Atkinson Hyperlegible Next"


def porta_do_servidor():
  spec = importlib.util.spec_from_file_location("servir", RAIZ / "demo" / "servir.py")
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod.PORTA


def no_ar(url):
  try:
    with urllib.request.urlopen(url, timeout=2) as r:
      return r.status == 200
  except OSError:
    return False


def subir_servidor(base):
  if no_ar(base + "apresentacao.html"):
    return None
  proc = subprocess.Popen([sys.executable, str(RAIZ / "demo" / "servir.py")], cwd=RAIZ,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  for _ in range(50):
    if no_ar(base + "apresentacao.html"):
      return proc
    time.sleep(0.2)
  proc.terminate()
  raise SystemExit("O servidor da interface (demo/servir.py) não respondeu.")


# ---------- conferência: nada de quadro de vídeo ou imagem ----------

CONFERIR = """(seletores) => {
  const problemas = [];
  for (const s of seletores) {
    for (const el of document.querySelectorAll(s)) {
      for (const n of [el, ...el.querySelectorAll("*")]) {
        const tag = n.tagName.toLowerCase();
        if (["img", "video", "canvas", "picture", "iframe", "object", "embed", "image"].includes(tag)) problemas.push(`<${tag}> em ${s}`);
        if (getComputedStyle(n).backgroundImage.includes("url(")) problemas.push(`imagem de fundo em ${s}`);
      }
    }
  }
  for (const v of document.querySelectorAll("video")) {
    if (v.currentSrc || v.getAttribute("src") || v.srcObject || v.readyState > 0) problemas.push("há vídeo carregado na página");
  }
  return problemas;
}"""


def conferir(pagina, seletores):
  problemas = pagina.evaluate(CONFERIR, seletores)
  if problemas:
    raise SystemExit("Captura interrompida, nada foi gravado: " + "; ".join(sorted(set(problemas))))


# ---------- recorte ----------

def recortar(pagina, seletores):
  """Recorta a união das caixas dos elementos (coordenadas da página)."""
  caixas = []
  for s in seletores:
    els = pagina.query_selector_all(s)
    if not els:
      raise SystemExit(f"Elemento não encontrado: {s}")
    for el in els:
      el.scroll_into_view_if_needed()
      caixas.append(pagina.evaluate("""(el) => {
        const r = el.getBoundingClientRect();
        return [r.left + scrollX, r.top + scrollY, r.right + scrollX, r.bottom + scrollY];
      }""", el))
  x0 = min(c[0] for c in caixas); y0 = min(c[1] for c in caixas)
  x1 = max(c[2] for c in caixas); y1 = max(c[3] for c in caixas)
  conferir(pagina, seletores)
  png = pagina.screenshot(full_page=True, animations="disabled", caret="hide",
                          clip={"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0})
  return Image.open(io.BytesIO(png)).convert("RGB")


def empilhar(recortes, fundo):
  vao = VAO * ESCALA
  w = max(r.width for r in recortes)
  h = sum(r.height for r in recortes) + vao * (len(recortes) - 1)
  img = Image.new("RGB", (w, h), fundo)
  y = 0
  for r in recortes:
    img.paste(r, (0, y))
    y += r.height + vao
  return img


def dividir(recortes, cabecalho):
  """Agrupa os recortes, em ordem, de modo que nenhuma parte passe de PROPORCAO_MAXIMA.
  O cabeçalho, se houver, abre cada parte."""
  inicio = [cabecalho] if cabecalho else []
  partes, atual = [], list(inicio)
  for r in recortes:
    teste = atual + [r]
    w = max(x.width for x in teste)
    h = sum(x.height for x in teste) + VAO * ESCALA * (len(teste) - 1)
    if len(atual) > len(inicio) and h / w > PROPORCAO_MAXIMA:
      partes.append(atual)
      atual = inicio + [r]
    else:
      atual = teste
  partes.append(atual)
  return partes


# cor de fundo atrás do elemento: a do primeiro ancestral com fundo não transparente
FUNDO = r"""(s) => {
  for (let n = document.querySelector(s).parentElement; n; n = n.parentElement) {
    const m = getComputedStyle(n).backgroundColor.match(/[\d.]+/g).map(Number);
    if (m.length < 4 || m[3] > 0) return m.slice(0, 3);
  }
  return [255, 255, 255];
}"""


def gravar(nome, blocos, pagina, saida, cabecalho=None):
  """blocos: lista de listas de seletores; cada bloco vira um recorte, e os
  recortes ficam um abaixo do outro. cabecalho: seletores de um recorte
  repetido no topo de cada parte (por exemplo, a legenda)."""
  fundo = tuple(pagina.evaluate(FUNDO, blocos[0][0]))
  recortes = [recortar(pagina, b) for b in blocos]
  partes = dividir(recortes, recortar(pagina, cabecalho) if cabecalho else None)
  gravados = []
  for i, parte in enumerate(partes, 1):
    arquivo = saida / (f"{nome}.png" if len(partes) == 1 else f"{nome}_{i}.png")
    img = empilhar(parte, fundo)
    img.save(arquivo, optimize=True)
    gravados.append(arquivo)
    print(f"{arquivo.relative_to(RAIZ) if arquivo.is_relative_to(RAIZ) else arquivo}: {img.width} x {img.height} px")
  # partes antigas de uma divisão diferente não podem ficar para trás
  for velho in saida.glob(f"{nome}*.png"):
    if velho not in gravados and (velho.stem == nome or velho.stem[len(nome):].lstrip("_").isdigit()):
      velho.unlink()
      print(f"removido: {velho.name}")


def abrir(pagina, url, pronto):
  pagina.goto(url)
  pagina.wait_for_selector(pronto)
  pagina.evaluate("document.fonts.ready")
  if not pagina.evaluate(f"document.fonts.check('12px \"{FONTE}\"')"):
    raise SystemExit(f"A fonte {FONTE} não carregou (ela vem do Google Fonts; é preciso estar conectado).")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--saida", type=Path, default=RAIZ / "figuras")
  ap.add_argument("--navegador", default="msedge", help="canal do Playwright: msedge, chrome ou chromium")
  args = ap.parse_args()
  saida = args.saida.resolve()
  saida.mkdir(parents=True, exist_ok=True)

  base = f"http://127.0.0.1:{porta_do_servidor()}/demo/"
  servidor = subir_servidor(base)
  try:
    with sync_playwright() as p:
      canal = None if args.navegador == "chromium" else args.navegador
      navegador = p.chromium.launch(channel=canal, headless=True)
      pagina = navegador.new_page(viewport={"width": LARGURA_JANELA, "height": ALTURA_JANELA},
                                  device_scale_factor=ESCALA, color_scheme="light")

      # modo apresentação
      abrir(pagina, base + "apresentacao.html?video=v_Spinning_22", "#posteriores-paineis .painel")
      if pagina.evaluate("new URLSearchParams(location.search).get('video')") != "v_Spinning_22":
        raise SystemExit("O modo apresentação não abriu o v_Spinning_22.")
      pagina.locator(".etapa-btn").filter(has=pagina.locator(".t", has_text=re.compile(r"^Trechos$"))).click()
      pagina.wait_for_selector("#etapa-visual .mini")
      if pagina.text_content("#etapa-titulo") != "Trechos":
        raise SystemExit("O modo apresentação não abriu a etapa Trechos.")
      pagina.mouse.move(0, 0)
      gravar("fig_interface_trechos", [["#etapa-visual"]], pagina, saida)
      gravar("fig_interface_confusao", [[".matrizes"]], pagina, saida)
      # o aviso do topo da seção ocupa a largura toda da página (1316 px)
      # e deixaria os painéis (uma coluna de 520 px nesta janela) num canto
      # da figura: ele fica para a legenda do texto, e a legenda dos
      # modelos abre cada parte
      gravar("fig_interface_achados",
             [[f"#posteriores-paineis .painel:nth-child({i})"]
              for i in range(1, pagina.locator("#posteriores-paineis .painel").count() + 1)],
             pagina, saida, cabecalho=["#legenda-modelos .item"])

      # modo revisão, sem abrir arquivo de vídeo
      abrir(pagina, base + "revisao.html?sessao=v_ArmFlapping_12", "#lista-trechos > *")
      if pagina.evaluate("document.getElementById('sel-sessao').value") != "v_ArmFlapping_12":
        raise SystemExit("O modo revisão não abriu o v_ArmFlapping_12.")
      pagina.mouse.move(0, 0)
      gravar("fig_interface_revisao", [[".linha-tempo"], [".painel-trechos"]], pagina, saida)
      navegador.close()
  finally:
    if servidor:
      servidor.terminate()


if __name__ == "__main__":
  main()
