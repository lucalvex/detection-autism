# ============================================================
# ARQUIVO: demo/servir.py
#
# O QUE FAZ: servidor local da interface de demonstração, só em
# 127.0.0.1. Entrega apenas:
#   - os arquivos da interface (demo/);
#   - os JSON de sessão exportados (saidas/*_sessao.json e
#     saidas/*_pontos.json);
#   - os JSON de resultado listados em RESULTADOS.
# Qualquer outro caminho responde 404 -- em particular, nenhum vídeo
# nem CSV de pose do projeto pode ser pedido pelo navegador. O vídeo
# é aberto pelo usuário com o seletor de arquivo da página e fica só
# na memória do navegador (URL.createObjectURL): não é copiado,
# embutido nem enviado.
#
# Câmera ao vivo: POST /camera/quadro com um quadro JPEG da webcam;
# devolve o esqueleto e as pontuações (demo/camera.py). Só aceita
# conexões de 127.0.0.1, e nada é gravado.
#
# Uso (da raiz do repositório): .venv/Scripts/python.exe demo/servir.py
# ============================================================

import http.server
import json
import os
import re
import threading
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
PORTA = 8765
MAX_QUADRO = 4 * 1024 * 1024  # bytes de um quadro enviado pela câmera

_camera, _trava_camera = None, threading.Lock()


def camera():
  """Carrega o YOLO e o modelo completo só no primeiro quadro."""
  global _camera
  with _trava_camera:
    if _camera is None:
      import sys
      sys.path.insert(0, str(RAIZ / "demo"))
      from camera import Camera
      _camera = Camera()
  return _camera

RESULTADOS = {
  "/data/results/metricas.json",
  "/data/results/comparacao_rodada_a/metricas.json",
  "/data/results/comparacao_base_reseeded/metricas.json",
  "/data/results/analises_posteriores.json",
  "/data/results/videos_desempenho_mediano.json",
  "/data/results/comparacao_final.json",
  "/data/results/resumo_dados.json",
}

PERMITIDOS = [
  re.compile(r"^/demo/[A-Za-z0-9_\-/]+\.(html|css|js|json)$"),
  # elementos visuais da interface, como a logo; nunca quadros de vídeo
  re.compile(r"^/demo/assets/[A-Za-z0-9_\-]+\.(png|svg)$"),
  re.compile(r"^/saidas/(final/)?v_[A-Za-z0-9_]+_(sessao|pontos)\.json$"),
  re.compile(r"^/saidas/final/indice\.json$"),
]


def permitido(caminho):
  if ".." in caminho or caminho.startswith("/demo/servir"):
    return False
  if caminho in RESULTADOS:
    return True
  return any(p.match(caminho) for p in PERMITIDOS)


class Handler(http.server.SimpleHTTPRequestHandler):

  # uma conexão que abre e não manda nada não pode prender a thread
  timeout = 10

  extensions_map = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".png": "image/png",
    ".svg": "image/svg+xml",
  }

  def __init__(self, *args, **kwargs):
    super().__init__(*args, directory=str(RAIZ), **kwargs)

  def do_GET(self):
    caminho = self.path.split("?", 1)[0]
    if caminho in ("/", "/demo", "/demo/"):
      self.send_response(302)
      self.send_header("Location", "/demo/index.html")
      self.end_headers()
      return
    if not permitido(caminho):
      self.send_error(404)
      return
    super().do_GET()

  def do_POST(self):
    if self.path != "/camera/quadro" or self.client_address[0] != "127.0.0.1":
      self.send_error(404)
      return
    try:
      n = int(self.headers.get("Content-Length", "0"))
      t = float(self.headers["X-Tempo"])
      sessao = self.headers["X-Sessao"][:64]
    except (TypeError, ValueError):
      self.send_error(400)
      return
    if not 0 < n <= MAX_QUADRO or self.headers.get("Content-Type") not in ("image/jpeg", "image/png"):
      self.send_error(400)
      return
    try:
      resposta = camera().processar(self.rfile.read(n), t, sessao)
    except ValueError:
      self.send_error(400)
      return
    corpo = json.dumps(resposta).encode("utf-8")
    self.send_response(200)
    self.send_header("Content-Type", "application/json; charset=utf-8")
    self.send_header("Content-Length", str(len(corpo)))
    self.end_headers()
    self.wfile.write(corpo)

  def do_HEAD(self):
    self.do_GET()

  def end_headers(self):
    self.send_header("Cache-Control", "no-store")
    super().end_headers()

  def list_directory(self, path):
    self.send_error(404)
    return None


if __name__ == "__main__":
  # várias conexões ao mesmo tempo: uma conexão presa do navegador não
  # bloqueia as outras. No Windows, SO_REUSEADDR deixa um segundo
  # servidor ocupar a mesma porta em silêncio (e um servidor antigo seguir
  # respondendo); lá ele fica desligado, para a porta ocupada dar erro.
  http.server.ThreadingHTTPServer.allow_reuse_address = os.name != "nt"
  with http.server.ThreadingHTTPServer(("127.0.0.1", PORTA), Handler) as httpd:
    print(f"Interface em http://127.0.0.1:{PORTA}/demo/index.html", flush=True)
    httpd.serve_forever()
