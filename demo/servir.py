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
# Uso (da raiz do repositório): python demo/servir.py
# ============================================================

import http.server
import re
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
PORTA = 8765

RESULTADOS = {
  "/data/results/comparacao_rodada_a/metricas.json",
  "/data/results/comparacao_base_reseeded/metricas.json",
  "/data/results/analises_posteriores.json",
  "/data/results/videos_desempenho_mediano.json",
  "/data/results/comparacao_final.json",
  "/data/results/resumo_dados.json",
}

PERMITIDOS = [
  re.compile(r"^/demo/[A-Za-z0-9_\-/]+\.(html|css|js|json)$"),
  re.compile(r"^/saidas/v_[A-Za-z0-9_]+_(sessao|pontos)\.json$"),
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
  }

  def __init__(self, *args, **kwargs):
    super().__init__(*args, directory=str(RAIZ), **kwargs)

  def do_GET(self):
    caminho = self.path.split("?", 1)[0]
    if caminho in ("/", "/demo", "/demo/"):
      self.send_response(302)
      self.send_header("Location", "/demo/apresentacao.html")
      self.end_headers()
      return
    if not permitido(caminho):
      self.send_error(404)
      return
    super().do_GET()

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
  # bloqueia as outras
  http.server.ThreadingHTTPServer.allow_reuse_address = True
  with http.server.ThreadingHTTPServer(("127.0.0.1", PORTA), Handler) as httpd:
    print(f"Interface em http://127.0.0.1:{PORTA}/demo/apresentacao.html", flush=True)
    httpd.serve_forever()
