# ============================================================
# ARQUIVO: tests/test_camera.py
#
# O QUE FAZ: testa o servidor da "Câmera ao vivo" (demo/servir.py +
# demo/camera.py) com um vídeo gravado no lugar da webcam, contra o
# pipeline em Python do modelo final (rodar_tudo.preparar_video +
# janelas_avaliacao) com o mesmo modelo completo:
#   A. preparar() da câmera dá exatamente o mesmo P que
#      rodar_tudo.preparar_video, nos 56 vídeos;
#   B. os quadros de VIDEO (os primeiros SEGUNDOS s, em PNG, sem perda)
#      vão por HTTP ao servidor, como a página faz:
#      B1. os pontos devolvidos são os da extração de pose do projeto;
#      B2. com a sequência inteira, as pontuações da câmera batem com as
#          do pipeline (diferença máxima < 1e-4);
#      B3. ao vivo (só o passado, buffer de 8 s), a concordância com as
#          pontuações do vídeo inteiro é medida e informada;
#   C. o caminho real da página (JPEG a cerca de 15 quadros/s) é medido
#      e informado, sem limite.
# Nenhum quadro é gravado; o vídeo é lido de data/videos (fora do git).
#
# Uso (da raiz do repositório): .venv/Scripts/python.exe tests/test_camera.py
# ============================================================

import importlib.util
import json
import sys
import threading
import urllib.request
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(RAIZ / "demo"), str(RAIZ / "experimentos"), str(RAIZ / "src")]

VIDEO = "v_Spinning_22"
SEGUNDOS = 12


def servidor():
  spec = importlib.util.spec_from_file_location("servir", RAIZ / "demo" / "servir.py")
  servir = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(servir)
  httpd = servir.http.server.ThreadingHTTPServer(("127.0.0.1", 0), servir.Handler)
  threading.Thread(target=httpd.serve_forever, daemon=True).start()
  return httpd, servir


def enviar(porta, img_bytes, tipo, t, sessao):
  req = urllib.request.Request(f"http://127.0.0.1:{porta}/camera/quadro", data=img_bytes, method="POST",
                               headers={"Content-Type": tipo, "X-Tempo": repr(t), "X-Sessao": sessao})
  with urllib.request.urlopen(req, timeout=60) as r:
    return json.loads(r.read())


def main():
  import cv2
  import camera as cam

  httpd, servir = servidor()
  porta = httpd.server_address[1]
  c = servir.camera()
  r = c.r

  # A
  for vid in r.videos():
    d = np.load(r.caminho("poses", c.cfg["pose"], f"{vid}.npz"))
    ref = r.preparar_video(vid, c.cfg)
    if ref is None:
      continue
    meu = cam.preparar(d["frames"].astype(np.int64), d["xy"].astype(np.float32), d["conf"].astype(np.float32), ref["fps"])
    assert np.array_equal(meu["P"], ref["P"]), vid
  print("A. preparar() igual a rodar_tudo.preparar_video nos 56 vídeos")

  # referência: o pipeline em Python no vídeo inteiro, com o modelo completo
  v = r.preparar_video(VIDEO, c.cfg)
  fps = v["fps"]
  X, _, fim = r.janelas_avaliacao(v, c.cfg)
  with c.torch.no_grad():
    ref = c.torch.softmax(c.modelo(c.q.transformar(c.torch.from_numpy(X), c.cfg)), 1).numpy()
  ref = dict(zip((fim - 1).tolist(), ref))

  # B: PNG, todos os quadros, pelo HTTP
  cap = cv2.VideoCapture(str(RAIZ / "data" / "videos" / f"{VIDEO}.mp4"))
  n = int(SEGUNDOS * fps)
  respostas = []
  for f in range(n):
    ok, img = cap.read()
    assert ok
    respostas.append(enviar(porta, cv2.imencode(".png", img)[1].tobytes(), "image/png", f / fps, "teste_png"))
  cap.release()

  d = np.load(r.caminho("poses", c.cfg["pose"], f"{VIDEO}.npz"))
  sel = d["frames"] < n
  quadros, xy, conf = d["frames"][sel].astype(np.int64), d["xy"][sel], d["conf"][sel]
  detectados = [f for f, x in enumerate(respostas) if x["pontos"] is not None]
  assert detectados == quadros.tolist(), "quadros com pessoa diferentes da extração"
  dif_px = max(abs(np.array(respostas[f]["pontos"])[:, :2] - xy[i]).max() for i, f in enumerate(detectados))
  assert dif_px <= 0.06, dif_px  # os pontos vêm arredondados a 0,1 px
  print(f"B1. mesmos {len(detectados)} quadros com pessoa; maior diferença nos pontos: {dif_px:.3f} px (arredondamento)")

  p, fins = c.pontuacoes(quadros / fps, xy, conf, todas=True)
  vt = cam.preparar(quadros, xy.astype(np.float32), conf.astype(np.float32), fps)
  Xt, _, fimt = r.janelas_avaliacao(vt, c.cfg)
  with c.torch.no_grad():
    pt = c.torch.softmax(c.modelo(c.q.transformar(c.torch.from_numpy(Xt), c.cfg)), 1).numpy()
  assert len(p) == len(pt) and len(p) > 0
  dif = abs(p - pt).max()
  assert dif < 1e-4, dif
  print(f"B2. sequência inteira: {len(p)} janelas, diferença máxima {dif:.1e} para o pipeline em Python")

  def concordancia(rotulo, resps, tempos):
    pares = [(np.array(x["pontuacoes"]), ref[int(round(t * fps))]) for x, t in zip(resps, tempos)
             if x["pontuacoes"] is not None and int(round(t * fps)) in ref]
    a = np.array([x for x, _ in pares])
    b = np.array([y for _, y in pares])
    print(f"{rotulo}: {len(pares)} janelas; diferença média {abs(a - b).mean():.3f}, máxima {abs(a - b).max():.3f}; "
          f"classe de maior pontuação igual em {(a.argmax(1) == b.argmax(1)).mean():.1%}")
    return a, b

  concordancia("B3. ao vivo (PNG, todos os quadros) contra o vídeo inteiro", respostas, [f / fps for f in range(n)])

  # C: o caminho real da página, JPEG a ~15 quadros/s
  cap = cv2.VideoCapture(str(RAIZ / "data" / "videos" / f"{VIDEO}.mp4"))
  passo = max(1, round(fps / 15))
  resps, tempos = [], []
  for f in range(n):
    ok, img = cap.read()
    if f % passo:
      continue
    resps.append(enviar(porta, cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes(), "image/jpeg", f / fps, "teste_jpg"))
    tempos.append(f / fps)
  cap.release()
  concordancia(f"C. ao vivo (JPEG, {fps / passo:.1f} quadros/s) contra o vídeo inteiro", resps, tempos)

  # o servidor recusa o que não é quadro
  for corpo, tipo in ((b"x", "image/jpeg"), (b"x", "text/plain")):
    try:
      enviar(porta, corpo, tipo, 0.0, "ruim")
      raise AssertionError("aceitou um corpo inválido")
    except urllib.error.HTTPError as e:
      assert e.code == 400
  httpd.shutdown()
  print("\nALL CAMERA TESTS PASSED")


if __name__ == "__main__":
  main()
