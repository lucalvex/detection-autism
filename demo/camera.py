# ============================================================
# ARQUIVO: demo/camera.py
#
# O QUE FAZ: o lado do servidor da "Câmera ao vivo". Recebe um quadro
# por vez (JPEG ou PNG, com o instante em segundos), roda o
# YOLO11n-Pose (primeira pessoa, como na extração) e o modelo completo
# (models/final_completo.pt, só para esta demonstração) com o mesmo
# pré-processamento do treino do modelo final:
#   - normalização robusta de rodar_tudo.preparar_video (pontos com
#     confiança < 0,5 interpolados, centro nos quadris, escala pela
#     altura do tronco, limite ±4);
#   - janela de 4 s reamostrada a 15 amostras/s, terminando no quadro
#     atual, de rodar_tudo.janelas_avaliacao (a mesma função, chamada
#     sobre o buffer).
# Diferença inevitável ao vivo: a interpolação dos pontos de baixa
# confiança só vê o passado (os últimos BUFFER_S segundos), enquanto a
# do treino usa o vídeo inteiro. tests/test_camera.py mede o efeito.
# Nada é gravado: quadros e pontos ficam só na memória, no buffer.
# ============================================================

import json
import sys
import threading
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(RAIZ / "experimentos"), str(RAIZ / "src")]

BUFFER_S = 8.0  # o dobro da janela; a interpolação ao vivo só vê este passado
MODELO_POSE = RAIZ / "models" / "yolo11n-pose.pt"
MODELO = RAIZ / "models" / "final_completo.pt"
REGISTRO = RAIZ / "experimentos" / "modelo-final-completo.json"


def preparar(quadros, xy, conf, fps):
  """Ramo "robusta" de rodar_tudo.preparar_video, sobre arrays em memória.
  tests/test_camera.py exige o mesmo P que a função original."""
  import rodar_tudo as r
  xy = xy.astype(np.float64)
  for k in range(17):
    ok = conf[:, k] >= r.LIMIAR_CONF
    if ok.any() and not ok.all():
      for c in (0, 1):
        xy[~ok, k, c] = np.interp(quadros[~ok], quadros[ok], xy[ok, k, c])
  quadril = (xy[:, 11] + xy[:, 12]) / 2
  escala = np.linalg.norm((xy[:, 5] + xy[:, 6]) / 2 - quadril, axis=1)
  ruim = escala < 1
  if ruim.all():
    escala[:] = 1
  elif ruim.any():
    idx = np.maximum.accumulate(np.where(ruim, -1, np.arange(len(escala))))
    escala = escala[np.where(idx < 0, np.argmax(~ruim), idx)]
  P = np.clip((xy - quadril[:, None]) / escala[:, None, None], -r.LIMITE_NORM, r.LIMITE_NORM).astype(np.float32)
  return {"quadros": quadros, "P": P, "fps": fps}


class Camera:

  def __init__(self):
    # a ultralytics antes do torch e 2 threads: a mesma ordem da extração das poses
    from ultralytics import YOLO
    import torch
    import rodar_q as rq
    torch.set_num_threads(2)
    self.r, self.q, self.torch = rq.r, rq.q, torch
    self.cfg = json.loads(REGISTRO.read_text(encoding="utf-8"))["configuracao"]
    self.yolo = YOLO(str(MODELO_POSE))
    self.modelo = self.q.rede(self.cfg)
    self.modelo.load_state_dict(torch.load(MODELO))
    self.modelo.eval()
    self.trava = threading.Lock()
    self.sessao = None
    self.limpar()

  def limpar(self):
    self.t, self.xy, self.conf = [], [], []

  def pose(self, img):
    """Pontos da primeira pessoa (17 x 2 em pixels, 17 confianças) ou None."""
    for res in self.yolo(img, verbose=False):
      if res.keypoints is None or len(res.keypoints.xy) == 0:
        continue
      return res.keypoints.xy.cpu().numpy()[0], res.keypoints.conf.cpu().numpy()[0]
    return None

  def pontuacoes(self, t, xy, conf, todas=False):
    """Pontuações (softmax) da janela de 4 s que termina no último quadro,
    ou de todas as janelas (todas=True). Tempos em segundos; o buffer usa
    milissegundos como índice de quadro, então fps = 1000."""
    quadros = np.round(np.asarray(t) * 1000).astype(np.int64)
    v = preparar(quadros, np.asarray(xy, np.float32), np.asarray(conf, np.float32), 1000.0)
    j = self.r.janelas_avaliacao(v, self.cfg)
    if j is None:
      return None
    X, _, fim = j
    if not todas:
      if fim[-1] != quadros[-1] + 1:  # o trecho atual ainda não tem 4 s
        return None
      X = X[-1:]
    with self.torch.no_grad():
      p = self.torch.softmax(self.modelo(self.q.transformar(self.torch.from_numpy(X), self.cfg)), 1).numpy()
    return (p, fim - 1) if todas else p[0]

  def processar(self, dados, t, sessao):
    """Um quadro codificado (bytes) no instante t (s) da sessão do navegador."""
    import cv2
    img = cv2.imdecode(np.frombuffer(dados, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
      raise ValueError("imagem inválida")
    with self.trava:
      if sessao != self.sessao or (self.t and t <= self.t[-1]):
        self.sessao = sessao
        self.limpar()
      p = self.pose(img)
      if p is not None:
        self.t.append(t)
        self.xy.append(p[0])
        self.conf.append(p[1])
        while self.t[0] < t - BUFFER_S:
          del self.t[0], self.xy[0], self.conf[0]
      ponts = self.pontuacoes(self.t, self.xy, self.conf) if p is not None else None
      # tempo contínuo com pessoa (lacunas acima de 0,5 s quebram o trecho, como no treino)
      continuo = 0.0
      if p is not None:
        i = len(self.t) - 1
        while i > 0 and self.t[i] - self.t[i - 1] <= self.r.MAX_LACUNA_S:
          i -= 1
        continuo = t - self.t[i]
    return {
      "largura": int(img.shape[1]), "altura": int(img.shape[0]),
      "pontos": None if p is None else [[round(float(a), 1), round(float(b), 1), round(float(c), 3)]
                                        for (a, b), c in zip(p[0], p[1])],
      "pontuacoes": None if ponts is None else [float(x) for x in ponts],
      "segundos_continuos": round(continuo, 2),
    }
