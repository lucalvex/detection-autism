# ============================================================
# ARQUIVO: tests/test_pose_features.py
#
# O QUE FAZ: testa normalize_frame_keypoints()/flatten_features()
# (src/pose_features.py) contra a fórmula original (inline, de antes
# da refatoração que unificou extract_pose.py/predict_video.py/
# episode_analysis.py) -- garante que a refatoração não mudou nenhum
# número. Rode com: python tests/test_pose_features.py
# ============================================================

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pose_features import flatten_features, normalize_frame_keypoints


def old_inline_normalize(kp, previous_kp):
  """Réplica exata da lógica original (pré-refatoração), para comparação."""

  hip_x = (kp[11][0] + kp[12][0]) / 2
  hip_y = (kp[11][1] + kp[12][1]) / 2
  left_shoulder = kp[5]
  right_shoulder = kp[6]
  body_size = np.sqrt(
    (right_shoulder[0] - left_shoulder[0]) ** 2 +
    (right_shoulder[1] - left_shoulder[1]) ** 2
  )
  if body_size < 1:
    body_size = 1

  current_kp = []
  row_x, row_y, row_dx, row_dy = [], [], [], []

  for i, (x, y) in enumerate(kp):
    x = (x - hip_x) / body_size
    y = (y - hip_y) / body_size
    current_kp.append((x, y))
    if previous_kp is None:
      dx, dy = 0.0, 0.0
    else:
      dx = x - previous_kp[i][0]
      dy = y - previous_kp[i][1]
    row_x.append(float(x))
    row_y.append(float(y))
    row_dx.append(float(dx))
    row_dy.append(float(dy))

  return row_x, row_y, row_dx, row_dy, current_kp


def test_regressao_contra_formula_antiga():

  rng = np.random.default_rng(42)

  kp1 = rng.uniform(0, 500, size=(17, 2))
  old_x1, old_y1, old_dx1, old_dy1, old_prev1 = old_inline_normalize(kp1, None)
  new_features1, new_prev1 = normalize_frame_keypoints(kp1, None)

  for i, (x, y, dx, dy) in enumerate(new_features1):
    assert abs(x - old_x1[i]) < 1e-9
    assert abs(y - old_y1[i]) < 1e-9
    assert abs(dx - old_dx1[i]) < 1e-9
    assert abs(dy - old_dy1[i]) < 1e-9

  assert old_prev1 == new_prev1

  kp2 = rng.uniform(0, 500, size=(17, 2))
  old_x2, old_y2, old_dx2, old_dy2, _ = old_inline_normalize(kp2, old_prev1)
  new_features2, _ = normalize_frame_keypoints(kp2, new_prev1)

  for i, (x, y, dx, dy) in enumerate(new_features2):
    assert abs(x - old_x2[i]) < 1e-9
    assert abs(y - old_y2[i]) < 1e-9
    assert abs(dx - old_dx2[i]) < 1e-9
    assert abs(dy - old_dy2[i]) < 1e-9

  flat = flatten_features(new_features2)
  assert len(flat) == 68
  assert flat[:8] == [old_x2[0], old_y2[0], old_dx2[0], old_dy2[0], old_x2[1], old_y2[1], old_dx2[1], old_dy2[1]]


if __name__ == "__main__":

  test_regressao_contra_formula_antiga()
  print("OK: normalize_frame_keypoints/flatten_features batem com a fórmula antiga (frame 1 e 2, ordem x,y,dx,dy)")

  print("\nALL POSE_FEATURES TESTS PASSED")
