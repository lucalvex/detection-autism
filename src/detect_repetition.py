import numpy as np
from scipy.signal import find_peaks

def calculate_motion(hand_positions):

  motion_signal = []

  previous = None

  for frame in hand_positions:

    if len(frame) == 0:
      continue

    current = np.array(frame[0])

    if previous is not None:
      dist = np.linalg.norm(current - previous)
      motion_signal.append(dist)

    previous = current

  return motion_signal


def detect_repetitive_pattern(signal):

  peaks, _ = find_peaks(signal, distance=5)

  if len(peaks) > 20:
    return True

  return False