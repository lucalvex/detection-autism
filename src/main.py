from extract_pose import extract_hand_positions
from detect_repetition import (
    calculate_motion,
    detect_repetitive_pattern
)

video = "videos/video.mp4"

positions = extract_hand_positions(video)

signal = calculate_motion(positions)

result = detect_repetitive_pattern(signal)

if result:
  print("Movimento repetitivo detectado")
else:
  print("Sem repetição significativa")