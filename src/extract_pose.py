# gera CVS com o keypoints

import cv2
import pandas as pd
from ultralytics import YOLO
import numpy as np

VIDEO_PATH = "data/videos/v_HeadBanging_18.mp4"
OUTPUT_CSV = "data/poses/HeadBanging.csv"

model = YOLO("models/yolo11n-pose.pt")

cap = cv2.VideoCapture(VIDEO_PATH)

data = []
frame_id = 0
previous_kp = None

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
      break

    results = model(frame, verbose=False)

    for result in results:

      if result.keypoints is None:
        continue

      people = result.keypoints.xy.cpu().numpy()

      if len(people) == 0:
        continue

      kp = people[0]
      
      # Centro do quadril
      hip_x = (kp[11][0] + kp[12][0]) / 2
      hip_y = (kp[11][1] + kp[12][1]) / 2
      
      left_shoulder = kp[5]
      right_shoulder = kp[6]

      body_size = np.sqrt(
        (right_shoulder[0] - left_shoulder[0]) ** 2 +
        (right_shoulder[1] - left_shoulder[1]) ** 2
      )
      
      current_kp = []

      if body_size < 1:
        body_size = 1

      row = {
        "frame": frame_id
      }

      for i, (x, y) in enumerate(kp):

        x = (x - hip_x) / body_size
        y = (y - hip_y) / body_size

        current_kp.append((x, y))

        if previous_kp is None:
          dx = 0.0
          dy = 0.0

        else:
          dx = x - previous_kp[i][0]
          dy = y - previous_kp[i][1]

        row[f"x{i}"] = float(x)
        row[f"y{i}"] = float(y)

        row[f"dx{i}"] = float(dx)
        row[f"dy{i}"] = float(dy)

      data.append(row)
      
      previous_kp = current_kp

    frame_id += 1

cap.release()

pd.DataFrame(data).to_csv(
  OUTPUT_CSV,
  index=False
)

print(f"Saved {len(data)} frames")