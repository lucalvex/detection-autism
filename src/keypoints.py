# Os keypoints são pontos anatômicos que o modelo de pose estimation identifica no corpo de pessoa
from ultralytics import YOLO

model = YOLO("yolo11n-pose.pt")

results = model("v_HeadBanging_18.mp4")

for result in results:
  print(result.keypoints.xy)