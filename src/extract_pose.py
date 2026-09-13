# ============================================================
# ARQUIVO: extract_pose.py
#
# O QUE FAZ: primeira etapa do pipeline. Para cada vídeo em
# data/videos/, roda a YOLO11n-Pose frame a frame, extrai os 17
# keypoints COCO da primeira pessoa detectada, normaliza essas
# coordenadas (centraliza no quadril + escala pela distância entre os
# ombros, para independer de posição/distância da câmera) e calcula o
# deslocamento de cada ponto em relação ao frame anterior (features de
# movimento). Salva 1 CSV por vídeo em data/poses/, nomeado pelo
# video_id (nome do arquivo de vídeo sem extensão) — esse video_id é
# usado depois por create_sequences.py e train_classifier.py para
# nunca misturar frames do mesmo vídeo entre treino e teste.
# ============================================================

import cv2
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

from pose_features import normalize_frame_keypoints

VIDEOS_DIR = Path("data/videos")
OUTPUT_DIR = Path("data/poses")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model = YOLO("models/yolo11n-pose.pt")


def extract_video(video_path: Path, output_csv: Path):

    cap = cv2.VideoCapture(str(video_path))

    data = []
    frame_id = 0
    previous_kp = None  # keypoints do frame anterior, para calcular dx/dy; resetado a cada vídeo

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # roda a detecção de pose neste frame
        results = model(frame, verbose=False)

        for result in results:

            if result.keypoints is None:
                continue

            # coordenadas (x, y) em pixels, uma linha por pessoa detectada
            people = result.keypoints.xy.cpu().numpy()

            if len(people) == 0:
                continue

            # assume 1 pessoa por vídeo: usa só a primeira detectada
            kp = people[0]

            # normalização (centro do quadril + escala pelos ombros) e
            # features de movimento (dx/dy) -- função compartilhada com
            # predict_video.py e episode_analysis.py
            features, previous_kp = normalize_frame_keypoints(kp, previous_kp)

            row = {
                "frame": frame_id
            }

            for i, (x, y, dx, dy) in enumerate(features):
                row[f"x{i}"] = x
                row[f"y{i}"] = y
                row[f"dx{i}"] = dx
                row[f"dy{i}"] = dy

            data.append(row)

        frame_id += 1

    cap.release()

    pd.DataFrame(data).to_csv(
        output_csv,
        index=False
    )

    print(f"Saved {len(data)} frames to {output_csv.name}")


if __name__ == "__main__":

    video_files = sorted(VIDEOS_DIR.glob("*.mp4"))

    if not video_files:
        print(f"No videos found in {VIDEOS_DIR}")

    for video_path in video_files:

        # video_id (stem do arquivo) identifica o vídeo de origem
        # e é usado depois para o split treino/teste por vídeo
        output_csv = OUTPUT_DIR / f"{video_path.stem}.csv"

        print(f"Processing {video_path.name}")

        extract_video(video_path, output_csv)
