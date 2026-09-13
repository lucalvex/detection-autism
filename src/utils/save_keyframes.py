# ============================================================
# ARQUIVO: save_keyframes.py
#
# O QUE FAZ: para cada uma das 3 classes de comportamento (Arm
# Flapping, Head Banging, Spinning), pega um vídeo de exemplo e salva,
# a cada N frames (SAVE_EVERY), uma imagem PNG com APENAS o esqueleto
# (keypoints + conexões) da YOLO11n-Pose desenhado sobre um fundo
# preto. Por privacidade, a pessoa do vídeo original nunca é desenhada
# -- só os pontos e linhas do esqueleto. Serve para ilustrar cada
# comportamento no TCC. Não alimenta o treino do modelo.
# ============================================================

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

VIDEOS_DIR = Path("data/videos")
OUTPUT_DIR = Path("data/results/keyframes")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAVE_EVERY = 100  # salva 1 imagem a cada 100 frames processados

CLASSES = ["ArmFlapping", "HeadBanging", "Spinning"]

# conexões do esqueleto COCO (17 keypoints, índices 0-based)
SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
    (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6),
]

CONF_THRESHOLD = 0.5

model = YOLO("models/yolo11n-pose.pt")


def draw_skeleton(frame_shape, keypoints_xy, keypoints_conf):
    """Desenha só os pontos + conexões do esqueleto sobre um fundo
    preto do mesmo tamanho do frame original -- o frame de vídeo em
    si (com a pessoa) nunca é usado na imagem salva."""

    canvas = np.zeros(frame_shape, dtype=np.uint8)

    visible = keypoints_conf >= CONF_THRESHOLD

    for i, j in SKELETON:
        if visible[i] and visible[j]:
            pt1 = tuple(keypoints_xy[i].astype(int))
            pt2 = tuple(keypoints_xy[j].astype(int))
            cv2.line(canvas, pt1, pt2, (0, 255, 0), 2)

    for i, (x, y) in enumerate(keypoints_xy):
        if visible[i]:
            cv2.circle(canvas, (int(x), int(y)), 4, (0, 0, 255), -1)

    return canvas


def save_class_keyframes(class_name: str):

    video_files = sorted(VIDEOS_DIR.glob(f"v_{class_name}_*.mp4"))

    if not video_files:
        print(f"No videos found for class {class_name}")
        return

    # usa só o primeiro vídeo da classe como exemplo representativo
    video_path = video_files[0]

    class_dir = OUTPUT_DIR / class_name
    class_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    frame_id = 0

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        # só roda a detecção quando o frame realmente vai ser salvo
        if frame_id % SAVE_EVERY == 0:

            results = model(frame, verbose=False)
            result = results[0]

            if result.keypoints is not None and len(result.keypoints.xy) > 0:

                # assume 1 pessoa por vídeo: usa só a primeira detectada
                keypoints_xy = result.keypoints.xy[0].cpu().numpy()
                keypoints_conf = result.keypoints.conf[0].cpu().numpy()

                skeleton_frame = draw_skeleton(frame.shape, keypoints_xy, keypoints_conf)

                filename = class_dir / f"frame_{frame_id:05d}.png"

                cv2.imwrite(str(filename), skeleton_frame)

                print(f"Saved {filename}")

        frame_id += 1

    cap.release()

    print(f"Finished {class_name} ({video_path.name})")


if __name__ == "__main__":

    for class_name in CLASSES:
        save_class_keyframes(class_name)

    print("Finished")
