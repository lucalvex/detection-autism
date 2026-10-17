# Passo 0: pontos ausentes na pose (YOLO11n-Pose, 56 vídeos)

Gerado por `experimentos/rodar_tudo.py` a partir da pose refeita com o mesmo modelo e a mesma regra de `extract_pose.py`.
Conferência com `data/poses/*.csv`: 56 de 56 vídeos com os mesmos quadros e os mesmos valores normalizados; maior diferença em qualquer vídeo: 0.

## 1. O que `keypoints.xy` devolve
Na Ultralytics instalada (8.4.66 no `.venv`; 8.4.60 no Python do sistema, com a mesma classe `Keypoints`), `Keypoints.xy` é só `self.data[..., :2]` (`ultralytics/engine/results.py`). Não há máscara por confiança: um ponto pouco confiável sai com a coordenada estimada, nunca zerado. A única alteração é `clip_coords` (`ultralytics/utils/ops.py`), chamada por `scale_coords` no pós-processamento da pose, que prende x em [0, largura] e y em [0, altura].
Nos dados: 0,000 dos quadros têm algum ponto em (0, 0).

## 2. Pontos com confiança baixa
Quadros com pessoa: 113671. Com algum ponto de confiança < 0,5: 0,961.
Fração de quadros com confiança < 0,5, por ponto (0 a 16): 0: 0,229; 1: 0,293; 2: 0,345; 3: 0,347; 4: 0,520; 5: 0,094; 6: 0,090; 7: 0,296; 8: 0,297; 9: 0,302; 10: 0,294; 11: 0,416; 12: 0,412; 13: 0,654; 14: 0,654; 15: 0,772; 16: 0,771.

## 3. Pontos da normalização (5, 6, 11, 12)
Quadros com algum deles com confiança < 0,5: 0,454.
`normalize_frame_keypoints` não olha a confiança: centraliza no meio dos quadris e divide pela distância entre ombros, e a única proteção é trocar a escala por 1 quando ela fica abaixo de 1 pixel (não há divisão por zero, mas um ombro mal estimado encolhe a escala). Distância entre ombros abaixo de 1 px: 74 quadros; percentis 1 e 50: 4,640 e 88,851 px.

## 4. Valores normalizados acima de 5 larguras de ombro
Fração de quadros: 0,108. Desses, com algum ponto da normalização ausente: 0,358.

| classe | quadros | algum de 5, 6, 11, 12 com conf. < 0,5 | algum valor > 5 |
|---|---|---|---|
| *ArmFlapping* | 52345 | 0,398 | 0,062 |
| *HeadBanging* | 25339 | 0,617 | 0,113 |
| *Spinning* | 35987 | 0,422 | 0,172 |
| total | 113671 | 0,454 | 0,108 |

## 5. Picos nos vídeos de *spinning*
Quadros anotados como giro: 0,469 dos quadros desses vídeos; picos dentro dos intervalos de giro: 0,428 dos picos. Taxa de picos dentro dos intervalos: 0,157; fora: 0,186.

## Decisão
Problema confirmado pelo critério do protocolo (≥ 0,05 dos quadros com ponto da normalização ausente e ≥ 0,01 com valor > 5): a etapa 3 roda.
