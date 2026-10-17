# Resumo dos experimentos

Decisão na validação (protocolo em `protocolo.md`): mantém o ajuste que supera o melhor modelo em pelo menos 2 das 3 métricas, com ganho ≥ 0,01.

| etapa | mudança | F1 evento IoU 0,3 | acurácia bal. presença | macro F1 quadros | mantida | tempo |
|---|---|---|---|---|---|---|
| 01_ponto_de_partida | configuração atual do modelo com rótulos por intervalo | 0,013 | 0,598 | 0,347 | ponto de partida | 30 min |
| 02_yolo11m | pose com YOLO11m-Pose | 0,012 | 0,596 | 0,347 | não | 30 min |
| 03_normalizacao_robusta | escala pelo tronco, confiança < 0,5 interpolada, limite ±4 | 0,027 | 0,624 | 0,328 | sim | 29 min |
| 04_interpolacao_suavizacao | lacunas até 0,5 s interpoladas, média móvel de 0,2 s | 0,044 | 0,606 | 0,336 | não | 30 min |
| 05_15fps_janela_1s | 15 quadros/s, janela de 1 s | 0,013 | 0,620 | 0,349 | não | 10 min |
| 06_15fps_janela_2s | 15 quadros/s, janela de 2 s | 0,035 | 0,656 | 0,389 | sim | 18 min |
| 07_sem_sobreposicao | janelas sem sobreposição no treino | 0,040 | 0,601 | 0,327 | não | 1 min |
| 08a_lr_0.0003 | taxa de aprendizado 0,0003 | 0,041 | 0,633 | 0,346 | não | 18 min |
| 08b_lr_0.003 | taxa de aprendizado 0,003 | 0,024 | 0,634 | 0,343 | não | 18 min |

## Modelo final

Configuração: `{"pose": "yolo11n", "normalizacao": "robusta", "interp_lacuna_s": 0.0, "suavizacao_s": 0.0, "fps_alvo": 15, "janela": 30, "passo_treino": 1, "lr": 0.001, "lote": 32, "oculto": 128, "camadas": 2, "max_epocas": 500, "paciencia": 20, "rotulos": "intervalos_4_classes", "subamostrar_fundo": true, "semente": 42}`

| teste (agregado, 56 vídeos) | F1 evento IoU 0,3 | acurácia bal. presença | macro F1 quadros | macro F1 quadros anotados |
|---|---|---|---|---|
| TCC, rótulos por intervalo | 0,014 | 0,569 | 0,386 | 0,322 |
| ponto de partida | 0,023 | 0,555 | 0,391 | 0,396 |
| modelo final (06_15fps_janela_2s) | 0,028 | 0,595 | 0,402 | 0,404 |

Estabilidade entre as sementes (média ± desvio de 3 execuções):

- validacao: f1_evento_iou_0.3 0,033 ± 0,002; acuracia_balanceada_presenca 0,639 ± 0,013; macro_f1_quadro_todos 0,364 ± 0,018
- teste: f1_evento_iou_0.3 0,032 ± 0,003; acuracia_balanceada_presenca 0,585 ± 0,007; macro_f1_quadro_todos 0,391 ± 0,010

Só janelas anotadas, 3 classes (teste): macro F1 por quadro nos quadros anotados 0,554 (média entre partições) e 0,563 (agregado), contra 0,553 no TCC (média entre partições, predição restrita às 3 classes).
