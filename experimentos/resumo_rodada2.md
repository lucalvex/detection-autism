# Resumo da rodada 2

Protocolo: `protocolo_rodada2.md`. Cada célula: agregado / média ± desvio entre partições (validação). Decisão pelo agregado: mantém se melhora ≥ 0,01 em pelo menos 2 das 3 métricas.

| etapa | mudança | F1 evento IoU 0,3 | acurácia bal. | macro F1 quadros | mantida | tempo |
|---|---|---|---|---|---|---|
| R | referência: 06_15fps_janela_2s (rodada 1) | 0,035 / 0,040 ± 0,024 | 0,656 / 0,653 ± 0,026 | 0,389 / 0,392 ± 0,083 | referência | 18 min |
| r2_A1_dxdy | só dx, dy | 0,016 / 0,008 ± 0,006 | 0,638 / 0,648 ± 0,011 | 0,344 / 0,359 ± 0,073 | não | 19 min |
| r2_A2_dxdy_angulos | dx, dy + velocidade e aceleração angulares | 0,005 / 0,008 ± 0,004 | 0,611 / 0,617 ± 0,042 | 0,336 / 0,360 ± 0,093 | não | 20 min |
