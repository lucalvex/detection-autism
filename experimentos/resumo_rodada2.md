# Resumo da rodada 2

Protocolo: `protocolo_rodada2.md`. Cada célula: agregado / média ± desvio entre partições (validação). Decisão pelo agregado: mantém se melhora ≥ 0,01 em pelo menos 2 das 3 métricas.

| etapa | mudança | F1 evento IoU 0,3 | acurácia bal. | macro F1 quadros | mantida | tempo |
|---|---|---|---|---|---|---|
| R | referência: 06_15fps_janela_2s (rodada 1) | 0,035 / 0,040 ± 0,024 | 0,656 / 0,653 ± 0,026 | 0,389 / 0,392 ± 0,083 | referência | 18 min |
| r2_A1_dxdy | só dx, dy | 0,016 / 0,008 ± 0,006 | 0,638 / 0,648 ± 0,011 | 0,344 / 0,359 ± 0,073 | não | 19 min |
| r2_A2_dxdy_angulos | dx, dy + velocidade e aceleração angulares | 0,005 / 0,008 ± 0,004 | 0,611 / 0,617 ± 0,042 | 0,336 / 0,360 ± 0,093 | não | 20 min |
| r2_B_2s_logreg | periodicidade, 2 s, logreg | 0,008 / 0,010 ± 0,005 | 0,547 / 0,546 ± 0,033 | 0,336 / 0,326 ± 0,004 | linha de base (não supera o melhor LSTM) | 2 min |
| r2_B_2s_gb | periodicidade, 2 s, gb | 0,009 / 0,012 ± 0,003 | 0,626 / 0,639 ± 0,031 | 0,420 / 0,418 ± 0,057 | linha de base (não supera o melhor LSTM) | 4 min |
| r2_B_4s_logreg | periodicidade, 4 s, logreg | 0,013 / 0,014 ± 0,006 | 0,597 / 0,598 ± 0,049 | 0,376 / 0,364 ± 0,019 | linha de base (não supera o melhor LSTM) | 1 min |
| r2_B_4s_gb | periodicidade, 4 s, gb | 0,017 / 0,023 ± 0,009 | 0,645 / 0,649 ± 0,018 | 0,410 / 0,418 ± 0,062 | linha de base (não supera o melhor LSTM) | 4 min |
| r2_C1_dropout_0.3 | dropout 0,3 | 0,027 / 0,027 ± 0,014 | 0,657 / 0,647 ± 0,033 | 0,353 / 0,362 ± 0,053 | não | 21 min |
| r2_C1_dropout_0.5 | dropout 0,5 | 0,029 / 0,030 ± 0,015 | 0,650 / 0,646 ± 0,010 | 0,362 / 0,376 ± 0,066 | não | 21 min |
| r2_C2_l2_1e-4 | penalização L2 1e-4 | 0,036 / 0,040 ± 0,020 | 0,675 / 0,667 ± 0,030 | 0,372 / 0,390 ± 0,070 | não | 20 min |
| r2_C2_l2_1e-3 | penalização L2 1e-3 | 0,030 / 0,040 ± 0,027 | 0,634 / 0,641 ± 0,030 | 0,376 / 0,370 ± 0,051 | não | 20 min |
| r2_C3_1x64 | 1 camada de 64 unidades | 0,028 / 0,033 ± 0,019 | 0,644 / 0,646 ± 0,047 | 0,352 / 0,358 ± 0,039 | não | 5 min |
