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
| r2_C4_aumento | aumento de dados | 0,040 / 0,049 ± 0,031 | 0,668 / 0,676 ± 0,021 | 0,414 / 0,432 ± 0,090 | sim | 26 min |

## Modelo final: r2_C4_aumento

Configuração: `{"modelo": "lstm", "entrada": "xy_dxdy", "dropout": 0.0, "peso_l2": 0.0, "aumento": true, "pose": "yolo11n", "normalizacao": "robusta", "interp_lacuna_s": 0.0, "suavizacao_s": 0.0, "fps_alvo": 15, "janela": 30, "passo_treino": 1, "lr": 0.001, "lote": 32, "oculto": 128, "camadas": 2, "max_epocas": 500, "paciencia": 20, "rotulos": "intervalos_4_classes", "subamostrar_fundo": true, "semente": 42}`

Teste (56 vídeos), agregado / média ± desvio entre partições:

| | F1 evento IoU 0,3 | acurácia bal. | macro F1 quadros | macro F1 anotados |
|---|---|---|---|---|
| TCC, rótulos por intervalo | 0,014 / 0,014 ± 0,006 | 0,569 / 0,579 ± 0,034 | 0,386 / 0,379 ± 0,057 | 0,322 / 0,336 ± 0,098 |
| R (rodada 1) | 0,028 / 0,026 ± 0,015 | 0,595 / 0,577 ± 0,026 | 0,402 / 0,368 ± 0,063 | 0,404 / 0,352 ± 0,128 |
| modelo final, semente 42 | 0,025 / 0,027 ± 0,010 | 0,619 / 0,613 ± 0,040 | 0,449 / 0,416 ± 0,068 | 0,454 / 0,403 ± 0,103 |

Três sementes (média ± desvio entre sementes; agregado | média entre partições): f1_evento_iou_0.3 0,037 ± 0,009 | 0,041 ± 0,010; acuracia_balanceada_presenca 0,594 ± 0,018 | 0,596 ± 0,012; macro_f1_quadro_todos 0,440 ± 0,006 | 0,411 ± 0,003; macro_f1_quadro_anotados 0,462 ± 0,009 | 0,424 ± 0,015.

Pós-processamento escolhido: mediana 2,0 s, junção 1,0 s; F1 por evento no teste (semente 42): 0,158 / 0,158 ± 0,050.
Ablações, macro F1 nos quadros anotados com 3 classes (agregado / média entre partições), sementes 42, 1000 e 2000: sem_ablacao: 0,631 / 0,616, 0,642 / 0,610, 0,646 / 0,625; ordem: 0,623 / 0,608, 0,620 / 0,589, 0,631 / 0,616; dxdy: 0,634 / 0,601, 0,631 / 0,589, 0,629 / 0,605.
Fração da classe do vídeo, dentro / fora dos episódios (semente 42, sem ablação): *ArmFlapping* 0,143 / 0,128; *HeadBanging* 0,268 / 0,114; *Spinning* 0,512 / 0,190.
*Head banging* com F1 zero: 7 de 17, 7 de 17, 9 de 17 (sementes 42, 1000, 2000). Por classe, matriz de confusão e curva de erro: `r2_avaliacao_final/avaliacao_final.json` e `r2_C4_aumento/curva_erro.pdf`.
