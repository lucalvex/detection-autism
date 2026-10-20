# Resumo da rodada 3

Protocolo: `protocolo_rodada3.md`. Cada célula: agregado / média ± desvio entre partições (validação). Decisão pelo agregado: mantém se melhora ≥ 0,01 em pelo menos 2 das 3 métricas.

| variante | mudança | F1 evento IoU 0,3 | acurácia bal. | macro F1 quadros | passa | tempo |
|---|---|---|---|---|---|---|
| r2_C4_aumento | referência (rodada 2) | 0,040 / 0,049 ± 0,031 | 0,668 / 0,676 ± 0,021 | 0,414 / 0,432 ± 0,090 | -- | -- |
| r3_I1_interpolacao_limitada | interpolação até 0,5 s, ausentes = 0 + marca, sem ombro/quadril = sem pessoa | 0,052 / 0,054 ± 0,025 | 0,651 / 0,667 ± 0,063 | 0,411 / 0,430 ± 0,050 | não | 14 min |
| r3_I2_limitada_confianca | I1 + confiança de cada ponto | 0,047 / 0,043 ± 0,019 | 0,665 / 0,677 ± 0,041 | 0,440 / 0,450 ± 0,070 | não | 15 min |
| r3_I3_treino_a | I3-treino, critério (a): descarta quadro sem os dois ombros e os dois quadris confiáveis só no treino | 0,050 / 0,046 ± 0,014 | 0,666 / 0,672 ± 0,038 | 0,444 / 0,426 ± 0,027 | não | 10 min |
| r3_I3_treino_b | I3-treino, critério (b): descarta quadro sem um ombro e um quadril confiáveis só no treino | 0,051 / 0,053 ± 0,027 | 0,680 / 0,675 ± 0,048 | 0,463 / 0,434 ± 0,050 | sim | 11 min |
| r3_I3_teste_a | I3-teste, critério (a): descarta quadro sem os dois ombros e os dois quadris confiáveis no treino, na validação e no teste (não concorre) | 0,065 / 0,075 ± 0,030 | 0,675 / 0,683 ± 0,069 | 0,473 / 0,446 ± 0,033 | não concorre | 10 min |
| r3_I3_teste_b | I3-teste, critério (b): descarta quadro sem um ombro e um quadril confiáveis no treino, na validação e no teste (não concorre) | 0,041 / 0,048 ± 0,019 | 0,687 / 0,697 ± 0,070 | 0,438 / 0,423 ± 0,023 | não concorre | 11 min |

**Decisão:** em andamento

Quadros removidos na I1 (ombro ou quadril ausente): 41761 de 113671 (0,367); *ArmFlapping* 0,303; *HeadBanging* 0,504; *Spinning* 0,365. Tabela por ponto: `r3_tabela_pontos/tabela_por_ponto.md`.

Descarte da I3 (fração dos quadros com pessoa / dos quadros anotados; treino = vídeos de treino das 5 partições, teste = os 56 vídeos):
- (a) treino: *ArmFlapping* 0,405 / 0,433, *HeadBanging* 0,621 / 0,533, *Spinning* 0,445 / 0,413; teste: *ArmFlapping* 0,398 / 0,434, *HeadBanging* 0,617 / 0,548, *Spinning* 0,422 / 0,406
- (b) treino: *ArmFlapping* 0,366 / 0,389, *HeadBanging* 0,568 / 0,480, *Spinning* 0,409 / 0,374; teste: *ArmFlapping* 0,356 / 0,385, *HeadBanging* 0,560 / 0,488, *Spinning* 0,388 / 0,369
Detalhes: `r3_I3_descarte/descarte.md`.
