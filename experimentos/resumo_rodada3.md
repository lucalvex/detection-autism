# Resumo da rodada 3

**A primeira avaliação foi substituída.** A regra da I1 tirava da validação e do teste os quadros sem ombro ou quadril recuperável (cerca de 44% dos quadros), e I1, I2 e I3-treino eram comparadas ao `r2_C4_aumento` em conjuntos de quadros diferentes. A decisão agora vale pela cobertura total: os mesmos modelos, sem novo treino, avaliados exatamente nos mesmos quadros do `r2_C4_aumento` (adendo de cobertura total em `protocolo_rodada3.md`).

## Cobertura total (validação; agregado / média ± desvio entre partições)

| variante | quadros de validação | F1 evento IoU 0,3 | acurácia bal. | macro F1 quadros | passa |
|---|---|---|---|---|---|
| r2_C4_aumento (referência) | 88717 | 0,040 / 0,049 ± 0,031 | 0,668 / 0,676 ± 0,021 | 0,414 / 0,432 ± 0,090 | -- |
| r3_I1_interpolacao_limitada | 88717 | 0,033 / 0,034 ± 0,011 | 0,631 / 0,635 ± 0,043 | 0,356 / 0,381 ± 0,071 | não |
| r3_I2_limitada_confianca | 88717 | 0,037 / 0,036 ± 0,012 | 0,628 / 0,633 ± 0,042 | 0,367 / 0,395 ± 0,089 | não |
| r3_I3_treino_a | 88717 | 0,028 / 0,029 ± 0,012 | 0,629 / 0,641 ± 0,020 | 0,376 / 0,383 ± 0,057 | não |
| r3_I3_treino_b | 88717 | 0,031 / 0,036 ± 0,023 | 0,642 / 0,642 ± 0,034 | 0,394 / 0,389 ± 0,068 | não |

**Decisão:** nenhuma variante passa no critério com cobertura total; o modelo final continua `r2_C4_aumento`.

Só como informação, no subconjunto de quadros que a I1 mantém (validação, agregado): r3_I1_interpolacao_limitada: 0,052 contra 0,067, 0,651 contra 0,673, 0,411 contra 0,472; r3_I2_limitada_confianca: 0,047 contra 0,067, 0,665 contra 0,673, 0,440 contra 0,472; r3_I3_treino_a: 0,050 contra 0,067, 0,666 contra 0,673, 0,444 contra 0,472; r3_I3_treino_b: 0,051 contra 0,067, 0,680 contra 0,673, 0,463 contra 0,472 (variante contra `r2_C4_aumento` nos mesmos 52963 quadros; métricas na ordem F1 evento, acurácia bal., macro F1 quadros).

Quadros removidos na I1 (ombro ou quadril ausente): 41761 de 113671 (0,367); *ArmFlapping* 0,303; *HeadBanging* 0,504; *Spinning* 0,365. Tabela por ponto: `r3_tabela_pontos/tabela_por_ponto.md`.
Descarte da I3 (fração dos quadros com pessoa / dos quadros anotados; treino = vídeos de treino das 5 partições, teste = os 56 vídeos):
- (a) treino: *ArmFlapping* 0,405 / 0,433, *HeadBanging* 0,621 / 0,533, *Spinning* 0,445 / 0,413; teste: *ArmFlapping* 0,398 / 0,434, *HeadBanging* 0,617 / 0,548, *Spinning* 0,422 / 0,406
- (b) treino: *ArmFlapping* 0,366 / 0,389, *HeadBanging* 0,568 / 0,480, *Spinning* 0,409 / 0,374; teste: *ArmFlapping* 0,356 / 0,385, *HeadBanging* 0,560 / 0,488, *Spinning* 0,388 / 0,369
Detalhes: `r3_I3_descarte/descarte.md`.

## Primeira avaliação (substituída: subconjunto de quadros)

| variante | mudança | F1 evento IoU 0,3 | acurácia bal. | macro F1 quadros | passa | tempo |
|---|---|---|---|---|---|---|
| r2_C4_aumento | referência (rodada 2) | 0,040 / 0,049 ± 0,031 | 0,668 / 0,676 ± 0,021 | 0,414 / 0,432 ± 0,090 | -- | -- |
| r3_I1_interpolacao_limitada | interpolação até 0,5 s, ausentes = 0 + marca, sem ombro/quadril = sem pessoa | 0,052 / 0,054 ± 0,025 | 0,651 / 0,667 ± 0,063 | 0,411 / 0,430 ± 0,050 | não | 14 min |
| r3_I2_limitada_confianca | I1 + confiança de cada ponto | 0,047 / 0,043 ± 0,019 | 0,665 / 0,677 ± 0,041 | 0,440 / 0,450 ± 0,070 | não | 15 min |
| r3_I3_treino_a | I3-treino, critério (a): descarta quadro sem os dois ombros e os dois quadris confiáveis só no treino | 0,050 / 0,046 ± 0,014 | 0,666 / 0,672 ± 0,038 | 0,444 / 0,426 ± 0,027 | não | 10 min |
| r3_I3_treino_b | I3-treino, critério (b): descarta quadro sem um ombro e um quadril confiáveis só no treino | 0,051 / 0,053 ± 0,027 | 0,680 / 0,675 ± 0,048 | 0,463 / 0,434 ± 0,050 | sim | 11 min |
| r3_I3_teste_a | I3-teste, critério (a): descarta quadro sem os dois ombros e os dois quadris confiáveis no treino, na validação e no teste (não concorre) | 0,065 / 0,075 ± 0,030 | 0,675 / 0,683 ± 0,069 | 0,473 / 0,446 ± 0,033 | não concorre | 10 min |
| r3_I3_teste_b | I3-teste, critério (b): descarta quadro sem um ombro e um quadril confiáveis no treino, na validação e no teste (não concorre) | 0,041 / 0,048 ± 0,019 | 0,687 / 0,697 ± 0,070 | 0,438 / 0,423 ± 0,023 | não concorre | 11 min |

Detalhes da primeira avaliação: `r3_avaliacao_final/` (não usar).
