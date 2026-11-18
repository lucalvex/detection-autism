# Resumo da rodada 3

Protocolo: `protocolo-rodada3.md`. Cada célula: agregado / média ± desvio entre partições (validação). Decisão pelo agregado: mantém se melhora ≥ 0,01 em pelo menos 2 das 3 métricas.

| variante | mudança | F1 evento IoU 0,3 | acurácia bal. | macro F1 quadros | passa | tempo |
|---|---|---|---|---|---|---|
| r2-C4-aumento | referência (rodada 2) | 0,040 / 0,049 ± 0,031 | 0,668 / 0,676 ± 0,021 | 0,414 / 0,432 ± 0,090 | -- | -- |
| r3-I1-interpolacao-limitada | interpolação até 0,5 s, ausentes = 0 + marca, sem ombro/quadril = sem pessoa | 0,052 / 0,054 ± 0,025 | 0,651 / 0,667 ± 0,063 | 0,411 / 0,430 ± 0,050 | não | 14 min |
| r3-I2-limitada-confianca | I1 + confiança de cada ponto | 0,047 / 0,043 ± 0,019 | 0,665 / 0,677 ± 0,041 | 0,440 / 0,450 ± 0,070 | não | 15 min |
| r3-I3-treino-a | I3-treino, critério (a): descarta quadro sem os dois ombros e os dois quadris confiáveis só no treino | 0,050 / 0,046 ± 0,014 | 0,666 / 0,672 ± 0,038 | 0,444 / 0,426 ± 0,027 | não | 10 min |
| r3-I3-treino-b | I3-treino, critério (b): descarta quadro sem um ombro e um quadril confiáveis só no treino | 0,051 / 0,053 ± 0,027 | 0,680 / 0,675 ± 0,048 | 0,463 / 0,434 ± 0,050 | sim | 11 min |
| r3-I3-teste-a | I3-teste, critério (a): descarta quadro sem os dois ombros e os dois quadris confiáveis no treino, na validação e no teste (não concorre) | 0,065 / 0,075 ± 0,030 | 0,675 / 0,683 ± 0,069 | 0,473 / 0,446 ± 0,033 | não concorre | 10 min |
| r3-I3-teste-b | I3-teste, critério (b): descarta quadro sem um ombro e um quadril confiáveis no treino, na validação e no teste (não concorre) | 0,041 / 0,048 ± 0,019 | 0,687 / 0,697 ± 0,070 | 0,438 / 0,423 ± 0,023 | não concorre | 11 min |

**Decisão:** r3-I3-treino-b melhora r2-C4-aumento pelo critério e passa a ser o modelo final.

Quadros removidos na I1 (ombro ou quadril ausente): 41761 de 113671 (0,367); *ArmFlapping* 0,303; *HeadBanging* 0,504; *Spinning* 0,365. Tabela por ponto: `r3-tabela-pontos/tabela-por-ponto.md`.

Descarte da I3 (fração dos quadros com pessoa / dos quadros anotados; treino = vídeos de treino das 5 partições, teste = os 56 vídeos):
- (a) treino: *ArmFlapping* 0,405 / 0,433, *HeadBanging* 0,621 / 0,533, *Spinning* 0,445 / 0,413; teste: *ArmFlapping* 0,398 / 0,434, *HeadBanging* 0,617 / 0,548, *Spinning* 0,422 / 0,406
- (b) treino: *ArmFlapping* 0,366 / 0,389, *HeadBanging* 0,568 / 0,480, *Spinning* 0,409 / 0,374; teste: *ArmFlapping* 0,356 / 0,385, *HeadBanging* 0,560 / 0,488, *Spinning* 0,388 / 0,369
Detalhes: `r3-I3-descarte/descarte.md`.

## Novo modelo final: r3-I3-treino-b

Teste (56 vídeos), agregado / média ± desvio entre partições; as métricas por quadro não dependem do pós-processamento.

| | F1 evento | acurácia bal. | macro F1 quadros | macro F1 anotados |
|---|---|---|---|---|
| r2-C4-aumento, semente 42 | 0,025 / 0,027 ± 0,010 | 0,619 / 0,613 ± 0,040 | 0,449 / 0,416 ± 0,068 | 0,454 / 0,403 ± 0,103 |
| r2-C4-aumento, 3 sementes (agregado) | 0,037 ± 0,009 | 0,594 ± 0,018 | 0,440 ± 0,006 | 0,462 ± 0,009 |
| r3-I3-treino-b, semente 42 | 0,054 / 0,054 ± 0,016 | 0,599 / 0,597 ± 0,030 | 0,470 / 0,434 ± 0,051 | 0,490 / 0,448 ± 0,111 |
| r3-I3-treino-b, 3 sementes (agregado) | 0,057 ± 0,005 | 0,611 ± 0,011 | 0,476 ± 0,028 | 0,488 ± 0,077 |

F1 por evento com o pós-processamento escolhido (mediana_2.0_juncao_1.0), agregado (média ± desvio entre sementes) | média entre partições:
- r3-I3-treino-b: sem 0,057 ± 0,005, com 0,206 ± 0,030 | com 0,190
- r2-C4-aumento: sem 0,037 ± 0,009, com 0,188 ± 0,022 | com 0,181
- TCC: sem 0,014 ± 0,000, com 0,137 ± 0,000 | com 0,142
Ablações, macro F1 anotados com 3 classes (agregado / média entre partições), por semente: sem_ablacao: 0,667 / 0,623, 0,688 / 0,617, 0,732 / 0,638; ordem: 0,669 / 0,629, 0,681 / 0,614, 0,727 / 0,634; dxdy: 0,665 / 0,619, 0,682 / 0,607, 0,729 / 0,635.
Fração da classe do vídeo, dentro / fora (semente 42): *ArmFlapping* 0,179 / 0,132; *HeadBanging* 0,173 / 0,063; *Spinning* 0,621 / 0,288.
*Head banging* com F1 zero: 6 de 14, 6 de 14, 7 de 14. Por classe e matriz de confusão: `r3-avaliacao-final/avaliacao-final.json`; curva de erro: `r3-I3-treino-b/curva-erro.pdf`.
