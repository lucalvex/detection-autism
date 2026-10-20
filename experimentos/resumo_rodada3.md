# Resumo da rodada 3

Protocolo: `protocolo_rodada3.md`. Cada célula: agregado / média ± desvio entre partições (validação). Decisão pelo agregado: mantém se melhora ≥ 0,01 em pelo menos 2 das 3 métricas.

| variante | mudança | F1 evento IoU 0,3 | acurácia bal. | macro F1 quadros | passa | tempo |
|---|---|---|---|---|---|---|
| r2_C4_aumento | referência (rodada 2) | 0,040 / 0,049 ± 0,031 | 0,668 / 0,676 ± 0,021 | 0,414 / 0,432 ± 0,090 | -- | -- |
| r3_I1_interpolacao_limitada | interpolação até 0,5 s, ausentes = 0 + marca, sem ombro/quadril = sem pessoa | 0,052 / 0,054 ± 0,025 | 0,651 / 0,667 ± 0,063 | 0,411 / 0,430 ± 0,050 | não | 14 min |
| r3_I2_limitada_confianca | I1 + confiança de cada ponto | 0,047 / 0,043 ± 0,019 | 0,665 / 0,677 ± 0,041 | 0,440 / 0,450 ± 0,070 | não | 15 min |

**Decisão:** em andamento

Quadros removidos na I1 (ombro ou quadril ausente): 41761 de 113671 (0,367); *ArmFlapping* 0,303; *HeadBanging* 0,504; *Spinning* 0,365. Tabela por ponto: `r3_tabela_pontos/tabela_pontos.md`.
