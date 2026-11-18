# Resumo da rodada 4

Protocolo: `protocolo-rodada4.md`. Validação nos 76542 quadros válidos por Q em que todos os modelos têm predição (agregado / média ± desvio entre partições). Dependência do movimento: macro F1 nos anotados com janelas reais menos com janelas congeladas.

| etapa | mudança | F1 evento | acurácia bal. | macro F1 quadros | dependência (AF / HB / SP) | situação | tempo |
|---|---|---|---|---|---|---|---|
| `r2-C4-aumento` | referência | 0,047 / 0,054 ± 0,029 | 0,667 / 0,681 ± 0,034 | 0,421 / 0,443 ± 0,099 | 0,037 (-0,028 / 0,012 / 0,128) | referência | -- |
| r4-P1a-trajetoria | P1a: trajetória centralizada, sem postura | 0,018 / 0,012 ± 0,005 | 0,607 / 0,623 ± 0,031 | 0,360 / 0,366 ± 0,049 | 0,233 (-0,040 / 0,089 / 0,650) | não | 22 min |
| r4-P1b-trajetoria-postura | P1b: trajetória centralizada + postura média após o LSTM | 0,031 / 0,035 ± 0,022 | 0,646 / 0,652 ± 0,020 | 0,387 / 0,394 ± 0,056 | 0,254 (-0,041 / 0,083 / 0,721) | não | 22 min |
| r4-P2-periodicidade | P2: + periodicidade após o LSTM | 0,025 / 0,025 ± 0,015 | 0,642 / 0,648 ± 0,035 | 0,445 / 0,455 ± 0,088 | 0,104 (0,365 / -0,021 / -0,031) | não | 21 min |
| r4-P3-aumento-postura | P3: + rotação ±30°, escala x/y ±20%, ruído | 0,039 / 0,056 ± 0,039 | 0,666 / 0,678 ± 0,028 | 0,410 / 0,433 ± 0,092 | 0,108 (-0,004 / 0,028 / 0,301) | não | 19 min |

Passo 0 (diagnóstico do `r2-C4-aumento`): `r4-passo0/diagnostico.md`.

Nenhuma variante foi mantida e nenhuma alternativa apareceu: o modelo final continua `r2-C4-aumento`. Dependência do movimento no teste (agregado): TCC 0,238, `r2-C4-aumento` 0,041.
