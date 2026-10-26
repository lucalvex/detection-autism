# Resumo da rodada 4c

Protocolo: `protocolo_rodada4c.md`. Referência: `r4b_C1_janela_4s` (modelo final atual). Validação nos quadros válidos por Q.

| variante | quadros | F1 evento | acurácia bal. | macro F1 quadros | dependência (AF / HB / SP) | situação | tempo |
|---|---|---|---|---|---|---|---|
| P2c: C1 + periodicidade corrigida (1,5-5 Hz; 1,5-7 Hz; giro 0,5-2 Hz) após o LSTM | 73277 | 0,020 vs 0,060 (média 0,030 vs 0,090) | 0,681 vs 0,659 (média 0,664 vs 0,659) | 0,435 vs 0,448 (média 0,428 vs 0,458) | 0,146 vs 0,086 (0,233 / 0,009 / 0,195 vs 0,087 / 0,034 / 0,135) | não | 46 min |

Cada célula: variante vs referência nos mesmos quadros (agregado; média entre partições).

A P2c não passou no critério nem se qualificou como alternativa; o modelo final continua `r4b_C1_janela_4s`.

## Análise posterior, não altera a decisão

F1 por evento com pós-processamento na validação, nos mesmos 73.277 quadros válidos por Q. O pós-processamento foi escolhido em cada modelo pelas opções de sempre (mediana 0; 0,5; 1 ou 2 s × junção 0; 0,5 ou 1 s); nos dois, venceu a mediana de 2 s com junção abaixo de 1 s.

| | sem pós (agregado / média ± desvio) | com pós (agregado / média ± desvio) |
|---|---|---|
| `r4b_C1_janela_4s` | 0,060 / 0,090 ± 0,047 | 0,276 / 0,269 ± 0,079 |
| P2c | 0,020 / 0,030 ± 0,020 | 0,298 / 0,291 ± 0,012 |

Com o pós-processamento, a P2c fica 0,022 acima da C1, e com variação entre partições bem menor. Mas o pós-processamento é escolhido e medido na mesma validação, então esses números são otimistas para os dois modelos. A decisão foi tomada pelo critério registrado, que usa o F1 por evento sem pós-processamento, e não muda. Dados: `r4c_P2c_periodicidade_corrigida/pos_processamento_validacao.json`.
