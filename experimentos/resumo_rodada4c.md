# Resumo da rodada 4c

Protocolo: `protocolo_rodada4c.md`. Referência: `r4b_C1_janela_4s` (modelo final atual). Validação nos quadros válidos por Q.

| variante | quadros | F1 evento | acurácia bal. | macro F1 quadros | dependência (AF / HB / SP) | situação | tempo |
|---|---|---|---|---|---|---|---|
| P2c: C1 + periodicidade corrigida (1,5-5 Hz; 1,5-7 Hz; giro 0,5-2 Hz) após o LSTM | 73277 | 0,020 vs 0,060 (média 0,030 vs 0,090) | 0,681 vs 0,659 (média 0,664 vs 0,659) | 0,435 vs 0,448 (média 0,428 vs 0,458) | 0,146 vs 0,086 (0,233 / 0,009 / 0,195 vs 0,087 / 0,034 / 0,135) | não | 46 min |

Cada célula: variante vs referência nos mesmos quadros (agregado; média entre partições).

A P2c não passou no critério nem se qualificou como alternativa; o modelo final continua `r4b_C1_janela_4s`.
