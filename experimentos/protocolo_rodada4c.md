# Protocolo da rodada 4c: periodicidade corrigida (P2c)

Registrado antes do primeiro treino. Executado por `experimentos/rodar_rodada4c.py`, com resultados em `experimentos/r4c_<id>/`.

Valem as regras de `protocolo_rodada4.md`:
- mesmas partições e mesma validação;
- avaliação nos quadros válidos por Q, com todos os modelos nos mesmos quadros e contagem registrada;
- decisão só na validação e teste só no fim;
- duas convenções (agregado e média ± desvio entre partições);
- critério de 2 em 3 métricas, com ganho ≥ 0,01;
- regra da alternativa;
- dependência do movimento, também por classe.

O modelo base, `data/results/` e as rodadas anteriores não são alterados.

**Referência:** o modelo final atual, `r4b_C1_janela_4s`: normalização robusta, 15 quadros/s, janela de 4 s (60 amostras) e aumento de dados. As sementes 42, 1000 e 2000 já estão treinadas e são lidas, sem retreino.

## P2c: a referência mais atributos de periodicidade corrigidos

Os atributos são calculados em cada janela de entrada, a partir das posições normalizadas, como na análise de frequência refeita (adendo de `protocolo_rodada4b.md`).

**Sinais:**
- punho em relação ao ombro do mesmo lado, esquerdo e direito (x, y);
- cotovelo em relação ao ombro do mesmo lado, esquerdo e direito (x, y);
- nariz em relação ao centro dos ombros (x, y);
- largura aparente dos ombros (distância entre eles).

**Atributos de cada sinal:**
- **amplitude:** o desvio-padrão do sinal filtrado de 1,5 a 5 Hz, com Butterworth passa-faixa de ordem 4 e filtragem sem atraso de fase (`filtfilt`), na janela. Com dois componentes, a raiz da soma das variâncias.
- **frequência dominante:** o pico da potência entre 1,5 e 7 Hz. Antes, remove-se a tendência linear de cada componente na janela e aplica-se a janela de Hann; o espectro é completado com zeros até 512 pontos, e as potências dos componentes se somam.
- **potência na frequência dominante:** entra em escala logarítmica, log(1 + p), porque varia muitas ordens de grandeza.

**Para *spinning*:** a largura dos ombros também entra na faixa de 0,5 a 2 Hz, com os mesmos três atributos (amplitude filtrada de 0,5 a 2 Hz, frequência dominante e log da potência nessa faixa).

**No modelo:** são 7 × 3 = 21 atributos por janela. Passam por normalização em lote (`BatchNorm1d`) e são concatenados à saída do LSTM antes da camada final, como na P2. Num lote de treino com uma única janela, a normalização usa as estatísticas acumuladas.

**Observação:** o filtro é aplicado em cada janela de 4 s, porque o modelo recebe uma janela de cada vez. Na faixa de 0,5 a 2 Hz, 4 s cobrem só de 2 a 8 ciclos, e os efeitos de borda do filtro pesam mais.

## Decisão e avaliação

A P2c é comparada com a `r4b_C1_janela_4s` na validação, nos mesmos quadros válidos por Q, pelo critério; vale também a regra da alternativa.

Se passar, ou se for alternativa, recebe a avaliação final completa da rodada 4b, ao lado da `r4b_C1_janela_4s` (3 sementes) e do TCC:
- 3 sementes;
- quatro métricas nas duas convenções, com e sem o pós-processamento escolhido na validação;
- por classe, com destaque para *arm flapping*, e matriz de confusão;
- dependência do movimento por classe;
- ablações e fração da classe do vídeo;
- *head banging* com F1 zero;
- `curva_erro.pdf`.

**Resumo:** `resumo_rodada4c.md`, no máximo uma página. Commits por etapa, sem push, com o índice limpo em caso de falha.
