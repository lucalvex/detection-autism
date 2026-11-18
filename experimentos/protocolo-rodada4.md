# Protocolo da rodada 4: reduzir a dependência da postura

Registrado antes do primeiro treino. Executado por `experimentos/rodar_rodada4.py`, com resultados em `experimentos/r4_<id>/`.

O modelo base, `data/results/` e as rodadas anteriores não são alterados; só se leem arquivos delas.

## Regras

- Mesmas partições (`models/oof/fold-assignments.json`) e mesmos vídeos de validação (`divisao-validacao.json`).
- Treino como no `r2-C4-aumento`, com todos os quadros com pessoa; muda só o que cada variante diz.
- **Avaliação sempre nos quadros válidos pelo critério Q:** os dois ombros e pelo menos um ponto da cabeça com confiança ≥ 0,5.
  - Todos os modelos comparados são avaliados exatamente nos mesmos quadros: os válidos por Q em que todos têm predição. A contagem é registrada.
  - Episódios anotados sem nenhum quadro válido saem do gabarito de eventos, como em `protocolo-rodada3.md`.
- Decisão só na validação; teste só no fim.
- Métricas nas duas convenções: agregada sobre os vídeos e média ± desvio entre as 5 partições.
- **Referência:** `r2-C4-aumento`, semente 42, lido da rodada 2 sem retreino.
- **Critério para manter uma variante:** melhorar o melhor modelo até então em pelo menos 2 destas 3 métricas de validação (agregado), com ganho ≥ 0,01 em cada:
  - F1 por evento com IoU 0,3, na regra principal;
  - acurácia balanceada da detecção de presença;
  - macro F1 por quadro em todos os quadros.

  Numa etapa com duas versões, fica a de maior soma de ganhos.
- **Regra da alternativa, registrada antes:** uma variante que não passa no critério é reportada como alternativa se cumprir duas condições:
  - não piora nenhuma das três métricas em mais de 0,01 em relação ao modelo com que foi comparada;
  - tem dependência do movimento pelo menos o dobro da do `r2-C4-aumento`, as duas medidas na mesma validação restrita a Q.

  Uma alternativa não substitui o melhor modelo na sequência.

## Dependência do movimento

- **Medida:** o macro F1 por quadro nos quadros anotados (3 classes, predição com fundo, como a métrica "macro F1 anotados") com as janelas reais, menos o mesmo F1 com as janelas congeladas. Também é reportada a diferença do F1 de cada classe.
- **Janela congelada:** em cada janela de entrada, as posições x e y de cada ponto são trocadas pela média delas na janela, repetida em todas as amostras, e dx e dy viram 0.
- **Aplicação:** a todos os modelos, com a mesma operação sobre as janelas de entrada de cada um. No modelo do TCC são as janelas de 30 linhas de `evaluate_temporal.py`.
- **Onde se mede:** na validação restrita a Q, para as decisões, e no teste restrito a Q, na avaliação final. O modelo do TCC não tem validação separada e entra só no teste.

## Passo 0: diagnóstico, sem treino

Com as predições de teste do `r2-C4-aumento` (média das 3 sementes, restritas a Q), para dois grupos de vídeos:
- os com Q > 90% e F1 < 0,1, pela tabela `qualidade-dataset/tabela-videos.csv`;
- os 3 de maior F1 de cada classe.

Mostra-se a fração dos quadros anotados prevista como cada classe e como fundo, e o mesmo para os quadros fora dos episódios.

## Variantes, acumulativas

Cada uma parte do melhor modelo até então.

**P1, trajetória centralizada na janela.** Em cada janela e para cada ponto, subtrai-se de x e y a média deles na janela. dx e dy não mudam.
- (a) Sem nenhuma informação de postura.
- (b) Com a postura média da janela (34 valores: x e y médios dos 17 pontos) concatenada à saída do LSTM antes da camada final.

**P2, periodicidade junto com o LSTM.** Os 36 atributos de periodicidade da rodada 2 (`protocolo-rodada2.md`, B), calculados de cada janela de entrada (2 s a 15 quadros/s), passam por uma normalização em lote (`BatchNorm1d`) e são concatenados à saída do LSTM antes da camada final.

**P3, aumento de dados de postura e ângulo.** É o aumento atual (espelhamento, velocidade de 0,8 a 1,2) com três mudanças:
- rotação de −30° a 30°, em vez de ±10°;
- escala independente na horizontal e na vertical, cada uma de 0,8 a 1,2, em vez da escala única de 0,9 a 1,1;
- ruído gaussiano independente em cada ponto e amostra, com desvio de 0,02 altura de tronco (dx e dy acompanham o ruído).

## Avaliação final

Feita para o modelo mantido ou, se nenhum for mantido, para a alternativa de maior dependência do movimento. Se não houver nenhum dos dois, o modelo final continua o `r2-C4-aumento`, e a dependência do movimento dele e a do TCC são reportadas no teste.

Sempre ao lado do `r2-C4-aumento` e do modelo do TCC, nos mesmos quadros válidos por Q:
- 3 sementes (42, 1000, 2000);
- as quatro métricas nas duas convenções, com e sem o pós-processamento escolhido na validação restrita, só pelo F1 por evento, entre:
  - mediana de 0; 0,5; 1 ou 2 s;
  - junção abaixo de 0; 0,5 ou 1 s;
- precisão, revocação e F1 por classe, incluindo fundo, e matriz de confusão nos quadros anotados;
- dependência do movimento, também por classe;
- ablações (ordem embaralhada e deslocamentos zerados) e fração da classe do vídeo dentro e fora dos episódios;
- *head banging* com F1 zero, entre os vídeos com quadros válidos;
- `curva-erro.pdf`.

**Commits:** um por etapa, sem push. Se um commit falhar, os arquivos são tirados do índice.
