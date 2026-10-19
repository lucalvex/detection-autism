# Adendo ao protocolo: rodada 2

Registrado antes de rodar. Executado por `experimentos/rodar_rodada2.py`; os resultados ficam em `experimentos/r2_<id>/`. Vale tudo de `protocolo.md`:

- mesmas partições e mesmos vídeos de validação (`divisao_validacao.json`);
- decisão só na validação e teste só no fim;
- até 500 épocas, com paciência de 20;
- critério para manter um ajuste: pelo menos 2 das 3 métricas de validação melhores, com ganho ≥ 0,01 em cada, calculadas no agregado dos vídeos de validação, como na rodada 1.

O modelo base, `data/results/` e os experimentos da rodada 1 não são alterados; da rodada 1, só se leem arquivos.

**Duas convenções.** Toda métrica é reportada agregada sobre os vídeos e como média ± desvio entre as 5 partições.

**Referência R.** O modelo final da rodada 1, `06_15fps_janela_2s`: normalização robusta, 15 quadros/s e janela de 2 s (30 amostras). Seus resultados são lidos da rodada 1, sem retreino. Cada etapa parte do melhor modelo até então.

## A. Entrada só com movimento
- **A1:** só os deslocamentos dx e dy dos 17 pontos (34 atributos).
- **A2:** dx e dy mais a velocidade e a aceleração angulares de 7 articulações, sem os ângulos em si.
  - Articulações: cotovelos (5-7-9, 6-8-10), ombros (7-5-11, 8-6-12), joelhos (11-13-15, 12-14-16) e pescoço (ângulo entre nariz menos centro dos ombros e centro dos ombros menos centro dos quadris).
  - Velocidade: diferença do ângulo entre amostras, reduzida a (−π, π]. Aceleração: diferença da velocidade.
  - Ambas são calculadas dentro de cada janela e valem 0 nas duas primeiras amostras, igual no treino e na avaliação.

## B. Periodicidade com classificador simples (linha de base)
**Sinais por janela,** com as posições da normalização robusta a 15 quadros/s:
- x e y de cada punho em relação ao ombro do mesmo lado;
- y do nariz em relação ao centro dos ombros;
- largura aparente dos ombros (distância entre eles).

**Atributos de cada sinal:**
- frequência dominante entre 0,5 e 6 Hz e a potência nela, com o sinal sem média e janela de Hann;
- entropia espectral normalizada nessa faixa;
- maior autocorrelação normalizada nos atrasos de 0,5 a 6 Hz;
- desvio-padrão;
- velocidade média (média de |Δ|).

Nenhuma posição média entra.

**Janelas de 2 s e de 4 s** (30 e 60 amostras), com a mesma regra de rótulo e a mesma avaliação por quadro.

**Classificadores:**
- regressão logística com padronização, C ∈ {0,01; 0,1; 1; 10};
- *gradient boosting* por histograma, taxa ∈ {0,05; 0,1} × folhas ∈ {15; 63}, 300 iterações.

Os dois usam `class_weight="balanced"` e todas as janelas de treino, sem subamostragem. Os hiperparâmetros são escolhidos na validação pela média das 3 métricas.

B é reportado mesmo que não supere o melhor modelo. Só substitui o LSTM como modelo final se superar, pelo critério, o melhor LSTM depois de C.

## C. Sobreajuste, acumulativo sobre o melhor LSTM
- **C1:** dropout de 0,3 e de 0,5, entre as camadas do LSTM e antes da camada final.
- **C2:** penalização L2 dos pesos no Adam (`weight_decay`) de 1e-4 e de 1e-3.
- **C3:** 1 camada de 64 unidades.
- **C4:** aumento de dados, só no treino, sorteado por janela:
  - espelhamento esquerda/direita com probabilidade 0,5 (x → −x e troca dos pares de pontos);
  - escala de 0,9 a 1,1;
  - rotação de −10° a 10°;
  - velocidade de 0,8 a 1,2, reamostrando no tempo em torno do centro da janela, com as bordas presas;
  - dx e dy acompanham a transformação; as posições continuam limitadas a ±4.

Em C1 e C2, se os dois valores passarem, fica o de maior soma de ganhos.

## D. Pós-processamento, sem retreino
Sobre as pontuações do modelo final:
- filtro de mediana de 0,5, 1 ou 2 s, ou nenhum, ao longo das janelas de cada vídeo;
- junção de trechos da mesma classe separados por menos de 0,5 s ou 1 s, ou nenhuma.

O pós-processamento só afeta a construção dos trechos, então as métricas por quadro continuam usando as pontuações sem filtro. A combinação é escolhida na validação só pelo F1 por evento.

## E. Curva de quantidade de dados
Com a configuração final, treino com 25%, 50%, 75% e 100% dos vídeos de treino de cada partição, por classe (mínimo de 1 por classe).
- Os subconjuntos são encaixados: os primeiros n de uma permutação por classe, sorteada com `default_rng(1000·sorteio + partição)`.
- Há 3 sorteios por fração, cada um com a semente de treino 42 + 1000·sorteio.
- A validação fica fixa.
- Figura: `curva_dados.pdf`, com as 3 métricas de validação em função do número médio de vídeos de treino e a faixa de ± 1 desvio entre os sorteios.

## Avaliação final (modelo final, depois de A a D)
- Teste nas duas convenções, comparado com R e com o TCC.
- Três sementes (42, 1000, 2000).
- Nas três sementes:
  - ablações de ordem embaralhada (uma permutação por janela, semente 42) e de deslocamentos zerados, se a entrada tiver dx e dy;
  - fração da classe do vídeo dentro e fora dos episódios.
- Precisão, revocação e F1 por classe, incluindo fundo, e matriz de confusão nos quadros anotados.
- Vídeos de *head banging* com F1 zero.
- `curva_erro.pdf` do modelo final.
