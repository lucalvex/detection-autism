# Protocolo dos experimentos do modelo com rótulos por intervalo

Registrado antes do primeiro treino. Executado por `experimentos/rodar_tudo.py`. O modelo base e os resultados de `data/results/` não são alterados.

## Dados e partições

- Mesmas 5 partições de `models/oof/fold-assignments.json` (56 vídeos).
- Validação: em cada partição, cerca de 20% dos vídeos de treino, por classe do vídeo (`round(0,2 × n)`, no mínimo 1), sorteados com `numpy.random.default_rng(42 + partição)`. A divisão fica em `experimentos/divisao-validacao.json` e é a mesma em todos os experimentos e sementes.
- Rótulo de cada janela, como em `create_sequences_annotated.py`: recebe a classe de um intervalo anotado se pelo menos 2/3 da janela estão dentro dele; recebe fundo se nenhum quadro está anotado; nos outros casos, é descartada.
- No treino, o fundo é subamostrado até igualar as janelas de estereotipia, como em `train_oof_seeded.py`. A perda de validação usa as janelas de validação com a mesma subamostragem e uma semente própria.

## Treino

- LSTM de 2 camadas e 128 unidades, lote 32, Adam, perda de entropia cruzada.
- Até 500 épocas. O treino para quando a perda de validação passa 20 épocas sem melhorar, e fica o modelo da melhor época.
- Perda de treino e de validação gravadas por época e por partição em `perdas.csv`.
- As 5 partições treinam em processos paralelos. Sementes por partição: inicialização, ordem dos lotes e subamostragem usam `semente + partição` (ver `seeds.json` de cada experimento).

## Avaliação

- Sempre na linha do tempo original de cada vídeo, com as funções de `src/evaluate_temporal.py`:
  - um rótulo por quadro, o do fim de cada janela;
  - gabarito com `round(t × fps)`;
  - trechos de `timeline.py`;
  - pareamento guloso por IoU.
- Nos experimentos com taxa de quadros padronizada, a janela de cada quadro original termina no próprio quadro, com passo de 1 quadro.
- Métricas de decisão, calculadas na validação: as predições dos 5 modelos sobre os seus vídeos de validação, somadas.
  1. F1 por evento com IoU 0,3, na regra principal: limiar 0,5, sem duração mínima adicional, macro entre as 3 classes.
  2. Acurácia balanceada da presença de estereotipia (qualquer das 3 classes contra fundo), em todos os quadros.
  3. Macro F1 por quadro em todos os quadros, com as 3 classes e o fundo.
- Um ajuste é mantido se superar o melhor modelo atual em pelo menos 2 das 3 métricas, com ganho mínimo de 0,01 em cada. Na etapa 8, se as duas taxas passarem, fica a de maior soma de ganhos.
- Teste (predições fora da partição, 56 vídeos): só no ponto de partida, no modelo final, nas duas sementes extras e no experimento separado de 3 classes.

## Etapas, cada uma partindo do melhor modelo até então

1. Ponto de partida: configuração atual do modelo com rótulos por intervalo (YOLO11n-Pose, normalização do TCC, janela de 30 linhas, passo 1, taxa de aprendizado 0,001).
2. Pose com YOLO11m-Pose, extraída numa pasta separada. Também se registram a fração de quadros com pessoa detectada e a confiança média dos pontos, comparadas com o YOLO11n.
3. Normalização robusta. Só roda se o passo 0 confirmar o problema, isto é, se pelo menos 5% dos quadros tiverem algum dos pontos 5, 6, 11 ou 12 com confiança abaixo de 0,5 e pelo menos 1% tiver algum valor normalizado acima de 5 larguras de ombro.
   - Escala pela altura do tronco: distância entre o centro dos ombros e o centro dos quadris.
   - Pontos com confiança abaixo de 0,5 viram ausentes e são interpolados entre os quadros vizinhos do mesmo ponto.
   - Valores limitados a ±4 alturas de tronco.
4. Interpolação e suavização.
   - Quadros sem pessoa em lacunas de até 0,5 s são interpolados; lacunas maiores ficam vazias.
   - Média móvel centrada de 0,2 s (2·round(0,1·fps)+1 quadros), dentro de cada trecho contínuo.
5. Reamostragem para 15 quadros por segundo, com janela de 15 quadros (1 s). A interpolação não atravessa lacunas maiores que 0,5 s.
6. Igual à etapa 5, com janela de 30 quadros (2 s).
7. Janelas sem sobreposição no treino; a avaliação continua com passo de 1 quadro.
8. Taxa de aprendizado 0,0003 e 0,003, contra a atual.

Depois das etapas:

- **Estabilidade:** o modelo final é treinado com as sementes 1000 e 2000, e se reportam média e desvio das três execuções na validação e no teste.
- **Experimento separado:** modelo final só com janelas dentro dos intervalos anotados, 3 classes e sem fundo. Reporta-se o macro F1 por quadro nos quadros anotados, comparado com o do TCC, lido de `data/results/analises-posteriores.json`.
