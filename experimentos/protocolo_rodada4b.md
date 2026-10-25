# Protocolo da rodada 4b: vídeos problemáticos, frequência e janela maior

Registrado antes de calcular ou treinar. Executado por `experimentos/rodar_rodada4b.py`, com resultados em `experimentos/r4b_<id>/`.

O modelo base, `data/results/` e as rodadas anteriores não são alterados; só se leem arquivos delas. Valem as regras de `protocolo_rodada4.md`:
- mesmas partições e mesma validação;
- avaliação nos quadros válidos por Q, com todos os modelos nos mesmos quadros e contagem registrada;
- decisão só na validação e teste só no fim;
- duas convenções (agregado e média ± desvio entre partições);
- critério de 2 em 3 métricas, com ganho ≥ 0,01;
- regra da alternativa;
- dependência do movimento, também por classe.

Situação na largada: a rodada 4 terminou sem variante mantida e sem alternativa; o modelo final continua `r2_C4_aumento`.

**Vídeos problemáticos:** `v_Spinning_12`, `v_Spinning_16`, `v_Spinning_09`, `v_ArmFlapping_03` e `v_HeadBanging_24`, apontados pela análise de qualidade (`qualidade_dataset/`).

## A. Vídeos problemáticos, sem treino

Avaliação de teste do `r2_C4_aumento` (3 sementes) e do modelo do TCC, nos quadros válidos por Q comuns aos dois, com e sem os 5 vídeos:
- as quatro métricas nas duas convenções;
- com e sem pós-processamento. Usa-se o mesmo pós-processamento escolhido na validação restrita a Q da rodada 3 (`r3_q_resultados.json`), para comparar com os valores já reportados.

Reportam-se os quadros e os episódios do gabarito que saem, por classe.

## B. Frequência dos movimentos, sem treino

**Dados.** Dentro dos intervalos anotados de cada classe, com as posições da normalização robusta do `r2_C4_aumento` (YOLO11n):
- a sequência é reamostrada a 15 quadros/s por interpolação linear, sem atravessar lacunas de detecção maiores que 0,5 s;
- janelas de 8 s (120 amostras) deslizam de 1 em 1 s, e só valem as inteiramente dentro do intervalo e de um trecho contínuo.

**Sinal por classe:**
- *arm flapping*: x e y de cada punho em relação ao ombro do mesmo lado (4 componentes);
- *head banging*: y do nariz em relação ao centro dos ombros;
- *spinning*: largura aparente dos ombros. Ela oscila duas vezes por volta; reporta-se também a frequência de rotação, igual à metade.

**Frequência dominante:** o pico da potência espectral entre 0,25 e 7 Hz.
- O sinal passa por remoção da média e janela de Hann, e o espectro é completado com zeros até 512 pontos.
- Com vários componentes, as potências se somam.
- Vídeos com menos de 15 quadros/s reais têm frequências acima da metade da sua taxa de quadros apenas interpoladas. A distribuição é reportada também sem esses vídeos.

**Relatório por classe:** mediana e intervalo entre os percentis 10 e 90, número de janelas, de episódios e de vídeos que contribuem, e quantos ciclos completos caberiam em 2 s e em 4 s, que é a frequência vezes a duração, arredondada para baixo.

## C. Treino, cada variante contra o `r2_C4_aumento` (sem acumular)

- **C1:** `r2_C4_aumento` com janela de 4 s (60 amostras a 15 quadros/s).
- **C2:** a P2 da rodada 4 (periodicidade concatenada à saída do LSTM) com janela de 4 s. Os atributos de periodicidade são calculados nas 60 amostras.
- **C3:** `r2_C4_aumento` com janela de 2 s, treinado sem os 5 vídeos problemáticos.
  - A validação e o teste usam todos os vídeos.
  - A exclusão vale para o treino de todas as sementes.

**Conjunto avaliado.** Com janela de 4 s, só têm predição os quadros com 4 s de histórico no mesmo trecho. A comparação usa os quadros válidos por Q comuns ao `r2_C4_aumento` e à variante, e a contagem é registrada.

**Decisão.** Se mais de uma variante passar, fica a de maior soma de ganhos. A vencedora recebe a avaliação final completa da rodada 4, ao lado do `r2_C4_aumento` e do TCC: 3 sementes, quatro métricas nas duas convenções com e sem pós-processamento, por classe, matriz de confusão, dependência do movimento por classe, ablações, fração da classe do vídeo, *head banging* com F1 zero e `curva_erro.pdf`. Se nenhuma passar, só uma alternativa, se houver, recebe essa avaliação.

**Resumo:** `resumo_rodada4b.md`, no máximo uma página. Commits por etapa, sem push, com o índice limpo em caso de falha.

## Adendo de 2026-10-25 16:00: parte B refeita (frequência dos movimentos)

**Por que refazer.** Na primeira versão, as medianas de *arm flapping* e *head banging* caíram perto do limite inferior da faixa (0,25 Hz). Isso indica que o pico de potência pegava a deriva lenta da postura, e não a oscilação do gesto. Além disso, cerca de metade dos episódios é mais curta que 8 s e ficou de fora. O resultado anterior (`r4b_B_frequencia/`) é mantido como está, só para registro.

**Mudanças**, sem treino. O resto da parte B não muda: mesmos sinais, posições da normalização robusta, 15 quadros/s, passo de 1 s, janela de Hann e espectro com zeros até 512 pontos.
1. Antes do espectro, remove-se a tendência linear de cada componente do sinal em cada janela (`scipy.signal.detrend`, tipo linear), em vez de só a média.
2. A frequência dominante é procurada entre 0,5 e 7 Hz.
3. Duas durações de janela: 8 s (120 amostras) e 4 s (60 amostras). A de 4 s alcança episódios curtos, com resolução espectral de 0,25 Hz contra 0,125 Hz na de 8 s.

**Relatório por classe e por duração de janela:**
- frequência dominante (mediana, percentis 10 e 90);
- ciclos completos em 2 s e em 4 s, arredondados para baixo;
- janelas, episódios com janela e vídeos.

Para *spinning*, também a frequência de rotação (metade da frequência da largura dos ombros). Resultado em `r4b_B2_frequencia/`.
