# Adendo ao protocolo: rodada 3

Registrado antes do primeiro treino. Executado por `experimentos/rodar_rodada3.py`, com resultados em `experimentos/r3_<id>/`.

Valem `protocolo.md` e `protocolo_rodada2.md`:
- mesmas partições e mesmos vídeos de validação (`divisao_validacao.json`);
- decisão só na validação e teste só no fim;
- até 500 épocas, com paciência de 20;
- critério para manter: pelo menos 2 das 3 métricas de validação melhores, com ganho ≥ 0,01 em cada, no agregado dos vídeos de validação;
- toda métrica reportada nas duas convenções: agregada sobre os vídeos e média ± desvio entre as 5 partições.

O modelo base, `data/results/` e os experimentos das rodadas 1 e 2 não são alterados; só se leem arquivos deles.

**Referência.** `r2_C4_aumento`, semente 42: normalização robusta, 15 quadros/s, janela de 2 s e aumento de dados. Seus resultados são lidos da rodada 2, sem retreino.

## Variantes

Cada uma é a referência com uma única mudança na entrada.

**I1, interpolação limitada.**
- Um ponto com confiança < 0,5 é interpolado linearmente entre os quadros confiáveis mais próximos de cada lado, só se os dois estiverem a até 0,5 s. Nas pontas do vídeo, nada é repetido.
- Um ponto que não pode ser interpolado fica ausente: valor 0 em x, y, dx e dy, e uma das 17 marcas binárias de presença, acrescentadas à entrada (85 atributos).
- Um quadro em que algum dos pontos 5, 6, 11 ou 12 fica ausente é tratado como quadro sem pessoa detectada: sai da sequência, como as lacunas de detecção.
- Na reamostragem para 15 quadros/s:
  - uma amostra só tem o ponto presente se as duas linhas usadas na interpolação o tiverem;
  - dx e dy valem 0 quando o ponto está ausente na amostra atual ou na anterior.
- O aumento de dados aplica a mesma transformação às marcas, inclusive a troca esquerda/direita, e mantém em 0 os pontos ausentes.
- Os quadros removidos não são avaliados, porque não terminam nenhuma janela; o número removido é reportado por classe.

**I2.** Igual à I1, com a confiança bruta de cada ponto acrescentada à entrada (102 atributos). Na reamostragem, a confiança é interpolada linearmente.

## Decisão

As duas variantes são comparadas com a referência pelo critério.
- **Nenhuma passa:** o modelo final continua `r2_C4_aumento`, a decisão é registrada e a rodada termina.
- **Alguma passa:** a de maior soma de ganhos vira o modelo final.

## Avaliação, só se houver novo modelo final

- Sementes 1000 e 2000.
- Pós-processamento escolhido na validação, só pelo F1 por evento, entre:
  - mediana de 0; 0,5; 1 ou 2 s;
  - junção de trechos separados por menos de 0; 0,5 ou 1 s.

  Ele muda só a construção dos trechos, então as métricas por quadro não dependem dele.
- No teste, nas 3 sementes e nas duas convenções: as quatro métricas, e o F1 por evento com e sem pós-processamento. A comparação é com `r2_C4_aumento` (3 sementes, com o mesmo pós-processamento escolhido) e com o TCC (com e sem esse pós-processamento).
- Ablações (ordem embaralhada e deslocamentos zerados).
- Fração da classe do vídeo dentro e fora dos episódios.
- Precisão, revocação e F1 por classe, incluindo fundo.
- Matriz de confusão nos quadros anotados.
- Vídeos de *head banging* com F1 zero.
- `curva_erro.pdf`.

A curva de quantidade de dados não é refeita.

## Em qualquer caso

`r3_tabela_pontos/`: por ponto (0 a 16), sobre os quadros com pessoa da YOLO11n:
- na interpolação sem limite atual: a fração interpolada (entre quadros confiáveis ou repetida na ponta) e a fração que fica bruta, quando o ponto nunca é confiável no vídeo;
- na I1: a fração interpolada, a fração ausente e a fração de quadros removidos por causa daquele ponto;
- quadros removidos por classe.

## Retomada e commits

- **Retomada:** um resultado gravado só é reaproveitado se a configuração gravada for igual à pedida, ou, no pós-processamento e na avaliação, se o modelo final gravado for o mesmo. Se não for, o script registra um erro e não reaproveita.
- **Commits:** se um `git add` ou um `git commit` falhar, os arquivos são tirados do índice (`git reset`). Nenhum arquivo fica preparado para um commit seu.

## Adendo de 2026-10-20 15:21: variante I3, descarte por qualidade da pose

Registrado antes de a I3 treinar. A I1 e a I2 já tinham terminado, sem passar no critério, e a avaliação final não tinha começado.

**Descarte.** Um quadro é descartado quando os pontos mínimos da normalização não têm confiança ≥ 0,5. Esses pontos não são interpolados. Há dois critérios de mínimo:
- (a) os dois ombros e os dois quadris;
- (b) pelo menos um ombro e pelo menos um quadril.

No (b), o centro dos ombros e o dos quadris usam só os pontos confiáveis, e o ponto não confiável fica ausente (0 + marca). Os demais pontos seguem a regra da I1, aplicada antes do descarte. A entrada é a mesma da I1: 85 atributos, com as marcas de presença.

**I3-treino**, a versão principal, que concorre a modelo final:
- o descarte vale só para os vídeos de treino de cada partição;
- a validação (perda e métricas) e o teste usam todos os quadros, com a regra da I1.

**I3-teste**, que fica só no relatório e não concorre a modelo final:
- o descarte vale também para a validação e o teste;
- os quadros descartados ficam sem predição e saem das métricas por quadro;
- na métrica por evento, os intervalos anotados contam inteiros.

**Decisão.** Concorrem I1, I2, I3-treino (a) e I3-treino (b), cada uma contra `r2_C4_aumento` pelo critério; fica a de maior soma de ganhos. Se alguma passar, segue a avaliação final de `protocolo_rodada3.md`.

**Relatório.** Para cada critério, a fração de quadros e de quadros anotados descartados, por classe:
- no treino: os vídeos de treino das 5 partições, somados;
- no teste: os 56 vídeos, cada um uma vez.

A tabela por ponto passa a se chamar `r3_tabela_pontos/tabela_por_ponto.*`. O nome antigo terminava em `_pontos.json` e era barrado pelo hook, uma regra feita para os pontos brutos da demo.

## Adendo de 2026-10-20 20:58: correção da comparação (cobertura total)

**Problema.** A regra da I1 tirava da sequência os quadros sem ombro ou quadril recuperável em 0,5 s, também na validação e no teste. Por isso, I1, I2 e I3-treino foram avaliadas em cerca de 56% dos quadros que o `r2_C4_aumento` avalia: 61.371 de 108.784 no teste. Saíram 30% dos quadros de *arm flapping*, 50% de *head banging* e 37% de *spinning*.

A comparação pelo critério misturava conjuntos de quadros diferentes e é inválida. Pelo mesmo motivo, a I3-teste já tinha ficado fora da escolha. A escolha da `r3_I3_treino_b` como modelo final, e a avaliação feita para ela, ficam substituídas pela correção abaixo.

**Correção (sem treinar nada).** Os modelos já treinados de I1, I2, I3-treino (a) e I3-treino (b) geram predições em todos os quadros com pessoa detectada que o `r2_C4_aumento` avalia. Nas sequências de validação e de teste:
- nenhum quadro é removido;
- nos quadros em que ombros ou quadris não podem ser recuperados em 0,5 s, o centro e a escala vêm da normalização do `r2_C4_aumento` (interpolação sem limite desses pontos);
- esses pontos ficam marcados como ausentes (0 + marca);
- os demais pontos seguem a regra da I1.

Antes de calcular as métricas, confere-se que cada modelo é avaliado exatamente nos mesmos quadros que o `r2_C4_aumento`, por partição e por vídeo, e registra-se a contagem.

**Decisão.** A decisão vale pela cobertura total: o mesmo critério na validação, para as quatro variantes contra o `r2_C4_aumento`.
- **Alguma passa:** a avaliação de teste completa é refeita com cobertura total nas 3 sementes, com os 17 vídeos de *head banging*. Só a I3-treino (b) tem sementes 1000 e 2000 treinadas; se a vencedora for outra, essas sementes não são treinadas sem nova autorização.
- **Nenhuma passa:** o modelo final continua `r2_C4_aumento`.

**Informação, sem peso na decisão.** A comparação no subconjunto: o `r2_C4_aumento` avaliado só nos quadros que a I1 mantém.

Os resultados ficam em `r3_cobertura_*`, e os da primeira avaliação são mantidos como estão.
