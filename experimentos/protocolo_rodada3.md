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
