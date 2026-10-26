# Modelo final

**`r4b_C1_janela_4s`** passou a ser o modelo final em 2026-10-25, por decisão do autor, depois da rodada 4b. Ele **substitui o `r2_C4_aumento`**, que era o modelo final desde a rodada 2.

## O que é

A configuração do `r2_C4_aumento` com uma única mudança: janela de 4 s em vez de 2 s.

| Item | Valor |
|---|---|
| Pose | YOLO11n-Pose |
| Normalização | robusta (centro nos quadris, escala pela altura do tronco, limite ±4) |
| Taxa de quadros | 15 quadros/s, janela de 60 amostras (4 s), passo 1 no treino |
| Rede | LSTM de 2 camadas e 128 unidades, Adam com taxa 0,001 |
| Treino | aumento de dados; até 500 épocas com paciência de 20 |
| Rótulos | por intervalo, 4 classes, fundo subamostrado |

**Sementes:** 42 em `r4b_C1_janela_4s/`, 1000 em `r4b_C1_janela_4s_semente_1000/` e 2000 em `r4b_C1_janela_4s_semente_2000/`.

**Pós-processamento dos trechos:** mediana de 2 s e junção de trechos separados por menos de 1 s, escolhidos na validação.

**Avaliação:** nos quadros válidos pelo critério Q (`protocolo_rodada3.md`), com o protocolo de `protocolo_rodada4.md` e `protocolo_rodada4b.md`.

## Por que substitui o `r2_C4_aumento`

**1. Passou no critério registrado antes, na validação.** A comparação foi feita nos mesmos 73.277 quadros válidos por Q, no agregado:

| Métrica | C1 | `r2_C4_aumento` | Ganho |
|---|---|---|---|
| F1 por evento | 0,060 | 0,049 | +0,011 |
| Macro F1 por quadro | 0,448 | 0,422 | +0,026 |
| Acurácia balanceada | 0,659 | 0,670 | −0,011 |

O critério pede ganho de pelo menos 0,01 em 2 das 3 métricas, e duas passaram.

**2. O teste confirmou nos eventos** (90.722 quadros comuns, 3 sementes, agregado):
- **F1 por evento com pós-processamento:** 0,272 ± 0,014, contra 0,219 ± 0,025 do `r2_C4_aumento` e 0,162 do TCC.
- **Sem pós-processamento:** 0,064 contra 0,046.
- **Macro F1 nos quadros anotados:** 0,510 contra 0,490.

**3. Ganho por classe nos quadros anotados (semente 42):** *arm flapping* 0,447 (+0,078) e *spinning* 0,749 (+0,062).

## Limitações, que continuam valendo

- **Acurácia balanceada da presença:** um pouco menor que a do `r2_C4_aumento`, 0,584 contra 0,595 no teste, agregado.
- **Macro F1 por quadro:** igual ao do `r2_C4_aumento`, 0,458 contra 0,462.
- ***Head banging*:** piora para 0,366 (−0,035). Os vídeos com F1 zero são 6, 7 e 9 de 16 nas três sementes.
- **Dependência do movimento:** continua baixa, 0,069 contra 0,056 do `r2_C4_aumento` e 0,235 do TCC, e quase toda em *spinning*. O modelo segue reconhecendo sobretudo a postura.
- **Ganho no F1 por evento:** na validação, passou o mínimo de 0,01 por pouco (+0,011).

## Histórico do modelo final

1. Rodada 1: `06_15fps_janela_2s`, chamado de R.
2. Rodada 2: `r2_C4_aumento`, que acrescenta aumento de dados.
3. Rodadas 3 e 4: sem mudança. Nenhuma variante passou nos quadros comparáveis: I1, I2, I3 e QF na rodada 3; P1, P2 e P3 na rodada 4.
4. Rodada 4b: `r4b_C1_janela_4s`.

Detalhes em `resumo_rodada4b.md` e `r4b_avaliacao_final/avaliacao_final.json`. O modelo do TCC, que foi para o texto, não muda: este registro vale para os experimentos.

## Versão para a demonstração ao vivo: `models/final_completo.pt`

Usada só pela câmera ao vivo da interface. **Não entra em nenhuma métrica**: viu todos os 56 vídeos no treino, então qualquer avaliação nesses vídeos seria com dados de treino.

- **Configuração:** a do `r4b_C1_janela_4s`, com semente 42, treinada com os 56 vídeos, sem validação.
- **Épocas:** a mediana das épocas escolhidas na validação das 5 partições do modelo final. As escolhidas foram 4, 1, 1, 1 e 1, então o treino durou **1 época**. A perda de validação do modelo final já era mínima depois da primeira época na maioria das partições, ou seja, ele sobreajusta cedo.
- **Arquivos:** o script é `treinar_final_completo.py`; o registro, `modelo_final_completo.json`. O `.pt` fica fora do git.
