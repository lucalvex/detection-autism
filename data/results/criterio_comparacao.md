# Critério de comparação: Rodada A vs. base re-seedado

Definido antes de rodar a avaliação temporal sobre os dois modelos, para
que a leitura do resultado não seja ajustada depois de vê-lo.

## Métricas principais

- **Macro F1 por quadro**, restrito aos quadros anotados (exclui
  Background do gabarito — ver `metricas.json` /
  `frame_level_report.json`, bloco `somente_quadros_anotados`).
- **F1 por evento com IoU 0,3**, na regra principal (limiar=0,5,
  duração mínima = 1 janela, sem pós-processamento — ver
  `metricas.json`, bloco `por_evento.regra_principal.por_iou["0.3"]`).

Ambas agregadas sobre os 56 vídeos (`pooled_56_videos`).

## Critério de decisão

A Rodada A é considerada **melhor** que o base re-seedado somente se:

1. melhorar as **duas** métricas principais (macro F1 por quadro
   restrito aos quadros anotados **e** F1 por evento IoU 0,3 na regra
   principal); **e**
2. a melhora ocorrer em **pelo menos 4 dos 5 folds** (comparando o
   valor por fold de cada modelo, não só a média agregada).

Se qualquer uma dessas duas condições não se cumprir, o resultado é
relatado como **"sem diferença distinguível"** — não como uma vitória
parcial de um lado ou outro.

## Ressalva sobre HeadBanging

Nos folds 3 e 5, o vídeo(s) de teste de HeadBanging têm poucas janelas
no dataset da Rodada A (218 no fold 3, 683 no fold 5 — ver contagem por
fold em `create_sequences_annotated.py`/`window_stats.json`). As
métricas por fold dessa classe nesses dois folds são reportadas com
essa ressalva: suporte pequeno torna o F1 por fold mais sensível a
poucos acertos/erros, e não deve ser lido com o mesmo peso que os
folds com suporte maior.
