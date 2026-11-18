# Resumo da rodada 4b

## Decisão (2026-10-25): `r4b-C1-janela-4s` é o novo modelo final

Substitui o `r2-C4-aumento`. Passou no critério da validação (F1 evento +0,011, macro F1 quadros +0,026; acurácia bal. −0,011) e o teste confirmou nos eventos (F1 com pós 0,272 contra 0,219, 3 sementes). Limitações: acurácia balanceada um pouco menor, *head banging* pior (−0,035) e dependência do movimento ainda baixa. Registro: `modelo-final.md` e `modelo-final.json`.

## B refeita (adendo): sem tendência linear, 0,5 a 7 Hz, janelas de 8 s e 4 s

A parte B acima (só a média removida, 0,25 a 7 Hz) fica só como registro; vale `r4b-B2-frequencia/frequencia.md`.

Protocolo: `protocolo-rodada4b.md`. Quadros válidos por Q; agregado / média entre partições.

## A. Sem os 5 vídeos problemáticos (ArmFlapping_03, HeadBanging_24, Spinning_09, Spinning_12, Spinning_16), teste
Saem 928 de 94490 quadros e 8 de 100 episódios do gabarito (*ArmFlapping*: 101 quadros, 1 episódios; *HeadBanging*: 140 quadros, 1 episódios; *Spinning*: 687 quadros, 6 episódios). Pós-processamento: mediana_2.0_juncao_1.0.

| | conjunto | F1 evento | acurácia bal. | macro F1 quadros | macro F1 anotados | F1 evento com pós |
|---|---|---|---|---|---|---|
| TCC | todos | 0,015 / 0,017 | 0,575 / 0,584 | 0,393 / 0,386 | 0,331 / 0,350 | 0,155 / 0,170 |
| `r2-C4-aumento` (3 sementes, agregado; média entre partições) | todos | 0,044 ± 0,011; 0,047 | 0,595 ± 0,018; 0,601 | 0,460 ± 0,008; 0,425 | 0,487 ± 0,009; 0,443 | 0,210; 0,202 |
| TCC | sem problematicos | 0,015 / 0,017 | 0,574 / 0,583 | 0,394 / 0,386 | 0,333 / 0,352 | 0,163 / 0,175 |
| `r2-C4-aumento` (3 sementes, agregado; média entre partições) | sem problematicos | 0,044 ± 0,011; 0,048 | 0,595 ± 0,018; 0,601 | 0,462 ± 0,008; 0,429 | 0,492 ± 0,010; 0,449 | 0,219; 0,211 |

## B. Frequência dominante nos intervalos anotados (janelas de 8 s)

| classe | frequência (Hz): mediana [p10; p90] | ciclos em 2 s | ciclos em 4 s | janelas / episódios com janela / vídeos |
|---|---|---|---|---|
| *ArmFlapping* | 0,322 [0,264; 2,080] | 0,000 [0,000; 4,000] | 1,000 [1,000; 8,000] | 245 / 25 de 49 / 13 |
| *HeadBanging* | 0,469 [0,264; 1,025] | 0,000 [0,000; 2,000] | 1,000 [1,000; 4,000] | 182 / 14 de 25 / 11 |
| *Spinning* | 0,879 [0,264; 1,699] | 1,000 [0,000; 3,000] | 3,000 [1,000; 6,000] | 528 / 18 de 27 / 15 |

*Spinning*: a largura dos ombros oscila duas vezes por volta; rotação 0,439 Hz [0,132; 0,850], 0,000 voltas em 2 s e 1,000 em 4 s (medianas).

## C. Variantes contra o `r2-C4-aumento` (validação nos mesmos quadros válidos por Q)

| variante | quadros | F1 evento | acurácia bal. | macro F1 quadros | dependência (AF / HB / SP) | situação | tempo |
|---|---|---|---|---|---|---|---|
| C1: janela de 4 s | 73277 | 0,060 vs 0,049 | 0,659 vs 0,670 | 0,448 vs 0,422 | 0,086 vs 0,038 (0,087 / 0,034 / 0,135) | passa | 35 min |
| C2: P2 (periodicidade) com janela de 4 s | 73277 | 0,030 vs 0,049 | 0,676 vs 0,670 | 0,455 vs 0,422 | -0,054 vs 0,038 (0,163 / -0,292 / -0,032) | não | 41 min |
| C3: janela de 2 s, treino sem os 5 vídeos problemáticos | 76542 | 0,036 vs 0,047 | 0,637 vs 0,667 | 0,368 vs 0,421 | 0,043 vs 0,037 (-0,010 / 0,020 / 0,119) | não | 18 min |
Cada célula: variante vs `r2-C4-aumento` nos mesmos quadros (agregado).

## Avaliação final: r4b-C1-janela-4s (variante que passou)

Teste nos 90722 quadros válidos por Q comuns a TCC, `r2-C4-aumento` e `r4b-C1-janela-4s`. Pós-processamento escolhido na validação: mediana_2.0_juncao_1.0. As métricas por quadro não dependem dele.

| | F1 evento | acurácia bal. | macro F1 quadros | macro F1 anotados | F1 evento com pós |
|---|---|---|---|---|---|
| TCC | 0,017 / 0,018 ± 0,008 | 0,574 / 0,584 ± 0,032 | 0,392 / 0,386 ± 0,055 | 0,328 / 0,348 ± 0,106 | 0,162 / 0,170 |
| `r2-C4-aumento`, 3 sementes, agregado | 0,046 ± 0,011 | 0,595 ± 0,018 | 0,462 ± 0,008 | 0,490 ± 0,010 | 0,219 ± 0,025 |
| `r2-C4-aumento`, 3 sementes, média entre partições | 0,050 ± 0,013 | 0,601 ± 0,011 | 0,427 ± 0,004 | 0,446 ± 0,016 | 0,211 ± 0,019 |
| `r4b-C1-janela-4s`, 3 sementes, agregado | 0,064 ± 0,005 | 0,584 ± 0,015 | 0,458 ± 0,018 | 0,510 ± 0,057 | 0,272 ± 0,014 |
| `r4b-C1-janela-4s`, 3 sementes, média entre partições | 0,073 ± 0,011 | 0,591 ± 0,015 | 0,428 ± 0,015 | 0,467 ± 0,049 | 0,265 ± 0,013 |

F1 por classe (semente 42, agregado; todos os quadros | anotados) e mudança em relação ao `r2-C4-aumento`:
- *ArmFlapping*: 0,254 (+0,007) | 0,447 (+0,078)
- *HeadBanging*: 0,304 (-0,027) | 0,366 (-0,035)
- *Spinning*: 0,601 (+0,017) | 0,749 (+0,062)
- *Background*: 0,658 (-0,073) | --

Dependência do movimento no teste (agregado; por classe AF / HB / SP): TCC 0,235 (0,098 / 0,056 / 0,550); r2-C4-aumento 0,056 (-0,013 / -0,023 / 0,203); r4b-C1-janela-4s 0,069 (-0,007 / -0,013 / 0,227) (média das sementes quando há 3).
Ablações, macro F1 anotados com 3 classes (agregado; sem / ordem / dxdy): r4b-C1-janela-4s: 0,634 / 0,601 / 0,616, 0,699 / 0,651 / 0,685, 0,694 / 0,663 / 0,673; TCC: 0,554 / 0,528 / 0,478.
Fração da classe do vídeo, dentro / fora (semente 42 e TCC): r4b-C1-janela-4s: ArmFlapping 0,187/0,193, HeadBanging 0,292/0,125, Spinning 0,595/0,292; TCC: ArmFlapping 0,116/0,066, HeadBanging 0,076/0,095, Spinning 0,481/0,212.
*Head banging* com F1 zero: r4b-C1-janela-4s: 6 de 16, 7 de 16, 9 de 16; r2-C4-aumento: 6 de 16, 7 de 16, 8 de 16; TCC: 8 de 16.
Matriz de confusão e por classe completos: `r4b-avaliacao-final/avaliacao-final.json`; curva de erro: `r4b-C1-janela-4s/curva-erro.pdf`.
