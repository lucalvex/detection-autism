# Resumo da rodada 3

**A primeira avaliação da I3 foi substituída.** A regra da I1 tirava os quadros sem ombro ou quadril recuperável só da avaliação dos modelos novos (cerca de 44% dos quadros), e a comparação com o `r2_C4_aumento` misturava conjuntos de quadros diferentes. Ela foi corrigida de duas formas, que valem juntas: a cobertura total (todos os modelos nos mesmos quadros) e o critério Q, decidido pelo orientador, que aplica a qualidade da pose como critério de dados a todos os modelos antes da separação entre treino, validação e teste (adendos de 2026-10-20 e 2026-10-21 em `protocolo_rodada3.md`).

## Cobertura total (todos os quadros com pessoa; validação, agregado: F1 evento / acurácia bal. / macro F1 quadros)

`r2_C4_aumento` 0,040 / 0,668 / 0,414; r3_I1_interpolacao_limitada: 0,033 / 0,631 / 0,356; r3_I2_limitada_confianca: 0,037 / 0,628 / 0,367; r3_I3_treino_a: 0,028 / 0,629 / 0,376; r3_I3_treino_b: 0,031 / 0,642 / 0,394. Nenhuma variante passa.
No subconjunto de 52963 quadros que a I1 mantém (resultado já calculado, citado): `r2_C4_aumento` 0,067 / 0,673 / 0,472; r3_I1_interpolacao_limitada: 0,052 / 0,651 / 0,411; r3_I2_limitada_confianca: 0,047 / 0,665 / 0,440; r3_I3_treino_a: 0,050 / 0,666 / 0,444; r3_I3_treino_b: 0,051 / 0,680 / 0,463.

## Critério Q

Quadro válido: os dois ombros e pelo menos um ponto da cabeça com confiança ≥ 0,5. Excluídos (quadros / anotados / episódios): *ArmFlapping* 0,087 / 0,043 / 0 de 49; *HeadBanging* 0,231 / 0,267 / 1 de 25; *Spinning* 0,142 / 0,121 / 0 de 27. 1 episódios saem do gabarito; vídeos sem quadro válido: 0. Detalhes: `r3_q_cobertura/cobertura.md`.

**Decisão (validação, 71219 quadros comuns; agregado / média ± desvio):**

| | F1 evento | acurácia bal. | macro F1 quadros |
|---|---|---|---|
| `r2_C4_aumento` restrito a Q | 0,050 / 0,057 ± 0,027 | 0,673 / 0,690 ± 0,039 | 0,430 / 0,459 ± 0,109 |
| QF | 0,034 / 0,039 ± 0,017 | 0,601 / 0,631 ± 0,064 | 0,358 / 0,373 ± 0,055 |

QF não passa no critério; modelo final: `r2_C4_aumento`.

**Teste do modelo final nos quadros válidos por Q (94490 quadros comuns; agregado / média ± desvio entre partições).** Pós-processamento escolhido na validação restrita: mediana_2.0_juncao_1.0; ele só muda o F1 por evento.

| | F1 evento | acurácia bal. | macro F1 quadros | macro F1 anotados | F1 evento com pós |
|---|---|---|---|---|---|
| TCC | 0,015 / 0,017 ± 0,009 | 0,575 / 0,584 ± 0,033 | 0,393 / 0,386 ± 0,055 | 0,331 / 0,350 ± 0,104 | 0,155 / 0,170 |
| `r2_C4_aumento` (final), 3 sementes, agregado (média ± desvio entre sementes) | 0,044 ± 0,011 | 0,595 ± 0,018 | 0,460 ± 0,008 | 0,487 ± 0,009 | 0,210 ± 0,025 |
| `r2_C4_aumento` (final), 3 sementes, média entre partições | 0,047 ± 0,012 | 0,601 ± 0,011 | 0,425 ± 0,004 | 0,443 ± 0,015 | 0,202 |

*TCC*, semente 42: F1 por classe em todos os quadros (P / R / F1): ArmFlapping 0,261 / 0,109 / 0,154; HeadBanging 0,208 / 0,112 / 0,145; Spinning 0,596 / 0,467 / 0,524; Background 0,678 / 0,838 / 0,750. Nos anotados: ArmFlapping 0,676 / 0,109 / 0,188; HeadBanging 0,514 / 0,112 / 0,183; Spinning 0,932 / 0,467 / 0,623. Matriz (linhas anotadas AF, HB, SP; colunas AF, HB, SP, fundo): [[1481, 520, 184, 11374], [477, 761, 305, 5271], [233, 200, 6666, 7160]]. *Head banging* com F1 zero: 8 de 16 vídeos com quadros válidos (de 17).

*r2_C4_aumento*, semente 42: F1 por classe em todos os quadros (P / R / F1): ArmFlapping 0,258 / 0,229 / 0,242; HeadBanging 0,304 / 0,349 / 0,325; Spinning 0,626 / 0,549 / 0,585; Background 0,718 / 0,746 / 0,732. Nos anotados: ArmFlapping 0,884 / 0,229 / 0,363; HeadBanging 0,462 / 0,349 / 0,398; Spinning 0,923 / 0,549 / 0,689. Matriz (linhas anotadas AF, HB, SP; colunas AF, HB, SP, fundo): [[3099, 105, 352, 10003], [395, 2378, 302, 3739], [11, 2661, 7832, 3755]]. *Head banging* com F1 zero: 6 de 16 vídeos com quadros válidos (de 17).

Ablações, macro F1 nos anotados com 3 classes (agregado / média): *r2_C4_aumento* por semente: sem_ablacao: 0,650 / 0,632, 0,664 / 0,629, 0,655 / 0,637; ordem: 0,640 / 0,622, 0,642 / 0,607, 0,644 / 0,630; dxdy: 0,656 / 0,620, 0,656 / 0,610, 0,641 / 0,619; *TCC*: sem_ablacao: 0,557 / 0,560; ordem: 0,533 / 0,529; dxdy: 0,484 / 0,487.
Fração da classe do vídeo, dentro / fora, sem ablação: *r2_C4_aumento* semente 42: ArmFlapping 0,145/0,132, HeadBanging 0,283/0,106, Spinning 0,514/0,197; *r2_C4_aumento* semente 1000: ArmFlapping 0,195/0,177, HeadBanging 0,220/0,080, Spinning 0,490/0,249; *r2_C4_aumento* semente 2000: ArmFlapping 0,167/0,165, HeadBanging 0,220/0,093, Spinning 0,536/0,191; *TCC*: ArmFlapping 0,114/0,070, HeadBanging 0,081/0,103, Spinning 0,484/0,215.
*Head banging* com F1 zero por semente (*r2_C4_aumento*): 6 de 16, 7 de 16, 8 de 16. Tudo em `r3_q_resultados.json`.

## Primeira avaliação (substituída: subconjunto de quadros só nos modelos novos)

| variante | mudança | F1 evento IoU 0,3 | acurácia bal. | macro F1 quadros | passa | tempo |
|---|---|---|---|---|---|---|
| r2_C4_aumento | referência (rodada 2) | 0,040 / 0,049 ± 0,031 | 0,668 / 0,676 ± 0,021 | 0,414 / 0,432 ± 0,090 | -- | -- |
| r3_I1_interpolacao_limitada | interpolação até 0,5 s, ausentes = 0 + marca, sem ombro/quadril = sem pessoa | 0,052 / 0,054 ± 0,025 | 0,651 / 0,667 ± 0,063 | 0,411 / 0,430 ± 0,050 | não | 14 min |
| r3_I2_limitada_confianca | I1 + confiança de cada ponto | 0,047 / 0,043 ± 0,019 | 0,665 / 0,677 ± 0,041 | 0,440 / 0,450 ± 0,070 | não | 15 min |
| r3_I3_treino_a | I3-treino, critério (a): descarta quadro sem os dois ombros e os dois quadris confiáveis só no treino | 0,050 / 0,046 ± 0,014 | 0,666 / 0,672 ± 0,038 | 0,444 / 0,426 ± 0,027 | não | 10 min |
| r3_I3_treino_b | I3-treino, critério (b): descarta quadro sem um ombro e um quadril confiáveis só no treino | 0,051 / 0,053 ± 0,027 | 0,680 / 0,675 ± 0,048 | 0,463 / 0,434 ± 0,050 | sim | 11 min |
| r3_I3_teste_a | I3-teste, critério (a): descarta quadro sem os dois ombros e os dois quadris confiáveis no treino, na validação e no teste (não concorre) | 0,065 / 0,075 ± 0,030 | 0,675 / 0,683 ± 0,069 | 0,473 / 0,446 ± 0,033 | não concorre | 10 min |
| r3_I3_teste_b | I3-teste, critério (b): descarta quadro sem um ombro e um quadril confiáveis no treino, na validação e no teste (não concorre) | 0,041 / 0,048 ± 0,019 | 0,687 / 0,697 ± 0,070 | 0,438 / 0,423 ± 0,023 | não concorre | 11 min |

Detalhes da primeira avaliação: `r3_avaliacao_final/` (não usar).
