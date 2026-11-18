# Pontos interpolados, ausentes e quadros removidos (YOLO11n, frações dos quadros com pessoa)

| ponto | atual: interpolado | atual: bruto | I1: interpolado | I1: ausente | I1: quadros removidos |
|---|---|---|---|---|---|
| 0 nariz | 0,229 | 0,000 | 0,095 | 0,134 | -- |
| 1 olho esq. | 0,293 | 0,000 | 0,092 | 0,201 | -- |
| 2 olho dir. | 0,345 | 0,000 | 0,110 | 0,235 | -- |
| 3 orelha esq. | 0,347 | 0,000 | 0,103 | 0,243 | -- |
| 4 orelha dir. | 0,476 | 0,044 | 0,110 | 0,410 | -- |
| 5 ombro esq. | 0,094 | 0,000 | 0,053 | 0,040 | 0,040 |
| 6 ombro dir. | 0,090 | 0,000 | 0,053 | 0,037 | 0,037 |
| 7 cotovelo esq. | 0,296 | 0,000 | 0,099 | 0,197 | -- |
| 8 cotovelo dir. | 0,297 | 0,000 | 0,112 | 0,185 | -- |
| 9 punho esq. | 0,302 | 0,000 | 0,123 | 0,179 | -- |
| 10 punho dir. | 0,294 | 0,000 | 0,122 | 0,171 | -- |
| 11 quadril esq. | 0,400 | 0,016 | 0,075 | 0,341 | 0,341 |
| 12 quadril dir. | 0,396 | 0,016 | 0,083 | 0,329 | 0,329 |
| 13 joelho esq. | 0,622 | 0,032 | 0,062 | 0,592 | -- |
| 14 joelho dir. | 0,607 | 0,047 | 0,059 | 0,595 | -- |
| 15 tornozelo esq. | 0,722 | 0,050 | 0,056 | 0,716 | -- |
| 16 tornozelo dir. | 0,721 | 0,050 | 0,057 | 0,714 | -- |

Quadros removidos na I1 (algum de 5, 6, 11, 12 ausente): 41761 de 113671 (0,367); *ArmFlapping* 15860 de 52345 (0,303); *HeadBanging* 12771 de 25339 (0,504); *Spinning* 13130 de 35987 (0,365).

- atual_interp: conf < 0,5 e o ponto tem algum quadro confiável no vídeo (interpolado sem limite ou repetido na ponta)
- atual_bruto: conf < 0,5 e o ponto nunca é confiável no vídeo (fica a estimativa bruta)
- i1_interp: conf < 0,5 e quadro confiável a até 0,5 s dos dois lados (interpolado)
- i1_ausente: conf < 0,5 sem quadro confiável a até 0,5 s dos dois lados (0 + marca de ausente)
- i1_remove: quadro removido por este ponto estar ausente (só 5, 6, 11, 12; um quadro pode ter mais de um)
- denominador: quadros com pessoa detectada pela YOLO11n, antes da remoção
