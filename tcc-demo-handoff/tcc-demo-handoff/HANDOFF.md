# Demonstração do TCC: passagem do design para implementação

Este pacote leva para o repositório do TCC o design da interface de demonstração (detecção automática de estereotipias motoras associadas ao TEA com YOLO11n-Pose + LSTM). O objetivo agora é trocar os dados sintéticos por dados reais do pipeline, sem perder o design.

## O que tem aqui

`design/revisao.dc.html` e `design/apresentacao.dc.html` são as duas telas desenhadas no canvas de design do Claude. Elas usam um runtime próprio daquele editor (`<x-dc>`, `<sc-for>`, `<sc-if>`, `{{holes}}`, `class Component extends DCLogic`) e **não rodam fora dele**. Servem como referência exata de layout, cores, tipografia, textos e comportamento: o HTML inline mostra a estrutura, e o método `renderVals()` de cada arquivo mostra a lógica (cálculo dos trechos, geometria da linha do tempo, desenho do esqueleto). A implementação deve reproduzir essas telas em tecnologia comum, não tentar executar esses arquivos.

`data/exemplo_sessao.json` e `data/exemplo_pontos.json` são dados **sintéticos** no formato de dados proposto abaixo, gerados com as mesmas funções da demonstração. Servem de fixture para construir a interface antes de o exportador real existir. Nada neles pode aparecer como resultado do trabalho.

## O que construir

Três peças, nesta ordem.

**1. Exportador (Python, dentro do pipeline existente).** Um script, por exemplo `exportar_sessao.py <video> --modelo <checkpoint> --saida <pasta>`, que roda o YOLO11n-Pose no vídeo, monta as janelas exatamente como no treino (mesmo tamanho, passo e normalização), aplica o LSTM treinado e grava `<id>_sessao.json` e `<id>_pontos.json` no formato abaixo. Reaproveitar o código de pré-processamento do treino em vez de reescrever: qualquer divergência entre inferência e treino invalida a demonstração.

**2. Exportador de métricas.** Um script que lê os resultados da avaliação já feita (com a divisão treino/teste por vídeo via `groups.npy`) e grava `metricas.json`. A tela de apresentação mostra essas métricas; hoje elas são `[valor]`.

**3. Interface web local.** Uma página que carrega um vídeo + os dois JSONs e reproduz as duas telas: modo revisão e modo apresentação. Sugestão de pilha simples: HTML + JavaScript (ou TypeScript com Vite), servida localmente com `python -m http.server` ou `npm run dev`. Sem backend obrigatório. Os vídeos ficam só na máquina local.

## Formato de dados

### `<id>_sessao.json`

```json
{
  "versao": 1,
  "sintetico": false,
  "sessao": { "id": "v042", "video": "videos/v042.mp4", "fps": 30, "total_quadros": 1800, "largura": 1280, "altura": 720 },
  "modelo": {
    "pose": "yolo11n-pose",
    "formato_pontos": "coco17",
    "classificador": "lstm",
    "ativacao_saida": "softmax",
    "janela_quadros": 60,
    "passo_quadros": 15
  },
  "classes": [
    { "id": "armflapping", "nome": "Balançar os braços" },
    { "id": "headbanging", "nome": "Bater a cabeça" },
    { "id": "spinning", "nome": "Girar o corpo" }
  ],
  "limiar_padrao": 0.5,
  "duracao_minima_s": 1.0,
  "janelas": [
    { "quadro_inicio": 0, "quadro_fim": 60, "pontuacoes": [0.04, 0.07, 0.03] }
  ],
  "pontos": { "arquivo": "v042_pontos.json" }
}
```

`pontuacoes` segue a ordem de `classes`. Os valores de `janela_quadros`, `passo_quadros`, `ativacao_saida`, `limiar_padrao` e `duracao_minima_s` no exemplo são provisórios: o exportador deve gravar os valores reais do modelo treinado.

### `<id>_pontos.json`

```json
{ "versao": 1, "formato": "coco17", "ordem": "[x_px, y_px, confianca] por ponto", "quadros": [ [[640.0, 163.6, 0.91], "... 17 pontos"], "... um item por quadro" ] }
```

Coordenadas em pixels do vídeo original. Quadro sem pessoa detectada: `null`. Se houver mais de uma pessoa, decidir e documentar qual é usada (a mesma regra do treino).

### `metricas.json`

```json
{ "divisao": "por vídeo (groups.npy)", "f1_macro": null, "acuracia_balanceada": null,
  "por_classe": [{ "id": "armflapping", "precisao": null, "revocacao": null, "f1": null, "suporte": null }],
  "matriz_confusao": null }
```

## Regras que a interface precisa seguir

**Da janela para a linha do tempo.** Cada janela é plotada no seu centro, em `(quadro_inicio + quadro_fim) / 2 / fps` segundos. Um trecho é uma sequência de janelas consecutivas com pontuação ≥ limiar na mesma classe. Ele vai do `quadro_inicio` da primeira janela ao `quadro_fim` da última, e só entra na lista se durar pelo menos `duracao_minima_s`. O pico é a maior pontuação entre essas janelas. O limiar é recalculado ao vivo quando o controle muda. No design isso está em `episodes()` do `revisao.dc.html`.

**Softmax sem classe de repouso.** Se `ativacao_saida` for `softmax` sobre só as três classes, toda janela soma 1. Mesmo sem estereotipia, alguma classe pontua alto, e a linha do tempo vira ruído. Antes de ligar a interface, confirmar como o modelo lida com janelas sem comportamento: classe de repouso, sigmoides independentes ou outro critério. Se não houver tratamento, a interface deve avisar isso em vez de esconder.

**Esqueleto sobre o vídeo.** Os pontos estão em pixels do vídeo original. O vídeo exibido é redimensionado (com `object-fit: contain` pode haver faixas pretas). A camada SVG precisa aplicar a mesma escala e o mesmo deslocamento. Sincronizar pelo quadro atual, `Math.floor(video.currentTime * fps)`, a cada `requestVideoFrameCallback` (ou `requestAnimationFrame` como alternativa).

**Revisões.** Confirmar e Descartar geram decisões que precisam sobreviver ao recarregar a página. Salvar em `localStorage` por sessão e oferecer "Exportar revisões" como JSON (`{ sessao, limiar, trechos: [{ classe, inicio_s, fim_s, decisao }] }`). Uma decisão pertence ao trecho, e o trecho muda com o limiar: guardar as decisões com o limiar em que foram tomadas.

**Vocabulário.** A interface diz "pontuação" (nunca "confiança" ou "probabilidade", a menos que o modelo seja calibrado) e "trecho sugerido" (nunca "diagnóstico" ou "resultado"). O aviso de triagem aparece sempre no modo revisão.

## Design (fixo)

A tipografia usa uma família só: Atkinson Hyperlegible Next (Google Fonts), pesos 400, 500 e 700. Números usam `font-variant-numeric: tabular-nums`.

| Papel | Cor |
|---|---|
| Fundo | `#E7EBEA` |
| Superfície | `#F6F8F7` |
| Texto | `#1E2B2F` |
| Texto secundário | `#4A585C` |
| Linhas | `#C3CBC8` |
| Balançar os braços | `#1F4F8F` |
| Bater a cabeça | `#E0A13A` |
| Girar o corpo | `#8E6FC0` |

As cores das classes diferem em luminosidade de propósito (daltonismo); manter isso ao adicionar as classes 4 a 6. Os alvos de toque têm pelo menos 44 px, o foco visível é um contorno de 3 px `#1F4F8F`, e os controles são `<button>`, `<input>` e `<label>` reais. As medidas exatas estão nos estilos inline dos arquivos em `design/`.

## Espaços reservados que continuam abertos

No modo apresentação: taxa de quadros, normalização dos pontos, se a confiança dos pontos entra no LSTM, tamanho e passo da janela, camadas e unidades do LSTM, ativação de saída, limiar padrão, duração mínima, F1 macro e acurácia balanceada. Também faltam nome do autor e orientação. Preencher a partir do código e dos resultados reais, nunca inventar.

## Privacidade

Vídeos, pontos e revisões não vão para o git: adicionar `videos/`, `saidas/` e `*_pontos.json` ao `.gitignore`. Mesmo o SSBD sendo público, qualquer vídeo clínico real exige os cuidados do comitê de ética.
