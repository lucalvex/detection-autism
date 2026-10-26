import { buildSegments } from "./timeline.js";
import {
  CORES, RESULTADOS, OSSOS, carregarJSON, carregarSessao, num, fmtTempo, fmtTempoFino,
  centroJanela, janelaMaisProxima, sequencias, caminhoCurva, svg, html, mostrarErro,
  nomeClasse, rotuloSessao, NOMES_EN,
} from "./comum.js";
import {
  CORES_MODELO, NOMES_MODELO, CURTO_MODELO, barrasHorizontais, mapaDeCalor, escalaSequencial, inteiro,
} from "./graficos.js";

// lista "a, b e c" de fragmentos ou textos
function juntar(itens) {
  const out = [];
  itens.forEach((x, i) => {
    if (i) out.push(i === itens.length - 1 ? " e " : ", ");
    out.push(x);
  });
  return out;
}

const $ = (id) => document.getElementById(id);
const SUB = { 0: "₀", 1: "₁", 2: "₂", 3: "₃", 4: "₄", 5: "₅", 6: "₆", 7: "₇", 8: "₈", 9: "₉" };
const subscrito = (n) => String(n).split("").map((d) => SUB[d]).join("");

const estado = { etapa: 0, dados: null, video: null };

async function iniciar() {
  try {
    const [medianos, rodadaA, base, posteriores, comparacao, resumo, baseOriginal] = await Promise.all([
      carregarJSON(RESULTADOS.medianos),
      carregarJSON(RESULTADOS.rodadaA),
      carregarJSON(RESULTADOS.base),
      carregarJSON(RESULTADOS.posteriores),
      carregarJSON(RESULTADOS.comparacao),
      carregarJSON(RESULTADOS.resumo),
      carregarJSON(RESULTADOS.baseOriginal),
    ]);
    estado.dados = { medianos, rodadaA, base, posteriores, comparacao, resumo, baseOriginal };

    const escolhidos = Object.values(medianos.escolhidos);
    const pedido = new URLSearchParams(location.search).get("video");
    const inicial = escolhidos.find((e) => e.video_id === pedido) || escolhidos[0];

    montarRodape();
    montarPosteriores();
    await selecionarVideo(inicial.video_id);
    montarRegra();
    montarMatrizes();
  } catch (e) {
    mostrarErro($("conteudo"), e.message + " Rode os scripts de src/ que geram os arquivos de resultado e sirva a interface com python demo/servir.py.");
  }
}

async function selecionarVideo(id) {
  const { sessao, pontos } = await carregarSessao(id);
  estado.video = { id, sessao, pontos };
  history.replaceState(null, "", "?video=" + encodeURIComponent(id));
  $("link-revisao").href = "revisao.html?sessao=" + encodeURIComponent(id);
  montarMedianos();
  montarEtapas();
  mostrarEtapa(estado.etapa);
}

// ---------- vídeos de desempenho mediano ----------

function montarMedianos() {
  const { medianos } = estado.dados;
  const lista = $("medianos-lista");
  lista.replaceChildren();
  const porId = Object.fromEntries(estado.video.sessao.classes.map((c) => [c.id, c]));
  for (const [classe, e] of Object.entries(medianos.escolhidos)) {
    const id = classe.toLowerCase();
    const empatados = medianos.empatados_com_o_escolhido[classe].length;
    const detalhe =
      `${rotuloSessao(e.video_id)} · F1 por quadro ${num(e.f1_quadro_anotados, 3)}, ${e.posicao_na_ordem + 1}º de ${e.n_videos_da_classe}` +
      (empatados > 1 ? `, empatado com outros ${empatados - 1}` : "");
    lista.appendChild(html("button", {
      type: "button",
      class: "mediano",
      "aria-pressed": String(e.video_id === estado.video.id),
      onclick: () => selecionarVideo(e.video_id),
    }, [
      html("span", { class: "quadradinho", estilo: `background:${CORES[id]}` }),
      html("span", { class: "textos" }, [
        html("span", { class: "nome" }, [nomeClasse(porId[id])]),
        html("span", { class: "detalhe", texto: detalhe }),
      ]),
    ]));
  }

  const m = estado.video.sessao.modelo;
  const aviso = $("aviso-particao");
  aviso.replaceChildren(
    html("strong", { texto: "Predições fora da partição." }),
    html("span", {
      texto: ` As etapas abaixo usam ${estado.video.id}, avaliado pelo modelo da partição ${m.particao}, treinado sem este vídeo. O vídeo em si não é exibido aqui: os quadros são desenhados a partir dos pontos do corpo exportados.`,
    })
  );
  aviso.hidden = false;
}

// regra de escolha, em linguagem simples (o registro técnico fica em
// videos_desempenho_mediano.json, campo "regra")
const SAIDAS_POR_EXTENSO = { 2: "duas", 3: "três", 4: "quatro", 5: "cinco", 6: "seis" };

function montarRegra() {
  const n = estado.video.sessao.modelo.n_saidas;
  const itens = [
    ["Regra", "Um vídeo por classe, o de desempenho mediano."],
    ["Medida", "F1 por quadro do vídeo, média sobre as classes anotadas nele."],
    ["Quadros avaliados", "Só os que estão dentro dos intervalos anotados no SSBD e têm predição."],
    ["Predição", `A classe de maior pontuação entre as ${SAIDAS_POR_EXTENSO[n] || n} saídas do modelo, incluindo fundo. Prever fundo num quadro anotado conta como erro.`],
    ["Posição", "O vídeo do meio na ordem crescente de F1. Com número par de vídeos, o de posição inferior."],
    ["Desempate", "Ordem alfabética do nome do arquivo."],
  ];
  $("regra-lista").replaceChildren(...itens.map(([k, v]) => html("div", {}, [html("dt", { texto: k }), html("dd", { texto: v })])));
}

// ---------- etapas ----------

function intervaloExemplo() {
  const s = estado.video.sessao;
  const classeVideo = s.classes.find((c) => estado.video.id.toLowerCase().includes(c.id));
  const a = s.anotacoes.find((x) => x.classe === classeVideo.id) || s.anotacoes[0];
  return { anotacao: a, classe: classeVideo, meio: (a.inicio_s + a.fim_s) / 2 };
}

function etapas() {
  const { sessao: s, pontos } = estado.video;
  const m = s.modelo;
  const nPontos = pontos.quadros.find((q) => q !== null).length;
  const porPonto = m.n_caracteristicas / nPontos;
  const ex = intervaloExemplo();
  const janela = janelaMaisProxima(s.janelas, s.sessao.fps, ex.meio);
  const nLongas = s.janelas.filter((j) => j.janela_longa).length;
  const saidas = juntar(s.classes.map((c) => nomeClasse(c, { apoio: false })));

  return [
    {
      titulo: "Vídeo", curto: "A sessão gravada, quadro a quadro.",
      texto: "O sistema recebe o vídeo de uma sessão gravada e o percorre quadro a quadro. O enquadramento importa: a pessoa precisa aparecer de corpo inteiro para que a etapa seguinte encontre os pontos do corpo.",
      params: [
        ["Entrada:", `vídeo da sessão, ${num(s.sessao.fps, 2)} quadros por segundo neste vídeo`],
        ["Este vídeo:", `${s.sessao.total_quadros} quadros, ${s.quadros_sem_deteccao.length} sem pessoa detectada`],
      ],
      visual: visualQuadros,
    },
    {
      titulo: "Pose", curto: `${nPontos} pontos do corpo por quadro.`,
      texto: `O YOLO11n-Pose localiza ${nPontos} pontos-chave em cada quadro: rosto, ombros, cotovelos, punhos, quadris, joelhos e tornozelos. Daqui em diante o classificador não vê mais a imagem, apenas a posição desses pontos ao longo do tempo.`,
      params: [
        ["Modelo:", "YOLO11n-Pose; com mais de uma pessoa, usa a primeira detectada"],
        ["Por quadro:", `${nPontos} pontos (x, y); a confiança dos pontos não entra no LSTM`],
        ["Normalização:", "posição em relação ao ponto médio dos quadris, dividida pela distância entre os ombros"],
        ["Características:", `${m.n_caracteristicas} por quadro: x, y e o deslocamento dx, dy em relação ao quadro anterior, ${porPonto} por ponto`],
      ],
      visual: visualPose,
    },
    {
      titulo: "Janelas", curto: "Trechos curtos e sobrepostos.",
      texto: "Uma estereotipia é um padrão que se repete no tempo, não uma pose isolada. Por isso a sequência de pontos é dividida em janelas sobrepostas, e cada janela é classificada como um todo.",
      params: [
        ["Tamanho da janela:", `${m.janela_quadros} quadros com pessoa detectada`],
        ["Passo:", `${m.passo_quadros} quadro`],
        ["Janelas longas:", `as que cobrem mais de ${s.limite_janela_longa_quadros} quadros do vídeo por causa de quadros sem detecção; ${nLongas} de ${s.janelas.length} neste vídeo`],
      ],
      visual: visualJanelas,
    },
    {
      titulo: "LSTM", curto: "Lê a janela em ordem.",
      texto: "A rede LSTM lê a janela quadro a quadro, carregando um estado interno que acumula o que já viu. No fim da janela, esse estado resume o movimento e alimenta a camada de saída, com uma unidade por classe, incluindo uma para a ausência de estereotipia.",
      params: [
        ["Camadas LSTM:", String(m.lstm_camadas)],
        ["Unidades por camada:", String(m.lstm_unidades)],
        ["Ativação de saída:", [`${m.ativacao_saida}, ${m.n_saidas} saídas (`, ...saidas, ")"]],
      ],
      visual: visualLSTM,
    },
    {
      titulo: "Pontuações", curto: "Uma pontuação por classe.",
      texto: "Cada janela recebe uma pontuação por classe, e as pontuações de uma janela somam 1. Uma janela isolada é ruidosa; o sinal que interessa é uma pontuação alta que se sustenta por várias janelas seguidas.",
      params: [
        ["Exemplo:", [`janela de ${fmtTempoFino(janela.quadro_inicio / s.sessao.fps)} a ${fmtTempoFino(janela.quadro_fim / s.sessao.fps)} de ${estado.video.id}, no meio do intervalo anotado de `, nomeClasse(ex.classe, { apoio: false })]],
      ],
      visual: visualPontuacoes,
    },
    {
      titulo: "Trechos", curto: "Sugestões para revisão.",
      texto: "Janelas consecutivas acima do limiar são unidas em um trecho, com início, fim e pontuação máxima. Cada trecho vai para um profissional, que o confirma ou descarta. O sistema aponta onde olhar; a decisão continua sendo humana.",
      params: [
        ["Limiar padrão:", `${num(s.limiar_padrao, 2)}, ajustável na revisão`],
        ["Duração mínima de um trecho:", `uma janela (${num(s.duracao_minima_s, 2)} s neste vídeo)`],
      ],
      visual: visualTrechos,
    },
  ];
}

function montarEtapas() {
  const ol = $("etapas");
  ol.replaceChildren();
  etapas().forEach((e, j) => {
    ol.appendChild(html("li", {}, [
      html("button", { type: "button", class: "etapa-btn", "data-etapa": j, onclick: () => mostrarEtapa(j) }, [
        html("span", { class: "n", texto: String(j + 1) }),
        html("span", { class: "t", texto: e.titulo }),
        html("span", { class: "c", texto: e.curto }),
      ]),
    ]));
  });
  $("btn-anterior").onclick = () => mostrarEtapa(Math.max(0, estado.etapa - 1));
  $("btn-proxima").onclick = () => mostrarEtapa(Math.min(etapas().length - 1, estado.etapa + 1));
}

function mostrarEtapa(j) {
  const lista = etapas();
  estado.etapa = j;
  const e = lista[j];
  for (const b of document.querySelectorAll(".etapa-btn")) {
    b.setAttribute("aria-current", Number(b.dataset.etapa) === j ? "step" : "false");
  }
  $("etapa-titulo").textContent = e.titulo;
  $("etapa-descricao").textContent = e.texto;
  const dl = $("etapa-params");
  dl.replaceChildren(...e.params.map(([k, v]) => html("div", {}, [html("dt", { texto: k }), html("dd", {}, Array.isArray(v) ? v : [v])])));
  $("btn-anterior").disabled = j === 0;
  $("btn-proxima").disabled = j === lista.length - 1;
  const visual = $("etapa-visual");
  visual.replaceChildren();
  e.visual(visual);
}

// ---------- visuais ----------

function caixaDosPontos(quadros) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const q of quadros) {
    if (!q) continue;
    for (const [x, y] of q) {
      x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y);
    }
  }
  return { x0, y0, x1, y1 };
}

function encaixar(caixa, largura, altura, margem) {
  const escala = Math.min((largura - 2 * margem) / Math.max(1, caixa.x1 - caixa.x0), (altura - 2 * margem) / Math.max(1, caixa.y1 - caixa.y0));
  const ox = (largura - (caixa.x1 - caixa.x0) * escala) / 2;
  const oy = (altura - (caixa.y1 - caixa.y0) * escala) / 2;
  return (x, y) => [ox + (x - caixa.x0) * escala, oy + (y - caixa.y0) * escala];
}

function caminhoOssos(q, tf) {
  let d = "";
  for (const [a, b] of OSSOS) {
    const p = tf(q[a][0], q[a][1]), r = tf(q[b][0], q[b][1]);
    d += `M${p[0].toFixed(1)} ${p[1].toFixed(1)}L${r[0].toFixed(1)} ${r[1].toFixed(1)}`;
  }
  return d;
}

function legenda(container, texto) {
  container.appendChild(html("p", { class: "legenda" }, Array.isArray(texto) ? texto : [texto]));
}

// <i>head banging</i>, sem o apoio em português
function en(id) {
  return nomeClasse({ id, nome: "" }, { apoio: false });
}

function rotuloTrilha(c) {
  return c.id === "background" ? "Fundo (sem estereotipia anotada)" : nomeClasse(c);
}

function visualQuadros(container) {
  const { sessao: s, pontos } = estado.video;
  const fps = s.sessao.fps;
  const ex = intervaloExemplo();
  const f0 = Math.min(s.sessao.total_quadros - 5, Math.round(ex.meio * fps));
  const idx = [0, 1, 2, 3, 4].map((k) => f0 + k);
  const quadros = idx.map((f) => pontos.quadros[f]);
  const tf = encaixar(caixaDosPontos(quadros), 104, 104, 8);
  const linha = html("div", { class: "quadros" });
  idx.forEach((f, k) => {
    const caixa = html("div", { class: "caixa" });
    if (quadros[k]) {
      caixa.appendChild(svg("svg", { width: 104, height: 104, "aria-hidden": "true" }, [
        svg("path", { d: caminhoOssos(quadros[k], tf), fill: "none", stroke: "#4A585C", "stroke-width": 4, "stroke-linecap": "round", "stroke-linejoin": "round" }),
      ]));
    } else {
      caixa.appendChild(html("span", { texto: "sem pessoa detectada" }));
    }
    linha.appendChild(html("div", { class: "quadro" }, [caixa, html("span", { class: "tc", texto: fmtTempoFino(f / fps) })]));
  });
  container.appendChild(linha);
  legenda(container, [
    `Cinco quadros seguidos de ${estado.video.id}, no meio do intervalo anotado de `, en(ex.classe.id),
    ` (${ex.anotacao.inicio_s} s a ${ex.anotacao.fim_s} s no XML). ` +
    "Desenhados a partir dos pontos exportados; quadro sem detecção fica vazio, sem preencher.",
  ]);
}

function visualPose(container) {
  const { sessao: s, pontos } = estado.video;
  const fps = s.sessao.fps;
  const ex = intervaloExemplo();
  let f = Math.round(ex.meio * fps);
  while (f < s.sessao.total_quadros && !pontos.quadros[f]) f++;
  if (f >= s.sessao.total_quadros) { f = Math.round(ex.meio * fps); while (f > 0 && !pontos.quadros[f]) f--; }
  const q = pontos.quadros[f];
  const tf = encaixar(caixaDosPontos([q]), 640, 310, 36);
  const p = q.map(([x, y]) => tf(x, y));
  const el = svg("svg", { width: 640, height: 360, "aria-hidden": "true" });
  el.appendChild(svg("path", { d: caminhoOssos(q, tf), fill: "none", stroke: "#1E2B2F", "stroke-width": 3, "stroke-linecap": "round" }));
  let juntas = "";
  for (const r of p) juntas += `M${r[0].toFixed(1)} ${r[1].toFixed(1)}l0.01 0`;
  el.appendChild(svg("path", { d: juntas, fill: "none", stroke: "#1E2B2F", "stroke-width": 11, "stroke-linecap": "round" }));
  el.appendChild(svg("path", { d: juntas, fill: "none", stroke: "#F6F8F7", "stroke-width": 6, "stroke-linecap": "round" }));
  container.appendChild(el);
  const ultimo = q.length - 1;
  p.forEach((r, n) => {
    if (n >= 1 && n <= 4) return;
    const dx = n === 0 ? -26 : (n % 2 === 1 ? 12 : -26);
    container.appendChild(html("span", {
      class: "num", texto: String(n),
      estilo: `position:absolute;left:${(r[0] + dx).toFixed(0)}px;top:${(r[1] - 10).toFixed(0)}px;font-size:12px;color:#4A585C`,
    }));
  });
  const rosto = p[2];
  container.appendChild(html("span", {
    texto: "1 a 4",
    estilo: `position:absolute;left:${(rosto[0] + 16).toFixed(0)}px;top:${(rosto[1] - 26).toFixed(0)}px;font-size:12px;color:#4A585C`,
  }));
  legenda(container,
    `${q.length} pontos por quadro, no formato COCO: 0 é o nariz, 1 a 4 olhos e orelhas, 5 a ${ultimo} o corpo. ` +
    `Quadro ${f} de ${estado.video.id} (${fmtTempoFino(f / fps)}), em pixels do vídeo original.`);
}

function visualJanelas(container) {
  const { sessao: s, pontos } = estado.video;
  const fps = s.sessao.fps;
  const ex = intervaloExemplo();
  const N = s.modelo.janela_quadros;
  const amostra = Math.round(N / 3);
  const PUNHO_ESQUERDO = 9;
  const fa = Math.round(ex.anotacao.inicio_s * fps);
  const fb = Math.min(s.sessao.total_quadros - 1, Math.round(Math.min(ex.anotacao.fim_s, ex.anotacao.inicio_s + 10) * fps));
  const ys = [];
  for (let f = fa; f <= fb; f++) {
    const q = pontos.quadros[f];
    if (q) ys.push(q[PUNHO_ESQUERDO][1]);
  }
  const yMin = Math.min(...ys), yMax = Math.max(...ys);
  const X = (f) => 20 + ((f - fa) / Math.max(1, fb - fa)) * 600;
  const Y = (y) => 22 + ((y - yMin) / Math.max(1, yMax - yMin)) * 200;

  const destaque = 4;
  const inicioDestaque = fa + destaque * amostra;
  container.appendChild(html("div", {
    estilo: `position:absolute;left:${X(inicioDestaque).toFixed(1)}px;top:22px;width:${(X(inicioDestaque + N) - X(inicioDestaque)).toFixed(1)}px;height:216px;background:#E1E8F2`,
  }));

  let d = "", aberto = false;
  for (let f = fa; f <= fb; f++) {
    const q = pontos.quadros[f];
    if (!q) { aberto = false; continue; }
    d += (aberto ? "L" : "M") + X(f).toFixed(1) + " " + Y(q[PUNHO_ESQUERDO][1]).toFixed(1);
    aberto = true;
  }
  container.appendChild(svg("svg", { width: 640, height: 360, "aria-hidden": "true", style: "position:absolute;left:0;top:0" }, [
    svg("path", { d, fill: "none", stroke: CORES[ex.classe.id], "stroke-width": 1.75, "stroke-linejoin": "round" }),
  ]));

  for (let k = 0; k < 9; k++) {
    const ini = fa + k * amostra;
    if (ini + N > fb) break;
    container.appendChild(html("div", {
      estilo: `position:absolute;left:${X(ini).toFixed(1)}px;top:${246 + (k % 3) * 12}px;width:${(X(ini + N) - X(ini)).toFixed(1)}px;height:9px;border:2px solid ${k === destaque ? "#1E2B2F" : "#AEB8B5"};border-top:0`,
    }));
  }
  container.appendChild(html("span", { class: "num", texto: fmtTempo(fa / fps), estilo: "position:absolute;left:20px;top:290px;font-size:12px;color:#4A585C" }));
  container.appendChild(html("span", { class: "num", texto: fmtTempo(fb / fps), estilo: "position:absolute;right:20px;top:290px;font-size:12px;color:#4A585C" }));
  legenda(container,
    `Altura do punho esquerdo (ponto ${PUNHO_ESQUERDO}) em ${estado.video.id}, para baixo é mais baixo na imagem; interrupções são quadros sem detecção. ` +
    `Abaixo, janelas de ${N} quadros: com passo ${s.modelo.passo_quadros}, começa uma janela em cada quadro; aqui aparece uma a cada ${amostra}.`);
}

function visualLSTM(container) {
  const { sessao: s } = estado.video;
  const m = s.modelo;
  const celulas = [["LSTM", "x₁"], ["LSTM", "x₂"], ["LSTM", "x₃"], ["…", ""], ["LSTM", "x" + subscrito(m.janela_quadros)]];
  const cadeia = html("div", { class: "cadeia" });
  for (const [rot, x] of celulas) {
    cadeia.appendChild(html("div", { class: "celula" }, [
      html("div", { class: "caixa" + (rot === "…" ? " tracejada" : ""), texto: rot }),
      html("span", { class: "x", texto: x }),
    ]));
    cadeia.appendChild(svg("svg", { width: 18, height: 48, "aria-hidden": "true" }, [
      svg("path", { d: "M1 24H15M10 19L15 24L10 29", fill: "none", stroke: "#1E2B2F", "stroke-width": 1.75, "stroke-linecap": "round", "stroke-linejoin": "round" }),
    ]));
  }
  cadeia.appendChild(html("div", { class: "saidas" }, s.classes.map((c) =>
    html("div", { class: "saida" }, [html("span", { class: "quadradinho", estilo: `width:10px;height:10px;background:${CORES[c.id]}` }), nomeClasse(c)])
  )));
  container.appendChild(html("div", { class: "lstm" }, [
    cadeia,
    html("p", {
      texto: `Cada caixa é o mesmo bloco LSTM (${m.lstm_camadas} camadas de ${m.lstm_unidades} unidades) aplicado a um quadro da janela, com ${m.n_caracteristicas} números por quadro. ` +
        "A seta leva o estado interno ao quadro seguinte; no fim da janela, esse estado alimenta a camada de saída.",
    }),
  ]));
}

function visualPontuacoes(container) {
  const { sessao: s } = estado.video;
  const fps = s.sessao.fps;
  const ex = intervaloExemplo();
  const j = janelaMaisProxima(s.janelas, fps, ex.meio);
  const longa = j.janela_longa
    ? ` Janela longa: por quadros sem detecção, ela cobre ${j.quadro_fim - j.quadro_inicio} quadros do vídeo para juntar ${s.modelo.janela_quadros} com pessoa.`
    : "";
  const barras = html("div", { class: "barras" }, [
    html("span", { class: "cab" }, [
      `Janela de ${fmtTempoFino(j.quadro_inicio / fps)} a ${fmtTempoFino(j.quadro_fim / fps)} de ${estado.video.id}; o XML anota `,
      en(ex.classe.id), ` de ${ex.anotacao.inicio_s} s a ${ex.anotacao.fim_s} s.${longa}`,
    ]),
  ]);
  s.classes.forEach((c, i) => {
    barras.appendChild(html("div", { class: "barra" }, [
      html("div", { class: "linha1" }, [html("span", {}, [rotuloTrilha(c)]), html("span", { class: "num", texto: num(j.pontuacoes[i], 2) })]),
      html("div", { class: "trilho" }, [html("div", { class: "valor", estilo: `width:${(j.pontuacoes[i] * 576).toFixed(0)}px;background:${CORES[c.id]}` })]),
    ]));
  });
  container.appendChild(barras);
  legenda(container, "Pontuações reais desta janela, fora da partição. É uma janela só: a revisão olha a sequência delas.");
}

function visualTrechos(container) {
  const { sessao: s } = estado.video;
  const fps = s.sessao.fps;
  const T = s.sessao.total_quadros / fps;
  const W = 576, ALT_ANOT = 6, ALT = 34;
  const X = (t) => (t / T) * W;
  const Y = (p) => ALT_ANOT + 4 + (1 - p) * (ALT - 10);
  const baseY = ALT_ANOT + ALT;
  const ids = s.classes.map((c) => c.id);
  const trechos = buildSegments(s.janelas, ids, s.limiar_padrao, s.duracao_minima_s, fps);
  const semDeteccao = sequencias(s.quadros_sem_deteccao);
  const bloco = html("div", { class: "mini" });

  s.classes.forEach((c, i) => {
    const neutra = c.id === "background";
    const cor = CORES[c.id];
    const altura = baseY + 4;
    const el = svg("svg", { width: W, height: altura, "aria-hidden": "true" });
    for (const [a, b] of semDeteccao) {
      el.appendChild(svg("rect", { x: X(a / fps).toFixed(1), y: 0, width: Math.max(1, X((b + 1) / fps) - X(a / fps)).toFixed(1), height: altura, fill: "#9AA5A2", "fill-opacity": 0.35 }));
    }
    if (!neutra) {
      for (const a of s.anotacoes.filter((x) => x.classe === c.id)) {
        el.appendChild(svg("rect", { x: X(a.inicio_s).toFixed(1), y: 0, width: Math.max(2, X(a.fim_s) - X(a.inicio_s)).toFixed(1), height: ALT_ANOT, fill: "none", stroke: cor, "stroke-width": 1.5 }));
      }
    }
    const d = caminhoCurva(s.janelas, fps, i, X, Y);
    el.appendChild(svg("path", { d: `${d}L${X(centroJanela(s.janelas[s.janelas.length - 1], fps)).toFixed(1)} ${baseY}L${X(centroJanela(s.janelas[0], fps)).toFixed(1)} ${baseY}Z`, fill: cor, "fill-opacity": 0.16 }));
    el.appendChild(svg("path", { d, fill: "none", stroke: cor, "stroke-width": 1.25, "stroke-linejoin": "round" }));
    if (!neutra) {
      el.appendChild(svg("line", { x1: 0, x2: W, y1: Y(s.limiar_padrao).toFixed(1), y2: Y(s.limiar_padrao).toFixed(1), stroke: "#4A585C", "stroke-width": 1, "stroke-dasharray": "4 4" }));
      for (const t of trechos.filter((x) => x.classe === c.id)) {
        el.appendChild(svg("rect", { x: X(t.inicio_s).toFixed(1), y: baseY - 2, width: Math.max(2, X(t.fim_s) - X(t.inicio_s)).toFixed(1), height: 5, rx: 2, fill: cor }));
      }
    }
    el.appendChild(svg("line", { x1: 0, x2: W, y1: baseY + 3, y2: baseY + 3, stroke: "#C3CBC8", "stroke-width": 1 }));
    bloco.appendChild(html("div", {}, [
      html("span", { class: "rot" }, [html("span", { class: "quadradinho", estilo: `width:10px;height:10px;background:${cor}` }), rotuloTrilha(c)]),
      el,
    ]));
  });
  container.appendChild(bloco);
  const nTrechos = trechos.filter((t) => t.classe !== "background").length;
  legenda(container,
    `${estado.video.id}, ${fmtTempo(T)} de vídeo, limiar ${num(s.limiar_padrao, 2)}: ${nTrechos} trechos sugeridos. ` +
    "Contorno no topo de cada classe: intervalo anotado no XML do SSBD. Barra na base: trecho sugerido pelo modelo. Cinza: sem pessoa detectada.");
}

// ---------- rodapé e achados posteriores ----------

function montarRodape() {
  const { resumo, comparacao } = estado.dados;
  const [m1, m2] = comparacao.metricas_principais;
  const rodape = $("rodape");
  const bloco = (titulo, filhos) => html("div", {}, [html("h3", { texto: titulo }), html("p", {}, filhos)]);
  const nClasses = Object.keys(resumo.comportamentos_por_categoria_videos_usados).length;
  // a frase sobre "pior" só vale se, de fato, os dois agregados do modelo
  // com rótulos por intervalo forem menores
  const ambasMenores = [m1, m2].every((m) => m.pooled_56_videos.rodada_a < m.pooled_56_videos.base_re_seedado);
  rodape.replaceChildren(
    bloco("Dados", [
      `SSBD: ${resumo.n_videos_usados} vídeos usados, de ${resumo.n_anotacoes_xml} anotações, com ${resumo.n_comportamentos_anotados_videos_usados} comportamentos anotados nesses vídeos, de ${nClasses} classes.`,
    ]),
    bloco("Validação", [
      `Divisão por vídeo em ${resumo.n_particoes} partições: cada vídeo é avaliado só pelo modelo da partição em que ficou fora do treino.`,
    ]),
    bloco("Resultados", [
      `Modelo com rótulos por intervalo contra o modelo base, pelo critério registrado antes da avaliação. ` +
      `${capitalizar(m1.metrica)}: ${num(m1.pooled_56_videos.rodada_a, 3)} contra ${num(m1.pooled_56_videos.base_re_seedado, 3)}. ` +
      `${capitalizar(m2.metrica)}: ${num(m2.pooled_56_videos.rodada_a, 3)} contra ${num(m2.pooled_56_videos.base_re_seedado, 3)}. Veredito: `,
      html("span", { class: "veredito", texto: comparacao.veredito }),
      ".",
      ambasMenores
        ? " O critério não previa o desfecho ‘pior’; nas duas métricas agregadas, os valores do modelo com rótulos por intervalo foram menores."
        : "",
    ]),
    bloco("Uso pretendido", ["Protótipo. Na avaliação deste trabalho, o desempenho não permite o uso para triagem ou diagnóstico."]),
  );
}

function capitalizar(t) {
  return t.charAt(0).toUpperCase() + t.slice(1);
}

function montarPosteriores() {
  const { posteriores, medianos, rodadaA } = estado.dados;
  $("posteriores-aviso").textContent =
    "Definidos depois de conhecer o resultado da comparação, para entender por que ele saiu assim. Não fazem parte do critério registrado e não mudam o veredito. Valores em média entre partições.";

  $("legenda-modelos").replaceChildren(...["base", "intervalos"].map((m) =>
    html("span", { class: "item" }, [
      html("span", { class: "amostra", estilo: `background:${CORES_MODELO[m]}` }),
      NOMES_MODELO[m],
    ])
  ));

  // chaves dos dois modelos em analises_posteriores.json
  const CHAVE = { intervalos: "rodada_a", base: "base_re_seedado" };
  const MODELOS = ["base", "intervalos"];
  const titulo = (m) => NOMES_MODELO[m].charAt(0).toUpperCase() + NOMES_MODELO[m].slice(1);
  const paineis = $("posteriores-paineis");
  paineis.replaceChildren();
  const painel = (tituloPainel, legendaPainel, montar, rodape) => {
    const area = html("div", { class: "grafico" });
    paineis.appendChild(html("figure", { class: "painel" }, [
      html("figcaption", {}, [html("span", { class: "titulo" }, tituloPainel), html("span", { class: "sub" }, legendaPainel)]),
      area,
      rodape ? html("p", { class: "frase" }, rodape) : null,
    ]));
    montar(area);
  };

  // a) presença de estereotipia: acurácia balanceada binária
  const bin = posteriores.analise_3_binario_estereotipia_vs_fundo;
  const acaso = 1 / Object.keys(bin.pooled_56_videos.report).filter((k) => k === "Estereotipia" || k === "Background").length;
  painel(["a) Presença de estereotipia"], ["Acurácia balanceada por quadro, qualquer estereotipia contra fundo."], (area) =>
    barrasHorizontais(area, {
      larguraRotulo: 78,
      referencia: { valor: acaso, rotulo: "acaso" },
      grupos: [{
        linhas: MODELOS.map((m) => {
          const v = bin.acuracia_balanceada_e_referencia[CHAVE[m]].acuracia_balanceada_media_entre_folds;
          return { rotulo: CURTO_MODELO[m], modelo: m, valor: v, dica: `${NOMES_MODELO[m]}: acurácia balanceada ${num(v, 3)} (acaso ${num(acaso, 3)})` };
        }),
      }],
    }));

  // b) ablações: F1 por quadro nos quadros anotados
  const ab = posteriores.analise_4_ablacoes_de_inferencia.ablacoes;
  const CHAVE_AB = { intervalos: "rodada_a_forcado_3_classes", base: "base_re_seedado_natural" };
  const f1 = (a, m) => a.macro_f1_quadro_anotados_media_entre_folds[CHAVE_AB[m]];
  const chance = rodadaA.por_quadro.somente_quadros_anotados.f1_chance_3_classes;
  const mudancaBase = Math.abs(f1(ab.shuffle, "base").com_ablacao - f1(ab.shuffle, "base").sem_ablacao);
  painel(["b) Ablações"], ["F1 por quadro nos quadros anotados, com a entrada alterada na inferência."], (area) =>
    barrasHorizontais(area, {
      larguraRotulo: 140,
      referencia: { valor: chance, rotulo: "F1 ao acaso" },
      grupos: MODELOS.map((m) => ({
        titulo: [titulo(m)],
        linhas: [
          ["sem ablação", f1(ab.shuffle, m).sem_ablacao],
          ["ordem embaralhada", f1(ab.shuffle, m).com_ablacao],
          ["deslocamentos zerados", f1(ab.zero_dxdy, m).com_ablacao],
        ].map(([rot, v]) => ({ rotulo: rot, modelo: m, valor: v, dica: `${NOMES_MODELO[m]}, ${rot}: F1 ${num(v, 3)}` })),
      })),
    }),
    [`No modelo base, embaralhar a ordem muda o F1 em ${num(mudancaBase, 3)}. No modelo com rótulos por intervalo, a predição fica restrita às estereotipias, sem fundo.`]);

  // fração dos quadros prevista como a classe do vídeo, dentro e fora dos episódios
  const fr = posteriores.analise_2_fracao_classe_do_video.media_por_classe_de_video;
  const linhasDentroFora = (classe, m) => {
    const x = fr[CHAVE[m]][classe];
    const nome = NOMES_EN[classe.toLowerCase()];
    return [
      { esquerda: CURTO_MODELO[m], rotulo: "dentro", modelo: m, valor: x.media_fracao_anotados, junto: true,
        dica: `${NOMES_MODELO[m]}: ${num(x.media_fracao_anotados, 3)} dos quadros dentro dos episódios anotados de ${nome} previstos como ${nome}` },
      { rotulo: "fora", modelo: m, valor: x.media_fracao_fundo, fora: true,
        dica: `${NOMES_MODELO[m]}: ${num(x.media_fracao_fundo, 3)} dos quadros fora dos episódios, nos vídeos de ${nome}, previstos como ${nome}` },
    ];
  };

  // c) head banging
  const rankingHB = medianos.ranking_por_classe.HeadBanging;
  const zerosHB = rankingHB.filter((r) => r.f1 === 0).length;
  painel(["c) ", en("headbanging")],
    ["Fração dos quadros dos vídeos de ", en("headbanging"), " prevista como ", en("headbanging"), ", dentro e fora dos episódios anotados."],
    (area) => barrasHorizontais(area, {
      larguraRotulo: 78,
      grupos: MODELOS.map((m) => ({
        titulo: [titulo(m)],
        linhas: linhasDentroFora("HeadBanging", m).map((l) => ({ ...l, esquerda: undefined })),
        nota: m === "intervalos" ? `F1 zero em ${zerosHB} de ${rankingHB.length} vídeos` : null,
      })),
    }));

  // d) classe do vídeo, por classe e modelo
  const classes = Object.keys(fr.rodada_a);
  painel(["d) Classe do vídeo dentro e fora dos episódios"],
    ["Fração dos quadros prevista como a classe do próprio vídeo."],
    (area) => barrasHorizontais(area, {
      larguraRotulo: 128,
      grupos: classes.map((c) => ({
        titulo: [{ t: NOMES_EN[c.toLowerCase()], italico: true }],
        linhas: MODELOS.flatMap((m) => linhasDentroFora(c, m)),
      })),
    }));
}

// ---------- matrizes de confusão (resultado registrado) ----------

function montarMatrizes() {
  const { baseOriginal, rodadaA, resumo } = estado.dados;
  const blocos = [
    ["Modelo base (treino original)", baseOriginal.por_quadro.somente_quadros_anotados.pooled_56_videos],
    ["Modelo com rótulos por intervalo", rodadaA.por_quadro.somente_quadros_anotados.pooled_56_videos],
  ];
  const labels = blocos[0][1].labels;
  if (!blocos.every(([, b]) => b.labels.join() === labels.join())) throw new Error("As matrizes de confusão usam rótulos diferentes.");
  const anotadas = labels.filter((l) => l !== "Background");
  const rotulo = (l) => (l === "Background" ? "fundo" : en(l.toLowerCase()));
  const nomeTexto = (l) => (l === "Background" ? "fundo" : NOMES_EN[l.toLowerCase()]);
  const grade = $("matrizes-grade");
  grade.replaceChildren(...blocos.map(([titulo, b]) => mapaDeCalor({
    titulo,
    linhas: anotadas.map((l) => ({ rotulo: rotulo(l) })),
    colunas: labels.map((l) => ({ rotulo: rotulo(l) })),
    matriz: anotadas.map((l) => b.confusion_matrix[labels.indexOf(l)]),
    descricaoCelula: (i, j, n, f, total) =>
      `${titulo}: ${inteiro(n)} de ${inteiro(total)} quadros anotados como ${nomeTexto(anotadas[i])} previstos como ${nomeTexto(labels[j])} (${num(f, 3)} da linha)`,
  })));
  grade.appendChild(escalaSequencial());

  const iFundo = labels.indexOf("Background");
  const baseSemFundo = anotadas.every((l) => blocos[0][1].confusion_matrix[labels.indexOf(l)][iFundo] === 0);
  $("matrizes-nota").textContent =
    `Predições fora da partição, quadros somados sobre os ${resumo.n_videos_usados} vídeos, só os que estão dentro dos intervalos anotados. ` +
    "Linhas: classe anotada; colunas: classe prevista. Cor: fração da linha; em cada célula, o número de quadros e essa fração. " +
    "No modelo base, fundo é a predição de qualquer uma das classes sem exemplo de treino" +
    (baseSemFundo ? "; ele não previu fundo em nenhum quadro anotado." : ".") +
    " Mesmos números da tabela de confusão do texto.";
}

iniciar();
