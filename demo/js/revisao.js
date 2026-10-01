import { buildSegments } from "./timeline.js";
import {
  CORES, RESULTADOS, carregarJSON, carregarSessao, num, fmtTempo, centroJanela,
  sequencias, trechosJanelasLongas, caminhoCurva, svg, html,
  desenharEsqueleto, mostrarErro,
} from "./comum.js";

const W = 1140;
const ALT_CLASSE = 48;
const ALT_DETECCAO = 24;
const INICIO_PLOT = 220; // 200 do rótulo + 20 de espaço, como no design
const VIDEO_W = 720;
const VIDEO_H = 405;
const PASSOS_ESCALA = [5, 10, 15, 30, 60, 120, 300, 600];

const $ = (id) => document.getElementById(id);

const estado = {
  sessao: null,
  pontos: null,
  fps: 30,
  T: 0,
  total: 0,
  t: 0,
  tocando: false,
  ultimoRelogio: 0,
  limiar: 0.5,
  segmentos: [],
  temVideo: false,
  urlVideo: null,
  ultimoQuadroDesenhado: -2,
  arrastando: false,
  classes: [],
  reais: [],
  indiceFundo: -1,
  refs: { trilhas: [] },
};

function chaveArmazenamento() {
  return `tcc-demo:revisoes:${estado.sessao.sessao.id}`;
}

function lerDecisoes() {
  try {
    return JSON.parse(localStorage.getItem(chaveArmazenamento()) || "{}");
  } catch {
    return {};
  }
}

function gravarDecisoes(d) {
  try {
    localStorage.setItem(chaveArmazenamento(), JSON.stringify(d));
  } catch {
    $("status-video").textContent = "Não foi possível salvar a decisão neste navegador (armazenamento indisponível).";
  }
}

function chaveTrecho(s, limiar) {
  return `${limiar.toFixed(2)}|${s.classe}|${s.quadro_inicio}|${s.quadro_fim}`;
}

const X = (t) => (t / estado.T) * W;

async function iniciar() {
  const conteudo = $("conteudo");
  try {
    const [medianos, rodadaA] = await Promise.all([
      carregarJSON(RESULTADOS.medianos),
      carregarJSON(RESULTADOS.rodadaA),
    ]);

    const escolhidos = Object.values(medianos.escolhidos);
    const ids = escolhidos.map((e) => e.video_id);
    const pedido = new URLSearchParams(location.search).get("sessao");
    const id = ids.includes(pedido) ? pedido : ids[0];

    const sel = $("sel-sessao");
    for (const e of escolhidos) {
      sel.appendChild(html("option", { value: e.video_id, texto: `${e.video_id} (${e.classe_do_video})` }));
    }
    sel.value = id;
    sel.addEventListener("change", () => {
      location.search = "?sessao=" + encodeURIComponent(sel.value);
    });
    $("link-apresentacao").href = "apresentacao.html?video=" + encodeURIComponent(id);

    const { sessao, pontos } = await carregarSessao(id);
    estado.sessao = sessao;
    estado.pontos = pontos;
    estado.fps = sessao.sessao.fps;
    estado.total = sessao.sessao.total_quadros;
    estado.T = estado.total / estado.fps;
    estado.classes = sessao.classes;
    estado.indiceFundo = sessao.classes.findIndex((c) => c.id === "background");
    estado.reais = sessao.classes.map((c, i) => ({ ...c, indice: i })).filter((c) => c.id !== "background");

    const limiares = rodadaA.por_evento.grade_completa.limiares;
    const faixa = $("limiar");
    faixa.min = limiares[0];
    faixa.max = limiares[limiares.length - 1];
    faixa.step = Math.round((limiares[1] - limiares[0]) * 100) / 100;
    faixa.value = sessao.limiar_padrao;
    estado.limiar = parseFloat(faixa.value);

    montarCabecalho();
    montarLinhaDoTempo();
    montarControles();
    recalcularTrechos();
    atualizarQuadro();
    estado.ultimoRelogio = performance.now();
    requestAnimationFrame(laco);
  } catch (e) {
    mostrarErro(conteudo, e.message + " Rode os exportadores (src/exportar_sessao.py, src/exportar_pontos.py) e sirva a interface com python demo/servir.py.");
    $("sessao-info").textContent = "";
  }
}

function montarCabecalho() {
  const s = estado.sessao;
  $("sessao-info").textContent =
    `Sessão ${s.sessao.id}, ${fmtTempo(estado.T)} de vídeo, ${s.sessao.total_quadros} quadros a ${num(estado.fps, 2)} quadros por segundo`;
  $("tc-total").textContent = fmtTempo(estado.T);

  const aviso = $("aviso-particao");
  aviso.replaceChildren(
    html("strong", { texto: "Predições fora da partição." }),
    html("span", {
      texto: ` Este vídeo do SSBD foi avaliado pelo modelo da partição ${s.modelo.particao} (${s.modelo.checkpoint}), treinado sem ele. As pontuações são as que o modelo daria a um vídeo novo.`,
    })
  );
  aviso.hidden = false;
}

function montarLinhaDoTempo() {
  const s = estado.sessao;
  const fps = estado.fps;
  const janelas = s.janelas;
  const trilhas = $("trilhas");
  trilhas.replaceChildren();

  const semDeteccao = sequencias(s.quadros_sem_deteccao);
  const longas = trechosJanelasLongas(janelas, fps);
  const nLongas = janelas.filter((j) => j.janela_longa).length;

  $("legenda-linha").textContent =
    "A curva é a pontuação do modelo em cada janela, marcada no centro dela; a linha tracejada é o limiar e a barra na base marca o trecho sugerido. " +
    `Cinza: quadros sem pessoa detectada. Hachurado: janelas que cobrem mais de ${s.limite_janela_longa_quadros} quadros reais, por causa de quadros sem detecção.`;

  const retangulosSemDeteccao = (altura, opacidade, cor) =>
    semDeteccao.map(([a, b]) => {
      const x = X(a / fps);
      return svg("rect", {
        x: x.toFixed(1), y: 0, width: Math.max(1, X((b + 1) / fps) - x).toFixed(1), height: altura,
        fill: cor, "fill-opacity": opacidade,
      });
    });

  // trilha de detecção
  {
    const linha = html("div", { class: "trilha", estilo: `height:${ALT_DETECCAO + 8}px` });
    linha.appendChild(html("div", { class: "trilha-rotulo" }, [
      html("span", { class: "textos" }, [
        html("span", { class: "nome", texto: "Detecção" }),
        html("span", { class: "resumo num", texto: `${s.quadros_sem_deteccao.length} sem pessoa, ${nLongas} longas` }),
      ]),
    ]));
    const plot = html("div", { class: "trilha-plot", estilo: `height:${ALT_DETECCAO + 8}px` });
    const el = svg("svg", { width: W, height: ALT_DETECCAO + 8, "aria-hidden": "true" });
    const defs = svg("defs", {}, [
      svg("pattern", { id: "hachura", width: 6, height: 6, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" }, [
        svg("rect", { width: 6, height: 6, fill: "#F6F8F7" }),
        svg("line", { x1: 0, y1: 0, x2: 0, y2: 6, stroke: "#1E2B2F", "stroke-width": 2 }),
      ]),
    ]);
    el.appendChild(defs);
    el.appendChild(svg("rect", { x: 0, y: 4, width: W, height: ALT_DETECCAO, fill: "#DCE2E0" }));
    for (const r of retangulosSemDeteccao(ALT_DETECCAO / 2, 1, "#9AA5A2")) { r.setAttribute("y", 4); el.appendChild(r); }
    for (const [a, b] of longas) {
      const x = X(a);
      el.appendChild(svg("rect", {
        x: x.toFixed(1), y: 4 + ALT_DETECCAO / 2, width: Math.max(2, X(b) - x).toFixed(1), height: ALT_DETECCAO / 2,
        fill: "url(#hachura)", stroke: "#1E2B2F", "stroke-width": 0.5,
      }));
    }
    plot.appendChild(el);
    linha.appendChild(plot);
    trilhas.appendChild(linha);
  }

  const Y = (p) => 6 + (1 - p) * (ALT_CLASSE - 18);
  const base = ALT_CLASSE - 12;

  const novaTrilha = (classe, neutra) => {
    const cor = CORES[classe.id] || "#7F8C89";
    const linha = html("div", { class: "trilha" + (neutra ? " neutra" : ""), estilo: `height:${ALT_CLASSE}px` });
    const resumo = html("span", { class: "resumo num" });
    linha.appendChild(html("div", { class: "trilha-rotulo" }, [
      html("span", { class: "quadradinho", estilo: `background:${cor}` }),
      html("span", { class: "textos" }, [html("span", { class: "nome", texto: classe.nome }), resumo]),
    ]));
    const plot = html("div", { class: "trilha-plot", estilo: `height:${ALT_CLASSE}px` });
    const el = svg("svg", { width: W, height: ALT_CLASSE, "aria-hidden": "true" });
    for (const r of retangulosSemDeteccao(ALT_CLASSE, 0.35, "#9AA5A2")) el.appendChild(r);
    const d = caminhoCurva(janelas, fps, classe.indice, X, Y);
    const x0 = X(centroJanela(janelas[0], fps)).toFixed(1);
    const x1 = X(centroJanela(janelas[janelas.length - 1], fps)).toFixed(1);
    el.appendChild(svg("path", { d: `${d}L${x1} ${base}L${x0} ${base}Z`, fill: cor, "fill-opacity": 0.16 }));
    el.appendChild(svg("path", { d, fill: "none", stroke: cor, "stroke-width": 1.5, "stroke-linejoin": "round" }));
    let linhaLimiar = null;
    if (!neutra) {
      linhaLimiar = svg("line", { x1: 0, x2: W, stroke: "#4A585C", "stroke-width": 1, "stroke-dasharray": "4 4" });
      el.appendChild(linhaLimiar);
    }
    const bandas = svg("g");
    el.appendChild(bandas);
    plot.appendChild(el);
    linha.appendChild(plot);
    trilhas.appendChild(linha);
    return { classe, cor, resumo, linhaLimiar, bandas, Y };
  };

  estado.refs.trilhas = estado.reais.map((c) => novaTrilha(c, false));
  if (estado.indiceFundo >= 0) {
    const fundo = { ...estado.classes[estado.indiceFundo], indice: estado.indiceFundo };
    const ref = novaTrilha(fundo, true);
    ref.resumo.textContent = "pontuação de fundo";
  }

  // escala de tempo
  const escala = $("escala");
  escala.replaceChildren();
  const passo = PASSOS_ESCALA.find((p) => estado.T / p <= 8) || PASSOS_ESCALA[PASSOS_ESCALA.length - 1];
  for (let s2 = 0; s2 <= estado.T + 1e-9; s2 += passo) {
    const ultimo = s2 + passo > estado.T + 1e-9;
    const tx = s2 === 0 ? "0" : (ultimo && X(s2) > W - 30 ? "-100%" : "-50%");
    escala.appendChild(html("span", { texto: fmtTempo(s2), estilo: `left:${X(s2).toFixed(1)}px;transform:translateX(${tx})` }));
  }

  const scrub = $("scrub");
  scrub.max = estado.T;
  scrub.step = 1 / estado.fps;
  scrub.value = 0;
}

function recalcularTrechos() {
  const s = estado.sessao;
  const ids = s.classes.map((c) => c.id);
  const todos = buildSegments(s.janelas, ids, estado.limiar, s.duracao_minima_s, estado.fps);
  estado.segmentos = todos.filter((t) => t.classe !== "background").sort((a, b) => a.inicio_s - b.inicio_s || a.classe.localeCompare(b.classe));
  estado.escolhido = -1;
  $("limiar-txt").textContent = num(estado.limiar, 2);

  for (const ref of estado.refs.trilhas) {
    const y = ref.Y(estado.limiar).toFixed(1);
    ref.linhaLimiar.setAttribute("y1", y);
    ref.linhaLimiar.setAttribute("y2", y);
    ref.bandas.replaceChildren();
    const meus = estado.segmentos.filter((t) => t.classe === ref.classe.id);
    for (const t of meus) {
      const x = X(t.inicio_s);
      ref.bandas.appendChild(svg("rect", {
        x: x.toFixed(1), y: ALT_CLASSE - 8, width: Math.max(2, X(t.fim_s) - x).toFixed(1), height: 6, rx: 2, fill: ref.cor,
      }));
    }
    const total = meus.reduce((acc, t) => acc + t.duracao_s, 0);
    ref.resumo.textContent = meus.length === 0
      ? "Nenhum trecho"
      : (meus.length === 1 ? "1 trecho, " : `${meus.length} trechos, `) + `${num(total, 1)} s`;
  }

  renderizarLista();
  if (estado.pontos) atualizarQuadro();
}

function renderizarLista() {
  const lista = $("lista-trechos");
  lista.replaceChildren();
  const decisoes = lerDecisoes();
  const nomes = Object.fromEntries(estado.classes.map((c) => [c.id, c.nome]));
  let revisados = 0;

  estado.segmentos.forEach((t, i) => {
    const chave = chaveTrecho(t, estado.limiar);
    const dec = decisoes[chave];
    if (dec) revisados++;
    const definir = (v) => () => {
      const d = lerDecisoes();
      if (v) d[chave] = v; else delete d[chave];
      gravarDecisoes(d);
      renderizarLista();
    };
    const acoes = dec
      ? html("div", { class: "trecho-acoes" }, [
          html("span", { class: "status", texto: dec === "confirmado" ? "Confirmado" : "Descartado" }),
          html("button", { type: "button", class: "botao botao-link", texto: "Desfazer", onclick: definir(null) }),
        ])
      : html("div", { class: "trecho-acoes" }, [
          html("button", { type: "button", class: "botao botao-primario", texto: "Confirmar", onclick: definir("confirmado") }),
          html("button", { type: "button", class: "botao botao-secundario", texto: "Descartar", onclick: definir("descartado") }),
        ]);
    lista.appendChild(html("div", { class: "trecho", "data-indice": i }, [
      html("span", { class: "quadradinho", estilo: `background:${CORES[t.classe]}` }),
      html("button", { type: "button", class: "trecho-ir", onclick: () => { estado.escolhido = i; irPara((t.quadro_inicio + 0.5) / estado.fps); } }, [
        html("span", { class: "nome", texto: nomes[t.classe] }),
        html("span", {
          class: "faixa",
          texto: `${fmtTempo(t.inicio_s)} a ${fmtTempo(t.fim_s)}, ${num(t.duracao_s, 1)} s, pico ${num(t.pico, 2)}`,
        }),
      ]),
      acoes,
    ]));
  });

  if (!estado.segmentos.length) {
    lista.appendChild(html("p", {
      class: "vazio",
      texto: "Nenhum trecho acima do limiar atual. Baixe o limiar na linha do tempo para ver trechos com pontuação menor.",
    }));
  }

  $("n-revisados").textContent = `${revisados} de ${estado.segmentos.length} revisados`;
}

function irPara(t) {
  const video = $("video");
  if (estado.temVideo) {
    video.pause();
    video.currentTime = t;
  } else {
    estado.tocando = false;
  }
  estado.t = t;
  atualizarIconePlay();
  atualizarQuadro();
}

function atualizarIconePlay() {
  const tocando = estado.temVideo ? !$("video").paused : estado.tocando;
  $("icone-play").setAttribute("d", tocando ? "M4 3h3.5v12H4zM10.5 3H14v12h-3.5z" : "M5 2.5L15.5 9L5 15.5z");
  $("btn-play").setAttribute("aria-label", tocando ? "Pausar" : "Reproduzir");
}

function montarControles() {
  const video = $("video");

  $("btn-play").addEventListener("click", () => {
    if (estado.temVideo) {
      if (video.paused) {
        if (video.ended) video.currentTime = 0;
        video.play();
      } else {
        video.pause();
      }
    } else {
      if (!estado.tocando && estado.t >= estado.T) estado.t = 0;
      estado.tocando = !estado.tocando;
      estado.ultimoRelogio = performance.now();
    }
    atualizarIconePlay();
  });
  video.addEventListener("play", atualizarIconePlay);
  video.addEventListener("pause", atualizarIconePlay);

  $("limiar").addEventListener("input", (e) => {
    estado.limiar = parseFloat(e.target.value);
    recalcularTrechos();
  });

  const scrub = $("scrub");
  scrub.addEventListener("input", () => { estado.arrastando = true; irPara(parseFloat(scrub.value)); estado.arrastando = false; });
  scrub.addEventListener("pointerdown", () => { estado.arrastando = true; });
  scrub.addEventListener("pointerup", () => { estado.arrastando = false; });

  $("chk-esqueleto").addEventListener("change", () => { estado.ultimoQuadroDesenhado = -2; atualizarQuadro(); });
  video.addEventListener("seeked", atualizarQuadro);

  $("btn-video").addEventListener("click", () => $("arq-video").click());
  $("arq-video").addEventListener("change", (e) => abrirVideo(e.target.files[0]));

  $("btn-exportar").addEventListener("click", exportarRevisoes);
}

function abrirVideo(arquivo) {
  if (!arquivo) return;
  const video = $("video");
  const status = $("status-video");
  if (estado.urlVideo) URL.revokeObjectURL(estado.urlVideo);
  // o arquivo fica só na memória deste navegador: não é enviado nem copiado
  estado.urlVideo = URL.createObjectURL(arquivo);
  video.src = estado.urlVideo;
  estado.temVideo = true;
  estado.tocando = false;
  $("placeholder-video").hidden = true;

  const esperado = estado.sessao.sessao.video;
  const avisos = [];
  if (arquivo.name !== esperado) avisos.push(`o nome do arquivo não é ${esperado}`);

  video.addEventListener("loadedmetadata", () => {
    video.currentTime = Math.min(estado.t, video.duration || 0);
    if (Math.abs(video.duration - estado.T) > 1) {
      avisos.push(`a duração (${fmtTempo(video.duration)}) não bate com a da sessão (${fmtTempo(estado.T)})`);
    }
    if (video.videoWidth !== estado.sessao.sessao.largura || video.videoHeight !== estado.sessao.sessao.altura) {
      avisos.push(`a resolução (${video.videoWidth}×${video.videoHeight}) não bate com a da sessão (${estado.sessao.sessao.largura}×${estado.sessao.sessao.altura})`);
    }
    status.className = avisos.length ? "alerta" : "secundario";
    status.textContent = avisos.length
      ? `Atenção: ${avisos.join("; ")}. Confira se é o mesmo vídeo da sessão.`
      : `${arquivo.name} aberto só neste navegador.`;
    estado.ultimoQuadroDesenhado = -2;
  }, { once: true });

  status.className = "secundario";
  status.textContent = `Abrindo ${arquivo.name}…`;
  atualizarIconePlay();
}

function exportarRevisoes() {
  const decisoes = lerDecisoes();
  const atuais = estado.segmentos.map((t) => ({
    classe: t.classe,
    inicio_s: t.inicio_s,
    fim_s: t.fim_s,
    pico: t.pico,
    decisao: decisoes[chaveTrecho(t, estado.limiar)] || "pendente",
  }));
  const chavesAtuais = new Set(estado.segmentos.map((t) => chaveTrecho(t, estado.limiar)));
  const outras = Object.entries(decisoes)
    .filter(([k]) => !chavesAtuais.has(k))
    .map(([k, v]) => {
      const [limiar, classe, qi, qf] = k.split("|");
      return {
        limiar: parseFloat(limiar), classe,
        inicio_s: parseInt(qi, 10) / estado.fps, fim_s: parseInt(qf, 10) / estado.fps, decisao: v,
      };
    });
  const conteudo = {
    sessao: estado.sessao.sessao.id,
    limiar: estado.limiar,
    trechos: atuais,
    decisoes_em_outros_limiares: outras,
  };
  const blob = new Blob([JSON.stringify(conteudo, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = html("a", { href: url, download: `${estado.sessao.sessao.id}_revisoes.json` });
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function transformarPonto() {
  const video = $("video");
  const largura = (estado.temVideo && video.videoWidth) || estado.sessao.sessao.largura;
  const altura = (estado.temVideo && video.videoHeight) || estado.sessao.sessao.altura;
  const escala = Math.min(VIDEO_W / largura, VIDEO_H / altura);
  const ox = (VIDEO_W - largura * escala) / 2;
  const oy = (VIDEO_H - altura * escala) / 2;
  return (x, y) => [ox + x * escala, oy + y * escala];
}

function laco(agora) {
  const video = $("video");
  if (estado.temVideo) {
    estado.t = video.currentTime;
  } else if (estado.tocando) {
    estado.t += (agora - estado.ultimoRelogio) / 1000;
    if (estado.t >= estado.T) {
      estado.t = estado.T;
      estado.tocando = false;
      atualizarIconePlay();
    }
  }
  estado.ultimoRelogio = agora;
  atualizarQuadro();
  requestAnimationFrame(laco);
}

// Janelas cujo intervalo [quadro_inicio, quadro_fim) contém o quadro.
// As janelas estão em ordem de quadro_inicio; uma janela longa pode
// cobrir mais quadros que o normal, por isso o recuo usa o maior vão.
function janelasQueCobrem(quadro) {
  const janelas = estado.sessao.janelas;
  if (estado.maiorVao === undefined) {
    estado.maiorVao = janelas.reduce((m, j) => Math.max(m, j.quadro_fim - j.quadro_inicio), 0);
  }
  let lo = 0, hi = janelas.length;
  while (lo < hi) {
    const meio = (lo + hi) >> 1;
    if (janelas[meio].quadro_inicio <= quadro) lo = meio + 1;
    else hi = meio;
  }
  const out = [];
  for (let i = lo - 1; i >= 0 && janelas[i].quadro_inicio > quadro - estado.maiorVao; i--) {
    if (janelas[i].quadro_fim > quadro) out.push(janelas[i]);
  }
  return out;
}

function atualizarQuadro() {
  const t = estado.t;
  $("tc").textContent = fmtTempo(t);
  if (!estado.arrastando) $("scrub").value = t;

  const trilhas = $("trilhas");
  const cabeca = $("cabeca");
  cabeca.style.left = (INICIO_PLOT + X(Math.min(t, estado.T))).toFixed(1) + "px";
  cabeca.style.top = trilhas.offsetTop + "px";
  cabeca.style.height = trilhas.offsetHeight + "px";

  const quadro = Math.min(estado.total - 1, Math.max(0, Math.floor(t * estado.fps)));
  const mostrar = $("chk-esqueleto").checked;
  if (quadro !== estado.ultimoQuadroDesenhado) {
    const g = $("esqueleto-g");
    const pontosQuadro = estado.pontos.quadros[quadro];
    if (mostrar) desenharEsqueleto(g, pontosQuadro, transformarPonto());
    else g.replaceChildren();
    $("sem-pessoa").hidden = !(mostrar && pontosQuadro === null);
    estado.ultimoQuadroDesenhado = quadro;
  }

  // trechos da mesma classe podem se sobrepor (cada um vai até o fim da
  // sua última janela); o escolhido na lista tem prioridade enquanto
  // contém o quadro atual
  const contem = (s) => s && quadro >= s.quadro_inicio && quadro < s.quadro_fim;
  const ativo = contem(estado.segmentos[estado.escolhido])
    ? estado.escolhido
    : estado.segmentos.findIndex(contem);
  const chip = $("chip");
  if (ativo >= 0) {
    const s = estado.segmentos[ativo];
    const classe = estado.classes.findIndex((c) => c.id === s.classe);
    // a maior pontuação entre as janelas ativas do trecho que cobrem
    // este quadro (o trecho é a união dessas janelas)
    let p = -1;
    for (const j of janelasQueCobrem(quadro)) {
      const v = j.pontuacoes[classe];
      if (v >= estado.limiar && j.quadro_inicio >= s.quadro_inicio && j.quadro_fim <= s.quadro_fim && v > p) p = v;
    }
    $("chip-cor").style.background = CORES[s.classe];
    $("chip-nome").textContent = estado.classes[classe].nome;
    $("chip-p").textContent = p >= 0 ? num(p, 2) : "—";
    chip.hidden = false;
  } else {
    chip.hidden = true;
  }
  for (const el of document.querySelectorAll(".trecho")) {
    el.classList.toggle("ativo", Number(el.dataset.indice) === ativo);
  }
}

iniciar();
