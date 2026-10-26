// Interface simplificada: modos "Vídeos", "Câmera ao vivo" e "Como
// funciona". Os dados dos vídeos vêm de saidas/final/ (exportados por
// src/exportar_sessoes_final.py); o arquivo de vídeo é aberto pelo
// usuário e fica só na memória do navegador.

import { CORES, NOMES_EN, carregarJSON, desenharEsqueleto, fmtTempo, html, mostrarErro, nomeClasse, num } from "./comum.js";
import { JUNTAR_S, MEDIANA_S, finalSegments } from "./timeline.js";

const $ = (id) => document.getElementById(id);
const CLASSES = ["armflapping", "headbanging", "spinning"];
const LIMIAR_CONF = 0.5;    // pontos abaixo disso aparecem esmaecidos
const SUAVIZACAO_S = 0.2;   // média móvel centrada do esqueleto, só na exibição
const QUADROS_POR_S = 15;   // envio da câmera
const params = new URLSearchParams(location.search);

// ---------- modos ----------

function mostrarModo(modo) {
  for (const b of document.querySelectorAll("[data-modo]")) {
    if (b.dataset.modo === modo) b.setAttribute("aria-current", "page");
    else b.removeAttribute("aria-current");
  }
  for (const m of ["videos", "camera", "como"]) $("modo-" + m).hidden = m !== modo;
  if (modo !== "videos") pausar();
  if (modo !== "camera") desligar();
  const u = new URL(location.href);
  u.searchParams.set("modo", modo);
  history.replaceState(null, "", u);
}

for (const b of document.querySelectorAll("[data-modo]")) b.addEventListener("click", () => mostrarModo(b.dataset.modo));

// o palco tem a proporção da imagem e no máximo 50% da altura da janela
function ajustarPalco(palco, largura, altura) {
  palco.style.aspectRatio = `${largura} / ${altura}`;
  palco.style.maxWidth = `calc(50vh * ${largura} / ${altura})`;
}

// ---------- Vídeos ----------

const est = { sessao: null, pontos: null, suaves: null, fps: 30, T: 0, t: 0, tocando: false, inicio: 0, t0: 0, ultimo: null, url: null };

// média móvel centrada de SUAVIZACAO_S por ponto, só sobre quadros com
// pessoa e ponto confiável; a confiança mostrada é a do próprio quadro
function suavizar(quadros, fps) {
  const h = Math.round((SUAVIZACAO_S * fps) / 2);
  const n = quadros.length;
  const sx = new Float64Array((n + 1) * 17), sy = new Float64Array((n + 1) * 17), sc = new Float64Array((n + 1) * 17);
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 17; k++) {
      const a = i * 17 + k, b = a + 17;
      const p = quadros[i] && quadros[i][k];
      const ok = p && p[2] >= LIMIAR_CONF;
      sx[b] = sx[a] + (ok ? p[0] : 0);
      sy[b] = sy[a] + (ok ? p[1] : 0);
      sc[b] = sc[a] + (ok ? 1 : 0);
    }
  }
  return quadros.map((q, i) => {
    if (!q) return null;
    const a = Math.max(0, i - h) * 17, b = Math.min(n, i + h + 1) * 17;
    return q.map((p, k) => {
      const c = sc[b + k] - sc[a + k];
      return c > 0 ? [(sx[b + k] - sx[a + k]) / c, (sy[b + k] - sy[a + k]) / c, p[2]] : p;
    });
  });
}

// trechos contínuos de quadros sem pessoa: [[início, fim), ...]
function lacunas(quadros) {
  const out = [];
  let i0 = null;
  quadros.forEach((q, i) => {
    if (!q && i0 === null) i0 = i;
    if (q && i0 !== null) { out.push([i0, i]); i0 = null; }
  });
  if (i0 !== null) out.push([i0, quadros.length]);
  return out;
}

async function iniciarVideos() {
  const indice = await carregarJSON("/saidas/final/indice.json");
  const sel = $("sel-video");
  for (const c of indice.classes) {
    sel.appendChild(html("optgroup", { label: c.nome }, c.videos.map((v) => html("option", { value: v.id, texto: v.rotulo }))));
  }
  const pedido = params.get("video");
  if (pedido && [...sel.options].some((o) => o.value === pedido)) sel.value = pedido;
  sel.addEventListener("change", () => carregarVideo(sel.value));
  $("limiar").addEventListener("input", () => { $("limiar").dataset.mexido = "1"; desenharLinhaTempo(); detalhes(); });
  $("suavizar").addEventListener("change", () => { est.ultimo = null; });
  $("tocar").addEventListener("click", () => (est.tocando ? pausar() : tocar()));
  $("arquivo").addEventListener("change", abrirArquivo);
  await carregarVideo(sel.value);
  requestAnimationFrame(laco);
}

async function carregarVideo(id) {
  pausar();
  fecharArquivo();
  const [sessao, pontos] = await Promise.all([
    carregarJSON(`/saidas/final/${id}_sessao.json`),
    carregarJSON(`/saidas/final/${id}_pontos.json`),
  ]);
  Object.assign(est, { sessao, pontos, fps: sessao.sessao.fps, T: sessao.sessao.total_quadros / sessao.sessao.fps, t: 0, ultimo: null });
  est.suaves = suavizar(pontos.quadros, est.fps);
  est.lacunas = lacunas(pontos.quadros);
  const { largura, altura } = sessao.sessao;
  $("sobre-video").setAttribute("viewBox", `0 0 ${largura} ${altura}`);
  ajustarPalco($("palco-video"), largura, altura);
  if (!$("limiar").dataset.mexido) $("limiar").value = sessao.limiar_padrao;
  const u = new URL(location.href);
  u.searchParams.set("video", id);
  history.replaceState(null, "", u);
  desenharLinhaTempo();
  detalhes();
}

function trechos() {
  const s = est.sessao;
  return finalSegments(s.janelas, s.classes.map((c) => c.id), Number($("limiar").value), est.fps,
    s.posprocessamento.mediana_s, s.posprocessamento.juntar_s);
}

function faixa(classe, inicio, fim, extra = {}) {
  return html("div", {
    class: classe,
    estilo: `left:${((100 * inicio) / est.T).toFixed(3)}%;width:${Math.max(0.15, (100 * (fim - inicio)) / est.T).toFixed(3)}%;${extra.estilo || ""}`,
    ...(extra.titulo ? { title: extra.titulo } : {}),
  });
}

function desenharLinhaTempo() {
  const lt = $("linha-tempo");
  lt.replaceChildren();
  const segs = trechos();
  const s = est.sessao;
  for (const c of CLASSES) {
    const classe = s.classes.find((x) => x.id === c);
    const trilha = html("div", { class: "trilha" });
    for (const [a, b] of est.lacunas) trilha.appendChild(faixa("faixa sem-pessoa", a / est.fps, b / est.fps));
    for (const a of s.anotacoes.filter((x) => x.classe === c)) {
      trilha.appendChild(faixa("faixa anotado", a.inicio_s, a.fim_s, { estilo: `background:${CORES[c]}`, titulo: `Anotado: ${fmtTempo(a.inicio_s)} a ${fmtTempo(a.fim_s)}` }));
    }
    for (const g of segs.filter((x) => x.classe === c)) {
      trilha.appendChild(faixa("faixa detectado", g.inicio_s, g.fim_s, { estilo: `background:${CORES[c]}`, titulo: `Detectado: ${fmtTempo(g.inicio_s)} a ${fmtTempo(g.fim_s)}` }));
    }
    trilha.appendChild(html("div", { class: "cabeca" }));
    trilha.addEventListener("click", (e) => {
      const r = trilha.getBoundingClientRect();
      irPara(((e.clientX - r.left) / r.width) * est.T);
    });
    const rotulo = html("div", { class: "trilha-rotulo" }, [html("span", { class: "quadradinho", estilo: `background:${CORES[c]}` }), html("span", {}, [nomeClasse(classe)])]);
    lt.appendChild(html("div", { class: "trilha-linha" }, [rotulo, trilha]));
  }
  lt.appendChild(html("div", { class: "trilha-linha regua" }, [html("div"), html("div", { class: "marcas num" },
    [0, 0.25, 0.5, 0.75, 1].map((f) => html("span", { estilo: `left:${100 * f}%`, texto: fmtTempo(f * est.T) })))]));
  $("sem-janelas")?.remove();
  if (!s.janelas.length) {
    lt.before(html("p", { id: "sem-janelas", class: "nota", texto: "Neste vídeo a pessoa quase nunca é detectada por 4 s seguidos, então o modelo não dá nenhuma sugestão." }));
  }
  est.ultimo = null;
}

function detalhes() {
  const s = est.sessao;
  const segs = trechos();
  $("limiar-valor").textContent = num(Number($("limiar").value), 2);
  const sem = s.quadros_sem_deteccao.length;
  $("detalhes-lista").replaceChildren(...[
    `Modelo ${s.modelo.nome} (semente ${s.modelo.semente}). Pontuações fora da partição: vêm do modelo da partição ${s.modelo.particao}, que não viu este vídeo no treino.`,
    `Janela de ${s.modelo.janela_s} s reamostrada a ${s.modelo.amostras_por_s} amostras/s; uma pontuação (softmax) por quadro, terminando nele.`,
    `Pós-processamento: mediana de ${num(s.posprocessamento.mediana_s, 0)} s nas pontuações e junção de trechos da mesma classe separados por menos de ${num(s.posprocessamento.juntar_s, 0)} s.`,
    `Pose: ${s.modelo.pose}. Pontos com confiança abaixo de ${num(LIMIAR_CONF, 1)} aparecem esmaecidos. Esqueleto suavizado: média móvel centrada de ${num(SUAVIZACAO_S, 1)} s por ponto, só na exibição.`,
    `${segs.length} trechos detectados com este limiar; ${sem} de ${s.sessao.total_quadros} quadros sem pessoa (${num((100 * sem) / s.sessao.total_quadros, 1)}%).`,
  ].map((t) => html("li", { texto: t })));
}

function abrirArquivo() {
  const f = $("arquivo").files[0];
  if (!f) return;
  fecharArquivo();
  est.url = URL.createObjectURL(f);
  const v = $("video");
  v.src = est.url;
  v.currentTime = est.t;
  v.hidden = false;
  $("sem-arquivo").hidden = true;
  const id = $("sel-video").value;
  $("arquivo-info").textContent = f.name.includes(id) ? f.name : `${f.name}: o nome não corresponde ao vídeo escolhido.`;
}

function fecharArquivo() {
  const v = $("video");
  v.pause();
  v.removeAttribute("src");
  v.load();
  v.hidden = true;
  if (est.url) URL.revokeObjectURL(est.url);
  est.url = null;
  $("arquivo").value = "";
  $("arquivo-info").textContent = "";
  $("sem-arquivo").hidden = false;
}

function tocar() {
  if (est.t >= est.T) irPara(0);
  est.tocando = true;
  est.inicio = performance.now();
  est.t0 = est.t;
  if (est.url) $("video").play();
  $("tocar").textContent = "Pausar";
}

function pausar() {
  est.tocando = false;
  if (est.url) $("video").pause();
  $("tocar").textContent = "Reproduzir";
}

function irPara(t) {
  est.t = Math.max(0, Math.min(est.T, t));
  est.t0 = est.t;
  est.inicio = performance.now();
  if (est.url) $("video").currentTime = est.t;
}

function laco() {
  if (est.sessao && !$("modo-videos").hidden) {
    if (est.tocando) {
      est.t = est.url ? $("video").currentTime : est.t0 + (performance.now() - est.inicio) / 1000;
      if (est.t >= est.T || (est.url && $("video").ended)) { est.t = Math.min(est.t, est.T); pausar(); }
    }
    $("tempo").textContent = `${fmtTempo(est.t)} / ${fmtTempo(est.T)}`;
    for (const c of document.querySelectorAll("#linha-tempo .cabeca")) c.style.left = `${((100 * est.t) / est.T).toFixed(3)}%`;
    const q = Math.min(est.pontos.quadros.length - 1, Math.floor(est.t * est.fps));
    if (q !== est.ultimo) {
      const quadros = $("suavizar").checked ? est.suaves : est.pontos.quadros;
      const larg = est.sessao.sessao.altura / 120;
      desenharEsqueleto($("esqueleto"), quadros[q], (x, y) => [x, y], { osso: larg, junta: larg * 3.2, miolo: larg * 1.8 }, LIMIAR_CONF);
      $("sem-pessoa").hidden = quadros[q] !== null;
      est.ultimo = q;
    }
  }
  requestAnimationFrame(laco);
}

// ---------- Câmera ao vivo ----------

const cam = { ativo: false, stream: null, sessao: "", t0: 0, historico: [], respostas: [] };
const NOMES_CAMERA = { background: "Sem estereotipia" };

function montarBarras() {
  $("barras").replaceChildren(...[...CLASSES, "background"].map((c) => html("div", { class: "barra-classe" }, [
    html("span", { class: "barra-nome" }, [c === "background" ? NOMES_CAMERA.background : html("i", { texto: NOMES_EN[c] })]),
    html("span", { class: "barra-trilho" }, [html("span", { class: "barra-valor", id: "barra-" + c, estilo: `background:${CORES[c]}` }), html("span", { class: "barra-meio" })]),
    html("span", { class: "barra-num num", id: "num-" + c, texto: "—" }),
  ])));
}

async function ligar() {
  if (cam.ativo) { desligar(); return; }
  try {
    cam.stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 } }, audio: false });
  } catch (e) {
    $("camera-estado").textContent = `Não consegui abrir a câmera (${e.name}).`;
    return;
  }
  const v = $("webcam");
  v.srcObject = cam.stream;
  await v.play();
  Object.assign(cam, { ativo: true, sessao: Math.random().toString(36).slice(2), t0: performance.now(), historico: [], respostas: [] });
  $("ligar").textContent = "Desligar a câmera";
  $("camera-estado").textContent = "A imagem não é gravada.";
  enviar();
}

function desligar() {
  cam.ativo = false;
  if (cam.stream) for (const t of cam.stream.getTracks()) t.stop();
  cam.stream = null;
  $("webcam").srcObject = null;
  $("ligar").textContent = "Ligar a câmera";
  $("esqueleto-camera").replaceChildren();
  $("camera-sem-pessoa").hidden = true;
  $("aguardando").hidden = true;
  $("taxa").textContent = "";
  mostrarPontuacoes(null);
}

const dormir = (ms) => new Promise((r) => setTimeout(r, ms));

async function enviar() {
  const v = $("webcam");
  const canvas = document.createElement("canvas"); // fora da página: só para codificar o quadro
  while (cam.ativo) {
    const inicio = performance.now();
    if (!v.videoWidth) { await dormir(50); continue; }
    const esc = Math.min(1, 640 / v.videoWidth);
    canvas.width = Math.round(v.videoWidth * esc);
    canvas.height = Math.round(v.videoHeight * esc);
    canvas.getContext("2d").drawImage(v, 0, 0, canvas.width, canvas.height);
    const corpo = await new Promise((r) => canvas.toBlob(r, "image/jpeg", 0.8));
    const t = (inicio - cam.t0) / 1000;
    let resp;
    try {
      const r = await fetch("/camera/quadro", { method: "POST", body: corpo, headers: { "Content-Type": "image/jpeg", "X-Tempo": String(t), "X-Sessao": cam.sessao } });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      resp = await r.json();
    } catch (e) {
      desligar();
      $("camera-estado").textContent = `O servidor da câmera não respondeu (${e.message}). Ele precisa estar rodando com o Python do projeto (.venv).`;
      return;
    }
    if (!cam.ativo) return;
    mostrarResposta(resp, t);
    const espera = 1000 / QUADROS_POR_S - (performance.now() - inicio);
    if (espera > 0) await dormir(espera);
  }
}

function mediana(xs) {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function mostrarResposta(resp, t) {
  $("sobre-camera").setAttribute("viewBox", `0 0 ${resp.largura} ${resp.altura}`);
  ajustarPalco($("palco-camera"), resp.largura, resp.altura);
  const larg = resp.altura / 120;
  desenharEsqueleto($("esqueleto-camera"), resp.pontos, (x, y) => [x, y], { osso: larg, junta: larg * 3.2, miolo: larg * 1.8 }, LIMIAR_CONF);
  $("camera-sem-pessoa").hidden = resp.pontos !== null;

  cam.respostas.push(performance.now());
  while (cam.respostas[0] < performance.now() - 2000) cam.respostas.shift();
  if (cam.respostas.length > 1) {
    const dur = (cam.respostas.at(-1) - cam.respostas[0]) / 1000;
    $("taxa").textContent = `${num((cam.respostas.length - 1) / dur, 1)} quadros/s processados`;
  }

  // a mesma mediana de 2 s do pós-processamento; ao vivo, só sobre o passado
  if (resp.pontuacoes) cam.historico.push({ t, p: resp.pontuacoes });
  else cam.historico = [];
  while (cam.historico.length && cam.historico[0].t < t - MEDIANA_S) cam.historico.shift();
  $("aguardando").hidden = resp.pontuacoes !== null || resp.pontos === null;
  mostrarPontuacoes(cam.historico.length ? [0, 1, 2, 3].map((c) => mediana(cam.historico.map((h) => h.p[c]))) : null);
}

function mostrarPontuacoes(p) {
  [...CLASSES, "background"].forEach((c, i) => {
    $("barra-" + c).style.width = p ? `${(100 * p[i]).toFixed(1)}%` : "0";
    $("num-" + c).textContent = p ? num(p[i], 2) : "—";
  });
  const dom = $("dominante");
  const i = p ? p.indexOf(Math.max(...p)) : -1;
  if (i >= 0 && p[i] > 0.5) {
    const c = [...CLASSES, "background"][i];
    dom.replaceChildren(c === "background" ? NOMES_CAMERA.background : html("i", { texto: NOMES_EN[c] }));
  } else {
    dom.textContent = "—";
  }
}

$("ligar").addEventListener("click", ligar);
montarBarras();

// ---------- início ----------

mostrarModo(["videos", "camera", "como"].includes(params.get("modo")) ? params.get("modo") : "videos");
iniciarVideos().catch((e) => mostrarErro($("linha-tempo"), `${e.message} Rode src/exportar_sessoes_final.py e abra a página pelo demo/servir.py.`));
