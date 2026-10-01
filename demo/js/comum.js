// Funções compartilhadas pelas duas telas. Nenhum número de dado ou
// de resultado é escrito aqui: tudo vem dos JSON carregados.

export const CORES = {
  armflapping: "#1F4F8F",
  headbanging: "#E0A13A",
  spinning: "#8E6FC0",
  background: "#7F8C89",
};

// Ligações entre os 17 pontos COCO, as mesmas do design.
export const OSSOS = [
  [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12], [11, 12], [11, 13],
  [13, 15], [12, 14], [14, 16], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6],
];

export const RESULTADOS = {
  rodadaA: "/data/results/comparacao_rodada_a/metricas.json",
  base: "/data/results/comparacao_base_reseeded/metricas.json",
  posteriores: "/data/results/analises_posteriores.json",
  medianos: "/data/results/videos_desempenho_mediano.json",
  comparacao: "/data/results/comparacao_final.json",
  resumo: "/data/results/resumo_dados.json",
};

export async function carregarJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Não consegui carregar ${url} (HTTP ${r.status}).`);
  return r.json();
}

export async function carregarSessao(id) {
  const [sessao, pontos] = await Promise.all([
    carregarJSON(`/saidas/${id}_sessao.json`),
    carregarJSON(`/saidas/${id}_pontos.json`),
  ]);
  if (pontos.quadros.length !== sessao.sessao.total_quadros) {
    throw new Error(
      `${id}: o _pontos.json tem ${pontos.quadros.length} quadros e o _sessao.json diz ${sessao.sessao.total_quadros}.`
    );
  }
  return { sessao, pontos };
}

export function num(x, casas) {
  return x.toFixed(casas).replace(".", ",");
}

export function fmtTempo(t) {
  const s = Math.max(0, Math.floor(t + 1e-6));
  return String(Math.floor(s / 60)).padStart(2, "0") + ":" + String(s % 60).padStart(2, "0");
}

export function fmtTempoFino(t) {
  const m = Math.floor(t / 60);
  const s = t - m * 60;
  return String(m).padStart(2, "0") + ":" + num(s, 2).padStart(5, "0");
}

export function centroJanela(j, fps) {
  return (j.quadro_inicio + j.quadro_fim) / 2 / fps;
}

// Janela cujo centro está mais perto do tempo t (busca binária; as
// janelas vêm em ordem crescente de quadro_inicio).
export function janelaMaisProxima(janelas, fps, t) {
  if (!janelas.length) return null;
  let lo = 0, hi = janelas.length - 1;
  while (lo < hi) {
    const meio = (lo + hi) >> 1;
    if (centroJanela(janelas[meio], fps) < t) lo = meio + 1;
    else hi = meio;
  }
  if (lo > 0 && Math.abs(centroJanela(janelas[lo - 1], fps) - t) <= Math.abs(centroJanela(janelas[lo], fps) - t)) lo -= 1;
  return janelas[lo];
}

// Sequências de quadros consecutivos: [[inicio, fim_inclusivo], ...]
export function sequencias(quadros) {
  const out = [];
  for (const q of quadros) {
    const ult = out[out.length - 1];
    if (ult && q === ult[1] + 1) ult[1] = q;
    else out.push([q, q]);
  }
  return out;
}

// Trechos de janelas longas consecutivas, em segundos (centro da
// primeira ao centro da última).
export function trechosJanelasLongas(janelas, fps) {
  const out = [];
  let inicio = null, anterior = null;
  janelas.forEach((j, i) => {
    if (j.janela_longa) {
      if (inicio === null) inicio = i;
      anterior = i;
    } else if (inicio !== null) {
      out.push([centroJanela(janelas[inicio], fps), centroJanela(janelas[anterior], fps), anterior - inicio + 1]);
      inicio = null;
    }
  });
  if (inicio !== null) out.push([centroJanela(janelas[inicio], fps), centroJanela(janelas[anterior], fps), anterior - inicio + 1]);
  return out;
}

export function caminhoCurva(janelas, fps, indiceClasse, X, Y) {
  let d = "";
  janelas.forEach((j, k) => {
    d += (k ? "L" : "M") + X(centroJanela(j, fps)).toFixed(1) + " " + Y(j.pontuacoes[indiceClasse]).toFixed(1);
  });
  return d;
}

export function svg(tag, atributos = {}, filhos = []) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(atributos)) el.setAttribute(k, v);
  for (const f of filhos) el.appendChild(f);
  return el;
}

export function html(tag, atributos = {}, filhos = []) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(atributos)) {
    if (k === "texto") el.textContent = v;
    else if (k === "estilo") el.setAttribute("style", v);
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v);
  }
  for (const f of filhos) if (f !== null && f !== undefined) el.appendChild(typeof f === "string" ? document.createTextNode(f) : f);
  return el;
}

// Desenha o esqueleto de um quadro dentro de um <g>. Um quadro sem
// detecção (null) não desenha nada: sem interpolar nem repetir o
// quadro anterior.
export function desenharEsqueleto(g, pontosQuadro, transformar, larguras = { osso: 3, junta: 11, miolo: 6 }) {
  g.replaceChildren();
  if (!pontosQuadro) return false;
  const p = pontosQuadro.map((q) => transformar(q[0], q[1]));
  let ossos = "", juntas = "";
  for (const [a, b] of OSSOS) ossos += `M${p[a][0].toFixed(1)} ${p[a][1].toFixed(1)}L${p[b][0].toFixed(1)} ${p[b][1].toFixed(1)}`;
  for (const q of p) juntas += `M${q[0].toFixed(1)} ${q[1].toFixed(1)}l0.01 0`;
  g.appendChild(svg("path", { d: ossos, fill: "none", stroke: "#1E2B2F", "stroke-width": larguras.osso, "stroke-linecap": "round" }));
  g.appendChild(svg("path", { d: juntas, fill: "none", stroke: "#1E2B2F", "stroke-width": larguras.junta, "stroke-linecap": "round" }));
  g.appendChild(svg("path", { d: juntas, fill: "none", stroke: "#F6F8F7", "stroke-width": larguras.miolo, "stroke-linecap": "round" }));
  return true;
}

export function mostrarErro(container, mensagem) {
  container.replaceChildren(html("p", { class: "erro", texto: mensagem }));
}
