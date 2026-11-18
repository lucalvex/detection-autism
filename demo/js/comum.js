// Funções compartilhadas da interface. Nenhum número de dado ou
// de resultado é escrito aqui: tudo vem dos JSON carregados.

export const CORES = {
  armflapping: "#1F4F8F",
  headbanging: "#E0A13A",
  spinning: "#8E6FC0",
  background: "#7F8C89",
};

// Nomes das classes como no texto do TCC: em inglês, em minúsculas e em
// itálico, com o nome em português (do _sessao.json) como apoio.
export const NOMES_EN = {
  armflapping: "arm flapping",
  headbanging: "head banging",
  spinning: "spinning",
};
export const NOME_FUNDO = "fundo";
export const DESCRICAO_FUNDO = "sem estereotipia anotada";

// classe: { id, nome } do _sessao.json. Devolve um fragmento:
// <i>arm flapping</i> (balançar os braços)
export function nomeClasse(classe, { apoio = true } = {}) {
  const frag = document.createDocumentFragment();
  if (classe.id === "background") {
    frag.append(apoio ? `${NOME_FUNDO} (${DESCRICAO_FUNDO})` : NOME_FUNDO);
    return frag;
  }
  const i = document.createElement("i");
  i.textContent = NOMES_EN[classe.id];
  frag.append(i);
  if (apoio) frag.append(` (${classe.nome.toLowerCase()})`);
  return frag;
}

// Ligações entre os 17 pontos COCO, as mesmas do design.
export const OSSOS = [
  [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12], [11, 12], [11, 13],
  [13, 15], [12, 14], [14, 16], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6],
];

export async function carregarJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Não consegui carregar ${url} (HTTP ${r.status}).`);
  return r.json();
}

export function num(x, casas) {
  return x.toFixed(casas).replace(".", ",");
}

export function fmtTempo(t) {
  const s = Math.max(0, Math.floor(t + 1e-6));
  return String(Math.floor(s / 60)).padStart(2, "0") + ":" + String(s % 60).padStart(2, "0");
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
// quadro anterior. Com esmaecer, pontos com confiança abaixo dele (e
// os ossos que tocam neles) ficam esmaecidos.
export function desenharEsqueleto(g, pontosQuadro, transformar, larguras = { osso: 3, junta: 11, miolo: 6 }, esmaecer = 0) {
  g.replaceChildren();
  if (!pontosQuadro) return false;
  const p = pontosQuadro.map((q) => transformar(q[0], q[1]));
  const fraco = pontosQuadro.map((q) => q[2] < esmaecer);
  for (const esmaecido of [true, false]) {
    let ossos = "", juntas = "";
    for (const [a, b] of OSSOS) {
      if ((fraco[a] || fraco[b]) === esmaecido) ossos += `M${p[a][0].toFixed(1)} ${p[a][1].toFixed(1)}L${p[b][0].toFixed(1)} ${p[b][1].toFixed(1)}`;
    }
    p.forEach((q, i) => { if (fraco[i] === esmaecido) juntas += `M${q[0].toFixed(1)} ${q[1].toFixed(1)}l0.01 0`; });
    if (!ossos && !juntas) continue;
    const grupo = svg("g", esmaecido ? { opacity: 0.3 } : {});
    grupo.appendChild(svg("path", { d: ossos, fill: "none", stroke: "#1E2B2F", "stroke-width": larguras.osso, "stroke-linecap": "round" }));
    grupo.appendChild(svg("path", { d: juntas, fill: "none", stroke: "#1E2B2F", "stroke-width": larguras.junta, "stroke-linecap": "round" }));
    grupo.appendChild(svg("path", { d: juntas, fill: "none", stroke: "#F6F8F7", "stroke-width": larguras.miolo, "stroke-linecap": "round" }));
    g.appendChild(grupo);
  }
  return true;
}

export function mostrarErro(container, mensagem) {
  container.replaceChildren(html("p", { class: "erro", texto: mensagem }));
}
