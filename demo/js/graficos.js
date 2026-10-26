// Gráficos pequenos do modo apresentação: barras horizontais (achados
// posteriores) e mapa de calor (matrizes de confusão). Os valores chegam
// prontos dos arquivos de resultado; aqui só se desenha.

import { svg, html, num } from "./comum.js";

// Cores dos dois modelos, distintas das cores das classes. Validadas com o
// validate_palette.js da skill de visualização (modo claro, superfície
// #F6F8F7): faixa de luminosidade, croma, separação para daltonismo
// (protan 11,9), separação normal (24,9) e contraste >= 3:1.
export const CORES_MODELO = { base: "#00897B", intervalos: "#C7522A" };
export const NOMES_MODELO = { base: "modelo base", intervalos: "modelo com rótulos por intervalo" };
export const CURTO_MODELO = { base: "base", intervalos: "intervalos" };

const TEXTO = "#1E2B2F";
const TEXTO_2 = "#4A585C";
const EIXO = "#C3CBC8";
const OPACIDADE_FORA = 0.42;

export const inteiro = (n) => n.toLocaleString("pt-BR");

// ---------- dica (tooltip) compartilhada ----------

let dica = null;

function elementoDica() {
  if (!dica) {
    dica = html("div", { class: "dica", role: "tooltip" });
    dica.hidden = true;
    document.body.appendChild(dica);
  }
  return dica;
}

export function comDica(el, texto) {
  const mover = (ev) => {
    const d = elementoDica();
    d.textContent = texto;
    d.hidden = false;
    const x = Math.min(ev.clientX + 14, window.innerWidth - d.offsetWidth - 8);
    d.style.left = x + "px";
    d.style.top = ev.clientY + 16 + "px";
  };
  el.addEventListener("mouseenter", mover);
  el.addEventListener("mousemove", mover);
  el.addEventListener("mouseleave", () => { elementoDica().hidden = true; });
  return el;
}

// ---------- texto em SVG ----------

// partes: string ou { t, italico }
function texto(atributos, partes) {
  const el = svg("text", { fill: TEXTO, "font-size": 12, ...atributos });
  for (const p of Array.isArray(partes) ? partes : [partes]) {
    if (typeof p === "string") el.appendChild(document.createTextNode(p));
    else el.appendChild(Object.assign(svg("tspan", p.italico ? { "font-style": "italic" } : {}), { textContent: p.t }));
  }
  return el;
}

// barra ancorada à esquerda, com a ponta direita arredondada (4 px)
function barra(x0, y, w, h, cor, opacidade) {
  const r = Math.min(4, h / 2, w);
  const d = w <= 0
    ? `M${x0} ${y}h0.5v${h}h-0.5Z`
    : `M${x0} ${y}h${(w - r).toFixed(2)}a${r} ${r} 0 0 1 ${r} ${r}v${h - 2 * r}a${r} ${r} 0 0 1 ${-r} ${r}h${-(w - r).toFixed(2)}Z`;
  return svg("path", { d, fill: cor, "fill-opacity": opacidade });
}

// ---------- barras horizontais ----------
//
// spec = {
//   grupos: [{ titulo?: partes, linhas: [{ esquerda?, rotulo, valor, modelo, fora?, junto?, dica }],
//              nota?: partes }],
//   referencia?: { valor, rotulo },
//   larguraRotulo,
// }
// junto: a linha seguinte é o par desta (2 px de vão em vez de 8).
// Escala de 0 a 1, a mesma em todos os painéis.

const ALT_BARRA = 12;
const PASSO_PAR = ALT_BARRA + 2;      // barras do mesmo par: 2 px de vão
const PASSO_LINHA = ALT_BARRA + 8;    // entre pares ou linhas soltas
const ALT_TITULO = 22;
const ALT_NOTA = 20;

export function barrasHorizontais(container, spec) {
  const desenhar = () => {
    const W = Math.max(240, container.clientWidth);
    const x0 = spec.larguraRotulo;
    const S = W - x0 - 46;
    const el = svg("svg", { width: W, role: "img", "aria-label": spec.descricao || "" });
    const camadaRef = svg("g");
    el.appendChild(camadaRef);
    let y = spec.referencia ? 20 : 4;
    const faixas = [];   // [y0, y1] das linhas de barras de cada grupo, para a linha de referência

    for (const g of spec.grupos) {
      if (g.titulo) {
        const t = texto({ x: 0, y: y + 13, "font-weight": 700, "font-size": 12.5 }, g.titulo);
        el.appendChild(t);
        y += ALT_TITULO;
      }
      const yGrupo = y;
      g.linhas.forEach((l, i) => {
        const cor = CORES_MODELO[l.modelo];
        const w = Math.max(0, l.valor) * S;
        if (l.esquerda !== undefined) el.appendChild(texto({ x: 0, y: y + 10, fill: TEXTO_2 }, l.esquerda));
        el.appendChild(texto({ x: x0 - 8, y: y + 10, "text-anchor": "end", fill: TEXTO_2 }, l.rotulo));
        el.appendChild(barra(x0, y, w, ALT_BARRA, cor, l.fora ? OPACIDADE_FORA : 1));
        el.appendChild(texto({ x: x0 + w + 5, y: y + 10, class: "num" }, num(l.valor, 3)));
        const alvo = svg("rect", { x: 0, y: y - 3, width: W, height: ALT_BARRA + 6, fill: "transparent" });
        el.appendChild(comDica(alvo, l.dica));
        y += l.junto && i < g.linhas.length - 1 ? PASSO_PAR : PASSO_LINHA;
      });
      faixas.push([yGrupo - 3, y - (PASSO_LINHA - ALT_BARRA) + 4]);
      if (g.nota) {
        el.appendChild(texto({ x: x0, y: y + 8, "font-weight": 700, "font-size": 12 }, g.nota));
        y += ALT_NOTA;
      }
      y += 6;
    }
    const yFim = y - 6;

    // eixo: marcas em 0, 0,5 e 1
    // linha de base só na altura das barras, como a de referência; a última vai até as marcas
    faixas.forEach(([a, b], i) => {
      camadaRef.appendChild(svg("line", { x1: x0, x2: x0, y1: a, y2: i === faixas.length - 1 ? yFim : b, stroke: EIXO, "stroke-width": 1 }));
    });
    for (const v of [0, 0.5, 1]) {
      const x = x0 + v * S;
      camadaRef.appendChild(svg("line", { x1: x, x2: x, y1: yFim, y2: yFim + 4, stroke: EIXO, "stroke-width": 1 }));
      camadaRef.appendChild(texto({ x, y: yFim + 16, "text-anchor": "middle", fill: TEXTO_2, "font-size": 11, class: "num" }, v === 0 ? "0" : num(v, v === 1 ? 0 : 1)));
    }
    if (spec.referencia) {
      const x = x0 + spec.referencia.valor * S;
      // tracejada só na altura das barras, sem cruzar os títulos dos grupos
      for (const [a, b] of faixas) {
        camadaRef.appendChild(svg("line", { x1: x, x2: x, y1: a, y2: b, stroke: TEXTO_2, "stroke-width": 1, "stroke-dasharray": "4 3" }));
      }
      el.appendChild(texto({ x, y: 11, "text-anchor": "middle", fill: TEXTO_2, "font-size": 11 }, `${spec.referencia.rotulo} ${num(spec.referencia.valor, 3)}`));
    }
    el.setAttribute("height", yFim + 20);
    container.replaceChildren(el);
  };
  desenhar();
  let ultima = container.clientWidth;
  new ResizeObserver(() => {
    if (container.clientWidth !== ultima) { ultima = container.clientWidth; desenhar(); }
  }).observe(container);
}

// ---------- mapa de calor ----------

const CLARO = [238, 242, 241];   // fração 0
const ESCURO = [31, 68, 80];     // fração 1 (um só matiz, do claro ao escuro)

export function corSequencial(f) {
  const c = CLARO.map((a, i) => Math.round(a + (ESCURO[i] - a) * f));
  return `rgb(${c.join(",")})`;
}

// Matriz como tabela: linhas = classe anotada, colunas = classe prevista.
// Cor = fração da linha; texto = número de quadros e a fração.
export function mapaDeCalor({ titulo, linhas, colunas, matriz, descricaoCelula }) {
  const tabela = html("table", { class: "calor" });
  tabela.appendChild(html("caption", {}, [titulo]));
  const cab = html("tr", {}, [html("th", { scope: "col", class: "canto", texto: "Anotada" })]);
  for (const c of colunas) cab.appendChild(html("th", { scope: "col" }, [c.rotulo]));
  tabela.appendChild(html("thead", {}, [cab]));
  const corpo = html("tbody");
  matriz.forEach((linha, i) => {
    const total = linha.reduce((a, b) => a + b, 0);
    const tr = html("tr", {}, [html("th", { scope: "row" }, [linhas[i].rotulo])]);
    linha.forEach((n, j) => {
      const f = total ? n / total : 0;
      const td = html("td", { estilo: `background:${corSequencial(f)};color:${f > 0.5 ? "#FFFFFF" : TEXTO}` }, [
        html("span", { class: "n num", texto: inteiro(n) }),
        html("span", { class: "f num", texto: num(f, 3) }),
      ]);
      tr.appendChild(comDica(td, descricaoCelula(i, j, n, f, total)));
    });
    corpo.appendChild(tr);
  });
  tabela.appendChild(corpo);
  return tabela;
}

export function escalaSequencial() {
  const passos = [0, 0.25, 0.5, 0.75, 1];
  return html("div", { class: "escala-calor" }, [
    html("span", { texto: "fração da linha" }),
    html("span", { class: "rampa", estilo: `background:linear-gradient(90deg,${passos.map((p) => corSequencial(p)).join(",")})` }),
    html("span", { class: "extremos num" }, [html("span", { texto: "0" }), html("span", { texto: "1" })]),
  ]);
}
