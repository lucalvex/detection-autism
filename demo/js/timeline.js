// Tradução linha a linha de build_segments() em src/timeline.py.
// A interface recalcula os trechos quando o limiar muda, então esta
// função tem que dar exatamente o mesmo resultado que a versão em
// Python -- tests/test_timeline_js.py compara as duas e exige
// igualdade. Qualquer mudança aqui ou lá tem que ser feita nas duas.

export function buildSegments(windows, classNames, threshold, minDurationSec, fps) {
  const segments = [];

  for (let classIdx = 0; classIdx < classNames.length; classIdx++) {
    const className = classNames[classIdx];
    let runStart = null;

    for (let i = 0; i < windows.length; i++) {
      const active = windows[i].pontuacoes[classIdx] >= threshold;

      if (active && runStart === null) runStart = i;

      const runEndsHere = runStart !== null && (!active || i === windows.length - 1);

      if (runEndsHere) {
        const runEnd = active ? i : i - 1;
        const quadroInicio = windows[runStart].quadro_inicio;
        const quadroFim = windows[runEnd].quadro_fim;
        const duracaoS = (quadroFim - quadroInicio) / fps;

        if (duracaoS >= minDurationSec) {
          let pico = windows[runStart].pontuacoes[classIdx];
          for (let j = runStart + 1; j <= runEnd; j++) {
            const p = windows[j].pontuacoes[classIdx];
            if (p > pico) pico = p;
          }
          segments.push({
            classe: className,
            quadro_inicio: quadroInicio,
            quadro_fim: quadroFim,
            inicio_s: quadroInicio / fps,
            fim_s: quadroFim / fps,
            duracao_s: duracaoS,
            pico: pico,
          });
        }

        runStart = null;
      }
    }
  }

  return segments;
}

// Pós-processamento do modelo final: tradução de median_scores(),
// merge_segments() e final_segments() em src/timeline.py (mesma
// exigência de igualdade exata no teste).

export const MEDIANA_S = 2.0;
export const JUNTAR_S = 1.0;

// mediana centrada, tamanho ímpar, bordas repetidas (scipy median_filter, mode="nearest")
export function medianScores(windows, fps, medianSec) {
  if (!medianSec || windows.length === 0) return windows;
  const k = Math.floor(Math.round(medianSec * fps) / 2) * 2 + 1;
  const h = (k - 1) / 2;
  const n = windows.length;
  const nc = windows[0].pontuacoes.length;
  const buf = new Float64Array(k);
  return windows.map((w, i) => {
    const p = new Array(nc);
    for (let c = 0; c < nc; c++) {
      for (let j = -h; j <= h; j++) buf[j + h] = windows[Math.min(n - 1, Math.max(0, i + j))].pontuacoes[c];
      buf.sort();
      p[c] = buf[h];
    }
    return { ...w, pontuacoes: p };
  });
}

export function mergeSegments(segments, gapSec) {
  const out = [];
  for (const c of new Set(segments.map((s) => s.classe))) {
    let atual = null;
    const daClasse = segments.filter((s) => s.classe === c).sort((a, b) => a.inicio_s - b.inicio_s);
    for (const s of daClasse) {
      if (atual && s.inicio_s - atual.fim_s < gapSec) {
        atual.fim_s = Math.max(atual.fim_s, s.fim_s);
        atual.quadro_fim = Math.max(atual.quadro_fim, s.quadro_fim);
        atual.pico = Math.max(atual.pico, s.pico);
        atual.duracao_s = atual.fim_s - atual.inicio_s;
      } else {
        atual = { ...s };
        out.push(atual);
      }
    }
  }
  return out;
}

export function finalSegments(windows, classNames, threshold, fps, medianSec = MEDIANA_S, gapSec = JUNTAR_S) {
  return mergeSegments(buildSegments(medianScores(windows, fps, medianSec), classNames, threshold, 0.0, fps), gapSec);
}
