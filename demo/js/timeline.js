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
