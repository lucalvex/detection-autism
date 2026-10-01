// Lê os casos gerados por tests/test_timeline_js.py, aplica a
// buildSegments() da interface e grava o resultado para o Python
// comparar. Uso: node tests/js/segmentos_js.mjs <casos.json> <saida.json>

import { readFileSync, writeFileSync } from "node:fs";
import { buildSegments } from "../../demo/js/timeline.js";

const [casosPath, saidaPath] = process.argv.slice(2);
const { videos, casos } = JSON.parse(readFileSync(casosPath, "utf-8"));

const resultados = casos.map((c) => {
  const v = videos[c.video];
  return buildSegments(v.windows, v.class_names, c.threshold, c.min_duration_sec, v.fps);
});

writeFileSync(saidaPath, JSON.stringify(resultados));
