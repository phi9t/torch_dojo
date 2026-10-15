"""Generated notebook cells for the 0073 paired trip bootstrap lesson."""

from __future__ import annotations

import textwrap

from browser_gym.lesson_ui import notebook_tensor_card_source


def cells(exercise_id: str) -> list[dict]:
    source = (
        textwrap.dedent(
            '''
        import json
        import numpy as np
        from IPython.display import HTML, display

        '''
        ).lstrip()
        + notebook_tensor_card_source()
        + "\n\n"
        + textwrap.dedent(
            r'''
        baseline_losses_probe = np.array([2.4, 3.2, 2.0, 4.6, 3.7, 2.8], dtype=np.float64)
        candidate_losses_probe = np.array([2.1, 3.4, 1.7, 4.0, 3.2, 2.9], dtype=np.float64)
        trip_improvements_probe = baseline_losses_probe - candidate_losses_probe
        display(tensor_card("trip_improvements_probe", trip_improvements_probe, axes=("trip",)))

        bootstrap_probe = {
            "differences": trip_improvements_probe.tolist(),
            "iterations": [50, 250, 450, 650, 850],
        }
        display(HTML("""
        <section id="dojo-0073-bootstrap" class="dojo-bootstrap-probe" role="region"
                 aria-labelledby="dojo-0073-bootstrap-title">
          <style>
            #dojo-0073-bootstrap {
              border: 1px solid var(--dojo-rule, #d1d5db);
              border-radius: 8px;
              padding: 12px;
              max-width: 680px;
              background: var(--dojo-surface, #ffffff);
              color: var(--dojo-fg, #111827);
            }
            #dojo-0073-bootstrap label {
              display: grid;
              gap: 4px;
              font-weight: 600;
              margin: 8px 0;
            }
            #dojo-0073-bootstrap input[type="range"] {
              width: min(100%, 360px);
            }
            #dojo-0073-bootstrap output,
            #dojo-0073-bootstrap .dojo-probe-runs,
            #dojo-0073-bootstrap .dojo-bootstrap-evidence {
              display: block;
              margin-top: 6px;
              font-variant-numeric: tabular-nums;
            }
            #dojo-0073-bootstrap svg {
              display: block;
              width: 100%;
              min-width: 240px;
              max-width: 600px;
              height: 240px;
              margin-top: 10px;
            }
            #dojo-0073-bootstrap svg:focus,
            #dojo-0073-bootstrap input:focus {
              outline: 3px solid var(--dojo-accent, #2563eb);
              outline-offset: 2px;
            }
          </style>
          <h3 id="dojo-0073-bootstrap-title">Grouped bootstrap uncertainty</h3>
          <label for="dojo-0073-bootstrap-iterations">
            Bootstrap iterations
            <input id="dojo-0073-bootstrap-iterations" aria-label="Bootstrap iterations"
                   type="range" min="50" max="850" step="200" value="50" />
          </label>
          <output id="dojo-bootstrap-ci-value" for="dojo-0073-bootstrap-iterations"
                  aria-live="polite" data-testid="dojo-bootstrap-ci-value">
            Mean and interval waiting for probe
          </output>
          <span class="dojo-bootstrap-evidence" data-role="bootstrap-evidence"></span>
          <span class="dojo-probe-runs" data-testid="dojo-bootstrap-probe-runs">Probe runs: 0</span>
          <svg id="dojo-bootstrap-plot" role="img" tabindex="0"
               aria-labelledby="dojo-bootstrap-plot-title dojo-bootstrap-plot-desc"
               viewBox="0 0 600 240">
            <title id="dojo-bootstrap-plot-title">Bootstrap replicate means</title>
            <desc id="dojo-bootstrap-plot-desc">Histogram of paired trip bootstrap mean improvements.</desc>
            <line x1="48" y1="198" x2="568" y2="198" stroke="#374151" stroke-width="1" />
            <line x1="48" y1="24" x2="48" y2="198" stroke="#374151" stroke-width="1" />
            <text x="506" y="222" font-size="12">mean improvement</text>
            <text x="8" y="24" font-size="12">count</text>
            <g data-role="bars"></g>
            <line data-role="ci-low" y1="28" y2="198" stroke="#b91c1c" stroke-width="2" />
            <line data-role="ci-high" y1="28" y2="198" stroke="#b91c1c" stroke-width="2" />
            <line data-role="mean" y1="28" y2="198" stroke="#0f766e" stroke-width="3" />
          </svg>
        </section>
        <script type="application/javascript">
        (() => {
          const root = document.getElementById("dojo-0073-bootstrap");
          if (!root || root.dataset.ready === "1") return;
          const probe = __BOOTSTRAP_PROBE__;
          const input = root.querySelector("#dojo-0073-bootstrap-iterations");
          const ciValue = root.querySelector("[data-testid='dojo-bootstrap-ci-value']");
          const evidence = root.querySelector("[data-role='bootstrap-evidence']");
          const runCount = root.querySelector("[data-testid='dojo-bootstrap-probe-runs']");
          const bars = root.querySelector("[data-role='bars']");
          const loLine = root.querySelector("[data-role='ci-low']");
          const hiLine = root.querySelector("[data-role='ci-high']");
          const meanLine = root.querySelector("[data-role='mean']");
          let runs = 0;

          function makeRandom() {
            let state = 20261006;
            return () => {
              state = (1664525 * state + 1013904223) >>> 0;
              return state / 4294967296;
            };
          }

          function percentile(sorted, p) {
            if (sorted.length === 1) return sorted[0];
            const rank = (p / 100) * (sorted.length - 1);
            const lower = Math.floor(rank);
            const upper = Math.ceil(rank);
            const weight = rank - lower;
            return sorted[lower] + (sorted[upper] - sorted[lower]) * weight;
          }

          function bootstrapMeans(iterations) {
            const random = makeRandom();
            const means = [];
            for (let j = 0; j < iterations; j++) {
              let total = 0;
              for (let k = 0; k < probe.differences.length; k++) {
                const index = Math.floor(random() * probe.differences.length);
                total += probe.differences[index];
              }
              means.push(total / probe.differences.length);
            }
            return means;
          }

          function projectX(value, minValue, maxValue) {
            const range = Math.max(maxValue - minValue, 1e-9);
            return 48 + ((value - minValue) / range) * 520;
          }

          function stepInput(direction) {
            const step = Number.parseInt(input.step || "1", 10);
            const min = Number.parseInt(input.min, 10);
            const max = Number.parseInt(input.max, 10);
            const next = Math.min(max, Math.max(min, Number.parseInt(input.value, 10) + direction * step));
            input.value = String(next);
            update();
          }

          function update() {
            runs += 1;
            const iterations = Number.parseInt(input.value, 10);
            const means = bootstrapMeans(iterations);
            const sorted = [...means].sort((a, b) => a - b);
            const low = percentile(sorted, 2.5);
            const high = percentile(sorted, 97.5);
            const mean = probe.differences.reduce((sum, value) => sum + value, 0) / probe.differences.length;
            const minValue = Math.min(...means, low, mean);
            const maxValue = Math.max(...means, high, mean);
            const binCount = 12;
            const counts = Array.from({ length: binCount }, () => 0);
            const range = Math.max(maxValue - minValue, 1e-9);
            for (const value of means) {
              const index = Math.min(binCount - 1, Math.floor(((value - minValue) / range) * binCount));
              counts[index] += 1;
            }
            const maxCount = Math.max(...counts, 1);
            bars.innerHTML = counts.map((count, index) => {
              const x = 48 + index * (520 / binCount) + 2;
              const width = 520 / binCount - 4;
              const height = (count / maxCount) * 160;
              const y = 198 - height;
              return `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${width.toFixed(1)}" height="${height.toFixed(1)}" fill="#2563eb" opacity="0.78" />`;
            }).join("");
            loLine.setAttribute("x1", projectX(low, minValue, maxValue).toFixed(1));
            loLine.setAttribute("x2", projectX(low, minValue, maxValue).toFixed(1));
            hiLine.setAttribute("x1", projectX(high, minValue, maxValue).toFixed(1));
            hiLine.setAttribute("x2", projectX(high, minValue, maxValue).toFixed(1));
            meanLine.setAttribute("x1", projectX(mean, minValue, maxValue).toFixed(1));
            meanLine.setAttribute("x2", projectX(mean, minValue, maxValue).toFixed(1));
            ciValue.textContent = `Mean ${mean.toFixed(4)}; 95% interval with ${iterations} replicates: [${low.toFixed(4)}, ${high.toFixed(4)}]`;
            evidence.textContent = `Evidence: ${probe.differences.length} paired trip differences, sampled by trip index.`;
            runCount.textContent = `Probe runs: ${runs}`;
          }

          input.addEventListener("keydown", event => {
            if (event.key === "ArrowRight" || event.key === "ArrowUp") {
              event.preventDefault();
              stepInput(1);
            } else if (event.key === "ArrowLeft" || event.key === "ArrowDown") {
              event.preventDefault();
              stepInput(-1);
            }
          });
          input.addEventListener("input", update);
          input.addEventListener("change", update);
          root.dataset.ready = "1";
          update();
        })();
        </script>
        """.replace("__BOOTSTRAP_PROBE__", json.dumps(bootstrap_probe))))
        '''
        ).lstrip()
    )
    return [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"dojo": {"exercise_id": exercise_id, "role": "exploration"}},
            "outputs": [],
            "source": source.splitlines(keepends=True),
        }
    ]
