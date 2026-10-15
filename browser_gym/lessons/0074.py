"""Generated notebook cells for the 0074 detection AP lesson."""

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
        detections_probe = [
            (0, 0.90, 0),
            (0, 0.65, -1),
            (0, 0.70, 1),
            (1, 0.60, 0),
        ]
        precision_recall_points = np.array([
            [1 / 3, 1.0],
            [2 / 3, 1.0],
            [2 / 3, 2 / 3],
            [1.0, 3 / 4],
        ], dtype=np.float64)
        display(tensor_card("precision_recall_points", precision_recall_points, axes=("detection", "metric")))

        pr_probe = {
            "base": [
                {"image": 0, "score": 0.90, "gt": 0, "label": "TP image 0 gt 0"},
                {"image": 0, "score": 0.65, "gt": -1, "label": "FP unmatched"},
                {"image": 0, "score": 0.70, "gt": 1, "label": "TP image 0 gt 1"},
                {"image": 1, "score": 0.60, "gt": 0, "label": "TP image 1 gt 0"},
            ],
            "total_gt": 3,
        }
        display(HTML("""
        <section id="dojo-0074-pr" class="dojo-pr-probe" role="region"
                 aria-labelledby="dojo-0074-pr-title">
          <style>
            #dojo-0074-pr {
              border: 1px solid var(--dojo-rule, #d1d5db);
              border-radius: 8px;
              padding: 12px;
              max-width: 680px;
              background: var(--dojo-surface, #ffffff);
              color: var(--dojo-fg, #111827);
            }
            #dojo-0074-pr label {
              display: grid;
              gap: 4px;
              font-weight: 600;
              margin: 8px 0;
            }
            #dojo-0074-pr input[type="range"] {
              width: min(100%, 360px);
            }
            #dojo-0074-pr output,
            #dojo-0074-pr .dojo-probe-runs,
            #dojo-0074-pr .dojo-pr-evidence {
              display: block;
              margin-top: 6px;
              font-variant-numeric: tabular-nums;
            }
            #dojo-0074-pr svg {
              display: block;
              width: 100%;
              min-width: 240px;
              max-width: 600px;
              height: 240px;
              margin-top: 10px;
            }
            #dojo-0074-pr svg:focus,
            #dojo-0074-pr input:focus {
              outline: 3px solid var(--dojo-accent, #2563eb);
              outline-offset: 2px;
            }
          </style>
          <h3 id="dojo-0074-pr-title">Precision-recall envelope</h3>
          <label for="dojo-0074-fp-score">
            False-positive score
            <input id="dojo-0074-fp-score" aria-label="False-positive score"
                   type="range" min="0.20" max="0.95" step="0.05" value="0.75" />
          </label>
          <output id="dojo-ap-value" for="dojo-0074-fp-score"
                  aria-live="polite" data-testid="dojo-ap-value">
            AP waiting for probe
          </output>
          <span class="dojo-pr-evidence" data-role="pr-evidence"></span>
          <span class="dojo-probe-runs" data-testid="dojo-pr-probe-runs">Probe runs: 0</span>
          <svg id="dojo-pr-plot" role="img" tabindex="0"
               aria-labelledby="dojo-pr-plot-title dojo-pr-plot-desc"
               viewBox="0 0 600 240">
            <title id="dojo-pr-plot-title">Precision-recall curve</title>
            <desc id="dojo-pr-plot-desc">All-point interpolated precision-recall envelope for one-class detection AP.</desc>
            <line x1="54" y1="198" x2="568" y2="198" stroke="#374151" stroke-width="1" />
            <line x1="54" y1="24" x2="54" y2="198" stroke="#374151" stroke-width="1" />
            <text x="520" y="222" font-size="12">recall</text>
            <text x="8" y="24" font-size="12">precision</text>
            <polyline data-role="pr-line" fill="none" stroke="#2563eb" stroke-width="3" points="" />
            <polyline data-role="envelope-line" fill="none" stroke="#0f766e" stroke-width="3" stroke-dasharray="6 4" points="" />
            <g data-role="points"></g>
          </svg>
        </section>
        <script type="application/javascript">
        (() => {
          const root = document.getElementById("dojo-0074-pr");
          if (!root || root.dataset.ready === "1") return;
          const probe = __PR_PROBE__;
          const input = root.querySelector("#dojo-0074-fp-score");
          const apValue = root.querySelector("[data-testid='dojo-ap-value']");
          const evidence = root.querySelector("[data-role='pr-evidence']");
          const runCount = root.querySelector("[data-testid='dojo-pr-probe-runs']");
          const prLine = root.querySelector("[data-role='pr-line']");
          const envelopeLine = root.querySelector("[data-role='envelope-line']");
          const pointsLayer = root.querySelector("[data-role='points']");
          let runs = 0;

          function orderedRows(falsePositiveScore) {
            return probe.base.map((row, index) => ({
              ...row,
              score: row.gt === -1 ? falsePositiveScore : row.score,
              index,
            })).sort((a, b) => (b.score - a.score) || (a.index - b.index));
          }

          function curve(falsePositiveScore) {
            const claimed = new Set();
            let truePositiveCount = 0;
            const rows = orderedRows(falsePositiveScore);
            const points = [];
            for (const row of rows) {
              const key = `${row.image}:${row.gt}`;
              const isTruePositive = row.gt >= 0 && !claimed.has(key);
              if (isTruePositive) {
                truePositiveCount += 1;
                claimed.add(key);
              }
              const rank = points.length + 1;
              points.push({
                recall: truePositiveCount / probe.total_gt,
                precision: truePositiveCount / rank,
                isTruePositive,
                label: row.label,
              });
            }
            const envelope = [];
            let best = 0;
            for (let index = points.length - 1; index >= 0; index--) {
              best = Math.max(best, points[index].precision);
              envelope[index] = best;
            }
            let previousRecall = 0;
            let ap = 0;
            for (let index = 0; index < points.length; index++) {
              ap += (points[index].recall - previousRecall) * envelope[index];
              previousRecall = points[index].recall;
            }
            return { rows, points, envelope, ap };
          }

          function project(point) {
            const x = 54 + point.recall * 514;
            const y = 198 - point.precision * 174;
            return [x, y];
          }

          function stepInput(direction) {
            const step = Number.parseFloat(input.step || "0.05");
            const min = Number.parseFloat(input.min);
            const max = Number.parseFloat(input.max);
            const next = Math.min(max, Math.max(min, Number.parseFloat(input.value) + direction * step));
            input.value = next.toFixed(2);
            update();
          }

          function update() {
            runs += 1;
            const falsePositiveScore = Number.parseFloat(input.value);
            const result = curve(falsePositiveScore);
            const rawPoints = [{ recall: 0, precision: 1 }, ...result.points];
            const envelopePoints = [{ recall: 0, precision: result.envelope[0] ?? 0 },
              ...result.points.map((point, index) => ({ recall: point.recall, precision: result.envelope[index] }))];
            prLine.setAttribute("points", rawPoints.map(point => project(point).map(number => number.toFixed(1)).join(",")).join(" "));
            envelopeLine.setAttribute("points", envelopePoints.map(point => project(point).map(number => number.toFixed(1)).join(",")).join(" "));
            pointsLayer.innerHTML = result.points.map(point => {
              const [x, y] = project(point);
              const fill = point.isTruePositive ? "#0f766e" : "#b91c1c";
              return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="5" fill="${fill}" />`;
            }).join("");
            const order = result.points.map(point => point.isTruePositive ? "TP" : "FP").join(" -> ");
            apValue.textContent = `AP at false-positive score ${falsePositiveScore.toFixed(2)}: ${result.ap.toFixed(6)}`;
            evidence.textContent = `Evidence: stable score order ${order}; recall denominator total_gt=${probe.total_gt}.`;
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
        """.replace("__PR_PROBE__", json.dumps(pr_probe))))
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
