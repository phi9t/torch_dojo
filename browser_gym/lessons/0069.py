"""Generated notebook cells for the 0069 calibration metrics lesson."""

from __future__ import annotations

import json
import textwrap

import numpy as np

from browser_gym.lesson_ui import notebook_tensor_card_source


def _summary_for_bins(probabilities: np.ndarray, labels: np.ndarray, bins: int) -> dict[str, object]:
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = predicted == labels
    bucket = np.clip(np.ceil(confidence * bins).astype(int) - 1, 0, bins - 1)
    one_hot = np.eye(probabilities.shape[1])[labels]
    ece = 0.0
    rows: list[dict[str, object]] = []
    for index in range(bins):
        selected = bucket == index
        count = int(selected.sum())
        lower = index / bins
        upper = (index + 1) / bins
        if count:
            accuracy = float(correct[selected].mean())
            mean_confidence = float(confidence[selected].mean())
            gap = abs(accuracy - mean_confidence)
            contribution = float(selected.mean() * gap)
            ece += contribution
        else:
            accuracy = None
            mean_confidence = None
            gap = None
            contribution = 0.0
        rows.append(
            {
                "index": index,
                "lower": round(lower, 6),
                "upper": round(upper, 6),
                "count": count,
                "accuracy": None if accuracy is None else round(accuracy, 6),
                "confidence": None if mean_confidence is None else round(mean_confidence, 6),
                "gap": None if gap is None else round(gap, 6),
                "contribution": round(contribution, 6),
            }
        )
    brier = float(np.square(probabilities - one_hot).sum(axis=1).mean())
    nll = float(-np.log(np.maximum(probabilities[np.arange(len(labels)), labels], np.finfo(np.float64).tiny)).mean())
    edge_probe = {
        "0.50": int(np.clip(np.ceil(0.50 * bins).astype(int) - 1, 0, bins - 1)) + 1,
        "0.75": int(np.clip(np.ceil(0.75 * bins).astype(int) - 1, 0, bins - 1)) + 1,
        "1.00": bins,
    }
    return {
        "bins": bins,
        "rows": rows,
        "ece": round(ece, 6),
        "brier": round(brier, 6),
        "nll": round(nll, 6),
        "edge_probe": edge_probe,
    }


def _lesson_data() -> dict[str, object]:
    probabilities = np.array(
        [
            [0.80, 0.20, 0.00],
            [0.60, 0.40, 0.00],
            [0.50, 0.50, 0.00],
            [0.75, 0.25, 0.00],
            [0.34, 0.33, 0.33],
            [0.20, 0.10, 0.70],
            [0.05, 0.90, 0.05],
            [0.55, 0.30, 0.15],
        ],
        dtype=np.float64,
    )
    labels = np.array([0, 1, 0, 1, 2, 2, 1, 0], dtype=np.int64)
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predicted == labels
    return {
        "probabilities": probabilities.tolist(),
        "labels": labels.tolist(),
        "predicted": predicted.astype(int).tolist(),
        "confidence": np.round(confidence, 6).tolist(),
        "correct": correct.astype(int).tolist(),
        "by_bins": {
            str(bins): _summary_for_bins(probabilities, labels, bins)
            for bins in range(2, 11)
        },
    }


def cells(exercise_id: str) -> list[dict]:
    lesson_data = _lesson_data()
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
            '''
        calibration_probabilities_probe = np.array(__PROBABILITIES__, dtype=np.float64)
        calibration_labels_probe = np.array(__LABELS__, dtype=np.int64)

        display(tensor_card("calibration_probabilities_probe", calibration_probabilities_probe, axes=("example", "class")))
        display(tensor_card("calibration_labels_probe", calibration_labels_probe, axes=("example",)))

        display(HTML("""
        <section id="dojo-0069-calibration" class="dojo-calibration-probe" role="region"
                 aria-labelledby="dojo-0069-calibration-title">
          <style>
            #dojo-0069-calibration {
              border: 1px solid var(--dojo-rule, #d1d5db);
              border-radius: 8px;
              padding: 12px;
              max-width: 760px;
              background: var(--dojo-surface, #ffffff);
              color: var(--dojo-fg, #111827);
            }
            #dojo-0069-calibration label {
              display: grid;
              gap: 4px;
              font-weight: 600;
              margin: 8px 0;
            }
            #dojo-0069-calibration input[type="range"] {
              width: min(100%, 360px);
            }
            #dojo-0069-calibration output,
            #dojo-0069-calibration .dojo-probe-runs {
              display: block;
              margin-top: 6px;
              font-variant-numeric: tabular-nums;
            }
            #dojo-0069-calibration .dojo-calibration-plots {
              display: grid;
              grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
              gap: 12px;
              margin-top: 10px;
            }
            #dojo-0069-calibration svg {
              display: block;
              width: 100%;
              min-width: 240px;
              max-width: 360px;
              height: 240px;
            }
            #dojo-0069-calibration svg:focus,
            #dojo-0069-calibration input:focus {
              outline: 3px solid var(--dojo-accent, #2563eb);
              outline-offset: 2px;
            }
          </style>
          <h3 id="dojo-0069-calibration-title">Calibration question: how far can confidence drift from accuracy?</h3>
          <p>Use the trained-model confidence idea from the path, then compare bin accuracy with bin confidence.</p>
          <label for="dojo-0069-bin-count">
            Calibration bins
            <input id="dojo-0069-bin-count" aria-label="Calibration bins"
                   type="range" min="2" max="10" step="1" value="5" />
          </label>
          <output id="dojo-calibration-summary" data-testid="dojo-calibration-summary"
                  for="dojo-0069-bin-count" aria-live="polite">
            Bins 5: waiting for probe
          </output>
          <output id="dojo-calibration-bin-probe" data-testid="dojo-calibration-bin-probe"
                  for="dojo-0069-bin-count" aria-live="polite">
            Edge probe waiting for bins
          </output>
          <span class="dojo-probe-runs" data-testid="dojo-calibration-probe-runs">Probe runs: 0</span>
          <div class="dojo-calibration-plots">
            <svg id="dojo-reliability-plot" data-testid="dojo-reliability-plot" role="img" tabindex="0"
                 aria-labelledby="dojo-reliability-title dojo-reliability-desc"
                 viewBox="0 0 360 240">
              <title id="dojo-reliability-title">Reliability diagram</title>
              <desc id="dojo-reliability-desc">Bin accuracy and confidence under the selected bin count.</desc>
            </svg>
            <svg id="dojo-confidence-histogram" data-testid="dojo-confidence-histogram" role="img" tabindex="0"
                 aria-labelledby="dojo-confidence-title dojo-confidence-desc"
                 viewBox="0 0 360 240">
              <title id="dojo-confidence-title">Confidence distribution</title>
              <desc id="dojo-confidence-desc">Counts of predictions assigned to each confidence bin.</desc>
            </svg>
          </div>
          <script type="application/json" id="dojo-0069-calibration-data">__LESSON_DATA__</script>
        </section>
        <script type="application/javascript">
        (() => {
          const root = document.getElementById("dojo-0069-calibration");
          if (!root || root.dataset.ready === "1") return;
          const data = JSON.parse(root.querySelector("#dojo-0069-calibration-data").textContent);
          const input = root.querySelector("#dojo-0069-bin-count");
          const summary = root.querySelector("[data-testid='dojo-calibration-summary']");
          const binProbe = root.querySelector("[data-testid='dojo-calibration-bin-probe']");
          const runCount = root.querySelector("[data-testid='dojo-calibration-probe-runs']");
          const reliability = root.querySelector("#dojo-reliability-plot");
          const histogram = root.querySelector("#dojo-confidence-histogram");
          let runs = 0;

          function clear(svg) {
            const keep = Array.from(svg.querySelectorAll("title, desc"));
            while (svg.firstChild) svg.removeChild(svg.firstChild);
            for (const child of keep) svg.append(child);
          }

          function node(name, attrs) {
            const el = document.createElementNS("http://www.w3.org/2000/svg", name);
            for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, String(value));
            return el;
          }

          function label(svg, value, x, y) {
            const el = node("text", { x, y, "font-size": 11, fill: "#111827" });
            el.textContent = value;
            svg.append(el);
          }

          function drawReliability(view) {
            clear(reliability);
            reliability.append(node("line", { x1: 44, y1: 196, x2: 330, y2: 196, stroke: "#374151" }));
            reliability.append(node("line", { x1: 44, y1: 24, x2: 44, y2: 196, stroke: "#374151" }));
            reliability.append(node("line", { x1: 44, y1: 196, x2: 330, y2: 24, stroke: "#9ca3af", "stroke-dasharray": "5 4" }));
            label(reliability, "conf", 306, 216);
            label(reliability, "acc", 12, 28);
            const band = 286 / view.bins;
            for (const row of view.rows) {
              const x = 44 + row.index * band;
              if (!row.count) {
                reliability.append(node("rect", { x, y: 196, width: Math.max(band - 2, 2), height: 0, fill: "#d1d5db" }));
                continue;
              }
              const accuracyHeight = row.accuracy * 172;
              const confidenceHeight = row.confidence * 172;
              reliability.append(node("rect", {
                x: x + 2,
                y: 196 - confidenceHeight,
                width: Math.max(band - 4, 2),
                height: confidenceHeight,
                fill: "#bfdbfe"
              }));
              reliability.append(node("rect", {
                x: x + 6,
                y: 196 - accuracyHeight,
                width: Math.max(band - 12, 2),
                height: accuracyHeight,
                fill: "#2563eb",
                opacity: 0.82
              }));
            }
          }

          function drawHistogram(view) {
            clear(histogram);
            histogram.append(node("line", { x1: 44, y1: 196, x2: 330, y2: 196, stroke: "#374151" }));
            histogram.append(node("line", { x1: 44, y1: 24, x2: 44, y2: 196, stroke: "#374151" }));
            label(histogram, "conf", 306, 216);
            label(histogram, "count", 10, 28);
            const maxCount = Math.max(...view.rows.map(row => row.count), 1);
            const band = 286 / view.bins;
            for (const row of view.rows) {
              const height = (row.count / maxCount) * 158;
              histogram.append(node("rect", {
                x: 44 + row.index * band + 3,
                y: 196 - height,
                width: Math.max(band - 6, 2),
                height,
                fill: "#059669"
              }));
            }
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
            const bins = String(Number.parseInt(input.value, 10));
            const view = data.by_bins[bins];
            summary.textContent =
              `Bins ${bins}: ECE ${Number(view.ece).toFixed(6)}, Brier ${Number(view.brier).toFixed(6)}, NLL ${Number(view.nll).toFixed(6)}`;
            binProbe.textContent =
              `Edge probe: confidence 0.50 -> bin ${view.edge_probe["0.50"]}, ` +
              `0.75 -> bin ${view.edge_probe["0.75"]}, 1.00 -> bin ${view.edge_probe["1.00"]}; ` +
              `non-empty bins ${view.rows.filter(row => row.count > 0).length}/${bins}`;
            runCount.textContent = `Probe runs: ${runs}`;
            drawReliability(view);
            drawHistogram(view);
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
        """))
        '''
        )
        .replace("__PROBABILITIES__", json.dumps(lesson_data["probabilities"]))
        .replace("__LABELS__", json.dumps(lesson_data["labels"]))
        .replace("__LESSON_DATA__", json.dumps(lesson_data, sort_keys=True))
        .lstrip()
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
