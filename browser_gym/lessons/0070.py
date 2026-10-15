"""Generated notebook cells for the 0070 anchor matching lesson."""

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
        anchor_iou = np.array([
            [0.20, 0.10],
            [0.45, 0.10],
            [0.62, 0.20],
            [0.10, 0.78],
            [0.52, 0.51],
        ], dtype=np.float64)
        collision_iou = np.array([
            [0.90, 0.80],
            [0.10, 0.20],
        ], dtype=np.float64)

        display(tensor_card("anchor_iou", anchor_iou, axes=("anchor", "ground truth")))
        display(tensor_card("collision_iou", collision_iou, axes=("anchor", "ground truth")))

        anchor_probe = {
            "iou": anchor_iou.tolist(),
            "negative_threshold": 0.30,
            "positive_thresholds": [0.40, 0.50, 0.60, 0.70],
        }
        collision_probe = {
            "iou": collision_iou.tolist(),
            "negative_threshold": 0.30,
            "positive_threshold": 0.50,
        }
        display(HTML(f"""
        <section id="dojo-0070-anchor-matching" class="dojo-anchor-matching-probe" role="region"
                 aria-labelledby="dojo-0070-anchor-matching-title">
          <style>
            #dojo-0070-anchor-matching {{
              border: 1px solid var(--dojo-rule, #d1d5db);
              border-radius: 8px;
              padding: 12px;
              max-width: 720px;
              background: var(--dojo-surface, #ffffff);
              color: var(--dojo-fg, #111827);
            }}
            #dojo-0070-anchor-matching label {{
              display: grid;
              gap: 4px;
              font-weight: 600;
              margin: 8px 0;
            }}
            #dojo-0070-anchor-matching input[type="range"] {{
              width: min(100%, 360px);
            }}
            #dojo-0070-anchor-matching output,
            #dojo-0070-anchor-matching .dojo-anchor-evidence {{
              display: block;
              margin-top: 8px;
              font-variant-numeric: tabular-nums;
            }}
            #dojo-0070-anchor-matching svg {{
              display: block;
              width: 100%;
              min-width: 260px;
              max-width: 640px;
              height: 230px;
              margin-top: 10px;
            }}
            #dojo-0070-anchor-matching svg:focus,
            #dojo-0070-anchor-matching input:focus {{
              outline: 3px solid var(--dojo-accent, #2563eb);
              outline-offset: 2px;
            }}
          </style>
          <h3 id="dojo-0070-anchor-matching-title">Anchor matching threshold</h3>
          <label for="dojo-0070-positive-threshold">
            Anchor matching threshold
            <input id="dojo-0070-positive-threshold" aria-label="Anchor matching threshold"
                   type="range" min="0.4" max="0.7" step="0.1" value="0.5" />
          </label>
          <output id="dojo-0070-anchor-labels" for="dojo-0070-positive-threshold"
                  aria-live="polite" data-testid="dojo-anchor-labels"></output>
          <output id="dojo-0070-anchor-shape" aria-live="polite"
                  data-testid="dojo-anchor-shape"></output>
          <span class="dojo-anchor-evidence" data-testid="dojo-anchor-collision"></span>
          <svg id="dojo-anchor-matching-plot" role="img" tabindex="0"
               aria-labelledby="dojo-anchor-matching-plot-title dojo-anchor-matching-plot-desc"
               viewBox="0 0 640 230">
            <title id="dojo-anchor-matching-plot-title">Anchor assignment plot</title>
            <desc id="dojo-anchor-matching-plot-desc">Anchor labels update as the positive threshold changes.</desc>
            <line x1="52" y1="184" x2="592" y2="184" stroke="#374151" stroke-width="1" />
            <text x="28" y="40" font-size="12">best IoU</text>
            <text x="596" y="188" font-size="12">anchor</text>
            <g data-role="bars"></g>
          </svg>
        </section>
        <script type="application/javascript">
        (() => {{
          const root = document.getElementById("dojo-0070-anchor-matching");
          if (!root || root.dataset.ready === "1") return;
          const anchorProbe = {json.dumps(anchor_probe)};
          const collisionProbe = {json.dumps(collision_probe)};
          const input = root.querySelector("#dojo-0070-positive-threshold");
          const labelsOutput = root.querySelector("[data-testid='dojo-anchor-labels']");
          const shapeOutput = root.querySelector("[data-testid='dojo-anchor-shape']");
          const collisionOutput = root.querySelector("[data-testid='dojo-anchor-collision']");
          const bars = root.querySelector("[data-role='bars']");

          function bestGt(row) {{
            let index = 0;
            for (let col = 1; col < row.length; col++) {{
              if (row[col] > row[index]) index = col;
            }}
            return index;
          }}

          function assign(matrix, negativeThreshold, positiveThreshold, forceBest) {{
            if (matrix.length === 0) return [];
            const groundTruths = matrix[0].length;
            if (groundTruths === 0) return matrix.map(() => -1);
            const result = matrix.map(row => {{
              const gt = bestGt(row);
              const value = row[gt];
              if (value < negativeThreshold) return -1;
              if (value >= positiveThreshold) return gt;
              return -2;
            }});
            if (!forceBest) return result;
            const claimed = new Set();
            for (let gt = 0; gt < groundTruths; gt++) {{
              let anchor = 0;
              for (let row = 1; row < matrix.length; row++) {{
                if (matrix[row][gt] > matrix[anchor][gt]) anchor = row;
              }}
              if (!claimed.has(anchor)) {{
                result[anchor] = gt;
                claimed.add(anchor);
              }}
            }}
            return result;
          }}

          function stepInput(direction) {{
            const step = Number.parseFloat(input.step || "0.1");
            const min = Number.parseFloat(input.min);
            const max = Number.parseFloat(input.max);
            const next = Math.min(max, Math.max(min, Number.parseFloat(input.value) + direction * step));
            input.value = next.toFixed(1);
            update();
          }}

          function labelText(label) {{
            if (label === -1) return "negative";
            if (label === -2) return "ignored";
            return `GT ${{label}}`;
          }}

          function renderBars(labels) {{
            bars.textContent = "";
            const bestValues = anchorProbe.iou.map(row => Math.max(...row));
            bestValues.forEach((value, index) => {{
              const x = 76 + index * 104;
              const height = Math.max(8, value * 150);
              const y = 184 - height;
              const fill = labels[index] === -1 ? "#9ca3af" : labels[index] === -2 ? "#d97706" : "#0f766e";
              const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
              rect.setAttribute("x", String(x));
              rect.setAttribute("y", y.toFixed(1));
              rect.setAttribute("width", "48");
              rect.setAttribute("height", height.toFixed(1));
              rect.setAttribute("rx", "4");
              rect.setAttribute("fill", fill);
              bars.append(rect);
              const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
              label.setAttribute("x", String(x + 24));
              label.setAttribute("y", "207");
              label.setAttribute("text-anchor", "middle");
              label.setAttribute("font-size", "12");
              label.textContent = `A${{index}}`;
              bars.append(label);
              const valueLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
              valueLabel.setAttribute("x", String(x + 24));
              valueLabel.setAttribute("y", String(Math.max(18, y - 6)));
              valueLabel.setAttribute("text-anchor", "middle");
              valueLabel.setAttribute("font-size", "12");
              valueLabel.textContent = `${{value.toFixed(2)}} / ${{labelText(labels[index])}}`;
              bars.append(valueLabel);
            }});
          }}

          function update() {{
            const positiveThreshold = Number.parseFloat(input.value);
            const labels = assign(anchorProbe.iou, anchorProbe.negative_threshold, positiveThreshold, true);
            const ordinary = assign(anchorProbe.iou, anchorProbe.negative_threshold, positiveThreshold, false);
            const collision = assign(collisionProbe.iou, collisionProbe.negative_threshold,
              collisionProbe.positive_threshold, true);
            labelsOutput.textContent =
              `Labels at positive threshold ${{positiveThreshold.toFixed(1)}}: [${{labels.join(", ")}}] ` +
              `(ordinary threshold labels: [${{ordinary.join(", ")}}])`;
            shapeOutput.textContent = `assignment shape: (${{labels.length}},), dtype: integer labels`;
            collisionOutput.textContent =
              `Collision evidence: IoU rows [[0.90, 0.80], [0.10, 0.20]] -> labels [${{collision.join(", ")}}]; ` +
              "GT 1 also prefers anchor 0, so the lower GT index keeps that anchor.";
            renderBars(labels);
          }}

          input.addEventListener("keydown", event => {{
            if (event.key === "ArrowRight" || event.key === "ArrowUp") {{
              event.preventDefault();
              stepInput(1);
            }} else if (event.key === "ArrowLeft" || event.key === "ArrowDown") {{
              event.preventDefault();
              stepInput(-1);
            }}
          }});
          input.addEventListener("input", update);
          input.addEventListener("change", update);
          root.dataset.ready = "1";
          update();
        }})();
        </script>
        """))
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
