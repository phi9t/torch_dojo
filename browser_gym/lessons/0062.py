"""Generated notebook cells for the 0062 pairwise IoU lesson."""

from __future__ import annotations

import textwrap

from browser_gym.lesson_ui import notebook_tensor_card_source


def cells(exercise_id: str) -> list[dict]:
    source = (
        textwrap.dedent(
            """
            import numpy as np
            from IPython.display import HTML, display

            """
        ).lstrip()
        + notebook_tensor_card_source()
        + "\n\n"
        + textwrap.dedent(
            r'''
            # Exercise probe: overlap and union under continuous xyxy boxes.
            left = np.array([[0.0, 0.0, 2.0, 2.0], [2.0, 0.0, 3.0, 1.0]], dtype=np.float64)
            right = np.array([[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0]], dtype=np.float64)

            display(tensor_card("left", left, axes=("box", "xyxy")))
            display(tensor_card("right", right, axes=("box", "xyxy")))

            width = np.maximum(0.0, np.minimum(left[:, None, 2], right[None, :, 2]) - np.maximum(left[:, None, 0], right[None, :, 0]))
            height = np.maximum(0.0, np.minimum(left[:, None, 3], right[None, :, 3]) - np.maximum(left[:, None, 1], right[None, :, 1]))
            intersection = width * height
            area_left = (left[:, 2] - left[:, 0]) * (left[:, 3] - left[:, 1])
            area_right = (right[:, 2] - right[:, 0]) * (right[:, 3] - right[:, 1])
            union = area_left[:, None] + area_right[None, :] - intersection
            iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
            display(tensor_card("iou", iou, axes=("left", "right")))

            # Visual probe: slide the right box and observe IoU against the first left box.
            probe = {"shifts": [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]}
            display(HTML(f"""
            <section id="dojo-0062-iou-shift" class="dojo-iou-shift-probe" role="region"
                     aria-labelledby="dojo-0062-iou-shift-title">
              <style>
                #dojo-0062-iou-shift {{
                  border: 1px solid var(--dojo-rule, #d1d5db);
                  border-radius: 8px;
                  padding: 12px;
                  max-width: 640px;
                  background: var(--dojo-surface, #ffffff);
                  color: var(--dojo-fg, #111827);
                }}
                #dojo-0062-iou-shift label {{
                  display: grid;
                  gap: 4px;
                  font-weight: 600;
                  margin: 8px 0;
                }}
                #dojo-0062-iou-shift input[type="range"] {{
                  width: min(100%, 360px);
                }}
                #dojo-0062-iou-shift output {{
                  display: block;
                  margin-top: 6px;
                  font-variant-numeric: tabular-nums;
                }}
                #dojo-0062-iou-shift svg {{
                  display: block;
                  width: 100%;
                  max-width: 560px;
                  height: 180px;
                  margin-top: 10px;
                }}
              </style>
              <h3 id="dojo-0062-iou-shift-title">IoU vs. horizontal shift</h3>
              <label for="dojo-0062-iou-shift-control">
                Shift x of the right box
                <input id="dojo-0062-iou-shift-control" aria-label="IoU shift"
                       type="range" min="0" max="2" step="0.25" value="0.5" />
              </label>
              <output id="dojo-0062-iou-value" aria-live="polite" data-testid="dojo-iou-value">
                IoU: waiting for probe
              </output>
              <span class="dojo-probe-runs" data-testid="dojo-0062-probe-runs">Probe runs: 0</span>
              <svg id="dojo-0062-iou-plot" role="img" tabindex="0"
                   aria-labelledby="dojo-0062-iou-plot-title dojo-0062-iou-plot-desc"
                   viewBox="0 0 560 180">
                <title id="dojo-0062-iou-plot-title">IoU curve</title>
                <desc id="dojo-0062-iou-plot-desc">IoU across a horizontal shift parameter.</desc>
                <line x1="44" y1="148" x2="532" y2="148" stroke="#374151" stroke-width="1" />
                <line x1="44" y1="20" x2="44" y2="148" stroke="#374151" stroke-width="1" />
                <text x="536" y="152" font-size="12">shift</text>
                <text x="8" y="20" font-size="12">IoU</text>
                <polyline data-role="iou-line" fill="none" stroke="#0f766e" stroke-width="3" points="" />
                <circle data-role="current-point" r="5" fill="#b91c1c" cx="44" cy="148" />
              </svg>
            </section>
            <script type="application/javascript">
            (() => {{
              const root = document.getElementById("dojo-0062-iou-shift");
              if (!root || root.dataset.ready === "1") return;
              const shifts = {probe["shifts"]};
              const input = root.querySelector("#dojo-0062-iou-shift-control");
              const label = root.querySelector("[data-testid='dojo-iou-value']");
              const runCount = root.querySelector("[data-testid='dojo-0062-probe-runs']");
              const line = root.querySelector("[data-role='iou-line']");
              const point = root.querySelector("[data-role='current-point']");
              let runs = 0;
              const baseShifted = shifts.map(shift => {{
                const overlapWidth = Math.max(0, 2 - Math.max(0, shift));
                const intersection = overlapWidth * 2;
                const union = 4 + 4 - intersection;
                return union > 0 ? intersection / union : 0;
              }});

              function project(shift, iou) {{
                const minShift = shifts[0];
                const maxShift = shifts[shifts.length - 1];
                const x = 44 + ((shift - minShift) / (maxShift - minShift)) * 488;
                const y = 148 - iou * 128;
                return [x, y];
              }}

              function stepInput(direction) {{
                const step = Number.parseFloat(input.step || "1");
                const min = Number.parseFloat(input.min);
                const max = Number.parseFloat(input.max);
                const next = Math.min(max, Math.max(min, Number.parseFloat(input.value) + direction * step));
                input.value = next.toFixed(2);
                update();
              }}

              function update() {{
                runs += 1;
                const shift = Number.parseFloat(input.value);
                const points = shifts.map((value, idx) => project(value, baseShifted[idx]).map(num => num.toFixed(1)).join(","));
                line.setAttribute("points", points.join(" "));
                const current = (() => {{
                  const overlapWidth = Math.max(0, 2 - Math.max(0, shift));
                  const intersection = overlapWidth * 2;
                  const union = 4 + 4 - intersection;
                  return union > 0 ? intersection / union : 0;
                }})();
                const [cx, cy] = project(shift, current);
                point.setAttribute("cx", cx.toFixed(1));
                point.setAttribute("cy", cy.toFixed(1));
                label.textContent = `IoU at shift ${{shift.toFixed(2)}}: ${{current.toFixed(6)}}`;
                runCount.textContent = `Probe runs: ${{runs}}`;
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
