"""Generated notebook cells for the 0064 point-in-rotated-box lesson."""

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
            # Evidence probe: inverse-rotate points into box-local coordinates.
            center = np.array([3.0, -2.0, 1.0], dtype=np.float64)
            size = np.array([4.0, 2.0, 6.0], dtype=np.float64)
            yaw = np.pi / 3

            c, s = float(np.cos(yaw)), float(np.sin(yaw))
            R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

            local_points = np.array(
                [
                    [2.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, 3.0],
                    [2.0 + 5e-11, 0.0, 0.0],
                    [2.0 + 2e-9, 0.0, 0.0],
                ],
                dtype=np.float64,
            )
            world_points = local_points @ R.T + center
            display(tensor_card("world_points", world_points, axes=("point", "xyz")))

            local_recovered = (world_points - center) @ R
            display(tensor_card("local_recovered", local_recovered, axes=("point", "xyz")))
            print("abs(local) <= size/2:", np.abs(local_recovered) <= size / 2 + 1e-10)

            # Visual probe: slide a point along world x and track inside/outside for yaw=0 axis-aligned box.
            display(HTML(r"""
            <section id="dojo-0064-membership-probe" role="region" aria-labelledby="dojo-0064-membership-title">
              <style>
                #dojo-0064-membership-probe {
                  border: 1px solid var(--dojo-rule, #d1d5db);
                  border-radius: 8px;
                  padding: 12px;
                  max-width: 640px;
                  background: var(--dojo-surface, #ffffff);
                  color: var(--dojo-fg, #111827);
                }
                #dojo-0064-membership-probe label {
                  display: grid;
                  gap: 4px;
                  font-weight: 600;
                  margin: 8px 0;
                }
                #dojo-0064-membership-probe input[type="range"] {
                  width: min(100%, 360px);
                }
                #dojo-0064-membership-probe output {
                  display: block;
                  margin-top: 6px;
                  font-variant-numeric: tabular-nums;
                }
                #dojo-0064-membership-probe svg {
                  display: block;
                  width: 100%;
                  max-width: 560px;
                  height: 180px;
                  margin-top: 10px;
                }
              </style>
              <h3 id="dojo-0064-membership-title">Axis-aligned membership on a face</h3>
              <label for="dojo-0064-point-control">
                Point x coordinate (y=z=0, box center 0, size 2)
                <input id="dojo-0064-point-control" aria-label="Point x"
                       type="range" min="-1.5" max="1.5" step="0.1" value="1.0" />
              </label>
              <output id="dojo-0064-membership-value" aria-live="polite" data-testid="dojo-membership-value">
                membership: waiting
              </output>
              <span class="dojo-probe-runs" data-testid="dojo-0064-probe-runs">Probe runs: 0</span>
              <svg id="dojo-0064-membership-plot" role="img" tabindex="0"
                   aria-labelledby="dojo-0064-membership-plot-title dojo-0064-membership-plot-desc"
                   viewBox="0 0 560 180">
                <title id="dojo-0064-membership-plot-title">Point position</title>
                <desc id="dojo-0064-membership-plot-desc">Shows the point along x and whether it lies within the box interval [-1, 1].</desc>
                <line x1="44" y1="120" x2="532" y2="120" stroke="#374151" stroke-width="1" />
                <rect x="204" y="80" width="168" height="80" fill="#dcfce7" stroke="#16a34a" />
                <circle id="dojo-0064-point" r="7" fill="#b91c1c" cx="288" cy="120" />
                <text x="200" y="78" font-size="12">inside region</text>
              </svg>
            </section>
            <script type="application/javascript">
            (() => {
              const root = document.getElementById("dojo-0064-membership-probe");
              if (!root || root.dataset.ready === "1") return;
              const input = root.querySelector("#dojo-0064-point-control");
              const label = root.querySelector("[data-testid='dojo-membership-value']");
              const runCount = root.querySelector("[data-testid='dojo-0064-probe-runs']");
              const point = root.querySelector("#dojo-0064-point");
              let runs = 0;
              function stepInput(direction) {
                const step = Number.parseFloat(input.step || "1");
                const min = Number.parseFloat(input.min);
                const max = Number.parseFloat(input.max);
                const next = Math.min(max, Math.max(min, Number.parseFloat(input.value) + direction * step));
                input.value = next.toFixed(1);
                update();
              }
              function update() {
                runs += 1;
                const x = Number.parseFloat(input.value);
                const inside = Math.abs(x) <= 1 + 1e-10;
                const cx = 44 + ((x + 1.5) / 3.0) * 488;
                point.setAttribute("cx", cx.toFixed(1));
                label.textContent = `x=${x.toFixed(2)} -> inside: ${inside ? "True" : "False"}`;
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
