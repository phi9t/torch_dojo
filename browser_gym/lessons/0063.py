"""Generated notebook cells for the 0063 oriented-box sampling lesson."""

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
            # Evidence probe: sample points in an oriented box and verify inverse transform bounds.
            center = np.array([2.0, 3.0, 4.0], dtype=np.float64)
            size = np.array([2.0, 4.0, 6.0], dtype=np.float64)
            yaw = np.pi / 3

            def rotation_z(yaw_value: float) -> np.ndarray:
                c, s = float(np.cos(yaw_value)), float(np.sin(yaw_value))
                return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

            def sample_oriented_box_reference(center_value, size_value, yaw_value, count, rng):
                local = rng.uniform(-0.5, 0.5, size=(count, 3)) * size_value
                R = rotation_z(yaw_value)
                return local @ R.T + center_value

            rng = np.random.default_rng(0)
            samples = sample_oriented_box_reference(center, size, yaw, 1024, rng)
            display(tensor_card("samples", samples[:8], axes=("sample", "xyz")))

            R = rotation_z(yaw)
            local = (samples - center) @ R
            print("max abs local coord:", np.abs(local).max(axis=0))
            print("expected half size:", size / 2)

            # Visual probe: yaw sweeps rotate a local +x point toward world +y.
            display(HTML(r"""
            <section id="dojo-0063-yaw-probe" role="region" aria-labelledby="dojo-0063-yaw-title">
              <style>
                #dojo-0063-yaw-probe {
                  border: 1px solid var(--dojo-rule, #d1d5db);
                  border-radius: 8px;
                  padding: 12px;
                  max-width: 640px;
                  background: var(--dojo-surface, #ffffff);
                  color: var(--dojo-fg, #111827);
                }
                #dojo-0063-yaw-probe label {
                  display: grid;
                  gap: 4px;
                  font-weight: 600;
                  margin: 8px 0;
                }
                #dojo-0063-yaw-probe input[type="range"] {
                  width: min(100%, 360px);
                }
                #dojo-0063-yaw-probe output {
                  display: block;
                  margin-top: 6px;
                  font-variant-numeric: tabular-nums;
                }
                #dojo-0063-yaw-probe svg {
                  display: block;
                  width: 100%;
                  max-width: 560px;
                  height: 220px;
                  margin-top: 10px;
                }
              </style>
              <h3 id="dojo-0063-yaw-title">Yaw rotates local +x into world</h3>
              <label for="dojo-0063-yaw-control">
                Yaw (radians)
                <input id="dojo-0063-yaw-control" aria-label="Yaw angle"
                       type="range" min="-3.1416" max="3.1416" step="0.2618" value="0" />
              </label>
              <output id="dojo-0063-yaw-value" aria-live="polite" data-testid="dojo-yaw-value">
                yaw: waiting
              </output>
              <span class="dojo-probe-runs" data-testid="dojo-0063-probe-runs">Probe runs: 0</span>
              <svg id="dojo-0063-vector-plot" role="img" tabindex="0"
                   aria-labelledby="dojo-0063-vector-plot-title dojo-0063-vector-plot-desc"
                   viewBox="0 0 560 220">
                <title id="dojo-0063-vector-plot-title">Rotation of unit vector</title>
                <desc id="dojo-0063-vector-plot-desc">Shows the world direction of a local unit x vector after yaw rotation.</desc>
                <rect x="44" y="20" width="488" height="160" fill="#f9fafb" stroke="#d1d5db" />
                <line x1="288" y1="100" x2="288" y2="100" stroke="#111827" />
                <line x1="44" y1="100" x2="532" y2="100" stroke="#374151" stroke-width="1" />
                <line x1="288" y1="20" x2="288" y2="180" stroke="#374151" stroke-width="1" />
                <line id="dojo-0063-arrow" x1="288" y1="100" x2="388" y2="100" stroke="#0f766e" stroke-width="4" />
              </svg>
            </section>
            <script type="application/javascript">
            (() => {
              const root = document.getElementById("dojo-0063-yaw-probe");
              if (!root || root.dataset.ready === "1") return;
              const input = root.querySelector("#dojo-0063-yaw-control");
              const label = root.querySelector("[data-testid='dojo-yaw-value']");
              const runCount = root.querySelector("[data-testid='dojo-0063-probe-runs']");
              const arrow = root.querySelector("#dojo-0063-arrow");
              let runs = 0;
              function stepInput(direction) {
                const step = Number.parseFloat(input.step || "1");
                const min = Number.parseFloat(input.min);
                const max = Number.parseFloat(input.max);
                const next = Math.min(max, Math.max(min, Number.parseFloat(input.value) + direction * step));
                input.value = next.toFixed(4);
                update();
              }
              function update() {
                runs += 1;
                const yaw = Number.parseFloat(input.value);
                const c = Math.cos(yaw);
                const s = Math.sin(yaw);
                const x = 288 + c * 120;
                const y = 100 - s * 120;
                arrow.setAttribute("x2", x.toFixed(1));
                arrow.setAttribute("y2", y.toFixed(1));
                label.textContent = `yaw ${yaw.toFixed(3)} -> world dir [${c.toFixed(3)}, ${s.toFixed(3)}, 0]`;
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
