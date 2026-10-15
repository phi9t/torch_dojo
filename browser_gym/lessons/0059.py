"""Generated notebook cells for the 0059 stable cross-entropy lesson."""

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
        logits_probe = np.array([[1000., 1001.], [-1000., -999.]], dtype=np.float64)
        with np.errstate(over="ignore"):
            print("Naive exponentials:", np.exp(logits_probe).sum(axis=1))

        display(tensor_card("logits_probe", logits_probe, axes=("batch", "class")))

        stable_probe = {
            "logits": [[2.0, -1.0, 0.5], [-0.25, 0.75, 1.25]],
            "targets": [0, 2],
            "scales": [0.25, 0.5, 1.0, 1.5, 2.0, 3.0],
        }
        display(HTML(f"""
        <section id="dojo-0059-logit-scale" class="dojo-logit-scale-probe" role="region"
                 aria-labelledby="dojo-0059-logit-scale-title">
          <style>
            #dojo-0059-logit-scale {{
              border: 1px solid var(--dojo-rule, #d1d5db);
              border-radius: 8px;
              padding: 12px;
              max-width: 640px;
              background: var(--dojo-surface, #ffffff);
              color: var(--dojo-fg, #111827);
            }}
            #dojo-0059-logit-scale label {{
              display: grid;
              gap: 4px;
              font-weight: 600;
              margin: 8px 0;
            }}
            #dojo-0059-logit-scale input[type="range"] {{
              width: min(100%, 360px);
            }}
            #dojo-0059-logit-scale output,
            #dojo-0059-logit-scale .dojo-probe-runs {{
              display: block;
              margin-top: 6px;
              font-variant-numeric: tabular-nums;
            }}
            #dojo-0059-logit-scale svg {{
              display: block;
              width: 100%;
              min-width: 240px;
              max-width: 560px;
              height: 220px;
              margin-top: 10px;
            }}
            #dojo-0059-logit-scale svg:focus,
            #dojo-0059-logit-scale input:focus {{
              outline: 3px solid var(--dojo-accent, #2563eb);
              outline-offset: 2px;
            }}
          </style>
          <h3 id="dojo-0059-logit-scale-title">Cross-entropy as logits sharpen</h3>
          <label for="dojo-0059-logit-scale-control">
            Logit scale
            <input id="dojo-0059-logit-scale-control" aria-label="Logit scale"
                   type="range" min="0.25" max="3" step="0.25" value="1" />
          </label>
          <output id="dojo-0059-loss-value" for="dojo-0059-logit-scale-control"
                  aria-live="polite" data-testid="dojo-loss-value">
            Loss at scale 1.00: waiting for probe
          </output>
          <span class="dojo-probe-runs" data-testid="dojo-probe-runs">Probe runs: 0</span>
          <svg id="dojo-loss-plot" role="img" tabindex="0"
               aria-labelledby="dojo-loss-plot-title dojo-loss-plot-desc"
               viewBox="0 0 560 220">
            <title id="dojo-loss-plot-title">Loss plot for scaled logits</title>
            <desc id="dojo-loss-plot-desc">Cross-entropy loss across logit scale values.</desc>
            <line x1="44" y1="184" x2="532" y2="184" stroke="#374151" stroke-width="1" />
            <line x1="44" y1="20" x2="44" y2="184" stroke="#374151" stroke-width="1" />
            <text x="536" y="188" font-size="12">scale</text>
            <text x="8" y="20" font-size="12">loss</text>
            <polyline data-role="loss-line" fill="none" stroke="#0f766e" stroke-width="3" points="" />
            <circle data-role="current-point" r="5" fill="#b91c1c" cx="44" cy="184" />
          </svg>
        </section>
        <script type="application/javascript">
        (() => {{
          const root = document.getElementById("dojo-0059-logit-scale");
          if (!root || root.dataset.ready === "1") return;
          const probe = {json.dumps(stable_probe)};
          const input = root.querySelector("#dojo-0059-logit-scale-control");
          const lossValue = root.querySelector("[data-testid='dojo-loss-value']");
          const runCount = root.querySelector("[data-testid='dojo-probe-runs']");
          const line = root.querySelector("[data-role='loss-line']");
          const point = root.querySelector("[data-role='current-point']");
          let runs = 0;

          function stableLoss(scale) {{
            let total = 0;
            for (let row = 0; row < probe.logits.length; row++) {{
              const scaled = probe.logits[row].map(value => value * scale);
              const maxLogit = Math.max(...scaled);
              const shifted = scaled.map(value => value - maxLogit);
              const mass = shifted.reduce((sum, value) => sum + Math.exp(value), 0);
              total += Math.log(mass) - shifted[probe.targets[row]];
            }}
            return total / probe.logits.length;
          }}

          function project(scale, loss, minLoss, maxLoss) {{
            const minScale = probe.scales[0];
            const maxScale = probe.scales[probe.scales.length - 1];
            const x = 44 + ((scale - minScale) / (maxScale - minScale)) * 488;
            const range = Math.max(maxLoss - minLoss, 1e-9);
            const y = 184 - ((loss - minLoss) / range) * 164;
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
            const scale = Number.parseFloat(input.value);
            const losses = probe.scales.map(value => stableLoss(value));
            const minLoss = Math.min(...losses);
            const maxLoss = Math.max(...losses);
            const points = probe.scales.map((value, index) => (
              project(value, losses[index], minLoss, maxLoss).map(number => number.toFixed(1)).join(",")
            ));
            const currentLoss = stableLoss(scale);
            const [cx, cy] = project(scale, currentLoss, minLoss, maxLoss);
            line.setAttribute("points", points.join(" "));
            point.setAttribute("cx", cx.toFixed(1));
            point.setAttribute("cy", cy.toFixed(1));
            lossValue.textContent = `Loss at scale ${{scale.toFixed(2)}}: ${{currentLoss.toFixed(6)}}`;
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
