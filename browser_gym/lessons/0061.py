"""Generated notebook cells for the 0061 K-means lesson."""

from __future__ import annotations

import json
import textwrap

from browser_gym.lesson_ui import notebook_tensor_card_source


def cells(exercise_id: str) -> list[dict]:
    # Fixed canonical probe data matching the 0061 baseline prompt.
    # Two clusters of two 1D points with well-separated initial centers.
    probe = {
        "points": [[0.0], [1.0], [9.0], [10.0], [2.0], [3.0], [8.0], [11.0]],
        "initial_centers": [[0.0], [10.0]],
        "k_options": [2, 3],
        "initial_centers_k3": [[0.0], [2.0], [10.0]],
        # Duplicate-points empty-center probe (see followup stage).
        "followup_points": [[0.0], [0.0], [2.0]],
        "followup_centers": [[0.0], [2.0], [50.0]],
        "max_iter_options": list(range(0, 11)),
        "default_max_iter": 5,
    }

    setup_and_tensor_cards = (
        textwrap.dedent(
            '''
        import json
        import numpy as np
        from IPython.display import HTML, display, clear_output

        '''
        ).lstrip()
        + notebook_tensor_card_source()
        + "\n\n"
        + textwrap.dedent(
            f'''
        probe_points = np.array({json.dumps(probe["points"])}, dtype=np.float64)
        probe_initial_centers = np.array({json.dumps(probe["initial_centers"])}, dtype=np.float64)
        followup_points = np.array({json.dumps(probe["followup_points"])}, dtype=np.float64)
        followup_centers = np.array({json.dumps(probe["followup_centers"])}, dtype=np.float64)

        print("Baseline probe: N={{}}, D={{}} points, K={{}} initial centers".format(
            probe_points.shape[0], probe_points.shape[1], probe_initial_centers.shape[0]
        ))
        display(tensor_card("probe_points", probe_points, axes=("point", "dimension")))
        display(tensor_card("probe_initial_centers", probe_initial_centers, axes=("cluster", "dimension")))

        # Reference probe runner so the browser notebook can also compute tensor evidence before
        # the learner implements kmeans. Runs in the exploration kernel only; it is never submitted.
        def _probe_assign(X, centers, block_size=16):
            labels = np.empty(len(X), dtype=np.int64)
            nearest = np.empty(len(X), dtype=np.float64)
            for start in range(0, len(X), block_size):
                pts = X[start:start + block_size]
                best = np.full(len(pts), np.inf)
                winners = np.zeros(len(pts), dtype=np.int64)
                for idx, c in enumerate(centers):
                    delta = pts - c
                    dist = np.einsum("nd,nd->n", delta, delta)
                    improved = dist < best
                    best[improved] = dist[improved]
                    winners[improved] = idx
                labels[start:start + len(pts)] = winners
                nearest[start:start + len(pts)] = best
            return labels, nearest

        def _probe_kmeans(X, initial_centers, max_iter, block_size=16):
            centers = np.asarray(initial_centers, dtype=float).copy()
            inertias = []
            for step in range(max_iter + 1):
                labels, dist = _probe_assign(X, centers, block_size)
                inertias.append(float(dist.sum()))
                if step == max_iter:
                    break
                updated = centers.copy()
                for idx in range(len(centers)):
                    selected = labels == idx
                    if np.any(selected):
                        updated[idx] = X[selected].mean(axis=0)
                if np.array_equal(updated, centers):
                    centers = updated
                    labels, dist = _probe_assign(X, centers, block_size)
                    inertias.append(float(dist.sum()))
                    break
                centers = updated
            labels, dist = _probe_assign(X, centers, block_size)
            if not inertias or not np.isclose(inertias[-1], float(dist.sum())):
                inertias.append(float(dist.sum()))
            return centers, labels, inertias

        _probe_initial_labels, _probe_initial_dist = _probe_assign(probe_points, probe_initial_centers)
        display(tensor_card("initial_assignment_labels", _probe_initial_labels.astype(np.int64), axes=("point",)))

        # Before running the interactive probe, write down your predicted first assignment
        # for X=[0,1,9,10] with centers=[0,10]. The probe records every re-run so you can
        # compare hand predictions with the actual iterates.
        '''
        ).strip()
        + "\n"
    )

    interactive_probe_html = textwrap.dedent(
        rf'''
        kmeans_probe = {json.dumps(probe)}
        display(HTML("""
        <section id="dojo-0061-kmeans" class="dojo-kmeans-probe" role="region"
                 aria-labelledby="dojo-0061-kmeans-title"
                 data-probe=''' + "'" + json.dumps(probe) + "'" + r'''>
          <style>
            #dojo-0061-kmeans {{
              border: 1px solid var(--dojo-rule, #d1d5db);
              border-radius: 8px;
              padding: 12px;
              max-width: 720px;
              background: var(--dojo-surface, #ffffff);
              color: var(--dojo-fg, #111827);
            }}
            #dojo-0061-kmeans h3 {{ margin: 0 0 8px; }}
            #dojo-0061-kmeans .dojo-controls {{
              display: grid;
              grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
              gap: 10px 16px;
              margin: 10px 0 14px;
            }}
            #dojo-0061-kmeans label {{
              display: grid;
              gap: 4px;
              font-weight: 600;
            }}
            #dojo-0061-kmeans input[type="range"] {{ width: 100%; }}
            #dojo-0061-kmeans button {{
              padding: 8px 14px;
              font-weight: 600;
              background: #2563eb;
              color: #fff;
              border: none;
              border-radius: 6px;
              cursor: pointer;
            }}
            #dojo-0061-kmeans button:hover {{ background: #1d4ed8; }}
            #dojo-0061-kmeans button:focus,
            #dojo-0061-kmeans input:focus,
            #dojo-0061-kmeans select:focus,
            #dojo-0061-kmeans svg:focus {{
              outline: 3px solid var(--dojo-accent, #2563eb);
              outline-offset: 2px;
            }}
            #dojo-0061-kmeans .dojo-readout {{
              font-variant-numeric: tabular-nums;
              background: var(--dojo-bg, #f9fafb);
              border: 1px solid var(--dojo-rule, #e5e7eb);
              border-radius: 6px;
              padding: 8px 10px;
              margin: 6px 0 12px;
            }}
            #dojo-0061-kmeans svg {{
              display: block;
              width: 100%;
              height: auto;
              border: 1px solid var(--dojo-rule, #e5e7eb);
              border-radius: 6px;
              background: var(--dojo-surface, #ffffff);
            }}
            #dojo-0061-kmeans .dojo-svg-row {{
              display: grid;
              grid-template-columns: 1fr 1fr;
              gap: 10px;
            }}
            @media (max-width: 520px) {{
              #dojo-0061-kmeans .dojo-svg-row {{ grid-template-columns: 1fr; }}
            }}
          </style>
          <h3 id="dojo-0061-kmeans-title">K-means assignment and convergence probe</h3>
          <div class="dojo-controls">
            <label for="dojo-0061-k">
              Cluster count (K)
              <select id="dojo-0061-k" aria-label="Cluster count (K)">
                <option value="2" selected>2 centers</option>
                <option value="3">3 centers (followup empty-center probe)</option>
              </select>
            </label>
            <label for="dojo-0061-max-iter">
              Max iterations
              <input id="dojo-0061-max-iter" type="range"
                     min="0" max="10" step="1" value="5" aria-label="Max iterations" />
            </label>
            <div class="dojo-run-control">
              <button id="dojo-0061-run" type="button"
                      data-testid="dojo-0061-run">Run k-means probe</button>
            </div>
          </div>
          <div class="dojo-readout" id="dojo-0061-readout" role="status" aria-live="polite"
               data-testid="dojo-0061-readout">
            Adjust K and max-iter, then click "Run k-means probe" to see assignments and convergence.
          </div>
          <div class="dojo-svg-row">
            <svg id="dojo-0061-cluster-plot" role="img" tabindex="0"
                 aria-labelledby="dojo-0061-cluster-title dojo-0061-cluster-desc"
                 viewBox="0 0 360 220">
              <title id="dojo-0061-cluster-title">Cluster assignments</title>
              <desc id="dojo-0061-cluster-desc">1D scatter of point positions colored by their assigned cluster, with current centroids highlighted.</desc>
              <line x1="24" y1="180" x2="336" y2="180" stroke="#374151" stroke-width="1" />
              <text x="340" y="184" font-size="10">x</text>
              <text x="6" y="180" font-size="10" text-anchor="end">D=1</text>
              <g id="dojo-0061-cluster-points" fill="#6b7280"></g>
              <g id="dojo-0061-cluster-centroids" fill="#b91c1c"></g>
            </svg>
            <svg id="dojo-0061-convergence-plot" role="img" tabindex="0"
                 aria-labelledby="dojo-0061-conv-title dojo-0061-conv-desc"
                 viewBox="0 0 360 220"
                 data-testid="dojo-kmeans-plot">
              <title id="dojo-0061-conv-title">Convergence (inertia per iteration)</title>
              <desc id="dojo-0061-conv-desc">Line plot of inertia (sum of within-cluster squared distances) at each Lloyd iteration.</desc>
              <line x1="36" y1="180" x2="340" y2="180" stroke="#374151" stroke-width="1" />
              <line x1="36" y1="20" x2="36" y2="180" stroke="#374151" stroke-width="1" />
              <text x="344" y="192" font-size="10">iteration</text>
              <text x="2" y="20" font-size="10" text-anchor="start">inertia</text>
              <polyline id="dojo-0061-inertia-line" fill="none" stroke="#0f766e" stroke-width="3" points="" />
              <g id="dojo-0061-inertia-dots"></g>
            </svg>
          </div>
          <div class="dojo-readout" id="dojo-0061-evidence" data-testid="dojo-0061-evidence"
               style="margin-top:10px">
            Probe runs: 0 · No shape evidence yet.
          </div>
        </section>
        <script type="application/javascript">
        (() => {{
          const root = document.getElementById("dojo-0061-kmeans");
          if (!root || root.dataset.ready === "1") return;
          const probe = {json.dumps(probe)};
          const COLORS = ["#2563eb", "#059669", "#b45309", "#7c3aed", "#be123c"];

          const kSel = root.querySelector("#dojo-0061-k");
          const maxIter = root.querySelector("#dojo-0061-max-iter");
          const runBtn = root.querySelector("#dojo-0061-run");
          const readout = root.querySelector("#dojo-0061-readout");
          const evidence = root.querySelector("#dojo-0061-evidence");
          const ptsG = root.querySelector("#dojo-0061-cluster-points");
          const cenG = root.querySelector("#dojo-0061-cluster-centroids");
          const inertiaLine = root.querySelector("#dojo-0061-inertia-line");
          const inertiaDots = root.querySelector("#dojo-0061-inertia-dots");

          let runs = 0;

          function assign(X, centers) {{
            const N = X.length, K = centers.length;
            const labels = new Array(N).fill(0);
            const dists = new Array(N).fill(0);
            for (let i = 0; i < N; i++) {{
              let best = Infinity, winner = 0;
              for (let k = 0; k < K; k++) {{
                let s = 0;
                for (let d = 0; d < X[i].length; d++) {{
                  const diff = X[i][d] - centers[k][d];
                  s += diff * diff;
                }}
                if (s < best) {{ best = s; winner = k; }}
              }}
              labels[i] = winner;
              dists[i] = best;
            }}
            return {{ labels, dists }};
          }}

          function lloyd(X, initial, iterations) {{
            let centers = initial.map(row => row.slice());
            const inertias = [];
            for (let step = 0; step <= iterations; step++) {{
              // Inertia at current centers (even before first update).
              const {{ labels, dists }} = assign(X, centers);
              inertias.push(dists.reduce((a, b) => a + b, 0));
              if (step === iterations) return {{ labels, centers, inertias, convergedAt: step }};
              const next = centers.map(row => row.slice());
              const counts = new Array(centers.length).fill(0);
              for (let i = 0; i < X.length; i++) {{
                counts[labels[i]] += 1;
                for (let d = 0; d < X[i].length; d++) next[labels[i]][d] += X[i][d];
              }}
              let changed = false;
              for (let k = 0; k < centers.length; k++) {{
                if (counts[k] === 0) continue;  // empty cluster keeps previous center
                for (let d = 0; d < next[k].length; d++) {{
                  const v = next[k][d] / counts[k];
                  if (v !== centers[k][d]) changed = true;
                  next[k][d] = v;
                }}
              }}
              if (!changed) {{
                // One final reassign for returned labels correctness, inertia already captured for current centers.
                const post = assign(X, next);
                inertias.push(post.dists.reduce((a, b) => a + b, 0));
                return {{ labels: post.labels, centers: next, inertias, convergedAt: step }};
              }}
              centers = next;
            }}
            const final = assign(X, centers);
            return {{ labels: final.labels, centers, inertias, convergedAt: iterations }};
          }}

          function setupData(K) {{
            if (K === 3) {{
              return {{ X: probe.followup_points.map(r => r.slice()), centers0: probe.followup_centers.map(r => r.slice()) }};
            }}
            return {{ X: probe.points.map(r => r.slice()), centers0: probe.initial_centers.map(r => r.slice()) }};
          }}

          function drawCluster(X, centers, labels) {{
            const N = X.length, K = centers.length;
            const minX = Math.min(...X.map(r => r[0]), ...centers.map(r => r[0])) - 0.5;
            const maxX = Math.max(...X.map(r => r[0]), ...centers.map(r => r[0])) + 0.5;
            const spanX = Math.max(maxX - minX, 1e-6);
            const project = (value) => 24 + ((value - minX) / spanX) * (336 - 24);
            ptsG.innerHTML = "";
            cenG.innerHTML = "";
            for (let i = 0; i < N; i++) {{
              const cx = project(X[i][0]).toFixed(1);
              const cy = 120 + ((i * 13) % 40) - 20;
              const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
              circle.setAttribute("cx", cx);
              circle.setAttribute("cy", String(cy));
              circle.setAttribute("r", "6");
              circle.setAttribute("fill", COLORS[labels[i] % COLORS.length]);
              circle.setAttribute("stroke", "#111827");
              circle.setAttribute("stroke-width", "1");
              ptsG.appendChild(circle);
            }}
            for (let k = 0; k < K; k++) {{
              const cx = project(centers[k][0]).toFixed(1);
              const tri = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
              const y1 = 158, y2 = 174, y3 = 158;
              tri.setAttribute("points", `${{cx - 7}},${{y1}} ${{cx}},${{y2}} ${{cx + 7}},${{y3}}`);
              tri.setAttribute("fill", COLORS[k % COLORS.length]);
              tri.setAttribute("stroke", "#7f1d1d");
              tri.setAttribute("stroke-width", "2");
              cenG.appendChild(tri);
            }}
          }}

          function drawConvergence(inertias) {{
            const steps = inertias.length - 1;
            const xMin = 0, xMax = Math.max(steps, 1);
            const yMin = Math.min(...inertias, 0);
            const yMax = Math.max(...inertias, 1);
            const ySpan = Math.max(yMax - yMin, 1e-6);
            const px = (i) => 36 + ((i - xMin) / (xMax - xMin)) * (340 - 36);
            const py = (v) => 180 - ((v - yMin) / ySpan) * (180 - 20);
            const pts = inertias.map((v, i) => `${{px(i).toFixed(1)}},${{py(v).toFixed(1)}}`);
            inertiaLine.setAttribute("points", pts.join(" "));
            inertiaDots.innerHTML = "";
            inertias.forEach((v, i) => {{
              const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
              c.setAttribute("cx", px(i).toFixed(1));
              c.setAttribute("cy", py(v).toFixed(1));
              c.setAttribute("r", "3");
              c.setAttribute("fill", "#b91c1c");
              inertiaDots.appendChild(c);
            }});
          }}

          function fmtVals(arr, fmt = v => v.toFixed(2)) {{
            return "[" + arr.map(fmt).join(", ") + "]";
          }}

          function run() {{
            runs += 1;
            const K = Number.parseInt(kSel.value, 10);
            const iters = Number.parseInt(maxIter.value, 10);
            const {{ X, centers0 }} = setupData(K);
            const result = lloyd(X, centers0, iters);
            drawCluster(X, result.centers, result.labels);
            drawConvergence(result.inertias);

            const N = X.length, D = X[0].length;
            const inertia = result.inertias[result.inertias.length - 1];
            readout.textContent =
              `K=${{K}}, max_iter=${{iters}}, convergedAt=${{result.convergedAt}}\\n` +
              `points shape (N,D) = (${{N}},${{D}})  labels shape (N,) = (${{result.labels.length}},)\\n` +
              `assignments labels = ${{fmtVals(result.labels, v => String(v))}}\\n` +
              `final centers = ${{result.centers.map(r => `[${{r.map(v => v.toFixed(3)).join(", ")}}]`).join(", ")}}\\n` +
              `final inertia = ${{inertia.toFixed(4)}}`;
            evidence.textContent =
              `Probe runs: ${{runs}} · shapes points(${{N}},${{D}}) centers(K=${{K}},${{D}}) labels(${{result.labels.length}}) ` +
              `· inertia steps = [${{result.inertias.map(v => v.toFixed(2)).join(", ")}}]`;
          }}

          maxIter.addEventListener("keydown", event => {{
            if (event.key === "ArrowRight" || event.key === "ArrowUp") {{
              event.preventDefault();
              const v = Math.min(10, Number(maxIter.value) + 1);
              maxIter.value = String(v);
            }} else if (event.key === "ArrowLeft" || event.key === "ArrowDown") {{
              event.preventDefault();
              const v = Math.max(0, Number(maxIter.value) - 1);
              maxIter.value = String(v);
            }}
          }});
          runBtn.addEventListener("click", run);
          kSel.addEventListener("change", run);
          root.dataset.ready = "1";
          // Initial render so tests can assert plot nodes exist even before explicit click.
          run();
        }})();
        </script>
        """))
        '''
    ).lstrip()
    # The HTML block above is authored with doubled braces so it can live next
    # to generator-time string interpolation. The emitted notebook cell is
    # ordinary Python/JavaScript, so normalize those escapes before writing it.
    interactive_probe_html = (
        interactive_probe_html
        .replace("{{", "{")
        .replace("}}", "}")
        .replace("{json.dumps(probe)}", json.dumps(probe))
    )

    full_source = setup_and_tensor_cards + "\n\n" + interactive_probe_html

    return [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"dojo": {"exercise_id": exercise_id, "role": "exploration"}},
            "outputs": [],
            "source": full_source.splitlines(keepends=True),
        }
    ]
