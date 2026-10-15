"""Generated notebook cells for the 0066 sparse voxel pooling lesson."""

from __future__ import annotations

import json
import textwrap

import numpy as np

from browser_gym.lesson_ui import notebook_tensor_card_source


def cells(exercise_id: str) -> list[dict]:
    points = np.array([[0.1, 0.1, 0.1], [0.9, 0.1, 0.1], [1.0, 0.0, 0.0]], dtype=np.float64)
    features = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    origin = np.zeros(3, dtype=np.float64)
    voxel_size = np.ones(3, dtype=np.float64)
    shape = np.array([2, 2, 2], dtype=np.int64)

    # Precompute canonical baseline result (matches eval reference logic).
    upper = origin + shape.astype(np.float64) * voxel_size
    inside = np.all((points >= origin) & (points < upper), axis=1)
    indices = np.floor((points[inside] - origin) / voxel_size).astype(np.int64)
    indices = np.minimum(indices, shape - 1)
    unique, local_inverse, counts = np.unique(indices, axis=0, return_inverse=True, return_counts=True)
    sums = np.zeros((len(unique), features.shape[1]), dtype=np.float64)
    np.add.at(sums, local_inverse, features[inside])
    means = sums / counts[:, None]
    inverse = np.full(len(points), -1, dtype=np.int64)
    inverse[inside] = local_inverse

    probe = {
        "points": points.tolist(),
        "features": features.tolist(),
        "unique": unique.tolist(),
        "counts": counts.tolist(),
        "means": means.tolist(),
        "inverse": inverse.tolist(),
        "voxel_size_initial": 1.0,
    }

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
            f'''
        probe = {json.dumps(probe)}
        points0 = np.array(probe["points"], dtype=np.float64)
        features0 = np.array(probe["features"], dtype=np.float64)
        origin = np.zeros(3)

        def _voxel_ref(points, features, vs):
            "Reference-only probe for voxel coordinates, mean features, and counts."
            shape = np.array([2, 2, 2])
            upper = origin + shape * vs
            inside = np.all((points >= origin) & (points < upper), axis=1)
            if not inside.any():
                return np.zeros((0,3), dtype=np.int64), np.full(len(points), -1, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros((0, features.shape[1]))
            idx = np.floor((points[inside] - origin) / vs).astype(np.int64)
            idx = np.minimum(idx, shape - 1)
            u, inv, n = np.unique(idx, axis=0, return_inverse=True, return_counts=True)
            sums = np.zeros((len(u), features.shape[1]))
            np.add.at(sums, inv, features[inside])
            means = sums / n[:, None]
            full_inv = np.full(len(points), -1, dtype=np.int64)
            full_inv[inside] = inv
            return u, full_inv, n, means

        display(tensor_card("points", points0, axes=("N", "xyz")))
        display(tensor_card("features", features0, axes=("N", "C")))

        display(HTML("""
        <section id="dojo-0066-voxel-probe" class="dojo-voxel-probe" role="region"
                 aria-labelledby="dojo-0066-voxel-title">
          <style>
            #dojo-0066-voxel-probe {{
              border: 1px solid var(--dojo-rule, #d1d5db);
              border-radius: 8px;
              padding: 12px;
              max-width: 680px;
              background: var(--dojo-surface, #ffffff);
              color: var(--dojo-fg, #111827);
            }}
            #dojo-0066-voxel-probe label {{
              display: grid;
              gap: 4px;
              font-weight: 600;
              margin: 8px 0;
            }}
            #dojo-0066-voxel-probe input[type="range"] {{
              width: min(100%, 360px);
            }}
            #dojo-0066-voxel-probe output {{
              display: block;
              margin-top: 4px;
              font-variant-numeric: tabular-nums;
            }}
            #dojo-0066-voxel-probe svg {{
              display: block;
              width: 100%;
              max-width: 560px;
              height: 240px;
              margin-top: 10px;
            }}
            #dojo-0066-voxel-probe svg:focus,
            #dojo-0066-voxel-probe input:focus {{
              outline: 3px solid var(--dojo-accent, #2563eb);
              outline-offset: 2px;
            }}
            #dojo-0066-voxel-probe .stat {{
              font-family: monospace;
              margin-top: 6px;
              white-space: pre-wrap;
            }}
          </style>
          <h3 id="dojo-0066-voxel-title">Voxel contributions at varying voxel size</h3>
          <p>Change voxel size to see which points fall into the same cell and how
             the pooled mean changes. Points on the max boundary stay outside
             (half-open interval). Probe only.</p>
          <label for="dojo-0066-vs">
            Voxel size (uniform)
            <input id="dojo-0066-vs" aria-label="Voxel size"
                   type="range" min="0.3" max="2.0" step="0.1" value="1.0" />
          </label>
          <output id="dojo-0066-vs-out" for="dojo-0066-vs" data-testid="dojo-0066-vs-out">
            voxel_size = 1.00
          </output>
          <output id="dojo-0066-stats" class="stat" data-testid="dojo-0066-stats" aria-live="polite">
            occupied=2 counts=[2,1]
          </output>
          <svg id="dojo-0066-grid-plot" role="img" tabindex="0"
               aria-labelledby="dojo-0066-vplot-title dojo-0066-vplot-desc"
               viewBox="0 0 560 260">
            <title id="dojo-0066-vplot-title">Voxel occupancy in the x-y plane</title>
            <desc id="dojo-0066-vplot-desc">Top-down view showing input points and their voxel cells colored by occupancy count.</desc>
            <g data-role="grid" stroke="#9ca3af" stroke-width="1"></g>
            <g data-role="cells"></g>
            <g data-role="points"></g>
            <text x="16" y="20" font-size="11" fill="#374151">y</text>
            <text x="540" y="248" font-size="11" fill="#374151">x</text>
          </svg>
        </section>
        <script type="application/javascript">
        (() => {{
          const root = document.getElementById("dojo-0066-voxel-probe");
          if (!root || root.dataset.ready === "1") return;
          const vsEl = root.querySelector("#dojo-0066-vs");
          const vsOut = root.querySelector("#dojo-0066-vs-out");
          const statsOut = root.querySelector("#dojo-0066-stats");
          const gridG = root.querySelector("[data-role='grid']");
          const cellsG = root.querySelector("[data-role='cells']");
          const pointsG = root.querySelector("[data-role='points']");

          const points = {json.dumps(points.tolist())};
          const SHAPE = [2, 2, 2];
          const PAD = 40;
          const SIZE = 180;
          const RANGE = SHAPE[0];  // grid spans up to shape*vs; vs=1 -> [0,2)
          const ORIG = PAD;

          function scale(x, vs) {{
            return ORIG + (x / (RANGE * vs)) * SIZE;
          }}

          function update() {{
            const vs = Number.parseFloat(vsEl.value);
            const upper = RANGE * vs;
            vsOut.textContent = `voxel_size = ${{vs.toFixed(2)}}`;

            // Grid lines
            gridG.innerHTML = "";
            for (let i = 0; i <= RANGE; i++) {{
              const t = ORIG + (i / RANGE) * SIZE;
              gridG.insertAdjacentHTML("beforeend",
                `<line x1="${{t}}" y1="${{ORIG}}" x2="${{t}}" y2="${{ORIG + SIZE}}" />`);
              gridG.insertAdjacentHTML("beforeend",
                `<line x1="${{ORIG}}" y1="${{t}}" x2="${{ORIG + SIZE}}" y2="${{t}}" />`);
            }}

            // Compute voxel membership in JS for the probe
            const coords = points.map(([x, y]) => {{
              if (x < 0 || y < 0 || x >= upper || y >= upper) return null;
              return [Math.min(Math.floor(x / vs), SHAPE[0] - 1),
                      Math.min(Math.floor(y / vs), SHAPE[1] - 1)];
            }});
            const buckets = new Map();
            coords.forEach((c, idx) => {{
              if (!c) return;
              const key = c.join(",");
              if (!buckets.has(key)) buckets.set(key, {{ c, n: 0, indices: [] }});
              buckets.get(key).n += 1;
              buckets.get(key).indices.push(idx);
            }});

            // Cells colored by count
            cellsG.innerHTML = "";
            const cellPx = SIZE / RANGE;
            const maxN = Math.max(1, ...[...buckets.values()].map(b => b.n));
            buckets.forEach(b => {{
              const [cx, cy] = b.c;
              const x = ORIG + cx * cellPx;
              const y = ORIG + (SHAPE[1] - 1 - cy) * cellPx;
              const alpha = 0.15 + 0.4 * (b.n / maxN);
              cellsG.insertAdjacentHTML("beforeend",
                `<rect x="${{x}}" y="${{y}}" width="${{cellPx}}" height="${{cellPx}}"
                       fill="#0f766e" fill-opacity="${{alpha.toFixed(2)}}"
                       stroke="#115e59" stroke-width="1.5" />`);
              cellsG.insertAdjacentHTML("beforeend",
                `<text x="${{x + cellPx/2}}" y="${{y + cellPx/2 + 4}}" text-anchor="middle"
                       font-size="14" font-weight="700" fill="#134e4a">n=${{b.n}}</text>`);
            }});

            // Points
            pointsG.innerHTML = "";
            points.forEach(([x, y], idx) => {{
              const px = scale(x, vs);
              const py = ORIG + SIZE - scale(y, vs) + ORIG;
              const out = coords[idx] === null;
              pointsG.insertAdjacentHTML("beforeend",
                `<circle cx="${{px.toFixed(1)}}" cy="${{py.toFixed(1)}}" r="5"
                         fill="${{out ? '#9ca3af' : '#b91c1c'}}"
                         stroke="#111827" stroke-width="1" />`);
            }});

            const counts = [...buckets.values()].map(b => b.n).sort((a,b)=>a-b);
            statsOut.textContent =
              `occupied=${{buckets.size}} counts=[${{counts.join(",")}}]`;
          }}

          vsEl.addEventListener("input", update);
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
