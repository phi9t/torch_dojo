"""Generated notebook cells for the 0065 temporal ego-coordinates lesson."""

from __future__ import annotations

import json
import textwrap

import numpy as np

from browser_gym.lesson_ui import notebook_tensor_card_source


def _rotation_z(angle_rad: float, dtype=np.float64) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.eye(4, dtype=dtype)
    R[:2, :2] = [[c, -s], [s, c]]
    return R


def _translation(x: float, y: float, z: float, dtype=np.float64) -> np.ndarray:
    T = np.eye(4, dtype=dtype)
    T[0, 3] = x
    T[1, 3] = y
    T[2, 3] = z
    return T


def _reference_transform(points_world, world_from_current):
    R = world_from_current[:3, :3]
    t = world_from_current[:3, 3]
    centered = points_world - t
    return (R.T @ centered.T).T


def cells(exercise_id: str) -> list[dict]:
    # Worked probe from the question script: point (1,0,0) in old ego frame,
    # rotated 90deg about z and translated +5 on x, current ego at +2 on x.
    rot90 = np.eye(4, dtype=np.float64)
    rot90[:2, :2] = [[0.0, -1.0], [1.0, 0.0]]
    rot90[0, 3] = 5.0
    current = _translation(2.0, 0.0, 0.0)
    point_old = np.array([[[[1.0, 0.0, 0.0]]]])  # [B=1,A=1,T=1,3]
    poses = rot90.reshape(1, 1, 4, 4)
    point_world = (poses[0, 0, :3, :3] @ point_old[0, 0, 0].T).T + poses[0, 0, :3, 3]
    point_current = _reference_transform(point_world, current)

    probe = {
        "point_old": point_old[0, 0, 0].tolist(),
        "point_world": point_world.tolist(),
        "point_current_expected": point_current.tolist(),
        "angle_deg_initial": 90.0,
        "translate_x_initial": 5.0,
        "current_x_initial": 2.0,
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

        def _rot_z(angle_rad):
            c, s = np.cos(angle_rad), np.sin(angle_rad)
            R = np.eye(4)
            R[0, 0] = c; R[0, 1] = -s
            R[1, 0] = s; R[1, 1] = c
            return R

        def _trans(x, y, z):
            T = np.eye(4)
            T[0, 3] = x; T[1, 3] = y; T[2, 3] = z
            return T

        def _frame_chain(point_old, angle_deg, world_tx, current_tx):
            "Apply T_world_from_t then inverse(T_world_from_current); reference-only probe."
            poses = _rot_z(np.deg2rad(angle_deg))
            poses[:3, 3] = [world_tx, 0., 0.]
            current = _trans(current_tx, 0., 0.)
            p_world = (poses[:3, :3] @ point_old) + poses[:3, 3]
            p_current = (current[:3, :3].T @ (p_world - current[:3, 3]))
            return p_world, p_current

        p_old = np.array(probe["point_old"])
        traj = p_old.reshape(1, 1, 1, 3)
        display(tensor_card("trajectories", traj, axes=("B", "A", "T", "xyz")))

        # Canonical worked example coordinates
        p_world_expected = np.array(probe["point_world"])
        p_current_expected = np.array(probe["point_current_expected"])
        display(tensor_card("expected world point", p_world_expected.reshape(1, 1, 1, 3),
                            axes=("B", "A", "T", "xyz")))
        display(tensor_card("expected current-ego point", p_current_expected.reshape(1, 1, 1, 3),
                            axes=("B", "A", "T", "xyz")))

        display(HTML("""
        <section id="dojo-0065-coordinate-probe" class="dojo-coordinate-probe" role="region"
                 aria-labelledby="dojo-0065-coordinate-title">
          <style>
            #dojo-0065-coordinate-probe {{
              border: 1px solid var(--dojo-rule, #d1d5db);
              border-radius: 8px;
              padding: 12px;
              max-width: 680px;
              background: var(--dojo-surface, #ffffff);
              color: var(--dojo-fg, #111827);
            }}
            #dojo-0065-coordinate-probe label {{
              display: grid;
              gap: 4px;
              font-weight: 600;
              margin: 8px 0;
            }}
            #dojo-0065-coordinate-probe input[type="range"] {{
              width: min(100%, 360px);
            }}
            #dojo-0065-coordinate-probe output {{
              display: block;
              margin-top: 4px;
              font-variant-numeric: tabular-nums;
            }}
            #dojo-0065-coordinate-probe svg {{
              display: block;
              width: 100%;
              max-width: 560px;
              height: 240px;
              margin-top: 10px;
            }}
            #dojo-0065-coordinate-probe svg:focus,
            #dojo-0065-coordinate-probe input:focus {{
              outline: 3px solid var(--dojo-accent, #2563eb);
              outline-offset: 2px;
            }}
            #dojo-0065-coordinate-probe .coord-text {{
              font-family: monospace;
            }}
          </style>
          <h3 id="dojo-0065-coordinate-title">Frame chain: old ego -> world -> current ego</h3>
          <p>Drag the rotation and translation controls to see how the world and current-ego
             coordinates change. Probe only — this does not launch a check.</p>
          <label for="dojo-0065-angle">
            Z-axis rotation (degrees)
            <input id="dojo-0065-angle" aria-label="Rotation angle degrees"
                   type="range" min="-180" max="180" step="5" value="90" />
          </label>
          <output id="dojo-0065-angle-out" for="dojo-0065-angle"
                  data-testid="dojo-0065-angle-out">θ = 90°</output>
          <label for="dojo-0065-world-tx">
            World translation x (old ego origin)
            <input id="dojo-0065-world-tx" aria-label="World translation x"
                   type="range" min="-5" max="10" step="0.5" value="5" />
          </label>
          <output id="dojo-0065-world-tx-out" for="dojo-0065-world-tx"
                  data-testid="dojo-0065-world-tx-out">t_world_x = 5.00</output>
          <label for="dojo-0065-current-tx">
            Current ego x
            <input id="dojo-0065-current-tx" aria-label="Current ego x"
                   type="range" min="-5" max="10" step="0.5" value="2" />
          </label>
          <output id="dojo-0065-current-tx-out" for="dojo-0065-current-tx"
                  data-testid="dojo-0065-current-tx-out">t_current_x = 2.00</output>
          <output id="dojo-0065-coords" class="coord-text" data-testid="dojo-0065-coords"
                  aria-live="polite">
            world = (5.00, 1.00, 0.00) | current = (3.00, 1.00, 0.00)
          </output>
          <svg id="dojo-0065-frame-plot" role="img" tabindex="0"
               aria-labelledby="dojo-0065-plot-title dojo-0065-plot-desc"
               viewBox="0 0 560 240">
            <title id="dojo-0065-plot-title">Transformed point in world and current ego</title>
            <desc id="dojo-0065-plot-desc">Top-down view of world frame, current ego frame, and the transformed point.</desc>
            <line x1="40" y1="200" x2="520" y2="200" stroke="#374151" stroke-width="1" />
            <line x1="280" y1="20" x2="280" y2="220" stroke="#374151" stroke-width="1" />
            <text x="525" y="204" font-size="11">x</text>
            <text x="284" y="18" font-size="11">y</text>
            <g data-role="current-frame" stroke="#0f766e" stroke-width="1.5">
              <line x1="280" y1="200" x2="280" y2="200" data-role="cx-x" />
              <line x1="280" y1="200" x2="280" y2="200" data-role="cx-z" />
            </g>
            <circle data-role="world-point" r="5" fill="#b91c1c" cx="280" cy="200" />
            <circle data-role="current-point" r="5" fill="#2563eb" cx="280" cy="200" />
          </svg>
        </section>
        <script type="application/javascript">
        (() => {{
          const root = document.getElementById("dojo-0065-coordinate-probe");
          if (!root || root.dataset.ready === "1") return;
          const angleEl = root.querySelector("#dojo-0065-angle");
          const worldTxEl = root.querySelector("#dojo-0065-world-tx");
          const currentTxEl = root.querySelector("#dojo-0065-current-tx");
          const angleOut = root.querySelector("#dojo-0065-angle-out");
          const worldOut = root.querySelector("#dojo-0065-world-tx-out");
          const currentOut = root.querySelector("#dojo-0065-current-tx-out");
          const coordsOut = root.querySelector("#dojo-0065-coords");
          const worldPt = root.querySelector("[data-role='world-point']");
          const currentPt = root.querySelector("[data-role='current-point']");
          const cxX = root.querySelector("[data-role='cx-x']");
          const cxZ = root.querySelector("[data-role='cx-z']");

          const PX = 40; // pixels per world unit
          const OX = 280, OY = 200;

          function project(x, y) {{
            return [OX + x * PX, OY - y * PX];
          }}

          function update() {{
            const deg = Number.parseFloat(angleEl.value);
            const wtx = Number.parseFloat(worldTxEl.value);
            const ctx = Number.parseFloat(currentTxEl.value);
            const rad = deg * Math.PI / 180;
            // Old point is (1,0,0) in old ego; rotate about z-axis then translate.
            const wx = Math.cos(rad) * 1 + wtx;
            const wy = Math.sin(rad) * 1;
            // Current frame shares orientation with world (zero rotation); translate by (-ctx).
            const cx = wx - ctx;
            const cy = wy;

            angleOut.textContent = `θ = ${{deg.toFixed(0)}}°`;
            worldOut.textContent = `t_world_x = ${{wtx.toFixed(2)}}`;
            currentOut.textContent = `t_current_x = ${{ctx.toFixed(2)}}`;
            coordsOut.textContent =
              `world = (${{wx.toFixed(2)}}, ${{wy.toFixed(2)}}, 0.00) | ` +
              `current = (${{cx.toFixed(2)}}, ${{cy.toFixed(2)}}, 0.00)`;

            const [wpx, wpy] = project(wx, wy);
            worldPt.setAttribute("cx", wpx.toFixed(1));
            worldPt.setAttribute("cy", wpy.toFixed(1));
            const [cpx, cpy] = project(ctx + cx, cy);
            currentPt.setAttribute("cx", cpx.toFixed(1));
            currentPt.setAttribute("cy", cpy.toFixed(1));
            // Current frame axes are identity (visualized at current ego position)
            const [axx, axy] = project(ctx + 1, 0);
            const [ayx, ayy] = project(ctx, 1);
            const [aox, aoy] = project(ctx, 0);
            cxX.setAttribute("x2", axx.toFixed(1)); cxX.setAttribute("y2", axy.toFixed(1));
            cxX.setAttribute("x1", aox.toFixed(1)); cxX.setAttribute("y1", aoy.toFixed(1));
            cxZ.setAttribute("x2", ayx.toFixed(1)); cxZ.setAttribute("y2", ayy.toFixed(1));
            cxZ.setAttribute("x1", aox.toFixed(1)); cxZ.setAttribute("y1", aoy.toFixed(1));
          }}

          [angleEl, worldTxEl, currentTxEl].forEach(el =>
            el.addEventListener("input", update));
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
