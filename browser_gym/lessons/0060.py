"""Generated notebook cells for the 0060 classifier-step lesson."""

from __future__ import annotations

import json
import textwrap

import numpy as np

from browser_gym.lesson_ui import notebook_tensor_card_source


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _loss_and_grads(
    X: np.ndarray,
    y: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    logits = X @ W + b
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probabilities = exp / exp.sum(axis=1, keepdims=True)
    rows = np.arange(len(y))
    loss = float(np.mean(np.log(exp.sum(axis=1)) - shifted[rows, y]))
    probabilities[rows, y] -= 1.0
    probabilities /= len(y)
    return loss, X.T @ probabilities, probabilities.sum(axis=0)


def _loss_curves(
    X: np.ndarray,
    y: np.ndarray,
    W0: np.ndarray,
    b0: np.ndarray,
) -> dict[str, list[float]]:
    curves: dict[str, list[float]] = {}
    for rate in np.round(np.arange(0.05, 0.85, 0.05), 2):
        W = W0.copy()
        b = b0.copy()
        losses = []
        for _ in range(20):
            loss, dW, db = _loss_and_grads(X, y, W, b)
            losses.append(round(loss, 6))
            W = W - float(rate) * dW
            b = b - float(rate) * db
        curves[f"{rate:.2f}"] = losses
    return curves


def cells(exercise_id: str) -> list[dict]:
    X_train = np.array(
        [[-2.0, -1.0], [-1.0, -1.5], [-1.5, 0.5], [1.5, -0.5], [1.0, 1.5], [2.0, 1.0]],
        dtype=np.float64,
    )
    y_train = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    W0 = np.zeros((2, 2), dtype=np.float64)
    b0 = np.zeros(2, dtype=np.float64)
    probabilities = _softmax(X_train @ W0 + b0)
    dZ = probabilities.copy()
    dZ[np.arange(len(y_train)), y_train] -= 1.0
    dZ /= len(y_train)
    dW = X_train.T @ dZ
    db = dZ.sum(axis=0)
    gradient_probe = {
        "dW_shape": list(dW.shape),
        "db_shape": list(db.shape),
        "dW_norm": round(float(np.linalg.norm(dW)), 6),
        "db_norm": round(float(np.linalg.norm(db)), 6),
        "preupdate_loss": round(float(np.log(2.0)), 6),
    }
    loss_curves = _loss_curves(X_train, y_train, W0, b0)
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
        X_train = np.array({json.dumps(X_train.tolist())}, dtype=np.float64)
        y_train = np.array({json.dumps(y_train.tolist())}, dtype=np.int64)
        gradient_probe = {json.dumps(gradient_probe, sort_keys=True)}
        loss_curves = {json.dumps(loss_curves, sort_keys=True)}

        display(tensor_card("X_train", X_train, axes=("example", "feature")))
        display(tensor_card("y_train", y_train, axes=("example",)))

        display(HTML("""
        <section id="dojo-0060-training" class="dojo-classifier-training" role="region"
                 aria-labelledby="dojo-0060-training-title">
          <style>
            #dojo-0060-training {{
              border: 1px solid var(--dojo-rule, #d1d5db);
              border-radius: 8px;
              padding: 12px;
              max-width: 700px;
              background: var(--dojo-surface, #ffffff);
              color: var(--dojo-fg, #111827);
            }}
            #dojo-0060-training label {{
              display: grid;
              gap: 4px;
              font-weight: 600;
              margin: 8px 0;
            }}
            #dojo-0060-training input[type="range"] {{
              width: min(100%, 360px);
            }}
            #dojo-0060-training button {{
              margin: 8px 0;
              padding: 6px 10px;
            }}
            #dojo-0060-training output,
            #dojo-0060-training .dojo-training-count,
            #dojo-0060-training .dojo-gradient-probe {{
              display: block;
              margin-top: 6px;
              font-variant-numeric: tabular-nums;
            }}
            #dojo-0060-training svg {{
              display: block;
              width: 100%;
              min-width: 240px;
              max-width: 600px;
              height: 230px;
              margin-top: 10px;
            }}
            #dojo-0060-training svg:focus,
            #dojo-0060-training button:focus,
            #dojo-0060-training input:focus {{
              outline: 3px solid var(--dojo-accent, #2563eb);
              outline-offset: 2px;
            }}
          </style>
          <h3 id="dojo-0060-training-title">Training question: how quickly does one honest classifier step reduce loss?</h3>
          <p>This uses the same stable log-sum-exp idea from 0059, then turns the logits gradient into dW and db.</p>
          <span class="dojo-gradient-probe" data-testid="dojo-0060-gradient-probe">
            Gradient probe: dW shape ({gradient_probe["dW_shape"][0]}, {gradient_probe["dW_shape"][1]}),
            db shape ({gradient_probe["db_shape"][0]}), dW norm {gradient_probe["dW_norm"]:.6f},
            pre-update loss {gradient_probe["preupdate_loss"]:.6f}
          </span>
          <label for="dojo-0060-learning-rate">
            Learning rate
            <input id="dojo-0060-learning-rate" aria-label="Learning rate"
                   type="range" min="0.05" max="0.80" step="0.05" value="0.20" />
          </label>
          <output id="dojo-0060-learning-rate-value" for="dojo-0060-learning-rate"
                  data-testid="dojo-0060-learning-rate-value">Learning rate: 0.20</output>
          <button type="button" id="dojo-0060-run-training" data-testid="dojo-0060-run-training">
            Run bounded training
          </button>
          <span class="dojo-training-count" data-testid="dojo-0060-run-count">Training runs: 0</span>
          <output id="dojo-0060-loss-summary" data-testid="dojo-0060-loss-summary" aria-live="polite">
            No training run yet
          </output>
          <svg id="dojo-0060-loss-curve" data-testid="dojo-0060-loss-curve" role="img" tabindex="0"
               aria-labelledby="dojo-0060-loss-title dojo-0060-loss-desc"
               viewBox="0 0 600 230">
            <title id="dojo-0060-loss-title">Bounded classifier training loss curve</title>
            <desc id="dojo-0060-loss-desc">Twenty full-batch classifier updates on a fixed two-class dataset.</desc>
            <line x1="48" y1="188" x2="560" y2="188" stroke="#374151" stroke-width="1" />
            <line x1="48" y1="24" x2="48" y2="188" stroke="#374151" stroke-width="1" />
            <text x="562" y="192" font-size="12">step</text>
            <text x="8" y="24" font-size="12">loss</text>
            <polyline data-role="loss-line" fill="none" stroke="#0f766e" stroke-width="3" points="" />
          </svg>
        </section>
        <script type="application/javascript">
        (() => {{
          const root = document.getElementById("dojo-0060-training");
          if (!root || root.dataset.ready === "1") return;
          const curves = {json.dumps(loss_curves, sort_keys=True)};
          const rateInput = root.querySelector("#dojo-0060-learning-rate");
          const rateValue = root.querySelector("[data-testid='dojo-0060-learning-rate-value']");
          const runButton = root.querySelector("[data-testid='dojo-0060-run-training']");
          const runCount = root.querySelector("[data-testid='dojo-0060-run-count']");
          const summary = root.querySelector("[data-testid='dojo-0060-loss-summary']");
          const line = root.querySelector("[data-role='loss-line']");
          let runs = 0;

          function project(index, loss, minLoss, maxLoss, length) {{
            const x = 48 + (index / Math.max(length - 1, 1)) * 512;
            const range = Math.max(maxLoss - minLoss, 1e-9);
            const yCoord = 188 - ((loss - minLoss) / range) * 164;
            return [x, yCoord];
          }}

          function renderCurve(losses) {{
            const minLoss = Math.min(...losses);
            const maxLoss = Math.max(...losses);
            const points = losses.map((loss, index) => (
              project(index, loss, minLoss, maxLoss, losses.length).map(value => value.toFixed(1)).join(",")
            ));
            line.setAttribute("points", points.join(" "));
            summary.textContent = `Loss ${{losses[0].toFixed(4)}} -> ${{losses[losses.length - 1].toFixed(4)}} over 20 steps`;
          }}

          function runTraining() {{
            const key = Number.parseFloat(rateInput.value).toFixed(2);
            const losses = curves[key];
            runs += 1;
            runCount.textContent = `Training runs: ${{runs}}`;
            renderCurve(losses);
          }}

          rateInput.addEventListener("input", () => {{
            rateValue.textContent = `Learning rate: ${{Number.parseFloat(rateInput.value).toFixed(2)}}`;
          }});
          runButton.addEventListener("click", runTraining);
          root.dataset.ready = "1";
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
