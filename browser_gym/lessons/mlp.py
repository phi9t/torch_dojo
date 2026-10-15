"""Generated notebook cells for the complete NumPy MLP lab."""

from __future__ import annotations

import json
import textwrap

import numpy as np

from browser_gym.lesson_ui import notebook_tensor_card_source
from browser_gym.labs.mlp import checks, fixtures, reference


def _lesson_data() -> dict[str, object]:
    data = fixtures.load_dataset()
    train_X, train_y = data["train"]
    valid_X, valid_y = data["valid"]
    batch = fixtures.checkpoint_batch()
    params = reference.init_params(
        fixtures.seeded_rng(),
        fixtures.INPUT_DIM,
        fixtures.HIDDEN_DIM,
        fixtures.CLASSES,
    )
    activations = reference.forward(batch.X, params)
    initial_loss, grads = reference.loss_and_grads(batch.X, batch.y, params)
    gradient_norms = {name: round(float(np.linalg.norm(block)), 6) for name, block in grads.items()}
    shape_probe = {
        "train_X": list(train_X.shape),
        "train_y": list(train_y.shape),
        "valid_X": list(valid_X.shape),
        "hidden": list(activations["hidden"].shape),
        "logits": list(activations["logits"].shape),
        "gradient_blocks": {name: list(value.shape) for name, value in grads.items()},
    }

    training_params = {name: value.copy() for name, value in params.items()}
    state: dict[str, object] | None = None
    rng = np.random.default_rng(fixtures.TRAIN_SEED)
    loss_curve = []
    for epoch in range(7):
        train_metrics = reference.evaluate_model(train_X, train_y, training_params)
        valid_metrics = reference.evaluate_model(valid_X, valid_y, training_params)
        loss_curve.append({
            "epoch": epoch,
            "train_loss": round(float(train_metrics["loss"]), 6),
            "valid_loss": round(float(valid_metrics["loss"]), 6),
            "valid_accuracy": round(float(valid_metrics["accuracy"]), 6),
        })
        if epoch < 6:
            training_params, state, _ = reference.train_epoch(
                train_X,
                train_y,
                training_params,
                state,
                rng,
                batch_size=fixtures.TRAIN_BATCH_SIZE,
                learning_rate=fixtures.TRAIN_LEARNING_RATE,
            )

    grid_x = np.linspace(-1.7, 1.7, 25)
    grid_y = np.linspace(-1.7, 1.7, 25)
    mesh = np.array([[x, y] for y in grid_y for x in grid_x], dtype=np.float64)
    probabilities = reference.predict_proba(mesh, training_params, batch_size=64)
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    valid_probabilities = reference.predict_proba(valid_X, training_params, batch_size=16)
    valid_confidence = valid_probabilities.max(axis=1)
    bins = np.linspace(0.0, 1.0, 6)
    hist, _ = np.histogram(valid_confidence, bins=bins)

    return {
        "shape_probe": shape_probe,
        "gradient_norms": gradient_norms,
        "initial_loss": round(float(initial_loss), 6),
        "loss_curve": loss_curve,
        "diagnosis": checks.diagnosis_comparison(),
        "decision_boundary": {
            "grid_x": np.round(grid_x, 4).tolist(),
            "grid_y": np.round(grid_y, 4).tolist(),
            "predictions": predictions.astype(int).reshape(len(grid_y), len(grid_x)).tolist(),
            "confidence": np.round(confidence.reshape(len(grid_y), len(grid_x)), 4).tolist(),
        },
        "confidence": {
            "bins": np.round(bins, 2).tolist(),
            "counts": hist.astype(int).tolist(),
            "mean": round(float(valid_confidence.mean()), 6),
            "min": round(float(valid_confidence.min()), 6),
            "max": round(float(valid_confidence.max()), 6),
        },
    }


def cells(exercise_id: str) -> list[dict]:
    data = fixtures.load_dataset()
    train_X, train_y = data["train"]
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
        train_X = np.array(__TRAIN_X__, dtype=np.float64)
        train_y = np.array(__TRAIN_Y__, dtype=np.int64)
        shape_probe = __SHAPE_PROBE__
        gradient_norms = __GRADIENT_NORMS__
        gradient_summary = ", ".join(f"{name}={value:.6f}" for name, value in gradient_norms.items())

        display(tensor_card("train_X", train_X, axes=("example", "feature")))
        display(tensor_card("train_y", train_y, axes=("example",)))

        display(HTML("""
        <section id="dojo-mlp-experiments" role="region" aria-labelledby="dojo-mlp-experiments-title">
          <style>
            #dojo-mlp-experiments {
              border: 1px solid var(--dojo-rule, #d1d5db);
              border-radius: 8px;
              padding: 12px;
              max-width: 760px;
              background: var(--dojo-surface, #ffffff);
              color: var(--dojo-fg, #111827);
            }
            #dojo-mlp-experiments .dojo-mlp-controls {
              display: flex;
              flex-wrap: wrap;
              gap: 8px;
              margin: 8px 0 12px;
            }
            #dojo-mlp-experiments button {
              border: 1px solid #9ca3af;
              border-radius: 6px;
              background: var(--dojo-bg, #f9fafb);
              color: var(--dojo-fg, #111827);
              cursor: pointer;
              font: inherit;
              padding: 6px 10px;
            }
            #dojo-mlp-experiments button:focus {
              outline: 3px solid var(--dojo-accent, #2563eb);
              outline-offset: 2px;
            }
            #dojo-mlp-experiments svg {
              display: block;
              width: 100%;
              min-width: 260px;
              max-width: 700px;
              height: 240px;
              margin-top: 8px;
            }
            #dojo-mlp-experiments pre {
              overflow: auto;
              white-space: pre-wrap;
              margin: 8px 0 0;
            }
          </style>
          <h3 id="dojo-mlp-experiments-title">MLP fixed-split experiments</h3>
          <div class="dojo-mlp-controls" role="group" aria-label="MLP experiment run actions">
            <button type="button" data-testid="dojo-mlp-loss-run">Run loss</button>
            <button type="button" data-testid="dojo-mlp-boundary-run">Run decision boundary</button>
            <button type="button" data-testid="dojo-mlp-confidence-run">Run confidence</button>
            <button type="button" data-testid="dojo-mlp-gradient-run">Run gradient norms</button>
            <button type="button" data-testid="dojo-mlp-diagnosis-run">Run diagnosis</button>
          </div>
          <output data-testid="dojo-mlp-loss-view" aria-live="polite">Loss view waiting for Run.</output>
          <svg id="dojo-mlp-loss-plot" data-testid="dojo-mlp-loss-plot" role="img"
               aria-label="Training and validation loss curve" viewBox="0 0 700 240"></svg>
          <output data-testid="dojo-mlp-boundary-view" aria-live="polite">Decision boundary view waiting for Run.</output>
          <svg id="dojo-mlp-boundary-plot" data-testid="dojo-mlp-boundary-plot" role="img"
               aria-label="Decision boundary on fixed validation grid" viewBox="0 0 700 240"></svg>
          <output data-testid="dojo-mlp-confidence-view" aria-live="polite">Confidence view waiting for Run.</output>
          <svg id="dojo-mlp-confidence-plot" data-testid="dojo-mlp-confidence-plot" role="img"
               aria-label="Validation confidence histogram" viewBox="0 0 700 240"></svg>
          <pre data-testid="dojo-mlp-gradient-view">Gradient block norms waiting for Run.</pre>
          <output data-testid="dojo-mlp-diagnosis-view" aria-live="polite">Diagnosis view waiting for Run. Planted defect: omitted_batch_mean.</output>
          <svg id="dojo-mlp-diagnosis-plot" data-testid="dojo-mlp-diagnosis-plot" role="img"
               aria-label="Broken and corrected validation loss comparison" viewBox="0 0 700 240"></svg>
          <pre aria-label="MLP shape probe">__SHAPE_JSON__</pre>
          <script type="application/json" id="dojo-mlp-experiment-data">__LESSON_DATA__</script>
        </section>
        <script type="application/javascript">
        (() => {
          const root = document.getElementById("dojo-mlp-experiments");
          if (!root || root.dataset.ready === "1") return;
          const data = JSON.parse(root.querySelector("#dojo-mlp-experiment-data").textContent);
          const colors = ["#2563eb", "#dc2626", "#059669"];

          function clear(svg) {
            while (svg.firstChild) svg.removeChild(svg.firstChild);
          }

          function node(name, attrs) {
            const el = document.createElementNS("http://www.w3.org/2000/svg", name);
            for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, String(value));
            return el;
          }

          function text(svg, value, x, y) {
            const label = node("text", { x, y, "font-size": 12, fill: "#111827" });
            label.textContent = value;
            svg.append(label);
          }

          function drawLoss() {
            const svg = root.querySelector("#dojo-mlp-loss-plot");
            clear(svg);
            const rows = data.loss_curve;
            const losses = rows.flatMap(row => [row.train_loss, row.valid_loss]);
            const minLoss = Math.min(...losses);
            const maxLoss = Math.max(...losses);
            const x = epoch => 48 + (epoch / (rows.length - 1)) * 600;
            const y = loss => 198 - ((loss - minLoss) / Math.max(maxLoss - minLoss, 1e-9)) * 156;
            svg.append(node("line", { x1: 44, y1: 198, x2: 668, y2: 198, stroke: "#374151" }));
            svg.append(node("line", { x1: 44, y1: 28, x2: 44, y2: 198, stroke: "#374151" }));
            for (const key of ["train_loss", "valid_loss"]) {
              const points = rows.map(row => `${x(row.epoch).toFixed(1)},${y(row[key]).toFixed(1)}`).join(" ");
              svg.append(node("polyline", {
                points,
                fill: "none",
                stroke: key === "train_loss" ? "#2563eb" : "#dc2626",
                "stroke-width": 3
              }));
            }
            text(svg, "train", 548, 42);
            text(svg, "valid", 548, 60);
            root.querySelector("[data-testid='dojo-mlp-loss-view']").textContent =
              `Loss view: epoch 0 train ${rows[0].train_loss.toFixed(4)}, epoch ${rows.at(-1).epoch} train ${rows.at(-1).train_loss.toFixed(4)}, valid accuracy ${rows.at(-1).valid_accuracy.toFixed(3)}.`;
          }

          function drawBoundary() {
            const svg = root.querySelector("#dojo-mlp-boundary-plot");
            clear(svg);
            const boundary = data.decision_boundary;
            const rows = boundary.predictions.length;
            const cols = boundary.predictions[0].length;
            const width = 600 / cols;
            const height = 170 / rows;
            for (let r = 0; r < rows; r++) {
              for (let c = 0; c < cols; c++) {
                svg.append(node("rect", {
                  x: 48 + c * width,
                  y: 28 + r * height,
                  width: width + 0.2,
                  height: height + 0.2,
                  fill: colors[boundary.predictions[r][c]],
                  opacity: 0.2 + 0.6 * boundary.confidence[r][c]
                }));
              }
            }
            svg.append(node("rect", { x: 48, y: 28, width: 600, height: 170, fill: "none", stroke: "#374151" }));
            text(svg, "x0", 652, 204);
            text(svg, "x1", 22, 28);
            root.querySelector("[data-testid='dojo-mlp-boundary-view']").textContent =
              `Decision boundary view: ${rows * cols} fixed grid points rendered from the trained reference parameters.`;
          }

          function drawConfidence() {
            const svg = root.querySelector("#dojo-mlp-confidence-plot");
            clear(svg);
            const counts = data.confidence.counts;
            const maxCount = Math.max(...counts, 1);
            svg.append(node("line", { x1: 44, y1: 198, x2: 668, y2: 198, stroke: "#374151" }));
            svg.append(node("line", { x1: 44, y1: 28, x2: 44, y2: 198, stroke: "#374151" }));
            counts.forEach((count, index) => {
              const x = 64 + index * 112;
              const h = (count / maxCount) * 150;
              svg.append(node("rect", { x, y: 198 - h, width: 74, height: h, fill: "#059669" }));
              text(svg, `${data.confidence.bins[index].toFixed(1)}-${data.confidence.bins[index + 1].toFixed(1)}`, x - 4, 218);
            });
            root.querySelector("[data-testid='dojo-mlp-confidence-view']").textContent =
              `Confidence view: validation confidence mean ${data.confidence.mean.toFixed(3)}, min ${data.confidence.min.toFixed(3)}, max ${data.confidence.max.toFixed(3)}.`;
          }

          function drawGradients() {
            root.querySelector("[data-testid='dojo-mlp-gradient-view']").textContent =
              `Gradient block norms: ${Object.entries(data.gradient_norms).map(([name, value]) => `${name}=${Number(value).toFixed(6)}`).join(", ")}\\n` +
              `Initial checkpoint loss: ${data.initial_loss.toFixed(6)}\\n` +
              JSON.stringify(data.shape_probe, null, 2);
          }

          function drawDiagnosis() {
            const svg = root.querySelector("#dojo-mlp-diagnosis-plot");
            clear(svg);
            const symptom = data.diagnosis.visible_symptom;
            const broken = symptom.broken_loss_curve;
            const corrected = symptom.corrected_loss_curve;
            const allLosses = broken.concat(corrected).flatMap(row => [row.valid_loss]);
            const minLoss = Math.min(...allLosses);
            const maxLoss = Math.max(...allLosses);
            const x = epoch => 48 + (epoch / (broken.length - 1)) * 600;
            const y = loss => 198 - ((loss - minLoss) / Math.max(maxLoss - minLoss, 1e-9)) * 156;
            svg.append(node("line", { x1: 44, y1: 198, x2: 668, y2: 198, stroke: "#374151" }));
            svg.append(node("line", { x1: 44, y1: 28, x2: 44, y2: 198, stroke: "#374151" }));
            for (const [rows, stroke] of [[broken, "#dc2626"], [corrected, "#2563eb"]]) {
              const points = rows.map(row => `${x(row.epoch).toFixed(1)},${y(row.valid_loss).toFixed(1)}`).join(" ");
              svg.append(node("polyline", { points, fill: "none", stroke, "stroke-width": 3 }));
            }
            text(svg, "broken", 538, 42);
            text(svg, "corrected", 538, 60);
            root.querySelector("[data-testid='dojo-mlp-diagnosis-view']").textContent =
              `Planted defect: ${data.diagnosis.defect_id}. ` +
              `Broken max abs_error ${Number(data.diagnosis.broken_max_abs_error).toFixed(6)}; ` +
              `corrected max abs_error ${Number(data.diagnosis.corrected_max_abs_error).toFixed(6)}. ` +
              `broken final valid loss ${Number(symptom.broken_final_valid_loss).toFixed(4)}; ` +
              `corrected final valid loss ${Number(symptom.corrected_final_valid_loss).toFixed(4)}. ` +
              data.diagnosis.repair_task;
          }

          root.querySelector("[data-testid='dojo-mlp-loss-run']").addEventListener("click", drawLoss);
          root.querySelector("[data-testid='dojo-mlp-boundary-run']").addEventListener("click", drawBoundary);
          root.querySelector("[data-testid='dojo-mlp-confidence-run']").addEventListener("click", drawConfidence);
          root.querySelector("[data-testid='dojo-mlp-gradient-run']").addEventListener("click", drawGradients);
          root.querySelector("[data-testid='dojo-mlp-diagnosis-run']").addEventListener("click", drawDiagnosis);
          root.dataset.ready = "1";
        })();
        </script>
        """))
        '''
        )
        .replace("__TRAIN_X__", json.dumps(train_X.tolist()))
        .replace("__TRAIN_Y__", json.dumps(train_y.tolist()))
        .replace("__SHAPE_PROBE__", json.dumps(lesson_data["shape_probe"], sort_keys=True))
        .replace("__GRADIENT_NORMS__", json.dumps(lesson_data["gradient_norms"], sort_keys=True))
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
