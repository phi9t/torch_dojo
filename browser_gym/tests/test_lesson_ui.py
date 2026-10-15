import json
import sys
import types
from pathlib import Path

import numpy as np

from browser_gym.build_catalog import build
from browser_gym.lesson_ui import notebook_tensor_card_source, tensor_card


ROOT = Path(__file__).resolve().parents[2]


def _lesson_source(out: Path, cells: list[dict]) -> str:
    sources = []
    for index, cell in enumerate(cells):
        exercise_id = cell["metadata"]["dojo"]["exercise_id"]
        loader = "".join(cell["source"])
        asset = f"{exercise_id}-{index}.py"
        assert f"/gym-assets/lessons/{asset}" in loader
        assert len(loader.splitlines()) <= 4
        sources.append((out / "lessons" / asset).read_text())
    return "\n".join(sources)


def _html(display_object):
    if hasattr(display_object, "data"):
        return display_object.data
    return display_object._repr_html_()


def test_tensor_card_shows_shape_dtype_axes_and_bounded_preview():
    html = _html(tensor_card("logits", np.arange(24, dtype=np.float64).reshape(4, 6), axes=("batch", "class")))

    assert "Tensor: logits" in html
    assert "shape" in html and "(4, 6)" in html
    assert "dtype" in html and "float64" in html
    assert "axes" in html and "batch x class" in html
    assert "0." in html
    assert "..." in html
    assert "23." not in html
    assert 'tabindex="0"' in html


def test_tensor_card_escapes_labels_and_values():
    html = _html(tensor_card("<logits>", np.array([["<bad>"]], dtype=object), axes=("<batch>",)))

    assert "&lt;logits&gt;" in html
    assert "&lt;batch&gt;" in html
    assert "&lt;bad&gt;" in html
    assert "<bad>" not in html


def test_build_inserts_0059_visual_lesson_without_solution_leakage(tmp_path):
    entries = build(ROOT, tmp_path, ["0059"])
    notebook = json.loads((tmp_path / entries[0]["notebook"]).read_text())

    solution_cells = [
        cell for cell in notebook["cells"]
        if cell["metadata"].get("dojo", {}).get("role") == "solution"
    ]
    exploration_cells = [
        cell for cell in notebook["cells"]
        if cell["metadata"].get("dojo", {}).get("role") == "exploration"
    ]

    assert len(solution_cells) == 1
    assert exploration_cells
    assert all(cell["cell_type"] == "code" for cell in exploration_cells)
    assert all(
        cell["metadata"].get("dojo", {}).get("exercise_id") == "0059"
        for cell in exploration_cells
    )
    assert all(
        cell["metadata"].get("dojo", {}).get("role") != "solution"
        for cell in exploration_cells
    )

    lesson_source = _lesson_source(tmp_path, exploration_cells)
    assert notebook_tensor_card_source().strip() in lesson_source
    assert "Logit scale" in lesson_source
    assert "dojo-loss-plot" in lesson_source
    assert "tensor_card" in lesson_source
    assert "softmax_cross_entropy(" not in lesson_source


def test_build_inserts_box_geometry_visual_lessons_without_solution_leakage(tmp_path):
    entries = build(ROOT, tmp_path, ["0062", "0063", "0064"])
    assert [entry["id"] for entry in entries] == ["0062", "0063", "0064"]
    expected = {
        "0062": ("dojo-0062-iou-shift-control", "dojo-iou-value", "dojo-0062-probe-runs", "pairwise_iou("),
        "0063": ("dojo-0063-yaw-control", "dojo-yaw-value", "dojo-0063-probe-runs", "sample_oriented_box("),
        "0064": ("dojo-0064-point-control", "dojo-membership-value", "dojo-0064-probe-runs", "point_in_rotated_box("),
    }

    for entry in entries:
        notebook = json.loads((tmp_path / entry["notebook"]).read_text())
        exploration_cells = [
            cell
            for cell in notebook["cells"]
            if cell["metadata"].get("dojo", {}).get("role") == "exploration"
        ]
        assert exploration_cells
        lesson_source = _lesson_source(tmp_path, exploration_cells)
        assert notebook_tensor_card_source().strip() in lesson_source
        assert "tensor_card(" in lesson_source
        for token in expected[entry["id"]][:3]:
            assert token in lesson_source
        assert expected[entry["id"]][3] not in lesson_source
        assert "pass" not in lesson_source


def test_build_inserts_0060_training_lesson_without_solution_leakage(tmp_path):
    entries = build(ROOT, tmp_path, ["0060"])
    notebook = json.loads((tmp_path / entries[0]["notebook"]).read_text())

    solution_cells = [
        cell for cell in notebook["cells"]
        if cell["metadata"].get("dojo", {}).get("role") == "solution"
    ]
    exploration_cells = [
        cell for cell in notebook["cells"]
        if cell["metadata"].get("dojo", {}).get("role") == "exploration"
    ]

    assert len(solution_cells) == 1
    assert exploration_cells
    assert all(cell["cell_type"] == "code" for cell in exploration_cells)
    assert all(
        cell["metadata"].get("dojo", {}).get("exercise_id") == "0060"
        for cell in exploration_cells
    )
    assert all(
        cell["metadata"].get("dojo", {}).get("role") != "solution"
        for cell in exploration_cells
    )

    lesson_source = _lesson_source(tmp_path, exploration_cells)
    assert notebook_tensor_card_source().strip() in lesson_source
    assert "Training question: how quickly does one honest classifier step reduce loss?" in lesson_source
    assert 'data-testid="dojo-0060-run-training"' in lesson_source
    assert 'data-testid="dojo-0060-run-count">Training runs: 0' in lesson_source
    assert 'data-testid="dojo-0060-loss-curve"' in lesson_source
    assert 'data-testid="dojo-0060-gradient-probe"' in lesson_source
    assert "classifier_step(" not in lesson_source
    assert "stableStep(" not in lesson_source
    assert "learningRate * dW" not in lesson_source
    assert "Math.exp" not in lesson_source


def test_build_inserts_0065_sensor_lesson_with_coordinate_probe(tmp_path):
    entries = build(ROOT, tmp_path, ["0065"])
    notebook = json.loads((tmp_path / entries[0]["notebook"]).read_text())
    solution_cells = [
        cell for cell in notebook["cells"]
        if cell["metadata"].get("dojo", {}).get("role") == "solution"
    ]
    exploration_cells = [
        cell for cell in notebook["cells"]
        if cell["metadata"].get("dojo", {}).get("role") == "exploration"
    ]

    assert len(solution_cells) == 1
    assert exploration_cells
    assert all(cell["cell_type"] == "code" for cell in exploration_cells)
    assert all(
        cell["metadata"].get("dojo", {}).get("exercise_id") == "0065"
        for cell in exploration_cells
    )
    assert all(
        cell["metadata"].get("dojo", {}).get("role") != "solution"
        for cell in exploration_cells
    )
    lesson_source = _lesson_source(tmp_path, exploration_cells)
    assert notebook_tensor_card_source().strip() in lesson_source
    assert "dojo-0065" in lesson_source
    assert "to_current_ego(" not in lesson_source
    assert "tensor_card" in lesson_source
    assert "Z-axis rotation" in lesson_source
    assert "World translation x" in lesson_source


def test_build_inserts_0066_voxel_lesson_with_contribution_probe(tmp_path):
    entries = build(ROOT, tmp_path, ["0066"])
    notebook = json.loads((tmp_path / entries[0]["notebook"]).read_text())
    solution_cells = [
        cell for cell in notebook["cells"]
        if cell["metadata"].get("dojo", {}).get("role") == "solution"
    ]
    exploration_cells = [
        cell for cell in notebook["cells"]
        if cell["metadata"].get("dojo", {}).get("role") == "exploration"
    ]

    assert len(solution_cells) == 1
    assert exploration_cells
    assert all(cell["cell_type"] == "code" for cell in exploration_cells)
    assert all(
        cell["metadata"].get("dojo", {}).get("exercise_id") == "0066"
        for cell in exploration_cells
    )
    assert all(
        cell["metadata"].get("dojo", {}).get("role") != "solution"
        for cell in exploration_cells
    )

    lesson_source = _lesson_source(tmp_path, exploration_cells)
    assert notebook_tensor_card_source().strip() in lesson_source
    assert "dojo-0066" in lesson_source
    assert "voxel_pool(" not in lesson_source
    assert "tensor_card" in lesson_source
    assert "Voxel size" in lesson_source


def test_build_inserts_0069_calibration_probe_without_solution_leakage(tmp_path):
    entries = build(ROOT, tmp_path, ["0069"])
    notebook = json.loads((tmp_path / entries[0]["notebook"]).read_text())

    solution_cells = [
        cell for cell in notebook["cells"]
        if cell["metadata"].get("dojo", {}).get("role") == "solution"
    ]
    exploration_cells = [
        cell for cell in notebook["cells"]
        if cell["metadata"].get("dojo", {}).get("role") == "exploration"
    ]

    assert len(solution_cells) == 1
    assert exploration_cells
    assert all(cell["cell_type"] == "code" for cell in exploration_cells)
    assert all(
        cell["metadata"].get("dojo", {}).get("exercise_id") == "0069"
        for cell in exploration_cells
    )
    assert all(
        cell["metadata"].get("dojo", {}).get("role") != "solution"
        for cell in exploration_cells
    )

    lesson_source = _lesson_source(tmp_path, exploration_cells)
    assert notebook_tensor_card_source().strip() in lesson_source
    assert "Calibration question: how far can confidence drift from accuracy?" in lesson_source
    assert 'data-testid="dojo-calibration-summary"' in lesson_source
    assert 'data-testid="dojo-calibration-bin-probe"' in lesson_source
    assert 'id="dojo-reliability-plot"' in lesson_source
    assert 'id="dojo-confidence-histogram"' in lesson_source
    assert "tensor_card" in lesson_source
    assert "calibration_metrics(" not in lesson_source


def test_build_inserts_0073_bootstrap_probe_without_solution_leakage(tmp_path):
    entries = build(ROOT, tmp_path, ["0073"])
    notebook = json.loads((tmp_path / entries[0]["notebook"]).read_text())

    solution_cells = [
        cell for cell in notebook["cells"]
        if cell["metadata"].get("dojo", {}).get("role") == "solution"
    ]
    exploration_cells = [
        cell for cell in notebook["cells"]
        if cell["metadata"].get("dojo", {}).get("role") == "exploration"
    ]

    assert len(solution_cells) == 1
    assert exploration_cells
    assert all(cell["cell_type"] == "code" for cell in exploration_cells)
    assert all(
        cell["metadata"].get("dojo", {}).get("exercise_id") == "0073"
        for cell in exploration_cells
    )
    assert all(
        cell["metadata"].get("dojo", {}).get("role") != "solution"
        for cell in exploration_cells
    )

    lesson_source = _lesson_source(tmp_path, exploration_cells)
    assert notebook_tensor_card_source().strip() in lesson_source
    assert "Bootstrap iterations" in lesson_source
    assert "dojo-bootstrap-plot" in lesson_source
    assert "dojo-bootstrap-ci-value" in lesson_source
    assert "tensor_card" in lesson_source
    assert "paired_trip_bootstrap(" not in lesson_source


def test_build_inserts_0061_visual_lesson_with_cluster_and_convergence(tmp_path):
    entries = build(ROOT, tmp_path, ["0061"])
    notebook = json.loads((tmp_path / entries[0]["notebook"]).read_text())

    solution_cells = [
        cell for cell in notebook["cells"]
        if cell["metadata"].get("dojo", {}).get("role") == "solution"
    ]
    exploration_cells = [
        cell for cell in notebook["cells"]
        if cell["metadata"].get("dojo", {}).get("role") == "exploration"
    ]

    assert len(solution_cells) == 1
    assert exploration_cells, "0061 needs exploration/visual lesson cells"
    assert all(cell["cell_type"] == "code" for cell in exploration_cells)
    assert all(
        cell["metadata"].get("dojo", {}).get("exercise_id") == "0061"
        for cell in exploration_cells
    )
    assert all(
        cell["metadata"].get("dojo", {}).get("role") != "solution"
        for cell in exploration_cells
    )

    lesson_source = _lesson_source(tmp_path, exploration_cells)
    # Tensor card with shape evidence
    assert notebook_tensor_card_source().strip() in lesson_source
    assert "tensor_card" in lesson_source
    # Cluster assignments or centroid probe
    assert "centroid" in lesson_source.lower() or "cluster" in lesson_source.lower() or "assign" in lesson_source.lower()
    # Convergence plot (inertia or iteration axis)
    assert "convergence" in lesson_source.lower() or "inertia" in lesson_source.lower() or "iteration" in lesson_source.lower()
    # Explicit controls (Run button or parameter input)
    assert "max_iter" in lesson_source or "k value" in lesson_source.lower() or "cluster" in lesson_source.lower()
    # No reference solution leakage
    assert "def kmeans(" not in lesson_source


def test_0061_lesson_has_tensor_shapes_for_points_centers_labels():
    from browser_gym.lessons import __init__  # noqa: ensure importable
    lesson_path = ROOT / "browser_gym" / "lessons" / "0061.py"
    assert lesson_path.exists(), "browser_gym/lessons/0061.py must exist"
    lesson_source = lesson_path.read_text()
    # Must return cells via a `cells()` callable
    assert "def cells(" in lesson_source
    assert "\"exploration\"" in lesson_source
    assert "dojo-0061" in lesson_source
    assert "kmeans" in lesson_source


def test_0061_exploration_cell_executes_and_renders_probe(tmp_path, monkeypatch):
    entries = build(ROOT, tmp_path, ["0061"])
    notebook = json.loads((tmp_path / entries[0]["notebook"]).read_text())
    exploration = next(
        cell for cell in notebook["cells"]
        if cell["metadata"].get("dojo", {}).get("role") == "exploration"
    )
    displayed = []

    class FakeHTML:
        def __init__(self, data):
            self.data = data

    display_module = types.ModuleType("IPython.display")
    display_module.HTML = FakeHTML
    display_module.display = lambda *objects: displayed.extend(objects)
    display_module.clear_output = lambda *args, **kwargs: None
    ipython_module = types.ModuleType("IPython")
    ipython_module.display = display_module
    monkeypatch.setitem(sys.modules, "IPython", ipython_module)
    monkeypatch.setitem(sys.modules, "IPython.display", display_module)

    source = _lesson_source(tmp_path, [exploration])
    exec(compile(source, "0061_exploration.py", "exec"), {})

    html_outputs = [item.data for item in displayed if hasattr(item, "data")]
    assert any("Tensor: probe_points" in output for output in html_outputs)
    assert any("Tensor: probe_initial_centers" in output for output in html_outputs)
    assert any("Tensor: initial_assignment_labels" in output for output in html_outputs)
    probe_html = next(output for output in html_outputs if "dojo-0061-kmeans" in output)
    assert "dojo-0061-convergence-plot" in probe_html
    assert "const probe = {" in probe_html
    assert "{json.dumps(probe)}" not in probe_html
    assert "((" in probe_html and "=> {" in probe_html
    assert "${cx - 7}" in probe_html
    assert "${{cx - 7}}" not in probe_html


def test_build_inserts_0074_precision_recall_probe_without_solution_leakage(tmp_path):
    entries = build(ROOT, tmp_path, ["0074"])
    notebook = json.loads((tmp_path / entries[0]["notebook"]).read_text())

    solution_cells = [
        cell for cell in notebook["cells"]
        if cell["metadata"].get("dojo", {}).get("role") == "solution"
    ]
    exploration_cells = [
        cell for cell in notebook["cells"]
        if cell["metadata"].get("dojo", {}).get("role") == "exploration"
    ]

    assert len(solution_cells) == 1
    assert exploration_cells
    assert all(cell["cell_type"] == "code" for cell in exploration_cells)
    assert all(
        cell["metadata"].get("dojo", {}).get("exercise_id") == "0074"
        for cell in exploration_cells
    )
    assert all(
        cell["metadata"].get("dojo", {}).get("role") != "solution"
        for cell in exploration_cells
    )

    lesson_source = _lesson_source(tmp_path, exploration_cells)
    assert notebook_tensor_card_source().strip() in lesson_source
    assert "False-positive score" in lesson_source
    assert "dojo-pr-plot" in lesson_source
    assert "dojo-ap-value" in lesson_source
    assert "tensor_card" in lesson_source
    assert "detection_ap(" not in lesson_source
