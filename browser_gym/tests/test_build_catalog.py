import json
from pathlib import Path

import pytest

from browser_gym.build_catalog import DEFAULT_CATALOG_IDS, build, extract_cell


ROOT = Path(__file__).resolve().parents[2]


def test_build_0059_from_canonical_pair(tmp_path):
    entries = build(ROOT, tmp_path, ["0059"])
    assert len(entries) == 1
    notebook = json.loads((tmp_path / entries[0]["notebook"]).read_text())
    assert notebook["metadata"]["dojo"]["exercise_id"] == "0059"
    brief = "".join(notebook["cells"][0]["source"])
    # 0059's brief is written with markers: the goal comes first, then each stage's contract.
    assert brief.index('<div class="dojo-brief-goal"><h4>Goal</h4>') < brief.index('dojo-stage-baseline')
    assert '<section class="dojo-brief-stage dojo-stage-baseline"><h3>Baseline</h3>' in brief
    assert '<section class="dojo-brief-stage dojo-stage-followup"><h3>Follow-up</h3>' in brief
    assert '<section class="dojo-brief-stage dojo-stage-production"><h3>Production</h3>' in brief
    assert '<div class="dojo-brief-watch-out"><h4>Watch out</h4>' in brief
    assert '<details class="dojo-brief-why"><summary>Why</summary>' in brief
    assert r'<span class="dojo-math">\(q = z - \max_c z_c\)</span>' in brief
    assert "Average over VALID rows" in brief
    kinds = {(section["kind"], section.get("stage")) for section in entries[0]["brief"]}
    assert {("goal", None), ("predict", "baseline"), ("watch-out", "followup")} <= kinds
    solutions = [cell for cell in notebook["cells"] if cell["metadata"].get("dojo", {}).get("role") == "solution"]
    assert len(solutions) == 1
    assert "".join(solutions[0]["source"]).endswith(
        extract_cell((ROOT / "dojo/0059_stable_cross_entropy.py").read_text(), "answer").strip() + "\n"
    )
    assert "import numpy as np" in "".join(solutions[0]["source"])
    assert all(cell["metadata"].get("dojo", {}).get("role") != "solution" for cell in notebook["cells"] if cell is not solutions[0])
    assert len(json.loads((tmp_path / entries[0]["checker"]).read_text())["verifiers"]) >= 5


def test_catalog_keeps_0059_assistance_and_mlp_checkpoints(tmp_path):
    entries = build(ROOT, tmp_path, ["0059", "mlp"])
    assert [entry["id"] for entry in entries] == ["0059", "mlp"]
    assert len(entries[0]["hints"]) == 3
    assert "def softmax_cross_entropy" in entries[0]["solution"]
    assert entries[0]["stages"] == ["baseline", "followup", "production"]
    assert entries[1]["title"] == "MLP Modeling Lab"
    assert entries[1]["notebook"] == "notebooks/mlp_modeling_lab.ipynb"
    assert entries[1]["checker"] == "checkers/mlp_modeling_lab.json"
    assert entries[1]["stages"] == ["init", "forward", "backward", "train", "inference", "evaluate", "state", "diagnose"]
    # Dojo stages are cumulative (ADR 0002); lab checkpoints are checked independently.
    assert entries[0]["stage_order"] == "cumulative"
    assert entries[1]["stage_order"] == "independent"
    assert entries[1]["reference_components"] == [
        "init_params",
        "forward",
        "loss_and_grads",
        "train_epoch",
        "predict_proba",
        "evaluate_model",
        "save_state",
        "restore_state",
        "diagnose_mlp_defect",
    ]
    bundle = json.loads((tmp_path / entries[1]["checker"]).read_text())
    assert bundle["kind"] == "mlp_lab"
    assert bundle["checkpoints"] == entries[1]["stages"]
    assert bundle["candidate_functions"] == entries[1]["reference_components"]


def test_builds_grouped_evaluation_exercises_from_canonical_pairs(tmp_path):
    entries = build(ROOT, tmp_path, ["0073", "0074"])

    assert [entry["id"] for entry in entries] == ["0073", "0074"]
    assert entries[0]["title"] == "Paired trip bootstrap"
    assert entries[1]["title"] == "Detection average precision"
    assert all(entry["stages"] == ["baseline", "followup", "production"] for entry in entries)
    assert all(len(entry["hints"]) >= 3 for entry in entries)
    assert all(entry["allowed_helpers"] == ["numpy"] for entry in entries)

    for entry in entries:
        notebook = json.loads((tmp_path / entry["notebook"]).read_text())
        bundle = json.loads((tmp_path / entry["checker"]).read_text())
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
            cell["metadata"].get("dojo", {}).get("exercise_id") == entry["id"]
            for cell in exploration_cells
        )
        assert all(
            cell["metadata"].get("dojo", {}).get("role") != "solution"
            for cell in exploration_cells
        )
        assert len(bundle["verifiers"]) >= 6
        assert bundle["exercise_id"] == entry["id"]
        assert entry["solution"].startswith("import numpy as np")
        assert "hypothesis" not in entry["solution"]


def test_default_catalog_includes_qualified_exercises(tmp_path):
    entries = build(ROOT, tmp_path, DEFAULT_CATALOG_IDS)
    assert [entry["id"] for entry in entries] == [
        "0059",
        "0060",
        "0061",
        "0062",
        "0063",
        "0064",
        "0065",
        "0066",
        "mlp",
        "0069",
        "0070",
        "0073",
        "0074",
        "0068",
        "0067",
        "0071",
        "0076",
        "0077",
        "0035",
        "0072",
        "0078",
        "0079",
        "0082",
        "0083",
        "0091",
        "0030",
        "0050",
        "0075",
        "0100",
        "0094",
        "0080",
    ]


def test_every_browser_workout_has_a_topic_for_the_home_page(tmp_path):
    entries = build(ROOT, tmp_path, DEFAULT_CATALOG_IDS)
    # The home page groups workouts by their first concept.
    assert [entry["id"] for entry in entries if not entry.get("concepts")] == []


def test_build_0060_from_canonical_pair_with_independent_assistance(tmp_path):
    entries = build(ROOT, tmp_path, ["0060"])
    assert len(entries) == 1
    entry = entries[0]
    assert entry["id"] == "0060"
    assert entry["title"] == "One honest classifier update"
    assert entry["stages"] == ["baseline", "followup", "production"]
    assert entry["hints"] == [
        "Reuse the stable 0059 log-sum-exp pattern before taking probabilities.",
        "Compute dW and db from one pre-update snapshot, then return new arrays.",
        "For ignored rows, divide by the valid-example count so uneven microbatches still add up."
    ]
    assert "def classifier_step" in entry["solution"]

    notebook = json.loads((tmp_path / entry["notebook"]).read_text())
    assert notebook["metadata"]["dojo"]["exercise_id"] == "0060"
    first_markdown = "".join(notebook["cells"][0]["source"])
    assert "# 0060 · One honest classifier update" in first_markdown
    assert "Use the stable <code>logits</code> loss from 0059" in first_markdown

    solutions = [cell for cell in notebook["cells"] if cell["metadata"].get("dojo", {}).get("role") == "solution"]
    assert len(solutions) == 1
    assert "".join(solutions[0]["source"]).endswith(
        extract_cell((ROOT / "dojo/0060_classifier_step.py").read_text(), "answer").strip() + "\n"
    )
    assert "import numpy as np" in "".join(solutions[0]["source"])
    assert all(cell["metadata"].get("dojo", {}).get("role") != "solution" for cell in notebook["cells"] if cell is not solutions[0])

    checker = json.loads((tmp_path / entry["checker"]).read_text())
    assert checker["candidate"] == "classifier_step"
    assert {item["name"] for item in checker["verifiers"]} >= {
        "verify_baseline_one_step_and_preupdate_loss",
        "verify_followup_ignore_and_all_ignored",
        "verify_production_weight_unequal_microbatches",
    }


def test_build_0069_from_canonical_pair_with_calibration_path(tmp_path):
    entries = build(ROOT, tmp_path, ["0059", "0060", "mlp", "0069"])
    entry = next(item for item in entries if item["id"] == "0069")
    assert entry["title"] == "Multiclass calibration metrics"
    assert entry["stages"] == ["baseline", "followup", "production"]
    assert entry["allowed_helpers"] == ["numpy"]
    assert entry["previous"]["id"] == "mlp"
    assert [item["id"] for item in entry["learning_path"]] == ["0059", "0060", "mlp", "0069"]
    assert "confidence bins" in " ".join(entry["hints"])
    assert "def calibration_metrics" in entry["solution"]

    notebook = json.loads((tmp_path / entry["notebook"]).read_text())
    assert notebook["metadata"]["dojo"]["exercise_id"] == "0069"
    first_markdown = "".join(notebook["cells"][0]["source"])
    assert "# 0069 · Multiclass calibration metrics" in first_markdown
    assert "Rows sum to one" in first_markdown

    solutions = [
        cell for cell in notebook["cells"]
        if cell["metadata"].get("dojo", {}).get("role") == "solution"
    ]
    assert len(solutions) == 1
    assert "".join(solutions[0]["source"]).endswith(
        extract_cell((ROOT / "dojo/0069_calibration_metrics.py").read_text(), "answer").strip() + "\n"
    )
    assert "import numpy as np" in "".join(solutions[0]["source"])
    assert all(
        cell["metadata"].get("dojo", {}).get("role") != "solution"
        for cell in notebook["cells"]
        if cell is not solutions[0]
    )

    checker = json.loads((tmp_path / entry["checker"]).read_text())
    assert checker["candidate"] == "calibration_metrics"
    assert {item["name"] for item in checker["verifiers"]} >= {
        "verify_baseline_known_denominators",
        "verify_followup_bin_edges_and_empty_bins",
        "verify_production_validation_and_immutability",
    }


def test_build_0070_from_canonical_pair_with_anchor_assistance(tmp_path):
    entries = build(ROOT, tmp_path, ["0070"])
    assert len(entries) == 1
    entry = entries[0]
    assert entry["id"] == "0070"
    assert entry["title"] == "Anchor to GT Matching"
    assert entry["stages"] == ["baseline", "followup", "production"]
    assert len(entry["hints"]) == 3
    assert "def match_anchors" in entry["solution"]

    notebook = json.loads((tmp_path / entry["notebook"]).read_text())
    assert notebook["metadata"]["dojo"]["exercise_id"] == "0070"
    solution_cells = [
        cell for cell in notebook["cells"]
        if cell["metadata"].get("dojo", {}).get("role") == "solution"
    ]
    exploration_cells = [
        cell for cell in notebook["cells"]
        if cell["metadata"].get("dojo", {}).get("role") == "exploration"
    ]
    assert len(solution_cells) == 1
    assert "".join(solution_cells[0]["source"]).endswith(
        extract_cell((ROOT / "dojo/0070_anchor_match.py").read_text(), "answer").strip() + "\n"
    )
    assert "import numpy as np" in "".join(solution_cells[0]["source"])
    assert exploration_cells
    lesson_source = (tmp_path / "lessons" / "0070-0.py").read_text()
    assert "Anchor matching threshold" in lesson_source
    assert "dojo-anchor-matching-plot" in lesson_source
    assert "collision_iou" in lesson_source
    assert "match_anchors(" not in lesson_source
    assert all(
        cell["metadata"].get("dojo", {}).get("exercise_id") == "0070"
        for cell in exploration_cells
    )
    assert all(
        cell["metadata"].get("dojo", {}).get("role") != "solution"
        for cell in exploration_cells
    )


def test_reject_missing_or_duplicate_markers(tmp_path):
    question = ROOT / "dojo/0059_stable_cross_entropy.py"
    with pytest.raises(ValueError, match="missing"):
        extract_cell(question.read_text().replace("# %% [answer]", "# %%"), "answer")
    with pytest.raises(ValueError, match="duplicate"):
        extract_cell(question.read_text() + "\n# %% [answer]\npass\n", "answer")


def test_build_does_not_execute_reference(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "root"
    dojo = root / "dojo"
    dojo.mkdir(parents=True)
    (dojo / "0059_sample.py").write_text("# %% [answer]\ndef answer():\n    pass\n")
    (dojo / "0059_sample_eval.py").write_text(
        "from pathlib import Path\nPath('should-not-exist').write_text('bad')\n"
        "# %% [verifiers]\ndef verify_answer(answer):\n    assert answer() == 1\n"
    )
    build(root, tmp_path / "out", ["0059"])
    assert not (tmp_path / "should-not-exist").exists()


def test_build_rejects_eval_without_verifier_marker(tmp_path):
    root = tmp_path / "root"
    dojo = root / "dojo"
    dojo.mkdir(parents=True)
    (dojo / "0059_sample.py").write_text("# %% [answer]\ndef answer():\n    pass\n")
    (dojo / "0059_sample_eval.py").write_text("def verify_answer(answer):\n    pass\n")
    with pytest.raises(ValueError, match=r"0059_sample_eval.py:1: missing.*verifiers"):
        build(root, tmp_path / "out", ["0059"])


def test_reject_unqualified_exercise(tmp_path):
    with pytest.raises(ValueError, match="not yet qualified"):
        build(ROOT, tmp_path, ["9999"])


def test_build_0061_from_canonical_pair(tmp_path):
    entries = build(ROOT, tmp_path, ["0061"])
    assert len(entries) == 1
    assert entries[0]["id"] == "0061"
    notebook = json.loads((tmp_path / entries[0]["notebook"]).read_text())
    assert notebook["metadata"]["dojo"]["exercise_id"] == "0061"
    solutions = [cell for cell in notebook["cells"] if cell["metadata"].get("dojo", {}).get("role") == "solution"]
    assert len(solutions) == 1
    answer = extract_cell((ROOT / "dojo/0061_kmeans.py").read_text(), "answer")
    assert "".join(solutions[0]["source"]).endswith(answer.strip() + "\n")
    assert "import numpy as np" in "".join(solutions[0]["source"])
    assert "def kmeans(" in "".join(solutions[0]["source"])
    assert all(cell["metadata"].get("dojo", {}).get("role") != "solution"
               for cell in notebook["cells"] if cell is not solutions[0])
    checker = json.loads((tmp_path / entries[0]["checker"]).read_text())
    assert len(checker["verifiers"]) >= 6
    assert checker["candidate"] == "kmeans"


def test_catalog_0061_title_hints_and_stages(tmp_path):
    entries = build(ROOT, tmp_path, ["0061"])
    entry = entries[0]
    assert "K-Means" in entry["title"]
    assert entry["stages"] == ["baseline", "followup", "production"]
    assert len(entry["hints"]) >= 3
    assert "def kmeans(" in entry["solution"]
    assert "centers" in entry["solution"] or "labels" in entry["solution"]
    notebook = json.loads((tmp_path / entry["notebook"]).read_text())
    h1 = next(cell for cell in notebook["cells"] if cell["cell_type"] == "markdown")
    assert "0061" in "".join(h1["source"])
    assert "K-Means" in "".join(h1["source"])


def test_build_0062_0063_0064_from_canonical_pairs(tmp_path):
    entries = build(ROOT, tmp_path, ["0062", "0063", "0064"])
    assert [entry["id"] for entry in entries] == ["0062", "0063", "0064"]
    expected = {
        "0062": ("Pairwise Box IoU", "Pairwise Box IoU", "def pairwise_iou", "0062_pairwise_iou.py"),
        "0063": ("Oriented Box Sampling", "Uniform Oriented-Box Sampling", "def sample_oriented_box", "0063_oriented_box_sampling.py"),
        "0064": ("Point in Rotated Box", "Point in Rotated Box", "def point_in_rotated_box", "0064_point_in_rotated_box.py"),
    }

    for entry in entries:
        title, heading, reference_name, question_name = expected[entry["id"]]
        assert entry["title"] == title
        assert entry["stages"] == ["baseline", "followup", "production"]
        assert len(entry["hints"]) == 3
        assert reference_name in entry["solution"]
        assert entry["notebook"] == f"notebooks/{question_name.removesuffix('.py')}.ipynb"
        notebook = json.loads((tmp_path / entry["notebook"]).read_text())
        assert notebook["metadata"]["dojo"]["exercise_id"] == entry["id"]
        assert heading in "".join(notebook["cells"][0]["source"])
        solutions = [
            cell
            for cell in notebook["cells"]
            if cell["metadata"].get("dojo", {}).get("role") == "solution"
        ]
        assert len(solutions) == 1
        solution_source = "".join(solutions[0]["source"])
        assert "pass" in solution_source
        assert "torch_judge" not in solution_source

        checker = json.loads((tmp_path / entry["checker"]).read_text())
        assert checker["exercise_id"] == entry["id"]
        assert checker["candidate"]
        assert len(checker["verifiers"]) >= 2


def test_build_0065_and_0066_keep_metadata(tmp_path):
    entries = build(ROOT, tmp_path, ["0065", "0066"])
    ids = [e["id"] for e in entries]
    assert ids == ["0065", "0066"]
    for entry in entries:
        assert entry["stages"] == ["baseline", "followup", "production"]
        assert len(entry["hints"]) == 3
        assert entry["solution"]  # reference solution is present
        assert entry["checker_version"]
    assert entries[0]["title"] == "Temporal Ego Coordinates"
    assert entries[1]["title"] == "Sparse Voxel Pooling"
    # Checker verifier bundles exist
    for entry in entries:
        bundle = json.loads((tmp_path / entry["checker"]).read_text())
        assert bundle["exercise_id"] == entry["id"]
        assert len(bundle["verifiers"]) >= 5
        assert bundle["candidate"] in {"to_current_ego", "voxel_pool"}


@pytest.mark.parametrize("exercise_id", ["0059", "mlp"])
def test_lesson_loaders_use_project_pages_prefix(tmp_path, exercise_id):
    entries = build(ROOT, tmp_path, [exercise_id], base_url="/torch_dojo/")
    notebook = json.loads((tmp_path / entries[0]["notebook"]).read_text())
    loaders = ["".join(cell["source"]) for cell in notebook["cells"]
               if cell["metadata"].get("dojo", {}).get("role") == "exploration"]
    assert loaders
    assert all("pyfetch('/torch_dojo/gym-assets/lessons/" in loader for loader in loaders)


@pytest.mark.parametrize("base_url", ["https://example.com/", "/missing-slash", "/../", "/bad'path/"])
def test_rejects_invalid_site_prefix(tmp_path, base_url):
    with pytest.raises(ValueError, match="base_url"):
        build(ROOT, tmp_path, ["0059"], base_url=base_url)


def test_briefs_are_readonly_with_code_and_typeset_equations(tmp_path):
    entries = build(ROOT, tmp_path, ["0060"])
    notebook = json.loads((tmp_path / entries[0]["notebook"]).read_text())
    brief = notebook["cells"][0]
    assert brief["metadata"]["editable"] is False
    source = "".join(brief["source"])
    assert '<span class="dojo-math">' in source
    assert r"\nabla_W" in source
    assert '<code>float64</code>' in source
    solution = next(cell for cell in notebook["cells"] if cell["metadata"].get("dojo", {}).get("role") == "solution")
    assert solution["metadata"].get("editable") is not False


def test_checker_bundle_carries_the_helpers_verifiers_call(tmp_path):
    entry = next(item for item in build(ROOT, tmp_path, ["0091"]) if item["id"] == "0091")
    bundle = json.loads((tmp_path / entry["checker"]).read_text())
    assert "def _chain" in bundle["helpers"]
    assert all(not item["name"].startswith("_") for item in bundle["verifiers"])
