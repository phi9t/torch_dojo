import ast
import re
from pathlib import Path

from browser_gym.build_catalog import build, extract_cell
from browser_gym.check_runtime import run_check


ROOT = Path(__file__).resolve().parents[2]


def _sources(exercise_id: str, question_name: str, function_name: str):
    question = (ROOT / f"dojo/{exercise_id}_{question_name}.py").read_text()
    reference = (ROOT / f"dojo/{exercise_id}_{question_name}_eval.py").read_text()
    tree = ast.parse(reference)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function_name)
    imports = [
        ast.get_source_segment(reference, node)
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom)) and node.lineno < function.lineno
    ]
    return extract_cell(question, "answer"), "\n".join(imports) + "\n" + ast.get_source_segment(reference, function)


def _dojo_reference(exercise_id, function_name):
    import ast
    reference = next(ROOT.glob(f"dojo/{exercise_id}_*_eval.py")).read_text()
    tree = ast.parse(reference)
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    return "import numpy as np\n" + ast.get_source_segment(reference, function)


def test_reference_starter_and_batch_gradient_mutant(tmp_path):
    build(ROOT, tmp_path, ["0059"])
    starter, reference = _sources("0059", "stable_cross_entropy", "softmax_cross_entropy")
    passed = run_check(reference, "0059", tmp_path, stage="production")
    assert passed["status"] == "passed"
    assert passed["runtime"] == "python"
    assert passed["package_identity"]["runtime"] == "python"
    assert passed["package_identity"]["numpy"]
    assert passed["fixture_id"] == "canonical-0059"
    assert run_check("import numpy as np\n" + starter, "0059", tmp_path, stage="production")["status"] == "failed"
    mutant = reference.replace("gradient[valid] = probabilities / count", "gradient[valid] = probabilities / (count * count)")
    assert mutant != reference
    result = run_check(mutant, "0059", tmp_path, stage="production")
    assert result["status"] == "failed"
    assert any("finite_difference" in item["name"] and item["status"] == "failed" for item in result["checks"])


def test_numerically_unstable_mutant_fails_extreme_logits(tmp_path):
    build(ROOT, tmp_path, ["0059"])
    _, reference = _sources("0059", "stable_cross_entropy", "softmax_cross_entropy")
    mutant = reference.replace("q = selected - selected.max(axis=1, keepdims=True)", "q = selected")
    assert mutant != reference
    result = run_check(mutant, "0059", tmp_path, stage="production")
    assert result["status"] == "failed"
    assert any("extreme_logits" in item["name"] and item["status"] == "failed" for item in result["checks"])


def test_stages_only_run_cumulative_verifiers(tmp_path):
    build(ROOT, tmp_path, ["0059"])
    baseline = """import numpy as np
def softmax_cross_entropy(logits, targets, ignore_index=-100):
    q = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(q)
    p = e / e.sum(axis=1, keepdims=True)
    grad = p.copy()
    grad[np.arange(len(logits)), targets] -= 1
    grad /= len(logits)
    loss = np.mean(np.log(e.sum(axis=1)) - q[np.arange(len(logits)), targets])
    return float(loss), grad
"""
    first = run_check(baseline, "0059", tmp_path)
    assert first["status"] == "passed"
    assert first["stage"] == "baseline"
    assert len(first["checks"]) == 3
    followup = run_check(baseline, "0059", tmp_path, stage="followup")
    assert followup["status"] == "failed"
    assert len(followup["checks"]) == 5


def test_invalid_source_and_missing_answer_have_specific_errors(tmp_path):
    build(ROOT, tmp_path, ["0059"])
    invalid = run_check("def softmax_cross_entropy(:\n    pass", "0059", tmp_path)
    assert invalid["status"] == "failed"
    assert "SyntaxError" in invalid["error"]
    assert "solution.py" in invalid["traceback"]
    missing = run_check("import numpy as np", "0059", tmp_path)
    assert missing["status"] == "failed"
    assert "Missing candidate function" in missing["error"]


def test_isolated_source_and_import_error(tmp_path):
    build(ROOT, tmp_path, ["0059"])
    _, reference = _sources("0059", "stable_cross_entropy", "softmax_cross_entropy")
    assert run_check(reference + "\nscratch_value = 2", "0059", tmp_path)["status"] == "passed"
    isolated = run_check("print(scratch_value)\n" + reference, "0059", tmp_path)
    assert isolated["status"] == "failed"
    assert "NameError" in isolated["error"]
    missing = run_check("import not_an_installed_package", "0059", tmp_path)
    assert missing["status"] == "failed"
    assert "ModuleNotFoundError" in missing["error"]


def _exercise_sources(exercise_id: str) -> tuple[str, str]:
    question_paths = [path for path in (ROOT / "dojo").glob(f"{exercise_id}_*.py") if not path.stem.endswith("_eval")]
    assert len(question_paths) == 1
    question_source = question_paths[0].read_text()
    starter = extract_cell(question_source, "answer")

    eval_source = question_paths[0].with_name(question_paths[0].stem + "_eval.py").read_text()
    tree = ast.parse(eval_source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("verify_")
    )
    imports = [
        ast.get_source_segment(eval_source, node)
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom)) and node.lineno < function.lineno
    ]
    reference = "\n".join(imports) + "\n" + ast.get_source_segment(eval_source, function) + "\n"
    return starter, reference


BOX_MUTANTS = {
    "0062": (
        lambda source: source.replace(
            "- np.maximum(aa[:, None, 0], bb[None, :, 0])",
            "+ np.maximum(aa[:, None, 0], bb[None, :, 0])",
        ),
        "generated_box_pairs",
    ),
    "0063": (
        lambda source: source.replace("local @ R.T + center", "local @ R + center"),
        "bounds_and_replay",
    ),
    "0064": (
        lambda source: source.replace("<= ss[None, :, 0] / 2 + 1e-10", "< ss[None, :, 0] / 2"),
        "faces_and_inverse_transform",
    ),
}


def test_reference_starter_and_plausible_mutants_for_box_geometry_exercises(tmp_path):
    build(ROOT, tmp_path, ["0062", "0063", "0064"])
    for exercise_id in ("0062", "0063", "0064"):
        starter, reference = _exercise_sources(exercise_id)
        passed = run_check(reference, exercise_id, tmp_path, stage="production")
        assert passed["status"] == "passed"
        assert passed["fixture_id"] == f"canonical-{exercise_id}"
        if exercise_id == "0062":
            assert "hypothesis" in (tmp_path / "checkers/0062_pairwise_iou.json").read_text()
        failed = run_check("import numpy as np\n" + starter, exercise_id, tmp_path, stage="production")
        assert failed["status"] == "failed"
        assert failed["error"] is None
        assert any(item["status"] == "failed" for item in failed["checks"])
        mutate, intended = BOX_MUTANTS[exercise_id]
        mutant = mutate(reference)
        assert mutant != reference
        result = run_check(mutant, exercise_id, tmp_path, stage="production")
        assert result["status"] == "failed"
        assert any(
            intended in item["name"] and item["status"] == "failed"
            for item in result["checks"]
        ), result


def _kmeans_sources():
    question = (ROOT / "dojo/0061_kmeans.py").read_text()
    reference = (ROOT / "dojo/0061_kmeans_eval.py").read_text()
    import ast
    tree = ast.parse(reference)
    answer = extract_cell(question, "answer")
    assign_fn = next(node for node in tree.body
                     if isinstance(node, ast.FunctionDef) and node.name == "_assign")
    kmeans_fn = next(node for node in tree.body
                     if isinstance(node, ast.FunctionDef) and node.name == "kmeans")
    # Only include stdlib/NumPy imports in the browser-compatible submission.
    # Hypothesis lives in the verifiers section of the canonical eval file; we
    # need only the actual runtime helpers the reference function uses.
    reference_source = (
        "import numpy as np\n\n"
        + ast.get_source_segment(reference, assign_fn)
        + "\n\n"
        + ast.get_source_segment(reference, kmeans_fn)
        + "\n"
    )
    return answer, reference_source


def test_0061_reference_starter_and_empty_cluster_mutant(tmp_path):
    build(ROOT, tmp_path, ["0061"])
    starter, reference = _kmeans_sources()
    passed = run_check(reference, "0061", tmp_path, stage="production")
    assert passed["status"] == "passed", [c for c in passed["checks"] if c["status"] == "failed"]
    assert passed["fixture_id"] == "canonical-0061"
    # Starter (pass) fails every verifier
    starter_check = run_check("import numpy as np\n" + starter, "0061", tmp_path, stage="baseline")
    assert starter_check["status"] == "failed"
    # Mutant: update unoccupied centers too, so zero counts produce NaN
    mutant = reference.replace(
        "occupied = counts > 0",
        "occupied = np.ones_like(counts, dtype=bool)"
    )
    assert mutant != reference, "could not apply empty-center mutation"
    mutant_result = run_check(mutant, "0061", tmp_path, stage="followup")
    assert mutant_result["status"] == "failed"
    assert any(
        "empty_center" in item["name"] and item["status"] == "failed"
        for item in mutant_result["checks"]
    ), [c["name"] for c in mutant_result["checks"] if c["status"] == "failed"]


def test_0061_stale_labels_mutant_fails_final_assignment(tmp_path):
    build(ROOT, tmp_path, ["0061"])
    _, reference = _kmeans_sources()
    # Remove the final reassign so labels come from the last update, not returned centers
    # The canonical reference pattern (before inertia):
    #     labels, distance = _assign(points, centers, block_size)
    #     inertia = float(distance.sum())
    mutant = re.sub(
        r"    labels, distance = _assign\(points, centers, block_size\)\n    inertia = float\(distance\.sum\(\)\)",
        "    # stale-labels mutant: reuse labels from the last update iteration\n"
        "    inertia = float(_assign(points, centers, block_size)[1].sum())",
        reference,
        count=1,
    )
    assert mutant != reference, "could not locate final assign block for stale-labels mutant"
    result = run_check(mutant, "0061", tmp_path, stage="production")
    assert result["status"] == "failed"


def test_0061_stages_and_isolation(tmp_path):
    build(ROOT, tmp_path, ["0061"])
    _, reference = _kmeans_sources()
    # Exploration kernel scratch name should not leak into checker
    poisoned = "exploration_result = {}\n" + reference
    assert run_check(poisoned, "0061", tmp_path, stage="baseline")["status"] == "passed"
    isolated = run_check("print(exploration_result)\n" + reference, "0061", tmp_path, stage="baseline")
    assert isolated["status"] == "failed"
    assert "NameError" in isolated["error"]
    # Baseline passes with a minimal correct implementation
    minimal = """import numpy as np
def kmeans(X, initial_centers, max_iter=100, *, block_size=1024):
    points = np.asarray(X, dtype=float)
    centers = np.asarray(initial_centers, dtype=float).copy()
    for _ in range(max_iter):
        diffs = points[:, None, :] - centers[None, :, :]
        dists = (diffs * diffs).sum(2)
        labels = dists.argmin(1).astype(np.int64)
        updated = centers.copy()
        for idx in range(len(centers)):
            sel = labels == idx
            if sel.any():
                updated[idx] = points[sel].mean(0)
        if np.array_equal(updated, centers):
            break
        centers = updated
    diffs = points[:, None, :] - centers[None, :, :]
    dists = (diffs * diffs).sum(2)
    labels = dists.argmin(1).astype(np.int64)
    inertia = float(dists[np.arange(len(labels)), labels].sum())
    return centers, labels, inertia
"""
    baseline_result = run_check(minimal, "0061", tmp_path, stage="baseline")
    assert baseline_result["status"] == "passed"
    followup_result = run_check(minimal, "0061", tmp_path, stage="followup")
    assert followup_result["status"] == "passed"


def test_grouped_bootstrap_reference_starter_and_metric_mutant(tmp_path):
    build(ROOT, tmp_path, ["0073"])
    starter, reference = _sources("0073", "paired_trip_bootstrap", "paired_trip_bootstrap")

    passed = run_check(reference, "0073", tmp_path, stage="production")
    assert passed["status"] == "passed"
    assert passed["fixture_id"] == "canonical-0073"
    assert any(item["name"] == "verify_followup_manual_paired_replay" for item in passed["checks"])

    starter_result = run_check("import numpy as np\n" + starter, "0073", tmp_path, stage="production")
    assert starter_result["status"] == "failed"
    assert any(item["name"] == "verify_baseline_constant_improvement" and item["status"] == "failed"
               for item in starter_result["checks"])

    wrong_sign = reference.replace(
        "differences = baseline.astype(np.float64) - candidate.astype(np.float64)",
        "differences = candidate.astype(np.float64) - baseline.astype(np.float64)",
    )
    assert wrong_sign != reference
    mutant = run_check(wrong_sign, "0073", tmp_path, stage="production")
    assert mutant["status"] == "failed"
    assert any("point_estimate_and_sign" in item["name"] and item["status"] == "failed"
               for item in mutant["checks"])


def test_detection_ap_reference_starter_and_metric_mutant(tmp_path):
    build(ROOT, tmp_path, ["0074"])
    starter, reference = _sources("0074", "detection_ap", "detection_ap")

    passed = run_check(reference, "0074", tmp_path, stage="production")
    assert passed["status"] == "passed"
    assert passed["fixture_id"] == "canonical-0074"
    assert any(item["name"] == "verify_followup_duplicate_and_precision_envelope" for item in passed["checks"])

    starter_result = run_check("import numpy as np\n" + starter, "0074", tmp_path, stage="production")
    assert starter_result["status"] == "failed"
    assert any(item["name"] == "verify_baseline_perfect_and_empty" and item["status"] == "failed"
               for item in starter_result["checks"])

    wrong_denominator = reference.replace(
        "recall = true_positives / total_gt",
        "recall = true_positives / max(len(claimed), 1)",
    )
    assert wrong_denominator != reference
    mutant = run_check(wrong_denominator, "0074", tmp_path, stage="production")
    assert mutant["status"] == "failed"
    assert any("missed_gt_denominator" in item["name"] and item["status"] == "failed"
               for item in mutant["checks"])


def _sensor_sources():
    question = (ROOT / "dojo/0065_temporal_transforms.py").read_text()
    reference = (ROOT / "dojo/0065_temporal_transforms_eval.py").read_text()
    import ast
    tree = ast.parse(reference)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "to_current_ego")
    return extract_cell(question, "answer"), "import numpy as np\n" + ast.get_source_segment(reference, function)

def test_0065_reference_passes_and_starter_fails(tmp_path):
    build(ROOT, tmp_path, ["0065"])
    starter, reference = _sensor_sources()
    assert run_check(reference, "0065", tmp_path, stage="production")["status"] == "passed"
    assert run_check("import numpy as np\n" + starter, "0065", tmp_path, stage="production")["status"] == "failed"

def test_0065_wrong_rotation_order_mutant_fails(tmp_path):
    build(ROOT, tmp_path, ["0065"])
    _, reference = _sensor_sources()
    # Plausible mutant: forgets to apply the time-step rotation (uses identity R_t),
    # so the rotated point stays in the old ego orientation.
    mutant = reference.replace(
        "world = np.einsum('btij,batj->bati', R_t, points) + t_t[:, None]",
        "world = points + t_t[:, None]",
    )
    assert mutant != reference
    result = run_check(mutant, "0065", tmp_path, stage="followup")
    assert result["status"] == "failed"
    assert any("rotation" in item["name"] and item["status"] == "failed" for item in result["checks"])

def _voxel_sources():
    question = (ROOT / "dojo/0066_voxel_pool.py").read_text()
    reference = (ROOT / "dojo/0066_voxel_pool_eval.py").read_text()
    import ast
    tree = ast.parse(reference)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "voxel_pool")
    return extract_cell(question, "answer"), "import numpy as np\n" + ast.get_source_segment(reference, function)

def test_0066_reference_passes_and_starter_fails(tmp_path):
    build(ROOT, tmp_path, ["0066"])
    starter, reference = _voxel_sources()
    assert run_check(reference, "0066", tmp_path, stage="production")["status"] == "passed"
    assert run_check("import numpy as np\n" + starter, "0066", tmp_path, stage="production")["status"] == "failed"

def test_0066_naive_assignment_mutant_fails_duplicate_points(tmp_path):
    build(ROOT, tmp_path, ["0066"])
    _, reference = _voxel_sources()
    # Mutant: uses fancy-indexed assignment instead of np.add.at, dropping duplicate features
    mutant = reference.replace(
        "np.add.at(sums, local_inverse, features[inside])",
        "sums[local_inverse] = features[inside]",
    )
    assert mutant != reference
    result = run_check(mutant, "0066", tmp_path, stage="baseline")
    assert result["status"] == "failed"
    assert any("repeated" in item["name"] and item["status"] == "failed" for item in result["checks"])

def test_0060_reference_starter_and_wrong_microbatch_weighting(tmp_path):
    build(ROOT, tmp_path, ["0060"])
    starter, reference = _sources("0060", "classifier_step", "classifier_step")

    passed = run_check(reference, "0060", tmp_path, stage="production")
    assert passed["status"] == "passed"
    assert passed["fixture_id"] == "canonical-0060"
    assert [item["name"] for item in passed["checks"]] == [
        "verify_baseline_one_step_and_preupdate_loss",
        "verify_baseline_loss_decreases",
        "verify_followup_ignore_and_all_ignored",
        "verify_followup_zero_rate_returns_copies",
        "verify_production_weight_unequal_microbatches",
        "verify_production_validation_and_no_mutation",
    ]

    starter_result = run_check("import numpy as np\n" + starter, "0060", tmp_path, stage="baseline")
    assert starter_result["status"] == "failed"
    assert any(
        item["name"] == "verify_baseline_one_step_and_preupdate_loss" and item["status"] == "failed"
        for item in starter_result["checks"]
    )

    mutant = reference.replace("probabilities /= count", "probabilities /= len(X)")
    assert mutant != reference
    result = run_check(mutant, "0060", tmp_path, stage="production")
    assert result["status"] == "failed"
    assert any(
        item["name"] == "verify_production_weight_unequal_microbatches" and item["status"] == "failed"
        for item in result["checks"]
    )


def test_0069_reference_starter_and_wrong_bin_edge_metric(tmp_path):
    build(ROOT, tmp_path, ["0069"])
    starter, reference = _sources("0069", "calibration_metrics", "calibration_metrics")

    passed = run_check(reference, "0069", tmp_path, stage="production")
    assert passed["status"] == "passed"
    assert passed["fixture_id"] == "canonical-0069"
    assert [item["name"] for item in passed["checks"]] == [
        "verify_baseline_perfect_and_wrong",
        "verify_baseline_known_denominators",
        "verify_followup_bin_edges_and_empty_bins",
        "verify_followup_argmax_tie_and_classwise_brier",
        "verify_production_validation_and_immutability",
        "verify_production_generated_one_hot",
    ]

    starter_result = run_check("import numpy as np\n" + starter, "0069", tmp_path, stage="baseline")
    assert starter_result["status"] == "failed"
    assert any(
        item["name"] == "verify_baseline_perfect_and_wrong" and item["status"] == "failed"
        for item in starter_result["checks"]
    )

    mutant = reference.replace(
        "bucket = np.clip(np.ceil(confidence * bins).astype(int) - 1, 0, bins - 1)",
        "bucket = np.clip(np.floor(confidence * bins).astype(int), 0, bins - 1)",
    )
    assert mutant != reference
    result = run_check(mutant, "0069", tmp_path, stage="production")
    assert result["status"] == "failed"
    assert any(
        item["name"] == "verify_followup_bin_edges_and_empty_bins" and item["status"] == "failed"
        for item in result["checks"]
    )


def test_0070_reference_starter_and_collision_mutant(tmp_path):
    build(ROOT, tmp_path, ["0070"])
    starter = "import numpy as np\n" + extract_cell((ROOT / "dojo/0070_anchor_match.py").read_text(), "answer")
    reference = _dojo_reference("0070", "match_anchors")

    passed = run_check(reference, "0070", tmp_path, stage="production")
    assert passed["status"] == "passed"
    assert passed["fixture_id"] == "canonical-0070"
    assert any(item["name"] == "verify_followup_forced_low_iou_and_collision" for item in passed["checks"])

    starter_result = run_check(starter, "0070", tmp_path, stage="production")
    assert starter_result["status"] == "failed"
    assert starter_result["checks"][0]["name"] == "verify_baseline_thresholds"

    mutant = reference.replace(
        "for gt in range(ground_truths):\n"
        "        anchor = int(iou[:, gt].argmax())\n"
        "        if anchor not in claimed:\n"
        "            result[anchor] = gt\n"
        "            claimed.add(anchor)",
        "for gt in range(ground_truths):\n"
        "        anchor = int(iou[:, gt].argmax())\n"
        "        result[anchor] = gt",
    )
    assert mutant != reference
    result = run_check(mutant, "0070", tmp_path, stage="production")
    assert result["status"] == "failed"
    assert any(
        item["name"] == "verify_followup_forced_low_iou_and_collision" and item["status"] == "failed"
        for item in result["checks"]
    )


def test_samples_capture_inputs_actual_and_expected_without_changing_verdict(tmp_path):
    import json
    from browser_gym.check_runtime import _unpack_value
    import numpy as np
    build(ROOT, tmp_path, ["0059"])
    source = '''import numpy as np
def softmax_cross_entropy(logits, targets, ignore_index=-100):
    q = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(q)
    p = e / e.sum(axis=1, keepdims=True)
    log_p = q - np.log(e.sum(axis=1, keepdims=True))
    return float(-(targets * log_p).sum(axis=1).mean()), (p-targets)/len(logits)
'''
    result = run_check(source, "0059", tmp_path)
    assert result["status"] == "failed"
    sample = result["checks"][0]["samples"][0]
    args = _unpack_value(sample["args"])
    actual = _unpack_value(sample["actual"])
    expected = _unpack_value(sample["expected"])
    assert args[1].shape == (1,)
    assert np.isclose(actual[0], 2 * np.log(2))
    assert np.isclose(expected[0], np.log(2))
    assert np.allclose(expected[1], [[.5, -.5]])
    assert sample["replayable"]
    json.dumps(result, allow_nan=False)


def test_samples_are_bounded_and_preserve_pre_mutation_inputs():
    import numpy as np
    from browser_gym.check_runtime import _CallSamples, _unpack_value
    def mutate(x):
        x[:] = -1
        return np.array([np.nan, np.inf])
    collector = _CallSamples(mutate, "mutate")
    for i in range(20):
        collector(np.array([float(i)]))
    samples = collector.finish(lambda x: x)
    assert [s["call"] for s in samples] == [1, 2, 3, 18, 19, 20]
    assert _unpack_value(samples[0]["args"])[0][0] == 0
    assert _unpack_value(samples[-1]["expected"])[0] == 19
    assert np.isnan(_unpack_value(samples[0]["actual"])[0])
    large = _CallSamples(lambda x: None, "large")
    large(np.zeros(300))
    assert large.finish(None)[0]["replayable"] is False


def test_sample_large_integer_roundtrip_survives_json():
    import json
    import numpy as np
    from browser_gym.check_runtime import _pack_value, _unpack_value
    value = np.array([2**63 - 1], dtype=np.int64)
    packed = _pack_value(value)
    restored = _unpack_value(json.loads(json.dumps(packed)))
    assert np.array_equal(restored, value)
