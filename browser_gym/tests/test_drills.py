import json
from pathlib import Path

import pytest

from browser_gym.build_catalog import BriefError, brief_sections, build, drills

ROOT = Path(__file__).resolve().parents[2]
STAGES = ["baseline", "followup", "production"]

QUESTION = """\
# %%
# 0059 · T
# Goal: Build it.

# %%
import numpy as np

# %% [drill]
# Paired or grid?
# Predict both printed arrays.
import numpy as np
x = np.arange(6).reshape(2, 3)
print(x[[0, 1], [2, 0]])

# %% [drill]
# Overflow
print(np.exp(1000.0))

# %% [answer]
def f():
    pass
"""

EVAL = """\
# %%
import numpy as np

# Drill 1: Paired indexing takes one value per row: $x_{0,2}$ and $x_{1,0}$.
# A slice would build a grid instead.

# Drill 2: `exp(1000)` overflows float64.

# %% [verifiers]
def verify_baseline_a(f):
    pass
"""


def test_drill_cells_pair_with_their_eval_reasoning():
    found = drills(QUESTION, EVAL, "q.py", "q_eval.py")
    assert [drill["title"] for drill in found] == ["Paired or grid?", "Overflow"]
    assert found[0]["prompt_html"] == "<p>Predict both printed arrays.</p>"
    assert found[0]["probe_source"] == "import numpy as np\nx = np.arange(6).reshape(2, 3)\nprint(x[[0, 1], [2, 0]])\n"
    assert found[0]["reasoning_html"].startswith("<p>Paired indexing takes one value per row: ")
    assert r'<span class="dojo-math">\(x_{0,2}\)</span>' in found[0]["reasoning_html"]
    assert "A slice would build a grid instead." in found[0]["reasoning_html"]
    assert found[1]["prompt_html"] == ""
    assert found[1]["reasoning_html"] == "<p><code>exp(1000)</code> overflows float64.</p>"


def test_drill_comments_stay_out_of_the_brief():
    sections = brief_sections(QUESTION, "0059", STAGES, "q.py")
    assert all("Paired or grid" not in section["html"] for section in sections)
    assert all("Overflow" not in section["html"] for section in sections)


@pytest.mark.parametrize(
    ("eval_source", "message"),
    [
        (EVAL.replace("# Drill 2: `exp(1000)` overflows float64.\n", ""), "q.py:15: drill 2 has no '# Drill 2:' reasoning in q_eval.py"),
        (EVAL + "# Drill 3: Nothing to explain.\n", "q_eval.py:12: '# Drill 3:' has no matching drill in q.py"),
        (EVAL.replace("# Drill 2:", "# Drill 1:"), "q_eval.py:7: duplicate '# Drill 1:' reasoning"),
    ],
)
def test_build_rejects_mismatched_drills_and_reasoning(eval_source, message):
    with pytest.raises(BriefError) as error:
        drills(QUESTION, eval_source, "q.py", "q_eval.py")
    assert message in str(error.value)


def test_scripts_without_drills_have_none():
    assert drills("# %%\n# 0059 · T\n\n# %% [answer]\ndef f():\n    pass\n", "", "q.py", "q_eval.py") == []


def test_0059_drills_become_catalog_entries_and_notebook_cells_never_checked(tmp_path):
    entry = build(ROOT, tmp_path, ["0059"])[0]
    assert len(entry["drills"]) >= 2
    assert all(drill["reasoning_html"] for drill in entry["drills"])
    notebook = json.loads((tmp_path / entry["notebook"]).read_text())
    cells = [cell for cell in notebook["cells"] if cell["metadata"].get("dojo", {}).get("role") == "drill"]
    assert ["".join(cell["source"]) for cell in cells] == [drill["probe_source"] for drill in entry["drills"]]
    assert [cell["metadata"]["dojo"]["drill"] for cell in cells] == list(range(len(cells)))
    # Only the answer cell is ever submitted to Check.
    solutions = [cell for cell in notebook["cells"] if cell["metadata"].get("dojo", {}).get("role") == "solution"]
    assert len(solutions) == 1
    assert all(drill["probe_source"] not in "".join(solutions[0]["source"]) for drill in entry["drills"])
    # Imports inside drill cells don't leak into the answer cell's carried imports.
    assert "".join(solutions[0]["source"]).count("import numpy as np") == 1


def test_mlp_lab_brief_is_typed_by_checkpoint_and_has_drills(tmp_path):
    entry = next(item for item in build(ROOT, tmp_path, ["mlp"]) if item["id"] == "mlp")
    kinds = [(section["kind"], section.get("stage")) for section in entry["brief"]]
    assert kinds[0] == ("goal", None)
    assert ("watch-out", None) in kinds
    assert ("watch-out", "backward") in kinds
    assert {stage for _, stage in kinds if stage} == set(entry["stages"])
    assert [drill["title"] for drill in entry["drills"]] == [
        "Mean or sum?", "Gradient through ReLU at zero", "Momentum's second step"]
    assert all(drill["reasoning_html"] for drill in entry["drills"])
    notebook = json.loads((tmp_path / entry["notebook"]).read_text())
    brief = "".join(notebook["cells"][0]["source"])
    assert brief.startswith("# MLP Modeling Lab")
    assert "<h3>Backward</h3>" in brief
    drill_cells = [cell for cell in notebook["cells"] if cell["metadata"].get("dojo", {}).get("role") == "drill"]
    assert len(drill_cells) == 3
    # The learner's starter carries no reasoning.
    solution = next(cell for cell in notebook["cells"] if cell["metadata"].get("dojo", {}).get("role") == "solution")
    assert "Drill 1" not in "".join(solution["source"])
