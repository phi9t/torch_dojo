import pytest

from browser_gym.build_catalog import BriefError, brief_sections

STAGES = ["baseline", "followup", "production"]

MARKED = """\
# %%
# 0059 · Stable cross-entropy
# Goal: Return the mean loss and its gradient for integer targets.
# Implement it in NumPy, with no autograd.

# %%
import numpy as np

# %%
# Predict: what is the shape of `normalizers - selected`?
# Watch out: a $(N,)$ minus an $(N,1)$ broadcasts to $(N,N)$.
#
# Plain prose after a blank comment line.
# Why: every pair appears once, so
# $$\\operatorname{mean}(a_i - b_j) = \\bar a - \\bar b$$
x = 1

# %%
# Baseline — Work with $q = z - \\max_c z_c$.
# Goal: the loss is $\\log \\sum_c e^{q_c} - q_y$.

# %% [answer]
def f():
    pass

# %%
# Follow-up — Ignore rows whose target is ignore_index.
# Watch out: validate targets before indexing.
"""


def kinds(sections):
    return [(section["kind"], section.get("stage")) for section in sections]


def test_markers_become_typed_sections_scoped_to_their_stage():
    sections = brief_sections(MARKED, "0059", STAGES, "0059_demo.py")
    assert kinds(sections) == [
        ("goal", None),
        ("predict", None),
        ("watch-out", None),
        ("prose", None),
        ("why", None),
        ("prose", "baseline"),
        ("goal", "baseline"),
        ("prose", "followup"),
        ("watch-out", "followup"),
    ]
    goal = sections[0]["html"]
    assert goal == "<p>Return the mean loss and its gradient for integer targets. Implement it in NumPy, with no autograd.</p>"


def test_math_passes_through_escaped_and_wrapped():
    sections = brief_sections(MARKED, "0059", STAGES, "0059_demo.py")
    watch = sections[2]["html"]
    assert r'<span class="dojo-math">\((N,)\)</span>' in watch
    why = sections[4]["html"]
    assert r'<div class="dojo-math-display">\[\operatorname{mean}(a_i - b_j) = \bar a - \bar b\]</div>' in why
    goal = sections[6]["html"]
    assert r'\(\log \sum_c e^{q_c} - q_y\)' in goal


def test_math_is_escaped_not_trusted_html():
    source = "# %%\n# 0001 · T\n# Goal: compare $a < b$ with <b>bold</b>.\n"
    html = brief_sections(source, "0001", STAGES, "s.py")[0]["html"]
    assert r"\(a &lt; b\)" in html
    assert "&lt;b&gt;bold&lt;/b&gt;" in html


def test_code_identifiers_render_in_the_code_face():
    sections = brief_sections(MARKED, "0059", STAGES, "0059_demo.py")
    assert "<code>normalizers - selected</code>" in sections[1]["html"]


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("# Watchout: indexing.", "unknown marker 'Watchout'; did you mean 'Watch out'?"),
        ("# goal: lower case.", "unknown marker 'goal'; did you mean 'Goal'?"),
        ("# Goal: pay $5 now.", "unbalanced $"),
        ("# Goal: $$x = 1$ oops", "unbalanced $"),
    ],
)
def test_rejects_malformed_marked_scripts_naming_script_and_line(line, message):
    source = "# %%\n# 0059 · T\n# Goal: fine.\n" + line + "\n"
    with pytest.raises(BriefError) as error:
        brief_sections(source, "0059", STAGES, "0059_demo.py")
    assert message in str(error.value)
    assert "0059_demo.py:4" in str(error.value)


def test_rejects_a_stage_the_workout_does_not_have():
    source = "# %%\n# 0059 · T\n# Goal: fine.\n# Production challenge — too far.\n"
    with pytest.raises(BriefError, match=r"0059_demo.py:4: stage 'production' is not one of this workout's stages"):
        brief_sections(source, "0059", ["baseline", "followup"], "0059_demo.py")


def test_scripts_opt_in_with_a_goal():
    unmarked = "# %%\n# 0059 · T\n# Implement it.\n# Predict: the shape.\n# Costs $5.\n\n# %%\n# Baseline — Do the thing.\n"
    assert brief_sections(unmarked, "0059", STAGES, "s.py") is None


def test_ordinary_prose_labels_are_not_markers():
    source = "# %%\n# 0059 · T\n# Goal: fine.\n#\n# Tiny example: the first row is uncertain.\n"
    sections = brief_sections(source, "0059", STAGES, "s.py")
    assert [section["kind"] for section in sections] == ["goal", "prose"]
    assert "Tiny example: the first row" in sections[1]["html"]


def test_escaped_dollars_and_dollars_in_code_are_not_math():
    source = "# %%\n# 0059 · T\n# Goal: pay \\$5, keep `a $ b` as code, and typeset $x$.\n"
    html = brief_sections(source, "0059", STAGES, "s.py")[0]["html"]
    assert "pay $5" in html
    assert "<code>a $ b</code>" in html
    assert r'<span class="dojo-math">\(x\)</span>' in html
