"""Derive browser notebooks and checker bundles without executing dojo files."""

from __future__ import annotations

import ast
import hashlib
import html
import importlib.util
import json
import os
import re
import sys
from pathlib import Path


# Also support `python build_catalog.py` from the standalone browser directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from torch_judge.review import motifs
from torch_judge.scripts import TaggedScript, script_cells as _cells

# Canonical dojo exercise metadata. Keys are exercise IDs with the browser
# qualification checklist accepted. Each entry supplies the notebook title,
# per-stage hints, and the human-friendly stage order.
QUALIFIED_DOJO_EXERCISES: dict[str, dict[str, object]] = {
    "0059": {
        "title": "Stable Softmax Cross-Entropy",
        "heading": "Stable Softmax Cross-Entropy",
        "stages": ["baseline", "followup", "production"],
        "concepts": ["training", "stability", "probabilities", "gradients"],
        "hints": [
            "Subtract each row maximum before exponentiating logits.",
            "Average only valid rows and leave ignored rows with zero gradient.",
            "Check the gradient by subtracting one from the target probability, then dividing by the valid-row count.",
        ],
        "allowed_helpers": ["numpy"],
    },
    "0060": {
        "title": "One honest classifier update",
        "heading": "One honest classifier update",
        "stages": ["baseline", "followup", "production"],
        "concepts": ["training", "stability", "gradients"],
        "hints": [
            "Reuse the stable 0059 log-sum-exp pattern before taking probabilities.",
            "Compute dW and db from one pre-update snapshot, then return new arrays.",
            "For ignored rows, divide by the valid-example count so uneven microbatches still add up.",
        ],
        "allowed_helpers": ["numpy"],
    },
    "0062": {
        "title": "Pairwise Box IoU",
        "concepts": ["geometry", "boxes", "overlap"],
        "hints": [
            "Compute intersection width/height with min/max and clamp at zero.",
            "IoU is intersection / union, where union = area(a) + area(b) - intersection.",
            "For empty inputs or zero-union pairs, return 0 without dividing by zero.",
        ],
        "allowed_helpers": ["numpy"],
    },
    "0063": {
        "title": "Oriented Box Sampling",
        "concepts": ["geometry", "rotation", "sampling"],
        "heading": "Uniform Oriented-Box Sampling",
        "hints": [
            "Sample local coordinates uniformly in [-0.5, 0.5] and scale by size.",
            "Rotate local -> world with world = local @ R.T + center for row-stored points.",
            "Validate inputs: shapes, finiteness, positive size, injected Generator, and integer nonnegative count.",
        ],
        "allowed_helpers": ["numpy"],
    },
    "0064": {
        "title": "Point in Rotated Box",
        "concepts": ["geometry", "rotation", "containment"],
        "hints": [
            "Compute local = (world - center) @ R for row-stored points.",
            "A point is inside when abs(local) <= size/2 (with the given tolerance).",
            "Block points/boxes to avoid a full [N,M,3] displacement tensor.",
        ],
        "allowed_helpers": ["numpy"],
    },
    "0065": {
        "title": "Temporal Ego Coordinates",
        "concepts": ["sensors", "rigid transforms", "time"],
        "hints": [
            "Compose inverse(T_world_from_current) with T_world_from_t; rotations apply to points, translations are added last.",
            "Zero out invalid outputs after the transform; never let NaNs from invalid slots leak into valid results.",
            "Validate proper rigid transforms: R^T R = I, det(R) = 1, bottom row is [0,0,0,1]; preserve float32/float64 dtype.",
        ],
        "allowed_helpers": ["numpy"],
    },
    "0066": {
        "title": "Sparse Voxel Pooling",
        "concepts": ["sensors", "voxels", "pooling"],
        "hints": [
            "Compute voxel coordinates via floor((point - origin) / voxel_size); use the half-open interval [origin, upper).",
            "Use np.unique(..., axis=0, return_inverse=True, return_counts=True) on the inside points to build the sparse map.",
            "Accumulate duplicate-point features with np.add.at; fancy indexing alone overwrites and drops contributions. Always use float64 for means.",
        ],
        "allowed_helpers": ["numpy"],
    },
    "0061": {
        "title": "K-Means Clustering",
        "concepts": ["training", "clustering", "distances"],
        "stages": ["baseline", "followup", "production"],
        "hints": [
            "For each point, compute squared distances to every center and pick the lowest index that wins ties.",
            "Only update a cluster center when at least one point belongs to it -- empty clusters keep the previous center.",
            "After the final update, reassign labels using the returned centers so labels always correspond to the current centroids. Limit workspace with a blockwise distance loop rather than a full [N,K,D] tensor.",
        ],
        "allowed_helpers": ["numpy"],
    },
    "0069": {
        "title": "Multiclass calibration metrics",
        "stages": ["baseline", "followup", "production"],
        "concepts": ["evaluation", "confidence", "calibration"],
        "hints": [
            "Group predictions into equal-width confidence bins before comparing mean accuracy with mean confidence.",
            "Use the left-bin convention for exact interior confidence edges; confidence 1.0 stays in the last bin.",
            "Brier is classwise squared error averaged by row; NLL clips only the true-class log input.",
        ],
        "allowed_helpers": ["numpy"],
    },
    "0070": {
        "title": "Anchor to GT Matching",
        "heading": "Anchor to GT Matching",
        "stages": ["baseline", "followup", "production"],
        "concepts": ["geometry", "assignment", "thresholds"],
        "hints": [
            "Start from each anchor row's best GT and best IoU value before applying thresholds.",
            "Forced matches are column-wise: visit GTs in increasing order and let the best anchor claim an unmatched GT.",
            "Validate finite [A, G] float IoUs and ordered thresholds before computing labels; do not mutate the input.",
        ],
        "allowed_helpers": ["numpy"],
    },
    "0073": {
        "title": "Paired trip bootstrap",
        "concepts": ["evaluation", "uncertainty", "resampling"],
        "hints": [
            "Compute paired trip improvements as baseline minus candidate; positive means the candidate has lower loss.",
            "Bootstrap the trip-level differences with one sampled index vector per replicate, not each model separately.",
            "Return the original trip mean with np.percentile bootstrap bounds, and reject invalid shapes, values, or iteration counts.",
        ],
        "allowed_helpers": ["numpy"],
    },
    "0074": {
        "title": "Detection average precision",
        "concepts": ["evaluation", "ranking", "precision-recall"],
        "hints": [
            "Sort detections by descending score while preserving input order for ties.",
            "Count only the first claim for each (image_id, gt_id) as a true positive; repeats and -1 are false positives.",
            "Use total_gt as the recall denominator, apply the right-to-left precision envelope, and validate malformed inputs.",
        ],
        "allowed_helpers": ["numpy"],
    },
    "0068": {
        "title": "Union of Half-Open Intervals",
        "concepts": ["arrays", "intervals", "sorting"],
        "hints": [
            "Sort the intervals by start, then sweep while tracking the current run's start and end.",
            "Merge when the next start is at or before the current end; touching half-open intervals join without adding length.",
            "Validate shape, finiteness and end >= start with ValueError before sorting, and sort a copy so the input is untouched.",
        ],
        "allowed_helpers": ["numpy"],
    },
    "0067": {
        "title": "Run-Length Random Access",
        "concepts": ["arrays", "run-length encoding", "prefix sums"],
        "hints": [
            "Walk the runs while adding up a running end, and never build the decoded string.",
            "A run's end is exclusive, so stop at the first run where index < end; an index at or past the total raises IndexError.",
            "Validate every run (single-character str, int count > 0, and reject bool before checking int) and keep scanning after finding the answer.",
        ],
        "allowed_helpers": ["numpy"],
    },
    "0071": {
        "title": "Grid Connected Components",
        "concepts": ["arrays", "flood fill", "connectivity"],
        "hints": [
            "Scan every cell, and start a new component at each True cell not yet visited.",
            "Grow it with an explicit stack using only the four up/down/left/right neighbours, marking cells as you push them.",
            "Reject non-2D or non-bool grids with ValueError, and keep the visited marks in a separate array so the input is untouched.",
        ],
        "allowed_helpers": ["numpy"],
    },
    "0076": {
        "title": "Where did the winning pixel come from?",
        "concepts": ["arrays", "max pooling", "argmax coordinates"],
        "hints": [
            "Loop over output windows, slice x[:, :, r*sh:r*sh+kh, c*sw:c*sw+kw], and take the argmax of each flattened window.",
            "Turn the flat index into window (row, col) with divmod by kw, then add the window's top-left corner to get original coordinates.",
            "Validate finite numeric 4-D input, positive non-bool integer kernel/stride pairs, and a kernel no larger than the image, all with ValueError.",
        ],
        "allowed_helpers": ["numpy"],
    },
    "0077": {
        "title": "A matrix made of coordinates",
        "concepts": ["arrays", "sparse COO", "scatter-add"],
        "hints": [
            "Store copies of row, col and value; Ax gathers x at each column and adds into that row.",
            "Use np.add.at so duplicate coordinates sum, and for y^T A gather y at rows and add into columns.",
            "Reject non-integer or out-of-range indices, a malformed shape, nonfinite values and wrong vector lengths with ValueError.",
        ],
        "allowed_helpers": ["numpy"],
    },
    "0035": {
        "title": "A small, deterministic byte-pair-style tokenizer",
        "concepts": ["parsing", "tokenization", "tie-breaking"],
        "hints": [
            "Represent each word as a tuple of characters plus '</w>', and keep a count per unique word.",
            "Each round, count adjacent pairs weighted by word frequency, merge the best pair left to right without overlap, and record it.",
            "Pick the pair with min(counts, key=lambda p: (-counts[p], p)), and have encode replay self.merges in that exact order.",
        ],
        "allowed_helpers": ["numpy"],
    },
    "0072": {
        "title": "Parse CSV records without losing the awkward ones",
        "concepts": ["parsing", "state machines", "streaming"],
        "hints": [
            "Walk the input one character at a time instead of splitting on commas or newlines.",
            "Track a small state (start, plain, quoted, quote-pending, after-quote) so commas and newlines inside quotes count as data.",
            "Keep the state and any pending CR across chunks, and after the loop emit a final row only if something was started.",
        ],
        "allowed_helpers": ["numpy"],
    },
    "0078": {
        "title": "What should a CSV consumer receive?",
        "concepts": ["parsing", "validation", "error records"],
        "hints": [
            "Check the header once against list(schema), and raise ValueError if it doesn't match.",
            "Make one record per data row, numbered from 2, and start data with every header key set to None.",
            "Record a width error, then convert each present field in its own try, catching ValueError, TypeError and OverflowError.",
        ],
        "allowed_helpers": ["numpy"],
    },
    "0079": {
        "title": "A pocket calculator with exactly two operators",
        "concepts": ["parsing", "operator precedence", "scanning"],
        "hints": [
            "Scan once, alternating between reading an integer and reading an operator, and skip whitespace in between.",
            "Keep a running total and a current term: * multiplies the term, + adds the term to the total and starts a new one.",
            "Accept only ASCII digits ('0' <= c <= '9'), and return total + term when the input ends.",
        ],
        "allowed_helpers": ["numpy"],
    },
    "0082": {
        "title": "How much is a policy worth?",
        "concepts": ["reinforcement learning", "policy evaluation", "Bellman expectation"],
        "hints": [
            "Repeat the Bellman expectation update until the values stop changing.",
            "Each sweep computes the action values `rewards + gamma * P @ V` from the previous whole vector, then averages them under the policy.",
            "Zero the terminal states in the new vector after every sweep, stop when the max absolute change is at most `tolerance`, and raise `RuntimeError` if the iterations run out.",
        ],
        "allowed_helpers": ["numpy"],
    },
    "0083": {
        "title": "Learn from a replay tape",
        "concepts": ["reinforcement learning", "Q-learning", "temporal difference"],
        "hints": [
            "Copy the table, then replay the transitions one at a time in recorded order.",
            "The target is `reward + gamma * max(Q[next_state])`, and Q moves by `alpha * (target - Q[s, a])`.",
            "When `done` is True the bootstrap is 0, and each later row reads the already-updated copy.",
        ],
        "allowed_helpers": ["numpy"],
    },
    "0091": {
        "title": "Follow a policy through a tiny world",
        "concepts": ["reinforcement learning", "on-policy sampling", "discounted return"],
        "hints": [
            "Each episode loops: pick an action from the policy row, then a successor from the transition row.",
            "Use the injected `rng.choice`, action first then successor; count the start state once per episode and every successor reached.",
            "Start `discount` at 1 and multiply it by gamma after adding each reward; break on a terminal state, and never act from a terminal start.",
        ],
        "allowed_helpers": ["numpy"],
    },
    "0030": {
        "title": "Cosine learning rate with linear warmup",
        "concepts": ["training", "learning-rate schedules", "warmup"],
        "hints": [
            "Split on the step: a linear warmup ramp, then cosine decay, then a hold at the end.",
            "Warmup is max_lr*step/warmup_steps for step < warmup_steps; cosine progress is (step-warmup)/(total-warmup).",
            "Return min_lr for step >= total_steps before using cosine, check integer counts and finite ordered rates, and return a float.",
        ],
        "allowed_helpers": ["numpy"],
    },
    "0050": {
        "title": "Reverse-mode differentiation through a tiny scalar graph",
        "concepts": ["training", "autodiff", "chain rule"],
        "hints": [
            "Seed the last node's gradient with 1 and visit nodes from last to first.",
            "add passes its gradient to both parents; mul passes the gradient times the other parent's value.",
            "Always use += so repeated and shared parents add up, and validate ops, parent counts and indices up front without recursing.",
        ],
        "allowed_helpers": ["numpy"],
    },
    "0075": {
        "title": "Choose K-means landmarks",
        "concepts": ["training", "clustering", "seeded sampling"],
        "hints": [
            "Pick the first row uniformly with the given rng, then sample each later row by squared distance to its nearest chosen center.",
            "Keep a running nearest-squared-distance vector, zero the used rows, and update it with np.minimum after each pick.",
            "If every remaining weight is 0, choose uniformly among unused indices; divide by the max before normalizing to avoid overflow.",
        ],
        "allowed_helpers": ["numpy"],
    },
    "0100": {
        "title": "Who gets the last expert slot?",
        "concepts": ["training", "mixture of experts", "capacity"],
        "hints": [
            "Visit routes from highest score to lowest and admit each while its pool has budget left.",
            "Use np.argsort(-scores.ravel(), kind='stable') so ties go to the lower flattened position.",
            "For policy='rank', pool by owner[expert] with budget count*capacity, and validate dtypes, ranges and capacity with ValueError.",
        ],
        "allowed_helpers": ["numpy"],
    },
    "0094": {
        "title": "Fit binary probability calibration",
        "concepts": ["evaluation", "calibration", "logistic regression"],
        "hints": [
            "Platt fits sigmoid(a*score+b) on raw scores; isotonic fits a non-decreasing step function.",
            "Minimize the stable logistic loss plus 1e-3*(a²+b²)/2 with Newton steps; for isotonic, average tied scores and then pool adjacent violators.",
            "Interpolate isotonic levels with np.interp, which clamps outside the range, and keep constant or one-class inputs finite.",
        ],
        "allowed_helpers": ["numpy"],
    },
    "0080": {
        "title": "Three coasters, one scoring rule",
        "concepts": ["systems", "business rules", "validation"],
        "hints": [
            "Compute base = min(1, 1/bumps, 1/speed), then branch on the coaster type.",
            "Steel uses scale 2 with comfort min(1, 1/bumps, 5/speed); Suspended uses scale 0.5 and adds 0.5 comfort for Cable.",
            "Use true float division, and raise ValueError for unknown names or nonfinite or nonpositive values before scoring.",
        ],
        "allowed_helpers": ["numpy"],
    },
}

DEFAULT_CATALOG_IDS = [
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

LEARNING_PATH = ["0059", "0060", "mlp", "0069"]

def extract_cell(source: str, tag: str) -> str:
    return TaggedScript(source).cell_source(tag)


def _code(source: str, role: str | None = None, exercise_id: str = "0059") -> dict:
    metadata = {"dojo": {"exercise_id": exercise_id, "role": role}} if role else {}
    return {"cell_type": "code", "execution_count": None, "metadata": metadata,
            "outputs": [], "source": source.splitlines(keepends=True)}


def _drill(source: str, index: int, exercise_id: str) -> dict:
    cell = _code(source, "drill", exercise_id)
    cell["metadata"]["dojo"]["drill"] = index
    return cell


def _markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {"editable": False, "dojo": {"role": "instruction"}}, "source": source.splitlines(keepends=True)}


def _brief_inline(text: str, *, legacy: bool = True) -> str:
    """Format trusted teaching notation while escaping every source fragment.

    The fixed equation table and the list of bare identifiers are a fallback
    for scripts not yet written with `$…$` math and backticked code; marked
    scripts pass ``legacy=False``.
    """
    equations = {
        "Z=XW+b": r"Z = XW + b",
        "dW=X.T@dZ": r"\nabla_W L = X^{\mathsf{T}}\nabla_Z L",
        "db=dZ.sum(axis=0)": r"\nabla_b L = \sum_n (\nabla_Z L)_{n,:}",
        "log(sum(exp(q))) - q[y]": r"L_i = \log\!\sum_c e^{q_{ic}} - q_{i,y_i}",
        "(softmax(q)-one_hot)/N": r"\nabla_z L = \frac{\operatorname{softmax}(q)-\operatorname{onehot}(y)}{N}",
    }
    if not legacy:
        equations = {}
    math_pattern = "|".join(re.escape(value) for value in equations) or "(?!)"
    code_pattern = r"`[^`]+`|\b[A-Za-z_]\w*(?:\.\w+)+(?:\([^()]*\))?|\[[A-Za-z0-9_, :]+\]|\([A-Z](?:,[A-Z0-9]+)*,?\)"
    if legacy:
        code_pattern += r"|\b(?:ignore_index|keepdims|float64|logits|targets|learning_rate|new_W|new_b|mean_loss|dlogits)\b"
    pattern = re.compile(f"({math_pattern})|({code_pattern})")
    result, start = [], 0
    for match in pattern.finditer(text):
        result.append(html.escape(text[start:match.start()]))
        value = match.group()
        if value in equations:
            result.append(r'<span class="dojo-math">\(' + html.escape(equations[value]) + r'\)</span>')
        else:
            result.append("<code>" + html.escape(value.strip("`")) + "</code>")
        start = match.end()
    result.append(html.escape(text[start:]))
    return "".join(result)


def _exercise_brief(source: str, exercise_id: str) -> str:
    """Turn canonical exercise comments into a readable stage brief."""
    blocks: list[str] = []
    current: list[str] = []
    drill_lines = _drill_lines(source)
    for number, line in enumerate(source.splitlines(), start=1):
        if number in drill_lines:
            line = ""
        if line.startswith("# ") and not line.startswith("# %%"):
            if line.startswith(f"# {exercise_id} · "):
                continue
            current.append(line[2:].strip())
        elif current:
            blocks.append(" ".join(current))
            current = []
    if current:
        blocks.append(" ".join(current))
    introduction: list[str] = []
    stages: list[tuple[str, list[str]]] = []
    for block in blocks:
        if not block:
            continue
        if " — " in block:
            label, detail = block.split(" — ", 1)
            if label.lower().startswith(("baseline", "follow-up", "production")):
                if stages and label.lower().startswith("baseline") and stages[-1][0].lower().startswith("baseline"):
                    stages[-1][1].append(detail)
                else:
                    stages.append((label, [detail]))
                continue
        (stages[-1][1] if stages else introduction).append(block)
    rendered = [f"<p>{_brief_inline(block)}</p>" for block in introduction]
    for label, body in stages:
        content = "".join(f"<p>{_brief_inline(block)}</p>" for block in body)
        title = html.escape(label)
        if label.lower().startswith("baseline"):
            rendered.append(f'<section class="dojo-brief-stage"><h3>{title}</h3>{content}</section>')
        else:
            rendered.append(f'<details class="dojo-brief-stage"><summary>{title}</summary>{content}</details>')
    return "\n".join(rendered)


class BriefError(ValueError):
    """A question script's brief markers or math are malformed."""


BRIEF_MARKERS = {"Goal": "goal", "Predict": "predict", "Watch out": "watch-out", "Why": "why"}
MARKER_LIKE = re.compile(r"([A-Za-z][A-Za-z -]{0,14}):(?:\s+|$)(.*)")
# A misspelt marker (`goal:`, `Watchout:`, `Watch-out:`) is an error; ordinary prose like `Note:` is not.
NEAR_MARKERS = {re.sub(r"[^a-z]", "", name.lower()): name for name in BRIEF_MARKERS}
CODE_SPAN = re.compile(r"(`[^`]*`)")
STAGE_IDS = {"Baseline": "baseline", "Follow-up": "followup", "Production": "production"}
BRIEF_LABELS = {"goal": "Goal", "predict": "Predict", "watch-out": "Watch out", "why": "Why"}


def _unmask(text: str) -> str:
    return text.replace("\x00", "$").replace("\x01", "$")


def _brief_text(text: str, where: str) -> str:
    """Escape prose, typeset `$…$` and `$$…$$` as math, and set code in the code face."""
    # `\$` is a literal dollar, and dollars inside `code` are not math.
    spans = CODE_SPAN.split(text.replace("\\$", "\x00"))
    masked = "".join(span.replace("$", "\x01") if index % 2 else span for index, span in enumerate(spans))
    parts = masked.split("$$")
    if len(parts) % 2 == 0:
        raise BriefError(f"{where}: unbalanced $$")
    rendered: list[str] = []
    for index, part in enumerate(parts):
        if index % 2:
            rendered.append('</p><div class="dojo-math-display">\\[' + html.escape(_unmask(part.strip()), quote=False) + "\\]</div><p>")
            continue
        pieces = part.split("$")
        if len(pieces) % 2 == 0:
            raise BriefError(f"{where}: unbalanced $")
        for inner, piece in enumerate(pieces):
            if inner % 2:
                rendered.append('<span class="dojo-math">\\(' + html.escape(_unmask(piece), quote=False) + "\\)</span>")
            else:
                rendered.append(_brief_inline(_unmask(piece), legacy=False))
    return ("<p>" + "".join(rendered) + "</p>").replace("<p></p>", "").replace("<p> ", "<p>").replace(" </p>", "</p>")


def brief_sections(source: str, exercise_id: str, stages: list[str], script: str,
                   stage_ids: dict[str, str] = STAGE_IDS) -> list[dict] | None:
    r"""Typed brief sections from a question script's comment markers.

    A script opts in with a `# Goal:` line. Then `# Goal:`, `# Predict:`,
    `# Watch out:` and `# Why:` each open a section that runs until a blank
    comment line, code, or the next marker. Stage headings (`# Baseline — …`,
    or a lab's own `stage_ids` labels such as `# Forward — …`) scope the
    sections after them. Math is written as `$…$` or `$$…$$` within
    one line, with `\$` for a literal dollar. Scripts without a goal return None
    and keep the prose brief.
    """
    lines = source.splitlines()
    if not any(line.startswith("# Goal:") for line in lines):
        return None
    drill_lines = _drill_lines(source)
    stage_heading = re.compile("(%s)[\\w ]*? — (.*)" % "|".join(map(re.escape, stage_ids)))
    sections: list[dict] = []
    current: dict | None = None
    stage: str | None = None
    for number, line in enumerate(lines, start=1):
        where = f"{script}:{number}"
        if (number in drill_lines or not line.startswith("#") or line.startswith("# %%")
                or line.startswith(f"# {exercise_id} · ")):
            current = None
            continue
        text = line[1:].strip()
        if not text:
            current = None
            continue
        heading = stage_heading.fullmatch(text)
        marker = MARKER_LIKE.fullmatch(text)
        if heading:
            stage = stage_ids[heading.group(1)]
            if stage not in stages:
                raise BriefError(f"{where}: stage {stage!r} is not one of this workout's stages")
            current = {"kind": "prose", "stage": stage, "lines": [(heading.group(2), where)]}
            sections.append(current)
        elif marker and marker.group(1) in BRIEF_MARKERS:
            kind = BRIEF_MARKERS[marker.group(1)]
            current = {"kind": kind, "stage": stage, "lines": [(marker.group(2), where)] if marker.group(2) else []}
            sections.append(current)
        elif marker and re.sub(r"[^a-z]", "", marker.group(1).lower()) in NEAR_MARKERS:
            intended = NEAR_MARKERS[re.sub(r"[^a-z]", "", marker.group(1).lower())]
            raise BriefError(f"{where}: unknown marker {marker.group(1)!r}; did you mean {intended!r}?")
        elif current is not None:
            current["lines"].append((text, where))
        else:
            current = {"kind": "prose", "stage": stage, "lines": [(text, where)]}
            sections.append(current)
    result = []
    for section in sections:
        rendered = []
        for text, where in section["lines"]:
            rendered.append(_brief_text(text, where))
        # Join the paragraph's lines into one paragraph unless a display equation splits it.
        html_text = "".join(rendered).replace("</p><p>", " ")
        entry = {"kind": section["kind"], "html": html_text}
        if section["stage"]:
            entry["stage"] = section["stage"]
        result.append(entry)
    return result


DRILL_REASONING = re.compile(r"# Drill (\d+):\s*(.*)")


def _drill_lines(source: str) -> set[int]:
    """1-based line numbers inside `# %% [drill]` cells, which never join the brief."""
    return {number for tag, start, end in _cells(source) if tag == "drill" for number in range(start, end)}


def drills(question: str, eval_source: str, question_name: str, eval_name: str) -> list[dict]:
    """Drill cards from `# %% [drill]` cells, each paired with `# Drill <n>:` reasoning in the eval script.

    A drill cell opens with comment lines: the first is its title and the rest
    its prompt. Its code is the probe the learner runs. The reasoning lives in
    the eval script so the learner's script doesn't give the answer away.
    """
    lines = question.splitlines()
    cells = [(start, end) for tag, start, end in _cells(question) if tag == "drill"]
    reasoning: dict[int, tuple[list[str], str]] = {}
    current: list[str] | None = None
    for number, line in enumerate(eval_source.splitlines(), start=1):
        match = DRILL_REASONING.fullmatch(line.rstrip())
        if match:
            index = int(match.group(1))
            if index in reasoning:
                raise BriefError(f"{eval_name}:{number}: duplicate '# Drill {index}:' reasoning")
            if not 1 <= index <= len(cells):
                raise BriefError(f"{eval_name}:{number}: '# Drill {index}:' has no matching drill in {question_name}")
            current = [match.group(2)] if match.group(2) else []
            reasoning[index] = (current, f"{eval_name}:{number}")
        elif current is not None and line.startswith("#") and line[1:].strip() and not line.startswith("# %%"):
            current.append(line[1:].strip())
        else:
            current = None
    result = []
    for index, (start, end) in enumerate(cells, start=1):
        if index not in reasoning:
            raise BriefError(f"{question_name}:{start}: drill {index} has no '# Drill {index}:' reasoning in {eval_name}")
        body = lines[start:end - 1]
        comments: list[str] = []
        while body and body[0].startswith("#"):
            comments.append(body.pop(0)[1:].strip())
        if not comments:
            raise BriefError(f"{question_name}:{start}: drill {index} needs a title comment")
        where = f"{question_name}:{start}"
        explanation, explained_at = reasoning[index]
        result.append({
            "title": comments[0],
            "prompt_html": _brief_text(" ".join(comments[1:]), where) if comments[1:] else "",
            "probe_source": "\n".join(body).strip() + "\n",
            "reasoning_html": _brief_text(" ".join(explanation), explained_at),
        })
    return result


def render_brief(sections: list[dict], stage_ids: dict[str, str] = STAGE_IDS) -> str:
    """The brief cell's HTML: the goal set apart first, then each stage's contract."""
    def block(section: dict) -> str:
        kind, body = section["kind"], section["html"]
        if kind == "prose":
            return body
        if kind == "why":
            return f'<details class="dojo-brief-why"><summary>Why</summary>{body}</details>'
        return f'<div class="dojo-brief-{kind}"><h4>{BRIEF_LABELS[kind]}</h4>{body}</div>'

    intro = [section for section in sections if not section.get("stage")]
    ordered = [section for section in intro if section["kind"] == "goal"] + \
              [section for section in intro if section["kind"] != "goal"]
    rendered = [block(section) for section in ordered]
    for label, stage in stage_ids.items():
        own = [section for section in sections if section.get("stage") == stage]
        if own:
            # The notebook's sanitizer keeps classes but drops data attributes.
            rendered.append(f'<section class="dojo-brief-stage dojo-stage-{stage}">'
                            f"<h3>{label}</h3>{''.join(block(section) for section in own)}</section>")
    return "\n".join(rendered)


def _lesson_cells(exercise_id: str, out: Path, base_url: str) -> list[dict]:
    path = Path(__file__).resolve().parent / "lessons" / f"{exercise_id}.py"
    if not path.exists():
        return []
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location(f"browser_gym_lesson_{exercise_id}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load lesson template: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cells = module.cells(exercise_id)
    for index, cell in enumerate(cells):
        dojo = cell.get("metadata", {}).get("dojo", {})
        if dojo.get("exercise_id") != exercise_id:
            raise ValueError(f"{exercise_id}: lesson cell {index} has wrong exercise metadata")
        if dojo.get("role") == "solution":
            raise ValueError(f"{exercise_id}: lesson cell {index} must not be a solution cell")
        if cell["cell_type"] == "code" and dojo.get("role") == "exploration":
            lesson_source = "".join(cell["source"])
            lesson_asset = f"{exercise_id}-{index}.py"
            lesson_path = out / "lessons" / lesson_asset
            lesson_path.parent.mkdir(parents=True, exist_ok=True)
            lesson_path.write_text(lesson_source)
            version = hashlib.sha256(lesson_source.encode()).hexdigest()[:12]
            loader = (
                f"# Run the {exercise_id} visual lab.\n"
                "from pyodide.http import pyfetch\n"
                f"exec(await (await pyfetch({base_url + 'gym-assets/lessons/' + lesson_asset + '?v=' + version!r})).string(), globals())\n"
            )
            cell["source"] = loader.splitlines(keepends=True)
    return cells


def _bundle(eval_source: str, filename: str = "eval.py") -> dict:
    program = TaggedScript(eval_source, filename).verifiers()
    verifiers = [node for node in program.nodes
                 if isinstance(node, ast.FunctionDef) and node.name in program.verifier_names]
    helpers = [node for node in program.nodes if node not in verifiers
               and not isinstance(node, (ast.Import, ast.ImportFrom))]
    imports = [node for node in program.nodes if isinstance(node, (ast.Import, ast.ImportFrom))]
    return {"candidate": program.candidate,
            "program": program.source(),
            "imports": "\n".join(ast.unparse(node) for node in imports),
            "helpers": "\n\n".join(ast.unparse(node) for node in helpers),
            "verifiers": [{"name": node.name, "source": ast.unparse(node)} for node in verifiers]}


def _reference_solution(eval_source: str, candidate: str) -> str:
    """The eval's reference answer: the candidate (a function or a class), every
    helper defined beside it before the verifiers, and the imports they use."""
    tree = ast.parse(eval_source)
    verifiers_start = next(cell[1] for cell in _cells(eval_source) if cell[0] == "verifiers")
    definitions = [node for node in tree.body if node.lineno < verifiers_start
                   and isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign))]
    if not any(isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == candidate for node in definitions):
        return ""
    used_names = {node.id for definition in definitions for node in ast.walk(definition) if isinstance(node, ast.Name)}

    def import_is_used(node: ast.Import | ast.ImportFrom) -> bool:
        for alias in node.names:
            local_name = alias.asname or alias.name.split(".")[0]
            if local_name in used_names:
                return True
        return False

    imports = [ast.get_source_segment(eval_source, node) for node in tree.body
               if isinstance(node, (ast.Import, ast.ImportFrom))
               and node.lineno < verifiers_start
               and import_is_used(node)]
    body = "\n\n\n".join(ast.get_source_segment(eval_source, node).strip() for node in definitions)
    return "\n".join(imports) + "\n\n" + body + "\n"

def _build_dojo_exercise(root: Path, out: Path, exercise_id: str, base_url: str) -> dict:
    metadata = QUALIFIED_DOJO_EXERCISES.get(exercise_id)
    if metadata is None:
        raise ValueError(f"{exercise_id}: browser lesson is not yet qualified")
    questions = list((root / "dojo").glob(f"{exercise_id}_*.py"))
    questions = [path for path in questions if not path.stem.endswith("_eval")]
    if len(questions) != 1:
        raise ValueError(f"{exercise_id}: expected one question script")
    question = questions[0]
    eval_path = question.with_name(question.stem + "_eval.py")
    question_source = question.read_text()
    eval_source = eval_path.read_text()
    solution = TaggedScript(question_source, question).answer().source()
    bundle = _bundle(eval_source, eval_path.name)
    reference_solution = _reference_solution(eval_source, bundle["candidate"])
    bundle["reference_source"] = reference_solution
    bundle["exercise_id"] = exercise_id
    bundle["checker_version"] = hashlib.sha256(eval_source.encode()).hexdigest()[:12]
    title = str(metadata["title"])
    stages = list(metadata["stages"]) if isinstance(metadata.get("stages"), list) else ["baseline", "followup", "production"]
    sections = brief_sections(question_source, exercise_id, stages, question.name)
    drill_cards = drills(question_source, eval_source, question.name, eval_path.name)
    instructions = render_brief(sections) if sections else _exercise_brief(question_source, exercise_id)
    hints = list(metadata["hints"]) if isinstance(metadata.get("hints"), list) else []
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python (Pyodide)", "language": "python", "name": "python"},
                             "dojo": {"exercise_id": exercise_id}},
                "cells": [
                    _markdown(f"# {exercise_id} · {metadata.get('heading', metadata['title'])}\n\n{instructions}\n"),
                    _markdown("### Explore\nRun the short cell below to see tensor shapes and interactive plots.\n"),
                    *_lesson_cells(exercise_id, out, base_url),
                    *[_drill(drill["probe_source"], index, exercise_id) for index, drill in enumerate(drill_cards)],
                    _markdown("Edit the solution below. Only this marked cell is sent to Check. "
                              "Run it with the notebook toolbar to explore normally.\n"),
                    _code(solution, "solution", exercise_id),
                    _markdown("Scratch space: try your own example. This cell is never checked.\n"),
                    _code("# Try a small example here.\n"),
                ]}
    notebook_path = f"notebooks/{question.stem}.ipynb"
    checker_path = f"checkers/{question.stem}.json"
    for relative, data in ((notebook_path, notebook), (checker_path, bundle)):
        path = out / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2) + "\n")
    return {"id": exercise_id, "title": title,
            "notebook": notebook_path, "checker": checker_path,
            "checker_version": bundle["checker_version"],
            "stages": stages,
            "stage_order": "cumulative",
            "hints": hints,
            "allowed_helpers": metadata["allowed_helpers"],
            "concepts": metadata.get("concepts", []),
            "solution": reference_solution,
            **({"brief": sections} if sections else {}),
            **({"drills": drill_cards} if drill_cards else {})}


def _build_mlp_lab(root: Path, out: Path, base_url: str) -> dict:
    lab = root / "browser_gym" / "labs" / "mlp"
    checkpoints = ["init", "forward", "backward", "train", "inference", "evaluate", "state", "diagnose"]
    candidate_functions = [
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
    sources = {
        "browser_gym.labs.__init__": (root / "browser_gym" / "labs" / "__init__.py").read_text(),
        "browser_gym.labs.mlp.__init__": (lab / "__init__.py").read_text(),
        "browser_gym.labs.mlp.fixtures": (lab / "fixtures.py").read_text(),
        "browser_gym.labs.mlp.reference": (lab / "reference.py").read_text(),
        "browser_gym.labs.mlp.checks": (lab / "checks.py").read_text(),
    }
    starter_source = (lab / "starter.py").read_text()
    # The lab's brief and drills follow the question-script markers, with a
    # heading per checkpoint. Drill reasoning has its own file, outside the checker bundle.
    checkpoint_ids = {checkpoint.capitalize(): checkpoint for checkpoint in checkpoints}
    brief_source = (lab / "brief.py").read_text()
    sections = brief_sections(brief_source, "mlp", checkpoints, "labs/mlp/brief.py", checkpoint_ids)
    if not sections:
        raise BriefError("labs/mlp/brief.py: the lab brief needs a '# Goal:' line")
    drill_cards = drills(brief_source, (lab / "reasoning.py").read_text(), "labs/mlp/brief.py", "labs/mlp/reasoning.py")
    checksum_material = "\n".join([starter_source, *sources.values()])
    bundle = {
        "kind": "mlp_lab",
        "exercise_id": "mlp",
        "checker_version": hashlib.sha256(checksum_material.encode()).hexdigest()[:12],
        "fixture_id": "mlp2d-complete-v1",
        "checkpoints": checkpoints,
        "candidate_functions": candidate_functions,
        "modules": sources,
    }
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python (Pyodide)", "language": "python", "name": "python"},
                             "dojo": {"exercise_id": "mlp"}},
                "cells": [
                    _markdown(f"# MLP Modeling Lab\n\n{render_brief(sections, checkpoint_ids)}\n"),
                    _markdown("### Explore\nRun the short cell below to see the MLP visuals.\n"),
                    *_lesson_cells("mlp", out, base_url),
                    *[_drill(drill["probe_source"], index, "mlp") for index, drill in enumerate(drill_cards)],
                    _markdown("Edit the solution below. Only this marked cell is sent to Check. "
                              "Use the stage menu to check each MLP checkpoint independently.\n"),
                    _code(starter_source, "solution", "mlp"),
                    _markdown("Scratch space: notebook variables here are useful for exploration but are not submitted.\n"),
                    _code("# Try a shape or gradient experiment here.\n"),
                ]}
    notebook_path = "notebooks/mlp_modeling_lab.ipynb"
    checker_path = "checkers/mlp_modeling_lab.json"
    for relative, data in ((notebook_path, notebook), (checker_path, bundle)):
        path = out / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2) + "\n")
    return {"id": "mlp", "title": "MLP Modeling Lab",
            "notebook": notebook_path, "checker": checker_path,
            "checker_version": bundle["checker_version"],
            "stages": bundle["checkpoints"],
            "stage_order": "independent",
            "reference_components": candidate_functions,
            "concepts": ["training", "gradients", "confidence"],
            "brief": sections,
            "drills": drill_cards}


def _attach_learning_path(entries: list[dict]) -> None:
    by_id = {entry["id"]: entry for entry in entries}
    path_entries = [
        {
            "id": item_id,
            "title": by_id[item_id]["title"],
            "concepts": by_id[item_id].get("concepts", []),
        }
        for item_id in LEARNING_PATH
        if item_id in by_id
    ]
    for entry in entries:
        if entry["id"] not in LEARNING_PATH:
            continue
        index = LEARNING_PATH.index(entry["id"])
        previous_id = LEARNING_PATH[index - 1] if index > 0 else None
        next_id = LEARNING_PATH[index + 1] if index + 1 < len(LEARNING_PATH) else None
        if previous_id in by_id:
            entry["previous"] = {"id": previous_id, "title": by_id[previous_id]["title"]}
        if next_id in by_id:
            entry["next"] = {"id": next_id, "title": by_id[next_id]["title"]}
        entry["learning_path"] = path_entries


def build(root: Path, out: Path, ids: list[str], *, base_url: str = "/") -> list[dict]:
    if not re.fullmatch(r"/(?:[A-Za-z0-9_-]+/)*", base_url):
        raise ValueError("base_url must be an absolute directory path with a trailing slash")
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    for exercise_id in ids:
        if exercise_id == "mlp":
            entries.append(_build_mlp_lab(root, out, base_url))
            continue
        if exercise_id not in QUALIFIED_DOJO_EXERCISES:
            raise ValueError(f"{exercise_id}: browser lesson is not yet qualified")
        entries.append(_build_dojo_exercise(root, out, exercise_id, base_url))
    _attach_learning_path(entries)
    (out / "motifs.json").write_text(json.dumps(motifs(), indent=2) + "\n")
    (out / "catalog.json").write_text(json.dumps(entries, indent=2) + "\n")
    return entries


if __name__ == "__main__":
    build(Path(__file__).resolve().parent.parent, Path(__file__).resolve().parent / "build", DEFAULT_CATALOG_IDS,
          base_url=os.environ.get("DOJO_BASE_URL", "/"))
