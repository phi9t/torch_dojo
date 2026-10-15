"""Reference and cumulative checks for a tiny reverse-mode scalar graph."""

# %%
import math

# Drill 1: Assigning prints `= [3.0, 1.0, 1.0]`; accumulating prints
# `+= [7.0, 1.0, 1.0]`. $x$ reaches the loss three ways: directly through the
# add (1) and twice through `x*x` (3 each), so $\partial L / \partial x = 1 + 3 + 3 = 7$.
# With `=` each later write throws away the earlier contributions, including
# the add's 1 and the other half of the multiply.

# Drill 2: Reverse prints `[15.0, 6.0, 3.0, 1.0, 1.0]`; forward prints
# `[0.0, 0.0, 1.0, 1.0, 1.0]`. Walking forward visits node 2 while its
# gradient is still 0, before nodes 3 and 4 have sent it anything, so nothing
# reaches the inputs. In reverse topological order every consumer runs before
# its parents, so a shared node has collected all its contributions first.

# %%
def backward_graph(nodes: list[tuple[str, tuple[int, ...], float]]) -> list[float]:
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("need a nonempty list of nodes")
    for index, node in enumerate(nodes):
        if not isinstance(node, tuple) or len(node) != 3:
            raise ValueError("each node must be (op, parents, value)")
        op, parents, value = node
        count = 0 if op == "input" else 2 if op in ("add", "mul") else -1
        if (count < 0 or not isinstance(parents, tuple) or len(parents) != count
                or any(type(parent) is not int or parent < 0 or parent >= index for parent in parents)
                or isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value)):
            raise ValueError("invalid node topology or value")
    gradients = [0.] * len(nodes)
    gradients[-1] = 1.
    for index in range(len(nodes)-1, -1, -1):
        op, parents, _ = nodes[index]
        incoming = gradients[index]
        if op == "add":
            left, right = parents
            gradients[left] += incoming
            gradients[right] += incoming
        elif op == "mul":
            left, right = parents
            gradients[left] += incoming * nodes[right][2]
            gradients[right] += incoming * nodes[left][2]
    return gradients


# %% [verifiers]
def verify_baseline_repeated_parent(backward_graph):
    nodes = [("input", (), 3.), ("mul", (0, 0), 9.), ("add", (1, 0), 12.)]
    assert backward_graph(nodes) == [7., 1., 1.]


def verify_followup_shared_intermediate(backward_graph):
    nodes = [("input", (), 2.), ("input", (), 3.), ("mul", (0, 1), 6.),
             ("mul", (2, 0), 12.), ("add", (2, 3), 18.)]
    assert backward_graph(nodes) == [15., 6., 3., 1., 1.]


def verify_followup_disconnected_and_single_node(backward_graph):
    assert backward_graph([("input", (), 4.), ("input", (), 8.), ("mul", (0, 0), 16.)]) == [8., 0., 1.]
    assert backward_graph([("input", (), -2.)]) == [1.]


def verify_production_invalid_graphs(backward_graph):
    invalid = ([], [("divide", (), 1.)], [("input", (0,), 1.)],
               [("input", (), 1.), ("add", (0,), 2.)],
               [("input", (), 1.), ("mul", (0, 2), 2.)],
               [("input", (), 1.), ("add", (0, -1), 2.)],
               [("input", (), float("nan"))])
    for graph in invalid:
        try:
            backward_graph(graph)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid graph {graph}")


def verify_production_deep_graph_is_iterative(backward_graph):
    nodes = [("input", (), 2.)]
    nodes.extend(("add", (i - 1, 0), float(i + 2)) for i in range(1, 3000))
    before = list(nodes)
    gradients = backward_graph(nodes)
    assert len(gradients) == len(nodes)
    assert gradients[-1] == 1. and gradients[0] == 3000.
    assert nodes == before


# %%
from torch_judge import evaluate
if not evaluate("0050_backward_graph_eval.py", backward_graph):
    raise SystemExit(1)
