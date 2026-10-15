# %%
# 0050 · Reverse-mode differentiation through a tiny scalar graph
# Goal: Return the gradient of the last node with respect to every node in a
# scalar graph, walking it backward once.
#
# Adapted from ferric_continuum/tnsr's backward graph traversal.
# Node=(op, parent indices, forward value); parents precede their consumer.

# %%
# Worked probe: $x = 3$, square $= x \cdot x$, loss $=$ square $+ x$.
# Predict: What is $\partial \text{loss} / \partial x$?
nodes = [("input", (), 3.), ("mul", (0, 0), 9.), ("add", (1, 0), 12.)]
print("values:", [node[2] for node in nodes])

# %% [drill]
# Assign or accumulate?
# Backward through `x*x` then `square + x`. Predict the gradients when each parent's slot is assigned and when it accumulates.
nodes = [("input", (), 3.), ("mul", (0, 0), 9.), ("add", (1, 0), 12.)]
for accumulate in (False, True):
    grads = [0., 0., 1.]
    for i in (2, 1):
        op, (left, right), _ = nodes[i]
        dl = 1. if op == "add" else nodes[right][2]
        dr = 1. if op == "add" else nodes[left][2]
        if accumulate:
            grads[left] += grads[i] * dl
            grads[right] += grads[i] * dr
        else:
            grads[left] = grads[i] * dl
            grads[right] = grads[i] * dr
    print("+=" if accumulate else "=", grads)

# %% [drill]
# Which way to walk?
# Node 2 feeds both node 3 and node 4. Predict the gradients from walking the nodes in reverse and in forward order.
shared = [("input", (), 2.), ("input", (), 3.), ("mul", (0, 1), 6.),
          ("mul", (2, 0), 12.), ("add", (2, 3), 18.)]
def run(order):
    grads = [0.] * 5
    grads[-1] = 1.
    for i in order:
        op, parents, _ = shared[i]
        if op == "add":
            grads[parents[0]] += grads[i]; grads[parents[1]] += grads[i]
        elif op == "mul":
            l, r = parents
            grads[l] += grads[i] * shared[r][2]; grads[r] += grads[i] * shared[l][2]
    return grads
print("reverse:", run([4, 3, 2, 1, 0]))
print("forward:", run([0, 1, 2, 3, 4]))

# %% [answer]
def backward_graph(nodes: list[tuple[str, tuple[int, ...], float]]) -> list[float]:
    """Return d(last node)/d(each node), including disconnected nodes."""
    pass

# %%
# Baseline — input/add/mul nodes, valid topological order, last node scalar
# loss. Seed its gradient with 1 and walk backward. The chain rule sends
# $\bar a \mathrel{+}= \bar c \, b$ and $\bar b \mathrel{+}= \bar c \, a$ through $c = a b$.
# Watch out: use `+=` for each parent: the same parent may appear twice in one multiply.
from torch_judge import check
check("0050_backward_graph_eval.py", stage="baseline")

# %%
# Follow-up — shared intermediate results must collect all downstream votes
# before their local backward step. Disconnected inputs have zero gradient;
# a single input serving as loss has gradient 1.
check("0050_backward_graph_eval.py", stage="followup")

# %%
# Production — reject empty graphs, unknown ops, wrong parent counts,
# forward/out-of-range parent indices, and non-finite values with `ValueError`.
# Do not recurse: a long graph should use $O(\text{nodes} + \text{edges})$ time and
# $O(\text{nodes})$ extra memory. Do not mutate the supplied nodes.
check("0050_backward_graph_eval.py", stage="production")
