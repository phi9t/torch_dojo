"""Reference and staged checks for four-connected boolean grid components."""

# %%
import numpy as np
from hypothesis import given, settings, strategies as st

# Drill 1: It prints `4-connected: 5  8-connected: 1`. The five cells of the X
# share only corners, so with up/down/left/right steps each is its own
# component; adding the four diagonal steps chains them all through the
# centre. The contract is 4-connectivity, so the neighbour list must have
# exactly four offsets.

# Drill 2: It prints `pushes, mark on push: 9  mark on pop: 13`. Marking on
# push guarantees each cell enters the stack once, so pushes equal the 9 cells.
# Marking only on pop lets a cell be pushed again by every neighbour that sees
# it before it is popped; the duplicates are skipped later but still cost stack
# space and time, and on large grids that blow-up breaks the $O(HW)$ bound.

# Drill 3: It prints `1000 3000`. A recursive flood fill on a 3000-cell chain
# nests one call per cell, three times Python's default limit, so it raises
# `RecursionError`; an explicit stack has no such limit.

# %%
def count_grid_components(grid: np.ndarray) -> int:
    grid = np.asarray(grid)
    if grid.ndim != 2 or grid.dtype != np.bool_:
        raise ValueError("grid must be 2D bool")
    height, width = grid.shape
    seen = np.zeros_like(grid)
    components = 0
    for row in range(height):
        for col in range(width):
            if not grid[row, col] or seen[row, col]:
                continue
            components += 1
            stack = [(row, col)]
            seen[row, col] = True
            while stack:
                r, c = stack.pop()
                for nr, nc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                    if 0 <= nr < height and 0 <= nc < width and grid[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
    return components


# %% [verifiers]
def verify_baseline_diagonal_and_cross(count_grid_components):
    assert count_grid_components(np.array([[1, 0], [0, 1]], dtype=bool)) == 2
    assert count_grid_components(np.array([[0, 1, 0], [1, 1, 1]], dtype=bool)) == 1


def verify_baseline_empty_and_hole(count_grid_components):
    ring = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool)
    assert count_grid_components(ring) == 1
    assert count_grid_components(np.zeros((0, 3), bool)) == 0


def verify_followup_long_chain_is_iterative(count_grid_components):
    assert count_grid_components(np.ones((1, 3000), bool)) == 1
    assert count_grid_components(np.ones((3000, 1), bool)) == 1


def verify_followup_invalid_shape_and_dtype(count_grid_components):
    for grid in (np.array([[1]]), np.array([True]), np.zeros((1, 1, 1), bool)):
        try:
            count_grid_components(grid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid grid accepted")


def verify_production_checkerboard_and_immutability(count_grid_components):
    grid = (np.indices((31, 35)).sum(axis=0) % 2 == 0)
    before = grid.copy()
    assert count_grid_components(grid) == int(grid.sum())
    assert np.array_equal(grid, before)


def verify_production_generated_transpose_invariance(count_grid_components):
    @settings(max_examples=50, deadline=None, derandomize=True)
    @given(st.integers(0, 8), st.integers(0, 8), st.data())
    def property_check(height, width, data):
        cells = data.draw(st.lists(st.booleans(), min_size=height*width, max_size=height*width))
        grid = np.asarray(cells, dtype=bool).reshape(height, width)
        assert count_grid_components(grid) == count_grid_components(grid.T)
    property_check()


# %%
from torch_judge import evaluate
if not evaluate("0071_grid_components_eval.py", count_grid_components):
    raise SystemExit(1)
