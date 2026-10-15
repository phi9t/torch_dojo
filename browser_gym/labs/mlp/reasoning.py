# Reasoning for the drill cards in brief.py, revealed on request in the
# browser. It stays out of the learner's starter, and out of the checker
# bundle so the lab's checker version doesn't change with it.

# Drill 1: The loss is a mean over the batch, so each row's gradient carries $1/B$ and `db2` is the second line, `[0.075 0.125 -0.2]`.
# The first line is $B = 4$ times too large. That missing division is exactly the defect planted for the Diagnose checkpoint.

# Drill 2: ReLU passes gradient only where its input is positive. This lab uses `hidden_pre > 0.0`, so an input of exactly 0 counts as inactive and gets none: `[0. 0. 1.]`.

# Drill 3: Velocity accumulates. After the first step $v = 1$ and the parameter is $0.9$; then $v = 0.9 \cdot 1 + 1 = 1.9$, so the second step moves $0.19$, to $0.71$.
# That carried velocity is why the State checkpoint must save and restore it: dropping it changes every later step.
