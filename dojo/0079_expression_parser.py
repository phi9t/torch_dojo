# %%
# 0079 · A pocket calculator with exactly two operators
# Goal: Evaluate an integer expression that uses only `+` and `*`, with the
# usual precedence, in a single left-to-right scan. RX02 adapts a visible
# +/* prompt. Grammar and error policy are authored. Python integers are
# unbounded except for runtime conversion limits.

# %%
from torch_judge import check

# %%
# Predict: Multiplication binds before addition; what does `2 + 3 * 4 + 5` print?
print('2 + 3 * 4 + 5 =', 2 + 3 * 4 + 5)
# Why: A running product completes a term before the term joins the sum.

# %%
# Grammar: `expression := integer ((+ | *) integer)*`, with optional
# whitespace between tokens. Integer := one or more ASCII digits.
# Watch out: empty input, missing operand, adjacent operators, and all other
# characters raise `ValueError`. Do not use `eval` or float arithmetic.

# %% [drill]
# Left to right?
# Predict what a plain left-to-right fold gives for `2 + 3 * 4 + 5`, next to the true value.
tokens = ['2', '+', '3', '*', '4', '+', '5']
acc = int(tokens[0])
for op, value in zip(tokens[1::2], tokens[2::2]):
    acc = acc + int(value) if op == '+' else acc * int(value)
print(acc, 2 + 3 * 4 + 5)

# %% [drill]
# Total and running term
# Predict `(total, term)` after each operand of `2 * 3 * 4 + 5`, and the final answer.
tokens = ['2', '*', '3', '*', '4', '+', '5']
total, term = 0, int(tokens[0])
print((total, term))
for op, value in zip(tokens[1::2], tokens[2::2]):
    if op == '*':
        term *= int(value)
    else:
        total, term = total + term, int(value)
    print((total, term))
print(total + term)

# %% [drill]
# Is int strict enough?
# Predict what `int` and `str.isdigit` say about Arabic-Indic digits and an underscore.
print(int('٢'), '٢'.isdigit(), int('1_000'))
print('0' <= '٢' <= '9')

# %% [answer]
def evaluate_expression(expression: str) -> int:
    pass

# %%
# Baseline — Valid single-digit sums/products with precedence.
check('0079_expression_parser_eval.py', stage='baseline')

# %%
# Follow-up — Multidigit operands, whitespace, zero, and large integers.
# Keep total/current-product state instead of building a syntax tree.
check('0079_expression_parser_eval.py', stage='followup')

# %%
# Production — Reject bad grammar and non-string inputs. Parentheses,
# division, signs, and implicit multiplication are outside this grammar.
check('0079_expression_parser_eval.py', stage='production')
