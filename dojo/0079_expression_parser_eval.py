"""Reference and verifiers for +/* expression scanning."""

# %%
import math
from hypothesis import given, settings, strategies as st

# Drill 1: It prints `25 19`. Folding left to right computes
# $((2 + 3) \cdot 4) + 5 = 25$, letting `+` act before `*`. Precedence means
# $3 \cdot 4$ must finish as one term before it joins the sum.

# Drill 2: It prints `(0, 2)`, `(0, 6)`, `(0, 24)`, `(24, 5)` and then `29`.
# `*` keeps growing the running term; `+` moves the finished term into the
# total and starts a new one. The last term is still pending when input
# ends, so the answer is `total + term`, not `total`.

# Drill 3: It prints `2 True 1000` and then `False`. `int` accepts
# Arabic-Indic digits and underscores, and `isdigit` agrees about the
# Arabic-Indic digit, but this grammar allows only ASCII digits. Comparing each character with `'0'` and `'9'`
# rejects them before `int` is called.

# %%
def evaluate_expression(expression: str) -> int:
    if not isinstance(expression, str):
        raise ValueError("expression must be text")
    index = 0
    length = len(expression)

    def number() -> int:
        nonlocal index
        while index < length and expression[index].isspace():
            index += 1
        start = index
        while index < length and "0" <= expression[index] <= "9":
            index += 1
        if index == start:
            raise ValueError("expected a nonnegative integer")
        return int(expression[start:index])

    term = number()
    total = 0
    while True:
        while index < length and expression[index].isspace():
            index += 1
        if index == length:
            return total + term
        operator = expression[index]
        if operator not in "+*":
            raise ValueError("expected + or *")
        index += 1
        value = number()
        if operator == "*":
            term *= value
        else:
            total += term
            term = value


# %% [verifiers]
def verify_baseline_precedence(evaluate_expression):
    assert evaluate_expression("2+3*4+5") == 19
    assert evaluate_expression("2*3*4+5") == 29


def verify_followup_spaces_zero_and_single_number(evaluate_expression):
    assert evaluate_expression(" 0 * 500 + 007 ") == 7
    assert evaluate_expression("12 * 3 + 40") == 76
    assert evaluate_expression("12345") == 12345


def verify_production_reject_malformed_grammar(evaluate_expression):
    for text in ("", " ", "2++3", "2*", "+2", "2 3", "2/1", "2-1", "2+(3)"):
        try:
            evaluate_expression(text)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid expression {text!r}")


def verify_followup_large_integer_without_float_rounding(evaluate_expression):
    n = 10 ** 40
    assert evaluate_expression(f"{n}*{n}+1") == n * n + 1


def verify_production_generated_sum_of_products(evaluate_expression):
    term = st.lists(st.integers(0, 100), min_size=1, max_size=4)

    @settings(max_examples=35, deadline=None, derandomize=True)
    @given(st.lists(term, min_size=1, max_size=5))
    def property_check(terms):
        text = " + ".join(" * ".join(map(str, factors)) for factors in terms)
        expected = sum(math.prod(factors) for factors in terms)
        assert evaluate_expression(text) == expected

    property_check()



def verify_production_reject_nontext_and_unicode_digits(evaluate_expression):
    for value in (None, 2, ['2+3'], '٢+٣'):
        try:
            evaluate_expression(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f'accepted invalid expression {value!r}')

# %%
from torch_judge import evaluate
if not evaluate("0079_expression_parser_eval.py", evaluate_expression):
    raise SystemExit(1)
