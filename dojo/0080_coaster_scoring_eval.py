"""Python adaptation of the three-type coaster scoring rules."""

# %%
import math

# Drill 1: It prints `0.25 0`. Python's `/` is true division, so
# $\min(1, 1/2, 5/20) = 0.25$. Floor division, like C++ integer division,
# truncates $1/2$ and $5/20$ to 0, so every comfort below 1 collapses to 0 and
# the score with it. Write the ratios with float literals such as `1. / speed`.

# Drill 2: With $5/\text{speed}$ every line prints a score of `10.0`; with the
# base $1/\text{speed}$ every line prints `2.0`. Once speed exceeds 5,
# $2 \cdot (5/\text{speed}) \cdot \text{speed} = 10$, so Steel scores are capped
# at 10 no matter how fast. Reusing Wooden's base comfort silently caps them at 2.

# %%
def coaster_score(coaster_type: str, speed: float, bumps_per_second: float, lift_type: str) -> float:
    if coaster_type not in {"Wooden", "Steel", "Suspended"} or lift_type not in {"Cable", "Chain", "Launched"}:
        raise ValueError("unknown coaster or lift")
    if not math.isfinite(speed) or not math.isfinite(bumps_per_second) or speed <= 0 or bumps_per_second <= 0:
        raise ValueError("speed and bump rate must be finite and positive")
    base = min(1., 1. / bumps_per_second, 1. / speed)
    if coaster_type == "Wooden":
        scale, comfort = 1., base
    elif coaster_type == "Steel":
        scale, comfort = 2., min(1., 1. / bumps_per_second, 5. / speed)
    else:
        scale = 0.5
        comfort = base + (0.5 if lift_type == "Cable" else 0.)
    return scale * comfort * speed


# %% [verifiers]
def verify_baseline_four_reported_examples(coaster_score):
    examples = [
        (("Wooden", 4., 1., "Chain"), 1.),
        (("Steel", 20., 2., "Cable"), 10.),
        (("Suspended", 2., 1.5, "Cable"), 1.),
        (("Suspended", 2., 1.5, "Chain"), 0.5),
    ]
    for args, expected in examples:
        assert math.isclose(coaster_score(*args), expected, rel_tol=1e-12)


def verify_followup_steel_speed_cap_and_lift_difference(coaster_score):
    assert math.isclose(coaster_score("Steel", 100., 1., "Chain"), 10.)
    cable = coaster_score("Suspended", 4., 1., "Cable")
    chain = coaster_score("Suspended", 4., 1., "Chain")
    assert math.isclose(cable - chain, 1.)


def verify_production_reject_unknown_names_and_nonpositive_values(coaster_score):
    bad = [("Other", 1., 1., "Chain"), ("Steel", 1., 1., "Other"),
           ("Wooden", 0., 1., "Cable"), ("Wooden", 1., -1., "Cable"),
           ("Wooden", float("inf"), 1., "Cable")]
    for args in bad:
        try:
            coaster_score(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid coaster input accepted: {args}")



def verify_production_nonfinite_and_no_shared_state(coaster_score):
    baseline = coaster_score('Wooden', 3., 2., 'Chain')
    for speed, bumps in [(float('nan'), 1.), (1., float('inf')),
                         (-1., 1.), (1., 0.)]:
        try:
            coaster_score('Wooden', speed, bumps, 'Cable')
        except ValueError:
            pass
        else:
            raise AssertionError('accepted nonfinite or nonpositive input')
    assert coaster_score('Wooden', 3., 2., 'Chain') == baseline

# %%
from torch_judge import evaluate
if not evaluate("0080_coaster_scoring_eval.py", coaster_score):
    raise SystemExit(1)
