# %%
# 0078 · What should a CSV consumer receive?
# Goal: Check already-parsed CSV rows against an ordered schema and return one
# record per data row, valid or not. DS07 companion. Ingestion and corrupt
# rows are reported topic families. This exact schema and error-record
# contract are authored practice.

# %%
from typing import Callable
from torch_judge import check

# %%
# Start AFTER lexical CSV parsing; exercise 0072 turns text into rows.
rows_probe = [['id', 'speed'], ['car-1', '12.5'], ['car-2', 'fast']]
schema_probe = {'id': str, 'speed': float}
print('Record 2 is valid; record 3 retains raw car-2, speed=None, and an error.')

# %%
# Each DATA row returns a dict: record (header is record 1), raw (copy),
# data (every header key, None where absent/conversion fails), is_valid,
# errors (strings).
# Watch out: never silently drop a malformed data row.
# Predict: Which fields stay usable in the car-2 record after speed conversion fails?

# %% [drill]
# Drop or keep?
# This consumer skips any row that fails to convert. Predict how many data rows arrive and how many records come out.
rows = [['id', 'speed'], ['car-1', '12.5'], ['car-2', 'fast'], ['car-3', '7']]
kept = []
for raw in rows[1:]:
    try:
        kept.append({'id': raw[0], 'speed': float(raw[1])})
    except ValueError:
        pass
print(len(rows) - 1, len(kept), [r['id'] for r in kept])

# %% [drill]
# Does zip check the width?
# Predict the dicts `zip` builds for a short row and for a row with an extra field.
header = ['id', 'speed']
print(dict(zip(header, ['b'])))
print(dict(zip(header, ['c', '3', 'extra'])))

# %% [drill]
# One bad field, one bad record
# Converting field by field, predict the data and errors for `car-2`.
schema = {'id': str, 'speed': float}
raw = ['car-2', 'fast']
data, errors = {}, []
for name, value in zip(schema, raw):
    try:
        data[name] = schema[name](value)
    except ValueError as exc:
        data[name] = None
        errors.append(f"{name}: {type(exc).__name__}")
print(data, errors)

# %% [answer]
def validate_rows(rows: list[list[str]], schema: dict[str, Callable[[str], object]]) -> list[dict]:
    pass

# %%
# Baseline — Ordered header and typed, valid data rows. Empty input is [].
check('0078_csv_consumer_eval.py', stage='baseline')

# %%
# Follow-up — Width and converter failures invalidate only that record;
# converter `ValueError`, `TypeError`, and `OverflowError` are local errors.
# Preserve extra raw fields and still convert fields that are present.
check('0078_csv_consumer_eval.py', stage='followup')

# %%
# Production — Duplicate/empty/reordered header is a global `ValueError`.
# Preserve input rows, including raw values on invalid records.
# This consumer does not parse quotes or reconstruct physical line numbers.
check('0078_csv_consumer_eval.py', stage='production')
