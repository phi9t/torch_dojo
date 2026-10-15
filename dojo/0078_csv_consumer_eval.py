"""Reference and verifiers for the CSV row consumer contract."""

# %%
from typing import Callable

# Drill 1: It prints `3 2 ['car-1', 'car-3']`: three data rows went in and
# only two records came out, with nothing saying `car-2` was ever there. A
# consumer that silently drops rows makes bad input look like less input.
# Returning one record per data row, with `is_valid` and `errors`, keeps the
# loss visible.

# Drill 2: It prints `{'id': 'b'}` and `{'id': 'c', 'speed': '3'}`. `zip`
# stops at the shorter list, so a short row loses the `speed` key and an
# extra field vanishes without a word. Width has to be checked explicitly,
# and `data` should still hold every header key.

# Drill 3: It prints `{'id': 'car-2', 'speed': None} ['speed: ValueError']`.
# Converting each field in its own `try` keeps the good `id` while recording
# exactly which field failed; one `try` around the whole row would discard
# both.

# %%
def validate_rows(rows: list[list[str]], schema: dict[str, Callable[[str], object]]) -> list[dict]:
    if not rows:
        return []
    header = rows[0]
    if not header or len(set(header)) != len(header) or any(not name for name in header) or header != list(schema):
        raise ValueError("header must match the ordered schema")
    records = []
    for number, raw in enumerate(rows[1:], start=2):
        data = {name: None for name in header}
        errors = []
        if len(raw) != len(header):
            errors.append(f"width: expected {len(header)}, received {len(raw)}")
        for index, name in enumerate(header):
            if index >= len(raw):
                continue
            try:
                data[name] = schema[name](raw[index])
            except (ValueError, TypeError, OverflowError) as exc:
                errors.append(f"{name}: {type(exc).__name__}")
        records.append(dict(record=number, raw=list(raw), data=data,
                            is_valid=not errors, errors=errors))
    return records


# %% [verifiers]
def verify_baseline_valid_row_and_record_number(validate_rows):
    records = validate_rows([["id", "speed"], ["car-1", "12.5"]], {"id": str, "speed": float})
    assert records == [dict(record=2, raw=["car-1", "12.5"],
                            data={"id": "car-1", "speed": 12.5}, is_valid=True, errors=[])]


def verify_followup_bad_rows_are_preserved(validate_rows):
    rows = [["id", "speed"], ["a", "bad"], ["b"], ["c", "3", "extra"]]
    records = validate_rows(rows, {"id": str, "speed": float})
    assert [r["record"] for r in records] == [2, 3, 4]
    assert all(not r["is_valid"] and r["errors"] for r in records)
    assert records[0]["raw"] == ["a", "bad"] and records[0]["data"]["speed"] is None
    assert records[1]["data"]["speed"] is None and records[2]["raw"][-1] == "extra"


def verify_production_bad_header_is_a_global_error(validate_rows):
    for header in (["id", "id"], ["speed", "id"], ["id", ""]):
        try:
            validate_rows([header, ["a", "1"]], {"id": str, "speed": float})
        except ValueError:
            pass
        else:
            raise AssertionError("invalid header accepted")


def verify_baseline_empty_input_has_no_records(validate_rows):
    assert validate_rows([], {"id": str}) == []
    assert validate_rows([["id"]], {"id": str}) == []



def verify_followup_converter_error_is_local(validate_rows):
    def parse(raw):
        if raw == 'type':
            raise TypeError('bad')
        if raw == 'overflow':
            raise OverflowError('bad')
        return int(raw)
    rows = [['n'], ['type'], ['4'], ['overflow']]
    records = validate_rows(rows, {'n': parse})
    assert [r['is_valid'] for r in records] == [False, True, False]
    assert [r['data']['n'] for r in records] == [None, 4, None]
    assert all('n' in r['errors'][0] for r in (records[0], records[2]))


def verify_production_preserves_input_and_all_data_records(validate_rows):
    rows = [['id', 'speed'], ['a', '1'], ['b'], ['c', '2', 'extra']]
    before = [r.copy() for r in rows]
    records = validate_rows(rows, {'id': str, 'speed': float})
    assert rows == before and len(records) == len(rows) - 1
    assert [r['record'] for r in records] == [2, 3, 4]
    assert records[2]['data']['speed'] == 2.
    assert records[2]['raw'] == ['c', '2', 'extra']

# %%
from torch_judge import evaluate
if not evaluate("0078_csv_consumer_eval.py", validate_rows):
    raise SystemExit(1)
