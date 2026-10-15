"""Three-stage reference for a strict CSV dialect with chunked input."""

# %%
from collections.abc import Iterable
from hypothesis import given, settings, strategies as st

# Drill 1: `split` prints `['Ada', '"red', ' blue"']`, three fields with the
# quotes kept, while a CSV reader prints `['Ada', 'red, blue']`. A comma is a
# separator only outside quotes, so the parser has to track whether it is
# inside a quoted field rather than split first.

# Drill 2: `splitlines` sees `2` lines, `['"first', 'second",x']`, but the
# reader returns one record, `[['first\nsecond', 'x']]`. A newline inside
# quotes is data, so records cannot be found by splitting lines first; the
# same quoted-state machine must decide what ends a record.

# Drill 3: Stripping prints `say ""hi`, which keeps the doubled quotes and
# loses the closing one, while the field really holds `say "hi"`. Inside a
# quoted field, a quote is either the end of the field or the first half of
# an escaped pair, and only the next character tells which.

# %%
def parse_csv_rows(text: str | Iterable[str]) -> list[list[str]]:
    if isinstance(text, str):
        chunks = (text,)
    else:
        try:
            chunks = iter(text)
        except TypeError as exc:
            raise ValueError("text or an iterable of text chunks required") from exc
    rows: list[list[str]] = []
    row: list[str] = []
    field: list[str] = []
    state = "start"  # start, plain, quoted, quote_pending, after
    pending_cr = False
    active = False

    for chunk in chunks:
        if not isinstance(chunk, str):
            raise ValueError("every input chunk must be text")
        for char in chunk:
            if pending_cr:
                if char != "\n":
                    raise ValueError("bare carriage return")
                row.append("".join(field))
                rows.append(row)
                row, field, state, pending_cr, active = [], [], "start", False, False
                continue
            if state == "quoted":
                if char == '"':
                    state = "quote_pending"
                else:
                    field.append(char)
                active = True
                continue
            if state == "quote_pending":
                if char == '"':
                    field.append('"')
                    state = "quoted"
                    continue
                state = "after"
            if char == ",":
                row.append("".join(field))
                field, state, active = [], "start", True
            elif char == "\n":
                row.append("".join(field))
                rows.append(row)
                row, field, state, active = [], [], "start", False
            elif char == "\r":
                pending_cr = True
            elif state == "after":
                raise ValueError("text after closing quote")
            elif char == '"':
                if state != "start":
                    raise ValueError("quote in unquoted field")
                state, active = "quoted", True
            else:
                field.append(char)
                state, active = "plain", True
    if pending_cr:
        raise ValueError("bare carriage return")
    if state == "quoted":
        raise ValueError("unterminated quoted field")
    if active or row or field or state != "start":
        row.append("".join(field))
        rows.append(row)
    return rows


# %% [verifiers]
def verify_baseline_quoted_comma_and_escaped_quote(parse_csv_rows):
    assert parse_csv_rows('a,"b,c"\n"say ""hi""",d') == [['a', 'b,c'], ['say "hi"', 'd']]


def verify_baseline_trailing_empty_field(parse_csv_rows):
    assert parse_csv_rows('a,b,') == [['a', 'b', '']]


def verify_followup_crlf_blank_line_and_multiline_quote(parse_csv_rows):
    assert parse_csv_rows('a,b,\r\nc,d,') == [['a', 'b', ''], ['c', 'd', '']]
    assert parse_csv_rows('') == [] and parse_csv_rows('\n') == [['']]
    assert parse_csv_rows('"first\nsecond",x') == [['first\nsecond', 'x']]


def verify_followup_malformed_quote_and_bare_cr(parse_csv_rows):
    for source in ('a"b', '"a"b', '"a', 'a\rb', 'a\r'):
        try:
            parse_csv_rows(source)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted malformed CSV: {source!r}")


def verify_production_every_split_boundary(parse_csv_rows):
    source = 'id,note\r\n1,"a,""b"""\r\n2,"line\none"'
    expected = [['id', 'note'], ['1', 'a,"b"'], ['2', 'line\none']]
    assert parse_csv_rows(source) == expected
    for split in range(len(source) + 1):
        assert parse_csv_rows((source[:split], source[split:])) == expected
    assert parse_csv_rows(iter(source)) == expected


def verify_production_reject_nontext_chunks(parse_csv_rows):
    for source in (42, ['a', 12]):
        try:
            parse_csv_rows(source)
        except ValueError:
            pass
        else:
            raise AssertionError("nontext input accepted")


def verify_production_generated_chunks(parse_csv_rows):
    @settings(max_examples=35, deadline=None, derandomize=True)
    @given(st.text(alphabet=st.characters(blacklist_characters=',"\r\n'), max_size=20),
           st.integers(min_value=0, max_value=20))
    def property_check(field, split):
        source = f'"{field}",x\r\n'
        expected = [[field, 'x']]
        assert parse_csv_rows([source[:split], source[split:]]) == expected

    property_check()


# %%
from torch_judge import evaluate
if not evaluate("0072_csv_rows_eval.py", parse_csv_rows):
    raise SystemExit(1)
