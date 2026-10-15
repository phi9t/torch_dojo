# %%
# 0072 · Parse CSV records without losing the awkward ones
# Goal: Turn text in a small CSV dialect into a list of records, each a list
# of string fields. This dialect is authored practice; the source report did
# not fix its grammar. Build one parser, then extend it through the follow-up
# and production stages.

# %%
from collections.abc import Iterable

# %%
# Baseline — Predict these rows before coding.
sample = 'name,note\nAda,"red, blue"'
print("The comma inside quotes stays in one field:", sample)
# Comma separates fields; a quote can open a field only at its start;
# doubled quotes inside a quoted field encode one literal quote.
# LF ends a record. A final LF does not make an extra record.

# %% [drill]
# Split on commas?
# Predict the fields `str.split(',')` gives for a row whose second field is quoted and contains a comma.
import csv
row = 'Ada,"red, blue"'
print(row.split(','))
print(next(csv.reader([row])))

# %% [drill]
# Lines or records?
# A quoted field contains a newline. Predict how many lines `splitlines` sees and how many records a CSV reader sees.
import csv, io
text = '"first\nsecond",x\n'
print(len(text.splitlines()), text.splitlines())
print(list(csv.reader(io.StringIO(text))))

# %% [drill]
# Doubled quotes
# Predict what stripping the outer quotes leaves, and what the field really holds.
import csv
field = '"say ""hi"""'
print(field.strip('"'))
print(next(csv.reader([field])))

# %% [answer]
def parse_csv_rows(text: str | Iterable[str]) -> list[list[str]]:
    pass

# %%
# Baseline check — Try an ordinary row, a quoted comma, and an escaped quote.
from torch_judge import check
check("0072_csv_rows_eval.py", stage="baseline")

# %%
# Follow-up — CRLF also ends a record, even when a chunk boundary splits it.
# Newlines inside quotes are data. An empty input gives no records; a blank
# line gives one empty field.
# Watch out: a quote in an unquoted field, text after a closing quote,
# unterminated quotes, and bare CR raise `ValueError`.
check("0072_csv_rows_eval.py", stage="followup")

# %%
# Production challenge — Accept either one str or an iterable of str chunks.
# A chunk may end anywhere: inside CRLF, a quoted field, or a doubled quote.
# Process each chunk without joining the entire input first. Preserve Unicode
# characters and rows exactly; reject a non-string chunk. Output is a list,
# so total output memory still grows with the parsed records.
# Predict: What API would you choose if even that output must be streamed?
check("0072_csv_rows_eval.py", stage="production")
