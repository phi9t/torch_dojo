# %%
# 0035 · A small, deterministic byte-pair-style tokenizer
# Goal: Learn byte-pair-style merges from a word list, then replay them to
# encode text. This character-level teaching tokenizer is not a byte-level
# production BPE.

# %%
# Baseline — `train(corpus, num_merges)` sees a list of NONEMPTY words and a
# nonnegative integer merge budget. Each word begins as characters plus
# `'</w>'`. Count adjacent pairs weighted by word occurrence. Merge the most
# frequent pair, recording it in `self.merges`. `encode(text)` splits on
# whitespace, applies merges in learned order, and returns `list[str]`.
# Predict: Which pair is merged first in `['aaa','aaa','bbb']`?
print("pair counts in 'aaa':", {('a', 'a'): 2, ('a', '</w>'): 1})

# %% [drill]
# Which pair wins a tie?
# Every pair in `['ac', 'ab']` occurs once. Predict what `most_common` picks and what the lexicographic tie-break picks.
from collections import Counter
counts = Counter()
for word in ['ac', 'ab']:
    symbols = list(word) + ['</w>']
    for pair in zip(symbols, symbols[1:]):
        counts[pair] += 1
print(counts.most_common(1)[0][0])
print(min(counts, key=lambda pair: (-counts[pair], pair)))

# %% [drill]
# Overlapping pairs
# `'aaa'` contains the pair `('a', 'a')` twice. Predict the count, and the symbols after one left-to-right merge.
symbols = list('aaa') + ['</w>']
print(sum(pair == ('a', 'a') for pair in zip(symbols, symbols[1:])))
merged, i = [], 0
while i < len(symbols):
    if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == ('a', 'a'):
        merged.append(symbols[i] + symbols[i + 1])
        i += 2
    else:
        merged.append(symbols[i])
        i += 1
print(merged)

# %% [drill]
# Replay in learned order
# Training learned `('b', 'c')` before `('a', 'b')`. Predict how `'abc'` encodes in that order and in the reverse order.
def apply(symbols, merges):
    for pair in merges:
        merged, i = [], 0
        while i < len(symbols):
            if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
                merged.append(symbols[i] + symbols[i + 1])
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        symbols = merged
    return symbols
learned = [('b', 'c'), ('a', 'b')]
print(apply(list('abc') + ['</w>'], learned))
print(apply(list('abc') + ['</w>'], learned[::-1]))

# %% [answer]
class SimpleBPE:
    def __init__(self):
        self.merges = []

    def train(self, corpus, num_merges):
        pass

    def encode(self, text):
        pass

# %%
from torch_judge import check
check("0035_bpe_eval.py", stage="baseline")

# %%
# Follow-up — Break equal-frequency pair ties lexicographically. Merge
# non-overlapping occurrences from left to right, so `'aaaa'` at one merge
# becomes `['aa','aa','</w>']`. Stop when no adjacent pair remains; a second
# train call replaces the old merges.
# Predict: Why can more requested merges yield fewer recorded merges?
check("0035_bpe_eval.py", stage="followup")

# %%
# Production — Reject malformed corpora, negative/bool budgets, and nonstring
# text with `ValueError`. Empty corpus and empty text are valid. Do not mutate
# the caller's list. Training may scan the current vocabulary per merge;
# keep counts per unique word instead of expanding duplicate corpus words.
check("0035_bpe_eval.py", stage="production")
