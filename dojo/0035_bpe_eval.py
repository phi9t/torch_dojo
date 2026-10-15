"""Three-stage reference for deterministic character-level BPE."""
# %%
from collections import Counter

# Drill 1: `most_common` prints `('a', 'c')` while the tie-break prints
# `('a', 'b')`. A `Counter` breaks ties by insertion order, so its winner
# depends on which word came first in the corpus. Sorting on
# `(-count, pair)` makes the choice depend only on the counts and the pairs,
# which is what makes training deterministic.

# Drill 2: It prints `2` and then `['aa', 'a', '</w>']`. Counting sees two
# overlapping `('a', 'a')` pairs, but they share the middle `a`, so one merge
# can only consume the first; the scan then skips past it. Pair counts can
# overstate what a merge actually replaces.

# Drill 3: Learned order gives `['a', 'bc', '</w>']`; reversed order gives
# `['ab', 'c', '</w>']`. Once `b` joins `c`, the pair `('a', 'b')` no longer
# exists, so applying merges in any other order produces tokens training
# never made. Encoding must replay `self.merges` exactly as recorded.

# %%
class SimpleBPE:
    def __init__(self):
        self.merges = []

    def train(self, corpus, num_merges):
        if (not isinstance(corpus, list) or any(not isinstance(word, str) or not word or word.isspace() or any(c.isspace() for c in word) for word in corpus)
                or isinstance(num_merges, bool) or not isinstance(num_merges, int) or num_merges < 0):
            raise ValueError("corpus must be a list of nonempty words; num_merges >= 0")
        vocab = Counter(tuple(word) + ('</w>',) for word in corpus)
        self.merges = []
        for _ in range(num_merges):
            counts = Counter()
            for symbols, frequency in vocab.items():
                for pair in zip(symbols, symbols[1:]):
                    counts[pair] += frequency
            if not counts:
                break
            best = min(counts, key=lambda pair: (-counts[pair], pair))
            self.merges.append(best)
            updated = Counter()
            for symbols, frequency in vocab.items():
                merged = []
                i = 0
                while i < len(symbols):
                    if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == best:
                        merged.append(symbols[i] + symbols[i + 1])
                        i += 2
                    else:
                        merged.append(symbols[i])
                        i += 1
                updated[tuple(merged)] += frequency
            vocab = updated

    def encode(self, text):
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        tokens = []
        for word in text.split():
            symbols = list(word) + ['</w>']
            for pair in self.merges:
                merged = []
                i = 0
                while i < len(symbols):
                    if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
                        merged.append(pair[0] + pair[1])
                        i += 2
                    else:
                        merged.append(symbols[i])
                        i += 1
                symbols = merged
            tokens.extend(symbols)
        return tokens

# %% [verifiers]
def verify_baseline_weighted_pair_count(SimpleBPE):
    bpe = SimpleBPE()
    bpe.train(['aaa', 'aaa', 'bbb'], 1)
    assert bpe.merges == [('a', 'a')]
    assert bpe.encode('aaa') == ['aa', 'a', '</w>']


def verify_baseline_reconstruction(SimpleBPE):
    bpe = SimpleBPE()
    bpe.train(['low', 'low', 'lower'], 4)
    tokens = bpe.encode('low lower')
    assert isinstance(tokens, list) and all(isinstance(t, str) for t in tokens)
    assert ''.join(token.replace('</w>', '') for token in tokens) == 'lowlower'
    assert sum('</w>' in token for token in tokens) == 2


def verify_followup_ties_and_overlap(SimpleBPE):
    bpe = SimpleBPE()
    bpe.train(['ab', 'ac'], 1)
    assert bpe.merges == [('a', 'b')]
    bpe.train(['aaaa'], 1)
    assert bpe.merges == [('a', 'a')]
    assert bpe.encode('aaaa') == ['aa', 'aa', '</w>']


def verify_followup_stop_and_retrain(SimpleBPE):
    bpe = SimpleBPE()
    bpe.train(['a'], 100)
    assert bpe.merges == [('a', '</w>')]
    bpe.train(['xyz'], 0)
    assert bpe.merges == [] and bpe.encode('x') == ['x', '</w>']


def verify_production_invalid_and_empty(SimpleBPE):
    bpe = SimpleBPE()
    for corpus, n in (([''], 1), (['a b'], 1), (['a', 2], 1), ('abc', 1), (['a'], -1), (['a'], True)):
        try:
            bpe.train(corpus, n)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted {corpus}, {n}")
    try:
        bpe.encode(3)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted nonstring text")
    bpe.train([], 3)
    assert bpe.merges == [] and bpe.encode('') == []


def verify_production_replay_and_input_immutability(SimpleBPE):
    corpus = ['banana', 'bandana', 'banana']
    original = corpus.copy()
    a, b = SimpleBPE(), SimpleBPE()
    a.train(corpus, 8)
    b.train(corpus, 8)
    assert corpus == original and a.merges == b.merges
    assert a.encode('banana bandana') == b.encode('banana bandana')

# %%
from torch_judge import evaluate
if not evaluate("0035_bpe_eval.py", SimpleBPE):
    raise SystemExit(1)
