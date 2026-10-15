# 05: Practice attention over paged KV cache

**What to build:** A learner can attend to a logical token sequence stored in nonsequential physical KV pages, including a partly filled final page. Reserve exercise 0093. The approved parent spec governs the common question/eval script contract.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] The question/eval pair uses top-level cells, one editable answer section, a page-table probe, and cumulative baseline, follow-up, and production verifiers.
- [x] Baseline matches dense stable attention for one and two physical blocks, independently per head.
- [x] Follow-up handles shuffled and repeated physical page IDs, a partial last block with adversarial unused values, and zero context returning zeros.
- [x] Production validates indices and context bounds, shape/dtype/device and immutability; its implementation processes blocks without reconstructing the full logical KV sequence while preserving query/cache gradients.
- [x] The reference passes direct evaluation; saved candidate stage selection works; wrong page order and unmasked tail cells fail checks. Review the block-bounded temporary storage rather than relying on a fragile process-memory threshold. Add the exercise to the dojo index with a source link and original wording.

## Comments

Merged exercise #93 and its original, source-linked README entry. The shared eval environment passed all 6 reference verifiers. Saved blank candidate baseline selection ran 1 check and failed as expected. Follow-up checks rejected reversed-page and unmasked-tail mutants. Reference inspection confirms one logical block at a time, with running softmax state; gradient parity and immutability pass.
