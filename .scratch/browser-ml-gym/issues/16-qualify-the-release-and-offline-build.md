# 16: Qualify the release and offline build

**What to build:** The full browser gym is verified as a reliable release, including every lab, recovery, export, and an offline reload.

**Blocked by:** 08: Diagnose a broken MLP; 09: Connect calibration with `0069`; 10: Explore K-means with `0061`; 11: Explore box geometry with `0062`–`0064`; 12: Explore sensor and voxel behavior with `0065`–`0066`; 13: Visualize anchor matching with `0070`; 14: Explore grouped evaluation with `0073`–`0074`; 15: Recover from stopped work

**Status:** resolved

- [x] All 12 existing NumPy exercises and the MLP checkpoints pass with references and reject starters and selected plausible wrong answers using the intended verifier.
- [x] Browser acceptance covers gradient repair, bounded training, an evaluation figure, widget probe update, tensor card, assistance events, isolated checks, a stopped check, a fresh attempt, and export/reimport with matching checked-source hashes.
- [x] Native regression checks, browser build checks, worker checks, and automated browser acceptance pass in the documented environment.
- [x] Required runtime assets are bundled and a previously loaded site works after reload with network disabled; release documentation states the tested offline scope.
