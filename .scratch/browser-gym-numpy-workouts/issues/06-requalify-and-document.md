# 06: Requalify the gym with 31 workouts

**What to build:** The gym with all 18 new workouts is qualified the way the polish release was, and the README lists them.

**Blocked by:** 02, 03, 04, 05

**Status:** done

- [x] The full e2e suite passes at the root and under the `/torch_dojo/` prefix; `offline_build.py --verify` passes; the Pages smoke passes.
- [x] The offline recovery scope still passes.
- [x] The gym README's workout list and topic groups include the new workouts.

## Comments

- Full e2e suite: 127 passed and 1 failed both at the root and under `/torch_dojo/` (with `DOJO_BASE_URL=/torch_dojo/` and `DOJO_PREFIX`). The failure was the switcher test filtering on "pool", which now also matches 0076 max pooling; it filters on "voxel" now and passes. The offline recovery scope passed in both runs, `make offline-build` verified the manifest, and the Pages smoke passed under the prefix.
- The gym README lists the 30 dojo workouts and the MLP lab by topic and says how to add a workout.
