# 12: Requalify the offline build and GitHub Pages

**What to build:** The polished gym is qualified the same way the current one is: the offline build works from its checksummed bundle with no network, and the deployed GitHub Pages site works under its repository prefix. The README describes the new workflow.

**Blocked by:** 01, 02, 03, 04, 05, 06, 07, 08, 09, 10, 11, 13

**Status:** done

- [x] The offline build's checksum manifest includes the bundled fonts and any new assets, and verification fails if one is missing or altered.
- [x] The qualified offline recovery scope passes with the new shell, including rep export and import.
- [x] The full e2e suite passes under the `/torch_dojo/` Pages prefix.
- [x] The gym's README covers the home page, the Bench layout, reps and splits, drills, the keyboard shortcuts, and the authoring markers.

## Comments

- Manifest: `offline-manifest.json` now lists every file of the gym extension (its scripts, styles and the bundled Newsreader and JetBrains Mono fonts) and MathJax's fonts. `offline_build.py --verify` names any missing or altered asset; `make offline-build` runs it after building, and a pytest tampers with a font and removes a lesson to prove it fails.
- Prefix: the e2e specs navigate relative to Playwright's `baseURL`, which carries `DOJO_PREFIX`, so the same suite runs under `/torch_dojo/`. With `DOJO_BASE_URL=/torch_dojo/ make -C browser_gym offline-build` served under the prefix, the full suite ran 87 passed, 1 failed; that failure (anchor_matching pinning the answer cell's notebook position, which ticket 11's drills moved) is fixed. The root-served full suite then ran 86 passed and 6 failed, all in ticket 13's new focus spec, fixed there.
- The qualified offline recovery scope (`recovery_offline.spec.ts`, including rep export and import) passes in both runs.
- `tests/pages-smoke.cjs` (run by the Pages workflow) was rewritten for the new shell: home page, a workout's typed brief, an explore widget, a passing check, the verifier grid, and the answer surviving a reload. It still fails the run on unexpected HTTP errors. It passed against the root-served site.
- The gym README covers the home page and Next up, the Bench, the keyboard shortcuts, reps and splits, drills, the authoring markers, the manifest verifier and prefix testing; the release hash for `catalog.json` is refreshed.
