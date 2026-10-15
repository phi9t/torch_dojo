# 02: Hide JupyterLab and apply the gym's look

**What to build:** When a workout opens, none of JupyterLab's chrome is visible: no menu bar, document tabs, status bar, file browser, launcher or side icon bars (ADR 0003). The page is still one column, but the Python editor, the rendered markdown/HTML and the gym's own panels all share the gym's calm, mathematical look in both light and dark. A theme toggle switches between them.

**Blocked by:** None (can start immediately)

**Status:** done

- [x] JupyterLab's menus, document tabs, status bar, file browser, launcher and side icon bars are turned off through the site's disabled-extensions config, not by hiding them with CSS.
- [x] A design-token set defines the light and dark palettes: warm paper ground, ink text, a muted ink-teal accent and a cinnabar seal colour. It overrides JupyterLab's own CSS variables, so the editor, outputs and rendered markdown follow it.
- [x] The type uses Newsreader for prose, JetBrains Mono for code and a math face that matches. The fonts ship with the site, so nothing loads from a font CDN.
- [x] The theme toggle follows the system setting by default, remembers an explicit choice, and switches without a reload.
- [x] An e2e test fails if any of the disabled chrome becomes visible, so a JupyterLite upgrade that renames a plugin ID is caught.
- [x] The existing e2e suite passes in both themes for at least the 0059 check flow.

## Comments

- Two pieces of chrome are not plugins. Document tabs go away through single-document mode (`browser_gym/lab/jupyter-lite.json`). The shell's own title and menu strips are hidden through LabShell's API in the extension. ADR 0003 now records this, and `shell.spec.ts` guards all of it (mutation-checked by re-enabling the status bar).
- The notebook toolbar is hidden too, and replaced by a quiet `Python · idle` kernel readout in the gym masthead. Interrupt now calls the kernel directly, because the menu command it used is gone.
- The light ground follows the spec's "cool paper" (`#f2f3f0`), not this ticket's "warm paper". Fail red is a crimson, kept distinct from the cinnabar seal.
- The math face is MathJax's bundled fonts, per the spec's Math row.
- Build fix: `make build` and `offline_build.py` now drop `.jupyterlite.doit.db`, because doit otherwise skipped re-merging `jupyter-lite.json` into a cleaned `dist/`.
- Known before this ticket, and still failing on unmodified HEAD: `cases.spec.ts` (0059 now reports 2 failed examples, not 1) and `qualification_0061.spec.ts`. `attempts.spec.ts:638` failed once in a full run and passed 3/3 alone.
- Not yet covered: dark-mode users see a brief light flash before the extension activates. The offline manifest listing the fonts is ticket 12.
