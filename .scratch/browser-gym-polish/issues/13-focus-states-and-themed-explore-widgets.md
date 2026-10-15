# 13: Focus states and themed explore widgets

**What to build:** Every control in the gym shows a visible focus state when reached by keyboard, and exploration widgets (tensor cards, plots, sliders) follow the gym's light and dark palettes, so dark mode never flashes a white plot. This closes spec stories 35 and 72, which no other ticket owned.

**Blocked by:** 02

**Status:** done

- [x] Every interactive control (rail, tabs, buttons, squares, seals, drill fields, switcher, home cards) has a `:focus-visible` style using the accent token, legible in both themes.
- [x] Explore panel outputs and widgets use the gym's tokens for ground, ink and accent in both themes.
- [x] e2e: tabbing through the 0059 Bench shows a visible focus indicator on each control it lands on (computed outline or box-shadow, not a class assert).
- [x] e2e: in dark mode, a rendered explore panel has no light (near-white) background.

## Comments

- One focus rule covers the whole gym (Bench, home page, switcher): an accent outline on `:focus-visible`, specific enough to beat JupyterLab's outline resets, with code editors and notebook containers excluded (they show focus their own way). The switcher's search field uses an inset accent rule instead.
- Explore panels: the lessons' neutral CSS colours now use the gym tokens with their old values as fallbacks (`var(--dojo-surface, #ffffff)` and so on). SVG presentation attributes can't take tokens, so `index.css` restyles the neutral, near-white and very dark stroke and fill values by attribute; inputs inside explore panels use the surface and accent tokens. Data series colours (teal, red, blue) are unchanged.
- `focus.spec.ts` tabs through the Bench, the drill cards, the switcher and the home page, failing on any interactive control that takes focus without an outline or inset rule; in dark mode it checks five lessons' explore panels for light backgrounds, light SVG fills and dark text.
- Tabbing showed JupyterLab's per-cell toolbar (duplicate, move, delete) on Bench cells, without a focus ring. It is chrome the spec removes, so `jupyter-lite.json` now disables `@jupyterlab/cell-toolbar-extension:plugin`, and the shell spec's chrome guard checks it stays gone. Notebook cells reached by Tab (the brief, drills, explore panels) get the accent ring too.
