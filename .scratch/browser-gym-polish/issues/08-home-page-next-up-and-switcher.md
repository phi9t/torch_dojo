# 08: Home page, Next up and ⌘K switcher

**What to build:** Opening the site root shows a home page instead of a default workout. Workout cards are grouped by topic, and each shows its best time, rep count, highest stage reached and last rep. A **Next up** card suggests what to practise. From any workout, ⌘/Ctrl+K opens a switcher that can be filtered by typing, and a link in the rail returns home. `?exercise=<id>` links still open a workout directly.

**Blocked by:** 02, 05

**Status:** done

- [x] Topics come from the catalog's concepts, and 0061–0066 gain concepts so every workout has a topic.
- [x] Next up is the workout with the oldest last rep. Workouts never repped come first, in learning-path order.
- [x] The switcher filters workouts by id, title and topic, and can be operated entirely from the keyboard.
- [x] Cards for workouts never repped show no stats, rather than zeros.
- [x] The home page follows the same focus rule as the Bench: one calm column of topics, with no dashboards or charts.
- [x] Each switcher row shows how long ago that workout's last rep was ("never" if none), so the choice matches the home page (spec story 8).
- [x] The rep summary's Next up link follows the same Next up rule as the home page, not the fixed learning-path successor (spec story 61).
- [x] pytest checks that every browser workout's catalog entry has at least one topic.
- [x] e2e covers: root opens home; a card opens its workout; Next up follows the rule given seeded rep data; ⌘K filter and open; a deep link bypasses home; the GitHub Pages prefix works.

## Comments

- The site root (`/lab/` with no `exercise`) shows the home page; every e2e spec that relied on `/lab/` opening 0059 now uses `?exercise=0059`.
- Topics: the home page groups each workout under its first concept. Concepts were reordered so the first is a broad area (training, evaluation, geometry, sensors), and 0061–0066 gained concepts. The switcher matches all of a workout's concepts.
- A rep counts on the home page only once it has a check or some active time, because opening a workout already creates a rep. Its date is its latest check, else its start, so autosaves don't make a workout look recent.
- ⌘/Ctrl+K is a window-level capture listener, not a Lumino keybinding, so it opens even from a drill prediction field (which suppresses Lumino shortcuts). It also works on the home page.
- The rail's brand line became the "← All workouts" link. The learning-path panel's "Next:" link still names the path successor; the rep summary's "Next up" follows the home page's rule.
- Pages prefix: `playwright.config.ts` honours `DOJO_PREFIX`; `home.spec.ts` passed with the built site served under `/torch_dojo/` (a symlinked directory). The full prefix run is ticket 12.
- `tests/pages-smoke.cjs` (run by the Pages workflow) was already stale against tickets 01–04 (labels and cell positions) and also opens `/lab/`; ticket 12 rewrites it.
