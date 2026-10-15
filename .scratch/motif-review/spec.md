# Reusable motifs and spaced recall

User direction: combine a motif library linked to workouts with a spaced-review
queue, usable in both the browser and native Jupyter/rootfs workflows.

- Shared authored library: concrete prediction, explanation, NumPy probe,
  transfer cue, links to real workouts.
- Due-first queue, then new cards. Voluntary early practice is allowed.
- Reveal before self-rating. Again = 10 minutes; hard = 1 day; good = at least
  3 days or twice the preceding interval, capped at 90 days.
- Reviews are separate from reps, active time, verifier results, and drill notes.
- Portable versioned archives merge the most recent review per card; malformed
  archives and storage failures must not erase progress or claim success.
- Native rich/text display requires no new runtime dependency. Browser cards
  and probes work offline; PyTorch-only workouts remain native-only.
