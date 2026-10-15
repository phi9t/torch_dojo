# Reliable, inviting practice loops

User direction: improve all three architecture-review candidates so the browser gym and GPU Jupyter insula are enjoyable places to learn.

Keep interactive question/eval scripts, cumulative stages, the JupyterLite Bench, and Emacs ownership of Org execution. Reduce the state and ordering rules callers must know.

1. Rep lifecycle owns switching, saving and recovery. Late checks stay attached to the submitted rep; changing reps never overwrites another rep's edits.
2. The Emacs scenario owns navigation and saved-result freshness. TUI drafts, focus and undo remain local. Unknown execution outcomes never trigger retries.
3. Native evaluation and browser generation share static tagged-script interpretation. Setup, helpers, decorators and source order retain their meaning.

Qualification covers existing browser race scenarios, native/browser source fixtures, Emacs scenario behavior, and Textual editor behavior. Learner feedback should explain the next action without exposing implementation mechanics.

## Implementation

- `RepLifecycle` owns the visible rep, notebook persistence, transition ordering,
  stale captures and selective import recovery. Jupyter notebook operations are
  supplied by an adapter; in-memory notebook and contents adapters test the same
  behavior. The old lock-only module and its tests are removed.
- The Emacs scenario owns snapshot projection, completed-step navigation and
  submitted-source/result freshness. The bridge serializes and dispatches. The
  TUI retains unsaved drafts and displays worker-owned stale state with its own
  key hints. Terminal Emacs retains its own key hints.
- `TaggedScript` selects validated answer/setup and verifier programs for both
  consumers. Browser bundles preserve verifier program order and use a distinct
  traceback filename so archived checks still resolve against their old source.
  Author comments remain in answer cells; drill imports stay with the drills.
