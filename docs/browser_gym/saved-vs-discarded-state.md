# Saved vs. discarded state in browser gym recovery

Every controller action that can stop execution (Check deadline, Stop button,
Interrupt, Restart kernel, page reload) makes an explicit trade between what
the learner loses and what the gym preserves. This document is the contract
that the panel strings, stopped/timeout receipt prefixes, and Restart dialog
text all implement.

## 1. Check deadline (check worker timeout)

Triggered when `runCheck()`'s controller-owned timer (default 120 s) fires
before the Pyodide worker returns a verifier result.

| State | Fate | Why |
| --- | --- | --- |
| Submitted `source` + `sha256` + `sourceBound` cell indices | **Saved** as `CheckReceipt.status = 'timeout'` | The source was actually dispatched to a worker; the receipt must be source-bound so the learner can see *which cells* they submitted, even after edits. |
| Controller deadline info (which stage, checker version, fixture id) | **Saved** inside the receipt's synthetic `CheckResult` | Qualification and export need stable metadata. |
| Pyodide worker interpreter, half-compiled verifier bytecode, partial result buffers, temp FS files inside the worker | **Discarded** via `worker.terminate()` | These are per-check scratch state and cannot be resumed from the controller; keeping them would leak a stuck Web Worker. |
| Prior check receipts, notebook contents, notes, assistance events, other reps | **Saved** (append-only receipts + `localStorage`) | The timeout receipt is appended; no prior receipt is mutated. |

Visible confirmation: the panel shows `TIMEOUT · source SHA-256 <hash>` and a
block `Saved: … / Discarded: …`.

## 2. Stop button (check-side abort)

Triggered by the learner clicking `Stop` while a check is in flight. Same
state table as §1, except:

- Receipt status is `'stopped'` instead of `'timeout'`.
- The status line starts with `STOPPED · source SHA-256 <hash>`.
- The saved-vs-discarded block uses "You stopped the check." wording instead
  of "hard-terminated at the controller deadline."

Behavior that makes this *source-bound* (the receipt is never a generic
"cancelled" marker):

1. `captureSolution()` is called *before* `runCheck()` and produces
   `{ source, sha256, sourceBound: number[] }`.
2. On `run.abort()`, `CheckStoppedError` carries that exact `source` and
   `sha256` to the panel, which copies them plus `sourceBound` into the
   receipt.
3. So even if the learner edits cells immediately after clicking Stop, the
   stored receipt still describes the source that was really sent.

## 3. Fresh worker recovery after Stop or timeout

Ticket acceptance: "A subsequent check starts in a fresh worker and succeeds
when the answer is valid; previous receipts remain intact."

- `runCheck()` always instantiates `new Worker(...)`, and every settled
  promise (resolve, reject, timeout, abort) calls `cleanupAndTerminate()`,
  which calls `worker.terminate()`. So the next Check is **guaranteed** a
  blank interpreter even if the previous worker was stuck.
- `appendReceipt()` enforces id uniqueness and only pushes; nothing in the
  Stop/timeout path removes prior receipts.
- The panel shows all receipts including the stopped/timeout ones so the
  learner can compare.

## 4. Exploration (notebook kernel) Interrupt

Triggered by the `Interrupt` button; falls back to `Restart kernel` when the
cooperative interrupt path is unavailable.

| State | Fate | Why |
| --- | --- | --- |
| Long-running cell execution (stuck `while True`, slow plot, etc.) | **Interrupted** if `kernelmenu:interrupt-kernel` is wired for the current Pyodide/lite kernel | Uses JupyterLite/Pyodide's interrupt buffer (SharedArrayBuffer + cross-origin isolation). |
| Live kernel variables, module imports, module-level side effects, widget state | **Kept** if interrupt succeeds | Cooperative interrupt is exactly that — no interpreter teardown. |
| Notebook file contents (edits not yet flushed through the 500 ms poll) | **Kept**; they are DOM state separate from the kernel | The attempt store's interval poll snapshots whatever is visible next tick. |
| Attempt metadata, receipts, notes, other notebooks | **Saved** | Stored outside the kernel. |

If interrupt is **unavailable** (no SharedArrayBuffer / COOP/COEP headers):

- The panel keeps the button but writes: *"Interrupt is unavailable because
  SharedArrayBuffer / cross-origin isolation is not active. Use Restart kernel
  to recover from a stuck cell."*
- `Restart kernel` remains explicitly available (§5) so there is always a
  recovery path.

## 5. Exploration (notebook kernel) Restart

Triggered by the `Restart kernel` button or by a JupyterLite toolbar restart.
The gym's own button shows a confirmation dialog before action.

| State | Fate | Why |
| --- | --- | --- |
| Live Pyodide kernel interpreter, in-memory globals, imported modules, half-initialized widget backends, kernel-side outputs | **DISCARDED** | A fresh kernel is booted; the old runtime cannot be carried over. |
| Cell outputs shown in the notebook DOM | **Cleared** (we issue `restart-kernel-and-clear-outputs`) | To avoid a misleading view: stale plots / printed values against a kernel that no longer has that state. |
| Editable notebook text (code/markdown cells, including solution cells) | **Saved** | Owned by JupyterLab Contents + the attempt store's snapshots. |
| Attempt id, notes, assistance events, every append-only receipt (including stopped/timeout ones) | **Saved** | Stored in `localStorage` under the attempt store key, independent of any kernel. |

Confirmation dialog mirrors this table in plain language:

```
Live kernel state (variables, imports, open file handles, half-loaded packages
in memory) is DISCARDED.

Saved state (rep log, editable notebooks, your notes, and every
append-only check receipt including stopped/timeout receipts bound to the
source you submitted) remains available. You can re-run cells to rebuild the
live state.
```

## 6. Page reload / accidental navigation

This is not an explicit button, but it is the common recovery scenario that
ticket 15 ("continue working without losing prior results") is really about.

- The attempt store is backed by `localStorage`.
- `app.restored` reopens the exercise notebook and calls
  `getOrCreateAttempt`, which finds the pre-existing attempt id in
  `activeByExercise`.
- The 500 ms poll re-snapshots the notebook and re-derives the current
  SHA-256 so stale-pass labels still work.
- The live kernel is obviously gone after reload (new browser context) so it
  counts as discarded, exactly like a Restart; that is the default recovery
  path.

## Common invariants across every case

1. **Source-bound receipts are never synthesized from thin air.** Every
   stopped/timeout receipt contains the exact `source`, `sha256`, and
   `sourceBound` cell indices that were captured before the worker was
   launched. They are traceable back to specific notebook cells.
2. **Only live runtime state is discarded.** Anything durable enough to live
   in a receipt, notebook snapshot, notes field, or attempt id is retained.
3. **Passing receipts become stale only when the source hash changes.**
   Timeout/stopped/failed receipts describe a historical event and never
   relabel as STALE — their status lines use the literal
   `TIMEOUT / STOPPED / FAILED` verbatim.
