# 04: Verifier grid and detail panel

**What to build:** After a check, the learner sees one square per verifier, grouped by stage, instead of the current list. Squares are green for pass, red for fail and outlined for not run; stages the check didn't reach are hatched. The first failing square is selected automatically. Its detail panel shows the verifier's name, timing, message and captured case, plus a **Load case into scratch** button that puts the case's inputs into the scratch cell so the learner can reproduce it.

**Blocked by:** 03

**Status:** done

The square model came from the prototype:

```ts
type SquareState = 'pass' | 'fail' | 'not-run' | 'locked';
interface VerifierSquare { stage: string; name: string; state: SquareState; elapsedMs?: number; case?: CapturedCase }
// After a check: select the first failing square, else none.
```

- [x] Before any check, the grid area shows a single quiet line instead of an empty grid.
- [x] While a check runs, a quiet indicator replaces the grid and a Stop control is offered. Stop and timeout still produce receipts exactly as today.
- [x] The squares are keyboard-navigable, and each has an accessible name stating its verifier and state.
- [x] Load case into scratch opens the scratch cell with a runnable reproduction, and does not alter the answer cell.
- [x] The stage selector, the runtime/version readout and the source-bound receipt details stay available.
- [x] e2e covers: starter → first baseline square selected; reference → all green; unstable answer → the expected failing square; locked stages hatched; stop and timeout.

## Comments

- Locked squares come from the checker bundle's verifier names (`verify_<stage>_…`), which the extension fetches once. Before it arrives, and for receipts saved without a `stage`, the grid still shows every verifier the check reported. Stop and timeout results now record the stage being checked, so their squares read "not run" rather than locked.
- Load case into scratch loads all captured cases (as before), and the new scratch cell points at the selected verifier's first case (`gym_case(n)`).
- Also done here, because the grid needed them: the Check button names its stage (story 40, "Check follow-up"), and a quiet readout line shows runtime · Python · NumPy · checker version (story 51).
- Independent checkpoints (MLP lab) never show hatched squares. Ticket 09 owns their grouping and evidence.
