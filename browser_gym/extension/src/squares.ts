import type { CheckResult } from './checker';
import { resultStage, type StageOrder } from './seals';

export type SquareState = 'pass' | 'fail' | 'not-run' | 'locked';

export type VerifierCheck = CheckResult['checks'][number];

/** One square per verifier, grouped by stage (or by checkpoint for labs). */
export interface VerifierSquare {
  stage: string;
  name: string;
  state: SquareState;
  elapsedMs?: number;
  /** The verifier's reported result, with its error, evidence and captured samples. */
  check?: VerifierCheck;
  /** From an earlier check of another checkpoint, not the check just shown. */
  earlier?: boolean;
  /** Reference components the checkpoint's check substituted for the learner's. */
  substitutions?: CheckResult['substitutions'];
}

/** Groups a checker bundle's verifier names by their `verify_<stage>_` prefix. */
export function stageVerifiers(stages: string[], names: string[]): Record<string, string[]> {
  const grouped: Record<string, string[]> = {};
  for (const stage of stages) {
    const own = names.filter(name => name.startsWith(`verify_${stage}_`));
    if (own.length) grouped[stage] = own;
  }
  return grouped;
}

/**
 * Squares for a check result. Independent checkpoints (labs) also show the
 * latest result of each other checkpoint (`others`), and a checkpoint never
 * checked shows one square for its `checkpoint_<name>` check.
 */
export function buildSquares(
  stages: string[],
  verifiers: Record<string, string[]>,
  result?: CheckResult,
  order: StageOrder = 'cumulative',
  others: CheckResult[] = []
): VerifierSquare[] {
  if (order === 'independent') return checkpointSquares(stages, verifiers, result, others);
  // A result whose stage can't be told still shows what it reported, under the first stage.
  const checked = result ? Math.max(0, resultStage(stages, result)) : -1;
  const reported = new Map((result?.checks ?? []).map(check => [check.name, check]));
  const squares: VerifierSquare[] = [];
  for (const [index, stage] of stages.entries()) {
    const reached = order === 'cumulative' ? index <= checked : index === checked;
    for (const name of verifiers[stage] ?? []) {
      const check = reported.get(name);
      reported.delete(name);
      const state: SquareState = !result ? 'not-run'
        : order === 'cumulative' && !reached ? 'locked'
          : check?.status === 'passed' ? 'pass'
            : check ? 'fail' : 'not-run';
      squares.push({ stage, name, state, elapsedMs: check?.elapsed_ms, check });
    }
    if (index === checked) {
      // Verifiers the bundle did not list (labs report by checkpoint) join the checked stage.
      for (const check of reported.values()) {
        squares.push({ stage, name: check.name, state: check.status === 'passed' ? 'pass' : 'fail', elapsedMs: check.elapsed_ms, check });
      }
    }
  }
  return squares;
}

function checkpointSquares(
  stages: string[],
  verifiers: Record<string, string[]>,
  result: CheckResult | undefined,
  others: CheckResult[]
): VerifierSquare[] {
  const squares: VerifierSquare[] = [];
  for (const [index, stage] of stages.entries()) {
    const current = result && resultStage(stages, result) === index ? result : undefined;
    const source = current ?? others.find(other => resultStage(stages, other) === index);
    const reported = new Map((source?.checks ?? []).map(check => [check.name, check]));
    const names = [...(verifiers[stage] ?? []), ...[...reported.keys()].filter(name => !verifiers[stage]?.includes(name))];
    if (!names.length) names.push(`checkpoint_${stage}`);
    for (const name of names) {
      const check = reported.get(name);
      const state: SquareState = !check ? 'not-run' : check.status === 'passed' ? 'pass' : 'fail';
      squares.push({
        stage, name, state, elapsedMs: check?.elapsed_ms, check,
        ...(source && !current ? { earlier: true } : {}),
        ...(source?.substitutions?.length ? { substitutions: source.substitutions } : {})
      });
    }
  }
  return squares;
}

/** For independent checkpoints, the latest result of each checkpoint other than the one shown. */
export function otherCheckpointResults(
  stages: string[],
  order: StageOrder,
  receipts: Array<{ result: CheckResult }>,
  shown?: CheckResult
): CheckResult[] {
  if (order !== 'independent') return [];
  const latest = new Map<string, CheckResult>();
  for (const receipt of [...receipts].reverse()) {
    const stage = stages[resultStage(stages, receipt.result)];
    if (stage && !latest.has(stage)) latest.set(stage, receipt.result);
  }
  if (shown) latest.delete(stages[resultStage(stages, shown)]);
  return [...latest.values()];
}

/** The first failing square of the check just shown, else none. */
export function firstFailing(squares: VerifierSquare[]): VerifierSquare | undefined {
  return squares.find(square => square.state === 'fail' && !square.earlier);
}
