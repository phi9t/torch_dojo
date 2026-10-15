/**
 * Stage seals in the side rail. Dojo workouts check cumulative stages
 * (ADR 0002): passing `followup` also passes `baseline`. The MLP lab's
 * checkpoints are independent.
 */
import { formatDuration, type Splits } from './splits';

export type StageOrder = 'cumulative' | 'independent';

export interface StagedResult {
  stage?: string;
  checks?: Array<{ name: string }>;
}

interface StagedReceipt {
  status: string;
  result?: StagedResult;
}

/**
 * The index of the stage a result was checked at, or -1. Results saved before
 * they carried a stage fall back to the furthest `verify_<stage>` verifier
 * they report.
 */
export function resultStage(stages: string[], result?: StagedResult): number {
  if (!result) return -1;
  if (result.stage) return stages.indexOf(result.stage);
  const reported = (result.checks ?? []).map(check =>
    stages.findIndex(stage => check.name === `verify_${stage}` || check.name.startsWith(`verify_${stage}_`)));
  return Math.max(-1, ...reported);
}

/** Whether a rep has passed its last stage, or every checkpoint when they are independent. */
export function isFinished(stages: string[], passed: Set<string>, order: StageOrder): boolean {
  return order === 'cumulative' ? passed.has(stages[stages.length - 1]) : stages.every(stage => passed.has(stage));
}

export function passedStages(stages: string[], receipts: StagedReceipt[], order: StageOrder): Set<string> {
  const passed = new Set<string>();
  for (const receipt of receipts) {
    if (receipt.status !== 'passed') continue;
    const index = resultStage(stages, receipt.result);
    if (index < 0) continue;
    if (order === 'independent') passed.add(stages[index]);
    else for (const stage of stages.slice(0, index + 1)) passed.add(stage);
  }
  return passed;
}

const STAGE_NAMES: Record<string, string> = {
  baseline: 'Baseline',
  followup: 'Follow-up',
  production: 'Production'
};

export function stageName(stage: string): string {
  return STAGE_NAMES[stage] ?? stage.charAt(0).toUpperCase() + stage.slice(1);
}

export interface SealTimes {
  /** This rep's split per stage; `null` when the split predates timing. */
  splits: Splits;
  /** The best split per stage across the workout's other reps. */
  best: Record<string, number>;
}

function delta(split: number, best: number): string {
  const difference = split - best;
  if (Math.abs(difference) < 1) return 'even';
  return `${difference < 0 ? '−' : '+'}${formatDuration(Math.abs(difference))}`;
}

export class StageSeals {
  readonly node = document.createElement('ol');
  private readonly items = new Map<string, { row: HTMLLIElement; seal: HTMLSpanElement; times: HTMLSpanElement }>();
  private shownRep?: string;
  private shownPassed = new Set<string>();

  constructor(stages: string[], order: StageOrder = 'cumulative') {
    this.node.className = 'dojo-seals';
    // A lab's rail lists its checkpoints instead of the three stages.
    this.node.setAttribute('aria-label', order === 'independent' ? 'Checkpoints' : 'Stages');
    for (const stage of stages) {
      const row = document.createElement('li');
      row.className = 'dojo-seal-row';
      const seal = document.createElement('span');
      seal.className = 'dojo-seal';
      seal.setAttribute('aria-hidden', 'true');
      seal.textContent = stageName(stage).charAt(0);
      const text = document.createElement('span');
      text.className = 'dojo-seal-text';
      const label = document.createElement('span');
      label.className = 'dojo-seal-label';
      label.textContent = stageName(stage);
      const times = document.createElement('span');
      times.className = 'dojo-seal-times';
      text.append(label, times);
      row.append(seal, text);
      this.items.set(stage, { row, seal, times });
      this.node.append(row);
    }
    this.render(new Set(), stages[0]);
  }

  /**
   * Shows each stage as passed, current or upcoming, with this rep's split and
   * the best split. A stage that passes while the learner watches is stamped.
   */
  render(passed: Set<string>, current: string, times: SealTimes = { splits: {}, best: {} }, repId?: string): void {
    const sameRep = repId !== undefined && repId === this.shownRep;
    for (const [stage, { row, seal, times: timesNode }] of this.items) {
      const state = passed.has(stage) ? 'passed' : stage === current ? 'current' : 'upcoming';
      row.dataset.state = state;
      const split = times.splits[stage];
      const best = times.best[stage];
      const parts: string[] = [];
      const spokenParts: string[] = [];
      if (state === 'passed') {
        parts.push(formatDuration(split));
        spokenParts.push(split === null || split === undefined ? 'time unknown' : `in ${formatDuration(split)}`);
      }
      if (best !== undefined) {
        parts.push(`best ${formatDuration(best)}`);
        spokenParts.push(`best ${formatDuration(best)}`);
        if (typeof split === 'number') parts.push(delta(split, best));
      }
      timesNode.textContent = parts.join(' · ');
      timesNode.dataset.trend = typeof split === 'number' && best !== undefined
        ? (split < best ? 'faster' : split > best ? 'slower' : 'even') : '';
      const status = state === 'passed' ? 'passed' : state === 'current' ? 'current stage' : 'not passed yet';
      row.setAttribute('aria-label', [`${stageName(stage)}: ${status}`, ...spokenParts].join(', '));
      if (sameRep && state === 'passed' && !this.shownPassed.has(stage)) {
        seal.classList.remove('dojo-stamping');
        void seal.offsetWidth;
        seal.classList.add('dojo-stamping');
      }
    }
    this.shownRep = repId;
    this.shownPassed = new Set(passed);
  }
}
