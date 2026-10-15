import { isFinished, resultStage, type StageOrder, type StagedResult } from './seals';

/**
 * A split is the rep's active time when a stage first passed (GLOSSARY: Split).
 * It is derived from receipts, never stored: the first passing receipt at a
 * stage carries `activeSecondsAtCheck`. Receipts saved before timing existed
 * carry none, so their split is unknown (`null`) and never becomes a best.
 */
export type Splits = Record<string, number | null>;

export interface SplitRep {
  receipts: Array<{ status: string; result?: StagedResult; activeSecondsAtCheck?: number }>;
}

export function repSplits(rep: SplitRep, stages: string[], order: StageOrder): Splits {
  const splits: Splits = {};
  for (const receipt of rep.receipts) {
    if (receipt.status !== 'passed') continue;
    const index = resultStage(stages, receipt.result);
    if (index < 0) continue;
    const passed = order === 'independent' ? [stages[index]] : stages.slice(0, index + 1);
    for (const stage of passed) {
      if (!(stage in splits)) splits[stage] = receipt.activeSecondsAtCheck ?? null;
    }
  }
  return splits;
}

/**
 * The rep's finish time: when its last stage passed, or its last checkpoint
 * when they pass independently. `undefined` until it finishes; `null` if the
 * finishing split predates timing.
 */
export function finishTime(rep: SplitRep, stages: string[], order: StageOrder): number | null | undefined {
  const splits = repSplits(rep, stages, order);
  if (!isFinished(stages, new Set(Object.keys(splits)), order)) return undefined;
  const times = stages.map(stage => splits[stage]);
  if (order === 'cumulative') return times[times.length - 1];
  return times.some(time => time === null) ? null : Math.max(...(times as number[]));
}

/** The fastest known split per stage across reps. */
export function bestSplits(reps: SplitRep[], stages: string[], order: StageOrder): Record<string, number> {
  const best: Record<string, number> = {};
  for (const rep of reps) {
    for (const [stage, split] of Object.entries(repSplits(rep, stages, order))) {
      if (split !== null && (best[stage] === undefined || split < best[stage])) best[stage] = split;
    }
  }
  return best;
}

export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return '—';
  const total = Math.floor(seconds);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = String(total % 60).padStart(2, '0');
  return hours ? `${hours}:${String(minutes).padStart(2, '0')}:${secs}` : `${minutes}:${secs}`;
}
