import { describe, expect, it } from 'vitest';
import { bestSplits, finishTime, formatDuration, repSplits, type SplitRep } from './splits';

const STAGES = ['baseline', 'followup', 'production'];

function receipt(stage: string, status: string, activeSecondsAtCheck?: number) {
  return { status, result: { stage }, activeSecondsAtCheck };
}

describe('repSplits', () => {
  it('takes each stage split from the first passing receipt at that stage', () => {
    const rep: SplitRep = {
      receipts: [receipt('baseline', 'failed', 30), receipt('baseline', 'passed', 70), receipt('baseline', 'passed', 90)]
    };
    expect(repSplits(rep, STAGES, 'cumulative')).toEqual({ baseline: 70 });
  });

  it('counts a cumulative pass as passing every earlier stage not yet split', () => {
    const rep: SplitRep = { receipts: [receipt('baseline', 'passed', 40), receipt('production', 'passed', 300)] };
    expect(repSplits(rep, STAGES, 'cumulative')).toEqual({ baseline: 40, followup: 300, production: 300 });
  });

  it('splits independent checkpoints only at the checked checkpoint', () => {
    const rep: SplitRep = { receipts: [receipt('forward', 'passed', 50)] };
    expect(repSplits(rep, ['init', 'forward'], 'independent')).toEqual({ forward: 50 });
  });

  it('marks a split unknown when its first passing receipt predates timing', () => {
    const rep: SplitRep = { receipts: [receipt('baseline', 'passed'), receipt('baseline', 'passed', 80)] };
    expect(repSplits(rep, STAGES, 'cumulative')).toEqual({ baseline: null });
  });
});

describe('finishTime', () => {
  it('is the final stage split once the last stage passes', () => {
    expect(finishTime({ receipts: [receipt('baseline', 'passed', 40)] }, STAGES, 'cumulative')).toBeUndefined();
    expect(finishTime({ receipts: [receipt('production', 'passed', 300)] }, STAGES, 'cumulative')).toBe(300);
  });

  it('is the latest checkpoint pass when checkpoints are independent', () => {
    const rep: SplitRep = { receipts: [receipt('forward', 'passed', 50), receipt('init', 'passed', 200)] };
    expect(finishTime(rep, ['init', 'forward'], 'independent')).toBe(200);
    expect(finishTime({ receipts: [receipt('forward', 'passed', 50)] }, ['init', 'forward'], 'independent')).toBeUndefined();
  });

  it('infers the stage of receipts saved without one from their verifier names', () => {
    const legacy = { status: 'passed', result: { checks: [{ name: 'verify_baseline' }] } };
    expect(repSplits({ receipts: [legacy] }, STAGES, 'cumulative')).toEqual({ baseline: null });
  });
});

describe('bestSplits', () => {
  it('is the minimum known split per stage across reps', () => {
    const reps: SplitRep[] = [
      { receipts: [receipt('followup', 'passed', 200)] },
      { receipts: [receipt('baseline', 'passed', 90), receipt('followup', 'passed', 150)] },
      { receipts: [receipt('production', 'passed')] }
    ];
    expect(bestSplits(reps, STAGES, 'cumulative')).toEqual({ baseline: 90, followup: 150 });
  });

  it('has no best for a stage nobody has passed with timing', () => {
    expect(bestSplits([{ receipts: [receipt('baseline', 'passed')] }], STAGES, 'cumulative')).toEqual({});
  });
});

describe('formatDuration', () => {
  it('formats minutes and seconds, and hours when needed', () => {
    expect(formatDuration(0)).toBe('0:00');
    expect(formatDuration(65.9)).toBe('1:05');
    expect(formatDuration(3725)).toBe('1:02:05');
  });

  it('shows an unknown split as a dash, never 0:00', () => {
    expect(formatDuration(null)).toBe('—');
    expect(formatDuration(undefined)).toBe('—');
  });
});
