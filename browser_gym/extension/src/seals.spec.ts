import { describe, expect, it } from 'vitest';
import { passedStages } from './seals';

const STAGES = ['baseline', 'followup', 'production'];

function receipt(stage: string | undefined, status: 'passed' | 'failed' | 'stopped' = 'passed') {
  return { status, result: { stage } };
}

describe('passedStages', () => {
  it('is empty before any passing check', () => {
    expect(passedStages(STAGES, [], 'cumulative')).toEqual(new Set());
    expect(passedStages(STAGES, [receipt('baseline', 'failed'), receipt('followup', 'stopped')], 'cumulative'))
      .toEqual(new Set());
  });

  it('counts every earlier stage as passed when stages are cumulative', () => {
    expect(passedStages(STAGES, [receipt('followup')], 'cumulative')).toEqual(new Set(['baseline', 'followup']));
  });

  it('counts only the checked checkpoint when checkpoints are independent', () => {
    const checkpoints = ['init', 'forward', 'backward'];
    expect(passedStages(checkpoints, [receipt('backward')], 'independent')).toEqual(new Set(['backward']));
  });

  it('keeps a stage passed after a later failing check', () => {
    expect(passedStages(STAGES, [receipt('baseline'), receipt('followup', 'failed')], 'cumulative'))
      .toEqual(new Set(['baseline']));
  });

  it('ignores receipts whose stage is unknown', () => {
    expect(passedStages(STAGES, [receipt(undefined), receipt('nonsense')], 'cumulative')).toEqual(new Set());
  });
});
