import { describe, expect, it } from 'vitest';
import type { CheckResult } from './checker';
import { buildSquares, firstFailing, stageVerifiers } from './squares';

const STAGES = ['baseline', 'followup', 'production'];
const VERIFIERS = stageVerifiers(STAGES, [
  'verify_baseline_a',
  'verify_baseline_b',
  'verify_production_e',
  'verify_followup_c',
  'verify_followup_d'
]);

function result(stage: string, checks: Array<[string, 'passed' | 'failed', number?]>, error: string | null = null): CheckResult {
  return {
    status: checks.every(([, status]) => status === 'passed') && !error ? 'passed' : 'failed',
    checks: checks.map(([name, status, elapsed]) => ({ name, status, elapsed_ms: elapsed, error: status === 'failed' ? 'AssertionError' : undefined })),
    error,
    checker_version: 'v1',
    runtime: 'pyodide',
    fixture_id: 'f',
    stage
  };
}

describe('stageVerifiers', () => {
  it('groups verifier names by their stage prefix, in stage order', () => {
    expect(VERIFIERS).toEqual({
      baseline: ['verify_baseline_a', 'verify_baseline_b'],
      followup: ['verify_followup_c', 'verify_followup_d'],
      production: ['verify_production_e']
    });
  });
});

describe('buildSquares', () => {
  it('shows every verifier as not run before any check', () => {
    const squares = buildSquares(STAGES, VERIFIERS);
    expect(squares.map(square => square.state)).toEqual(['not-run', 'not-run', 'not-run', 'not-run', 'not-run']);
    expect(squares.map(square => square.stage)).toEqual(['baseline', 'baseline', 'followup', 'followup', 'production']);
  });

  it('locks the stages beyond the checked one', () => {
    const squares = buildSquares(STAGES, VERIFIERS, result('baseline', [['verify_baseline_a', 'passed', 3], ['verify_baseline_b', 'failed', 4]]));
    expect(squares.map(square => [square.name, square.state])).toEqual([
      ['verify_baseline_a', 'pass'],
      ['verify_baseline_b', 'fail'],
      ['verify_followup_c', 'locked'],
      ['verify_followup_d', 'locked'],
      ['verify_production_e', 'locked']
    ]);
    expect(squares[0].elapsedMs).toBe(3);
    expect(squares[1].check?.error).toBe('AssertionError');
  });

  it('marks a verifier of a checked stage that did not report as not run', () => {
    const squares = buildSquares(STAGES, VERIFIERS, result('followup', [['verify_baseline_a', 'passed']], 'SyntaxError'));
    expect(squares.slice(0, 4).map(square => square.state)).toEqual(['pass', 'not-run', 'not-run', 'not-run']);
  });

  it('keeps reported verifiers the bundle did not list, under the checked stage', () => {
    const squares = buildSquares(STAGES, {}, result('baseline', [['verify_something', 'failed']]));
    expect(squares).toEqual([expect.objectContaining({ stage: 'baseline', name: 'verify_something', state: 'fail' })]);
  });

  it('places a result saved without a stage at the furthest stage it reports', () => {
    const legacy = { ...result('baseline', [['verify_baseline_a', 'passed'], ['verify_followup_c', 'failed']]), stage: undefined };
    const squares = buildSquares(STAGES, VERIFIERS, legacy);
    expect(squares.map(square => square.state)).toEqual(['pass', 'not-run', 'fail', 'not-run', 'locked']);
  });

  it('treats independent checkpoints as unlocked and unchecked', () => {
    const checkpoints = ['init', 'forward'];
    const squares = buildSquares(checkpoints, {}, result('forward', [['forward_shapes', 'passed']]), 'independent');
    expect(squares).toEqual([
      expect.objectContaining({ stage: 'init', name: 'checkpoint_init', state: 'not-run' }),
      expect.objectContaining({ stage: 'forward', name: 'forward_shapes', state: 'pass' })
    ]);
  });
});

describe('firstFailing', () => {
  it('picks the first failing square, else none', () => {
    const squares = buildSquares(STAGES, VERIFIERS, result('followup', [
      ['verify_baseline_a', 'passed'], ['verify_baseline_b', 'passed'], ['verify_followup_c', 'failed'], ['verify_followup_d', 'failed']
    ]));
    expect(firstFailing(squares)?.name).toBe('verify_followup_c');
    expect(firstFailing(buildSquares(STAGES, VERIFIERS))).toBeUndefined();
  });
});

describe('buildSquares for independent checkpoints', () => {
  const CHECKPOINTS = ['init', 'forward', 'backward'];

  it('shows each checkpoint as one square not run until it is checked', () => {
    const squares = buildSquares(CHECKPOINTS, {}, undefined, 'independent');
    expect(squares.map(square => [square.stage, square.name, square.state])).toEqual([
      ['init', 'checkpoint_init', 'not-run'],
      ['forward', 'checkpoint_forward', 'not-run'],
      ['backward', 'checkpoint_backward', 'not-run']
    ]);
  });

  it('keeps the latest result of every other checkpoint beside the one just checked', () => {
    const forward = result('forward', [['checkpoint_forward', 'failed']]);
    const init = result('init', [['checkpoint_init', 'passed']]);
    const squares = buildSquares(CHECKPOINTS, {}, forward, 'independent', [init]);
    expect(squares.map(square => [square.name, square.state, Boolean(square.earlier)])).toEqual([
      ['checkpoint_init', 'pass', true],
      ['checkpoint_forward', 'fail', false],
      ['checkpoint_backward', 'not-run', false]
    ]);
    // Only the check just run picks the first failing square.
    expect(firstFailing(squares)?.name).toBe('checkpoint_forward');
    expect(firstFailing(buildSquares(CHECKPOINTS, {}, init, 'independent', [forward]))).toBeUndefined();
  });

  it('carries a checkpoint result’s reference substitutions on its squares', () => {
    const assisted = {
      ...result('backward', [['substitution_loss_and_grads', 'passed'], ['checkpoint_backward', 'passed']]),
      substitutions: [{ function: 'loss_and_grads', checkpoint: 'backward', independent_credit: false }]
    };
    const squares = buildSquares(CHECKPOINTS, {}, assisted, 'independent');
    expect(squares.filter(square => square.stage === 'backward').map(square => square.substitutions?.[0]?.function))
      .toEqual(['loss_and_grads', 'loss_and_grads']);
  });
});
