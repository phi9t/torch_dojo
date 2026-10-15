import { describe, expect, it } from 'vitest';
import { failurePlaces } from './failure';

const VERIFIER = `def verify_baseline_overlap(interval_union):
    x = np.array([[0., 2.], [1., 3.]])
    assert interval_union(x) == 3.0
    assert interval_union(np.empty((0, 2))) == 0.0`;

const ANSWER = `import numpy as np

def interval_union(intervals):
    ordered = intervals[np.argsort(intervals[:, 0])]
    start, end = ordered[0]
    return float(end - start)`;

function traceback(...frames: Array<[string, number, string]>) {
  return 'Traceback (most recent call last):\n' +
    frames.map(([file, line, fn]) => `  File "${file}", line ${line}, in ${fn}\n    whatever\n`).join('') +
    'IndexError: index 0 is out of bounds\n';
}

describe('failurePlaces', () => {
  it('locates shared-program helpers without changing how archived checks are read', () => {
    const sources = { verifiers: { verify_baseline_overlap: VERIFIER }, program: 'import numpy as np\n\ndef helper():\n    assert False' };
    const fresh = failurePlaces(traceback(['checker_program.py', 4, 'helper']), sources);
    expect(fresh[0].lines.find(line => line.failing)?.text).toBe('    assert False');
    const archived = failurePlaces(traceback(['checker.py', 3, 'verify_baseline_overlap']), sources);
    expect(archived[0].lines.find(line => line.failing)?.text).toBe('    assert interval_union(x) == 3.0');
  });

  it('points at the check line that failed, with the lines around it', () => {
    const places = failurePlaces(traceback(['/home/pyodide/check_runtime.py', 208, '_run_bundle'], ['checker.py', 3, 'verify_baseline_overlap']),
      { verifiers: { verify_baseline_overlap: VERIFIER }, submitted: ANSWER });
    expect(places).toEqual([{
      label: 'In the check verify_baseline_overlap, line 3',
      lines: [
        { number: 1, text: 'def verify_baseline_overlap(interval_union):', failing: false },
        { number: 2, text: '    x = np.array([[0., 2.], [1., 3.]])', failing: false },
        { number: 3, text: '    assert interval_union(x) == 3.0', failing: true },
        { number: 4, text: '    assert interval_union(np.empty((0, 2))) == 0.0', failing: false }
      ]
    }]);
  });

  it('puts the line of the answer first when the answer raised, then the check that called it', () => {
    const places = failurePlaces(traceback(['checker.py', 4, 'verify_baseline_overlap'], ['solution.py', 5, 'interval_union']),
      { verifiers: { verify_baseline_overlap: VERIFIER }, submitted: ANSWER });
    expect(places.map(place => place.label)).toEqual([
      'In your answer, line 5 (interval_union)',
      'In the check verify_baseline_overlap, line 4'
    ]);
    expect(places[0].lines.find(line => line.failing)?.text).toBe('    start, end = ordered[0]');
  });

  it('reads helper frames from the shared helpers, and skips frames it has no source for', () => {
    const places = failurePlaces(traceback(['checker.py', 2, 'verify_x'], ['checker_helpers.py', 2, '_chain'], ['<frozen>', 1, 'inner']),
      { verifiers: {}, helpers: 'def _chain():\n    raise ValueError("bad")' });
    expect(places.map(place => place.label)).toEqual(['In the check helper _chain, line 2']);
  });
});
