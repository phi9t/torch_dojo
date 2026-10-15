import { describe, expect, it } from 'vitest';
import { captureSolution, type NotebookSnapshot } from './source';

describe('solution capture', () => {
  const notebook: NotebookSnapshot = {
    metadata: { dojo: { exercise_id: '0059' } },
    cells: [
      { cell_type: 'code', source: 'scratch_value = 99', metadata: {} },
      { cell_type: 'code', source: ['first'], metadata: { dojo: { role: 'solution', exercise_id: '0059' } } },
      { cell_type: 'code', source: 'print(scratch_value)', metadata: {} },
      { cell_type: 'code', source: 'second', metadata: { dojo: { role: 'solution', exercise_id: '0059' } } }
    ]
  };

  it('captures only marked cells in document order with an exact hash and source-bound cell indices', async () => {
    expect(await captureSolution(notebook, '0059')).toEqual({
      source: 'first\n\nsecond',
      sha256: '2dd7ae357f63549559e40349e8a1f5aae104da8f8074d61101699e759ac5863d',
      sourceBound: [1, 3]
    });
  });

  it('binds a single solution cell to its notebook index', async () => {
    const single: NotebookSnapshot = {
      metadata: { dojo: { exercise_id: '0059' } },
      cells: [
        { cell_type: 'markdown', source: 'intro' },
        { cell_type: 'code', source: 'only_solution = 1', metadata: { dojo: { role: 'solution', exercise_id: '0059' } } },
        { cell_type: 'code', source: 'scratch' }
      ]
    };
    const captured = await captureSolution(single, '0059');
    expect(captured.sourceBound).toEqual([1]);
  });

  it('rejects a missing or foreign answer', async () => {
    await expect(captureSolution({ ...notebook, cells: [] }, '0059')).rejects.toThrow(/no valid solution/);
    await expect(captureSolution(notebook, '0060')).rejects.toThrow(/active exercise/);
    await expect(captureSolution({ ...notebook, cells: [{ cell_type: 'code', source: 'x', metadata: { dojo: { role: 'solution', exercise_id: '0060' } } }] }, '0059')).rejects.toThrow(/no valid solution/);
  });
});
