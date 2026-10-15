export interface NotebookCell {
  cell_type: string;
  source: string | string[];
  metadata?: {
    dojo?: { role?: string; exercise_id?: string };
    jupyter?: { source_hidden?: boolean };
  };
}

export interface NotebookSnapshot {
  metadata?: { dojo?: { exercise_id?: string } };
  cells: NotebookCell[];
}

export interface CapturedSolution {
  source: string;
  sha256: string;
  /**
   * Notebook cell indices (0-based, document order) that the concatenated
   * `source` was built from. A receipt carrying `sourceBound` is "source-bound":
   * it records exactly which notebook cells contributed to the checked source,
   * even after subsequent edits, Stop, or timeout. Stop/timeout receipts must
   * always carry this field so the learner can recover which cells were
   * submitted to the terminated worker.
   */
  sourceBound: number[];
}

export async function captureSolution(notebook: NotebookSnapshot, exerciseId: string): Promise<CapturedSolution> {
  if (notebook.metadata?.dojo?.exercise_id !== exerciseId) {
    throw new Error('Open the active exercise notebook before checking.');
  }
  const sourceBound: number[] = [];
  const markedContents: string[] = [];
  notebook.cells.forEach((cell, index) => {
    if (cell.metadata?.dojo?.role === 'solution') {
      if (cell.cell_type !== 'code' || cell.metadata?.dojo?.exercise_id !== exerciseId) {
        return; // will be caught by the validation below
      }
      sourceBound.push(index);
      markedContents.push(Array.isArray(cell.source) ? cell.source.join('') : cell.source);
    }
  });
  if (!sourceBound.length) {
    throw new Error('This notebook has no valid solution cells for the active exercise.');
  }
  const source = markedContents.join('\n\n');
  const bytes = new TextEncoder().encode(source);
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  const sha256 = Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('');
  return { source, sha256, sourceBound };
}
