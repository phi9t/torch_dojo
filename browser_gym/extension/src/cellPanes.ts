/**
 * Where a workout notebook cell lives on the Bench.
 *
 * - `brief`, `explore`, `drill`: the left pane's tabs
 * - `answer`: the answer cell, alone in the right pane
 * - `scratch`: the collapsed scratch area under the answer
 * - `answer-note`: generated guidance the Bench replaces with its own label
 */
export type Pane = 'brief' | 'explore' | 'drill' | 'answer' | 'answer-note' | 'scratch';

export interface PaneCell {
  cellType: string;
  source: string;
  role?: string;
}

const CODE_ROLES: Record<string, Pane> = {
  solution: 'answer',
  exploration: 'explore',
  drill: 'drill',
  scratch: 'scratch'
};

export function paneForCell(cell: PaneCell): Pane {
  if (cell.cellType === 'code') return CODE_ROLES[cell.role ?? ''] ?? 'scratch';
  // Instruction cells carry no finer role, and reps saved before roles existed
  // carry none at all, so their generated opening line decides.
  if (cell.source.startsWith('### Explore')) return 'explore';
  if (cell.source.startsWith('Edit the solution below')) return 'answer-note';
  if (cell.source.startsWith('Scratch space:')) return 'scratch';
  return 'brief';
}
