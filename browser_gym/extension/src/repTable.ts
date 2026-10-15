import type { AttemptRecord } from './attempts';
import { stageName, type StageOrder } from './seals';
import { finishTime, formatDuration, repSplits } from './splits';

/** The help a rep used: hints shown and whether the reference solution was revealed. */
export function helpUsed(rep: Pick<AttemptRecord, 'assistance'>): string {
  const hints = rep.assistance.filter(event => event.kind === 'hint').length;
  const solution = rep.assistance.some(event => event.kind === 'solution');
  return [hints ? `${hints} hint${hints === 1 ? '' : 's'}` : '', solution ? 'solution revealed' : ''].filter(Boolean).join(' + ');
}

/** The rep with the fastest known finish, if any rep finished with timing. */
export function bestRepId(reps: AttemptRecord[], stages: string[], order: StageOrder): string | undefined {
  let best: { id: string; time: number } | undefined;
  for (const rep of reps) {
    const time = finishTime(rep, stages, order);
    if (typeof time === 'number' && (!best || time < best.time)) best = { id: rep.id, time };
  }
  return best?.id;
}

/** Every rep of the workout, with its date, split per stage, hints used and notes. */
export function repTable(reps: AttemptRecord[], stages: string[], order: StageOrder, activeId?: string): HTMLTableElement {
  const table = document.createElement('table');
  table.className = 'dojo-rep-table';
  const caption = document.createElement('caption');
  caption.textContent = 'Splits per rep';
  const head = table.createTHead().insertRow();
  for (const label of ['Rep', 'Date', ...stages.map(stageName), 'Help', 'Notes']) {
    const cell = document.createElement('th');
    cell.scope = 'col';
    cell.textContent = label;
    head.append(cell);
  }
  table.prepend(caption);
  const best = bestRepId(reps, stages, order);
  const body = table.createTBody();
  for (const [index, rep] of reps.entries()) {
    const row = body.insertRow();
    if (rep.id === best) row.dataset.best = 'true';
    if (rep.id === activeId) row.dataset.current = 'true';
    const name = document.createElement('th');
    name.scope = 'row';
    name.textContent = `Rep ${index + 1}${rep.id === activeId ? ' (current)' : ''}${rep.id === best ? ' · best' : ''}`;
    row.append(name);
    row.insertCell().textContent = new Date(rep.createdAt).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
    const splits = repSplits(rep, stages, order);
    for (const stage of stages) row.insertCell().textContent = stage in splits ? formatDuration(splits[stage]) : '';
    row.insertCell().textContent = helpUsed(rep) || '—';
    row.insertCell().textContent = rep.notes.split('\n')[0];
  }
  return table;
}
