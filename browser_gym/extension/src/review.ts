/** Portable motif review archive. All times and intervals are Unix seconds. */
export interface Motif {
  id: string; title: string; topic: string; workouts: string[];
  prompt: string; answer: string; probe: string; transfer: string;
}
export type Rating = 'again' | 'hard' | 'good';
export interface ReviewState { due: number; reviewed: number; interval: number; rating: Rating }
export interface ReviewArchive { version: 1; cards: Record<string, ReviewState> }
const DAY = 86400;
export const REVIEW_KEY = 'torch-dojo.motif-review.v1';
export function schedule(previous: ReviewState | undefined, rating: Rating, now: number): ReviewState {
  const interval = rating === 'again' ? 600 : rating === 'hard' ? DAY : Math.min(90 * DAY, Math.max(3 * DAY, (previous?.interval ?? 0) * 2));
  return { due: now + interval, reviewed: now, interval, rating };
}
export function validateArchive(value: unknown, motifs: Motif[]): ReviewArchive {
  const archive = value as ReviewArchive;
  if (!archive || archive.version !== 1 || !archive.cards || typeof archive.cards !== 'object' || Array.isArray(archive.cards)) throw new Error('Expected a motif review archive (version 1).');
  const known = new Set(motifs.map(m => m.id));
  for (const [id, state] of Object.entries(archive.cards)) {
    if (!known.has(id) || !state || !['again', 'hard', 'good'].includes(state.rating) ||
      ![state.due, state.reviewed, state.interval].every(n => typeof n === 'number' && Number.isFinite(n) && n >= 0) ||
      state.interval > 90 * DAY || state.due < state.reviewed) throw new Error(`Invalid review state for ${id}.`);
  }
  return archive;
}
export function reviewQueue(motifs: Motif[], archive: ReviewArchive, now: number): Motif[] {
  return [...motifs.filter(m => archive.cards[m.id]?.due <= now).sort((a, b) => archive.cards[a.id].due - archive.cards[b.id].due),
    ...motifs.filter(m => !archive.cards[m.id])];
}
/** Read before each mutation, preserving updates to other cards from other tabs. */
export class ReviewStore {
  constructor(private motifs: Motif[], private storage: Pick<Storage, 'getItem' | 'setItem'>) {}
  load(): ReviewArchive {
    const raw = this.storage.getItem(REVIEW_KEY);
    return raw === null ? { version: 1, cards: {} } : validateArchive(JSON.parse(raw), this.motifs);
  }
  rate(id: string, rating: Rating, now: number): ReviewState {
    const archive = this.load();
    const state = schedule(archive.cards[id], rating, now);
    archive.cards[id] = state;
    this.storage.setItem(REVIEW_KEY, JSON.stringify(validateArchive(archive, this.motifs)));
    return state;
  }
  import(value: unknown): void {
    const incoming = validateArchive(value, this.motifs);
    const archive = this.load();
    for (const [id, state] of Object.entries(incoming.cards)) {
      if (!archive.cards[id] || state.reviewed > archive.cards[id].reviewed) archive.cards[id] = state;
    }
    this.storage.setItem(REVIEW_KEY, JSON.stringify(archive));
  }
}
