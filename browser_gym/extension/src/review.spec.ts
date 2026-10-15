import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { ReviewStore, REVIEW_KEY, reviewQueue, schedule, validateArchive, type Motif, type Rating, type ReviewState } from './review';
const motifs: Motif[] = JSON.parse(readFileSync(new URL('../../../torch_judge/motifs.json', import.meta.url), 'utf8'));
const fixtures: Array<{previous: ReviewState | null; rating: Rating; now: number; expected: ReviewState}> = JSON.parse(readFileSync(new URL('./fixtures/review.json', import.meta.url), 'utf8'));
function memory() {
  const data = new Map<string, string>();
  return { getItem: (key: string) => data.get(key) ?? null, setItem: (key: string, value: string) => { data.set(key, value); } };
}
describe('motif reviews', () => {
  it('matches the native scheduling contract', () => {
    for (const f of fixtures) expect(schedule(f.previous ?? undefined, f.rating, f.now)).toEqual(f.expected);
  });
  it('puts overdue cards before fresh ones and excludes future reviews', () => {
    const store = new ReviewStore(motifs, memory());
    store.rate(motifs[0].id, 'good', 1000);
    store.rate(motifs[1].id, 'again', 1000);
    const queue = reviewQueue(motifs, store.load(), 1600);
    expect(queue[0]).toEqual(motifs[1]);
    expect(queue).not.toContain(motifs[0]);
    expect(queue[1]).toEqual(motifs[2]);
  });
  it('reloads state and preserves other tabs and newer ratings on import', () => {
    const storage = memory();
    const one = new ReviewStore(motifs, storage), two = new ReviewStore(motifs, storage);
    one.rate(motifs[0].id, 'hard', 1000);
    const old = one.load();
    two.rate(motifs[1].id, 'good', 1001);
    one.rate(motifs[0].id, 'good', 2000);
    two.import(old);
    expect(Object.keys(two.load().cards)).toHaveLength(2);
    expect(two.load().cards[motifs[0].id].reviewed).toBe(2000);
  });
  it('rejects the whole invalid import and preserves corrupt storage', () => {
    const storage = memory(), store = new ReviewStore(motifs, storage);
    store.rate(motifs[0].id, 'hard', 1000);
    const before = storage.getItem(REVIEW_KEY);
    expect(() => store.import({ version: 1, cards: { unknown: schedule(undefined, 'good', 0) } })).toThrow();
    expect(storage.getItem(REVIEW_KEY)).toBe(before);
    storage.setItem(REVIEW_KEY, '{broken');
    expect(() => store.rate(motifs[0].id, 'good', 1000)).toThrow();
    expect(storage.getItem(REVIEW_KEY)).toBe('{broken');
  });
  it('does not claim success when saving is blocked', () => {
    const store = new ReviewStore(motifs, { getItem: () => null, setItem: () => { throw new Error('Quota'); } });
    expect(() => store.rate(motifs[0].id, 'good', 0)).toThrow('Quota');
  });
  it('rejects invalid timestamps and intervals', () => {
    for (const due of [NaN, Infinity, -1, true]) {
      expect(() => validateArchive({ version: 1, cards: { [motifs[0].id]: { ...schedule(undefined, 'good', 0), due } } }, motifs)).toThrow();
    }
  });
});
