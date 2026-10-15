import { afterEach, describe, expect, it, vi } from 'vitest';
import { ActiveTimer } from './activeTimer';

function timer() {
  vi.useFakeTimers();
  vi.stubGlobal('document', {
    visibilityState: 'visible', addEventListener() {},
    createElement: () => ({ dataset: {}, hidden: false, append() {}, setAttribute() {} })
  });
  vi.stubGlobal('window', {
    localStorage: { getItem: () => null }, setInterval
  });
  return new ActiveTimer();
}
afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals(); });

describe('practice time', () => {
  it('keeps counting while repeated renders unpause an already active rep', () => {
    const active = timer();
    for (let i = 0; i < 10; i++) {
      vi.advanceTimersByTime(500);
      active.setPaused(false);
    }
    expect(active.value).toBe(5);
  });
  it('does not charge notebook loading time to the learner', () => {
    const active = timer();
    vi.advanceTimersByTime(2000);
    active.setPaused(true);
    vi.advanceTimersByTime(10000);
    expect(active.value).toBe(2);
    active.setPaused(false);
    vi.advanceTimersByTime(1000);
    expect(active.value).toBe(3);
  });
});
