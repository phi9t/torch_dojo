import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { runCheck, type CheckRun } from './checker';

/**
 * A minimal fake Worker implementation we drive from the test.
 * `postMessage` to the worker is captured in `inbox`;
 * the test can call `respond(result | error)` or `fireError(msg)`
 * to simulate worker → main events.
 * `terminateCalls` records each `terminate()` call.
 */
class FakeWorker {
  inbox: Array<Record<string, unknown>> = [];
  onmessage: ((ev: { data: Record<string, unknown> }) => void) | null = null;
  onerror: ((ev: { message: string }) => void) | null = null;
  terminateCalls = 0;
  constructor() {}
  postMessage(data: Record<string, unknown>) {
    this.inbox.push(data);
    // **Do not auto-dispatch** — the test drives responses manually via
    // respond()/fireError(). This prevents the inbox copy from being
    // immediately consumed by the onmessage handler before the test inspects it.
  }
  respond(data: Record<string, unknown>) {
    queueMicrotask(() => this.onmessage?.({ data }));
  }
  fireError(message: string) {
    queueMicrotask(() => this.onerror?.({ message }));
  }
  terminate() {
    this.terminateCalls++;
  }
}

let fakeWorker: FakeWorker | null = null;
const OriginalWorker = globalThis.Worker;

beforeEach(() => {
  fakeWorker = null;
  // Replace global Worker with a factory that returns our fake.
  // runCheck constructs via `new Worker(url, opts)`; we return the singleton.
  globalThis.Worker = vi.fn((_url: string | URL, _opts?: WorkerOptions) => {
    fakeWorker = new FakeWorker();
    return fakeWorker as unknown as globalThis.Worker;
  }) as typeof globalThis.Worker;
  vi.useFakeTimers();
});

afterEach(() => {
  globalThis.Worker = OriginalWorker;
  vi.useRealTimers();
  vi.restoreAllMocks();
});

function currentInbox() {
  if (!fakeWorker) throw new Error('Fake worker not instantiated');
  return fakeWorker.inbox;
}

describe('runCheck (abortable checker client)', () => {
  const bundleUrl = new URL('http://example/bundle.json');
  const baseHref = 'http://localhost/lab/index.html';

  it('resolves with the worker result and always terminates the worker', async () => {
    const run: CheckRun = runCheck('answer = 1', 'abc123', bundleUrl, 'baseline', { timeoutMs: 30_000, baseHref });
    await new Promise<void>(resolve => { queueMicrotask(() => resolve()); });
    expect(currentInbox().length).toBe(1);
    const requestId = currentInbox()[0].requestId as string;

    fakeWorker!.respond({
      requestId,
      result: {
        status: 'passed', checks: [], error: null,
        checker_version: 'v1', runtime: 'pyodide', fixture_id: 'canonical-0059'
      }
    });
    const result = await run.result;
    expect(result.status).toBe('passed');
    expect(fakeWorker!.terminateCalls).toBe(1);
  });

  it('rejects with CheckTimeoutError when the deadline fires before a response', async () => {
    let captured: unknown = null;
    const run: CheckRun = runCheck('while True: pass', 'sha256x', bundleUrl, 'production', { timeoutMs: 50, baseHref });
    // Attach a rejection handler FIRST so node never sees an unhandled rejection.
    run.result.catch(err => { captured = err; });
    await new Promise<void>(resolve => { queueMicrotask(() => resolve()); });
    expect(currentInbox().length).toBe(1);

    // Advance timers, then assert the captured rejection.
    await vi.advanceTimersByTimeAsync(200);
    // Give microtasks (the timer callback + externalReject) a chance to fire.
    await new Promise<void>(resolve => { queueMicrotask(() => resolve()); });
    expect(captured).toMatchObject({ name: 'CheckTimeoutError' });
    await expect(run.result).rejects.toMatchObject({ name: 'CheckTimeoutError' });
    expect(fakeWorker!.terminateCalls).toBe(1);
  });

  it('rejects with CheckStoppedError with source-bound details when abort() is called before completion', async () => {
    const source = 'runaway_loop()';
    const sha = 'deadbeefcafebabe000000000000000000000000000000000000000000000000';
    const run: CheckRun = runCheck(source, sha, bundleUrl, 'baseline', { timeoutMs: 120_000, baseHref });
    // Flush the postMessage microtask so inbox is populated, but do NOT flush
    // all timers (that would fire the 120s timeout before we get to abort).
    await new Promise<void>(resolve => { queueMicrotask(() => resolve()); });
    expect(currentInbox().length).toBe(1);

    // Stop the check before the worker replies.
    run.abort();
    // Give the abort microtask (settlement) a chance to run.
    await new Promise<void>(resolve => { queueMicrotask(() => resolve()); });
    await expect(run.result).rejects.toMatchObject({
      name: 'CheckStoppedError',
      // Stopped receipts must be source-bound: carry the originally submitted source + hash.
      source,
      sha256: sha
    });

    // abort() must terminate the worker immediately (not wait for timeout).
    expect(fakeWorker!.terminateCalls).toBeGreaterThanOrEqual(1);
  });

  it('keeps a second check independent: a stopped first check does not poison the next worker', async () => {
    // First check → stopped.
    const first = runCheck('infinite_loop()', 'sha-a', bundleUrl, 'baseline', { timeoutMs: 120_000, baseHref });
    await new Promise<void>(resolve => { queueMicrotask(() => resolve()); });
    const firstWorker = fakeWorker!;
    first.abort();
    await new Promise<void>(resolve => { queueMicrotask(() => resolve()); });
    await expect(first.result).rejects.toMatchObject({ name: 'CheckStoppedError' });
    expect(firstWorker.terminateCalls).toBeGreaterThanOrEqual(1);

    // Second check → succeeds in a brand new worker.
    const second = runCheck('ok = 1', 'sha-b', bundleUrl, 'baseline', { timeoutMs: 120_000, baseHref });
    await new Promise<void>(resolve => { queueMicrotask(() => resolve()); });
    const secondWorker = fakeWorker!;
    // A second worker must have been created.
    expect(secondWorker).not.toBe(firstWorker);
    const requestId = secondWorker.inbox[0].requestId as string;
    secondWorker.respond({
      requestId,
      result: {
        status: 'passed', checks: [{ name: 'c1', status: 'passed', elapsed_ms: 1 }], error: null,
        checker_version: 'v1', runtime: 'pyodide', fixture_id: 'canonical-0059'
      }
    });
    await expect(second.result).resolves.toMatchObject({ status: 'passed' });
    expect(secondWorker.terminateCalls).toBe(1);
  });

  it('ignores stale responses from a prior (terminated) worker', async () => {
    const run = runCheck('x', 's1', bundleUrl, 'baseline', { timeoutMs: 120_000, baseHref });
    await new Promise<void>(resolve => { queueMicrotask(() => resolve()); });
    const goodId = currentInbox()[0].requestId as string;

    // Simulate a stale, mismatched requestId arriving first.
    fakeWorker!.respond({ requestId: 'stale-id-nobody-asked-for', result: { nope: true } });
    // Then the real, matching response.
    fakeWorker!.respond({
      requestId: goodId,
      result: {
        status: 'passed', checks: [], error: null,
        checker_version: 'v1', runtime: 'pyodide', fixture_id: 'canonical-0059'
      }
    });

    const result = await run.result;
    expect(result.status).toBe('passed');
    expect(fakeWorker!.terminateCalls).toBe(1);
  });
});
