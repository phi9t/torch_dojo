export interface CheckSample {
  function: string;
  call: number;
  argument_names?: string[];
  args: unknown;
  kwargs: unknown;
  actual?: unknown;
  expected?: unknown;
  error?: string;
  expected_error?: string;
  replayable: boolean;
}

export interface CheckResult {
  status: 'passed' | 'failed';
  checks: Array<{ name: string; status: string; error?: string; traceback?: string; locals?: Record<string, string>; elapsed_ms?: number; evidence?: unknown; samples?: CheckSample[]; sample_calls?: number }>;
  error: string | null;
  traceback?: string;
  checker_version: string;
  runtime: string;
  package_identity?: Record<string, string>;
  fixture_id: string;
  stage?: string;
  checkpoint?: string;
  evidence?: unknown;
  reference_overrides?: string[];
  substitutions?: Array<{ function: string; checkpoint: string; independent_credit: boolean }>;
  owned_checkpoint_passed?: boolean;
}

/**
 * The checker worker URL is resolved relative to the page. Tests inject a
 * different base via the optional `baseHref` parameter so this module can be
 * exercised outside a browser page without reaching for `window`.
 */
function resolveWorkerUrl(baseHref?: string): URL {
  if (baseHref) return new URL('../gym-assets/checker-worker.js', baseHref);
  if (typeof window !== 'undefined') return new URL('../gym-assets/checker-worker.js', window.location.href);
  throw new Error('Cannot resolve checker worker URL: no baseHref and no window.location');
}

export class CheckTimeoutError extends Error {
  readonly name = 'CheckTimeoutError';
  constructor(message = 'Check timed out before the worker returned results.') {
    super(message);
  }
}

export class CheckStoppedError extends Error {
  readonly name = 'CheckStoppedError';
  constructor(
    message: string,
    readonly source: string,
    readonly sha256: string
  ) {
    super(message);
  }
}

export interface CheckRun {
  /** Resolves with the verifier result on success; rejects on any failure. */
  readonly result: Promise<CheckResult>;
  /**
   * Hard-stop the in-flight worker immediately. Rejects `result` with
   * `CheckStoppedError` carrying the original source snapshot so the caller
   * can write a source-bound stopped receipt. Safe to call more than once.
   */
  abort(reason?: string): void;
}

interface RunCheckOptions {
  /** Milliseconds before the controller hard-terminates the worker. Default 120 s. */
  timeoutMs?: number;
  /** Explicit base URL for resolving the worker script; mostly for tests. */
  baseHref?: string;
}

/**
 * Launch one disposable Pyodide worker to check `source`, with a
 * controller-owned deadline and explicit Stop support.
 *
 * Guarantees:
 * - The worker is always terminated (success, error, timeout, or Stop).
 * - Timeout rejects with `CheckTimeoutError`.
 * - `run.abort()` rejects with a source-bound `CheckStoppedError`.
 * - Responses whose `requestId` does not match the current invocation are
 *   ignored — a late reply from a terminated prior worker can never feed the
 *   wrong check's result.
 */
export function runCheck(
  source: string,
  sha256: string,
  bundleUrl: URL,
  stage: string,
  optsOrTimeoutMs: RunCheckOptions | number = {}
): CheckRun {
  const options: RunCheckOptions = typeof optsOrTimeoutMs === 'number'
    ? { timeoutMs: optsOrTimeoutMs }
    : optsOrTimeoutMs;
  const timeoutMs = options.timeoutMs ?? 120_000;

  const workerUrl = resolveWorkerUrl(options.baseHref);
  // eslint-disable-next-line @typescript-eslint/no-unsafe-return, @typescript-eslint/no-explicit-any
  const worker: Worker = new (globalThis.Worker as any)(workerUrl, { type: 'module' });
  const requestId =
    (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function')
      ? crypto.randomUUID()
      : `req-${Date.now()}-${Math.random().toString(16).slice(2)}`;

  let settled = false;
  let abortCalled = false;
  let externalResolve: (value: CheckResult) => void;
  let externalReject: (reason: unknown) => void;
  const result = new Promise<CheckResult>((resolve, reject) => {
    externalResolve = resolve;
    externalReject = reject;
  });

  const timer =
    (typeof window !== 'undefined' ? window.setTimeout : setTimeout)(() => {
      if (settled) return;
      settled = true;
      cleanupAndTerminate();
      externalReject!(new CheckTimeoutError());
    }, timeoutMs);

  function clearTimer() {
    (typeof window !== 'undefined' ? window.clearTimeout : clearTimeout)(timer);
  }

  function cleanupAndTerminate() {
    clearTimer();
    try { worker.onmessage = null; } catch { /* noop */ }
    try { worker.onerror = null; } catch { /* noop */ }
    worker.terminate();
  }

  worker.onmessage = event => {
    if (event.data?.requestId !== requestId) return; // ignore stale / wrong-ID messages
    if (settled) return;
    settled = true;
    cleanupAndTerminate();
    if (event.data.error) {
      externalReject!(new Error(String(event.data.error)));
    } else {
      externalResolve!(event.data.result as CheckResult);
    }
  };
  worker.onerror = event => {
    if (settled) return;
    settled = true;
    cleanupAndTerminate();
    externalReject!(new Error(event.message || 'Checker worker failed.'));
  };

  // Kick off the worker. If abort() was called synchronously before this
  // function returns, the postMessage is harmless: the subsequent abort path
  // terminates before anything can resolve.
  worker.postMessage({ requestId, source, sha256, bundleUrl: bundleUrl.href, stage });

  return {
    result,
    abort(reason = 'Check stopped by the learner before the worker completed.') {
      if (abortCalled) return;
      abortCalled = true;
      if (settled) return;
      settled = true;
      cleanupAndTerminate();
      externalReject!(new CheckStoppedError(reason, source, sha256));
    }
  };
}
