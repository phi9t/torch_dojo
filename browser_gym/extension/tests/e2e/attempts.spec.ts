import { expect, test, type Page } from '@playwright/test';
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { answerPane, openTab } from './bench';

const evalSource = readFileSync(resolve(__dirname, '../../../../dojo/0059_stable_cross_entropy_eval.py'), 'utf8');
const referenceFunction = evalSource.match(/def softmax_cross_entropy[\s\S]*?(?=\n\n# %% \[verifiers\])/)?.[0];
if (!referenceFunction) throw new Error('Could not find canonical reference function');
const reference = `import numpy as np\n${referenceFunction}`;

function answerCell(page: Page) {
  // The Bench's answer pane when the workout is showing; otherwise another open notebook's answer, the code cell just before scratch.
  const bench = page.locator('#torch-dojo-bench:not(.lm-mod-hidden)');
  const otherNotebook = page.locator('#jp-main-dock-panel > .jp-NotebookPanel:not(.lm-mod-hidden)');
  return bench.getByRole('region', { name: 'Answer', exact: true })
    .locator('.jp-CodeCell').filter({ visible: true }).first().locator('.cm-content')
    .or(otherNotebook.locator('.jp-CodeCell').nth(-2).locator('.cm-content'));
}

function visibleNotebookHeading(page: Page) {
  return page.locator('#jp-main-dock-panel > :not(.lm-mod-hidden) .jp-MarkdownCell h1').first();
}

async function loadGym(page: Page, url = 'lab/?exercise=0059') {
  await page.goto(url);
  await expect(visibleNotebookHeading(page)).toContainText('0059', { timeout: 60_000 });
  const panel = page.locator('body');
  const status = page.locator('#torch-dojo-gym [role="status"]');
  const check = panel.getByRole('button', { name: /^Check (?!log)/ });
  await expect(panel).toBeAttached({ timeout: 60_000 });
  await expect(check).toBeAttached({ timeout: 60_000 });
  // A `path=` URL opens another notebook in front of the workout once it loads.
  const otherPath = new URL(url, 'http://gym.test').searchParams.get('path');
  if (otherPath) {
    await selectNotebook(page, otherPath.split('/').pop()!);
    await selectNotebook(page, '0059_stable_cross_entropy.ipynb');
  }
  await expect(check).toBeVisible({ timeout: 60_000 });
  await expect(check).toBeEnabled({ timeout: 60_000 });
  await openTab(page, 'Reps');
  return { panel, status, check };
}

async function replaceAnswer(page: Page, source: string) {
  await answerCell(page).click();
  await page.keyboard.press('ControlOrMeta+A');
  await page.keyboard.insertText(source);
}

async function installTransitionControls(page: Page) {
  await page.addInitScript(() => {
    type TransitionKind = 'resume' | 'new' | 'import';
    type CheckPhase = 'before-receipt' | 'after-receipt' | 'after-render';
    type RenderAttemptPhase = 'before-ui-write';
    type RenderHistoryPhase = 'after-list';
    let armedKind: TransitionKind | undefined;
    let armedPhase: string | undefined;
    let armedCheckPhase: CheckPhase | undefined;
    let armedRenderPhase: RenderAttemptPhase | undefined;
    let armedRenderHistoryPhase: RenderHistoryPhase | undefined;
    let failure:
      | { kind: TransitionKind; phase: string; message: string; remaining: number }
      | undefined;
    const testWindow = window as Window & {
      __dojoArmAttemptTransition?: (kind: TransitionKind, phase?: string) => void;
      __dojoFailAttemptTransition?: (kind: TransitionKind, phase: string, message: string) => void;
      __dojoObservedAttemptTransitions?: Array<{ kind: TransitionKind; attemptId: string; phase?: string }>;
      __dojoPendingAttemptTransition?: { kind: TransitionKind; attemptId: string; phase?: string };
      __dojoResolveAttemptTransition?: () => void;
      __dojoRunAutosaveTick?: () => Promise<void>;
      __dojoOpenNotebookPath?: (path: string) => Promise<void>;
      __dojoArmCheckGate?: (phase?: CheckPhase) => void;
      __dojoPendingCheckGate?: { attemptId: string; phase: CheckPhase };
      __dojoResolveCheckGate?: () => void;
      __dojoArmRenderAttemptGate?: (phase?: RenderAttemptPhase) => void;
      __dojoPendingRenderAttemptGate?: { attemptId: string; phase: RenderAttemptPhase };
      __dojoResolveRenderAttemptGate?: () => void;
      __dojoArmRenderHistoryGate?: (phase?: RenderHistoryPhase) => void;
      __dojoPendingRenderHistoryGate?: { attemptId?: string; phase: RenderHistoryPhase };
      __dojoResolveRenderHistoryGate?: () => void;
      __dojoPendingAutosaveTick?: Promise<unknown>;
      __torchDojoCheckGate?: (event: { attemptId: string; phase: CheckPhase }) => Promise<void> | void;
      __torchDojoRenderAttemptGate?: (event: { attemptId: string; phase: RenderAttemptPhase }) => Promise<void> | void;
      __torchDojoRenderHistoryGate?: (event: { attemptId?: string; phase: RenderHistoryPhase }) => Promise<void> | void;
      __torchDojoAttemptTransitionGate?: (event: {
        kind: TransitionKind;
        attemptId: string;
        phase?: string;
      }) => Promise<void> | void;
    };

    testWindow.__dojoObservedAttemptTransitions = [];
    testWindow.__dojoArmAttemptTransition = (kind, phase = 'before-swap') => {
      armedKind = kind;
      armedPhase = phase;
      testWindow.__dojoPendingAttemptTransition = undefined;
      testWindow.__dojoResolveAttemptTransition = undefined;
    };
    testWindow.__dojoFailAttemptTransition = (kind, phase, message) => {
      failure = { kind, phase, message, remaining: 1 };
    };
    testWindow.__torchDojoAttemptTransitionGate = event => {
      testWindow.__dojoObservedAttemptTransitions?.push(event);
      const eventPhase = event.phase ?? 'before-swap';
      if (failure && event.kind === failure.kind && eventPhase === failure.phase && failure.remaining > 0) {
        failure.remaining -= 1;
        throw new Error(failure.message);
      }
      if (event.kind !== armedKind || eventPhase !== (armedPhase ?? 'before-swap')) return undefined;
      return new Promise<void>(resolve => {
        testWindow.__dojoPendingAttemptTransition = event;
        testWindow.__dojoResolveAttemptTransition = () => {
          armedKind = undefined;
          armedPhase = undefined;
          testWindow.__dojoPendingAttemptTransition = undefined;
          testWindow.__dojoResolveAttemptTransition = undefined;
          resolve();
        };
      });
    };
    testWindow.__dojoArmCheckGate = (phase = 'before-receipt') => {
      armedCheckPhase = phase;
      testWindow.__dojoPendingCheckGate = undefined;
      testWindow.__dojoResolveCheckGate = undefined;
    };
    testWindow.__torchDojoCheckGate = event => {
      if (event.phase !== armedCheckPhase) return undefined;
      return new Promise<void>(resolve => {
        testWindow.__dojoPendingCheckGate = event;
        testWindow.__dojoResolveCheckGate = () => {
          armedCheckPhase = undefined;
          testWindow.__dojoPendingCheckGate = undefined;
          testWindow.__dojoResolveCheckGate = undefined;
          resolve();
        };
      });
    };
    testWindow.__dojoArmRenderAttemptGate = (phase = 'before-ui-write') => {
      armedRenderPhase = phase;
      testWindow.__dojoPendingRenderAttemptGate = undefined;
      testWindow.__dojoResolveRenderAttemptGate = undefined;
    };
    testWindow.__torchDojoRenderAttemptGate = event => {
      if (event.phase !== armedRenderPhase) return undefined;
      if (testWindow.__dojoPendingRenderAttemptGate) return undefined;
      return new Promise<void>(resolve => {
        testWindow.__dojoPendingRenderAttemptGate = event;
        testWindow.__dojoResolveRenderAttemptGate = () => {
          armedRenderPhase = undefined;
          testWindow.__dojoPendingRenderAttemptGate = undefined;
          testWindow.__dojoResolveRenderAttemptGate = undefined;
          resolve();
        };
      });
    };
    testWindow.__dojoArmRenderHistoryGate = (phase = 'after-list') => {
      armedRenderHistoryPhase = phase;
      testWindow.__dojoPendingRenderHistoryGate = undefined;
      testWindow.__dojoResolveRenderHistoryGate = undefined;
    };
    testWindow.__torchDojoRenderHistoryGate = event => {
      if (event.phase !== armedRenderHistoryPhase) return undefined;
      if (testWindow.__dojoPendingRenderHistoryGate) return undefined;
      return new Promise<void>(resolve => {
        testWindow.__dojoPendingRenderHistoryGate = event;
        testWindow.__dojoResolveRenderHistoryGate = () => {
          armedRenderHistoryPhase = undefined;
          testWindow.__dojoPendingRenderHistoryGate = undefined;
          testWindow.__dojoResolveRenderHistoryGate = undefined;
          resolve();
        };
      });
    };
  });
}

async function installControlledChecker(page: Page) {
  await page.addInitScript(() => {
    type ControlledWorker = Worker & {
      onmessage: ((event: MessageEvent) => void) | null;
      onerror: ((event: ErrorEvent) => void) | null;
    };
    const OriginalWorker = window.Worker;
    const pendingChecks: Array<{ worker: ControlledWorker; message: { requestId: string } }> = [];
    const testWindow = window as Window & {
      __dojoPendingControlledChecks?: Array<{ requestId: string }>;
      __dojoResolveNextControlledCheck?: () => void;
    };
    testWindow.__dojoPendingControlledChecks = [];
    testWindow.__dojoResolveNextControlledCheck = () => {
      const pending = pendingChecks.shift();
      const visiblePending = testWindow.__dojoPendingControlledChecks;
      if (visiblePending) visiblePending.shift();
      if (!pending) throw new Error('No pending controlled check.');
      const result = {
        status: 'passed',
        checks: [{ name: 'controlled-check', status: 'passed', elapsed_ms: 1 }],
        error: null,
        checker_version: 'controlled-checker',
        runtime: 'controlled-worker',
        package_identity: { runtime: 'controlled-worker' },
        fixture_id: 'canonical-0059'
      };
      pending.worker.onmessage?.({ data: { requestId: pending.message.requestId, result } } as MessageEvent);
    };
    window.Worker = function Worker(scriptURL: string | URL, options?: WorkerOptions) {
      if (!String(scriptURL).includes('checker-worker.js')) {
        return new OriginalWorker(scriptURL, options);
      }
      const worker = {
        onmessage: null,
        onerror: null,
        postMessage(message: { requestId: string }) {
          pendingChecks.push({ worker: worker as ControlledWorker, message });
          testWindow.__dojoPendingControlledChecks?.push({ requestId: message.requestId });
        },
        terminate() {},
        addEventListener() {},
        removeEventListener() {},
        dispatchEvent() { return true; }
      };
      return worker as ControlledWorker;
    } as unknown as typeof Worker;
  });
}

async function openNotebookPath(page: Page, path: string) {
  await page.evaluate(async notebookPath => {
    const openPath = (window as Window & {
      __dojoOpenNotebookPath?: (path: string) => Promise<void>;
    }).__dojoOpenNotebookPath;
    if (!openPath) throw new Error('Notebook path opener is not installed.');
    await openPath(notebookPath);
  }, path);
}

async function armCheckGate(page: Page, phase = 'before-receipt') {
  await page.evaluate(gatePhase => {
    (window as Window & {
      __dojoArmCheckGate?: (phase?: 'before-receipt' | 'after-receipt' | 'after-render') => void;
    }).__dojoArmCheckGate?.(gatePhase as 'before-receipt');
  }, phase);
}

async function waitForCheckGate(page: Page, phase = 'before-receipt') {
  await page.waitForFunction(gatePhase => {
    return (window as Window & {
      __dojoPendingCheckGate?: { attemptId: string; phase: 'before-receipt' | 'after-receipt' | 'after-render' };
    }).__dojoPendingCheckGate?.phase === gatePhase;
  }, phase, { timeout: 10_000 });
}

async function releaseCheckGate(page: Page) {
  await page.evaluate(() => {
    const release = (window as Window & {
      __dojoResolveCheckGate?: () => void;
    }).__dojoResolveCheckGate;
    if (!release) throw new Error('No check gate is paused.');
    release();
  });
}

async function armRenderAttemptGate(page: Page, phase = 'before-ui-write') {
  await page.evaluate(gatePhase => {
    (window as Window & {
      __dojoArmRenderAttemptGate?: (phase?: 'before-ui-write') => void;
    }).__dojoArmRenderAttemptGate?.(gatePhase as 'before-ui-write');
  }, phase);
}

async function waitForRenderAttemptGate(page: Page, phase = 'before-ui-write') {
  await page.waitForFunction(gatePhase => {
    return (window as Window & {
      __dojoPendingRenderAttemptGate?: { attemptId: string; phase: 'before-ui-write' };
    }).__dojoPendingRenderAttemptGate?.phase === gatePhase;
  }, phase, { timeout: 10_000 });
}

async function releaseRenderAttemptGate(page: Page) {
  await page.evaluate(() => {
    const release = (window as Window & {
      __dojoResolveRenderAttemptGate?: () => void;
    }).__dojoResolveRenderAttemptGate;
    if (!release) throw new Error('No render attempt gate is paused.');
    release();
  });
}

async function armRenderHistoryGate(page: Page, phase = 'after-list') {
  await page.evaluate(gatePhase => {
    (window as Window & {
      __dojoArmRenderHistoryGate?: (phase?: 'after-list') => void;
    }).__dojoArmRenderHistoryGate?.(gatePhase as 'after-list');
  }, phase);
}

async function waitForRenderHistoryGate(page: Page, phase = 'after-list', attemptId?: string) {
  await page.waitForFunction(({ gatePhase, expectedAttemptId }) => {
    const pending = (window as Window & {
      __dojoPendingRenderHistoryGate?: { attemptId?: string; phase: 'after-list' };
    }).__dojoPendingRenderHistoryGate;
    return pending?.phase === gatePhase &&
      (expectedAttemptId === undefined || pending.attemptId === expectedAttemptId);
  }, { gatePhase: phase, expectedAttemptId: attemptId }, { timeout: 10_000 });
}

async function releaseRenderHistoryGate(page: Page) {
  await page.evaluate(() => {
    const release = (window as Window & {
      __dojoResolveRenderHistoryGate?: () => void;
    }).__dojoResolveRenderHistoryGate;
    if (!release) throw new Error('No render history gate is paused.');
    release();
  });
}

async function startControlledAutosave(page: Page) {
  await page.evaluate(() => {
    const testWindow = window as Window & {
      __dojoRunAutosaveTick?: () => Promise<unknown>;
      __dojoPendingAutosaveTick?: Promise<unknown>;
    };
    const autosave = testWindow.__dojoRunAutosaveTick;
    if (!autosave) throw new Error('Autosave control is not installed.');
    testWindow.__dojoPendingAutosaveTick = autosave();
  });
}

async function waitForStartedAutosave(page: Page) {
  await page.evaluate(async () => {
    const testWindow = window as Window & {
      __dojoPendingAutosaveTick?: Promise<unknown>;
    };
    const pending = testWindow.__dojoPendingAutosaveTick;
    if (!pending) throw new Error('No autosave tick is pending.');
    try {
      await pending;
    } finally {
      testWindow.__dojoPendingAutosaveTick = undefined;
    }
  });
}

async function waitForPendingControlledCheck(page: Page) {
  await page.waitForFunction(() => {
    return ((window as Window & {
      __dojoPendingControlledChecks?: Array<{ requestId: string }>;
    }).__dojoPendingControlledChecks?.length ?? 0) > 0;
  }, undefined, { timeout: 10_000 });
}

async function resolveNextControlledCheck(page: Page) {
  await page.evaluate(() => {
    const resolveCheck = (window as Window & {
      __dojoResolveNextControlledCheck?: () => void;
    }).__dojoResolveNextControlledCheck;
    if (!resolveCheck) throw new Error('Controlled checker is not installed.');
    resolveCheck();
  });
}

async function armTransitionGate(page: Page, kind: 'resume' | 'new' | 'import', phase = 'before-swap') {
  await page.evaluate(({ gateKind, gatePhase }) => {
    (window as Window & {
      __dojoArmAttemptTransition?: (kind: 'resume' | 'new' | 'import', phase?: string) => void;
    }).__dojoArmAttemptTransition?.(gateKind, gatePhase);
  }, { gateKind: kind, gatePhase: phase });
}

async function failTransitionAt(page: Page, kind: 'resume' | 'new' | 'import', phase: string, message: string) {
  await page.evaluate(({ gateKind, gatePhase, errorMessage }) => {
    (window as Window & {
      __dojoFailAttemptTransition?: (kind: 'resume' | 'new' | 'import', phase: string, message: string) => void;
    }).__dojoFailAttemptTransition?.(gateKind, gatePhase, errorMessage);
  }, { gateKind: kind, gatePhase: phase, errorMessage: message });
}

async function waitForTransitionGate(page: Page, kind: 'resume' | 'new' | 'import', phase = 'before-swap') {
  await page.waitForFunction(({ gateKind, gatePhase }) => {
    return (window as Window & {
      __dojoPendingAttemptTransition?: { kind: 'resume' | 'new' | 'import'; attemptId: string; phase?: string };
    }).__dojoPendingAttemptTransition?.kind === gateKind &&
      ((window as Window & {
        __dojoPendingAttemptTransition?: { kind: 'resume' | 'new' | 'import'; attemptId: string; phase?: string };
      }).__dojoPendingAttemptTransition?.phase ?? 'before-swap') === gatePhase;
  }, { gateKind: kind, gatePhase: phase });
}

async function runControlledAutosave(page: Page) {
  await page.evaluate(async () => {
    const autosave = (window as Window & {
      __dojoRunAutosaveTick?: () => Promise<void>;
    }).__dojoRunAutosaveTick;
    if (!autosave) throw new Error('Autosave control is not installed.');
    await autosave();
  });
}

async function releaseTransitionGate(page: Page) {
  await page.evaluate(() => {
    const release = (window as Window & {
      __dojoResolveAttemptTransition?: () => void;
    }).__dojoResolveAttemptTransition;
    if (!release) throw new Error('No attempt transition is paused.');
    release();
  });
}

async function dispatchButtonClick(page: Page, name: string) {
  await page.evaluate(buttonName => {
    const button = Array.from(document.querySelectorAll<HTMLButtonElement>('#torch-dojo-bench button'))
      .find(item => item.textContent === buttonName);
    if (!button) throw new Error(`Button not found: ${buttonName}`);
    button.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
  }, name);
}

async function selectNotebook(page: Page, name: string) {
  // Workouts open in single-document mode, so there are no document tabs to click.
  await expect.poll(() => page.evaluate(async notebookName => {
    try {
      await (window as Window & { __dojoActivateNotebook?: (name: string) => Promise<void> })
        .__dojoActivateNotebook?.(notebookName);
      return true;
    } catch {
      return false;
    }
  }, name), { timeout: 60_000 }).toBe(true);
}

async function failImportAfterLoadAndSelectSharedActive(page: Page, attemptId: string) {
  await page.evaluate(selectedAttemptId => {
    (window as Window & {
      __torchDojoAttemptTransitionGate?: (event: {
        kind: 'resume' | 'new' | 'import';
        attemptId: string;
        phase?: string;
      }) => void | Promise<void>;
    }).__torchDojoAttemptTransitionGate = event => {
      if (event.kind !== 'import' || event.phase !== 'after-load') return undefined;
      const key = 'torch-dojo.browser-gym.attempts.v1';
      const raw = window.localStorage.getItem(key);
      if (!raw) throw new Error('Attempt store missing before concurrent active selection');
      const state = JSON.parse(raw) as { activeByExercise?: Record<string, string> };
      state.activeByExercise = { ...(state.activeByExercise ?? {}), '0059': selectedAttemptId };
      window.localStorage.setItem(key, JSON.stringify(state));
      throw new Error('Injected import failure after concurrent active selection');
    };
  }, attemptId);
}

async function exportAttemptArchive(page: Page) {
  const panel = page.locator('body');
  const download = await Promise.all([
    page.waitForEvent('download'),
    panel.getByRole('button', { name: 'Export rep' }).click()
  ]).then(([item]) => item);
  const archivePath = await download.path();
  expect(archivePath).toBeTruthy();
  return JSON.parse(readFileSync(archivePath!, 'utf8'));
}

async function dispatchImportArchiveChange(page: Page, archive: unknown) {
  await page.evaluate(archiveText => {
    const input = document.querySelector<HTMLInputElement>('#torch-dojo-bench input[type="file"]');
    if (!input) throw new Error('Import input not found.');
    const file = new File([archiveText], 'attempts-import-while-locked.json', { type: 'application/json' });
    const transfer = new DataTransfer();
    transfer.items.add(file);
    Object.defineProperty(input, 'files', { configurable: true, value: transfer.files });
    input.dispatchEvent(new Event('change', { bubbles: true }));
  }, JSON.stringify(archive));
}

async function exportAllArchive(page: Page) {
  const download = await Promise.all([
    page.waitForEvent('download'),
    dispatchButtonClick(page, 'Export all')
  ]).then(([item]) => item);
  const archivePath = await download.path();
  expect(archivePath).toBeTruthy();
  return JSON.parse(readFileSync(archivePath!, 'utf8'));
}

test('learner-facing gym copy says rep, never attempt', async ({ page }) => {
  const { panel, status } = await loadGym(page);
  await panel.getByRole('button', { name: 'New rep' }).click();
  await expect(status).toContainText('Rep 2');
  await expect(panel.getByRole('button', { name: 'Resume rep 1' })).toBeVisible();
  await expect(page.getByRole('region', { name: 'Rep log' })).toContainText('Rep log: 2 reps');
  await expect(panel.getByRole('button', { name: 'Export rep' })).toBeVisible();
  await expect(panel.getByLabel('Rep notes')).toBeVisible();

  const learnerCopy = await page.locator('#torch-dojo-bench').evaluate(root => {
    const parts: string[] = [];
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    for (let node = walker.nextNode(); node; node = walker.nextNode()) parts.push(node.textContent ?? '');
    for (const element of root.querySelectorAll('*')) {
      for (const name of ['aria-label', 'title', 'placeholder']) {
        const value = element.getAttribute(name);
        if (value) parts.push(value);
      }
    }
    return parts.join('\n');
  });
  expect(learnerCopy).not.toMatch(/attempt/i);
});

test('hints, solution reveal, new attempts, and archives preserve attempt history', async ({ page }, testInfo) => {
  const { panel, status, check } = await loadGym(page);
  const answer = () => answerCell(page);
  const answerBeforeReveal = await answer().textContent();

  const hint = panel.getByRole('button', { name: 'Hint', exact: true });
  await hint.click();
  await expect(panel).toContainText('Subtract each row');
  await hint.click();
  await expect(panel).toContainText('Average only valid rows');
  await panel.getByRole('button', { name: 'Collapse hints' }).click();
  await expect(panel).not.toContainText('Average only valid rows');
  await expect(status).toContainText('Assistance recorded: hint 1, hint 2');

  await panel.getByRole('button', { name: 'Reveal solution' }).click();
  await expect(panel).toContainText('def softmax_cross_entropy');
  await expect(answer()).toHaveText(answerBeforeReveal ?? '');
  await panel.getByRole('button', { name: 'Hide solution' }).click();
  await expect(panel).not.toContainText('gradient[valid] = probabilities / count');
  await expect(status).toContainText('solution reveal');

  await panel.getByLabel('Rep notes').fill('Remember the stable denominator.');
  await answer().click();
  await page.keyboard.press('ControlOrMeta+A');
  await page.keyboard.insertText(reference);
  await check.click();
  await expect(status).toContainText('ASSISTED PASS', { timeout: 120_000 });
  const assistedText = await status.textContent();
  const assistedHash = assistedText?.match(/Receipt source SHA-256 ([0-9a-f]{64})/)?.[1];
  expect(assistedHash).toBeTruthy();

  const editedAfterCheck = `${reference}\n# edited after check`;
  await answer().click();
  await page.keyboard.press('ControlOrMeta+A');
  await page.keyboard.insertText(editedAfterCheck);
  await expect(status).toContainText('STALE PASS');

  const singleDownload = await Promise.all([
    page.waitForEvent('download'),
    panel.getByRole('button', { name: 'Export rep' }).click()
  ]).then(([download]) => download);
  const singleArchivePath = await singleDownload.path();
  expect(singleArchivePath).toBeTruthy();
  const singleArchive = JSON.parse(readFileSync(singleArchivePath!, 'utf8'));
  expect(singleArchive.attempts).toHaveLength(1);
  expect(singleArchive.attempts[0].notes).toBe('Remember the stable denominator.');
  expect(singleArchive.attempts[0].assistance.map((event: { kind: string }) => event.kind)).toContain('solution');
  expect(singleArchive.files[`attempts/${singleArchive.attempts[0].id}/solution.py`]).toBe(editedAfterCheck);
  expect(singleArchive.attempts[0].receipts[0].source).toBe(reference);
  expect(singleArchive.attempts[0].receipts[0].sha256).toBe(assistedHash);

  await panel.getByRole('button', { name: 'New rep' }).click();
  await expect(status).toContainText('Rep 2');
  await expect(answerPane(page).locator('#torch-dojo-gym')).toBeVisible();
  await expect(status).toContainText('Rep log: 2 reps');
  await expect(answer()).not.toContainText('gradient[valid] = probabilities / count');

  await answer().click();
  await page.keyboard.press('ControlOrMeta+A');
  await page.keyboard.insertText(reference);
  await check.click();
  await answer().click();
  await page.keyboard.press('ControlOrMeta+End');
  await page.keyboard.insertText('\n# edited while check was running');
  await expect(status).toContainText('STALE PASS', { timeout: 120_000 });

  await answer().click();
  await page.keyboard.press('ControlOrMeta+A');
  await page.keyboard.insertText(reference);
  await check.click();
  await expect(status).toContainText('INDEPENDENT PASS', { timeout: 120_000 });

  const allDownload = await Promise.all([
    page.waitForEvent('download'),
    panel.getByRole('button', { name: 'Export all' }).click()
  ]).then(([download]) => download);
  const allArchivePath = await allDownload.path();
  expect(allArchivePath).toBeTruthy();
  const allArchive = JSON.parse(readFileSync(allArchivePath!, 'utf8'));
  expect(allArchive.attempts).toHaveLength(2);
  expect(allArchive.attempts[0].receipts[0].sha256).toBe(assistedHash);
  expect(allArchive.attempts[1].assistance).toEqual([]);

  const importPath = testInfo.outputPath('attempts-import.json');
  writeFileSync(importPath, JSON.stringify(allArchive));
  await page.evaluate(() => localStorage.clear());
  await expect(panel.locator('input[type="file"]')).toBeEnabled();
  await panel.locator('input[type="file"]').setInputFiles(importPath);
  await expect(status).toContainText('Imported reps: 2');
  await expect(status).toContainText('Rep log: 2 reps');
  await expect(status).toContainText(assistedHash!);

  const restoredDownload = await Promise.all([
    page.waitForEvent('download'),
    panel.getByRole('button', { name: 'Export all' }).click()
  ]).then(([download]) => download);
  const restoredArchivePath = await restoredDownload.path();
  expect(restoredArchivePath).toBeTruthy();
  const restoredArchive = JSON.parse(readFileSync(restoredArchivePath!, 'utf8'));
  expect(restoredArchive.files[`attempts/${allArchive.attempts[0].id}/solution.py`]).toBe(editedAfterCheck);
  expect(restoredArchive.attempts[0].receipts[0].source).toBe(reference);
});

test('learner can resume a prior attempt without losing the current attempt', async ({ page }) => {
  const { panel, status, check } = await loadGym(page);
  const notes = panel.getByLabel('Rep notes');
  const firstSource = `${reference}\n# first attempt survives resume`;
  const secondSource = 'import numpy as np\n# second attempt draft survives resume';

  await replaceAnswer(page, firstSource);
  await notes.fill('first notes survive resume');
  await check.click();
  await expect(status).toContainText('INDEPENDENT PASS', { timeout: 120_000 });

  await panel.getByRole('button', { name: 'New rep' }).click();
  await expect(status).toContainText('Rep 2');
  await replaceAnswer(page, secondSource);
  await notes.fill('second notes survive resume');

  await panel.getByRole('button', { name: 'Resume rep 1' }).click();
  await expect(answerPane(page).locator('#torch-dojo-gym')).toBeVisible();
  await expect(notes).toHaveValue('first notes survive resume');
  await expect(answerCell(page)).toContainText('# first attempt survives resume');
  await expect(status).toContainText('Rep 1 (current)');
  await check.click();
  await expect(status).toContainText('INDEPENDENT PASS', { timeout: 120_000 });

  const download = await Promise.all([
    page.waitForEvent('download'),
    panel.getByRole('button', { name: 'Export all' }).click()
  ]).then(([item]) => item);
  const archivePath = await download.path();
  expect(archivePath).toBeTruthy();
  const archive = JSON.parse(readFileSync(archivePath!, 'utf8'));
  expect(archive.attempts).toHaveLength(2);
  expect(archive.attempts[0].notes).toBe('first notes survive resume');
  expect(archive.files[`attempts/${archive.attempts[0].id}/solution.py`]).toBe(firstSource);
  expect(archive.attempts[0].receipts.at(-1).source).toBe(firstSource);
  expect(archive.attempts[1].notes).toBe('second notes survive resume');
  expect(archive.files[`attempts/${archive.attempts[1].id}/solution.py`]).toBe(secondSource);
});

test('successful resume preserves unsaved background gym edits when another notebook tab is selected', async ({ page }) => {
  await installTransitionControls(page);
  const { panel, status } = await loadGym(page, 'lab/index.html?exercise=0059&path=notebooks/0060_classifier_step.ipynb');
  const notes = panel.getByLabel('Rep notes');
  const firstSource = `${reference}\n# background first attempt remains selectable`;
  const secondSource = 'import numpy as np\n# unsaved background second attempt survives resume';

  await selectNotebook(page, '0059_stable_cross_entropy.ipynb');
  await expect(visibleNotebookHeading(page)).toContainText('0059', { timeout: 60_000 });
  await replaceAnswer(page, firstSource);
  await notes.fill('background first notes');

  await panel.getByRole('button', { name: 'New rep' }).click();
  await expect(status).toContainText('Rep 2');
  await selectNotebook(page, '0059_stable_cross_entropy.ipynb');
  await expect(visibleNotebookHeading(page)).toContainText('0059', { timeout: 60_000 });
  await replaceAnswer(page, secondSource);
  await expect(answerCell(page)).toContainText('# unsaved background second attempt survives resume');
  await notes.fill('background second notes');

  await selectNotebook(page, '0060_classifier_step.ipynb');
  await expect(visibleNotebookHeading(page)).toContainText('0060 · One honest classifier update', {
    timeout: 60_000
  });

  // The inline UI is hidden on another tab; invoke the controller directly.
  await dispatchButtonClick(page, 'Resume rep 1');
  await expect(notes).toHaveValue('background first notes');
  await expect(answerCell(page)).toContainText('# background first attempt remains selectable');
  await expect(status).toContainText('Rep 1 (current)');

  const archive = await exportAllArchive(page);
  expect(archive.attempts).toHaveLength(2);
  expect(archive.activeByExercise['0059']).toBe(archive.attempts[0].id);
  expect(archive.files[`attempts/${archive.attempts[0].id}/solution.py`]).toBe(firstSource);
  expect(archive.attempts[0].notes).toBe('background first notes');
  expect(archive.files[`attempts/${archive.attempts[1].id}/solution.py`]).toBe(secondSource);
  expect(archive.attempts[1].notes).toBe('background second notes');
});

test('resume post-load failure restores the background gym notebook when an unrelated tab is selected', async ({ page }) => {
  await installTransitionControls(page);
  const { panel, status } = await loadGym(page, 'lab/index.html?exercise=0059&path=notebooks/0060_classifier_step.ipynb');
  const notes = panel.getByLabel('Rep notes');
  const firstSource = `${reference}\n# rollback target must not replace active background source`;
  const secondSource = 'import numpy as np\n# active background source survives failed resume';

  await selectNotebook(page, '0059_stable_cross_entropy.ipynb');
  await expect(visibleNotebookHeading(page)).toContainText('0059', { timeout: 60_000 });
  await replaceAnswer(page, firstSource);
  await notes.fill('rollback first notes');

  await panel.getByRole('button', { name: 'New rep' }).click();
  await expect(status).toContainText('Rep 2');
  await selectNotebook(page, '0059_stable_cross_entropy.ipynb');
  await expect(visibleNotebookHeading(page)).toContainText('0059', { timeout: 60_000 });
  await replaceAnswer(page, secondSource);
  await expect(answerCell(page)).toContainText('# active background source survives failed resume');
  await notes.fill('rollback active notes');

  await selectNotebook(page, '0060_classifier_step.ipynb');
  await expect(visibleNotebookHeading(page)).toContainText('0060 · One honest classifier update', {
    timeout: 60_000
  });

  await failTransitionAt(page, 'resume', 'after-load', 'Injected background resume failure after notebook load');
  // The inline UI is hidden on another tab; invoke the controller directly.
  await dispatchButtonClick(page, 'Resume rep 1');
  await expect(status).toContainText('Injected background resume failure after notebook load');
  await expect(status).toContainText('restored');
  await expect(notes).toHaveValue('rollback active notes');
  await expect(answerCell(page)).toContainText('# active background source survives failed resume');

  await runControlledAutosave(page);
  const archive = await exportAllArchive(page);
  expect(archive.attempts).toHaveLength(2);
  expect(archive.activeByExercise['0059']).toBe(archive.attempts[1].id);
  expect(archive.files[`attempts/${archive.attempts[0].id}/solution.py`]).toBe(firstSource);
  expect(archive.attempts[0].notes).toBe('rollback first notes');
  expect(archive.files[`attempts/${archive.attempts[1].id}/solution.py`]).toBe(secondSource);
  expect(archive.attempts[1].notes).toBe('rollback active notes');
});

test('same-exercise resume keeps the selected attempt source across an autosave tick during transition', async ({ page }) => {
  await installTransitionControls(page);
  const { panel, status } = await loadGym(page);
  const notes = panel.getByLabel('Rep notes');
  const firstSource = `${reference}\n# first attempt selected across transition`;
  const secondSource = 'import numpy as np\n# second attempt must not overwrite first during resume';

  await replaceAnswer(page, firstSource);
  await notes.fill('first notes selected across transition');

  await panel.getByRole('button', { name: 'New rep' }).click();
  await expect(status).toContainText('Rep 2');
  await replaceAnswer(page, secondSource);
  await notes.fill('second notes stay with second attempt');

  await armTransitionGate(page, 'resume');
  await panel.getByRole('button', { name: 'Resume rep 1' }).click();
  await waitForTransitionGate(page, 'resume');
  await runControlledAutosave(page);
  await releaseTransitionGate(page);

  await expect(notes).toHaveValue('first notes selected across transition');
  await expect(answerCell(page)).toContainText('# first attempt selected across transition');
  await expect(status).toContainText('Rep 1 (current)');

  const archive = await exportAllArchive(page);
  expect(archive.attempts).toHaveLength(2);
  expect(archive.attempts[0].notes).toBe('first notes selected across transition');
  expect(archive.files[`attempts/${archive.attempts[0].id}/solution.py`]).toBe(firstSource);
  expect(archive.attempts[1].notes).toBe('second notes stay with second attempt');
  expect(archive.files[`attempts/${archive.attempts[1].id}/solution.py`]).toBe(secondSource);
});

test('new attempt keeps a fresh notebook source across an autosave tick during transition', async ({ page }) => {
  await installTransitionControls(page);
  const { panel, status } = await loadGym(page);
  const oldSource = 'import numpy as np\n# old draft must not overwrite the fresh attempt';

  await replaceAnswer(page, oldSource);

  await armTransitionGate(page, 'new');
  await panel.getByRole('button', { name: 'New rep' }).click();
  await waitForTransitionGate(page, 'new');
  await runControlledAutosave(page);
  const archiveDuringTransition = await exportAllArchive(page);
  await releaseTransitionGate(page);

  await expect(status).toContainText('Rep 2');
  await expect(answerCell(page)).not.toContainText('# old draft must not overwrite the fresh attempt');
  expect(archiveDuringTransition.attempts).toHaveLength(2);
  const freshAttemptId = archiveDuringTransition.attempts[1].id;
  expect(archiveDuringTransition.files[`attempts/${freshAttemptId}/solution.py`])
    .not.toContain('# old draft must not overwrite the fresh attempt');
});

test('overlapping new attempt click is ignored while resume transition is locked', async ({ page }) => {
  await installTransitionControls(page);
  const { panel, status } = await loadGym(page);
  const notes = panel.getByLabel('Rep notes');
  const firstSource = `${reference}\n# first source survives overlapping transition`;
  const secondSource = 'import numpy as np\n# second source survives overlapping transition';

  await replaceAnswer(page, firstSource);
  await notes.fill('first notes survive overlapping transition');

  await panel.getByRole('button', { name: 'New rep' }).click();
  await expect(status).toContainText('Rep 2');
  await replaceAnswer(page, secondSource);
  await notes.fill('second notes survive overlapping transition');

  await armTransitionGate(page, 'resume');
  await panel.getByRole('button', { name: 'Resume rep 1' }).click();
  await waitForTransitionGate(page, 'resume');
  await expect(panel.getByRole('button', { name: 'New rep' })).toBeDisabled();

  await dispatchButtonClick(page, 'New rep');
  await runControlledAutosave(page);
  await releaseTransitionGate(page);

  await expect(notes).toHaveValue('first notes survive overlapping transition');
  await expect(answerCell(page)).toContainText('# first source survives overlapping transition');
  await expect(status).toContainText('Rep 1 (current)');

  await runControlledAutosave(page);
  const archive = await exportAllArchive(page);
  expect(archive.attempts).toHaveLength(2);
  expect(archive.activeByExercise['0059']).toBe(archive.attempts[0].id);
  expect(archive.files[`attempts/${archive.attempts[0].id}/solution.py`]).toBe(firstSource);
  expect(archive.attempts[0].notes).toBe('first notes survive overlapping transition');
  expect(archive.files[`attempts/${archive.attempts[1].id}/solution.py`]).toBe(secondSource);
  expect(archive.attempts[1].notes).toBe('second notes survive overlapping transition');
});

test('open exercise cannot overwrite the visible model while resume transition is locked', async ({ page }) => {
  await installTransitionControls(page);
  const { panel, status } = await loadGym(page);
  const notes = panel.getByLabel('Rep notes');
  const firstSource = `${reference}\n# first source survives overlapping open`;
  const secondSource = 'import numpy as np\n# second source must not overwrite target during open';

  await replaceAnswer(page, firstSource);
  await notes.fill('first notes survive overlapping open');

  await panel.getByRole('button', { name: 'New rep' }).click();
  await expect(status).toContainText('Rep 2');
  await replaceAnswer(page, secondSource);
  await notes.fill('second notes survive overlapping open');

  await armTransitionGate(page, 'resume', 'after-load');
  await panel.getByRole('button', { name: 'Resume rep 1' }).click();
  await waitForTransitionGate(page, 'resume', 'after-load');
  await expect(panel.getByRole('button', { name: 'Open exercise' })).toBeDisabled();
  await expect(panel.getByRole('button', { name: /^Check (?!log)/ })).toBeDisabled();

  await dispatchButtonClick(page, 'Open exercise');
  await releaseTransitionGate(page);

  await expect(notes).toHaveValue('first notes survive overlapping open');
  await expect(answerCell(page)).toContainText('# first source survives overlapping open');
  await expect(status).toContainText('Rep 1 (current)');

  await runControlledAutosave(page);
  const archive = await exportAllArchive(page);
  expect(archive.attempts).toHaveLength(2);
  expect(archive.activeByExercise['0059']).toBe(archive.attempts[0].id);
  expect(archive.files[`attempts/${archive.attempts[0].id}/solution.py`]).toBe(firstSource);
  expect(archive.files[`attempts/${archive.attempts[1].id}/solution.py`]).toBe(secondSource);
});

test('open exercise preserves an unsaved active notebook edit before autosave', async ({ page }) => {
  await installTransitionControls(page);
  const { panel } = await loadGym(page);
  const unsavedSource = 'import numpy as np\n# unsaved source survives immediate open';

  await replaceAnswer(page, unsavedSource);
  await panel.getByRole('button', { name: 'Open exercise' }).click();

  await expect(answerCell(page)).toContainText('# unsaved source survives immediate open');
  const archive = await exportAttemptArchive(page);
  expect(archive.attempts).toHaveLength(1);
  expect(archive.files[`attempts/${archive.attempts[0].id}/solution.py`]).toBe(unsavedSource);
});

test('import archive is ignored while resume transition is locked', async ({ page }) => {
  await installTransitionControls(page);
  const { panel, status } = await loadGym(page);
  const notes = panel.getByLabel('Rep notes');
  const importSource = 'import numpy as np\n# imported archive must wait for transition lock';
  const firstSource = `${reference}\n# first source survives overlapping import`;
  const secondSource = 'import numpy as np\n# second source survives overlapping import';

  await replaceAnswer(page, importSource);
  const importArchive = await exportAttemptArchive(page);
  await replaceAnswer(page, firstSource);
  await notes.fill('first notes survive overlapping import');

  await panel.getByRole('button', { name: 'New rep' }).click();
  await expect(status).toContainText('Rep 2');
  await replaceAnswer(page, secondSource);
  await notes.fill('second notes survive overlapping import');

  await armTransitionGate(page, 'resume', 'after-load');
  await panel.getByRole('button', { name: 'Resume rep 1' }).click();
  await waitForTransitionGate(page, 'resume', 'after-load');
  await expect(panel.locator('input[type="file"]')).toBeDisabled();
  await dispatchImportArchiveChange(page, importArchive);
  await releaseTransitionGate(page);

  await expect(notes).toHaveValue('first notes survive overlapping import');
  await expect(answerCell(page)).toContainText('# first source survives overlapping import');
  await expect(status).toContainText('Rep 1 (current)');

  await runControlledAutosave(page);
  const archive = await exportAllArchive(page);
  expect(archive.attempts).toHaveLength(2);
  expect(archive.activeByExercise['0059']).toBe(archive.attempts[0].id);
  expect(archive.files[`attempts/${archive.attempts[0].id}/solution.py`]).toBe(firstSource);
  expect(archive.files[`attempts/${archive.attempts[1].id}/solution.py`]).toBe(secondSource);
  expect(JSON.stringify(archive)).not.toContain('# imported archive must wait for transition lock');
});

test('import archive post-load failure restores prior attempt before autosave resumes', async ({ page }) => {
  await installTransitionControls(page);
  const { panel, status } = await loadGym(page);
  const notes = panel.getByLabel('Rep notes');
  const importedSource = 'import numpy as np\n# imported archive must not survive failed swap';
  const oldSource = 'import numpy as np\n# original source survives failed import';

  await replaceAnswer(page, importedSource);
  const importArchive = await exportAttemptArchive(page);
  await replaceAnswer(page, oldSource);
  await notes.fill('original notes survive failed import');
  await runControlledAutosave(page);

  await failTransitionAt(page, 'import', 'after-load', 'Injected import failure after notebook load');
  await dispatchImportArchiveChange(page, importArchive);

  await expect(status).toContainText('Injected import failure after notebook load');
  await expect(status).toContainText('restored');
  await expect(notes).toHaveValue('original notes survive failed import');
  await expect(answerCell(page)).toContainText('# original source survives failed import');

  await runControlledAutosave(page);
  const archive = await exportAllArchive(page);
  expect(archive.attempts).toHaveLength(1);
  expect(archive.activeByExercise['0059']).toBe(archive.attempts[0].id);
  expect(archive.files[`attempts/${archive.attempts[0].id}/solution.py`]).toBe(oldSource);
  expect(JSON.stringify(archive)).not.toContain('# imported archive must not survive failed swap');
});

test('failed import keeps local check bound to restored attempt when another tab selects a different active attempt', async ({ page }) => {
  await installTransitionControls(page);
  await installControlledChecker(page);
  const { panel, status, check } = await loadGym(page);
  const notes = panel.getByLabel('Rep notes');
  const importedSource = 'import numpy as np\n# imported source must be removed after failed import';
  const firstSource = `${reference}\n# local restored attempt checked after failed import`;
  const secondSource = `${reference}\n# other tab active attempt must remain untouched`;

  await replaceAnswer(page, importedSource);
  const importArchive = await exportAttemptArchive(page);
  await replaceAnswer(page, firstSource);
  await notes.fill('local restored attempt notes');
  await runControlledAutosave(page);

  await panel.getByRole('button', { name: 'New rep' }).click();
  await expect(status).toContainText('Rep 2');
  await replaceAnswer(page, secondSource);
  await notes.fill('other tab active notes');
  await runControlledAutosave(page);
  await check.click();
  await waitForPendingControlledCheck(page);
  await resolveNextControlledCheck(page);
  await expect(status).toContainText('Receipt 1');
  await expect(panel.locator('input[type="file"]')).toBeEnabled();

  const beforeImport = await exportAllArchive(page);
  const firstAttemptId = beforeImport.attempts[0].id;
  const secondAttemptId = beforeImport.attempts[1].id;
  expect(beforeImport.files[`attempts/${secondAttemptId}/solution.py`]).toBe(secondSource);
  expect(beforeImport.attempts[1].receipts).toHaveLength(1);
  expect(beforeImport.attempts[1].receipts[0].source).toBe(secondSource);

  await panel.getByRole('button', { name: 'Resume rep 1' }).click();
  await expect(status).toContainText('Rep 1 (current)');
  await expect(notes).toHaveValue('local restored attempt notes');
  await expect(answerCell(page)).toContainText('# local restored attempt checked after failed import');

  await failImportAfterLoadAndSelectSharedActive(page, secondAttemptId);
  await dispatchImportArchiveChange(page, importArchive);
  await expect(status).toContainText('Injected import failure after concurrent active selection');
  await expect(status).toContainText('restored');
  await expect(notes).toHaveValue('local restored attempt notes');
  await expect(answerCell(page)).toContainText('# local restored attempt checked after failed import');

  await check.click();
  await waitForPendingControlledCheck(page);
  await resolveNextControlledCheck(page);
  await expect(status).toContainText('Receipt 1');
  await expect(panel.locator('input[type="file"]')).toBeEnabled();

  const archive = await exportAllArchive(page);
  const attemptsById = Object.fromEntries(archive.attempts.map((attempt: { id: string }) => [attempt.id, attempt]));
  const firstAttempt = attemptsById[firstAttemptId] as { notes: string; receipts: Array<{ source: string }> };
  const secondAttempt = attemptsById[secondAttemptId] as { notes: string; receipts: Array<{ source: string }> };
  expect(archive.attempts).toHaveLength(2);
  expect(archive.activeByExercise['0059']).toBe(secondAttemptId);
  expect(firstAttempt.notes).toBe('local restored attempt notes');
  expect(firstAttempt.receipts).toHaveLength(1);
  expect(firstAttempt.receipts[0].source).toBe(firstSource);
  expect(archive.files[`attempts/${firstAttemptId}/solution.py`]).toBe(firstSource);
  expect(secondAttempt.notes).toBe('other tab active notes');
  expect(secondAttempt.receipts).toHaveLength(1);
  expect(secondAttempt.receipts[0].source).toBe(secondSource);
  expect(archive.files[`attempts/${secondAttemptId}/solution.py`]).toBe(secondSource);
  expect(JSON.stringify(archive)).not.toContain('# imported source must be removed after failed import');
});

test('import archive is blocked while a completed check is appending its receipt', async ({ page }) => {
  await installTransitionControls(page);
  await installControlledChecker(page);
  const { panel, status, check } = await loadGym(page);
  const notes = panel.getByLabel('Rep notes');
  const importedSource = 'import numpy as np\n# import must not run while check is committing';
  const checkedSource = `${reference}\n# checked receipt must survive blocked import`;

  await replaceAnswer(page, importedSource);
  const importArchive = await exportAttemptArchive(page);
  await replaceAnswer(page, checkedSource);
  await notes.fill('checked notes survive blocked import');

  await armCheckGate(page, 'before-receipt');
  await check.click();
  await waitForPendingControlledCheck(page);
  await resolveNextControlledCheck(page);
  await waitForCheckGate(page, 'before-receipt');
  await expect(panel.locator('input[type="file"]')).toBeDisabled();
  await failTransitionAt(page, 'import', 'after-load', 'Injected import failure should be unreachable');
  await dispatchImportArchiveChange(page, importArchive);
  await releaseCheckGate(page);

  await expect(status).toContainText('INDEPENDENT PASS');
  await expect(status).toContainText('Receipt 1');
  const archive = await exportAllArchive(page);
  expect(archive.attempts).toHaveLength(1);
  expect(archive.attempts[0].notes).toBe('checked notes survive blocked import');
  expect(archive.attempts[0].receipts).toHaveLength(1);
  expect(archive.attempts[0].receipts[0].source).toBe(checkedSource);
  expect(archive.files[`attempts/${archive.attempts[0].id}/solution.py`]).toBe(checkedSource);
  expect(JSON.stringify(archive)).not.toContain('# import must not run while check is committing');
});

test('check completion after a new-attempt switch preserves active notes and stores the old receipt', async ({ page }) => {
  await installTransitionControls(page);
  await installControlledChecker(page);
  const { panel, status, check } = await loadGym(page);
  const notes = panel.getByLabel('Rep notes');
  const checkedSource = `${reference}\n# checked before new attempt switch`;
  const newSource = 'import numpy as np\n# active new attempt after stale check';

  await replaceAnswer(page, checkedSource);
  await notes.fill('checked attempt notes');
  await check.click();
  await waitForPendingControlledCheck(page);

  await panel.getByRole('button', { name: 'New rep' }).click();
  await expect(status).toContainText('Rep 2');
  await replaceAnswer(page, newSource);
  await notes.fill('active new notes');
  await resolveNextControlledCheck(page);

  await expect(status).toContainText('Receipt 1');
  await expect(status).toContainText('Rep 2 (current)');
  await expect(notes).toHaveValue('active new notes');

  await notes.fill('active new notes after old check');
  await runControlledAutosave(page);
  const archive = await exportAllArchive(page);
  expect(archive.attempts).toHaveLength(2);
  expect(archive.activeByExercise['0059']).toBe(archive.attempts[1].id);
  expect(archive.attempts[0].notes).toBe('checked attempt notes');
  expect(archive.attempts[0].receipts).toHaveLength(1);
  expect(archive.attempts[0].receipts[0].source).toBe(checkedSource);
  expect(archive.files[`attempts/${archive.attempts[0].id}/solution.py`]).toBe(checkedSource);
  expect(archive.attempts[1].notes).toBe('active new notes after old check');
  expect(archive.files[`attempts/${archive.attempts[1].id}/solution.py`]).toBe(newSource);
});

test('check completion after a resume switch preserves resumed notes and stores the old receipt', async ({ page }) => {
  await installTransitionControls(page);
  await installControlledChecker(page);
  const { panel, status, check } = await loadGym(page);
  const notes = panel.getByLabel('Rep notes');
  const firstSource = 'import numpy as np\n# resumed attempt remains current after old check';
  const secondSource = `${reference}\n# checked before resume switch`;

  await replaceAnswer(page, firstSource);
  await notes.fill('resumed notes before check');
  await panel.getByRole('button', { name: 'New rep' }).click();
  await expect(status).toContainText('Rep 2');
  await replaceAnswer(page, secondSource);
  await notes.fill('checked second notes');

  await check.click();
  await waitForPendingControlledCheck(page);
  await panel.getByRole('button', { name: 'Resume rep 1' }).click();
  await expect(status).toContainText('Rep 1 (current)');
  await expect(notes).toHaveValue('resumed notes before check');
  await resolveNextControlledCheck(page);

  await expect(status).toContainText('Receipt 1');
  await expect(status).toContainText('Rep 1 (current)');
  await expect(notes).toHaveValue('resumed notes before check');

  await notes.fill('resumed notes after old check');
  await runControlledAutosave(page);
  const archive = await exportAllArchive(page);
  expect(archive.attempts).toHaveLength(2);
  expect(archive.activeByExercise['0059']).toBe(archive.attempts[0].id);
  expect(archive.attempts[0].notes).toBe('resumed notes after old check');
  expect(archive.files[`attempts/${archive.attempts[0].id}/solution.py`]).toBe(firstSource);
  expect(archive.attempts[1].notes).toBe('checked second notes');
  expect(archive.attempts[1].receipts).toHaveLength(1);
  expect(archive.attempts[1].receipts[0].source).toBe(secondSource);
  expect(archive.files[`attempts/${archive.attempts[1].id}/solution.py`]).toBe(secondSource);
});

test('pending autosave render does not replace notes typed during the same attempt render', async ({ page }) => {
  await installTransitionControls(page);
  const { panel } = await loadGym(page);
  const notes = panel.getByLabel('Rep notes');
  const initialSource = 'import numpy as np\n# render race initial source';
  const changedSource = 'import numpy as np\n# render race changed source';

  await replaceAnswer(page, initialSource);
  await notes.fill('notes before render race');
  await runControlledAutosave(page);

  await replaceAnswer(page, changedSource);
  await notes.fill('stale notes before pending render');
  await armRenderAttemptGate(page);
  await startControlledAutosave(page);
  await waitForRenderAttemptGate(page);

  await notes.fill('newer notes typed during pending render');
  await releaseRenderAttemptGate(page);
  await waitForStartedAutosave(page);
  await expect(notes).toHaveValue('newer notes typed during pending render');

  await notes.fill('newer notes after pending render');
  await runControlledAutosave(page);
  const archive = await exportAllArchive(page);
  expect(archive.attempts).toHaveLength(1);
  expect(archive.attempts[0].notes).toBe('newer notes after pending render');
  expect(archive.files[`attempts/${archive.attempts[0].id}/solution.py`]).toBe(changedSource);
});

test('pending autosave render cannot write prior attempt notes after a new-attempt switch', async ({ page }) => {
  await installTransitionControls(page);
  const { panel, status } = await loadGym(page);
  const notes = panel.getByLabel('Rep notes');
  const firstSource = 'import numpy as np\n# first source saved by pending render';
  const secondSource = 'import numpy as np\n# second source after stale render release';

  await replaceAnswer(page, firstSource);
  await notes.fill('first notes before stale render');
  await runControlledAutosave(page);

  await replaceAnswer(page, `${firstSource}\n# autosave render pending across new attempt`);
  await armRenderAttemptGate(page);
  await startControlledAutosave(page);
  await waitForRenderAttemptGate(page);

  await panel.getByRole('button', { name: 'New rep' }).click();
  await expect(status).toContainText('Rep 2');
  await replaceAnswer(page, secondSource);
  await notes.fill('second notes during stale render');

  await releaseRenderAttemptGate(page);
  await waitForStartedAutosave(page);
  await expect(notes).toHaveValue('second notes during stale render');

  await notes.fill('second notes after stale render');
  await runControlledAutosave(page);
  const archive = await exportAllArchive(page);
  expect(archive.attempts).toHaveLength(2);
  expect(archive.activeByExercise['0059']).toBe(archive.attempts[1].id);
  expect(archive.attempts[0].notes).toBe('first notes before stale render');
  expect(archive.files[`attempts/${archive.attempts[0].id}/solution.py`])
    .toBe(`${firstSource}\n# autosave render pending across new attempt`);
  expect(archive.attempts[1].notes).toBe('second notes after stale render');
  expect(archive.files[`attempts/${archive.attempts[1].id}/solution.py`]).toBe(secondSource);
});

test('pending history render cannot rewrite status after a resume switch', async ({ page }) => {
  await installTransitionControls(page);
  const { panel, status } = await loadGym(page);
  const notes = panel.getByLabel('Rep notes');
  const firstSource = 'import numpy as np\n# first source restored during history race';
  const secondSource = 'import numpy as np\n# second source before history race';
  const pendingSecondSource = `${secondSource}\n# pending history render must stay in attempt 2`;

  await replaceAnswer(page, firstSource);
  await notes.fill('first notes before history race');
  await runControlledAutosave(page);

  await panel.getByRole('button', { name: 'New rep' }).click();
  await expect(status).toContainText('Rep 2');
  await replaceAnswer(page, secondSource);
  await notes.fill('second notes before history race');
  await runControlledAutosave(page);

  const beforeResume = await exportAllArchive(page);
  const firstAttemptId = beforeResume.attempts[0].id;
  const secondAttemptId = beforeResume.attempts[1].id;

  await replaceAnswer(page, pendingSecondSource);
  await armRenderHistoryGate(page);
  await startControlledAutosave(page);
  await waitForRenderHistoryGate(page, 'after-list', secondAttemptId);

  await panel.getByRole('button', { name: 'Resume rep 1' }).click();
  await expect(status).toContainText('Resumed rep 1');
  await expect(status).toContainText('Rep 1 (current)');
  await expect(notes).toHaveValue('first notes before history race');

  await releaseRenderHistoryGate(page);
  await waitForStartedAutosave(page);

  await expect(status).toContainText('Resumed rep 1');
  await expect(status).toContainText('Rep 1 (current)');
  await expect(status).not.toContainText('Rep 2 (current)');
  await expect(notes).toHaveValue('first notes before history race');

  const archive = await exportAllArchive(page);
  expect(archive.activeByExercise['0059']).toBe(firstAttemptId);
  expect(archive.files[`attempts/${firstAttemptId}/solution.py`]).toBe(firstSource);
  expect(archive.attempts[0].notes).toBe('first notes before history race');
  expect(archive.files[`attempts/${secondAttemptId}/solution.py`]).toBe(pendingSecondSource);
  expect(archive.attempts[1].notes).toBe('second notes before history race');
});

test('resume post-load failure rolls back model and active attempt before autosave resumes', async ({ page }) => {
  await installTransitionControls(page);
  const { panel, status } = await loadGym(page);
  const notes = panel.getByLabel('Rep notes');
  const firstSource = `${reference}\n# selected resume source must remain intact`;
  const secondSource = 'import numpy as np\n# active source must survive failed resume';

  await replaceAnswer(page, firstSource);
  await notes.fill('selected notes must remain intact');

  await panel.getByRole('button', { name: 'New rep' }).click();
  await expect(status).toContainText('Rep 2');
  await replaceAnswer(page, secondSource);
  await notes.fill('active notes must survive failed resume');

  await failTransitionAt(page, 'resume', 'after-load', 'Injected resume failure after notebook load');
  await panel.getByRole('button', { name: 'Resume rep 1' }).click();
  await expect(status).toContainText('Injected resume failure after notebook load');
  await expect(status).toContainText('restored');
  await expect(notes).toHaveValue('active notes must survive failed resume');
  await expect(answerCell(page)).toContainText('# active source must survive failed resume');

  await runControlledAutosave(page);
  const archive = await exportAllArchive(page);
  expect(archive.attempts).toHaveLength(2);
  expect(archive.activeByExercise['0059']).toBe(archive.attempts[1].id);
  expect(archive.files[`attempts/${archive.attempts[0].id}/solution.py`]).toBe(firstSource);
  expect(archive.attempts[0].notes).toBe('selected notes must remain intact');
  expect(archive.files[`attempts/${archive.attempts[1].id}/solution.py`]).toBe(secondSource);
  expect(archive.attempts[1].notes).toBe('active notes must survive failed resume');
});

test('new attempt post-load failure restores the prior notebook before autosave resumes', async ({ page }) => {
  await installTransitionControls(page);
  const { panel, status } = await loadGym(page);
  const oldSource = 'import numpy as np\n# old source must survive failed new attempt';

  await replaceAnswer(page, oldSource);

  await failTransitionAt(page, 'new', 'after-load', 'Injected new failure after notebook load');
  await panel.getByRole('button', { name: 'New rep' }).click();
  await expect(status).toContainText('Injected new failure after notebook load');
  await expect(status).toContainText('restored');
  await expect(answerCell(page)).toContainText('# old source must survive failed new attempt');

  await runControlledAutosave(page);
  const archive = await exportAllArchive(page);
  expect(archive.attempts).toHaveLength(2);
  expect(archive.activeByExercise['0059']).toBe(archive.attempts[0].id);
  expect(archive.files[`attempts/${archive.attempts[0].id}/solution.py`]).toBe(oldSource);
  expect(archive.files[`attempts/${archive.attempts[1].id}/solution.py`])
    .not.toContain('# old source must survive failed new attempt');
});

test('new attempt post-activation swap failure restores the prior notebook before autosave resumes', async ({ page }) => {
  await installTransitionControls(page);
  const { panel, status } = await loadGym(page);
  const oldSource = 'import numpy as np\n# old source must survive post-activation failure';

  await replaceAnswer(page, oldSource);

  await failTransitionAt(page, 'new', 'after-activation', 'Injected new failure after activation');
  await panel.getByRole('button', { name: 'New rep' }).click();
  await expect(status).toContainText('Injected new failure after activation');
  await expect(status).toContainText('restored');
  await expect(answerCell(page)).toContainText('# old source must survive post-activation failure');

  await runControlledAutosave(page);
  const archive = await exportAllArchive(page);
  expect(archive.attempts).toHaveLength(2);
  expect(archive.activeByExercise['0059']).toBe(archive.attempts[0].id);
  expect(archive.files[`attempts/${archive.attempts[0].id}/solution.py`]).toBe(oldSource);
  expect(archive.files[`attempts/${archive.attempts[1].id}/solution.py`])
    .not.toContain('# old source must survive post-activation failure');
});

test('autosave ignores an older same-exercise attempt tab while a newer attempt is active', async ({ page }) => {
  await installTransitionControls(page);
  let loaded = await loadGym(page);
  let { panel, status } = loaded;
  const firstSource = 'import numpy as np\n# first attempt opened in a separate tab';
  const secondSource = 'import numpy as np\n# active second attempt must remain bound';
  const staleTabSource = 'import numpy as np\n# stale same-exercise tab must not overwrite active';

  await replaceAnswer(page, firstSource);
  await runControlledAutosave(page);
  await panel.getByRole('button', { name: 'New rep' }).click();
  await expect(status).toContainText('Rep 2');
  await replaceAnswer(page, secondSource);
  await runControlledAutosave(page);

  const before = await exportAllArchive(page);
  const firstAttempt = before.attempts[0];
  const secondAttempt = before.attempts[1];
  await openNotebookPath(page, firstAttempt.notebookPath);
  await selectNotebook(page, `${firstAttempt.id}.ipynb`);
  await expect(answerCell(page)).toContainText('# first attempt opened in a separate tab');

  await replaceAnswer(page, staleTabSource);
  await runControlledAutosave(page);

  const archive = await exportAllArchive(page);
  expect(archive.activeByExercise['0059']).toBe(secondAttempt.id);
  expect(archive.files[`attempts/${secondAttempt.id}/solution.py`]).toBe(secondSource);
  expect(JSON.stringify(archive)).not.toContain('# stale same-exercise tab must not overwrite active');
  await expect(answerCell(page)).toContainText('# stale same-exercise tab must not overwrite active');
});

test('background autosave interval persists edits without a controlled tick', async ({ page }) => {
  await loadGym(page);
  const source = 'import numpy as np\n# background timer must still autosave';

  await replaceAnswer(page, source);

  await expect.poll(async () => {
    return page.evaluate(() => window.localStorage.getItem('torch-dojo.browser-gym.attempts.v1') ?? '');
  }, { timeout: 5_000 }).toContain('# background timer must still autosave');
});

test('autosave ignores unrelated notebook tabs while the gym panel remains active', async ({ page }) => {
  const { panel, status, check } = await loadGym(page, 'lab/?exercise=0059');
  const notes = panel.getByLabel('Rep notes');
  const savedSource = `${reference}\n# source that must survive unrelated tab`;

  await replaceAnswer(page, savedSource);
  await notes.fill('notes that must survive unrelated tab');
  await check.click();
  await expect(status).toContainText('INDEPENDENT PASS', { timeout: 120_000 });
  const checkedText = await status.textContent();
  const checkedHash = checkedText?.match(/Receipt source SHA-256 ([0-9a-f]{64})/)?.[1];
  expect(checkedHash).toBeTruthy();

  await page.goto('lab/index.html?exercise=0059&path=notebooks/0060_classifier_step.ipynb');
  await selectNotebook(page, '0060_classifier_step.ipynb');
  await expect(visibleNotebookHeading(page)).toContainText('0060 · One honest classifier update', {
    timeout: 60_000
  });
  await replaceAnswer(page, 'import numpy as np\n# unrelated notebook edit must not become attempt source');
  await page.waitForTimeout(1200);

  await dispatchButtonClick(page, 'Open exercise');
  await expect(visibleNotebookHeading(page)).toContainText('0059 · Stable Softmax Cross-Entropy', {
    timeout: 60_000
  });
  await expect(answerCell(page)).toContainText('# source that must survive unrelated tab');
  await expect(notes).toHaveValue('notes that must survive unrelated tab');
  await expect(status).toContainText(`Receipt source SHA-256 ${checkedHash}`);

  await openTab(page, 'Reps');
  const download = await Promise.all([
    page.waitForEvent('download'),
    panel.getByRole('button', { name: 'Export rep' }).click()
  ]).then(([item]) => item);
  const archivePath = await download.path();
  expect(archivePath).toBeTruthy();
  const archive = JSON.parse(readFileSync(archivePath!, 'utf8'));
  expect(archive.attempts).toHaveLength(1);
  expect(archive.attempts[0].notes).toBe('notes that must survive unrelated tab');
  expect(archive.attempts[0].receipts[0].source).toBe(savedSource);
  expect(archive.files[`attempts/${archive.attempts[0].id}/solution.py`]).toBe(savedSource);
});
