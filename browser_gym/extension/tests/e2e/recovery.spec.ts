import { expect, test } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { answerEditor, answerPane, runExplorePanel, scratchEditor } from './bench';

const evalSource = readFileSync(resolve(__dirname, '../../../../dojo/0059_stable_cross_entropy_eval.py'), 'utf8');
const referenceFunction = evalSource.match(/def softmax_cross_entropy[\s\S]*?(?=\n\n# %% \[verifiers\])/)?.[0];
if (!referenceFunction) throw new Error('Could not find canonical reference function');
const reference = `import numpy as np\n${referenceFunction}`;

const INF_LOOP = `${reference}\nimport time\nwhile True:\n    time.sleep(10)`;

test('stop a runaway check, write a source-bound stopped receipt, then recover with a fresh worker', async ({ page }) => {
  await page.goto('lab/?exercise=0059');
  await expect(page.locator('.jp-MarkdownCell h1').first()).toContainText('0059 · Stable Softmax Cross-Entropy');
  const status = page.locator('#torch-dojo-gym [role="status"]');
  const answer = answerEditor(page);

  async function replaceAnswer(source: string) {
    await answer.click();
    await page.keyboard.press('ControlOrMeta+A');
    await page.keyboard.insertText(source);
  }

  const check = page.getByRole('button', { name: /^Check (?!log)/ });
  const stop = page.getByRole('button', { name: 'Stop' });
  await expect(check).toBeVisible();
  // Stop starts disabled because no check is in flight.
  await expect(stop).toBeVisible();
  await expect(stop).toBeDisabled();

  // Seed a known good receipt first so we can prove it survives Stop/recovery.
  await replaceAnswer(reference);
  await check.click();
  await expect(status).toContainText('PASSED · source SHA-256', { timeout: 180_000 });
  await expect(status).toContainText('Receipt 1');
  const goodReceipt = await status.textContent();
  const goodHash = goodReceipt?.match(/PASSED · source SHA-256 ([0-9a-f]{64})/)?.[1];
  expect(goodHash).toBeTruthy();

  // Submit a runaway infinite loop. Stop button must enable while the check is
  // in flight. We click Stop instead of waiting 120s.
  await replaceAnswer(INF_LOOP);
  const runawaySubmit = check.click();
  // Give the panel time to disable the check button and enable Stop.
  await expect(check).toBeDisabled({ timeout: 15_000 });
  await expect(stop).toBeEnabled({ timeout: 15_000 });
  await stop.click();
  // Buttons reset once the stopped receipt is appended.
  await expect(check).toBeEnabled({ timeout: 15_000 });
  await expect(stop).toBeDisabled({ timeout: 15_000 });
  await runawaySubmit.catch(() => { /* swallow the rejected Playwright click; the DOM flow is what we verify */ });

  // A stopped receipt must appear and be source-bound (shows the source hash
  // and the submitted source cell indices).
  await expect(status).toContainText('STOPPED · source SHA-256', { timeout: 15_000 });
  await expect(status).toContainText('Receipt 2: STOPPED');
  await expect(status).toContainText('source cells');
  await expect(status).toContainText('Receipt 1'); // prior receipt must remain visible
  // Prior receipt was valid, but after the second answer replace with INF_LOOP
  // its source hash differs, so the panel correctly labels it STALE PASS.
  const receipt1Text = await status.textContent();
  expect(receipt1Text).toMatch(/Receipt 1: (PASSED|STALE PASS)/);

  // Stopped receipts must explain saved vs discarded state.
  const stoppedText = await status.textContent();
  expect(stoppedText).toContain('Saved:');
  expect(stoppedText).toContain('Discarded:');

  // A subsequent valid check must run in a *fresh* worker and succeed.
  await replaceAnswer(reference);
  await check.click();
  await expect(status).toContainText('PASSED · source SHA-256', { timeout: 180_000 });
  await expect(status).toContainText('Receipt 3: PASSED');
  // Receipt 2 (stopped) and 1 (earlier pass) must still be visible.
  await expect(status).toContainText('Receipt 2: STOPPED');
  await expect(status).toContainText('Receipt 1: PASSED');
});

test('exploration kernel offers interrupt with a clear fallback restart path', async ({ page }) => {
  await page.goto('lab/?exercise=0059');
  await expect(page.locator('.jp-MarkdownCell h1').first()).toContainText('0059');

  // Narrow to the gym notebook controls to avoid colliding with
  // JupyterLab toolbar buttons like "Interrupt the kernel".
  const gymPanel = page.getByRole('complementary', { name: 'Workout' });
  const interrupt = gymPanel.getByRole('button', { name: 'Interrupt', exact: true });
  const restart = gymPanel.getByRole('button', { name: 'Restart kernel', exact: true });
  await expect(interrupt).toBeVisible();
  await expect(restart).toBeVisible();

  // Interrupt has an accessible title/tooltip distinguishing the cooperative
  // path vs. fallback restart.
  await expect(interrupt).toHaveAttribute('title', /SharedArrayBuffer|cross-origin|Restart kernel/);
  await expect(restart).toHaveAttribute('title', /Discard.*kernel state|remain available/);

  // Restart dialog explicitly separates discarded live state from saved state.
  page.once('dialog', dialog => {
    const msg = dialog.message();
    expect(msg).toMatch(/DISCARDED/);
    expect(msg).toMatch(/Live kernel state|variables|imports/);
    expect(msg).toMatch(/Saved state|remain available/);
    expect(msg).toMatch(/receipts|notebooks|notes|reps/);
    void dialog.dismiss();
  });
  await restart.click();
  // Dialog handler above confirms the explanation text; we dismissed it so
  // status should not show "kernel restarted" yet.
  const status = page.locator('#torch-dojo-gym [role="status"]');
  await expect(status).not.toContainText('kernel restarted', { timeout: 5_000 });
});

test('check deadline produces a timeout receipt without waiting forever', async ({ page }) => {
  // Exercise the controller-owned deadline path directly by temporarily
  // forcing the timeout down via page-exposed override.
  await page.goto('lab/?exercise=0059');
  await expect(page.locator('.jp-MarkdownCell h1').first()).toContainText('0059');
  const status = page.locator('#torch-dojo-gym [role="status"]');
  const check = page.getByRole('button', { name: /^Check (?!log)/ });
  const stop = page.getByRole('button', { name: 'Stop' });
  const answer = answerEditor(page);

  async function replaceAnswer(source: string) {
    await answer.click();
    await page.keyboard.press('ControlOrMeta+A');
    await page.keyboard.insertText(source);
  }

  // Install a shim that reduces controller deadline to 5s so the test doesn't
  // actually wait 120s. We do this by monkeypatching the checker module via
  // the plugin's internal `runCheck` default options. The safest public seam
  // is to replace window.setTimeout during checkout; the panel always waits
  // for the worker or the timeout, whichever fires first.
  await page.addInitScript(() => {
    const originalSetTimeout = window.setTimeout;
    // The controller's 120_000ms check deadline is a call to window.setTimeout.
    // Shorten *only* abnormally long timeouts to keep short ones intact.
    (window as typeof window & { __dojoShortenDeadline?: boolean }).__dojoShortenDeadline = true;
    (window.setTimeout as typeof setTimeout) = ((fn: TimerHandler, delay?: number, ...args: unknown[]) => {
      const d = (window as typeof window & { __dojoShortenDeadline?: boolean }).__dojoShortenDeadline
        ? (delay && delay >= 100_000 ? 5_000 : delay)
        : delay;
      return originalSetTimeout(fn, d, ...args);
    }) as typeof setTimeout;
  });

  // Reload so the init script applies.
  await page.reload();
  await expect(page.locator('.jp-MarkdownCell h1').first()).toContainText('0059', { timeout: 60_000 });

  await replaceAnswer(`${reference}\nimport time\nwhile True:\n    time.sleep(60)`);
  await check.click();
  await expect(check).toBeDisabled({ timeout: 10_000 });
  // Stop stays enabled until the controller deadline fires.
  await expect(stop).toBeEnabled({ timeout: 10_000 });

  // Deadline will fire at 5s instead of 120s. We expect a TIMEOUT receipt
  // that is source-bound and explains saved/discarded state.
  await expect(status).toContainText('TIMEOUT · source SHA-256', { timeout: 20_000 });
  await expect(status).toContainText('Receipt 1: TIMEOUT');
  await expect(status).toContainText('source cells');
  const timeoutText = await status.textContent();
  expect(timeoutText).toContain('Saved:');
  expect(timeoutText).toContain('Discarded:');
  const results = page.getByRole('region', { name: 'Check results' });
  await expect(results).toContainText('Check deadline reached');
  await expect(results.getByRole('button', { name: /: not run$/ })).toHaveCount(3);

  // After timeout a fresh worker still works.
  await replaceAnswer(reference);
  await check.click();
  await expect(status).toContainText('PASSED · source SHA-256', { timeout: 180_000 });
  await expect(status).toContainText('Receipt 2: PASSED');
  await expect(status).toContainText('Receipt 1: TIMEOUT');
});
