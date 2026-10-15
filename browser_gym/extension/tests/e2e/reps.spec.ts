import { expect, test, type Page } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { openTab, replaceAnswer, seal, watchStamp } from './bench';

const evalSource = readFileSync(resolve(__dirname, '../../../../dojo/0059_stable_cross_entropy_eval.py'), 'utf8');
const reference = 'import numpy as np\n' + evalSource.match(/def softmax_cross_entropy[\s\S]*?(?=\n\n# %% \[verifiers\])/)![0];

async function openWorkout(page: Page) {
  await page.clock.install();
  await page.goto('lab/?exercise=0059');
  await expect(page.getByRole('button', { name: /^Check (?!log)/ })).toBeEnabled({ timeout: 60_000 });
}

let pointer = 0;
/** Learner input: the timer only counts while there is some within the last two minutes. */
async function nudge(page: Page) {
  pointer = (pointer + 37) % 400;
  await page.mouse.move(200 + pointer, 300);
}

async function setPageVisible(page: Page, visible: boolean) {
  await page.evaluate(isVisible => {
    Object.defineProperty(document, 'visibilityState', { configurable: true, get: () => (isVisible ? 'visible' : 'hidden') });
    document.dispatchEvent(new Event('visibilitychange'));
  }, visible);
}

async function checkStage(page: Page, stage: 'baseline' | 'production') {
  await page.getByLabel('Check stage').selectOption(stage);
  await page.getByRole('button', { name: `Check ${stage}` }).click();
  await expect(page.locator('#torch-dojo-gym [role="status"]')).toContainText('PASSED · source SHA-256', { timeout: 120_000 });
}

test('the rep timer counts only active time and pauses while hidden or idle', async ({ page }) => {
  await openWorkout(page);
  const timer = page.getByRole('timer');
  await nudge(page);
  await page.clock.runFor(30_000);
  await expect(timer).toHaveText(/^0:(29|30|31)$/);

  // A hidden tab adds nothing.
  await setPageVisible(page, false);
  await page.clock.runFor(60_000);
  await expect(timer).toHaveText(/^0:(29|30|31)$/);
  await setPageVisible(page, true);
  await nudge(page);
  await page.clock.runFor(10_000);
  await expect(timer).toHaveText(/^0:(39|40|41)$/);

  // Two minutes without input and the timer stops counting.
  await page.clock.runFor(180_000);
  await expect(timer).toHaveText(/^2:(29|30|31)$/);
  await expect(timer).toHaveAttribute('aria-label', /^Active time 2 minutes (29|30|31) seconds$/);

  // It can be hidden, and stays hidden.
  await page.getByRole('button', { name: 'Hide timer' }).click();
  await expect(timer).toBeHidden();
  await page.reload();
  await expect(page.getByRole('button', { name: 'Show timer' })).toBeVisible({ timeout: 60_000 });
});

test('splits are recorded per stage, stamped, summarised, and compared across reps', async ({ page }) => {
  await openWorkout(page);
  await nudge(page);
  await page.clock.runFor(60_000);
  await replaceAnswer(page, reference);
  const stamped = await watchStamp(page, 'Baseline');
  await checkStage(page, 'baseline');
  await expect(seal(page, 'Baseline')).toHaveAttribute('aria-label', /^Baseline: passed, in (0:59|1:0\d)$/);
  await expect(seal(page, 'Follow-up')).toHaveAttribute('aria-label', 'Follow-up: not passed yet');
  // Passing a stage stamps its seal with a short animation.
  await expect.poll(stamped).toBe(true);

  await nudge(page);
  await page.clock.runFor(60_000);
  await checkStage(page, 'production');
  await expect(seal(page, 'Production')).toHaveAttribute('aria-label', /^Production: passed, in (1:59|2:0\d)$/);
  await expect(seal(page, 'Follow-up')).toHaveAttribute('aria-label', /^Follow-up: passed, in (1:59|2:0\d)$/);
  const summary = page.getByText(/^Rep done in (1:59|2:0\d) · your first timed finish · no hints\./);
  await expect(summary).toBeVisible();
  await expect(page.getByRole('link', { name: /^Next up: 0060/ })).toBeVisible();
  // The final stage passed, so the rep's time is final.
  const finalTime = await page.getByRole('timer').textContent();
  await nudge(page);
  await page.clock.runFor(30_000);
  await expect(page.getByRole('timer')).toHaveText(finalTime!);

  // A new rep starts from zero, resets the answer and keeps the first rep's best times.
  await openTab(page, 'Reps');
  await page.getByRole('button', { name: 'New rep' }).click();
  await expect(page.getByRole('timer')).toHaveText('0:00');
  await expect(summary).toBeHidden();
  await nudge(page);
  await page.clock.runFor(20_000);
  await replaceAnswer(page, reference);
  await checkStage(page, 'baseline');
  await expect(seal(page, 'Baseline')).toHaveAttribute('aria-label', /^Baseline: passed, in 0:(19|2\d), best (0:59|1:0\d)$/);
  await expect(seal(page, 'Baseline')).toContainText(/−0:\d\d/);

  const table = page.getByRole('table', { name: 'Splits per rep' });
  await expect(table.getByRole('row')).toHaveCount(3);
  await expect(table.getByRole('row', { name: /Rep 1 · best/ })).toContainText(/(0:59|1:0\d).*(1:59|2:0\d).*(1:59|2:0\d)/);
  await expect(table.getByRole('row', { name: /Rep 2 \(current\)/ })).toContainText(/0:(19|2\d)/);
});

test('with reduced motion the stamp appears without animation', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await openWorkout(page);
  await replaceAnswer(page, reference);
  const stamped = await watchStamp(page, 'Baseline');
  await checkStage(page, 'baseline');
  await expect(seal(page, 'Baseline')).toHaveAttribute('aria-label', /^Baseline: passed/);
  await page.waitForTimeout(1000);
  expect(await stamped()).toBe(false);
});

test('a rep saved before timing shows unknown splits as a dash, never 0:00', async ({ page }) => {
  await openWorkout(page);
  await openTab(page, 'Reps');
  await page.getByLabel('Import reps').setInputFiles(resolve(__dirname, '../../src/fixtures/archive-before-rep-rename.json'));
  await expect(page.locator('#torch-dojo-gym [role="status"]')).toContainText('Imported reps: 1', { timeout: 60_000 });
  await expect(seal(page, 'Baseline')).toHaveAttribute('aria-label', 'Baseline: passed, time unknown');
  await expect(seal(page, 'Baseline')).toContainText('—');
  const row = page.getByRole('table', { name: 'Splits per rep' }).getByRole('row', { name: /\(current\)/ });
  await expect(row).toContainText('—');
  await expect(row).not.toContainText('0:00');
  await expect(row).not.toContainText('best');
});
