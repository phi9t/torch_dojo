import { expect, test, type Page } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { replaceAnswer } from './bench';

const evalSource = readFileSync(resolve(__dirname, '../../../../dojo/0059_stable_cross_entropy_eval.py'), 'utf8');
const reference = 'import numpy as np\n' + evalSource.match(/def softmax_cross_entropy[\s\S]*?(?=\n\n# %% \[verifiers\])/)![0];

async function openWorkout(page: Page) {
  await page.goto('lab/?exercise=0059');
  await expect(page.getByRole('button', { name: 'Check baseline' })).toBeEnabled({ timeout: 60_000 });
}

function results(page: Page) {
  return page.getByRole('region', { name: 'Check results' });
}

test('the grid shows one square per verifier, selects the first failure, and hatches unreached stages', async ({ page }) => {
  await openWorkout(page);
  const grid = results(page);
  // Before any check: one quiet line, no grid.
  await expect(grid).toHaveText('Check your answer to see each verifier here.');
  await expect(grid.getByRole('button')).toHaveCount(0);

  // The starter fails at baseline; follow-up and production are not reached.
  await page.getByRole('button', { name: 'Check baseline' }).click();
  await expect(grid).toContainText('0 passed', { timeout: 120_000 });
  const baseline = grid.getByRole('group', { name: 'Baseline' });
  await expect(baseline.getByRole('button')).toHaveCount(3);
  const first = baseline.getByRole('button').first();
  await expect(first).toHaveAccessibleName(/^baseline equal logits and single example: failed$/);
  await expect(first).toHaveAttribute('aria-pressed', 'true');
  await expect(grid.getByRole('group', { name: 'Follow-up' }).getByRole('img', { name: /not reached$/ })).toHaveCount(2);
  await expect(grid.getByRole('group', { name: 'Production' }).getByRole('img', { name: /not reached$/ })).toHaveCount(4);
  const detail = grid.getByRole('region', { name: 'Verifier detail' });
  await expect(detail).toContainText('Baseline · failed');
  await expect(detail).toContainText('Inputs');
  await expect(detail).toContainText('Expected');
  await expect(detail).toContainText('Your output');

  // Squares are keyboard-navigable.
  await first.focus();
  await page.keyboard.press('ArrowRight');
  await expect(baseline.getByRole('button').nth(1)).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(baseline.getByRole('button').nth(1)).toHaveAttribute('aria-pressed', 'true');
  await expect(detail).toContainText('baseline mean reduction');

  // The stage selector, the runtime readout and the receipt log stay available.
  const stage = page.getByLabel('Check stage');
  await stage.selectOption('production');
  await expect(page.getByRole('button', { name: 'Check production' })).toBeVisible();
  await expect(page.getByText(/pyodide · Python [\d.]+ · NumPy [\d.]+ · checker \w+/)).toBeVisible();
  await expect(page.locator('#torch-dojo-gym [role="status"]')).toContainText('Receipt 1');

  // The reference passes every verifier.
  await replaceAnswer(page, reference);
  await page.getByRole('button', { name: 'Check production' }).click();
  await expect(grid).toContainText('9 passed', { timeout: 120_000 });
  await expect(grid).toContainText('0 failed');
  await expect(grid.getByRole('button', { name: /: passed$/ })).toHaveCount(9);
  await expect(grid.getByRole('img', { name: /not reached$/ })).toHaveCount(0);
  await expect(grid.getByRole('region', { name: 'Verifier detail' })).toBeHidden();

  // An unstable answer fails the extreme-logits verifier first, and that square is selected.
  const unstable = reference.replace('q = selected - selected.max(axis=1, keepdims=True)', 'q = selected');
  expect(unstable).not.toBe(reference);
  await replaceAnswer(page, unstable);
  await page.getByRole('button', { name: 'Check production' }).click();
  await expect(grid).toContainText('2 failed', { timeout: 120_000 });
  await expect(grid.getByRole('button', { name: 'production extreme logits and row shifts: failed' }))
    .toHaveAttribute('aria-pressed', 'true');
});

test('a running check shows a quiet indicator, and a stopped check still writes its receipt', async ({ page }) => {
  await openWorkout(page);
  const grid = results(page);
  await replaceAnswer(page, `${reference}\nimport time\nwhile True:\n    time.sleep(10)`);
  await page.getByRole('button', { name: 'Check baseline' }).click();
  await expect(grid).toContainText('Checking baseline in a fresh worker');
  await expect(page.getByRole('button', { name: 'Stop' })).toBeEnabled({ timeout: 15_000 });
  await page.getByRole('button', { name: 'Stop' }).click();
  const status = page.locator('#torch-dojo-gym [role="status"]');
  await expect(status).toContainText('STOPPED · source SHA-256', { timeout: 15_000 });
  await expect(status).toContainText('Receipt 1: STOPPED');
  await expect(grid).toContainText('Check stopped by the learner');
  await expect(grid.getByRole('button', { name: /: not run$/ })).toHaveCount(3);
});
