import { expect, test, type Page } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { openTab, problemPane, replaceAnswer, seal, watchStamp } from './bench';

const reference = readFileSync(resolve(__dirname, '../../../labs/mlp/reference.py'), 'utf8');
const substitutedDiagnosis = `
REFERENCE_OVERRIDES = {"loss_and_grads"}


def diagnose_mlp_defect():
    return {
        "defect_id": "omitted_batch_mean",
        "diagnosis": "The planted backward pass omits the mini-batch mean scale for dlogits.",
        "failed_invariant": "Directional finite differences must match every parameter-block gradient.",
        "evidence": "The broken analytic directional derivatives are about the mini-batch size larger than finite differences.",
        "repair": "Divide dlogits by the mini-batch size before computing W1, b1, W2, and b2 gradients.",
        "regression_check": "Run the backward checkpoint and require abs_error below 1e-6 for every block.",
    }
`;

async function openLab(page: Page) {
  await page.goto('lab/?exercise=mlp');
  await expect(page.getByRole('button', { name: /^Check (?!log)/ })).toBeEnabled({ timeout: 60_000 });
}

async function checkCheckpoint(page: Page, checkpoint: string, outcome = 'PASSED') {
  await page.getByLabel('Check stage').selectOption(checkpoint);
  await page.getByRole('button', { name: /^Check (?!log)/ }).click();
  await expect(page.locator('#torch-dojo-gym [role="status"]')).toContainText(`${outcome} · source SHA-256`, { timeout: 120_000 });
}

test('the lab brief is typed by checkpoint, and its drills run', async ({ page }) => {
  await openLab(page);
  const brief = problemPane(page);
  await expect(brief.getByRole('heading', { name: 'Goal', exact: true })).toBeVisible();
  await expect(brief.getByRole('heading', { name: /^Backward/ })).toBeVisible();
  await expect(brief.getByRole('heading', { name: 'Watch out', exact: true }).first()).toBeVisible();

  await openTab(page, 'Drills');
  const drill = problemPane(page).getByRole('group', { name: 'Gradient through ReLU at zero' });
  await drill.getByLabel('Your prediction').fill('[0, 0, 1]');
  await drill.getByRole('button', { name: 'Run', exact: true }).click();
  await expect(drill.locator('.jp-OutputArea')).toContainText('[0. 0. 1.]', { timeout: 60_000 });
  await drill.getByRole('button', { name: 'Reveal reasoning' }).click();
  await expect(drill).toContainText('counts as inactive');
});

test('checkpoints pass in any order, each with its own split, stamp and squares', async ({ page }) => {
  await page.clock.install();
  await openLab(page);
  await expect(seal(page, 'Forward', 'Checkpoints')).toHaveAttribute('aria-label', 'Forward: not passed yet');
  await page.mouse.move(220, 300);
  await page.clock.runFor(40_000);
  await replaceAnswer(page, reference);

  const forwardStamped = await watchStamp(page, 'Forward', 'Checkpoints');
  const initStamped = await watchStamp(page, 'Init', 'Checkpoints');
  await checkCheckpoint(page, 'forward');
  await expect.poll(forwardStamped).toBe(true);
  expect(await initStamped()).toBe(false);
  await expect(seal(page, 'Forward', 'Checkpoints')).toHaveAttribute('aria-label', /^Forward: passed, in 0:(39|4\d)$/);
  await expect(seal(page, 'Init', 'Checkpoints')).toHaveAttribute('aria-label', 'Init: not passed yet');
  // Every checkpoint has its own group; ones not checked yet show as not run.
  const results = page.getByRole('region', { name: 'Check results' });
  await expect(results.getByRole('group', { name: 'Forward' }).getByRole('button', { name: /checkpoint forward: passed/ })).toBeVisible();
  await expect(results.getByRole('group', { name: 'Init' }).getByRole('button', { name: /checkpoint init: not run/ })).toBeVisible();

  await page.mouse.move(260, 320);
  await page.clock.runFor(20_000);
  await checkCheckpoint(page, 'init');
  await expect.poll(initStamped).toBe(true);
  await expect(seal(page, 'Init', 'Checkpoints')).toHaveAttribute('aria-label', /^Init: passed, in (0:59|1:0\d)$/);
  // The forward result from the earlier check stays on the grid.
  const forward = results.getByRole('group', { name: 'Forward' }).getByRole('button', { name: /checkpoint forward: passed/ });
  await forward.click();
  await expect(results.getByRole('region', { name: 'Verifier detail' })).toContainText('from an earlier check');
});

test('a pass that used a reference component says so in the detail panel', async ({ page }) => {
  await openLab(page);
  await replaceAnswer(page, substitutedDiagnosis);
  await checkCheckpoint(page, 'diagnose');
  const results = page.getByRole('region', { name: 'Check results' });
  await results.getByRole('group', { name: 'Diagnose' }).getByRole('button', { name: /checkpoint diagnose: passed/ }).click();
  await expect(results.getByRole('region', { name: 'Verifier detail' }))
    .toContainText('Reference substitution: this check used the reference loss_and_grads (Backward checkpoint), so its pass is not independent credit.');
});
