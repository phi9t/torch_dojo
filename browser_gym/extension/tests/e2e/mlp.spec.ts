import { expect, test } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { answerEditor, answerPane, openTab, runExplorePanel, scratchEditor } from './bench';

const reference = readFileSync(resolve(__dirname, '../../../labs/mlp/reference.py'), 'utf8');
const starter = readFileSync(resolve(__dirname, '../../../labs/mlp/starter.py'), 'utf8');
const mutant = readFileSync(resolve(__dirname, '../../../labs/mlp/mutants.py'), 'utf8');
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

async function replaceSolution(page: import('@playwright/test').Page, source: string) {
  const solution = answerEditor(page);
  await solution.click();
  await page.keyboard.press('ControlOrMeta+A');
  await page.keyboard.insertText(source);
}

test('MLP lab checks every checkpoint in fresh workers with bounded experiment views', async ({ page }) => {
  await page.goto('lab/?exercise=mlp');
  await expect(page.locator('.jp-MarkdownCell h1').first()).toContainText('MLP Modeling Lab');

  await expect(page.getByRole('button', { name: 'Run cell', exact: true })).toBeEnabled();
  await runExplorePanel(page);
  await expect(page.getByRole('group', { name: 'Tensor card for train_X' })).toContainText('(96, 2)', { timeout: 60_000 });
  await expect(page.getByTestId('dojo-mlp-gradient-view')).toContainText('waiting for Run', { timeout: 60_000 });
  await page.getByTestId('dojo-mlp-loss-run').click();
  await expect(page.getByTestId('dojo-mlp-loss-view')).toContainText('Loss view: epoch 0 train');
  await expect(page.getByTestId('dojo-mlp-loss-plot').locator('polyline')).toHaveCount(2);
  await page.getByTestId('dojo-mlp-boundary-run').click();
  await expect(page.getByTestId('dojo-mlp-boundary-view')).toContainText('625 fixed grid points');
  await expect(page.getByTestId('dojo-mlp-boundary-plot').locator('rect')).toHaveCount(626);
  await page.getByTestId('dojo-mlp-confidence-run').click();
  await expect(page.getByTestId('dojo-mlp-confidence-view')).toContainText('validation confidence mean');
  await expect(page.getByTestId('dojo-mlp-confidence-plot').locator('rect')).toHaveCount(5);
  await page.getByTestId('dojo-mlp-gradient-run').click();
  await expect(page.getByTestId('dojo-mlp-gradient-view')).toContainText('Gradient block norms:');
  await page.getByTestId('dojo-mlp-diagnosis-run').click();
  await expect(page.getByTestId('dojo-mlp-diagnosis-view')).toContainText('Planted defect: omitted_batch_mean');
  await expect(page.getByTestId('dojo-mlp-diagnosis-view')).toContainText('broken final valid loss');
  await expect(page.getByTestId('dojo-mlp-diagnosis-plot').locator('polyline')).toHaveCount(2);

  const status = page.locator('#torch-dojo-gym [role="status"]');
  const check = page.getByRole('button', { name: /^Check (?!log)/ });
  const stage = page.getByLabel('Check stage');
  await expect(stage).toHaveValue('init');

  await replaceSolution(page, reference);
  for (const checkpoint of ['init', 'forward', 'backward', 'train', 'inference', 'evaluate', 'state', 'diagnose']) {
    await stage.selectOption(checkpoint);
    await check.click();
    await expect(status).toContainText('PASSED · source SHA-256', { timeout: 120_000 });
    await expect(status).toContainText(`checkpoint_${checkpoint}`);
    if (checkpoint === 'state') {
      await expect(status).toContainText('"parameters_restored":true');
    }
  }

  await replaceSolution(page, starter);
  await stage.selectOption('forward');
  await check.click();
  await expect(status).toContainText('FAILED · source SHA-256', { timeout: 120_000 });
  await expect(status).toContainText('NotImplementedError');

  await replaceSolution(page, mutant);
  await stage.selectOption('backward');
  await check.click();
  await expect(status).toContainText('FAILED · source SHA-256', { timeout: 120_000 });
  await expect(status).toContainText('gradient magnitude');

  await stage.selectOption('diagnose');
  await check.click();
  await expect(status).toContainText('FAILED · source SHA-256', { timeout: 120_000 });
  await expect(status).toContainText('omitted mini-batch mean');
  await expect(status).toContainText('candidate_max_abs_error');

  await replaceSolution(page, reference);
  await stage.selectOption('diagnose');
  await check.click();
  await expect(status).toContainText('PASSED · source SHA-256', { timeout: 120_000 });
  await expect(status).toContainText('candidate_max_abs_error');

  await replaceSolution(page, substitutedDiagnosis);
  await stage.selectOption('diagnose');
  await check.click();
  await expect(status).toContainText('PASSED · source SHA-256', { timeout: 120_000 });
  await expect(status).toContainText('substitution_loss_and_grads');
  await expect(status).toContainText('"independent_credit":false');
  await expect(status).toContainText('REFERENCE-ASSISTED CONTINUATION');

  await openTab(page, 'Reps');
  const exported = await Promise.all([
    page.waitForEvent('download'),
    page.getByRole('button', { name: 'Export all' }).click()
  ]).then(([download]) => download);
  const archivePath = await exported.path();
  expect(archivePath).toBeTruthy();
  const archive = JSON.parse(readFileSync(archivePath!, 'utf8'));
  const receipts = archive.attempts.flatMap((attempt: { receipts: unknown[] }) => attempt.receipts);
  const plantedReceipt = receipts.find((receipt: { source: string; status: string; result: { stage?: string } }) =>
    receipt.source === mutant && receipt.status === 'failed' && receipt.result.stage === 'diagnose'
  );
  expect(plantedReceipt).toBeTruthy();
  const correctedReceipt = receipts.find((receipt: { source: string; status: string; result: { stage?: string } }) =>
    receipt.source === reference && receipt.status === 'passed' && receipt.result.stage === 'diagnose'
  );
  expect(correctedReceipt).toBeTruthy();
  const substitutedReceipt = receipts.find((receipt: { source: string; status: string; result: { stage?: string; substitutions?: unknown[] } }) =>
    receipt.source === substitutedDiagnosis &&
    receipt.status === 'passed' &&
    receipt.result.stage === 'diagnose' &&
    (receipt.result.substitutions?.length ?? 0) === 1
  );
  expect(substitutedReceipt).toBeTruthy();

  await replaceSolution(page, `from browser_gym.labs.mlp.reference import train_epoch\n`);
  await stage.selectOption('train');
  await check.click();
  await expect(status).toContainText('FAILED · source SHA-256', { timeout: 120_000 });
  await expect(status).toContainText('reference components are not importable');

  await replaceSolution(page, `
import numpy as np

REFERENCE_OVERRIDES = {"loss_and_grads"}
MOMENTUM = 0.9


def train_epoch(X, y, params, state, rng, batch_size, learning_rate):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    next_params = {name: np.asarray(value, dtype=np.float64).copy() for name, value in params.items()}
    raw_velocity = state.get("velocity", {}) if isinstance(state, dict) else {}
    step = int(state.get("step", 0)) if isinstance(state, dict) else 0
    velocity = {
        name: np.asarray(raw_velocity.get(name, np.zeros_like(value)), dtype=np.float64).copy()
        for name, value in next_params.items()
    }
    order = rng.permutation(len(X))
    batch_losses = []
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        loss, grads = loss_and_grads(X[indices], y[indices], next_params)
        batch_losses.append(float(loss))
        for name in next_params:
            velocity[name] = MOMENTUM * velocity[name] + np.asarray(grads[name], dtype=np.float64)
            next_params[name] = next_params[name] - learning_rate * velocity[name]
        step += 1
    return next_params, {"step": step, "velocity": velocity}, {
        "batch_losses": np.asarray(batch_losses, dtype=np.float64),
        "mean_loss": float(np.mean(batch_losses)),
        "batch_count": len(batch_losses),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
    }
`);
  await stage.selectOption('train');
  await check.click();
  await expect(status).toContainText('PASSED · source SHA-256', { timeout: 120_000 });
  await expect(status).toContainText('substitution_loss_and_grads');
  await expect(status).toContainText('"independent_credit":false');
  await expect(status).toContainText('REFERENCE-ASSISTED CONTINUATION');

  await replaceSolution(page, `print(shape_probe)\n${reference}`);
  await stage.selectOption('init');
  await check.click();
  await expect(status).toContainText('NameError', { timeout: 120_000 });
});
