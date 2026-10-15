import { expect, test } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { answerEditor, answerPane, explorePanels, openTab, runExplorePanel } from './bench';

const evalSource = readFileSync(resolve(__dirname, '../../../../dojo/0059_stable_cross_entropy_eval.py'), 'utf8');
const referenceFunction = evalSource.match(/def softmax_cross_entropy[\s\S]*?(?=\n\n# %% \[verifiers\])/)?.[0];
if (!referenceFunction) throw new Error('Could not find canonical reference function');
const reference = `import numpy as np\n${referenceFunction}`;

const classifierEvalSource = readFileSync(resolve(__dirname, '../../../../dojo/0060_classifier_step_eval.py'), 'utf8');
const classifierReferenceFunction = classifierEvalSource.match(/def classifier_step[\s\S]*?(?=\n\n# %% \[verifiers\])/)?.[0];
if (!classifierReferenceFunction) throw new Error('Could not find canonical classifier reference function');
const classifierReference = `import numpy as np\n${classifierReferenceFunction}`;

test('0059 visual probe shows tensor evidence and updates loss without joining submitted source', async ({ page }) => {
  await page.goto('lab/?exercise=0059');
  await expect(page.locator('.jp-MarkdownCell h1').first()).toContainText('0059 · Stable Softmax Cross-Entropy');

  await expect(answerPane(page).locator('#torch-dojo-gym')).toBeVisible();
  await openTab(page, 'Reps');
  await expect(page.getByLabel('Rep notes')).toBeVisible();
  // Explore panels show output and a Run control; the loader code stays behind a disclosure.
  const probeCell = (await explorePanels(page)).first();
  await expect(probeCell.locator('.cm-content')).toBeHidden();
  await probeCell.getByRole('button', { name: 'Show code' }).click();
  await expect(probeCell.locator('.cm-content')).toContainText('pyfetch');
  await expect(probeCell.locator('.cm-content')).not.toContainText('def _preview');
  await probeCell.getByRole('button', { name: 'Hide code' }).click();
  await expect(probeCell.locator('.cm-content')).toBeHidden();
  await expect(answerEditor(page)).toBeVisible();
  await runExplorePanel(page);

  const card = page.getByRole('group', { name: 'Tensor card for logits_probe' });
  await expect(card).toContainText('shape', { timeout: 60_000 });
  await expect(card).toContainText('(2, 2)');
  await expect(card).toContainText('dtype');
  await expect(card).toContainText('float64');
  await expect(card).toContainText('batch x class');
  await card.focus();
  await expect(card).toBeFocused();

  const loss = page.getByTestId('dojo-loss-value');
  await expect(loss).toContainText('Loss at scale 1.00:', { timeout: 60_000 });
  const firstLoss = await loss.textContent();

  const slider = page.getByRole('slider', { name: 'Logit scale' });
  await slider.focus();
  await expect(slider).toBeFocused();
  await slider.press('ArrowRight');
  await expect(loss).not.toHaveText(firstLoss ?? '');
  await expect(slider).toHaveValue('1.25');
  await expect(page.getByTestId('dojo-probe-runs')).toContainText('Probe runs: 2');
  await expect(page.locator('#dojo-loss-plot')).toHaveAttribute('role', 'img');
  await page.locator('#dojo-loss-plot').focus();
  await expect(page.locator('#dojo-loss-plot')).toBeFocused();
  await page.setViewportSize({ width: 390, height: 760 });
  await expect(card).toBeVisible();
  await expect(slider).toBeVisible();
  await expect(page.locator('#dojo-loss-plot')).toBeVisible();
  await page.setViewportSize({ width: 1280, height: 720 });

  const answer = answerEditor(page);
  await answer.click();
  await page.keyboard.press('ControlOrMeta+A');
  await page.keyboard.insertText(`print(logits_probe)\n${reference}`);

  const status = page.locator('#torch-dojo-gym [role="status"]');
  await page.getByRole('button', { name: /^Check (?!log)/ }).click();
  await expect(status).toContainText('NameError', { timeout: 120_000 });
  await expect(status).toContainText('source SHA-256');

  await answer.click();
  await page.keyboard.press('ControlOrMeta+A');
  await page.keyboard.insertText(reference);
  await page.getByRole('button', { name: /^Check (?!log)/ }).click();
  await expect(status).toContainText('PASSED · source SHA-256', { timeout: 120_000 });
});

test('0060 opens directly and runs bounded classifier training only on explicit action', async ({ page }) => {
  await page.goto('lab/?exercise=0060');
  await expect(page.locator('.jp-MarkdownCell h1').first()).toContainText('0060 · One honest classifier update');
  const panel = answerPane(page);
  const status = panel.getByRole('status');

  await panel.getByRole('button', { name: 'Hint', exact: true }).click();
  await expect(panel).toContainText('Reuse the stable 0059 log-sum-exp pattern');

  await runExplorePanel(page);
  const card = page.getByRole('group', { name: 'Tensor card for X_train' });
  await expect(card).toContainText('(6, 2)', { timeout: 60_000 });
  await expect(page.getByTestId('dojo-0060-gradient-probe')).toContainText('dW shape (2, 2)', { timeout: 60_000 });
  await expect(page.getByTestId('dojo-0060-run-count')).toContainText('Training runs: 0');

  const learningRate = page.getByRole('slider', { name: 'Learning rate' });
  await learningRate.fill('0.35');
  await expect(page.getByTestId('dojo-0060-run-count')).toContainText('Training runs: 0');
  await expect(page.getByTestId('dojo-0060-loss-summary')).toContainText('No training run yet');

  await page.getByTestId('dojo-0060-run-training').click();
  await expect(page.getByTestId('dojo-0060-run-count')).toContainText('Training runs: 1');
  await expect(page.getByTestId('dojo-0060-loss-summary')).toContainText('Loss 0.6931 ->', { timeout: 60_000 });
  await expect(page.getByTestId('dojo-0060-loss-curve')).toHaveAttribute('role', 'img');

  const answer = answerEditor(page);
  const check = panel.getByRole('button', { name: /^Check (?!log)/ });
  const stage = panel.getByLabel('Check stage');

  await check.click();
  await expect(status).toContainText('FAILED · source SHA-256', { timeout: 120_000 });
  await expect(status).toContainText('verify_baseline_one_step_and_preupdate_loss');

  await answer.click();
  await page.keyboard.press('ControlOrMeta+A');
  await page.keyboard.insertText(classifierReference);
  await stage.selectOption('production');
  await check.click();
  await expect(status).toContainText('PASSED · source SHA-256', { timeout: 120_000 });
  await expect(status).toContainText('✓ verify_production_weight_unequal_microbatches');
  await expect(status).toContainText('Receipt source SHA-256');
  await expect(status).toContainText('source cells [');

  const wrong = classifierReference.replace('probabilities /= count', 'probabilities /= len(X)');
  expect(wrong).not.toBe(classifierReference);
  await answer.click();
  await page.keyboard.press('ControlOrMeta+A');
  await page.keyboard.insertText(wrong);
  await check.click();
  await expect(status).toContainText('FAILED · source SHA-256', { timeout: 120_000 });
  await expect(status).toContainText('✗ verify_production_weight_unequal_microbatches');
});
