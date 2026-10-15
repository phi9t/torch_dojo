import { expect, test } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { answerEditor, answerPane, runExplorePanel, scratchEditor } from './bench';
const evaluation = readFileSync(resolve(__dirname, '../../../../dojo/0059_stable_cross_entropy_eval.py'), 'utf8');
const reference = 'import numpy as np\n' + evaluation.match(/def softmax_cross_entropy[\s\S]*?(?=\n\n# %% \[verifiers\])/)![0];

test('real passing and failing examples load as independent NumPy scratch variables', async ({ page }) => {
  await page.goto('lab/?exercise=0059');
  // Wait for the rep to load, or it replaces what is typed.
  await expect(page.getByRole('button', { name: /^Check (?!log)/ })).toBeEnabled({ timeout: 60_000 });
  const answer = answerEditor(page);
  await answer.click();
  await page.keyboard.press('ControlOrMeta+A');
  await page.keyboard.insertText(reference.replace('gradient[valid] = probabilities / count', 'gradient[valid] = probabilities / (count * count)'));
  await page.getByRole('button', { name: 'Run cell', exact: true }).click();
  await page.getByRole('button', { name: /^Check (?!log)/ }).click();
  const results = page.getByRole('region', { name: 'Check results' });
  await expect(results).toContainText('1 passed', { timeout: 120_000 });
  await expect(results).toContainText('2 failed');
  // The first failing square is selected, and its detail shows the captured case.
  const firstFailure = results.getByRole('button', { name: 'baseline mean reduction: failed' });
  await expect(firstFailure).toHaveAttribute('aria-pressed', 'true');
  const detail = results.getByRole('region', { name: 'Verifier detail' });
  await expect(detail).toContainText('baseline mean reduction');
  // The failing example comes first; the check's earlier calls fold away below it.
  await expect(detail.getByRole('region', { name: 'The failing example' })).toContainText('Example 2');
  await detail.getByText(/^Earlier calls in this check/).click();
  await expect(detail).toContainText('shape (3, 3)');
  const answerBefore = await answer.textContent();
  // Loading cases puts the failing example first in line.
  await detail.getByRole('button', { name: 'Load case into scratch' }).click();
  await expect(detail).toContainText('Ready: gym_case(2)', { timeout: 60_000 });
  // Loading a case opens the scratch area with a runnable reproduction and leaves the answer alone.
  await expect(page.getByRole('button', { name: 'Hide scratch cell' })).toBeVisible();
  await expect(answerPane(page).locator('.jp-CodeCell').filter({ visible: true }).last()).toContainText('case = gym_case(2)');
  expect(await answer.textContent()).toBe(answerBefore);
  await results.getByRole('button', { name: 'baseline equal logits and single example: passed' }).click();
  await expect(detail).toContainText('Example 0');
  await scratchEditor(page);
  const scratch = answerPane(page).locator('.jp-CodeCell').filter({ visible: true }).last();
  await scratch.locator('.cm-content').click();
  await page.keyboard.press('ControlOrMeta+A');
  await page.keyboard.insertText(`assert gym_passed and gym_failed
assert gym_case(0)['args'][0].shape == (1, 2)
assert gym_case(0)['args'][1].dtype.kind in 'iu'
copy = gym_case(0)
copy['args'][0][:] = 123
assert (gym_case(0)['args'][0] == 0).all()
assert np.isclose(gym_try(0)[0], np.log(2))
assert gym_case(0)['expected'][1].shape == (1, 2)
print('SCRATCH_CASES_OK')`);
  await page.getByRole('button', { name: 'Run cell', exact: true }).click();
  await expect(scratch).toContainText('SCRATCH_CASES_OK');
  await expect(scratch.locator('.jp-OutputArea')).toContainText('SCRATCH_CASES_OK', { timeout: 60_000 });
  await answer.click();
  await page.keyboard.press('ControlOrMeta+End');
  await page.keyboard.insertText('\n# edited after this check');
  await expect(results).toContainText('Code changed since this check');
  await page.reload();
  await expect(results).toContainText('2 failed', { timeout: 60_000 });
  await expect(results).toContainText('Code changed since this check');
  await results.getByText(/^Earlier calls in this check/).click();
  await expect(results).toContainText('shape (3, 3)');
});
