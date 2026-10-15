import { expect, test } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { problemPane, replaceAnswer } from './bench';

const evalSource = readFileSync(resolve(__dirname, '../../../../dojo/0059_stable_cross_entropy_eval.py'), 'utf8');
const reference = 'import numpy as np\n' + evalSource.match(/def softmax_cross_entropy[\s\S]*?(?=\n\n# %% \[verifiers\])/)![0];

for (const colorScheme of ['light', 'dark'] as const) {
  test.describe(`${colorScheme} theme`, () => {
    test.use({ colorScheme });

    test('the brief shows typed sections and typeset math', async ({ page }) => {
      await page.goto('lab/?exercise=0059');
      const brief = problemPane(page);
      await expect(brief.getByRole('heading', { name: 'Goal', exact: true })).toBeVisible({ timeout: 60_000 });
      // The workout title shows once, above the tabs; the brief's own copy is for screen readers.
      const repeated = await brief.getByRole('heading', { level: 1 }).filter({ hasText: '0059 ·' }).boundingBox();
      expect(repeated!.width).toBeLessThanOrEqual(1);
      // The goal comes first, before any stage.
      const headings = await brief.getByRole('heading').allTextContents();
      const goal = headings.findIndex(text => text.startsWith('Goal'));
      expect(goal).toBeGreaterThanOrEqual(0);
      expect(goal).toBeLessThan(headings.findIndex(text => text.startsWith('Baseline')));
      await expect(brief.getByRole('heading', { name: 'Predict', exact: true }).first()).toBeVisible();
      await expect(brief.getByRole('heading', { name: 'Watch out', exact: true })).toContainText('Watch out');
      const why = brief.locator('summary', { hasText: 'Why' });
      await expect(why).toBeVisible();
      await why.click();
      await expect(brief.getByText('Checking only the scalar mean cannot reveal this mistake.')).toBeVisible();
      // Math is typeset, inline and display, rather than left as dollar signs.
      await expect(brief.locator('mjx-container').first()).toBeVisible({ timeout: 30_000 });
      expect(await brief.locator('mjx-container[display="true"]').count()).toBeGreaterThan(0);
      await expect(brief.getByText('$q = z')).toHaveCount(0);
      // Code identifiers in prose use the code face.
      await expect(brief.locator('code', { hasText: 'ignore_index' }).first()).toBeVisible();
    });
  });
}

test('the brief marks the stage being checked, passed stages, and links to help', async ({ page }) => {
  await page.goto('lab/?exercise=0059');
  const brief = problemPane(page);
  await expect(brief.getByRole('heading', { name: /^Baseline/ })).toContainText('checking now', { timeout: 60_000 });
  await expect(brief.getByRole('heading', { name: /^Follow-up/ })).not.toContainText('checking now');
  // The rep loads into the notebook before Check is enabled; typing earlier can be overwritten.
  await expect(page.getByRole('button', { name: 'Check baseline' })).toBeEnabled({ timeout: 60_000 });

  await replaceAnswer(page, reference);
  await page.getByRole('button', { name: 'Check baseline' }).click();
  await expect(page.locator('#torch-dojo-gym [role="status"]')).toContainText('PASSED · source SHA-256', { timeout: 120_000 });
  await expect(brief.getByRole('heading', { name: /^Baseline/ })).toContainText('passed');
  await page.getByLabel('Check stage').selectOption('followup');
  await expect(brief.getByRole('heading', { name: /^Follow-up/ })).toContainText('checking now');

  await brief.getByRole('button', { name: 'Show a hint' }).click();
  await expect(page.getByText('Assistance recorded: hint 1').first()).toBeAttached();
  await expect(page.locator('#torch-dojo-gym ol li').first()).toBeVisible();
  await brief.getByRole('button', { name: 'See the reference solution' }).click();
  await expect(page.locator('#torch-dojo-gym pre').filter({ hasText: 'def softmax_cross_entropy' })).toBeVisible();
});
