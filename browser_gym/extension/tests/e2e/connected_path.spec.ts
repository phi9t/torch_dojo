import { expect, test } from '@playwright/test';

test('learning path links 0059, 0060, MLP, and 0069 with shared concepts', async ({ page }) => {
  await page.goto('lab/?exercise=mlp');
  await expect(page.locator('.jp-MarkdownCell h1').first()).toContainText('MLP Modeling Lab');

  await page.getByRole('button', { name: 'Learning path', exact: true }).click();
  const path = page.getByTestId('dojo-learning-path');
  await expect(path).toContainText('0059 · Stable Softmax Cross-Entropy');
  await expect(path).toContainText('0060 · One honest classifier update');
  await expect(path).toContainText('mlp · MLP Modeling Lab');
  await expect(path).toContainText('0069 · Multiclass calibration metrics');
  await expect(path).toContainText('stability');
  await expect(path).toContainText('training');
  await expect(path).toContainText('confidence');
  await expect(path).toContainText('calibration');

  await expect(page.getByTestId('dojo-next-exercise')).toContainText('Next: 0069');
  await page.getByTestId('dojo-path-link-0069').click();
  await expect(page.locator('.jp-MarkdownCell h1').first()).toContainText(
    '0069 · Multiclass calibration metrics',
    { timeout: 60_000 }
  );
  await expect(page.getByTestId('dojo-next-exercise')).toContainText('End of path');
});
