import { expect, test, type Page } from '@playwright/test';
import { replaceAnswer } from './bench';

async function checkBaseline(page: Page, answer: string) {
  await page.goto('lab/?exercise=0068');
  await expect(page.getByRole('button', { name: /^Check (?!log)/ })).toBeEnabled({ timeout: 60_000 });
  await replaceAnswer(page, answer);
  await page.getByRole('button', { name: 'Check baseline' }).click();
  await expect(page.locator('#torch-dojo-gym [role="status"]')).toContainText('FAILED · source SHA-256', { timeout: 120_000 });
  return page.getByRole('region', { name: 'Verifier detail' });
}

test('a failed assertion shows the check line, the failing example and the check’s variables', async ({ page }) => {
  // Summing durations double-counts overlaps.
  const detail = await checkBaseline(page, 'import numpy as np\n\ndef interval_union(intervals):\n    return float(sum(hi - lo for lo, hi in intervals))\n');
  await expect(detail).toContainText('AssertionError: the highlighted assertion was false');
  const where = detail.getByRole('region', { name: 'Where it failed' });
  await expect(where).toContainText('In the check verify_baseline_overlap_and_touching, line 3');
  await expect(where.getByLabel(/^Failing line 3: assert interval_union\(x\) == 5\.0$/)).toBeVisible();

  const example = detail.getByRole('region', { name: 'The failing example' });
  await expect(example).toContainText('[[0, 2], [1, 3], [3, 5]]');
  await expect(example).toContainText(/Expected\s*5\s*Your output\s*6/);
  const variables = detail.getByRole('region', { name: 'The check’s variables when it failed' });
  await expect(variables).toContainText('x');
  await expect(variables).toContainText('array([[0., 2.]');
});

test('an answer that raises shows its own line first, then the check that called it', async ({ page }) => {
  // Works until the input is empty: `ordered[0]` has nothing to unpack.
  const answer = [
    'import numpy as np',
    '',
    'def interval_union(intervals):',
    '    ordered = intervals[np.argsort(intervals[:, 0])]',
    '    start, end = ordered[0]',
    '    total = 0.0',
    '    for lo, hi in ordered[1:]:',
    '        if lo > end:',
    '            total, start, end = total + end - start, lo, hi',
    '        else:',
    '            end = max(end, hi)',
    '    return float(total + end - start)',
    ''
  ].join('\n');
  const detail = await checkBaseline(page, answer);
  await expect(detail).toContainText('IndexError');
  const where = detail.getByRole('region', { name: 'Where it failed' });
  const labels = await where.locator('figcaption').allInnerTexts();
  expect(labels).toEqual(['In your answer, line 5 (interval_union)', 'In the check verify_baseline_empty_and_nested, line 2']);
  await expect(where.getByLabel(/^Failing line 5: start, end = ordered\[0\]$/)).toBeVisible();
  await expect(detail.getByRole('region', { name: 'The failing example' })).toContainText('(0, 2)');
});
