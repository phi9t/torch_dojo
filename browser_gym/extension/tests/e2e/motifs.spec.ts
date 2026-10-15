import { expect, test } from '@playwright/test';

test('recall, rate, reload, browse and merge a portable archive', async ({ page }) => {
  await page.goto('lab/');
  const panel = page.getByRole('region', { name: 'Motif library and review' });
  await expect(panel).toBeVisible({ timeout: 60_000 });
  await expect(panel).toContainText('16 new');
  await expect(panel.getByLabel('Import motif review archive')).toBeHidden();
  await panel.getByRole('button', { name: 'Review next motif', exact: true }).click();
  await expect(panel.getByRole('heading', { name: 'Shift before you exponentiate' })).toBeVisible();
  await expect(panel.getByRole('button', { name: 'Comfortable', exact: true })).toBeHidden();
  await panel.getByRole('button', { name: 'Reveal explanation' }).click();
  await expect(panel).toContainText('Subtract 1001');
  await expect(panel.getByRole('link', { name: /0059/ })).toHaveAttribute('href', /\?exercise=0059$/);
  await panel.getByRole('button', { name: 'Comfortable', exact: true }).click();
  await expect(panel.getByRole('status')).toHaveText('Saved. This motif returns in 3 days.');
  await page.reload();
  await expect(panel).toContainText('15 new', { timeout: 60_000 });
  await panel.getByRole('button', { name: 'Review next motif', exact: true }).click();
  await expect(panel.getByRole('heading', { name: 'Collisions are contributions' })).toBeVisible();
  await panel.getByRole('button', { name: 'Reveal explanation' }).click();
  await panel.getByRole('button', { name: 'Again', exact: true }).click();
  await expect(panel.getByRole('status')).toContainText('10 minutes');
  await panel.getByText('Browse all 16 motifs', { exact: true }).click();
  await panel.getByRole('searchbox', { name: 'Find a motif' }).fill('0117');
  await expect(panel.getByRole('button', { name: 'Compact, remember, restore' })).toBeVisible();
  await expect(panel.getByRole('button', { name: 'Shift before you exponentiate', exact: true })).toHaveCount(0);
  const archive = { version: 1, cards: { 'scatter-collisions': { due: 0, reviewed: 0, interval: 600, rating: 'again' } } };
  await panel.getByLabel('Import motif review archive').setInputFiles({ name: 'review.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(archive)) });
  await expect(panel.getByRole('status')).toContainText('Reviews merged');
  // Older imported rating cannot undo the review just saved.
  await expect(panel).toContainText('0 due · 14 new');
  const download = page.waitForEvent('download');
  await panel.getByRole('button', { name: 'Export motif reviews' }).click();
  expect((await download).suggestedFilename()).toBe('motif-reviews.json');
});

test('failed saves keep the revealed card available to retry', async ({ page }) => {
  await page.goto('lab/');
  const panel = page.getByRole('region', { name: 'Motif library and review' });
  await expect(panel).toBeVisible({ timeout: 60_000 });
  await panel.getByRole('button', { name: 'Review next motif', exact: true }).click();
  await panel.getByRole('button', { name: 'Reveal explanation' }).click();
  await page.evaluate(() => {
    const original = Storage.prototype.setItem;
    Storage.prototype.setItem = function (key, value) {
      if (key === 'torch-dojo.motif-review.v1') throw new DOMException('Storage is full', 'QuotaExceededError');
      return original.call(this, key, value);
    };
  });
  await panel.getByRole('button', { name: 'Comfortable', exact: true }).click();
  await expect(panel.getByRole('status')).toContainText('Storage is full');
  await expect(panel.getByRole('heading', { name: 'Shift before you exponentiate' })).toBeVisible();
  await expect(panel).toContainText('16 new');
});

test('previously loaded motif library remains usable offline', async ({ page }) => {
  await page.goto('lab/');
  const panel = page.getByRole('region', { name: 'Motif library and review' });
  await expect(panel).toBeVisible({ timeout: 60_000 });
  await page.evaluate(async () => { await navigator.serviceWorker.ready; });
  await page.reload({ waitUntil: 'networkidle' });
  await expect(panel).toBeVisible();
  await page.context().setOffline(true);
  try {
    await page.reload({ waitUntil: 'domcontentloaded' });
    await expect(panel).toBeVisible({ timeout: 60_000 });
    await panel.getByRole('button', { name: 'Review next motif', exact: true }).click();
    await panel.getByRole('button', { name: 'Reveal explanation' }).click();
    await panel.getByRole('button', { name: 'With effort', exact: true }).click();
    await expect(panel.getByRole('status')).toContainText('Saved. This motif returns in 1 day.');
  } finally {
    await page.context().setOffline(false);
  }
});
