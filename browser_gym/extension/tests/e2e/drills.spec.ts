import { expect, test, type Page } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { openTab, problemPane } from './bench';

async function openDrills(page: Page) {
  await page.goto('lab/?exercise=0059');
  await expect(page.getByRole('button', { name: /^Check (?!log)/ })).toBeEnabled({ timeout: 60_000 });
  await openTab(page, 'Drills');
}

function drill(page: Page, title: string) {
  return problemPane(page).getByRole('group', { name: title });
}

/** Waits until a prediction is saved with the rep, rather than sleeping past the save delay. */
async function expectSavedPrediction(page: Page, text: string) {
  await expect.poll(() => page.evaluate(() => window.localStorage.getItem('torch-dojo.browser-gym.attempts.v1') ?? ''))
    .toContain(text);
}

test('a drill card: predict, run the probe, compare, then reveal the reasoning', async ({ page }) => {
  await openDrills(page);
  await expect(page.getByRole('tab', { name: /^Drills/ })).toContainText('3');
  const card = drill(page, 'Paired or grid?');
  await expect(card.getByRole('heading', { name: 'Paired or grid?' })).toBeVisible();
  await expect(card).toContainText('Predict both printed arrays');
  // The probe code is on the card.
  await expect(card.locator('.cm-content')).toContainText('drill_logits[np.arange(2), drill_targets]');

  const prediction = card.getByLabel('Your prediction');
  await prediction.fill('[12, 20] then a 2x2 grid');
  // Shortcuts don't fire while typing a prediction.
  await prediction.press('ControlOrMeta+Enter');
  await prediction.press('Shift+Enter');
  await page.waitForTimeout(500);
  await expect(page.getByRole('region', { name: 'Check results' })).toHaveText('Check your answer to see each verifier here.');

  await card.getByRole('button', { name: 'Run', exact: true }).click();
  await expect(card.locator('.jp-OutputArea')).toContainText('[12. 20.]', { timeout: 60_000 });
  await expect(card.locator('.jp-OutputArea')).toContainText('[22. 20.]]');
  await expect(prediction).toHaveValue('[12, 20] then a 2x2 grid');

  const reveal = card.getByRole('button', { name: 'Reveal reasoning' });
  await expect(card.getByText('Only the paired form gives each example its own logit.')).toBeHidden();
  await reveal.click();
  await expect(card.getByText(/Only the paired form gives each example its own logit\./)).toBeVisible();
  await expect(card.locator('mjx-container').first()).toBeVisible();

  // Drills are never checked: the receipt log stays empty of drill code.
  await expect(page.locator('#torch-dojo-gym [role="status"]')).not.toContainText('drill_logits');
});

test('reasoning can be revealed before running, and predictions persist with the rep', async ({ page }) => {
  await openDrills(page);
  const card = drill(page, 'Which rows count in the mean?');
  await card.getByRole('button', { name: 'Reveal reasoning' }).click();
  await expect(card).toContainText('Averaging over the 2 valid rows');
  await expect(card.locator('.jp-OutputArea')).not.toContainText('1.25');
  await card.getByLabel('Your prediction').fill('0.83 and 1.25');
  await expectSavedPrediction(page, '0.83 and 1.25');

  await page.reload();
  await expect(page.getByRole('button', { name: /^Check (?!log)/ })).toBeEnabled({ timeout: 60_000 });
  // The Drills tab is remembered, and so is the prediction.
  await expect(page.getByRole('tab', { name: /^Drills/ })).toHaveAttribute('aria-selected', 'true');
  await expect(drill(page, 'Which rows count in the mean?').getByLabel('Your prediction')).toHaveValue('0.83 and 1.25');

  await openTab(page, 'Reps');
  const archive = await Promise.all([
    page.waitForEvent('download'),
    page.getByRole('button', { name: 'Export rep' }).click()
  ]).then(async ([download]) => JSON.parse(readFileSync((await download.path())!, 'utf8')));
  expect(archive.attempts[0].predictions).toEqual({ 2: '0.83 and 1.25' });

  // A new rep starts with empty predictions; the old rep keeps its own.
  await page.getByRole('button', { name: 'New rep' }).click();
  await openTab(page, 'Drills');
  await expect(drill(page, 'Which rows count in the mean?').getByLabel('Your prediction')).toHaveValue('');
});
