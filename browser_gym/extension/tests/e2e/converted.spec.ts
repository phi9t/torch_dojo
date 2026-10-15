import { expect, test } from '@playwright/test';
import { openTab, problemPane } from './bench';

// Every workout converted to brief markers opens with a typed brief and a drill that runs.
const CONVERTED = ['0060', '0061', '0069', '0073', '0074', '0062', '0063', '0064', '0065', '0066', '0070', '0068', '0067', '0071', '0076', '0077', '0035', '0072', '0078', '0079', '0082', '0083', '0091', '0030', '0050', '0075', '0100', '0094', '0080'];

for (const id of CONVERTED) {
  test(`${id} opens with a typed brief and a drill that runs`, async ({ page }) => {
    await page.goto(`lab/?exercise=${id}`);
    await expect(page.getByRole('button', { name: /^Check (?!log)/ })).toBeEnabled({ timeout: 60_000 });
    const brief = problemPane(page);
    await expect(brief.getByRole('heading', { name: 'Goal', exact: true })).toBeVisible();
    await expect(brief.getByRole('heading', { name: /^Baseline/ })).toBeVisible();

    await openTab(page, 'Drills');
    const drill = problemPane(page).getByRole('group').filter({ has: page.getByRole('button', { name: 'Reveal reasoning' }) }).first();
    await drill.getByRole('button', { name: 'Run', exact: true }).click();
    await expect(drill.locator('.jp-OutputArea')).not.toBeEmpty({ timeout: 60_000 });
    await drill.getByRole('button', { name: 'Reveal reasoning' }).click();
    await expect(problemPane(page).getByRole('button', { name: 'Hide reasoning' })).toBeVisible();
  });
}
