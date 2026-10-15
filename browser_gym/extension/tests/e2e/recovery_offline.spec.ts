import { expect, test } from '@playwright/test';
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { answerEditor, answerPane, openTab, runExplorePanel, scratchEditor } from './bench';

const evalSource = readFileSync(resolve(__dirname, '../../../../dojo/0059_stable_cross_entropy_eval.py'), 'utf8');
const referenceFunction = evalSource.match(/def softmax_cross_entropy[\s\S]*?(?=\n\n# %% \[verifiers\])/)?.[0];
if (!referenceFunction) throw new Error('Could not find canonical reference function');
const reference = `import numpy as np\n${referenceFunction}`;

async function replaceAnswer(page: import('@playwright/test').Page, source: string) {
  const answer = answerEditor(page);
  await answer.click();
  await page.keyboard.press('ControlOrMeta+A');
  await page.keyboard.insertText(source);
}

async function runProbeCell(page: import('@playwright/test').Page) {
  const probeCell = await runExplorePanel(page);
  const card = page.getByRole('group', { name: 'Tensor card for logits_probe' });
  await expect(card).toContainText('(2, 2)', { timeout: 90_000 });
  await expect(card).toContainText('float64');
  await expect(page.getByTestId('dojo-loss-value')).toContainText('Loss at scale 1.00:', { timeout: 90_000 });
  const firstLoss = await page.getByTestId('dojo-loss-value').textContent();
  const slider = page.getByRole('slider', { name: 'Logit scale' });
  await slider.press('ArrowRight');
  await expect(slider).toHaveValue('1.25');
  await expect(page.getByTestId('dojo-loss-value')).not.toHaveText(firstLoss ?? '');
  await expect(page.getByTestId('dojo-probe-runs')).toContainText('Probe runs: 2');
  await expect(page.locator('#dojo-loss-plot')).toHaveAttribute('role', 'img');
}

test('offline reload keeps 0059 probe, isolated check, and attempt archive usable', async ({ page }, testInfo) => {
  await page.goto('lab/?exercise=0059');
  await expect(page.locator('.jp-MarkdownCell h1').first()).toContainText('0059 · Stable Softmax Cross-Entropy');
  await page.evaluate(async () => {
    if (!navigator.serviceWorker) throw new Error('service worker is unavailable');
    await navigator.serviceWorker.ready;
  });

  await runProbeCell(page);
  await replaceAnswer(page, reference);
  const panel = page.locator('body');
  const status = page.locator('#torch-dojo-gym [role="status"]');
  await panel.getByRole('button', { name: /^Check (?!log)/ }).click();
  await expect(status).toContainText('PASSED · source SHA-256', { timeout: 180_000 });
  const firstStatus = await status.textContent();
  const firstHash = firstStatus?.match(/PASSED · source SHA-256 ([0-9a-f]{64})/)?.[1];
  expect(firstHash).toBeTruthy();

  // Give the service worker one controlled online navigation after the runtime
  // and checker assets have been fetched. The offline scope is intentionally
  // "previously-loaded site", not first-load-without-network.
  await page.reload({ waitUntil: 'networkidle' });
  await expect(page.locator('.jp-MarkdownCell h1').first()).toContainText('0059 · Stable Softmax Cross-Entropy', {
    timeout: 90_000
  });

  await page.context().setOffline(true);
  try {
    await page.reload({ waitUntil: 'domcontentloaded' });
    await expect(page.locator('.jp-MarkdownCell h1').first()).toContainText('0059 · Stable Softmax Cross-Entropy', {
      timeout: 90_000
    });

    await runProbeCell(page);
    await replaceAnswer(page, reference);
    await panel.getByRole('button', { name: /^Check (?!log)/ }).click();
    await expect(status).toContainText('PASSED · source SHA-256', { timeout: 180_000 });
    await expect(status).toContainText(`Receipt source SHA-256 ${firstHash}`);

    const staleSource = `${reference}\n# offline edit after check`;
    await replaceAnswer(page, staleSource);
    await expect(status).toContainText('STALE PASS');

    await openTab(page, 'Reps');
    const download = await Promise.all([
      page.waitForEvent('download'),
      panel.getByRole('button', { name: 'Export all' }).click()
    ]).then(([item]) => item);
    const archivePath = await download.path();
    expect(archivePath).toBeTruthy();
    const archive = JSON.parse(readFileSync(archivePath!, 'utf8'));
    const checkedHashes = archive.attempts.flatMap((attempt: { receipts: Array<{ sha256: string }> }) =>
      attempt.receipts.map(receipt => receipt.sha256)
    );
    expect(checkedHashes).toContain(firstHash);

    const importPath = testInfo.outputPath('offline-import.json');
    writeFileSync(importPath, JSON.stringify(archive));
    await page.evaluate(() => localStorage.clear());
    await panel.locator('input[type="file"]').setInputFiles(importPath);
    await expect(status).toContainText('Imported reps');
    await expect(status).toContainText(firstHash!);
  } finally {
    await page.context().setOffline(false);
  }
});

test('hard restart after runaway exploration cell leaves a usable fresh kernel', async ({ page }) => {
  await page.goto('lab/?exercise=0059');
  await expect(page.locator('.jp-MarkdownCell h1').first()).toContainText('0059 · Stable Softmax Cross-Entropy');

  const scratch = await scratchEditor(page);
  await scratch.click();
  await page.keyboard.press('ControlOrMeta+A');
  await page.keyboard.insertText('import time\nwhile True:\n    time.sleep(60)');
  await page.getByRole('button', { name: 'Run cell' }).click();

  const panel = page.locator('body');
  const status = page.locator('#torch-dojo-gym [role="status"]');
  page.once('dialog', dialog => {
    expect(dialog.message()).toMatch(/Live kernel state/);
    void dialog.accept();
  });
  await panel.getByRole('button', { name: 'Restart kernel', exact: true }).click();
  await expect(status).toContainText('Notebook kernel restarted', { timeout: 120_000 });

  await runProbeCell(page);
  await replaceAnswer(page, reference);
  await panel.getByRole('button', { name: /^Check (?!log)/ }).click();
  await expect(status).toContainText('PASSED · source SHA-256', { timeout: 180_000 });
});
