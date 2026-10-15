import { expect, test, type Locator, type Page } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { answerEditor, runExplorePanel } from './bench';

const evalSource = readFileSync(resolve(__dirname, '../../../../dojo/0061_kmeans_eval.py'), 'utf8');
const assignMatch = evalSource.match(/def _assign[\s\S]*?(?=\n\n\ndef kmeans\()/)?.[0];
const kmeansMatch = evalSource.match(/def kmeans\([\s\S]*?(?=\n\n# %% \[verifiers\])/)?.[0];
if (!assignMatch || !kmeansMatch) throw new Error('Could not find canonical 0061 reference');
const reference = `import numpy as np\n\n${assignMatch}\n\n${kmeansMatch}\n`;

const starter = `import numpy as np\ndef kmeans(X: np.ndarray, initial_centers: np.ndarray, max_iter: int = 100,\n           *, block_size: int = 1024) -> tuple[np.ndarray, np.ndarray, float]:\n    pass\n`;

async function replaceAnswer(page: Page, source: string) {
  const cell = answerEditor(page);
  await cell.click();
  await page.keyboard.press('ControlOrMeta+A');
  await page.keyboard.insertText(source);
}

test('0061 K-means qualification: reference passes, starter fails, empty-cluster mutant fails, visual probe renders', async ({ page }) => {
  await page.goto('lab/?exercise=0061');

  const title = page.locator('.jp-MarkdownCell h1').first();
  await expect(title).toContainText('0061', { timeout: 30_000 });
  await expect(title).toContainText('K-Means');

  const status = page.locator('#torch-dojo-gym [role="status"]');
  const check = page.getByRole('button', { name: /^Check (?!log)/ });
  await expect(check).toBeVisible({ timeout: 30_000 });
  const stage = page.getByLabel('Check stage');
  await expect(stage).toHaveValue('baseline');

  // ---- Visual lesson probe ----
  await runExplorePanel(page);

  // Tensor card shape evidence for at least one of: points, initial_centers, labels
  const anyTensorCard = page.getByRole('group', { name: /Tensor card for/i }).first();
  await expect(anyTensorCard).toBeVisible({ timeout: 180_000 });
  await expect(anyTensorCard).toContainText('shape', { timeout: 180_000 });
  await expect(anyTensorCard).toContainText('dtype');

  // K-means section with cluster assignments + convergence plot
  const kmeansSection = page.locator('#dojo-0061-kmeans');
  await expect(kmeansSection).toBeVisible({ timeout: 60_000 });

  // Cluster assignment scatter plot
  const clusterPlot = page.locator('#dojo-0061-cluster-plot');
  await expect(clusterPlot).toBeVisible({ timeout: 60_000 });
  await expect(clusterPlot).toHaveAttribute('role', 'img');

  // Convergence plot (inertia per iteration)
  const convergence = page.locator('#dojo-0061-convergence-plot');
  await expect(convergence).toBeVisible({ timeout: 60_000 });
  await expect(convergence).toHaveAttribute('role', 'img');

  // Explicit controls: Run button for k-means probe, K select, and max-iter slider
  const runBtn = page.getByRole('button', { name: 'Run k-means probe' });
  await expect(runBtn).toBeVisible();
  const kSelect = page.getByLabel('Cluster count (K)');
  await expect(kSelect).toBeVisible();
  const maxIterSlider = page.getByLabel('Max iterations');
  await expect(maxIterSlider).toBeVisible();

  // Evidence region shows shape text after the probe runs
  const evidence = page.getByTestId('dojo-0061-evidence');
  await expect(evidence).toContainText(/shapes points\(/i, { timeout: 60_000 });

  // ---- Starter fails ----
  // First code cell is the probe; second is the solution
  await stage.selectOption('baseline');
  await replaceAnswer(page, starter);
  await check.click();
  await expect(status).toContainText('FAILED · source SHA-256', { timeout: 120_000 });

  // ---- Reference passes ----
  await replaceAnswer(page, reference);
  await stage.selectOption('production');
  await check.click();
  await expect(status).toContainText('PASSED · source SHA-256', { timeout: 180_000 });
  await expect(status).toContainText('verify_production_generated_batch_equivalence');

  // ---- Exploration isolation: scratch kernel value does not leak to next check ----
  await replaceAnswer(page, 'print(probe_points)\n' + reference);
  await check.click();
  await expect(status).toContainText('FAILED · source SHA-256', { timeout: 120_000 });
  await expect(status).toContainText('NameError');

  // ---- Plausible wrong answer (empty cluster collapse) ----
  // Divide every center by its count, so an empty cluster collapses to NaN instead of keeping its center.
  const emptyClusterMutant = reference.replace(
    'updated[occupied] = sums[occupied] / counts[occupied, None]',
    'updated = sums / counts[:, None]'
  );
  expect(emptyClusterMutant).not.toBe(reference);
  await replaceAnswer(page, emptyClusterMutant);
  await stage.selectOption('followup');
  await check.click();
  await expect(status).toContainText('FAILED · source SHA-256', { timeout: 120_000 });
  await expect(status).toContainText('empty_center');

  // ---- Cleanup: reference passes production again ----
  await replaceAnswer(page, reference);
  await stage.selectOption('production');
  await check.click();
  await expect(status).toContainText('PASSED · source SHA-256', { timeout: 180_000 });
});
