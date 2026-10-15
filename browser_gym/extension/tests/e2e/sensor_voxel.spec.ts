import { expect, test, type Page } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { answerEditor, answerPane, runExplorePanel, scratchEditor } from './bench';

const CELL_OUTPUT_TIMEOUT = 120_000;

test.setTimeout(240_000);

function referenceSource(evalFile: string, functionName: string): string {
  const evalSource = readFileSync(resolve(__dirname, '../../../../dojo', evalFile), 'utf8');
  const pattern = new RegExp(`def ${functionName}[\\s\\S]*?(?=\\n\\n# %% \\[verifiers\\])`);
  const referenceFunction = evalSource.match(pattern)?.[0];
  if (!referenceFunction) throw new Error(`Could not find canonical reference function ${functionName}`);
  return `import numpy as np\n${referenceFunction}`;
}

async function answerText(page: Page): Promise<string> {
  return await answerEditor(page).textContent() ?? '';
}

async function replaceAnswer(page: Page, source: string): Promise<void> {
  const answer = answerEditor(page);
  await answer.click();
  await page.keyboard.press('ControlOrMeta+A');
  await page.keyboard.insertText(source);
}

async function setRangeValue(page: Page, label: string, value: string): Promise<void> {
  await page.getByRole('slider', { name: label }).evaluate((element, nextValue) => {
    const input = element as HTMLInputElement;
    input.value = nextValue as string;
    input.dispatchEvent(new Event('input', { bubbles: true }));
  }, value);
}

test('0065 temporal transform probe updates coordinates without changing submitted source', async ({ page }) => {
  await page.goto('lab/?exercise=0065');
  await expect(page.locator('.jp-MarkdownCell h1').first()).toContainText('0065 · Temporal Ego Coordinates');
  await expect(page.getByTestId('dojo-kernel-status')).toHaveText('Python · idle', { timeout: 60_000 });

  const answerBefore = await answerText(page);
  await runExplorePanel(page);

  const trajectoryCard = page.getByRole('group', { name: 'Tensor card for trajectories' });
  await expect(trajectoryCard).toContainText('shape', { timeout: CELL_OUTPUT_TIMEOUT });
  await expect(trajectoryCard).toContainText('(1, 1, 1, 3)');
  await expect(trajectoryCard).toContainText('float64');
  await expect(trajectoryCard).toContainText('B x A x T x xyz');

  const coords = page.getByTestId('dojo-0065-coords');
  await expect(coords).toContainText('world = (5.00, 1.00, 0.00)');
  await expect(coords).toContainText('current = (3.00, 1.00, 0.00)');
  const worldPoint = page.locator('#dojo-0065-frame-plot [data-role="world-point"]');
  const firstCx = await worldPoint.getAttribute('cx');

  const worldTranslation = page.getByRole('slider', { name: 'World translation x' });
  await setRangeValue(page, 'World translation x', '5.5');
  await expect(worldTranslation).toHaveValue('5.5');
  await expect(coords).toContainText('world = (5.50, 1.00, 0.00)');
  await expect(coords).toContainText('current = (3.50, 1.00, 0.00)');
  await expect(worldPoint).not.toHaveAttribute('cx', firstCx ?? '');
  await expect(page.locator('#dojo-0065-frame-plot')).toHaveAttribute('role', 'img');
  expect(await answerText(page)).toBe(answerBefore);

  await page.getByLabel('Check stage').selectOption('production');
  await replaceAnswer(page, referenceSource('0065_temporal_transforms_eval.py', 'to_current_ego'));
  await page.getByRole('button', { name: /^Check (?!log)/ }).click();
  const status = page.locator('#torch-dojo-gym [role="status"]');
  await expect(status).toContainText('PASSED · source SHA-256', { timeout: 180_000 });
  await expect(status).toContainText('✓ verify_production_invalid_inputs');
});

test('0066 voxel probe updates occupied cells without changing submitted source', async ({ page }) => {
  await page.goto('lab/?exercise=0066');
  await expect(page.locator('.jp-MarkdownCell h1').first()).toContainText('0066 · Sparse Voxel Pooling');
  await expect(page.getByTestId('dojo-kernel-status')).toHaveText('Python · idle', { timeout: 60_000 });

  const answerBefore = await answerText(page);
  await runExplorePanel(page);

  const pointsCard = page.getByRole('group', { name: 'Tensor card for points' });
  await expect(pointsCard).toContainText('shape', { timeout: CELL_OUTPUT_TIMEOUT });
  await expect(pointsCard).toContainText('(3, 3)');
  await expect(pointsCard).toContainText('float64');
  await expect(pointsCard).toContainText('N x xyz');

  const stats = page.getByTestId('dojo-0066-stats');
  await expect(stats).toContainText('occupied=2 counts=[1,2]');
  const occupiedRects = page.locator('#dojo-0066-grid-plot [data-role="cells"] rect');
  await expect(occupiedRects).toHaveCount(2);
  const firstRectY = Number(await occupiedRects.first().getAttribute('y'));
  expect(firstRectY).toBeLessThan(220);

  const voxelSize = page.getByRole('slider', { name: 'Voxel size' });
  await setRangeValue(page, 'Voxel size', '1.1');
  await expect(voxelSize).toHaveValue('1.1');
  await expect(page.getByTestId('dojo-0066-vs-out')).toContainText('voxel_size = 1.10');
  await expect(stats).toContainText('occupied=1 counts=[3]');
  await expect(occupiedRects).toHaveCount(1);
  await expect(page.locator('#dojo-0066-grid-plot')).toHaveAttribute('role', 'img');
  expect(await answerText(page)).toBe(answerBefore);

  await page.getByLabel('Check stage').selectOption('production');
  await replaceAnswer(page, referenceSource('0066_voxel_pool_eval.py', 'voxel_pool'));
  await page.getByRole('button', { name: /^Check (?!log)/ }).click();
  const status = page.locator('#torch-dojo-gym [role="status"]');
  await expect(status).toContainText('PASSED · source SHA-256', { timeout: 180_000 });
  await expect(status).toContainText('✓ verify_production_generated_count_conservation');
});
