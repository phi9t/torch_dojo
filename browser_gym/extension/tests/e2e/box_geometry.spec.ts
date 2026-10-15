import { expect, test, type Page } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { answerEditor, answerPane, runExplorePanel, scratchEditor } from './bench';

interface GeometryCase {
  id: '0062' | '0063' | '0064';
  title: string;
  evalFile: string;
  functionName: string;
  tensorCard: string;
  sliderName: string;
  outputTestId: string;
  runCountTestId: string;
  passingVerifier: string;
  scratchName: string;
}

const cases: GeometryCase[] = [
  {
    id: '0062',
    title: 'Pairwise Box IoU',
    evalFile: '0062_pairwise_iou_eval.py',
    functionName: 'pairwise_iou',
    tensorCard: 'Tensor card for iou',
    sliderName: 'IoU shift',
    outputTestId: 'dojo-iou-value',
    runCountTestId: 'dojo-0062-probe-runs',
    passingVerifier: 'verify_production_generated_box_pairs',
    scratchName: 'left'
  },
  {
    id: '0063',
    title: 'Uniform Oriented-Box Sampling',
    evalFile: '0063_oriented_box_sampling_eval.py',
    functionName: 'sample_oriented_box',
    tensorCard: 'Tensor card for samples',
    sliderName: 'Yaw angle',
    outputTestId: 'dojo-yaw-value',
    runCountTestId: 'dojo-0063-probe-runs',
    passingVerifier: 'verify_production_immutability_and_large_batch',
    scratchName: 'samples'
  },
  {
    id: '0064',
    title: 'Point in Rotated Box',
    evalFile: '0064_point_in_rotated_box_eval.py',
    functionName: 'point_in_rotated_box',
    tensorCard: 'Tensor card for local_recovered',
    sliderName: 'Point x',
    outputTestId: 'dojo-membership-value',
    runCountTestId: 'dojo-0064-probe-runs',
    passingVerifier: 'verify_production_large_grid_and_translated_membership',
    scratchName: 'world_points'
  }
];

function referenceSource(item: GeometryCase): string {
  const evalSource = readFileSync(resolve(__dirname, `../../../../dojo/${item.evalFile}`), 'utf8');
  const pattern = new RegExp(`def ${item.functionName}[\\s\\S]*?(?=\\n\\n# %% \\[verifiers\\])`);
  const match = evalSource.match(pattern);
  if (!match) throw new Error(`Could not find canonical reference function for ${item.id}`);
  return `import numpy as np\n${match[0]}`;
}

async function replaceSolution(page: Page, source: string) {
  const solution = answerEditor(page);
  await solution.click();
  await page.keyboard.press('ControlOrMeta+A');
  await page.keyboard.insertText(source);
}

for (const item of cases) {
  test(`${item.id} opens with visual probe and checks canonical reference`, async ({ page }) => {
    await page.goto(`lab/?exercise=${item.id}`);
    await expect(page.locator('.jp-MarkdownCell h1').first()).toContainText(`${item.id} · ${item.title}`);

    const panel = answerPane(page);
    const hint = panel.getByRole('button', { name: 'Hint', exact: true });
    await expect(hint).toBeEnabled();
    await hint.click();
    await expect(panel).toContainText('Assistance recorded: hint 1');

    await runExplorePanel(page);
    await expect(page.getByRole('group', { name: item.tensorCard })).toContainText('shape', { timeout: 60_000 });

    const probeOutput = page.getByTestId(item.outputTestId);
    await expect(probeOutput).not.toContainText('waiting', { timeout: 60_000 });
    const firstProbeText = await probeOutput.textContent();
    const slider = page.getByRole('slider', { name: item.sliderName });
    await slider.focus();
    await slider.press('ArrowRight');
    await expect(probeOutput).not.toHaveText(firstProbeText ?? '');
    await expect(page.getByTestId(item.runCountTestId)).toContainText('Probe runs: 2');

    await replaceSolution(page, referenceSource(item));
    await page.getByLabel('Check stage').selectOption('production');
    await panel.getByRole('button', { name: /^Check (?!log)/ }).click();
    const status = panel.locator('[role="status"]');
    await expect(status).toContainText('PASSED · source SHA-256', { timeout: 180_000 });
    await expect(status).toContainText(item.passingVerifier);
    await expect(status).toContainText('Receipt 1: PASSED');

    await replaceSolution(page, `print(${item.scratchName})\n${referenceSource(item)}`);
    await panel.getByRole('button', { name: /^Check (?!log)/ }).click();
    await expect(status).toContainText('NameError', { timeout: 120_000 });
    await expect(status).toContainText('Receipt 2: FAILED');
  });
}
