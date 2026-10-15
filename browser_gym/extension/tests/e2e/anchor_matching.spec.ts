import { expect, test } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { answerEditor, answerPane, runExplorePanel, scratchEditor } from './bench';

const evalSource = readFileSync(resolve(__dirname, '../../../../dojo/0070_anchor_match_eval.py'), 'utf8');
const referenceFunction = evalSource.match(/def match_anchors[\s\S]*?(?=\n\n# %% \[verifiers\])/)?.[0];
if (!referenceFunction) throw new Error('Could not find canonical reference function');
const reference = `import numpy as np\n${referenceFunction}`;

test('0070 visualizes threshold effects and qualifies canonical matching', async ({ page }) => {
  await page.goto('lab/?exercise=0070');
  await expect(page.locator('.jp-MarkdownCell h1').first()).toContainText('0070 · Anchor to GT Matching');
  await expect(page.getByTestId('dojo-kernel-status')).toHaveText('Python · idle', { timeout: 60_000 });
  const panel = answerPane(page);
  const status = panel.getByRole('status');
  const check = panel.getByRole('button', { name: /^Check (?!log)/ });

  const answer = answerEditor(page);
  const probeCell = await runExplorePanel(page);

  const anchorCard = page.getByRole('group', { name: 'Tensor card for anchor_iou' });
  await expect(anchorCard).toContainText('shape', { timeout: 60_000 });
  await expect(anchorCard).toContainText('(5, 2)');
  await expect(anchorCard).toContainText('float64');
  await expect(anchorCard).toContainText('anchor x ground truth');

  await expect(page.getByRole('group', { name: 'Tensor card for collision_iou' })).toContainText('(2, 2)');

  const labels = page.getByTestId('dojo-anchor-labels');
  await expect(labels).toContainText('Labels at positive threshold 0.5: [-1, -2, 0, 1, 0]', { timeout: 60_000 });
  await expect(page.getByTestId('dojo-anchor-shape')).toContainText('assignment shape: (5,), dtype: integer labels');
  await expect(page.getByTestId('dojo-anchor-collision')).toContainText('Collision evidence');
  await expect(page.getByTestId('dojo-anchor-collision')).toContainText('labels [0, -1]');

  const slider = page.getByRole('slider', { name: 'Anchor matching threshold' });
  await slider.focus();
  await slider.press('ArrowRight');
  await expect(labels).toContainText('Labels at positive threshold 0.6: [-1, -2, 0, 1, -2]');
  await expect(page.locator('#dojo-anchor-matching-plot')).toHaveAttribute('role', 'img');
  await page.locator('#dojo-anchor-matching-plot').focus();
  await expect(page.locator('#dojo-anchor-matching-plot')).toBeFocused();

  const answerBeforeReveal = await answer.textContent();
  await panel.getByRole('button', { name: 'Hint', exact: true }).click();
  await expect(panel).toContainText("Start from each anchor row's best GT");
  await panel.getByRole('button', { name: 'Collapse hints' }).click();
  await expect(status).toContainText('Assistance recorded: hint 1');
  await panel.getByRole('button', { name: 'Reveal solution' }).click();
  await expect(panel).toContainText('def match_anchors');
  await expect(answer).toHaveText(answerBeforeReveal ?? '');
  await panel.getByRole('button', { name: 'Hide solution' }).click();

  const stage = panel.getByLabel('Check stage');
  await stage.selectOption('production');
  await check.click();
  await expect(status).toContainText('FAILED · source SHA-256', { timeout: 120_000 });
  await expect(status).toContainText('verify_baseline_thresholds');
  await expect(status).toContainText('Receipt 1');
  // Only the answer cell is submitted, wherever drills put it in the notebook.
  await expect(status).toContainText(/source cells \[\d+\]/);

  async function replaceAnswer(source: string) {
    await answer.click();
    await page.keyboard.press('ControlOrMeta+A');
    await page.keyboard.insertText(source);
  }

  const scratch = await scratchEditor(page);
  await scratch.click();
  await page.keyboard.press('ControlOrMeta+A');
  await page.keyboard.insertText('scratch_labels = [-999]');
  await page.getByRole('button', { name: 'Run cell' }).click();
  await replaceAnswer(`print(scratch_labels)\n${reference}`);
  await check.click();
  await expect(status).toContainText('NameError', { timeout: 120_000 });

  await replaceAnswer(reference);
  await check.click();
  await expect(status).toContainText('PASSED · source SHA-256', { timeout: 120_000 });
  await expect(status).toContainText('✓ verify_followup_forced_low_iou_and_collision');
  await expect(status).toContainText('✓ verify_production_validation_and_no_mutation');

  const collisionMutant = reference.replace(
    `if anchor not in claimed:
            result[anchor] = gt
            claimed.add(anchor)`,
    'result[anchor] = gt'
  );
  expect(collisionMutant).not.toBe(reference);
  await replaceAnswer(collisionMutant);
  await check.click();
  await expect(status).toContainText('FAILED · source SHA-256', { timeout: 120_000 });
  await expect(status).toContainText('✗ verify_followup_forced_low_iou_and_collision');
});
