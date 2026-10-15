import { expect, test } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { answerEditor, runExplorePanel, scratchEditor } from './bench';

const evalSource = readFileSync(resolve(__dirname, '../../../../dojo/0059_stable_cross_entropy_eval.py'), 'utf8');
const referenceFunction = evalSource.match(/def softmax_cross_entropy[\s\S]*?(?=\n\n# %% \[verifiers\])/)?.[0];
if (!referenceFunction) throw new Error('Could not find canonical reference function');
const reference = `import numpy as np\n${referenceFunction}`;

for (const colorScheme of ['light', 'dark'] as const) {
  test.describe(`${colorScheme} theme`, () => {
    test.use({ colorScheme });

    test('learner runs cells and checks only the marked answer in a fresh worker', async ({ page }) => {
      await page.goto('lab/?exercise=0059');
      await expect(page.locator('.jp-MarkdownCell h1').first()).toContainText('0059 · Stable Softmax Cross-Entropy');
      const ground = await page.evaluate(() => getComputedStyle(document.body).backgroundColor);
      const [r, g, b] = ground.match(/\d+/g)!.map(Number);
      expect(r + g + b > 382, `${colorScheme} page ground ${ground}`).toBe(colorScheme === 'light');
      const status = page.locator('#torch-dojo-gym [role="status"]');
      const check = page.getByRole('button', { name: /^Check (?!log)/ });
      await expect(check).toBeVisible();
      const stage = page.getByLabel('Check stage');
      await expect(stage).toHaveValue('baseline');
      await runExplorePanel(page);
      await expect(page.getByText('Naive exponentials:').first()).toBeVisible({ timeout: 60_000 });

      await check.click();
      await expect(status).toContainText('FAILED · source SHA-256', { timeout: 120_000 });
      await expect(status).toContainText('verify_baseline_equal_logits_and_single_example');
      await expect(status).toContainText('ms');
      await expect(status).toContainText('checker.py');

      const answer = answerEditor(page);
      async function replaceAnswer(source: string) {
        await answer.click();
        await page.keyboard.press('ControlOrMeta+A');
        await page.keyboard.insertText(source);
      }

      await replaceAnswer(reference);
      await check.click();
      await expect(status).toContainText('PASSED · source SHA-256', { timeout: 120_000 });
      await expect(status).toContainText('✓ verify_baseline_mean_reduction');
      const passedText = await status.textContent();
      const passedHash = passedText?.match(/source SHA-256 ([0-9a-f]{64})/)?.[1];
      expect(passedHash).toBeTruthy();
      await expect(status).toContainText('Receipt 1');
      await expect(status).toContainText('Receipt source SHA-256');
      await expect(status).not.toContainText('verify_production_generated_shift_invariant');

      await replaceAnswer(`${reference}\n# edited after the passing check`);
      const editedHash = await page.evaluate(async source => {
        const bytes = new TextEncoder().encode(source as string);
        const digest = await crypto.subtle.digest('SHA-256', bytes);
        return Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('');
      }, `${reference}\n# edited after the passing check`);
      await expect(status).toContainText('STALE PASS');
      await expect(status).toContainText(`Receipt source SHA-256 ${passedHash}`);
      await expect(status).not.toContainText(`source SHA-256 ${editedHash}`);
      await replaceAnswer(reference);

      await stage.selectOption('production');
      await check.click();
      await expect(status).toContainText('✓ verify_production_generated_shift_invariant', { timeout: 120_000 });
      await expect(status).toContainText('Receipt 2');

      const unstable = reference.replace('q = selected - selected.max(axis=1, keepdims=True)', 'q = selected');
      expect(unstable).not.toBe(reference);
      await replaceAnswer(unstable);
      await check.click();
      await expect(status).toContainText('FAILED · source SHA-256', { timeout: 120_000 });
      await expect(status).toContainText('✗ verify_production_extreme_logits_and_row_shifts');

      const mutant = reference.replace('gradient[valid] = probabilities / count',
        'gradient[valid] = probabilities / (count * count)');
      expect(mutant).not.toBe(reference);
      await replaceAnswer(mutant);
      await check.click();
      await expect(status).toContainText('FAILED · source SHA-256', { timeout: 120_000 });
      await expect(status).toContainText('✗ verify_production_finite_difference_and_immutability');

      // A scratch cell can run in the notebook, but the checker never receives it.
      const scratch = await scratchEditor(page);
      await scratch.click();
      await page.keyboard.press('ControlOrMeta+A');
      await page.keyboard.insertText('secret_for_scratch = 1');
      await page.getByRole('button', { name: 'Run cell' }).click();
      await replaceAnswer(`print(secret_for_scratch)\n${reference}`);
      await check.click();
      await expect(status).toContainText('NameError', { timeout: 120_000 });

      await replaceAnswer('def softmax_cross_entropy(:\n    pass');
      await check.click();
      await expect(status).toContainText('SyntaxError', { timeout: 120_000 });
      await expect(status).toContainText('solution.py');

      await replaceAnswer('import numpy as np');
      await check.click();
      await expect(status).toContainText('Missing candidate function', { timeout: 120_000 });
    });
  });
}
