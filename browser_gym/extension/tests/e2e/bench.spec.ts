import { expect, test, type Page } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { answerEditor, answerPane, explorePanels, openTab, problemPane, replaceAnswer, scratchEditor } from './bench';

const evalSource = readFileSync(resolve(__dirname, '../../../../dojo/0059_stable_cross_entropy_eval.py'), 'utf8');
const reference = 'import numpy as np\n' + evalSource.match(/def softmax_cross_entropy[\s\S]*?(?=\n\n# %% \[verifiers\])/)![0];

async function openWorkout(page: Page, id = '0059') {
  await page.goto(`lab/?exercise=${id}`);
  await expect(page.getByRole('button', { name: /^Check (?!log)/ })).toBeEnabled({ timeout: 60_000 });
}

test('the problem and the answer sit side by side, one tab at a time', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await openWorkout(page);
  const problem = problemPane(page);
  const answer = answerPane(page);
  const problemBox = (await problem.boundingBox())!;
  const answerBox = (await answer.boundingBox())!;
  expect(answerBox.x).toBeGreaterThan(problemBox.x + problemBox.width - 2);
  expect(Math.abs(answerBox.y - problemBox.y)).toBeLessThan(2);

  await expect(problem.getByRole('heading', { name: 'Stable Softmax Cross-Entropy', exact: true })).toBeVisible();
  const tabs = problem.getByRole('tablist', { name: 'Workout views' });
  await expect(tabs.getByRole('tab')).toHaveText([/^Brief/, /^Drills\s*3/, /^Explore\s*1/, /^Reps/]);
  const brief = tabs.getByRole('tab', { name: 'Brief' });
  await expect(brief).toHaveAttribute('aria-selected', 'true');

  // Brief: the brief only.
  await expect(problem.locator('.jp-MarkdownCell h1').filter({ visible: true })).toContainText('0059');
  await expect(problem.getByRole('button', { name: 'New rep' })).toBeHidden();
  // The answer pane shows only the answer cell; the brief and exploration live on the left.
  await expect(answer.locator('.jp-Cell').filter({ visible: true })).toHaveCount(1);
  await expect(answerEditor(page)).toContainText('def softmax_cross_entropy');

  // Explore: panels only.
  await brief.focus();
  await page.keyboard.press('ArrowRight');
  await expect(tabs.getByRole('tab', { name: /Drills/ })).toBeFocused();
  await page.keyboard.press('ArrowRight');
  const explore = tabs.getByRole('tab', { name: /Explore/ });
  await expect(explore).toBeFocused();
  await expect(explore).toHaveAttribute('aria-selected', 'true');
  await expect(problem.locator('.jp-MarkdownCell h1').filter({ visible: true })).toHaveCount(0);
  const panels = problem.locator('.jp-CodeCell').filter({ visible: true });
  await expect(panels).toHaveCount(1);
  await expect(panels.first().getByRole('button', { name: 'Run', exact: true })).toBeVisible();

  // Reps: the rep log and notes.
  await page.keyboard.press('End');
  const reps = tabs.getByRole('tab', { name: /Reps/ });
  await expect(reps).toBeFocused();
  await expect(problem.getByRole('button', { name: 'New rep' })).toBeVisible();
  await expect(problem.getByLabel('Rep notes')).toBeVisible();
  await expect(problem.locator('.jp-Cell').filter({ visible: true })).toHaveCount(0);
  await page.keyboard.press('ArrowRight');
  await expect(brief).toHaveAttribute('aria-selected', 'true');
  await page.keyboard.press('ArrowLeft');
  await expect(reps).toHaveAttribute('aria-selected', 'true');

  // The chosen tab is remembered per workout.
  await page.reload();
  await expect(page.getByRole('tab', { name: /Reps/ })).toHaveAttribute('aria-selected', 'true', { timeout: 60_000 });
  await openWorkout(page, '0060');
  await expect(page.getByRole('tab', { name: 'Brief' })).toHaveAttribute('aria-selected', 'true');
});

test('explore panels show their output, with the loader code behind a disclosure', async ({ page }) => {
  await openWorkout(page);
  const panel = (await explorePanels(page)).first();
  await expect(panel.getByText('pyfetch')).toBeHidden();
  await expect(panel.getByRole('button', { name: 'Show code' })).toHaveAttribute('aria-expanded', 'false');
  await panel.getByRole('button', { name: 'Run', exact: true }).click();
  await expect(page.getByRole('group', { name: 'Tensor card for logits_probe' })).toBeVisible({ timeout: 60_000 });
  // Outputs render once, in the pane that shows them.
  await expect(page.getByRole('group', { name: 'Tensor card for logits_probe' })).toHaveCount(1);
  await expect(panel.getByText('pyfetch')).toBeHidden();
  await panel.getByRole('button', { name: 'Show code' }).click();
  await expect(panel.locator('.cm-content')).toContainText('pyfetch');
});

test('the scratch cell starts collapsed and never reaches the checked source', async ({ page }) => {
  await openWorkout(page);
  const toggle = page.getByRole('button', { name: 'Scratch cell', exact: true });
  await expect(toggle).toHaveAttribute('aria-expanded', 'false');
  await expect(answerPane(page).locator('.jp-CodeCell').filter({ visible: true })).toHaveCount(1);
  const scratch = await scratchEditor(page);
  await expect(page.getByRole('button', { name: 'Hide scratch cell' })).toHaveAttribute('aria-expanded', 'true');
  await scratch.click();
  await page.keyboard.press('ControlOrMeta+A');
  await page.keyboard.insertText('scratch_only = 7');
  await replaceAnswer(page, reference);
  await page.getByRole('button', { name: /^Check (?!log)/ }).click();
  const status = page.locator('#torch-dojo-gym [role="status"]');
  await expect(status).toContainText('PASSED · source SHA-256', { timeout: 120_000 });
  await page.getByRole('tab', { name: /Reps/ }).click();
  const archive = await Promise.all([
    page.waitForEvent('download'),
    page.getByRole('button', { name: 'Export rep' }).click()
  ]).then(async ([download]) => JSON.parse(readFileSync((await download.path())!, 'utf8')));
  expect(archive.attempts[0].receipts[0].source).toBe(reference);
  expect(archive.attempts[0].receipts[0].source).not.toContain('scratch_only');
});

test('keyboard: Shift+Enter runs, Accel+Enter checks, Accel+. stops', async ({ page }) => {
  await openWorkout(page);
  const status = page.locator('#torch-dojo-gym [role="status"]');
  const check = page.getByRole('button', { name: /^Check (?!log)/ });
  const stop = page.getByRole('button', { name: 'Stop' });

  // Run works before the learner has clicked into any cell: it runs the answer cell.
  const answerCell = answerPane(page).locator('.jp-CodeCell').filter({ visible: true }).first();
  await page.getByRole('button', { name: 'Run cell' }).click();
  await expect(answerCell.locator('.jp-InputPrompt')).toContainText(/\[\d+\]/, { timeout: 60_000 });

  await replaceAnswer(page, 'print("ran from the keyboard")\nimport numpy as np');
  await page.keyboard.press('Shift+Enter');
  await expect(answerPane(page).getByText('ran from the keyboard').first()).toBeVisible({ timeout: 60_000 });
  // Shift+Enter keeps the learner in the answer cell rather than moving to a hidden cell.
  await page.keyboard.insertText('\n# still here');
  await expect(answerEditor(page)).toContainText('# still here');

  await page.keyboard.press('ControlOrMeta+Enter');
  await expect(status).toContainText('FAILED · source SHA-256', { timeout: 120_000 });

  await replaceAnswer(page, `${reference}\nimport time\nwhile True:\n    time.sleep(10)`);
  await page.keyboard.press('ControlOrMeta+Enter');
  await expect(stop).toBeEnabled({ timeout: 15_000 });
  await page.keyboard.press('ControlOrMeta+.');
  await expect(status).toContainText('STOPPED · source SHA-256', { timeout: 15_000 });
  await expect(check).toBeEnabled();
});

test('below 1100px the rail, problem and answer stack without horizontal scroll', async ({ page }) => {
  await page.setViewportSize({ width: 900, height: 800 });
  await openWorkout(page);
  const rail = page.getByRole('complementary', { name: 'Workout' });
  const railBox = (await rail.boundingBox())!;
  const problemBox = (await problemPane(page).boundingBox())!;
  const answerBox = (await answerPane(page).boundingBox())!;
  expect(problemBox.y).toBeGreaterThanOrEqual(railBox.y + railBox.height - 2);
  expect(answerBox.y).toBeGreaterThanOrEqual(problemBox.y + problemBox.height - 2);
  expect(await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)).toBeLessThanOrEqual(0);
  await answerPane(page).scrollIntoViewIfNeeded();
  await expect(answerEditor(page)).toBeVisible();
});

test('the answer can take the whole page, and stays that way across reloads', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await openWorkout(page);
  const normalHeight = (await answerEditor(page).boundingBox())!.height;
  expect(normalHeight).toBeGreaterThanOrEqual(200);

  await page.getByRole('button', { name: 'Expand', exact: true }).click();
  await expect(problemPane(page)).toBeHidden();
  expect((await answerPane(page).boundingBox())!.width).toBeGreaterThan(1100);
  expect((await answerEditor(page).boundingBox())!.height).toBeGreaterThan(500);

  await page.reload();
  await expect(page.getByRole('button', { name: /^Check (?!log)/ })).toBeEnabled({ timeout: 60_000 });
  await expect(page.getByRole('button', { name: 'Show problem' })).toHaveAttribute('aria-pressed', 'true');
  await expect(problemPane(page)).toBeHidden();

  // ⌘/Ctrl+Shift+E brings the problem back.
  await answerEditor(page).click();
  await page.keyboard.press('ControlOrMeta+Shift+E');
  await expect(problemPane(page)).toBeVisible();
  await expect(page.getByRole('button', { name: 'Expand', exact: true })).toHaveAttribute('aria-pressed', 'false');
});

test('⌘/Ctrl+Shift+Enter runs the answer and then the scratch cells', async ({ page }) => {
  await openWorkout(page);
  await replaceAnswer(page, 'def answer_probe():\n    return 41 + 1\n');
  const scratch = await scratchEditor(page);
  await scratch.click();
  await page.keyboard.press('ControlOrMeta+A');
  await page.keyboard.insertText('print("probe", answer_probe())');
  const output = answerPane(page).locator('.jp-CodeCell').filter({ visible: true }).last().locator('.jp-OutputArea');

  await answerEditor(page).click();
  await page.keyboard.press('ControlOrMeta+Shift+Enter');
  await expect(output).toContainText('probe 42', { timeout: 60_000 });

  // The answer reruns first, so scratch sees the edit without a separate run.
  await replaceAnswer(page, 'def answer_probe():\n    return 7\n');
  await page.getByRole('button', { name: 'Run scratch' }).click();
  await expect(output).toContainText('probe 7');
});

test('the scratch area explains itself and lists the keyboard shortcuts', async ({ page }) => {
  await openWorkout(page);
  const runScratch = page.getByRole('button', { name: 'Run scratch' });
  await expect(runScratch).toBeHidden();
  await page.getByRole('button', { name: 'Scratch cell', exact: true }).click();
  await expect(runScratch).toBeVisible();
  await expect(runScratch).toHaveAttribute('title', /Run the answer, then each scratch cell \((⌘|Ctrl)\+Shift\+Enter\)/);
  await expect(page.getByText(/reruns your answer first, then each scratch cell/)).toBeVisible();
  await expect(page.getByRole('button', { name: /^Check (?!log)/ })).toHaveAttribute('title', /\((⌘|Ctrl)\+Enter\)$/);

  const shortcuts = page.getByRole('button', { name: 'Shortcuts' });
  await shortcuts.click();
  await expect(shortcuts).toHaveAttribute('aria-expanded', 'true');
  const list = page.getByLabel('Keyboard shortcuts');
  await expect(list).toContainText('Run the answer, then each scratch cell');
  await expect(list).toContainText('Expand the answer, or show the problem again');
  await shortcuts.click();
  await expect(list).toBeHidden();
});

test('Shift+Enter runs the cell being typed in, whichever pane it is in', async ({ page }) => {
  await page.goto('lab/?exercise=0068');
  await expect(page.getByRole('button', { name: /^Check (?!log)/ })).toBeEnabled({ timeout: 60_000 });
  // Pressed before the kernel is idle: the run waits for it rather than being dropped.
  const scratch = await scratchEditor(page);
  await scratch.click();
  await page.keyboard.press('ControlOrMeta+A');
  await page.keyboard.insertText('print("scratch ran", 6 * 7)');
  await page.keyboard.press('Shift+Enter');
  const scratchCell = answerPane(page).locator('.jp-CodeCell').filter({ visible: true }).last();
  await expect(scratchCell).toContainText('scratch ran 42', { timeout: 60_000 });

  // After running the answer, Shift+Enter in scratch still runs scratch, not the brief.
  await answerEditor(page).click();
  await page.keyboard.press('Shift+Enter');
  await scratch.click();
  await page.keyboard.press('ControlOrMeta+A');
  await page.keyboard.insertText('print("again", 2 + 2)');
  await page.keyboard.press('Shift+Enter');
  await expect(scratchCell).toContainText('again 4');

  // In the problem pane it runs the drill being edited.
  await openTab(page, 'Drills');
  const drill = problemPane(page).getByRole('group').first();
  await drill.locator('.cm-content').click();
  await page.keyboard.press('Shift+Enter');
  await expect(drill.locator('.jp-OutputArea')).not.toBeEmpty({ timeout: 30_000 });
});
