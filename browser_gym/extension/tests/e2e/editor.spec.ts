import { expect, test, type Page } from '@playwright/test';
import { answerEditor } from './bench';

async function openReady(page: Page) {
  await page.goto('lab/?exercise=0068');
  await expect(page.getByRole('button', { name: /^Check (?!log)/ })).toBeEnabled({ timeout: 60_000 });
  await expect(page.getByTestId('dojo-kernel-status')).toHaveText('Python · idle', { timeout: 60_000 });
}

async function newLineInAnswer(page: Page) {
  await answerEditor(page).click();
  await page.keyboard.press('ControlOrMeta+End');
  await page.keyboard.press('Enter');
}

const lastLine = async (page: Page) => (await answerEditor(page).innerText()).split('\n').pop();

test('NumPy names complete as you type, and Tab accepts one', async ({ page }) => {
  await openReady(page);
  await newLineInAnswer(page);
  // No cell has run yet: the kernel already has `np` for completion.
  await page.keyboard.type('np.argm', { delay: 50 });
  const suggestions = page.getByRole('listbox');
  await expect(suggestions.getByRole('option', { name: /^argmax/ })).toBeVisible({ timeout: 15_000 });
  await expect(suggestions.getByRole('option', { name: /^argmin/ })).toBeVisible();
  await page.keyboard.press('Tab');
  expect(await lastLine(page)).toBe('np.argmax');
});

test('Emacs keys are opt-in, edit the line, and keep Ctrl+K in the editor', async ({ page }) => {
  await openReady(page);
  await page.getByRole('button', { name: 'Scratch cell', exact: true }).click();
  await page.getByRole('button', { name: 'Shortcuts' }).click();
  const emacs = page.getByLabel(/Emacs keys in the editor/);
  await expect(emacs).not.toBeChecked();
  await emacs.check();

  await newLineInAnswer(page);
  await page.keyboard.type('abc');
  await page.keyboard.press('Control+a');
  await page.keyboard.type('X');
  await page.keyboard.press('Control+e');
  await page.keyboard.type('Y');
  expect(await lastLine(page)).toBe('XabcY');
  await page.keyboard.press('Control+a');
  await page.keyboard.press('Control+k');
  await expect(page.getByRole('dialog', { name: 'Switch workout' })).toBeHidden();
  await page.keyboard.type('killed: ');
  await page.keyboard.press('Control+y');
  expect(await lastLine(page)).toBe('killed: XabcY');

  // The choice is remembered.
  await page.reload();
  await expect(page.getByRole('button', { name: /^Check (?!log)/ })).toBeEnabled({ timeout: 60_000 });
  await page.getByRole('button', { name: 'Scratch cell', exact: true }).click();
  await page.getByRole('button', { name: 'Shortcuts' }).click();
  await expect(page.getByLabel(/Emacs keys in the editor/)).toBeChecked();
});
