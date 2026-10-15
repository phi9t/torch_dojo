import type { Page } from '@playwright/test';

/** Locators for the Bench: a problem pane with tabs, and an answer pane holding the answer cell. */

export function answerPane(page: Page) {
  return page.getByRole('region', { name: 'Answer', exact: true });
}

export function problemPane(page: Page) {
  return page.getByRole('region', { name: 'Problem', exact: true });
}

export function answerEditor(page: Page) {
  return answerPane(page).locator('.jp-CodeCell').filter({ visible: true }).first().locator('.cm-content');
}

/** Opens the scratch cells under the answer (if closed) and returns the last scratch editor. */
export async function scratchEditor(page: Page) {
  const toggle = page.getByRole('button', { name: 'Scratch cell', exact: true });
  if (await toggle.isVisible()) await toggle.click();
  return answerPane(page).locator('.jp-CodeCell').filter({ visible: true }).last().locator('.cm-content');
}

export async function openTab(page: Page, name: 'Brief' | 'Drills' | 'Explore' | 'Reps') {
  await page.getByRole('tab', { name: new RegExp(`^${name}`) }).click();
}

/** The visible exploration panels in the Explore tab. */
export async function explorePanels(page: Page) {
  await openTab(page, 'Explore');
  return problemPane(page).locator('.jp-CodeCell').filter({ visible: true });
}

/** Runs the n-th exploration panel from its own Run control. */
export async function runExplorePanel(page: Page, index = 0) {
  const panels = await explorePanels(page);
  await panels.nth(index).getByRole('button', { name: 'Run', exact: true }).click();
  return panels.nth(index);
}

export function briefHeading(page: Page) {
  return page.locator('.jp-MarkdownCell h1').filter({ visible: true }).first();
}

export async function replaceAnswer(page: Page, source: string) {
  await answerEditor(page).click();
  await page.keyboard.press('ControlOrMeta+A');
  await page.keyboard.insertText(source);
}

/** A stage's seal in the rail; labs list checkpoints instead of stages. */
export function seal(page: Page, name: string, list: 'Stages' | 'Checkpoints' = 'Stages') {
  return page.getByRole('list', { name: list }).getByRole('listitem').filter({ hasText: name });
}

/** Records whether a seal starts its stamp animation from now on. */
export async function watchStamp(page: Page, name: string, list: 'Stages' | 'Checkpoints' = 'Stages'): Promise<() => Promise<boolean>> {
  const mark = seal(page, name, list).locator('[aria-hidden="true"]');
  await mark.evaluate(node => {
    node.addEventListener('animationstart', () => { node.setAttribute('data-test-stamped', 'true'); });
  });
  return async () => (await mark.getAttribute('data-test-stamped')) === 'true';
}
