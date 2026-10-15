import { expect, test, type Page } from '@playwright/test';
import { answerEditor } from './bench';

async function openWorkout(page: Page) {
  await page.goto('lab/?exercise=0059');
  await expect(page.locator('.jp-MarkdownCell h1').first()).toContainText('0059 · Stable Softmax Cross-Entropy', {
    timeout: 60_000
  });
  await expect(page.getByRole('button', { name: /^Check (?!log)/ })).toBeEnabled({ timeout: 60_000 });
}

/** Relative luminance (0 dark … 1 light) of an element's computed background. */
async function backgroundLuminance(page: Page, selector: string): Promise<number> {
  return page.locator(selector).first().evaluate(element => {
    const match = getComputedStyle(element).backgroundColor.match(/[\d.]+/g);
    if (!match) throw new Error('No background colour');
    const [r, g, b] = match.slice(0, 3).map(Number).map(channel => {
      const c = channel / 255;
      return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
    });
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  });
}

const pageGround = 'body';
const editor = '[aria-label="Answer"] .jp-CodeCell .jp-Cell-inputWrapper';
const prose = '.jp-MarkdownCell .jp-RenderedHTMLCommon';

test('a workout shows none of the JupyterLab chrome', async ({ page }) => {
  await openWorkout(page);
  // Each of these is turned off by disabledExtensions in jupyter-lite.json (ADR 0003).
  // A JupyterLite upgrade that renames one of those plugin IDs brings it back and fails here.
  const chrome = {
    'menu bar': page.getByRole('menubar'),
    'File menu': page.getByRole('menuitem', { name: 'File' }),
    'document tabs': page.locator('.lm-DockPanel-tabBar'),
    'title and menu strips': page.locator('#jp-top-panel, #jp-menu-panel'),
    'status bar': page.locator('#jp-main-statusbar'),
    'side icon bars': page.locator('.jp-SideBar'),
    'side panels': page.locator('#jp-left-stack, #jp-right-stack'),
    'file browser': page.locator('#filebrowser'),
    launcher: page.locator('.jp-Launcher'),
    'running sessions': page.locator('#jp-running-sessions'),
    'table of contents': page.locator('#table-of-contents'),
    'notebook toolbar': page.locator('.jp-NotebookPanel-toolbar'),
    'cell toolbar': page.getByRole('button', { name: 'Create a duplicate of this cell below' })
  };
  for (const [name, locator] of Object.entries(chrome)) {
    await expect(locator.filter({ visible: true }), `${name} should be hidden`).toHaveCount(0);
  }

  await page.keyboard.press('ControlOrMeta+Shift+C');
  await expect(page.locator('#command-palette').filter({ visible: true }), 'command palette').toHaveCount(0);

  // What remains: the gym's own controls, the cell editor and rendered markdown.
  await expect(page.getByRole('button', { name: 'Run cell' })).toBeVisible();
  await expect(answerEditor(page)).toBeVisible();
  await expect(page.locator(prose).first()).toBeVisible();
  await expect(page.getByTestId('dojo-kernel-status')).toHaveText('Python · idle', { timeout: 60_000 });
});

test('fonts load from the site itself, never from a font CDN', async ({ page, baseURL }) => {
  const fontRequests: string[] = [];
  page.on('request', request => {
    if (request.resourceType() === 'font' || /fonts\.(googleapis|gstatic)\.com/.test(request.url())) {
      fontRequests.push(request.url());
    }
  });
  await openWorkout(page);
  const loaded = await page.evaluate(async () => {
    await document.fonts.ready;
    return Array.from(document.fonts)
      .filter(face => face.status === 'loaded')
      .map(face => face.family.replace(/"/g, ''));
  });
  expect(loaded).toContain('Newsreader');
  expect(loaded).toContain('JetBrains Mono');
  expect(fontRequests.length).toBeGreaterThan(0);
  for (const url of fontRequests) expect(url.startsWith(new URL(baseURL!).origin)).toBe(true);
});

test('the theme follows the system until the toggle overrides it, and the choice is remembered', async ({ page }) => {
  await page.emulateMedia({ colorScheme: 'dark' });
  await openWorkout(page);
  for (const surface of [pageGround, editor, prose]) {
    await expect.poll(() => backgroundLuminance(page, surface), { message: `${surface} in system dark` }).toBeLessThan(0.05);
  }

  await page.emulateMedia({ colorScheme: 'light' });
  for (const surface of [pageGround, editor, prose]) {
    await expect.poll(() => backgroundLuminance(page, surface), { message: `${surface} in system light` }).toBeGreaterThan(0.8);
  }

  await page.emulateMedia({ colorScheme: 'dark' });
  await page.evaluate(() => { (window as Window & { __sameView?: boolean }).__sameView = true; });
  await page.getByRole('button', { name: 'Switch to light theme' }).click();
  for (const surface of [pageGround, editor, prose]) {
    await expect.poll(() => backgroundLuminance(page, surface), { message: `${surface} after toggle` }).toBeGreaterThan(0.8);
  }
  expect(await page.evaluate(() => (window as Window & { __sameView?: boolean }).__sameView)).toBe(true);
  await expect(page.getByRole('button', { name: 'Switch to dark theme' })).toBeVisible();

  await page.reload();
  await openWorkout(page);
  await expect.poll(() => backgroundLuminance(page, pageGround), { message: 'explicit light survives reload under system dark' }).toBeGreaterThan(0.8);
  await page.getByRole('button', { name: 'Switch to dark theme' }).click();
  await expect.poll(() => backgroundLuminance(page, editor)).toBeLessThan(0.05);
});

test('the workout reads at phone width without horizontal page scroll', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await openWorkout(page);
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  expect(overflow).toBeLessThanOrEqual(0);
  await expect(page.getByRole('button', { name: 'Learning path' })).toBeVisible();
});
