import { expect, test, type Page } from '@playwright/test';
import { runExplorePanel } from './bench';

const CONTROLS = 'a[href], button, input, select, textarea, [role="tab"], [role="option"], [tabindex="0"]';

/** Whether the focused control draws a visible indicator: an outline, or an inset rule like the switcher field's. */
async function focusedWithoutRing(page: Page): Promise<string | null> {
  return page.evaluate(controls => {
    const element = document.activeElement as HTMLElement | null;
    // Code editors are text areas the learner types in; they show their caret, not a ring.
    if (!element || element === document.body || element.isContentEditable || !element.matches(controls)) return null;
    const style = getComputedStyle(element);
    if (style.outlineStyle !== 'none' && parseFloat(style.outlineWidth) > 0) return null;
    if (style.boxShadow.includes('inset')) return null;
    return `${element.tagName.toLowerCase()} "${(element.getAttribute('aria-label') ?? element.textContent ?? '').trim().slice(0, 40)}"`;
  }, CONTROLS);
}

/** Tabs through the page and returns every control that took focus without a visible indicator. */
async function controlsWithoutFocusRing(page: Page, presses: number): Promise<string[]> {
  const missing: string[] = [];
  for (let i = 0; i < presses; i++) {
    await page.keyboard.press('Tab');
    const found = await focusedWithoutRing(page);
    if (found) missing.push(found);
  }
  return [...new Set(missing)];
}

test('every control reached by Tab on the Bench shows a focus indicator', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('lab/?exercise=0059');
  await expect(page.getByRole('button', { name: /^Check (?!log)/ })).toBeEnabled({ timeout: 60_000 });
  await page.getByRole('link', { name: '← All workouts' }).focus();
  expect(await controlsWithoutFocusRing(page, 30)).toEqual([]);
  // The drill cards: prediction fields, Run and Reveal reasoning.
  await page.getByRole('tab', { name: /^Drills/ }).click();
  await page.getByLabel('Your prediction').first().focus();
  expect(await controlsWithoutFocusRing(page, 8)).toEqual([]);
  // The ⌘K switcher's search field.
  await page.keyboard.press('ControlOrMeta+K');
  await expect(page.getByRole('combobox', { name: 'Find a workout' })).toBeFocused();
  await page.keyboard.press('Shift+Tab');
  await page.keyboard.press('Tab');
  expect(await focusedWithoutRing(page)).toBeNull();
});

test('every control reached by Tab on the home page shows a focus indicator', async ({ page }) => {
  await page.goto('lab/');
  await expect(page.getByRole('region', { name: 'Workouts', exact: true })).toBeVisible({ timeout: 60_000 });
  expect(await controlsWithoutFocusRing(page, 20)).toEqual([]);
});

test.describe('dark theme', () => {
  test.use({ colorScheme: 'dark' });

  for (const id of ['0059', '0061', '0064', '0066', '0069']) {
    test(`${id}'s explore panel never glares or hides its ink`, async ({ page }) => {
      await page.goto(`lab/?exercise=${id}`);
      await expect(page.getByRole('button', { name: /^Check (?!log)/ })).toBeEnabled({ timeout: 60_000 });
      const panel = await runExplorePanel(page);
      // The lesson's own drawing, not a toolbar icon.
      await expect(panel.locator('svg').filter({ visible: true }).first()).toBeVisible({ timeout: 60_000 });
      await page.waitForTimeout(1_000);
      const light = await panel.evaluate(root => {
        const found: string[] = [];
        const channels = (colour: string) => (colour.match(/[\d.]+/g) ?? []).map(Number);
        for (const element of [root, ...root.querySelectorAll('*')] as Element[]) {
          const style = getComputedStyle(element);
          const [r, g, b, a = 1] = channels(style.backgroundColor);
          if (a > 0.5 && r + g + b > 600) found.push(`background of ${element.tagName.toLowerCase()}`);
          if (element instanceof SVGElement && !(element instanceof SVGSVGElement)) {
            const [fr, fg, fb] = channels(style.fill);
            // Light shapes glare on the dark ground; dark text vanishes into it.
            if (element.tagName !== 'text' && style.fill.startsWith('rgb') && fr + fg + fb > 600) found.push(`light fill on ${element.tagName}`);
            if (element.tagName === 'text' && style.fill.startsWith('rgb') && fr + fg + fb < 200) found.push(`dark text "${element.textContent}"`);
          }
        }
        return [...new Set(found)];
      });
      expect(light).toEqual([]);
    });
  }
});
