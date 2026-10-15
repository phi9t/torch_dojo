import { expect, test, type Page } from '@playwright/test';
import { openTab } from './bench';

const STORE = 'torch-dojo.browser-gym.attempts.v1';
/** Every workout in the built catalog, so adding one never edits this spec. */
async function workoutIds(page: Page): Promise<string[]> {
  const catalog = await (await page.request.get('gym-assets/catalog.json')).json();
  return catalog.map((entry: { id: string }) => entry.id);
}

function daysAgo(days: number): string {
  return new Date(Date.now() - days * 24 * 60 * 60 * 1000).toISOString();
}

function seededRep(exerciseId: string, updatedAt: string, finishSeconds?: number) {
  const receipts = finishSeconds === undefined ? [] : [{
    id: `receipt-${exerciseId}`, exerciseId, checkedAt: updatedAt, source: '', sha256: 'seeded', status: 'passed',
    checkerVersion: 'seeded', runtime: 'seeded', fixtureId: 'seeded', result: { stage: 'production', checks: [] },
    activeSecondsAtCheck: finishSeconds
  }];
  return {
    id: `rep-${exerciseId}`, exerciseId, createdAt: updatedAt, updatedAt, activeSeconds: 60,
    notebook: { cells: [], metadata: {}, nbformat: 4, nbformat_minor: 5 }, receipts
  };
}

async function seedReps(page: Page, reps: ReturnType<typeof seededRep>[]) {
  await page.addInitScript(([key, state]) => {
    if (!window.sessionStorage.getItem('seeded')) {
      window.localStorage.setItem(key, state);
      window.sessionStorage.setItem('seeded', 'yes');
    }
  }, [STORE, JSON.stringify({ attempts: reps, activeByExercise: {} })] as const);
}

function home(page: Page) {
  return page.getByRole('region', { name: 'Workouts', exact: true });
}

test('the site root opens a home page of every workout, grouped by topic', async ({ page }) => {
  await seedReps(page, [seededRep('0059', daysAgo(3), 252)]);
  // The site root, under any Pages prefix in the base URL.
  await page.goto('./');
  await expect(home(page)).toBeVisible({ timeout: 60_000 });
  await expect(page).not.toHaveURL(/exercise=/);
  for (const topic of ['Training', 'Evaluation', 'Geometry', 'Sensors']) {
    await expect(home(page).getByRole('region', { name: topic })).toBeVisible();
  }
  await expect(home(page).getByRole('region', { name: 'Geometry' })).toContainText('0062 · Pairwise Box IoU');
  await expect(home(page).getByRole('region', { name: 'Sensors' })).toContainText('0066 · Sparse Voxel Pooling');
  // A repped workout shows its standing; one never repped shows no stats rather than zeros.
  const repped = home(page).getByRole('listitem').filter({ hasText: '0059 · Stable Softmax Cross-Entropy' });
  await expect(repped).toContainText('best 4:12 · 1 rep · passed Production · last rep 3 days ago');
  const fresh = home(page).getByRole('listitem').filter({ hasText: '0061 · K-Means Clustering' });
  await expect(fresh).toHaveText('0061 · K-Means Clustering');
  // The theme toggle is a compact control beside the heading, not a bar across the page.
  const toggle = home(page).getByRole('button', { name: /^Switch to (dark|light) theme$/ });
  expect((await toggle.boundingBox())!.width).toBeLessThan(80);
  // No JupyterLab chrome on the home page either.
  await expect(page.locator('#jp-menu-panel')).toBeHidden();

  await home(page).getByRole('region', { name: 'Training' }).getByRole('link', { name: '0060 · One honest classifier update' }).click();
  await expect(page).toHaveURL(/\/lab\/.*\?exercise=0060$/);
  await expect(page.getByRole('button', { name: /^Check (?!log)/ })).toBeEnabled({ timeout: 60_000 });
  await page.getByRole('link', { name: '← All workouts' }).click();
  await expect(home(page)).toBeVisible({ timeout: 60_000 });
});

test('Next up suggests a workout never repped, in learning-path order', async ({ page }) => {
  await seedReps(page, [seededRep('0059', daysAgo(1))]);
  await page.goto('lab/');
  const next = home(page).getByRole('region', { name: 'Next up' });
  await expect(next).toBeVisible({ timeout: 60_000 });
  await expect(next.getByRole('link')).toHaveText('0060 · One honest classifier update');
  await expect(next).toContainText('not done a rep');
});

test('once every workout has a rep, Next up is the one with the oldest last rep', async ({ page }) => {
  const ids = await workoutIds(page);
  await seedReps(page, ids.map((id, index) => seededRep(id, daysAgo(id === '0065' ? 40 : (index % 20) + 1))));
  await page.goto('lab/');
  const next = home(page).getByRole('region', { name: 'Next up' });
  await expect(next.getByRole('link')).toHaveText('0065 · Temporal Ego Coordinates', { timeout: 60_000 });
  await expect(next).toContainText('5 weeks ago');
});

test('⌘K filters workouts and opens one, even from a drill prediction', async ({ page }) => {
  await seedReps(page, [seededRep('0066', daysAgo(2))]);
  await page.goto('lab/?exercise=0059');
  await expect(page.getByRole('button', { name: /^Check (?!log)/ })).toBeEnabled({ timeout: 60_000 });
  // A deep link bypasses the home page.
  await expect(home(page)).toHaveCount(0);

  await openTab(page, 'Drills');
  await page.getByLabel('Your prediction').first().click();
  await page.keyboard.press('ControlOrMeta+K');
  const switcher = page.getByRole('dialog', { name: 'Switch workout' });
  await expect(switcher).toBeVisible();
  const search = switcher.getByRole('combobox', { name: 'Find a workout' });
  await expect(search).toBeFocused();
  await expect(switcher.getByRole('option')).toHaveCount((await workoutIds(page)).length);
  await expect(switcher.getByRole('option', { name: /0061 · K-Means/ })).toContainText('never repped');

  await search.fill('voxel');
  await expect(switcher.getByRole('option')).toHaveCount(1);
  await expect(switcher.getByRole('option')).toContainText('last rep 2 days ago');
  await search.fill('geometry');
  await expect(switcher.getByRole('option')).toHaveCount(4);
  await search.press('ArrowDown');
  await expect(switcher.getByRole('option', { selected: true })).toContainText('0063 · Oriented Box Sampling');
  await search.press('Escape');
  await expect(switcher).toBeHidden();

  await page.keyboard.press('ControlOrMeta+K');
  await search.fill('voxel');
  await search.press('Enter');
  await expect(page).toHaveURL(/\?exercise=0066$/);
  await expect(page.getByRole('button', { name: /^Check (?!log)/ })).toBeEnabled({ timeout: 60_000 });
});
