const { chromium, expect } = require('@playwright/test');
const fs = require('node:fs/promises');
const path = require('node:path');

// Runs against the packaged page under its Pages prefix as well as local source.
(async () => {
  const target = process.argv[2] || 'http://localhost:8765/site/fundamentals/';
  const browser = await chromium.launch({
    ...(process.env.DOJO_BROWSER_EXECUTABLE ? { executablePath: process.env.DOJO_BROWSER_EXECUTABLE } : {}),
    args: ['--no-sandbox']
  });
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
    page.on('response', r => { if (r.status() >= 400) errors.push(`${r.status()} ${r.url()}`); });
    const lessons = ['start', 'masks', 'contrastive', 'conv', 'iou', 'huber', 'voxel', 'ego'];
    const snapshots = process.env.DOJO_SCREENSHOT_DIR;
    if (snapshots) await fs.mkdir(snapshots, { recursive: true });
    const select = async id => {
      await page.goto(`${target}#${id}`);
      await expect(page.locator('#view h1')).toBeVisible();
      await expect(page.locator(`#tab-${id}`)).toHaveAttribute('aria-selected', 'true');
    };
    const range = async (name, value) => {
      const slider = page.getByRole('slider', { name, exact: true });
      await slider.fill(String(value));
      await slider.dispatchEvent('input');
    };
    for (const theme of ['dark', 'light']) {
      for (const width of [320, 390, 768, 1440]) {
        await page.setViewportSize({ width, height: 1000 });
        for (const id of lessons) {
          await select(id);
          if (await page.locator('html').getAttribute('data-theme') !== theme) await page.locator('#theme').click();
          const fits = await page.evaluate(() => ({
            page: document.documentElement.scrollWidth <= innerWidth,
            stages: [...document.querySelectorAll('.stage')].every(e => e.scrollWidth <= e.clientWidth + 1)
          }));
          expect(fits, `${id}, ${theme}, ${width}px`).toEqual({ page: true, stages: true });
          if (snapshots && [390, 1440].includes(width)) {
            await page.screenshot({ path: path.join(snapshots, `${id}-${theme}-${width}.png`), fullPage: true });
          }
        }
      }
    }
    await page.setViewportSize({ width: 1440, height: 1000 });
    await select('start');
    await page.getByRole('tab', { name: 'Start' }).focus();
    await page.keyboard.press('ArrowRight');
    await expect(page.locator('#tab-masks')).toBeFocused();
    await expect(page.locator('#view')).toHaveAttribute('aria-labelledby', 'tab-masks');

    // Empty rows must become NaN only in the deliberately unsafe demo.
    await page.getByRole('button', { name: 'safe softmax', exact: true }).click();
    await expect(page.locator('.stage')).toContainText('NaN row');
    await page.getByRole('button', { name: 'safe softmax', exact: true }).click();
    for (const i of [0, 1, 3, 4]) await page.getByRole('button', { name: `t${i}`, exact: true }).click();
    await expect(page.locator('.readout')).toContainText('5');
    await expect(page.locator('.stage')).not.toContainText('NaN');

    await select('contrastive');
    await range('temperature', 0.05);
    for (let i = 0; i < 4; i++) await page.getByRole('button', { name: `p${i}`, exact: true }).click();
    await expect(page.locator('.stage')).toContainText('No valid pairs → loss 0');

    await select('conv');
    await page.getByRole('button', { name: 'output 1,1 = -4', exact: true }).press('Enter');
    await expect(page.locator('.kv')).toContainText('out[1,1]');
    await page.getByRole('button', { name: '5×5', exact: true }).click();
    await page.getByRole('group', { name: 'padding', exact: true }).getByRole('button', { name: '2', exact: true }).click();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);

    await select('iou');
    const diagram = page.getByRole('img', { name: 'two boxes', exact: true });
    const box = await diagram.boundingBox();
    const original = await page.locator('.readout').innerText();
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width / 2 + 15, box.y + box.height / 2, { steps: 3 });
    const middle = await page.locator('.readout').innerText();
    await page.mouse.move(box.x + box.width / 2 + 45, box.y + box.height / 2, { steps: 6 });
    const end = await page.locator('.readout').innerText();
    await page.mouse.up();
    expect(middle).not.toBe(original);
    expect(end).not.toBe(middle); // Repainting must not lose pointer capture.
    await page.getByRole('button', { name: 'Move B left', exact: true }).press('Enter');
    expect(await page.locator('.readout').innerText()).not.toBe(end);
    await range('B width', 0);
    await expect(page.locator('.readout')).toContainText('0.000');

    await select('huber');
    await page.getByRole('button', { name: 'multiply (× mask)', exact: true }).click();
    await expect(page.locator('.readout').last()).toContainText('NaN');
    await page.getByRole('button', { name: 'select (np.where)', exact: true }).click();
    await expect(page.locator('.readout').last()).not.toContainText('NaN');
    await range('δ', 3);

    await select('voxel');
    const initialVoxels = await page.locator('.readout').innerText();
    await range('voxel size', 0.5);
    expect(await page.locator('.readout').innerText()).not.toBe(initialVoxels);
    await range('voxel size', 2.5);
    await range('origin x', -1.5);

    await select('ego');
    await expect(page.locator('.kv')).toContainText('p_current = R_cᵀ·(p_world − t_c) = [5.00, 0.00]');
    await page.getByRole('button', { name: 'rotate 90°', exact: true }).click();
    await expect(page.locator('.kv')).toContainText('p_current = R_cᵀ·(p_world − t_c) = [0.00, 1.00]');
    await range('history x', 16);
    await range('history y', 6);
    await range('point y', 3);
    const pointFits = await page.getByRole('img', { name: 'world with two ego frames' }).evaluate(svg => {
      const v = svg.viewBox.baseVal, p = svg.querySelector('circle');
      return p.cx.baseVal.value >= v.x && p.cx.baseVal.value <= v.x + v.width && p.cy.baseVal.value >= v.y && p.cy.baseVal.value <= v.y + v.height;
    });
    expect(pointFits).toBe(true);
    const theme = await page.locator('html').getAttribute('data-theme');
    await page.reload();
    await expect(page.locator('html')).toHaveAttribute('data-theme', theme);
    expect(errors).toEqual([]);
    console.log('Fundamentals passed: 64 theme/viewport views, keyboard navigation, seven interactive widgets, continuous drag, and theme persistence.');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
