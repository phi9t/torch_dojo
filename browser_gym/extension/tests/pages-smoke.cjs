const { chromium, expect } = require('@playwright/test');
// Smoke-checks a deployed site (GitHub Pages, under its repository prefix):
// the home page, one workout's brief, an exploration widget, a passing check
// in a fresh worker, and the answer surviving a reload.
(async () => {
  const target = process.argv[2] || 'http://localhost:8152/torch_dojo/lab/';
  const browser = await chromium.launch({...(process.env.DOJO_BROWSER_EXECUTABLE ? {executablePath: process.env.DOJO_BROWSER_EXECUTABLE} : {}), args: ['--no-sandbox']});
  const page = await browser.newPage();
  const failures = [];
  page.on('pageerror', e => console.log('PAGE ERROR', e.message));
  page.setDefaultTimeout(120000);
  page.on('response', r => {
    // A fresh browser has no saved rep index yet; that lookup 404s by design.
    if (r.status() >= 400 && !(r.status() === 404 && /\/api\/contents\/attempts\/(?:0059\/)?all\.json$/.test(new URL(r.url()).pathname))) {
      failures.push(`${r.status()} ${r.url()}`);
      console.log('HTTP ERROR', r.status(), r.url());
    }
  });

  await page.goto(target);
  const home = page.getByRole('region', {name: 'Workouts', exact: true});
  await expect(home).toBeVisible({timeout: 90000});
  await expect(home.getByRole('region', {name: 'Next up'})).toBeVisible();
  await home.getByRole('region', {name: 'Training'}).getByRole('link', {name: /^0059 · /}).click();
  const check = page.getByRole('button', {name: /^Check (?!log)/});
  await expect(check).toBeEnabled({timeout: 90000});

  const problem = page.getByRole('region', {name: 'Problem', exact: true});
  await expect(problem.getByRole('heading', {name: 'Goal', exact: true})).toBeVisible();
  await expect(problem.locator('mjx-container').first()).toBeVisible();

  await page.getByRole('tab', {name: /^Explore/}).click();
  const panel = problem.locator('.jp-CodeCell').filter({visible: true}).first();
  await panel.getByRole('button', {name: 'Run', exact: true}).click();
  await expect(page.getByRole('group', {name: 'Tensor card for logits_probe'})).toContainText('(2, 2)', {timeout: 90000});
  const loss = page.getByTestId('dojo-loss-value');
  await expect(loss).toContainText('Loss at scale 1.00:', {timeout: 90000});
  const oldLoss = await loss.textContent();
  const slider = page.getByRole('slider', {name: 'Logit scale'});
  await slider.focus();
  await slider.press('ArrowRight');
  await expect(loss).not.toHaveText(oldLoss);

  const catalog = await (await page.request.get(new URL('../gym-assets/catalog.json', target).href)).json();
  const solution = catalog.find(entry => entry.id === '0059').solution;
  const answer = page.getByRole('region', {name: 'Answer', exact: true}).locator('.jp-CodeCell').filter({visible: true}).first().locator('.cm-content');
  await answer.click();
  await page.keyboard.press('ControlOrMeta+A');
  await page.keyboard.insertText(solution);
  await check.click();
  await expect(page.locator('#torch-dojo-gym [role="status"]')).toContainText('PASSED · source SHA-256', {timeout: 120000});
  await expect(page.getByRole('region', {name: 'Check results'}).getByRole('button', {name: /: passed$/}).first()).toBeVisible();

  await page.reload();
  await expect(check).toBeEnabled({timeout: 90000});
  await expect(answer).toContainText('def softmax_cross_entropy');
  await expect(page.locator('#torch-dojo-gym [role="status"]')).toContainText('INDEPENDENT PASS');
  console.log(JSON.stringify({url: target, home: 'passed', brief: 'passed', widget: 'passed', isolated_checker: 'passed', restored: 'passed', http_errors: failures}));
  await browser.close();
  if (failures.length) process.exitCode = 1;
})().catch(error => {
  console.error(error);
  process.exit(1);
});
