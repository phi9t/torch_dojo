import { defineConfig } from '@playwright/test';

const port = Number(process.env.DOJO_TEST_PORT ?? 8000);
// Set DOJO_PREFIX (for example /torch_dojo) to test a site already served under a Pages prefix.
const prefix = process.env.DOJO_PREFIX ?? '';

export default defineConfig({
  testDir: 'tests/e2e',
  timeout: 180_000,
  workers: 1,
  use: {
    // Specs navigate relative to this, so they also run under a Pages prefix.
    baseURL: `http://localhost:${port}${prefix}/`,
    browserName: 'chromium',
    launchOptions: { executablePath: '/usr/bin/google-chrome', args: ['--no-sandbox'] }
  },
  webServer: {
    command: `../.venv/bin/python -m http.server ${port} --directory ../dist`,
    url: `http://localhost:${port}${prefix}/lab/`,
    reuseExistingServer: true,
    timeout: 30_000
  }
});
