import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  timeout: 180_000,
  expect: { timeout: 15_000 },
  reporter: [['list']],
  use: { baseURL: 'http://127.0.0.1:4173', headless: true, acceptDownloads: true },
  webServer: {
    // Source-level browser tests import /src modules directly. The CI workflow
    // already runs a production Vite build and static-asset gates before E2E,
    // so serve E2E through Vite dev mode where those source module URLs exist.
    command: 'npm run dev -- --host 127.0.0.1 --port 4173',
    url: 'http://127.0.0.1:4173',
    reuseExistingServer: false,
    timeout: 30_000,
  },
  projects: [{ name: 'chrome', use: { browserName: 'chromium', channel: 'chrome' } }],
});
