import { defineConfig, devices } from '@playwright/test'

export default defineConfig({
  testDir: './e2e',
  timeout: 30_000,
  expect: { timeout: 5_000 },
  fullyParallel: false,
  reporter: [['list']],
  use: {
    baseURL: process.env.PANELTONE_E2E_BASE_URL || 'http://127.0.0.1:8765',
    ...devices['Desktop Chrome'],
    trace: 'retain-on-failure',
  },
})
