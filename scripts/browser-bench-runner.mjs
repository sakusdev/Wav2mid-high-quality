import { spawn } from 'node:child_process';
import process from 'node:process';
import { chromium } from '@playwright/test';

const DEFAULT_PORT = 4173;

export class BrowserBenchRunner {
  constructor(options = {}) {
    this.port = Number(options.port ?? DEFAULT_PORT);
    this.baseUrl = `http://127.0.0.1:${this.port}`;
    this.preview = null;
    this.browser = null;
  }

  async start({ build = true } = {}) {
    if (build) await runCommand(process.platform === 'win32' ? 'npm.cmd' : 'npm', ['run', 'build']);
    this.preview = spawn(process.platform === 'win32' ? 'npm.cmd' : 'npm', ['run', 'preview', '--', '--host', '127.0.0.1', '--port', String(this.port)], {
      stdio: ['ignore', 'pipe', 'pipe'],
      env: process.env,
    });
    let previewLog = '';
    this.preview.stdout.on('data', chunk => { previewLog += chunk.toString(); });
    this.preview.stderr.on('data', chunk => { previewLog += chunk.toString(); });
    await waitForHttp(this.baseUrl, 20_000).catch(error => {
      throw new Error(`Vite preview failed to start. ${error.message}\n${previewLog.slice(-3000)}`);
    });
    this.browser = await chromium.launch({ channel: 'chrome', headless: true });
  }

  async stop() {
    await this.browser?.close().catch(() => {});
    this.browser = null;
    if (this.preview && !this.preview.killed) this.preview.kill('SIGTERM');
    this.preview = null;
  }

  async run(audioPath, config = {}) {
    if (!this.browser) throw new Error('BrowserBenchRunner.start() must be called first.');
    const page = await this.browser.newPage();
    const timeoutMs = Number(config.timeoutMs ?? 10 * 60_000);
    page.setDefaultTimeout(timeoutMs);
    await page.addInitScript(() => { window.__WAV2MID_BENCHMARK__ = true; });

    try {
      await page.goto(this.baseUrl, { waitUntil: 'domcontentloaded' });
      await page.waitForFunction(() => document.querySelector('#backendLabel')?.textContent?.includes('backend:') && !document.querySelector('#backendLabel')?.textContent?.includes('loading'));

      const mode = String(config.mode ?? 'pro').toLowerCase();
      if (['fast', 'pro', 'insane'].includes(mode)) await page.locator(`button[data-mode="${mode}"]`).click();
      if (config.backend && config.backend !== 'auto') await page.locator('#backendSelect').selectOption(config.backend);
      if (config.neural) await page.locator('#neuralOption').click();
      if (config.sensitivity != null) {
        await page.locator('#sensitivity').evaluate((element, value) => {
          element.value = String(value);
          element.dispatchEvent(new Event('input', { bubbles: true }));
        }, Number(config.sensitivity));
      }
      if (config.minPitch != null) await page.locator('#minPitch').fill(String(config.minPitch));
      if (config.maxPitch != null) await page.locator('#maxPitch').fill(String(config.maxPitch));

      await page.evaluate(advanced => {
        window.__WAV2MID_BENCH_OPTIONS__ = advanced ?? {};
        window.__WAV2MID_LAST_RESULT__ = null;
        window.__WAV2MID_LAST_ERROR__ = null;
      }, config.advanced ?? {});

      await page.locator('#fileInput').setInputFiles(audioPath);
      await page.locator('#analyzeBtn').waitFor({ state: 'visible' });
      await page.waitForFunction(() => !document.querySelector('#analyzeBtn')?.disabled);
      await page.locator('#analyzeBtn').click();
      await page.waitForFunction(() => window.__WAV2MID_LAST_RESULT__ || window.__WAV2MID_LAST_ERROR__, null, { timeout: timeoutMs });

      const state = await page.evaluate(() => ({
        result: window.__WAV2MID_LAST_RESULT__ ?? null,
        error: window.__WAV2MID_LAST_ERROR__ ?? null,
      }));
      if (state.error) throw new Error(`Browser transcription failed: ${state.error}`);
      return state.result;
    } finally {
      await page.close();
    }
  }
}

async function waitForHttp(url, timeoutMs) {
  const start = Date.now();
  let lastError;
  while (Date.now() - start < timeoutMs) {
    try {
      const response = await fetch(url);
      if (response.ok) return;
    } catch (error) {
      lastError = error;
    }
    await new Promise(resolve => setTimeout(resolve, 200));
  }
  throw lastError ?? new Error(`Timed out waiting for ${url}`);
}

export function runCommand(command, args, options = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      stdio: options.capture ? ['ignore', 'pipe', 'pipe'] : 'inherit',
      env: options.env ?? process.env,
      cwd: options.cwd ?? process.cwd(),
      shell: false,
    });
    let stdout = '';
    let stderr = '';
    if (options.capture) {
      child.stdout.on('data', chunk => { stdout += chunk.toString(); });
      child.stderr.on('data', chunk => { stderr += chunk.toString(); });
    }
    child.on('error', reject);
    child.on('close', code => {
      if (code === 0) resolve({ stdout, stderr });
      else reject(new Error(`${command} exited with ${code}${stderr ? `\n${stderr}` : ''}`));
    });
  });
}
