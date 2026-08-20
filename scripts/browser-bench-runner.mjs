import { spawn } from 'node:child_process';
import fs from 'node:fs/promises';
import path from 'node:path';
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
    const viteBin = path.resolve('node_modules/vite/bin/vite.js');
    this.preview = spawn(process.execPath, [viteBin, 'preview', '--host', '127.0.0.1', '--port', String(this.port), '--strictPort'], {
      stdio: ['ignore', 'pipe', 'pipe'],
      env: process.env,
    });
    let previewLog = '';
    this.preview.stdout.on('data', chunk => { previewLog += chunk.toString(); });
    this.preview.stderr.on('data', chunk => { previewLog += chunk.toString(); });
    this.preview.once('exit', code => {
      if (code && code !== 0) previewLog += `\nVite preview exited with ${code}`;
    });
    await waitForHttp(this.baseUrl, 20_000).catch(error => {
      throw new Error(`Vite preview failed to start. ${error.message}\n${previewLog.slice(-3000)}`);
    });
    this.browser = await chromium.launch({ channel: 'chrome', headless: true });
  }

  async stop() {
    await this.browser?.close().catch(() => {});
    this.browser = null;
    const preview = this.preview;
    this.preview = null;
    if (!preview || preview.exitCode != null) return;
    const exited = new Promise(resolve => preview.once('exit', resolve));
    preview.kill('SIGTERM');
    const graceful = await Promise.race([
      exited.then(() => true),
      new Promise(resolve => setTimeout(() => resolve(false), 3000)),
    ]);
    if (!graceful && preview.exitCode == null) {
      preview.kill('SIGKILL');
      await Promise.race([exited, new Promise(resolve => setTimeout(resolve, 1000))]);
    }
  }

  async run(audioPath, config = {}) {
    if (!this.browser) throw new Error('BrowserBenchRunner.start() must be called first.');
    const page = await this.browser.newPage({ acceptDownloads: true });
    const timeoutMs = Number(config.timeoutMs ?? 10 * 60_000);
    page.setDefaultTimeout(timeoutMs);

    try {
      await page.goto(this.baseUrl, { waitUntil: 'domcontentloaded' });
      await page.waitForFunction(() => {
        const text = document.querySelector('#backendLabel')?.textContent ?? '';
        return text.includes('backend:') && !text.includes('loading');
      });

      const mode = String(config.mode ?? 'pro').toLowerCase();
      if (['fast', 'pro', 'insane'].includes(mode)) await page.locator(`button[data-mode="${mode}"]`).click();
      if (config.backend && config.backend !== 'auto') {
        await page.locator('#backendSelect').selectOption(config.backend);
        await page.waitForFunction(expected => (document.querySelector('#backendLabel')?.textContent ?? '').toLowerCase().includes(expected), String(config.backend).toLowerCase());
      }
      if (config.neural) await page.locator('#neuralOption').click();
      if (config.sensitivity != null) {
        await page.locator('#sensitivity').evaluate((element, value) => {
          element.value = String(value);
          element.dispatchEvent(new Event('input', { bubbles: true }));
        }, Number(config.sensitivity));
      }
      if (config.minPitch != null) await page.locator('#minPitch').fill(String(config.minPitch));
      if (config.maxPitch != null) await page.locator('#maxPitch').fill(String(config.maxPitch));

      await page.locator('#fileInput').setInputFiles(audioPath);
      await page.waitForFunction(() => !document.querySelector('#analyzeBtn')?.disabled);
      await page.locator('#analyzeBtn').click();
      await page.waitForFunction(() => {
        const results = document.querySelector('#results');
        const progress = document.querySelector('#progressText')?.textContent ?? '';
        return (results && !results.hidden) || progress.startsWith('解析失敗:');
      }, null, { timeout: timeoutMs });

      const failure = await page.locator('#progressText').textContent();
      if (failure?.startsWith('解析失敗:')) throw new Error(failure);

      const [download] = await Promise.all([
        page.waitForEvent('download'),
        page.locator('#jsonBtn').click(),
      ]);
      const downloadPath = await download.path();
      if (!downloadPath) throw new Error('Browser did not expose the analysis JSON download path.');
      return JSON.parse(await fs.readFile(downloadPath, 'utf8'));
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
