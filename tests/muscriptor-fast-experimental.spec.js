import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { test, expect } from '@playwright/test';

const root = process.cwd();
const fastSourcePath = join(root, 'src', 'muscriptor-fast-experimental.js');
const coreSourcePath = join(root, 'src', 'muscriptor-browser-core.js');
const ortSourcePath = join(root, 'src', 'ort-cdn-shim.js');

test('MuScriptor exposes an opt-in FAST experimental toggle', async ({ page }) => {
  await page.goto('http://127.0.0.1:4173');

  await expect(page.locator('#muscriptorFastToggle')).toHaveCount(1);
  expect(await page.locator('#muscriptorFastToggle').isChecked()).toBe(false);
  expect(await page.evaluate(() => globalThis.__WAV2MID_MUSCRIPTOR_FAST_STATE__?.enabled)).toBe(false);

  await page.locator('#muscriptorFastToggle').check({ force: true });
  expect(await page.evaluate(() => globalThis.__WAV2MID_MUSCRIPTOR_FAST_STATE__?.enabled)).toBe(true);
  expect(await page.locator('#muscriptorFastState').textContent()).toContain('ON');
});

test('FAST profile limits autoregressive work without changing normal MuScriptor', async () => {
  const [fastSource, coreSource] = await Promise.all([
    readFile(fastSourcePath, 'utf8'),
    readFile(coreSourcePath, 'utf8'),
  ]);

  expect(coreSource).toContain('export const MUSCRIPTOR_MAX_GENERATION = 2000;');
  expect(fastSource).toContain('const FAST_MAX_GENERATION = 512;');
  expect(fastSource).toContain('const FAST_CHUNK_TICKS = 500;');
  expect(fastSource).toContain('if (!state.enabled) return normalGenerateChunk.apply(this, args);');
  expect(fastSource).toContain('shift >= FAST_CHUNK_TICKS');
  expect(fastSource).toContain('generated < FAST_MAX_GENERATION');
});

test('FAST profile tunes the existing SIMD-threaded ORT WASM/JSEP runtime', async () => {
  const [fastSource, ortSource] = await Promise.all([
    readFile(fastSourcePath, 'utf8'),
    readFile(ortSourcePath, 'utf8'),
  ]);

  expect(fastSource).toContain("configureOrtRuntimeProfile?.(state.enabled ? 'fast' : 'default')");
  expect(ortSource).toContain("artifact: 'simd-threaded-jsep'");
  expect(ortSource).toContain('wasm.numThreads = requestedThreads');
  expect(ortSource).toContain('Math.min(4, Math.floor(cores / 2))');
  expect(ortSource).toContain('wasm.proxy = false');
  expect(ortSource).toContain('crossOriginIsolated');
});
