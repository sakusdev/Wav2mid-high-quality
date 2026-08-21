import { readFile } from 'node:fs/promises';
import { test, expect } from '@playwright/test';

const coreUrl = new URL('../src/muscriptor-browser-core.js', import.meta.url);

test('MuScriptor exposes throttled per-token decoder progress', async () => {
  const source = await readFile(coreUrl, 'utf8');

  expect(source).toContain('TOKEN_PROGRESS_EVERY = 16');
  expect(source).toContain('TOKEN_PROGRESS_MAX_SILENCE_MS = 500');
  expect(source).toContain('tok ${progress.generated}/${progress.max}');
  expect(source).toContain('tok/s');
  expect(source).toContain('elapsed.toFixed(1)');
});

test('MuScriptor treats the 2000-token no-EOS boundary like upstream', async () => {
  const source = await readFile(coreUrl, 'utf8');

  expect(source).toContain('upstream no_eos_is_ok=True');
  expect(source).toContain('capped: true');
  expect(source).toContain('tok cap · continuing');
  expect(source).toContain('cappedChunks');
  expect(source).not.toContain('MuScriptor chunk did not emit EOS within ${MUSCRIPTOR_MAX_GENERATION} tokens.');
});

test('MuScriptor stops immediately after the final allowed token', async () => {
  const source = await readFile(coreUrl, 'utf8');
  const capCheck = source.indexOf('if (generatedCount === MUSCRIPTOR_MAX_GENERATION)');
  const nextDecoder = source.indexOf('step = await this.runDecoder(emptyPrefix', capCheck);

  expect(capCheck).toBeGreaterThan(-1);
  expect(nextDecoder).toBeGreaterThan(capCheck);
});
