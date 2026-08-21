import { test, expect } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';

const JSEP_GLUE = 'ort-wasm-simd-threaded.jsep.mjs';

test('ORT JSEP worker glue is staged on the app origin', async ({ request }) => {
  const staged = path.join(process.cwd(), 'public', 'ort-wasm', JSEP_GLUE);
  expect(fs.existsSync(staged)).toBeTruthy();
  expect(fs.statSync(staged).size).toBeGreaterThan(1_000);

  const response = await request.get(`/ort-wasm/${JSEP_GLUE}`);
  expect(response.ok()).toBeTruthy();
  expect((await response.body()).byteLength).toBeGreaterThan(1_000);
});

test('ORT shim keeps only the oversized WASM payload on jsDelivr', async () => {
  const shim = fs.readFileSync(path.join(process.cwd(), 'src', 'ort-cdn-shim.js'), 'utf8');
  expect(shim).toContain("mjs: new URL(ORT_JSEP_MJS, location.origin).href");
  expect(shim).toContain('wasm: ORT_JSEP_WASM');
  expect(shim).toContain('ort-wasm-simd-threaded.jsep.wasm');

  const runtimePrep = fs.readFileSync(path.join(process.cwd(), 'scripts', 'copy-runtime.mjs'), 'utf8');
  expect(runtimePrep).toContain("const ortJsepGlue = 'ort-wasm-simd-threaded.jsep.mjs'");
  expect(runtimePrep).not.toContain("copyFile(join(ortDistSrc, 'ort-wasm-simd-threaded.jsep.wasm')");
});
