import { test, expect } from '@playwright/test';

test('MuScriptor diagnostics stay lazy until the loader is called', async ({ page }) => {
  const requests = [];
  page.on('request', request => requests.push(request.url()));
  await page.goto('http://127.0.0.1:4173');

  expect(await page.evaluate(() => typeof globalThis.__WAV2MID_MUSCRIPTOR_MODEL_LOADER__)).toBe('function');
  expect(requests.some(url => /muscriptor-small|ort\.all\.min\.js|ort-wasm/i.test(url))).toBeFalsy();
});

test('MuScriptor reports a missing deployed manifest with a stable error code', async ({ page }) => {
  await page.route('**/models/muscriptor-small/manifest.json', async route => {
    await route.fulfill({ status: 404, contentType: 'application/json', body: '{}' });
  });
  await page.goto('http://127.0.0.1:4173');

  const error = await page.evaluate(async () => {
    try {
      await globalThis.__WAV2MID_MUSCRIPTOR_MODEL_LOADER__();
      return null;
    } catch (caught) {
      return {
        name: caught?.name,
        code: caught?.code,
        stage: caught?.stage,
        message: caught?.message,
        hint: caught?.hint,
        trace: globalThis.__WAV2MID_MUSCRIPTOR_DIAGNOSTICS__,
      };
    }
  });

  expect(error?.name).toBe('MuScriptorDiagnosticError');
  expect(error?.code).toBe('MUSCRIPTOR_MANIFEST_HTTP');
  expect(error?.stage).toBe('manifest');
  expect(error?.message).toContain('HTTP 404');
  expect(error?.hint).toContain('モデルがデプロイされていません');
  expect(error?.trace?.at(-1)?.status).toBe('error');
});

test('MuScriptor distinguishes conditioner asset delivery failure', async ({ page }) => {
  await page.route('**/models/muscriptor-small/manifest.json', async route => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        format: 'wav2mid-muscriptor-browser/v1',
        architecture: { layers: 4, heads: 4, dim: 256, maxCache: 1024 },
        files: {
          conditioner: { url: '/models/muscriptor-small/conditioner.onnx', bytes: 1234 },
          decoder: { url: '/models/muscriptor-small/decoder.onnx', bytes: 5678 },
        },
      }),
    });
  });
  await page.route('**/models/muscriptor-small/conditioner.onnx', async route => {
    await route.fulfill({ status: 404, body: '' });
  });
  await page.goto('http://127.0.0.1:4173');

  const error = await page.evaluate(async () => {
    try {
      await globalThis.__WAV2MID_MUSCRIPTOR_MODEL_LOADER__();
      return null;
    } catch (caught) {
      return { code: caught?.code, stage: caught?.stage, message: caught?.message };
    }
  });

  expect(error?.code).toBe('MUSCRIPTOR_ASSET_HTTP');
  expect(error?.stage).toBe('conditioner-asset');
  expect(error?.message).toContain('conditioner');
  expect(error?.message).toContain('HTTP 404');
});
