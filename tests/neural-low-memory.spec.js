import { test, expect } from '@playwright/test';

// This test intentionally does not load HTDemucs. It verifies that the public
// NEURAL entry point is now the low-memory streaming wrapper while preserving
// lazy model loading.
test('NEURAL HQ exposes low-memory streaming profile without eager model fetch', async ({ page }) => {
  const requests = [];
  page.on('request', request => requests.push(request.url()));
  await page.goto('http://127.0.0.1:4173');
  const info = await page.evaluate(async () => {
    const neural = await import('/src/neural-transcribe.js');
    return neural.neuralModelDescription();
  });
  expect(info.name).toContain('low-memory streaming');
  expect(info.sizeHint).toContain('chunks');
  expect(requests.some(url => /htdemucs|\.onnx(?:\?|$)/i.test(url))).toBeFalsy();
});
