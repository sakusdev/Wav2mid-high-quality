import { test, expect } from '@playwright/test';

// Production preview does not expose /src modules directly. Verify the rendered
// NEURAL metadata and make sure viewing the page still does not fetch HTDemucs.
test('NEURAL HQ exposes low-memory streaming profile without eager model fetch', async ({ page }) => {
  const requests = [];
  page.on('request', request => requests.push(request.url()));
  await page.goto('http://127.0.0.1:4173');
  await expect(page.locator('#neuralOption')).toContainText('NEURAL HQ');
  await expect(page.locator('#neuralOption')).toContainText('chunks');
  expect(requests.some(url => /htdemucs|\.onnx(?:\?|$)/i.test(url))).toBeFalsy();
});

test('Android NEURAL HQ stays below one Demucs internal segment', async ({ browser }) => {
  const context = await browser.newContext({
    userAgent: 'Mozilla/5.0 (Linux; Android 17; Pixel 7a) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/151.0 Mobile Safari/537.36',
    viewport: { width: 412, height: 915 },
  });
  const page = await context.newPage();
  const requests = [];
  page.on('request', request => requests.push(request.url()));
  await page.goto('http://127.0.0.1:4173');
  await expect(page.locator('#neuralOption')).toContainText('7.5s chunks');
  expect(requests.some(url => /htdemucs|\.onnx(?:\?|$)/i.test(url))).toBeFalsy();
  await context.close();
});
