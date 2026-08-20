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
