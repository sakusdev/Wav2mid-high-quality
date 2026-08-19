import { test, expect } from '@playwright/test';

test('NEURAL HQ is lazy, visible and does not fetch neural runtime/model until analysis starts', async ({ page }) => {
  const neuralRequests = [];
  page.on('request', request => {
    if (/htdemucs|huggingface|onnxruntime-web|cdn\.jsdelivr\.net/i.test(request.url())) neuralRequests.push(request.url());
  });

  await page.goto('/');
  const toggle = page.locator('#neuralToggle');
  const option = page.locator('#neuralOption');
  await expect(toggle).toHaveCount(1);
  await expect(option).toContainText('NEURAL HQ · HTDemucs');
  await expect(option).toContainText('~172 MB');

  // The checkbox is intentionally visually hidden; real users click the label card.
  await option.click();
  await expect(toggle).toBeChecked();
  await expect(option).toHaveClass(/active/);
  await expect(page.locator('#qualityGroup')).toHaveClass(/disabled/);
  await expect(page.locator('#modeNote')).toContainText('HTDemucs 4-stem neural');
  expect(neuralRequests).toEqual([]);

  await option.click();
  await expect(toggle).not.toBeChecked();
  await expect(page.locator('#qualityGroup')).not.toHaveClass(/disabled/);
  expect(neuralRequests).toEqual([]);
});
