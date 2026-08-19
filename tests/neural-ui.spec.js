import { test, expect } from '@playwright/test';

test('NEURAL HQ is lazy, visible and does not fetch the model until analysis starts', async ({ page }) => {
  const modelRequests = [];
  page.on('request', request => {
    if (/htdemucs|huggingface/i.test(request.url())) modelRequests.push(request.url());
  });

  await page.goto('/');
  const toggle = page.locator('#neuralToggle');
  await expect(toggle).toHaveCount(1);
  await expect(page.locator('#neuralOption')).toContainText('NEURAL HQ · HTDemucs');
  await expect(page.locator('#neuralOption')).toContainText('~172 MB');
  await toggle.check();
  await expect(page.locator('#neuralOption')).toHaveClass(/active/);
  await expect(page.locator('#qualityGroup')).toHaveClass(/disabled/);
  await expect(page.locator('#modeNote')).toContainText('HTDemucs 4-stem neural');
  expect(modelRequests).toEqual([]);

  await toggle.uncheck();
  await expect(page.locator('#qualityGroup')).not.toHaveClass(/disabled/);
  expect(modelRequests).toEqual([]);
});
