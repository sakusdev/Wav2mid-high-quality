import { test, expect } from '@playwright/test';

test('MuScriptor ULTRA is visible, NC-labelled and lazy', async ({ page }) => {
  const requests = [];
  page.on('request', request => requests.push(request.url()));
  await page.goto('http://127.0.0.1:4173');

  const option = page.locator('#ultraOption');
  await expect(option).toContainText('MuScriptor ULTRA');
  await expect(option).toContainText('NC');
  await expect(page.locator('#ultraConfig')).toBeHidden();
  expect(requests.some(url => /127\.0\.0\.1:8223|localhost:8223/i.test(url))).toBeFalsy();

  await option.click();
  await expect(page.locator('#ultraToggle')).toBeChecked();
  await expect(page.locator('#ultraConfig')).toBeVisible();
  await expect(page.locator('#muscriptorEndpoint')).toHaveValue('http://127.0.0.1:8223');
  expect(requests.some(url => /127\.0\.0\.1:8223|localhost:8223/i.test(url))).toBeFalsy();
});
