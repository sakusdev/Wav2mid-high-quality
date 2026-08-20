import { test, expect } from '@playwright/test';
import fs from 'node:fs';

function makeTinyWav(path) {
  const sampleRate = 16000;
  const duration = 0.35;
  const samples = Math.round(sampleRate * duration);
  const dataBytes = samples * 2;
  const buffer = Buffer.alloc(44 + dataBytes);
  let offset = 0;
  const text = value => { buffer.write(value, offset, 'ascii'); offset += value.length; };
  const u16 = value => { buffer.writeUInt16LE(value, offset); offset += 2; };
  const u32 = value => { buffer.writeUInt32LE(value, offset); offset += 4; };
  text('RIFF'); u32(36 + dataBytes); text('WAVE');
  text('fmt '); u32(16); u16(1); u16(1); u32(sampleRate); u32(sampleRate * 2); u16(2); u16(16);
  text('data'); u32(dataBytes);
  for (let i = 0; i < samples; i += 1) {
    const t = i / sampleRate;
    const envelope = Math.min(1, t / 0.01) * Math.min(1, (duration - t) / 0.04);
    const sample = Math.sin(2 * Math.PI * 261.6256 * t) * envelope * 0.25;
    buffer.writeInt16LE(Math.round(sample * 32767), 44 + i * 2);
  }
  fs.writeFileSync(path, buffer);
}

function bridgeHeaders(extra = {}) {
  return {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
    'Access-Control-Allow-Headers': '*',
    'Access-Control-Allow-Private-Network': 'true',
    ...extra,
  };
}

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

test('MuScriptor ULTRA consumes bridge SSE and renders the result', async ({ page }, testInfo) => {
  const wavPath = testInfo.outputPath('muscriptor-input.wav');
  makeTinyWav(wavPath);

  const sse = [
    'data: {"type":"progress","completed":0,"total":1}',
    '',
    'data: {"type":"start","pitch":60,"start_time":0.05,"index":1,"instrument":"acoustic_piano"}',
    '',
    'data: {"type":"end","end_time":0.25,"start_event_index":1}',
    '',
    'data: {"type":"progress","completed":1,"total":1}',
    '',
    'data: {"type":"transcription_complete","data":"TVRoZA==","beat_grid":{"bpm":120,"beats_per_bar":4,"first_downbeat":0,"onset_delay":0}}',
    '',
    '',
  ].join('\n');

  await page.route('http://127.0.0.1:8223/**', async route => {
    const request = route.request();
    const url = new URL(request.url());
    if (request.method() === 'OPTIONS') {
      await route.fulfill({ status: 204, headers: bridgeHeaders() });
      return;
    }
    if (url.pathname === '/health') {
      await route.fulfill({ status: 200, contentType: 'application/json', headers: bridgeHeaders(), body: '{"status":"ok"}' });
      return;
    }
    if (url.pathname === '/transcribe') {
      await route.fulfill({
        status: 200,
        headers: bridgeHeaders({ 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache' }),
        body: sse,
      });
      return;
    }
    await route.fulfill({ status: 404, headers: bridgeHeaders() });
  });

  await page.goto('http://127.0.0.1:4173');
  await page.locator('#fileInput').setInputFiles(wavPath);
  await expect(page.locator('#analyzeBtn')).toBeEnabled();
  await page.locator('#ultraOption').click();
  await page.locator('#analyzeBtn').click();

  const results = page.locator('#results');
  await expect(results).toBeVisible();
  await expect(results).toHaveAttribute('data-engine', 'muscriptor');
  await expect(page.locator('#statNotes')).toHaveText('1');
  await expect(page.locator('#statTempo')).toHaveText('120 BPM');
  await expect(page.locator('#backendLabel')).toContainText('MUSCRIPTOR');
  await expect(page.locator('#pipelineList')).toContainText('MuScriptor transformer');
  await expect(page.locator('#progressText')).toContainText('完了');
  await expect(page.locator('#ultraState')).toHaveText('ready');
});
