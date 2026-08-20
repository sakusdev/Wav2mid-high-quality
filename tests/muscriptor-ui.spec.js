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

test('MuScriptor ULTRA is visible, NC-labelled and browser-lazy', async ({ page }) => {
  const requests = [];
  page.on('request', request => requests.push(request.url()));
  await page.goto('http://127.0.0.1:4173');

  const option = page.locator('#ultraOption');
  await expect(option).toContainText('MuScriptor ULTRA');
  await expect(option).toContainText('NC');
  await expect(option).toContainText('Browser WebGPU');
  await expect(page.locator('#ultraState')).toHaveText('lazy');
  await expect(page.locator('#ultraConfig')).toBeHidden();
  expect(requests.some(url => /127\.0\.0\.1:8223|localhost:8223/i.test(url))).toBeFalsy();
  expect(requests.some(url => /muscriptor-small|ort\.all\.min\.js/i.test(url))).toBeFalsy();

  await option.click();
  await expect(page.locator('#ultraToggle')).toBeChecked();
  await expect(page.locator('#ultraConfig')).toBeVisible();
  await expect(page.locator('#ultraConfig')).toContainText('端末内推論');
  await expect(page.locator('#muscriptorEndpoint')).toHaveCount(0);
  expect(requests.some(url => /127\.0\.0\.1:8223|localhost:8223/i.test(url))).toBeFalsy();
  expect(requests.some(url => /muscriptor-small|ort\.all\.min\.js/i.test(url))).toBeFalsy();
});

test('MuScriptor ULTRA uses the browser model path and renders the result', async ({ page }, testInfo) => {
  const wavPath = testInfo.outputPath('muscriptor-input.wav');
  makeTinyWav(wavPath);

  await page.addInitScript(() => {
    globalThis.__WAV2MID_MUSCRIPTOR_MODEL_LOADER__ = async () => ({
      transcribe: async (_audioBuffer, _options, onProgress) => {
        onProgress?.({ stage: 'muscriptor-load', value: 0.1, detail: 'model ready' });
        onProgress?.({ stage: 'muscriptor-decode', value: 1, detail: '1/1 chunks' });
        const note = {
          pitchMidi: 60,
          startTimeSeconds: 0.05,
          durationSeconds: 0.2,
          instrument: 'acoustic_piano',
          program: 0,
          confidence: 1,
          amplitude: 0.8,
        };
        return {
          notes: [note],
          drums: [],
          rawNotes: [note],
          chunks: 1,
          model: {
            engine: 'MuScriptor small',
            architecture: { decoderActivationType: 'float32' },
          },
        };
      },
    });
  });

  const localhostRequests = [];
  page.on('request', request => {
    if (/127\.0\.0\.1:8223|localhost:8223/i.test(request.url())) localhostRequests.push(request.url());
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
  await expect(page.locator('#backendLabel')).toContainText('MUSCRIPTOR · WEBGPU');
  await expect(page.locator('#pipelineList')).toContainText('MuScriptor transformer');
  await expect(page.locator('#pipelineList')).toContainText('INT4 weight-only decoder');
  await expect(page.locator('#progressText')).toContainText('完了');
  await expect(page.locator('#ultraState')).toHaveText('ready');
  expect(localhostRequests).toEqual([]);
});
