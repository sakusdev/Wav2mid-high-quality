import { test, expect } from '@playwright/test';
import fs from 'node:fs';

function makeFixtureWav(path) {
  const sampleRate = 22050;
  const duration = 3;
  const totalSamples = sampleRate * duration;
  const dataBytes = totalSamples * 2;
  const buf = Buffer.alloc(44 + dataBytes);
  let o = 0;
  const write = s => { buf.write(s, o, 'ascii'); o += s.length; };
  const u32 = n => { buf.writeUInt32LE(n, o); o += 4; };
  const u16 = n => { buf.writeUInt16LE(n, o); o += 2; };
  write('RIFF'); u32(36 + dataBytes); write('WAVE');
  write('fmt '); u32(16); u16(1); u16(1); u32(sampleRate); u32(sampleRate * 2); u16(2); u16(16);
  write('data'); u32(dataBytes);

  const events = [
    [0.15, 0.95, [60]],
    [1.05, 1.85, [64]],
    [1.95, 2.85, [60, 64, 67]],
  ];
  for (let i = 0; i < totalSamples; i += 1) {
    const t = i / sampleRate;
    let x = 0;
    for (const [start, end, pitches] of events) {
      if (t < start || t >= end) continue;
      const local = t - start;
      const envelope = Math.min(1, local / 0.012) * Math.min(1, (end - t) / 0.06) * (0.8 + 0.2 * Math.exp(-1.1 * local));
      for (const midi of pitches) {
        const f = 440 * 2 ** ((midi - 69) / 12);
        x += envelope * (Math.sin(2 * Math.PI * f * t) + 0.35 * Math.sin(4 * Math.PI * f * t) + 0.15 * Math.sin(6 * Math.PI * f * t)) / pitches.length;
      }
    }
    buf.writeInt16LE(Math.max(-32767, Math.min(32767, Math.round(x * 0.3 * 32767))), 44 + i * 2);
  }
  fs.writeFileSync(path, buf);
}

test('real browser transcribes audio and downloads valid MIDI', async ({ page }, testInfo) => {
  test.setTimeout(180_000);
  const wavPath = testInfo.outputPath('known-notes.wav');
  makeFixtureWav(wavPath);
  const browserErrors = [];
  page.on('pageerror', error => browserErrors.push(String(error)));

  await page.goto('/');
  await expect(page).toHaveTitle(/Wav2mid HQ/);
  expect(await page.evaluate(() => crossOriginIsolated)).toBe(true);
  await expect(page.locator('#backendLabel')).not.toContainText('unavailable');
  await page.locator('#fileInput').setInputFiles(wavPath);
  await expect(page.locator('#analyzeBtn')).toBeEnabled();
  await expect(page.locator('#fileInfo')).toContainText('0:03');

  // Prove the portable WASM backend initializes under the same production-style isolation headers.
  await page.locator('#backendSelect').selectOption('wasm');
  await expect(page.locator('#backendLabel')).toContainText('WASM');

  // Use the normal Auto path for the full E2E inference; on Chromium this selects the fastest available backend.
  await page.locator('#backendSelect').selectOption('auto');
  await expect(page.locator('#backendLabel')).not.toContainText('loading');
  await page.locator('#analyzeBtn').click();
  try {
    await expect(page.locator('#results')).toBeVisible({ timeout: 120_000 });
  } catch (error) {
    console.log('progress:', await page.locator('#progressText').innerText(), await page.locator('#progressPct').innerText());
    console.log('hint:', await page.locator('#progressHint').innerText());
    console.log('backend:', await page.locator('#backendLabel').innerText());
    throw error;
  }

  const noteCount = Number((await page.locator('#statNotes').innerText()).replaceAll(',', ''));
  expect(noteCount).toBeGreaterThan(0);
  await expect(page.locator('#statRange')).not.toHaveText('—');

  const downloadPromise = page.waitForEvent('download');
  await page.locator('#midiBtn').click();
  const download = await downloadPromise;
  const midiPath = testInfo.outputPath('result.mid');
  await download.saveAs(midiPath);
  const midi = fs.readFileSync(midiPath);
  expect(midi.length).toBeGreaterThan(32);
  expect(midi.subarray(0, 4).toString('ascii')).toBe('MThd');
  expect(browserErrors).toEqual([]);
});
