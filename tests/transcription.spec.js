import { test, expect } from '@playwright/test';
import fs from 'node:fs';

function makeFixtureWav(path) {
  const sampleRate = 22050;
  const duration = 6;
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
    [0.25, 1.65, [60]],
    [1.75, 3.15, [64]],
    [3.25, 4.65, [67]],
    [4.75, 5.85, [60, 64, 67]],
  ];
  for (let i = 0; i < totalSamples; i += 1) {
    const t = i / sampleRate;
    let x = 0;
    for (const [start, end, pitches] of events) {
      if (t < start || t >= end) continue;
      const local = t - start;
      const attack = Math.min(1, local / 0.015);
      const release = Math.min(1, (end - t) / 0.08);
      const envelope = attack * release * (0.8 + 0.2 * Math.exp(-1.2 * local));
      for (const midi of pitches) {
        const f = 440 * 2 ** ((midi - 69) / 12);
        x += envelope * (
          Math.sin(2 * Math.PI * f * t) +
          0.35 * Math.sin(2 * Math.PI * f * 2 * t) +
          0.15 * Math.sin(2 * Math.PI * f * 3 * t)
        ) / pitches.length;
      }
    }
    const pcm = Math.max(-32767, Math.min(32767, Math.round(x * 0.28 * 32767)));
    buf.writeInt16LE(pcm, 44 + i * 2);
  }
  fs.writeFileSync(path, buf);
}

test('real browser transcribes audio and downloads a valid MIDI file', async ({ page }, testInfo) => {
  const wavPath = testInfo.outputPath('known-notes.wav');
  makeFixtureWav(wavPath);

  const browserErrors = [];
  page.on('pageerror', error => browserErrors.push(String(error)));

  await page.goto('/');
  await expect(page).toHaveTitle(/Wav2mid HQ/);
  await expect(page.locator('#backendLabel')).not.toContainText('unavailable');

  await page.locator('#fileInput').setInputFiles(wavPath);
  await expect(page.locator('#analyzeBtn')).toBeEnabled();
  await expect(page.locator('#fileInfo')).toContainText('0:06');

  await page.locator('#backendSelect').selectOption('wasm');
  await expect(page.locator('#backendLabel')).toContainText('WASM');
  await page.locator('#qualityGroup button[data-mode="pro"]').click();

  await page.locator('#analyzeBtn').click();
  await expect(page.locator('#results')).toBeVisible({ timeout: 90_000 });

  const noteCount = Number((await page.locator('#statNotes').innerText()).replaceAll(',', ''));
  expect(noteCount).toBeGreaterThan(0);
  await expect(page.locator('#statRange')).not.toHaveText('—');
  await expect(page.locator('#cleanupSummary')).toContainText('WASM');

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
