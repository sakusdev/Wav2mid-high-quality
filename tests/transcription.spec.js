import { test, expect } from '@playwright/test';
import fs from 'node:fs';

function makeFixtureWav(path) {
  const sampleRate = 44100;
  const channels = 2;
  const duration = 4;
  const totalSamples = sampleRate * duration;
  const dataBytes = totalSamples * channels * 2;
  const buf = Buffer.alloc(44 + dataBytes);
  let o = 0;
  const write = s => { buf.write(s, o, 'ascii'); o += s.length; };
  const u32 = n => { buf.writeUInt32LE(n, o); o += 4; };
  const u16 = n => { buf.writeUInt16LE(n, o); o += 2; };
  write('RIFF'); u32(36 + dataBytes); write('WAVE');
  write('fmt '); u32(16); u16(1); u16(channels); u32(sampleRate); u32(sampleRate * channels * 2); u16(channels * 2); u16(16);
  write('data'); u32(dataBytes);

  const tonalEvents = [
    [0.10, 0.95, [48, 60]],
    [1.05, 1.90, [52, 64]],
    [2.00, 2.85, [55, 67]],
    [2.95, 3.90, [48, 55, 60, 64, 67]],
  ];
  const kickTimes = [0.45, 2.35, 3.45];
  const snareTimes = [1.35, 2.85];
  const hatTimes = [0.9, 1.8, 2.7, 3.6];

  const transient = (t, times, kind) => {
    let value = 0;
    for (const hit of times) {
      const local = t - hit;
      if (local < 0 || local > 0.16) continue;
      if (kind === 'kick') value += Math.exp(-22 * local) * Math.sin(2 * Math.PI * (72 - local * 150) * local);
      if (kind === 'snare') value += Math.exp(-28 * local) * (Math.sin(2 * Math.PI * 1650 * t) + 0.7 * Math.sin(2 * Math.PI * 2870 * t));
      if (kind === 'hat') value += Math.exp(-55 * local) * (Math.sin(2 * Math.PI * 5900 * t) + 0.5 * Math.sin(2 * Math.PI * 7600 * t));
    }
    return value;
  };

  for (let i = 0; i < totalSamples; i += 1) {
    const t = i / sampleRate;
    let tonal = 0;
    for (const [start, end, pitches] of tonalEvents) {
      if (t < start || t >= end) continue;
      const local = t - start;
      const envelope = Math.min(1, local / 0.012) * Math.min(1, (end - t) / 0.07) * (0.82 + 0.18 * Math.exp(-1.2 * local));
      for (const midi of pitches) {
        const f = 440 * 2 ** ((midi - 69) / 12);
        tonal += envelope * (Math.sin(2 * Math.PI * f * t) + 0.28 * Math.sin(4 * Math.PI * f * t) + 0.10 * Math.sin(6 * Math.PI * f * t)) / pitches.length;
      }
    }
    const drums = transient(t, kickTimes, 'kick') * 0.75 + transient(t, snareTimes, 'snare') * 0.20 + transient(t, hatTimes, 'hat') * 0.12;
    const left = Math.max(-1, Math.min(1, tonal * 0.28 + drums * 0.34));
    const right = Math.max(-1, Math.min(1, tonal * 0.25 + drums * 0.30));
    const byte = 44 + i * 4;
    buf.writeInt16LE(Math.round(left * 32767), byte);
    buf.writeInt16LE(Math.round(right * 32767), byte + 2);
  }
  fs.writeFileSync(path, buf);
}

function assertThreeTrackMidi(midiBytes) {
  expect(midiBytes.length).toBeGreaterThan(64);
  expect(midiBytes.subarray(0, 4).toString('ascii')).toBe('MThd');
  expect(midiBytes.readUInt16BE(8)).toBe(1); // SMF format 1
  expect(midiBytes.readUInt16BE(10)).toBe(3); // Harmony / Bass / Drums
  const binary = midiBytes.toString('latin1');
  expect(binary).toContain('Wav2mid HQ · Harmony');
  expect(binary).toContain('Wav2mid HQ · Bass');
  expect(binary).toContain('Wav2mid HQ · Drums');
  expect([...midiBytes].some(byte => byte === 0x99)).toBe(true); // note-on, MIDI channel 10
}

test('PRO pipeline separates, ensembles, transcribes drums and exports multi-track MIDI', async ({ page }, testInfo) => {
  test.setTimeout(300_000);
  const wavPath = testInfo.outputPath('polyphonic-drums-44k-stereo.wav');
  makeFixtureWav(wavPath);
  const browserErrors = [];
  page.on('pageerror', error => browserErrors.push(String(error)));

  await page.goto('/');
  await expect(page).toHaveTitle(/Wav2mid HQ/);
  expect(await page.evaluate(() => crossOriginIsolated)).toBe(true);
  await expect(page.locator('#backendLabel')).not.toContainText('unavailable');
  await page.locator('#fileInput').setInputFiles(wavPath);
  await expect(page.locator('#analyzeBtn')).toBeEnabled();
  await expect(page.locator('#fileInfo')).toContainText('0:04');
  await expect(page.locator('#fileInfo')).toContainText('2ch');

  // Portable fallback must still initialize.
  await page.locator('#backendSelect').selectOption('wasm');
  await expect(page.locator('#backendLabel')).toContainText('WASM');

  // WebGPU is exposed when supported; Auto handles backend fallback if a model op is unavailable.
  await page.locator('#backendSelect').selectOption('auto');
  await expect(page.locator('#backendLabel')).not.toContainText('loading');
  await page.locator('[data-mode="pro"]').click();
  await page.locator('#analyzeBtn').click();
  try {
    await expect(page.locator('#results')).toBeVisible({ timeout: 260_000 });
  } catch (error) {
    console.log('progress:', await page.locator('#progressText').innerText(), await page.locator('#progressPct').innerText());
    console.log('hint:', await page.locator('#progressHint').innerText());
    console.log('backend:', await page.locator('#backendLabel').innerText());
    throw error;
  }

  const noteCount = Number((await page.locator('#statNotes').innerText()).replaceAll(',', ''));
  const drumCount = Number((await page.locator('#statDrums').innerText()).replaceAll(',', ''));
  expect(noteCount).toBeGreaterThan(0);
  expect(drumCount).toBeGreaterThan(0);
  await expect(page.locator('#statRange')).not.toHaveText('—');
  await expect(page.locator('#pipelineList')).toContainText('harmonic');
  await expect(page.locator('#pipelineList')).toContainText('confidence ensemble');
  await expect(page.locator('#pipelineList')).toContainText('drum onset classifier');

  const midiDownloadPromise = page.waitForEvent('download');
  await page.locator('#midiBtn').click();
  const midiDownload = await midiDownloadPromise;
  const midiPath = testInfo.outputPath('result.mid');
  await midiDownload.saveAs(midiPath);
  assertThreeTrackMidi(fs.readFileSync(midiPath));

  const jsonDownloadPromise = page.waitForEvent('download');
  await page.locator('#jsonBtn').click();
  const jsonDownload = await jsonDownloadPromise;
  const jsonPath = testInfo.outputPath('result.json');
  await jsonDownload.saveAs(jsonPath);
  const analysis = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));
  expect(analysis.format).toBe('wav2mid-hq/v2');
  expect(analysis.pipeline.ensemble).toBe(true);
  expect(analysis.pipeline.contextDecoder).toBe(true);
  expect(analysis.pipeline.stemPasses).toEqual(expect.arrayContaining(['mix', 'harmonic', 'bass']));
  expect(analysis.notes.length).toBeGreaterThan(0);
  expect(analysis.drums.length).toBeGreaterThan(0);
  expect(browserErrors).toEqual([]);
});
