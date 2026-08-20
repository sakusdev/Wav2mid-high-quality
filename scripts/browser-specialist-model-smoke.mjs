import { spawn } from 'node:child_process';
import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { chromium } from '@playwright/test';

const args = parseArgs(process.argv.slice(2));
if (!args.model) {
  console.error('Usage: node scripts/browser-specialist-model-smoke.mjs --model /path/model.onnx [--port 4174] [--seconds 1.0]');
  process.exit(2);
}

const modelPath = path.resolve(args.model);
await fs.access(modelPath);
const port = Number(args.port ?? 4174);
const seconds = Number(args.seconds ?? 1.0);
const baseUrl = `http://127.0.0.1:${port}`;
const npm = process.platform === 'win32' ? 'npm.cmd' : 'npm';
const server = spawn(npm, ['run', 'dev', '--', '--host', '127.0.0.1', '--port', String(port)], {
  stdio: ['ignore', 'pipe', 'pipe'],
  env: process.env,
});
let serverLog = '';
server.stdout.on('data', chunk => { serverLog += chunk.toString(); });
server.stderr.on('data', chunk => { serverLog += chunk.toString(); });

let browser;
try {
  await waitForHttp(baseUrl, 30_000);
  browser = await chromium.launch({ channel: 'chrome', headless: true });
  const page = await browser.newPage();
  const pageErrors = [];
  page.on('pageerror', error => pageErrors.push(String(error)));
  await page.route('**/__specialist-smoke.onnx', route => route.fulfill({
    path: modelPath,
    contentType: 'application/octet-stream',
    headers: { 'Cross-Origin-Resource-Policy': 'same-origin' },
  }));
  await page.goto(baseUrl, { waitUntil: 'domcontentloaded' });
  const result = await page.evaluate(async ({ seconds }) => {
    const runtime = await import('/src/specialist-runtime.js');
    const sampleRate = 16000;
    const count = Math.max(1, Math.round(sampleRate * seconds));
    const samples = new Float32Array(count);
    for (let i = 0; i < count; i += 1) {
      const t = i / sampleRate;
      const attack = Math.min(1, t / 0.01);
      const decay = Math.exp(-0.7 * t);
      samples[i] = 0.3 * attack * decay * (
        Math.sin(2 * Math.PI * 440 * t) +
        0.18 * Math.sin(2 * Math.PI * 880 * t)
      );
    }
    const started = performance.now();
    const transcription = await runtime.transcribeWithSpecialist({
      sampleRate,
      samples,
      length: samples.length,
      numberOfChannels: 1,
    }, {
      name: 'browser-smoke-piano',
      instrument: 'piano',
      url: '/__specialist-smoke.onnx',
      executionProvider: 'wasm',
      classes: 88,
      beginNote: 21,
      framesPerSecond: 100,
    });
    return {
      elapsedSeconds: (performance.now() - started) / 1000,
      stats: transcription.stats,
      shapes: Object.fromEntries(Object.entries(transcription.outputs).map(([key, value]) => [key, value.dims])),
      firstNotes: transcription.notes.slice(0, 8).map(note => ({
        pitchMidi: note.pitchMidi,
        startTimeSeconds: note.startTimeSeconds,
        durationSeconds: note.durationSeconds,
        confidence: note.confidence,
      })),
      crossOriginIsolated: globalThis.crossOriginIsolated,
    };
  }, { seconds });
  await page.close();

  if (pageErrors.length) throw new Error(`Browser page errors: ${pageErrors.join('\n')}`);
  const expectedFrames = 1001;
  for (const key of ['reg_onset_output', 'reg_offset_output', 'frame_output', 'velocity_output']) {
    const dims = result.shapes[key];
    if (!dims || dims[0] < 1 || dims[1] !== 88) throw new Error(`Unexpected ${key} shape: ${JSON.stringify(dims)}`);
    if (dims[0] !== Math.floor(seconds * 100) + 1) {
      // deframe trims model output back to the real (unpadded) input duration.
      throw new Error(`Unexpected trimmed frame count for ${key}: ${dims[0]}`);
    }
  }
  if (!result.crossOriginIsolated) throw new Error('Specialist browser smoke did not run cross-origin isolated.');
  console.log(JSON.stringify(result, null, 2));
  console.log(`BROWSER SPECIALIST SMOKE PASS · ${result.elapsedSeconds.toFixed(2)} sec · ${result.stats.noteCount} note(s)`);
} finally {
  await browser?.close().catch(() => {});
  if (!server.killed) server.kill('SIGTERM');
  await Promise.race([
    new Promise(resolve => server.once('close', resolve)),
    new Promise(resolve => setTimeout(resolve, 3000)),
  ]).catch(() => {});
  if (server.exitCode && server.exitCode !== 0) console.error(serverLog.slice(-4000));
}

async function waitForHttp(url, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  let lastError;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url);
      if (response.ok) return;
    } catch (error) { lastError = error; }
    await new Promise(resolve => setTimeout(resolve, 200));
  }
  throw lastError ?? new Error(`Timed out waiting for ${url}`);
}

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (!token.startsWith('--')) continue;
    const key = token.slice(2);
    if (argv[i + 1] && !argv[i + 1].startsWith('--')) out[key] = argv[++i];
    else out[key] = true;
  }
  return out;
}
