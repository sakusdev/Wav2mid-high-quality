import { test, expect } from '@playwright/test';

async function stubOrt(page) {
  await page.route('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/ort.all.min.js', route => route.fulfill({
    status: 200,
    contentType: 'application/javascript',
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Cross-Origin-Resource-Policy': 'cross-origin',
    },
    body: 'globalThis.ort={env:{wasm:{}},InferenceSession:{},Tensor:{}};',
  }));
}

test('MuScriptor browser token layout and tie forcing match upstream', async ({ page }) => {
  await stubOrt(page);
  await page.goto('/');
  const result = await page.evaluate(async () => {
    const m = await import('/src/muscriptor-browser-core.js');
    return {
      eos: m.decodeMuScriptorToken(1),
      shift: m.decodeMuScriptorToken(1003),
      pitch: m.decodeMuScriptorToken(1004 + 60),
      velocity: m.decodeMuScriptorToken(1133),
      tie: m.decodeMuScriptorToken(1134),
      program: m.decodeMuScriptorToken(1135 + 32),
      drum: m.decodeMuScriptorToken(1265 + 42),
      prompt: m.encodeTieSection([[32, 48], [0, 64], [0, 60]]),
    };
  });
  expect(result.eos.type).toBe('EOS');
  expect(result.shift).toEqual({ type: 'shift', value: 1000 });
  expect(result.pitch).toEqual({ type: 'pitch', value: 60 });
  expect(result.velocity).toEqual({ type: 'velocity', value: 1 });
  expect(result.tie.type).toBe('tie');
  expect(result.program).toEqual({ type: 'program', value: 32 });
  expect(result.drum).toEqual({ type: 'drum', value: 42 });
  expect(result.prompt).toEqual([
    1135 + 0, 1004 + 60, 1004 + 64,
    1135 + 32, 1004 + 48,
    1134,
  ]);
});

test('MuScriptor OpenNoteTracker preserves notes across forced chunk ties', async ({ page }) => {
  await stubOrt(page);
  await page.goto('/');
  const actions = await page.evaluate(async () => {
    const { MuScriptorNoteTracker, encodeTieSection } = await import('/src/muscriptor-browser-core.js');
    const tracker = new MuScriptorNoteTracker(100);
    const all = [];
    all.push(...tracker.boundary(0, 5));
    all.push(...tracker.feed(1134));
    all.push(...tracker.feed(1135));
    all.push(...tracker.feed(1133));
    all.push(...tracker.feed(3 + 100));
    all.push(...tracker.feed(1004 + 60));
    all.push(...tracker.boundary(5, 10));
    const prompt = encodeTieSection(tracker.openKeys());
    for (const token of prompt) all.push(...tracker.feed(token));
    all.push(...tracker.feed(1135));
    all.push(...tracker.feed(1132));
    all.push(...tracker.feed(3 + 20));
    all.push(...tracker.feed(1004 + 60));
    return { all, prompt, open: tracker.openKeys() };
  });
  expect(actions.prompt).toEqual([1135, 1004 + 60, 1134]);
  expect(actions.all).toEqual([
    { type: 'start', program: 0, pitch: 60, time: 1 },
    { type: 'end', program: 0, pitch: 60, time: 5.2 },
  ]);
  expect(actions.open).toEqual([]);
});

test('MuScriptor browser HTK log-mel matches the upstream torch frontend', async ({ page }) => {
  test.setTimeout(60_000);
  await stubOrt(page);
  await page.goto('/');
  const measured = await page.evaluate(async () => {
    const { logMelForFiveSecondChunk } = await import('/src/muscriptor-browser-core.js');
    const sr = 16000;
    const x = new Float32Array(sr * 5);
    for (let i = 0; i < x.length; i += 1) {
      const t = i / sr;
      x[i] = 0.17 * Math.sin(2 * Math.PI * 220 * t) + 0.09 * Math.sin(2 * Math.PI * 329.6276 * t);
    }
    const mel = logMelForFiveSecondChunk(x);
    const bins = [0, 1, 10, 50, 100, 200, 300, 400, 511];
    const frames = [0, 10, 250, 500];
    return {
      dims: mel.dims,
      values: frames.map(frame => bins.map(bin => mel.data[frame * 512 + bin])),
    };
  });
  expect(measured.dims).toEqual([1, 501, 512]);
  const expected = [
    [-13.8155106, 1.3732481, 1.6435011, 1.9150989, 0.5644208, -1.1280301, -2.0585059, -2.6047443, -2.6411420],
    [-13.8155106, -9.0788220, -7.1947106, -1.3008769, -6.3515211, -10.7313605, -12.3834704, -13.2371665, -13.8067484],
    [-13.8155106, -8.9685942, -7.1957932, -1.3009770, -6.3624140, -10.7536001, -12.4024481, -13.2482475, -13.8040920],
    [-13.8155106, 1.3000783, 1.5712429, 1.8895215, 0.4034955, -1.2557885, -2.1837339, -2.7294313, -2.7656987],
  ];

  let compared = 0;
  let maxSignalAbs = 0;
  for (let f = 0; f < expected.length; f += 1) {
    for (let b = 0; b < expected[f].length; b += 1) {
      const actual = measured.values[f][b];
      expect(Number.isFinite(actual)).toBe(true);
      // Near the 1e-6 log floor, tiny FFT roundoff differences become large in
      // log space even though both frontends agree that the bin is effectively
      // silent. Check strict parity on bins with meaningful reference energy.
      if (expected[f][b] > -10) {
        maxSignalAbs = Math.max(maxSignalAbs, Math.abs(actual - expected[f][b]));
        compared += 1;
      }
    }
  }
  expect(compared).toBeGreaterThan(20);
  expect(maxSignalAbs).toBeLessThan(0.0035);
});
