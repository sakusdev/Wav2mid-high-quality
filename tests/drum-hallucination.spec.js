import { test, expect } from '@playwright/test';

const EXPECTED_DRUMS = [0.30, 0.50, 0.80, 1.05, 1.40, 1.60, 1.90, 2.20];

function scoreOnsets(actual, expected, tolerance = 0.12) {
  const used = new Set();
  let matched = 0;
  for (const target of expected) {
    let best = -1;
    let distance = Infinity;
    for (let i = 0; i < actual.length; i += 1) {
      if (used.has(i)) continue;
      const d = Math.abs(actual[i].time - target);
      if (d <= tolerance && d < distance) { best = i; distance = d; }
    }
    if (best >= 0) { used.add(best); matched += 1; }
  }
  return { matched, recall: matched / expected.length };
}

test('drum classifier rejects pitched attacks without losing synthetic drum recall', async ({ page }) => {
  await page.goto('/');
  const result = await page.evaluate(async () => {
    const { detectDrumEvents } = await import('/src/transcribe.js');
    const sampleRate = 22050;
    const duration = 3;
    const length = Math.round(sampleRate * duration);

    const piano = new Float32Array(length);
    const pianoEvents = [
      [0.30, [48, 55, 60, 64, 67]],
      [0.90, [52, 59, 64, 67, 71]],
      [1.50, [60, 64, 67, 72]],
      [2.10, [72, 76, 79, 84]],
      [2.55, [84, 88, 91]],
    ];
    for (let i = 0; i < length; i += 1) {
      const t = i / sampleRate;
      let value = 0;
      for (const [hit, pitches] of pianoEvents) {
        const local = t - hit;
        if (local < 0 || local >= 0.45) continue;
        const envelope = Math.exp(-5 * local) * Math.min(1, local / 0.003);
        let chord = 0;
        for (const midi of pitches) {
          const f = 440 * 2 ** ((midi - 69) / 12);
          chord += Math.sin(2 * Math.PI * f * t)
            + 0.38 * Math.sin(4 * Math.PI * f * t)
            + 0.18 * Math.sin(6 * Math.PI * f * t)
            + 0.08 * Math.sin(8 * Math.PI * f * t);
        }
        value += 0.5 * envelope * chord / pitches.length;
      }
      piano[i] = value;
    }

    const drums = new Float32Array(length);
    const kickTimes = [0.30, 1.40];
    const snareTimes = [0.80, 1.90];
    const hatTimes = [0.50, 1.05, 1.60, 2.20];
    for (let i = 0; i < length; i += 1) {
      const t = i / sampleRate;
      let value = 0;
      for (const hit of kickTimes) {
        const local = t - hit;
        if (local >= 0 && local < 0.18) {
          value += 0.8 * Math.exp(-22 * local)
            * Math.sin(2 * Math.PI * (72 - 150 * local) * local);
        }
      }
      for (const hit of snareTimes) {
        const local = t - hit;
        if (local >= 0 && local < 0.16) {
          value += 0.25 * Math.exp(-28 * local)
            * (Math.sin(2 * Math.PI * 1650 * t) + 0.7 * Math.sin(2 * Math.PI * 2870 * t));
        }
      }
      for (const hit of hatTimes) {
        const local = t - hit;
        if (local >= 0 && local < 0.10) {
          value += 0.15 * Math.exp(-55 * local)
            * (Math.sin(2 * Math.PI * 5900 * t) + 0.5 * Math.sin(2 * Math.PI * 7600 * t));
        }
      }
      drums[i] = value;
    }

    return {
      piano: detectDrumEvents(piano, sampleRate),
      drums: detectDrumEvents(drums, sampleRate),
    };
  });

  const score = scoreOnsets(result.drums, EXPECTED_DRUMS);
  console.log('drum hallucination regression:', JSON.stringify({
    pianoFalsePositives: result.piano.length,
    drums: result.drums.length,
    score,
  }));
  expect(result.piano.length).toBeLessThanOrEqual(1);
  expect(score.recall).toBeGreaterThanOrEqual(0.75);
});
