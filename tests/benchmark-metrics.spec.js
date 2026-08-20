import { test, expect } from '@playwright/test';
import { scoreTranscription } from '../scripts/benchmark-metrics.mjs';

function note(pitchMidi, startTimeSeconds, durationSeconds) {
  return { pitchMidi, startTimeSeconds, durationSeconds, velocity: 0.8 };
}

test('benchmark metrics return perfect scores for identical transcription', () => {
  const reference = {
    tonal: [note(60, 0.0, 0.5), note(64, 0.5, 0.4), note(67, 1.0, 0.6)],
    drums: [note(36, 0.25, 0.05), note(38, 0.75, 0.05)],
  };
  const metrics = scoreTranscription(reference, structuredClone(reference));
  expect(metrics.onset.f1).toBeCloseTo(1, 8);
  expect(metrics.offset.f1).toBeCloseTo(1, 8);
  expect(metrics.frame.f1).toBeCloseTo(1, 8);
  expect(metrics.drums.f1).toBeCloseTo(1, 8);
  expect(metrics.objective).toBeCloseTo(1, 8);
});

test('benchmark onset tolerance accepts 40 ms and rejects 80 ms shifts', () => {
  const reference = { tonal: [note(60, 1.0, 0.5)], drums: [] };
  const accepted = scoreTranscription(reference, { tonal: [note(60, 1.04, 0.5)], drums: [] });
  const rejected = scoreTranscription(reference, { tonal: [note(60, 1.08, 0.5)], drums: [] });
  expect(accepted.onset.f1).toBe(1);
  expect(rejected.onset.f1).toBe(0);
});

test('frame metric measures duration overlap instead of onset count', () => {
  const reference = { tonal: [note(60, 0, 1)], drums: [] };
  const prediction = { tonal: [note(60, 0, 0.5)], drums: [] };
  const metrics = scoreTranscription(reference, prediction);
  expect(metrics.onset.f1).toBe(1);
  expect(metrics.frame.precision).toBeCloseTo(1, 8);
  expect(metrics.frame.recall).toBeCloseTo(0.5, 8);
  expect(metrics.frame.f1).toBeCloseTo(2 / 3, 8);
});

test('drum metric requires matching GM drum class', () => {
  const reference = { tonal: [], drums: [note(36, 0.5, 0.05)] };
  const wrongClass = { tonal: [], drums: [note(38, 0.5, 0.05)] };
  const metrics = scoreTranscription(reference, wrongClass);
  expect(metrics.drums.f1).toBe(0);
});

test('drum hallucinations reduce objective on a drumless reference', () => {
  const reference = { tonal: [note(60, 0, 1)], drums: [] };
  const clean = scoreTranscription(reference, { tonal: [note(60, 0, 1)], drums: [] });
  const hallucinated = scoreTranscription(reference, { tonal: [note(60, 0, 1)], drums: [note(38, 0.5, 0.05)] });
  expect(clean.objective).toBeCloseTo(1, 8);
  expect(hallucinated.drums.f1).toBe(0);
  expect(hallucinated.objective).toBeLessThan(clean.objective);
  expect(hallucinated.objective).toBeCloseTo(0.9, 8);
});
