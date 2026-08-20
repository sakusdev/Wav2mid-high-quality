import { test, expect } from '@playwright/test';
import { refineResult } from '../src/quality-refine.js';

test('real-world refiner merges fragments, rejects a short harmonic ghost and stabilizes context', () => {
  const input = {
    notes: [
      { pitchMidi: 60, startTimeSeconds: 0.00, durationSeconds: 0.50, amplitude: 0.86, confidence: 0.91, agreement: 2, sources: ['mix', 'harmonic'], source: 'ensemble', instrument: 'harmony' },
      { pitchMidi: 60, startTimeSeconds: 0.55, durationSeconds: 0.40, amplitude: 0.80, confidence: 0.88, agreement: 2, sources: ['mix', 'harmonic'], source: 'ensemble', instrument: 'harmony' },
      { pitchMidi: 64, startTimeSeconds: 0.02, durationSeconds: 0.88, amplitude: 0.78, confidence: 0.84, agreement: 2, sources: ['mix', 'harmonic'], source: 'ensemble', instrument: 'harmony' },
      { pitchMidi: 67, startTimeSeconds: 0.03, durationSeconds: 0.84, amplitude: 0.75, confidence: 0.82, agreement: 2, sources: ['mix', 'harmonic'], source: 'ensemble', instrument: 'harmony' },
      { pitchMidi: 96, startTimeSeconds: 0.01, durationSeconds: 0.10, amplitude: 0.35, confidence: 0.44, agreement: 1, sources: ['harmonic'], source: 'harmonic', instrument: 'harmony' },
    ],
    drums: [],
    tempo: 120,
    key: 'A minor',
    keyDetail: { root: 9, mode: 'minor', label: 'A minor', score: 1 },
    chords: [],
    stats: { duration: 1, candidateCount: 9, chunks: 1, passCount: 3 },
    pipeline: { mode: 'PRO', ensemble: true, backend: 'wasm', stemPasses: ['mix', 'harmonic', 'bass'] },
  };

  const result = refineResult(input, 1);

  expect(result.pipeline.qualityRefiner).toBe('realworld-v1');
  expect(result.pipeline.chordWindow).toBe('tempo-aware-two-beat');
  expect(result.notes.filter(note => note.pitchMidi === 60)).toHaveLength(1);
  expect(result.notes.find(note => note.pitchMidi === 60).durationSeconds).toBeGreaterThanOrEqual(0.94);
  expect(result.notes.some(note => note.pitchMidi === 96)).toBe(false);
  expect(result.key).toBe('C major');
  expect(result.chords[0]?.name).toBe('C');
  expect(result.stats.noteCount).toBe(3);
  expect(result.stats.maxPolyphony).toBe(3);
});
