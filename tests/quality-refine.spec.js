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

  expect(result.pipeline.qualityRefiner).toBe('realworld-v2');
  expect(result.pipeline.chordWindow).toBe('tempo-aware-two-beat');
  expect(result.notes.filter(note => note.pitchMidi === 60)).toHaveLength(1);
  expect(result.notes.find(note => note.pitchMidi === 60).durationSeconds).toBeGreaterThanOrEqual(0.94);
  expect(result.notes.some(note => note.pitchMidi === 96)).toBe(false);
  expect(result.key).toBe('C major');
  expect(result.chords[0]?.name).toBe('C');
  expect(result.stats.noteCount).toBe(3);
  expect(result.stats.maxPolyphony).toBe(3);
});

test('INSANE requires cross-pass consensus except for exceptional sustained two-pass notes', () => {
  const common = { amplitude: 0.8, source: 'ensemble', instrument: 'harmony' };
  const input = {
    notes: [
      { ...common, pitchMidi: 60, startTimeSeconds: 0.00, durationSeconds: 0.40, confidence: 0.80, agreement: 3, sources: ['mix', 'harmonic', 'presence'] },
      { ...common, pitchMidi: 64, startTimeSeconds: 0.02, durationSeconds: 0.18, confidence: 0.88, agreement: 2, sources: ['harmonic', 'presence'] },
      { ...common, pitchMidi: 67, startTimeSeconds: 0.03, durationSeconds: 0.30, confidence: 0.95, agreement: 2, sources: ['mix', 'harmonic'] },
      { ...common, pitchMidi: 72, startTimeSeconds: 0.04, durationSeconds: 0.20, confidence: 0.99, agreement: 1, sources: ['presence'], source: 'presence' },
    ],
    drums: [],
    tempo: 120,
    stats: { duration: 1, candidateCount: 4, chunks: 1, passCount: 4 },
    pipeline: { mode: 'INSANE', ensemble: true, backend: 'wasm', stemPasses: ['mix', 'harmonic', 'bass', 'presence'] },
  };

  const result = refineResult(input, 1);
  const pitches = result.notes.map(note => note.pitchMidi);

  expect(result.pipeline.consensusGate).toBe('3-of-4-or-exceptional');
  expect(pitches).toContain(60);
  expect(pitches).toContain(67);
  expect(pitches).not.toContain(64);
  expect(pitches).not.toContain(72);
});
