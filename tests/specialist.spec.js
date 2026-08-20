import { test, expect } from '@playwright/test';
import {
  deframeSpecialistOutputs,
  logMelFromWaveform,
  makeSpecialistSegments,
  SPECIALIST_FRONTEND,
} from '../src/specialist-frontend.js';
import { postprocessSpecialistOutputs } from '../src/specialist-postprocess.js';
import { promotedSpecialists } from '../src/specialist-runtime.js';

test('specialist frontend produces torchlibrosa-compatible shape and finite log-mel values', () => {
  const samples = new Float32Array(1600);
  for (let i = 0; i < samples.length; i += 1) {
    const t = i / SPECIALIST_FRONTEND.sampleRate;
    samples[i] = 0.4 * Math.sin(2 * Math.PI * 440 * t);
  }
  const feature = logMelFromWaveform(samples);
  expect(feature.dims).toEqual([1, 1, 11, 229]);
  expect(feature.data.length).toBe(11 * 229);
  expect([...feature.data].every(Number.isFinite)).toBe(true);
  expect(Math.max(...feature.data)).toBeGreaterThan(-80);
});

test('specialist segmentation and deframe use 10 second windows with 50% overlap', () => {
  const samples = new Float32Array(12 * 16000);
  const segmented = makeSpecialistSegments(samples, 10);
  expect(segmented.segments).toHaveLength(3);
  expect(segmented.segments[0].length).toBe(160000);

  const segmentOutputs = [1, 2, 3].map(value => ({
    reg_onset_output: tensor(value),
    reg_offset_output: tensor(value),
    frame_output: tensor(value),
    velocity_output: tensor(value),
  }));
  const merged = deframeSpecialistOutputs(segmentOutputs, samples.length);
  expect(merged.frame_output.dims).toEqual([1201, 1]);
  expect(merged.frame_output.data[0]).toBe(1);
  expect(merged.frame_output.data[749]).toBe(1);
  expect(merged.frame_output.data[750]).toBe(2);
  expect(merged.frame_output.data[1200]).toBe(2);
});

test('regression postprocessor detects onset, offset, duration and velocity', () => {
  const frames = 30;
  const onset = matrix(frames, 1, 0.01);
  setCurve(onset.data, 5, [0.1, 0.3, 0.8, 0.4, 0.2], 3);
  const offset = matrix(frames, 1, 0.01);
  setCurve(offset.data, 15, [0.05, 0.1, 0.2, 0.4, 0.8, 0.4, 0.2, 0.1, 0.05], 11);
  const frame = matrix(frames, 1, 0.02);
  for (let i = 5; i <= 15; i += 1) frame.data[i] = 0.8;
  const velocity = matrix(frames, 1, 0.1);
  velocity.data[5] = 0.7;

  const notes = postprocessSpecialistOutputs({
    reg_onset_output: onset,
    reg_offset_output: offset,
    frame_output: frame,
    velocity_output: velocity,
  }, { classes: 1, beginNote: 60, source: 'specialist:test' });

  expect(notes).toHaveLength(1);
  expect(notes[0].pitchMidi).toBe(60);
  expect(notes[0].startTimeSeconds).toBeCloseTo(0.05, 3);
  expect(notes[0].durationSeconds).toBeCloseTo(0.10, 2);
  expect(notes[0].velocity).toBe(89);
  expect(notes[0].confidence).toBeGreaterThan(0.6);
  expect(notes[0].source).toBe('specialist:test');
});

test('only benchmark-promoted specialist models are selected', () => {
  const manifest = {
    version: 1,
    models: {
      bass: { url: '/bass.onnx', stem: 'bass', promoted: true },
      piano: { url: '/piano.onnx', stem: 'other', promoted: false },
      guitar: { url: '/guitar.onnx', stem: 'other', promoted: true, enabled: false },
    },
  };
  expect(promotedSpecialists(manifest).map(item => item.name)).toEqual(['bass']);
  expect(promotedSpecialists(manifest, 'bass')).toHaveLength(1);
  expect(promotedSpecialists(manifest, 'other')).toHaveLength(0);
});

function tensor(value) {
  const data = new Float32Array(1001);
  data.fill(value);
  return { data, dims: [1, 1001, 1] };
}

function matrix(frames, classes, fill) {
  const data = new Float32Array(frames * classes);
  data.fill(fill);
  return { data, dims: [frames, classes] };
}

function setCurve(target, center, values, start) {
  void center;
  for (let i = 0; i < values.length; i += 1) target[start + i] = values[i];
}
