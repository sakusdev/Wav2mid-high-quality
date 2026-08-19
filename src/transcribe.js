import * as tf from '@tensorflow/tfjs';
import { setWasmPaths } from '@tensorflow/tfjs-backend-wasm';
import '@tensorflow/tfjs-backend-wasm';
import {
  BasicPitch,
  addPitchBendsToNoteEvents,
  noteFramesToTime,
  outputToNotesPoly,
} from '@spotify/basic-pitch';
import { Midi } from '@tonejs/midi';

const MODEL_URL = '/model/basic-pitch/model.json';
const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

export const MODES = {
  fast: { label: 'FAST', onsetThreshold: 0.36, frameThreshold: 0.34, minNoteLength: 7, minDuration: 0.07, harmonicSuppression: 0, mergeGap: 0.025 },
  pro: { label: 'PRO', onsetThreshold: 0.28, frameThreshold: 0.28, minNoteLength: 5, minDuration: 0.055, harmonicSuppression: 0.42, mergeGap: 0.045 },
  insane: { label: 'INSANE', onsetThreshold: 0.22, frameThreshold: 0.23, minNoteLength: 4, minDuration: 0.045, harmonicSuppression: 0.52, mergeGap: 0.055 },
};

let basicPitchInstance = null;

export async function configureBackend(preference = 'auto') {
  setWasmPaths('/tfjs-wasm/');
  const choices = preference === 'auto' ? ['webgl', 'wasm', 'cpu'] : [preference, 'wasm', 'cpu'];
  let lastError;
  for (const backend of [...new Set(choices)]) {
    try {
      const ok = await tf.setBackend(backend);
      if (!ok) continue;
      await tf.ready();
      basicPitchInstance = null;
      return tf.getBackend();
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError ?? new Error('TensorFlow.js backend could not be initialized.');
}

export async function transcribe(audioBuffer, options = {}, onProgress = () => {}) {
  const mode = MODES[options.mode] ?? MODES.pro;
  const frames = [];
  const onsets = [];
  const contours = [];
  if (!basicPitchInstance) basicPitchInstance = new BasicPitch(MODEL_URL);

  onProgress({ stage: 'infer', value: 0 });
  await basicPitchInstance.evaluateModel(
    audioBuffer,
    (frameChunk, onsetChunk, contourChunk) => {
      frames.push(...frameChunk);
      onsets.push(...onsetChunk);
      contours.push(...contourChunk);
    },
    progress => onProgress({ stage: 'infer', value: progress }),
  );

  onProgress({ stage: 'decode', value: 0.93 });
  const rawNotes = noteFramesToTime(
    addPitchBendsToNoteEvents(
      contours,
      outputToNotesPoly(frames, onsets, mode.onsetThreshold, mode.frameThreshold, mode.minNoteLength),
    ),
  );

  onProgress({ stage: 'clean', value: 0.96 });
  const notes = postProcessNotes(rawNotes, mode, options);
  const tempo = estimateTempo(notes);
  const key = estimateKey(notes);
  const chords = estimateChords(notes, audioBuffer.duration);
  const stats = buildStats(notes, audioBuffer.duration);
  onProgress({ stage: 'done', value: 1 });
  return { notes, rawNotes, tempo, key, chords, stats };
}

function postProcessNotes(rawNotes, mode, options) {
  const minPitch = Number(options.minPitch ?? 21);
  const maxPitch = Number(options.maxPitch ?? 108);
  const sensitivity = Number(options.sensitivity ?? 1);
  const minAmp = Math.max(0.025, 0.08 / sensitivity);

  let notes = rawNotes
    .map(note => ({ ...note }))
    .filter(note => note.pitchMidi >= minPitch && note.pitchMidi <= maxPitch)
    .filter(note => note.durationSeconds >= mode.minDuration)
    .filter(note => note.amplitude >= minAmp)
    .sort((a, b) => a.startTimeSeconds - b.startTimeSeconds || a.pitchMidi - b.pitchMidi);

  notes = mergeAdjacentNotes(notes, mode.mergeGap);
  if (mode.harmonicSuppression > 0) notes = suppressLikelyHarmonics(notes, mode.harmonicSuppression);
  notes = removeNearDuplicates(notes);

  return notes.map(note => ({
    ...note,
    amplitude: Math.min(1, Math.max(0.05, Math.sqrt(note.amplitude))),
  }));
}

function mergeAdjacentNotes(notes, maxGap) {
  const byPitch = new Map();
  for (const note of notes) {
    const list = byPitch.get(note.pitchMidi) ?? [];
    list.push(note);
    byPitch.set(note.pitchMidi, list);
  }

  const merged = [];
  for (const list of byPitch.values()) {
    if (!list.length) continue;
    list.sort((a, b) => a.startTimeSeconds - b.startTimeSeconds);
    let current = { ...list[0] };
    for (let i = 1; i < list.length; i += 1) {
      const next = list[i];
      const end = current.startTimeSeconds + current.durationSeconds;
      const gap = next.startTimeSeconds - end;
      if (gap <= maxGap && gap >= -0.08) {
        const nextEnd = next.startTimeSeconds + next.durationSeconds;
        current.durationSeconds = Math.max(end, nextEnd) - current.startTimeSeconds;
        current.amplitude = Math.max(current.amplitude, next.amplitude);
        if (current.pitchBends && next.pitchBends) current.pitchBends.push(...next.pitchBends);
      } else {
        merged.push(current);
        current = { ...next };
      }
    }
    merged.push(current);
  }
  return merged.sort((a, b) => a.startTimeSeconds - b.startTimeSeconds || a.pitchMidi - b.pitchMidi);
}

function suppressLikelyHarmonics(notes, strength) {
  const suspectIntervals = new Set([12, 19, 24, 28, 31, 36]);
  const removed = new Set();
  for (let i = 0; i < notes.length; i += 1) {
    const low = notes[i];
    for (let j = i + 1; j < notes.length; j += 1) {
      const high = notes[j];
      if (high.startTimeSeconds - low.startTimeSeconds > 0.035) break;
      const interval = high.pitchMidi - low.pitchMidi;
      if (!suspectIntervals.has(interval) || high.pitchMidi < 76) continue;
      const durationSimilarity = Math.min(low.durationSeconds, high.durationSeconds) / Math.max(low.durationSeconds, high.durationSeconds);
      const amplitudeRatio = high.amplitude / Math.max(0.001, low.amplitude);
      const threshold = 0.30 + strength * 0.38;
      if (durationSimilarity > 0.68 && amplitudeRatio < threshold) removed.add(j);
    }
  }
  return notes.filter((_, index) => !removed.has(index));
}

function removeNearDuplicates(notes) {
  const kept = [];
  for (const note of notes) {
    const duplicate = kept.find(existing =>
      existing.pitchMidi === note.pitchMidi &&
      Math.abs(existing.startTimeSeconds - note.startTimeSeconds) < 0.012 &&
      Math.abs(existing.durationSeconds - note.durationSeconds) < 0.025,
    );
    if (!duplicate) kept.push(note);
    else if (note.amplitude > duplicate.amplitude) Object.assign(duplicate, note);
  }
  return kept;
}

export function exportMidi(result, filename = 'transcription.mid') {
  const midi = new Midi();
  const bpm = Number.isFinite(result.tempo) ? result.tempo : 120;
  midi.header.setTempo(bpm);
  const track = midi.addTrack();
  track.name = 'Wav2mid HQ';

  for (const note of result.notes) {
    track.addNote({ midi: note.pitchMidi, time: note.startTimeSeconds, duration: note.durationSeconds, velocity: note.amplitude });
    if (note.pitchBends?.length) {
      const maxBends = 48;
      const stride = Math.max(1, Math.ceil(note.pitchBends.length / maxBends));
      for (let i = 0; i < note.pitchBends.length; i += stride) {
        track.addPitchBend({
          time: note.startTimeSeconds + note.durationSeconds * (i / note.pitchBends.length),
          value: note.pitchBends[i],
        });
      }
    }
  }
  downloadBlob(new Blob([midi.toArray()], { type: 'audio/midi' }), filename);
}

export function exportJson(result, filename = 'transcription.json') {
  const payload = JSON.stringify({
    format: 'wav2mid-hq/v1',
    tempo: result.tempo,
    key: result.key,
    chords: result.chords,
    stats: result.stats,
    notes: result.notes,
  }, null, 2);
  downloadBlob(new Blob([payload], { type: 'application/json' }), filename);
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function estimateTempo(notes) {
  if (notes.length < 6) return 120;
  const onsets = [...new Set(notes.map(n => Math.round(n.startTimeSeconds * 50) / 50))].slice(0, 500);
  const votes = new Map();
  for (let i = 0; i < onsets.length; i += 1) {
    for (let j = i + 1; j < Math.min(onsets.length, i + 10); j += 1) {
      const dt = onsets[j] - onsets[i];
      if (dt < 0.18 || dt > 2.0) continue;
      let bpm = 60 / dt;
      while (bpm < 70) bpm *= 2;
      while (bpm > 180) bpm /= 2;
      const rounded = Math.round(bpm);
      votes.set(rounded, (votes.get(rounded) ?? 0) + 1 / (j - i));
    }
  }
  if (!votes.size) return 120;
  return [...votes.entries()].sort((a, b) => b[1] - a[1])[0][0];
}

function estimateKey(notes) {
  if (!notes.length) return '—';
  const majorProfile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
  const minorProfile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];
  const histogram = Array(12).fill(0);
  for (const note of notes) histogram[note.pitchMidi % 12] += note.durationSeconds * note.amplitude;
  let best = { score: -Infinity, label: '—' };
  for (let root = 0; root < 12; root += 1) {
    for (const [profile, suffix] of [[majorProfile, 'major'], [minorProfile, 'minor']]) {
      let score = 0;
      for (let pc = 0; pc < 12; pc += 1) score += histogram[pc] * profile[(pc - root + 12) % 12];
      if (score > best.score) best = { score, label: `${NOTE_NAMES[root]} ${suffix}` };
    }
  }
  return best.label;
}

const CHORD_TEMPLATES = [
  ['maj', [0, 4, 7]], ['min', [0, 3, 7]], ['dim', [0, 3, 6]], ['aug', [0, 4, 8]],
  ['sus2', [0, 2, 7]], ['sus4', [0, 5, 7]], ['7', [0, 4, 7, 10]], ['maj7', [0, 4, 7, 11]],
  ['m7', [0, 3, 7, 10]], ['mMaj7', [0, 3, 7, 11]], ['6', [0, 4, 7, 9]], ['m6', [0, 3, 7, 9]],
  ['add9', [0, 2, 4, 7]], ['madd9', [0, 2, 3, 7]], ['9', [0, 2, 4, 7, 10]], ['m9', [0, 2, 3, 7, 10]],
];

function estimateChords(notes, duration) {
  const windowSize = 0.5;
  const segments = [];
  let previous = null;
  for (let t = 0; t < duration; t += windowSize) {
    const active = notes.filter(n => n.startTimeSeconds < t + windowSize && n.startTimeSeconds + n.durationSeconds > t);
    if (active.length < 2) continue;
    const weights = Array(12).fill(0);
    for (const n of active) weights[n.pitchMidi % 12] += n.amplitude * Math.min(n.durationSeconds, windowSize);
    const chord = bestChord(weights);
    if (!chord || chord.score < 0.58) continue;
    if (previous?.name === chord.name && t - previous.end <= windowSize * 1.5) {
      previous.end = t + windowSize;
      previous.confidence = Math.max(previous.confidence, chord.score);
    } else {
      previous = { name: chord.name, start: t, end: t + windowSize, confidence: chord.score };
      segments.push(previous);
    }
  }
  return segments.slice(0, 300);
}

function bestChord(weights) {
  const total = weights.reduce((a, b) => a + b, 0);
  if (!total) return null;
  let best = null;
  for (let root = 0; root < 12; root += 1) {
    for (const [suffix, intervals] of CHORD_TEMPLATES) {
      const pcs = new Set(intervals.map(i => (root + i) % 12));
      const inWeight = weights.reduce((sum, w, pc) => sum + (pcs.has(pc) ? w : 0), 0);
      const coverage = intervals.filter(i => weights[(root + i) % 12] > total * 0.025).length / intervals.length;
      const score = (inWeight / total) * 0.72 + coverage * 0.28;
      if (!best || score > best.score) {
        const suffixLabel = suffix === 'maj' ? '' : suffix === 'min' ? 'm' : suffix;
        best = { name: `${NOTE_NAMES[root]}${suffixLabel}`, score };
      }
    }
  }
  return best;
}

function buildStats(notes, duration) {
  let maxPolyphony = 0;
  const events = [];
  for (const n of notes) {
    events.push([n.startTimeSeconds, 1]);
    events.push([n.startTimeSeconds + n.durationSeconds, -1]);
  }
  events.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  let poly = 0;
  for (const [, delta] of events) {
    poly += delta;
    maxPolyphony = Math.max(maxPolyphony, poly);
  }
  return {
    duration,
    noteCount: notes.length,
    maxPolyphony,
    lowestNote: notes.length ? Math.min(...notes.map(n => n.pitchMidi)) : null,
    highestNote: notes.length ? Math.max(...notes.map(n => n.pitchMidi)) : null,
  };
}

export function midiName(midi) {
  if (midi == null) return '—';
  return `${NOTE_NAMES[midi % 12]}${Math.floor(midi / 12) - 1}`;
}
