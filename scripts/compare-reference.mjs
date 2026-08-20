import { readFile } from 'node:fs/promises';
import { Midi } from '@tonejs/midi';

const args = Object.fromEntries(process.argv.slice(2).map((arg, i, all) => {
  if (!arg.startsWith('--')) return [arg, true];
  const [key, inline] = arg.slice(2).split('=', 2);
  if (inline !== undefined) return [key, inline];
  const next = all[i + 1];
  return [key, next && !next.startsWith('--') ? next : true];
}));

if (!args.reference || !args.analysis) {
  console.error('Usage: node scripts/compare-reference.mjs --reference reference.mid --analysis transcription.analysis.json [--onset-tolerance 0.20]');
  process.exit(2);
}

const onsetTolerance = Number(args['onset-tolerance'] ?? 0.20);
const frameStep = Number(args['frame-step'] ?? 0.01);
const searchOffset = Number(args['search-offset'] ?? 0.35);

const midi = new Midi(await readFile(String(args.reference)));
const analysis = JSON.parse(await readFile(String(args.analysis), 'utf8'));
const referenceNotes = midi.tracks
  .filter(track => track.channel !== 9)
  .flatMap(track => track.notes.map(note => ({
    pitchMidi: note.midi,
    start: note.time,
    end: note.time + note.duration,
  })));
const referenceDrums = midi.tracks
  .filter(track => track.channel === 9)
  .flatMap(track => track.notes.map(note => ({ midi: note.midi, time: note.time })));
const actualNotes = (analysis.notes ?? []).map(note => ({
  pitchMidi: note.pitchMidi,
  start: note.startTimeSeconds,
  end: note.startTimeSeconds + note.durationSeconds,
}));
const actualDrums = (analysis.drums ?? []).map(drum => ({ midi: drum.midi, time: drum.time }));

function matchEvents(reference, actual, tolerance, offset) {
  const used = new Set();
  let matched = 0;
  let onsetError = 0;
  for (const target of reference) {
    let best = null;
    for (let i = 0; i < actual.length; i += 1) {
      if (used.has(i) || actual[i].pitchMidi !== target.pitchMidi) continue;
      const error = Math.abs(actual[i].start + offset - target.start);
      if (error <= tolerance && (!best || error < best.error)) best = { i, error };
    }
    if (best) {
      used.add(best.i);
      matched += 1;
      onsetError += best.error;
    }
  }
  const precision = matched / Math.max(1, actual.length);
  const recall = matched / Math.max(1, reference.length);
  const f1 = precision + recall ? 2 * precision * recall / (precision + recall) : 0;
  return { matched, precision, recall, f1, meanOnsetError: matched ? onsetError / matched : null };
}

function frameScore(reference, actual, step, offset) {
  const referenceFrames = new Set();
  const actualFrames = new Set();
  for (const note of reference) {
    for (let frame = Math.floor(note.start / step); frame < Math.ceil(note.end / step); frame += 1) referenceFrames.add(`${frame}:${note.pitchMidi}`);
  }
  for (const note of actual) {
    const start = note.start + offset;
    const end = note.end + offset;
    for (let frame = Math.max(0, Math.floor(start / step)); frame < Math.ceil(end / step); frame += 1) actualFrames.add(`${frame}:${note.pitchMidi}`);
  }
  let matched = 0;
  for (const frame of actualFrames) if (referenceFrames.has(frame)) matched += 1;
  const precision = matched / Math.max(1, actualFrames.size);
  const recall = matched / Math.max(1, referenceFrames.size);
  const f1 = precision + recall ? 2 * precision * recall / (precision + recall) : 0;
  return { matched, precision, recall, f1, referenceFrames: referenceFrames.size, actualFrames: actualFrames.size };
}

let best = null;
for (let offset = -searchOffset; offset <= searchOffset + 1e-9; offset += 0.01) {
  const event = matchEvents(referenceNotes, actualNotes, onsetTolerance, offset);
  const frame = frameScore(referenceNotes, actualNotes, frameStep, offset);
  const score = event.f1 * 0.7 + frame.f1 * 0.3;
  if (!best || score > best.score) best = { score, offset, event, frame };
}

function matchDrumOnsets(reference, actual, tolerance = 0.12) {
  const used = new Set();
  let matched = 0;
  for (const target of reference) {
    let best = null;
    for (let i = 0; i < actual.length; i += 1) {
      if (used.has(i)) continue;
      const error = Math.abs(actual[i].time + bestOffset - target.time);
      if (error <= tolerance && (!best || error < best.error)) best = { i, error };
    }
    if (best) { used.add(best.i); matched += 1; }
  }
  const precision = matched / Math.max(1, actual.length);
  const recall = matched / Math.max(1, reference.length);
  const f1 = precision + recall ? 2 * precision * recall / (precision + recall) : 0;
  return { matched, precision, recall, f1, reference: reference.length, actual: actual.length };
}

const bestOffset = best.offset;
const drum = matchDrumOnsets(referenceDrums, actualDrums);
const result = {
  reference: { notes: referenceNotes.length, drums: referenceDrums.length, duration: midi.duration, tempos: midi.header.tempos.map(item => item.bpm) },
  analysis: { notes: actualNotes.length, drums: actualDrums.length, tempo: analysis.tempo, key: analysis.key },
  bestOffsetSeconds: Number(bestOffset.toFixed(3)),
  onsetToleranceSeconds: onsetTolerance,
  tonalEvents: best.event,
  tonalFrames: best.frame,
  drums: drum,
};

console.log(JSON.stringify(result, null, 2));
