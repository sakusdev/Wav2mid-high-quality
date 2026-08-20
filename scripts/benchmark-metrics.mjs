import fs from 'node:fs/promises';
import midiPackage from '@tonejs/midi';

const { Midi } = midiPackage;

export async function readMidiNotes(path) {
  const bytes = await fs.readFile(path);
  const midi = new Midi(bytes);
  const tonal = [];
  const drums = [];
  for (const track of midi.tracks) {
    const isDrum = track.channel === 9;
    for (const note of track.notes) {
      const item = {
        pitchMidi: note.midi,
        startTimeSeconds: note.time,
        durationSeconds: note.duration,
        velocity: note.velocity,
        track: track.name || '',
        channel: track.channel,
        instrument: track.instrument?.name || '',
      };
      (isDrum ? drums : tonal).push(item);
    }
  }
  return { tonal, drums, duration: midi.duration };
}

export async function readPrediction(path) {
  if (path.toLowerCase().endsWith('.json')) {
    const payload = JSON.parse(await fs.readFile(path, 'utf8'));
    return {
      tonal: (payload.notes ?? []).map(normalizeNote),
      drums: (payload.drums ?? []).map(drum => ({
        pitchMidi: Number(drum.midi),
        startTimeSeconds: Number(drum.time),
        durationSeconds: Number(drum.duration ?? 0.05),
        velocity: Number(drum.velocity ?? 1),
      })),
      duration: Number(payload.stats?.duration ?? 0),
    };
  }
  return readMidiNotes(path);
}

function normalizeNote(note) {
  return {
    pitchMidi: Number(note.pitchMidi ?? note.midi),
    startTimeSeconds: Number(note.startTimeSeconds ?? note.time),
    durationSeconds: Number(note.durationSeconds ?? note.duration),
    velocity: Number(note.amplitude ?? note.velocity ?? 1),
  };
}

export function scoreTranscription(reference, prediction, options = {}) {
  const onsetTolerance = Number(options.onsetTolerance ?? 0.05);
  const offsetTolerance = Number(options.offsetTolerance ?? 0.05);
  const offsetRatio = Number(options.offsetRatio ?? 0.2);
  const drumTolerance = Number(options.drumTolerance ?? 0.05);

  const onset = matchNotes(reference.tonal, prediction.tonal, {
    onsetTolerance,
    requireOffset: false,
  });
  const offset = matchNotes(reference.tonal, prediction.tonal, {
    onsetTolerance,
    offsetTolerance,
    offsetRatio,
    requireOffset: true,
  });
  const frame = frameMetrics(reference.tonal, prediction.tonal);
  const drums = matchDrums(reference.drums ?? [], prediction.drums ?? [], drumTolerance);

  return {
    onset,
    offset,
    frame,
    drums,
    objective: objectiveScore({ onset, offset, frame, drums }),
  };
}

export function matchNotes(reference, prediction, options = {}) {
  const onsetTolerance = Number(options.onsetTolerance ?? 0.05);
  const offsetTolerance = Number(options.offsetTolerance ?? 0.05);
  const offsetRatio = Number(options.offsetRatio ?? 0.2);
  const requireOffset = Boolean(options.requireOffset);
  const used = new Set();
  let matched = 0;

  const refSorted = [...reference].sort((a, b) => a.startTimeSeconds - b.startTimeSeconds || a.pitchMidi - b.pitchMidi);
  const predSorted = [...prediction].sort((a, b) => a.startTimeSeconds - b.startTimeSeconds || a.pitchMidi - b.pitchMidi);

  for (const ref of refSorted) {
    let bestIndex = -1;
    let bestCost = Infinity;
    for (let i = 0; i < predSorted.length; i += 1) {
      if (used.has(i)) continue;
      const pred = predSorted[i];
      if (pred.pitchMidi !== ref.pitchMidi) continue;
      const onsetError = Math.abs(pred.startTimeSeconds - ref.startTimeSeconds);
      if (onsetError > onsetTolerance) continue;
      let offsetError = 0;
      if (requireOffset) {
        const refEnd = ref.startTimeSeconds + ref.durationSeconds;
        const predEnd = pred.startTimeSeconds + pred.durationSeconds;
        offsetError = Math.abs(predEnd - refEnd);
        const allowed = Math.max(offsetTolerance, Math.abs(ref.durationSeconds) * offsetRatio);
        if (offsetError > allowed) continue;
      }
      const cost = onsetError + offsetError * 0.35;
      if (cost < bestCost) {
        bestCost = cost;
        bestIndex = i;
      }
    }
    if (bestIndex >= 0) {
      used.add(bestIndex);
      matched += 1;
    }
  }

  return prf(matched, prediction.length, reference.length);
}

export function matchDrums(reference, prediction, tolerance = 0.05) {
  const used = new Set();
  let matched = 0;
  for (const ref of reference) {
    let best = -1;
    let bestError = Infinity;
    for (let i = 0; i < prediction.length; i += 1) {
      if (used.has(i)) continue;
      const pred = prediction[i];
      if (pred.pitchMidi !== ref.pitchMidi) continue;
      const error = Math.abs(pred.startTimeSeconds - ref.startTimeSeconds);
      if (error <= tolerance && error < bestError) {
        best = i;
        bestError = error;
      }
    }
    if (best >= 0) {
      used.add(best);
      matched += 1;
    }
  }
  return prf(matched, prediction.length, reference.length);
}

export function frameMetrics(reference, prediction) {
  const pitches = new Set([...reference.map(n => n.pitchMidi), ...prediction.map(n => n.pitchMidi)]);
  let refDuration = 0;
  let predDuration = 0;
  let intersection = 0;

  for (const pitch of pitches) {
    const refIntervals = unionIntervals(reference.filter(n => n.pitchMidi === pitch).map(toInterval));
    const predIntervals = unionIntervals(prediction.filter(n => n.pitchMidi === pitch).map(toInterval));
    refDuration += totalDuration(refIntervals);
    predDuration += totalDuration(predIntervals);
    intersection += intersectionDuration(refIntervals, predIntervals);
  }

  const precision = predDuration ? intersection / predDuration : (refDuration ? 0 : 1);
  const recall = refDuration ? intersection / refDuration : (predDuration ? 0 : 1);
  return {
    matchedSeconds: intersection,
    predictedSeconds: predDuration,
    referenceSeconds: refDuration,
    precision,
    recall,
    f1: f1(precision, recall),
  };
}

function toInterval(note) {
  const start = Number(note.startTimeSeconds);
  const end = start + Math.max(0, Number(note.durationSeconds));
  return [start, end];
}

function unionIntervals(intervals) {
  const sorted = intervals.filter(([a, b]) => Number.isFinite(a) && Number.isFinite(b) && b > a).sort((a, b) => a[0] - b[0]);
  const out = [];
  for (const interval of sorted) {
    const last = out.at(-1);
    if (!last || interval[0] > last[1]) out.push([...interval]);
    else last[1] = Math.max(last[1], interval[1]);
  }
  return out;
}

function totalDuration(intervals) {
  return intervals.reduce((sum, [a, b]) => sum + (b - a), 0);
}

function intersectionDuration(a, b) {
  let i = 0;
  let j = 0;
  let sum = 0;
  while (i < a.length && j < b.length) {
    const start = Math.max(a[i][0], b[j][0]);
    const end = Math.min(a[i][1], b[j][1]);
    if (end > start) sum += end - start;
    if (a[i][1] < b[j][1]) i += 1;
    else j += 1;
  }
  return sum;
}

function prf(matched, predicted, reference) {
  const precision = predicted ? matched / predicted : (reference ? 0 : 1);
  const recall = reference ? matched / reference : (predicted ? 0 : 1);
  return { matched, predicted, reference, precision, recall, f1: f1(precision, recall) };
}

function f1(precision, recall) {
  return precision + recall ? (2 * precision * recall) / (precision + recall) : 0;
}

export function objectiveScore(metrics) {
  // Drum hallucinations must hurt even on drumless references (e.g. piano attacks misread as drums).
  const drumWeight = metrics.drums.reference > 0 || metrics.drums.predicted > 0 ? 0.10 : 0;
  const tonalWeight = 1 - drumWeight;
  const tonal = metrics.onset.f1 * 0.55 + metrics.offset.f1 * 0.20 + metrics.frame.f1 * 0.25;
  return tonal * tonalWeight + metrics.drums.f1 * drumWeight;
}

export function aggregateScores(items) {
  if (!items.length) return null;
  const macro = key => items.reduce((sum, item) => sum + key(item), 0) / items.length;
  return {
    count: items.length,
    onsetF1: macro(item => item.metrics.onset.f1),
    offsetF1: macro(item => item.metrics.offset.f1),
    frameF1: macro(item => item.metrics.frame.f1),
    drumsF1: macro(item => item.metrics.drums.f1),
    objective: macro(item => item.metrics.objective),
  };
}
