const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
const MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];
const SUSPECT_HARMONIC_INTERVALS = new Set([12, 19, 24, 28, 31, 36]);
const CHORD_TEMPLATES = [
  ['maj', [0, 4, 7]], ['min', [0, 3, 7]], ['dim', [0, 3, 6]], ['aug', [0, 4, 8]],
  ['sus2', [0, 2, 7]], ['sus4', [0, 5, 7]], ['7', [0, 4, 7, 10]], ['maj7', [0, 4, 7, 11]],
  ['m7', [0, 3, 7, 10]], ['mMaj7', [0, 3, 7, 11]], ['6', [0, 4, 7, 9]], ['m6', [0, 3, 7, 9]],
  ['add9', [0, 2, 4, 7]], ['madd9', [0, 2, 3, 7]], ['9', [0, 2, 4, 7, 10]], ['m9', [0, 2, 3, 7, 10]],
  ['maj9', [0, 2, 4, 7, 11]], ['m11', [0, 2, 3, 5, 7, 10]],
];

export function refineResult(input, duration) {
  if (!input?.notes?.length) return input;
  const result = { ...input, pipeline: { ...(input.pipeline ?? {}) } };
  const mode = String(result.pipeline.mode ?? '').toUpperCase();
  const ensemble = Boolean(result.pipeline.ensemble);
  let notes = result.notes.map(note => ({ ...note, sources: [...(note.sources ?? [])] }));

  if (ensemble) {
    const mergeGap = mode.includes('INSANE') ? 0.09 : 0.08;
    notes = mergeAdjacent(notes, mergeGap);
    notes = pruneShortHighHarmonics(notes);
  }

  notes.sort((a, b) => a.startTimeSeconds - b.startTimeSeconds || a.pitchMidi - b.pitchMidi);
  const keyDetail = estimateKeyDetail(notes);
  const tempo = Number.isFinite(result.tempo) ? result.tempo : 120;
  const chords = estimateChords(notes, Number(duration ?? result.stats?.duration ?? 0), tempo);

  result.notes = notes;
  result.keyDetail = keyDetail;
  result.key = keyDetail.label;
  result.chords = chords;
  result.stats = rebuildStats(result, notes, duration);
  result.pipeline.qualityRefiner = 'realworld-v1';
  result.pipeline.chordWindow = 'tempo-aware-two-beat';
  return result;
}

function mergeAdjacent(notes, maxGap) {
  const byPitch = new Map();
  for (const note of notes) {
    const list = byPitch.get(note.pitchMidi) ?? [];
    list.push(note);
    byPitch.set(note.pitchMidi, list);
  }
  const output = [];
  for (const list of byPitch.values()) {
    list.sort((a, b) => a.startTimeSeconds - b.startTimeSeconds);
    let current = { ...list[0], sources: [...(list[0]?.sources ?? [])] };
    for (let i = 1; i < list.length; i += 1) {
      const next = list[i];
      const currentEnd = current.startTimeSeconds + current.durationSeconds;
      const gap = next.startTimeSeconds - currentEnd;
      if (gap >= -0.08 && gap <= maxGap) {
        const nextEnd = next.startTimeSeconds + next.durationSeconds;
        current.durationSeconds = Math.max(currentEnd, nextEnd) - current.startTimeSeconds;
        current.amplitude = Math.max(current.amplitude, next.amplitude);
        current.confidence = Math.max(current.confidence ?? 0, next.confidence ?? 0);
        current.sources = [...new Set([...(current.sources ?? []), ...(next.sources ?? [])])];
        current.agreement = Math.max(current.agreement ?? 1, next.agreement ?? 1, current.sources.length);
        if (current.pitchBends && next.pitchBends) current.pitchBends = [...current.pitchBends, ...next.pitchBends];
      } else {
        output.push(current);
        current = { ...next, sources: [...(next.sources ?? [])] };
      }
    }
    if (current) output.push(current);
  }
  return output;
}

function pruneShortHighHarmonics(notes) {
  const removed = new Set();
  for (let highIndex = 0; highIndex < notes.length; highIndex += 1) {
    const high = notes[highIndex];
    if (high.pitchMidi < 88 || high.durationSeconds > 0.18 || (high.agreement ?? 1) > 2) continue;
    const highConfidence = high.confidence ?? high.amplitude ?? 0;
    for (let lowIndex = 0; lowIndex < notes.length; lowIndex += 1) {
      if (lowIndex === highIndex) continue;
      const low = notes[lowIndex];
      const interval = high.pitchMidi - low.pitchMidi;
      if (!SUSPECT_HARMONIC_INTERVALS.has(interval)) continue;
      if (Math.abs(high.startTimeSeconds - low.startTimeSeconds) > 0.08) continue;
      if (low.durationSeconds < high.durationSeconds * 1.5) continue;
      const lowConfidence = low.confidence ?? low.amplitude ?? 0;
      if (highConfidence <= lowConfidence) {
        removed.add(highIndex);
        break;
      }
    }
  }
  return notes.filter((_, index) => !removed.has(index));
}

function estimateKeyDetail(notes) {
  if (!notes.length) return { root: 0, mode: 'major', label: '—', score: 0 };
  const histogram = Array(12).fill(0);
  for (const note of notes) {
    histogram[note.pitchMidi % 12] += note.durationSeconds * note.amplitude * (0.7 + 0.3 * (note.confidence ?? 1));
  }

  const candidates = [];
  for (let root = 0; root < 12; root += 1) {
    for (const [profile, mode] of [[MAJOR_PROFILE, 'major'], [MINOR_PROFILE, 'minor']]) {
      let score = 0;
      for (let pc = 0; pc < 12; pc += 1) score += histogram[pc] * profile[(pc - root + 12) % 12];
      candidates.push({ root, mode, score, tonicWeight: histogram[root] });
    }
  }
  candidates.sort((a, b) => b.score - a.score);
  let best = candidates[0];

  // Krumhansl profiles often leave relative major/minor almost tied. When that happens,
  // prefer the relative key whose tonic has clearly more sustained energy instead of
  // letting a tiny floating-point/profile difference flip the displayed key.
  if (best.mode === 'minor') {
    const relativeMajorRoot = (best.root + 3) % 12;
    const relativeMajor = candidates.find(item => item.root === relativeMajorRoot && item.mode === 'major');
    if (relativeMajor && relativeMajor.score >= best.score * 0.992 && relativeMajor.tonicWeight >= best.tonicWeight * 1.15) best = relativeMajor;
  } else {
    const relativeMinorRoot = (best.root + 9) % 12;
    const relativeMinor = candidates.find(item => item.root === relativeMinorRoot && item.mode === 'minor');
    if (relativeMinor && relativeMinor.score >= best.score * 0.992 && relativeMinor.tonicWeight >= best.tonicWeight * 1.15) best = relativeMinor;
  }

  return { ...best, label: `${NOTE_NAMES[best.root]} ${best.mode}` };
}

function estimateChords(notes, duration, tempo) {
  if (!notes.length || !duration) return [];
  const safeTempo = Math.max(60, Math.min(220, Number(tempo) || 120));
  const windowSize = clamp(120 / safeTempo, 0.5, 1.0); // two beats
  const ordered = [...notes].sort((a, b) => a.startTimeSeconds - b.startTimeSeconds);
  const segments = [];
  let previous = null;
  let cursor = 0;

  for (let t = 0; t < duration; t += windowSize) {
    while (cursor < ordered.length && ordered[cursor].startTimeSeconds + ordered[cursor].durationSeconds < t - windowSize) cursor += 1;
    const active = [];
    for (let i = cursor; i < ordered.length; i += 1) {
      const note = ordered[i];
      if (note.startTimeSeconds >= t + windowSize) break;
      if (note.startTimeSeconds + note.durationSeconds > t) active.push(note);
    }
    if (active.length < 2) continue;
    const weights = Array(12).fill(0);
    for (const note of active) {
      const overlapStart = Math.max(t, note.startTimeSeconds);
      const overlapEnd = Math.min(t + windowSize, note.startTimeSeconds + note.durationSeconds);
      const overlap = Math.max(0, overlapEnd - overlapStart);
      weights[note.pitchMidi % 12] += note.amplitude * overlap * (0.72 + 0.28 * (note.confidence ?? 1));
    }
    const chord = bestChord(weights);
    if (!chord || chord.score < 0.56) continue;
    if (previous?.name === chord.name && t - previous.end <= windowSize * 0.55) {
      previous.end = Math.min(duration, t + windowSize);
      previous.confidence = Math.max(previous.confidence, chord.score);
    } else {
      previous = {
        name: chord.name,
        root: chord.root,
        pitchClasses: chord.pitchClasses,
        start: t,
        end: Math.min(duration, t + windowSize),
        confidence: chord.score,
      };
      segments.push(previous);
    }
  }
  return segments.slice(0, 600);
}

function bestChord(weights) {
  const total = weights.reduce((sum, value) => sum + value, 0);
  if (!total) return null;
  let best = null;
  for (let root = 0; root < 12; root += 1) {
    for (const [suffix, intervals] of CHORD_TEMPLATES) {
      const pitchClasses = intervals.map(interval => (root + interval) % 12);
      const pitchClassSet = new Set(pitchClasses);
      const inWeight = weights.reduce((sum, weight, pitchClass) => sum + (pitchClassSet.has(pitchClass) ? weight : 0), 0);
      const coverage = intervals.filter(interval => weights[(root + interval) % 12] > total * 0.022).length / intervals.length;
      const complexityPenalty = Math.max(0, intervals.length - 4) * 0.012;
      const score = (inWeight / total) * 0.72 + coverage * 0.28 - complexityPenalty;
      if (!best || score > best.score) {
        const suffixLabel = suffix === 'maj' ? '' : suffix === 'min' ? 'm' : suffix;
        best = { name: `${NOTE_NAMES[root]}${suffixLabel}`, root, pitchClasses, score };
      }
    }
  }
  return best;
}

function rebuildStats(result, notes, duration) {
  const original = result.stats ?? {};
  const events = [];
  for (const note of notes) {
    events.push([note.startTimeSeconds, 1]);
    events.push([note.startTimeSeconds + note.durationSeconds, -1]);
  }
  events.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  let polyphony = 0;
  let maxPolyphony = 0;
  for (const [, delta] of events) {
    polyphony += delta;
    maxPolyphony = Math.max(maxPolyphony, polyphony);
  }
  const ensembleBacked = notes.filter(note => (note.agreement ?? 1) >= 2).length;
  return {
    ...original,
    duration: Number(duration ?? original.duration ?? 0),
    noteCount: notes.length,
    drumCount: (result.drums ?? []).length,
    maxPolyphony,
    lowestNote: notes.length ? Math.min(...notes.map(note => note.pitchMidi)) : null,
    highestNote: notes.length ? Math.max(...notes.map(note => note.pitchMidi)) : null,
    ensembleBacked,
    ensembleRatio: notes.length ? ensembleBacked / notes.length : 0,
  };
}

function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }
