export const DEFAULT_SPECIALIST_POSTPROCESS = Object.freeze({
  framesPerSecond: 100,
  beginNote: 21,
  classes: 88,
  onsetThreshold: 0.3,
  offsetThreshold: 0.3,
  frameThreshold: 0.1,
  velocityScale: 128,
  maxNoteFrames: 600,
});

export function postprocessSpecialistOutputs(outputs, options = {}) {
  const config = { ...DEFAULT_SPECIALIST_POSTPROCESS, ...options };
  const onset = requireMatrix(outputs, 'reg_onset_output');
  const offset = requireMatrix(outputs, 'reg_offset_output');
  const frame = requireMatrix(outputs, 'frame_output');
  const velocity = requireMatrix(outputs, 'velocity_output');
  const frames = Math.min(onset.dims[0], offset.dims[0], frame.dims[0], velocity.dims[0]);
  const classes = Math.min(config.classes, onset.dims[1], offset.dims[1], frame.dims[1], velocity.dims[1]);

  const onsetPeaks = binarizeRegression(onset, frames, classes, config.onsetThreshold, 2);
  const offsetPeaks = binarizeRegression(offset, frames, classes, config.offsetThreshold, 4);
  const notes = [];

  for (let pitchClass = 0; pitchClass < classes; pitchClass += 1) {
    const detected = detectPitchEvents({
      pitchClass,
      frames,
      frame,
      velocity,
      onsetPeaks,
      offsetPeaks,
      frameThreshold: config.frameThreshold,
      maxNoteFrames: config.maxNoteFrames,
    });
    for (const event of detected) {
      const onsetSeconds = (event.begin + event.onsetShift) / config.framesPerSecond;
      const offsetSeconds = Math.max(onsetSeconds + 1 / config.framesPerSecond,
        (event.end + event.offsetShift) / config.framesPerSecond);
      const normalizedVelocity = clamp(event.velocity, 0, 1);
      notes.push({
        pitchMidi: pitchClass + config.beginNote,
        startTimeSeconds: Math.max(0, onsetSeconds),
        durationSeconds: Math.max(0.01, offsetSeconds - onsetSeconds),
        amplitude: normalizedVelocity,
        velocity: Math.max(1, Math.min(127, Math.floor(normalizedVelocity * config.velocityScale))),
        confidence: clamp(event.confidence, 0, 1),
        source: options.source ?? 'specialist-crnn',
        sources: [options.source ?? 'specialist-crnn'],
        agreement: 1,
        instrument: options.instrument ?? 'harmony',
      });
    }
  }

  notes.sort((a, b) => a.startTimeSeconds - b.startTimeSeconds || a.pitchMidi - b.pitchMidi);
  return notes;
}

export function binarizeRegression(matrix, frames, classes, threshold, neighbour) {
  const binary = new Uint8Array(frames * classes);
  const shifts = new Float32Array(frames * classes);

  for (let pitch = 0; pitch < classes; pitch += 1) {
    for (let frame = neighbour; frame < frames - neighbour; frame += 1) {
      const center = valueAt(matrix, frame, pitch);
      if (center <= threshold || !isMonotonicNeighbour(matrix, frame, pitch, neighbour)) continue;
      const before = valueAt(matrix, frame - 1, pitch);
      const after = valueAt(matrix, frame + 1, pitch);
      let denominator;
      let shift;
      if (before > after) {
        denominator = center - after;
        shift = Math.abs(denominator) < 1e-12 ? 0 : (after - before) / denominator / 2;
      } else {
        denominator = center - before;
        shift = Math.abs(denominator) < 1e-12 ? 0 : (after - before) / denominator / 2;
      }
      const index = frame * classes + pitch;
      binary[index] = 1;
      shifts[index] = Number.isFinite(shift) ? clamp(shift, -1, 1) : 0;
    }
  }
  return { binary, shifts, frames, classes };
}

function detectPitchEvents({ pitchClass, frames, frame, velocity, onsetPeaks, offsetPeaks, frameThreshold, maxNoteFrames }) {
  const out = [];
  let begin = null;
  let frameDisappear = null;
  let offsetOccur = null;

  const finish = (end, offsetShift = 0) => {
    if (begin === null) return;
    const beginIndex = begin * onsetPeaks.classes + pitchClass;
    const onsetConfidence = valueAtRaw(onsetPeaks.source ?? null, begin, pitchClass);
    const frameConfidence = meanFrameConfidence(frame, begin, end, pitchClass);
    out.push({
      begin,
      end: Math.max(begin + 1, end),
      onsetShift: onsetPeaks.shifts[beginIndex] ?? 0,
      offsetShift,
      velocity: valueAt(velocity, begin, pitchClass),
      confidence: Number.isFinite(onsetConfidence)
        ? onsetConfidence * 0.65 + frameConfidence * 0.35
        : frameConfidence,
    });
    begin = null;
    frameDisappear = null;
    offsetOccur = null;
  };

  for (let i = 0; i < frames; i += 1) {
    const index = i * onsetPeaks.classes + pitchClass;
    if (onsetPeaks.binary[index] === 1) {
      if (begin !== null) finish(Math.max(i - 1, 0), 0);
      begin = i;
    }

    if (begin !== null && i > begin) {
      if (valueAt(frame, i, pitchClass) <= frameThreshold && frameDisappear === null) frameDisappear = i;
      if (offsetPeaks.binary[index] === 1 && offsetOccur === null) offsetOccur = i;

      if (frameDisappear !== null) {
        let end;
        if (offsetOccur !== null && offsetOccur - begin > frameDisappear - offsetOccur) end = offsetOccur;
        else end = frameDisappear;
        const offsetIndex = end * offsetPeaks.classes + pitchClass;
        finish(end, offsetPeaks.shifts[offsetIndex] ?? 0);
      }

      if (begin !== null && (i - begin >= maxNoteFrames || i === frames - 1)) {
        const offsetIndex = i * offsetPeaks.classes + pitchClass;
        finish(i, offsetPeaks.shifts[offsetIndex] ?? 0);
      }
    }
  }
  return out;
}

function isMonotonicNeighbour(matrix, frame, pitch, neighbour) {
  for (let i = 0; i < neighbour; i += 1) {
    if (valueAt(matrix, frame - i, pitch) < valueAt(matrix, frame - i - 1, pitch)) return false;
    if (valueAt(matrix, frame + i, pitch) < valueAt(matrix, frame + i + 1, pitch)) return false;
  }
  return true;
}

function meanFrameConfidence(matrix, begin, end, pitch) {
  let sum = 0;
  let count = 0;
  for (let i = begin; i <= end && i < matrix.dims[0]; i += 1) {
    sum += valueAt(matrix, i, pitch);
    count += 1;
  }
  return count ? sum / count : 0;
}

function requireMatrix(outputs, key) {
  const tensor = outputs?.[key];
  if (!tensor?.data || !Array.isArray(tensor.dims) || tensor.dims.length < 2) {
    throw new Error(`Specialist output is missing ${key}.`);
  }
  const dims = tensor.dims.length === 3 && tensor.dims[0] === 1 ? tensor.dims.slice(1) : tensor.dims.slice(-2);
  return { data: tensor.data, dims };
}

function valueAt(matrix, frame, pitch) {
  if (!matrix || frame < 0 || pitch < 0 || frame >= matrix.dims[0] || pitch >= matrix.dims[1]) return 0;
  return Number(matrix.data[frame * matrix.dims[1] + pitch] ?? 0);
}

function valueAtRaw(matrix, frame, pitch) {
  return matrix ? valueAt(matrix, frame, pitch) : Number.NaN;
}

function clamp(value, min, max) { return Math.max(min, Math.min(max, Number(value) || 0)); }
