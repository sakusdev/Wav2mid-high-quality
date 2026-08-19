import { transcribe } from './transcribe.js';

const DEMUCS_SAMPLE_RATE = 44100;
const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
let processorPromise = null;
let separatorBackend = 'uninitialized';
let activeProgress = () => {};

export async function transcribeNeural(audioBuffer, options = {}, onProgress = () => {}) {
  activeProgress = onProgress;
  onProgress({ stage: 'neural-prepare', value: 0.01, detail: '44.1 kHz stereo normalization' });
  const stereo = normalizeStereo44100(audioBuffer);

  onProgress({ stage: 'neural-load', value: 0.03, detail: 'HTDemucs runtime/model' });
  const processor = await getProcessor();

  onProgress({ stage: 'neural-separate', value: 0.16, detail: `HTDemucs · ${separatorBackend}` });
  const stems = await processor.separate(stereo.left, stereo.right);
  onProgress({ stage: 'neural-separate', value: 0.46, detail: 'drums / bass / other / vocals ready' });

  const passSpecs = [
    { name: 'mix', stem: stereo, weight: 0.92, minPitch: options.minPitch ?? 21, maxPitch: options.maxPitch ?? 108 },
    { name: 'other', stem: stems.other, weight: 1.18, minPitch: Math.max(28, Number(options.minPitch ?? 21)), maxPitch: options.maxPitch ?? 108 },
    { name: 'bass', stem: stems.bass, weight: 1.30, minPitch: Math.max(21, Number(options.minPitch ?? 21)), maxPitch: Math.min(64, Number(options.maxPitch ?? 108)) },
    { name: 'vocals', stem: stems.vocals, weight: 1.14, minPitch: Math.max(36, Number(options.minPitch ?? 21)), maxPitch: Math.min(96, Number(options.maxPitch ?? 108)) },
  ];

  const passResults = {};
  const candidates = [];
  for (let index = 0; index < passSpecs.length; index += 1) {
    const spec = passSpecs[index];
    const pseudoBuffer = stereoLikeToAudioBuffer(spec.stem);
    const result = await transcribe(pseudoBuffer, {
      mode: 'fast',
      sensitivity: Number(options.sensitivity ?? 1),
      minPitch: spec.minPitch,
      maxPitch: spec.maxPitch,
    }, ({ value, detail }) => {
      const mapped = 0.48 + ((index + Math.max(0, Math.min(1, value ?? 0))) / passSpecs.length) * 0.42;
      onProgress({ stage: 'neural-amt', value: mapped, detail: `${spec.name}${detail ? ` · ${detail}` : ''}` });
    });
    passResults[spec.name] = result;
    for (const note of result.notes) {
      candidates.push({
        ...note,
        source: spec.name,
        sources: [spec.name],
        agreement: 1,
        neuralWeight: spec.weight,
        confidence: clamp((note.confidence ?? note.amplitude) * spec.weight, 0, 1),
        instrument: spec.name === 'bass' ? 'bass' : 'harmony',
      });
    }
  }

  onProgress({ stage: 'neural-ensemble', value: 0.91, detail: `${candidates.length} stem candidates` });
  const harmonicReference = passResults.other?.chords?.length ? passResults.other : passResults.mix;
  let notes = fuseStemCandidates(candidates);
  notes = contextCorrect(notes, harmonicReference?.chords ?? [], harmonicReference?.keyDetail ?? passResults.mix?.keyDetail);
  notes = suppressWeakHarmonics(notes);
  notes = mergeAdjacent(notes, 0.05);

  onProgress({ stage: 'neural-drums', value: 0.95, detail: 'isolated drum-stem onset classification' });
  const drums = detectDrums(stems.drums, DEMUCS_SAMPLE_RATE);
  const tempo = harmonicReference?.tempo ?? passResults.mix?.tempo ?? 120;
  const key = harmonicReference?.key ?? passResults.mix?.key ?? '—';
  const keyDetail = harmonicReference?.keyDetail ?? passResults.mix?.keyDetail;
  const chords = harmonicReference?.chords ?? passResults.mix?.chords ?? [];
  const stats = buildStats(notes, drums, audioBuffer.duration, candidates.length);
  onProgress({ stage: 'done', value: 1, detail: `${notes.length} notes · ${drums.length} drums` });

  return {
    notes,
    rawNotes: candidates,
    drums,
    tempo,
    key,
    keyDetail,
    chords,
    stats,
    pipeline: {
      mode: 'NEURAL HQ',
      backend: passResults.mix?.pipeline?.backend ?? 'unknown',
      separatorBackend,
      sourceSeparation: 'HTDemucs 4-stem neural (ONNX Runtime Web)',
      stemPasses: ['mix', 'other', 'bass', 'vocals'],
      ensemble: true,
      contextDecoder: true,
      drumTranscription: true,
      chunks: passResults.mix?.pipeline?.chunks ?? 1,
      neural: true,
      model: 'htdemucs_embedded.onnx',
    },
  };
}

async function getProcessor() {
  if (processorPromise) return processorPromise;
  processorPromise = (async () => {
    const gpuUsable = await canUseWebGPU();
    let ort;
    if (gpuUsable) {
      try {
        ort = await import('onnxruntime-web/webgpu');
        separatorBackend = 'webgpu';
      } catch (error) {
        console.warn('ONNX Runtime WebGPU import failed; falling back to WASM.', error);
      }
    }
    if (!ort) {
      ort = await import('onnxruntime-web');
      separatorBackend = 'wasm';
    }

    ort.env.wasm.wasmPaths = '/ort-wasm/';
    ort.env.wasm.numThreads = globalThis.crossOriginIsolated
      ? Math.max(1, Math.min(8, Number(navigator.hardwareConcurrency ?? 4)))
      : 1;
    if (ort.env.webgpu && separatorBackend === 'webgpu') {
      ort.env.webgpu.powerPreference = 'high-performance';
    }

    const { DemucsProcessor, CONSTANTS } = await import('demucs-web');
    const processor = new DemucsProcessor({
      ort,
      sessionOptions: {
        executionProviders: separatorBackend === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm'],
        enableCpuMemArena: false,
        enableMemPattern: false,
        graphOptimizationLevel: 'basic',
      },
      onProgress: info => {
        const progress = typeof info === 'number' ? info : Number(info?.progress ?? 0);
        activeProgress({
          stage: 'neural-separate',
          value: 0.16 + clamp(progress, 0, 1) * 0.30,
          detail: typeof info === 'object' && info ? `segment ${info.currentSegment ?? '?'} / ${info.totalSegments ?? '?'}` : 'HTDemucs',
        });
      },
      onDownloadProgress: (loaded, total) => {
        const ratio = total > 0 ? loaded / total : 0;
        const loadedMB = loaded / 1024 / 1024;
        const totalMB = total / 1024 / 1024;
        activeProgress({
          stage: 'neural-load',
          value: 0.03 + clamp(ratio, 0, 1) * 0.12,
          detail: total > 0 ? `${loadedMB.toFixed(1)} / ${totalMB.toFixed(1)} MB` : `${loadedMB.toFixed(1)} MB`,
        });
      },
      onLog: (phase, message) => console.debug(`[HTDemucs:${phase}] ${message}`),
    });
    await processor.loadModel(CONSTANTS.DEFAULT_MODEL_URL);
    return processor;
  })().catch(error => {
    processorPromise = null;
    throw error;
  });
  return processorPromise;
}

async function canUseWebGPU() {
  if (!navigator?.gpu) return false;
  try {
    return Boolean(await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' }));
  } catch {
    return false;
  }
}

function normalizeStereo44100(audioBuffer) {
  const sourceRate = Number(audioBuffer.sampleRate);
  const sourceLength = Number(audioBuffer.length);
  if (!sourceRate || !sourceLength || !audioBuffer.numberOfChannels) throw new Error('Decoded audio is empty.');
  const leftSource = audioBuffer.getChannelData(0);
  const rightSource = audioBuffer.numberOfChannels > 1 ? audioBuffer.getChannelData(1) : leftSource;
  if (sourceRate === DEMUCS_SAMPLE_RATE) {
    return { left: new Float32Array(leftSource), right: new Float32Array(rightSource), sampleRate: DEMUCS_SAMPLE_RATE };
  }
  const targetLength = Math.max(1, Math.round(sourceLength * DEMUCS_SAMPLE_RATE / sourceRate));
  return {
    left: linearResample(leftSource, targetLength),
    right: linearResample(rightSource, targetLength),
    sampleRate: DEMUCS_SAMPLE_RATE,
  };
}

function linearResample(source, targetLength) {
  const output = new Float32Array(targetLength);
  if (source.length === 1) { output.fill(source[0]); return output; }
  const ratio = (source.length - 1) / Math.max(1, targetLength - 1);
  for (let i = 0; i < targetLength; i += 1) {
    const pos = i * ratio;
    const left = Math.floor(pos);
    const right = Math.min(source.length - 1, left + 1);
    const frac = pos - left;
    output[i] = source[left] + (source[right] - source[left]) * frac;
  }
  return output;
}

function stereoLikeToAudioBuffer(stem) {
  const left = stem.left;
  const right = stem.right ?? left;
  const length = Math.min(left.length, right.length);
  return {
    sampleRate: DEMUCS_SAMPLE_RATE,
    numberOfChannels: 2,
    length,
    duration: length / DEMUCS_SAMPLE_RATE,
    getChannelData(channel) { return channel === 0 ? left : right; },
  };
}

function fuseStemCandidates(candidates) {
  const byPitch = new Map();
  for (const note of candidates) {
    const list = byPitch.get(note.pitchMidi) ?? [];
    list.push(note);
    byPitch.set(note.pitchMidi, list);
  }
  const fused = [];
  for (const [pitch, list] of byPitch) {
    list.sort((a, b) => a.startTimeSeconds - b.startTimeSeconds);
    let cluster = [];
    const flush = () => {
      if (!cluster.length) return;
      const sources = new Set(cluster.map(note => note.source));
      let weightSum = 0;
      let startSum = 0;
      let endSum = 0;
      let amplitude = 0;
      let probabilityMiss = 1;
      let strongest = cluster[0];
      for (const note of cluster) {
        const confidence = clamp(note.confidence ?? note.amplitude, 0.02, 1);
        const weight = Math.max(0.04, confidence * (note.neuralWeight ?? 1));
        weightSum += weight;
        startSum += note.startTimeSeconds * weight;
        endSum += (note.startTimeSeconds + note.durationSeconds) * weight;
        amplitude = Math.max(amplitude, note.amplitude);
        probabilityMiss *= 1 - clamp(confidence * 0.76, 0.02, 0.93);
        if ((note.confidence ?? 0) > (strongest.confidence ?? 0)) strongest = note;
      }
      const start = startSum / weightSum;
      const end = Math.max(start + 0.02, endSum / weightSum);
      const agreement = sources.size;
      const confidence = clamp(1 - probabilityMiss + Math.max(0, agreement - 1) * 0.055, 0, 1);
      fused.push({
        pitchMidi: pitch,
        startTimeSeconds: start,
        durationSeconds: end - start,
        amplitude: clamp(Math.sqrt(amplitude) * (0.92 + agreement * 0.03), 0.05, 1),
        confidence,
        agreement,
        sources: [...sources],
        source: agreement > 1 ? 'neural-ensemble' : [...sources][0],
        instrument: sources.has('bass') && pitch <= 64 ? 'bass' : 'harmony',
        pitchBends: strongest.pitchBends,
      });
      cluster = [];
    };
    for (const note of list) {
      if (!cluster.length) { cluster.push(note); continue; }
      const center = cluster.reduce((sum, item) => sum + item.startTimeSeconds, 0) / cluster.length;
      if (Math.abs(note.startTimeSeconds - center) <= 0.085) cluster.push(note);
      else { flush(); cluster.push(note); }
    }
    flush();
  }
  return fused.sort((a, b) => a.startTimeSeconds - b.startTimeSeconds || a.pitchMidi - b.pitchMidi);
}

function contextCorrect(notes, chords, keyDetail) {
  const scaleMajor = new Set([0, 2, 4, 5, 7, 9, 11]);
  const scaleMinor = new Set([0, 2, 3, 5, 7, 8, 10]);
  const scale = keyDetail?.mode === 'minor' ? scaleMinor : scaleMajor;
  const root = Number(keyDetail?.root ?? 0);
  return notes.filter(note => {
    let score = note.confidence ?? note.amplitude;
    if ((note.agreement ?? 1) >= 2) score += 0.07 * Math.min(3, note.agreement - 1);
    const relative = ((note.pitchMidi % 12) - root + 12) % 12;
    score += scale.has(relative) ? 0.035 : -0.04;
    const midpoint = note.startTimeSeconds + note.durationSeconds / 2;
    const chord = chords.find(item => midpoint >= item.start && midpoint < item.end);
    if (chord?.pitchClasses) score += chord.pitchClasses.includes(note.pitchMidi % 12) ? 0.055 : -0.025;
    if (note.sources?.includes('vocals')) score += 0.025;
    note.confidence = clamp(score, 0, 1);
    return note.confidence >= 0.18 || (note.agreement ?? 1) >= 2;
  });
}

function suppressWeakHarmonics(notes) {
  const intervals = new Set([12, 19, 24, 28, 31, 36]);
  const remove = new Set();
  for (let i = 0; i < notes.length; i += 1) {
    const low = notes[i];
    for (let j = i + 1; j < notes.length; j += 1) {
      const high = notes[j];
      if (high.startTimeSeconds - low.startTimeSeconds > 0.045) break;
      if (!intervals.has(high.pitchMidi - low.pitchMidi) || high.pitchMidi < 76) continue;
      if ((high.agreement ?? 1) >= 2 || high.sources?.includes('vocals')) continue;
      if ((high.confidence ?? 0) < (low.confidence ?? 0) * 0.55) remove.add(j);
    }
  }
  return notes.filter((_, index) => !remove.has(index));
}

function mergeAdjacent(notes, gap) {
  const byPitch = new Map();
  for (const note of notes) {
    const list = byPitch.get(note.pitchMidi) ?? [];
    list.push(note);
    byPitch.set(note.pitchMidi, list);
  }
  const output = [];
  for (const list of byPitch.values()) {
    list.sort((a, b) => a.startTimeSeconds - b.startTimeSeconds);
    let current = { ...list[0] };
    for (let i = 1; i < list.length; i += 1) {
      const next = list[i];
      const currentEnd = current.startTimeSeconds + current.durationSeconds;
      if (next.startTimeSeconds - currentEnd <= gap && next.startTimeSeconds - currentEnd >= -0.08) {
        const nextEnd = next.startTimeSeconds + next.durationSeconds;
        current.durationSeconds = Math.max(currentEnd, nextEnd) - current.startTimeSeconds;
        current.amplitude = Math.max(current.amplitude, next.amplitude);
        current.confidence = Math.max(current.confidence ?? 0, next.confidence ?? 0);
        current.sources = [...new Set([...(current.sources ?? []), ...(next.sources ?? [])])];
        current.agreement = current.sources.length;
      } else {
        output.push(current);
        current = { ...next };
      }
    }
    output.push(current);
  }
  return output.sort((a, b) => a.startTimeSeconds - b.startTimeSeconds || a.pitchMidi - b.pitchMidi);
}

function detectDrums(stem, sampleRate) {
  const left = stem.left;
  const right = stem.right ?? left;
  const length = Math.min(left.length, right.length);
  const mono = new Float32Array(length);
  for (let i = 0; i < length; i += 1) mono[i] = (left[i] + right[i]) * 0.5;
  const low = onePoleLowPass(mono, sampleRate, 180);
  const high = onePoleHighPass(mono, sampleRate, 2600);
  const frameSize = 2048;
  const hop = 512;
  const frames = [];
  let previousRms = 0;
  for (let start = 0; start + frameSize <= length; start += hop) {
    let energy = 0, lowEnergy = 0, highEnergy = 0, crossings = 0;
    for (let i = 0; i < frameSize; i += 1) {
      const index = start + i;
      const x = mono[index];
      energy += x * x;
      lowEnergy += low[index] * low[index];
      highEnergy += high[index] * high[index];
      if (i > 0 && ((x < 0) !== (mono[index - 1] < 0))) crossings += 1;
    }
    const rms = Math.sqrt(energy / frameSize);
    frames.push({ start, rms, onset: Math.max(0, rms - previousRms * 0.86), low: Math.sqrt(lowEnergy / frameSize), high: Math.sqrt(highEnergy / frameSize), zcr: crossings / frameSize });
    previousRms = rms;
  }
  if (!frames.length) return [];
  const strengths = frames.map(item => item.onset).sort((a, b) => a - b);
  const p70 = strengths[Math.floor(strengths.length * 0.70)] ?? 0;
  const mean = strengths.reduce((sum, value) => sum + value, 0) / strengths.length;
  const threshold = Math.max(0.003, p70 * 1.35, mean * 1.6);
  const maxStrength = Math.max(threshold, ...strengths);
  const events = [];
  let last = -1;
  for (let i = 1; i < frames.length - 1; i += 1) {
    const frame = frames[i];
    if (frame.onset < threshold || frame.onset < frames[i - 1].onset || frame.onset < frames[i + 1].onset) continue;
    const time = frame.start / sampleRate;
    if (time - last < 0.055) continue;
    const lowRatio = frame.low / Math.max(1e-6, frame.rms);
    const highRatio = frame.high / Math.max(1e-6, frame.rms);
    let midi = 38, name = 'Snare';
    if (lowRatio > 0.60) { midi = 36; name = 'Kick'; }
    else if (highRatio > 0.50 || frame.zcr > 0.23) { midi = 42; name = 'Hi-hat'; }
    events.push({ midi, name, time, duration: 0.055, velocity: clamp(0.30 + 0.70 * frame.onset / maxStrength, 0.12, 1) });
    last = time;
  }
  return events;
}

function onePoleLowPass(input, sampleRate, cutoff) {
  const output = new Float32Array(input.length);
  const dt = 1 / sampleRate, rc = 1 / (2 * Math.PI * cutoff), alpha = dt / (rc + dt);
  let y = 0;
  for (let i = 0; i < input.length; i += 1) { y += alpha * (input[i] - y); output[i] = y; }
  return output;
}

function onePoleHighPass(input, sampleRate, cutoff) {
  const output = new Float32Array(input.length);
  const dt = 1 / sampleRate, rc = 1 / (2 * Math.PI * cutoff), alpha = rc / (rc + dt);
  let y = 0, previous = input[0] ?? 0;
  for (let i = 0; i < input.length; i += 1) {
    const x = input[i]; y = alpha * (y + x - previous); previous = x; output[i] = y;
  }
  return output;
}

function buildStats(notes, drums, duration, candidateCount) {
  const events = [];
  for (const note of notes) {
    events.push([note.startTimeSeconds, 1]);
    events.push([note.startTimeSeconds + note.durationSeconds, -1]);
  }
  events.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  let poly = 0, maxPolyphony = 0;
  for (const [, delta] of events) { poly += delta; maxPolyphony = Math.max(maxPolyphony, poly); }
  const ensembleBacked = notes.filter(note => (note.agreement ?? 1) >= 2).length;
  return {
    duration,
    noteCount: notes.length,
    drumCount: drums.length,
    candidateCount,
    maxPolyphony,
    lowestNote: notes.length ? Math.min(...notes.map(note => note.pitchMidi)) : null,
    highestNote: notes.length ? Math.max(...notes.map(note => note.pitchMidi)) : null,
    ensembleBacked,
    ensembleRatio: notes.length ? ensembleBacked / notes.length : 0,
    chunks: 1,
    passCount: 4,
  };
}

export function neuralModelDescription() {
  return {
    name: 'HTDemucs 4-stem',
    sampleRate: DEMUCS_SAMPLE_RATE,
    tracks: ['drums', 'bass', 'other', 'vocals'],
    sizeHint: '~172 MB first download',
  };
}

export function midiPitchName(midi) {
  return midi == null ? '—' : `${NOTE_NAMES[midi % 12]}${Math.floor(midi / 12) - 1}`;
}

function clamp(value, min, max) { return Math.min(max, Math.max(min, value)); }
