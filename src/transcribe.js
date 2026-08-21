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
import { createSourceStems } from './separation.js';

const MODEL_URL = '/model/basic-pitch/model.json';
const TARGET_SAMPLE_RATE = 22050;
const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const MAJOR_SCALE = new Set([0, 2, 4, 5, 7, 9, 11]);
const MINOR_SCALE = new Set([0, 2, 3, 5, 7, 8, 10]);

export const MODES = {
  fast: {
    label: 'FAST',
    onsetThreshold: 0.36,
    frameThreshold: 0.34,
    minNoteLength: 7,
    minDuration: 0.07,
    harmonicSuppression: 0.12,
    mergeGap: 0.025,
    chunkSeconds: 28,
    overlapSeconds: 0.65,
    separation: false,
    drums: false,
    context: false,
    passNames: ['mix'],
    contextThreshold: 0.18,
  },
  pro: {
    label: 'PRO',
    onsetThreshold: 0.28,
    frameThreshold: 0.28,
    minNoteLength: 5,
    minDuration: 0.055,
    harmonicSuppression: 0.42,
    mergeGap: 0.045,
    chunkSeconds: 18,
    overlapSeconds: 0.9,
    separation: true,
    drums: true,
    context: true,
    passNames: ['mix', 'harmonic', 'bass'],
    contextThreshold: 0.22,
  },
  insane: {
    label: 'INSANE',
    onsetThreshold: 0.22,
    frameThreshold: 0.23,
    minNoteLength: 4,
    minDuration: 0.045,
    harmonicSuppression: 0.52,
    mergeGap: 0.055,
    chunkSeconds: 14,
    overlapSeconds: 1.1,
    separation: true,
    drums: true,
    context: true,
    passNames: ['mix', 'harmonic', 'bass', 'presence'],
    contextThreshold: 0.18,
  },
};

let basicPitchInstance = null;
let webgpuRegistered = false;

export async function configureBackend(preference = 'auto') {
  setWasmPaths('/tfjs-wasm/');
  const hasWebGPU = typeof navigator !== 'undefined' && Boolean(navigator.gpu);
  const choices = preference === 'auto'
    ? [...(hasWebGPU ? ['webgpu'] : []), 'webgl', 'wasm', 'cpu']
    : [preference, ...(preference === 'webgpu' ? ['webgl'] : []), 'wasm', 'cpu'];
  let lastError;

  for (const backend of [...new Set(choices)]) {
    try {
      if (backend === 'webgpu') {
        if (!hasWebGPU) continue;
        if (!webgpuRegistered) {
          await import('@tensorflow/tfjs-backend-webgpu');
          webgpuRegistered = true;
        }
      }
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
  onProgress({ stage: 'prepare', value: 0.005, detail: '22.05 kHz mono normalization' });
  const normalizedAudio = normalizeAudio(audioBuffer);
  const chunks = makeChunks(normalizedAudio, mode.chunkSeconds, mode.overlapSeconds);
  const candidateNotes = [];
  const drumEvents = [];
  const separationStats = [];
  let completedPasses = 0;
  const totalPasses = chunks.length * mode.passNames.length;

  for (let chunkIndex = 0; chunkIndex < chunks.length; chunkIndex += 1) {
    const chunk = chunks[chunkIndex];
    let stems = { mix: chunk.samples, stats: { method: 'none', frames: 0 } };

    if (mode.separation) {
      const separationBase = 0.02 + (chunkIndex / Math.max(1, chunks.length)) * 0.12;
      onProgress({
        stage: 'separate',
        value: separationBase,
        detail: `chunk ${chunkIndex + 1}/${chunks.length}`,
      });
      stems = createSourceStems(chunk.samples, TARGET_SAMPLE_RATE, options.mode ?? 'pro');
      separationStats.push(stems.stats);
      if (mode.drums) {
        const localDrums = detectDrumEvents(stems.percussive, TARGET_SAMPLE_RATE);
        for (const drum of localDrums) drumEvents.push({ ...drum, time: drum.time + chunk.offsetSeconds });
      }
    }

    for (const passName of mode.passNames) {
      const pass = buildPass(passName, mode, options);
      const samples = stems[passName] ?? stems.mix;
      const passNotes = await inferStem(samples, pass, progress => {
        const overall = 0.14 + ((completedPasses + progress) / Math.max(1, totalPasses)) * 0.70;
        onProgress({
          stage: 'infer',
          value: overall,
          detail: `${passName} · chunk ${chunkIndex + 1}/${chunks.length}`,
        });
      });
      for (const note of passNotes) {
        candidateNotes.push({ ...note, startTimeSeconds: note.startTimeSeconds + chunk.offsetSeconds });
      }
      completedPasses += 1;
      if (typeof tf.nextFrame === 'function') await tf.nextFrame();
    }
  }

  onProgress({ stage: 'ensemble', value: 0.86, detail: `${candidateNotes.length} candidates` });
  let notes = fuseCandidates(candidateNotes);
  notes = postProcessNotes(notes, mode, options);

  const tempoBeforeContext = estimateTempo(notes);
  const keyDetail = estimateKeyDetail(notes);
  const preliminaryChords = estimateChords(notes, audioBuffer.duration);

  if (mode.context) {
    onProgress({ stage: 'context', value: 0.91, detail: 'key/chord/agreement correction' });
    notes = contextCorrect(notes, preliminaryChords, keyDetail, mode);
    notes = mergeAdjacentNotes(notes, mode.mergeGap);
  }

  const drums = dedupeDrums(drumEvents);
  const tempo = estimateTempo(notes.length ? notes : candidateNotes) || tempoBeforeContext || 120;
  const finalKey = estimateKeyDetail(notes);
  const chords = estimateChords(notes, audioBuffer.duration);
  const stats = buildStats(notes, drums, audioBuffer.duration, candidateNotes.length, chunks.length, mode.passNames.length);
  onProgress({ stage: 'done', value: 1, detail: `${notes.length} notes · ${drums.length} drums` });

  return {
    notes,
    rawNotes: candidateNotes,
    drums,
    tempo,
    key: finalKey.label,
    keyDetail: finalKey,
    chords,
    stats,
    pipeline: {
      mode: mode.label,
      backend: tf.getBackend(),
      sourceSeparation: mode.separation ? 'STFT harmonic/percussive soft-mask' : 'off',
      stemPasses: [...mode.passNames],
      ensemble: mode.passNames.length > 1,
      contextDecoder: mode.context,
      drumTranscription: mode.drums,
      chunks: chunks.length,
      separationFrames: separationStats.reduce((sum, item) => sum + (item.frames ?? 0), 0),
    },
  };
}

function buildPass(name, mode, options) {
  const userMin = Number(options.minPitch ?? 21);
  const userMax = Number(options.maxPitch ?? 108);
  const base = {
    name,
    onsetThreshold: mode.onsetThreshold,
    frameThreshold: mode.frameThreshold,
    minNoteLength: mode.minNoteLength,
    minPitch: userMin,
    maxPitch: userMax,
    weight: 1,
  };
  if (name === 'harmonic') return { ...base, onsetThreshold: Math.max(0.16, mode.onsetThreshold - 0.035), frameThreshold: Math.max(0.18, mode.frameThreshold - 0.025), minPitch: Math.max(userMin, 30), weight: 1.12 };
  if (name === 'bass') return { ...base, onsetThreshold: Math.max(0.16, mode.onsetThreshold - 0.055), frameThreshold: Math.max(0.17, mode.frameThreshold - 0.035), minPitch: Math.max(userMin, 21), maxPitch: Math.min(userMax, 62), weight: 1.22 };
  if (name === 'presence') return { ...base, onsetThreshold: Math.max(0.15, mode.onsetThreshold - 0.045), frameThreshold: Math.max(0.17, mode.frameThreshold - 0.035), minPitch: Math.max(userMin, 45), weight: 0.94 };
  return base;
}

async function inferStem(samples, pass, progressCallback, retried = false) {
  if (!basicPitchInstance) basicPitchInstance = new BasicPitch(MODEL_URL);
  const frames = [];
  const onsets = [];
  const contours = [];

  try {
    await basicPitchInstance.evaluateModel(
      samples,
      (frameChunk, onsetChunk, contourChunk) => {
        frames.push(...frameChunk);
        onsets.push(...onsetChunk);
        contours.push(...contourChunk);
      },
      progressCallback,
    );
  } catch (error) {
    if (!retried && tf.getBackend() === 'webgpu') {
      await configureBackend('webgl');
      return inferStem(samples, pass, progressCallback, true);
    }
    throw error;
  }

  const decoded = noteFramesToTime(
    addPitchBendsToNoteEvents(
      contours,
      outputToNotesPoly(frames, onsets, pass.onsetThreshold, pass.frameThreshold, pass.minNoteLength),
    ),
  );

  return decoded
    .filter(note => note.pitchMidi >= pass.minPitch && note.pitchMidi <= pass.maxPitch)
    .filter(note => Number.isFinite(note.startTimeSeconds) && Number.isFinite(note.durationSeconds) && Number.isFinite(note.amplitude))
    .map(note => ({
      ...note,
      source: pass.name,
      sources: [pass.name],
      agreement: 1,
      confidence: clamp(note.amplitude * pass.weight, 0, 1),
      passWeight: pass.weight,
    }));
}

function makeChunks(audio, chunkSeconds, overlapSeconds) {
  const chunkLength = Math.max(1, Math.round(chunkSeconds * TARGET_SAMPLE_RATE));
  const overlap = Math.max(0, Math.round(overlapSeconds * TARGET_SAMPLE_RATE));
  const step = Math.max(1, chunkLength - overlap);
  const chunks = [];
  for (let start = 0; start < audio.length; start += step) {
    const end = Math.min(audio.length, start + chunkLength);
    chunks.push({
      samples: audio.slice(start, end),
      offsetSeconds: start / TARGET_SAMPLE_RATE,
      startSample: start,
      endSample: end,
    });
    if (end >= audio.length) break;
  }
  return chunks;
}

function normalizeAudio(audioBuffer) {
  const sourceRate = audioBuffer.sampleRate;
  const channels = audioBuffer.numberOfChannels;
  const sourceLength = audioBuffer.length;
  if (!sourceRate || !sourceLength || !channels) throw new Error('Decoded audio is empty.');

  const mono = new Float32Array(sourceLength);
  for (let ch = 0; ch < channels; ch += 1) {
    const data = audioBuffer.getChannelData(ch);
    for (let i = 0; i < sourceLength; i += 1) mono[i] += data[i] / channels;
  }
  if (sourceRate === TARGET_SAMPLE_RATE) return mono;

  const targetLength = Math.max(1, Math.round(sourceLength * TARGET_SAMPLE_RATE / sourceRate));
  const output = new Float32Array(targetLength);
  const ratio = sourceRate / TARGET_SAMPLE_RATE;
  for (let i = 0; i < targetLength; i += 1) {
    const pos = i * ratio;
    const left = Math.min(sourceLength - 1, Math.floor(pos));
    const right = Math.min(sourceLength - 1, left + 1);
    const frac = pos - left;
    output[i] = mono[left] + (mono[right] - mono[left]) * frac;
  }
  return output;
}

function fuseCandidates(candidates) {
  const byPitch = new Map();
  for (const note of candidates) {
    const list = byPitch.get(note.pitchMidi) ?? [];
    list.push(note);
    byPitch.set(note.pitchMidi, list);
  }

  const fused = [];
  for (const [pitch, list] of byPitch.entries()) {
    list.sort((a, b) => a.startTimeSeconds - b.startTimeSeconds);
    let cluster = [];
    const flush = () => {
      if (!cluster.length) return;
      let weightSum = 0;
      let start = 0;
      let end = 0;
      let amplitude = 0;
      let probabilityMiss = 1;
      const sources = new Set();
      let strongest = cluster[0];
      for (const note of cluster) {
        const weight = Math.max(0.05, note.passWeight ?? 1) * Math.max(0.03, note.amplitude);
        weightSum += weight;
        start += note.startTimeSeconds * weight;
        end += (note.startTimeSeconds + note.durationSeconds) * weight;
        amplitude = Math.max(amplitude, note.amplitude);
        probabilityMiss *= 1 - clamp((note.confidence ?? note.amplitude) * 0.78, 0.02, 0.92);
        sources.add(note.source ?? 'mix');
        if ((note.confidence ?? 0) > (strongest.confidence ?? 0)) strongest = note;
      }
      const fusedStart = start / weightSum;
      const fusedEnd = Math.max(fusedStart + 0.01, end / weightSum);
      const agreement = sources.size;
      const confidence = clamp(1 - probabilityMiss + Math.min(0.18, (agreement - 1) * 0.07), 0, 1);
      fused.push({
        pitchMidi: pitch,
        startTimeSeconds: fusedStart,
        durationSeconds: fusedEnd - fusedStart,
        amplitude: clamp(amplitude * (1 + Math.min(0.18, (agreement - 1) * 0.07)), 0.03, 1),
        confidence,
        agreement,
        sources: [...sources],
        source: agreement > 1 ? 'ensemble' : [...sources][0],
        pitchBends: strongest.pitchBends,
        instrument: sources.has('bass') && pitch <= 57 ? 'bass' : 'harmony',
      });
      cluster = [];
    };

    for (const note of list) {
      if (!cluster.length) {
        cluster.push(note);
        continue;
      }
      const center = cluster.reduce((sum, n) => sum + n.startTimeSeconds, 0) / cluster.length;
      if (Math.abs(note.startTimeSeconds - center) <= 0.075) cluster.push(note);
      else { flush(); cluster.push(note); }
    }
    flush();
  }
  return fused.sort((a, b) => a.startTimeSeconds - b.startTimeSeconds || a.pitchMidi - b.pitchMidi);
}

function postProcessNotes(rawNotes, mode, options) {
  const minPitch = Number(options.minPitch ?? 21);
  const maxPitch = Number(options.maxPitch ?? 108);
  const sensitivity = Number(options.sensitivity ?? 1);
  const minAmp = Math.max(0.02, 0.065 / sensitivity);
  const minConfidence = Math.max(0.12, 0.20 / sensitivity);

  let notes = rawNotes
    .map(note => ({ ...note }))
    .filter(note => note.pitchMidi >= minPitch && note.pitchMidi <= maxPitch)
    .filter(note => note.durationSeconds >= mode.minDuration)
    .filter(note => note.amplitude >= minAmp)
    .filter(note => (note.confidence ?? note.amplitude) >= minConfidence)
    .sort((a, b) => a.startTimeSeconds - b.startTimeSeconds || a.pitchMidi - b.pitchMidi);

  notes = mergeAdjacentNotes(notes, mode.mergeGap);
  notes = suppressLikelyHarmonics(notes, mode.harmonicSuppression);
  notes = removeNearDuplicates(notes);
  return notes.map(note => ({ ...note, amplitude: clamp(Math.sqrt(note.amplitude), 0.05, 1) }));
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
    list.sort((a, b) => a.startTimeSeconds - b.startTimeSeconds);
    let current = { ...list[0], sources: [...(list[0]?.sources ?? [])] };
    for (let i = 1; i < list.length; i += 1) {
      const next = list[i];
      const end = current.startTimeSeconds + current.durationSeconds;
      const gap = next.startTimeSeconds - end;
      if (gap <= maxGap && gap >= -0.08) {
        const nextEnd = next.startTimeSeconds + next.durationSeconds;
        current.durationSeconds = Math.max(end, nextEnd) - current.startTimeSeconds;
        current.amplitude = Math.max(current.amplitude, next.amplitude);
        current.confidence = Math.max(current.confidence ?? 0, next.confidence ?? 0);
        current.sources = [...new Set([...(current.sources ?? []), ...(next.sources ?? [])])];
        current.agreement = current.sources.length;
        if (current.pitchBends && next.pitchBends) current.pitchBends.push(...next.pitchBends);
      } else {
        merged.push(current);
        current = { ...next, sources: [...(next.sources ?? [])] };
      }
    }
    if (current) merged.push(current);
  }
  return merged.sort((a, b) => a.startTimeSeconds - b.startTimeSeconds || a.pitchMidi - b.pitchMidi);
}

function suppressLikelyHarmonics(notes, strength) {
  if (strength <= 0) return notes;
  const suspectIntervals = new Set([12, 19, 24, 28, 31, 36]);
  const removed = new Set();
  for (let i = 0; i < notes.length; i += 1) {
    const low = notes[i];
    for (let j = i + 1; j < notes.length; j += 1) {
      const high = notes[j];
      if (high.startTimeSeconds - low.startTimeSeconds > 0.045) break;
      const interval = high.pitchMidi - low.pitchMidi;
      if (!suspectIntervals.has(interval) || high.pitchMidi < 74) continue;
      if ((high.agreement ?? 1) >= 2) continue;
      const durationSimilarity = Math.min(low.durationSeconds, high.durationSeconds) / Math.max(low.durationSeconds, high.durationSeconds);
      const confidenceRatio = (high.confidence ?? high.amplitude) / Math.max(0.001, low.confidence ?? low.amplitude);
      const threshold = 0.34 + strength * 0.42;
      if (durationSimilarity > 0.62 && confidenceRatio < threshold) removed.add(j);
    }
  }
  return notes.filter((_, index) => !removed.has(index));
}

function removeNearDuplicates(notes) {
  const kept = [];
  for (const note of notes) {
    const duplicate = kept.find(existing =>
      existing.pitchMidi === note.pitchMidi &&
      Math.abs(existing.startTimeSeconds - note.startTimeSeconds) < 0.018 &&
      Math.abs(existing.durationSeconds - note.durationSeconds) < 0.04,
    );
    if (!duplicate) kept.push(note);
    else if ((note.confidence ?? note.amplitude) > (duplicate.confidence ?? duplicate.amplitude)) Object.assign(duplicate, note);
  }
  return kept;
}

function contextCorrect(notes, chords, keyDetail, mode) {
  const scale = keyDetail.mode === 'minor' ? MINOR_SCALE : MAJOR_SCALE;
  const result = [];
  for (const note of notes) {
    let score = note.confidence ?? note.amplitude;
    const agreement = note.agreement ?? 1;
    if (agreement >= 2) score += Math.min(0.20, 0.07 * (agreement - 1));
    const keyRelative = (note.pitchMidi % 12 - keyDetail.root + 12) % 12;
    score += scale.has(keyRelative) ? 0.045 : -0.055;

    const midpoint = note.startTimeSeconds + note.durationSeconds * 0.5;
    const chord = chords.find(item => midpoint >= item.start && midpoint < item.end);
    if (chord) score += chord.pitchClasses.includes(note.pitchMidi % 12) ? 0.07 : -0.035;

    if (note.pitchMidi >= 84 && agreement === 1 && note.source !== 'presence') score -= 0.06;
    score = clamp(score, 0, 1);
    if (score >= mode.contextThreshold || agreement >= 2) {
      result.push({ ...note, confidence: score, amplitude: clamp(note.amplitude * (0.88 + score * 0.22), 0.04, 1) });
    }
  }
  return result;
}

export function detectDrumEvents(percussive, sampleRate) {
  if (!percussive?.length) return [];
  const low = onePoleLowPass(percussive, sampleRate, 180);
  const high = onePoleHighPass(percussive, sampleRate, 2600);
  const frameSize = 1024;
  const hop = 256;
  const frames = [];
  let previousRms = 0;
  for (let start = 0; start + frameSize <= percussive.length; start += hop) {
    let energy = 0;
    let lowEnergy = 0;
    let highEnergy = 0;
    let crossings = 0;
    for (let i = 0; i < frameSize; i += 1) {
      const x = percussive[start + i];
      energy += x * x;
      lowEnergy += low[start + i] * low[start + i];
      highEnergy += high[start + i] * high[start + i];
      if (i > 0 && Math.signbit(x) !== Math.signbit(percussive[start + i - 1])) crossings += 1;
    }
    const rms = Math.sqrt(energy / frameSize);
    const onset = Math.max(0, rms - previousRms * 0.86);
    frames.push({ start, rms, onset, low: Math.sqrt(lowEnergy / frameSize), high: Math.sqrt(highEnergy / frameSize), zcr: crossings / frameSize });
    previousRms = rms;
  }
  if (!frames.length) return [];
  const strengths = frames.map(frame => frame.onset).sort((a, b) => a - b);
  const p75 = strengths[Math.floor(strengths.length * 0.75)] ?? 0;
  const mean = strengths.reduce((sum, x) => sum + x, 0) / strengths.length;
  const threshold = Math.max(0.004, p75 * 1.45, mean * 1.8);
  const maxStrength = Math.max(threshold, ...strengths);
  const drums = [];
  let lastTime = -1;

  for (let i = 1; i < frames.length - 1; i += 1) {
    const frame = frames[i];
    if (frame.onset < threshold || frame.onset < frames[i - 1].onset || frame.onset < frames[i + 1].onset) continue;
    const time = frame.start / sampleRate;
    if (time - lastTime < 0.055) continue;
    const lowRatio = frame.low / Math.max(1e-6, frame.rms);
    const highRatio = frame.high / Math.max(1e-6, frame.rms);
    let midi = 38;
    let name = 'Snare';
    if (lowRatio > 0.62) {
      midi = 36;
      name = 'Kick';
    } else if (highRatio > 0.52 || frame.zcr > 0.24) {
      midi = 42;
      name = 'Hi-hat';
    } else if (highRatio < 0.34 && frame.zcr < 0.075) {
      // Pitched attacks (especially piano/guitar) leak into an HPSS percussive
      // stem but usually lack the broadband/high-ZCR evidence of a snare.
      continue;
    }
    const velocity = clamp(0.3 + 0.7 * (frame.onset / maxStrength), 0.12, 1);
    drums.push({ midi, name, time, duration: 0.055, velocity });
    lastTime = time;
  }
  return drums;
}

function dedupeDrums(drums) {
  const sorted = drums.sort((a, b) => a.time - b.time || a.midi - b.midi);
  const kept = [];
  for (const drum of sorted) {
    const duplicate = kept.find(existing => existing.midi === drum.midi && Math.abs(existing.time - drum.time) < 0.07);
    if (!duplicate) kept.push(drum);
    else if (drum.velocity > duplicate.velocity) Object.assign(duplicate, drum);
  }
  return kept;
}

function onePoleLowPass(input, sampleRate, cutoff) {
  const output = new Float32Array(input.length);
  const dt = 1 / sampleRate;
  const rc = 1 / (2 * Math.PI * cutoff);
  const alpha = dt / (rc + dt);
  let y = 0;
  for (let i = 0; i < input.length; i += 1) { y += alpha * (input[i] - y); output[i] = y; }
  return output;
}

function onePoleHighPass(input, sampleRate, cutoff) {
  const output = new Float32Array(input.length);
  const dt = 1 / sampleRate;
  const rc = 1 / (2 * Math.PI * cutoff);
  const alpha = rc / (rc + dt);
  let y = 0;
  let previous = input[0] ?? 0;
  for (let i = 0; i < input.length; i += 1) {
    const x = input[i];
    y = alpha * (y + x - previous);
    previous = x;
    output[i] = y;
  }
  return output;
}

export function exportMidi(result, filename = 'transcription.mid') {
  const midi = new Midi();
  const bpm = Number.isFinite(result.tempo) ? result.tempo : 120;
  midi.header.setTempo(bpm);
  const harmonyTrack = midi.addTrack();
  harmonyTrack.name = 'Wav2mid HQ · Harmony';
  const bassTrack = midi.addTrack();
  bassTrack.name = 'Wav2mid HQ · Bass';
  const drumTrack = midi.addTrack();
  drumTrack.name = 'Wav2mid HQ · Drums';
  drumTrack.channel = 9;

  for (const note of result.notes) {
    const track = note.instrument === 'bass' ? bassTrack : harmonyTrack;
    track.addNote({ midi: note.pitchMidi, time: note.startTimeSeconds, duration: note.durationSeconds, velocity: note.amplitude });
    if (note.pitchBends?.length) {
      const maxBends = 48;
      const stride = Math.max(1, Math.ceil(note.pitchBends.length / maxBends));
      for (let i = 0; i < note.pitchBends.length; i += stride) {
        track.addPitchBend({
          time: note.startTimeSeconds + note.durationSeconds * (i / note.pitchBends.length),
          value: clamp(note.pitchBends[i], -1, 1),
        });
      }
    }
  }
  for (const drum of result.drums ?? []) {
    drumTrack.addNote({ midi: drum.midi, time: drum.time, duration: drum.duration, velocity: drum.velocity });
  }
  downloadBlob(new Blob([midi.toArray()], { type: 'audio/midi' }), filename);
}

export function exportJson(result, filename = 'transcription.json') {
  const payload = JSON.stringify({
    format: 'wav2mid-hq/v2',
    tempo: result.tempo,
    key: result.key,
    chords: result.chords,
    stats: result.stats,
    pipeline: result.pipeline,
    notes: result.notes,
    drums: result.drums,
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
  const onsets = [...new Set(notes.map(n => Math.round(n.startTimeSeconds * 50) / 50))].slice(0, 900);
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

function estimateKeyDetail(notes) {
  if (!notes.length) return { root: 0, mode: 'major', label: '—', score: 0 };
  const majorProfile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
  const minorProfile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];
  const histogram = Array(12).fill(0);
  for (const note of notes) histogram[note.pitchMidi % 12] += note.durationSeconds * note.amplitude * (0.7 + 0.3 * (note.confidence ?? 1));
  let best = { root: 0, mode: 'major', label: '—', score: -Infinity };
  for (let root = 0; root < 12; root += 1) {
    for (const [profile, mode] of [[majorProfile, 'major'], [minorProfile, 'minor']]) {
      let score = 0;
      for (let pc = 0; pc < 12; pc += 1) score += histogram[pc] * profile[(pc - root + 12) % 12];
      if (score > best.score) best = { root, mode, score, label: `${NOTE_NAMES[root]} ${mode}` };
    }
  }
  return best;
}

const CHORD_TEMPLATES = [
  ['maj', [0, 4, 7]], ['min', [0, 3, 7]], ['dim', [0, 3, 6]], ['aug', [0, 4, 8]],
  ['sus2', [0, 2, 7]], ['sus4', [0, 5, 7]], ['7', [0, 4, 7, 10]], ['maj7', [0, 4, 7, 11]],
  ['m7', [0, 3, 7, 10]], ['mMaj7', [0, 3, 7, 11]], ['6', [0, 4, 7, 9]], ['m6', [0, 3, 7, 9]],
  ['add9', [0, 2, 4, 7]], ['madd9', [0, 2, 3, 7]], ['9', [0, 2, 4, 7, 10]], ['m9', [0, 2, 3, 7, 10]],
  ['maj9', [0, 2, 4, 7, 11]], ['m11', [0, 2, 3, 5, 7, 10]],
];

function estimateChords(notes, duration) {
  const windowSize = 0.5;
  const segments = [];
  let previous = null;
  let cursor = 0;
  const ordered = [...notes].sort((a, b) => a.startTimeSeconds - b.startTimeSeconds);
  for (let t = 0; t < duration; t += windowSize) {
    while (cursor < ordered.length && ordered[cursor].startTimeSeconds + ordered[cursor].durationSeconds < t - windowSize) cursor += 1;
    const active = [];
    for (let i = cursor; i < ordered.length; i += 1) {
      const n = ordered[i];
      if (n.startTimeSeconds >= t + windowSize) break;
      if (n.startTimeSeconds + n.durationSeconds > t) active.push(n);
    }
    if (active.length < 2) continue;
    const weights = Array(12).fill(0);
    for (const n of active) weights[n.pitchMidi % 12] += n.amplitude * Math.min(n.durationSeconds, windowSize) * (0.75 + 0.25 * (n.confidence ?? 1));
    const chord = bestChord(weights);
    if (!chord || chord.score < 0.56) continue;
    if (previous?.name === chord.name && t - previous.end <= windowSize * 1.5) {
      previous.end = t + windowSize;
      previous.confidence = Math.max(previous.confidence, chord.score);
    } else {
      previous = { name: chord.name, root: chord.root, pitchClasses: chord.pitchClasses, start: t, end: t + windowSize, confidence: chord.score };
      segments.push(previous);
    }
  }
  return segments.slice(0, 600);
}

function bestChord(weights) {
  const total = weights.reduce((a, b) => a + b, 0);
  if (!total) return null;
  let best = null;
  for (let root = 0; root < 12; root += 1) {
    for (const [suffix, intervals] of CHORD_TEMPLATES) {
      const pitchClasses = intervals.map(i => (root + i) % 12);
      const pcs = new Set(pitchClasses);
      const inWeight = weights.reduce((sum, w, pc) => sum + (pcs.has(pc) ? w : 0), 0);
      const coverage = intervals.filter(i => weights[(root + i) % 12] > total * 0.022).length / intervals.length;
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

function buildStats(notes, drums, duration, candidateCount, chunkCount, passCount) {
  let maxPolyphony = 0;
  const events = [];
  for (const n of notes) {
    events.push([n.startTimeSeconds, 1]);
    events.push([n.startTimeSeconds + n.durationSeconds, -1]);
  }
  events.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  let poly = 0;
  for (const [, delta] of events) { poly += delta; maxPolyphony = Math.max(maxPolyphony, poly); }
  const ensembleBacked = notes.filter(note => (note.agreement ?? 1) >= 2).length;
  return {
    duration,
    noteCount: notes.length,
    drumCount: drums.length,
    candidateCount,
    maxPolyphony,
    lowestNote: notes.length ? Math.min(...notes.map(n => n.pitchMidi)) : null,
    highestNote: notes.length ? Math.max(...notes.map(n => n.pitchMidi)) : null,
    ensembleBacked,
    ensembleRatio: notes.length ? ensembleBacked / notes.length : 0,
    chunks: chunkCount,
    passCount,
  };
}

export function midiName(midi) {
  if (midi == null) return '—';
  return `${NOTE_NAMES[midi % 12]}${Math.floor(midi / 12) - 1}`;
}

function clamp(value, min, max) { return Math.min(max, Math.max(min, value)); }
