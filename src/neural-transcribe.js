const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
let corePromise = null;

export async function transcribeNeural(audioBuffer, options = {}, onProgress = () => {}) {
  const profile = chooseMemoryProfile();
  installThreadCap(profile.wasmThreads);
  const core = await getCore();

  const duration = Number(audioBuffer?.duration ?? 0);
  if (!Number.isFinite(duration) || duration <= 0) throw new Error('Decoded audio is empty.');

  if (duration <= profile.chunkSeconds + 1) {
    const result = await core.transcribeNeural(audioBuffer, options, onProgress);
    result.pipeline = {
      ...(result.pipeline ?? {}),
      lowMemoryStreaming: false,
      memoryProfile: profile.name,
      wasmThreadCap: profile.wasmThreads,
      chunkSeconds: profile.chunkSeconds,
    };
    return result;
  }

  const overlapSeconds = Math.min(profile.overlapSeconds, profile.chunkSeconds * 0.2);
  const stepSeconds = profile.chunkSeconds - overlapSeconds;
  const totalChunks = Math.max(1, Math.ceil((duration - overlapSeconds) / stepSeconds));
  const notes = [];
  const drums = [];
  const tempos = [];
  const keys = new Map();
  const chunkMeta = [];
  let backend = 'unknown';
  let separatorBackend = 'unknown';

  for (let index = 0; index < totalChunks; index += 1) {
    const nominalStart = index * stepSeconds;
    const start = Math.max(0, nominalStart);
    const end = Math.min(duration, start + profile.chunkSeconds);
    const chunkDuration = Math.max(0.01, end - start);
    const chunk = sliceAudioBuffer(audioBuffer, start, end);

    onProgress({
      stage: 'neural-separate',
      value: index / totalChunks,
      detail: `low-memory chunk ${index + 1}/${totalChunks} · ${profile.name}`,
    });

    const result = await core.transcribeNeural(chunk, options, progress => {
      const local = clamp(Number(progress?.value ?? 0), 0, 1);
      onProgress({
        ...progress,
        value: (index + local) / totalChunks,
        detail: `chunk ${index + 1}/${totalChunks}${progress?.detail ? ` · ${progress.detail}` : ''}`,
      });
    });

    backend = result.pipeline?.backend ?? backend;
    separatorBackend = result.pipeline?.separatorBackend ?? separatorBackend;
    if (Number.isFinite(result.tempo)) tempos.push(Number(result.tempo));
    if (result.key) keys.set(result.key, (keys.get(result.key) ?? 0) + chunkDuration);

    const leftKeep = index === 0 ? 0 : overlapSeconds * 0.5;
    const rightKeep = index === totalChunks - 1 ? chunkDuration : chunkDuration - overlapSeconds * 0.5;

    for (const note of result.notes ?? []) {
      const midpoint = Number(note.startTimeSeconds ?? 0) + Number(note.durationSeconds ?? 0) * 0.5;
      if (midpoint < leftKeep || midpoint >= rightKeep) continue;
      notes.push({ ...note, startTimeSeconds: Number(note.startTimeSeconds ?? 0) + start });
    }
    for (const drum of result.drums ?? []) {
      const localTime = Number(drum.time ?? 0);
      if (localTime < leftKeep || localTime >= rightKeep) continue;
      drums.push({ ...drum, time: localTime + start });
    }

    chunkMeta.push({ index, start, end, noteCount: result.notes?.length ?? 0, drumCount: result.drums?.length ?? 0 });

    // Release all stem/tensor references from this chunk before the next one.
    // Yielding gives the browser a chance to reclaim detached ArrayBuffers and GPU/WASM temporaries.
    await new Promise(resolve => setTimeout(resolve, profile.yieldMs));
  }

  const mergedNotes = mergeBoundaryNotes(notes, 0.10);
  const mergedDrums = dedupeDrums(drums, 0.05);
  const tempo = robustMedian(tempos) ?? 120;
  const key = [...keys.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] ?? '—';
  const stats = buildStats(mergedNotes, mergedDrums, duration, totalChunks);

  onProgress({ stage: 'done', value: 1, detail: `${mergedNotes.length} notes · ${mergedDrums.length} drums · ${totalChunks} chunks` });
  return {
    notes: mergedNotes,
    rawNotes: [],
    drums: mergedDrums,
    tempo,
    key,
    keyDetail: null,
    chords: [],
    stats,
    pipeline: {
      mode: 'NEURAL HQ',
      backend,
      separatorBackend,
      sourceSeparation: 'HTDemucs 4-stem neural · low-memory streaming',
      stemPasses: ['mix', 'other', 'bass', 'vocals'],
      ensemble: true,
      contextDecoder: true,
      drumTranscription: true,
      chunks: totalChunks,
      neural: true,
      model: 'htdemucs_embedded.onnx',
      lowMemoryStreaming: true,
      rawCandidatesDropped: true,
      memoryProfile: profile.name,
      chunkSeconds: profile.chunkSeconds,
      overlapSeconds,
      wasmThreadCap: profile.wasmThreads,
      deviceMemoryGiB: Number(navigator?.deviceMemory ?? 0) || null,
      hardwareConcurrency: Number(navigator?.hardwareConcurrency ?? 0) || null,
      chunkMeta,
    },
  };
}

export function neuralModelDescription() {
  const profile = chooseMemoryProfile();
  return {
    name: 'HTDemucs 4-stem · low-memory streaming',
    sampleRate: 44100,
    tracks: ['drums', 'bass', 'other', 'vocals'],
    sizeHint: `~172 MB first download · ${profile.chunkSeconds}s chunks`,
  };
}

export function midiPitchName(midi) {
  return midi == null ? '—' : `${NOTE_NAMES[midi % 12]}${Math.floor(midi / 12) - 1}`;
}

async function getCore() {
  if (!corePromise) corePromise = import('./neural-transcribe-core.js');
  return corePromise;
}

function chooseMemoryProfile() {
  const deviceMemory = Number(globalThis.navigator?.deviceMemory ?? 0);
  const cores = Number(globalThis.navigator?.hardwareConcurrency ?? 4);
  const mobile = /Android|iPhone|iPad|Mobile/i.test(globalThis.navigator?.userAgent ?? '');

  if (mobile || (deviceMemory > 0 && deviceMemory <= 4)) {
    return { name: 'MOBILE_SAFE', chunkSeconds: 12, overlapSeconds: 2, wasmThreads: 2, yieldMs: 32 };
  }
  if ((deviceMemory > 0 && deviceMemory <= 8) || cores <= 6) {
    return { name: 'BALANCED', chunkSeconds: 18, overlapSeconds: 2, wasmThreads: Math.min(3, Math.max(2, Math.floor(cores / 2))), yieldMs: 20 };
  }
  return { name: 'DESKTOP', chunkSeconds: 24, overlapSeconds: 3, wasmThreads: Math.min(4, Math.max(2, Math.floor(cores / 2))), yieldMs: 12 };
}

function installThreadCap(cap) {
  const nav = globalThis.navigator;
  if (!nav || nav.__wav2midThreadCapInstalled) return;
  const reported = Number(nav.hardwareConcurrency ?? 4);
  try {
    Object.defineProperty(nav, 'hardwareConcurrency', {
      configurable: true,
      get: () => Math.max(1, Math.min(reported, cap)),
    });
    Object.defineProperty(nav, '__wav2midThreadCapInstalled', { configurable: true, value: true });
  } catch {
    // Some browsers expose a non-extensible Navigator. Streaming still lowers peak RAM even without this cap.
  }
}

function sliceAudioBuffer(buffer, startSeconds, endSeconds) {
  const sampleRate = Number(buffer.sampleRate);
  const start = Math.max(0, Math.floor(startSeconds * sampleRate));
  const end = Math.min(Number(buffer.length), Math.ceil(endSeconds * sampleRate));
  const length = Math.max(1, end - start);
  const channels = Math.max(1, Number(buffer.numberOfChannels ?? 1));
  const data = [];
  for (let channel = 0; channel < channels; channel += 1) {
    const source = buffer.getChannelData(channel);
    const copy = new Float32Array(length);
    copy.set(source.subarray(start, end));
    data.push(copy);
  }
  return {
    sampleRate,
    numberOfChannels: channels,
    length,
    duration: length / sampleRate,
    getChannelData(channel) { return data[Math.min(channel, data.length - 1)]; },
  };
}

function mergeBoundaryNotes(notes, maxGap) {
  const byPitch = new Map();
  for (const note of notes) {
    const pitch = Number(note.pitchMidi);
    const list = byPitch.get(pitch) ?? [];
    list.push({ ...note });
    byPitch.set(pitch, list);
  }
  const output = [];
  for (const list of byPitch.values()) {
    list.sort((a, b) => a.startTimeSeconds - b.startTimeSeconds);
    let current = list[0];
    for (let i = 1; i < list.length; i += 1) {
      const next = list[i];
      const currentEnd = current.startTimeSeconds + current.durationSeconds;
      const gap = next.startTimeSeconds - currentEnd;
      if (gap >= -0.06 && gap <= maxGap) {
        const nextEnd = next.startTimeSeconds + next.durationSeconds;
        current.durationSeconds = Math.max(currentEnd, nextEnd) - current.startTimeSeconds;
        current.confidence = Math.max(current.confidence ?? 0, next.confidence ?? 0);
        current.amplitude = Math.max(current.amplitude ?? 0, next.amplitude ?? 0);
        current.sources = [...new Set([...(current.sources ?? []), ...(next.sources ?? [])])];
        current.agreement = Math.max(current.agreement ?? 1, next.agreement ?? 1, current.sources.length);
      } else {
        output.push(current);
        current = next;
      }
    }
    if (current) output.push(current);
  }
  return output.sort((a, b) => a.startTimeSeconds - b.startTimeSeconds || a.pitchMidi - b.pitchMidi);
}

function dedupeDrums(drums, tolerance) {
  const sorted = [...drums].sort((a, b) => a.time - b.time || a.midi - b.midi);
  const output = [];
  for (const event of sorted) {
    const previous = output[output.length - 1];
    if (previous && previous.midi === event.midi && Math.abs(previous.time - event.time) <= tolerance) {
      if ((event.velocity ?? 0) > (previous.velocity ?? 0)) output[output.length - 1] = event;
    } else output.push(event);
  }
  return output;
}

function robustMedian(values) {
  if (!values.length) return null;
  const sorted = [...values].filter(Number.isFinite).sort((a, b) => a - b);
  if (!sorted.length) return null;
  const mid = Math.floor(sorted.length / 2);
  return Math.round(sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2);
}

function buildStats(notes, drums, duration, chunks) {
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
    duration,
    noteCount: notes.length,
    drumCount: drums.length,
    candidateCount: notes.length,
    maxPolyphony,
    lowestNote: notes.length ? Math.min(...notes.map(note => note.pitchMidi)) : null,
    highestNote: notes.length ? Math.max(...notes.map(note => note.pitchMidi)) : null,
    ensembleBacked,
    ensembleRatio: notes.length ? ensembleBacked / notes.length : 0,
    chunks,
    passCount: 4,
  };
}

function clamp(value, min, max) { return Math.min(max, Math.max(min, value)); }
