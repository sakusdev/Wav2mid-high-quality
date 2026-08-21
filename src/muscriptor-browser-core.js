import FFT from 'fft.js';
import * as ort from 'onnxruntime-web/webgpu';

export const MUSCRIPTOR_SAMPLE_RATE = 16000;
export const MUSCRIPTOR_SEGMENT_SECONDS = 5;
export const MUSCRIPTOR_CARD = 1393;
export const MUSCRIPTOR_INITIAL_TOKEN = 1393;
export const MUSCRIPTOR_EOS = 1;
export const MUSCRIPTOR_MAX_GENERATION = 2000;

const FFT_SIZE = 2048;
const HOP = 160;
const MEL_BINS = 512;
const FREQ_BINS = FFT_SIZE / 2 + 1;
const EPS = 1e-6;
const MODEL_MANIFEST_URL = '/models/muscriptor-small/manifest.json';

const PERIODIC_HANN = new Float32Array(FFT_SIZE);
for (let i = 0; i < FFT_SIZE; i += 1) {
  PERIODIC_HANN[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / FFT_SIZE);
}
const MEL_FILTERS = buildHtkMelFilters();

const PROGRAM_NAMES = new Map([
  [0, 'acoustic_piano'], [2, 'electric_piano'], [8, 'chromatic_percussion'],
  [16, 'organ'], [24, 'acoustic_guitar'], [26, 'clean_electric_guitar'],
  [29, 'distorted_electric_guitar'], [32, 'acoustic_bass'], [33, 'electric_bass'],
  [40, 'violin'], [41, 'viola'], [42, 'cello'], [43, 'contrabass'],
  [46, 'orchestral_harp'], [47, 'timpani'], [48, 'string_ensemble'],
  [50, 'synth_strings'], [52, 'voice'], [55, 'orchestra_hit'], [56, 'trumpet'],
  [57, 'trombone'], [58, 'tuba'], [60, 'french_horn'], [61, 'brass_section'],
  [64, 'soprano_and_alto_sax'], [66, 'tenor_sax'], [67, 'baritone_sax'],
  [68, 'oboe'], [69, 'english_horn'], [70, 'bassoon'], [71, 'clarinet'],
  [72, 'flutes'], [80, 'synth_lead'], [88, 'synth_pad'],
]);

const INSTRUMENT_GROUPS = Object.freeze({
  acoustic_piano: 0, electric_piano: 1, chromatic_percussion: 2, organ: 3,
  acoustic_guitar: 4, clean_electric_guitar: 5, distorted_electric_guitar: 6,
  acoustic_bass: 7, electric_bass: 8, violin: 9, viola: 10, cello: 11,
  contrabass: 12, orchestral_harp: 13, timpani: 14, string_ensemble: 15,
  synth_strings: 16, voice: 17, orchestra_hit: 18, trumpet: 19, trombone: 20,
  tuba: 21, french_horn: 22, brass_section: 23, soprano_and_alto_sax: 24,
  tenor_sax: 25, baritone_sax: 26, oboe: 27, english_horn: 28, bassoon: 29,
  clarinet: 30, flutes: 31, synth_lead: 32, synth_pad: 33, drums: 36,
});

const GROUP_PROGRAMS = Object.freeze({
  0: [0, 1, 3, 6, 7], 1: [2, 4, 5], 2: range(8, 15), 3: range(16, 23),
  4: [24, 25], 5: [26, 27, 28], 6: [29, 30, 31], 7: [32, 35],
  8: [33, 34, 36, 37, 38, 39], 9: [40], 10: [41], 11: [42], 12: [43],
  13: [46], 14: [47], 15: [48, 49, 44, 45], 16: [50, 51],
  17: [52, 53, 54], 18: [55], 19: [56, 59], 20: [57], 21: [58],
  22: [60], 23: [61, 62, 63], 24: [64, 65], 25: [66], 26: [67],
  27: [68], 28: [69], 29: [70], 30: [71], 31: range(72, 79),
  32: range(80, 87), 33: range(88, 95), 34: [100], 35: [101],
});

let modelPromise = null;

export function decodeMuScriptorToken(token) {
  const t = Number(token);
  if (t === 0) return { type: 'PAD', value: 0 };
  if (t === 1) return { type: 'EOS', value: 0 };
  if (t === 2) return { type: 'UNK', value: 0 };
  if (t >= 3 && t <= 1003) return { type: 'shift', value: t - 3 };
  if (t >= 1004 && t <= 1131) return { type: 'pitch', value: t - 1004 };
  if (t >= 1132 && t <= 1133) return { type: 'velocity', value: t - 1132 };
  if (t === 1134) return { type: 'tie', value: 0 };
  if (t >= 1135 && t <= 1264) return { type: 'program', value: t - 1135 };
  if (t >= 1265 && t <= 1392) return { type: 'drum', value: t - 1265 };
  return { type: 'unknown', value: t };
}

export function encodeTieSection(openKeys) {
  const tokens = [];
  let programState = null;
  const sorted = [...openKeys].sort((a, b) => (a[0] - b[0]) || (a[1] - b[1]));
  for (const [program, pitch] of sorted) {
    if (program !== programState) {
      tokens.push(1135 + program);
      programState = program;
    }
    tokens.push(1004 + pitch);
  }
  tokens.push(1134);
  return tokens;
}

export class MuScriptorNoteTracker {
  constructor(frameRate = 100) {
    this.frameRate = frameRate;
    this.open = new Map();
    this.seekTime = 0;
    this.nextSeekTime = null;
    this.startTick = 0;
    this.tickState = 0;
    this.program = null;
    this.velocity = null;
    this.inPrologue = true;
    this.skipRest = false;
    this.tieSet = new Set();
    this.chunkStarted = false;
  }

  boundary(seekTime, nextSeekTime = null) {
    let actions = [];
    if (this.chunkStarted && this.inPrologue) actions = this.endAll(this.seekTime);
    this.seekTime = Number(seekTime);
    this.nextSeekTime = nextSeekTime == null ? null : Number(nextSeekTime);
    this.startTick = Math.round(this.seekTime * this.frameRate);
    this.tickState = this.startTick;
    this.program = null;
    this.velocity = null;
    this.inPrologue = true;
    this.skipRest = false;
    this.tieSet = new Set();
    this.chunkStarted = true;
    return actions;
  }

  feed(token) {
    const event = decodeMuScriptorToken(token);
    if (this.inPrologue) {
      if (event.type === 'tie') {
        this.inPrologue = false;
        this.velocity = null;
        const ended = [];
        for (const key of [...this.open.keys()]) {
          if (!this.tieSet.has(key)) {
            const [program, pitch] = key.split(':').map(Number);
            this.open.delete(key);
            ended.push({ type: 'end', program, pitch, time: this.seekTime });
          }
        }
        return ended;
      }
      if (event.type === 'shift') {
        this.inPrologue = false;
        this.skipRest = true;
        return this.endAll(this.seekTime);
      }
      if (event.type === 'program') this.program = event.value;
      else if (event.type === 'pitch' && this.program != null) {
        this.tieSet.add(keyFor(this.program, event.value));
      }
      return [];
    }

    if (this.skipRest) return [];
    if (event.type === 'shift') {
      if (event.value > 0) this.tickState = this.startTick + event.value;
      return [];
    }
    if (event.type === 'program') { this.program = event.value; return []; }
    if (event.type === 'velocity') { this.velocity = event.value; return []; }
    if (event.type === 'drum') {
      const time = this.tickState / this.frameRate;
      if (this.nextSeekTime == null || time < this.nextSeekTime) {
        return [{ type: 'drum', pitch: event.value, time }];
      }
      return [];
    }
    if (event.type !== 'pitch' || this.program == null || this.velocity == null) return [];

    const time = this.tickState / this.frameRate;
    if (this.nextSeekTime != null && time >= this.nextSeekTime) return [];
    const key = keyFor(this.program, event.value);
    const actions = [];
    if (this.open.has(key)) {
      this.open.delete(key);
      actions.push({ type: 'end', program: this.program, pitch: event.value, time });
    }
    if (this.velocity > 0) {
      this.open.set(key, time);
      actions.push({ type: 'start', program: this.program, pitch: event.value, time });
    }
    return actions;
  }

  finish() {
    if (this.chunkStarted && this.inPrologue) return this.endAll(this.seekTime);
    const actions = [];
    for (const [key, onset] of this.open) {
      const [program, pitch] = key.split(':').map(Number);
      actions.push({ type: 'end', program, pitch, time: onset + 0.01 });
    }
    this.open.clear();
    return actions;
  }

  openKeys() {
    return [...this.open.keys()]
      .map(key => key.split(':').map(Number))
      .sort((a, b) => (a[0] - b[0]) || (a[1] - b[1]));
  }

  endAll(time) {
    const actions = [];
    for (const key of this.open.keys()) {
      const [program, pitch] = key.split(':').map(Number);
      actions.push({ type: 'end', program, pitch, time });
    }
    this.open.clear();
    return actions;
  }
}

export async function audioBufferTo16kMono(audioBuffer) {
  if (!audioBuffer) throw new Error('MuScriptor received no AudioBuffer.');
  const targetLength = Math.max(1, Math.round(audioBuffer.duration * MUSCRIPTOR_SAMPLE_RATE));
  if (typeof OfflineAudioContext === 'function') {
    try {
      const ctx = new OfflineAudioContext(1, targetLength, MUSCRIPTOR_SAMPLE_RATE);
      const source = ctx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(ctx.destination);
      source.start(0);
      const rendered = await ctx.startRendering();
      return new Float32Array(rendered.getChannelData(0));
    } catch (error) {
      console.warn('OfflineAudioContext resample failed; using linear resample.', error);
    }
  }
  const mono = new Float32Array(audioBuffer.length);
  for (let ch = 0; ch < audioBuffer.numberOfChannels; ch += 1) {
    const data = audioBuffer.getChannelData(ch);
    for (let i = 0; i < mono.length; i += 1) mono[i] += data[i] / audioBuffer.numberOfChannels;
  }
  return linearResample(mono, audioBuffer.sampleRate, MUSCRIPTOR_SAMPLE_RATE);
}

export function logMelForFiveSecondChunk(input) {
  if (!(input instanceof Float32Array) || input.length !== MUSCRIPTOR_SAMPLE_RATE * MUSCRIPTOR_SEGMENT_SECONDS) {
    throw new Error('MuScriptor mel frontend expects exactly 80,000 samples.');
  }
  const frames = Math.floor(input.length / HOP) + 1; // 501 with centered STFT.
  const out = new Float32Array(frames * MEL_BINS);
  const fft = new FFT(FFT_SIZE);
  const frame = new Array(FFT_SIZE).fill(0);
  const spectrum = fft.createComplexArray();
  const mel = new Float64Array(MEL_BINS);

  for (let t = 0; t < frames; t += 1) {
    const start = t * HOP - FFT_SIZE / 2;
    for (let n = 0; n < FFT_SIZE; n += 1) {
      frame[n] = input[reflectIndex(start + n, input.length)] * PERIODIC_HANN[n];
    }
    fft.realTransform(spectrum, frame);
    mel.fill(0);
    for (let k = 0; k < FREQ_BINS; k += 1) {
      const re = spectrum[k * 2];
      const im = spectrum[k * 2 + 1];
      const magnitude = Math.hypot(re, im);
      const filters = MEL_FILTERS[k];
      for (let j = 0; j < filters.length; j += 2) {
        mel[filters[j]] += magnitude * filters[j + 1];
      }
    }
    const base = t * MEL_BINS;
    for (let m = 0; m < MEL_BINS; m += 1) out[base + m] = Math.log(mel[m] + EPS);
  }
  return { data: out, dims: [1, frames, MEL_BINS] };
}

export async function loadMuScriptorBrowserModel(manifestUrl = MODEL_MANIFEST_URL) {
  if (modelPromise) return modelPromise;
  modelPromise = createBrowserModel(manifestUrl).catch(error => {
    modelPromise = null;
    throw error;
  });
  return modelPromise;
}

async function createBrowserModel(manifestUrl) {
  if (!navigator.gpu) throw new Error('MuScriptor ULTRA browser mode requires WebGPU (Chromium recommended).');
  const manifestResponse = await fetch(manifestUrl, { cache: 'force-cache' });
  if (!manifestResponse.ok) throw new Error(`MuScriptor model manifest HTTP ${manifestResponse.status}`);
  const manifest = await manifestResponse.json();
  if (manifest.format !== 'wav2mid-muscriptor-browser/v1') throw new Error('Unsupported MuScriptor browser model manifest.');

  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No WebGPU adapter is available.');
  const device = await adapter.requestDevice();
  device.lost.then(info => console.warn('MuScriptor WebGPU device lost:', info));

  const ep = { name: 'webgpu', device, storageBufferCacheMode: 'simple' };
  const conditioner = await ort.InferenceSession.create(manifest.files.conditioner.url, {
    executionProviders: [ep],
    graphOptimizationLevel: 'all',
    preferredOutputLocation: { prefix_embeddings: 'gpu-buffer' },
  });
  const preferred = { logits: 'cpu' };
  for (let i = 0; i < manifest.architecture.layers; i += 1) {
    preferred[`new_k_${i}`] = 'gpu-buffer';
    preferred[`new_v_${i}`] = 'gpu-buffer';
  }
  const decoder = await ort.InferenceSession.create(manifest.files.decoder.url, {
    executionProviders: [ep],
    graphOptimizationLevel: 'all',
    preferredOutputLocation: preferred,
  });
  return new BrowserMuScriptor({ manifest, conditioner, decoder, device });
}

class BrowserMuScriptor {
  constructor({ manifest, conditioner, decoder, device }) {
    this.manifest = manifest;
    this.conditioner = conditioner;
    this.decoder = decoder;
    this.device = device;
    this.layers = manifest.architecture.layers;
    this.heads = manifest.architecture.heads;
    this.headDim = manifest.architecture.dim / this.heads;
    this.maxCache = manifest.architecture.maxCache;
    this.decoderActivationType = manifest.architecture.decoderActivationType
      || manifest.architecture.activationType
      || 'float16';
    if (!['float16', 'float32'].includes(this.decoderActivationType)) {
      throw new Error(`Unsupported MuScriptor decoder activation type: ${this.decoderActivationType}`);
    }
    const elementBytes = this.decoderActivationType === 'float32' ? 4 : 2;
    this.cacheRowBytes = this.heads * this.headDim * elementBytes;
  }

  async transcribe(audioBuffer, options = {}, onProgress) {
    const waveform = await audioBufferTo16kMono(audioBuffer);
    const chunkSamples = MUSCRIPTOR_SAMPLE_RATE * MUSCRIPTOR_SEGMENT_SECONDS;
    const chunks = Math.max(1, Math.ceil(waveform.length / chunkSamples));
    const tracker = new MuScriptorNoteTracker(100);
    const openEvents = new Map();
    const notes = [];
    const drums = [];
    const selectedInstruments = normalizeInstrumentNames(options.instruments || []);
    const instrumentIds = selectedInstruments.length
      ? selectedInstruments.map(name => INSTRUMENT_GROUPS[name] + 2)
      : [1]; // upstream null ClassConditioner maps to table index 1.
    const forbidden = selectedInstruments.length ? forbiddenTokens(selectedInstruments) : null;

    onProgress?.({ stage: 'muscriptor-load', value: 0.02, detail: 'model ready' });
    for (let chunkIndex = 0; chunkIndex < chunks; chunkIndex += 1) {
      const seek = chunkIndex * MUSCRIPTOR_SEGMENT_SECONDS;
      const nextSeek = chunkIndex + 1 < chunks ? (chunkIndex + 1) * MUSCRIPTOR_SEGMENT_SECONDS : null;
      consumeActions(tracker.boundary(seek, nextSeek), openEvents, notes, drums);
      const prompt = chunkIndex > 0 ? encodeTieSection(tracker.openKeys()) : [];
      for (const tok of prompt) consumeActions(tracker.feed(tok), openEvents, notes, drums);

      const chunk = new Float32Array(chunkSamples);
      chunk.set(waveform.subarray(chunkIndex * chunkSamples, Math.min(waveform.length, (chunkIndex + 1) * chunkSamples)));
      const mel = logMelForFiveSecondChunk(chunk);
      const prefix = await this.condition(mel, instrumentIds);
      await this.generateChunk(prefix, prompt, forbidden, tracker, openEvents, notes, drums);

      onProgress?.({
        stage: 'muscriptor-decode',
        value: 0.04 + 0.94 * ((chunkIndex + 1) / chunks),
        detail: `${chunkIndex + 1}/${chunks} chunks`,
      });
      await yieldToUi();
    }
    consumeActions(tracker.finish(), openEvents, notes, drums);
    notes.sort((a, b) => a.startTimeSeconds - b.startTimeSeconds || a.pitchMidi - b.pitchMidi);
    drums.sort((a, b) => a.time - b.time || a.midi - b.midi);
    return {
      notes,
      drums,
      rawNotes: [...notes, ...drums.map(d => ({
        pitchMidi: d.midi, startTimeSeconds: d.time, durationSeconds: 0.01,
        instrument: 'drums', program: 128,
      }))],
      chunks,
      model: this.manifest,
    };
  }

  async condition(mel, instrumentIds) {
    const logMel = new ort.Tensor('float32', mel.data, mel.dims);
    const ids = new ort.Tensor('int64', BigInt64Array.from(instrumentIds, BigInt), [1, instrumentIds.length]);
    try {
      const outputs = await this.conditioner.run({ log_mel: logMel, instrument_embed_ids: ids });
      return outputs.prefix_embeddings;
    } finally {
      logMel.dispose?.();
      ids.dispose?.();
    }
  }

  async generateChunk(prefix, prompt, forbidden, tracker, openEvents, notes, drums) {
    const caches = this.createCaches();
    let step = null;
    try {
      const firstTokens = BigInt64Array.from([MUSCRIPTOR_INITIAL_TOKEN, ...prompt], BigInt);
      const prefixLength = prefix.dims[1];
      let pastLen = 0;
      try {
        step = await this.runDecoder(prefix, firstTokens, pastLen, caches);
      } finally {
        prefix.dispose?.();
      }
      pastLen += prefixLength + firstTokens.length;

      for (let generated = 0; generated < MUSCRIPTOR_MAX_GENERATION; generated += 1) {
        let token;
        try {
          token = argmax(step.logits.data, forbidden);
          await this.appendNewKv(step, caches, pastLen - step.queryLength);
        } finally {
          step.dispose();
          step = null;
        }
        if (token === MUSCRIPTOR_EOS) return;
        consumeActions(tracker.feed(token), openEvents, notes, drums);
        const emptyData = this.decoderActivationType === 'float32'
          ? new Float32Array(0)
          : new Uint16Array(0);
        const emptyPrefix = new ort.Tensor(
          this.decoderActivationType,
          emptyData,
          [1, 0, this.manifest.architecture.dim],
        );
        try {
          step = await this.runDecoder(emptyPrefix, BigInt64Array.of(BigInt(token)), pastLen, caches);
        } finally {
          emptyPrefix.dispose?.();
        }
        pastLen += 1;
        if (pastLen >= this.maxCache - 1) throw new Error('MuScriptor KV cache limit reached before EOS.');
      }
      throw new Error(`MuScriptor chunk did not emit EOS within ${MUSCRIPTOR_MAX_GENERATION} tokens.`);
    } finally {
      step?.dispose();
      for (const cache of caches) {
        cache.tensor.dispose?.();
        cache.buffer.destroy();
      }
    }
  }

  createCaches() {
    const caches = [];
    const bytes = this.maxCache * this.cacheRowBytes;
    for (let i = 0; i < this.layers * 2; i += 1) {
      const buffer = this.device.createBuffer({
        size: bytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      const tensor = ort.Tensor.fromGpuBuffer(buffer, {
        dataType: this.decoderActivationType,
        dims: [1, this.maxCache, this.heads, this.headDim],
      });
      caches.push({ buffer, tensor });
    }
    return caches;
  }

  async runDecoder(prefix, tokenData, pastLen, caches) {
    const tokenIds = tokenData instanceof BigInt64Array ? tokenData : BigInt64Array.from(tokenData, BigInt);
    const tokenIdsTensor = new ort.Tensor('int64', tokenIds, [1, tokenIds.length]);
    const pastLenTensor = new ort.Tensor('int64', BigInt64Array.of(BigInt(pastLen)), []);
    const feeds = {
      prefix_embeddings: prefix,
      token_ids: tokenIdsTensor,
      past_len: pastLenTensor,
    };
    for (let i = 0; i < this.layers; i += 1) {
      feeds[`cache_k_${i}`] = caches[i * 2].tensor;
      feeds[`cache_v_${i}`] = caches[i * 2 + 1].tensor;
    }
    try {
      const outputs = await this.decoder.run(feeds);
      const queryLength = outputs.new_k_0?.dims?.[1] ?? tokenIds.length;
      return new DecoderStep(outputs, this.layers, queryLength);
    } finally {
      tokenIdsTensor.dispose?.();
      pastLenTensor.dispose?.();
    }
  }

  async appendNewKv(step, caches, dstRow) {
    const encoder = this.device.createCommandEncoder();
    const bytes = step.queryLength * this.cacheRowBytes;
    const dstOffset = dstRow * this.cacheRowBytes;
    for (let i = 0; i < this.layers; i += 1) {
      const k = step.outputs[`new_k_${i}`];
      const v = step.outputs[`new_v_${i}`];
      encoder.copyBufferToBuffer(k.gpuBuffer, 0, caches[i * 2].buffer, dstOffset, bytes);
      encoder.copyBufferToBuffer(v.gpuBuffer, 0, caches[i * 2 + 1].buffer, dstOffset, bytes);
    }
    this.device.queue.submit([encoder.finish()]);
  }
}

class DecoderStep {
  constructor(outputs, layers, queryLength) {
    this.outputs = outputs;
    this.layers = layers;
    this.queryLength = queryLength;
    this.logits = outputs.logits;
  }
  dispose() {
    for (let i = 0; i < this.layers; i += 1) {
      this.outputs[`new_k_${i}`]?.dispose?.();
      this.outputs[`new_v_${i}`]?.dispose?.();
    }
    this.logits?.dispose?.();
  }
}

function consumeActions(actions, openEvents, notes, drums) {
  for (const action of actions) {
    if (action.type === 'drum') {
      drums.push({ midi: action.pitch, name: 'drums', time: action.time, duration: 0.01, velocity: 0.8 });
      continue;
    }
    const key = keyFor(action.program, action.pitch);
    if (action.type === 'start') {
      openEvents.set(key, action.time);
      continue;
    }
    const onset = openEvents.get(key);
    if (onset == null) continue;
    openEvents.delete(key);
    notes.push({
      pitchMidi: action.pitch,
      startTimeSeconds: onset,
      durationSeconds: Math.max(0.01, action.time - onset),
      instrument: instrumentForProgram(action.program),
      program: action.program,
      confidence: 1,
      amplitude: 1,
    });
  }
}

function normalizeInstrumentNames(names) {
  const unique = [...new Set(names.map(String).map(x => x.trim()).filter(Boolean))];
  for (const name of unique) {
    if (!(name in INSTRUMENT_GROUPS)) throw new Error(`Unknown MuScriptor instrument: ${name}`);
  }
  return unique;
}

function forbiddenTokens(names) {
  const allowDrums = names.includes('drums');
  const allowedPrograms = new Set();
  for (const name of names) {
    if (name === 'drums') continue;
    const group = INSTRUMENT_GROUPS[name];
    const programs = GROUP_PROGRAMS[group] || [];
    for (const program of programs) allowedPrograms.add(program);
  }
  const forbidden = new Set();
  for (let program = 0; program <= 129; program += 1) {
    if (!allowedPrograms.has(program)) forbidden.add(1135 + program);
  }
  if (!allowDrums) for (let pitch = 0; pitch < 128; pitch += 1) forbidden.add(1265 + pitch);
  return forbidden;
}

function argmax(values, forbidden) {
  let best = -1;
  let bestValue = -Infinity;
  const limit = Math.min(MUSCRIPTOR_CARD, values.length);
  for (let i = 0; i < limit; i += 1) {
    if (forbidden?.has(i)) continue;
    const v = values[i];
    if (v > bestValue) { bestValue = v; best = i; }
  }
  if (best < 0) throw new Error('MuScriptor produced no valid next token.');
  return best;
}

function instrumentForProgram(program) {
  return PROGRAM_NAMES.get(program) || `program_${program}`;
}

function buildHtkMelFilters() {
  const filters = Array.from({ length: FREQ_BINS }, () => []);
  const hzToMel = hz => 2595 * Math.log10(1 + hz / 700);
  const melToHz = mel => 700 * (10 ** (mel / 2595) - 1);
  const minMel = hzToMel(0);
  const maxMel = hzToMel(MUSCRIPTOR_SAMPLE_RATE / 2);
  const points = new Float64Array(MEL_BINS + 2);
  for (let i = 0; i < points.length; i += 1) {
    points[i] = melToHz(minMel + (maxMel - minMel) * i / (MEL_BINS + 1));
  }
  for (let m = 0; m < MEL_BINS; m += 1) {
    const left = points[m];
    const center = points[m + 1];
    const right = points[m + 2];
    for (let k = 0; k < FREQ_BINS; k += 1) {
      const hz = (MUSCRIPTOR_SAMPLE_RATE / 2) * k / (FREQ_BINS - 1);
      let w = 0;
      if (hz >= left && hz <= center && center > left) w = (hz - left) / (center - left);
      else if (hz > center && hz <= right && right > center) w = (right - hz) / (right - center);
      if (w > 0) filters[k].push(m, w);
    }
  }
  return filters;
}

function reflectIndex(index, length) {
  if (length <= 1) return 0;
  let i = index;
  while (i < 0 || i >= length) {
    if (i < 0) i = -i;
    if (i >= length) i = 2 * length - 2 - i;
  }
  return i;
}

function linearResample(input, sourceRate, targetRate) {
  if (sourceRate === targetRate) return new Float32Array(input);
  const length = Math.max(1, Math.round(input.length * targetRate / sourceRate));
  const out = new Float32Array(length);
  const scale = sourceRate / targetRate;
  for (let i = 0; i < length; i += 1) {
    const x = i * scale;
    const a = Math.min(input.length - 1, Math.floor(x));
    const b = Math.min(input.length - 1, a + 1);
    const frac = x - a;
    out[i] = input[a] * (1 - frac) + input[b] * frac;
  }
  return out;
}

function keyFor(program, pitch) { return `${program}:${pitch}`; }
function range(a, b) { return Array.from({ length: b - a + 1 }, (_, i) => a + i); }
function yieldToUi() { return new Promise(resolve => setTimeout(resolve, 0)); }
