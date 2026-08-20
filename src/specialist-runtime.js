import { deframeSpecialistOutputs, logMelFromWaveform, makeSpecialistSegments, normalizeMono16k } from './specialist-frontend.js';
import { postprocessSpecialistOutputs } from './specialist-postprocess.js';

const DEFAULT_MANIFEST_URL = '/specialists/manifest.json';
const sessionCache = new Map();
let manifestPromise = null;

export async function loadSpecialistManifest(url = DEFAULT_MANIFEST_URL, { refresh = false } = {}) {
  if (!refresh && manifestPromise && url === DEFAULT_MANIFEST_URL) return manifestPromise;
  const promise = fetch(url, { cache: 'no-cache' })
    .then(async response => {
      if (!response.ok) throw new Error(`Specialist manifest HTTP ${response.status}`);
      const manifest = await response.json();
      validateManifest(manifest);
      return manifest;
    })
    .catch(error => {
      console.warn('Specialist manifest unavailable; continuing without specialist models.', error);
      return { version: 1, models: {} };
    });
  if (url === DEFAULT_MANIFEST_URL) manifestPromise = promise;
  return promise;
}

export function promotedSpecialists(manifest, stem = null) {
  return Object.entries(manifest?.models ?? {})
    .filter(([, model]) => model?.enabled !== false && model?.promoted === true && model?.url)
    .filter(([, model]) => !stem || model.stem === stem)
    .map(([name, model]) => ({ name, ...model }));
}

export async function transcribeWithSpecialist(audioLike, modelConfig, onProgress = () => {}) {
  if (!modelConfig?.url) throw new Error('Specialist model URL is missing.');
  onProgress({ stage: 'specialist-prepare', value: 0.01, detail: `${modelConfig.name ?? modelConfig.instrument ?? 'model'} · 16 kHz mono` });
  const samples = normalizeMono16k(audioLike);
  const { segments, originalSamples } = makeSpecialistSegments(samples, Number(modelConfig.segmentSeconds ?? 10));
  const session = await getSession(modelConfig, onProgress);
  const segmentOutputs = [];

  for (let index = 0; index < segments.length; index += 1) {
    const feature = logMelFromWaveform(segments[index]);
    const { Tensor } = await getOrt();
    const inputName = modelConfig.inputName ?? session.inputNames?.[0] ?? 'logmel';
    const feeds = { [inputName]: new Tensor('float32', feature.data, feature.dims) };
    const results = await session.run(feeds);
    segmentOutputs.push(normalizeOutputs(results, modelConfig));
    onProgress({
      stage: 'specialist-infer',
      value: 0.08 + ((index + 1) / Math.max(1, segments.length)) * 0.82,
      detail: `${modelConfig.name ?? modelConfig.instrument ?? 'model'} · ${index + 1}/${segments.length}`,
    });
    await new Promise(resolve => setTimeout(resolve, 0));
  }

  const outputs = deframeSpecialistOutputs(segmentOutputs, originalSamples);
  const notes = postprocessSpecialistOutputs(outputs, {
    classes: Number(modelConfig.classes ?? 88),
    beginNote: Number(modelConfig.beginNote ?? 21),
    framesPerSecond: Number(modelConfig.framesPerSecond ?? 100),
    onsetThreshold: Number(modelConfig.onsetThreshold ?? 0.3),
    offsetThreshold: Number(modelConfig.offsetThreshold ?? 0.3),
    frameThreshold: Number(modelConfig.frameThreshold ?? 0.1),
    source: `specialist:${modelConfig.name ?? modelConfig.instrument ?? 'crnn'}`,
    instrument: modelConfig.instrument === 'bass' ? 'bass' : 'harmony',
  });

  const maxPitch = Number(modelConfig.maxPitch ?? 127);
  const minPitch = Number(modelConfig.minPitch ?? 0);
  const filtered = notes.filter(note => note.pitchMidi >= minPitch && note.pitchMidi <= maxPitch);
  onProgress({ stage: 'specialist-done', value: 1, detail: `${filtered.length} notes` });
  return {
    notes: filtered,
    outputs,
    model: modelConfig,
    stats: {
      segments: segments.length,
      noteCount: filtered.length,
      inputSamples: originalSamples,
      sampleRate: 16000,
    },
  };
}

export async function runPromotedStemSpecialists(stems, manifest, onProgress = () => {}) {
  const results = [];
  const models = promotedSpecialists(manifest);
  for (let index = 0; index < models.length; index += 1) {
    const model = models[index];
    const stem = stems?.[model.stem];
    if (!stem) continue;
    const result = await transcribeWithSpecialist(stemLikeAudio(stem), model, progress => {
      onProgress({
        ...progress,
        value: (index + Number(progress.value ?? 0)) / Math.max(1, models.length),
      });
    });
    results.push({ ...result, name: model.name, stem: model.stem });
  }
  return results;
}

async function getSession(modelConfig, onProgress) {
  const cacheKey = `${modelConfig.url}|${modelConfig.executionProvider ?? 'auto'}`;
  if (sessionCache.has(cacheKey)) return sessionCache.get(cacheKey);
  const promise = (async () => {
    onProgress({ stage: 'specialist-load', value: 0.03, detail: modelConfig.name ?? modelConfig.instrument ?? 'CRNN' });
    const ort = await getOrt();
    const useWebGpu = modelConfig.executionProvider !== 'wasm' && await hasWebGPU();
    const providers = useWebGpu ? ['webgpu', 'wasm'] : ['wasm'];
    if (ort.env?.webgpu && useWebGpu) ort.env.webgpu.powerPreference = 'high-performance';
    if (ort.env?.wasm) {
      ort.env.wasm.numThreads = globalThis.crossOriginIsolated
        ? Math.max(1, Math.min(8, Number(navigator.hardwareConcurrency ?? 4)))
        : 1;
    }
    return ort.InferenceSession.create(modelConfig.url, {
      executionProviders: providers,
      graphOptimizationLevel: 'all',
      enableCpuMemArena: false,
      enableMemPattern: false,
    });
  })().catch(error => {
    sessionCache.delete(cacheKey);
    throw error;
  });
  sessionCache.set(cacheKey, promise);
  return promise;
}

let ortPromise = null;
async function getOrt() {
  if (!ortPromise) ortPromise = import('./ort-cdn-shim.js');
  return ortPromise;
}

function normalizeOutputs(results, modelConfig) {
  const names = {
    reg_onset_output: modelConfig.outputNames?.reg_onset_output ?? 'reg_onset_output',
    reg_offset_output: modelConfig.outputNames?.reg_offset_output ?? 'reg_offset_output',
    frame_output: modelConfig.outputNames?.frame_output ?? 'frame_output',
    velocity_output: modelConfig.outputNames?.velocity_output ?? 'velocity_output',
  };
  const out = {};
  for (const [canonical, name] of Object.entries(names)) {
    const tensor = results[name];
    if (!tensor) throw new Error(`Specialist ONNX output ${name} (${canonical}) is missing.`);
    out[canonical] = {
      data: tensor.data instanceof Float32Array ? tensor.data : Float32Array.from(tensor.data),
      dims: [...tensor.dims],
    };
  }
  return out;
}

function validateManifest(manifest) {
  if (!manifest || Number(manifest.version ?? 0) < 1 || typeof manifest.models !== 'object') {
    throw new Error('Invalid specialist manifest.');
  }
}

async function hasWebGPU() {
  if (!globalThis.navigator?.gpu) return false;
  try { return Boolean(await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })); }
  catch { return false; }
}

function stemLikeAudio(stem) {
  if (stem?.sampleRate && stem?.samples) return stem;
  const left = stem?.left ?? stem;
  const right = stem?.right ?? left;
  if (!(left instanceof Float32Array)) throw new Error('Specialist stem has no Float32 audio data.');
  return {
    sampleRate: Number(stem?.sampleRate ?? 44100),
    numberOfChannels: right === left ? 1 : 2,
    length: Math.min(left.length, right.length),
    left,
    right,
  };
}
