const LOADER_KEY = '__WAV2MID_MUSCRIPTOR_MODEL_LOADER__';
const STATE_KEY = '__WAV2MID_MUSCRIPTOR_FAST_STATE__';
const STORAGE_KEY = 'wav2mid.muscriptor.fastExperimental';

const FAST_MAX_GENERATION = 512;
const FAST_CHUNK_TICKS = 500; // 5 seconds at MuScriptor's 100 Hz event clock.
const MUSCRIPTOR_INITIAL_TOKEN = 1393;
const MUSCRIPTOR_EOS = 1;
const MUSCRIPTOR_CARD = 1393;
const TOKEN_PROGRESS_EVERY = 16;
const TOKEN_PROGRESS_MAX_SILENCE_MS = 500;

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

const state = {
  enabled: readPreference(),
  maxTokensPerChunk: FAST_MAX_GENERATION,
  stopAtChunkBoundary: true,
  wasm: null,
  modelLoaded: false,
};
globalThis[STATE_KEY] = state;

installLoaderWrapper();
initUiWhenReady();

function installLoaderWrapper() {
  const loader = globalThis[LOADER_KEY];
  if (typeof loader !== 'function') {
    setTimeout(installLoaderWrapper, 0);
    return;
  }
  if (loader.__wav2midFastExperimental) return;

  const wrapped = async (...args) => {
    const ort = await import('./ort-cdn-shim.js');
    state.wasm = ort.configureOrtRuntimeProfile?.(state.enabled ? 'fast' : 'default') || null;
    const model = await loader(...args);
    installFastGenerator(model, ort);
    state.modelLoaded = true;
    refreshFastUi();
    return model;
  };

  Object.defineProperty(wrapped, '__wav2midFastExperimental', { value: true });
  globalThis[LOADER_KEY] = wrapped;
}

function initUiWhenReady() {
  const config = document.getElementById('ultraConfig');
  if (!config) {
    requestAnimationFrame(initUiWhenReady);
    return;
  }
  if (document.getElementById('muscriptorFastToggle')) return;

  const box = document.createElement('label');
  box.className = 'muscriptor-fast-experimental';
  box.innerHTML = `
    <span class="muscriptor-fast-row">
      <input id="muscriptorFastToggle" type="checkbox" />
      <strong>FAST <em>EXPERIMENTAL</em></strong>
      <small id="muscriptorFastState"></small>
    </span>
    <span class="muscriptor-fast-copy">512 tok/chunk · 5秒境界で早期終了 · ORT WASM/JSEP 2–4 threads</span>
  `;
  config.appendChild(box);

  installFastStyles();
  const toggle = document.getElementById('muscriptorFastToggle');
  toggle.checked = state.enabled;
  toggle.addEventListener('change', () => {
    state.enabled = Boolean(toggle.checked);
    writePreference(state.enabled);
    refreshFastUi();
  });
  refreshFastUi();
}

function refreshFastUi() {
  const toggle = document.getElementById('muscriptorFastToggle');
  const status = document.getElementById('muscriptorFastState');
  if (toggle) toggle.checked = state.enabled;
  if (!status) return;

  if (!state.enabled) {
    status.textContent = 'OFF · normal 2000 tok';
    return;
  }
  const threads = state.wasm?.numThreads;
  const wasmText = Number.isFinite(threads) ? `${threads}t WASM` : 'WASM tune on load';
  status.textContent = state.modelLoaded ? `ON · ${wasmText}` : `ON · ${wasmText}`;
}

function installFastStyles() {
  if (document.getElementById('muscriptorFastStyles')) return;
  const style = document.createElement('style');
  style.id = 'muscriptorFastStyles';
  style.textContent = `
    .muscriptor-fast-experimental{grid-column:1/-1;display:grid;gap:4px;border:1px dashed #3b3b40;border-radius:6px;padding:8px 9px;background:#111113;cursor:pointer}
    .muscriptor-fast-row{display:grid;grid-template-columns:auto auto 1fr;gap:7px;align-items:center}
    .muscriptor-fast-row input{width:14px;height:14px;margin:0;accent-color:#d7d7db}
    .muscriptor-fast-row strong{font:600 10px ui-monospace,SFMono-Regular,Menlo,monospace;color:#d7d7db}
    .muscriptor-fast-row em{font-style:normal;font-weight:500;color:#8d8d94}
    .muscriptor-fast-row small{justify-self:end;color:#8d8d94;font:500 9px ui-monospace,SFMono-Regular,Menlo,monospace}
    .muscriptor-fast-copy{color:#77777e;font-size:9px;line-height:1.45}
  `;
  document.head.appendChild(style);
}

function installFastGenerator(model, ort) {
  if (!model || typeof model.generateChunk !== 'function' || model.__wav2midFastGeneratorInstalled) return;
  const normalGenerateChunk = model.generateChunk;

  model.generateChunk = async function generateChunkWithProfile(...args) {
    if (!state.enabled) return normalGenerateChunk.apply(this, args);
    return generateChunkFast.call(this, ort, ...args);
  };

  try {
    Object.defineProperty(model, '__wav2midFastGeneratorInstalled', { value: true });
  } catch {
    model.__wav2midFastGeneratorInstalled = true;
  }
}

async function generateChunkFast(ort, prefix, prompt, forbidden, tracker, openEvents, notes, drums, onTokenProgress) {
  const caches = this.createCaches();
  const startedAt = performance.now();
  let lastProgressAt = startedAt;
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

    for (let generated = 0; generated < FAST_MAX_GENERATION; generated += 1) {
      let token;
      try {
        token = argmax(step.logits.data, forbidden);
        await this.appendNewKv(step, caches, pastLen - step.queryLength);
      } finally {
        step.dispose();
        step = null;
      }

      const generatedCount = generated + 1;
      const now = performance.now();
      const shift = decodeShift(token);
      const reachedBoundary = shift != null && shift >= FAST_CHUNK_TICKS;
      const forceProgress = generatedCount === 1
        || token === MUSCRIPTOR_EOS
        || reachedBoundary
        || generatedCount === FAST_MAX_GENERATION;

      if (forceProgress
        || generatedCount % TOKEN_PROGRESS_EVERY === 0
        || now - lastProgressAt >= TOKEN_PROGRESS_MAX_SILENCE_MS) {
        const elapsedSeconds = Math.max((now - startedAt) / 1000, 0.001);
        onTokenProgress?.({
          generated: generatedCount,
          max: FAST_MAX_GENERATION,
          token,
          elapsedSeconds,
          tokPerSecond: generatedCount / elapsedSeconds,
          eos: token === MUSCRIPTOR_EOS,
          fast: true,
          boundary: reachedBoundary,
        });
        lastProgressAt = now;
      }

      if (token === MUSCRIPTOR_EOS) {
        return resultFor(generatedCount, startedAt, { eos: true });
      }

      consumeActions(tracker.feed(token), openEvents, notes, drums);

      // Any event at/after +5.00s is discarded by the normal tracker because
      // the next MuScriptor chunk owns that time range. In FAST mode there is
      // therefore no value in autoregressively decoding beyond this boundary.
      if (reachedBoundary) {
        return resultFor(generatedCount, startedAt, { fastBoundary: true });
      }

      if (generatedCount === FAST_MAX_GENERATION) {
        console.warn(`MuScriptor FAST chunk reached ${FAST_MAX_GENERATION} tokens; continuing with the next chunk.`);
        return resultFor(generatedCount, startedAt, { capped: true });
      }

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
    throw new Error('MuScriptor FAST generation loop exited unexpectedly.');
  } finally {
    step?.dispose();
    for (const cache of caches) {
      cache.tensor.dispose?.();
      cache.buffer.destroy();
    }
  }
}

function resultFor(tokens, startedAt, extra = {}) {
  return {
    eos: false,
    capped: false,
    tokens,
    elapsedSeconds: Math.max((performance.now() - startedAt) / 1000, 0),
    fast: true,
    ...extra,
  };
}

function decodeShift(token) {
  const value = Number(token);
  return value >= 3 && value <= 1003 ? value - 3 : null;
}

function argmax(values, forbidden) {
  let best = -1;
  let bestValue = -Infinity;
  const limit = Math.min(MUSCRIPTOR_CARD, values.length);
  for (let i = 0; i < limit; i += 1) {
    if (forbidden?.has(i)) continue;
    const value = values[i];
    if (value > bestValue) {
      bestValue = value;
      best = i;
    }
  }
  if (best < 0) throw new Error('MuScriptor FAST produced no valid next token.');
  return best;
}

function consumeActions(actions, openEvents, notes, drums) {
  for (const action of actions) {
    if (action.type === 'drum') {
      drums.push({ midi: action.pitch, name: 'drums', time: action.time, duration: 0.01, velocity: 0.8 });
      continue;
    }
    const key = `${action.program}:${action.pitch}`;
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
      instrument: PROGRAM_NAMES.get(action.program) || `program_${action.program}`,
      program: action.program,
      confidence: 1,
      amplitude: 1,
    });
  }
}

function readPreference() {
  try {
    return globalThis.localStorage?.getItem(STORAGE_KEY) === '1';
  } catch {
    return false;
  }
}

function writePreference(enabled) {
  try {
    globalThis.localStorage?.setItem(STORAGE_KEY, enabled ? '1' : '0');
  } catch {
    // Preference persistence is optional.
  }
}
