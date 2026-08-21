const MODEL_MANIFEST_URL = '/models/muscriptor-small/manifest.json';
const DIAGNOSTIC_KEY = '__WAV2MID_MUSCRIPTOR_DIAGNOSTICS__';
const LOADER_KEY = '__WAV2MID_MUSCRIPTOR_MODEL_LOADER__';
const MAX_TRACE = 32;
const HEARTBEAT_MS = 5000;
const SLOW_STAGE_MS = 30000;

if (!globalThis[LOADER_KEY]) {
  globalThis[LOADER_KEY] = loadMuScriptorWithDiagnostics;
}

export async function loadMuScriptorWithDiagnostics() {
  const trace = [];
  publish(trace, 'manifest', 'running', MODEL_MANIFEST_URL);

  const manifest = await fetchManifest(trace);
  await probeModelAssets(manifest, trace);

  publish(trace, 'webgpu-adapter', 'running', 'browser WebGPU API');
  if (!globalThis.navigator?.gpu) {
    throw diagnosticError(trace, 'MUSCRIPTOR_WEBGPU_UNAVAILABLE', 'webgpu-adapter',
      'WebGPU APIが利用できません。',
      'Chromium系ブラウザでWebGPUを有効にし、対応GPU/ドライバを使用してください。');
  }
  // Do not allocate a throwaway adapter/device here. The browser core creates
  // the one real device that ONNX Runtime sessions will keep using. Creating a
  // probe GPUDevice and immediately destroying it was redundant and can upset
  // mobile GPU drivers immediately before the heavy ORT WebGPU initialization.
  publish(trace, 'webgpu-adapter', 'ok', 'WebGPU API ready; adapter/device delegated to model core');

  publish(trace, 'ort-runtime', 'running', 'ONNX Runtime WebGPU');
  let ort;
  try {
    ort = await import('onnxruntime-web/webgpu');
    publish(trace, 'ort-runtime', 'ok', 'ORT module ready');
  } catch (cause) {
    throw wrapRuntimeError(trace, cause, 'ort-runtime', 'MUSCRIPTOR_ORT_IMPORT');
  }

  const restoreOrt = instrumentOrtSessions(ort, manifest, trace);
  const stopHeartbeat = startLoadHeartbeat(trace);
  try {
    publish(trace, 'model-loader', 'running', 'MuScriptor browser core');
    const { loadMuScriptorBrowserModel } = await import('./muscriptor-browser-core.js');
    const model = await loadMuScriptorBrowserModel(MODEL_MANIFEST_URL);
    publish(trace, 'model-loader', 'ok', 'conditioner + decoder ready');
    return wrapModel(model, trace);
  } catch (cause) {
    if (cause?.name === 'MuScriptorDiagnosticError') throw cause;
    throw wrapRuntimeError(trace, cause, 'model-loader', 'MUSCRIPTOR_MODEL_LOAD');
  } finally {
    stopHeartbeat();
    restoreOrt();
  }
}

async function fetchManifest(trace) {
  let response;
  try {
    response = await fetch(MODEL_MANIFEST_URL, { cache: 'no-store' });
  } catch (cause) {
    throw diagnosticError(trace, 'MUSCRIPTOR_MANIFEST_FETCH', 'manifest',
      `モデルmanifest取得失敗: ${messageOf(cause)}`,
      'モデル配信先への接続、Cloudflareのデプロイ状態、ネットワーク制限を確認してください。', cause);
  }
  if (!response.ok) {
    throw diagnosticError(trace, 'MUSCRIPTOR_MANIFEST_HTTP', 'manifest',
      `モデルmanifest HTTP ${response.status}`,
      response.status === 404
        ? 'MuScriptorモデルがデプロイされていません。モデルexport/staging付きの本番デプロイを使用してください。'
        : 'モデル配信先のHTTPエラーです。Cloudflare Worker/Static Assetsを確認してください。');
  }

  let manifest;
  try {
    manifest = await response.json();
  } catch (cause) {
    throw diagnosticError(trace, 'MUSCRIPTOR_MANIFEST_JSON', 'manifest',
      `モデルmanifest JSON解析失敗: ${messageOf(cause)}`,
      'manifest.jsonが途中で壊れていないか、HTMLエラーページが返っていないか確認してください。', cause);
  }
  if (manifest?.format !== 'wav2mid-muscriptor-browser/v1') {
    throw diagnosticError(trace, 'MUSCRIPTOR_MANIFEST_FORMAT', 'manifest',
      `未対応のmanifest形式: ${manifest?.format || 'missing'}`,
      '現在のexport_muscriptor_browser_v3.pyでモデルを再exportしてください。');
  }
  if (!manifest?.files?.conditioner?.url || !manifest?.files?.decoder?.url) {
    throw diagnosticError(trace, 'MUSCRIPTOR_MANIFEST_FILES', 'manifest',
      'conditioner/decoderのURLがmanifestにありません。',
      'MuScriptor browser modelを再exportしてCloudflare stagingをやり直してください。');
  }
  publish(trace, 'manifest', 'ok', `format=${manifest.format}`);
  return manifest;
}

async function probeModelAssets(manifest, trace) {
  for (const name of ['conditioner', 'decoder']) {
    const entry = manifest.files[name];
    const stage = `${name}-asset`;
    publish(trace, stage, 'running', entry.url);
    let response;
    try {
      response = await fetch(entry.url, { method: 'HEAD', cache: 'no-store' });
    } catch (cause) {
      throw diagnosticError(trace, 'MUSCRIPTOR_ASSET_FETCH', stage,
        `${name}モデル配信確認失敗: ${messageOf(cause)}`,
        'Cloudflare Static Assets/streaming Workerとネットワーク接続を確認してください。', cause);
    }
    // Some generic static hosts reject HEAD. In that case ORT will perform the
    // real GET later, so do not turn 405/501 into a false negative.
    if (!response.ok && response.status !== 405 && response.status !== 501) {
      throw diagnosticError(trace, 'MUSCRIPTOR_ASSET_HTTP', stage,
        `${name}モデル HTTP ${response.status}: ${entry.url}`,
        response.status === 404
          ? `${name}モデルがデプロイに含まれていません。MuScriptor model stagingを確認してください。`
          : 'Cloudflareのモデル配信経路を確認してください。');
    }
    const actualBytes = Number(response.headers.get('content-length') || 0);
    const expectedBytes = Number(entry.bytes || 0);
    if (response.ok && actualBytes > 0 && expectedBytes > 0 && actualBytes !== expectedBytes) {
      throw diagnosticError(trace, 'MUSCRIPTOR_ASSET_SIZE', stage,
        `${name}モデルのサイズ不一致: expected ${expectedBytes}, got ${actualBytes}`,
        '分割モデルのstream-map/partsが古い可能性があります。モデルを再stageして再デプロイしてください。');
    }
    publish(trace, stage, 'ok', actualBytes > 0 ? `${formatMiB(actualBytes)} MiB` : `HTTP ${response.status}`);
  }
}

function instrumentOrtSessions(ort, manifest, trace) {
  const Session = ort?.InferenceSession;
  if (!Session || typeof Session.create !== 'function') return () => {};
  const originalCreate = Session.create;
  const conditionerUrl = String(manifest.files.conditioner.url);
  const decoderUrl = String(manifest.files.decoder.url);

  const wrappedCreate = async function diagnosticSessionCreate(source, options) {
    const sourceText = typeof source === 'string' ? source : '';
    const kind = sourceText === conditionerUrl || /conditioner/i.test(sourceText)
      ? 'conditioner'
      : sourceText === decoderUrl || /decoder/i.test(sourceText)
        ? 'decoder'
        : 'ort';
    const stage = `${kind}-session`;
    publish(trace, stage, 'running', sourceText || 'buffer');
    let session;
    try {
      session = await originalCreate.call(Session, source, options);
    } catch (cause) {
      throw wrapRuntimeError(trace, cause, stage, `MUSCRIPTOR_${kind.toUpperCase()}_SESSION`);
    }
    publish(trace, stage, 'ok', 'session ready');
    wrapSessionRun(session, kind, trace);
    return session;
  };

  let patched = false;
  try {
    Session.create = wrappedCreate;
    patched = Session.create === wrappedCreate;
  } catch {
    // Some ORT builds may freeze static members. Diagnostics still retain
    // manifest/asset/WebGPU stages and the outer model-loader classification.
  }
  return () => {
    if (!patched) return;
    try { Session.create = originalCreate; } catch { /* no-op */ }
  };
}

function wrapSessionRun(session, kind, trace) {
  if (!session || typeof session.run !== 'function' || session.__wav2midDiagnosticRun) return;
  const originalRun = session.run.bind(session);
  const wrapped = async (...args) => {
    const stage = `${kind}-inference`;
    publish(trace, stage, 'running', 'session.run');
    try {
      const result = await originalRun(...args);
      publish(trace, stage, 'ok', 'run complete');
      return result;
    } catch (cause) {
      throw wrapRuntimeError(trace, cause, stage, `MUSCRIPTOR_${kind.toUpperCase()}_INFERENCE`);
    }
  };
  try {
    session.run = wrapped;
    Object.defineProperty(session, '__wav2midDiagnosticRun', { value: true });
  } catch {
    // Non-extensible session objects simply keep the coarser outer diagnostic.
  }
}

function wrapModel(model, trace) {
  if (!model || typeof model.transcribe !== 'function') return model;
  return new Proxy(model, {
    get(target, prop, receiver) {
      if (prop !== 'transcribe') return Reflect.get(target, prop, receiver);
      return async (...args) => {
        publish(trace, 'transcription', 'running', 'audio -> MuScriptor');
        try {
          const result = await target.transcribe(...args);
          publish(trace, 'transcription', 'ok', `${result?.notes?.length ?? 0} notes`);
          return result;
        } catch (cause) {
          if (cause?.name === 'MuScriptorDiagnosticError') throw cause;
          throw wrapRuntimeError(trace, cause, 'transcription', 'MUSCRIPTOR_TRANSCRIPTION');
        }
      };
    },
  });
}

function wrapRuntimeError(trace, cause, stage, fallbackCode) {
  const text = messageOf(cause);
  const lower = text.toLowerCase();
  if (/no webgpu adapter|requestadapter|adapter.*unavailable/.test(lower)) {
    return diagnosticError(trace, 'MUSCRIPTOR_WEBGPU_ADAPTER', stage,
      `WebGPU adapter取得失敗: ${text}`,
      'GPUドライバ、ブラウザのWebGPU設定、ハードウェアアクセラレーションを確認してください。', cause);
  }
  if (/requestdevice|failed to create.*device|gpu device.*failed/.test(lower)) {
    return diagnosticError(trace, 'MUSCRIPTOR_WEBGPU_DEVICE', stage,
      `WebGPU device作成失敗: ${text}`,
      'GPUメモリ不足またはドライバ制限の可能性があります。ほかのGPU負荷を下げて再試行してください。', cause);
  }
  if (/out of memory|oom|device lost|gpu.*lost|failed to allocate|createbuffer/.test(lower)) {
    return diagnosticError(trace, 'MUSCRIPTOR_GPU_MEMORY', stage,
      `GPUメモリ/デバイスエラー: ${text}`,
      'ほかのGPU負荷を終了し、短い音源で再試行してください。Androidでは特にGPUメモリ上限に注意してください。', cause);
  }
  if (/initwasm|jsep|failed to fetch dynamically imported module|ort-wasm/.test(lower)) {
    return diagnosticError(trace, 'MUSCRIPTOR_ORT_RUNTIME_FETCH', stage,
      `ONNX Runtime JSEP/WASM初期化失敗: ${text}`,
      'ページを再読み込みしてORT初期化状態をリセットし、/ort-wasm/*.mjs と jsDelivr WASMへの接続を確認してください。', cause);
  }
  if (/kv cache limit/i.test(text)) {
    return diagnosticError(trace, 'MUSCRIPTOR_DECODER_KV_LIMIT', stage, text,
      '1チャンクの生成が長すぎます。モデルexportのmaxCacheまたは生成上限を確認してください。', cause);
  }
  if (/did not emit eos|within .* tokens/i.test(text)) {
    return diagnosticError(trace, 'MUSCRIPTOR_DECODER_NO_EOS', stage, text,
      'decoderがEOSを生成できませんでした。checkpoint/export互換性を確認してください。', cause);
  }
  return diagnosticError(trace, fallbackCode, stage, text,
    '表示されたstage/codeを使って該当するモデル配信・ORT・WebGPU工程を確認してください。', cause);
}

function diagnosticError(trace, code, stage, message, hint, cause) {
  publish(trace, stage, 'error', `${code}: ${message}`);
  const error = new Error(`[${code}] ${stage}: ${message}`);
  error.name = 'MuScriptorDiagnosticError';
  error.code = code;
  error.stage = stage;
  error.hint = hint;
  if (cause !== undefined) error.cause = cause;
  surfaceHint(hint, code);
  return error;
}

function publish(trace, stage, status, detail) {
  trace.push({ stage, status, detail: String(detail || ''), at: performance.now() });
  if (trace.length > MAX_TRACE) trace.splice(0, trace.length - MAX_TRACE);
  globalThis[DIAGNOSTIC_KEY] = trace.map(item => ({ ...item }));
  console.debug(`[MuScriptor:${stage}] ${status}${detail ? ` · ${detail}` : ''}`);
  const state = document.getElementById('ultraState');
  if (state && status === 'running') state.textContent = shortStage(stage);
  surfaceProgress(stage, status);
}

function startLoadHeartbeat(trace) {
  const timer = setInterval(() => {
    const running = latestRunningStage(trace);
    if (!running) return;
    const elapsedMs = Math.max(0, performance.now() - running.at);
    surfaceProgress(running.stage, 'running', elapsedMs);
    if (elapsedMs >= SLOW_STAGE_MS && /session|model-loader/.test(running.stage)) {
      const target = document.getElementById('progressHint');
      if (target) {
        target.textContent = '初回はモデル取得とWebGPUコンパイルに時間がかかります。表示中のstageと経過秒数が動いていれば処理は継続中です。';
      }
    }
  }, HEARTBEAT_MS);
  return () => clearInterval(timer);
}

function latestRunningStage(trace) {
  const latestByStage = new Map();
  for (const item of trace) latestByStage.set(item.stage, item);
  const running = [...latestByStage.values()].filter(item => item.status === 'running');
  return running.sort((a, b) => b.at - a.at)[0] || null;
}

function surfaceProgress(stage, status, elapsedMs = 0) {
  if (status !== 'running') return;
  const stages = {
    manifest: [0.010, 'MuScriptor: manifest確認中…'],
    'conditioner-asset': [0.014, 'MuScriptor: conditioner配信確認中…'],
    'decoder-asset': [0.018, 'MuScriptor: decoder配信確認中…'],
    'webgpu-adapter': [0.022, 'MuScriptor: WebGPU API確認中…'],
    'ort-runtime': [0.026, 'MuScriptor: ONNX Runtime初期化中…'],
    'model-loader': [0.030, 'MuScriptor: GPU adapter/device作成中…'],
    'conditioner-session': [0.034, 'MuScriptor: conditionerモデル読込・compile中…'],
    'decoder-session': [0.038, 'MuScriptor: decoderモデル読込・compile中…'],
  };
  const entry = stages[stage];
  if (!entry) return;
  const [value, label] = entry;
  const seconds = elapsedMs >= HEARTBEAT_MS ? ` · ${Math.round(elapsedMs / 1000)}秒` : '';
  const panel = document.getElementById('progressPanel');
  const text = document.getElementById('progressText');
  const pct = document.getElementById('progressPct');
  const bar = document.getElementById('progressBar');
  if (panel) panel.hidden = false;
  if (text) text.textContent = `${label}${seconds}`;
  if (pct) pct.textContent = `${Math.max(1, Math.round(value * 100))}%`;
  if (bar) bar.style.width = `${Math.max(1, value * 100)}%`;
  const state = document.getElementById('ultraState');
  if (state && elapsedMs >= HEARTBEAT_MS) state.textContent = `${shortStage(stage)} ${Math.round(elapsedMs / 1000)}s`;
}

function surfaceHint(hint, code) {
  setTimeout(() => {
    const target = document.getElementById('progressHint');
    if (target) target.textContent = `${hint} (${code})`;
  }, 0);
}

function shortStage(stage) {
  const names = {
    manifest: 'manifest',
    'conditioner-asset': 'model',
    'decoder-asset': 'model',
    'webgpu-adapter': 'gpu',
    'webgpu-device': 'gpu',
    'ort-runtime': 'ort',
    'model-loader': 'init',
    'conditioner-session': 'cond',
    'decoder-session': 'dec',
    'conditioner-inference': 'cond',
    'decoder-inference': 'decode',
    transcription: 'infer',
  };
  return names[stage] || 'loading';
}

function messageOf(error) {
  return String(error?.message || error || 'unknown error');
}

function formatMiB(bytes) {
  return (bytes / 1024 / 1024).toFixed(1);
}
