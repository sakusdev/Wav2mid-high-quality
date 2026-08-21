// Cloudflare Workers Static Assets rejects individual files over 25 MiB.
// Keep the small ORT JSEP module/worker glue on this app's origin, while the
// large WASM payload remains on a version-pinned CDN. A cross-origin JSEP .mjs
// can fail Worker construction/dynamic import in Chromium and poison ORT's
// shared WASM initialization, making every later backend report initWasm errors.
const ORT_VERSION = '1.27.0';
const ORT_DIST = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const ORT_SCRIPT = `${ORT_DIST}ort.all.min.js`;
const ORT_JSEP_MJS = '/ort-wasm/ort-wasm-simd-threaded.jsep.mjs';
const ORT_JSEP_WASM = `${ORT_DIST}ort-wasm-simd-threaded.jsep.wasm`;

const ort = await loadOrt();
installWasmPathGuard(ort);

export const env = ort.env;
export const InferenceSession = ort.InferenceSession;
export const Tensor = ort.Tensor;
export const TRACE = ort.TRACE;
export const TRACE_FUNC_BEGIN = ort.TRACE_FUNC_BEGIN;
export const TRACE_FUNC_END = ort.TRACE_FUNC_END;
export default ort;

// Experimental knob used by MuScriptor FAST. The JSEP artifact is already the
// SIMD + threaded build; this selects how many pthread workers ORT may use for
// WASM/JSEP-side work before the first session initializes the runtime.
// The normal profile intentionally leaves ORT defaults untouched.
export function configureOrtRuntimeProfile(profile = 'default') {
  const wasm = ort?.env?.wasm;
  if (!wasm) {
    return { profile, applied: false, numThreads: 1, proxy: false, reason: 'env.wasm unavailable' };
  }

  if (profile !== 'fast') {
    return {
      profile: 'default',
      applied: false,
      numThreads: Number(wasm.numThreads) || null,
      proxy: Boolean(wasm.proxy),
    };
  }

  const isolated = globalThis.crossOriginIsolated === true;
  const cores = Math.max(1, Number(globalThis.navigator?.hardwareConcurrency) || 4);
  // Keep enough CPU headroom for the UI/audio thread on mobile. Pixel-class
  // 8-core devices land at 4 threads, 4-core devices at 2, and non-isolated
  // pages must remain single-threaded because SharedArrayBuffer is unavailable.
  const requestedThreads = isolated
    ? Math.max(2, Math.min(4, Math.floor(cores / 2)))
    : 1;

  try { wasm.numThreads = requestedThreads; } catch { /* ORT may already be initialized. */ }
  try { wasm.proxy = false; } catch { /* Keep direct calls if this build exposes proxy as readonly. */ }

  const effectiveThreads = Math.max(1, Number(wasm.numThreads) || requestedThreads);
  return {
    profile: 'fast',
    applied: true,
    numThreads: effectiveThreads,
    requestedThreads,
    proxy: Boolean(wasm.proxy),
    crossOriginIsolated: isolated,
    hardwareConcurrency: cores,
    artifact: 'simd-threaded-jsep',
  };
}

function stableWasmPaths() {
  return {
    mjs: new URL(ORT_JSEP_MJS, location.origin).href,
    wasm: ORT_JSEP_WASM,
  };
}

function installWasmPathGuard(runtime) {
  const wasm = runtime?.env?.wasm;
  if (!wasm) throw new Error('ONNX Runtime Web loaded without env.wasm.');

  let configuredPaths = stableWasmPaths();
  Object.defineProperty(wasm, 'wasmPaths', {
    configurable: true,
    enumerable: true,
    get() {
      return configuredPaths;
    },
    set(value) {
      // Compatibility guard for older NEURAL HQ code that still assigns the
      // legacy /ort-wasm/ prefix. That prefix intentionally contains only the
      // JSEP .mjs glue; the oversized .wasm must remain on the CDN.
      if (value === '/ort-wasm/' || value === `${location.origin}/ort-wasm/`) {
        configuredPaths = stableWasmPaths();
        return;
      }
      configuredPaths = value;
    },
  });
  wasm.wasmPaths = stableWasmPaths();
}

async function loadOrt() {
  if (globalThis.ort?.InferenceSession) return globalThis.ort;
  const existing = document.querySelector('script[data-wav2mid-ort]');
  if (existing) {
    await new Promise((resolve, reject) => {
      if (globalThis.ort?.InferenceSession) return resolve();
      existing.addEventListener('load', resolve, { once: true });
      existing.addEventListener('error', () => reject(new Error('ONNX Runtime CDN load failed.')), { once: true });
    });
    return globalThis.ort;
  }

  await new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = ORT_SCRIPT;
    script.async = true;
    script.crossOrigin = 'anonymous';
    script.dataset.wav2midOrt = ORT_VERSION;
    script.onload = resolve;
    script.onerror = () => reject(new Error(`Failed to load ONNX Runtime Web ${ORT_VERSION} from jsDelivr.`));
    document.head.appendChild(script);
  });

  if (!globalThis.ort?.InferenceSession) {
    throw new Error('ONNX Runtime Web loaded without the expected global API.');
  }
  return globalThis.ort;
}
