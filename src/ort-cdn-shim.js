// Cloudflare Workers Static Assets rejects individual files over 25 MiB.
// ONNX Runtime Web 1.27's WebGPU JSEP WASM exceeds that limit, so the optional
// NEURAL HQ path loads the version-pinned ORT distribution from jsDelivr only
// when this lazily imported module is evaluated.
const ORT_VERSION = '1.27.0';
const ORT_DIST = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const ORT_SCRIPT = `${ORT_DIST}ort.all.min.js`;

const ort = await loadOrt();
ort.env.wasm.wasmPaths = ORT_DIST;

export const env = ort.env;
export const InferenceSession = ort.InferenceSession;
export const Tensor = ort.Tensor;
export const TRACE = ort.TRACE;
export const TRACE_FUNC_BEGIN = ort.TRACE_FUNC_BEGIN;
export const TRACE_FUNC_END = ort.TRACE_FUNC_END;
export default ort;

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
