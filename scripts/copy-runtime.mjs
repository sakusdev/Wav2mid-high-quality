import { cp, mkdir, readdir, copyFile, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const publicDir = join(root, 'public');
const modelSrc = join(root, 'node_modules', '@spotify', 'basic-pitch', 'model');
const modelDst = join(publicDir, 'model', 'basic-pitch');
const tfWasmSrc = join(root, 'node_modules', '@tensorflow', 'tfjs-backend-wasm', 'dist');
const tfWasmDst = join(publicDir, 'tfjs-wasm');
const ortDistSrc = join(root, 'node_modules', 'onnxruntime-web', 'dist');
const ortWasmDst = join(publicDir, 'ort-wasm');
const ortJsepGlue = 'ort-wasm-simd-threaded.jsep.mjs';

await mkdir(publicDir, { recursive: true });
await rm(modelDst, { recursive: true, force: true });
await mkdir(modelDst, { recursive: true });
await cp(modelSrc, modelDst, { recursive: true });

await rm(tfWasmDst, { recursive: true, force: true });
await mkdir(tfWasmDst, { recursive: true });
for (const name of await readdir(tfWasmSrc)) {
  if (name.endsWith('.wasm')) {
    await copyFile(join(tfWasmSrc, name), join(tfWasmDst, name));
  }
}

// Keep ORT's JavaScript worker/JSEP glue on the app origin. Chromium may reject
// constructing a Worker from a cross-origin CDN module even when the CDN itself
// has CORS enabled. The large JSEP WASM binary still stays on the version-pinned
// CDN because it exceeds Cloudflare Workers Static Assets' 25 MiB/file limit.
await rm(ortWasmDst, { recursive: true, force: true });
await mkdir(ortWasmDst, { recursive: true });
await copyFile(join(ortDistSrc, ortJsepGlue), join(ortWasmDst, ortJsepGlue));

console.log('Prepared Basic Pitch model, TensorFlow.js WASM, and same-origin ORT JSEP glue. Large NEURAL HQ ORT WASM stays on-demand.');
