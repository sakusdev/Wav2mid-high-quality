import { cp, mkdir, readdir, copyFile, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const publicDir = join(root, 'public');
const modelSrc = join(root, 'node_modules', '@spotify', 'basic-pitch', 'model');
const modelDst = join(publicDir, 'model', 'basic-pitch');
const tfWasmSrc = join(root, 'node_modules', '@tensorflow', 'tfjs-backend-wasm', 'dist');
const tfWasmDst = join(publicDir, 'tfjs-wasm');

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

// ONNX Runtime is loaded only by NEURAL HQ. Its WebGPU JSEP WASM is larger
// than Cloudflare Workers Static Assets' 25 MiB/file limit, so NEURAL HQ
// points ORT at its versioned jsDelivr distribution instead of copying it here.
console.log('Prepared Basic Pitch model and TensorFlow.js WASM runtime. NEURAL HQ ORT runtime stays on-demand.');
