import { cp, mkdir, readdir, copyFile, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const publicDir = join(root, 'public');
const modelSrc = join(root, 'node_modules', '@spotify', 'basic-pitch', 'model');
const modelDst = join(publicDir, 'model', 'basic-pitch');
const tfWasmSrc = join(root, 'node_modules', '@tensorflow', 'tfjs-backend-wasm', 'dist');
const tfWasmDst = join(publicDir, 'tfjs-wasm');
const ortSrc = join(root, 'node_modules', 'onnxruntime-web', 'dist');
const ortDst = join(publicDir, 'ort-wasm');

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

// Keep ORT's glue modules and WASM binaries on the same origin. Loading the
// JavaScript bundle from one origin and letting it dynamically import the JSEP
// glue from another origin is fragile (and one failed init poisons later WASM
// fallback attempts in the same ORT instance).
await rm(ortDst, { recursive: true, force: true });
await mkdir(ortDst, { recursive: true });
const ortFiles = [
  'ort.all.min.js',
  'ort-wasm-simd-threaded.jsep.mjs',
  'ort-wasm-simd-threaded.jsep.wasm',
  'ort-wasm-simd-threaded.mjs',
  'ort-wasm-simd-threaded.wasm',
];
for (const name of ortFiles) {
  await copyFile(join(ortSrc, name), join(ortDst, name));
}

console.log('Prepared Basic Pitch, TensorFlow.js WASM, and same-origin ONNX Runtime Web assets.');
