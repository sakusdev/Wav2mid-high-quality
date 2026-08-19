import { cp, mkdir, readdir, copyFile, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const publicDir = join(root, 'public');
const modelSrc = join(root, 'node_modules', '@spotify', 'basic-pitch', 'model');
const modelDst = join(publicDir, 'model', 'basic-pitch');
const wasmSrc = join(root, 'node_modules', '@tensorflow', 'tfjs-backend-wasm', 'dist');
const wasmDst = join(publicDir, 'tfjs-wasm');

await mkdir(publicDir, { recursive: true });
await rm(modelDst, { recursive: true, force: true });
await mkdir(modelDst, { recursive: true });
await cp(modelSrc, modelDst, { recursive: true });

await rm(wasmDst, { recursive: true, force: true });
await mkdir(wasmDst, { recursive: true });
for (const name of await readdir(wasmSrc)) {
  if (name.endsWith('.wasm')) {
    await copyFile(join(wasmSrc, name), join(wasmDst, name));
  }
}

console.log('Prepared Basic Pitch model and TensorFlow.js WASM runtime.');
