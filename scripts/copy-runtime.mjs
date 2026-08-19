import { cp, mkdir, readdir, copyFile, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const publicDir = join(root, 'public');
const modelSrc = join(root, 'node_modules', '@spotify', 'basic-pitch', 'model');
const modelDst = join(publicDir, 'model', 'basic-pitch');
const tfWasmSrc = join(root, 'node_modules', '@tensorflow', 'tfjs-backend-wasm', 'dist');
const tfWasmDst = join(publicDir, 'tfjs-wasm');
const ortWasmSrc = join(root, 'node_modules', 'onnxruntime-web', 'dist');
const ortWasmDst = join(publicDir, 'ort-wasm');

await mkdir(publicDir, { recursive: true });
await rm(modelDst, { recursive: true, force: true });
await mkdir(modelDst, { recursive: true });
await cp(modelSrc, modelDst, { recursive: true });

await copyWasmFiles(tfWasmSrc, tfWasmDst);
await copyWasmFiles(ortWasmSrc, ortWasmDst);

console.log('Prepared Basic Pitch model, TensorFlow.js WASM and ONNX Runtime Web WASM assets.');

async function copyWasmFiles(source, destination) {
  await rm(destination, { recursive: true, force: true });
  await mkdir(destination, { recursive: true });
  for (const name of await readdir(source)) {
    if (name.endsWith('.wasm')) {
      await copyFile(join(source, name), join(destination, name));
    }
  }
}
