import { access, rm } from 'node:fs/promises';
import { constants } from 'node:fs';
import { resolve } from 'node:path';
import { spawnSync } from 'node:child_process';

const root = resolve(new URL('..', import.meta.url).pathname);
const modelManifest = resolve(root, 'public/models/muscriptor-small/manifest.json');
const sourceDir = resolve(root, '.muscriptor-source');
const browserDir = resolve(root, '.muscriptor-browser');
const stagedDir = resolve(root, 'public/models/muscriptor-small');

if (process.env.WORKERS_CI !== '1') {
  console.log('MuScriptor Workers Builds preparation: skipped (WORKERS_CI != 1).');
  process.exit(0);
}

if (await readable(modelManifest)) {
  console.log('MuScriptor Workers Builds preparation: staged model already present; validating.');
  run(process.execPath, ['scripts/require-muscriptor-model.mjs', '--root=public']);
  process.exit(0);
}

console.log('MuScriptor Workers Builds preparation: exporting non-commercial muscriptor-small for browser WebGPU.');
console.log('This runs only inside Cloudflare Workers Builds and keeps ordinary local/CI builds lightweight.');

run('python', ['--version']);
run('python', [
  '-m', 'pip', 'install', '--disable-pip-version-check',
  '--index-url', 'https://download.pytorch.org/whl/cpu',
  'torch',
]);
run('python', [
  '-m', 'pip', 'install', '--disable-pip-version-check',
  'git+https://github.com/muscriptor/muscriptor.git',
  'huggingface_hub', 'onnx', 'onnxruntime', 'onnxconverter-common', 'onnx-ir',
]);

await rm(sourceDir, { recursive: true, force: true });
await rm(browserDir, { recursive: true, force: true });
await rm(stagedDir, { recursive: true, force: true });

run('python', ['tools/fetch_muscriptor_weights.py']);
run('python', [
  'tools/export_muscriptor_browser_v3.py',
  '--weights', '.muscriptor-source/model.safetensors',
  '--output', '.muscriptor-browser',
  '--source-file', '.muscriptor-source/SOURCE.txt',
]);
run('python', [
  'tools/stage_muscriptor_cloudflare.py',
  '.muscriptor-browser', 'public/models/muscriptor-small', '--chunk-mib', '4.5',
]);
run(process.execPath, ['scripts/require-muscriptor-model.mjs', '--root=public']);
console.log('MuScriptor Workers Builds preparation: ready.');

async function readable(path) {
  try {
    await access(path, constants.R_OK);
    return true;
  } catch {
    return false;
  }
}

function run(command, args) {
  const result = spawnSync(command, args, {
    cwd: root,
    env: {
      ...process.env,
      PYTHONUNBUFFERED: '1',
      PIP_DISABLE_PIP_VERSION_CHECK: '1',
    },
    stdio: 'inherit',
  });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(`${command} ${args.join(' ')} failed with exit code ${result.status}`);
  }
}
