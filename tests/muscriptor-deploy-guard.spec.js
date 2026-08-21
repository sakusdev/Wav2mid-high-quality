import { test, expect } from '@playwright/test';
import { mkdtemp, mkdir, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';

const guard = join(process.cwd(), 'scripts', 'require-muscriptor-model.mjs');

function runGuard(root) {
  return spawnSync(process.execPath, [guard, `--root=${root}`], {
    cwd: process.cwd(),
    encoding: 'utf8',
  });
}

test('MuScriptor deploy guard rejects a model-less deployment', async () => {
  const root = await mkdtemp(join(tmpdir(), 'wav2mid-no-model-'));
  const result = runGuard(root);
  expect(result.status).not.toBe(0);
  expect(result.stderr).toContain('missing');
  expect(result.stderr).toContain('Refusing to deploy');
});

test('MuScriptor deploy guard accepts staged streamed model parts', async () => {
  const root = await mkdtemp(join(tmpdir(), 'wav2mid-model-'));
  const dir = join(root, 'models', 'muscriptor-small');
  await mkdir(dir, { recursive: true });

  const manifest = {
    format: 'wav2mid-muscriptor-browser/v1',
    files: {
      conditioner: { url: '/models/muscriptor-small/conditioner.onnx' },
      decoder: { url: '/models/muscriptor-small/decoder.onnx' },
    },
  };
  const streamMap = {
    format: 'wav2mid-static-model-parts/v1',
    models: {
      '/models/muscriptor-small/conditioner.onnx': '/models/muscriptor-small/conditioner.onnx.parts.json',
      '/models/muscriptor-small/decoder.onnx': '/models/muscriptor-small/decoder.onnx.parts.json',
    },
  };

  await writeFile(join(dir, 'manifest.json'), JSON.stringify(manifest));
  await writeFile(join(dir, 'stream-map.json'), JSON.stringify(streamMap));

  for (const name of ['conditioner', 'decoder']) {
    const partName = `${name}.onnx.part-000.bin`;
    await writeFile(join(dir, partName), `${name}-bytes`);
    await writeFile(join(dir, `${name}.onnx.parts.json`), JSON.stringify({
      format: 'wav2mid-streamed-asset/v1',
      bytes: 8,
      parts: [{ url: `/models/muscriptor-small/${partName}` }],
    }));
  }

  const result = runGuard(root);
  expect(result.status, result.stderr).toBe(0);
  expect(result.stdout).toContain('MuScriptor deploy guard: OK');
});
