import { test, expect } from '@playwright/test';
import { mkdtemp, mkdir, readFile, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';

const guard = join(process.cwd(), 'scripts', 'require-muscriptor-model.mjs');
const workersPrep = join(process.cwd(), 'scripts', 'prepare-muscriptor-workers-build.mjs');

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

test('ordinary builds do not invoke the heavy Workers MuScriptor exporter', () => {
  const result = spawnSync(process.execPath, [workersPrep], {
    cwd: process.cwd(),
    encoding: 'utf8',
    env: { ...process.env, WORKERS_CI: '0' },
  });
  expect(result.status, result.stderr).toBe(0);
  expect(result.stdout).toContain('skipped');
});

test('Workers Builds preparation contains the complete export and staging path', async () => {
  const source = await readFile(workersPrep, 'utf8');
  const pythonVersion = await readFile(join(process.cwd(), '.python-version'), 'utf8');
  expect(pythonVersion.trim()).toBe('3.12');
  expect(source).toContain("process.env.WORKERS_CI !== '1'");
  expect(source).toContain('tools/fetch_muscriptor_weights.py');
  expect(source).toContain('tools/export_muscriptor_browser_v3.py');
  expect(source).toContain("'--source', source");
  expect(source).toContain('tools/stage_muscriptor_cloudflare.py');
  expect(source).toContain("'--root=public'");
  expect(source).toContain('e34b397bf0584e67bfd81dc591c390e6dcb03350');
});

test('MuScriptor small checkpoint is made self-describing before load_model', async () => {
  const fetchSource = await readFile(join(process.cwd(), 'tools', 'fetch_muscriptor_weights.py'), 'utf8');
  const exporterSource = await readFile(join(process.cwd(), 'tools', 'export_muscriptor_browser_v3.py'), 'utf8');

  for (const source of [fetchSource, exporterSource]) {
    expect(source).toContain('768');
    expect(source).toContain('12');
    expect(source).toContain('14');
    expect(source).toContain('1393');
    expect(source).toContain('config.json');
  }

  expect(exporterSource).toContain('ensure_small_checkpoint_config');
  expect(exporterSource).toContain('safe_open');
  expect(exporterSource).toContain('Refusing to export with the wrong architecture');
});
