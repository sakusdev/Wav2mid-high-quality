import { access, readFile } from 'node:fs/promises';
import { constants } from 'node:fs';
import { join, resolve } from 'node:path';

const rootArg = process.argv.find(arg => arg.startsWith('--root='));
const root = resolve(rootArg ? rootArg.slice('--root='.length) : 'public');
const modelDir = join(root, 'models', 'muscriptor-small');
const manifestPath = join(modelDir, 'manifest.json');
const streamMapPath = join(modelDir, 'stream-map.json');

function fail(message) {
  console.error(`MuScriptor deploy guard: ${message}`);
  process.exit(1);
}

async function exists(path) {
  try {
    await access(path, constants.R_OK);
    return true;
  } catch {
    return false;
  }
}

if (!(await exists(manifestPath))) {
  fail(`missing ${manifestPath}. Refusing to deploy an app that would return the SPA HTML fallback for MuScriptor.`);
}

let manifest;
try {
  manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
} catch (error) {
  fail(`invalid manifest JSON at ${manifestPath}: ${error.message}`);
}

if (manifest?.format !== 'wav2mid-muscriptor-browser/v1') {
  fail(`unexpected manifest format: ${manifest?.format ?? 'missing'}`);
}

let streamMap = { models: {} };
if (await exists(streamMapPath)) {
  try {
    streamMap = JSON.parse(await readFile(streamMapPath, 'utf8'));
  } catch (error) {
    fail(`invalid stream-map JSON: ${error.message}`);
  }
}

for (const name of ['conditioner', 'decoder']) {
  const entry = manifest?.files?.[name];
  if (!entry?.url) fail(`manifest is missing files.${name}.url`);

  const logicalPath = join(root, entry.url.replace(/^\//, ''));
  if (await exists(logicalPath)) continue;

  const partManifestUrl = streamMap?.models?.[entry.url];
  if (!partManifestUrl) {
    fail(`${name} model is neither present as ${logicalPath} nor mapped in stream-map.json`);
  }

  const partManifestPath = join(root, String(partManifestUrl).replace(/^\//, ''));
  if (!(await exists(partManifestPath))) fail(`missing streamed ${name} part manifest: ${partManifestPath}`);

  let partManifest;
  try {
    partManifest = JSON.parse(await readFile(partManifestPath, 'utf8'));
  } catch (error) {
    fail(`invalid ${name} part manifest JSON: ${error.message}`);
  }
  if (!Array.isArray(partManifest?.parts) || !partManifest.parts.length) {
    fail(`${name} part manifest contains no parts`);
  }
  for (const part of partManifest.parts) {
    const partPath = join(root, String(part.url || '').replace(/^\//, ''));
    if (!(await exists(partPath))) fail(`missing streamed ${name} part: ${partPath}`);
  }
}

console.log(`MuScriptor deploy guard: OK (${root})`);
