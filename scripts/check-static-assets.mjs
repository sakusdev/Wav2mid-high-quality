import { readdir, stat } from 'node:fs/promises';
import { join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const dist = join(root, 'dist');
const maxBytes = 25 * 1024 * 1024;
let largest = { path: '', size: 0 };
const oversized = [];

await walk(dist);

console.log(`Largest static asset: ${largest.path} (${(largest.size / 1024 / 1024).toFixed(2)} MiB)`);
if (oversized.length) {
  for (const item of oversized) {
    console.error(`Cloudflare asset limit exceeded: ${item.path} (${(item.size / 1024 / 1024).toFixed(2)} MiB > 25 MiB)`);
  }
  process.exitCode = 1;
} else {
  console.log('Cloudflare Workers Static Assets 25 MiB/file gate: PASS');
}

async function walk(directory) {
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) {
      await walk(path);
      continue;
    }
    if (!entry.isFile()) continue;
    const info = await stat(path);
    const item = { path: relative(dist, path), size: info.size };
    if (item.size > largest.size) largest = item;
    if (item.size > maxBytes) oversized.push(item);
  }
}
