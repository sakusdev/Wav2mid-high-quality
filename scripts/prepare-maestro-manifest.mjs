import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';

const args = parseArgs(process.argv.slice(2));
if (!args.root) {
  console.error('Usage: npm run benchmark:prepare-maestro -- --root /path/to/maestro-v3.0.0 [--csv maestro-v3.0.0.csv] [--output bench/maestro.json] [--split test] [--limit 50]');
  process.exit(2);
}

const root = path.resolve(args.root);
const csvPath = path.resolve(root, args.csv ?? 'maestro-v3.0.0.csv');
const output = path.resolve(args.output ?? 'bench/maestro-v3.manifest.json');
const splitFilter = args.split ? String(args.split) : null;
const limit = args.limit ? Number(args.limit) : Infinity;
const rows = parseCsv(await fs.readFile(csvPath, 'utf8'));
const items = [];

for (const row of rows) {
  if (splitFilter && row.split !== splitFilter) continue;
  const audio = row.audio_filename;
  const reference = row.midi_filename;
  if (!audio || !reference) continue;
  items.push({
    id: reference.replace(/\.midi?$/i, '').replace(/[^a-zA-Z0-9._-]+/g, '_'),
    split: row.split,
    audio,
    reference,
    instruments: ['acoustic_piano'],
    tags: ['maestro-v3', 'real-recording', 'piano'],
    metadata: {
      composer: row.canonical_composer,
      title: row.canonical_title,
      year: Number(row.year),
      duration: Number(row.duration),
    },
  });
  if (items.length >= limit) break;
}

const manifest = {
  name: 'MAESTRO v3 aligned real-piano benchmark',
  root,
  metrics: { onsetTolerance: 0.05, offsetTolerance: 0.05, offsetRatio: 0.2, drumTolerance: 0.05 },
  items,
};
await fs.mkdir(path.dirname(output), { recursive: true });
await fs.writeFile(output, JSON.stringify(manifest, null, 2));
console.log(`Wrote ${items.length} MAESTRO entries to ${output}`);

function parseCsv(text) {
  const records = [];
  let record = [];
  let field = '';
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    if (quoted) {
      if (char === '"' && text[i + 1] === '"') { field += '"'; i += 1; }
      else if (char === '"') quoted = false;
      else field += char;
    } else if (char === '"') quoted = true;
    else if (char === ',') { record.push(field); field = ''; }
    else if (char === '\n') { record.push(field.replace(/\r$/, '')); records.push(record); record = []; field = ''; }
    else field += char;
  }
  if (field.length || record.length) { record.push(field.replace(/\r$/, '')); records.push(record); }
  const header = records.shift() ?? [];
  return records.filter(row => row.some(Boolean)).map(row => Object.fromEntries(header.map((key, index) => [key, row[index] ?? ''])));
}

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (!token.startsWith('--')) continue;
    const key = token.slice(2);
    if (argv[i + 1] && !argv[i + 1].startsWith('--')) out[key] = argv[++i];
    else out[key] = true;
  }
  return out;
}
