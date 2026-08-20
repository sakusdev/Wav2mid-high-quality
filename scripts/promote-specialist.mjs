import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';

const args = parseArgs(process.argv.slice(2));
for (const required of ['benchmark', 'candidate', 'baseline', 'metadata']) {
  if (!args[required]) {
    console.error('Usage: node scripts/promote-specialist.mjs --benchmark result.json --candidate MODEL --baseline wav2mid-insane --metadata model.json [--instrument bass] [--stem bass] [--min-delta 0.01] [--manifest public/specialists/manifest.json]');
    process.exit(2);
  }
}

const reportPath = path.resolve(args.benchmark);
const metadataPath = path.resolve(args.metadata);
const manifestPath = path.resolve(args.manifest ?? 'public/specialists/manifest.json');
const report = JSON.parse(await fs.readFile(reportPath, 'utf8'));
const metadata = JSON.parse(await fs.readFile(metadataPath, 'utf8'));
const manifest = JSON.parse(await fs.readFile(manifestPath, 'utf8'));
const candidate = report.adapters?.[args.candidate];
const baseline = report.adapters?.[args.baseline];
if (!candidate || !baseline) throw new Error('Candidate or baseline adapter is missing from benchmark report.');

const instrument = args.instrument ? String(args.instrument).toLowerCase() : null;
const candidateItems = filterItems(candidate.items ?? [], instrument);
const baselineById = new Map(filterItems(baseline.items ?? [], instrument).map(item => [String(item.id), item]));
const paired = candidateItems
  .map(item => [item, baselineById.get(String(item.id))])
  .filter(([, base]) => Boolean(base));
if (!paired.length) throw new Error(`No paired benchmark items${instrument ? ` for instrument ${instrument}` : ''}.`);

const candidateObjective = mean(paired.map(([item]) => Number(item.metrics?.objective ?? 0)));
const baselineObjective = mean(paired.map(([, item]) => Number(item.metrics?.objective ?? 0)));
const delta = candidateObjective - baselineObjective;
const minimum = Number(args['min-delta'] ?? 0.01);

console.log(`candidate ${args.candidate}: ${(candidateObjective * 100).toFixed(2)}%`);
console.log(`baseline  ${args.baseline}: ${(baselineObjective * 100).toFixed(2)}%`);
console.log(`delta: ${(delta * 100).toFixed(2)} pp over ${paired.length} paired item(s)`);
if (!(delta >= minimum)) {
  throw new Error(`Promotion denied: ${delta.toFixed(6)} < required ${minimum.toFixed(6)}.`);
}

if (metadata.schema !== 'wav2mid-specialist/v1') throw new Error('Unsupported specialist metadata schema.');
const modelName = String(args.name ?? metadata.name ?? args.candidate);
const entry = {
  ...metadata,
  name: modelName,
  stem: args.stem ?? metadata.stem ?? defaultStem(metadata.instrument),
  promoted: true,
  enabled: true,
  promotion: {
    benchmark: report.benchmark ?? path.basename(reportPath),
    generatedAt: report.generatedAt ?? null,
    candidateAdapter: args.candidate,
    baselineAdapter: args.baseline,
    instrument,
    pairedItems: paired.length,
    candidateObjective,
    baselineObjective,
    delta,
    minimumDelta: minimum,
  },
};

manifest.version = Math.max(1, Number(manifest.version ?? 1));
manifest.models ??= {};
manifest.models[modelName] = entry;
await fs.writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
console.log(`PROMOTED ${modelName} -> ${manifestPath}`);

function filterItems(items, instrument) {
  if (!instrument) return items;
  return items.filter(item => {
    const instruments = (item.instruments ?? []).map(value => String(value).toLowerCase());
    const tags = (item.tags ?? []).map(value => String(value).toLowerCase());
    return instruments.includes(instrument) || tags.includes(instrument) || tags.includes(`instrument:${instrument}`);
  });
}

function defaultStem(instrument) {
  if (instrument === 'bass') return 'bass';
  if (instrument === 'vocal' || instrument === 'vocals') return 'vocals';
  if (instrument === 'drums') return 'drums';
  return 'other';
}

function mean(values) { return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length); }

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
