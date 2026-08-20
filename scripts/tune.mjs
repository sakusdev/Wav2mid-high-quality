import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { BrowserBenchRunner } from './browser-bench-runner.mjs';
import { aggregateScores, readMidiNotes, scoreTranscription } from './benchmark-metrics.mjs';

const args = parseArgs(process.argv.slice(2));
if (!args.manifest) {
  console.error('Usage: npm run tune -- --manifest bench.json [--split validation] [--max-items 8] [--backend wasm] [--modes pro,insane] [--sensitivities 0.8,0.9,1,1.1,1.2]');
  process.exit(2);
}

const manifestPath = path.resolve(args.manifest);
const manifest = JSON.parse(await fs.readFile(manifestPath, 'utf8'));
const root = path.resolve(path.dirname(manifestPath), manifest.root ?? '.');
const split = args.split ?? 'validation';
let items = (manifest.items ?? manifest.entries ?? []).filter(item => !split || String(item.split ?? '') === String(split));
if (!items.length && split === 'validation') items = manifest.items ?? manifest.entries ?? [];
if (args['max-items']) items = items.slice(0, Number(args['max-items']));
if (!items.length) throw new Error('No tuning items found.');

const modes = String(args.modes ?? 'pro,insane').split(',').map(x => x.trim()).filter(Boolean);
const sensitivities = String(args.sensitivities ?? '0.80,0.90,1.00,1.10,1.20').split(',').map(Number).filter(Number.isFinite);
const configs = [];
for (const mode of modes) for (const sensitivity of sensitivities) configs.push({ mode, sensitivity });

const runner = new BrowserBenchRunner({ port: args.port });
await runner.start({ build: !args['no-build'] });
const evaluations = [];

try {
  for (let configIndex = 0; configIndex < configs.length; configIndex += 1) {
    const config = configs[configIndex];
    const itemReports = [];
    console.log(`\n[${configIndex + 1}/${configs.length}] ${config.mode} sensitivity=${config.sensitivity.toFixed(2)}`);
    for (let itemIndex = 0; itemIndex < items.length; itemIndex += 1) {
      const item = items[itemIndex];
      const audio = resolve(root, item.audio);
      const referencePath = resolve(root, item.reference ?? item.midi);
      const payload = await runner.run(audio, {
        mode: config.mode,
        sensitivity: config.sensitivity,
        backend: args.backend ?? 'wasm',
        timeoutMs: args.timeout ? Number(args.timeout) : undefined,
      });
      const prediction = {
        tonal: (payload.notes ?? []).map(note => ({
          pitchMidi: Number(note.pitchMidi),
          startTimeSeconds: Number(note.startTimeSeconds),
          durationSeconds: Number(note.durationSeconds),
          velocity: Number(note.amplitude ?? 1),
        })),
        drums: (payload.drums ?? []).map(drum => ({
          pitchMidi: Number(drum.midi),
          startTimeSeconds: Number(drum.time),
          durationSeconds: Number(drum.duration ?? 0.05),
          velocity: Number(drum.velocity ?? 1),
        })),
      };
      const reference = await readMidiNotes(referencePath);
      const metrics = scoreTranscription(reference, prediction, manifest.metrics ?? {});
      itemReports.push({ id: item.id ?? itemIndex, metrics });
      console.log(`  ${itemIndex + 1}/${items.length} ${item.id ?? itemIndex}: ${(metrics.objective * 100).toFixed(2)}`);
    }
    const aggregate = aggregateScores(itemReports);
    evaluations.push({ config, aggregate });
    console.log(`  objective ${(aggregate.objective * 100).toFixed(2)} · onset ${(aggregate.onsetF1 * 100).toFixed(2)} · offset ${(aggregate.offsetF1 * 100).toFixed(2)} · frame ${(aggregate.frameF1 * 100).toFixed(2)}`);
  }
} finally {
  await runner.stop();
}

evaluations.sort((a, b) => b.aggregate.objective - a.aggregate.objective);
const winner = evaluations[0];
const output = {
  benchmark: manifest.name ?? path.basename(manifestPath),
  split,
  itemCount: items.length,
  generatedAt: new Date().toISOString(),
  backend: args.backend ?? 'wasm',
  winner,
  evaluations,
};
const outputPath = path.resolve(args.output ?? 'benchmark-results/tuned-profile.json');
await fs.mkdir(path.dirname(outputPath), { recursive: true });
await fs.writeFile(outputPath, JSON.stringify(output, null, 2));
console.log(`\nWINNER: ${winner.config.mode} sensitivity=${winner.config.sensitivity.toFixed(2)} objective=${(winner.aggregate.objective * 100).toFixed(2)}%`);
console.log(`Saved ${outputPath}`);

function resolve(rootDir, value) {
  if (!value) throw new Error('Manifest item is missing a path.');
  return path.isAbsolute(value) ? value : path.resolve(rootDir, value);
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
