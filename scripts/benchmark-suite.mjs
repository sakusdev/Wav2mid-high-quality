import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { BrowserBenchRunner, runCommand } from './browser-bench-runner.mjs';
import { aggregateScores, readMidiNotes, readPrediction, scoreTranscription } from './benchmark-metrics.mjs';

const args = parseArgs(process.argv.slice(2));
if (!args.manifest || !args.adapters) {
  console.error('Usage: npm run benchmark:suite -- --manifest bench.json --adapters adapters.json [--adapter wav2mid-insane,muscriptor-small] [--split test] [--max-items 10]');
  process.exit(2);
}

const manifestPath = path.resolve(args.manifest);
const adaptersPath = path.resolve(args.adapters);
const manifest = JSON.parse(await fs.readFile(manifestPath, 'utf8'));
const adapterFile = JSON.parse(await fs.readFile(adaptersPath, 'utf8'));
const manifestRoot = path.resolve(path.dirname(manifestPath), manifest.root ?? '.');
const selectedNames = args.adapter ? new Set(String(args.adapter).split(',').map(x => x.trim()).filter(Boolean)) : null;
const allAdapters = adapterFile.adapters ?? adapterFile;
const adapters = Object.entries(allAdapters).filter(([name]) => !selectedNames || selectedNames.has(name));
if (!adapters.length) throw new Error('No benchmark adapters selected.');

let items = manifest.items ?? manifest.entries ?? [];
if (args.split) items = items.filter(item => String(item.split ?? '') === String(args.split));
if (args['max-items']) items = items.slice(0, Number(args['max-items']));
if (!items.length) throw new Error('Manifest has no matching items.');

const cacheRoot = path.resolve(args.cache ?? '.benchmark-cache');
const reportRoot = path.resolve(args.output ?? 'benchmark-results');
await fs.mkdir(cacheRoot, { recursive: true });
await fs.mkdir(reportRoot, { recursive: true });

const needsBrowser = adapters.some(([, adapter]) => adapter.type === 'browser');
const browser = needsBrowser ? new BrowserBenchRunner({ port: args.port }) : null;
if (browser) await browser.start({ build: !args['no-build'] });

const report = {
  benchmark: manifest.name ?? path.basename(manifestPath),
  generatedAt: new Date().toISOString(),
  manifest: manifestPath,
  itemCount: items.length,
  adapters: {},
};

try {
  for (const [adapterName, adapter] of adapters) {
    console.log(`\n=== ${adapterName} ===`);
    const itemReports = [];
    for (let index = 0; index < items.length; index += 1) {
      const item = items[index];
      const id = sanitize(item.id ?? `item-${index + 1}`);
      const audioPath = resolveFromRoot(manifestRoot, item.audio);
      const referencePath = resolveFromRoot(manifestRoot, item.reference ?? item.midi);
      if (!audioPath || !referencePath) throw new Error(`Manifest item ${id} needs audio and reference/midi paths.`);
      const adapterDir = path.join(cacheRoot, sanitize(adapterName));
      await fs.mkdir(adapterDir, { recursive: true });

      const started = performance.now();
      const predictionPath = await runAdapter({ adapterName, adapter, browser, audioPath, item, adapterDir, id });
      const reference = await readMidiNotes(referencePath);
      const prediction = await readPrediction(predictionPath);
      const metrics = scoreTranscription(reference, prediction, manifest.metrics ?? {});
      const elapsedSeconds = (performance.now() - started) / 1000;
      itemReports.push({
        id: item.id ?? id,
        split: item.split ?? null,
        tags: item.tags ?? [],
        instruments: item.instruments ?? [],
        audio: item.audio,
        reference: item.reference ?? item.midi,
        prediction: path.relative(process.cwd(), predictionPath),
        elapsedSeconds,
        metrics,
      });
      console.log(`${index + 1}/${items.length} ${id}: onset ${(metrics.onset.f1 * 100).toFixed(1)} · offset ${(metrics.offset.f1 * 100).toFixed(1)} · frame ${(metrics.frame.f1 * 100).toFixed(1)} · objective ${(metrics.objective * 100).toFixed(1)}`);
    }
    report.adapters[adapterName] = {
      config: adapter,
      aggregate: aggregateScores(itemReports),
      items: itemReports,
    };
  }
} finally {
  await browser?.stop();
}

const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
const jsonPath = path.join(reportRoot, `benchmark-${timestamp}.json`);
const mdPath = path.join(reportRoot, `benchmark-${timestamp}.md`);
await fs.writeFile(jsonPath, JSON.stringify(report, null, 2));
await fs.writeFile(mdPath, renderMarkdown(report));
console.log(`\nSaved ${jsonPath}`);
console.log(`Saved ${mdPath}`);

async function runAdapter({ adapterName, adapter, browser, audioPath, item, adapterDir, id }) {
  if (adapter.type === 'browser') {
    const output = path.join(adapterDir, `${id}.json`);
    const payload = await browser.run(audioPath, {
      mode: adapter.mode ?? 'insane',
      backend: adapter.backend ?? 'wasm',
      sensitivity: adapter.sensitivity ?? 1,
      minPitch: adapter.minPitch,
      maxPitch: adapter.maxPitch,
      neural: Boolean(adapter.neural),
      timeoutMs: adapter.timeoutMs,
    });
    await fs.writeFile(output, JSON.stringify(payload, null, 2));
    return output;
  }

  if (adapter.type === 'precomputed') {
    const template = adapter.path ?? adapter.output;
    if (!template) throw new Error(`${adapterName}: precomputed adapter needs path.`);
    return path.resolve(substitute(template, { audio: audioPath, id, instruments: instrumentString(item) }));
  }

  if (adapter.type === 'command') {
    if (!Array.isArray(adapter.command) || !adapter.command.length) throw new Error(`${adapterName}: command must be an argv array.`);
    const extension = adapter.format === 'json' ? '.json' : '.mid';
    const output = path.join(adapterDir, `${id}${extension}`);
    if (!adapter.alwaysRun && await exists(output)) return output;
    const values = {
      audio: audioPath,
      output,
      id,
      instruments: instrumentString(item),
    };
    const [command, ...commandArgs] = adapter.command.map(token => substitute(String(token), values));
    const env = { ...process.env };
    for (const [key, value] of Object.entries(adapter.env ?? {})) env[key] = substitute(String(value), values);
    console.log(`  run ${adapterName}: ${command} ${commandArgs.join(' ')}`);
    await runCommand(command, commandArgs, { env, cwd: adapter.cwd ? path.resolve(adapter.cwd) : process.cwd() });
    if (!await exists(output)) throw new Error(`${adapterName} completed but did not create ${output}`);
    return output;
  }

  throw new Error(`Unsupported adapter type: ${adapter.type}`);
}

function instrumentString(item) {
  return Array.isArray(item.instruments) ? item.instruments.join(',') : String(item.instruments ?? '');
}

function resolveFromRoot(root, value) {
  if (!value) return null;
  return path.isAbsolute(value) ? value : path.resolve(root, value);
}

function substitute(template, values) {
  return template.replace(/\{(audio|output|id|instruments)\}/g, (_, key) => String(values[key] ?? ''));
}

function sanitize(value) {
  return String(value).replace(/[^a-zA-Z0-9._-]+/g, '_').slice(0, 160);
}

async function exists(file) {
  try { await fs.access(file); return true; } catch { return false; }
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

function renderMarkdown(data) {
  const rows = Object.entries(data.adapters)
    .map(([name, value]) => ({ name, ...value.aggregate }))
    .sort((a, b) => b.objective - a.objective);
  const lines = [
    `# ${data.benchmark} leaderboard`,
    '',
    `Generated: ${data.generatedAt}`,
    '',
    '| Rank | Adapter | Onset F1 | Offset F1 | Frame F1 | Drums F1 | Objective |',
    '| ---: | --- | ---: | ---: | ---: | ---: | ---: |',
  ];
  rows.forEach((row, index) => {
    lines.push(`| ${index + 1} | ${row.name} | ${pct(row.onsetF1)} | ${pct(row.offsetF1)} | ${pct(row.frameF1)} | ${pct(row.drumsF1)} | **${pct(row.objective)}** |`);
  });
  lines.push('', '> Metrics are computed by the same evaluator for every adapter. Dataset/model licenses are not changed by this harness.');
  return `${lines.join('\n')}\n`;
}

function pct(value) { return `${(Number(value ?? 0) * 100).toFixed(2)}%`; }
