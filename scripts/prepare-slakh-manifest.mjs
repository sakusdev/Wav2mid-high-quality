import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import midiPackage from '@tonejs/midi';

const { Midi } = midiPackage;

const args = parseArgs(process.argv.slice(2));
if (!args.root) {
  console.error('Usage: npm run benchmark:prepare-slakh -- --root /path/to/Slakh2100 [--output bench/slakh.json] [--limit 50]');
  process.exit(2);
}

const root = path.resolve(args.root);
const output = path.resolve(args.output ?? 'bench/slakh2100.manifest.json');
const mergedRoot = path.resolve(args['reference-cache'] ?? '.benchmark-cache/slakh-reference');
const limit = args.limit ? Number(args.limit) : Infinity;
await fs.mkdir(mergedRoot, { recursive: true });

const splitCandidates = [
  ['train', path.join(root, 'train')],
  ['validation', path.join(root, 'validation')],
  ['test', path.join(root, 'test')],
];
const splitDirs = [];
for (const [split, dir] of splitCandidates) {
  if (await isDirectory(dir)) splitDirs.push([split, dir]);
}
if (!splitDirs.length) splitDirs.push(['unspecified', root]);

const items = [];
for (const [split, splitDir] of splitDirs) {
  const entries = (await fs.readdir(splitDir, { withFileTypes: true }))
    .filter(entry => entry.isDirectory() && /^Track\d+$/i.test(entry.name))
    .sort((a, b) => a.name.localeCompare(b.name));

  for (const entry of entries) {
    if (items.length >= limit) break;
    const trackDir = path.join(splitDir, entry.name);
    const audio = path.join(trackDir, 'mix.flac');
    const midiDir = path.join(trackDir, 'MIDI');
    if (!await exists(audio) || !await isDirectory(midiDir)) continue;
    const sourceMidis = (await fs.readdir(midiDir))
      .filter(name => /^S\d+\.mid$/i.test(name))
      .sort()
      .map(name => path.join(midiDir, name));
    if (!sourceMidis.length) continue;

    const reference = path.join(mergedRoot, `${entry.name}.mid`);
    await mergeMidiSources(sourceMidis, reference);
    items.push({
      id: entry.name,
      split,
      audio: path.relative(root, audio),
      reference,
      tags: ['slakh2100', 'multi-instrument', 'synthetic-render'],
    });
  }
}

const manifest = {
  name: 'Slakh2100 aligned multi-instrument benchmark',
  root,
  metrics: { onsetTolerance: 0.05, offsetTolerance: 0.05, offsetRatio: 0.2, drumTolerance: 0.05 },
  items,
  notes: 'Reference MIDI is merged from MIDI/Sxx.mid because those files exactly drove the rendered stems; all_src.mid can differ due to synthesis heuristics.',
};
await fs.mkdir(path.dirname(output), { recursive: true });
await fs.writeFile(output, JSON.stringify(manifest, null, 2));
console.log(`Wrote ${items.length} Slakh entries to ${output}`);

async function mergeMidiSources(files, output) {
  const merged = new Midi();
  for (const file of files) {
    const source = new Midi(await fs.readFile(file));
    for (const sourceTrack of source.tracks) {
      const track = merged.addTrack();
      track.name = sourceTrack.name || path.basename(file, '.mid');
      track.channel = sourceTrack.channel;
      if (sourceTrack.instrument?.number != null) track.instrument.number = sourceTrack.instrument.number;
      for (const note of sourceTrack.notes) {
        track.addNote({ midi: note.midi, time: note.time, duration: note.duration, velocity: note.velocity });
      }
      for (const bend of sourceTrack.pitchBends ?? []) track.addPitchBend({ time: bend.time, value: bend.value });
    }
  }
  await fs.writeFile(output, Buffer.from(merged.toArray()));
}

async function exists(file) { try { await fs.access(file); return true; } catch { return false; } }
async function isDirectory(dir) { try { return (await fs.stat(dir)).isDirectory(); } catch { return false; } }

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
