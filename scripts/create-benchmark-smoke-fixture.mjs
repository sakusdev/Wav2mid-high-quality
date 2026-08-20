import fs from 'node:fs/promises';
import path from 'node:path';
import midiPackage from '@tonejs/midi';

const { Midi } = midiPackage;
const root = path.resolve('.benchmark-smoke');
await fs.rm(root, { recursive: true, force: true });
await fs.mkdir(root, { recursive: true });

const wavPath = path.join(root, 'smoke.wav');
const midiPath = path.join(root, 'smoke.mid');
const manifestPath = path.join(root, 'manifest.json');
const adaptersPath = path.join(root, 'adapters.json');

const notes = [
  { midi: 60, start: 0.10, end: 0.58 },
  { midi: 64, start: 0.66, end: 1.12 },
  { midi: 67, start: 1.20, end: 1.66 },
  { midi: 60, start: 1.72, end: 2.30 },
  { midi: 64, start: 1.72, end: 2.30 },
  { midi: 67, start: 1.72, end: 2.30 },
];
await fs.writeFile(wavPath, makeWav(notes));

const midi = new Midi();
midi.header.setTempo(120);
const track = midi.addTrack();
track.name = 'Ground Truth Piano';
track.instrument.number = 0;
for (const note of notes) {
  track.addNote({ midi: note.midi, time: note.start, duration: note.end - note.start, velocity: 0.9 });
}
await fs.writeFile(midiPath, Buffer.from(midi.toArray()));

await fs.writeFile(manifestPath, JSON.stringify({
  name: 'CI benchmark browser smoke',
  root,
  metrics: {
    onsetTolerance: 0.20,
    offsetTolerance: 0.16,
    offsetRatio: 0.35,
    drumTolerance: 0.08
  },
  items: [{
    id: 'smoke',
    split: 'test',
    audio: 'smoke.wav',
    reference: 'smoke.mid',
    instruments: ['acoustic_piano'],
    tags: ['ci', 'synthetic']
  }]
}, null, 2));

await fs.writeFile(adaptersPath, JSON.stringify({
  adapters: {
    'wav2mid-fast-smoke': {
      type: 'browser',
      mode: 'fast',
      backend: 'wasm',
      sensitivity: 1.0,
      timeoutMs: 180000
    }
  }
}, null, 2));

console.log(`Created benchmark smoke fixture under ${root}`);

function makeWav(events) {
  const sampleRate = 44100;
  const channels = 2;
  const duration = 2.45;
  const totalSamples = Math.ceil(sampleRate * duration);
  const bytesPerSample = 2;
  const dataBytes = totalSamples * channels * bytesPerSample;
  const buffer = Buffer.alloc(44 + dataBytes);
  let offset = 0;
  const write = text => { buffer.write(text, offset, 'ascii'); offset += text.length; };
  const u32 = value => { buffer.writeUInt32LE(value, offset); offset += 4; };
  const u16 = value => { buffer.writeUInt16LE(value, offset); offset += 2; };
  write('RIFF'); u32(36 + dataBytes); write('WAVE');
  write('fmt '); u32(16); u16(1); u16(channels); u32(sampleRate);
  u32(sampleRate * channels * bytesPerSample); u16(channels * bytesPerSample); u16(16);
  write('data'); u32(dataBytes);

  for (let i = 0; i < totalSamples; i += 1) {
    const t = i / sampleRate;
    let sample = 0;
    for (const event of events) {
      if (t < event.start || t >= event.end) continue;
      const local = t - event.start;
      const attack = Math.min(1, local / 0.012);
      const release = Math.min(1, (event.end - t) / 0.055);
      const decay = 0.78 + 0.22 * Math.exp(-1.1 * local);
      const env = attack * release * decay;
      const frequency = 440 * 2 ** ((event.midi - 69) / 12);
      sample += env * (
        Math.sin(2 * Math.PI * frequency * t) +
        0.20 * Math.sin(4 * Math.PI * frequency * t) +
        0.06 * Math.sin(6 * Math.PI * frequency * t)
      );
    }
    sample *= 0.24;
    sample = Math.max(-0.95, Math.min(0.95, sample));
    const left = sample;
    const right = sample * 0.96;
    const byteOffset = 44 + i * channels * bytesPerSample;
    buffer.writeInt16LE(Math.round(left * 32767), byteOffset);
    buffer.writeInt16LE(Math.round(right * 32767), byteOffset + 2);
  }
  return buffer;
}
