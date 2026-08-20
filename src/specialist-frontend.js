import FFT from 'fft.js';

export const SPECIALIST_FRONTEND = Object.freeze({
  sampleRate: 16000,
  nFft: 2048,
  hopLength: 160,
  winLength: 2048,
  melBins: 229,
  fmin: 30,
  fmax: 8000,
  amin: 1e-10,
});

const WINDOW = makePeriodicHann(SPECIALIST_FRONTEND.winLength);
const MEL_BANK = makeSlaneyMelBank(SPECIALIST_FRONTEND);

export function normalizeMono16k(audioLike) {
  const sourceRate = Number(audioLike?.sampleRate);
  const channels = Number(audioLike?.numberOfChannels ?? 1);
  const sourceLength = Number(audioLike?.length ?? audioLike?.samples?.length ?? 0);
  if (!sourceRate || !sourceLength || channels < 1) throw new Error('Specialist frontend received empty audio.');

  let mono;
  if (audioLike.samples instanceof Float32Array) {
    mono = new Float32Array(audioLike.samples);
  } else if (typeof audioLike.getChannelData === 'function') {
    mono = new Float32Array(sourceLength);
    for (let channel = 0; channel < channels; channel += 1) {
      const data = audioLike.getChannelData(channel);
      for (let i = 0; i < sourceLength; i += 1) mono[i] += data[i] / channels;
    }
  } else if (audioLike.left instanceof Float32Array) {
    const left = audioLike.left;
    const right = audioLike.right instanceof Float32Array ? audioLike.right : left;
    const length = Math.min(left.length, right.length);
    mono = new Float32Array(length);
    for (let i = 0; i < length; i += 1) mono[i] = (left[i] + right[i]) * 0.5;
  } else {
    throw new Error('Unsupported specialist audio container.');
  }

  if (sourceRate === SPECIALIST_FRONTEND.sampleRate) return mono;
  const targetLength = Math.max(1, Math.round(mono.length * SPECIALIST_FRONTEND.sampleRate / sourceRate));
  return linearResample(mono, targetLength);
}

export function makeSpecialistSegments(samples, segmentSeconds = 10) {
  if (!(samples instanceof Float32Array) || !samples.length) throw new Error('Specialist segmentation received empty samples.');
  const segmentSamples = Math.round(SPECIALIST_FRONTEND.sampleRate * segmentSeconds);
  const paddedLength = Math.ceil(samples.length / segmentSamples) * segmentSamples;
  const padded = new Float32Array(paddedLength);
  padded.set(samples);
  const hop = Math.floor(segmentSamples / 2);
  const segments = [];
  for (let start = 0; start + segmentSamples <= padded.length; start += hop) {
    segments.push(padded.slice(start, start + segmentSamples));
  }
  return { segments, originalSamples: samples.length, segmentSamples };
}

export function logMelFromWaveform(samples) {
  if (!(samples instanceof Float32Array) || !samples.length) throw new Error('Log-mel frontend received empty samples.');
  const { nFft, hopLength, melBins, amin } = SPECIALIST_FRONTEND;
  const pad = nFft >> 1;
  const frames = Math.floor(samples.length / hopLength) + 1;
  const fft = new FFT(nFft);
  const spectrum = fft.createComplexArray();
  const frame = new Array(nFft).fill(0);
  const out = new Float32Array(frames * melBins);

  for (let t = 0; t < frames; t += 1) {
    const centerStart = t * hopLength - pad;
    for (let i = 0; i < nFft; i += 1) {
      frame[i] = samples[reflectIndex(centerStart + i, samples.length)] * WINDOW[i];
    }
    fft.realTransform(spectrum, frame);
    const power = new Float32Array(nFft / 2 + 1);
    for (let k = 0; k < power.length; k += 1) {
      const re = spectrum[k * 2];
      const im = spectrum[k * 2 + 1];
      power[k] = re * re + im * im;
    }
    const base = t * melBins;
    for (let mel = 0; mel < melBins; mel += 1) {
      const weights = MEL_BANK[mel];
      let energy = 0;
      for (let k = 0; k < power.length; k += 1) energy += power[k] * weights[k];
      out[base + mel] = 10 * Math.log10(Math.max(amin, energy));
    }
  }

  return { data: out, dims: [1, 1, frames, melBins], frames };
}

export function deframeSpecialistOutputs(segmentOutputs, originalSamples) {
  if (!segmentOutputs.length) return {};
  const keys = Object.keys(segmentOutputs[0]);
  const finalFrames = Math.floor(originalSamples / SPECIALIST_FRONTEND.hopLength) + 1;
  const result = {};

  for (const key of keys) {
    const arrays = segmentOutputs.map(item => item[key]);
    const classes = arrays[0].dims.at(-1);
    if (arrays.length === 1) {
      const frames = Math.min(finalFrames, arrays[0].dims.at(-2));
      result[key] = {
        data: arrays[0].data.slice(0, frames * classes),
        dims: [frames, classes],
      };
      continue;
    }

    const segmentFramesWithCenter = arrays[0].dims.at(-2);
    const usableSegmentFrames = segmentFramesWithCenter - 1;
    if (usableSegmentFrames % 4 !== 0) throw new Error(`Unexpected specialist segment frame count: ${usableSegmentFrames}`);
    const quarter = usableSegmentFrames / 4;
    const chunks = [];
    chunks.push(sliceFrames(arrays[0], 0, quarter * 3, classes));
    for (let i = 1; i < arrays.length - 1; i += 1) {
      chunks.push(sliceFrames(arrays[i], quarter, quarter * 3, classes));
    }
    chunks.push(sliceFrames(arrays.at(-1), quarter, usableSegmentFrames, classes));

    const totalFrames = Math.min(finalFrames, chunks.reduce((sum, chunk) => sum + chunk.frames, 0));
    const merged = new Float32Array(totalFrames * classes);
    let frameOffset = 0;
    for (const chunk of chunks) {
      const framesToCopy = Math.min(chunk.frames, totalFrames - frameOffset);
      if (framesToCopy <= 0) break;
      merged.set(chunk.data.subarray(0, framesToCopy * classes), frameOffset * classes);
      frameOffset += framesToCopy;
    }
    result[key] = { data: merged, dims: [totalFrames, classes] };
  }

  return result;
}

function sliceFrames(tensor, start, end, classes) {
  const frames = Math.max(0, end - start);
  return {
    frames,
    data: tensor.data.slice(start * classes, end * classes),
  };
}

function reflectIndex(index, length) {
  if (length <= 1) return 0;
  let i = index;
  while (i < 0 || i >= length) {
    if (i < 0) i = -i;
    if (i >= length) i = 2 * length - 2 - i;
  }
  return i;
}

function makePeriodicHann(length) {
  const out = new Float32Array(length);
  for (let i = 0; i < length; i += 1) out[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / length);
  return out;
}

function linearResample(source, targetLength) {
  const output = new Float32Array(targetLength);
  if (source.length === 1) { output.fill(source[0]); return output; }
  const ratio = (source.length - 1) / Math.max(1, targetLength - 1);
  for (let i = 0; i < targetLength; i += 1) {
    const position = i * ratio;
    const left = Math.floor(position);
    const right = Math.min(source.length - 1, left + 1);
    const frac = position - left;
    output[i] = source[left] + (source[right] - source[left]) * frac;
  }
  return output;
}

function makeSlaneyMelBank({ sampleRate, nFft, melBins, fmin, fmax }) {
  const fftBins = nFft / 2 + 1;
  const fftFreqs = new Float64Array(fftBins);
  for (let i = 0; i < fftBins; i += 1) fftFreqs[i] = i * sampleRate / nFft;

  const melMin = hzToSlaneyMel(fmin);
  const melMax = hzToSlaneyMel(fmax);
  const melPoints = new Float64Array(melBins + 2);
  const hzPoints = new Float64Array(melBins + 2);
  for (let i = 0; i < melPoints.length; i += 1) {
    melPoints[i] = melMin + (melMax - melMin) * i / (melPoints.length - 1);
    hzPoints[i] = slaneyMelToHz(melPoints[i]);
  }

  const bank = [];
  for (let m = 0; m < melBins; m += 1) {
    const lower = hzPoints[m];
    const center = hzPoints[m + 1];
    const upper = hzPoints[m + 2];
    const weights = new Float32Array(fftBins);
    const norm = 2 / Math.max(1e-12, upper - lower);
    for (let k = 0; k < fftBins; k += 1) {
      const frequency = fftFreqs[k];
      const lowerSlope = (frequency - lower) / Math.max(1e-12, center - lower);
      const upperSlope = (upper - frequency) / Math.max(1e-12, upper - center);
      weights[k] = Math.max(0, Math.min(lowerSlope, upperSlope)) * norm;
    }
    bank.push(weights);
  }
  return bank;
}

function hzToSlaneyMel(frequency) {
  const fSp = 200 / 3;
  const minLogHz = 1000;
  const minLogMel = minLogHz / fSp;
  const logStep = Math.log(6.4) / 27;
  return frequency >= minLogHz
    ? minLogMel + Math.log(frequency / minLogHz) / logStep
    : frequency / fSp;
}

function slaneyMelToHz(mel) {
  const fSp = 200 / 3;
  const minLogHz = 1000;
  const minLogMel = minLogHz / fSp;
  const logStep = Math.log(6.4) / 27;
  return mel >= minLogMel
    ? minLogHz * Math.exp(logStep * (mel - minLogMel))
    : fSp * mel;
}
