import FFT from 'fft.js';

const FFT_SIZE = 2048;
const HOP_SIZE = 512;
const BINS = FFT_SIZE / 2 + 1;
const TEMPORAL_RADIUS = 3;
const FREQUENCY_RADIUS = 4;
const EPSILON = 1e-12;
const WINDOW = new Float32Array(FFT_SIZE);
for (let i = 0; i < FFT_SIZE; i += 1) {
  WINDOW[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (FFT_SIZE - 1));
}

export function createSourceStems(input, sampleRate = 22050, detail = 'pro') {
  if (!(input instanceof Float32Array) || input.length === 0) {
    throw new Error('Source separation received empty audio.');
  }

  const { harmonic, percussive, frameCount } = spectralHarmonicPercussive(input);
  const bass = lowPass(harmonic, sampleRate, detail === 'insane' ? 340 : 300);
  const presence = highPass(harmonic, sampleRate, 720);

  return {
    mix: input,
    harmonic,
    percussive,
    bass,
    presence,
    stats: {
      fftSize: FFT_SIZE,
      hopSize: HOP_SIZE,
      frames: frameCount,
      method: 'STFT harmonic/percussive soft-mask',
    },
  };
}

function spectralHarmonicPercussive(input) {
  const frameCount = Math.max(1, Math.ceil(Math.max(0, input.length - FFT_SIZE) / HOP_SIZE) + 1);
  const magnitudes = new Float32Array(frameCount * BINS);
  const harmonicScore = new Float32Array(frameCount * BINS);
  const percussiveScore = new Float32Array(frameCount * BINS);
  const fft = new FFT(FFT_SIZE);
  const frame = new Array(FFT_SIZE).fill(0);
  const spectrum = fft.createComplexArray();

  for (let frameIndex = 0; frameIndex < frameCount; frameIndex += 1) {
    loadWindowedFrame(input, frameIndex * HOP_SIZE, frame);
    fft.realTransform(spectrum, frame);
    const base = frameIndex * BINS;
    for (let k = 0; k < BINS; k += 1) {
      const re = spectrum[k * 2];
      const im = spectrum[k * 2 + 1];
      magnitudes[base + k] = Math.hypot(re, im);
    }
  }

  // Horizontal smoothing favours sustained/harmonic energy.
  for (let k = 0; k < BINS; k += 1) {
    let sum = 0;
    let count = 0;
    for (let t = 0; t < frameCount; t += 1) {
      if (t === 0) {
        const end = Math.min(frameCount - 1, TEMPORAL_RADIUS);
        for (let q = 0; q <= end; q += 1) { sum += magnitudes[q * BINS + k]; count += 1; }
      } else {
        const add = t + TEMPORAL_RADIUS;
        const remove = t - TEMPORAL_RADIUS - 1;
        if (add < frameCount) { sum += magnitudes[add * BINS + k]; count += 1; }
        if (remove >= 0) { sum -= magnitudes[remove * BINS + k]; count -= 1; }
      }
      harmonicScore[t * BINS + k] = sum / Math.max(1, count);
    }
  }

  // Vertical smoothing favours broadband/percussive energy.
  for (let t = 0; t < frameCount; t += 1) {
    const base = t * BINS;
    let sum = 0;
    let count = 0;
    for (let k = 0; k < BINS; k += 1) {
      if (k === 0) {
        const end = Math.min(BINS - 1, FREQUENCY_RADIUS);
        for (let q = 0; q <= end; q += 1) { sum += magnitudes[base + q]; count += 1; }
      } else {
        const add = k + FREQUENCY_RADIUS;
        const remove = k - FREQUENCY_RADIUS - 1;
        if (add < BINS) { sum += magnitudes[base + add]; count += 1; }
        if (remove >= 0) { sum -= magnitudes[base + remove]; count -= 1; }
      }
      percussiveScore[base + k] = sum / Math.max(1, count);
    }
  }

  const harmonic = new Float32Array(input.length);
  const percussive = new Float32Array(input.length);
  const norm = new Float32Array(input.length);
  const hSpectrum = fft.createComplexArray();
  const pSpectrum = fft.createComplexArray();
  const hInverse = fft.createComplexArray();
  const pInverse = fft.createComplexArray();
  const hReal = new Array(FFT_SIZE).fill(0);
  const pReal = new Array(FFT_SIZE).fill(0);

  for (let frameIndex = 0; frameIndex < frameCount; frameIndex += 1) {
    const start = frameIndex * HOP_SIZE;
    loadWindowedFrame(input, start, frame);
    fft.realTransform(spectrum, frame);
    hSpectrum.fill(0);
    pSpectrum.fill(0);
    const base = frameIndex * BINS;

    for (let k = 0; k < BINS; k += 1) {
      const h = harmonicScore[base + k];
      const p = percussiveScore[base + k];
      const h2 = h * h;
      const p2 = p * p;
      const hMask = h2 / (h2 + p2 + EPSILON);
      const pMask = 1 - hMask;
      const idx = k * 2;
      hSpectrum[idx] = spectrum[idx] * hMask;
      hSpectrum[idx + 1] = spectrum[idx + 1] * hMask;
      pSpectrum[idx] = spectrum[idx] * pMask;
      pSpectrum[idx + 1] = spectrum[idx + 1] * pMask;
    }

    fft.completeSpectrum(hSpectrum);
    fft.completeSpectrum(pSpectrum);
    fft.inverseTransform(hInverse, hSpectrum);
    fft.inverseTransform(pInverse, pSpectrum);
    fft.fromComplexArray(hInverse, hReal);
    fft.fromComplexArray(pInverse, pReal);

    for (let i = 0; i < FFT_SIZE; i += 1) {
      const outIndex = start + i;
      if (outIndex >= input.length) break;
      const w = WINDOW[i];
      harmonic[outIndex] += hReal[i] * w;
      percussive[outIndex] += pReal[i] * w;
      norm[outIndex] += w * w;
    }
  }

  for (let i = 0; i < input.length; i += 1) {
    const n = norm[i];
    if (n > 1e-6) {
      harmonic[i] /= n;
      percussive[i] /= n;
    } else {
      harmonic[i] = input[i];
      percussive[i] = 0;
    }
  }

  normalizePeak(harmonic, 0.98);
  normalizePeak(percussive, 0.98);
  return { harmonic, percussive, frameCount };
}

function loadWindowedFrame(input, start, frame) {
  for (let i = 0; i < FFT_SIZE; i += 1) {
    frame[i] = (input[start + i] ?? 0) * WINDOW[i];
  }
}

function lowPass(input, sampleRate, cutoff) {
  const out = new Float32Array(input.length);
  const dt = 1 / sampleRate;
  const rc = 1 / (2 * Math.PI * cutoff);
  const alpha = dt / (rc + dt);
  let y1 = 0;
  let y2 = 0;
  for (let i = 0; i < input.length; i += 1) {
    y1 += alpha * (input[i] - y1);
    y2 += alpha * (y1 - y2);
    out[i] = y2;
  }
  normalizePeak(out, 0.92);
  return out;
}

function highPass(input, sampleRate, cutoff) {
  const out = new Float32Array(input.length);
  const dt = 1 / sampleRate;
  const rc = 1 / (2 * Math.PI * cutoff);
  const alpha = rc / (rc + dt);
  let y1 = 0;
  let y2 = 0;
  let prevX1 = 0;
  let prevX2 = 0;
  for (let i = 0; i < input.length; i += 1) {
    const x = input[i];
    y1 = alpha * (y1 + x - prevX1);
    prevX1 = x;
    y2 = alpha * (y2 + y1 - prevX2);
    prevX2 = y1;
    out[i] = y2;
  }
  normalizePeak(out, 0.92);
  return out;
}

function normalizePeak(data, target) {
  let peak = 0;
  for (let i = 0; i < data.length; i += 1) peak = Math.max(peak, Math.abs(data[i]));
  if (peak < 1e-7 || peak <= target) return data;
  const gain = target / peak;
  for (let i = 0; i < data.length; i += 1) data[i] *= gain;
  return data;
}
