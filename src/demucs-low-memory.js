import { CONSTANTS } from 'demucs-web/constants';
import { fft, ifft, stft, istft, reflectPad, getHannWindow } from 'demucs-web/fft';
import { standaloneIspec, prepareModelInput } from 'demucs-web/processor';

const {
  TRAINING_SAMPLES,
  MODEL_SPEC_BINS,
  MODEL_SPEC_FRAMES,
  SEGMENT_OVERLAP,
  TRACKS,
} = CONSTANTS;

export { CONSTANTS, fft, ifft, stft, istft, reflectPad, getHannWindow, standaloneIspec, prepareModelInput };

/**
 * Memory-oriented drop-in replacement for demucs-web's DemucsProcessor.
 *
 * The upstream processor temporarily expands all four frequency-domain stems
 * at once, then allocates another four pairs of time-domain buffers. On mobile
 * browsers that peak sits on top of the ~172 MB model/session and can kill the
 * renderer. This implementation processes one stem at a time and disposes ORT
 * tensors immediately after a segment is folded into the output.
 *
 * It also gives remote model URLs directly to ORT instead of buffering the
 * entire response into chunks and then copying those chunks into a second
 * contiguous ArrayBuffer.
 */
export class DemucsProcessor {
  constructor(options = {}) {
    this.ort = options.ort || null;
    this.session = null;
    this.modelPath = options.modelPath || './htdemucs_embedded.onnx';
    this.sessionOptions = options.sessionOptions || {};
    this.onProgress = options.onProgress || (() => {});
    this.onLog = options.onLog || (() => {});
    this.onDownloadProgress = options.onDownloadProgress || (() => {});
    this.mobileSafe = isMobileLike();
    this.backend = 'uninitialized';
  }

  async loadModel(modelPathOrBuffer) {
    if (!this.ort) throw new Error('ONNX Runtime not provided. Pass ort in constructor options.');

    const source = modelPathOrBuffer || this.modelPath;
    const requestedProviders = this.sessionOptions.executionProviders || ['webgpu', 'wasm'];
    const executionProviders = this.mobileSafe ? ['wasm'] : requestedProviders;
    this.backend = executionProviders[0] || 'wasm';

    this.onLog('model', this.mobileSafe
      ? 'Loading model directly with mobile-safe WASM backend…'
      : `Loading model directly with ${this.backend} backend…`);

    // ORT accepts a URL, ArrayBuffer, or Uint8Array. Passing the URL straight
    // through avoids demucs-web's fetch-chunks + combine-copy peak allocation.
    this.session = await this.ort.InferenceSession.create(source, {
      graphOptimizationLevel: 'basic',
      ...this.sessionOptions,
      executionProviders,
      enableCpuMemArena: false,
      enableMemPattern: false,
    });

    this.onLog('model', `Model loaded successfully (${this.backend})`);
    return this.session;
  }

  async separate(leftChannel, rightChannel) {
    if (!this.session) throw new Error('Model not loaded. Call loadModel() first.');

    const totalSamples = Math.min(leftChannel.length, rightChannel.length);
    const stride = Math.floor(TRAINING_SAMPLES * (1 - SEGMENT_OVERLAP));
    const numSegments = Math.max(1, Math.ceil((totalSamples - TRAINING_SAMPLES) / stride) + 1);
    const outputs = TRACKS.map(() => ({
      left: new Float32Array(totalSamples),
      right: new Float32Array(totalSamples),
    }));
    const weights = new Float32Array(totalSamples);

    let segmentIdx = 0;
    for (let start = 0; start < totalSamples; start += stride) {
      const end = Math.min(start + TRAINING_SAMPLES, totalSamples);
      const segmentLength = end - start;
      const segLeft = new Float32Array(TRAINING_SAMPLES);
      const segRight = new Float32Array(TRAINING_SAMPLES);
      segLeft.set(leftChannel.subarray(start, end));
      segRight.set(rightChannel.subarray(start, end));

      const input = prepareModelInput(segLeft, segRight);
      const waveformTensor = new this.ort.Tensor('float32', input.waveform, [1, 2, TRAINING_SAMPLES]);
      const magSpecTensor = new this.ort.Tensor(
        'float32',
        input.magSpec,
        [1, 4, MODEL_SPEC_BINS, MODEL_SPEC_FRAMES],
      );

      const feeds = { [this.session.inputNames[0]]: waveformTensor };
      if (this.session.inputNames.length > 1) feeds[this.session.inputNames[1]] = magSpecTensor;

      let inferResults;
      try {
        inferResults = await this.session.run(feeds);
        const parsed = parseOutputs(this.session.outputNames, inferResults);
        if (!parsed.timeData) throw new Error('Could not find time-domain output tensor');

        const overlapWindow = makeOverlapWindow(segmentLength, stride);
        for (let track = 0; track < TRACKS.length; track += 1) {
          // Crucially, expand and iSTFT only one frequency stem at a time.
          const freqOutput = parsed.freqData
            ? standaloneIspec(extractTrackSpec(parsed.freqData, track), TRAINING_SAMPLES)
            : null;

          foldTrack({
            destination: outputs[track],
            start,
            segmentLength,
            totalSamples,
            track,
            timeData: parsed.timeData,
            timeShape: parsed.timeShape,
            freqOutput,
            overlapWindow,
          });

          // Let large per-track iSTFT arrays become unreachable before the next
          // stem is expanded. A task yield helps mobile Chromium reclaim them.
          if (this.mobileSafe) await taskYield();
        }

        for (let i = 0; i < segmentLength && start + i < totalSamples; i += 1) {
          weights[start + i] += overlapWindow[i];
        }
      } finally {
        waveformTensor.dispose?.();
        magSpecTensor.dispose?.();
        if (inferResults) {
          for (const tensor of Object.values(inferResults)) tensor?.dispose?.();
        }
      }

      segmentIdx += 1;
      this.onProgress({
        progress: segmentIdx / numSegments,
        currentSegment: segmentIdx,
        totalSegments: numSegments,
        backend: this.backend,
        lowMemory: true,
      });
      if (this.mobileSafe) await taskYield(20);
    }

    for (let track = 0; track < TRACKS.length; track += 1) {
      const stem = outputs[track];
      for (let i = 0; i < totalSamples; i += 1) {
        const weight = weights[i];
        if (weight > 0) {
          stem.left[i] /= weight;
          stem.right[i] /= weight;
        }
      }
    }

    return {
      drums: outputs[0],
      bass: outputs[1],
      other: outputs[2],
      vocals: outputs[3],
      __wav2midBackend: this.backend,
      __wav2midLowMemory: true,
    };
  }
}

function parseOutputs(outputNames, results) {
  let timeData = null;
  let timeShape = null;
  let freqData = null;
  for (const name of outputNames) {
    const tensor = results[name];
    if (tensor?.dims?.length === 4 && tensor.dims[2] === 2) {
      timeData = tensor.data;
      timeShape = tensor.dims;
    } else if (tensor?.dims?.length === 5 && tensor.dims[2] === 4) {
      freqData = tensor.data;
    }
  }
  return { timeData, timeShape, freqData };
}

function extractTrackSpec(freqData, track) {
  const bins = MODEL_SPEC_BINS;
  const frames = MODEL_SPEC_FRAMES;
  const channelStride = bins * frames;
  const trackBase = track * 4 * channelStride;
  const spec = {
    leftReal: new Float32Array(channelStride),
    leftImag: new Float32Array(channelStride),
    rightReal: new Float32Array(channelStride),
    rightImag: new Float32Array(channelStride),
  };

  // Model layout is channel-major [track, complex-channel, bin, frame], while
  // standaloneIspec expects each complex plane flattened as bin*frames+frame.
  spec.leftReal.set(freqData.subarray(trackBase, trackBase + channelStride));
  spec.leftImag.set(freqData.subarray(trackBase + channelStride, trackBase + 2 * channelStride));
  spec.rightReal.set(freqData.subarray(trackBase + 2 * channelStride, trackBase + 3 * channelStride));
  spec.rightImag.set(freqData.subarray(trackBase + 3 * channelStride, trackBase + 4 * channelStride));
  return spec;
}

function foldTrack({
  destination,
  start,
  segmentLength,
  totalSamples,
  track,
  timeData,
  timeShape,
  freqOutput,
  overlapWindow,
}) {
  const channels = timeShape[2];
  const samples = timeShape[3];
  const trackBase = track * channels * samples;
  const rightBase = trackBase + samples;

  for (let i = 0; i < segmentLength && start + i < totalSamples; i += 1) {
    const left = timeData[trackBase + i] + (freqOutput?.left?.[i] || 0);
    const right = timeData[rightBase + i] + (freqOutput?.right?.[i] || 0);
    const weight = overlapWindow[i];
    destination.left[start + i] += left * weight;
    destination.right[start + i] += right * weight;
  }
}

function makeOverlapWindow(segmentLength, stride) {
  const window = new Float32Array(segmentLength);
  for (let i = 0; i < segmentLength; i += 1) {
    const fadeIn = Math.min(i / Math.max(1, stride * 0.5), 1);
    const fadeOut = Math.min((segmentLength - i) / Math.max(1, stride * 0.5), 1);
    window[i] = Math.min(fadeIn, fadeOut);
  }
  return window;
}

function isMobileLike() {
  const ua = globalThis.navigator?.userAgent || '';
  const memory = Number(globalThis.navigator?.deviceMemory || 0);
  return /Android|iPhone|iPad|Mobile/i.test(ua) || (memory > 0 && memory <= 4);
}

function taskYield(ms = 0) {
  return new Promise(resolve => setTimeout(resolve, ms));
}
