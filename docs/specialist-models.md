# Browser specialist models

Wav2mid can run benchmark-promoted high-resolution CRNN transcription specialists in the browser through ONNX Runtime Web. This layer is deliberately separate from the default Basic Pitch pipeline: a specialist is not enabled merely because a checkpoint exists.

## Runtime contract

A specialist model receives a frontendless tensor:

```text
16 kHz mono waveform
  ↓ periodic Hann STFT (n_fft 2048, hop 160, center=True, reflect padding)
229-bin Slaney-normalized mel filter bank (30 Hz → 8 kHz)
  ↓ 10 * log10(power), amin=1e-10, top_db=None
[batch, 1, frames, 229]
  ↓ ONNX CRNN body
reg_onset_output
reg_offset_output
frame_output
velocity_output
```

The browser frontend mirrors TorchLibrosa/Librosa settings used by the Kong/ByteDance high-resolution transcription architecture. Long audio is split into 10-second segments with 50% overlap. Predictions are stitched with the same 75%/50%/75% deframe policy used by the reference inference implementation, then trimmed to the real input duration.

The postprocessor keeps the original regression semantics: onset local maxima use a 2-frame monotonic neighborhood, offsets use 4 frames, sub-frame shifts are recovered from the neighboring regression values, frame disappearance competes with the predicted offset, and velocity is mapped to MIDI velocity.

## Export a checkpoint

Install the exporter dependencies in a separate Python environment; they are not required for normal web development.

```bash
python -m venv .venv-specialist
source .venv-specialist/bin/activate
pip install torch piano-transcription-inference onnx onnxruntime

python scripts/export-crnn-specialist.py \
  --checkpoint /models/filobass_20000_iterations.pth \
  --output /models/onnx/bass.onnx \
  --instrument bass \
  --model-url https://MODEL_HOST/bass.onnx
```

The exporter removes the TorchLibrosa frontend from the graph so WebGPU sees mostly Conv/BatchNorm/GRU/Linear/Sigmoid operations. It accepts both standalone note checkpoints and combined piano `Note_pedal` checkpoints by extracting `note_model.*` when needed.

If `onnxruntime` is installed, export also runs a PyTorch-vs-ONNX forward parity check and writes the maximum absolute error into the generated JSON sidecar. The sidecar also records the source checkpoint SHA256 and all frontend/postprocess parameters.

Do not commit large checkpoints or ONNX weights to this repository by default. Host them in a location whose redistribution terms allow it, then set the sidecar/model manifest URL accordingly.

## Benchmark before promotion

Run the same held-out manifest through the current Wav2mid baseline and the candidate model. The benchmark harness supports command adapters, so a PyTorch specialist can be scored before spending time exporting it to ONNX.

```bash
npm run benchmark:suite -- \
  --manifest bench/local-slakh-test.json \
  --adapters bench/adapters.local.json \
  --adapter wav2mid-insane,crnn-bass-candidate \
  --split test
```

For an instrument specialist, use benchmark items tagged with the corresponding instrument. Never tune thresholds on the held-out test split.

## Promote only a winner

After exporting the winning checkpoint, gate promotion against the benchmark report:

```bash
npm run specialist:promote -- \
  --benchmark benchmark-results/benchmark-....json \
  --candidate crnn-bass-candidate \
  --baseline wav2mid-insane \
  --instrument bass \
  --stem bass \
  --metadata /models/onnx/bass.json \
  --min-delta 0.01
```

Promotion is refused unless the candidate objective exceeds the baseline by at least the requested delta on paired items. A successful promotion writes the model metadata and benchmark evidence into `public/specialists/manifest.json` with `promoted: true`.

The browser runtime ignores entries that are not promoted, are explicitly disabled, have no URL, or do not match the requested stem.

## Model hosting and Cloudflare

Specialist weights can be far larger than the Workers Static Assets per-file limit. Keep large ONNX files outside the normal `dist/` bundle (for example, an appropriate model host or object storage) and load them lazily. The user's audio remains local; only model/runtime assets are fetched.

Cross-origin model hosting must permit browser fetches from the Wav2mid origin. Prefer immutable/versioned URLs and record a checksum in metadata.

## Promotion policy

A production specialist should satisfy all of the following:

1. license and redistribution terms have been checked;
2. held-out benchmark objective beats the current production baseline by the configured margin;
3. ONNX forward parity is within the accepted numerical tolerance;
4. frontend compatibility tests pass;
5. WebGPU and WASM fallback paths complete on representative hardware;
6. latency/memory do not make the user experience worse than the accuracy gain justifies.

The goal is not to accumulate models. The goal is to keep only specialists that measurably improve the final MIDI.
