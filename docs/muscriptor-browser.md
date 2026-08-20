# MuScriptor ULTRA — browser WebGPU path

MuScriptor ULTRA is an optional, non-commercial transcription mode that runs the MuScriptor-small transformer directly in the browser. It does **not** require the localhost bridge used by the earlier prototype.

## Runtime path

```text
Audio file
  ↓ Web Audio decode
16 kHz mono
  ↓
5 s chunks
  ↓
HTK log-mel frontend
  - n_fft 2048
  - hop 160 (100 fps)
  - 512 mel bins
  - periodic Hann
  - centered reflect padding
  - magnitude / power=1.0
  - log(mel + 1e-6)
  ↓
MuScriptor conditioner (ONNX / WebGPU)
  ↓ GPU prefix embeddings
Streaming MuScriptor decoder
  - INT4 MatMulNBits weight-only weights
  - persistent GPU KV cache
  - autoregressive token generation
  - tie prompt across 5 s chunk boundaries
  ↓
Multi-instrument notes + drums
  ↓
Browser MIDI / JSON / piano-roll output
```

The mel frontend intentionally uses magnitude (`power=1.0`). This matches the MuScriptor inference conditioner; changing it to power-squared changes the model input distribution.

## Lazy loading and privacy

The normal application startup does not load MuScriptor, ONNX Runtime, or MuScriptor model weights. The ULTRA UI dynamically imports the browser runtime only after ULTRA analysis starts.

Audio is decoded and transcribed on the user's device. The audio file is not uploaded to a transcription server. Network requests made by ULTRA are limited to application/runtime/model assets required to execute the model.

## Model distribution on Cloudflare

MuScriptor model weights and derived browser ONNX files are **not committed to this repository**. Deployment generates the derived browser graphs from the upstream checkpoint only when a real Cloudflare deployment is being built.

The INT4 decoder is roughly 55 MiB, larger than Workers Static Assets' single-file gate and the repository's stricter 5 MiB temporary-preview gate. `tools/stage_muscriptor_cloudflare.py` therefore splits oversized ONNX files into <=4.5 MiB immutable static parts.

The public browser manifest deliberately keeps the original `.onnx` URL. `worker.js` intercepts that URL and streams the static parts in order from the `ASSETS` binding, so ONNX Runtime Web receives the exact original byte stream and does not need a custom model loader. The heavy export CI verifies the reconstructed decoder's byte count and SHA-256 against the original exported ONNX file.

```text
browser /models/.../decoder.onnx
          ↓
Cloudflare worker.js
          ↓
stream-map.json → decoder.parts.json
          ↓
part-000.bin
part-001.bin
...
          ↓ streamed, no full Worker-side buffering
exact original ONNX byte stream
```

This lets both normal and temporary Cloudflare deployments satisfy their per-file limits while keeping the Git repository free of large binary model artifacts.

The application code is MIT. MuScriptor model weights/derived model artifacts retain their upstream **CC BY-NC 4.0** terms. ULTRA is therefore explicitly marked `NC` in the UI and must not be represented as a commercial-use model path.

## Exported graph contract

The exporter creates two final graphs:

- `conditioner`: `[1, 501, 512]` float32 log-mel plus instrument condition IDs → prefix embeddings.
- `decoder`: prefix embeddings + token IDs + `past_len` + per-layer K/V cache → logits + newly generated K/V rows.

The safe production baseline uses FP32 activations for both graphs and INT4 weight-only quantization for decoder MatMul weights. Keeping conditioner and decoder activation types identical allows the conditioner GPU output to feed the decoder directly without an intermediate CPU cast/copy.

Mixed FP16 activation is an optimization candidate, not a correctness requirement. It should only replace the FP32 baseline after both ONNX Runtime validation and Chrome WebGPU execution pass.

## Validation gates

A browser artifact is not considered usable merely because ONNX export succeeds. The gates check, in order:

1. PyTorch wrapper parity against the loaded MuScriptor-small checkpoint.
2. FP32 ONNX Runtime parity.
3. Final conditioner → INT4 decoder execution with ONNX Runtime CPU.
4. Output dtype/shape/finiteness and every layer's K/V shape.
5. Deterministic first-token fixture recorded into the model manifest.
6. Cloudflare-safe chunk staging with every static asset <=5 MiB.
7. Local Wrangler reconstruction of the streamed decoder with exact byte-count and SHA-256 parity.
8. Chrome WebGPU execution of the **final conditioner and final INT4 decoder**, using the same fixture when the CI machine exposes a WebGPU adapter.
9. When WebGPU is available, its first token must equal the CPU ORT first token. Adapter absence on GitHub's hosted Linux runner is reported as an infrastructure skip rather than a model failure.

The ordinary CI remains lightweight: it verifies the UI is lazy and uses a mocked browser model. It does not download the large non-commercial checkpoint on every commit.

## Local bridge

`tools/muscriptor_bridge.py` may remain useful as a development/reference implementation, but the user-facing ULTRA mode does not depend on it. A missing localhost service must never be a prerequisite for browser ULTRA.
