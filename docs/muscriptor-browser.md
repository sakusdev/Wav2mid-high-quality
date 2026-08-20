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

## Model distribution

MuScriptor model weights and derived browser ONNX files are **not committed to this repository**. The decoder is larger than Cloudflare Workers Static Assets' per-file deployment gate, so production must serve the derived ONNX assets from an external model host and point the generated manifest URLs at that host.

The application code is MIT. MuScriptor model weights/derived model artifacts retain their upstream **CC BY-NC 4.0** terms. ULTRA is therefore explicitly marked `NC` in the UI and must not be represented as a commercial-use model path.

## Exported graph contract

The exporter creates two final graphs:

- `conditioner`: `[1, 501, 512]` float32 log-mel plus instrument condition IDs → prefix embeddings.
- `decoder`: prefix embeddings + token IDs + `past_len` + per-layer K/V cache → logits + newly generated K/V rows.

The safe production baseline uses FP32 activations for both graphs and INT4 weight-only quantization for decoder MatMul weights. Keeping conditioner and decoder activation types identical allows the conditioner GPU output to feed the decoder directly without an intermediate CPU cast/copy.

Mixed FP16 activation is an optimization candidate, not a correctness requirement. It should only replace the FP32 baseline after both ONNX Runtime validation and Chrome WebGPU execution pass.

## Validation gates

A browser artifact is not considered usable merely because ONNX export succeeds. The PR gate checks, in order:

1. PyTorch wrapper parity against the loaded MuScriptor-small checkpoint.
2. FP32 ONNX Runtime parity.
3. Final conditioner → INT4 decoder execution with ONNX Runtime CPU.
4. Output dtype/shape/finiteness and every layer's K/V shape.
5. Deterministic first-token fixture recorded into the model manifest.
6. Chrome WebGPU execution of the **final conditioner and final INT4 decoder**, using the same fixture.
7. WebGPU first token must equal the CPU ORT first token.

The ordinary CI remains lightweight: it verifies the UI is lazy and uses a mocked browser model. It does not download the large non-commercial checkpoint on every commit.

## Local bridge

`tools/muscriptor_bridge.py` may remain useful as a development/reference implementation, but the user-facing ULTRA mode does not depend on it. A missing localhost service must never be a prerequisite for browser ULTRA.
