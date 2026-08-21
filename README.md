# Wav2mid HQ

Browser-first high-quality audio → multi-track MIDI transcription. Audio stays on-device: decoding, source separation, AMT inference, ensemble fusion, musical-context correction, drum detection and MIDI generation run in the browser.

## Usable feature set

- Drag/drop WAV, MP3, M4A, OGG and other browser-decodable audio
- Automatic sample-rate/channel normalization
- Long-audio chunking with overlap for the lightweight pipeline
- Spotify Basic Pitch polyphonic AMT foundation
- TensorFlow.js WebGPU → WebGL → WASM → CPU fallback chain for AMT
- FAST / PRO / INSANE lightweight quality presets
- Optional **NEURAL HQ**: HTDemucs 4-stem neural separation in-browser
- Chunk-local STFT harmonic/percussive soft-mask separation for PRO/INSANE
- Stem-specialized AMT passes over mix / harmonic / bass / presence signals
- Confidence ensemble and INSANE cross-pass consensus
- Key- and chord-aware context correction
- Conservative harmonic ghost-note suppression and same-pitch fragment repair
- Drum onset detection and kick/snare/hi-hat classification
- Tempo, key and extended-chord timeline estimation
- Pitch-bend export
- Piano-roll preview with confidence-weighted display
- Three musical MIDI tracks: Harmony / Bass / Drums
- Analysis JSON including pipeline metadata and confidence information
- Reproducible SOTA benchmark lab with browser / command / precomputed model adapters
- MAESTRO v3 and Slakh2100 benchmark-manifest importers
- Validation-set mode/sensitivity auto-tuner
- Cloudflare Workers Static Assets deployment
- Build-time 25 MiB/file production and 5 MiB/file temporary-deploy gates
- No audio upload/API required

## Quality modes

| Mode | Pipeline |
|---|---|
| FAST | Mix-only AMT. No source separation, drums or context decoder. |
| PRO | STFT separation → mix + harmonic + bass AMT → ensemble → context correction + drums. |
| INSANE | PRO plus presence AMT and dense overlap; final notes use a 3-of-4 cross-pass consensus rule with a narrow high-confidence exception. |
| NEURAL HQ | HTDemucs → drums / bass / other / vocals → four AMT passes → neural stem ensemble → context correction + isolated-drum analysis. |

PRO is the default and needs no huge separator model. INSANE is precision-oriented: the extra pass is used as evidence instead of merely lowering thresholds. NEURAL HQ is the slow/high-quality separation path and downloads the HTDemucs model on first use (roughly 172 MB); the model is cached by normal browser HTTP caching where available.

## NEURAL HQ

NEURAL HQ loads `demucs-web` only when the switch is enabled and analysis actually starts. The separator expects 44.1 kHz stereo, so the input is normalized to that format before HTDemucs inference.

```text
Audio
  ↓ 44.1 kHz stereo normalization
HTDemucs 4-stem (ONNX Runtime Web)
  ├─ drums ───────────────→ drum onset/classification
  ├─ bass ──→ Basic Pitch bass pass ┐
  ├─ other ─→ Basic Pitch harmony   ├─→ confidence ensemble
  ├─ vocals → Basic Pitch melody    │
  └─ mix ───→ Basic Pitch baseline ┘
                     ↓
               key/chord context
                     ↓
          Harmony + Bass + Drums MIDI
```

The HTDemucs model is fetched from the model URL supplied by `demucs-web`; the user's audio is not uploaded. ONNX Runtime Web is also loaded only by NEURAL HQ from a version-pinned jsDelivr release. The small JSEP module/worker glue is staged on the app origin to avoid cross-origin module-worker failures, while the oversized JSEP WASM payload remains on jsDelivr because it exceeds Cloudflare Workers Static Assets' 25 MiB per-file limit. The normal FAST/PRO/INSANE application remains lightweight, while the optional neural path fetches its neural runtime and model only on demand.

## MuScriptor ULTRA diagnostics

The non-commercial MuScriptor browser path stays lazy: neither its models nor ONNX Runtime are fetched until MuScriptor analysis actually starts. When it does start, a lightweight diagnostic layer records the exact failure stage before and during model execution.

Typical stage/error pairs include:

- `manifest` → `MUSCRIPTOR_MANIFEST_*`
- `conditioner-asset` / `decoder-asset` → `MUSCRIPTOR_ASSET_*`
- `webgpu-adapter` / `webgpu-device` → `MUSCRIPTOR_WEBGPU_*`
- `ort-runtime` → `MUSCRIPTOR_ORT_*`
- `conditioner-session` / `decoder-session` → `MUSCRIPTOR_*_SESSION`
- `conditioner-inference` / `decoder-inference` → `MUSCRIPTOR_*_INFERENCE`
- GPU OOM/device loss → `MUSCRIPTOR_GPU_MEMORY`
- JSEP/WASM fetch/init failure → `MUSCRIPTOR_ORT_RUNTIME_FETCH`

The latest trace is exposed as `window.__WAV2MID_MUSCRIPTOR_DIAGNOSTICS__` for debugging. It contains stage/status/detail records only; it does not contain the user's audio.

## Benchmark lab — proving "strongest"

`bench/` is the reproducible comparison layer. Every adapter is scored by the same evaluator instead of comparing incompatible numbers copied from different papers.

Metrics include exact-pitch onset F1, onset+offset F1, interval/frame F1 and exact GM drum-class onset F1. The combined objective also penalizes hallucinated drums on drumless references, which matters for piano attacks that resemble percussion.

Supported adapter types:

- **browser** — drives the production Wav2mid UI in Chrome and downloads its real JSON export;
- **command** — runs a local competitor/specialist executable with `{audio}` / `{output}` placeholders;
- **precomputed** — scores existing MIDI/JSON, useful for YouTabs or externally generated results.

`bench/adapters.example.json` contains slots for Wav2mid PRO / INSANE / NEURAL, MuScriptor, Neural Semi-CRF focused models, instrument-specific piano/bass/guitar CRNN models, ADTOF drums, YourMT3+ and YouTabs.

```bash
# Build a real-piano held-out manifest
npm run benchmark:prepare-maestro -- --dataset /path/to/maestro-v3 --output bench/manifests/maestro-v3.json

# Build a multitrack/instrument manifest
npm run benchmark:prepare-slakh -- --dataset /path/to/slakh2100_flac_redux --output bench/manifests/slakh2100.json

# Evaluate one browser adapter
npm run benchmark:suite -- --manifest bench/manifests/maestro-v3.json --adapters bench/adapters.example.json --adapter wav2mid-pro --split validation --max-items 20

# Tune mode + sensitivity on a validation set
npm run tune -- --manifest bench/manifests/maestro-v3.json --adapters bench/adapters.example.json --adapter wav2mid-pro --split validation --max-items 20
```

The benchmark tooling is intentionally separate from the UI pipeline: model changes can be compared against exactly the same held-out set and metric definitions before promoting them.
