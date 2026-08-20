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

The HTDemucs model is fetched from the model URL supplied by `demucs-web`; the user's audio is not uploaded. ONNX Runtime Web is also loaded only by NEURAL HQ from a version-pinned jsDelivr release. This is deliberate: ONNX Runtime's WebGPU JSEP WASM is larger than Cloudflare Workers Static Assets' 25 MiB per-file limit. The normal FAST/PRO/INSANE application remains fully self-hosted, while the optional neural path fetches its neural runtime and model only on demand.

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
npm run benchmark:prepare-maestro -- \
  --root /datasets/maestro-v3.0.0 \
  --split test \
  --output bench/local-maestro-test.json

# Build an aligned multi-instrument manifest
npm run benchmark:prepare-slakh -- \
  --root /datasets/Slakh2100 \
  --output bench/local-slakh.json \
  --limit 50

# Same evaluator, multiple systems
npm run benchmark:suite -- \
  --manifest bench/local-maestro-test.json \
  --adapters bench/adapters.local.json \
  --adapter wav2mid-insane,muscriptor-small
```

The Slakh importer merges the actual `MIDI/Sxx.mid` files used for stem rendering rather than treating `all_src.mid` as exact synthesis ground truth.

### Auto-tuning

Tune only on a validation split, then evaluate once on an untouched test split.

```bash
npm run tune -- \
  --manifest bench/local-validation.json \
  --split validation \
  --backend wasm \
  --modes pro,insane \
  --sensitivities 0.80,0.90,1.00,1.10,1.20
```

The tuner executes the actual production browser pipeline for every candidate and ranks the objective. A specialist model is a **candidate**, not an automatic upgrade: it should be promoted to the browser/WebGPU pipeline only after it beats the current path on the relevant held-out split and its model license is compatible with deployment.

CI includes a complete benchmark smoke: it generates aligned ground-truth WAV+MIDI, launches the production app, transcribes through Chrome, downloads the real analysis JSON, scores it with the common evaluator, writes a leaderboard, and enforces a minimum objective.

See [`bench/README.md`](bench/README.md) for metric definitions, adapters, licenses and reproducibility rules.

## Local development

Requirements: Node.js 20+.

```bash
npm install
npm run dev
```

`npm run prepare-runtime` copies the Basic Pitch model and TensorFlow.js WASM binaries into `public/`. The large neural separator model and ONNX Runtime neural runtime are not copied into the Cloudflare static bundle.

## Production build and E2E

```bash
npm run build
npm run test:e2e
npm run test:bench:smoke
npm run preview
```

`npm run build` also scans every output file and fails if any asset exceeds 25 MiB, matching Cloudflare Workers Static Assets' individual-file limit.

The standard browser E2E creates a deterministic 44.1 kHz stereo fixture containing polyphonic tonal material plus synthetic kick/snare/hat transients. It verifies browser decoding/resampling, STFT separation, multi-pass PRO AMT, ensemble/context stages, drum events, JSON output and a valid multi-track Standard MIDI File. CI intentionally does not download the ~172 MB HTDemucs model; it separately smoke-checks that NEURAL HQ is visible and that neither ONNX Runtime nor the HTDemucs model is fetched merely by toggling the option.

## Lightweight architecture

```text
Audio file
  ↓ Web Audio decode
22.05 kHz mono normalization
  ↓
Overlapped chunks
  ↓
STFT harmonic/percussive soft masking (PRO/INSANE)
  ├─ mix
  ├─ harmonic
  ├─ bass pass
  ├─ presence pass (INSANE)
  └─ percussive → drum onset/classification
        ↓
TensorFlow.js Basic Pitch AMT
        ↓
Candidate ensemble / INSANE consensus
  ↓
Key + chord context decoder
  ↓
Real-world continuity + harmonic refiner
  ↓
Harmony + Bass + Drums MIDI / JSON / piano roll
```

### Context decoder

The context stage is a deterministic confidence decoder. It combines cross-stem agreement, estimated key membership, local chord membership, high-register harmonic-ghost heuristics, note confidence and duration. Multi-pass agreement can rescue an unusual note while a weak single-pass high harmonic can be removed.

## Deploy to Cloudflare Workers

The project uses Workers Static Assets (`wrangler.jsonc`).

```bash
npx wrangler login
npm run deploy
```

For GitHub Actions deployment, configure repository secrets:

- `CLOUDFLARE_API_TOKEN`
- `CLOUDFLARE_ACCOUNT_ID`

The deploy workflow runs on `main` and can also be started manually. If those secrets are absent, it reports that deployment was skipped instead of pretending a public deployment exists.

## Privacy

Audio files are read with browser APIs and are not uploaded by this app. Cloudflare serves the application and lightweight AMT assets. NEURAL HQ fetches the pinned ONNX Runtime browser distribution and HTDemucs model, then performs separation and transcription locally on the user's device.

## License

Project code: MIT (see `LICENSE`). Spotify Basic Pitch, `demucs-web`, ONNX Runtime Web and FFT.js are external dependencies under their own licenses. Benchmark datasets and third-party model checkpoints retain their own licenses and are never relicensed by this repository.
