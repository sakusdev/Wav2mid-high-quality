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
- Specialist AMT passes over isolated stems
- Confidence ensemble across specialist passes
- Key- and chord-aware context correction
- Conservative harmonic ghost-note suppression
- Drum onset detection and kick/snare/hi-hat classification
- Tempo, key and extended-chord timeline estimation
- Pitch-bend export
- Piano-roll preview with confidence-weighted display
- Three musical MIDI tracks: Harmony / Bass / Drums
- Analysis JSON including pipeline metadata and confidence information
- Cloudflare Workers Static Assets deployment
- No audio upload/API required

## Quality modes

| Mode | Pipeline |
|---|---|
| FAST | Mix-only AMT. No source separation, drums or context decoder. |
| PRO | STFT separation → mix + harmonic + bass AMT → ensemble → context correction + drums. |
| INSANE | PRO plus a presence specialist AMT pass, lower thresholds and denser chunk overlap. |
| NEURAL HQ | HTDemucs → drums / bass / other / vocals → four AMT passes → neural stem ensemble → context correction + isolated-drum analysis. |

PRO is the default and needs no huge separator model. INSANE increases recall and cross-pass agreement. NEURAL HQ is the slow/high-quality path and downloads the HTDemucs model on first use (roughly 172 MB); the model is cached by normal browser HTTP caching where available.

## NEURAL HQ

NEURAL HQ dynamically loads `demucs-web` + ONNX Runtime Web only when the switch is enabled. The separator expects 44.1 kHz stereo, so the input is normalized to that format before HTDemucs inference.

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

The HTDemucs model is fetched from the model URL supplied by `demucs-web`; the user's audio is not uploaded. ONNX Runtime Web prefers WebGPU when available and uses the bundled WASM runtime as fallback. `npm run prepare-runtime` copies ONNX Runtime WASM binaries into `public/ort-wasm/` so the fallback is self-hosted with the app.

## Local development

Requirements: Node.js 20+.

```bash
npm install
npm run dev
```

`npm run prepare-runtime` copies:

- the Basic Pitch model
- TensorFlow.js WASM binaries
- ONNX Runtime Web WASM binaries

into `public/`, avoiding third-party runtime-CDN dependencies. The large HTDemucs model remains on-demand so ordinary FAST/PRO/INSANE users do not download it.

## Production build and E2E

```bash
npm run build
npm run test:e2e
npm run preview
```

The standard browser E2E creates a deterministic 44.1 kHz stereo fixture containing polyphonic tonal material plus synthetic kick/snare/hat transients. It verifies browser decoding/resampling, STFT separation, multi-pass PRO AMT, ensemble/context stages, drum events, JSON output and a valid multi-track Standard MIDI File. CI intentionally does not download the ~172 MB HTDemucs model; it smoke-checks that the NEURAL HQ control and lazy neural bundle are present while the normal PRO regression remains fast and deterministic.

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
  ├─ bass specialist
  ├─ presence specialist (INSANE)
  └─ percussive → drum onset/classification
        ↓
TensorFlow.js Basic Pitch AMT
        ↓
Candidate ensemble
  ↓
Key + chord context decoder
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

Audio files are read with browser APIs and are not uploaded by this app. Cloudflare serves application/runtime assets; Basic Pitch and HTDemucs inference run on the user's device. NEURAL HQ does fetch its model file on first use.

## License

Project code: MIT (see `LICENSE`). Spotify Basic Pitch, `demucs-web`, ONNX Runtime Web and FFT.js are external dependencies under their own licenses.
