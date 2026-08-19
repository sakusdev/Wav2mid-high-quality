# Wav2mid HQ

Browser-first high-quality audio → multi-track MIDI transcription. Audio stays on-device: decoding, source separation, AMT inference, ensemble fusion, musical-context correction, drum detection and MIDI generation all run in the browser.

## Usable feature set

- Drag/drop WAV, MP3, M4A, OGG and other browser-decodable audio
- Automatic 22.05 kHz mono normalization from stereo/multichannel sources
- Long-audio chunking with overlap so full songs do not need one giant inference tensor
- Spotify Basic Pitch polyphonic AMT foundation
- TensorFlow.js WebGPU → WebGL → WASM → CPU fallback chain
- FAST / PRO / INSANE quality presets
- Chunk-local STFT harmonic/percussive soft-mask source separation
- Specialist AMT passes over mix / harmonic / bass / presence stems
- Confidence ensemble across specialist passes
- Key- and chord-aware context correction
- Conservative harmonic ghost-note suppression
- Drum onset detection and kick/snare/hi-hat classification
- Tempo, key and extended-chord timeline estimation
- Pitch-bend export
- Piano-roll preview with confidence-weighted display
- Three-track MIDI export: Harmony / Bass / Drums
- Analysis JSON including pipeline metadata and confidence information
- Cloudflare Workers Static Assets deployment
- No audio upload/API required

## Quality modes

| Mode | Pipeline |
|---|---|
| FAST | Mix-only AMT. No source separation, drums or context decoder. |
| PRO | STFT separation → mix + harmonic + bass AMT → ensemble → context correction + drums. |
| INSANE | PRO plus a presence specialist AMT pass, lower thresholds and denser chunk overlap. |

PRO is the default. INSANE intentionally trades time for recall and cross-pass agreement.

## Local development

Requirements: Node.js 20+.

```bash
npm install
npm run dev
```

`npm run prepare-runtime` copies the Basic Pitch model and TensorFlow.js WASM binaries from `node_modules` into `public/`, so inference does not depend on a third-party runtime CDN.

## Production build and E2E

```bash
npm run build
npm run test:e2e
npm run preview
```

The browser E2E creates a deterministic 44.1 kHz stereo fixture containing polyphonic tonal material plus synthetic kick/snare/hat transients. It verifies browser decoding/resampling, source separation, multi-pass AMT, ensemble/context stages, drum events, JSON v2 output and a valid three-track Standard MIDI File.

## Architecture

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
  ├─ WebGPU when usable
  ├─ WebGL fallback
  ├─ WASM portable fallback
  └─ CPU last resort
        ↓
Candidate ensemble
  ↓
Short-note / duplicate / harmonic cleanup
  ↓
Key + chord context decoder
  ↓
Tempo / key / chord timeline
  ↓
Harmony + Bass + Drums MIDI / JSON / piano roll
```

### What “source separation” means here

The current browser-native separator is an STFT soft-mask separator, not a server-side Demucs-sized neural separator. It compares temporal and frequency-smoothed spectral energy to reconstruct harmonic and percussive signals, then derives bass/presence specialist stems. This is intentionally small enough to run fully client-side and acts as a strong front-end for ensemble AMT. A future neural separator can replace this module without changing the rest of the pipeline.

### Context decoder

The context stage is a deterministic confidence decoder, not a language-model claim. It combines:

- cross-stem agreement
- estimated key membership
- local chord membership
- high-register harmonic-ghost heuristics
- note confidence and duration

Multi-pass agreement can rescue a note that looks unusual harmonically, while a weak single-pass high harmonic can be removed.

## Deploy to Cloudflare Workers

The project uses Workers Static Assets (`wrangler.jsonc`).

```bash
npx wrangler login
npm run deploy
```

For GitHub Actions deployment, configure repository secrets:

- `CLOUDFLARE_API_TOKEN`
- `CLOUDFLARE_ACCOUNT_ID`

The deploy workflow runs on `main` and can also be started manually. If those secrets are absent, the workflow reports that deployment was skipped instead of pretending a public deployment exists.

## Privacy

Audio files are read with browser APIs and are not uploaded by this app. Cloudflare serves application/model assets only; transcription runs on the user's device.

## License

Project code: MIT (see `LICENSE`). Spotify Basic Pitch is an external dependency under its own license. FFT.js is MIT licensed and used for the browser STFT separator.
