# Wav2mid HQ

Browser-first high quality audio → MIDI transcription. The audio stays on the device: decoding, AMT inference, cleanup, chord/key estimation and MIDI generation all run in the browser.

## Current usable feature set

- Drag/drop WAV, MP3, M4A, OGG and other browser-decodable audio
- Polyphonic transcription using Spotify Basic Pitch
- TensorFlow.js WebGL acceleration with WASM / CPU fallback
- FAST / PRO / INSANE quality presets
- Harmonic ghost-note suppression, short-note filtering and same-pitch merging
- Pitch-bend export
- Estimated tempo, key and chord timeline
- Piano-roll preview
- `.mid` and analysis `.json` downloads
- Cloudflare Workers Static Assets deployment
- No audio upload/API required

> This is a usable v0.1 foundation, not a claim that the current model already beats YouTabs. The quality roadmap is focused on source separation, instrument-specific AMT, ensemble inference and context-aware correction.

## Local development

Requirements: Node.js 20+.

```bash
npm install
npm run dev
```

`npm run prepare-runtime` copies the Basic Pitch model and TensorFlow.js WASM binaries from `node_modules` into `public/`, so inference does not depend on a third-party runtime CDN.

## Production build

```bash
npm run build
npm run preview
```

## Deploy to Cloudflare Workers

The project uses Workers Static Assets (`wrangler.jsonc`).

```bash
npx wrangler login
npm run deploy
```

For GitHub Actions deployment, create repository secrets:

- `CLOUDFLARE_API_TOKEN`
- `CLOUDFLARE_ACCOUNT_ID`

Then run the **Deploy Cloudflare Worker** workflow manually or push to `main`.

## Architecture

```text
Audio file
  ↓ Web Audio decode
TensorFlow.js backend
  ├─ WebGL (auto, usually fastest)
  └─ WASM (portable fallback)
  ↓
Spotify Basic Pitch AMT
  ↓
HQ cleanup
  ├─ short note rejection
  ├─ adjacent note merge
  ├─ duplicate suppression
  └─ conservative harmonic suppression
  ↓
Context analysis
  ├─ tempo estimate
  ├─ key estimate
  └─ chord timeline
  ↓
Piano roll + MIDI/JSON export
```

## Quality roadmap

1. **v0.2 — source separation**: optional vocals / drums / bass / keys / other stems.
2. **v0.3 — specialist models**: piano, guitar/bass and drum-specific transcription paths.
3. **v0.4 — ensemble**: multiple windows / thresholds with confidence fusion.
4. **v0.5 — context decoder**: beat-aware and chord-aware false positive correction.
5. **v0.6 — WebGPU**: ONNX/WebGPU path for larger models while retaining WASM fallback.
6. **v1.0 — HQ benchmark suite**: reproducible audio/MIDI benchmark and regression gates.

## Privacy

Audio files are read with browser APIs and are not uploaded by this app. The Cloudflare Worker serves static assets; transcription happens client-side.

## License

Project code: MIT (see `LICENSE`). Spotify Basic Pitch is an external dependency distributed under its own license (Apache-2.0/GPL-2.0 dual licensing; this project uses it as a package dependency).
