# AMT benchmark lab

This directory turns "best" into a reproducible claim. Every model is evaluated by the same Node evaluator and the same aligned audio/MIDI manifest.

## Metrics

The evaluator reports:

- note onset precision / recall / F1, exact MIDI pitch with a default 50 ms onset tolerance;
- note onset+offset precision / recall / F1, with offset tolerance `max(50 ms, 20% of the reference duration)` by default;
- frame precision / recall / F1 from exact interval overlap per MIDI pitch;
- GM drum onset precision / recall / F1 by exact MIDI drum class;
- a single objective used by the tuner: tonal score = 55% onset F1 + 20% offset F1 + 25% frame F1. A 10% drum term is activated whenever either the reference or prediction contains drums, so hallucinated drums on a drumless piano recording are explicitly penalized.

All adapters use those same metrics. Do not compare a number produced by another repository's evaluator directly with this leaderboard unless the metric definition is identical.

## Dataset manifests

A manifest is intentionally just paths and metadata; datasets are not vendored into this repository.

```bash
npm run benchmark:prepare-maestro -- --root /datasets/maestro-v3.0.0 --split test --output bench/local-maestro-test.json
npm run benchmark:prepare-slakh -- --root /datasets/Slakh2100 --output bench/local-slakh.json --limit 50
```

MAESTRO is useful for real acoustic-piano timing and is distributed separately under CC BY-NC-SA 4.0. Slakh2100 is useful for aligned multi-instrument mixtures and is distributed separately under CC BY 4.0.

The Slakh importer deliberately merges `MIDI/Sxx.mid` files for the reference instead of using `all_src.mid`: the Sxx MIDI files are the exact files used to synthesize the distributed stems, while `all_src.mid` can differ because of synthesis heuristics.

For any private/user-created benchmark, copy `manifest.example.json` and point it at aligned audio and MIDI. Do not commit copyrighted evaluation audio unless its license permits redistribution.

## Model adapters

Copy `adapters.example.json`, delete adapters you do not have installed, and edit local paths. Three adapter types are supported:

- `browser`: runs the production Wav2mid UI in headless Chrome, uploads the audio through the real file input, waits for analysis, then downloads the real JSON export.
- `command`: executes an argv array without a shell. `{audio}`, `{output}`, `{id}`, and `{instruments}` placeholders are supported.
- `precomputed`: scores an existing MIDI/JSON result, useful for YouTabs, YourMT3+, or any externally generated output.

The example includes:

- Wav2mid PRO / INSANE / NEURAL;
- MuScriptor small and instrument-conditioned medium;
- focused Neural Semi-CRF adapters (`bass_v2`, `vocal`, `guitar_v1_5`, `drums`);
- precomputed YourMT3+ and YouTabs slots.

MuScriptor code is MIT, but its published model weights are CC BY-NC 4.0 and require Hugging Face license acceptance. It is therefore a benchmark/reference adapter and is not bundled into the Cloudflare/browser application. Check every third-party checkpoint license before redistribution or commercial use.

## Run a leaderboard

```bash
npm run benchmark:suite -- \
  --manifest bench/local-maestro-test.json \
  --adapters bench/adapters.local.json \
  --adapter wav2mid-insane,muscriptor-small \
  --split test
```

Results are written as both JSON and Markdown under `benchmark-results/`. Command-adapter outputs are cached under `.benchmark-cache/` unless `alwaysRun` is set.

For CI or a controlled experiment, `--min-objective 0.50` can enforce a minimum best-adapter objective. CI itself generates a tiny ground-truth WAV/MIDI fixture and runs the production-browser adapter end to end, so the leaderboard path cannot silently rot while the main app remains green.

## Auto-tune Wav2mid

Use a validation split only. Never tune on the held-out test split.

```bash
npm run tune -- \
  --manifest bench/local-maestro-validation.json \
  --split validation \
  --max-items 8 \
  --backend wasm \
  --modes pro,insane \
  --sensitivities 0.80,0.90,1.00,1.10,1.20
```

The tuner runs the actual production browser pipeline for every candidate and saves a ranked `benchmark-results/tuned-profile.json`. Evaluate the winner once on the untouched test split before changing defaults.

## What counts as "strongest"

A strongest claim needs at least:

1. a held-out real-recording benchmark (MAESTRO covers piano; add a licensed real multi-instrument set when available);
2. a held-out aligned multi-instrument benchmark such as Slakh2100;
3. the same metric implementation for Wav2mid and competitors;
4. no test-set tuning;
5. published adapter configs, manifest version, and benchmark JSON.

As of 2026-07, Kyutai/Mirelo report MuScriptor as the strongest open multi-instrument transcription model in their held-out real-recording comparison, so it is the primary open reference target rather than an older MT3-only baseline.
