# Reference MIDI benchmarking

Wav2mid HQ can be compared against a local reference MIDI without uploading the audio or the reference file anywhere.

```bash
node scripts/compare-reference.mjs \
  --reference /path/to/reference.mid \
  --analysis /path/to/transcription.analysis.json
```

The report searches a small global timing offset and reports:

- exact-pitch onset precision / recall / F1
- mean onset error for matched notes
- 10 ms note-occupancy (frame) precision / recall / F1, which penalizes duration errors
- drum-onset precision / recall / F1
- reference and analysis note counts, duration, tempo and key metadata

Useful options:

```bash
--onset-tolerance 0.20
--frame-step 0.01
--search-offset 0.35
```

This benchmark intentionally does not require committing reference audio or MIDI to the repository. Keep copyrighted or private evaluation material local.
