# NEURAL HQ low-memory streaming

NEURAL HQ no longer keeps a full-song 4-stem separation in memory.

Profiles are selected from browser hints:

- `MOBILE_SAFE`: 12 s chunks, 2 WASM threads
- `BALANCED`: 18 s chunks, 2–3 WASM threads
- `DESKTOP`: 24 s chunks, up to 4 WASM threads

Chunks overlap by 2–3 seconds. Only the center ownership region is kept, so note/drum events are not duplicated at boundaries. After each chunk, only MIDI-like events and small metadata remain referenced; separated stem arrays and inference temporaries can be reclaimed before the next chunk.

This stays on wasm32. Memory64/wasm64 is intentionally not required for this path.
