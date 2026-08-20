# MuScriptor ULTRA (NC)

Wav2mid HQ can use MuScriptor as an optional non-commercial desktop transcription engine through a localhost bridge.

This mode is intentionally separate from the browser-only FAST / PRO / INSANE / NEURAL HQ paths:

- MuScriptor code: MIT
- MuScriptor published model weights: CC BY-NC 4.0
- model weights are **not** bundled in this repository
- audio is sent only from the browser to the loopback bridge (`127.0.0.1`)
- the public site does not contact the bridge until ULTRA is enabled and analysis/check is explicitly started

## Setup

1. Install MuScriptor and authenticate with Hugging Face after accepting the model license.

```bash
pip install muscriptor
hf auth login
```

2. From this repository, start the bridge.

```bash
python tools/muscriptor_bridge.py --model small
```

The default endpoint is:

```text
http://127.0.0.1:8223
```

3. Open Wav2mid HQ, enable **MuScriptor ULTRA NC**, press **CHECK**, then analyze an audio file.

The bridge accepts `small`, `medium`, or `large`:

```bash
python tools/muscriptor_bridge.py --model medium --device cuda
```

For CPU-only machines, `small` is the practical starting point.

## Why a bridge?

The upstream MuScriptor web client uses a FastAPI/PyTorch server. The released model is an autoregressive transformer decoder, so it is not currently a drop-in ONNX Runtime Web model. Wav2mid therefore uses the official streaming `/transcribe` API while a browser-native WebGPU port is developed separately.

The wrapper adds CORS to a **loopback-only** server. It refuses non-loopback bind addresses so the permissive CORS configuration does not accidentally expose MuScriptor to the LAN.

## Browser notes

Chrome and Firefox can access loopback HTTP from an HTTPS page. Safari/WebKit may block this pattern; use Chrome/Firefox for the localhost bridge mode.
