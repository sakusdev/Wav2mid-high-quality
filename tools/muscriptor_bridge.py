#!/usr/bin/env python3
"""Run MuScriptor as a localhost bridge for the static Wav2mid HQ UI.

The upstream MuScriptor server intentionally does not enable cross-origin
requests. Wav2mid HQ is a static HTTPS site, so this wrapper adds permissive
CORS only to a loopback-bound server. The audio never leaves the user's PC.

MuScriptor code is MIT. Published model weights are CC BY-NC 4.0 and are not
bundled by this repository; MuScriptor downloads them after the user accepts
the Hugging Face license and authenticates locally.
"""

from __future__ import annotations

import argparse

import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from muscriptor import TranscriptionModel
from muscriptor.server import create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local MuScriptor bridge for Wav2mid HQ")
    parser.add_argument("--model", default="small", choices=("small", "medium", "large"))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, mps, ...")
    parser.add_argument("--dtype", default=None, choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8223)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("Refusing a non-loopback host. Use the upstream server if remote access is intended.")

    device = None if args.device == "auto" else args.device
    print(f"Loading MuScriptor {args.model} on {args.device} ...", flush=True)
    model = TranscriptionModel.load_model(
        weights_path=args.model,
        device=device,
        dtype=args.dtype,
    )
    app = create_app(model)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["Content-Disposition"],
    )
    print(f"MuScriptor bridge ready: http://127.0.0.1:{args.port}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
