#!/usr/bin/env python3
"""Stage exported MuScriptor browser artifacts for Cloudflare Static Assets.

The INT4 decoder is ~55 MiB, above Workers Static Assets' per-file limit and
also above this project's stricter 5 MiB temporary-preview gate. Keep the
conditioner as a normal asset, but split oversized model files into <=4.5 MiB
pieces. A tiny Worker (`worker.js`) reassembles those pieces as a streaming HTTP
response at the original ONNX URL, so ONNX Runtime Web does not need a custom
loader and never sees the chunking scheme.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--chunk-mib", type=float, default=4.5)
    args = ap.parse_args()

    manifest_path = args.model_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chunk_bytes = int(args.chunk_mib * 1024 * 1024)
    if chunk_bytes <= 0:
        raise SystemExit("chunk size must be positive")

    staged = {"format": "wav2mid-static-model-parts/v1", "models": {}}

    for logical_name, entry in manifest["files"].items():
        src = args.model_dir / entry["name"]
        if not src.is_file():
            raise SystemExit(f"missing exported model file: {src}")

        url = entry["url"]
        size = src.stat().st_size
        if size <= chunk_bytes:
            shutil.copyfile(src, args.output_dir / src.name)
            continue

        parts = []
        with src.open("rb") as f:
            index = 0
            while True:
                data = f.read(chunk_bytes)
                if not data:
                    break
                part_name = f"{src.name}.part-{index:03d}.bin"
                part_path = args.output_dir / part_name
                part_path.write_bytes(data)
                parts.append({
                    "url": f"/models/muscriptor-small/{part_name}",
                    "bytes": len(data),
                    "sha256": sha256(part_path),
                })
                index += 1

        part_manifest_name = f"{src.name}.parts.json"
        part_manifest = {
            "format": "wav2mid-streamed-asset/v1",
            "target": url,
            "bytes": size,
            "sha256": entry.get("sha256"),
            "contentType": "application/octet-stream",
            "parts": parts,
        }
        (args.output_dir / part_manifest_name).write_text(
            json.dumps(part_manifest, indent=2) + "\n"
        )
        staged["models"][url] = f"/models/muscriptor-small/{part_manifest_name}"

    # Keep the original browser manifest untouched: its decoder URL remains a
    # normal .onnx URL. worker.js intercepts that URL and streams the parts.
    shutil.copyfile(manifest_path, args.output_dir / "manifest.json")
    (args.output_dir / "stream-map.json").write_text(json.dumps(staged, indent=2) + "\n")

    largest = max((p.stat().st_size for p in args.output_dir.iterdir() if p.is_file()), default=0)
    if largest > 5 * 1024 * 1024:
        raise SystemExit(f"staged model asset exceeds 5 MiB: {largest} bytes")

    print(json.dumps({
        "output": str(args.output_dir),
        "chunk_mib": args.chunk_mib,
        "largest_bytes": largest,
        "streamed_models": list(staged["models"]),
    }, indent=2))


if __name__ == "__main__":
    main()
