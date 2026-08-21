#!/usr/bin/env python3
"""MuScriptor-small browser exporter with a robust INT4/FP32 baseline.

The v2 exporter proved exact PyTorch/FP32-ONNX parity, but converting the whole
streaming decoder graph to float16 made ONNX Runtime reject a position/Cast edge
before session creation. Weight-only INT4 already supplies almost all of the
size reduction, so this compatibility layer keeps browser activations FP32 and
quantizes the original FP32 decoder MatMul weights directly.

The conditioner is intentionally FP32 as well. Mixing a float16 conditioner
output with the FP32 decoder creates an invalid browser hand-off unless an
explicit cast/copy is inserted, which would also defeat the zero-copy WebGPU
path. Establish one type-safe browser baseline first; mixed precision is probed
separately and can be promoted only after it passes ORT and Chrome WebGPU smoke.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import export_muscriptor_browser as base
from export_muscriptor_browser_v2 import StaticBrowserLayer


base.BrowserLayer = StaticBrowserLayer


def keep_fp32(src: Path, dst: Path) -> None:
    """Preserve FP32 activations for both browser graphs."""
    shutil.copyfile(src, dst)


base.convert_to_fp16 = keep_fp32


def output_dir_from_argv(argv: list[str]) -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    args, _ = parser.parse_known_args(argv)
    return args.output


def _rename_file_entry(output: Path, entry: dict, old_token: str, new_token: str) -> None:
    old_path = output / entry["name"]
    new_name = entry["name"].replace(old_token, new_token)
    new_path = output / new_name
    old_path.rename(new_path)
    entry["name"] = new_name
    entry["url"] = entry["url"].replace(old_token, new_token)
    # bytes and sha256 are unchanged by rename.


def rewrite_manifest(output: Path) -> None:
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    _rename_file_entry(output, manifest["files"]["conditioner"], "conditioner-f16", "conditioner-f32")
    _rename_file_entry(output, manifest["files"]["decoder"], "int4-f16", "int4-f32")

    arch = manifest["architecture"]
    arch.pop("activationType", None)
    arch["conditionerActivationType"] = "float32"
    arch["decoderActivationType"] = "float32"
    arch["decoderQuantization"] = "int4 MatMulNBits block128 symmetric; fp32 activations"
    manifest["exportNotes"] = [
        "conditioner and decoder activations intentionally remain float32 for browser type safety",
        "decoder weights use INT4 MatMulNBits weight-only quantization",
        "mixed precision is experimental until ORT and Chrome WebGPU smoke both pass",
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    output = output_dir_from_argv(sys.argv[1:])
    base.main()
    rewrite_manifest(output)


if __name__ == "__main__":
    main()
