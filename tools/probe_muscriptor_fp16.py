#!/usr/bin/env python3
"""Probe safe mixed-precision variants of the MuScriptor INT4 decoder.

The fully-float16 conversion of the streaming decoder breaks ONNX type
inference around the dynamic sinusoidal position path. Starting from the known
valid INT4/FP32 decoder, try increasingly conservative op block lists and run a
real one-step ORT execution for every candidate. The first valid candidate is
kept beside the baseline model and summarized in fp16-probe.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxconverter_common.float16 import convert_float_to_float16


CANDIDATES = [
    ("range", ["Range"]),
    ("trig", ["Range", "Cos", "Sin"]),
    ("position", ["Range", "Cos", "Sin", "Div"]),
    ("position-cast", ["Range", "Cos", "Sin", "Div", "Cast"]),
]


def type_to_dtype(type_string: str):
    if type_string == "tensor(float16)":
        return np.float16
    if type_string == "tensor(float)":
        return np.float32
    if type_string == "tensor(int64)":
        return np.int64
    raise RuntimeError(f"unsupported input type: {type_string}")


def run_step(session: ort.InferenceSession, arch: dict, prefix_len: int) -> dict:
    dim = int(arch["dim"])
    heads = int(arch["heads"])
    layers = int(arch["layers"])
    max_cache = int(arch["maxCache"])
    head_dim = dim // heads
    feeds = {}
    for meta in session.get_inputs():
        dtype = type_to_dtype(meta.type)
        if meta.name == "prefix_embeddings":
            feeds[meta.name] = np.zeros((1, prefix_len, dim), dtype=dtype)
        elif meta.name == "token_ids":
            feeds[meta.name] = np.asarray([[int(arch["card"])]], dtype=np.int64)
        elif meta.name == "past_len":
            feeds[meta.name] = np.asarray(0, dtype=np.int64)
        elif meta.name.startswith("cache_k_") or meta.name.startswith("cache_v_"):
            feeds[meta.name] = np.zeros((1, max_cache, heads, head_dim), dtype=dtype)
        else:
            raise RuntimeError(f"unexpected decoder input: {meta.name} {meta.type}")

    outputs = session.run(None, feeds)
    logits = outputs[0]
    if logits.shape != (1, int(arch["card"])) or not np.isfinite(logits).all():
        raise RuntimeError(f"bad logits: shape={logits.shape} finite={np.isfinite(logits).all()}")
    expected = (1, prefix_len + 1, heads, head_dim)
    for i in range(layers):
        k = outputs[1 + i * 2]
        v = outputs[2 + i * 2]
        if k.shape != expected or v.shape != expected:
            raise RuntimeError(f"layer {i} KV mismatch: {k.shape} {v.shape} != {expected}")
        if not np.isfinite(k).all() or not np.isfinite(v).all():
            raise RuntimeError(f"layer {i} non-finite KV")
    return {
        "argmax": int(np.argmax(logits[0])),
        "logitsType": str(logits.dtype),
        "cacheType": str(outputs[1].dtype),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()

    manifest_path = args.model_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    arch = manifest["architecture"]
    source = args.model_dir / manifest["files"]["decoder"]["name"]
    prefix_len = int(manifest.get("parity", {}).get("pytorch", {}).get("prefix_tokens", 503))

    report = {
        "source": source.name,
        "sourceBytes": source.stat().st_size,
        "candidates": [],
        "winner": None,
    }

    for label, blocked_ops in CANDIDATES:
        candidate = args.model_dir / f"muscriptor-small-decoder-int4-f16-{label}.onnx"
        entry = {"label": label, "blockedOps": blocked_ops, "file": candidate.name}
        try:
            model = onnx.load(str(source), load_external_data=True)
            converted = convert_float_to_float16(
                model,
                keep_io_types=False,
                disable_shape_infer=False,
                op_block_list=blocked_ops,
            )
            onnx.save(converted, str(candidate))
            entry["bytes"] = candidate.stat().st_size
            session = ort.InferenceSession(str(candidate), providers=["CPUExecutionProvider"])
            entry.update(run_step(session, arch, prefix_len))
            entry["status"] = "ok"
            report["winner"] = entry.copy()
            report["candidates"].append(entry)
            print(json.dumps(entry, indent=2))
            break
        except Exception as exc:  # probe intentionally records failures
            entry["status"] = "failed"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            report["candidates"].append(entry)
            print(json.dumps(entry, indent=2))
            candidate.unlink(missing_ok=True)

    (args.model_dir / "fp16-probe.json").write_text(json.dumps(report, indent=2) + "\n")
    if report["winner"] is None:
        print("No mixed-precision FP16 decoder candidate passed; keeping validated FP32 baseline.")
    else:
        winner = report["winner"]
        saved = report["sourceBytes"] - winner["bytes"]
        print(f"FP16 probe winner: {winner['label']} ({winner['bytes']} bytes, saved {saved} bytes)")


if __name__ == "__main__":
    main()
