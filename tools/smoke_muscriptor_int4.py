#!/usr/bin/env python3
"""Execute the exported MuScriptor browser pipeline with ONNX Runtime CPU.

PyTorch/FP32-ONNX numeric parity is checked by the exporter. This smoke test
loads the final browser conditioner and INT4 decoder, runs the real
conditioner->decoder hand-off, validates shapes/dtypes/finiteness, and records a
deterministic zero-input first-token fixture for the Chrome WebGPU smoke test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def numpy_dtype(name: str):
    if name == "float16":
        return np.float16
    if name == "float32":
        return np.float32
    raise RuntimeError(f"unsupported activation type: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()

    manifest_path = args.model_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    arch = manifest["architecture"]
    conditioner_path = args.model_dir / manifest["files"]["conditioner"]["name"]
    decoder_path = args.model_dir / manifest["files"]["decoder"]["name"]

    conditioner_activation = arch.get(
        "conditionerActivationType", arch.get("activationType", "float16")
    )
    decoder_activation = arch.get(
        "decoderActivationType", arch.get("activationType", "float16")
    )
    decoder_dtype = numpy_dtype(decoder_activation)

    cond_sess = ort.InferenceSession(
        str(conditioner_path), providers=["CPUExecutionProvider"]
    )
    log_mel = np.zeros((1, 501, int(arch["melBins"])), dtype=np.float32)
    instrument_ids = np.asarray([[1]], dtype=np.int64)
    prefix = cond_sess.run(
        None,
        {"log_mel": log_mel, "instrument_embed_ids": instrument_ids},
    )[0]

    dim = int(arch["dim"])
    prefix_len = int(manifest.get("parity", {}).get("pytorch", {}).get("prefix_tokens", 503))
    expected_prefix_shape = (1, prefix_len, dim)
    if prefix.shape != expected_prefix_shape:
        raise RuntimeError(
            f"unexpected conditioner prefix shape: {prefix.shape} expected {expected_prefix_shape}"
        )
    if not np.isfinite(prefix).all():
        raise RuntimeError("conditioner produced non-finite prefix embeddings")
    if prefix.dtype != decoder_dtype:
        raise RuntimeError(
            "conditioner/decoder activation mismatch: "
            f"conditioner output {prefix.dtype} vs decoder {decoder_dtype}"
        )

    decoder = ort.InferenceSession(
        str(decoder_path), providers=["CPUExecutionProvider"]
    )

    heads = int(arch["heads"])
    layers = int(arch["layers"])
    max_cache = int(arch["maxCache"])
    head_dim = dim // heads
    feeds: dict[str, np.ndarray] = {
        "prefix_embeddings": prefix,
        "token_ids": np.asarray([[int(arch["card"])]], dtype=np.int64),
        "past_len": np.asarray(0, dtype=np.int64),
    }
    cache = np.zeros((1, max_cache, heads, head_dim), dtype=decoder_dtype)
    for i in range(layers):
        feeds[f"cache_k_{i}"] = cache
        feeds[f"cache_v_{i}"] = cache

    outputs = decoder.run(None, feeds)
    logits = outputs[0]
    if logits.shape != (1, int(arch["card"])):
        raise RuntimeError(f"unexpected logits shape: {logits.shape}")
    if not np.isfinite(logits).all():
        raise RuntimeError("INT4 decoder produced non-finite logits")

    expected_query = prefix_len + 1
    for i in range(layers):
        k = outputs[1 + i * 2]
        v = outputs[2 + i * 2]
        expected = (1, expected_query, heads, head_dim)
        if k.shape != expected or v.shape != expected:
            raise RuntimeError(
                f"layer {i} KV shape mismatch: k={k.shape} v={v.shape} expected={expected}"
            )
        if not np.isfinite(k).all() or not np.isfinite(v).all():
            raise RuntimeError(f"layer {i} produced non-finite KV")

    first_token = int(np.argmax(logits[0]))
    manifest["browserSmoke"] = {
        "fixture": "zero-log-mel-null-instrument",
        "logMelShape": [1, 501, int(arch["melBins"])],
        "instrumentEmbedIds": [1],
        "prefixTokens": prefix_len,
        "conditionerActivationType": conditioner_activation,
        "decoderActivationType": decoder_activation,
        "expectedFirstToken": first_token,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(json.dumps({
        "provider": decoder.get_providers()[0],
        "conditioner": conditioner_path.name,
        "decoder": decoder_path.name,
        "conditioner_activation": conditioner_activation,
        "decoder_activation": decoder_activation,
        "prefix_shape": list(prefix.shape),
        "logits_shape": list(logits.shape),
        "argmax": first_token,
        "query_tokens": expected_query,
        "layers": layers,
        "status": "ok",
    }, indent=2))


if __name__ == "__main__":
    main()
