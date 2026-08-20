#!/usr/bin/env python3
"""Export MuScriptor-small to browser-friendly ONNX graphs.

The browser runtime keeps the autoregressive loop in JavaScript and the model
math in ONNX Runtime Web/WebGPU. A tiny conditioner graph projects the 512-bin
log-mel frames and class ids. A single decoder graph is used for both prefill
and one-token decode; it receives persistent KV-cache tensors and returns only
the K/V rows produced by the current input so the browser can append them to
preallocated GPU buffers without copying the whole cache every token.

The source MuScriptor code is MIT. Published MuScriptor weights are CC BY-NC
4.0. This exporter does not bundle or download weights by itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx import TensorProto, helper
from onnxconverter_common import float16
from onnxruntime.quantization.matmul_nbits_quantizer import (
    DefaultWeightOnlyQuantConfig,
    MatMulNBitsQuantizer,
)
from onnxruntime.quantization.quant_utils import QuantFormat

from muscriptor import TranscriptionModel
from muscriptor.modules.streaming import increment_steps, init_states


DEFAULT_MAX_CACHE = 3072
OPSET = 18


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class BrowserConditioner(nn.Module):
    """Equivalent of MuScriptor's three conditioning streams.

    Input log_mel is already log(amplitude_mel + 1e-6), shape [B,T,512].
    instrument_embed_ids are *embedding table indices*: 1 is the null class;
    a MT3_FULL_PLUS group id g is g + 2, exactly matching the upstream
    ClassConditioner tokenize()+forward() pair.
    """

    def __init__(self, lm: nn.Module):
        super().__init__()
        provider = lm.condition_provider.conditioners
        mel = provider["self_wav"]
        inst = provider["instrument_group"]
        dataset = provider["dataset_name"]

        self.register_buffer("mel_w", mel.output_proj.weight.detach().t().contiguous())
        self.register_buffer("mel_b", mel.output_proj.bias.detach().contiguous())
        self.register_buffer("inst_w", inst.embed.weight.detach().contiguous())
        self.register_buffer("dataset_w", dataset.embed.weight.detach().contiguous())

    def forward(self, log_mel: torch.Tensor, instrument_embed_ids: torch.Tensor):
        mel = torch.matmul(log_mel, self.mel_w) + self.mel_b
        # For a 5 s / 16 kHz chunk the upstream length mask is 500 while
        # centered STFT emits 501 frames, so the final projected frame is zero.
        mel = torch.cat([mel[:, :-1], torch.zeros_like(mel[:, -1:])], dim=1)
        instrument = F.embedding(instrument_embed_ids, self.inst_w)
        dataset_ids = torch.ones(
            (log_mel.shape[0], 1), dtype=torch.long, device=log_mel.device
        )
        dataset = F.embedding(dataset_ids, self.dataset_w)
        # LMModel.forward prepends each dict value in insertion order
        # instrument -> dataset -> self_wav, producing wav,dataset,instrument.
        return torch.cat([mel, dataset, instrument], dim=1)


class BrowserLayer(nn.Module):
    def __init__(self, layer: nn.Module, heads: int):
        super().__init__()
        dim = layer.norm1.weight.numel()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.register_buffer("norm1_w", layer.norm1.weight.detach().contiguous())
        self.register_buffer("norm1_b", layer.norm1.bias.detach().contiguous())
        self.register_buffer(
            "qkv_w", layer.self_attn.in_proj_weight.detach().t().contiguous()
        )
        self.register_buffer(
            "attn_out_w", layer.self_attn.out_proj.weight.detach().t().contiguous()
        )
        self.register_buffer("norm2_w", layer.norm2.weight.detach().contiguous())
        self.register_buffer("norm2_b", layer.norm2.bias.detach().contiguous())
        self.register_buffer("ff1_w", layer.linear1.weight.detach().t().contiguous())
        self.register_buffer("ff2_w", layer.linear2.weight.detach().t().contiguous())

    def forward(
        self,
        x: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        past_len: torch.Tensor,
    ):
        b, q_len, dim = x.shape
        h = F.layer_norm(x, (dim,), self.norm1_w, self.norm1_b, 1e-5)
        packed = torch.matmul(h, self.qkv_w).reshape(
            b, q_len, 3, self.heads, self.head_dim
        )
        q = packed[:, :, 0]
        k_new = packed[:, :, 1]
        v_new = packed[:, :, 2]

        # cache layout is [B, MAX_SEQ, H, Dh], so appending Q rows is a single
        # contiguous GPU copy per K/V tensor in the browser.
        p = past_len.reshape(()).to(torch.long)
        past_k = cache_k[:, :p]
        past_v = cache_v[:, :p]
        k_all = torch.cat([past_k, k_new], dim=1)
        v_all = torch.cat([past_v, v_new], dim=1)

        qt = q.permute(0, 2, 1, 3)
        kt = k_all.permute(0, 2, 1, 3)
        vt = v_all.permute(0, 2, 1, 3)
        scores = torch.matmul(qt, kt.transpose(-1, -2)) * self.scale

        # Bottom-right causal mask. For decode q_len=1 and all cached keys are
        # visible; for prefill this is the ordinary square causal mask.
        q_pos = torch.arange(q_len, device=x.device, dtype=torch.long) + p
        k_pos = torch.arange(k_all.shape[1], device=x.device, dtype=torch.long)
        allow = k_pos.reshape(1, 1, 1, -1) <= q_pos.reshape(1, 1, -1, 1)
        scores = torch.where(allow, scores, torch.full_like(scores, -1.0e4))
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, vt).permute(0, 2, 1, 3).reshape(b, q_len, dim)
        x = x + torch.matmul(attended, self.attn_out_w)

        h2 = F.layer_norm(x, (dim,), self.norm2_w, self.norm2_b, 1e-5)
        x = x + torch.matmul(F.gelu(torch.matmul(h2, self.ff1_w)), self.ff2_w)
        return x, k_new, v_new


class BrowserDecoder(nn.Module):
    def __init__(self, lm: nn.Module):
        super().__init__()
        self.dim = int(lm.dim)
        self.card = int(lm.card)
        self.max_period = float(lm.transformer.max_period)
        self.heads = int(lm.transformer.layers[0].self_attn.num_heads)

        self.register_buffer("token_w", lm.emb.weight.detach().contiguous())
        self.layers = nn.ModuleList(
            [BrowserLayer(layer, self.heads) for layer in lm.transformer.layers]
        )
        self.register_buffer("out_norm_w", lm.out_norm.weight.detach().contiguous())
        self.register_buffer("out_norm_b", lm.out_norm.bias.detach().contiguous())
        self.register_buffer("lm_w", lm.linear.weight.detach().t().contiguous())

    def _positions(self, x: torch.Tensor, past_len: torch.Tensor) -> torch.Tensor:
        q_len = x.shape[1]
        p = past_len.reshape(()).to(torch.long)
        positions = torch.arange(q_len, device=x.device, dtype=torch.float32) + p.float()
        half = self.dim // 2
        adim = torch.arange(half, device=x.device, dtype=torch.float32)
        denom = torch.pow(
            torch.tensor(self.max_period, dtype=torch.float32, device=x.device),
            adim / float(half - 1),
        )
        phase = positions.reshape(1, -1, 1) / denom.reshape(1, 1, -1)
        return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1).to(x.dtype)

    def forward(self, prefix_embeddings, token_ids, past_len, *caches):
        token = F.embedding(token_ids.clamp(min=0), self.token_w)
        x = torch.cat([prefix_embeddings, token], dim=1)
        x = x + self._positions(x, past_len)

        new_kv: list[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            cache_k = caches[i * 2]
            cache_v = caches[i * 2 + 1]
            x, k_new, v_new = layer(x, cache_k, cache_v, past_len)
            new_kv.extend([k_new, v_new])

        x = F.layer_norm(x, (self.dim,), self.out_norm_w, self.out_norm_b, 1e-5)
        logits = torch.matmul(x[:, -1], self.lm_w)
        return (logits, *new_kv)


def official_prefix(lm: nn.Module, cfg_conditions: dict) -> torch.Tensor:
    ordered = None
    for cond, _ in cfg_conditions.values():
        ordered = cond if ordered is None else torch.cat([cond, ordered], dim=1)
    assert ordered is not None
    return ordered


def make_conditions(transcriber: TranscriptionModel, waveform: torch.Tensor):
    attrs = transcriber._build_conditions(waveform, None)
    lm = transcriber._model
    prepared = lm.condition_provider.tokenize(attrs)
    cfg = lm.condition_provider(prepared)
    log_mel = lm.condition_provider.conditioners["self_wav"]._mel_embedding(
        prepared["self_wav"]
    )
    # null instrument maps to embedding index 1 in upstream tokenize+forward.
    instrument_embed_ids = torch.ones((1, 1), dtype=torch.long)
    return cfg, log_mel, instrument_embed_ids


def parity_pytorch(transcriber: TranscriptionModel, max_cache: int):
    torch.manual_seed(0)
    lm = transcriber._model.eval()
    t = torch.arange(80_000, dtype=torch.float32) / 16_000.0
    waveform = (
        0.17 * torch.sin(2 * math.pi * 220.0 * t)
        + 0.09 * torch.sin(2 * math.pi * 329.6276 * t)
    ).reshape(1, -1)
    cfg, log_mel, inst_ids = make_conditions(transcriber, waveform)

    conditioner = BrowserConditioner(lm).eval()
    prefix = conditioner(log_mel, inst_ids)
    ref_prefix = official_prefix(lm, cfg)
    prefix_err = (prefix - ref_prefix).abs().max().item()
    if prefix_err > 2e-5:
        raise RuntimeError(f"conditioner parity failed: max abs {prefix_err}")

    decoder = BrowserDecoder(lm).eval()
    head_dim = lm.dim // lm.transformer.layers[0].self_attn.num_heads
    heads = lm.transformer.layers[0].self_attn.num_heads
    empty = [
        torch.zeros((1, max_cache, heads, head_dim), dtype=torch.float32)
        for _ in range(len(lm.transformer.layers) * 2)
    ]
    token = torch.tensor([[lm.initial_token_id]], dtype=torch.long)
    p0 = torch.tensor(0, dtype=torch.long)
    wrapped = decoder(prefix, token, p0, *empty)

    state = init_states(lm, batch_size=1, sequence_length=max_cache)
    ref_logits = lm(token, cfg, first_step=True, model_state=state)[:, -1]
    logit_err = (wrapped[0] - ref_logits).abs().max().item()
    if logit_err > 2e-4:
        raise RuntimeError(f"prefill logit parity failed: max abs {logit_err}")

    q_len = prefix.shape[1] + token.shape[1]
    cache_tensors = []
    for i in range(len(lm.transformer.layers)):
        cache = state[f"transformer.layers.{i}.self_attn"]["cache"]
        ref_k = cache[0, :, :q_len]
        ref_v = cache[1, :, :q_len]
        got_k = wrapped[1 + i * 2]
        got_v = wrapped[2 + i * 2]
        if (got_k - ref_k).abs().max().item() > 3e-4:
            raise RuntimeError(f"layer {i} prefill K parity failed")
        if (got_v - ref_v).abs().max().item() > 3e-4:
            raise RuntimeError(f"layer {i} prefill V parity failed")
        kfull = torch.zeros_like(empty[0])
        vfull = torch.zeros_like(empty[0])
        kfull[:, :q_len] = got_k
        vfull[:, :q_len] = got_v
        cache_tensors.extend([kfull, vfull])

    increment_steps(lm.transformer, state, increment=q_len)
    next_token = wrapped[0].argmax(dim=-1).reshape(1, 1)
    wrapped2 = decoder(prefix[:, :0], next_token, torch.tensor(q_len), *cache_tensors)
    ref_logits2 = lm(next_token, cfg, first_step=False, model_state=state)[:, -1]
    logit2_err = (wrapped2[0] - ref_logits2).abs().max().item()
    if logit2_err > 3e-4:
        raise RuntimeError(f"decode logit parity failed: max abs {logit2_err}")

    return {
        "conditioner_max_abs": prefix_err,
        "prefill_logits_max_abs": logit_err,
        "decode_logits_max_abs": logit2_err,
        "first_token": int(next_token.item()),
        "prefix_tokens": int(prefix.shape[1]),
    }


def export_fp32(
    transcriber: TranscriptionModel,
    out_dir: Path,
    max_cache: int,
):
    lm = transcriber._model.eval()
    conditioner = BrowserConditioner(lm).eval()
    decoder = BrowserDecoder(lm).eval()
    heads = lm.transformer.layers[0].self_attn.num_heads
    head_dim = lm.dim // heads

    cond_path = out_dir / "muscriptor-small-conditioner-fp32.onnx"
    decoder_path = out_dir / "muscriptor-small-decoder-fp32.onnx"

    dummy_mel = torch.zeros((1, 501, 512), dtype=torch.float32)
    dummy_inst = torch.ones((1, 1), dtype=torch.long)
    torch.onnx.export(
        conditioner,
        (dummy_mel, dummy_inst),
        cond_path,
        opset_version=OPSET,
        input_names=["log_mel", "instrument_embed_ids"],
        output_names=["prefix_embeddings"],
        dynamic_axes={
            "instrument_embed_ids": {1: "instrument_count"},
            "prefix_embeddings": {1: "prefix_len"},
        },
        do_constant_folding=True,
        dynamo=False,
    )

    prefix = conditioner(dummy_mel, dummy_inst)
    token = torch.tensor([[lm.initial_token_id]], dtype=torch.long)
    past_len = torch.tensor(0, dtype=torch.long)
    caches = [
        torch.zeros((1, max_cache, heads, head_dim), dtype=torch.float32)
        for _ in range(len(lm.transformer.layers) * 2)
    ]
    cache_names = []
    new_names = []
    dynamic = {
        "prefix_embeddings": {1: "prefix_len"},
        "token_ids": {1: "token_len"},
    }
    for i in range(len(lm.transformer.layers)):
        cache_names += [f"cache_k_{i}", f"cache_v_{i}"]
        new_names += [f"new_k_{i}", f"new_v_{i}"]
        dynamic[f"new_k_{i}"] = {1: "query_len"}
        dynamic[f"new_v_{i}"] = {1: "query_len"}

    torch.onnx.export(
        decoder,
        (prefix, token, past_len, *caches),
        decoder_path,
        opset_version=OPSET,
        input_names=["prefix_embeddings", "token_ids", "past_len", *cache_names],
        output_names=["logits", *new_names],
        dynamic_axes=dynamic,
        do_constant_folding=True,
        dynamo=False,
    )
    return cond_path, decoder_path


def ort_parity_fp32(
    transcriber: TranscriptionModel,
    cond_path: Path,
    decoder_path: Path,
    max_cache: int,
):
    lm = transcriber._model.eval()
    heads = lm.transformer.layers[0].self_attn.num_heads
    head_dim = lm.dim // heads
    t = torch.arange(80_000, dtype=torch.float32) / 16_000.0
    waveform = (0.17 * torch.sin(2 * math.pi * 220.0 * t)).reshape(1, -1)
    cfg, log_mel, inst_ids = make_conditions(transcriber, waveform)
    ref_prefix = official_prefix(lm, cfg).detach().numpy()

    cond_sess = ort.InferenceSession(str(cond_path), providers=["CPUExecutionProvider"])
    prefix = cond_sess.run(
        None,
        {
            "log_mel": log_mel.detach().numpy(),
            "instrument_embed_ids": inst_ids.numpy().astype(np.int64),
        },
    )[0]
    cond_err = float(np.max(np.abs(prefix - ref_prefix)))
    if cond_err > 3e-5:
        raise RuntimeError(f"ONNX conditioner parity failed: {cond_err}")

    cache = np.zeros((1, max_cache, heads, head_dim), dtype=np.float32)
    feeds = {
        "prefix_embeddings": prefix,
        "token_ids": np.asarray([[lm.initial_token_id]], dtype=np.int64),
        "past_len": np.asarray(0, dtype=np.int64),
    }
    for i in range(len(lm.transformer.layers)):
        feeds[f"cache_k_{i}"] = cache
        feeds[f"cache_v_{i}"] = cache
    dec_sess = ort.InferenceSession(str(decoder_path), providers=["CPUExecutionProvider"])
    out = dec_sess.run(None, feeds)

    state = init_states(lm, batch_size=1, sequence_length=max_cache)
    token = torch.tensor([[lm.initial_token_id]], dtype=torch.long)
    ref_logits = lm(token, cfg, first_step=True, model_state=state)[:, -1].detach().numpy()
    dec_err = float(np.max(np.abs(out[0] - ref_logits)))
    if dec_err > 5e-4:
        raise RuntimeError(f"ONNX decoder parity failed: {dec_err}")
    return {"conditioner_max_abs": cond_err, "decoder_max_abs": dec_err}


def convert_to_fp16(src: Path, dst: Path):
    model = onnx.load(str(src))
    model = float16.convert_float_to_float16(
        model,
        keep_io_types=False,
        disable_shape_infer=False,
    )
    onnx.save(model, str(dst))


def force_logits_float32(path: Path):
    model = onnx.load(str(path))
    output = next(o for o in model.graph.output if o.name == "logits")
    original_name = "logits_f16_internal"
    # Rename the graph producer output only; consumers do not exist because
    # logits is a terminal output.
    for node in model.graph.node:
        for i, name in enumerate(node.output):
            if name == "logits":
                node.output[i] = original_name
    cast = helper.make_node(
        "Cast", [original_name], ["logits"], name="logits_to_fp32", to=TensorProto.FLOAT
    )
    model.graph.node.append(cast)
    output.type.tensor_type.elem_type = TensorProto.FLOAT
    onnx.save(model, str(path))


def quantize_int4(src: Path, dst: Path):
    model = onnx.load(str(src))
    config = DefaultWeightOnlyQuantConfig(
        block_size=128,
        is_symmetric=True,
        accuracy_level=4,
        quant_format=QuantFormat.QOperator,
        op_types_to_quantize=("MatMul",),
        quant_axes=(("MatMul", 0),),
        bits=4,
    )
    quant = MatMulNBitsQuantizer(model, algo_config=config)
    quant.process()
    quant.model.save_model_to_file(str(dst), use_external_data_format=False)


def check_model(path: Path):
    model = onnx.load(str(path))
    onnx.checker.check_model(model)


def file_info(path: Path):
    return {
        "name": path.name,
        "bytes": path.stat().st_size,
        "mib": round(path.stat().st_size / (1024 * 1024), 3),
        "sha256": sha256(path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-cache", type=int, default=DEFAULT_MAX_CACHE)
    ap.add_argument("--source", default="MuScriptor/muscriptor-small")
    ap.add_argument("--release-base", default="/models/muscriptor-small")
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    if not args.weights.exists():
        raise SystemExit(f"missing weights: {args.weights}")

    print("Loading MuScriptor small...")
    transcriber = TranscriptionModel.load_model(
        str(args.weights), device="cpu", dtype="float32"
    )
    lm = transcriber._model
    if int(lm.dim) != 768 or len(lm.transformer.layers) != 14:
        raise RuntimeError("exporter currently targets muscriptor-small only")

    pt_parity = parity_pytorch(transcriber, args.max_cache)
    print("PyTorch parity:", json.dumps(pt_parity, indent=2))

    cond_fp32, dec_fp32 = export_fp32(transcriber, args.output, args.max_cache)
    ort_parity = ort_parity_fp32(
        transcriber, cond_fp32, dec_fp32, args.max_cache
    )
    print("ORT FP32 parity:", json.dumps(ort_parity, indent=2))

    cond_f16 = args.output / "muscriptor-small-conditioner-f16.onnx"
    dec_f16 = args.output / "muscriptor-small-decoder-f16.onnx"
    dec_int4 = args.output / "muscriptor-small-decoder-int4-f16.onnx"
    convert_to_fp16(cond_fp32, cond_f16)
    convert_to_fp16(dec_fp32, dec_f16)
    force_logits_float32(dec_f16)
    quantize_int4(dec_f16, dec_int4)
    for p in (cond_f16, dec_int4):
        check_model(p)

    manifest = {
        "format": "wav2mid-muscriptor-browser/v1",
        "engine": "MuScriptor small",
        "modelLicense": "CC BY-NC 4.0",
        "codeLicense": "MIT",
        "source": args.source,
        "architecture": {
            "dim": int(lm.dim),
            "heads": int(lm.transformer.layers[0].self_attn.num_heads),
            "layers": len(lm.transformer.layers),
            "card": int(lm.card),
            "sampleRate": 16000,
            "segmentSeconds": 5,
            "melBins": 512,
            "frameRate": 100,
            "maxCache": args.max_cache,
            "weightQuantization": "int4 MatMulNBits block128 symmetric",
            "activationType": "float16",
        },
        "files": {
            "conditioner": {
                **file_info(cond_f16),
                "url": f"{args.release_base}/{cond_f16.name}",
            },
            "decoder": {
                **file_info(dec_int4),
                "url": f"{args.release_base}/{dec_int4.name}",
            },
        },
        "parity": {"pytorch": pt_parity, "onnxFp32": ort_parity},
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    # Do not publish intermediate full-precision/fp16 decoder copies.
    for p in (cond_fp32, dec_fp32, dec_f16):
        p.unlink(missing_ok=True)

    print(json.dumps({k: file_info(v) for k, v in {
        "conditioner": cond_f16,
        "decoder": dec_int4,
        "manifest": manifest_path,
    }.items()}, indent=2))


if __name__ == "__main__":
    main()
