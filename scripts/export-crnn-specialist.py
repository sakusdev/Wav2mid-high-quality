#!/usr/bin/env python3
"""Export Kong/ByteDance-style high-resolution transcription CRNNs for Wav2mid.

The ONNX graph intentionally starts *after* torchlibrosa Spectrogram/LogmelFilterBank.
Wav2mid reproduces that frontend in JavaScript, keeping the ONNX graph to Conv/BN/GRU/
Linear/Sigmoid operators that map well to ONNX Runtime WebGPU.

Example:
  python scripts/export-crnn-specialist.py \
    --checkpoint /models/filobass_20000_iterations.pth \
    --output public/specialists/bass.onnx --instrument bass
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--instrument", default="piano")
    parser.add_argument("--classes", type=int, default=88)
    parser.add_argument("--begin-note", type=int, default=21)
    parser.add_argument("--frames-per-second", type=int, default=100)
    parser.add_argument("--onset-threshold", type=float, default=0.3)
    parser.add_argument("--offset-threshold", type=float, default=0.3)
    parser.add_argument("--frame-threshold", type=float, default=0.1)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--metadata-output")
    parser.add_argument("--source", default="local-checkpoint")
    parser.add_argument("--model-url", default=None,
                        help="URL to write into the generated metadata/manifest entry")
    parser.add_argument("--skip-parity", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import numpy as np
        import torch
        import torch.nn as nn
        from piano_transcription_inference.models import Regress_onset_offset_frame_velocity_CRNN
    except Exception as exc:
        print("Exporter dependencies are missing. Install torch, piano-transcription-inference, onnx and optionally onnxruntime.", file=sys.stderr)
        raise

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base = Regress_onset_offset_frame_velocity_CRNN(
        frames_per_second=args.frames_per_second,
        classes_num=args.classes,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    raw_state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    state = normalize_state_dict(raw_state)
    load_result = base.load_state_dict(state, strict=False)
    unexpected = [key for key in load_result.unexpected_keys if not key.startswith("pedal_model.")]
    missing_core = [key for key in load_result.missing_keys if not key.startswith(("spectrogram_extractor.", "logmel_extractor."))]
    if unexpected:
        print(f"warning: unexpected keys: {unexpected[:12]}", file=sys.stderr)
    if missing_core:
        print(f"warning: missing core keys: {missing_core[:12]}", file=sys.stderr)

    model = FrontendlessCrnn(base).eval()
    dummy = torch.randn(1, 1, 1001, 229, dtype=torch.float32)
    output_names = ["reg_onset_output", "reg_offset_output", "frame_output", "velocity_output"]
    dynamic_axes = {"logmel": {2: "frames"}}
    for name in output_names:
        dynamic_axes[name] = {1: "frames"}

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["logmel"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
    )

    parity = None
    if not args.skip_parity:
        try:
            import onnxruntime as ort
            with torch.no_grad():
                torch_outputs = model(dummy)
            session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
            ort_outputs = session.run(output_names, {"logmel": dummy.numpy()})
            errors = {}
            for name, torch_value, ort_value in zip(output_names, torch_outputs, ort_outputs):
                errors[name] = float(np.max(np.abs(torch_value.numpy() - ort_value)))
            parity = {"maxAbsoluteError": errors, "maximum": max(errors.values())}
            print(f"ONNX parity maximum absolute error: {parity['maximum']:.8g}")
        except ImportError:
            print("onnxruntime not installed; parity check skipped", file=sys.stderr)

    metadata_path = Path(args.metadata_output).expanduser().resolve() if args.metadata_output else output_path.with_suffix(".json")
    metadata = {
        "schema": "wav2mid-specialist/v1",
        "name": output_path.stem,
        "instrument": args.instrument,
        "url": args.model_url or output_path.name,
        "source": args.source,
        "checkpointSha256": sha256(checkpoint_path),
        "classes": args.classes,
        "beginNote": args.begin_note,
        "framesPerSecond": args.frames_per_second,
        "sampleRate": 16000,
        "frontend": {
            "nFft": 2048,
            "hopLength": 160,
            "melBins": 229,
            "fmin": 30,
            "fmax": 8000,
            "window": "hann-periodic",
            "center": True,
            "padMode": "reflect",
            "melScale": "slaney",
            "melNorm": "slaney",
            "topDb": None,
        },
        "onsetThreshold": args.onset_threshold,
        "offsetThreshold": args.offset_threshold,
        "frameThreshold": args.frame_threshold,
        "inputName": "logmel",
        "outputNames": {name: name for name in output_names},
        "onnxOpset": args.opset,
        "parity": parity,
        "promoted": False,
        "enabled": True,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")
    print(f"Wrote {metadata_path}")
    return 0


def normalize_state_dict(raw_state):
    state = {}
    if not isinstance(raw_state, dict):
        raise TypeError("Checkpoint state is not a dict")
    keys = list(raw_state.keys())
    has_note_prefix = any(strip_module_prefix(key).startswith("note_model.") for key in keys)
    for key, value in raw_state.items():
        key = strip_module_prefix(key)
        if has_note_prefix:
            if not key.startswith("note_model."):
                continue
            key = key[len("note_model."):]
        state[key] = value
    return state


def strip_module_prefix(key: str) -> str:
    while key.startswith("module."):
        key = key[len("module."):]
    return key


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Defined after dependency import in main on purpose: py_compile can validate the script
# without forcing PyTorch into normal Node/Cloudflare development environments.
def FrontendlessCrnn(base):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _FrontendlessCrnn(nn.Module):
        def __init__(self, source):
            super().__init__()
            self.bn0 = source.bn0
            self.frame_model = source.frame_model
            self.reg_onset_model = source.reg_onset_model
            self.reg_offset_model = source.reg_offset_model
            self.velocity_model = source.velocity_model
            self.reg_onset_gru = source.reg_onset_gru
            self.reg_onset_fc = source.reg_onset_fc
            self.frame_gru = source.frame_gru
            self.frame_fc = source.frame_fc

        def forward(self, logmel):
            x = logmel.transpose(1, 3)
            x = self.bn0(x)
            x = x.transpose(1, 3)

            frame_output = self.frame_model(x)
            reg_onset_output = self.reg_onset_model(x)
            reg_offset_output = self.reg_offset_model(x)
            velocity_output = self.velocity_model(x)

            x = torch.cat((reg_onset_output, (reg_onset_output ** 0.5) * velocity_output.detach()), dim=2)
            x, _ = self.reg_onset_gru(x)
            reg_onset_output = torch.sigmoid(self.reg_onset_fc(x))

            x = torch.cat((frame_output, reg_onset_output.detach(), reg_offset_output.detach()), dim=2)
            x, _ = self.frame_gru(x)
            frame_output = torch.sigmoid(self.frame_fc(x))
            return reg_onset_output, reg_offset_output, frame_output, velocity_output

    return _FrontendlessCrnn(base)


if __name__ == "__main__":
    raise SystemExit(main())
