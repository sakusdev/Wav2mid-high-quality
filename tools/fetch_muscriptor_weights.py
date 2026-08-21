#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download


SMALL_CONFIG = {
    'dim': 768,
    'num_heads': 12,
    'num_layers': 14,
    'card': 1393,
}


def main() -> None:
    out = Path('.muscriptor-source')
    out.mkdir(exist_ok=True)

    token = os.environ.get('HF_TOKEN') or None
    repo = 'MuScriptor/muscriptor-small' if token else 'cocktailpeanut/muscriptor-small'
    try:
        model = hf_hub_download(repo_id=repo, filename='model.safetensors', token=token)
    except Exception:
        if repo == 'cocktailpeanut/muscriptor-small':
            raise
        repo = 'cocktailpeanut/muscriptor-small'
        model = hf_hub_download(repo_id=repo, filename='model.safetensors')

    target = out / 'model.safetensors'
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(Path(model).resolve())

    # Current MuScriptor determines a local checkpoint architecture from a
    # sibling config.json. Without it, load_model() falls back to the large
    # 1536-dim / 48-layer model and a valid small checkpoint fails state_dict
    # loading. Keep mirrored small checkpoints self-describing.
    (out / 'config.json').write_text(json.dumps(SMALL_CONFIG, indent=2) + '\n')
    (out / 'SOURCE.txt').write_text(repo + '\n')
    print(f'weight source: {repo} ({target.stat().st_size} bytes)')
    print('weight config: muscriptor-small 768d / 12h / 14L / card1393')


if __name__ == '__main__':
    main()
