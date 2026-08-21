#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import hf_hub_download


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
    (out / 'SOURCE.txt').write_text(repo + '\n')
    print(f'weight source: {repo} ({target.stat().st_size} bytes)')


if __name__ == '__main__':
    main()
