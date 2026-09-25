#!/usr/bin/env python
"""Prepare a local safetensors copy of ProsusAI/finbert for the sentiment backend.

v-playbook-desk-b1-2026-09-25. The hub copy of ProsusAI/finbert ships only
`pytorch_model.bin` (a pickle checkpoint). transformers >= 5 refuses pickle
checkpoints on torch < 2.6 (CVE-2025-32434), so `core.sentiment_backend`
loads from a local directory holding `model.safetensors` instead.

This script builds that directory (default `models/finbert`, git-ignored;
override with `--dest` or env `SENTIMENT_FINBERT_PATH`):

  1. locates the Hugging Face cache snapshot for the model (downloading the
     config, tokenizer and weights if they are not cached yet),
  2. copies config.json and the tokenizer files,
  3. writes model.safetensors, either by reusing a safetensors blob already
     in the cache or by converting the pickle checkpoint with
     `torch.load(weights_only=True)` (the safe loader), and
  4. verifies every tensor in the safetensors file equals the pickle
     checkpoint before reporting success.

Usage:
    python scripts/prepare_finbert.py [--dest models/finbert] [--force]
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

MODEL_ID = "ProsusAI/finbert"
CONFIG_FILES = ("config.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt")
WEIGHTS_BIN = "pytorch_model.bin"
WEIGHTS_ST = "model.safetensors"


def _default_dest() -> str:
    return os.getenv("SENTIMENT_FINBERT_PATH") or "models/finbert"


def _cached_snapshot(model_id: str) -> Path:
    """Return the local snapshot dir for `model_id`, downloading if needed."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        model_id,
        allow_patterns=list(CONFIG_FILES) + [WEIGHTS_BIN],
    )
    return Path(path)


def _find_cached_safetensors(snapshot: Path) -> Path | None:
    """Look for a model.safetensors already present in any snapshot of the
    same repo (older sessions may have converted it)."""
    repo_root = snapshot.parent.parent  # <cache>/models--ProsusAI--finbert
    snapshots_dir = repo_root / "snapshots"
    if not snapshots_dir.is_dir():
        return None
    for candidate in sorted(snapshots_dir.glob(f"*/{WEIGHTS_ST}")):
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def _load_bin_state_dict(bin_path: Path) -> dict:
    import torch

    return torch.load(str(bin_path), map_location="cpu", weights_only=True)


def _write_safetensors(bin_path: Path, dest: Path) -> None:
    from safetensors.torch import save_file

    state = _load_bin_state_dict(bin_path)
    # save_file requires contiguous tensors and rejects shared storage.
    state = {k: v.contiguous().clone() for k, v in state.items()}
    save_file(state, str(dest), metadata={"format": "pt"})


def _verify(bin_path: Path, st_path: Path) -> tuple[int, list[str]]:
    import torch
    from safetensors.torch import load_file

    bin_sd = _load_bin_state_dict(bin_path)
    st_sd = load_file(str(st_path))
    problems: list[str] = []
    for key in sorted(set(bin_sd) | set(st_sd)):
        if key not in st_sd:
            problems.append(f"missing in safetensors: {key}")
        elif key not in bin_sd:
            problems.append(f"extra in safetensors: {key}")
        elif not torch.equal(bin_sd[key], st_sd[key]):
            problems.append(f"tensor differs: {key}")
    return len(bin_sd), problems


def prepare(dest_dir: str, force: bool = False) -> Path:
    dest = Path(dest_dir)
    if not force and all((dest / name).is_file() for name in ("config.json", WEIGHTS_ST)):
        print(f"already prepared: {dest} (use --force to rebuild)")
        return dest

    snapshot = _cached_snapshot(MODEL_ID)
    missing = [name for name in CONFIG_FILES + (WEIGHTS_BIN,) if not (snapshot / name).is_file()]
    if missing:
        raise SystemExit(f"snapshot {snapshot} is missing {missing}")

    dest.mkdir(parents=True, exist_ok=True)
    for name in CONFIG_FILES:
        shutil.copyfile(snapshot / name, dest / name)

    st_target = dest / WEIGHTS_ST
    cached = _find_cached_safetensors(snapshot)
    if cached is not None:
        print(f"reusing cached safetensors: {cached}")
        shutil.copyfile(cached, st_target)
    else:
        print(f"converting {snapshot / WEIGHTS_BIN} -> {st_target}")
        _write_safetensors(snapshot / WEIGHTS_BIN, st_target)

    n_tensors, problems = _verify(snapshot / WEIGHTS_BIN, st_target)
    if problems:
        st_target.unlink(missing_ok=True)
        raise SystemExit("verification failed:\n  " + "\n  ".join(problems[:20]))
    print(f"verified {n_tensors} tensors; prepared {dest}")
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", default=_default_dest(), help="output directory (default: %(default)s)")
    parser.add_argument("--force", action="store_true", help="rebuild even if already prepared")
    args = parser.parse_args(argv)
    try:
        prepare(args.dest, force=args.force)
    except SystemExit:
        raise
    except Exception as exc:  # report, never trace-dump, for a CLI user
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
