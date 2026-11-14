"""Checkpoint utilities."""

from __future__ import annotations

import os

import torch


def safe_torch_load(path: str, map_location="cpu"):
    """Load a checkpoint, allowing non-tensor objects such as RNG state."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        return torch.load(path, map_location=map_location, weights_only=False)


def state_dict_of(obj) -> dict:
    """Return the bare state dict from a checkpoint or a dict of tensors."""
    if isinstance(obj, dict):
        for key in ("model", "state_dict"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj


def load_sage_ckpt(
    model: torch.nn.Module, path: str, require_prefixes: tuple[str, ...] = ("softselect.",)
) -> None:
    """Load a SAGE checkpoint and verify that the selection head was loaded.

    ``strict=False`` alone is not enough: a checkpoint whose keys do not match
    the model would be partially ignored without any error. Every prefix in
    ``require_prefixes`` must therefore match at least one parameter.
    """
    try:
        obj = safe_torch_load(path)
    except Exception as exc:
        raise RuntimeError(f"cannot read checkpoint {path}: {exc}") from exc
    state = state_dict_of(obj)
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"{path} does not contain a state dict")
    missing, unexpected = model.load_state_dict(state, strict=False)
    for prefix in require_prefixes:
        if not any(k.startswith(prefix) for k in state):
            raise RuntimeError(
                f"{path} contains no parameters with prefix {prefix!r}; it is not a SAGE checkpoint"
            )
        if any(k.startswith(prefix) for k in missing):
            bad = [k for k in missing if k.startswith(prefix)][:5]
            raise RuntimeError(f"{path} is missing {prefix} parameters of this model, e.g. {bad}")
    if unexpected:
        print(f"[sage] ignored {len(unexpected)} unexpected checkpoint entries")
    if missing:
        print(f"[sage] {len(missing)} model parameters keep their initial value")


def atomic_save(obj, path: str) -> None:
    """Write to ``path`` atomically so that an interrupted save cannot corrupt it."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(obj, tmp)
    os.replace(tmp, path)
