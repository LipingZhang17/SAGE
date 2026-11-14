"""Valid-pixel masks for MPI Sintel.

Two rules are used, in this order:

1. use the devkit mask ``training/invalid/<scene>/<frame>.png`` when present;
2. otherwise mark a pixel invalid when ``|fx| >= 1000`` or ``|fy| >= 1000``.
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image

FLOW_SENTINEL = 1000.0

VALID_CHOICES = ("auto", "invalid", "flow")
VALID_DEFAULT = "auto"


def devkit_mask_path(sintel_root: str, scene: str, frame_name: str, split: str = "training") -> str:
    return os.path.join(sintel_root, split, "invalid", scene, frame_name + ".png")


def devkit_valid_mask(
    sintel_root: str, scene: str, frame_name: str, split: str = "training"
) -> np.ndarray | None:
    """Devkit mask as ``1 = valid``; ``None`` when the file is absent."""
    path = devkit_mask_path(sintel_root, scene, frame_name, split)
    if not os.path.isfile(path):
        return None
    invalid = np.array(Image.open(path)) > 0
    return 1.0 - invalid.astype(np.float32)


def flow_valid_mask(flow) -> np.ndarray:
    """Fallback rule on a ``(H, W, 2)`` flow field."""
    f = np.asarray(flow)
    if f.ndim != 3 or f.shape[-1] < 2:
        raise ValueError(f"flow must be (H, W, 2), got {f.shape}")
    ok = (np.abs(f[..., 0]) < FLOW_SENTINEL) & (np.abs(f[..., 1]) < FLOW_SENTINEL)
    return ok.astype(np.float32)


def resolve_valid_mask(
    flow,
    sintel_root: str,
    scene: str,
    frame_name: str,
    split: str = "training",
    mode: str = VALID_DEFAULT,
) -> np.ndarray:
    """Return the valid mask for one frame under the selected rule."""
    if mode not in VALID_CHOICES:
        raise ValueError(f"valid mode must be one of {VALID_CHOICES}, got {mode!r}")
    if mode in ("auto", "invalid"):
        mask = devkit_valid_mask(sintel_root, scene, frame_name, split)
        if mask is not None:
            return mask
        if mode == "invalid":
            raise FileNotFoundError(devkit_mask_path(sintel_root, scene, frame_name, split))
    return flow_valid_mask(flow)
