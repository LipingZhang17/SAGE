"""Runtime bootstrap.

SAGE uses the public WAFT implementation as its host model. This module puts
the WAFT tree on ``sys.path`` and applies two runtime fixes without editing any
file under ``third_party``:

* DINOv3 feature extractor: resolve its repository path and local weights;
* ``utils.frame_utils.readFlow``: make the magic-number check work on NumPy 2.x.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)


def _find_waft_root() -> str:
    candidates = []
    env = os.environ.get("WAFT_DIR")
    if env:
        candidates.append(env)
    candidates += [
        os.path.join(_PROJECT_ROOT, "third_party", "WAFT"),
        os.path.join(_PROJECT_ROOT, "..", "WAFT"),
    ]
    for c in candidates:
        c = os.path.abspath(c)
        if os.path.isdir(os.path.join(c, "model")):
            return c
    raise RuntimeError(
        "WAFT not found. Place it at third_party/WAFT or set WAFT_DIR to a "
        "directory that contains model/."
    )


WAFT_ROOT = _find_waft_root()
# WAFT is appended rather than prepended: the project has no packages named
# ``model``, ``utils``, ``criterion`` or ``dataloader``, so WAFT still resolves,
# while the project's own entry modules are not shadowed by WAFT's ``train.py``
# or ``evaluate.py``.
if WAFT_ROOT in sys.path:
    sys.path.remove(WAFT_ROOT)
sys.path.append(WAFT_ROOT)


def fix_dinov3_paths() -> None:
    """Resolve the DINOv3 hub repository.

    The upstream default is ``thirdparty/dinov3`` relative to the WAFT root.
    If that copy is absent, ``third_party/dinov3`` of this project is used.
    """
    try:
        import model.backbone.dinov3 as d3
    except Exception:
        return
    if not getattr(d3, "REPO_DIR", None) or os.path.isabs(d3.REPO_DIR):
        return
    upstream = os.path.join(WAFT_ROOT, d3.REPO_DIR)
    local = os.path.join(_PROJECT_ROOT, "third_party", "dinov3")
    if not os.path.isdir(upstream) and os.path.isdir(local):
        d3.REPO_DIR = local
    else:
        d3.REPO_DIR = upstream


_DINOV3_WEIGHTS = {
    "vits": "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    "vitb": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "vitl": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
}


def dinov3_weight_dirs():
    dirs = [
        os.path.join(_PROJECT_ROOT, "model_zoo", "dinov3-ckpts"),
        os.path.join(_PROJECT_ROOT, "model_zoo"),
        os.path.join(WAFT_ROOT, "thirdparty", "dinov3", "weights"),
        os.path.join(_PROJECT_ROOT, "third_party", "dinov3", "weights"),
        os.path.join(_PROJECT_ROOT, "third_party", "dinov3"),
        os.path.expanduser("~/.cache/torch/hub/checkpoints"),
    ]
    env = os.environ.get("DINOV3_WEIGHTS_DIR")
    if env:
        dirs.insert(0, os.path.abspath(env))
    seen, out = set(), []
    for d in dirs:
        d = os.path.abspath(d)
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def find_dinov3_weight(name: str = "vits"):
    env_file = os.environ.get(f"DINOV3_WEIGHTS_{name.upper()}")
    if env_file and os.path.isfile(env_file):
        return os.path.abspath(env_file)
    fname = _DINOV3_WEIGHTS.get(name)
    if not fname:
        return None
    for d in dinov3_weight_dirs():
        cand = os.path.join(d, fname)
        if os.path.isfile(cand):
            return os.path.abspath(cand)
    return None


def fix_dinov3_weights(verbose: bool = False) -> bool:
    """Point the DINOv3 backbone at local weights when they exist."""
    try:
        import model.backbone.dinov3 as d3
    except Exception:
        return False
    urls = getattr(d3, "WEIGHTS_URLS", None)
    if not isinstance(urls, dict) or not urls:
        return False
    ready = False
    for name in list(urls):
        cur = urls.get(name)
        if (
            cur
            and isinstance(cur, str)
            and (cur.startswith(("http://", "https://", "file://")) or os.path.isfile(cur))
        ):
            if name == "vits":
                ready = True
            continue
        if cur and not isinstance(cur, str):
            if name == "vits":
                ready = True
            continue
        path = find_dinov3_weight(name)
        if path:
            urls[name] = path
            if verbose:
                print(f"[sage] DINOv3 {name} weights: {path}")
            if name == "vits":
                ready = True
        elif _DINOV3_WEIGHTS.get(name):
            urls[name] = os.path.join(dinov3_weight_dirs()[0], _DINOV3_WEIGHTS[name])
    return ready


def patch_read_flow() -> None:
    """Make WAFT's ``.flo`` reader work on NumPy 2.x (scalar magic check)."""
    try:
        from utils import frame_utils
    except Exception:
        return
    import numpy as np

    def read_flow(fn):
        with open(fn, "rb") as f:
            magic = float(np.fromfile(f, np.float32, count=1)[0])
            if magic != 202021.25:
                raise ValueError(f"invalid .flo file: {fn}")
            w = int(np.fromfile(f, np.int32, count=1)[0])
            h = int(np.fromfile(f, np.int32, count=1)[0])
            data = np.fromfile(f, np.float32, count=2 * w * h)
        return data.reshape(h, w, 2)

    frame_utils.readFlow = read_flow


def install() -> None:
    fix_dinov3_paths()
    fix_dinov3_weights()
    patch_read_flow()


install()
