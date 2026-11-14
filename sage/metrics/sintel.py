"""MPI Sintel benchmark metrics.

Reported quantities per pass (clean / final):

* EPE over all, matched and unmatched pixels;
* EPE bucketed by distance to the nearest motion boundary (``d0-10``,
  ``d10-60``, ``d60-140``);
* EPE bucketed by ground-truth flow magnitude (``s0-10``, ``s10-40``, ``s40+``).

All values are averaged over valid pixels (micro average). A motion boundary is
a pixel whose 4-neighbour flow difference exceeds 0.5 pixels.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

OVERALL_LABELS = (
    ("all", "EPE all"),
    ("matched", "EPE matched"),
    ("unmatched", "EPE unmatched"),
    ("d0_10", "d0-10"),
    ("d10_60", "d10-60"),
    ("d60_140", "d60-140"),
    ("s0_10", "s0-10"),
    ("s10_40", "s10-40"),
    ("s40_", "s40+"),
)


def epe_map(pred_hw2: np.ndarray, gt_hw2: np.ndarray) -> np.ndarray:
    """End-point error of a ``(H, W, 2)`` prediction against the ground truth."""
    return np.linalg.norm(
        np.asarray(pred_hw2, dtype=np.float64) - np.asarray(gt_hw2, dtype=np.float64), axis=-1
    )


def distance_to_motion_boundary(flow_np: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    """Euclidean distance to the nearest flow discontinuity, in pixels."""
    fx, fy = flow_np[0], flow_np[1]
    h, w = fx.shape
    if h < 2 and w < 2:
        return np.zeros((h, w), dtype=np.float32)
    boundary = np.zeros((h, w), dtype=bool)
    boundary[:, :-1] |= (np.abs(fx[:, 1:] - fx[:, :-1]) > thresh) | (
        np.abs(fy[:, 1:] - fy[:, :-1]) > thresh
    )
    boundary[:-1, :] |= (np.abs(fx[1:, :] - fx[:-1, :]) > thresh) | (
        np.abs(fy[1:, :] - fy[:-1, :]) > thresh
    )
    return distance_transform_edt(~boundary).astype(np.float32)


def bucket_masks(flow_np: np.ndarray, valid: np.ndarray, occ: np.ndarray) -> dict:
    """Boolean masks of the reported subsets for one frame.

    ``flow_np`` is ``(2, H, W)``, ``valid`` and ``occ`` are ``(H, W)``.
    """
    mag = np.linalg.norm(flow_np, axis=0)
    dist = distance_to_motion_boundary(flow_np)
    ok = valid > 0.5
    return {
        "all": ok,
        "matched": ok & (occ < 0.5),
        "unmatched": ok & (occ > 0.5),
        "d0_10": ok & (dist < 10.0),
        "d10_60": ok & (dist >= 10.0) & (dist < 60.0),
        "d60_140": ok & (dist >= 60.0),
        "s0_10": ok & (mag < 10.0),
        "s10_40": ok & (mag >= 10.0) & (mag < 40.0),
        "s40_": ok & (mag >= 40.0),
    }


class SintelAccum:
    """Accumulate Sintel EPE per pass, per subset and per scene."""

    def __init__(self) -> None:
        keys = [k for k, _ in OVERALL_LABELS]
        self.sums = {p: {k: 0.0 for k in keys} for p in ("clean", "final")}
        self.counts = {p: {k: 0 for k in keys} for p in ("clean", "final")}
        self.scene_sums = {p: {} for p in ("clean", "final")}
        self.scene_counts = {p: {} for p in ("clean", "final")}

    def update(self, pass_name: str, scene: str, epe: np.ndarray, masks: dict) -> None:
        for key, mask in masks.items():
            n = int(mask.sum())
            if n == 0:
                continue
            self.sums[pass_name][key] += float(epe[mask].sum())
            self.counts[pass_name][key] += n
        ok = masks["all"]
        n = int(ok.sum())
        if n:
            self.scene_sums[pass_name][scene] = self.scene_sums[pass_name].get(scene, 0.0) + float(
                epe[ok].sum()
            )
            self.scene_counts[pass_name][scene] = self.scene_counts[pass_name].get(scene, 0) + n

    def epe(self, pass_name: str, key: str) -> float:
        n = self.counts[pass_name][key]
        return self.sums[pass_name][key] / n if n else float("nan")

    def scene_epe(self, pass_name: str, scene: str) -> float:
        n = self.scene_counts[pass_name].get(scene, 0)
        return self.scene_sums[pass_name].get(scene, 0.0) / n if n else float("nan")
