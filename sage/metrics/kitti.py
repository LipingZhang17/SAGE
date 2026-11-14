"""KITTI-15 metrics: F1-all and F1-epe.

``F1-all`` is the percentage of bad pixels over all valid pixels of the
dataset. A pixel is bad when ``EPE > 3px`` and ``EPE > 5% * |gt|``.
``F1-epe`` is the mean endpoint error per frame, averaged over frames, which is
the convention used by RAFT and WAFT for KITTI.
"""

from __future__ import annotations

import numpy as np

KITTI_PX_THRESHOLD = 3.0
KITTI_REL_THRESHOLD = 0.05


def epe_map(pred_hw2: np.ndarray, gt_hw2: np.ndarray) -> np.ndarray:
    """End-point error of a ``(H, W, 2)`` prediction against the ground truth."""
    return np.linalg.norm(
        np.asarray(pred_hw2, dtype=np.float64) - np.asarray(gt_hw2, dtype=np.float64), axis=-1
    )


def magnitude_map(gt_hw2: np.ndarray) -> np.ndarray:
    """Ground-truth flow magnitude of a ``(H, W, 2)`` field."""
    return np.linalg.norm(np.asarray(gt_hw2, dtype=np.float64), axis=-1)


def outlier_all(epe: np.ndarray, mag: np.ndarray) -> np.ndarray:
    return (np.asarray(epe) > KITTI_PX_THRESHOLD) & (
        np.asarray(epe) > KITTI_REL_THRESHOLD * np.asarray(mag)
    )


class KittiAccum:
    """Accumulator for F1-all (per pixel) and F1-epe (per frame)."""

    def __init__(self) -> None:
        self.px = 0
        self.n_frame = 0
        self.n_empty = 0
        self.fl_all_sum = 0.0
        self.frame_epe_sum = 0.0
        self.per_frame: list[tuple] = []

    def update(self, frame_id: str, pred_hw2, gt_hw2, valid_hw) -> bool:
        v = np.asarray(valid_hw) > 0.5
        n = int(v.sum())
        if n == 0:
            self.n_empty += 1
            return False
        e = epe_map(pred_hw2, gt_hw2)[v]
        m = magnitude_map(gt_hw2)[v]
        bad = outlier_all(e, m)

        self.px += n
        self.fl_all_sum += float(bad.sum())
        frame_mean = float(e.mean())
        self.n_frame += 1
        self.frame_epe_sum += frame_mean
        self.per_frame.append((frame_id, frame_mean, 100.0 * float(bad.mean()), n))
        return True

    def metrics(self) -> dict:
        def ratio(num: float, den: float) -> float:
            return (num / den) if den else float("nan")

        return {
            "fl_all": 100.0 * ratio(self.fl_all_sum, self.px),
            "fl_epe": ratio(self.frame_epe_sum, self.n_frame),
            "n_px": self.px,
            "n_frame": self.n_frame,
        }
