"""Spring metrics: EPE, 1px, Fl and speed-bucket EPE.

All metrics are aggregated over valid pixels (micro). A pixel counts as
``1px`` when its EPE is at least one pixel and as ``Fl`` when
``EPE > 3px`` and ``EPE > 5% * |gt|``. Speed buckets split the error by the
ground-truth flow magnitude with thresholds at 10 and 40 pixels.
"""

from __future__ import annotations

import numpy as np

SPRING_PX1_THRESHOLD = 1.0
SPRING_FL_PX_THRESHOLD = 3.0
SPRING_FL_REL_THRESHOLD = 0.05

SPEED_BUCKETS = (("s0_10", 0.0, 10.0), ("s10_40", 10.0, 40.0), ("s40_", 40.0, None))


def epe_map(pred_hw2: np.ndarray, gt_hw2: np.ndarray) -> np.ndarray:
    """End-point error of a ``(H, W, 2)`` prediction against the ground truth."""
    return np.linalg.norm(
        np.asarray(pred_hw2, dtype=np.float64) - np.asarray(gt_hw2, dtype=np.float64), axis=-1
    )


def magnitude_map(gt_hw2: np.ndarray) -> np.ndarray:
    """Ground-truth flow magnitude of a ``(H, W, 2)`` field."""
    return np.linalg.norm(np.asarray(gt_hw2, dtype=np.float64), axis=-1)


def px1_mask(epe: np.ndarray) -> np.ndarray:
    return np.asarray(epe) >= SPRING_PX1_THRESHOLD


def outlier_fl(epe: np.ndarray, mag: np.ndarray) -> np.ndarray:
    return (np.asarray(epe) > SPRING_FL_PX_THRESHOLD) & (
        np.asarray(epe) > SPRING_FL_REL_THRESHOLD * np.asarray(mag)
    )


class SpringAccum:
    """Accumulator for the Spring metrics."""

    def __init__(self) -> None:
        self.px = 0
        self.n_frame = 0
        self.n_empty = 0
        self.epe_sum = 0.0
        self.px1_sum = 0.0
        self.fl_sum = 0.0
        self.buckets = {name: [0.0, 0.0] for name, _, _ in SPEED_BUCKETS}
        self.per_frame: list[tuple] = []

    def update(self, frame_id: str, pred_hw2, gt_hw2, valid_hw) -> bool:
        v = np.asarray(valid_hw) > 0.5
        n = int(v.sum())
        if n == 0:
            self.n_empty += 1
            return False

        e_all = epe_map(pred_hw2, gt_hw2)
        m_all = magnitude_map(gt_hw2)
        e = e_all[v]
        m = m_all[v]

        n_px1 = int(px1_mask(e).sum())
        n_fl = int(outlier_fl(e, m).sum())
        epe_sum = float(e.sum())

        self.px += n
        self.epe_sum += epe_sum
        self.px1_sum += n_px1
        self.fl_sum += n_fl

        for name, lo, hi in SPEED_BUCKETS:
            inside = (m >= lo) if hi is None else ((m >= lo) & (m < hi))
            nb = int(inside.sum())
            if nb:
                self.buckets[name][0] += float(e[inside].sum())
                self.buckets[name][1] += nb

        self.n_frame += 1
        self.per_frame.append((frame_id, epe_sum / n, 100.0 * n_px1 / n, 100.0 * n_fl / n, n))
        return True

    def metrics(self) -> dict:
        def ratio(num: float, den: float) -> float:
            return (num / den) if den else float("nan")

        out = {
            "epe": ratio(self.epe_sum, self.px),
            "px1": 100.0 * ratio(self.px1_sum, self.px),
            "fl": 100.0 * ratio(self.fl_sum, self.px),
            "n_px": self.px,
            "n_frame": self.n_frame,
        }
        for name, _, _ in SPEED_BUCKETS:
            num, den = self.buckets[name]
            out[name] = ratio(num, den)
            out[f"{name}_n"] = den
        return out
