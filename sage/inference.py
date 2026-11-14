"""Inference helper shared by evaluation and submission."""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def infer_flow(model, img1: torch.Tensor, img2: torch.Tensor, scale: float = 0.0) -> torch.Tensor:
    """Predict flow for one image pair.

    ``scale`` follows the WAFT convention: the inputs are resized by
    ``2 ** scale`` before the forward pass and the prediction is resized back,
    with the flow magnitude scaled accordingly. ``scale = 0`` is a no-op.
    """
    if scale != 0:
        img1 = F.interpolate(img1, scale_factor=2**scale, mode="bilinear", align_corners=True)
        img2 = F.interpolate(img2, scale_factor=2**scale, mode="bilinear", align_corners=True)
    flow = model(img1, img2)["flow"][-1]
    if scale != 0:
        flow = F.interpolate(flow, scale_factor=0.5**scale, mode="bilinear", align_corners=True) * (
            0.5**scale
        )
    return flow
