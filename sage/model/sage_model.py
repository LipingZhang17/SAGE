"""SAGE model: WAFT with a learned soft feature read-out.

The model keeps the WAFT architecture (feature encoder, feature network,
update transformer, flow head and upsampling) and replaces the hard warp
``F2[p + f[p]]`` inside the iterative update by the soft read-out defined in
:class:`sage.model.soft_select.SoftSelect`::

    F2_soft = sum_j A[p, j] * F2[p + f[p] + delta_j]

``lam`` interpolates between the hard warp (``1 - lam``) and the soft warp
(``lam``); ``lam = 0`` reproduces the WAFT baseline exactly.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from model.waft_a2 import WAFTv2
from utils.utils import Padder, bilinear_sampler, coords_grid

from sage import bootstrap  # noqa: F401  (adds WAFT to sys.path)

from .soft_select import SoftSelect


def structure_tensor(f: torch.Tensor, win: int = 5) -> torch.Tensor:
    """Per-pixel 2x2 structure tensor ``S = sum_c grad f_c grad f_c^T``.

    Gradients are computed with Sobel filters and averaged over a ``win x win``
    window. Returns a symmetric tensor of shape ``(B, 2, 2, H, W)``.
    """
    b, c, h, w = f.shape
    dev, dt = f.device, f.dtype
    kx = (
        torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=dev, dtype=dt)
        .view(1, 1, 3, 3)
        .repeat(c, 1, 1, 1)
    )
    ky = (
        torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=dev, dtype=dt)
        .view(1, 1, 3, 3)
        .repeat(c, 1, 1, 1)
    )
    gx = F.conv2d(f, kx, padding=1, groups=c)
    gy = F.conv2d(f, ky, padding=1, groups=c)
    s11 = (gx * gx).sum(1, keepdim=True)
    s12 = (gx * gy).sum(1, keepdim=True)
    s22 = (gy * gy).sum(1, keepdim=True)
    p = win // 2
    s11 = F.avg_pool2d(s11, win, stride=1, padding=p)
    s12 = F.avg_pool2d(s12, win, stride=1, padding=p)
    s22 = F.avg_pool2d(s22, win, stride=1, padding=p)
    s = torch.zeros(b, 2, 2, h, w, device=dev, dtype=dt)
    s[:, 0, 0] = s11[:, 0]
    s[:, 0, 1] = s12[:, 0]
    s[:, 1, 0] = s12[:, 0]
    s[:, 1, 1] = s22[:, 0]
    return s


class SAGEModel(WAFTv2):
    """WAFT with a soft, flow-conditioned feature read-out.

    Args:
        args: configuration namespace shared with WAFT.
        lam: weight of the soft read-out in ``[0, 1]``.
        k: candidate window side length.
        learnable: attach a learnable :class:`SoftSelect` head.
        softselect_kwargs: keyword arguments forwarded to :class:`SoftSelect`.
        beta: isotropic lower bound of the structure-tensor precision, used
            only by :meth:`precision_pair` when the anisotropy regularizer is
            enabled.
        win: window size of the structure tensor.
    """

    def __init__(
        self,
        args,
        lam: float = 1.0,
        k: int = 5,
        learnable: bool = True,
        softselect_kwargs: dict | None = None,
        beta: float = 0.1,
        win: int = 5,
    ):
        super().__init__(args)
        self.lam = float(lam)
        self.k = int(k)
        self.beta = float(beta)
        self.win = int(win)
        self.softselect = None
        if learnable:
            self.softselect = SoftSelect(self.iter_dim, k=self.k, **(softselect_kwargs or {}))
        elif self.lam > 0:
            raise ValueError("a learnable selection head is required when lam > 0")

    def precision_pair(self, image1: torch.Tensor, image2: torch.Tensor):
        """Return the learned and the structure-tensor precisions.

        Both have shape ``(B, 2, 2, H/2, W/2)``. The structure-tensor precision
        is the trace-normalized structure tensor of the reference features plus
        ``beta * I``. The learned precision is ``L^T L`` of the selection head.
        """
        with torch.no_grad():
            image1 = self.normalize_image(image1)
            image2 = self.normalize_image(image2)
            h, w = image1.shape[-2:]
            padder = Padder(image1.shape, factor=self.factor)
            image1 = padder.pad(image1)
            image2 = padder.pad(image2)
            fmap1 = self.fmap_conv(torch.cat([self.encoder(image1), self.fnet(image1)[0]], dim=1))
            fmap2 = self.fmap_conv(torch.cat([self.encoder(image2), self.fnet(image2)[0]], dim=1))

        s = structure_tensor(fmap1, win=self.win)
        tr = s[:, 0, 0] + s[:, 1, 1] + 1e-6
        scale = 2.0 / tr
        prec_struct = torch.zeros_like(s)
        prec_struct[:, 0, 0] = s[:, 0, 0] * scale + self.beta
        prec_struct[:, 0, 1] = s[:, 0, 1] * scale
        prec_struct[:, 1, 0] = s[:, 0, 1] * scale
        prec_struct[:, 1, 1] = s[:, 1, 1] * scale + self.beta

        prec_learned = None
        if self.softselect is not None:
            prec_learned = self.softselect.precision_matrix(fmap1, fmap2)

        left, _, top, _ = padder._pad
        lh, th = left // 2, top // 2
        hh, wh = h // 2, w // 2
        prec_struct = prec_struct[..., th : th + hh, lh : lh + wh]
        if prec_learned is not None:
            prec_learned = prec_learned[..., th : th + hh, lh : lh + wh]
        return prec_learned, prec_struct

    def forward(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        iters: int | None = None,
        flow_gt: torch.Tensor | None = None,
    ):
        """Run the iterative update.

        Returns a dict with the flow and information predictions for every
        iteration. When ``flow_gt`` is given, negative log-likelihood maps are
        added under the ``nf`` key for training.
        """
        if iters is None:
            iters = self.args.iters
        image1 = self.normalize_image(image1)
        image2 = self.normalize_image(image2)
        padder = Padder(image1.shape, factor=self.factor)
        image1 = padder.pad(image1)
        image2 = padder.pad(image2)

        flow_predictions = []
        info_predictions = []
        n, _, h, w = image1.shape
        fmap1 = self.fmap_conv(torch.cat([self.encoder(image1), self.fnet(image1)[0]], dim=1))
        fmap2 = self.fmap_conv(torch.cat([self.encoder(image2), self.fnet(image2)[0]], dim=1))
        net = self.hidden_conv(torch.cat([fmap1, fmap2], dim=1))
        flow_2x = torch.zeros(n, 2, h // 2, w // 2, device=image1.device)

        for _ in range(iters):
            flow_2x = flow_2x.detach()
            if self.lam < 1.0:
                coords2 = (coords_grid(n, h // 2, w // 2, device=image1.device) + flow_2x).detach()
                warp_hard = bilinear_sampler(fmap2, coords2.permute(0, 2, 3, 1))

            if self.lam > 0:
                warp_soft, _ = self.softselect(fmap1, fmap2, flow_2x)
                if self.lam >= 1.0:
                    warp_2x = warp_soft
                else:
                    warp_2x = self.lam * warp_soft + (1.0 - self.lam) * warp_hard
            else:
                warp_2x = warp_hard

            refine_inp = self.warp_linear(torch.cat([fmap1, warp_2x, net, flow_2x], dim=1))
            refine_outs = self.refine_net(refine_inp)
            net = self.refine_transform(torch.cat([refine_outs["out"], net], dim=1))
            flow_update = self.flow_head(net)
            weight_update = 0.25 * self.upsample_weight(net)
            flow_2x = flow_2x + flow_update[:, :2]
            info_2x = flow_update[:, 2:]
            flow_up, info_up = self.upsample_data(flow_2x, info_2x, weight_update)
            flow_predictions.append(flow_up)
            info_predictions.append(info_up)

        for i in range(len(info_predictions)):
            flow_predictions[i] = padder.unpad(flow_predictions[i])
            info_predictions[i] = padder.unpad(info_predictions[i])

        if flow_gt is not None:
            nf_predictions = []
            for i in range(len(info_predictions)):
                raw_b = info_predictions[i][:, 2:]
                log_b = torch.zeros_like(raw_b)
                weight = info_predictions[i][:, :2]
                log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=self.args.var_max)
                log_b[:, 1] = torch.clamp(raw_b[:, 1], min=self.args.var_min, max=0)
                term2 = ((flow_gt - flow_predictions[i]).abs().unsqueeze(2)) * (
                    torch.exp(-log_b).unsqueeze(1)
                )
                term1 = weight - math.log(2) - log_b
                nf_loss = torch.logsumexp(weight, dim=1, keepdim=True) - torch.logsumexp(
                    term1.unsqueeze(1) - term2, dim=2
                )
                nf_predictions.append(nf_loss)
            return {"flow": flow_predictions, "info": info_predictions, "nf": nf_predictions}

        return {"flow": flow_predictions, "info": info_predictions}
