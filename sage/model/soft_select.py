"""Soft feature selection operator for SAGE.

Given the current flow estimate ``f``, the operator replaces the hard warp
``F2[p + f[p]]`` with a weighted average over a ``k x k`` candidate window::

    q_j       = p + f[p] + delta_j
    A[p, j]   = softmax_j ( g(p, j) + c(p, j) )
    F2_soft   = sum_j A[p, j] * F2[q_j]

``g`` is the geometry logit and ``c`` an optional cosine content term. With
``anisotropic=True`` the geometry logit is a quadratic form with a per-pixel
SPD precision matrix ``Sigma^-1 = L^T L``, where ``L`` is a lower triangular
Cholesky factor predicted by a small convolutional head.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Bounds applied to the raw scale parameters before exponentiation. The
# anisotropic factor is ``l11 = 1 / sigma`` while the isotropic scale is
# ``b = sigma``; the symmetric interval makes both branches cover
# ``sigma in [0.25, 4]``.
_LOG_SCALE_BOUND = math.log(4.0)
_L21_BOUND = 2.0


def _det_hash(idx: torch.Tensor, salt: float = 0.0) -> torch.Tensor:
    """Deterministic pseudo-random values in [0, 1) derived from indices."""
    v = torch.sin(idx * 12.9898 + 78.233 + float(salt) * 0.017) * 43758.5453
    return v - v.floor()


def _det_semi_orthogonal(rows: int, cols: int, salt: float = 0.0) -> torch.Tensor:
    """Deterministic row-orthogonal matrix scaled by 1/sqrt(3)."""
    idx = torch.arange(rows * cols, dtype=torch.float64)
    a = (_det_hash(idx, salt) * 2.0 - 1.0).reshape(rows, cols)
    if rows <= cols:
        q, r = torch.linalg.qr(a.t())
        d = torch.diagonal(r)
        q = q * torch.where(d < 0, -1.0, 1.0).unsqueeze(0)
        w = q.t()
    else:
        q, r = torch.linalg.qr(a)
        d = torch.diagonal(r)
        w = q * torch.where(d < 0, -1.0, 1.0).unsqueeze(0)
    return w * (1.0 / math.sqrt(3.0))


def deterministic_orthogonal_(
    tensor: torch.Tensor, seed: int = 0, fan_in: int | None = None
) -> torch.Tensor:
    """Fill ``tensor`` in place without touching the global RNG.

    Weights are row-orthogonal with the same Frobenius norm as the default
    Kaiming initialization; biases are uniform in ``1 / sqrt(fan_in)``. This
    makes the selection head initialization independent of the training seed.
    """
    with torch.no_grad():
        if tensor.dim() == 1:
            if fan_in is None:
                raise ValueError("fan_in is required for 1-D tensors")
            bound = 1.0 / math.sqrt(int(fan_in)) if int(fan_in) > 0 else 1.0
            idx = torch.arange(tensor.numel(), dtype=torch.float64, device=tensor.device)
            v = _det_hash(idx, seed)
            out = (v * 2.0 - 1.0) * bound
        else:
            rows = int(tensor.shape[0])
            cols = tensor.numel() // rows
            out = _det_semi_orthogonal(rows, cols, salt=seed)
        tensor.copy_(out.reshape(tensor.shape).to(tensor.dtype))
    return tensor


class SoftSelect(nn.Module):
    """Flow-conditioned soft feature read-out.

    Args:
        dim: channel width of the feature maps.
        k: side length of the candidate window.
        kernel: ``gaussian``, ``laplace`` or ``mixture``.
        n_mix: number of mixture components (only for ``kernel='mixture'``).
        use_content: add a cosine similarity term to the geometry logit.
        learn_mu: predict a per-pixel window centre offset.
        anisotropic: learn a full 2x2 precision matrix instead of one scale.
        temperature_init: initial temperature of the content term.
        init_seed: if given, initialize the head randomly with this seed;
            otherwise the head is initialized deterministically.
    """

    def __init__(
        self,
        dim: int,
        k: int = 5,
        kernel: str = "laplace",
        n_mix: int = 1,
        use_content: bool = False,
        learn_mu: bool = True,
        anisotropic: bool = False,
        temperature_init: float = 4.0,
        init_seed: int | None = None,
    ):
        super().__init__()
        if kernel != "mixture" and n_mix != 1:
            raise ValueError("n_mix > 1 requires kernel='mixture'")
        self.k = k
        self.kernel = kernel
        self.n_mix = n_mix
        self.use_content = use_content
        self.learn_mu = learn_mu
        self.anisotropic = anisotropic

        if use_content:
            self.log_temperature = nn.Parameter(torch.log(torch.tensor(float(temperature_init))))

        r = torch.arange(-(k // 2), k // 2 + 1, dtype=torch.float32)
        dy, dx = torch.meshgrid(r, r, indexing="ij")
        self.register_buffer("off", torch.stack([dx, dy], dim=0).reshape(2, -1))

        out_c = 0
        if learn_mu:
            out_c += n_mix * 2
        out_c += n_mix * 3 if anisotropic else n_mix
        if kernel == "mixture":
            out_c += n_mix
        self.geometry_head = nn.Sequential(
            nn.Conv2d(dim * 3, dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, out_c, 1),
        )
        # Zero the last layer so that the initial layout is the one implied by
        # ``kernel``: identity precision, no centre offset, uniform mixture.
        nn.init.zeros_(self.geometry_head[-1].weight)
        nn.init.zeros_(self.geometry_head[-1].bias)

        w0 = self.geometry_head[0].weight
        fan_in = w0.shape[1] * w0.shape[2] * w0.shape[3]
        if init_seed is None:
            deterministic_orthogonal_(w0)
            deterministic_orthogonal_(self.geometry_head[0].bias, seed=1, fan_in=fan_in)
        else:
            g = torch.Generator().manual_seed(int(init_seed))
            nn.init.kaiming_uniform_(w0, a=math.sqrt(5), generator=g)
            bound = 1.0 / math.sqrt(fan_in)
            with torch.no_grad():
                self.geometry_head[0].bias.uniform_(-bound, bound, generator=g)

    def build_candidates(self, f2: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Sample ``F2`` at the ``k * k`` candidate positions of every pixel."""
        b, c, h, w = f2.shape
        dev, dt = f2.device, f2.dtype
        yy, xx = torch.meshgrid(
            torch.arange(h, device=dev, dtype=dt),
            torch.arange(w, device=dev, dtype=dt),
            indexing="ij",
        )
        base = torch.stack([xx, yy], dim=0)
        coords = (
            base.reshape(2, h * w, 1)
            + flow.reshape(b, 2, h * w, 1)
            + self.off.reshape(2, 1, self.k * self.k)
        )
        cn = coords.clone()
        cn[:, 0] = cn[:, 0] / (w - 1) * 2 - 1
        cn[:, 1] = cn[:, 1] / (h - 1) * 2 - 1
        cn = cn.permute(0, 2, 3, 1)
        cand = F.grid_sample(f2, cn, mode="bilinear", align_corners=True)
        return cand.reshape(b, c, h * w, self.k * self.k).permute(0, 2, 3, 1)

    def content_logit(self, f1: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
        b, c, h, w = f1.shape
        q = f1.reshape(b, c, h * w).permute(0, 2, 1).unsqueeze(2)
        sim = (F.normalize(q, dim=-1) * F.normalize(cand, dim=-1)).sum(-1)
        return self.log_temperature.exp() * sim

    @staticmethod
    def decode_cholesky(chol: torch.Tensor, dim: int = -1):
        """Decode three raw channels into the Cholesky factor ``(l11, l21, l22)``."""

        def take(i):
            return chol.select(dim, i) if dim != -1 else chol[..., i]

        b = _LOG_SCALE_BOUND
        l11 = take(0).clamp(min=-b, max=b).exp()
        l21 = take(1).clamp(min=-_L21_BOUND, max=_L21_BOUND)
        l22 = take(2).clamp(min=-b, max=b).exp()
        return l11, l21, l22

    def geometry_logit(
        self,
        f1: torch.Tensor,
        f2_warp: torch.Tensor,
        cand: torch.Tensor | None = None,
        content_fn=None,
    ) -> torch.Tensor:
        """Per-candidate geometry logits, shape ``(B, H*W, k*k)``.

        ``cand`` and ``content_fn`` are only passed by :meth:`forward` when the
        content term is enabled.
        """
        b, _, h, w = f1.shape
        x = torch.cat([f1, f2_warp, (f1 - f2_warp).abs()], dim=1)
        p = self.geometry_head(x).reshape(b, -1, h * w).permute(0, 2, 1)

        idx = 0
        if self.learn_mu:
            mu = p[..., : self.n_mix * 2].reshape(b, -1, self.n_mix, 2)
            idx = self.n_mix * 2
        else:
            mu = torch.zeros(b, h * w, self.n_mix, 2, device=f1.device, dtype=f1.dtype)

        d = self.off.view(1, 1, 2, self.k * self.k) - mu.unsqueeze(-1)
        d1, d2 = d[..., 0, :], d[..., 1, :]

        if self.anisotropic:
            chol = p[..., idx : idx + self.n_mix * 3].reshape(b, -1, self.n_mix, 3)
            idx += self.n_mix * 3
            l11, l21, l22 = self.decode_cholesky(chol)
            t1 = l11.unsqueeze(-1) * d1
            t2 = l21.unsqueeze(-1) * d1 + l22.unsqueeze(-1) * d2
            if self.kernel == "gaussian":
                logits = -0.5 * (t1 * t1 + t2 * t2)
            else:
                logits = -(t1.abs() + t2.abs())
        else:
            log_b = p[..., idx : idx + self.n_mix]
            idx += self.n_mix
            b_scale = log_b.clamp(min=-_LOG_SCALE_BOUND, max=_LOG_SCALE_BOUND).exp()
            if self.kernel == "gaussian":
                logits = -0.5 * (d**2).sum(-2) / (b_scale.unsqueeze(-1) ** 2)
            else:
                logits = -d.abs().sum(-2) / b_scale.unsqueeze(-1)

        if self.kernel == "mixture":
            logit_pi = p[..., idx : idx + self.n_mix]
            logits = torch.logsumexp(logits + F.log_softmax(logit_pi, dim=-1).unsqueeze(-1), dim=2)
        else:
            logits = logits.squeeze(2)
        return content_fn(f1, cand) + logits if content_fn is not None else logits

    def precision_matrix(self, f1: torch.Tensor, f2_warp: torch.Tensor) -> torch.Tensor:
        """Return ``Sigma^-1 = L^T L``, shape ``(B, 2, 2, H, W)``."""
        if self.n_mix != 1:
            raise ValueError("precision_matrix requires a single mixture component")
        b, _, h, w = f1.shape
        x = torch.cat([f1, f2_warp, (f1 - f2_warp).abs()], dim=1)
        p = self.geometry_head(x)
        idx = self.n_mix * 2 if self.learn_mu else 0
        prec = torch.zeros(b, 2, 2, h, w, device=f1.device, dtype=f1.dtype)
        if self.anisotropic:
            chol = p[:, idx : idx + self.n_mix * 3]
            l11, l21, l22 = self.decode_cholesky(chol, dim=1)
            prec[:, 0, 0] = l11 * l11 + l21 * l21
            prec[:, 0, 1] = l21 * l22
            prec[:, 1, 0] = l21 * l22
            prec[:, 1, 1] = l22 * l22
        else:
            log_b = p[:, idx : idx + self.n_mix]
            b_scale = log_b.clamp(min=-_LOG_SCALE_BOUND, max=_LOG_SCALE_BOUND).exp()
            inv_b2 = 1.0 / (b_scale * b_scale)
            prec[:, 0, 0] = inv_b2[:, 0]
            prec[:, 1, 1] = inv_b2[:, 0]
        return prec

    def forward(
        self, f1: torch.Tensor, f2: torch.Tensor, flow: torch.Tensor, return_attn: bool = False
    ):
        """Read ``F2`` softly.

        Returns ``(f2_soft, confidence)``, where ``confidence`` is the largest
        selection weight per pixel. With ``return_attn`` the ``(B, H*W, k*k)``
        attention is returned as a third value.
        """
        b, c, h, w = f1.shape
        cand = self.build_candidates(f2, flow)
        center = (self.k * self.k) // 2
        f2_warp = cand[:, :, center, :].permute(0, 2, 1).reshape(b, c, h, w)

        z = self.geometry_logit(
            f1, f2_warp, cand=cand, content_fn=self.content_logit if self.use_content else None
        )

        attn = F.softmax(z, dim=-1)
        j = self.k * self.k
        f2_soft = (
            torch.bmm(
                attn.reshape(b * h * w, 1, j),
                cand.reshape(b * h * w, j, c),
            )
            .reshape(b, h * w, c)
            .permute(0, 2, 1)
            .reshape(b, c, h, w)
        )
        conf = attn.max(-1)[0].reshape(b, 1, h, w)
        if return_attn:
            return f2_soft, conf, attn.reshape(b, h * w, j)
        return f2_soft, conf
