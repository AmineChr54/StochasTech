"""Calibration losses: NLL of returns and energy distance.

See ``docs/math/calibration-losses.md``.
"""
from __future__ import annotations

import math

import torch


def energy_distance(simulated: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    """Energy distance between two 1D samples.

    $\\mathcal{E}(P, Q) = 2\\,E\\|X - Y\\| - E\\|X - X'\\| - E\\|Y - Y'\\|$.

    For 1D returns the norm collapses to absolute value; the implementation
    forms full $n_s \\times n_o$ pairwise-distance matrices, so memory scales as
    $O(n^2)$ — chunk inputs above ~5_000 samples per side. Differentiable in
    ``simulated``.
    """
    s = simulated.flatten()
    o = observed.flatten()
    if s.numel() < 2 or o.numel() < 2:
        raise ValueError("energy_distance needs at least 2 samples per side")
    cross = (s.unsqueeze(1) - o.unsqueeze(0)).abs().mean()
    self_s = (s.unsqueeze(1) - s.unsqueeze(0)).abs().mean()
    self_o = (o.unsqueeze(1) - o.unsqueeze(0)).abs().mean()
    return 2.0 * cross - self_s - self_o


def nll_loss(
    simulated_returns: torch.Tensor,
    observed_returns: torch.Tensor,
    bandwidth: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """Negative log-likelihood under a Gaussian-KDE of the simulated sample.

    Builds the kernel density $\\hat p(x) = \\frac{1}{n h} \\sum_i \\phi((x - s_i)/h)$
    from ``simulated_returns`` and returns $-\\frac{1}{m} \\sum_j \\log \\hat p(o_j)$
    for observed returns ``o``. Uses Silverman's rule for the bandwidth when
    ``bandwidth=None``. Differentiable in ``simulated_returns``.
    """
    s = simulated_returns.flatten()
    o = observed_returns.flatten()
    n = s.numel()
    if n < 2:
        raise ValueError("nll_loss needs at least 2 simulated samples")

    if bandwidth is None:
        sigma = s.std(unbiased=True)
        h = 1.06 * sigma * (n ** (-1.0 / 5.0))
    else:
        h = bandwidth if torch.is_tensor(bandwidth) else torch.as_tensor(
            float(bandwidth), dtype=s.dtype, device=s.device
        )
    h = torch.clamp(h, min=1e-8)

    diff = (o.unsqueeze(1) - s.unsqueeze(0)) / h
    log_kernel = -0.5 * diff * diff - 0.5 * math.log(2.0 * math.pi)
    log_density = torch.logsumexp(log_kernel, dim=1) - math.log(n) - torch.log(h)
    return -log_density.mean()
