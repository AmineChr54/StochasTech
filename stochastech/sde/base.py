"""Generic Euler-Maruyama core.

See ``docs/math/euler-maruyama.md`` for the scheme and convergence orders.
"""
from __future__ import annotations

from collections.abc import Callable

import torch


def euler_maruyama(
    drift: Callable[[torch.Tensor, float], torch.Tensor],
    diffusion: Callable[[torch.Tensor, float], torch.Tensor],
    x0: torch.Tensor,
    dt: float,
    n_steps: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Simulate paths under ``dX = drift(X, t) dt + diffusion(X, t) dW``.

    Diagonal-noise convention: each component of ``X`` is driven by its own
    independent scalar Brownian motion, scaled by ``diffusion(X, t)`` evaluated
    componentwise. For correlated multi-factor models (e.g. Heston) compose this
    primitive with a correlation transform on the standard normal draws before
    passing them to ``diffusion``, or write a model-specific stepper.

    Args:
        drift: function ``(x, t) -> mu(x, t)``, broadcasting over the path batch.
        diffusion: function ``(x, t) -> sigma(x, t)``, same shape contract.
        x0: initial state with shape ``(n_paths,)`` or ``(n_paths, d)``.
        dt: step size.
        n_steps: number of steps; output has ``n_steps + 1`` time points.
        generator: optional ``torch.Generator`` for reproducible draws.

    Returns:
        Tensor of shape ``(n_steps + 1, *x0.shape)``.
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")

    sqrt_dt = float(dt) ** 0.5
    out = torch.empty((n_steps + 1, *x0.shape), dtype=x0.dtype, device=x0.device)
    out[0] = x0
    x = x0
    t = 0.0
    for i in range(n_steps):
        z = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
        x = x + drift(x, t) * dt + diffusion(x, t) * sqrt_dt * z
        out[i + 1] = x
        t += dt
    return out
