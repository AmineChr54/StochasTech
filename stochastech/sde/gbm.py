"""Geometric Brownian Motion simulator.

See ``doc/math/03_gbm.md``.
"""
from __future__ import annotations

import torch

from stochastech.sde.base import euler_maruyama


def simulate_gbm(
    s0: float,
    mu: float,
    sigma: float,
    dt: float,
    n_steps: int,
    n_paths: int,
    generator: torch.Generator | None = None,
    log_euler: bool = True,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Simulate ``dS = mu S dt + sigma S dW`` paths.

    Args:
        s0: initial price, ``> 0``.
        mu: drift.
        sigma: volatility, ``>= 0``.
        dt: step size.
        n_steps: number of steps.
        n_paths: number of independent paths.
        generator: optional ``torch.Generator``.
        log_euler: if True, simulate ``log S`` (exact + positivity-preserving).
        dtype: float dtype for the output tensor.
        device: torch device.

    Returns:
        Tensor of shape ``(n_steps + 1, n_paths)``.
    """
    if s0 <= 0:
        raise ValueError(f"s0 must be positive, got {s0}")
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if n_paths < 1:
        raise ValueError(f"n_paths must be >= 1, got {n_paths}")

    if log_euler:
        # Exact GBM increments: log S_{t+dt} - log S_t ~ N((mu - sigma^2/2) dt, sigma^2 dt).
        sqrt_dt = float(dt) ** 0.5
        drift_log = (mu - 0.5 * sigma * sigma) * dt
        log_s = torch.empty((n_steps + 1, n_paths), dtype=dtype, device=device)
        log_s[0] = torch.log(torch.tensor(s0, dtype=dtype, device=device))
        for i in range(n_steps):
            z = torch.randn(n_paths, dtype=dtype, device=device, generator=generator)
            log_s[i + 1] = log_s[i] + drift_log + sigma * sqrt_dt * z
        return torch.exp(log_s)

    x0 = torch.full((n_paths,), float(s0), dtype=dtype, device=device)
    return euler_maruyama(
        drift=lambda x, _t: mu * x,
        diffusion=lambda x, _t: sigma * x,
        x0=x0,
        dt=dt,
        n_steps=n_steps,
        generator=generator,
    )
