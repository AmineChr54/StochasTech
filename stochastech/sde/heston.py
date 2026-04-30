"""Heston stochastic-volatility simulator (full-truncation Euler).

See ``docs/math/heston.md``.
"""
from __future__ import annotations

import math

import torch


def simulate_heston(
    s0: float,
    v0: float,
    mu: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    dt: float,
    n_steps: int,
    n_paths: int,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulate Heston paths under full-truncation Euler.

    Scheme (Lord, Koekkoek, van Dijk 2010): the raw variance trajectory ``v_t``
    is allowed to drift negative, but every appearance of ``v_t`` inside drift
    or diffusion coefficients is replaced by ``v_t^+ = max(v_t, 0)``. Log-price
    is integrated to preserve positivity of ``S_t`` exactly.

    Args:
        s0: initial price, ``> 0``.
        v0: initial variance, ``>= 0``.
        mu: drift.
        kappa: mean-reversion speed, ``>= 0``.
        theta: long-run variance, ``>= 0``.
        xi: vol of vol, ``>= 0``.
        rho: correlation between price and variance Brownians, in ``[-1, 1]``.
        dt: step size.
        n_steps: number of steps.
        n_paths: number of paths.
        generator: optional ``torch.Generator``.
        dtype: float dtype for output tensors.
        device: torch device.

    Returns:
        Tuple ``(S, V)`` each of shape ``(n_steps + 1, n_paths)``. ``V`` carries
        the raw (possibly slightly negative) full-truncation trajectory; clamp
        to ``>= 0`` if a strictly-non-negative variance series is needed.
    """
    if s0 <= 0:
        raise ValueError(f"s0 must be positive, got {s0}")
    if v0 < 0:
        raise ValueError(f"v0 must be non-negative, got {v0}")
    if kappa < 0:
        raise ValueError(f"kappa must be non-negative, got {kappa}")
    if theta < 0:
        raise ValueError(f"theta must be non-negative, got {theta}")
    if xi < 0:
        raise ValueError(f"xi must be non-negative, got {xi}")
    if not -1.0 <= rho <= 1.0:
        raise ValueError(f"rho must be in [-1, 1], got {rho}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if n_paths < 1:
        raise ValueError(f"n_paths must be >= 1, got {n_paths}")

    sqrt_dt = math.sqrt(dt)
    sqrt_1_rho2 = math.sqrt(max(0.0, 1.0 - rho * rho))

    log_s = torch.empty((n_steps + 1, n_paths), dtype=dtype, device=device)
    v = torch.empty((n_steps + 1, n_paths), dtype=dtype, device=device)
    log_s[0] = math.log(s0)
    v[0] = v0

    for i in range(n_steps):
        z_v = torch.randn(n_paths, dtype=dtype, device=device, generator=generator)
        z_perp = torch.randn(n_paths, dtype=dtype, device=device, generator=generator)
        z_s = rho * z_v + sqrt_1_rho2 * z_perp

        v_pos = v[i].clamp(min=0.0)
        sqrt_v_pos = v_pos.sqrt()

        v[i + 1] = v[i] + kappa * (theta - v_pos) * dt + xi * sqrt_v_pos * sqrt_dt * z_v
        log_s[i + 1] = log_s[i] + (mu - 0.5 * v_pos) * dt + sqrt_v_pos * sqrt_dt * z_s

    return torch.exp(log_s), v


def feller_condition(kappa: float, theta: float, xi: float) -> bool:
    """Return True iff ``2 kappa theta >= xi^2`` (continuous-SDE positivity)."""
    return 2.0 * kappa * theta >= xi * xi


def simulate_heston_diff(
    s0: float | torch.Tensor,
    v0: torch.Tensor,
    mu: torch.Tensor,
    kappa: torch.Tensor,
    theta: torch.Tensor,
    xi: torch.Tensor,
    rho: torch.Tensor,
    dt: float,
    n_steps: int,
    n_paths: int,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable Heston full-truncation Euler.

    Same scheme as ``simulate_heston`` but accepts tensor-valued parameters with
    ``requires_grad`` and propagates gradients through the entire forward path
    (BPTT). Skips the input-validation block — the caller is responsible for
    keeping ``kappa, theta, xi, v0 >= 0`` and ``rho in (-1, 1)``; for
    optimization, parameterize via ``softplus``/``tanh`` upstream.

    A small floor ``eps = 1e-12`` is added inside ``sqrt(v_t^+)`` to keep the
    derivative finite at $v = 0$; the numerical perturbation on the forward
    path is below ``1e-6`` and well inside Monte Carlo noise.
    """
    eps = 1e-12
    sqrt_dt = math.sqrt(dt)
    one_minus_rho2 = (1.0 - rho * rho).clamp(min=eps)
    sqrt_1_rho2 = one_minus_rho2.sqrt()

    log_s0_scalar = torch.log(torch.as_tensor(s0, dtype=dtype, device=device))
    log_s_steps: list[torch.Tensor] = [log_s0_scalar.expand(n_paths)]
    v_init = v0 if v0.dim() > 0 else v0.expand(n_paths)
    v_steps: list[torch.Tensor] = [v_init]

    for i in range(n_steps):
        z_v = torch.randn(n_paths, dtype=dtype, device=device, generator=generator)
        z_perp = torch.randn(n_paths, dtype=dtype, device=device, generator=generator)
        z_s = rho * z_v + sqrt_1_rho2 * z_perp

        v_pos = v_steps[i].clamp(min=0.0)
        sqrt_v_pos = (v_pos + eps).sqrt()

        v_next = v_steps[i] + kappa * (theta - v_pos) * dt + xi * sqrt_v_pos * sqrt_dt * z_v
        log_s_next = log_s_steps[i] + (mu - 0.5 * v_pos) * dt + sqrt_v_pos * sqrt_dt * z_s
        v_steps.append(v_next)
        log_s_steps.append(log_s_next)

    log_s = torch.stack(log_s_steps, dim=0)
    v = torch.stack(v_steps, dim=0)
    return torch.exp(log_s), v
