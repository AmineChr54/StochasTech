"""Value-at-Risk and Expected Shortfall estimators.

Sign convention: input ``returns`` is signed (positive = profit). Loss is
``-returns``. VaR and ES are reported as positive numbers when the tail is
on the loss side. See ``doc/math/06_monte_carlo_var.md``.
"""
from __future__ import annotations

import math

import torch


def _validate_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")


def _tail_index(alpha: float, n: int) -> int:
    if n < 1:
        raise ValueError(f"Need at least 1 sample, got n={n}.")
    # Round-trip via ceil with a tiny epsilon tolerates FP slop at integer boundaries
    # (e.g. (1 - 0.95) * 20 = 1.0000…001 in IEEE-754 and would naively yield k=2).
    k = math.ceil((1.0 - alpha) * n - 1e-9)
    if k < 1:
        raise ValueError(
            f"Sample too small for alpha={alpha}: need ceil((1-alpha)*n) >= 1, "
            f"got n={n}, k={k}."
        )
    return k


def value_at_risk(returns: torch.Tensor, alpha: float = 0.95) -> torch.Tensor:
    """Empirical VaR at level ``alpha`` from a sample of returns.

    Loss is ``-returns``; VaR is the ``(1 - alpha)`` quantile of the loss
    distribution, equivalently ``-R_{(k)}`` for ``k = ceil((1 - alpha) * n)``
    on the ascending order statistic of returns.
    """
    _validate_alpha(alpha)
    flat = returns.flatten()
    n = flat.numel()
    k = _tail_index(alpha, n)
    sorted_r, _ = torch.sort(flat)
    return -sorted_r[k - 1]


def expected_shortfall(returns: torch.Tensor, alpha: float = 0.95) -> torch.Tensor:
    """Empirical ES at level ``alpha`` — mean loss in the worst ``(1-alpha)`` tail."""
    _validate_alpha(alpha)
    flat = returns.flatten()
    n = flat.numel()
    k = _tail_index(alpha, n)
    sorted_r, _ = torch.sort(flat)
    return -sorted_r[:k].mean()


def gbm_mle_var_forecast(
    log_returns: torch.Tensor,
    alpha: float = 0.95,
    horizon: int = 1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """One-step (or h-step) VaR forecast under closed-form GBM-MLE.

    Estimates per-step mean ``m_hat`` and variance ``v_hat`` of log-returns
    assuming the increments are iid Gaussian (constant-vol GBM). The h-step
    log-return is then ``N(m_hat * h, v_hat * h)`` and

        VaR_alpha = sqrt(v_hat * h) * Phi^{-1}(alpha) - m_hat * h.

    Returns the positive VaR forecast plus the fitted moments.
    """
    _validate_alpha(alpha)
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    r = log_returns.flatten()
    if r.numel() < 2:
        raise ValueError("need at least 2 returns to estimate variance")
    m = r.mean()
    v = r.var(unbiased=True)
    m_h = m * horizon
    v_h = v * horizon
    z_alpha = float(torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(alpha)))
    sd_h = v_h.sqrt()
    var_value = sd_h * z_alpha - m_h
    return var_value, {
        "mu_per_step": float(m.item()),
        "sigma2_per_step": float(v.item()),
    }


def historical_var_forecast(
    log_returns: torch.Tensor,
    alpha: float = 0.95,
) -> torch.Tensor:
    """Historical-simulation VaR — empirical (1-alpha) quantile of past returns."""
    return value_at_risk(log_returns, alpha=alpha)


def heston_var_forecast(
    fitted_params,  # HestonParams; avoid circular import.
    dt: float,
    alpha: float = 0.95,
    n_paths: int = 20_000,
    horizon: int = 1,
    s0: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """One-step VaR forecast from simulated Heston paths.

    Simulates ``n_paths`` Heston trajectories of length ``horizon`` from the
    fitted parameters and reports the empirical VaR of the cumulative log-return.
    """
    _validate_alpha(alpha)
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    from stochastech.sde.heston import simulate_heston

    s, _ = simulate_heston(
        s0=s0,
        v0=float(fitted_params.v0.item()),
        mu=float(fitted_params.mu.item()),
        kappa=float(fitted_params.kappa.item()),
        theta=float(fitted_params.theta.item()),
        xi=float(fitted_params.xi.item()),
        rho=float(fitted_params.rho.item()),
        dt=dt, n_steps=horizon, n_paths=n_paths,
        generator=generator,
    )
    cumulative_log_return = (s[-1] / s[0]).log()
    return value_at_risk(cumulative_log_return, alpha=alpha)


def antithetic_normals(
    n: int,
    n_paths: int,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Draw ``n_paths`` antithetic standard-normal samples of length ``n``.

    Returns a ``(n, n_paths)`` tensor whose columns split into two halves
    ``Z`` and ``-Z`` (the second half mirrors the first). For odd
    ``n_paths`` the final column is independent. Antithetic variates reduce
    variance for monotone-in-Z integrands; see ``doc/math/06_monte_carlo_var.md``.
    """
    if n_paths < 1:
        raise ValueError(f"n_paths must be >= 1, got {n_paths}")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    half = n_paths // 2
    base = torch.randn(n, half, dtype=dtype, device=device, generator=generator)
    pieces = [base, -base]
    if n_paths % 2 == 1:
        extra = torch.randn(n, 1, dtype=dtype, device=device, generator=generator)
        pieces.append(extra)
    return torch.cat(pieces, dim=1)
