"""GBM-MLE / historical / Heston VaR forecaster tests."""
from __future__ import annotations

import math

import pytest
import torch

from stochastech.calibration.heston_fit import HestonParams
from stochastech.risk.var import (
    gbm_mle_var_forecast,
    heston_var_forecast,
    historical_var_forecast,
    value_at_risk,
)


def _gen(seed: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def test_gbm_mle_var_matches_normal_quantile() -> None:
    # Returns drawn from N(0, sigma^2 * dt) -> VaR_alpha = sigma * sqrt(dt) * Phi^{-1}(alpha).
    sigma = 0.02
    n = 50_000
    r = torch.randn(n, generator=_gen(0), dtype=torch.float64) * sigma
    var_value, fit = gbm_mle_var_forecast(r, alpha=0.95, horizon=1)
    var_true = sigma * 1.6448536269514722
    # Sample sigma stderr ~ sigma / sqrt(2n); VaR stderr ~ z_alpha * sigma / sqrt(2n).
    stderr = 1.6449 * sigma / math.sqrt(2 * n)
    assert abs(var_value.item() - var_true) < 5 * stderr
    # Fitted moments should match the data-generating moments.
    assert abs(fit["sigma2_per_step"] - sigma * sigma) / (sigma * sigma) < 0.05


def test_gbm_mle_var_horizon_scales_with_sqrt_h() -> None:
    sigma = 0.02
    r = torch.randn(20_000, generator=_gen(1), dtype=torch.float64) * sigma
    v1, _ = gbm_mle_var_forecast(r, alpha=0.95, horizon=1)
    v4, _ = gbm_mle_var_forecast(r, alpha=0.95, horizon=4)
    # 4-step VaR scales as sqrt(4) = 2x the 1-step (drift contribution near zero on this seed).
    assert abs(v4.item() / v1.item() - 2.0) < 0.05


def test_gbm_mle_var_rejects_bad_inputs() -> None:
    r = torch.randn(100)
    with pytest.raises(ValueError):
        gbm_mle_var_forecast(r, alpha=0.0)
    with pytest.raises(ValueError):
        gbm_mle_var_forecast(r, alpha=0.95, horizon=0)
    with pytest.raises(ValueError):
        gbm_mle_var_forecast(torch.tensor([0.01]), alpha=0.95)


def test_historical_var_forecast_equivalent_to_value_at_risk() -> None:
    r = torch.randn(2_000, generator=_gen(2), dtype=torch.float64) * 0.02
    a = historical_var_forecast(r, alpha=0.95)
    b = value_at_risk(r, alpha=0.95)
    assert torch.equal(a, b)


def test_heston_var_forecast_returns_positive_scalar() -> None:
    p = HestonParams(
        mu=torch.tensor(0.0, dtype=torch.float64),
        kappa=torch.tensor(2.0, dtype=torch.float64),
        theta=torch.tensor(0.04, dtype=torch.float64),
        xi=torch.tensor(0.3, dtype=torch.float64),
        rho=torch.tensor(-0.5, dtype=torch.float64),
        v0=torch.tensor(0.04, dtype=torch.float64),
    )
    var = heston_var_forecast(p, dt=1 / 252, alpha=0.95, n_paths=5_000,
                              horizon=1, generator=_gen(0))
    assert var.dim() == 0
    assert var.item() > 0


def test_heston_var_close_to_gbm_at_zero_xi() -> None:
    # Heston with xi=0 and v0=theta is GBM(sigma=sqrt(theta)). VaR forecasts should match.
    sigma2 = 0.04
    p = HestonParams(
        mu=torch.tensor(0.0, dtype=torch.float64),
        kappa=torch.tensor(2.0, dtype=torch.float64),
        theta=torch.tensor(sigma2, dtype=torch.float64),
        xi=torch.tensor(0.0, dtype=torch.float64),
        rho=torch.tensor(0.0, dtype=torch.float64),
        v0=torch.tensor(sigma2, dtype=torch.float64),
    )
    heston_var = heston_var_forecast(p, dt=1 / 252, alpha=0.95, n_paths=20_000,
                                     horizon=1, generator=_gen(0)).item()
    sigma = math.sqrt(sigma2)
    dt = 1 / 252
    # VaR analytic for log-return r ~ N(-0.5 sigma^2 dt, sigma^2 dt): r_(0.05) quantile = m - sigma*sqrt(dt)*z.
    # VaR = -((-0.5 sigma^2 dt) - sigma*sqrt(dt)*z) = sigma*sqrt(dt)*z + 0.5*sigma^2*dt.
    var_analytic = sigma * math.sqrt(dt) * 1.6448536269514722 + 0.5 * sigma2 * dt
    # 5% relative tolerance due to MC noise.
    assert abs(heston_var - var_analytic) / var_analytic < 0.05


def test_heston_var_rejects_bad_inputs() -> None:
    p = HestonParams(
        mu=torch.tensor(0.0), kappa=torch.tensor(2.0), theta=torch.tensor(0.04),
        xi=torch.tensor(0.3), rho=torch.tensor(-0.5), v0=torch.tensor(0.04),
    )
    with pytest.raises(ValueError):
        heston_var_forecast(p, dt=1 / 252, alpha=0.0)
    with pytest.raises(ValueError):
        heston_var_forecast(p, dt=1 / 252, alpha=0.95, horizon=0)
