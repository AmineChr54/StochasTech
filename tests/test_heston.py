"""Heston full-truncation Euler simulator tests."""
from __future__ import annotations

import math

import pytest
import torch

from stochastech.sde.heston import feller_condition, simulate_heston


def _gen(seed: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def test_heston_output_shapes() -> None:
    s, v = simulate_heston(
        s0=100.0, v0=0.04, mu=0.05, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
        dt=1 / 252, n_steps=10, n_paths=8, generator=_gen(),
    )
    assert s.shape == (11, 8)
    assert v.shape == (11, 8)


def test_heston_initial_state() -> None:
    s, v = simulate_heston(
        s0=100.0, v0=0.04, mu=0.05, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
        dt=1 / 252, n_steps=5, n_paths=4, generator=_gen(),
    )
    assert torch.allclose(s[0], torch.full((4,), 100.0, dtype=s.dtype))
    assert torch.allclose(v[0], torch.full((4,), 0.04, dtype=v.dtype))


def test_heston_deterministic_with_seed() -> None:
    args = dict(s0=100.0, v0=0.04, mu=0.05, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
                dt=1 / 252, n_steps=20, n_paths=16)
    s1, v1 = simulate_heston(**args, generator=_gen(7))
    s2, v2 = simulate_heston(**args, generator=_gen(7))
    assert torch.equal(s1, s2)
    assert torch.equal(v1, v2)


def test_heston_price_strictly_positive() -> None:
    # Log-Euler on price preserves positivity even when raw v dips negative.
    s, _ = simulate_heston(
        s0=1.0, v0=0.04, mu=0.0, kappa=1.0, theta=0.04, xi=0.6, rho=-0.5,
        dt=1 / 52, n_steps=52, n_paths=2_000, generator=_gen(1),
    )
    assert torch.all(s > 0)


def test_heston_zero_xi_collapses_to_gbm() -> None:
    # xi=0 with v0=theta -> v_t == theta forever; price is GBM with sigma^2=theta.
    s0, v0, mu, T = 100.0, 0.04, 0.07, 1.0
    n_paths, n_steps = 50_000, 50
    dt = T / n_steps
    s, v = simulate_heston(
        s0=s0, v0=v0, mu=mu, kappa=1.0, theta=v0, xi=0.0, rho=0.0,
        dt=dt, n_steps=n_steps, n_paths=n_paths, generator=_gen(2),
    )
    # Variance trajectory must be identically v0.
    assert torch.allclose(v, torch.full_like(v, v0), atol=1e-12)
    # Terminal price moments match GBM with sigma^2 = v0.
    sigma2 = v0
    analytic_mean = s0 * math.exp(mu * T)
    analytic_var = s0 * s0 * math.exp(2 * mu * T) * (math.exp(sigma2 * T) - 1)
    sample_mean = s[-1].mean().item()
    stderr = math.sqrt(analytic_var / n_paths)
    assert abs(sample_mean - analytic_mean) < 4 * stderr


def test_heston_variance_long_run_mean() -> None:
    # On a long horizon variance should mean-revert to theta.
    s0, v0, theta = 100.0, 0.10, 0.04
    s, v = simulate_heston(
        s0=s0, v0=v0, mu=0.0, kappa=4.0, theta=theta, xi=0.3, rho=-0.5,
        dt=1 / 252, n_steps=2_500, n_paths=20_000, generator=_gen(3),
    )
    del s
    # Average over the last 1y of the simulation across paths and time.
    tail_mean = v[-252:].mean().item()
    assert abs(tail_mean - theta) / theta < 0.05, f"E[v] {tail_mean} vs theta {theta}"


def test_heston_leverage_sign() -> None:
    # rho < 0 -> negative correlation between log returns and dv. Use long history.
    rho = -0.8
    s, v = simulate_heston(
        s0=100.0, v0=0.04, mu=0.0, kappa=2.0, theta=0.04, xi=0.4, rho=rho,
        dt=1 / 252, n_steps=2_000, n_paths=4_000, generator=_gen(4),
    )
    log_returns = (s[1:] / s[:-1]).log().flatten()
    dv = (v[1:] - v[:-1]).flatten()
    # Pearson correlation.
    lr = log_returns - log_returns.mean()
    dvc = dv - dv.mean()
    corr = (lr * dvc).sum() / ((lr.pow(2).sum().sqrt()) * (dvc.pow(2).sum().sqrt()))
    assert corr.item() < -0.3, f"corr {corr.item()} should be strongly negative for rho={rho}"


def test_heston_full_truncation_keeps_negativity_rare_under_feller() -> None:
    # Feller-respecting params + small dt should keep negative excursions of v rare.
    kappa, theta, xi = 3.0, 0.04, 0.2
    assert feller_condition(kappa, theta, xi)
    _, v = simulate_heston(
        s0=100.0, v0=theta, mu=0.0, kappa=kappa, theta=theta, xi=xi, rho=-0.5,
        dt=1 / 252, n_steps=252, n_paths=5_000, generator=_gen(5),
    )
    neg_fraction = (v < 0).float().mean().item()
    assert neg_fraction < 0.01, f"negative-v fraction {neg_fraction} too high"


def test_feller_condition() -> None:
    assert feller_condition(2.0, 0.04, 0.2)            # 0.16 >= 0.04
    assert not feller_condition(0.5, 0.04, 0.5)        # 0.04 < 0.25
    assert feller_condition(1.0, 0.04, 0.2)            # 0.08 >= 0.04 (interior)
    assert not feller_condition(1.0, 0.04, 0.4)        # 0.08 < 0.16


def test_heston_rejects_bad_inputs() -> None:
    base = dict(s0=100.0, v0=0.04, mu=0.05, kappa=2.0, theta=0.04, xi=0.3, rho=-0.5,
                dt=1 / 252, n_steps=10, n_paths=4)
    with pytest.raises(ValueError):
        simulate_heston(**{**base, "s0": 0.0})
    with pytest.raises(ValueError):
        simulate_heston(**{**base, "v0": -0.01})
    with pytest.raises(ValueError):
        simulate_heston(**{**base, "kappa": -0.1})
    with pytest.raises(ValueError):
        simulate_heston(**{**base, "theta": -0.1})
    with pytest.raises(ValueError):
        simulate_heston(**{**base, "xi": -0.1})
    with pytest.raises(ValueError):
        simulate_heston(**{**base, "rho": 1.5})
    with pytest.raises(ValueError):
        simulate_heston(**{**base, "rho": -1.5})
    with pytest.raises(ValueError):
        simulate_heston(**{**base, "dt": 0.0})
    with pytest.raises(ValueError):
        simulate_heston(**{**base, "n_steps": 0})
    with pytest.raises(ValueError):
        simulate_heston(**{**base, "n_paths": 0})
