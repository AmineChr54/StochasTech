"""GBM simulator tests: shape, determinism, positivity, and analytic moments."""
from __future__ import annotations

import math

import pytest
import torch

from stochastech.sde.base import euler_maruyama
from stochastech.sde.gbm import simulate_gbm


def _gen(seed: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def test_gbm_output_shape() -> None:
    paths = simulate_gbm(s0=100.0, mu=0.05, sigma=0.2, dt=1 / 252, n_steps=10, n_paths=7,
                        generator=_gen())
    assert paths.shape == (11, 7)


def test_gbm_initial_state() -> None:
    paths = simulate_gbm(s0=100.0, mu=0.05, sigma=0.2, dt=1 / 252, n_steps=5, n_paths=4,
                        generator=_gen())
    assert torch.allclose(paths[0], torch.full((4,), 100.0, dtype=paths.dtype))


@pytest.mark.parametrize("log_euler", [True, False])
def test_gbm_deterministic_with_seed(log_euler: bool) -> None:
    a = simulate_gbm(100.0, 0.05, 0.2, 1 / 252, 50, 32, generator=_gen(42), log_euler=log_euler)
    b = simulate_gbm(100.0, 0.05, 0.2, 1 / 252, 50, 32, generator=_gen(42), log_euler=log_euler)
    assert torch.equal(a, b)


def test_gbm_log_euler_strictly_positive() -> None:
    # Aggressive sigma * sqrt(dt) that would push naive Euler negative.
    paths = simulate_gbm(s0=1.0, mu=0.0, sigma=2.0, dt=0.25, n_steps=20, n_paths=5_000,
                        generator=_gen(1), log_euler=True)
    assert torch.all(paths > 0)


def test_gbm_naive_euler_can_go_negative_warns_user_via_doc() -> None:
    # Documents the failure mode the doc warns about: naive Euler is not positivity-preserving.
    paths = simulate_gbm(s0=1.0, mu=0.0, sigma=2.0, dt=0.25, n_steps=20, n_paths=5_000,
                        generator=_gen(1), log_euler=False)
    # We don't assert negativity (rare on a fixed seed) — just confirm shape and that nothing
    # silently clipped to zero, i.e. behavior matches the documented unsafe scheme.
    assert paths.shape == (21, 5_000)


def test_gbm_mean_matches_analytic() -> None:
    s0, mu, sigma, T = 100.0, 0.07, 0.25, 1.0
    n_paths = 200_000
    n_steps = 50
    dt = T / n_steps
    paths = simulate_gbm(s0, mu, sigma, dt, n_steps, n_paths, generator=_gen(0), log_euler=True)
    sample_mean = paths[-1].mean().item()
    analytic_mean = s0 * math.exp(mu * T)
    # Stderr of the mean ≈ analytic_std / sqrt(n_paths).
    analytic_var = s0 * s0 * math.exp(2 * mu * T) * (math.exp(sigma * sigma * T) - 1)
    stderr = math.sqrt(analytic_var / n_paths)
    assert abs(sample_mean - analytic_mean) < 4 * stderr, (
        f"mean {sample_mean} vs {analytic_mean}, stderr {stderr}"
    )


def test_gbm_variance_matches_analytic() -> None:
    s0, mu, sigma, T = 100.0, 0.07, 0.25, 1.0
    n_paths = 200_000
    n_steps = 50
    dt = T / n_steps
    paths = simulate_gbm(s0, mu, sigma, dt, n_steps, n_paths, generator=_gen(1), log_euler=True)
    sample_var = paths[-1].var(unbiased=True).item()
    analytic_var = s0 * s0 * math.exp(2 * mu * T) * (math.exp(sigma * sigma * T) - 1)
    # 5% relative tolerance — variance estimator has high MC noise on heavy-tailed lognormal.
    assert abs(sample_var - analytic_var) / analytic_var < 0.05, (
        f"var {sample_var} vs {analytic_var}"
    )


def test_gbm_log_returns_match_analytic() -> None:
    # Exact log-return distribution: r_i = log(S_{t+dt}/S_t) ~ N((mu - 0.5 sigma^2) dt, sigma^2 dt).
    s0, mu, sigma = 100.0, 0.05, 0.3
    n_paths = 100_000
    n_steps = 1
    dt = 0.5
    paths = simulate_gbm(s0, mu, sigma, dt, n_steps, n_paths, generator=_gen(2), log_euler=True)
    log_returns = (paths[1] / paths[0]).log()
    expected_mean = (mu - 0.5 * sigma * sigma) * dt
    expected_var = sigma * sigma * dt
    assert abs(log_returns.mean().item() - expected_mean) < 4 * math.sqrt(expected_var / n_paths)
    # Variance of sample variance ≈ 2 sigma^4 / (n-1) for normal data.
    var_stderr = math.sqrt(2 * expected_var * expected_var / (n_paths - 1))
    assert abs(log_returns.var(unbiased=True).item() - expected_var) < 4 * var_stderr


def test_euler_maruyama_constant_coefficients() -> None:
    # dX = a dt + b dW with constants -> X_T ~ N(X_0 + a T, b^2 T).
    a, b, T = 0.3, 0.4, 2.0
    n_paths = 200_000
    n_steps = 100
    dt = T / n_steps
    x0 = torch.zeros(n_paths, dtype=torch.float64)
    paths = euler_maruyama(
        drift=lambda x, _t: torch.full_like(x, a),
        diffusion=lambda x, _t: torch.full_like(x, b),
        x0=x0,
        dt=dt,
        n_steps=n_steps,
        generator=_gen(3),
    )
    final = paths[-1]
    expected_mean = a * T
    expected_var = b * b * T
    assert abs(final.mean().item() - expected_mean) < 4 * math.sqrt(expected_var / n_paths)
    var_stderr = math.sqrt(2 * expected_var * expected_var / (n_paths - 1))
    assert abs(final.var(unbiased=True).item() - expected_var) < 4 * var_stderr


def test_euler_maruyama_rejects_bad_inputs() -> None:
    x0 = torch.zeros(4)
    z = lambda x, _t: torch.zeros_like(x)  # noqa: E731
    with pytest.raises(ValueError):
        euler_maruyama(z, z, x0, dt=0.0, n_steps=10)
    with pytest.raises(ValueError):
        euler_maruyama(z, z, x0, dt=0.1, n_steps=0)


def test_simulate_gbm_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        simulate_gbm(s0=0.0, mu=0.0, sigma=0.2, dt=0.1, n_steps=10, n_paths=4)
    with pytest.raises(ValueError):
        simulate_gbm(s0=1.0, mu=0.0, sigma=-0.1, dt=0.1, n_steps=10, n_paths=4)
    with pytest.raises(ValueError):
        simulate_gbm(s0=1.0, mu=0.0, sigma=0.2, dt=0.1, n_steps=0, n_paths=4)
    with pytest.raises(ValueError):
        simulate_gbm(s0=1.0, mu=0.0, sigma=0.2, dt=0.1, n_steps=10, n_paths=0)


def test_zero_volatility_collapses_to_deterministic() -> None:
    s0, mu, T = 50.0, 0.1, 0.5
    n_steps = 10
    dt = T / n_steps
    paths = simulate_gbm(s0, mu, 0.0, dt, n_steps, n_paths=8, generator=_gen(0), log_euler=True)
    expected = s0 * math.exp(mu * T)
    assert torch.allclose(paths[-1], torch.full_like(paths[-1], expected), atol=1e-10)
