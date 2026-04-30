"""VaR / ES estimator tests."""
from __future__ import annotations

import math

import pytest
import torch

from stochastech.risk.var import antithetic_normals, expected_shortfall, value_at_risk


def _gen(seed: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def test_var_known_order_statistic() -> None:
    # Returns -10..-1, 0, 1..9 (n=20). For alpha=0.95, k=ceil(0.05*20)=1, R_(1) = -10.
    returns = torch.arange(-10, 10, dtype=torch.float64)
    assert value_at_risk(returns, alpha=0.95).item() == 10.0


def test_var_alpha_90_known() -> None:
    # n=20, alpha=0.9 -> k=2, R_(2) = -9 -> VaR = 9.
    returns = torch.arange(-10, 10, dtype=torch.float64)
    assert value_at_risk(returns, alpha=0.9).item() == 9.0


def test_es_known_order_statistic() -> None:
    # n=20, alpha=0.9, k=2: mean of {-10, -9} = -9.5, ES = 9.5.
    returns = torch.arange(-10, 10, dtype=torch.float64)
    assert expected_shortfall(returns, alpha=0.9).item() == 9.5


def test_var_matches_normal_quantile() -> None:
    # For X ~ N(0, sigma^2), VaR_alpha = sigma * Phi^{-1}(alpha) (loss = -X).
    sigma = 0.02
    n = 200_000
    z = torch.randn(n, generator=_gen(0), dtype=torch.float64) * sigma
    var_hat = value_at_risk(z, alpha=0.95).item()
    var_true = sigma * 1.6448536269514722  # Phi^{-1}(0.95)
    # Quantile estimator stderr: sqrt(alpha (1-alpha)/n) / f(F^{-1}(1-alpha)).
    f = math.exp(-1.6449**2 / 2) / math.sqrt(2 * math.pi) / sigma
    stderr = math.sqrt(0.95 * 0.05 / n) / f
    assert abs(var_hat - var_true) < 5 * stderr, f"{var_hat} vs {var_true}, stderr {stderr}"


def test_es_matches_normal_analytic() -> None:
    # ES_alpha for N(0, sigma^2) = sigma * phi(z_alpha) / (1 - alpha).
    sigma = 0.02
    alpha = 0.95
    n = 200_000
    z = torch.randn(n, generator=_gen(1), dtype=torch.float64) * sigma
    es_hat = expected_shortfall(z, alpha=alpha).item()
    z_alpha = 1.6448536269514722
    phi = math.exp(-z_alpha**2 / 2) / math.sqrt(2 * math.pi)
    es_true = sigma * phi / (1 - alpha)
    # Sanity: 5% relative tolerance.
    assert abs(es_hat - es_true) / es_true < 0.05, f"{es_hat} vs {es_true}"


def test_es_geq_var() -> None:
    # ES is the average of the worst tail, so ES >= VaR by construction.
    z = torch.randn(10_000, generator=_gen(2), dtype=torch.float64)
    var_hat = value_at_risk(z, 0.95)
    es_hat = expected_shortfall(z, 0.95)
    assert es_hat.item() >= var_hat.item() - 1e-10


def test_var_monotone_in_alpha() -> None:
    z = torch.randn(20_000, generator=_gen(3), dtype=torch.float64)
    v90 = value_at_risk(z, 0.90).item()
    v95 = value_at_risk(z, 0.95).item()
    v99 = value_at_risk(z, 0.99).item()
    assert v90 <= v95 <= v99


def test_var_rejects_bad_alpha() -> None:
    z = torch.randn(1_000)
    with pytest.raises(ValueError):
        value_at_risk(z, alpha=0.0)
    with pytest.raises(ValueError):
        value_at_risk(z, alpha=1.0)
    with pytest.raises(ValueError):
        expected_shortfall(z, alpha=-0.1)


def test_var_rejects_empty_sample() -> None:
    with pytest.raises(ValueError):
        value_at_risk(torch.empty(0), alpha=0.95)
    with pytest.raises(ValueError):
        expected_shortfall(torch.empty(0), alpha=0.95)


def test_var_works_on_2d_input() -> None:
    # Flatten convention: a (n_paths, n_horizons) tensor of returns is treated as one pool.
    z = torch.randn(1_000, 4, generator=_gen(4), dtype=torch.float64)
    v_flat = value_at_risk(z.flatten(), 0.95).item()
    v_2d = value_at_risk(z, 0.95).item()
    assert v_flat == v_2d


def test_antithetic_normals_mirror_structure() -> None:
    z = antithetic_normals(n=5, n_paths=8, generator=_gen(0))
    assert z.shape == (5, 8)
    # First 4 columns mirror last 4 (when n_paths even).
    assert torch.allclose(z[:, :4], -z[:, 4:8])


def test_antithetic_normals_odd_count_has_extra_column() -> None:
    z = antithetic_normals(n=3, n_paths=5, generator=_gen(0))
    assert z.shape == (3, 5)
    assert torch.allclose(z[:, :2], -z[:, 2:4])
    # Final column independent — no mirror.


def test_antithetic_normals_zero_mean_exactly() -> None:
    # Antithetic columns sum to zero by construction (even n_paths).
    z = antithetic_normals(n=10, n_paths=100, generator=_gen(0))
    assert torch.allclose(z.sum(dim=1), torch.zeros(10, dtype=z.dtype), atol=1e-12)


def test_antithetic_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        antithetic_normals(n=0, n_paths=10)
    with pytest.raises(ValueError):
        antithetic_normals(n=5, n_paths=0)
