"""Calibration-loss tests: identifiability, gradients, sanity properties."""
from __future__ import annotations

import math

import pytest
import torch

from stochastech.calibration.losses import energy_distance, nll_loss


def _gen(seed: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def test_energy_distance_zero_on_identical_samples() -> None:
    x = torch.randn(500, generator=_gen(0), dtype=torch.float64)
    ed = energy_distance(x, x)
    # Self-energy is exactly zero by construction.
    assert abs(ed.item()) < 1e-10


def test_energy_distance_small_on_same_distribution() -> None:
    x = torch.randn(2_000, generator=_gen(1), dtype=torch.float64)
    y = torch.randn(2_000, generator=_gen(2), dtype=torch.float64)
    ed = energy_distance(x, y).item()
    # Two independent N(0,1) samples: energy distance ~ O(1/sqrt(n)) and >= 0.
    assert ed >= -1e-6
    assert ed < 0.05


def test_energy_distance_large_on_shifted_distribution() -> None:
    x = torch.randn(2_000, generator=_gen(3), dtype=torch.float64)
    y = torch.randn(2_000, generator=_gen(4), dtype=torch.float64) + 2.0
    ed = energy_distance(x, y).item()
    assert ed > 0.5


def test_energy_distance_differentiable() -> None:
    s = torch.randn(200, generator=_gen(0), dtype=torch.float64, requires_grad=True)
    o = torch.randn(200, generator=_gen(1), dtype=torch.float64) + 0.5
    ed = energy_distance(s, o)
    ed.backward()
    assert s.grad is not None
    assert torch.isfinite(s.grad).all()
    assert s.grad.abs().sum().item() > 0


def test_energy_distance_decreases_when_pulling_simulated_toward_observed() -> None:
    o = torch.randn(500, generator=_gen(5), dtype=torch.float64) + 2.0
    s = torch.randn(500, generator=_gen(6), dtype=torch.float64).requires_grad_(True)
    ed_before = energy_distance(s, o).item()

    # Single Adam step targeting the energy distance should move s toward o.
    opt = torch.optim.Adam([s], lr=0.1)
    for _ in range(20):
        opt.zero_grad()
        loss = energy_distance(s, o)
        loss.backward()
        opt.step()
    ed_after = energy_distance(s, o).item()
    assert ed_after < ed_before


def test_energy_distance_rejects_tiny_samples() -> None:
    with pytest.raises(ValueError):
        energy_distance(torch.tensor([1.0]), torch.tensor([1.0, 2.0]))
    with pytest.raises(ValueError):
        energy_distance(torch.tensor([1.0, 2.0]), torch.tensor([1.0]))


def test_nll_finite_on_overlapping_samples() -> None:
    s = torch.randn(1_000, generator=_gen(0), dtype=torch.float64)
    o = torch.randn(500, generator=_gen(1), dtype=torch.float64)
    nll = nll_loss(s, o).item()
    # Per-sample NLL of N(0,1) is ~ 0.5 log(2 pi e) ≈ 1.42 nats.
    assert math.isfinite(nll)
    assert 1.0 < nll < 2.5


def test_nll_higher_for_shifted_observations() -> None:
    s = torch.randn(2_000, generator=_gen(0), dtype=torch.float64)
    o_close = torch.randn(500, generator=_gen(1), dtype=torch.float64)
    o_far = torch.randn(500, generator=_gen(2), dtype=torch.float64) + 5.0
    nll_close = nll_loss(s, o_close).item()
    nll_far = nll_loss(s, o_far).item()
    assert nll_far > nll_close + 1.0


def test_nll_differentiable_in_simulated() -> None:
    s = torch.randn(500, generator=_gen(0), dtype=torch.float64, requires_grad=True)
    o = torch.randn(200, generator=_gen(1), dtype=torch.float64)
    nll = nll_loss(s, o)
    nll.backward()
    assert s.grad is not None
    assert torch.isfinite(s.grad).all()
    assert s.grad.abs().sum().item() > 0


def test_nll_rejects_too_few_simulated() -> None:
    with pytest.raises(ValueError):
        nll_loss(torch.tensor([1.0]), torch.tensor([1.0, 2.0]))
