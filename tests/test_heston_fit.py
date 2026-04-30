"""Differentiable Heston calibration tests.

The deliverable for Week 3 is a smoke-test that a single Adam step actually
moves the parameters and reduces the loss. A full mini-fit is also exercised
on synthetic data to confirm the optimizer can reduce the energy distance
toward zero.
"""
from __future__ import annotations

import torch

from stochastech.calibration.heston_fit import (
    HestonParams,
    fit_heston,
    heston_loss_and_grad,
)
from stochastech.sde.heston import simulate_heston, simulate_heston_diff


def _gen(seed: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _truth_returns(seed: int = 0, n_steps: int = 252) -> torch.Tensor:
    """Generate a return path under known Heston params to fit against."""
    s, _ = simulate_heston(
        s0=1.0, v0=0.04, mu=0.0, kappa=2.0, theta=0.04, xi=0.3, rho=-0.5,
        dt=1 / 252, n_steps=n_steps, n_paths=1, generator=_gen(seed),
    )
    return (s[1:, 0] / s[:-1, 0]).log()


def _params(**overrides: float) -> HestonParams:
    base = dict(mu=0.0, kappa=2.0, theta=0.04, xi=0.3, rho=-0.5, v0=0.04)
    base.update(overrides)
    return HestonParams(**{
        k: torch.tensor(v, dtype=torch.float64) for k, v in base.items()
    })


def test_simulate_heston_diff_matches_simulate_heston_at_same_seed() -> None:
    # Determinism: tensor-param simulator with the same seed/dt/n_steps must produce
    # the same path family as the float-param reference, modulo the eps-floor inside
    # sqrt(v_pos) which is < 1e-6 on the price scale.
    args = dict(s0=1.0, v0=0.04, mu=0.0, kappa=2.0, theta=0.04, xi=0.3, rho=-0.5,
                dt=1 / 252, n_steps=10, n_paths=4)
    s_ref, v_ref = simulate_heston(**args, generator=_gen(0))
    p = _params()
    s_diff, v_diff = simulate_heston_diff(
        s0=1.0, v0=p.v0, mu=p.mu, kappa=p.kappa, theta=p.theta, xi=p.xi, rho=p.rho,
        dt=args["dt"], n_steps=args["n_steps"], n_paths=args["n_paths"],
        generator=_gen(0),
    )
    assert torch.allclose(s_ref, s_diff, atol=1e-6, rtol=1e-6)
    assert torch.allclose(v_ref, v_diff, atol=1e-6, rtol=1e-6)


def test_grads_flow_through_simulator() -> None:
    p = _params()
    for f in ("mu", "kappa", "theta", "xi", "rho", "v0"):
        getattr(p, f).requires_grad_(True)
    s, _ = simulate_heston_diff(
        s0=1.0, v0=p.v0, mu=p.mu, kappa=p.kappa, theta=p.theta, xi=p.xi, rho=p.rho,
        dt=1 / 252, n_steps=20, n_paths=128, generator=_gen(0),
    )
    s[-1].mean().backward()
    for f in ("mu", "kappa", "theta", "xi", "rho", "v0"):
        g = getattr(p, f).grad
        assert g is not None, f"no grad for {f}"
        assert torch.isfinite(g).all(), f"non-finite grad for {f}"


def test_single_adam_step_moves_params() -> None:
    obs = _truth_returns(seed=1, n_steps=64)
    init = _params(kappa=1.0, theta=0.06, xi=0.4, rho=0.0, v0=0.06)
    init_snapshot = {f: getattr(init, f).clone() for f in ("kappa", "theta", "xi", "rho", "v0")}
    fitted, history = fit_heston(
        returns=obs, dt=1 / 252, init=init, loss="energy",
        n_paths=128, n_iters=1, lr=5e-2, generator=_gen(0),
    )
    moved = sum(
        1 for f in init_snapshot
        if not torch.allclose(getattr(fitted, f), init_snapshot[f])
    )
    assert moved >= 4, f"only {moved}/5 params moved after one Adam step"
    assert len(history) == 1


def test_mini_fit_reduces_loss() -> None:
    # Synthetic-data fit: 30 Adam steps should reduce the energy distance from a
    # deliberately wrong init toward the truth.
    obs = _truth_returns(seed=2, n_steps=126)
    init = _params(kappa=4.0, theta=0.10, xi=0.5, rho=0.3, v0=0.10)
    _, history = fit_heston(
        returns=obs, dt=1 / 252, init=init, loss="energy",
        n_paths=256, n_iters=30, lr=5e-2, generator=_gen(0),
    )
    early = sum(history[:5]) / 5
    late = sum(history[-5:]) / 5
    assert late < early, f"loss did not decrease: early={early}, late={late}"


def test_fit_heston_rejects_unknown_loss() -> None:
    obs = _truth_returns(seed=0, n_steps=8)
    init = _params()
    try:
        fit_heston(returns=obs, dt=1 / 252, init=init, loss="bogus",
                   n_paths=8, n_iters=1, lr=1e-2, generator=_gen(0))
    except ValueError:
        return
    raise AssertionError("expected ValueError on unknown loss name")


def test_heston_loss_and_grad_populates_all_params() -> None:
    obs = _truth_returns(seed=3, n_steps=32)
    p = _params(kappa=1.5, theta=0.05, xi=0.25, rho=-0.2, v0=0.05)
    for f in ("mu", "kappa", "theta", "xi", "rho", "v0"):
        getattr(p, f).requires_grad_(True)
    loss, grads = heston_loss_and_grad(
        returns=obs, dt=1 / 252, params=p, loss="energy",
        n_paths=256, generator=_gen(0),
    )
    assert torch.isfinite(loss)
    for f, g in grads.items():
        assert g is not None, f"no grad for {f}"
        assert torch.isfinite(g).all(), f"non-finite grad for {f}"


def test_finite_difference_matches_autograd_for_kappa() -> None:
    # Sanity check: BPTT gradient on the energy loss should agree in sign and
    # rough magnitude with a central-difference estimate. We use a fixed
    # generator so both sides see the same Brownian path.
    obs = _truth_returns(seed=4, n_steps=32)
    base_kwargs = dict(s0=1.0, mu=0.0, theta=0.04, xi=0.3, rho=-0.5, v0=0.04,
                       dt=1 / 252, n_steps=32, n_paths=32)

    def loss_at(kappa_val: float) -> float:
        p = HestonParams(
            mu=torch.tensor(base_kwargs["mu"], dtype=torch.float64),
            kappa=torch.tensor(kappa_val, dtype=torch.float64),
            theta=torch.tensor(base_kwargs["theta"], dtype=torch.float64),
            xi=torch.tensor(base_kwargs["xi"], dtype=torch.float64),
            rho=torch.tensor(base_kwargs["rho"], dtype=torch.float64),
            v0=torch.tensor(base_kwargs["v0"], dtype=torch.float64),
        )
        s, _ = simulate_heston_diff(
            s0=base_kwargs["s0"], v0=p.v0, mu=p.mu, kappa=p.kappa, theta=p.theta,
            xi=p.xi, rho=p.rho, dt=base_kwargs["dt"],
            n_steps=base_kwargs["n_steps"], n_paths=base_kwargs["n_paths"],
            generator=_gen(99),
        )
        sim_returns = (s[1:] / s[:-1]).log().flatten()
        # Subsample to keep the O(n^2) energy distance bounded.
        if sim_returns.numel() > 512:
            idx = torch.randperm(sim_returns.numel(), generator=_gen(7))[:512]
            sim_returns = sim_returns[idx]
        from stochastech.calibration.losses import energy_distance
        return energy_distance(sim_returns, obs).item()

    # Autograd gradient at kappa=2.5.
    kappa_t = torch.tensor(2.5, dtype=torch.float64, requires_grad=True)
    p = HestonParams(
        mu=torch.tensor(0.0, dtype=torch.float64),
        kappa=kappa_t,
        theta=torch.tensor(0.04, dtype=torch.float64),
        xi=torch.tensor(0.3, dtype=torch.float64),
        rho=torch.tensor(-0.5, dtype=torch.float64),
        v0=torch.tensor(0.04, dtype=torch.float64),
    )
    s, _ = simulate_heston_diff(
        s0=1.0, v0=p.v0, mu=p.mu, kappa=p.kappa, theta=p.theta, xi=p.xi, rho=p.rho,
        dt=1 / 252, n_steps=32, n_paths=32, generator=_gen(99),
    )
    sim_returns = (s[1:] / s[:-1]).log().flatten()
    if sim_returns.numel() > 512:
        idx = torch.randperm(sim_returns.numel(), generator=_gen(7))[:512]
        sim_returns = sim_returns[idx]
    from stochastech.calibration.losses import energy_distance
    energy_distance(sim_returns, obs).backward()
    autograd_grad = kappa_t.grad.item()

    # Central difference at the same kappa, same generator seed (same Brownian
    # path), with a step large enough to overcome float precision.
    eps = 1e-3
    fd_grad = (loss_at(2.5 + eps) - loss_at(2.5 - eps)) / (2 * eps)

    # Loose tolerance: BPTT and FD share the same Brownian draws so this is a
    # consistency check, not a Monte Carlo bound.
    assert abs(autograd_grad - fd_grad) < max(1e-2, 0.5 * abs(fd_grad)), (
        f"autograd {autograd_grad} vs FD {fd_grad}"
    )
