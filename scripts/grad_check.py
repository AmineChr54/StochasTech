"""Gradient-check for the Heston BPTT calibrator.

Compares the autograd gradient produced by ``simulate_heston_diff`` +
``energy_distance`` against a central-difference estimate, with both sides
sharing the same Brownian path (fixed RNG seed). Prints a table of
``autograd | finite-difference | rel-error`` per Heston parameter.

Usage:
    pixi run -e dev python scripts/grad_check.py
    pixi run -e dev python scripts/grad_check.py --n-paths 256 --seed 42
"""
from __future__ import annotations

import argparse
import math

import torch

from stochastech.calibration.heston_fit import HestonParams, heston_loss_and_grad
from stochastech.calibration.losses import energy_distance
from stochastech.sde.heston import simulate_heston, simulate_heston_diff


def _gen(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _truth_returns(seed: int, n_steps: int) -> torch.Tensor:
    s, _ = simulate_heston(
        s0=1.0, v0=0.04, mu=0.0, kappa=2.0, theta=0.04, xi=0.3, rho=-0.5,
        dt=1 / 252, n_steps=n_steps, n_paths=1, generator=_gen(seed),
    )
    return (s[1:, 0] / s[:-1, 0]).log()


def _make_params(values: dict[str, float]) -> HestonParams:
    return HestonParams(**{
        k: torch.tensor(v, dtype=torch.float64) for k, v in values.items()
    })


def _loss_at(values: dict[str, float], obs: torch.Tensor, n_paths: int,
             n_steps: int, dt: float, brownian_seed: int, subsample_seed: int,
             max_samples: int) -> float:
    p = _make_params(values)
    s, _ = simulate_heston_diff(
        s0=1.0, v0=p.v0, mu=p.mu, kappa=p.kappa, theta=p.theta, xi=p.xi, rho=p.rho,
        dt=dt, n_steps=n_steps, n_paths=n_paths, generator=_gen(brownian_seed),
    )
    sim = (s[1:] / s[:-1]).log().flatten()
    if sim.numel() > max_samples:
        idx = torch.randperm(sim.numel(), generator=_gen(subsample_seed))[:max_samples]
        sim = sim[idx]
    return energy_distance(sim, obs).item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-paths", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=99,
                        help="Brownian seed shared by autograd and FD evaluations.")
    parser.add_argument("--data-seed", type=int, default=4)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--max-samples", type=int, default=512)
    args = parser.parse_args()

    obs = _truth_returns(seed=args.data_seed, n_steps=args.n_steps)
    base = dict(mu=0.0, kappa=2.5, theta=0.05, xi=0.32, rho=-0.55, v0=0.05)

    # Autograd grads via heston_loss_and_grad — but it draws fresh randomness internally.
    # For a fair FD comparison we instead inline an autograd pass with a fixed seed.
    p = _make_params(base)
    for f in ("mu", "kappa", "theta", "xi", "rho", "v0"):
        getattr(p, f).requires_grad_(True)

    s, _ = simulate_heston_diff(
        s0=1.0, v0=p.v0, mu=p.mu, kappa=p.kappa, theta=p.theta, xi=p.xi, rho=p.rho,
        dt=1 / 252, n_steps=args.n_steps, n_paths=args.n_paths,
        generator=_gen(args.seed),
    )
    sim = (s[1:] / s[:-1]).log().flatten()
    subsample_seed = args.seed + 7
    if sim.numel() > args.max_samples:
        idx = torch.randperm(sim.numel(), generator=_gen(subsample_seed))[:args.max_samples]
        sim = sim[idx]
    loss = energy_distance(sim, obs)
    loss.backward()

    autograd_grads = {f: float(getattr(p, f).grad) for f in
                      ("mu", "kappa", "theta", "xi", "rho", "v0")}

    # Central-difference grads, same Brownian + same subsample.
    fd_grads = {}
    for f in autograd_grads:
        plus = base.copy()
        minus = base.copy()
        plus[f] = base[f] + args.eps
        minus[f] = base[f] - args.eps
        l_plus = _loss_at(plus, obs, args.n_paths, args.n_steps, 1 / 252,
                          args.seed, subsample_seed, args.max_samples)
        l_minus = _loss_at(minus, obs, args.n_paths, args.n_steps, 1 / 252,
                           args.seed, subsample_seed, args.max_samples)
        fd_grads[f] = (l_plus - l_minus) / (2 * args.eps)

    print(f"{'param':<8} {'autograd':>14} {'finite-diff':>14} {'abs-err':>12} {'rel-err':>10}")
    print("-" * 62)
    worst_rel = 0.0
    for f in autograd_grads:
        a = autograd_grads[f]
        d = fd_grads[f]
        abs_err = abs(a - d)
        rel_err = abs_err / max(abs(d), 1e-8)
        worst_rel = max(worst_rel, rel_err)
        print(f"{f:<8} {a:>14.6e} {d:>14.6e} {abs_err:>12.3e} {rel_err:>10.2%}")
    print("-" * 62)
    print(f"loss at base: {loss.item():.6e}")
    print(f"worst relative error: {worst_rel:.2%}")
    if not all(math.isfinite(v) for v in autograd_grads.values()):
        raise SystemExit("autograd produced non-finite gradients")
    if worst_rel > 0.20:
        raise SystemExit(f"gradient check failed: worst rel error {worst_rel:.2%} > 20%")
    print("OK: autograd matches central differences within 20%.")


if __name__ == "__main__":
    main()
