"""Differentiable Heston calibration via the stochastic adjoint.

See ``doc/math/07_adjoint_sde.md``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from stochastech.calibration.losses import energy_distance, nll_loss
from stochastech.sde.heston import simulate_heston_diff


@dataclass
class HestonParams:
    mu: torch.Tensor
    kappa: torch.Tensor
    theta: torch.Tensor
    xi: torch.Tensor
    rho: torch.Tensor
    v0: torch.Tensor


@dataclass
class _RawParams:
    """Unconstrained leaf tensors backing the optimizer.

    ``log_*`` fields parameterize positives via ``exp``; ``rho_raw`` parameterizes
    correlation via ``tanh``. Initial values are the inverse-transform of the
    user-supplied ``HestonParams``.
    """
    mu: torch.Tensor
    log_kappa: torch.Tensor
    log_theta: torch.Tensor
    log_xi: torch.Tensor
    log_v0: torch.Tensor
    rho_raw: torch.Tensor

    @classmethod
    def from_constrained(cls, p: HestonParams, dtype: torch.dtype) -> _RawParams:
        def t(x: torch.Tensor) -> torch.Tensor:
            return x.detach().to(dtype=dtype).clone()

        rho_clamped = t(p.rho).clamp(-0.999, 0.999)
        return cls(
            mu=t(p.mu).requires_grad_(True),
            log_kappa=torch.log(t(p.kappa).clamp(min=1e-8)).requires_grad_(True),
            log_theta=torch.log(t(p.theta).clamp(min=1e-8)).requires_grad_(True),
            log_xi=torch.log(t(p.xi).clamp(min=1e-8)).requires_grad_(True),
            log_v0=torch.log(t(p.v0).clamp(min=1e-8)).requires_grad_(True),
            rho_raw=torch.atanh(rho_clamped).requires_grad_(True),
        )

    def constrained(self) -> HestonParams:
        return HestonParams(
            mu=self.mu,
            kappa=self.log_kappa.exp(),
            theta=self.log_theta.exp(),
            xi=self.log_xi.exp(),
            rho=torch.tanh(self.rho_raw),
            v0=self.log_v0.exp(),
        )

    def leaves(self) -> list[torch.Tensor]:
        return [self.mu, self.log_kappa, self.log_theta, self.log_xi,
                self.log_v0, self.rho_raw]


def _simulate_log_returns(
    params: HestonParams,
    s0: float,
    dt: float,
    n_steps: int,
    n_paths: int,
    generator: torch.Generator | None,
    dtype: torch.dtype,
    device: torch.device | str,
    max_samples: int | None = None,
) -> torch.Tensor:
    s, _ = simulate_heston_diff(
        s0=s0, v0=params.v0, mu=params.mu, kappa=params.kappa, theta=params.theta,
        xi=params.xi, rho=params.rho, dt=dt, n_steps=n_steps, n_paths=n_paths,
        generator=generator, dtype=dtype, device=device,
    )
    flat = (s[1:] / s[:-1]).log().flatten()
    if max_samples is not None and flat.numel() > max_samples:
        # Energy distance / NLL form $O(n^2)$ pairwise matrices — cap the sample
        # size to keep the loss computation tractable. Random subsampling keeps
        # the U-statistic estimator unbiased.
        idx = torch.randperm(flat.numel(), generator=generator, device=flat.device)[:max_samples]
        flat = flat[idx]
    return flat


def _loss_fn(
    name: str, simulated: torch.Tensor, observed: torch.Tensor
) -> torch.Tensor:
    if name == "energy":
        return energy_distance(simulated, observed)
    if name == "nll":
        return nll_loss(simulated, observed)
    raise ValueError(f"Unknown loss '{name}'; choose 'energy' or 'nll'.")


def fit_heston(
    returns: torch.Tensor,
    dt: float,
    init: HestonParams,
    loss: str = "energy",
    n_paths: int = 4096,
    n_iters: int = 500,
    lr: float = 1e-2,
    s0: float = 1.0,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
    max_samples: int | None = 2048,
) -> tuple[HestonParams, list[float]]:
    """Fit Heston parameters to historical returns by Adam through the SDE.

    The Heston SDE is differentiated via backprop-through-time on the
    full-truncation Euler simulator (see ``simulate_heston_diff`` in
    ``stochastech/sde/heston.py``). For Heston's small parameter dimension
    ($|\\theta| = 6$ here) BPTT is competitive in wall-clock with the stochastic
    adjoint and far simpler to keep numerically stable; see
    ``doc/math/07_adjoint_sde.md`` for the adjoint theory and the trade-off.

    Returns the fitted parameters and the per-iteration loss history.
    """
    obs = returns.flatten().to(dtype=dtype, device=device)
    raw = _RawParams.from_constrained(init, dtype=dtype)
    optimizer = torch.optim.Adam(raw.leaves(), lr=lr)
    n_steps = obs.numel()

    history: list[float] = []
    for _ in range(n_iters):
        optimizer.zero_grad()
        params = raw.constrained()
        sim_returns = _simulate_log_returns(
            params, s0, dt, n_steps, n_paths, generator, dtype, device, max_samples,
        )
        loss_value = _loss_fn(loss, sim_returns, obs)
        loss_value.backward()
        optimizer.step()
        history.append(float(loss_value.detach()))

    final = raw.constrained()
    return HestonParams(
        mu=final.mu.detach(),
        kappa=final.kappa.detach(),
        theta=final.theta.detach(),
        xi=final.xi.detach(),
        rho=final.rho.detach(),
        v0=final.v0.detach(),
    ), history


def heston_loss_and_grad(
    returns: torch.Tensor,
    dt: float,
    params: HestonParams,
    loss: str = "energy",
    n_paths: int = 1024,
    s0: float = 1.0,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
    max_samples: int | None = 2048,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """One-shot loss + per-parameter gradient for the gradient-check script.

    ``params`` must contain leaf tensors with ``requires_grad=True`` set on the
    constrained values; the function calls ``.backward()`` and returns the
    populated ``.grad`` for each of ``mu, kappa, theta, xi, rho, v0``.
    """
    obs = returns.flatten().to(dtype=dtype, device=device)
    n_steps = obs.numel()
    sim_returns = _simulate_log_returns(
        params, s0, dt, n_steps, n_paths, generator, dtype, device, max_samples,
    )
    loss_value = _loss_fn(loss, sim_returns, obs)
    loss_value.backward()
    grads = {
        "mu": params.mu.grad,
        "kappa": params.kappa.grad,
        "theta": params.theta.grad,
        "xi": params.xi.grad,
        "rho": params.rho.grad,
        "v0": params.v0.grad,
    }
    return loss_value.detach(), grads


def fit_heston_gradient_free(
    returns: torch.Tensor,
    dt: float,
    init: HestonParams,
    loss: str = "energy",
    n_paths: int = 1024,
    s0: float = 1.0,
    seed: int = 0,
    method: str = "Nelder-Mead",
    maxiter: int = 200,
    max_samples: int | None = 2048,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> tuple[HestonParams, dict]:
    """Gradient-free Heston fit via ``scipy.optimize.minimize``.

    Baseline for the BPTT calibrator; the comparison sanity-checks that the
    differentiable approach is at least as good as a derivative-free optimizer
    on the same objective. The optimizer works in the same unconstrained
    ``(mu, log_kappa, log_theta, log_xi, log_v0, rho_raw)`` space as
    ``fit_heston``; the simulation seed is fixed across function evaluations to
    keep the objective deterministic in $\\theta$ (common-random-numbers).
    """
    from scipy.optimize import minimize

    obs = returns.flatten().to(dtype=dtype, device=device)
    n_steps = obs.numel()
    raw0 = _RawParams.from_constrained(init, dtype=dtype)
    x0 = torch.stack([
        raw0.mu, raw0.log_kappa, raw0.log_theta,
        raw0.log_xi, raw0.log_v0, raw0.rho_raw,
    ]).detach().cpu().numpy()

    def objective(x) -> float:
        params = HestonParams(
            mu=torch.tensor(float(x[0]), dtype=dtype, device=device),
            kappa=torch.tensor(float(x[1]), dtype=dtype, device=device).exp(),
            theta=torch.tensor(float(x[2]), dtype=dtype, device=device).exp(),
            xi=torch.tensor(float(x[3]), dtype=dtype, device=device).exp(),
            v0=torch.tensor(float(x[4]), dtype=dtype, device=device).exp(),
            rho=torch.tanh(torch.tensor(float(x[5]), dtype=dtype, device=device)),
        )
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        with torch.no_grad():
            sim_returns = _simulate_log_returns(
                params, s0, dt, n_steps, n_paths, gen, dtype, device, max_samples,
            )
            loss_value = _loss_fn(loss, sim_returns, obs)
        return float(loss_value.item())

    result = minimize(objective, x0, method=method, options={"maxiter": maxiter})
    x = result.x
    fitted = HestonParams(
        mu=torch.tensor(float(x[0]), dtype=dtype, device=device),
        kappa=torch.tensor(float(x[1]), dtype=dtype, device=device).exp(),
        theta=torch.tensor(float(x[2]), dtype=dtype, device=device).exp(),
        xi=torch.tensor(float(x[3]), dtype=dtype, device=device).exp(),
        v0=torch.tensor(float(x[4]), dtype=dtype, device=device).exp(),
        rho=torch.tanh(torch.tensor(float(x[5]), dtype=dtype, device=device)),
    )
    diagnostics = {
        "method": method,
        "loss_final": float(result.fun),
        "n_iterations": int(result.nit) if hasattr(result, "nit") else None,
        "n_function_evals": int(result.nfev) if hasattr(result, "nfev") else None,
        "success": bool(result.success),
        "message": str(result.message),
    }
    return fitted, diagnostics


def _params_to_dict(p: HestonParams) -> dict[str, float]:
    return {k: float(v.detach().cpu().item()) for k, v in asdict(p).items()}


def rolling_window_calibration(
    returns: torch.Tensor,
    dt: float,
    init: HestonParams,
    window: int,
    step: int | None = None,
    loss: str = "energy",
    n_paths: int = 1024,
    n_iters: int = 200,
    lr: float = 1e-2,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
    method: str = "bptt",
) -> list[dict]:
    """Run rolling-window Heston calibration over a return series.

    For each window starting at indices ``0, step, 2 step, ...`` (default
    ``step = window``, i.e. non-overlapping), fit Heston on
    ``returns[start : start + window]`` and emit a record with the fitted
    parameters and final loss. The previous window's fit seeds the next
    window's init — warm-start gives smoother parameter trajectories and
    converges faster than re-initializing from scratch.

    ``method='bptt'`` uses Adam through the differentiable simulator (default);
    ``method='gradient_free'`` uses the scipy baseline.
    """
    if window < 8:
        raise ValueError(f"window must be >= 8, got {window}")
    if step is None:
        step = window
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")

    obs = returns.flatten().to(dtype=dtype, device=device)
    n = obs.numel()
    if n < window:
        raise ValueError(f"need at least {window} returns, got {n}")

    records: list[dict] = []
    current_init = init
    starts = list(range(0, n - window + 1, step))
    for win_idx, start in enumerate(starts):
        seg = obs[start:start + window]
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed + win_idx)

        if method == "bptt":
            fitted, history = fit_heston(
                returns=seg, dt=dt, init=current_init, loss=loss,
                n_paths=n_paths, n_iters=n_iters, lr=lr,
                generator=gen, dtype=dtype, device=device,
            )
            final_loss = history[-1]
            extra = {"loss_initial": history[0], "n_iters": len(history)}
        elif method == "gradient_free":
            fitted, diag = fit_heston_gradient_free(
                returns=seg, dt=dt, init=current_init, loss=loss,
                n_paths=n_paths, seed=seed + win_idx, maxiter=n_iters,
                dtype=dtype, device=device,
            )
            final_loss = diag["loss_final"]
            extra = {"n_function_evals": diag["n_function_evals"], "success": diag["success"]}
        else:
            raise ValueError(f"Unknown method '{method}'; use 'bptt' or 'gradient_free'.")

        records.append({
            "window_idx": win_idx,
            "start": int(start),
            "end": int(start + window),
            "loss_final": float(final_loss),
            "params": _params_to_dict(fitted),
            **extra,
        })
        current_init = fitted

    return records
