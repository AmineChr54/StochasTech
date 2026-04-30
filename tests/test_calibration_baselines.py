"""Gradient-free baseline + rolling-window calibration tests."""
from __future__ import annotations

import torch

from stochastech.calibration.heston_fit import (
    HestonParams,
    fit_heston_gradient_free,
    rolling_window_calibration,
)
from stochastech.sde.heston import simulate_heston


def _gen(seed: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _truth_returns(n_steps: int = 252, seed: int = 0) -> torch.Tensor:
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


def test_gradient_free_returns_valid_params() -> None:
    obs = _truth_returns(n_steps=64, seed=0)
    init = _params(kappa=1.5, theta=0.05, xi=0.4, rho=-0.2, v0=0.05)
    fitted, diag = fit_heston_gradient_free(
        returns=obs, dt=1 / 252, init=init, loss="energy",
        n_paths=64, seed=0, maxiter=20,
    )
    # Constraints: positives stay positive, rho stays in (-1, 1).
    assert fitted.kappa.item() > 0
    assert fitted.theta.item() > 0
    assert fitted.xi.item() > 0
    assert fitted.v0.item() > 0
    assert -1.0 < fitted.rho.item() < 1.0
    assert "loss_final" in diag
    assert diag["loss_final"] >= 0  # energy distance is non-negative


def test_gradient_free_reduces_loss_vs_init() -> None:
    obs = _truth_returns(n_steps=64, seed=1)
    init = _params(kappa=4.0, theta=0.10, xi=0.5, rho=0.3, v0=0.10)
    fitted, diag = fit_heston_gradient_free(
        returns=obs, dt=1 / 252, init=init, loss="energy",
        n_paths=128, seed=1, maxiter=40,
    )
    del fitted
    # Compare to loss at init by rerunning the inner objective once.
    from stochastech.calibration.heston_fit import _loss_fn, _simulate_log_returns
    gen = torch.Generator()
    gen.manual_seed(1)
    sim = _simulate_log_returns(init, 1.0, 1 / 252, obs.numel(), 128, gen,
                                torch.float64, "cpu", 2048)
    init_loss = _loss_fn("energy", sim, obs).item()
    assert diag["loss_final"] < init_loss


def test_rolling_window_emits_one_record_per_window() -> None:
    obs = _truth_returns(n_steps=200, seed=2)
    init = _params()
    records = rolling_window_calibration(
        returns=obs, dt=1 / 252, init=init, window=64, step=64,
        loss="energy", n_paths=64, n_iters=5, lr=5e-2, seed=0, method="bptt",
    )
    # 200 returns, window 64, step 64 -> windows starting at 0, 64, 128 -> 3 records.
    assert len(records) == 3
    for r in records:
        for k in ("window_idx", "start", "end", "loss_final", "params"):
            assert k in r
        assert set(r["params"].keys()) == {"mu", "kappa", "theta", "xi", "rho", "v0"}
        assert r["params"]["kappa"] > 0
        assert -1 < r["params"]["rho"] < 1


def test_rolling_window_warm_starts() -> None:
    # Warm start: window i+1 init equals window i fitted params. Record this by
    # checking that fitted params change smoothly across windows on a stationary
    # path — large jumps would imply re-initialization.
    obs = _truth_returns(n_steps=180, seed=3)
    init = _params(kappa=3.0, theta=0.06, xi=0.4, rho=-0.3, v0=0.06)
    records = rolling_window_calibration(
        returns=obs, dt=1 / 252, init=init, window=60, step=60,
        loss="energy", n_paths=64, n_iters=5, lr=5e-2, seed=0,
    )
    assert len(records) == 3
    # Window 1 should differ from the cold init by less than what would happen
    # without a warm start — equivalently, params at window 1 should be closer
    # to window 0's fit than to the cold init.
    init_kappa = init.kappa.item()
    win0_kappa = records[0]["params"]["kappa"]
    win1_kappa = records[1]["params"]["kappa"]
    assert abs(win1_kappa - win0_kappa) < abs(win1_kappa - init_kappa) + 1e-6


def test_rolling_window_supports_overlap() -> None:
    obs = _truth_returns(n_steps=100, seed=4)
    records = rolling_window_calibration(
        returns=obs, dt=1 / 252, init=_params(), window=50, step=10,
        loss="energy", n_paths=32, n_iters=3, lr=5e-2, seed=0,
    )
    # 100 returns, window 50, step 10 -> starts 0, 10, 20, 30, 40, 50 -> 6 windows.
    assert len(records) == 6
    starts = [r["start"] for r in records]
    assert starts == [0, 10, 20, 30, 40, 50]


def test_rolling_window_rejects_bad_inputs() -> None:
    obs = _truth_returns(n_steps=60, seed=0)
    init = _params()
    try:
        rolling_window_calibration(returns=obs, dt=1/252, init=init, window=4,
                                    n_paths=8, n_iters=1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on too-small window")
    try:
        rolling_window_calibration(returns=obs, dt=1/252, init=init, window=200,
                                    n_paths=8, n_iters=1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when window > n_returns")
    try:
        rolling_window_calibration(returns=obs, dt=1/252, init=init, window=20,
                                    method="bogus", n_paths=8, n_iters=1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on unknown method")


def test_rolling_window_gradient_free_method() -> None:
    obs = _truth_returns(n_steps=120, seed=5)
    init = _params()
    records = rolling_window_calibration(
        returns=obs, dt=1 / 252, init=init, window=60, step=60,
        loss="energy", n_paths=32, n_iters=10, seed=0, method="gradient_free",
    )
    assert len(records) == 2
    assert all("n_function_evals" in r for r in records)
