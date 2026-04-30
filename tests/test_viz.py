"""Plot helpers smoke-tested via headless Agg backend."""
from __future__ import annotations

from pathlib import Path

import pytest

from stochastech.viz.plots import (
    plot_coverage_bars,
    plot_loss_curve,
    plot_param_trajectories,
    plot_var_violations,
)


def test_plot_loss_curve_writes_pdf(tmp_path: Path) -> None:
    out = tmp_path / "loss.pdf"
    plot_loss_curve([1.0, 0.5, 0.25, 0.1, 0.05], out, title="test")
    assert out.exists() and out.stat().st_size > 0


def test_plot_param_trajectories_writes_pdf(tmp_path: Path) -> None:
    windows = [
        {"end_date": "2020-01-01", "params": {"kappa": 1.0, "theta": 0.04, "xi": 0.3,
                                              "rho": -0.5, "v0": 0.04}},
        {"end_date": "2020-06-01", "params": {"kappa": 1.5, "theta": 0.05, "xi": 0.4,
                                              "rho": -0.6, "v0": 0.05}},
        {"end_date": "2020-12-01", "params": {"kappa": 2.0, "theta": 0.06, "xi": 0.35,
                                              "rho": -0.55, "v0": 0.045}},
    ]
    out = tmp_path / "params.pdf"
    plot_param_trajectories(windows, out, title="trajectories")
    assert out.exists() and out.stat().st_size > 0


def test_plot_param_trajectories_rejects_empty() -> None:
    with pytest.raises(ValueError):
        plot_param_trajectories([], Path("/tmp/x.pdf"))


def test_plot_param_trajectories_handles_forecast_date_key(tmp_path: Path) -> None:
    windows = [
        {"forecast_date": "2020-01-01", "params": {"kappa": 1.0, "theta": 0.04,
                                                    "xi": 0.3, "rho": -0.5,
                                                    "v0": 0.04}},
        {"forecast_date": "2020-06-01", "params": {"kappa": 1.5, "theta": 0.05,
                                                    "xi": 0.4, "rho": -0.6,
                                                    "v0": 0.05}},
    ]
    out = tmp_path / "params2.pdf"
    plot_param_trajectories(windows, out)
    assert out.exists()


def test_plot_var_violations_writes_pdf(tmp_path: Path) -> None:
    dates = [f"2020-{m:02d}-01" for m in range(1, 7)]
    realized = [-0.01, 0.02, -0.03, 0.005, -0.025, 0.015]
    forecasts = {
        "gbm_mle": [0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
        "heston": [0.025, 0.022, 0.03, 0.025, 0.028, 0.022],
        "historical": [0.024] * 6,
    }
    violations = {
        "gbm_mle": [0, 0, 1, 0, 1, 0],
        "heston": [0, 0, 1, 0, 0, 0],
        "historical": [0, 0, 1, 0, 1, 0],
    }
    out = tmp_path / "viol.pdf"
    plot_var_violations(dates, realized, forecasts, violations, out, title="t")
    assert out.exists() and out.stat().st_size > 0


def test_plot_coverage_bars_writes_pdf(tmp_path: Path) -> None:
    summaries = {
        "SPY": {
            "gbm_mle": {"observed_rate": 0.06},
            "heston": {"observed_rate": 0.05},
            "historical": {"observed_rate": 0.07},
        },
        "AAPL": {
            "gbm_mle": {"observed_rate": 0.08},
            "heston": {"observed_rate": 0.055},
            "historical": {"observed_rate": 0.09},
        },
    }
    out = tmp_path / "coverage.pdf"
    plot_coverage_bars(summaries, out, expected_rate=0.05, title="t")
    assert out.exists() and out.stat().st_size > 0
