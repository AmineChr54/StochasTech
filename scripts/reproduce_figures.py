"""Regenerate every figure in the paper from cached results JSON.

Reads:
    results/calibration_<method>.json   (from scripts/run_calibration.py)
    results/backtest_alpha<NN>.json     (from scripts/run_backtest.py)

Writes:
    paper/figures/param_trajectories_<TICKER>.pdf
    paper/figures/var_violations_<TICKER>.pdf
    paper/figures/coverage_alpha<NN>.pdf

Designed to run end-to-end in under 30 seconds once the result JSONs exist.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from stochastech.viz.plots import (
    plot_coverage_bars,
    plot_param_trajectories,
    plot_var_violations,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "paper" / "figures"


def _figures_from_calibration(path: Path, figures_dir: Path) -> None:
    if not path.exists():
        print(f"  [skip] {path.name} not found")
        return
    with open(path) as f:
        payload = json.load(f)
    for ticker_result in payload["results"]:
        ticker = ticker_result["ticker"]
        windows = ticker_result["windows"]
        out = figures_dir / f"param_trajectories_{ticker}.pdf"
        plot_param_trajectories(windows, out,
                                title=f"{ticker} rolling-window calibration")
        print(f"  wrote {out.relative_to(REPO_ROOT)}")


def _figures_from_backtest(path: Path, figures_dir: Path) -> None:
    if not path.exists():
        print(f"  [skip] {path.name} not found")
        return
    with open(path) as f:
        payload = json.load(f)
    alpha = payload["config"]["alpha"]
    summaries = {}
    for ticker_result in payload["results"]:
        ticker = ticker_result["ticker"]
        summaries[ticker] = ticker_result["summary"]

        # Per-ticker violation plot needs the forecast values per method, which
        # only land in the JSON when run_backtest.py is invoked with the full
        # detail dump. Skip cleanly when absent.
        forecast_dates = ticker_result.get("forecast_dates", [])
        realized = ticker_result.get("realized_returns", [])
        violations = ticker_result.get("violations", {})
        forecasts = ticker_result.get("forecasts_per_method")
        if forecast_dates and realized and violations and forecasts:
            out = figures_dir / f"var_violations_{ticker}.pdf"
            plot_var_violations(forecast_dates, realized, forecasts,
                                violations, out,
                                title=f"{ticker} 1-day VaR vs realized loss "
                                      f"(alpha={alpha})")
            print(f"  wrote {out.relative_to(REPO_ROOT)}")

    if summaries:
        out = figures_dir / f"coverage_alpha{int(alpha*100):02d}.pdf"
        plot_coverage_bars(summaries, out, expected_rate=1 - alpha,
                           title=f"Observed violation rate (alpha={alpha})")
        print(f"  wrote {out.relative_to(REPO_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--figures-dir", default=str(FIGURES_DIR))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading results from {results_dir}")
    print(f"Writing figures to {figures_dir}")

    print("\nCalibration figures:")
    for path in sorted(results_dir.glob("calibration_*.json")):
        _figures_from_calibration(path, figures_dir)

    print("\nBacktest figures:")
    for path in sorted(results_dir.glob("backtest_alpha*.json")):
        _figures_from_backtest(path, figures_dir)


if __name__ == "__main__":
    main()
