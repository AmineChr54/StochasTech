"""Out-of-sample VaR backtest: GBM-MLE vs Heston-AI vs historical.

For each ticker:
  1. Pull adjusted-close prices, compute daily log-returns.
  2. Walk a rolling estimation window forward by ``--step`` days.
  3. At each window: fit Heston via BPTT, fit GBM-MLE in closed form, compute
     historical VaR. Forecast 1-day-ahead VaR under each method.
  4. Compare to the realized next-day return; record a violation if
     ``L_t = -r_t > VaR_t``.
  5. Run Kupiec POF, Christoffersen independence, joint conditional-coverage
     tests on the accumulated violation series per (ticker, method).

Writes ``results/backtest.json`` with the per-(ticker, method) summary and the
violation series.

Usage:
    pixi run -e dev python scripts/run_backtest.py
    pixi run -e dev python scripts/run_backtest.py --tickers SPY \\
        --start 2018-01-01 --end 2024-12-31 --window 504 --step 21 --alpha 0.95
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def _is_tty() -> bool:
    try:
        return sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def _bar(pct: float, width: int = 20) -> str:
    pct = max(0.0, min(1.0, pct))
    filled = int(round(width * pct))
    return "[" + "#" * filled + "-" * (width - filled) + "]"

from stochastech.calibration.heston_fit import HestonParams, fit_heston
from stochastech.data.loaders import load_prices, log_returns
from stochastech.risk.backtest import conditional_coverage
from stochastech.risk.var import (
    gbm_mle_var_forecast,
    heston_var_forecast,
    historical_var_forecast,
)


def _default_init() -> HestonParams:
    return HestonParams(
        mu=torch.tensor(0.05, dtype=torch.float64),
        kappa=torch.tensor(2.0, dtype=torch.float64),
        theta=torch.tensor(0.04, dtype=torch.float64),
        xi=torch.tensor(0.3, dtype=torch.float64),
        rho=torch.tensor(-0.5, dtype=torch.float64),
        v0=torch.tensor(0.04, dtype=torch.float64),
    )


def _backtest_ticker(
    ticker: str, start: str, end: str, window: int, step: int, alpha: float,
    n_paths: int, n_iters: int, lr: float, seed: int, mc_paths: int,
    column: str = "Close", verbose: bool = True, progress_every: int = 25,
) -> dict:
    print(f"  fetching prices for {ticker} {start}..{end} ...", flush=True)
    df = load_prices(ticker, start, end, column=column)
    rets_series = log_returns(df[column])
    rets = torch.tensor(rets_series.to_numpy(), dtype=torch.float64)
    dates = list(rets_series.index)
    n = rets.numel()
    if n < window + 1:
        raise RuntimeError(f"{ticker}: need >= {window + 1} returns, have {n}.")

    forecast_records: list[dict] = []
    violations = {"gbm_mle": [], "heston": [], "historical": []}
    forecasts_per_method = {"gbm_mle": [], "heston": [], "historical": []}
    realized_returns: list[float] = []
    forecast_dates: list[str] = []
    current_init = _default_init()

    starts = list(range(0, n - window, step))
    print(
        f"  [{ticker}] {len(starts)} forecast windows over {n} returns "
        f"(window={window} train days, step={step} fwd days, "
        f"n_paths={n_paths} n_iters={n_iters} mc_paths={mc_paths})",
        flush=True,
    )
    inline = verbose and _is_tty()
    t0 = time.time()
    for idx, s in enumerate(starts):
        train = rets[s : s + window]
        # Forecast covers a single next-day return at index s + window.
        realized = float(rets[s + window].item())
        realized_returns.append(realized)
        forecast_dates.append(str(dates[s + window].date()))
        fcast_date = forecast_dates[-1]

        # GBM-MLE.
        gbm_var, _ = gbm_mle_var_forecast(train, alpha=alpha, horizon=1)
        # Historical.
        hist_var = historical_var_forecast(train, alpha=alpha)
        # Heston: warm-start from the previous fit.
        gen = torch.Generator()
        gen.manual_seed(seed + idx)
        if verbose and not inline:
            print(
                f"    win {idx + 1}/{len(starts)} forecast={fcast_date}: "
                f"fitting Heston ({n_iters} iters) ...",
                flush=True,
            )
        fitted, hist = fit_heston(
            returns=train, dt=1 / 252, init=current_init, loss="energy",
            n_paths=n_paths, n_iters=n_iters, lr=lr, generator=gen,
            verbose=inline, progress_every=max(1, n_iters // 10),
            progress_prefix=f"    win {idx + 1}/{len(starts)} {fcast_date} ",
            finalize_progress=not inline,
        )
        current_init = fitted
        gen2 = torch.Generator()
        gen2.manual_seed(seed + idx + 10_000)
        heston_var = heston_var_forecast(fitted, dt=1 / 252, alpha=alpha,
                                         n_paths=mc_paths, horizon=1, generator=gen2)

        forecasts = {
            "gbm_mle": float(gbm_var.item()),
            "heston": float(heston_var.item()),
            "historical": float(hist_var.item()),
        }
        for method, v in forecasts.items():
            violations[method].append(int(-realized > v))
            forecasts_per_method[method].append(v)
        forecast_records.append({
            "idx": idx,
            "forecast_date": forecast_dates[-1],
            "realized_return": realized,
            "forecasts": forecasts,
            "params": {k: float(getattr(fitted, k).item())
                       for k in ("mu", "kappa", "theta", "xi", "rho", "v0")},
        })

        if verbose:
            viol = {m: violations[m][-1] for m in violations}
            elapsed = time.time() - t0
            eta = elapsed / (idx + 1) * (len(starts) - idx - 1)
            cum = {m: sum(violations[m]) for m in violations}
            summary_msg = (
                f"  [{ticker}] {_bar((idx + 1) / len(starts))} "
                f"{idx + 1}/{len(starts)} ({fcast_date}) "
                f"VaR g={forecasts['gbm_mle']:+.4f} h={forecasts['heston']:+.4f} "
                f"H={forecasts['historical']:+.4f} | "
                f"realized={realized:+.4f} "
                f"breach[{viol['gbm_mle']}{viol['heston']}{viol['historical']}] "
                f"cum[{cum['gbm_mle']}/{cum['heston']}/{cum['historical']}] "
                f"ETA {eta:5.0f}s"
            )
            if inline:
                sys.stdout.write("\r" + summary_msg + "\x1b[K\n")
                sys.stdout.flush()
            elif (idx + 1) % progress_every == 0 or idx == len(starts) - 1:
                print(summary_msg, flush=True)

    # Coverage tests per method.
    summary = {}
    for method, vs in violations.items():
        v_t = torch.tensor(vs, dtype=torch.long)
        cc = conditional_coverage(v_t, alpha=alpha)
        summary[method] = {
            "n_forecasts": len(vs),
            "n_violations": int(v_t.sum().item()),
            "observed_rate": float(v_t.float().mean().item()),
            "expected_rate": 1 - alpha,
            "kupiec_pvalue": cc["kupiec"]["pvalue"],
            "kupiec_stat": cc["kupiec"]["stat"],
            "independence_pvalue": cc["independence"]["pvalue"],
            "independence_stat": cc["independence"]["stat"],
            "conditional_coverage_pvalue": cc["pvalue"],
            "conditional_coverage_stat": cc["stat"],
        }

    return {
        "ticker": ticker,
        "alpha": alpha,
        "window": window,
        "step": step,
        "n_returns": n,
        "summary": summary,
        "violations": violations,
        "forecasts_per_method": forecasts_per_method,
        "realized_returns": realized_returns,
        "forecast_dates": forecast_dates,
        "forecasts": forecast_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=["SPY", "AAPL", "MSFT"])
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--window", type=int, default=504)
    parser.add_argument("--step", type=int, default=21,
                        help="forecast cadence in trading days; 21 = ~monthly refit")
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--n-paths", type=int, default=256)
    parser.add_argument("--n-iters", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--mc-paths", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress per-window progress lines")
    parser.add_argument("--progress-every", type=int, default=25,
                        help="print ETA every N windows")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for ticker in args.tickers:
        print(f"=== {ticker} ===")
        out = _backtest_ticker(
            ticker=ticker, start=args.start, end=args.end,
            window=args.window, step=args.step, alpha=args.alpha,
            n_paths=args.n_paths, n_iters=args.n_iters, lr=args.lr,
            seed=args.seed, mc_paths=args.mc_paths,
            verbose=not args.quiet, progress_every=args.progress_every,
        )
        all_results.append(out)
        print(f"\nCoverage at alpha={args.alpha}:")
        print(f"{'method':<14} {'obs%':>8} {'exp%':>8} {'kupiec p':>10} "
              f"{'ind p':>10} {'CC p':>10}")
        for method, s in out["summary"].items():
            print(f"{method:<14} {100*s['observed_rate']:>7.2f}% {100*s['expected_rate']:>7.2f}% "
                  f"{s['kupiec_pvalue']:>10.4f} "
                  f"{s['independence_pvalue']:>10.4f} "
                  f"{s['conditional_coverage_pvalue']:>10.4f}")
        print()

    out_path = results_dir / f"backtest_alpha{int(args.alpha*100):02d}.json"
    # Strip detailed forecasts from summary file to keep it readable; keep the
    # violations / realized series for plotting.
    summary_for_json = []
    for r in all_results:
        summary_for_json.append({
            **{k: v for k, v in r.items() if k != "forecasts"},
        })
    with open(out_path, "w") as f:
        json.dump({"config": vars(args), "results": summary_for_json}, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
