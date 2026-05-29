"""Driver: pull data for tickers, fit Heston by rolling window, persist results.

Produces the Week 4 deliverable — calibration table for N tickers x M windows
plus loss curves and parameter trajectories — written under ``results/``.

Usage:
    pixi run -e dev python scripts/run_calibration.py
    pixi run -e dev python scripts/run_calibration.py --tickers SPY AAPL MSFT \\
        --start 2018-01-01 --end 2024-12-31 --window 504 --step 252
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from stochastech.calibration.heston_fit import HestonParams, rolling_window_calibration
from stochastech.data.loaders import load_prices, log_returns


def _default_init() -> HestonParams:
    return HestonParams(
        mu=torch.tensor(0.05, dtype=torch.float64),
        kappa=torch.tensor(2.0, dtype=torch.float64),
        theta=torch.tensor(0.04, dtype=torch.float64),
        xi=torch.tensor(0.3, dtype=torch.float64),
        rho=torch.tensor(-0.5, dtype=torch.float64),
        v0=torch.tensor(0.04, dtype=torch.float64),
    )


def _calibrate_ticker(
    ticker: str, start: str, end: str, window: int, step: int,
    n_paths: int, n_iters: int, lr: float, seed: int, method: str,
    column: str = "Close", verbose: bool = True, progress_every: int = 20,
) -> dict:
    print(f"  fetching prices for {ticker} {start}..{end} ...", flush=True)
    df = load_prices(ticker, start, end, column=column)
    rets = log_returns(df[column])
    rets_t = torch.tensor(rets.to_numpy(), dtype=torch.float64)
    n_windows = max(0, (rets_t.numel() - window) // step + 1)
    print(
        f"  loaded {rets_t.numel()} returns "
        f"({rets.index[0].date()}..{rets.index[-1].date()}); "
        f"rolling fit: window={window} step={step} -> {n_windows} windows",
        flush=True,
    )
    records = rolling_window_calibration(
        returns=rets_t, dt=1 / 252, init=_default_init(), window=window, step=step,
        loss="energy", n_paths=n_paths, n_iters=n_iters, lr=lr, seed=seed,
        method=method, verbose=verbose, progress_every=progress_every,
    )
    # Attach the actual date range for each window (more useful in the paper than indices).
    dates = list(rets.index)
    for r in records:
        r["start_date"] = str(dates[r["start"]].date()) if r["start"] < len(dates) else None
        r["end_date"] = str(dates[min(r["end"] - 1, len(dates) - 1)].date())
    return {
        "ticker": ticker,
        "start": start,
        "end": end,
        "n_returns": int(rets_t.numel()),
        "window": window,
        "step": step,
        "method": method,
        "windows": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=["SPY", "AAPL", "MSFT"])
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--window", type=int, default=504,
                        help="rolling-window length in trading days (~2y)")
    parser.add_argument("--step", type=int, default=252,
                        help="rolling-window step in trading days (~1y)")
    parser.add_argument("--n-paths", type=int, default=512)
    parser.add_argument("--n-iters", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", default="bptt", choices=["bptt", "gradient_free"])
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--progress-every", type=int, default=20,
                        help="print per-iter loss line every N iterations")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress per-iter progress lines")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for ticker in args.tickers:
        print(f"=== {ticker} ===")
        result = _calibrate_ticker(
            ticker=ticker, start=args.start, end=args.end,
            window=args.window, step=args.step, n_paths=args.n_paths,
            n_iters=args.n_iters, lr=args.lr, seed=args.seed, method=args.method,
            verbose=not args.quiet, progress_every=args.progress_every,
        )
        all_results.append(result)
        for r in result["windows"]:
            p = r["params"]
            print(f"  window {r['window_idx']} [{r['start_date']}..{r['end_date']}]: "
                  f"loss={r['loss_final']:.4e} "
                  f"kappa={p['kappa']:.3f} theta={p['theta']:.4f} "
                  f"xi={p['xi']:.3f} rho={p['rho']:+.3f} v0={p['v0']:.4f}")

    out_path = results_dir / f"calibration_{args.method}.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": vars(args),
            "results": all_results,
        }, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
