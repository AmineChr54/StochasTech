# Driver Scripts

> Command-line scripts for running the full pipeline.

## `scripts/grad_check.py`

Confirms autograd matches central differences for each Heston parameter. Uses common random numbers (fixed Brownian seed) on both sides so the test is consistency, not Monte Carlo.

```bash
pixi run -e dev python scripts/grad_check.py
pixi run -e dev python scripts/grad_check.py --n-paths 256 --seed 42 --eps 1e-3
```

Exits non-zero if worst relative error exceeds 20%.

---

## `scripts/run_calibration.py`

Pulls historical prices for N tickers, runs rolling-window Heston calibration on log-returns, persists results.

```bash
pixi run -e dev python scripts/run_calibration.py \
    --tickers SPY AAPL MSFT \
    --start 2018-01-01 --end 2024-12-31 \
    --window 504 --step 252 \
    --n-paths 512 --n-iters 200 --lr 1e-2 \
    --method bptt \
    --results-dir results
```

**Output:** `results/calibration_<method>.json` plus per-window $\kappa, \theta, \xi, \rho, v_0$ printed to stdout.

---

## `scripts/run_backtest.py`

Walk-forward 1-day VaR backtest. For each step in the rolling window: fits Heston (warm-started from the previous fit), computes GBM-MLE in closed form, takes historical empirical quantile. Forecasts next-day VaR under each method, records violation. After the walk, runs Kupiec/Christoffersen/CC tests.

```bash
pixi run -e dev python scripts/run_backtest.py \
    --tickers SPY AAPL MSFT \
    --window 504 --step 21 --alpha 0.95 \
    --n-paths 256 --n-iters 100 --mc-paths 20000
```

**Output:** `results/backtest_alpha<NN>.json` with summary, violation series, forecast series. Stdout prints a coverage table per ticker:

```
=== SPY ===
Coverage at alpha=0.95:
method            obs%     exp%   kupiec p      ind p       CC p
gbm_mle          6.40%    5.00%     0.1812      0.0651     0.0703
heston           5.20%    5.00%     0.8104      0.4218     0.4326
historical       5.80%    5.00%     0.4221      0.2103     0.2078
```

---

## `scripts/reproduce_figures.py`

Reads `results/calibration_*.json` and `results/backtest_alpha*.json`; writes `paper/figures/*.pdf`.

```bash
pixi run -e dev python scripts/reproduce_figures.py
pixi run -e dev python scripts/reproduce_figures.py \
    --results-dir my_results --figures-dir my_figures
```

Skips cleanly when result JSONs are absent — useful for first-time clones.
