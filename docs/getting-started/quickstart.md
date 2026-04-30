# Quickstart

> Reproduce every numerical result in the paper in five commands.

## End-to-end commands

Run from the repo root after [installing](installation.md):

```bash
# 1. Verify autograd against central differences (Week 3 sanity check).
#    Prints a per-parameter table; passes if worst rel-err < 20%.
pixi run -e dev python scripts/grad_check.py

# 2. Rolling-window calibration on three tickers.
#    First call hits yfinance and writes data_cache/*.csv; subsequent calls are offline.
pixi run -e dev python scripts/run_calibration.py \
    --tickers SPY AAPL MSFT \
    --start 2018-01-01 --end 2024-12-31 \
    --window 504 --step 252

# 3. Out-of-sample VaR backtest with monthly Heston refit.
#    Re-uses the data cache from step 2.
pixi run -e dev python scripts/run_backtest.py \
    --tickers SPY AAPL MSFT \
    --start 2018-01-01 --end 2024-12-31 \
    --window 504 --step 21 --alpha 0.95

# 4. Generate every paper figure from the JSON results.
pixi run -e dev python scripts/reproduce_figures.py

# 5. Build the paper PDF.
cd paper
pdflatex stochastech.tex
bibtex stochastech
pdflatex stochastech.tex
pdflatex stochastech.tex      # re-run for cross-references
cd ..
```

## Time budget

| Step | What | Time (laptop CPU) |
|------|------|--------------------|
| 1 | Gradient check | ~30 seconds |
| 2 | Rolling-window calibration | 5–10 minutes |
| 3 | VaR backtest | 30–60 minutes |
| 4 | Generate figures | A few seconds |
| 5 | Build paper | A few seconds |

## Fast smoke run

To shrink runtime for a quick sanity test, scale down the simulation parameters:

```bash
pixi run -e dev python scripts/run_calibration.py \
    --tickers SPY --window 252 --step 252 --n-iters 50 --n-paths 128

pixi run -e dev python scripts/run_backtest.py \
    --tickers SPY --window 252 --step 63 --n-iters 30 --n-paths 64 --mc-paths 5000
```

!!! tip "Data caching"
    The first run pulls data from yfinance and writes CSV files to `data_cache/`. Subsequent runs are fully offline.
