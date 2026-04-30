# StochasTech

Differentiable Heston-model calibrator with Monte Carlo VaR backtesting on real equity data. Phase 1 = Python + PyTorch + paper. C++/CUDA is Phase 2 (deferred).

See [doc/README.md](doc/README.md) for the documentation index, [doc/01_overview.md](doc/01_overview.md) for what the project is, and [doc/02_roadmap.md](doc/02_roadmap.md) for the 6-week sprint schedule.

## Quickstart

```bash
pixi install
pixi run test                                    # 120+ tests, ~25s
pixi run -e dev python scripts/grad_check.py     # verify autograd vs central differences
```

End-to-end paper-result reproduction (needs network for the first run; cached after):

```bash
# 1. Rolling-window calibration on SPY/AAPL/MSFT, 2-year window stepping yearly.
pixi run -e dev python scripts/run_calibration.py \
    --tickers SPY AAPL MSFT --window 504 --step 252

# 2. Out-of-sample 1-day VaR backtest with monthly refit.
pixi run -e dev python scripts/run_backtest.py \
    --tickers SPY AAPL MSFT --window 504 --step 21 --alpha 0.95

# 3. Generate paper figures from the JSON results.
pixi run -e dev python scripts/reproduce_figures.py

# 4. Build the paper.
cd paper && pdflatex stochastech.tex && bibtex stochastech && pdflatex stochastech.tex
```

## Layout

| Path | Purpose |
|------|---------|
| [stochastech/](stochastech/) | Library source — SDE simulators, calibration, risk, viz |
| [tests/](tests/) | pytest suite (120+ tests, includes math-doc parity) |
| [doc/](doc/) | Overview, roadmap, math derivations, methods, architecture |
| [doc/math/](doc/math/) | LaTeX-bearing math derivations paired 1:1 with code modules |
| [scripts/](scripts/) | `grad_check.py`, `run_calibration.py`, `run_backtest.py`, `reproduce_figures.py` |
| [paper/](paper/) | LaTeX source + bib + figures for the paper |
| [notebooks/](notebooks/) | Exploratory notebooks |
| [cpp/](cpp/) | Placeholder for Phase 2 |
| [results/](results/) | JSON results (gitignored; produced by run_calibration / run_backtest) |

## What this implements

- **Week 1.** Generic Euler–Maruyama core ([stochastech/sde/base.py](stochastech/sde/base.py)) and GBM with both naive- and log-Euler ([stochastech/sde/gbm.py](stochastech/sde/gbm.py)).
- **Week 2.** Heston full-truncation Euler with Cholesky-correlated noise ([stochastech/sde/heston.py](stochastech/sde/heston.py)) and order-statistic VaR/ES + antithetic helper ([stochastech/risk/var.py](stochastech/risk/var.py)).
- **Week 3.** Differentiable Heston (`simulate_heston_diff`) wired into a BPTT calibrator ([stochastech/calibration/heston_fit.py](stochastech/calibration/heston_fit.py)) with energy-distance + Gaussian-KDE NLL losses ([stochastech/calibration/losses.py](stochastech/calibration/losses.py)). Gradient-check vs central differences in [scripts/grad_check.py](scripts/grad_check.py).
- **Week 4.** yfinance loader with CSV cache ([stochastech/data/loaders.py](stochastech/data/loaders.py)), gradient-free baseline (scipy Nelder-Mead), rolling-window calibration runner.
- **Week 5.** Kupiec POF + Christoffersen independence + joint conditional-coverage ([stochastech/risk/backtest.py](stochastech/risk/backtest.py)). GBM-MLE / Heston-AI / historical VaR forecasters. End-to-end backtest runner. Paper draft v0 in [paper/stochastech.tex](paper/stochastech.tex).
- **Week 6.** Reproducible-figures script, paper-figure helpers ([stochastech/viz/plots.py](stochastech/viz/plots.py)), polish.

## Binding rule

Every code module under `stochastech/sde/` and `stochastech/calibration/` ships with a paired LaTeX-bearing math doc under [doc/math/](doc/math/). PRs without the math doc fail CI ([tests/test_doc_parity.py](tests/test_doc_parity.py)). See [doc/03_scope.md](doc/03_scope.md).

## Status

Phase 1 — Week 6 (polish). Tag `v0.1.0` on completion.
