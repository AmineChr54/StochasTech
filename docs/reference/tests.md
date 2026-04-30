# Test Suite

> 120+ pytest tests organized by sprint week.

## Running tests

```bash
pixi run pytest                               # all 120 tests
pixi run pytest tests/test_heston.py          # one file
pixi run pytest tests/test_heston.py -v       # verbose
pixi run pytest -k "feller"                   # filter by name
pixi run pytest --cov=stochastech --cov-report=term-missing
pixi run -e dev pytest tests/test_doc_parity.py -v   # the binding-rule check alone
```

## Test breakdown

| File | Count | What it verifies |
|------|------:|------------------|
| `test_doc_parity.py` | 11 | Every code module has a math doc that references it |
| `test_smoke.py` | 16 | Every package + version string imports |
| `test_gbm.py` | 13 | Mean/var match analytic, log-Euler positive, deterministic with seed |
| `test_heston.py` | 10 | Full-truncation, leverage sign for $\rho<0$, $\xi=0$ collapses to GBM |
| `test_var.py` | 14 | Order-statistic exactness, ES≥VaR, antithetic mirror |
| `test_losses.py` | 10 | Energy=0 on identical samples, gradient flow, GD reduces |
| `test_heston_fit.py` | 7 | BPTT parity with non-diff sim, Adam moves params, FD-vs-autograd |
| `test_loaders.py` | 9 | yfinance cache write/read, refresh, MultiIndex, edge cases |
| `test_calibration_baselines.py` | 7 | Scipy fit constraints preserved, rolling warm-start smoothness |
| `test_backtest.py` | 10 | Kupiec accepts iid Bernoulli, rejects biased; independence detects clustering |
| `test_var_forecasts.py` | 7 | GBM-MLE matches normal quantile, Heston→GBM at $\xi=0$ |
| `test_viz.py` | 6 | Each plot helper writes a non-empty PDF |

## Notes

- Network is fully mocked in `test_loaders.py` via `sys.modules['yfinance']` patching, so the suite runs **offline**.
- Tests are parametrized over `float32` and `float64` where convergence rates are verified.
- The doc-parity test (`test_doc_parity.py`) enforces the [binding rule](../project/scope.md) — every code module under `stochastech/sde/` and `stochastech/calibration/` must have a paired math doc.
