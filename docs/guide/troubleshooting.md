# Troubleshooting

> Common issues and their fixes.

## Environment issues

!!! warning "`pixi: command not found`"
    `pixi install` script didn't put the binary on PATH. Restart the shell or add `~/.pixi/bin` to PATH.

!!! warning "`pyarrow.lib.ArrowKeyError: A type extension with name pandas.period already defined`"
    A pyarrow/pandas version conflict on Windows when both system Python and pixi Python see different installs. We sidestepped this in Week 4 by using CSV instead of parquet for the data cache.

## Data issues

!!! warning "`yfinance returned no data for SPY`"
    Usually rate-limiting or a transient yfinance issue. Wait a few minutes and retry. Or pass `--start 2010-01-01` for a longer history that's more likely to be fully cached on yfinance's end.

## Performance issues

!!! warning "Out-of-memory during calibration"
    The energy distance forms an $O(M^2)$ pairwise matrix. The default `max_samples=2048` keeps it under 50MB. If you increase `n_paths` to large values without raising `max_samples`, memory will explode. Both are arguments to `fit_heston`.

!!! tip "Slow Heston backtest"
    The dominant cost is the per-window Adam refit. Cut `--n-iters 100 → 40` for a quick sanity run; cut `--step 21 → 63` to refit quarterly instead of monthly.

## Test issues

!!! warning "`test_finite_difference_matches_autograd_for_kappa` flaky"
    It uses common random numbers, so the result is deterministic for a given seed. If it does fail, the most likely cause is a numerical change in `simulate_heston_diff` — check the sqrt-v floor and the rho clamping.

!!! warning "Lint failures in CI but green locally"
    Usually a Python version mismatch. CI pins to whatever pixi resolves; bump locally with `pixi update` and re-run `pixi run -e dev lint`.

## Figure issues

!!! warning "Paper figures not generated"
    `scripts/reproduce_figures.py` skips silently when result JSONs are missing. Verify `results/calibration_bptt.json` and `results/backtest_alpha95.json` exist before running it.
