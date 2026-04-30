# Customization

> Recipes for common modifications.

## Add a fourth ticker

```bash
pixi run -e dev python scripts/run_calibration.py --tickers SPY AAPL MSFT NVDA ...
pixi run -e dev python scripts/run_backtest.py --tickers SPY AAPL MSFT NVDA ...
```

The plotting helpers handle arbitrary ticker counts; the figures' x-axis ticks auto-thin past 12 windows.

## Try the gradient-free baseline instead of BPTT

```bash
pixi run -e dev python scripts/run_calibration.py --method gradient_free --tickers SPY
```

`fit_heston_gradient_free` calls scipy Nelder-Mead with a fixed simulator seed (common random numbers), so the objective is deterministic in $\theta$ and the optimizer can converge. It's slower per evaluation but uses no autograd memory.

## Use NLL instead of energy distance

In `scripts/run_calibration.py`, the `loss="energy"` argument is hardcoded inside `_calibrate_ticker`. Change to `loss="nll"`. Or add a CLI flag.

## Calibrate at a different VaR level

```bash
pixi run -e dev python scripts/run_backtest.py --alpha 0.99 --tickers SPY
```

Output filename includes the level: `results/backtest_alpha99.json`. Figures pick this up automatically.

## Add a new code module

1. Write the math doc first: `doc/math/10_my_topic.md` with the [six-section template](../math/index.md).
2. Add the code: `stochastech/sde/my_module.py` (or `calibration/`) referencing the doc in its module docstring.
3. Update `MODULE_TO_DOC` in `tests/test_doc_parity.py`.
4. Make sure the math doc text contains the literal string `stochastech/sde/my_module.py` (the parity test greps for the path).
5. `pixi run pytest tests/test_doc_parity.py` should pass.
6. Add unit tests under `tests/test_my_module.py`.

## Plug in a different stochastic-volatility model

Both `simulate_heston_diff` and `fit_heston` are the only Heston-specific things. To fit, e.g., SABR or 3/2: write the new `simulate_<model>_diff(...)` (must be differentiable in tensor parameters), then mirror `_RawParams` and `fit_heston` for the new parameter set.
