# Source Module Reference

> API reference for all public functions in the `stochastech` package.

## `stochastech/sde/base.py`

```python
euler_maruyama(drift, diffusion, x0, dt, n_steps, generator=None) -> Tensor
```

Generic SDE stepper. `drift` and `diffusion` are callables `(x, t) -> Tensor` matching the shape of `x0`. Returns `(n_steps + 1, *x0.shape)`.

---

## `stochastech/sde/gbm.py`

```python
simulate_gbm(s0, mu, sigma, dt, n_steps, n_paths, log_euler=True, ...) -> (n_steps+1, n_paths)
```

Default `log_euler=True` integrates $\log S$ exactly (lognormal increments) and is strictly positivity-preserving. `log_euler=False` falls through to `base.euler_maruyama` for direct comparison; can produce negative prices for large $\sigma\sqrt{\Delta t}$.

---

## `stochastech/sde/heston.py`

```python
simulate_heston(s0, v0, mu, kappa, theta, xi, rho, dt, n_steps, n_paths, ...) -> (S, V)
simulate_heston_diff(...same args as tensors...) -> (S, V)        # BPTT-friendly
feller_condition(kappa, theta, xi) -> bool
```

Full-truncation Euler. `simulate_heston_diff` is the autograd-friendly variant used by the calibrator; the `eps` floor inside `sqrt(v^+)` keeps gradients finite without measurably affecting forward values.

---

## `stochastech/risk/var.py`

```python
value_at_risk(returns, alpha=0.95)              # order-statistic, signed-returns input
expected_shortfall(returns, alpha=0.95)         # mean of worst (1-alpha) tail

gbm_mle_var_forecast(log_returns, alpha=0.95, horizon=1)
historical_var_forecast(log_returns, alpha=0.95)
heston_var_forecast(fitted_params, dt, alpha=0.95, n_paths=20_000, horizon=1, ...)

antithetic_normals(n, n_paths, generator=None)  # paired-mirror draws
```

!!! info "Sign convention"
    Positive returns = profit, loss = $-r$, VaR is reported as a positive number when the tail is on the loss side.

---

## `stochastech/risk/backtest.py`

```python
kupiec_pof(violations, alpha)                   # -> {stat, pvalue, n, n_violations, ...}
christoffersen_independence(violations)         # -> {stat, pvalue, n00, n01, n10, n11, ...}
conditional_coverage(violations, alpha)         # -> {stat, pvalue, kupiec, independence, ...}
```

All three return dicts and use `scipy.stats.chi2.sf` for p-values. Boundary cases ($\hat\pi = 0$, $N_{i\cdot} = 0$) handled via the $0 \log 0 = 0$ convention.

---

## `stochastech/calibration/losses.py`

```python
energy_distance(simulated, observed)            # 2 E|X-Y| - E|X-X'| - E|Y-Y'|
nll_loss(simulated, observed, bandwidth=None)   # Silverman default bandwidth
```

Energy distance is the squared MMD with the negative-Euclidean kernel; zero iff the two distributions match. Both are differentiable in `simulated`.

---

## `stochastech/calibration/heston_fit.py`

```python
HestonParams(mu, kappa, theta, xi, rho, v0)     # dataclass of constrained tensor values

fit_heston(returns, dt, init, loss="energy", n_paths=4096, n_iters=500, lr=1e-2,
           max_samples=2048, ...) -> (HestonParams, history: list[float])

fit_heston_gradient_free(returns, dt, init, loss="energy", method="Nelder-Mead",
                         maxiter=200, ...) -> (HestonParams, diagnostics: dict)

rolling_window_calibration(returns, dt, init, window, step=None,
                           method="bptt", ...) -> list[dict]

heston_loss_and_grad(returns, dt, params, ...) -> (loss, grads_dict)
```

`fit_heston` is the workhorse. Internally it reparameterizes positive parameters via `exp` and `rho` via `tanh`, runs Adam on the unconstrained leaves, and subsamples the simulated returns to bound the $O(M^2)$ pairwise-distance memory cost.

---

## `stochastech/data/loaders.py`

```python
load_prices(ticker, start, end, refresh=False, column="Close") -> DataFrame
log_returns(prices: Series) -> Series
```

CSV cache under `data_cache/`. Lazy `import yfinance` so unit tests don't pay the startup cost.

---

## `stochastech/viz/plots.py`

```python
plot_loss_curve(history, out_path, title="")
plot_param_trajectories(windows, out_path, params=("kappa","theta","xi","rho","v0"), title="")
plot_var_violations(forecast_dates, realized_returns, forecasts, violations, out_path, title="")
plot_coverage_bars(summaries, out_path, expected_rate, title="")
```

Headless Agg backend. All write vector PDFs.
