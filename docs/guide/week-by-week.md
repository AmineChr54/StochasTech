# Week-by-Week Build Log

> What was built each week during the 6-week sprint.

## Day 1 — Documentation rewrite

The starting point was a folder of legacy planning docs. They were consolidated into a structured documentation tree: a top-level `README.md` for the doc folder, four numbered orientation files, a `math/` directory with stub files for nine derivations, `methods/` for architectural choices, `architecture/` for repo layout and the deferred Phase 2 plan, `reference/` for the reading list, and `archive/` preserving the originals. **The math-doc binding rule was introduced**: every code module under `stochastech/sde/` and `stochastech/calibration/` must ship with its paired math doc, and `tests/test_doc_parity.py` enforces this in CI.

---

## Week 0 — Repo skeleton

`pyproject.toml`, `pixi.toml`, `.gitignore`, root `README.md`. Module stubs that raise `NotImplementedError` with pointers to their math docs. Smoke-import test parametrized over all modules. Doc-parity test with three checks (mapping completeness, doc existence, doc-references-module). Empty driver scripts with their CLI shapes already locked. CI workflow.

---

## Week 1 — Foundations (GBM + Euler–Maruyama)

**Code.**

- `stochastech/sde/base.py`: `euler_maruyama(drift, diffusion, x0, dt, n_steps, generator=None)`. Diagonal-noise convention.
- `stochastech/sde/gbm.py`: `simulate_gbm(s0, mu, sigma, dt, n_steps, n_paths, log_euler=True, ...)`. Log-Euler is the default and is exact for GBM.
- `tests/test_gbm.py`: shape, determinism, positivity, analytic mean/variance match within MC stderr.

**Math docs filled.** `01_brownian_motion.md` (Donsker construction, $\langle W \rangle_t = t$, nowhere differentiability), `02_ito_calculus.md` (partition+Taylor derivation of Itô's lemma, Itô isometry), `03_gbm.md` (closed form, lognormal moments, MLE), `05_euler_maruyama.md` (left-endpoint freeze, strong-1/2 + weak-1 convergence).

---

## Week 2 — Heston + Monte Carlo VaR

**Code.**

- `stochastech/sde/heston.py`: `simulate_heston(...)` with full-truncation Euler. Cholesky-correlated noise via $Z^S = \rho Z^v + \sqrt{1-\rho^2}Z^\perp$. `feller_condition(kappa, theta, xi)` helper.
- `stochastech/risk/var.py`: order-statistic `value_at_risk` and `expected_shortfall` (FP-stable tail index), `antithetic_normals` paired-mirror draws.
- `tests/test_heston.py`, `tests/test_var.py`: full-truncation invariants, leverage sign for $\rho < 0$, ES ≥ VaR, monotonicity in $\alpha$.

**Math docs filled.** `04_heston.md` (CIR rationale, leverage effect, full-truncation discretization), `06_monte_carlo_var.md` (two-bond counterexample for VaR subadditivity, Bahadur stderr, antithetic variance reduction).

---

## Week 3 — Differentiable Heston + calibration loss

**Code.**

- Added `simulate_heston_diff(...)` to `stochastech/sde/heston.py`: same full-truncation scheme but accepts tensor-valued parameters and stitches a BPTT-friendly autograd graph. An `eps = 1e-12` floor inside $\sqrt{v^+}$ keeps the gradient finite at $v=0$.
- `stochastech/calibration/losses.py`: `energy_distance(simulated, observed)` (1D plug-in U-statistic), `nll_loss` (Silverman-bandwidth Gaussian KDE).
- `stochastech/calibration/heston_fit.py`: `HestonParams` dataclass, `_RawParams` reparameterizing $\log \kappa, \log \theta, \log \xi, \log v_0, \tanh^{-1} \rho$, `fit_heston(...)` running Adam with random subsampling cap on the $O(M^2)$ pairwise distances, and `heston_loss_and_grad(...)` for the gradient-check script.
- `scripts/grad_check.py`: autograd vs central-difference per parameter, fixed Brownian seed for common-random-numbers.

**Math docs filled.** `07_adjoint_sde.md` (Lagrangian → backward SDE, virtual Brownian tree, BPTT-vs-adjoint trade-off, sqrt-v Lipschitz caveat), `08_calibration_losses.md` (KL fragility under misspecification, energy distance ≡ MMD with $-\|\cdot\|$ kernel).

---

## Week 4 — Real data + rolling-window calibration

**Code.**

- `stochastech/data/loaders.py`: `load_prices(ticker, start, end)` with lazy yfinance import and CSV cache (CSV not parquet to dodge the pyarrow extension-type clash on Windows multi-pandas installs). `log_returns(series)` validates positivity.
- Added `fit_heston_gradient_free(...)` (scipy Nelder-Mead with common-random-numbers via fixed seed) and `rolling_window_calibration(...)` (warm-start across windows, supports both `bptt` and `gradient_free`) to `heston_fit.py`.
- `scripts/run_calibration.py`: full driver. CLI: tickers, dates, window, step, optimizer, paths, iters. Writes `results/calibration_<method>.json`.

**Math doc updated.** Added a Week-4 section to `08_calibration_losses.md` covering $\rho/\xi$ identifiability via the Fisher near-zero eigenvalue, regime-break diagnostics, and forward-chaining (expanding-window) cross-validation.

---

## Week 5 — VaR backtesting + paper draft

**Code.**

- `stochastech/risk/backtest.py`: `kupiec_pof(violations, alpha)` (binomial LR, $\chi^2_1$), `christoffersen_independence(violations)` (2×2 transition-count Markov LR, $\chi^2_1$), `conditional_coverage(violations, alpha)` (joint, $\chi^2_2$). All boundary cases handled via the $0 \log 0 = 0$ convention.
- Added `gbm_mle_var_forecast`, `historical_var_forecast`, `heston_var_forecast` to `stochastech/risk/var.py`.
- `scripts/run_backtest.py`: walk-forward window with monthly Heston refit, computes 1-day VaR under all three methods, runs Kupiec/Christoffersen/CC tests on the accumulated breach series, writes `results/backtest_alpha95.json`.

**Math doc.** New `09_var_backtesting.md`: Kupiec POF derivation from Wilks' theorem, Christoffersen Markov-chain LR with codimension argument, CC orthogonality.

**Paper draft v0.** `paper/stochastech.tex` filled — abstract, introduction with three-bullet contribution list, math model, method (full-truncation + energy distance + BPTT-vs-adjoint argument), VaR forecasting + backtesting, experiments setup, placeholder results table, discussion.

---

## Week 6 — Polish

**Code.**

- `stochastech/viz/plots.py`: four plot helpers — `plot_loss_curve`, `plot_param_trajectories`, `plot_var_violations`, `plot_coverage_bars`. Headless Agg backend. All take prepared data + an output `Path` and write vector PDFs.
- `scripts/reproduce_figures.py`: reads every `results/*.json` in the results directory and writes `paper/figures/*.pdf`. Skips cleanly when results JSONs are missing.
- `.github/workflows/ci.yml`: switched from raw pip to `prefix-dev/setup-pixi@v0.8.1`, so CI runs identically to the local dev loop.
- `README.md`: rewritten with a full Week 1–5 quickstart, layout table, end-to-end recipe.
- `tests/test_viz.py`: 6 smoke tests covering each plot helper (writes a real PDF to `tmp_path`).
