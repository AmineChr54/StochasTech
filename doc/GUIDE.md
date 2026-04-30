# StochasTech — Complete Guide

This is the single document that walks through the project from "what is this?" to "I have a calibrated Heston model and PDF figures on my disk." It assembles, in one place, what each part of the codebase does, how the pieces fit together, and exactly which commands to run to reproduce the paper.

---

## Table of contents

1. [What this project is](#1-what-this-project-is)
2. [Repository layout](#2-repository-layout)
3. [Setup (one-time)](#3-setup-one-time)
4. [The full pipeline at a glance](#4-the-full-pipeline-at-a-glance)
5. [End-to-end commands](#5-end-to-end-commands)
6. [Visualizing and inspecting results](#6-visualizing-and-inspecting-results)
7. [What was built, week by week](#7-what-was-built-week-by-week)
8. [Source-module reference](#8-source-module-reference)
9. [Driver-script reference](#9-driver-script-reference)
10. [Math-doc reference](#10-math-doc-reference)
11. [Test suite](#11-test-suite)
12. [Building the paper](#12-building-the-paper)
13. [Troubleshooting](#13-troubleshooting)
14. [Customization recipes](#14-customization-recipes)

---

## 1. What this project is

**Goal.** Calibrate the Heston stochastic-volatility model to historical equity returns by *backpropagating through the SDE solver*, then evaluate the calibrated model on out-of-sample 1-day Value-at-Risk (VaR) forecasts against two simpler baselines.

**Three deliverables**:

1. A self-contained PyTorch library (`stochastech/`) that simulates GBM and Heston, fits Heston by gradient descent, and runs VaR backtests.
2. A LaTeX paper draft (`paper/stochastech.tex`) with abstract, math model, method, and a results table seeded from generated JSON.
3. A folder of math derivations (`doc/math/`) — every code module is paired 1:1 with a LaTeX-bearing markdown file, and CI fails if a code change ships without its math doc.

**Headline contribution.** For Heston's small parameter dimension (six parameters), vanilla backprop-through-time (BPTT) is competitive with the more-complex stochastic-adjoint method, so the calibrator is short, easy to debug, and produces stable parameter trajectories under rolling-window calibration. Energy distance is used as the loss because it is more forgiving than NLL when the model is misspecified — a known issue for Heston on equity tails.

---

## 2. Repository layout

```
StochasTech/
├── README.md                    # Entry point; links to this guide and doc/
├── pyproject.toml               # Python package config + ruff/pytest settings
├── pixi.toml                    # Reproducible env (pixi = conda-forge wrapper)
├── .github/workflows/ci.yml     # Lint + doc-parity + pytest on every push
│
├── stochastech/                 # Library source — five sub-packages
│   ├── sde/                     # SDE simulators
│   │   ├── base.py              # Generic Euler–Maruyama
│   │   ├── gbm.py               # Geometric Brownian Motion (log-Euler default)
│   │   └── heston.py            # Heston full-truncation Euler + differentiable variant
│   ├── risk/                    # Risk metrics + backtests
│   │   ├── var.py               # VaR/ES + GBM-MLE/historical/Heston VaR forecasters
│   │   └── backtest.py          # Kupiec POF + Christoffersen independence + conditional coverage
│   ├── calibration/             # Differentiable model fitting
│   │   ├── losses.py            # Energy distance, Gaussian-KDE NLL
│   │   └── heston_fit.py        # Adam-through-SDE fit, gradient-free baseline, rolling window
│   ├── data/loaders.py          # yfinance pull + CSV cache
│   └── viz/plots.py             # Matplotlib helpers for paper figures (headless Agg)
│
├── tests/                       # 120 pytest tests
│   ├── test_doc_parity.py       # Enforces the binding rule (CI fails on missing math docs)
│   ├── test_smoke.py            # Imports every module + version string check
│   ├── test_gbm.py              # Week 1: GBM moments, positivity, determinism (13)
│   ├── test_heston.py           # Week 2: full-truncation, leverage sign, Feller (10)
│   ├── test_var.py              # Week 2: order-statistic VaR/ES, antithetic (14)
│   ├── test_losses.py           # Week 3: energy distance + NLL gradient flow (10)
│   ├── test_heston_fit.py       # Week 3: BPTT calibration end-to-end (7)
│   ├── test_loaders.py          # Week 4: yfinance cache + log-returns (9)
│   ├── test_calibration_baselines.py  # Week 4: scipy + rolling window (7)
│   ├── test_backtest.py         # Week 5: Kupiec / Christoffersen / CC (10)
│   ├── test_var_forecasts.py    # Week 5: GBM-MLE / historical / Heston VaR (7)
│   └── test_viz.py              # Week 6: figure helpers smoke (6)
│
├── scripts/                     # End-user drivers
│   ├── grad_check.py            # Autograd vs central differences for Heston
│   ├── run_calibration.py       # Pull data → rolling-window fit → results/calibration_*.json
│   ├── run_backtest.py          # Walk-forward VaR backtest → results/backtest_alpha*.json
│   └── reproduce_figures.py     # JSON → paper/figures/*.pdf
│
├── doc/                         # All documentation
│   ├── README.md                # Documentation index
│   ├── 01_overview.md           # What the project is
│   ├── 02_roadmap.md            # 6-week sprint schedule
│   ├── 03_scope.md              # Binding rule + non-goals
│   ├── GUIDE.md                 # ← This file
│   ├── math/                    # 9 LaTeX-bearing math derivations
│   │   ├── 00_index.md
│   │   ├── 01_brownian_motion.md
│   │   ├── 02_ito_calculus.md
│   │   ├── 03_gbm.md
│   │   ├── 04_heston.md
│   │   ├── 05_euler_maruyama.md
│   │   ├── 06_monte_carlo_var.md
│   │   ├── 07_adjoint_sde.md
│   │   ├── 08_calibration_losses.md
│   │   └── 09_var_backtesting.md
│   ├── methods/                 # Architectural choices
│   ├── architecture/            # Repo layout, tech stack, deferred phase 2
│   ├── reference/               # Inspiration + reading list
│   └── archive/                 # Pre-rewrite originals
│
├── paper/                       # LaTeX source
│   ├── stochastech.tex          # Paper draft v0
│   ├── refs.bib                 # 9 references
│   └── figures/                 # Generated PDFs (gitignored)
│
├── data_cache/                  # CSV cache from yfinance (gitignored)
├── results/                     # JSON outputs from runner scripts (gitignored)
├── notebooks/                   # Reserved for exploration
└── cpp/                         # Phase 2 placeholder + relocated legacy_main.cpp
```

---

## 3. Setup (one-time)

### Install pixi

`pixi` is a reproducible, conda-forge-backed environment manager. The single-source-of-truth `pixi.toml` pins every dependency.

```bash
# macOS / Linux
curl -fsSL https://pixi.sh/install.sh | bash

# Windows (PowerShell)
iwr -useb https://pixi.sh/install.ps1 | iex
```

### Install the project's environment

```bash
cd StochasTech
pixi install
```

This creates the default environment (Python 3.11/3.12, NumPy, SciPy, PyTorch, pandas, matplotlib) and the `dev` environment (adds pytest, pytest-cov, ruff). First install takes ~5 minutes; subsequent calls are instant.

### Verify the install

```bash
pixi run test                         # 120 tests, ~25 seconds
pixi run -e dev lint                  # ruff check, must be clean
```

If these pass, you're ready to run the pipeline.

---

## 4. The full pipeline at a glance

Reading order if you want to understand the data flow:

```
                    ┌────────────────────┐
                    │ yfinance (network) │
                    └──────────┬─────────┘
                               │  load_prices(ticker, start, end)
                               ▼
                    ┌────────────────────┐
                    │ data_cache/*.csv   │ ← survive across runs
                    └──────────┬─────────┘
                               │  log_returns(prices)
                               ▼
                    ┌────────────────────┐
                    │ daily log-returns  │
                    └──────────┬─────────┘
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
    ┌──────────────┐  ┌──────────────────┐  ┌──────────────┐
    │ run_         │  │ run_backtest.py  │  │ grad_check.  │
    │ calibration  │  │ (Week 5 driver)  │  │ py           │
    │ .py (Week 4) │  │                  │  │ (Week 3)     │
    └──────┬───────┘  └────────┬─────────┘  └──────────────┘
           │                   │
           ▼                   ▼
    ┌─────────────────────────────────┐
    │ results/calibration_<method>.json│
    │ results/backtest_alpha95.json    │
    └─────────────────┬───────────────┘
                      │  reproduce_figures.py
                      ▼
            ┌──────────────────────┐
            │ paper/figures/*.pdf  │
            └──────────┬───────────┘
                       │  pdflatex paper/stochastech.tex
                       ▼
                ┌─────────────┐
                │ paper.pdf   │
                └─────────────┘
```

Each box is one command. Go through them in section 5.

---

## 5. End-to-end commands

The following block reproduces every numerical result in the paper. Run from the repo root.

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

**Time budget on a laptop CPU.** Step 1: 30s. Step 2: 5–10 min (3 tickers × 5 windows × 200 Adam iters at n_paths=512). Step 3: 30–60 min (200+ refits, that's the long one). Step 4: a few seconds. Step 5: a few seconds.

To shrink runtime for a fast smoke run, scale down:

```bash
pixi run -e dev python scripts/run_calibration.py \
    --tickers SPY --window 252 --step 252 --n-iters 50 --n-paths 128

pixi run -e dev python scripts/run_backtest.py \
    --tickers SPY --window 252 --step 63 --n-iters 30 --n-paths 64 --mc-paths 5000
```

---

## 6. Visualizing and inspecting results

### What ends up where

| Artifact | Path | Format |
|----------|------|--------|
| Cached prices | `data_cache/SPY_2018-01-01_2024-12-31.csv` | CSV, indexed by date |
| Calibration parameters per window | `results/calibration_bptt.json` | One record per (ticker, window) |
| Backtest violations + coverage | `results/backtest_alpha95.json` | One record per ticker with violation series |
| Parameter trajectories | `paper/figures/param_trajectories_<TICKER>.pdf` | Multi-row plot of $\kappa, \theta, \xi, \rho, v_0$ over time |
| VaR vs realized | `paper/figures/var_violations_<TICKER>.pdf` | Realized loss curve with three VaR forecast lines + breach markers |
| Coverage bars | `paper/figures/coverage_alpha95.pdf` | Grouped bar of observed-rate per (ticker, method) with 5% nominal line |

### Reading the JSON output

`results/calibration_bptt.json`:

```json
{
  "config": {"tickers": ["SPY"], "window": 504, ...},
  "results": [
    {
      "ticker": "SPY",
      "windows": [
        {
          "window_idx": 0,
          "start_date": "2018-01-02",
          "end_date": "2019-12-31",
          "loss_final": 1.42e-04,
          "loss_initial": 6.30e-04,
          "n_iters": 200,
          "params": {"mu": 0.05, "kappa": 1.85, "theta": 0.041, ...}
        }
      ]
    }
  ]
}
```

`results/backtest_alpha95.json` — per ticker:

```json
{
  "ticker": "SPY",
  "summary": {
    "gbm_mle":    {"observed_rate": 0.064, "kupiec_pvalue": 0.18, "conditional_coverage_pvalue": 0.07},
    "heston":     {"observed_rate": 0.052, "kupiec_pvalue": 0.81, "conditional_coverage_pvalue": 0.43},
    "historical": {"observed_rate": 0.058, "kupiec_pvalue": 0.42, "conditional_coverage_pvalue": 0.21}
  },
  "violations": {"gbm_mle": [0,0,1,0,...], ...},
  "forecasts_per_method": {"gbm_mle": [0.018, 0.019, ...], ...},
  "realized_returns": [-0.012, 0.004, ...],
  "forecast_dates": ["2020-01-21", "2020-02-19", ...]
}
```

A model "passes" if its `conditional_coverage_pvalue` exceeds 0.05 (the standard 5% significance level): we cannot reject correct coverage.

### Quick interactive inspection

```python
# After running the pipeline:
import json
with open("results/backtest_alpha95.json") as f:
    data = json.load(f)

for r in data["results"]:
    print(r["ticker"])
    for method, s in r["summary"].items():
        print(f"  {method:12s}  obs={s['observed_rate']:.2%}  CC p={s['conditional_coverage_pvalue']:.3f}")
```

### Generating a single figure manually

```python
from pathlib import Path
from stochastech.viz.plots import plot_loss_curve, plot_param_trajectories

plot_loss_curve([1.0, 0.6, 0.4, 0.3, 0.25], Path("/tmp/loss.pdf"), title="My run")
```

All four plot helpers in `stochastech/viz/plots.py` follow the same shape: take prepared data + an output `Path`, write a vector PDF, return None.

---

## 7. What was built, week by week

### Day 1 — Documentation rewrite

The starting point was a folder of legacy planning docs. They were consolidated into the structure above: a top-level `README.md` for the doc folder, four numbered orientation files, a `math/` directory with stub files for nine derivations, `methods/` for architectural choices, `architecture/` for repo layout and the deferred Phase 2 plan, `reference/` for the reading list, and `archive/` preserving the originals. **The math-doc binding rule was introduced**: every code module under `stochastech/sde/` and `stochastech/calibration/` must ship with its paired math doc, and `tests/test_doc_parity.py` enforces this in CI.

### Week 0 — Repo skeleton

`pyproject.toml`, `pixi.toml`, `.gitignore`, root `README.md`. Module stubs that raise `NotImplementedError` with pointers to their math docs. Smoke-import test parametrized over all modules. Doc-parity test with three checks (mapping completeness, doc existence, doc-references-module). Empty driver scripts with their CLI shapes already locked. CI workflow.

### Week 1 — Foundations (GBM + Euler–Maruyama)

**Code.**
- `stochastech/sde/base.py`: `euler_maruyama(drift, diffusion, x0, dt, n_steps, generator=None)`. Diagonal-noise convention.
- `stochastech/sde/gbm.py`: `simulate_gbm(s0, mu, sigma, dt, n_steps, n_paths, log_euler=True, ...)`. Log-Euler is the default and is exact for GBM.
- `tests/test_gbm.py`: shape, determinism, positivity, analytic mean/variance match within MC stderr.

**Math docs filled.** `01_brownian_motion.md` (Donsker construction, $\langle W \rangle_t = t$, nowhere differentiability), `02_ito_calculus.md` (partition+Taylor derivation of Itô's lemma, Itô isometry), `03_gbm.md` (closed form, lognormal moments, MLE), `05_euler_maruyama.md` (left-endpoint freeze, strong-1/2 + weak-1 convergence).

### Week 2 — Heston + Monte Carlo VaR

**Code.**
- `stochastech/sde/heston.py`: `simulate_heston(...)` with full-truncation Euler. Cholesky-correlated noise via $Z^S = \rho Z^v + \sqrt{1-\rho^2}Z^\perp$. `feller_condition(kappa, theta, xi)` helper.
- `stochastech/risk/var.py`: order-statistic `value_at_risk` and `expected_shortfall` (FP-stable tail index), `antithetic_normals` paired-mirror draws.
- `tests/test_heston.py`, `tests/test_var.py`: full-truncation invariants, leverage sign for $\rho < 0$, ES ≥ VaR, monotonicity in $\alpha$.

**Math docs filled.** `04_heston.md` (CIR rationale, leverage effect, full-truncation discretization), `06_monte_carlo_var.md` (two-bond counterexample for VaR subadditivity, Bahadur stderr, antithetic variance reduction).

### Week 3 — Differentiable Heston + calibration loss

**Code.**
- Added `simulate_heston_diff(...)` to `stochastech/sde/heston.py`: same full-truncation scheme but accepts tensor-valued parameters and stitches a BPTT-friendly autograd graph. An `eps = 1e-12` floor inside $\sqrt{v^+}$ keeps the gradient finite at $v=0$.
- `stochastech/calibration/losses.py`: `energy_distance(simulated, observed)` (1D plug-in U-statistic), `nll_loss` (Silverman-bandwidth Gaussian KDE).
- `stochastech/calibration/heston_fit.py`: `HestonParams` dataclass, `_RawParams` reparameterizing $\log \kappa, \log \theta, \log \xi, \log v_0, \tanh^{-1} \rho$, `fit_heston(...)` running Adam with random subsampling cap on the $O(M^2)$ pairwise distances, and `heston_loss_and_grad(...)` for the gradient-check script.
- `scripts/grad_check.py`: autograd vs central-difference per parameter, fixed Brownian seed for common-random-numbers.

**Math docs filled.** `07_adjoint_sde.md` (Lagrangian → backward SDE, virtual Brownian tree, BPTT-vs-adjoint trade-off, sqrt-v Lipschitz caveat), `08_calibration_losses.md` (KL fragility under misspecification, energy distance ≡ MMD with $-\|\cdot\|$ kernel).

### Week 4 — Real data + rolling-window calibration

**Code.**
- `stochastech/data/loaders.py`: `load_prices(ticker, start, end)` with lazy yfinance import and CSV cache (CSV not parquet to dodge the pyarrow extension-type clash on Windows multi-pandas installs). `log_returns(series)` validates positivity.
- Added `fit_heston_gradient_free(...)` (scipy Nelder-Mead with common-random-numbers via fixed seed) and `rolling_window_calibration(...)` (warm-start across windows, supports both `bptt` and `gradient_free`) to `heston_fit.py`.
- `scripts/run_calibration.py`: full driver. CLI: tickers, dates, window, step, optimizer, paths, iters. Writes `results/calibration_<method>.json`.

**Math doc updated.** Added a Week-4 section to `08_calibration_losses.md` covering $\rho/\xi$ identifiability via the Fisher near-zero eigenvalue, regime-break diagnostics, and forward-chaining (expanding-window) cross-validation.

### Week 5 — VaR backtesting + paper draft

**Code.**
- `stochastech/risk/backtest.py`: `kupiec_pof(violations, alpha)` (binomial LR, $\chi^2_1$), `christoffersen_independence(violations)` (2×2 transition-count Markov LR, $\chi^2_1$), `conditional_coverage(violations, alpha)` (joint, $\chi^2_2$). All boundary cases handled via the $0 \log 0 = 0$ convention.
- Added `gbm_mle_var_forecast`, `historical_var_forecast`, `heston_var_forecast` to `stochastech/risk/var.py`.
- `scripts/run_backtest.py`: walk-forward window with monthly Heston refit, computes 1-day VaR under all three methods, runs Kupiec/Christoffersen/CC tests on the accumulated breach series, writes `results/backtest_alpha95.json`.

**Math doc.** New `09_var_backtesting.md`: Kupiec POF derivation from Wilks' theorem, Christoffersen Markov-chain LR with codimension argument, CC orthogonality.

**Paper draft v0.** `paper/stochastech.tex` filled — abstract, introduction with three-bullet contribution list, math model, method (full-truncation + energy distance + BPTT-vs-adjoint argument), VaR forecasting + backtesting, experiments setup, placeholder results table, discussion. `paper/refs.bib` extended with Szekely-Rizzo, Gretton et al., Barndorff-Nielsen-Shephard.

### Week 6 — Polish

**Code.**
- `stochastech/viz/plots.py`: four plot helpers — `plot_loss_curve`, `plot_param_trajectories`, `plot_var_violations`, `plot_coverage_bars`. Headless Agg backend. All take prepared data + an output `Path` and write vector PDFs.
- `scripts/reproduce_figures.py`: reads every `results/*.json` in the results directory and writes `paper/figures/*.pdf`. Skips cleanly when results JSONs are missing (so it works in a fresh checkout before you've run the calibrators).
- `scripts/run_backtest.py` extended with `forecasts_per_method` in the JSON output, so `var_violations_<TICKER>.pdf` can plot VaR series alongside realized losses.
- `.github/workflows/ci.yml`: switched from raw pip to `prefix-dev/setup-pixi@v0.8.1`, so CI runs identically to the local dev loop.
- `.gitignore`: `results/` and `paper/figures/*.pdf` (regenerable artifacts).
- `README.md`: rewritten with a full Week 1–5 quickstart, layout table, end-to-end recipe.
- `tests/test_viz.py`: 6 smoke tests covering each plot helper (writes a real PDF to `tmp_path`).

---

## 8. Source-module reference

### `stochastech/sde/base.py`

```
euler_maruyama(drift, diffusion, x0, dt, n_steps, generator=None) -> Tensor
```

Generic SDE stepper. `drift` and `diffusion` are callables `(x, t) -> Tensor` matching the shape of `x0`. Returns `(n_steps + 1, *x0.shape)`.

### `stochastech/sde/gbm.py`

```
simulate_gbm(s0, mu, sigma, dt, n_steps, n_paths, log_euler=True, ...) -> (n_steps+1, n_paths)
```

Default `log_euler=True` integrates $\log S$ exactly (lognormal increments) and is strictly positivity-preserving. `log_euler=False` falls through to `base.euler_maruyama` for direct comparison; can produce negative prices for large $\sigma\sqrt{\Delta t}$.

### `stochastech/sde/heston.py`

```
simulate_heston(s0, v0, mu, kappa, theta, xi, rho, dt, n_steps, n_paths, ...) -> (S, V)
simulate_heston_diff(...same args as tensors...) -> (S, V)        # BPTT-friendly
feller_condition(kappa, theta, xi) -> bool
```

Full-truncation Euler. `simulate_heston_diff` is the autograd-friendly variant used by the calibrator; the `eps` floor inside `sqrt(v^+)` keeps gradients finite without measurably affecting forward values.

### `stochastech/risk/var.py`

```
value_at_risk(returns, alpha=0.95)              # order-statistic, signed-returns input
expected_shortfall(returns, alpha=0.95)         # mean of worst (1-alpha) tail

gbm_mle_var_forecast(log_returns, alpha=0.95, horizon=1)
historical_var_forecast(log_returns, alpha=0.95)
heston_var_forecast(fitted_params, dt, alpha=0.95, n_paths=20_000, horizon=1, ...)

antithetic_normals(n, n_paths, generator=None)  # paired-mirror draws
```

Sign convention throughout: positive returns = profit, loss = $-r$, VaR is reported as a positive number when the tail is on the loss side.

### `stochastech/risk/backtest.py`

```
kupiec_pof(violations, alpha)                   # -> {stat, pvalue, n, n_violations, ...}
christoffersen_independence(violations)         # -> {stat, pvalue, n00, n01, n10, n11, ...}
conditional_coverage(violations, alpha)         # -> {stat, pvalue, kupiec, independence, ...}
```

All three return dicts and use `scipy.stats.chi2.sf` for p-values. Boundary cases ($\hat\pi = 0$, $N_{i\cdot} = 0$) handled via the $0 \log 0 = 0$ convention.

### `stochastech/calibration/losses.py`

```
energy_distance(simulated, observed)            # 2 E|X-Y| - E|X-X'| - E|Y-Y'|
nll_loss(simulated, observed, bandwidth=None)   # Silverman default bandwidth
```

Energy distance is the squared MMD with the negative-Euclidean kernel; zero iff the two distributions match. Both are differentiable in `simulated`.

### `stochastech/calibration/heston_fit.py`

```
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

### `stochastech/data/loaders.py`

```
load_prices(ticker, start, end, refresh=False, column="Close") -> DataFrame
log_returns(prices: Series) -> Series
```

CSV cache under `data_cache/`. Lazy `import yfinance` so unit tests don't pay the startup cost.

### `stochastech/viz/plots.py`

```
plot_loss_curve(history, out_path, title="")
plot_param_trajectories(windows, out_path, params=("kappa","theta","xi","rho","v0"), title="")
plot_var_violations(forecast_dates, realized_returns, forecasts, violations, out_path, title="")
plot_coverage_bars(summaries, out_path, expected_rate, title="")
```

Headless Agg backend. All write vector PDFs.

---

## 9. Driver-script reference

### `scripts/grad_check.py`

Confirms autograd matches central differences for each Heston parameter. Uses common random numbers (fixed Brownian seed) on both sides so the test is consistency, not Monte Carlo.

```bash
pixi run -e dev python scripts/grad_check.py
pixi run -e dev python scripts/grad_check.py --n-paths 256 --seed 42 --eps 1e-3
```

Exits non-zero if worst relative error exceeds 20%.

### `scripts/run_calibration.py`

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

Output: `results/calibration_<method>.json` plus per-window `kappa, theta, xi, rho, v0` printed to stdout.

### `scripts/run_backtest.py`

Walk-forward 1-day VaR backtest. For each step in the rolling window: fits Heston (warm-started from the previous fit), computes GBM-MLE in closed form, takes historical empirical quantile. Forecasts next-day VaR under each method, records violation. After the walk, runs Kupiec/Christoffersen/CC tests.

```bash
pixi run -e dev python scripts/run_backtest.py \
    --tickers SPY AAPL MSFT \
    --window 504 --step 21 --alpha 0.95 \
    --n-paths 256 --n-iters 100 --mc-paths 20000
```

Output: `results/backtest_alpha<NN>.json` with summary, violation series, forecast series. Stdout prints a coverage table per ticker:

```
=== SPY ===
Coverage at alpha=0.95:
method            obs%     exp%   kupiec p      ind p       CC p
gbm_mle          6.40%    5.00%     0.1812      0.0651     0.0703
heston           5.20%    5.00%     0.8104      0.4218     0.4326
historical       5.80%    5.00%     0.4221      0.2103     0.2078
```

### `scripts/reproduce_figures.py`

Reads `results/calibration_*.json` and `results/backtest_alpha*.json`; writes `paper/figures/*.pdf`.

```bash
pixi run -e dev python scripts/reproduce_figures.py
pixi run -e dev python scripts/reproduce_figures.py \
    --results-dir my_results --figures-dir my_figures
```

Skips cleanly when result JSONs are absent — useful for first-time clones.

---

## 10. Math-doc reference

Each file follows the same six-section template: **Statement** → **Derivation** → **Discretization** → **Assumptions and failure modes** → **References**. They are written so equations can be lifted straight into the paper.

| File | Topic | Paired code |
|------|-------|-------------|
| [`doc/math/01_brownian_motion.md`](math/01_brownian_motion.md) | Donsker, quadratic variation $\langle W \rangle_t = t$, nowhere differentiability | (foundational, no code) |
| [`doc/math/02_ito_calculus.md`](math/02_ito_calculus.md) | Itô's lemma derivation, box calculus, Itô isometry | (foundational) |
| [`doc/math/03_gbm.md`](math/03_gbm.md) | GBM closed form, lognormal moments, MLE | `stochastech/sde/gbm.py` |
| [`doc/math/04_heston.md`](math/04_heston.md) | CIR rationale, leverage, joint dynamics | `stochastech/sde/heston.py` |
| [`doc/math/05_euler_maruyama.md`](math/05_euler_maruyama.md) | Left-endpoint freeze, strong-1/2 + weak-1 orders | `stochastech/sde/base.py` |
| [`doc/math/06_monte_carlo_var.md`](math/06_monte_carlo_var.md) | VaR/ES coherence, Bahadur stderr, antithetic | `stochastech/risk/var.py` |
| [`doc/math/07_adjoint_sde.md`](math/07_adjoint_sde.md) | Backward SDE, virtual Brownian tree, BPTT trade-off | `stochastech/calibration/heston_fit.py` |
| [`doc/math/08_calibration_losses.md`](math/08_calibration_losses.md) | Energy distance ≡ MMD, identifiability, CV | `stochastech/calibration/losses.py` |
| [`doc/math/09_var_backtesting.md`](math/09_var_backtesting.md) | Kupiec POF, Christoffersen, conditional coverage | `stochastech/risk/backtest.py` |

The doc-parity test (`tests/test_doc_parity.py`) iterates every `.py` under `stochastech/sde/` and `stochastech/calibration/` and asserts (a) it has a mapping to a math doc, (b) that math doc exists, (c) that math doc references the module by its repo path. Adding a new code module without its math doc fails CI.

---

## 11. Test suite

```bash
pixi run pytest                               # all 120 tests
pixi run pytest tests/test_heston.py          # one file
pixi run pytest tests/test_heston.py -v       # verbose
pixi run pytest -k "feller"                   # filter by name
pixi run pytest --cov=stochastech --cov-report=term-missing
pixi run -e dev pytest tests/test_doc_parity.py -v   # the binding-rule check alone
```

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

Network is fully mocked in `test_loaders.py` via `sys.modules['yfinance']` patching, so the suite runs offline.

---

## 12. Building the paper

```bash
cd paper
pdflatex stochastech.tex
bibtex stochastech
pdflatex stochastech.tex
pdflatex stochastech.tex
```

Three pdflatex passes are needed for cross-references (one for the bib, two more for forward-references settling). The output is `paper/stochastech.pdf` (gitignored).

The paper draft v0 includes:

- Abstract pulled from `doc/01_overview.md`.
- Introduction with a three-bullet contribution list.
- Math model (eqs. 1–4) lifted from `doc/math/04_heston.md`.
- Method section explaining full-truncation, energy distance, and the BPTT-vs-adjoint trade-off (with an explicit "for $|\theta|=6$, BPTT is competitive" claim).
- VaR forecasting + backtesting section.
- Experiments with a placeholder Table 1 that you fill from `results/backtest_alpha95.json`.
- Discussion + future work (jumps, multi-asset, CUDA).

Filling Table 1 is currently manual — read the JSON, paste the rows into the LaTeX `tabular`. A future iteration could templatize this.

---

## 13. Troubleshooting

**`pixi: command not found`** — `pixi install` script didn't put the binary on PATH. Restart the shell or add `~/.pixi/bin` to PATH.

**`yfinance returned no data for SPY`** — usually rate-limiting or a transient yfinance issue. Wait a few minutes and retry. Or pass `--start 2010-01-01` for a longer history that's more likely to be fully cached on yfinance's end.

**Out-of-memory during calibration** — the energy distance forms an $O(M^2)$ pairwise matrix. The default `max_samples=2048` keeps it under 50MB. If you increase `n_paths` to large values without raising `max_samples`, memory will explode. Both are arguments to `fit_heston`.

**`pyarrow.lib.ArrowKeyError: A type extension with name pandas.period already defined`** — a pyarrow/pandas version conflict on Windows when both system Python and pixi Python see different installs. We sidestepped this in Week 4 by using CSV instead of parquet for the data cache.

**Test `test_finite_difference_matches_autograd_for_kappa` flaky** — it uses common random numbers, so the result is deterministic for a given seed. If it does fail, the most likely cause is a numerical change in `simulate_heston_diff` — check the sqrt-v floor and the rho clamping.

**Lint failures in CI but green locally** — usually a Python version mismatch. CI pins to whatever pixi resolves; bump locally with `pixi update` and re-run `pixi run -e dev lint`.

**Paper figures not generated** — `scripts/reproduce_figures.py` skips silently when result JSONs are missing. Verify `results/calibration_bptt.json` and `results/backtest_alpha95.json` exist before running it.

**Slow Heston backtest** — the dominant cost is the per-window Adam refit. Cut `--n-iters 100 → 40` for a quick sanity run; cut `--step 21 → 63` to refit quarterly instead of monthly.

---

## 14. Customization recipes

### Add a fourth ticker

```bash
pixi run -e dev python scripts/run_calibration.py --tickers SPY AAPL MSFT NVDA ...
pixi run -e dev python scripts/run_backtest.py --tickers SPY AAPL MSFT NVDA ...
```

The plotting helpers handle arbitrary ticker counts; the figures' x-axis ticks auto-thin past 12 windows.

### Try the gradient-free baseline instead of BPTT

```bash
pixi run -e dev python scripts/run_calibration.py --method gradient_free --tickers SPY
```

`fit_heston_gradient_free` calls scipy Nelder-Mead with a fixed simulator seed (common random numbers), so the objective is deterministic in $\theta$ and the optimizer can converge. It's slower per evaluation but uses no autograd memory.

### Use NLL instead of energy distance

In `scripts/run_calibration.py`, the `loss="energy"` argument is hardcoded inside `_calibrate_ticker`. Change to `loss="nll"`. Or add a CLI flag.

### Calibrate at a different VaR level

```bash
pixi run -e dev python scripts/run_backtest.py --alpha 0.99 --tickers SPY
```

Output filename includes the level: `results/backtest_alpha99.json`. Figures pick this up automatically.

### Add a new code module

1. Write the math doc first: `doc/math/10_my_topic.md` with the six-section template.
2. Add the code: `stochastech/sde/my_module.py` (or `calibration/`) referencing the doc in its module docstring.
3. Update `MODULE_TO_DOC` in `tests/test_doc_parity.py`.
4. Make sure the math doc text contains the literal string `stochastech/sde/my_module.py` (the parity test grep for the path).
5. `pixi run pytest tests/test_doc_parity.py` should pass.
6. Add unit tests under `tests/test_my_module.py`.

### Plug in a different stochastic-volatility model

Both `simulate_heston_diff` and `fit_heston` are the only Heston-specific things. To fit, e.g., SABR or 3/2: write the new `simulate_<model>_diff(...)` (must be differentiable in tensor parameters), then mirror `_RawParams` and `fit_heston` for the new parameter set.

---

That's everything. Run section 5 in order, inspect the JSONs from section 6, and `paper/stochastech.pdf` is the final deliverable.
