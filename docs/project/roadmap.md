# Roadmap — 6-week solo sprint

> Week-by-week schedule. Pairs a math-learning track with a build track each week so the theory is internalized by implementing it.

## Pacing rule

Every week with a math track ships its corresponding math docs in [math/](../math/index.md) **before or alongside** the code. A code module without its math doc is incomplete (CI enforces this — see [03_scope.md](../project/scope.md)).

---

## Week 1 — Foundations

**Math.** Brownian motion, Itô's lemma, Itô isometry, GBM as the canonical SDE.
- Source: Shreve, *Stochastic Calculus for Finance II*, ch. 3–4. Alternative: Øksendal, *Stochastic Differential Equations*, ch. 3–5.

**Build.** Python repo skeleton (`pyproject.toml`, `pixi`, `ruff`, `pytest`). Implement Euler–Maruyama for GBM. NumPy first (vectorize over paths), then port to PyTorch tensors. Unit-test simulated mean and variance against analytic GBM moments.

**Math docs (mandatory).**
- [math/brownian-motion.md](../math/brownian-motion.md)
- [math/ito-calculus.md](../math/ito-calculus.md)
- [math/gbm.md](../math/gbm.md)
- [math/euler-maruyama.md](../math/euler-maruyama.md)

**Deliverable.** `stochastech/sde/gbm.py` + tests + the four math docs, merged together.

---

## Week 2 — Heston + Monte Carlo risk

**Math.** Heston SDE system, Feller condition, why stochastic volatility matters. Pick one variance-reduction technique (antithetic variates recommended; control variates as a stretch).

**Build.** Heston Euler–Maruyama with full truncation enforcing $v_t \ge 0$. VaR / ES estimators from Monte Carlo paths. Benchmark sims/sec on CPU.

**Math docs (mandatory).**
- [math/heston.md](../math/heston.md) — model derivation + Feller condition
- [math/monte-carlo-var.md](../math/monte-carlo-var.md) — estimator + convergence rates

**Deliverable.** `stochastech/sde/heston.py`, `stochastech/risk/var.py`, notebook reproducing the classic Heston volatility smile, plus the two math docs.

---

## Week 3 — Differentiable SDE + calibration loss

**Math.** Adjoint method for SDEs (Li et al., *Scalable Gradients for Stochastic Differential Equations*, 2020). Why GBM has closed-form MLE but Heston does not. Choice of loss: log-likelihood of returns vs distributional distance (energy / MMD).

**Build.** Wrap Heston solver with `torchsde.sdeint_adjoint` so $(\kappa, \theta, \xi, \rho, v_0)$ are leaf tensors. Implement loss. Smoke-test that a single Adam step actually moves the parameters.

**Math docs (mandatory).**
- [math/adjoint-sde.md](../math/adjoint-sde.md)
- [math/calibration-losses.md](../math/calibration-losses.md)

**Deliverable.** `stochastech/calibration/heston_fit.py`, gradient-check script, two math docs.

---

## Week 4 — Real data + calibration runs

**Math.** Estimation pitfalls — identifiability of $\rho$ vs $\xi$, leverage effect, regime breaks. Cross-validation for SDE models.

**Build.** Pull SPY + 2 single names via `yfinance`. Compute log-returns. Run rolling-window calibration. Persist fitted parameters and diagnostics. Compare against a `scipy.optimize` gradient-free baseline as a sanity check.

**Deliverable.** Calibration results table for 3 tickers × 3 windows. Loss curves and parameter trajectories.

---

## Week 5 — Evaluation + paper draft

**Math.** VaR backtesting methodology — Kupiec POF and Christoffersen independence tests. Reading list converted into the paper's references.

**Build.** Out-of-sample VaR backtest: GBM-MLE vs Heston-AI-calibrated vs historical VaR. Coverage tables, plots.

**Paper.** Draft sections — abstract, intro, math model, method, experiments, results. Pull abstract seed from [01_overview.md](../getting-started/overview.md) and equations from [math/](../math/index.md) directly.

**Deliverable.** Paper draft v0 in `paper/`. Numerical results frozen.

---

## Week 6 — Polish + (optional) CUDA stretch

**Polish.** Quickstart README, reproducible-figures script, docstrings, CI on push, tag `v0.1.0`.

**Paper.** v1 pass, advisor / friend review, plan arXiv submission.

**Stretch (only if ahead of schedule).** Write **one** CUDA kernel — the path-generation inner loop — behind a `--gpu` flag. Reference benchmark in paper appendix. This is the seed of the deferred [architecture/phase2_cpp.md](../architecture/phase2-cpp.md) library.

**Status (2026-05-01).** Polish complete: end-to-end quickstart in [README.md](../index.md), reproducible figures via [scripts/reproduce_figures.py](https://github.com/AmineChr54/StochasTech/blob/main/scripts/reproduce_figures.py) with helpers in [stochastech/viz/plots.py](https://github.com/AmineChr54/StochasTech/blob/main/stochastech/viz/plots.py), CI runs lint + doc-parity + pytest under pixi ([.github/workflows/ci.yml](https://github.com/AmineChr54/StochasTech/blob/main/.github/workflows/ci.yml)), 120 tests green. CUDA stretch deferred to Phase 2.

---

## Risks and cuts

| Risk | First cut | Last resort |
|------|-----------|-------------|
| Adjoint training fails to converge in Week 3 | Fall back to gradient-free CMA-ES (`cma` package). Frame as a contribution in the paper. | — |
| Real-data calibration unstable in Week 4 | Restrict to SPY + 1 ticker on long windows. | SPY-only, single window. |
| Schedule slips by Week 5 | Drop the CUDA stretch. | Drop second/third tickers. Drop ES (keep VaR). **Never drop the paper** — a finished paper on one ticker beats unfinished code on three. |
