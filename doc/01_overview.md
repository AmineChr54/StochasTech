# Overview

> What StochasTech is, what it produces, and why.

## One paragraph

StochasTech is a simulation-based framework for financial risk analysis. It models equity price evolution under two stochastic differential equations — Geometric Brownian Motion (GBM, constant volatility) and the Heston model (stochastic volatility) — and uses Monte Carlo simulation to compute Value-at-Risk (VaR) and Expected Shortfall (ES). On top of the simulator sits a differentiable calibration layer that fits Heston parameters $(\kappa, \theta, \xi, \rho, v_0)$ to historical returns by backpropagating through the SDE solver (the "AI" component). The headline result is a head-to-head comparison: does AI-calibrated Heston produce better-calibrated VaR than GBM-MLE on out-of-sample data?

## What gets built

1. **SDE simulator.** Vectorized Euler–Maruyama for GBM and Heston in PyTorch. Antithetic variates for variance reduction.
2. **Risk module.** VaR and ES estimators from Monte Carlo paths. Kupiec POF and Christoffersen independence tests for backtesting.
3. **Differentiable calibrator.** `torchsde`-backed adjoint training of Heston parameters against a chosen loss (log-likelihood and energy/MMD distance both supported).
4. **Real-data harness.** `yfinance` loaders for SPY + 2 single names, rolling-window calibration, persistence of fitted parameters and diagnostics.
5. **Paper.** 8–12 page writeup with derivations, method, experiments, results.

## Final objective

A reproducible system that:
- Simulates financial markets via SDEs.
- Estimates risk via Monte Carlo.
- Compares model realism against real return distributions.
- Calibrates parameters by gradient-based optimization through the SDE solver.
- Produces a published paper alongside the code.

## Key claim of the paper

Accurate financial risk forecasting depends not only on model choice but on parameter calibration. Differentiable calibration through the SDE solver delivers better VaR coverage than closed-form GBM-MLE on out-of-sample data.

## Strategic positioning

The project is designed as a **portfolio + paper** combo:
- For quant firms: demonstrates SDE numerics, Monte Carlo risk, calibration, and backtesting on real data.
- For ML research: demonstrates differentiable simulation, adjoint methods, and end-to-end gradient training through a non-trivial physical model.

## Phase boundaries

| Phase | Content | Status |
|-------|---------|--------|
| 1 | Python/PyTorch MVP described above + paper | This 6-week sprint |
| 2 | C++/CUDA `neuro_ito` library — see [architecture/phase2_cpp.md](architecture/phase2_cpp.md) | Backlog |
| 3 | Integration with the live ORIA portfolio pipeline — see [reference/oria_inspiration.md](reference/oria_inspiration.md) | Backlog |
