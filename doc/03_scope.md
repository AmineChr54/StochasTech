# Scope — v1

> Explicit in/out for the 6-week Phase 1 sprint. Anything not listed as in-scope is out-of-scope by default; if you want to add something, write down why and which in-scope item it displaces.

## In scope

- **GBM and Heston SDE simulators.** Euler–Maruyama, full truncation for Heston variance, antithetic variates.
- **Risk metrics.** VaR (95%, 99%) and ES from Monte Carlo paths.
- **Differentiable calibration.** Adjoint-method gradient training of Heston parameters $(\kappa, \theta, \xi, \rho, v_0)$ via `torchsde`.
- **Calibration loss.** Log-likelihood of returns and one distributional distance (energy or MMD).
- **Real-data harness.** SPY + 2 single names via `yfinance`. Rolling-window calibration. 3 tickers × 3 windows.
- **Backtesting.** Kupiec POF + Christoffersen independence tests. Out-of-sample coverage tables.
- **Paper.** 8–12 pages, reproducible from `scripts/reproduce_figures.py`.
- **CI.** Tests + math-doc parity check on push.

## Out of scope (deferred to later phases)

| Item | Phase | Notes |
|------|-------|-------|
| C++/CUDA `neuro_ito` library | 2 | Spec preserved in [architecture/phase2_cpp.md](architecture/phase2_cpp.md). One CUDA kernel is allowed as a Week 6 stretch and that's it. |
| ORIA portfolio integration | 3 | The live $100k ORIA fund stays untouched. See [reference/oria_inspiration.md](reference/oria_inspiration.md) for context — paper cites it as motivation only. |
| Fourier Neural Operators / PINN-FNO hybrids | 3+ | From the archived Aether-Flow vision. See [methods/future_work.md](methods/future_work.md). |
| Transformer / GNN signal generators | 3+ | See [methods/future_work.md](methods/future_work.md). |
| Multi-domain physics (Navier–Stokes, fluid dynamics, Hamiltonian mechanics) | Never (in this project) | The original `roadmap.md` Aether-Flow scope is permanently archived. Finance only. |
| High-frequency / limit-order-book data | 3+ | Daily closes only for v1. |
| Live deployment (FastAPI / NVIDIA Triton / Docker images) | 3+ | Paper + reproducible repo only. |

## Doc parity rule (binding)

A pull request that adds or modifies any module under `stochastech/sde/` or `stochastech/calibration/` is **incomplete** unless the corresponding `doc/math/*.md` is added or updated in the same PR. CI enforces this with a name-matching check.

The math doc template is documented in [math/00_index.md](math/00_index.md). Required sections: Statement, Derivation, Discretization, Assumptions and failure modes, References.

## Cut-priority order (when slipping)

1. Drop the Week 6 CUDA stretch.
2. Drop the second and third tickers (keep SPY only).
3. Drop ES (keep VaR).
4. Drop one of the two losses (keep log-likelihood).

**Never drop the paper.** A finished paper on one ticker outranks unfinished code on three.
