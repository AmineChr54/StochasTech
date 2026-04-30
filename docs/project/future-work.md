# Future-work parking lot

> ML methods considered and explicitly deferred. Recorded so they don't keep re-entering scope discussions.

The original `methods_to_use.md` brainstorm cast a wide net across architectures. v1 commits to Neural SDEs only (see [neural_sde.md](methods.md)). Everything below is parked for Phase 3+.

## Physics-Informed Neural Networks (PINNs)

**Where they win:** structured derivative-pricing problems with a known governing PDE — Black–Scholes, Heston pricing PDE. Useful for computing the Greeks (Delta, Gamma, Vega) by autodiff, and for data-scarce exotic instruments where the PDE provides regularization.

**Where they lose:** high-frequency directional prediction. Markets are adversarial — there's no universal physical law, so forcing returns into a PDE residual leads to model misspecification. PINNs also converge slowly because the loss carries a PDE-residual term.

**Verdict:** Parked. If we extend to option-pricing benchmarks in a Phase 2 paper, PINNs become relevant. Reference: [maziarraissi/PINNs](https://github.com/maziarraissi/PINNs).

## Transformers (e.g., Temporal Fusion)

Strong for stock/crypto sequence prediction — captures long-range dependencies and regime shifts better than LSTMs. But this is a *price-prediction* tool, not a *risk-modeling* tool. It would belong in a separate alpha-signal layer that feeds ORIA (Phase 3), not in the SDE / VaR pipeline.

## Graph Neural Networks (GNNs)

Useful for cross-asset correlation structure (e.g., NVDA → TSMC propagation). Out of scope because v1 is single-asset univariate Heston. A multivariate Heston extension could use a GNN to parameterize the correlation structure — this is a credible Phase 2 paper extension.

## Reinforcement Learning (PPO / SAC)

For execution and portfolio management — when to buy/sell to minimize market impact and maximize Sharpe. Belongs in the ORIA Execution Governor layer, not in the SDE/risk pipeline.

## Hybrid PINN-FNO

Combines PINNs with Fourier Neural Operators to handle non-linear volatility spikes and multi-scale dynamics. Mathematically interesting but needs the C++/CUDA spectral solver from [../architecture/phase2_cpp.md](../architecture/phase2-cpp.md) to be efficient. Phase 3+.

## Diffusion / Langevin-dynamics generative models

Modern diffusion models are mathematically Langevin SDEs — deeply related to what we're doing. Worth flagging because it's an obvious paper-extension direction: generate full price-path distributions rather than fitting parametric Heston. Probably the most natural Phase 2 research paper.

## References

- Original brainstorm: archived [`../archive/methods_to_use.md`]() (post-migration).
- Video: [The Physics of A.I.](https://www.youtube.com/watch?v=dRkehLL19Wo) — context for the physics-ML intersection.
- Video: [Chaos: The Science of the Butterfly Effect](https://youtu.be/fDek6cYijxI) — chaos vs randomness.
