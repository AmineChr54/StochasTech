# Neural SDEs — the chosen approach

> The ML method actually used in this project. Distilled from the original `methods_to_use.md` brainstorm.

## What we're doing

We treat the **Heston SDE itself as the model** and learn its parameters end-to-end. This is the "Neural SDE" framing of Li et al. (2020) and the Chen et al. (2018) Neural ODE family — the network is not a generic MLP that predicts prices; it is the SDE solver, with $(\kappa, \theta, \xi, \rho, v_0)$ as the learnable leaf tensors.

Concretely:
1. Forward solve Heston via Euler–Maruyama (full truncation) to produce simulated returns. See [../math/heston.md](../math/heston.md).
2. Compute a calibration loss against historical returns. See [../math/calibration-losses.md](../math/calibration-losses.md).
3. Backpropagate the loss through the solver via the stochastic adjoint. See [../math/adjoint-sde.md](../math/adjoint-sde.md).
4. Update parameters with Adam.

## Why this and not a generic deep model

| Alternative | Why not for v1 |
|-------------|----------------|
| LSTM / Transformer that predicts $\hat{S}_{t+1}$ from history | Black box, not differentiable through a known physical model, no natural way to extract VaR / ES with calibrated tail behavior. The paper's contribution disappears. |
| PINN that imposes Black–Scholes / Heston PDE residual loss | Useful for derivative pricing; overkill for terminal-distribution VaR on real data, and the "physics" is only as good as the assumed PDE. See [future_work.md](future-work.md). |
| Gradient-free optimization (CMA-ES, random search) of Heston parameters against the same losses | Works, but loses the "differentiable simulation" novelty that motivates the paper. **Used as a sanity-check baseline in Week 4.** |

The Neural-SDE framing also gives us something the alternatives don't: **the ability to swap loss functions cheaply** (NLL vs energy distance vs anything that's a function of simulated paths) and have gradients flow correctly. That's the experiment that makes the paper interesting.

## Implementation

`torchsde` ([google-research/torchsde](https://github.com/google-research/torchsde)) provides `sdeint_adjoint`. We subclass `torchsde.SDEIto` for Heston, expose drift and diffusion, and let the library handle the Brownian-tree bookkeeping.

## References

- Li, Wong, Chen, Duvenaud, *Scalable Gradients for Stochastic Differential Equations*, AISTATS 2020.
- Chen, Rubanova, Bettencourt, Duvenaud, *Neural Ordinary Differential Equations*, NeurIPS 2018.
- Kidger, Foster, Li, Lyons, *Neural SDEs as Infinite-Dimensional GANs*, ICML 2021 — the generative-modeling extension, future-work-tier.
