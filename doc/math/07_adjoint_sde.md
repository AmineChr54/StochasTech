# Adjoint method for SDEs

> The technique that lets us backpropagate through the Heston SDE solver to fit parameters by gradient descent. Code: [stochastech/calibration/heston_fit.py](../../stochastech/calibration/heston_fit.py).

## Statement

For an SDE $dX_t = f_\theta(X_t, t)\,dt + g_\theta(X_t, t)\,dW_t$ with parameters $\theta$ and a scalar loss $\mathcal{L}(X_T)$, naive backprop through the solver costs $O(N \cdot |\theta|)$ memory in the number of time steps $N$.

The **stochastic adjoint** (Li et al., 2020) replaces this with a backward SDE that recomputes the forward path on the fly, giving $O(|\theta|)$ memory at the cost of one extra forward pass.

The adjoint state $a_t = \partial \mathcal{L} / \partial X_t$ satisfies the backward SDE:

$$
da_t = -a_t^\top \left( \frac{\partial f_\theta}{\partial X_t}\, dt + \frac{\partial g_\theta}{\partial X_t}\, dW_t \right),
$$

integrated from $t = T$ down to $t = 0$ with terminal condition $a_T = \nabla_{X_T} \mathcal{L}$, against the **same** Brownian path used in the forward solve.

The parameter gradient is then:

$$
\nabla_\theta \mathcal{L} = -\int_0^T a_t^\top \left( \frac{\partial f_\theta}{\partial \theta}\, dt + \frac{\partial g_\theta}{\partial \theta}\, dW_t \right).
$$

## Derivation

### Lagrangian for the constrained loss

Define the SDE constraint in integral form $X_T = X_0 + \int_0^T f_\theta(X_s, s)\,ds + \int_0^T g_\theta(X_s, s)\,dW_s$. The constrained-optimization Lagrangian is

$$
\mathcal{L}_a(\theta, X, a) = \mathbb{E}\Big[ \mathcal{L}(X_T) - \int_0^T a_s^\top \big( dX_s - f_\theta\,ds - g_\theta\,dW_s \big) \Big],
$$

with multiplier process $a_t$ adapted to $\mathcal{F}_t$. At a stationary point $\delta \mathcal{L}_a = 0$ for variations of $X$ and $\theta$ separately.

### Backward SDE for the adjoint

Vary $X_t \to X_t + \delta X_t$ with $\delta X_0 = 0$. By Itô's lemma applied to the inner product $a^\top X$:

$$
\int_0^T \delta\!\left( a_s^\top dX_s \right) = a_T^\top \delta X_T - \int_0^T (\delta X_s)^\top da_s - \int_0^T \delta(\partial_X f_\theta\, a_s + \partial_X g_\theta\, a_s\, dW_s) \cdot \delta X_s \cdots
$$

(performing the integration by parts in the Itô sense). Collecting terms, $\delta \mathcal{L}_a / \delta X = 0$ forces the adjoint to satisfy the **backward SDE**

$$
\boxed{\,da_t = -\Big( \partial_X f_\theta(X_t, t) \Big)^\top a_t\, dt - \Big( \partial_X g_\theta(X_t, t) \Big)^\top a_t\, dW_t,\,}
$$

with terminal condition $a_T = \nabla_{X_T} \mathcal{L}(X_T)$. Note the sign: $a_t$ is integrated **backwards** in time. Crucially the Itô integral on the right-hand side is against the **same** Brownian path $W$ used in the forward solve — the adjoint cannot use a fresh draw.

### Parameter gradient

With the adjoint solved, varying $\theta$:

$$
\nabla_\theta \mathcal{L} = -\mathbb{E}\!\left[ \int_0^T \big( \partial_\theta f_\theta(X_t, t) \big)^\top a_t\, dt + \int_0^T \big( \partial_\theta g_\theta(X_t, t) \big)^\top a_t\, dW_t \right].
$$

The first integral is a Riemann integral of the drift sensitivity weighted by the costate; the second is an Itô integral of the diffusion sensitivity. Both can be accumulated alongside the backward solve.

### Brownian reconstruction

The backward sweep needs $W_t$ at arbitrary times — but storing $W$ on the forward pass would defeat the $O(|\theta|)$ memory goal. The fix is the **virtual Brownian tree**: fix a seed and a depth $L$; on the forward pass, query $W_t$ via a deterministic function of $(t, \text{seed})$ that splits intervals lazily. The same tree replayed on the backward pass produces bit-identical increments without ever materializing the full path.

Implementation note: `torchsde` ships this as `torchsde.BrownianTree` and wires it automatically when `sdeint_adjoint` is called.

### When BPTT wins instead

The adjoint pays a forward-recompute cost (twice through the SDE) for the memory win. For models with $|\theta| = O(1)$ and short time horizons — exactly Heston's situation, with 6 calibrated parameters and ~250 daily steps — vanilla **backprop-through-time** stores $O(N)$ activations that fit comfortably in RAM and runs at ~1× cost rather than 2×. This project's calibrator (see `fit_heston` in [stochastech/calibration/heston_fit.py](../../stochastech/calibration/heston_fit.py) and `simulate_heston_diff` in [stochastech/sde/heston.py](../../stochastech/sde/heston.py)) defaults to BPTT for that reason. The adjoint is the right tool when $N \gg 10^3$ or $|\theta| \gg 10^2$ (e.g., neural-SDE drift networks); it is over-engineered for plain Heston.

### Unbiasedness conditions

Li et al. prove the adjoint estimator is unbiased provided $f_\theta, g_\theta$ are uniformly Lipschitz in $X$ with at-most-linear growth and the loss $\mathcal{L}$ is $C^1$ with at-most-polynomial growth. Heston's $g_\theta(v) = \xi \sqrt{v}$ violates Lipschitz at $v = 0$ — formally the theorem does not apply, and bias does enter through the full-truncation clamp. Empirically, on moderate-volatility regimes the bias is dominated by Monte Carlo noise; the gradient-check script (see `scripts/grad_check.py`) confirms agreement between BPTT and central differences to several significant figures.

## Discretization

Use `torchsde.sdeint_adjoint` rather than rolling our own — it handles the Brownian-tree bookkeeping correctly. The Heston solver is wired into `torchsde` by subclassing `torchsde.SDEIto` and exposing `f`, `g`, `noise_type='general'`, `sde_type='ito'`.

## Assumptions and failure modes

- Discretization bias: adjoint is unbiased only in the continuous limit. Empirical check — gradient-finite-difference smoke test in `tests/test_adjoint_gradcheck.py`.
- Stiff dynamics: tight Heston regimes (small $\kappa$, large $\xi$) can produce noisy gradients ⇒ averaging across path batches before parameter update is essential.
- Memory wins are real but compute cost is ~2× a vanilla solve. For small $|\theta|$ (Heston has 5 params), backprop-through-time may be competitive — benchmark both.

## References

- Li, Wong, Chen, Duvenaud, *Scalable Gradients for Stochastic Differential Equations*, AISTATS 2020 — the canonical reference.
- Chen, Rubanova, Bettencourt, Duvenaud, *Neural Ordinary Differential Equations*, NeurIPS 2018 — deterministic-case adjoint.
- [google-research/torchsde](https://github.com/google-research/torchsde) — implementation we use.
