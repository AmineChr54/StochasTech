# Euler–Maruyama discretization

> The numerical scheme used to simulate every SDE in this project. Code: [stochastech/sde/base.py](../../stochastech/sde/base.py).

## Statement

For an Itô SDE $dX_t = \mu(X_t, t)\, dt + \sigma(X_t, t)\, dW_t$, the Euler–Maruyama update on a uniform grid $t_i = i\Delta t$ is:

$$
X_{t_{i+1}} = X_{t_i} + \mu(X_{t_i}, t_i)\, \Delta t + \sigma(X_{t_i}, t_i)\, \sqrt{\Delta t}\, Z_i, \quad Z_i \overset{\text{iid}}{\sim} \mathcal{N}(0, 1).
$$

**Convergence orders** (Kloeden & Platen):

- **Strong** (path-wise, $L^2$): $\mathbb{E}\,|X_T - X_T^{\Delta t}| \le C\, \Delta t^{1/2}$.
- **Weak** (distributional, on smooth functionals $g$): $|\mathbb{E}\,g(X_T) - \mathbb{E}\,g(X_T^{\Delta t})| \le C\, \Delta t$.

## Derivation

### Scheme from the integral form

Integrate $dX_t = \mu(X_t, t)\, dt + \sigma(X_t, t)\, dW_t$ over $[t_i, t_{i+1}]$:

$$
X_{t_{i+1}} = X_{t_i} + \int_{t_i}^{t_{i+1}} \mu(X_s, s)\, ds + \int_{t_i}^{t_{i+1}} \sigma(X_s, s)\, dW_s.
$$

Freeze each integrand at the **left endpoint** (Itô convention — the Stratonovich scheme would use the midpoint and produce a different drift):

$$
\int_{t_i}^{t_{i+1}} \mu(X_s, s)\, ds \approx \mu(X_{t_i}, t_i)\, \Delta t,
$$

$$
\int_{t_i}^{t_{i+1}} \sigma(X_s, s)\, dW_s \approx \sigma(X_{t_i}, t_i)\, \Delta W_i, \qquad \Delta W_i \sim \mathcal{N}(0, \Delta t).
$$

Substituting and writing $\Delta W_i = \sqrt{\Delta t}\, Z_i$ with $Z_i \overset{\text{iid}}{\sim} \mathcal{N}(0, 1)$ gives the scheme stated above.

### Strong order 1/2

Expand $\sigma(X_s, s)$ around $(X_{t_i}, t_i)$ inside the diffusion integral using Itô's lemma applied to $\sigma$:

$$
\sigma(X_s, s) = \sigma(X_{t_i}, t_i) + \int_{t_i}^{s} \mathcal{L}\sigma\, du + \int_{t_i}^{s} \sigma_x \sigma\, dW_u,
$$

where $\mathcal{L}\sigma = \sigma_t + \mu\, \sigma_x + \tfrac{1}{2}\sigma^2 \sigma_{xx}$. The leading omitted term in the diffusion integral is therefore

$$
\int_{t_i}^{t_{i+1}} \int_{t_i}^{s} \sigma_x \sigma\, dW_u\, dW_s,
$$

which is $O(\Delta t)$ in $L^2$ (the integrand under the inner integral is $O(\sqrt{\Delta t})$, and the outer Itô integral preserves that order). The drift integral contributes $O(\Delta t^{3/2})$ in $L^2$ and is dominated. Sum the squared local errors over $\lfloor T/\Delta t \rfloor$ steps:

$$
\mathbb{E}\,|X_T - X_T^{\Delta t}|^2 \le C\, \Delta t,
$$

so $\mathbb{E}\,|X_T - X_T^{\Delta t}| \le C\, \Delta t^{1/2}$ — strong order $1/2$. Including the missing Itô–Taylor term $\sigma_x \sigma\, \tfrac{1}{2}((\Delta W_i)^2 - \Delta t)$ recovers strong order $1$ (Milstein); we do **not** add it here because (a) it requires $\sigma_x$ in closed form and (b) for marginal-distribution / VaR work the relevant rate is the weak rate.

### Weak order 1

For smooth payoffs $g$, expand $\mathbb{E}\,g(X_T) - \mathbb{E}\,g(X_T^{\Delta t})$ using the backward Kolmogorov PDE $u(t, x) = \mathbb{E}[g(X_T)\mid X_t = x]$. Local truncation per step is $O(\Delta t^2)$ — odd-power Brownian moments vanish in expectation, so the $(\Delta W_i)^2 \to \Delta t$ substitution is **exact in mean**. Summing $T/\Delta t$ steps yields total weak error $O(\Delta t)$. This is the regime our Monte Carlo VaR (see [06_monte_carlo_var.md](06_monte_carlo_var.md)) lives in: distributional accuracy at $\Delta t$ rate, with no need for Milstein.

### Why this matters for our project

GBM is special — log-Euler is **exact** (the increment of $\log S$ is Gaussian with no truncation error), so the strong-$1/2$ caveat is moot for the Week 1 baseline. Heston is not so lucky: the variance leg has $\sigma(v) = \xi \sqrt{v}$ with non-Lipschitz $\sigma_x$ at $v = 0$, breaking the standard convergence proof. Full truncation (see [04_heston.md](04_heston.md)) restores order in practice.

## Discretization

The implementation in `stochastech/sde/base.py` exposes:

```python
def euler_maruyama(drift, diffusion, x0, dt, n_steps, generator=None):
    # x0 carries the path-batch shape; output is (n_steps + 1, *x0.shape).
    ...
```

GBM and Heston are special cases; see [03_gbm.md](03_gbm.md) (log-Euler variant) and [04_heston.md](04_heston.md) (full truncation).

## Assumptions and failure modes

- Strong order $\tfrac{1}{2}$ is poor for path-dependent payoffs — use Milstein for those if needed (out of scope v1).
- Stiffness: $\Delta t$ small enough that $|\mu \Delta t|$ and $\sigma \sqrt{\Delta t}$ are both $\ll |X|$. Daily steps are fine for daily-frequency calibration.
- Naive Euler can violate positivity for square-root diffusions (Heston variance) — see full-truncation note in [04_heston.md](04_heston.md).

## References

- Kloeden & Platen, *Numerical Solution of Stochastic Differential Equations*, 1992, ch. 9–10.
- Higham, *An Algorithmic Introduction to Numerical Simulation of SDEs*, SIAM Review, 2001.
