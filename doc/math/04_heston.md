# Heston stochastic volatility model

> Stochastic-volatility model whose differentiable calibration is the project's headline contribution. Code: [stochastech/sde/heston.py](../../stochastech/sde/heston.py).

## Statement

Heston (1993) couples the asset price $S_t$ to a stochastic variance $v_t$:

$$
\begin{aligned}
dS_t &= \mu S_t\, dt + \sqrt{v_t}\, S_t\, dW_t^S, \\
dv_t &= \kappa(\theta - v_t)\, dt + \xi \sqrt{v_t}\, dW_t^v, \\
d\langle W^S, W^v \rangle_t &= \rho\, dt.
\end{aligned}
$$

**Parameters:**

| Symbol | Meaning | Typical range |
|--------|---------|---------------|
| $\mu$ | drift | $\sim 0.05$/yr |
| $\kappa$ | mean-reversion speed | $0.5$–$5$ |
| $\theta$ | long-run variance | $0.02$–$0.10$ |
| $\xi$ | vol of vol | $0.1$–$0.6$ |
| $\rho$ | correlation $W^S, W^v$ | $-0.9$–$0$ (leverage effect) |
| $v_0$ | initial variance | $\sim \theta$ |

## Derivation

### Why a CIR variance process

The variance leg $dv_t = \kappa(\theta - v_t)\, dt + \xi \sqrt{v_t}\, dW_t^v$ is the Cox–Ingersoll–Ross (CIR) process. Three properties make it the canonical choice:

1. **Mean reversion.** Drift $\kappa(\theta - v_t)$ pulls $v_t$ toward $\theta$ at rate $\kappa$. Empirically variance does not random-walk to infinity; it oscillates around a long-run level.
2. **Positivity (Feller).** Diffusion vanishes as $v \to 0^+$ at rate $\xi\sqrt{v}$, slow enough that the upward drift $\kappa\theta$ dominates near zero whenever $2\kappa\theta \ge \xi^2$ (the **Feller condition**). Under that condition $v_t > 0$ almost surely. A constant-coefficient diffusion ($\sigma\, dW$) would be Gaussian and routinely cross zero.
3. **Tractability.** $v_t$ has a known noncentral-chi-squared transition density, and the joint $(\log S, v)$ characteristic function is closed-form — that is what makes the original Heston paper a *closed-form* options pricer.

The square-root form is also the Bessel-process scaling that comes out of $v_t = X_t^\top X_t$ for a multidimensional OU $X$ — variance-as-squared-distance is a natural model.

### Log-price dynamics by Itô

Apply Itô's lemma to $f(s) = \log s$ on $dS_t = \mu S_t\, dt + \sqrt{v_t}\, S_t\, dW_t^S$. With $f_s = 1/s$, $f_{ss} = -1/s^2$ and $\sigma(S, v) = \sqrt{v}\, S$:

$$
d(\log S_t) = \left( \mu - \tfrac{1}{2} v_t \right) dt + \sqrt{v_t}\, dW_t^S.
$$

The Itô correction is now path-dependent — $-\tfrac{1}{2} v_t\, dt$ rather than the constant $-\tfrac{1}{2}\sigma^2\, dt$ of GBM. This is the source of fat tails: when $v_t$ spikes, log-returns get a big random shock and a big negative drift simultaneously.

### Correlated Brownians and the leverage effect

The two driving Brownians satisfy $d\langle W^S, W^v\rangle_t = \rho\, dt$. Construct them from independent $\tilde W^v, \tilde W^\perp$ via the Cholesky factor

$$
W^v_t = \tilde W^v_t, \qquad W^S_t = \rho\, \tilde W^v_t + \sqrt{1 - \rho^2}\, \tilde W^\perp_t,
$$

so that $\mathrm{Var}(W^S_t) = t$ and $\mathrm{Cov}(W^S_t, W^v_t) = \rho t$.

**Leverage effect.** Empirically equity volatility rises when prices fall — Black 1976. In Heston this is encoded by $\rho < 0$: a negative shock to $W^S$ depresses returns; the same shock (correlated through $\rho$) lifts variance, so the very next return has a larger expected absolute size. Calibrated $\rho$ on US equities is typically $-0.6$ to $-0.8$.

$\rho = 0$ shuts the leverage channel off and produces a symmetric implied-volatility smile; $\rho \ne 0$ produces the empirically observed **smirk** (steeper on the OTM-put side).

### Joint dynamics for simulation

Combining, the simulation-ready system is

$$
\begin{aligned}
d(\log S_t) &= \left( \mu - \tfrac{1}{2} v_t \right) dt + \sqrt{v_t}\, dW_t^S, \\
dv_t &= \kappa(\theta - v_t)\, dt + \xi \sqrt{v_t}\, dW_t^v, \\
W^S_t &= \rho \tilde W^v_t + \sqrt{1 - \rho^2}\, \tilde W^\perp_t, \quad W^v_t = \tilde W^v_t.
\end{aligned}
$$

This is what the discretization below digitizes.

## Discretization (full truncation)

Naive Euler–Maruyama on $v_t$ can drive variance negative. **Full-truncation Euler** — the scheme used in `heston.py` — replaces $v_t$ with $v_t^+ = \max(v_t, 0)$ inside the diffusion and drift:

$$
\begin{aligned}
v_{t+\Delta t} &= v_t + \kappa(\theta - v_t^+)\,\Delta t + \xi \sqrt{v_t^+}\, \sqrt{\Delta t}\, Z^v, \\
\log S_{t+\Delta t} &= \log S_t + \left(\mu - \tfrac{1}{2} v_t^+\right) \Delta t + \sqrt{v_t^+}\, \sqrt{\Delta t}\, Z^S, \\
Z^S &= \rho Z^v + \sqrt{1 - \rho^2}\, Z^\perp, \quad Z^v, Z^\perp \overset{\text{iid}}{\sim} \mathcal{N}(0,1).
\end{aligned}
$$

The implementation is `simulate_heston(...)` in [stochastech/sde/heston.py](../../stochastech/sde/heston.py); a helper `feller_condition(kappa, theta, xi)` in the same module reports whether the calibrated parameters satisfy $2\kappa\theta \ge \xi^2$.

## Assumptions and failure modes

- **Feller condition:** $2\kappa\theta \ge \xi^2$ keeps $v_t > 0$ almost surely in the continuous SDE. Discrete schemes can violate it ⇒ full truncation needed. Document whether calibrated parameters satisfy Feller.
- **Identifiability:** $\rho$ and $\xi$ are jointly hard to identify from price data alone — implied-vol surfaces help but are out of scope here. Expect wide posterior on these two.
- **Regime breaks:** Heston is single-regime; calibration on data spanning a regime change (e.g., COVID March 2020) will produce inflated $\xi$.

## References

- Heston, *A Closed-Form Solution for Options with Stochastic Volatility*, Review of Financial Studies, 1993.
- Lord, Koekkoek, van Dijk, *A Comparison of Biased Simulation Schemes for Stochastic Volatility Models*, 2010 (full-truncation analysis).
- Andersen, *Efficient Simulation of the Heston Stochastic Volatility Model*, 2008.
