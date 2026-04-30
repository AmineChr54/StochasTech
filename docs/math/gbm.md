# Geometric Brownian Motion (GBM)

> Constant-volatility baseline model for asset prices. Code: [stochastech/sde/gbm.py](https://github.com/AmineChr54/StochasTech/blob/main/stochastech/sde/gbm.py).

## Statement

GBM is the SDE

$$
dS_t = \mu S_t\, dt + \sigma S_t\, dW_t, \quad S_0 > 0,
$$

with constant drift $\mu \in \mathbb{R}$ and constant volatility $\sigma > 0$.

**Closed-form solution:**

$$
S_t = S_0 \exp\!\left( \left(\mu - \tfrac{1}{2}\sigma^2\right) t + \sigma W_t \right).
$$

**Moments:**

$$
\mathbb{E}[S_t] = S_0 e^{\mu t}, \qquad \mathrm{Var}(S_t) = S_0^2 e^{2\mu t} \left( e^{\sigma^2 t} - 1 \right).
$$

## Derivation

### Closed form via Itô's lemma

Apply Itô's lemma (see [ito-calculus.md](ito-calculus.md)) to $f(s) = \log s$, with $f_s = 1/s$, $f_{ss} = -1/s^2$, $f_t = 0$. Plug into

$$
df = \left( f_t + \mu S\, f_s + \tfrac{1}{2} \sigma^2 S^2\, f_{ss} \right) dt + \sigma S\, f_s\, dW_t,
$$

so

$$
d(\log S_t) = \left( \mu - \tfrac{1}{2}\sigma^2 \right) dt + \sigma\, dW_t.
$$

Integrate from $0$ to $t$:

$$
\log S_t - \log S_0 = \left( \mu - \tfrac{1}{2}\sigma^2 \right) t + \sigma W_t,
$$

and exponentiate:

$$
S_t = S_0 \exp\!\left( \left( \mu - \tfrac{1}{2}\sigma^2 \right) t + \sigma W_t \right). \tag{$\star$}
$$

### Moments

From $(\star)$, $\log(S_t / S_0) \sim \mathcal{N}((\mu - \tfrac{1}{2}\sigma^2) t,\, \sigma^2 t)$. For $X \sim \mathcal{N}(m, v)$ the lognormal moments are $\mathbb{E}[e^X] = e^{m + v/2}$ and $\mathrm{Var}(e^X) = e^{2m + v}(e^v - 1)$. Substituting $m = (\mu - \tfrac{1}{2}\sigma^2) t$, $v = \sigma^2 t$:

$$
\mathbb{E}[S_t] = S_0 e^{m + v/2} = S_0 e^{\mu t},
$$

$$
\mathrm{Var}(S_t) = S_0^2 e^{2m + v}(e^v - 1) = S_0^2 e^{2 \mu t}\,( e^{\sigma^2 t} - 1).
$$

The drift $\mu$ governs the mean; the $-\tfrac{1}{2}\sigma^2$ correction inside the exponent is the volatility drag — it disappears from $\mathbb{E}[S_t]$ but shows up in any path-wise quantity (median, log-return mean).

### MLE on log-returns

From $(\star)$, the daily log-return $r_i = \log(S_{t_{i+1}}/S_{t_i}) \sim \mathcal{N}((\mu - \tfrac{1}{2}\sigma^2) \Delta t,\, \sigma^2 \Delta t)$ iid. Sample mean $\bar r$ and variance $s_r^2$ are sufficient:

$$
\hat\sigma^2 = s_r^2 / \Delta t, \qquad \hat\mu = \bar r / \Delta t + \tfrac{1}{2}\hat\sigma^2.
$$

This closed-form is the GBM-MLE baseline that Week 5 backtests against the Heston-AI calibration.

## Discretization

Two equivalent schemes are implemented:

- **Naive Euler–Maruyama on $S_t$:**
  $S_{t+\Delta t} = S_t + \mu S_t\, \Delta t + \sigma S_t\, \sqrt{\Delta t}\, Z$, $Z \sim \mathcal{N}(0,1)$. Can produce negative prices for large $\sigma\sqrt{\Delta t}$ — avoid for production.
- **Log-Euler (preferred):**
  $\log S_{t+\Delta t} = \log S_t + (\mu - \tfrac{1}{2}\sigma^2)\,\Delta t + \sigma\sqrt{\Delta t}\, Z$. Exact for GBM and strictly positive.

The implementation is `simulate_gbm(...)` in [stochastech/sde/gbm.py](https://github.com/AmineChr54/StochasTech/blob/main/stochastech/sde/gbm.py); pass `log_euler=True` (default) for the exact scheme, `log_euler=False` to fall through to the generic [stochastech/sde/base.py](https://github.com/AmineChr54/StochasTech/blob/main/stochastech/sde/base.py) Euler–Maruyama for comparison.

## Assumptions and failure modes

- Constant $\sigma$ — empirically false (volatility clustering). Motivation for Heston (see [heston.md](heston.md)).
- Log-normal returns ⇒ no fat tails. VaR will under-estimate tail risk on real data; this is **the empirical comparison Week 5 makes**.
- MLE for $(\mu, \sigma)$ on log-returns has closed form: $\hat{\mu} = \bar{r}/\Delta t + \tfrac{1}{2}\hat{\sigma}^2$, $\hat{\sigma}^2 = \mathrm{Var}(r)/\Delta t$, where $r_i = \log(S_{t_{i+1}}/S_{t_i})$.

## References

- Shreve, ch. 5.
- Hull, *Options, Futures, and Other Derivatives*, ch. 14.
