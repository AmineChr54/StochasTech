# Calibration losses

> The objectives optimized to fit Heston parameters $(\kappa, \theta, \xi, \rho, v_0)$ to historical returns. Code: [stochastech/calibration/losses.py](https://github.com/AmineChr54/StochasTech/blob/main/stochastech/calibration/losses.py).

## Statement

Two losses are implemented. Both are minimized with Adam through the adjoint solver (see [adjoint-sde.md](adjoint-sde.md)).

### Loss A — Negative log-likelihood of returns

Approximate the conditional density $p_\theta(r_t \mid \mathcal{F}_{t-1})$ by simulating $M$ paths and forming a kernel density estimate, or by computing the per-step log-density of the discrete Euler update under the Gaussian increment assumption:

$$
\mathcal{L}_{\mathrm{NLL}}(\theta) = -\sum_{t=1}^{T} \log p_\theta(r_t \mid \mathcal{F}_{t-1}).
$$

For full-truncation Euler on Heston, the one-step log-return given $v_t$ is approximately Gaussian with mean $(\mu - \tfrac{1}{2} v_t^+) \Delta t$ and variance $v_t^+ \Delta t$ — but $v_t$ itself is latent ⇒ marginalization via simulation.

### Loss B — Energy distance / MMD

Distribution-matching loss. Given empirical return sample $\{r_i\}$ and simulated sample $\{\tilde{r}_j(\theta)\}$:

$$
\mathcal{L}_{\mathrm{ED}}(\theta) = 2\, \mathbb{E}\,\|R - \tilde{R}\| - \mathbb{E}\,\|R - R'\| - \mathbb{E}\,\|\tilde{R} - \tilde{R}'\|,
$$

a U-statistic of pairwise distances. Energy distance is zero iff distributions match, differentiable in $\theta$ through $\tilde{R}$, and robust to the path-density approximation that NLL needs.

## Derivation

### NLL: optimal under correct specification, fragile otherwise

If the true data-generating process really is Heston with parameters $\theta^\star$, then maximum-likelihood is asymptotically efficient: $\sqrt{T}(\hat\theta_{\mathrm{NLL}} - \theta^\star) \Rightarrow \mathcal{N}(0, I(\theta^\star)^{-1})$ where $I$ is the Fisher information. No other consistent estimator achieves a smaller asymptotic covariance — that is the Cramér–Rao bound.

The fragility kicks in when the model is **misspecified**. NLL minimizes the KL divergence $\mathrm{KL}(p_{\mathrm{data}} \,\|\, p_\theta)$, which is asymmetric: it penalizes regions where $p_{\mathrm{data}} > 0$ but $p_\theta \approx 0$ catastrophically. Heston's lognormal-conditional return distribution under-weights the deep tails of equity returns (heavier than lognormal — see Mandelbrot 1963, Cont 2001). NLL responds by inflating $\xi$ (vol of vol) to manufacture tails, distorting the bulk of the distribution to compensate. The fitted parameters then poorly reproduce the *bulk* properties (mean reversion, leverage) the model was supposed to capture.

Energy distance / MMD-style losses minimize a symmetric integral-probability metric: $|\!|p_{\mathrm{data}} - p_\theta|\!|$ in some RKHS. They tolerate misspecification more gracefully — a tail mismatch contributes a bounded amount, not infinite KL.

### Energy distance as squared MMD

Energy distance and MMD are the same object viewed from two angles. For the kernel $k(x, y) = -\|x - y\|$ (conditionally negative-definite, so MMD is well-defined),

$$
\mathrm{MMD}^2_k(P, Q) = \mathbb{E}_{X,X' \sim P}\, k(X, X') - 2\,\mathbb{E}_{X \sim P, Y \sim Q}\, k(X, Y) + \mathbb{E}_{Y, Y' \sim Q}\, k(Y, Y').
$$

Substituting $k(x,y) = -\|x - y\|$ and rearranging:

$$
\mathrm{MMD}^2(P, Q) = 2\,\mathbb{E}\,\|X - Y\| - \mathbb{E}\,\|X - X'\| - \mathbb{E}\,\|Y - Y'\| = \mathcal{E}(P, Q),
$$

so the energy distance is exactly squared MMD with the negative-Euclidean kernel. It is zero iff $P = Q$ (Székely & Rizzo 2013). This justifies treating it as a proper distance for distribution matching and connects it to the kernel-two-sample-test literature (Gretton et al. 2012).

### Plug-in estimator and its bias

Given iid samples $\{X_i\}_{i=1}^n \sim P$ and $\{Y_j\}_{j=1}^m \sim Q$, the **plug-in** estimator replaces expectations by sample means:

$$
\widehat{\mathcal{E}} = \frac{2}{nm}\sum_{i,j}\|X_i - Y_j\| - \frac{1}{n^2}\sum_{i,i'}\|X_i - X_{i'}\| - \frac{1}{m^2}\sum_{j,j'}\|Y_j - Y_{j'}\|.
$$

This is what `energy_distance(...)` in [stochastech/calibration/losses.py](https://github.com/AmineChr54/StochasTech/blob/main/stochastech/calibration/losses.py) computes. The "self" sums include the $i = i'$ diagonal, which adds $\|X_i - X_i\| = 0$ — so the diagonal contributes nothing to the magnitude but inflates the denominator from $n(n-1)$ to $n^2$, giving a small downward bias of order $1/n$. The unbiased $U$-statistic excludes the diagonal:

$$
\widetilde{\mathcal{E}}_{\mathrm{self}} = \frac{1}{n(n-1)}\sum_{i \ne i'} \|X_i - X_{i'}\|.
$$

For $n \gtrsim 10^3$ the bias is below Monte Carlo noise, so the plug-in form is the default; switch to the U-statistic if $n < 100$ and bias matters.

Both losses cost $O(n_s n_o + n_s^2 + n_o^2)$ memory in pairwise distances. The calibrator subsamples the simulated returns to a cap (default 2048) before forming the loss matrix.

### Why energy distance for Heston specifically

The energy distance differentiates cleanly through `simulate_heston_diff` because the only operations on $\theta$ are (i) the SDE step (smooth in parameters), (ii) `(s[1:]/s[:-1]).log()`, and (iii) pairwise absolute differences (subdifferentiable everywhere; differentiable a.s.). NLL with KDE is equally smooth but the bandwidth choice introduces a hyperparameter the user must tune, and the per-sample log-density inflates noise from the rare tail observations. Energy distance is therefore the **default** loss; NLL remains available for reproducibility against MLE-based literature.

## Discretization

Both losses are computed on PyTorch tensors of simulated and historical returns. Energy distance has $O(M^2)$ memory in batch size — chunk if $M > 5000$.

## Assumptions and failure modes

- **NLL** is sharper but requires that the simulator's density is well-defined. Variance reduction via control variates can bias the density estimate — keep antithetic only.
- **Energy distance** is more forgiving but has weaker identification: two distinct $(\rho, \xi)$ pairs can produce nearly indistinguishable return distributions on short windows. Mitigate with longer windows or by adding a penalty on parameter drift between windows.
- Both losses are non-convex in $\theta$ — Adam with multiple random restarts; check convergence on the loss curve, not just final value.

## Week 4 — Empirical estimation pitfalls

### $\rho$ vs $\xi$ identifiability

The two parameters governing the variance leg's volatility-of-volatility ($\xi$) and its correlation with returns ($\rho$) trade off in their effect on the return distribution. Both control the conditional skewness of returns through different channels:

- $\rho < 0$ produces left skew via the leverage channel (variance up when returns down).
- Larger $\xi$ produces fatter tails by widening the conditional distribution at high-variance times, which compounds with $\rho$ to produce left skew when $\rho < 0$.

On price data alone (no implied-volatility surface), the Fisher information matrix has a near-zero eigenvalue along the direction $(\Delta \rho \cdot \xi + \rho \cdot \Delta \xi) = 0$ in $(\rho, \xi)$-space. Calibration on a single ticker × short window will produce wide posteriors on $(\rho, \xi)$; the rolling-window sweep in `scripts/run_calibration.py` exposes this as parameter-trajectory noise on the $(\rho, \xi)$ pair while $(\kappa, \theta, v_0)$ remain stable.

Mitigations: (a) calibrate jointly on returns + implied-volatility (out of scope here, but the standard fix), (b) regularize $\rho$ near $-0.5$ to $-0.7$ as an empirical prior, (c) report posterior dispersion across windows rather than point estimates.

### Regime breaks

Heston is a single-regime model. Calibrating across COVID March 2020, the GFC (2008–09), or other vol-regime transitions inflates $\xi$ to absorb the inter-regime variance jump that the model can't represent structurally. Diagnostic: sharp jumps in fitted $\xi$ at window boundaries that straddle a known regime break. Workarounds: window the data to exclude regime changes, or fit a switching-Heston (out of scope v1).

### Cross-validation for SDE models

Standard $k$-fold CV breaks the time-series structure — random folds leak future variance into training. Use **expanding-window** or **forward-chaining** CV instead: train on $[0, t]$, validate on $[t, t+h]$, advance $t$. The rolling-window calibrator in `rolling_window_calibration` (in [stochastech/calibration/heston_fit.py](https://github.com/AmineChr54/StochasTech/blob/main/stochastech/calibration/heston_fit.py)) is the same pattern with the validation step deferred to Week 5's VaR backtest, where each window's fitted parameters generate the next window's out-of-sample VaR forecasts.

## References

- Gretton, Borgwardt, Rasch, Schölkopf, Smola, *A Kernel Two-Sample Test*, JMLR 2012 (MMD).
- Székely, Rizzo, *Energy statistics: A class of statistics based on distances*, JSPI 2013.
- Aït-Sahalia & Kimmel, *Maximum likelihood estimation of stochastic volatility models*, JFE 2007.
