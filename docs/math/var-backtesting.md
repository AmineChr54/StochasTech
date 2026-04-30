# VaR backtesting — Kupiec POF and Christoffersen tests

> Out-of-sample evaluation of VaR forecasts. Code: [stochastech/risk/backtest.py](https://github.com/AmineChr54/StochasTech/blob/main/stochastech/risk/backtest.py).

## Statement

Given a sequence of one-step-ahead VaR forecasts $\widehat{\mathrm{VaR}}_t$ at level $\alpha$ and realized losses $L_t = -r_t$, define the violation indicator

$$
I_t = \mathbf{1}\{ L_t > \widehat{\mathrm{VaR}}_t \}, \quad t = 1, \ldots, n.
$$

A correctly specified VaR model satisfies two properties (Christoffersen 1998):

1. **Unconditional coverage.** $\mathbb{E}[I_t] = 1 - \alpha$.
2. **Independence.** $I_t \perp \mathcal{F}_{t-1}$ — violations are not serially correlated.

Failing (1) means the model is biased; failing (2) means it misses volatility clustering even if its average level is right. The Kupiec POF test addresses (1); the Christoffersen test addresses (2); the joint conditional-coverage test combines them.

### Kupiec POF (proportion of failures)

With $x = \sum_t I_t$ violations in $n$ observations under $H_0: \pi = \pi_0 = 1 - \alpha$:

$$
LR_{\mathrm{POF}} = -2 \log \frac{(1-\pi_0)^{n - x}\, \pi_0^x}{(1-\hat\pi)^{n-x}\, \hat\pi^x},
\qquad \hat\pi = x/n,
$$

asymptotically $\chi^2_1$ under $H_0$. The test is two-sided: too few **and** too many violations both reject.

### Christoffersen independence

Build the $2 \times 2$ count of consecutive states $N_{ij}$ for $i, j \in \{0, 1\}$ (transition from $I_{t-1} = i$ to $I_t = j$). Estimate per-row transition probabilities $\hat\pi_{i1} = N_{i1} / (N_{i0} + N_{i1})$. Under $H_0$ (independence) $\pi_{01} = \pi_{11} = \hat\pi$ (the marginal):

$$
LR_{\mathrm{IND}} = -2 \log \frac{(1-\hat\pi)^{N_{00} + N_{10}}\, \hat\pi^{N_{01} + N_{11}}}{(1-\hat\pi_{01})^{N_{00}}\, \hat\pi_{01}^{N_{01}}\, (1-\hat\pi_{11})^{N_{10}}\, \hat\pi_{11}^{N_{11}}},
$$

asymptotically $\chi^2_1$.

### Conditional coverage (joint test)

$$
LR_{\mathrm{CC}} = LR_{\mathrm{POF}} + LR_{\mathrm{IND}} \xrightarrow{d} \chi^2_2.
$$

A model that passes $LR_{\mathrm{CC}}$ has both the right average breach rate **and** independent breaches — the standard "well-calibrated VaR" criterion.

## Derivation

### From iid Bernoulli to LR statistic

Under correct specification, $\{I_t\}$ is iid Bernoulli($\pi_0$). The likelihood is
$L(\pi) = \pi^x (1 - \pi)^{n - x}$. The MLE is $\hat\pi = x / n$. By Wilks' theorem the deviance

$$
2 \log \frac{L(\hat\pi)}{L(\pi_0)} \xrightarrow{d} \chi^2_1
$$

since the parameter space restriction has codimension 1. Negating recovers $LR_{\mathrm{POF}}$ as written above.

### Independence as a Markov-chain LR

Allow first-order Markov dependence: $I_t \mid I_{t-1} = i \sim \mathrm{Bernoulli}(\pi_{i1})$. The unrestricted MLE has two free probabilities $(\pi_{01}, \pi_{11})$; the restricted MLE collapses them to a single $\hat\pi$. Two minus one = one degree of freedom for $LR_{\mathrm{IND}}$. Independence within higher-order Markov chains generalizes via the same argument; the codimension equals the number of equality constraints.

### Why $LR_{\mathrm{POF}} + LR_{\mathrm{IND}} \sim \chi^2_2$

The two tests target orthogonal aspects of the joint hypothesis:

- $LR_{\mathrm{POF}}$ tests the marginal breach rate.
- $LR_{\mathrm{IND}}$ tests **conditional** rates given the previous violation, holding the marginal at its MLE.

The constraints they impose commute, so under $H_0$ the two statistics are asymptotically independent (Christoffersen 1998, prop. 3). Sum of two independent $\chi^2_1$ is $\chi^2_2$.

### Power and pitfalls

- **Sample size.** For $\alpha = 0.99$ on one year of daily data, expected violations $= 252 \cdot 0.01 \approx 2.5$. The discrete-binomial nature of the test means the achievable type-I error rates are coarse — Kupiec is famously low-power on annual samples. Use 5+ years where possible.
- **Boundary cases.** $\hat\pi = 0$ or $1$ makes $\log \hat\pi$ undefined; the implementation uses the convention $0 \log 0 = 0$ (matching the limit of the binomial likelihood). Same for the row probabilities in the independence test.
- **Independence test failure modes.** If no violations transition into a violation ($N_{11} = 0$), the test still has well-defined statistic via the same boundary convention. If $n_1 = 0$ (no violations at all), independence is vacuous and the LR statistic collapses to zero.

## Discretization

The implementation lives in [stochastech/risk/backtest.py](https://github.com/AmineChr54/StochasTech/blob/main/stochastech/risk/backtest.py): `kupiec_pof(violations, alpha)`, `christoffersen_independence(violations)`, and `conditional_coverage(violations, alpha)`. Each returns a dict with the LR statistic, p-value, and supporting counts. P-values come from `scipy.stats.chi2.sf`.

## Assumptions and failure modes

- Both tests assume iid violations under $H_0$. If the underlying loss distribution itself drifts (regime change), violations will look serially correlated even if the model is "right" within each regime — confounding regime breaks with model misspecification.
- The asymptotic $\chi^2$ approximation is poor for very small expected counts. Use Monte Carlo p-values when $n(1-\alpha) < 10$.
- The tests are blind to the **size** of breaches; a model that violates 5% of the time but every breach is a $10 \sigma$ tail event will pass Kupiec but fail any Expected Shortfall or quadratic-loss backtest. ES backtesting (Acerbi & Szekely 2014) is the next layer up; it is out of scope for v1.

## References

- Kupiec, *Techniques for Verifying the Accuracy of Risk Measurement Models*, Journal of Derivatives, 1995.
- Christoffersen, *Evaluating Interval Forecasts*, International Economic Review, 1998.
- Acerbi & Szekely, *Backtesting Expected Shortfall*, Risk Magazine, 2014.
