# Monte Carlo VaR / ES

> Risk estimators on top of Monte Carlo SDE simulations. Code: [stochastech/risk/var.py](https://github.com/AmineChr54/StochasTech/blob/main/stochastech/risk/var.py).

## Statement

Let $L = -R$ be the loss over horizon $T$, with $R = (S_T - S_0)/S_0$ (or log-return). For a confidence level $\alpha \in (0,1)$:

**Value-at-Risk:**

$$
\mathrm{VaR}_\alpha = \inf\{\ell \in \mathbb{R} : \Pr(L \ge \ell) \le 1 - \alpha\} = -F_R^{-1}(1 - \alpha).
$$

**Expected Shortfall:**

$$
\mathrm{ES}_\alpha = \mathbb{E}[L \mid L \ge \mathrm{VaR}_\alpha].
$$

**Monte Carlo estimators** from $M$ simulated paths:

$$
\widehat{\mathrm{VaR}}_\alpha = -R_{(\lceil (1-\alpha)M \rceil)}, \qquad \widehat{\mathrm{ES}}_\alpha = -\frac{1}{\lceil (1-\alpha)M \rceil} \sum_{i=1}^{\lceil (1-\alpha)M \rceil} R_{(i)},
$$

where $R_{(1)} \le \cdots \le R_{(M)}$ is the order statistic of returns.

## Derivation

### Coherent risk measures

Artzner, Delbaen, Eber, Heath (1999) call a risk measure $\rho$ **coherent** if it is monotone, translation-invariant, positively homogeneous, and **subadditive**: $\rho(L_1 + L_2) \le \rho(L_1) + \rho(L_2)$. Subadditivity is the formal statement that *diversification cannot make risk worse*.

**VaR fails subadditivity.** Two-bond example: each defaults independently with probability $0.04$, loss $1$ on default, $0$ otherwise. At $\alpha = 0.95$:

- $\mathrm{VaR}_{0.95}(L_i) = 0$ (single default has prob $0.04 < 0.05$, so the 95% quantile of $L_i$ is $0$).
- $\Pr(L_1 + L_2 \ge 1) = 1 - 0.96^2 = 0.0784 > 0.05$, so $\mathrm{VaR}_{0.95}(L_1 + L_2) \ge 1$.

Hence $\mathrm{VaR}(L_1 + L_2) > \mathrm{VaR}(L_1) + \mathrm{VaR}(L_2)$. A regulator using VaR could *penalize* a diversified portfolio.

**ES is coherent.** ES averages the entire tail beyond the VaR cut, and the average of a tail is monotone-and-convex in the loss distribution. Formally,

$$
\mathrm{ES}_\alpha(L) = \frac{1}{1-\alpha} \int_\alpha^1 \mathrm{VaR}_u(L)\, du,
$$

and an integral of an Lipschitz-monotone functional preserves subadditivity (proof: Acerbi & Tasche, 2002). This is why Basel III moved from VaR to ES for market-risk capital.

This project still reports VaR alongside ES — VaR is what regulators historically backtest (Kupiec, Christoffersen) and what every previous study has produced numbers for. ES is the better risk measure but the harder one to backtest cleanly (no elicitable single statistic; see Gneiting 2011).

### MC VaR convergence rate

Let $F$ be the return CDF and $q = F^{-1}(1-\alpha)$ the true quantile, so $\mathrm{VaR}_\alpha = -q$. The MC estimator $\hat q_M$ is the $\lceil(1-\alpha)M\rceil$-th order statistic of $M$ iid samples. By Bahadur's representation,

$$
\hat q_M = q + \frac{(1-\alpha) - \hat F_M(q)}{f(q)} + o_p(M^{-1/2}),
$$

where $f = F'$ and $\hat F_M$ is the empirical CDF. Since $\hat F_M(q) - F(q) = O_p(M^{-1/2})$ with variance $\alpha(1-\alpha)/M$, the asymptotic standard error of the VaR estimator is

$$
\mathrm{se}(\widehat{\mathrm{VaR}}_\alpha) \approx \frac{\sqrt{\alpha(1-\alpha)}}{f(q)\,\sqrt{M}}.
$$

The $1/f(q)$ blow-up is the cost of estimating a tail quantile: the further into the tail, the smaller $f(q)$, and the more samples are needed for the same standard error. For $\alpha = 0.99$ on a Gaussian return, $f(q) \approx 0.027 / \sigma$, so the stderr is roughly $3.7 \sigma / \sqrt{M}$ — about $4\times$ worse than the bulk-quantile rate.

### Antithetic variates

Replace iid $Z_1, \ldots, Z_M$ with paired $Z_1, -Z_1, Z_2, -Z_2, \ldots$. For any payoff $g$,

$$
\mathrm{Var}\!\left( \tfrac{1}{2}[g(Z) + g(-Z)] \right) = \tfrac{1}{4}\!\left( \mathrm{Var}\,g(Z) + \mathrm{Var}\,g(-Z) + 2\,\mathrm{Cov}(g(Z), g(-Z)) \right).
$$

Per pair, the variance reduction over the iid case is

$$
1 - \mathrm{corr}\big(g(Z), g(-Z)\big).
$$

For payoffs **monotone in $Z$**, $\mathrm{corr}(g(Z), g(-Z)) < 0$, so antithetic strictly reduces variance — sometimes by an order of magnitude. GBM terminal price $S_T = S_0\, e^{(\mu - \sigma^2/2)T + \sigma\sqrt{T}\, Z}$ is monotone in $Z$, so $\mathbb{E}[S_T]$ benefits. The Heston log-price is monotone in $\tilde W^\perp$ at fixed variance trajectory, so antithetic on the $\perp$ component remains a free win at the price-leg level even though the variance leg breaks monotonicity.

For VaR specifically the gain is smaller because the tail estimator is non-smooth in the inputs; treat antithetic as a default-on baseline rather than the headline variance-reduction trick.

## Discretization

Already discrete — VaR/ES operate on the simulated terminal-return sample. The estimators are `value_at_risk(returns, alpha)` and `expected_shortfall(returns, alpha)` in [stochastech/risk/var.py](https://github.com/AmineChr54/StochasTech/blob/main/stochastech/risk/var.py); paired-mirror antithetic draws are produced by `antithetic_normals(n, n_paths, ...)` in the same module.

## Assumptions and failure modes

- Tail estimation needs $M$ large enough that $(1-\alpha) M \gg 1$. For $\alpha = 0.99$ that means $M \gg 100$. Default to $M \ge 50{,}000$.
- Antithetic variance reduction breaks down for non-monotone-in-$Z$ payoffs (rare for plain VaR/ES on terminal price).
- Backtesting validity: see [02_roadmap.md](../project/roadmap.md) Week 5 for Kupiec POF and Christoffersen tests.

## References

- Artzner, Delbaen, Eber, Heath, *Coherent Measures of Risk*, Mathematical Finance, 1999.
- Glasserman, *Monte Carlo Methods in Financial Engineering*, ch. 9.
- McNeil, Frey, Embrechts, *Quantitative Risk Management*, ch. 2.
