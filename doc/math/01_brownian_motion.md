# Brownian motion / Wiener process

> Foundational random process underlying every SDE in this project. No direct code module — this is referenced by every solver.

## Statement

A standard Wiener process $W_t$ on $t \ge 0$ is a stochastic process with:

1. $W_0 = 0$ almost surely.
2. **Independent increments.** For $0 \le s < t$, $W_t - W_s$ is independent of $\mathcal{F}_s$ (the filtration up to time $s$).
3. **Gaussian increments.** $W_t - W_s \sim \mathcal{N}(0,\, t - s)$.
4. **Continuous paths.** $t \mapsto W_t$ is continuous almost surely.

In differential notation: $dW_t \sim \mathcal{N}(0, dt)$, equivalently $dW_t = \sqrt{dt}\,\varepsilon$ with $\varepsilon \sim \mathcal{N}(0, 1)$.

## Derivation

### Random-walk construction (Donsker)

Let $\xi_1, \xi_2, \ldots$ be iid with $\mathbb{E}[\xi_i] = 0$ and $\mathrm{Var}(\xi_i) = 1$. Define the rescaled cumulative sum on $[0, 1]$ by interpolation:

$$
W^{(n)}_t = \frac{1}{\sqrt{n}} \sum_{i=1}^{\lfloor n t \rfloor} \xi_i + \frac{n t - \lfloor n t \rfloor}{\sqrt{n}}\, \xi_{\lfloor n t \rfloor + 1}.
$$

By the **CLT** at any fixed $t$: $W^{(n)}_t \xrightarrow{d} \mathcal{N}(0, t)$. Donsker's invariance principle promotes this to weak convergence of the entire path $W^{(n)} \Rightarrow W$ in $C([0,1])$. The limit $W$ has Gaussian increments, independent on disjoint intervals (by independence of the $\xi_i$), and continuous paths (by tightness).

### Variance and covariance

From property (3), $W_t = (W_t - W_0) \sim \mathcal{N}(0, t)$, so $\mathrm{Var}(W_t) = t$. For $0 \le s \le t$, write $W_t = W_s + (W_t - W_s)$ with the two terms independent:

$$
\mathrm{Cov}(W_s, W_t) = \mathbb{E}[W_s W_t] = \mathbb{E}[W_s^2] + \mathbb{E}[W_s (W_t - W_s)] = s + 0 = s = \min(s, t).
$$

### Quadratic variation

Partition $[0, t]$ into $n$ equal subintervals of length $\Delta t = t/n$. Let $\Delta W_i = W_{t_{i+1}} - W_{t_i} \sim \mathcal{N}(0, \Delta t)$, independent. The quadratic-variation sum is

$$
Q_n = \sum_{i=0}^{n-1} (\Delta W_i)^2.
$$

Each term has mean $\Delta t$ and variance $2(\Delta t)^2$ (variance of $\chi^2_1$ scaled). So $\mathbb{E}[Q_n] = n \Delta t = t$ and $\mathrm{Var}(Q_n) = 2 n (\Delta t)^2 = 2 t \Delta t \to 0$ as $n \to \infty$. By Chebyshev, $Q_n \to t$ in $L^2$, hence in probability:

$$
\langle W \rangle_t \;=\; \lim_{\|\pi\|\to 0} \sum_i (\Delta W_i)^2 \;=\; t \quad \text{(in probability).}
$$

This is the single most important fact about Brownian motion. It says $(dW_t)^2 = dt$ in the box-calculus sense — the second-order term in any Taylor expansion does not vanish, which is exactly what produces Itô's correction (see [02_ito_calculus.md](02_ito_calculus.md)).

Contrast: a smooth path $f \in C^1$ has $\sum (\Delta f)^2 \sim \sum (f' \Delta t)^2 = O(\Delta t) \to 0$. Brownian quadratic variation is $\Theta(1)$ — the path is "too rough."

### Nowhere differentiability

Heuristic: if $W$ were differentiable at $t$, then $\Delta W / \Delta t \to W'(t)$ would be finite. But $\Delta W \sim \mathcal{N}(0, \Delta t)$ has typical magnitude $\sqrt{\Delta t}$, so $\Delta W / \Delta t \sim 1/\sqrt{\Delta t} \to \infty$. The rigorous statement (Paley–Wiener–Zygmund, 1933): $\mathbb{P}(W \text{ is differentiable at any } t) = 0$. A modulus-of-continuity argument shows $W$ is Hölder-$\alpha$ for every $\alpha < 1/2$ but not for $\alpha = 1/2$.

## Discretization

For simulation, sample $\Delta W_i = W_{t_{i+1}} - W_{t_i} \sim \mathcal{N}(0, \Delta t)$ at each step. In code: `dW = torch.randn(N, M) * sqrt(dt)` (M paths, N steps).

## Assumptions and failure modes

- Discretization error scales as $\sqrt{\Delta t}$ for path-level quantities.
- Subtle: pseudo-RNG quality matters for tail estimates — use 64-bit-state RNGs (PyTorch default is fine).

## References

- Shreve, *Stochastic Calculus for Finance II*, ch. 3.
- Øksendal, *SDEs*, ch. 2.
- Lawler, [Stochastic Calculus and Brownian Motion](https://www.math.uchicago.edu/~lawler/finbook.pdf).
- Video: [Brownian motion and Wiener processes explained](https://www.youtube.com/watch?v=ksrcU-foiRU).
