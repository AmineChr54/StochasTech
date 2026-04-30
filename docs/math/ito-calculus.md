# Itô calculus

> The chain rule for stochastic processes. Foundation for every SDE in this project.

## Statement

**Itô's lemma.** Let $X_t$ satisfy the SDE $dX_t = \mu(X_t, t)\,dt + \sigma(X_t, t)\,dW_t$, and let $f(x, t)$ be twice differentiable. Then:

$$
df(X_t, t) = \left( \frac{\partial f}{\partial t} + \mu \frac{\partial f}{\partial x} + \frac{1}{2} \sigma^2 \frac{\partial^2 f}{\partial x^2} \right) dt + \sigma \frac{\partial f}{\partial x}\, dW_t.
$$

The extra $\tfrac{1}{2}\sigma^2 f_{xx}$ term — absent in classical calculus — is what makes stochastic calculus stochastic.

**Itô isometry.** For an adapted process $\phi_t$ in $L^2$:

$$
\mathbb{E}\left[ \left( \int_0^T \phi_t\, dW_t \right)^2 \right] = \mathbb{E}\left[ \int_0^T \phi_t^2\, dt \right].
$$

## Derivation

### Itô's lemma

Fix a partition $0 = t_0 < t_1 < \cdots < t_n = T$ with $\Delta t_i = t_{i+1} - t_i$ and $\Delta X_i = X_{t_{i+1}} - X_{t_i}$. By telescoping,

$$
f(X_T, T) - f(X_0, 0) = \sum_{i=0}^{n-1} \big[ f(X_{t_{i+1}}, t_{i+1}) - f(X_{t_i}, t_i) \big].
$$

Taylor-expand each summand around $(X_{t_i}, t_i)$ to second order:

$$
\Delta f_i = f_t \Delta t_i + f_x \Delta X_i + \tfrac{1}{2} f_{xx} (\Delta X_i)^2 + f_{tx} \Delta t_i \Delta X_i + \tfrac{1}{2} f_{tt} (\Delta t_i)^2 + R_i,
$$

with partial derivatives evaluated at $(X_{t_i}, t_i)$ and $R_i = O((\Delta t_i)^{3/2})$.

Substitute $\Delta X_i = \mu_i \Delta t_i + \sigma_i \Delta W_i$ where $\Delta W_i \sim \mathcal{N}(0, \Delta t_i)$, and expand:

$$
(\Delta X_i)^2 = \mu_i^2 (\Delta t_i)^2 + 2 \mu_i \sigma_i \Delta t_i \Delta W_i + \sigma_i^2 (\Delta W_i)^2.
$$

Now sum and take $\|\pi\| \to 0$. Using the quadratic-variation result from [brownian-motion.md](brownian-motion.md):

- $\sum f_x \Delta X_i \to \int_0^T f_x\, dX_s$ (pathwise + Itô integral).
- $\sum \tfrac{1}{2} f_{xx} \sigma_i^2 (\Delta W_i)^2 \to \int_0^T \tfrac{1}{2} f_{xx} \sigma^2\, ds$ — the **Itô correction**, by quadratic variation.
- $\sum \tfrac{1}{2} f_{xx} \mu_i^2 (\Delta t_i)^2 = O(\Delta t) \to 0$.
- $\sum \mu_i \sigma_i f_{xx} \Delta t_i \Delta W_i \to 0$ (variance $O(\Delta t)$).
- $\sum f_{tx} \Delta t_i \Delta X_i$ and $\sum f_{tt} (\Delta t_i)^2 \to 0$ similarly.
- $\sum R_i \to 0$.

Collecting:

$$
df = \left( f_t + \mu f_x + \tfrac{1}{2} \sigma^2 f_{xx} \right) dt + \sigma f_x\, dW_t.
$$

The single non-vanishing second-order contribution is $\tfrac{1}{2} \sigma^2 f_{xx}\, dt$ — precisely the Itô correction.

### Box calculus

The substitutions above are summarized by the multiplication table:

| · | $dt$ | $dW_t$ |
|---|------|--------|
| $dt$ | 0 | 0 |
| $dW_t$ | 0 | $dt$ |

Read formally: $(dW_t)^2 = dt$, $dt \cdot dW_t = 0$, $(dt)^2 = 0$. These are not algebraic equalities — they are shorthand for the limiting behavior of partition sums.

### Itô isometry

For a simple adapted step process $\phi_t = \sum_j \phi_{t_j} \mathbf{1}_{[t_j, t_{j+1})}(t)$,

$$
\int_0^T \phi_t\, dW_t = \sum_j \phi_{t_j} (W_{t_{j+1}} - W_{t_j}).
$$

Squaring and taking expectations: cross terms vanish by independence of the $W$ increments and adaptedness of $\phi_{t_j}$ (so $\phi_{t_j}$ is independent of $W_{t_{j+1}} - W_{t_j}$), giving

$$
\mathbb{E}\!\left[ \left( \int_0^T \phi_t\, dW_t \right)^2 \right] = \sum_j \mathbb{E}[\phi_{t_j}^2]\, (t_{j+1} - t_j) = \mathbb{E}\!\left[ \int_0^T \phi_t^2\, dt \right].
$$

Density of simple processes in $L^2(\Omega \times [0, T])$ extends the identity to all adapted $\phi \in L^2$.

## Discretization

Itô's lemma is a continuous-time identity; the discrete analogue used in simulation is the Euler–Maruyama update — see [euler-maruyama.md](euler-maruyama.md).

## Assumptions and failure modes

- Requires $f \in C^{2,1}$. For non-smooth payoffs (e.g., American options), generalize via local time / Tanaka's formula.
- Itô vs Stratonovich: this project uses Itô throughout. The two interpretations give different drifts when $\sigma$ depends on $X_t$ — be explicit.

## References

- Shreve, ch. 4.
- Øksendal, ch. 4.
- Video: [Itô's Lemma From Scratch](https://www.youtube.com/watch?v=ZXsqxRRcH6g).
