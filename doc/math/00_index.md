# Math docs — index

> One file per concept. Each math doc is paired with a code module of the same conceptual name; the code merges only when its math doc exists. The math docs **double as the source of truth for the paper** — Week 5 paper drafting assembles sections by copying from these files.

## Template

Every math doc has this structure:

```markdown
# <Topic>

> One-line purpose. Link back to the code module: [stochastech/.../foo.py](../../stochastech/.../foo.py).

## Statement
The SDE / theorem / algorithm in formal LaTeX.

## Derivation
Step-by-step. Show the work. Add intuition sidebars where the algebra alone hides what's happening.

## Discretization / numerical scheme
How the continuous math becomes the code. Cite the function and line range.

## Assumptions and failure modes
When does this break? Feller condition, stiffness, identifiability, etc.

## References
Books, papers, page numbers.
```

LaTeX uses GitHub-native `$...$` (inline) and `$$...$$` (display). For multi-line derivations use `aligned`. Define every symbol the first time it appears. If a derivation runs longer than ~2 pages, split it under `derivations/`.

## Files

| # | File | Code module | Sprint week |
|---|------|-------------|-------------|
| 01 | [01_brownian_motion.md](01_brownian_motion.md) | foundation (no direct module) | Week 1 |
| 02 | [02_ito_calculus.md](02_ito_calculus.md) | foundation | Week 1 |
| 03 | [03_gbm.md](03_gbm.md) | `stochastech/sde/gbm.py` | Week 1 |
| 04 | [04_heston.md](04_heston.md) | `stochastech/sde/heston.py` | Week 2 |
| 05 | [05_euler_maruyama.md](05_euler_maruyama.md) | `stochastech/sde/base.py` | Week 1 |
| 06 | [06_monte_carlo_var.md](06_monte_carlo_var.md) | `stochastech/risk/var.py` | Week 2 |
| 07 | [07_adjoint_sde.md](07_adjoint_sde.md) | `stochastech/calibration/heston_fit.py` | Week 3 |
| 08 | [08_calibration_losses.md](08_calibration_losses.md) | `stochastech/calibration/losses.py` | Week 3 |
