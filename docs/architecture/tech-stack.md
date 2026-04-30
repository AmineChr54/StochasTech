# Tech stack — Phase 1

> Python-first stack for the 6-week sprint. The C++/CUDA stack is preserved in [phase2_cpp.md](phase2-cpp.md) for the deferred neuro_ito library.

## Languages

- **Python 3.11+** — everything in Phase 1.

## Core libraries

| Library | Use | Notes |
|---------|-----|-------|
| **PyTorch** (≥ 2.2) | Tensors, autograd, optimizers | Default device CPU; GPU optional via `--device cuda`. |
| **torchsde** | Differentiable SDE solving with adjoint | The whole point — see [../math/adjoint-sde.md](../math/adjoint-sde.md). |
| **NumPy** | Reference implementations, tests | Used to cross-check PyTorch ports in Week 1; not in production paths. |
| **SciPy** | `scipy.optimize` baseline (Week 4) | Gradient-free Heston fit for sanity check. |
| **yfinance** | Equity price loaders | SPY + 2 single-name tickers. |
| **pandas** | Data wrangling | Returns, rolling windows, persistence. |
| **matplotlib** | Plotting | Paper figures generated headless. |

## Tooling

| Tool | Use |
|------|-----|
| **pixi** | Environment + lockfile (`pixi.lock` committed). |
| **ruff** | Lint + format (replaces black + flake8). |
| **pytest** | Test runner. |
| **pytest-cov** | Coverage on PRs. |
| **GitHub Actions** | CI on push: tests + doc-parity check. |

## Why these choices

- **PyTorch over JAX:** mature `torchsde` is the only mainstream adjoint-SDE library; equivalent JAX option (`diffrax`) exists but adds another framework to learn during a 6-week sprint.
- **pixi over conda/poetry:** language-agnostic envs (will manage C++/CUDA toolchains uniformly when Phase 2 lands). Explicitly suggested in the original tech-stack notes.
- **ruff over black + flake8 + isort:** one tool, fast, single config.
- **No FastAPI / Triton / Docker:** v1 is a paper + reproducible repo, not a serving system. Defer to Phase 3.

## Deferred to Phase 2

C++20, CUDA C++ (nvcc), CMake, Eigen 3, nanobind/pybind11, Google Test, Clang-Format, Clang-Tidy. All preserved in [phase2_cpp.md](phase2-cpp.md).
