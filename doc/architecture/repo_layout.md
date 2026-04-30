# Repo layout — Phase 1 (Python-first)

> Current source tree. Designed to slot the Phase 2 C++/CUDA library — see [phase2_cpp.md](phase2_cpp.md) — under `cpp/` later without restructuring.

```
StochasTech/
├── pyproject.toml
├── pixi.toml
├── pixi.lock
├── stochastech/
│   ├── __init__.py
│   ├── sde/
│   │   ├── __init__.py
│   │   ├── base.py           # Euler–Maruyama core; see ../doc/math/05_euler_maruyama.md
│   │   ├── gbm.py            # see ../doc/math/03_gbm.md
│   │   └── heston.py         # see ../doc/math/04_heston.md
│   ├── risk/
│   │   ├── __init__.py
│   │   ├── var.py            # see ../doc/math/06_monte_carlo_var.md
│   │   └── backtest.py       # Kupiec POF, Christoffersen
│   ├── calibration/
│   │   ├── __init__.py
│   │   ├── heston_fit.py     # see ../doc/math/07_adjoint_sde.md
│   │   └── losses.py         # see ../doc/math/08_calibration_losses.md
│   ├── data/
│   │   ├── __init__.py
│   │   └── loaders.py        # yfinance wrappers, caching
│   └── viz/
│       ├── __init__.py
│       └── plots.py
├── tests/
│   ├── test_gbm.py
│   ├── test_heston.py
│   ├── test_var.py
│   ├── test_adjoint_gradcheck.py
│   └── test_doc_parity.py    # enforces math-doc parity rule
├── notebooks/
│   ├── 01_gbm_paths.ipynb
│   ├── 02_heston_smile.ipynb
│   └── ...
├── scripts/
│   ├── reproduce_figures.py  # regenerates every paper figure
│   └── run_calibration.py
├── paper/
│   ├── stochastech.tex
│   ├── refs.bib
│   └── figures/
├── doc/                       # documentation (this folder)
└── cpp/                       # placeholder; populated only in Phase 2
```

## Module conventions

- All numerical functions accept and return `torch.Tensor` (CPU or GPU) — no NumPy in the core paths once Week 1 ports complete.
- Random number generation: every public simulator takes a `torch.Generator` argument. No implicit global RNG state.
- Dtype: `float32` default for speed; tests parametrize over `float32` and `float64` to verify scheme convergence rates.
- Shapes: `(n_paths, n_steps + 1)` for path tensors; `(n_paths,)` for terminal samples. Document this in every public function's docstring.

## Doc parity rule

`tests/test_doc_parity.py` walks `stochastech/sde/` and `stochastech/calibration/`, asserts every `.py` (excluding `__init__.py`) has a corresponding `doc/math/*.md` whose link to the module resolves. CI runs this on push. See [../03_scope.md](../03_scope.md).
