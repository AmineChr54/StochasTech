# Phase 2 — C++/CUDA `neuro_ito` library

> **Status: deferred.** Phase 1 ships in PyTorch only. This doc preserves the original engineering spec for when the Python MVP is shipped and we want to accelerate the inner loop.

When we re-engage, the goal is not to "rewrite everything in C++" — it is to write **one CUDA kernel** for the Monte Carlo path-generation inner loop, expose it via nanobind, and call it from `stochastech/sde/heston.py` behind a `--gpu` flag. Benchmark vs PyTorch as a paper-appendix figure.

## Tech stack (Phase 2 only)

- **C++20** — Concepts, smart pointers, modern templates.
- **CUDA C++** (nvcc) — custom kernels for matrix mults and parallel stochastic paths.
- **CMake** — non-negotiable build system.
- **Eigen 3** — header-only CPU linear algebra baseline. Test custom kernels against Eigen's results before trusting them.
- **nanobind** (preferred) or **pybind11** — Python bridge for PyTorch interop.
- **Google Test** — C++ test framework.
- **Clang-Format** + **Clang-Tidy** — automated lint/format.
- **pixi** — same env tool as Phase 1, manages both C++ and Python toolchains.

## Library layout (when populated)

```
neuro_ito/
├── CMakeLists.txt              # master build script
├── README.md                   # 6-second pitch + benchmark graphs
├── .clang-format
├── .gitignore
├── include/                    # PUBLIC HEADERS (the interface)
│   └── neuro_ito/
│       ├── core/               # base tensors, memory pools
│       ├── sde/                # Itô / Stratonovich definitions
│       └── fno/                # Fourier Neural Operator definitions
├── src/                        # CPU C++ implementations
│   ├── core/                   # CPU tensor operations
│   └── sde/                    # Euler–Maruyama and Runge–Kutta CPU solvers
├── cuda/                       # GPU implementations
│   ├── kernels/                # raw .cu files (matmul, path generation)
│   └── bindings/               # safe C++ wrappers around CUDA kernels
├── python/                     # the bridge
│   ├── neuro_ito_py.cpp        # nanobind/pybind11 module
│   └── setup.py
├── tests/                      # GTest suite
│   ├── test_tensor.cpp
│   └── test_sde_convergence.cpp
└── benchmarks/
    └── bench_sde_speed.cpp     # CPU vs GPU latency
```

## Phase 2 scope when re-engaged

1. Path-generation CUDA kernel for parallel Monte Carlo (10k+ simultaneous paths).
2. nanobind binding so Python can call it as a drop-in replacement for the PyTorch path generator.
3. Memory pooling to avoid GPU allocator pressure under millions of short-lived path tensors.
4. Hard benchmark in the paper appendix: "Custom CUDA path generator runs Nx faster than PyTorch on a single GPU."

## Phase 2 explicitly out of scope (still)

- Custom tensor library / template metaprogramming for matmul. PyTorch's tensor implementation is good enough; competing with it is a project of its own.
- AVX-512 / SIMD for CPU paths. Modern PyTorch already vectorizes well.
- Spectral methods / FNO / multi-domain physics. Those belong to the original Aether-Flow vision and are not part of this project. See [`../archive/roadmap.md`](../archive/roadmap.md).

## Triggers to re-engage Phase 2

Only after Phase 1 ships:
- Paper is submitted.
- Phase 1 repo is tagged `v0.1.0`.
- Phase 1 calibration takes more than ~30 minutes per ticker — i.e., the inner loop has become the bottleneck for whatever experiment we want to run next.

If those conditions aren't met, don't start Phase 2.
