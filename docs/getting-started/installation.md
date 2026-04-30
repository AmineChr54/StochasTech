# Installation

> One-time setup to get the project running locally.

## Prerequisites

- **Git** — to clone the repository
- **pixi** — the reproducible, conda-forge-backed environment manager

## Install pixi

`pixi` is a reproducible, conda-forge-backed environment manager. The single-source-of-truth `pixi.toml` pins every dependency.

=== "macOS / Linux"

    ```bash
    curl -fsSL https://pixi.sh/install.sh | bash
    ```

=== "Windows (PowerShell)"

    ```powershell
    iwr -useb https://pixi.sh/install.ps1 | iex
    ```

## Clone and install

```bash
git clone https://github.com/AmineChr54/StochasTech.git
cd StochasTech
pixi install
```

This creates the default environment (Python 3.11/3.12, NumPy, SciPy, PyTorch, pandas, matplotlib) and the `dev` environment (adds pytest, pytest-cov, ruff). First install takes ~5 minutes; subsequent calls are instant.

## Verify the install

```bash
pixi run test                         # 120 tests, ~25 seconds
pixi run -e dev lint                  # ruff check, must be clean
```

If these pass, you're ready to run the pipeline. Head to the [Quickstart](quickstart.md) to reproduce all paper results.

## Troubleshooting

!!! warning "`pixi: command not found`"
    The `pixi install` script didn't put the binary on PATH. Restart the shell or add `~/.pixi/bin` to PATH.

!!! warning "`pyarrow.lib.ArrowKeyError: A type extension with name pandas.period already defined`"
    A pyarrow/pandas version conflict on Windows when both system Python and pixi Python see different installs. We sidestepped this by using CSV instead of parquet for the data cache.
