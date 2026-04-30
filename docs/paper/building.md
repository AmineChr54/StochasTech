# Building the Paper

> How to compile the LaTeX paper from source.

## Prerequisites

You'll need a LaTeX distribution installed:

=== "macOS"

    ```bash
    brew install --cask mactex
    ```

=== "Linux (Ubuntu/Debian)"

    ```bash
    sudo apt-get install texlive-full
    ```

=== "Windows"

    Install [MiKTeX](https://miktex.org/download) or [TeX Live](https://tug.org/texlive/).

## Compile

Three pdflatex passes are needed for cross-references (one for the bib, two more for forward-references settling):

```bash
cd paper
pdflatex stochastech.tex
bibtex stochastech
pdflatex stochastech.tex
pdflatex stochastech.tex
```

The output is `paper/stochastech.pdf` (gitignored).

## What's in the paper

The paper draft v0 includes:

- **Abstract** — pulled from the [Overview](../getting-started/overview.md)
- **Introduction** — three-bullet contribution list
- **Math model** — equations lifted from the [Heston math doc](../math/heston.md)
- **Method** — full-truncation + energy distance + BPTT-vs-adjoint argument
- **VaR forecasting + backtesting** — methodology from [VaR Backtesting](../math/var-backtesting.md)
- **Experiments** — setup with placeholder Table 1 (fill from `results/backtest_alpha95.json`)
- **Discussion + future work** — jumps, multi-asset, CUDA

## Generating figures first

Make sure to run the pipeline before building the paper:

```bash
pixi run -e dev python scripts/reproduce_figures.py
```

This reads `results/*.json` and writes `paper/figures/*.pdf`. See [Inspecting Results](../guide/results.md) for details on the output format.

!!! tip "Filling Table 1"
    Currently manual — read the JSON, paste the rows into the LaTeX `tabular`. A future iteration could templatize this.
