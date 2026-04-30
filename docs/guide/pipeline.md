# Pipeline

> Reading order if you want to understand the full data flow.

## Architecture diagram

```mermaid
graph TD
    A["🌐 yfinance (network)"] -->|"load_prices(ticker, start, end)"| B["💾 data_cache/*.csv"]
    B -->|"log_returns(prices)"| C["📊 Daily log-returns"]

    C --> D["🔧 run_calibration.py<br/>(Week 4 driver)"]
    C --> E["📈 run_backtest.py<br/>(Week 5 driver)"]
    C --> F["🧪 grad_check.py<br/>(Week 3)"]

    D --> G["📋 results/calibration_*.json"]
    E --> H["📋 results/backtest_alpha*.json"]

    G --> I["🎨 reproduce_figures.py"]
    H --> I

    I --> J["📄 paper/figures/*.pdf"]
    J --> K["📜 pdflatex → paper.pdf"]

    style A fill:#1b3a4b,stroke:#d4a843,color:#e0e1dd
    style K fill:#1b3a4b,stroke:#16a085,color:#e0e1dd
```

## Step-by-step

Each box in the diagram above is one command. See the [Quickstart](../getting-started/quickstart.md) for the exact commands.

1. **Data loading** — `stochastech/data/loaders.py` pulls equity prices from yfinance and caches as CSV.
2. **Log-returns** — computed from adjusted close prices.
3. **Calibration** — rolling-window Heston fit via BPTT or gradient-free baseline.
4. **Backtesting** — walk-forward 1-day VaR with monthly refit. Three methods compared.
5. **Figures** — reads JSON results and writes vector PDFs.
6. **Paper** — pdflatex compiles the LaTeX source with generated figures.
