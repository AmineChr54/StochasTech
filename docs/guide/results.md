# Inspecting Results

> What the pipeline produces and how to read the output.

## Output artifacts

| Artifact | Path | Format |
|----------|------|--------|
| Cached prices | `data_cache/SPY_2018-01-01_2024-12-31.csv` | CSV, indexed by date |
| Calibration parameters per window | `results/calibration_bptt.json` | One record per (ticker, window) |
| Backtest violations + coverage | `results/backtest_alpha95.json` | One record per ticker with violation series |
| Parameter trajectories | `paper/figures/param_trajectories_<TICKER>.pdf` | Multi-row plot of $\kappa, \theta, \xi, \rho, v_0$ over time |
| VaR vs realized | `paper/figures/var_violations_<TICKER>.pdf` | Realized loss curve with three VaR forecast lines + breach markers |
| Coverage bars | `paper/figures/coverage_alpha95.pdf` | Grouped bar of observed-rate per (ticker, method) with 5% nominal line |

## Reading the JSON output

### Calibration results

`results/calibration_bptt.json`:

```json
{
  "config": {"tickers": ["SPY"], "window": 504, ...},
  "results": [
    {
      "ticker": "SPY",
      "windows": [
        {
          "window_idx": 0,
          "start_date": "2018-01-02",
          "end_date": "2019-12-31",
          "loss_final": 1.42e-04,
          "loss_initial": 6.30e-04,
          "n_iters": 200,
          "params": {"mu": 0.05, "kappa": 1.85, "theta": 0.041, ...}
        }
      ]
    }
  ]
}
```

### Backtest results

`results/backtest_alpha95.json` — per ticker:

```json
{
  "ticker": "SPY",
  "summary": {
    "gbm_mle":    {"observed_rate": 0.064, "kupiec_pvalue": 0.18, "conditional_coverage_pvalue": 0.07},
    "heston":     {"observed_rate": 0.052, "kupiec_pvalue": 0.81, "conditional_coverage_pvalue": 0.43},
    "historical": {"observed_rate": 0.058, "kupiec_pvalue": 0.42, "conditional_coverage_pvalue": 0.21}
  },
  "violations": {"gbm_mle": [0,0,1,0,...], ...},
  "forecasts_per_method": {"gbm_mle": [0.018, 0.019, ...], ...},
  "realized_returns": [-0.012, 0.004, ...],
  "forecast_dates": ["2020-01-21", "2020-02-19", ...]
}
```

!!! info "Pass/fail criterion"
    A model "passes" if its `conditional_coverage_pvalue` exceeds 0.05 (the standard 5% significance level): we cannot reject correct coverage.

## Quick interactive inspection

```python
# After running the pipeline:
import json
with open("results/backtest_alpha95.json") as f:
    data = json.load(f)

for r in data["results"]:
    print(r["ticker"])
    for method, s in r["summary"].items():
        print(f"  {method:12s}  obs={s['observed_rate']:.2%}  CC p={s['conditional_coverage_pvalue']:.3f}")
```

## Generating a single figure manually

```python
from pathlib import Path
from stochastech.viz.plots import plot_loss_curve, plot_param_trajectories

plot_loss_curve([1.0, 0.6, 0.4, 0.3, 0.25], Path("/tmp/loss.pdf"), title="My run")
```

All four plot helpers in `stochastech/viz/plots.py` follow the same shape: take prepared data + an output `Path`, write a vector PDF, return None.
