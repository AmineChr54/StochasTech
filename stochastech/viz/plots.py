"""Plotting helpers for paper figures. Headless-safe (Agg backend).

All plot functions take prepared data (numpy arrays / dicts) and an output
``Path`` and write a vector PDF. They never read JSON or load tickers --
that's ``scripts/reproduce_figures.py``'s job.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PARAM_LABELS = {
    "kappa": r"$\kappa$",
    "theta": r"$\theta$",
    "xi": r"$\xi$",
    "rho": r"$\rho$",
    "v0": r"$v_0$",
    "mu": r"$\mu$",
}


def plot_loss_curve(history: list[float], out_path: Path, title: str = "") -> None:
    """Single-series loss-vs-iteration plot. Log y-axis."""
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(history, lw=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_param_trajectories(
    windows: list[dict],
    out_path: Path,
    params: tuple[str, ...] = ("kappa", "theta", "xi", "rho", "v0"),
    title: str = "",
) -> None:
    """Per-parameter trajectory across rolling windows.

    ``windows`` is a list of records with ``"end_date"`` (or ``"forecast_date"``)
    and ``"params"`` keys -- the format emitted by ``scripts/run_calibration.py``
    or ``scripts/run_backtest.py``.
    """
    if not windows:
        raise ValueError("no windows to plot")
    date_key = "forecast_date" if "forecast_date" in windows[0] else "end_date"
    dates = [w[date_key] for w in windows]
    fig, axes = plt.subplots(len(params), 1, figsize=(7, 1.4 * len(params)),
                             sharex=True)
    if len(params) == 1:
        axes = [axes]
    for ax, p in zip(axes, params, strict=True):
        values = [w["params"][p] for w in windows]
        ax.plot(dates, values, lw=1.0, marker="o", markersize=2)
        ax.set_ylabel(PARAM_LABELS.get(p, p))
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("window end date")
    # Thin out date ticks if many windows.
    if len(dates) > 12:
        every = max(1, len(dates) // 8)
        for ax in axes:
            ax.set_xticks(range(0, len(dates), every))
            ax.set_xticklabels([dates[i] for i in range(0, len(dates), every)],
                                rotation=30, ha="right")
    if title:
        fig.suptitle(title, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_var_violations(
    forecast_dates: list[str],
    realized_returns: list[float],
    forecasts: dict[str, list[float]],
    violations: dict[str, list[int]],
    out_path: Path,
    title: str = "",
) -> None:
    """Realized loss vs VaR forecasts; mark violation points by method.

    ``forecasts[method]`` is a list of positive VaR forecasts;
    ``violations[method]`` is the corresponding 0/1 indicator series.
    """
    losses = [-r for r in realized_returns]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(forecast_dates, losses, color="black", lw=0.8, label="realized loss",
            alpha=0.7)
    colors = {"gbm_mle": "#1f77b4", "heston": "#d62728", "historical": "#2ca02c"}
    for method, fc in forecasts.items():
        col = colors.get(method)
        ax.plot(forecast_dates, fc, lw=1.0, label=f"VaR ({method})", color=col)
        # Mark violations.
        viol_idx = [i for i, v in enumerate(violations[method]) if v == 1]
        if viol_idx:
            ax.scatter([forecast_dates[i] for i in viol_idx],
                       [losses[i] for i in viol_idx],
                       s=18, color=col, zorder=5, edgecolor="black", linewidth=0.4)
    ax.set_ylabel("loss")
    if len(forecast_dates) > 12:
        every = max(1, len(forecast_dates) // 8)
        ax.set_xticks(range(0, len(forecast_dates), every))
        ax.set_xticklabels(
            [forecast_dates[i] for i in range(0, len(forecast_dates), every)],
            rotation=30, ha="right",
        )
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_coverage_bars(
    summaries: dict[str, dict[str, dict]],
    out_path: Path,
    expected_rate: float,
    title: str = "",
) -> None:
    """Grouped-bar chart of observed violation rate per (ticker, method).

    ``summaries[ticker][method]`` -> dict from ``run_backtest.py`` summary.
    """
    tickers = list(summaries.keys())
    methods = ["gbm_mle", "heston", "historical"]
    width = 0.25
    x = np.arange(len(tickers))
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for j, m in enumerate(methods):
        rates = [summaries[t][m]["observed_rate"] for t in tickers]
        ax.bar(x + (j - 1) * width, rates, width, label=m)
    ax.axhline(expected_rate, color="black", linestyle="--", lw=1.0,
               label=f"nominal {expected_rate:.0%}")
    ax.set_xticks(x)
    ax.set_xticklabels(tickers)
    ax.set_ylabel("observed violation rate")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
