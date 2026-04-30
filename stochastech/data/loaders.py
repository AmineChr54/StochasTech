"""Equity price loaders backed by yfinance with on-disk caching."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[2] / "data_cache"


def _cache_path(ticker: str, start: str, end: str) -> Path:
    safe = ticker.replace("/", "_").replace("\\", "_")
    return CACHE_DIR / f"{safe}_{start}_{end}.csv"


def load_prices(
    ticker: str,
    start: str,
    end: str,
    refresh: bool = False,
    column: str = "Close",
) -> pd.DataFrame:
    """Load adjusted close prices for ``ticker`` between ``start`` and ``end``.

    Caches the raw download under ``data_cache/{ticker}_{start}_{end}.csv``;
    pass ``refresh=True`` to re-download. Returns a DataFrame indexed by date
    with at minimum the requested ``column``.

    yfinance is imported lazily so unit tests that don't hit the network — and
    environments where yfinance isn't installed — pay no import cost. CSV cache
    is used over parquet to avoid pyarrow extension-type conflicts on systems
    with multiple pandas installs.
    """
    cache_path = _cache_path(ticker, start, end)
    if cache_path.exists() and not refresh:
        return pd.read_csv(cache_path, index_col=0, parse_dates=[0])

    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker} {start}..{end}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if column not in df.columns:
        raise KeyError(f"Column {column!r} not in download; got {list(df.columns)}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path)
    return df


def log_returns(prices: pd.Series) -> pd.Series:
    """Log returns ``log(S_t / S_{t-1})`` with the first NaN dropped."""
    if not isinstance(prices, pd.Series):
        raise TypeError(f"prices must be pd.Series, got {type(prices).__name__}")
    if prices.empty:
        raise ValueError("prices is empty")
    if (prices <= 0).any():
        raise ValueError("prices must be strictly positive for log-returns")
    return np.log(prices / prices.shift(1)).dropna()
