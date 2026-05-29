"""Equity price loaders backed by Tiingo with on-disk caching.

Tiingo daily prices REST endpoint:
    GET https://api.tiingo.com/tiingo/daily/{ticker}/prices
        ?startDate=YYYY-MM-DD&endDate=YYYY-MM-DD&format=json
    Header: Authorization: Token <API_KEY>

Returned rows contain ``date``, raw OHLCV, and split/dividend-adjusted
``adjOpen/adjHigh/adjLow/adjClose/adjVolume``. We expose ``adjClose`` as the
``Close`` column to keep the calling-code contract identical to the previous
yfinance-backed loader (which used ``auto_adjust=True``).

Docs: https://www.tiingo.com/documentation/end-of-day
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "data_cache" / "tiingo"
TIINGO_BASE_URL = "https://api.tiingo.com/tiingo/daily"

# .env candidates: in priority order. First match wins.
_ENV_PATHS = (
    Path(__file__).resolve().parents[1] / ".env",  # stochastech/.env
    REPO_ROOT / ".env",                            # project root .env
)


def _read_env_file(path: Path) -> dict[str, str]:
    """Tiny .env parser. Skips blanks and ``#`` comments; strips surrounding quotes."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep:
            continue
        value = value.strip().strip('"').strip("'")
        out[key.strip()] = value
    return out


def _get_tiingo_key() -> str:
    """Resolve Tiingo API key from .env files or ``TIINGO_API_KEY`` env var."""
    for env_path in _ENV_PATHS:
        env = _read_env_file(env_path)
        if env.get("TIINGO_API_KEY"):
            return env["TIINGO_API_KEY"]
    key = os.environ.get("TIINGO_API_KEY", "").strip()
    if key:
        return key
    raise RuntimeError(
        "TIINGO_API_KEY not found. Put it in stochastech/.env "
        "(see stochastech/.env.example) or export TIINGO_API_KEY."
    )


def _cache_path(ticker: str, start: str, end: str) -> Path:
    safe = ticker.replace("/", "_").replace("\\", "_").upper()
    return CACHE_DIR / f"{safe}_{start}_{end}.csv"


def _fetch_tiingo(ticker: str, start: str, end: str, api_key: str) -> pd.DataFrame:
    """One Tiingo HTTP call. Returns a DataFrame indexed by date with adj OHLCV."""
    query = urllib.parse.urlencode({
        "startDate": start,
        "endDate": end,
        "format": "json",
    })
    url = f"{TIINGO_BASE_URL}/{urllib.parse.quote(ticker)}/prices?{query}"
    req = urllib.request.Request(
        url,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Token {api_key}",
            "User-Agent": "stochastech/0.1 (+https://github.com/AmineChr54/StochasTech)",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
        raise RuntimeError(
            f"Tiingo HTTP {e.code} for {ticker} {start}..{end}: {body[:200]}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Tiingo network error for {ticker}: {e}") from e

    rows = json.loads(payload)
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"Tiingo returned no rows for {ticker} {start}..{end}")

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)
    df = df.set_index("date").sort_index()
    if "adjClose" not in df.columns:
        raise RuntimeError(
            f"Tiingo payload missing adjClose; got columns {list(df.columns)}"
        )
    # Expose adjClose as Close so the rest of the pipeline (which expects an
    # already-adjusted Close) doesn't need to care about Tiingo's column scheme.
    df["Close"] = df["adjClose"]
    df["Open"] = df.get("adjOpen", df.get("open"))
    df["High"] = df.get("adjHigh", df.get("high"))
    df["Low"] = df.get("adjLow", df.get("low"))
    df["Volume"] = df.get("adjVolume", df.get("volume"))
    return df


def load_prices(
    ticker: str,
    start: str,
    end: str,
    refresh: bool = False,
    column: str = "Close",
) -> pd.DataFrame:
    """Load split/dividend-adjusted prices for ``ticker`` between ``start`` and ``end``.

    Source: Tiingo daily-prices REST API. Caches each ``(ticker, start, end)``
    request under ``data_cache/tiingo/{TICKER}_{start}_{end}.csv``; pass
    ``refresh=True`` to re-download.

    The returned DataFrame is indexed by trading-date and always contains a
    ``Close`` column (mapped from Tiingo's ``adjClose``), so the calling code
    can stay agnostic of the upstream column scheme.
    """
    cache_path = _cache_path(ticker, start, end)
    if cache_path.exists() and not refresh:
        df = pd.read_csv(cache_path, index_col=0, parse_dates=[0])
        if column in df.columns:
            return df
        # Cache file from an older schema — fall through and re-download.

    api_key = _get_tiingo_key()
    df = _fetch_tiingo(ticker, start, end, api_key)
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
