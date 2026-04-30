"""Data loader tests. Network mocked to keep tests offline + deterministic."""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from stochastech.data.loaders import load_prices, log_returns


def _fake_yf_module(df: pd.DataFrame) -> SimpleNamespace:
    return SimpleNamespace(download=lambda *a, **kw: df)


def _make_price_df(n: int = 30, start: str = "2024-01-02") -> pd.DataFrame:
    idx = pd.bdate_range(start=start, periods=n)
    rng = np.random.default_rng(0)
    rets = rng.normal(loc=0.0005, scale=0.01, size=n)
    prices = 100.0 * np.exp(np.cumsum(rets))
    return pd.DataFrame({"Close": prices, "Open": prices * 0.999,
                         "High": prices * 1.01, "Low": prices * 0.99,
                         "Volume": rng.integers(1_000_000, 5_000_000, size=n)},
                        index=idx)


def test_load_prices_writes_and_reads_cache(tmp_path) -> None:
    df = _make_price_df(20)
    fake_yf = _fake_yf_module(df)
    with patch.dict(sys.modules, {"yfinance": fake_yf}), \
         patch("stochastech.data.loaders.CACHE_DIR", tmp_path):
        out = load_prices("XYZ", "2024-01-01", "2024-02-01")
        np.testing.assert_allclose(out["Close"].values, df["Close"].values)
        cache_files = list(tmp_path.glob("XYZ_*.csv"))
        assert len(cache_files) == 1
        # Second call should hit the cache — break the network mock to verify.
        with patch.dict(sys.modules, {"yfinance": SimpleNamespace(
                download=lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("network!"))
        )}):
            cached = load_prices("XYZ", "2024-01-01", "2024-02-01")
            np.testing.assert_allclose(cached["Close"].values, df["Close"].values)


def test_load_prices_refresh_flag_bypasses_cache(tmp_path) -> None:
    df1 = _make_price_df(10)
    df2 = _make_price_df(10, start="2024-02-01")
    with patch.dict(sys.modules, {"yfinance": _fake_yf_module(df1)}), \
         patch("stochastech.data.loaders.CACHE_DIR", tmp_path):
        load_prices("AAA", "2024-01-01", "2024-03-01")
    with patch.dict(sys.modules, {"yfinance": _fake_yf_module(df2)}), \
         patch("stochastech.data.loaders.CACHE_DIR", tmp_path):
        out = load_prices("AAA", "2024-01-01", "2024-03-01", refresh=True)
        np.testing.assert_allclose(out["Close"].values, df2["Close"].values)


def test_load_prices_raises_on_empty_response(tmp_path) -> None:
    with (
        patch.dict(sys.modules, {"yfinance": _fake_yf_module(pd.DataFrame())}),
        patch("stochastech.data.loaders.CACHE_DIR", tmp_path),
        pytest.raises(RuntimeError, match="no data"),
    ):
        load_prices("EMPTY", "2024-01-01", "2024-02-01")


def test_load_prices_unwraps_multiindex_columns(tmp_path) -> None:
    df = _make_price_df(15)
    df.columns = pd.MultiIndex.from_tuples([(c, "XYZ") for c in df.columns])
    with patch.dict(sys.modules, {"yfinance": _fake_yf_module(df)}), \
         patch("stochastech.data.loaders.CACHE_DIR", tmp_path):
        out = load_prices("XYZ", "2024-01-01", "2024-02-01")
        assert "Close" in out.columns
        assert not isinstance(out.columns, pd.MultiIndex)


def test_load_prices_raises_when_column_missing(tmp_path) -> None:
    df = _make_price_df(10).drop(columns=["Close"])
    with (
        patch.dict(sys.modules, {"yfinance": _fake_yf_module(df)}),
        patch("stochastech.data.loaders.CACHE_DIR", tmp_path),
        pytest.raises(KeyError),
    ):
        load_prices("XYZ", "2024-01-01", "2024-02-01")


def test_log_returns_basic() -> None:
    s = pd.Series([100.0, 101.0, 99.0, 102.0])
    r = log_returns(s)
    assert len(r) == 3
    assert abs(r.iloc[0] - np.log(101 / 100)) < 1e-12
    assert abs(r.iloc[1] - np.log(99 / 101)) < 1e-12


def test_log_returns_rejects_non_series() -> None:
    with pytest.raises(TypeError):
        log_returns([100.0, 101.0])


def test_log_returns_rejects_empty() -> None:
    with pytest.raises(ValueError):
        log_returns(pd.Series([], dtype=float))


def test_log_returns_rejects_nonpositive() -> None:
    with pytest.raises(ValueError):
        log_returns(pd.Series([100.0, 0.0, 50.0]))
