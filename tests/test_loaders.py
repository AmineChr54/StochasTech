"""Data loader tests. Network mocked to keep tests offline + deterministic."""
from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from stochastech.data.loaders import load_prices, log_returns


def _fake_tiingo_rows(n: int = 30, start: str = "2024-01-02") -> list[dict]:
    """Build a Tiingo-shaped response: list of per-day dicts."""
    idx = pd.bdate_range(start=start, periods=n)
    rng = np.random.default_rng(0)
    rets = rng.normal(loc=0.0005, scale=0.01, size=n)
    prices = 100.0 * np.exp(np.cumsum(rets))
    rows = []
    for i, date in enumerate(idx):
        rows.append({
            "date": date.strftime("%Y-%m-%dT00:00:00.000Z"),
            "close": float(prices[i]),
            "high": float(prices[i] * 1.01),
            "low": float(prices[i] * 0.99),
            "open": float(prices[i] * 0.999),
            "volume": int(rng.integers(1_000_000, 5_000_000)),
            "adjClose": float(prices[i]),
            "adjHigh": float(prices[i] * 1.01),
            "adjLow": float(prices[i] * 0.99),
            "adjOpen": float(prices[i] * 0.999),
            "adjVolume": int(rng.integers(1_000_000, 5_000_000)),
            "divCash": 0.0,
            "splitFactor": 1.0,
        })
    return rows


def _fake_urlopen(rows: list[dict]):
    """Return a context manager that mimics ``urllib.request.urlopen``."""
    class _Resp:
        def __enter__(self_inner):
            return self_inner
        def __exit__(self_inner, *a):
            return False
        def read(self_inner):
            return json.dumps(rows).encode("utf-8")
    return lambda *_a, **_kw: _Resp()


def test_load_prices_writes_and_reads_cache(tmp_path) -> None:
    rows = _fake_tiingo_rows(20)
    with patch("stochastech.data.loaders.CACHE_DIR", tmp_path), \
         patch("stochastech.data.loaders._get_tiingo_key", return_value="dummy"), \
         patch("stochastech.data.loaders.urllib.request.urlopen",
               new=_fake_urlopen(rows)):
        out = load_prices("XYZ", "2024-01-01", "2024-02-01")
        expected_close = np.array([r["adjClose"] for r in rows])
        np.testing.assert_allclose(out["Close"].values, expected_close)
        cache_files = list(tmp_path.glob("XYZ_*.csv"))
        assert len(cache_files) == 1
        # Second call must hit the cache — break the mock to verify offline read.
        def _boom(*_a, **_kw):
            raise RuntimeError("network!")
        with patch("stochastech.data.loaders.urllib.request.urlopen", new=_boom):
            cached = load_prices("XYZ", "2024-01-01", "2024-02-01")
            np.testing.assert_allclose(cached["Close"].values, expected_close)


def test_load_prices_refresh_flag_bypasses_cache(tmp_path) -> None:
    rows1 = _fake_tiingo_rows(10)
    rows2 = _fake_tiingo_rows(10, start="2024-02-01")
    with patch("stochastech.data.loaders.CACHE_DIR", tmp_path), \
         patch("stochastech.data.loaders._get_tiingo_key", return_value="dummy"):
        with patch("stochastech.data.loaders.urllib.request.urlopen",
                   new=_fake_urlopen(rows1)):
            load_prices("AAA", "2024-01-01", "2024-03-01")
        with patch("stochastech.data.loaders.urllib.request.urlopen",
                   new=_fake_urlopen(rows2)):
            out = load_prices("AAA", "2024-01-01", "2024-03-01", refresh=True)
            expected = np.array([r["adjClose"] for r in rows2])
            np.testing.assert_allclose(out["Close"].values, expected)


def test_load_prices_raises_on_empty_response(tmp_path) -> None:
    with patch("stochastech.data.loaders.CACHE_DIR", tmp_path), \
         patch("stochastech.data.loaders._get_tiingo_key", return_value="dummy"), \
         patch("stochastech.data.loaders.urllib.request.urlopen",
               new=_fake_urlopen([])), \
         pytest.raises(RuntimeError, match="no rows"):
        load_prices("EMPTY", "2024-01-01", "2024-02-01")


def test_load_prices_raises_on_missing_adjclose(tmp_path) -> None:
    bad_rows = [{"date": "2024-01-02T00:00:00.000Z", "close": 100.0}]
    with patch("stochastech.data.loaders.CACHE_DIR", tmp_path), \
         patch("stochastech.data.loaders._get_tiingo_key", return_value="dummy"), \
         patch("stochastech.data.loaders.urllib.request.urlopen",
               new=_fake_urlopen(bad_rows)), \
         pytest.raises(RuntimeError, match="adjClose"):
        load_prices("BAD", "2024-01-01", "2024-02-01")


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


def test_env_key_resolution(tmp_path, monkeypatch) -> None:
    from stochastech.data import loaders
    fake = tmp_path / ".env"
    fake.write_text('TIINGO_API_KEY = "abc123"\n# comment\n')
    monkeypatch.setattr(loaders, "_ENV_PATHS", (fake,))
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)
    assert loaders._get_tiingo_key() == "abc123"


def test_env_key_falls_back_to_env_var(tmp_path, monkeypatch) -> None:
    from stochastech.data import loaders
    monkeypatch.setattr(loaders, "_ENV_PATHS", (tmp_path / "nope.env",))
    monkeypatch.setenv("TIINGO_API_KEY", "from-env")
    assert loaders._get_tiingo_key() == "from-env"
