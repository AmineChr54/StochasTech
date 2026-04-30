"""Backtest test-suite: Kupiec POF, Christoffersen independence, joint CC."""
from __future__ import annotations

import math

import torch

from stochastech.risk.backtest import (
    christoffersen_independence,
    conditional_coverage,
    kupiec_pof,
)


def _gen(seed: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def test_kupiec_pof_perfect_calibration_passes() -> None:
    # Bernoulli(0.05) with seed-fixed expected count -> p-value should be > 0.05.
    n = 5_000
    alpha = 0.95
    v = (torch.rand(n, generator=_gen(0)) < (1 - alpha)).int()
    out = kupiec_pof(v, alpha=alpha)
    assert out["n"] == n
    assert 0.0 <= out["pvalue"] <= 1.0
    assert out["pvalue"] > 0.05


def test_kupiec_pof_too_many_violations_rejects() -> None:
    # 20% violations with alpha=0.95 -> strong rejection.
    v = (torch.rand(2_000, generator=_gen(1)) < 0.20).int()
    out = kupiec_pof(v, alpha=0.95)
    assert out["pvalue"] < 0.001
    assert out["observed_rate"] > out["expected_rate"]


def test_kupiec_pof_too_few_violations_rejects() -> None:
    # 0.5% violations with alpha=0.95 -> reject with too few.
    v = (torch.rand(2_000, generator=_gen(2)) < 0.005).int()
    out = kupiec_pof(v, alpha=0.95)
    assert out["pvalue"] < 0.001
    assert out["observed_rate"] < out["expected_rate"]


def test_kupiec_pof_zero_violations() -> None:
    # All zeros: pi_hat=0, LL_alt=0, LL_null = n log(1-p0). Statistic is finite.
    v = torch.zeros(500, dtype=torch.int64)
    out = kupiec_pof(v, alpha=0.95)
    assert math.isfinite(out["stat"])
    assert out["pvalue"] < 1e-3  # 0/500 vs expected 25 strongly rejects.


def test_kupiec_pof_rejects_bad_inputs() -> None:
    try:
        kupiec_pof(torch.zeros(10), alpha=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on alpha=0")
    try:
        kupiec_pof(torch.empty(0), alpha=0.95)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on empty tensor")
    try:
        kupiec_pof(torch.tensor([0, 1, 2]), alpha=0.95)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on non-binary input")


def test_christoffersen_independence_iid_passes() -> None:
    # iid Bernoulli(0.05) -> independence p-value should not reject at 5%.
    v = (torch.rand(5_000, generator=_gen(3)) < 0.05).int()
    out = christoffersen_independence(v)
    assert math.isfinite(out["stat"])
    assert out["pvalue"] > 0.05


def test_christoffersen_independence_clustered_rejects() -> None:
    # Build an obviously clustered violation series: 50 leading violations + 950 zeros.
    v = torch.cat([torch.ones(50, dtype=torch.long), torch.zeros(950, dtype=torch.long)])
    out = christoffersen_independence(v)
    # Strongly non-iid -> independence should reject.
    assert out["pvalue"] < 0.05


def test_christoffersen_independence_handles_zero_violations() -> None:
    v = torch.zeros(200, dtype=torch.long)
    out = christoffersen_independence(v)
    # No transitions into 1 -> stat should be 0 / NaN; either is acceptable so long
    # as we don't crash.
    assert math.isnan(out["stat"]) or out["stat"] == 0.0


def test_conditional_coverage_combines_pof_and_independence() -> None:
    v = (torch.rand(3_000, generator=_gen(4)) < 0.05).int()
    cc = conditional_coverage(v, alpha=0.95)
    assert cc["df"] == 2
    assert math.isclose(cc["stat"], cc["kupiec"]["stat"] + cc["independence"]["stat"],
                        rel_tol=1e-9, abs_tol=1e-9)
    assert 0.0 <= cc["pvalue"] <= 1.0


def test_conditional_coverage_rejects_biased_iid() -> None:
    # Right marginal (5%) but wrong clustering vs right clustering wrong marginal:
    # extreme case — half the series is 0/1/0/1/... at exactly 50% violations.
    pattern = torch.tensor([0, 1] * 1_000, dtype=torch.long)
    cc = conditional_coverage(pattern, alpha=0.95)
    # Both POF (50% vs 5%) and IND (perfect alternation) should reject hard.
    assert cc["pvalue"] < 1e-6
