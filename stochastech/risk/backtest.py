"""VaR backtesting: Kupiec POF and Christoffersen independence + conditional coverage.

See ``docs/math/var-backtesting.md``.
"""
from __future__ import annotations

import math

import torch
from scipy.stats import chi2


def _to_int_array(violations: torch.Tensor) -> torch.Tensor:
    """Cast a 0/1 tensor of violation flags to long, validating values."""
    v = violations.flatten().long()
    if v.numel() == 0:
        raise ValueError("violations tensor is empty")
    if not torch.all((v == 0) | (v == 1)):
        raise ValueError("violations must contain only 0 and 1")
    return v


def kupiec_pof(violations: torch.Tensor, alpha: float) -> dict:
    """Kupiec proportion-of-failures (POF) likelihood-ratio test.

    Tests $H_0: \\pi = 1 - \\alpha$ where $\\pi$ is the violation rate.
    Statistic: $LR_{POF} = -2 \\log \\frac{(1-\\pi_0)^{n - x} \\pi_0^x}{(1-\\hat\\pi)^{n-x} \\hat\\pi^x}$
    with $\\pi_0 = 1 - \\alpha$, $\\hat\\pi = x / n$. Asymptotically $\\chi^2_1$.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    v = _to_int_array(violations)
    n = int(v.numel())
    x = int(v.sum().item())
    p0 = 1.0 - alpha
    pi_hat = x / n if n > 0 else 0.0

    def _log_bin(k: int, m: int, p: float) -> float:
        # log of p^k (1-p)^(m-k), treating 0*log(0) = 0.
        out = 0.0
        if k > 0:
            out += k * math.log(p) if p > 0 else float("-inf")
        if m - k > 0:
            out += (m - k) * math.log(1.0 - p) if p < 1 else float("-inf")
        return out

    log_lik_null = _log_bin(x, n, p0)
    log_lik_alt = _log_bin(x, n, pi_hat) if 0 < pi_hat < 1 else 0.0
    stat = -2.0 * (log_lik_null - log_lik_alt)
    # Numerical guard for the boundary cases where -2(LL_0 - LL_alt) can dip below 0.
    stat = max(stat, 0.0)
    p_value = float(chi2.sf(stat, df=1))
    return {
        "stat": float(stat),
        "pvalue": p_value,
        "n": n,
        "n_violations": x,
        "expected_rate": p0,
        "observed_rate": pi_hat,
        "df": 1,
    }


def christoffersen_independence(violations: torch.Tensor) -> dict:
    """Christoffersen LR test for serial independence of VaR breaches.

    Constructs the 2x2 transition-count matrix $N_{ij}$ for the Markov chain on
    $\\{0, 1\\}$ and tests $H_0: \\pi_{01} = \\pi_{11}$ against the unconstrained
    alternative. Statistic distributed $\\chi^2_1$.
    """
    v = _to_int_array(violations)
    if v.numel() < 2:
        raise ValueError("need at least 2 observations for independence test")
    prev = v[:-1]
    curr = v[1:]
    n00 = int(((prev == 0) & (curr == 0)).sum().item())
    n01 = int(((prev == 0) & (curr == 1)).sum().item())
    n10 = int(((prev == 1) & (curr == 0)).sum().item())
    n11 = int(((prev == 1) & (curr == 1)).sum().item())

    n0 = n00 + n01
    n1 = n10 + n11
    n_total = n0 + n1

    pi01 = n01 / n0 if n0 > 0 else 0.0
    pi11 = n11 / n1 if n1 > 0 else 0.0
    pi_hat = (n01 + n11) / n_total if n_total > 0 else 0.0

    def _xlog(x: float) -> float:
        return 0.0 if x == 0.0 else x * math.log(x)

    # log L under the constrained model (single transition prob).
    log_lik_null = (
        _xlog(1.0 - pi_hat) * (n00 + n10) + _xlog(pi_hat) * (n01 + n11)
    ) if 0 < pi_hat < 1 else 0.0
    # log L under the unconstrained model.
    log_lik_alt_part_0 = (
        _xlog(1.0 - pi01) * n00 + _xlog(pi01) * n01
    ) if 0 < pi01 < 1 else (0.0 if (n01 == 0 and pi01 == 0) or (n00 == 0 and pi01 == 1) else float("-inf"))
    log_lik_alt_part_1 = (
        _xlog(1.0 - pi11) * n10 + _xlog(pi11) * n11
    ) if 0 < pi11 < 1 else (0.0 if (n11 == 0 and pi11 == 0) or (n10 == 0 and pi11 == 1) else float("-inf"))
    log_lik_alt = log_lik_alt_part_0 + log_lik_alt_part_1

    # Re-do with x*log(x) convention which already collapses degenerate terms to 0.
    # The sums above use _xlog so 0 log 0 = 0 is already handled; we only need to use
    # straight x*log(x*total) for the per-pair products.
    def _ll(n_a: int, n_b: int, p: float) -> float:
        # n_a counts under (1-p), n_b counts under p.
        terms = 0.0
        if n_a > 0:
            terms += n_a * math.log(1.0 - p) if p < 1 else float("-inf")
        if n_b > 0:
            terms += n_b * math.log(p) if p > 0 else float("-inf")
        return terms

    log_lik_null = _ll(n00 + n10, n01 + n11, pi_hat)
    log_lik_alt = _ll(n00, n01, pi01) + _ll(n10, n11, pi11)
    stat = -2.0 * (log_lik_null - log_lik_alt)
    stat = max(stat, 0.0) if math.isfinite(stat) else float("nan")
    p_value = float(chi2.sf(stat, df=1)) if math.isfinite(stat) else float("nan")
    return {
        "stat": stat,
        "pvalue": p_value,
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
        "pi_01": pi01,
        "pi_11": pi11,
        "df": 1,
    }


def conditional_coverage(violations: torch.Tensor, alpha: float) -> dict:
    """Christoffersen joint conditional-coverage test (POF + independence).

    $LR_{CC} = LR_{POF} + LR_{IND}$ asymptotically $\\chi^2_2$.
    """
    pof = kupiec_pof(violations, alpha)
    ind = christoffersen_independence(violations)
    stat = pof["stat"] + ind["stat"]
    p_value = float(chi2.sf(stat, df=2)) if math.isfinite(stat) else float("nan")
    return {
        "stat": stat,
        "pvalue": p_value,
        "kupiec": pof,
        "independence": ind,
        "df": 2,
    }
