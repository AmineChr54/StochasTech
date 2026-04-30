"""Enforce the binding rule from doc/03_scope.md.

Every module under ``stochastech/sde/`` and ``stochastech/calibration/``
(excluding ``__init__.py``) must:

1. Have a corresponding ``doc/math/*.md`` file (matched by the table below).
2. The math doc must reference the module by its repo-relative path.

Failing this test is how a code change without a math doc gets blocked at CI.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MATH_DIR = REPO_ROOT / "doc" / "math"

MODULE_TO_DOC = {
    "stochastech/sde/base.py": "doc/math/05_euler_maruyama.md",
    "stochastech/sde/gbm.py": "doc/math/03_gbm.md",
    "stochastech/sde/heston.py": "doc/math/04_heston.md",
    "stochastech/calibration/heston_fit.py": "doc/math/07_adjoint_sde.md",
    "stochastech/calibration/losses.py": "doc/math/08_calibration_losses.md",
}

WATCHED_DIRS = ["stochastech/sde", "stochastech/calibration"]


def _watched_modules() -> list[Path]:
    out: list[Path] = []
    for rel in WATCHED_DIRS:
        for path in (REPO_ROOT / rel).glob("*.py"):
            if path.name == "__init__.py":
                continue
            out.append(path.relative_to(REPO_ROOT).as_posix())
    return sorted(out)


def test_every_watched_module_has_mapping() -> None:
    """A new .py under a watched dir without a mapping is a failure."""
    unmapped = [m for m in _watched_modules() if m not in MODULE_TO_DOC]
    assert not unmapped, (
        f"Add a math doc for {unmapped} and register it in MODULE_TO_DOC."
    )


@pytest.mark.parametrize("module,doc", sorted(MODULE_TO_DOC.items()))
def test_math_doc_exists(module: str, doc: str) -> None:
    if not (REPO_ROOT / module).exists():
        pytest.skip(f"{module} not present yet")
    assert (REPO_ROOT / doc).exists(), f"Missing math doc: {doc} for {module}"


@pytest.mark.parametrize("module,doc", sorted(MODULE_TO_DOC.items()))
def test_math_doc_references_module(module: str, doc: str) -> None:
    doc_path = REPO_ROOT / doc
    if not doc_path.exists():
        pytest.skip(f"{doc} not present yet")
    text = doc_path.read_text(encoding="utf-8")
    assert module in text, (
        f"{doc} must reference its code module by path '{module}'."
    )
