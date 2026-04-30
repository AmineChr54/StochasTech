"""Sanity-check that every module imports."""
import importlib

import pytest

MODULES = [
    "stochastech",
    "stochastech.sde",
    "stochastech.sde.base",
    "stochastech.sde.gbm",
    "stochastech.sde.heston",
    "stochastech.risk",
    "stochastech.risk.var",
    "stochastech.risk.backtest",
    "stochastech.calibration",
    "stochastech.calibration.heston_fit",
    "stochastech.calibration.losses",
    "stochastech.data",
    "stochastech.data.loaders",
    "stochastech.viz",
    "stochastech.viz.plots",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    importlib.import_module(name)


def test_version_string() -> None:
    import stochastech

    assert isinstance(stochastech.__version__, str)
    assert stochastech.__version__.count(".") == 2
