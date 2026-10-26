"""Constants the manuscript engine reads from the code, without importing es.run_legacy.

es.run_legacy is a heavy module that other tasks edit, so the engine never imports it (an import-time error there
used to stop every macro). SEEDS, ALPHAS and BANDS are mirrored here instead; es.run_legacy stays the source of truth,
tests/es/test_paper_pins.py asserts that the copies equal it, and es.paper_macros checks the seeds in the names of the
legacy output files against SEEDS. PINS are never typed: each is read once from the default argument of the function
that uses it (es.bootstrap, es.policies, es.measures), and a missing default is an error.

This is not a light module. Reading those defaults imports es.bootstrap, es.policies and es.measures (numpy, pandas),
and the registry that imports it also imports es.scores (scikit-learn, xgboost) and the pipeline's own v2 and src
modules. What the engine no longer has is its import-time link to es.run_legacy."""
from __future__ import annotations

import importlib
import inspect
from decimal import Decimal

from .paper_tex import round_half_up

#: Mirror of es.run_legacy.SEEDS, ALPHAS and BANDS (tests/es/test_paper_pins.py: equal to the originals).
SEEDS = (0, 1, 7, 13, 42)
ALPHAS = (0.01, 0.025, 0.05, 0.10)
BANDS = {"0.5_2": (0.5, 2.0), "0.67_1.5": (2 / 3, 1.5), "0.8_1.25": (0.8, 1.25)}


def signature_default(module: str, func: str, param: str):
    """A constant read from the default argument of es.<module>.<func>. Raises ImportError or AttributeError when the
    module or the function is missing, and LookupError when the parameter is missing or has lost its default: a pin is
    never typed here and never silently inspect.Parameter.empty or None."""
    fn = getattr(importlib.import_module(f".{module}", __package__), func)
    parameters = inspect.signature(fn).parameters
    if param not in parameters:
        raise LookupError(f"es.{module}.{func} has no parameter {param!r}")
    default = parameters[param].default
    if default is inspect.Parameter.empty:
        raise LookupError(f"es.{module}.{func}: parameter {param!r} has no default any more, so the pin cannot be read")
    return default


def pct_of(fraction) -> float:
    """A fraction as a percentage, shifted by 100 in Decimal (100 * 0.07 is 7.000000000000001 in binary)."""
    return float(Decimal(repr(float(fraction))).scaleb(2))


def band_text(band) -> str:
    """A failure band as text, each bound to two decimals (half up) without trailing zeros: [0.67, 1.5]."""
    def one(x) -> str:
        s = round_half_up(x, 2)
        return s.rstrip("0").rstrip(".") if "." in s else s
    return f"[{one(band[0])}, {one(band[1])}]"


#: The pins every generator and the implementation table share, read once from the code's own defaults.
PINS = {"bootB": signature_default("bootstrap", "block_contrast_bootstrap", "n_rep"),
        "bootBlockDays": signature_default("bootstrap", "block_contrast_bootstrap", "mean_block"),
        "bootSeed": signature_default("bootstrap", "block_contrast_bootstrap", "seed"),
        "rotMinShiftDays": signature_default("bootstrap", "rotation_test", "min_shift"),
        "pFourWindow": signature_default("policies", "p4_rolling", "window_days"),
        "esFrac": pct_of(signature_default("measures", "expected_shortfall", "frac"))}
