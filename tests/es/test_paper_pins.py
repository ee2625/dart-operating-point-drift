import inspect
import subprocess
import sys
from pathlib import Path

import pytest

DEV_ROOT = Path(__file__).resolve().parents[2]


def test_mirrored_constants_equal_run_legacy():
    """es.paper_pins mirrors SEEDS, ALPHAS and BANDS so that the manuscript engine never imports es.run_legacy (a heavy
    module other tasks edit). run_legacy stays the source of truth: this test is the guard against drift."""
    from es import paper_pins, run_legacy
    assert paper_pins.SEEDS == run_legacy.SEEDS
    assert paper_pins.ALPHAS == run_legacy.ALPHAS
    assert paper_pins.BANDS == run_legacy.BANDS


def test_engine_modules_do_not_import_run_legacy():
    code = "import sys, es.paper_registry, es.paper_macros, es.paper_lint; print('es.run_legacy' in sys.modules)"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=DEV_ROOT, timeout=300)
    assert r.returncode == 0, r.stderr[-1500:]
    assert r.stdout.strip() == "False"


def test_signature_default_raises_when_a_default_is_missing_or_removed():
    from es.paper_pins import signature_default
    assert signature_default("bootstrap", "block_contrast_bootstrap", "n_rep") == 2000
    with pytest.raises(LookupError, match="scores_alt"):                 # a parameter without a default
        signature_default("bootstrap", "block_contrast_bootstrap", "scores_alt")
    with pytest.raises(LookupError, match="no_such_param"):
        signature_default("bootstrap", "block_contrast_bootstrap", "no_such_param")
    with pytest.raises(AttributeError):
        signature_default("bootstrap", "no_such_function", "n_rep")


def test_band_text_prints_two_decimals_without_trailing_zeros():
    from es.paper_pins import band_text
    assert band_text((2 / 3, 1.5)) == "[0.67, 1.5]" and band_text((0.5, 2.0)) == "[0.5, 2]"
    assert band_text((0.8, 1.25)) == "[0.8, 1.25]"


def test_registry_constants_come_from_the_pins_module_and_the_code():
    from es.paper_pins import BANDS, PINS, SEEDS, band_text
    from es.paper_registry import CODE_CONSTANTS
    from es.scores import score_analysis
    assert CODE_CONSTANTS["bandMid"] == band_text(BANDS["0.67_1.5"]) == "[0.67, 1.5]"       # read from BANDS, not typed
    assert CODE_CONSTANTS["bandWide"] == band_text(BANDS["0.5_2"]) and CODE_CONSTANTS["bandTight"] == band_text(BANDS["0.8_1.25"])
    assert CODE_CONSTANTS["nSeeds"] == len(SEEDS) and CODE_CONSTANTS["seedList"] == "0, 1, 7, 13, 42"
    assert CODE_CONSTANTS["seedPrimary"] == inspect.signature(score_analysis).parameters["seed"].default == 42
    assert CODE_CONSTANTS["alphaGrid"] == "1, 2.5, 5, 10" and CODE_CONSTANTS["lamGrid"] == "0, 0.25, 0.5, 1"
    assert CODE_CONSTANTS["feeGrid"] == "0.00, 0.10, 0.25, 0.50, 1.00"
    assert all(CODE_CONSTANTS[k] == v for k, v in PINS.items())          # the pins are the code's defaults, once


def test_the_rotation_pin_reads_the_signature_default():
    """Ruling 69 changes rotation_test's default min_shift; the pin follows the code and no test fixes its number."""
    from es.bootstrap import rotation_test
    from es.paper_pins import PINS
    from es.paper_registry import CODE_CONSTANTS
    default = inspect.signature(rotation_test).parameters["min_shift"].default
    assert PINS["rotMinShiftDays"] == CODE_CONSTANTS["rotMinShiftDays"] == default
