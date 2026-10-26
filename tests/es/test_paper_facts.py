"""Ruling 74 and the hardening round (Ruling 76): the numbers the prose states, as macros. Every macro is tested three
ways: its value comes from its source, a changed source changes the value, and a missing source resolves to pending only
where the ruling says so (bhLevel and famAlpha while es.run_inference is missing, protocolSha, and the facts read from
output files, derived ones included); anywhere else a missing source is an error."""
import dataclasses
import inspect
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SEALED_MACROS = ("sealedDays", "sealedDaysTestB")
ROTATION_MACROS = ("rotGridSingle", "rotGridPooled", "rotPMinSingle", "rotPMinPooled")
LEVEL_MACROS = ("labelPct", "labelPercentile", "ciLevel", "alertAlpha")     # alertAlpha: the alert level, es.config.ALPHA
BH_MACROS = ("bhLevel", "famAlpha")                                           # both from es.run_inference; famAlpha is holm's alpha
POLICY_MACROS = ("pFiveClipLo", "pFiveClipHi", "pFourRateLo", "pFourRateHi", "pFourIters")
SPEC_MACROS = ("pFourWindowMin", "pFourWindowMax")
MONITOR_MACROS = ("earlyDays", "evalStartDay", "psiBins")
STRESS_MACROS = ("stressWindowH", "stressWindowDays")
DUP_MACROS = ("dupRows", "dupBoundaries")
D10_MACROS = ("dTenLabelMissed", "dTenHoldoutExtra", "dTenHoldoutMissed", "dTenHoldoutRows")
CAISO_MACROS = ("caisoFallbackRows",)
PROTOCOL_MACROS = ("protocolSha",)
STRATA_TEXT_MACROS = ("strataPrimaryText", "strataDecileText")
PLACEBO_MACROS = ("placeboFallbackDays",)
LAG_MACROS = ("lagMinH", "lagMaxH", "lookbackH")                  # Ruling 71; read at resolve time since Ruling 76
NEW_MACROS = set(SEALED_MACROS + ROTATION_MACROS + LEVEL_MACROS + BH_MACROS + POLICY_MACROS + SPEC_MACROS
                 + MONITOR_MACROS + STRESS_MACROS + DUP_MACROS + D10_MACROS + CAISO_MACROS + PROTOCOL_MACROS
                 + STRATA_TEXT_MACROS + PLACEBO_MACROS)


def _resolve(name, tmp_path, docs=None):
    from es.paper_macros import Sources, resolve
    from es.paper_registry import entries_by_name
    src = Sources(tmp_path / "out", tmp_path / "paper", docs or tmp_path / "docs_sealed")
    return resolve(entries_by_name()[name], src)


def _value(name, tmp_path, docs=None):
    r = _resolve(name, tmp_path, docs)
    assert r["status"] == "ok", (name, r)
    return r["value"]


def _with_default(monkeypatch, fn, param, value):
    """Replace one default argument of ``fn``: inspect.signature, which the registry reads, sees the new default."""
    names = list(inspect.signature(fn).parameters)
    defaults = list(fn.__defaults__)
    defaults[names.index(param) - (len(names) - len(defaults))] = value
    monkeypatch.setattr(fn, "__defaults__", tuple(defaults))


def _check(tmp_path, group, live, patched):
    """Each macro of the group has a live value and a different value once its source is patched."""
    assert set(live) == set(group) == set(patched)
    assert all(patched[n] != live[n] for n in group)                     # patching the source must move every macro
    assert {n: _value(n, tmp_path) for n in group} == live
    return lambda: {n: _value(n, tmp_path) for n in group}


def _days(window):
    return (date.fromisoformat(window.end) - date.fromisoformat(window.start)).days + 1



def test_the_groups_cover_every_new_macro_and_the_registry_pins_their_formats_stages_and_sources():
    from es.paper_registry import ENTRIES, SPEC_FIXED, entries_by_name
    from es.paper_tex import FORMATS, PAIR_FORMATS
    by = entries_by_name()
    assert NEW_MACROS <= set(by)
    expected = {name: ("int", "const") for name in
                SEALED_MACROS + ("rotGridSingle", "rotGridPooled", "pFourIters", "pFourWindowMin", "pFourWindowMax")
                + MONITOR_MACROS + STRESS_MACROS}
    expected.update({name: ("g3", "const") for name in
                     ("rotPMinSingle", "rotPMinPooled") + LEVEL_MACROS + BH_MACROS
                     + ("pFiveClipLo", "pFiveClipHi", "pFourRateLo", "pFourRateHi")})
    expected.update({name: ("raw", "const") for name in STRATA_TEXT_MACROS})
    expected.update({name: ("int", "dev") for name in DUP_MACROS + D10_MACROS + CAISO_MACROS + PLACEBO_MACROS})
    expected.update({"protocolSha": ("text", "sealedA")})
    assert set(expected) == NEW_MACROS
    for name, (fmt, stage) in expected.items():
        assert (by[name].fmt, by[name].stage) == (fmt, stage), name
        assert by[name].fmt in FORMATS or by[name].fmt in PAIR_FORMATS
    assert {n for n in NEW_MACROS if by[n].source.startswith("code:")} == set(
        SEALED_MACROS + ROTATION_MACROS + LEVEL_MACROS + BH_MACROS + POLICY_MACROS + SPEC_MACROS + MONITOR_MACROS
        + STRESS_MACROS + STRATA_TEXT_MACROS)
    assert by["placeboFallbackDays"].source == "derived:placeboFallbackDays"
    assert set(SPEC_MACROS) <= set(SPEC_FIXED)
    assert len(by) == len(ENTRIES)                                       # names stay unique


# ------------------------------------------------------------------------------------------------ code-derived macros
def test_sealed_window_macros_follow_es_analyses(tmp_path, monkeypatch):
    from es import analyses
    from es.windows import DateWindow
    a, b = analyses.SEALED["SA"], analyses.SEALED["SB"]
    live = {"sealedDays": str(_days(a.deploy)), "sealedDaysTestB": str(_days(b.deploy))}
    patched = {"sealedDays": "30", "sealedDaysTestB": "45"}
    now = _check(tmp_path, SEALED_MACROS, live, patched)
    assert live["sealedDays"] == "92"                                    # July to September 2026, local settlement days
    monkeypatch.setitem(analyses.SEALED, "SA", dataclasses.replace(a, deploy=DateWindow("2026-07-01", "2026-07-30")))
    monkeypatch.setitem(analyses.SEALED, "SB", dataclasses.replace(b, deploy=DateWindow("2026-10-01", "2026-11-14")))
    assert now() == patched


def test_rotation_grid_macros_follow_the_default_shift_set_and_the_sealed_windows(tmp_path, monkeypatch):
    from es import analyses, bootstrap
    from es.bootstrap import RowCube, rotation_test
    from es.paper_tex import FORMATS
    a, b = analyses.SEALED["SA"], analyses.SEALED["SB"]
    d_a, d_b = _days(a.deploy), _days(b.deploy)
    m = inspect.signature(rotation_test).parameters["min_shift"].default

    def expected(d_a, d_b, m):
        cells_a, cells_b = d_a - 2 * m + 2, d_b - 2 * m + 2              # {0} u {m, ..., D - m}: D - 2m + 2 cells
        return {"rotGridSingle": FORMATS["int"](cells_a), "rotGridPooled": FORMATS["int"](cells_a * cells_b),
                "rotPMinSingle": FORMATS["g3"](1 / cells_a), "rotPMinPooled": FORMATS["g3"](1 / (cells_a * cells_b))}

    shifted = 14 if m != 14 else 20                                      # a restricted shift set, whatever the live default
    live = expected(d_a, d_b, m)
    patched = expected(d_a, d_b, shifted)
    now = _check(tmp_path, ROTATION_MACROS, live, patched)
    if m == 1:                                                           # Ruling 69: the full cyclic group, D cells
        assert (live["rotGridSingle"], live["rotGridPooled"]) == (str(d_a), "8{,}464")
        assert (live["rotPMinSingle"], live["rotPMinPooled"]) == ("0.0109", "0.000118")

    def cube(n_days, seed):
        rng = np.random.default_rng(seed)
        day = np.arange(n_days)
        return RowCube.from_arrays(day, n_days, np.zeros(n_days, dtype=int), np.zeros(n_days, dtype=int),
                                   rng.normal(size=n_days), rng.integers(0, 2, n_days))

    single, pooled = rotation_test([cube(d_a, 1)]), rotation_test([cube(d_a, 1), cube(d_b, 2)])
    assert live["rotGridSingle"] == FORMATS["int"](single["n_null"] + 1)      # the macros count rotation_test's own grid
    assert live["rotGridPooled"] == FORMATS["int"](pooled["n_null"] + 1)
    monkeypatch.setattr(bootstrap.rotation_test, "__defaults__", (shifted,))
    assert now() == patched                                              # e.g. 66 and 4,356 cells for a shift of 14
    restricted = rotation_test([cube(d_a, 1)])
    assert now()["rotGridSingle"] == FORMATS["int"](restricted["n_null"] + 1)



def test_label_and_alert_level_macros_come_from_the_defaults(tmp_path, monkeypatch):
    from es import bootstrap, config, scores
    from es.paper_registry import SPEC_FIXED
    live = {"labelPct": "5", "labelPercentile": "95", "ciLevel": "95", "alertAlpha": "0.05"}
    patched = {"labelPct": "10", "labelPercentile": "90", "ciLevel": "90", "alertAlpha": "0.1"}
    now = _check(tmp_path, LEVEL_MACROS, live, patched)
    assert inspect.signature(scores.cached_scores).parameters["label_alpha"].default == 0.05
    assert inspect.signature(bootstrap.percentile_ci).parameters["level"].default == 0.95 and config.ALPHA == 0.05
    assert not set(LEVEL_MACROS) & set(SPEC_FIXED)                       # all four are read from code
    _with_default(monkeypatch, scores.cached_scores, "label_alpha", 0.1)
    _with_default(monkeypatch, bootstrap.percentile_ci, "level", 0.9)
    monkeypatch.setattr(config, "ALPHA", 0.1)
    assert now() == patched



def test_inference_levels_read_run_inference_and_are_pending_while_it_is_missing(tmp_path, monkeypatch):
    """bhLevel is bh's q default and famAlpha, the family-wise level, is holm's alpha default; the alert level is a
    different macro (alertAlpha, es.config.ALPHA), so a change of one never moves the others."""
    from es import config, run_inference
    from es.paper_macros import PENDING
    live = {"bhLevel": "0.05", "famAlpha": "0.05"}
    now = _check(tmp_path, BH_MACROS, live, {"bhLevel": "0.1", "famAlpha": "0.025"})
    assert inspect.signature(run_inference.bh).parameters["q"].default == 0.05
    assert inspect.signature(run_inference.holm).parameters["alpha"].default == 0.05
    with monkeypatch.context() as m:
        _with_default(m, run_inference.bh, "q", 0.1)
        assert now() == {"bhLevel": "0.1", "famAlpha": "0.05"}
    with monkeypatch.context() as m:
        _with_default(m, run_inference.holm, "alpha", 0.025)
        assert now() == {"bhLevel": "0.05", "famAlpha": "0.025"}
    with monkeypatch.context() as m:                                     # the alert level is not the family-wise level
        m.setattr(config, "ALPHA", 0.2)
        assert now() == live and _value("alertAlpha", tmp_path) == "0.2"

    def pending(name, reason):
        r = _resolve(name, tmp_path)
        assert r["status"] == "pending" and r["value"] == PENDING["const"], r      # pending, not an error
        assert reason in r["detail"], r["detail"]

    with monkeypatch.context() as m:                                     # the module is not written yet
        m.setitem(sys.modules, "es.run_inference", None)
        pending("bhLevel", "run_inference")
        pending("famAlpha", "run_inference")
    with monkeypatch.context() as m:                                     # written, without bh (holm still there)
        m.delattr(run_inference, "bh")
        pending("bhLevel", "bh")
        assert _value("famAlpha", tmp_path) == "0.05"
    with monkeypatch.context() as m:                                     # written, without holm
        m.delattr(run_inference, "holm")
        pending("famAlpha", "holm")
        assert _value("bhLevel", tmp_path) == "0.05"
    with monkeypatch.context() as m:                                     # holm without an alpha default
        def holm(pvals, alpha):
            raise NotImplementedError

        m.setattr(run_inference, "holm", holm)
        pending("famAlpha", "default")
    assert now() == live                                                 # and both resolve as soon as they exist


def test_policy_bound_macros_come_from_es_policies(tmp_path, monkeypatch):
    from es import policies
    live = {"pFiveClipLo": "0.001", "pFiveClipHi": "0.5", "pFourRateLo": "0.0001", "pFourRateHi": "0.5",
            "pFourIters": "40"}
    patched = {"pFiveClipLo": "0.01", "pFiveClipHi": "0.4", "pFourRateLo": "0.001", "pFourRateHi": "0.6",
               "pFourIters": "30"}
    now = _check(tmp_path, POLICY_MACROS, live, patched)
    clip = inspect.signature(policies.p5_self_correcting).parameters
    assert (clip["lo"].default, clip["hi"].default) == (0.001, 0.5)       # the sources, as the paper states them
    assert (policies._RATE_LO, policies._RATE_HI) == (1e-4, 0.5)
    assert inspect.signature(policies.p4_matched).parameters["iters"].default == 40
    _with_default(monkeypatch, policies.p5_self_correcting, "lo", 0.01)
    _with_default(monkeypatch, policies.p5_self_correcting, "hi", 0.4)
    monkeypatch.setattr(policies, "_RATE_LO", 0.001)
    monkeypatch.setattr(policies, "_RATE_HI", 0.6)
    _with_default(monkeypatch, policies.p4_matched, "iters", 30)
    assert now() == patched



def test_window_candidates_are_fixed_by_the_spec_and_say_so(tmp_path, monkeypatch):
    """No code constant carries them: v1 compared W = 28, 56 and 84 days. The citation is the exact place in the spec:
    section 12, the first table (round 1), the Minor row, "Correction in v3". 56 is the code's pin."""
    from es.paper_registry import CODE_CONSTANTS, SPEC_FIXED
    live = {"pFourWindowMin": "28", "pFourWindowMax": "84"}
    now = _check(tmp_path, SPEC_MACROS, live, {"pFourWindowMin": "7", "pFourWindowMax": "90"})
    assert CODE_CONSTANTS["pFourWindowMin"] < CODE_CONSTANTS["pFourWindow"] < CODE_CONSTANTS["pFourWindowMax"]
    for name in SPEC_MACROS:
        detail = _resolve(name, tmp_path)["detail"]
        assert detail.startswith("fixed by spec v3 §12, first table"), detail
        assert "Minor row" in detail and "Correction in v3" in detail and "28, 56 and 84" in detail, detail
        assert name in SPEC_FIXED
    monkeypatch.setitem(CODE_CONSTANTS, "pFourWindowMin", 7)             # one typed constant per number, in the registry
    monkeypatch.setitem(CODE_CONSTANTS, "pFourWindowMax", 90)
    assert now() == {"pFourWindowMin": "7", "pFourWindowMax": "90"}


def test_monitor_macros_come_from_run_monitors(tmp_path, monkeypatch):
    from es import run_monitors
    live = {"earlyDays": str(run_monitors.MONITOR_DAYS), "evalStartDay": str(run_monitors.MONITOR_DAYS + 1),
            "psiBins": str(inspect.signature(run_monitors.psi).parameters["bins"].default)}
    now = _check(tmp_path, MONITOR_MACROS, live, {"earlyDays": "21", "evalStartDay": "22", "psiBins": "20"})
    assert (live["earlyDays"], live["evalStartDay"], live["psiBins"]) == ("14", "15", "10")
    monkeypatch.setattr(run_monitors, "MONITOR_DAYS", 21)
    _with_default(monkeypatch, run_monitors.psi, "bins", 20)
    assert now() == {"earlyDays": "21", "evalStartDay": "22", "psiBins": "20"}


def test_stress_window_macros_come_from_the_stress_threshold_column(tmp_path, monkeypatch):
    from es import positions
    live = {"stressWindowH": "168", "stressWindowDays": "7"}
    now = _check(tmp_path, STRESS_MACROS, live, {"stressWindowH": "72", "stressWindowDays": "3"})
    assert "abs_mean_168h" in inspect.getsource(positions.stress_threshold)

    def stress_threshold(train_rows):                                    # the same function over a 72-hour mean
        return train_rows.groupby(["market", "node"])["abs_mean_72h"].quantile(0.9)

    monkeypatch.setattr(positions, "stress_threshold", stress_threshold)
    assert now() == {"stressWindowH": "72", "stressWindowDays": "3"}

    def not_a_pipeline_window(train_rows):                               # 100 h is not a rolling window of the pipeline
        return train_rows.groupby(["market", "node"])["abs_mean_100h"].quantile(0.9)

    monkeypatch.setattr(positions, "stress_threshold", not_a_pipeline_window)
    with pytest.raises(LookupError, match="stressWindowH"):
        _resolve("stressWindowH", tmp_path)

    def two_windows(train_rows):
        return train_rows["abs_mean_24h"] + train_rows["abs_mean_72h"]

    monkeypatch.setattr(positions, "stress_threshold", two_windows)
    with pytest.raises(LookupError, match="stressWindowDays"):
        _resolve("stressWindowDays", tmp_path)



def test_a_missing_code_source_is_an_error_except_where_the_ruling_says_pending(tmp_path, monkeypatch):
    cases = (("sealedDays", "es.analyses"), ("rotGridSingle", "es.bootstrap"), ("labelPct", "es.scores"),
             ("ciLevel", "es.bootstrap"), ("alertAlpha", "es.config"), ("pFiveClipLo", "es.policies"),
             ("pFourRateHi", "es.policies"), ("pFourIters", "es.policies"), ("earlyDays", "es.run_monitors"),
             ("psiBins", "es.run_monitors"), ("stressWindowH", "es.positions"), ("lagMinH", "v2.panel_prep"),
             ("lookbackH", "v2.panel_prep"), ("strataPrimaryText", "es.measures"))
    for name, module in cases:
        with monkeypatch.context() as m:
            m.setitem(sys.modules, module, None)                         # the import fails
            with pytest.raises(LookupError, match=name):
                _resolve(name, tmp_path)


def test_provenance_hashes_cover_derived_and_missing_sources(tmp_path, monkeypatch):
    import hashlib

    import es
    from es.paper_registry import CODE_SOURCES
    es_dir = Path(es.__file__).resolve().parent

    def digest(name):
        return hashlib.sha256((es_dir / f"{name}.py").read_bytes()).hexdigest()

    assert _resolve("sealedDays", tmp_path)["sha"] == digest("analyses")[:16]
    assert _resolve("earlyDays", tmp_path)["sha"] == digest("run_monitors")[:16]
    both = hashlib.sha256((digest("analyses") + digest("bootstrap")).encode()).hexdigest()[:16]
    assert _resolve("rotGridSingle", tmp_path)["sha"] == both            # a number derived from two modules: one digest
    panel_prep = hashlib.sha256((es_dir.parent / "v2" / "panel_prep.py").read_bytes()).hexdigest()
    stress = hashlib.sha256((digest("positions") + panel_prep).encode()).hexdigest()[:16]
    assert _resolve("stressWindowH", tmp_path)["sha"] == stress          # the function's source and the pipeline's windows
    assert _resolve("lookbackH", tmp_path)["sha"] == panel_prep[:16]
    assert _resolve("strataDecileText", tmp_path)["sha"] == digest("measures")[:16]
    assert _resolve("alertAlpha", tmp_path)["sha"] == digest("config")[:16]
    assert _resolve("famAlpha", tmp_path)["sha"] == digest("run_inference")[:16]      # the family-wise level is holm's
    monkeypatch.setitem(CODE_SOURCES, "bhLevel", "no_such_module")      # a source file that is not there: nothing to hash
    assert _resolve("bhLevel", tmp_path)["sha"] == ""


def test_lag_and_lookback_macros_follow_the_pipeline_at_resolve_time(tmp_path, monkeypatch):
    """They were import-time snapshots that equalled the typed numbers, so a re-typed 168 or 216 went unseen."""
    import v2.panel_prep as pipeline
    from es.paper_tex import FORMATS
    live = {"lagMinH": FORMATS["int"](min(pipeline.G1_LAGS)), "lagMaxH": FORMATS["int"](max(pipeline.G1_LAGS)),
            "lookbackH": FORMATS["int"](max(pipeline.G1_ROLL_WINDOWS) + min(pipeline.G1_LAGS))}
    now = _check(tmp_path, LAG_MACROS, live, {"lagMinH": "24", "lagMaxH": "336", "lookbackH": "264"})
    assert (live["lagMinH"], live["lagMaxH"], live["lookbackH"]) == ("48", "168", "216")
    monkeypatch.setattr(pipeline, "G1_LAGS", (24, 72, 336))
    monkeypatch.setattr(pipeline, "G1_ROLL_WINDOWS", (12, 48, 240))
    assert now() == {"lagMinH": "24", "lagMaxH": "336", "lookbackH": "264"}        # 240 + 24


def test_sealed_windows_must_be_inclusive_local_date_windows(tmp_path, monkeypatch):
    """A UtcWindow is half-open: its day count would be off by one, silently."""
    from es import analyses
    from es.windows import UtcWindow
    monkeypatch.setitem(analyses.SEALED, "SA", dataclasses.replace(
        analyses.SEALED["SA"], deploy=UtcWindow("2026-07-01", "2026-10-01")))
    for name in ("sealedDays", "rotGridSingle", "rotGridPooled", "rotPMinSingle", "rotPMinPooled"):
        with pytest.raises(LookupError, match=rf"{name}.*DateWindow"):
            _resolve(name, tmp_path)
    assert _value("sealedDaysTestB", tmp_path) == "92"                   # the other test's window is still a DateWindow
    monkeypatch.setitem(analyses.SEALED, "SB", dataclasses.replace(
        analyses.SEALED["SB"], deploy=UtcWindow("2026-10-01", "2027-01-01")))
    with pytest.raises(LookupError, match=r"sealedDaysTestB.*DateWindow"):
        _resolve("sealedDaysTestB", tmp_path)


def test_strata_text_macros_put_the_strata_columns_into_words(tmp_path, monkeypatch):
    from es import measures
    live = {"strataPrimaryText": "node × local hour × weekday class × month",
            "strataDecileText": "node × decile of the calendar-only model's score"}
    patched = {"strataPrimaryText": "node × month", "strataDecileText": "node × month × local hour"}
    now = _check(tmp_path, STRATA_TEXT_MACROS, live, patched)            # market is dropped: strata are within node
    monkeypatch.setattr(measures, "STRATA_PRIMARY", ["market", "node", "month"])
    monkeypatch.setattr(measures, "STRATA_DECILE", ["market", "node", "month", "settle_hour"])
    assert now() == patched
    monkeypatch.setattr(measures, "STRATA_PRIMARY", ["market", "node", "year"])             # no words for this column
    with pytest.raises(LookupError, match=r"strataPrimaryText.*'year'"):
        _resolve("strataPrimaryText", tmp_path)
    monkeypatch.setattr(measures, "STRATA_DECILE", ["market", "node", "cal_decile", "vol_tercile"])
    with pytest.raises(LookupError, match=r"strataDecileText.*'vol_tercile'"):
        _resolve("strataDecileText", tmp_path)


def test_e4_contrast_stats_are_registered_with_labels_formats_and_twins(tmp_path):
    """Ruling 81: the paired contrasts SELC(P1) - SELC(comparator) of Task 12 and the sealed run."""
    from es.paper_registry import STAT_FMT, STAT_LABEL, STAT_ORDER, STAT_WORD, entries_by_name
    cal, vol = "E4:dSELC|calendar@count", "E4:dSELC|VOL@count"
    assert STAT_WORD[cal] == "dselcCalCount" and STAT_WORD[vol] == "dselcVolCount"
    assert STAT_FMT[cal] == STAT_FMT[vol] == ("pct1", "cipct1")
    # \textminus{}, not U+2212: this class stops with "Unicode character - (U+2212) not set up for use with LaTeX"
    assert STAT_LABEL[cal] == "E4: SELC(P1) \\textminus{} SELC(calendar-only), matched count"
    assert STAT_LABEL[vol] == "E4: SELC(P1) \\textminus{} SELC(VOL), matched count"
    assert STAT_LABEL["E4:SELC|calendar@count"] == "E4: SELC of the calendar-only model, matched count"
    assert STAT_LABEL["E4:SELC|VOL@count"] == "E4: SELC of VOL, matched count"
    assert STAT_ORDER[7:9] == (cal, vol) and STAT_ORDER.index(cal) > STAT_ORDER.index("E4:SELC|VOL@count")       # Ruling 118 appends four ESred rows
    by = entries_by_name()
    for suffix in ("AqThree", "AqFour", "AqOne", "AqTwo", "YY", "HTwo", "PostOpen", "TestA", "TestB"):
        for word in ("dselcCalCount", "dselcVolCount", "selcCalCount", "selcVolCount"):   # the level names stay
            assert f"{word}{suffix}" in by and f"{word}{suffix}CI" in by, (word, suffix)
    assert by["dselcVolCountAqThree"].query == f'analysis == "AQ3" and stat == "{vol}"'
    assert by["dselcCalCountTestA"].source == "sealed/A/tables/estimands.csv" and by["dselcCalCountTestA"].query == f'stat == "{cal}"'
    out = tmp_path / "out"
    (out / "economics").mkdir(parents=True)
    pd.DataFrame([{"analysis": "AQ3", "stat": cal, "estimate": -0.0123, "lo": -0.05, "hi": 0.02, "approximate": False,
                   "n_windows": 1}]).to_csv(out / "economics" / "bootstrap_ci.csv", index=False)
    assert _value("dselcCalCountAqThree", tmp_path) == "\\textminus{}1.2\\%"
    assert _value("dselcCalCountAqThreeCI", tmp_path) == "[\\textminus{}5.0\\%, 2.0\\%]"
    (out / "sealed" / "A" / "tables").mkdir(parents=True)
    pd.DataFrame([{"stat": vol, "estimate": 0.031, "lo": 0.01, "hi": 0.05}]).to_csv(
        out / "sealed" / "A" / "tables" / "estimands.csv", index=False)
    assert _value("dselcVolCountTestA", tmp_path) == "3.1\\%" and _value("dselcVolCountTestACI", tmp_path) == "[1.0\\%, 5.0\\%]"


# ------------------------------------------------------------------------------------------------ facts in output files
def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _implementation(out, rows=390, boundaries=39, nodes=10, extra_note="NYISO panel: both rows of the repeated hour are absent."):
    note = (f"Duplicates: {rows} duplicate rows, all at NYISO month boundaries (each at 00:00 local on the 1st of a month: "
            f"{boundaries} month starts, {nodes} nodes; FORENSICS D11); per-market counts: NYISO {rows}, ISONE 0.")
    _write(out / "paper" / "implementation.json", json.dumps({
        "coverage": [{"series": "NYISO", "nodes": nodes, "duplicates_dropped_total": rows},
                     {"series": "ISONE", "nodes": 9, "duplicates_dropped_total": rows}],
        "coverage_notes": [extra_note, note]}))



def test_duplicate_macros_read_implementation_json(tmp_path):
    from es.paper_macros import PENDING
    out = tmp_path / "out"
    _implementation(out)
    assert [_value(n, tmp_path) for n in DUP_MACROS] == ["390", "39"]
    _implementation(out, rows=410, boundaries=41)                       # a changed source changes the values
    assert [_value(n, tmp_path) for n in DUP_MACROS] == ["410", "41"]
    for name in DUP_MACROS:
        (out / "paper" / "implementation.json").unlink(missing_ok=True)
        r = _resolve(name, tmp_path)
        assert r["status"] == "pending" and r["value"] == PENDING["dev"], r          # a missing output file is pending
        _implementation(out)
    # the pattern is as strict as the sentence: months mentioned elsewhere do not disturb it. A pattern loosened to
    # "(\d+) month" would find "2 month" in "2 months" and "3 month" in "3 month ends", count three, and refuse.
    _implementation(out, extra_note="CAISO: 2 months of the panel precede the FMM start (3 month ends).")
    assert _value("dupBoundaries", tmp_path) == "39"
    # dupBoundaries is read from a sentence of the notes, so it refuses what it cannot read with certainty: a second
    # "N month starts" in another note, or in the same note, is never picked silently
    _implementation(out, extra_note="Also 5 month starts in another note.")
    with pytest.raises(LookupError, match="dupBoundaries.*matched 2"):
        _resolve("dupBoundaries", tmp_path)
    _write(out / "paper" / "implementation.json", json.dumps({
        "coverage": [{"duplicates_dropped_total": 390}], "coverage_notes": ["5 month starts, then 39 month starts."]}))
    with pytest.raises(LookupError, match="dupBoundaries.*matched 2"):
        _resolve("dupBoundaries", tmp_path)
    _write(out / "paper" / "implementation.json", json.dumps({"coverage": [{"duplicates_dropped_total": 390}], "coverage_notes": ["none"]}))
    with pytest.raises(LookupError, match="dupBoundaries.*matched 0"):
        _resolve("dupBoundaries", tmp_path)
    assert _value("dupRows", tmp_path) == "390"                          # the structured total does not depend on the notes


def _d10(out, scale=1):
    rows = []
    for window, extra, missed, inside in ((("2023-01-01", "2024-01-01"), (0, 0, 0, 0), (50, 45, 40, 48), 274177),
                                          (("2026-01-01", "2026-07-01"), (50, 45, 40, 48), (40, 36, 40, 42), 143352)):
        for market, e, m in zip(("NYISO", "ISONE", "MISO", "CAISO"), extra, missed):
            rows.append({"window_start": window[0], "window_end": window[1], "market": market, "rows_utc": inside // 4,
                         "rows_local": 0, "extra_other_dates": e * scale, "missed_local_dates": m * scale, "used_by": "x"})
    pd.DataFrame(rows).to_csv(out / "legacy" / "d10_boundaries.csv", index=False)


def test_d10_macros_sum_the_boundary_rows_of_the_two_windows(tmp_path):
    from es.paper_macros import PENDING
    out = tmp_path / "out"
    (out / "legacy").mkdir(parents=True)
    _d10(out)
    live = {n: _value(n, tmp_path) for n in D10_MACROS}
    assert live == {"dTenLabelMissed": "183", "dTenHoldoutExtra": "183", "dTenHoldoutMissed": "158",
                    "dTenHoldoutRows": "143{,}352"}
    _d10(out, scale=2)                                                   # a changed source changes the values
    assert {n: _value(n, tmp_path) for n in D10_MACROS} == {"dTenLabelMissed": "366", "dTenHoldoutExtra": "366",
                                                            "dTenHoldoutMissed": "316", "dTenHoldoutRows": "143{,}352"}
    (out / "legacy" / "d10_boundaries.csv").unlink()
    for name in D10_MACROS:
        r = _resolve(name, tmp_path)
        assert r["status"] == "pending" and r["value"] == PENDING["dev"]



def test_caisoFallbackRows_is_the_panel_rows_without_an_fmm_hour(tmp_path):
    from es.paper_macros import PENDING
    out = tmp_path / "out"
    other = [{"series": "NYISO", "hourly_rows": 306390},
             {"series": "NYISO integrated RT LBMP (settlement leg)", "hourly_rows": 459615}]

    def coverage(*rows):
        _write(out / "paper" / "coverage.csv", pd.DataFrame(list(rows) + other).to_csv(index=False))

    caiso = lambda n: {"series": "CAISO", "hourly_rows": n}                              # noqa: E731
    fmm = lambda n: {"series": "CAISO FMM (settlement leg)", "hourly_rows": n}           # noqa: E731
    coverage(caiso(169200), fmm(156096))
    assert _value("caisoFallbackRows", tmp_path) == "13{,}104"
    coverage(caiso(170000), fmm(156096))                                 # a changed source changes the value
    assert _value("caisoFallbackRows", tmp_path) == "13{,}904"
    # range is the gap between exactly two rows: a missing or a repeated series is an error, never a silent 0 or a wrong gap
    for rows in ((caiso(169200),), (fmm(156096),), (caiso(169200), fmm(156096), fmm(156096)), ()):
        coverage(*rows)
        with pytest.raises(LookupError, match=r"caisoFallbackRows.*exactly 2"):
            _resolve("caisoFallbackRows", tmp_path)
    (out / "paper" / "coverage.csv").unlink()
    r = _resolve("caisoFallbackRows", tmp_path)
    assert r["status"] == "pending" and r["value"] == PENDING["dev"]


def test_protocolSha_reads_the_protocol_hash_file_and_is_pending_until_it_exists(tmp_path):
    from es.paper_macros import PENDING
    docs = tmp_path / "docs_sealed"
    r = _resolve("protocolSha", tmp_path, docs)
    assert r["status"] == "pending" and r["value"] == PENDING["sealedA"], r      # the file does not exist yet
    digest = "AB" * 32
    _write(docs / "PROTOCOL.sha256", digest + "\n")
    assert _value("protocolSha", tmp_path, docs) == "ab" * 32                    # lower case, as in the registration record
    _write(docs / "PROTOCOL.sha256", ("cd" * 32) + "  PROTOCOL.md\n")           # shasum's own format works too
    assert _value("protocolSha", tmp_path, docs) == "cd" * 32
    for bad in ("", "\n\n", "abc", "zz" * 32, "ab" * 31):                        # an unreadable hash is an error, not pending
        _write(docs / "PROTOCOL.sha256", bad)
        with pytest.raises(LookupError, match="protocolSha"):
            _resolve("protocolSha", tmp_path, docs)


def test_bh_test_counts_include_only_the_rows_bh_ranks_when_the_column_exists(tmp_path):
    """Ruling 80: bh.csv gains a boolean `included` (p is not NaN); the counts then equal BH's m. Ruling 118 (M5): the audit-robustness
    family is descriptive, so it keeps its test count and has no rejection count at all; the economic family stands in for the check."""
    from es.paper_macros import PENDING
    out = tmp_path / "out"
    (out / "inference").mkdir(parents=True)
    rows = [{"family": "audit robustness", "test": "t1", "estimate": 1.0, "p": 0.01, "bh_reject": True, "included": True},
            {"family": "audit robustness", "test": "t2", "estimate": 0.2, "p": 0.40, "bh_reject": False, "included": True},
            {"family": "audit robustness", "test": "t3", "estimate": 0.1, "p": float("nan"), "bh_reject": False, "included": False},
            {"family": "economic robustness", "test": "t6", "estimate": 1.0, "p": 0.01, "bh_reject": True, "included": True},
            {"family": "economic robustness", "test": "t7", "estimate": 0.2, "p": 0.40, "bh_reject": False, "included": True},
            {"family": "economic robustness", "test": "t8", "estimate": 0.1, "p": float("nan"), "bh_reject": False, "included": False},
            {"family": "drift monitors", "test": "t4", "estimate": 0.3, "p": float("nan"), "bh_reject": False, "included": False},
            {"family": "drift monitors", "test": "t5", "estimate": 0.4, "p": 0.03, "bh_reject": True, "included": True}]
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "inference" / "bh.csv", index=False)
    counts = lambda: [_value(n, tmp_path) for n in ("bhTestsAuditRobustness", "bhTestsEconomicRobustness", "bhRejectionsEconomicRobustness",       # noqa: E731
                                                       "bhTestsDriftMonitors", "bhRejectionsDriftMonitors")]
    assert counts() == ["2", "2", "1", "1", "1"]                         # the rows with no p are not tests
    assert "included == True" in _resolve("bhTestsAuditRobustness", tmp_path)["detail"]       # provenance names the filter
    assert _resolve("bhRejectionsEconomicRobustness", tmp_path)["detail"] == ""               # the rejections need none
    frame.drop(columns="included").to_csv(out / "inference" / "bh.csv", index=False)
    assert counts() == ["3", "3", "1", "2", "1"]                         # a file without the column counts every row, as before
    assert _resolve("bhTestsAuditRobustness", tmp_path)["detail"] == ""                       # and says nothing about a filter
    (out / "inference" / "bh.csv").unlink()
    assert _resolve("bhTestsAuditRobustness", tmp_path)["status"] == "pending"
    assert _resolve("bhTestsAuditRobustness", tmp_path)["value"] == PENDING["dev"]


def _coverage_dates(out, caiso_first="2023-04-13", fmm_first="2023-07-13"):
    _write(out / "paper" / "coverage.csv", pd.DataFrame([
        {"series": "CAISO", "first_local_date": caiso_first, "hourly_rows": 169200},
        {"series": "CAISO FMM (settlement leg)", "first_local_date": fmm_first, "hourly_rows": 156096},
        {"series": "NYISO", "first_local_date": "2023-01-01", "hourly_rows": 306390}]).to_csv(index=False))


def test_placebo_fallback_days_count_placebo_pairing_days_before_the_fmm_start(tmp_path, monkeypatch):
    """Spec 6.2(a): the analogue alerts of AQ1-AQ4 are paired with P&L from the same quarter one and two years earlier.
    CAISO P&L falls back to the panel series from the CAISO panel start until the FMM first date."""
    import hashlib

    import es
    from es import analyses
    from es.paper_macros import PENDING, Sources, _shift_back, resolve
    from es.paper_registry import entries_by_name
    from es.windows import DateWindow, UtcWindow
    out = tmp_path / "out"
    _coverage_dates(out)
    r = _resolve("placeboFallbackDays", tmp_path)
    assert r["status"] == "ok" and r["value"] == "12", r                 # 1 to 12 July 2023, the 2023-Q3 window of AQ3
    # provenance: the inputs and what was derived
    es_dir = Path(es.__file__).resolve().parent
    src = Sources(out, tmp_path / "paper", tmp_path / "docs_sealed")
    expected_sha = hashlib.sha256((src.sha("paper/coverage.csv") + hashlib.sha256(
        (es_dir / "analyses.py").read_bytes()).hexdigest()).encode()).hexdigest()[:16]
    assert r["sha"] == expected_sha and r["source"] == "derived:placeboFallbackDays"
    assert "2023-07-13" in r["detail"] and "2023-04-13" in r["detail"] and "AQ1-AQ4" in r["detail"], r["detail"]
    # the same source as fmmFirstDate: moving that date moves the count
    _coverage_dates(out, fmm_first="2023-07-20")
    assert (_value("fmmFirstDate", tmp_path), _value("placeboFallbackDays", tmp_path)) == ("2023-07-20", "19")
    _coverage_dates(out, caiso_first="2023-07-05")                       # no CAISO P&L at all before the panel starts
    assert _value("placeboFallbackDays", tmp_path) == "8"
    _coverage_dates(out)
    # the analogues' windows are the code's: another deploy window, shifted back one and two years, adds its days
    a = analyses.ANALOGS["AQ1"]
    with monkeypatch.context() as m:
        m.setitem(analyses.ANALOGS, "AQ1", dataclasses.replace(a, deploy=DateWindow("2024-05-01", "2024-05-31")))
        m.setitem(analyses.ANALOGS, "AQ4", dataclasses.replace(a, deploy=analyses.ANALOGS["AQ3"].deploy))   # a repeat adds none
        m.setitem(analyses.ANALOGS, "AQ3F", dataclasses.replace(a, deploy=DateWindow("2024-06-01", "2024-06-10")))  # not AQ1-AQ4
        assert _value("placeboFallbackDays", tmp_path) == "43"           # 12 + the 31 days of May 2023, and not the 10 of June
    with monkeypatch.context() as m:                                     # Feb 29 maps to Feb 28 in a year without one
        m.setitem(analyses.ANALOGS, "AQ1", dataclasses.replace(a, deploy=DateWindow("2024-02-29", "2024-03-10")))
        _coverage_dates(out, caiso_first="2023-02-01", fmm_first="2023-03-05")
        assert _value("placeboFallbackDays", tmp_path) == "5"            # 2023-02-28 to 2023-03-04
    _coverage_dates(out)
    assert (_shift_back(date(2024, 2, 29), 1), _shift_back(date(2024, 2, 29), 4)) == (date(2023, 2, 28), date(2020, 2, 29))
    assert _shift_back(date(2024, 3, 1), 2) == date(2022, 3, 1)
    with monkeypatch.context() as m:                                     # a half-open UTC window is refused
        m.setitem(analyses.ANALOGS, "AQ2", dataclasses.replace(analyses.ANALOGS["AQ2"], deploy=UtcWindow("2026-04-01", "2026-07-01")))
        with pytest.raises(LookupError, match=r"placeboFallbackDays.*DateWindow"):
            _resolve("placeboFallbackDays", tmp_path)
    # a missing input file is pending; a coverage file without the FMM row is an error that names the macro
    (out / "paper" / "coverage.csv").unlink()
    r = _resolve("placeboFallbackDays", tmp_path)
    assert r["status"] == "pending" and r["value"] == PENDING["dev"] and "coverage.csv" in r["detail"]
    _write(out / "paper" / "coverage.csv", pd.DataFrame([{"series": "CAISO", "first_local_date": "2023-04-13"}]).to_csv(index=False))
    with pytest.raises(LookupError, match=r"placeboFallbackDays.*fmmFirstDate"):
        resolve(entries_by_name()["placeboFallbackDays"], Sources(out, tmp_path / "paper", tmp_path / "docs_sealed"))


def test_the_real_placebo_fallback_days_are_the_twelve_days_of_july_2023():
    from es.paper_macros import Sources, resolve
    from es.paper_registry import entries_by_name
    if not (REAL_OUT / "paper" / "coverage.csv").exists():
        pytest.skip("Task 3 outputs not generated")
    assert resolve(entries_by_name()["placeboFallbackDays"], Sources())["value"] == "12"


# ------------------------------------------------------------------------------------------------ the real outputs
REAL_OUT = Path(__file__).resolve().parents[2] / "outputs" / "es"


@pytest.mark.skipif(not (REAL_OUT / "paper" / "implementation.json").exists(), reason="Task 3 outputs not generated")
def test_data_fact_macros_agree_with_the_real_outputs_and_their_prose_records():
    from es.paper_macros import Sources, resolve
    from es.paper_registry import entries_by_name
    by, src = entries_by_name(), Sources()
    value = {n: resolve(by[n], src)["value"] for n in DUP_MACROS + D10_MACROS + CAISO_MACROS + ("dupDropped",)}
    assert value["dupRows"] == value["dupDropped"] == "390"              # the structured total and the coverage.csv total
    coverage = pd.read_csv(REAL_OUT / "paper" / "coverage.csv")
    nyiso_nodes = int(coverage.loc[coverage.series == "NYISO", "nodes"].iloc[0])
    assert int(value["dupBoundaries"]) * nyiso_nodes == int(value["dupRows"])    # 39 month starts x 10 nodes = 390 rows
    changes = (REAL_OUT / "legacy" / "CHANGES.md").read_text()               # the D10 sentence of the legacy report
    assert f"= {value['dTenHoldoutMissed']}" in changes and value["dTenHoldoutMissed"] == "158"
    assert value["dTenHoldoutExtra"] == value["dTenLabelMissed"] == "183" and value["dTenHoldoutRows"] == "143{,}352"
    assert "143,352 rows inside the window" in changes
    settlement = (REAL_OUT / "settlement_status.md").read_text()             # the settlement report states the same count
    assert value["caisoFallbackRows"] == "13{,}104" and "CAISO approximate rows: 13,104 of 169,200" in settlement
