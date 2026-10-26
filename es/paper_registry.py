"""Every macro the manuscript may use: name, source, selector, format and stage.

Names are letters only (LaTeX). Analysis ids, blocks, estimators and metrics map to words through
the tables below, and nowhere else. Sources are paths relative to outputs/es/ (CSV or JSON), or
"code:<key>" for a constant read from the package, "author:<key>" for paper/author_inputs.json (the identifiers only the author can
supply; Ruling 133: the protocol is not registered with a third party and the code is not archived at a DOI, so no macro names an OSF or
Zenodo identifier or reads a registration record), "docsfile:<name>" for a file under docs/sealed/ that holds
one value (a hash), "derived:<name>" for a number that es.paper_macros computes from output files and the code's own
definitions when it resolves the macro (DERIVED there). The registry is data; es.paper_macros resolves it and writes
paper/generated/registry.csv so reviewers can read it as a table.

A code constant is either a snapshot in CODE_CONSTANTS (taken when this module is imported) or, for the numbers the
prose states (Ruling 74), a reader in CODE_READERS that es.paper_macros calls when it resolves the macro, so that the
value follows its source and a source that is not there yet (es.run_inference while it is being written) resolves to
the stage sentinel. A number that no code carries is typed once, in CODE_CONSTANTS, and named in SPEC_FIXED.

SEEDS, ALPHAS, BANDS and PINS come from es.paper_pins, so that the engine never imports es.run_legacy.
CODE_SOURCES names the module behind each code constant; its SHA-256 goes into provenance.csv.
Entry.nullable marks the values a JSON null or a CSV NaN may take (the Holm decisions and the replication verdicts of Test B, a power
that failed), which print "n/a".
Entry.agg picks the value out of the rows a query selects: one (exactly one row), count, sum, median, min, max, range (the
gap between exactly two rows), same (every selected row has this value), argmax (the row where the column `by` is largest,
the first on a tie, and a missing `by` is an error: the worst case of the registration rule, Ruling 101). A format of
es.paper_tex.ROW_FORMATS takes all the columns of Entry.column ("law,scenario,persistence") as one text. On a JSON source agg
"count" is the number of entries of the object or list at the pointer. Entry.estimand_of names a hypothesis (H-B1): when the
run's family (results.json["family"], the ids of its confirmatory hypotheses) does not list it, the macro prints "n/a (estimand)".

Where the controller fills \\pullRecordTestA (Task 9, Ruling 121): the sentence on the pull's own log (chunks accepted incomplete and days
excused, or none) is in no output file, so it is the one macro of Test A that a person writes. It is paper/author_inputs.json, key
"pull_record_test_a": plain ASCII, one line, ending with a full stop (es.paper_macros.VALIDATORS refuses anything else). While the value is
empty the macro prints the sentinel \\PENDINGA, and the strict build of stage sealedA fails on it (_pull_record_entries)."""
from __future__ import annotations

import importlib
import importlib.metadata as md
import inspect
import platform
import re
from dataclasses import dataclass, replace
from datetime import date
from decimal import Decimal

from v2.panel_prep import G1_LAGS, G1_ROLL_WINDOWS

from .analyses import ANALOGS, LEGACY, SEALED
from .blocks import BLOCKS
from .config import ALPHA, SETTLE_TZ
from .measures import E1_KEYS, LAMS, STRATA_DECILE, STRATA_PRIMARY
from .paper_pins import ALPHAS, BANDS, PINS, SEEDS, band_text, pct_of, signature_default  # noqa: F401 (PINS is re-exported)
from .paper_tex import fmt_dec, fmt_g, tex_escape
from .positions import CALIBRATED_FEES, UNIFORM_FEES
from .scores import MIN_POSITIVES

ANALYSIS_WORD = {"24H1-25H1": "HOne", "24H2-25H2": "HTwo", "24H2-25H2-train2023": "HTwoOldFit",
                 "2024-2025": "YY", "25H1-26H1": "PostOpen", "sealed26H1": "SealedHold",
                 "frozen-26H1": "Frozen", "AQ3": "AqThree", "AQ4": "AqFour", "AQ1": "AqOne",
                 "AQ2": "AqTwo", "AQ3F": "AqThreeFrozen", "SA": "TestA", "SB": "TestB",
                 "LC-full-q2023": "LcFullQold", "LC-full-q2324": "LcFullQnew",
                 "LC-h1-q2023": "LcHalfQold", "LC-h1-q2324": "LcHalfQnew"}
BLOCK_WORD = {"G6": "Cal", "LEV6": "Lev", "MID3": "Mid", "DART3": "Dart", "MIDDART6": "MidDart",
              "ROLL15": "Roll", "ROLL_DISP9": "Disp", "ROLL_LOC6": "Loc", "DART18": "Rel", "G1_24": "Comb"}
EST_WORD = {"xgb": "", "logit": "Logit"}
METRIC_WORD = {"failures": "fail", "median_R": "medR", "median_log_error": "medLogErr",
               "median_auprc": "auprc", "auprc_lift": "lift", "prevalence": "prev", "n_zero_activity": "zeroAct"}
METRIC_FMT = {"failures": "int", "median_R": "dec2", "median_log_error": "dec2", "median_auprc": "dec2",
              "auprc_lift": "dec2", "prevalence": "dec3", "n_zero_activity": "int"}
MARKET_WORD = {"NYISO": "Nyiso", "ISONE": "Isone", "MISO": "Miso", "CAISO": "Caiso"}
LEGACY_IDS = tuple(LEGACY) + ("LC-full-q2023", "LC-full-q2324", "LC-h1-q2023", "LC-h1-q2324")
#: The four quarterly analogues (AQ3F, the frozen-origin arm, is not a quarter of its own): what "on the quarters" means.
QUARTER_ANALOGUES = ("AQ1", "AQ2", "AQ3", "AQ4")
HB_WORD = {"H-B1": "hbOne", "H-B2": "hbTwo"}
LAM_WORD = {0.0: "Zero", 0.25: "Quarter", 0.5: "Half", 1.0: "One"}
#: E4 (Ruling 81): the levels SELC of the calendar-only model and of VOL at the matched alert count, and the paired
#: contrasts SELC(P1) - SELC(comparator) that the table reads (the differences have their own bootstrap interval).
#: Ruling 118 (G1, M6) adds the same two for the tail measure: each comparator's own ESred (E4:ESred|c) and the paired
#: contrast dESred|c = ESred(P1) - ESred(c), the tail protection of the alerts beyond a rule that alerts as often. E3's ESred has no
#: matched-activity benchmark of its own, so the text reads it as the tail effect of the policy and E4 as the evidence about skill.
STAT_WORD = {"SELC": "selc", "NET": "net", "E2:SNET": "snet", "E3:ESred": "esred",
             "E4:SELC_decile": "selcDecile", "E4:SELC|calendar@count": "selcCalCount",
             "E4:SELC|VOL@count": "selcVolCount", "E4:dSELC|calendar@count": "dselcCalCount",
             "E4:dSELC|VOL@count": "dselcVolCount",
             "E4:ESred|calendar@count": "esredCalCount", "E4:ESred|VOL@count": "esredVolCount",
             "E4:dESred|calendar@count": "desredCalCount", "E4:dESred|VOL@count": "desredVolCount"}
STAT_FMT = {"SELC": ("pct1", "cipct1"), "NET": ("usd0", "ciusd0"), "E2:SNET": ("usd2", "ciusd2"),
            "E3:ESred": ("usd0", "ciusd0"), "E4:SELC_decile": ("pct1", "cipct1"),
            "E4:SELC|calendar@count": ("pct1", "cipct1"), "E4:SELC|VOL@count": ("pct1", "cipct1"),
            "E4:dSELC|calendar@count": ("pct1", "cipct1"), "E4:dSELC|VOL@count": ("pct1", "cipct1"),
            "E4:ESred|calendar@count": ("usd0", "ciusd0"), "E4:ESred|VOL@count": ("usd0", "ciusd0"),
            "E4:dESred|calendar@count": ("usd0", "ciusd0"), "E4:dESred|VOL@count": ("usd0", "ciusd0")}
#: The one stat-to-label table (Tasks 4 and 6 import STAT_FMT, STAT_LABEL and STAT_ORDER from here). The labels are LaTeX:
#: the minus of a difference is \textminus{}, because the class cannot typeset the Unicode minus sign U+2212 (it stops with
#: "Unicode character - (U+2212) not set up for use with LaTeX").
STAT_LABEL = {"SELC": "SELC", "NET": "NET", "E2:SNET": "E2: SNET (\\$/MW-day)", "E3:ESred": "E3: ESred",
              "E4:SELC_decile": "E4: SELC, decile strata",
              "E4:SELC|calendar@count": "E4: SELC of the calendar-only model, matched count",
              "E4:SELC|VOL@count": "E4: SELC of VOL, matched count",
              "E4:dSELC|calendar@count": "E4: SELC(P1) \\textminus{} SELC(calendar-only), matched count",
              "E4:dSELC|VOL@count": "E4: SELC(P1) \\textminus{} SELC(VOL), matched count",
              "E4:ESred|calendar@count": "E4: ESred of the calendar-only model, matched count",
              "E4:ESred|VOL@count": "E4: ESred of VOL, matched count",
              "E4:dESred|calendar@count": "E4: ESred(P1) \\textminus{} ESred(calendar-only), matched count",
              "E4:dESred|VOL@count": "E4: ESred(P1) \\textminus{} ESred(VOL), matched count"}
STAT_ORDER = tuple(STAT_LABEL)
#: The four statistics of Ruling 118 (the comparators' own ESred and the paired contrasts dESred|c): the ones the text cites for
#: the pooled rehearsal as well, and the ones the Fig. 2 right panel draws.
E4_ESRED_STATS = tuple(s for s in STAT_ORDER if s.startswith(("E4:ESred", "E4:dESred")))
E1_FMT = {"NET": ("usd0", "ciusd0"), "ESred": ("usd0", "ciusd0")}
E1_FMT.update({f"J_{lam}": ("usd0", "ciusd0") for lam in LAMS})


@dataclass(frozen=True)
class Entry:
    name: str
    source: str
    query: str
    column: str
    fmt: str
    stage: str
    agg: str = "one"
    note: str = ""
    nullable: bool = False      # a JSON null prints "n/a" (the Holm decisions of Test B) instead of being an error
    pattern: str = ""           # JSON text: the regex whose one group is the value, searched in the strings at `query`
    when_column: str = ""       # CSV: when the file has this column, `when_query` is applied as well (and-ed with `query`);
    when_query: str = ""        # a file written before the column existed is read as it always was
    by: str = ""                # agg "argmax": the column whose largest value picks the row (the first one on a tie)
    estimand_of: str = ""       # JSON: the hypothesis whose estimand status replaces the value by "n/a (estimand)" (Ruling 107)


def _ver(pkg: str) -> str:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return "not installed"


#: version macro -> installed distribution (the macro prints md.version of it).
PACKAGES = {"verPandas": "pandas", "verNumpy": "numpy", "verXgboost": "xgboost", "verSklearn": "scikit-learn",
            "verPyarrow": "pyarrow", "verScipy": "scipy", "verStatsmodels": "statsmodels", "verMatplotlib": "matplotlib"}

CODE_CONSTANTS: dict[str, object] = {
    "alphaPct": pct_of(ALPHA), "nBlocks": len(BLOCKS), "minPositives": MIN_POSITIVES,
    "nSeeds": len(SEEDS), "seedPrimary": signature_default("scores", "score_analysis", "seed"),
    "seedList": ", ".join(str(s) for s in SEEDS),
    "alphaGrid": ", ".join(fmt_g(pct_of(a), 6) for a in ALPHAS),
    "bandWide": band_text(BANDS["0.5_2"]), "bandMid": band_text(BANDS["0.67_1.5"]),
    "bandTight": band_text(BANDS["0.8_1.25"]),
    "lamGrid": ", ".join(fmt_g(lam, 6) for lam in LAMS), "nEoneKeys": len(E1_KEYS),
    "strataPrimary": " $\\times$ ".join(tex_escape(s) for s in STRATA_PRIMARY),
    "strataDecile": " $\\times$ ".join(tex_escape(s) for s in STRATA_DECILE),
    "feeGrid": ", ".join(fmt_dec(c, 2) for c in UNIFORM_FEES),
    **{f"fee{MARKET_WORD[m]}{'Inc' if d == 1 else 'Dec'}": CALIBRATED_FEES[m][d]
       for m in CALIBRATED_FEES for d in (1, -1)},
    "tzMiso": SETTLE_TZ["MISO"], "posMW": 1,
    "lagMinH": min(G1_LAGS), "lagMaxH": max(G1_LAGS),
    "lookbackH": max(G1_ROLL_WINDOWS) + min(G1_LAGS),      # rolling windows end lagMinH before the row (impl_table asserts it)
    **PINS, "pFourWindowAlt": 28, "pFourWindowMin": 28, "pFourWindowMax": 84, "covMinNodes": 28, "covMinShare": 90,
    "nLegacyAnalyses": len(LEGACY), "nAnalogs": len(ANALOGS), "nSealed": len(SEALED),
    "verPython": platform.python_version(), **{key: _ver(pkg) for key, pkg in PACKAGES.items()},
}

#: The module that defines each code constant (es/<name>.py, or v2.<name> for the pipeline's own), "pkg:<distribution>" or
#: "python". es.paper_macros records the SHA-256 of that file (or of the package METADATA) in provenance.csv.
#: "paper_registry" means typed in this file (see SPEC_FIXED).
#: SEEDS, ALPHAS and BANDS are mirrored in es.paper_pins, but run_legacy is their source, so run_legacy is what is hashed.
_SOURCE_MODULE = {
    "alphaPct": "config", "tzMiso": "config", "nBlocks": "blocks", "minPositives": "scores", "seedPrimary": "scores",
    "nSeeds": "run_legacy", "seedList": "run_legacy", "alphaGrid": "run_legacy", "bandWide": "run_legacy",
    "bandMid": "run_legacy", "bandTight": "run_legacy", "lamGrid": "measures", "nEoneKeys": "measures",
    "strataPrimary": "measures", "strataDecile": "measures", "bootB": "bootstrap", "bootBlockDays": "bootstrap",
    "bootSeed": "bootstrap", "rotMinShiftDays": "bootstrap", "pFourWindow": "policies", "esFrac": "measures",
    "nLegacyAnalyses": "analyses", "nAnalogs": "analyses", "nSealed": "analyses", "verPython": "python",
    "lagMinH": "v2.panel_prep", "lagMaxH": "v2.panel_prep", "lookbackH": "v2.panel_prep"}
#: The code constants that no code carries (grep es/, v2/ and src/): spec v3 fixes them and they are typed above, once.
#: The provenance row says "fixed by spec v3 §..." for them; tests/es/test_impl_table.py compares impl_table's twins.
_P4_WINDOWS_SPEC = ('spec v3 §12, first table (round 1), Minor row, "Correction in v3": W = 56 was chosen in v1 development, '
                    'which compared 28, 56 and 84 days on pre-2026 data, and is kept fixed')
SPEC_FIXED = {"posMW": "spec v3 §4.2 (standing 1 MW virtual positions)",
              "pFourWindowAlt": "spec v3 §4.3 (the 28-day P4 sensitivity window, in days)",
              "pFourWindowMin": f"{_P4_WINDOWS_SPEC}; the shortest candidate, in days",
              "pFourWindowMax": f"{_P4_WINDOWS_SPEC}; the longest candidate, in days",
              "covMinNodes": "spec v3 §6.1 (coverage rule: usable nodes)",
              "covMinShare": "spec v3 §6.1 (node exclusion: share of expected deployment hours, in percent)"}
CODE_SOURCES: dict[str, str] = {
    key: ("positions" if key.startswith("fee") else f"pkg:{PACKAGES[key]}" if key in PACKAGES
          else _SOURCE_MODULE.get(key, "paper_registry")) for key in CODE_CONSTANTS}


class CodeUnavailable(LookupError):
    """A code constant whose source is not there yet (a module another task is still writing): the macro resolves to its
    stage sentinel, with this message as the detail, instead of being an error."""


def _module(name: str):
    return importlib.import_module(f".{name}", __package__)


def _pipeline():
    """The pipeline's feature module, imported now: the lags and rolling windows are read when a macro is resolved."""
    return importlib.import_module("v2.panel_prep")


def date_window(window, what: str):
    """``window`` if it is an es.windows.DateWindow (inclusive local settlement dates), else a LookupError that names
    ``what``. A UtcWindow is half-open, so counting its days as end - start + 1 would be off by one, silently."""
    windows = _module("windows")
    if not isinstance(window, windows.DateWindow):
        raise LookupError(f"{what} is a {type(window).__name__}, expected a DateWindow (inclusive local days): "
                          f"its days cannot be counted as end - start + 1")
    return window


def _window_days(test_id: str) -> int:
    """Days of a sealed test's deployment window: inclusive local settlement dates (es.analyses.SEALED)."""
    window = date_window(_module("analyses").SEALED[test_id].deploy, f"es.analyses.SEALED[{test_id!r}].deploy")
    return (date.fromisoformat(window.end) - date.fromisoformat(window.start)).days + 1


def _rotation_cells(days: int) -> int:
    """Cells of one window's rotation grid, identity included: the shift set {0} u {m, ..., D - m} of the default
    min_shift m of es.bootstrap.rotation_test has D - 2m + 2 elements (D for the full cyclic group, m = 1: Ruling 69).
    tests/es/test_paper_facts.py checks the formula against rotation_test's own n_null."""
    m = signature_default("bootstrap", "rotation_test", "min_shift")
    if days < 2 * m:
        raise LookupError(f"a window of {days} days is shorter than 2 * min_shift = {2 * m}: rotation_test refuses it")
    return days - 2 * m + 2


def _stress_window_hours() -> int:
    """The trailing window of the stress threshold, in hours. es.positions carries it only in the name of the column that
    stress_threshold reads (abs_mean_<hours>h), so it is read from that function's source; it must be one of the
    pipeline's rolling windows (v2.panel_prep.G1_ROLL_WINDOWS)."""
    hours = {int(h) for h in re.findall(r"abs_mean_(\d+)h", inspect.getsource(_module("positions").stress_threshold))}
    if len(hours) != 1:
        raise LookupError(f"es.positions.stress_threshold names the windows {sorted(hours)}, expected exactly one")
    (window,) = hours
    rolling = _pipeline().G1_ROLL_WINDOWS
    if window not in rolling:
        raise LookupError(f"the stress window of {window} h is not a rolling window of the pipeline {rolling}")
    return window


#: The strata columns in words (es.measures.STRATA_PRIMARY and STRATA_DECILE). "market" is dropped: a node belongs to one
#: market, so the strata are within a node. A column without words here is an error, never a raw column name in the text.
STRATA_WORDS = {"node": "node", "settle_hour": "local hour", "wclass": "weekday class", "month": "month",
                "cal_decile": "decile of the calendar-only model's score"}
STRATA_DROPPED = ("market",)


def _strata_text(attribute: str) -> str:
    columns = getattr(_module("measures"), attribute)
    unknown = [c for c in columns if c not in STRATA_WORDS and c not in STRATA_DROPPED]
    if unknown:
        raise LookupError(f"es.measures.{attribute} names the column {unknown[0]!r}, which has no words in STRATA_WORDS "
                          f"(es/paper_registry.py; known: {sorted(STRATA_WORDS)})")
    return " \u00d7 ".join(STRATA_WORDS[c] for c in columns if c not in STRATA_DROPPED)


def _optional(read, what: str):
    """A reader whose source may not exist yet: any failure becomes CodeUnavailable (the macro is pending)."""
    def reader():
        try:
            return read()
        except Exception as exc:                  # absent, half written, renamed: another task's work in progress
            raise CodeUnavailable(f"{what} cannot be read yet ({type(exc).__name__}: {exc})") from exc
    return reader


#: Code constants read when the macro is resolved (see the module docstring); READER_SOURCES names the module(s) hashed.
CODE_READERS = {
    "sealedDays": lambda: _window_days("SA"),                       # Test A's deployment window, local settlement days
    "sealedDaysTestB": lambda: _window_days("SB"),
    "rotGridSingle": lambda: _rotation_cells(_window_days("SA")),   # cells of one window's rotation grid, identity included
    "rotGridPooled": lambda: _rotation_cells(_window_days("SA")) * _rotation_cells(_window_days("SB")),
    "rotPMinSingle": lambda: 1 / _rotation_cells(_window_days("SA")),
    "rotPMinPooled": lambda: 1 / (_rotation_cells(_window_days("SA")) * _rotation_cells(_window_days("SB"))),
    "labelPct": lambda: pct_of(signature_default("scores", "cached_scores", "label_alpha")),
    "labelPercentile": lambda: float(Decimal(100) - Decimal(repr(pct_of(
        signature_default("scores", "cached_scores", "label_alpha"))))),
    "ciLevel": lambda: pct_of(signature_default("bootstrap", "percentile_ci", "level")),
    "alertAlpha": lambda: _module("config").ALPHA,                  # the alert level (the label tail), not a family-wise level
    "famAlpha": _optional(lambda: signature_default("run_inference", "holm", "alpha"),
                          "es.run_inference.holm (its alpha default, the family-wise level)"),
    "bhLevel": _optional(lambda: signature_default("run_inference", "bh", "q"), "es.run_inference.bh (its q default)"),
    "pFiveClipLo": lambda: signature_default("policies", "p5_self_correcting", "lo"),
    "pFiveClipHi": lambda: signature_default("policies", "p5_self_correcting", "hi"),
    "pFourRateLo": lambda: _module("policies")._RATE_LO,            # the bisection bounds of P4@r*
    "pFourRateHi": lambda: _module("policies")._RATE_HI,
    "pFourIters": lambda: signature_default("policies", "p4_matched", "iters"),
    "earlyDays": lambda: _module("run_monitors").MONITOR_DAYS,
    "evalStartDay": lambda: _module("run_monitors").MONITOR_DAYS + 1,
    "psiBins": lambda: signature_default("run_monitors", "psi", "bins"),
    "stressWindowH": _stress_window_hours,
    "stressWindowDays": lambda: _stress_window_hours() / 24,
    "lagMinH": lambda: min(_pipeline().G1_LAGS),                    # the snapshots in CODE_CONSTANTS stay for impl_table
    "lagMaxH": lambda: max(_pipeline().G1_LAGS),
    "lookbackH": lambda: max(_pipeline().G1_ROLL_WINDOWS) + min(_pipeline().G1_LAGS),   # windows end lagMinH before the row
    "strataPrimaryText": lambda: _strata_text("STRATA_PRIMARY"),
    "strataDecileText": lambda: _strata_text("STRATA_DECILE"),
    # the trailing windows of Figure 2 (plan 2C, T1b): the length of the window and of its first, shorter windows, in weeks. es.run_driftpath is
    # imported when the macro is resolved, not here (it imports es.paper_macros, which imports this module)
    "driftWindowWeeks": lambda: _module("run_driftpath").WINDOW_WEEKS,
    "driftWindowMinWeeks": lambda: _module("run_driftpath").WINDOW_MIN_WEEKS,
}
READER_SOURCES = {
    "sealedDays": "analyses", "sealedDaysTestB": "analyses", "rotGridSingle": "analyses+bootstrap",
    "rotGridPooled": "analyses+bootstrap", "rotPMinSingle": "analyses+bootstrap", "rotPMinPooled": "analyses+bootstrap",
    "labelPct": "scores", "labelPercentile": "scores", "ciLevel": "bootstrap", "alertAlpha": "config",
    "famAlpha": "run_inference", "bhLevel": "run_inference", "pFiveClipLo": "policies", "pFiveClipHi": "policies", "pFourRateLo": "policies",
    "pFourRateHi": "policies", "pFourIters": "policies", "earlyDays": "run_monitors", "evalStartDay": "run_monitors",
    "psiBins": "run_monitors", "stressWindowH": "positions+v2.panel_prep", "stressWindowDays": "positions+v2.panel_prep",
    "strataPrimaryText": "measures", "strataDecileText": "measures",
    "driftWindowWeeks": "run_driftpath", "driftWindowMinWeeks": "run_driftpath"}
CODE_SOURCES.update(READER_SOURCES)
CONST_FMT = {"alphaPct": "g3", "feeGrid": "raw", "seedList": "raw", "alphaGrid": "raw", "bandWide": "raw",
             "bandMid": "raw", "bandTight": "raw", "lamGrid": "raw", "strataPrimary": "raw",
             "strataDecile": "raw", "strataPrimaryText": "raw", "strataDecileText": "raw", "tzMiso": "text",
             "bootSeed": "raw", "esFrac": "g3",
             "pFourWindowMin": "int", "pFourWindowMax": "int",
             **{k: "int" for k in ("sealedDays", "sealedDaysTestB", "rotGridSingle", "rotGridPooled", "pFourIters",
                                   "earlyDays", "evalStartDay", "psiBins", "stressWindowH", "stressWindowDays",
                                   "driftWindowWeeks", "driftWindowMinWeeks")},
             **{k: "g3" for k in ("rotPMinSingle", "rotPMinPooled", "labelPct", "labelPercentile", "ciLevel", "alertAlpha",
                                  "famAlpha", "bhLevel", "pFiveClipLo", "pFiveClipHi", "pFourRateLo", "pFourRateHi")}}


def _const_entries() -> list[Entry]:
    out = []
    for key in list(CODE_CONSTANTS) + [k for k in CODE_READERS if k not in CODE_CONSTANTS]:
        value = CODE_CONSTANTS.get(key)
        fmt = CONST_FMT.get(key) or ("dec2" if (isinstance(value, float) and key.startswith("fee"))
                                     else "int" if isinstance(value, int) else "text")
        out.append(Entry(key, f"code:{key}", "", "", fmt, "const"))
    return out


def _legacy_entries() -> list[Entry]:
    out = []
    for aid in LEGACY_IDS:
        aw = ANALYSIS_WORD[aid]
        for blk, bw in BLOCK_WORD.items():
            for est, ew in EST_WORD.items():
                for metric, mw in METRIC_WORD.items():
                    q = (f'analysis == "{aid}" and block == "{blk}" and estimator == "{est}" '
                         f'and metric == "{metric}"')
                    out.append(Entry(f"{mw}{bw}{aw}{ew}", "legacy/regenerated_long.csv", q,
                                     "regenerated", METRIC_FMT[metric], "dev"))
                for market, mk in MARKET_WORD.items():
                    q = (f'analysis == "{aid}" and block == "{blk}" and estimator == "{est}" '
                         f'and metric == "failures_{market}"')
                    out.append(Entry(f"fail{bw}{aw}{ew}{mk}", "legacy/regenerated_long.csv", q,
                                     "regenerated", "int", "dev"))
    # alpha and band sweeps (D8), seeds (D8), derived numbers (Fig. 2, "49 of 53", sign tests)
    for aid in ("2024-2025", "25H1-26H1"):
        aw = ANALYSIS_WORD[aid]
        for blk in ("LEV6", "DART18", "G1_24", "DART3"):
            bw = BLOCK_WORD[blk]
            for a, word in ((0.01, "One"), (0.025, "TwoHalf"), (0.1, "Ten")):
                out.append(Entry(f"fail{bw}{aw}Alpha{word}", "legacy/regenerated_long.csv",
                                 f'analysis == "{aid}" and block == "{blk}" and estimator == "xgb" and metric == "failures@a{a:g}"',
                                 "regenerated", "int", "dev"))
            for metric, word in (("failures@band1.5", "BandMid"), ("failures@band1.25", "BandTight")):
                out.append(Entry(f"fail{bw}{aw}{word}", "legacy/regenerated_long.csv",
                                 f'analysis == "{aid}" and block == "{blk}" and estimator == "xgb" and metric == "{metric}"',
                                 "regenerated", "int", "dev"))
        for blk in ("G6", "LEV6", "DART3", "ROLL_DISP9", "DART18"):
            bw = BLOCK_WORD[blk]
            for metric, word in (("failures_seed_min", "SeedMin"), ("failures_seed_max", "SeedMax")):
                out.append(Entry(f"fail{bw}{aw}{word}", "legacy/regenerated_long.csv",
                                 f'analysis == "{aid}" and block == "{blk}" and estimator == "xgb" and metric == "{metric}"',
                                 "regenerated", "int", "dev"))
            if blk != "G6":
                for metric, word, fmt in (("delta_auprc_median", "DeltaAuprc", "dec3"),
                                          ("delta_auprc_npos", "DeltaAuprcNpos", "int"),
                                          ("delta_auprc_n", "DeltaAuprcN", "int")):
                    out.append(Entry(f"{word[0].lower() + word[1:]}{bw}{aw}", "legacy/regenerated_long.csv",
                                     f'analysis == "{aid}" and block == "{blk}" and estimator == "xgb" and metric == "{metric}"',
                                     "regenerated", fmt, "dev"))
        for market, mk in MARKET_WORD.items():
            out.append(Entry(f"opContrast{aw}{mk}", "legacy/regenerated_long.csv",
                             f'analysis == "{aid}" and block == "DART18-LEV6" and estimator == "xgb" and metric == "opcontrast_{market}"',
                             "regenerated", "dec2", "dev"))
    for aid in ("24H1-25H1", "24H2-25H2", "2024-2025", "25H1-26H1"):
        aw = ANALYSIS_WORD[aid]
        out += [Entry(f"signK{aw}", "legacy/regenerated_long.csv",
                      f'analysis == "{aid}" and block == "LEV6" and estimator == "xgb" and metric == "signtest_k"',
                      "regenerated", "int", "dev"),
                Entry(f"signP{aw}", "legacy/regenerated_long.csv",
                      f'analysis == "{aid}" and block == "LEV6" and estimator == "xgb" and metric == "signtest_p"',
                      "regenerated", "pval", "dev")]
    out += [
        Entry("failCombYYInsample", "legacy/regenerated_long.csv",
              'analysis == "2024-2025" and block == "G1_24" and estimator == "xgb" and metric == "failures@P2_all_train"',
              "regenerated", "int", "dev"),
        Entry("medRCombYYInsample", "legacy/regenerated_long.csv",
              'analysis == "2024-2025" and block == "G1_24" and estimator == "xgb" and metric == "median_R@P2_all_train"',
              "regenerated", "dec2", "dev"),
        Entry("nodesDownYY", "legacy/regenerated_long.csv",
              'analysis == "2024-2025" and block == "G1_24" and estimator == "xgb" and metric == "nodes_down"',
              "regenerated", "int", "dev"),
        Entry("medRRatioYY", "legacy/regenerated_long.csv",
              'analysis == "2024-2025" and block == "G1_24" and estimator == "xgb" and metric == "median_R_ratio"',
              "regenerated", "dec1", "dev"),
        Entry("nFailingNodeBlocks", "legacy/regenerated_long.csv",
              'analysis == "agg:2pairs" and metric == "n_failing_nodeblocks"', "regenerated", "int", "dev"),
        Entry("nFailingAbovePrev", "legacy/regenerated_long.csv",
              'analysis == "agg:2pairs" and metric == "n_failing_above_prev"', "regenerated", "int", "dev"),
        Entry("nZeroActivityAll", "legacy/regenerated_long.csv",
              'analysis == "agg:paper" and metric == "n_zero_activity"', "regenerated", "int", "dev"),
        Entry("nNodes", "legacy/regenerated_long.csv", 'analysis == "panel" and block == "all" and metric == "n_nodes"',
              "regenerated", "int", "dev"),
        Entry("spearmanDriftTwentySix", "legacy/regenerated_long.csv",
              'analysis == "drift2026" and metric == "spearman_2026"', "regenerated", "dec2", "dev"),
        Entry("pNinetyFiveDartYYpooled", "legacy/regenerated_long.csv",
              'analysis == "fig1" and block == "2024" and metric == "p95_abs_dart_pooled"', "regenerated", "dec1", "dev"),
        Entry("pNinetyFiveDartTwentySixPooled", "legacy/regenerated_long.csv",
              'analysis == "fig1" and block == "2026H1" and metric == "p95_abs_dart_pooled"', "regenerated", "dec1", "dev"),
        Entry("sourceActivityMin", "legacy/d5_d9.json", "source_activity_min", "", "pct1", "dev"),
        Entry("sourceActivityMax", "legacy/d5_d9.json", "source_activity_max", "", "pct1", "dev"),
        Entry("nPublishedNumbers", "legacy/comparison.csv", "", "match", "int", "dev", agg="count"),
        Entry("nPublishedMismatch", "legacy/comparison.csv", "match == False", "match", "int", "dev", agg="count"),
    ]
    for market, mk in MARKET_WORD.items():
        out += [Entry(f"rows{mk}", "legacy/regenerated_long.csv",
                      f'analysis == "panel" and block == "{market}" and metric == "rows"', "regenerated", "int", "dev"),
                Entry(f"nodes{mk}", "legacy/regenerated_long.csv",
                      f'analysis == "panel" and block == "{market}" and metric == "n_nodes"', "regenerated", "int", "dev"),
                Entry(f"prevTwentyFour{mk}", "legacy/regenerated_long.csv",
                      f'analysis == "panel-q2023" and block == "{market}" and metric == "prevalence"', "regenerated", "dec3", "dev")]
    return out


def _analog_entries() -> list[Entry]:
    out = []
    for aid in ANALOGS:
        aw = ANALYSIS_WORD[aid]
        for blk, bw in BLOCK_WORD.items():
            for pol, pw in (("P1", ""), ("P4_56", "PFour"), ("P4_28", "PFourShort"), ("P6", "PSix")):
                base = f'analysis == "{aid}" and block == "{blk}" and estimator == "xgb" and policy == "{pol}"'
                out += [Entry(f"fail{bw}{aw}{pw}", "analogs/activity_all.csv", base, "fail_0.5_2", "int", "dev", agg="sum"),
                        Entry(f"medLogErr{bw}{aw}{pw}", "analogs/activity_all.csv", base, "log_error", "dec2", "dev", agg="median"),
                        Entry(f"medR{bw}{aw}{pw}", "analogs/activity_all.csv", base, "R", "dec2", "dev", agg="median")]
    out += [Entry("etaStar", "analogs/protocol_inputs.json", "eta", "", "g3", "dev"),
            Entry("rhoStar", "analogs/protocol_inputs.json", "rho_star", "", "g3", "dev"),
            Entry("heConfirmatory", "analogs/protocol_inputs.json", "family/H-E/confirmatory", "", "yesno", "dev"),
            Entry("hbOneConfirmatory", "analogs/protocol_inputs.json", "family/H-B1/confirmatory", "", "yesno", "dev"),
            Entry("hbTwoConfirmatory", "analogs/protocol_inputs.json", "family/H-B2/confirmatory", "", "yesno", "dev"),
            # Ruling 101: only the counted pairings (the 1-year ones: four single quarters and the pooled AQ3+AQ4-1y) are tests of the
            # rule; the 2-year pairings are descriptive (nPlaceboPairsDescriptive in _simulation_entries)
            Entry("nPlaceboPairs", "analogs/placebo_he.csv", "counted == True and p == p", "p", "int", "dev", agg="count",
                  note="the counted (1-year) pairings that have a p-value"),
            Entry("nPlaceboRejectFive", "analogs/placebo_he.csv", "counted == True and p <= 0.05", "p", "int", "dev", agg="count",
                  note="the counted (1-year) pairings that reject at 0.05")]
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Ruling 101: the simulation stage (analogs/simulation.csv, protocol_inputs.json, placebo_he.csv, eta_selection.csv)
# ---------------------------------------------------------------------------------------------------------------------
SIM, PROTOCOL, PLACEBO = "analogs/simulation.csv", "analogs/protocol_inputs.json", "analogs/placebo_he.csv"
#: The two levels every size and power is read at: the first Holm level of Test A (alpha / 3) and the family-wise level.
#: The columns of simulation.csv are named after them; paper_macros.check_simulation_rule compares them with the rule's.
SIM_LEVELS = (("Third", "reject_0.0166667"), ("Five", "reject_0.05"))
#: The laws whose size rows are the nulls of an H-B test (es.run_analogs.HB_PAIRS); the rule echoes them in protocol_inputs.json
#: and paper_macros.check_simulation_rule compares them at every build.
HB_LAWS = {"H-B1": ("DART3", "LEV6"), "H-B2": ("DART3", "ROLL_DISP9")}
HYPOTHESIS_WORD = {"H-E": "he", "H-B1": "hbOne", "H-B2": "hbTwo"}
#: The rows the registration rule reads (es.run_analogs._test_rows): the primary strata (the decile strata are the sensitivity of
#: Ruling 68), a row without strata included, and for an H-B test the rows of its laws (a row without a law is read by every test).
PRIMARY_ROWS = '(strata.isna() or strata == "primary")'
#: size macro prefix -> (hypothesis, the test's name in simulation.csv, its laws, the key of the rule's row count)
SIZE_TESTS = {"heSizePooled": ("H-E", "H-E pooled A+B, size", (), "k_rows/H-E"),
              "heSizeSingle": ("H-E", "H-E single A, size", (), "k_rows_interim"),
              "hbOneSize": ("H-B1", "H-B size", HB_LAWS["H-B1"], "k_rows/H-B1"),
              "hbTwoSize": ("H-B2", "H-B size", HB_LAWS["H-B2"], "k_rows/H-B2")}
#: The couplings of the power curve in words (RhoFive is rho = 0.05); the file's grid must be exactly these (check_simulation_rule).
RHO_WORD = {0.05: "RhoFive", 0.1: "RhoTen", 0.2: "RhoTwenty", 0.3: "RhoThirty", 0.5: "RhoFifty", 0.7: "RhoSeventy", 0.9: "RhoNinety"}
CURVE_TEST, TARGET_TEST = "H-E pooled A+B, power curve", "H-E pooled A+B, power"


def size_query(test: str, laws=()) -> str:
    """The rows of ``test`` that the registration rule reads, as a pandas query."""
    return f'test == "{test}" and {PRIMARY_ROWS}' + (f" and (law.isna() or law in {list(laws)})" if laws else "")


def _simulation_entries() -> list[Entry]:
    """What the simulation stage wrote, read the way the registered rule reads it. A worst case is the maximum over the rows the
    rule reads (agg "argmax" on the rate) and the row that governs it is the first with that maximum: its law, scenario and
    persistence are read from that row, so a value and its row can never come from different rows."""
    out: list[Entry] = []
    for prefix, (hid, test, laws, _) in SIZE_TESTS.items():
        q = size_query(test, laws)
        reads = f"'{test}' under the primary strata" + (f", laws {', '.join(laws)}" if laws else "")
        out.append(Entry(f"{prefix}NSim", SIM, q, "n_sim", "int", "dev", agg="same",
                         note=f"replications behind each rate of {reads}: one number for all of them"))
        for lw, col in SIM_LEVELS:
            at = "alpha / 3" if lw == "Third" else "the family-wise level"
            out += [Entry(f"{prefix}{lw}", SIM, q, col, "dec3", "dev", agg="argmax", by=col,
                          note=f"worst-case size at {at}: the highest rate over {reads}"),
                    Entry(f"{prefix}{lw}WorstRow", SIM, q, "law,scenario,persistence", "where", "dev", agg="argmax", by=col,
                          note=f"law, scenario and persistence of the row that governs {prefix}{lw} (the first row on a tie)")]
            out += [Entry(f"{prefix}{lw}{attr.capitalize()}", SIM, q, attr, "text", "dev", agg="argmax", by=col, nullable=attr == "law",
                          note=f"the {attr} of the row that governs {prefix}{lw}" + ("; an H-E row has none" if attr == "law" else ""))
                    for attr in ("law", "scenario", "persistence")]
    third, five = SIM_LEVELS[0][1], SIM_LEVELS[1][1]
    # power: the H-E curve by coupling rho, the H-E row at rho* (the target), and the two H-B tests (a failed calibration is a NaN)
    curve = f'test == "{CURVE_TEST}"'
    for rho, rw in RHO_WORD.items():
        q = f"{curve} and rho == {rho!r}"
        out += [Entry(f"hePowerCurveThird{rw}", SIM, q, third, "dec2", "dev", note=f"H-E power at alpha / 3, coupling rho = {rho}"),
                Entry(f"hePowerCurveFive{rw}", SIM, q, five, "dec2", "dev", note=f"H-E power at the family-wise level, rho = {rho}"),
                Entry(f"hePowerCurveSelc{rw}", SIM, q, "mean_selc", "pct2", "dev", note=f"mean SELC of the simulated paths at rho = {rho}")]
    out.append(Entry("hePowerCurveNSim", SIM, curve, "n_sim", "int", "dev", agg="same", note="replications behind each point of the curve"))
    target = f'test == "{TARGET_TEST}"'
    out += [Entry("hePowerTargetThird", SIM, target, third, "dec2", "dev", note="H-E power at alpha / 3 at the coupling rho*"),
            Entry("hePowerTargetFive", SIM, target, five, "dec2", "dev", note="H-E power at the family-wise level at rho*"),
            Entry("hePowerTargetRho", SIM, target, "rho", "g3", "dev", note="the coupling of the target row (the same number as rhoStar)"),
            Entry("hePowerTargetSelc", SIM, target, "target_selc", "pct2", "dev",
                  note="the effect the coupling is searched for: half the loss-weighted pooled SELC of AQ3 and AQ4"),
            Entry("hePowerTargetMeanSelc", SIM, target, "mean_selc", "pct2", "dev", note="mean SELC of the simulated paths at rho*"),
            Entry("hePowerTargetNSim", SIM, target, "n_sim", "int", "dev", note="replications behind the power at rho*"),
            Entry("hePowerTargetNote", SIM, target, "note", "text", "dev", nullable=True,
                  note="what the search says about the target (for example that it is below the grid); n/a when nothing")]
    for hid, hw in (("H-B1", "hbOne"), ("H-B2", "hbTwo")):
        for lw, col in SIM_LEVELS:
            at = "alpha / 3" if lw == "Third" else "the family-wise level"
            out.append(Entry(f"{hw}Power{lw}", SIM, f'test == "{hid} power"', col, "dec2", "dev", nullable=True,
                             note=f"{hid} power at {at}; n/a when the calibration failed (a NaN rate)"))
    # the registered family (protocol_inputs.json), hypothesis by hypothesis
    for hid, hw in HYPOTHESIS_WORD.items():
        key = f"family/{hid}/"
        out += [Entry(f"{hw}FamilyStatus", PROTOCOL, key + "confirmatory", "", "registered", "dev",
                      note="confirmatory, or estimand when the size (or, for H-B, the power) check demoted the test"),
                Entry(f"{hw}Approximate", PROTOCOL, key + "approximate", "", "yesno", "dev",
                      note="whether the interval is labelled approximate: the estimated size is significantly above nominal"),
                Entry(f"{hw}Reasons", PROTOCOL, key + "reasons", "", "reasons", "dev", note="the reasons of a demotion, or none"),
                Entry(f"{hw}FalseDemotion", PROTOCOL, key + "false_demotion_bound", "", "dec3", "dev",
                      note="probability that a correctly sized test is demoted, under independent rows (the rule's bound)")]
    out.append(Entry("heInterimApproximate", PROTOCOL, "family/H-E/interim_approximate", "", "yesno", "dev",
                     note="whether Test A's interim p-value is labelled approximate (the single-quarter test)"))
    out += [Entry(f"{hw}PowerLowerBound", PROTOCOL, f"family/{hid}/power_lower_bound", "", "yesno", "dev",
                  note=f"whether the power of {hid} is only a lower bound (the target is above the calibration grid)")
            for hid, hw in (("H-B1", "hbOne"), ("H-B2", "hbTwo"))]
    out += [Entry("heFalseDemotionSim", "derived:heFalseDemotionSim", "", "", "dec3", "dev",
                  note="the part of H-E's false-demotion bound that comes from the simulation rows, recomputed from the rule's constants"),
            Entry("heFalseDemotionPlacebo", "derived:heFalseDemotionPlacebo", "", "", "dec4", "dev",
                  note="the part that comes from the counted pooled placebo pairings, recomputed from the rule's constants")]
    # eta* and whether it sits at the top edge of the grid it was chosen from
    out.append(Entry("etaAtGridTop", "derived:etaAtGridTop", "", "", "yesno", "dev",
                     note="whether eta* is the largest value of the grid in analogs/eta_selection.csv: the optimum is then at its edge"))
    # the rule's constants and the persistence medians
    rule = (("ruleFactor", "factor", "g3"), ("ruleApproxFactor", "approximate_factor", "g3"),
            ("ruleBinomLevel", "binomial_level", "g3"), ("ruleMinPower", "min_power", "g3"),
            ("ruleLevelThird", "levels/0", "g3"), ("ruleLevelFive", "levels/1", "g3"),
            ("ruleKHe", "k_rows/H-E", "int"), ("ruleKHbOne", "k_rows/H-B1", "int"), ("ruleKHbTwo", "k_rows/H-B2", "int"),
            ("ruleKHeInterim", "k_rows_interim", "int"), ("nSimHeSize", "n_sim/he_size", "int"), ("nSimHeRho", "n_sim/he_rho", "int"),
            ("nSimHePower", "n_sim/he_power", "int"), ("nSimHbSize", "n_sim/hb_size", "int"), ("nSimHbPower", "n_sim/hb_power", "int"),
            ("nSimHbCalibrate", "n_sim/hb_calibrate", "int"), ("ruleNRep", "n_rep", "int"))
    out += [Entry(name, PROTOCOL, f"rule/{key}", "", fmt, "dev", note="a constant of the designation rule, as the run used it")
            for name, key, fmt in rule]
    for law in ("DART3", "LEV6", "ROLL_DISP9"):
        out += [Entry(f"phiCorrected{BLOCK_WORD[law]}", PROTOCOL, f"phi/{law}/phi_score_median", "", "dec3", "dev",
                      note=f"median over nodes of the corrected persistence of the {law} score"),
                Entry(f"phiRaw{BLOCK_WORD[law]}", PROTOCOL, f"phi/{law}/phi_score_raw_median", "", "dec3", "dev",
                      note=f"median over nodes of the uncorrected persistence of the {law} score")]
    # the placebo pairings that do not count (the 2-year ones, descriptive) and the counted pooled and single ones
    pooled = 'pair.str.startswith("AQ3+AQ4")'
    out += [Entry("nPlaceboPairsDescriptive", PLACEBO, "counted == False and p == p", "p", "int", "dev", agg="count",
                  note="the pairings that do not count (the 2-year ones, whose P&L year is in the training window) that have a p-value"),
            Entry("nPlaceboRejectFiveDescriptive", PLACEBO, "counted == False and p <= 0.05", "p", "int", "dev", agg="count",
                  note="the pairings that do not count and reject at 0.05"),
            Entry("placeboPooledP", PLACEBO, f"counted == True and {pooled}", "p", "pval", "dev",
                  note="rotation p-value of the counted pooled pairing AQ3+AQ4-1y"),
            Entry("placeboPooledSelc", PLACEBO, f"counted == True and {pooled}", "selc_obs", "pct1", "dev",
                  note="SELC of the counted pooled pairing AQ3+AQ4-1y"),
            Entry("placeboSinglesPMin", PLACEBO, f"counted == True and ~{pooled}", "p", "pval", "dev", agg="min",
                  note="smallest rotation p-value of the counted single-quarter pairings")]
    # the names the sections already use stay, as aliases of the primary entries: nothing matches more than one row
    primary = {e.name: e for e in out}

    def alias(old: str, new: str) -> Entry:
        return replace(primary[new], name=old, note=f"alias of {new} (Ruling 101): the old name stays so that the text that uses it compiles")

    laws = tuple(dict.fromkeys(law for pair in HB_LAWS.values() for law in pair))
    out += [alias("heSizePooled", "heSizePooledFive"), alias("heSizeSingle", "heSizeSingleFive"),
            alias("hePowerPooled", "hePowerTargetThird"), alias("hbOnePower", "hbOnePowerThird"), alias("hbTwoPower", "hbTwoPowerThird"),
            Entry("hbSize", SIM, size_query("H-B size", laws), five, "dec3", "dev", agg="argmax", by=five,
                  note="the larger of hbOneSizeFive and hbTwoSizeFive: the worst-case size of the portability bootstrap at the "
                       "family-wise level over the rows of both H-B tests (Ruling 101; the old name stays)")]
    return out


def _econ_entries() -> list[Entry]:
    out = []
    econ_ids = ("AQ3", "AQ4", "AQ1", "AQ2", "2024-2025", "24H2-25H2", "25H1-26H1")
    for aid in econ_ids:
        aw = ANALYSIS_WORD[aid]
        for stat, sw in STAT_WORD.items():
            pf, cf = STAT_FMT[stat]
            q = f'analysis == "{aid}" and stat == "{stat}"'
            out += [Entry(f"{sw}{aw}", "economics/bootstrap_ci.csv", q, "estimate", pf, "dev"),
                    Entry(f"{sw}{aw}CI", "economics/bootstrap_ci.csv", q, "lo,hi", cf, "dev")]
        for key, (pf, cf) in E1_FMT.items():
            kw = "Net" if key == "NET" else "Esred" if key == "ESred" else "J" + LAM_WORD[float(key[2:])]
            for part in ("timing", "level", "total"):
                q = f'analysis == "{aid}" and stat == "E1:{key}:{part}"'
                out += [Entry(f"eOne{kw}{part.capitalize()}{aw}", "economics/bootstrap_ci.csv", q, "estimate", pf, "dev"),
                        Entry(f"eOne{kw}{part.capitalize()}{aw}CI", "economics/bootstrap_ci.csv", q, "lo,hi", cf, "dev")]
        for pol, pw in (("P1", "POne"), ("P4", "PFour"), ("P5", "PFive"), ("P6", "PSix"), ("P4@r*", "PFourMatched")):
            q = f'analysis == "{aid}" and policy == "{pol}" and position == "premium" and fees == "calibrated" and comparator == "none"'
            for col, cw, fmt in (("SELC", "selc", "pct1"), ("NET", "net", "usd0"), ("SNET", "snet", "usd2"),
                                 ("ESred", "esred", "usd0"), ("CAP", "cap", "pct1"), ("n_days", "nDays", "int"),
                                 ("alert_rate", "alertRate", "pct1")):
                out.append(Entry(f"{cw}{pw}{aw}", "economics/summary.csv", q, col, fmt, "dev"))
        for expo, ew in (("long_spikes", "Long"), ("short_spikes", "Short")):          # P1 by the position's spike exposure
            q = (f'analysis == "{aid}" and policy == "P1" and position == "premium[{expo}]" and fees == "calibrated" '
                 f'and comparator == "none"')
            for col, cw, fmt in (("SELC", "selc", "pct1"), ("NET", "net", "usd0"), ("ESred", "esred", "usd0")):
                out.append(Entry(f"{cw}{ew}POne{aw}", "economics/summary.csv", q, col, fmt, "dev"))
        out += [Entry(f"rotP{aw}", "economics/rotation.csv", f'window == "{aid}"', "p", "pval", "dev"),
                Entry(f"sharpeAlerted{aw}", "economics/volmanagement_check.csv", f'analysis == "{aid}"', "sharpe_alerted", "dec3", "dev"),
                Entry(f"sharpeNotAlerted{aw}", "economics/volmanagement_check.csv", f'analysis == "{aid}"', "sharpe_not_alerted", "dec3", "dev")]
    out += [Entry("rotPPooledRehearsal", "economics/rotation.csv", 'window.str.startswith("AQ3+AQ4")', "p", "pval", "dev"),
            Entry("selcPooledRehearsal", "economics/bootstrap_ci.csv", 'analysis.str.startswith("AQ3+AQ4") and stat == "SELC"', "estimate", "pct1", "dev"),
            Entry("selcPooledRehearsalCI", "economics/bootstrap_ci.csv", 'analysis.str.startswith("AQ3+AQ4") and stat == "SELC"', "lo,hi", "cipct1", "dev"),
            Entry("netPooledRehearsal", "economics/bootstrap_ci.csv", 'analysis.str.startswith("AQ3+AQ4") and stat == "NET"', "estimate", "usd0", "dev"),
            Entry("netPooledRehearsalCI", "economics/bootstrap_ci.csv", 'analysis.str.startswith("AQ3+AQ4") and stat == "NET"', "lo,hi", "ciusd0", "dev"),
            Entry("esredPooledRehearsal", "economics/bootstrap_ci.csv", 'analysis.str.startswith("AQ3+AQ4") and stat == "E3:ESred"', "estimate", "usd0", "dev"),
            Entry("esredPooledRehearsalCI", "economics/bootstrap_ci.csv", 'analysis.str.startswith("AQ3+AQ4") and stat == "E3:ESred"', "lo,hi", "ciusd0", "dev")]
    for stat in E4_ESRED_STATS:                  # Ruling 118: the tail measure at matched activity, for the pooled rehearsal as well
        pf, cf = STAT_FMT[stat]
        q = f'analysis.str.startswith("AQ3+AQ4") and stat == "{stat}"'
        out += [Entry(f"{STAT_WORD[stat]}PooledRehearsal", "economics/bootstrap_ci.csv", q, "estimate", pf, "dev"),
                Entry(f"{STAT_WORD[stat]}PooledRehearsalCI", "economics/bootstrap_ci.csv", q, "lo,hi", cf, "dev")]
    for stat in ("E4:dSELC|calendar@count", "E4:dSELC|VOL@count"):      # the SELC contrasts of the pooled rehearsal (Ruling 148: the secondary analysis is read beside them)
        pf, cf = STAT_FMT[stat]
        q = f'analysis.str.startswith("AQ3+AQ4") and stat == "{stat}"'
        out += [Entry(f"{STAT_WORD[stat]}PooledRehearsal", "economics/bootstrap_ci.csv", q, "estimate", pf, "dev"),
                Entry(f"{STAT_WORD[stat]}PooledRehearsalCI", "economics/bootstrap_ci.csv", q, "lo,hi", cf, "dev")]
    return out


#: The exploratory secondary economic analysis (controller Ruling 148): the same development economic analysis with the plain-DART-lag block
#: (DART3) as the alert model, written by ``es.run_economics_block`` into a folder of its own beside the primary one. The macros carry the
#: infix SecDart (``dselcCalCountSecDartAqThree``), so that no name of the primary analysis is touched.
SEC_DART_DIR = "economics_block/DART3"
SEC_DART_STATS = ("SELC", "NET", "E3:ESred", "E4:dSELC|calendar@count", "E4:dESred|calendar@count", "E4:dSELC|VOL@count",
                  "E4:dESred|VOL@count")


def _econ_block_entries() -> list[Entry]:
    """The estimates and intervals of ``SEC_DART_STATS`` for the four quarterly analogues and the pooled AQ3 + AQ4 rehearsal, the alert rate of
    P1 and the rotation p-value of each quarter, all of the secondary analysis with the plain DART lags as the alert model."""
    out = []
    boot, summary, rotation = (f"{SEC_DART_DIR}/{name}.csv" for name in ("bootstrap_ci", "summary", "rotation"))
    for aid in QUARTER_ANALOGUES:
        aw = ANALYSIS_WORD[aid]
        for stat in SEC_DART_STATS:
            pf, cf = STAT_FMT[stat]
            q = f'analysis == "{aid}" and stat == "{stat}"'
            out += [Entry(f"{STAT_WORD[stat]}SecDart{aw}", boot, q, "estimate", pf, "dev"),
                    Entry(f"{STAT_WORD[stat]}SecDart{aw}CI", boot, q, "lo,hi", cf, "dev")]
        q = f'analysis == "{aid}" and policy == "P1" and position == "premium" and fees == "calibrated" and comparator == "none"'
        out += [Entry(f"alertRatePOneSecDart{aw}", summary, q, "alert_rate", "pct1", "dev"),
                Entry(f"rotPSecDart{aw}", rotation, f'window == "{aid}"', "p", "pval", "dev")]
    for stat in SEC_DART_STATS:
        pf, cf = STAT_FMT[stat]
        q = f'analysis.str.startswith("AQ3+AQ4") and stat == "{stat}"'
        out += [Entry(f"{STAT_WORD[stat]}SecDartPooledRehearsal", boot, q, "estimate", pf, "dev"),
                Entry(f"{STAT_WORD[stat]}SecDartPooledRehearsalCI", boot, q, "lo,hi", cf, "dev")]
    out.append(Entry("rotPSecDartPooledRehearsal", rotation, 'window.str.startswith("AQ3+AQ4")', "p", "pval", "dev"))
    return out


def _other_dev_entries() -> list[Entry]:
    out = []
    for aid in ("2024-2025", "25H1-26H1", "AQ3", "AQ4"):
        aw = ANALYSIS_WORD[aid]
        for blk in ("G6", "DART3", "LEV6", "ROLL_DISP9", "DART18", "G1_24"):
            bw = BLOCK_WORD[blk]
            q = f'analysis == "{aid}" and block == "{blk}" and estimator == "xgb"'
            out += [Entry(f"precQ{bw}{aw}", "labels_pr/pr_all.csv", q, "precision_q", "dec2", "dev", agg="median"),
                    Entry(f"recQ{bw}{aw}", "labels_pr/pr_all.csv", q, "recall_q", "dec2", "dev", agg="median"),
                    Entry(f"precEval{bw}{aw}", "labels_pr/pr_all.csv", q, "precision_eval", "dec2", "dev", agg="median"),
                    Entry(f"precLift{bw}{aw}", "labels_pr/pr_all.csv", q, "precision_lift", "dec2", "dev", agg="median")]
    for aid in ("LC-full-q2023", "LC-full-q2324", "LC-h1-q2023", "LC-h1-q2324"):
        aw = ANALYSIS_WORD[aid]
        for blk in ("LEV6", "DART18", "G1_24", "DART3"):
            bw = BLOCK_WORD[blk]
            q = f'analysis == "{aid}" and block == "{blk}"'
            out += [Entry(f"failLc{bw}{aw}", "labels_pr/label_control_2x2.csv", q, "failures", "int", "dev"),
                    Entry(f"medRLc{bw}{aw}", "labels_pr/label_control_2x2.csv", q, "median_R", "dec2", "dev")]
    for mon, mw in (("W1_norm", "Wone"), ("PSI", "Psi"), ("domain_auc", "DomainAuc"), ("early_rate", "EarlyRate")):
        for stat, sw, fmt in (("spearman", "Rho", "dec2"), ("auroc_fail", "Auroc", "dec2")):
            out.append(Entry(f"mon{mw}{sw}Median", "monitors/monitor_eval.csv", f'monitor == "{mon}"', stat, fmt, "dev", agg="median"))
            for aid in ("25H1-26H1", "AQ3", "AQ4"):
                aw = ANALYSIS_WORD[aid]
                out.append(Entry(f"mon{mw}{sw}{aw}", "monitors/monitor_eval.csv",
                                 f'monitor == "{mon}" and analysis == "{aid}"', stat, fmt, "dev", agg="median"))
            # Ruling 98: "on the quarters" is the median over the four quarterly analogues (AQ1-AQ4) and their blocks
            out.append(Entry(f"mon{mw}{sw}Quarters", "monitors/monitor_eval.csv",
                             f'monitor == "{mon}" and analysis in {list(QUARTER_ANALOGUES)}', stat, fmt, "dev", agg="median",
                             note="median over the blocks of AQ1, AQ2, AQ3 and AQ4 (the legacy analyses are left out)"))
    # Ruling 83: the no-drift reference of the domain classifier, on the quarterly analogues (descriptive; it enters no test)
    for word, how in (("Min", "min"), ("Max", "max")):
        out.append(Entry(f"monDomainPlacebo{word}", "monitors/monitor_eval.csv",
                         f'monitor == "domain_auc" and analysis in {list(QUARTER_ANALOGUES)}', "placebo_median", "dec2", "dev",
                         agg=how, note=f"the {how} over the blocks of AQ1-AQ4 of the median no-drift AUROC (a split inside the calibration window)"))
    # Ruling 100 (Section 4.6): the no-drift context of the audit counts, from the size study of the portability bootstrap: the 95th
    # percentile over replications of the number of nodes outside the band without drift, as the smallest and the largest over the
    # score laws and persistences of the study (descriptive context for the counts; it enters no test)
    for bw, band in (("Wide", "0.5_2"), ("Mid", "0.67_1.5"), ("Tight", "0.8_1.25")):
        for word, how in (("Min", "min"), ("Max", "max")):
            out.append(Entry(f"noDriftFailNinetyFive{bw}{word}", "analogs/simulation.csv", 'test == "H-B size" and scenario == "no_drift"',
                             f"nodrift_fail_q95_{band}", "g3", "dev", agg=how,
                             note=f"the {how} over the laws DART3, LEV6, ROLL_DISP9 and the fitted and upper persistence of the 95th "
                                  f"percentile of the number of nodes outside the band {band} without drift"))
    # Ruling 94: only the one-year placebo pairings count in the registration rule; the two-year pairings are descriptive
    for word, tag in (("OneYear", "-1y"), ("TwoYear", "-2y")):
        out += [Entry(f"nPlacebo{word}", "analogs/placebo_he.csv", f'pair.str.endswith("{tag}")', "p", "int", "dev", agg="count"),
                Entry(f"nPlacebo{word}Reject", "analogs/placebo_he.csv", f'pair.str.endswith("{tag}") and p <= 0.05', "p", "int",
                      "dev", agg="count"),
                Entry(f"placebo{word}PMin", "analogs/placebo_he.csv", f'pair.str.endswith("{tag}")', "p", "pval", "dev", agg="min")]
    # which tests carry the Benjamini-Hochberg rejections on the rehearsal window (inference/bh.csv)
    out += [Entry("bhEconomicDecTests", "inference/bh.csv", 'family == "economic robustness" and test.str.startswith("P1 dec ")',
                  "test", "int", "dev", agg="count", note="the always-DEC tests of the economic family, one per fee setting"),
            Entry("bhEconomicDecRejections", "inference/bh.csv",
                  'family == "economic robustness" and test.str.startswith("P1 dec ") and bh_reject == True', "test", "int", "dev",
                  agg="count"),
            Entry("bhMonitorsEarlyRateTests", "inference/bh.csv",
                  'family == "drift monitors" and test.str.endswith(" early_rate")', "test", "int", "dev", agg="count"),
            Entry("bhMonitorsEarlyRateRejections", "inference/bh.csv",
                  'family == "drift monitors" and test.str.endswith(" early_rate") and bh_reject == True', "test", "int", "dev",
                  agg="count")]
    return out


#: The summary of the trailing windows of Figure 2 (es.run_driftpath, plan 2C T1b): the file the macros read, and for each of its metrics the macro prefix, the
#: format and what it counts. The macro is <prefix><policy word><analysis word>, for P1 (POne) and P4 (PFour) in 2024-2025 (YY) and 25H1-26H1 (PostOpen).
DRIFT_SUMMARY = "paper/drift_summary.csv"
DRIFT_METRICS = (
    ("driftOut", "nodes_outside", "int", "nodes whose R over the whole deployment period is outside the band (R below its lower or above its upper end)"),
    ("driftAbove", "nodes_above", "int", "nodes whose R over the whole deployment period is above the band"),
    ("driftBelow", "nodes_below", "int", "nodes whose R over the whole deployment period is below the band"),
    ("driftWinOutMean", "window_outside_mean", "dec1", "mean over the trailing windows of the number of nodes whose window R is outside the band"),
    ("driftWinOutMin", "window_outside_min", "int", "smallest number, over the trailing windows, of nodes whose window R is outside the band"),
    ("driftWinOutMax", "window_outside_max", "int", "largest number, over the trailing windows, of nodes whose window R is outside the band"),
    ("driftWinMedMin", "window_median_min", "dec2", "smallest, over the trailing windows, of the median over nodes of the window R"),
    ("driftWinMedMax", "window_median_max", "dec2", "largest, over the trailing windows, of the median over nodes of the window R"),
    ("driftWinN", "window_n", "int", "number of trailing windows"))


def _drift_entries() -> list[Entry]:
    """The macros of drift_summary.csv (written by es.run_driftpath): for the fixed (P1) and the rolling (P4) cutoff of the two main analyses, how many nodes
    are outside the band over the whole period and how many of them above and below it, and what the trailing windows of eight weeks (the macros
    driftWindowWeeks and driftWindowMinWeeks) do: their number, the nodes outside the band in each, and the median over nodes of the window R."""
    out = []
    for aid in ("2024-2025", "25H1-26H1"):
        aw = ANALYSIS_WORD[aid]
        for pol, pw in (("P1", "POne"), ("P4", "PFour")):
            for prefix, metric, fmt, what in DRIFT_METRICS:
                q = f'analysis == "{aid}" and policy == "{pol}" and metric == "{metric}"'
                out.append(Entry(f"{prefix}{pw}{aw}", DRIFT_SUMMARY, q, "value", fmt, "dev", note=f"{pol}, {aid}: {what}"))
    return out


def sealed_entries(test_id: str) -> list[Entry]:
    tw = ANALYSIS_WORD["SA" if test_id == "A" else "SB"]
    stage = f"sealed{test_id}"
    res, est, e5 = f"sealed/{test_id}/results.json", f"sealed/{test_id}/tables/estimands.csv", f"sealed/{test_id}/tables/E5_blocks.csv"
    out = [Entry(f"ran{tw}", res, "coverage/ran", "", "yesno", stage),
           Entry(f"usableNodes{tw}", res, "coverage/usable_nodes", "", "int", stage),
           Entry(f"rowsRead{tw}", res, "rows_read", "", "int", stage),
           Entry(f"rowsKept{tw}", res, "rows_kept", "", "int", stage),
           Entry(f"tauMaxDiff{tw}", res, "tau_check/max_abs_diff", "", "g3", stage),
           Entry(f"started{tw}", res, "started_utc", "", "date", stage),
           Entry(f"protocolSha{tw}", res, "protocol_sha256", "", "text", stage),
           Entry(f"eta{tw}", res, "eta", "", "g3", stage),
           Entry(f"heP{tw}", res, "H-E/p", "", "pval", stage),
           Entry(f"heTobs{tw}", res, "H-E/t_obs", "", "dec2", stage),          # studentized (Ruling 55): unitless, not a SELC
           Entry(f"heStatus{tw}", res, "H-E/status", "", "text", stage),
           Entry(f"heNnull{tw}", res, "H-E/n_null", "", "int", stage),
           Entry(f"estimandsScope{tw}", res, "estimands_scope", "", "text", stage)]
    for hid, hw in HB_WORD.items():
        # Ruling 110: the percentile interval is the primary one. For an estimand that the registration rule demoted, the basic
        # interval inherits the bootstrap shift that drove the over-rejection; the basic one stays, as CIBasic.
        ci = Entry(f"{hw}CI{tw}", res, f"{hid}/ci", "", "ci3", stage,
                   note="the percentile bootstrap interval, the primary one (Ruling 110): for a demoted estimand the basic interval "
                        "inherits the bootstrap shift that drove the over-rejection")
        out += [Entry(f"{hw}T{tw}", res, f"{hid}/t_hat", "", "dec3", stage),
                Entry(f"{hw}P{tw}", res, f"{hid}/p", "", "pval", stage, estimand_of=hid,
                      note=f"the basic-bootstrap p-value; n/a (estimand) when the run's family does not list {hid} as confirmatory"),
                ci,
                Entry(f"{hw}CIBasic{tw}", res, f"{hid}/ci_basic", "", "ci3", stage,
                      note="the basic bootstrap interval (it goes with the basic p-value; spec 6.2 names it, Ruling 110 makes the "
                           "percentile one primary)"),
                replace(ci, name=f"{hw}CIPercentile{tw}", note=f"alias of {ci.name} (Ruling 110): the percentile interval, by its name"),
                Entry(f"{hw}N{tw}", res, f"{hid}/n_nodes", "", "int", stage),
                Entry(f"{hw}Status{tw}", res, f"{hid}/status", "", "text", stage)]
    out += [Entry(f"runStatus{tw}", res, "status", "", "text", stage,
                  note="results.json status of the run: completed, completed with secondary errors, completed with errors, or "
                       "not run (coverage rule)"),
            Entry(f"nRunErrors{tw}", res, "errors", "", "int", stage, agg="count",
                  note="the number of analyses that failed and were recorded in results.json errors")]
    # Task 9 (the Test A text): the run record that the execution record states. All four are header keys of results.json, which a test
    # that did not run still has (es.paper_macros.NOT_RUN_KEYS).
    out += [Entry(f"departures{tw}", res, "departures", "", "int", stage, agg="count",
                  note="the number of files of the code that differ from the manifest of the protocol, each recorded as a departure from the "
                       "protocol (RUN_LOG.txt, departures.txt); 0 when the run used the code fixed in the protocol"),
            Entry(f"heApprox{tw}", res, "H-E/approximate", "", "approxlabel", stage,
                  note="whether the designation rule labels H-E's p-value approximate (the interim single-window one in Test A, the pooled one in Test B)"),
            Entry(f"startedUtc{tw}", res, "started_utc", "", "utcmin", stage,
                  note=f"the start of the run, UTC to the minute (started{tw} prints the date alone)"),
            Entry(f"registeredUtc{tw}", res, "registered_utc", "", "utcmin", stage,
                  note="the time of the protocol record that the run verified at its start (docs/sealed/REGISTRATION.md), UTC to the minute")]
    for pc, pw in (("P2_season", "PcSeason"), ("P2_all", "PcAll")):
        out += [Entry(f"{pw[0].lower() + pw[1:]}Diff{tw}", res, f"PC/{pc}/median_diff", "", "dec3", stage),
                Entry(f"{pw[0].lower() + pw[1:]}CI{tw}", res, f"PC/{pc}/ci", "", "ci3", stage)]
    for est_name, ew in EST_WORD.items():
        out += [Entry(f"hsFail{ew}{tw}", res, f"H-S/{est_name}/failures", "", "int", stage),
                Entry(f"hsFail{ew}CI{tw}", res, f"H-S/{est_name}/ci", "", "ci1", stage)]
    for stat, sw in STAT_WORD.items():
        pf, cf = STAT_FMT[stat]
        q = f'stat == "{stat}"'
        out += [Entry(f"{sw}{tw}", est, q, "estimate", pf, stage), Entry(f"{sw}{tw}CI", est, q, "lo,hi", cf, stage)]
    for key, (pf, cf) in E1_FMT.items():
        kw = "Net" if key == "NET" else "Esred" if key == "ESred" else "J" + LAM_WORD[float(key[2:])]
        for part in ("timing", "level", "total"):
            q = f'stat == "E1:{key}:{part}"'
            out += [Entry(f"eOne{kw}{part.capitalize()}{tw}", est, q, "estimate", pf, stage),
                    Entry(f"eOne{kw}{part.capitalize()}{tw}CI", est, q, "lo,hi", cf, stage)]
    for blk, bw in BLOCK_WORD.items():
        q = f'block == "{blk}"'
        out += [Entry(f"eFiveMedLogErr{bw}{tw}", e5, q, "median_log_error", "dec2", stage),
                Entry(f"eFiveMedLogErr{bw}{tw}CI", e5, q, "lo,hi", "ci2", stage),
                Entry(f"eFiveFail{bw}{tw}", e5, q, "failures", "int", stage),
                Entry(f"eFiveFail{bw}{tw}CI", e5, q, "failures_lo,failures_hi", "ci1", stage)]
    if test_id == "B":
        no_verdict = "null for an estimand, which has no replication verdict (Ruling 113), and in a Test B that ran without Test A"
        out += [Entry("heWording", res, "H-E/wording", "", "text", stage),
                Entry("holmHE", res, "holm/H-E", "", "yesno", stage, nullable=True),
                Entry("holmHBOne", res, "holm/H-B1", "", "yesno", stage, nullable=True),
                Entry("holmHBTwo", res, "holm/H-B2", "", "yesno", stage, nullable=True),
                Entry("hbOneReplicated", res, "H-B1/replicated", "", "yesno", stage, nullable=True, note=no_verdict),
                Entry("hbTwoReplicated", res, "H-B2/replicated", "", "yesno", stage, nullable=True, note=no_verdict)]
    return out


def _author_entries() -> list[Entry]:
    return [Entry("paperTitle", "author:title", "", "", "text", "author"),
            Entry("paperShortTitle", "author:short_title", "", "", "text", "author"),
            Entry("authorEmail", "author:email", "", "", "text", "author"),
            Entry("authorDept", "author:department", "", "", "text", "author"),
            Entry("authorOrcid", "author:orcid", "", "", "text", "author"),
            Entry("codeRepoURL", "author:github_url", "", "", "url", "author"),
            Entry("ssrnId", "author:ssrn_abstract_id", "", "", "text", "author"),
            Entry("aiTools", "author:ai_tools", "", "", "text", "author")]


def _pull_record_entries() -> list[Entry]:
    """The pull record of Test A (Ruling 121; Task 9): one sentence from the pull's own log (PULL_LOG and the entry of PULLS.json) that
    says which chunks were accepted incomplete (incomplete_chunks) and which days were excused (allow_missing_days), or that there were
    none. sealed_run does not read either, so no output file holds it: the controller writes the sentence after the pull into
    paper/author_inputs.json, key pull_record_test_a, as plain ASCII text of one line that ends with a full stop (the validator of
    es.paper_macros says what it refuses). An empty value keeps the macro pending: it prints \\PENDINGA, and the strict build of stage
    sealedA fails on it, like on every other macro of Test A that has no value yet."""
    return [Entry("pullRecordTestA", "author:pull_record_test_a", "", "", "text", "sealedA",
                  note="the controller's sentence on the pull's own log: chunks accepted incomplete and days excused, or none (Ruling 121); "
                       "paper/author_inputs.json, key pull_record_test_a; pending while it is empty")]


def _coverage_entries() -> list[Entry]:
    """Data coverage, written by es.impl_table --with-data (Task 3); pending until then."""
    out = []
    for market, mk in MARKET_WORD.items():
        out += [Entry(f"firstDate{mk}", "paper/coverage.csv", f'series == "{market}"', "first_local_date", "date", "dev"),
                Entry(f"lastDate{mk}", "paper/coverage.csv", f'series == "{market}"', "last_local_date", "date", "dev")]
    out += [Entry("fmmFirstDate", "paper/coverage.csv", 'series == "CAISO FMM (settlement leg)"', "first_local_date", "date", "dev"),
            Entry("nyisoIntFirstDate", "paper/coverage.csv", 'series == "NYISO integrated RT LBMP (settlement leg)"', "first_local_date", "date", "dev"),
            Entry("dupDropped", "paper/coverage.csv", "", "duplicates_dropped_total", "int", "dev", agg="max")]
    return out


def _fact_entries() -> list[Entry]:
    """Facts of the data and the protocol that the prose states (Ruling 74), each read from an output file."""
    impl, cov, d10 = "paper/implementation.json", "paper/coverage.csv", "legacy/d10_boundaries.csv"
    holdout = 'window_start == "2026-01-01" and window_end == "2026-07-01"'      # the 2026-H1 deployment window
    label = 'window_start == "2023-01-01" and window_end == "2024-01-01"'        # the 2023 label window
    return [
        Entry("dupRows", impl, "coverage/0/duplicates_dropped_total", "", "int", "dev",
              note="NYISO month-boundary duplicates dropped; every coverage row carries the same total (D11)"),
        Entry("dupBoundaries", impl, "coverage_notes", "", "int", "dev", pattern=r"(\d+) month starts",
              note="read from the one note that states it: implementation.json has no structured field for it yet"),
        Entry("dTenLabelMissed", d10, label, "missed_local_dates", "int", "dev", agg="sum",
              note="D10: rows of the 2023 label window's local dates that its UTC interval misses, four operators"),
        Entry("dTenHoldoutExtra", d10, holdout, "extra_other_dates", "int", "dev", agg="sum",
              note="D10: rows of 31 Dec 2025 that the UTC interval of the 2026-H1 window includes, four operators"),
        Entry("dTenHoldoutMissed", d10, holdout, "missed_local_dates", "int", "dev", agg="sum",
              note="D10: rows of the last local evening hours of 30 Jun 2026 that it misses, four operators"),
        Entry("dTenHoldoutRows", d10, holdout, "rows_utc", "int", "dev", agg="sum",
              note="D10: rows inside the 2026-H1 UTC interval, four operators"),
        Entry("caisoFallbackRows", cov, 'series == "CAISO" or series == "CAISO FMM (settlement leg)"', "hourly_rows", "int",
              "dev", agg="range",
              note="CAISO panel rows minus the rows of its FMM settlement leg: the hours settled at the real-time mean"),
        Entry("placeboFallbackDays", "derived:placeboFallbackDays", "", "", "int", "dev",
              note="days on which the placebo pairing of AQ1-AQ4 (the same quarter one and two years earlier) uses CAISO's "
                   "real-time-mean fallback: from the CAISO panel start to the day before the FMM first date (spec 6.2(a))"),
        Entry("protocolSha", "docsfile:PROTOCOL.sha256", "", "", "text", "sealedA",
              note="SHA-256 of the protocol, from docs/sealed/PROTOCOL.sha256; pending until the protocol is frozen"),
    ]


BH_FAMILY_WORD = {"audit robustness": "AuditRobustness", "economic robustness": "EconomicRobustness",
                  "drift monitors": "DriftMonitors"}
#: Ruling 118 (G1, M5): the audit-robustness family is descriptive, as H-B is (Ruling 110). Its p-values come from the
#: calibration-and-deployment bootstrap that the size study found oversized, so its BH flags are diagnostics and no count of its
#: rejections is reported or interpreted: it has no `bhRejections` macro, in development or in the sealed tests, and the tables
#: print the number of its tests and nothing of its rejections.
DESCRIPTIVE_BH_FAMILIES = ("audit robustness",)


def _bh_entries() -> list[Entry]:
    """Benjamini-Hochberg within the three exploratory families (spec 5.2; Ruling 37): development from
    inference/bh.csv (family, test, estimate, p, bh_reject, included), sealed from results.json["secondary"]."""
    out = []
    for fam, fw in BH_FAMILY_WORD.items():
        counted = fam not in DESCRIPTIVE_BH_FAMILIES                 # a descriptive family has tests, and no rejections to count
        out.append(Entry(f"bhTests{fw}", "inference/bh.csv", f'family == "{fam}"', "test", "int", "dev", agg="count",
                         when_column="included", when_query="included == True",
                         note="the tests BH ranks: the rows with a p-value (Ruling 80); a file without `included` counts every row"))
        if counted:
            out.append(Entry(f"bhRejections{fw}", "inference/bh.csv", f'family == "{fam}" and bh_reject == True', "test", "int", "dev",
                             agg="count"))
        for test_id in ("A", "B"):
            tw, res = ANALYSIS_WORD["SA" if test_id == "A" else "SB"], f"sealed/{test_id}/results.json"
            out.append(Entry(f"bhTests{fw}{tw}", res, f"secondary/{fam}/tests", "", "int", f"sealed{test_id}"))
            if counted:
                out.append(Entry(f"bhRejections{fw}{tw}", res, f"secondary/{fam}/bh_rejections", "", "int", f"sealed{test_id}"))
    return out


def _size_check_entries() -> list[Entry]:
    """The supplementary size check of gate G1 (Ruling 118, M8), derived from the archived probe outputs of docs/reviews/g1_probes
    (es.paper_macros.DERIVED): the worst rejection rate of H-E's rotation test over the scenarios that the size study does not
    contain, for the pooled test and for one window, at alpha / 3 and at the family-wise level; the replications per scenario; and
    the largest daily persistence among them. They describe a check made at the review, not the designation rule."""
    where = "the archived probe outputs docs/reviews/g1_probes (he_common_500.txt, he_trend_500.txt); not part of the designation rule"
    rows = (("PooledThird", "dec3", "worst pooled rejection rate at alpha / 3"), ("PooledFive", "dec3", "worst pooled rate at the family-wise level"),
            ("SingleThird", "dec3", "worst single-window rate at alpha / 3"), ("SingleFive", "dec3", "worst single-window rate at the family-wise level"),
            ("Reps", "int", "replications per scenario"), ("Phi", "dec2", "the largest daily persistence among the scenarios"))
    return [Entry(f"sizeCheck{word}", f"derived:sizeCheck{word}", "", "", fmt, "dev", note=f"{what} of the supplementary size check, from {where}")
            for word, fmt, what in rows]


ENTRIES: list[Entry] = (_const_entries() + _legacy_entries() + _analog_entries() + _simulation_entries() + _size_check_entries() + _econ_entries() + _econ_block_entries()
                        + _other_dev_entries() + _drift_entries() + _coverage_entries() + _fact_entries() + _bh_entries()
                        + sealed_entries("A") + sealed_entries("B") + _pull_record_entries() + _author_entries())


def entries_by_name() -> dict[str, Entry]:
    by = {e.name: e for e in ENTRIES}
    if len(by) != len(ENTRIES):
        dup = sorted({e.name for e in ENTRIES if sum(x.name == e.name for x in ENTRIES) > 1})
        raise ValueError(f"duplicate macro names: {dup}")
    return by
