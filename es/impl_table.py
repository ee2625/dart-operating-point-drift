"""The implementation table (spec v3 section 5 item 7), read from the code and the data.

Where each number comes from. (1) The code: a constant or a default argument (SEEDS, ALPHAS and BANDS from the light
module es.paper_pins, which mirrors es.run_legacy's and is tested equal; MIN_POSITIVES, the fees, es.paper_pins.PINS,
v2.panel_prep.G1_LAGS and G1_ROLL_WINDOWS, the bounds and iteration count of es.policies, the settings of
es.run_monitors including MONITOR_DAYS, src.transfer.fit_scaler's minimum_scale, the default seed of
es.scores.score_analysis). (2) Probes of
the code on toy data, for settings that are literals inside a function (the stress quantile and fee multiplier, the
calendar definitions, the domain classifier's splitter and arguments, the rotation test's reference set, the
rolling-window tolerance), so the table says what the function does. (3) The data, with --with-data. (4) A number that the design of the study fixes
and that no code constant carries yet: the eta grid and the 28-day P4 window (spec v3 §4.3), the 1 MW position (§4.2), the 48 h shift (§3, I3), the
216 h lookback and the 28-node and 90% coverage rule (§6.1). Those are typed once below (SPEC_*), checked against the code where the code
allows, and printed in plain words with the section of the paper that describes them ("set by the design of the study (§3.7.3)") in
rules()["spec_fixed"] and in the rule text: a reader of the paper has no design specification. tests/es/test_impl_table.py
compares those that es.paper_registry.CODE_CONSTANTS (the source of the macros) also carries. The remaining strings only
name the function that implements each rule. Every "file.py:123" in the text is found in the source when the table is
built (_cite), so it cannot go stale, and the build refuses to run when src.transfer.fit_scaler, fit_model or the XGBoost
factory stop doing what the text says. static_drift compares a stored implementation.json with the code, ignoring the line
numbers of citations (an edit elsewhere in a cited file moves them without changing what the table says).

Static part (fast): hyperparameters, windows, blocks, scaler, label and alert rules, seeds, bands, time zones,
bootstrap pins, fees, versions. Data part (--with-data; reads development-period rows only; seconds to about a minute,
once per data vintage): coverage per market, FMM and NYISO-integrated start dates, duplicates dropped with their split
by market and their timestamps (coverage_notes, and its structured twin coverage_duplicates), and the hours absent from
each series. A static run keeps the data part of the last --with-data run. Writes outputs/es/paper/implementation.json and outputs/es/paper/coverage.csv,
files this generator owns.
Run: DART_PROJECT_ROOT=... /opt/anaconda3/bin/python3 -m es.impl_table [--with-data]"""
from __future__ import annotations

import argparse
import calendar
import importlib.metadata as md
import inspect
import json
import platform
import re
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator
from xgboost import XGBClassifier

from src import transfer
from v2.panel_prep import G1_LAGS, G1_ROLL_WINDOWS, MARKET_TZ, add_features

from . import policies, positions, run_monitors
from . import scores as scores_module
from .analyses import ANALOGS, LABEL_CONTROL, LEGACY, SEALED
from .blocks import BLOCKS, G6, PAPER_BLOCK_NAMES, features
from .config import ALPHA, DATA_ROOT, DEV_ROOT, MARKETS, SEAL_LAST_DEV_DATE, SETTLE_TZ
from .measures import LAMS, STRATA_DECILE, STRATA_PRIMARY
from .paper_paths import PAPER_OUT_DIR, ensure_paper_dirs
from .paper_pins import ALPHAS, BANDS, PINS, SEEDS, signature_default
from .paper_tex import fmt_g
from .positions import CALIBRATED_FEES, UNIFORM_FEES
from .scores import MIN_POSITIVES, logit_factory, xgb_factory
from .windows import UtcWindow

XGB_KEYS = ("n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "reg_lambda",
            "eval_metric", "random_state")
LOGIT_KEYS = ("penalty", "C", "class_weight", "max_iter", "solver")
FMM_SERIES = "CAISO FMM (settlement leg)"
NYI_SERIES = "NYISO integrated RT LBMP (settlement leg)"

# Numbers that the design of the study (spec v3) fixes and that no code constant carries yet (see the module docstring).
SPEC_P4_ALT_DAYS = 28            # §4.3, P4: W = 28 days is the sensitivity window; 56, the primary, is p4_rolling's default
SPEC_P5_ETA_GRID = (0.005, 0.01, 0.02, 0.05)     # §4.3, P5; the development plan's ETAS is not in the code yet
SPEC_POSITION_MW = 1             # §4.2: standing 1 MW virtual positions (es/positions.py has the sentence, not a constant)
SPEC_SHIFT_H = 48                # §3, I3: rolling windows are shifted 48 h; rules() checks it against min(G1_LAGS)
SPEC_LOOKBACK_H = 216            # §6.1: 168 h lags plus the 48 h shift; rules() checks it against the code
SPEC_COVERAGE_MIN_NODES = 28     # §6.1, coverage rule
SPEC_COVERAGE_MIN_SHARE_PCT = 90     # §6.1, node exclusion: at least 90% of expected deployment hours
#: The primary seed is the default of score_analysis's `seed` parameter (es/scores.py), read from the signature.
PRIMARY_SEED = signature_default("scores", "score_analysis", "seed")
#: The four ingesters; each inner-merges the day-ahead and the real-time series (v2/ingest_<market>.py).
INGESTERS = tuple(f"v2/ingest_{market.lower()}.py" for market in MARKETS)


# --------------------------------------------------------------------------------------------
# Citations: a "file.py:123" in the table is found in the source when the table is built.
# --------------------------------------------------------------------------------------------

def _find(where, needle) -> tuple[str, int, str]:
    """(path relative to dart_dev, line number, line) of the one source line that holds ``needle`` (a substring, or a compiled
    regular expression searched in the line). ``where`` is a function (its source is searched) or a Path (the whole file).
    Raises when no line holds it, or when several do."""
    if isinstance(where, Path):
        path, lines, first = where, where.read_text().splitlines(), 1
    else:
        path = Path(inspect.getsourcefile(where))
        lines, first = inspect.getsourcelines(where)
    hits = [(first + i, line) for i, line in enumerate(lines)
            if (needle.search(line) if isinstance(needle, re.Pattern) else needle in line)]
    if not hits:
        raise RuntimeError(f"no line of {where} contains {needle!r}: the code moved, update es.impl_table")
    if len(hits) > 1:
        raise RuntimeError(f"{needle!r} is on {len(hits)} lines of {where}, expected exactly one")
    return str(path.resolve().relative_to(DEV_ROOT)), hits[0][0], hits[0][1]


def _cite(where, needle) -> str:
    path, line, _ = _find(where, needle)
    return f"{path}:{line}"


def _and(nums) -> str:
    items = [str(n) for n in nums]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


# --------------------------------------------------------------------------------------------
# Hyperparameters
# --------------------------------------------------------------------------------------------

def _check_xgb_settings(params: dict) -> None:
    """The table says every parameter but XGB_KEYS, scale_pos_weight and n_jobs is at the xgboost default: check it."""
    default = XGBClassifier().get_params()
    changed = {k for k, v in params.items() if not (v == default.get(k) or (v != v and default.get(k) != default.get(k)))}   # nan == nan
    expected = set(XGB_KEYS) | {"scale_pos_weight", "n_jobs"}
    if changed != expected:
        raise RuntimeError(f"the XGBoost factory differs from the xgboost {md.version('xgboost')} defaults in {sorted(changed)}, but the "
                           f"table describes {sorted(expected)} (not described: {sorted(changed - expected)}; described but at the "
                           f"default: {sorted(expected - changed)}). Report; do not change the pipeline.")


def _check_class_weight() -> None:
    """src.transfer.fit_model passes scale_pos_weight = n_negative / n_positive, counted over the feature-complete training
    rows, as a keyword to the factory. A toy frame whose incomplete rows would change the ratio shows that."""
    seen = {}

    class Recorder:
        def fit(self, X, y):
            seen["rows"] = len(y)

    def factory(**kwargs):
        seen.update(kwargs)
        return Recorder()

    toy = pd.DataFrame({"f": [0.0] * 60 + [1.0] * 30 + [np.nan] * 10, "label": [0] * 60 + [1] * 30 + [1] * 10})
    transfer.fit_model(factory, toy, ["f"], "label")
    if seen.get("scale_pos_weight") != 2.0 or seen.get("rows") != 90:
        raise RuntimeError(f"src.transfer.fit_model passed scale_pos_weight={seen.get('scale_pos_weight')!r} and fitted "
                           f"{seen.get('rows')} rows on a toy frame whose 90 feature-complete rows hold 60 negatives and 30 "
                           "positives (2.0 expected), but the table says n_negative / n_positive of the feature-complete training "
                           "rows. Report; do not change the pipeline.")


def xgb_params(seed: int = PRIMARY_SEED) -> dict:
    params = xgb_factory(seed)(1.0).get_params()         # 1.0 is a placeholder: fit_model passes the node's real weight
    _check_xgb_settings(params)
    _check_class_weight()
    out = {k: params[k] for k in XGB_KEYS}
    out["scale_pos_weight"] = (
        f"n_negative / n_positive of the node's feature-complete training rows "
        f"({_cite(transfer.fit_model, 'model_factory(scale_pos_weight=')}); every other parameter at the xgboost "
        f"{md.version('xgboost')} default except n_jobs = {params['n_jobs']} (the thread count)")
    return out


def logit_params() -> dict:
    p = logit_factory(PRIMARY_SEED)(1.0).get_params()
    return {k: p[k] for k in LOGIT_KEYS}


# --------------------------------------------------------------------------------------------
# The scaler
# --------------------------------------------------------------------------------------------

def _fit_scaler_floor() -> float:
    """The scale below which the fallback applies. src/transfer.py:452 declares it as the default of fit_scaler's
    `minimum_scale` parameter and there is no module-level constant, so it is read from the signature."""
    return inspect.signature(transfer.fit_scaler).parameters["minimum_scale"].default


def calendar_fallback_features() -> list[str]:
    """The G6 calendar features that take the standard-deviation fallback: those whose 1.4826 x MAD is at most the
    threshold over a year of hourly timestamps, which are the 0/1 flags (MAD is 0 unless a flag is split 50/50).
    Raises when the real transfer.fit_scaler gives a listed feature anything but its sample standard deviation, or any
    other feature anything but 1.4826 x MAD."""
    ts = pd.date_range("2023-01-01", "2023-12-31 23:00", freq="h", tz="UTC")
    cal = add_features(pd.DataFrame({"market": "NYISO", "node": "n", "timestamp_utc": ts, "da_lmp": 1.0, "rt_lmp": 1.0}))
    mad = (cal[G6] - cal[G6].median()).abs().median() * 1.4826
    applies = [c for c in G6 if mad[c] <= _fit_scaler_floor()]
    _, spread = transfer.fit_scaler(cal, list(G6))
    for c in G6:
        want = cal[c].std() if c in applies else mad[c]
        if abs(float(spread[c]) - float(want)) > 1e-9:
            raise RuntimeError(f"src.transfer.fit_scaler gives {c} the spread {float(spread[c])}, expected {float(want)} "
                               f"({'standard deviation fallback' if c in applies else '1.4826 x MAD'}). Report; do not change the pipeline.")
    return applies


def scaler_spec() -> dict:
    """Checks at build time that src.transfer.fit_scaler is median / (1.4826 x MAD) on training rows, and that its
    fallback is the one described: src/transfer.py:464-466 take the sample standard deviation where 1.4826 x MAD is at
    most the threshold (:452), then 1.0 where that is undefined or zero."""
    x = pd.DataFrame({"f": [1.0, 2.0, 4.0, 7.0, 11.0, 16.0, 22.0]})
    centre, spread = transfer.fit_scaler(x, ["f"])
    med = float(x["f"].median())
    mad = float((x["f"] - med).abs().median())
    if abs(float(centre["f"]) - med) > 1e-12 or abs(float(spread["f"]) - 1.4826 * mad) > 1e-9:
        raise RuntimeError(f"src.transfer.fit_scaler is not median/1.4826*MAD: centre={float(centre['f'])}, "
                           f"spread={float(spread['f'])}; expected {med}, {1.4826 * mad}. Report; do not change the pipeline.")
    flags = pd.DataFrame({"flag": [0.0] * 8 + [1.0] * 4, "const": [3.0] * 12})            # MAD is 0 for both
    _, fb = transfer.fit_scaler(flags, ["flag", "const"])
    if abs(float(fb["flag"]) - float(flags["flag"].std())) > 1e-12 or float(fb["const"]) != 1.0:
        raise RuntimeError("src.transfer.fit_scaler's fallback is not 'sample standard deviation, then 1.0': "
                           f"flag={float(fb['flag'])} (expected {float(flags['flag'].std())}), const={float(fb['const'])} "
                           "(expected 1.0). Report; do not change the pipeline.")
    floor = f"{_fit_scaler_floor():.0e}".replace("e-0", "e-")                              # 1e-06 -> 1e-6
    return {"centre": "median of the training rows", "spread": "1.4826 x median absolute deviation of the training rows",
            "fallback": (f"training standard deviation (n - 1) when 1.4826 x MAD <= {floor}, then 1.0 where that is "
                         "undefined or zero; in the pipeline it applies to the 0/1 calendar flags "
                         f"{' and '.join(calendar_fallback_features())}"),
            "fit_on": "training rows only, never calibration or deployment rows", "applied_to": "every row of the node"}


# --------------------------------------------------------------------------------------------
# Windows and blocks
# --------------------------------------------------------------------------------------------

def _win(w) -> dict:
    conv = "UTC half-open [start, end)" if isinstance(w, UtcWindow) else "local settlement dates [start, end]"
    return {"start": w.start, "end": w.end, "convention": conv}


def windows() -> list[dict]:
    rows = []
    for group, reg in (("legacy", LEGACY), ("label_control", LABEL_CONTROL), ("analog", ANALOGS), ("sealed", SEALED)):
        for a in reg.values():
            rows.append({"group": group, "id": a.id, "status": a.status, "maintenance": a.maintenance,
                         "train": _win(a.train), "calib": _win(a.calib), "deploy": _win(a.deploy),
                         "label_train": _win(a.label_train)})
    return rows


def blocks() -> dict:
    out = {"calendar_G6": list(G6)}
    for b, f in BLOCKS.items():
        out[b] = {"paper_name": PAPER_BLOCK_NAMES[b], "n_price_features": len(f), "price_features": list(f),
                  "n_model_features": len(features(b))}
    return out


def window_conventions() -> str:
    """Which analyses use which window convention, read from windows(): the label-control analyses are new but use UTC."""
    groups: dict[str, dict[str, int]] = {}
    for r in windows():
        conventions = {r[w]["convention"] for w in ("train", "calib", "deploy", "label_train")}
        if len(conventions) != 1:
            raise RuntimeError(f"analysis {r['id']} mixes window conventions {sorted(conventions)}")
        by_group = groups.setdefault(conventions.pop(), {})
        by_group[r["group"]] = by_group.get(r["group"], 0) + 1
    parts = [f"{convention}: " + ", ".join(f"{g} ({n} {'analysis' if n == 1 else 'analyses'})" for g, n in by_group.items())
             for convention, by_group in groups.items()]
    return "; ".join(parts) + "; the two conventions are never pooled (§3.4)"


# --------------------------------------------------------------------------------------------
# Rules: missing data, policies, fees, monitors, calendar, seeds
# --------------------------------------------------------------------------------------------

def _roll_min_periods(w: int) -> int:
    """v2/panel_prep.py:92, `mp = max(2, w // 2)`, is an inline expression with no constant to read, so it is cited
    here and pinned by tests/es/test_impl_table.py::test_missing_data_rule_matches_add_g1_features, which bisects the
    real add_g1_features and fails if the tolerance changes."""
    return max(2, w // 2)


def _check_spec_numbers() -> None:
    """The spec-fixed shift and lookback must agree with the code's lags and rolling windows (v2.panel_prep)."""
    if SPEC_SHIFT_H != min(G1_LAGS) or SPEC_LOOKBACK_H != max(G1_ROLL_WINDOWS) + SPEC_SHIFT_H:
        raise RuntimeError(f"spec v3 fixes a {SPEC_SHIFT_H} h shift and a {SPEC_LOOKBACK_H} h lookback, but v2.panel_prep has "
                           f"lags {G1_LAGS} and rolling windows {G1_ROLL_WINDOWS}. Report; do not change the pipeline.")


_MERGE_KEYS = re.compile(r"\.merge\(\s*rt\s*,\s*on\s*=\s*\[([^\]]*)\]")
_KEY_WORDS = {"timestamp_utc": "timestamp", "node": "node", "node_type": "node type"}
_MARKET_WORDS = {"ISONE": "ISO-NE"}


def _ingester_merges() -> tuple[str, str]:
    """(citations, wording) of the DA/RT merge of each ingester: an inner merge, so an hour with a price on one side only never
    reaches the panel. The keys are read from the merge lines: those every ingester shares, then, for each other key, the
    markets that add it (node type for ISO-NE and MISO)."""
    cites, keys = [], {}
    for market, path in zip(MARKETS, INGESTERS):
        rel, line, source = _find(DEV_ROOT / path, "da.merge(rt")
        if 'how="inner"' not in source:
            raise RuntimeError(f"{rel}:{line} no longer inner-merges the day-ahead and real-time series: {source.strip()}")
        match = _MERGE_KEYS.search(source)
        if match is None:
            raise RuntimeError(f"{rel}:{line} merges on keys this table cannot read: {source.strip()}")
        keys[market] = [k.strip().strip("'\"") for k in match.group(1).split(",") if k.strip()]
        cites.append(f"{rel}:{line}")
    word = lambda key: _KEY_WORDS.get(key, key.replace("_", " "))                                   # noqa: E731
    shared = [k for k in keys[MARKETS[0]] if all(k in ks for ks in keys.values())]
    extra: dict[str, list[str]] = {}
    for market, ks in keys.items():
        for k in ks:
            if k not in shared:
                extra.setdefault(k, []).append(_MARKET_WORDS.get(market, market))
    wording = f"on {_and(word(k) for k in shared)}"
    if extra:
        wording += " (and " + "; ".join(f"{word(k)} for {_and(markets)}" for k, markets in extra.items()) + ")"
    return ", ".join(cites), wording


def missing_data_rule() -> str:
    """Nothing is imputed. Each ingester inner-merges the DA and RT series (v2/ingest_<market>.py), so a one-sided gap is an
    absent hour. Lags and rolling windows are built on a complete hourly grid per node (v2/panel_prep.py:85), so a lag takes
    exactly the hour h before (:86-89) and a rolling statistic needs max(2, w // 2) observed hours (:92-93); duplicate
    timestamps are dropped per node, keeping the first (:82); rows with an incomplete feature are then dropped."""
    mins = ", ".join(f"{w} h: {_roll_min_periods(w)}" for w in G1_ROLL_WINDOWS)
    cites, on = _ingester_merges()
    return (f"nothing is imputed. Each ingester inner-merges the day-ahead and real-time series {on}, so an hour "
            f"with a price on one side only is absent from the panel ({cites}). "
            f"A lag feature (lags of {_and(G1_LAGS)} h) takes the observation at exactly that hour before "
            "the row on a complete hourly grid, so it is missing when that hour is absent or its price is missing; "
            f"a rolling statistic (windows of {_and(G1_ROLL_WINDOWS)} h, ending {SPEC_SHIFT_H} h before the row) is computed when "
            f"at least max(2, w // 2) of its w hours are observed ({mins}) and is missing otherwise; "
            "each node drops duplicate timestamps, keeping the first (v2.panel_prep.add_g1_features); "
            "a row with any missing feature is then dropped for that block (es.scores._prepare)")


def p5_clip_rule() -> str:
    lo = fmt_g(signature_default("policies", "p5_self_correcting", "lo"), 6)
    hi = fmt_g(signature_default("policies", "p5_self_correcting", "hi"), 6)
    return (f"after each deployment day the alert rate is clipped to [{lo}, {hi}]: alpha_(d+1) = min({hi}, max({lo}, "
            "alpha_d + eta (alpha - r_d))), with alpha_1 = alpha and r_d the day's realized alert rate "
            f"(es.policies.p5_self_correcting, {_cite(policies.p5_self_correcting, 'def p5_self_correcting')})")


def p4_matched_rule() -> str:
    iters = signature_default("policies", "p4_matched", "iters")
    lo, hi = fmt_g(policies._RATE_LO, 6), fmt_g(policies._RATE_HI, 6)
    return (f"P4@r*: per node, a bisection of the P4 alert rate on [{lo}, {hi}] with {iters} iterations, to the target alert count "
            "(that of P1, §3.7.3); es.policies.p4_matched at "
            f"{_cite(policies.p4_matched, 'def p4_matched')}, the bounds at {_cite(Path(policies.__file__), '_RATE_LO, _RATE_HI =')}")


def _stress_window_and_lag() -> tuple[int, int]:
    """The window and the lag, in hours, of the stress statistic. es.positions reads abs_mean_168h, the unscaled rolling feature
    that es.scores names dart_spread_roll_abs_mean_<w>h_shift_<s>h: the trailing w-hour mean of |DART|, lagged s hours."""
    source = scores_module._ABS_MEAN_SOURCE
    match = re.fullmatch(r"dart_spread_roll_abs_mean_(\d+)h_shift_(\d+)h", source)
    if match is None:
        raise RuntimeError(f"es.scores._ABS_MEAN_SOURCE is {source!r}, not of the form dart_spread_roll_abs_mean_<w>h_shift_<s>h: "
                           "update es.impl_table.stress_fee_rule")
    window, lag = int(match.group(1)), int(match.group(2))
    if window not in G1_ROLL_WINDOWS or lag != SPEC_SHIFT_H:
        raise RuntimeError(f"es.scores._ABS_MEAN_SOURCE ({source}) does not match v2.panel_prep: rolling windows {G1_ROLL_WINDOWS} and "
                           f"a shift of {SPEC_SHIFT_H} h. Report; do not change the pipeline.")
    return window, lag


def stress_fee_rule() -> str:
    """Probes es.positions on toy rows: the training quantile of stress_threshold, then the day rule and the fee multiplier
    of attach_pnl (a first hour at the threshold doubles the day, a lower first hour does not, whatever the later hours)."""
    train = pd.DataFrame({"market": "M", "node": "N", "abs_mean_168h": np.arange(101.0)})
    thr = positions.stress_threshold(train)
    level = float(thr.iloc[0])
    quantile = round(level / 100.0, 6)
    day1, day2 = pd.Timestamp("2024-01-10").date(), pd.Timestamp("2024-01-11").date()
    rows = pd.DataFrame({
        "market": "M", "node": "N", "settle_date": [day1] * 3 + [day2] * 3,
        "timestamp_utc": pd.to_datetime(["2024-01-10 05:00", "2024-01-10 06:00", "2024-01-10 07:00",
                                         "2024-01-11 05:00", "2024-01-11 06:00", "2024-01-11 07:00"], utc=True),
        "abs_mean_168h": [level, 0.0, 0.0, level - 5.0, level + 9.0, level + 9.0], "dart_settle": 0.0})
    fee = positions.attach_pnl(rows, "inc", fees=1.0, stress=thr).groupby("settle_date")["fee"].agg(["min", "max"])
    multiplier = float(fee.loc[day1, "min"])
    if not (fee.loc[day1, "max"] == multiplier > 1.0 and fee.loc[day2].tolist() == [1.0, 1.0]):
        raise RuntimeError("es.positions no longer doubles the fee of every hour of a day whose first hour is at or above the "
                           f"threshold and of no other day: fees per day {fee.to_dict('index')}. Report; do not change the pipeline.")
    window, lag = _stress_window_and_lag()
    return (f"fees double on stressed days: a node-day is stressed when the trailing {window}-hour mean of |DART|, lagged {lag} h, "
            f"at its first hour is at or above the node's {fmt_g(100 * quantile, 6)}th percentile over the training rows, and the fee is then multiplied by {fmt_g(multiplier, 6)} "
            f"in every hour of that day (es.positions.stress_threshold, {_cite(positions.stress_threshold, 'def stress_threshold')}; "
            f"_stressed_days, {_cite(positions._stressed_days, 'def _stressed_days')}; "
            f"attach_pnl, {_cite(positions.attach_pnl, 'def attach_pnl')})")


def _splitter_name() -> str:
    """The cross-validation splitter class es.run_monitors uses, found by its base class among the module's names."""
    found = [name for name, obj in vars(run_monitors).items() if isinstance(obj, type) and issubclass(obj, BaseCrossValidator)]
    if len(found) != 1:
        raise RuntimeError(f"es.run_monitors names {len(found)} cross-validation splitters ({found}), expected exactly one: "
                           "update es.impl_table")
    return found[0]


def _toy_node() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """A calibration window of 30 settlement days and a deployment window of MONITOR_DAYS days, four hourly rows a day, two features."""
    rng = np.random.default_rng(0)

    def window(start: str, days: int) -> pd.DataFrame:
        return pd.DataFrame({"settle_date": np.repeat(pd.date_range(start, periods=days, freq="D").date, 4),
                             "f1": rng.normal(size=4 * days), "f2": rng.normal(size=4 * days)})

    return window("2024-01-01", 30), window("2024-02-01", run_monitors.MONITOR_DAYS), ["f1", "f2"]


def _run_recorded_domain_classifier() -> tuple[dict, dict]:
    """Runs run_monitors.input_statistics on a toy node with the splitter and the classifier recorded: the arguments they are built
    with, and the labels, groups and folds of the split. Returns (seen, meta); _check_domain_record turns them into facts."""
    name = _splitter_name()
    splitter_class, classifier_class = getattr(run_monitors, name), run_monitors.LogisticRegression
    cal, dep, feats = _toy_node()
    seen: dict = {"classifiers": []}

    class Recorded:
        def __init__(self, inner):
            self.inner = inner

        def split(self, X, y=None, groups=None):
            folds = list(self.inner.split(X, y, groups))
            seen.update(y=np.asarray(y), groups=np.asarray(groups), folds=folds)
            return iter(folds)

    def splitter(*args, **kwargs):
        seen["splitter_args"], seen["splitter_kwargs"] = args, kwargs
        return Recorded(splitter_class(*args, **kwargs))

    def classifier(*args, **kwargs):
        seen["classifiers"].append((args, kwargs))
        return classifier_class(*args, **kwargs)

    with mock.patch.object(run_monitors, name, splitter), mock.patch.object(run_monitors, "LogisticRegression", classifier):
        run_monitors.input_statistics(cal, dep, feats)
    return seen, {"name": name, "cal_dates": list(cal["settle_date"]), "dep_dates": list(dep["settle_date"])}


def _check_domain_record(seen: dict, meta: dict) -> dict:
    """Checks what the recorded run shows against what the table says of the domain classifier: N_FOLDS folds of an unshuffled
    splitter, one CV group per local settlement day of each side, both classes in the training and test part of every fold,
    no group in both parts, and N_FOLDS identical classifiers. Returns the facts the text uses."""
    n_folds, name = run_monitors.N_FOLDS, meta["name"]
    if "folds" not in seen:
        raise RuntimeError(f"run_monitors.input_statistics never split the data with {name}: update es.impl_table")
    if seen["splitter_args"]:
        raise RuntimeError(f"run_monitors builds {name} with positional arguments {seen['splitter_args']}: update es.impl_table")
    kwargs = seen["splitter_kwargs"]
    if kwargs.get("n_splits") != n_folds:
        raise RuntimeError(f"run_monitors builds {name} with n_splits={kwargs.get('n_splits')!r}, but N_FOLDS is {n_folds}")
    if kwargs.get("shuffle", False) is not False:
        raise RuntimeError(f"run_monitors builds {name} with shuffle={kwargs['shuffle']!r}, but the table says it never shuffles")
    folds, y, groups = seen["folds"], seen["y"], seen["groups"]
    if len(folds) != n_folds:
        raise RuntimeError(f"{name} split the toy node into {len(folds)} folds, not N_FOLDS = {n_folds}")
    rows = pd.DataFrame({"side": ["cal"] * len(meta["cal_dates"]) + ["dep"] * len(meta["dep_dates"]),
                         "date": meta["cal_dates"] + meta["dep_dates"], "group": groups})
    days = rows[["side", "date"]].drop_duplicates()
    if len(rows[["side", "date", "group"]].drop_duplicates()) != len(days) or rows["group"].nunique() != len(days):
        raise RuntimeError("the CV groups of run_monitors are not one group per local settlement day of each side: "
                           f"{rows['group'].nunique()} groups for {len(days)} calibration and deployment days")
    for k, (train, test) in enumerate(folds):
        if any(len(set(y[part])) < 2 for part in (train, test)):
            raise RuntimeError(f"fold {k} of {name} has a training or test part without both classes, so the CV is not "
                               "class-stratified")
        if set(groups[train]) & set(groups[test]):
            raise RuntimeError(f"fold {k} of {name} puts a group in both its training and its test part, so the CV is not "
                               "blocked by day")
    classifiers = seen["classifiers"]
    if len(classifiers) != n_folds or any(c != classifiers[0] for c in classifiers) or classifiers[0][0]:
        raise RuntimeError(f"run_monitors.domain_auc fitted {len(classifiers)} classifiers with arguments {classifiers}, expected "
                           f"{n_folds} identical ones with keyword arguments only. Report; do not change the pipeline.")
    return {"splitter": name, "n_splits": kwargs["n_splits"], "classifier": dict(classifiers[0][1])}


def _check_cv_groups() -> None:
    """A calibration day and a deployment day with the same date must not share a CV group (cv_groups shifts one side)."""
    shared = run_monitors.cv_groups(np.array([7, 7]), 1)               # one calibration and one deployment row of the same day id
    if shared[0] == shared[1]:
        raise RuntimeError("run_monitors.cv_groups lets a calibration day and a deployment day share a group")


def _check_domain_guards() -> None:
    """domain_auc refuses what the text says it refuses: fewer than N_FOLDS distinct day groups, and a layout in which a fold
    would hold one class (a single deployment day)."""
    rng = np.random.default_rng(0)
    n = run_monitors.N_FOLDS
    layouts = ((f"fewer than {n} distinct day groups", np.repeat(np.arange(n // 2), 3), np.repeat(np.arange(n - 1 - n // 2), 3)),
               ("a layout with one deployment day", np.repeat(np.arange(n + 1), 3), np.repeat([0], 2 * n)))
    for what, cal_ids, dep_ids in layouts:
        try:
            run_monitors.domain_auc(rng.normal(size=(len(cal_ids), 2)), rng.normal(size=(len(dep_ids), 2)), np.r_[cal_ids, dep_ids])
        except ValueError:
            continue
        raise RuntimeError(f"run_monitors.domain_auc did not refuse {what}, but the table says it does. "
                           "Report; do not change the pipeline.")


def _check_placebo() -> list[str]:
    """The placebo columns of run_monitors: each mirrors a monitor without being one and is a row column, the evaluation reports
    their median (placebo_median), and placebo_split takes the last MONITOR_DAYS calibration days against the earlier ones.
    Returns their names."""
    for monitor, column in run_monitors.PLACEBO_COLUMNS.items():
        if monitor not in run_monitors.MONITORS:
            raise RuntimeError(f"the placebo column {column} mirrors {monitor}, which is not in MONITORS")
        if column in run_monitors.MONITORS:
            raise RuntimeError(f"the placebo column {column} is a monitor (it is in MONITORS), but the table calls it a reference")
        if column not in run_monitors.ROW_COLUMNS:
            raise RuntimeError(f"the placebo column {column} is not in ROW_COLUMNS")
    if "placebo_median" not in run_monitors.EVAL_COLUMNS:
        raise RuntimeError("placebo_median is not in EVAL_COLUMNS, but the table says the evaluation reports the placebo median")
    days = run_monitors.MONITOR_DAYS
    cal = pd.DataFrame({"settle_date": np.repeat(pd.date_range("2024-01-01", periods=days + run_monitors.N_FOLDS + 10,
                                                               freq="D").date, 2)})
    earlier, last = run_monitors.placebo_split(cal)
    if (last["settle_date"].nunique() != days or earlier["settle_date"].nunique() != cal["settle_date"].nunique() - days
            or earlier["settle_date"].max() >= last["settle_date"].min()):
        raise RuntimeError(f"run_monitors.placebo_split no longer takes the last {days} calibration days against the earlier ones")
    return list(run_monitors.PLACEBO_COLUMNS.values())


_MONITOR_DAYS_LINE = re.compile(r"^MONITOR_DAYS\b")        # the definition in es/run_monitors.py (a comment above it does not match)


def monitor_days() -> int:
    """The monitoring window (controller Ruling 66b): every monitor sees deployment days 1 to MONITOR_DAYS, and the outcome they
    are scored against comes from the days after. Read from es.run_monitors.MONITOR_DAYS with no fallback: without the constant
    the table cannot say what the monitors do."""
    days = getattr(run_monitors, "MONITOR_DAYS", None)
    if not isinstance(days, int) or isinstance(days, bool) or days < 1:
        raise RuntimeError(f"es.run_monitors.MONITOR_DAYS is {'missing' if days is None else repr(days)}, but the monitors entry of the "
                           "implementation table states the monitoring window from it (controller Ruling 66b): es/run_monitors.py "
                           "must define it as a positive integer (Task 14). Report; do not type a default here.")
    return days


def _check_early_rate() -> None:
    """early_rate covers deployment days 1 to MONITOR_DAYS by default. Probed on a toy node rather than read off the signature (its
    default is None and it reads MONITOR_DAYS when it is called): an alert on the last of those days counts, one on the next does not."""
    days = run_monitors.MONITOR_DAYS
    dates = pd.date_range("2024-01-01", periods=days + 2, freq="D").date

    def rate(alert_day: int) -> float:
        alerts = pd.DataFrame({"market": "M", "node": "N", "settle_date": dates,
                               "alert": [1.0 if i == alert_day else 0.0 for i in range(len(dates))]})
        return float(run_monitors.early_rate(alerts).iloc[0])

    last_early, first_late = rate(days - 1), rate(days)
    if abs(last_early - 1.0 / days) > 1e-12 or first_late != 0.0:
        raise RuntimeError(f"run_monitors.early_rate does not cover exactly the first {days} deployment days (MONITOR_DAYS): on a toy "
                           f"node an alert on day {days} gave the rate {last_early} (expected {1.0 / days}) and one on day {days + 1} "
                           f"gave {first_late} (expected 0.0). Report; do not change the pipeline.")


def monitor_rule() -> str:
    days = monitor_days()
    folds = run_monitors.N_FOLDS
    bins = signature_default("run_monitors", "psi", "bins")
    _check_early_rate()
    monitors = list(run_monitors.MONITORS)
    if "early_rate" not in monitors:
        raise RuntimeError(f"es.run_monitors.MONITORS is {monitors}, which has no early_rate: update es.impl_table.monitor_rule")
    inputs = [m for m in monitors if m != "early_rate"]
    seen, meta = _run_recorded_domain_classifier()
    domain = _check_domain_record(seen, meta)
    _check_cv_groups()
    _check_domain_guards()
    placebo = _check_placebo()
    classifier = ", ".join(f"{k} = {v}" for k, v in domain["classifier"].items())
    here = Path(run_monitors.__file__)
    return (f"the monitors are {_and(monitors)}; every monitor sees only deployment days 1 to {days}: {_and(inputs)} compare the "
            f"calibration rows with the deployment rows of those days, and early_rate is the alert rate over those days; the outcome "
            f"they are scored against, |log10 R| and the failure flag, comes from deployment days {days + 1} to the end. The domain "
            f"classifier is a logistic regression ({classifier}) scored by out-of-fold AUROC over {domain['n_splits']}-fold "
            f"{domain['splitter']} CV, which is class-stratified and blocked by local settlement day (every day stays whole in one "
            f"fold, no shuffling; at least {folds} distinct day groups are required, and so are both classes in the training and test "
            f"part of every fold). PSI uses {bins} reference-quantile bins. The descriptive placebo reference columns "
            f"{_and(placebo)} repeat the three input statistics on "
            f"a split inside the calibration window, its last {days} days against the earlier ones; they are not monitors (they are "
            f"not in MONITORS), and the evaluation reports their median per analysis, block and monitor as placebo_median "
            f"(es.run_monitors: MONITOR_DAYS, {_cite(here, _MONITOR_DAYS_LINE)}; N_FOLDS, {_cite(here, 'N_FOLDS = ')}; "
            f"domain_auc, {_cite(run_monitors.domain_auc, 'class_weight=')}; "
            f"placebo_split, {_cite(run_monitors.placebo_split, 'def placebo_split')})")


def _flag_keys(flag: pd.Series, key) -> list[int]:
    """The values of ``key`` at which the 0/1 ``flag`` is 1, after checking that the flag is a function of the key."""
    by = flag.groupby(np.asarray(key)).agg(["min", "max"])
    if (by["min"] != by["max"]).any():
        raise RuntimeError(f"{flag.name} is not a function of one clock field, so the table cannot name its values")
    return [int(k) for k in by.index[by["max"] == 1]]


def _period(cal: pd.DataFrame, name: str, values) -> float:
    """The period P with which v2.panel_prep.add_features encodes a clock field as sin and cos of 2 pi x / P, solved at x = 1
    and then checked on every row."""
    values = np.asarray(values)
    sin, cos = cal[f"calendar_{name}_sin"].to_numpy(), cal[f"calendar_{name}_cos"].to_numpy()
    at_one = values == 1
    period = round(float(2 * np.pi / np.arctan2(sin[at_one], cos[at_one]).mean()), 2)
    if not (np.allclose(np.sin(2 * np.pi * values / period), sin) and np.allclose(np.cos(2 * np.pi * values / period), cos)):
        raise RuntimeError(f"calendar_{name} is not the sin and cos of 2 pi x / {period}. Report; do not change the pipeline.")
    return period


def calendar_rule() -> str:
    """The calendar features of v2.panel_prep.add_features, read from a synthetic hourly year in the New York clock."""
    apart = [m for m in MARKETS if MARKET_TZ[m] != SETTLE_TZ[m]]
    if apart:
        raise RuntimeError(f"v2.panel_prep.MARKET_TZ differs from es.config.SETTLE_TZ for {apart}")
    ts = pd.date_range("2023-01-01", "2023-12-31 23:00", freq="h", tz="UTC")
    cal = add_features(pd.DataFrame({"market": "NYISO", "node": "n", "timestamp_utc": ts, "da_lmp": 1.0, "rt_lmp": 1.0}))
    local = ts.tz_convert(MARKET_TZ["NYISO"])
    peak = _flag_keys(cal["calendar_is_peak_hour"], local.hour)
    weekend = _flag_keys(cal["calendar_is_weekend"], local.dayofweek)
    if peak != list(range(peak[0], peak[-1] + 1)):
        raise RuntimeError(f"the peak hours {peak} are not a contiguous range")
    periods = {name: _period(cal, name, values) for name, values in (
        ("hour", local.hour), ("day_of_week", local.dayofweek), ("day_of_year", local.dayofyear), ("month", local.month))}
    return (f"the calendar flags and encodings use the market-local clock, which is the settlement clock; peak hour = local hours "
            f"{peak[0]} to {peak[-1]}; weekend = {_and(calendar.day_name[d] for d in weekend)}; each cyclic feature is the sin and "
            f"cos of 2 pi x / period, with periods hour {fmt_g(periods['hour'], 6)}, day of week {fmt_g(periods['day_of_week'], 6)}, "
            f"day of year {fmt_g(periods['day_of_year'], 6)}, month {fmt_g(periods['month'], 6)} "
            f"(v2.panel_prep.add_features, {_cite(add_features, 'def add_features')})")


def rotation_reference_set() -> str:
    """The reference set of the H-E rotation test under rotation_test's default shift limit (Rulings 55 and 69), probed on toy
    windows: a window of D days has D cells (the shifts 0, 1, ..., D-1, shift 0 being the observed state), and several windows
    have the grid of their shift sets, so n_null is the product of the D minus 1 and a single window's p is a multiple of 1/D."""
    from . import bootstrap

    def window(days: int):
        n = 2 * days                                                  # one stratum, two hourly cells a day, a third of them alerted
        rng = np.random.default_rng(days)
        return bootstrap.RowCube.from_arrays(np.repeat(np.arange(days), 2), days, np.tile([0, 1], days), np.zeros(n, dtype=int),
                                             rng.normal(size=n), (np.arange(n) % 3 == 0).astype(float))

    d1, d2 = 5, 7
    one, both = bootstrap.rotation_test([window(d1)]), bootstrap.rotation_test([window(d1), window(d2)])
    if one["n_null"] != d1 - 1 or both["n_null"] != d1 * d2 - 1 or abs(one["p"] * d1 - round(one["p"] * d1)) > 1e-9:
        raise RuntimeError(f"es.bootstrap.rotation_test's default reference set is not the full cyclic group: a window of {d1} days "
                           f"gave n_null = {one['n_null']} (expected {d1 - 1}) and windows of {d1} and {d2} days gave "
                           f"{both['n_null']} (expected {d1 * d2 - 1}). Report; do not change the pipeline.")
    return ("the H-E reference set is the full cyclic group: each window is rotated by every shift 0, 1, ..., D-1 (D its number of "
            "days, shift 0 the observed state), independently across windows, so the reference set is the grid of those shifts and "
            f"n_null = (product of the windows' D) - 1 (es.bootstrap.rotation_test, {_cite(bootstrap.rotation_test, 'def rotation_test')})")


def seed_scope() -> str:
    """Which runs use the extra seeds, read from es.run_legacy.run_plan (a call-time import: run_legacy is heavy and other
    tasks edit it, so importing this module never imports it)."""
    from .run_legacy import run_plan
    plan = list(run_plan())
    extra = [(a.id, block, est, seed, variants) for a, block, est, seed, variants in plan if seed != PRIMARY_SEED]
    if sorted({run[3] for run in plan}) != sorted(SEEDS):
        raise RuntimeError(f"run_plan uses the seeds {sorted({run[3] for run in plan})}, but es.paper_pins.SEEDS is {sorted(SEEDS)}")
    seeds = sorted({run[3] for run in extra})
    analyses = list(dict.fromkeys(run[0] for run in extra))
    blocks_ = list(dict.fromkeys(run[1] for run in extra))
    estimators = {run[2] for run in extra}
    variants = {tuple(run[4]) for run in extra}
    if (len(extra) != len(analyses) * len(blocks_) * len(seeds) or len(estimators) != 1 or len(variants) != 1
            or len(next(iter(variants))) != 1):
        raise RuntimeError("the extra-seed runs of run_plan are not one estimator and one policy over a full grid of "
                           "analyses, blocks and seeds, so the table cannot summarise them in one sentence")
    ((policy, alpha),) = next(iter(variants))
    estimator = {"xgb": "XGBoost", "logit": "logistic regression"}[next(iter(estimators))]
    return (f"the {len(seeds)} extra seeds ({', '.join(map(str, seeds))}) run only for the seed-sensitivity blocks "
            f"({', '.join(blocks_)}) of {len(analyses)} analyses ({', '.join(analyses)}), {estimator}, {policy} at alpha "
            f"{fmt_g(alpha, 6)}; every other run uses the primary seed {PRIMARY_SEED} "
            f"(es.run_legacy.run_plan, {_cite(run_plan, '# D8: seeds')})")


def rules() -> dict:
    _check_spec_numbers()
    return {
        "label": (f"spike = |DART| >= q, with q the node's {100 * (1 - ALPHA):g}th percentile of |DART| on the "
                  "analysis's label-training rows (linear interpolation); NaN |DART| stays unlabelled (es.labels.label_training_q)"),
        "cutoff_P1": "tau = numpy.quantile(calibration scores, 1 - alpha), linear interpolation; alert when score >= tau (es.policies.p1_fixed)",
        "alpha": ALPHA, "alpha_sweep": list(ALPHAS), "failure_bands": {k: list(v) for k, v in BANDS.items()},
        "zero_alert_floor": "R floored at 1/(2 n alpha) for log statistics; zero-activity cells counted as failures (es.policies.activity)",
        "feasibility": f"a node is fitted only with at least {MIN_POSITIVES} positive training labels and two classes (es.scores._gate)",
        "missing_data": missing_data_rule(),
        "deduplication": ("drop_duplicates on (market, node, timestamp_utc), keep first (es.data.load_dev_raw); the count is in "
                          "coverage.csv and its split by market in coverage_notes"),
        "information_policy": (f"realized lags {_and(G1_LAGS)} h (v2.panel_prep.G1_LAGS); rolling windows of "
                               f"{_and(G1_ROLL_WINDOWS)} h (G1_ROLL_WINDOWS) end at target minus {SPEC_SHIFT_H} h (set by the information "
                               "policy, which keeps every input known before the day-ahead bidding deadline, §3.3); "
                               f"feature lookback {SPEC_LOOKBACK_H} h = {max(G1_ROLL_WINDOWS)} h + {SPEC_SHIFT_H} h (the longest rolling "
                               "window plus the shift, §3.3; v2.panel_prep.add_g1_features)"),
        "calendar": calendar_rule(),
        "seeds": {"primary": PRIMARY_SEED, "all": list(SEEDS), "extra_seeds_scope": seed_scope()},
        "time_zones": dict(SETTLE_TZ), "miso_note": "MISO settles in fixed EST (Etc/GMT+5)",
        "window_conventions": window_conventions(),
        "dst": "spring-forward days have 23 hours; on fall-back days CAISO keeps both 01:00 rows, NYISO and ISO-NE drop both (§6.1.1)",
        "sealed_last_dev_date": SEAL_LAST_DEV_DATE,
        "P4_windows_days": [PINS["pFourWindow"], SPEC_P4_ALT_DAYS], "P5_eta_grid": list(SPEC_P5_ETA_GRID),
        "P5_clip": p5_clip_rule(), "P4_matched": p4_matched_rule(),
        "bootstrap": {"replicates": PINS["bootB"], "mean_block_days": PINS["bootBlockDays"], "seed": PINS["bootSeed"],
                      "rotation_min_shift_days": PINS["rotMinShiftDays"],
                      "rotation_reference_set": rotation_reference_set()},
        "es_fraction_pct": PINS["esFrac"],
        "fees_calibrated": {m: {"INC": v[1], "DEC": v[-1]} for m, v in CALIBRATED_FEES.items()},
        "fees_uniform_grid": list(UNIFORM_FEES), "position_mw": SPEC_POSITION_MW,
        "stress_fee": stress_fee_rule(),
        "lambda_grid": list(LAMS), "strata_primary": list(STRATA_PRIMARY), "strata_sensitivity": list(STRATA_DECILE),
        "monitors": monitor_rule(),
        "coverage_rule": (f"a sealed test runs only with at least {SPEC_COVERAGE_MIN_NODES} usable nodes and all four operators; a node "
                          f"needs at least {SPEC_COVERAGE_MIN_SHARE_PCT}% of expected deployment hours (set before the sealed windows "
                          "were opened, §6.1.1)"),
        # Numbers that no code constant carries yet (module docstring, item 4), each with the section that fixes it.
        "spec_fixed": {
            "P4_windows_days": (f"the {SPEC_P4_ALT_DAYS}-day sensitivity window is set by the design of the study (§3.7.3); the "
                                f"{PINS['pFourWindow']}-day primary is the default of es.policies.p4_rolling"),
            "P5_eta_grid": ("the grid from which the step size is chosen on the quarterly analogues (§3.7.3); "
                            "es.policies.p5_self_correcting takes eta as an argument and no code constant holds the grid yet"),
            "position_mw": "the standing position of the economic design (§3.7.1); es/positions.py states it in its docstring, not as a constant",
            "information_shift_h": f"{SPEC_SHIFT_H} h rolling shift, set by the information policy (§3.3); checked against min(v2.panel_prep.G1_LAGS)",
            "feature_lookback_h": (f"{SPEC_LOOKBACK_H} h = {max(G1_ROLL_WINDOWS)} h + {SPEC_SHIFT_H} h, the longest rolling window plus the "
                                   "shift (§3.3); checked against v2.panel_prep.G1_ROLL_WINDOWS"),
            "coverage_rule": (f"{SPEC_COVERAGE_MIN_NODES} usable nodes and {SPEC_COVERAGE_MIN_SHARE_PCT}% of expected deployment "
                              "hours, set before the sealed windows were opened (§6.1.1)"),
        },
    }


def versions() -> dict:
    out = {"python": platform.python_version()}
    for pkg in ("numpy", "pandas", "xgboost", "scikit-learn", "pyarrow", "scipy", "statsmodels", "matplotlib"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = "not installed"
    return out


def collect_static() -> dict:
    return {"xgboost": xgb_params(), "logistic": logit_params(), "scaler": scaler_spec(), "windows": windows(),
            "blocks": blocks(), "rules": rules(), "versions": versions()}


_LINE_REFERENCE = re.compile(r"(\.py):\d+(?:-\d+)?")


def mask_line_references(obj):
    """``obj`` (JSON-like) with the line number of every 'file.py:123' or 'file.py:123-130' in its strings replaced by '#'. An edit
    elsewhere in a cited file moves the numbers without changing what the table says; the file paths and all other text stay."""
    if isinstance(obj, str):
        return _LINE_REFERENCE.sub(r"\1:#", obj)
    if isinstance(obj, dict):
        return {k: mask_line_references(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [mask_line_references(v) for v in obj]
    return obj


def _differences(a, b, path: str = "") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for key in sorted(set(a) | set(b), key=str):
            where = f"{path}.{key}" if path else str(key)
            out += [where] if (key not in a or key not in b) else _differences(a[key], b[key], where)
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in _differences(x, y, f"{path}.{i}" if path else str(i))]
    return [] if a == b else [path]


def static_drift(stored: dict, fresh: dict | None = None) -> list[str]:
    """The dotted paths at which the static part of a stored implementation.json differs from a fresh collect_static() (or from
    ``fresh``), with the line numbers of citations masked; [] when the stored file still says what the code does. The data part
    (coverage, coverage_notes) is not compared: only --with-data refreshes it."""
    if fresh is None:
        fresh = json.loads(json.dumps(collect_static(), default=str))
    return _differences(mask_line_references({k: stored.get(k) for k in fresh}), mask_line_references(fresh))


# --------------------------------------------------------------------------------------------
# Coverage
# --------------------------------------------------------------------------------------------

def _series_row(name: str, nodes, rows, ts: pd.Series, tz: str) -> dict:
    ts = pd.to_datetime(ts, utc=True)
    local = ts.dt.tz_convert(tz).dt.date
    return {"series": name, "nodes": int(nodes), "hourly_rows": int(rows),
            "first_local_date": str(local.min()) if len(local) else "", "last_local_date": str(local.max()) if len(local) else "",
            "first_utc": str(ts.min()) if len(ts) else "", "last_utc": str(ts.max()) if len(ts) else ""}


def load_dev_raw_with_dropped():
    """(raw, dropped): the frame es.data.load_dev_raw returns and the rows it dropped as duplicates (market, node,
    timestamp_utc). load_dev_raw does not return the frame it deduplicates and es/data.py is not this module's to change, so
    the frame is taken where the loader passes it on, as the result of restrict_to_node_map, the last step before
    drop_duplicates. The loader read it through its own sealed-window filter, so nothing is read here, and its count must
    equal the one the loader reports."""
    from . import data
    taps: list[pd.DataFrame] = []
    real = data.restrict_to_node_map

    def tap(frame):
        out = real(frame)
        taps.append(out[["market", "node", "timestamp_utc"]].copy())
        return out

    with mock.patch.object(data, "restrict_to_node_map", tap):
        raw = data.load_dev_raw()
    if len(taps) != 1:
        raise RuntimeError(f"es.data.load_dev_raw called restrict_to_node_map {len(taps)} times, not once before deduplicating: "
                           "update es.impl_table.load_dev_raw_with_dropped")
    pre = taps[0]
    dropped = pre[pre.duplicated(["market", "node", "timestamp_utc"], keep="first")]
    if len(dropped) != raw.attrs["duplicates_dropped"] or len(pre) - len(dropped) != len(raw):
        raise RuntimeError(f"the frame before deduplication holds {len(dropped)} duplicate rows ({len(pre)} rows, {len(raw)} kept), but "
                           f"es.data.load_dev_raw reports {raw.attrs['duplicates_dropped']} dropped")
    return raw, dropped


def _month_start(dropped: pd.DataFrame) -> pd.Series:
    """True for the dropped rows at 00:00 local on the 1st of a month (local: the settlement clock of the row's market)."""
    out = pd.Series(False, index=dropped.index)
    for market in MARKETS:
        sel = (dropped["market"] == market).to_numpy()
        if sel.any():
            local = dropped.loc[sel, "timestamp_utc"].dt.tz_convert(SETTLE_TZ[market])
            out.loc[sel] = ((local.dt.hour == 0) & (local.dt.day == 1)).to_numpy()
    return out


def summarize_dropped(dropped: pd.DataFrame) -> dict:
    """Per market: rows dropped, at how many distinct hours and nodes, how many at 00:00 local on the 1st of a month, and at how
    many distinct such month starts."""
    start = _month_start(dropped)
    out = {}
    for market in MARKETS:
        sel = (dropped["market"] == market).to_numpy()
        d = dropped[sel]
        out[market] = {"rows": int(sel.sum()), "hours": int(d["timestamp_utc"].nunique()), "nodes": int(d["node"].nunique()),
                       "month_start_local": int(start[sel].sum()),
                       "month_boundaries": int(d.loc[start[sel], "timestamp_utc"].nunique())}
    return out


def duplicates_field(dropped: pd.DataFrame) -> dict:
    """The structured twin of the duplicates note (Ruling 76a), for implementation.json["coverage_duplicates"]: the rows dropped,
    how many sit at 00:00 local on the 1st of a month, at how many distinct month starts, in how many nodes, in which markets,
    and the same per market. Derived from the frame the loader deduplicates, like the note."""
    by_market = summarize_dropped(dropped)
    start = _month_start(dropped)
    return {"rows": int(len(dropped)), "at_month_start": int(start.sum()),
            "month_boundaries": int(dropped.loc[start, "timestamp_utc"].nunique()),
            "nodes": int(len(dropped[["market", "node"]].drop_duplicates())),
            "markets": [m for m in MARKETS if by_market[m]["rows"]], "by_market": by_market}


def _duplicates_field_of(raw: pd.DataFrame) -> dict:
    """raw.attrs' structured field, or, when the rows were not examined, the total with null values (never a sentinel)."""
    total = int(raw.attrs["duplicates_dropped"])
    field = raw.attrs.get("duplicates_dropped_field")
    if field is None:
        return {"rows": total, "at_month_start": None, "month_boundaries": None, "nodes": None, "markets": None, "by_market": None}
    if field["rows"] != total:
        raise ValueError(f"the duplicates field counts {field['rows']} rows, which does not match the {total} that "
                         "es.data.load_dev_raw reports: the frame and the field come from different loads")
    return field


def _duplicates_note(raw: pd.DataFrame) -> str:
    total = int(raw.attrs["duplicates_dropped"])
    where = raw.attrs.get("duplicates_dropped_where")
    if where is None:
        return (f"Duplicates: {total} duplicate rows dropped by es.data.load_dev_raw (key market, node, timestamp_utc), first copy "
                "kept; the split by market and the timestamps were not examined (a review of the earlier version of this work found them at "
                "NYISO month boundaries).")
    split = {m: w["rows"] for m, w in where.items()}
    if sum(split.values()) != total:
        raise ValueError(f"the per-market duplicate counts {split} do not add up to the total {total} that "
                         "es.data.load_dev_raw reports: the frame and the split come from different loads")
    counts = ", ".join(f"{m} {where[m]['rows']}" for m in MARKETS)
    hit = [m for m in MARKETS if where[m]["rows"]]
    if not hit:
        return "Duplicates: es.data.load_dev_raw dropped no duplicate rows."
    at_start = sum(where[m]["month_start_local"] for m in hit)
    if at_start != total:
        return (f"Duplicates: {total} duplicate rows ({counts}), {at_start} of them at 00:00 local on the 1st of a month. The first "
                "copy is kept.")
    if hit == ["NYISO"]:
        w = where["NYISO"]
        return (f"Duplicates: {total} duplicate rows, all at NYISO month boundaries (each at 00:00 local on the 1st of a month: "
                f"{w['hours']} month start{'s' if w['hours'] != 1 else ''}, {w['nodes']} node{'s' if w['nodes'] != 1 else ''}); "
                f"per-market counts from es.data.load_dev_raw: {counts}. The first copy is kept.")
    return (f"Duplicates: {total} duplicate rows, all at 00:00 local on the 1st of a month; per-market counts from "
            f"es.data.load_dev_raw: {counts}. The first copy is kept.")


def _series_gaps(frame: pd.DataFrame, tz: str) -> dict:
    """Per node, the hours absent against a continuous hourly UTC index from the series' first to its last timestamp, and
    the repeated local hours inside that span (the two UTC hours that share one wall-clock time on a fall-back day)."""
    ts = pd.to_datetime(frame["timestamp_utc"], utc=True)
    if (ts != ts.dt.floor("h")).any():
        raise ValueError("timestamps are not on the hour, so they cannot be compared with a continuous hourly index")
    full = pd.date_range(ts.min(), ts.max(), freq="h")
    repeated = full[full.tz_convert(tz).tz_localize(None).duplicated(keep=False)]
    missing = {node: full.difference(pd.DatetimeIndex(g)) for node, g in ts.groupby(frame["node"], sort=True)}
    return {"tz": tz, "full": full, "repeated": repeated, "missing": missing}


def _hour_label(t: pd.Timestamp) -> str:
    return t.strftime("%Y-%m-%d %H:%MZ")


def _runs(hours: pd.DatetimeIndex) -> list[pd.DatetimeIndex]:
    """Maximal runs of consecutive hours in a sorted DatetimeIndex."""
    if len(hours) == 0:
        return []
    breaks = np.flatnonzero((hours[1:] - hours[:-1]) != pd.Timedelta(hours=1)) + 1
    return [hours[idx[0]:idx[-1] + 1] for idx in np.split(np.arange(len(hours)), breaks)]


def _run_label(run: pd.DatetimeIndex) -> str:
    first, last = run[0], run[-1]
    if len(run) == 1:
        return _hour_label(first)
    if first.date() == last.date():
        return f"{_hour_label(first)} to {last.strftime('%H:%MZ')}"
    return f"{_hour_label(first)} to {_hour_label(last)}"


def _local_day_hours(day, tz: str) -> int:
    """Hours in a local settlement day: 24, or 23 and 25 on the DST days."""
    start = pd.Timestamp(day).tz_localize(tz)
    return len(pd.date_range(start, (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize(tz), freq="h", inclusive="left"))


def _fallback_note(label: str, gaps: dict, july_downloader: bool):
    """Are the rows of the repeated local hour on fall-back days present or absent? None when the series has none."""
    repeated = gaps["repeated"]
    if len(repeated) == 0:
        return None
    local = repeated.tz_convert(gaps["tz"])
    dates = sorted({d.isoformat() for d in local.date})
    clock = " and ".join(f"{h:02d}:00" for h in sorted(set(local.hour)))
    days = (f"the fall-back day ({dates[0]})" if len(dates) == 1
            else f"each of the {len(dates)} fall-back days ({', '.join(dates)})")
    counts = [len(m.intersection(repeated)) for m in gaps["missing"].values()]
    lo, hi = min(counts), max(counts)
    if lo == hi == len(repeated):
        return (f"{label}: both rows of the repeated {clock} local hour are absent on {days}, {len(repeated)} hours per node."
                + (" The July downloaders drop them (§6.1.1)." if july_downloader else ""))
    if hi == 0:
        return f"{label}: both rows of the repeated {clock} local hour are present on {days}."
    span = str(lo) if lo == hi else f"{lo} to {hi}"
    return (f"{label}: of the {len(repeated)} rows of the repeated {clock} local hour on {days}, "
            f"{span} hour{'' if hi == 1 else 's'} per node {'is' if lo == hi == 1 else 'are'} absent.")


def _other_gap_note(label: str, gaps: dict, settle):
    """The hours absent for any other reason, as runs of consecutive hours with their local settlement dates, and whether the
    settlement series has them."""
    tz, full, repeated, missing = gaps["tz"], gaps["full"], gaps["repeated"], gaps["missing"]
    other = {n: m.difference(repeated) for n, m in missing.items()}
    hours = pd.DatetimeIndex(sorted(set().union(*(set(m) for m in other.values()))))
    if len(hours) == 0:
        return None
    full_dates = np.asarray(full.tz_convert(tz).date)

    def describe(run: pd.DatetimeIndex) -> str:
        affected = [n for n, m in other.items() if len(m.intersection(run))]
        dates = sorted(set(run.tz_convert(tz).date))
        parts = []
        for d in dates:
            present = [int(((full_dates == d) & ~full.isin(missing[n])).sum()) for n in affected]
            lo, hi = min(present), max(present)
            parts.append(f"{d.isoformat()}: {lo if lo == hi else f'{lo} to {hi}'} of {_local_day_hours(d, tz)} hours present")
        return (f"{_run_label(run)} ({len(affected)} of {len(other)} nodes; local settlement "
                f"date{'' if len(dates) == 1 else 's'} {', '.join(parts)})")

    runs = _runs(hours)
    shown = [describe(r) for r in runs[:6]]
    if len(runs) > 6:
        shown.append(f"{len(runs) - 6} more runs")
    text = (f"{label}: {len(hours)} hour{'s' if len(hours) != 1 else ''} absent outside the fall-back rule: "
            f"{'; '.join(shown)}.")
    if settle is not None:
        name, table = settle
        affected = [n for n, m in other.items() if len(m)]
        in_table = set(table["node"].unique())
        shared = [n for n in affected if n in in_table]
        if shared:
            have = {n: pd.DatetimeIndex(pd.to_datetime(table.loc[table["node"] == n, "timestamp_utc"], utc=True)) for n in shared}
            total = sum(len(other[n]) for n in shared)
            lacking = sum(int((~other[n].isin(have[n])).sum()) for n in shared)
            text += (f" The {name} series has every one of these node-hours." if lacking == 0
                     else f" The {name} series lacks {lacking} of these {total} node-hours.")
            if len(shared) < len(affected):
                unshared = len(affected) - len(shared)
                text += f" ({unshared} affected node{'' if unshared == 1 else 's'} not in that series.)"
    return text


def _names_note(label: str, panel_nodes, table: pd.DataFrame):
    """A settlement series is keyed by its own names: which are not panel nodes, and which panel nodes it lacks."""
    names, panel_nodes = set(table["node"].unique()), set(panel_nodes)
    extra, lacking = sorted(names - panel_nodes), sorted(panel_nodes - names)
    parts = []
    if extra:
        parts.append(f"{len(names)} name{'' if len(names) == 1 else 's'}, {len(extra)} of them not panel nodes ({', '.join(extra)})")
    if lacking:
        parts.append(f"{len(lacking)} panel node{'' if len(lacking) == 1 else 's'} not in it ({', '.join(lacking)})")
    return f"{label}: {'; '.join(parts)}." if parts else None


def coverage_notes(raw: pd.DataFrame, panel: pd.DataFrame, fmm, nyi) -> list[str]:
    """Plain-text notes for the coverage table, derived from the loaded series and nothing else: the duplicates (from
    raw.attrs, which main fills from the rows the loader dropped), and for every series the hours absent against a
    continuous hourly UTC index, as fall-back hours (the repeated local hour, which the July NYISO and ISO-NE downloaders
    drop) and as other gaps with their local settlement dates, then the names of the settlement series."""
    notes = [_duplicates_note(raw)]
    settle = {}                                           # the settlement leg whose nodes are the panel's, per market
    if nyi is not None:
        settle["NYISO"] = ("NYISO integrated", nyi)
    if fmm is not None:
        settle["CAISO"] = ("CAISO FMM", fmm)
    july_drops = ("NYISO", "ISONE")                       # spec v3 §6.1: the July downloaders of these two drop both fall-back rows
    series = [(f"{m} panel", SETTLE_TZ[m], panel[panel["market"] == m], settle.get(m), m in july_drops, None) for m in MARKETS]
    if fmm is not None:
        series.append((FMM_SERIES, SETTLE_TZ["CAISO"], fmm, None, False, "CAISO"))
    if nyi is not None:
        series.append((NYI_SERIES, SETTLE_TZ["NYISO"], nyi, None, False, "NYISO"))
    for label, tz, frame, comparator, july, panel_market in series:
        if len(frame) == 0:
            continue
        gaps = _series_gaps(frame, tz)
        extra = _names_note(label, panel.loc[panel["market"] == panel_market, "node"], frame) if panel_market else None
        notes += [n for n in (_fallback_note(label, gaps, july), _other_gap_note(label, gaps, comparator), extra) if n]
    return notes


def collect_coverage(raw: pd.DataFrame, panel: pd.DataFrame, fmm, nyi) -> pd.DataFrame:
    """One row per series (coverage.csv). The returned frame's attrs["notes"] holds coverage_notes(...) and attrs["duplicates"]
    the structured duplicates field."""
    if "duplicates_dropped" not in raw.attrs:
        raise KeyError("raw.attrs['duplicates_dropped'] is missing: pass the frame returned by es.data.load_dev_raw")
    rows = []
    for market in MARKETS:
        sel = panel["market"] == market
        rows.append(_series_row(market, panel.loc[sel, "node"].nunique(), int(sel.sum()),
                                panel.loc[sel, "timestamp_utc"], SETTLE_TZ[market]))
    if fmm is not None:
        rows.append(_series_row(FMM_SERIES, fmm["node"].nunique(), len(fmm), fmm["timestamp_utc"], SETTLE_TZ["CAISO"]))
    if nyi is not None:
        rows.append(_series_row(NYI_SERIES, nyi["node"].nunique(), len(nyi), nyi["timestamp_utc"], SETTLE_TZ["NYISO"]))
    df = pd.DataFrame(rows)
    df["duplicates_dropped_total"] = int(raw.attrs["duplicates_dropped"])
    df.attrs["notes"] = coverage_notes(raw, panel, fmm, nyi)
    df.attrs["duplicates"] = _duplicates_field_of(raw)
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-data", action="store_true",
                    help="also compute coverage from the development data (development-period rows only; seconds to about a minute)")
    a = ap.parse_args(argv)
    ensure_paper_dirs()
    out = PAPER_OUT_DIR / "implementation.json"
    info = collect_static()
    if a.with_data:
        from .data import build_panel
        from .settlement import load_caiso_fmm, load_nyiso_integrated
        raw, dropped = load_dev_raw_with_dropped()
        raw.attrs["duplicates_dropped_where"] = summarize_dropped(dropped)
        raw.attrs["duplicates_dropped_field"] = duplicates_field(dropped)
        panel = build_panel(raw)
        cov = collect_coverage(raw, panel, load_caiso_fmm(DATA_ROOT / "caiso_fmm_dev"),
                               load_nyiso_integrated(DATA_ROOT / "nyiso_rtint_dev"))
        cov.to_csv(PAPER_OUT_DIR / "coverage.csv", index=False)
        info["coverage"] = cov.to_dict(orient="records")
        info["coverage_notes"] = cov.attrs["notes"]
        info["coverage_duplicates"] = cov.attrs["duplicates"]
    elif out.exists():                                    # a static run keeps the data part of the last --with-data run
        stored = json.loads(out.read_text())
        info.update({k: stored[k] for k in ("coverage", "coverage_notes", "coverage_duplicates") if k in stored})
    out.write_text(json.dumps(info, indent=2, default=str))
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
