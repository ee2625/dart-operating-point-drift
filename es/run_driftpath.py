"""The alert rate of the fixed and the rolling cutoff through the deployment window (Figure 2 of the revision).

For the two main analyses (``2024-2025`` and ``25H1-26H1``) this reads the cached scores of the primary pipeline (the combined
block, XGBoost, seed 42), applies the fixed cutoff (P1, ``p1_fixed``) and the rolling cutoff (P4, ``p4_rolling`` over 56 days) at the
5 % target, and counts for every node and every Monday-start calendar week of the deployment window the hours and the alerts:
``outputs/es/paper/drift_path.csv``, one row per analysis, policy, node and week with the columns ``analysis, policy, market, node,
week_start, n_hours, alerts, rate, R, full_week``. ``R`` is the weekly rate over the target. A week is ``full_week`` when it has seven
settlement days and every one of them has all the hours of its local day (24, or 23 and 25 on the daylight-saving days of the three
markets that observe them; MISO settles in fixed EST). Weeks are those of the local settlement date, the one that the panel's
``settle_date`` holds.

Trailing windows (T1b). The rate of one week is too noisy to show what a cutoff does, so the figure and the text read the rate over a
trailing window of ``WINDOW_WEEKS`` (8) Monday-start weeks, the length of the rolling cutoff's window (56 days; asserted at import).
``window_r`` gives, for every node and every week from the ``WINDOW_MIN_WEEKS``-th (4th) of the deployment window on, the alerts and the
node-hours summed over the last eight weeks that end with that week (the weeks so far at the start) and ``R = alerts / (0.05 * hours)``:
a ratio of sums, so that a partial week or a week with missing hours counts with the hours it has (``full_week`` is not read). A node's
window with fewer than ``WINDOW_MIN_HOURS_SHARE`` (half) of the nominal hours of its weeks (24 a day, seven days a week), or with none, is
left out. A window is drawn at its ``week_end``: the Monday after its last week, or, for the last week of the deployment window, the day
after its last settlement day. A cumulative rate is not used: it would hide the excursions of the rolling cutoff.

The second file, ``outputs/es/paper/drift_summary.csv`` (``drift_summary_frame``), has the columns ``analysis, policy, metric, value``,
one row per analysis, policy and metric, in a fixed order, at full precision: ``nodes``; over the whole window ``nodes_outside`` (R below
0.5 or above 2, as ``failing_nodes``), ``nodes_above`` and ``nodes_below``; and over the windows ``window_n`` (how many),
``window_outside_mean``, ``window_outside_min`` and ``window_outside_max`` (the nodes whose window R is outside the band) and
``window_median_min`` and ``window_median_max`` (the median over nodes of the window R). The macros ``drift*`` of the registry read it.

Development data only: the scores are those of the development windows, and the run stops if one settlement date is after 2026-06-30.

A cache hit or a stop. ``cached_scores`` fits the models when its cache has no entry for the panel at hand; this script never does. While
it asks the cache, ``es.scores.score_analysis`` (the only thing that fits) is replaced by a refusal, so a cold cache ends the run with
``ColdCache`` and nothing is fitted or written. The only files the run creates are ``drift_path.csv`` and ``drift_summary.csv`` (written
at the end, after every check, each through a temporary file, so a failed check leaves no half file); its log is the standard output.

Consistency checks, each of which stops the run when it fails:

1. Counts. The weekly sums reproduce the node counts of ``es.policies.activity`` (hours, alerts and R), and for P1 the node counts of
   the per-node files behind Table 3 (``legacy/<analysis>__G1_24__xgb__s42__P1__a0.05.csv``). A node's R over the whole window is
   ``sum(alerts) / (0.05 * sum(n_hours))``, and the number of nodes with R outside [0.5, 2] equals the combined-block, XGBoost cell of
   ``tab_block_audit`` (Table 3) for the same period, read from the frame that table is built from (``legacy/summary.csv``).
2. Economics. The pooled alert rate (sum of alerts over sum of node-hours) equals the ``alert_rate`` of ``economics/summary.csv`` for the
   same analysis and policy. The two are one quantity over the same nodes and hours once the economics' own trimming is applied:
   ``es.run_economics`` relabels, before any policy runs, the partial local days at the edges of a UtcWindow deployment (an edge day with
   fewer than 23 rows; ``es.measures.drop_partial_edge_days``), and ``alert_rate`` of the position ``inc`` (which drops no row) with the
   calibrated fees is the mean of the alerts over what is left. P1 and P4 do not change on the days that stay: the P1 cutoff comes from
   the calibration rows, and the rolling window of P4 reads every non-training row, the relabelled ones included. The rate over all hours
   is logged next to it.
3. The summary. For every analysis and policy ``nodes_above + nodes_below`` equals ``nodes_outside``, and for P1 ``nodes_outside`` (of
   ``nodes``) is again the cell of Table 3, read by the function that ``check_table3`` uses, from the rows about to be written.

Run: DART_PROJECT_ROOT=... /opt/anaconda3/bin/python3 -m es.run_driftpath > outputs/es/paper/drift_path.log"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import scores as _scores
from .analyses import LEGACY
from .config import ALPHA, SEAL_LAST_DEV_DATE, SETTLE_TZ
from .data import build_panel, load_dev_raw
from .guard import SealedDataError
from .measures import drop_partial_edge_days
from .paper_macros import Sources
from .paper_paths import PAPER_OUT_DIR
from .paper_pins import BANDS, PINS, band_text
from .paper_tables import _cell, _summary
from .policies import activity, p1_fixed, p4_rolling
from .scores import cached_scores
from .windows import UtcWindow

#: The two main analyses of the audit (the deployment year 2025 and the first half of 2026).
ANALYSES = ("2024-2025", "25H1-26H1")
#: The primary pipeline: the combined block, XGBoost, seed 42.
BLOCK, ESTIMATOR, SEED = "G1_24", "xgb", 42
#: The window of the rolling cutoff, in days: the default of ``p4_rolling`` (the registry's pFourWindow), not typed here.
P4_WINDOW_DAYS = PINS["pFourWindow"]
POLICIES = ("P1", "P4")
#: The files as Sources names them (relative to outputs/es), and where they are written: the weekly path, and its summary over trailing windows.
SOURCE = "paper/drift_path.csv"
OUT_PATH = PAPER_OUT_DIR / "drift_path.csv"
SUMMARY_SOURCE = "paper/drift_summary.csv"
SUMMARY_PATH = PAPER_OUT_DIR / "drift_summary.csv"
COLUMNS = ["analysis", "policy", "market", "node", "week_start", "n_hours", "alerts", "rate", "R", "full_week"]
SUMMARY_COLUMNS = ["analysis", "policy", "metric", "value"]
#: The metrics of the summary, one row each per analysis and policy, in this order. nodes_*: over the whole deployment window (whole_period_r);
#: window_*: over the trailing windows (window_r), the number of windows, the nodes whose window R is outside the band (mean, smallest and largest
#: over the windows) and the median over nodes of the window R (smallest and largest over the windows).
SUMMARY_METRICS = ("nodes", "nodes_outside", "nodes_above", "nodes_below", "window_n", "window_outside_mean", "window_outside_min",
                   "window_outside_max", "window_median_min", "window_median_max")
#: The failure band of R of the audit (fail_0.5_2): outside it is a failure, the two ends are inside.
BAND = BANDS["0.5_2"]
LAST_DEV_DAY = date.fromisoformat(SEAL_LAST_DEV_DATE)
#: The days of a week (a week is ``full_week`` when every one of its seven days has all the hours of its local day).
DAYS_PER_WEEK = 7
#: The nominal hours of a day and of a week (daylight-saving days are ignored on purpose): what a window's hours are compared with.
HOURS_PER_DAY = 24
NOMINAL_WEEK_HOURS = HOURS_PER_DAY * DAYS_PER_WEEK
#: The trailing windows of plan 2C (T1b). A window is the last WINDOW_WEEKS Monday-start weeks of the deployment window that end with a given
#: week; at the start of the window it is the weeks so far, from the WINDOW_MIN_WEEKS-th week on. Eight weeks are the 56 days of P4's rolling window.
WINDOW_WEEKS = 8
WINDOW_MIN_WEEKS = 4
#: A node's window is left out when it has fewer hours than this share of the nominal hours of its weeks (24 a day, seven days a week), or none.
WINDOW_MIN_HOURS_SHARE = 0.5
if WINDOW_WEEKS * DAYS_PER_WEEK != P4_WINDOW_DAYS:                  # not an assert: it must hold under python -O too
    raise ImportError(f"WINDOW_WEEKS = {WINDOW_WEEKS} weeks is {WINDOW_WEEKS * DAYS_PER_WEEK} days, but the rolling window of P4 is {P4_WINDOW_DAYS} days")
#: The floating point tolerance of the comparisons of two rates or two Rs that are one number computed in two ways.
REL_TOL = 1e-9


class DriftPathError(RuntimeError):
    """A consistency check of the drift path failed."""


class ColdCache(DriftPathError):
    """The score cache holds no entry for an analysis: this script reads the cache and never refits a model."""


# ---------------------------------------------------------------- the day clock and the weekly counts
def assert_development_only(frame: pd.DataFrame) -> pd.DataFrame:
    """``frame`` unchanged, or SealedDataError if one of its settlement dates is after the last development day (2026-06-30)."""
    latest = pd.Timestamp(frame["settle_date"].max()).date()
    if latest > LAST_DEV_DAY:
        raise SealedDataError(f"the score frame holds a settlement date as late as {latest}, after the last development day {LAST_DEV_DAY}; "
                              "this script reads development windows only")
    return frame


def hours_in_local_day(market: str, day) -> int:
    """The hours of a settlement day on its market's clock: 24, or 23 and 25 on the days that the clock changes (KeyError for an unknown market)."""
    zone = SETTLE_TZ[market]
    midnight = pd.Timestamp(day).normalize()
    return int(((midnight + pd.Timedelta(days=1)).tz_localize(zone) - midnight.tz_localize(zone)) / pd.Timedelta(hours=1))


def weekly_counts(alerts: pd.DataFrame, alpha: float = ALPHA) -> pd.DataFrame:
    """One row per node and Monday-start week of the settlement dates in ``alerts`` (a frame with market, node, settle_date and alert): the
    hours (``n_hours``), the alerts, ``rate`` = alerts / hours, ``R`` = rate / alpha and ``full_week`` (seven settlement days, each with all
    the hours of its local day). A week that the window cuts is counted over the hours that it has, and is not full."""
    missing = [c for c in ("market", "node", "settle_date", "alert") if c not in alerts.columns]
    if missing:
        raise LookupError(f"weekly_counts: missing columns {missing}; have {list(alerts.columns)}")
    day = pd.to_datetime(alerts["settle_date"])
    rows = pd.DataFrame({"market": alerts["market"].to_numpy(), "node": alerts["node"].to_numpy(), "settle_date": day.to_numpy(),
                         "alert": alerts["alert"].to_numpy(dtype=bool)})
    rows["week_start"] = rows["settle_date"] - pd.to_timedelta(rows["settle_date"].dt.weekday, unit="D")
    per_day = (rows.groupby(["market", "node", "week_start", "settle_date"], sort=True)
               .agg(n=("alert", "size"), alerts=("alert", "sum")).reset_index())
    clock = {(m, d): hours_in_local_day(m, d) for m, d in per_day[["market", "settle_date"]].drop_duplicates().itertuples(index=False, name=None)}
    per_day["whole"] = per_day["n"].to_numpy() == np.array([clock[(m, d)] for m, d in zip(per_day["market"], per_day["settle_date"])])
    weekly = (per_day.groupby(["market", "node", "week_start"], sort=True)
              .agg(n_hours=("n", "sum"), alerts=("alerts", "sum"), whole_days=("whole", "sum")).reset_index())
    weekly["full_week"] = weekly["whole_days"] == DAYS_PER_WEEK            # a week has at most seven distinct days, so seven whole days are all of them
    weekly["rate"] = weekly["alerts"] / weekly["n_hours"]
    weekly["R"] = weekly["rate"] / alpha
    return weekly[["market", "node", "week_start", "n_hours", "alerts", "rate", "R", "full_week"]]


def whole_period_r(weekly: pd.DataFrame, alpha: float = ALPHA) -> pd.DataFrame:
    """Per node, over the whole window and from the weekly sums: ``n_hours``, ``alerts`` and ``R = sum(alerts) / (alpha * sum(n_hours))``."""
    sums = weekly.groupby(["market", "node"], sort=True)[["n_hours", "alerts"]].sum().reset_index()
    sums["R"] = sums["alerts"] / (alpha * sums["n_hours"])
    return sums


def failing_nodes(r: pd.DataFrame, band: Sequence[float] = BAND) -> int:
    """The nodes whose R is outside the band (the audit's fail_0.5_2: R < lo or R > hi, so the two ends are inside)."""
    lo, hi = band
    return int(((r["R"] < lo) | (r["R"] > hi)).sum())


def nodes_above_below(r: pd.DataFrame, band: Sequence[float] = BAND) -> tuple[int, int]:
    """(nodes with R above the band, nodes with R below it): R > hi and R < lo, so that the two together are ``failing_nodes``."""
    lo, hi = band
    return int((r["R"] > hi).sum()), int((r["R"] < lo).sum())


def window_r(weekly: pd.DataFrame, weeks: int = WINDOW_WEEKS, min_weeks: int = WINDOW_MIN_WEEKS, alpha: float = ALPHA) -> pd.DataFrame:
    """The alert rate of every node over trailing windows of ``weeks`` Monday-start weeks.

    ``weekly`` is the weekly path of one analysis and one policy (the rows of ``weekly_counts`` or of drift_path.csv: market, node, week_start, n_hours and
    alerts are read, ``full_week`` is not). The weeks are the calendar weeks from the first ``week_start`` to the last, a week that no node has included. For
    each week from the ``min_weeks``-th on, the window is the last ``weeks`` weeks that end with it (the weeks so far at the start of the deployment window).
    A node's ``n_hours`` and ``alerts`` are the sums over its window and ``R = alerts / (alpha * n_hours)``: a ratio of sums, so a partial week or a week
    with missing hours counts with the hours it has (the rate is alerts per hour). A node's window with fewer hours than WINDOW_MIN_HOURS_SHARE of the nominal
    hours of its weeks (24 a day, seven days a week, ``weeks`` of them in the window), or with none, is left out of that window.

    ``week_end`` is where the window is drawn: the Monday after its last week, and for the last week of the frame the day after its last settlement day. That
    cap is taken from the frame, not from es.analyses: the last ``week_start`` plus the days that week has, read as the largest ``n_hours`` of the week over the
    nodes in days, rounded up, at most seven (daylight-saving days fall on Sundays, so a week with a 25-hour day is a whole week). For the two main analyses it
    is the end of ``LEGACY[analysis].deploy`` (a test on the real tree).

    One row per node and window, ordered by market, node and ``week_end``: market, node, week_end, ``weeks`` (the weeks in the window), n_hours, alerts, R."""
    need = ["market", "node", "week_start", "n_hours", "alerts"]
    missing = [c for c in need if c not in weekly.columns]
    if missing:
        raise LookupError(f"window_r: missing columns {missing}; have {list(weekly.columns)}")
    if not 1 <= min_weeks <= weeks:
        raise ValueError(f"window_r: min_weeks must be between 1 and weeks ({weeks}), got {min_weeks}")
    columns = ["market", "node", "week_end", "weeks", "n_hours", "alerts", "R"]
    if weekly.empty:
        return pd.DataFrame(columns=columns)
    rows = weekly[need].assign(week_start=pd.to_datetime(weekly["week_start"]))
    if rows.isna().any().any():
        raise ValueError("window_r: a missing value in market, node, week_start, n_hours or alerts (a missing number is never read as no hours or no alerts)")
    if rows.duplicated(["market", "node", "week_start"]).any():
        raise ValueError("window_r: more than one row for a node and a week: the frame must be of one analysis and one policy")
    first, last = rows["week_start"].min(), rows["week_start"].max()
    if ((rows["week_start"] - first).dt.days % DAYS_PER_WEEK).any():
        raise ValueError(f"window_r: week_start values that are not a whole number of weeks apart (first {first.date()})")
    grid = pd.date_range(first, last, freq=f"{DAYS_PER_WEEK}D")
    hours = rows.pivot(index=["market", "node"], columns="week_start", values="n_hours").reindex(columns=grid).fillna(0).sort_index()
    alerts = rows.pivot(index=["market", "node"], columns="week_start", values="alerts").reindex(columns=grid).fillna(0).sort_index()
    h_all, a_all = hours.to_numpy(float), alerts.to_numpy(float)
    nodes = hours.index.to_frame(index=False)
    cap = grid[-1] + pd.Timedelta(days=max(1, min(DAYS_PER_WEEK, math.ceil(h_all[:, -1].max() / HOURS_PER_DAY))))
    parts = []
    for i in range(min_weeks - 1, len(grid)):
        first_week = max(0, i - weeks + 1)
        in_window = i - first_week + 1                                  # the weeks in this window: fewer than `weeks` at the start
        h, a = h_all[:, first_week:i + 1].sum(axis=1), a_all[:, first_week:i + 1].sum(axis=1)
        keep = (h > 0) & (h >= WINDOW_MIN_HOURS_SHARE * NOMINAL_WEEK_HOURS * in_window)
        if keep.any():
            parts.append(pd.DataFrame({"market": nodes["market"].to_numpy()[keep], "node": nodes["node"].to_numpy()[keep],
                                       "week_end": min(grid[i] + pd.Timedelta(days=DAYS_PER_WEEK), cap), "weeks": in_window,
                                       "n_hours": h[keep].astype("int64"), "alerts": a[keep].astype("int64")}))
    if not parts:
        return pd.DataFrame(columns=columns)
    out = pd.concat(parts, ignore_index=True).sort_values(["market", "node", "week_end"], kind="mergesort").reset_index(drop=True)
    out["R"] = out["alerts"] / (alpha * out["n_hours"])
    return out[columns]


# ---------------------------------------------------------------- the scores: a cache hit or a stop
def policy_alerts(scores: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The fixed cutoff (P1) and the rolling cutoff (P4, over ``P4_WINDOW_DAYS``) of one score frame at the 5 % target."""
    return {"P1": p1_fixed(scores, ALPHA), "P4": p4_rolling(scores, ALPHA, window_days=P4_WINDOW_DAYS)}


def primary_scores(panel, analysis, *, fetch: Callable = cached_scores) -> pd.DataFrame:
    """The primary pipeline's scores of ``analysis`` from the cache (``fetch`` is ``cached_scores``): ColdCache if the cache would have to refit,
    SealedDataError if a score is dated after 2026-06-30. For the duration of the call ``es.scores.score_analysis`` is a refusal, and it is the
    original again afterwards, whatever happens."""
    original = _scores.score_analysis

    def refuse(*_args, **_kwargs):
        raise ColdCache(f"the score cache has no entry for {analysis.id} / {BLOCK} / {ESTIMATOR} / seed {SEED}: reading it would refit the "
                        "models, which this script never does (a cache hit or a stop)")

    _scores.score_analysis = refuse
    try:
        frame = fetch(panel, analysis, BLOCK, ESTIMATOR, SEED)
    finally:
        _scores.score_analysis = original
    return assert_development_only(frame)


# ---------------------------------------------------------------- what the checks compare with
def table3_failures(src: Sources, analysis: str) -> tuple[int, int]:
    """(nodes outside the band, nodes) of the combined block with XGBoost in the audit of ``analysis``: the cell of ``tab_block_audit`` (Table 3),
    read from the frame that table is built from (legacy/summary.csv, the primary seed and the 5 % cutoff, policy P1)."""
    summary = _summary(src)
    if "n_nodes" not in summary.columns:
        raise LookupError(f"legacy/summary.csv: missing column n_nodes; have {list(summary.columns)}")
    failures, nodes = (_cell(summary, analysis, BLOCK, ESTIMATOR, column, int) for column in ("failures", "n_nodes"))
    if isinstance(failures, str) or isinstance(nodes, str):                 # _cell answers a row that is not there with its pending sentinel
        raise LookupError(f"legacy/summary.csv has no row for analysis {analysis!r}, block {BLOCK}, estimator {ESTIMATOR}: the cell of Table 3")
    return failures, nodes


def economics_alert_rate(src: Sources, analysis: str, policy: str) -> float:
    """``alert_rate`` of economics/summary.csv for the policy, the position ``inc`` (which drops no row) and the calibrated fees."""
    summary = src.frame("economics/summary.csv")
    need = ["analysis", "policy", "comparator", "position", "fees", "alert_rate"]
    if [c for c in need if c not in summary.columns]:
        raise LookupError(f"economics/summary.csv: missing columns {[c for c in need if c not in summary.columns]}; have {list(summary.columns)}")
    rows = summary[(summary["analysis"] == analysis) & (summary["policy"] == policy) & (summary["comparator"] == "none")
                   & (summary["position"] == "inc") & (summary["fees"].astype(str) == "calibrated")]
    if len(rows) != 1:
        raise LookupError(f"economics/summary.csv: {len(rows)} rows for analysis {analysis!r}, policy {policy!r}, position inc, calibrated fees; expected one")
    return float(rows["alert_rate"].iloc[0])


def economics_hours(alerts: pd.DataFrame, *, trim_edges: bool) -> pd.DataFrame:
    """The rows over which the economics measures the alert rate: for a UtcWindow deployment (``trim_edges``) the rows without the partial days at
    the edges of each node's window (es.measures.drop_partial_edge_days, the rule es.run_economics applies before any policy runs); else all rows."""
    return drop_partial_edge_days(alerts) if trim_edges else alerts


# ---------------------------------------------------------------- the checks
def _same(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=REL_TOL, abs_tol=0.0)


def check_counts(analysis: str, policy: str, alerts: pd.DataFrame, weekly: pd.DataFrame) -> str:
    """The weekly sums are the node counts of ``es.policies.activity`` (hours, alerts and R, node by node)."""
    sums = whole_period_r(weekly).set_index(["market", "node"])
    act = activity(alerts, ALPHA).set_index(["market", "node"])
    if not sums.index.equals(act.index):
        raise DriftPathError(f"{analysis} {policy}: the weekly path has {len(sums)} nodes and the alerts have {len(act)}, or other nodes")
    for column, other in (("n_hours", "n"), ("alerts", "alerts")):
        bad = sums.index[(sums[column].to_numpy() != act[other].to_numpy())]
        if len(bad):
            raise DriftPathError(f"{analysis} {policy}: the weekly sums of {column} differ from the node counts of the alerts for "
                                 f"{'/'.join(bad[0])} ({sums.loc[bad[0], column]} against {act.loc[bad[0], other]})")
    if not np.allclose(sums["R"].to_numpy(), act["R"].to_numpy(), rtol=REL_TOL, atol=0):
        raise DriftPathError(f"{analysis} {policy}: R from the weekly sums differs from the R of the alerts")
    return (f"[check counts] {analysis} {policy}: the weekly sums reproduce the node counts of the alerts: {int(sums['n_hours'].sum()):,} node-hours, "
            f"{int(sums['alerts'].sum()):,} alerts, {len(sums)} nodes - matched")


def _match_table3(analysis: str, got: tuple[int, int], what: str, src: Sources, tag: str) -> str:
    """P1: ``got`` = (nodes with R outside the band, nodes), as ``what`` gives them, against the cell of Table 3 (combined block, XGBoost) of the same period."""
    expected = table3_failures(src, analysis)
    if got != expected:
        raise DriftPathError(f"{analysis} P1: {what} give {got[0]} of {got[1]} nodes with R outside {band_text(BAND)}, "
                             f"Table 3 (combined block, XGBoost) has {expected[0]} of {expected[1]}; "
                             + ("the number of nodes differs" if got[1] != expected[1] else "the failure count differs"))
    return (f"[{tag}] {analysis} P1: {what} give {got[0]} of {got[1]} nodes with R outside {band_text(BAND)}; Table 3 (combined "
            f"block, XGBoost) has {expected[0]} of {expected[1]} - matched")


def check_table3(analysis: str, weekly_p1: pd.DataFrame, src: Sources) -> str:
    """P1: the nodes with R outside the band, from the weekly sums, are the cell of Table 3 (combined block, XGBoost) of the same period."""
    sums = whole_period_r(weekly_p1)
    return _match_table3(analysis, (failing_nodes(sums), len(sums)), "the weekly sums", src, "check Table 3")


def check_per_node(analysis: str, weekly_p1: pd.DataFrame, src: Sources) -> str:
    """P1: hours, alerts and R of every node equal the per-node file behind Table 3."""
    name = f"legacy/{analysis}__{BLOCK}__{ESTIMATOR}__s{SEED}__P1__a{ALPHA}.csv"
    legacy = src.frame(name)
    need = ["market", "node", "n", "alerts", "R"]
    if [c for c in need if c not in legacy.columns]:
        raise LookupError(f"{name}: missing columns {[c for c in need if c not in legacy.columns]}; have {list(legacy.columns)}")
    sums, legacy = whole_period_r(weekly_p1).set_index(["market", "node"]), legacy.set_index(["market", "node"])
    if not sums.index.equals(legacy.index.sort_values()):
        raise DriftPathError(f"{analysis} P1: the nodes of the per-node file behind Table 3 ({name}) are not the nodes of the weekly path")
    legacy = legacy.reindex(sums.index)
    for node in sums.index:
        got, want = sums.loc[node], legacy.loc[node]
        if (got["n_hours"], got["alerts"]) != (want["n"], want["alerts"]) or not _same(got["R"], want["R"]):
            raise DriftPathError(f"{analysis} P1: the per-node file behind Table 3 differs from the weekly sums for {'/'.join(node)}: hours "
                                 f"{got['n_hours']} against {want['n']}, alerts {got['alerts']} against {want['alerts']}, R {got['R']:.6g} against {want['R']:.6g}")
    return f"[check per-node] {analysis} P1: hours, alerts and R of the {len(sums)} nodes equal the per-node file behind Table 3 ({name}) - matched"


def check_economics(analysis: str, policy: str, alerts: pd.DataFrame, src: Sources, *, trim_edges: bool) -> str:
    """The pooled alert rate over the hours the economics measures equals the alert_rate of economics/summary.csv for the same analysis and policy."""
    kept = economics_hours(alerts, trim_edges=trim_edges)
    pooled, every_hour = float(kept["alert"].mean()), float(alerts["alert"].mean())
    expected = economics_alert_rate(src, analysis, policy)
    if not _same(pooled, expected):
        raise DriftPathError(f"{analysis} {policy}: the pooled alert rate over the {len(kept):,} hours the economics measures is {pooled:.6f}, "
                             f"alert_rate of economics/summary.csv is {expected:.6f}")
    return (f"[check economics] {analysis} {policy}: pooled alert rate {pooled:.6f} over {len(kept):,} node-hours (partial edge days "
            f"{'left out' if trim_edges else 'not trimmed for this window'}; all {len(alerts):,} hours: {every_hour:.6f}) equals alert_rate of "
            f"economics/summary.csv {expected:.6f} - matched")


def check_summary(summary: pd.DataFrame, src: Sources) -> list[str]:
    """The rows of drift_summary.csv, before they are written: for every analysis and policy ``nodes_above + nodes_below`` is ``nodes_outside``, and for P1
    ``nodes_outside`` (of ``nodes``) is the cell of Table 3, compared by the function that check_table3 uses. One log line per analysis and policy."""
    lines = []
    for (analysis, policy), rows in summary.groupby(["analysis", "policy"], sort=False):
        v = dict(zip(rows["metric"], rows["value"]))
        missing = [m for m in SUMMARY_METRICS if m not in v]
        if missing:
            raise DriftPathError(f"{analysis} {policy}: the summary lacks the metrics {missing}")
        if v["nodes_above"] + v["nodes_below"] != v["nodes_outside"]:
            raise DriftPathError(f"{analysis} {policy}: nodes_above ({v['nodes_above']}) + nodes_below ({v['nodes_below']}) is not nodes_outside "
                                 f"({v['nodes_outside']})")
        lines.append(f"[check summary] {analysis} {policy}: {v['nodes_above']} above + {v['nodes_below']} below = {v['nodes_outside']} of "
                     f"{v['nodes']} nodes outside {band_text(BAND)} over the whole window - matched")
        if policy == "P1":
            lines.append(_match_table3(analysis, (v["nodes_outside"], v["nodes"]), "the rows of drift_summary.csv", src, "check summary vs Table 3"))
    return lines


# ---------------------------------------------------------------- the files
def drift_summary_frame(weekly_by_key: Mapping[tuple[str, str], pd.DataFrame], weeks: int = WINDOW_WEEKS, min_weeks: int = WINDOW_MIN_WEEKS,
                        alpha: float = ALPHA) -> pd.DataFrame:
    """The summary of the weekly paths: columns ``analysis, policy, metric, value``, one row per (analysis, policy) key in the order of the mapping and per
    metric of SUMMARY_METRICS. Counts are integers, the others floats at full precision (the ``value`` column holds objects, so a count is written as 12 and
    not as 12.0).

    Over the whole window, from ``whole_period_r``: ``nodes``; ``nodes_outside`` (R < lo or R > hi, the two ends inside, as ``failing_nodes``);
    ``nodes_above`` (R > hi) and ``nodes_below`` (R < lo). Over the trailing windows of ``window_r`` (windows in which at least one node has a window R): the
    number of windows ``window_n``; the nodes whose window R is outside the band, as their plain ``window_outside_mean`` over the windows and their smallest
    and largest number (``window_outside_min`` and ``window_outside_max``); and the median over nodes of the window R, smallest and largest over the windows
    (``window_median_min`` and ``window_median_max``). A node whose window is left out does not count in that window."""
    lo, hi = BAND
    analysis_col, policy_col, metric_col, value_col = [], [], [], []
    for (analysis, policy), weekly in weekly_by_key.items():
        whole = whole_period_r(weekly, alpha)
        above, below = nodes_above_below(whole)
        win = window_r(weekly, weeks, min_weeks, alpha)
        if win.empty:
            raise DriftPathError(f"{analysis} {policy}: no window of {min_weeks} to {weeks} weeks has a node with enough hours "
                                 f"({weekly['week_start'].nunique()} weeks in the weekly path)")
        per_window = win.assign(outside=(win["R"] < lo) | (win["R"] > hi)).groupby("week_end").agg(outside=("outside", "sum"), median=("R", "median"))
        values = {"nodes": int(len(whole)), "nodes_outside": failing_nodes(whole), "nodes_above": above, "nodes_below": below,
                  "window_n": int(len(per_window)), "window_outside_mean": float(per_window["outside"].mean()),
                  "window_outside_min": int(per_window["outside"].min()), "window_outside_max": int(per_window["outside"].max()),
                  "window_median_min": float(per_window["median"].min()), "window_median_max": float(per_window["median"].max())}
        for metric in SUMMARY_METRICS:
            analysis_col.append(analysis)
            policy_col.append(policy)
            metric_col.append(metric)
            value_col.append(values[metric])
    return pd.DataFrame({"analysis": analysis_col, "policy": policy_col, "metric": metric_col, "value": pd.Series(value_col, dtype=object)})[SUMMARY_COLUMNS]


def _frame(weekly_by_key: Mapping[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    """The rows of every (analysis, policy) in the order given, as written: the columns of COLUMNS, the week as an ISO date."""
    parts = []
    for (analysis, policy), weekly in weekly_by_key.items():
        part = weekly.assign(analysis=analysis, policy=policy)
        part["week_start"] = part["week_start"].dt.strftime("%Y-%m-%d")
        parts.append(part[COLUMNS])
    return pd.concat(parts, ignore_index=True)


def _write(frame: pd.DataFrame, path: Path) -> None:
    """Through a temporary file in the same folder, then ``os.replace``: a failure leaves no half file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        frame.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def window_lines(summary: pd.DataFrame, weeks: int = WINDOW_WEEKS, min_weeks: int = WINDOW_MIN_WEEKS) -> list[str]:
    """One log line per analysis and policy on its trailing windows (what drift_summary.csv says about them)."""
    lines = []
    for (analysis, policy), rows in summary.groupby(["analysis", "policy"], sort=False):
        v = dict(zip(rows["metric"], rows["value"]))
        lines.append(f"[windows] {analysis} {policy}: {v['window_n']} windows of up to {weeks} weeks (from week {min_weeks}): nodes outside "
                     f"{band_text(BAND)} per window {v['window_outside_min']} to {v['window_outside_max']} (mean {v['window_outside_mean']:.2f}), median over "
                     f"nodes of the window R {v['window_median_min']:.2f} to {v['window_median_max']:.2f}")
    return lines


def run(panel, src: Sources, *, analyses: Sequence[str] = ANALYSES, fetch: Callable = cached_scores, out_path: Path = OUT_PATH,
        summary_path: Path | None = None, weeks: int = WINDOW_WEEKS, min_weeks: int = WINDOW_MIN_WEEKS, log: Callable[[str], None] = print) -> pd.DataFrame:
    """Everything: read the cached scores of each analysis, make the weekly paths of P1 and P4 and their summary over trailing windows of ``weeks`` weeks (from
    the ``min_weeks``-th: the paper's windows unless a test with a few weeks says otherwise), run every check, and only then write ``out_path`` (the weekly path)
    and ``summary_path`` (default: drift_summary.csv next to ``out_path``). Returns the weekly path."""
    summary_path = Path(summary_path) if summary_path is not None else out_path.with_name(SUMMARY_PATH.name)
    weekly_by_key: dict[tuple[str, str], pd.DataFrame] = {}
    for aid in analyses:
        analysis = LEGACY[aid]
        scores = primary_scores(panel, analysis, fetch=fetch)
        deploy = scores[scores["split"] == "deploy"]
        log(f"[scores] {aid}: {len(scores):,} score rows from the cache ({len(deploy):,} deployment rows, {deploy[['market', 'node']].drop_duplicates().shape[0]} "
            f"nodes), settlement dates up to {pd.Timestamp(scores['settle_date'].max()).date()}")
        trim = isinstance(analysis.deploy, UtcWindow)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            alerts_by_policy = policy_alerts(scores)
        for w in caught:
            log(f"[warning] {aid}: {w.message}")
        for policy in POLICIES:
            alerts = alerts_by_policy[policy]
            weekly = weekly_counts(alerts)
            log(check_counts(aid, policy, alerts, weekly))
            if policy == "P1":
                log(check_table3(aid, weekly, src))
                log(check_per_node(aid, weekly, src))
            log(check_economics(aid, policy, alerts, src, trim_edges=trim))
            full = weekly["full_week"]
            log(f"[weeks] {aid} {policy}: {len(weekly)} node-weeks, {int(full.sum())} full ({weekly['week_start'].nunique()} Monday-start weeks, "
                f"{weekly.loc[full, 'week_start'].nunique()} with a full node-week)")
            weekly_by_key[(aid, policy)] = weekly
    frame = _frame(weekly_by_key)
    summary = drift_summary_frame(weekly_by_key, weeks, min_weeks)
    for line in check_summary(summary, src) + window_lines(summary, weeks, min_weeks):
        log(line)
    _write(frame, out_path)                                             # nothing is written before every check has passed
    log(f"[write] {out_path.name}: {len(frame):,} rows")
    _write(summary, summary_path)
    log(f"[write] {summary_path.name}: {len(summary):,} rows")
    return frame


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Write the weekly alert-rate path of the fixed and the rolling cutoff and its summary over trailing "
                                             "windows (a cache hit or a stop).")
    ap.add_argument("--out", default=str(OUT_PATH), help="the weekly path to write (default outputs/es/paper/drift_path.csv); the summary "
                                                         "drift_summary.csv is written next to it")
    a = ap.parse_args(argv)
    started = time.time()

    def log(message: str) -> None:
        print(f"[{time.time() - started:6.1f}s] {message}", flush=True)

    panel = build_panel(load_dev_raw())
    log(f"panel: {len(panel):,} rows, settlement dates {panel['settle_date'].min()} to {panel['settle_date'].max()}")
    run(panel, Sources(), out_path=Path(a.out), log=log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
