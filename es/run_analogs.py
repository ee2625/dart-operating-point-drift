"""Quarterly analogues, placebo pairings, eta choice, and simulation-based size and power, all before
registration (spec v3 §6.2; review round 2, N1, N2, N7; controller Rulings 29, 49, 55, 63, 64, 68, 69, 77, 84b, 92b
and 94).

Two stages (Ruling 77), run as ``python -m es.run_analogs [--stage analogs|sim]``; with no ``--stage`` both run, in
order:

- ``analogs`` (simulation-free): ``activity_all.csv`` (every policy on the four analogue quarters and the frozen arm),
  ``eta_selection.csv`` (the self-correcting policy's eta), ``placebo_he.csv`` (the H-E placebo pairings: only the
  1-year ones are counted by the rule, Ruling 94) and a first ``protocol_inputs.json`` = {"eta", "family": null,
  "stage": "analogs", "placebo"}.
- ``sim``: reads the placebo rows and the eta back from those files, runs the parametric size and power study
  (``simulation.csv``) and rewrites ``protocol_inputs.json`` with the registered ``family`` (``registration_family``,
  computed, never decided by hand) and "stage": "complete". Re-running ``analogs`` resets the marker to "analogs": run
  ``sim`` again after it. Every size and power row carries how the persistence corrections of its block's nodes ended
  (``phi_corrected``, ``phi_clipped_high`` and the others), and a note when more than a tenth of them were clipped high.

A stage computes everything first and writes at its end: every output goes to a temporary file, then ``os.replace``
moves them into place, ``protocol_inputs.json`` last. A stage that fails writes nothing. The files hold no timing field,
so the two-stage and the one-shot run leave the same bytes; progress and timings go to standard output.

Data: the development loaders only (``load_dev_raw``, ``load_caiso_fmm`` and ``load_nyiso_integrated`` with their
default ``dev_only=True``), nothing dated after 2026-06-30. The simulated 2026-H2 quarters are calendar dates, not data.
Heavy: the first stage reads about a hundred cached score frames, the second simulates for a long time (controller
Ruling 62: run it serialized with other heavy jobs)."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import time
import warnings
from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from .analyses import ANALOGS, Analysis
from .blocks import BLOCKS
from .bootstrap import RowCube, rotation_test
from .config import DATA_ROOT, OUT_DIR, ensure_dirs
from .data import build_panel, load_dev_raw
from .measures import add_calendar_deciles
from .policies import activity, p1_fixed, p4_rolling, p5_self_correcting, p6_exact
from .positions import attach_pnl, premium_direction
from .scores import cached_scores
from .settlement import attach_settlement, load_caiso_fmm, load_nyiso_integrated
from .simulate import calibrate_delta, fit_node, hb_size, he_size_power
from .windows import DateWindow as D

ANALOG_DIR = OUT_DIR / "analogs"
ETAS = (0.005, 0.01, 0.02, 0.05)
#: Ruling 94 (C1): the pairings the registered rule counts, the four 1-year quarters and the pooled AQ3+AQ4-1y. A 2-year
#: P&L quarter lies inside the alert model's training window (its day-of-year features let the alerts reproduce that
#: year's spike days), so a 2-year pairing is not a no-skill null: it is computed and reported, never counted.
COUNTED_PAIRINGS = ("AQ3-1y", "AQ4-1y", "AQ1-1y", "AQ2-1y", "AQ3+AQ4-1y")
UNCOUNTED_NOTE = "P&L year inside the alert model's training window"
#: Out-of-sample scores to fit the simulation to: train to 2025-06-30, then 2025-07 to 2026-06 (development data only).
SIMFIT = Analysis("SIMFIT", D("2023-01-01", "2025-06-30"), D("2025-07-01", "2025-09-30"),
                  D("2025-10-01", "2026-06-30"), D("2023-01-01", "2025-06-30"), "rolling", "analog")


def shift_years(day, years):
    """``day`` moved ``years`` years back, keeping month and day (29 Feb becomes 28 Feb)."""
    try:
        return day.replace(year=day.year - years)
    except ValueError:                                   # 29 Feb -> 28 Feb
        return day.replace(year=day.year - years, day=28)


def _settled_rows(panel, analysis, fmm, nyi):
    """The analysis's primary scores (combined model, XGBoost, seed 42) and the premium-position direction learnt on
    its training rows, from the settlement price (invariant I4: no calibration or deployment row enters it)."""
    primary = cached_scores(panel, analysis, "G1_24", "xgb", 42)
    train = attach_settlement(primary[primary["split"] == "train"], fmm, nyi)
    direction = premium_direction(train, col="dart_settle")
    return primary, direction


def placebo_cube(panel, analysis, years_back, fmm, nyiso_int):
    """Analogue alerts paired with the P&L of the same quarter ``years_back`` years earlier (spec §6.2(a)).

    The alerts are those of P1 on the analogue's deployment quarter; each is moved to the same calendar date
    ``years_back`` years earlier (``shift_years``) and meets that day's P&L of the same node and local hour: the premium
    position's, with the direction learnt on the analogue's own training rows and the calibrated fees. Alerts and P&L
    come from different years, so a rotation test on the pair has no timing skill to find and its rejection rate is a
    size check. The cube has the primary strata of Ruling 68 (node x hour x weekday class x month, those of the P&L
    date). Raises ValueError for a placebo window without rows. A date that exists on one side only (29 Feb, or an hour
    a daylight-saving day repeats or lacks) is left out of the pairing, and one UserWarning says how many rows."""
    primary, direction = _settled_rows(panel, analysis, fmm, nyiso_int)
    g6 = cached_scores(panel, analysis, "G6", "xgb", 42)
    alerts = p1_fixed(primary, 0.05)[["market", "node", "settle_date", "settle_hour", "alert"]].copy()
    alerts["placebo_date"] = alerts["settle_date"].map(lambda d: shift_years(d, years_back))
    start = shift_years(dt.date.fromisoformat(analysis.deploy.start), years_back)
    end = shift_years(dt.date.fromisoformat(analysis.deploy.end), years_back)
    other = primary[(primary["settle_date"] >= start) & (primary["settle_date"] <= end)]
    if other.empty:
        raise ValueError(f"placebo_cube: {analysis.id} has no rows between {start} and {end}, the placebo window "
                         f"{years_back} year(s) before its deployment quarter")
    other = attach_pnl(attach_settlement(other, fmm, nyiso_int), "premium", direction)
    other = add_calendar_deciles(other, g6)
    key = ["market", "node", "settle_date", "settle_hour"]
    right = alerts.drop(columns="settle_date").rename(columns={"placebo_date": "settle_date", "alert": "alert_placebo"})
    paired = other.merge(right, on=key, how="inner")
    paired = paired.drop(columns=[c for c in ("alert",) if c in paired]).rename(columns={"alert_placebo": "alert"})
    n_inner = len(paired)
    paired = paired.drop_duplicates(key)
    n_pnl_only = len(other) - int(other[key].merge(right[key].drop_duplicates(), on=key, how="inner").shape[0])
    n_alert_only = len(right.drop_duplicates(key)) - len(paired)
    if n_pnl_only or n_alert_only or n_inner != len(paired):
        warnings.warn(f"placebo_cube({analysis.id!r}, {years_back} y): {n_pnl_only} P&L row(s) without a paired alert "
                      f"and {n_alert_only} alert row(s) without a P&L row are left out, and {n_inner - len(paired)} "
                      "repeated (node, date, hour) pairing(s) are kept once", UserWarning, stacklevel=2)
    cube = RowCube.from_rows(paired)
    cube.n_rows = int(len(paired))                              # Ruling 94 (M1): how many of the P&L rows rest on
    cube.n_settle_approx = int(paired["settle_approx"].sum())   # the approximate (5-minute mean) settlement leg
    return cube


# ---------------------------------------------------------------------------------------------------------------
# Stage "analogs": activity of every policy, eta, and the H-E placebo pairings (no simulation)
# ---------------------------------------------------------------------------------------------------------------

#: The cutoff policies of the activity table: name and how each reads a score frame (alpha 0.05).
POLICIES = (("P1", lambda sc: p1_fixed(sc, 0.05)), ("P4_56", lambda sc: p4_rolling(sc, 0.05, 56)),
            ("P4_28", lambda sc: p4_rolling(sc, 0.05, 28)), ("P6", lambda sc: p6_exact(sc, 0.05)))
ESTIMATORS = ("xgb", "logit")


def activity_table(panel):
    """Per node, the activity of every policy on every analogue (the four quarters and the frozen arm AQ3F), block and
    estimator: ``activity_all.csv``. Scores are read through ``cached_scores`` (seed 42)."""
    acts = []
    for a in ANALOGS.values():
        for block in BLOCKS:
            for est in ESTIMATORS:
                sc = cached_scores(panel, a, block, est, 42)
                for pol, make in POLICIES:
                    acts.append(activity(make(sc), 0.05).assign(analysis=a.id, block=block, estimator=est, policy=pol))
    return pd.concat(acts, ignore_index=True)


def eta_table(panel):
    """The panel-median |log10 R| of the self-correcting policy P5 on every analogue at every eta:
    ``eta_selection.csv``."""
    rows = []
    for a in ANALOGS.values():
        primary = cached_scores(panel, a, "G1_24", "xgb", 42)
        for eta in ETAS:
            act = activity(p5_self_correcting(primary, 0.05, eta), 0.05)
            rows.append({"analysis": a.id, "eta": eta, "median_log_error": float(act["log_error"].median())})
    return pd.DataFrame(rows)


def select_eta(eta_tab):
    """The eta of least mean panel-median |log10 R| over the four quarterly analogues (the frozen arm AQ3F is not
    pooled); ties go to the smaller eta (spec §4.3)."""
    pooled = eta_tab[eta_tab["analysis"] != "AQ3F"].groupby("eta")["median_log_error"].mean()
    return float(pooled[pooled == pooled.min()].index.min())


def _placebo_row(pair, cubes, counted):
    """The rotation test of one pairing as a ``placebo_he.csv`` row: the studentized T of the observed state (Ruling
    64), its p, the size of the null and the pooled SELC, the number of rows paired and how many of them rest on the
    approximate settlement leg (Ruling 94, M1), and whether the pairing is ``counted`` by the registered rule (a
    pairing that is not carries the reason as its ``note``). A window too short for the test (Rulings 49 and 69) or any
    other ValueError of the test is a row with ``error`` and no ``p``."""
    flags = {"counted": counted, **({} if counted else {"note": UNCOUNTED_NOTE})}
    try:
        res = rotation_test(cubes)
    except (ValueError, KeyError) as exc:
        return {"pair": pair, "error": str(exc), **flags}
    return {"pair": pair, "t_obs": res["t_obs"], "p": res["p"], "n_null": res["n_null"], "selc_obs": res["selc_obs"],
            "n_rows": int(sum(getattr(c, "n_rows", 0) for c in cubes)),
            "n_settle_approx": int(sum(getattr(c, "n_settle_approx", 0) for c in cubes)), **flags}


def placebo_table(panel, fmm, nyi):
    """The H-E placebo pairings of spec §6.2(a): AQ3, AQ4, AQ1 and AQ2 against the quarter one and two years earlier,
    then AQ3 + AQ4 pooled for each. All ten are computed and written, but only the 1-year ones are ``counted`` (Ruling
    94, C1): the P&L of a 2-year pairing comes from the alert model's own training window. A pairing that cannot be
    built or tested is a row with ``error`` and no ``p``, never a missing row; ``run_sim_stage`` refuses to run unless
    every counted pairing has a p."""
    rows, cubes = [], {}
    for aid in ("AQ3", "AQ4", "AQ1", "AQ2"):
        for k in (1, 2):
            pair = f"{aid}-{k}y"
            try:
                cubes[(aid, k)] = placebo_cube(panel, ANALOGS[aid], k, fmm, nyi)
            except (ValueError, KeyError) as exc:                       # a quarter outside the data
                rows.append({"pair": pair, "error": str(exc), "counted": k == 1,
                             **({} if k == 1 else {"note": UNCOUNTED_NOTE})})
                continue
            rows.append(_placebo_row(pair, [cubes[(aid, k)]], k == 1))
    for k in (1, 2):
        pair = f"AQ3+AQ4-{k}y"
        absent = [f"{aid}-{k}y" for aid in ("AQ3", "AQ4") if (aid, k) not in cubes]
        if absent:
            rows.append({"pair": pair, "error": f"no cube for {', '.join(absent)}", "counted": k == 1,
                         **({} if k == 1 else {"note": UNCOUNTED_NOTE})})
        else:
            rows.append(_placebo_row(pair, [cubes[("AQ3", k)], cubes[("AQ4", k)]], k == 1))
    return rows


# ---------------------------------------------------------------------------------------------------------------
# Stage "sim": parametric size and power (spec v3 §6.2(b); Rulings 55, 63 and 68; appendix B of the brief)
# ---------------------------------------------------------------------------------------------------------------

#: H-E's coupling grid (Ruling 63); the coupling whose mean SELC is nearest the effect target powers the study.
RHO_GRID = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
PERSISTENCE = ("fit", "upper")
#: The score laws H-B's size is studied under (Ruling 84b): both blocks drawn from the SIMFIT fit of this block. DART3
#: is the base block of both contrasts; LEV6 and ROLL_DISP9 are the alternatives, whose fits the test is run on.
HB_LAWS = ("DART3", "LEV6", "ROLL_DISP9")
#: H-B's drift scenarios: no drift, and a common deployment shift of both blocks by this many score SDs.
HB_DRIFTS = (("no_drift", 0.0), ("common_drift_0.25", 0.25), ("common_drift_0.5", 0.5))
#: The base law of H-B: the block both contrasts compare against, and the law the power runs draw from.
HB_BASE_LAW = "DART3"
#: Seeds (Ruling 94, I2). Every size row has a seed of its own so that the rows are independent draws, as the Bonferroni
#: correction assumes: row i of the study takes SIZE_SEED + i, the 16 H-E size rows first (test single then pooled,
#: persistence fit then upper, scenario homoskedastic then heteroskedastic, strata primary then decile), then the 18 H-B
#: size rows (law DART3, LEV6, ROLL_DISP9, persistence, drift no_drift, common_drift_0.25, common_drift_0.5). The
#: power curve (the rho search), the H-E target row and the H-B power runs keep their own seeds, and every row records
#: its own.
SIZE_SEED = 20261022
HB_SIZE_FIRST_INDEX = 16
SEED_RHO, SEED_HE_POWER, SEED_HB_POWER, SEED_CALIBRATE = 7, 11, 13, 20261022
#: calibrate_delta's grid (its default, passed explicitly): the largest alternative shift it reaches is GRID_TOP SDs.
CALIBRATION_GRID = (0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8)
GRID_TOP = CALIBRATION_GRID[-1]
#: Ruling 94 (N-2 of the Task 17 re-review): how ``fit_node``'s persistence correction ended for a block's nodes,
#: reported next to every size and power row. ``NodeParams.phi_status`` holds one of these ``PHI_STATUSES`` for each
#: component, "score" and "pnl". A block's simulation draws only the components it uses: H-E both, H-B the score alone
#: (it reads no P&L: the H-B fits are made with a constant zero P&L, whose correction always ends in "no_signal").
PHI_STATUSES = ("corrected", "clipped_high", "clipped_low", "no_signal", "uncorrected")
HE_COMPONENTS = ("score", "pnl")
HB_COMPONENTS = ("score",)
#: A clipped-high node is one whose plain persistence lay above what the correction can reach, so the top of its search
#: range was used: the simulated null then differs from the data's in a direction nobody knows. A block with MORE than
#: this share of its nodes clipped high says in the ``note`` of each of its rows that the row is not a bound.
CLIPPED_HIGH_SHARE = 0.10


def _span(start, end):
    return pd.date_range(start, end).date.tolist()


@dataclass(frozen=True)
class SimCounts:
    """Replications of the registered study. ``hb_calibrate`` is calibrate_delta's n_sim: Task 17 measured +-0.015 of
    gap noise at 30 (appendix B). ``hb_rep`` is 299 in every H-B simulation: with 299 bootstrap replicates the p-value
    is a multiple of 1/300, so alpha / 3 and 0.05 are exactly attainable (Ruling 84b). Tests pass small counts through
    ``SIM_COUNTS``."""
    he_size: int = 300         # each H-E size row
    he_rho: int = 100          # each point of the rho search
    he_power: int = 300        # H-E power at the chosen rho
    hb_size: int = 300         # each H-B size row
    hb_power: int = 300        # each H-B power run (Ruling 94, M11: at 150 its standard error near 0.5 is 0.041)
    hb_rep: int = 299          # bootstrap replicates of one H-B simulation (size and power)
    hb_calibrate: int = 100    # calibrate_delta


@dataclass(frozen=True)
class SimWindows:
    """The simulated quarters: Test A's calibration and deployment quarter (``cal_a``, ``dep_a``) and Test B's. They are
    calendar dates only: the simulation draws its own prices, so no price of the sealed 2026-H2 quarters is read."""
    cal_a: list
    cal_b: list
    dep_a: list
    dep_b: list

    @classmethod
    def default(cls):
        return cls(_span("2025-07-01", "2025-09-30"), _span("2025-10-01", "2025-12-31"),
                   _span("2026-07-01", "2026-09-30"), _span("2026-10-01", "2026-12-31"))


#: What ``run_sim_stage`` uses; a test replaces them to run the study on a handful of replications and short windows.
SIM_COUNTS = SimCounts()
SIM_WINDOWS = SimWindows.default()


def analog_selc_parts(panel, fmm, nyi):
    """[(excess, loss)] of P1 on AQ3 and on AQ4 under the primary strata: the pieces of SELC, so that the two quarters
    pool by summation, as the H-E statistic does (spec §6.3: (excess_A + excess_B) / (L_A + L_B))."""
    out = []
    for aid in ("AQ3", "AQ4"):
        pri, drc = _settled_rows(panel, ANALOGS[aid], fmm, nyi)
        g6 = cached_scores(panel, ANALOGS[aid], "G6", "xgb", 42)
        rows = add_calendar_deciles(attach_pnl(attach_settlement(p1_fixed(pri, 0.05), fmm, nyi), "premium", drc), g6)
        excess, loss = RowCube.from_rows(rows).selc_parts()
        out.append((float(excess), float(loss)))
    return out


def pooled_selc(parts):
    """The loss-weighted pooled SELC of ``analog_selc_parts``: the sum of the excesses over the sum of the losses
    (0.0 when there is no loss at all, as ``measures.selc``). The H-E effect target is half of it (Ruling 94, M12)."""
    loss = sum(l for _, l in parts)
    return float(sum(e for e, _ in parts) / loss) if loss > 0 else 0.0


def analog_selc(panel, fmm, nyi):
    """P1's SELC (primary strata) on AQ3 and on AQ4 separately: the analogue point estimates, reported next to the
    pooled one that the effect target is taken from."""
    return [float(e / l) if l > 0 else 0.0 for e, l in analog_selc_parts(panel, fmm, nyi)]


def fit_he_params(panel, fmm, nyi):
    """One ``NodeParams`` per (market, node), fitted to SIMFIT's out-of-sample scores and premium-position P&L, 2025-07
    to 2026-06."""
    fit_scores, direction = _settled_rows(panel, SIMFIT, fmm, nyi)
    oos = fit_scores[fit_scores["split"].isin(["calib", "deploy"])]
    oos = attach_pnl(attach_settlement(oos, fmm, nyi), "premium", direction)
    return {k: fit_node(g) for k, g in oos.groupby(["market", "node"])}


def fit_hb_params(panel, block="DART3"):
    """The same for H-B, from the scores of ``block`` (by default the plain-DART-lags block, the base block of both
    contrasts), out-of-sample rows only; H-B reads no P&L."""
    hb_scores = cached_scores(panel, SIMFIT, block, "xgb", 42)
    hb_oos = hb_scores[hb_scores["split"].isin(["calib", "deploy"])].assign(pnl_open=0.0)
    return {k: fit_node(g) for k, g in hb_oos.groupby(["market", "node"])}


def fit_hb_laws(panel):
    """{law: {(market, node): NodeParams}} for the three laws of ``HB_LAWS`` (Ruling 84b)."""
    return {law: fit_hb_params(panel, law) for law in HB_LAWS}


def _phi_medians(params_by_node):
    """Median over nodes of the score's fitted persistence (Ruling 84b item 6): ``phi_score_median`` is the ``phi_s``
    the simulation draws with (corrected by ``fit_node`` for what the fit does to a daily component),
    ``phi_score_raw_median`` the plain AR(1) estimate it had before the correction, ``NodeParams.phi_s_raw`` (the ruling
    called it ``phi_raw``, which is read when ``phi_s_raw`` is absent). NaN when no fit carries one."""
    def median(*attrs):
        values = []
        for p in params_by_node.values():
            v = next((x for x in (getattr(p, a, None) for a in attrs) if x is not None), None)
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(v):
                values.append(v)
        return float(np.median(values)) if values else float("nan")

    return {"phi_score_median": median("phi_s"), "phi_score_raw_median": median("phi_s_raw", "phi_raw")}


def phi_status_counts(params_by_node, components):
    """How the persistence correction ended for the nodes of one block (Ruling 94, N-2). ``components`` are the daily
    components the block's simulation draws (``HE_COMPONENTS`` or ``HB_COMPONENTS``).

    Returns ``{"components", "phi_nodes", "phi_corrected", "phi_clipped_high", "phi_clipped_low", "phi_no_signal",
    "phi_uncorrected", "by_component"}``. Each ``phi_<status>`` is the number of nodes that have the status in at least
    one of the components, so "nodes clipped high" is a count of nodes whichever component was clipped; with two
    components a node can be in two statuses and the counts add up to more than ``phi_nodes``. ``by_component`` counts
    each component on its own. A node with no recorded status (a hand-built ``NodeParams``) is ``uncorrected``: no
    correction ran. Raises ValueError for a status ``fit_node`` does not make and TypeError for a ``phi_status`` that
    is not a dict."""
    by_component = {c: dict.fromkeys(PHI_STATUSES, 0) for c in components}
    nodes = dict.fromkeys(PHI_STATUSES, 0)
    for key, params in params_by_node.items():
        recorded = getattr(params, "phi_status", None) or {}
        if not isinstance(recorded, dict):
            raise TypeError(f"phi_status_counts: node {key}: phi_status must be a dict {{component: status}}, got "
                            f"{recorded!r}")
        statuses = set()
        for component in components:
            status = recorded.get(component, "uncorrected")
            if status not in PHI_STATUSES:
                raise ValueError(f"phi_status_counts: node {key}: unknown {component} status {status!r}; "
                                 f"fit_node makes {', '.join(PHI_STATUSES)}")
            by_component[component][status] += 1
            statuses.add(status)
        for status in statuses:
            nodes[status] += 1
    return {"components": list(components), "phi_nodes": len(params_by_node),
            **{f"phi_{s}": nodes[s] for s in PHI_STATUSES}, "by_component": by_component}


def clipped_high_note(counts):
    """The note for every row of a block with more than ``CLIPPED_HIGH_SHARE`` of its nodes clipped high, from
    ``phi_status_counts``; None for a block at or below the share (or without nodes)."""
    nodes, high = counts["phi_nodes"], counts["phi_clipped_high"]
    if not nodes or high / nodes <= CLIPPED_HIGH_SHARE:
        return None
    return (f"{high} of {nodes} nodes ({high / nodes:.0%}) are clipped high: the fitted persistence of their "
            f"{' or '.join(counts['components'])} lay above what the correction can reach and the top of its search "
            "range was used, so this row is not a bound in a known direction")


def _phi_row(row, counts):
    """``row`` with the status counts of its block and, for a flagged block, the note appended to any it has."""
    out = {**row, **{k: v for k, v in counts.items() if k.startswith("phi_")}}
    note = clipped_high_note(counts)
    if note:
        out["note"] = f"{out['note']}; {note}" if out.get("note") else note
    return out


def he_size_rows(params, windows, counts):
    """H-E's size matrix (appendix B): single quarter A and pooled A + B, each at the fitted and the upper-end
    persistence (Ruling 63), with pooled and fitted heteroskedastic P&L noise (Ruling 55), under the primary and the
    decile strata (Ruling 68): 16 rows, no timing skill (rho = 0). The decile rows are reported and never govern the
    registration. Row i has the seed SIZE_SEED + i (Ruling 94, I2), recorded in the row. Every row carries the status
    counts of the fit's persistence correction (``phi_status_counts``) and, past ``CLIPPED_HIGH_SHARE``, the note that
    it is no bound in a known direction (N-2)."""
    phi = phi_status_counts(params, HE_COMPONENTS)
    rows = []
    for test, cal, dep in (("H-E single A, size", [windows.cal_a], [windows.dep_a]),
                           ("H-E pooled A+B, size", [windows.cal_a, windows.cal_b], [windows.dep_a, windows.dep_b])):
        for persistence in PERSISTENCE:
            for scenario, hetero in (("homoskedastic", False), ("heteroskedastic", True)):
                for strata in ("primary", "decile"):
                    seed = SIZE_SEED + len(rows)
                    res = he_size_power(params, cal, dep, 0.0, counts.he_size, seed=seed, hetero=hetero,
                                        persistence=persistence, strata=strata)
                    rows.append(_phi_row({"test": test, "scenario": scenario, "seed": seed, **res}, phi))
    if len(rows) != HB_SIZE_FIRST_INDEX:
        raise RuntimeError(f"he_size_rows made {len(rows)} rows but the H-B size seeds start at {HB_SIZE_FIRST_INDEX}")
    return rows


def he_power_rows(params, windows, counts, target):
    """H-E's power as a curve and as a target row, and the coupling rho* behind the latter (Rulings 63 and 92b).

    The registration never reads H-E's power (H-E stays in whatever its power), so it is reported over the whole rho
    grid: one row "H-E pooled A+B, power curve" per grid point, with the rejection rates at alpha/3 and 0.05 and the
    mean SELC of ``counts.he_rho`` simulations (the rho search itself, so the curve costs no extra run). Then the single
    target row "H-E pooled A+B, power": rho* is the grid point whose mean SELC is nearest ``target``, rerun with
    ``counts.he_power`` simulations. If no grid point reaches ``target``, rho* is 0.9 and the row's note says the target
    is not bracketed. A target below every grid point also gets the nearest point (the lowest) and a note, since the
    power read there is that of the weakest coupling, not of the target. Every row carries the fit's status counts and
    the clipped-high note (N-2). Returns (the curve rows followed by the target row, rho*)."""
    cal, dep = [windows.cal_a, windows.cal_b], [windows.dep_a, windows.dep_b]
    phi = phi_status_counts(params, HE_COMPONENTS)
    runs = [he_size_power(params, cal, dep, r, counts.he_rho, seed=SEED_RHO) for r in RHO_GRID]
    means = {r: run["mean_selc"] for r, run in zip(RHO_GRID, runs)}
    curve_text = " | ".join(f"rho {r:g}: SELC {run['mean_selc']:.4f}, power {run['reject_0.0166667']:.2f} at alpha/3, "
                            f"{run['reject_0.05']:.2f} at 0.05" for r, run in zip(RHO_GRID, runs))
    print(f"  rho search: {curve_text}; target SELC {target:.4f}", flush=True)
    note = None
    if max(means.values()) < target:
        rho_star, note = RHO_GRID[-1], "target not bracketed"
    else:
        rho_star = min(RHO_GRID, key=lambda r: abs(means[r] - target))
        if target < min(means.values()):
            note = "target below the grid"
    curve = [{"test": "H-E pooled A+B, power curve", "scenario": "homoskedastic", "seed": SEED_RHO, **run}
             for run in runs]
    res = he_size_power(params, cal, dep, rho_star, counts.he_power, seed=SEED_HE_POWER)
    row = {"test": "H-E pooled A+B, power", "scenario": "homoskedastic", "target_selc": target, "seed": SEED_HE_POWER,
           **res}
    if note:
        row["note"] = note
    return [_phi_row(r, phi) for r in curve + [row]], rho_star


def hb_size_rows(laws, windows, counts, first_index=HB_SIZE_FIRST_INDEX):
    """H-B's size matrix (Rulings 63, 84b and 94): for each law of ``HB_LAWS`` the two blocks are drawn from that
    block's SIMFIT fit (equal portability), at the fitted and the upper-end persistence, with no drift and with a
    common drift of 0.25 and of 0.5 score SDs in both blocks: 3 x 2 x 3 = 18 rows of ``counts.hb_size`` simulations of
    ``counts.hb_rep`` bootstrap replicates. Every row carries its ``law`` and that law's fitted persistence
    (``_phi_medians``), the status counts of that law's persistence corrections and, past ``CLIPPED_HIGH_SHARE``, the
    clipped-high note (N-2), and its own seed SIZE_SEED + ``first_index`` + its position (Ruling 94, I2)."""
    rows = []
    for law in HB_LAWS:
        phi, status = _phi_medians(laws[law]), phi_status_counts(laws[law], HB_COMPONENTS)
        for persistence in PERSISTENCE:
            for scenario, shift in HB_DRIFTS:
                seed = SIZE_SEED + first_index + len(rows)
                res = hb_size(laws[law], windows.cal_a, windows.dep_a, counts.hb_size, counts.hb_rep, seed=seed,
                              persistence=persistence, common_shift=shift)
                rows.append(_phi_row({"test": "H-B size", "law": law, "scenario": scenario, "seed": seed, **res, **phi},
                                     status))
    return rows


def _aq3_level_gap(panel, block):
    """Half the AQ3 median over nodes of |log10 R|(block) - |log10 R|(plain DART lags) under P1: the registered H-B
    effect (Ruling 84b item 5: the median, the scale of the test's own statistic, not the mean)."""
    alt = activity(p1_fixed(cached_scores(panel, ANALOGS["AQ3"], block, "xgb", 42), 0.05), 0.05)
    dar = activity(p1_fixed(cached_scores(panel, ANALOGS["AQ3"], "DART3", "xgb", 42), 0.05), 0.05)
    both = alt.merge(dar, on=["market", "node"], suffixes=("_alt", "_dart"))
    return 0.5 * float((both["log_error_alt"] - both["log_error_dart"]).median())


def hb_power_rows(panel, laws, windows, counts):
    """One power row per H-B test, both blocks drawn from DART3's fit: ``calibrate_delta`` finds the alternative
    block's shift whose simulated gap is the registered effect (``_aq3_level_gap``), and the power is simulated at that
    delta (not tuned by hand). The row records the target, the calibrated delta and the simulated gaps; a note says so
    if the simulated gap on the target's scale, ``mean_t_hat`` (the mean over simulations of the median over nodes,
    which is what ``calibrate_delta`` matches), is more than a factor of 2 from the target.

    If the calibration raises ValueError the target is outside what the grid reaches (Ruling 94, M4). Which side is
    read from the draw that raised, not from another (NB1): ``hb_size`` at delta = ``GRID_TOP`` with the calibration's
    own seed and simulation count is the gap ``calibrate_delta`` measured there (the same simulations). Above the grid
    (that gap stays below the target: a very large effect) the row records the power of the usual power run at delta =
    ``GRID_TOP`` as ``power_lower_bound`` with the calibration's message, and the test is not demoted on power. Below
    it, or for a target that is not finite, the row has an ``error`` and no rejection rates (and no power run is made),
    and ``registration_family`` counts its power as below the floor. Every row, an error row too, carries the status
    counts of DART3's persistence corrections (both blocks are drawn from them) and the clipped-high note (N-2)."""
    hb_params = laws[HB_BASE_LAW]
    phi, status = _phi_medians(hb_params), phi_status_counts(hb_params, HB_COMPONENTS)
    rows = []

    def power_run(delta):
        return hb_size(hb_params, windows.cal_a, windows.dep_a, counts.hb_power, counts.hb_rep, delta=delta,
                       seed=SEED_HB_POWER)

    def calibration_gap_at_top():
        """The simulated gap at the grid's top as ``calibrate_delta`` measured it: its mean over simulations of the
        median gap is what ``hb_size`` reports as ``mean_t_hat`` for the same arguments and seed (NB1)."""
        return hb_size(hb_params, windows.cal_a, windows.dep_a, counts.hb_calibrate, counts.hb_rep, delta=GRID_TOP,
                       seed=SEED_CALIBRATE)["mean_t_hat"]

    for hid, block in HB_PAIRS:
        gap = _aq3_level_gap(panel, block)
        base = {"test": f"{hid} power", "law": HB_BASE_LAW, "scenario": "no_drift", "target_log_error_gap": gap, **phi}
        seeds = {"seed": SEED_HB_POWER, "calibration_seed": SEED_CALIBRATE}
        try:
            cal = calibrate_delta(hb_params, windows.cal_a, windows.dep_a, gap, grid=CALIBRATION_GRID,
                                  n_sim=counts.hb_calibrate, n_rep=counts.hb_rep, seed=SEED_CALIBRATE)
        except ValueError as exc:
            if math.isfinite(gap) and calibration_gap_at_top() < gap:            # above the grid: a very large effect
                rows.append({**base, **seeds, "power_lower_bound": True, "calibration_error": str(exc),
                             "note": f"target above the grid: the power at delta = {GRID_TOP:g} is a lower bound on "
                                     "the power at the target", **power_run(GRID_TOP)})
            else:
                rows.append({**base, "persistence": "fit", "error": str(exc)})
            continue
        res = power_run(cal["delta"])
        row = {**base, **seeds, "calibrated_delta": cal["delta"], **res}
        simulated = res["mean_t_hat"]
        ratio = simulated / gap if gap else float("inf")
        if not 0.5 <= ratio <= 2.0:
            row["note"] = (f"simulated median gap {simulated:.4f} differs from the target {gap:.4f} "
                           "by more than a factor of 2")
        rows.append(row)
    return [_phi_row(r, status) for r in rows]


# ---------------------------------------------------------------------------------------------------------------
# The registration rule (spec v3 §6.2; controller Rulings 29, 63, 68, 84b, 92b and 94)
# ---------------------------------------------------------------------------------------------------------------

#: The two levels every size is read at: the first Holm level of Test A (alpha / 3) and the family-wise level.
NOMINAL = (0.05 / 3, 0.05)
#: H-B1 and H-B2: the hypothesis and the block whose level model is contrasted with plain DART lags (DART3).
HB_PAIRS = (("H-B1", "LEV6"), ("H-B2", "ROLL_DISP9"))
#: Level of the one-sided binomial test of a rejection count against nominal (Ruling 63).
BINOMIAL_LEVEL = 0.05
#: Placebo pairings named like this are the pooled A + B ones; every other pairing is a single quarter.
POOLED_PREFIX = "AQ3+AQ4"
#: Ruling 92b: a test is labelled approximate when its estimated size is significantly above nominal, which is the same
#: combined condition as the demotion's with this factor (rate > a, and the binomial tail below 0.05 / rows).
APPROXIMATE_FACTOR = 1.0
#: The demotion factor (Ruling 29) and the power floor of an H-B test at alpha / 3.
FACTOR = 1.5
MIN_POWER = 0.5


def _hb_laws(block):
    """The laws whose size rows are nulls of the H-B test of ``block`` (Ruling 94, M3): the base law and its own."""
    return (HB_BASE_LAW, block)


def _binom_upper(k, n, p):
    """P(X >= k) for X ~ Binomial(n, p): the one-sided tail of ``k`` rejections in ``n`` against the rate ``p``."""
    return float(sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(max(k, 0), n + 1)))


def _oversized(rate, n, a, factor, rows=1):
    """Ruling 63: a size check demotes a test only when the estimated rejection rate exceeds ``factor * a`` AND a
    one-sided binomial test of the rejection count k = round(rate * n) against the nominal ``a`` rejects at 0.05.
    The second condition is what keeps a correctly sized test from being demoted by Monte Carlo noise at the replication
    counts used; the first is what keeps a test that is only slightly oversized, and significantly so, in.

    Ruling 84b (M-6): the rate is the worst of ``rows`` simulation rows, so the binomial test must reject at 0.05 /
    ``rows`` (Bonferroni over the rows entering the maximum); the rate condition is unchanged. A placebo count is not
    a maximum over rows: ``rows`` stays 1."""
    n = int(n)
    return bool(n > 0 and rate > factor * a and _binom_upper(round(rate * n), n, a) < BINOMIAL_LEVEL / rows)


@lru_cache(maxsize=None)
def _smallest_demoting_count(n, a, factor, rows=1):
    """The least number of rejections among ``n`` simulations (or placebo pairings) at level ``a`` that ``_oversized``
    calls oversized, with ``rows`` rows entering the maximum, or None when no count of them is."""
    return next((k for k in range(1, n + 1) if _oversized(k / n, n, a, factor, rows)), None)


def _test_rows(sim, test, laws=None):
    """The rows named ``test`` that the registration rule reads. Rows under the decile strata are the sensitivity of
    Ruling 68: they are reported and never count, so a status rests on the primary-strata rows (and on rows that carry
    no strata at all, H-B's). With ``laws`` only the rows of those laws are read (a row without a law is read by every
    test): H-B1 reads the DART3 and LEV6 rows, H-B2 the DART3 and ROLL_DISP9 rows (Ruling 94, M3). Raises ValueError if
    there is no such row."""
    rows = sim[sim["test"] == test] if "test" in sim.columns else sim.iloc[0:0]
    if "strata" in rows.columns:
        rows = rows[rows["strata"].isna() | (rows["strata"] == "primary")]
    if laws is not None and "law" in rows.columns:
        rows = rows[rows["law"].isna() | rows["law"].isin(laws)]
    if rows.empty:
        raise ValueError(f"simulation has no '{test}' row" + (" under the primary strata" if "strata" in sim.columns
                                                              else ""))
    return rows


def _worst(sim, test):
    """The highest simulated rejection rate of ``test`` at each nominal level, over the rows the rule reads (a plain
    maximum, for reporting: the labels and the demotion apply the combined condition to the rows themselves)."""
    rows = _test_rows(sim, test)
    return {a: float(rows[f"reject_{a:g}"].max()) for a in NOMINAL}


def _where(row):
    """'law, scenario, persistence' of a simulation row, as far as it carries them."""
    return ", ".join(str(row[c]) for c in ("law", "scenario", "persistence") if c in row.index and pd.notna(row[c]))


def _size_reasons(sim, test, factor, laws=None):
    """One reason per nominal level at which some row of ``test`` is oversized (``_oversized``): the worst such row. The
    rows are the maximum's: the binomial check of each is made at 0.05 / their number (Ruling 84b, M-6). A NaN
    rejection rate raises ValueError (Ruling 94, M5): a size check must not pass on a missing number."""
    rows = _test_rows(sim, test, laws)
    if "n_sim" not in rows.columns or rows["n_sim"].isna().any():
        raise ValueError(f"the simulation rows of '{test}' need n_sim: the binomial check of Ruling 63 needs the "
                         "number of replications behind each rate")
    k_rows = len(rows)
    reasons = []
    for a in NOMINAL:
        col = f"reject_{a:g}"
        rates = rows[col].astype(float)
        if rates.isna().any():
            raise ValueError(f"the simulation row of '{test}' ({_where(rows[rates.isna()].iloc[0]) or 'no label'}) has "
                             f"a NaN rejection rate at {a:.4f}: a size check cannot pass on a missing number")
        bad = [r for _, r in rows.iterrows() if _oversized(float(r[col]), r["n_sim"], a, factor, k_rows)]
        if bad:
            w = max(bad, key=lambda r: float(r[col]))
            n, where = int(w["n_sim"]), _where(w)
            detail = ", ".join(x for x in (f"{round(float(w[col]) * n)} of {n} replications",
                                           f"worst of {k_rows} rows" if k_rows > 1 else "") if x)
            reasons.append(f"simulated size {float(w[col]):.3f} > {factor} x {a:.4f} "
                           f"({detail}{': ' + where if where else ''})")
    return reasons


def _size_bound(sim, test, factor, laws=None):
    """The probability, under independent rows, that a correctly sized test has some row of ``test`` oversized
    (Ruling 94, M7): per level 1 - the product over the rows of (1 - the tail of the smallest oversized count), summed
    over the two levels (a bound, since both levels read the same rows)."""
    rows = _test_rows(sim, test, laws)
    k_rows = len(rows)
    total = 0.0
    for a in NOMINAL:
        none = 1.0
        for n in rows["n_sim"]:
            k = _smallest_demoting_count(int(n), a, factor, k_rows)
            if k is not None:
                none *= 1.0 - _binom_upper(k, int(n), a)
        total += 1.0 - none
    return min(total, 1.0)


def _placebo_bound(n, factor):
    """The same bound for the pooled placebo check on ``n`` counted pairings: the sum over the two levels of the tail of
    the smallest demoting count (0 for n = 0). For the one registered 1-year pairing it is alpha/3."""
    total = 0.0
    for a in NOMINAL:
        k = _smallest_demoting_count(n, a, factor) if n else None
        if k is not None:
            total += _binom_upper(k, n, a)
    return min(total, 1.0)


def _placebo_reasons(rows, factor):
    """The counted placebo pairings (rows with a p) whose rejection counts are oversized, by the same condition."""
    reasons = []
    for a in NOMINAL:
        k = sum(r["p"] <= a for r in rows)
        if k and _oversized(k / len(rows), len(rows), a, factor):
            reasons.append(f"{k}/{len(rows)} placebo rejections at {a:.4f}")
    return reasons


def _counted(row):
    """Whether a placebo row is counted by the rule (Ruling 94, C1): its own flag, or for a row written without one the
    pairing's name (the 1-year pairings count)."""
    flag = row.get("counted")
    if flag is not None and pd.notna(flag):
        return bool(flag)
    return str(row.get("pair", "")).endswith("-1y")


def _placebo_notes(pooled, factor):
    """What the pooled placebo check does to a correctly sized test (Ruling 94, M7). It rests on the counted 1-year
    pairings: for the one registered, AQ3+AQ4-1y, H-E is demoted if and only if its p is at most alpha/3, which a
    correctly sized test does with probability at most 0.0167, and nothing can demote at 0.05 (a single rejection has
    probability 0.05, not below it). The 2-year pairings are reported and do not enter. The check is retained as
    registered."""
    n = len(pooled)
    uncounted = ("The 2-year pairings (their P&L quarter lies inside the alert model's training window) are reported "
                 "with counted = false and do not enter the rule.")
    if n == 0:
        return ["The pooled placebo check did not run: none of its counted 1-year pairings has a p-value, so only the "
                f"simulation governs H-E's status. {uncounted}"]
    demote, cannot = [], []
    for a in NOMINAL:
        k = _smallest_demoting_count(n, a, factor)
        if k is None:
            cannot.append(f"{a:.4f}")
        else:
            when = f"its p is at most {a:.4f}" if n == 1 else f"{k} or more of its {n} pairings reject at {a:.4f}"
            demote.append(f"{when} (false-demotion probability under a correctly sized test ≤ "
                          f"{_binom_upper(k, n, a):.3g})")
    text = f"Pooled placebo check ({n} counted 1-year pairing{'s' if n != 1 else ''}): "
    text += ("H-E is replaced by its estimand when " + " or when ".join(demote) + "." if demote
             else "no count of rejections replaces H-E.")
    if cannot and demote:
        text += f" It cannot demote at {', '.join(cannot)}."
    return [f"{text} {uncounted} The check is retained as registered."]


def _power(sim, hid):
    """(power, errors, lower_bound_only) of H-B test ``hid``. ``power`` is its simulated power at alpha / 3, the lowest
    over its power rows, NaN if any of them has no rejection rate (a failed calibration, Ruling 63). A row marked
    ``power_lower_bound`` is the power at the top of the calibration grid for a target above it (Ruling 94, M4): it is
    left out of that minimum, and a test whose power rows are all such bounds is not demoted on power
    (``lower_bound_only``). A NaN rate with no error text is a bug and raises ValueError (M5)."""
    rows = _test_rows(sim, f"{hid} power")
    col = f"reject_{NOMINAL[0]:g}"
    values = rows[col].to_numpy(dtype=float) if col in rows.columns else np.full(len(rows), np.nan)
    lower = (rows["power_lower_bound"].eq(True).to_numpy() if "power_lower_bound" in rows.columns
             else np.zeros(len(rows), dtype=bool))
    has_error = rows["error"].notna().to_numpy() if "error" in rows.columns else np.zeros(len(rows), dtype=bool)
    if (np.isnan(values) & ~has_error).any():
        raise ValueError(f"a row of '{hid} power' has a NaN rejection rate and no error text: a power that failed must "
                         "say why")
    errors = "; ".join(map(str, rows["error"].dropna())) if "error" in rows.columns else ""
    normal = values[~lower]
    if normal.size:
        return (float("nan") if np.isnan(normal).any() else float(normal.min())), errors, False
    return float(values.min()), errors, True


def registration_family(sim_rows, placebo_rows, factor=FACTOR, min_power=MIN_POWER):
    """Spec v3 §6.2 registration rule, as read in controller Rulings 29, 63, 68, 84b, 92b and 94.

    A test is replaced by its estimand when its size is oversized (``_oversized``: estimated rate above ``factor`` x
    nominal and a one-sided binomial test of its rejection count against nominal rejecting at 0.05, at 0.05 / k for the
    worst of k simulation rows), in the simulation or in the placebo pairings; an H-B test also when its simulated
    power at 0.05/3 is below ``min_power`` (H-E stays in whatever its power, which is never read; a power that is only a
    lower bound never demotes). The simulation size is the worst over the rows of the test: H-E's over its scenarios and
    persistence settings under the primary strata only; H-B1's over the DART3 and LEV6 laws and H-B2's over the DART3
    and ROLL_DISP9 laws, with their persistence settings and drifts (k = 12 each).

    Only the counted placebo pairings are read, the 1-year ones (Ruling 94, C1; the 2-year P&L quarters lie inside the
    alert model's training window): the pooled AQ3+AQ4-1y (n = 1) demotes H-E if and only if its p is at most alpha/3;
    the four single quarters only label Test A's interim p-value (2 of 4 at either level). A placebo row without a p
    (a pairing that failed, or NaN from a CSV) counts as nothing; ``run_sim_stage`` refuses to run without them.

    ``approximate`` (and ``interim_approximate``) is size significantly above nominal (Ruling 92b): the same combined
    condition with factor 1.0, Bonferroni over the rows of the test, so a correctly sized test is not labelled; H-E's is
    also true when the placebo check demotes it (M8). Each hypothesis carries its ``k_rows`` and its
    ``false_demotion_bound``; ``notes`` states the pooled check's and the bounds. Raises ValueError for a missing
    simulation row, a size row without n_sim, and a NaN rejection rate."""
    sim = pd.DataFrame(sim_rows)
    counted = [r for r in placebo_rows if pd.notna(r.get("p")) and _counted(r)]
    pooled = [r for r in counted if str(r["pair"]).startswith(POOLED_PREFIX)]
    single = [r for r in counted if not str(r["pair"]).startswith(POOLED_PREFIX)]
    he, he_single = "H-E pooled A+B, size", "H-E single A, size"
    placebo_reasons = _placebo_reasons(pooled, factor)
    reasons = _size_reasons(sim, he, factor) + placebo_reasons
    interim = bool(_size_reasons(sim, he_single, APPROXIMATE_FACTOR) or _placebo_reasons(single, APPROXIMATE_FACTOR))
    out = {"H-E": {"confirmatory": not reasons, "reasons": reasons,
                   "approximate": bool(_size_reasons(sim, he, APPROXIMATE_FACTOR)) or bool(placebo_reasons),
                   "interim_approximate": interim, "k_rows": len(_test_rows(sim, he)),
                   "k_rows_interim": len(_test_rows(sim, he_single)),
                   "placebo_counted": {"single": len(single), "pooled": len(pooled)},
                   "false_demotion_bound": min(1.0, _size_bound(sim, he, factor)
                                               + _placebo_bound(len(pooled), factor))}}
    for hid, block in HB_PAIRS:
        laws = _hb_laws(block)
        reasons = _size_reasons(sim, "H-B size", factor, laws)
        power, errors, lower_only = _power(sim, hid)
        if not lower_only and not power >= min_power:                       # a NaN power is below the floor
            reasons.append(f"simulated power is unavailable ({errors or 'no rejection rate'}), counted as below "
                           f"{min_power}" if np.isnan(power) else
                           f"simulated power {power:.2f} at 0.05/3 is below {min_power}")
        out[hid] = {"confirmatory": not reasons, "reasons": reasons, "power": power, "power_lower_bound": lower_only,
                    "approximate": bool(_size_reasons(sim, "H-B size", APPROXIMATE_FACTOR, laws)),
                    "k_rows": len(_test_rows(sim, "H-B size", laws)),
                    "false_demotion_bound": _size_bound(sim, "H-B size", factor, laws)}
    bounds = (f"False-demotion bounds under independent rows, for a correctly sized test (union over the two levels): "
              f"H-E {out['H-E']['false_demotion_bound']:.3f} (simulation, {out['H-E']['k_rows']} rows, "
              f"{_size_bound(sim, he, factor):.3f}; placebo {_placebo_bound(len(pooled), factor):.4f}), "
              f"H-B1 {out['H-B1']['false_demotion_bound']:.3f} ({out['H-B1']['k_rows']} rows), "
              f"H-B2 {out['H-B2']['false_demotion_bound']:.3f} ({out['H-B2']['k_rows']} rows).")
    out["notes"] = _placebo_notes(pooled, factor) + [bounds]
    return out


# ---------------------------------------------------------------------------------------------------------------
# Files, stages and the command line (Ruling 77)
# ---------------------------------------------------------------------------------------------------------------

STAGES = ("analogs", "sim")


def _csv(frame):
    return frame.to_csv(index=False, lineterminator="\n")


def _jsonable(obj):
    """``obj`` as strict-JSON data: numpy scalars as Python ones, NaN and infinity as None, dates as ISO strings."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (dt.date, pd.Timestamp)):
        return obj.isoformat()
    return obj


def _json(obj):
    return json.dumps(_jsonable(obj), indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n"


def _publish(files, stale=()):
    """Write ``files`` (path -> text) so that none appears half-written and none at all if one cannot be written: every
    file goes to a temporary sibling first, then ``os.replace`` moves each into place, in the order given (callers put
    ``protocol_inputs.json``, which carries the stage marker, last). The ``stale`` paths are removed once everything is
    staged and before the first replace, so a file of an earlier run never survives beside the new ones (Ruling 94,
    M6); a failure while staging removes nothing."""
    staged = {}
    try:
        for path, text in files.items():
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(text, encoding="utf-8", newline="\n")
            staged[path] = tmp
        for path in stale:
            path.unlink(missing_ok=True)
        for path, tmp in staged.items():
            os.replace(tmp, path)
    finally:
        for tmp in staged.values():
            tmp.unlink(missing_ok=True)


def _missing(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


def _present(row):
    """``row`` without its missing values (a column that does not apply to it)."""
    return {k: v for k, v in row.items() if not _missing(v)}


def _placebo_records(frame):
    """The rows of ``placebo_he.csv`` as the dicts ``placebo_table`` made: no NaN for a column that did not apply
    (a failed pairing has an error and no p), the counts ints and ``counted`` a bool."""
    rows = []
    for rec in frame.to_dict(orient="records"):
        rec = _present(rec)
        for column in ("n_null", "n_rows", "n_settle_approx"):
            if column in rec:
                rec[column] = int(rec[column])
        if "counted" in rec:
            rec["counted"] = bool(rec["counted"])
        rows.append(rec)
    return rows


def _read_stage_one():
    """(eta, placebo rows) as stage ``analogs`` wrote them. Raises FileNotFoundError, naming the command to run, when
    they are not there, and ValueError for a ``protocol_inputs.json`` of another stage."""
    inputs, placebo = ANALOG_DIR / "protocol_inputs.json", ANALOG_DIR / "placebo_he.csv"
    absent = [str(f) for f in (inputs, placebo) if not f.exists()]
    if absent:
        raise FileNotFoundError(f"stage 'sim' reads the output of stage 'analogs', but {', '.join(absent)} is missing: "
                                "run `python -m es.run_analogs --stage analogs` first")
    meta = json.loads(inputs.read_text())
    if meta.get("stage") not in STAGES[:1] + ("complete",):
        raise ValueError(f"{inputs}: its stage is {meta.get('stage')!r}, not 'analogs' or 'complete'")
    return float(meta["eta"]), _placebo_records(pd.read_csv(placebo, float_precision="round_trip"))


def _require_counted_placebo(placebo):
    """Raises ValueError unless every counted pairing (``COUNTED_PAIRINGS``) has a finite p (Ruling 94, I1). A counted
    pairing that failed would silently remove a registered check (or leave the pooled one with nothing to test), and H-E
    would be registered on the simulation alone. The 2-year pairings are descriptive and may fail."""
    by_pair = {r["pair"]: r for r in placebo}
    bad = []
    for pair in COUNTED_PAIRINGS:
        row = by_pair.get(pair)
        if row is None:
            bad.append(f"{pair} (no row)")
        elif not (pd.notna(row.get("p")) and math.isfinite(float(row["p"]))):
            bad.append(f"{pair} ({row.get('error', 'no finite p')})")
    if bad:
        raise ValueError("stage 'sim' needs a finite p for every counted placebo pairing (the four 1-year quarters and "
                         "the pooled AQ3+AQ4-1y), but: " + "; ".join(bad) + ". Fix the cause and rerun "
                         "`python -m es.run_analogs --stage analogs`.")


def _stage_one_protocol(eta, placebo):
    """The ``protocol_inputs.json`` of stage ``analogs``: eta and the placebo rows, no family."""
    return {"eta": eta, "family": None, "stage": "analogs", "placebo": placebo}


def _current_stage():
    return json.loads((ANALOG_DIR / "protocol_inputs.json").read_text()).get("stage")


def run_analogs_stage(panel, fmm, nyi):
    """Stage ``analogs``: the activity table, eta, the placebo pairings, and a provisional ``protocol_inputs.json``
    (family null). Writes its four files at the end and returns the protocol inputs. Writing removes the
    ``simulation.csv`` and the ``protocol_inputs.json`` of any earlier run first (Ruling 94, M6), and puts the marker
    last: stage ``sim`` must be run again after this one."""
    t0 = time.time()
    acts = activity_table(panel)
    print(f"[analogs] activity: {len(acts)} rows, {time.time() - t0:.0f} s", flush=True)
    eta_tab = eta_table(panel)
    eta_star = select_eta(eta_tab)
    print(f"[analogs] eta* = {eta_star:g} ({time.time() - t0:.0f} s)\n"
          + eta_tab.pivot(index="analysis", columns="eta", values="median_log_error").round(4).to_string(), flush=True)
    placebo = placebo_table(panel, fmm, nyi)
    for r in placebo:
        print("[analogs] placebo " + ", ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
                                               for k, v in r.items()), flush=True)
    have_p = {r["pair"] for r in placebo if pd.notna(r.get("p"))}
    n_ok = sum(pair in have_p for pair in COUNTED_PAIRINGS)
    print(f"[analogs] counted placebo pairings with a p: {n_ok} of {len(COUNTED_PAIRINGS)}"
          + ("" if n_ok == len(COUNTED_PAIRINGS) else "  <-- stage sim will refuse to run until they all have one"),
          flush=True)
    protocol = _stage_one_protocol(eta_star, placebo)
    marker = ANALOG_DIR / "protocol_inputs.json"
    _publish({ANALOG_DIR / "activity_all.csv": _csv(acts), ANALOG_DIR / "eta_selection.csv": _csv(eta_tab),
              ANALOG_DIR / "placebo_he.csv": _csv(pd.DataFrame(placebo)), marker: _json(protocol)},
             stale=(ANALOG_DIR / "simulation.csv", marker))
    return protocol


def _lead(frame, first=("test", "law", "scenario", "persistence", "strata", "n_sim", "seed", "reject_0.0166667",
                        "reject_0.05")):
    lead = [c for c in first if c in frame.columns]
    return frame[lead + [c for c in frame.columns if c not in lead]]


def _rule_record(counts, family):
    """Every constant of the registered rule and every seed of the study, as this run used them (Ruling 94, M10)."""
    return {"factor": FACTOR, "binomial_level": BINOMIAL_LEVEL, "approximate_factor": APPROXIMATE_FACTOR,
            "min_power": MIN_POWER, "levels": list(NOMINAL), "clipped_high_share": CLIPPED_HIGH_SHARE,
            "k_rows": {hid: family[hid]["k_rows"] for hid in ("H-E", "H-B1", "H-B2")},
            "k_rows_interim": family["H-E"]["k_rows_interim"],
            "laws": {hid: list(_hb_laws(block)) for hid, block in HB_PAIRS},
            "n_rep": counts.hb_rep, "n_sim": {k: v for k, v in asdict(counts).items() if k != "hb_rep"},
            "seeds": {"size_base": SIZE_SEED, "he_size_first_index": 0, "hb_size_first_index": HB_SIZE_FIRST_INDEX,
                      "rho_search": SEED_RHO, "he_power": SEED_HE_POWER, "hb_power": SEED_HB_POWER,
                      "calibration": SEED_CALIBRATE},
            "calibration_grid": list(CALIBRATION_GRID), "counted_pairings": list(COUNTED_PAIRINGS)}


def run_sim_stage(panel, fmm, nyi):
    """Stage ``sim``: reads eta and the placebo rows back from stage ``analogs`` (and refuses to run unless every
    counted pairing has a p, Ruling 94 I1), runs the size and power study (``SIM_COUNTS`` replications on
    ``SIM_WINDOWS``), computes the registered family from its rows and the placebo rows, and writes ``simulation.csv``
    and the complete ``protocol_inputs.json`` with the rule's constants and seeds. The effect target of the H-E power
    is half the loss-weighted pooled SELC of AQ3 and AQ4 (M12). The file also holds, per block (H-E and the three H-B
    laws), how its nodes' persistence corrections ended (``phi_status``, N-2), the same counts as the rows carry. A kill
    cannot leave the old family beside new files:
    the marker is first put back to stage 1's form, then the old simulation.csv is removed as the new files go in,
    marker last (M6). Returns the protocol inputs."""
    eta, placebo = _read_stage_one()
    _require_counted_placebo(placebo)
    counts, windows = SIM_COUNTS, SIM_WINDOWS
    t0 = time.time()

    def stamp(what, rows):
        print(f"[sim] {what}: {len(rows)} row(s), {time.time() - t0:.0f} s", flush=True)

    parts = analog_selc_parts(panel, fmm, nyi)
    analog = [float(e / l) if l > 0 else 0.0 for e, l in parts]
    pooled = pooled_selc(parts)
    target = 0.5 * pooled                                                 # the conservative effect (review N7, M12)
    print(f"[sim] SELC of P1 on AQ3 and AQ4: {analog}; loss-weighted pooled {pooled:.4f}; effect target {target:.4f}",
          flush=True)
    params, laws = fit_he_params(panel, fmm, nyi), fit_hb_laws(panel)
    phi = {law: _phi_medians(laws[law]) for law in HB_LAWS}
    phi_status = {"H-E": phi_status_counts(params, HE_COMPONENTS),
                  **{law: phi_status_counts(laws[law], HB_COMPONENTS) for law in HB_LAWS}}
    phi_status = {block: {**c, "note": clipped_high_note(c)} for block, c in phi_status.items()}
    print(f"[sim] fitted {len(params)} H-E node models and {len(laws[HB_BASE_LAW])} per H-B law, "
          f"{time.time() - t0:.0f} s; score persistence (median over nodes): "
          + "; ".join(f"{law} {v['phi_score_median']:.3f} (raw {v['phi_score_raw_median']:.3f})"
                      for law, v in phi.items()), flush=True)
    print("[sim] persistence corrections ending clipped high / clipped low / no signal / uncorrected (nodes with the "
          "status in a component the block draws, of all its nodes): "
          + "; ".join(f"{block} {c['phi_clipped_high']}/{c['phi_clipped_low']}/{c['phi_no_signal']}/"
                      f"{c['phi_uncorrected']} of {c['phi_nodes']}" for block, c in phi_status.items()), flush=True)
    for block, c in phi_status.items():
        if c["note"]:
            print(f"[sim] FLAG {block}: {c['note']}", flush=True)
    sim_rows = he_size_rows(params, windows, counts)
    stamp("H-E size", sim_rows)
    power_rows, rho_star = he_power_rows(params, windows, counts, target)
    sim_rows += power_rows
    stamp("H-E power", sim_rows)
    sim_rows += hb_size_rows(laws, windows, counts)
    stamp("H-B size", sim_rows)
    sim_rows += hb_power_rows(panel, laws, windows, counts)
    stamp("H-B power", sim_rows)
    family = registration_family(sim_rows, placebo)
    sim = _lead(pd.DataFrame(sim_rows))
    print(sim.to_string(), flush=True)
    for hid in ("H-E", "H-B1", "H-B2"):
        f = family[hid]
        print(f"[sim] {hid}: {'confirmatory' if f['confirmatory'] else 'ESTIMAND'}"
              + (", approximate" if f["approximate"] else "")
              + (", power is a lower bound" if f.get("power_lower_bound") else "")
              + f", false-demotion bound {f['false_demotion_bound']:.3f}"
              + ("; " + "; ".join(f["reasons"]) if f["reasons"] else ""), flush=True)
    for note in family["notes"]:
        print(f"[sim] note: {note}", flush=True)
    protocol = {"eta": eta, "family": family, "stage": "complete", "rule": _rule_record(counts, family),
                "analog_selc": analog, "analog_selc_pooled": pooled, "rho_star": rho_star, "phi": phi,
                "phi_status": phi_status, "simulation": [_present(r) for r in sim_rows], "placebo": placebo}
    sim_csv, marker = ANALOG_DIR / "simulation.csv", ANALOG_DIR / "protocol_inputs.json"
    if _current_stage() == "complete":
        _publish({marker: _json(_stage_one_protocol(eta, placebo))})      # withdraw the old family before anything new
    _publish({sim_csv: _csv(sim), marker: _json(protocol)}, stale=(sim_csv,))
    return protocol


def _load_inputs():
    """(panel, fmm, nyi) through the development loaders only: the sealed-window guard is on (``dev_only=True``)."""
    fmm = load_caiso_fmm(DATA_ROOT / "caiso_fmm_dev")
    nyi = load_nyiso_integrated(DATA_ROOT / "nyiso_rtint_dev")
    panel = build_panel(load_dev_raw())
    return panel, fmm, nyi


def _parser():
    ap = argparse.ArgumentParser(prog="python -m es.run_analogs",
                                 description="Quarterly analogues, placebo pairings, eta, size and power study.")
    ap.add_argument("--stage", choices=STAGES, default=None,
                    help="'analogs' (activity, eta, placebo pairings) or 'sim' (size and power, registration family; "
                         "needs 'analogs' first). Default: both, in order.")
    return ap


def main(argv=None):
    args = _parser().parse_args(argv)
    stages = STAGES if args.stage is None else (args.stage,)
    if stages[0] == "sim":
        _require_counted_placebo(_read_stage_one()[1])   # stage 'sim' needs stage 'analogs' (with a p for every counted
        #                                                  pairing): stop now, not after loading the panel
    ensure_dirs()
    ANALOG_DIR.mkdir(parents=True, exist_ok=True)
    panel, fmm, nyi = _load_inputs()
    run = {"analogs": run_analogs_stage, "sim": run_sim_stage}
    for stage in stages:
        t0 = time.time()
        run[stage](panel, fmm, nyi)
        print(f"[{stage}] done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
