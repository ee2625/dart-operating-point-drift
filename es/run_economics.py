"""Development economic analysis (spec v3 §4; exploratory). Primary pipeline: combined
model (G1_24), XGBoost seed 42, rolling-origin, premium position (direction from settlement
DART on training rows), calibrated fees, strata node x local hour x weekday class x month (Ruling 68;
calendar-propensity deciles are the E4 sensitivity).

What is here, and the controller rulings it carries:
- ``economics_for``: the cutoff policies' alerts as the P&L of a standing 1 MW position, summarised (SELC, SNET,
  ES reduction, the J_lambda family, ...). ``keep_rows=True`` also returns the premium-position, calibrated-fee rows of
  ``KEPT`` (Ruling 27). Every row of policy P1 carries ``rotation_p``, the H-E randomization p-value of that row's
  position and fee setting, the economic-robustness family of ``es.secondary`` (Ruling 37).
  Every policy and comparator ranks and matches on P1's rows only (Ruling 82, review items M1 and M2): deployment rows that
  P1 does not have are dropped from their score frames first; one warning counts them per policy and the counts also come
  back as ``kept.attrs["dropped_rows"]``. For a UtcWindow
  deployment the partial local days at the window's edges leave the deployment split before any policy runs, so P1's
  count, the matched counts and the measured rows coincide (M3).
- ``p1_rows_for``: the P1 rows the H-E test is computed on.
- ``estimand_cis``: E1-E4 with day-bootstrap percentile intervals, quarters pooled by their additive pieces (Ruling 27).
  E4 reports each comparator's own SELC and, paired over the same resampled days, the contrast SELC(P1) - SELC(comparator)
  (``E4:dSELC|<comparator>``, Ruling 81). It reports the same two for ESred (Ruling 118): each comparator's own ESred
  (``E4:ESred|<comparator>``) and, paired in the same way, the contrast ESred(P1) - ESred(comparator)
  (``E4:dESred|<comparator>``). E3's ESred has no benchmark at matched activity of its own: sitting out any share of the
  hours shrinks the exposure, and alerts concentrated in risky calendar cells shrink the tail further without any timing
  skill. The comparators are P1's rows with alerts at P1's counts, placed by the calendar-only score (``calendar@count``) or
  by the VOL score (``VOL@count``), so the contrast is what the model's alerts remove beyond them. Both new statistics are
  approximate, like E3's.
- ``rotation_row``: one rotation.csv row. The rotation test's reference set is the full cyclic group (Ruling 69), and its
  per-window ``retained_share`` is flattened to scalars (Ruling 64).
- ``run_all`` / ``main``: the five files of outputs/es/economics/, with the pooled AQ3 + AQ4 rehearsal and a run without
  CAISO for every analysis (spec §4.1, Ruling 41). The files are rewritten atomically after every analysis and a
  ``run_meta.json`` records the run's settings (Ruling 82, M4).
Run: DART_PROJECT_ROOT=... /opt/anaconda3/bin/python3 -m es.run_economics"""
from __future__ import annotations

import datetime as dt
import inspect
import json
import os
import time
import warnings

import numpy as np
import pandas as pd

from .analyses import ANALOGS, LEGACY
from .bootstrap import RowCube, percentile_ci, rotation_test, stationary_counts
from .config import ALPHA, DATA_ROOT, OUT_DIR, SEAL_LAST_DEV_DATE, ensure_dirs
from .data import build_panel, load_dev_raw
from .measures import (E1_KEYS, LAMS, STRATA_DECILE, STRATA_PRIMARY, add_calendar_deciles, decomposition,
                       drop_partial_edge_days, j_key, summarize)
from .policies import p1_fixed, p4_matched, p4_rolling, p5_self_correcting, p6_exact, top_count
from .positions import UNIFORM_FEES, attach_pnl, premium_direction, spike_exposure, stress_threshold
from .scores import cached_scores
from .settlement import attach_settlement, load_caiso_fmm, load_nyiso_integrated
from .windows import UtcWindow

ECON_DIR = OUT_DIR / "economics"

#: The policies whose premium-position, calibrated-fee rows ``economics_for(keep_rows=True)`` returns, and which
#: ``estimand_cis`` needs (P1, P4@r* and P4 for E1-E3, the two comparators for E4).
KEPT = ("P1", "P4@r*", "P4", "calendar@count", "VOL@count")
E4_COMPARATORS = ("calendar@count", "VOL@count")
#: The quarters whose rows are kept for the pooled rehearsal (memory: only these two stay alive past their analysis).
POOLED = ("AQ3", "AQ4")
#: The key of an hourly row.
KEY = ["market", "node", "timestamp_utc"]
#: P4's trailing window, in days.
WINDOW_DAYS = 56
#: The day bootstrap of ``estimand_cis``: resamples, mean block length (days) and seed. Recorded in run_meta.json.
N_REP, MEAN_BLOCK, SEED = 2000, 7, 20261022


class KeptRows(dict):
    """The ``kept`` mapping of ``economics_for(keep_rows=True)``: {name: premium-position, calibrated-fee rows} for each
    name in ``KEPT``. A plain dict in every respect but one: ``attrs``, a dict of facts about the call, in the manner of
    ``DataFrame.attrs``. ``attrs["dropped_rows"]`` is {policy: the deployment rows dropped from that policy's frame because
    P1 does not have them (Ruling 82)} for every policy and comparator, P1 itself being 0. It carries the numbers of the
    warning, so that a caller that does not capture warnings (the sealed entry point) can record them."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attrs = {}


def _vol_scores(primary):
    return primary.assign(score=primary["abs_mean_168h"]).dropna(subset=["score"])


def _counts(alerts):
    return alerts.groupby(["market", "node"])["alert"].sum().astype(int).to_dict()


def _prepare(alerts, g6, fmm, nyi):
    """Settlement legs and calendar deciles for a policy's alerts. The partial local days of a UtcWindow deployment are
    not dropped here: they left the deployment split before the policies ran (``_trim_partial_edges``)."""
    return add_calendar_deciles(attach_settlement(alerts, fmm, nyi), g6)


def _trim_partial_edges(primary, g6, analysis):
    """For a UtcWindow deployment (spec §4.6) relabel to split "other", in both score frames, the deployment rows on the
    partial local days at the edges of each node's window: ``drop_partial_edge_days``' rule (an edge day with fewer than
    23 rows), applied once to the combined frame's deployment rows. Done before any policy runs (controller Ruling 82, M3):
    P4's trailing window still reads these rows ("other" is not "train"), but P1's count, the matched counts and the
    measured rows then coincide. A DateWindow deployment has no partial day, and the frames are returned as they are."""
    if not isinstance(analysis.deploy, UtcWindow):
        return primary, g6
    day = ["market", "node", "settle_date"]
    dep = primary[primary["split"] == "deploy"]
    if dep.empty:
        return primary, g6
    whole = drop_partial_edge_days(dep)[day].drop_duplicates().assign(_whole=True)
    partial = dep[day].drop_duplicates().merge(whole, on=day, how="left")
    partial = partial.loc[partial["_whole"].isna(), day].assign(_partial=True)
    if partial.empty:
        return primary, g6

    def relabel(frame):
        at = np.flatnonzero((frame["split"] == "deploy").to_numpy())
        hit = frame.iloc[at][day].merge(partial, on=day, how="left")["_partial"].notna().to_numpy()
        split = frame["split"].to_numpy(dtype=object).copy()
        split[at[hit]] = "other"
        return frame.assign(split=split)
    return relabel(primary), relabel(g6)


def _scores_for(panel, analysis, score_fn):
    """The two primary score frames (combined model, calendar model), with the partial edge days already trimmed."""
    primary = score_fn(panel, analysis, "G1_24", "xgb", 42)
    g6 = score_fn(panel, analysis, "G6", "xgb", 42)
    return _trim_partial_edges(primary, g6, analysis)


def _within(scores, keys):
    """``scores`` cut down to P1's rows, and how many deployment rows were cut (controller Ruling 82, M1 and M2).

    A node P1 has no rows for (it cannot place one without calibration rows) leaves the frame with all its rows, and any
    other deployment row whose (market, node, timestamp_utc) is not in ``keys`` is dropped; rows that are not deployment
    rows of a node P1 handles (training, calibration, "other") stay, since the policies read them. The frame is returned
    as it is when nothing has to go."""
    deploy = (scores["split"] == "deploy").to_numpy()
    if not deploy.any():
        return scores, 0
    nodes = set(zip(keys["market"], keys["node"]))
    m_codes, m_levels = pd.factorize(scores["market"])
    n_codes, n_levels = pd.factorize(scores["node"])
    handled = np.array([[(m, n) in nodes for n in n_levels] for m in m_levels], dtype=bool).reshape(len(m_levels), len(n_levels))
    keep = handled[m_codes, n_codes]
    at = np.flatnonzero(deploy & keep)
    hit = scores.iloc[at][KEY].merge(keys.assign(_p=True), on=KEY, how="left", validate="m:1")["_p"].notna().to_numpy()
    keep[at[~hit]] = False
    return (scores if keep.all() else scores[keep]), int((deploy & ~keep).sum())


def economics_for(panel, analysis, fmm, nyi, eta, score_fn=cached_scores, keep_rows=False):
    """`score_fn(panel, analysis, block, estimator, seed)` supplies scores. Development runs use the
    cache; the sealed entry point passes es.scores.score_analysis, because sealed runs must never
    read the score cache (controller Ruling 15).

    Returns one summary dict per policy (or comparator) x position x fee setting, plus the premium position split by
    spike exposure. Every P1 dict carries ``rotation_p`` (Ruling 37). With ``keep_rows=True`` it returns
    ``(out, kept)``, ``kept[name]`` being the premium-position, calibrated-fee rows of each name in ``KEPT``.

    Every other policy and comparator ranks and matches on P1's rows only (Ruling 82): deployment rows that P1 does not
    have are dropped from its score frame first, never silently: one UserWarning counts them per policy, and with
    ``keep_rows=True`` the same counts, zeros included, come back as ``kept.attrs["dropped_rows"]`` (see ``KeptRows``).
    Nothing here raises over them, so a sealed run is not aborted after its data are read."""
    primary, g6 = _scores_for(panel, analysis, score_fn)
    train = attach_settlement(primary[primary["split"] == "train"], fmm, nyi)
    direction = premium_direction(train, col="dart_settle")
    exposure = spike_exposure(train, direction, col="dart_settle")
    stress = stress_threshold(train)
    p1 = p1_fixed(primary, ALPHA)
    target = _counts(p1)
    keys = p1[KEY]
    prim, n_primary = _within(primary, keys)                  # the frames the other policies rank and match on
    vol, n_vol = _within(_vol_scores(primary), keys)
    g6, n_g6 = _within(g6, keys)
    policies = {"P1": p1, "P4": p4_rolling(prim, ALPHA, WINDOW_DAYS), "P5": p5_self_correcting(prim, ALPHA, eta),
                "P6": p6_exact(prim, ALPHA), "P4@r*": p4_matched(prim, target, WINDOW_DAYS)}
    comparators = {"calendar@count": top_count(g6, target), "VOL_P1": p1_fixed(vol, ALPHA),
                   "VOL_P4": p4_rolling(vol, ALPHA, WINDOW_DAYS), "VOL_P6": p6_exact(vol, ALPHA),
                   "VOL@count": top_count(vol, target)}
    for alpha_c in (0.01, 0.025, 0.05, 0.10):
        comparators[f"curve_P1_a{alpha_c}"] = p1_fixed(prim, alpha_c)
        comparators[f"curve_P6_a{alpha_c}"] = p6_exact(prim, alpha_c)
    dropped = [(n_primary, ["P4", "P5", "P6", "P4@r*"] + [c for c in comparators if c.startswith("curve_")]),
               (n_vol, [c for c in comparators if c.startswith("VOL")]), (n_g6, ["calendar@count"])]
    if any(n for n, _ in dropped):
        warnings.warn(f"economics_for({analysis.id!r}): deployment rows that P1 does not have were dropped before the policies "
                      "ranked or matched on them (Ruling 82), rows per policy: "
                      + ", ".join(f"{name} {n}" for n, names in dropped for name in names if n), UserWarning, stacklevel=2)
    by_name = {"P1": 0, **{name: n for n, names in dropped for name in names}}
    out, kept = [], KeptRows()
    kept.attrs["dropped_rows"] = {name: by_name[name] for name in (*policies, *comparators)}
    for name, alerts in list(policies.items()) + list(comparators.items()):
        base = _prepare(alerts, g6, fmm, nyi)
        for position in ("premium", "inc", "dec"):
            for fees in ("calibrated",) + UNIFORM_FEES + ("stress",):
                rows = attach_pnl(base, position, direction,
                                  fees="calibrated" if fees == "stress" else fees,
                                  stress=stress if fees == "stress" else None)
                s = summarize(rows)
                s.update({"analysis": analysis.id, "policy": name if name in policies else "comparator",
                          "comparator": name if name not in policies else "none", "position": position,
                          "fees": str(fees),
                          "settle_approx_share": float(rows["settle_approx"].mean())})
                if name == "P1":                                             # Ruling 37: economic-robustness p-values
                    s["rotation_p"] = rotation_test([RowCube.from_rows(rows)])["p"]
                out.append(s)
                if keep_rows and position == "premium" and fees == "calibrated" and name in KEPT:
                    kept[name] = rows
                if position == "premium" and fees == "calibrated" and name in policies:
                    split = rows.join(exposure, on=["market", "node", "settle_hour"])
                    for ex, sub in split.groupby("exposure"):
                        t = summarize(sub)
                        t.update({"analysis": analysis.id, "policy": name, "comparator": "none",
                                  "position": f"premium[{ex}]", "fees": "calibrated",
                                  "settle_approx_share": float(sub["settle_approx"].mean())})
                        if name == "P1":
                            t["rotation_p"] = rotation_test([RowCube.from_rows(sub)])["p"]
                        out.append(t)
    return (out, kept) if keep_rows else out


def p1_rows_for(panel, analysis, fmm, nyi, score_fn=cached_scores):
    """P1 alerts on the primary pipeline, as premium-position rows with calibrated fees,
    the primary-strata columns and calendar deciles: the rows the H-E statistic is computed on."""
    primary, g6 = _scores_for(panel, analysis, score_fn)
    train = attach_settlement(primary[primary["split"] == "train"], fmm, nyi)
    return attach_pnl(_prepare(p1_fixed(primary, ALPHA), g6, fmm, nyi), "premium",
                      premium_direction(train, col="dart_settle"))


# ---------------------------------------------------------------------------------------------------------------
# E1-E4 with day-bootstrap intervals (spec v3 §6.3; Ruling 27)
# ---------------------------------------------------------------------------------------------------------------

def _is_sealed(frame):
    """True if ``frame`` holds a settlement date after the last development date: the rows of a sealed-window run."""
    return bool(pd.Timestamp(frame["settle_date"].max()) > pd.Timestamp(SEAL_LAST_DEV_DATE))


def _check_keys(kept):
    """Every other frame of ``kept`` must hold exactly P1's rows (market, node, timestamp_utc), neither fewer nor more.
    ``economics_for`` restricts the frames to P1's rows, so this cannot fire on its output. On rows of the development
    window a mismatch raises (a bug to find); on rows of the sealed window it only warns, because a sealed run cannot be
    repeated and must not be aborted after its data are read (Ruling 82)."""
    base = pd.MultiIndex.from_frame(kept["P1"][KEY])
    problems = []
    for name in KEPT[1:]:
        other = pd.MultiIndex.from_frame(kept[name][KEY])
        n_missing, n_extra = len(base.difference(other, sort=False)), len(other.difference(base, sort=False))
        if n_missing or n_extra:
            problems.append(f"{name}: {n_missing} P1 row(s) missing, {n_extra} row(s) that P1 does not have")
    if not problems:
        return
    message = "estimand_cis: these frames do not hold exactly P1's rows (market, node, timestamp_utc): " + "; ".join(problems)
    if _is_sealed(kept["P1"]):
        warnings.warn(message + ". The rows are of the sealed window, so the run goes on.", UserWarning, stacklevel=4)
    else:
        raise ValueError(message)


def _aligned_alerts(base, other, name):
    key = ["market", "node", "timestamp_utc"]
    m = base[key].merge(other[key + ["alert"]], on=key, how="left", validate="1:1")["alert"]
    if m.isna().any():
        raise ValueError(f"{name}: alerts missing for {int(m.isna().sum())} P1 row(s)")
    return m.to_numpy(dtype=float)


def _window(kept):
    cube = RowCube.from_rows(kept["P1"])
    _check_keys(kept)
    alerts = {"P1": cube.alert, "P4@r*": _aligned_alerts(kept["P1"], kept["P4@r*"], "P4@r*"),
              "P4": _aligned_alerts(kept["P1"], kept["P4"], "P4")}
    decile = RowCube.from_rows(kept["P1"], strata=STRATA_DECILE)   # E4 sensitivity strata (Ruling 68)
    comps = {c: RowCube.from_rows(kept[c]) for c in E4_COMPARATORS}
    for c, cc in comps.items():
        if not np.array_equal(cc.days, cube.days):
            raise ValueError(f"{c}: its deployment days differ from P1's")
    return cube, alerts, decile, comps


def _parts(win, w):
    """Additive pieces of every estimand in one window, so quarters pool by summation."""
    cube, alerts, decile, comps = win
    D = cube.day_weight(w)
    p = {"D": D}
    for pol, a in alerts.items():
        p[f"NET|{pol}"] = cube.net(w, a)
        p[f"ESD|{pol}"] = D * cube.esred(w, alert=a)          # day-weighted ESred
    p["selc_ex"], p["selc_L"] = cube.selc_parts(w)
    p["snet_ex"], p["node_days"] = cube.snet_parts(w)
    p["decile_ex"], p["decile_L"] = decile.selc_parts(w)
    for c, cc in comps.items():
        p[f"cmp_ex|{c}"], p[f"cmp_L|{c}"] = cc.selc_parts(w)
        p[f"cmp_ESD|{c}"] = D * cc.esred(w, alert=cc.alert)    # Ruling 118: the comparator's day-weighted ESred, like ESD|P1
    return p


def _stats(p):
    v = {}
    for pol in ("P1", "P4@r*", "P4"):
        v[pol] = {"NET": p[f"NET|{pol}"], "ESred": p[f"ESD|{pol}"] / p["D"]}
        for lam in LAMS:
            v[pol][j_key(lam)] = p[f"NET|{pol}"] + lam * p[f"ESD|{pol}"]
    out = {}
    for k in E1_KEYS:
        out[f"E1:{k}:timing"] = v["P1"][k] - v["P4@r*"][k]
        out[f"E1:{k}:level"] = v["P4@r*"][k] - v["P4"][k]
        out[f"E1:{k}:total"] = v["P1"][k] - v["P4"][k]
    out["E2:SNET"] = p["snet_ex"] / p["node_days"]
    out["E3:ESred"] = v["P1"]["ESred"]
    out["NET"] = v["P1"]["NET"]
    out["SELC"] = p["selc_ex"] / p["selc_L"]
    out["E4:SELC_decile"] = p["decile_ex"] / p["decile_L"]
    for c in E4_COMPARATORS:
        out[f"E4:SELC|{c}"] = p[f"cmp_ex|{c}"] / p[f"cmp_L|{c}"]
        out[f"E4:dSELC|{c}"] = out["SELC"] - out[f"E4:SELC|{c}"]       # Ruling 81: the contrast, paired within a replicate
    for c in E4_COMPARATORS:                                           # Ruling 118: ESred at matched activity
        out[f"E4:ESred|{c}"] = p[f"cmp_ESD|{c}"] / p["D"]
        out[f"E4:dESred|{c}"] = out["E3:ESred"] - out[f"E4:ESred|{c}"]    # paired within a replicate, like dSELC (Ruling 81)
    return out


def _approximate(stat):
    return "ESred" in stat or (stat.startswith("E1:J_") and not stat.startswith("E1:J_0.0"))


def estimand_cis(windows, n_rep=N_REP, mean_block=MEAN_BLOCK, seed=SEED):
    """E1-E4 with percentile CIs (spec v3 §6.3). One `kept` dict per quarter; with two quarters the
    estimands pool by summing their additive pieces (excess, loss, NET, day-weighted ESred, node-days),
    each quarter resampled independently over days. Alerts stay fixed on the real path.

    E4 carries, per comparator c, its own SELC (``E4:SELC|c``) and the contrast ``E4:dSELC|c`` = SELC(P1) - SELC(c)
    (Ruling 81). Both are evaluated on the same resampled days in each replicate, so the contrast's percentile interval is
    that of the paired differences, not the difference of the two marginal intervals; pooled, it is the difference of the
    pooled ratios. E4 carries the same two for ESred (Ruling 118): ``E4:ESred|c``, the comparator's own ESred, and
    ``E4:dESred|c`` = ESred(P1) - ESred(c), paired in the same way. The comparator's additive piece is its day-weighted ESred
    (``cmp_ESD|c``, D times the quarter's ESred, like P1's), so pooled quarters give the day-weighted mean of the quarters'
    ESred, and the contrast is the difference of the pooled levels. Both are approximate (``_approximate``). The four rows
    follow the others in the output. The frames of ``windows`` must hold P1's rows and no others (``_check_keys``)."""
    if not windows:
        raise ValueError("estimand_cis: needs at least one window")
    wins = [_window(k) for k in windows]

    def total(ws):
        parts = [_parts(win, w) for win, w in zip(wins, ws)]
        return _stats({k: sum(p[k] for p in parts) for k in parts[0]})

    est = total([None] * len(wins))
    draws = [stationary_counts(win[0].n_days, mean_block, n_rep, seed + 1000 * i) for i, win in enumerate(wins)]
    boot = [total([d[b] for d in draws]) for b in range(n_rep)]
    rows = []
    for stat, value in est.items():
        lo, hi = percentile_ci(np.array([b[stat] for b in boot]))
        rows.append({"stat": stat, "estimate": value, "lo": lo, "hi": hi,
                     "approximate": _approximate(stat), "n_windows": len(wins)})
    return rows


# ---------------------------------------------------------------------------------------------------------------
# The files
# ---------------------------------------------------------------------------------------------------------------

def _without_caiso(kept):
    """``kept`` without its CAISO rows (spec §4.1, Ruling 41). Every model is fitted per node, so a row filter is a run
    without CAISO."""
    return {k: v[v["market"] != "CAISO"] for k, v in kept.items()}


def rotation_row(window, cubes):
    """One row of rotation.csv: the H-E test on ``cubes`` (one RowCube per window), flattened to scalars.

    ``t_obs`` (the full-grid studentized T, unitless: it is not SELC), ``p``, ``n_null`` (from the result: D - 1 for a
    window of D days, prod(D_w) - 1 pooled) and ``selc_obs`` (the pooled SELC, the reported estimand) are the test's own.
    ``retained_share`` is one {"min", "mean"} per window; the row keeps ``retained_min``, the minimum over windows of
    each window's min, and ``retained_mean``, the mean over windows of each window's mean. A NaN (a window without
    alerts) is not skipped."""
    res = rotation_test(cubes)
    shares = res["retained_share"]
    return {"window": window, "t_obs": res["t_obs"], "p": res["p"], "n_null": res["n_null"],
            "selc_obs": res["selc_obs"], "retained_min": float(np.min([s["min"] for s in shares])),
            "retained_mean": float(np.mean([s["mean"] for s in shares]))}


def _now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rotation_defaults():
    """The defaults of rotation_test's keyword parameters: what the run uses, since it never passes one (Ruling 69)."""
    return {name: p.default for name, p in inspect.signature(rotation_test).parameters.items()
            if p.default is not inspect.Parameter.empty}


def _atomic(path, write):
    """Write ``path`` through a temporary file beside it and ``os.replace``, so that neither a reader nor a crash ever finds
    a half-written file."""
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def run_all(panel, fmm, nyi, eta, analyses, econ_dir, n_rep=N_REP, eta_source=None):
    """Run every analysis of ``analyses`` and write the five files into ``econ_dir``; returns them as DataFrames.

    Per analysis: ``economics_for(keep_rows=True)`` (summary.csv, decomposition.csv), ``estimand_cis`` on its kept rows and
    on the same rows without CAISO (bootstrap_ci.csv, analysis ``<id>|ex-CAISO``), the H-E test and the volatility-management
    check on the P1 rows (rotation.csv, volmanagement_check.csv). The kept rows of AQ3 and AQ4 only are held on, for the
    pooled rehearsal after the loop (memory).

    The five files are rewritten after each analysis (atomically: a temporary file, then ``os.replace``), so a failure in
    analysis k leaves analyses 1..k-1 on disk; the pooled rows come only after the last analysis. ``run_meta.json`` records
    what the run used (resamples, seed, block length, eta and ``eta_source``, alpha, W, the strata, rotation_test's defaults),
    its start and finish, which analyses are done, and its status: running, complete or failed (then with the analysis it
    failed on and the error), so that a partial set of files is never mistaken for a finished run (Ruling 82, M4).
    ``analyses_done`` names an analysis only once all five files have been replaced with its rows, and before run_meta.json
    is written: a write that fails part-way leaves the failed analysis out of the list (it is ``failed_on``), so every
    analysis named there is in every file."""
    econ_dir.mkdir(parents=True, exist_ok=True)
    analyses = list(analyses)
    meta = {"status": "running", "analyses": [a.id for a in analyses], "analyses_done": [], "n_rep": n_rep, "seed": SEED,
            "mean_block": MEAN_BLOCK, "eta": eta, "eta_source": eta_source, "alpha": ALPHA, "W": WINDOW_DAYS,
            "strata_primary": list(STRATA_PRIMARY), "strata_decile": list(STRATA_DECILE),
            "rotation_test_defaults": _rotation_defaults(), "started_utc": _now(), "finished_utc": None}
    rows, decomp, cis, rot, vm = [], [], [], [], []
    cubes, pool = {}, {}

    def write_meta():
        _atomic(econ_dir / "run_meta.json", lambda tmp: tmp.write_text(json.dumps(meta, indent=2, default=str) + "\n"))

    def write_csvs():
        out = {"summary": pd.DataFrame(rows), "decomposition": pd.DataFrame(decomp), "bootstrap_ci": pd.DataFrame(cis),
               "rotation": pd.DataFrame(rot), "volmanagement_check": pd.DataFrame(vm)}
        for name, frame in out.items():
            _atomic(econ_dir / f"{name}.csv", lambda tmp, frame=frame: frame.to_csv(tmp, index=False))
        return out
    write_meta()
    stage = None
    try:
        for a in analyses:
            stage = a.id
            t0 = time.time()
            res, kept = economics_for(panel, a, fmm, nyi, eta, keep_rows=True)
            t1 = time.time()
            rows += res
            pick = {r["policy"]: r for r in res
                    if r["position"] == "premium" and r["fees"] == "calibrated" and r["comparator"] == "none"}
            d = decomposition(pick["P1"], pick["P4@r*"], pick["P4"])
            decomp.append({"analysis": a.id, **{f"{k}_{part}": v[part] for k, v in d.items() for part in v}})
            cis += [{"analysis": a.id, **r} for r in estimand_cis([kept], n_rep=n_rep)]
            cis += [{"analysis": f"{a.id}|ex-CAISO", **r} for r in estimand_cis([_without_caiso(kept)], n_rep=n_rep)]
            t2 = time.time()
            p1rows = kept["P1"]                                  # equals p1_rows_for(panel, a, fmm, nyi)
            cube = RowCube.from_rows(p1rows)
            rot.append(rotation_row(a.id, [cube]))
            alerted = p1rows["alert"].to_numpy(bool)
            pnl = p1rows["pnl_open"].to_numpy(float)
            vm.append({"analysis": a.id, "sharpe_alerted": float(pnl[alerted].mean() / pnl[alerted].std()),
                       "sharpe_not_alerted": float(pnl[~alerted].mean() / pnl[~alerted].std())})
            if a.id in POOLED:
                cubes[a.id], pool[a.id] = cube, kept
            write_csvs()
            meta["analyses_done"].append(a.id)         # only now: its rows are in all five files
            write_meta()
            print(f"{a.id}: economics_for {t1 - t0:.0f} s; estimand_cis (2 runs) {t2 - t1:.0f} s; files written", flush=True)
            del kept, p1rows, res
        stage = "pooled rehearsal"
        if all(i in pool for i in POOLED):
            rot.append(rotation_row("AQ3+AQ4 (rehearsal of the pooled H-E test)", [cubes[i] for i in POOLED]))
            cis += [{"analysis": "AQ3+AQ4 (rehearsal of the pooled estimands)", **r}
                    for r in estimand_cis([pool[i] for i in POOLED], n_rep=n_rep)]
        out = write_csvs()
        meta.update(status="complete", finished_utc=_now())
        write_meta()
    except Exception as exc:
        meta.update(status="failed", failed_on=stage, error=f"{type(exc).__name__}: {exc}", finished_utc=_now())
        write_meta()
        raise
    print(out["rotation"].to_string())
    return out


def main():
    ensure_dirs()
    ECON_DIR.mkdir(parents=True, exist_ok=True)
    eta_path = OUT_DIR / "analogs" / "protocol_inputs.json"
    with open(eta_path) as f:
        eta = json.load(f)["eta"]
    fmm = load_caiso_fmm(DATA_ROOT / "caiso_fmm_dev")
    nyi = load_nyiso_integrated(DATA_ROOT / "nyiso_rtint_dev")
    panel = build_panel(load_dev_raw())
    analyses = list(ANALOGS.values()) + [LEGACY[k] for k in ("2024-2025", "24H2-25H2", "25H1-26H1")]
    run_all(panel, fmm, nyi, eta, analyses, ECON_DIR, eta_source=str(eta_path))


if __name__ == "__main__":
    main()
