"""Label-free drift monitors (spec v3 §5.5; they form one of the three BH families of §5.2): the
existing 1-Wasserstein summary (the July pipeline's ``nwd``), PSI on inputs, a day-blocked domain
classifier (Ruling 73), and the early realized alert rate. Evaluated by Spearman with |log10 R| and AUROC
for node failure.

Spec v3 §5.5 as refined by Ruling 66b: every monitor is computable by deployment day 14, and the outcome
excludes those days. For each node the deployment rows split by local settlement day, counted from the
node's first deployment day, into ``early`` (the first ``MONITOR_DAYS`` = 14 days) and ``late`` (the rest).
W1, PSI and the domain classifier compare the calibration rows with the early rows only, the realized
alert rate is the P1 alert rate on the early days, and the outcome |log10 R| and the failure flag are P1's
on the late days, with the calibration cutoff unchanged. Reason: the early-rate monitor read deployment
days 1-14, which also entered the full-window |log10 R| it was scored against, so part of its correlation
with that outcome was mechanical.

``monitor_rows`` and ``evaluate_monitors`` are pure functions of their arguments. They read no
score cache and no global state, so the sealed run (plan 2A) calls them on its own panel and its
own scores. Only ``main`` (the development run) builds the panel and reads the cache.

Controller rulings built in:
- 37: ``monitor_rows`` / ``evaluate_monitors`` are reusable, the Spearman p-value is one-sided, and
  the domain classifier's CV groups come from each side's own rows;
- 40: the domain classifier is class-balanced;
- 44: W1 is the July ``nwd`` (calibration-IQR scale, median over the family), PSI is NaN when it is
  undefined, NaN monitor values are dropped (and counted) at evaluation, and every per-node failure
  names the analysis, block, market and node;
- 66b: the early/late split above (the monitors read days 1-14, the outcome the days after), with
  ``n_early`` and ``n_late`` reported per node;
- 73 (amends spec §5.5 "blocked by week"): the domain classifier's CV is class-stratified (5 folds) and
  blocked by local settlement day, deterministic, with a both-classes-in-every-fold guard. With the 14-day early
  window, week blocks left 2-3 deployment groups, a null AUC far below 0.5 (about 0.26 on AQ3's calendar) and
  folds that crashed; see ``domain_auc``. With no drift the day-blocked AUC is somewhat above 0.5 for persistent
  inputs: about 0.55 for AR(0.9) features in simulation, on every window length tried, and up to about 0.9 for
  168-hour rolling means. The placebo columns give each block's empirical no-drift reference, and only ranks
  across nodes are interpreted; ``calib_days`` is descriptive;
- 83: descriptive placebo reference columns ``W1_norm_placebo``, ``psi_placebo`` and ``domain_auc_placebo``: the
  same three statistics, with the same functions, day-blocked CV and guards, on a split inside the calibration
  window (its last ``MONITOR_DAYS`` days against the earlier days). They are never monitors: ``MONITORS`` and
  the evaluation are unchanged, they have no p-value and nothing enters a BH family. A 14-day window of a slow
  feature (168-hour rolling means) is a localized excursion, so input-distribution statistics are high even
  without drift (simulated domain AUC about 0.9 for a 168-hour mean, about 0.6 for a 24-hour one). The placebo columns
  show the real-data level of this effect per block; ``evaluate_monitors`` reports their median per
  (analysis, block, monitor) as ``placebo_median`` and the node count behind it as ``placebo_n``.
Run: DART_PROJECT_ROOT=... /opt/anaconda3/bin/python3 -m es.run_monitors"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wasserstein_distance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from .analyses import ANALOGS, LEGACY
from .blocks import BLOCKS
from .config import ALPHA, OUT_DIR, ensure_dirs
from .data import build_panel, load_dev_raw
from .policies import activity, p1_fixed
from .scores import cached_scores

MON_DIR = OUT_DIR / "monitors"

#: Folds of the domain classifier's CV: class-stratified and blocked by local settlement day (Ruling 73, which
#: amends spec v3 §5.5's "blocked by week").
N_FOLDS = 5
#: Deployment days the monitors may read, counted from each node's first deployment day (Ruling 66b). The
#: outcome (|log10 R| and the failure flag) is scored on the days after them.
MONITOR_DAYS = 14
#: The four monitors, in the order ``evaluate_monitors`` reports them.
MONITORS = ("W1_norm", "PSI", "domain_auc", "early_rate")
#: The descriptive placebo statistics of ``monitor_rows`` (Ruling 83), by the monitor each one mirrors. They are
#: never monitors: they are not in ``MONITORS``, have no p-value and enter no BH family.
PLACEBO_COLUMNS = {"W1_norm": "W1_norm_placebo", "PSI": "psi_placebo", "domain_auc": "domain_auc_placebo"}
#: Columns of ``monitor_rows`` (monitor_scores.csv) and of ``evaluate_monitors`` (monitor_eval.csv).
ROW_COLUMNS = ["analysis", "block", "market", "node", *MONITORS, "log_error", "fail", "n_early", "n_late", "calib_days",
               *PLACEBO_COLUMNS.values()]
EVAL_COLUMNS = ["analysis", "block", "monitor", "spearman", "spearman_p_one_sided", "auroc_fail", "n_used", "calib_days",
                "placebo_median", "placebo_n"]


# The July pipeline's 1-Wasserstein drift summary (spec v3 §5.5: "the existing 1-Wasserstein summaries"),
# copied verbatim from v2/scripts/open_holdout.py:39-43. That script opens the sealed holdout when it runs, so it
# is not imported, and es/run_legacy.py is not used either (another task's module). A test compiles the original
# function from the script's source and checks that this copy agrees with it.
def nwd(a, d):
    a, d = a[np.isfinite(a)], d[np.isfinite(d)]
    if len(a) < 50 or len(d) < 50: return np.nan
    iqr = np.subtract(*np.percentile(a, [75, 25])) or 1e-9
    return wasserstein_distance(a, d) / iqr


def psi(reference, current, bins=10):
    """Population stability index of ``current`` against ``reference``, with ``bins`` reference-quantile bins.

    NaN when the reference leaves fewer than two bins: a constant reference has one quantile edge, and deciles
    that collapse onto two values leave a single bin. PSI is then 0 for every current sample, and 0.0 would say
    "no drift" when the deployment may have moved.
    """
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return float("nan")
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(reference, edges)[0] / len(reference)
    c = np.histogram(current, edges)[0] / len(current)
    r, c = np.maximum(r, 1e-4), np.maximum(c, 1e-4)
    return float(np.sum((c - r) * np.log(c / r)))


def day_ids(dates):
    """Integer id of each local settlement day: days since 1970-01-01. The CV group of the domain classifier.

    One id per calendar day whatever the time of day, consecutive days differ by one, and no two days share an
    id, across year ends too (Ruling 73 blocks by day; Ruling 40's ISO-year-and-week key is no longer needed).
    """
    days = pd.to_datetime(pd.Series(dates)).dt.normalize()
    return (days - pd.Timestamp("1970-01-01")).dt.days.to_numpy(dtype="int64")


def cv_groups(ids, n_cal):
    """CV groups for the domain classifier: each side's rows keep their own settlement days (Ruling 73).

    ``ids`` are the day ids (``day_ids``) of the ``n_cal`` calibration rows, then of the deployment rows.
    Deployment groups are shifted by 10,000 so a calibration day and a deployment day never share a group.
    """
    ids = np.asarray(ids)
    return np.r_[ids[:n_cal], ids[n_cal:] + 10_000]


def domain_auc(cal_X, dep_X, ids):
    """Out-of-fold AUROC of a logistic classifier that separates calibration from deployment rows.

    Class-stratified 5-fold CV blocked by local settlement day (Ruling 73, which amends spec v3 §5.5's "blocked
    by week"): ``StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=False)`` keeps every day whole and spreads both
    classes over the folds, and with no shuffling it is deterministic. ``ids`` are the day ids (``day_ids``) of the
    calibration rows, then of the deployment rows, in the order of ``vstack([cal_X, dep_X])``; any other length
    raises ValueError, and so do fewer than ``N_FOLDS`` distinct day groups across both windows.

    Guard: the training part and the test part of every fold must hold both classes, otherwise a ValueError says
    which fold and part lack which class, and how many calibration and deployment days there are
    (``monitor_rows`` adds the node). With 5 or more deployment days it did not fire in any layout tested, including
    4,400 random missing-hours patterns on the real analysis windows; it is a guard, not a guarantee.

    Why days. With the 14-day early deployment window of Ruling 66b, week blocks leave 2-3 deployment groups.
    That gave a null AUC far below 0.5 (about 0.26 on AQ3's calendar, and 0.46 under a real +0.5 sd shift; 40
    seeds, AR 0.9 features) and, with some calendars, folds trained on one class (the frozen sealed definitions
    SAF and SBF raised). Day groups avoid both.

    Levels. With no drift the day-blocked AUC is somewhat above 0.5 for persistent inputs: about 0.55 for AR(0.9)
    features in simulation, on every window length tried (0.54 on a quarter, 0.57 on a half-year, 0.56 on a year;
    standard errors about 0.015), and up to about 0.9 for 168-hour rolling means. For iid features it is 0.50, and a
    +0.5 sd shift scores about 0.8. The placebo columns of ``monitor_rows`` (Ruling 83) give each block's empirical
    no-drift reference, and only the ranks of the AUC across nodes (Spearman, AUROC for failure) are interpreted,
    not its level. ``calib_days`` in ``monitor_eval.csv`` is descriptive: the calibration window's length in
    local settlement days.

    Caveat. On slowly moving features a 14-day deployment window is a small, localized sample that day blocks
    cannot protect against: with no drift the AUC was about 0.9 on a 168-hour rolling mean of white noise (0.91
    over 30 seeds; 0.57 on a 24-hour one and 0.50 on white noise). Read the AUC of blocks with rolling features
    against their placebo (``domain_auc_placebo``, Ruling 83), not against 0.5.

    The fit is class-balanced (Ruling 40). Each group holds one class, so an unweighted intercept follows the
    training folds' class balance, which runs against the held-out fold's balance, and the pooled out-of-fold
    AUC falls below 0.5 even when nothing drifted.
    """
    if len(ids) != len(cal_X) + len(dep_X):
        raise ValueError("domain_auc: the day ids must cover the calibration rows, then the deployment rows")
    X = np.vstack([cal_X, dep_X])
    y = np.r_[np.zeros(len(cal_X)), np.ones(len(dep_X))]
    groups = cv_groups(ids, len(cal_X))
    n_groups = len(np.unique(groups))
    if n_groups < N_FOLDS:
        raise ValueError(f"domain_auc: {n_groups} distinct day group(s) across the calibration and deployment "
                         f"windows, but {N_FOLDS}-fold CV blocked by day needs at least {N_FOLDS}")
    folds = list(StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=False).split(X, y, groups))
    for k, (train, test) in enumerate(folds):
        for part, idx in (("training", train), ("test", test)):
            present = set(y[idx])
            if len(present) < 2:
                missing = "calibration" if 0.0 not in present else "deployment"
                n_cal_days, n_dep_days = (len(np.unique(groups[:len(cal_X)])), len(np.unique(groups[len(cal_X):])))
                raise ValueError(f"domain_auc: the {part} part of fold {k} has no {missing} rows; class-stratified "
                                 f"{N_FOLDS}-fold CV blocked by day needs at least {N_FOLDS} calibration days and "
                                 f"{N_FOLDS} deployment days (here {n_cal_days} and {n_dep_days})")
    pred = np.zeros(len(y))
    for train, test in folds:
        model = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X[train], y[train])
        pred[test] = model.predict_proba(X[test])[:, 1]
    return float(roc_auc_score(y, pred))


def deployment_day(frame):
    """Local settlement days since each node's first deployment day, per row (day 0 is the first).

    A node's first deployment day is the earliest ``settle_date`` among its rows in ``frame``, so a node whose
    data starts later counts its days from there. Rows with a day below ``MONITOR_DAYS`` are 'early'.
    """
    first = frame.groupby(["market", "node"])["settle_date"].transform("min")
    return (pd.to_datetime(frame["settle_date"]) - pd.to_datetime(first)).dt.days


def early_rate(alerts, days=None):
    """The P1 alert rate per node over its first ``days`` deployment days; by default ``MONITOR_DAYS``, read when
    the function is called (like every other use of the constant), not when the module is imported."""
    days = MONITOR_DAYS if days is None else days
    early = alerts[deployment_day(alerts) < days]
    return early.groupby(["market", "node"])["alert"].mean()


def input_statistics(cal, dep, feats, strict=True):
    """W1, PSI and the domain AUC of the ``dep`` rows against the ``cal`` rows: the three input-distribution
    statistics of ``monitor_rows``, and, on the placebo split, its placebo (Ruling 83) too, so both use the same
    functions, the same day-blocked class-stratified CV and the same guards.

    ``W1`` is the median over ``feats`` of the July ``nwd`` on the raw values and ``PSI`` the mean over the
    features that have one. The domain classifier sees the features scaled by the ``cal`` median and std. With
    ``strict=False`` a domain classifier that cannot run (its guard raised a ValueError) gives NaN instead of raising.
    """
    w1 = np.nanmedian([nwd(cal[f].to_numpy(float), dep[f].to_numpy(float)) for f in feats])
    p = np.nanmean([psi(cal[f].to_numpy(), dep[f].to_numpy()) for f in feats])
    med = cal[feats].median()                    # the domain classifier sees median/std-scaled inputs
    sd = cal[feats].std().replace(0, 1)
    days = day_ids(pd.concat([cal["settle_date"], dep["settle_date"]]))
    try:
        auc = domain_auc(((cal[feats] - med) / sd).to_numpy(), ((dep[feats] - med) / sd).to_numpy(), days)
    except ValueError:
        if strict:
            raise
        auc = np.nan
    return w1, p, auc


def placebo_split(cal):
    """Split a node's calibration rows for the placebo (Ruling 83): the pseudo-deployment is the last
    ``MONITOR_DAYS`` local settlement days that have rows, the pseudo-calibration all the earlier days. Returns
    ``(pseudo_cal, pseudo_dep)``; with fewer than ``MONITOR_DAYS`` days everything is pseudo-deployment."""
    last = set(sorted({*cal["settle_date"]})[-MONITOR_DAYS:])
    is_last = cal["settle_date"].isin(last).to_numpy()
    return cal[~is_last], cal[is_last]


def _unmonitorable(message):
    """The ValueError for a node whose windows cannot carry the monitors (too little data, not a bug). It is a plain
    ValueError, so strict mode raises exactly what it always did; ``monitor_rows(strict=False)`` recognises the tag."""
    exc = ValueError(message)
    exc.unmonitorable = True
    return exc


def monitor_rows(panel, analysis, scores_by_block, strict=True):
    """One row per (block, node): the four monitors, the node's P1 outcome and the sample sizes behind them.

    ``scores_by_block[block]`` is that block's xgb seed-42 score frame (columns as in ``es.scores``). The
    calibration and deployment windows come from ``analysis``. A node contributes a row if P1 can place a
    cutoff for it. Columns: ``ROW_COLUMNS``.

    Each node's deployment rows split by local settlement day, counted from the node's first deployment day
    (``deployment_day``): the ``early`` rows are the first ``MONITOR_DAYS`` days, the ``late`` rows the rest.
    Every monitor is computable by day ``MONITOR_DAYS``, and the outcome excludes those days (Ruling 66b).
    - ``W1_norm``, ``PSI`` and ``domain_auc`` compare the calibration rows with the early rows only.
      ``W1_norm`` is the median over the block's features of the July ``nwd`` on the raw values (NaN for a
      feature with fewer than 50 finite values on either side). ``PSI`` is the mean over the features that
      have one. Either is NaN for a node with no feature to average; ``evaluate_monitors`` drops those.
      ``domain_auc`` is class-stratified 5-fold CV blocked by local settlement day (Ruling 73).
    - ``early_rate`` is the P1 alert rate on the early days.
    - ``log_error`` and ``fail`` are the outcome: ``activity`` of the P1 alerts on the late days, with the
      calibration cutoff that P1 fixed from the whole calibration window.
    - ``n_early`` is the number of early deployment rows the monitors read, ``n_late`` the number of late rows
      the outcome rests on.
    - ``calib_days`` is the number of distinct local settlement days in the node's calibration rows, that is, the
      calibration window's length in days (and the number of CV day groups on that side).
    - ``W1_norm_placebo``, ``psi_placebo`` and ``domain_auc_placebo`` (Ruling 83) are the same three statistics
      on a split inside the node's calibration rows (``placebo_split``): the last ``MONITOR_DAYS`` calibration
      days play the deployment and the earlier days the calibration. They show the block's no-drift level, which
      for slow features is high (a 14-day window of a 168-hour rolling mean is a localized excursion). They are
      descriptive, never monitors. With fewer than ``N_FOLDS`` pseudo-calibration days all three are NaN, and
      a placebo domain classifier whose guard fires gives a NaN AUC. Neither raises: both are counted in
      ``attrs["placebo_unavailable"]`` (``cases``, ``too_small``, ``domain_guard``) and reported in one
      UserWarning.

    A node with fewer than ``MONITOR_DAYS`` early days, or with no later day, raises a ValueError. So does any
    other failure while a node is being monitored (an empty window, a fold without a class, ...). Each names
    the analysis, block, market and node, and is chained to the original error.

    ``strict=False`` (controller Ruling 104(d); the sealed run, which must not lose a family of tests to one thin node)
    changes only that: a node whose windows cannot carry the monitors (no calibration or deployment rows with complete
    features, fewer than ``MONITOR_DAYS`` early days, or no deployment day after them) is skipped and counted, not raised.
    So is a node that the block's score frame does not cover (P1 places no cutoff for it there; controller Ruling 109, M3):
    in strict mode that node is passed over without a word, as it always was, and in this mode it is counted too.
    The frame's ``attrs["skipped_nodes"]`` lists {"analysis", "block", "market", "node", "reason"} for each, and one
    UserWarning gives their number. Every other failure still raises. The default is bit-identical to what it was: the same
    errors, the same frame and the same ``attrs`` (``skipped_nodes`` is set only when ``strict`` is False).
    """
    rows, skipped = [], []
    placebo_count = {"cases": 0, "too_small": 0, "domain_guard": 0}
    for block, sc in scores_by_block.items():
        feats = BLOCKS[block]
        if not feats:
            raise ValueError(f"monitor_rows: block {block!r} has no price-history features to monitor")
        alerts = p1_fixed(sc, ALPHA)
        first_day = alerts.groupby(["market", "node"])["settle_date"].min()
        act = activity(alerts[deployment_day(alerts) >= MONITOR_DAYS], ALPHA).set_index(["market", "node"])
        er = early_rate(alerts)
        for (m, n), g in panel.groupby(["market", "node"]):
            if (m, n) not in first_day.index:
                # Ruling 109, M3: a node the block's frame does not cover is counted, not lost (strict mode never reports it)
                skipped.append({"analysis": analysis.id, "block": block, "market": m, "node": n,
                                "reason": f"absent from the {block} score frame (P1 places no cutoff for it there)"})
                continue
            try:
                cal = g[analysis.calib.mask(g).to_numpy()].dropna(subset=feats)
                dep_all = g[analysis.deploy.mask(g).to_numpy()].dropna(subset=feats)
                for side, window in (("calibration", cal), ("deployment", dep_all)):
                    if window.empty:
                        raise _unmonitorable(f"no {side} rows with complete {block} features")
                dep_day = (pd.to_datetime(dep_all["settle_date"]) - pd.Timestamp(first_day[(m, n)])).dt.days
                dep = dep_all[(dep_day < MONITOR_DAYS).to_numpy()]            # the early days: all the monitors read
                n_days = dep["settle_date"].nunique()
                if n_days < MONITOR_DAYS:
                    raise _unmonitorable(f"only {n_days} early deployment day(s) with complete {block} features, "
                                         f"but the monitors read the first {MONITOR_DAYS}")
                if (m, n) not in act.index:
                    raise _unmonitorable(f"no deployment day after the first {MONITOR_DAYS} to score the outcome on")
                w1, p, auc = input_statistics(cal, dep, feats)
                pseudo_cal, pseudo_dep = placebo_split(cal)
                placebo_count["cases"] += 1
                if pseudo_cal["settle_date"].nunique() < N_FOLDS:             # too small a placebo split: NaN, counted
                    placebo = (np.nan, np.nan, np.nan)
                    placebo_count["too_small"] += 1
                else:
                    placebo = input_statistics(pseudo_cal, pseudo_dep, feats, strict=False)
                    placebo_count["domain_guard"] += bool(np.isnan(placebo[2]))
                rows.append({"analysis": analysis.id, "block": block, "market": m, "node": n, "W1_norm": w1,
                             "PSI": p, "domain_auc": auc, "early_rate": float(er.get((m, n), np.nan)),
                             "log_error": float(act.loc[(m, n), "log_error"]),
                             "fail": bool(act.loc[(m, n), "fail_0.5_2"]),
                             "n_early": len(dep), "n_late": int(act.loc[(m, n), "n"]),
                             "calib_days": int(cal["settle_date"].nunique()),
                             "W1_norm_placebo": placebo[0], "psi_placebo": placebo[1], "domain_auc_placebo": placebo[2]})
            except Exception as exc:
                if not strict and getattr(exc, "unmonitorable", False):
                    skipped.append({"analysis": analysis.id, "block": block, "market": m, "node": n, "reason": str(exc)})
                    continue
                raise ValueError(f"monitor_rows({analysis.id!r}, {block!r}): node {m}/{n}: "
                                 f"{type(exc).__name__}: {exc}") from exc
    out = pd.DataFrame(rows, columns=ROW_COLUMNS)
    out.attrs["placebo_unavailable"] = dict(placebo_count)
    if not strict:
        out.attrs["skipped_nodes"] = skipped
        if skipped:
            warnings.warn(f"monitor_rows({analysis.id!r}): {len(skipped)} node-block(s) skipped because their windows "
                          "cannot carry the monitors or the block's score frame does not cover them: "
                          + "; ".join(f"{d['block']} {d['market']}/{d['node']} ({d['reason']})" for d in skipped),
                          UserWarning, stacklevel=2)
    if placebo_count["too_small"] or placebo_count["domain_guard"]:
        warnings.warn(f"monitor_rows({analysis.id!r}): no placebo for {placebo_count['too_small']} of "
                      f"{placebo_count['cases']} node-block(s) (fewer than {N_FOLDS} pseudo-calibration days) and no "
                      f"placebo domain AUC for {placebo_count['domain_guard']} of {placebo_count['cases']} (the "
                      "day-blocked classifier cannot run)", UserWarning, stacklevel=2)
    return out


def _spearman_one_sided(x, y):
    """Spearman rho and the one-sided p-value for a positive association (alternative: rho > 0).

    The p-value comes from the two-sided one: p/2 when rho > 0, 1 - p/2 otherwise. A NaN rho
    (constant input, fewer than two nodes) gives a NaN p.
    """
    res = spearmanr(x, y)
    rho, p = float(res.statistic), float(res.pvalue)
    if np.isnan(rho) or np.isnan(p):
        return rho, np.nan
    return rho, (p / 2 if rho > 0 else 1 - p / 2)


def evaluate_monitors(df):
    """Per (analysis, block, monitor): Spearman with |log10 R| over nodes, its one-sided p-value
    (positive association) and the AUROC for node failure. |log10 R| and the failure flag are the late-day
    outcome of ``monitor_rows`` (``log_error``, ``fail``). The realized-rate monitor enters as
    |log10 (early rate / alpha)|, floored at R = 1e-3.

    A node whose value for the monitor is NaN is dropped for that (analysis, block, monitor) before
    Spearman and AUROC, and ``n_used`` is the number of nodes left. ``calib_days`` is the calibration window
    length in local settlement days (the longest node's count; NaN if the frame has no such column). It is
    descriptive: only the ranks of the AUC across nodes are interpreted, not its level, which is read against
    the placebo columns (see ``domain_auc``).

    ``placebo_median`` and ``placebo_n`` (Ruling 83) are descriptive: the median over the group's nodes of the
    monitor's placebo column (``PLACEBO_COLUMNS``) and the number of nodes that have a value, NaN for
    ``early_rate`` (it has no placebo) or when the frame has no placebo columns. They are not monitors and are
    used nowhere in the Spearman, p-value or AUROC above. Columns: ``EVAL_COLUMNS``."""
    ev = []
    for (aid, block), g in df.groupby(["analysis", "block"]):
        longest = g["calib_days"].max() if "calib_days" in g else np.nan
        calib_days = int(longest) if pd.notna(longest) else np.nan
        for mon in MONITORS:
            x = g[mon].to_numpy(float)
            if mon == "early_rate":
                x = np.abs(np.log10(np.maximum(x / ALPHA, 1e-3)))
            keep = ~np.isnan(x)
            x, error, failed = x[keep], g["log_error"].to_numpy(float)[keep], g["fail"].to_numpy()[keep]
            rho, p = _spearman_one_sided(x, error)
            auc = roc_auc_score(failed, x) if pd.Series(failed).nunique() == 2 else np.nan
            pcol = PLACEBO_COLUMNS.get(mon)
            if pcol is not None and pcol in g:               # descriptive only: never used above
                placebo_median, placebo_n = float(g[pcol].median()), int(g[pcol].notna().sum())
            else:
                placebo_median, placebo_n = np.nan, np.nan
            ev.append({"analysis": aid, "block": block, "monitor": mon, "spearman": rho,
                       "spearman_p_one_sided": p, "auroc_fail": auc, "n_used": int(keep.sum()),
                       "calib_days": calib_days, "placebo_median": placebo_median, "placebo_n": placebo_n})
    return pd.DataFrame(ev, columns=EVAL_COLUMNS)


def main():
    ensure_dirs()
    MON_DIR.mkdir(parents=True, exist_ok=True)
    panel = build_panel(load_dev_raw())
    frames = []
    for a in list(ANALOGS.values()) + [LEGACY[k] for k in ("2024-2025", "25H1-26H1")]:
        scores = {b: cached_scores(panel, a, b, "xgb", 42) for b in BLOCKS if b != "G6"}
        frames.append(monitor_rows(panel, a, scores))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(MON_DIR / "monitor_scores.csv", index=False)
    evaluate_monitors(df).to_csv(MON_DIR / "monitor_eval.csv", index=False)


if __name__ == "__main__":
    main()
