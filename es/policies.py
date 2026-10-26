"""Cutoff policies (spec v2 §4.3). All read one score frame; none refits a model.

Each policy returns the deployment rows of the nodes it can handle, plus ``tau`` and ``alert``. A node
it cannot handle is never dropped silently: one consolidated UserWarning per call names every skipped
node and why, and also what else the call glossed over (P4 days with an empty trailing window,
``target_counts`` keys that match no node). If no node qualifies the call raises a ValueError naming the
policy and the reasons, rather than returning an empty frame.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

#: Search bounds of the rate that ``p4_matched`` root-finds.
_RATE_LO, _RATE_HI = 1e-4, 0.5

_NO_CAL, _NO_DEP, _NO_TARGET = "no calibration rows", "no deployment rows", "not in target_counts"


def _tau(values, alpha: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), 1 - alpha))


def _per_node(scores: pd.DataFrame):
    for key, g in scores.groupby(["market", "node"], sort=True):
        yield key, g.sort_values("timestamp_utc")


def _with_tau(dep: pd.DataFrame, tau) -> pd.DataFrame:
    tau_arr = np.broadcast_to(np.asarray(tau, dtype=float), (len(dep),))
    return dep.assign(tau=tau_arr, alert=dep["score"].to_numpy(dtype=float) >= tau_arr)


def _why(*checks) -> str:
    """The reasons that hold, joined. ``checks`` are (reason, holds) pairs."""
    return " and ".join(reason for reason, holds in checks if holds)


def _key_name(key) -> str:
    return "/".join(map(str, key)) if isinstance(key, tuple) else str(key)


def _describe(skipped) -> str:
    return "; ".join(f"{_key_name(key)} ({reason})" for key, reason in skipped)


def _empty_window_note(empty) -> str:
    """``empty`` holds (node key, number of deployment days whose trailing window held no row)."""
    return (f"{len(empty)} node(s) have days with an empty trailing window (tau is NaN there, so no alert "
            "fires): " + "; ".join(f"{_key_name(key)} ({k} day{'' if k == 1 else 's'})" for key, k in empty))


def _unknown_targets_note(target_counts, seen) -> list[str]:
    """One note naming the ``target_counts`` keys that match no node in the score frame, if there are any."""
    unknown = sorted((k for k in target_counts if k not in seen), key=str)
    if not unknown:
        return []
    return [f"{len(unknown)} target_counts key(s) match no node in the score frame: "
            + ", ".join(map(_key_name, unknown))]


def _collect(policy: str, parts, skipped, notes=()) -> pd.DataFrame:
    """Stack the per-node results. Warn once about all the call glossed over; raise if no node qualified.

    ``notes`` are further findings, already worded. They share the skipped nodes' warning, so a call
    emits one consolidated warning and not one per kind of finding.
    """
    if not parts:
        detail = " | ".join([_describe(skipped) or "the score frame has no nodes", *notes])
        raise ValueError(f"{policy}: no node qualifies, so there is nothing to return ({detail})")
    findings = ([f"{len(skipped)} node(s) skipped: {_describe(skipped)}"] if skipped else []) + list(notes)
    if findings:
        warnings.warn(f"{policy}: {' | '.join(findings)}", UserWarning, stacklevel=3)
    return pd.concat(parts, ignore_index=True)


def p1_fixed(scores, alpha):
    parts, skipped = [], []
    for key, g in _per_node(scores):
        cal, dep = g.loc[g.split == "calib", "score"], g[g.split == "deploy"]
        if len(cal) and len(dep):
            parts.append(_with_tau(dep, _tau(cal, alpha)))
        else:
            skipped.append((key, _why((_NO_CAL, cal.empty), (_NO_DEP, dep.empty))))
    return _collect("p1_fixed", parts, skipped)


def p2_insample(scores, alpha, months):
    parts, skipped = [], []
    for key, g in _per_node(scores):
        tr = g[(g.split == "train") & g["settle_date"].map(lambda d: d.month).isin(months)]
        dep = g[g.split == "deploy"]
        if len(tr) and len(dep):
            parts.append(_with_tau(dep, _tau(tr["score"], alpha)))
        else:
            no_train = f"no training rows in the season months {sorted(int(m) for m in months)}"
            skipped.append((key, _why((no_train, tr.empty), (_NO_DEP, dep.empty))))
    return _collect("p2_insample", parts, skipped)


def p6_exact(scores, alpha):
    parts, skipped = [], []
    for key, g in _per_node(scores):
        dep = g[g.split == "deploy"]
        if len(dep):
            parts.append(_with_tau(dep, _tau(dep["score"], alpha)))
        else:
            skipped.append((key, _NO_DEP))
    return _collect("p6_exact", parts, skipped)


def _p4_node(g, alpha, window_days):
    """P4 on one node (``g`` sorted by time).

    Returns its deployment rows with ``tau`` and ``alert``, and the number of deployment days whose trailing
    window held no row (their tau is NaN, so they never alert). Returns (None, 0) if it has no deployment
    rows.
    """
    dep = g[g.split == "deploy"]
    if dep.empty:
        return None, 0
    # Deliberate: only train rows are in-sample; calib, gap and earlier deployment days may fill the window.
    hist = g[g.split != "train"]                     # in-sample scores never enter the window
    hist_days = pd.to_datetime(hist["settle_date"]).to_numpy()
    hist_scores = hist["score"].to_numpy(dtype=float)
    order = np.argsort(hist_days, kind="stable")
    hist_days, hist_scores = hist_days[order], hist_scores[order]
    taus, n_empty = {}, 0
    for d in sorted(set(dep["settle_date"])):
        day = np.datetime64(pd.Timestamp(d))
        lo = np.searchsorted(hist_days, day - np.timedelta64(window_days, "D"), side="left")
        hi = np.searchsorted(hist_days, day, side="left")
        n_empty += int(hi <= lo)
        taus[d] = _tau(hist_scores[lo:hi], alpha) if hi > lo else np.nan
    return _with_tau(dep, dep["settle_date"].map(taus).to_numpy(dtype=float)), n_empty


def p4_rolling(scores, alpha, window_days=56):
    """Rolling-alpha cutoff (spec P4): tau for day d is the (1 - alpha) quantile of the node's non-training
    scores dated d - window_days .. d - 1. A day whose window is empty gets tau NaN and no alert, and the
    warning names those days."""
    parts, skipped, empty = [], [], []
    for key, g in _per_node(scores):
        res, n_empty = _p4_node(g, alpha, window_days)
        if res is None:
            skipped.append((key, _NO_DEP))
            continue
        parts.append(res)
        if n_empty:
            empty.append((key, n_empty))
    return _collect("p4_rolling", parts, skipped, [_empty_window_note(empty)] if empty else [])


def p5_self_correcting(scores, alpha, eta, lo=0.001, hi=0.5):
    parts, skipped = [], []
    for key, g in _per_node(scores):
        cal = g.loc[g.split == "calib", "score"].to_numpy(dtype=float)
        dep = g[g.split == "deploy"]
        if cal.size == 0 or dep.empty:
            skipped.append((key, _why((_NO_CAL, cal.size == 0), (_NO_DEP, dep.empty))))
            continue
        a, taus = alpha, {}
        for d, day_rows in dep.groupby("settle_date", sort=True):
            tau = _tau(cal, a)
            taus[d] = tau
            r = float((day_rows["score"].to_numpy(dtype=float) >= tau).mean())
            a = min(hi, max(lo, a + eta * (alpha - r)))
        parts.append(_with_tau(dep, dep["settle_date"].map(taus).to_numpy(dtype=float)))
    return _collect("p5_self_correcting", parts, skipped)


def p4_matched(scores, target_counts, window_days=56, iters=40):
    """P4 with its target rate root-found per node so the alert count matches (spec P4@r*).

    Adds ``matched_rate`` (the rate found) and ``matched_count`` (the alert count it realizes). A second
    UserWarning names every node whose count misses its target by more than max(3, 1% of the target) or
    whose rate sits on a search bound, i.e. whose target was out of reach. Days with an empty trailing
    window never alert at any rate; the first warning names them, as ``p4_rolling`` does.
    """
    parts, skipped, empty, unmatched, seen = [], [], [], [], set()
    edge = 2 * (_RATE_HI - _RATE_LO) / 2 ** iters    # the bisection cannot end closer to a bound than this
    for key, g in _per_node(scores):
        seen.add(key)
        no_target, no_dep = key not in target_counts, not (g.split == "deploy").any()
        if no_target or no_dep:
            skipped.append((key, _why((_NO_TARGET, no_target), (_NO_DEP, no_dep))))
            continue
        target = int(target_counts[key])
        lo, hi = _RATE_LO, _RATE_HI
        run = lambda r: _p4_node(g, r, window_days)[0]                     # noqa: E731
        res_lo, n_empty = _p4_node(g, lo, window_days)
        res_hi = run(hi)
        if n_empty:
            empty.append((key, n_empty))
        for _ in range(iters):
            mid = (lo + hi) / 2
            res = run(mid)
            if int(res["alert"].sum()) < target:
                lo, res_lo = mid, res
            else:
                hi, res_hi = mid, res
        c_lo, c_hi = int(res_lo["alert"].sum()), int(res_hi["alert"].sum())
        chosen, rate = (res_lo, lo) if abs(c_lo - target) <= abs(c_hi - target) else (res_hi, hi)
        count = int(chosen["alert"].sum())
        problems = []
        if abs(count - target) > max(3, 0.01 * target):
            problems.append("alert count differs from the target by more than max(3, 1% of target)")
        on_bound = [b for b in (_RATE_LO, _RATE_HI) if abs(rate - b) <= edge]
        problems += [f"rate sits on the search bound {b:g}" for b in on_bound]
        if problems:
            unmatched.append(f"{key[0]}/{key[1]} (target {target}, matched {count} at rate {rate:.6g}: "
                             f"{' and '.join(problems)})")
        parts.append(chosen.assign(matched_rate=rate, matched_count=count))
    notes = ([_empty_window_note(empty)] if empty else []) + _unknown_targets_note(target_counts, seen)
    out = _collect("p4_matched", parts, skipped, notes)
    if unmatched:
        warnings.warn(f"p4_matched: {len(unmatched)} node(s) not matched to their target: "
                      f"{'; '.join(unmatched)}", UserWarning, stacklevel=2)
    return out


def top_count(scores, target_counts):
    """After-the-fact benchmark: alert on the T highest deployment scores per node.

    Raises ValueError if any T is negative: it would slice from the end of the ranking.
    """
    negative = {key: int(t) for key, t in target_counts.items() if int(t) < 0}
    if negative:
        bad = ", ".join(f"{_key_name(k)}: {t}" for k, t in negative.items())
        raise ValueError("top_count: target_counts must be non-negative (a negative T would slice from the "
                         f"end of the ranking): {bad}")
    parts, skipped, seen = [], [], set()
    for key, g in _per_node(scores):
        seen.add(key)
        dep = g[g.split == "deploy"]
        if key not in target_counts or dep.empty:
            skipped.append((key, _why((_NO_TARGET, key not in target_counts), (_NO_DEP, dep.empty))))
            continue
        t = int(target_counts[key])
        order = np.argsort(-dep["score"].to_numpy(dtype=float), kind="stable")
        alert = np.zeros(len(dep), dtype=bool)
        alert[order[:t]] = True
        parts.append(dep.assign(tau=np.nan, alert=alert))
    return _collect("top_count", parts, skipped, _unknown_targets_note(target_counts, seen))


def activity(alerts, alpha):
    g = alerts.groupby(["market", "node"], sort=True)
    s = pd.DataFrame({"n": g.size(), "alerts": g["alert"].sum().astype(int)})
    s["rate"] = s["alerts"] / s["n"]
    s["R"] = s["rate"] / alpha
    s["zero_activity"] = s["alerts"] == 0
    s["R_floored"] = np.maximum(s["R"], 1.0 / (2.0 * s["n"] * alpha))
    s["log_error"] = np.abs(np.log10(s["R_floored"]))
    s["fail_0.5_2"] = (s["R"] < 0.5) | (s["R"] > 2)
    return s.reset_index()
