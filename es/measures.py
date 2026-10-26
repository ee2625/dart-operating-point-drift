"""Economic measures at matched activity (spec v3 §4.4). Sitting out removes the open
position's loss or profit. SELC compares alerted loss with the within-stratum random
expectation at the same counts. Primary strata (controller Ruling 68) are node x local hour x weekday class x month,
``STRATA_PRIMARY``; node x calendar-propensity decile, ``STRATA_DECILE``, is the sensitivity. Deciles stopped being
primary because, under a calendar-only null, they bias SELC upward and miscentre the rotation test: a decile can hold
rows from calendar groups with different alert rates and loss levels, so inside it an alert carries information about
loss that the stratification was meant to remove.

Each measure first calls ``_require`` on the columns it reads. It raises a ValueError naming the column if the
frame is empty, the column is missing, or the column holds a NaN, so no row is skipped by a sum or dropped by a
groupby (which drops NaN keys). ``expected_shortfall`` likewise rejects an empty series and a NaN day. Two ratios
have a documented value where their denominator is zero: SELC is 0.0 when the rows hold no loss, and CAP is 0.0
when they hold no |DART|."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

#: Primary strata (Ruling 68): node x local hour x weekday class x month.
STRATA_PRIMARY = ["market", "node", "settle_hour", "wclass", "month"]
#: Sensitivity strata: node x calendar-propensity decile (the primary strata of spec v3 before Ruling 68).
STRATA_DECILE = ["market", "node", "cal_decile"]
#: The earlier name of the primary strata, kept as an alias so that imports of it do not break.
STRATA_MONTH = STRATA_PRIMARY

#: The registered lambdas of the value function J_lambda = NET + lambda * D * ESred (spec §4.4).
LAMS = (0.0, 0.25, 0.5, 1.0)


def j_key(lam):
    """The ``summarize`` key of J_lambda for one lambda: ``f"J_{float(lam)}"``. For LAMS that is J_0.0, J_0.25, J_0.5
    and J_1.0; an integer lambda is formatted as a float too, so ``j_key(2)`` is J_2.0."""
    return f"J_{float(lam)}"


#: What the spec's estimand E1 decomposes: NET, ESred and J_lambda for every registered lambda. These are
#: the keys of ``summarize`` output, so ``decomposition`` of summaries covers all of them by default.
E1_KEYS = ("NET", "ESred") + tuple(j_key(lam) for lam in LAMS)


def _require(rows, cols, caller=""):
    """Raise ValueError unless ``rows`` is not empty and each column of ``cols`` is present and holds no NaN.

    The message names the offending column and, if given, the measure that asked. A NaN would otherwise be
    skipped silently by a sum, or would drop its row out of a stratum or a day (groupby drops NaN keys).
    """
    where = f"{caller}: " if caller else ""
    cols = list(dict.fromkeys(cols))
    if len(rows) == 0:
        raise ValueError(f"{where}the frame is empty, so there is nothing to measure (it needs {', '.join(cols)})")
    for col in cols:
        if col not in rows.columns:
            raise ValueError(f"{where}column {col!r} is missing")
        n_nan = int(rows[col].isna().sum())
        if n_nan:
            raise ValueError(f"{where}column {col!r} has {n_nan} NaN value(s)")


def _deciles_by_value(score):
    """Decile codes of one node's calendar-model scores, taken by value (see ``add_calendar_deciles``).

    ``pd.qcut`` is applied to ``rank(method="min")`` instead of the raw scores: equal scores share their lowest
    rank, so they share a decile, and with distinct scores the ranks are 1..n, the very input the order-based
    assignment used, so no decile changes. (On the raw scores the interpolated quantile edges differ from the
    rank edges in floating point. When n - 1 is divisible by 10 an edge falls exactly on an order statistic and
    that can move one row across a boundary: it did in 14 of 2,472 random node-windows tried.)
    ``duplicates="drop"`` merges the edges that coincide inside a tie. A node whose scores are all equal, a
    node with one row included, gives qcut a single edge and nothing to bin: it is one stratum, decile 0.
    """
    if score.nunique() < 2:
        return pd.Series(0, index=score.index)
    return pd.qcut(score.rank(method="min"), 10, labels=False, duplicates="drop")


def add_calendar_deciles(rows, g6_scores):
    """Add ``g6_score``, ``cal_decile`` and ``month`` to ``rows``.

    ``cal_decile`` is the decile of the calendar-only model's score within the window, per (market, node), over
    the rows given. It is taken by VALUE: rows with equal scores have the same calendar propensity, so they
    share a decile, and the assignment does not depend on the order of ``rows``. Ties therefore make the strata
    unequal in size, and where quantile edges coincide fewer than ten are used; the codes lie in 0..9 and only
    label strata (they need not be consecutive). A node whose scores are all equal, a node with one row
    included, is one stratum, decile 0. With distinct scores nothing changes against the order-based rule this
    replaced: the deciles are near-equal in size, and their boundaries come from the quantile arithmetic of the
    pinned pandas 2.2.2 (for n = 91 or 171, say, the largest and smallest decile differ by two rows, not one).

    Raises ValueError if ``rows`` already has ``g6_score``, ``cal_decile`` or ``month`` (all three are added
    here, so the call was probably repeated), if a row has no score in ``g6_scores``, and, through
    ``validate="m:1"``, if ``g6_scores`` holds a (market, node, timestamp_utc) key twice (pandas MergeError, a
    ValueError).
    """
    key = ["market", "node", "timestamp_utc"]
    clash = [c for c in ("g6_score", "cal_decile", "month") if c in rows.columns]
    if clash:
        raise ValueError(f"add_calendar_deciles: rows already has {' and '.join(clash)}, which it adds itself "
                         "(was it applied twice?)")
    g6 = g6_scores[key + ["score"]].rename(columns={"score": "g6_score"})
    r = rows.merge(g6, on=key, how="left", validate="m:1")
    if r["g6_score"].isna().any():
        raise ValueError("every row needs a calendar-model score")
    r["cal_decile"] = (r.groupby(["market", "node"], observed=True)["g6_score"]
                       .transform(_deciles_by_value).astype(int))
    r["month"] = pd.to_datetime(r["settle_date"]).dt.month
    return r


def drop_partial_edge_days(rows, min_hours=23):
    key = ["market", "node", "settle_date"]
    counts = rows.groupby(key, observed=True).size().rename("n").reset_index()
    first = counts.groupby(["market", "node"], observed=True)["settle_date"].transform("min")
    last = counts.groupby(["market", "node"], observed=True)["settle_date"].transform("max")
    bad = counts.loc[((counts["settle_date"] == first) | (counts["settle_date"] == last))
                     & (counts["n"] < min_hours), key]
    merged = rows.merge(bad.assign(_bad=True), on=key, how="left")
    return merged[merged["_bad"].isna()].drop(columns="_bad").reset_index(drop=True)


def _loss_profit(rows):
    pnl = rows["pnl_open"].to_numpy(dtype=float)
    return np.clip(-pnl, 0, None), np.clip(pnl, 0, None)


def la_pg_net(rows):
    _require(rows, ["alert", "pnl_open"], "la_pg_net")
    loss, profit = _loss_profit(rows)
    a = rows["alert"].to_numpy(dtype=bool)
    la, pg = float(loss[a].sum()), float(profit[a].sum())
    return {"LA": la, "PG": pg, "NET": la - pg}


def _stratified(rows, values, strata):
    a = rows["alert"].to_numpy(dtype=float)
    frame = rows[list(strata)].assign(v=values, av=a * values, a=a, n=1.0)
    g = frame.groupby(list(strata), sort=False, observed=True)[["v", "av", "a", "n"]].sum()
    expected = (g["a"] / g["n"]) * g["v"]
    return float((g["av"] - expected).sum()), float(g["v"].sum())


def selc(rows, strata=STRATA_PRIMARY):
    """Stratified excess loss capture (spec §4.4): [sum_s (captured loss_s - m_s/N_s * L_s)] / sum_s L_s, with m_s
    alerted hours out of N_s and loss L_s in stratum s. The strata default to ``STRATA_PRIMARY`` (Ruling 68: node x
    hour x weekday class x month); pass ``STRATA_DECILE`` for the decile sensitivity. By convention SELC is 0.0 when
    the rows hold no loss at all (sum_s L_s = 0), where the ratio is undefined."""
    _require(rows, ["alert", "pnl_open", *strata], "selc")
    loss, _ = _loss_profit(rows)
    excess, total = _stratified(rows, loss, strata)
    return excess / total if total > 0 else 0.0


def snet(rows, strata=STRATA_PRIMARY):
    """The stratified version of NET, in $ per MW-day: the NET of the alerted hours beyond what random selection
    within each stratum would give, divided by the node-days, i.e. the sum over nodes of the distinct settlement
    days each node has (nodes x days on a balanced panel). The strata default to ``STRATA_PRIMARY`` (Ruling 68)."""
    _require(rows, ["alert", "pnl_open", "settle_date", "market", "node", *strata], "snet")
    loss, profit = _loss_profit(rows)
    excess, _ = _stratified(rows, loss - profit, strata)
    mw_days = rows.groupby(["market", "node"], observed=True)["settle_date"].nunique().sum()   # >= 1 after _require
    return excess / mw_days


def daily_portfolio(rows, with_alerts):
    _require(rows, ["pnl_open", "settle_date", *(["alert"] if with_alerts else [])], "daily_portfolio")
    pnl = rows["pnl_open"].to_numpy(dtype=float)
    if with_alerts:
        pnl = pnl * (1 - rows["alert"].to_numpy(dtype=float))
    return pd.Series(pnl, index=rows.index).groupby(rows["settle_date"], observed=True).sum().sort_index()


def expected_shortfall(daily, frac=0.05):
    """ES: the mean of the k = max(1, ceil(frac * D)) worst daily P&Ls, as a positive loss. Raises ValueError for an
    empty series and for one holding a NaN, which would otherwise sort last and drop out of the worst k."""
    values = np.sort(np.asarray(daily, dtype=float))
    if values.size == 0:
        raise ValueError("expected_shortfall: no daily P&L values (the series is empty)")
    n_nan = int(np.isnan(values).sum())
    if n_nan:
        raise ValueError(f"expected_shortfall: {n_nan} NaN value(s) among the daily P&Ls")
    k = max(1, math.ceil(frac * len(values)))
    return float(-values[:k].mean())


def esred(rows, frac=0.05):
    _require(rows, ["alert", "pnl_open", "settle_date"], "esred")
    return (expected_shortfall(daily_portfolio(rows, False), frac)
            - expected_shortfall(daily_portfolio(rows, True), frac))


def cap(rows):
    """CAP: the share of the total |DART| that falls in the alerted hours; ``excess`` is CAP minus the alert rate.
    By convention CAP is 0.0 when the rows hold no |DART| at all (total |DART| = 0), so ``excess`` is then minus
    the alert rate."""
    _require(rows, ["dart_settle", "alert"], "cap")
    absd = np.abs(rows["dart_settle"].to_numpy(dtype=float))
    a = rows["alert"].to_numpy(dtype=bool)
    share = float(absd[a].sum() / absd.sum()) if absd.sum() > 0 else 0.0
    return {"CAP": share, "excess": share - float(a.mean())}


def j_lambda(net, esred_value, n_days, lam):
    return float(net + lam * n_days * esred_value)


def summarize(rows, strata=STRATA_PRIMARY, lams=LAMS):
    """Every measure of one set of rows, as one flat dict.

    SELC and SNET are taken under ``strata`` (default ``STRATA_PRIMARY``, Ruling 68). ``SELC_decile`` is SELC under
    ``STRATA_DECILE``, the sensitivity, and is added only when the ``cal_decile`` column is present; it replaces the
    ``SELC_month`` of earlier versions, which is now just SELC.

    Keys, in order: LA, PG, NET, SELC, SELC_decile (only when cal_decile is present), SNET,
    ESred, ES5_without, CAP, excess, n_days, alert_rate, and one J key per lambda, named by ``j_key``:
    ``f"J_{float(lam)}"``, so the default LAMS give J_0.0, J_0.25, J_0.5 and J_1.0, and integer lams are
    formatted as floats too (lams=(0, 1) gives J_0.0, J_1.0). ``decomposition`` is given these names (its
    default ``E1_KEYS`` are NET, ESred and the four J keys), so a caller that passes other lams must name
    the matching keys, built with ``j_key``.
    """
    cols = ["alert", "pnl_open", "dart_settle", "settle_date", "market", "node", *strata]
    has_deciles = set(STRATA_DECILE) <= set(rows.columns)
    if has_deciles:
        cols += STRATA_DECILE
    _require(rows, cols, "summarize")
    out = la_pg_net(rows)
    out["SELC"] = selc(rows, strata)
    if has_deciles:
        out["SELC_decile"] = selc(rows, STRATA_DECILE)
    out["SNET"] = snet(rows, strata)
    out["ESred"] = esred(rows)
    out["ES5_without"] = expected_shortfall(daily_portfolio(rows, False))
    out.update(cap(rows))
    n_days = rows["settle_date"].nunique()
    out["n_days"] = n_days
    out["alert_rate"] = float(rows["alert"].mean())
    for lam in lams:
        out[j_key(lam)] = j_lambda(out["NET"], out["ESred"], n_days, lam)
    return out


def decomposition(v_p1, v_matched, v_alpha, keys=E1_KEYS):
    """Timing and level effects of spec §4.4: timing = V(P1) - V(P4@r*), level = V(P4@r*) - V(P4@alpha),
    total = V(P1) - V(P4@alpha), for each key. The inputs are ``summarize`` dicts (or any mapping holding the
    keys); ``keys`` defaults to ``E1_KEYS``, the measures that estimand E1 decomposes. A missing key raises."""
    return {k: {"timing": v_p1[k] - v_matched[k], "level": v_matched[k] - v_alpha[k],
                "total": v_p1[k] - v_alpha[k]} for k in keys}
