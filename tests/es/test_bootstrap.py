import datetime as dt
import hashlib
import itertools
import math
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import pytest


def test_counts_are_resamples_of_full_length():
    from es.bootstrap import stationary_counts
    c = stationary_counts(92, mean_block=7, n_rep=500, seed=1)
    assert c.shape == (500, 92) and (c.sum(axis=1) == 92).all()


def test_circular_scheme_covers_window_ends():
    from es.bootstrap import stationary_counts
    c = stationary_counts(92, mean_block=7, n_rep=4000, seed=2)
    included = (c > 0).mean(axis=0)
    assert abs(included[0] - included[46]) < 0.05


BAD_DAYS = {3, 17, 29, 41, 52}          # irregular, so no non-zero rotation realigns them all


def _rows(seed=0, n_days=30, aligned=False):
    # Ruling 68: the default strata are node x hour x weekday class x month, so the frame carries `month` (the brief's
    # helper had no such column; it carries cal_decile for the sensitivity strata).
    rng = np.random.default_rng(seed)
    out = []
    for d in range(n_days):
        day = dt.date(2025, 7, 1) + dt.timedelta(days=d)
        bad_day = d in BAD_DAYS
        for h in range(24):
            pnl = (-40.0 if (aligned and bad_day and h == 5) else rng.normal(0.5, 3.0))
            alert = (bad_day and h == 5) if aligned else bool(pnl < -4)
            out.append({"market": "M", "node": "A", "settle_date": day, "settle_hour": h,
                        "wclass": int(day.weekday() >= 5), "month": day.month, "cal_decile": h % 10,
                        "pnl_open": pnl, "dart_settle": pnl, "alert": alert})
    return pd.DataFrame(out)


def test_cube_statistics_equal_direct_measures():
    from es import measures
    from es.bootstrap import RowCube
    rows = _rows()
    cube = RowCube.from_rows(rows)
    assert np.isclose(cube.selc(), measures.selc(rows))
    assert np.isclose(cube.net(), measures.la_pg_net(rows)["NET"])
    assert np.isclose(cube.esred(), measures.esred(rows))


def test_rotation_by_full_window_is_identity_and_preserves_counts():
    from es.bootstrap import RowCube
    cube = RowCube.from_rows(_rows())
    assert np.array_equal(cube.rotated_alerts(len(cube.days)), cube.alert)
    assert cube.rotated_alerts(7).sum() == cube.alert.sum()


def test_rotation_test_detects_perfect_alignment():
    from es.bootstrap import RowCube, rotation_test
    cube = RowCube.from_rows(_rows(n_days=60, aligned=True))
    res = rotation_test([cube], min_shift=14)
    assert res["n_null"] == 60 - 2 * 14 + 1
    assert res["p"] == pytest.approx(1.0 / (1 + res["n_null"]))
    full = rotation_test([cube])                         # Ruling 69: the default reference set is the full cyclic group
    assert full["n_null"] == 60 - 1 and full["p"] == pytest.approx(1.0 / 60)


def test_pooled_rotation_null_size():
    from es.bootstrap import RowCube, rotation_test
    a = RowCube.from_rows(_rows(seed=1, n_days=40))
    b = RowCube.from_rows(_rows(seed=2, n_days=40))
    res = rotation_test([a, b], min_shift=14)
    assert res["n_null"] == (40 - 28 + 2) ** 2 - 1      # Ruling 55: each window's {0} + 13 shifts, less the identity
    assert rotation_test([a, b])["n_null"] == 40 * 40 - 1       # Ruling 69: each window's D rotations, no identity


def _scores(shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for node in ("A", "B", "C"):
        for split, start, mu in (("calib", dt.date(2024, 7, 1), 0.0), ("deploy", dt.date(2025, 7, 1), shift)):
            for d in range(40):
                day = start + dt.timedelta(days=d)
                for h in range(24):
                    rows.append({"market": "M", "node": node, "split": split, "settle_date": day,
                                 "score": rng.normal(mu, 1.0)})
    return pd.DataFrame(rows)


def test_block_contrast_zero_for_identical_blocks():
    from es.bootstrap import block_contrast_bootstrap
    s = _scores()
    res = block_contrast_bootstrap(s, s, n_rep=50)
    assert res["t_hat"] == 0.0 and np.allclose(res["t_star"], 0.0)


def test_block_contrast_positive_when_alt_drifts():
    from es.bootstrap import block_contrast_bootstrap
    res = block_contrast_bootstrap(_scores(shift=1.0, seed=3), _scores(shift=0.0, seed=4), n_rep=50)
    assert res["t_hat"] > 0.2 and res["n_nodes"] == 3


def test_p_value_and_ci():
    from es.bootstrap import basic_p_one_sided, percentile_ci
    t_star = np.random.default_rng(3).normal(1.0, 0.1, 2000)
    assert basic_p_one_sided(1.0, t_star) < 0.01
    lo, hi = percentile_ci(t_star)
    assert lo < 1.0 < hi


def _two_node_rows():
    a = _rows(seed=0)
    b = _rows(seed=5).assign(node="B")
    b = b[b["settle_date"] != b["settle_date"].min()]          # node B misses the first day
    return pd.concat([a, b], ignore_index=True)


def test_from_rows_refuses_nan_and_empty():
    from es.bootstrap import RowCube
    rows = _rows()
    with pytest.raises(ValueError, match="empty"):
        RowCube.from_rows(rows.iloc[:0])
    # the brief looped over cal_decile, which the default strata read before Ruling 68; they read wclass and month
    for col in ("wclass", "month", "pnl_open", "alert"):
        bad = rows.copy()
        bad[col] = bad[col].astype(float)
        bad.loc[3, col] = np.nan
        with pytest.raises(ValueError, match=col):
            RowCube.from_rows(bad)


def test_snet_and_j_equal_direct_measures():
    from es import measures
    from es.bootstrap import RowCube
    rows = _two_node_rows()
    cube = RowCube.from_rows(rows)
    assert np.isclose(cube.snet(), measures.snet(rows))
    s = measures.summarize(rows)
    for lam in measures.LAMS:
        assert np.isclose(cube.j_lambda(lam), s[measures.j_key(lam)])


def test_weighted_statistics_equal_the_expanded_frame():
    """Day weights w give the statistics of the frame in which day d appears w_d times, as distinct days."""
    from es import measures
    from es.bootstrap import RowCube, stationary_counts
    rows = _two_node_rows()
    cube = RowCube.from_rows(rows)
    w = stationary_counts(cube.n_days, 7, 1, seed=11)[0]
    parts = []
    for i, d in enumerate(cube.days):
        for _ in range(int(w[i])):
            parts.append(rows[rows["settle_date"] == d].assign(
                settle_date=dt.date(2030, 1, 1) + dt.timedelta(days=len(parts))))
    big = pd.concat(parts, ignore_index=True)
    assert np.isclose(cube.selc(w), measures.selc(big))
    assert np.isclose(cube.snet(w), measures.snet(big))
    assert np.isclose(cube.net(w), measures.la_pg_net(big)["NET"])
    assert np.isclose(cube.esred(w), measures.esred(big))
    assert np.isclose(cube.j_lambda(0.5, w), measures.summarize(big)["J_0.5"])


def test_block_level_bootstrap_matches_activity():
    from es.bootstrap import block_level_bootstrap
    from es.policies import activity, p1_fixed
    s = _scores(shift=0.5, seed=6)
    s = s.assign(timestamp_utc=pd.Timestamp("2024-01-01", tz="UTC")
                 + pd.to_timedelta(s.groupby(["market", "node"]).cumcount(), unit="h"))
    res = block_level_bootstrap(s, n_rep=50)
    expected = float(np.median(activity(p1_fixed(s, 0.05), 0.05)["log_error"]))
    assert res["t_hat"] == pytest.approx(expected) and res["n_nodes"] == 3
    assert res["t_star"].shape == (50,) and res["ci"][0] <= res["ci"][1]


def test_block_level_failures_match_activity():
    from es.bootstrap import block_level_bootstrap
    from es.policies import activity, p1_fixed
    s = _scores(shift=1.5, seed=8)
    s = s.assign(timestamp_utc=pd.Timestamp("2024-01-01", tz="UTC")
                 + pd.to_timedelta(s.groupby(["market", "node"]).cumcount(), unit="h"))
    res = block_level_bootstrap(s, n_rep=20, measure="failures")
    assert res["t_hat"] == float(activity(p1_fixed(s, 0.05), 0.05)["fail_0.5_2"].sum())
    with pytest.raises(ValueError):
        block_level_bootstrap(s, n_rep=2, measure="mean")


# ====================================================================================================
# Pre-review round (controller Ruling 45). Independent references for the statistics, the registered defaults, and
# one test per guard. A test that takes more than about 3 s is marked slow.
# ====================================================================================================
def _random_panel(seed, n_days=24, nodes=(("M", "A"), ("M", "B"), ("N", "A")), p_day=0.9, p_hour=0.7, p_alert=0.15,
                  start=dt.date(2025, 7, 1)):
    """An unbalanced hourly panel: nodes miss days and hours, node A exists in two markets, P&L has ties and zeros."""
    rng = np.random.default_rng(seed)
    out = []
    for market, node in nodes:
        for d in range(n_days):
            if rng.random() > p_day:
                continue
            day = start + dt.timedelta(days=d)
            for h in range(24):
                if rng.random() > p_hour:
                    continue
                pnl = float(rng.normal(0.3, 3.0))
                if rng.random() < 0.1:
                    pnl = float(round(pnl))
                if rng.random() < 0.05:
                    pnl = 0.0
                out.append({"market": market, "node": node, "settle_date": day, "settle_hour": h,
                            "wclass": int(day.weekday() >= 5), "month": day.month,
                            "cal_decile": int(rng.integers(0, 4)), "pnl_open": pnl, "dart_settle": pnl,
                            "alert": bool(rng.random() < p_alert)})
    return pd.DataFrame(out)


def _expand(rows, cube, w, alert=None):
    """The frame in which day d appears w_d times, each copy on its own distinct date."""
    r = rows.reset_index(drop=True)
    if alert is not None:
        r = r.assign(alert=np.asarray(alert) > 0.5)
    parts = []
    for i, d in enumerate(cube.days):
        day_rows = r[r["settle_date"] == d]
        for _ in range(int(w[i])):
            parts.append(day_rows.assign(settle_date=dt.date(2030, 1, 1) + dt.timedelta(days=len(parts))))
    return pd.concat(parts, ignore_index=True)


def _close(a, b, what=""):
    assert np.isclose(a, b, rtol=1e-9, atol=1e-9), f"{what}: {a!r} != {b!r}"


# ---------------------------------------------------------------------------------------------- the sampler
def _exact_count_law(n, mean_block):
    """The exact law of the count vector of the circular stationary bootstrap, by enumerating every index path."""
    p = 1.0 / mean_block
    law = {}
    for path in itertools.product(range(n), repeat=n):
        prob = 1.0 / n
        for t in range(1, n):
            prob *= p / n + ((1.0 - p) if path[t] == (path[t - 1] + 1) % n else 0.0)
        key = tuple(np.bincount(path, minlength=n).tolist())
        law[key] = law.get(key, 0.0) + prob
    return law


@pytest.mark.parametrize("n,mean_block,reps,seed",
                         [(4, 2, 60000, 1), (5, 3, 60000, 2), (4, 1, 40000, 3), (3, 7, 40000, 4)])
def test_stationary_counts_follow_the_exact_law_of_the_index_chain(n, mean_block, reps, seed):
    """Not just the first moments: the whole joint law of the count vector, against the enumerated exact law."""
    from scipy.stats import chi2
    from es.bootstrap import stationary_counts
    law = _exact_count_law(n, mean_block)
    assert abs(sum(law.values()) - 1.0) < 1e-12
    counts = stationary_counts(n, mean_block, reps, seed)
    assert counts.shape == (reps, n) and (counts.sum(axis=1) == n).all()
    seen = Counter(map(tuple, counts.tolist()))
    assert set(seen) <= set(law), "a count vector that cannot occur was drawn"
    stat, bins, pooled_expected, pooled_seen = 0.0, 0, 0.0, 0.0
    for key, prob in law.items():                    # cells with fewer than 5 expected draws are pooled into one
        expected, observed = reps * prob, seen.get(key, 0)
        if expected >= 5:
            stat += (observed - expected) ** 2 / expected
            bins += 1
        else:
            pooled_expected, pooled_seen = pooled_expected + expected, pooled_seen + observed
    if pooled_expected >= 5:
        stat += (pooled_seen - pooled_expected) ** 2 / pooled_expected
        bins += 1
    assert chi2.sf(stat, bins - 1) > 1e-4, f"chi2 = {stat:.1f} on {bins - 1} df"
    for key, prob in law.items():                    # and no single cell is off by more than five sigma
        sd = math.sqrt(reps * prob * (1 - prob)) + 1e-9
        assert abs(seen.get(key, 0) - reps * prob) < 5 * sd + 2, f"cell {key}"


def _exact_coverage(n, mean_block):
    """P(day d is among the n drawn indices), for every d, from the transition matrix of the index chain."""
    p = 1.0 / mean_block
    transition = np.full((n, n), p / n)
    transition[np.arange(n), (np.arange(n) + 1) % n] += 1.0 - p
    out = np.empty(n)
    for d in range(n):
        keep = np.arange(n) != d
        step = transition[np.ix_(keep, keep)]
        v = np.full(n - 1, 1.0 / n)                  # P(I_1 = j) for j != d
        for _ in range(n - 1):
            v = v @ step
        out[d] = 1.0 - v.sum()
    return out


@pytest.mark.parametrize("n,mean_block,reps,seed", [(92, 7, 20000, 5), (31, 4, 20000, 6)])
def test_stationary_counts_include_each_day_with_the_exact_probability(n, mean_block, reps, seed):
    from es.bootstrap import stationary_counts
    coverage = _exact_coverage(n, mean_block)
    assert np.ptp(coverage) < 1e-10, "the circular scheme must treat the window ends like any other day"
    counts = stationary_counts(n, mean_block, reps, seed)
    included = counts > 0
    se = np.sqrt(coverage * (1 - coverage) / reps)
    assert (np.abs(included.mean(axis=0) - coverage) < 5 * se).all()
    share = included.mean(axis=1)
    assert abs(share.mean() - coverage.mean()) < 5 * share.std() / math.sqrt(reps)
    assert (np.abs(counts.mean(axis=0) - 1.0) < 5 * counts.std(axis=0) / math.sqrt(reps)).all()    # E[count_d] = 1


# ------------------------------------------------------------------------------------ weighted statistics
def _explicit_es(daily, weights, frac):
    x = np.sort(np.repeat(np.asarray(daily, float), np.asarray(weights).astype(int)))
    return float(-x[:max(1, math.ceil(frac * len(x)))].mean())


def test_weighted_expected_shortfall_equals_the_explicitly_repeated_values():
    from es.bootstrap import RowCube
    rng = np.random.default_rng(12)
    for trial in range(600):
        n = int(rng.integers(1, 60))
        daily = (rng.normal(0, 5, n) if trial % 3 == 0 else np.round(rng.normal(0, 3, n)) if trial % 3 == 1
                 else np.where(rng.random(n) < 0.4, 0.0, rng.normal(0, 4, n)))                  # ties and zeros
        w = rng.integers(0, 6, n).astype(float)
        if trial % 7 == 0:
            w[rng.integers(n)] = float(rng.integers(50, 300))
        if w.sum() == 0:
            w[rng.integers(n)] = 1.0
        for frac in (0.01, 0.05, 0.1, 0.25, 0.5, 1.0):
            got, want = RowCube._es(daily, w, frac), _explicit_es(daily, w, frac)
            assert np.isclose(got, want, rtol=1e-12, atol=1e-12), f"trial {trial}, frac {frac}: {got} != {want}"


@pytest.mark.parametrize("seed", range(10))
def test_weighted_statistics_equal_the_expanded_frame_on_random_unbalanced_panels(seed):
    from es import measures
    from es.bootstrap import RowCube, stationary_counts
    rows = _random_panel(seed)
    cube = RowCube.from_rows(rows)                  # the primary strata of Ruling 68
    decile_cube = RowCube.from_rows(rows, strata=measures.STRATA_DECILE)
    rng = np.random.default_rng(1000 + seed)
    sparse = np.zeros(cube.n_days)
    sparse[int(rng.integers(cube.n_days))] = 3.0
    weight_sets = [stationary_counts(cube.n_days, 3 + seed % 5, 1, seed=seed)[0], np.ones(cube.n_days), sparse,
                   stationary_counts(cube.n_days, 2, 1, seed=seed + 77)[0]]
    for wi, w in enumerate(weight_sets):
        for override in (None, (rng.random(len(rows)) < 0.25).astype(float)):
            tag = f"seed {seed}, weights {wi}, {'override' if override is not None else 'real alerts'}"
            big = _expand(rows, cube, w, override)
            _close(cube.selc(w, override), measures.selc(big), tag + " selc")
            _close(decile_cube.selc(w, override), measures.selc(big, measures.STRATA_DECILE), tag + " selc_decile")
            _close(cube.snet(w, override), measures.snet(big), tag + " snet")
            _close(cube.net(w, override), measures.la_pg_net(big)["NET"], tag + " net")
            _close(cube.esred(w, alert=override), measures.esred(big), tag + " esred")
            _close(cube.esred(w, 0.2, override), measures.esred(big, 0.2), tag + " esred(0.2)")
            _close(cube.day_weight(w), big["settle_date"].nunique(), tag + " day_weight")
            summary = measures.summarize(big)
            for lam in measures.LAMS:
                _close(cube.j_lambda(lam, w, override), summary[measures.j_key(lam)], tag + f" J {lam}")
            excess, total = cube.selc_parts(w, override)
            _close(total, float(np.clip(-big["pnl_open"], 0, None).sum()), tag + " total loss")
            _close(cube.snet_parts(w, override)[1], big.groupby(["market", "node"])["settle_date"].nunique().sum(),
                   tag + " node-days")
            _close(excess / total if total > 0 else 0.0, measures.selc(big), tag + " excess / total")


def test_selc_is_zero_when_there_is_no_loss_as_in_measures():
    from es import measures
    from es.bootstrap import RowCube
    panel = _random_panel(6, n_days=10)
    rows = panel.assign(pnl_open=panel["pnl_open"].abs(), dart_settle=panel["pnl_open"].abs())
    cube = RowCube.from_rows(rows)
    excess, total = cube.selc_parts()
    assert total == 0.0 and excess == 0.0 and cube.selc() == 0.0 == measures.selc(rows)
    assert cube.selc(np.ones(cube.n_days) * 2) == 0.0


def test_from_arrays_called_positionally_has_one_node_unless_told_and_counts_node_days():
    """Task 17 calls from_arrays(day_idx, n_days, cell_idx, sid, pnl, alert) positionally, with a bool alert."""
    from es.bootstrap import RowCube
    rng = np.random.default_rng(0)
    n = 30
    day_idx = np.concatenate([np.repeat(np.arange(n), 24) for _ in range(2)])
    cells = np.concatenate([k * 24 + np.tile(np.arange(24), n) for k in range(2)])
    sid = np.concatenate([k * 10 + np.tile(np.arange(24), n) % 10 for k in range(2)])
    pnl, alert = rng.normal(0.5, 3, len(day_idx)), rng.random(len(day_idx)) < 0.1
    cube = RowCube.from_arrays(day_idx, n, cells, sid, pnl, alert)
    assert cube.alert.dtype == float and cube.days is None
    assert (cube.node_idx == 0).all() and np.array_equal(cube.nd_day, np.arange(n))
    assert cube.snet_parts()[1] == n and cube.day_weight() == n
    node = np.repeat([0, 1], 24 * n)
    cube2 = RowCube.from_arrays(day_idx, n, cells, sid, pnl, alert, None, node)
    assert cube2.snet_parts()[1] == 2 * n
    w = np.zeros(n)
    w[[2, 5]] = [3, 1]
    assert cube2.snet_parts(w)[1] == 2 * 4 and cube.snet_parts(w)[1] == 4
    relabelled = RowCube.from_arrays(day_idx, n, cells, sid, pnl, alert, node_idx=np.repeat([7, 3], 24 * n))
    assert np.array_equal(np.sort(relabelled.nd_day), np.sort(cube2.nd_day))     # labels only tell nodes apart
    assert cube2.snet(w) == cube2.snet_parts(w)[0] / 8 and cube2.snet(np.zeros(n)) == 0.0


def test_from_rows_validation_names_the_column():
    from es.bootstrap import RowCube
    from es.measures import STRATA_DECILE
    rows = _random_panel(4, n_days=10)
    # the primary strata read wclass and month besides the row columns; the decile strata read cal_decile instead, so
    # there the hour (which the cube needs for its cells) is checked as a row column, not as a stratum column
    for strata, columns in ((None, ("market", "node", "settle_date", "settle_hour", "pnl_open", "alert", "wclass",
                                    "month")),
                            (STRATA_DECILE, ("market", "node", "settle_date", "settle_hour", "pnl_open", "alert",
                                             "cal_decile"))):
        kwargs = {} if strata is None else {"strata": strata}
        for col in columns:
            bad = rows.copy()
            bad[col] = bad[col].astype(object)
            bad.loc[bad.index[2], col] = np.nan
            with pytest.raises(ValueError, match=rf"NaN in .*{col}"):
                RowCube.from_rows(bad, **kwargs)
            with pytest.raises(ValueError, match=rf"missing column.*{col}"):
                RowCube.from_rows(rows.drop(columns=[col]), **kwargs)
    for col in ("market", "node"):                   # read even when the strata do not name them
        bad = rows.copy()
        bad[col] = bad[col].astype(object)
        bad.loc[bad.index[2], col] = np.nan
        with pytest.raises(ValueError, match=col):
            RowCube.from_rows(bad, strata=("settle_hour",))
    bad = rows.assign(cal_decile=rows["cal_decile"].astype(float))
    bad.loc[bad.index[2], "cal_decile"] = np.nan
    RowCube.from_rows(bad)                           # the decile is not read with the primary strata (Ruling 68)
    with pytest.raises(ValueError, match="cal_decile"):
        RowCube.from_rows(bad, strata=STRATA_DECILE)
    with pytest.raises(ValueError, match=r"missing column.*cal_decile"):
        RowCube.from_rows(rows.drop(columns=["cal_decile"]), strata=STRATA_DECILE)
    RowCube.from_rows(rows.drop(columns=["cal_decile"]))
    RowCube.from_rows(rows.assign(dart_settle=np.nan))      # the cube does not read dart_settle


# ---------------------------------------------------------------------------------------------------- rotation
# The primary strata of Ruling 68, spelled out so that the references below do not import what they check.
_PRIMARY_STRATA = ("market", "node", "settle_hour", "wclass", "month")


def _brute_path(rows, cube, shift, strata=_PRIMARY_STRATA):
    """One window with every alert moved `shift` days later within its (market, node, hour) cell, in plain Python and
    from scratch: (excess, variance, total loss, alerts kept). Per stratum, with N rows, A alerted, loss L, captured
    loss C and squared losses Q: excess = sum [C - A L / N] and variance = sum over N >= 2 of A (N - A) / (N (N - 1)) *
    (Q - L^2 / N). An alert whose target (day, cell) has no row is lost."""
    pos = {d: i for i, d in enumerate(cube.days)}
    r = rows.reset_index(drop=True)
    days = [pos[d] for d in r["settle_date"]]
    cells = list(zip(r["market"], r["node"], r["settle_hour"]))
    alert_at = {(c, d): bool(a) for c, d, a in zip(cells, days, r["alert"])}
    assert len(alert_at) == len(r), "the pure-Python rotation needs unique (cell, day) rows"
    count, hits, loss, captured, squares = Counter(), Counter(), Counter(), Counter(), Counter()
    kept = 0
    for cell, day, key, pnl in zip(cells, days, zip(*[r[s] for s in strata]), r["pnl_open"]):
        hit = alert_at.get((cell, (day - shift) % cube.n_days), False)
        lost = max(-float(pnl), 0.0)
        count[key] += 1
        hits[key] += hit
        loss[key] += lost
        captured[key] += lost if hit else 0.0
        squares[key] += lost * lost
        kept += hit
    excess = sum(captured[k] - hits[k] * loss[k] / count[k] for k in count)
    variance = sum(hits[k] * (count[k] - hits[k]) / (count[k] * (count[k] - 1))
                   * max(squares[k] - loss[k] ** 2 / count[k], 0.0) for k in count if count[k] >= 2)
    return excess, variance, sum(loss.values()), kept


def _brute_paths(rows, cube, min_shift):
    """_brute_path at the identity and at every admissible shift: the window's shift set {0} u {m, ..., D - m}."""
    return [_brute_path(rows, cube, s) for s in [0] + list(range(min_shift, cube.n_days - min_shift + 1))]


def _brute_test(window_paths):
    """rotation_test by explicit enumeration of the full grid, the statistic of every cell computed from scratch:
    T = (sum of the windows' excess) / sqrt(sum of their variances), 0 where both are 0."""
    cells = list(itertools.product(*[range(len(paths)) for paths in window_paths]))     # the identity cell is first
    stats = []
    for cell in cells:
        excess = sum(window_paths[w][i][0] for w, i in enumerate(cell))
        variance = sum(window_paths[w][i][1] for w, i in enumerate(cell))
        stats.append(excess / math.sqrt(variance) if variance > 0 else 0.0)
    total = sum(paths[0][2] for paths in window_paths)
    selc = sum(paths[0][0] for paths in window_paths) / total if total > 0 else 0.0
    return {"t_obs": stats[0], "p": (1 + sum(t >= stats[0] - 1e-9 for t in stats[1:])) / len(cells),
            "n_null": len(cells) - 1, "selc_obs": selc}


def test_rotated_excess_equals_a_pure_python_rotation():
    from es.bootstrap import RowCube, rotation_excess
    for seed in (1, 2, 3):
        panel = _random_panel(seed, n_days=34)
        cube = RowCube.from_rows(panel)
        for min_shift in (None, 3, 8):                  # None: the default, every non-zero shift (Ruling 69)
            shifts, got = rotation_excess(cube) if min_shift is None else rotation_excess(cube, min_shift)
            first = 1 if min_shift is None else min_shift
            want = np.array([_brute_path(panel, cube, int(s))[0] for s in shifts])
            assert np.array_equal(shifts, np.arange(first, cube.n_days - first + 1))
            assert len(shifts) == (cube.n_days - 1 if min_shift is None else cube.n_days - 2 * first + 1)
            assert np.allclose(got, want, rtol=1e-10, atol=1e-10), f"seed {seed}, min_shift {min_shift}"


def test_rotation_test_equals_explicit_enumeration_of_the_full_grid():
    """The reference set of Rulings 55 and 69: each window's shifts are {0} u {m, ..., D - m} (m = 1, the default, makes
    that the full cyclic group {0, ..., D - 1}), the grid is their product, the identity cell is the observed one, and
    p = (1 + #{other cells with T >= T_obs}) / (number of cells)."""
    from es.bootstrap import RowCube, rotation_test
    for seed in (1, 2, 3):
        panels = [_random_panel(seed, n_days=34), _random_panel(seed + 40, n_days=31)]
        cubes = [RowCube.from_rows(panel) for panel in panels]
        for min_shift in (None, 3, 8):                  # None: the default, the full cyclic group of each window
            first = 1 if min_shift is None else min_shift
            paths = [_brute_paths(panel, cube, first) for panel, cube in zip(panels, cubes)]
            for chosen in ([0], [1], [0, 1], [1, 0]):                   # A alone, B alone, A and B pooled, B and A
                want = _brute_test([paths[i] for i in chosen])
                windows = [cubes[i] for i in chosen]
                got = rotation_test(windows) if min_shift is None else rotation_test(windows, min_shift)
                where = f"seed {seed}, min_shift {min_shift}, windows {chosen}"
                assert got["n_null"] == want["n_null"] == np.prod([len(paths[i]) for i in chosen]) - 1, where
                assert np.isclose(got["t_obs"], want["t_obs"], rtol=1e-9, atol=1e-12), where
                assert np.isclose(got["selc_obs"], want["selc_obs"], rtol=1e-9, atol=1e-12), where
                assert got["p"] == want["p"], f"{where}: p {got['p']} vs {want['p']}"


def test_rotation_ties_give_p_one():
    from es.bootstrap import RowCube, rotation_test
    base = _random_panel(9, n_days=30, p_hour=1.0, p_day=1.0)       # complete grid: on a gappy one nothing is invariant
    for name, alert in (("no alerts", False), ("all alerts", True),
                        ("the same hour every day", base["settle_hour"] == 5)):
        cube = RowCube.from_rows(base.assign(alert=alert))
        res = rotation_test([cube], min_shift=5)
        assert res["p"] == 1.0 and res["n_null"] == 30 - 2 * 5 + 1, name
        full = rotation_test([cube])                                # the default, the full cyclic group, ties alike
        assert full["p"] == 1.0 and full["n_null"] == 30 - 1, name
    worst = base.assign(alert=base["pnl_open"] < base["pnl_open"].quantile(0.03))        # alerts on the worst hours
    for kwargs in ({"min_shift": 5}, {}):                           # restricted set, then the default
        res = rotation_test([RowCube.from_rows(worst)], **kwargs)
        assert res["t_obs"] > 0 and res["p"] <= 3.0 / (1 + res["n_null"]), kwargs


def test_a_repeated_hour_is_a_cell_of_its_own_so_the_full_window_rotation_is_the_identity():
    from es import measures
    from es.bootstrap import RowCube
    rows = _random_panel(3, n_days=20, p_hour=1.0, p_day=1.0)
    first = sorted(rows["settle_date"].unique())
    mask = (rows["settle_hour"] == 1) & rows["settle_date"].isin([first[3], first[9]])      # as on a DST fall-back day
    rows = rows.assign(alert=np.where(mask, False, rows["alert"]).astype(bool))
    repeated = pd.concat([rows, rows[mask].assign(alert=True, pnl_open=-2.0, dart_settle=-2.0)], ignore_index=True)
    cube = RowCube.from_rows(repeated)
    assert cube.n_cells == 3 * 24 + 3
    assert np.array_equal(cube.rotated_alerts(cube.n_days), cube.alert)
    assert np.isclose(cube.selc(), measures.selc(repeated))


@pytest.mark.parametrize("p_hour,documented", [(1.0, 1.0), (0.995, 0.996), (0.98, 0.980), (0.95, 0.953)])
def test_rotation_loses_only_the_alerts_whose_target_cell_has_no_row(p_hour, documented):
    """The module docstring quotes these: a rotated path keeps the observed alert count exactly on a complete grid,
    and about the share of existing cells otherwise (99.6, 98.0 and 95.3% at 0.5, 2 and 5% missing hours), over the 91
    non-zero shifts of the default reference set (Ruling 69; the shifts 14..78 of the earlier default gave 98.1 and
    95.2%)."""
    from es.bootstrap import RowCube
    nodes = tuple(("M", f"n{i}") for i in range(12))
    cube = RowCube.from_rows(_random_panel(11, n_days=92, nodes=nodes, p_day=1.0, p_hour=p_hour, p_alert=0.05))
    kept = np.array([cube.rotated_alerts(s).sum() for s in range(1, 92)]) / cube.alert.sum()
    if p_hour == 1.0:
        assert (kept == 1.0).all()
    else:
        assert kept.max() < 1.0 and abs(kept.mean() - p_hour) < 0.01
    assert round(kept.mean(), 3) == documented


@pytest.mark.parametrize("n_days", [59, 60, 92])
def test_smallest_rotation_p_of_a_window_is_one_over_its_number_of_days(n_days):
    """Ruling 69. The default reference set is the full cyclic group, so a window has D cells and n_null = D - 1. A
    perfectly aligned window ranks first among them: p_min = 1 / D, and a window rejects at alpha/3 = 0.05/3 from 60
    days on (1/60 and 0.05/3 are the same double); 59 days cannot, however well aligned."""
    from es.bootstrap import RowCube, rotation_test
    res = rotation_test([RowCube.from_rows(_rows(n_days=n_days, aligned=True))])
    assert res["n_null"] == n_days - 1
    assert res["p"] == 1.0 / n_days
    assert (res["p"] <= 0.05 / 3) == (n_days >= 60)
    if n_days == 92:
        assert round(res["p"], 4) == 0.0109


@pytest.mark.parametrize("n_days", [60, 85, 86, 92])
def test_smallest_rotation_p_with_a_restricted_shift_set_is_one_over_d_minus_2m_plus_2(n_days):
    """min_shift stays a parameter (Ruling 69). With m = min_shift > 1 the shift set is {0} u {m, ..., D - m}: n_null =
    D - 2m + 1 and p_min = 1 / (D - 2m + 2), i.e. 1 / (D - 26) for m = 14: 0.0152 at D = 92, and 85 days or fewer cannot
    reject at alpha/3 (86 can). That set is not a group, so no validity is claimed for it."""
    from es.bootstrap import RowCube, rotation_test
    res = rotation_test([RowCube.from_rows(_rows(n_days=n_days, aligned=True))], min_shift=14)
    assert res["n_null"] == n_days - 2 * 14 + 1
    assert res["p"] == 1.0 / (n_days - 2 * 14 + 2)
    assert (res["p"] <= 0.05 / 3) == (n_days >= 86)
    if n_days == 92:
        assert round(res["p"], 4) == 0.0152


# ------------------------------------------------------------------- the calibration-and-deployment bootstrap
def _score_frame(seed, shift, nodes, n_cal, n_dep, rounded=False, start_cal=dt.date(2024, 7, 1),
                 start_dep=dt.date(2025, 7, 1), hours=24, node_shift=None, node_keep=None):
    """Scores of `nodes` in the calibration and deployment splits. `node_shift` gives a node's deployment mean (default
    `shift`); `node_keep` the (calibration, deployment) day indices a node has (default: every day)."""
    node_shift, node_keep = node_shift or {}, node_keep or {}
    rng = np.random.default_rng(seed)
    rows = []
    for market, node in nodes:
        for split, start, n, mu in (("calib", start_cal, n_cal, 0.0),
                                    ("deploy", start_dep, n_dep, node_shift.get((market, node), shift))):
            keep = node_keep.get((market, node))
            for d in range(n):
                if keep is not None and d not in keep[0 if split == "calib" else 1]:
                    continue
                for _ in range(hours):
                    score = rng.normal(mu, 1.0)
                    rows.append({"market": market, "node": node, "split": split,
                                 "settle_date": start + dt.timedelta(days=d),
                                 "score": round(score, 1) if rounded else score})
    frame = pd.DataFrame(rows)
    return frame.assign(timestamp_utc=pd.Timestamp("2024-01-01", tz="UTC")
                        + pd.to_timedelta(frame.groupby(["market", "node"]).cumcount(), unit="h"))


def _days_of(frames, split):
    return sorted(set().union(*(set(f.loc[f["split"] == split, "settle_date"]) for f in frames)))


def _policy_view(frame, cal_days, dep_days, wc, wd, alpha, allow_skips=False):
    """activity(p1_fixed(...)) of the frame in which every calibration and deployment day is repeated by its weight."""
    from es.policies import activity, p1_fixed
    cal_pos, dep_pos = {d: i for i, d in enumerate(cal_days)}, {d: i for i, d in enumerate(dep_days)}
    parts = []
    for split, pos, weights in (("calib", cal_pos, wc), ("deploy", dep_pos, wd)):
        for day, day_rows in frame[frame["split"] == split].groupby("settle_date"):
            for _ in range(int(weights[pos[day]])):
                offset = np.arange(len(day_rows)) + 10_000 * len(parts)
                parts.append(day_rows.assign(timestamp_utc=pd.Timestamp("2000-01-01", tz="UTC")
                                             + pd.to_timedelta(offset, unit="h")))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore" if allow_skips else "error")      # otherwise no node may be skipped silently
        return activity(p1_fixed(pd.concat(parts, ignore_index=True), alpha), alpha).set_index(["market", "node"])


# two ordinary nodes, and two zero-activity nodes with missing days (so that the floor binds and the weighted n varies)
_L1_NODES = (("M", "A"), ("N", "A"), ("M", "Z1"), ("N", "Z2"))
_L1_SHIFT = {("M", "Z1"): -6.0, ("N", "Z2"): -6.0}
_L1_KEEP = {("M", "Z1"): (set(range(22)) - {3, 7, 11, 15, 19}, set(range(21)) - {2, 6, 10, 14}),
            ("N", "Z2"): (set(range(22)) - {0, 5, 9, 20}, set(range(21)) - {1, 8, 13, 19})}


@pytest.mark.parametrize("rounded", [False, True])          # rounded scores make `score >= tau` ties common
def test_bootstrap_replicates_equal_the_policy_pipeline_on_explicitly_resampled_data(rounded):
    from es.bootstrap import block_contrast_bootstrap, block_level_bootstrap, stationary_counts
    alt = _score_frame(21, 0.6, _L1_NODES, 22, 21, rounded=rounded, hours=6, node_shift=_L1_SHIFT, node_keep=_L1_KEEP)
    base = _score_frame(22, 0.1, _L1_NODES, 22, 21, rounded=rounded, hours=6, node_shift=_L1_SHIFT, node_keep=_L1_KEEP)
    base_missing = base[~((base["node"] == "A") & (base["market"] == "N") & (base["split"] == "deploy")
                          & (base["settle_date"] < dt.date(2025, 7, 4)))]
    last_cal, last_dep = max(_days_of([base], "calib")), max(_days_of([base], "deploy"))
    base_short = base[~(((base["split"] == "calib") & (base["settle_date"] > last_cal - dt.timedelta(days=2)))
                        | ((base["split"] == "deploy") & (base["settle_date"] > last_dep - dt.timedelta(days=3))))]
    base_no_node = base[~((base["market"] == "N") & (base["node"] == "A"))]
    n_rep, mean_block, seed, alpha = 5, 4, 77, 0.1
    cal_days, dep_days = _days_of([alt, base], "calib"), _days_of([alt, base], "deploy")
    assert len(cal_days) != len(dep_days)
    cal_counts = stationary_counts(len(cal_days), mean_block, n_rep, seed)
    dep_counts = stationary_counts(len(dep_days), mean_block, n_rep, seed + 1)
    for pair, frames in (("balanced", [alt, base]), ("base misses days", [alt, base_missing]),
                         ("base covers fewer days", [alt, base_short]), ("base lacks a node", [alt, base_no_node])):
        contrast = block_contrast_bootstrap(frames[0], frames[1], alpha=alpha, n_rep=n_rep, mean_block=mean_block,
                                            seed=seed)
        level = block_level_bootstrap(frames[0], alpha=alpha, n_rep=n_rep, mean_block=mean_block, seed=seed)
        failures = block_level_bootstrap(frames[0], alpha=alpha, n_rep=n_rep, mean_block=mean_block, seed=seed,
                                         measure="failures")
        for b in [None] + list(range(n_rep)):                # None: the real days, i.e. t_hat
            wc = np.ones(len(cal_days)) if b is None else cal_counts[b]
            wd = np.ones(len(dep_days)) if b is None else dep_counts[b]
            act = [_policy_view(f, cal_days, dep_days, wc, wd, alpha) for f in frames]
            common = sorted(set(act[0].index) & set(act[1].index))
            want_contrast = float(np.median([act[0].log_error[k] - act[1].log_error[k] for k in common]))
            want_level = float(np.median([act[0].log_error[k] for k in act[0].index]))
            want_failures = float(sum(act[0]["fail_0.5_2"][k] for k in act[0].index))
            got = [(r["t_hat"] if b is None else r["t_star"][b]) for r in (contrast, level, failures)]
            tag = f"{pair}, replicate {b}"
            assert np.isclose(got[0], want_contrast, rtol=1e-9, atol=1e-12), \
                f"{tag}: contrast {got[0]} vs {want_contrast}"
            assert np.isclose(got[1], want_level, rtol=1e-9, atol=1e-12), f"{tag}: median {got[1]} vs {want_level}"
            assert got[2] == want_failures, f"{tag}: failures {got[2]} vs {want_failures}"
        assert contrast["n_nodes"] == len(common) and level["n_nodes"] == 4 and failures["n_nodes"] == 4
    real = _policy_view(alt, cal_days, dep_days, np.ones(len(cal_days)), np.ones(len(dep_days)), alpha)
    assert real.loc[[("M", "Z1"), ("N", "Z2")], "zero_activity"].all()      # the floor really is exercised


def test_a_sparse_node_drops_out_of_a_replicate_that_gives_it_no_calibration_or_deployment_weight():
    """p1_fixed skips a node with no calibration or no deployment rows; the bootstrap must do the same, not floor it."""
    from es.bootstrap import block_contrast_bootstrap, block_level_bootstrap, stationary_counts
    nodes = (("M", "A"), ("N", "A"), ("M", "X"))
    keep = {("M", "X"): ({4}, {3})}                          # one calibration day and one deployment day
    alt = _score_frame(31, 0.5, nodes, 22, 21, hours=6, node_keep=keep)
    base = _score_frame(32, 0.0, nodes, 22, 21, hours=6, node_keep=keep)
    n_rep, mean_block, seed, alpha = 14, 2, 5, 0.1
    cal_days, dep_days = _days_of([alt, base], "calib"), _days_of([alt, base], "deploy")
    cal_counts = stationary_counts(len(cal_days), mean_block, n_rep, seed)
    dep_counts = stationary_counts(len(dep_days), mean_block, n_rep, seed + 1)
    x_rows = alt[alt["node"] == "X"]
    x_cal = [i for i, d in enumerate(cal_days) if d in set(x_rows[x_rows["split"] == "calib"]["settle_date"])]
    x_dep = [i for i, d in enumerate(dep_days) if d in set(x_rows[x_rows["split"] == "deploy"]["settle_date"])]
    no_cal = [b for b in range(n_rep) if cal_counts[b][x_cal].sum() == 0 and dep_counts[b][x_dep].sum() > 0]
    no_dep = [b for b in range(n_rep) if dep_counts[b][x_dep].sum() == 0 and cal_counts[b][x_cal].sum() > 0]
    assert no_cal and no_dep, "both empty-resample cases must occur, or this test checks nothing"
    contrast = block_contrast_bootstrap(alt, base, alpha=alpha, n_rep=n_rep, mean_block=mean_block, seed=seed)
    level = block_level_bootstrap(alt, alpha=alpha, n_rep=n_rep, mean_block=mean_block, seed=seed)
    for b in range(n_rep):
        act = [_policy_view(f, cal_days, dep_days, cal_counts[b], dep_counts[b], alpha, allow_skips=True)
               for f in (alt, base)]
        both = sorted(set(act[0].index) & set(act[1].index))
        want = np.median([act[0].log_error[k] - act[1].log_error[k] for k in both])
        assert np.isclose(contrast["t_star"][b], want, rtol=1e-9, atol=1e-12), f"contrast, replicate {b}"
        assert np.isclose(level["t_star"][b], np.median(act[0].log_error), rtol=1e-9, atol=1e-12), \
            f"median, replicate {b}"


def test_failure_rule_and_zero_activity_floor_match_activity_at_the_band_edges():
    """R = 0.5 and R = 2 exactly are not failures (activity uses strict < 0.5 and > 2); just outside they are. A
    node with no alert is floored at R = 1 / (2 n alpha) in the median, and fails the band."""
    from es.bootstrap import block_level_bootstrap
    from es.policies import activity, p1_fixed
    rows = []
    calibration = np.arange(1000, dtype=float)               # tau (alpha = 5%) = 949.05
    # n = 800 deployment rows per node, so R = alerts / 40
    for node, alerts in (("e05", 20), ("e2", 80), ("lo", 19), ("hi", 81), ("mid", 40), ("zero", 0)):
        deployment = np.concatenate([np.full(alerts, 5000.0), np.full(800 - alerts, 1.0)])
        for split, values in (("calib", calibration), ("deploy", deployment)):
            for i, value in enumerate(values):
                rows.append({"market": "M", "node": node, "split": split,
                             "settle_date": dt.date(2024, 1, 1) + dt.timedelta(days=i // 10), "score": value,
                             "timestamp_utc": pd.Timestamp("2024-01-01", tz="UTC")
                             + pd.Timedelta(hours=i + (0 if split == "calib" else 5000))})
    frame = pd.DataFrame(rows)
    act = activity(p1_fixed(frame, 0.05), 0.05).set_index("node")
    assert act["R"].round(6).to_dict() == {"e05": 0.5, "e2": 2.0, "lo": 0.475, "hi": 2.025, "mid": 1.0, "zero": 0.0}
    assert act["fail_0.5_2"].sum() == 3
    assert block_level_bootstrap(frame, n_rep=2, measure="failures")["t_hat"] == 3.0
    median = block_level_bootstrap(frame, n_rep=2)["t_hat"]
    assert np.isclose(median, np.median(act["log_error"]), rtol=1e-12)
    assert np.isclose(act.loc["zero", "log_error"], abs(math.log10(1 / (2 * 800 * 0.05))))
    with pytest.raises(ValueError, match="unknown measure"):
        block_level_bootstrap(frame, n_rep=2, measure="mean")


def test_basic_p_value_on_hand_worked_cases():
    """p = (1 + #{t* - t_hat >= t_hat}) / (B + 1), worked by hand on small cases."""
    from es.bootstrap import basic_p_one_sided
    # t* - t_hat = [-.5, 0, .5, .5, 1.5], three of them >= t_hat = .5: (1 + 3) / 6
    assert basic_p_one_sided(0.5, [0.0, 0.5, 1.0, 1.0, 2.0]) == pytest.approx(4 / 6)
    # t* - t_hat = [-1, -.5, 0, .5, 1, 2], two of them >= 1: (1 + 2) / 7
    assert basic_p_one_sided(1.0, [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]) == pytest.approx(3 / 7)
    # a negative t_hat: t* - t_hat = [-1, 0, 1, 2] are all >= -1, so p = 1
    assert basic_p_one_sided(-1.0, [-2.0, -1.0, 0.0, 1.0]) == 1.0
    assert basic_p_one_sided(0.0, np.zeros(5)) == 1.0       # every resample ties t_hat = 0: no evidence at all
    assert basic_p_one_sided(100.0, np.zeros(9)) == pytest.approx(1 / 10)       # none reaches 2 * t_hat: the smallest p


def test_basic_p_value_rejects_at_about_the_nominal_rate_when_the_null_is_true():
    """The property the formula exists for, not the formula: bootstrap the mean of n = 40 draws centred on 0 and test
    H0: mean <= 0 with the basic p. Over 600 samples it rejects at about its level."""
    from es.bootstrap import basic_p_one_sided
    rng = np.random.default_rng(17)
    p = []
    for _ in range(600):
        x = rng.normal(0.0, 1.0, 40)
        resamples = rng.choice(x, size=(399, 40)).mean(axis=1)
        p.append(basic_p_one_sided(x.mean(), resamples))
    p = np.array(p)
    assert abs(np.mean(p <= 0.05) - 0.05) < 0.025 and abs(np.mean(p <= 0.2) - 0.2) < 0.045
    shifted = []                                              # and with a real effect (mean 0.8) it nearly always does
    for _ in range(100):
        x = rng.normal(0.8, 1.0, 40)
        shifted.append(basic_p_one_sided(x.mean(), rng.choice(x, size=(399, 40)).mean(axis=1)))
    assert np.mean(np.array(shifted) <= 0.05) > 0.95


def test_p_value_monotonicity_and_interval_levels_and_ties():
    from es.bootstrap import basic_p_one_sided, block_contrast_bootstrap, percentile_ci
    rng = np.random.default_rng(5)
    t_star = rng.normal(0, 1, 500)
    p_by_t_hat = [basic_p_one_sided(t, t_star) for t in np.linspace(-1.5, 1.5, 31)]
    assert all(a >= b for a, b in zip(p_by_t_hat, p_by_t_hat[1:])), "p must not rise with t_hat"
    assert basic_p_one_sided(100.0, t_star) == 1.0 / 501 and basic_p_one_sided(-100.0, t_star) == 1.0
    for level in (0.95, 0.9, 0.5):
        assert percentile_ci(t_star, level) == (float(np.percentile(t_star, 100 * (1 - level) / 2)),
                                                float(np.percentile(t_star, 100 * (1 - (1 - level) / 2))))
    lo, hi = percentile_ci(t_star)                           # the default level is 95%
    assert (t_star < lo).mean() < 0.03 and (t_star > hi).mean() < 0.03 and lo < 0 < hi
    identical = _score_frame(3, 0.0, (("M", "A"), ("M", "B")), 12, 12, hours=3)
    res = block_contrast_bootstrap(identical, identical, n_rep=20)
    assert res["t_hat"] == 0.0 and res["p"] == 1.0           # identical blocks: no evidence at all


def test_registered_defaults_are_pinned():
    """Spec v3 section 5.1 and 6.3 as amended: mean block 7, B = 2000, seed 20261022, shifts from 1 day (the full
    cyclic group of each window, Ruling 69), 95% intervals, ES at 5%, P1 at 5%, the primary strata of Ruling 68; and the
    argument order that Tasks 11, 12 and 17 rely on."""
    import dataclasses
    import inspect
    from es import bootstrap as b
    from es.measures import STRATA_PRIMARY

    def defaults(fn):
        return {k: v.default for k, v in inspect.signature(fn).parameters.items()
                if v.default is not inspect.Parameter.empty}
    assert defaults(b.stationary_counts) == {"mean_block": 7, "n_rep": 2000, "seed": 20261022}
    assert defaults(b.block_contrast_bootstrap) == {"alpha": 0.05, "n_rep": 2000, "mean_block": 7, "seed": 20261022}
    assert defaults(b.block_level_bootstrap) == {"alpha": 0.05, "n_rep": 2000, "mean_block": 7, "seed": 20261022,
                                                 "measure": "median", "aligned": False}          # Ruling 118: off by default
    assert "aligned" not in inspect.signature(b.block_contrast_bootstrap).parameters       # H-B keeps the registered scheme
    assert defaults(b.rotation_excess) == {"min_shift": 1} and defaults(b.rotation_test) == {"min_shift": 1}
    assert defaults(b.percentile_ci) == {"level": 0.95}
    assert defaults(b.RowCube.esred) == {"w": None, "frac": 0.05, "alert": None}
    assert defaults(b.RowCube.from_rows)["strata"] == STRATA_PRIMARY
    assert list(defaults(b.RowCube.from_rows)["strata"]) == ["market", "node", "settle_hour", "wclass", "month"]
    assert list(inspect.signature(b.RowCube.from_arrays).parameters) == [
        "day_idx", "n_days", "cell_idx", "sid", "pnl", "alert", "days", "node_idx"]
    assert [f.name for f in dataclasses.fields(b.RowCube)][-6:] == [
        "days", "node_idx", "nd_day", "strat_n", "strat_loss", "strat_ss"]


# ------------------------------------------------------------------------------------- guards (Ruling 45)
def test_p_value_refuses_a_nan_statistic_or_resample():
    """A NaN statistic used to give the smallest possible p, 1 / (B + 1), as if it were the strongest evidence."""
    from es.bootstrap import basic_p_one_sided
    for t_hat, t_star in ((float("nan"), np.zeros(20)), (0.1, np.array([0.1, np.nan, 0.3])),
                          (float("nan"), np.full(20, np.nan))):
        with pytest.raises(ValueError, match="NaN"):
            basic_p_one_sided(t_hat, t_star)
    assert basic_p_one_sided(0.0, np.zeros(20)) == 1.0       # clean input still gives a p-value


def test_block_bootstraps_refuse_frames_that_share_no_node():
    from es.bootstrap import block_contrast_bootstrap, block_level_bootstrap
    alt = _score_frame(1, 0.0, (("M", "A"), ("M", "B")), 12, 12, hours=2)
    other = _score_frame(2, 0.0, (("M", "C"), ("M", "D")), 12, 12, hours=2)
    with pytest.raises(ValueError, match="no node"):
        block_contrast_bootstrap(alt, other, n_rep=5)
    # a node counts only if it has calibration AND deployment rows in every frame
    split_apart = pd.concat([alt[(alt["node"] == "A") & (alt["split"] == "calib")],
                             alt[(alt["node"] == "B") & (alt["split"] == "deploy")]], ignore_index=True)
    with pytest.raises(ValueError, match="no node"):
        block_level_bootstrap(split_apart, n_rep=5)
    with pytest.raises(ValueError, match="no node"):
        block_contrast_bootstrap(alt, split_apart, n_rep=5)
    with pytest.raises(ValueError, match="no node"):
        block_level_bootstrap(alt[alt["split"] == "calib"], n_rep=5)                # no deployment rows at all
    one_node = block_contrast_bootstrap(alt, alt[alt["node"] == "A"], n_rep=5)
    assert one_node["n_nodes"] == 1                                                  # one common node is enough


@pytest.mark.parametrize("column", ["market", "node", "split", "settle_date", "score"])
def test_block_bootstraps_validate_the_score_frames_and_name_the_column(column):
    from es.bootstrap import block_contrast_bootstrap, block_level_bootstrap
    frame = _score_frame(1, 0.0, (("M", "A"), ("M", "B")), 12, 12, hours=2)
    bad = frame.assign(**{column: frame[column].astype(object)})
    bad.loc[bad.index[3], column] = np.nan
    for call in (lambda f: block_level_bootstrap(f, n_rep=3), lambda f: block_contrast_bootstrap(f, frame, n_rep=3),
                 lambda f: block_contrast_bootstrap(frame, f, n_rep=3)):
        with pytest.raises(ValueError, match=rf"NaN in .*{column}"):
            call(bad)
        with pytest.raises(ValueError, match=rf"missing column.*{column}"):
            call(frame.drop(columns=[column]))


def test_block_bootstraps_refuse_an_empty_score_frame_and_ignore_columns_they_do_not_read():
    from es.bootstrap import block_contrast_bootstrap, block_level_bootstrap
    frame = _score_frame(1, 0.0, (("M", "A"), ("M", "B")), 12, 12, hours=2)
    for call in (lambda f: block_level_bootstrap(f, n_rep=3), lambda f: block_contrast_bootstrap(f, frame, n_rep=3)):
        with pytest.raises(ValueError, match="empty"):
            call(frame.iloc[:0])
    noisy = frame.assign(note=np.nan, timestamp_utc=pd.NaT)
    assert block_level_bootstrap(noisy, n_rep=3)["t_hat"] == block_level_bootstrap(frame, n_rep=3)["t_hat"]


def test_rotation_test_refuses_a_window_whose_null_would_be_empty():
    """n_null = D - 2 * min_shift + 1: 27 days leave none with min_shift 14, 28 days leave one. The default min_shift
    of 1 needs 2 days, whose single rotation is the weakest null there is."""
    from es.bootstrap import RowCube, rotation_test

    def cube(n):
        return RowCube.from_rows(_random_panel(3, n_days=n, p_hour=1.0, p_day=1.0))
    one_day, two_days = cube(1), cube(2)
    with pytest.raises(ValueError, match="1 days"):
        rotation_test([one_day])
    res = rotation_test([two_days])
    assert res["n_null"] == 1 and res["p"] in (0.5, 1.0)
    short, edge, longer = cube(27), cube(28), cube(40)
    assert (short.n_days, edge.n_days, longer.n_days) == (27, 28, 40)
    with pytest.raises(ValueError, match="27 days"):
        rotation_test([short], min_shift=14)
    res = rotation_test([edge], min_shift=14)                # 2 * min_shift days leave one rotation: a weak null
    assert res["n_null"] == 1 and res["p"] in (0.5, 1.0)
    with pytest.raises(ValueError, match="27 days"):
        rotation_test([longer, short], min_shift=14)         # one short window is enough
    with pytest.raises(ValueError, match="40 days"):
        rotation_test([longer], min_shift=21)
    assert rotation_test([longer], min_shift=20)["n_null"] == 1


def test_rotation_test_with_no_loss_returns_p_one_and_the_true_null_size():
    """SELC is 0.0 when there is no loss, so no rotation can beat it: T 0.0 and p 1.0, not a ZeroDivisionError."""
    from es.bootstrap import RowCube, rotation_test

    def no_loss(seed, n):
        rows = _random_panel(seed, n_days=n, p_hour=1.0, p_day=1.0)
        return RowCube.from_rows(rows.assign(pnl_open=rows["pnl_open"].abs(), dart_settle=rows["pnl_open"].abs()))
    a, b = no_loss(3, 40), no_loss(4, 35)
    single, pooled = rotation_test([a]), rotation_test([a, b])
    assert (single["t_obs"], single["p"], single["n_null"]) == (0.0, 1.0, 40 - 1)
    assert (pooled["t_obs"], pooled["p"], pooled["n_null"]) == (0.0, 1.0, 40 * 35 - 1)
    assert single["selc_obs"] == pooled["selc_obs"] == 0.0
    restricted = rotation_test([a, b], min_shift=14)                 # and the restricted set of Ruling 55 still works
    assert (restricted["t_obs"], restricted["p"], restricted["n_null"]) == (0.0, 1.0, (40 - 27 + 1) * (35 - 27 + 1) - 1)
    with_loss = RowCube.from_rows(_random_panel(5, n_days=40, p_hour=1.0, p_day=1.0))
    mixed, alone = rotation_test([a, with_loss]), rotation_test([with_loss])
    assert mixed["n_null"] == 40 * 40 - 1                    # a window with no loss still has 39 shifts + the identity
    assert (mixed["t_obs"], mixed["p"]) == (alone["t_obs"], alone["p"])      # but it adds neither excess nor variance


def test_rotation_test_needs_at_least_one_window():
    from es.bootstrap import rotation_test
    with pytest.raises(ValueError, match="at least one"):
        rotation_test([])


def test_categorical_key_columns_give_the_same_numbers_and_no_pandas_warning():
    """Without observed=True pandas warns (FutureWarning) in each groupby on a categorical key; es.measures is clean."""
    from es import measures
    from es.bootstrap import RowCube, block_level_bootstrap
    rows = _random_panel(5)
    categories = {"market": ["M", "N", "Z"], "node": ["A", "B", "Z"], "wclass": [0, 1, 2]}    # "Z" and 2 are never used
    cat = rows.assign(**{c: pd.Categorical(rows[c], categories=k) for c, k in categories.items()})
    plain = RowCube.from_rows(rows)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cube = RowCube.from_rows(cat)
        decile = RowCube.from_rows(cat, strata=measures.STRATA_DECILE)
    assert (cube.n_cells, cube.n_strata) == (plain.n_cells, plain.n_strata)
    assert cube.selc() == plain.selc() and cube.snet() == plain.snet() and np.array_equal(cube.nd_day, plain.nd_day)
    assert np.array_equal(cube.rotated_alerts(7), plain.rotated_alerts(7))
    assert decile.selc() == RowCube.from_rows(rows, strata=measures.STRATA_DECILE).selc()
    scores = _score_frame(1, 0.3, (("M", "A"), ("N", "A")), 12, 12, hours=2)
    cat_scores = scores.assign(market=pd.Categorical(scores["market"], ["M", "N", "Z"]),
                               node=pd.Categorical(scores["node"], ["A", "Z"]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = block_level_bootstrap(cat_scores, n_rep=4)
    want = block_level_bootstrap(scores, n_rep=4)
    assert res["t_hat"] == want["t_hat"] and np.array_equal(res["t_star"], want["t_star"]) and res["n_nodes"] == 2


def test_day_weights_must_have_one_entry_per_day():
    """Weights longer than n_days used to be accepted silently (the first n_days were used)."""
    from es.bootstrap import RowCube
    cube = RowCube.from_rows(_two_node_rows())
    n = cube.n_days
    good = np.arange(n) % 3 + 1.0
    statistics = {"selc_parts": cube.selc_parts, "selc": cube.selc, "snet_parts": cube.snet_parts, "snet": cube.snet,
                  "net": cube.net, "esred": cube.esred, "day_weight": cube.day_weight,
                  "j_lambda": lambda w: cube.j_lambda(0.5, w)}
    for name, statistic in statistics.items():
        statistic(None)
        assert np.all(np.asarray(statistic(good)) == np.asarray(statistic(list(good)))), name      # a list is fine
        for bad in (np.ones(n + 1), np.ones(n - 1), np.ones(2 * n), np.ones((n, 1)), np.ones((1, n)), np.array([]),
                    3.0):
            with pytest.raises(ValueError, match="day weights"):
                statistic(bad)
    with pytest.raises(ValueError, match="day weights"):
        cube.esred(np.ones(n + 1), alert=cube.alert)
    with pytest.raises(ValueError, match="day weights"):
        cube.net(np.ones(n + 1), cube.alert)


def test_from_arrays_refuses_a_repeated_day_cell_pair():
    """Two rows in one (day, cell) would silently overwrite each other when the alerts are rotated."""
    from es.bootstrap import RowCube
    day_idx, cell_idx = np.repeat(np.arange(5), 3), np.tile(np.arange(3), 5)
    pnl, alert, sid = np.linspace(-2, 2, 15), np.arange(15) % 4 == 0, np.zeros(15, dtype=int)
    cube = RowCube.from_arrays(day_idx, 5, cell_idx, sid, pnl, alert)
    assert cube.n_days == 5 and cube.n_cells == 3
    with pytest.raises(ValueError, match=r"day 2, cell 1"):
        RowCube.from_arrays(np.append(day_idx, 2), 5, np.append(cell_idx, 1), np.append(sid, 0),
                            np.append(pnl, 1.0), np.append(alert, False))
    with pytest.raises(ValueError, match=r"2 \(day_idx, cell_idx\) pair"):
        RowCube.from_arrays(np.append(day_idx, [2, 4]), 5, np.append(cell_idx, [1, 0]), np.append(sid, [0, 0]),
                            np.append(pnl, [1.0, 2.0]), np.append(alert, [False, True]))
    # the same cell on different days, and the same day in different cells, are fine
    RowCube.from_arrays(np.append(day_idx, 4), 5, np.append(cell_idx, 3), np.append(sid, 0), np.append(pnl, 1.0),
                        np.append(alert, False))


# --------------------------------------------------------------------------- fix round 1 (controller Ruling 55)
@pytest.mark.parametrize("min_shift,cells", [(14, 66), (1, 92)])
def test_pooling_with_a_window_that_has_no_alerts_reproduces_the_single_window_p_and_statistic(min_shift, cells):
    """The reviewer's case. A window ranking 2nd of 66 gave p = 2/66 alone and 0.0156 (below alpha/3) pooled with a
    window without alerts, because the cells in which one window stays unrotated were missing from the reference set.
    66 cells are the shift set {0} u {14, ..., 78} of 92 days; the default min_shift of 1 gives the full cyclic group of
    92 cells, where rank 2 is p = 2/92 = 0.0217, still above alpha/3."""
    from es.bootstrap import RowCube, rotation_test
    n_days, hours = 92, 24
    day_idx, cell = np.repeat(np.arange(n_days), hours), np.tile(np.arange(hours), n_days)

    def cube(rng, rate):
        return RowCube.from_arrays(day_idx, n_days, cell, cell % 10, rng.normal(0.5, 3, n_days * hours),
                                   rng.random(n_days * hours) < rate)
    rng = np.random.default_rng(1)
    for _ in range(5000):
        a = cube(rng, 0.05)
        if rotation_test([a], min_shift)["p"] == 2 / cells:
            break
    else:
        pytest.fail(f"no window ranked 2nd of {cells} in 5000 draws")
    quiet = cube(rng, 0.0)
    alone = rotation_test([a], min_shift)
    assert alone["p"] == 2 / cells and alone["n_null"] == cells - 1
    for pooled in (rotation_test([a, quiet], min_shift), rotation_test([quiet, a], min_shift)):
        assert (pooled["p"], pooled["t_obs"]) == (alone["p"], alone["t_obs"])
        assert pooled["n_null"] == cells * cells - 1 and pooled["p"] > 0.05 / 3
    for seed in range(5):                                           # not only that window
        b = cube(np.random.default_rng(100 + seed), 0.08)
        single, pooled = rotation_test([b], min_shift), rotation_test([b, quiet], min_shift)
        assert (pooled["p"], pooled["t_obs"]) == (single["p"], single["t_obs"])


def test_rotation_test_reports_the_pooled_selc_next_to_the_studentized_statistic():
    from es.bootstrap import RowCube, rotation_test
    a, b = (RowCube.from_rows(_random_panel(s, n_days=34)) for s in (1, 2))
    res = rotation_test([a, b], min_shift=8)
    assert set(res) == {"t_obs", "p", "n_null", "selc_obs", "retained_share"}
    pooled_selc = (a.selc_parts()[0] + b.selc_parts()[0]) / (a.selc_parts()[1] + b.selc_parts()[1])
    assert res["selc_obs"] == pytest.approx(pooled_selc, rel=1e-12)
    assert rotation_test([a], min_shift=8)["selc_obs"] == pytest.approx(a.selc(), rel=1e-12)
    (excess_a, var_a), (excess_b, var_b) = a.excess_var(), b.excess_var()
    assert res["t_obs"] == pytest.approx((excess_a + excess_b) / math.sqrt(var_a + var_b), rel=1e-12)
    signal = rotation_test([RowCube.from_rows(_rows(n_days=60, aligned=True))])      # alerts exactly on the worst hours
    assert signal["t_obs"] > 5 and 0 < signal["selc_obs"] < 1      # a z-like statistic and a share of the loss


def test_retained_share_is_the_share_of_observed_alerts_that_survive_each_nonzero_rotation():
    from es.bootstrap import RowCube, rotation_test
    complete = RowCube.from_rows(_random_panel(2, n_days=40, p_hour=1.0, p_day=1.0))
    gappy_panel = _random_panel(3, n_days=40, p_hour=0.9, p_day=0.95)
    gappy = RowCube.from_rows(gappy_panel)
    quiet = RowCube.from_rows(_random_panel(4, n_days=40, p_hour=1.0, p_day=1.0).assign(alert=False))
    shares = rotation_test([complete, gappy, quiet])["retained_share"]
    assert len(shares) == 3 and all(set(s) == {"min", "mean"} for s in shares)
    assert shares[0] == {"min": 1.0, "mean": 1.0}                   # a complete grid keeps every alert
    paths = _brute_paths(gappy_panel, gappy, 1)
    observed = paths[0][3]
    want = [path[3] / observed for path in paths[1:]]               # the identity is not a rotation
    assert len(want) == 40 - 1                                      # the default reference set: every non-zero shift
    assert shares[1]["min"] == pytest.approx(min(want)) and shares[1]["mean"] == pytest.approx(np.mean(want))
    assert 0.5 < shares[1]["min"] < shares[1]["mean"] < 1.0
    assert math.isnan(shares[2]["min"]) and math.isnan(shares[2]["mean"])       # no observed alerts: no share to report
    restricted = rotation_test([gappy], min_shift=14)["retained_share"][0]      # the shifts 14..26 alone
    want14 = [path[3] / observed for path in _brute_paths(gappy_panel, gappy, 14)[1:]]
    assert len(want14) == 40 - 27
    assert restricted["min"] == pytest.approx(min(want14)) and restricted["mean"] == pytest.approx(np.mean(want14))


def test_the_cube_precomputes_per_stratum_counts_losses_and_spread():
    from es.bootstrap import RowCube
    cube = RowCube.from_rows(_random_panel(7))
    frame = pd.DataFrame({"sid": cube.sid, "loss": cube.loss})
    by = frame.groupby("sid")["loss"]
    spread = ((frame["loss"] - by.transform("mean")) ** 2).groupby(frame["sid"]).sum()
    assert np.array_equal(cube.strat_n, by.size().to_numpy())
    assert np.allclose(cube.strat_loss, by.sum().to_numpy(), rtol=1e-12, atol=1e-12)
    assert np.allclose(cube.strat_ss, spread.to_numpy(), rtol=1e-9, atol=1e-9)
    assert (cube.strat_ss >= 0).all() and len(cube.strat_n) == cube.n_strata


def test_the_variance_in_the_statistic_is_the_exact_variance_of_captured_loss_under_random_selection():
    """Enumerate every way of choosing the alerted rows within each stratum (SELC's reference) on strata small enough to
    list: the captured loss has mean A L / N and the variance the statistic divides by. Strata are independent."""
    from es.bootstrap import RowCube
    losses = [[0.0, 1.0, 4.0, 2.5, 0.0, 7.0], [3.0, 3.0, 0.5, 9.0], [2.0], [1.0, 5.0, 5.0], [0.0, 0.0, 0.0]]
    sid = np.concatenate([np.full(len(v), k) for k, v in enumerate(losses)])
    flat = np.concatenate(losses)
    n_rows = len(flat)
    cube0 = RowCube.from_arrays(np.zeros(n_rows, dtype=int), 1, np.arange(n_rows), sid, -flat, np.zeros(n_rows))
    # stratum 0: 3 of 6; 1: 2 of 4; 2: its only row; 3: all three rows; 4: one of three rows that all lose nothing
    for counts in ([3, 2, 1, 3, 1], [1, 1, 0, 3, 0], [5, 3, 1, 0, 2]):
        alert = np.zeros(len(flat))
        for k, n_alert in enumerate(counts):                         # the first n_alert rows of each stratum
            alert[np.flatnonzero(sid == k)[:n_alert]] = 1.0
        choices = [list(itertools.combinations(range(len(v)), c)) for v, c in zip(losses, counts)]
        totals = np.array([sum(sum(losses[k][i] for i in pick[k]) for k in range(len(losses)))
                           for pick in itertools.product(*choices)])
        excess, variance = cube0.excess_var(alert)
        assert excess == pytest.approx(float(alert @ flat) - totals.mean(), abs=1e-12)
        assert variance == pytest.approx(totals.var(), rel=1e-12, abs=1e-12), counts
    assert cube0.excess_var(np.zeros(len(flat))) == (0.0, 0.0)       # no alerts: nothing captured, nothing to vary
    assert cube0.excess_var(np.ones(len(flat))) == (0.0, 0.0)        # every row alerted: nothing was selected


@pytest.mark.parametrize("what", ["pnl", "alert"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_pnl_or_alert_is_refused_before_it_can_reach_rotation_test(what, bad):
    """One NaN alert used to give the smallest possible p, 1/66; a NaN P&L gave selc() = 0.0 silently."""
    from es.bootstrap import RowCube, rotation_test
    n_days, hours = 40, 24
    day_idx, cell = np.repeat(np.arange(n_days), hours), np.tile(np.arange(hours), n_days)
    rng = np.random.default_rng(0)
    arrays = {"pnl": rng.normal(0.5, 3, n_days * hours), "alert": (rng.random(n_days * hours) < 0.05).astype(float)}
    base = RowCube.from_arrays(day_idx, n_days, cell, cell % 10, arrays["pnl"], arrays["alert"])
    assert np.isfinite(rotation_test([base])["t_obs"])
    arrays[what] = arrays[what].copy()
    arrays[what][100] = bad
    with pytest.raises(ValueError, match=rf"{what} holds 1 non-finite value"):       # named so, for the alert too
        RowCube.from_arrays(day_idx, n_days, cell, cell % 10, arrays["pnl"], arrays["alert"])


@pytest.mark.parametrize("field", ["alert", "loss", "strat_n", "strat_loss", "strat_ss"])
def test_rotation_test_refuses_a_cube_that_was_corrupted_after_construction(field):
    """Including the per-stratum statistics: a NaN count reads as an empty stratum and would leave a finite T."""
    from es.bootstrap import RowCube, rotation_test
    n_days, hours = 40, 24
    day_idx, cell = np.repeat(np.arange(n_days), hours), np.tile(np.arange(hours), n_days)
    rng = np.random.default_rng(0)

    def make():
        return RowCube.from_arrays(day_idx, n_days, cell, cell % 10, rng.normal(0.5, 3, n_days * hours),
                                   rng.random(n_days * hours) < 0.05)
    good, broken = make(), make()
    getattr(broken, field)[3 if field.startswith("strat") else 100] = np.nan
    match = "0 or 1" if field == "alert" else "finite"           # a NaN alert is not a 0 or 1 (the alert guard)
    with pytest.raises(ValueError, match=match):
        rotation_test([good, broken])
    with pytest.raises(ValueError, match=match):
        rotation_test([broken])


def test_a_cube_built_without_the_per_stratum_statistics_is_refused_by_the_variance():
    import dataclasses
    from es.bootstrap import RowCube, rotation_test
    n_days, hours = 40, 24
    day_idx, cell = np.repeat(np.arange(n_days), hours), np.tile(np.arange(hours), n_days)
    rng = np.random.default_rng(0)
    cube = RowCube.from_arrays(day_idx, n_days, cell, cell % 10, rng.normal(0.5, 3, n_days * hours),
                               rng.random(n_days * hours) < 0.05)
    bare = dataclasses.replace(cube, strat_n=None, strat_loss=None, strat_ss=None)
    with pytest.raises(ValueError, match="missing"):
        bare.excess_var()
    with pytest.raises(ValueError, match="missing"):
        rotation_test([bare])


_INDEX_CASES = [
    ("day_idx", np.arange(10) / 2.0, "day_idx must hold integers"),                       # floats, even if integral
    ("day_idx", np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, -1]), r"day_idx must lie in \[0, 5\)"),
    ("day_idx", np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 5]), r"day_idx must lie in \[0, 5\)"),
    ("cell_idx", np.array([0, 1] * 4 + [1, -1]), "cell_idx must not be negative"),
    ("cell_idx", np.tile([0.0, 1.0], 5), "cell_idx must hold integers"),
    ("sid", np.array([0, 1] * 4 + [1, -1]), "sid must not be negative"),
    ("sid", np.tile([0.0, 1.0], 5), "sid must hold integers"),
    ("node_idx", np.array([0] * 9 + [-1]), "node_idx must not be negative"),
    ("node_idx", np.zeros(10), "node_idx must hold integers"),
    ("pnl", np.arange(9.0), "one entry per row"),
    ("alert", np.zeros(11), "one entry per row"),
]


@pytest.mark.parametrize("field,bad,match", _INDEX_CASES)
def test_from_arrays_refuses_malformed_indices_and_lengths(field, bad, match):
    from es.bootstrap import RowCube
    good = {"day_idx": np.repeat(np.arange(5), 2), "cell_idx": np.tile(np.arange(2), 5),
            "sid": np.tile(np.arange(2), 5),
            "pnl": np.linspace(-1, 1, 10), "alert": np.arange(10) % 2, "node_idx": np.zeros(10, dtype=int)}
    RowCube.from_arrays(good["day_idx"], 5, good["cell_idx"], good["sid"], good["pnl"], good["alert"],
                        node_idx=good["node_idx"])
    good[field] = bad
    with pytest.raises(ValueError, match=match):
        RowCube.from_arrays(good["day_idx"], 5, good["cell_idx"], good["sid"], good["pnl"], good["alert"],
                            node_idx=good["node_idx"])


def test_from_arrays_refuses_no_rows():
    from es.bootstrap import RowCube
    empty = np.array([], dtype=int)
    with pytest.raises(ValueError, match="no rows"):
        RowCube.from_arrays(empty, 5, empty, empty, np.array([]), np.array([]))


@pytest.mark.filterwarnings("ignore::RuntimeWarning")           # the NaN propagates through numpy first
def test_selc_and_snet_refuse_a_non_finite_total_instead_of_returning_zero():
    from es.bootstrap import RowCube
    cube = RowCube.from_rows(_two_node_rows())
    weights = np.ones(cube.n_days)
    weights[3] = np.nan
    broken_alerts = [cube.alert.copy(), cube.alert.copy()]
    broken_alerts[0][2], broken_alerts[1][2] = np.nan, np.inf
    for statistic in (cube.selc, cube.snet):
        with pytest.raises(ValueError, match="finite"):
            statistic(weights)
        for alert in broken_alerts:
            with pytest.raises(ValueError, match="finite"):
                statistic(alert=alert)
    assert np.isfinite(cube.selc()) and np.isfinite(cube.snet())
    no_loss = RowCube.from_rows(_two_node_rows().assign(pnl_open=lambda d: d["pnl_open"].abs()))
    assert no_loss.selc() == 0.0                                   # a finite zero total keeps its convention


@pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning")
def test_failures_of_a_replicate_with_no_evaluable_node_is_nan_not_zero():
    """With no node to evaluate there is no failure count: the bootstrap distribution must say so, not show a 0."""
    from es.bootstrap import block_level_bootstrap, stationary_counts
    rng = np.random.default_rng(0)
    rows = []
    for node in ("X", "Y"):                                          # one calibration day and one deployment day each
        for split, start, day in (("calib", dt.date(2024, 1, 1), 4), ("deploy", dt.date(2024, 6, 1), 3)):
            rows += [{"market": "M", "node": node, "split": split, "settle_date": start + dt.timedelta(days=day),
                      "score": float(rng.normal())} for _ in range(24)]
    for split, start in (("calib", dt.date(2024, 1, 1)), ("deploy", dt.date(2024, 6, 1))):   # pad nodes fix day sets
        rows += [{"market": "M", "node": "P_" + split, "split": split, "settle_date": start + dt.timedelta(days=d),
                  "score": float(rng.normal())} for d in range(20)]
    frame = pd.DataFrame(rows)
    seed, n_rep = 20261022, 200
    failures = block_level_bootstrap(frame, n_rep=n_rep, measure="failures", seed=seed)
    median = block_level_bootstrap(frame, n_rep=n_rep, seed=seed)
    calibration, deployment = stationary_counts(20, 7, n_rep, seed), stationary_counts(20, 7, n_rep, seed + 1)
    empty = (calibration[:, 4] == 0) | (deployment[:, 3] == 0)       # X and Y vanish together
    assert failures["n_nodes"] == 2 and empty.sum() > 20 and (~empty).sum() > 20
    assert np.isnan(failures["t_star"][empty]).all() and np.isnan(median["t_star"][empty]).all()
    assert np.isfinite(failures["t_star"][~empty]).all() and (failures["t_star"][~empty] >= 0).all()
    assert np.isfinite(failures["t_hat"])


def test_failures_count_the_evaluable_nodes_when_only_some_nodes_vanish_from_a_replicate():
    """NaN means that no node could be evaluated. If one node is left, its failure (or not) is the count."""
    from es.bootstrap import block_level_bootstrap, stationary_counts
    rng = np.random.default_rng(1)
    rows = []
    for split, start in (("calib", dt.date(2024, 1, 1)), ("deploy", dt.date(2024, 6, 1))):
        for node, days in (("Z", range(20)), ("X", [4])):            # Z has every day, X a single day
            rows += [{"market": "M", "node": node, "split": split, "settle_date": start + dt.timedelta(days=d),
                      "score": float(rng.normal())} for d in days for _ in range(24)]
    frame = pd.DataFrame(rows)
    seed, n_rep = 20261022, 200
    failures = block_level_bootstrap(frame, n_rep=n_rep, measure="failures", seed=seed)
    calibration, deployment = stationary_counts(20, 7, n_rep, seed), stationary_counts(20, 7, n_rep, seed + 1)
    x_gone = (calibration[:, 4] == 0) | (deployment[:, 4] == 0)
    assert failures["n_nodes"] == 2 and x_gone.sum() > 20 and (~x_gone).sum() > 20
    assert np.isfinite(failures["t_star"]).all()                     # Z is always there to be counted
    assert set(np.unique(failures["t_star"])) <= {0.0, 1.0, 2.0}
    assert failures["t_star"][x_gone].max() <= 1.0                   # one evaluable node: at most one failure


def test_basic_interval_reflects_the_percentile_interval_about_t_hat():
    from es.bootstrap import basic_ci, block_contrast_bootstrap, block_level_bootstrap, percentile_ci
    # q.025 = 2.5 and q.975 = 97.5 of 0..100, so the interval is (20 - 97.5, 20 - 2.5)
    assert basic_ci(10.0, np.arange(101.0)) == pytest.approx((-77.5, 17.5))
    assert basic_ci(1.0, np.linspace(0, 2, 201), level=0.5) == (pytest.approx(0.5), pytest.approx(1.5))
    alt = _score_frame(41, 0.5, (("M", "A"), ("M", "B"), ("N", "A")), 20, 20, hours=3)
    base = _score_frame(42, 0.0, (("M", "A"), ("M", "B"), ("N", "A")), 20, 20, hours=3)
    contrast = block_contrast_bootstrap(alt, base, n_rep=80)
    level = block_level_bootstrap(alt, n_rep=80)
    assert set(contrast) == {"t_hat", "t_star", "p", "ci", "ci_basic", "n_nodes"}
    assert set(level) == {"t_hat", "t_star", "ci", "ci_basic", "n_nodes"}
    for res in (contrast, level):
        q_lo, q_hi = np.quantile(res["t_star"], [0.025, 0.975])
        assert res["ci"] == percentile_ci(res["t_star"])                             # still the percentile interval
        assert res["ci"] == pytest.approx((q_lo, q_hi), rel=1e-9)
        assert res["ci_basic"] == pytest.approx((2 * res["t_hat"] - q_hi, 2 * res["t_hat"] - q_lo), rel=1e-9)
        assert res["ci_basic"][0] <= res["ci_basic"][1]


# --------------------------------------------------------------------------- fix round 4 (controller Rulings 68, 69)
def test_the_default_strata_are_the_primary_strata_of_ruling_68_and_the_sensitivity_strata_are_a_choice():
    """node x hour x weekday class x month by default; node x calendar-propensity decile when asked for."""
    from es import measures
    from es.bootstrap import RowCube
    rows = _random_panel(8, n_days=70)                              # July to mid September: three months
    cube, decile = RowCube.from_rows(rows), RowCube.from_rows(rows, strata=measures.STRATA_DECILE)
    primary = rows.groupby(["market", "node", "settle_hour", "wclass", "month"], sort=True).ngroup().to_numpy()
    deciles = rows.groupby(["market", "node", "cal_decile"], sort=True).ngroup().to_numpy()
    assert cube.n_strata == primary.max() + 1 and np.array_equal(cube.sid, primary)
    assert decile.n_strata == deciles.max() + 1 and np.array_equal(decile.sid, deciles)
    assert cube.n_strata > 3 * decile.n_strata                      # finer strata: many more of them
    assert cube.selc() == pytest.approx(measures.selc(rows), rel=1e-12)
    assert decile.selc() == pytest.approx(measures.selc(rows, measures.STRATA_DECILE), rel=1e-12)
    assert cube.selc() != decile.selc()                             # and a different number, the point of Ruling 68
    assert measures.STRATA_MONTH == measures.STRATA_PRIMARY         # the alias that keeps old imports working
    assert RowCube.from_rows(rows, strata=measures.STRATA_MONTH).selc() == cube.selc()


def _null_window(rng, n_days, hours=24, rate=0.1):
    """A complete day x hour grid, strata hour % 4, independent N(0.5, 3) P&L and alerts: a true null."""
    from es.bootstrap import RowCube
    day, hour = np.repeat(np.arange(n_days), hours), np.tile(np.arange(hours), n_days)
    return RowCube.from_arrays(day, n_days, hour, hour % 4, rng.normal(0.5, 3.0, n_days * hours),
                               rng.random(n_days * hours) < rate)


@pytest.mark.parametrize("n_days", [12, 31, 92])
def test_a_single_windows_p_under_the_default_is_k_over_its_number_of_days(n_days):
    """Ruling 69: the reference set is the D rotations of the window, n_null = D - 1 and p = (1 + #{rotations with T >=
    T_obs}) / D = k / D with k = 1, ..., D. Two windows give k / (D_1 D_2) on the product grid. k is not a constant
    either: over 40 null windows it takes many values."""
    from es.bootstrap import rotation_test
    rng = np.random.default_rng(n_days)
    ks = []
    for _ in range(40):
        res = rotation_test([_null_window(rng, n_days)])
        k = res["p"] * n_days
        assert res["n_null"] == n_days - 1
        assert abs(k - round(k)) < 1e-9 and 1 <= round(k) <= n_days, res["p"]
        ks.append(round(k))
    assert len(set(ks)) > 5 and max(ks) > n_days // 2 and min(ks) < n_days // 2          # a spread of ranks
    pooled = rotation_test([_null_window(rng, n_days), _null_window(rng, n_days - 1)])     # D_1 = D, D_2 = D - 1
    cells = n_days * (n_days - 1)
    k = pooled["p"] * cells
    assert pooled["n_null"] == cells - 1 and abs(k - round(k)) < 1e-6 and 1 <= round(k) <= cells


def _exchangeable_cube(n_days, hours, seed):
    """A complete day x hour grid with fixed P&L of stratum-dependent spread and a base alert path; strata hour % 3."""
    from es.bootstrap import RowCube
    rng = np.random.default_rng(seed)
    day, hour = np.repeat(np.arange(n_days), hours), np.tile(np.arange(hours), n_days)
    pnl = 0.5 + (1.0 + (hour % 3)) * rng.normal(size=n_days * hours)
    alert = (rng.random(n_days * hours) < 0.2).astype(float)
    return RowCube.from_arrays(day, n_days, hour, hour % 3, pnl, alert)


def test_p_values_are_exactly_uniform_when_the_observed_path_is_a_uniformly_random_rotation_of_a_base_path():
    """The validity claim of Ruling 69, with no simulation. If the observed alert path is the base path rotated by a
    uniformly random shift u, it is exchangeable with its D rotations, and the set the test rotates through is the
    same D paths whatever u was (on a complete grid a rotation of a rotation is a rotation). Each u then has its own
    rank among the same D statistics, so the D p-values are exactly {1/D, 2/D, ..., 1} and P(p <= a) <= a for every a. A
    shift set that leaves any rotation out is no longer the orbit of the observed path, and this fails."""
    import dataclasses
    from es.bootstrap import rotation_test
    n_days = 15
    base = _exchangeable_cube(n_days, hours=6, seed=3)
    p = []
    for u in range(n_days):
        cube = dataclasses.replace(base, alert=base.rotated_alerts(u))
        res = rotation_test([cube])
        assert res["n_null"] == n_days - 1
        p.append(res["p"])
    ranks = np.sort(p) * n_days
    assert np.allclose(ranks, np.arange(1, n_days + 1), atol=1e-9), np.round(ranks, 6)
    for level in np.arange(1, n_days + 1) / n_days:
        assert np.mean(np.array(p) <= level + 1e-12) <= level + 1e-12


def test_pooled_p_values_are_exactly_uniform_on_the_product_of_the_two_rotation_groups():
    """The same for two windows: the grid of the windows' rotations is the orbit of the pair under the product group, so
    over all D_1 x D_2 equally likely pairs of shifts the p-values are exactly {1/(D_1 D_2), ..., 1}."""
    import dataclasses
    from es.bootstrap import rotation_test
    a, b = _exchangeable_cube(7, hours=6, seed=4), _exchangeable_cube(9, hours=6, seed=5)
    p = []
    for u in range(a.n_days):
        for v in range(b.n_days):
            cubes = [dataclasses.replace(a, alert=a.rotated_alerts(u)),
                     dataclasses.replace(b, alert=b.rotated_alerts(v))]
            p.append(rotation_test(cubes)["p"])
    cells = a.n_days * b.n_days
    assert np.allclose(np.sort(p) * cells, np.arange(1, cells + 1), atol=1e-9)


@pytest.mark.parametrize("bad", [2.0, -1.0, 0.5, 1.0 + 1e-9, 1e-9, 3, np.nan, np.inf, -np.inf])
def test_alert_values_must_be_zero_or_one_at_construction_and_in_the_variance(bad):
    """Re-review, New Breakage 2: an alert is a yes/no flag. The variance A (N - A) / (N (N - 1)) Q is that of choosing
    A rows only if A counts rows, so a 2.0 or a 0.5 gave a statistic with no meaning (and no error)."""
    from es.bootstrap import RowCube, rotation_test
    n_days, hours = 40, 24
    day_idx, cell = np.repeat(np.arange(n_days), hours), np.tile(np.arange(hours), n_days)
    rng = np.random.default_rng(0)
    pnl, alert = rng.normal(0.5, 3, n_days * hours), (rng.random(n_days * hours) < 0.05).astype(float)

    def build(a):
        return RowCube.from_arrays(day_idx, n_days, cell, cell % 10, pnl, a)
    good = build(alert)
    wrong = alert.copy()
    wrong[100] = bad
    with pytest.raises(ValueError, match="alert"):
        build(wrong)
    with pytest.raises(ValueError, match="0 or 1"):
        good.excess_var(wrong)
    corrupted = build(alert.copy())                                  # (a float array is not copied by the cube)
    corrupted.alert[100] = bad                                       # a cube changed after construction
    with pytest.raises(ValueError, match="0 or 1"):
        corrupted.excess_var()
    with pytest.raises(ValueError, match="0 or 1"):
        rotation_test([good, corrupted])
    assert np.isfinite(good.excess_var(alert)[1]) and good.excess_var(alert) == good.excess_var()


def test_alert_values_of_zero_and_one_are_accepted_whatever_their_type():
    from es.bootstrap import RowCube
    n_days, hours = 20, 24
    day_idx, cell = np.repeat(np.arange(n_days), hours), np.tile(np.arange(hours), n_days)
    rng = np.random.default_rng(1)
    pnl, flags = rng.normal(0.5, 3, n_days * hours), rng.random(n_days * hours) < 0.1
    reference = RowCube.from_arrays(day_idx, n_days, cell, cell % 10, pnl, flags.astype(float))
    for kind in (flags, flags.astype(int), flags.astype(np.int8), flags.astype(float), list(flags.astype(int)),
                 np.where(flags, 1.0, -0.0)):
        cube = RowCube.from_arrays(day_idx, n_days, cell, cell % 10, pnl, kind)
        assert cube.excess_var() == reference.excess_var() and cube.excess_var(kind) == reference.excess_var()
    from es.bootstrap import rotation_test
    from_bool = RowCube.from_arrays(day_idx, n_days, cell, cell % 10, pnl, flags)
    assert rotation_test([reference])["p"] == rotation_test([from_bool])["p"]


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, False, 2.0, np.float64(3.0), np.bool_(True), None, "3"])
def test_min_shift_must_be_an_integer_of_at_least_one(bad):
    """The shift set is {0} u {min_shift, ..., D - min_shift}. A min_shift of 0 would count the observed path again as
    the shifts 0 and D, a negative or fractional one is no day count, and a bool is an int nobody means as 1 day."""
    from es.bootstrap import rotation_excess, rotation_test
    cube = _null_window(np.random.default_rng(0), 20)
    for call in (lambda: rotation_test([cube], min_shift=bad), lambda: rotation_test([cube], bad),
                 lambda: rotation_excess(cube, min_shift=bad), lambda: rotation_excess(cube, bad)):
        with pytest.raises(ValueError, match="min_shift must be an integer of at least 1"):
            call()


def test_min_shift_accepts_python_and_numpy_integers_and_gives_the_same_numbers():
    from es.bootstrap import rotation_excess, rotation_test
    cube = _null_window(np.random.default_rng(1), 20)
    want = rotation_test([cube], min_shift=3)
    assert want["n_null"] == 20 - 2 * 3 + 1
    shifts, excess = rotation_excess(cube, 3)
    for good in (np.int64(3), np.int32(3), np.uint8(3), np.int8(3)):
        got = rotation_test([cube], min_shift=good)
        assert (got["t_obs"], got["p"], got["n_null"]) == (want["t_obs"], want["p"], want["n_null"])
        got_shifts, got_excess = rotation_excess(cube, good)
        assert np.array_equal(got_shifts, shifts) and np.array_equal(got_excess, excess)
    assert rotation_test([cube], min_shift=1)["n_null"] == 19 == rotation_test([cube])["n_null"]


def test_alert_values_other_than_zero_and_one_are_refused_by_from_rows():
    from es.bootstrap import RowCube
    rows = _rows()
    with pytest.raises(ValueError, match="alert"):
        RowCube.from_rows(rows.assign(alert=rows["alert"].astype(float) * 2))
    RowCube.from_rows(rows.assign(alert=rows["alert"].astype(float)))


def _weekday_window(rng, effect=0.0):
    """92 days x 24 hours whose strata are the days of the week. Alerts (12% on weekday 0, 1% elsewhere) and the
    dispersion of the P&L (SD 10 on weekday 0, 1 elsewhere) are tied to the weekday only, so given its stratum an alert
    is independent of the loss: the null is true. A rotation keeps an alert on weekday 0 only when the shift is a
    multiple of 7, so the rotated paths put their alerts where the loss is far less dispersed. effect > 0 adds that
    loss on every alerted row."""
    n_days, hours = 92, 24
    day, hour = np.repeat(np.arange(n_days), hours), np.tile(np.arange(hours), n_days)
    weekday = day % 7
    alert = rng.random(n_days * hours) < np.where(weekday == 0, 0.12, 0.01)
    pnl = 0.5 + np.where(weekday == 0, 10.0, 1.0) * rng.normal(size=n_days * hours) - effect * alert
    from es.bootstrap import RowCube
    return RowCube.from_arrays(day, n_days, hour, weekday, pnl, alert)


@pytest.mark.slow
def test_size_with_stratum_dependent_loss_dispersion_stays_below_nominal_and_the_test_still_has_power():
    """The case the studentized statistic is for, under the default reference set (the full cyclic group of Ruling 69).
    Unstudentized, the same simulation rejects about 6% of true nulls at alpha/3 and 17% at 0.05 (the shifted paths are
    less dispersed than the observed one; 2,000 windows); the studentized test rejects about 0.6% and 3.2%. The bounds
    below, 0.03 and 0.07 over 500 windows, sit about 7 and 5 standard deviations above the studentized rates; the
    bound at 0.05 is 6 standard deviations below the unstudentized rate (the one at alpha/3 is 3 below it)."""
    from es.bootstrap import rotation_test
    rng = np.random.default_rng(2026)
    p = np.array([rotation_test([_weekday_window(rng)])["p"] for _ in range(500)])
    assert np.mean(p <= 0.05 / 3) <= 0.03 and np.mean(p <= 0.05) <= 0.07
    rng = np.random.default_rng(2027)                                # and it is not simply a test that never rejects
    q = np.array([rotation_test([_weekday_window(rng, effect=4.0)])["p"] for _ in range(80)])
    assert np.mean(q <= 0.05) > 0.8


@pytest.mark.slow
def test_pooled_size_with_stratum_dependent_loss_dispersion_stays_below_nominal():
    """Two such windows pooled: the full grid and the studentized statistic together (the old test rejected about 12% at
    alpha/3 and 20% at 0.05 here; the unstudentized statistic on the full grid, 11% and 19%)."""
    from es.bootstrap import rotation_test
    rng = np.random.default_rng(2028)
    p = np.array([rotation_test([_weekday_window(rng), _weekday_window(rng)])["p"] for _ in range(300)])
    assert np.mean(p <= 0.05 / 3) <= 0.045 and np.mean(p <= 0.05) <= 0.085


def _persistent_null_window(rng, phi):
    """92 days x 24 hours, strata hour % 10, and a true null: the alert path and the P&L are independent. Alerts come
    from a latent daily AR(1) process z (about 5% of the hours) and the dispersion of the P&L from another, u; both
    have day-to-day correlation phi, so alerts cluster on days and so does the loss dispersion (the reviewer's design
    for Ruling 69)."""
    from es.bootstrap import RowCube
    n_days, hours = 92, 24
    day, hour = np.repeat(np.arange(n_days), hours), np.tile(np.arange(hours), n_days)

    def ar1():
        x = np.empty(n_days)
        x[0] = rng.normal()
        for k in range(1, n_days):
            x[k] = phi * x[k - 1] + math.sqrt(1 - phi ** 2) * rng.normal()
        return x
    z, u = ar1(), ar1()
    alert = 0.8 * z[day] + 0.6 * rng.normal(size=n_days * hours) > 1.645
    pnl = 0.5 + 3.0 * np.exp(0.5 * u[day]) * rng.normal(size=n_days * hours)
    return RowCube.from_arrays(day, n_days, hour, hour % 10, pnl, alert)


@pytest.mark.slow
def test_size_under_day_to_day_persistence_stays_at_the_attainable_level_of_the_full_cyclic_group():
    """Ruling 69. With the full cyclic group the smallest p of a window is 1/92 = 0.0109 and p <= 0.05 needs a rank of 4
    or better (4/92 = 0.0435): those are the levels the test can attain, and under persistence (phi = 0.9, a true null)
    it stays at them: about 1.1% and 5.1% over 3,000 simulations. The set {0} u {14, ..., 78} of Ruling 55 is not a
    group and rejected 5.0% and 9.2% of the same nulls. The bounds, 0.03 and 0.08 over 500 windows, are 4 and 3
    standard deviations above the full group's rates."""
    from es.bootstrap import rotation_test
    rng = np.random.default_rng(2029)
    p = np.array([rotation_test([_persistent_null_window(rng, 0.9)])["p"] for _ in range(500)])
    assert np.mean(p <= 0.05 / 3) <= 0.03 and np.mean(p <= 0.05) <= 0.08


# ---------------------------------------------------------------------------------------------------------------
# Ruling 118 (pre-registration review M2): the calendar-aligned option of the level bootstrap
# ---------------------------------------------------------------------------------------------------------------
def _calendar_scores(cal_range, dep_range, drop=(), seed=0, nodes=(("M", "A"), ("M", "B"), ("N", "A")), hours=24):
    """Scores that are a fixed function of (node, month, day, hour): one table indexed by the calendar, read by both
    windows. A calibration day and the deployment day with the same (month, day) carry the same numbers, as a
    calendar-driven score (the G6 block) does. ``cal_range`` and ``dep_range`` are (first, last) days, inclusive;
    ``drop`` are days that no node has."""
    rng = np.random.default_rng(seed)
    table = {node: rng.random((12, 31, hours)) for node in nodes}
    rows = []
    for market, node in nodes:
        for split, (first, last) in (("calib", cal_range), ("deploy", dep_range)):
            for k in range((last - first).days + 1):
                day = first + dt.timedelta(days=k)
                if day in drop:
                    continue
                for h in range(hours):
                    rows.append({"market": market, "node": node, "split": split, "settle_date": day,
                                 "score": float(table[(market, node)][day.month - 1, day.day - 1, h])})
    frame = pd.DataFrame(rows)
    return frame.assign(timestamp_utc=pd.Timestamp("2024-01-01", tz="UTC")
                        + pd.to_timedelta(frame.groupby(["market", "node"]).cumcount(), unit="h"))


_Q3_CAL = (dt.date(2024, 7, 1), dt.date(2024, 9, 30))           # the same calendar quarter, one year apart: 92 days each
_Q3_DEP = (dt.date(2025, 7, 1), dt.date(2025, 9, 30))


def test_aligned_interval_of_a_calendar_driven_score_contains_the_estimate_and_is_much_narrower():
    """Ruling 118 (M2). The score depends on (month, day, hour) only, so the two quarters hold the same numbers on the
    same dates and the real R is as near 1 as the data allow. Independent draws break that matching in every replicate
    and the interval then measures the re-randomized calendar; the aligned draw keeps the matching."""
    from es.bootstrap import block_level_bootstrap
    scores = _calendar_scores(_Q3_CAL, _Q3_DEP)
    aligned = block_level_bootstrap(scores, n_rep=400, aligned=True)
    independent = block_level_bootstrap(scores, n_rep=400)
    lo, hi = aligned["ci"]
    # t_hat is the smallest |log10 R| a replicate can take here (no ties among the weights), so lo equals t_hat
    assert lo <= aligned["t_hat"] <= hi
    assert (hi - lo) < 0.2 * (independent["ci"][1] - independent["ci"][0])     # 9 to 23 times narrower over 12 fixtures
    assert independent["ci"][0] > independent["t_hat"]       # the registered interval lies above its own estimate


def _pin_frames():
    nodes = (("M", "A"), ("M", "B"), ("N", "A"), ("N", "C"))
    keep = {("M", "B"): (set(range(26)) - {4, 11}, set(range(23)) - {7})}          # node B misses three days
    alt = _score_frame(61, 0.5, nodes, 26, 23, rounded=True, hours=5, node_keep=keep)      # rounded: ties; 26 != 23 days
    base = _score_frame(62, 0.0, nodes, 26, 23, hours=5)
    return alt, base


def _pin_sparse_frame():
    """Two nodes with one calibration day and one deployment day each, and two padding nodes that fix the day sets: a
    replicate that draws neither day of them has no node to evaluate, so its statistic is NaN."""
    rng = np.random.default_rng(3)
    rows = []
    for node in ("X", "Y"):
        for split, start, day in (("calib", dt.date(2024, 1, 1), 4), ("deploy", dt.date(2024, 6, 1), 3)):
            rows += [{"market": "M", "node": node, "split": split, "settle_date": start + dt.timedelta(days=day),
                      "score": float(rng.normal())} for _ in range(24)]
    for split, start in (("calib", dt.date(2024, 1, 1)), ("deploy", dt.date(2024, 6, 1))):
        rows += [{"market": "M", "node": "P_" + split, "split": split, "settle_date": start + dt.timedelta(days=d),
                  "score": float(rng.normal())} for d in range(20)]
    return pd.DataFrame(rows)


def _pin_results(**level_kwargs):
    """The four calls the pin covers; ``level_kwargs`` go to the level calls only (the contrast has no such option)."""
    from es.bootstrap import block_contrast_bootstrap, block_level_bootstrap
    alt, base = _pin_frames()
    return {"level": block_level_bootstrap(alt, n_rep=40, **level_kwargs),
            "failures": block_level_bootstrap(alt, n_rep=40, measure="failures", **level_kwargs),
            "contrast": block_contrast_bootstrap(alt, base, n_rep=40),
            "sparse": block_level_bootstrap(_pin_sparse_frame(), n_rep=200, **level_kwargs)}       # NaN replicates


def _pin_digest(result):
    """sha256 of a bootstrap result, bit for bit: its keys in sorted order, an array by its bytes, a float by its hex."""
    parts = []
    for key in sorted(result):
        value = result[key]
        if isinstance(value, np.ndarray):
            parts.append(f"{key}=" + np.ascontiguousarray(value, dtype="<f8").tobytes().hex())
        elif isinstance(value, tuple):
            parts.append(f"{key}=" + ",".join(float(x).hex() for x in value))
        elif isinstance(value, float):
            parts.append(f"{key}={value.hex()}")
        else:
            parts.append(f"{key}={value!r}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def _pin_digests(results):
    """{call: (sha256 of its t_star, sha256 of the whole result)}."""
    return {name: (hashlib.sha256(np.ascontiguousarray(r["t_star"], dtype="<f8").tobytes()).hexdigest(), _pin_digest(r))
            for name, r in results.items()}


#: The digests of ``_pin_results()``, computed on es/bootstrap.py as it stood before Ruling 118 added ``aligned`` (the
#: replicates, the dict keys and every value), then pasted here.
PIN_BEFORE_RULING_118 = {
    "level": ("f915f166a222e1e9032e00515292e4090b4ccd1b953958d187f30831f84dfe9c",
              "92dc1cf0053062cbae823eabb3f38e8c364c966823ad8fc0066397efa4c5d92a"),
    "failures": ("95af63ad7e4ea3bd14948f5e697cd5b9418b6d02d84083da2846e423bcfe40cd",
                 "33d5b643b10a5496d26235e2d93792bec89a92e0d3654cc07db9f4c964c42b04"),
    "contrast": ("eecda804e967b625a7ef57dc0f00a84af9643607cdbda0fa15db384d8384a0c8",
                 "e2b8773fbf8f665cb6da6c433f7c4a0aa9d8e55e70361b815afce29f4ea9c5ae"),
    "sparse": ("e83c95023c055fe50fefc82d5a502dcdea84a7305c6885ac29c187cbd4b39746",
               "d8c7c3288bdb0050e4542e569887674e3da55fd2465702736d26b848163b1f1c")}


@pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning")
def test_independent_scheme_is_bit_identical_to_the_code_before_ruling_118():
    """aligned=False (the default, spelled out or not) draws the calibration days with `seed` and the deployment days with
    `seed + 1`, as before: the same replicates, the same keys, the same values. The contrast bootstrap has no such option."""
    from es.bootstrap import _cal_dep_bootstrap
    for extra in ({}, {"aligned": False}):
        assert _pin_digests(_pin_results(**extra)) == PIN_BEFORE_RULING_118, extra
    alt, base = _pin_frames()                       # and the positional call that es.sealed_run._contrast makes
    t_hat, t_star, n_nodes = _cal_dep_bootstrap([alt, base], lambda e: e[0] - e[1], 0.05, 40, 7, 20261022)
    digest = hashlib.sha256(np.ascontiguousarray(t_star, dtype="<f8").tobytes()).hexdigest()
    assert digest == PIN_BEFORE_RULING_118["contrast"][0] and n_nodes == 4


def test_aligned_counts_of_two_windows_with_the_same_calendar_are_one_vector():
    """Windows that carry the same (month, day) positions get the same count at each position: one draw, applied to both."""
    from es.bootstrap import _aligned_counts, stationary_counts
    cal = [_Q3_CAL[0] + dt.timedelta(days=k) for k in range(92)]
    dep = [_Q3_DEP[0] + dt.timedelta(days=k) for k in range(92)]
    wc, wd = _aligned_counts(cal, dep, 7, 50, 3)
    assert wc.shape == wd.shape == (50, 92) and np.array_equal(wc, wd)
    assert np.array_equal(wc, stationary_counts(92, 7, 50, 3))      # the single draw is made with `seed`, not `seed + 1`
    assert (wc.sum(axis=1) == 92).all()


def test_aligned_draw_handles_a_leap_day_and_missing_days_through_the_calendar_index():
    """A leap day that one window alone has, and days that a window lacks, need no special case: the index is the sorted
    union of the windows' (month, day), one draw is made over it, and each day takes the count of its position. The
    replicates equal the policy pipeline on the explicitly repeated days."""
    from es.bootstrap import _aligned_counts, block_level_bootstrap, stationary_counts
    cal = (dt.date(2024, 2, 10), dt.date(2024, 3, 20))              # holds 29 February 2024: 20 + 20 days
    dep = (dt.date(2025, 2, 10), dt.date(2025, 3, 21))              # 19 + 21 days: no 29 February, one more day in March
    missing = (dt.date(2024, 2, 20), dt.date(2025, 3, 5))           # one day lost in each window, to every node
    scores = _calendar_scores(cal, dep, drop=missing, hours=6)
    cal_days, dep_days = _days_of([scores], "calib"), _days_of([scores], "deploy")
    assert (len(cal_days), len(dep_days)) == (39, 39)

    def month_day(days):
        return [(d.month, d.day) for d in days]
    index = sorted(set(month_day(cal_days)) | set(month_day(dep_days)))
    assert len(index) == 41 and (2, 29) in month_day(cal_days) and (2, 29) not in month_day(dep_days)
    n_rep, mean_block, seed, alpha = 8, 7, 11, 0.1
    draw = stationary_counts(len(index), mean_block, n_rep, seed)       # ONE draw over the index, with `seed`
    at = {md: i for i, md in enumerate(index)}
    wc, wd = _aligned_counts(cal_days, dep_days, mean_block, n_rep, seed)
    assert np.array_equal(wc, draw[:, [at[md] for md in month_day(cal_days)]])
    assert np.array_equal(wd, draw[:, [at[md] for md in month_day(dep_days)]])
    shared = sorted(set(month_day(cal_days)) & set(month_day(dep_days)))
    assert len(shared) == 37                                            # 2 days only the calibration has, 2 only deployment
    for md in shared:                                                   # a position shared by both windows has one count
        assert np.array_equal(wc[:, month_day(cal_days).index(md)], wd[:, month_day(dep_days).index(md)])
    assert not np.array_equal(wc.sum(axis=1), np.full(n_rep, 39))       # the totals vary: not a resample of each window
    assert (wc.sum(axis=1) > 0).all() and (wd.sum(axis=1) > 0).all()   # (and no node vanishes from these replicates)
    res = block_level_bootstrap(scores, alpha=alpha, n_rep=n_rep, mean_block=mean_block, seed=seed, aligned=True)
    assert res["n_nodes"] == 3 and np.isfinite(res["t_star"]).all() and np.isfinite(res["t_hat"])
    for b in [None] + list(range(n_rep)):                               # None: the real days, all weights 1, i.e. t_hat
        weights = (np.ones(39), np.ones(39)) if b is None else (wc[b], wd[b])
        act = _policy_view(scores, cal_days, dep_days, *weights, alpha)
        got = res["t_hat"] if b is None else res["t_star"][b]
        assert np.isclose(got, float(np.median(act["log_error"])), rtol=1e-9, atol=1e-12), f"replicate {b}"


def test_aligned_bootstrap_with_the_same_seed_is_reproducible_and_the_seed_matters():
    from es.bootstrap import block_level_bootstrap
    scores = _score_frame(71, 0.5, (("M", "A"), ("M", "B"), ("N", "A")), 24, 24, hours=4)
    first = block_level_bootstrap(scores, n_rep=60, seed=5, aligned=True)
    again = block_level_bootstrap(scores, n_rep=60, seed=5, aligned=True)
    other = block_level_bootstrap(scores, n_rep=60, seed=6, aligned=True)
    independent = block_level_bootstrap(scores, n_rep=60, seed=5)
    assert _pin_digest(first) == _pin_digest(again)
    assert not np.array_equal(first["t_star"], other["t_star"], equal_nan=True)
    assert not np.array_equal(first["t_star"], independent["t_star"], equal_nan=True)     # a different draw
    assert first["t_hat"] == independent["t_hat"]                                      # on the real days either way


def test_the_aligned_option_is_on_the_level_bootstrap_only_and_labels_only_its_own_result():
    """The key "scheme" is added to the aligned result alone: the default result keeps the keys it always had. t_hat is
    on the real days (all weights 1) in both."""
    import inspect
    from es import bootstrap as b
    assert inspect.signature(b._cal_dep_bootstrap).parameters["aligned"].default is False
    assert inspect.signature(b.block_level_bootstrap).parameters["aligned"].default is False
    assert "aligned" not in inspect.signature(b.block_contrast_bootstrap).parameters
    scores = _score_frame(72, 0.3, (("M", "A"), ("M", "B")), 20, 20, hours=3)
    for measure in ("median", "failures"):
        default = b.block_level_bootstrap(scores, n_rep=20, measure=measure)
        spelled = b.block_level_bootstrap(scores, n_rep=20, measure=measure, aligned=False)
        aligned = b.block_level_bootstrap(scores, n_rep=20, measure=measure, aligned=True)
        assert set(default) == set(spelled) == {"t_hat", "t_star", "ci", "ci_basic", "n_nodes"}
        assert set(aligned) == set(default) | {"scheme"} and aligned["scheme"] == "aligned"
        assert aligned["t_hat"] == default["t_hat"] and aligned["n_nodes"] == default["n_nodes"]
        assert aligned["t_star"].shape == (20,)
