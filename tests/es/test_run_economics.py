"""Tests of es/run_economics.py (Task 12): the development economic analysis.

Three layers:
- the brief's own tests (the slow real-data one, and the two estimand_cis tests of the Ruling 27 amendment);
- unit tests of what the amendments and the controller's appended rules add (rotation.csv flattening, the pooled and
  ex-CAISO estimands, the full cyclic group of Ruling 69, the paired E4 contrasts of Ruling 81);
- `economics_for` / `p1_rows_for` end to end on small synthetic score frames through the injectable ``score_fn``
  (no cache, no real data): the plain case, frames that disagree with P1's rows (Ruling 82, M1 and M2), and a
  deployment window in UTC time (M3); and the output wiring of ``run_all`` with a stubbed ``economics_for``
  (incremental atomic writes and run_meta.json, M4).
"""
import datetime as dt
import hashlib
import json
import re
import time
import warnings

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------------------------------------------
# The brief's slow test (Step 1, extended by Ruling 37), on the real development panel
# ---------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_panel():
    """The development panel, built once for the slow tests (build_panel(load_dev_raw()) is sealed-window guarded)."""
    from es.data import build_panel, load_dev_raw
    t0 = time.time()
    panel = build_panel(load_dev_raw())
    print(f"\n[timing] build_panel {time.time() - t0:.1f} s, {len(panel)} rows", flush=True)
    return panel


@pytest.fixture(scope="module")
def aq3_real(real_panel):
    """AQ3 on the real development panel, computed once for the slow tests: (rows, kept, dropped_notes).

    ``fmm=None, nyi=None`` as in the brief's test: CAISO and NYISO settle on the panel's 5-minute-mean series and are
    flagged approximate. Scores come from the cache (the default score_fn). ``dropped_notes`` are the warnings that
    say rows outside P1's were dropped before the policies ranked (Ruling 82): AQ3 has none."""
    from es.analyses import ANALOGS
    from es.run_economics import economics_for
    t0 = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rows, kept = economics_for(real_panel, ANALOGS["AQ3"], fmm=None, nyi=None, eta=0.01, keep_rows=True)
    print(f"\n[timing] economics_for(AQ3, keep_rows=True) {time.time() - t0:.1f} s, {len(rows)} summary rows", flush=True)
    return rows, kept, [str(w.message) for w in caught if "P1 does not have" in str(w.message)]


@pytest.mark.slow
def test_economics_rows_for_one_analog(aq3_real):
    rows, _, _ = aq3_real
    kinds = {(r["policy"], r["position"], r["fees"]) for r in rows}
    assert ("P1", "premium", "calibrated") in kinds and ("P4@r*", "premium", "calibrated") in kinds
    p1 = next(r for r in rows if r["policy"] == "P1" and r["position"] == "premium"
              and r["fees"] == "calibrated" and r["comparator"] == "none")
    assert -1.0 <= p1["SELC"] <= 1.0 and "SELC_decile" in p1 and p1["n_days"] == 92
    assert 0 < p1["rotation_p"] <= 1                                    # Ruling 37
    print(f"\n[result] AQ3 P1 premium calibrated: SELC {p1['SELC']:.4f}, SELC_decile {p1['SELC_decile']:.4f}, "
          f"NET {p1['NET']:.1f}, SNET {p1['SNET']:.4f}, ESred {p1['ESred']:.1f}, rotation_p {p1['rotation_p']:.4f}, "
          f"alert_rate {p1['alert_rate']:.4f}, settle_approx_share {p1['settle_approx_share']:.3f}", flush=True)


@pytest.mark.slow
def test_estimand_cis_on_the_real_rows_of_one_analog(aq3_real, real_panel):
    """Real rows through estimand_cis at the default 2000 resamples: its time per window (reported), its agreement
    with the summary row of economics_for, the row sets of P1 / P4 / P4@r* and of the E4 comparators, exact count
    equality for the top_count comparators (a DateWindow analysis), p1_rows_for against kept["P1"], and the full
    cyclic group (Ruling 69)."""
    from es.analyses import ANALOGS
    from es.bootstrap import RowCube, rotation_test
    from es.run_economics import estimand_cis, p1_rows_for
    rows, kept, dropped_notes = aq3_real
    assert dropped_notes == []                                  # AQ3 has no row outside P1's (Ruling 82)
    t0 = time.time()
    cis = estimand_cis([kept])
    seconds = time.time() - t0
    print(f"\n[timing] estimand_cis(1 window, n_rep=2000) {seconds:.1f} s", flush=True)
    est = {r["stat"]: r for r in cis}
    p1 = next(r for r in rows if r["policy"] == "P1" and r["position"] == "premium"
              and r["fees"] == "calibrated" and r["comparator"] == "none")
    assert est["SELC"]["estimate"] == pytest.approx(p1["SELC"])
    assert est["E2:SNET"]["estimate"] == pytest.approx(p1["SNET"])
    assert est["E3:ESred"]["estimate"] == pytest.approx(p1["ESred"])
    assert est["E4:SELC_decile"]["estimate"] == pytest.approx(p1["SELC_decile"])
    assert all(r["lo"] <= r["hi"] and np.isfinite([r["estimate"], r["lo"], r["hi"]]).all() for r in cis)
    assert {r["n_windows"] for r in cis} == {1}
    for stat in ("SELC", "NET", "E2:SNET", "E3:ESred", "E4:SELC_decile", "E4:SELC|calendar@count", "E4:SELC|VOL@count",
                 "E4:dSELC|calendar@count", "E4:dSELC|VOL@count"):
        r = est[stat]
        print(f"[result] AQ3 {stat}: {r['estimate']:.4f}  [{r['lo']:.4f}, {r['hi']:.4f}]", flush=True)
    for c in ("calendar@count", "VOL@count"):                   # Ruling 81: the contrast is SELC(P1) - SELC(comparator)
        assert est[f"E4:dSELC|{c}"]["estimate"] == pytest.approx(est["SELC"]["estimate"] - est[f"E4:SELC|{c}"]["estimate"])
    key = ["market", "node", "timestamp_utc"]
    p1_keys = set(map(tuple, kept["P1"][key].to_numpy()))
    for other in ("P4", "P4@r*", "calendar@count", "VOL@count"):        # none may hold a row P1 lacks, or lack one P1 has
        assert set(map(tuple, kept[other][key].to_numpy())) == p1_keys, other
    n1 = kept["P1"].groupby(["market", "node"])["alert"].sum()
    for name in ("calendar@count", "VOL@count"):                        # top_count: exactly P1's count in every node
        assert (kept[name].groupby(["market", "node"])["alert"].sum() == n1).all(), name
    gap = (kept["P4@r*"].groupby(["market", "node"])["alert"].sum() - n1).abs()
    print(f"[result] AQ3 P4@r* count minus P1 count: max |gap| {int(gap.max())} alerts (nodes {len(gap)})", flush=True)
    pd.testing.assert_frame_equal(p1_rows_for(real_panel, ANALOGS["AQ3"], None, None), kept["P1"])
    res = rotation_test([RowCube.from_rows(kept["P1"])])
    assert res["n_null"] == 92 - 1
    assert 0 < res["p"] <= 1 and res["p"] == p1["rotation_p"]
    print(f"[result] AQ3 rotation: t_obs {res['t_obs']:.3f}, p {res['p']:.4f}, n_null {res['n_null']}, "
          f"selc_obs {res['selc_obs']:.4f}, retained_share {res['retained_share']}", flush=True)


@pytest.mark.slow
def test_deployment_windows_of_every_analysis_hold_no_missing_prices(real_panel):
    """summarize refuses a NaN P&L, so main() would stop at an analysis whose deployment rows lack a DA or RT price.
    Checked on the panel itself, with nothing fitted, for every analysis main() runs."""
    from es.analyses import ANALOGS, LEGACY
    from es.blocks import features
    complete = real_panel.dropna(subset=features("G1_24"))              # the rows score_analysis scores
    for a in list(ANALOGS.values()) + [LEGACY[k] for k in ("2024-2025", "24H2-25H2", "25H1-26H1")]:
        dep = complete[a.deploy.mask(complete).to_numpy()]
        assert len(dep) > 0, a.id
        missing = dep[["da_lmp", "rt_lmp"]].isna().sum()
        assert missing.sum() == 0, f"{a.id}: {missing.to_dict()} deployment rows without a price"


@pytest.mark.slow
def test_calendar_frame_rows_that_p1_lacks_are_dropped_before_top_count_ranks_on_aq4(real_panel):
    """The review's M1 case on real scores (cached AQ4, nothing fitted): around the 2 Nov 2025 fall-back the calendar
    frame holds deployment rows that the combined frame lacks (its 48/72/168 h lags are missing there). The restriction
    of Ruling 82 drops them, counts them, and top_count then ranks on P1's rows only: same rows, same counts."""
    from es.analyses import ANALOGS
    from es.policies import p1_fixed, top_count
    from es.run_economics import _counts, _scores_for, _within
    from es.scores import cached_scores
    primary, g6 = _scores_for(real_panel, ANALOGS["AQ4"], cached_scores)
    p1 = p1_fixed(primary, 0.05)
    key = ["market", "node", "timestamp_utc"]
    p1_keys = set(map(tuple, p1[key].to_numpy()))
    g6_deploy = g6[g6["split"] == "deploy"]
    extra = [k for k in map(tuple, g6_deploy[key].to_numpy()) if k not in p1_keys]
    assert extra, "AQ4 is expected to have calendar-frame deployment rows that P1 lacks"
    restricted, n_dropped = _within(g6, p1[key])
    assert n_dropped == len(extra)
    ranked = top_count(restricted, _counts(p1))
    assert set(map(tuple, ranked[key].to_numpy())) == p1_keys
    assert (ranked.groupby(["market", "node"])["alert"].sum() == p1.groupby(["market", "node"])["alert"].sum()).all()
    print(f"\n[result] AQ4: {n_dropped} calendar-frame deployment rows are not P1's rows; "
          f"after the restriction calendar@count has P1's rows and counts exactly", flush=True)


# ---------------------------------------------------------------------------------------------------------------
# The brief's estimand_cis tests (Ruling 27)
# ---------------------------------------------------------------------------------------------------------------

def _kept(seed, start=dt.date(2025, 7, 1), n_days=30):
    rng = np.random.default_rng(seed)
    rows = []
    for node in ("A", "B"):
        for d in range(n_days):
            day = start + dt.timedelta(days=d)
            for h in range(24):
                rows.append({"market": "M", "node": node, "settle_date": day, "settle_hour": h,
                             "timestamp_utc": pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=h),
                             "wclass": int(day.weekday() >= 5), "month": day.month, "cal_decile": h % 10,
                             "pnl_open": rng.normal(0.5, 3.0)})
    base = pd.DataFrame(rows)
    base["dart_settle"] = base["pnl_open"]
    return {name: base.assign(alert=rng.random(len(base)) < 0.08)
            for name in ("P1", "P4@r*", "P4", "calendar@count", "VOL@count")}


def test_estimand_cis_match_direct_measures_and_pool():
    from es import measures
    from es.bootstrap import RowCube
    from es.run_economics import estimand_cis
    a, b = _kept(1), _kept(2, start=dt.date(2025, 10, 1))
    est = {r["stat"]: r for r in estimand_cis([a], n_rep=30)}
    s = {k: measures.summarize(v) for k, v in a.items()}
    assert est["SELC"]["estimate"] == pytest.approx(s["P1"]["SELC"])
    assert est["E2:SNET"]["estimate"] == pytest.approx(s["P1"]["SNET"])
    assert est["E3:ESred"]["estimate"] == pytest.approx(s["P1"]["ESred"])
    assert est["E4:SELC_decile"]["estimate"] == pytest.approx(s["P1"]["SELC_decile"])
    assert est["E4:SELC|calendar@count"]["estimate"] == pytest.approx(s["calendar@count"]["SELC"])
    d = measures.decomposition(s["P1"], s["P4@r*"], s["P4"])
    for k in measures.E1_KEYS:
        for part in ("timing", "level", "total"):
            assert est[f"E1:{k}:{part}"]["estimate"] == pytest.approx(d[k][part])
    assert all(r["lo"] <= r["hi"] for r in est.values())
    assert est["E3:ESred"]["approximate"] and not est["E2:SNET"]["approximate"]
    pooled = {r["stat"]: r for r in estimand_cis([a, b], n_rep=30)}
    (ea, la), (eb, lb) = RowCube.from_rows(a["P1"]).selc_parts(), RowCube.from_rows(b["P1"]).selc_parts()
    assert pooled["SELC"]["estimate"] == pytest.approx((ea + eb) / (la + lb))
    sb = measures.summarize(b["P1"])
    assert pooled["NET"]["estimate"] == pytest.approx(s["P1"]["NET"] + sb["NET"])
    assert pooled["E3:ESred"]["estimate"] == pytest.approx(
        (s["P1"]["n_days"] * s["P1"]["ESred"] + sb["n_days"] * sb["ESred"]) / (s["P1"]["n_days"] + sb["n_days"]))
    assert {r["n_windows"] for r in pooled.values()} == {2}


def test_estimand_cis_refuse_misaligned_policies():
    from es.run_economics import estimand_cis
    a = _kept(3)
    a["P4"] = a["P4"].iloc[5:]
    with pytest.raises(ValueError, match="P4"):
        estimand_cis([a], n_rep=5)


# ---------------------------------------------------------------------------------------------------------------
# estimand_cis: more of the contract
# ---------------------------------------------------------------------------------------------------------------

def test_estimand_cis_cover_every_estimand_once_with_flags():
    from es.measures import E1_KEYS
    from es.run_economics import estimand_cis
    got = estimand_cis([_kept(4)], n_rep=20)
    stats = [r["stat"] for r in got]
    want = ([f"E1:{k}:{part}" for k in E1_KEYS for part in ("timing", "level", "total")]
            + ["E2:SNET", "E3:ESred", "NET", "SELC", "E4:SELC_decile", "E4:SELC|calendar@count", "E4:SELC|VOL@count",
               "E4:dSELC|calendar@count", "E4:dSELC|VOL@count"]             # the levels stay; the contrasts are Ruling 81's
            + ["E4:ESred|calendar@count", "E4:ESred|VOL@count",
               "E4:dESred|calendar@count", "E4:dESred|VOL@count"])          # ESred against matched activity: Ruling 118
    assert sorted(stats) == sorted(want) and len(set(stats)) == len(stats) == 31
    assert {r["n_windows"] for r in got} == {1}
    flag = {r["stat"]: r["approximate"] for r in got}
    # exact: the sums and ratios of additive pieces; approximate: anything that holds the day-bootstrapped ESred
    assert not any(flag[s] for s in ("NET", "SELC", "E2:SNET", "E4:SELC_decile", "E4:SELC|calendar@count",
                                     "E4:SELC|VOL@count", "E4:dSELC|calendar@count", "E4:dSELC|VOL@count",
                                     "E1:NET:timing", "E1:NET:level", "E1:NET:total",
                                     "E1:J_0.0:timing", "E1:J_0.0:level", "E1:J_0.0:total"))
    assert all(flag[s] for s in ("E3:ESred", "E1:ESred:timing", "E1:J_0.25:total", "E1:J_0.5:level", "E1:J_1.0:timing",
                                 "E4:ESred|calendar@count", "E4:ESred|VOL@count",
                                 "E4:dESred|calendar@count", "E4:dESred|VOL@count"))


def test_estimand_cis_e1_total_is_timing_plus_level_and_j0_is_net():
    from es.measures import E1_KEYS
    from es.run_economics import estimand_cis
    est = {r["stat"]: r["estimate"] for r in estimand_cis([_kept(5)], n_rep=10)}
    for k in E1_KEYS:
        assert est[f"E1:{k}:total"] == pytest.approx(est[f"E1:{k}:timing"] + est[f"E1:{k}:level"])
    for part in ("timing", "level", "total"):
        assert est[f"E1:J_0.0:{part}"] == pytest.approx(est[f"E1:NET:{part}"])


def test_estimand_cis_are_reproducible_and_the_seed_matters():
    from es.run_economics import estimand_cis
    a = _kept(6)
    first, again = estimand_cis([a], n_rep=40), estimand_cis([a], n_rep=40)
    other = estimand_cis([a], n_rep=40, seed=1)
    assert first == again
    assert [r["estimate"] for r in first] == [r["estimate"] for r in other]
    assert [(r["lo"], r["hi"]) for r in first] != [(r["lo"], r["hi"]) for r in other]


def test_e4_contrasts_are_paired_over_the_same_resampled_days():
    """Ruling 81: E4:dSELC|c = SELC(P1) - SELC(c). Its interval is that of the per-replicate differences (P1 and c are
    evaluated on the same resampled days), which is not the difference of the two marginal intervals."""
    from es.bootstrap import RowCube, percentile_ci, stationary_counts
    from es.run_economics import estimand_cis
    a = _kept(21)
    est = {r["stat"]: r for r in estimand_cis([a], n_rep=60)}
    cube = RowCube.from_rows(a["P1"])
    w = stationary_counts(cube.n_days, 7, 60, 20261022)
    for c in ("calendar@count", "VOL@count"):
        d, level = est[f"E4:dSELC|{c}"], est[f"E4:SELC|{c}"]
        cmp_cube = RowCube.from_rows(a[c])
        assert d["estimate"] == pytest.approx(est["SELC"]["estimate"] - level["estimate"])
        lo, hi = percentile_ci(np.array([cube.selc(wb) - cmp_cube.selc(wb) for wb in w]))
        assert (d["lo"], d["hi"]) == pytest.approx((lo, hi))
        marginal = (est["SELC"]["lo"] - level["hi"], est["SELC"]["hi"] - level["lo"])
        assert (d["lo"], d["hi"]) != pytest.approx(marginal)
        assert not d["approximate"] and d["n_windows"] == 1
        assert level["stat"] == f"E4:SELC|{c}"                          # the level is kept next to the contrast


def test_e4_contrast_of_a_comparator_that_is_p1_is_exactly_zero_with_a_degenerate_interval():
    from es.run_economics import estimand_cis
    a = _kept(22)
    a["calendar@count"] = a["P1"].copy()
    est = {r["stat"]: r for r in estimand_cis([a], n_rep=40)}
    d = est["E4:dSELC|calendar@count"]
    assert d["estimate"] == pytest.approx(0, abs=1e-12)
    assert d["lo"] == pytest.approx(0, abs=1e-12) and d["hi"] == pytest.approx(0, abs=1e-12)
    level = est["E4:SELC|calendar@count"]
    assert est["SELC"]["hi"] - est["SELC"]["lo"] > 0.01 and level["hi"] - level["lo"] > 0.01   # the marginal intervals are not


def test_e4_contrast_pools_as_the_difference_of_the_pooled_ratios():
    from es.bootstrap import RowCube
    from es.run_economics import estimand_cis
    a, b = _kept(23), _kept(24, start=dt.date(2025, 10, 1))
    pooled = {r["stat"]: r for r in estimand_cis([a, b], n_rep=20)}
    single = {r["stat"]: r for r in estimand_cis([a], n_rep=20)}

    def parts(kept, name):
        return RowCube.from_rows(kept[name]).selc_parts()
    for c in ("calendar@count", "VOL@count"):
        (xa, la), (xb, lb) = parts(a, "P1"), parts(b, "P1")
        (ya, ma), (yb, mb) = parts(a, c), parts(b, c)
        want = (xa + xb) / (la + lb) - (ya + yb) / (ma + mb)
        assert pooled[f"E4:dSELC|{c}"]["estimate"] == pytest.approx(want)
        mean_of_quarters = ((xa / la - ya / ma) + (xb / lb - yb / mb)) / 2
        assert pooled[f"E4:dSELC|{c}"]["estimate"] != pytest.approx(mean_of_quarters)
        assert pooled[f"E4:dSELC|{c}"]["estimate"] == pytest.approx(
            pooled["SELC"]["estimate"] - pooled[f"E4:SELC|{c}"]["estimate"])
        assert single[f"E4:dSELC|{c}"]["estimate"] != pytest.approx(pooled[f"E4:dSELC|{c}"]["estimate"])
        assert pooled[f"E4:dSELC|{c}"]["n_windows"] == 2


# ---------------------------------------------------------------------------------------------------------------
# Ruling 118 (pre-registration review M6): ESred against the matched-activity comparators, paired like dSELC
# ---------------------------------------------------------------------------------------------------------------

E4_COMPARATORS = ("calendar@count", "VOL@count")
E4_ESRED = tuple(f"E4:{kind}|{c}" for c in E4_COMPARATORS for kind in ("ESred", "dESred"))


def test_e4_desred_is_esred_of_p1_minus_esred_of_the_comparator_paired_over_the_same_resampled_days():
    """E4:ESred|c is the comparator's own ESred and E4:dESred|c = ESred(P1) - ESred(c). Like dSELC (Ruling 81) the contrast
    is evaluated on the same resampled days in each replicate, so its interval is that of the paired differences, which is
    not the difference of the two marginal intervals."""
    from es import measures
    from es.bootstrap import RowCube, percentile_ci, stationary_counts
    from es.run_economics import estimand_cis
    a = _kept(41)
    est = {r["stat"]: r for r in estimand_cis([a], n_rep=60)}
    s = {k: measures.summarize(v) for k, v in a.items()}
    cube = RowCube.from_rows(a["P1"])
    w = stationary_counts(cube.n_days, 7, 60, 20261022)
    for c in E4_COMPARATORS:
        level, d = est[f"E4:ESred|{c}"], est[f"E4:dESred|{c}"]
        assert level["estimate"] == pytest.approx(s[c]["ESred"])                      # es.measures, not the cube
        assert d["estimate"] == pytest.approx(s["P1"]["ESred"] - s[c]["ESred"])
        assert d["estimate"] == pytest.approx(est["E3:ESred"]["estimate"] - level["estimate"])
        cmp_cube = RowCube.from_rows(a[c])
        lo, hi = percentile_ci(np.array([cube.esred(wb) - cmp_cube.esred(wb) for wb in w]))
        assert (d["lo"], d["hi"]) == pytest.approx((lo, hi))
        marginal = (est["E3:ESred"]["lo"] - level["hi"], est["E3:ESred"]["hi"] - level["lo"])
        assert (d["lo"], d["hi"]) != pytest.approx(marginal)
        assert (level["stat"], d["stat"]) == (f"E4:ESred|{c}", f"E4:dESred|{c}") and d["n_windows"] == 1


def test_e4_desred_of_a_comparator_with_p1s_alerts_is_zero_in_every_replicate():
    """A comparator whose alerts are P1's removes exactly what P1 removes on any resample, so the paired contrast is 0 in
    every replicate and its interval is a point, while the marginal ESred intervals are not."""
    from es.bootstrap import stationary_counts
    from es.run_economics import _parts, _stats, _window, estimand_cis
    a = _kept(42)
    a["calendar@count"] = a["P1"].copy()
    a["VOL@count"] = a["P1"].sample(frac=1.0, random_state=3)             # P1's rows and alerts, in another order
    win = _window(a)
    for b, w in enumerate(stationary_counts(win[0].n_days, 7, 40, 20261022)):
        stats = _stats(_parts(win, w))
        assert stats["E4:dESred|calendar@count"] == 0.0, b
        assert stats["E4:ESred|calendar@count"] == stats["E3:ESred"], b
        assert stats["E4:dESred|VOL@count"] == pytest.approx(0.0, abs=1e-9), b      # a sum in another row order: to rounding
    est = {r["stat"]: r for r in estimand_cis([a], n_rep=40)}
    d = est["E4:dESred|calendar@count"]
    assert (d["estimate"], d["lo"], d["hi"]) == (0.0, 0.0, 0.0)
    level = est["E4:ESred|calendar@count"]
    assert est["E3:ESred"]["hi"] - est["E3:ESred"]["lo"] > 0 and level["hi"] - level["lo"] > 0


def test_e4_esred_rows_are_in_the_estimands_once_each_flagged_approximate_and_finite():
    from es.run_economics import _approximate, estimand_cis
    got = estimand_cis([_kept(43)], n_rep=20)
    stats = [r["stat"] for r in got]
    assert all(stats.count(s) == 1 for s in E4_ESRED) and stats[-4:] == list(E4_ESRED)      # after the 27 that were there
    new = [r for r in got if r["stat"] in E4_ESRED]
    assert len(new) == 4 and all(r["approximate"] for r in new) and {r["n_windows"] for r in new} == {1}
    assert all(r["lo"] <= r["hi"] and np.isfinite([r["estimate"], r["lo"], r["hi"]]).all() for r in new)
    assert all(_approximate(s) for s in E4_ESRED)                  # "ESred" is in the name of the level and of the contrast
    assert not _approximate("E4:dSELC|calendar@count") and not _approximate("E4:SELC|VOL@count")      # and SELC's stay exact


def test_e4_esred_pieces_pool_by_their_day_weights_like_e3():
    """The comparator's day-weighted ESred is an additive piece like P1's: quarters pool by summing it (and D), so the
    pooled level is the day-weighted mean of the quarters' ESred and the pooled contrast the difference of the pooled levels,
    not the mean of the quarters' contrasts (the windows differ in length here, so the two differ)."""
    from es import measures
    from es.run_economics import estimand_cis
    a, b = _kept(44, n_days=30), _kept(45, start=dt.date(2025, 10, 1), n_days=20)
    pooled = {r["stat"]: r for r in estimand_cis([a, b], n_rep=20)}
    single = {r["stat"]: r for r in estimand_cis([a], n_rep=20)}
    sa, sb = ({k: measures.summarize(v) for k, v in kept.items()} for kept in (a, b))
    na, nb = sa["P1"]["n_days"], sb["P1"]["n_days"]
    assert (na, nb) == (30, 20)

    def day_weighted(name):
        return (na * sa[name]["ESred"] + nb * sb[name]["ESred"]) / (na + nb)
    for c in E4_COMPARATORS:
        assert pooled[f"E4:ESred|{c}"]["estimate"] == pytest.approx(day_weighted(c))
        assert pooled[f"E4:dESred|{c}"]["estimate"] == pytest.approx(day_weighted("P1") - day_weighted(c))
        assert pooled[f"E4:dESred|{c}"]["estimate"] == pytest.approx(
            pooled["E3:ESred"]["estimate"] - pooled[f"E4:ESred|{c}"]["estimate"])
        mean_of_quarters = ((sa["P1"]["ESred"] - sa[c]["ESred"]) + (sb["P1"]["ESred"] - sb[c]["ESred"])) / 2
        assert pooled[f"E4:dESred|{c}"]["estimate"] != pytest.approx(mean_of_quarters)
        assert single[f"E4:dESred|{c}"]["estimate"] != pytest.approx(pooled[f"E4:dESred|{c}"]["estimate"])
        assert pooled[f"E4:dESred|{c}"]["n_windows"] == pooled[f"E4:ESred|{c}"]["n_windows"] == 2
        assert pooled[f"E4:dESred|{c}"]["approximate"] and pooled[f"E4:ESred|{c}"]["approximate"]


def _ci_digest(rows):
    """sha256 of estimand_cis rows, bit for bit and in order: stat, estimate, lo, hi (float.hex), approximate, n_windows."""
    text = "\n".join(f"{r['stat']}|{float(r['estimate']).hex()}|{float(r['lo']).hex()}|{float(r['hi']).hex()}|"
                     f"{bool(r['approximate'])}|{r['n_windows']}" for r in rows)
    return hashlib.sha256(text.encode()).hexdigest()


def _pin_estimands():
    """estimand_cis on two fixtures: one window, and two windows of different length pooled."""
    from es.run_economics import estimand_cis
    return {"one window": estimand_cis([_kept(31)], n_rep=40),
            "two windows": estimand_cis([_kept(32), _kept(33, start=dt.date(2025, 10, 1), n_days=20)], n_rep=40)}


#: ``_ci_digest`` of the 27 rows ``_pin_estimands()`` gave on es/run_economics.py as it stood before Ruling 118 added
#: the four ESred rows (estimates, intervals, flags and order), computed before the change and then pasted here.
PIN_ESTIMANDS_BEFORE_RULING_118 = {
    "one window": "945d2998557b9ed202cce2c17fc99d4702df1eca1934ef7ee7b0034acea33a46",
    "two windows": "c81b240776347ece4e9ef60f6225ad628688ef4eee2c9abd3eebd03089e94ff6"}


def test_the_existing_estimands_are_unchanged_by_the_esred_contrasts():
    """Pinned before and after: the 27 statistics of Rulings 27, 68 and 81 keep their estimates, intervals, flags and
    order bit for bit; the four new ones follow them."""
    got = _pin_estimands()
    existing = {k: [r for r in rows if r["stat"] not in E4_ESRED] for k, rows in got.items()}
    assert {k: len(v) for k, v in existing.items()} == {"one window": 27, "two windows": 27}
    assert {k: _ci_digest(v) for k, v in existing.items()} == PIN_ESTIMANDS_BEFORE_RULING_118
    for k, rows in got.items():
        assert [r["stat"] for r in rows[:27]] == [r["stat"] for r in existing[k]]


def _extra_rows(frame, n=6):
    """``frame`` plus ``n`` rows that P1 does not have: a second occurrence of an existing local hour half an hour later
    (the shape of a fall-back day)."""
    extra = frame.iloc[:n].assign(timestamp_utc=frame["timestamp_utc"].iloc[:n] + pd.Timedelta(minutes=30))
    return pd.concat([frame, extra], ignore_index=True)


def test_estimand_cis_refuse_rows_that_p1_does_not_have_or_lacks_in_development():
    """Ruling 82: a two-sided check of every frame's (market, node, timestamp_utc) keys against P1's. Before it, six extra
    comparator rows were accepted silently (the day check sees only days)."""
    from es.run_economics import estimand_cis
    for name in ("calendar@count", "VOL@count", "P4", "P4@r*"):
        a = _kept(7)
        a[name] = _extra_rows(a[name])
        with pytest.raises(ValueError, match=rf"{re.escape(name)}: 0 P1 row\(s\) missing, 6 row\(s\) that P1 does not have"):
            estimand_cis([a], n_rep=5)
    for name in ("calendar@count", "VOL@count"):                      # missing rows: the E4 comparators were unchecked too
        a = _kept(7)
        a[name] = a[name].iloc[7:]
        with pytest.raises(ValueError, match=rf"{re.escape(name)}: 7 P1 row\(s\) missing, 0 row\(s\)"):
            estimand_cis([a], n_rep=5)


def test_estimand_cis_only_warn_about_row_mismatches_on_rows_of_the_sealed_window():
    """Never a check that aborts a sealed run after its data are read: on rows dated after 2026-06-30 a key mismatch is a
    warning and the run goes on. The day check of the brief is kept (it raises)."""
    from es.run_economics import estimand_cis
    a = _kept(8, start=dt.date(2026, 7, 5))
    a["calendar@count"] = _extra_rows(a["calendar@count"])
    with pytest.warns(UserWarning, match=r"calendar@count: 0 P1 row\(s\) missing, 6 row\(s\) that P1 does not have"):
        got = estimand_cis([a], n_rep=5)
    assert len(got) == 31 and all(r["lo"] <= r["hi"] for r in got)
    b = _kept(9, start=dt.date(2026, 7, 5))
    cal = b["calendar@count"]
    b["calendar@count"] = cal[cal["settle_date"] != cal["settle_date"].min()]      # a whole day missing
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match=r"calendar@count.*days differ"):
            estimand_cis([b], n_rep=5)
    assert any("P1 row(s) missing" in str(w.message) for w in caught)
    c = _kept(10, start=dt.date(2026, 7, 5))
    c["P4"] = c["P4"].iloc[5:]              # the brief's alignment of alerts is not softened: a missing alert means NaN estimates
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match=r"P4: alerts missing for 5 P1 row"):
            estimand_cis([c], n_rep=5)


def test_estimand_cis_need_a_window():
    from es.run_economics import estimand_cis
    with pytest.raises(ValueError, match="window"):
        estimand_cis([], n_rep=5)


def _with_caiso(kept):
    """The same frames with node A in market CAISO (node B stays in M)."""
    return {name: frame.assign(market=np.where(frame["node"] == "A", "CAISO", "M")) for name, frame in kept.items()}


def test_without_caiso_is_a_row_filter_and_its_estimands_are_those_of_the_remaining_rows():
    from es import measures
    from es.run_economics import _without_caiso, estimand_cis
    kept = _with_caiso(_kept(8))
    ex = _without_caiso(kept)
    assert set(ex) == set(kept)
    for name, frame in ex.items():
        assert set(frame["market"]) == {"M"} and len(frame) == (kept[name]["market"] != "CAISO").sum()
        assert frame.index.equals(kept[name].index[kept[name]["market"] != "CAISO"])        # a filter, nothing else
    assert (kept["P1"]["market"] == "CAISO").any()                                          # the input is not mutated
    est = {r["stat"]: r for r in estimand_cis([ex], n_rep=20)}
    direct = {k: measures.summarize(v) for k, v in ex.items()}
    assert est["NET"]["estimate"] == pytest.approx(direct["P1"]["NET"])
    assert est["SELC"]["estimate"] == pytest.approx(direct["P1"]["SELC"])
    assert est["E4:SELC|VOL@count"]["estimate"] == pytest.approx(direct["VOL@count"]["SELC"])
    assert est["E3:ESred"]["estimate"] == pytest.approx(direct["P1"]["ESred"])
    full = {r["stat"]: r["estimate"] for r in estimand_cis([kept], n_rep=20)}
    assert full["NET"] != pytest.approx(est["NET"]["estimate"])                             # CAISO's rows do count


# ---------------------------------------------------------------------------------------------------------------
# rotation.csv rows (controller's appended rule, Rulings 55, 64, 69)
# ---------------------------------------------------------------------------------------------------------------

def _gappy(seed, n_days):
    """P1 rows in which node B has no row on days 3..8: alerts rotated into the gap are lost (retained share < 1)."""
    p1 = _kept(seed, n_days=n_days)["P1"]
    gap = (p1["node"] == "B") & p1["settle_date"].between(dt.date(2025, 7, 4), dt.date(2025, 7, 9))
    return p1[~gap]


def test_rotation_row_flattens_two_windows_into_scalars():
    from es.bootstrap import RowCube, rotation_test
    from es.run_economics import rotation_row
    whole, gappy = RowCube.from_rows(_kept(1, n_days=12)["P1"]), RowCube.from_rows(_gappy(2, 15))
    res = rotation_test([whole, gappy])
    shares = res["retained_share"]
    assert shares[0] == {"min": 1.0, "mean": 1.0}                    # a complete grid keeps every alert
    assert shares[1]["min"] < shares[1]["mean"] < 1.0                # the gappy window loses some
    row = rotation_row("two windows", [whole, gappy])
    assert list(row) == ["window", "t_obs", "p", "n_null", "selc_obs", "retained_min", "retained_mean"]
    assert row["window"] == "two windows"
    assert all(isinstance(v, (int, float)) and not isinstance(v, (list, dict)) for k, v in row.items() if k != "window")
    assert row["retained_min"] == min(s["min"] for s in shares) == shares[1]["min"]
    assert row["retained_mean"] == pytest.approx((shares[0]["mean"] + shares[1]["mean"]) / 2)
    assert row["retained_mean"] != pytest.approx(shares[1]["mean"])  # the mean over windows, not the worst window's mean
    assert (row["t_obs"], row["p"], row["selc_obs"]) == (res["t_obs"], res["p"], res["selc_obs"])
    assert row["n_null"] == res["n_null"] == 12 * 15 - 1             # Ruling 69: each window's D rotations, less the identity


def test_rotation_row_of_one_window_is_the_full_cyclic_group():
    from es.bootstrap import RowCube
    from es.run_economics import rotation_row
    row = rotation_row("AQ3", [RowCube.from_rows(_kept(9, n_days=30)["P1"])])
    assert row["n_null"] == 30 - 1 and 1.0 / 30 <= row["p"] <= 1.0
    assert row["retained_min"] == row["retained_mean"] == 1.0


def test_rotation_row_n_null_for_92_day_windows_is_91_alone_and_8463_pooled():
    from es.bootstrap import RowCube
    from es.run_economics import rotation_row
    a, b = (RowCube.from_rows(_kept(s, n_days=92)["P1"]) for s in (10, 11))
    assert rotation_row("AQ3", [a])["n_null"] == 91
    assert rotation_row("AQ3+AQ4", [a, b])["n_null"] == 92 * 92 - 1 == 8463


def test_run_economics_never_passes_min_shift():
    """Ruling 69: the full cyclic group is rotation_test's default; no caller restricts it."""
    import inspect

    import es.run_economics as module
    assert "min_shift" not in inspect.getsource(module)


def test_public_signatures_are_those_the_sealed_entry_point_calls():
    import inspect

    from es.run_economics import KEPT, economics_for, estimand_cis, p1_rows_for, run_all
    from es.scores import cached_scores
    assert list(inspect.signature(run_all).parameters) == ["panel", "fmm", "nyi", "eta", "analyses", "econ_dir", "n_rep",
                                                           "eta_source"]
    sig = inspect.signature(economics_for)
    assert list(sig.parameters) == ["panel", "analysis", "fmm", "nyi", "eta", "score_fn", "keep_rows"]
    assert sig.parameters["score_fn"].default is cached_scores and sig.parameters["keep_rows"].default is False
    sig = inspect.signature(p1_rows_for)
    assert list(sig.parameters) == ["panel", "analysis", "fmm", "nyi", "score_fn"]
    assert sig.parameters["score_fn"].default is cached_scores
    sig = inspect.signature(estimand_cis)
    assert list(sig.parameters) == ["windows", "n_rep", "mean_block", "seed"]
    assert [p.default for p in sig.parameters.values()][1:] == [2000, 7, 20261022]
    assert KEPT == ("P1", "P4@r*", "P4", "calendar@count", "VOL@count")


# ---------------------------------------------------------------------------------------------------------------
# economics_for and p1_rows_for on synthetic scores (score_fn injected: no cache, no real data)
# ---------------------------------------------------------------------------------------------------------------

N_TRAIN, N_CALIB, N_DEPLOY = 45, 20, 28
NODES = (("NYISO", "A"), ("ISONE", "B"), ("CAISO", "C"))


def _syn_analysis(deploy=None):
    from es.analyses import Analysis
    from es.windows import DateWindow as D
    return Analysis("SYN", D("2025-05-01", "2025-06-14"), D("2025-06-15", "2025-07-04"),
                    deploy or D("2025-07-05", "2025-08-01"), D("2025-05-01", "2025-06-14"), "rolling", "analog")


def _syn_scores(seed=0):
    """Combined-model and calendar-model score frames of three nodes (NYISO A, ISONE B, CAISO C), shaped like the
    output of es.scores: 45 training, 20 calibration and 28 deployment days in May-August 2025 (no clock change).
    DART averages +4 with a rare spike of -14 in hours 16-19; the combined score sees the spikes, the calendar score
    only the hour and the weekday (so it has many ties)."""
    from es.config import SETTLE_TZ
    rng = np.random.default_rng(seed)
    records = []
    for market, node in NODES:
        for i in range(N_TRAIN + N_CALIB + N_DEPLOY):
            day = dt.date(2025, 5, 1) + dt.timedelta(days=i)
            split = "train" if i < N_TRAIN else "calib" if i < N_TRAIN + N_CALIB else "deploy"
            spike_day = rng.random() < 0.1
            for h in range(24):
                hot = spike_day and 16 <= h <= 19
                dart = rng.normal(4.0, 2.0) - (14.0 if hot else 0.0)
                da = 40.0 + rng.normal(0.0, 3.0)
                ts = (pd.Timestamp(day) + pd.Timedelta(hours=h)).tz_localize(SETTLE_TZ[market]).tz_convert("UTC")
                records.append({"market": market, "node": node, "node_type": "zone", "timestamp_utc": ts,
                                "settle_date": day, "settle_hour": h, "weekday": day.weekday(),
                                "wclass": int(day.weekday() >= 5), "da_lmp": da, "rt_lmp": da - dart,
                                "dart_spread": dart, "label": float(hot),
                                "abs_mean_168h": abs(rng.normal(3.0, 1.0)), "score": 2.0 * hot + rng.normal(0, 1),
                                "split": split})
    primary = pd.DataFrame(records)
    g6 = primary.assign(score=0.5 * primary["settle_hour"].between(16, 19) + 0.01 * primary["weekday"])
    return primary, g6


def _flipped_fmm(primary):
    """Settlement series for CAISO node C that reverses the sign of its DART: dart_settle = da - fmm = -(da - rt)."""
    c = primary[primary["market"] == "CAISO"]
    return pd.DataFrame({"node": c["node"], "timestamp_utc": c["timestamp_utc"], "fmm_lmp": 2 * c["da_lmp"] - c["rt_lmp"]})


def _stub(primary, g6):
    """A score_fn that serves the two synthetic frames and records how it is called."""
    calls = []

    def score_fn(panel, analysis, block, estimator, seed):
        calls.append((block, estimator, seed))
        return {"G1_24": primary, "G6": g6}[block].copy()
    score_fn.calls = calls
    return score_fn


@pytest.fixture(scope="module")
def syn():
    """economics_for on the synthetic nodes, run once for the module (with keep_rows=True and with the default)."""
    from es.run_economics import economics_for
    primary, g6 = _syn_scores()
    fmm = _flipped_fmm(primary)
    score_fn = _stub(primary, g6)
    t0 = time.time()
    rows, kept = economics_for(None, _syn_analysis(), fmm, None, 0.01, score_fn=score_fn, keep_rows=True)
    seconds = time.time() - t0
    plain = economics_for(None, _syn_analysis(), fmm, None, 0.01, score_fn=_stub(primary, g6))
    return {"rows": rows, "kept": kept, "plain": plain, "primary": primary, "g6": g6, "fmm": fmm,
            "calls": score_fn.calls, "seconds": seconds}


def test_economics_for_covers_every_policy_comparator_position_and_fee(syn):
    rows = syn["rows"]
    policies = {"P1", "P4", "P5", "P6", "P4@r*"}
    comparators = ({"calendar@count", "VOL_P1", "VOL_P4", "VOL_P6", "VOL@count"}
                   | {f"curve_P{k}_a{a}" for k in (1, 6) for a in (0.01, 0.025, 0.05, 0.1)})
    fees = {"calibrated", "0.0", "0.1", "0.25", "0.5", "1.0", "stress"}
    main = [r for r in rows if r["position"] in ("premium", "inc", "dec")]
    assert {r["policy"] for r in main} == policies | {"comparator"}
    assert all((r["comparator"] == "none") == (r["policy"] in policies) for r in main)
    grid = [(r["policy"] if r["policy"] != "comparator" else r["comparator"], r["position"], r["fees"]) for r in main]
    assert len(grid) == len(set(grid)) == len(policies | comparators) * 3 * len(fees)
    assert set(grid) == {(n, p, f) for n in policies | comparators for p in ("premium", "inc", "dec") for f in fees}
    assert all(r["analysis"] == "SYN" for r in rows)


def test_economics_for_splits_the_premium_rows_by_spike_exposure(syn):
    rows = syn["rows"]
    split = [r for r in rows if r["position"].startswith("premium[")]
    assert {r["position"] for r in split} <= {"premium[short_spikes]", "premium[long_spikes]"}
    assert {r["policy"] for r in split} == {"P1", "P4", "P5", "P6", "P4@r*"}
    assert all(r["fees"] == "calibrated" and r["comparator"] == "none" for r in split)
    assert {r["position"] for r in split if r["policy"] == "P1"} == {"premium[short_spikes]", "premium[long_spikes]"}
    whole = next(r for r in rows if (r["policy"], r["position"], r["fees"]) == ("P1", "premium", "calibrated"))
    parts = [r for r in split if r["policy"] == "P1"]
    for key in ("LA", "PG", "NET"):                      # additive over a partition of the rows
        assert sum(r[key] for r in parts) == pytest.approx(whole[key])


def test_economics_for_summary_row_is_summarize_of_the_kept_rows_plus_labels(syn):
    from es import measures
    rows, kept = syn["rows"], syn["kept"]
    p1 = next(r for r in rows if (r["policy"], r["position"], r["fees"], r["comparator"])
              == ("P1", "premium", "calibrated", "none"))
    direct = measures.summarize(kept["P1"])
    for key, value in direct.items():
        assert p1[key] == pytest.approx(value), key
    assert p1["n_days"] == N_DEPLOY and "SELC_decile" in p1 and -1.0 <= p1["SELC"] <= 1.0
    assert p1["settle_approx_share"] == pytest.approx(1 / 3)       # NYISO (no integrated table) only; CAISO has the FMM
    assert p1["SELC"] > 0                                          # the synthetic combined score sees the spikes


def test_economics_for_rotation_p_is_on_every_p1_row_and_nowhere_else(syn):
    from es.bootstrap import RowCube, rotation_test
    rows, kept = syn["rows"], syn["kept"]
    p1 = [r for r in rows if r["policy"] == "P1"]
    assert len(p1) == 3 * 7 + 2                                    # 3 positions x 7 fee settings + 2 spike-exposure splits
    assert all(0 < r["rotation_p"] <= 1 for r in p1)
    assert all("rotation_p" not in r for r in rows if r["policy"] != "P1")        # not VOL_P1, curve_P1_*, P4, ...
    whole = next(r for r in p1 if (r["position"], r["fees"]) == ("premium", "calibrated"))
    assert whole["rotation_p"] == rotation_test([RowCube.from_rows(kept["P1"])])["p"]
    assert whole["rotation_p"] >= 1 / N_DEPLOY                     # the full cyclic group has N_DEPLOY cells


def test_economics_for_keep_rows_returns_the_kept_frames_and_default_returns_the_list(syn):
    from es.positions import CALIBRATED_FEES
    from es.run_economics import KEPT
    kept, rows, plain = syn["kept"], syn["rows"], syn["plain"]
    assert isinstance(plain, list) and plain == rows               # keep_rows does not change the rows
    assert set(kept) == set(KEPT)
    for name, frame in kept.items():                               # premium position, calibrated fees
        assert (frame["fee"] == [CALIBRATED_FEES[m][d] for m, d in zip(frame["market"], frame["d"])]).all(), name
        assert np.allclose(frame["pnl_open"], frame["d"] * frame["dart_settle"] - frame["fee"]), name
        assert {"alert", "cal_decile", "month", "wclass", "settle_hour", "timestamp_utc"} <= set(frame.columns), name


def test_p1_rows_for_is_the_kept_p1_frame(syn):
    from es.run_economics import p1_rows_for
    score_fn = _stub(syn["primary"], syn["g6"])
    out = p1_rows_for(None, _syn_analysis(), syn["fmm"], None, score_fn=score_fn)
    pd.testing.assert_frame_equal(out, syn["kept"]["P1"])
    assert set(score_fn.calls) == {("G1_24", "xgb", 42), ("G6", "xgb", 42)} and len(score_fn.calls) == 2


def test_economics_for_asks_its_score_fn_for_the_two_primary_frames_only(syn):
    """Sealed runs pass score_fn=score_analysis and must never read the cache: the injected function is the only
    source of scores, called as score_fn(panel, analysis, block, estimator, seed)."""
    assert sorted(syn["calls"]) == [("G1_24", "xgb", 42), ("G6", "xgb", 42)]


def test_economics_for_alerts_follow_p1_and_matched_counts(syn):
    kept = syn["kept"]
    n1 = kept["P1"].groupby("node")["alert"].sum()
    got = kept["P4@r*"].groupby("node")["alert"].sum()              # p4_matched: within its own tolerance of the target
    assert (got - n1).abs().max() <= max(3, 0.01 * n1.max())
    for name in ("calendar@count", "VOL@count"):                    # top_count: exactly P1's count in every node (M6)
        assert (kept[name].groupby("node")["alert"].sum() == n1).all(), name
    assert set(kept["P1"]["node"]) == {"A", "B", "C"} and kept["P1"]["settle_date"].nunique() == N_DEPLOY


def test_economics_for_keeps_every_frame_on_p1s_rows(syn):
    """M6: the E4 comparators, like P4 and P4@r*, hold P1's rows and no others."""
    kept = syn["kept"]
    key = ["market", "node", "timestamp_utc"]
    p1_keys = set(map(tuple, kept["P1"][key].to_numpy()))
    for name in ("P4", "P4@r*", "calendar@count", "VOL@count"):
        assert set(map(tuple, kept[name][key].to_numpy())) == p1_keys, name
        assert len(kept[name]) == len(kept["P1"]), name


def test_premium_direction_comes_from_the_settlement_dart_of_the_training_rows(syn):
    primary, fmm, kept = syn["primary"], syn["fmm"], syn["kept"]
    train = primary[primary["split"] == "train"]
    joined = train.merge(fmm, on=["node", "timestamp_utc"], how="left")        # CAISO node C settles on the FMM series
    settle = joined["da_lmp"] - joined["fmm_lmp"].fillna(joined["rt_lmp"])
    keys = ["market", "node", "settle_hour"]
    settle_mean = settle.groupby([joined[k] for k in keys]).mean()
    want = pd.Series(np.where(settle_mean >= 0, 1, -1), index=settle_mean.index)
    panel_mean = train.groupby(keys)["dart_spread"].mean()
    panel_side = pd.Series(np.where(panel_mean >= 0, 1, -1), index=panel_mean.index)
    assert (want != panel_side).any()                              # not vacuous: the settlement series flips node C's sign
    got = kept["P1"].groupby(keys)["d"].agg(["min", "max"])
    assert (got["min"] == got["max"]).all()
    assert (got["min"].reindex(want.index) == want).all()
    p1 = kept["P1"]
    assert set(p1.loc[p1["node"] == "C", "d"]) == {-1} and set(p1.loc[p1["node"] != "C", "d"]) == {1}
    c = kept["P1"][kept["P1"]["node"] == "C"]
    assert np.allclose(c["dart_settle"], -c["dart_spread"])        # P&L is on the settlement series, not the panel's


def test_fee_settings_are_wired_through_exactly(syn):
    """With the alerts fixed, NET = sum over alerts of (fee - d * DART): linear in a uniform fee with slope = alerts."""
    rows, kept = syn["rows"], syn["kept"]
    n_alerts = int(kept["P1"]["alert"].sum())
    net = {r["fees"]: r["NET"] for r in rows if (r["policy"], r["position"]) == ("P1", "premium")}
    for fee in (0.1, 0.25, 0.5, 1.0):
        assert net[str(fee)] == pytest.approx(net["0.0"] + fee * n_alerts)
    assert net["stress"] > net["calibrated"]                        # fees double on stressed days, which hold alerts
    inc = {r["fees"]: r["NET"] for r in rows if (r["policy"], r["position"]) == ("P1", "inc")}
    dec = {r["fees"]: r["NET"] for r in rows if (r["policy"], r["position"]) == ("P1", "dec")}
    for fee in (0.0, 0.1, 0.25, 0.5, 1.0):                          # INC and DEC mirror each other around the fee
        assert inc[str(fee)] + dec[str(fee)] == pytest.approx(2 * fee * n_alerts)


# ---------------------------------------------------------------------------------------------------------------
# Frames that disagree with P1's rows (Ruling 82; review items M1 and M2)
# ---------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def syn_messy():
    """economics_for where the score frames hold rows that P1 does not have, in the two ways the review found:
    - a node P1 cannot place (a copy of C with no calibration rows) whose deployment rows the other policies would keep (M2);
    - twelve deployment rows that only the calendar frame has, with its top score, as on a fall-back day (M1).
    The fee grid is cut to two settings (calibrated, stress) to keep the run short."""
    import es.run_economics as module
    primary, g6 = _syn_scores(seed=3)
    dep = primary[(primary["split"] == "deploy") & primary["node"].isin(["A", "B"]) & primary["settle_hour"].isin([0, 1])]
    gone = dep[dep["settle_date"].isin(sorted(dep["settle_date"].unique())[10:13])]
    assert len(gone) == 12
    g6 = g6.assign(score=g6["score"].where(~g6.index.isin(gone.index), 5.0))
    d_primary = primary[(primary["node"] == "C") & (primary["split"] != "calib")].assign(node="D", market="MISO")
    d_g6 = g6.loc[d_primary.index].assign(node="D", market="MISO")
    primary = pd.concat([primary.drop(index=gone.index), d_primary], ignore_index=True)
    g6 = pd.concat([g6, d_g6], ignore_index=True)
    with pytest.MonkeyPatch.context() as mp, warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mp.setattr(module, "UNIFORM_FEES", ())
        rows, kept = module.economics_for(None, _syn_analysis(), None, None, 0.01, score_fn=_stub(primary, g6), keep_rows=True)
    notes = [str(w.message) for w in caught if "P1 does not have" in str(w.message)]
    return {"rows": rows, "kept": kept, "notes": notes, "n_extra": len(gone), "n_d": int((d_primary["split"] == "deploy").sum())}


def test_frames_are_restricted_to_p1s_rows_before_the_policies_rank_or_match(syn_messy):
    from es.run_economics import KEPT
    kept = syn_messy["kept"]
    key = ["market", "node", "timestamp_utc"]
    p1_keys = set(map(tuple, kept["P1"][key].to_numpy()))
    for name in KEPT:
        assert set(map(tuple, kept[name][key].to_numpy())) == p1_keys, name
    assert set(kept["P4"]["node"]) == {"A", "B", "C"}              # D, which P1 cannot place, is out of every policy (M2)
    n1 = kept["P1"].groupby("node")["alert"].sum()
    for name in ("calendar@count", "VOL@count"):                    # top_count ranks on P1's rows only: P1's count exactly (M1)
        assert (kept[name].groupby("node")["alert"].sum() == n1).all(), name
    assert (kept["P4@r*"].groupby("node")["alert"].sum() - n1).abs().max() <= max(3, 0.01 * n1.max())


def test_within_cuts_the_nodes_and_deployment_rows_p1_does_not_have_and_counts_them():
    from es.run_economics import _within
    primary, _ = _syn_scores(seed=5)
    key = ["market", "node", "timestamp_utc"]
    ab_deploy = primary[(primary["split"] == "deploy") & (primary["node"] != "C")]
    keys = ab_deploy[key].iloc[:-5]                                 # P1 has no row of node C and lacks B's last five
    out, n = _within(primary, keys)
    assert n == 5 + N_DEPLOY * 24                                   # the five, and node C's whole deployment window
    assert set(out["node"]) == {"A", "B"}                           # C leaves with all its rows
    for node in ("A", "B"):                                         # the rows the policies read stay: train, calib
        assert (out[out["node"] == node]["split"].value_counts().drop("deploy")
                == primary[primary["node"] == node]["split"].value_counts().drop("deploy")).all()
    assert set(map(tuple, out.loc[out["split"] == "deploy", key].to_numpy())) == set(map(tuple, keys.to_numpy()))
    ab = primary[primary["node"] != "C"]
    same, none = _within(ab, ab.loc[ab["split"] == "deploy", key])
    assert same is ab and none == 0                                 # nothing to cut: the frame itself comes back
    no_deploy = primary[primary["split"] != "deploy"]
    same_frame, zero = _within(no_deploy, keys)                     # no deployment row at all
    assert same_frame is no_deploy and zero == 0


def test_dropped_rows_are_counted_per_policy_in_one_warning(syn_messy):
    notes, n_extra, n_d = syn_messy["notes"], syn_messy["n_extra"], syn_messy["n_d"]
    assert len(notes) == 1                                          # one warning for the whole call
    msg = notes[0]

    def count(name):
        m = re.search(rf"(?<![\w@*]){re.escape(name)} (\d+)", msg)
        return int(m.group(1)) if m else 0
    assert count("calendar@count") == n_extra + n_d                 # the twelve extra rows and the node P1 cannot place
    for name in ("P4", "P5", "P6", "P4@r*", "VOL_P1", "VOL_P4", "VOL_P6", "VOL@count", "curve_P1_a0.05", "curve_P6_a0.01"):
        assert count(name) == n_d, name
    assert count("P1") == 0                                         # P1 defines the rows; it loses none


ALL_NAMES = ({"P1", "P4", "P5", "P6", "P4@r*", "calendar@count", "VOL_P1", "VOL_P4", "VOL_P6", "VOL@count"}
             | {f"curve_P{k}_a{a}" for k in (1, 6) for a in (0.01, 0.025, 0.05, 0.1)})


def test_dropped_row_counts_come_back_with_kept_and_agree_with_the_warning(syn_messy):
    """N2: the sealed entry point does not capture warnings, so the counts of Ruling 82 are also returned, as
    kept.attrs["dropped_rows"] = {policy: deployment rows dropped from its frame} for every policy and comparator."""
    kept, notes, n_extra, n_d = syn_messy["kept"], syn_messy["notes"], syn_messy["n_extra"], syn_messy["n_d"]
    assert isinstance(kept, dict) and isinstance(kept.attrs, dict)         # still a dict of frames, with attrs besides
    dropped = kept.attrs["dropped_rows"]
    assert set(dropped) == ALL_NAMES and len(dropped) == 18
    assert dropped["P1"] == 0                                              # P1 defines the rows
    assert dropped["calendar@count"] == n_extra + n_d
    assert all(dropped[name] == n_d for name in ALL_NAMES - {"P1", "calendar@count"})
    tail = notes[0].split("rows per policy: ")[1]                          # the warning lists the nonzero ones, and agrees
    listed = {entry.rsplit(" ", 1)[0]: int(entry.rsplit(" ", 1)[1]) for entry in tail.split(", ")}
    assert listed == {name: n for name, n in dropped.items() if n}


def test_dropped_row_counts_are_explicit_zeros_when_nothing_is_dropped(syn, syn_utc):
    for kept in (syn["kept"], syn_utc["kept"]):             # the UTC case: its edge rows were relabelled, not dropped
        dropped = kept.attrs["dropped_rows"]
        assert set(dropped) == ALL_NAMES and set(dropped.values()) == {0}


def test_summary_and_interval_estimands_describe_the_same_rows(syn_messy):
    """Before the restriction a node that only P4 handled moved decomposition.csv (P4's full rows) and not
    bootstrap_ci.csv (P1's rows): E1:NET:level changed sign between the two files (review, M2)."""
    from es import measures
    from es.run_economics import estimand_cis
    rows, kept = syn_messy["rows"], syn_messy["kept"]
    pick = {r["policy"]: r for r in rows if (r["position"], r["fees"], r["comparator"]) == ("premium", "calibrated", "none")}
    want = measures.decomposition(pick["P1"], pick["P4@r*"], pick["P4"])
    est = {r["stat"]: r["estimate"] for r in estimand_cis([kept], n_rep=10)}
    for k in measures.E1_KEYS:
        for part in ("timing", "level", "total"):
            assert est[f"E1:{k}:{part}"] == pytest.approx(want[k][part]), (k, part)


# ---------------------------------------------------------------------------------------------------------------
# Partial local days (spec §4.6; Ruling 82, review item M3)
# ---------------------------------------------------------------------------------------------------------------

def _syn_analysis_utc():
    from es.analyses import Analysis
    from es.windows import UtcWindow as U
    return Analysis("SYNU", U("2025-05-01", "2025-06-15"), U("2025-06-15", "2025-07-05"), U("2025-07-05", "2025-08-02"),
                    U("2025-05-01", "2025-06-15"), "rolling", "legacy")


def _syn_scores_utc(seed=0):
    """The three nodes hourly on the UTC clock and split by UTC windows, as the legacy analyses are. The deployment window
    cuts the first and the last local day of every node (4 and 20 hours in New York, 7 and 17 in California). The combined
    score is highest on those edge rows, so P1 alerts there; the calendar and volatility scores are lowest there, so a
    matched benchmark would not. Returns (combined frame, calendar frame, mask of the edge rows)."""
    from es.guard import settlement_frame
    rng = np.random.default_rng(seed)
    analysis = _syn_analysis_utc()
    frames = []
    for market, node in NODES:
        ts = pd.Series(pd.date_range("2025-05-01", "2025-08-02", freq="h", tz="UTC", inclusive="left"))
        s = settlement_frame(ts, pd.Series(market, index=ts.index))
        n = len(ts)
        hot = (rng.random(n // 24 + 1) < 0.1)[np.arange(n) // 24] & s["settle_hour"].between(16, 19).to_numpy()
        dart = rng.normal(4.0, 2.0, n) - 14.0 * hot
        da = 40.0 + rng.normal(0.0, 3.0, n)
        frame = pd.DataFrame({"market": market, "node": node, "node_type": "zone", "timestamp_utc": ts,
                              "settle_date": s["settle_date"], "settle_hour": s["settle_hour"], "weekday": s["weekday"],
                              "wclass": (s["weekday"] >= 5).astype(int), "da_lmp": da, "rt_lmp": da - dart,
                              "dart_spread": dart, "label": hot.astype(float),
                              "abs_mean_168h": np.abs(rng.normal(3.0, 1.0, n)), "score": 2.0 * hot + rng.normal(0, 1, n)})
        frame["split"] = np.select([analysis.train.mask(frame).to_numpy(), analysis.calib.mask(frame).to_numpy(),
                                    analysis.deploy.mask(frame).to_numpy()], ["train", "calib", "deploy"], default="other")
        frames.append(frame)
    primary = pd.concat(frames, ignore_index=True)
    edge = (primary["split"] == "deploy") & primary["settle_date"].isin([dt.date(2025, 7, 4), dt.date(2025, 8, 1)])
    primary.loc[edge, "score"] = 10.0
    primary.loc[edge, "abs_mean_168h"] = 0.0
    g6 = primary.assign(score=0.5 * primary["settle_hour"].between(16, 19) + 0.01 * primary["weekday"])
    g6.loc[edge, "score"] = -10.0
    return primary, g6, edge


def test_trim_partial_edges_relabels_the_edge_deployment_rows_only():
    from es.measures import drop_partial_edge_days
    from es.run_economics import _trim_partial_edges
    primary, g6, edge = _syn_scores_utc(seed=1)
    assert int(edge.sum()) == (4 + 20) + (4 + 20) + (7 + 17)        # New York, New England, California
    p, g = _trim_partial_edges(primary, g6, _syn_analysis_utc())
    assert (p.loc[edge, "split"] == "other").all() and (g.loc[edge, "split"] == "other").all()   # P4's window still sees them
    assert (p.loc[~edge, "split"] == primary.loc[~edge, "split"]).all()
    assert (g.loc[~edge, "split"] == g6.loc[~edge, "split"]).all()
    assert (primary.loc[edge, "split"] == "deploy").all() and (g6.loc[edge, "split"] == "deploy").all()      # inputs untouched
    assert len(p) == len(primary) and list(p.columns) == list(primary.columns)
    dep = p[p["split"] == "deploy"]
    assert len(drop_partial_edge_days(dep)) == len(dep)             # nothing partial is left at the new edges
    same_p, same_g = _trim_partial_edges(primary, g6, _syn_analysis())          # a DateWindow deployment is not touched
    assert same_p is primary and same_g is g6


@pytest.fixture(scope="module")
def syn_utc():
    """economics_for and p1_rows_for on a UtcWindow deployment (fee grid cut to calibrated and stress)."""
    import es.run_economics as module
    primary, g6, _ = _syn_scores_utc()
    analysis = _syn_analysis_utc()
    with pytest.MonkeyPatch.context() as mp, warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mp.setattr(module, "UNIFORM_FEES", ())
        rows, kept = module.economics_for(None, analysis, None, None, 0.01, score_fn=_stub(primary, g6), keep_rows=True)
        p1_rows = module.p1_rows_for(None, analysis, None, None, score_fn=_stub(primary, g6))
    return {"rows": rows, "kept": kept, "p1_rows": p1_rows,
            "notes": [str(w.message) for w in caught if "P1 does not have" in str(w.message)]}


def test_partial_edge_days_of_a_utc_window_are_out_before_the_counts_are_matched(syn_utc):
    """Before: target = P1's count on the whole UTC window, then each policy lost its own edge alerts. P1 alerts on all
    24 + 24 + 24 edge rows here and the benchmarks on none, so the measured counts differed by the edge alerts."""
    kept = syn_utc["kept"]
    p1 = kept["P1"]
    assert (p1["settle_date"].min(), p1["settle_date"].max()) == (dt.date(2025, 7, 5), dt.date(2025, 7, 31))
    assert p1["settle_date"].nunique() == 27 and len(p1) == 3 * 27 * 24
    n1 = p1.groupby("node")["alert"].sum()
    assert (n1 > 0).all()
    for name in ("calendar@count", "VOL@count"):
        assert (kept[name].groupby("node")["alert"].sum() == n1).all(), name       # exactly P1's count on the measured rows
    assert (kept["P4@r*"].groupby("node")["alert"].sum() - n1).abs().max() <= max(3, 0.01 * n1.max())
    key = ["market", "node", "timestamp_utc"]
    for name in ("P4", "P4@r*", "calendar@count", "VOL@count"):
        assert set(map(tuple, kept[name][key].to_numpy())) == set(map(tuple, p1[key].to_numpy())), name
    assert syn_utc["notes"] == []             # the edge rows left both frames together: nothing was dropped as "not P1's"


def test_p1_rows_for_is_the_kept_p1_frame_for_a_utc_window_too(syn_utc):
    pd.testing.assert_frame_equal(syn_utc["p1_rows"], syn_utc["kept"]["P1"])


def _utc_alerts():
    from es.guard import settlement_frame
    ts = pd.Series(pd.date_range("2025-07-01", periods=240, freq="h", tz="UTC"))
    s = settlement_frame(ts, pd.Series("NYISO", index=ts.index))
    rng = np.random.default_rng(0)
    alerts = pd.DataFrame({"market": "NYISO", "node": "A", "timestamp_utc": ts, "settle_date": s["settle_date"],
                           "settle_hour": s["settle_hour"], "weekday": s["weekday"],
                           "wclass": (s["weekday"] >= 5).astype(int), "da_lmp": 40.0,
                           "rt_lmp": 40.0 - rng.normal(1.0, 2.0, 240), "abs_mean_168h": 1.0,
                           "score": rng.normal(size=240), "split": "deploy", "alert": rng.random(240) < 0.1})
    g6 = alerts[["market", "node", "timestamp_utc"]].assign(score=rng.normal(size=240))
    return alerts, g6


def test_prepare_attaches_settlement_and_deciles_and_drops_no_row():
    """The partial-day rule is applied once, to the score frames, before the policies run (_trim_partial_edges); _prepare
    does not apply it again, so it cannot undo the match of the counts."""
    from es.run_economics import _prepare
    alerts, g6 = _utc_alerts()      # UTC 1-10 July, NYISO: local days 30 June (4 h), 1-9 July (24 h), 10 July (20 h)
    out = _prepare(alerts, g6, None, None)
    assert len(out) == 240
    assert {"rt_settle", "dart_settle", "settle_approx", "g6_score", "cal_decile", "month"} <= set(out.columns)
    assert out["settle_approx"].all()                                # NYISO without the integrated table


# ---------------------------------------------------------------------------------------------------------------
# run_all: the files main() writes, with economics_for stubbed
# ---------------------------------------------------------------------------------------------------------------

def _analysis_with_id(analysis_id):
    from dataclasses import replace
    return replace(_syn_analysis(), id=analysis_id)


def _fake_economics_for(record):
    """economics_for stand-in: the summaries and kept rows of three premium policies on _kept data (node A in CAISO)."""
    from es import measures
    starts = {"AQ3": (11, dt.date(2025, 7, 1)), "AQ4": (12, dt.date(2025, 10, 1)), "AQ1": (13, dt.date(2026, 1, 1))}

    def fake(panel, analysis, fmm, nyi, eta, score_fn=None, keep_rows=False):
        record.append((analysis.id, eta, keep_rows))
        seed, start = starts[analysis.id]
        kept = _with_caiso(_kept(seed, start=start))
        rows = [{**measures.summarize(kept[p]), "analysis": analysis.id, "policy": p, "comparator": "none",
                 "position": "premium", "fees": "calibrated"} for p in ("P1", "P4@r*", "P4")]
        rows.append({**measures.summarize(kept["P4"]), "analysis": analysis.id, "policy": "P1", "comparator": "none",
                     "position": "premium", "fees": "0.1"})                  # a decoy the decomposition must not pick
        rows.append({**measures.summarize(kept["P4"]), "analysis": analysis.id, "policy": "P1", "comparator": "none",
                     "position": "inc", "fees": "calibrated"})
        rows.append({**measures.summarize(kept["P4"]), "analysis": analysis.id, "policy": "comparator",
                     "comparator": "VOL_P1", "position": "premium", "fees": "calibrated"})
        return (rows, kept) if keep_rows else rows
    return fake


def test_run_all_writes_the_five_files(tmp_path, monkeypatch):
    import es.run_economics as module
    from es import measures
    from es.bootstrap import RowCube, rotation_test
    record = []
    monkeypatch.setattr(module, "economics_for", _fake_economics_for(record))
    out = module.run_all("panel", "fmm", "nyi", 0.01, [_analysis_with_id(i) for i in ("AQ3", "AQ4", "AQ1")],
                         tmp_path, n_rep=12, eta_source="protocol_inputs.json (test)")
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "bootstrap_ci.csv", "decomposition.csv", "rotation.csv", "run_meta.json", "summary.csv", "volmanagement_check.csv"]
    assert record == [("AQ3", 0.01, True), ("AQ4", 0.01, True), ("AQ1", 0.01, True)]
    summary, decomp, cis, rot, vm = (pd.read_csv(tmp_path / f"{n}.csv") for n in
                                     ("summary", "decomposition", "bootstrap_ci", "rotation", "volmanagement_check"))
    assert len(summary) == 18 and set(out) == {"summary", "decomposition", "bootstrap_ci", "rotation", "volmanagement_check"}

    # decomposition.csv: the E1 keys' timing / level / total, from the premium calibrated rows only
    assert list(decomp["analysis"]) == ["AQ3", "AQ4", "AQ1"]
    s = {p: measures.summarize(_with_caiso(_kept(11, start=dt.date(2025, 7, 1)))[p]) for p in ("P1", "P4@r*", "P4")}
    want = measures.decomposition(s["P1"], s["P4@r*"], s["P4"])
    for k in measures.E1_KEYS:
        for part in ("timing", "level", "total"):
            assert decomp.loc[0, f"{k}_{part}"] == pytest.approx(want[k][part])

    # bootstrap_ci.csv: every analysis, its ex-CAISO run, and the pooled rehearsal of AQ3 + AQ4
    pooled = "AQ3+AQ4 (rehearsal of the pooled estimands)"
    assert list(dict.fromkeys(cis["analysis"])) == ["AQ3", "AQ3|ex-CAISO", "AQ4", "AQ4|ex-CAISO", "AQ1", "AQ1|ex-CAISO", pooled]
    assert list(cis.columns) == ["analysis", "stat", "estimate", "lo", "hi", "approximate", "n_windows"]
    assert set(cis.loc[cis["analysis"] == pooled, "n_windows"]) == {2}
    assert set(cis.loc[cis["analysis"] != pooled, "n_windows"]) == {1}
    for name in dict.fromkeys(cis["analysis"]):                    # Ruling 81: the paired contrasts are in the file
        stats = set(cis.loc[cis["analysis"] == name, "stat"])
        assert {"E4:dSELC|calendar@count", "E4:dSELC|VOL@count", "E4:SELC|calendar@count", "E4:SELC|VOL@count"} <= stats, name
        assert {"E4:dESred|calendar@count", "E4:dESred|VOL@count",
                "E4:ESred|calendar@count", "E4:ESred|VOL@count"} <= stats, name
    k3, k4 = _with_caiso(_kept(11, start=dt.date(2025, 7, 1))), _with_caiso(_kept(12, start=dt.date(2025, 10, 1)))

    def net(name, stat="NET"):
        return float(cis[(cis["analysis"] == name) & (cis["stat"] == stat)]["estimate"].iloc[0])
    assert net("AQ3") == pytest.approx(measures.summarize(k3["P1"])["NET"])
    assert net("AQ3|ex-CAISO") == pytest.approx(measures.summarize(k3["P1"][k3["P1"]["market"] != "CAISO"])["NET"])
    assert net(pooled) == pytest.approx(measures.summarize(k3["P1"])["NET"] + measures.summarize(k4["P1"])["NET"])

    # rotation.csv: scalars only, the full cyclic group, the pooled rehearsal on its own row
    assert list(rot.columns) == ["window", "t_obs", "p", "n_null", "selc_obs", "retained_min", "retained_mean"]
    assert list(rot["window"]) == ["AQ3", "AQ4", "AQ1", "AQ3+AQ4 (rehearsal of the pooled H-E test)"]
    assert list(rot["n_null"]) == [30 - 1, 30 - 1, 30 - 1, 30 * 30 - 1]
    c3, c4 = RowCube.from_rows(k3["P1"]), RowCube.from_rows(k4["P1"])
    both = rotation_test([c3, c4])
    assert rot.loc[3, "p"] == pytest.approx(both["p"]) and rot.loc[3, "t_obs"] == pytest.approx(both["t_obs"])
    assert rot.loc[3, "selc_obs"] == pytest.approx(both["selc_obs"])
    assert rot.loc[0, "selc_obs"] == pytest.approx(measures.summarize(k3["P1"])["SELC"])
    assert (rot["retained_min"] == 1.0).all() and (rot["retained_mean"] == 1.0).all()      # complete grids

    # volmanagement_check.csv: Sharpe-like ratios of the alerted and the non-alerted hours of P1
    p1 = k3["P1"]
    alerted, pnl = p1["alert"].to_numpy(bool), p1["pnl_open"].to_numpy(float)
    assert vm.loc[0, "sharpe_alerted"] == pytest.approx(pnl[alerted].mean() / pnl[alerted].std())
    assert vm.loc[0, "sharpe_not_alerted"] == pytest.approx(pnl[~alerted].mean() / pnl[~alerted].std())

    # run_meta.json: what the run used, and that it finished
    meta = json.loads((tmp_path / "run_meta.json").read_text())
    assert meta["status"] == "complete" and meta["analyses_done"] == ["AQ3", "AQ4", "AQ1"]
    assert meta["analyses"] == meta["analyses_done"]
    assert (meta["n_rep"], meta["seed"], meta["mean_block"]) == (12, 20261022, 7)
    assert (meta["eta"], meta["eta_source"]) == (0.01, "protocol_inputs.json (test)")
    assert (meta["alpha"], meta["W"]) == (0.05, 56)
    assert meta["strata_primary"] == ["market", "node", "settle_hour", "wclass", "month"]
    assert meta["strata_decile"] == ["market", "node", "cal_decile"]
    assert meta["rotation_test_defaults"] == {"min_shift": 1}      # the run never passes one (Ruling 69)
    started, finished = (dt.datetime.fromisoformat(meta[k].replace("Z", "+00:00")) for k in ("started_utc", "finished_utc"))
    assert started <= finished


def test_run_all_without_both_pooling_quarters_writes_no_pooled_rows(tmp_path, monkeypatch):
    import es.run_economics as module
    monkeypatch.setattr(module, "economics_for", _fake_economics_for([]))
    module.run_all("panel", "fmm", "nyi", 0.01, [_analysis_with_id("AQ3"), _analysis_with_id("AQ1")], tmp_path, n_rep=8)
    cis, rot = pd.read_csv(tmp_path / "bootstrap_ci.csv"), pd.read_csv(tmp_path / "rotation.csv")
    assert not cis["analysis"].str.contains("rehearsal").any() and not rot["window"].str.contains("rehearsal").any()


def test_run_all_rewrites_the_files_after_each_analysis_so_a_later_failure_keeps_the_earlier_rows(tmp_path, monkeypatch):
    import es.run_economics as module
    fake = _fake_economics_for([])

    def failing_on_the_second(panel, analysis, fmm, nyi, eta, score_fn=None, keep_rows=False):
        if analysis.id == "AQ4":
            raise RuntimeError("boom in AQ4")
        return fake(panel, analysis, fmm, nyi, eta, score_fn, keep_rows)
    monkeypatch.setattr(module, "economics_for", failing_on_the_second)
    with pytest.raises(RuntimeError, match="boom in AQ4"):
        module.run_all("panel", "fmm", "nyi", 0.01, [_analysis_with_id(i) for i in ("AQ3", "AQ4", "AQ1")], tmp_path,
                       n_rep=8, eta_source="x.json")
    summary, decomp, cis, rot, vm = (pd.read_csv(tmp_path / f"{n}.csv") for n in
                                     ("summary", "decomposition", "bootstrap_ci", "rotation", "volmanagement_check"))
    assert set(summary["analysis"]) == {"AQ3"} and len(summary) == 6
    assert list(decomp["analysis"]) == ["AQ3"] and list(rot["window"]) == ["AQ3"] and list(vm["analysis"]) == ["AQ3"]
    assert list(dict.fromkeys(cis["analysis"])) == ["AQ3", "AQ3|ex-CAISO"]
    meta = json.loads((tmp_path / "run_meta.json").read_text())
    assert meta["status"] == "failed" and meta["analyses_done"] == ["AQ3"] and meta["failed_on"] == "AQ4"
    assert "boom in AQ4" in meta["error"] and meta["finished_utc"] is not None
    assert (meta["n_rep"], meta["eta_source"]) == (8, "x.json")
    assert not list(tmp_path.glob("*.tmp"))                         # nothing half-written is left behind


def _atomic_failing_when(module, fail_when):
    """A stand-in for module._atomic that raises OSError when fail_when(file name, how many times it was written)."""
    real, seen = module._atomic, {}

    def atomic(path, write):
        seen[path.name] = seen.get(path.name, 0) + 1
        if fail_when(path.name, seen[path.name]):
            raise OSError(f"disk full writing {path.name}")
        return real(path, write)
    return atomic


def _analyses_in_files(econ_dir):
    """The analysis ids each of the five files holds ('<id>|ex-CAISO' counts as <id>)."""
    out = {}
    for name in ("summary", "decomposition", "bootstrap_ci", "rotation", "volmanagement_check"):
        col = "window" if name == "rotation" else "analysis"
        out[name] = set(pd.read_csv(econ_dir / f"{name}.csv")[col].str.split("|").str[0])
    return out


def test_analyses_done_names_only_analyses_whose_files_were_all_replaced(tmp_path, monkeypatch):
    """N1: a write that fails part-way (the second write of bootstrap_ci.csv, i.e. while AQ4 is being written) leaves
    summary.csv and decomposition.csv with AQ4 and the other three without. run_meta.json must not claim AQ4."""
    import es.run_economics as module
    monkeypatch.setattr(module, "economics_for", _fake_economics_for([]))
    monkeypatch.setattr(module, "_atomic", _atomic_failing_when(module, lambda name, k: name == "bootstrap_ci.csv" and k == 2))
    with pytest.raises(OSError, match="disk full writing bootstrap_ci.csv"):
        module.run_all("panel", "fmm", "nyi", 0.01, [_analysis_with_id(i) for i in ("AQ3", "AQ4", "AQ1")], tmp_path, n_rep=6)
    meta = json.loads((tmp_path / "run_meta.json").read_text())
    assert meta["status"] == "failed" and meta["failed_on"] == "AQ4" and "disk full" in meta["error"]
    assert meta["analyses_done"] == ["AQ3"]                         # not AQ4: its files are only partly there
    held = _analyses_in_files(tmp_path)
    assert held["summary"] == held["decomposition"] == {"AQ3", "AQ4"}              # replaced before the failure
    assert held["bootstrap_ci"] == held["rotation"] == held["volmanagement_check"] == {"AQ3"}
    assert all(done in ids for done in meta["analyses_done"] for ids in held.values())     # every named analysis is in all five


def test_analyses_done_includes_an_analysis_whose_files_are_all_there_when_only_the_meta_write_fails(tmp_path, monkeypatch):
    """The id is appended after the five CSVs and before run_meta.json is written: if that write fails, the five files hold
    AQ4 completely, so the failure record (written again by the handler) names it."""
    import es.run_economics as module
    monkeypatch.setattr(module, "economics_for", _fake_economics_for([]))
    monkeypatch.setattr(module, "_atomic", _atomic_failing_when(module, lambda name, k: name == "run_meta.json" and k == 3))
    with pytest.raises(OSError, match="disk full writing run_meta.json"):
        module.run_all("panel", "fmm", "nyi", 0.01, [_analysis_with_id(i) for i in ("AQ3", "AQ4", "AQ1")], tmp_path, n_rep=6)
    meta = json.loads((tmp_path / "run_meta.json").read_text())
    assert meta["status"] == "failed" and meta["failed_on"] == "AQ4" and meta["analyses_done"] == ["AQ3", "AQ4"]
    held = _analyses_in_files(tmp_path)
    assert all(ids == {"AQ3", "AQ4"} for ids in held.values())


def test_run_all_writes_every_file_by_replacing_a_temporary_one(tmp_path, monkeypatch):
    import os
    from pathlib import Path

    import es.run_economics as module
    monkeypatch.setattr(module, "economics_for", _fake_economics_for([]))
    calls, real_replace = [], os.replace

    def spy(src, dst):
        calls.append((Path(src).name, Path(dst).name))
        return real_replace(src, dst)
    monkeypatch.setattr(module.os, "replace", spy)
    module.run_all("panel", "fmm", "nyi", 0.01, [_analysis_with_id("AQ3"), _analysis_with_id("AQ1")], tmp_path, n_rep=8)
    names = {"summary.csv", "decomposition.csv", "bootstrap_ci.csv", "rotation.csv", "volmanagement_check.csv", "run_meta.json"}
    assert {dst for _, dst in calls} == names
    assert all(src == dst + ".tmp" for src, dst in calls)           # written aside, then moved over the real name
    assert all(sum(d == n for _, d in calls) >= 3 for n in names)   # after each of the two analyses, and at the end
    assert not list(tmp_path.glob("*.tmp"))
