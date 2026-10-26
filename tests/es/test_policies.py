import datetime as dt
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _scores(n_cal=60, n_dep=60, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    start = dt.date(2025, 1, 1)
    for split, ndays, mu in (("train", 30, 0.0), ("calib", n_cal, 0.0), ("deploy", n_dep, shift)):
        for k in range(ndays):
            day = start + dt.timedelta(days=len(rows) // 24)
            for h in range(24):
                rows.append({"market": "M", "node": "A", "split": split, "settle_date": day,
                             "timestamp_utc": pd.Timestamp(day) + pd.Timedelta(hours=h),
                             "score": rng.normal(mu, 1.0)})
    return pd.DataFrame(rows)


def test_p1_fixed_uses_calibration_quantile():
    from es.policies import p1_fixed
    s = _scores()
    out = p1_fixed(s, 0.05)
    tau = np.quantile(s.loc[s.split == "calib", "score"], 0.95)
    assert np.isclose(out["tau"].iloc[0], tau)
    assert (out["alert"] == (out["score"] >= tau)).all()
    assert set(out["split"]) == {"deploy"}


def test_p6_exact_hits_alpha():
    from es.policies import activity, p6_exact
    out = p6_exact(_scores(shift=1.0), 0.05)
    assert abs(activity(out, 0.05)["R"].iloc[0] - 1.0) < 0.02


def test_p4_rolling_is_causal_and_adapts():
    from es.policies import activity, p1_fixed, p4_rolling
    s = _scores(shift=1.5)
    base = p4_rolling(s, 0.05, window_days=14)
    late = s.copy()
    last_day = late.loc[late.split == "deploy", "settle_date"].max()
    late.loc[late.settle_date == last_day, "score"] += 100.0      # perturb the future only
    pert = p4_rolling(late, 0.05, window_days=14)
    earlier = base.settle_date < last_day
    assert (base.loc[earlier, "alert"].to_numpy() == pert.loc[earlier.to_numpy(), "alert"].to_numpy()).all()
    assert activity(base, 0.05)["R"].iloc[0] < activity(p1_fixed(s, 0.05), 0.05)["R"].iloc[0]


def test_p4_rolling_excludes_training_scores():
    from es.policies import p4_rolling
    s = _scores()
    s.loc[s.split == "train", "score"] = 1e9                    # would dominate any window
    out = p4_rolling(s, 0.05, window_days=200)
    assert np.isfinite(out["tau"]).all() and out["tau"].max() < 1e8


def test_p5_moves_alpha_against_overfiring():
    from es.policies import activity, p1_fixed, p5_self_correcting
    s = _scores(shift=1.0)
    r5 = activity(p5_self_correcting(s, 0.05, eta=0.05), 0.05)["R"].iloc[0]
    r1 = activity(p1_fixed(s, 0.05), 0.05)["R"].iloc[0]
    assert r5 < r1


def _p5_case(days, n_cal=100):
    """Calibration scores 0..n_cal-1, so the (1 - a) quantile is (n_cal - 1) * (1 - a).
    Then one list of scores per deployment day."""
    t0, d0 = pd.Timestamp("2025-01-01"), pd.Timestamp("2025-03-01")
    rows = [("calib", t0 + pd.Timedelta(hours=i), float(i)) for i in range(n_cal)]
    rows += [("deploy", d0 + pd.Timedelta(days=k, hours=h), float(sc))
             for k, day in enumerate(days) for h, sc in enumerate(day)]
    split, ts, score = zip(*rows)
    return pd.DataFrame({"market": "M", "node": "A", "split": split, "timestamp_utc": ts,
                         "settle_date": [t.date() for t in ts], "score": score})


def test_p5_follows_the_hand_computed_alpha_recursion():
    from es.policies import p5_self_correcting
    s = _p5_case([[95, 50, 99, 10], [99, 98, 5, 6], [1, 2, 3, 4]])
    out = p5_self_correcting(s, 0.1, eta=0.5)
    # Day 1 uses a_1 = alpha = 0.1, so tau = 99 * 0.9 = 89.1. Alerts: 95 and 99, so r_1 = 0.5 and
    # a_2 = clip(0.1 + 0.5 * (0.1 - 0.5)) = clip(-0.1) = 0.001.
    # Day 2 uses a_2: tau = 99 * 0.999 = 98.901. Alert: 99 only (98 < 98.901), so r_2 = 0.25 and
    # a_3 = clip(0.001 + 0.5 * (0.1 - 0.25)) = 0.001 again, so day 3 also uses 98.901.
    assert np.allclose(out["tau"], [89.1] * 4 + [98.901] * 8, rtol=0, atol=1e-9)
    assert out["alert"].tolist() == [True, False, True, False, True, False, False, False] + [False] * 4
    # The clip is an argument: with lo = 0.01 day 2 and day 3 use 99 * 0.99.
    wide = p5_self_correcting(s, 0.1, eta=0.5, lo=0.01)
    assert np.allclose(wide["tau"], [89.1] * 4 + [98.01] * 8, rtol=0, atol=1e-9)


def test_p5_upper_clip_caps_alpha_when_a_large_eta_overshoots():
    from es.policies import p5_self_correcting
    s = _p5_case([[1, 2, 3, 4], [5, 6, 7, 8], [60, 40, 55, 45]])
    out = p5_self_correcting(s, 0.1, eta=5.0)
    # No alert on day 1 (tau 89.1), so a_2 = min(0.5, 0.1 + 5 * 0.1) = 0.5 and tau = 99 * 0.5 = 49.5.
    # None on day 2 either, and a_3 = min(0.5, 0.5 + 0.5) = 0.5, so day 3 stays at 49.5.
    assert np.allclose(out["tau"], [89.1] * 4 + [49.5] * 8, rtol=0, atol=1e-9)
    assert out["alert"].tolist() == [False] * 8 + [True, False, True, False]
    # The cap is an argument: with hi = 0.2, a_2 = a_3 = 0.2 and tau = 99 * 0.8.
    capped = p5_self_correcting(s, 0.1, eta=5.0, hi=0.2)
    assert np.allclose(capped["tau"].to_numpy()[4:], 79.2, rtol=0, atol=1e-9)


def test_p5_realized_rate_counts_a_score_equal_to_tau_as_an_alert():
    from es.policies import p5_self_correcting
    s = _p5_case([[75, 80, 10, 20], [1, 2, 3, 4]], n_cal=101)       # calibration 0..100: tau = 100 * (1 - a)
    out = p5_self_correcting(s, 0.25, eta=0.5)
    # Day 1: tau = 75.0 exactly, so 75 and 80 both alert (>=): r_1 = 0.5 and a_2 = 0.25 + 0.5 * (0.25 - 0.5)
    # = 0.125, giving tau_2 = 87.5. Were 75 not counted, r_1 = 0.25, a_2 = 0.25 and tau_2 would stay 75.
    assert np.allclose(out["tau"], [75.0] * 4 + [87.5] * 4, rtol=0, atol=1e-9)
    assert out["alert"].tolist() == [True, True, False, False] + [False] * 4


def test_p4_matched_hits_target_count():
    from es.policies import p4_matched
    s = _scores(shift=1.0)
    out = p4_matched(s, {("M", "A"): 200}, window_days=14)
    assert abs(int(out["alert"].sum()) - 200) <= 3


def test_p2_uses_only_season_matched_training_rows():
    from es.policies import p2_insample
    s = _scores()
    # The helper's 30 training days all fall in January, which cannot tell a season filter from none.
    # Spread the training rows over Jan-Apr 2024 (6 a day) and make January the only month scoring 0.
    tr = s.index[s.split == "train"]
    s.loc[tr, "settle_date"] = [dt.date(2024, 1, 1) + dt.timedelta(days=i // 6) for i in range(len(tr))]
    month = s.loc[tr, "settle_date"].map(lambda d: d.month)
    s.loc[tr, "score"] = np.where(month == 1, 0.0, 50.0)
    assert p2_insample(s, 0.05, months={1})["tau"].iloc[0] == 0.0
    assert p2_insample(s, 0.05, months={2, 3})["tau"].iloc[0] == 50.0


def test_activity_floor_and_failure():
    from es.policies import activity
    a = pd.DataFrame({"market": "M", "node": "A", "alert": [False] * 100})
    s = activity(a, 0.05).iloc[0]
    assert s.zero_activity and s["fail_0.5_2"] and np.isclose(s.R_floored, 1 / (2 * 100 * 0.05))


def test_top_count_selects_highest_scores():
    from es.policies import top_count
    s = _scores()
    out = top_count(s, {("M", "A"): 10})
    dep = s[s.split == "deploy"].sort_values("score", ascending=False)
    assert set(out.loc[out.alert, "score"]) == set(dep["score"].iloc[:10])


def test_p4_window_is_d_minus_w_to_d_minus_1_of_non_train_rows():
    from es.policies import p4_rolling
    one = dt.timedelta(days=1)
    D, W = dt.date(2025, 3, 1), 7

    def row(day, split, score, h=0):
        return {"market": "M", "node": "A", "split": split, "settle_date": day,
                "timestamp_utc": pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=h), "score": score}

    s = pd.DataFrame([
        row(D - (W + 1) * one, "calib", 1000.0),    # one day too old: excluded
        row(D - W * one, "calib", 1.0),             # first day of the window: included
        row(D - 3 * one, "train", 5000.0),          # in-sample: excluded
        row(D - one, "other", 2.0),                 # gap row: included
        row(D, "deploy", 10.0, h=0), row(D, "deploy", 3.0, h=1),
        row(D, "other", 9000.0, h=2),               # same day as the cutoff: excluded
        row(D + one, "deploy", 7.0),
    ])
    out = p4_rolling(s, 0.5, window_days=W)          # alpha = 0.5, so tau is the median
    assert (out.loc[out.settle_date == D, "tau"] == 1.5).all()              # median of {1, 2}
    # the next day's window holds D's own rows: earlier deployment days are out-of-sample history
    assert (out.loc[out.settle_date == D + one, "tau"] == 6.5).all()        # median of {2, 3, 10, 9000}


def test_alert_fires_at_equality_with_the_cutoff():
    from es.policies import p6_exact
    s = pd.DataFrame({"market": "M", "node": "A", "split": "deploy", "settle_date": dt.date(2025, 1, 1),
                      "timestamp_utc": pd.Timestamp("2025-01-01") + pd.to_timedelta(np.arange(21), "h"),
                      "score": np.arange(21, dtype=float)})
    out = p6_exact(s, 0.05)                          # (n - 1) * 0.95 = 19 exactly, so tau is itself a score
    assert out["tau"].iloc[0] == 19.0
    assert int(out["alert"].sum()) == 2              # scores 19 and 20: >=, not >


def test_activity_band_is_closed_and_zero_activity_log_error_uses_the_floor():
    from es.policies import activity

    def a(k, n=40):
        return pd.DataFrame({"market": "M", "node": "A", "alert": [True] * k + [False] * (n - k)})

    assert activity(a(1), 0.05).iloc[0].R == 0.5 and not activity(a(1), 0.05).iloc[0]["fail_0.5_2"]
    assert activity(a(4), 0.05).iloc[0].R == 2.0 and not activity(a(4), 0.05).iloc[0]["fail_0.5_2"]
    assert activity(a(5), 0.05).iloc[0]["fail_0.5_2"]
    z = activity(a(0, n=100), 0.05).iloc[0]
    assert np.isclose(z.log_error, 1.0)              # |log10(1 / (2 * 100 * 0.05))|


def test_p4_matched_is_p4_rolling_at_its_matched_rate():
    from es.policies import p4_matched, p4_rolling
    s = _scores(shift=1.0)
    out = p4_matched(s, {("M", "A"): 200}, window_days=14)
    ref = p4_rolling(s, float(out["matched_rate"].iloc[0]), window_days=14)
    assert (out["alert"].to_numpy() == ref["alert"].to_numpy()).all()
    assert int(out["alert"].sum()) == 200


# --- skipped nodes warn once, an empty result raises, an unmatched P4@r* is flagged -------------

def _targets(both):
    return {("M", "A"): 100, ("M", "B"): 100} if both else {("M", "A"): 100}


def _two_nodes(lacks=()):
    """Nodes A and B, both complete, except that B has no rows in the splits named in ``lacks``."""
    s = pd.concat([_scores(), _scores(seed=1).assign(node="B")], ignore_index=True)
    return s[~((s.node == "B") & s.split.isin(lacks))].reset_index(drop=True)


#: (policy, call(P, frame, both_nodes_targeted), splits that node B lacks, reason that must be named)
SKIP_CASES = [
    ("p1_fixed", lambda P, s, both: P.p1_fixed(s, 0.05), ("calib",), "no calibration rows"),
    ("p2_insample", lambda P, s, both: P.p2_insample(s, 0.05, months={1}), ("train",),
     "no training rows in the season months"),
    ("p4_rolling", lambda P, s, both: P.p4_rolling(s, 0.05, window_days=14), ("deploy",),
     "no deployment rows"),
    ("p5_self_correcting", lambda P, s, both: P.p5_self_correcting(s, 0.05, eta=0.05), ("calib",),
     "no calibration rows"),
    ("p6_exact", lambda P, s, both: P.p6_exact(s, 0.05), ("deploy",), "no deployment rows"),
    ("p4_matched", lambda P, s, both: P.p4_matched(s, _targets(both), window_days=14), (),
     "not in target_counts"),
    ("top_count", lambda P, s, both: P.top_count(s, _targets(both)), (), "not in target_counts"),
]


@pytest.mark.parametrize("name,call,lacks,reason", SKIP_CASES, ids=[c[0] for c in SKIP_CASES])
def test_a_skipped_node_gets_one_consolidated_warning_naming_it_and_why(name, call, lacks, reason):
    import es.policies as P
    with pytest.warns(UserWarning) as rec:
        out = call(P, _two_nodes(lacks), False)
    ours = [w for w in rec if issubclass(w.category, UserWarning)]
    assert len(ours) == 1
    msg = str(ours[0].message)
    assert msg.startswith(f"{name}: 1 node(s) skipped: ") and f"M/B ({reason}" in msg and "M/A" not in msg
    assert Path(ours[0].filename).name == Path(__file__).name       # attributed to the caller
    assert set(out["node"]) == {"A"}
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)                 # nothing skipped, so nothing said
        full = call(P, _two_nodes(), True)
    assert set(full["node"]) == {"A", "B"}


def test_the_warning_names_every_skipped_node_with_each_of_its_reasons():
    from es.policies import p1_fixed
    lacks = {"B": ("calib",), "C": ("deploy",), "D": ("calib", "deploy")}
    s = pd.concat([_scores(seed=i).assign(node=n) for i, n in enumerate("ABCD")], ignore_index=True)
    s = s[[sp not in lacks.get(n, ()) for sp, n in zip(s.split, s.node)]]
    with pytest.warns(UserWarning) as rec:
        out = p1_fixed(s, 0.05)
    (msg,) = [str(w.message) for w in rec]
    assert msg.startswith("p1_fixed: 3 node(s) skipped: ")
    for part in ("M/B (no calibration rows)", "M/C (no deployment rows)",
                 "M/D (no calibration rows and no deployment rows)"):
        assert part in msg
    assert set(out["node"]) == {"A"}


#: (policy, call(P, frame), splits that the only node lacks, reason that must be named)
NO_NODE_CASES = [
    ("p1_fixed", lambda P, s: P.p1_fixed(s, 0.05), ("calib",), "no calibration rows"),
    ("p2_insample", lambda P, s: P.p2_insample(s, 0.05, months={1}), ("train",),
     "no training rows in the season months"),
    ("p4_rolling", lambda P, s: P.p4_rolling(s, 0.05, window_days=14), ("deploy",), "no deployment rows"),
    ("p5_self_correcting", lambda P, s: P.p5_self_correcting(s, 0.05, eta=0.05), ("calib",),
     "no calibration rows"),
    ("p6_exact", lambda P, s: P.p6_exact(s, 0.05), ("deploy",), "no deployment rows"),
    ("p4_matched", lambda P, s: P.p4_matched(s, {}, window_days=14), (), "not in target_counts"),
    ("p4_matched", lambda P, s: P.p4_matched(s, {("M", "A"): 5}, window_days=14), ("deploy",),
     "no deployment rows"),
    ("top_count", lambda P, s: P.top_count(s, {}), (), "not in target_counts"),
]


@pytest.mark.parametrize("name,call,lacks,reason", NO_NODE_CASES,
                         ids=[f"{c[0]}:{c[3]}" for c in NO_NODE_CASES])
def test_no_qualifying_node_raises_a_value_error_naming_the_policy_and_why(name, call, lacks, reason):
    import es.policies as P
    s = _scores()
    with pytest.raises(ValueError) as exc:
        call(P, s[~s.split.isin(lacks)])
    msg = str(exc.value)
    assert msg.startswith(f"{name}: no node qualifies") and f"M/A ({reason}" in msg


def test_an_empty_score_frame_raises_a_value_error_too():
    from es.policies import p6_exact
    with pytest.raises(ValueError, match="p6_exact: no node qualifies.*the score frame has no nodes"):
        p6_exact(_scores().iloc[:0], 0.05)


def test_p4_matched_reports_its_count_and_warns_when_the_target_cannot_be_met():
    from es.policies import p4_matched
    s = _scores(shift=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)                 # a reachable target is silent
        ok = p4_matched(s, {("M", "A"): 200}, window_days=14)
    assert ok["matched_count"].iloc[0] == int(ok["alert"].sum()) == 200
    assert 1e-4 < ok["matched_rate"].iloc[0] < 0.5
    assert list(ok.columns[-4:]) == ["tau", "alert", "matched_rate", "matched_count"]
    # 10**6 is more than rate 0.5 can give, so the search ends on the upper bound; 0 is fewer than
    # rate 1e-4 gives, so it ends on the lower bound
    for target, bound in ((10 ** 6, "0.5"), (0, "0.0001")):
        with pytest.warns(UserWarning) as rec:
            out = p4_matched(s, {("M", "A"): target}, window_days=14)
        (w,) = rec
        msg = str(w.message)
        realized = int(out["alert"].sum())
        assert out["matched_count"].iloc[0] == realized and realized != target
        assert msg.startswith("p4_matched: 1 node(s) not matched")
        assert f"M/A (target {target}, matched {realized} " in msg
        assert "differs from the target" in msg and f"search bound {bound}" in msg
        assert abs(out["matched_rate"].iloc[0] - float(bound)) < 1e-9
        assert Path(w.filename).name == Path(__file__).name        # attributed to the caller


def _two_level(n_dep):
    """Each day's last hour scores 1.0 and the rest 0.0. Every P4 window then looks alike, so P4's
    alert count only jumps, from n_dep to 24 * n_dep, at one interior rate (about 4.2%): a target
    in between is unreachable."""
    s = _scores(n_dep=n_dep)
    s["score"] = (s["timestamp_utc"].dt.hour == 23).astype(float)
    return s


# The reachable counts are 100 and 2400, nothing between. The tolerance is max(3, 1% of the target).
@pytest.mark.parametrize("target,expect", [
    (103, None), (104, "differs"),       # 3 away is within the tolerance, 4 away is not
    (2380, None), (2370, "differs"),     # 1% is 23.8 (20 away is fine) and 23.7 (30 away is not)
    (100, "bound"),                      # count matched exactly, but the rate sits on the lower search bound
])
def test_p4_matched_flags_a_count_gap_beyond_max_3_or_one_percent_and_a_rate_on_a_bound(target, expect):
    from es.policies import p4_matched
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = p4_matched(_two_level(100), {("M", "A"): target}, window_days=14)
    msgs = [str(w.message) for w in rec if issubclass(w.category, UserWarning)]
    assert out["matched_count"].iloc[0] in (100, 2400)
    if expect is None:
        assert msgs == []
    else:
        (msg,) = msgs
        assert ("differs from the target" in msg) == (expect == "differs")
        assert ("search bound" in msg) == (expect == "bound")


# --- guards: empty P4 windows, target keys without a node, a negative T -------------------------

def _train_and_deploy_only(deploy_days, node="A"):
    """One node: a day of training rows and 24 deployment rows on each given day. No calib or gap rows."""
    def day_rows(day, split):
        return [{"market": "M", "node": node, "split": split, "settle_date": day,
                 "timestamp_utc": pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=h), "score": float(h)}
                for h in range(24)]
    return pd.DataFrame(day_rows(dt.date(2025, 1, 1), "train")
                        + [r for d in deploy_days for r in day_rows(d, "deploy")])


@pytest.mark.parametrize("days,window,n_empty,words", [
    # only the first day lacks history, the later days see the earlier deployment days
    ([dt.date(2025, 4, 1), dt.date(2025, 4, 2), dt.date(2025, 4, 3)], 14, 1, "1 day"),
    # the days are further apart than the window, so every day lacks history
    ([dt.date(2025, 4, 1), dt.date(2025, 4, 10), dt.date(2025, 4, 20)], 3, 3, "3 days"),
])
def test_p4_days_with_an_empty_trailing_window_are_named_in_the_one_warning(days, window, n_empty, words):
    from es.policies import p4_rolling
    s = pd.concat([_train_and_deploy_only(days), _train_and_deploy_only([], node="B")], ignore_index=True)
    with pytest.warns(UserWarning) as rec:
        out = p4_rolling(s, 0.05, window_days=window)
    (msg,) = [str(w.message) for w in rec]               # B is skipped and A has empty days: one warning
    assert msg.startswith("p4_rolling: 1 node(s) skipped: M/B (no deployment rows)")
    assert f"M/A ({words})" in msg and "empty trailing window" in msg
    empty = out["tau"].isna()
    assert empty.sum() == 24 * n_empty and not out.loc[empty, "alert"].any()
    assert set(out["node"]) == {"A"} and out["settle_date"].nunique() == len(days)


def test_p4_matched_names_empty_trailing_window_days_too():
    from es.policies import p4_matched
    s = _train_and_deploy_only([dt.date(2025, 4, 1), dt.date(2025, 4, 2), dt.date(2025, 4, 3)])
    with pytest.warns(UserWarning) as rec:
        p4_matched(s, {("M", "A"): 5}, window_days=14)
    msgs = [str(w.message) for w in rec]
    assert any(m.startswith("p4_matched: ") and "M/A (1 day)" in m and "empty trailing window" in m
               for m in msgs)


@pytest.mark.parametrize("name,call", [
    ("p4_matched", lambda P, s, targets: P.p4_matched(s, targets, window_days=14)),
    ("top_count", lambda P, s, targets: P.top_count(s, targets)),
])
def test_target_keys_that_match_no_node_are_named_in_the_warning(name, call):
    import es.policies as P
    with pytest.warns(UserWarning) as rec:
        out = call(P, _scores(), {("M", "A"): 100, ("M", "Z"): 5, ("X", "A"): 7})
    (msg,) = [str(w.message) for w in rec]
    assert msg.startswith(f"{name}: 2 target_counts key(s) match no node in the score frame: M/Z, X/A")
    assert "skipped" not in msg and set(out["node"]) == {"A"}
    assert Path(rec[0].filename).name == Path(__file__).name       # attributed to the caller


def test_a_mistyped_only_target_key_is_named_in_the_error():
    from es.policies import top_count
    with pytest.raises(ValueError, match=r"top_count: no node qualifies.*M/A \(not in target_counts\).*M/Z"):
        top_count(_scores(), {("M", "Z"): 5})


def test_top_count_rejects_a_negative_count():
    from es.policies import top_count
    s = _scores()
    with pytest.raises(ValueError, match=r"top_count.*non-negative.*M/A: -1"):
        top_count(s, {("M", "A"): -1})
    with pytest.raises(ValueError, match=r"top_count.*non-negative.*M/Z: -3"):  # even for a key with no node
        top_count(s, {("M", "A"): 10, ("M", "Z"): -3})
    with pytest.raises(ValueError) as exc:                                      # every offending key is named
        top_count(s, {("M", "A"): -1, ("M", "Z"): -3})
    assert "M/A: -1" in str(exc.value) and "M/Z: -3" in str(exc.value)
    assert int(top_count(s, {("M", "A"): 0})["alert"].sum()) == 0               # zero is a valid T
