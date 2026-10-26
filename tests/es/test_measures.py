import datetime as dt
import inspect
import warnings

import numpy as np
import pandas as pd
import pytest


def _rows(alert, pnl, deciles=None, days=None, hours=None, **over):
    """The brief's helper, extended with `hours` and column overrides (a superset of its signature)."""
    n = len(pnl)
    df = pd.DataFrame({
        "market": "M", "node": "A",
        "settle_date": days or [dt.date(2025, 7, 1 + i // 2) for i in range(n)],
        "settle_hour": hours if hours is not None else [i % 2 for i in range(n)], "wclass": 0, "month": 7,
        "cal_decile": deciles if deciles is not None else [i % 2 for i in range(n)],
        "alert": np.asarray(alert, bool), "pnl_open": np.asarray(pnl, float),
        "dart_settle": np.asarray(pnl, float)})
    return df.assign(**over)


def _summary_rows():
    # One node, 8 hours on 4 days. Day 1: rows 0, 1 (-10 alerted, +2); day 2: rows 2, 3 (-6 alerted, +3);
    # day 3: rows 4, 5 (-1, +4 alerted); day 4: rows 6, 7 (+5, -2).
    # Hours 0, 1, 0, 1, ...; deciles 0, 0, 1, 1, 0, 0, 1, 1.
    return _rows([1, 0, 1, 0, 0, 1, 0, 0], [-10, 2, -6, 3, -1, 4, 5, -2], deciles=[0, 0, 1, 1, 0, 0, 1, 1])


def test_la_pg_net():
    from es.measures import la_pg_net
    assert la_pg_net(_rows([1, 1, 0, 0], [-10, 4, -1, 2])) == {"LA": 10.0, "PG": 4.0, "NET": 6.0}


def test_selc_zero_for_within_stratum_random_expectation():
    from es.measures import STRATA_DECILE, selc
    r = _rows([1, 0, 1, 0], [-10, -10, -10, -10], deciles=[0, 0, 1, 1])
    assert np.isclose(selc(r, STRATA_DECILE), 0.0)


def test_selc_positive_when_alerts_pick_losses():
    from es.measures import STRATA_DECILE, selc
    r = _rows([1, 0, 1, 0], [-10, 0, -10, 0], deciles=[0, 0, 1, 1])
    assert np.isclose(selc(r, STRATA_DECILE), 0.5)


def test_selc_primary_and_decile_strata():
    from es.measures import STRATA_DECILE, STRATA_PRIMARY, selc
    r = _summary_rows()                        # alerts on rows 0, 2, 5; total loss 19
    # Primary strata (Ruling 68: hour, weekday class, month). Hour 0 = rows 0, 2, 4, 6: losses 10, 6, 1, 0; alerted loss
    # 16, expected 17 * 2/4 = 8.5, excess 7.5. Hour 1 = rows 1, 3, 5, 7: losses 0, 0, 0, 2; alerted loss 0, expected
    # 2 * 1/4 = 0.5, excess -0.5.
    assert np.isclose(selc(r), 7.0 / 19) and np.isclose(selc(r, STRATA_PRIMARY), 7.0 / 19)
    # Decile strata (the sensitivity). Decile 0 = rows 0, 1, 4, 5: losses 10, 0, 1, 0; alerted loss 10, expected
    # 11 * 2/4 = 5.5, excess 4.5. Decile 1 = rows 2, 3, 6, 7: losses 6, 0, 0, 2; alerted loss 6, expected
    # 8 * 1/4 = 2, excess 4.
    assert np.isclose(selc(r, STRATA_DECILE), 8.5 / 19)


def test_snet_is_per_mw_day():
    from es.measures import STRATA_DECILE, snet
    r = _rows([1, 0, 1, 0], [-10, 0, -10, 0], deciles=[0, 0, 1, 1])   # 1 node, 2 days
    assert np.isclose(snet(r, STRATA_DECILE), 10.0 / (1 * 2))          # excess 10 $ over 2 MW-days


def test_add_calendar_deciles_equal_size():
    from es.measures import add_calendar_deciles
    ts = pd.date_range("2025-07-01", periods=100, freq="h", tz="UTC")
    rows = pd.DataFrame({"market": "M", "node": "A", "timestamp_utc": ts,
                         "settle_date": ts.date, "alert": False, "pnl_open": 0.0})
    g6 = pd.DataFrame({"market": "M", "node": "A", "timestamp_utc": ts, "score": np.arange(100.0)})
    out = add_calendar_deciles(rows, g6)
    assert out["cal_decile"].value_counts().tolist() == [10] * 10
    assert out.loc[out["g6_score"] >= 90, "cal_decile"].eq(9).all()


def test_drop_partial_edge_days():
    from es.measures import drop_partial_edge_days
    days = [dt.date(2025, 7, 1)] * 5 + [dt.date(2025, 7, 2)] * 24 + [dt.date(2025, 7, 3)] * 24
    r = pd.DataFrame({"market": "M", "node": "A", "settle_date": days})
    out = drop_partial_edge_days(r)
    assert set(out["settle_date"]) == {dt.date(2025, 7, 2), dt.date(2025, 7, 3)}


def test_expected_shortfall_and_esred():
    from es.measures import daily_portfolio, esred, expected_shortfall
    r = _rows([1, 0] * 10, [-50, 1] * 10)
    assert np.isclose(expected_shortfall(daily_portfolio(r, with_alerts=False)), 49.0)
    assert np.isclose(expected_shortfall(daily_portfolio(r, with_alerts=True)), -1.0)
    assert np.isclose(esred(r), 50.0)


def test_j_lambda_and_decomposition_add_up():
    from es.measures import decomposition, j_lambda
    assert j_lambda(net=100.0, esred_value=2.0, n_days=10, lam=0.5) == 110.0
    d = decomposition({"NET": 5.0, "ESred": 3.0}, {"NET": 2.0, "ESred": 1.0}, {"NET": 1.0, "ESred": 0.5},
                      keys=("NET", "ESred"))
    for k in ("NET", "ESred"):
        assert np.isclose(d[k]["timing"] + d[k]["level"], d[k]["total"])


def test_cap_excess():
    from es.measures import cap
    out = cap(_rows([1, 0, 0, 0], [-30, 10, 10, 10]))
    assert np.isclose(out["CAP"], 0.5) and np.isclose(out["excess"], 0.25)


# ====================================================================================================
# Supplementary tests (fix round). Each closes a gap that a mutation pass found in the tests above.
# Expected values are worked out by hand in the comments, not read back from the implementation.
# ====================================================================================================
D0 = dt.date(2025, 7, 1)


def day(i):
    return D0 + dt.timedelta(days=i)


def _hours(counts, node="A", market="M", first=D0):
    return pd.DataFrame([(market, node, first + dt.timedelta(days=i), h)
                         for i, c in enumerate(counts) for h in range(c)],
                        columns=["market", "node", "settle_date", "settle_hour"])


def _scored(n_rows, nodes=("A",), offsets=(0.0,), start="2025-07-01", score=None):
    ts = pd.date_range(start, periods=n_rows, freq="h", tz="UTC")
    rows = pd.concat([pd.DataFrame({"market": "M", "node": node, "timestamp_utc": ts, "settle_date": ts.date,
                                    "alert": False, "pnl_open": 0.0}) for node in nodes], ignore_index=True)
    g6 = pd.concat([pd.DataFrame({"market": "M", "node": node, "timestamp_utc": ts,
                                  "score": (np.arange(n_rows, dtype=float) if score is None else score) + off})
                    for node, off in zip(nodes, offsets)], ignore_index=True)
    return rows, g6


# ---------------------------------------------------------------- SELC and its strata
def test_selc_expectation_is_the_alert_rate_not_its_complement():
    from es.measures import STRATA_DECILE, STRATA_PRIMARY, selc
    # one stratum, 1 of 4 hours alerted, on the only loss: captured 8, expected 8 * 1/4 = 2 -> (8 - 2) / 8
    one = _rows([1, 0, 0, 0], [-8, 0, 0, 0], deciles=[0, 0, 0, 0], hours=[0, 0, 0, 0])
    assert np.isclose(selc(one, STRATA_DECILE), 0.75) and np.isclose(selc(one, STRATA_PRIMARY), 0.75)


@pytest.mark.parametrize("which", ["STRATA_PRIMARY", "STRATA_DECILE"])
def test_selc_strata_do_not_pool_nodes_or_markets(which):
    from es import measures
    strata = getattr(measures, which)
    pick = _rows([1, 0], [-10, -10], deciles=[0, 0], hours=[0, 0])    # one of two equal losses alerted: 0 excess
    idle = _rows([0, 0], [0, 0], deciles=[0, 0], hours=[0, 0])        # a second cell with no loss and no alert
    # pooled over the stratum the pair would show 10 - 20/4 = 5 of excess, i.e. SELC 0.25, not 0
    assert np.isclose(measures.selc(pd.concat([pick, idle.assign(node="B")], ignore_index=True), strata), 0.0)
    assert np.isclose(measures.selc(pd.concat([pick, idle.assign(market="N")], ignore_index=True), strata), 0.0)


def test_selc_primary_strata_separate_weekday_class_and_month():
    from es.measures import STRATA_PRIMARY, selc
    pick = _rows([1, 0], [-10, -10], hours=[0, 0])
    idle = _rows([0, 0], [0, 0], hours=[0, 0])
    assert np.isclose(selc(pd.concat([pick, idle], ignore_index=True)), 0.25)     # same cell: pooled, 5 of excess
    assert np.isclose(selc(pd.concat([pick, idle.assign(wclass=1)], ignore_index=True), STRATA_PRIMARY), 0.0)
    assert np.isclose(selc(pd.concat([pick, idle.assign(month=8)], ignore_index=True), STRATA_PRIMARY), 0.0)
    assert np.isclose(selc(pd.concat([pick, idle.assign(settle_hour=1)], ignore_index=True), STRATA_PRIMARY), 0.0)


# ---------------------------------------------------------------- SNET
def test_snet_counts_profit_given_up_as_well_as_loss_avoided():
    from es.measures import STRATA_DECILE, snet
    # one stratum; value = -pnl = [10, -6, 0, 0]; alerted rows 0 and 1 hold 4 of the stratum total 4;
    # expected = 4 * 2/4 = 2; excess = 2; over 2 days = 1.0   (loss-only would give 5 / 2 = 2.5)
    assert np.isclose(snet(_rows([1, 1, 0, 0], [-10, 6, 0, 0], deciles=[0, 0, 0, 0]), STRATA_DECILE), 1.0)


def test_snet_is_per_node_day_across_nodes_and_markets():
    from es.measures import STRATA_DECILE, snet
    one = _rows([1, 0, 1, 0], [-10, 0, -10, 0], deciles=[0, 0, 1, 1])           # 10 $ of excess, 1 node x 2 days
    three = pd.concat([one, one.assign(node="B"), one.assign(market="N")], ignore_index=True)
    assert np.isclose(snet(one, STRATA_DECILE), 5.0)
    assert np.isclose(snet(three, STRATA_DECILE), 5.0)   # 30 $ over 6 node-days (not 2 days = 15, nor 4 market-days)


def test_snet_divides_by_the_node_days_of_an_unbalanced_panel():
    from es.measures import STRATA_DECILE, snet
    a = _rows([1, 0, 1, 0], [-10, 0, -10, 0], deciles=[0, 0, 1, 1])          # node A: 2 days, excess 5 + 5
    b = _rows([1, 0], [-20, 0], deciles=[0, 0]).assign(node="B")             # node B: 1 day, excess 20 - 20 * 1/2
    assert np.isclose(snet(a, STRATA_DECILE), 10 / 2) and np.isclose(snet(b, STRATA_DECILE), 10 / 1)
    # 20 $ of excess over 2 + 1 = 3 node-days. Not over nodes x global days = 2 x 2 = 4 (5.0), nor over the longest
    # node's days x nodes = 2 x 2 = 4 (5.0), nor over the 2 distinct days alone (10.0).
    assert np.isclose(snet(pd.concat([a, b], ignore_index=True), STRATA_DECILE), 20 / 3)


# ---------------------------------------------------------------- expected shortfall, CAP, decomposition
def test_expected_shortfall_is_the_mean_of_the_k_worst_days_as_a_positive_loss():
    from es.measures import expected_shortfall
    daily = pd.Series(np.arange(41.0) - 10)                     # -10 ... 30, D = 41, k = ceil(2.05) = 3
    assert np.isclose(expected_shortfall(daily), 9.0)           # worst three: -10, -9, -8
    assert np.isclose(expected_shortfall(daily.sample(frac=1.0, random_state=0)), 9.0)
    assert np.isclose(expected_shortfall(pd.Series(np.arange(21.0))), -0.5)     # D = 21, k = 2: mean(0, 1) = 0.5
    assert np.isclose(expected_shortfall(pd.Series(np.arange(20.0))), 0.0)      # D = 20, k = 1: worst day = 0
    assert np.isclose(expected_shortfall(pd.Series([-3.0])), 3.0)               # D = 1, k = 1
    assert np.isclose(expected_shortfall(daily, frac=0.1), 8.0)                 # k = ceil(4.1) = 5: mean(-10..-6) = -8


def test_daily_portfolio_pools_nodes_by_settlement_date():
    from es.measures import daily_portfolio
    a = _rows([1, 0, 0, 0], [-4, 1, 2, 3])                      # days Jul 1, Jul 2
    both = pd.concat([a, a.assign(node="B", market="N")], ignore_index=True)
    assert daily_portfolio(both, with_alerts=False).to_dict() == {dt.date(2025, 7, 1): -6.0, dt.date(2025, 7, 2): 10.0}
    assert daily_portfolio(both, with_alerts=True).to_dict() == {dt.date(2025, 7, 1): 2.0, dt.date(2025, 7, 2): 10.0}


def test_cap_is_the_alerted_share_of_absolute_dart_and_ignores_the_position():
    from es.measures import cap
    r = _rows([1, 0, 0, 0], [1, 1, 1, 1]).assign(dart_settle=[-45.0, 5.0, 5.0, 5.0])
    out = cap(r)
    assert np.isclose(out["CAP"], 0.75) and np.isclose(out["excess"], 0.5)      # 45/60, minus the alert rate 0.25


def test_zero_denominator_conventions_are_pinned():
    from es.measures import cap, selc
    no_loss = _rows([1, 0, 1, 0], [5, 3, 2, 1], deciles=[0, 0, 1, 1])
    assert selc(no_loss) == 0.0                        # no loss at all (sum of L_s = 0): SELC is 0.0 by convention
    zero_dart = _rows([1, 0], [-5, 3]).assign(dart_settle=[0.0, 0.0])
    assert cap(zero_dart) == {"CAP": 0.0, "excess": -0.5}   # no |DART| at all: CAP is 0.0, so excess = -alert rate


def test_decomposition_timing_is_p1_minus_matched_and_level_is_matched_minus_alpha():
    from es.measures import decomposition
    d = decomposition({"NET": 5.0}, {"NET": 2.0}, {"NET": 1.0}, keys=("NET",))
    assert d == {"NET": {"timing": 3.0, "level": 1.0, "total": 4.0}}


# ---------------------------------------------------------------- drop_partial_edge_days
def test_drop_partial_edge_days_keeps_23_and_25_hour_edge_days():
    from es.measures import drop_partial_edge_days
    assert sorted(set(drop_partial_edge_days(_hours([23, 24, 23]))["settle_date"])) == [day(0), day(1), day(2)]
    assert sorted(set(drop_partial_edge_days(_hours([25, 24, 25]))["settle_date"])) == [day(0), day(1), day(2)]
    assert sorted(set(drop_partial_edge_days(_hours([22, 24, 24, 23]))["settle_date"])) == [day(1), day(2), day(3)]
    assert sorted(set(drop_partial_edge_days(_hours([23, 24, 23]), min_hours=24)["settle_date"])) == [day(1)]


def test_drop_partial_edge_days_drops_a_partial_last_day_but_not_an_interior_one():
    from es.measures import drop_partial_edge_days
    assert sorted(set(drop_partial_edge_days(_hours([24, 24, 5]))["settle_date"])) == [day(0), day(1)]
    assert sorted(set(drop_partial_edge_days(_hours([24, 10, 24]))["settle_date"])) == [day(0), day(1), day(2)]


def test_drop_partial_edge_days_judges_each_node_on_its_own_edges():
    from es.measures import drop_partial_edge_days
    both = pd.concat([_hours([5, 24, 24], "A"), _hours([24, 24, 3], "B"), _hours([12, 24, 24], "C", first=day(1))],
                     ignore_index=True)
    out = drop_partial_edge_days(both)
    assert {n: sorted(set(g["settle_date"])) for n, g in out.groupby("node")} == {
        "A": [day(1), day(2)], "B": [day(0), day(1)], "C": [day(2), day(3)]}


def test_drop_partial_edge_days_keeps_the_other_columns_and_resets_the_index():
    from es.measures import drop_partial_edge_days
    f = _hours([5, 24, 24]).assign(x=np.arange(53), ts=pd.date_range("2025-07-01", periods=53, freq="h", tz="UTC"))
    shuffled = f.sample(frac=1.0, random_state=1)
    out = drop_partial_edge_days(shuffled)
    assert list(out.index) == list(range(len(out))) and len(out) == 48
    assert list(out["x"]) == [x for x in shuffled["x"] if x >= 5]               # survivors keep their order
    assert str(out["ts"].dtype) == "datetime64[ns, UTC]" and out["x"].dtype.kind == "i"
    clean = _hours([24, 24, 24])
    assert drop_partial_edge_days(clean).equals(clean)                          # nothing to drop: unchanged


# ---------------------------------------------------------------- add_calendar_deciles
def test_calendar_deciles_are_per_market_and_node():
    from es.measures import add_calendar_deciles
    rows, g6 = _scored(100, nodes=("A", "B"), offsets=(0.0, 1000.0))            # every B score above every A score
    out = add_calendar_deciles(rows, g6)
    assert (out.groupby("node")["cal_decile"].value_counts() == 10).all()
    # the same node names in a second market, with scores far above the first market's: still ten per decile each
    both = add_calendar_deciles(pd.concat([rows, rows.assign(market="X")], ignore_index=True),
                                pd.concat([g6, g6.assign(market="X", score=g6["score"] + 5000.0)], ignore_index=True))
    assert len(both) == 400 and (both.groupby(["market", "node"])["cal_decile"].value_counts() == 10).all()


def test_calendar_deciles_give_each_tied_score_one_stratum():
    from es.measures import add_calendar_deciles
    rows, g6 = _scored(100, score=(np.arange(100) % 5).astype(float))           # five values x 20 rows
    out = add_calendar_deciles(rows, g6)
    assert (out.groupby("g6_score")["cal_decile"].nunique() == 1).all()         # equal score, equal decile
    assert out["cal_decile"].value_counts().tolist() == [20] * 5                # five strata, not ten equal ones
    assert out.sort_values("g6_score")["cal_decile"].is_monotonic_increasing


def test_calendar_deciles_add_the_settlement_month_and_keep_row_order():
    from es.measures import add_calendar_deciles
    rows, g6 = _scored(24, start="2025-06-30 12:00")                            # 12 rows on Jun 30, 12 on Jul 1
    out = add_calendar_deciles(rows, g6)
    assert out["month"].tolist() == [6] * 12 + [7] * 12
    assert out["timestamp_utc"].tolist() == rows["timestamp_utc"].tolist()


def test_calendar_deciles_refuse_rows_without_a_score():
    from es.measures import add_calendar_deciles
    rows, g6 = _scored(100)
    with pytest.raises(ValueError, match="calendar-model score"):
        add_calendar_deciles(rows, g6.iloc[:-1])


# ---------------------------------------------------------------- summarize (hand-worked on 8 rows, 4 days)
def test_summarize_values_are_hand_worked():
    from es.measures import STRATA_DECILE, snet, summarize
    s = summarize(_summary_rows())
    assert list(s) == ["LA", "PG", "NET", "SELC", "SELC_decile", "SNET", "ESred", "ES5_without", "CAP", "excess",
                       "n_days", "alert_rate", "J_0.0", "J_0.25", "J_0.5", "J_1.0"]
    assert "SELC_month" not in s                    # it is SELC now (Ruling 68)
    assert (s["LA"], s["PG"], s["NET"]) == (16.0, 4.0, 12.0)
    assert np.isclose(s["SELC"], 7.0 / 19)          # primary strata: hour 0: 16 - 17/2 = 7.5; hour 1: 0 - 2/4 = -0.5
    assert np.isclose(s["SELC_decile"], 8.5 / 19)   # decile 0: 10 - 11/2 = 4.5; decile 1: 6 - 8/4 = 4; losses 19
    # SNET, primary: value = -pnl = 10, -2, 6, -3, 1, -4, -5, 2. Hour 0 = rows 0, 2, 4, 6 (value 12, alerted 10 + 6):
    # 16 - 12 * 2/4 = 10. Hour 1 = rows 1, 3, 5, 7 (value -7, alerted -4): -4 + 7 * 1/4 = -2.25. 7.75 over 4 node-days.
    assert np.isclose(s["SNET"], 7.75 / 4)
    # and under the decile strata, as it was: decile 0: 6 - 5/2 = 3.5; decile 1: 6 - 0 = 6; over 4 node-days
    assert np.isclose(snet(_summary_rows(), STRATA_DECILE), 9.5 / 4)
    assert np.isclose(s["ES5_without"], 8.0)        # daily -8, -3, 3, 3; k = 1
    assert np.isclose(s["ESred"], 8.0 - 1.0)        # with alerts: 2, 3, -1, 3
    assert np.isclose(s["CAP"], 20 / 33) and np.isclose(s["excess"], 20 / 33 - 3 / 8)
    assert s["n_days"] == 4 and np.isclose(s["alert_rate"], 3 / 8)
    assert [s[k] for k in ("J_0.0", "J_0.25", "J_0.5", "J_1.0")] == [12.0, 19.0, 26.0, 40.0]    # 12 + lam * 4 * 7


def test_summarize_uses_the_strata_it_is_given_and_the_decile_sensitivity_only_when_cal_decile_is_present():
    from es.measures import STRATA_DECILE, STRATA_PRIMARY, selc, snet, summarize
    r = _summary_rows()
    assert not np.isclose(selc(r, STRATA_PRIMARY), selc(r, STRATA_DECILE))
    assert not np.isclose(snet(r, STRATA_PRIMARY), snet(r, STRATA_DECILE))
    t = summarize(r, strata=STRATA_DECILE)
    assert np.isclose(t["SELC"], selc(r, STRATA_DECILE)) and np.isclose(t["SNET"], snet(r, STRATA_DECILE))
    assert np.isclose(t["SELC_decile"], t["SELC"])           # named strata or not, the sensitivity is the deciles
    default = summarize(r)
    assert np.isclose(default["SELC"], selc(r)) and np.isclose(default["SNET"], snet(r))
    assert np.isclose(default["SELC_decile"], selc(r, STRATA_DECILE))
    assert "SELC_decile" not in summarize(r.drop(columns="cal_decile"))
    assert summarize(r, lams=(2.0,))["J_2.0"] == 12.0 + 2.0 * 4 * 7


# ====================================================================================================
# F2. Calendar deciles are taken by value: tied scores share a decile, whatever the order of the rows.
# ====================================================================================================
def test_calendar_deciles_tied_scores_straddling_a_decile_boundary_share_a_decile():
    from es.measures import add_calendar_deciles
    score = np.arange(100.0)
    score[5:15] = 5.0                       # ten rows tie; by rank they would straddle the decile 0 / 1 boundary
    rows, g6 = _scored(100, score=score)
    out = add_calendar_deciles(rows, g6)
    tied = out.loc[out["g6_score"] == 5.0, "cal_decile"]
    assert len(tied) == 10 and tied.nunique() == 1
    # sorted: 0 .. 4, ten 5s, 15 .. 99. The first quantile edge falls inside the tie (value 5), so decile 0 holds
    # the five values below it and all ten tied rows: 15 rows, not 10.
    assert (out["cal_decile"] == 0).sum() == 15 and tied.iloc[0] == 0
    assert out.sort_values("g6_score")["cal_decile"].is_monotonic_increasing


def test_calendar_deciles_do_not_depend_on_the_order_of_the_rows():
    from es.measures import add_calendar_deciles
    rows, g6 = _scored(100, nodes=("A", "B"), offsets=(0.0, 3.0), score=(np.arange(100) % 7).astype(float))
    key = ["market", "node", "timestamp_utc"]
    base = add_calendar_deciles(rows, g6).set_index(key)["cal_decile"]
    shuffled = add_calendar_deciles(rows.sample(frac=1.0, random_state=3), g6.sample(frac=1.0, random_state=4))
    again = shuffled.set_index(key)["cal_decile"]
    assert len(again) == len(base) and (again.loc[base.index] == base).all()


def test_calendar_deciles_single_row_and_all_equal_nodes_get_decile_zero():
    from es.measures import add_calendar_deciles
    ts = pd.date_range("2025-07-01", periods=50, freq="h", tz="UTC")

    def node_rows(node, n):
        return pd.DataFrame({"market": "M", "node": node, "timestamp_utc": ts[:n], "settle_date": ts[:n].date,
                             "alert": False, "pnl_open": 0.0})

    rows = pd.concat([node_rows("ONE", 1), node_rows("FLAT", 30), node_rows("SPREAD", 50)], ignore_index=True)
    g6 = rows[["market", "node", "timestamp_utc"]].assign(score=np.r_[[4.2], np.full(30, 7.0), np.arange(50.0)])
    out = add_calendar_deciles(rows, g6)
    assert out.loc[out["node"] == "ONE", "cal_decile"].tolist() == [0]
    assert out.loc[out["node"] == "FLAT", "cal_decile"].eq(0).all()
    assert out.loc[out["node"] == "SPREAD", "cal_decile"].value_counts().tolist() == [5] * 10    # unaffected
    assert out["cal_decile"].dtype.kind == "i"


@pytest.mark.parametrize("n", [91, 171, 181, 331, 341, 361])
def test_distinct_scores_keep_the_deciles_of_the_order_based_assignment(n):
    # The first version took pd.qcut of the rank by order. With distinct scores the by-value rule must give the very
    # same deciles, also when n - 1 is divisible by 10 and a quantile edge falls on an order statistic: qcut on the
    # raw scores moves a row across a boundary there (floating-point interpolation), qcut on the ranks does not.
    from es.measures import add_calendar_deciles
    rng = np.random.default_rng(n)
    rows, g6 = _scored(n, score=1e6 + rng.permutation(n) * 1e-6)
    expected = pd.qcut(g6["score"].rank(method="first"), 10, labels=False)
    assert add_calendar_deciles(rows, g6)["cal_decile"].tolist() == expected.tolist()


@pytest.mark.parametrize("seed", range(6))
def test_calendar_deciles_on_tie_heavy_scores_are_valid_strata(seed):
    from es.measures import add_calendar_deciles
    rng = np.random.default_rng(seed)
    n, k = int(rng.integers(30, 400)), int(rng.integers(2, 25))
    rows, g6 = _scored(n, score=rng.integers(0, k, n).astype(float))
    out = add_calendar_deciles(rows, g6)
    assert out["cal_decile"].between(0, 9).all()
    assert (out.groupby("g6_score")["cal_decile"].nunique() == 1).all()         # equal score, equal decile
    assert out.sort_values("g6_score")["cal_decile"].is_monotonic_increasing    # higher score, no lower decile


# ====================================================================================================
# F3. The G6 merge is many-to-one, and the new columns are never overwritten.
# ====================================================================================================
def test_calendar_deciles_accept_a_key_repeated_in_rows_and_give_the_repeats_the_same_decile():
    from es.measures import add_calendar_deciles
    rows, g6 = _scored(100)
    out = add_calendar_deciles(pd.concat([rows, rows.iloc[:5]], ignore_index=True), g6)       # many rows, one G6 score
    assert len(out) == 105
    assert out["cal_decile"].iloc[100:].tolist() == out["cal_decile"].iloc[:5].tolist()


def test_calendar_deciles_refuse_duplicate_keys_in_the_g6_frame():
    from es.measures import add_calendar_deciles
    rows, g6 = _scored(100)
    with pytest.raises(pd.errors.MergeError):
        add_calendar_deciles(rows, pd.concat([g6, g6.iloc[[0]]], ignore_index=True))


def test_calendar_deciles_refuse_rows_that_already_have_g6_score():
    from es.measures import add_calendar_deciles
    rows, g6 = _scored(100)
    with pytest.raises(ValueError, match="g6_score"):
        add_calendar_deciles(rows.assign(g6_score=0.0), g6)


def test_calendar_deciles_refuse_rows_that_already_have_cal_decile():
    from es.measures import add_calendar_deciles
    rows, g6 = _scored(100)
    with pytest.raises(ValueError, match="cal_decile"):
        add_calendar_deciles(rows.assign(cal_decile=0), g6)


def test_calendar_deciles_refuse_rows_that_already_have_month():
    from es.measures import add_calendar_deciles
    rows, g6 = _scored(100)
    with pytest.raises(ValueError, match="month"):
        add_calendar_deciles(rows.assign(month=7), g6)          # the settlement month is derived, never supplied


# ====================================================================================================
# F4. Nothing is dropped silently: an empty frame, a missing column or a NaN raises and names the column.
# ====================================================================================================
#: The columns each public measure reads (Ruling 68: the primary strata are hour, weekday class and month, so the
#: decile column is read only by `summarize`, and only when it is present: it then adds SELC_decile).
READS = {
    "la_pg_net": ["alert", "pnl_open"],
    "selc": ["alert", "pnl_open", "market", "node", "settle_hour", "wclass", "month"],
    "snet": ["alert", "pnl_open", "settle_date", "market", "node", "settle_hour", "wclass", "month"],
    "daily_portfolio[alerts]": ["pnl_open", "settle_date", "alert"],
    "daily_portfolio[open]": ["pnl_open", "settle_date"],
    "esred": ["alert", "pnl_open", "settle_date"],
    "cap": ["dart_settle", "alert"],
    "summarize": ["alert", "pnl_open", "dart_settle", "settle_date", "market", "node", "settle_hour", "wclass",
                  "month", "cal_decile"],
}
OPTIONAL = {"summarize": {"cal_decile"}}
NAN_CASES = [(name, col) for name, cols in READS.items() for col in cols]
MISSING_CASES = [(name, col) for name, col in NAN_CASES if col not in OPTIONAL.get(name, ())]


def _call(name, rows):
    from es import measures
    if name.startswith("daily_portfolio"):
        return measures.daily_portfolio(rows, with_alerts=name.endswith("[alerts]"))
    return getattr(measures, name)(rows)


def _with_nan(rows, column):
    values = rows[column].astype(object)
    values.iloc[0] = np.nan
    return rows.assign(**{column: values})


def _caller(name):
    return name.split("[")[0]


# The messages start with the measure's own name, so each public measure is shown to run its own check (a check
# that only a function it calls ran would name that function instead).
@pytest.mark.parametrize("name,column", NAN_CASES)
def test_a_nan_in_a_column_a_measure_reads_raises_and_names_the_column(name, column):
    with pytest.raises(ValueError, match=rf"^{_caller(name)}: column '{column}' has 1 NaN"):
        _call(name, _with_nan(_summary_rows(), column))


@pytest.mark.parametrize("name,column", MISSING_CASES)
def test_a_missing_column_a_measure_reads_raises_and_names_the_column(name, column):
    with pytest.raises(ValueError, match=rf"^{_caller(name)}: column '{column}' is missing"):
        _call(name, _summary_rows().drop(columns=column))


@pytest.mark.parametrize("name", list(READS))
def test_an_empty_frame_raises(name):
    with pytest.raises(ValueError, match=rf"^{_caller(name)}: the frame is empty"):
        _call(name, _summary_rows().iloc[0:0])


@pytest.mark.parametrize("name", list(READS))
def test_a_one_row_frame_is_not_empty(name):
    _call(name, _summary_rows().iloc[:1])


@pytest.mark.parametrize("name", [n for n in READS if n != "summarize"])
def test_a_nan_in_a_column_a_measure_does_not_read_is_ignored(name):
    base = _summary_rows()
    for column in sorted(set(base.columns) - set(READS[name])):
        _call(name, _with_nan(base, column))


@pytest.mark.parametrize("column", ["settle_hour", "wclass", "month"])
def test_a_nan_in_a_primary_strata_column_raises(column):
    from es.measures import selc, snet, summarize
    for measure in (selc, snet, summarize):
        with pytest.raises(ValueError, match=column):
            measure(_with_nan(_summary_rows(), column))


def test_a_nan_in_the_decile_column_raises_where_the_deciles_are_read_and_is_ignored_where_they_are_not():
    from es.measures import STRATA_DECILE, selc, snet, summarize
    r = _summary_rows().assign(cal_decile=[0.0, np.nan, 1, 1, 0, 0, 1, 1])
    for measure in (selc, snet):
        with pytest.raises(ValueError, match="cal_decile"):
            measure(r, STRATA_DECILE)
        measure(r)                                  # the primary strata do not read it
    with pytest.raises(ValueError, match="cal_decile"):
        summarize(r)                                # SELC_decile does


def test_snet_and_summarize_check_market_and_node_even_when_the_strata_do_not_include_them():
    from es.measures import snet, summarize
    r = _summary_rows()                                 # strata=["cal_decile"] names neither market nor node
    for measure in (snet, summarize):
        name = measure.__name__
        with pytest.raises(ValueError, match=rf"^{name}: column 'market' has 1 NaN"):
            measure(_with_nan(r, "market"), strata=["cal_decile"])
        with pytest.raises(ValueError, match=rf"^{name}: column 'node' is missing"):
            measure(r.drop(columns="node"), strata=["cal_decile"])


def test_a_nan_pnl_on_an_alerted_row_raises_instead_of_being_skipped():
    from es.measures import esred, la_pg_net, selc, snet, summarize
    r = _summary_rows()
    r.loc[0, "pnl_open"] = np.nan                           # row 0 is alerted
    assert r.loc[0, "alert"]
    for measure in (la_pg_net, selc, snet, esred, summarize):
        with pytest.raises(ValueError, match="pnl_open"):
            measure(r)


def test_an_empty_frame_raises_for_every_measure():
    from es.measures import cap, esred, la_pg_net, selc, snet, summarize
    empty = _summary_rows().iloc[0:0]
    for measure in (la_pg_net, selc, snet, esred, cap, summarize):
        with pytest.raises(ValueError, match="empty"):
            measure(empty)


def test_summarize_without_the_decile_column_adds_no_decile_sensitivity_and_ignores_nan_in_unread_columns():
    from es.measures import summarize
    r = _summary_rows().drop(columns="cal_decile")          # no SELC_decile, and nothing reads a decile
    s = summarize(r)
    assert "SELC_decile" not in s and np.isclose(s["SELC"], 7.0 / 19)
    extra = r.assign(g6_score=np.nan)                       # a column nobody reads may hold NaN
    assert summarize(extra) == s


def test_expected_shortfall_of_no_days_raises():
    from es.measures import expected_shortfall
    for empty in (pd.Series([], dtype=float), [], np.array([])):
        with pytest.raises(ValueError, match="no daily"):
            expected_shortfall(empty)


def test_expected_shortfall_rejects_a_nan_day_instead_of_ranking_it_last():
    from es.measures import expected_shortfall
    for bad in (pd.Series([1.0, np.nan, -3.0]), [np.nan], np.array([0.5, np.nan, 2.0])):
        with pytest.raises(ValueError, match="NaN"):
            expected_shortfall(bad)


# ====================================================================================================
# F5 and F6. J keys have float formatting; decomposition covers the E1 estimand (NET, ESred and every J).
# ====================================================================================================
def test_summarize_names_its_j_keys_with_float_formatting():
    from es.measures import summarize
    r = _summary_rows()
    assert [k for k in summarize(r) if k.startswith("J_")] == ["J_0.0", "J_0.25", "J_0.5", "J_1.0"]
    assert [k for k in summarize(r, lams=(0, 1)) if k.startswith("J_")] == ["J_0.0", "J_1.0"]


def test_j_key_is_the_one_name_of_a_j_column():
    from es.measures import E1_KEYS, LAMS, j_key, summarize
    assert [j_key(x) for x in (0, 1, 0.25, 0.5, np.float64(2.0), np.int64(3))] == [
        "J_0.0", "J_1.0", "J_0.25", "J_0.5", "J_2.0", "J_3.0"]
    assert E1_KEYS == ("NET", "ESred") + tuple(j_key(lam) for lam in LAMS)
    s = summarize(_summary_rows(), lams=(2, 0.1))
    assert [k for k in s if k.startswith("J_")] == [j_key(2), j_key(0.1)] == ["J_2.0", "J_0.1"]


def test_lams_and_e1_keys_and_the_defaults_that_use_them():
    from es.measures import E1_KEYS, LAMS, decomposition, summarize
    assert LAMS == (0.0, 0.25, 0.5, 1.0)
    assert E1_KEYS == ("NET", "ESred", "J_0.0", "J_0.25", "J_0.5", "J_1.0")
    assert inspect.signature(summarize).parameters["lams"].default == LAMS
    assert inspect.signature(decomposition).parameters["keys"].default == E1_KEYS


def test_decomposition_of_summaries_returns_exactly_the_e1_keys_and_each_part_is_additive():
    from es.measures import E1_KEYS, decomposition, summarize
    p1 = summarize(_summary_rows())
    matched = summarize(_summary_rows().assign(alert=[True, False, False, False, False, True, False, False]))
    alpha = summarize(_summary_rows().assign(alert=[False, True, False, True, True, False, False, True]))
    assert len({p1["J_0.5"], matched["J_0.5"], alpha["J_0.5"]}) == 3          # three different runs
    d = decomposition(p1, matched, alpha)
    assert tuple(d) == E1_KEYS
    for k in E1_KEYS:
        assert set(d[k]) == {"timing", "level", "total"}
        assert np.isclose(d[k]["timing"], p1[k] - matched[k])
        assert np.isclose(d[k]["level"], matched[k] - alpha[k])
        assert np.isclose(d[k]["total"], p1[k] - alpha[k])
        assert np.isclose(d[k]["timing"] + d[k]["level"], d[k]["total"])


# ====================================================================================================
# Categorical key columns give the numbers of object columns and no pandas FutureWarning (observed=True).
# ====================================================================================================
def test_categorical_key_columns_give_the_same_numbers_and_no_warning():
    from es.measures import (STRATA_DECILE, add_calendar_deciles, daily_portfolio, drop_partial_edge_days, esred,
                             selc, snet, summarize)
    a = _summary_rows()
    plain = pd.concat([a, a.assign(node="B", market="N", wclass=1)], ignore_index=True)       # two markets, two nodes
    categories = {"market": ["M", "N", "Z"], "node": ["A", "B", "Z"], "wclass": [0, 1, 2]}    # "Z" and 2 are never used
    cat = plain.assign(**{c: pd.Categorical(plain[c], categories=k) for c, k in categories.items()})
    with warnings.catch_warnings():
        warnings.simplefilter("error")             # without observed=True pandas warns (FutureWarning) in each groupby
        assert selc(cat) == selc(plain) and snet(cat) == snet(plain)
        assert selc(cat, STRATA_DECILE) == selc(plain, STRATA_DECILE)
        assert snet(cat, STRATA_DECILE) == snet(plain, STRATA_DECILE)
        assert summarize(cat) == summarize(plain)
        # a categorical settle_date keeps unused categories, e.g. the dates of rows filtered out: they must not turn
        # into days with a P&L of 0 in the daily portfolio
        dates = sorted(set(plain["settle_date"])) + [dt.date(2025, 8, 1)]
        by_date = plain.assign(settle_date=pd.Categorical(plain["settle_date"], categories=dates))
        assert (daily_portfolio(by_date, with_alerts=True).to_numpy().tolist()
                == daily_portfolio(plain, with_alerts=True).to_numpy().tolist())
        assert esred(by_date) == esred(plain)
        # the two functions that group a frame by market and node before the measures see it
        rows, g6 = _scored(100)
        cat_rows = rows.assign(market=pd.Categorical(rows["market"], ["M", "Z"]),
                               node=pd.Categorical(rows["node"], ["A", "Z"]))
        cat_g6 = g6.assign(market=pd.Categorical(g6["market"], ["M", "Z"]), node=pd.Categorical(g6["node"], ["A", "Z"]))
        assert (add_calendar_deciles(cat_rows, cat_g6)["cal_decile"].tolist()
                == add_calendar_deciles(rows, g6)["cal_decile"].tolist())
        days = _hours([5, 24, 24])
        cat_days = days.assign(market=pd.Categorical(days["market"], ["M", "Z"]),
                               node=pd.Categorical(days["node"], ["A", "Z"]))
        assert sorted(set(drop_partial_edge_days(cat_days)["settle_date"])) == [day(1), day(2)]


# ====================================================================================================
# Ruling 68. Primary strata are node x hour x weekday class x month; the deciles are the sensitivity.
# ====================================================================================================
def test_the_strata_are_as_ruled_and_strata_month_is_an_alias_of_the_primary_strata():
    from es import measures
    assert measures.STRATA_PRIMARY == ["market", "node", "settle_hour", "wclass", "month"]
    assert measures.STRATA_DECILE == ["market", "node", "cal_decile"]
    assert measures.STRATA_MONTH is measures.STRATA_PRIMARY            # imports of the old name keep working


def test_selc_snet_and_summarize_default_to_the_primary_strata():
    from es.measures import STRATA_PRIMARY, selc, snet, summarize
    for fn in (selc, snet, summarize):
        assert inspect.signature(fn).parameters["strata"].default is STRATA_PRIMARY


def test_decile_strata_that_straddle_calendar_groups_bias_selc_upward_while_the_primary_strata_do_not():
    """Hour 0 and hour 1 are two calendar groups. Alerts are independent of the loss within each group: hour 0 loses 10
    on every row and alerts on 3 of 4, hour 1 loses 1 on every row and alerts on 1 of 4, so whichever rows are alerted
    the captured loss is the same: the null is true. A decile that holds both groups cannot tell them apart."""
    from es.measures import STRATA_DECILE, STRATA_PRIMARY, selc, summarize
    r = _rows([1, 1, 1, 0, 1, 0, 0, 0], [-10] * 4 + [-1] * 4, deciles=[0] * 8, hours=[0] * 4 + [1] * 4)
    # primary: hour 0 captures 30 and expects 40 * 3/4 = 30; hour 1 captures 1 and expects 4 * 1/4 = 1: SELC 0
    assert np.isclose(selc(r, STRATA_PRIMARY), 0.0)
    # one decile: captured 31, expected 44 * 4/8 = 22, excess 9 of 44: a calendar-only null looks like timing skill
    assert np.isclose(selc(r, STRATA_DECILE), 9.0 / 44)
    s = summarize(r)
    assert np.isclose(s["SELC"], 0.0) and np.isclose(s["SELC_decile"], 9.0 / 44)
