"""es.run_driftpath: the weekly alert-rate path of the fixed and the rolling cutoff (plan 2C, Figure 2 of the revision).

Every expected number below is worked out by hand in the comment next to it, never computed with the code under test. The small
score frames use MISO (a fixed UTC-5 settlement clock: no daylight-saving day), so a settlement day always has 24 rows; the
daylight-saving days are tested on NYISO alert frames directly."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

D = dt.date
ALPHA = 0.05


# ---------------------------------------------------------------------------------------------------------------------
# helpers: score frames, alert frames and the output files of the pipeline that the checks read
# ---------------------------------------------------------------------------------------------------------------------
def _day_rows(market, node, day, scores, split):
    """24 hourly rows of one MISO settlement day (local midnight is 05:00 UTC): the columns that the policies read."""
    start = pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=5)
    return pd.DataFrame({"market": market, "node": node, "timestamp_utc": [start + pd.Timedelta(hours=h) for h in range(24)],
                         "settle_date": day, "settle_hour": range(24), "split": split, "score": np.asarray(scores, float)})


def _days(first, n):
    return [first + dt.timedelta(days=i) for i in range(n)]


def _node_a():
    """Node A of the hand case. 56 calibration days (2024-11-01 to 2024-12-26) of score 1.0, then 17 deployment days
    (2024-12-27 to 2025-01-12): days 1 to 3 score 3.0 in every hour; days 4 to 10 have 2 hours at 3.0, 3 at 2.0, 19 at 0.5; days 11 to 17
    have 4 hours at 3.0, 4 at 2.0, 16 at 0.5."""
    parts = [_day_rows("MISO", "A", d, [1.0] * 24, "calib") for d in _days(D(2024, 11, 1), 56)]
    shape = {range(0, 3): [3.0] * 24, range(3, 10): [3.0] * 2 + [2.0] * 3 + [0.5] * 19, range(10, 17): [3.0] * 4 + [2.0] * 4 + [0.5] * 16}
    for i, d in enumerate(_days(D(2024, 12, 27), 17)):
        parts.append(_day_rows("MISO", "A", d, next(v for k, v in shape.items() if i in k), "deploy"))
    return pd.concat(parts, ignore_index=True)


def _node_b():
    """Node B: the same 56 calibration days at 1.0; in deployment 20 hours at 2.0 (10 on 2025-01-01, 10 on 2025-01-08), the rest 0.5.
    Its 20 high hours are 1.5 % of any 1344-row window, so the 95 % cutoff of the rolling window stays at 1.0."""
    parts = [_day_rows("MISO", "B", d, [1.0] * 24, "calib") for d in _days(D(2024, 11, 1), 56)]
    for d in _days(D(2024, 12, 27), 17):
        scores = [2.0] * 10 + [0.5] * 14 if d in (D(2025, 1, 1), D(2025, 1, 8)) else [0.5] * 24
        parts.append(_day_rows("MISO", "B", d, scores, "deploy"))
    return pd.concat(parts, ignore_index=True)


def _scores():
    return pd.concat([_node_a(), _node_b()], ignore_index=True)


def _alerts(market, node, days, per_day, hours=24):
    """An alert frame (what the policies return, reduced to the columns of weekly_counts): `per_day` alerts on each day."""
    rows = []
    for d, k in zip(days, per_day):
        rows += [{"market": market, "node": node, "settle_date": d, "alert": h < k} for h in range(hours)]
    return pd.DataFrame(rows)


def _write(path: Path, frame: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _sources(tmp_path, *, failures=1, nodes=2, rate_p1=183 / 816, rate_p4=134 / 816, legacy_alerts=(163, 20), legacy_n=(408, 408)):
    """The three files that the checks read, written the way the pipeline writes them, for the two-node hand case:
    legacy/summary.csv (Table 3), the per-node file behind it, and economics/summary.csv."""
    from es.paper_macros import Sources
    out = tmp_path / "out"
    row = {"analysis": "2024-2025", "block": "G1_24", "estimator": "xgb", "seed": 42, "policy": "P1", "alpha": 0.05, "failures": failures,
           "n_nodes": nodes, "median_R": 1.0, "median_log_error": 0.1, "median_auprc": 0.1, "prevalence": 0.05}
    other = dict(row, block="LEV6", failures=99)                                  # another block: must not be read
    _write(out / "legacy" / "summary.csv", pd.DataFrame([row, other]))
    per_node = pd.DataFrame({"market": ["MISO", "MISO"], "node": ["A", "B"], "n": list(legacy_n), "alerts": list(legacy_alerts)})
    per_node["R"] = per_node.alerts / (ALPHA * per_node.n)
    _write(out / "legacy" / "2024-2025__G1_24__xgb__s42__P1__a0.05.csv", per_node)
    econ = []
    for policy, rate in (("P1", rate_p1), ("P4", rate_p4)):
        for position, fees in (("premium", "calibrated"), ("inc", "calibrated"), ("inc", "0.5"), ("dec", "calibrated")):
            econ.append({"analysis": "2024-2025", "policy": policy, "comparator": "none", "position": position, "fees": fees, "alert_rate": rate})
    econ.append({"analysis": "2024-2025", "policy": "comparator", "comparator": "calendar@count", "position": "inc", "fees": "calibrated",
                 "alert_rate": 0.0})
    econ.append({"analysis": "AQ3", "policy": "P4", "comparator": "none", "position": "inc", "fees": "calibrated", "alert_rate": 0.5})
    _write(out / "economics" / "summary.csv", pd.DataFrame(econ))
    return Sources(out, tmp_path / "paper", tmp_path / "docs")


def _fetch(frames):
    """A stand-in for es.scores.cached_scores that serves a frame per analysis id and records what it was asked for."""
    calls = []

    def fetch(panel, analysis, block, estimator, seed):
        calls.append((analysis.id, block, estimator, seed))
        return frames[analysis.id]
    fetch.calls = calls
    return fetch


# ---------------------------------------------------------------------------------------------------------------------
# the day clock: how many hours a settlement day has
# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("market,day,hours", [
    ("NYISO", D(2025, 1, 15), 24), ("NYISO", D(2025, 3, 9), 23), ("NYISO", D(2025, 11, 2), 25),     # 2025: spring forward 9 March, fall back 2 Nov
    ("ISONE", D(2026, 3, 8), 23), ("CAISO", D(2025, 11, 2), 25), ("CAISO", D(2025, 3, 9), 23),
    ("MISO", D(2025, 3, 9), 24), ("MISO", D(2025, 11, 2), 24)])                                       # MISO settles in fixed EST
def test_a_settlement_day_has_23_24_or_25_hours_by_the_clock_of_its_market(market, day, hours):
    from es.run_driftpath import hours_in_local_day
    assert hours_in_local_day(market, day) == hours


def test_an_unknown_market_is_refused():
    from es.run_driftpath import hours_in_local_day
    with pytest.raises(KeyError, match="ERCOT"):
        hours_in_local_day("ERCOT", D(2025, 1, 1))


# ---------------------------------------------------------------------------------------------------------------------
# weekly counts
# ---------------------------------------------------------------------------------------------------------------------
def test_a_week_starts_on_monday_so_a_sunday_belongs_to_the_week_before():
    from es.run_driftpath import weekly_counts
    w = weekly_counts(_alerts("MISO", "A", [D(2025, 1, 5), D(2025, 1, 6)], [1, 1]))      # Sunday 5 Jan 2025, Monday 6 Jan 2025
    assert list(w.week_start) == [pd.Timestamp("2024-12-30"), pd.Timestamp("2025-01-06")]
    assert list(w.n_hours) == [24, 24] and list(w.alerts) == [1, 1]


def test_a_full_week_has_seven_whole_days_and_its_counts_are_the_sums():
    from es.run_driftpath import weekly_counts
    days = _days(D(2025, 1, 6), 7)                                                       # Monday 6 to Sunday 12 January 2025
    w = weekly_counts(_alerts("MISO", "A", days, [0, 1, 2, 3, 0, 0, 2]))                  # 8 alerts in 7 x 24 = 168 hours
    row = w.iloc[0]
    assert len(w) == 1 and row.n_hours == 168 and row.alerts == 8 and bool(row.full_week)
    assert row.rate == pytest.approx(8 / 168) and row.R == pytest.approx((8 / 168) / 0.05)      # R = 0.952381


def test_a_week_with_a_missing_hour_or_a_missing_day_is_counted_but_not_full():
    from es.run_driftpath import weekly_counts
    days = _days(D(2025, 1, 6), 7)
    complete = _alerts("MISO", "A", days, [1] * 7)
    one_hour_short = complete.drop(index=complete.index[30])                              # an hour gone from the second day
    short = weekly_counts(one_hour_short).iloc[0]
    assert short.n_hours == 167 and short.alerts == 7                                   # counted over what is there (hour 6 of day 2 was no alert)
    assert short.rate == pytest.approx(7 / 167) and not bool(short.full_week)
    six_days = weekly_counts(_alerts("MISO", "A", days[:6], [1] * 6)).iloc[0]
    assert six_days.n_hours == 144 and six_days.alerts == 6 and not bool(six_days.full_week)


def test_the_first_and_last_weeks_of_a_deployment_window_are_partial():
    from es.run_driftpath import weekly_counts
    w = weekly_counts(_alerts("MISO", "A", _days(D(2025, 1, 1), 14), [1] * 14))        # Wednesday 1 Jan to Tuesday 14 Jan 2025
    assert list(w.week_start) == [pd.Timestamp("2024-12-30"), pd.Timestamp("2025-01-06"), pd.Timestamp("2025-01-13")]
    assert list(w.n_hours) == [5 * 24, 7 * 24, 2 * 24]                                  # Wed-Sun, Mon-Sun, Mon-Tue
    assert list(w.full_week) == [False, True, False]


def test_a_week_with_a_daylight_saving_day_is_full_when_the_day_has_its_23_or_25_rows():
    from es.run_driftpath import weekly_counts
    spring = _days(D(2025, 3, 3), 7)                                                      # Monday 3 to Sunday 9 March: Sunday has 23 hours
    frame = pd.concat([_alerts("NYISO", "N", spring[:6], [1] * 6), _alerts("NYISO", "N", spring[6:], [1], hours=23)], ignore_index=True)
    assert bool(weekly_counts(frame).iloc[0].full_week) and weekly_counts(frame).iloc[0].n_hours == 6 * 24 + 23
    fall = _days(D(2025, 10, 27), 7)                                                      # Monday 27 Oct to Sunday 2 Nov: Sunday has 25 hours
    whole = pd.concat([_alerts("NYISO", "N", fall[:6], [1] * 6), _alerts("NYISO", "N", fall[6:], [1], hours=25)], ignore_index=True)
    assert bool(weekly_counts(whole).iloc[0].full_week) and weekly_counts(whole).iloc[0].n_hours == 6 * 24 + 25
    short = pd.concat([_alerts("NYISO", "N", fall[:6], [1] * 6), _alerts("NYISO", "N", fall[6:], [1], hours=23)], ignore_index=True)
    assert not bool(weekly_counts(short).iloc[0].full_week)                             # the data lack the two repeated hours: not complete


def test_two_nodes_and_two_markets_are_counted_apart():
    from es.run_driftpath import weekly_counts
    days = _days(D(2025, 1, 6), 7)
    frame = pd.concat([_alerts("MISO", "A", days, [2] * 7), _alerts("MISO", "B", days, [4] * 7), _alerts("CAISO", "A", days, [0] * 7)],
                      ignore_index=True)
    w = weekly_counts(frame).set_index(["market", "node"])
    assert w.loc[("MISO", "A"), "alerts"] == 14 and w.loc[("MISO", "B"), "alerts"] == 28 and w.loc[("CAISO", "A"), "alerts"] == 0
    assert w.loc[("MISO", "B"), "R"] == pytest.approx((28 / 168) / 0.05)               # 3.333


def test_weekly_counts_names_a_missing_column():
    from es.run_driftpath import weekly_counts
    with pytest.raises(LookupError, match="alert"):
        weekly_counts(_alerts("MISO", "A", [D(2025, 1, 6)], [1]).drop(columns="alert"))


# ---------------------------------------------------------------------------------------------------------------------
# the cutoffs by hand: node A and node B of the two-node case
# ---------------------------------------------------------------------------------------------------------------------
def _paths():
    from es.run_driftpath import policy_alerts, weekly_counts
    alerts = policy_alerts(_scores())
    return alerts, {p: weekly_counts(a).set_index(["market", "node", "week_start"]) for p, a in alerts.items()}


def test_the_fixed_cutoff_of_node_a_by_hand():
    """Calibration is constant 1.0, so the P1 cutoff is 1.0 and an hour alerts when its score is at least 1.0. Days 1 to 3 (24 x 3.0): 24 a day = 72
    in the partial week of 23 Dec. Days 4 to 10: 2 + 3 = 5 a day = 35 in the week of 30 Dec. Days 11 to 17: 4 + 4 = 8 a day = 56 in the week of 6 Jan."""
    _, w = _paths()
    a = w["P1"].loc["MISO", "A"]
    assert list(a.alerts) == [72, 35, 56] and list(a.n_hours) == [72, 168, 168] and list(a.full_week) == [False, True, True]
    assert list(a.R) == pytest.approx([(72 / 72) / 0.05, (35 / 168) / 0.05, (56 / 168) / 0.05])     # 20, 4.1667, 6.6667


def test_the_rolling_cutoff_of_node_a_adapts_by_hand():
    """P4: the cutoff of a day is the 95 % quantile of the non-training scores of the 56 days before it. Day 1 and day 2 and day 3: 1.0 (at most 72 of 1344
    window values are 3.0, and index 0.95 x 1343 = 1275.85 is still among the 1296 or more values of 1.0) - 24 alerts each. From day 4 on the 72 hours at
    3.0 fill the top 72 of the 1344 window values (index 1275 is among them), so the cutoff is 3.0 until they leave the window and an hour alerts only
    at 3.0: 2 a day in the week of 30 Dec = 14, 4 a day in the week of 6 Jan = 28."""
    _, w = _paths()
    a = w["P4"].loc["MISO", "A"]
    assert list(a.alerts) == [72, 14, 28] and list(a.full_week) == [False, True, True]
    assert list(a.R) == pytest.approx([20.0, (14 / 168) / 0.05, (28 / 168) / 0.05])               # 20, 1.6667, 3.3333


def test_the_rolling_window_is_56_days_neither_shorter_nor_longer():
    """Node C: 84 calibration days (2024-10-04 to 2024-12-26) in three levels of 28 days: 9.0, then 5.0, then 1.0. Its one deployment day (2024-12-27)
    has 6 hours at 7.0, 6 at 3.0 and 12 at 0.5. P1 reads all of calibration: 95 % quantile of 2016 values (672 each) at index 0.95 x 2015 = 1914.25
    is 9.0, so no hour alerts. P4 over 56 days reads 2024-11-01 to 2024-12-26, 672 values of 5.0 and 672 of 1.0: index 0.95 x 1343 = 1275.85 is 5.0, so only
    the 6 hours at 7.0 alert. A window of 28 days would read the 1.0s only (cutoff 1.0: 12 alerts), one of 84 days would read the 9.0s as well (cutoff 9.0: none)."""
    from es.run_driftpath import policy_alerts, weekly_counts
    parts = [_day_rows("MISO", "C", d, [level] * 24, "calib") for level, first in ((9.0, D(2024, 10, 4)), (5.0, D(2024, 11, 1)), (1.0, D(2024, 11, 29)))
             for d in _days(first, 28)]
    parts.append(_day_rows("MISO", "C", D(2024, 12, 27), [7.0] * 6 + [3.0] * 6 + [0.5] * 12, "deploy"))
    alerts = policy_alerts(pd.concat(parts, ignore_index=True))
    assert int(weekly_counts(alerts["P1"]).alerts.sum()) == 0
    assert int(weekly_counts(alerts["P4"]).alerts.sum()) == 6


def test_node_b_has_the_same_alerts_under_both_cutoffs():
    """Node B alerts on its 20 hours at 2.0 (at least the cutoff 1.0 under either policy; they are 1.5 % of a window, so the rolling cutoff stays at 1.0):
    10 on 1 Jan 2025 (week of 30 Dec) and 10 on 8 Jan 2025 (week of 6 Jan)."""
    _, w = _paths()
    for policy in ("P1", "P4"):
        b = w[policy].loc["MISO", "B"]
        assert list(b.alerts) == [0, 10, 10] and list(b.n_hours) == [72, 168, 168]


def test_the_whole_period_r_from_the_weekly_sums_and_the_failing_nodes():
    """Node A, P1: 163 alerts in 408 hours, R = 163 / (0.05 x 408) = 7.99; node B: 20 alerts, R = 20 / 20.4 = 0.98."""
    from es.run_driftpath import failing_nodes, weekly_counts, whole_period_r
    alerts, _ = _paths()
    r = whole_period_r(weekly_counts(alerts["P1"])).set_index("node").R
    assert r["A"] == pytest.approx(163 / 20.4) and r["B"] == pytest.approx(20 / 20.4)
    assert failing_nodes(whole_period_r(weekly_counts(alerts["P1"]))) == 1                   # A is above 2, B is inside [0.5, 2]
    r4 = whole_period_r(weekly_counts(alerts["P4"]))
    assert r4.set_index("node").R["A"] == pytest.approx(114 / 20.4)                          # 72 + 14 + 28 = 114 alerts: 5.588
    assert failing_nodes(r4) == 1


def test_the_band_edges_are_inside_the_band():
    """R = 0.5 and R = 2 are not failures (the audit's fail_0.5_2 is R < 0.5 or R > 2). n = 400 hours: 10 alerts give 0.5 and 40 give 2."""
    from es.run_driftpath import failing_nodes
    r = pd.DataFrame({"market": "MISO", "node": list("ABCD"), "n_hours": 400, "alerts": [10, 9, 40, 41]})
    r["R"] = r.alerts / (0.05 * r.n_hours)                                                     # 0.5, 0.45, 2.0, 2.05
    assert failing_nodes(r) == 2


# ---------------------------------------------------------------------------------------------------------------------
# trailing windows (T1b): the alert rate of a node over the last weeks, from the weekly path
# ---------------------------------------------------------------------------------------------------------------------
WEEKS5 = pd.date_range("2025-01-06", periods=5, freq="7D")                    # the Mondays 6, 13, 20, 27 January and 3 February 2025


def _weekly(nodes):
    """A weekly path from (market, node, hours per week, alerts per week) over the five weeks of WEEKS5; an hours entry of None is a week without a row."""
    rows = []
    for market, node, hours, alerts in nodes:
        for week, h, a in zip(WEEKS5, hours, alerts):
            if h is not None:
                rows.append({"market": market, "node": node, "week_start": week, "n_hours": h, "alerts": a})
    return pd.DataFrame(rows)


def _hand_case():
    """Two nodes, five weeks; the last week has four days (96 hours). Node A (MISO): 160 hours in each of the first four weeks, 96 in the last, alerts 8, 24, 16,
    24, 20. Node B (NYISO): hours 160, 160, 40, 72, 96, alerts 4, 2, 1, 1, 1. With alpha = 0.05, 0.05 x 160 = 8, so the windows of two weeks (weeks=2, min_weeks=1;
    a node-window needs half of 2 x 168 = 336 hours, that is 168, and the first window, of one week, needs 84) are, by hand:

      window ends   weeks   A: hours alerts  R = alerts / (0.05 hours)        B: hours alerts  R
      13 Jan          1          160     8   8 / 8 = 1.0                         160     4    4 / 8 = 0.5
      20 Jan          2          320    32   32 / 16 = 2.0                       320     6    6 / 16 = 0.375
      27 Jan          2          320    40   40 / 16 = 2.5                       200     3    3 / 10 = 0.3
       3 Feb          2          320    40   40 / 16 = 2.5                       (40 + 72 = 112 hours < 168: left out)
       7 Feb          2          256    44   44 / 12.8 = 3.4375                  168     2    2 / 8.4 = 0.238095

    The last week has four days (its largest n_hours is 96), so the last window ends on Friday 7 February, the day after Thursday 6 February, and not on Monday 10."""
    return _weekly([("MISO", "A", [160, 160, 160, 160, 96], [8, 24, 16, 24, 20]), ("NYISO", "B", [160, 160, 40, 72, 96], [4, 2, 1, 1, 1])])


def _ends(*days):
    return [pd.Timestamp(d) for d in days]


def test_the_summary_is_written_where_the_registry_reads_it():
    """The macros read paper/drift_summary.csv through Sources (outputs/es/ + the name), and the run writes SUMMARY_PATH next to drift_path.csv: one file."""
    from es.paper_macros import Sources
    from es.paper_registry import DRIFT_SUMMARY
    from es.run_driftpath import OUT_PATH, SOURCE, SUMMARY_PATH, SUMMARY_SOURCE
    assert (SOURCE, SUMMARY_SOURCE) == ("paper/drift_path.csv", "paper/drift_summary.csv") and DRIFT_SUMMARY == SUMMARY_SOURCE
    assert Sources().path(SUMMARY_SOURCE) == SUMMARY_PATH and Sources().path(SOURCE) == OUT_PATH
    assert SUMMARY_PATH.name == "drift_summary.csv" and SUMMARY_PATH.parent == OUT_PATH.parent


def test_the_window_constants_are_the_rolling_window_of_p4_in_weeks():
    from es.run_driftpath import P4_WINDOW_DAYS, WINDOW_MIN_HOURS_SHARE, WINDOW_MIN_WEEKS, WINDOW_WEEKS
    assert (WINDOW_WEEKS, WINDOW_MIN_WEEKS, WINDOW_MIN_HOURS_SHARE) == (8, 4, 0.5)
    assert WINDOW_WEEKS * 7 == P4_WINDOW_DAYS == 56                             # eight weeks are the 56 days of p4_rolling


def test_a_window_that_is_not_the_window_of_p4_stops_the_import(monkeypatch):
    """The relation is checked when the module is imported (and not with an assert statement, which python -O would skip)."""
    import importlib

    import es.paper_pins
    import es.run_driftpath
    monkeypatch.setitem(es.paper_pins.PINS, "pFourWindow", 28)
    try:
        with pytest.raises(ImportError, match="8 weeks is 56 days.*28 days"):
            importlib.reload(es.run_driftpath)
    finally:
        monkeypatch.undo()
        importlib.reload(es.run_driftpath)                                      # the module as it was, for the tests that follow
    assert es.run_driftpath.P4_WINDOW_DAYS == 56


def test_a_window_is_the_weeks_that_end_with_a_week_and_its_rate_is_the_ratio_of_the_sums():
    from es.run_driftpath import window_r
    w = window_r(_hand_case(), weeks=2, min_weeks=1)
    assert list(w.columns) == ["market", "node", "week_end", "weeks", "n_hours", "alerts", "R"]
    a, b = w[w.node == "A"], w[w.node == "B"]
    assert list(a.week_end) == _ends("2025-01-13", "2025-01-20", "2025-01-27", "2025-02-03", "2025-02-07")
    assert list(a.weeks) == [1, 2, 2, 2, 2] and list(a.n_hours) == [160, 320, 320, 320, 256] and list(a.alerts) == [8, 32, 40, 40, 44]
    assert list(a.R) == pytest.approx([8 / 8, 32 / 16, 40 / 16, 40 / 16, 44 / 12.8])                       # 1.0, 2.0, 2.5, 2.5, 3.4375
    assert list(b.week_end) == _ends("2025-01-13", "2025-01-20", "2025-01-27", "2025-02-07")               # 3 Feb: left out
    assert list(b.weeks) == [1, 2, 2, 2] and list(b.n_hours) == [160, 320, 200, 168] and list(b.alerts) == [4, 6, 3, 2]
    assert list(b.R) == pytest.approx([4 / 8, 6 / 16, 3 / 10, 2 / 8.4])                                    # 0.5, 0.375, 0.3, 0.238095
    assert list(w.market) == ["MISO"] * 5 + ["NYISO"] * 4                                                 # by market and node, then window end


def test_the_first_window_ends_with_the_min_weeks_th_week_and_the_windows_slide_once_they_have_all_their_weeks():
    """weeks=3, min_weeks=2 on the hand case: windows end with weeks 2, 3, 4 and 5; the first two have the weeks so far, the last two the last three weeks.
    A: weeks 1-2: 320 hours, 32 alerts, R = 32 / 16 = 2.0; weeks 1-3: 480, 48, 48 / 24 = 2.0; weeks 2-4: 480, 24 + 16 + 24 = 64, 64 / 24 = 2.6667;
    weeks 3-5: 160 + 160 + 96 = 416, 16 + 24 + 20 = 60, 60 / 20.8 = 2.884615. B (3 weeks need half of 504 = 252 hours): 320 and 6 alerts, 6 / 16 = 0.375;
    360 and 7, 7 / 18 = 0.388889; 160 + 40 + 72 = 272 and 4, 4 / 13.6 = 0.294118; weeks 3-5: 40 + 72 + 96 = 208 < 252, left out."""
    from es.run_driftpath import window_r
    w = window_r(_hand_case(), weeks=3, min_weeks=2)
    a, b = w[w.node == "A"], w[w.node == "B"]
    assert list(a.week_end) == _ends("2025-01-20", "2025-01-27", "2025-02-03", "2025-02-07") and list(a.weeks) == [2, 3, 3, 3]
    assert list(a.n_hours) == [320, 480, 480, 416] and list(a.alerts) == [32, 48, 64, 60]
    assert list(a.R) == pytest.approx([32 / 16, 48 / 24, 64 / 24, 60 / 20.8])
    assert list(b.week_end) == _ends("2025-01-20", "2025-01-27", "2025-02-03") and list(b.weeks) == [2, 3, 3]
    assert list(b.n_hours) == [320, 360, 272] and list(b.alerts) == [6, 7, 4]
    assert list(b.R) == pytest.approx([6 / 16, 7 / 18, 4 / 13.6])


def test_a_node_window_with_fewer_than_half_the_nominal_hours_is_left_out_and_exactly_half_is_kept():
    """B's window of 7 February has 72 + 96 = 168 hours, exactly half of 2 x 168: kept. With 71 hours in the fourth week it has 167 and is left out."""
    from es.run_driftpath import window_r
    kept = window_r(_hand_case(), weeks=2, min_weeks=1)
    assert 168 in set(kept[(kept.node == "B") & (kept.week_end == "2025-02-07")].n_hours)
    short = _weekly([("MISO", "A", [160, 160, 160, 160, 96], [8, 24, 16, 24, 20]), ("NYISO", "B", [160, 160, 40, 71, 96], [4, 2, 1, 1, 1])])
    w = window_r(short, weeks=2, min_weeks=1)
    assert list(w[w.node == "B"].week_end) == _ends("2025-01-13", "2025-01-20", "2025-01-27")           # 71 + 96 = 167 < 168 (and 40 + 71 = 111): nothing after 27 Jan
    assert list(w[w.node == "A"].week_end) == _ends("2025-01-13", "2025-01-20", "2025-01-27", "2025-02-03", "2025-02-07")


def test_the_nominal_hours_are_those_of_the_weeks_in_the_window():
    """The window of one week needs 84 hours: A has 84 and is kept, C has 83 and is left out. The next window has two weeks and needs 168: C has 83 + 100 = 183
    and is kept, A has 84 + 100 = 184."""
    from es.run_driftpath import window_r
    w = window_r(_weekly([("MISO", "A", [84, 100, 100, 100, 96], [1] * 5), ("MISO", "C", [83, 100, 100, 100, 96], [1] * 5)]), weeks=2, min_weeks=1)
    assert list(w[w.week_end == "2025-01-13"].node) == ["A"]
    second = w[w.week_end == "2025-01-20"]
    assert list(second.node) == ["A", "C"] and list(second.n_hours) == [184, 183]


def test_a_week_that_a_node_has_no_row_for_counts_as_no_hours_in_every_window_that_holds_it():
    """G has 100 hours in weeks 1, 2 and 4 and no row in weeks 3 and 5; H has all five weeks (100 hours; 96 in the last). The windows are calendar weeks:
    {1}: G 100 >= 84, kept; {1, 2}: 200 >= 168, kept; {2, 3}: G has week 2 only, 100 < 168: left out; {3, 4}: week 4 only, left out; {4, 5}: week 4 only, left out. If
    the window were the node's own last two rows, G would have 200 hours in {3, 4} and be kept. Only H has all five windows."""
    from es.run_driftpath import window_r
    w = window_r(_weekly([("MISO", "G", [100, 100, None, 100, None], [5] * 5), ("MISO", "H", [100, 100, 100, 100, 96], [5] * 5)]), weeks=2, min_weeks=1)
    assert list(w[w.node == "G"].week_end) == _ends("2025-01-13", "2025-01-20")
    assert list(w[w.node == "G"].R) == pytest.approx([5 / 5, 10 / 10])                                     # 1.0, 1.0
    assert list(w[w.node == "H"].week_end) == _ends("2025-01-13", "2025-01-20", "2025-01-27", "2025-02-03", "2025-02-07")


def test_a_node_window_with_no_hour_is_left_out_whatever_the_share(monkeypatch):
    """With the share at zero, 0 >= 0 would keep a window of no hours: the rule 'or is zero' leaves it out. Z has no hour in the first two weeks, so the windows
    that end with week 1 ({1}) and week 2 ({1, 2}) have none: left out. The windows {2, 3}, {3, 4} and {4, 5} have 100, 200 and 100 + 96 = 196 hours: kept at
    share zero."""
    import es.run_driftpath as module
    monkeypatch.setattr(module, "WINDOW_MIN_HOURS_SHARE", 0.0)
    w = module.window_r(_weekly([("MISO", "Z", [0, 0, 100, 100, 96], [0, 0, 5, 5, 5])]), weeks=2, min_weeks=1)
    assert list(w.week_end) == _ends("2025-01-27", "2025-02-03", "2025-02-07") and list(w.n_hours) == [100, 200, 196]
    assert list(w.weeks) == [2, 2, 2] and list(w.alerts) == [5, 10, 10]


def test_the_rate_of_a_window_is_a_ratio_of_sums_and_partial_weeks_and_missing_hours_count_with_their_hours():
    """A partial first week (96 hours, 9 alerts) and a week of 168 hours (6 alerts): the weekly rates are 9 / 4.8 = 1.875 and 6 / 8.4 = 0.714286, their mean is
    1.294643, the ratio of sums is 15 / (0.05 x 264) = 15 / 13.2 = 1.136364. Nothing is left out because the first week is partial (full_week is not read)."""
    from es.run_driftpath import window_r
    weekly = pd.DataFrame({"market": "MISO", "node": "A", "week_start": WEEKS5[:2], "n_hours": [96, 168], "alerts": [9, 6], "full_week": [False, True]})
    w = window_r(weekly, weeks=2, min_weeks=2)
    assert list(w.n_hours) == [264] and list(w.alerts) == [15] and list(w.R) == pytest.approx([15 / 13.2])
    assert w.R.iloc[0] != pytest.approx((9 / 4.8 + 6 / 8.4) / 2)
    no_flag = window_r(weekly.drop(columns="full_week"), weeks=2, min_weeks=2)
    assert list(no_flag.R) == list(w.R)                                                                    # the column is not needed


@pytest.mark.parametrize("last_hours,days", [(168, 7), (169, 7), (167, 7), (96, 4), (95, 4), (97, 5), (24, 1), (20, 1), (1, 1)])
def test_the_last_window_ends_the_day_after_the_last_settlement_day_and_never_after_the_next_monday(last_hours, days):
    """The last week starts on Monday 3 February 2025. Its days are its largest n_hours over the nodes in days, rounded up and at most seven (a week with a 25-hour
    Sunday has 169 hours): 168 and 169 and 167 hours are 7 days (the Monday after, 10 February), 96 and 95 hours are 4, 97 are 5, 24, 20 and 1 hour are 1 day.
    B has fewer hours in that week: the largest of the nodes counts."""
    from es.run_driftpath import window_r
    full = [168, 168, 168, 168]
    w = window_r(_weekly([("MISO", "A", full + [last_hours], [1] * 5), ("MISO", "B", full + [max(1, last_hours // 2)], [1] * 5)]), weeks=2, min_weeks=1)
    assert w.week_end.max() == pd.Timestamp("2025-02-03") + pd.Timedelta(days=days)
    assert w.week_end.nunique() == 5 and (w.groupby("week_end").size() == 2).all()                        # five windows of both nodes


def test_the_cap_is_the_end_of_the_deployment_window_of_each_main_analysis():
    """The two main analyses have UTC windows [2025-01-01, 2026-01-01) and [2026-01-01, 2026-07-01); the day after their last settlement day is the end of the
    window: 2026-01-01 (the last week starts on Monday 29 December, with three days) and 2026-07-01 (Monday 29 June, two days). Rows as the settlement clock of
    NYISO makes them: the window ends at 19:00 local in December (EST: the last day has 19 hours, 24 + 24 + 19 = 67 in the week) and at 20:00 in June (EDT: 20 hours,
    24 + 20 = 44)."""
    from es.analyses import LEGACY
    from es.run_driftpath import window_r
    for analysis, last_monday, days, last_day_hours in (("2024-2025", "2025-12-29", 3, 19), ("25H1-26H1", "2026-06-29", 2, 20)):
        first = pd.Timestamp(last_monday) - pd.Timedelta(days=7 * 5)
        weeks = pd.date_range(first, periods=6, freq="7D")
        hours = [168] * 5 + [24 * (days - 1) + last_day_hours]
        weekly = pd.DataFrame([{"market": "NYISO", "node": "N", "week_start": wk, "n_hours": h, "alerts": 1} for wk, h in zip(weeks, hours)])
        assert window_r(weekly, weeks=2, min_weeks=1).week_end.max() == pd.Timestamp(LEGACY[analysis].deploy.end)


def test_a_window_as_long_as_the_whole_period_is_the_whole_period_rate():
    """A: 92 alerts in 736 hours, 0.05 x 736 = 36.8, R = 2.5; B: 9 alerts in 528 hours, 0.05 x 528 = 26.4, R = 0.340909 (528 >= half of 5 x 168 = 420)."""
    from es.run_driftpath import whole_period_r, window_r
    w = window_r(_hand_case(), weeks=5, min_weeks=5).set_index("node")
    assert list(w.weeks) == [5, 5] and w.loc["A", "R"] == pytest.approx(92 / 36.8) and w.loc["B", "R"] == pytest.approx(9 / 26.4)
    whole = whole_period_r(_hand_case()).set_index("node")
    assert list(w.R) == pytest.approx(list(whole.R)) and list(w.n_hours) == list(whole.n_hours)


def test_dates_given_as_text_are_read_and_a_path_with_fewer_weeks_than_the_first_window_has_no_window():
    from es.run_driftpath import window_r
    as_text = _hand_case().assign(week_start=lambda d: d.week_start.dt.strftime("%Y-%m-%d"))
    assert list(window_r(as_text, weeks=2, min_weeks=1).R) == list(window_r(_hand_case(), weeks=2, min_weeks=1).R)
    none = window_r(_hand_case(), weeks=8, min_weeks=6)                                                   # five weeks only
    assert none.empty and list(none.columns) == ["market", "node", "week_end", "weeks", "n_hours", "alerts", "R"]
    assert window_r(_hand_case().iloc[:0], weeks=2, min_weeks=1).empty


def test_window_r_refuses_what_it_cannot_read():
    from es.run_driftpath import window_r
    case = _hand_case()
    with pytest.raises(LookupError, match="n_hours"):
        window_r(case.drop(columns="n_hours"))
    for bad in (3, 0):
        with pytest.raises(ValueError, match="min_weeks"):
            window_r(case, weeks=2, min_weeks=bad)
    with pytest.raises(ValueError, match="one analysis and one policy"):
        window_r(pd.concat([case, case], ignore_index=True))                                              # two policies in one frame
    off = case.copy()
    off.loc[off.index[1], "week_start"] += pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="whole number of weeks"):
        window_r(off, weeks=2, min_weeks=1)
    for column in ("n_hours", "alerts", "node"):                                                           # a missing value is not read as zero
        gap = case.astype({"n_hours": float, "alerts": float}).copy()
        gap.loc[gap.index[2], column] = np.nan
        with pytest.raises(ValueError, match="missing value"):
            window_r(gap, weeks=2, min_weeks=1)


def test_nodes_above_and_below_the_band_are_the_failing_nodes_split_in_two():
    """n = 400 hours (0.05 x 400 = 20): 10 alerts R = 0.5 and 40 alerts R = 2.0 are inside; 9 (0.45) is below; 41 (2.05) and 100 (5.0) are above."""
    from es.run_driftpath import failing_nodes, nodes_above_below
    r = pd.DataFrame({"market": "MISO", "node": list("ABCDE"), "n_hours": 400, "alerts": [10, 9, 40, 41, 100]})
    r["R"] = r.alerts / (0.05 * r.n_hours)
    assert nodes_above_below(r) == (2, 1) and failing_nodes(r) == 3


# ---------------------------------------------------------------------------------------------------------------------
# the summary of the windows (drift_summary.csv)
# ---------------------------------------------------------------------------------------------------------------------
HAND_SUMMARY = {"nodes": 2, "nodes_outside": 2, "nodes_above": 1, "nodes_below": 1, "window_n": 5, "window_outside_mean": 1.2, "window_outside_min": 0,
                "window_outside_max": 2, "window_median_min": 0.75, "window_median_max": 2.5}


def test_the_summary_of_the_hand_case_has_every_metric_worked_out_by_hand():
    """The windows of two weeks of the hand case (the table of _hand_case), with the band [0.5, 2] and both ends inside it:
    whole period: A 92 / 36.8 = 2.5 (above), B 9 / 26.4 = 0.3409 (below): 2 nodes, 2 outside, 1 above, 1 below.
    window ends 13 Jan: A 1.0, B 0.5 (an end of the band: inside): 0 outside, median (1.0 + 0.5) / 2 = 0.75; 20 Jan: A 2.0 (the other end: inside), B 0.375: 1 outside,
    median (2.0 + 0.375) / 2 = 1.1875; 27 Jan: A 2.5, B 0.3: 2 outside, median 1.4; 3 Feb: A 2.5 alone (B is left out): 1 outside, median 2.5;
    7 Feb: A 3.4375, B 0.238095: 2 outside, median 1.837798. So 5 windows; the outside counts 0, 1, 2, 1, 2 have mean 6 / 5 = 1.2, minimum 0, maximum 2;
    the medians have minimum 0.75 and maximum 2.5."""
    from es.run_driftpath import SUMMARY_COLUMNS, SUMMARY_METRICS, drift_summary_frame
    s = drift_summary_frame({("T", "P1"): _hand_case()}, weeks=2, min_weeks=1)
    assert list(s.columns) == ["analysis", "policy", "metric", "value"] == SUMMARY_COLUMNS
    assert list(s.metric) == ["nodes", "nodes_outside", "nodes_above", "nodes_below", "window_n", "window_outside_mean", "window_outside_min",
                              "window_outside_max", "window_median_min", "window_median_max"] == list(SUMMARY_METRICS)
    assert set(s.analysis) == {"T"} and set(s.policy) == {"P1"}
    got = dict(zip(s.metric, s.value))
    assert got == pytest.approx(HAND_SUMMARY)
    assert [type(got[m]) for m in HAND_SUMMARY] == [int, int, int, int, int, float, int, int, float, float]      # counts are whole numbers, the rest are floats


def test_a_node_exactly_on_an_end_of_the_band_is_inside_in_the_summary_too():
    """One week of 400 hours (0.05 x 400 = 20) for four nodes: 10 alerts R = 0.5 and 40 alerts R = 2.0 are inside the band, 9 (0.45) is below and 41 (2.05) above.
    One window, with the median of 0.45, 0.5, 2.0 and 2.05: (0.5 + 2.0) / 2 = 1.25."""
    from es.run_driftpath import drift_summary_frame
    weekly = pd.DataFrame({"market": "MISO", "node": list("ABCD"), "week_start": WEEKS5[0], "n_hours": 400, "alerts": [10, 9, 40, 41]})
    got = dict(zip(*drift_summary_frame({("T", "P1"): weekly}, weeks=1, min_weeks=1)[["metric", "value"]].to_numpy().T))
    assert (got["nodes"], got["nodes_outside"], got["nodes_above"], got["nodes_below"]) == (4, 2, 1, 1)
    assert (got["window_n"], got["window_outside_min"], got["window_outside_max"], got["window_outside_mean"]) == (1, 2, 2, 2.0)
    assert got["window_median_min"] == got["window_median_max"] == pytest.approx(1.25)


def test_the_summary_has_one_block_per_analysis_and_policy_in_the_order_of_the_mapping():
    from es.run_driftpath import SUMMARY_METRICS, drift_summary_frame
    s = drift_summary_frame({("B", "P4"): _hand_case(), ("A", "P1"): _hand_case(), ("A", "P4"): _hand_case()}, weeks=2, min_weeks=1)
    n = len(SUMMARY_METRICS)
    assert len(s) == 3 * n and [(s.analysis[i], s.policy[i]) for i in (0, n, 2 * n)] == [("B", "P4"), ("A", "P1"), ("A", "P4")]
    for block in range(3):
        rows = s.iloc[block * n:(block + 1) * n]
        assert list(rows.metric) == list(SUMMARY_METRICS) and len(set(zip(rows.analysis, rows.policy))) == 1


def test_counts_are_written_as_whole_numbers_and_the_rest_at_full_precision(tmp_path):
    """Written and read back, every float is the same number, and a value with a long decimal part keeps it (1 / 3: 5 alerts in 300 hours, 0.05 x 300 = 15)."""
    from es.run_driftpath import drift_summary_frame
    weekly = pd.DataFrame({"market": ["MISO"], "node": ["A"], "week_start": [WEEKS5[0]], "n_hours": [300], "alerts": [5]})
    s = drift_summary_frame({("T", "P1"): weekly, ("T", "P4"): _hand_case()}, weeks=2, min_weeks=1)
    path = tmp_path / "s.csv"
    s.to_csv(path, index=False)
    text = path.read_text().splitlines()
    assert text[0] == "analysis,policy,metric,value" and "T,P1,nodes,1" in text and "T,P4,window_n,5" in text and "T,P4,window_outside_mean,1.2" in text
    third = [line for line in text if line.startswith("T,P1,window_median_min,")][0].split(",")[3]
    assert third.startswith("0.3333333333") and len(third.split(".")[1]) >= 15                              # not rounded to a few digits
    back = pd.read_csv(path)
    assert list(back.value) == pytest.approx([float(v) for v in s.value], rel=1e-15, abs=0)                  # the same numbers


def test_a_policy_with_no_window_stops_the_summary():
    from es.run_driftpath import DriftPathError, drift_summary_frame
    three_weeks = _hand_case()[lambda d: d.week_start < WEEKS5[3]]
    with pytest.raises(DriftPathError, match="T P1.*no window of 4 to 8 weeks.*3 weeks"):
        drift_summary_frame({("T", "P1"): three_weeks})                                                   # the paper's windows start with the fourth week


def test_the_summary_check_wants_above_plus_below_to_be_outside_and_p1_to_be_the_cell_of_table_3(tmp_path):
    """The hand case has 2 of 2 nodes outside the band. Table 3 (the _sources frame) says 2 of 2: matched, and only P1 is compared (P4 has 0 outside here)."""
    from es.run_driftpath import DriftPathError, check_summary, drift_summary_frame
    calm = _weekly([("MISO", "A", [160] * 5, [8] * 5), ("NYISO", "B", [160] * 5, [8] * 5)])               # R = 1.0 in every window: nobody outside
    s = drift_summary_frame({("2024-2025", "P1"): _hand_case(), ("2024-2025", "P4"): calm}, weeks=2, min_weeks=1)
    lines = check_summary(s, _sources(tmp_path, failures=2, nodes=2))
    assert any("2024-2025 P1: 1 above + 1 below = 2 of 2 nodes outside" in line for line in lines)
    assert any("2024-2025 P4: 0 above + 0 below = 0 of 2" in line for line in lines)
    assert any("Table 3" in line and "2 of 2" in line and "matched" in line and "P1" in line for line in lines)
    assert len([line for line in lines if "Table 3" in line]) == 1                                         # P4 is not compared with it
    with pytest.raises(DriftPathError, match="Table 3") as err:
        check_summary(s, _sources(tmp_path / "other", failures=1, nodes=2))
    assert "2 of 2" in str(err.value) and "1 of 2" in str(err.value) and "drift_summary.csv" in str(err.value)
    with pytest.raises(DriftPathError, match="number of nodes differs"):
        check_summary(s, _sources(tmp_path / "three", failures=2, nodes=3))
    broken = s.copy()
    broken.loc[(broken.policy == "P1") & (broken.metric == "nodes_below"), "value"] = 0
    with pytest.raises(DriftPathError, match=r"nodes_above \(1\) \+ nodes_below \(0\) is not nodes_outside \(2\)"):
        check_summary(broken, _sources(tmp_path / "again", failures=2, nodes=2))


# ---------------------------------------------------------------------------------------------------------------------
# development windows only
# ---------------------------------------------------------------------------------------------------------------------
def test_the_last_development_day_is_allowed_and_the_next_day_is_refused():
    from es.guard import SealedDataError
    from es.run_driftpath import assert_development_only
    ok = _alerts("MISO", "A", [D(2026, 6, 30)], [1])
    assert assert_development_only(ok) is ok
    with pytest.raises(SealedDataError, match="2026-07-01"):
        assert_development_only(_alerts("MISO", "A", [D(2026, 6, 30), D(2026, 7, 1)], [1, 1]))


def test_a_run_over_a_score_frame_that_reaches_july_2026_writes_nothing(tmp_path):
    from es.guard import SealedDataError
    from es.run_driftpath import run
    scores = _scores()
    late = _day_rows("MISO", "A", D(2026, 7, 1), [3.0] * 24, "deploy")
    out = tmp_path / "drift_path.csv"
    with pytest.raises(SealedDataError):
        run(None, _sources(tmp_path), analyses=("2024-2025",), fetch=_fetch({"2024-2025": pd.concat([scores, late], ignore_index=True)}), out_path=out, log=lambda *_: None)
    assert not out.exists()


# ---------------------------------------------------------------------------------------------------------------------
# the cache: a hit is read, a miss is never refitted
# ---------------------------------------------------------------------------------------------------------------------
def test_the_scores_are_the_primary_pipelines_combined_block_xgboost_seed_42():
    from es.analyses import LEGACY
    from es.run_driftpath import primary_scores
    served = _scores()
    fetch = _fetch({"2024-2025": served})
    assert primary_scores("panel", LEGACY["2024-2025"], fetch=fetch) is served
    assert fetch.calls == [("2024-2025", "G1_24", "xgb", 42)]


def test_the_cutoffs_use_the_alpha_and_the_window_of_the_brief():
    from es.run_driftpath import ALPHA, P4_WINDOW_DAYS
    assert (ALPHA, P4_WINDOW_DAYS) == (0.05, 56)


def test_a_cold_cache_is_refused_and_no_model_is_fitted(monkeypatch):
    """A stand-in for cached_scores on a miss: it runs the builder, which is es.scores.score_analysis (a refit). The refusal must stop it before any
    fit, and es.scores.score_analysis must be the original again afterwards."""
    import es.scores as scores_module
    from es.analyses import LEGACY
    from es.run_driftpath import ColdCache, primary_scores
    fitted = []

    def spy(*args, **kwargs):
        fitted.append(args)
        return pd.DataFrame()

    monkeypatch.setattr(scores_module, "score_analysis", spy)

    def cold(panel, analysis, block, estimator, seed):
        return scores_module.score_analysis(panel, analysis, block, estimator, seed)

    with pytest.raises(ColdCache, match="2024-2025.*G1_24"):
        primary_scores("panel", LEGACY["2024-2025"], fetch=cold)
    assert fitted == [] and scores_module.score_analysis is spy
    assert primary_scores("panel", LEGACY["2024-2025"], fetch=lambda *a: _scores()) is not None       # a hit does not touch the builder
    assert fitted == []


def test_the_refusal_is_restored_even_when_the_fetch_fails_for_another_reason(monkeypatch):
    import es.scores as scores_module
    from es.analyses import LEGACY
    from es.run_driftpath import primary_scores
    original = scores_module.score_analysis

    def broken(*args):
        raise OSError("disk")

    with pytest.raises(OSError, match="disk"):
        primary_scores("panel", LEGACY["2024-2025"], fetch=broken)
    assert scores_module.score_analysis is original


def test_a_cold_cache_stops_a_run_before_anything_is_written(tmp_path):
    import es.scores as scores_module
    from es.run_driftpath import ColdCache, run
    out = tmp_path / "drift_path.csv"
    with pytest.raises(ColdCache):
        run(None, _sources(tmp_path), analyses=("2024-2025",), fetch=lambda p, a, b, e, s: scores_module.score_analysis(p, a, b, e, s),
            out_path=out, log=lambda *_: None)
    assert not out.exists()


# ---------------------------------------------------------------------------------------------------------------------
# the consistency checks
# ---------------------------------------------------------------------------------------------------------------------
def test_the_expected_failures_are_the_cell_of_table_3_for_the_combined_block_and_xgboost(tmp_path):
    """read from the frame behind tab_block_audit (legacy/summary.csv, the primary seed and cutoff, policy P1), not typed"""
    from es.run_driftpath import table3_failures
    assert table3_failures(_sources(tmp_path, failures=7, nodes=33), "2024-2025") == (7, 33)         # the LEV6 row (99) is not the one
    with pytest.raises(LookupError, match="2025-H2"):
        table3_failures(_sources(tmp_path), "2025-H2")


def test_failures_that_differ_from_table_3_stop_the_run(tmp_path):
    from es.run_driftpath import DriftPathError, run
    out = tmp_path / "drift_path.csv"
    with pytest.raises(DriftPathError, match="Table 3") as err:
        run(None, _sources(tmp_path, failures=2), analyses=("2024-2025",), fetch=_fetch({"2024-2025": _scores()}), out_path=out, log=lambda *_: None)
    assert "2024-2025" in str(err.value) and "1 of 2" in str(err.value) and "2 of 2" in str(err.value)       # the weekly sums say 1 of 2, Table 3 2 of 2
    assert not out.exists()


def test_a_node_count_that_differs_from_table_3_stops_the_run(tmp_path):
    from es.run_driftpath import DriftPathError, run
    with pytest.raises(DriftPathError, match="nodes"):
        run(None, _sources(tmp_path, nodes=3), analyses=("2024-2025",), fetch=_fetch({"2024-2025": _scores()}), out_path=tmp_path / "x.csv", log=lambda *_: None)


def test_node_counts_of_hours_and_alerts_must_equal_the_per_node_file_behind_table_3(tmp_path):
    from es.run_driftpath import DriftPathError, run
    with pytest.raises(DriftPathError, match="per-node.*A"):
        run(None, _sources(tmp_path, legacy_alerts=(164, 20)), analyses=("2024-2025",), fetch=_fetch({"2024-2025": _scores()}),
            out_path=tmp_path / "x.csv", log=lambda *_: None)


def test_the_pooled_alert_rate_must_equal_the_economics_alert_rate_of_the_same_policy(tmp_path):
    """P1: (163 + 20) / (2 x 408) = 183 / 816; P4: (114 + 20) / 816 = 134 / 816, over the rows the economics measures (no partial edge day here)."""
    from es.run_driftpath import DriftPathError, run
    lines = []
    run(None, _sources(tmp_path), analyses=("2024-2025",), fetch=_fetch({"2024-2025": _scores()}), out_path=tmp_path / "ok.csv", min_weeks=1,
        log=lines.append)                                      # (T1b) the three weeks of this case are fewer than the four that the paper's first window needs
    assert any("P4" in line and "economics" in line and "0.164216" in line for line in lines), lines
    with pytest.raises(DriftPathError, match="P4.*economics.*0.164216"):
        run(None, _sources(tmp_path, rate_p4=0.1642), analyses=("2024-2025",), fetch=_fetch({"2024-2025": _scores()}),
            out_path=tmp_path / "bad.csv", log=lambda *_: None)
    assert not (tmp_path / "bad.csv").exists()


def test_the_economics_rate_is_over_the_hours_without_the_partial_edge_days():
    """es.run_economics relabels the partial days at the edges of a UtcWindow deployment (an edge day with fewer than 23 rows) before the policies run, so
    its alert rate leaves them out. Node A: two edge days of 5 rows (1 alert each) and 2 whole days of 24 rows (2 alerts each): the rate over all hours is
    6 / 58 and over the economics' hours 4 / 48."""
    from es.run_driftpath import economics_hours
    frame = pd.concat([_alerts("MISO", "A", [D(2025, 1, 1)], [1], hours=5), _alerts("MISO", "A", [D(2025, 1, 2), D(2025, 1, 3)], [2, 2]),
                       _alerts("MISO", "A", [D(2025, 1, 4)], [1], hours=5)], ignore_index=True)
    assert frame.alert.mean() == pytest.approx(6 / 58)
    assert economics_hours(frame, trim_edges=True).alert.mean() == pytest.approx(4 / 48)
    assert economics_hours(frame, trim_edges=False).alert.mean() == pytest.approx(6 / 58)       # a DateWindow deployment has no partial day


def test_a_missing_economics_row_is_an_error_not_a_pass(tmp_path):
    from es.run_driftpath import economics_alert_rate
    src = _sources(tmp_path)
    assert economics_alert_rate(src, "2024-2025", "P4") == pytest.approx(134 / 816)
    with pytest.raises(LookupError, match="2025-H2"):
        economics_alert_rate(src, "2025-H2", "P4")


# ---------------------------------------------------------------------------------------------------------------------
# the file
# ---------------------------------------------------------------------------------------------------------------------
def _run(tmp_path, name="drift_path.csv", **kw):
    """One run over the two-node hand case. Its deployment is three weeks long, so the windows of the summary start at the first week (min_weeks=1; T1b)."""
    from es.run_driftpath import run
    lines = []
    kw.setdefault("min_weeks", 1)
    frame = run(None, _sources(tmp_path), analyses=("2024-2025",), fetch=_fetch({"2024-2025": _scores()}), out_path=tmp_path / name, log=lines.append, **kw)
    return frame, lines


def test_the_file_has_the_columns_of_the_brief_one_row_per_node_and_week_and_policy(tmp_path):
    frame, _ = _run(tmp_path)
    on_disk = pd.read_csv(tmp_path / "drift_path.csv")
    assert list(on_disk.columns) == ["analysis", "policy", "market", "node", "week_start", "n_hours", "alerts", "rate", "R", "full_week"]
    assert len(on_disk) == 2 * 2 * 3                       # 2 policies x 2 nodes x 3 weeks (23 Dec, 30 Dec, 6 Jan)
    assert set(on_disk.analysis) == {"2024-2025"} and set(on_disk.policy) == {"P1", "P4"}
    a = on_disk[(on_disk.policy == "P1") & (on_disk.node == "A")]
    assert list(a.week_start) == ["2024-12-23", "2024-12-30", "2025-01-06"] and list(a.alerts) == [72, 35, 56]
    assert list(a.full_week) == [False, True, True]
    assert on_disk.R.tolist() == pytest.approx((on_disk.rate / 0.05).tolist())
    assert list(frame.columns) == list(on_disk.columns)


def test_the_rows_are_in_a_fixed_order_and_two_runs_write_the_same_bytes(tmp_path):
    _run(tmp_path, "one.csv")
    _run(tmp_path, "two.csv")
    assert (tmp_path / "one.csv").read_bytes() == (tmp_path / "two.csv").read_bytes()
    d = pd.read_csv(tmp_path / "one.csv")
    assert list(d.policy[:6]) == ["P1"] * 6 and list(d.node[:6]) == ["A"] * 3 + ["B"] * 3         # policy, then market, node, week


def test_the_log_names_each_check_that_matched(tmp_path):
    _, lines = _run(tmp_path)
    text = "\n".join(map(str, lines))
    assert "Table 3" in text and "1 of 2" in text           # the P1 failures against Table 3
    assert "per-node" in text and "economics" in text


def test_writing_leaves_no_temporary_file_behind(tmp_path):
    _run(tmp_path)
    assert sorted(p.name for p in tmp_path.glob("drift_path*")) == ["drift_path.csv"]
    assert sorted(p.name for p in tmp_path.glob("drift_summary*")) == ["drift_summary.csv"]          # (T1b) the summary, next to the weekly path


def test_the_summary_file_is_written_next_to_the_weekly_path_with_the_values_worked_out_by_hand(tmp_path):
    """The two-node case (the weekly paths of test_the_fixed_cutoff_of_node_a_by_hand and its rolling twin) with the windows of up to eight weeks from the first
    week. Weekly hours: A and B 72, 168, 168. P1 alerts: A 72, 35, 56; B 0, 10, 10. A window of one week needs 84 hours and the first week has 72, so the window
    of week 1 has no node and is not counted; weeks 1-2: 240 hours (0.05 x 240 = 12), A 107 alerts R = 8.9167, B 10 alerts R = 0.8333; weeks 1-3: 408 hours (20.4),
    A 163 alerts R = 7.9902, B 20 alerts R = 0.9804. So 2 windows, with one node outside (A) in each: mean 1.0, minimum 1, maximum 1; the medians are
    (107 + 10) / 24 = 4.875 and (163 + 20) / 40.8 = 4.4853. Over the whole period A is above the band (7.99), B inside it: 2 nodes, 1 outside, 1 above, 0 below.
    P4 alerts of A are 72, 14, 28: weeks 1-2: 86 alerts, 86 / 12 = 7.1667, median (86 + 10) / 24 = 4.0; weeks 1-3: 114 / 20.4 = 5.5882, median
    (114 + 20) / 40.8 = 3.2843. Whole period: A 5.59 above, B inside."""
    _run(tmp_path)
    on_disk = pd.read_csv(tmp_path / "drift_summary.csv")
    assert list(on_disk.columns) == ["analysis", "policy", "metric", "value"] and len(on_disk) == 20 and set(on_disk.analysis) == {"2024-2025"}
    for policy, medians in (("P1", (183 / 40.8, 4.875)), ("P4", (134 / 40.8, 4.0))):
        got = dict(zip(*on_disk[on_disk.policy == policy][["metric", "value"]].to_numpy().T))
        assert got == pytest.approx({"nodes": 2, "nodes_outside": 1, "nodes_above": 1, "nodes_below": 0, "window_n": 2, "window_outside_mean": 1.0,
                                     "window_outside_min": 1, "window_outside_max": 1, "window_median_min": medians[0], "window_median_max": medians[1]}), policy


def test_two_runs_write_the_same_summary_bytes_and_the_log_names_the_windows_and_the_checks(tmp_path):
    _, lines = _run(tmp_path, "one.csv", summary_path=tmp_path / "s1.csv")
    _run(tmp_path, "two.csv", summary_path=tmp_path / "s2.csv")
    assert (tmp_path / "s1.csv").read_bytes() == (tmp_path / "s2.csv").read_bytes() and (tmp_path / "one.csv").read_bytes() == (tmp_path / "two.csv").read_bytes()
    text = "\n".join(map(str, lines))
    assert "[windows] 2024-2025 P1: 2 windows of up to 8 weeks (from week 1)" in text and "[windows] 2024-2025 P4: 2 windows" in text
    assert "[check summary] 2024-2025 P1: 1 above + 0 below = 1 of 2 nodes outside" in text and "[check summary vs Table 3] 2024-2025 P1" in text
    assert "[write] s1.csv: 20 rows" in text


@pytest.mark.parametrize("edits,message", [
    ({"nodes_below": 1}, r"nodes_above \(1\) \+ nodes_below \(1\) is not nodes_outside \(1\)"),       # 1 above + 1 below is not the 1 outside
    ({"nodes_outside": 2, "nodes_above": 2}, "Table 3")])                        # 2 above + 0 below = 2 outside is consistent, but Table 3 has 1 of 2
def test_a_summary_that_fails_its_check_stops_the_run_before_either_file_is_written(tmp_path, monkeypatch, edits, message):
    """The weekly checks pass (Table 3 says 1 of 2 and the weekly sums give 1 of 2), then the rows of the summary are checked: above + below must be outside, and P1's
    outside must be Table 3's. The summary is doctored here after it is made. Neither file is written when one of the two checks fails."""
    import es.run_driftpath as module
    real = module.drift_summary_frame

    def doctored(*args, **kwargs):
        s = real(*args, **kwargs)
        for metric, value in edits.items():
            s.loc[(s.policy == "P1") & (s.metric == metric), "value"] = value
        return s

    monkeypatch.setattr(module, "drift_summary_frame", doctored)
    with pytest.raises(module.DriftPathError, match=message):
        module.run(None, _sources(tmp_path), analyses=("2024-2025",), fetch=_fetch({"2024-2025": _scores()}), out_path=tmp_path / "p.csv", min_weeks=1,
                   log=lambda *_: None)
    assert not (tmp_path / "p.csv").exists() and not (tmp_path / "drift_summary.csv").exists()
    assert sorted(p.name for p in tmp_path.glob("*.tmp")) == []


def test_a_window_too_short_for_the_paper_stops_a_run_with_no_file(tmp_path):
    """The two-node case has three weeks, fewer than the four of the paper's first window: the run stops, and nothing is written."""
    from es.run_driftpath import DriftPathError, run
    with pytest.raises(DriftPathError, match="no window of 4 to 8 weeks"):
        run(None, _sources(tmp_path), analyses=("2024-2025",), fetch=_fetch({"2024-2025": _scores()}), out_path=tmp_path / "p.csv", log=lambda *_: None)
    assert not (tmp_path / "p.csv").exists() and not (tmp_path / "drift_summary.csv").exists()


# ---------------------------------------------------------------------------------------------------------------------
# the real tree (read only): the file on disk satisfies the checks that the script made
# ---------------------------------------------------------------------------------------------------------------------
REAL_OUT = Path(__file__).resolve().parents[2] / "outputs" / "es"
needs_drift_path = pytest.mark.skipif(not (REAL_OUT / "paper" / "drift_path.csv").exists() or not (REAL_OUT / "legacy" / "summary.csv").exists(),
                                      reason="outputs/es/paper/drift_path.csv has not been generated (python -m es.run_driftpath)")


@needs_drift_path
def test_the_drift_path_on_disk_reproduces_table_3_and_the_per_node_files_behind_it():
    from es.paper_macros import Sources
    from es.run_driftpath import ANALYSES, SOURCE, failing_nodes, table3_failures, whole_period_r
    src = Sources(REAL_OUT)
    path = src.frame(SOURCE)
    assert set(path.analysis) == set(ANALYSES) and set(path.policy) == {"P1", "P4"}
    assert path.week_start.max() <= "2026-06-30"
    for analysis in ANALYSES:
        sums = whole_period_r(path[(path.analysis == analysis) & (path.policy == "P1")])
        failures, nodes = table3_failures(src, analysis)
        assert (failing_nodes(sums), len(sums)) == (failures, nodes)
        legacy = src.frame(f"legacy/{analysis}__G1_24__xgb__s42__P1__a0.05.csv").set_index(["market", "node"])
        got = sums.set_index(["market", "node"])
        assert (got.alerts == legacy.alerts.reindex(got.index)).all() and (got.n_hours == legacy.n.reindex(got.index)).all()
        assert got.R.to_numpy() == pytest.approx(legacy.R.reindex(got.index).to_numpy(), rel=1e-12)


# ---- the summary and the windows of the real tree (read only) ----------------------------------------------------------
REAL_PAPER = Path(__file__).resolve().parents[3] / "paper"
needs_drift_summary = pytest.mark.skipif(
    not all((REAL_OUT / f).exists() for f in ("paper/drift_path.csv", "paper/drift_summary.csv", "legacy/summary.csv")),
    reason="outputs/es/paper/drift_summary.csv has not been generated (python -m es.run_driftpath)")


def _real_weekly_by_key():
    from es.paper_macros import Sources
    from es.run_driftpath import SOURCE
    path = Sources(REAL_OUT).frame(SOURCE)
    return {key: group for key, group in path.groupby(["analysis", "policy"], sort=False)}


@needs_drift_summary
def test_the_summary_on_disk_is_the_summary_of_the_path_on_disk():
    from es.paper_macros import Sources
    from es.run_driftpath import SUMMARY_SOURCE, drift_summary_frame
    on_disk, again = Sources(REAL_OUT).frame(SUMMARY_SOURCE), drift_summary_frame(_real_weekly_by_key())
    assert list(on_disk.columns) == ["analysis", "policy", "metric", "value"] and len(on_disk) == 40
    assert list(on_disk[["analysis", "policy", "metric"]].itertuples(index=False, name=None)) == list(again[["analysis", "policy", "metric"]].itertuples(index=False, name=None))
    assert list(on_disk.value) == pytest.approx([float(v) for v in again.value], rel=1e-12)


@needs_drift_summary
def test_the_whole_period_counts_of_the_summary_are_the_controllers_and_the_cells_of_table_3():
    """P1: 12 of 33 nodes outside the band in 2025, all of them above; 8 in 2026-H1, 1 above and 7 below; P4: none, in both periods (the controller's
    counts from the weekly sums of drift_path.csv, and Table 3 for P1)."""
    from es.paper_macros import Sources
    from es.run_driftpath import SUMMARY_SOURCE, table3_failures
    src = Sources(REAL_OUT)
    s = src.frame(SUMMARY_SOURCE)

    def value(analysis, policy, metric):
        return s[(s.analysis == analysis) & (s.policy == policy) & (s.metric == metric)].value.iloc[0]

    for analysis, policy, expected in (("2024-2025", "P1", (12, 12, 0)), ("25H1-26H1", "P1", (8, 1, 7)), ("2024-2025", "P4", (0, 0, 0)), ("25H1-26H1", "P4", (0, 0, 0))):
        assert tuple(value(analysis, policy, m) for m in ("nodes_outside", "nodes_above", "nodes_below")) == expected, (analysis, policy)
        assert value(analysis, policy, "nodes") == 33
    for analysis in ("2024-2025", "25H1-26H1"):
        assert (value(analysis, "P1", "nodes_outside"), value(analysis, "P1", "nodes")) == table3_failures(src, analysis)


@needs_drift_summary
def test_windows_of_exactly_eight_weeks_give_the_numbers_the_controller_found():
    """The controller's own recomputation from drift_path.csv with windows of exactly eight weeks (min_weeks = 8): the number of windows, the smallest and the
    largest median over nodes of the window R (two decimals), and the smallest and largest number of nodes outside the band in a window."""
    from es.run_driftpath import drift_summary_frame
    s = drift_summary_frame(_real_weekly_by_key(), weeks=8, min_weeks=8)
    want = {("2024-2025", "P1"): (46, "0.43", "2.42", 8, 30), ("2024-2025", "P4"): (46, "0.23", "2.31", 0, 24),
            ("25H1-26H1", "P1"): (20, "0.07", "1.07", 8, 30), ("25H1-26H1", "P4"): (20, "0.31", "1.93", 3, 25)}
    for (analysis, policy), (n, med_lo, med_hi, out_lo, out_hi) in want.items():
        v = dict(zip(*s[(s.analysis == analysis) & (s.policy == policy)][["metric", "value"]].to_numpy().T))
        assert (v["window_n"], f"{v['window_median_min']:.2f}", f"{v['window_median_max']:.2f}", v["window_outside_min"], v["window_outside_max"]) == \
            (n, med_lo, med_hi, out_lo, out_hi), (analysis, policy)


@needs_drift_summary
def test_the_first_window_of_the_real_path_ends_with_its_fourth_week_and_the_last_with_the_deployment_window():
    """The weeks of the deployment year start on Monday 30 December 2024 (the window opens on 1 January in UTC), so the fourth week is that of 20 January and its window
    ends on Monday 27 January; the weeks of 2026-H1 start on Monday 29 December 2025, the fourth is that of 19 January, the window ends on Monday 26 January.
    The last window ends the day after the last settlement day, which is the end of the UTC window of the analysis (2026-01-01 and 2026-07-01)."""
    from es.analyses import LEGACY
    from es.run_driftpath import window_r
    for (analysis, policy), weekly in _real_weekly_by_key().items():
        w = window_r(weekly)
        first = {"2024-2025": "2025-01-27", "25H1-26H1": "2026-01-26"}[analysis]
        assert w.week_end.min() == pd.Timestamp(first) and w.week_end.max() == pd.Timestamp(LEGACY[analysis].deploy.end), (analysis, policy)
        assert sorted(w.weeks.unique()) == [4, 5, 6, 7, 8]
        assert w.groupby("week_end").size().min() >= 1


# ---------------------------------------------------------------------------------------------------------------------
# the macros of the summary and the constants of the window (registry: _drift_entries and the two code readers)
# ---------------------------------------------------------------------------------------------------------------------
#: macro prefix -> (metric, format)
DRIFT_MACROS = {"driftOut": ("nodes_outside", "int"), "driftAbove": ("nodes_above", "int"), "driftBelow": ("nodes_below", "int"),
                "driftWinOutMean": ("window_outside_mean", "dec1"), "driftWinOutMin": ("window_outside_min", "int"),
                "driftWinOutMax": ("window_outside_max", "int"), "driftWinMedMin": ("window_median_min", "dec2"),
                "driftWinMedMax": ("window_median_max", "dec2"), "driftWinN": ("window_n", "int")}
#: (policy word, period word) -> (policy, analysis): POne is P1 and PFour is P4; YY is the deployment year 2025 and PostOpen is 2026-H1
DRIFT_GROUPS = {("POne", "YY"): ("P1", "2024-2025"), ("POne", "PostOpen"): ("P1", "25H1-26H1"),
                ("PFour", "YY"): ("P4", "2024-2025"), ("PFour", "PostOpen"): ("P4", "25H1-26H1")}
#: A small summary file with a different number in every cell that a macro reads (nothing in it is a real result), and the macros it must give, worked out by hand
#: (half up: 18.78 -> 18.8, 0.225847 -> 0.23, 0.215 -> 0.22, 2.305 -> 2.31, 19.45 -> 19.5, 1.345 -> 1.35, 13.04 -> 13.0, 1.935 -> 1.94).
SMALL = {("P1", "2024-2025"): {"nodes": 33, "nodes_outside": 12, "nodes_above": 11, "nodes_below": 1, "window_n": 50, "window_outside_mean": 18.78,
                               "window_outside_min": 8, "window_outside_max": 30, "window_median_min": 0.225847, "window_median_max": 2.423336},
         ("P4", "2024-2025"): {"nodes": 33, "nodes_outside": 6, "nodes_above": 4, "nodes_below": 2, "window_n": 51, "window_outside_mean": 9.84,
                               "window_outside_min": 1, "window_outside_max": 24, "window_median_min": 0.215, "window_median_max": 2.305},
         ("P1", "25H1-26H1"): {"nodes": 33, "nodes_outside": 8, "nodes_above": 1, "nodes_below": 7, "window_n": 24, "window_outside_mean": 19.45,
                               "window_outside_min": 9, "window_outside_max": 31, "window_median_min": 0.07, "window_median_max": 1.345},
         ("P4", "25H1-26H1"): {"nodes": 33, "nodes_outside": 3, "nodes_above": 2, "nodes_below": 1, "window_n": 25, "window_outside_mean": 13.04,
                               "window_outside_min": 3, "window_outside_max": 25, "window_median_min": 0.31, "window_median_max": 1.935}}
SMALL_MACROS = {("POne", "YY"): {"driftOut": "12", "driftAbove": "11", "driftBelow": "1", "driftWinOutMean": "18.8", "driftWinOutMin": "8", "driftWinOutMax": "30",
                                 "driftWinMedMin": "0.23", "driftWinMedMax": "2.42", "driftWinN": "50"},
                ("PFour", "YY"): {"driftOut": "6", "driftAbove": "4", "driftBelow": "2", "driftWinOutMean": "9.8", "driftWinOutMin": "1", "driftWinOutMax": "24",
                                  "driftWinMedMin": "0.22", "driftWinMedMax": "2.31", "driftWinN": "51"},
                ("POne", "PostOpen"): {"driftOut": "8", "driftAbove": "1", "driftBelow": "7", "driftWinOutMean": "19.5", "driftWinOutMin": "9", "driftWinOutMax": "31",
                                       "driftWinMedMin": "0.07", "driftWinMedMax": "1.35", "driftWinN": "24"},
                ("PFour", "PostOpen"): {"driftOut": "3", "driftAbove": "2", "driftBelow": "1", "driftWinOutMean": "13.0", "driftWinOutMin": "3", "driftWinOutMax": "25",
                                        "driftWinMedMin": "0.31", "driftWinMedMax": "1.94", "driftWinN": "25"}}


def _small_sources(tmp_path, drop=None):
    """Sources over a folder that holds only paper/drift_summary.csv (the SMALL table, one row per cell; `drop` is a (policy, analysis, metric) left out)."""
    from es.paper_macros import Sources
    out = tmp_path / "out"
    (out / "paper").mkdir(parents=True)
    lines = ["analysis,policy,metric,value"]
    for (policy, analysis), values in SMALL.items():
        lines += [f"{analysis},{policy},{metric},{value}" for metric, value in values.items() if (policy, analysis, metric) != drop]
    (out / "paper" / "drift_summary.csv").write_text("\n".join(lines) + "\n")
    return Sources(out, tmp_path / "paper", tmp_path / "docs")


def _drift_names():
    return {prefix + pol + per for prefix in DRIFT_MACROS for pol, per in DRIFT_GROUPS}


def test_the_registry_has_the_36_macros_of_the_summary_and_the_two_window_constants():
    from es.paper_registry import ENTRIES, entries_by_name
    by = entries_by_name()
    names = _drift_names()
    assert len(names) == 36 and names <= set(by) and {n for n in by if n.startswith("drift")} == names | {"driftWindowWeeks", "driftWindowMinWeeks"}
    assert {e.name for e in ENTRIES if e.source == "paper/drift_summary.csv"} == names
    for (pol_word, per_word), (policy, analysis) in DRIFT_GROUPS.items():
        for prefix, (metric, fmt) in DRIFT_MACROS.items():
            e = by[prefix + pol_word + per_word]
            assert (e.source, e.column, e.fmt, e.stage, e.agg) == ("paper/drift_summary.csv", "value", fmt, "dev", "one"), e.name
            assert e.query == f'analysis == "{analysis}" and policy == "{policy}" and metric == "{metric}"', e.name
    for name in ("driftWindowWeeks", "driftWindowMinWeeks"):
        assert (by[name].source, by[name].stage, by[name].fmt) == (f"code:{name}", "const", "int")


def test_each_macro_reads_the_row_of_its_own_analysis_policy_and_metric_and_prints_it_rounded_half_up(tmp_path):
    from es.paper_macros import resolve
    from es.paper_registry import entries_by_name
    by, src = entries_by_name(), _small_sources(tmp_path)
    for (pol_word, per_word), expected in SMALL_MACROS.items():
        for prefix, value in expected.items():
            r = resolve(by[prefix + pol_word + per_word], src)
            assert (r["status"], r["value"]) == ("ok", value), (prefix + pol_word + per_word, r)
            assert r["sha"] == src.sha("paper/drift_summary.csv") and r["sha"] != ""


def test_a_macro_of_a_summary_that_is_not_there_is_pending_and_names_the_file_by_a_path_in_the_project(tmp_path):
    from es.paper_macros import PENDING, Sources, resolve
    from es.paper_registry import entries_by_name
    by, src = entries_by_name(), Sources(tmp_path / "out", tmp_path / "paper", tmp_path / "docs")
    for name in sorted(_drift_names()):
        r = resolve(by[name], src)
        assert (r["status"], r["value"], r["detail"]) == ("pending", PENDING["dev"], "missing outputs/es/paper/drift_summary.csv"), name
    assert src.shown("paper/drift_summary.csv") == "outputs/es/paper/drift_summary.csv"


def test_a_summary_without_the_row_of_a_macro_is_an_error_that_names_the_macro(tmp_path):
    from es.paper_macros import resolve
    from es.paper_registry import entries_by_name
    src = _small_sources(tmp_path, drop=("P4", "25H1-26H1", "window_n"))
    with pytest.raises(LookupError, match="driftWinNPFourPostOpen"):
        resolve(entries_by_name()["driftWinNPFourPostOpen"], src)
    assert resolve(entries_by_name()["driftWinNPFourYY"], src)["value"] == "51"                     # the other rows are read as before


def test_the_window_constants_are_read_from_the_code_when_the_macro_is_resolved(tmp_path, monkeypatch):
    import hashlib

    import es.run_driftpath as module
    from es.paper_macros import Sources, resolve
    from es.paper_registry import entries_by_name
    by, src = entries_by_name(), Sources(tmp_path / "out", tmp_path / "paper", tmp_path / "docs")
    got = {n: resolve(by[n], src) for n in ("driftWindowWeeks", "driftWindowMinWeeks")}
    assert (got["driftWindowWeeks"]["value"], got["driftWindowMinWeeks"]["value"]) == ("8", "4") and {r["status"] for r in got.values()} == {"ok"}
    digest = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()[:16]                    # hashed with the module that defines it
    assert {r["sha"] for r in got.values()} == {digest}
    monkeypatch.setattr(module, "WINDOW_WEEKS", 6)
    monkeypatch.setattr(module, "WINDOW_MIN_WEEKS", 3)
    assert (resolve(by["driftWindowWeeks"], src)["value"], resolve(by["driftWindowMinWeeks"], src)["value"]) == ("6", "3")      # not a snapshot


def test_the_generated_files_name_the_summary_by_a_path_relative_to_outputs_and_hold_no_absolute_path(tmp_path):
    from es.paper_macros import resolve_all, write_outputs
    from es.paper_registry import entries_by_name
    by = entries_by_name()
    names = sorted(_drift_names()) + ["driftWindowWeeks", "driftWindowMinWeeks"]
    generated = tmp_path / "generated"
    write_outputs(resolve_all([by[n] for n in names], _small_sources(tmp_path)), generated, {"A": False, "B": False, "Adidnotrun": False, "Bdidnotrun": False})
    tex = (generated / "macros.tex").read_text()
    assert "\\newcommand{\\driftOutPOneYY}{12}" in tex and "\\newcommand{\\driftWinMedMaxPFourPostOpen}{1.94}" in tex and "\\newcommand{\\driftWindowWeeks}{8}" in tex
    provenance, registry = pd.read_csv(generated / "provenance.csv", dtype=str, keep_default_na=False), pd.read_csv(generated / "registry.csv", dtype=str, keep_default_na=False)
    assert set(provenance.name) == set(names) and set(names) <= set(registry.name)                  # provenance.csv has the macros that were resolved, registry.csv all of them
    for table in (provenance, registry):
        ours = table[table.name.isin(names)]
        rows = ours[ours.name.isin(_drift_names())]
        assert set(rows.source) == {"paper/drift_summary.csv"} and len(rows) == 36 and len(ours) == 38
        assert not any(str(cell).startswith("/") or "/Users/" in str(cell) for column in ours.columns for cell in ours[column])        # no absolute path in any cell


REAL_MACROS = REAL_PAPER / "generated" / "macros.tex"


def _real_macro(name):
    import re
    m = re.search(r"\\newcommand\{\\" + name + r"\}\{(.*)\}\s*$", REAL_MACROS.read_text(), re.M)
    assert m, f"\\{name} is not in paper/generated/macros.tex: run paper/build.sh macros"
    return m.group(1)


@pytest.mark.skipif(not REAL_MACROS.exists(), reason="paper/generated/macros.tex has not been built (paper/build.sh macros)")
def test_the_whole_period_macros_of_p1_are_the_counts_of_table_3_in_the_real_macros_tex():
    assert _real_macro("driftOutPOneYY") == _real_macro("failCombYY") and _real_macro("driftOutPOnePostOpen") == _real_macro("failCombPostOpen")
    assert (_real_macro("driftOutPOneYY"), _real_macro("driftAbovePOneYY"), _real_macro("driftBelowPOneYY")) == ("12", "12", "0")
    assert (_real_macro("driftOutPOnePostOpen"), _real_macro("driftAbovePOnePostOpen"), _real_macro("driftBelowPOnePostOpen")) == ("8", "1", "7")
    for period in ("YY", "PostOpen"):
        assert (_real_macro(f"driftOutPFour{period}"), _real_macro(f"driftAbovePFour{period}"), _real_macro(f"driftBelowPFour{period}")) == ("0", "0", "0")
    assert (_real_macro("driftWindowWeeks"), _real_macro("driftWindowMinWeeks")) == ("8", "4")


@needs_drift_summary
@pytest.mark.skipif(not REAL_MACROS.exists(), reason="paper/generated/macros.tex has not been built (paper/build.sh macros)")
def test_every_real_macro_of_the_summary_is_its_value_in_the_file_rounded_half_up():
    """Rounded by an independent routine (decimal, half up), from the file on disk, to the figures the registry prints: one decimal for the mean, two for the
    medians and none for the counts."""
    from decimal import ROUND_HALF_UP, Decimal
    from es.paper_macros import Sources
    from es.run_driftpath import SUMMARY_SOURCE
    s = Sources(REAL_OUT).frame(SUMMARY_SOURCE)
    for (pol_word, per_word), (policy, analysis) in DRIFT_GROUPS.items():
        rows = s[(s.analysis == analysis) & (s.policy == policy)]
        for prefix, (metric, fmt) in DRIFT_MACROS.items():
            raw = Decimal(repr(float(rows[rows.metric == metric].value.iloc[0])))
            places = {"int": "1", "dec1": "0.1", "dec2": "0.01"}[fmt]
            assert _real_macro(prefix + pol_word + per_word) == f"{raw.quantize(Decimal(places), rounding=ROUND_HALF_UP):f}", (prefix, pol_word, per_word)
