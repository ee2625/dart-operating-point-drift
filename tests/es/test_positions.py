import datetime as dt
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _train():
    return pd.DataFrame({"market": "MISO", "node": "A", "settle_hour": [0, 0, 1, 1],
                         "dart_spread": [2.0, 1.0, -3.0, -1.0], "dart_settle": [-2.0, -1.0, 3.0, 1.0],
                         "abs_mean_168h": [1, 2, 3, 4]})


def _days(abs_means, market="MISO", node="A", first_day=dt.date(2025, 3, 1)):
    """Hourly rows of consecutive settlement days: abs_means[i][h] is abs_mean_168h at hour h of day i."""
    return pd.DataFrame([
        {"market": market, "node": node, "settle_date": first_day + dt.timedelta(days=i), "settle_hour": h,
         "timestamp_utc": pd.Timestamp(first_day + dt.timedelta(days=i), tz="UTC") + pd.Timedelta(hours=h),
         "dart_settle": 0.0, "abs_mean_168h": value}
        for i, day in enumerate(abs_means) for h, value in enumerate(day)])


def _stress_table(thresholds):
    """What stress_threshold returns, built from {(market, node): threshold}."""
    index = pd.MultiIndex.from_tuples(list(thresholds), names=["market", "node"])
    return pd.Series(list(thresholds.values()), index=index, name="stress_thr")


def _direction(d_by_key):
    """What premium_direction returns, built from {(market, node, hour): d}."""
    index = pd.MultiIndex.from_tuples(list(d_by_key), names=["market", "node", "settle_hour"])
    return pd.Series(list(d_by_key.values()), index=index, name="d")


# --- direction, fees, stress and exposure: the plan's tests (the stress one is per node-day, spec §4.2) --

def test_premium_direction_sign_of_training_mean():
    from es.positions import premium_direction
    d = premium_direction(_train())
    assert d[("MISO", "A", 0)] == 1 and d[("MISO", "A", 1)] == -1


def test_premium_direction_on_settlement_column():
    from es.positions import premium_direction
    d = premium_direction(_train(), col="dart_settle")
    assert d[("MISO", "A", 0)] == -1 and d[("MISO", "A", 1)] == 1


def test_calibrated_fees_are_side_specific():
    from es.positions import attach_pnl, premium_direction
    rows = pd.DataFrame({"market": "MISO", "node": "A", "settle_hour": [0, 1],
                         "dart_settle": [10.0, 10.0], "abs_mean_168h": [0.0, 0.0]})
    out = attach_pnl(rows, "premium", premium_direction(_train()))
    assert np.allclose(out["fee"], [0.15, 0.05])
    assert np.allclose(out["pnl_open"], [10 - 0.15, -10 - 0.05])


def test_uniform_fee_and_bookends():
    from es.positions import attach_pnl
    rows = pd.DataFrame({"market": "NYISO", "node": "A", "settle_hour": [0],
                         "dart_settle": [5.0], "abs_mean_168h": [0.0]})
    assert attach_pnl(rows, "inc", fees=0.5)["pnl_open"].iloc[0] == 4.5
    assert attach_pnl(rows, "dec", fees=0.5)["pnl_open"].iloc[0] == -5.5


def test_stress_doubles_fee_on_volatile_days():
    from es.positions import attach_pnl, stress_threshold
    thr = stress_threshold(_train())                      # MISO/A: 90th percentile of 1, 2, 3, 4 = 3.7
    # A day's first hour decides all of it. Day 1 opens calm and turns volatile later (1, 10, 10), day 2
    # opens volatile and calms down (10, 1, 1): day 1 pays the base fee all day, day 2 pays double all day.
    out = attach_pnl(_days([[1.0, 10.0, 10.0], [10.0, 1.0, 1.0]]), "inc", fees=0.10, stress=thr)
    assert out["fee"].tolist() == [0.10] * 3 + [0.20] * 3


def test_spike_exposure_from_skew():
    from es.positions import premium_direction, spike_exposure
    tr = pd.DataFrame({"market": "M", "node": "A", "settle_hour": 0,
                       "dart_spread": [1, 1, 1, 1, -20.0], "abs_mean_168h": 0.0})
    d = premium_direction(tr)[("M", "A", 0)]
    expected = "short_spikes" if pd.Series(d * tr["dart_spread"]).skew() < 0 else "long_spikes"
    assert spike_exposure(tr, premium_direction(tr))[("M", "A", 0)] == expected


# --- direction, exposure and threshold, per node-hour ---------------------------------------------------

def test_premium_direction_follows_the_mean_not_the_median():
    from es.positions import premium_direction
    # The means are +2.67 (node A) and -2.67 (node B); the medians are -1 and +1.
    tr = pd.DataFrame({"market": "M", "node": ["A"] * 3 + ["B"] * 3, "settle_hour": 0,
                       "dart_spread": [-1.0, -1.0, 10.0, 1.0, 1.0, -10.0]})
    assert premium_direction(tr).to_dict() == {("M", "A", 0): 1, ("M", "B", 0): -1}


def test_premium_direction_is_per_node_and_hour_and_a_zero_mean_is_an_inc():
    from es.positions import premium_direction
    tr = pd.DataFrame({"market": ["M"] * 6 + ["N"] * 2, "node": ["A"] * 4 + ["B"] * 2 + ["A"] * 2,
                       "settle_hour": [0, 0, 1, 1, 0, 0, 0, 0],
                       "dart_spread": [1.0, -1.0, -2.0, -2.0, 5.0, 5.0, -5.0, -5.0]})
    d = premium_direction(tr)
    assert d.to_dict() == {("M", "A", 0): 1, ("M", "A", 1): -1, ("M", "B", 0): 1, ("N", "A", 0): -1}
    assert list(d.index.names) == ["market", "node", "settle_hour"] and d.name == "d"


def test_stress_threshold_is_the_training_90th_percentile_per_node():
    from es.positions import stress_threshold
    tr = pd.DataFrame({"market": ["M"] * 10 + ["N"] * 5, "node": ["A"] * 5 + ["B"] * 5 + ["A"] * 5,
                       "abs_mean_168h": [0, 1, 2, 3, 4, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500]})
    thr = stress_threshold(tr)
    assert thr.to_dict() == pytest.approx({("M", "A"): 3.6, ("M", "B"): 46.0, ("N", "A"): 460.0})
    assert list(thr.index.names) == ["market", "node"] and thr.name == "stress_thr"


def test_spike_exposure_names_short_and_long_spikes_by_the_skew_of_d_times_dart():
    from es.positions import premium_direction, spike_exposure
    # (node, hour): (training spreads, expected). d * spread is the position's own P&L before fees.
    cases = {
        ("A", 0): ([3.0, 3.0, 3.0, 3.0, -8.0], "short_spikes"),      # INC (mean 0.8); rare big loss: skew < 0
        ("A", 1): ([-1.0, -1.0, -1.0, -1.0, 20.0], "long_spikes"),   # INC (mean 3.2); rare big gain: skew > 0
        ("B", 0): ([-3.0, -3.0, -3.0, -3.0, 8.0], "short_spikes"),   # DEC (mean -0.8); d * x = 3, 3, 3, 3, -8
        ("B", 1): ([1.0, 1.0, 1.0, 1.0, -20.0], "long_spikes"),      # DEC (mean -3.2); d * x = -1, -1, -1, -1, 20
        ("A", 2): ([5.0, 5.0, 5.0, 5.0, 0.1], "short_spikes"),       # INC (mean 4.02); a rare dip: skew < 0
        ("C", 0): ([4.0, 4.0, 3.0, 2.0, 5.0], "short_spikes"),       # INC (mean 3.6); any negative skew counts: -0.40
        ("C", 1): ([6.0, 6.0, 7.0, 8.0, 5.0], "long_spikes"),        # INC (mean 6.4); a mild positive skew: +0.40
    }
    tr = pd.concat([pd.DataFrame({"market": "M", "node": n, "settle_hour": h, "dart_spread": v})
                    for (n, h), (v, _) in cases.items()], ignore_index=True)
    out = spike_exposure(tr, premium_direction(tr))
    assert out.to_dict() == {("M", n, h): label for (n, h), (_, label) in cases.items()}
    assert list(out.index.names) == ["market", "node", "settle_hour"] and out.name == "exposure"


def test_spike_exposure_on_settlement_column():
    from es.positions import premium_direction, spike_exposure
    tr = pd.DataFrame({"market": "M", "node": "A", "settle_hour": 0,
                       "dart_spread": [3.0, 3.0, 3.0, 3.0, -8.0],        # an INC with a rare big loss
                       "dart_settle": [-1.0, -1.0, -1.0, -1.0, 20.0]})   # an INC with a rare big gain
    assert spike_exposure(tr, premium_direction(tr))[("M", "A", 0)] == "short_spikes"
    d = premium_direction(tr, col="dart_settle")
    assert spike_exposure(tr, d, col="dart_settle")[("M", "A", 0)] == "long_spikes"


def test_a_symmetric_sample_with_zero_skew_counts_as_long_spikes():
    from es.positions import premium_direction, spike_exposure
    tr = pd.DataFrame({"market": "M", "node": "A", "settle_hour": 0, "dart_spread": [-5.0, -1.0, 0.0, 1.0, 5.0]})
    assert spike_exposure(tr, premium_direction(tr))[("M", "A", 0)] == "long_spikes"


# --- fees: every market and side, and the uniform grid --------------------------------------------------

def test_calibrated_fee_dict_is_pinned_exactly():
    """Rulings 78 and 90: every level is the latest value published on or before 2026-06-30, rounded to 0.05
    (docs/fee_rederivation_2026-10-23.md). MISO omits the Schedule 17 and 24 adders, because no posting dated on or
    before June 2026 could be retrieved. A change to any level needs a new ruling."""
    from es.positions import CALIBRATED_FEES
    assert CALIBRATED_FEES == {"NYISO": {1: 0.20, -1: 0.15}, "ISONE": {1: 0.70, -1: 0.70},
                               "MISO": {1: 0.15, -1: 0.05}, "CAISO": {1: 0.55, -1: 0.15}}


@pytest.mark.parametrize("market,inc,dec", [("NYISO", 0.20, 0.15), ("ISONE", 0.70, 0.70),
                                            ("MISO", 0.15, 0.05), ("CAISO", 0.55, 0.15)])
def test_calibrated_fee_table_for_every_market_and_side(market, inc, dec):
    from es.positions import attach_pnl
    rows = pd.DataFrame({"market": market, "node": "A", "settle_hour": [0, 1],
                         "dart_settle": [10.0, -4.0], "abs_mean_168h": 0.0})
    for position, d, fee in (("inc", 1, inc), ("dec", -1, dec)):
        out = attach_pnl(rows, position)
        assert out["d"].tolist() == [d, d] and out["fee"].tolist() == [fee, fee]
        assert np.allclose(out["pnl_open"], [d * 10.0 - fee, d * -4.0 - fee])


def test_uniform_fee_grid_applies_to_every_market_and_side():
    from es.positions import UNIFORM_FEES, attach_pnl
    assert UNIFORM_FEES == (0.0, 0.10, 0.25, 0.50, 1.00)
    rows = pd.DataFrame({"market": ["NYISO", "ISONE", "MISO", "CAISO"], "node": "A", "settle_hour": 0,
                         "dart_settle": 2.0, "abs_mean_168h": 0.0})
    for c in UNIFORM_FEES:
        for position, d in (("inc", 1), ("dec", -1)):
            out = attach_pnl(rows, position, fees=c)
            assert out["fee"].tolist() == [c] * 4 and np.allclose(out["pnl_open"], d * 2.0 - c)


def test_an_unknown_position_is_rejected():
    from es.positions import attach_pnl
    rows = pd.DataFrame({"market": "MISO", "node": "A", "settle_hour": [0], "dart_settle": [1.0]})
    with pytest.raises(ValueError, match="both"):
        attach_pnl(rows, "both")


def test_attach_pnl_leaves_its_input_untouched():
    from es.positions import attach_pnl
    rows = _days([[10.0, 1.0]])
    before = rows.copy(deep=True)
    attach_pnl(rows, "inc", fees=0.10, stress=_stress_table({("MISO", "A"): 3.7}))
    pd.testing.assert_frame_equal(rows, before)


def test_attach_pnl_adds_exactly_d_fee_and_pnl_open_and_keeps_everything_else():
    from es.positions import attach_pnl, premium_direction, stress_threshold
    rows = _days([[10.0, 1.0]]).assign(extra="keep me")
    for position, direction in (("inc", None), ("dec", None), ("premium", premium_direction(_train()))):
        out = attach_pnl(rows, position, direction, stress=stress_threshold(_train()))
        assert list(out.columns) == list(rows.columns) + ["d", "fee", "pnl_open"]
        pd.testing.assert_frame_equal(out[list(rows.columns)], rows)


def test_an_empty_result_still_has_float_fee_and_pnl_columns():
    from es.positions import attach_pnl
    stress = _stress_table({("MISO", "A"): 3.7})
    out = attach_pnl(_days([[10.0, 1.0]]).iloc[:0], "inc", fees=0.10, stress=stress)
    assert out.empty and out["fee"].dtype == float and out["pnl_open"].dtype == float
    with pytest.warns(UserWarning, match=r"2 row\(s\) dropped"):              # a direction table that covers nothing
        out = attach_pnl(_days([[10.0, 1.0]]), "premium", _direction({("MISO", "Z", 0): 1}), stress=stress)
    assert out.empty and out["fee"].dtype == float and out["pnl_open"].dtype == float


# --- stress: per node-day, decided by the day's first hour ----------------------------------------------

def test_stress_boundary_is_at_or_above_the_threshold():
    from es.positions import attach_pnl
    t = 3.7
    rows = pd.concat([_days([[t]]), _days([[np.nextafter(t, 0.0)]], first_day=dt.date(2025, 3, 2))],
                     ignore_index=True)
    out = attach_pnl(rows, "inc", fees=0.10, stress=_stress_table({("MISO", "A"): t}))
    assert out["fee"].tolist() == [0.20, 0.10]


def test_stress_doubles_the_side_specific_fee_on_a_stressed_day_only():
    from es.positions import attach_pnl, premium_direction, stress_threshold
    rows = pd.concat([_days([[10.0, 1.0]]), _days([[1.0, 10.0]], first_day=dt.date(2025, 3, 2))],
                     ignore_index=True).assign(dart_settle=10.0)
    out = attach_pnl(rows, "premium", premium_direction(_train()), stress=stress_threshold(_train()))
    # MISO hour 0 is an INC (0.15), hour 1 a DEC (0.05). Day 1 opens volatile, day 2 opens calm.
    assert out["fee"].tolist() == [0.30, 0.10, 0.15, 0.05]
    assert np.allclose(out["pnl_open"], [10 - 0.30, -10 - 0.10, 10 - 0.15, -10 - 0.05])


@pytest.mark.parametrize("values,expected", [
    ([10.0, 1.0, 1.0], 0.20),       # the earliest hour is volatile; the hours listed first are calm
    ([1.0, 10.0, 10.0], 0.10),      # the earliest hour is calm; the hours listed first are volatile
])
def test_the_earliest_timestamp_decides_not_the_row_listed_first(values, expected):
    from es.positions import attach_pnl
    latest_first = _days([values]).iloc[[2, 1, 0]]
    out = attach_pnl(latest_first, "inc", fees=0.10, stress=_stress_table({("MISO", "A"): 3.7}))
    assert out["fee"].tolist() == [expected] * 3


def test_stress_is_judged_per_market_node_and_day():
    from es.positions import attach_pnl
    rows = pd.concat([_days([[10.0, 1.0]], node="A"),                    # opens volatile
                      _days([[1.0, 10.0]], node="B"),                    # opens calm
                      _days([[10.0, 1.0]], market="NYISO", node="A")],   # same name, other market, own threshold
                     ignore_index=True)
    stress = _stress_table({("MISO", "A"): 3.7, ("MISO", "B"): 3.7, ("NYISO", "A"): 50.0})
    out = attach_pnl(rows, "inc", fees=0.10, stress=stress)
    assert out["fee"].tolist() == [0.20, 0.20, 0.10, 0.10, 0.10, 0.10]


def test_a_stressed_day_stays_stressed_when_premium_drops_its_first_hour():
    from es.positions import attach_pnl
    rows = _days([[10.0, 1.0]])                           # hour 0 is volatile but has no direction
    with pytest.warns(UserWarning, match="dropped"):
        out = attach_pnl(rows, "premium", _direction({("MISO", "A", 1): 1}), fees=0.10,
                         stress=_stress_table({("MISO", "A"): 3.7}))
    assert out["settle_hour"].tolist() == [1] and out["fee"].tolist() == [0.20]


@pytest.mark.parametrize("missing", ["settle_date", "timestamp_utc", "abs_mean_168h"])
def test_stress_needs_the_columns_that_define_a_day(missing):
    from es.positions import attach_pnl
    rows = _days([[10.0, 1.0]]).drop(columns=missing)
    with pytest.raises(ValueError, match=rf"\(missing: {missing}\)"):
        attach_pnl(rows, "inc", fees=0.10, stress=_stress_table({("MISO", "A"): 3.7}))
    assert attach_pnl(rows, "inc", fees=0.10)["fee"].tolist() == [0.10, 0.10]    # no stress, no need


# --- nothing is dropped or skipped silently: one UserWarning per call -----------------------------------

def test_a_nan_training_mean_gives_no_direction_and_one_consolidated_warning():
    from es.positions import premium_direction
    nan = np.nan
    # The rows are listed out of order (C, B, A); the warning names the node-hours in key order.
    tr = pd.DataFrame({"market": "MISO", "node": ["C"] * 2 + ["B"] * 2 + ["A"] * 4,
                       "settle_hour": [0, 0, 0, 0, 1, 1, 0, 0],
                       "dart_spread": [nan, nan, nan, 3.0, nan, nan, 2.0, 1.0]})
    with pytest.warns(UserWarning) as rec:
        d = premium_direction(tr)
    (w,) = rec
    msg = str(w.message)
    assert msg.startswith("premium_direction: 2 node-hour(s)") and "'dart_spread'" in msg
    assert msg.endswith("MISO/A/1, MISO/C/0")                      # all-NaN groups only; B/0 has a value
    assert d.isna().to_dict() == {("MISO", "A", 0): False, ("MISO", "A", 1): True,    # NaN: no position, not a DEC
                                  ("MISO", "B", 0): False, ("MISO", "C", 0): True}
    assert d[("MISO", "A", 0)] == 1 and d[("MISO", "B", 0)] == 1    # a NaN among the values is just skipped
    assert Path(w.filename).name == Path(__file__).name             # attributed to the caller


def test_premium_rows_without_a_direction_are_dropped_with_one_warning():
    from es.positions import attach_pnl, premium_direction
    # Directions exist for MISO/A hours 0 and 1 only. Rows 0, 1 and 4 have none and are listed out of order.
    rows = pd.DataFrame({"market": "MISO", "node": ["B", "A", "A", "A", "B"], "settle_hour": [0, 2, 0, 1, 0],
                         "dart_settle": 10.0, "abs_mean_168h": 0.0})
    with pytest.warns(UserWarning) as rec:
        out = attach_pnl(rows, "premium", premium_direction(_train()))
    (w,) = rec
    msg = str(w.message)
    assert msg.startswith("attach_pnl: 3 row(s) dropped") and msg.endswith("has no direction: MISO/A/2, MISO/B/0")
    assert out.index.tolist() == [2, 3] and out["d"].tolist() == [1, -1] and out["d"].dtype.kind == "i"
    assert Path(w.filename).name == Path(__file__).name


def test_a_nan_direction_drops_its_rows_and_names_them():
    from es.positions import attach_pnl
    rows = pd.DataFrame({"market": "MISO", "node": "A", "settle_hour": [0, 1], "dart_settle": 10.0})
    with pytest.warns(UserWarning) as rec:
        out = attach_pnl(rows, "premium", _direction({("MISO", "A", 0): 1.0, ("MISO", "A", 1): np.nan}))
    assert out["settle_hour"].tolist() == [0] and str(rec[0].message).endswith("MISO/A/1")


def test_a_node_without_a_stress_threshold_is_never_stressed_and_is_named_in_one_warning():
    from es.positions import attach_pnl
    rows = pd.concat([_days([[10.0]], node="A"), _days([[10.0]], node="Z"),
                      _days([[10.0]], market="CAISO", node="Y"), _days([[10.0]], node="N")], ignore_index=True)
    stress = _stress_table({("MISO", "A"): 3.7, ("MISO", "N"): np.nan})      # Z and CAISO/Y absent, N is NaN
    with pytest.warns(UserWarning) as rec:
        out = attach_pnl(rows, "inc", fees=0.10, stress=stress)
    (w,) = rec
    msg = str(w.message)
    assert msg.startswith("attach_pnl: no stress threshold for 3 node(s)") and msg.endswith("CAISO/Y, MISO/N, MISO/Z")
    assert "MISO/A" not in msg
    assert out["fee"].tolist() == [0.20, 0.10, 0.10, 0.10]
    assert Path(w.filename).name == Path(__file__).name


def test_a_call_with_both_findings_still_warns_once():
    from es.positions import attach_pnl
    rows = pd.concat([_days([[10.0]], node="A"), _days([[10.0]], node="Y"), _days([[10.0]], node="Z")],
                     ignore_index=True)
    direction = _direction({("MISO", "A", 0): 1, ("MISO", "Y", 0): 1})        # Z has no direction: dropped
    with pytest.warns(UserWarning) as rec:                                    # Y has no threshold: never stressed
        out = attach_pnl(rows, "premium", direction, fees=0.10, stress=_stress_table({("MISO", "A"): 3.7}))
    (w,) = rec
    msg = str(w.message)
    assert msg.startswith("attach_pnl: 1 row(s) dropped") and " | " in msg and msg.endswith("MISO/Y")
    assert "MISO/Z/0" in msg and out["fee"].tolist() == [0.20, 0.10]


def test_nothing_is_said_when_nothing_is_dropped_or_missing():
    from es.positions import attach_pnl, premium_direction, spike_exposure, stress_threshold
    rows = _days([[10.0, 1.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        d = premium_direction(_train())
        spike_exposure(_train(), d)
        thr = stress_threshold(_train())
        attach_pnl(rows, "premium", d, stress=thr)
        attach_pnl(rows, "inc", stress=thr)
        attach_pnl(rows, "dec", fees=0.25)


# --- invariant I4: direction, exposure and threshold are fitted on training rows only -------------------

@pytest.mark.parametrize("other", ["calib", "deploy", "other", np.nan], ids=["calib", "deploy", "other", "nan"])
@pytest.mark.parametrize("name", ["premium_direction", "spike_exposure", "stress_threshold"])
def test_training_side_functions_refuse_rows_outside_the_training_split(name, other):
    import es.positions as P
    clean = _train().assign(split="train")
    calls = {"premium_direction": P.premium_direction,
             "spike_exposure": lambda f: P.spike_exposure(f, P.premium_direction(clean)),
             "stress_threshold": P.stress_threshold}
    with pytest.raises(ValueError) as exc:
        calls[name](clean.assign(split=["train", "train", "train", other]))
    msg = str(exc.value)
    assert msg.startswith(f"{name}: ") and "I4" in msg and str(other) in msg
    calls[name](clean)                                           # all rows are train: fine
    calls[name](clean.drop(columns="split"))                     # no split column at all: fine
