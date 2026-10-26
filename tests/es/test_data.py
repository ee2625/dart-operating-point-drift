import datetime as dt

import pytest


@pytest.fixture(scope="module")
def raw():
    from es.data import load_dev_raw
    return load_dev_raw()


def test_no_row_after_seal(raw):
    from es.guard import settlement_frame
    s = settlement_frame(raw["timestamp_utc"], raw["market"])
    assert max(s["settle_date"]) == dt.date(2026, 6, 30)


def test_caiso_coverage_reaches_last_development_hour(raw):
    # Coverage of the real files. That sealed rows are excluded at read time is pinned on
    # synthetic files in test_data_seal.py, because the real files end at the seal.
    import pandas as pd
    caiso = raw[raw["market"] == "CAISO"]
    assert caiso["timestamp_utc"].max() == pd.Timestamp("2026-07-01 06:00", tz="UTC")


def test_duplicates_removed_and_counted(raw):
    assert raw.attrs["duplicates_dropped"] > 0          # NYISO month boundaries (FORENSICS §7)
    assert not raw.duplicated(["market", "node", "timestamp_utc"]).any()


def test_node_map_restriction(raw):
    assert raw.groupby("market")["node"].nunique().to_dict() == {
        "CAISO": 6, "ISONE": 9, "MISO": 8, "NYISO": 10}


def test_panel_columns(raw):
    from es.data import build_panel
    sample = raw[raw["node"].isin(["N.Y.C.", "TH_SP15_GEN-APND"])]
    p = build_panel(sample)
    for col in ["dart_spread", "mid_lag_48h", "settle_date", "settle_hour", "wclass",
                "dart_spread_roll_abs_mean_168h_shift_48h", "calendar_hour_sin"]:
        assert col in p.columns
    assert set(p["wclass"].unique()) <= {0, 1}
