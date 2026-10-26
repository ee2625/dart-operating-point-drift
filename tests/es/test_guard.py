import pandas as pd
import pytest


def test_first_sealed_utc_per_market():
    from es.guard import first_sealed_utc
    assert first_sealed_utc("NYISO") == pd.Timestamp("2026-07-01 04:00", tz="UTC")
    assert first_sealed_utc("ISONE") == pd.Timestamp("2026-07-01 04:00", tz="UTC")
    assert first_sealed_utc("MISO") == pd.Timestamp("2026-07-01 05:00", tz="UTC")
    assert first_sealed_utc("CAISO") == pd.Timestamp("2026-07-01 07:00", tz="UTC")


def test_settlement_frame_uses_market_clocks():
    from es.guard import settlement_frame
    ts = pd.Series(pd.to_datetime(["2026-06-30 04:00", "2026-06-30 04:00"], utc=True))
    mk = pd.Series(["NYISO", "MISO"])
    s = settlement_frame(ts, mk)
    assert str(s.loc[0, "settle_date"]) == "2026-06-30" and s.loc[0, "settle_hour"] == 0
    assert str(s.loc[1, "settle_date"]) == "2026-06-29" and s.loc[1, "settle_hour"] == 23


def test_guard_raises_on_a_sealed_row():
    from es.guard import SealedDataError, assert_development_safe
    frame = pd.DataFrame({"market": ["CAISO"],
                          "timestamp_utc": pd.to_datetime(["2026-07-01 07:00"], utc=True)})
    with pytest.raises(SealedDataError) as err:
        assert_development_safe(frame)
    # Ruling 140: the windows stay sealed until a registration record of the protocol exists, not until an OSF registration
    assert "stay sealed until a registration record of the protocol exists " \
           "(docs/sealed/REGISTRATION.md; EnergySystems_Revision/CLAUDE.md)" in str(err.value)
    assert "OSF" not in str(err.value) and "1 rows have a settlement date after" in str(err.value)


def test_guard_passes_last_development_hour():
    from es.guard import assert_development_safe
    frame = pd.DataFrame({"market": ["CAISO"],
                          "timestamp_utc": pd.to_datetime(["2026-07-01 06:00"], utc=True)})
    assert assert_development_safe(frame) is frame
