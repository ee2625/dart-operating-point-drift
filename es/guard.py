"""Sealed-window guard (spec invariant I1). Development code must never see a row whose
settlement date is after SEAL_LAST_DEV_DATE in the market's settlement clock."""
from __future__ import annotations

import pandas as pd

from .config import SEAL_LAST_DEV_DATE, SETTLE_TZ


class SealedDataError(RuntimeError):
    """Development code met data from a sealed window."""


def first_sealed_utc(market: str) -> pd.Timestamp:
    """UTC instant at which the first sealed settlement day starts in ``market``."""
    day_after = pd.Timestamp(SEAL_LAST_DEV_DATE) + pd.Timedelta(days=1)
    return day_after.tz_localize(SETTLE_TZ[market]).tz_convert("UTC")


def settlement_frame(timestamps_utc: pd.Series, markets: pd.Series) -> pd.DataFrame:
    """Local settlement date, hour and weekday for each row."""
    ts = pd.to_datetime(timestamps_utc, utc=True)
    unknown = sorted(set(markets.unique()) - set(SETTLE_TZ))
    if unknown:
        raise ValueError(f"unknown market(s): {unknown}")
    out = pd.DataFrame(index=ts.index, columns=["settle_date", "settle_hour", "weekday"], dtype=object)
    for market, tz in SETTLE_TZ.items():
        sel = (markets == market).to_numpy()
        if sel.any():
            local = ts[sel].dt.tz_convert(tz)
            out.loc[sel, "settle_date"] = local.dt.date.to_numpy()
            out.loc[sel, "settle_hour"] = local.dt.hour.to_numpy()
            out.loc[sel, "weekday"] = local.dt.dayofweek.to_numpy()
    out["settle_hour"] = out["settle_hour"].astype(int)
    out["weekday"] = out["weekday"].astype(int)
    return out


def assert_development_safe(frame: pd.DataFrame) -> pd.DataFrame:
    """Return ``frame`` unchanged, or raise if any row is in a sealed window."""
    s = settlement_frame(frame["timestamp_utc"], frame["market"])
    last = pd.Timestamp(SEAL_LAST_DEV_DATE).date()
    n_bad = int((s["settle_date"] > last).sum())
    if n_bad:
        raise SealedDataError(
            f"{n_bad} rows have a settlement date after {SEAL_LAST_DEV_DATE}; these windows "
            "stay sealed until a registration record of the protocol exists (docs/sealed/REGISTRATION.md; "
            "EnergySystems_Revision/CLAUDE.md)")
    return frame
