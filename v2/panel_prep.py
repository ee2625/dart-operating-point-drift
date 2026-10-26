"""v2 panel_prep: raw ingested prices -> features/label the cell_builder needs.

Between the ingesters (which emit market, node, node_type, timestamp_utc, da_lmp,
rt_lmp) and `cell_builder`. Adds:

  - DART = da_lmp - rt_lmp;
  - the 10 G6 calendar features, computed in **market-local** time (electricity
    delivery is local), with the exact v1 column names so cell_builder's
    `g6_features()` list matches;
  - the top-alpha |DART| label per (market, node), thresholded on training rows
    only (leakage-safe), matching v1's `label_high_spread`.

Calendar features are deterministic functions of the local timestamp, so no price
information leaks. Nothing here fits a model or reads the network.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: Calendar tz per market for the G6 features. Most markets settle in *prevailing*
#: (DST-aware) local time, so the feature clock == the settlement clock. MISO is the
#: exception: it settles in **fixed Eastern Standard Time** (UTC-5, no DST — verified
#: from the report header and the invariant 24 hour-endings on both DST-transition
#: days). Using a DST-aware zone for MISO would drift its market-hour label by 1h
#: seasonally, so MISO uses a fixed UTC-5 offset (Etc/GMT+5) to match its HE grid.
MARKET_TZ = {"NYISO": "America/New_York", "ISONE": "America/New_York",
             "MISO": "Etc/GMT+5", "SPP": "America/Chicago",
             "CAISO": "America/Los_Angeles"}

G6_COLUMNS = ["calendar_hour_sin", "calendar_hour_cos",
              "calendar_day_of_week_sin", "calendar_day_of_week_cos",
              "calendar_day_of_year_sin", "calendar_day_of_year_cos",
              "calendar_month_sin", "calendar_month_cos",
              "calendar_is_peak_hour", "calendar_is_weekend"]


def _cyc(v, period):
    a = 2 * np.pi * v / period
    return np.sin(a), np.cos(a)


def add_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add DART and the 10 market-local G6 calendar features."""
    p = panel.copy()
    p["timestamp_utc"] = pd.to_datetime(p["timestamp_utc"], utc=True)
    p["dart_spread"] = p["da_lmp"] - p["rt_lmp"]
    parts = []
    for market, g in p.groupby("market", sort=False):
        local = g["timestamp_utc"].dt.tz_convert(MARKET_TZ.get(market, "UTC"))
        h, dow, doy, mon = local.dt.hour, local.dt.dayofweek, local.dt.dayofyear, local.dt.month
        g = g.copy()
        g["calendar_hour_sin"], g["calendar_hour_cos"] = _cyc(h, 24)
        g["calendar_day_of_week_sin"], g["calendar_day_of_week_cos"] = _cyc(dow, 7)
        g["calendar_day_of_year_sin"], g["calendar_day_of_year_cos"] = _cyc(doy, 365.25)
        g["calendar_month_sin"], g["calendar_month_cos"] = _cyc(mon, 12)
        g["calendar_is_peak_hour"] = ((h >= 7) & (h <= 22)).astype(int)  # HE 8-23 local
        g["calendar_is_weekend"] = (dow >= 5).astype(int)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


#: v1's G1 price-history group (24 features). Node-specific, leakage-safe: price
#: lags >= 48h and rolling windows shifted 48h. Names match common_panel_groups.json.
G1_LAGS = (48, 72, 168)
G1_ROLL_WINDOWS = (24, 72, 168)


def add_g1_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add v1's 24 G1 price-history features per (market, node). Each node's series
    is reindexed to a complete hourly grid so integer shifts equal clock hours, then
    restricted back to the observed timestamps. All features end at t-48h or earlier
    (48/72/168h lags; rolling windows shifted 48h) -> leakage-safe per the info policy.
    """
    p = panel.copy()
    p["timestamp_utc"] = pd.to_datetime(p["timestamp_utc"], utc=True)
    if "dart_spread" not in p.columns:
        p["dart_spread"] = p["da_lmp"] - p["rt_lmp"]
    parts = []
    for _, g in p.groupby(["market", "node"], sort=False):
        g = g.sort_values("timestamp_utc").set_index("timestamp_utc")
        g = g[~g.index.duplicated(keep="first")]   # drop year-boundary parquet overlaps
        obs = g.index
        grid = pd.date_range(obs.min(), obs.max(), freq="h")
        g = g.reindex(grid)
        for h in G1_LAGS:
            g[f"da_lmp_lag_{h}h"] = g["da_lmp"].shift(h)
            g[f"rt_lmp_lag_{h}h"] = g["rt_lmp"].shift(h)
            g[f"dart_spread_lag_{h}h"] = g["dart_spread"].shift(h)
        ds = g["dart_spread"]
        for w in G1_ROLL_WINDOWS:
            mp = max(2, w // 2)
            rw = ds.rolling(w, min_periods=mp)
            g[f"dart_spread_roll_mean_{w}h_shift_48h"] = rw.mean().shift(48)
            g[f"dart_spread_roll_std_{w}h_shift_48h"] = rw.std().shift(48)
            g[f"dart_spread_roll_maximum_{w}h_shift_48h"] = rw.max().shift(48)
            g[f"dart_spread_roll_minimum_{w}h_shift_48h"] = rw.min().shift(48)
            g[f"dart_spread_roll_abs_mean_{w}h_shift_48h"] = (
                ds.abs().rolling(w, min_periods=mp).mean().shift(48))
        parts.append(g.loc[obs].reset_index().rename(columns={"index": "timestamp_utc"}))
    return pd.concat(parts, ignore_index=True)


def g1_feature_names() -> list[str]:
    names = []
    for h in G1_LAGS:
        names += [f"da_lmp_lag_{h}h", f"rt_lmp_lag_{h}h", f"dart_spread_lag_{h}h"]
    for w in G1_ROLL_WINDOWS:
        names += [f"dart_spread_roll_{s}_{w}h_shift_48h"
                  for s in ("mean", "std", "maximum", "minimum", "abs_mean")]
    return sorted(names)


def add_label(panel: pd.DataFrame, train_end: str, alpha: float = 0.05,
              col: str = "label_high_spread") -> pd.DataFrame:
    """Top-alpha |DART| label per (market, node); threshold on rows before
    ``train_end`` only, then applied to all rows (leakage-safe)."""
    p = panel.copy()
    p["timestamp_utc"] = pd.to_datetime(p["timestamp_utc"], utc=True)
    cut = pd.Timestamp(train_end, tz="UTC")
    absd = p["dart_spread"].abs()
    thr = (p.assign(_a=absd)[p["timestamp_utc"] < cut]
           .groupby(["market", "node"])["_a"].quantile(1 - alpha))
    key = list(zip(p["market"], p["node"]))
    p[col] = (absd.to_numpy() >= p.set_index(["market", "node"]).index.map(thr).to_numpy()).astype(int)
    return p


if __name__ == "__main__":  # validate the transform on the ingested samples
    import glob, os
    root = os.path.expanduser("~/dart-spread-data")
    files = sum((glob.glob(f"{root}/{m}_da_rt_*.parquet")
                 for m in ("nyiso", "miso", "isone", "caiso")), [])
    raw = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    feat = add_features(raw)
    labelled = add_label(feat, train_end="2024-02-01")
    miss = [c for c in G6_COLUMNS if c not in labelled.columns]
    print(f"markets: {sorted(labelled['market'].unique())}")
    print(f"rows {len(labelled):,}  G6 cols present: {not miss}  "
          f"label mean {labelled['label_high_spread'].mean():.3f}")
    print(f"DART std by market:\n{labelled.groupby('market')['dart_spread'].std().round(2)}")
