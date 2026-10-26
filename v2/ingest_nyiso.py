"""NYISO DA/RT LMP ingester for the v2 panel (plan W1).

NYISO publishes zonal LBMP as free, no-key monthly zip archives back to ~1999:

    DA (hourly): http://mis.nyiso.com/public/csv/damlbmp/<YYYYMM01>damlbmp_zone_csv.zip
    RT (5-min):  http://mis.nyiso.com/public/csv/realtime/<YYYYMM01>realtime_zone_csv.zip

Each monthly zip holds one CSV per day with columns
``Time Stamp, Name, PTID, LBMP ($/MWHr), Marginal Cost Losses, Marginal Cost
Congestion``. Timestamps are New York prevailing time. This module downloads a
date range, aggregates RT 5-minute prices to the hourly settlement resolution,
converts to UTC (handling DST duplicate/missing hours), and writes one parquet per
market in the ``PANEL_SCHEMA`` the decomposition consumes.

Stdlib networking only (urllib), so it runs anywhere with outbound HTTP; writes to
a **local** data root (default ``~/dart-spread-data``), never the Drive cache.

    python3.12 v2/ingest_nyiso.py --start 2024-01 --end 2024-03
"""
from __future__ import annotations

import argparse
import http.client
import io
import socket
import time
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import pandas as pd

_TRANSIENT = (URLError, socket.timeout, ConnectionResetError, http.client.IncompleteRead)


def _urlread(url: str, retries: int = 5, timeout: int = 60) -> bytes:
    """Fetch with retry on transient network errors; HTTPError re-raised immediately."""
    for i in range(retries):
        try:
            return urlopen(url, timeout=timeout).read()
        except HTTPError:
            raise
        except _TRANSIENT:
            if i == retries - 1:
                raise
            time.sleep(2 * (i + 1))

DA_URL = "http://mis.nyiso.com/public/csv/damlbmp/{ym}01damlbmp_zone_csv.zip"
RT_URL = "http://mis.nyiso.com/public/csv/realtime/{ym}01realtime_zone_csv.zip"
TZ = "America/New_York"
MARKET = "NYISO"
DEFAULT_ROOT = Path.home() / "dart-spread-data"


def months(start: str, end: str) -> list[str]:
    """Inclusive list of 'YYYYMM' strings from start to end ('YYYY-MM')."""
    rng = pd.period_range(start=start, end=end, freq="M")
    return [p.strftime("%Y%m") for p in rng]


def _read_zip(url: str) -> pd.DataFrame:
    blob = _urlread(url)
    frames = []
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for name in zf.namelist():
            if name.lower().endswith(".csv"):
                frames.append(pd.read_csv(zf.open(name)))
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"Time Stamp": "ts_local", "Name": "node",
                            "LBMP ($/MWHr)": "lmp"})
    # DA stamps are "MM/DD/YYYY HH:MM"; RT 5-min stamps add ":SS".
    ts = df["ts_local"].astype(str)
    fmt = "%m/%d/%Y %H:%M:%S" if ts.iloc[0].count(":") == 2 else "%m/%d/%Y %H:%M"
    df["ts_local"] = pd.to_datetime(ts, format=fmt)
    return df[["ts_local", "node", "lmp"]]


def _to_utc_hourly(df: pd.DataFrame, aggregate: bool) -> pd.DataFrame:
    """Localize NY time -> UTC (DST-safe), optionally mean-aggregate to the hour."""
    if aggregate:  # RT 5-min -> hourly settlement (uniform intervals => simple mean)
        df = (df.assign(hour=df["ts_local"].dt.floor("h"))
                .groupby(["node", "hour"], as_index=False)["lmp"].mean()
                .rename(columns={"hour": "ts_local"}))
    out = []
    for node, g in df.groupby("node", sort=False):
        idx = g["ts_local"]
        # DST edges: drop the spring-forward gap and the (single, un-repeated in
        # these 24-HE files) fall-back ambiguous hour rather than guess.
        loc = idx.dt.tz_localize(TZ, ambiguous="NaT", nonexistent="NaT")
        gg = g.assign(timestamp_utc=loc.dt.tz_convert("UTC")).dropna(subset=["timestamp_utc"])
        out.append(gg[["timestamp_utc", "node", "lmp"]])
    return pd.concat(out, ignore_index=True)


def build(start: str, end: str, root: Path = DEFAULT_ROOT) -> Path:
    da_parts, rt_parts = [], []
    for ym in months(start, end):
        da_parts.append(_to_utc_hourly(_read_zip(DA_URL.format(ym=ym)), aggregate=False))
        rt_parts.append(_to_utc_hourly(_read_zip(RT_URL.format(ym=ym)), aggregate=True))
        print(f"  {ym} ok", flush=True)
    da = pd.concat(da_parts, ignore_index=True).rename(columns={"lmp": "da_lmp"})
    rt = pd.concat(rt_parts, ignore_index=True).rename(columns={"lmp": "rt_lmp"})
    panel = da.merge(rt, on=["timestamp_utc", "node"], how="inner")
    panel["market"] = MARKET
    panel["node_type"] = "load_zone"          # NYISO zonal LBMP series
    panel = panel[["market", "node", "node_type", "timestamp_utc", "da_lmp", "rt_lmp"]]
    panel = panel.sort_values(["node", "timestamp_utc"]).reset_index(drop=True)

    root.mkdir(parents=True, exist_ok=True)
    out = root / f"nyiso_da_rt_{start}_{end}.parquet".replace("-", "")
    panel.to_parquet(out, index=False)
    print(f"\n{len(panel):,} rows, {panel['node'].nunique()} nodes, "
          f"{panel['timestamp_utc'].min()} .. {panel['timestamp_utc'].max()}")
    print(f"wrote {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM")
    ap.add_argument("--end", required=True, help="YYYY-MM")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    a = ap.parse_args()
    build(a.start, a.end, Path(a.root))


if __name__ == "__main__":
    main()
