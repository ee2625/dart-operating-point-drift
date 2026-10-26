"""ISO-NE DA/RT LMP ingester for the v2 panel (plan W1). No key required.

ISO Express publishes hourly LMP as public daily CSVs (this is how v1 got ISO-NE;
`webservices.iso-ne.com` is the auth-gated route and is NOT used):

    DA: https://www.iso-ne.com/static-transform/csv/histRpts/da-lmp/WW_DALMP_ISO_<YYYYMMDD>.csv
    RT: https://www.iso-ne.com/static-transform/csv/histRpts/rt-lmp/lmp_rt_final_<YYYYMMDD>.csv

Files are C/H/D-tagged: 'C' comments, one 'H' header, 'D' data rows with
``Date, Hour Ending, Location ID, Location Name, Location Type, LMP, Energy,
Congestion, Loss``. Hour Ending is Eastern *Prevailing* Time (DST-aware), 1..24.
We keep the frozen NODE_MAP locations, localize (DST-safe) to UTC, write
``PANEL_SCHEMA`` parquet to a local root.

    python3.12 v2/ingest_isone.py --start 2024-01-01 --end 2024-01-03
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import http.client
import socket
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import pandas as pd

#: transient network faults worth retrying (IncompleteRead = truncated response)
_TRANSIENT = (URLError, socket.timeout, ConnectionResetError, http.client.IncompleteRead)


def _urlread(url: str, retries: int = 5, timeout: int = 60) -> bytes:
    """Fetch with retry on transient network errors; HTTPError (e.g. 404) is
    re-raised immediately so the caller can skip that day."""
    for i in range(retries):
        try:
            return urlopen(url, timeout=timeout).read()
        except HTTPError:
            raise
        except _TRANSIENT:
            if i == retries - 1:
                raise
            time.sleep(2 * (i + 1))

DA_URL = "https://www.iso-ne.com/static-transform/csv/histRpts/da-lmp/WW_DALMP_ISO_{d}.csv"
RT_URL = "https://www.iso-ne.com/static-transform/csv/histRpts/rt-lmp/lmp_rt_final_{d}.csv"
TZ = "America/New_York"          # Eastern Prevailing Time (DST-aware)
MARKET = "ISONE"
DEFAULT_ROOT = Path.home() / "dart-spread-data"

#: frozen ISO-NE node set (hub + 8 load zones) from NODE_MAP.csv
NODES = {".H.INTERNAL_HUB": "trading_hub", ".Z.MAINE": "load_zone",
         ".Z.NEWHAMPSHIRE": "load_zone", ".Z.VERMONT": "load_zone",
         ".Z.CONNECTICUT": "load_zone", ".Z.RHODEISLAND": "load_zone",
         ".Z.SEMASS": "load_zone", ".Z.WCMASS": "load_zone",
         ".Z.NEMASSBOST": "load_zone"}


def _read_day(url: str, day: pd.Timestamp) -> pd.DataFrame:
    raw = _urlread(url).decode("utf-8", "replace")
    rows = [r for r in csv.reader(io.StringIO(raw)) if r and r[0] == "D"]
    df = pd.DataFrame(rows, columns=["tag", "date", "he", "loc_id", "node",
                                     "loc_type", "lmp", "energy", "cong", "loss"])
    df = df[df["node"].isin(NODES)].copy()
    df["he"] = df["he"].map(lambda s: int(re.sub(r"\D", "", s)))   # "02X" -> 2
    df["lmp"] = pd.to_numeric(df["lmp"], errors="coerce")
    local = day + pd.to_timedelta(df["he"] - 1, unit="h")          # hour-beginning, local
    ts = (pd.DatetimeIndex(local).tz_localize(TZ, ambiguous="NaT", nonexistent="NaT")
          .tz_convert("UTC"))
    out = df.assign(timestamp_utc=ts, node_type=df["node"].map(NODES))
    return out.dropna(subset=["timestamp_utc", "lmp"])[
        ["timestamp_utc", "node", "node_type", "lmp"]]


def _series(url_tmpl: str, days: pd.DatetimeIndex, value: str) -> pd.DataFrame:
    parts, misses = [], 0
    for day in days:
        try:
            parts.append(_read_day(url_tmpl.format(d=day.strftime("%Y%m%d")), day))
        except (HTTPError, URLError) as e:
            misses += 1
            if misses <= 5:
                why = getattr(e, "code", None) or getattr(e, "reason", e)
                print(f"  skip {day.date()} {value}: {why}", flush=True)
    print(f"  {value}: {len(days) - misses}/{len(days)} days", flush=True)
    return pd.concat(parts, ignore_index=True).rename(columns={"lmp": value})


def build(start: str, end: str, root: Path = DEFAULT_ROOT) -> Path:
    days = pd.date_range(start, end, freq="D")
    da = _series(DA_URL, days, "da_lmp")
    rt = _series(RT_URL, days, "rt_lmp")
    panel = da.merge(rt, on=["timestamp_utc", "node", "node_type"], how="inner")
    panel["market"] = MARKET
    panel = panel[["market", "node", "node_type", "timestamp_utc", "da_lmp", "rt_lmp"]]
    panel = panel.sort_values(["node", "timestamp_utc"]).reset_index(drop=True)

    root.mkdir(parents=True, exist_ok=True)
    out = root / f"isone_da_rt_{start}_{end}.parquet".replace("-", "")
    panel.to_parquet(out, index=False)
    print(f"\n{len(panel):,} rows, {panel['node'].nunique()} nodes, "
          f"types={sorted(panel['node_type'].unique())}")
    print(f"wrote {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    a = ap.parse_args()
    build(a.start, a.end, Path(a.root))


if __name__ == "__main__":
    main()
