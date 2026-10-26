"""MISO DA/RT LMP ingester for the v2 panel (plan W1). No key required.

MISO publishes daily market reports as public CSVs:

    DA: https://docs.misoenergy.org/marketreports/<YYYYMMDD>_da_expost_lmp.csv
    RT: https://docs.misoenergy.org/marketreports/<YYYYMMDD>_rt_lmp_final.csv

Both are wide: a short preamble, then ``Node,Type,Value,HE 1..HE 24``. ``Value``
is one of LMP / MCC (congestion) / MLC (loss); we keep LMP. The header states
"All Hours-Ending are Eastern Standard Time (EST)", i.e. a fixed UTC-5 offset all
year (no DST shift), so HE h maps to a fixed UTC hour. HE h is the hour *ending*
at h, so its hour-beginning local time is (h-1):00 EST.

Writes the ``PANEL_SCHEMA`` parquet to a local (non-Drive) root.

    python3.12 v2/ingest_miso.py --start 2024-01-01 --end 2024-01-31
"""
from __future__ import annotations

import argparse
import io
import http.client
import socket
import time
from datetime import timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import pandas as pd

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

DA_URL = "https://docs.misoenergy.org/marketreports/{d}_da_expost_lmp.csv"
RT_URL = "https://docs.misoenergy.org/marketreports/{d}_rt_lmp_final.csv"
EST_OFFSET = pd.Timedelta(hours=5)          # EST = UTC-5, fixed (report convention)
MARKET = "MISO"
DEFAULT_ROOT = Path.home() / "dart-spread-data"
TYPE_MAP = {"Loadzone": "load_zone", "Interface": "interface",
            "Hub": "trading_hub", "Gennode": "gen_node"}

#: frozen MISO node set (the 8 trading hubs) from NODE_MAP.csv. The daily report
#: carries ~2,400 nodes; keeping only these makes the multi-year pull small.
NODES = {"ARKANSAS.HUB", "ILLINOIS.HUB", "INDIANA.HUB", "LOUISIANA.HUB",
         "MICHIGAN.HUB", "MINN.HUB", "MS.HUB", "TEXAS.HUB"}


def _read_day(url: str, day: pd.Timestamp) -> pd.DataFrame:
    """One daily wide report -> long [timestamp_utc, node, node_type, lmp]."""
    text = _urlread(url).decode("utf-8", "replace")
    lines = text.splitlines()
    hdr = next(i for i, l in enumerate(lines) if l.startswith("Node,Type,Value"))
    df = pd.read_csv(io.StringIO("\n".join(lines[hdr:])))
    df = df[(df["Value"] == "LMP") & (df["Node"].isin(NODES))]
    he_cols = [c for c in df.columns if c.replace(" ", "").upper().startswith("HE")]
    long = df.melt(id_vars=["Node", "Type"], value_vars=he_cols,
                   var_name="he", value_name="lmp")
    long["he"] = long["he"].str.extract(r"(\d+)").astype(int)
    long["lmp"] = pd.to_numeric(long["lmp"], errors="coerce")
    # HE h -> local hour-beginning (h-1) EST -> UTC
    long["timestamp_utc"] = (day + pd.to_timedelta(long["he"] - 1, unit="h") + EST_OFFSET)
    long["node_type"] = long["Type"].map(TYPE_MAP).fillna("other")
    return long.rename(columns={"Node": "node"})[
        ["timestamp_utc", "node", "node_type", "lmp"]].dropna(subset=["lmp"])


def _series(url_tmpl: str, days: pd.DatetimeIndex, value: str) -> pd.DataFrame:
    parts, misses = [], 0
    for day in days:
        url = url_tmpl.format(d=day.strftime("%Y%m%d"))
        try:
            parts.append(_read_day(url, day))
        except (HTTPError, URLError) as e:
            misses += 1
            if misses <= 5:
                why = getattr(e, "code", None) or getattr(e, "reason", e)
                print(f"  skip {day.date()} {value}: {why}", flush=True)
    out = pd.concat(parts, ignore_index=True).rename(columns={"lmp": value})
    print(f"  {value}: {len(days) - misses}/{len(days)} days", flush=True)
    return out


def build(start: str, end: str, root: Path = DEFAULT_ROOT) -> Path:
    days = pd.date_range(start, end, freq="D")
    da = _series(DA_URL, days, "da_lmp")
    rt = _series(RT_URL, days, "rt_lmp")
    panel = da.merge(rt, on=["timestamp_utc", "node", "node_type"], how="inner")
    panel["market"] = MARKET
    panel = panel[["market", "node", "node_type", "timestamp_utc", "da_lmp", "rt_lmp"]]
    panel = panel.sort_values(["node", "timestamp_utc"]).reset_index(drop=True)

    root.mkdir(parents=True, exist_ok=True)
    out = root / f"miso_da_rt_{start}_{end}.parquet".replace("-", "")
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
