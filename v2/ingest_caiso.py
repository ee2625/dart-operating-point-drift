"""CAISO DA/RT LMP ingester for the v2 panel (plan W1). No key required.

CAISO OASIS SingleZip returns a zipped XML report; ``INTERVAL_START_GMT`` is
already UTC, so no timezone handling is needed. One node per query, so we loop the
frozen CAISO NODE_MAP nodes; OASIS rate-limits to ~1 request / 5 s and caps a
request at 31 days, so we chunk and sleep.

    DA: queryname=PRC_LMP,        market_run_id=DAM  (hourly)
    RT: queryname=PRC_INTVL_LMP,  market_run_id=RTM  (5-min -> hourly mean)

Keeps DATA_ITEM=LMP_PRC (total LMP). Writes PANEL_SCHEMA parquet to a local root.

    python3.12 v2/ingest_caiso.py --start 2024-01-01 --end 2024-01-02 --nodes TH_SP15_GEN-APND
"""
from __future__ import annotations

import argparse
import io
import time
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

OASIS = ("http://oasis.caiso.com/oasisapi/SingleZip?queryname={qn}"
         "&startdatetime={s}T08:00-0000&enddatetime={e}T08:00-0000"
         "&market_run_id={mri}&node={node}&version=1")
NS = {"o": "http://www.caiso.com/soa/OASISReport_v1.xsd"}
MARKET = "CAISO"
DEFAULT_ROOT = Path.home() / "dart-spread-data"
PAUSE = 5.5                       # OASIS rate limit (~1 req / 5 s)
NODE_TYPE = {"TH_SP15_GEN-APND": "trading_hub", "TH_NP15_GEN-APND": "trading_hub",
             "TH_ZP26_GEN-APND": "trading_hub", "DLAP_PGAE-APND": "load_zone",
             "DLAP_SCE-APND": "load_zone", "DLAP_SDGE-APND": "load_zone"}


def _fetch(qn: str, mri: str, node: str, s: pd.Timestamp, e: pd.Timestamp) -> pd.DataFrame:
    url = OASIS.format(qn=qn, mri=mri, node=node,
                       s=s.strftime("%Y%m%d"), e=e.strftime("%Y%m%d"))
    blob = urlopen(url, timeout=90).read()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        xml = zf.read(zf.namelist()[0])
    root = ET.fromstring(xml)
    rows = []
    for it in root.findall(".//o:REPORT_DATA", NS):
        d = {c.tag.split("}")[-1]: c.text for c in it}
        if d.get("DATA_ITEM") == "LMP_PRC":
            rows.append((d["INTERVAL_START_GMT"], d["RESOURCE_NAME"], float(d["VALUE"])))
    df = pd.DataFrame(rows, columns=["timestamp_utc", "node", "lmp"])
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df


def _series(qn: str, mri: str, nodes: list[str], days: pd.DatetimeIndex,
            value: str, hourly: bool) -> pd.DataFrame:
    chunks = pd.date_range(days[0], days[-1] + pd.Timedelta(days=1), freq="30D")
    if chunks[-1] <= days[-1]:
        chunks = chunks.append(pd.DatetimeIndex([days[-1] + pd.Timedelta(days=1)]))
    parts = []
    for node in nodes:
        for s, e in zip(chunks[:-1], chunks[1:]):
            parts.append(_fetch(qn, mri, node, s, e))
            time.sleep(PAUSE)
        print(f"  {value} {node} done", flush=True)
    df = pd.concat(parts, ignore_index=True)
    if hourly:                     # RT 5-min -> hourly settlement mean
        df = (df.assign(h=df["timestamp_utc"].dt.floor("h"))
                .groupby(["node", "h"], as_index=False)["lmp"].mean()
                .rename(columns={"h": "timestamp_utc"}))
    return df.rename(columns={"lmp": value})


def build(start: str, end: str, nodes: list[str], root: Path = DEFAULT_ROOT) -> Path:
    days = pd.date_range(start, end, freq="D")
    da = _series("PRC_LMP", "DAM", nodes, days, "da_lmp", hourly=False)
    rt = _series("PRC_INTVL_LMP", "RTM", nodes, days, "rt_lmp", hourly=True)
    panel = da.merge(rt, on=["timestamp_utc", "node"], how="inner")
    panel["market"] = MARKET
    panel["node_type"] = panel["node"].map(NODE_TYPE).fillna("other")
    panel = panel[["market", "node", "node_type", "timestamp_utc", "da_lmp", "rt_lmp"]]
    panel = panel.sort_values(["node", "timestamp_utc"]).reset_index(drop=True)

    root.mkdir(parents=True, exist_ok=True)
    out = root / f"caiso_da_rt_{start}_{end}.parquet".replace("-", "")
    panel.to_parquet(out, index=False)
    print(f"\n{len(panel):,} rows, {panel['node'].nunique()} nodes, "
          f"types={sorted(panel['node_type'].unique())}")
    print(f"wrote {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--nodes", nargs="*", default=list(NODE_TYPE))
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    a = ap.parse_args()
    build(a.start, a.end, a.nodes, Path(a.root))


if __name__ == "__main__":
    main()
