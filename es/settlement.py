"""Settlement real-time legs (spec v3 §4.1). CAISO virtual bids settle on the hourly mean of
four 15-minute FMM LMPs; NYISO virtual transactions settle on the
integrated hourly real-time LBMP. The panel's CAISO and NYISO RT series are 5-minute means,
so P&L uses the settlement series where obtainable and flags the rest as approximate."""
from __future__ import annotations

import io
import time
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from .config import SEAL_LAST_DEV_DATE
from .guard import SealedDataError, first_sealed_utc

CAISO_NODES = ("TH_SP15_GEN-APND", "TH_NP15_GEN-APND", "TH_ZP26_GEN-APND",
               "DLAP_PGAE-APND", "DLAP_SCE-APND", "DLAP_SDGE-APND")
CAISO_URL = ("https://oasis.caiso.com/oasisapi/SingleZip?queryname=PRC_RTPD_LMP"
             "&startdatetime={s}T07:00-0000&enddatetime={e}T07:00-0000&market_run_id=RTPD"
             "&node={node}&version=1&resultformat=6")
NYISO_URL = "http://mis.nyiso.com/public/csv/rtlbmp/{ym}01rtlbmp_zone_csv.zip"


def _check_dev_end(end: str) -> None:
    if pd.Timestamp(end) > pd.Timestamp(SEAL_LAST_DEV_DATE):
        raise SealedDataError(f"pull end {end} is after {SEAL_LAST_DEV_DATE}; that window is sealed")


def _assert_before_seal(table: pd.DataFrame, market: str) -> None:
    """Raise if any row of a loaded settlement table is at or after the market's first sealed instant."""
    cut = first_sealed_utc(market)
    n_sealed = int((pd.to_datetime(table["timestamp_utc"], utc=True) >= cut).sum())
    if n_sealed:
        raise SealedDataError(f"{n_sealed} {market} settlement rows are at or after {cut}; that window is sealed "
                              "(dev_only=False is reserved for the sealed-run entry point)")


def _fetch(url: str, attempts: int = 5) -> bytes:
    last = None
    for k in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                return resp.read()
        except Exception as exc:          # retried, then re-raised; never silently skipped
            last = exc
            time.sleep(6 * (k + 1))
    raise last


def _parse_caiso(blob: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        names = [n for n in z.namelist() if n.endswith(".csv")]
        if not names:
            raise RuntimeError("OASIS returned no CSV (error or no-data response)")
        df = pd.read_csv(z.open(names[0]))
    # Header verified on 22 Oct 2026 (resultformat=6): INTERVALSTARTTIME_GMT, ..., NODE,
    # MARKET_RUN_ID, LMP_TYPE (LMP/MCC/MCE/MCL), XML_DATA_ITEM, ..., PRC, OPR_INTERVAL. The price
    # column is PRC; 96 intervals x 4 components per node-day.
    df = df[df["LMP_TYPE"] == "LMP"]
    return pd.DataFrame({"node": df["NODE"],
                         "timestamp_utc": pd.to_datetime(df["INTERVALSTARTTIME_GMT"], utc=True),
                         "fmm15": df["PRC"].astype(float)})


def pull_caiso_fmm(start, end, root, nodes=CAISO_NODES):
    _check_dev_end(end)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    frames, coverage = [], []
    for node in nodes:
        for chunk_start in pd.date_range(start, end, freq="30D"):
            chunk_end = min(chunk_start + pd.Timedelta(days=30), pd.Timestamp(end) + pd.Timedelta(days=1))
            url = CAISO_URL.format(s=chunk_start.strftime("%Y%m%d"), e=chunk_end.strftime("%Y%m%d"), node=node)
            part = _parse_caiso(_fetch(url))
            frames.append(part)
            coverage.append({"node": node, "start": chunk_start.date(), "rows15": len(part), "dropped_duplicates": 0})
            time.sleep(5.5)
    stacked = pd.concat(frames, ignore_index=True)
    raw = stacked.drop_duplicates(["node", "timestamp_utc"])
    # Last coverage row, for the whole pull: rows15 = LMP intervals kept, dropped_duplicates = intervals removed here
    # (chunk rows15 add up to kept + removed, and the hourly n15 add up to kept).
    coverage.append({"node": "ALL", "start": "drop_duplicates", "rows15": len(raw),
                     "dropped_duplicates": len(stacked) - len(raw)})
    hourly = (raw.assign(timestamp_utc=raw["timestamp_utc"].dt.floor("h"))
                 .groupby(["node", "timestamp_utc"], as_index=False)["fmm15"].agg(["mean", "count"])
                 .rename(columns={"mean": "fmm_lmp", "count": "n15"}))
    path = root / f"caiso_fmm_{start.replace('-', '')}_{end.replace('-', '')}.parquet"
    hourly.to_parquet(path, index=False)
    pd.DataFrame(coverage).to_csv(root / "coverage.csv", index=False)
    return path


def load_caiso_fmm(root, dev_only=True):
    """Hourly FMM means with all four intervals. ``dev_only=False`` is reserved for the sealed-run entry point."""
    files = sorted(Path(root).glob("caiso_fmm_*.parquet")) if Path(root).exists() else []
    if not files:
        return None
    fmm = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if dev_only:
        _assert_before_seal(fmm, "CAISO")          # any file, any row, whether or not the hour is complete
    fmm = fmm[fmm["n15"] == 4]                     # an hour needs all four 15-minute intervals
    return fmm[["node", "timestamp_utc", "fmm_lmp"]]


def _parse_nyiso(blob: bytes) -> pd.DataFrame:
    frames = []
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for name in sorted(n for n in z.namelist() if n.endswith(".csv")):
            frames.append(pd.read_csv(z.open(name)))
    df = pd.concat(frames, ignore_index=True)
    local = pd.to_datetime(df["Time Stamp"], format="%m/%d/%Y %H:%M")
    ts = (local.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
               .dt.tz_convert("UTC"))
    out = pd.DataFrame({"node": df["Name"], "timestamp_utc": ts,
                        "rt_int_lmp": df["LBMP ($/MWHr)"].astype(float)})
    kept = out.dropna(subset=["timestamp_utc"]).reset_index(drop=True)
    # Rows whose local time is ambiguous (the repeated fall-back hour) or nonexistent are dropped, never guessed;
    # the count travels with the frame so that the puller can log it.
    kept.attrs["dropped_nat"] = len(out) - len(kept)
    return kept


def pull_nyiso_integrated(start_month, end_month, root):
    _check_dev_end(str(pd.Period(end_month, "M").end_time.date()))
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    frames, coverage = [], []
    for period in pd.period_range(start_month, end_month, freq="M"):
        part = _parse_nyiso(_fetch(NYISO_URL.format(ym=period.strftime("%Y%m"))))
        frames.append(part)
        dropped_nat = part.attrs["dropped_nat"]
        coverage.append({"month": str(period), "raw_rows": len(part) + dropped_nat, "dropped_nat": dropped_nat,
                         "rows": len(part), "dropped_duplicates": 0})
        time.sleep(2.0)
    stacked = pd.concat(frames, ignore_index=True)
    out = stacked.drop_duplicates(["node", "timestamp_utc"])
    # Last coverage row, for the whole pull: raw_rows = rows entering drop_duplicates, rows = rows kept, so that
    # raw_rows - dropped_nat - dropped_duplicates == rows holds on every row of coverage.csv.
    coverage.append({"month": "drop_duplicates", "raw_rows": len(stacked), "dropped_nat": 0, "rows": len(out),
                     "dropped_duplicates": len(stacked) - len(out)})
    path = root / f"nyiso_rtint_{start_month.replace('-', '')}_{end_month.replace('-', '')}.parquet"
    out.to_parquet(path, index=False)
    pd.DataFrame(coverage).to_csv(root / "coverage.csv", index=False)
    return path


def load_nyiso_integrated(root, dev_only=True):
    """Integrated hourly LBMP. ``dev_only=False`` is reserved for the sealed-run entry point."""
    files = sorted(Path(root).glob("nyiso_rtint_*.parquet")) if Path(root).exists() else []
    if not files:
        return None
    nyi = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if dev_only:
        _assert_before_seal(nyi, "NYISO")
    return nyi[["node", "timestamp_utc", "rt_int_lmp"]]


def attach_settlement(rows, fmm=None, nyiso_int=None):
    if not rows.index.is_unique:               # results are written back by index label
        raise ValueError("attach_settlement: rows must have a unique index")
    if fmm is not None and "n15" in fmm.columns:
        fmm = fmm.loc[fmm["n15"] == 4]         # a raw hourly table: an hour needs all four 15-minute intervals
    out = rows.copy()
    out["rt_settle"] = out["rt_lmp"].astype(float)
    out["settle_approx"] = False
    for market, table, col in (("CAISO", fmm, "fmm_lmp"), ("NYISO", nyiso_int, "rt_int_lmp")):
        sel = (out["market"] == market).to_numpy()
        if not sel.any():
            continue
        if table is None:
            out.loc[sel, "settle_approx"] = True
            continue
        vals = (out.loc[sel, ["node", "timestamp_utc"]]
                .merge(table[["node", "timestamp_utc", col]], on=["node", "timestamp_utc"], how="left")[col]
                .to_numpy())
        have = ~pd.isna(vals)
        idx = out.index[sel]
        out.loc[idx[have], "rt_settle"] = vals[have].astype(float)
        out.loc[idx[~have], "settle_approx"] = True
    out["dart_settle"] = out["da_lmp"].astype(float) - out["rt_settle"]
    return out
