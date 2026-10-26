import ast
import collections
import datetime as dt
import functools
import hashlib
import io
import json
import os
import pathlib
import re
import stat
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import pandas as pd
import pytest

NO_SLEEP = dict(pause=0, sleep=lambda _s: None)
DART_DEV = Path(__file__).resolve().parents[2]
REAL_SEALED_DIR = Path(__file__).resolve().parents[3] / "docs" / "sealed"


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """No test in this file reaches a real endpoint: responses come from injected fakes. urllib's urlopen is blocked
    and so is the urlopen each v2 ingester bound at import (patching urllib.request alone leaves those open); the
    socket layer is closed for the whole suite in conftest.py."""
    import urllib.request

    import v2.ingest_isone
    import v2.ingest_miso
    import v2.ingest_nyiso

    def blocked(*args, **kwargs):
        raise AssertionError("network access attempted in a unit test")

    monkeypatch.setattr(urllib.request, "urlopen", blocked)
    for module in (v2.ingest_isone, v2.ingest_miso, v2.ingest_nyiso):
        monkeypatch.setattr(module, "urlopen", blocked)


@pytest.fixture(autouse=True)
def _the_real_sealed_dir_is_left_alone():
    """PULLS.json and the pull locks live in docs/sealed. No test may create, change or remove them there."""
    def snapshot():
        record = REAL_SEALED_DIR / "PULLS.json"
        locks = sorted(p.name for p in REAL_SEALED_DIR.glob("PULL_*.lock")) if REAL_SEALED_DIR.exists() else []
        return record.stat().st_mtime_ns if record.exists() else None, locks

    before = snapshot()
    yield
    assert snapshot() == before, "a test wrote PULLS.json or a pull lock in the real docs/sealed"


# =====================================================================================================
# The brief's tests. Fix round 1 changed how they call pull_sealed and nothing else: `registration=` is gone (the
# `registered` fixture points the gate at its record), the fakes write a node of their own, so they pass the logged
# check_days=False, and pull_sealed returns (root, manifest_sha256).
# =====================================================================================================
def test_windows_start_ten_days_early():
    from es.sealed_pull import SEALED_WINDOWS
    assert SEALED_WINDOWS["A"] == ("2026-06-21", "2026-07-01", "2026-09-30", "sealed_2026Q3")
    assert SEALED_WINDOWS["B"] == ("2026-09-21", "2026-10-01", "2026-12-31", "sealed_2026Q4")


def test_local_day_bounds_and_expected_hours():
    from es.sealed_pull import expected_hours, local_day_bounds_utc
    s, e = local_day_bounds_utc("2026-07-01", "2026-09-30", "America/Los_Angeles")
    assert s == pd.Timestamp("2026-07-01 07:00", tz="UTC") and e == pd.Timestamp("2026-10-01 07:00", tz="UTC")
    s, e = local_day_bounds_utc("2026-10-01", "2026-12-31", "America/Los_Angeles")
    assert s == pd.Timestamp("2026-10-01 07:00", tz="UTC") and e == pd.Timestamp("2027-01-01 08:00", tz="UTC")
    assert expected_hours("CAISO", "2026-11-01", "2026-11-01") == 25
    assert expected_hours("NYISO", "2026-03-08", "2026-03-08") == 23
    assert expected_hours("MISO", "2026-11-01", "2026-11-01") == 24
    assert expected_hours("ISONE", "2026-07-01", "2026-09-30") == 92 * 24


def _ts(url, key):
    v = url.split(key + "=")[1][:14]                       # e.g. 20260601T07:00
    return pd.Timestamp(f"{v[:4]}-{v[4:6]}-{v[6:8]} {v[9:14]}", tz="UTC")


def _oasis_xml(node, start, hours, step_minutes, value=10.0):
    items = []
    for k in range(int(hours * 60 / step_minutes)):
        ts = (start + pd.Timedelta(minutes=step_minutes * k)).strftime("%Y-%m-%dT%H:%M:%S-00:00")
        items.append(f"<REPORT_DATA><DATA_ITEM>LMP_PRC</DATA_ITEM><RESOURCE_NAME>{node}</RESOURCE_NAME>"
                     f"<INTERVAL_START_GMT>{ts}</INTERVAL_START_GMT><VALUE>{value + k}</VALUE></REPORT_DATA>")
    xml = ('<OASISReport xmlns="http://www.caiso.com/soa/OASISReport_v1.xsd"><MessagePayload><RTO>'
           + "".join(items) + "</RTO></MessagePayload></OASISReport>")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("report.xml", xml)
    return buf.getvalue()


def test_caiso_exact_requests_exact_utc_bounds_and_keeps_schema(tmp_path):
    from es.sealed_pull import caiso_exact
    seen = []

    def fake_fetch(url):
        seen.append(url)
        start, end = _ts(url, "startdatetime"), _ts(url, "enddatetime")
        node = url.split("node=")[1].split("&")[0]
        step = 5 if "PRC_INTVL_LMP" in url else 60
        return _oasis_xml(node, start.tz_localize(None), (end - start) / pd.Timedelta(hours=1), step)

    df = pd.read_parquet(caiso_exact("2026-06-01", "2026-06-02", tmp_path / "root", fetch=fake_fetch, **NO_SLEEP))
    assert list(df.columns) == ["market", "node", "node_type", "timestamp_utc", "da_lmp", "rt_lmp"]
    assert df["timestamp_utc"].min() == pd.Timestamp("2026-06-01 07:00", tz="UTC")
    assert df["timestamp_utc"].max() == pd.Timestamp("2026-06-03 06:00", tz="UTC")
    assert df.groupby("node").size().eq(48).all() and df["node"].nunique() == 6
    assert all(_ts(u, "startdatetime") == pd.Timestamp("2026-06-01 07:00", tz="UTC") for u in seen)


def test_caiso_empty_responses_are_retried_then_raised(tmp_path):
    from es.sealed_pull import caiso_exact
    calls = []

    def empty_fetch(url):
        calls.append(url)
        return _oasis_xml("X", pd.Timestamp("2026-06-01 07:00"), 0, 60)

    with pytest.raises(RuntimeError, match="attempts"):
        caiso_exact("2026-06-01", "2026-06-02", tmp_path / "root", fetch=empty_fetch, **NO_SLEEP)
    assert len(calls) == 8                                   # one chunk, eight attempts


def test_cached_responses_are_reused(tmp_path):
    from es.sealed_pull import caiso_exact
    calls = []

    def fetch(url):
        calls.append(url)
        start, end = _ts(url, "startdatetime"), _ts(url, "enddatetime")
        node = url.split("node=")[1].split("&")[0]
        return _oasis_xml(node, start.tz_localize(None), (end - start) / pd.Timedelta(hours=1),
                          5 if "PRC_INTVL_LMP" in url else 60)

    a = caiso_exact("2026-06-01", "2026-06-01", tmp_path / "a", fetch=fetch, cache_dir=tmp_path / "raw", **NO_SLEEP)
    n = len(calls)
    b = caiso_exact("2026-06-01", "2026-06-01", tmp_path / "b", fetch=fetch, cache_dir=tmp_path / "raw", **NO_SLEEP)
    assert len(calls) == n and pd.read_parquet(a).equals(pd.read_parquet(b))


def test_pull_sealed_refuses_without_registration(tmp_path, monkeypatch):
    import es.registration as registration
    from es.registration import RegistrationError
    from es.sealed_pull import pull_sealed
    monkeypatch.setattr(registration, "PROTOCOL_PATH", tmp_path / "PROTOCOL.md")
    monkeypatch.setattr(registration, "REGISTRATION_PATH", tmp_path / "REGISTRATION.md")
    with pytest.raises(RegistrationError):
        pull_sealed("A", "0" * 64, data_root=tmp_path)
    assert not (tmp_path / "sealed_2026Q3").exists()


def _price_builder(market, calls):
    def build(start, end, root):
        calls.append((market, start, end))
        ts = pd.to_datetime(["2026-06-25 18:00", "2026-07-15 18:00", "2026-10-01 18:00"], utc=True)
        p = root / f"{market.lower()}_da_rt_test.parquet"
        pd.DataFrame({"market": market, "node": "X", "node_type": "load_zone", "timestamp_utc": ts,
                      "da_lmp": [1.0, 2.0, 3.0], "rt_lmp": [1.0, 2.0, 3.0]}).to_parquet(p, index=False)
        return p
    return build


def _side_builder(sub, calls):
    def build(start, end, root):
        calls.append((sub, start, end))
        (root / sub).mkdir()
        pd.DataFrame({"node": ["X"], "timestamp_utc": pd.to_datetime(["2026-07-15 18:00"], utc=True)}).to_parquet(
            root / sub / f"{sub}_test.parquet", index=False)
    return build


def test_pull_sealed_trims_logs_manifests_and_runs_once(registered):
    from es.sealed_pull import pull_sealed, verify_manifest
    write, tmp = registered
    proto, reg, sha = write()
    calls = []
    builders = {m: _price_builder(m, calls) for m in ("NYISO", "ISONE", "MISO", "CAISO")}
    builders.update({"FMM": _side_builder("fmm", calls), "NYISO_INT": _side_builder("nyiso_rtint", calls)})
    root, _ = pull_sealed("A", sha, data_root=tmp, builders=builders, reference_dirs=[], check_days=False)
    assert ("NYISO", "2026-06", "2026-09") in calls and ("ISONE", "2026-06-21", "2026-09-30") in calls
    assert ("CAISO", "2026-06-21", "2026-09-30") in calls
    for f in root.glob("*_da_rt_*.parquet"):
        assert pd.read_parquet(f)["timestamp_utc"].max() < pd.Timestamp("2026-10-01", tz="UTC")
    log = (root / "PULL_LOG.txt").read_text()
    assert "trimmed 1 row(s)" in log and "10.17605/OSF.IO/ABCDE" in log
    assert (root / "OVERLAP_CHECK.csv").exists() and (root / "COVERAGE.csv").exists()
    verify_manifest(root)
    (root / "isone_da_rt_test.parquet").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="changed"):
        verify_manifest(root)
    with pytest.raises(FileExistsError):
        pull_sealed("A", sha, data_root=tmp, builders=builders, reference_dirs=[], check_days=False)


def test_interrupted_pull_resumes_without_redoing_finished_steps(registered):
    from es.sealed_pull import pull_sealed, verify_manifest
    write, tmp = registered
    proto, reg, sha = write()
    calls, failures = [], {"left": 1}
    builders = {m: _price_builder(m, calls) for m in ("NYISO", "ISONE", "MISO")}
    good_caiso = _price_builder("CAISO", calls)

    def flaky_caiso(start, end, root):
        if failures["left"]:
            failures["left"] -= 1
            raise OSError("nodename nor servname provided")       # DNS failure after sleep
        return good_caiso(start, end, root)

    builders.update({"CAISO": flaky_caiso, "FMM": _side_builder("fmm", calls),
                     "NYISO_INT": _side_builder("nyiso_rtint", calls)})
    kw = dict(data_root=tmp, builders=builders, reference_dirs=[], check_days=False)
    with pytest.raises(OSError):
        pull_sealed("A", sha, **kw)
    with pytest.raises(FileExistsError, match="interrupted pull; call again with resume=True"):
        pull_sealed("A", sha, **kw)
    root, _ = pull_sealed("A", sha, resume=True, **kw)
    assert [c[0] for c in calls].count("NYISO") == 1          # finished steps are not redone
    verify_manifest(root)
    assert not (root / ".state.json").exists()
    assert (root / "PULL_LOG.txt").read_text().count("pull_started_utc") == 2
    with pytest.raises(FileExistsError, match="PULLS.json records a completed pull"):
        pull_sealed("A", sha, resume=True, **kw)


def test_overlap_check_counts_matches_and_differences(tmp_path):
    from es.sealed_pull import overlap_check
    ts = pd.to_datetime(["2026-06-25 14:00", "2026-06-25 15:00", "2026-06-25 16:00"], utc=True)
    base = pd.DataFrame({"market": "ISONE", "node": "X", "node_type": "hub", "timestamp_utc": ts,
                         "da_lmp": [1.0, 2.0, 3.0], "rt_lmp": [1.0, 2.0, 3.0]})
    (tmp_path / "ref").mkdir()
    (tmp_path / "new").mkdir()
    base.to_parquet(tmp_path / "ref" / "isone_da_rt_ref.parquet", index=False)
    new = base.iloc[:2].copy()
    new.loc[1, "rt_lmp"] = 2.5
    new.to_parquet(tmp_path / "new" / "isone_da_rt_new.parquet", index=False)
    out = overlap_check(tmp_path / "new", [tmp_path / "ref"], "2026-06-21", "2026-07-01").set_index("market")
    assert out.loc["ISONE", "matched"] == 2 and out.loc["ISONE", "reference_only"] == 1
    assert out.loc["ISONE", "rows_differing"] == 1 and out.loc["ISONE", "max_abs_diff_rt"] == pytest.approx(0.5)


# =====================================================================================================
# Helpers shared by the tests that follow. Fakes only; every request a test makes goes to a function it injects.
# =====================================================================================================
T0 = pd.Timestamp("2026-06-01 07:00", tz="UTC")                        # 1 June 00:00 PDT
T = lambda stamp: pd.Timestamp(stamp, tz="UTC")
STEPS = ("NYISO", "ISONE", "MISO", "CAISO", "FMM", "NYISO_INT")
MARKETS = ("NYISO", "ISONE", "MISO", "CAISO")
PRICE_COLUMNS = ["market", "node", "node_type", "timestamp_utc", "da_lmp", "rt_lmp"]
FMM_COLUMNS = ["INTERVALSTARTTIME_GMT", "INTERVALENDTIME_GMT", "OPR_DT", "OPR_HR", "NODE_ID_XML", "NODE_ID", "NODE",
               "MARKET_RUN_ID", "LMP_TYPE", "XML_DATA_ITEM", "PNODE_RESMRID", "GRP_TYPE", "POS", "PRC",
               "OPR_INTERVAL", "GROUP"]
FMM_ITEM = {"LMP": "LMP_PRC", "MCC": "LMP_CONG_PRC", "MCE": "LMP_ENE_PRC", "MCL": "LMP_LOSS_PRC"}
NYISO_HEADER = ('"Time Stamp","Name","PTID","LBMP ($/MWHr)","Marginal Cost Losses ($/MWHr)",'
                '"Marginal Cost Congestion ($/MWHr)"')
SDGE = "DLAP_SDGE-APND"


def _node(url):
    return url.split("node=")[1].split("&")[0]


def _grid(start, end, step_minutes, price_of, inclusive_end=False):
    """[(interval start, price)] from start up to end (end included when inclusive_end)."""
    out, t = [], start
    while t < end or (inclusive_end and t == end):
        out.append((t, price_of(t)))
        t += pd.Timedelta(minutes=step_minutes)
    return out


def _stamp(t):
    return t.strftime("%Y-%m-%dT%H:%M:%S-00:00")


def _oasis_zip(node, priced):
    """A PRC_LMP or PRC_INTVL_LMP answer (XML in a zip) for [(interval start, price)]."""
    items = "".join(f"<REPORT_DATA><DATA_ITEM>LMP_PRC</DATA_ITEM><RESOURCE_NAME>{node}</RESOURCE_NAME>"
                    f"<INTERVAL_START_GMT>{_stamp(t)}</INTERVAL_START_GMT><VALUE>{v}</VALUE></REPORT_DATA>"
                    for t, v in priced)
    xml = ('<OASISReport xmlns="http://www.caiso.com/soa/OASISReport_v1.xsd"><MessagePayload><RTO>'
           + items + "</RTO></MessagePayload></OASISReport>")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("report.xml", xml)
    return buf.getvalue()


def _fmm_zip(node, priced, repeated=None):
    """A PRC_RTPD_LMP answer (resultformat=6): a zipped CSV, one row per 15-minute interval and price component.
    The LMP component carries the price and the other three are 900, far from it on purpose. The interval at
    position `repeated` of `priced` arrives twice."""
    rows = [{"INTERVALSTARTTIME_GMT": _stamp(t), "INTERVALENDTIME_GMT": _stamp(t + pd.Timedelta(minutes=15)),
             "OPR_DT": "2026-06-01", "OPR_HR": 1, "NODE_ID_XML": node, "NODE_ID": node, "NODE": node,
             "MARKET_RUN_ID": "RTPD", "LMP_TYPE": kind, "XML_DATA_ITEM": FMM_ITEM[kind], "PNODE_RESMRID": node,
             "GRP_TYPE": "ALL_APNODES", "POS": 0, "PRC": v if kind == "LMP" else 900.0, "OPR_INTERVAL": 1, "GROUP": 1}
            for t, v in priced for kind in FMM_ITEM]
    df = pd.DataFrame(rows, columns=FMM_COLUMNS)
    if repeated is not None:
        n = len(FMM_ITEM)
        df = pd.concat([df, df.iloc[repeated * n:(repeated + 1) * n]], ignore_index=True)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("report.csv", df.to_csv(index=False))
    return buf.getvalue()


def _nyiso_zip(days):
    """A monthly NYISO 'rtlbmp' zip of daily CSVs. days: {'YYYYMMDD': [(stamp, name, lbmp)]}."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for day, records in days.items():
            lines = [NYISO_HEADER] + [f'"{s}","{n}",61761,{v},1.0,2.0' for s, n, v in records]
            z.writestr(f"{day}rtlbmp_zone.csv", "\n".join(lines) + "\n")
    return buf.getvalue()


def _kind(url):
    return "fmm" if "PRC_RTPD_LMP" in url else ("rtm" if "PRC_INTVL_LMP" in url else "dam")


@functools.lru_cache(maxsize=None)
def _blob(kind, node, start, end, days=None):
    """The answer to one OASIS request: the whole interval grid of [start, end), or of its first `days` days.
    Memoised across tests (zip headers carry the clock, so the same bytes also keep byte comparisons stable)."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    if days is not None:
        e = min(e, s + pd.Timedelta(days=days))
    if kind == "fmm":
        return _fmm_zip(node, _grid(s, e, 15, lambda t: 100.0))
    return _oasis_zip(node, _grid(s, e, 5 if kind == "rtm" else 60, lambda t: 10.0))


def _good_oasis(url, days=None):
    """The complete answer to a PRC_LMP, PRC_INTVL_LMP or PRC_RTPD_LMP request, or only its first `days` days (a
    short answer)."""
    return _blob(_kind(url), _node(url), str(_ts(url, "startdatetime")), str(_ts(url, "enddatetime")), days)


def _memoised_oasis():
    """A fetch that answers every URL with the same complete bytes each time."""
    calls = []

    def fetch(url):
        calls.append(url)
        return _good_oasis(url)

    return fetch, calls


def _complete_builders(calls):
    """The brief's fakes for all six steps (one node of their own; pull them with check_days=False)."""
    builders = {m: _price_builder(m, calls) for m in MARKETS}
    builders.update({"FMM": _side_builder("fmm", calls), "NYISO_INT": _side_builder("nyiso_rtint", calls)})
    return builders


def _stamped_builder(market, stamps, node="X", calls=None, rt=1.0):
    def build(start, end, root):
        if calls is not None:
            calls.append((market, start, end))
        stamped = pd.to_datetime(list(stamps), utc=True)
        pd.DataFrame({"market": market, "node": node, "node_type": "hub", "timestamp_utc": stamped,
                      "da_lmp": 1.0, "rt_lmp": rt}).to_parquet(root / f"{market.lower()}_da_rt_test.parquet",
                                                                index=False)
    return build


def _a_pull(tmp, sha, calls=None, **kw):
    """Test A pulled with the brief's fakes into tmp. Returns (root, manifest_sha256)."""
    from es.sealed_pull import pull_sealed
    calls = [] if calls is None else calls
    kw = {"data_root": tmp, "builders": _complete_builders(calls), "reference_dirs": [], "check_days": False, **kw}
    return pull_sealed("A", sha, **kw)


def _pulls(tmp):
    path = tmp / "docs_sealed" / "PULLS.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _node_map_builder(market, calls, stamps=("2026-06-25 18:00", "2026-07-15 18:00", "2026-07-15 19:00",
                                              "2026-10-01 18:00"), skip=(), extra=()):
    """A July-downloader fake: the node map's nodes of `market` less `skip`, plus `extra` data-only nodes."""
    def build(start, end, root):
        from es.sealed_pull import _node_map_nodes
        calls.append((market, start, end))
        nodes = [n for m, n in _node_map_nodes() if m == market and (m, n) not in skip] + list(extra)
        stamps_utc = pd.to_datetime(list(stamps), utc=True)
        pd.DataFrame([(market, n, "hub", t, 1.0, 1.0) for n in nodes for t in stamps_utc],
                     columns=PRICE_COLUMNS).to_parquet(root / f"{market.lower()}_da_rt_test.parquet", index=False)
    return build


DEV_WINDOW = ("2026-06-21", "2026-06-22", "2026-06-23")                  # pull start, deployment start, deployment end


def _nodes_by_market():
    from es.sealed_pull import _node_map_nodes
    out = {}
    for market, node in _node_map_nodes():
        out.setdefault(market, []).append(node)
    return out


def _clock_hours(market, first, last):
    """Every clock hour (UTC instants) of the local days first..last ('YYYY-MM-DD') of a market: 24 a day, 23 on a
    spring-forward day, 25 on a fall-back day, and 24 every day for MISO (fixed EST)."""
    from es.config import SETTLE_TZ
    from es.sealed_pull import local_day_bounds_utc
    s, e = local_day_bounds_utc(first, last, SETTLE_TZ[market])
    return pd.date_range(s, e, freq="h", inclusive="left")


def _downloader_hours(market, first, last):
    """The hours the July downloaders of NYISO and ISO-NE (and the NYISO integrated parser) keep: the clock hours
    without the two rows of the repeated hour of a fall-back day (23 of 25). The other markets keep every hour."""
    from es.config import SETTLE_TZ
    hours = _clock_hours(market, first, last)
    if market not in ("NYISO", "ISONE"):
        return hours
    wall = pd.Series(hours.tz_convert(SETTLE_TZ[market]).tz_localize(None))
    return hours[~wall.duplicated(keep=False).to_numpy()]


def _around(first, last):
    """The day before `first` and the day after `last` ('YYYY-MM-DD'): the fakes write a day on each side of a pull."""
    day = pd.Timedelta(days=1)
    return f"{pd.Timestamp(first) - day:%Y-%m-%d}", f"{pd.Timestamp(last) + day:%Y-%m-%d}"


def _full_price_rows(market, nodes, first, last, skip=(), prices=None):
    """The rows a July downloader returns for `nodes` over the local days first..last ('YYYY-MM-DD'): every hour
    (_downloader_hours), every price 1.0 unless `prices` has (node, 'YYYY-MM-DD', local hour): (da_lmp, rt_lmp);
    `skip` holds (node, 'YYYY-MM-DD') pairs to leave out."""
    from es.config import SETTLE_TZ
    prices = prices or {}
    local = _downloader_hours(market, first, last).tz_convert(SETTLE_TZ[market])
    return [(market, n, "hub", t.tz_convert("UTC"), *prices.get((n, f"{t:%Y-%m-%d}", t.hour), (1.0, 1.0)))
            for n in nodes for t in local if (n, f"{t:%Y-%m-%d}") not in skip]


@functools.lru_cache(maxsize=None)
def _nyiso_month_blob(month, first, last, extra=()):
    """The NYISO monthly archive for 'YYYYMM': every node-map zone, every clock hour of the local days from the day
    before `first` to the day after `last` that fall in the month (the two rows of a repeated hour are both there; the
    parser drops them), plus the `extra` records [(stamp, name, value)]."""
    zones = _nodes_by_market()["NYISO"]
    by_day = {}
    for t in _clock_hours("NYISO", *_around(first, last)).tz_convert("America/New_York"):
        if f"{t:%Y%m}" == month:
            by_day.setdefault(f"{t:%Y%m%d}", []).extend((f"{t:%m/%d/%Y %H:%M}", z, 5.0) for z in zones)
    for stamp, name, value in extra:
        if f"{stamp[6:10]}{stamp[:2]}" == month:                      # an archive holds the records of its own month
            by_day.setdefault(f"{stamp[6:10]}{stamp[:2]}{stamp[3:5]}", []).append((stamp, name, value))
    return _nyiso_zip(by_day)


def _use_default_builders(monkeypatch, urls, fail=lambda url: False, window=DEV_WINDOW, skip=None,
                          builder_calls=None, nyiso_extra=(), fetch_override=None, prices=None, drop=()):
    """Set up pull_sealed("A", ...) to run its own default builders, with the three July downloaders and the network
    replaced by fakes and the pauses removed.
    - The window of Test A moves to `window` (development dates), unless window is None.
    - The July downloaders' fakes write every node-map node of their market for every hour of every day of the pull,
      one day before it and one day after it (as the downloaders do: _downloader_hours), except the
      (market, node, 'YYYY-MM-DD') triples in `skip`, a set the test may change. Every price is 1.0 unless `prices`
      has (market, node, 'YYYY-MM-DD', local hour): (da_lmp, rt_lmp) for the row; the rows of the (market, node,
      'YYYY-MM-DD', local hour) keys in `drop` are left out, as a downloader leaves out a row it cannot read.
    - OASIS is answered with complete interval grids, NYISO with complete monthly archives (plus nyiso_extra records).
    - fetch_override(url) may return other bytes for a URL, or None to leave it to the fake."""
    import v2.ingest_isone
    import v2.ingest_miso
    import v2.ingest_nyiso

    import es.sealed_pull as sp

    skip = set() if skip is None else skip
    prices = {} if prices is None else prices
    nodes = _nodes_by_market()
    if window is not None:
        monkeypatch.setitem(sp.SEALED_WINDOWS, "A", (*window, "pull_unit"))

    def july_downloader(market):
        def build(start, end, root):
            if builder_calls is not None:
                builder_calls.append(market)
            from es.config import SETTLE_TZ
            first, _, last, _ = sp.SEALED_WINDOWS["A"]
            local = _downloader_hours(market, *_around(first, last)).tz_convert(SETTLE_TZ[market])
            stamps = [(t.tz_convert("UTC"), f"{t:%Y-%m-%d}", t.hour) for t in local]
            rows = [(market, n, "hub", t, *prices.get((market, n, day, hour), (1.0, 1.0)))
                    for n in nodes[market] for t, day, hour in stamps
                    if (market, n, day) not in skip and (market, n, day, hour) not in drop]
            pd.DataFrame(rows, columns=PRICE_COLUMNS).to_parquet(root / f"{market.lower()}_da_rt_x.parquet",
                                                                 index=False)
        return build

    for module, market in ((v2.ingest_nyiso, "NYISO"), (v2.ingest_isone, "ISONE"), (v2.ingest_miso, "MISO")):
        monkeypatch.setattr(module, "build", july_downloader(market))

    def fake_fetch(url, attempts=5):
        urls.append(url)
        if fail(url):
            raise OSError("nodename nor servname provided")
        if fetch_override is not None:
            got = fetch_override(url)
            if got is not None:
                return got
        if "mis.nyiso.com" in url:
            first, _, last, _ = sp.SEALED_WINDOWS["A"]
            return _nyiso_month_blob(url.rsplit("/", 1)[1][:6], first, last, tuple(nyiso_extra))
        return _good_oasis(url)

    monkeypatch.setattr(sp, "_fetch", fake_fetch)
    for name in ("caiso_exact", "caiso_fmm_exact", "nyiso_integrated_months"):
        monkeypatch.setattr(sp, name, functools.partial(getattr(sp, name), pause=0, sleep=lambda _s: None))


# ----------------------------------------------------------------------------- C1: the gate is the module's alone
def test_pull_sealed_takes_no_registration_argument(registered):
    import inspect

    from es.sealed_pull import pull_sealed
    write, tmp = registered
    proto, reg, sha = write()
    parameters = inspect.signature(pull_sealed).parameters
    assert list(parameters) == [
        "test_id", "protocol_sha256", "data_root", "builders", "reference_dirs", "resume", "allow_missing_days",
        "check_days", "allow_incomplete_chunks"]
    keyword_only = [n for n, q in parameters.items() if q.kind is inspect.Parameter.KEYWORD_ONLY]
    assert keyword_only == ["allow_incomplete_chunks"]
    with pytest.raises(TypeError, match="registration"):                  # the old way of handing over a record is gone
        pull_sealed("A", sha, data_root=tmp, registration=(proto, reg), builders=_complete_builders([]),
                    reference_dirs=[], check_days=False)
    with pytest.raises(TypeError, match="takes from 2 to 8 positional arguments but 9 were given"):
        pull_sealed("A", sha, tmp, _complete_builders([]), [], False, None, False, {"fmm": []})   # nor a 9th argument
    assert not (tmp / "sealed_2026Q3").exists()


def test_a_record_the_gate_does_not_point_at_opens_nothing(registered, monkeypatch):
    import es.registration as registration
    from es.registration import RegistrationError
    from es.sealed_pull import pull_sealed
    write, tmp = registered
    proto, reg, sha = write()                                       # a perfectly valid record, at the fixture's paths
    monkeypatch.setattr(registration, "PROTOCOL_PATH", tmp / "elsewhere" / "PROTOCOL.md")
    monkeypatch.setattr(registration, "REGISTRATION_PATH", tmp / "elsewhere" / "REGISTRATION.md")
    calls = []
    with pytest.raises(RegistrationError, match="no registration record"):
        pull_sealed("A", sha, data_root=tmp, builders=_complete_builders(calls), reference_dirs=[], check_days=False)
    assert calls == [] and not (tmp / "sealed_2026Q3").exists()


def test_the_default_builders_cannot_reach_the_network_under_the_test_block(registered, socket_attempts):
    """The reviewer's check A: a pull with the real default builders and nothing replaced. The block stops the
    first request, and no socket is touched."""
    from es.sealed_pull import pull_sealed
    write, tmp = registered
    proto, reg, sha = write()
    before = len(socket_attempts)
    with pytest.raises(AssertionError, match="network access attempted"):
        pull_sealed("A", sha, data_root=tmp)
    assert len(socket_attempts) == before


def test_the_socket_layer_is_closed_and_counts_an_attempt(socket_attempts):
    import socket
    before = len(socket_attempts)
    with pytest.raises(AssertionError, match="network access attempted"):
        socket.create_connection(("example.invalid", 80))
    with pytest.raises(AssertionError, match="network access attempted"):
        socket.getaddrinfo("example.invalid", 80)
    sock = socket.socket()
    try:
        with pytest.raises(AssertionError, match="network access attempted"):
            sock.connect(("127.0.0.1", 9))
    finally:
        sock.close()
    assert [a[1] for a in socket_attempts[before:]] == ["create_connection", "getaddrinfo", "connect"]
    del socket_attempts[before:]                                    # these three were made on purpose


@pytest.mark.parametrize("module", ["v2.ingest_nyiso", "v2.ingest_isone", "v2.ingest_miso"])
def test_the_july_downloaders_own_urlopen_is_blocked(module):
    import importlib
    with pytest.raises(AssertionError, match="network access attempted"):
        importlib.import_module(module).urlopen("http://example.invalid/")


# ----------------------------------------------------------------------------- I1: the readers stop at the seal
@pytest.mark.parametrize("market, utc_offset_hours, naive", [("NYISO", 4, False), ("ISONE", 4, False),
                                                              ("MISO", 5, True), ("CAISO", 7, False)])
def test_the_readers_never_load_a_row_at_or_after_the_seal(tmp_path, monkeypatch, market, utc_offset_hours, naive):
    """A stand-in for the H1 CAISO file (one row at 1 July 00:00 PDT), for every market's own midnight."""
    import pyarrow.parquet as pq

    from es.sealed_pull import _missing_days, coverage_manifest, overlap_check
    seal = T("2026-07-01") + pd.Timedelta(hours=utc_offset_hours)
    stamps = [seal - pd.Timedelta(hours=1), seal, seal + pd.Timedelta(hours=5)]
    if naive:
        stamps = [s.tz_localize(None) for s in stamps]
    h1 = tmp_path / "h1like"
    h1.mkdir()
    pd.DataFrame({"market": market, "node": "X", "node_type": "hub", "timestamp_utc": stamps,
                  "da_lmp": [1.0, 999.0, 999.0], "rt_lmp": [1.0, 999.0, 999.0]}).to_parquet(
        h1 / f"{market.lower()}_da_rt_h1like.parquet", index=False)
    real, read = pq.read_table, []

    def spy(*args, **kwargs):
        table = real(*args, **kwargs)
        read.append(table)
        return table

    monkeypatch.setattr(pq, "read_table", spy)
    calls = {"coverage_manifest": lambda: coverage_manifest(h1, "2026-06-01", "2026-06-30"),
             "overlap_check (end past the seal)": lambda: overlap_check(h1, [h1], "2026-06-21", "2026-07-02"),
             "_missing_days": lambda: _missing_days(h1, market, "2026-06-01", "2026-07-02")}
    for label, call in calls.items():
        read.clear()
        call()
        assert read, f"{label} did not read through a predicate"
        for table in read:
            loaded = pd.to_datetime(table.column("timestamp_utc").to_pandas(), utc=True)
            assert len(loaded) == 1 and loaded.max() < seal, label          # only the hour before the seal


def test_the_private_sealed_flag_lifts_the_cap_and_only_pull_sealed_passes_it(tmp_path):
    from es.sealed_pull import coverage_manifest, overlap_check
    h1 = tmp_path / "h1like"
    h1.mkdir()
    stamps = pd.to_datetime(["2026-06-30 18:00", "2026-07-01 07:00"], utc=True)  # 30 June 11:00 | 1 July 00:00 PDT
    pd.DataFrame({"market": "CAISO", "node": "TH_SP15_GEN-APND", "node_type": "hub", "timestamp_utc": stamps,
                  "da_lmp": [1.0, 2.0], "rt_lmp": [1.0, 2.0]}).to_parquet(h1 / "caiso_da_rt_x.parquet", index=False)

    def hours(df):
        return int(df[df["node"] == "TH_SP15_GEN-APND"]["hours"].iloc[0])

    assert hours(coverage_manifest(h1, "2026-06-01", "2026-07-02")) == 1
    assert hours(coverage_manifest(h1, "2026-06-01", "2026-07-02", _sealed=True)) == 2
    capped = overlap_check(h1, [h1], "2026-06-21", "2026-07-02").set_index("market")
    lifted = overlap_check(h1, [h1], "2026-06-21", "2026-07-02", _sealed=True).set_index("market")
    assert capped.loc["CAISO", "matched"] == 1 and lifted.loc["CAISO", "matched"] == 2


# =====================================================================================================
# Tests from the earlier rounds (Rulings 58 and 60), kept and adapted to fix round 1.
# =====================================================================================================
def test_caiso_fmm_exact_uses_exact_bounds_the_lmp_component_and_the_hourly_mean(tmp_path):
    from es.sealed_pull import caiso_fmm_exact
    seen = []
    price = lambda t: 100.0 + (t - T0) / pd.Timedelta(minutes=15)             # one price per 15-minute interval

    def fetch(url):
        seen.append(url)
        start, end = _ts(url, "startdatetime"), _ts(url, "enddatetime")
        return _fmm_zip(_node(url), _grid(start, end, 15, price), repeated=2)    # the window's third interval twice

    df = pd.read_parquet(caiso_fmm_exact("2026-06-01", "2026-06-01", tmp_path / "fmm", fetch=fetch, **NO_SLEEP))
    assert list(df.columns) == ["node", "timestamp_utc", "fmm_lmp", "n15"]
    assert df["timestamp_utc"].min() == T0 and df["timestamp_utc"].max() == T0 + pd.Timedelta(hours=23)
    assert df.groupby("node").size().eq(24).all() and df["node"].nunique() == 6 and df["n15"].eq(4).all()
    sp15 = df[df["node"] == "TH_SP15_GEN-APND"].set_index("timestamp_utc")["fmm_lmp"]
    assert sp15[T0] == pytest.approx(101.5)                                   # mean(100, 101, 102, 103)
    assert sp15[T0 + pd.Timedelta(hours=2)] == pytest.approx(109.5)           # mean(108, 109, 110, 111)
    assert len(seen) == 6 and all("queryname=PRC_RTPD_LMP" in u and "market_run_id=RTPD" in u for u in seen)
    assert all(u.endswith("&version=1&resultformat=6") for u in seen)
    assert {(_ts(u, "startdatetime"), _ts(u, "enddatetime")) for u in seen} == {(T0, T0 + pd.Timedelta(hours=24))}


def test_nyiso_integrated_months_requests_each_month_dedupes_and_caches(tmp_path):
    from es.sealed_pull import nyiso_integrated_months
    from es.settlement import NYISO_URL
    seen = []

    def fetch(url):
        seen.append(url)
        if "20260501" in url:
            return _nyiso_zip({"20260531": [("05/31/2026 22:00", "N.Y.C.", 10.0),
                                            ("05/31/2026 23:00", "N.Y.C.", 11.0)]})
        return _nyiso_zip({"20260601": [("06/01/2026 00:00", "N.Y.C.", 12.0), ("06/01/2026 00:00", "N.Y.C.", 12.0)]})

    kw = dict(fetch=fetch, cache_dir=tmp_path / "raw", **NO_SLEEP)
    out = nyiso_integrated_months("2026-05", "2026-06", tmp_path / "nyiso_rtint", **kw)
    df = pd.read_parquet(out)
    assert out.name == "nyiso_rtint_202605_202606.parquet"
    assert list(df.columns) == ["node", "timestamp_utc", "rt_int_lmp"]
    assert seen == [NYISO_URL.format(ym="202605"), NYISO_URL.format(ym="202606")]
    assert df["timestamp_utc"].tolist() == [pd.Timestamp(s, tz="UTC") for s in
                                            ("2026-06-01 02:00", "2026-06-01 03:00", "2026-06-01 04:00")]
    assert df["rt_int_lmp"].tolist() == [10.0, 11.0, 12.0]                    # the row sent twice counts once
    nyiso_integrated_months("2026-05", "2026-06", tmp_path / "again", **kw)
    assert len(seen) == 2                                                     # the second run came from the cache


def test_retries_back_off_from_6_to_60_seconds_and_keep_the_cause(tmp_path):
    from es.sealed_pull import caiso_exact
    sleeps = []

    def dead_fetch(url):
        raise OSError("nodename nor servname provided")

    with pytest.raises(RuntimeError, match="after 8 attempts") as err:
        caiso_exact("2026-06-01", "2026-06-01", tmp_path / "root", fetch=dead_fetch, pause=0, sleep=sleeps.append)
    assert sleeps == [6, 12, 24, 48, 60, 60, 60]             # seven pauses between eight attempts, none after the last
    assert isinstance(err.value.__cause__, OSError)


def test_error_bodies_and_network_errors_are_retried_and_the_good_response_is_cached(tmp_path):
    from es.sealed_pull import caiso_exact
    calls, sleeps = [], []

    def flaky(url):
        calls.append(url)
        if len(calls) == 1:
            raise OSError("nodename nor servname provided")
        if len(calls) == 2:
            return b"<html>429 Too Many Requests</html>"                  # an error page, not a zip
        return _good_oasis(url)

    cache = tmp_path / "raw"
    df = pd.read_parquet(caiso_exact("2026-06-01", "2026-06-01", tmp_path / "root", fetch=flaky, pause=5.5,
                                     sleep=sleeps.append, cache_dir=cache))
    assert len(df) == 6 * 24 and len(calls) == 12 + 2                      # 12 requests, the first one tried 3 times
    assert calls[0] == calls[1] == calls[2] and sleeps[:3] == [6, 12, 5.5] and sleeps.count(5.5) == 12
    assert len(list(cache.glob("*.bin"))) == 12                            # only good responses are cached


def test_a_window_over_30_days_goes_out_in_contiguous_chunks_and_a_duplicate_row_changes_no_hourly_mean(tmp_path):
    """32 days go out as a 30-day and a 2-day chunk. A response that repeats a row inside itself (OASIS has done worse)
    must still give the mean of the twelve distinct 5-minute prices."""
    from es.sealed_pull import caiso_exact
    seen = []
    da_price = lambda t: 20.0 + (t - T0) / pd.Timedelta(hours=1)
    rt_price = lambda t: 50.0 + (t - T0) / pd.Timedelta(minutes=5)

    def fetch(url):
        seen.append(url)
        start, end = _ts(url, "startdatetime"), _ts(url, "enddatetime")
        grid = _grid(start, end, 5 if "PRC_INTVL_LMP" in url else 60, rt_price if "PRC_INTVL_LMP" in url else da_price)
        return _oasis_zip(_node(url), grid + grid[:1])                      # the first interval twice

    df = pd.read_parquet(caiso_exact("2026-05-15", "2026-06-15", tmp_path / "root", fetch=fetch, **NO_SLEEP))
    first = pd.Timestamp("2026-05-15 07:00", tz="UTC")
    assert len(seen) == 2 * 6 * 2                                           # 32 days: a 30-day and a 2-day chunk
    assert {_ts(u, "startdatetime") for u in seen} == {first, first + pd.Timedelta(days=30)}
    assert df["timestamp_utc"].min() == first
    assert df["timestamp_utc"].max() == pd.Timestamp("2026-06-16 06:00", tz="UTC")
    assert df.groupby("node").size().eq(32 * 24).all()
    assert (df["da_lmp"] - df["timestamp_utc"].map(da_price)).abs().max() < 1e-9
    assert (df["rt_lmp"] - (df["timestamp_utc"].map(rt_price) + 5.5)).abs().max() < 1e-9   # mean of 0..11 is 5.5


@pytest.mark.parametrize("market, utc_offset_hours, naive", [("NYISO", 4, False), ("ISONE", 4, False),
                                                              ("MISO", 5, True), ("MISO", 5, False),
                                                              ("CAISO", 7, False)])
def test_trim_cuts_at_each_markets_local_midnight(tmp_path, market, utc_offset_hours, naive):
    """Local midnight in June is 04:00 UTC in New York, 05:00 UTC for MISO (fixed EST) and 07:00 UTC in California.
    v2/ingest_miso writes timezone-naive UTC stamps; the other downloaders write tz-aware ones."""
    from es.sealed_pull import _trim
    midnight = lambda day: pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=utc_offset_hours)
    stamps = [midnight("2026-06-01") - pd.Timedelta(hours=1), midnight("2026-06-01"),      # last hour before | first in
              midnight("2026-06-16") - pd.Timedelta(hours=1), midnight("2026-06-16")]      # last hour in | first after
    if naive:
        stamps = [s.tz_localize(None) for s in stamps]
    path = tmp_path / f"{market.lower()}_da_rt_t.parquet"
    pd.DataFrame({"market": market, "node": "X", "node_type": "hub", "timestamp_utc": stamps,
                  "da_lmp": 1.0, "rt_lmp": 1.0}).to_parquet(path, index=False)
    assert _trim(path, "2026-06-01", "2026-06-15") == 2
    assert pd.read_parquet(path)["timestamp_utc"].tolist() == stamps[1:3]


def test_pull_reads_tz_naive_miso_beside_tz_aware_files(registered):
    from es.sealed_pull import pull_sealed
    write, tmp = registered
    proto, reg, sha = write()
    calls = []

    def naive_miso(start, end, root):
        calls.append(("MISO", start, end))
        ts = pd.to_datetime(["2026-06-25 18:00", "2026-07-15 18:00", "2026-10-01 18:00"])        # no utc=True
        pd.DataFrame({"market": "MISO", "node": "X", "node_type": "trading_hub", "timestamp_utc": ts,
                      "da_lmp": [1.0, 2.0, 3.0], "rt_lmp": [1.0, 2.0, 3.0]}).to_parquet(
            root / "miso_da_rt_test.parquet", index=False)

    builders = {m: _price_builder(m, calls) for m in ("NYISO", "ISONE", "CAISO")}
    builders.update({"MISO": naive_miso, "FMM": _side_builder("fmm", calls),
                     "NYISO_INT": _side_builder("nyiso_rtint", calls)})
    ref = tmp / "ref"                                       # an opened pull: the 25 June row of two markets
    ref.mkdir()
    ts = pd.to_datetime(["2026-06-25 18:00"])
    pd.DataFrame({"market": "MISO", "node": "X", "node_type": "trading_hub", "timestamp_utc": ts,
                  "da_lmp": 1.0, "rt_lmp": 1.0}).to_parquet(ref / "miso_da_rt_ref.parquet", index=False)
    pd.DataFrame({"market": "ISONE", "node": "X", "node_type": "load_zone", "timestamp_utc": ts.tz_localize("UTC"),
                  "da_lmp": 1.0, "rt_lmp": 9.0}).to_parquet(ref / "isone_da_rt_ref.parquet", index=False)
    root, _ = pull_sealed("A", sha, data_root=tmp, builders=builders, reference_dirs=["ref"], check_days=False)
    overlap = pd.read_csv(root / "OVERLAP_CHECK.csv")
    overlap = overlap[overlap["series"] == "price"].set_index("market")
    assert overlap.loc["MISO", "matched"] == 1 and overlap.loc["MISO", "rows_differing"] == 0
    assert overlap.loc["ISONE", "matched"] == 1 and overlap.loc["ISONE", "rows_differing"] == 1
    assert overlap.loc["NYISO", "new_only"] == 1 and overlap.loc["CAISO", "new_only"] == 1
    assert set(pd.read_csv(root / "COVERAGE.csv")["market"]) == {"NYISO", "ISONE", "MISO", "CAISO"}
    assert pd.read_parquet(root / "miso_da_rt_test.parquet")["timestamp_utc"].tolist() == list(
        pd.to_datetime(["2026-06-25 18:00", "2026-07-15 18:00"]))             # trimmed, and still tz-naive


def test_default_builders_cache_responses_and_write_the_side_directories(registered, monkeypatch):
    from es.sealed_pull import pull_sealed, verify_manifest
    from es.settlement import load_caiso_fmm, load_nyiso_integrated
    write, tmp = registered
    proto, reg, sha = write()
    urls = []
    _use_default_builders(monkeypatch, urls)
    root, _ = pull_sealed("A", sha, data_root=tmp, reference_dirs=[])
    assert root == tmp / "pull_unit" and len(urls) == 12 + 6 + 1             # DA and RT, FMM per node, one NYISO month
    assert len(list((root / "raw").glob("*.bin"))) == 19                     # every response is cached ...
    manifest = (root / "MANIFEST.sha256").read_text().splitlines()
    assert sum(line.split("  ", 1)[1].startswith("raw/") for line in manifest) == 19      # ... and in the manifest
    assert len(load_caiso_fmm(root / "fmm", dev_only=False)) == 6 * 72       # six nodes, three days of hours
    assert len(load_nyiso_integrated(root / "nyiso_rtint", dev_only=False)) == 10 * 72  # ten zones, three days of hours
    log = (root / "PULL_LOG.txt").read_text()
    for key in ("NYISO", "ISONE", "MISO", "CAISO", "fmm", "nyiso_rtint"):  # every day check ran, and says so
        assert (f"step {key}: every node-map node has the hours of every day "
                "from 2026-06-21 to 2026-06-23, with finite prices") in log
    caiso_log = (root / "logs" / "CAISO.log").read_text()                    # one line per request, with its size
    assert caiso_log.count("bytes on attempt 1") == 12 and "PRC_INTVL_LMP TH_SP15_GEN-APND" in caiso_log
    verify_manifest(root)


def test_resumed_default_pull_requests_only_what_is_missing(registered, monkeypatch):
    """The Task 6 failure: the FMM pull died part-way. On resume only the nodes after the failure are requested."""
    from es.sealed_pull import pull_sealed, verify_manifest
    write, tmp = registered
    proto, reg, sha = write()
    urls, broken = [], {"on": True}
    _use_default_builders(monkeypatch, urls,
                          fail=lambda url: broken["on"] and "PRC_RTPD_LMP" in url and "node=DLAP_PGAE-APND" in url)
    kw = dict(data_root=tmp, reference_dirs=[])
    with pytest.raises(RuntimeError, match="attempts"):
        pull_sealed("A", sha, **kw)
    assert len(urls) == 12 + 3 + 8                           # DA and RT, three FMM nodes, 8 failed tries
    fmm_log = (tmp / "pull_unit" / "logs" / "FMM.log").read_text()
    assert all(f"attempt {k}/8 failed" in fmm_log for k in range(1, 9)) and "OSError" in fmm_log
    urls.clear()
    broken["on"] = False
    root, _ = pull_sealed("A", sha, resume=True, **kw)
    assert (root / "logs" / "FMM.log").read_text().count("attempt started") == 2           # both attempts are on record
    assert [("PRC_RTPD_LMP" in u, _node(u) if "oasis" in u else "nyiso") for u in urls] == [
        (True, "DLAP_PGAE-APND"), (True, "DLAP_SCE-APND"), (True, "DLAP_SDGE-APND"), (False, "nyiso")]
    verify_manifest(root)


@pytest.mark.parametrize("day, hours", [("2025-11-02", 25), ("2026-03-08", 23), ("2026-06-15", 24)])
def test_caiso_exact_returns_the_clock_hours_of_a_dst_day(tmp_path, day, hours):
    """Fall back (25 h), spring forward (23 h) and an ordinary day, on development dates."""
    from es.sealed_pull import caiso_exact, expected_hours
    df = pd.read_parquet(caiso_exact(day, day, tmp_path / "root", fetch=_good_oasis, **NO_SLEEP))
    assert expected_hours("CAISO", day, day) == hours
    assert df["node"].nunique() == 6 and df.groupby("node").size().eq(hours).all()
    assert df.groupby("node")["timestamp_utc"].apply(lambda s: s.is_unique).all()          # one row per node and hour
    assert (df["timestamp_utc"].max() - df["timestamp_utc"].min()) == pd.Timedelta(hours=hours - 1)  # no gap


# ----------------------------------------------------------------------------- Ruling 58 items 1 to 3, as adapted
def test_cache_entries_are_written_through_a_temp_file_in_the_same_directory(tmp_path, monkeypatch):
    from es.sealed_pull import caiso_exact
    fetch, seen = _memoised_oasis()
    real, renames = os.replace, []

    def spy(src, dst):
        src, dst = Path(src), Path(dst)
        renames.append({"same_dir": src.parent == dst.parent, "other_name": src.name != dst.name,
                        "suffix": dst.suffix, "final_absent": not dst.exists(), "temp_bytes": src.stat().st_size})
        return real(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    cache = tmp_path / "raw"
    caiso_exact("2026-06-01", "2026-06-01", tmp_path / "root", fetch=fetch, cache_dir=cache, **NO_SLEEP)
    assert len(renames) == len(seen) == 12
    assert all(r["same_dir"] and r["other_name"] and r["suffix"] == ".bin" and r["final_absent"] and r["temp_bytes"] > 0
               for r in renames)
    names = {hashlib.sha256(u.encode()).hexdigest() + ".bin": u for u in seen}
    assert sorted(p.name for p in cache.iterdir()) == sorted(names)            # no temp file is left behind
    assert all((cache / n).read_bytes() == _good_oasis(u) for n, u in names.items())


def test_a_crash_before_the_rename_leaves_no_cache_entry_under_its_final_name(tmp_path, monkeypatch):
    from es.sealed_pull import caiso_exact
    fetch, calls = _memoised_oasis()
    real = os.replace

    def killed(src, dst):
        raise KeyboardInterrupt("killed after the temp file was written and before it was renamed")

    cache = tmp_path / "raw"
    monkeypatch.setattr(os, "replace", killed)
    with pytest.raises(KeyboardInterrupt):
        caiso_exact("2026-06-01", "2026-06-01", tmp_path / "a", fetch=fetch, cache_dir=cache, **NO_SLEEP)
    assert list(cache.glob("*.bin")) == []                      # nothing half-written under a final name
    monkeypatch.setattr(os, "replace", real)
    calls.clear()
    caiso_exact("2026-06-01", "2026-06-01", tmp_path / "b", fetch=fetch, cache_dir=cache, **NO_SLEEP)
    assert len(calls) == 12 and len(list(cache.glob("*.bin"))) == 12
    assert list(cache.glob("*.tmp")) == []                      # the leftover temp file was overwritten and renamed


@pytest.mark.parametrize("damage", ["truncated", "empty", "not_a_zip", "no_rows"])
def test_an_unusable_cache_entry_is_a_miss_that_is_refetched_and_overwritten(tmp_path, damage, capsys):
    from es.sealed_pull import caiso_exact
    fetch, calls = _memoised_oasis()
    cache = tmp_path / "raw"
    first = pd.read_parquet(caiso_exact("2026-06-01", "2026-06-01", tmp_path / "a", fetch=fetch, cache_dir=cache,
                                        **NO_SLEEP))
    victim = sorted(cache.glob("*.bin"))[0]
    good = victim.read_bytes()
    victim.write_bytes({"truncated": good[:50], "empty": b"", "not_a_zip": b"<html>429</html>",
                        "no_rows": _oasis_zip("X", [])}[damage])
    n = len(calls)
    capsys.readouterr()
    again = pd.read_parquet(caiso_exact("2026-06-01", "2026-06-01", tmp_path / "b", fetch=fetch, cache_dir=cache,
                                        **NO_SLEEP))
    notice = capsys.readouterr().out
    assert victim.name in notice and "fetching it again" in notice          # the step log will say what happened
    assert len(calls) == n + 1                                   # only the damaged entry was requested again
    assert again.equals(first) and victim.read_bytes() == good   # and the fresh response replaced it


# ----------------------------------------------------------------------------- the manifest
def _small_record(tmp_path):
    from es.sealed_pull import write_manifest
    root = tmp_path / "record"
    (root / "sub").mkdir(parents=True)
    (root / "a.parquet").write_bytes(b"alpha")
    (root / "sub" / "b.log").write_bytes(b"beta")
    write_manifest(root)
    return root


def test_verify_manifest_accepts_the_record_and_ignores_the_transient_files(tmp_path):
    from es.sealed_pull import verify_manifest
    root = _small_record(tmp_path)
    verify_manifest(root)
    for name in (".state.json", ".state.json.tmp", ".lock", "MANIFEST.sha256.tmp"):      # what a crash can leave behind
        (root / name).write_text("{}")
    verify_manifest(root)


@pytest.mark.parametrize("how, kind, name", [("edit", "changed", "a.parquet"),
                                             ("add", "not in the manifest", "sub/extra.bin"),
                                             ("delete", "missing", "sub/b.log")])
def test_verify_manifest_names_the_file_that_was_edited_added_or_deleted(tmp_path, how, kind, name):
    from es.sealed_pull import verify_manifest
    root = _small_record(tmp_path)
    if how == "edit":
        (root / "a.parquet").write_bytes(b"ALPHA")
    elif how == "add":
        (root / "sub" / "extra.bin").write_bytes(b"x")
    else:
        (root / "sub" / "b.log").unlink()
    with pytest.raises(RuntimeError, match=kind) as err:
        verify_manifest(root)
    assert name in str(err.value)


def test_verify_manifest_names_every_offending_file_at_once(tmp_path):
    from es.sealed_pull import verify_manifest
    root = _small_record(tmp_path)
    (root / "a.parquet").write_bytes(b"ALPHA")
    (root / "sub" / "b.log").unlink()
    (root / "c.parquet").write_bytes(b"gamma")
    with pytest.raises(RuntimeError) as err:
        verify_manifest(root)
    assert all(name in str(err.value) for name in ("a.parquet", "sub/b.log", "c.parquet"))


def test_verify_manifest_sees_a_file_whose_manifest_line_was_removed(tmp_path):
    from es.sealed_pull import verify_manifest
    root = _small_record(tmp_path)
    lines = (root / "MANIFEST.sha256").read_text().splitlines()
    (root / "MANIFEST.sha256").write_text("\n".join(line for line in lines if not line.endswith("a.parquet")) + "\n")
    (root / "a.parquet").write_bytes(b"ALPHA")
    with pytest.raises(RuntimeError, match="not in the manifest") as err:
        verify_manifest(root)
    assert "a.parquet" in str(err.value)


def test_a_missing_manifest_is_a_runtime_error(tmp_path):
    from es.sealed_pull import manifest_sha256, verify_manifest
    for call in (verify_manifest, manifest_sha256):
        with pytest.raises(RuntimeError, match="MANIFEST.sha256"):
            call(tmp_path)


def test_manifest_sha256_is_the_hash_of_the_manifest_file(tmp_path):
    from es.sealed_pull import manifest_sha256, write_manifest
    root = _small_record(tmp_path)
    first = manifest_sha256(root)
    assert first == hashlib.sha256((root / "MANIFEST.sha256").read_bytes()).hexdigest()
    (root / "a.parquet").write_bytes(b"changed")
    write_manifest(root)
    assert manifest_sha256(root) != first                         # the hash moves with the record


def test_the_manifest_leaves_out_the_transient_files_and_itself(tmp_path):
    from es.sealed_pull import write_manifest
    root = _small_record(tmp_path)
    for name in (".state.json", ".state.json.tmp", ".lock", "MANIFEST.sha256.tmp"):
        (root / name).write_text("{}")
    write_manifest(root)
    listed = [line.split("  ", 1)[1] for line in (root / "MANIFEST.sha256").read_text().splitlines()]
    assert listed == ["a.parquet", "sub/b.log"]


def test_a_malformed_or_repeated_manifest_line_is_refused(tmp_path):
    from es.sealed_pull import verify_manifest
    root = _small_record(tmp_path)
    good = (root / "MANIFEST.sha256").read_text()
    for text, why in ((good + "not a manifest line\n", "malformed"), (good + good.splitlines()[0] + "\n", "twice")):
        (root / "MANIFEST.sha256").write_text(text)
        with pytest.raises(RuntimeError, match=why):
            verify_manifest(root)


# ----------------------------------------------------------------------------- the crash windows of a finished pull
def test_a_crash_between_the_manifest_and_the_state_file_stops_as_completed(registered, monkeypatch):
    from es.sealed_pull import verify_manifest
    write, tmp = registered
    proto, reg, sha = write()
    real_unlink = pathlib.Path.unlink

    def crash(self, *args, **kwargs):
        if self.name == ".state.json":
            raise KeyboardInterrupt("killed after the manifest was written")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "unlink", crash)
    with pytest.raises(KeyboardInterrupt):
        _a_pull(tmp, sha)
    monkeypatch.setattr(pathlib.Path, "unlink", real_unlink)
    root = tmp / "sealed_2026Q3"
    assert (root / "MANIFEST.sha256").exists() and (root / ".state.json").exists()
    verify_manifest(root)                                        # the record is whole; the state file is not part of it
    for resume in (False, True):
        with pytest.raises(FileExistsError, match="PULLS.json records a completed pull"):
            _a_pull(tmp, sha, resume=resume)


def test_a_crash_while_the_manifest_is_written_leaves_no_manifest_and_resume_completes(registered, monkeypatch):
    from es.sealed_pull import verify_manifest
    write, tmp = registered
    proto, reg, sha = write()
    real = os.replace

    def killed(src, dst):
        if Path(dst).name == "MANIFEST.sha256":
            raise KeyboardInterrupt("killed while the manifest was being written")
        return real(src, dst)

    monkeypatch.setattr(os, "replace", killed)
    with pytest.raises(KeyboardInterrupt):
        _a_pull(tmp, sha)
    monkeypatch.setattr(os, "replace", real)
    root = tmp / "sealed_2026Q3"
    assert not (root / "MANIFEST.sha256").exists() and (root / ".state.json").exists()     # never a truncated manifest
    root, _ = _a_pull(tmp, sha, resume=True)
    verify_manifest(root)
    assert not (root / ".state.json").exists() and list(root.rglob("*.tmp")) == []


def test_a_crash_while_the_state_is_written_leaves_the_earlier_state_readable(registered, monkeypatch):
    from es.sealed_pull import verify_manifest
    write, tmp = registered
    proto, reg, sha = write()
    real, state_writes = os.replace, []

    def killed(src, dst):
        if Path(dst).name == ".state.json":
            state_writes.append(src)
            if len(state_writes) == 5:  # the first write opens the state; the fifth is the fourth step's (CAISO)
                raise KeyboardInterrupt("killed while the state of the fourth step was being written")
        return real(src, dst)

    monkeypatch.setattr(os, "replace", killed)
    with pytest.raises(KeyboardInterrupt):
        _a_pull(tmp, sha)
    monkeypatch.setattr(os, "replace", real)
    root = tmp / "sealed_2026Q3"
    assert json.loads((root / ".state.json").read_text())["done"] == ["NYISO", "ISONE", "MISO"]      # never truncated
    root, _ = _a_pull(tmp, sha, resume=True)                                                         # CAISO runs again
    verify_manifest(root)


# ----------------------------------------------------------------------------- coverage over the node map
def test_the_node_universe_is_the_33_nodes_restrict_to_node_map_keeps():
    from v2.run_decomposition import restrict_to_node_map

    from es.sealed_pull import _node_map_nodes
    nodes = _node_map_nodes()
    assert len(nodes) == len(set(nodes)) == 33
    assert collections.Counter(m for m, _ in nodes) == {"NYISO": 10, "ISONE": 9, "MISO": 8, "CAISO": 6}
    probe = pd.DataFrame({"market": [m for m, _ in nodes] + ["NYISO", "ISONE"],
                          "node": [n for _, n in nodes] + ["NOT_A_ZONE", "NOT_A_NODE"], "node_type": "x"})
    assert sorted(zip(*[restrict_to_node_map(probe)[c] for c in ("market", "node")])) == sorted(nodes)


def _write_node_map_prices(directory, absent=(), extra=()):
    stamps = pd.to_datetime(["2026-06-10 18:00", "2026-06-10 19:00", "2026-06-10 20:00"], utc=True)
    from es.sealed_pull import _node_map_nodes
    for market in MARKETS:
        nodes = [n for m, n in _node_map_nodes() if m == market and (m, n) not in absent]
        nodes += [n for m, n in extra if m == market]
        pd.DataFrame([(market, n, "hub", t, 1.0, 1.0) for n in nodes for t in stamps],
                     columns=PRICE_COLUMNS).to_parquet(directory / f"{market.lower()}_da_rt_t.parquet", index=False)


def test_coverage_lists_every_node_map_node_and_gives_an_absent_one_zero_hours(tmp_path):
    from es.sealed_pull import _node_map_nodes, coverage_manifest
    absent = ("CAISO", "DLAP_SDGE-APND")
    _write_node_map_prices(tmp_path, absent=[absent], extra=[("NYISO", "OTHER_ZONE")])
    out = coverage_manifest(tmp_path, "2026-06-10", "2026-06-12")
    assert list(out.columns) == ["series", "market", "node", "hours", "expected_hours", "nonfinite_rows", "share"]
    price = out[out["series"] == "price"]
    assert sorted(zip(price["market"], price["node"])) == sorted(_node_map_nodes())  # SPP and OTHER_ZONE are not in it
    gone = price[(price["market"] == "CAISO") & (price["node"] == "DLAP_SDGE-APND")].iloc[0]
    assert gone["hours"] == 0 and gone["share"] == 0.0 and gone["expected_hours"] == 72
    rest = price.drop(gone.name)
    assert rest["hours"].eq(3).all() and (rest["share"] == 3 / 72).all()
    assert out.attrs["not_in_node_map"] == [("NYISO", "OTHER_ZONE")]                      # reported, not listed


def test_coverage_of_a_pull_without_price_files_is_all_zeros_for_every_series(tmp_path):
    from es.sealed_pull import coverage_manifest
    out = coverage_manifest(tmp_path, "2026-06-10", "2026-06-12")
    assert out.groupby("series").size().to_dict() == {"fmm": 6, "nyiso_rtint": 10, "price": 33}
    assert out["hours"].eq(0).all() and out["share"].eq(0.0).all()


def test_coverage_csv_of_a_pull_lists_the_node_map_and_the_log_names_data_only_nodes(registered):
    from es.sealed_pull import pull_sealed
    write, tmp = registered
    proto, reg, sha = write()
    calls = []
    builders = {m: _node_map_builder(m, calls, skip={("CAISO", "DLAP_SDGE-APND")},
                                     extra=["OTHER_ZONE"] if m == "NYISO" else ()) for m in MARKETS}
    builders.update({"FMM": _side_builder("fmm", calls), "NYISO_INT": _side_builder("nyiso_rtint", calls)})
    root, _ = pull_sealed("A", sha, data_root=tmp, builders=builders, reference_dirs=[], check_days=False)
    cov = pd.read_csv(root / "COVERAGE.csv")
    price = cov[cov["series"] == "price"]
    assert len(price) == 33
    gone = price[(price["market"] == "CAISO") & (price["node"] == "DLAP_SDGE-APND")].iloc[0]
    assert gone["hours"] == 0 and gone["share"] == 0.0
    assert price[price["hours"] > 0]["hours"].eq(2).all() and (price["hours"] > 0).sum() == 32
    assert "OTHER_ZONE" in (root / "PULL_LOG.txt").read_text()


# ----------------------------------------------------------------------------- builder logs
def _noisy(builder, step):
    def build(start, end, root):
        print(f"{step} said hello on stdout")
        print(f"{step} said hello on stderr", file=sys.stderr)
        return builder(start, end, root)
    return build


def test_each_builders_output_is_captured_in_a_log_that_is_part_of_the_record(registered, capsys):
    from es.sealed_pull import pull_sealed, verify_manifest
    write, tmp = registered
    proto, reg, sha = write()
    builders = {step: _noisy(b, step) for step, b in _complete_builders([]).items()}
    root, _ = pull_sealed("A", sha, data_root=tmp, builders=builders, reference_dirs=[], check_days=False)
    for step in STEPS:
        log = (root / "logs" / f"{step}.log").read_text()
        assert f"{step} said hello on stdout" in log and f"{step} said hello on stderr" in log
    shown = capsys.readouterr()
    assert "said hello" not in shown.out + shown.err                      # captured, not echoed
    manifest = (root / "MANIFEST.sha256").read_text()
    assert all(f"  logs/{step}.log" in manifest for step in STEPS)
    verify_manifest(root)
    (root / "logs" / "MISO.log").write_text("edited")
    with pytest.raises(RuntimeError, match="changed"):
        verify_manifest(root)


def test_a_failed_step_leaves_its_traceback_in_the_log_and_a_resume_appends_to_it(registered):
    from es.sealed_pull import pull_sealed
    write, tmp = registered
    proto, reg, sha = write()
    calls, failures = [], {"left": 1}
    builders = _complete_builders(calls)
    good_caiso = builders["CAISO"]

    def flaky_caiso(start, end, root):
        if failures["left"]:
            failures["left"] -= 1
            raise OSError("nodename nor servname provided")
        return good_caiso(start, end, root)

    builders["CAISO"] = flaky_caiso
    kw = dict(data_root=tmp, builders=builders, reference_dirs=[], check_days=False)
    with pytest.raises(OSError):
        pull_sealed("A", sha, **kw)
    log = (tmp / "sealed_2026Q3" / "logs" / "CAISO.log").read_text()
    assert "OSError" in log and "nodename nor servname provided" in log and log.count("attempt started") == 1
    root, _ = pull_sealed("A", sha, resume=True, **kw)
    log = (root / "logs" / "CAISO.log").read_text()
    assert log.count("attempt started") == 2 and "nodename nor servname provided" in log     # the first attempt is kept


# ----------------------------------------------------------------------------- the puller guard
@pytest.mark.parametrize("puller, args", [("caiso_exact", ("2026-06-30", "2026-07-01")),
                                          ("caiso_fmm_exact", ("2026-06-30", "2026-07-01")),
                                          ("nyiso_integrated_months", ("2026-06", "2026-07"))])
def test_a_direct_call_with_a_sealed_end_raises_before_any_fetch(tmp_path, puller, args):
    import es.sealed_pull as sp
    from es.guard import SealedDataError

    def must_not_fetch(url):
        raise AssertionError(f"the fetch function was reached: {url}")

    with pytest.raises(SealedDataError):
        getattr(sp, puller)(*args, tmp_path / "root", fetch=must_not_fetch, cache_dir=tmp_path / "raw", **NO_SLEEP)
    assert not (tmp_path / "root").exists() and not (tmp_path / "raw").exists()


def test_the_last_development_day_is_not_refused(tmp_path):
    from es.sealed_pull import caiso_exact, caiso_fmm_exact, nyiso_integrated_months
    caiso_exact("2026-06-30", "2026-06-30", tmp_path / "a", fetch=_good_oasis, **NO_SLEEP)
    caiso_fmm_exact("2026-06-30", "2026-06-30", tmp_path / "b", fetch=_good_oasis, **NO_SLEEP)
    nyiso_integrated_months("2026-06", "2026-06", tmp_path / "c",
                            fetch=lambda url: _nyiso_zip({"20260630": [("06/30/2026 00:00", "N.Y.C.", 1.0)]}),
                            **NO_SLEEP)


@pytest.mark.parametrize("puller, args", [("caiso_exact", ("2026-07-01", "2026-07-01")),
                                          ("caiso_fmm_exact", ("2026-07-01", "2026-07-01")),
                                          ("nyiso_integrated_months", ("2026-07", "2026-07"))])
def test_the_private_sealed_flag_opens_the_guard(tmp_path, puller, args):
    import es.sealed_pull as sp
    fetch = _good_oasis if puller != "nyiso_integrated_months" else (
        lambda url: _nyiso_zip({"20260701": [("07/01/2026 00:00", "N.Y.C.", 1.0)]}))
    assert getattr(sp, puller)(*args, tmp_path / "root", fetch=fetch, _sealed=True, **NO_SLEEP).exists()


def test_only_pull_sealed_and_its_default_builders_pass_the_sealed_flag():
    """A static scan of es/*.py. The pullers get _sealed=True from the three real downloaders (_download_caiso,
    _download_fmm, _download_nyiso_integrated) alone; the readers get it from pull_sealed alone. A recorder test
    cannot show 'only'; the source can."""
    hits = set()

    class Finder(ast.NodeVisitor):
        def __init__(self):
            self.stack = []

        def visit_FunctionDef(self, node):
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_Call(self, node):
            for kw in node.keywords:
                if kw.arg == "_sealed":
                    hits.add((self.path, self.stack[-1] if self.stack else "<module>", ast.unparse(node.func)))
            self.generic_visit(node)

    for path in sorted((DART_DEV / "es").glob("*.py")):
        finder = Finder()
        finder.path = path.name
        finder.visit(ast.parse(path.read_text()))
    assert hits == {("sealed_pull.py", "_download_caiso", "caiso_exact"),
                    ("sealed_pull.py", "_download_fmm", "caiso_fmm_exact"),
                    ("sealed_pull.py", "_download_nyiso_integrated", "nyiso_integrated_months"),
                    ("sealed_pull.py", "pull_sealed", "coverage_manifest"),
                    ("sealed_pull.py", "pull_sealed", "overlap_check")}


def test_the_default_builders_pass_the_sealed_flag(monkeypatch, tmp_path):
    import es.sealed_pull as sp
    seen = {}
    for name in ("caiso_exact", "caiso_fmm_exact", "nyiso_integrated_months"):
        monkeypatch.setattr(sp, name, lambda *a, _name=name, **k: seen.setdefault(_name, k))
    builders = sp._default_builders()
    for step in ("CAISO", "FMM", "NYISO_INT"):
        builders[step]("2026-06-21", "2026-06-23", tmp_path)
    assert {name: kw.get("_sealed") for name, kw in seen.items()} == {
        "caiso_exact": True, "caiso_fmm_exact": True, "nyiso_integrated_months": True}


def test_the_default_builders_open_the_guard_for_test_a_and_request_its_exact_window(registered, monkeypatch):
    from es.sealed_pull import pull_sealed, verify_manifest
    write, tmp = registered
    proto, reg, sha = write()
    urls = []
    _use_default_builders(monkeypatch, urls, window=None)                      # the real window of Test A, fakes only
    root, _ = pull_sealed("A", sha, data_root=tmp, reference_dirs=[])
    oasis = [u for u in urls if "oasis" in u]
    assert len(oasis) == 3 * 6 * 4                                              # three queries, six nodes, four chunks
    assert sorted({(_ts(u, "startdatetime"), _ts(u, "enddatetime")) for u in oasis}) == [
        (T("2026-06-21 07:00"), T("2026-07-21 07:00")), (T("2026-07-21 07:00"), T("2026-08-20 07:00")),
        (T("2026-08-20 07:00"), T("2026-09-19 07:00")), (T("2026-09-19 07:00"), T("2026-10-01 07:00"))]
    assert [u.rsplit("/", 1)[1][:6] for u in urls if "nyiso" in u] == ["202606", "202607", "202608", "202609"]
    assert root == tmp / "sealed_2026Q3"
    verify_manifest(root)


def test_the_pullers_request_the_exact_window_of_test_b(tmp_path):
    from es.sealed_pull import caiso_exact
    seen = []

    def fetch(url):
        seen.append(url)
        return _good_oasis(url)

    caiso_exact("2026-09-21", "2026-12-31", tmp_path / "root", fetch=fetch, _sealed=True, **NO_SLEEP)
    chunks = sorted({(_ts(u, "startdatetime"), _ts(u, "enddatetime")) for u in seen})
    assert chunks[0][0] == T("2026-09-21 07:00") and chunks[-1][1] == T("2027-01-01 08:00")     # PDT start, PST end
    assert all(a[1] == b[0] for a, b in zip(chunks, chunks[1:])) and len(chunks) == 4


# -------------------------------------------------------------------- Ruling 60 A: reference reads stay in their window
@pytest.mark.parametrize("market, utc_offset_hours, naive", [("NYISO", 4, False), ("ISONE", 4, False),
                                                              ("MISO", 5, True), ("CAISO", 7, False)])
def test_overlap_check_never_loads_a_row_at_or_after_the_local_midnight_of_end_exclusive(
        tmp_path, monkeypatch, market, utc_offset_hours, naive):
    """The H1 CAISO file holds six rows of 1 July 00:00 PDT. A read spy sees the tables pyarrow hands back."""
    import pyarrow.parquet as pq

    from es.sealed_pull import overlap_check
    midnight = lambda day: T(day) + pd.Timedelta(hours=utc_offset_hours)                      # local 00:00 of `day`
    stamps = [midnight("2026-06-15") + pd.Timedelta(hours=10), midnight("2026-06-16") - pd.Timedelta(hours=1),
              midnight("2026-06-16"), midnight("2026-06-17") + pd.Timedelta(hours=3)]     # two inside | the cut | later
    if naive:
        stamps = [s.tz_localize(None) for s in stamps]
    frame = pd.DataFrame({"market": market, "node": "X", "node_type": "hub", "timestamp_utc": stamps,
                          "da_lmp": [1.0, 2.0, 3.0, 4.0], "rt_lmp": [1.0, 2.0, 3.0, 4.0]})
    for sub in ("ref", "new"):
        (tmp_path / sub).mkdir()
        frame.to_parquet(tmp_path / sub / f"{market.lower()}_da_rt_{sub}.parquet", index=False)
    real, read = pq.read_table, []

    def spy(*args, **kwargs):
        table = real(*args, **kwargs)
        read.append(table)
        return table

    monkeypatch.setattr(pq, "read_table", spy)
    out = overlap_check(tmp_path / "new", [tmp_path / "ref"], "2026-06-10", "2026-06-16").set_index("market")
    assert len(read) == 2                                                      # one read for each side
    for table in read:
        loaded = pd.to_datetime(table.column("timestamp_utc").to_pandas(), utc=True)
        assert len(loaded) == 2 and loaded.max() < midnight("2026-06-16")      # the cut row and later ones never load
    assert out.loc[market, "matched"] == 2 and out.loc[market, "new_only"] == 0
    assert out.loc[market, "reference_only"] == 0 and out.loc[market, "rows_differing"] == 0


# ----------------------------------------------------------------------------- Ruling 60 B: no silently lost days
def test_missing_days_reports_each_node_and_day_without_rows(tmp_path):
    from es.sealed_pull import _missing_days
    isone = _nodes_by_market()["ISONE"]
    gone = {(".Z.MAINE", "2026-06-22"), (".Z.VERMONT", "2026-06-21")}
    rows = _full_price_rows("ISONE", isone, "2026-06-21", "2026-06-23", skip=gone)
    rows += _full_price_rows("ISONE", ["NOT_IN_THE_MAP"], "2026-06-21", "2026-06-23")               # ignored
    rows += _full_price_rows("ISONE", [".Z.MAINE"], "2026-06-20", "2026-06-20")                     # outside the window
    pd.DataFrame(rows, columns=PRICE_COLUMNS).to_parquet(tmp_path / "isone_da_rt_t.parquet", index=False)
    assert _missing_days(tmp_path, "ISONE", "2026-06-21", "2026-06-23") == [
        (".Z.MAINE", dt.date(2026, 6, 22)), (".Z.VERMONT", dt.date(2026, 6, 21))]


def test_missing_days_counts_a_row_on_the_day_of_its_local_clock(tmp_path):
    from es.sealed_pull import _missing_days
    caiso = _nodes_by_market()["CAISO"]
    rows = _full_price_rows("CAISO", caiso, "2026-06-21", "2026-06-21")       # 07:00Z on the 21st to 06:00Z on the 22nd
    pd.DataFrame(rows, columns=PRICE_COLUMNS).to_parquet(tmp_path / "caiso_da_rt_t.parquet", index=False)
    assert _missing_days(tmp_path, "CAISO", "2026-06-21", "2026-06-22") == [(n, dt.date(2026, 6, 22))
                                                                            for n in sorted(caiso)]
    # the last six hours of the 21st fall on the 22nd by the UTC calendar; by the local clock they are still the 21st


def test_missing_days_of_a_market_without_files_is_every_node_and_day(tmp_path):
    from es.sealed_pull import _missing_days
    assert len(_missing_days(tmp_path, "MISO", "2026-06-21", "2026-06-23")) == 8 * 3


def test_gaps_are_written_as_node_by_day_ranges():
    from es.sealed_pull import _fmt_gaps
    d = lambda day: dt.date(2026, 6, day)
    gaps = [("A", d(22)), ("A", d(23)), ("A", d(24)), ("A", d(30)), ("B", d(22))]
    assert _fmt_gaps(gaps) == "A: 2026-06-22 .. 2026-06-24, 2026-06-30; B: 2026-06-22"


def test_a_day_the_july_downloader_skipped_stops_the_step_before_it_is_recorded(registered, monkeypatch):
    from es.sealed_pull import MissingDaysError, pull_sealed, verify_manifest
    write, tmp = registered
    proto, reg, sha = write()
    urls, ran = [], []
    skip = {("ISONE", ".Z.MAINE", "2026-06-22")}
    _use_default_builders(monkeypatch, urls, skip=skip, builder_calls=ran)
    kw = dict(data_root=tmp, reference_dirs=[])
    with pytest.raises(MissingDaysError, match=r"ISONE.*\.Z\.MAINE.*2026-06-22"):
        pull_sealed("A", sha, **kw)
    root = tmp / "pull_unit"
    assert json.loads((root / ".state.json").read_text())["done"] == ["NYISO"]        # ISONE is not recorded as done
    assert not (root / "MANIFEST.sha256").exists()
    log = (root / "PULL_LOG.txt").read_text()
    assert ".Z.MAINE" in log and "2026-06-22" in log                                   # the gap is in the record
    skip.clear()                                                                      # the ISO serves the day next time
    ran.clear()
    root, _ = pull_sealed("A", sha, resume=True, **kw)
    assert ran == ["ISONE", "MISO"]                          # the failed step is re-run; the finished one is not
    verify_manifest(root)


def test_a_node_that_never_arrives_is_reported_for_every_day(registered, monkeypatch):
    from es.sealed_pull import MissingDaysError, pull_sealed
    write, tmp = registered
    proto, reg, sha = write()
    skip = {("MISO", "TEXAS.HUB", d) for d in ("2026-06-21", "2026-06-22", "2026-06-23")}
    _use_default_builders(monkeypatch, [], skip=skip)
    with pytest.raises(MissingDaysError, match=r"MISO.*TEXAS\.HUB: 2026-06-21 \.\. 2026-06-23"):
        pull_sealed("A", sha, data_root=tmp, reference_dirs=[])


def test_the_error_says_to_excuse_a_day_only_if_the_iso_never_published_it(registered, monkeypatch):
    from es.sealed_pull import MissingDaysError, pull_sealed
    write, tmp = registered
    proto, reg, sha = write()
    _use_default_builders(monkeypatch, [], skip={("ISONE", ".Z.MAINE", "2026-06-22")})
    with pytest.raises(MissingDaysError) as err:
        pull_sealed("A", sha, data_root=tmp, reference_dirs=[])
    assert "only if the ISO never published" in str(err.value) and "resume=True" in str(err.value)


def test_allow_missing_days_excuses_listed_days_only_and_logs_them(registered, monkeypatch):
    from es.sealed_pull import MissingDaysError, pull_sealed, verify_manifest
    write, tmp = registered
    proto, reg, sha = write()
    skip = {("MISO", n, "2026-06-22") for n in _nodes_by_market()["MISO"]}             # MISO never published the 22nd
    skip.add(("MISO", "ILLINOIS.HUB", "2026-06-23"))                                   # and one gap nobody excused
    _use_default_builders(monkeypatch, [], skip=skip)
    kw = dict(data_root=tmp, reference_dirs=[], allow_missing_days={"MISO": ["2026-06-22"]})
    with pytest.raises(MissingDaysError, match=r"ILLINOIS\.HUB: 2026-06-23") as err:
        pull_sealed("A", sha, **kw)
    assert "2026-06-22" not in str(err.value)                                          # the excused day is not an error
    skip.discard(("MISO", "ILLINOIS.HUB", "2026-06-23"))
    root, _ = pull_sealed("A", sha, resume=True, **kw)
    log = (root / "PULL_LOG.txt").read_text()
    assert log.count('allow_missing_days: {"MISO": ["2026-06-22"]}') == 2           # every attempt records the override
    assert "allowed by allow_missing_days" in log and "2026-06-22" in log              # and the gaps it excused
    cov = pd.read_csv(root / "COVERAGE.csv")
    assert cov[(cov["market"] == "MISO") & (cov["node"] == "TEXAS.HUB")]["hours"].iloc[0] == 24     # the 23rd only
    verify_manifest(root)


def test_allow_missing_days_may_hold_sets_or_tuples_and_is_logged_sorted(registered, monkeypatch):
    from es.sealed_pull import pull_sealed
    write, tmp = registered
    proto, reg, sha = write()
    skip = {("MISO", n, d) for n in _nodes_by_market()["MISO"] for d in ("2026-06-22", "2026-06-23")}
    _use_default_builders(monkeypatch, [], skip=skip)
    root, _ = pull_sealed("A", sha, data_root=tmp, reference_dirs=[],
                          allow_missing_days={"MISO": {"2026-06-23", "2026-06-22"}})  # a set: not JSON, still fine
    assert 'allow_missing_days: {"MISO": ["2026-06-22", "2026-06-23"]}' in (root / "PULL_LOG.txt").read_text()


@pytest.mark.parametrize("bad, why", [({"SPP": ["2026-06-22"]}, "unknown market or series"),
                                       ({"MISO": ["2026-6-22"]}, "not a 'YYYY-MM-DD' day"),
                                       ({"MISO": ["2026-06-19"]}, "outside the pull"),
                                       ({"MISO": "2026-06-22"}, "needs a list"),
                                       ({"MISO": ["2026-06-22", 5]}, "not a 'YYYY-MM-DD' day")])
def test_allow_missing_days_refuses_what_cannot_be_meant(registered, monkeypatch, bad, why):
    from es.sealed_pull import pull_sealed
    write, tmp = registered
    proto, reg, sha = write()
    _use_default_builders(monkeypatch, [])
    with pytest.raises(ValueError, match="allow_missing_days") as err:
        pull_sealed("A", sha, data_root=tmp, reference_dirs=[], allow_missing_days=bad)
    assert why in str(err.value)
    assert not (tmp / "pull_unit").exists()                                            # refused before anything is made


# ----------------------------------------------------------------------- Ruling 60 C: side files are trimmed and logged
def test_side_tables_are_trimmed_by_settlement_day_and_the_counts_are_logged(registered, monkeypatch):
    import es.sealed_pull as sp
    write, tmp = registered
    proto, reg, sha = write()
    monkeypatch.setitem(sp.SEALED_WINDOWS, "A", ("2026-06-10", "2026-06-12", "2026-06-15", "pull_unit"))
    calls = []
    builders = {m: _node_map_builder(m, calls, stamps=("2026-06-11 18:00", "2026-06-13 18:00")) for m in MARKETS}

    def fmm(start, end, root):                       # CAISO time, UTC-7: edges of the pull at 07:00 UTC
        (root / "fmm").mkdir()
        stamps = [T("2026-06-10 06:00"), T("2026-06-10 07:00"), T("2026-06-16 06:00"), T("2026-06-16 07:00")]
        pd.DataFrame({"node": "TH_SP15_GEN-APND", "timestamp_utc": stamps, "fmm_lmp": 1.0, "n15": 4}).to_parquet(
            root / "fmm" / "caiso_fmm_x.parquet", index=False)

    def nyiso_int(start, end, root):                 # New York time, UTC-4: edges of the pull at 04:00 UTC
        (root / "nyiso_rtint").mkdir()
        stamps = [T("2026-06-10 03:00"), T("2026-06-10 04:00"), T("2026-06-16 03:00"), T("2026-06-16 04:00")]
        pd.DataFrame({"node": "N.Y.C.", "timestamp_utc": stamps, "rt_int_lmp": 1.0}).to_parquet(
            root / "nyiso_rtint" / "nyiso_rtint_x.parquet", index=False)

    builders.update({"FMM": fmm, "NYISO_INT": nyiso_int})
    root, _ = sp.pull_sealed("A", sha, data_root=tmp, builders=builders, reference_dirs=[], check_days=False)
    assert pd.read_parquet(root / "fmm" / "caiso_fmm_x.parquet")["timestamp_utc"].tolist() == [
        T("2026-06-10 07:00"), T("2026-06-16 06:00")]
    assert pd.read_parquet(root / "nyiso_rtint" / "nyiso_rtint_x.parquet")["timestamp_utc"].tolist() == [
        T("2026-06-10 04:00"), T("2026-06-16 03:00")]
    log = (root / "PULL_LOG.txt").read_text()
    assert "trimmed 2 row(s) outside 2026-06-10 .. 2026-06-15 from fmm/caiso_fmm_x.parquet" in log
    assert "trimmed 2 row(s) outside 2026-06-10 .. 2026-06-15 from nyiso_rtint/nyiso_rtint_x.parquet" in log


# ----------------------------------------------------------------------------- Ruling 60 D: module-level imports
def test_the_july_downloaders_are_imported_with_the_module():
    """The protocol's code archive follows es.sealed_pull's imports, so the three downloaders must be among them."""
    import v2.ingest_isone
    import v2.ingest_miso
    import v2.ingest_nyiso

    import es.sealed_pull as sp
    assert (sp.ingest_isone, sp.ingest_miso, sp.ingest_nyiso) == (v2.ingest_isone, v2.ingest_miso, v2.ingest_nyiso)
    here = Path(__file__).resolve()
    env = dict(os.environ, DART_PROJECT_ROOT=str(here.parents[3]))
    code = ("import sys; import es.sealed_pull; "
            "print([m for m in ('v2.ingest_isone', 'v2.ingest_miso', 'v2.ingest_nyiso') if m not in sys.modules])")
    out = subprocess.run([sys.executable, "-c", code], cwd=here.parents[2], env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"                                    # none of the three is missing after the import


# ----------------------------------------------------------------------------- Ruling 60 E: the smaller fixes
def test_empty_builders_never_fall_through_to_the_real_endpoints(registered, monkeypatch):
    import es.sealed_pull as sp
    write, tmp = registered
    proto, reg, sha = write()

    def real_endpoints():
        raise AssertionError("the default builders were reached")

    monkeypatch.setattr(sp, "_default_builders", real_endpoints)
    with pytest.raises(ValueError, match="builders"):
        sp.pull_sealed("A", sha, data_root=tmp, builders={}, reference_dirs=[], check_days=False)
    assert not (tmp / "sealed_2026Q3").exists()                            # refused before anything is made


def test_an_assertion_error_from_the_fetch_function_is_not_retried(tmp_path):
    from es.sealed_pull import caiso_exact
    calls, sleeps = [], []

    def blocked(url):
        calls.append(url)
        raise AssertionError("network access attempted in a unit test")

    with pytest.raises(AssertionError, match="network access"):
        caiso_exact("2026-06-01", "2026-06-01", tmp_path / "root", fetch=blocked, pause=0, sleep=sleeps.append)
    assert len(calls) == 1 and sleeps == []


def test_nyiso_integrated_months_logs_the_rows_dropped_for_ambiguous_local_times(tmp_path):
    """The clocks went back on 2 November 2025: both 01:00 rows are ambiguous and are dropped, never guessed."""
    from es.sealed_pull import nyiso_integrated_months
    fall_back_day = [("11/02/2025 00:00", "N.Y.C.", 1.0), ("11/02/2025 01:00", "N.Y.C.", 2.0),
                     ("11/02/2025 01:00", "N.Y.C.", 3.0), ("11/02/2025 02:00", "N.Y.C.", 4.0)]
    months = {"202510": _nyiso_zip({"20251031": [("10/31/2025 10:00", "N.Y.C.", 5.0)]}),
              "202511": _nyiso_zip({"20251102": fall_back_day})}
    lines = []
    out = nyiso_integrated_months("2025-10", "2025-11", tmp_path / "x",
                                  fetch=lambda url: months[url.rsplit("/", 1)[1][:6]], log=lines.append, **NO_SLEEP)
    assert len(pd.read_parquet(out)) == 3
    assert len(lines) == 2
    assert "2025-10" in lines[0] and "1 rows kept, 0 dropped" in lines[0]
    assert "2025-11" in lines[1] and "2 rows kept, 2 dropped" in lines[1]


def test_the_dropped_rows_per_month_reach_the_pull_log(registered, monkeypatch):
    from es.sealed_pull import pull_sealed
    write, tmp = registered
    proto, reg, sha = write()
    _use_default_builders(monkeypatch, [], window=("2025-10-31", "2025-11-01", "2025-11-03"),
                          nyiso_extra=(("11/02/2025 01:00", "N.Y.C.", 2.0), ("11/02/2025 01:00", "N.Y.C.", 3.0)))
    root, _ = pull_sealed("A", sha, data_root=tmp, reference_dirs=[])
    log = (root / "PULL_LOG.txt").read_text()
    # ten zones, 30 and 31 October: 48 hours each
    assert "NYISO integrated 2025-10: 480 rows kept, 0 dropped" in log
    # 1 to 4 November: 24 + 23 + 24 + 24 hours kept each; on the 2nd both 01:00 rows of each zone are ambiguous, and so
    # are the two extra 01:00 records (10 zones x 2 + 2 dropped)
    assert "NYISO integrated 2025-11: 950 rows kept, 22 dropped" in log
    assert "NYISO integrated 2025-11: 950 rows kept, 22 dropped" in (root / "logs" / "NYISO_INT.log").read_text()


# =====================================================================================================
# Fix round 1 (controller Ruling 70). Each test below fails on the code that preceded the round.
# =====================================================================================================
# ----------------------------------------------------------------------------- I3 and M6: once only, what is recorded
def test_the_pull_returns_its_root_and_manifest_hash_and_records_both_outside_the_data_root(registered):
    from es.sealed_pull import completed_pulls, manifest_sha256
    write, tmp = registered
    proto, reg, sha = write()
    root, manifest = _a_pull(tmp, sha)
    assert re.fullmatch("[0-9a-f]{64}", manifest) and manifest == manifest_sha256(root)
    record = completed_pulls()["A"]
    assert record == _pulls(tmp)["A"]
    assert (record["root"], record["manifest_sha256"]) == (str(root), manifest)
    assert (record["protocol_sha256"], record["osf_doi"], record["registered_utc"]) == (
        sha, "10.17605/OSF.IO/ABCDE", "2026-10-20T15:04:00Z")
    started, finished = (dt.datetime.fromisoformat(record[k].replace("Z", "+00:00"))
                         for k in ("started_utc", "finished_utc"))
    assert started < finished
    logged = (root / "PULL_LOG.txt").read_text().splitlines()
    assert record["started_utc"] == [ln.split(": ", 1)[1] for ln in logged if ln.startswith("pull_started_utc: ")][0]
    assert record["finished_utc"] == [ln.split(": ", 1)[1] for ln in logged if ln.startswith("pull_finished_utc: ")][-1]
    assert (tmp / "docs_sealed" / "PULLS.json").exists() and not list(root.rglob("PULLS.json"))
    assert not (tmp / "docs_sealed" / "PULL_A.lock").exists()


def test_a_completed_pull_blocks_another_pull_of_the_same_test_whatever_the_data_root(registered):
    write, tmp = registered
    proto, reg, sha = write()
    _a_pull(tmp, sha)
    other = tmp / "elsewhere"
    other.mkdir()
    calls = []
    for resume in (False, True):
        with pytest.raises(FileExistsError, match="PULLS.json records a completed pull"):
            _a_pull(other, sha, calls, resume=resume)
    assert calls == [] and not (other / "sealed_2026Q3").exists()


def test_a_corrupt_record_of_completed_pulls_stops_every_pull_before_any_download(registered):
    write, tmp = registered
    proto, reg, sha = write()
    (tmp / "docs_sealed" / "PULLS.json").write_text("{not json")
    calls = []
    with pytest.raises(RuntimeError, match="PULLS.json"):
        _a_pull(tmp, sha, calls)
    assert calls == [] and not (tmp / "sealed_2026Q3").exists()


def test_the_record_is_written_atomically_and_a_resume_adopts_the_finished_pull(registered, monkeypatch):
    from es.sealed_pull import manifest_sha256, verify_manifest
    write, tmp = registered
    proto, reg, sha = write()
    real, calls = os.replace, []

    def killed(src, dst):
        if Path(dst).name == "PULLS.json":
            raise KeyboardInterrupt("killed while PULLS.json was being written")
        return real(src, dst)

    monkeypatch.setattr(os, "replace", killed)
    with pytest.raises(KeyboardInterrupt):
        _a_pull(tmp, sha, calls)
    monkeypatch.setattr(os, "replace", real)
    root = tmp / "sealed_2026Q3"
    assert _pulls(tmp) == {} and (root / "MANIFEST.sha256").exists()          # sealed, but not yet recorded
    calls.clear()
    with pytest.raises(FileExistsError, match="holds a completed pull"):      # a manifest without a record, no resume
        _a_pull(tmp, sha, calls)
    adopted, manifest = _a_pull(tmp, sha, calls, resume=True)
    assert calls == [] and adopted == root                                   # no download: the finished pull is adopted
    assert _pulls(tmp)["A"]["manifest_sha256"] == manifest == manifest_sha256(root)
    logged = (root / "PULL_LOG.txt").read_text().splitlines()
    started = [ln.split(": ", 1)[1] for ln in logged if ln.startswith("pull_started_utc: ")]
    finished = [ln.split(": ", 1)[1] for ln in logged if ln.startswith("pull_finished_utc: ")]
    assert (_pulls(tmp)["A"]["started_utc"], _pulls(tmp)["A"]["finished_utc"]) == (started[0], finished[-1])
    assert not (root / ".state.json").exists()
    verify_manifest(root)
    with pytest.raises(FileExistsError, match="PULLS.json records a completed pull"):
        _a_pull(tmp, sha, calls, resume=True)


def test_a_finished_pull_that_no_longer_verifies_is_not_adopted(registered, monkeypatch):
    write, tmp = registered
    proto, reg, sha = write()
    real = os.replace

    def killed(src, dst):
        if Path(dst).name == "PULLS.json":
            raise KeyboardInterrupt("killed while PULLS.json was being written")
        return real(src, dst)

    monkeypatch.setattr(os, "replace", killed)
    with pytest.raises(KeyboardInterrupt):
        _a_pull(tmp, sha)
    monkeypatch.setattr(os, "replace", real)
    (tmp / "sealed_2026Q3" / "isone_da_rt_test.parquet").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="changed"):
        _a_pull(tmp, sha, resume=True)
    assert _pulls(tmp) == {}


def test_a_manifest_in_a_pull_whose_log_never_finished_is_not_adopted(registered):
    """Only a pull that wrote its finish line is adopted: a hand-made manifest in an unfinished pull proves nothing."""
    from es.sealed_pull import write_manifest
    write, tmp = registered
    proto, reg, sha = write()
    builders = _failing_then_fine_builders()
    with pytest.raises(OSError):
        _a_pull(tmp, sha, builders=builders)
    root = tmp / "sealed_2026Q3"
    write_manifest(root)                                                          # a hand-made manifest of half a pull
    calls = []
    with pytest.raises(RuntimeError, match="does not say the pull finished"):
        _a_pull(tmp, sha, calls, resume=True)
    assert calls == [] and _pulls(tmp) == {}


def test_a_resume_is_refused_once_the_log_says_finished_and_the_manifest_and_state_are_gone(registered):
    """The reviewer's check D: the manifest deleted, then resume=True re-ran all six steps."""
    write, tmp = registered
    proto, reg, sha = write()
    root, _ = _a_pull(tmp, sha)
    (root / "MANIFEST.sha256").unlink()
    (tmp / "docs_sealed" / "PULLS.json").unlink()           # and the record outside the root: only the log can object
    calls = []
    with pytest.raises(FileExistsError, match="holds a finished pull"):
        _a_pull(tmp, sha, calls, resume=True)
    (root / ".state.json").write_text(json.dumps({"done": [], "protocol_sha256": sha,
                                                  "osf_doi": "10.17605/OSF.IO/ABCDE"}))      # a state put back by hand
    with pytest.raises(FileExistsError, match="holds a finished pull"):
        _a_pull(tmp, sha, calls, resume=True)
    (root / ".state.json").write_text("{not json")                                 # or one that cannot be read
    with pytest.raises(FileExistsError, match="holds a finished pull"):
        _a_pull(tmp, sha, calls, resume=True)
    assert calls == [] and not (root / "MANIFEST.sha256").exists()


def test_a_kill_between_the_finish_line_and_the_manifest_is_finalised_without_any_download(registered, monkeypatch):
    import es.sealed_pull as sp
    from es.sealed_pull import manifest_sha256
    write, tmp = registered
    proto, reg, sha = write()
    real, calls = sp.write_manifest, []

    def killed(root):
        raise KeyboardInterrupt("killed before the manifest was written")

    monkeypatch.setattr(sp, "write_manifest", killed)
    with pytest.raises(KeyboardInterrupt):
        _a_pull(tmp, sha, calls)
    monkeypatch.setattr(sp, "write_manifest", real)
    root = tmp / "sealed_2026Q3"
    assert "pull_finished_utc" in (root / "PULL_LOG.txt").read_text()
    assert not (root / "MANIFEST.sha256").exists() and (root / ".state.json").exists()
    calls.clear()
    root, manifest = _a_pull(tmp, sha, calls, resume=True)
    assert calls == []                                                          # all six steps were done: nothing runs
    assert (root / "PULL_LOG.txt").read_text().count("pull_finished_utc") == 2
    assert _pulls(tmp)["A"]["manifest_sha256"] == manifest == manifest_sha256(root)


# ----------------------------------------------------------------------------- I3: the lock
def test_an_exclusive_lock_is_held_for_the_whole_pull_and_released_after_it(registered):
    import es.registration as registration
    write, tmp = registered
    proto, reg, sha = write()
    lock = registration.SEALED_DIR / "PULL_A.lock"
    held = {}
    builders = _complete_builders([])
    for step in ("NYISO", "NYISO_INT"):                                          # the first step and the last
        inner = builders[step]

        def spy(start, end, root, _inner=inner, _step=step):
            held[_step] = lock.exists() and json.loads(lock.read_text())["pid"] == os.getpid()
            return _inner(start, end, root)

        builders[step] = spy
    assert not lock.exists()
    _a_pull(tmp, sha, builders=builders)
    assert held == {"NYISO": True, "NYISO_INT": True} and not lock.exists()


def test_a_second_pull_started_while_the_first_runs_is_refused(registered):
    """The reviewer's check J: a second pull_sealed(resume=True) used to finish into the same root."""
    write, tmp = registered
    proto, reg, sha = write()
    calls, refused = [], {}
    builders = _complete_builders(calls)
    outer = builders["NYISO"]

    def nyiso_then_a_second_process(start, end, root):
        outer(start, end, root)
        try:
            _a_pull(tmp, sha, builders=builders, resume=True)
        except FileExistsError as exc:
            refused["error"] = str(exc)

    builders["NYISO"] = nyiso_then_a_second_process
    root, _ = _a_pull(tmp, sha, builders=builders)
    assert "PULL_A.lock" in refused["error"] and "remove" in refused["error"]
    assert [c[0] for c in calls].count("NYISO") == 1 and len(calls) == 6          # each step ran once
    assert (root / "PULL_LOG.txt").read_text().count("pull_finished_utc") == 1


@pytest.mark.parametrize("failure", [OSError("dns"), KeyboardInterrupt("killed"), RuntimeError("builder bug")])
def test_the_lock_is_released_when_the_pull_fails(registered, failure):
    import es.registration as registration
    write, tmp = registered
    proto, reg, sha = write()
    builders = _complete_builders([])

    def broken(start, end, root):
        raise failure

    builders["MISO"] = broken
    with pytest.raises(type(failure)):
        _a_pull(tmp, sha, builders=builders)
    assert not (registration.SEALED_DIR / "PULL_A.lock").exists()
    builders["MISO"] = _complete_builders([])["MISO"]
    _a_pull(tmp, sha, builders=builders, resume=True)                            # and the pull can go on


def test_a_leftover_lock_blocks_the_pull_and_says_how_to_clear_it(registered):
    import es.registration as registration
    write, tmp = registered
    proto, reg, sha = write()
    lock = registration.SEALED_DIR / "PULL_A.lock"
    lock.write_text(json.dumps({"pid": 4242, "host": "somewhere", "started_utc": "2026-10-23T01:00:00Z"}))
    calls = []
    with pytest.raises(FileExistsError, match="PULL_A.lock") as err:
        _a_pull(tmp, sha, calls)
    assert "4242" in str(err.value) and "remove" in str(err.value)
    assert calls == [] and not (tmp / "sealed_2026Q3").exists() and lock.exists()      # the lock is not ours to remove


# ----------------------------------------------------------------------------- M2: Test B needs Test A
def _b_fakes(rows, calls=None):
    builders = {m: _stamped_builder(m, rows, calls=calls) for m in MARKETS}
    builders.update({"FMM": _side_builder("fmm", calls if calls is not None else []),
                     "NYISO_INT": _side_builder("nyiso_rtint", calls if calls is not None else [])})
    return builders


A_ROWS = ("2026-06-25 18:00", "2026-09-25 18:00", "2026-10-01 18:00")
B_ROWS = ("2026-09-25 18:00", "2026-10-15 18:00", "2027-01-02 18:00")


def test_pulling_test_b_requires_a_verified_pull_of_test_a_and_reads_it_from_the_record(registered):
    from es.sealed_pull import pull_sealed
    write, tmp = registered
    proto, reg, sha = write()
    (tmp / "a_data").mkdir()
    (tmp / "b_data").mkdir()
    b_calls = []
    kw_b = dict(data_root=tmp / "b_data", builders=_b_fakes(B_ROWS, b_calls), check_days=False)
    with pytest.raises(RuntimeError, match="Test A"):
        pull_sealed("B", sha, **kw_b)                                 # Test A has not been pulled
    assert b_calls == [] and not (tmp / "b_data" / "sealed_2026Q4").exists()
    a_root, _ = pull_sealed("A", sha, data_root=tmp / "a_data", builders=_b_fakes(A_ROWS), reference_dirs=[],
                            check_days=False)
    b_root, _ = pull_sealed("B", sha, **kw_b)                         # A is found through the record, not data_root
    overlap = pd.read_csv(b_root / "OVERLAP_CHECK.csv")
    price = overlap[overlap["series"] == "price"].set_index("market")
    assert price["matched"].to_dict() == {m: 1 for m in MARKETS}      # 25 September, read from A's folder
    assert (price["new_only"] == 0).all() and (price["reference_only"] == 0).all()
    assert (a_root / "MANIFEST.sha256").exists() and a_root.parent == tmp / "a_data"
    from es.sealed_pull import completed_pulls
    pulls = completed_pulls()
    assert sorted(pulls) == ["A", "B"] and pulls["A"]["root"] == str(a_root) and pulls["B"]["root"] == str(b_root)


def test_test_b_is_refused_when_test_as_record_or_manifest_no_longer_verifies(registered):
    from es.sealed_pull import pull_sealed, write_manifest
    write, tmp = registered
    proto, reg, sha = write()
    a_root, _ = _a_pull(tmp, sha)
    kw_b = dict(data_root=tmp, builders=_b_fakes(B_ROWS), check_days=False, reference_dirs=[])
    (a_root / "isone_da_rt_test.parquet").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="changed"):
        pull_sealed("B", sha, **kw_b)                                 # a file of Test A was edited
    write_manifest(a_root)                                            # ... and the manifest rewritten to match
    with pytest.raises(RuntimeError, match="recorded"):
        pull_sealed("B", sha, **kw_b)                                 # but PULLS.json holds the hash of the original
    assert not (tmp / "sealed_2026Q4").exists()


# ----------------------------------------------------------------------------- I4: the trim is atomic, part of the step
def test_trimming_goes_through_a_temp_file_and_leaves_an_untouched_file_alone(tmp_path, monkeypatch):
    from es.sealed_pull import _trim
    path = tmp_path / "isone_da_rt_t.parquet"
    stamps = pd.to_datetime(["2026-06-25 18:00", "2026-10-01 18:00"], utc=True)
    pd.DataFrame({"market": "ISONE", "node": "X", "node_type": "hub", "timestamp_utc": stamps,
                  "da_lmp": 1.0, "rt_lmp": 1.0}).to_parquet(path, index=False)
    real, renames = os.replace, []

    def spy(src, dst):
        renames.append((Path(src).parent == Path(dst).parent, Path(src).name != Path(dst).name, Path(dst).name))
        return real(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    assert _trim(path, "2026-06-21", "2026-09-30") == 1
    assert renames == [(True, True, "isone_da_rt_t.parquet")] and list(tmp_path.glob("*.tmp")) == []
    before = path.stat().st_mtime_ns
    assert _trim(path, "2026-06-21", "2026-09-30") == 0               # nothing to cut: the file is not rewritten
    assert path.stat().st_mtime_ns == before and len(renames) == 1


def test_a_kill_during_the_trim_of_a_step_resumes_successfully(registered, monkeypatch):
    """The reviewer's check E: all steps were recorded before the trim loop, so a kill during it was unrecoverable."""
    from es.sealed_pull import verify_manifest
    write, tmp = registered
    proto, reg, sha = write()
    real, calls = os.replace, []

    def killed(src, dst):
        if Path(dst).name == "isone_da_rt_test.parquet":
            raise KeyboardInterrupt("killed while the trim of the ISO-NE file was being written")
        return real(src, dst)

    monkeypatch.setattr(os, "replace", killed)
    with pytest.raises(KeyboardInterrupt):
        _a_pull(tmp, sha, calls)
    monkeypatch.setattr(os, "replace", real)
    root = tmp / "sealed_2026Q3"
    assert json.loads((root / ".state.json").read_text())["done"] == ["NYISO"]        # ISONE is not recorded as done
    assert pd.read_parquet(root / "isone_da_rt_test.parquet")["timestamp_utc"].max() > T("2026-10-01")  # untrimmed
    calls.clear()
    root, _ = _a_pull(tmp, sha, calls, resume=True)
    assert [c[0] for c in calls] == ["ISONE", "MISO", "CAISO", "fmm", "nyiso_rtint"]  # the interrupted step runs again
    assert pd.read_parquet(root / "isone_da_rt_test.parquet")["timestamp_utc"].max() < T("2026-10-01")
    verify_manifest(root)
    assert list(root.rglob("*.tmp")) == []


def test_each_step_is_trimmed_before_it_is_recorded_as_done(registered):
    write, tmp = registered
    proto, reg, sha = write()
    seen = []
    builders = _complete_builders([])
    inner = builders["ISONE"]

    def spy(start, end, root):
        state = json.loads((root / ".state.json").read_text())["done"]
        seen.append((state, pd.read_parquet(root / "nyiso_da_rt_test.parquet")["timestamp_utc"].max()))
        return inner(start, end, root)

    builders["ISONE"] = spy
    _a_pull(tmp, sha, builders=builders)
    assert seen == [(["NYISO"], T("2026-07-15 18:00"))]              # NYISO was trimmed (no October row) and recorded


def test_leftover_temp_files_of_an_interrupted_attempt_are_removed_and_logged_on_resume(registered):
    write, tmp = registered
    proto, reg, sha = write()
    builders = _complete_builders([])
    good = builders["MISO"]
    failures = {"left": 1}

    def flaky(start, end, root):
        if failures["left"]:
            failures["left"] -= 1
            (root / "isone_da_rt_test.parquet.tmp").write_bytes(b"half")      # what a kill mid-write leaves
            raise OSError("dns")
        return good(start, end, root)

    builders["MISO"] = flaky
    with pytest.raises(OSError):
        _a_pull(tmp, sha, builders=builders)
    root, _ = _a_pull(tmp, sha, builders=builders, resume=True)
    assert list(root.rglob("*.tmp")) == []
    assert "removed 1 leftover temp file(s): isone_da_rt_test.parquet.tmp" in (root / "PULL_LOG.txt").read_text()


# ----------------------------------------------------------------------------- I5: the day check always runs
def test_the_day_check_applies_to_builders_passed_explicitly(registered, monkeypatch):
    """The reviewer's check H: builders={**_default_builders()} used to switch the check off."""
    import es.sealed_pull as sp
    write, tmp = registered
    proto, reg, sha = write()
    _use_default_builders(monkeypatch, [], skip={("ISONE", ".Z.MAINE", "2026-06-22")})
    with pytest.raises(sp.MissingDaysError, match=r"ISONE.*\.Z\.MAINE.*2026-06-22"):
        sp.pull_sealed("A", sha, data_root=tmp, builders={**sp._default_builders()}, reference_dirs=[])


def test_one_node_fakes_fail_the_day_check_unless_it_is_switched_off_and_the_switch_is_logged(registered):
    from es.sealed_pull import MissingDaysError
    write, tmp = registered
    proto, reg, sha = write()
    (tmp / "strict").mkdir()
    with pytest.raises(MissingDaysError, match="NYISO"):
        _a_pull(tmp / "strict", sha, check_days=True)
    root, _ = _a_pull(tmp, sha)                                       # the helper's default is check_days=False
    log = (root / "PULL_LOG.txt").read_text()
    assert "check_days: False" in log
    assert "every node-map node has rows" not in log                  # and no day check ran


def test_the_day_check_is_logged_as_on_when_it_runs(registered, monkeypatch):
    from es.sealed_pull import pull_sealed
    write, tmp = registered
    proto, reg, sha = write()
    _use_default_builders(monkeypatch, [])
    root, _ = pull_sealed("A", sha, data_root=tmp, reference_dirs=[])
    assert "check_days: True" in (root / "PULL_LOG.txt").read_text()


def test_check_days_false_is_refused_for_the_real_downloaders(registered, monkeypatch):
    import es.sealed_pull as sp
    write, tmp = registered
    proto, reg, sha = write()
    monkeypatch.setattr(sp, "_default_builders", lambda: pytest.fail("the default builders were built"))
    with pytest.raises(ValueError, match="check_days=False is for injected builders"):
        sp.pull_sealed("A", sha, data_root=tmp, check_days=False)
    assert not (tmp / "sealed_2026Q3").exists()


@pytest.mark.parametrize("bad", [0, 1, None, "False"])
def test_check_days_must_be_a_bool(registered, bad):
    write, tmp = registered
    proto, reg, sha = write()
    with pytest.raises(ValueError, match="check_days"):
        _a_pull(tmp, sha, check_days=bad)
    assert not (tmp / "sealed_2026Q3").exists()


# ----------------------------------------------------------------------------- I6: the settlement tables, checked too
def _window_days_utc(tz):
    import es.sealed_pull as sp
    first, _, last, _ = sp.SEALED_WINDOWS["A"]
    s, e = sp.local_day_bounds_utc(first, last, tz)
    return pd.date_range(s, e, freq="h", inclusive="left")


def _fmm_table_builder(skip=(), partial=(), nonfinite=None):
    """The FMM step as a fake: a complete hourly table for the pull (every CAISO node, every local hour, n15 == 4),
    except the (node, 'YYYY-MM-DD') pairs in `skip` (no rows that day) and `partial` (only n15 == 3 hours that day),
    and the pairs in `nonfinite` {(node, day): value}, whose first hour holds `value` as its price."""
    nonfinite = {} if nonfinite is None else nonfinite

    def build(start, end, root):
        rows, bad_done = [], set()
        for node in _nodes_by_market()["CAISO"]:
            for h in _window_days_utc("America/Los_Angeles"):
                day = f"{h.tz_convert('America/Los_Angeles'):%Y-%m-%d}"
                if (node, day) not in skip:
                    first_hour = (node, day) not in bad_done
                    price = nonfinite[(node, day)] if (node, day) in nonfinite and first_hour else 100.0
                    bad_done.add((node, day))
                    rows.append((node, h, price, 3 if (node, day) in partial else 4))
        (root / "fmm").mkdir(exist_ok=True)
        pd.DataFrame(rows, columns=["node", "timestamp_utc", "fmm_lmp", "n15"]).to_parquet(
            root / "fmm" / "caiso_fmm_x.parquet", index=False)
    return build


def _nyiso_table_builder(skip=(), nonfinite=None):
    """The NYISO integrated step as a fake; `nonfinite` {(zone, 'YYYY-MM-DD'): value} sets the first hour's price."""
    nonfinite = {} if nonfinite is None else nonfinite

    def build(start, end, root):
        rows, bad_done = [], set()
        for zone in _nodes_by_market()["NYISO"]:
            for h in _window_days_utc("America/New_York"):
                day = f"{h.tz_convert('America/New_York'):%Y-%m-%d}"
                if (zone, day) not in skip:
                    first_hour = (zone, day) not in bad_done
                    price = nonfinite[(zone, day)] if (zone, day) in nonfinite and first_hour else 5.0
                    bad_done.add((zone, day))
                    rows.append((zone, h, price))
        (root / "nyiso_rtint").mkdir(exist_ok=True)
        pd.DataFrame(rows, columns=["node", "timestamp_utc", "rt_int_lmp"]).to_parquet(
            root / "nyiso_rtint" / "nyiso_rtint_x.parquet", index=False)
    return build


@pytest.mark.parametrize("how", ["no rows that day", "only incomplete hours that day"])
def test_a_day_without_a_complete_fmm_hour_stops_the_step_like_a_price_day(registered, monkeypatch, how):
    import es.sealed_pull as sp
    write, tmp = registered
    proto, reg, sha = write()
    _use_default_builders(monkeypatch, [])
    gap = {(SDGE, "2026-06-22")}
    fmm_gap = {"skip": gap} if how[0] == "n" else {"partial": gap}
    builders = {**sp._default_builders(), "FMM": _fmm_table_builder(**fmm_gap)}
    kw = dict(data_root=tmp, reference_dirs=[])
    with pytest.raises(sp.MissingDaysError, match=r"fmm.*DLAP_SDGE-APND: 2026-06-22"):
        sp.pull_sealed("A", sha, builders=builders, **kw)
    root = tmp / "pull_unit"
    assert json.loads((root / ".state.json").read_text())["done"] == ["NYISO", "ISONE", "MISO", "CAISO"]
    assert "fmm" in (root / "PULL_LOG.txt").read_text() and "DLAP_SDGE-APND" in (root / "PULL_LOG.txt").read_text()
    builders["FMM"] = _fmm_table_builder()                            # the next attempt is complete
    root, _ = sp.pull_sealed("A", sha, builders=builders, resume=True, **kw)
    sp.verify_manifest(root)


def test_a_zone_day_missing_from_the_nyiso_integrated_table_stops_the_step(registered, monkeypatch):
    import es.sealed_pull as sp
    write, tmp = registered
    proto, reg, sha = write()
    _use_default_builders(monkeypatch, [])
    builders = {**sp._default_builders(), "NYISO_INT": _nyiso_table_builder(skip={("N.Y.C.", "2026-06-22")})}
    with pytest.raises(sp.MissingDaysError, match=r"nyiso_rtint.*N\.Y\.C\.: 2026-06-22"):
        sp.pull_sealed("A", sha, data_root=tmp, builders=builders, reference_dirs=[])
    assert json.loads((tmp / "pull_unit" / ".state.json").read_text())["done"] == [
        "NYISO", "ISONE", "MISO", "CAISO", "FMM"]


def test_allow_missing_days_is_keyed_by_series_for_the_settlement_tables(registered, monkeypatch):
    import es.sealed_pull as sp
    write, tmp = registered
    proto, reg, sha = write()
    _use_default_builders(monkeypatch, [])
    builders = {**sp._default_builders(), "FMM": _fmm_table_builder(skip={(SDGE, "2026-06-22")}),
                "NYISO_INT": _nyiso_table_builder(skip={("N.Y.C.", "2026-06-23")})}
    root, _ = sp.pull_sealed("A", sha, data_root=tmp, builders=builders, reference_dirs=[],
                             allow_missing_days={"fmm": ["2026-06-22"], "nyiso_rtint": ["2026-06-23"]})
    log = (root / "PULL_LOG.txt").read_text()
    excused = "1 (node, day) pair(s) without rows, allowed by allow_missing_days: "
    assert f"step fmm: {excused}DLAP_SDGE-APND: 2026-06-22" in log
    assert f"step nyiso_rtint: {excused}N.Y.C.: 2026-06-23" in log
    cov = pd.read_csv(root / "COVERAGE.csv")
    assert cov[(cov["series"] == "fmm") & (cov["node"] == SDGE)]["hours"].iloc[0] == 24  # the 23rd; the 22nd is excused
    assert cov[(cov["series"] == "nyiso_rtint") & (cov["node"] == "N.Y.C.")]["hours"].iloc[0] == 24   # the 22nd only
    assert cov.groupby("series").size().to_dict() == {"fmm": 6, "nyiso_rtint": 10, "price": 33}
    assert list(cov.columns) == ["series", "market", "node", "hours", "expected_hours", "nonfinite_rows", "share"]
    fmm = cov[cov["series"] == "fmm"]
    assert set(fmm["market"]) == {"CAISO"} and fmm["expected_hours"].eq(48).all()
    assert fmm[fmm["node"] != SDGE]["hours"].eq(48).all()


# ----------------------------------------------------------------------------- I7 and M9: short responses and bounds
@pytest.mark.parametrize("query", ["PRC_LMP", "PRC_INTVL_LMP", "PRC_RTPD_LMP"])
def test_a_short_response_is_never_cached_and_the_retry_completes_with_two_requests(tmp_path, query):
    """DAM: the hours of the chunk; RTM: 12 x hours; FMM: 4 x hours. One chunk is served 2 days of 3, then in full."""
    from es.sealed_pull import caiso_exact, caiso_fmm_exact
    calls, short = collections.Counter(), {"left": 1}

    def fetch(url):
        calls[url] += 1
        if f"queryname={query}&" in url and _node(url) == SDGE and short["left"]:
            short["left"] -= 1
            return _good_oasis(url, days=2)
        return _good_oasis(url)

    cache = tmp_path / "raw"
    puller = caiso_fmm_exact if query == "PRC_RTPD_LMP" else caiso_exact
    out = pd.read_parquet(puller("2026-06-21", "2026-06-23", tmp_path / "root", fetch=fetch, cache_dir=cache,
                                 **NO_SLEEP))
    target = [u for u in calls if f"queryname={query}&" in u and _node(u) == SDGE]
    assert len(target) == 1 and calls[target[0]] == 2                    # the short answer, then the complete one
    assert all(c == 1 for u, c in calls.items() if u != target[0])
    assert out.groupby("node").size().eq(72).all()                       # nothing is missing from the table
    cached = cache / (hashlib.sha256(target[0].encode()).hexdigest() + ".bin")
    assert cached.read_bytes() == _good_oasis(target[0])                 # and what was cached is the complete answer


def test_a_chunk_served_short_once_is_requested_exactly_twice_and_the_pull_completes(registered, monkeypatch):
    """The reviewer's check F, through pull_sealed: the short answer used to be cached for good."""
    from es.sealed_pull import pull_sealed, verify_manifest
    write, tmp = registered
    proto, reg, sha = write()
    served, urls = {"short": 0}, []

    def short_first(url):
        if "queryname=PRC_LMP&" in url and _node(url) == SDGE and served["short"] == 0:
            served["short"] = 1
            return _good_oasis(url, days=2)
        return None

    _use_default_builders(monkeypatch, urls, fetch_override=short_first)
    root, _ = pull_sealed("A", sha, data_root=tmp, reference_dirs=[])
    target = [u for u in urls if "queryname=PRC_LMP&" in u and _node(u) == SDGE]
    assert len(target) == 2 and target[0] == target[1]                   # requested exactly twice
    cov = pd.read_csv(root / "COVERAGE.csv")
    assert cov[(cov["series"] == "price") & (cov["node"] == SDGE)]["share"].iloc[0] == 1.0
    verify_manifest(root)


def test_a_response_cut_off_mid_day_is_retried_not_accepted(tmp_path):
    """The reviewer's check F': a response that stopped at 06:00 on the last day passed the day check."""
    from es.sealed_pull import caiso_exact
    calls, served = [], {"short": 0}
    cache = tmp_path / "raw"

    def fetch(url):
        calls.append(url)
        if "queryname=PRC_LMP&" in url and _node(url) == SDGE and not served["short"]:
            served["short"] = 1
            s = _ts(url, "startdatetime")
            return _oasis_zip(SDGE, _grid(s, s + pd.Timedelta(days=2, hours=6), 60, lambda t: 10.0))
        return _good_oasis(url)

    out = pd.read_parquet(caiso_exact("2026-06-21", "2026-06-23", tmp_path / "root", fetch=fetch, cache_dir=cache,
                                      **NO_SLEEP))
    assert out.groupby("node").size().eq(72).all() and len(calls) == 12 + 1


def test_a_response_that_never_covers_its_grid_raises_with_the_count_and_caches_nothing(tmp_path):
    from es.sealed_pull import caiso_exact
    calls, cache = [], tmp_path / "raw"

    def always_short(url):
        calls.append(url)
        return _good_oasis(url, days=2)

    with pytest.raises(RuntimeError, match="after 8 attempts") as err:
        caiso_exact("2026-06-21", "2026-06-23", tmp_path / "root", fetch=always_short, cache_dir=cache, **NO_SLEEP)
    assert len(calls) == 8 and "short: 24 of 72" in str(err.value)  # one day of three is missing from the first chunk
    assert "allow_missing_days" not in str(err.value)               # that dict excuses days, not short answers
    assert not cache.exists() or list(cache.glob("*.bin")) == []


def test_an_rtm_answer_missing_a_five_minute_interval_is_retried_even_if_every_hour_is_there(tmp_path):
    """Every hour has an interval, so an hourly grid would pass it; the 5-minute grid does not."""
    from es.sealed_pull import caiso_exact
    cache, calls, short = tmp_path / "raw", [], {"once": True}

    def fetch(url):
        calls.append(url)
        if "queryname=PRC_INTVL_LMP&" in url and _node(url) == SDGE and short["once"]:
            short["once"] = False
            s, e = _ts(url, "startdatetime"), _ts(url, "enddatetime")
            missing = s + pd.Timedelta(hours=5, minutes=35)
            return _oasis_zip(SDGE, [(t, v) for t, v in _grid(s, e, 5, lambda t: 10.0) if t != missing])
        return _good_oasis(url)

    out = pd.read_parquet(caiso_exact("2026-06-21", "2026-06-21", tmp_path / "root", fetch=fetch, cache_dir=cache,
                                      **NO_SLEEP))
    target = [u for u in calls if "queryname=PRC_INTVL_LMP&" in u and _node(u) == SDGE]
    assert len(target) == 2 and target[0] == target[1] and len(calls) == 12 + 1
    assert out.groupby("node").size().eq(24).all() and out["rt_lmp"].eq(10.0).all()       # the retry was whole


def test_an_fmm_answer_missing_a_quarter_hour_is_retried_even_if_every_hour_is_there(tmp_path):
    """FMM: 4 x hours intervals for each node's chunk."""
    from es.sealed_pull import caiso_fmm_exact
    cache, calls, short = tmp_path / "raw", [], {"once": True}

    def fetch(url):
        calls.append(url)
        if _node(url) == SDGE and short["once"]:
            short["once"] = False
            s, e = _ts(url, "startdatetime"), _ts(url, "enddatetime")
            missing = s + pd.Timedelta(hours=5, minutes=30)
            return _fmm_zip(SDGE, [(t, v) for t, v in _grid(s, e, 15, lambda t: 100.0) if t != missing])
        return _good_oasis(url)

    out = pd.read_parquet(caiso_fmm_exact("2026-06-21", "2026-06-21", tmp_path / "fmm", fetch=fetch, cache_dir=cache,
                                          **NO_SLEEP))
    target = [u for u in calls if _node(u) == SDGE]
    assert len(target) == 2 and target[0] == target[1] and len(calls) == 6 + 1
    assert out["n15"].eq(4).all() and out.groupby("node").size().eq(24).all()


def test_a_short_response_left_in_the_cache_by_an_earlier_run_is_fetched_again(tmp_path):
    from es.sealed_pull import caiso_exact
    fetch, calls = _memoised_oasis()
    cache = tmp_path / "raw"
    cache.mkdir()
    url = ("https://oasis.caiso.com/oasisapi/SingleZip?queryname=PRC_LMP&startdatetime=20260621T07:00-0000"
           f"&enddatetime=20260624T07:00-0000&market_run_id=DAM&node={SDGE}&version=1")
    (cache / (hashlib.sha256(url.encode()).hexdigest() + ".bin")).write_bytes(_good_oasis(url, days=2))
    caiso_exact("2026-06-21", "2026-06-23", tmp_path / "root", fetch=fetch, cache_dir=cache, **NO_SLEEP)
    assert calls.count(url) == 1 and len(calls) == 12                    # every URL once; the stale entry was replaced
    assert (cache / (hashlib.sha256(url.encode()).hexdigest() + ".bin")).read_bytes() == _good_oasis(url)


def test_rows_outside_the_requested_interval_are_rejected_and_retried(tmp_path):
    """M9, sealed or not: the interval that starts at the chunk's end is not part of the answer."""
    from es.sealed_pull import caiso_exact
    calls = []

    def fetch(url):
        calls.append(url)
        if len(calls) == 1:                                              # the first answer reaches one hour too far
            s, e = _ts(url, "startdatetime"), _ts(url, "enddatetime")
            return _oasis_zip(_node(url), _grid(s, e, 60, lambda t: 10.0, inclusive_end=True))
        return _good_oasis(url)

    out = pd.read_parquet(caiso_exact("2026-07-01", "2026-07-01", tmp_path / "root", fetch=fetch, _sealed=True,
                                      **NO_SLEEP))
    assert len(calls) == 12 + 1 and out["timestamp_utc"].max() == T("2026-07-02 06:00")


def test_a_development_pull_that_receives_a_sealed_row_raises_at_once(tmp_path):
    from es.guard import SealedDataError
    from es.sealed_pull import caiso_exact
    calls = []

    def fetch(url):
        calls.append(url)
        s, e = _ts(url, "startdatetime"), _ts(url, "enddatetime")
        return _oasis_zip(_node(url), _grid(s, e, 60, lambda t: 10.0, inclusive_end=True))       # 1 July 00:00 PDT too

    with pytest.raises(SealedDataError, match="sealed"):
        caiso_exact("2026-06-30", "2026-06-30", tmp_path / "root", fetch=fetch, **NO_SLEEP)
    assert len(calls) == 1                                               # not retried: it is not a transient fault


def test_nyiso_integrated_rows_outside_their_month_are_rejected_and_retried(tmp_path):
    """M9: an archive holds the local days of its own month; a row of the month before it is a short or wrong answer."""
    from es.sealed_pull import nyiso_integrated_months
    calls = []

    def fetch(url):
        calls.append(url)
        if len(calls) == 1:
            return _nyiso_zip({"20260531": [("05/31/2026 23:00", "N.Y.C.", 9.0)],
                               "20260630": [("06/30/2026 23:00", "N.Y.C.", 1.0)]})
        return _nyiso_zip({"20260630": [("06/30/2026 23:00", "N.Y.C.", 1.0)]})

    out = pd.read_parquet(nyiso_integrated_months("2026-06", "2026-06", tmp_path / "x", fetch=fetch, **NO_SLEEP))
    assert len(calls) == 2 and out["rt_int_lmp"].tolist() == [1.0]


def test_an_empty_nyiso_month_is_retried_then_raised(tmp_path):
    """The month validator has no grid to compare with, so the 'no rows' check is what refuses an empty archive."""
    from es.sealed_pull import nyiso_integrated_months
    calls = []

    def empty(url):
        calls.append(url)
        return _nyiso_zip({"20260630": []})

    with pytest.raises(RuntimeError, match="no rows for NYISO integrated 2026-06"):
        nyiso_integrated_months("2026-06", "2026-06", tmp_path / "x", fetch=empty, **NO_SLEEP)
    assert len(calls) == 8 and not (tmp_path / "x").exists()


def test_a_cached_nyiso_month_with_no_rows_is_fetched_again(tmp_path):
    from es.sealed_pull import NYISO_URL, nyiso_integrated_months
    cache, calls = tmp_path / "raw", []
    cache.mkdir()
    url = NYISO_URL.format(ym="202606")
    (cache / (hashlib.sha256(url.encode()).hexdigest() + ".bin")).write_bytes(_nyiso_zip({"20260630": []}))
    good = _nyiso_zip({"20260630": [("06/30/2026 23:00", "N.Y.C.", 1.0)]})     # one object: zip headers hold the clock

    def fetch(u):
        calls.append(u)
        return good

    out = pd.read_parquet(nyiso_integrated_months("2026-06", "2026-06", tmp_path / "x", fetch=fetch, cache_dir=cache,
                                                  **NO_SLEEP))
    assert calls == [url] and out["rt_int_lmp"].tolist() == [1.0]
    assert (cache / (hashlib.sha256(url.encode()).hexdigest() + ".bin")).read_bytes() == good      # replaced


def test_a_sealed_pull_rejects_nyiso_rows_of_the_next_month_and_retries(tmp_path):
    from es.sealed_pull import nyiso_integrated_months
    calls = []

    def fetch(url):
        calls.append(url)
        if len(calls) == 1:
            return _nyiso_zip({"20260731": [("07/31/2026 23:00", "N.Y.C.", 1.0), ("08/01/2026 00:00", "N.Y.C.", 2.0)]})
        return _nyiso_zip({"20260731": [("07/31/2026 23:00", "N.Y.C.", 1.0)]})

    out = pd.read_parquet(nyiso_integrated_months("2026-07", "2026-07", tmp_path / "x", fetch=fetch, _sealed=True,
                                                  **NO_SLEEP))
    assert len(calls) == 2 and out["rt_int_lmp"].tolist() == [1.0]


def test_a_development_pull_that_receives_a_sealed_nyiso_row_raises_at_once(tmp_path):
    from es.guard import SealedDataError
    from es.sealed_pull import nyiso_integrated_months
    calls = []

    def fetch(url):
        calls.append(url)
        return _nyiso_zip({"20260630": [("06/30/2026 23:00", "N.Y.C.", 1.0), ("07/01/2026 00:00", "N.Y.C.", 2.0)]})

    with pytest.raises(SealedDataError, match="sealed"):
        nyiso_integrated_months("2026-06", "2026-06", tmp_path / "x", fetch=fetch, **NO_SLEEP)
    assert len(calls) == 1


# ----------------------------------------------------------------------------- M4 and M10: cache hits
def test_cache_hits_are_logged_with_a_timestamp_and_no_pause_follows_them(tmp_path, capsys):
    from es.sealed_pull import caiso_exact
    fetch, calls = _memoised_oasis()
    cache, pauses = tmp_path / "raw", []
    caiso_exact("2026-06-21", "2026-06-21", tmp_path / "a", fetch=fetch, cache_dir=cache, pause=5.5,
                sleep=pauses.append)
    assert pauses.count(5.5) == 12                                       # a pause after each of the 12 requests
    first = capsys.readouterr().out
    stamp = r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ "
    assert len(re.findall(stamp + r".*: \d[\d,]* bytes on attempt 1$", first, flags=re.M)) == 12
    pauses.clear()
    caiso_exact("2026-06-21", "2026-06-21", tmp_path / "b", fetch=fetch, cache_dir=cache, pause=5.5,
                sleep=pauses.append)
    assert pauses == [] and len(calls) == 12                             # nothing was requested, nothing waited for
    second = capsys.readouterr().out
    assert len(re.findall(stamp + r".*: cache hit [0-9a-f]{64}\.bin$", second, flags=re.M)) == 12


# ----------------------------------------------------------------------------- I8: the override is a dict
@pytest.mark.parametrize("bad", [True, False, ["MISO"], "MISO", 5])
def test_allow_missing_days_must_be_a_dict_or_none(registered, bad):
    """The plan's _fake_pull passed True: that used to crash on bool.items()."""
    write, tmp = registered
    proto, reg, sha = write()
    with pytest.raises(ValueError, match="allow_missing_days must be a dict"):
        _a_pull(tmp, sha, allow_missing_days=bad)
    assert not (tmp / "sealed_2026Q3").exists()


def test_an_empty_dict_and_none_mean_no_override(registered):
    write, tmp = registered
    proto, reg, sha = write()
    root, _ = _a_pull(tmp, sha, allow_missing_days={})
    assert "allow_missing_days: none" in (root / "PULL_LOG.txt").read_text()


# ----------------------------------------------------------------------------- M1: a resume continues the first attempt
def _failing_then_fine_builders(calls=None):
    builders = _complete_builders([] if calls is None else calls)
    good, failures = builders["CAISO"], {"left": 1}

    def flaky(start, end, root):
        if failures["left"]:
            failures["left"] -= 1
            raise OSError("nodename nor servname provided")
        return good(start, end, root)

    builders["CAISO"] = flaky
    return builders


def test_the_state_holds_the_first_attempts_registration_and_a_resume_must_match_it(registered):
    from es.registration import RegistrationError
    write, tmp = registered
    proto, reg, sha = write("protocol v1\n")
    builders = _failing_then_fine_builders()
    with pytest.raises(OSError):
        _a_pull(tmp, sha, builders=builders)
    state = json.loads((tmp / "sealed_2026Q3" / ".state.json").read_text())
    assert (state["protocol_sha256"], state["osf_doi"], state["registered_utc"]) == (
        sha, "10.17605/OSF.IO/ABCDE", "2026-10-20T15:04:00Z")
    proto2, reg2, sha2 = write("protocol v2, edited after the first attempt\n")          # another protocol, registered
    with pytest.raises(RegistrationError, match="started under protocol"):
        _a_pull(tmp, sha2, builders=builders, resume=True)
    proto3, reg3, sha3 = write("protocol v1\n", reg_fields={"osf_doi": "10.17605/OSF.IO/ZZZZZ"})  # another DOI
    assert sha3 == sha
    with pytest.raises(RegistrationError, match="is another registration"):
        _a_pull(tmp, sha3, builders=builders, resume=True)
    write("protocol v1\n")                                                 # the registration the pull began under
    root, _ = _a_pull(tmp, sha, builders=builders, resume=True)
    assert (root / "MANIFEST.sha256").exists()


# ----------------------------------------------------------------------------- addendum 2: DOI, OSF address, or both
DOI, URL = "10.17605/OSF.IO/ABCDE", "https://osf.io/abcde/"
ONLY_URL = {"osf_doi": None, "osf_registration": "abcde"}                # the OSF id as the author may write it


def test_a_pull_under_an_osf_registration_without_a_doi_is_recorded_under_the_osf_address(registered):
    from es.sealed_pull import completed_pulls
    write, tmp = registered
    proto, reg, sha = write(reg_fields=ONLY_URL)
    builders = _failing_then_fine_builders()
    with pytest.raises(OSError):
        _a_pull(tmp, sha, builders=builders)
    root = tmp / "sealed_2026Q3"
    state = json.loads((root / ".state.json").read_text())
    assert (state["osf_doi"], state["osf_registration"], state["registration_identifier"]) == (None, URL, URL)
    root, manifest = _a_pull(tmp, sha, builders=builders, resume=True)
    record = completed_pulls()["A"]
    assert record["started_utc"] == state["started_utc"]                         # the first attempt's, not the resume's
    assert (record["osf_doi"], record["osf_registration"], record["registration_identifier"]) == (None, URL, URL)
    assert record["protocol_sha256"] == sha and record["manifest_sha256"] == manifest
    log = (root / "PULL_LOG.txt").read_text()
    assert log.count("osf_doi: none") == 2 and log.count(f"osf_registration: {URL}") == 2         # once per attempt
    assert log.count(f"registration_identifier: {URL}") == 2
    assert log.count("registration_id: abcde") == 2                                 # the OSF id both forms stand for


def test_a_pull_under_both_identifiers_records_both_and_the_doi_as_the_identifier(registered):
    from es.sealed_pull import completed_pulls
    write, tmp = registered
    proto, reg, sha = write(reg_fields={"osf_registration": URL})
    root, _ = _a_pull(tmp, sha)
    record = completed_pulls()["A"]
    assert (record["osf_doi"], record["osf_registration"], record["registration_identifier"]) == (DOI, URL, DOI)
    log = (root / "PULL_LOG.txt").read_text()
    assert f"osf_doi: {DOI}" in log and f"osf_registration: {URL}" in log and f"registration_identifier: {DOI}" in log
    assert "registration_id: abcde\n" in log and record["registration_id"] == "abcde"


def test_a_pull_under_a_doi_alone_logs_the_osf_address_as_none(registered):
    from es.sealed_pull import completed_pulls
    write, tmp = registered
    proto, reg, sha = write()
    root, _ = _a_pull(tmp, sha)
    record = completed_pulls()["A"]
    assert (record["osf_doi"], record["osf_registration"], record["registration_identifier"]) == (DOI, None, DOI)
    log = (root / "PULL_LOG.txt").read_text()
    assert "osf_registration: none" in log and "registration_id: abcde\n" in log


def test_a_resume_after_the_doi_appears_continues_the_pull_while_the_osf_address_stays(registered):
    """The embargo ends between two attempts and OSF mints the DOI: the author adds a line, and the pull goes on."""
    from es.sealed_pull import completed_pulls
    write, tmp = registered
    proto, reg, sha = write(reg_fields=ONLY_URL)
    builders = _failing_then_fine_builders()
    with pytest.raises(OSError):
        _a_pull(tmp, sha, builders=builders)
    write(reg_fields={"osf_registration": "abcde"})                       # the same registration, now with its DOI
    root, _ = _a_pull(tmp, sha, builders=builders, resume=True)
    record = completed_pulls()["A"]
    assert (record["osf_doi"], record["osf_registration"], record["registration_identifier"]) == (DOI, URL, DOI)
    log = (root / "PULL_LOG.txt").read_text()
    assert f"registration_identifier: {URL}" in log and f"registration_identifier: {DOI}" in log      # both attempts


def test_a_resume_under_another_osf_registration_is_refused(registered):
    from es.registration import RegistrationError
    write, tmp = registered
    proto, reg, sha = write(reg_fields=ONLY_URL)
    builders = _failing_then_fine_builders()
    with pytest.raises(OSError):
        _a_pull(tmp, sha, builders=builders)
    for now in ({"osf_doi": None, "osf_registration": "zzzzz"}, {"osf_doi": "10.17605/OSF.IO/ZZZZZ"}):
        write(reg_fields=now)                                              # another address, and another DOI
        calls = []
        with pytest.raises(RegistrationError, match="is another registration") as err:
            _a_pull(tmp, sha, calls, resume=True)
        assert URL in str(err.value) and calls == []                       # it names what the pull started under


@pytest.mark.parametrize("first, second, recorded",
                         [(ONLY_URL, {"osf_registration": None}, (DOI, None)),
                          ({"osf_registration": None}, ONLY_URL, (None, URL))],
                         ids=["address replaced by its own DOI", "DOI replaced by its own address"])
def test_a_resume_accepts_an_identifier_replaced_by_the_other_form_of_the_same_registration(
        registered, first, second, recorded):
    from es.sealed_pull import completed_pulls
    write, tmp = registered
    proto, reg, sha = write(reg_fields=first)
    builders = _failing_then_fine_builders()
    with pytest.raises(OSError):
        _a_pull(tmp, sha, builders=builders)
    write(reg_fields=second)
    root, _ = _a_pull(tmp, sha, builders=builders, resume=True)
    record = completed_pulls()["A"]
    assert record["registration_id"] == "abcde" and (root / "MANIFEST.sha256").exists()
    assert (record["osf_doi"], record["osf_registration"]) == recorded           # the second attempt's own form


def test_the_identifiers_are_compared_in_one_case(registered):
    write, tmp = registered
    proto, reg, sha = write()
    builders = _failing_then_fine_builders()
    with pytest.raises(OSError):
        _a_pull(tmp, sha, builders=builders)
    write(reg_fields={"osf_doi": "10.17605/osf.io/abcde"})                 # the same DOI in lower case
    root, _ = _a_pull(tmp, sha, builders=builders, resume=True)
    assert (root / "MANIFEST.sha256").exists()


def test_a_state_without_any_identifier_is_refused_on_resume(registered):
    from es.registration import RegistrationError
    write, tmp = registered
    proto, reg, sha = write()
    builders = _failing_then_fine_builders()
    with pytest.raises(OSError):
        _a_pull(tmp, sha, builders=builders)
    state_path = tmp / "sealed_2026Q3" / ".state.json"
    state = json.loads(state_path.read_text())
    for dropped in (("osf_doi", "osf_registration", "public_record", "local_record", "registration_identifier"),
                    ("protocol_sha256",)):
        state_path.write_text(json.dumps({k: v for k, v in state.items() if k not in dropped}))
        with pytest.raises(RegistrationError, match="does not say which registration"):
            _a_pull(tmp, sha, builders=builders, resume=True)


def test_a_state_that_names_two_registrations_is_matched_on_neither(registered):
    """parse_registration never lets a record name two registrations, so a state that does was not written by a
    pull. It is refused, not matched on one of its two ids."""
    from es.registration import RegistrationError
    write, tmp = registered
    proto, reg, sha = write()
    builders = _failing_then_fine_builders()
    with pytest.raises(OSError):
        _a_pull(tmp, sha, builders=builders)
    state_path = tmp / "sealed_2026Q3" / ".state.json"
    state = json.loads(state_path.read_text())
    state["osf_registration"] = "https://osf.io/zzzzz/"                       # beside the DOI of abcde
    state_path.write_text(json.dumps(state))
    with pytest.raises(RegistrationError, match=r"\.state\.json was started under .* is another registration"):
        _a_pull(tmp, sha, builders=builders, resume=True)


def test_a_finished_pull_is_adopted_under_an_osf_registration_and_refused_under_another(registered, monkeypatch):
    from es.sealed_pull import completed_pulls
    write, tmp = registered
    proto, reg, sha = write(reg_fields=ONLY_URL)
    real = os.replace

    def killed(src, dst):
        if Path(dst).name == "PULLS.json":
            raise KeyboardInterrupt("killed while PULLS.json was being written")
        return real(src, dst)

    monkeypatch.setattr(os, "replace", killed)
    with pytest.raises(KeyboardInterrupt):
        _a_pull(tmp, sha)
    monkeypatch.setattr(os, "replace", real)
    from es.registration import RegistrationError
    write(reg_fields={"osf_doi": None, "osf_registration": "zzzzz"})
    with pytest.raises(RegistrationError, match="is another registration") as err:
        _a_pull(tmp, sha, resume=True)
    assert f"started under the registration {URL}; the one now verified, https://osf.io/zzzzz/," in str(err.value)
    assert completed_pulls() == {}
    write(reg_fields=ONLY_URL)
    root, manifest = _a_pull(tmp, sha, resume=True)
    record = completed_pulls()["A"]
    assert (record["osf_doi"], record["osf_registration"], record["manifest_sha256"]) == (None, URL, manifest)


CODE_FILES = ("es/sealed_pull.py", "es/registration.py", "es/settlement.py", "es/guard.py", "es/config.py",
              "es/data.py", "v2/run_decomposition.py", "v2/ingest_nyiso.py", "v2/ingest_isone.py",
              "v2/ingest_miso.py", "v2/NODE_MAP.csv")


def test_the_fingerprint_covers_what_the_pull_runs_and_nothing_it_does_not(tmp_path):
    import es.sealed_pull as sp
    from es.registration import sha256_file
    assert sp.CODE_FILES == CODE_FILES                                        # written out here: a change is deliberate
    fingerprint = sp._code_fingerprint()
    assert list(fingerprint) == list(CODE_FILES)
    assert all(fingerprint[rel] == sha256_file(DART_DEV / rel) for rel in CODE_FILES)
    assert "v2/ingest_caiso.py" not in fingerprint and "v2/ingest_spp.py" not in fingerprint  # it never runs them


def test_every_repository_module_that_sealed_pull_imports_is_in_the_fingerprint():
    """A static scan: a new import in es/sealed_pull.py that the fingerprint does not cover fails here."""
    import es.sealed_pull as sp
    tree = ast.parse((DART_DEV / "es" / "sealed_pull.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 1:             # from .config import x / from . import x
            imported |= {f"es/{name}.py" for name in ([node.module] if node.module else [a.name for a in node.names])}
        elif isinstance(node, ast.ImportFrom) and node.module == "v2":       # from v2 import ingest_isone, ...
            imported |= {f"v2/{alias.name}.py" for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("v2."):
            imported.add(node.module.replace(".", "/") + ".py")              # from v2.run_decomposition import HERE
    assert len(imported) >= 9 and imported <= set(sp.CODE_FILES), sorted(imported - set(sp.CODE_FILES))


def test_a_code_fingerprint_is_logged_for_every_attempt(registered):
    from es.registration import sha256_file
    write, tmp = registered
    proto, reg, sha = write()
    builders = _failing_then_fine_builders()
    with pytest.raises(OSError):
        _a_pull(tmp, sha, builders=builders)
    root, _ = _a_pull(tmp, sha, builders=builders, resume=True)
    log = (root / "PULL_LOG.txt").read_text()
    for rel in CODE_FILES:
        assert log.count(f"code_sha256: {rel} {sha256_file(DART_DEV / rel)}") == 2          # once per attempt
    assert log.count("code: unchanged since the first attempt") == 1


def test_a_change_of_code_between_attempts_is_logged(registered, monkeypatch):
    import es.sealed_pull as sp
    write, tmp = registered
    proto, reg, sha = write()
    builders = _failing_then_fine_builders()
    with pytest.raises(OSError):
        _a_pull(tmp, sha, builders=builders)
    real = sp._code_fingerprint
    monkeypatch.setattr(sp, "_code_fingerprint", lambda: {**real(), "es/sealed_pull.py": "f" * 64})
    root, _ = _a_pull(tmp, sha, builders=builders, resume=True)
    assert "code: CHANGED since the first attempt: es/sealed_pull.py" in (root / "PULL_LOG.txt").read_text()


# ----------------------------------------------------------------------------- M2: reference_dirs
@pytest.mark.parametrize("bad", ["ref", 5, ["ref", 5], {"ref"}, [["ref"]]])
def test_reference_dirs_must_be_a_list_of_paths(registered, bad):
    write, tmp = registered
    proto, reg, sha = write()
    with pytest.raises(ValueError, match="reference_dirs"):
        _a_pull(tmp, sha, reference_dirs=bad)
    assert not (tmp / "sealed_2026Q3").exists()


def test_reference_dirs_may_be_names_under_the_data_root_or_absolute_paths(registered):
    write, tmp = registered
    proto, reg, sha = write()
    ref = tmp / "ref"
    ref.mkdir()
    pd.DataFrame({"market": "ISONE", "node": "X", "node_type": "hub",
                  "timestamp_utc": pd.to_datetime(["2026-06-25 18:00"], utc=True),
                  "da_lmp": 1.0, "rt_lmp": 9.0}).to_parquet(ref / "isone_da_rt_ref.parquet", index=False)
    root, _ = _a_pull(tmp, sha, reference_dirs=(ref,))                       # a tuple of absolute Paths
    overlap = pd.read_csv(root / "OVERLAP_CHECK.csv")
    assert overlap[(overlap["series"] == "price") & (overlap["market"] == "ISONE")]["matched"].iloc[0] == 1


def test_the_overlap_summary_is_logged_in_the_pull_log(registered):
    write, tmp = registered
    proto, reg, sha = write()
    ref = tmp / "ref"
    ref.mkdir()
    pd.DataFrame({"market": "ISONE", "node": "X", "node_type": "hub",
                  "timestamp_utc": pd.to_datetime(["2026-06-25 18:00"], utc=True),
                  "da_lmp": 1.0, "rt_lmp": 9.0}).to_parquet(ref / "isone_da_rt_ref.parquet", index=False)
    root, _ = _a_pull(tmp, sha, reference_dirs=["ref"])
    log = (root / "PULL_LOG.txt").read_text()
    assert ("overlap price ISONE: matched 1, new_only 0, reference_only 0, rows_differing 1, "
            "max_abs_diff_da 0.0, max_abs_diff_rt 8.0") in log
    assert "overlap price NYISO: matched 0, new_only 1, reference_only 0" in log     # no reference rows for the others


# ----------------------------------------------------------------------------- M3: the settlement tables overlap too
def _write_side(directory, series, rows):
    directory.mkdir(parents=True, exist_ok=True)
    cols = ["node", "timestamp_utc", "fmm_lmp", "n15"] if series == "fmm" else ["node", "timestamp_utc", "rt_int_lmp"]
    pd.DataFrame([(n, T(s), *rest) for n, s, *rest in rows], columns=cols).to_parquet(
        directory / f"{series}_x.parquet", index=False)


def _side_writer(series, rows):
    """A fake FMM or NYISO integrated step that writes `rows` into <root>/<series>."""
    return lambda start, end, root: _write_side(root / series, series, rows)


def test_overlap_check_compares_the_settlement_tables_for_the_overlap_days(tmp_path):
    """Overlap days 21 to 30 June. Rows of 20 June (new side) and 19 June (reference) are outside them; the row of
    1 July is in the sealed window; one hour differs in price and another only in its number of intervals."""
    from es.sealed_pull import overlap_check
    new, ref_fmm, ref_nyi = tmp_path / "new", tmp_path / "ref_fmm", tmp_path / "ref_nyi"
    new.mkdir()
    _write_side(new / "fmm", "fmm", [("A", "2026-06-25 18:00", 1.0, 4), ("A", "2026-06-25 19:00", 2.0, 4),
                                     ("A", "2026-06-25 21:00", 4.0, 4), ("A", "2026-06-20 18:00", 7.0, 4),
                                     ("A", "2026-07-01 08:00", 99.0, 4)])
    _write_side(ref_fmm, "fmm", [("A", "2026-06-25 18:00", 1.0, 4), ("A", "2026-06-25 19:00", 2.5, 4),
                                 ("A", "2026-06-25 20:00", 3.0, 4), ("A", "2026-06-25 21:00", 4.0, 3),
                                 ("A", "2026-06-19 18:00", 8.0, 4), ("A", "2026-07-01 08:00", 99.0, 4)])
    _write_side(new / "nyiso_rtint", "nyiso_rtint", [("Z", "2026-06-25 18:00", 5.0), ("Z", "2026-06-25 19:00", 6.0),
                                                     ("Z", "2026-06-20 18:00", 7.0)])
    _write_side(ref_nyi, "nyiso_rtint", [("Z", "2026-06-25 18:00", 5.0), ("Z", "2026-06-19 18:00", 8.0)])
    refs = {"fmm": [ref_fmm], "nyiso_rtint": [ref_nyi]}
    for end_exclusive in ("2026-07-01", "2026-07-02"):  # the seal cap holds even when the caller asks for more
        out = overlap_check(new, [], "2026-06-21", end_exclusive, side_references=refs)
        fmm = out[out["series"] == "fmm"].iloc[0]
        assert (fmm["market"], fmm["matched"], fmm["new_only"], fmm["reference_only"], fmm["rows_differing"]) == (
            "CAISO", 3, 0, 1, 2)                      # 25 June 18:00, 19:00 and 21:00; 20:00 is only in the reference
        assert fmm["max_abs_diff_rt"] == pytest.approx(0.5)
        nyi = out[out["series"] == "nyiso_rtint"].iloc[0]
        assert (nyi["market"], nyi["matched"], nyi["new_only"], nyi["reference_only"], nyi["rows_differing"]) == (
            "NYISO", 1, 1, 0, 0)
    out = overlap_check(new, [], "2026-06-21", "2026-07-02", _sealed=True, side_references=refs)
    assert out[out["series"] == "fmm"].iloc[0]["matched"] == 4          # pull_sealed's own call lifts the cap
    out = overlap_check(new, [], "2026-06-21", "2026-07-01", side_references={"fmm": [tmp_path / "none"]})
    assert out[out["series"] == "fmm"].iloc[0]["note"] == "no reference files"


def test_a_settlement_table_without_the_value_columns_is_noted_in_the_overlap_not_a_crash(tmp_path):
    """The overlap check is logged, not enforced: a table of another shape ends up as a note row, not a crash."""
    from es.sealed_pull import overlap_check
    new, ref = tmp_path / "new", tmp_path / "ref"
    (new / "fmm").mkdir(parents=True)                                  # the new table sits under new/fmm, as in a pull
    pd.DataFrame({"node": ["A"], "timestamp_utc": [T("2026-06-25 18:00")]}).to_parquet(
        new / "fmm" / "fmm_bare.parquet", index=False)
    _write_side(ref, "fmm", [("A", "2026-06-25 18:00", 1.0, 4)])
    out = overlap_check(new, [], "2026-06-21", "2026-07-01", side_references={"fmm": [ref]})
    row = out[out["series"] == "fmm"].iloc[0]
    assert row["market"] == "CAISO" and "fmm_lmp" in row["note"] and "n15" in row["note"]


def test_pull_sealed_overlaps_the_settlement_tables_with_the_development_files(registered):
    from es.sealed_pull import pull_sealed
    write, tmp = registered
    proto, reg, sha = write()
    _write_side(tmp / "caiso_fmm_dev", "fmm", [("TH_SP15_GEN-APND", "2026-06-25 18:00", 1.0, 4),
                                               ("TH_SP15_GEN-APND", "2026-06-25 19:00", 2.5, 4)])
    _write_side(tmp / "nyiso_rtint_dev", "nyiso_rtint", [("N.Y.C.", "2026-06-25 18:00", 5.0)])
    builders = _complete_builders([])
    builders["FMM"] = _side_writer("fmm", [("TH_SP15_GEN-APND", "2026-06-25 18:00", 1.0, 4),
                                           ("TH_SP15_GEN-APND", "2026-06-25 19:00", 2.0, 4)])
    builders["NYISO_INT"] = _side_writer("nyiso_rtint", [("N.Y.C.", "2026-06-25 18:00", 5.0)])
    root, _ = pull_sealed("A", sha, data_root=tmp, builders=builders, check_days=False)         # default reference_dirs
    overlap = pd.read_csv(root / "OVERLAP_CHECK.csv")
    fmm = overlap[overlap["series"] == "fmm"].iloc[0]
    assert (fmm["matched"], fmm["rows_differing"]) == (2, 1)
    assert overlap[overlap["series"] == "nyiso_rtint"].iloc[0]["matched"] == 1
    assert overlap[overlap["series"] == "price"]["note"].iloc[0] == "no reference files"  # no holdout_2026H1 here
    log = (root / "PULL_LOG.txt").read_text()
    assert "overlap fmm CAISO: matched 2, new_only 0, reference_only 0, rows_differing 1" in log
    assert "overlap nyiso_rtint NYISO: matched 1" in log


def test_the_default_references_of_test_a_are_the_h1_files_and_the_development_tables(registered):
    from es.sealed_pull import REFERENCE_DIRS, SIDE_REFERENCE_DIRS
    assert REFERENCE_DIRS == {"A": ("holdout_2026H1",), "B": ("sealed_2026Q3",)}
    assert SIDE_REFERENCE_DIRS == {"A": {"fmm": ("caiso_fmm_dev",), "nyiso_rtint": ("nyiso_rtint_dev",)}}
    write, tmp = registered
    proto, reg, sha = write()
    h1 = tmp / "holdout_2026H1"
    h1.mkdir()
    pd.DataFrame({"market": "ISONE", "node": "X", "node_type": "hub", "timestamp_utc": [T("2026-06-25 18:00")],
                  "da_lmp": 1.0, "rt_lmp": 1.0}).to_parquet(h1 / "isone_da_rt_h1.parquet", index=False)
    root, _ = _a_pull(tmp, sha, reference_dirs=None)                               # the defaults
    overlap = pd.read_csv(root / "OVERLAP_CHECK.csv")
    isone = overlap[(overlap["series"] == "price") & (overlap["market"] == "ISONE")].iloc[0]
    assert (isone["matched"], isone["new_only"], isone["reference_only"], isone["rows_differing"]) == (1, 0, 0, 0)


def test_explicit_reference_dirs_replace_the_defaults_including_the_settlement_tables(registered):
    write, tmp = registered
    proto, reg, sha = write()
    _write_side(tmp / "caiso_fmm_dev", "fmm", [("X", "2026-06-25 18:00", 1.0, 4)])
    _write_side(tmp / "nyiso_rtint_dev", "nyiso_rtint", [("X", "2026-06-25 18:00", 5.0)])
    (tmp / "other_ref").mkdir()
    root, _ = _a_pull(tmp, sha, reference_dirs=["other_ref"])
    assert set(pd.read_csv(root / "OVERLAP_CHECK.csv")["series"]) == {"price"}      # no fmm or nyiso_rtint rows at all


def test_test_bs_settlement_tables_are_compared_with_test_as(registered):
    from es.sealed_pull import pull_sealed
    write, tmp = registered
    proto, reg, sha = write()
    a_builders = _b_fakes(A_ROWS)
    a_builders["FMM"] = _side_writer("fmm", [("TH_SP15_GEN-APND", "2026-09-25 18:00", 1.0, 4)])
    a_builders["NYISO_INT"] = _side_writer("nyiso_rtint", [("N.Y.C.", "2026-09-25 18:00", 5.0)])
    (tmp / "a_data").mkdir()
    pull_sealed("A", sha, data_root=tmp / "a_data", builders=a_builders, reference_dirs=[], check_days=False)
    b_builders = _b_fakes(B_ROWS)
    b_builders["FMM"] = _side_writer("fmm", [("TH_SP15_GEN-APND", "2026-09-25 18:00", 2.0, 4)])
    b_builders["NYISO_INT"] = _side_writer("nyiso_rtint", [("N.Y.C.", "2026-09-25 18:00", 5.0)])
    (tmp / "b_data").mkdir()
    b_root, _ = pull_sealed("B", sha, data_root=tmp / "b_data", builders=b_builders, check_days=False)
    overlap = pd.read_csv(b_root / "OVERLAP_CHECK.csv")
    fmm = overlap[overlap["series"] == "fmm"].iloc[0]
    assert (fmm["matched"], fmm["rows_differing"], fmm["max_abs_diff_rt"]) == (1, 1, 1.0)
    assert overlap[overlap["series"] == "nyiso_rtint"].iloc[0]["rows_differing"] == 0


# ----------------------------------------------------------------------------- M8: durable writes
def test_atomic_writes_flush_the_file_then_rename_then_flush_the_directory(tmp_path, monkeypatch):
    import es.sealed_pull as sp
    events, real_replace = [], os.replace

    def fsync(fd):
        events.append("fsync " + ("dir" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file"))

    def replace(src, dst):
        events.append("replace")
        return real_replace(src, dst)

    monkeypatch.setattr(sp, "_fsync", fsync)
    monkeypatch.setattr(os, "replace", replace)
    sp._atomic_write(tmp_path / "x.bin", b"payload")
    assert events == ["fsync file", "replace", "fsync dir"] and (tmp_path / "x.bin").read_bytes() == b"payload"
    events.clear()
    sp._atomic_to_parquet(pd.DataFrame({"a": [1]}), tmp_path / "y.parquet")
    assert events == ["fsync file", "replace", "fsync dir"]


def test_fsync_uses_the_full_flush_where_the_platform_has_it(monkeypatch):
    import es.sealed_pull as sp
    seen = []
    monkeypatch.setattr(os, "fsync", lambda fd: seen.append(("fsync", fd)))
    full = types.SimpleNamespace(F_FULLFSYNC=51, fcntl=lambda fd, cmd: seen.append(("full", fd, cmd)))
    monkeypatch.setattr(sp, "fcntl", full)
    sp._fsync(7)
    assert seen == [("full", 7, 51)]  # macOS: plain fsync does not flush the drive cache
    seen.clear()

    def refuses(fd, cmd):
        raise OSError("not supported on this file system")

    monkeypatch.setattr(sp, "fcntl", types.SimpleNamespace(F_FULLFSYNC=51, fcntl=refuses))
    sp._fsync(8)
    assert seen == [("fsync", 8)]                                     # fall back to fsync
    seen.clear()
    monkeypatch.setattr(sp, "fcntl", types.SimpleNamespace())          # no F_FULLFSYNC (Linux)
    sp._fsync(9)
    assert seen == [("fsync", 9)]


# =====================================================================================================
# Addendum 3 to fix round 1 (controller Ruling 82): only rows whose prices are finite numbers count as present
# =====================================================================================================
NAN, INF = float("nan"), float("inf")
HOURS = tuple(range(14, 22))                   # eight UTC hours that fall on one local day in every market in June


def test_a_row_counts_only_when_its_prices_are_finite_numbers():
    from es.sealed_pull import _row_flags
    prices = pd.DataFrame({"da_lmp": [1.0, NAN, INF, 1.0, -INF, 1.0], "rt_lmp": [1.0, 1.0, 1.0, NAN, 1.0, 2.0]})
    finite, usable = _row_flags(prices, "ISONE")
    assert finite.tolist() == [True, False, False, False, False, True] and usable.tolist() == finite.tolist()
    fmm = pd.DataFrame({"fmm_lmp": [1.0, NAN, 1.0, INF], "n15": [4, 4, 3, 4]})
    finite, usable = _row_flags(fmm, "fmm")                        # the hour of three intervals is finite, not usable
    assert finite.tolist() == [True, False, True, False] and usable.tolist() == [True, False, False, False]
    assert _row_flags(pd.DataFrame({"rt_int_lmp": [5.0, NAN]}), "nyiso_rtint")[0].tolist() == [True, False]
    text = pd.DataFrame({"da_lmp": [1.0, 2.0], "rt_lmp": ["n/a", "3"]})        # text that is no number is not finite
    assert _row_flags(text, "MISO")[0].tolist() == [False, True]


def test_a_missing_price_column_is_not_finite_and_raises_in_the_day_check():
    from es.sealed_pull import _row_flags
    lacking = pd.DataFrame({"da_lmp": [1.0, 2.0]})
    assert _row_flags(lacking, "MISO")[0].tolist() == [False, False]
    with pytest.raises(RuntimeError, match=r"have no \['rt_lmp'\] column"):
        _row_flags(lacking, "MISO", strict=True)
    no_count = pd.DataFrame({"fmm_lmp": [1.0]})
    assert _row_flags(no_count, "fmm")[1].tolist() == [False]
    with pytest.raises(RuntimeError, match=r"have no \['n15'\] column"):
        _row_flags(no_count, "fmm", strict=True)


def test_coverage_counts_finite_hours_and_non_finite_rows_separately(tmp_path):
    from es.sealed_pull import coverage_manifest
    root = tmp_path / "pull"
    root.mkdir()
    stamps = ["2026-06-22 18:00", "2026-06-22 19:00", "2026-06-22 20:00",
              "2026-06-20 18:00"]                                                  # the last is before the window
    pd.DataFrame({"market": "CAISO", "node": "TH_SP15_GEN-APND", "node_type": "hub",
                  "timestamp_utc": [T(x) for x in stamps], "da_lmp": [1.0, 1.0, INF, 1.0],
                  "rt_lmp": [1.0, NAN, 1.0, NAN]}).to_parquet(root / "caiso_da_rt_x.parquet", index=False)
    _write_side(root / "fmm", "fmm", [(SDGE, "2026-06-22 18:00", 1.0, 4), (SDGE, "2026-06-22 19:00", NAN, 4),
                                      (SDGE, "2026-06-22 20:00", 1.0, 3)])
    _write_side(root / "nyiso_rtint", "nyiso_rtint", [("N.Y.C.", "2026-06-22 18:00", 5.0),
                                                      ("N.Y.C.", "2026-06-22 19:00", -INF),
                                                      ("N.Y.C.", "2026-06-22 20:00", NAN)])
    cov = coverage_manifest(root, "2026-06-22", "2026-06-23")

    def counts(series, node):
        row = cov[(cov["series"] == series) & (cov["node"] == node)].iloc[0]
        return int(row["hours"]), int(row["nonfinite_rows"])

    # a NaN rt and an infinite da; the row before the window is not counted, nor is the hour of three intervals
    assert counts("price", "TH_SP15_GEN-APND") == (1, 2)
    assert counts("fmm", SDGE) == (1, 1)
    assert counts("nyiso_rtint", "N.Y.C.") == (1, 2)
    rest = cov[~cov["node"].isin(["TH_SP15_GEN-APND", SDGE, "N.Y.C."])]
    assert (rest["hours"] == 0).all() and (rest["nonfinite_rows"] == 0).all()
    assert (cov["share"] == cov["hours"] / cov["expected_hours"]).all()           # the share counts finite hours only


def test_day_gaps_tell_days_without_usable_rows_from_days_short_of_their_hours(tmp_path):
    from es.sealed_pull import _day_gaps, _missing_days
    root = tmp_path / "pull"
    root.mkdir()
    a, b, c, d = _nodes_by_market()["ISONE"][:4]                                   # d has nothing wrong with it
    first, last = "2026-06-21", "2026-06-23"
    rows = _full_price_rows("ISONE", _nodes_by_market()["ISONE"], first, last,
                            skip={(c, "2026-06-23")}, prices={(b, "2026-06-22", 14): (1.0, NAN),       # one bad hour
                                                              **{(c, "2026-06-21", h): (INF, 1.0) for h in range(24)}})
    rows = [r for r in rows if not (r[1] == a and r[3] == T("2026-06-22 09:00"))]       # a row the downloader dropped
    pd.DataFrame(rows, columns=PRICE_COLUMNS).to_parquet(root / "isone_da_rt_x.parquet", index=False)
    no_rows, short, over = _day_gaps(root, "ISONE", first, last)
    assert no_rows == [(c, dt.date(2026, 6, 21)), (c, dt.date(2026, 6, 23))]       # all infinite; absent
    assert short == {(a, dt.date(2026, 6, 22)): (23, 24, 0, 0), (b, dt.date(2026, 6, 22)): (23, 24, 1, 0)}
    assert over == {}
    assert _missing_days(root, "ISONE", first, last) == sorted(no_rows + list(short) + list(over))
    assert d not in {n for n, _ in no_rows} | {n for n, _ in short}


@pytest.mark.parametrize("key, spring, fall, other", [("CAISO", 23, 25, 24), ("fmm", 23, 25, 24), ("MISO", 24, 24, 24),
                                                      ("NYISO", 23, 23, 24), ("ISONE", 23, 23, 24),
                                                      ("nyiso_rtint", 23, 23, 24)])
def test_the_hours_a_day_must_hold_follow_the_clock_and_the_known_fall_back_conventions(key, spring, fall, other):
    from es.sealed_pull import _expected_day_hours
    # the clocks went forward (not in MISO: fixed EST) and back (25 hours; 23 where the repeated hour is dropped)
    assert _expected_day_hours(key, dt.date(2026, 3, 8)) == spring
    assert _expected_day_hours(key, dt.date(2026, 11, 1)) == fall
    assert _expected_day_hours(key, dt.date(2026, 6, 22)) == other


def test_a_fall_back_day_of_nyiso_and_isone_holds_23_hours_and_of_caiso_25(tmp_path):
    """The conventions of the development files (usable rows per node on 2 November 2025: 23, 23 and 25)."""
    from es.sealed_pull import _day_gaps
    for market, hours in (("NYISO", 23), ("ISONE", 23), ("CAISO", 25)):
        root = tmp_path / market
        root.mkdir()
        nodes = _nodes_by_market()[market]
        rows = _full_price_rows(market, nodes, "2025-11-02", "2025-11-02")
        assert len(rows) == hours * len(nodes)                                # what a downloader keeps
        pd.DataFrame(rows, columns=PRICE_COLUMNS).to_parquet(root / f"{market.lower()}_da_rt_x.parquet", index=False)
        assert _day_gaps(root, market, "2025-11-02", "2025-11-02") == ([], {}, {})
        short = [r for r in rows if r[1] != nodes[0] or r[3] != rows[0][3]]   # the first node loses its first hour
        pd.DataFrame(short, columns=PRICE_COLUMNS).to_parquet(root / f"{market.lower()}_da_rt_x.parquet", index=False)
        assert _day_gaps(root, market, "2025-11-02", "2025-11-02") == (
            [], {(nodes[0], dt.date(2025, 11, 2)): (hours - 1, hours, 0, 0)}, {})


@pytest.mark.parametrize("market, node", [("NYISO", "N.Y.C."), ("ISONE", ".Z.MAINE"), ("MISO", "TEXAS.HUB")])
@pytest.mark.parametrize("how", ["a NaN real-time price", "an infinite day-ahead price",
                                 "a row the downloader dropped"])
def test_one_bad_hour_in_an_otherwise_complete_day_is_reported_like_a_missing_day(
        registered, monkeypatch, market, node, how):
    """The July downloaders' fakes write every hour of every day; one hour of one day is bad."""
    from es.sealed_pull import MissingDaysError, pull_sealed, verify_manifest
    write, tmp = registered
    proto, reg, sha = write()
    day, key = "2026-06-22", (market, node, "2026-06-22", 12)
    dropped = how == "a row the downloader dropped"
    _use_default_builders(monkeypatch, [], **(dict(drop={key}) if dropped else
                                              dict(prices={key: (1.0, NAN) if "NaN" in how else (INF, 1.0)})))
    kw = dict(data_root=tmp, reference_dirs=[])
    bad = 0 if dropped else 1
    note = "" if dropped else ", 1 non-finite price(s)"
    gap = (rf"{market}: fewer usable hours than the day has on 1 \(node, day\) pair\(s\): "
           rf"{re.escape(node)}: {day} \(23 of 24 hours{re.escape(note)}\)")
    with pytest.raises(MissingDaysError, match=gap) as err:
        pull_sealed("A", sha, **kw)
    assert "no rows for" not in str(err.value)
    assert "only if the ISO never published it with finite prices" in str(err.value)
    root = tmp / "pull_unit"
    assert json.loads((root / ".state.json").read_text())["done"] == list(STEPS[:STEPS.index(market)])    # not recorded
    logged = (root / "PULL_LOG.txt").read_text()
    assert f"step {market}: 1 (node, day) pair(s) short of their hours: {node}: {day} (23 of 24 hours{note})" in logged
    assert f"step {market}: every node-map node has the hours" not in logged     # the step was not logged as complete
    with pytest.raises(MissingDaysError):                                       # another market's key excuses nothing
        pull_sealed("A", sha, resume=True, allow_missing_days={"CAISO": [day]}, **kw)
    root, _ = pull_sealed("A", sha, resume=True, allow_missing_days={market: [day]}, **kw)
    log = (root / "PULL_LOG.txt").read_text()
    excused = "1 (node, day) pair(s) short of their hours, allowed by allow_missing_days"
    assert f"step {market}: {excused}: {node}: {day} (23 of 24 hours{note})" in log
    cov = pd.read_csv(root / "COVERAGE.csv")
    price = cov[cov["series"] == "price"]
    row = price[price["node"] == node].iloc[0]
    assert (row["nonfinite_rows"], row["hours"]) == (bad, 47)                     # two deployment days, one hour lost
    assert price["nonfinite_rows"].sum() == bad
    assert cov[cov["series"] != "price"]["nonfinite_rows"].sum() == 0
    if bad:
        assert f"(series market node rows): price {market} {node} 1" in log         # the record says how many were left
    verify_manifest(root)


def test_a_day_with_no_finite_price_at_all_is_reported_as_without_rows(registered, monkeypatch):
    from es.sealed_pull import MissingDaysError, pull_sealed
    write, tmp = registered
    proto, reg, sha = write()
    _use_default_builders(monkeypatch, [], prices={("ISONE", ".Z.MAINE", "2026-06-22", h): (NAN, NAN)
                                                   for h in range(24)})
    gap = r"ISONE: no rows for 1 \(node, day\) pair\(s\): \.Z\.MAINE: 2026-06-22"
    with pytest.raises(MissingDaysError, match=gap) as err:
        pull_sealed("A", sha, data_root=tmp, reference_dirs=[])
    assert "fewer usable hours" not in str(err.value)
    assert "step ISONE: 1 (node, day) pair(s) without rows: .Z.MAINE: 2026-06-22" in (
        tmp / "pull_unit" / "PULL_LOG.txt").read_text()


def _isone_csv(day, bad=None):
    """A synthetic ISO-NE daily report in the real format: one row per node and hour ending; `bad` (node, hour ending)
    has an LMP that is not a number."""
    import v2.ingest_isone as isone
    header = ["Date", "Hour Ending", "Location ID", "Location Name", "Location Type", "LMP", "Energy", "Congestion",
              "Loss"]
    lines = ['"C","synthetic ISO-NE report"', '"H",' + ",".join(f'"{name}"' for name in header)]
    for node in isone.NODES:
        for he in range(1, 25):
            lmp = "N/A" if (node, he) == bad else "25.00"
            lines.append(f'"D","{day:%m/%d/%Y}","{he:02d}","4000","{node}","Hub","{lmp}","24.0","0.5","0.5"')
    return ("\n".join(lines) + "\n").encode()


def _miso_csv(bad=None):
    """A synthetic MISO wide report in the real format; the LMP of `bad` (node, hour ending) is empty."""
    import v2.ingest_miso as miso
    head = "Node,Type,Value," + ",".join(f"HE {i}" for i in range(1, 25))
    rows = [f"{node},Hub,LMP," + ",".join("" if (node, i) == bad else "25.0" for i in range(1, 25))
            for node in sorted(miso.NODES)]
    return ("synthetic MISO report\nAll Hours-Ending are Eastern Standard Time (EST)\n\n" + head + "\n"
            + "\n".join(rows) + "\n").encode()


@pytest.mark.parametrize("market", ["ISONE", "MISO"])
def test_the_real_isone_and_miso_parsers_drop_a_row_they_cannot_read_and_the_day_check_reports_the_day(
        registered, monkeypatch, market):
    """The reviewer's case: one real-time LMP is not a number. The v2 downloader drops the row, so no non-finite price
    ever reaches the table; the day is short of its hours. These runs use the real parsers, fed synthetic reports."""
    import v2.ingest_isone
    import v2.ingest_miso

    import es.sealed_pull as sp
    write, tmp = registered
    proto, reg, sha = write()
    _use_default_builders(monkeypatch, [])
    module, node = (v2.ingest_isone, ".Z.MAINE") if market == "ISONE" else (v2.ingest_miso, "TEXAS.HUB")
    monkeypatch.setattr(module, "build", sp.REAL_DOWNLOADERS[market])               # the real downloader, fake reports

    def urlread(url, *args, **kwargs):
        name = url.rsplit("/", 1)[1]
        day = pd.Timestamp(name.rsplit("_", 1)[-1][:8] if market == "ISONE" else name[:8])
        real_time = "rt" in name
        bad_day = real_time and day == pd.Timestamp("2026-06-22")
        if market == "ISONE":
            return _isone_csv(day, (node, 14) if bad_day else None)
        return _miso_csv((node, 14) if bad_day else None)

    monkeypatch.setattr(module, "_urlread", urlread)
    kw = dict(data_root=tmp, reference_dirs=[])
    gap = (rf"{market}: fewer usable hours than the day has on 1 \(node, day\) pair\(s\): "
           rf"{re.escape(node)}: 2026-06-22 \(23 of 24 hours\)")
    with pytest.raises(sp.MissingDaysError, match=gap):
        sp.pull_sealed("A", sha, **kw)
    root, _ = sp.pull_sealed("A", sha, resume=True, allow_missing_days={market: ["2026-06-22"]}, **kw)
    cov = pd.read_csv(root / "COVERAGE.csv")
    row = cov[(cov["series"] == "price") & (cov["node"] == node)].iloc[0]
    assert (row["hours"], row["nonfinite_rows"]) == (47, 0)                  # one hour short, nothing non-finite in it


@pytest.mark.parametrize("key, step, market, node, day",
                         [("fmm", "FMM", "CAISO", SDGE, "2026-06-22"),
                          ("nyiso_rtint", "NYISO_INT", "NYISO", "N.Y.C.", "2026-06-23")])
@pytest.mark.parametrize("value", [NAN, INF])
def test_a_non_finite_settlement_price_is_reported_and_excused_by_series(
        registered, monkeypatch, key, step, market, node, day, value):
    import es.sealed_pull as sp
    write, tmp = registered
    proto, reg, sha = write()
    _use_default_builders(monkeypatch, [])
    make = _fmm_table_builder if key == "fmm" else _nyiso_table_builder
    builders = {**sp._default_builders(), step: make(nonfinite={(node, day): value})}
    kw = dict(data_root=tmp, reference_dirs=[], builders=builders)
    gap = (rf"{key}: fewer usable hours than the day has on 1 \(node, day\) pair\(s\): {re.escape(node)}: {day} "
           r"\(23 of 24 hours, 1 non-finite price\(s\)\)")
    with pytest.raises(sp.MissingDaysError, match=gap):
        sp.pull_sealed("A", sha, **kw)
    assert json.loads((tmp / "pull_unit" / ".state.json").read_text())["done"] == list(STEPS[:STEPS.index(step)])
    with pytest.raises(sp.MissingDaysError):                                  # the market's key is not the series' key
        sp.pull_sealed("A", sha, resume=True, allow_missing_days={market: [day]}, **kw)
    root, _ = sp.pull_sealed("A", sha, resume=True, allow_missing_days={key: [day]}, **kw)
    excused = "1 (node, day) pair(s) short of their hours, allowed by allow_missing_days"
    log = (root / "PULL_LOG.txt").read_text()
    assert f"step {key}: {excused}: {node}: {day} (23 of 24 hours, 1 non-finite price(s))" in log
    assert f"(series market node rows): {key} {market} {node} 1" in log
    cov = pd.read_csv(root / "COVERAGE.csv")
    side = cov[cov["series"] == key]
    assert side[side["node"] == node]["nonfinite_rows"].tolist() == [1]
    assert side[side["node"] == node]["hours"].tolist() == [47]                  # 48 hours in the two days, one lost
    assert side["nonfinite_rows"].sum() == 1 and cov[cov["series"] != key]["nonfinite_rows"].sum() == 0


def test_an_hour_with_a_non_finite_five_minute_price_is_nan(tmp_path):
    """The hourly real-time price is the mean of twelve 5-minute prices; pandas would skip a NaN and average eleven."""
    from es.sealed_pull import _hourly_rt_mean
    stamps = pd.date_range(T("2026-06-01 07:00"), periods=36, freq="5min")           # three hours
    lmp = [10.0] * 36
    lmp[5], lmp[14], lmp[27] = NAN, INF, -INF                                            # one bad price in each hour
    frame = pd.DataFrame({"node": "A", "timestamp_utc": stamps, "lmp": lmp})
    clean = pd.DataFrame({"node": "B", "timestamp_utc": stamps, "lmp": [float(i) for i in range(36)]})
    out = _hourly_rt_mean(pd.concat([frame, clean], ignore_index=True)).set_index(["node", "timestamp_utc"])["lmp"]
    assert list(_hourly_rt_mean(frame).columns) == ["node", "timestamp_utc", "lmp"]
    assert out["A"].isna().all() and len(out["A"]) == 3
    assert out["B"].tolist() == pytest.approx([5.5, 17.5, 29.5])                          # the mean of twelve prices


def test_an_hour_with_a_non_finite_fifteen_minute_price_has_a_nan_price_and_fewer_than_four_intervals():
    from es.sealed_pull import _hourly_fmm_mean
    stamps = pd.date_range(T("2026-06-01 07:00"), periods=16, freq="15min")           # four hours
    price = [100.0 + i for i in range(16)]
    price[1], price[6], price[11] = NAN, INF, -INF                                   # hours 1, 2 and 3 each have one
    price[12:16] = [NAN] * 4                                                         # hour 4 has none at all
    out = _hourly_fmm_mean(pd.DataFrame({"node": "A", "timestamp_utc": stamps, "fmm15": price}))
    assert list(out.columns) == ["node", "timestamp_utc", "fmm_lmp", "n15"] and out["n15"].dtype.kind == "i"
    assert out["n15"].tolist() == [3, 3, 3, 0]                                           # counted over finite intervals
    assert out["fmm_lmp"].isna().all()
    good = pd.DataFrame({"node": "B", "timestamp_utc": stamps[:8], "fmm15": [float(i) for i in range(8)]})
    clean = _hourly_fmm_mean(good)
    assert clean["n15"].tolist() == [4, 4] and clean["fmm_lmp"].tolist() == pytest.approx([1.5, 5.5])


@pytest.mark.parametrize("kind", ["DAM", "RTM", "FMM", "NYISO"])
def test_a_response_with_a_non_finite_price_is_retried_and_never_cached(tmp_path, kind):
    from es.sealed_pull import NYISO_URL, caiso_exact, caiso_fmm_exact, nyiso_integrated_months
    cache, calls, answers = tmp_path / "raw", [], {"bad": 2}

    def poisoned(url):
        calls.append(url)
        s, e = (_ts(url, "startdatetime"), _ts(url, "enddatetime")) if "oasis" in url else (None, None)
        if kind == "NYISO":
            bad = "nan" if answers["bad"] else "1.0"
            answers["bad"] = max(answers["bad"] - 1, 0)
            return _nyiso_zip({"20260630": [("06/30/2026 23:00", "N.Y.C.", bad)]})
        want = {"DAM": "queryname=PRC_LMP&", "RTM": "queryname=PRC_INTVL_LMP&", "FMM": "PRC_RTPD_LMP"}[kind]
        if want in url and _node(url) == SDGE and answers["bad"]:
            answers["bad"] -= 1
            step = {"DAM": 60, "RTM": 5, "FMM": 15}[kind]
            bad_price = lambda t: NAN if t == s + pd.Timedelta(hours=3) else 10.0
            return (_fmm_zip if kind == "FMM" else (lambda n, g: _oasis_zip(n, g)))(SDGE, _grid(s, e, step, bad_price))
        return _good_oasis(url)

    if kind == "NYISO":
        nyiso_integrated_months("2026-06", "2026-06", tmp_path / "x", fetch=poisoned, cache_dir=cache, **NO_SLEEP)
        target = NYISO_URL.format(ym="202606")
    elif kind == "FMM":
        caiso_fmm_exact("2026-06-21", "2026-06-21", tmp_path / "x", fetch=poisoned, cache_dir=cache, **NO_SLEEP)
        target = next(u for u in calls if _node(u) == SDGE)
    else:
        caiso_exact("2026-06-21", "2026-06-21", tmp_path / "x", fetch=poisoned, cache_dir=cache, **NO_SLEEP)
        want = "queryname=PRC_LMP&" if kind == "DAM" else "queryname=PRC_INTVL_LMP&"
        target = next(u for u in calls if want in u and _node(u) == SDGE)
    assert calls.count(target) == 3                                      # two poisoned answers, then a good one
    assert len(list(cache.glob("*.bin"))) == len({u for u in calls})     # only good answers were written, once per URL


@pytest.mark.parametrize("kind", ["DAM", "NYISO"])
def test_a_response_that_is_never_finite_raises_with_the_count_and_caches_nothing(tmp_path, kind):
    from es.sealed_pull import caiso_exact, nyiso_integrated_months
    cache, calls = tmp_path / "raw", []

    def fetch(url):
        calls.append(url)
        if kind == "NYISO":
            return _nyiso_zip({"20260630": [("06/30/2026 23:00", "N.Y.C.", "inf")]})
        s, e = _ts(url, "startdatetime"), _ts(url, "enddatetime")
        return _oasis_zip(_node(url), _grid(s, e, 60, lambda t: INF))

    with pytest.raises(RuntimeError, match="after 8 attempts") as err:
        if kind == "NYISO":
            nyiso_integrated_months("2026-06", "2026-06", tmp_path / "x", fetch=fetch, cache_dir=cache, **NO_SLEEP)
        else:
            caiso_exact("2026-06-21", "2026-06-21", tmp_path / "x", fetch=fetch, cache_dir=cache, **NO_SLEEP)
    assert len(calls) == 8 and "are not finite numbers" in str(err.value)
    assert not cache.exists() or list(cache.glob("*.bin")) == []


def test_a_cached_response_with_a_non_finite_price_is_fetched_again(tmp_path):
    from es.sealed_pull import OASIS_XML, _fmt, caiso_exact, local_day_bounds_utc
    cache, calls = tmp_path / "raw", []
    cache.mkdir()
    s, e = local_day_bounds_utc("2026-06-21", "2026-06-21", "America/Los_Angeles")
    url = OASIS_XML.format(qn="PRC_LMP", mri="DAM", node=SDGE, s=_fmt(s), e=_fmt(e))
    poisoned = _oasis_zip(SDGE, _grid(s, e, 60, lambda t: NAN if t == s else 10.0))
    path = cache / (hashlib.sha256(url.encode()).hexdigest() + ".bin")
    path.write_bytes(poisoned)                                            # what the old code cached for good

    def fetch(u):
        calls.append(u)
        return _good_oasis(u)

    out = pd.read_parquet(caiso_exact("2026-06-21", "2026-06-21", tmp_path / "x", fetch=fetch, cache_dir=cache,
                                      **NO_SLEEP))
    assert calls.count(url) == 1 and out["da_lmp"].notna().all()         # the poisoned entry was a miss; it is replaced
    assert path.read_bytes() != poisoned


def test_a_pull_whose_caiso_answer_is_never_finite_stops_at_the_caiso_step_and_a_resume_asks_again(
        registered, monkeypatch):
    """The reviewer's F3: the NaN answer used to be cached, so the failure came back on every resume with no request."""
    import es.sealed_pull as sp
    write, tmp = registered
    proto, reg, sha = write()
    state, count = {"poisoned": True}, collections.Counter()

    def one_nan(url):
        if "queryname=PRC_INTVL_LMP&" in url and _node(url) == "TH_SP15_GEN-APND" and state["poisoned"]:
            count[url] += 1
            s, e = _ts(url, "startdatetime"), _ts(url, "enddatetime")
            return _oasis_zip(_node(url), _grid(s, e, 5, lambda t: NAN if t == s + pd.Timedelta(hours=30) else 10.0))
        return None

    _use_default_builders(monkeypatch, [], fetch_override=one_nan)
    kw = dict(data_root=tmp, reference_dirs=[])
    with pytest.raises(RuntimeError, match="are not finite numbers"):
        sp.pull_sealed("A", sha, **kw)
    assert sum(count.values()) == 8
    state["poisoned"] = False                                              # OASIS answers properly this time
    root, _ = sp.pull_sealed("A", sha, resume=True, **kw)
    assert sum(count.values()) == 8 and (root / "MANIFEST.sha256").exists()


@pytest.mark.parametrize("key, step, drop", [("fmm", "FMM", "n15"), ("fmm", "FMM", "fmm_lmp"),
                                             ("nyiso_rtint", "NYISO_INT", "rt_int_lmp")])
def test_a_settlement_table_that_lacks_a_column_stops_the_day_check(registered, monkeypatch, key, step, drop):
    import es.sealed_pull as sp
    write, tmp = registered
    proto, reg, sha = write()
    _use_default_builders(monkeypatch, [])
    inner = (_fmm_table_builder if key == "fmm" else _nyiso_table_builder)()

    def build(start, end, root):
        inner(start, end, root)
        table = next((root / key).glob("*.parquet"))
        pd.read_parquet(table).drop(columns=[drop]).to_parquet(table, index=False)

    builders = {**sp._default_builders(), step: build}
    with pytest.raises(RuntimeError, match=rf"the rows of {key} have no \['{drop}'\] column"):
        sp.pull_sealed("A", sha, data_root=tmp, reference_dirs=[], builders=builders)
    assert json.loads((tmp / "pull_unit" / ".state.json").read_text())["done"] == list(STEPS[:STEPS.index(step)])


def test_a_price_file_without_rt_lmp_stops_the_day_check_and_shows_as_non_finite_in_coverage(registered, monkeypatch):
    import es.sealed_pull as sp
    write, tmp = registered
    proto, reg, sha = write()
    _use_default_builders(monkeypatch, [])

    def no_rt(start, end, root):
        rows = [("ISONE", n, "hub", T(f"{d} 18:00"), 1.0) for n in _nodes_by_market()["ISONE"]
                for d in ("2026-06-21", "2026-06-22", "2026-06-23")]
        pd.DataFrame(rows, columns=["market", "node", "node_type", "timestamp_utc", "da_lmp"]).to_parquet(
            root / "isone_da_rt_x.parquet", index=False)

    builders = {**sp._default_builders(), "ISONE": no_rt}
    with pytest.raises(RuntimeError, match=r"the rows of ISONE have no \['rt_lmp'\] column"):
        sp.pull_sealed("A", sha, data_root=tmp, reference_dirs=[], builders=builders)
    rehearsal = {**_complete_builders([]), "ISONE": no_rt}                           # injected builders only
    (tmp / "rehearsal").mkdir()
    root, _ = sp.pull_sealed("A", sha, data_root=tmp / "rehearsal", reference_dirs=[], builders=rehearsal,
                             check_days=False)
    cov = pd.read_csv(root / "COVERAGE.csv")
    isone = cov[(cov["series"] == "price") & (cov["market"] == "ISONE")]
    assert (isone["hours"] == 0).all() and (isone["nonfinite_rows"] == 2).all()   # no price, no hour; 22 and 23 June


# ----------------------------------------------------------------------------- I1 for the settlement tables
@pytest.mark.parametrize("series, market, utc_offset_hours", [("fmm", "CAISO", 7), ("nyiso_rtint", "NYISO", 4)])
def test_the_settlement_table_readers_never_load_a_row_at_or_after_the_seal(
        tmp_path, monkeypatch, series, market, utc_offset_hours):
    import pyarrow.parquet as pq

    from es.sealed_pull import _missing_days, coverage_manifest, overlap_check
    seal = T("2026-07-01") + pd.Timedelta(hours=utc_offset_hours)
    node, tail = ("TH_SP15_GEN-APND", (4,)) if series == "fmm" else ("N.Y.C.", ())

    def stamp(offset_hours):
        return (seal + pd.Timedelta(hours=offset_hours)).strftime("%Y-%m-%d %H:%M")

    root = tmp_path / "pull"
    _write_side(root / series, series, [(node, stamp(-1), 1.0, *tail), (node, stamp(0), 999.0, *tail),
                                        (node, stamp(5), 999.0, *tail)])
    real, read = pq.read_table, []

    def spy(*args, **kwargs):
        table = real(*args, **kwargs)
        read.append(table)
        return table

    monkeypatch.setattr(pq, "read_table", spy)
    calls = {"coverage_manifest": lambda: coverage_manifest(root, "2026-06-01", "2026-06-30"),
             "_missing_days": lambda: _missing_days(root, series, "2026-06-01", "2026-07-02"),
             "overlap_check": lambda: overlap_check(root, [], "2026-06-21", "2026-07-02",
                                                    side_references={series: [root / series]})}
    for label, call in calls.items():
        read.clear()
        call()
        assert read, f"{label} did not read through a predicate"
        for table in read:
            loaded = pd.to_datetime(table.column("timestamp_utc").to_pandas(), utc=True)
            assert len(loaded) == 1 and loaded.max() < seal, label


# ----------------------------------------------------------------------------- _fetch_parse, called directly
def test_a_cached_response_that_raises_in_the_parser_or_the_validator_is_raised_not_fetched_again(tmp_path):
    from es.guard import SealedDataError
    from es.sealed_pull import _fetch_parse
    cache, calls, url = tmp_path / "raw", [], "https://example.invalid/x"
    cache.mkdir()
    (cache / (hashlib.sha256(url.encode()).hexdigest() + ".bin")).write_bytes(b"cached")

    def fetch(u):
        calls.append(u)
        return b"fresh"

    def blocked(blob):
        raise AssertionError("network access attempted in a unit test")

    with pytest.raises(AssertionError, match="network access attempted"):
        _fetch_parse(url, blocked, "x", fetch, lambda s: None, cache)

    def sealed(part):
        raise SealedDataError("a row of the sealed window")

    with pytest.raises(SealedDataError):
        _fetch_parse(url, lambda blob: pd.DataFrame({"a": [1]}), "x", fetch, lambda s: None, cache, validate=sealed)
    assert calls == []


def test_a_cached_response_with_a_sealed_row_raises_in_a_development_pull(tmp_path):
    from es.guard import SealedDataError
    from es.sealed_pull import OASIS_XML, _fmt, caiso_exact, local_day_bounds_utc
    cache, calls = tmp_path / "raw", []
    cache.mkdir()
    s, e = local_day_bounds_utc("2026-06-30", "2026-06-30", "America/Los_Angeles")
    url = OASIS_XML.format(qn="PRC_LMP", mri="DAM", node="TH_SP15_GEN-APND", s=_fmt(s), e=_fmt(e))
    grid = _grid(s, e, 60, lambda t: 10.0, inclusive_end=True)                    # one hour too many: 1 July 00:00 PDT
    (cache / (hashlib.sha256(url.encode()).hexdigest() + ".bin")).write_bytes(_oasis_zip("TH_SP15_GEN-APND", grid))

    def fetch(u):
        calls.append(u)
        return _good_oasis(u)

    with pytest.raises(SealedDataError, match="sealed"):
        caiso_exact("2026-06-30", "2026-06-30", tmp_path / "x", fetch=fetch, cache_dir=cache, **NO_SLEEP)
    assert calls == []                                                                   # raised, not fetched again


# ----------------------------------------------------------------------------- adoption records the last finish line
def test_a_pull_adopted_after_two_finish_lines_records_the_last_one(registered, monkeypatch):
    import es.sealed_pull as sp
    write, tmp = registered
    proto, reg, sha = write()
    real_manifest, real_replace = sp.write_manifest, os.replace

    def no_manifest(root):
        raise KeyboardInterrupt("killed before the manifest was written")

    def no_record(src, dst):
        if Path(dst).name == "PULLS.json":
            raise KeyboardInterrupt("killed while PULLS.json was being written")
        return real_replace(src, dst)

    monkeypatch.setattr(sp, "write_manifest", no_manifest)
    with pytest.raises(KeyboardInterrupt):
        _a_pull(tmp, sha)                                                        # the first finish line, no manifest
    monkeypatch.setattr(sp, "write_manifest", real_manifest)
    monkeypatch.setattr(os, "replace", no_record)
    with pytest.raises(KeyboardInterrupt):
        _a_pull(tmp, sha, resume=True)                                           # the second finish line, a manifest
    monkeypatch.setattr(os, "replace", real_replace)
    root = tmp / "sealed_2026Q3"
    finished = [ln.split(": ", 1)[1] for ln in (root / "PULL_LOG.txt").read_text().splitlines()
                if ln.startswith("pull_finished_utc: ")]
    assert len(finished) == 2 and finished[0] != finished[1] and _pulls(tmp) == {}
    _a_pull(tmp, sha, resume=True)                                               # adopted
    assert _pulls(tmp)["A"]["finished_utc"] == finished[1]


# ----------------------------------------------------------------------------- the descriptor of a directory flush
def test_fsync_dir_closes_the_descriptor_it_opens(tmp_path, monkeypatch):
    import es.sealed_pull as sp
    opened, closed = [], []
    real_open, real_close = os.open, os.close

    def spy_open(*args, **kwargs):
        fd = real_open(*args, **kwargs)
        opened.append(fd)
        return fd

    def spy_close(fd):
        closed.append(fd)
        real_close(fd)

    with monkeypatch.context() as patched:
        patched.setattr(os, "open", spy_open)
        patched.setattr(os, "close", spy_close)
        sp._fsync_dir(tmp_path)
    assert len(opened) == 1 and closed == opened


def test_fsync_dir_closes_the_descriptor_when_the_flush_fails(tmp_path, monkeypatch):
    import es.sealed_pull as sp
    opened, closed = [], []
    real_open, real_close = os.open, os.close
    monkeypatch.setattr(sp, "_fsync", lambda fd: (_ for _ in ()).throw(OSError("the disk went away")))

    def spy_open(*args, **kwargs):
        opened.append(real_open(*args, **kwargs))
        return opened[-1]

    def spy_close(fd):
        closed.append(fd)
        real_close(fd)

    with monkeypatch.context() as patched:
        patched.setattr(os, "open", spy_open)
        patched.setattr(os, "close", spy_close)
        with pytest.raises(OSError, match="the disk went away"):
            sp._fsync_dir(tmp_path)
    assert len(opened) == 1 and closed == opened


# ----------------------------------------------------------------------------- settlement tables: duplicates, extras
def test_a_row_in_two_files_of_a_settlement_table_counts_once(tmp_path):
    import shutil

    from es.sealed_pull import _day_gaps, coverage_manifest
    root = tmp_path / "pull"
    _write_side(root / "fmm", "fmm", [(SDGE, "2026-06-22 18:00", 1.0, 4)])
    shutil.copy(root / "fmm" / "fmm_x.parquet", root / "fmm" / "fmm_y.parquet")       # the same hour in a second file
    cov = coverage_manifest(root, "2026-06-22", "2026-06-23")
    assert cov[(cov["series"] == "fmm") & (cov["node"] == SDGE)]["hours"].tolist() == [1]
    no_rows, short, over = _day_gaps(root, "fmm", "2026-06-22", "2026-06-22")
    assert short[(SDGE, dt.date(2026, 6, 22))] == (1, 24, 0, 0) and over == {}      # the hour counts once, not twice


def test_nodes_of_the_settlement_tables_outside_the_node_map_are_named_and_left_out_of_coverage(tmp_path):
    from es.sealed_pull import coverage_manifest
    root = tmp_path / "pull"
    _write_side(root / "fmm", "fmm", [(SDGE, "2026-06-22 18:00", 1.0, 4),
                                      ("EXTRA_NODE", "2026-06-22 18:00", 1.0, 4)])
    _write_side(root / "nyiso_rtint", "nyiso_rtint", [("N.Y.C.", "2026-06-22 18:00", 5.0),
                                                      ("OTHER_ZONE", "2026-06-22 18:00", 5.0)])
    cov = coverage_manifest(root, "2026-06-22", "2026-06-23")
    assert not {"EXTRA_NODE", "OTHER_ZONE"} & set(cov["node"])
    assert cov.attrs["side_not_in_node_map"] == [("fmm", "EXTRA_NODE"), ("nyiso_rtint", "OTHER_ZONE")]
    assert cov.attrs["not_in_node_map"] == []


def test_a_pull_logs_the_settlement_table_nodes_that_are_not_in_the_node_map(registered, monkeypatch):
    from es.sealed_pull import pull_sealed
    write, tmp = registered
    proto, reg, sha = write()
    _use_default_builders(monkeypatch, [], nyiso_extra=(("06/22/2026 10:00", "OTHER_ZONE", 7.0),))
    root, _ = pull_sealed("A", sha, data_root=tmp, reference_dirs=[])
    log = (root / "PULL_LOG.txt").read_text()
    assert ("in the settlement tables but not in the node map, so not in COVERAGE.csv: "
            "[('nyiso_rtint', 'OTHER_ZONE')]") in log
    assert "OTHER_ZONE" not in set(pd.read_csv(root / "COVERAGE.csv")["node"])


# ----------------------------------------------------------------------------- the last survivors
def test_no_pause_follows_a_cache_hit_in_the_fmm_and_nyiso_pullers_either(tmp_path):
    from es.sealed_pull import caiso_fmm_exact, nyiso_integrated_months
    fetch, calls = _memoised_oasis()
    month = _nyiso_zip({"20260621": [("06/21/2026 10:00", "N.Y.C.", 1.0)]})        # one object: zip headers differ
    cache, pauses = tmp_path / "raw", []
    for where in ("a", "b"):
        caiso_fmm_exact("2026-06-21", "2026-06-21", tmp_path / where / "fmm", fetch=fetch, cache_dir=cache, pause=5.5,
                        sleep=pauses.append)
        nyiso_integrated_months("2026-06", "2026-06", tmp_path / where / "nyiso_rtint", fetch=lambda url: month,
                                cache_dir=cache, pause=2.0, sleep=pauses.append)
        if where == "a":
            assert sorted(pauses) == [2.0] + [5.5] * 6                  # a pause after each of the seven requests
            pauses.clear()
    assert pauses == [] and len(calls) == 6                              # the second round came from the cache alone


def test_a_record_of_completed_pulls_that_is_not_an_object_stops_every_pull(registered):
    write, tmp = registered
    proto, reg, sha = write()
    (tmp / "docs_sealed" / "PULLS.json").write_text("[1, 2]")
    calls = []
    with pytest.raises(RuntimeError, match="must hold a JSON object keyed by test id"):
        _a_pull(tmp, sha, calls)
    assert calls == [] and not (tmp / "sealed_2026Q3").exists()


def test_a_price_row_in_two_files_counts_once(tmp_path):
    import shutil

    from es.sealed_pull import coverage_manifest
    root = tmp_path / "pull"
    root.mkdir()
    pd.DataFrame({"market": "CAISO", "node": "TH_SP15_GEN-APND", "node_type": "hub",
                  "timestamp_utc": [T("2026-06-22 18:00")], "da_lmp": 1.0, "rt_lmp": 1.0}).to_parquet(
        root / "caiso_da_rt_a.parquet", index=False)
    shutil.copy(root / "caiso_da_rt_a.parquet", root / "caiso_da_rt_b.parquet")                # the same hour twice
    cov = coverage_manifest(root, "2026-06-22", "2026-06-23")
    assert cov[(cov["series"] == "price") & (cov["node"] == "TH_SP15_GEN-APND")]["hours"].tolist() == [1]


def test_overlap_check_flags_a_day_ahead_difference_alone_and_includes_the_first_day(tmp_path):
    from es.sealed_pull import overlap_check
    ts = pd.to_datetime(["2026-06-20 18:00", "2026-06-21 18:00", "2026-06-22 18:00"], utc=True)  # before, on, after
    base = pd.DataFrame({"market": "ISONE", "node": "X", "node_type": "hub", "timestamp_utc": ts,
                         "da_lmp": [1.0, 2.0, 3.0], "rt_lmp": [1.0, 2.0, 3.0]})
    (tmp_path / "ref").mkdir()
    (tmp_path / "new").mkdir()
    base.to_parquet(tmp_path / "ref" / "isone_da_rt_ref.parquet", index=False)
    new = base.copy()
    new.loc[1, "da_lmp"] = 2.75                     # 21 June, the first day: only the day-ahead price differs
    new.loc[0, "da_lmp"] = 99.0                     # 20 June is before the overlap days: not compared
    new.to_parquet(tmp_path / "new" / "isone_da_rt_new.parquet", index=False)
    out = overlap_check(tmp_path / "new", [tmp_path / "ref"], "2026-06-21", "2026-07-01").set_index("market")
    assert out.loc["ISONE", "matched"] == 2 and out.loc["ISONE", "rows_differing"] == 1
    assert out.loc["ISONE", "max_abs_diff_da"] == pytest.approx(0.75) and out.loc["ISONE", "max_abs_diff_rt"] == 0.0


def test_the_manifest_lists_its_files_in_sorted_order_whatever_order_the_file_system_gives(tmp_path, monkeypatch):
    from es.sealed_pull import write_manifest
    root = tmp_path / "r"
    (root / "z").mkdir(parents=True)
    for name in ("z/b.bin", "m.bin", "z/a.bin", "a.bin"):
        (root / name).write_bytes(name.encode())
    real = pathlib.Path.rglob
    monkeypatch.setattr(pathlib.Path, "rglob", lambda self, pattern: iter(sorted(real(self, pattern), reverse=True)))
    paths = [ln.split("  ", 1)[1] for ln in write_manifest(root).read_text().splitlines()]
    assert paths == ["a.bin", "m.bin", "z/a.bin", "z/b.bin"]


# =====================================================================================================
# Fix round 2 (controller Ruling 106, from the Opus re-review of fix round 1)
# =====================================================================================================
# ----------------------------------------------------------------------------- N3: a pull is never started over
def _failed_first_attempt(registered, protocol="protocol v1\n", **record):
    write, tmp = registered
    proto, reg, sha = write(protocol, reg_fields=record or None)
    builders = _failing_then_fine_builders()
    with pytest.raises(OSError):
        _a_pull(tmp, sha, builders=builders)
    return write, tmp, sha, builders, tmp / "sealed_2026Q3"


def test_a_root_whose_state_was_deleted_is_never_started_over_whatever_registration_resumes_it(registered):
    """The reviewer's N3: .state.json deleted, then a resume under another protocol and DOI used to complete, and
    PULLS.json recorded only the second attempt."""
    from es.sealed_pull import completed_pulls
    write, tmp, sha1, builders, root = _failed_first_attempt(registered)
    (root / ".state.json").unlink()
    log_before, files_before = (root / "PULL_LOG.txt").read_text(), sorted(p.name for p in root.rglob("*"))
    proto2, reg2, sha2 = write("protocol v2\n", reg_fields={"osf_doi": "10.17605/OSF.IO/QQ7RT"})
    calls = []
    with pytest.raises(FileExistsError, match=r"\.state\.json is missing, so the pull cannot be resumed") as err:
        _a_pull(tmp, sha2, calls, resume=True)
    assert "it is not started over" in str(err.value) and "move the whole folder aside" in str(err.value)
    assert "PULL_LOG.txt stays there as evidence" in str(err.value) and "PULLS.json is not touched" in str(err.value)
    assert calls == [] and completed_pulls() == {}
    assert (root / "PULL_LOG.txt").read_text() == log_before and sorted(p.name for p in root.rglob("*")) == files_before
    write("protocol v1\n")                                                  # even the registration it began under
    with pytest.raises(FileExistsError, match=r"\.state\.json is missing"):
        _a_pull(tmp, sha1, builders=builders, resume=True)
    with pytest.raises(FileExistsError, match="interrupted pull; call again with resume=True"):
        _a_pull(tmp, sha1, builders=builders)                               # and without resume it is the older refusal


@pytest.mark.parametrize("damage", ["{not json", "[1, 2]", '{"done": "all"}', ""])
def test_a_root_with_an_unreadable_state_is_refused_the_same_way(registered, damage):
    from es.sealed_pull import completed_pulls
    write, tmp, sha, builders, root = _failed_first_attempt(registered)
    (root / ".state.json").write_text(damage)
    with pytest.raises(FileExistsError, match=r"\.state\.json is unreadable, so the pull cannot be resumed"):
        _a_pull(tmp, sha, builders=builders, resume=True)
    assert completed_pulls() == {} and (root / ".state.json").read_text() == damage


def test_moving_the_root_aside_is_the_way_out_and_leaves_the_log_as_evidence(registered):
    from es.sealed_pull import completed_pulls
    write, tmp, sha, builders, root = _failed_first_attempt(registered)
    (root / ".state.json").unlink()
    evidence = (root / "PULL_LOG.txt").read_text()
    root.rename(tmp / "sealed_2026Q3_broken")
    fresh, _ = _a_pull(tmp, sha, builders=builders)                         # no resume needed: there is no root
    assert (fresh / "MANIFEST.sha256").exists() and completed_pulls()["A"]["root"] == str(fresh)
    assert (tmp / "sealed_2026Q3_broken" / "PULL_LOG.txt").read_text() == evidence


def test_a_root_holding_only_the_temp_file_of_a_state_that_was_never_written_starts_afresh(registered):
    write, tmp = registered
    proto, reg, sha = write()
    root = tmp / "sealed_2026Q3"
    root.mkdir()
    (root / ".state.json.tmp").write_text("{")                              # a kill while the first state was written
    with pytest.raises(FileExistsError, match="interrupted pull; call again with resume=True"):
        _a_pull(tmp, sha)
    calls = []
    done, _ = _a_pull(tmp, sha, calls, resume=True)
    assert len(calls) == 6 and (done / "MANIFEST.sha256").exists() and not (done / ".state.json.tmp").exists()


def test_an_empty_root_is_started_afresh_with_resume(registered):
    write, tmp = registered
    proto, reg, sha = write()
    (tmp / "sealed_2026Q3").mkdir()                                         # a kill between mkdir and the first state
    calls = []
    done, _ = _a_pull(tmp, sha, calls, resume=True)
    assert len(calls) == 6 and (done / "MANIFEST.sha256").exists()


def test_the_first_attempts_log_is_checked_as_well_as_its_state(registered):
    """A state rewritten to name the registration now verified does not carry a pull over to it."""
    from es.registration import RegistrationError
    from es.sealed_pull import completed_pulls
    write, tmp, sha1, builders, root = _failed_first_attempt(registered)
    proto2, reg2, sha2 = write("protocol v2\n", reg_fields={"osf_doi": "10.17605/OSF.IO/QQ7RT"})
    state = json.loads((root / ".state.json").read_text())
    state.update(protocol_sha256=sha2, osf_doi="10.17605/OSF.IO/QQ7RT", registration_identifier="10.17605/OSF.IO/QQ7RT",
                 registration_id="qq7rt")
    (root / ".state.json").write_text(json.dumps(state))
    with pytest.raises(RegistrationError, match=r"PULL_LOG\.txt was started under protocol"):
        _a_pull(tmp, sha2, builders=builders, resume=True)
    assert completed_pulls() == {}


def test_the_state_checks_the_registration_when_the_log_is_gone(registered):
    from es.registration import RegistrationError
    write, tmp, sha1, builders, root = _failed_first_attempt(registered)
    (root / "PULL_LOG.txt").unlink()
    proto2, reg2, sha2 = write("protocol v2\n")
    with pytest.raises(RegistrationError, match=r"\.state\.json was started under protocol"):
        _a_pull(tmp, sha2, builders=builders, resume=True)
    write("protocol v1\n")                                                  # under its own registration it goes on
    done, _ = _a_pull(tmp, sha1, builders=builders, resume=True)
    assert (done / "MANIFEST.sha256").exists()


def test_the_start_in_pulls_json_is_the_first_attempts_start_from_the_log(registered):
    from es.sealed_pull import completed_pulls
    write, tmp, sha, builders, root = _failed_first_attempt(registered)
    first = [ln.split(": ", 1)[1] for ln in (root / "PULL_LOG.txt").read_text().splitlines()
             if ln.startswith("pull_started_utc: ")][0]
    state = json.loads((root / ".state.json").read_text())
    state["started_utc"] = "1999-01-01T00:00:00+00:00"                      # the state is not what the record trusts
    (root / ".state.json").write_text(json.dumps(state))
    done, _ = _a_pull(tmp, sha, builders=builders, resume=True)
    assert completed_pulls()["A"]["started_utc"] == first
    assert (done / "PULL_LOG.txt").read_text().count("pull_started_utc: ") == 2


def test_a_third_attempt_goes_back_to_the_first_attempts_log_lines_not_the_seconds(registered):
    """The log holds a block of lines per attempt. The start that PULLS.json records is the first block's, and the
    registration may change its form between attempts (the address, then the address and the DOI, then the DOI)."""
    from es.sealed_pull import completed_pulls
    write, tmp = registered
    proto, reg, sha = write(reg_fields=ONLY_URL)                            # attempt 1: the OSF address alone (embargo)
    builders = _complete_builders([])
    good, failures = builders["CAISO"], {"left": 2}

    def flaky(start, end, root):
        if failures["left"]:
            failures["left"] -= 1
            raise OSError("nodename nor servname provided")
        return good(start, end, root)

    builders["CAISO"] = flaky
    with pytest.raises(OSError):
        _a_pull(tmp, sha, builders=builders)
    write(reg_fields={"osf_registration": "abcde"})                         # attempt 2: the DOI has joined the address
    with pytest.raises(OSError):
        _a_pull(tmp, sha, builders=builders, resume=True)
    write()                                                                 # attempt 3: the DOI alone
    root, _ = _a_pull(tmp, sha, builders=builders, resume=True)
    starts = [ln.split(": ", 1)[1] for ln in (root / "PULL_LOG.txt").read_text().splitlines()
              if ln.startswith("pull_started_utc: ")]
    assert len(starts) == 3 and len(set(starts)) == 3
    record = completed_pulls()["A"]
    assert record["started_utc"] == starts[0]                               # the first attempt's, not the second's
    assert (record["osf_doi"], record["osf_registration"], record["registration_id"]) == (DOI, None, "abcde")


def test_a_resume_after_a_kill_between_the_first_state_and_the_first_log_line_goes_on_from_the_state(registered):
    write, tmp, sha, builders, root = _failed_first_attempt(registered)
    (root / "PULL_LOG.txt").unlink()
    state_before = json.loads((root / ".state.json").read_text())
    done, _ = _a_pull(tmp, sha, builders=builders, resume=True)
    assert (done / "MANIFEST.sha256").exists() and state_before["done"] == ["NYISO", "ISONE", "MISO"]


# ----------------------------------------------------------------------------- N5: real downloaders, by identity
def test_check_days_false_is_refused_whenever_a_step_is_a_real_downloader(registered, monkeypatch):
    import es.sealed_pull as sp
    write, tmp = registered
    proto, reg, sha = write()
    kw = dict(data_root=tmp, reference_dirs=[], check_days=False)
    injected = _complete_builders([])
    with pytest.raises(ValueError, match=r"step\(s\) \['CAISO'\]"):                 # one real step among fakes
        sp.pull_sealed("A", sha, builders={**injected, "CAISO": sp.REAL_DOWNLOADERS["CAISO"]}, **kw)
    with pytest.raises(ValueError, match=r"step\(s\) \['NYISO'\]"):
        sp.pull_sealed("A", sha, builders={**injected, "NYISO": sp.REAL_DOWNLOADERS["NYISO"]}, **kw)
    with pytest.raises(ValueError, match=r"\['CAISO', 'FMM', 'ISONE', 'MISO', 'NYISO', 'NYISO_INT'\]"):
        sp.pull_sealed("A", sha, builders={**sp._default_builders()}, **kw)       # the explicit defaults, the old hole
    _use_default_builders(monkeypatch, [])                                           # fakes where the v2 builds were
    with pytest.raises(ValueError, match=r"\['CAISO', 'FMM', 'ISONE', 'MISO', 'NYISO', 'NYISO_INT'\]"):
        sp.pull_sealed("A", sha, builders={**sp._default_builders()}, **kw)       # still: they are what would run
    original = sp.REAL_DOWNLOADERS["NYISO"]                                         # a fake is in the module now
    with pytest.raises(ValueError, match=r"step\(s\) \['NYISO'\]"):                 # but the original is still refused
        sp.pull_sealed("A", sha, builders={**injected, "NYISO": original}, **kw)
    with pytest.raises(ValueError, match=r"step\(s\) \['MISO'\]"):                  # and the fake that is in place
        sp.pull_sealed("A", sha, builders={**injected, "MISO": sp._default_builders()["MISO"]}, **kw)
    assert not (tmp / "pull_unit").exists() and not (tmp / "sealed_2026Q3").exists()
    root, _ = sp.pull_sealed("A", sha, builders=injected, **kw)                      # all injected: allowed
    assert "check_days: False" in (root / "PULL_LOG.txt").read_text()


def test_the_default_builders_are_the_same_objects_each_time_so_that_identity_can_tell_them(monkeypatch):
    import es.sealed_pull as sp
    first, second = sp._default_builders(), sp._default_builders()
    assert all(first[name] is second[name] for name in sp.STEPS)
    assert all(first[name] is sp.REAL_DOWNLOADERS[name] for name in sp.STEPS)
    assert sp.REAL_DOWNLOADERS["CAISO"] is sp._download_caiso and sp.REAL_DOWNLOADERS["NYISO"] is sp.ingest_nyiso.build


# ----------------------------------------------------------------------------- N8 and N9
def test_a_relative_data_root_is_stored_resolved(registered, monkeypatch):
    from es.sealed_pull import completed_pulls
    write, tmp = registered
    proto, reg, sha = write()
    (tmp / "rel").mkdir()
    monkeypatch.chdir(tmp)
    root, _ = _a_pull(tmp, sha, data_root="rel")
    assert root.is_absolute() and root == (tmp / "rel" / "sealed_2026Q3").resolve()
    assert completed_pulls()["A"]["root"] == str(root) and Path(completed_pulls()["A"]["root"]).is_absolute()


def test_a_manifest_without_its_record_says_that_resume_adopts_it(registered, monkeypatch):
    write, tmp = registered
    proto, reg, sha = write()
    real = os.replace

    def killed(src, dst):
        if Path(dst).name == "PULLS.json":
            raise KeyboardInterrupt("killed while PULLS.json was being written")
        return real(src, dst)

    monkeypatch.setattr(os, "replace", killed)
    with pytest.raises(KeyboardInterrupt):
        _a_pull(tmp, sha)
    monkeypatch.setattr(os, "replace", real)
    with pytest.raises(FileExistsError, match=r"call again with resume=True, which verifies its manifest and adop"):
        _a_pull(tmp, sha)


# =====================================================================================================
# Fix round 3 (controller Ruling 111a): allow_incomplete_chunks, for a chunk that never comes complete
# =====================================================================================================
SPAN = ("2026-06-21", "2026-06-23")                       # the pull of DEV_WINDOW: its first day and its deployment end
CHUNK_TIMES = "20260621T07:00..20260624T07:00"            # its one CAISO chunk: 21 June 00:00 to 24 June 00:00 PDT
FMM_SDGE, FMM_PGAE = f"PRC_RTPD_LMP/{SDGE}/{CHUNK_TIMES}", f"PRC_RTPD_LMP/DLAP_PGAE-APND/{CHUNK_TIMES}"
DAM_SDGE, RTM_SDGE = f"PRC_LMP/{SDGE}/{CHUNK_TIMES}", f"PRC_INTVL_LMP/{SDGE}/{CHUNK_TIMES}"
SHORT_FMM = "short: 192 of 288 15-minute intervals missing, first 2026-06-22 07:00:00+00:00"   # only the first day came


def _key_of(url):
    """The key of an OASIS request, as PULL_LOG.txt prints it: the query, the node and the UTC interval."""
    return (f"{url.split('queryname=')[1].split('&')[0]}/{_node(url)}/"
            f"{_ts(url, 'startdatetime'):%Y%m%dT%H:%M}..{_ts(url, 'enddatetime'):%Y%m%dT%H:%M}")


def _short_answer(key, days=1):
    """fetch_override for _use_default_builders: the chunk `key` is answered with its first `days` days only."""
    return lambda url: _good_oasis(url, days=days) if "oasis" in url and _key_of(url) == key else None


def _pull(tmp, sha, **kw):
    import es.sealed_pull as sp
    return sp.pull_sealed("A", sha, data_root=tmp, reference_dirs=[], **kw)


def _log_of(tmp):
    return (tmp / "pull_unit" / "PULL_LOG.txt").read_text()


def _state_of(tmp):
    return json.loads((tmp / "pull_unit" / ".state.json").read_text())


def test_a_chunk_that_never_completes_is_accepted_only_by_two_explicit_decisions(registered, monkeypatch):
    import es.sealed_pull as sp
    write, tmp = registered
    proto, reg, sha = write()
    urls, down, short = [], {"nyiso": False}, _short_answer(FMM_SDGE)

    def answers(url):
        if down["nyiso"] and "mis.nyiso.com" in url:
            raise OSError("nodename nor\nservname provided")                       # a message of two lines
        return short(url)

    _use_default_builders(monkeypatch, urls, fetch_override=answers)
    sdge = lambda: [u for u in urls if "PRC_RTPD_LMP" in u and _node(u) == SDGE]

    # attempt 1: eight answers that the validator refuses. The step stops, and the log names the chunk.
    with pytest.raises(RuntimeError, match="no data after 8 attempts") as err:
        _pull(tmp, sha)
    assert isinstance(err.value, sp.ChunkError)
    assert (err.value.key, err.value.attempts, err.value.validator_failures) == (FMM_SDGE, 8, 8) and len(sdge()) == 8
    assert f"chunk_failed: fmm {FMM_SDGE} | 8 of 8 attempts failed the validator | {SHORT_FMM}\n" in _log_of(tmp)
    assert _state_of(tmp)["incomplete_chunks"] == {} and "allow_incomplete_chunks: none\n" in _log_of(tmp)

    # attempt 2: the first decision alone. The chunk is accepted, and the day check still stops the step.
    only_chunks = {"allow_incomplete_chunks": {"fmm": [FMM_SDGE]}}
    with pytest.raises(sp.MissingDaysError, match=r"fmm: no rows for 2 \(node, day\) pair\(s\): "
                                                  r"DLAP_SDGE-APND: 2026-06-22 \.\. 2026-06-23"):
        _pull(tmp, sha, resume=True, **only_chunks)
    assert len(sdge()) == 16                                # the usual eight attempts, then the last answer is taken
    log = _log_of(tmp)
    assert f'allow_incomplete_chunks: {{"fmm": ["{FMM_SDGE}"]}}\n' in log
    assert f"chunk_accepted: fmm {FMM_SDGE} | {SHORT_FMM}\n" in log
    assert (f"chunk_accepted_detail: fmm {FMM_SDGE} | 192 interval(s) of 15 minutes missing: "
            "2026-06-22T07:00Z .. 2026-06-24T06:45Z (192)\n") in log
    state = _state_of(tmp)
    assert "FMM" not in state["done"] and state["incomplete_chunks"] == {}        # the step is not done: not recorded
    assert not (tmp / "pull_unit" / "raw" / (hashlib.sha256(sdge()[0].encode()).hexdigest() + ".bin")).exists()

    # attempt 3: both decisions. The FMM step completes and is recorded in the state; NYISO's archive is down.
    down["nyiso"] = True
    both = {**only_chunks, "allow_missing_days": {"fmm": ["2026-06-22", "2026-06-23"]}}
    with pytest.raises(sp.ChunkError):
        _pull(tmp, sha, resume=True, **both)
    state = _state_of(tmp)
    assert "FMM" in state["done"] and state["incomplete_chunks"] == {"fmm": {FMM_SDGE: SHORT_FMM}}
    log = _log_of(tmp)
    assert f'allow_incomplete_chunks: {{"fmm": ["{FMM_SDGE}"]}}\n' in log
    assert 'allow_missing_days: {"fmm": ["2026-06-22", "2026-06-23"]}\n' in log
    assert ("step fmm: 2 (node, day) pair(s) without rows, allowed by allow_missing_days: "
            "DLAP_SDGE-APND: 2026-06-22 .. 2026-06-23") in log
    assert ("chunk_failed: nyiso_rtint 2026-06 | 0 of 8 attempts failed the validator | NYISO integrated 2026-06: "
            "no data after 8 attempts; the last problem: nodename nor servname provided\n") in log     # one line
    assert "came complete this time" not in log
    assert (f"accepted as incomplete, the answer with the fewest problems of 8 ({SHORT_FMM}); not cached\n"
            in (tmp / "pull_unit" / "logs" / "FMM.log").read_text())

    # attempt 4: a plain resume. The step is done, and PULLS.json carries what the state recorded.
    down["nyiso"] = False
    root, manifest = _pull(tmp, sha, resume=True)
    record = sp.completed_pulls()["A"]
    assert record["incomplete_chunks"] == {"fmm": {FMM_SDGE: SHORT_FMM}} and record["manifest_sha256"] == manifest
    assert not (root / ".state.json").exists()
    cov = pd.read_csv(root / "COVERAGE.csv")
    fmm = cov[cov["series"] == "fmm"].set_index("node")
    assert fmm.loc[SDGE, "hours"] == 0 and fmm.drop(index=SDGE)["share"].eq(1.0).all()


def test_a_pull_without_the_override_records_no_incomplete_chunk(registered, monkeypatch):
    import es.sealed_pull as sp
    write, tmp = registered
    proto, reg, sha = write()
    _use_default_builders(monkeypatch, [])
    root, _ = _pull(tmp, sha)
    assert sp.completed_pulls()["A"]["incomplete_chunks"] == {}
    assert "allow_incomplete_chunks: none\n" in (root / "PULL_LOG.txt").read_text()
    assert "chunk_failed" not in (root / "PULL_LOG.txt").read_text()


def test_a_chunk_that_is_not_listed_gives_up_again_as_before(registered, monkeypatch):
    import es.sealed_pull as sp
    write, tmp = registered
    proto, reg, sha = write()
    urls = []
    _use_default_builders(monkeypatch, urls, fetch_override=_short_answer(FMM_SDGE))
    with pytest.raises(sp.ChunkError):
        _pull(tmp, sha)
    for listed in (None, {}, {"fmm": []}, {"CAISO": [DAM_SDGE]}):         # nothing, or another chunk
        with pytest.raises(sp.ChunkError, match="no data after 8 attempts") as err:
            _pull(tmp, sha, resume=True, allow_incomplete_chunks=listed)
        assert err.value.key == FMM_SDGE and "chunk_accepted" not in _log_of(tmp)
    assert _state_of(tmp)["incomplete_chunks"] == {}
    assert len([u for u in urls if "PRC_RTPD_LMP" in u and _node(u) == SDGE]) == 8 * 5


def test_a_listed_chunk_without_an_earlier_failure_gives_up_as_before_and_is_open_next_time(registered, monkeypatch):
    """Two chunks are always short. The first attempt stops at the first of them; the second has never been asked
    for, so listing it changes nothing until it has failed all its attempts in an attempt of its own."""
    import es.sealed_pull as sp
    write, tmp = registered
    proto, reg, sha = write()
    urls = []
    short_pgae, short_sdge = _short_answer(FMM_PGAE), _short_answer(FMM_SDGE)
    _use_default_builders(monkeypatch, urls, fetch_override=lambda url: short_pgae(url) or short_sdge(url))
    asked = lambda node: len([u for u in urls if "PRC_RTPD_LMP" in u and _node(u) == node])
    with pytest.raises(sp.ChunkError) as err:
        _pull(tmp, sha)
    assert err.value.key == FMM_PGAE and asked("DLAP_PGAE-APND") == 8 and asked(SDGE) == 0
    both = {"fmm": [FMM_PGAE, FMM_SDGE]}
    with pytest.raises(sp.ChunkError, match="no data after 8 attempts") as err:
        _pull(tmp, sha, resume=True, allow_incomplete_chunks=both)
    assert err.value.key == FMM_SDGE and asked("DLAP_PGAE-APND") == 16 and asked(SDGE) == 8
    log = _log_of(tmp)
    assert f"chunk_accepted: fmm {FMM_PGAE} | {SHORT_FMM}\n" in log and f"chunk_accepted: fmm {FMM_SDGE}" not in log
    assert (f"allow_incomplete_chunks: fmm {FMM_SDGE} did not fail the validator in all 8 attempts in an earlier "
            "attempt, so it is requested and validated as usual\n") in log
    assert f'allow_incomplete_chunks: {{"fmm": ["{FMM_PGAE}"]}}\n' in log             # only the open one is in force
    assert f"chunk_failed: fmm {FMM_SDGE} | 8 of 8 attempts failed the validator | {SHORT_FMM}\n" in log
    days = ["2026-06-22", "2026-06-23"]
    with pytest.raises(sp.MissingDaysError, match=r"DLAP_PGAE-APND: 2026-06-22 \.\. 2026-06-23; "
                                                  r"DLAP_SDGE-APND: 2026-06-22 \.\. 2026-06-23"):
        _pull(tmp, sha, resume=True, allow_incomplete_chunks=both)              # now both are open
    root, _ = _pull(tmp, sha, resume=True, allow_incomplete_chunks=both, allow_missing_days={"fmm": days})
    record = sp.completed_pulls()["A"]
    assert record["incomplete_chunks"] == {"fmm": {FMM_PGAE: SHORT_FMM, FMM_SDGE: SHORT_FMM}}
    assert (root / "MANIFEST.sha256").exists()


@pytest.mark.parametrize("resume", [False, True])
def test_the_override_is_a_dict_of_lists_of_known_chunk_keys(registered, monkeypatch, resume):
    write, tmp = registered
    proto, reg, sha = write()
    _use_default_builders(monkeypatch, [])                  # the pull of DEV_WINDOW: FMM_SDGE and its like are keys
    bad = [(["fmm"], "must be a dict"), ("fmm", "must be a dict"), (True, "must be a dict"),
           ({"MISO": [FMM_SDGE]}, "unknown series 'MISO'"), ({"nyiso": []}, "unknown series 'nyiso'"),
           ({"fmm": FMM_SDGE}, "fmm needs a list of chunk keys"), ({"fmm": 5}, "fmm needs a list of chunk keys"),
           ({"fmm": {FMM_SDGE: 1}}, "fmm needs a list of chunk keys"),
           ({"fmm": [5]}, "5 is not a chunk of fmm"), ({"fmm": [None]}, "None is not a chunk of fmm"),
           ({"fmm": [DAM_SDGE]}, "is not a chunk of fmm"), ({"CAISO": [FMM_SDGE]}, "is not a chunk of CAISO"),
           ({"fmm": [FMM_SDGE.replace("T07:00.", "T07:01.")]}, "is not a chunk of fmm"),
           ({"fmm": [FMM_SDGE.replace("SDGE", "SDG")]}, "is not a chunk of fmm"),
           ({"nyiso_rtint": ["2026-07"]}, "is not a chunk of nyiso_rtint"),
           ({"nyiso_rtint": ["202606"]}, "is not a chunk of nyiso_rtint"),
           ({"fmm": [FMM_SDGE, "typo"]}, "'typo' is not a chunk of fmm")]
    for value, why in bad:
        with pytest.raises(ValueError, match=why):
            _pull(tmp, sha, resume=resume, allow_incomplete_chunks=value)
    assert not (tmp / "pull_unit").exists() and not list((tmp / "docs_sealed").glob("PULL_*.lock"))
    with pytest.raises(ValueError, match="e.g. 'PRC_LMP/TH_SP15_GEN-APND/20260621T07:00..20260624T07:00'"):
        _pull(tmp, sha, resume=resume, allow_incomplete_chunks={"CAISO": ["x"]})     # it shows what a key looks like


def test_the_accepted_forms_of_the_override():
    import es.sealed_pull as sp
    good = {"fmm": [FMM_SDGE], "CAISO": (DAM_SDGE, RTM_SDGE, DAM_SDGE), "nyiso_rtint": {"2026-06"}}
    assert sp._allowed_chunks(good, *SPAN) == {"fmm": {FMM_SDGE}, "CAISO": {DAM_SDGE, RTM_SDGE},
                                               "nyiso_rtint": {"2026-06"}}
    nothing = (sp._allowed_chunks(value, *SPAN) for value in ({"fmm": []}, {}, None))
    assert list(nothing) == [{}, {}, {}]
    assert sp._allowed_chunks({"fmm": [], "CAISO": [DAM_SDGE]}, *SPAN) == {"CAISO": {DAM_SDGE}}


def test_the_override_needs_resume_and_creates_nothing_without_it(registered, monkeypatch):
    write, tmp = registered
    proto, reg, sha = write()
    _use_default_builders(monkeypatch, [])
    with pytest.raises(ValueError, match=r"allow_incomplete_chunks needs resume=True: it applies only to a chunk that "
                                         r"failed the validator in all of its attempts in an earlier attempt"):
        _pull(tmp, sha, allow_incomplete_chunks={"fmm": [FMM_SDGE]})
    assert not (tmp / "pull_unit").exists() and not list((tmp / "docs_sealed").glob("PULL_*.lock"))
    root, _ = _pull(tmp, sha, allow_incomplete_chunks={"fmm": []})               # naming nothing needs no resume
    assert (root / "MANIFEST.sha256").exists()


def test_the_chunk_keys_are_the_requests_of_a_pull_and_nothing_else(registered, monkeypatch):
    import es.sealed_pull as sp
    write, tmp = registered
    proto, reg, sha = write()
    urls = []
    _use_default_builders(monkeypatch, urls)
    _pull(tmp, sha)
    asked = {"CAISO": [_key_of(u) for u in urls if "oasis" in u and "PRC_RTPD_LMP" not in u],
             "fmm": [_key_of(u) for u in urls if "PRC_RTPD_LMP" in u],
             "nyiso_rtint": [f"{u.rsplit('/', 1)[1][:4]}-{u.rsplit('/', 1)[1][4:6]}" for u in urls if "nyiso" in u]}
    for series, requested in asked.items():
        assert requested == sp._chunk_keys(series, *SPAN) and len(set(requested)) == len(requested)
    assert [len(v) for v in asked.values()] == [12, 6, 1] and asked["nyiso_rtint"] == ["2026-06"]
    assert sp._chunk_keys("fmm", "2026-06-21", "2026-09-30")[:4] == [          # the real window: four chunks a node
        "PRC_RTPD_LMP/TH_SP15_GEN-APND/20260621T07:00..20260721T07:00",
        "PRC_RTPD_LMP/TH_SP15_GEN-APND/20260721T07:00..20260820T07:00",
        "PRC_RTPD_LMP/TH_SP15_GEN-APND/20260820T07:00..20260919T07:00",
        "PRC_RTPD_LMP/TH_SP15_GEN-APND/20260919T07:00..20261001T07:00"]
    assert sp._chunk_keys("nyiso_rtint", "2026-09-21", "2026-12-31") == ["2026-09", "2026-10", "2026-11", "2026-12"]


def _nyiso_month_with_nan(month, zone, stamps):
    """The archive of _nyiso_month_blob for 'YYYYMM', but the price of `zone` at the local times `stamps` is NaN."""
    by_day = {}
    for t in _clock_hours("NYISO", *_around(*SPAN)).tz_convert("America/New_York"):
        if f"{t:%Y%m}" == month:
            when = f"{t:%m/%d/%Y %H:%M}"
            by_day.setdefault(f"{t:%Y%m%d}", []).extend(
                (when, z, NAN if z == zone and when in stamps else 5.0) for z in _nodes_by_market()["NYISO"])
    return _nyiso_zip(by_day)


def _never_complete(kind):
    """(fetch_override, series, chunk key, the validator's reason, the detail line, the days that allow_missing_days
    must name, the day check's log line) for a chunk of `kind` that never comes complete."""
    s0, three = T("2026-06-21 07:00"), pd.Timedelta(hours=3)

    def only(key, answer):
        return lambda url: answer(url) if "oasis" in url and _key_of(url) == key else None

    def grid_of(url, step, price_of):
        return _grid(_ts(url, "startdatetime"), _ts(url, "enddatetime"), step, price_of)

    none_in_two_days = ("2026-06-22", "2026-06-23")
    poisoned = lambda t: s0 + three <= t < s0 + three + pd.Timedelta(minutes=30)          # two 15-minute intervals
    no_rows = ("{} (node, day) pair(s) without rows, allowed by allow_missing_days: "
               "DLAP_SDGE-APND: 2026-06-22 .. 2026-06-23")
    nan_hour = ("1 (node, day) pair(s) short of their hours, allowed by allow_missing_days: {}: 2026-06-21 "
                "({} of 24 hours, {} non-finite price(s))")
    return {
        "fmm short": (only(FMM_SDGE, lambda url: _good_oasis(url, days=1)), "fmm", FMM_SDGE, SHORT_FMM,
                      "192 interval(s) of 15 minutes missing: 2026-06-22T07:00Z .. 2026-06-24T06:45Z (192)",
                      none_in_two_days, "step fmm: " + no_rows.format(2)),
        "fmm nan": (only(FMM_SDGE, lambda url: _fmm_zip(SDGE, grid_of(url, 15, lambda t: NAN if poisoned(t) else 1e2))),
                    "fmm", FMM_SDGE, "2 price(s) in fmm15 are not finite numbers",
                    f"2 price(s) not finite: {SDGE}: 2026-06-21T10:00Z .. 2026-06-21T10:15Z (2)", ("2026-06-21",),
                    "step fmm: " + nan_hour.format(SDGE, 23, 1)),
        "dam short": (only(DAM_SDGE, lambda url: _good_oasis(url, days=1)), "CAISO", DAM_SDGE,
                      "short: 48 of 72 60-minute intervals missing, first 2026-06-22 07:00:00+00:00",
                      "48 interval(s) of 60 minutes missing: 2026-06-22T07:00Z .. 2026-06-24T06:00Z (48)",
                      none_in_two_days, "step CAISO: " + no_rows.format(2)),
        "rtm hole": (only(RTM_SDGE, lambda url: _oasis_zip(SDGE, [
            (t, v) for t, v in grid_of(url, 5, lambda t: 10.0)
            if t not in (s0 + pd.Timedelta(hours=2, minutes=5), s0 + pd.Timedelta(hours=2, minutes=10))])),
                     "CAISO", RTM_SDGE, "short: 2 of 864 5-minute intervals missing, first 2026-06-21 09:05:00+00:00",
                     "2 interval(s) of 5 minutes missing: 2026-06-21T09:05Z .. 2026-06-21T09:10Z (2)", ("2026-06-21",),
                     "step CAISO: " + nan_hour.format(SDGE, 23, 1)),
        "nyiso nan": (lambda url: _nyiso_month_with_nan("202606", "N.Y.C.", ("06/21/2026 03:00", "06/21/2026 04:00"))
                      if "mis.nyiso.com" in url else None, "nyiso_rtint", "2026-06",
                      "2 price(s) in rt_int_lmp are not finite numbers",
                      "2 price(s) not finite: N.Y.C.: 2026-06-21T07:00Z .. 2026-06-21T08:00Z (2)", ("2026-06-21",),
                      "step nyiso_rtint: " + nan_hour.format("N.Y.C.", 22, 2)),
    }[kind]


@pytest.mark.parametrize("kind", ["fmm short", "fmm nan", "dam short", "rtm hole", "nyiso nan"])
def test_each_kind_of_incomplete_chunk_is_accepted_with_its_intervals_logged_and_the_days_still_reported(
        registered, monkeypatch, kind):
    import es.sealed_pull as sp
    write, tmp = registered
    proto, reg, sha = write()
    override, series, key, reason, detail, days, day_line = _never_complete(kind)
    assert "N.Y.C." in _nodes_by_market()["NYISO"]
    _use_default_builders(monkeypatch, [], fetch_override=override)
    with pytest.raises(sp.ChunkError) as err:
        _pull(tmp, sha)
    assert (err.value.key, err.value.validator_failures, err.value.reason) == (key, 8, reason)
    assert f"chunk_failed: {series} {key} | 8 of 8 attempts failed the validator | {reason}\n" in _log_of(tmp)
    with pytest.raises(sp.MissingDaysError, match=series):                    # the first decision alone is not enough
        _pull(tmp, sha, resume=True, allow_incomplete_chunks={series: [key]})
    root, _ = _pull(tmp, sha, resume=True, allow_incomplete_chunks={series: [key]}, allow_missing_days={series: days})
    log = (root / "PULL_LOG.txt").read_text()
    assert f"chunk_accepted: {series} {key} | {reason}\n" in log
    assert f"chunk_accepted_detail: {series} {key} | {detail}\n" in log
    assert day_line in log
    assert sp.completed_pulls()["A"]["incomplete_chunks"] == {series: {key: reason}}


def test_a_chunk_with_a_network_failure_among_its_attempts_is_not_open_to_the_override(registered, monkeypatch):
    import es.sealed_pull as sp
    write, tmp = registered
    proto, reg, sha = write()
    asked = collections.Counter()

    def short_but_once(url):
        if "oasis" in url and _key_of(url) == FMM_SDGE:
            asked[FMM_SDGE] += 1
            if asked[FMM_SDGE] == 4:
                raise OSError("nodename nor servname provided")
            return _good_oasis(url, days=1)

    _use_default_builders(monkeypatch, [], fetch_override=short_but_once)
    with pytest.raises(sp.ChunkError) as err:
        _pull(tmp, sha)
    assert (err.value.validator_failures, err.value.attempts) == (7, 8)
    assert f"chunk_failed: fmm {FMM_SDGE} | 7 of 8 attempts failed the validator | {SHORT_FMM}\n" in _log_of(tmp)
    listed = {"allow_incomplete_chunks": {"fmm": [FMM_SDGE]}}
    with pytest.raises(sp.ChunkError) as err:                       # listed, but seven are not eight: asked as usual
        _pull(tmp, sha, resume=True, **listed)
    assert err.value.validator_failures == 8 and asked[FMM_SDGE] == 16
    assert "did not fail the validator in all 8 attempts in an earlier attempt" in _log_of(tmp)
    assert "chunk_accepted" not in _log_of(tmp)
    with pytest.raises(sp.MissingDaysError):                        # the second attempt did fail eight times
        _pull(tmp, sha, resume=True, **listed)
    assert f"chunk_accepted: fmm {FMM_SDGE} |" in _log_of(tmp)


def test_rows_outside_the_interval_are_never_accepted_as_incomplete(registered, monkeypatch):
    import es.sealed_pull as sp
    write, tmp = registered
    proto, reg, sha = write()

    def spills_over(url):
        if "oasis" in url and _key_of(url) == FMM_SDGE:
            s, e = _ts(url, "startdatetime"), _ts(url, "enddatetime")      # the first day is missing, one row too many
            return _fmm_zip(SDGE, _grid(s + pd.Timedelta(days=1), e + pd.Timedelta(minutes=15), 15, lambda t: 100.0))

    _use_default_builders(monkeypatch, [], fetch_override=spills_over)
    reason = "1 row(s) outside the requested 2026-06-21 07:00:00+00:00 .. 2026-06-24 07:00:00+00:00"
    with pytest.raises(sp.ChunkError, match="outside the requested"):
        _pull(tmp, sha)
    assert f"chunk_failed: fmm {FMM_SDGE} | 8 of 8 attempts failed the validator | {reason}\n" in _log_of(tmp)
    with pytest.raises(sp.ChunkError, match="outside the requested") as err:              # open, and still refused
        _pull(tmp, sha, resume=True, allow_incomplete_chunks={"fmm": [FMM_SDGE]})
    assert err.value.validator_failures == 8 and "chunk_accepted" not in _log_of(tmp)
    assert _state_of(tmp)["incomplete_chunks"] == {}


def test_a_listed_chunk_that_comes_complete_this_time_is_not_recorded_as_incomplete(registered, monkeypatch):
    import es.sealed_pull as sp
    write, tmp = registered
    proto, reg, sha = write()
    broken, short = {"on": True}, _short_answer(FMM_SDGE)
    _use_default_builders(monkeypatch, [], fetch_override=lambda url: short(url) if broken["on"] else None)
    with pytest.raises(sp.ChunkError):
        _pull(tmp, sha)
    broken["on"] = False                                               # OASIS has filled the gap
    root, _ = _pull(tmp, sha, resume=True, allow_incomplete_chunks={"fmm": [FMM_SDGE]})     # no allow_missing_days
    log = (root / "PULL_LOG.txt").read_text()
    assert f"allow_incomplete_chunks: fmm {FMM_SDGE} came complete this time; nothing is accepted as incomplete" in log
    assert "chunk_accepted" not in log and sp.completed_pulls()["A"]["incomplete_chunks"] == {}
    cov = pd.read_csv(root / "COVERAGE.csv")
    assert cov[cov["series"] == "fmm"]["share"].eq(1.0).all()


@pytest.mark.parametrize("state_kept", [True, False], ids=["from the state", "from the log, the state being gone"])
def test_a_pull_killed_before_its_record_is_adopted_with_its_incomplete_chunks(registered, monkeypatch, state_kept):
    import es.sealed_pull as sp
    write, tmp = registered
    proto, reg, sha = write()
    _use_default_builders(monkeypatch, [], fetch_override=_short_answer(FMM_SDGE))
    with pytest.raises(sp.ChunkError):
        _pull(tmp, sha)
    real = os.replace

    def killed(src, dst):
        if Path(dst).name == "PULLS.json":
            raise KeyboardInterrupt("killed while PULLS.json was being written")
        return real(src, dst)

    monkeypatch.setattr(os, "replace", killed)
    with pytest.raises(KeyboardInterrupt):
        _pull(tmp, sha, resume=True, allow_incomplete_chunks={"fmm": [FMM_SDGE]},
              allow_missing_days={"fmm": ["2026-06-22", "2026-06-23"]})
    monkeypatch.setattr(os, "replace", real)
    root = tmp / "pull_unit"
    assert (root / "MANIFEST.sha256").exists() and sp.completed_pulls() == {}
    if not state_kept:
        (root / ".state.json").unlink()
    _pull(tmp, sha, resume=True)                                           # adopted: nothing is downloaded
    assert sp.completed_pulls()["A"]["incomplete_chunks"] == {"fmm": {FMM_SDGE: SHORT_FMM}}


def test_the_record_holds_the_chunks_accepted_by_the_attempt_that_finished_the_step(registered, monkeypatch):
    """Two chunks are accepted in an attempt whose day check then stops the step. The next attempt names only one of
    them: the other has come complete. The state, and PULLS.json after an adoption, hold the one, not both; the log, the
    only record left once the state is gone, says what was accepted at any attempt."""
    import es.sealed_pull as sp
    write, tmp = registered
    proto, reg, sha = write()
    sdge_broken, short_pgae, short_sdge = {"on": True}, _short_answer(FMM_PGAE), _short_answer(FMM_SDGE)
    _use_default_builders(monkeypatch, [], fetch_override=lambda url: short_pgae(url) or (
        short_sdge(url) if sdge_broken["on"] else None))
    days = ["2026-06-22", "2026-06-23"]
    with pytest.raises(sp.ChunkError):                                         # PGAE gives up
        _pull(tmp, sha)
    with pytest.raises(sp.ChunkError):                                         # PGAE is taken; SDGE gives up
        _pull(tmp, sha, resume=True, allow_incomplete_chunks={"fmm": [FMM_PGAE, FMM_SDGE]})
    with pytest.raises(sp.MissingDaysError):                                   # both taken; days not excused
        _pull(tmp, sha, resume=True, allow_incomplete_chunks={"fmm": [FMM_PGAE, FMM_SDGE]})
    assert _log_of(tmp).count(f"chunk_accepted: fmm {FMM_SDGE} |") == 1
    sdge_broken["on"] = False                                                  # SDGE comes complete
    real = os.replace

    def killed(src, dst):
        if Path(dst).name == "PULLS.json":
            raise KeyboardInterrupt("killed while PULLS.json was being written")
        return real(src, dst)

    monkeypatch.setattr(os, "replace", killed)
    with pytest.raises(KeyboardInterrupt):
        _pull(tmp, sha, resume=True, allow_incomplete_chunks={"fmm": [FMM_PGAE]}, allow_missing_days={"fmm": days})
    monkeypatch.setattr(os, "replace", real)
    assert _state_of(tmp)["incomplete_chunks"] == {"fmm": {FMM_PGAE: SHORT_FMM}}
    assert _log_of(tmp).count(f"chunk_accepted: fmm {FMM_PGAE} |") == 3 and sp._logged_accepted_chunks(
        tmp / "pull_unit") == {"fmm": {FMM_PGAE: SHORT_FMM, FMM_SDGE: SHORT_FMM}}
    _pull(tmp, sha, resume=True)                                               # adopted from the state
    assert sp.completed_pulls()["A"]["incomplete_chunks"] == {"fmm": {FMM_PGAE: SHORT_FMM}}


def test_a_validator_that_finds_nothing_to_accept_is_not_accepted():
    """A response that a validator refuses, whose facts show nothing missing or not finite, is not one the override
    covers."""
    import es.sealed_pull as sp
    s, e, key, real = _hourly_chunk()
    lines = []
    odd = lambda part: "refused for some other reason"
    odd.problems = lambda part: {"outside": 0, "missing": pd.DatetimeIndex([]), "nonfinite": part.iloc[:0]}
    odd.texts, odd.step = lambda found: ["refused for some other reason"], 60
    part = sp._parse_oasis_xml(_oasis_zip("X", [(s, 10.0)]))
    inc = sp.IncompleteChunks("CAISO", {key}, lambda *ls: lines.extend(ls))
    assert inc.take(key, part, odd) is False and inc.accepted == {} and lines == []
    assert inc.take(key, part, real) is True and list(inc.accepted) == [key]       # the real validator does find them


def test_a_state_from_before_the_override_resumes_with_an_empty_record(registered):
    from es.sealed_pull import completed_pulls
    write, tmp, sha, builders, root = _failed_first_attempt(registered)
    state = json.loads((root / ".state.json").read_text())
    assert state["incomplete_chunks"] == {}
    del state["incomplete_chunks"]
    (root / ".state.json").write_text(json.dumps(state))
    _a_pull(tmp, sha, builders=builders, resume=True)
    assert completed_pulls()["A"]["incomplete_chunks"] == {}


# ----------------------------------------------------------------------------- _fetch_parse and the small helpers
def _hourly_chunk():
    """One day of hourly prices (24 intervals) and what a fetch of it needs."""
    import es.sealed_pull as sp
    s, e = T("2026-06-21 07:00"), T("2026-06-22 07:00")
    key = "PRC_LMP/X/20260621T07:00..20260622T07:00"
    return s, e, key, sp._grid_validator(s, e, 60, "CAISO", True, "lmp")


def test_fetch_parse_still_raises_unless_the_chunk_is_listed_and_the_response_is_one_the_override_covers(tmp_path):
    import es.sealed_pull as sp
    s, e, key, validate = _hourly_chunk()
    rows = lambda n: [(t, 10.0) for t, _ in _grid(s, e, 60, lambda t: 0)][:n]
    answers = {"short": _oasis_zip("X", rows(20)), "outside": _oasis_zip("X", rows(20) + [(e, 10.0)]),
               "network": None}

    def run(kind, *, listed=True, name=key, check=validate, incomplete=True):
        sent = []

        def fetch(url):
            sent.append(url)
            if answers[kind] is None:
                raise OSError("nodename nor servname provided")
            return answers[kind]

        inc = sp.IncompleteChunks("CAISO", {key} if listed else set(), lambda *ls: None) if incomplete else None
        with pytest.raises(sp.ChunkError, match="after 8 attempts") as err:
            sp._fetch_parse("http://x/y", sp._parse_oasis_xml, "what", fetch, lambda _s: None, validate=check,
                            key=name, incomplete=inc)
        return err.value, inc, len(sent)

    error, _, sent = run("short", incomplete=False)                          # no override at all
    assert (error.key, error.attempts, error.validator_failures, sent) == (key, 8, 8, 8)
    assert error.reason == "short: 4 of 24 60-minute intervals missing, first 2026-06-22 03:00:00+00:00"
    assert isinstance(error, RuntimeError) and "no data after 8 attempts; the last problem:" in str(error)
    for kwargs in ({"listed": False}, {"name": "PRC_LMP/X/another"}, {"name": None},
                   {"check": lambda part: "short"}):                         # not listed; another key; no key; no facts
        error, inc, _ = run("short", **kwargs)
        assert inc.accepted == {} and error.validator_failures == 8
    error, inc, _ = run("outside")                                           # listed, but a row is outside the interval
    assert inc.accepted == {} and error.validator_failures == 8 and "outside the requested" in error.reason
    error, inc, sent = run("network")                                        # listed, but no whole response ever came
    assert inc.accepted == {} and (error.validator_failures, error.reason, sent) == (0, None, 8)


def test_a_chunk_error_is_a_runtime_error_with_the_facts_of_the_chunk():
    import es.sealed_pull as sp
    assert issubclass(sp.ChunkError, RuntimeError)
    error = sp.ChunkError("x", key="k", attempts=8, validator_failures=3, reason="why")
    assert (str(error), error.key, error.attempts, error.validator_failures, error.reason) == ("x", "k", 8, 3, "why")
    bare = sp.ChunkError("y")
    assert (bare.key, bare.attempts, bare.validator_failures, bare.reason) == (None, 0, 0, None)


def test_runs_name_every_stretch_of_consecutive_intervals():
    import es.sealed_pull as sp
    t = lambda minutes: T("2026-06-21 07:00") + pd.Timedelta(minutes=minutes)
    assert sp._runs([t(0)], 15) == "2026-06-21T07:00Z"
    assert sp._runs([t(135), t(30), t(0), t(15), t(60), t(120)], 15) == (
        "2026-06-21T07:00Z .. 2026-06-21T07:30Z (3), 2026-06-21T08:00Z, 2026-06-21T09:00Z .. 2026-06-21T09:15Z (2)")
    assert sp._runs(pd.DatetimeIndex([t(0), t(5), t(10)]), 5) == "2026-06-21T07:00Z .. 2026-06-21T07:10Z (3)"
    assert sp._runs([t(0), t(15)], 5) == "2026-06-21T07:00Z, 2026-06-21T07:15Z"    # 15 minutes apart: not consecutive


def test_an_hour_with_fewer_than_twelve_five_minute_prices_is_nan():
    from es.sealed_pull import _hourly_rt_mean
    stamps = pd.date_range(T("2026-06-01 07:00"), periods=36, freq="5min")           # three hours
    frame = pd.DataFrame({"node": "A", "timestamp_utc": stamps.delete(17), "lmp": [10.0] * 35})
    out = _hourly_rt_mean(frame)["lmp"]
    assert out.iloc[0] == 10.0 and pd.isna(out.iloc[1]) and out.iloc[2] == 10.0       # the second hour has eleven


def test_only_builders_that_declare_incomplete_are_handed_the_override(tmp_path):
    import es.sealed_pull as sp
    for name in ("CAISO", "FMM", "NYISO_INT"):
        assert sp._takes_incomplete(sp.REAL_DOWNLOADERS[name])
    for name in ("NYISO", "ISONE", "MISO"):
        assert not sp._takes_incomplete(sp.REAL_DOWNLOADERS[name])
    assert not sp._takes_incomplete(object())                                         # no signature to read
    got = []
    sp._run_step("X", lambda s, e, r: got.append("plain"), "a", "b", tmp_path, incomplete="the handler")
    takes = lambda s, e, r, incomplete=None: got.append(incomplete)
    sp._run_step("Y", takes, "a", "b", tmp_path, incomplete="the handler")
    sp._run_step("Z", takes, "a", "b", tmp_path)
    assert got == ["plain", "the handler", None]


# =====================================================================================================
# Ruling 116 (Mn4): the day check counts only rows on the hour, and a day with more rows than hours fails
# =====================================================================================================
MAINE = ".Z.MAINE"


def _isone_rows(first, last, nodes=None):
    """The rows of ISO-NE's July downloader for the local days first..last: every node, every hour, on the hour."""
    return _full_price_rows("ISONE", _nodes_by_market()["ISONE"] if nodes is None else nodes, first, last)


def _after(rows, minutes):
    """The same rows, `minutes` after the hour (the stamp is the fourth field)."""
    return [(*r[:3], r[3] + pd.Timedelta(minutes=minutes), *r[4:]) for r in rows]


def _write_isone(root, rows):
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=PRICE_COLUMNS).to_parquet(root / "isone_da_rt_x.parquet", index=False)


def test_a_row_is_on_the_hour_when_minute_second_and_fraction_are_zero_in_utc():
    from es.sealed_pull import _on_the_hour
    stamps = ["2026-06-22 18:00", "2026-06-22 18:30", "2026-06-22 18:00:01", "2026-06-22 18:00:00.000001",
              "2026-06-22 00:00"]
    frame = pd.DataFrame({"timestamp_utc": [T(s) for s in stamps]})
    assert _on_the_hour(frame).tolist() == [True, False, False, False, True]
    naive = pd.DataFrame({"timestamp_utc": pd.to_datetime(["2026-06-22 18:00", "2026-06-22 18:15"])})   # MISO's files
    assert _on_the_hour(naive).tolist() == [True, False]
    est = pd.DataFrame({"timestamp_utc": [pd.Timestamp("2026-06-22 13:00", tz="America/New_York"),
                                          pd.Timestamp("2026-06-22 13:30", tz="America/New_York")]})
    assert _on_the_hour(est).tolist() == [True, False]                       # any zone: whole hours apart from UTC


def test_a_series_stamped_on_the_half_hour_has_no_hours(tmp_path):
    from es.sealed_pull import _day_gaps
    first, last = "2026-06-21", "2026-06-23"
    nodes = _nodes_by_market()["ISONE"]
    _write_isone(tmp_path, _after(_isone_rows(first, last), 30))
    no_rows, short, over = _day_gaps(tmp_path, "ISONE", first, last)
    assert no_rows == [(n, dt.date(2026, 6, d)) for n in nodes for d in (21, 22, 23)] and short == {} and over == {}


def test_a_day_with_each_hour_twice_has_more_rows_than_hours(tmp_path):
    from es.sealed_pull import _day_gaps
    first, last = "2026-06-21", "2026-06-23"
    rows = _isone_rows(first, last) + _after(_isone_rows("2026-06-22", "2026-06-22", [MAINE]), 30)
    _write_isone(tmp_path, rows)
    no_rows, short, over = _day_gaps(tmp_path, "ISONE", first, last)
    assert (no_rows, short) == ([], {}) and over == {(MAINE, dt.date(2026, 6, 22)): (48, 24, 24)}
    from es.sealed_pull import _missing_days
    assert _missing_days(tmp_path, "ISONE", first, last) == [(MAINE, dt.date(2026, 6, 22))]    # it is a missing day


def test_one_stray_row_is_one_row_too_many(tmp_path):
    from es.sealed_pull import _day_gaps
    first, last = "2026-06-21", "2026-06-23"
    stray = _after(_isone_rows("2026-06-22", "2026-06-22", [MAINE])[:1], 15)
    _write_isone(tmp_path, _isone_rows(first, last) + stray)
    assert _day_gaps(tmp_path, "ISONE", first, last) == ([], {}, {(MAINE, dt.date(2026, 6, 22)): (25, 24, 1)})


def test_a_day_with_a_missing_hour_and_two_stray_rows_is_short_and_says_how_many_are_off_the_hour(tmp_path):
    from es.sealed_pull import _day_gaps
    first, last = "2026-06-21", "2026-06-23"
    rows = [r for r in _isone_rows(first, last) if not (r[1] == MAINE and r[3] == T("2026-06-22 09:00"))]
    stray = _after(_isone_rows("2026-06-22", "2026-06-22", [MAINE])[:2], 30)
    _write_isone(tmp_path, rows + stray)
    no_rows, short, over = _day_gaps(tmp_path, "ISONE", first, last)
    assert (no_rows, over) == ([], {})                   # 25 rows for 24 hours, but 23 on the hour: the day is short
    assert short == {(MAINE, dt.date(2026, 6, 22)): (23, 24, 0, 2)}


def test_the_details_of_short_days_and_days_over_their_hours_are_written_with_their_counts():
    from es.sealed_pull import _fmt_over, _fmt_short
    day, other = dt.date(2026, 6, 22), dt.date(2026, 6, 23)
    assert _fmt_short({("A", day): (23, 24, 0, 0), ("A", other): (22, 24, 1, 2), ("B", day): (0, 24, 0, 24)}) == (
        "A: 2026-06-22 (23 of 24 hours), 2026-06-23 (22 of 24 hours, 1 non-finite price(s), 2 off-hour row(s)); "
        "B: 2026-06-22 (0 of 24 hours, 24 off-hour row(s))")
    assert _fmt_over({("A", day): (25, 24, 0), ("B", day): (48, 24, 24), ("B", other): (25, 23, 1)}) == (
        "A: 2026-06-22 (25 rows for 24 hours); B: 2026-06-22 (48 rows for 24 hours, 24 off-hour row(s)), "
        "2026-06-23 (25 rows for 23 hours, 1 off-hour row(s))")


def test_a_row_that_is_there_twice_is_one_row(tmp_path):
    """The readers keep the first of rows with the same market, node and instant, so an exact repeat adds nothing."""
    from es.sealed_pull import _day_gaps
    first, last = "2026-06-21", "2026-06-23"
    one = _isone_rows("2026-06-22", "2026-06-22", [MAINE])
    _write_isone(tmp_path, _isone_rows(first, last) + one)
    twice = tmp_path / "isone_da_rt_y.parquet"                                              # the same file again
    twice.write_bytes((tmp_path / "isone_da_rt_x.parquet").read_bytes())
    assert _day_gaps(tmp_path, "ISONE", first, last) == ([], {}, {})


def test_a_fall_back_day_that_kept_the_repeated_hour_has_more_rows_than_hours_in_nyiso(tmp_path):
    """NYISO's July downloader drops the two rows of the repeated hour: 23 on 2 November 2025. 25 would be a change."""
    from es.sealed_pull import _day_gaps
    nodes = _nodes_by_market()["NYISO"]
    rows = [("NYISO", n, "hub", t, 1.0, 1.0) for n in nodes for t in _clock_hours("NYISO", "2025-11-02", "2025-11-02")]
    pd.DataFrame(rows, columns=PRICE_COLUMNS).to_parquet(tmp_path / "nyiso_da_rt_x.parquet", index=False)
    no_rows, short, over = _day_gaps(tmp_path, "NYISO", "2025-11-02", "2025-11-02")
    assert (no_rows, short) == ([], {}) and over == {(n, dt.date(2025, 11, 2)): (25, 23, 0) for n in nodes}


@pytest.mark.parametrize("series, market, node", [("fmm", "CAISO", SDGE), ("nyiso_rtint", "NYISO", "N.Y.C.")])
def test_the_settlement_tables_are_counted_on_the_hour_too(tmp_path, series, market, node):
    from es.sealed_pull import _day_gaps
    value = (100.0, 4) if series == "fmm" else (5.0,)
    hours = [f"{t:%Y-%m-%d %H:%M}" for t in _clock_hours(market, "2026-06-22", "2026-06-22")]
    half = [f"{t + pd.Timedelta(minutes=30):%Y-%m-%d %H:%M}" for t in _clock_hours(market, "2026-06-22", "2026-06-22")]
    day = dt.date(2026, 6, 22)
    _write_side(tmp_path / series, series, [(node, h, *value) for h in half])
    assert _day_gaps(tmp_path, series, "2026-06-22", "2026-06-22")[0].count((node, day)) == 1       # no hour at all
    _write_side(tmp_path / series, series, [(node, h, *value) for h in hours])
    assert _day_gaps(tmp_path, series, "2026-06-22", "2026-06-22")[0].count((node, day)) == 0       # all on the hour
    _write_side(tmp_path / series, series, [(node, h, *value) for h in hours + half[:1]])
    no_rows, short, over = _day_gaps(tmp_path, series, "2026-06-22", "2026-06-22")
    assert (node, day) not in no_rows and short == {} and over == {(node, day): (25, 24, 1)}


def test_coverage_counts_only_the_hours_that_are_on_the_hour(tmp_path):
    from es.sealed_pull import coverage_manifest
    rows = _isone_rows("2026-06-22", "2026-06-22", [MAINE])
    _write_isone(tmp_path, rows + _after(rows, 30))
    cov = coverage_manifest(tmp_path, "2026-06-22", "2026-06-22")
    row = cov[(cov["series"] == "price") & (cov["node"] == MAINE)].iloc[0]
    assert (row["hours"], row["expected_hours"], row["share"], row["nonfinite_rows"]) == (24, 24, 1.0, 0)
    _write_isone(tmp_path, _after(rows, 30))
    cov = coverage_manifest(tmp_path, "2026-06-22", "2026-06-22")
    assert cov[(cov["series"] == "price") & (cov["node"] == MAINE)].iloc[0]["hours"] == 0


def _isone_builder(rows_of):
    """A July downloader for ISO-NE as a fake: `rows_of(first, last)` gives its rows for the days of the pull, plus the
    day on each side that the real downloaders also write."""
    def build(start, end, root):
        import es.sealed_pull as sp
        first, _, last, _ = sp.SEALED_WINDOWS["A"]
        _write_isone(root, rows_of(*_around(first, last)))
    return build


def test_a_series_stamped_on_the_half_hour_stops_its_step_and_the_log_counts_the_rows_off_the_hour(
        registered, monkeypatch):
    import es.sealed_pull as sp
    write, tmp = registered
    proto, reg, sha = write()
    _use_default_builders(monkeypatch, [])
    builders = {**sp._default_builders(), "ISONE": _isone_builder(lambda a, b: _after(_isone_rows(a, b), 30))}
    kw = dict(data_root=tmp, reference_dirs=[])
    nodes = _nodes_by_market()["ISONE"]
    gap = rf"ISONE: no rows for {3 * len(nodes)} \(node, day\) pair\(s\)"
    with pytest.raises(sp.MissingDaysError, match=gap) as err:
        sp.pull_sealed("A", sha, builders=builders, **kw)
    assert f"{72 * len(nodes)} row(s) are not on the hour and were not counted as hours" in str(err.value)
    assert "Inspect the rows of a day before naming it" in str(err.value)
    log = (tmp / "pull_unit" / "PULL_LOG.txt").read_text()
    per_node = "; ".join(f"{n}: 72" for n in nodes)
    assert (f"step ISONE: {72 * len(nodes)} row(s) are not on the hour (minute or second not 0, UTC) and do not count "
            f"as hours: {per_node}\n") in log
    assert json.loads((tmp / "pull_unit" / ".state.json").read_text())["done"] == ["NYISO"]
    days = ["2026-06-21", "2026-06-22", "2026-06-23"]
    root, _ = sp.pull_sealed("A", sha, builders=builders, resume=True, allow_missing_days={"ISONE": days}, **kw)
    log = (root / "PULL_LOG.txt").read_text()
    assert f"step ISONE: {3 * len(nodes)} (node, day) pair(s) without rows, allowed by allow_missing_days:" in log
    cov = pd.read_csv(root / "COVERAGE.csv")
    isone = cov[(cov["series"] == "price") & (cov["market"] == "ISONE")]
    assert (isone["hours"] == 0).all() and (isone["nonfinite_rows"] == 0).all()


def test_a_day_with_each_hour_twice_stops_its_step_and_is_excused_only_by_name(registered, monkeypatch):
    import es.sealed_pull as sp
    write, tmp = registered
    proto, reg, sha = write()
    _use_default_builders(monkeypatch, [])
    builders = {**sp._default_builders(), "ISONE": _isone_builder(
        lambda a, b: _isone_rows(a, b) + _after(_isone_rows("2026-06-22", "2026-06-22", [MAINE]), 30))}
    kw = dict(data_root=tmp, reference_dirs=[])
    gap = (r"ISONE: more rows than the day has hours on 1 \(node, day\) pair\(s\): \.Z\.MAINE: 2026-06-22 "
           r"\(48 rows for 24 hours, 24 off-hour row\(s\)\)")
    with pytest.raises(sp.MissingDaysError, match=gap) as err:
        sp.pull_sealed("A", sha, builders=builders, **kw)
    assert "fewer usable hours" not in str(err.value) and "no rows for" not in str(err.value)
    log = (tmp / "pull_unit" / "PULL_LOG.txt").read_text()
    assert ("step ISONE: 1 (node, day) pair(s) over their hours: .Z.MAINE: 2026-06-22 "
            "(48 rows for 24 hours, 24 off-hour row(s))") in log
    assert ("step ISONE: 24 row(s) are not on the hour (minute or second not 0, UTC) and do not count as hours: "
            ".Z.MAINE: 24") in log
    assert "step ISONE: every node-map node has the hours" not in log            # the step is not logged as complete
    with pytest.raises(sp.MissingDaysError):                                    # another day, another market: no excuse
        sp.pull_sealed("A", sha, builders=builders, resume=True, allow_missing_days={"ISONE": ["2026-06-23"]}, **kw)
    root, _ = sp.pull_sealed("A", sha, builders=builders, resume=True,
                             allow_missing_days={"ISONE": ["2026-06-22"]}, **kw)
    log = (root / "PULL_LOG.txt").read_text()
    assert ("step ISONE: 1 (node, day) pair(s) over their hours, allowed by allow_missing_days: .Z.MAINE: 2026-06-22 "
            "(48 rows for 24 hours, 24 off-hour row(s))") in log
    cov = pd.read_csv(root / "COVERAGE.csv")
    maine = cov[(cov["series"] == "price") & (cov["node"] == MAINE)].iloc[0]
    assert (maine["hours"], maine["share"]) == (48, 1.0)                        # the 48 hours on the hour, never 72


@pytest.mark.parametrize("series, step, builder", [("fmm", "FMM", _fmm_table_builder),
                                                   ("nyiso_rtint", "NYISO_INT", _nyiso_table_builder)])
def test_a_settlement_table_stamped_off_the_hour_stops_its_step(registered, monkeypatch, series, step, builder):
    import es.sealed_pull as sp
    write, tmp = registered
    proto, reg, sha = write()
    _use_default_builders(monkeypatch, [])
    inner = builder()

    def half_past(start, end, root):
        inner(start, end, root)
        table = next((root / series).glob("*.parquet"))
        frame = pd.read_parquet(table)
        frame["timestamp_utc"] = frame["timestamp_utc"] + pd.Timedelta(minutes=30)
        frame.to_parquet(table, index=False)

    with pytest.raises(sp.MissingDaysError, match=rf"{series}: no rows for") as err:
        sp.pull_sealed("A", sha, data_root=tmp, reference_dirs=[], builders={**sp._default_builders(), step: half_past})
    assert "row(s) are not on the hour and were not counted as hours" in str(err.value)
    assert f"step {series}: " in (tmp / "pull_unit" / "PULL_LOG.txt").read_text()


# =====================================================================================================
# Ruling 121 (final fixes). F1: an answer that arrived whole but holds no data is a whole answer, for eligibility and
# for acceptance. F2: an incomplete chunk is accepted only if every attempt of the accepting call was a whole answer,
# and the answer kept is the one with the fewest problems. F4: a cached NYISO month that is short says how to have it
# fetched again.
# =====================================================================================================
OASIS_ERROR = "OASIS error 1000: No data returned for the specified selection"


def _oasis_error_zip(code="1000", desc="No data returned for the specified selection"):
    """What OASIS answers when it has nothing to give: a zip with one XML report that holds an error and no data items,
    and no CSV (so a resultformat=6 request gets a zip without a CSV)."""
    xml = ('<?xml version="1.0" encoding="UTF-8"?><m:OASISReport xmlns:m="http://www.caiso.com/soa/OASISReport_v1.xsd">'
           f"<m:MessagePayload><m:RTO><m:name>CAISO</m:name><m:ERROR><m:ERR_CODE>{code}</m:ERR_CODE>"
           f"<m:ERR_DESC>{desc}</m:ERR_DESC></m:ERROR></m:RTO></m:MessagePayload></m:OASISReport>")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("20260621_20260624_PRC_RTPD_LMP.xml", xml)
    return buf.getvalue()


def _empty_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w"):
        pass
    return buf.getvalue()


def _zip_of(name, text):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(name, text)
    return buf.getvalue()


#: (id, blob, parser, what the empty table says the server said): well-formed answers that hold no data.
NO_DATA = [("a CSV with a header and no rows (FMM)", lambda: _fmm_zip("X", []), "_parse_fmm_answer", None),
           ("a zip with an error and no CSV (FMM)", _oasis_error_zip, "_parse_fmm_answer", OASIS_ERROR),
           ("a zip with no file in it (FMM)", _empty_zip, "_parse_fmm_answer", "the zip holds no CSV file"),
           ("an XML report with no data items", lambda: _oasis_zip("X", []), "_parse_oasis_xml",
            "the report holds no data items"),
           ("an XML report that is an OASIS error", _oasis_error_zip, "_parse_oasis_xml", OASIS_ERROR),
           ("a zip with no file in it (XML)", _empty_zip, "_parse_oasis_xml", "the zip holds no files"),
           ("a monthly archive with no CSV file", _empty_zip, "_parse_nyiso_answer", "the zip holds no CSV file"),
           ("a monthly archive of CSV files without rows", lambda: _nyiso_zip({"20260630": []}),
            "_parse_nyiso_answer", None),
           ("a CSV of no bytes (FMM)", lambda: _zip_of("report.csv", ""), "_parse_fmm_answer",
            "the CSV in the zip is empty"),
           ("a CSV of no bytes (NYISO)", lambda: _zip_of("20260630rtlbmp_zone.csv", ""), "_parse_nyiso_answer",
            "the CSV in the zip is empty"),
           ("an XML report with data items of another kind", lambda: _zip_of("report.xml", (
               '<OASISReport xmlns="http://www.caiso.com/soa/OASISReport_v1.xsd"><MessagePayload><RTO><REPORT_DATA>'
               "<DATA_ITEM>LMP_CONG_PRC</DATA_ITEM><RESOURCE_NAME>X</RESOURCE_NAME>"
               "<INTERVAL_START_GMT>2026-06-21T07:00:00-00:00</INTERVAL_START_GMT><VALUE>1.0</VALUE></REPORT_DATA>"
               "</RTO></MessagePayload></OASISReport>")), "_parse_oasis_xml", "the report holds no LMP items")]
NO_DATA_IDS = [row[0] for row in NO_DATA]
EMPTY_COLUMNS = {"_parse_oasis_xml": ["timestamp_utc", "node", "lmp"],
                 "_parse_fmm_answer": ["node", "timestamp_utc", "fmm15"],
                 "_parse_nyiso_answer": ["node", "timestamp_utc", "rt_int_lmp"]}


@pytest.mark.parametrize("label, blob, parser, answer", NO_DATA, ids=NO_DATA_IDS)
def test_f1_a_well_formed_answer_without_data_is_an_empty_table_that_says_what_the_server_said(
        label, blob, parser, answer):
    import es.sealed_pull as sp
    table = getattr(sp, parser)(blob())
    assert len(table) == 0 and list(table.columns) == EMPTY_COLUMNS[parser]
    assert str(table["timestamp_utc"].dtype) == "datetime64[ns, UTC]" and table["node"].dtype == object
    assert table[EMPTY_COLUMNS[parser][-1]].dtype == float
    assert table.attrs.get("answer") == answer


@pytest.mark.parametrize("parser", ["_parse_oasis_xml", "_parse_fmm_answer", "_parse_nyiso_answer"])
def test_f1_an_answer_that_cannot_be_read_at_all_still_raises(parser):
    import es.sealed_pull as sp
    whole = {"_parse_oasis_xml": _oasis_zip("X", [(T("2026-06-21 07:00"), 1.0)]),
             "_parse_fmm_answer": _fmm_zip("X", [(T("2026-06-21 07:00"), 1.0)]),
             "_parse_nyiso_answer": _nyiso_zip({"20260630": [("06/30/2026 23:00", "N.Y.C.", 1.0)]})}[parser]
    assert len(getattr(sp, parser)(whole)) == 1                                  # the same parser reads a whole answer
    for garbage in (b"<html>503 Service Unavailable</html>", whole[:40], b""):
        with pytest.raises(zipfile.BadZipFile):
            getattr(sp, parser)(garbage)


def test_f1_the_parsers_of_complete_answers_return_what_the_settlement_parsers_return():
    import es.sealed_pull as sp
    from es.settlement import _parse_caiso, _parse_nyiso
    fmm = _fmm_zip(SDGE, _grid(T("2026-06-21 07:00"), T("2026-06-22 07:00"), 15, lambda t: 100.0))
    assert sp._parse_fmm_answer(fmm).equals(_parse_caiso(fmm)) and sp._parse_fmm_answer(fmm).attrs == {}
    month = _nyiso_month_blob("202606", *SPAN)
    ours, theirs = sp._parse_nyiso_answer(month), _parse_nyiso(month)
    assert ours.equals(theirs) and ours.attrs == theirs.attrs and len(ours) > 0


def _network_failures():
    import http.client
    import urllib.error
    return [OSError("nodename nor servname provided"), TimeoutError("timed out"), ConnectionResetError("reset by peer"),
            urllib.error.URLError("unreachable"), http.client.IncompleteRead(b"ab", 10),
            urllib.error.HTTPError("http://x/y", 404, "Not Found", {}, None),
            urllib.error.HTTPError("http://x/y", 500, "Internal Server Error", {}, None),
            urllib.error.HTTPError("http://x/y", 503, "Service Unavailable", {}, None)]


def _eight_times(item):
    """A fetch that returns `item` (bytes) or raises it (an exception) at every request."""
    def fetch(url):
        if isinstance(item, BaseException):
            raise item
        return item
    return fetch


@pytest.mark.parametrize("label, blob, parser, answer", NO_DATA, ids=NO_DATA_IDS)
def test_f1_an_answer_with_no_data_counts_as_a_whole_answer_that_failed_the_validator(label, blob, parser, answer):
    import es.sealed_pull as sp
    with pytest.raises(sp.ChunkError, match="no data after 8 attempts") as err:
        sp._fetch_parse("http://x/y", getattr(sp, parser), "what", _eight_times(blob()), lambda _s: None, key="k")
    assert (err.value.validator_failures, err.value.attempts) == (8, 8)
    assert err.value.reason == "no rows" + (f" ({answer})" if answer else "")
    assert f"no rows for what{': ' + answer if answer else ''}" in str(err.value)


@pytest.mark.parametrize("error", _network_failures(), ids=lambda e: type(e).__name__ + str(getattr(e, "code", "")))
def test_f1_a_network_failure_is_not_a_whole_answer(error):
    """A timeout, a connection error and an HTTP error status stay network failures: the request itself failed."""
    import es.sealed_pull as sp
    with pytest.raises(sp.ChunkError) as err:
        sp._fetch_parse("http://x/y", sp._parse_fmm_answer, "what", _eight_times(error), lambda _s: None, key="k")
    assert (err.value.validator_failures, err.value.attempts, err.value.reason) == (0, 8, None)


@pytest.mark.parametrize("label, blob", [("garbage that is no zip", b"<html>busy</html>"),
                                         ("a zip cut short", _fmm_zip("X", [(T("2026-06-21 07:00"), 1.0)])[:40])])
def test_f1_an_answer_that_cannot_be_read_stays_with_the_network_failures(label, blob):
    import es.sealed_pull as sp
    with pytest.raises(sp.ChunkError) as err:
        sp._fetch_parse("http://x/y", sp._parse_fmm_answer, "what", _eight_times(blob), lambda _s: None, key="k")
    assert (err.value.validator_failures, err.value.reason) == (0, None)


def test_f1_seven_answers_with_no_data_and_one_network_failure_are_seven_of_eight():
    import es.sealed_pull as sp
    items = [_oasis_error_zip()] * 3 + [OSError("reset")] + [_oasis_error_zip()] * 4
    seen = iter(items)

    def fetch(url):
        item = next(seen)
        if isinstance(item, BaseException):
            raise item
        return item

    with pytest.raises(sp.ChunkError) as err:
        sp._fetch_parse("http://x/y", sp._parse_oasis_xml, "what", fetch, lambda _s: None, key="k")
    assert err.value.validator_failures == 7


def test_f1_a_chunk_of_no_data_answers_is_accepted_as_zero_rows_with_what_the_server_said(tmp_path):
    import es.sealed_pull as sp
    s, e, key, validate = _hourly_chunk()
    lines, cache = [], tmp_path / "raw"
    inc = sp.IncompleteChunks("CAISO", {key}, lambda *ls: lines.extend(ls))
    part, cached = sp._fetch_parse("http://x/y", sp._parse_oasis_xml, "what", _eight_times(_oasis_error_zip()),
                                   lambda _s: None, cache, validate=validate, key=key, incomplete=inc)
    assert (len(part), cached) == (0, False) and not cache.exists()
    reason = ("short: 24 of 24 60-minute intervals missing, first 2026-06-21 07:00:00+00:00; "
              f"no rows came ({OASIS_ERROR})")
    assert inc.accepted == {key: reason}
    assert lines == [f"chunk_accepted: CAISO {key} | {reason}",
                     f"chunk_accepted_detail: CAISO {key} | 24 interval(s) of 60 minutes missing: "
                     "2026-06-21T07:00Z .. 2026-06-22T06:00Z (24)"]


def test_f1_a_month_of_no_data_answers_is_accepted_as_zero_rows(tmp_path):
    import es.sealed_pull as sp
    period = pd.Period("2026-06", "M")
    lines = []
    inc = sp.IncompleteChunks("nyiso_rtint", {"2026-06"}, lambda *ls: lines.extend(ls))
    part, cached = sp._fetch_parse("http://x/y", sp._parse_nyiso_answer, "what", _eight_times(_empty_zip()),
                                   lambda _s: None, validate=sp._month_validator(period, True), key="2026-06",
                                   incomplete=inc)
    assert (len(part), cached) == (0, False)
    reason = ("short: the archive of 2026-06 holds no rows (720 hours missing in every zone); "
              "no rows came (the zip holds no CSV file)")
    assert inc.accepted == {"2026-06": reason}
    assert lines == [f"chunk_accepted: nyiso_rtint 2026-06 | {reason}",
                     "chunk_accepted_detail: nyiso_rtint 2026-06 | 720 interval(s) of 60 minutes missing: "
                     "2026-06-01T04:00Z .. 2026-07-01T03:00Z (720)"]


def _scripted(key, script):
    """fetch_override for _use_default_builders: the requests for the OASIS chunk `key` get the items of `script`, one
    each (bytes, a function of the URL, or an exception to raise); when it is used up the chunk is answered in full."""
    items = list(script)

    def override(url):
        if "oasis" in url and _key_of(url) == key and items:
            item = items.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item(url) if callable(item) else item
        return None

    override.items = items
    return override


def _no_data_pull_cases():
    """(id, series, chunk, the answer for its URLs, the reason in the chunk_failed line, the accepted reason's start)"""
    first = "first 2026-06-21 07:00:00+00:00"
    return [
        ("fmm csv without rows", "fmm", FMM_SDGE, lambda url: _fmm_zip(SDGE, []), "no rows",
         f"short: 288 of 288 15-minute intervals missing, {first}; no rows came"),
        ("fmm error body", "fmm", FMM_SDGE, lambda url: _oasis_error_zip(), f"no rows ({OASIS_ERROR})",
         f"short: 288 of 288 15-minute intervals missing, {first}; no rows came ({OASIS_ERROR})"),
        ("dam error body", "CAISO", DAM_SDGE, lambda url: _oasis_error_zip(), f"no rows ({OASIS_ERROR})",
         f"short: 72 of 72 60-minute intervals missing, {first}; no rows came ({OASIS_ERROR})"),
        ("rtm report with no items", "CAISO", RTM_SDGE, lambda url: _oasis_zip(SDGE, []),
         "no rows (the report holds no data items)",
         f"short: 864 of 864 5-minute intervals missing, {first}; no rows came (the report holds no data items)"),
    ]


@pytest.mark.parametrize("case", _no_data_pull_cases(), ids=lambda case: case[0])
def test_f1_a_chunk_never_answered_with_data_can_be_accepted_by_the_two_decisions(registered, monkeypatch, case):
    import es.sealed_pull as sp
    write, tmp = registered
    proto, reg, sha = write()
    days = ["2026-06-21", "2026-06-22", "2026-06-23"]
    label, series, key, answer, failed, accepted = case
    _use_default_builders(monkeypatch, [], fetch_override=lambda url: (
        answer(url) if "oasis" in url and _key_of(url) == key else None))
    with pytest.raises(sp.ChunkError) as err:
        _pull(tmp, sha)
    assert (err.value.key, err.value.validator_failures, err.value.reason) == (key, 8, failed)
    assert f"chunk_failed: {series} {key} | 8 of 8 attempts failed the validator | {failed}\n" in _log_of(tmp)
    with pytest.raises(sp.MissingDaysError, match=series):
        _pull(tmp, sha, resume=True, allow_incomplete_chunks={series: [key]})
    root, _ = _pull(tmp, sha, resume=True, allow_incomplete_chunks={series: [key]}, allow_missing_days={series: days})
    log = (root / "PULL_LOG.txt").read_text()
    assert f"chunk_accepted: {series} {key} | {accepted}" in log
    assert (f"step {series}: 3 (node, day) pair(s) without rows, allowed by allow_missing_days: "
            f"{SDGE}: 2026-06-21 .. 2026-06-23") in log
    record = sp.completed_pulls()["A"]["incomplete_chunks"]
    assert list(record) == [series] and list(record[series]) == [key] and record[series][key].startswith(accepted)
    cov = pd.read_csv(root / "COVERAGE.csv")
    row = cov[(cov["node"] == SDGE) & (cov["series"] == ("fmm" if series == "fmm" else "price"))].iloc[0]
    assert (row["hours"], row["nonfinite_rows"]) == (0, 0)


def test_f1_a_nyiso_month_never_answered_with_data_can_be_accepted_by_the_two_decisions(registered, monkeypatch):
    import es.sealed_pull as sp
    write, tmp = registered
    proto, reg, sha = write()
    _use_default_builders(monkeypatch, [], fetch_override=lambda url: _empty_zip() if "mis.nyiso.com" in url else None)
    with pytest.raises(sp.ChunkError) as err:
        _pull(tmp, sha)
    assert (err.value.key, err.value.validator_failures) == ("2026-06", 8)
    assert ("chunk_failed: nyiso_rtint 2026-06 | 8 of 8 attempts failed the validator | "
            "no rows (the zip holds no CSV file)\n") in _log_of(tmp)
    days = ["2026-06-21", "2026-06-22", "2026-06-23"]
    root, _ = _pull(tmp, sha, resume=True, allow_incomplete_chunks={"nyiso_rtint": ["2026-06"]},
                    allow_missing_days={"nyiso_rtint": days})
    log = (root / "PULL_LOG.txt").read_text()
    assert "step nyiso_rtint: 30 (node, day) pair(s) without rows, allowed by allow_missing_days:" in log
    cov = pd.read_csv(root / "COVERAGE.csv")
    assert (cov[cov["series"] == "nyiso_rtint"]["hours"] == 0).all()
    assert list(sp.completed_pulls()["A"]["incomplete_chunks"]["nyiso_rtint"]) == ["2026-06"]


def test_f1_a_step_whose_every_chunk_is_accepted_as_empty_still_writes_its_tables(tmp_path):
    import es.sealed_pull as sp
    kw = dict(fetch=lambda url: _oasis_error_zip(), **NO_SLEEP)
    inc = sp.IncompleteChunks("CAISO", sp._chunk_keys("CAISO", *SPAN), lambda *ls: None)
    price = pd.read_parquet(sp.caiso_exact(*SPAN, tmp_path / "caiso", incomplete=inc, **kw))
    assert len(price) == 0 and list(price.columns) == PRICE_COLUMNS
    assert sorted(inc.accepted) == sorted(sp._chunk_keys("CAISO", *SPAN))
    inc = sp.IncompleteChunks("fmm", sp._chunk_keys("fmm", *SPAN), lambda *ls: None)
    fmm = pd.read_parquet(sp.caiso_fmm_exact(*SPAN, tmp_path / "fmm", incomplete=inc, **kw))
    assert len(fmm) == 0 and list(fmm.columns) == ["node", "timestamp_utc", "fmm_lmp", "n15"]
    inc = sp.IncompleteChunks("nyiso_rtint", ["2026-06"], lambda *ls: None)
    nyiso = pd.read_parquet(sp.nyiso_integrated_months("2026-06", "2026-06", tmp_path / "nyiso", incomplete=inc,
                                                       fetch=lambda url: _empty_zip(), **NO_SLEEP))
    assert len(nyiso) == 0 and list(nyiso.columns) == ["node", "timestamp_utc", "rt_int_lmp"]


def _legacy_fetch_parse(url, parse, what, fetch, sleep, cache_dir=None, attempts=8, validate=None):
    """_fetch_parse as it was before Ruling 111a, verbatim: the reference of the normal path."""
    import es.sealed_pull as sp
    path = Path(cache_dir) / f"{hashlib.sha256(url.encode()).hexdigest()}.bin" if cache_dir else None
    if path is not None and path.exists():
        try:
            part = parse(path.read_bytes())
            why = "holds no rows" if not len(part) else (validate(part) if validate else None)
        except (AssertionError, sp.SealedDataError):
            raise
        except Exception as exc:
            why = f"does not parse ({exc!r})"
        if why is None:
            print(f"{sp._stamp_now()} {what}: cache hit {path.name}", flush=True)
            return part, True
        print(f"{sp._stamp_now()} {what}: cached response {path.name} {why}; fetching it again", flush=True)
    last = None
    for k in range(attempts):
        try:
            blob = fetch(url)
            part = parse(blob)
            if not len(part):
                last = RuntimeError(f"no rows for {what}")
            else:
                why = validate(part) if validate else None
                if why is None:
                    if path is not None:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        sp._atomic_write(path, blob)
                    print(f"{sp._stamp_now()} {what}: {len(blob):,} bytes on attempt {k + 1}", flush=True)
                    return part, False
                last = RuntimeError(f"{what}: {why}")
        except (AssertionError, sp.SealedDataError):
            raise
        except Exception as exc:                          # error bodies and network failures alike
            last = exc
        print(f"{sp._stamp_now()} {what}: attempt {k + 1}/{attempts} failed: {last!r}", flush=True)
        if k < attempts - 1:
            sleep(min(60, 6 * 2 ** k))
    raise RuntimeError(f"{what}: no data after {attempts} attempts; the last problem: {last}") from last


@pytest.mark.parametrize("scenario", [
    "a complete answer", "a cache hit", "a cached answer that is short", "a network failure, then a complete answer",
    "an empty answer, then a complete one", "a short answer, then a complete one", "a complete answer, no cache",
    "a damaged cache entry"])
def test_f1_the_normal_path_is_unchanged_from_the_fetch_before_ruling_121(tmp_path, capsys, scenario):
    """The same scripted requests through the function before the ruling and through the one now: the same rows, the
    same cache flag, the same requests and pauses, the same files in the cache byte for byte, the same log lines."""
    import es.sealed_pull as sp
    s, e, key, validate = _hourly_chunk()
    full = _oasis_zip("X", [(t, 10.0) for t, _ in _grid(s, e, 60, lambda t: 0)])
    short = _oasis_zip("X", [(t, 10.0) for t, _ in _grid(s, e, 60, lambda t: 0)][:20])
    plan = {"a complete answer": ([full], None), "a cache hit": ([], full),
            "a cached answer that is short": ([full], short),
            "a network failure, then a complete answer": ([OSError("reset"), full], None),
            "an empty answer, then a complete one": ([_oasis_zip("X", []), full], None),
            "a short answer, then a complete one": ([short, full], None),
            "a complete answer, no cache": ([full], False), "a damaged cache entry": ([full], b"not a zip")}[scenario]
    answers, cached_entry = plan

    def run(function, name):
        cache = None if cached_entry is False else tmp_path / name
        if cached_entry:
            cache.mkdir()
            (cache / (hashlib.sha256(b"http://x/y").hexdigest() + ".bin")).write_bytes(cached_entry)
        items, calls, pauses = list(answers), [], []

        def fetch(url):
            calls.append(url)
            item = items.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item

        capsys.readouterr()
        part, flag = function("http://x/y", sp._parse_oasis_xml, "what", fetch, pauses.append, cache, validate=validate)
        printed = re.sub(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ ", "", capsys.readouterr().out, flags=re.M)
        files = {} if cache is None or not cache.exists() else {p.name: p.read_bytes() for p in sorted(cache.glob("*"))}
        return part, flag, calls, pauses, files, printed

    new, old = run(sp._fetch_parse, "new"), run(_legacy_fetch_parse, "old")
    if scenario == "an empty answer, then a complete one":    # the one change: the line of the empty answer says why
        new = (*new[:5], new[5].replace("no rows for what: the report holds no data items", "no rows for what"))
    assert new[0].equals(old[0]) and new[1:] == old[1:]
    assert old[2] == ["http://x/y"] * len(answers)                          # the script was played as written


def test_f1_the_downloaders_give_the_same_tables_for_complete_answers(tmp_path):
    """Value for value, with no pause and one request per chunk: the tables of the three downloaders for complete
    answers. (The June check against the development files, bit for bit, was run on this path before the change.)"""
    import es.sealed_pull as sp
    pauses, calls = [], []

    def fetch(url):
        calls.append(url)
        return _good_oasis(url)

    kw = dict(fetch=fetch, pause=0, sleep=pauses.append)
    price = pd.read_parquet(sp.caiso_exact("2026-06-21", "2026-06-22", tmp_path / "a", **kw))
    nodes = sorted(sp.CAISO_NODE_TYPE)
    expected = pd.DataFrame(
        [("CAISO", n, sp.CAISO_NODE_TYPE[n], t, 10.0, 10.0) for n in nodes
         for t in pd.date_range(T("2026-06-21 07:00"), T("2026-06-23 07:00"), freq="h", inclusive="left")],
        columns=PRICE_COLUMNS)
    pd.testing.assert_frame_equal(price, expected, check_exact=True)
    fmm = pd.read_parquet(sp.caiso_fmm_exact("2026-06-21", "2026-06-22", tmp_path / "b", **kw))
    hours = pd.date_range(T("2026-06-21 07:00"), T("2026-06-23 07:00"), freq="h", inclusive="left")
    expected = pd.DataFrame([(n, t, 100.0, 4) for n in nodes for t in hours],
                            columns=["node", "timestamp_utc", "fmm_lmp", "n15"])
    pd.testing.assert_frame_equal(fmm, expected, check_exact=True)
    assert len(calls) == 6 * 2 + 6 and pauses == [0] * 18                  # one request per chunk, none retried
    nyiso = pd.read_parquet(sp.nyiso_integrated_months(
        "2026-06", "2026-06", tmp_path / "c", fetch=lambda url: _nyiso_month_blob("202606", *SPAN), **NO_SLEEP))
    zones = _nodes_by_market()["NYISO"]
    expected = pd.DataFrame([(z, t.tz_convert("UTC"), 5.0) for t in _clock_hours("NYISO", *_around(*SPAN))
                             if f"{t.tz_convert('America/New_York'):%Y%m}" == "202606" for z in zones],
                            columns=["node", "timestamp_utc", "rt_int_lmp"])
    pd.testing.assert_frame_equal(nyiso, expected, check_exact=True)


# ----------------------------------------------------------------------------- F2: the accepting call
def test_f2_one_whole_answer_then_seven_network_failures_is_not_accepted(registered, monkeypatch):
    """The reviewer's B5. A chunk that failed 8 of 8 earlier is listed; during the resume the network goes away after
    the first (short) answer, and OASIS may meanwhile have published the whole chunk. Nothing is accepted."""
    import es.sealed_pull as sp
    write, tmp = registered
    proto, reg, sha = write()
    short, net = (lambda url: _good_oasis(url, days=1)), OSError("nodename nor servname provided")
    override = _scripted(FMM_SDGE, [short] * 8)
    _use_default_builders(monkeypatch, [], fetch_override=lambda url: override(url))
    with pytest.raises(sp.ChunkError):
        _pull(tmp, sha)                                                      # attempt 1: 8 of 8, the chunk is open
    listed = {"allow_incomplete_chunks": {"fmm": [FMM_SDGE]}}
    override.items[:] = [short] + [net] * 7
    with pytest.raises(sp.ChunkError, match="no data after 8 attempts") as err:
        _pull(tmp, sha, resume=True, **listed)
    assert (err.value.validator_failures, err.value.attempts) == (1, 8)
    assert f"chunk_failed: fmm {FMM_SDGE} | 1 of 8 attempts failed the validator | {SHORT_FMM}\n" in _log_of(tmp)
    assert "chunk_accepted" not in _log_of(tmp) and _state_of(tmp)["incomplete_chunks"] == {}
    override.items[:] = [net] + [short] * 7                                  # the first request failing is the same
    with pytest.raises(sp.ChunkError) as err:
        _pull(tmp, sha, resume=True, **listed)
    assert err.value.validator_failures == 7 and "chunk_accepted" not in _log_of(tmp)
    override.items[:] = [short] * 3 + [net] + [short] * 4                    # one failure anywhere in the call
    with pytest.raises(sp.ChunkError) as err:
        _pull(tmp, sha, resume=True, **listed)
    assert err.value.validator_failures == 7 and "chunk_accepted" not in _log_of(tmp)
    override.items[:] = [short] * 8                                          # eight whole answers: now it is taken
    with pytest.raises(sp.MissingDaysError):                                 # (the days still have to be excused)
        _pull(tmp, sha, resume=True, **listed)
    assert f"chunk_accepted: fmm {FMM_SDGE} | {SHORT_FMM}\n" in _log_of(tmp)
    assert f"chunk_failed: fmm {FMM_SDGE} | 8 of 8 attempts failed the validator" in _log_of(tmp)     # still there


def test_f2_the_answer_with_the_fewest_problems_is_the_one_accepted_and_the_last_of_equals_wins(tmp_path, capsys):
    import es.sealed_pull as sp
    s, e, key, validate = _hourly_chunk()
    hours = [t for t, _ in _grid(s, e, 60, lambda t: 0)]
    lines, sent = [], []
    missing = [6, 3, 5, 1, 4, 2, 1, 7]            # hours missing from the eight answers: the 4th and 7th are best

    def fetch(url):
        sent.append(url)
        return _oasis_zip("X", [(t, 10.0 + len(sent)) for t in hours[:24 - missing[len(sent) - 1]]])    # price: attempt

    inc = sp.IncompleteChunks("CAISO", {key}, lambda *ls: lines.extend(ls))
    part, cached = sp._fetch_parse("http://x/y", sp._parse_oasis_xml, "what", fetch, lambda _s: None, tmp_path / "raw",
                                   validate=validate, key=key, incomplete=inc)
    assert len(sent) == 8 and cached is False and not (tmp_path / "raw").exists()
    assert len(part) == 23 and set(part["lmp"]) == {17.0}                # the 7th: the last of two equals, not the 8th
    assert inc.accepted == {key: "short: 1 of 24 60-minute intervals missing, first 2026-06-22 06:00:00+00:00"}
    assert lines[1] == f"chunk_accepted_detail: CAISO {key} | 1 interval(s) of 60 minutes missing: 2026-06-22T06:00Z"
    assert ("what: accepted as incomplete, the answer with the fewest problems of 8 (short: 1 of 24 60-minute "
            "intervals missing, first 2026-06-22 06:00:00+00:00); not cached\n") in capsys.readouterr().out


def test_f2_the_error_of_a_chunk_names_the_last_problem_and_how_many_answers_were_refused(tmp_path):
    import es.sealed_pull as sp
    s, e, key, validate = _hourly_chunk()
    hours = [t for t, _ in _grid(s, e, 60, lambda t: 0)]
    missing = iter([6, 3, 5, 1, 4, 2, 1, 7])
    with pytest.raises(sp.ChunkError) as err:
        sp._fetch_parse("http://x/y", sp._parse_oasis_xml, "what", lambda url: _oasis_zip(
            "X", [(t, 1.0) for t in hours[:24 - next(missing)]]), lambda _s: None, validate=validate, key=key)
    assert err.value.reason == "short: 7 of 24 60-minute intervals missing, first 2026-06-22 00:00:00+00:00"
    assert err.value.validator_failures == 8


def test_f2_facts_that_cannot_be_read_are_no_reason_to_accept_and_no_error(tmp_path):
    import es.sealed_pull as sp
    s, e, key, real = _hourly_chunk()
    hours = [t for t, _ in _grid(s, e, 60, lambda t: 0)]

    def check(part):
        return real(part)

    def facts(part):
        if part.attrs.get("odd"):
            raise KeyError("timestamp_utc")                  # a table the facts cannot be read from
        return real.problems(part)

    check.problems, check.texts, check.step = facts, real.texts, real.step

    def parse(blob):
        table = sp._parse_oasis_xml(blob)
        table.attrs["odd"] = len(table) == 23                # the one answer with a single hour missing
        return table

    inc = sp.IncompleteChunks("CAISO", {key}, lambda *ls: None)
    odd = _oasis_zip("X", [(t, 1.0) for t in hours[:23]])
    plain = _oasis_zip("X", [(t, 1.0) for t in hours[:20]])
    answers = iter([odd] + [plain] * 7)
    part, _ = sp._fetch_parse("http://x/y", parse, "what", lambda url: next(answers), lambda _s: None, validate=check,
                              key=key, incomplete=inc)
    assert len(part) == 20                                   # the odd one misses the fewest hours: never the best
    inc = sp.IncompleteChunks("CAISO", {key}, lambda *ls: None)
    answers = iter([odd] * 8)
    with pytest.raises(sp.ChunkError, match="no data after 8 attempts"):       # all odd: nothing to accept, no KeyError
        sp._fetch_parse("http://x/y", parse, "what", lambda url: next(answers), lambda _s: None, validate=check,
                        key=key, incomplete=inc)
    assert inc.accepted == {} and inc.take(key, parse(odd), check) is False
    with pytest.raises(sp.ChunkError, match="no data after 0 attempts"):       # no attempt, no answer: nothing to take
        sp._fetch_parse("http://x/y", parse, "what", lambda url: b"", lambda _s: None, validate=check, key=key,
                        incomplete=inc, attempts=0)


def test_f2_a_better_answer_before_a_worse_one_is_kept_and_rows_outside_the_interval_rank_last(tmp_path):
    import es.sealed_pull as sp
    s, e, key, validate = _hourly_chunk()
    hours = [t for t, _ in _grid(s, e, 60, lambda t: 0)]
    full_plus_one = _oasis_zip("X", [(t, 1.0) for t in hours] + [(e, 1.0)])      # 24 hours and a row outside: worst
    five_short = _oasis_zip("X", [(t, 2.0) for t in hours[:19]])
    ten_short = _oasis_zip("X", [(t, 3.0) for t in hours[:14]])
    empty = _oasis_error_zip()
    five_nan = _oasis_zip("X", [(t, NAN if i < 5 else 4.0) for i, t in enumerate(hours)])    # 24 hours, 5 not finite
    two_short = _oasis_zip("X", [(t, 5.0) for t in hours[:22]])
    for order, kept in (([five_nan, two_short] * 4, 22), ([two_short, five_nan] * 4, 22),
                        ([five_short, ten_short, empty, full_plus_one] * 2, 19),
                        ([ten_short, full_plus_one, empty, five_short] + [empty] * 4, 19),
                        ([empty] * 7 + [ten_short], 14), ([ten_short] + [empty] * 7, 14)):
        items = iter(order)
        inc = sp.IncompleteChunks("CAISO", {key}, lambda *ls: None)
        part, _ = sp._fetch_parse("http://x/y", sp._parse_oasis_xml, "what", lambda url: next(items), lambda _s: None,
                                  validate=validate, key=key, incomplete=inc)
        assert len(part) == kept
    items = iter([full_plus_one] * 8)                                          # every answer has a row outside: refused
    inc = sp.IncompleteChunks("CAISO", {key}, lambda *ls: None)
    with pytest.raises(sp.ChunkError):
        sp._fetch_parse("http://x/y", sp._parse_oasis_xml, "what", lambda url: next(items), lambda _s: None,
                        validate=validate, key=key, incomplete=inc)
    assert inc.accepted == {}


# ----------------------------------------------------------------------------- F4: a cached NYISO month that is short
def _nyiso_month_without(month, zone, stamp):
    """The archive of _nyiso_month_blob for 'YYYYMM' without the record of `zone` at the local time `stamp`."""
    by_day = {}
    for t in _clock_hours("NYISO", *_around(*SPAN)).tz_convert("America/New_York"):
        if f"{t:%Y%m}" == month:
            when = f"{t:%m/%d/%Y %H:%M}"
            by_day.setdefault(f"{t:%Y%m%d}", []).extend(
                (when, z, 5.0) for z in _nodes_by_market()["NYISO"] if (z, when) != (zone, stamp))
    return _nyiso_zip(by_day)


def test_f4_a_cached_nyiso_month_that_is_short_says_how_to_have_it_fetched_again(registered, monkeypatch):
    import es.sealed_pull as sp
    write, tmp = registered
    proto, reg, sha = write()
    urls, source = [], {"short": True}
    _use_default_builders(monkeypatch, urls, fetch_override=lambda url: (
        _nyiso_month_without("202606", "N.Y.C.", "06/22/2026 05:00") if "mis.nyiso.com" in url and source["short"]
        else None))
    named = f"raw/{hashlib.sha256(sp.NYISO_URL.format(ym='202606').encode()).hexdigest()}.bin"
    gap = (r"nyiso_rtint: fewer usable hours than the day has on 1 \(node, day\) pair\(s\): "
           r"N\.Y\.C\.: 2026-06-22 \(23 of 24 hours\)")
    with pytest.raises(sp.MissingDaysError, match=gap) as err:
        _pull(tmp, sha)
    message = " ".join(str(err.value).split())
    assert (f"The archive of a month is kept in raw/ and reused as it is on every resume: if NYISO has since published "
            f"the missing hours, delete {named} (the archive of 2026-06) in the pull root, and the next resume fetches "
            "the month again") in message
    assert (tmp / "pull_unit" / named).is_file()
    root = tmp / "pull_unit"
    assert sp._cached_archive_hint(root, "fmm", [(SDGE, dt.date(2026, 6, 22))]) == ""          # not for another series
    july = [("N.Y.C.", dt.date(2026, 7, 1))]
    assert sp._cached_archive_hint(root, "nyiso_rtint", july) == ""                       # nor a month not cached
    requests = lambda: len([u for u in urls if "mis.nyiso.com" in u])
    source["short"] = False                                                   # NYISO has filled the gap
    with pytest.raises(sp.MissingDaysError, match=gap):
        _pull(tmp, sha, resume=True)
    assert requests() == 1                                                    # the caching is what it was: reused
    (tmp / "pull_unit" / named).unlink()                                      # what the message says to do
    root, _ = _pull(tmp, sha, resume=True)
    assert requests() == 2 and (root / "MANIFEST.sha256").exists()
    assert (root / named).is_file()                                           # the new archive is cached in its place


def test_f4_the_sentence_names_every_cached_month_behind_a_gap(tmp_path):
    import es.sealed_pull as sp
    names = {ym: f"raw/{hashlib.sha256(sp.NYISO_URL.format(ym=ym).encode()).hexdigest()}.bin" for ym in
             ("202606", "202607", "202608")}
    (tmp_path / "raw").mkdir()
    for ym in ("202606", "202607"):                                           # August is not cached
        (tmp_path / names[ym]).write_bytes(b"x")
    pairs = [("N.Y.C.", dt.date(2026, 7, 1)), ("WEST", dt.date(2026, 6, 30)), ("WEST", dt.date(2026, 7, 2)),
             ("WEST", dt.date(2026, 8, 3))]
    assert sp._cached_archive_hint(tmp_path, "nyiso_rtint", pairs) == (
        " The archive of a month is kept in raw/ and reused as it is on every resume: if NYISO has since published the "
        f"missing hours, delete {names['202606']} (the archive of 2026-06) and {names['202607']} (the archive of "
        "2026-07) in the pull root, and the next resume fetches those months again.")
    assert sp._cached_archive_hint(tmp_path, "nyiso_rtint", []) == ""


def test_f4_only_the_months_behind_gaps_that_are_not_excused_are_named(tmp_path):
    import es.sealed_pull as sp
    names = {ym: f"raw/{hashlib.sha256(sp.NYISO_URL.format(ym=ym).encode()).hexdigest()}.bin" for ym in
             ("202606", "202607")}
    (tmp_path / "raw").mkdir()
    for name in names.values():
        (tmp_path / name).write_bytes(b"x")
    with pytest.raises(sp.MissingDaysError) as err:           # no table at all: every zone has no rows on both days
        sp._require_complete(tmp_path, "nyiso_rtint", "2026-06-30", "2026-07-01", {"2026-07-01"})
    assert names["202606"] in str(err.value) and names["202607"] not in str(err.value)       # July's gaps are excused


def test_f4_other_series_and_uncached_months_get_no_such_sentence(registered, monkeypatch):
    import es.sealed_pull as sp
    write, tmp = registered
    proto, reg, sha = write()
    _use_default_builders(monkeypatch, [], drop={("ISONE", ".Z.MAINE", "2026-06-22", 12)})
    with pytest.raises(sp.MissingDaysError, match="ISONE: fewer usable hours") as err:
        _pull(tmp, sha)
    assert "raw/" not in str(err.value) and "fetches the month again" not in str(err.value)
    assert sp._cached_archive_hint(tmp / "pull_unit", "nyiso_rtint", [("N.Y.C.", dt.date(2026, 6, 22))]) == ""
    assert sp._cached_archive_hint(tmp / "pull_unit", "fmm", [(SDGE, dt.date(2026, 6, 22))]) == ""


# ----------------------------------------------------------------------------- Ruling 122, C2-1: a ready protocol
def test_c2_pull_sealed_refuses_a_protocol_that_is_not_ready_before_anything_is_spent(registered, monkeypatch):
    """The gate refuses a PROTOCOL.md that still holds a marker: no lock, no folder, no request."""
    import es.sealed_pull as sp
    from es.registration import RegistrationError
    write, tmp = registered
    proto, reg, sha = write("protocol v1\n\nName and date: [AUTHOR: Erce Erez, date]\n")
    urls, calls = [], []
    _use_default_builders(monkeypatch, urls)
    for how in ("the default builders", "injected builders"):
        with pytest.raises(RegistrationError, match=r"is not ready to register: 1 item still open"):
            if how == "the default builders":
                sp.pull_sealed("A", sha, data_root=tmp, reference_dirs=[])
            else:
                _a_pull(tmp, sha, calls)
    assert urls == [] and calls == []
    assert not (tmp / "pull_unit").exists() and not (tmp / "sealed_2026Q3").exists()
    assert sorted(p.name for p in (tmp / "docs_sealed").iterdir()) == []       # no lock, no PULLS.json
    proto, reg, sha = write("protocol v1\n\nName and date: Erce Erez, 24 October 2026\n")   # filled in: a new SHA-256
    root, manifest = _a_pull(tmp, sha, calls)
    assert len(calls) == 6 and (root / "MANIFEST.sha256").exists() and sp.completed_pulls()["A"]["root"] == str(root)



# =====================================================================================================
# Ruling 140 (26 Oct 2026): a pull is bound to an OSF registration, a public release or a local record. osf_id is
# renamed registration_id, and a registration_kind (osf, public_release, local) is kept wherever the identifier is.
# =====================================================================================================
RELEASE_URL = "https://github.com/ee2625/dart-spread/releases/tag/v1.0"
LOCAL_TOKEN = "author-statement-2026-10-26"
#: The `registered` fixture's record in each of the three kinds (None takes its default DOI line out).
KINDS = {"osf": {}, "public_release": {"osf_doi": None, "public_record": RELEASE_URL},
         "local": {"osf_doi": None, "local_record": LOCAL_TOKEN}}
#: (registration id, the identifier to log, the key that holds it) of each kind.
FACTS = {"osf": ("abcde", DOI, "osf_doi"),
         "public_release": ("github:ee2625/dart-spread/releases/tag/v1.0", RELEASE_URL, "public_record"),
         "local": ("local:author-statement-2026-10-26", LOCAL_TOKEN, "local_record")}
IDENTIFIER_KEYS = ("osf_doi", "osf_registration", "public_record", "local_record")
PAIRS = [(a, b) for a in KINDS for b in KINDS if a != b]


@pytest.mark.parametrize("kind", list(KINDS))
def test_a_pull_records_the_id_the_kind_and_the_identifier_of_its_registration(registered, kind):
    from es.sealed_pull import completed_pulls
    write, tmp = registered
    proto, reg, sha = write(reg_fields=KINDS[kind])
    builders = _failing_then_fine_builders()
    with pytest.raises(OSError):
        _a_pull(tmp, sha, builders=builders)
    state = json.loads((tmp / "sealed_2026Q3" / ".state.json").read_text())
    root, manifest = _a_pull(tmp, sha, builders=builders, resume=True)
    record = completed_pulls()["A"]
    rid, identifier, key = FACTS[kind]
    for where in (state, record):                                         # .state.json of the first attempt, PULLS.json
        assert (where["registration_id"], where["registration_identifier"], where["registration_kind"]) == (
            rid, identifier, kind)
        assert where[key] == identifier and [k for k in IDENTIFIER_KEYS if where[k] is not None] == [key]
        assert "osf_id" not in where
    log = (root / "PULL_LOG.txt").read_text()
    assert log.count(f"registration_id: {rid}\n") == 2 and log.count(f"registration_kind: {kind}\n") == 2   # per attempt
    assert log.count(f"registration_identifier: {identifier}\n") == 2 and log.count(f"{key}: {identifier}\n") == 2
    assert log.count("none\n") >= 2 * 3 and "osf_id" not in log          # the three other identifiers are logged as none
    assert record["protocol_sha256"] == sha and record["manifest_sha256"] == manifest


@pytest.mark.parametrize("first, second", PAIRS)
def test_a_resume_under_another_kind_of_registration_is_refused(registered, first, second):
    """The same protocol, recorded another way, is another registration: the pull that began under one does not go on under
    the other (as it did not under another OSF id), and goes on under its own."""
    from es.registration import RegistrationError
    write, tmp = registered
    proto, reg, sha = write(reg_fields=KINDS[first])
    builders = _failing_then_fine_builders()
    with pytest.raises(OSError):
        _a_pull(tmp, sha, builders=builders)
    write(reg_fields=KINDS[second])
    assert write(reg_fields=KINDS[second])[2] == sha                       # the same protocol hash
    calls = []
    with pytest.raises(RegistrationError, match="is another registration") as err:
        _a_pull(tmp, sha, calls, resume=True)
    assert FACTS[first][1] in str(err.value) and FACTS[second][1] in str(err.value) and calls == []   # it names both
    write(reg_fields=KINDS[first])
    root, _ = _a_pull(tmp, sha, builders=builders, resume=True)
    assert (root / "MANIFEST.sha256").exists()


@pytest.mark.parametrize("first, second", PAIRS)
def test_a_finished_pull_is_adopted_under_its_own_kind_and_refused_under_another(registered, monkeypatch, first, second):
    from es.registration import RegistrationError
    from es.sealed_pull import completed_pulls
    write, tmp = registered
    proto, reg, sha = write(reg_fields=KINDS[first])
    real = os.replace

    def killed(src, dst):
        if Path(dst).name == "PULLS.json":
            raise KeyboardInterrupt("killed while PULLS.json was being written")
        return real(src, dst)

    monkeypatch.setattr(os, "replace", killed)
    with pytest.raises(KeyboardInterrupt):
        _a_pull(tmp, sha)
    monkeypatch.setattr(os, "replace", real)
    write(reg_fields=KINDS[second])
    with pytest.raises(RegistrationError, match="is another registration"):
        _a_pull(tmp, sha, resume=True)
    assert completed_pulls() == {}
    write(reg_fields=KINDS[first])
    root, manifest = _a_pull(tmp, sha, resume=True)
    record = completed_pulls()["A"]
    assert (record["registration_kind"], record["registration_id"], record["manifest_sha256"]) == (
        first, FACTS[first][0], manifest)


def test_a_state_that_names_two_kinds_is_matched_on_neither(registered):
    """A state (or a log) that carries the identifiers of two kinds was not written by a pull: it is refused."""
    from es.registration import RegistrationError
    write, tmp = registered
    proto, reg, sha = write(reg_fields=KINDS["local"])
    builders = _failing_then_fine_builders()
    with pytest.raises(OSError):
        _a_pull(tmp, sha, builders=builders)
    state_path = tmp / "sealed_2026Q3" / ".state.json"
    state = json.loads(state_path.read_text())
    state["public_record"] = RELEASE_URL                                     # beside the local record
    state_path.write_text(json.dumps(state))
    with pytest.raises(RegistrationError, match=r"\.state\.json was started under .* is another registration"):
        _a_pull(tmp, sha, builders=builders, resume=True)


@pytest.mark.parametrize("kind", ["public_release", "local"])
def test_a_state_without_any_identifier_of_the_new_kinds_is_refused_on_resume(registered, kind):
    from es.registration import RegistrationError
    write, tmp = registered
    proto, reg, sha = write(reg_fields=KINDS[kind])
    builders = _failing_then_fine_builders()
    with pytest.raises(OSError):
        _a_pull(tmp, sha, builders=builders)
    state_path = tmp / "sealed_2026Q3" / ".state.json"
    state = json.loads(state_path.read_text())
    state_path.write_text(json.dumps({k: v for k, v in state.items() if k not in IDENTIFIER_KEYS}))
    with pytest.raises(RegistrationError, match="does not say which registration"):
        _a_pull(tmp, sha, builders=builders, resume=True)
