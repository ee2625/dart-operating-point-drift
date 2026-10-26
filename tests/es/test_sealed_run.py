import datetime as dt
import json
import os
import re
import shutil
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

DEV_ROOT = Path(__file__).resolve().parents[2]
FAMILY = {h: {"confirmatory": True, "reasons": [], "approximate": False, "interim_approximate": False}
          for h in ("H-E", "H-B1", "H-B2")}
#: One node of v2/NODE_MAP.csv per market (restrict_to_node_map drops any other name).
NODE = {"NYISO": "N.Y.C.", "ISONE": ".H.INTERNAL_HUB", "MISO": "ILLINOIS.HUB", "CAISO": "TH_SP15_GEN-APND"}


REAL_SEALED_DIR = DEV_ROOT.parent / "docs" / "sealed"
REAL_SEALED_OUT = DEV_ROOT / "outputs" / "es" / "sealed"


@pytest.fixture(autouse=True)
def _the_real_sealed_files_are_left_alone():
    """The pull and run records, the pull locks, the registration and the real sealed output folder are not for tests: no
    test here creates, changes or removes them (the `registered` fixture points the gate at a temporary copy)."""
    def snapshot():
        names = ("PULLS.json", "RUNS.json", "REGISTRATION.md", "PULL_A.lock", "PULL_B.lock", "RUN_A.claim", "RUN_B.claim",
                 "ENV_PINS.json")
        files = {n: (REAL_SEALED_DIR / n).stat().st_mtime_ns for n in names if (REAL_SEALED_DIR / n).exists()}
        return files, sorted(p.name for p in REAL_SEALED_OUT.glob("*")) if REAL_SEALED_OUT.exists() else None

    before = snapshot()
    yield
    assert snapshot() == before, "a test wrote to the real docs/sealed or outputs/es/sealed"


@pytest.fixture(autouse=True)
def _the_registered_output_folders_are_under_tmp(tmp_path, monkeypatch):
    """default_config(t).out_dir is outputs/es/sealed/A and /B, and a run with a sealed arm writes only there (Ruling 122, N2b).
    Every test here moves them under its own tmp, as it moves N_REP and DEV_DATA_DIRS, so that nothing in this module can
    create the real folders: tmp/outputs_es/sealed/A and /B."""
    from es import sealed_run
    monkeypatch.setattr(sealed_run, "OUT_DIR", tmp_path / "outputs_es")


def _out(tmp, test_id="A"):
    """The registered output folder of a sealed test, as default_config names it in these tests."""
    return tmp / "outputs_es" / "sealed" / test_id


def _protocol_lines(cut, inp, man):
    """The lines es.registration.verify_bound looks for: '<file name> SHA-256: <hash>'. (The brief's draft printed
    'cutoffs <hash>' and so on, which is not the format the landed gate reads.)"""
    from es.registration import sha256_file
    return "".join(f"{p.name} SHA-256: {sha256_file(p)}\n" for p in (cut, inp, man))


def _side_tables(tmp):
    """The development settlement tables, by name only: the pre-lock check (Ruling 109, M9) lists the folders and reads no row,
    and a test that stops at the read never opens them. (The slow tests write real ones.)"""
    for sub, name in (("caiso_fmm_dev", "caiso_fmm_x.parquet"), ("nyiso_rtint_dev", "nyiso_rtint_x.parquet")):
        (tmp / sub).mkdir(exist_ok=True)
        (tmp / sub / name).write_bytes(b"")


def _bound(tmp, write, protocol_extra="", real_code=False):
    """A cutoff table, an inputs file and a code manifest, all bound in a temporary protocol. The manifest lists two small
    files of tmp/pkg (the code_root), or, with real_code, two files of this repository (code_root = the folder es was imported
    from, which a sealed run requires, Ruling 109 M8). Returns (proto, reg, sha, cfg_kwargs)."""
    from es.registration import sha256_file
    cut, inp, man = tmp / "cutoffs.csv", tmp / "inputs.json", tmp / "CODE_MANIFEST.sha256"
    cut.write_text("analysis\n")
    inp.write_text(json.dumps({"eta": 0.01, "family": FAMILY}))
    if real_code:
        man.write_text("".join(f"{sha256_file(DEV_ROOT / rel)}  {rel}\n" for rel in ("es/cutoffs.py", "es/sealed_run.py")))
    else:
        (tmp / "pkg").mkdir()
        (tmp / "pkg" / "a.py").write_text("A = 1\n")
        (tmp / "pkg" / "b.py").write_text("B = 2\n")
        man.write_text("".join(f"{sha256_file(tmp / 'pkg' / f)}  {f}\n" for f in ("a.py", "b.py")))
    _side_tables(tmp)
    proto, reg, sha = write(_protocol_lines(cut, inp, man) + protocol_extra)
    return proto, reg, sha, dict(registration=(proto, reg), cutoffs=cut, inputs=inp, code_manifest=man,
                                 code_root=DEV_ROOT if real_code else tmp / "pkg", data_root=tmp)


def _cfg_a(tmp, kw, **over):
    """A rehearsal of Test A: the analog arm AQ1 on both sides and no sealed pull, whose once-only records are written beside
    its out_dir (here tmp itself), never in docs/sealed (Ruling 109, C1)."""
    from es.analyses import ANALOGS
    from es.sealed_run import SealedConfig
    arm = ANALOGS["AQ1"]
    return SealedConfig(**{"test_id": "A", "rolling": arm, "frozen": arm, "sealed_dirs": [], "window_end": "2026-03-31",
                           "out_dir": tmp / "out", **kw, **over})


def _sealed_cfg(tmp, kw, test_id="A", **over):
    """The registered config of a sealed test (default_config), pointed at temporary files. `kw` should come from
    _bound(..., real_code=True). The sealed pull itself is not there: tests that go past check_sealed_dirs either stub it
    (_no_pull_check) or make a fake pull with _fake_pull."""
    from es.sealed_run import default_config
    cfg = default_config(test_id)
    assert Path(cfg.out_dir) == _out(tmp, test_id)         # the registered folders (the autouse fixture above), under tmp
    for name, value in {**kw, **over}.items():
        setattr(cfg, name, value)
    return cfg


def _no_pull_check(monkeypatch):
    """check_sealed_dirs is tested on its own; a sealed-arm test of something else stands it in with the manifest hashes the
    sealed pulls would have."""
    from es import sealed_run
    monkeypatch.setattr(sealed_run, "check_sealed_dirs", lambda cfg, record: {d: "f" * 64 for d in cfg.sealed_dirs})


def _no_read(monkeypatch, why="a price row was read before the checks passed"):
    """Make the loader fail loudly, so that a refusal that reaches it is told apart from one that does not."""
    from es import sealed_run

    def boom(*_a, **_k):
        raise AssertionError(why)

    monkeypatch.setattr(sealed_run, "_load_sealed_raw", boom)


def _spy_parquet(monkeypatch):
    """Record every parquet file opened from here on, whichever way (pandas, pyarrow's reader or file class)."""
    import pyarrow.parquet as pq
    opened = []
    real_pd, real_table, real_file = pd.read_parquet, pq.read_table, pq.ParquetFile

    def pd_read(path, *a, **k):
        opened.append(str(path))
        return real_pd(path, *a, **k)

    def table(source, *a, **k):
        opened.append(str(source))
        return real_table(source, *a, **k)

    class File(real_file):
        def __init__(self, source, *a, **k):
            opened.append(str(source))
            super().__init__(source, *a, **k)

    monkeypatch.setattr(pd, "read_parquet", pd_read)
    monkeypatch.setattr(pq, "read_table", table)
    monkeypatch.setattr(pq, "ParquetFile", File)
    return opened


def _stop_at_the_read(monkeypatch):
    from es import sealed_run
    monkeypatch.setattr(sealed_run, "_load_sealed_raw", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("stop")))


# =====================================================================================================
# The refusals. Nothing in this section reads a price row.
# =====================================================================================================
def test_run_refuses_without_registration(registered):
    """The registered Test A reads es.registration's own record, and there is none: refused, with no lock, no record and no claim."""
    from es.registration import RegistrationError
    from es.sealed_run import default_config, run_sealed
    write, tmp = registered
    cfg = default_config("A")                                     # out_dir: the registered folder, under tmp in these tests
    with pytest.raises(RegistrationError, match="no registration record"):
        run_sealed(cfg, "0" * 64)
    assert not cfg.out_dir.exists() and not list((tmp / "docs_sealed").iterdir())


def test_run_refuses_unbound_inputs(registered, monkeypatch):
    """A protocol that prints no hash line for the bound files, and one that prints another hash, both stop the run before
    its lock. (The brief expected 'not the registered file' for a protocol naming no files; the landed gate says 'not
    bound' there and 'not the registered file' for a wrong hash.)"""
    from es.registration import RegistrationError
    from es.sealed_run import default_config, run_sealed
    _no_read(monkeypatch)
    write, tmp = registered
    proto, reg, sha = write("a protocol that names no files\n")
    (tmp / "inputs.json").write_text('{"eta": 0.01}')
    (tmp / "cutoffs.csv").write_text("analysis\n")
    (tmp / "CODE_MANIFEST.sha256").write_text("")
    cfg = default_config("A")
    cfg.registration = (proto, reg)
    cfg.inputs, cfg.cutoffs, cfg.code_manifest = tmp / "inputs.json", tmp / "cutoffs.csv", tmp / "CODE_MANIFEST.sha256"
    with pytest.raises(RegistrationError, match="not bound"):
        run_sealed(cfg, sha)
    assert not cfg.out_dir.exists()
    proto, reg, sha = write("cutoffs.csv SHA-256: " + "0" * 64 + "\n")
    cfg.registration = (proto, reg)
    with pytest.raises(RegistrationError, match="not the registered file"):
        run_sealed(cfg, sha)
    assert not cfg.out_dir.exists()


@pytest.mark.parametrize("changed", ["cutoffs", "inputs", "code_manifest"])
def test_run_refuses_a_bound_file_changed_after_registration(registered, monkeypatch, changed):
    from es.registration import RegistrationError
    from es.sealed_run import run_sealed
    _no_read(monkeypatch)
    write, tmp = registered
    proto, reg, sha, kw = _bound(tmp, write)
    p = kw[changed]
    p.write_text(p.read_text() + "\n# edited after registration\n")
    with pytest.raises(RegistrationError, match="not the registered file"):
        run_sealed(_cfg_a(tmp, kw), sha)
    assert not (tmp / "out").exists() and not (tmp / "docs_sealed" / "RUNS.json").exists()      # a refusal records no run


def test_run_once_lock_precedes_the_data_read_and_a_crash_counts(registered, monkeypatch):
    from es import sealed_run
    from es.sealed_run import run_sealed
    write, tmp = registered
    proto, reg, sha, kw = _bound(tmp, write)

    def crash(*_a, **_k):
        raise RuntimeError("crash while reading the sealed data")

    monkeypatch.setattr(sealed_run, "_load_sealed_raw", crash)
    cfg = _cfg_a(tmp, kw)
    with pytest.raises(RuntimeError, match="crash"):
        run_sealed(cfg, sha)
    log = (tmp / "out" / "RUN_LOG.txt").read_text()
    assert log.startswith("started ") and "finished" not in log and f"protocol {sha}" in log
    assert not (tmp / "out" / "results.json").exists()
    with pytest.raises(FileExistsError, match="runs once"):          # the crashed run was the one run
        run_sealed(cfg, sha)
    assert (tmp / "out" / "RUN_LOG.txt").read_text() == log


def test_a_crash_is_written_to_the_run_log_with_its_traceback(registered, monkeypatch):
    """The log is the record of a run that cannot be repeated: it says where it died."""
    from es import sealed_run
    from es.sealed_run import run_sealed
    write, tmp = registered
    proto, reg, sha, kw = _bound(tmp, write)
    monkeypatch.setattr(sealed_run, "_load_sealed_raw",
                        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("the disk went away")))
    with pytest.raises(RuntimeError, match="disk"):
        run_sealed(_cfg_a(tmp, kw), sha)
    lines = (tmp / "out" / "RUN_LOG.txt").read_text().splitlines()
    crash = [ln for ln in lines if ln.startswith("crashed ")]
    assert len(crash) == 1 and "RuntimeError: the disk went away" in crash[0]
    assert any("Traceback" in ln for ln in lines) and lines[0].startswith("started ")
    assert not any(ln.startswith("finished") for ln in lines)


def test_the_log_records_what_identifies_the_run(registered, monkeypatch):
    from es.registration import sha256_file
    from es.sealed_run import run_sealed
    write, tmp = registered
    proto, reg, sha, kw = _bound(tmp, write)
    _stop_at_the_read(monkeypatch)
    with pytest.raises(RuntimeError, match="stop"):
        run_sealed(_cfg_a(tmp, kw), sha, n_rep=123)
    log = (tmp / "out" / "RUN_LOG.txt").read_text()
    assert "test A" in log and f"protocol {sha}" in log and "n_rep 123" in log
    assert "registration 10.17605/OSF.IO/ABCDE registered_utc 2026-10-20T15:04:00Z" in log
    assert "\nregistration_kind osf\n" in log                                        # Ruling 140: beside the identifier
    for name in ("cutoffs", "inputs", "code_manifest"):
        assert f"bound {kw[name]} {sha256_file(kw[name])}" in log
    assert "departures 0" in log


def test_code_departures_refuse_unless_allowed_and_are_then_recorded(registered, monkeypatch):
    from es.registration import RegistrationError, sha256_file
    from es.sealed_run import code_departures, run_sealed
    write, tmp = registered
    proto, reg, sha, kw = _bound(tmp, write)
    assert code_departures(kw["code_manifest"], tmp / "pkg") == []
    (tmp / "pkg" / "b.py").write_text("B = 3\n")                        # a fix after registration
    dep = code_departures(kw["code_manifest"], tmp / "pkg")
    assert [d["file"] for d in dep] == ["b.py"] and dep[0]["current"] == sha256_file(tmp / "pkg" / "b.py")
    assert dep[0]["registered"] != dep[0]["current"]
    cfg = _cfg_a(tmp, kw)
    with pytest.raises(RegistrationError, match="departure"):
        run_sealed(cfg, sha)
    assert not (tmp / "out").exists() and not (tmp / "docs_sealed" / "RUNS.json").exists()
    _stop_at_the_read(monkeypatch)
    with pytest.raises(RuntimeError, match="stop"):
        run_sealed(cfg, sha, allow_departure=True)
    log = (tmp / "out" / "RUN_LOG.txt").read_text()
    assert log.startswith("started ") and "departures 1" in log
    assert (tmp / "out" / "departures.txt").read_text() == f"{dep[0]['registered']}  {dep[0]['current']}  b.py\n"
    (tmp / "pkg" / "b.py").unlink()
    assert code_departures(kw["code_manifest"], tmp / "pkg")[0]["current"] is None


def test_code_departures_names_every_changed_file_and_ignores_unlisted_ones(tmp_path):
    from es.registration import sha256_file
    from es.sealed_run import code_departures
    (tmp_path / "a.py").write_text("A = 1\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.py").write_text("C = 1\n")
    man = tmp_path / "man.sha256"
    man.write_text(f"{sha256_file(tmp_path / 'a.py')}  a.py\n\n{sha256_file(tmp_path / 'sub' / 'c.py')}  sub/c.py\n")
    (tmp_path / "new.py").write_text("NEW = 1\n")                       # not listed: not a departure
    assert code_departures(man, tmp_path) == []
    (tmp_path / "a.py").write_text("A = 2\n")
    (tmp_path / "sub" / "c.py").unlink()
    got = {d["file"]: d["current"] for d in code_departures(man, tmp_path)}
    assert got == {"a.py": sha256_file(tmp_path / "a.py"), "sub/c.py": None}


def test_a_malformed_manifest_line_is_refused_before_any_run(tmp_path):
    from es.registration import RegistrationError
    from es.sealed_run import code_departures
    man = tmp_path / "man.sha256"
    man.write_text("not a manifest line\n")
    with pytest.raises(RegistrationError, match="malformed"):
        code_departures(man, tmp_path)


# =====================================================================================================
# A development directory and a fake sealed pull made through the landed pull_sealed (Ruling 70).
# =====================================================================================================
def _price_file(market, stamps, path, naive=False, da=None, rt=None):
    ts = pd.to_datetime(stamps, utc=True)
    ts = ts.tz_localize(None) if naive else ts                        # MISO files hold naive UTC (es/data.py)
    values = np.arange(len(ts), dtype=float)
    pd.DataFrame({"market": market, "node": NODE[market], "node_type": "x", "timestamp_utc": ts,
                  "da_lmp": values if da is None else da, "rt_lmp": values if rt is None else rt}
                 ).to_parquet(path, index=False)


def _fmm_builder(start, end, root):
    (root / "fmm").mkdir()
    ts = pd.to_datetime(["2026-06-25 18:00", "2026-07-15 18:00"], utc=True)
    pd.DataFrame({"node": NODE["CAISO"], "timestamp_utc": ts, "fmm_lmp": [30.0, 31.0], "n15": [4, 4]}).to_parquet(
        root / "fmm" / "caiso_fmm_test.parquet", index=False)


def _nyint_builder(start, end, root):
    (root / "nyiso_rtint").mkdir()
    ts = pd.to_datetime(["2026-06-25 18:00", "2026-07-15 18:00"], utc=True)
    pd.DataFrame({"node": NODE["NYISO"], "timestamp_utc": ts, "rt_int_lmp": [40.0, 41.0]}).to_parquet(
        root / "nyiso_rtint" / "nyiso_rtint_test.parquet", index=False)


def _dev_dir(tmp):
    dev = tmp / "dev"
    dev.mkdir()
    for m in NODE:
        _price_file(m, ["2026-06-20 18:00", "2026-06-21 18:00"], dev / f"{m.lower()}_da_rt_dev.parquet", m == "MISO")
    return dev


def _fake_pull(tmp, proto, reg, sha, bad=None, foreign=None):
    """A development directory with two June rows per market and, through pull_sealed with fake builders, a
    sealed_2026Q3 pull whose price files hold one June, one July and (before the pull trims it) one October row.
    `bad` maps a market to extra (stamp, da, rt) rows of its sealed file, for the non-finite tests; `foreign` maps a market to
    a number of rows to copy, first rows first, under a node that is not in the node map."""
    from es.sealed_pull import pull_sealed
    dev = _dev_dir(tmp)

    def price(market):
        def build(start, end, root):
            stamps = ["2026-06-25 18:00", "2026-07-15 18:00", "2026-10-01 18:00"]
            da, rt = [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]
            for stamp, d, r in (bad or {}).get(market, ()):
                stamps.append(stamp)
                da.append(d)
                rt.append(r)
            path = root / f"{market.lower()}_da_rt_test.parquet"
            _price_file(market, stamps, path, market == "MISO", da=np.array(da), rt=np.array(rt))
            if (foreign or {}).get(market):
                frame = pd.read_parquet(path)
                pd.concat([frame, frame.iloc[:foreign[market]].assign(node="NOT_A_HUB")]).to_parquet(path, index=False)
        return build

    builders = {m: price(m) for m in NODE}
    builders.update({"FMM": _fmm_builder, "NYISO_INT": _nyint_builder})
    # Ruling 70: no registration= argument. The registered fixture points es.registration's PROTOCOL_PATH,
    # REGISTRATION_PATH and SEALED_DIR (where PULLS.json lives) at tmp.
    root, _ = pull_sealed("A", sha, data_root=tmp, builders=builders, reference_dirs=[],
                          check_days=False)           # one node and three rows per market: not complete days (logged)
    return dev, root


def _cfg_pull(tmp, proto, reg, dev, **over):
    from es.analyses import SEALED
    from es.sealed_run import SealedConfig
    return SealedConfig("A", SEALED["SA"], SEALED["SAF"], ["sealed_2026Q3"], "2026-09-30", registration=(proto, reg),
                        out_dir=tmp / "out", data_root=tmp, dev_dirs=(dev,), **over)


def _claimed(cfg, sha):
    """The claim that run_sealed takes at its lock, made for a test of the loader: the real _claim_run, for cfg's test and
    out_dir. (Nothing else makes one; the loader and _execute refuse any other, Ruling 109 I1.)"""
    from es.registration import verify_registration
    from es.sealed_run import _claim_run
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    record = verify_registration(sha, *cfg.registration)
    return _claim_run(cfg, record, "2026-10-23T00:00:00+00:00", out, {"code_manifest_sha256": "x", "files": {}}, 2000, False, 0)


def test_sealed_loader_reads_the_pull_cuts_at_window_end_and_records_manifests(registered):
    from es.guard import SealedDataError, first_sealed_utc
    from es.sealed_pull import manifest_sha256
    from es.sealed_run import _load_sealed_raw
    from es.settlement import load_caiso_fmm, load_nyiso_integrated
    write, tmp = registered
    proto, reg, sha = write()
    dev, root = _fake_pull(tmp, proto, reg, sha)
    cfg = _cfg_pull(tmp, proto, reg, dev)
    claim = _claimed(cfg, sha)
    raw = _load_sealed_raw(cfg, sha, claim)
    assert raw.attrs["sealed_manifests"] == {"sealed_2026Q3": manifest_sha256(root)}
    assert raw.attrs["rows_read"] == 16 and raw.attrs["rows_kept"] == 16       # 8 development + 8 sealed rows
    assert (raw.attrs["node_map_dropped"], raw.attrs["duplicates_dropped"], raw.attrs["beyond_window_end"]) == (0, 0, 0)
    assert raw.attrs["off_hour_dropped"] == {} and raw.attrs["nonfinite_dropped"] == {}
    assert raw["timestamp_utc"].max() == pd.Timestamp("2026-07-15 18:00", tz="UTC")   # the pull trimmed October
    assert str(raw["timestamp_utc"].dtype) == "datetime64[ns, UTC]" and raw["node_type"].isin(("x",)).sum() == 0
    assert raw.loc[raw["market"] == "MISO", "timestamp_utc"].max() == pd.Timestamp("2026-07-15 18:00", tz="UTC")
    cfg.window_end = "2026-07-01"
    cut = _load_sealed_raw(cfg, sha, claim)
    assert cut["timestamp_utc"].max() == pd.Timestamp("2026-06-25 18:00", tz="UTC") and cut.attrs["rows_kept"] == 12
    assert cut.attrs["beyond_window_end"] == 4 and cut.attrs["rows_read"] == 16
    fmm = load_caiso_fmm(root / "fmm", dev_only=False)
    nyi = load_nyiso_integrated(root / "nyiso_rtint", dev_only=False)
    assert list(fmm.columns) == ["node", "timestamp_utc", "fmm_lmp"] and len(fmm) == 2
    assert fmm["timestamp_utc"].max() >= first_sealed_utc("CAISO")
    assert list(nyi.columns) == ["node", "timestamp_utc", "rt_int_lmp"] and len(nyi) == 2
    with pytest.raises(SealedDataError):
        load_caiso_fmm(root / "fmm")                                  # the development loader refuses the same files
    with pytest.raises(SealedDataError):
        load_nyiso_integrated(root / "nyiso_rtint")
    (root / "isone_da_rt_test.parquet").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="changed"):
        _load_sealed_raw(cfg, sha, claim)


def test_sealed_loader_returns_sorted_rows_and_the_development_file_wins_a_duplicate(registered):
    """The pull's first ten days repeat H1's last ten (the overlap check compares them): one row per key, development first."""
    from es.sealed_run import _load_sealed_raw
    write, tmp = registered
    proto, reg, sha = write()
    dev, root = _fake_pull(tmp, proto, reg, sha)
    _price_file("ISONE", ["2026-06-25 18:00"], dev / "isone_da_rt_dev.parquet", da=np.array([99.0]), rt=np.array([99.0]))
    cfg = _cfg_pull(tmp, proto, reg, dev)
    raw = _load_sealed_raw(cfg, sha, _claimed(cfg, sha))
    iso = raw[raw["market"] == "ISONE"]
    assert len(iso) == 2 and iso["timestamp_utc"].is_monotonic_increasing
    assert iso.loc[iso["timestamp_utc"] == pd.Timestamp("2026-06-25 18:00", tz="UTC"), "da_lmp"].iloc[0] == 99.0
    assert raw.attrs["rows_read"] - raw.attrs["rows_kept"] == 1 and raw.attrs["duplicates_dropped"] == 1
    assert not raw[["market", "node", "timestamp_utc"]].duplicated().any()
    assert raw.equals(raw.sort_values(["market", "node", "timestamp_utc"]).reset_index(drop=True))


def test_nonfinite_prices_are_dropped_counted_and_never_raise(registered):
    """Ruling 82: a raise after the data are read would spend the one run, so a non-finite DA or RT price is dropped and
    counted per market and node. Every finite row stays."""
    from es.sealed_run import _load_sealed_raw
    write, tmp = registered
    proto, reg, sha = write()
    bad = {"ISONE": [("2026-07-16 18:00", 5.0, np.nan)],                     # a NaN RT price
           "NYISO": [("2026-07-16 18:00", np.inf, 5.0), ("2026-07-17 18:00", 5.0, -np.inf)]}
    dev, root = _fake_pull(tmp, proto, reg, sha, bad=bad)
    cfg = _cfg_pull(tmp, proto, reg, dev)
    claim = _claimed(cfg, sha)
    raw = _load_sealed_raw(cfg, sha, claim)
    assert raw.attrs["nonfinite_dropped"] == {"ISONE/.H.INTERNAL_HUB": 1, "NYISO/N.Y.C.": 2}
    assert raw.attrs["rows_read"] == 19 and raw.attrs["rows_kept"] == 16                   # 16 + the 3 bad rows
    assert np.isfinite(raw[["da_lmp", "rt_lmp"]].to_numpy()).all()
    assert (raw["market"] + "/" + raw["node"]).value_counts().to_dict() == {
        "NYISO/N.Y.C.": 4, "ISONE/.H.INTERNAL_HUB": 4, "MISO/ILLINOIS.HUB": 4, "CAISO/TH_SP15_GEN-APND": 4}
    assert _load_sealed_raw(cfg, sha, claim).attrs["nonfinite_dropped"] == raw.attrs["nonfinite_dropped"]


def test_nonfinite_rows_after_the_window_end_are_not_counted(registered):
    from es.sealed_run import _load_sealed_raw
    write, tmp = registered
    proto, reg, sha = write()
    dev, root = _fake_pull(tmp, proto, reg, sha, bad={"ISONE": [("2026-07-16 18:00", 5.0, np.nan)]})
    cfg = _cfg_pull(tmp, proto, reg, dev)
    cfg.window_end = "2026-07-01"                    # the bad row is dated 16 July: outside this run's window
    raw = _load_sealed_raw(cfg, sha, _claimed(cfg, sha))
    assert raw.attrs["nonfinite_dropped"] == {} and raw.attrs["rows_kept"] == 12


def test_the_loader_counts_each_kind_of_dropped_row_separately_and_drops_the_off_hour_stamps(registered):
    """Ruling 109, M2 and M4: rows outside the node map, repeated hours, rows past the window end, rows stamped off the hour
    (which the panel build refuses, after the lock) and rows with a non-finite price are each counted on their own, and
    rows_read is all of them plus the rows kept."""
    from es.sealed_run import _load_sealed_raw
    write, tmp = registered
    proto, reg, sha = write()
    bad = {"NYISO": [("2026-07-16 18:30", 5.0, 5.0), ("2026-07-16 19:00", 6.0, 6.0)],         # one off the hour, one fine
           "CAISO": [("2026-07-17 18:00", 5.0, 5.0), ("2026-07-16 20:00", np.nan, 5.0)]}      # past the window, a NaN price
    dev, root = _fake_pull(tmp, proto, reg, sha, bad=bad, foreign={"ISONE": 2})   # two rows of a node outside the node map
    ny = pd.read_parquet(dev / "nyiso_da_rt_dev.parquet")
    pd.concat([ny, ny.iloc[:1]]).to_parquet(dev / "nyiso_da_rt_dev.parquet", index=False)      # a repeated hour in a development file
    iso = pd.read_parquet(dev / "isone_da_rt_dev.parquet")
    repeated = iso.iloc[:1].assign(timestamp_utc=pd.Timestamp("2026-06-25 18:00", tz="UTC"), da_lmp=99.0, rt_lmp=99.0)
    pd.concat([iso, repeated]).to_parquet(dev / "isone_da_rt_dev.parquet", index=False)         # the pull has 06-25 too
    cfg = _cfg_pull(tmp, proto, reg, dev)
    cfg.window_end = "2026-07-16"
    raw = _load_sealed_raw(cfg, sha, _claimed(cfg, sha))
    a = raw.attrs
    assert a["rows_read"] == 23 and a["rows_kept"] == len(raw) == 17
    assert (a["node_map_dropped"], a["duplicates_dropped"], a["beyond_window_end"]) == (2, 1, 1)
    assert a["dev_duplicates_dropped"] == 1                                    # the development loader's own count (not in rows_read)
    assert a["off_hour_dropped"] == {"NYISO/N.Y.C.": 1} and a["nonfinite_dropped"] == {"CAISO/TH_SP15_GEN-APND": 1}
    assert a["rows_read"] == a["rows_kept"] + 2 + 1 + 1 + 1 + 1                 # every row is accounted for, once
    assert (raw["timestamp_utc"] == raw["timestamp_utc"].dt.floor("h")).all()   # nothing off the hour reaches the panel build
    iso_rows = raw[raw["market"] == "ISONE"]
    assert iso_rows.loc[iso_rows["timestamp_utc"] == pd.Timestamp("2026-06-25 18:00", tz="UTC"), "da_lmp"].iloc[0] == 99.0


def test_the_loader_is_gated_on_its_own(registered, monkeypatch):
    """A direct call needs the registration and intact, bound manifests too; nothing is read before they pass. (It needs the
    claim as well, below.)"""
    from es import sealed_run
    from es.registration import RegistrationError
    from es.sealed_run import _load_sealed_raw
    write, tmp = registered
    proto, reg, sha = write()
    dev, root = _fake_pull(tmp, proto, reg, sha)
    reads = []
    real = sealed_run.load_dev_raw
    monkeypatch.setattr(sealed_run, "load_dev_raw", lambda *a, **k: reads.append(a) or real(*a, **k))
    cfg = _cfg_pull(tmp, proto, reg, dev)
    claim = _claimed(cfg, sha)
    with pytest.raises(RegistrationError):
        _load_sealed_raw(cfg, "0" * 64, claim)                           # not the registered protocol
    reg.rename(tmp / "REGISTRATION.gone")
    with pytest.raises(RegistrationError, match="no registration record"):
        _load_sealed_raw(cfg, sha, claim)                                # no record at all
    (tmp / "REGISTRATION.gone").rename(reg)
    self_sha, self_kw = _self_made(tmp)                                   # a record of its own, put into cfg.registration:
    cfg.registration = self_kw["registration"]                            # a sealed run reads es.registration's paths only
    with pytest.raises(RegistrationError, match="differs from the one supplied"):
        _load_sealed_raw(cfg, self_sha, claim)
    cfg.registration = (proto, reg)
    (tmp / "docs_sealed" / "PULLS.json").write_text("{}")                # the pull is not on record
    with pytest.raises(RegistrationError, match="no completed pull record"):
        _load_sealed_raw(cfg, sha, claim)
    assert reads == []


# =====================================================================================================
# check_sealed_dirs binds each sealed directory to the verified registration through PULLS.json (Ruling 70).
# =====================================================================================================
def _record_of(sha, proto, reg):
    from es.registration import verify_registration
    return verify_registration(sha, proto, reg)


def _edit_pulls(tmp, **changes):
    path = tmp / "docs_sealed" / "PULLS.json"
    pulls = json.loads(path.read_text())
    pulls["A"].update(changes)
    path.write_text(json.dumps(pulls))


def test_check_sealed_dirs_accepts_a_pull_made_under_the_verified_registration(registered, monkeypatch):
    from es.sealed_pull import manifest_sha256
    from es.sealed_run import check_sealed_dirs
    write, tmp = registered
    proto, reg, sha = write()
    dev, root = _fake_pull(tmp, proto, reg, sha)
    opened = _spy_parquet(monkeypatch)
    got = check_sealed_dirs(_cfg_pull(tmp, proto, reg, dev), _record_of(sha, proto, reg))
    assert got == {"sealed_2026Q3": manifest_sha256(root)}
    assert opened == []                                                    # hashed and bound, no price row read


def test_check_sealed_dirs_without_directories_reads_nothing_and_binds_nothing(registered):
    from es.sealed_run import check_sealed_dirs
    write, tmp = registered
    proto, reg, sha = write()
    (tmp / "docs_sealed" / "PULLS.json").write_text("{not json")        # would raise if it were read
    cfg = _cfg_pull(tmp, proto, reg, tmp / "dev")
    cfg.sealed_dirs = []
    assert check_sealed_dirs(cfg, _record_of(sha, proto, reg)) == {}


def test_check_sealed_dirs_refuses_a_directory_with_no_pull_record(registered):
    from es.registration import RegistrationError
    from es.sealed_run import check_sealed_dirs
    write, tmp = registered
    proto, reg, sha = write()
    dev, root = _fake_pull(tmp, proto, reg, sha)
    record = _record_of(sha, proto, reg)
    (tmp / "docs_sealed" / "PULLS.json").unlink()                         # a valid manifest, and no record of the pull
    with pytest.raises(RegistrationError, match="no completed pull record"):
        check_sealed_dirs(_cfg_pull(tmp, proto, reg, dev), record)
    (tmp / "docs_sealed" / "PULLS.json").write_text(json.dumps({"A": {"root": str(tmp / "elsewhere")}}))
    with pytest.raises(RegistrationError, match="no completed pull record"):          # a record of another root
        check_sealed_dirs(_cfg_pull(tmp, proto, reg, dev), record)


def test_check_sealed_dirs_refuses_a_pull_under_another_protocol(registered):
    from es.registration import RegistrationError
    from es.sealed_run import check_sealed_dirs
    write, tmp = registered
    proto, reg, sha = write()
    dev, root = _fake_pull(tmp, proto, reg, sha)
    _edit_pulls(tmp, protocol_sha256="1" * 64)
    with pytest.raises(RegistrationError, match="different protocol"):
        check_sealed_dirs(_cfg_pull(tmp, proto, reg, dev), _record_of(sha, proto, reg))


def test_check_sealed_dirs_refuses_a_pull_under_another_registration(registered):
    from es.registration import RegistrationError
    from es.sealed_run import check_sealed_dirs
    write, tmp = registered
    proto, reg, sha = write()
    dev, root = _fake_pull(tmp, proto, reg, sha)
    proto, reg, sha = write(reg_fields={"osf_doi": "10.17605/OSF.IO/ZZZZZ"})           # the same protocol, registered again
    with pytest.raises(RegistrationError, match="different registration"):
        check_sealed_dirs(_cfg_pull(tmp, proto, reg, dev), _record_of(sha, proto, reg))


def test_an_embargoed_registration_that_gains_a_doi_still_matches_through_its_osf_address(registered):
    """The pull ran when the record gave only the OSF address; the DOI appeared later. One identifier in common is the
    same registration (the rule pull_sealed applies to a resume), and a record with none in common is not."""
    from es.registration import RegistrationError
    from es.sealed_run import check_sealed_dirs
    write, tmp = registered
    proto, reg, sha = write(reg_fields={"osf_doi": None, "osf_registration": "https://osf.io/abcde/"})
    dev, root = _fake_pull(tmp, proto, reg, sha)
    assert json.loads((tmp / "docs_sealed" / "PULLS.json").read_text())["A"]["osf_doi"] is None
    proto, reg, sha = write(reg_fields={"osf_registration": "https://osf.io/abcde/"})            # the DOI is back
    record = _record_of(sha, proto, reg)
    assert record["osf_doi"] == "10.17605/OSF.IO/ABCDE"
    assert check_sealed_dirs(_cfg_pull(tmp, proto, reg, dev), record)
    proto, reg, sha = write(reg_fields={"osf_doi": None, "osf_registration": "https://osf.io/qqqqq/"})
    with pytest.raises(RegistrationError, match="different registration"):
        check_sealed_dirs(_cfg_pull(tmp, proto, reg, dev), _record_of(sha, proto, reg))


def test_the_pull_is_matched_to_the_record_on_the_registration_id_whatever_the_case(registered):
    """The match is on the registration id the entry in PULLS.json carries (registration_id(record)), not on the DOI or the
    address as written: a DOI added when an embargo ends, an address replaced by its DOI, or an id in another case is the
    same registration; another id is not."""
    from es.registration import RegistrationError
    from es.sealed_run import check_sealed_dirs
    write, tmp = registered
    proto, reg, sha = write(reg_fields={"osf_registration": "https://osf.io/abcde/"})     # DOI and OSF address of abcde
    dev, root = _fake_pull(tmp, proto, reg, sha)
    assert json.loads((tmp / "docs_sealed" / "PULLS.json").read_text())["A"]["registration_id"] == "abcde"
    record = _record_of(sha, proto, reg)
    assert check_sealed_dirs(_cfg_pull(tmp, proto, reg, dev), record)
    _edit_pulls(tmp, osf_doi=None, osf_registration=None, registration_identifier=None)   # only the id is read
    assert check_sealed_dirs(_cfg_pull(tmp, proto, reg, dev), record)
    _edit_pulls(tmp, registration_id="ABCDE")                                            # whatever the case
    assert check_sealed_dirs(_cfg_pull(tmp, proto, reg, dev), record)
    proto, reg, sha = write(reg_fields={"osf_doi": "10.17605/OSF.IO/ZZZZZ", "osf_registration": "https://osf.io/zzzzz/"})
    _edit_pulls(tmp, protocol_sha256=sha)                              # the same protocol, registered under another id
    with pytest.raises(RegistrationError, match=r"pulled under a different registration \(registration id abcde\).*\(zzzzz\)"):
        check_sealed_dirs(_cfg_pull(tmp, proto, reg, dev), _record_of(sha, proto, reg))
    proto, reg, sha = write(reg_fields={"osf_doi": None, "osf_registration": "https://osf.io/abcde/"})
    _edit_pulls(tmp, protocol_sha256=sha, registration_id=None)         # an entry with no id is not a pull_sealed record
    with pytest.raises(RegistrationError, match="names no registration_id"):
        check_sealed_dirs(_cfg_pull(tmp, proto, reg, dev), _record_of(sha, proto, reg))
    _edit_pulls(tmp, registration_id="zzzzz")                           # an entry that names another registration than the
    with pytest.raises(RegistrationError, match=r"pulled under a different registration \(registration id zzzzz\).*\(abcde\)"):
        check_sealed_dirs(_cfg_pull(tmp, proto, reg, dev), _record_of(sha, proto, reg))     # record and than the pull's log
    _edit_pulls(tmp, registration_id="abcde")
    assert check_sealed_dirs(_cfg_pull(tmp, proto, reg, dev), _record_of(sha, proto, reg))


def test_the_pull_s_own_log_cannot_name_another_registration_than_its_pulls_json_entry(registered):
    """The log is part of the hashed record. An entry edited to match another registration, with the log of the pull as it
    was, is refused; a log edited after the pull no longer verifies, and one rewritten to match is not the recorded manifest."""
    from es.registration import RegistrationError
    from es.sealed_pull import write_manifest
    from es.sealed_run import _logged_registration_ids, check_sealed_dirs
    write, tmp = registered
    proto, reg, sha = write()
    dev, root = _fake_pull(tmp, proto, reg, sha)
    assert _logged_registration_ids(root) == {"abcde"}
    proto, reg, sha = write(reg_fields={"osf_doi": "10.17605/OSF.IO/ZZZZZ"})
    _edit_pulls(tmp, registration_id="zzzzz", protocol_sha256=sha)       # the entry now names the new registration, the log not
    with pytest.raises(RegistrationError, match=r"PULL_LOG.txt names a different registration \(registration id abcde\)"):
        check_sealed_dirs(_cfg_pull(tmp, proto, reg, dev), _record_of(sha, proto, reg))
    proto, reg, sha = write()
    _edit_pulls(tmp, registration_id="abcde", protocol_sha256=sha)
    record = _record_of(sha, proto, reg)
    assert check_sealed_dirs(_cfg_pull(tmp, proto, reg, dev), record)
    (root / "PULL_LOG.txt").write_text((root / "PULL_LOG.txt").read_text() + "registration_id: zzzzz\n")
    with pytest.raises(RuntimeError, match="changed"):
        check_sealed_dirs(_cfg_pull(tmp, proto, reg, dev), record)
    write_manifest(root)                                                  # rewritten to match: not the one recorded
    with pytest.raises(RegistrationError, match="not the one recorded"):
        check_sealed_dirs(_cfg_pull(tmp, proto, reg, dev), record)


def test_a_pull_is_found_through_a_symlinked_data_root_and_a_symlinked_record(registered):
    """PULLS.json holds the root as a resolved absolute path (symlinks resolved), and both sides are resolved here: a data
    root given through a symlink finds the pull, and so does a record that was written with the symlink in its path."""
    from es.sealed_run import check_sealed_dirs
    write, tmp = registered
    proto, reg, sha = write()
    dev, root = _fake_pull(tmp, proto, reg, sha)
    record = _record_of(sha, proto, reg)
    link = tmp.parent / f"{tmp.name}_link"
    link.symlink_to(tmp, target_is_directory=True)
    assert json.loads((tmp / "docs_sealed" / "PULLS.json").read_text())["A"]["root"] == str(root.resolve())
    through_link = _cfg_pull(tmp, proto, reg, dev)
    through_link.data_root = link
    assert check_sealed_dirs(through_link, record)
    _edit_pulls(tmp, root=str(link / "sealed_2026Q3"))                   # an entry with the symlink in its path
    assert check_sealed_dirs(_cfg_pull(tmp, proto, reg, dev), record)
    assert check_sealed_dirs(through_link, record)


def test_check_sealed_dirs_refuses_a_pull_that_started_before_the_registration(registered):
    from es.registration import RegistrationError
    from es.sealed_run import check_sealed_dirs
    write, tmp = registered
    proto, reg, sha = write()
    dev, root = _fake_pull(tmp, proto, reg, sha)
    record = _record_of(sha, proto, reg)
    assert record["registered_utc"] == "2026-10-20T15:04:00Z"
    _edit_pulls(tmp, started_utc="2026-10-20T15:03:59+00:00")
    with pytest.raises(RegistrationError, match="started before the registration"):
        check_sealed_dirs(_cfg_pull(tmp, proto, reg, dev), record)
    _edit_pulls(tmp, started_utc="2026-10-20T15:04:00+00:00")           # at the registration time: allowed
    assert check_sealed_dirs(_cfg_pull(tmp, proto, reg, dev), record)
    _edit_pulls(tmp, started_utc="2026-10-20T10:04:00-05:00")           # the same instant written in another offset
    assert check_sealed_dirs(_cfg_pull(tmp, proto, reg, dev), record)


def test_check_sealed_dirs_refuses_a_manifest_that_is_not_the_one_recorded(registered):
    """A file changed and the manifest rewritten to match verifies as a set, but is not the pull that was recorded."""
    from es.registration import RegistrationError
    from es.sealed_pull import verify_manifest, write_manifest
    from es.sealed_run import check_sealed_dirs
    write, tmp = registered
    proto, reg, sha = write()
    dev, root = _fake_pull(tmp, proto, reg, sha)
    record = _record_of(sha, proto, reg)
    _price_file("ISONE", ["2026-06-25 18:00"], root / "isone_da_rt_test.parquet", da=np.array([7.0]), rt=np.array([7.0]))
    write_manifest(root)
    verify_manifest(root)                                                 # internally consistent again
    with pytest.raises(RegistrationError, match="not the one recorded"):
        check_sealed_dirs(_cfg_pull(tmp, proto, reg, dev), record)


def test_check_sealed_dirs_refuses_files_changed_added_or_missing(registered):
    from es.sealed_run import check_sealed_dirs
    write, tmp = registered
    proto, reg, sha = write()
    dev, root = _fake_pull(tmp, proto, reg, sha)
    record = _record_of(sha, proto, reg)
    cfg = _cfg_pull(tmp, proto, reg, dev)
    (root / "extra_da_rt_x.parquet").write_bytes(b"x")                     # a price file the manifest does not list
    with pytest.raises(RuntimeError, match="not in the manifest"):
        check_sealed_dirs(cfg, record)
    (root / "extra_da_rt_x.parquet").unlink()
    (root / "fmm" / "caiso_fmm_test.parquet").unlink()
    with pytest.raises(RuntimeError, match="missing"):
        check_sealed_dirs(cfg, record)


def test_a_sealed_directory_without_a_manifest_is_refused(registered):
    from es.sealed_run import check_sealed_dirs
    write, tmp = registered
    proto, reg, sha = write()
    dev, root = _fake_pull(tmp, proto, reg, sha)
    (root / "MANIFEST.sha256").unlink()
    with pytest.raises(RuntimeError, match="no MANIFEST.sha256"):
        check_sealed_dirs(_cfg_pull(tmp, proto, reg, dev), _record_of(sha, proto, reg))


# =====================================================================================================
# run_sealed's order: every check that reads no price row comes before the lock, and the lock before any read.
# =====================================================================================================
def test_an_unrecorded_sealed_directory_stops_the_run_before_its_lock(registered, monkeypatch):
    from es.registration import RegistrationError
    from es.sealed_run import run_sealed
    write, tmp = registered
    proto, reg, sha, kw = _bound(tmp, write, real_code=True)
    _no_read(monkeypatch)
    cfg = _sealed_cfg(tmp, kw)
    (tmp / "sealed_2026Q3").mkdir()
    (tmp / "sealed_2026Q3" / "MANIFEST.sha256").write_text("")          # a manifest, and no pull on record
    opened = _spy_parquet(monkeypatch)
    with pytest.raises(RegistrationError, match="no completed pull record"):
        run_sealed(cfg, sha)
    assert not (tmp / "outputs_es").exists() and opened == []
    assert not list((tmp / "docs_sealed").iterdir())                    # no record, no claim


@pytest.mark.parametrize("what", ["out_dir", "window_end", "window_too_late", "stray_dir", "b_without_prior",
                                  "a_with_prior", "n_rep"])
def test_a_misconfigured_run_is_refused_before_its_lock(registered, monkeypatch, what):
    from es.analyses import SEALED
    from es.sealed_run import SealedConfig, run_sealed
    write, tmp = registered
    proto, reg, sha, kw = _bound(tmp, write)
    _no_read(monkeypatch)
    cfg = _cfg_a(tmp, kw)
    n_rep = 2000
    if what == "out_dir":
        cfg.out_dir = None
    elif what == "window_end":
        cfg.window_end = "end of September"
    elif what == "window_too_late":
        cfg.window_end = "2026-10-01"                   # one day past Test A's own window
    elif what == "stray_dir":
        cfg.sealed_dirs = ["sealed_2026Q4"]             # Test B's pull in a Test A run
    elif what == "b_without_prior":
        cfg = SealedConfig("B", SEALED["SB"], SEALED["SBF"], [], "2026-12-31", out_dir=tmp / "out", **kw)
    elif what == "a_with_prior":
        cfg.prior_test_dir = tmp / "prior"
    elif what == "n_rep":
        n_rep = 0
    with pytest.raises(ValueError):
        run_sealed(cfg, sha, n_rep=n_rep)
    assert not (tmp / "out").exists() and not (tmp / "docs_sealed" / "RUNS.json").exists()


@pytest.mark.parametrize("doc,match", [
    ({"eta": 0.01, "family": None}, "registered family"),
    ({"eta": 0.01, "family": {**FAMILY, "H-B2": {"confirmatory": True, "reasons": []}}}, r"H-B2.*approximate"),
    ({"eta": 0.01, "family": {**FAMILY, "H-E": {"confirmatory": True, "reasons": [], "approximate": False}}},
     "interim_approximate"),
    ({"eta": 0.01, "family": {k: v for k, v in FAMILY.items() if k != "H-B1"}}, "H-B1"),
    ({"family": FAMILY}, "eta"),
    ({"eta": "fast", "family": FAMILY}, "eta"),
    ([0.01, FAMILY], "JSON object"),
])
def test_inputs_without_a_complete_registered_family_are_refused_before_the_lock(registered, monkeypatch, doc, match):
    """The inputs the run reads after its lock are all checked before it: a missing key there would spend the run."""
    from es.sealed_run import run_sealed
    write, tmp = registered
    _no_read(monkeypatch)
    proto, reg, sha, kw = _bound(tmp, write)
    kw["inputs"].write_text(json.dumps(doc))
    proto, reg, sha = write(_protocol_lines(kw["cutoffs"], kw["inputs"], kw["code_manifest"]))   # re-bound after the edit
    kw["registration"] = (proto, reg)
    with pytest.raises(ValueError, match=match):
        run_sealed(_cfg_a(tmp, kw), sha)
    assert not (tmp / "out").exists()


# --- Test B pools with Test A's stored output: everything it needs from there is checked before the lock.
STARTED_A = "2026-10-21T12:00:00+00:00"


def _record_run(store, test_id, out, sha, started, manifest=None, claim=True, **over):
    """The RUNS.json entry (and the claim file) that a finished run leaves in `store`; `over` replaces or adds fields."""
    path = store / "RUNS.json"
    runs = json.loads(path.read_text()) if path.exists() else {}
    runs[test_id] = {"started_utc": started, "protocol_sha256": sha, "out_dir": str(Path(out).resolve()),
                     "finished_utc": "2026-10-21T13:00:00+00:00", "manifest_sha256": manifest, "status": "completed", **over}
    path.write_text(json.dumps(runs))
    if claim:
        (store / f"RUN_{test_id}.claim").write_text("{}")


def _prior_dir(tmp, sha, record=True, results=True, no_kept=(), folder=None, **over):
    """A fake Test A output directory (results.json, kept/*.parquet and a manifest) and, with `record`, the RUNS.json entry and
    claim file of the run that made it, in the store beside it (a rehearsal's, here tmp itself): the chain Test B checks
    (Ruling 109 I3). `over` replaces fields of results.json; `no_kept` names stored frames that Test A never wrote (its
    economics failed), in a folder whose manifest and record are otherwise in order. `folder` is where it is made (default
    tmp/A_out; Test B of a sealed config reads Test A's registered output folder, _out(tmp, "A"))."""
    from es.run_economics import KEPT
    from es.sealed_pull import manifest_sha256, write_manifest
    from es.sealed_run import _safe
    prior = tmp / "A_out" if folder is None else Path(folder)
    (prior / "kept").mkdir(parents=True)
    for name in KEPT:
        if name not in no_kept:
            pd.DataFrame({"x": [1]}).to_parquet(prior / "kept" / f"{_safe(name)}.parquet", index=False)
    if results:
        res = {"test": "A", "protocol_sha256": sha, "started_utc": STARTED_A, "coverage": {"ran": True, "usable_nodes": 33},
               "finished_utc": "2026-10-21T13:00:00+00:00", "status": "completed",
               "H-B1": {"p": 0.01, "t_hat": 0.2}, "H-B2": {"p": 0.2, "t_hat": -0.1}}
        res.update(over)
        (prior / "results.json").write_text(json.dumps(res))
    write_manifest(prior)
    if record:
        _record_run(tmp, "A", prior, sha, STARTED_A, manifest_sha256(prior))
    return prior


def _cfg_b(tmp, kw, prior):
    """A rehearsal of Test B: the analog arm AQ2 on both sides, pooled with the output folder `prior`."""
    from es.analyses import ANALOGS
    from es.sealed_run import SealedConfig
    arm = ANALOGS["AQ2"]
    return SealedConfig("B", arm, arm, [], "2026-06-30", prior_test_dir=prior, out_dir=tmp / "outB", **kw)


def _rewrite_manifest(prior):
    from es.sealed_pull import write_manifest
    (prior / "MANIFEST.sha256").unlink()
    write_manifest(prior)


@pytest.mark.parametrize("defect,exc,match", [
    ("stray", RuntimeError, "not in the manifest"),             # a file added to Test A's directory after it finished
    ("kept_deleted", RuntimeError, "missing"),                  # a stored file deleted after it finished
    ("kept_deleted_manifest_rewritten", "RegistrationError", "not the one recorded"),     # ... and the manifest rewritten to hide it
    ("kept_replaced_manifest_rewritten", "RegistrationError", "not the one recorded"),    # rows no recorded run produced
    ("results_of_test_b", ValueError, "Test A"),
    ("other_protocol", "RegistrationError", "different protocol"),
    ("other_protocol_in_the_record", "RegistrationError", "different protocol"),
    ("another_folder", "RegistrationError", "names the folder"),   # the record is of a run that wrote elsewhere
    ("another_start", "RegistrationError", "recorded run"),        # results.json is of another run than the record
    ("hb2_without_p", ValueError, r"H-B2.*'p'"),
    ("no_results", ValueError, "results.json"),
    ("unfinished_record_with_a_manifest", RuntimeError, r"complete.*lacks \['finished_utc', 'manifest_sha256'\]"),     # N2
    ("record_without_the_manifest_sha", RuntimeError, r"complete.*lacks \['manifest_sha256'\]"),
    ("record_without_finished_utc", RuntimeError, r"complete.*lacks \['finished_utc'\]"),    # the manifest SHA alone is not "finished"
    # N2b (the reviewer's V2): the record names the folder Test A wrote, whichever folder Test B was pointed at
    ("unfinished_record_complete_output_elsewhere", RuntimeError, r"A_out: Test A's output is complete.*lacks.*Point Test B's prior_test_dir"),
    ("unfinished_record_crashed_run_elsewhere", "RegistrationError", r"names the folder .*A_out"),
    ("record_not_an_object", RuntimeError, "record of Test A is not an object"),
    ("no_record", ValueError, "Test A has not been run"),       # the reviewer's P5: a fabricated folder, Test A never started
    ("no_record_no_folder", ValueError, "Test A has not been run"),
])
def test_test_b_checks_test_a_s_output_before_its_lock(registered, monkeypatch, defect, exc, match):
    from es.registration import RegistrationError
    from es.sealed_run import run_sealed
    exc = RegistrationError if exc == "RegistrationError" else exc
    write, tmp = registered
    proto, reg, sha, kw = _bound(tmp, write)
    _no_read(monkeypatch)
    over = {"results_of_test_b": {"test": "B"}, "other_protocol": {"protocol_sha256": "2" * 64},
            "another_start": {"started_utc": "2026-10-21T12:00:01+00:00"},
            "hb2_without_p": {"H-B2": {"t_hat": 0.1}}}.get(defect, {})
    prior = _prior_dir(tmp, sha, record=defect not in ("no_record", "no_record_no_folder"), results=defect != "no_results", **over)
    runs, where = tmp / "RUNS.json", prior
    if defect == "stray":
        (prior / "stray.txt").write_text("a file added after Test A finished")
    elif defect == "kept_deleted":
        (prior / "kept" / "P1.parquet").unlink()
    elif defect == "kept_deleted_manifest_rewritten":
        (prior / "kept" / "P4.parquet").unlink()
        _rewrite_manifest(prior)
    elif defect == "kept_replaced_manifest_rewritten":
        pd.DataFrame({"x": [2]}).to_parquet(prior / "kept" / "P1.parquet", index=False)
        _rewrite_manifest(prior)
    elif defect == "other_protocol_in_the_record":
        record = json.loads(runs.read_text())
        record["A"]["protocol_sha256"] = "3" * 64
        runs.write_text(json.dumps(record))
    elif defect == "another_folder":
        record = json.loads(runs.read_text())
        record["A"]["out_dir"] = str(tmp / "somewhere_else")
        runs.write_text(json.dumps(record))
    elif defect == "record_not_an_object":
        runs.write_text(json.dumps({"A": "started, they say"}))
    elif defect in ("unfinished_record_with_a_manifest", "record_without_the_manifest_sha", "record_without_finished_utc",
                    "unfinished_record_complete_output_elsewhere", "unfinished_record_crashed_run_elsewhere"):
        record = json.loads(runs.read_text())                            # what a failed _record_finished leaves behind
        for field in {"unfinished_record_with_a_manifest": ("finished_utc", "manifest_sha256", "status"),
                      "record_without_the_manifest_sha": ("manifest_sha256",),
                      "record_without_finished_utc": ("finished_utc", "status"),
                      "unfinished_record_complete_output_elsewhere": ("finished_utc", "manifest_sha256", "status"),
                      "unfinished_record_crashed_run_elsewhere": ("finished_utc", "manifest_sha256", "status")}[defect]:
            record["A"].pop(field)
        runs.write_text(json.dumps(record))
        if defect.endswith("crashed_run_elsewhere"):
            (prior / "MANIFEST.sha256").unlink()                          # a crash leaves no manifest
        if defect.endswith("elsewhere"):
            where = tmp / "A"                                             # Test B is pointed at another folder than the record's
    elif defect == "no_record_no_folder":
        shutil.rmtree(prior)
    with pytest.raises(exc, match=match):
        run_sealed(_cfg_b(tmp, kw, where), sha)
    assert not (tmp / "outB").exists() and not (tmp / "A").exists()
    assert not (tmp / "RUN_B.claim").exists() and "B" not in (json.loads(runs.read_text()) if runs.exists() else {})


def test_test_b_looks_for_test_a_s_record_beside_test_a_s_folder_not_beside_its_own(registered, monkeypatch):
    """A rehearsal keeps its records in the folder that holds its output. Test B's output may be anywhere else; Test A's record
    is still found beside Test A's folder, and Test B's own is written beside Test B's."""
    from es.sealed_run import run_sealed
    write, tmp = registered
    proto, reg, sha, kw = _bound(tmp, write)
    _stop_at_the_read(monkeypatch)
    prior = _prior_dir(tmp, sha)                                    # Test A's folder and record are in tmp
    cfg = _cfg_b(tmp, kw, prior)
    cfg.out_dir = tmp / "elsewhere" / "B"
    with pytest.raises(RuntimeError, match="stop"):
        run_sealed(cfg, sha)
    state = json.loads([ln for ln in (tmp / "elsewhere" / "B" / "RUN_LOG.txt").read_text().splitlines()
                        if ln.startswith("test_a ")][0].split(" ", 1)[1])
    assert state["state"] == "pooled" and "B" in json.loads((tmp / "elsewhere" / "RUNS.json").read_text())
    assert set(json.loads((tmp / "RUNS.json").read_text())) == {"A"} and (tmp / "elsewhere" / "RUN_B.claim").exists()


def test_the_registered_test_b_finds_test_a_s_record_in_docs_sealed_and_nowhere_else(registered, monkeypatch):
    """The sealed Test B reads Test A's record from docs/sealed (es.registration's SEALED_DIR), as the sealed Test A wrote it;
    a record beside Test A's folder, which is where a rehearsal keeps its own, is not Test A's."""
    from es.sealed_pull import manifest_sha256
    from es.sealed_run import run_sealed
    write, tmp = registered
    proto, reg, sha, kw = _bound(tmp, write, real_code=True)
    _no_pull_check(monkeypatch)
    _no_read(monkeypatch)
    prior = _prior_dir(tmp, sha, folder=_out(tmp, "A"))             # Test A's registered folder; its record is beside it, in tmp
    with pytest.raises(ValueError, match="Test A has not been run"):
        run_sealed(_sealed_cfg(tmp, kw, test_id="B"), sha)
    _record_run(tmp / "docs_sealed", "A", prior, sha, STARTED_A, manifest_sha256(prior))
    _stop_at_the_read(monkeypatch)
    with pytest.raises(RuntimeError, match="stop"):
        run_sealed(_sealed_cfg(tmp, kw, test_id="B"), sha)
    assert set(_sealed_runs(tmp)) == {"A", "B"} and (tmp / "docs_sealed" / "RUN_B.claim").exists()
    assert _sealed_runs(tmp)["B"]["out_dir"] == str(_out(tmp, "B").resolve()) and set(json.loads(_runs_file(tmp).read_text())) == {"A"}
    log = (_out(tmp, "B") / "RUN_LOG.txt").read_text()
    assert '"state": "pooled"' in log and "sealed_manifest sealed_2026Q4" in log and "test B" in log


def test_test_b_with_a_sound_test_a_output_gets_through_to_the_data_read(registered, monkeypatch):
    from es.sealed_pull import manifest_sha256
    from es.sealed_run import run_sealed
    write, tmp = registered
    proto, reg, sha, kw = _bound(tmp, write)
    _stop_at_the_read(monkeypatch)
    prior = _prior_dir(tmp, sha)
    opened = _spy_parquet(monkeypatch)
    with pytest.raises(RuntimeError, match="stop"):
        run_sealed(_cfg_b(tmp, kw, prior), sha)
    assert opened == []                                   # Test A's stored rows are not read before the lock either
    log = (tmp / "outB" / "RUN_LOG.txt").read_text()
    assert log.startswith("started ") and "test B" in log
    state = json.loads([ln for ln in log.splitlines() if ln.startswith("test_a ")][0].split(" ", 1)[1])
    assert state["state"] == "pooled" and state["started_utc"] == STARTED_A
    assert state["manifest_sha256"] == manifest_sha256(prior)
    with pytest.raises(FileExistsError, match="runs once"):
        run_sealed(_cfg_b(tmp, kw, prior), sha)


@pytest.mark.parametrize("case", ["not_run", "unfinished", "claimed_only"])
def test_test_b_still_runs_when_test_a_left_nothing_to_pool(registered, monkeypatch, case):
    """Ruling 109, I4. Test A failed its coverage rule (NOT_RUN), or was claimed and never finished, or was claimed and never
    recorded: there is no stored row to pool, and Test B runs on its own window (H-E is then not confirmatory, below)."""
    from es.sealed_run import run_sealed
    write, tmp = registered
    proto, reg, sha, kw = _bound(tmp, write)
    _stop_at_the_read(monkeypatch)
    if case == "not_run":                                                   # a NOT_RUN results.json holds no H-B numbers
        prior = _prior_dir(tmp, sha, coverage={"ran": False, "usable_nodes": 20}, status="not run (coverage rule)",
                           **{"H-B1": None, "H-B2": None})
    elif case == "unfinished":
        prior = tmp / "A_out"
        prior.mkdir()
        (prior / "RUN_LOG.txt").write_text("started x\ncrashed y\n")
        _record_run(tmp, "A", prior, sha, STARTED_A, finished_utc=None)
    else:
        prior = tmp / "A_out"
        prior.mkdir()
        (tmp / "RUN_A.claim").write_text("{}")
    with pytest.raises(RuntimeError, match="stop"):
        run_sealed(_cfg_b(tmp, kw, prior), sha)
    log = (tmp / "outB" / "RUN_LOG.txt").read_text()
    state = json.loads([ln for ln in log.splitlines() if ln.startswith("test_a ")][0].split(" ", 1)[1])["state"]
    assert state == {"not_run": "not run (coverage rule)", "unfinished": "unfinished", "claimed_only": "claimed, not recorded"}[case]
    assert "B" in json.loads((tmp / "RUNS.json").read_text())


def _test_a_state(tmp, name="outB"):
    log = (tmp / name / "RUN_LOG.txt").read_text()
    return json.loads([ln for ln in log.splitlines() if ln.startswith("test_a ")][0].split(" ", 1)[1])


def test_a_complete_test_a_whose_record_lost_its_finish_fields_is_refused_with_the_repair_and_pools_once_repaired(
        registered, monkeypatch):
    """Ruling 116, N2 (the reviewer's R6). Test A's folder is complete (it has its manifest) but the update of its record
    did not reach RUNS.json: Test B would run alone and the pooled H-E would be lost. It is refused before its lock instead,
    with the repair: the three fields _record_finished names and where each value comes from. Repaired, it pools."""
    from es.sealed_pull import manifest_sha256
    from es.sealed_run import run_sealed
    write, tmp = registered
    proto, reg, sha, kw = _bound(tmp, write)
    _stop_at_the_read(monkeypatch)
    prior = _prior_dir(tmp, sha, finished_utc="2026-10-21T13:00:07+00:00", status="completed with secondary errors")
    runs = json.loads((tmp / "RUNS.json").read_text())
    want = {k: runs["A"].pop(k) for k in ("finished_utc", "manifest_sha256", "status")}
    (tmp / "RUNS.json").write_text(json.dumps(runs))
    with pytest.raises(RuntimeError) as err:
        run_sealed(_cfg_b(tmp, kw, prior), sha)
    message = str(err.value)
    assert "finished_utc" in message and "2026-10-21T13:00:07+00:00" in message and "results.json" in message
    assert manifest_sha256(prior) in message and "MANIFEST.sha256" in message and "completed with secondary errors" in message
    assert "run Test B again" in message
    assert not (tmp / "outB").exists() and not (tmp / "RUN_B.claim").exists()           # nothing spent
    runs["A"].update({"finished_utc": "2026-10-21T13:00:07+00:00", "manifest_sha256": manifest_sha256(prior),
                      "status": "completed with secondary errors"})                       # the repair, as the message says
    (tmp / "RUNS.json").write_text(json.dumps(runs))
    with pytest.raises(RuntimeError, match="stop"):
        run_sealed(_cfg_b(tmp, kw, prior), sha)
    assert _test_a_state(tmp)["state"] == "pooled" and want["manifest_sha256"] == manifest_sha256(prior)


def test_test_b_pools_h_e_with_test_a_s_p1_rows_when_its_other_stored_rows_are_missing(registered, monkeypatch):
    """Ruling 116, N1(b) (the reviewer's R5). Test A finished with kept/P1 and without E1-E4's frames (say its economic
    summaries failed after H-E's rows were written): H-E is pooled all the same, and the missing frames are listed."""
    from es.run_economics import KEPT
    from es.sealed_run import run_sealed
    write, tmp = registered
    proto, reg, sha, kw = _bound(tmp, write)
    _stop_at_the_read(monkeypatch)
    prior = _prior_dir(tmp, sha, no_kept=tuple(n for n in KEPT if n != "P1"))
    with pytest.raises(RuntimeError, match="stop"):
        run_sealed(_cfg_b(tmp, kw, prior), sha)
    state = _test_a_state(tmp)
    assert state["state"] == "pooled" and state["missing_kept"] == ["P4@r*", "P4", "calendar@count", "VOL@count"]
    assert state["started_utc"] == STARTED_A and "B" in json.loads((tmp / "RUNS.json").read_text())


def test_test_b_runs_alone_when_test_a_finished_without_its_p1_rows(registered, monkeypatch):
    """Ruling 116, N1(c) (the reviewer's R5). Nothing to pool H-E with: Test B runs alone, as it does for a NOT_RUN Test A, and
    says why in its own state."""
    from es.run_economics import KEPT
    from es.sealed_run import run_sealed
    write, tmp = registered
    proto, reg, sha, kw = _bound(tmp, write)
    _stop_at_the_read(monkeypatch)
    prior = _prior_dir(tmp, sha, no_kept=KEPT, **{"H-E": {"status": "interim", "p": 0.3}})
    with pytest.raises(RuntimeError, match="stop"):
        run_sealed(_cfg_b(tmp, kw, prior), sha)
    state = _test_a_state(tmp)
    assert state["state"] == "finished, without stored P1 rows" and state["missing_kept"] == list(KEPT)
    assert state["manifest_sha256"] and "B" in json.loads((tmp / "RUNS.json").read_text())


def test_a_not_run_test_a_still_has_to_be_the_recorded_run(registered, monkeypatch):
    """A NOT_RUN results.json is checked like any other (the record, the manifest, the start) before it ends Test A's part."""
    from es.registration import RegistrationError
    from es.sealed_run import run_sealed
    write, tmp = registered
    proto, reg, sha, kw = _bound(tmp, write)
    _no_read(monkeypatch)
    prior = _prior_dir(tmp, sha, coverage={"ran": False})
    record = json.loads((tmp / "RUNS.json").read_text())
    record["A"]["manifest_sha256"] = "0" * 64                               # not the manifest of this folder
    (tmp / "RUNS.json").write_text(json.dumps(record))
    with pytest.raises(RegistrationError, match="not the one recorded"):
        run_sealed(_cfg_b(tmp, kw, prior), sha)


def test_a_demoted_h_b_hypothesis_needs_nothing_from_test_a_and_a_confirmatory_one_needs_p_and_t_hat(registered, monkeypatch):
    """Ruling 109, I4. The registered family has H-E alone confirmatory: Test A's results.json may hold an error entry for
    H-B1 (its bootstrap failed there) and Test B still runs. If H-B1 were confirmatory, Holm would need its p."""
    from es.sealed_run import run_sealed
    write, tmp = registered
    proto, reg, sha, kw = _bound(tmp, write)
    family = {**FAMILY, "H-B1": {"confirmatory": False, "reasons": ["size 0.19"], "approximate": True},
              "H-B2": {"confirmatory": False, "reasons": ["size 0.19"], "approximate": True}}
    kw["inputs"].write_text(json.dumps({"eta": 0.01, "family": family}))
    proto, reg, sha = write(_protocol_lines(kw["cutoffs"], kw["inputs"], kw["code_manifest"]))
    kw["registration"] = (proto, reg)
    prior = _prior_dir(tmp, sha, **{"H-B1": {"error": "ValueError: the contrast has nothing to infer from"}, "H-B2": {}})
    _stop_at_the_read(monkeypatch)
    with pytest.raises(RuntimeError, match="stop"):
        run_sealed(_cfg_b(tmp, kw, prior), sha)                                # reached the read: nothing was needed of H-B1


# =====================================================================================================
# The pieces of the run, tested on their own.
# =====================================================================================================
def test_the_34_score_sets_cover_everything_the_audit_asks_for(monkeypatch):
    """audit_family asks for 18 contrasts' frames (Ruling 37); run_sealed must have fitted every one of them, and
    economics_for's two (G1_24 and G6, xgb, seed 42, rolling) must be kept in full."""
    import es.secondary as secondary
    from es.blocks import BLOCKS
    from es.cutoffs import CUTOFF_BLOCKS
    from es.sealed_run import FULL, score_keys
    keys = score_keys()
    assert len(keys) == 34 == len(set(keys))
    assert sum(k[0] == "roll" and k[2] == "xgb" and k[3] == 42 for k in keys) == len(BLOCKS) == 10
    assert sorted(k[1] for k in keys if k[2] == "logit") == sorted(("DART3", "LEV6", "ROLL_DISP9", "G1_24"))
    assert sorted(k[1] for k in keys if k[0] == "frozen") == sorted(CUTOFF_BLOCKS)
    asked = []
    monkeypatch.setattr(secondary, "block_contrast_bootstrap", lambda *_a, **_k: {"t_hat": 0.0, "p": 1.0})
    secondary.audit_family(lambda kind, block, est, seed: asked.append((kind, block, est, seed)), n_rep=1)
    assert len(asked) == 36 and set(asked) <= set(keys)
    assert set(FULL) == {("roll", "G1_24", "xgb", 42), ("roll", "G6", "xgb", 42)} and set(FULL) <= set(keys)


def test_the_score_lookup_never_reaches_the_cache_and_raises_for_an_unknown_key():
    from es.analyses import SEALED
    from es.sealed_run import SealedConfig, _score_lookup
    cfg = SealedConfig("A", SEALED["SA"], SEALED["SAF"], [], "2026-09-30")
    sc = {("roll", "G1_24", "xgb", 42): "roll frame", ("frozen", "G1_24", "xgb", 42): "frozen frame"}
    lookup = _score_lookup(cfg, sc)
    assert lookup("panel", cfg.rolling, "G1_24", "xgb", 42) == "roll frame"
    assert lookup("panel", cfg.frozen, "G1_24", "xgb", 42, label_alpha=0.05) == "frozen frame"
    with pytest.raises(KeyError):
        lookup("panel", cfg.rolling, "G6", "xgb", 42)                      # not precomputed: never fitted on demand
    with pytest.raises(KeyError):
        lookup("panel", cfg.rolling, "G1_24", "xgb", 7)


def test_the_rolling_and_frozen_arms_are_told_apart_by_identity():
    from es.analyses import ANALOGS
    from es.sealed_run import SealedConfig, _score_lookup
    a = ANALOGS["AQ1"]
    cfg = SealedConfig("A", a, a, [], "2026-03-31")                          # the rehearsal: the frozen arm is the rolling one
    lookup = _score_lookup(cfg, {("roll", "G6", "xgb", 42): "roll", ("frozen", "G6", "xgb", 42): "frozen"})
    assert lookup(None, a, "G6", "xgb", 42) == "roll"


def _tiny_scores(markets=("M", "N")):
    """Score frames shaped like SEALED['SA']'s: training in July-August 2024, calibration in July 2025, deployment in
    July 2026, one node per market."""
    return _scores_in(markets, (("train", dt.date(2024, 7, 1), 60), ("calib", dt.date(2025, 7, 1), 30),
                                ("deploy", dt.date(2026, 7, 1), 30)))


def _scores_in(markets, layout, seed=3):
    """One node per market, named by position (A, B, ...), so that the category product of (market, node) has pairs that do
    not exist: that is what makes a groupby without observed=True add phantom rows."""
    rng = np.random.default_rng(seed)
    rows = []
    for i, m in enumerate(markets):
        for split, start, n in layout:
            for d in range(n):
                day = start + dt.timedelta(days=d)
                for h in range(24):
                    rows.append({"market": m, "node": chr(ord("A") + i), "split": split, "settle_date": day,
                                 "timestamp_utc": pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=h),
                                 "score": rng.normal(), "label": float(rng.random() < 0.05)})
    return pd.DataFrame(rows)


def test_slim_keeps_seven_columns_with_categorical_keys_and_plain_restores_objects():
    from es.sealed_run import SLIM, _plain, _slim
    full = _tiny_scores().assign(extra=1.0, da_lmp=2.0)
    slim = _slim(full)
    assert list(slim.columns) == SLIM and all(isinstance(slim[c].dtype, pd.CategoricalDtype)
                                              for c in ("market", "node", "split"))
    plain = _plain(slim)
    assert plain["market"].dtype == object and plain["node"].dtype == object
    assert isinstance(plain["split"].dtype, pd.CategoricalDtype)       # the default restores market and node only
    assert isinstance(slim["market"].dtype, pd.CategoricalDtype)       # the input is left alone
    every = _plain(slim, ("market", "node", "split"))
    assert every["split"].dtype == object
    assert every[["market", "node", "split", "score"]].equals(full[["market", "node", "split", "score"]])


def test_as_calib_relabels_the_p2_training_rows_and_keeps_deployment():
    from es.sealed_run import _as_calib
    s = _tiny_scores(("M",))
    out = _as_calib(s, {7})                                                  # July only: the first 31 training days
    assert set(out["split"]) == {"calib", "deploy"}
    july_train = ((s["split"] == "train") & s["settle_date"].map(lambda d: d.month == 7)).sum()
    assert july_train == 31 * 24 and (out["split"] == "calib").sum() == july_train
    assert (out["split"] == "deploy").sum() == (s["split"] == "deploy").sum()
    assert len(out) == july_train + (s["split"] == "deploy").sum()          # the real calibration rows are gone


def test_coverage_counts_deployment_hours_per_node_against_the_expected_clock_hours():
    from es.analyses import SEALED
    from es.sealed_pull import expected_hours
    from es.sealed_run import coverage
    a = SEALED["SA"]
    assert {expected_hours(m, a.deploy.start, a.deploy.end) for m in NODE} == {2208}       # 92 days of 24 hours

    def block(market, node, hours):
        days = [d.date() for d in pd.date_range("2026-07-01", periods=-(-hours // 24), freq="D")]
        return pd.DataFrame({"market": market, "node": node,
                             "settle_date": [d for d in days for _ in range(24)][:hours]})

    outside = pd.DataFrame({"market": ["MISO"], "node": ["ILLINOIS.HUB"], "settle_date": [dt.date(2026, 6, 30)]})
    panel = pd.concat([block("NYISO", "N.Y.C.", 2208), block("NYISO", "WEST", 1987),       # 90% of 2208 is 1987.2
                       block("ISONE", ".H.INTERNAL_HUB", 1988), outside,
                       block("CAISO", "TH_SP15_GEN-APND", 2208)], ignore_index=True)
    cov = coverage(panel, a).set_index(["market", "node"])
    assert cov.loc[("NYISO", "N.Y.C."), "hours"] == 2208 and cov.loc[("NYISO", "N.Y.C."), "ok"]
    assert cov.loc[("NYISO", "WEST"), "hours"] == 1987 and not cov.loc[("NYISO", "WEST"), "ok"]   # just under 90%
    assert cov.loc[("ISONE", ".H.INTERNAL_HUB"), "ok"]
    assert cov.loc[("ISONE", ".H.INTERNAL_HUB"), "share"] == pytest.approx(1988 / 2208)
    assert cov.loc[("MISO", "ILLINOIS.HUB"), "hours"] == 0 and not cov.loc[("MISO", "ILLINOIS.HUB"), "ok"]
    assert set(cov["expected_hours"]) == {2208} and len(cov) == 5


def test_coverage_expected_hours_follow_the_market_clock_on_a_fall_back_quarter():
    from es.analyses import SEALED
    from es.sealed_run import coverage
    panel = pd.DataFrame({"market": ["CAISO", "MISO"], "node": ["TH_SP15_GEN-APND", "ILLINOIS.HUB"],
                          "settle_date": [dt.date(2026, 11, 1)] * 2})
    cov = coverage(panel, SEALED["SB"]).set_index("market")             # 2026-10-01 .. 2026-12-31; DST ends 1 November
    assert cov.loc["CAISO", "expected_hours"] == 2209 and cov.loc["MISO", "expected_hours"] == 2208


def test_a_node_that_lacks_only_the_repeated_hour_of_the_fall_back_day_is_covered():
    """The NYISO and ISO-NE downloaders drop the two rows of the repeated hour of 1 November, so those nodes hold 2207 of
    Test B's 2209 clock hours (a share of 0.999; COVERAGE.csv says so too). The coverage rule is 90%, not 100%."""
    from es.analyses import SEALED
    from es.sealed_pull import expected_hours
    from es.sealed_run import coverage
    a = SEALED["SB"]
    assert expected_hours("NYISO", a.deploy.start, a.deploy.end) == 2209
    days = [d.date() for d in pd.date_range("2026-10-01", "2026-12-31", freq="D")]
    rows = [d for d in days for _ in range(24)] + [dt.date(2026, 11, 1)]               # 2209 hours: the day has 25 ...
    panel = pd.concat([pd.DataFrame({"market": "NYISO", "node": "N.Y.C.", "settle_date": rows[:-2]}),   # ... the node keeps 23 of them
                       pd.DataFrame({"market": "CAISO", "node": "TH_SP15_GEN-APND", "settle_date": rows})], ignore_index=True)
    cov = coverage(panel, a).set_index("market")
    assert cov.loc["NYISO", "hours"] == 2207 and cov.loc["NYISO", "share"] == pytest.approx(2207 / 2209) and cov.loc["NYISO", "ok"]
    assert cov.loc["CAISO", "share"] == 1.0 and cov.loc["CAISO", "ok"]


def _cut_cfg(tmp_path, rows):
    """A config whose registered cutoff table is `rows` and whose two arms are both SA (the rehearsal's shape)."""
    from es.analyses import SEALED
    from es.sealed_run import SealedConfig
    cut = tmp_path / "cutoffs.csv"
    pd.DataFrame(rows).to_csv(cut, index=False)
    return SealedConfig("A", SEALED["SA"], SEALED["SA"], [], "2026-09-30", cutoffs=cut)


def _tau_inputs():
    from es.analyses import SEALED
    from es.cutoffs import CUTOFF_BLOCKS, cutoffs_from_scores
    s = _tiny_scores()
    rows = [r for b in CUTOFF_BLOCKS for r in cutoffs_from_scores(s, SEALED["SA"], b)]
    sc = {(kind, b, "xgb", 42): s for kind in ("roll", "frozen") for b in CUTOFF_BLOCKS}
    return s, rows, sc


def test_tau_check_is_exactly_zero_on_the_cutoffs_it_registered(tmp_path):
    from es.sealed_run import _tau_check
    s, rows, sc = _tau_inputs()
    got = _tau_check(sc, _cut_cfg(tmp_path, rows))
    assert got == {"max_abs_diff": 0.0, "nan_pattern_mismatches": 0, "n_compared": 8,
                   "registered_only": 0, "recomputed_only": 0}


def test_tau_check_reports_a_difference_a_nan_pattern_and_unmatched_rows(tmp_path):
    from es.sealed_run import _tau_check
    s, rows, sc = _tau_inputs()
    reg = pd.DataFrame(rows)
    reg.loc[0, "tau_p1"] += 0.25                                           # one cutoff differs by 0.25
    reg.loc[1, "tau_p2_season"] = np.nan                                   # one is NaN where the run has a number
    reg = reg.drop(index=7)                                                # one node-block the registered table lacks
    reg = pd.concat([reg, reg.iloc[[0]].assign(node="GONE"),               # a registered node the run does not have
                     pd.DataFrame([{"analysis": "AQ1", "block": "G1_24", "market": "M", "node": "A"}])])  # another analysis
    got = _tau_check(sc, _cut_cfg(tmp_path, reg.to_dict("records")))
    assert got["max_abs_diff"] == pytest.approx(0.25) and got["nan_pattern_mismatches"] == 1
    assert got == {"max_abs_diff": got["max_abs_diff"], "nan_pattern_mismatches": 1, "n_compared": 7,
                   "registered_only": 1, "recomputed_only": 1}             # AQ1 is not part of this run: not compared


def test_tau_check_with_nothing_to_compare_is_nan_not_an_error(tmp_path):
    from es.sealed_run import _tau_check
    s, rows, sc = _tau_inputs()
    cfg = _cut_cfg(tmp_path, [{"analysis": "AQ1", "block": "G1_24", "market": "M", "node": "A", "tau_p1": 1.0,
                               "tau_p2_season": 1.0, "tau_p2_all": 1.0}])
    got = _tau_check(sc, cfg)
    assert np.isnan(got["max_abs_diff"]) and got["n_compared"] == 0 and got["recomputed_only"] == 8


def test_audit_runs_every_policy_on_slim_frames_with_categorical_keys():
    """The audit table: P1 on all 34 score sets (a plain copy of each, because es.policies adds one n = 0 row per
    unobserved pair on categorical keys) and the other six policies on the primary pipeline."""
    from es.analyses import SEALED
    from es.sealed_run import SealedConfig, _audit, _slim, score_keys
    s = _tiny_scores()
    cfg = SealedConfig("A", SEALED["SA"], SEALED["SAF"], [], "2026-09-30")
    sc = {k: _slim(s) for k in score_keys()}
    sc[("roll", "G1_24", "xgb", 42)] = s                                   # the primary frame is kept in full, object keys
    out = _audit(sc, cfg, eta=0.01)
    assert len(out) == (34 + 6) * 2 and out["n"].gt(0).all()               # two nodes; no phantom (market, node) pairs
    assert set(out["policy"]) == {"P1", "P4_56", "P4_28", "P5", "P6", "P2_season", "P2_all"}
    p1 = out[out["policy"] == "P1"]
    assert len(p1) == 34 * 2 and set(zip(p1["maintenance"], p1["estimator"])) == {
        ("roll", "xgb"), ("roll", "logit"), ("frozen", "xgb")}
    assert sorted(set(p1["seed"])) == [0, 1, 7, 13, 42]


def test_audit_policies_take_the_registered_parameters():
    """P4 at 56 and 28 days, P5 at the registered eta, P2 over the deployment season's months and over the year, P6 exact:
    each audit row equals the policy called with those parameters on the primary frame."""
    from es.analyses import SEALED
    from es.policies import activity, p1_fixed, p2_insample, p4_rolling, p5_self_correcting, p6_exact
    from es.sealed_run import SealedConfig, _audit, _slim, score_keys
    s = _scores_in(("M", "N"), (("train", dt.date(2024, 3, 1), 150), ("calib", dt.date(2026, 5, 1), 61),
                                ("deploy", dt.date(2026, 7, 1), 61)))        # deployment follows calibration: P4's windows differ
    cfg = SealedConfig("A", SEALED["SA"], SEALED["SAF"], [], "2026-09-30")
    sc = {k: _slim(s) for k in score_keys()}
    sc[("roll", "G1_24", "xgb", 42)] = s
    out = _audit(sc, cfg, eta=0.03)
    want = {"P1": activity(p1_fixed(s, 0.05), 0.05), "P4_56": activity(p4_rolling(s, 0.05, 56), 0.05),
            "P4_28": activity(p4_rolling(s, 0.05, 28), 0.05), "P5": activity(p5_self_correcting(s, 0.05, 0.03), 0.05),
            "P6": activity(p6_exact(s, 0.05), 0.05), "P2_season": activity(p2_insample(s, 0.05, {7, 8, 9}), 0.05),
            "P2_all": activity(p2_insample(s, 0.05, set(range(1, 13))), 0.05)}
    for policy, frame in want.items():
        got = out[(out["policy"] == policy) & (out["maintenance"] == "roll") & (out["block"] == "G1_24")
                  & (out["estimator"] == "xgb") & (out["seed"] == 42)]
        pd.testing.assert_frame_equal(got[list(frame.columns)].reset_index(drop=True), frame)
    assert want["P4_56"]["alerts"].tolist() != want["P4_28"]["alerts"].tolist()       # the windows are not interchangeable here
    assert want["P2_season"]["alerts"].tolist() != want["P2_all"]["alerts"].tolist()
    assert out["policy"].eq("P1").sum() == 34 * 2 and len(out) == (34 + 6) * 2


def test_secondary_counts_tests_by_the_rows_bh_adjusted_not_by_the_rows():
    """Ruling 80: a NaN p-value is no test that ran, so it is in `rows` and not in `tests`."""
    from es.secondary import bh_table
    from es.sealed_run import _secondary_counts
    rows = [{"family": "audit", "test": "a", "estimate": 1.0, "p": 0.001}, {"family": "audit", "test": "b", "estimate": 1.0, "p": 0.9},
            {"family": "audit", "test": "c", "estimate": 1.0, "p": float("nan")},
            {"family": "monitors", "test": "a", "estimate": 1.0, "p": float("nan")},
            {"family": "econ", "test": "a", "estimate": 1.0, "p": 0.01}]
    assert _secondary_counts(bh_table(rows)) == {"audit": {"tests": 2, "rows": 3, "bh_rejections": 1},
                                                 "econ": {"tests": 1, "rows": 1, "bh_rejections": 1},
                                                 "monitors": {"tests": 0, "rows": 1, "bh_rejections": 0}}
    assert _secondary_counts(bh_table([])) == {}


# --- the decision rule's wording, given the numbers (_decide is pure)
HE = {"t_obs": 3.0, "p": 0.01, "n_null": 100, "selc_obs": 0.05, "retained_share": [{"min": 1.0, "mean": 1.0}]}


def _hb(p1, t1, p2, t2, confirmatory=(True, True)):
    return {"H-B1": {"t_hat": t1, "p": p1, "confirmatory": confirmatory[0]},
            "H-B2": {"t_hat": t2, "p": p2, "confirmatory": confirmatory[1]}}


def _family(he_interim_approx=False, reasons=("size too big", "power 0.4"), he_approx=False):
    return {"H-E": {"confirmatory": True, "reasons": [], "approximate": he_approx, "interim_approximate": he_interim_approx},
            "H-B1": {"confirmatory": True, "reasons": [], "approximate": False},
            "H-B2": {"confirmatory": False, "reasons": list(reasons), "approximate": True}}


def test_test_a_judges_each_confirmatory_h_b_test_at_alpha_over_the_family_size():
    from es.sealed_run import _decide
    hb = _hb(0.016, 0.3, 0.017, 0.2)                                  # 0.05 / 3 = 0.016667
    out, holm = _decide(HE, hb, _family(True), ["H-E", "H-B1", "H-B2"], None)
    assert holm is None
    assert hb["H-B1"]["status"] == "rejected in Test A at 0.05/3; replication pending"
    assert hb["H-B2"]["status"] == "not rejected at 0.05/3 in Test A; full Holm after Test B"
    assert out["status"].startswith("interim") and out["approximate"] is True and out["confirmatory"] is True
    assert {k: out[k] for k in HE} == HE and "wording" not in out and "replicated" not in hb["H-B1"]
    hb = _hb(0.02, 0.3, 0.2, 0.2)
    _decide(HE, hb, _family(), ["H-B1", "H-B2"], None)                 # without H-E in the family: alpha / 2 = 0.025
    assert hb["H-B1"]["status"] == "rejected in Test A at 0.05/2; replication pending"
    assert hb["H-B2"]["status"].startswith("not rejected at 0.05/2")
    out, _ = _decide(HE, _hb(0.5, 0, 0.5, 0), _family(), ["H-B1", "H-B2"], None)
    assert out["confirmatory"] is False and out["approximate"] is False       # demoted to an estimand, still reported


def test_test_a_rejects_at_exactly_alpha_over_m():
    from es.sealed_run import _decide
    hb = _hb(0.05 / 3, 0.3, 0.05 / 3 + 1e-9, 0.2)
    _decide(HE, hb, _family(), ["H-E", "H-B1", "H-B2"], None)
    assert hb["H-B1"]["status"].startswith("rejected in Test A") and hb["H-B2"]["status"].startswith("not rejected")


def test_test_a_reports_a_demoted_h_b_test_as_an_estimand_with_its_reasons():
    from es.sealed_run import _decide
    hb = _hb(0.0001, 0.3, 0.0001, 0.2, confirmatory=(True, False))
    _decide(HE, hb, _family(), ["H-E", "H-B1"], None)
    assert hb["H-B2"]["status"] == "estimand (registration rule): size too big; power 0.4"
    assert hb["H-B1"]["status"] == "rejected in Test A at 0.05/2; replication pending"


def _prior(p1, t1, p2, t2):
    return {"H-B1": {"p": p1, "t_hat": t1}, "H-B2": {"p": p2, "t_hat": t2}}


def test_test_b_runs_holm_over_the_confirmatory_family_with_the_pooled_h_e_p_value():
    from es.sealed_run import _decide
    he = {**HE, "p": 0.001}
    hb = _hb(0.01, 0.4, 0.5, 0.1)                                      # Test B: H-B1 significant with Test A's sign
    out, holm = _decide(he, hb, _family(), ["H-E", "H-B1", "H-B2"], _prior(0.02, 0.2, 0.2, -0.1))
    # sorted p: 0.001 <= 0.05/3, 0.02 <= 0.05/2, 0.2 > 0.05/1: H-E and H-B1 rejected, H-B2 not
    assert holm == {"H-E": True, "H-B1": True, "H-B2": False}
    assert out["status"] == "decided (pooled A+B)" and out["wording"] == "timing skill within calendar cells detected"
    assert hb["H-B1"]["replicated"] is True and hb["H-B1"]["status"] == "larger operating-point error in both 2026 quarters"
    assert hb["H-B2"]["replicated"] is False and hb["H-B2"]["status"] == "no difference in operating-point error detected"
    assert {k: out[k] for k in HE if k != "p"} == {k: HE[k] for k in HE if k != "p"} and out["p"] == 0.001


def test_test_b_wording_when_h_e_is_not_rejected_or_not_replicated():
    from es.sealed_run import _decide
    he = {**HE, "p": 0.3}
    hb = _hb(0.2, 0.4, 0.01, -0.1)
    out, holm = _decide(he, hb, _family(), ["H-E", "H-B1", "H-B2"], _prior(0.001, 0.2, 0.2, 0.3))
    # sorted p: H-B1 0.001 <= 0.05/3 is rejected; H-B2 0.2 > 0.05/2 stops the step-down; H-E (0.3) is not rejected
    assert holm == {"H-E": False, "H-B1": True, "H-B2": False}
    # H-B1: Test B has Test A's sign but p = 0.2 >= 0.05: it does not replicate
    assert hb["H-B1"]["replicated"] is False
    assert hb["H-B1"]["status"] == "larger operating-point error in Q3 2026; not replicated in Q4"
    assert out["wording"] == "timing skill within calendar cells not detected"
    assert hb["H-B2"]["replicated"] is False and hb["H-B2"]["status"] == "no difference in operating-point error detected"


@pytest.mark.parametrize("t_b,t_a,p_b,replicated", [
    (0.5, 0.2, 0.049, True), (0.5, 0.2, 0.05, False),                  # p < alpha, strictly
    (-0.5, 0.2, 0.001, False), (0.0, 0.2, 0.001, False),               # opposite sign; no effect at all
    (0.5, 0.0, 0.001, False), (float("nan"), 0.2, 0.001, False), (-0.5, -0.2, 0.001, True),
    (0.0, 0.0, 0.001, False)])                                         # no effect in either quarter is not a replication
def test_test_b_replication_needs_the_same_sign_and_p_below_alpha(t_b, t_a, p_b, replicated):
    from es.sealed_run import _decide
    hb = _hb(p_b, t_b, 0.9, 0.0)
    _decide(HE, hb, _family(), ["H-E", "H-B1", "H-B2"], _prior(0.001, t_a, 0.9, 0.0))
    assert hb["H-B1"]["replicated"] is replicated


def test_test_b_without_a_confirmatory_family_reports_estimands_only():
    from es.sealed_run import _decide
    hb = _hb(0.001, 0.4, 0.001, 0.4, confirmatory=(False, False))
    out, holm = _decide(HE, hb, _family(), [], _prior(0.001, 0.2, 0.001, 0.2))
    assert holm == {"H-E": None, "H-B1": None, "H-B2": None}
    assert out["confirmatory"] is False and out["wording"] == "estimand (registration rule): pooled SELC with CI"
    # Ruling 109, M11: Test B's status keeps the reasons of the registration rule, as Test A's does; no reason, no colon
    assert hb["H-B1"]["status"] == "estimand (registration rule)"
    assert hb["H-B2"]["status"] == "estimand (registration rule): size too big; power 0.4"
    assert all(hb[h]["replicated"] is None for h in hb)                  # an estimand has no replication verdict (I4, I5)
    hb = _hb(0.001, 0.4, 0.001, 0.4, confirmatory=(True, False))       # H-E demoted, H-B1 kept: Holm over H-B1 alone
    out, holm = _decide(HE, hb, _family(), ["H-B1"], _prior(0.04, 0.2, 0.001, 0.2))
    assert holm == {"H-E": None, "H-B1": True, "H-B2": None} and out["wording"].startswith("estimand")
    assert hb["H-B1"]["status"] == "larger operating-point error in both 2026 quarters" and hb["H-B1"]["replicated"] is True
    assert hb["H-B2"]["status"] == "estimand (registration rule): size too big; power 0.4" and hb["H-B2"]["replicated"] is None


def test_a_demoted_h_b_hypothesis_needs_nothing_of_test_a_in_the_decision():
    """Ruling 109, I4 and I5. Test A's p and t_hat are read only for the confirmatory H-B hypotheses: a demoted one may have an
    error entry there, and Test B decides H-E and the confirmatory family without it."""
    from es.sealed_run import _decide
    hb = {"H-B1": {"t_hat": 0.4, "confirmatory": False, "diagnostics": {"bootstrap_p_one_sided": 0.001}},
          "H-B2": {"t_hat": 0.4, "confirmatory": False, "diagnostics": {"bootstrap_p_one_sided": 0.001}}}
    prior = {"H-B1": {"error": "ValueError: nothing to infer from"}, "H-B2": {}}
    out, holm = _decide({**HE, "p": 0.01}, hb, _family(), ["H-E"], prior)
    assert holm == {"H-E": True, "H-B1": None, "H-B2": None} and out["status"] == "decided (pooled A+B)"
    for hid, reasons in (("H-B1", ""), ("H-B2", ": size too big; power 0.4")):
        assert hb[hid]["status"] == "estimand (registration rule)" + reasons and hb[hid]["replicated"] is None
        assert "p" not in hb[hid]


def test_test_b_alone_tests_h_e_on_its_own_window_and_is_not_confirmatory():
    """Ruling 109, I4. Test A's output is not available (prior None, test B): H-E is not confirmatory, nothing replicates, no
    step-down runs. The state says why (Ruling 116, Mn6); the wording does not claim that Test A did not run."""
    from es.sealed_run import _decide
    hb = _hb(0.001, 0.4, 0.001, 0.4, confirmatory=(True, False))
    out, holm = _decide({**HE, "p": 0.001}, hb, _family(he_interim_approx=True, he_approx=False), ["H-E", "H-B1"], None, "B")
    assert out["status"] == "not confirmatory (Test A's output is not available; the registered pooled test is unavailable)"
    assert out["confirmatory"] is False and out["approximate"] is True           # one window: the interim flag
    assert out["wording"] == "estimand (Test A's output is not available): SELC of the Test B window alone" and out["p"] == 0.001
    assert holm == {"H-E": None, "H-B1": None, "H-B2": None}
    assert hb["H-B1"]["status"] == "undetermined: Test A's output is not available, so there is no replication and no step-down"
    assert hb["H-B2"]["status"] == "estimand (registration rule): size too big; power 0.4"
    assert hb["H-B1"]["replicated"] is None and hb["H-B2"]["replicated"] is None
    out, holm = _decide(None, _hb(0.5, 0, 0.5, 0, confirmatory=(False, False)), _family(), [], None, "B")
    assert out is None and holm == {"H-E": None, "H-B1": None, "H-B2": None}     # H-E failed: nothing else changes
    out_a, _ = _decide(HE, _hb(0.5, 0, 0.5, 0), _family(), ["H-E"], None)         # the same arguments in Test A: interim
    assert out_a["status"].startswith("interim")


def test_a_demoted_hypothesis_is_left_out_of_the_holm_family_whatever_its_p_value():
    """Holm runs over the confirmatory hypotheses only. H-E's pooled p is 0.9 and H-B2's is 0.001; with H-E and H-B2
    demoted, H-B1 (0.04) is alone in its family and is rejected at 0.05."""
    from es.sealed_run import _decide
    hb = _hb(0.001, 0.4, 0.001, 0.4, confirmatory=(True, False))
    out, holm = _decide({**HE, "p": 0.9}, hb, _family(), ["H-B1"], _prior(0.04, 0.2, 0.001, 0.2))
    assert holm == {"H-E": None, "H-B1": True, "H-B2": None}
    hb = _hb(0.001, 0.4, 0.001, 0.4, confirmatory=(True, True))        # all three in: 0.001 first, then 0.04 > 0.05/2
    _, holm = _decide({**HE, "p": 0.9}, hb, _family(), ["H-E", "H-B1", "H-B2"], _prior(0.04, 0.2, 0.001, 0.2))
    assert holm == {"H-E": False, "H-B1": False, "H-B2": True}


def test_the_loader_checks_its_own_window(registered):
    """The loader checks the window as well, with the claim it needs: a window past the test's own end, or a later test's
    directory, is refused before anything is read."""
    from es.analyses import SEALED
    from es.sealed_run import SealedConfig, _check_window, _load_sealed_raw
    write, tmp = registered
    proto, reg, sha = write()
    base = dict(registration=(proto, reg), out_dir=tmp / "out")
    cfg = SealedConfig("A", SEALED["SA"], SEALED["SAF"], [], "2026-10-01", **base)
    claim = _claimed(cfg, sha)
    with pytest.raises(ValueError, match="after Test A's own deployment end"):
        _load_sealed_raw(cfg, sha, claim)
    cfg = SealedConfig("A", SEALED["SA"], SEALED["SAF"], ["sealed_2026Q4"], "2026-09-30", **base)
    with pytest.raises(ValueError, match="sealed_2026Q4"):
        _load_sealed_raw(cfg, sha, claim)
    cfg = SealedConfig("B", SEALED["SB"], SEALED["SBF"], ["sealed_2026Q3", "sealed_2026Q4"], "2026-12-31")
    assert _check_window(cfg) == dt.date(2026, 12, 31)                     # Test B may read both pulls


def test_safe_names_and_the_kept_round_trip_through_parquet(tmp_path):
    from es.run_economics import KEPT
    from es.sealed_run import _load_kept, _safe
    assert [_safe(n) for n in KEPT] == ["P1", "P4_at_rstar", "P4", "calendar_at_count", "VOL_at_count"]
    assert len({_safe(n) for n in KEPT}) == len(KEPT)
    frame = pd.DataFrame({"market": ["M"], "node": ["A"], "settle_date": [dt.date(2026, 7, 1)], "x": [1.5],
                          "timestamp_utc": [pd.Timestamp("2026-07-01 05:00", tz="UTC")]})
    for name in KEPT:
        frame.assign(name=name).to_parquet(tmp_path / f"{_safe(name)}.parquet", index=False)
    back = _load_kept(tmp_path)
    assert list(back) == list(KEPT) and back["P4@r*"]["name"].iloc[0] == "P4@r*"
    assert back["P1"]["settle_date"].iloc[0] == dt.date(2026, 7, 1)          # a date object, as the cube's days need
    assert back["P1"]["timestamp_utc"].iloc[0] == pd.Timestamp("2026-07-01 05:00", tz="UTC")


def test_concat_opt_skips_missing_tables_and_drops_repeated_hours():
    from es.sealed_run import _concat_opt
    ts = pd.to_datetime(["2026-06-25 18:00", "2026-06-25 19:00"], utc=True)
    a = pd.DataFrame({"node": "X", "timestamp_utc": ts, "v": [1.0, 2.0]})
    b = pd.DataFrame({"node": "X", "timestamp_utc": ts[1:], "v": [9.0]})
    assert _concat_opt([None, None]) is None and _concat_opt([]) is None
    both = _concat_opt([None, a, b])
    assert len(both) == 2 and both.loc[both["timestamp_utc"] == ts[1], "v"].iloc[0] == 2.0    # the first table wins


def test_results_json_keeps_numbers_numbers_and_the_manifest_comes_last(tmp_path):
    from es.sealed_run import _finish
    out = tmp_path / "o"
    out.mkdir()
    (out / "RUN_LOG.txt").write_text("started x\n")
    res = _finish(out, {"n": np.int64(5), "x": np.float64(0.5), "ok": np.bool_(True), "d": dt.date(2026, 7, 1),
                        "nan": float("nan"), "p": Path("/a/b"), "s": {"b", "a"}})
    back = json.loads((out / "results.json").read_text())
    assert back["n"] == 5 and isinstance(back["n"], int) and back["x"] == 0.5 and back["ok"] is True
    assert back["d"] == "2026-07-01" and np.isnan(back["nan"]) and back["p"] == "/a/b" and back["s"] == ["a", "b"]
    assert res["finished_utc"] == back["finished_utc"]
    assert (out / "RUN_LOG.txt").read_text().splitlines()[-1] == f"finished {res['finished_utc']}"
    from es.sealed_pull import verify_manifest
    verify_manifest(out)                                                     # the manifest lists the finished log too


def test_default_config_for_each_test(registered):
    import es.registration as registration
    from es.analyses import SEALED
    from es.sealed_run import default_config
    write, tmp = registered
    a, b = default_config("A"), default_config("B")
    assert (a.rolling, a.frozen) == (SEALED["SA"], SEALED["SAF"]) and (b.rolling, b.frozen) == (SEALED["SB"], SEALED["SBF"])
    assert a.sealed_dirs == ["sealed_2026Q3"] and b.sealed_dirs == ["sealed_2026Q3", "sealed_2026Q4"]
    assert (a.window_end, b.window_end) == ("2026-09-30", "2026-12-31")
    assert a.prior_test_dir is None and Path(b.prior_test_dir) == Path(a.out_dir)
    assert a.out_dir.name == "A" and b.out_dir.name == "B" and a.out_dir.parent.name == "sealed"
    # the registration, the bound files and the pull record all come from es.registration as it is when the config is made
    assert a.registration == (registration.PROTOCOL_PATH, registration.REGISTRATION_PATH)
    assert a.registration[0] == tmp / "PROTOCOL.md"                       # the fixture's temporary protocol, not docs/sealed
    assert (a.cutoffs, a.inputs) == (registration.CUTOFFS_PATH, registration.INPUTS_PATH)
    assert a.code_manifest == registration.SEALED_DIR / "CODE_MANIFEST.sha256" and a.code_root == DEV_ROOT
    assert a.dev_dirs == __import__("es.config", fromlist=["DEV_DATA_DIRS"]).DEV_DATA_DIRS


def test_the_hypothesis_pairs_and_seeds_are_the_secondary_families():
    from es import secondary
    from es.sealed_run import HB_PAIRS, LOGIT_BLOCKS, SEEDS
    from es.secondary import AUDIT_VARIANTS
    assert HB_PAIRS == secondary.HB_PAIRS
    assert sorted(v[3] for v in AUDIT_VARIANTS if v[0].startswith("seed=")) == sorted(SEEDS)
    assert {"DART3", "LEV6", "ROLL_DISP9"} <= set(LOGIT_BLOCKS)


# =====================================================================================================
# Controller Ruling 104: the run record, the guards, the monitors' skipped nodes and the NaN replicates.
# =====================================================================================================
def _runs_file(tmp):
    """The record of a rehearsal run: beside its out_dir (here tmp itself), never in docs/sealed (Ruling 109, C1)."""
    return tmp / "RUNS.json"


def _read_runs(tmp):
    return json.loads(_runs_file(tmp).read_text())


def _sealed_runs_file(tmp):
    """The record of a sealed run: docs/sealed/RUNS.json (the `registered` fixture's temporary copy)."""
    return tmp / "docs_sealed" / "RUNS.json"


# --- (a) docs/sealed/RUNS.json: the record of a run, written at the lock, so that deleting out_dir cannot re-arm a test
def test_the_lock_writes_the_run_record_after_the_started_line_and_before_any_price_row(registered, monkeypatch):
    from es import sealed_run
    from es.registration import sha256_file
    from es.sealed_run import run_sealed
    write, tmp = registered
    proto, reg, sha, kw = _bound(tmp, write)
    seen = {}

    def about_to_read(*_a, **_k):
        seen["log"] = (tmp / "out" / "RUN_LOG.txt").read_text()
        seen["runs"] = _read_runs(tmp)                       # what is on disk when the first price row would be read
        raise RuntimeError("stop")

    monkeypatch.setattr(sealed_run, "_load_sealed_raw", about_to_read)
    with pytest.raises(RuntimeError, match="stop"):
        run_sealed(_cfg_a(tmp, kw), sha, n_rep=77)
    first, rec = seen["log"].splitlines()[0], seen["runs"]["A"]
    assert first.startswith("started ") and rec["started_utc"] == first.split(" ", 1)[1] and "run recorded in" in seen["log"]
    assert rec["protocol_sha256"] == sha and rec["out_dir"] == str((tmp / "out").resolve())
    assert rec["registration_identifier"] == "10.17605/OSF.IO/ABCDE" and rec["registered_utc"] == "2026-10-20T15:04:00Z"
    assert rec["registration_kind"] == "osf"                                       # Ruling 140: beside the identifier
    assert rec["code"] == {"code_manifest_sha256": sha256_file(kw["code_manifest"]),
                           "files": {f: sha256_file(tmp / "pkg" / f) for f in ("a.py", "b.py")}}
    assert (rec["n_rep"], rec["allow_departure"], rec["departures"]) == (77, False, 0)
    assert rec["environment"]["executable"] == sys.executable and rec["environment"]["versions"]["python"] == sys.version.split()[0]
    assert rec["environment"]["pins"] == "not bound"
    assert _read_runs(tmp) == seen["runs"] and not list(_runs_file(tmp).parent.glob("*.tmp"))     # the crash leaves it be
    claim = json.loads((tmp / "RUN_A.claim").read_text())                      # made before the record, and never removed
    assert claim["test"] == "A" and claim["started_utc"] == rec["started_utc"] and claim["out_dir"] == rec["out_dir"]
    assert not (tmp / "docs_sealed" / "RUNS.json").exists() and not list((tmp / "docs_sealed").iterdir())   # a rehearsal


def test_the_record_names_the_departures_and_the_code_it_ran_on(registered, monkeypatch):
    from es.registration import sha256_file
    from es.sealed_run import run_sealed
    write, tmp = registered
    proto, reg, sha, kw = _bound(tmp, write)
    (tmp / "pkg" / "b.py").write_text("B = 3\n")
    _stop_at_the_read(monkeypatch)
    with pytest.raises(RuntimeError, match="stop"):
        run_sealed(_cfg_a(tmp, kw), sha, allow_departure=True)
    rec = _read_runs(tmp)["A"]
    assert rec["allow_departure"] is True and rec["departures"] == 1
    assert rec["code"]["files"]["b.py"] == sha256_file(tmp / "pkg" / "b.py")        # the code as it was run, not as registered


def test_deleting_out_dir_does_not_re_arm_a_test(registered, monkeypatch):
    from es import sealed_run
    from es.sealed_run import run_sealed
    write, tmp = registered
    proto, reg, sha, kw = _bound(tmp, write)
    _stop_at_the_read(monkeypatch)
    cfg = _cfg_a(tmp, kw)
    with pytest.raises(RuntimeError, match="stop"):
        run_sealed(cfg, sha)
    record = _runs_file(tmp).read_text()
    shutil.rmtree(tmp / "out")                                     # how a test used to be re-armed
    reads = []
    monkeypatch.setattr(sealed_run, "_load_sealed_raw", lambda *a, **k: reads.append(a))
    broken = _cfg_a(tmp, kw, registration=(tmp / "none.md", tmp / "none2.md"))     # anything else refuses first
    for other in (cfg, _cfg_a(tmp, kw, out_dir=tmp / "elsewhere"), broken):
        with pytest.raises(FileExistsError, match="runs once"):
            run_sealed(other, sha)
    with pytest.raises(FileExistsError, match="runs once"):
        run_sealed(cfg, sha, allow_departure=True)
    assert reads == [] and not (tmp / "out").exists() and not (tmp / "elsewhere").exists()
    assert _runs_file(tmp).read_text() == record
    # the claim file alone does it too (RUNS.json deleted as well), and it is never removed (Ruling 109, I9)
    _runs_file(tmp).unlink()
    with pytest.raises(FileExistsError, match=r"RUN_A.claim exists.*runs once"):
        run_sealed(_cfg_a(tmp, kw, out_dir=tmp / "elsewhere"), sha)
    assert (tmp / "RUN_A.claim").exists() and not (tmp / "elsewhere").exists() and reads == []
    # and the record alone, with the claim file gone: refused before the lock as well, not after it
    _runs_file(tmp).write_text(record)
    (tmp / "RUN_A.claim").unlink()
    with pytest.raises(FileExistsError, match=r"records a run started.*runs once"):
        run_sealed(_cfg_a(tmp, kw, out_dir=tmp / "elsewhere"), sha)
    assert not (tmp / "elsewhere").exists() and not (tmp / "RUN_A.claim").exists() and reads == []


def test_each_test_keeps_its_own_record(registered, monkeypatch):
    from es.sealed_run import run_sealed
    write, tmp = registered
    proto, reg, sha, kw = _bound(tmp, write)
    _stop_at_the_read(monkeypatch)
    with pytest.raises(RuntimeError, match="stop"):
        run_sealed(_cfg_a(tmp, kw), sha)
    a_record = _read_runs(tmp)["A"]
    with pytest.raises(RuntimeError, match="stop"):                    # Test A crashed at its read: Test B goes on, alone
        run_sealed(_cfg_b(tmp, kw, tmp / "out"), sha)
    runs = _read_runs(tmp)
    assert set(runs) == {"A", "B"} and runs["A"] == a_record and runs["B"]["out_dir"] == str((tmp / "outB").resolve())
    assert (tmp / "RUN_A.claim").exists() and (tmp / "RUN_B.claim").exists()


@pytest.mark.parametrize("content,match", [("{not json", "not valid JSON"), ("[1, 2]", "JSON object")])
def test_a_corrupt_run_record_stops_every_run_before_anything_else(registered, monkeypatch, content, match):
    from es.sealed_run import run_sealed
    write, tmp = registered
    proto, reg, sha, kw = _bound(tmp, write)
    _no_read(monkeypatch)
    _runs_file(tmp).write_text(content)
    with pytest.raises(RuntimeError, match=match):
        run_sealed(_cfg_a(tmp, kw), sha)
    assert not (tmp / "out").exists()


def test_the_run_record_is_written_atomically_and_a_failed_write_stops_the_run_before_any_read(registered, monkeypatch):
    from es import sealed_run
    from es.sealed_run import run_sealed
    write, tmp = registered
    proto, reg, sha, kw = _bound(tmp, write)
    real, reads = os.replace, []

    def killed(src, dst):
        if Path(dst).name == "RUNS.json":
            raise OSError("disk full")
        return real(src, dst)

    monkeypatch.setattr(os, "replace", killed)
    monkeypatch.setattr(sealed_run, "_load_sealed_raw", lambda *a, **k: reads.append(a))
    with pytest.raises(OSError, match="disk full"):
        run_sealed(_cfg_a(tmp, kw), sha)
    assert reads == [] and not _runs_file(tmp).exists() and not list(_runs_file(tmp).parent.glob("*.tmp"))
    log = (tmp / "out" / "RUN_LOG.txt").read_text()
    assert log.startswith("started ") and "crashed " in log and "finished" not in log
    assert (tmp / "RUN_A.claim").exists()                               # the claim came first, and it stays: the run counts
    monkeypatch.setattr(os, "replace", real)
    with pytest.raises(FileExistsError, match="runs once"):          # the folder exists, so this run counts as made
        run_sealed(_cfg_a(tmp, kw), sha)
    with pytest.raises(FileExistsError, match="runs once"):          # and so does the claim file, whatever the folder
        run_sealed(_cfg_a(tmp, kw, out_dir=tmp / "elsewhere"), sha)


def _claim_args(tmp, proto, reg, sha, kw, name="out", **over):
    out = tmp / name
    out.mkdir(exist_ok=True)
    return (_cfg_a(tmp, kw, out_dir=out), _record_of(sha, proto, reg), "2026-10-23T00:00:00+00:00", out,
            {"code_manifest_sha256": "x", "files": {}}, 2000, False, 0)


def test_claim_run_takes_an_exclusive_claim_before_it_writes_the_record_and_never_removes_it(registered, monkeypatch):
    """Ruling 109, I9: RUN_<test>.claim is created with O_CREAT|O_EXCL before RUNS.json is written, an existing claim file
    refuses the run like a record, and neither is ever removed."""
    from es import sealed_run
    from es.sealed_run import _claim_run, recorded_runs
    write, tmp = registered
    proto, reg, sha, kw = _bound(tmp, write)
    assert recorded_runs(tmp) == {}
    seen = {}
    real = sealed_run._atomic_write_json

    def spy(path, obj):
        seen["claim_there_at_the_write"], seen["record_there_before"] = (tmp / "RUN_A.claim").exists(), _runs_file(tmp).exists()
        return real(path, obj)

    monkeypatch.setattr(sealed_run, "_atomic_write_json", spy)
    args = _claim_args(tmp, proto, reg, sha, kw)
    claim = _claim_run(*args)
    assert seen == {"claim_there_at_the_write": True, "record_there_before": False}
    held = json.loads((tmp / "RUN_A.claim").read_text())
    assert held["nonce"] == claim.nonce and held["test"] == "A" and held["started_utc"] == args[2]
    before_claim, before_record = (tmp / "RUN_A.claim").read_text(), _runs_file(tmp).read_text()
    with pytest.raises(FileExistsError, match=r"RUN_A.claim exists.*runs once"):     # another run holds the claim
        _claim_run(*_claim_args(tmp, proto, reg, sha, kw, name="again"))
    assert (tmp / "RUN_A.claim").read_text() == before_claim and _runs_file(tmp).read_text() == before_record
    (tmp / "RUN_A.claim").unlink()                                                    # a record whose claim file is gone
    with pytest.raises(FileExistsError, match="records a run started.*runs once"):
        _claim_run(*_claim_args(tmp, proto, reg, sha, kw, name="third"))
    assert _runs_file(tmp).read_text() == before_record


def test_two_runs_started_together_cannot_both_claim_the_test(registered, monkeypatch):
    """The reviewer's P7. Two claims with different out_dirs reach the exclusive create at the same moment (a barrier around
    the open of the claim file): one gets it, the other is refused, and RUNS.json holds the winner's record."""
    from es.sealed_run import _claim_run, recorded_runs
    write, tmp = registered
    proto, reg, sha, kw = _bound(tmp, write)
    barrier, real_open, outcomes = threading.Barrier(2), os.open, []

    def open_together(path, flags, *a, **k):
        if str(path).endswith(".claim"):
            barrier.wait(timeout=10)
        return real_open(path, flags, *a, **k)

    monkeypatch.setattr(os, "open", open_together)

    def claim(name):
        args = _claim_args(tmp, proto, reg, sha, kw, name=name)
        try:
            _claim_run(*args)
            outcomes.append((name, "claimed"))
        except FileExistsError:
            outcomes.append((name, "refused"))

    threads = [threading.Thread(target=claim, args=(n,)) for n in ("out_p", "out_q")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(o for _, o in outcomes) == ["claimed", "refused"]
    winner = next(n for n, o in outcomes if o == "claimed")
    assert recorded_runs(tmp)["A"]["out_dir"] == str((tmp / winner).resolve())
    assert json.loads((tmp / "RUN_A.claim").read_text())["out_dir"] == str((tmp / winner).resolve())


# --- (b) Test B's decided H-E carries the pooled test's approximate flag
def test_test_b_carries_the_pooled_approximate_flag_and_test_a_the_interim_one():
    from es.sealed_run import _decide
    names = ["H-E", "H-B1", "H-B2"]
    for pooled in (True, False):
        out, _ = _decide(HE, _hb(0.5, 0.4, 0.5, 0.4), _family(he_interim_approx=not pooled, he_approx=pooled), names,
                         _prior(0.5, 0.2, 0.5, 0.2))
        assert out["approximate"] is pooled and out["status"] == "decided (pooled A+B)"
        out, _ = _decide(HE, _hb(0.5, 0.4, 0.5, 0.4), _family(he_interim_approx=pooled, he_approx=not pooled), names, None)
        assert out["approximate"] is pooled and out["status"].startswith("interim")


def test_decide_without_h_e_keeps_everything_that_does_not_need_it():
    from es.sealed_run import _decide
    names = ["H-E", "H-B1", "H-B2"]
    hb = _hb(0.001, 0.4, 0.5, 0.4)
    out, holm = _decide(None, hb, _family(), names, None)                     # Test A: an H-B status never needed H-E
    assert out is None and holm is None and hb["H-B1"]["status"].startswith("rejected in Test A")
    hb = _hb(0.001, 0.4, 0.5, 0.4)
    out, holm = _decide(None, hb, _family(), names, _prior(0.02, 0.2, 0.2, 0.3))     # Test B: the step-down does
    assert out is None and holm == {"error": "not computed: H-E, a confirmatory hypothesis, did not complete"}
    assert hb["H-B1"]["replicated"] is True and hb["H-B1"]["status"].startswith("undetermined")
    hb = _hb(0.001, 0.4, 0.5, 0.4)
    out, holm = _decide(None, hb, _family(), ["H-B1", "H-B2"], _prior(0.01, 0.2, 0.2, 0.3))     # H-E demoted: all there is
    assert out is None and holm == {"H-E": None, "H-B1": True, "H-B2": False}
    assert hb["H-B1"]["status"] == "larger operating-point error in both 2026 quarters"


def test_decide_runs_the_step_down_when_an_h_b_bootstrap_did_not_complete():
    """Holm uses Test A's p for the H-B tests: Test B's own bootstrap of one of them failing costs it only its status."""
    from es.sealed_run import _decide
    hb = {"H-B1": {"t_hat": 0.4, "p": 0.001, "confirmatory": True}}                     # H-B2 is absent
    out, holm = _decide({**HE, "p": 0.001}, hb, _family(), ["H-E", "H-B1", "H-B2"], _prior(0.02, 0.2, 0.2, 0.3))
    assert holm == {"H-E": True, "H-B1": True, "H-B2": False} and list(hb) == ["H-B1"]
    assert out["wording"] == "timing skill within calendar cells detected"


# --- helpers for the stage tests: a _Run on a temporary folder, the 34 score sets, economic rows, fake monitors
def _stage_run(tmp, prior=None, test_id=None, **over):
    """A _Run as run_sealed builds it for the stages after the lock, on a temporary output folder (no side tables there).
    With `prior` (Test A's output folder) it is Test B's, pooled; `test_id="B"` without a prior is Test B alone."""
    from es.analyses import SEALED
    from es.sealed_run import SealedConfig, _Run
    out = tmp / "run_out"
    (out / "tables").mkdir(parents=True)
    test_id = test_id or ("A" if prior is None else "B")
    a, b = (SEALED["SA"], SEALED["SAF"]), (SEALED["SB"], SEALED["SBF"])
    cfg = SealedConfig(test_id, *(a if test_id == "A" else b), [], "2026-09-30" if test_id == "A" else "2026-12-31",
                       prior_test_dir=prior, out_dir=out, data_root=tmp)
    args = dict(cfg=cfg, n_rep=5, out=out, results={"test": test_id, "coverage": {}}, eta=0.01, family=FAMILY,
                confirmatory=["H-E", "H-B1", "H-B2"], prior=prior)
    args.update(over)
    return _Run(**args)


def _all_sets(frame=None):
    """The 34 score sets as the run holds them: the two economics_for reads stay full with object keys, the rest are slim."""
    from es.sealed_run import FULL, _slim, score_keys
    s = _tiny_scores() if frame is None else frame
    return {k: (s if k in FULL else _slim(s)) for k in score_keys()}


def _econ_rows():
    return [{"policy": "P1", "comparator": "none", "position": pos, "fees": fee, "SELC": 0.1, "rotation_p": p}
            for pos, fee, p in (("premium", "calibrated", 0.01), ("inc", "calibrated", 0.2), ("dec", "0.5", 0.6))]


def _fake_monitor_rows(skipped=()):
    """monitor_rows without the classifiers: eight nodes per block with fixed random monitors, the skipped nodes it is told of."""
    calls = []

    def fake(panel, analysis, scores_by_block, strict=True):
        (block,) = scores_by_block
        calls.append(strict)
        rng = np.random.default_rng(sum(map(ord, block)))
        n = 8
        frame = pd.DataFrame({"analysis": analysis.id, "block": block, "market": "M", "node": [f"n{i}" for i in range(n)],
                              "W1_norm": rng.random(n), "PSI": rng.random(n), "domain_auc": rng.random(n),
                              "early_rate": 0.01 + 0.2 * rng.random(n), "log_error": rng.random(n),
                              "fail": rng.random(n) > 0.5, "n_early": 336, "n_late": 500, "calib_days": 90})
        frame.attrs["skipped_nodes"] = [d for d in skipped if d["block"] == block]
        return frame

    return fake, calls


def test_a_guarded_analysis_is_logged_with_its_traceback_only_in_the_run_log(tmp_path):
    from es.sealed_run import FAILED
    run = _stage_run(tmp_path)

    def explode():
        raise ValueError("p1_fixed: no node qualifies, so there is nothing to return")

    assert run.guard("policy", explode) is FAILED
    message = "ValueError: p1_fixed: no node qualifies, so there is nothing to return"
    assert run.errors == {"policy": message} and run.error_entry("policy") == {"error": message}
    log = (tmp_path / "run_out" / "RUN_LOG.txt").read_text()
    assert f"error policy: {message}" in log and "Traceback (most recent call last)" in log and "in explode" in log
    assert "Traceback" not in json.dumps(run.error_entry("policy"))
    assert run.guard("fine", lambda: 5) == 5 and "fine" not in run.errors
    with pytest.raises(KeyboardInterrupt):                                        # an interrupt still ends the run
        run.guard("interrupted", lambda: (_ for _ in ()).throw(KeyboardInterrupt()))


def test_the_final_status_names_what_failed(tmp_path):
    run = _stage_run(tmp_path)
    assert run.status() == "completed"
    run.record("audit robustness", RuntimeError("x"), secondary=True)
    run.record("BH tables/file", OSError("y"), secondary=True)
    assert run.status() == "completed with secondary errors"
    run.record("H-E", ValueError("z"))
    assert run.status() == "completed with errors"


def test_a_bug_in_a_stage_is_recorded_and_the_later_stages_still_run(tmp_path):
    from es.sealed_run import _stage_call
    run = _stage_run(tmp_path)
    assert _stage_call(run, "audit", lambda: 1 / 0, "fallback") == "fallback"
    assert run.errors == {"stage/audit": "ZeroDivisionError: division by zero"} and _stage_call(run, "fine", lambda: 7, None) == 7


# --- (c) the secondary analyses, each in its own guard
def test_the_secondary_stage_runs_its_three_families_and_the_bh_tables(tmp_path, monkeypatch):
    from es import sealed_run
    from es.sealed_run import _stage_secondary
    fake, calls = _fake_monitor_rows()
    monkeypatch.setattr(sealed_run, "monitor_rows", fake)
    run = _stage_run(tmp_path)
    _stage_secondary(run, None, _all_sets(), _econ_rows())
    sec = run.results["secondary"]
    assert set(sec) == {"audit robustness", "economic robustness", "drift monitors"}
    assert (sec["audit robustness"]["tests"], sec["economic robustness"]["tests"], sec["drift monitors"]["tests"]) == (18, 3, 36)
    assert sec["drift monitors"]["skipped_nodes"] == [] and calls == [False] * 9       # strict=False, one call per block
    assert run.errors == {} and run.status() == "completed"
    tables = tmp_path / "run_out" / "tables"
    assert sorted(pd.read_csv(tables / "secondary_bh.csv")["family"].unique()) == sorted(sec)
    assert len(pd.read_csv(tables / "monitor_scores.csv")) == 9 * 8


@pytest.mark.parametrize("name,target", [("audit robustness", "audit_family"), ("economic robustness", "economic_family"),
                                         ("drift monitors", "monitor_rows"), ("drift monitors", "evaluate_monitors"),
                                         ("drift monitors", "monitor_family"), ("BH tables", "bh_table")])
def test_each_secondary_analysis_has_its_own_guard(tmp_path, monkeypatch, name, target):
    from es import sealed_run
    from es.sealed_run import _stage_secondary
    fake, _ = _fake_monitor_rows()
    monkeypatch.setattr(sealed_run, "monitor_rows", fake)

    def boom(*_a, **_k):
        raise RuntimeError(f"injected into {name}")

    monkeypatch.setattr(sealed_run, target, boom)
    run = _stage_run(tmp_path)
    _stage_secondary(run, None, _all_sets(), _econ_rows())                          # the stage does not raise
    sec, message = run.results["secondary"], f"RuntimeError: injected into {name}"
    assert sec[name] == {"error": message} and run.errors == {name: message}
    assert run.status() == "completed with secondary errors"
    tables = tmp_path / "run_out" / "tables"
    others = {"audit robustness", "economic robustness", "drift monitors"} - {name}
    if name == "BH tables":              # the families completed, and could not be adjusted: no counts, no table
        assert set(sec) == {"BH tables"} and not (tables / "secondary_bh.csv").exists()
    else:                                # the other families ran, and are adjusted without the failed one
        assert others <= set(sec) and all("error" not in sec[o] for o in others)
        assert set(pd.read_csv(tables / "secondary_bh.csv")["family"]) == others
    log = (tmp_path / "run_out" / "RUN_LOG.txt").read_text()
    assert f"error {name}: {message}" in log and "Traceback (most recent call last)" in log
    assert "Traceback" not in json.dumps(run.results)                              # the traceback is in RUN_LOG only


def test_when_every_family_fails_there_is_nothing_to_adjust(tmp_path, monkeypatch):
    from es import sealed_run
    from es.sealed_run import _stage_secondary

    def boom(*_a, **_k):
        raise RuntimeError("injected")

    adjusted = []
    for target in ("audit_family", "economic_family", "monitor_rows"):
        monkeypatch.setattr(sealed_run, target, boom)
    monkeypatch.setattr(sealed_run, "bh_table", lambda rows: adjusted.append(rows))
    run = _stage_run(tmp_path)
    _stage_secondary(run, None, _all_sets(), _econ_rows())
    assert set(run.results["secondary"]) == {"audit robustness", "economic robustness", "drift monitors"}
    assert all(v == {"error": "RuntimeError: injected"} for v in run.results["secondary"].values())
    assert adjusted == [] and not (tmp_path / "run_out" / "tables" / "secondary_bh.csv").exists()


def test_the_monitors_run_in_sealed_mode_and_their_skipped_nodes_are_recorded(tmp_path, monkeypatch):
    from es import sealed_run
    from es.sealed_run import _stage_secondary
    skipped = [{"analysis": "SA", "block": "LEV6", "market": "NYISO", "node": "N.Y.C.",
                "reason": "only 9 early deployment day(s) with complete LEV6 features, but the monitors read the first 14"},
               {"analysis": "SA", "block": "G1_24", "market": "CAISO", "node": "TH_SP15_GEN-APND",
                "reason": "no deployment day after the first 14 to score the outcome on"}]
    fake, calls = _fake_monitor_rows(skipped)
    monkeypatch.setattr(sealed_run, "monitor_rows", fake)
    run = _stage_run(tmp_path)
    _stage_secondary(run, None, _all_sets(), _econ_rows())
    assert run.results["secondary"]["drift monitors"]["skipped_nodes"] == skipped and set(calls) == {False}
    log = (tmp_path / "run_out" / "RUN_LOG.txt").read_text()
    assert "monitor_skipped LEV6 NYISO/N.Y.C.: only 9 early deployment day" in log
    assert "monitor_skipped G1_24 CAISO/TH_SP15_GEN-APND: no deployment day after the first 14" in log
    assert run.errors == {} and run.status() == "completed"                      # a skipped node is counted, not an error


def test_a_family_with_no_row_to_adjust_is_reported_with_zero_tests(tmp_path, monkeypatch):
    """Every monitored node skipped: the drift-monitor family completes with no row. It is in the report with zero tests
    (and its skipped nodes), not missing, and the other families are adjusted as usual."""
    from es import sealed_run
    from es.run_monitors import ROW_COLUMNS
    from es.sealed_run import _stage_secondary
    nodes = [{"analysis": "SA", "block": b, "market": "M", "node": f"n{i}", "reason": "only 3 early deployment day(s)"}
             for b in ("LEV6", "MID3", "DART3", "MIDDART6", "ROLL15", "ROLL_DISP9", "ROLL_LOC6", "DART18", "G1_24") for i in range(2)]

    def all_skipped(panel, analysis, scores_by_block, strict=True):
        (block,) = scores_by_block
        frame = pd.DataFrame(columns=ROW_COLUMNS)
        frame.attrs["skipped_nodes"] = [d for d in nodes if d["block"] == block]
        return frame

    monkeypatch.setattr(sealed_run, "monitor_rows", all_skipped)
    run = _stage_run(tmp_path)
    _stage_secondary(run, None, _all_sets(), _econ_rows())
    sec = run.results["secondary"]
    assert sec["drift monitors"] == {"tests": 0, "rows": 0, "bh_rejections": 0, "skipped_nodes": nodes}
    assert sec["audit robustness"]["tests"] == 18 and run.errors == {}


def test_a_score_set_that_fails_is_left_out_and_the_others_are_fitted_and_written(tmp_path, monkeypatch):
    from es import sealed_run
    from es.scores import CACHED_COLUMNS
    from es.sealed_run import FULL, _stage_scores, score_keys

    def fake(panel, analysis, block, estimator, seed):
        if (block, estimator, seed) == ("LEV6", "logit", 42):
            raise ValueError("score_analysis: no node is feasible")
        return _tiny_scores()

    monkeypatch.setattr(sealed_run, "score_analysis", fake)
    run = _stage_run(tmp_path)
    sc = _stage_scores(run, None)
    assert len(sc) == 33 and ("roll", "LEV6", "logit", 42) not in sc and set(sc) < set(score_keys())
    assert run.errors == {"scores/roll/LEV6/logit/s42": "ValueError: score_analysis: no node is feasible"}
    files = sorted((tmp_path / "run_out" / "scores").glob("*.parquet"))
    assert len(files) == 33 and "roll__LEV6__logit__s42.parquet" not in [f.name for f in files]
    slim = pd.read_parquet(tmp_path / "run_out" / "scores" / "roll__DART3__xgb__s42.parquet")
    assert list(slim.columns) == CACHED_COLUMNS and slim["market"].dtype == object        # object keys, like the cache files
    assert all(isinstance(sc[k]["market"].dtype, pd.CategoricalDtype) == (k not in FULL) for k in sc)    # only two stay full


def test_a_failed_tau_check_is_its_own_error(tmp_path):
    from es.sealed_run import _stage_tau
    run = _stage_run(tmp_path)
    _stage_tau(run, {})
    assert run.results["tau_check"]["error"].startswith("LookupError: score set ('") and "tau_check" in run.errors
    assert "tau_check {" not in (tmp_path / "run_out" / "RUN_LOG.txt").read_text()


def test_economics_that_completes_is_written_with_its_dropped_row_counts(tmp_path, monkeypatch):
    from es import sealed_run
    from es.run_economics import KEPT, KeptRows
    from es.sealed_run import _stage_economics
    kept = KeptRows(_kept_windows(9))
    kept.attrs["dropped_rows"] = {"P1": 0, "P4": 2, "calendar@count": 114}
    seen = {}

    def fake(panel, analysis, fmm, nyi, eta, score_fn, keep_rows):
        seen.update(analysis=analysis.id, eta=eta, keep_rows=keep_rows, fmm=fmm, nyi=nyi)
        return [{"policy": "P1", "comparator": "none", "SELC": 0.1}], kept

    monkeypatch.setattr(sealed_run, "economics_for", fake)
    run = _stage_run(tmp_path)
    econ, got = _stage_economics(run, None, _all_sets())
    assert got is kept and econ == [{"policy": "P1", "comparator": "none", "SELC": 0.1}]
    assert seen == {"analysis": "SA", "eta": 0.01, "keep_rows": True, "fmm": None, "nyi": None}      # no side tables there
    assert run.results["coverage"]["comparator_rows_dropped"] == {"P1": 0, "P4": 2, "calendar@count": 114}
    assert 'comparator_rows_dropped {"P1": 0, "P4": 2, "calendar@count": 114}' in (tmp_path / "run_out" / "RUN_LOG.txt").read_text()
    assert len(pd.read_csv(tmp_path / "run_out" / "tables" / "economics.csv")) == 1
    assert sorted(p.name for p in (tmp_path / "run_out" / "kept").glob("*.parquet")) == sorted(
        f"{n.replace('@', '_at_').replace('*', 'star')}.parquet" for n in KEPT) and run.errors == {}


# --- (f) a failure in one analysis is that analysis' own error; only the data load aborts
def test_a_policy_that_finds_no_node_is_one_recorded_error_and_the_others_still_run(tmp_path, monkeypatch):
    from es import sealed_run
    from es.sealed_run import _stage_audit

    def none_qualifies(scores, alpha):
        raise ValueError("p6_exact: no node qualifies, so there is nothing to return (the score frame has no nodes)")

    monkeypatch.setattr(sealed_run, "p6_exact", none_qualifies)
    run = _stage_run(tmp_path)
    _stage_audit(run, _all_sets())
    assert run.errors == {"audit_activity/P6": "ValueError: p6_exact: no node qualifies, so there is nothing to return "
                                                "(the score frame has no nodes)"}
    table = pd.read_csv(tmp_path / "run_out" / "tables" / "audit_activity.csv")
    assert set(table["policy"]) == {"P1", "P4_56", "P4_28", "P5", "P2_season", "P2_all"} and len(table) == (34 + 5) * 2
    assert run.status() == "completed with errors"


def test_an_audit_with_every_part_failed_is_one_error_and_writes_no_table(tmp_path, monkeypatch):
    from es import sealed_run
    from es.sealed_run import _stage_audit
    for name in ("p1_fixed", "p4_rolling", "p5_self_correcting", "p6_exact", "p2_insample"):
        monkeypatch.setattr(sealed_run, name, lambda *a, **k: (_ for _ in ()).throw(ValueError("no node qualifies")))
    run = _stage_run(tmp_path)
    _stage_audit(run, _all_sets())
    assert "audit_activity" in run.errors and "No objects to concatenate" in run.errors["audit_activity"]
    assert not (tmp_path / "run_out" / "tables" / "audit_activity.csv").exists()


def test_a_missing_score_set_fails_only_the_analyses_that_need_it(tmp_path):
    from es.sealed_run import _stage_bootstraps
    sc = _all_sets()
    del sc[("roll", "LEV6", "xgb", 42)]
    run = _stage_run(tmp_path)
    hb = _stage_bootstraps(run, sc)
    message = "LookupError: score set ('roll', 'LEV6', 'xgb', 42) is not available (its fit did not complete)"
    assert set(hb) == {"H-B2"} and run.results["H-B1"] == {"error": message}
    assert set(run.errors) == {"H-B1", "E5/LEV6"}                     # PC, H-S and the nine other E5 blocks ran
    assert all("error" not in v for v in run.results["PC"].values()) and "error" not in run.results["H-S"]["logit"]
    e5 = {row["block"]: row for row in run.results["E5"]}
    assert e5["LEV6"] == {"block": "LEV6", "error": message} and "error" not in e5["G1_24"]
    assert len(pd.read_csv(tmp_path / "run_out" / "tables" / "E5_blocks.csv")) == 10


def _write_prior_kept(prior, frames, skip=()):
    """Test A's stored rows (kept/*.parquet) in `prior`, and its manifest: Test B checks every file it reads against that
    manifest after its lock (Ruling 116, Mn3)."""
    from es.run_economics import KEPT
    from es.sealed_pull import write_manifest
    from es.sealed_run import _safe
    (prior / "kept").mkdir(parents=True, exist_ok=True)
    for name in KEPT:
        if name not in skip:
            frames[name].to_parquet(prior / "kept" / f"{_safe(name)}.parquet", index=False)
    write_manifest(prior)


def _kept_windows(seed, start=dt.date(2025, 7, 1), n_days=20, losses=True):
    rng = np.random.default_rng(seed)
    rows = []
    for node in ("A", "B"):
        for d in range(n_days):
            day = start + dt.timedelta(days=d)
            for h in range(24):
                pnl = rng.normal(0.5, 3.0) if losses else abs(rng.normal(0.5, 3.0)) + 0.1
                rows.append({"market": "M" if node == "A" else "CAISO", "node": node, "settle_date": day, "settle_hour": h,
                             "timestamp_utc": pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=h),
                             "wclass": int(day.weekday() >= 5), "month": day.month, "cal_decile": h % 10, "pnl_open": pnl})
    base = pd.DataFrame(rows)
    base["dart_settle"] = base["pnl_open"]
    from es.run_economics import KEPT
    return {name: base.assign(alert=rng.random(len(base)) < 0.08) for name in KEPT}


def test_a_nan_in_the_economic_rows_fails_h_e_alone(tmp_path):
    from es.sealed_run import FAILED, _stage_decision, _stage_he
    kept = _kept_windows(1)
    kept["P1"] = kept["P1"].assign(pnl_open=lambda f: f["pnl_open"].where(f.index != 7))             # one NaN price result
    run = _stage_run(tmp_path)
    he = _stage_he(run, kept, [kept])
    assert he is FAILED and run.errors["H-E"].startswith("ValueError:") and "NaN" in run.errors["H-E"]
    hb = {"H-B1": {"t_hat": 0.4, "p": 0.001, "confirmatory": True}}
    _stage_decision(run, he, hb)                                       # the H-B statuses are still given
    assert run.results["H-E"] == run.error_entry("H-E") and hb["H-B1"]["status"].startswith("rejected in Test A")
    assert run.results["H-B1"]["status"].startswith("rejected in Test A") and "holm" not in run.results


def test_a_failed_economics_is_recorded_and_what_needs_it_says_so(tmp_path, monkeypatch):
    from es import sealed_run
    from es.sealed_run import FAILED, _stage_economics, _stage_estimands, _stage_he, _stage_secondary
    monkeypatch.setattr(sealed_run, "economics_for", lambda *a, **k: (_ for _ in ()).throw(
        ValueError("p1_fixed: no node qualifies, so there is nothing to return")))
    fake, _ = _fake_monitor_rows()
    monkeypatch.setattr(sealed_run, "monitor_rows", fake)
    run = _stage_run(tmp_path)
    sc = _all_sets()
    econ, kept = _stage_economics(run, None, sc)
    assert (econ, kept) == (None, None)
    assert run.results["economics"] == {"error": "ValueError: p1_fixed: no node qualifies, so there is nothing to return"}
    windows = _stage_estimands(run, kept)
    assert windows is None and run.results["estimands"]["error"].startswith("LookupError: needs the economics rows")
    assert run.results["estimands_scope"] == "single window (interim)"
    assert _stage_he(run, kept, windows) is FAILED and "needs the economic rows" in run.errors["H-E"]
    _stage_secondary(run, None, sc, econ)
    sec = run.results["secondary"]
    assert sec["economic robustness"]["error"].startswith("LookupError: needs the economic rows")
    assert "error" not in sec["audit robustness"] and "error" not in sec["drift monitors"]       # the others still ran
    assert {"economics", "estimands", "estimands_ex_caiso", "H-E", "economic robustness"} <= set(run.errors)
    assert run.status() == "completed with errors"


def test_the_quarter_alone_survives_when_test_a_s_stored_rows_cannot_be_loaded(tmp_path):
    from es.run_economics import KEPT
    from es.sealed_run import _safe, _stage_estimands
    prior = tmp_path / "A_out"
    (prior / "kept").mkdir(parents=True)
    for name in KEPT:                                                     # Test A's stored rows: not readable
        (prior / "kept" / f"{_safe(name)}.parquet").write_bytes(b"not parquet")
    run = _stage_run(tmp_path, prior=prior)
    windows = _stage_estimands(run, _kept_windows(2))
    tables = tmp_path / "run_out" / "tables"
    assert windows is None and (tables / "estimands_this_quarter.csv").exists()          # this quarter alone was computed
    assert not (tables / "estimands.csv").exists() and not (tables / "estimands_ex_caiso.csv").exists()
    assert {"prior_kept", "estimands", "estimands_ex_caiso"} <= set(run.errors) and "estimands_this_quarter" not in run.errors
    assert run.results["estimands"]["error"].startswith("LookupError: needs Test A's stored rows")
    assert run.results["estimands_scope"] == "pooled A+B"


def test_test_b_pools_the_estimands_with_test_a_s_stored_rows(tmp_path):
    from es.sealed_run import _stage_estimands
    prior = tmp_path / "A_out"
    a, b = _kept_windows(7), _kept_windows(8, start=dt.date(2025, 10, 1))
    _write_prior_kept(prior, a)
    run = _stage_run(tmp_path, prior=prior)
    windows = _stage_estimands(run, b)
    tables = tmp_path / "run_out" / "tables"
    assert run.errors == {} and len(windows) == 2 and windows[1] is b
    pooled, quarter = (pd.read_csv(tables / f).set_index("stat") for f in ("estimands.csv", "estimands_this_quarter.csv"))
    assert (pooled["n_windows"] == 2).all() and (quarter["n_windows"] == 1).all()
    assert (tables / "estimands_ex_caiso.csv").exists() and run.results["estimands_scope"] == "pooled A+B"


def test_the_estimands_of_a_single_window_are_written_with_their_nan_columns(tmp_path):
    from es.sealed_run import _stage_estimands
    run = _stage_run(tmp_path)
    kept = _kept_windows(3)
    windows = _stage_estimands(run, kept)
    tables = tmp_path / "run_out" / "tables"
    assert windows == [kept] and run.errors == {} and not (tables / "estimands_this_quarter.csv").exists()
    est = pd.read_csv(tables / "estimands.csv").set_index("stat")
    assert (est["n_nan_replicates"] == 0).all() and (est["n_windows"] == 1).all()
    assert set(run.results["estimands"]["SELC"]) == {"estimate", "lo", "hi", "approximate", "n_nan_replicates"}
    ex = pd.read_csv(tables / "estimands_ex_caiso.csv").set_index("stat")
    assert list(ex.index) == list(est.index) and not np.allclose(est.loc["NET", "estimate"], ex.loc["NET", "estimate"])


def test_a_data_load_failure_aborts_the_run_and_the_run_still_counts(registered, monkeypatch):
    """Only the data load may abort: here the panel build, which is part of it. The record and the crash line are there."""
    from es import sealed_run
    from es.sealed_run import run_sealed
    write, tmp = registered
    proto, reg, sha, kw = _bound(tmp, write)
    raw = pd.DataFrame({"x": [1]})
    raw.attrs.update({"rows_read": 1, "rows_kept": 1, "nonfinite_dropped": {}, "off_hour_dropped": {}, "sealed_manifests": {},
                      "node_map_dropped": 0, "duplicates_dropped": 0, "dev_duplicates_dropped": 0, "beyond_window_end": 0})
    monkeypatch.setattr(sealed_run, "_load_sealed_raw", lambda *a, **k: raw)
    monkeypatch.setattr(sealed_run, "build_panel", lambda r: (_ for _ in ()).throw(RuntimeError("the panel would not build")))
    with pytest.raises(RuntimeError, match="would not build"):
        run_sealed(_cfg_a(tmp, kw), sha)
    assert "A" in _read_runs(tmp) and "crashed " in (tmp / "out" / "RUN_LOG.txt").read_text()
    assert not (tmp / "out" / "results.json").exists()


# --- (e) NaN bootstrap replicates
def test_nan_replicates_are_counted_and_more_than_one_percent_marks_the_result_approximate():
    from es.sealed_run import _split_nan
    finite, n_nan, approximate = _split_nan([1.0, np.nan, 2.0] + [0.5] * 197)       # 1 of 200
    assert (len(finite), n_nan, approximate) == (199, 1, False)
    assert _split_nan([np.nan, np.nan] + [0.5] * 198)[1:] == (2, False)             # 2 of 200 is exactly 1%: not more than 1%
    assert _split_nan([np.nan] * 3 + [0.5] * 197)[1:] == (3, True)
    assert _split_nan([0.5, 0.6])[1:] == (0, False) and _split_nan([np.nan, np.nan])[0].size == 0


def _contrast_pair(cal_alt=40, cal_base=40, dep=15):
    """Two score frames for the contrast: calibration on different days, deployment on the same ones, different scores."""
    alt = _scores_in(("M", "N"), (("calib", dt.date(2024, 7, 1), cal_alt), ("deploy", dt.date(2026, 7, 1), dep)), seed=5)
    base = _scores_in(("M", "N"), (("calib", dt.date(2025, 7, 1), cal_base), ("deploy", dt.date(2026, 7, 1), dep)), seed=6)
    return alt, base


def test_a_contrast_without_nan_replicates_is_block_contrast_bootstrap_number_for_number():
    from es.bootstrap import block_contrast_bootstrap
    from es.sealed_run import _contrast
    alt, base = _contrast_pair()
    ref = block_contrast_bootstrap(alt, base, alpha=0.05, n_rep=60)
    got = _contrast(alt, base, 0.05, 60)
    assert {k: got[k] for k in ("t_hat", "p", "ci", "ci_basic", "n_nodes")} == {k: ref[k] for k in (
        "t_hat", "p", "ci", "ci_basic", "n_nodes")}
    assert got["n_nan_replicates"] == 0 and got["nan_approximate"] is False


def test_a_contrast_with_nan_replicates_is_computed_on_the_finite_ones():
    """Two calibration days per frame, on different days: a replicate often draws none of one frame's days, and its
    contrast is NaN. block_contrast_bootstrap refuses; _contrast infers from the replicates that exist."""
    from es.bootstrap import (_cal_dep_bootstrap, basic_ci, basic_p_one_sided, block_contrast_bootstrap, percentile_ci)
    from es.sealed_run import _contrast
    alt, base = _contrast_pair(cal_alt=2, cal_base=2, dep=10)
    with pytest.raises(ValueError, match="basic_p_one_sided"):
        block_contrast_bootstrap(alt, base, alpha=0.05, n_rep=60)
    got = _contrast(alt, base, 0.05, 60)
    t_hat, t_star, n_nodes = _cal_dep_bootstrap([alt, base], lambda e: e[0] - e[1], 0.05, 60, 7, 20261022)
    finite = t_star[~np.isnan(t_star)]
    assert 0 < len(finite) < 60
    assert got["n_nan_replicates"] == 60 - len(finite) and got["nan_approximate"] is True
    assert got["t_hat"] == t_hat and got["n_nodes"] == n_nodes
    assert got["p"] == basic_p_one_sided(t_hat, finite) and got["ci"] == percentile_ci(finite)
    assert got["ci_basic"] == basic_ci(t_hat, finite)
    assert np.isfinite([got["p"], *got["ci"], *got["ci_basic"]]).all()


def test_a_contrast_with_nothing_to_infer_from_is_an_error(monkeypatch):
    from es import sealed_run
    from es.sealed_run import _contrast
    monkeypatch.setattr(sealed_run, "block_contrast_bootstrap", lambda *a, **k: (_ for _ in ()).throw(ValueError("nan")))
    for t_hat, t_star in ((0.3, np.full(10, np.nan)), (np.nan, np.linspace(0, 1, 10))):
        monkeypatch.setattr(sealed_run, "_cal_dep_bootstrap", lambda *a, **k: (t_hat, t_star, 4))
        with pytest.raises(ValueError, match="nothing to infer from"):
            _contrast(None, None, 0.05, 10)


def test_a_contrast_the_bootstrap_refuses_for_another_reason_is_refused_again(monkeypatch):
    from es.sealed_run import _contrast
    alt, base = _contrast_pair()
    with pytest.raises(ValueError, match="missing column"):
        _contrast(alt.drop(columns=["score"]), base, 0.05, 10)


def _stub_level_bootstrap(monkeypatch, aligned_star, independent_star, t_hat=0.4, n_nodes=3, failures=None):
    """block_level_bootstrap that answers each draw (aligned=True, and the registered independent one) with its own replicates
    and intervals that are NaN whenever a replicate is, as the real function's are. `failures` is the (aligned, independent)
    pair of replicates for measure="failures", when it should differ from the median's."""
    from es import sealed_run
    seen = []

    def stub(scores, alpha=0.05, n_rep=2000, mean_block=7, seed=20261022, measure="median", aligned=False):
        seen.append(aligned)
        pair = failures if (failures is not None and measure == "failures") else (aligned_star, independent_star)
        star = pair[0] if aligned else pair[1]
        return {"t_hat": t_hat, "t_star": star, "ci": (np.nan, np.nan), "ci_basic": (np.nan, np.nan), "n_nodes": n_nodes}

    monkeypatch.setattr(sealed_run, "block_level_bootstrap", stub)
    return seen


def test_the_level_bootstrap_recomputes_the_intervals_of_each_draw_on_its_finite_replicates(monkeypatch):
    """Ruling 104(e) for both draws of E5 and H-S (Ruling 118, M2): the aligned draw is the primary one, the independent draw
    gives ci_independent, and each has its own NaN count and its own 1% rule."""
    from es.bootstrap import basic_ci, percentile_ci
    from es.sealed_run import _level
    aligned_star = np.r_[np.linspace(0.0, 1.0, 190), np.full(10, np.nan)]          # 10 of 200: 5%
    independent_star = np.r_[np.linspace(0.2, 0.9, 199), np.nan]                    # 1 of 200: 0.5%
    seen = _stub_level_bootstrap(monkeypatch, aligned_star, independent_star)
    got = _level(None, 0.05, 200)
    assert sorted(seen) == [False, True]
    assert got["n_nan_replicates"] == 10 and got["nan_approximate"] is True and got["n_nodes"] == 3
    assert got["ci"] == percentile_ci(aligned_star[:190]) and got["ci_basic"] == basic_ci(0.4, aligned_star[:190])
    assert got["ci_independent"] == percentile_ci(independent_star[:199])
    assert got["n_nan_replicates_independent"] == 1 and got["nan_approximate_independent"] is False
    assert got["scheme"] == "aligned" and got["approximate"] is True
    _stub_level_bootstrap(monkeypatch, aligned_star, independent_star, t_hat=np.nan, n_nodes=0)
    none = _level(None, 0.05, 200)                                                  # no point estimate: nothing to recompute
    assert np.isnan(none["ci"]).all() and np.isnan(none["ci_independent"]).all() and none["approximate"] is True


def test_a_level_bootstrap_is_the_calendar_aligned_draw_with_the_independent_interval_beside_it():
    """Ruling 118, M2. E5 and H-S take the aligned draw as the primary one (t_hat, ci, ci_basic), keep the registered
    independent draws' percentile interval as ci_independent, a diagnostic, and say which: scheme 'aligned'. The result is
    approximate whatever the NaN count (the aligned draw is not the registered scheme)."""
    from es.bootstrap import block_level_bootstrap
    from es.sealed_run import _level
    s = _tiny_scores()
    for measure in ("median", "failures"):
        aligned = block_level_bootstrap(s, alpha=0.05, n_rep=40, measure=measure, aligned=True)
        independent = block_level_bootstrap(s, alpha=0.05, n_rep=40, measure=measure)            # the registered scheme, as it was
        got = _level(s, 0.05, 40, measure)
        assert (got["t_hat"], got["ci"], got["ci_basic"], got["n_nodes"]) == (aligned["t_hat"], aligned["ci"], aligned["ci_basic"],
                                                                              aligned["n_nodes"])
        assert got["ci_independent"] == independent["ci"] and got["scheme"] == "aligned" and got["approximate"] is True
        assert got["n_nan_replicates"] == 0 and got["nan_approximate"] is False
        assert got["n_nan_replicates_independent"] == 0 and got["nan_approximate_independent"] is False
    median = _level(s, 0.05, 40)
    assert median["ci"] != median["ci_independent"]                  # the two draws are not the same interval on this fixture


def test_the_hypothesis_results_record_their_nan_replicates_and_stay_approximate_when_many(tmp_path):
    """Two-day calibration windows give many NaN replicates; H-B1 and H-B2 are still computed, on the finite ones."""
    from es.sealed_run import _stage_bootstraps
    alt, base = _contrast_pair(cal_alt=2, cal_base=2, dep=10)
    sc = _all_sets(base)
    sc[("roll", "LEV6", "xgb", 42)] = alt                               # H-B1's block: calibration on other days than DART3's
    sc[("roll", "ROLL_DISP9", "xgb", 42)] = alt
    run = _stage_run(tmp_path, n_rep=60)
    hb = _stage_bootstraps(run, sc)
    assert set(hb) == {"H-B1", "H-B2"} and run.errors.get("H-B1") is None
    for r in hb.values():
        assert r["n_nan_replicates"] > 0 and r["ci_approximate"] is True and np.isfinite([r["p"], *r["ci"], *r["ci_basic"]]).all()
    log = (tmp_path / "run_out" / "RUN_LOG.txt").read_text()
    assert "nan_replicates H-B1:" in log and "(more than 1%: approximate)" in log


def test_estimands_without_a_non_finite_endpoint_have_no_nan_replicates():
    """E1-E4: estimand_cis cannot produce a NaN replicate. Its inputs are checked finite and its sums are Python floats, so
    a degenerate replicate raises ZeroDivisionError (here: no loss anywhere) and is the estimands' recorded error."""
    from es.run_economics import estimand_cis
    from es.sealed_run import _estimand_rows
    windows = [_kept_windows(4), _kept_windows(5, start=dt.date(2025, 10, 1))]
    rows, ref = _estimand_rows(windows, 12), estimand_cis(windows, n_rep=12)
    assert [r["n_nan_replicates"] for r in rows] == [0] * len(ref)
    assert [{k: r[k] for k in ref[0]} for r in rows] == ref                  # estimand_cis' own numbers and flags
    with pytest.raises(ZeroDivisionError):
        estimand_cis([_kept_windows(6, losses=False)], n_rep=5)


def test_an_estimand_with_a_non_finite_endpoint_is_marked_approximate(monkeypatch):
    from es import sealed_run
    from es.sealed_run import _estimand_rows
    rows = [{"stat": "SELC", "estimate": 0.1, "lo": np.nan, "hi": 0.3, "approximate": False, "n_windows": 1},
            {"stat": "NET", "estimate": 5.0, "lo": 1.0, "hi": 9.0, "approximate": False, "n_windows": 1},
            {"stat": "E3:ESred", "estimate": 2.0, "lo": 1.0, "hi": 3.0, "approximate": True, "n_windows": 1}]
    monkeypatch.setattr(sealed_run, "estimand_cis", lambda windows, n_rep: [dict(r) for r in rows])
    got = {r["stat"]: r for r in _estimand_rows([], 10)}
    assert got["SELC"]["approximate"] is True and got["SELC"]["n_nan_replicates"] is None
    assert got["NET"]["approximate"] is False and got["NET"]["n_nan_replicates"] == 0
    assert got["E3:ESred"]["approximate"] is True and got["E3:ESred"]["n_nan_replicates"] == 0


# =====================================================================================================
# Controller Ruling 109 (fix round 1 of Task 2A-3): what a config can be (C1, C2, I2, M8), the claim (I1, I9), the chain from
# Test A to Test B (I3, I4), the interval and decision fields of H-B (I5), H-E's rows (I6), the environment (I7), the disk and
# the settlement tables (I8, M9), and the minors. The reviewer's probes P1-P8 and S1-S7 are each a test here.
# =====================================================================================================
def _sealed_runs(tmp):
    path = _sealed_runs_file(tmp)
    return json.loads(path.read_text()) if path.exists() else {}


def _nothing_recorded(tmp):
    """No record, no claim file and no output folder of a refused run, in docs/sealed or beside the output."""
    assert not [p for p in (tmp / "docs_sealed").iterdir() if p.name != "ENV_PINS.json"] and not (tmp / "RUNS.json").exists()
    assert not list(tmp.glob("RUN_*.claim")) and not (tmp / "out").exists() and not (tmp / "outB").exists()
    assert not (tmp / "outputs_es").exists()                         # nor the registered folders of a sealed run (Ruling 122, N2b)


@pytest.mark.parametrize("test_id", ["X", "a", "A2", "AB", "B ", "", None, 1, ["A"]])
def test_a_test_id_other_than_a_or_b_is_refused_before_anything_else_runs(registered, monkeypatch, test_id):
    """C1, the reviewer's P1: an id that names no window is refused first of all, before the record is read, and nothing
    is created. (An id outside the two used to pass every check and record itself under its own key.)"""
    from es import sealed_run
    from es.sealed_run import default_config, run_sealed
    write, tmp = registered
    proto, reg, sha, kw = _bound(tmp, write)
    _no_read(monkeypatch)
    monkeypatch.setattr(sealed_run, "recorded_runs", lambda *a, **k: (_ for _ in ()).throw(AssertionError("RUNS.json was read")))
    cfg = _cfg_a(tmp, kw)
    cfg.test_id = test_id
    with pytest.raises(ValueError, match="not one of the two sealed tests"):
        run_sealed(cfg, sha)
    _nothing_recorded(tmp)
    with pytest.raises(ValueError, match="not one of the two sealed tests"):
        default_config(test_id)


def test_the_two_test_ids_are_the_two_registered_windows():
    from es.sealed_pull import SEALED_WINDOWS
    from es.sealed_run import TEST_IDS
    assert TEST_IDS == ("A", "B") and set(SEALED_WINDOWS) == set(TEST_IDS)


def test_after_test_a_is_recorded_no_other_id_and_no_unregistered_config_reads_the_pull_again(registered, monkeypatch):
    """C1 with the sealed arms, the reviewer's P1, P1b and P8. Test A reads its pull once. The same arms, pull and window under
    the id X, a, A2 or B are refused, and so is Test B's id with Test A's arms and window: each before any read."""
    from es import sealed_run
    from es.sealed_run import run_sealed
    write, tmp = registered
    proto, reg, sha, kw = _bound(tmp, write, real_code=True)
    _no_pull_check(monkeypatch)
    reads = []
    monkeypatch.setattr(sealed_run, "_load_sealed_raw",
                        lambda *a, **k: reads.append(a) or (_ for _ in ()).throw(RuntimeError("stop")))
    with pytest.raises(RuntimeError, match="stop"):
        run_sealed(_sealed_cfg(tmp, kw), sha)                           # the one Test A run, recorded
    assert len(reads) == 1 and set(_sealed_runs(tmp)) == {"A"}
    for test_id in ("X", "a", "A2", "B"):
        cfg = _sealed_cfg(tmp, kw, out_dir=tmp / f"out_{test_id}")
        cfg.test_id = test_id
        if test_id == "B":
            cfg.prior_test_dir = _out(tmp, "A")                         # Test A's folder that exists, so that B is complete
        with pytest.raises(ValueError, match="sealed tests|registered one"):
            run_sealed(cfg, sha)
        assert not (tmp / f"out_{test_id}").exists()
    both = _sealed_cfg(tmp, kw, out_dir=tmp / "out_both", sealed_dirs=["sealed_2026Q3", "sealed_2026Q4"], window_end="2026-12-31")
    both.test_id = "X"
    with pytest.raises(ValueError, match="sealed tests"):               # P1b: _check_config used to accept this
        run_sealed(both, sha)
    assert len(reads) == 1 and set(_sealed_runs(tmp)) == {"A"} and not list(tmp.glob("RUN_B.claim"))
    assert not (tmp / "docs_sealed" / "RUN_B.claim").exists()


@pytest.mark.parametrize("what", ["window_end_early", "window_end_late", "test_b_arms", "frozen_is_rolling", "no_sealed_dirs",
                                  "extra_pull", "mixed_arms", "mixed_arms_the_other_way", "arm_with_other_windows", "n_rep_50",
                                  "n_rep_2001"])
def test_a_sealed_config_must_be_the_registered_one_and_is_refused_before_its_lock(registered, monkeypatch, what):
    """C1 and C2, the reviewer's P6. A config with a sealed arm that is not default_config('A') (a window ending early, Test B's
    arms, no pull, 50 replicates) used to pass every pre-lock check and spend Test A. Now each is refused, with nothing
    created, and the corrected config still runs."""
    import dataclasses
    from es.analyses import ANALOGS, SEALED
    from es.sealed_run import run_sealed
    from es.windows import DateWindow as D
    write, tmp = registered
    proto, reg, sha, kw = _bound(tmp, write, real_code=True)
    _no_pull_check(monkeypatch)
    _no_read(monkeypatch)
    over, n_rep = {}, None
    if what == "window_end_early":
        over["window_end"] = "2026-08-31"
    elif what == "window_end_late":
        over["window_end"] = "2026-10-01"
    elif what == "test_b_arms":
        over.update(rolling=SEALED["SB"], frozen=SEALED["SBF"])
    elif what == "frozen_is_rolling":
        over["frozen"] = SEALED["SA"]
    elif what == "no_sealed_dirs":
        over["sealed_dirs"] = []
    elif what == "extra_pull":
        over["sealed_dirs"] = ["sealed_2026Q3", "sealed_2026Q4"]
    elif what == "mixed_arms":                                           # one sealed arm makes it a sealed run; no pull
        over.update(frozen=ANALOGS["AQ3F"], sealed_dirs=[])              # would otherwise let it pass as a rehearsal
    elif what == "mixed_arms_the_other_way":
        over.update(rolling=ANALOGS["AQ3"], sealed_dirs=[])
    elif what == "arm_with_other_windows":                              # the registered id, other windows
        over["rolling"] = dataclasses.replace(SEALED["SA"], deploy=D("2026-07-01", "2026-08-31"))
    elif what == "n_rep_50":
        n_rep = 50
    elif what == "n_rep_2001":
        n_rep = 2001
    with pytest.raises(ValueError):
        run_sealed(_sealed_cfg(tmp, kw, **over), sha, **({} if n_rep is None else {"n_rep": n_rep}))
    _nothing_recorded(tmp)
    _stop_at_the_read(monkeypatch)                                       # the registered config is not refused
    with pytest.raises(RuntimeError, match="stop"):
        run_sealed(_sealed_cfg(tmp, kw), sha)
    assert set(_sealed_runs(tmp)) == {"A"}


def test_n_rep_of_a_sealed_run_is_the_module_constant_that_tests_lower(registered, monkeypatch):
    """C2: the registered number of replicates is a module constant, N_REP, and the default of run_sealed's n_rep (es.protocol
    reads it from the signature); a sealed run uses exactly that. A test lowers the constant as it lowers MIN_USABLE_NODES, and
    then passes the lowered number. A rehearsal may use any."""
    import inspect
    from es import sealed_run
    from es.sealed_run import run_sealed
    assert sealed_run.N_REP == 2000 and inspect.signature(run_sealed).parameters["n_rep"].default == 2000
    write, tmp = registered
    proto, reg, sha, kw = _bound(tmp, write, real_code=True)
    _no_pull_check(monkeypatch)
    monkeypatch.setattr(sealed_run, "N_REP", 20)
    seen = []
    monkeypatch.setattr(sealed_run, "_load_sealed_raw", lambda *a, **k: seen.append(1) or (_ for _ in ()).throw(RuntimeError("stop")))
    for refused in ({}, {"n_rep": 2000}, {"n_rep": 21}):                # the default is the constant as it was when defined
        with pytest.raises(ValueError, match=r"registered n_rep, 20, not \d+"):
            run_sealed(_sealed_cfg(tmp, kw), sha, **refused)
    _nothing_recorded(tmp)
    with pytest.raises(RuntimeError, match="stop"):
        run_sealed(_sealed_cfg(tmp, kw), sha, n_rep=20)
    assert _sealed_runs(tmp)["A"]["n_rep"] == 20 and seen == [1]


def test_a_rehearsal_cannot_keep_its_records_in_docs_sealed(registered, monkeypatch):
    """C1, I2: the records of a rehearsal sit in the folder that holds its out_dir. An out_dir directly in docs/sealed would
    make them the real RUNS.json and claim files, and block the real Test A."""
    from es.sealed_run import run_sealed
    write, tmp = registered
    proto, reg, sha, kw = _bound(tmp, write, real_code=True)
    _no_read(monkeypatch)
    with pytest.raises(ValueError, match="in docs/sealed or around it"):
        run_sealed(_cfg_a(tmp, kw, out_dir=tmp / "docs_sealed" / "rehearsal"), sha)
    assert not list((tmp / "docs_sealed").iterdir())


@pytest.mark.parametrize("test_id", ["A", "B"])
def test_a_rehearsal_cannot_write_the_output_folder_of_a_sealed_test(registered, monkeypatch, test_id):
    """The sealed test refuses to run where its out_dir exists, so a rehearsal that wrote it would block the real run (and
    would be a run whose records sit in outputs/es/sealed)."""
    from es import sealed_run
    from es.sealed_run import default_config, run_sealed
    write, tmp = registered
    proto, reg, sha, kw = _bound(tmp, write, real_code=True)
    _no_read(monkeypatch)
    monkeypatch.setattr(sealed_run, "OUT_DIR", tmp / "outputs_es")      # the sealed tests' "real" folders, here under tmp
    real_out = default_config(test_id).out_dir
    assert real_out == tmp / "outputs_es" / "sealed" / test_id
    with pytest.raises(ValueError, match="output folder of a sealed test"):
        run_sealed(_cfg_a(tmp, kw, out_dir=real_out), sha)
    assert not real_out.exists() and not (tmp / "outputs_es").exists()


def test_a_rehearsal_reads_no_sealed_pull_and_keeps_its_records_beside_its_own_output(registered, monkeypatch):
    """C1: a config with no sealed arm reads no sealed pull, may use any n_rep, and writes its once-only records in the folder
    that holds its output, never in docs/sealed (where the real records are)."""
    from es.sealed_run import run_sealed
    write, tmp = registered
    proto, reg, sha, kw = _bound(tmp, write, real_code=True)
    _no_read(monkeypatch)
    with pytest.raises(ValueError, match="reads no sealed pull"):
        run_sealed(_cfg_a(tmp, kw, sealed_dirs=["sealed_2026Q3"]), sha)
    _nothing_recorded(tmp)
    _stop_at_the_read(monkeypatch)
    folder = tmp / "rehearsals" / "a"
    with pytest.raises(RuntimeError, match="stop"):
        run_sealed(_cfg_a(tmp, kw, out_dir=folder / "A"), sha, n_rep=7)
    assert json.loads((folder / "RUNS.json").read_text())["A"]["n_rep"] == 7 and (folder / "RUN_A.claim").exists()
    assert not list((tmp / "docs_sealed").iterdir()) and not (tmp / "RUNS.json").exists()
    _no_pull_check(monkeypatch)                                          # the registered Test A is not blocked by it
    with pytest.raises(RuntimeError, match="stop"):                      # (the reviewer's P3 wrote docs/sealed/RUNS.json['A'])
        run_sealed(_sealed_cfg(tmp, kw), sha)
    assert set(_sealed_runs(tmp)) == {"A"} and (tmp / "docs_sealed" / "RUN_A.claim").exists()


def _self_made(tmp):
    """A protocol and registration that the caller wrote itself (not es.registration's paths), binding its own files: what a
    rehearsal outside pytest looks like. Returns (sha, cfg kwargs)."""
    from es.registration import sha256_file
    d = tmp / "self"
    d.mkdir()
    cut, inp, man = d / "cutoffs.csv", d / "inputs.json", d / "CODE_MANIFEST.sha256"
    cut.write_text("analysis\n")
    inp.write_text(json.dumps({"eta": 0.01, "family": FAMILY}))
    man.write_text("")
    proto = d / "PROTOCOL.md"
    proto.write_text("a self-made protocol\n" + _protocol_lines(cut, inp, man))
    sha = sha256_file(proto)
    reg = d / "REGISTRATION.md"
    reg.write_text(f"protocol_sha256: {sha}\nosf_doi: 10.17605/OSF.IO/QQQQQ\nregistered_utc: 2026-10-01T00:00:00Z\n")
    return sha, dict(registration=(proto, reg), cutoffs=cut, inputs=inp, code_manifest=man, code_root=d)


def test_a_sealed_run_reads_es_registration_s_paths_only_whatever_cfg_registration_says(registered, monkeypatch):
    """I2, the reviewer's P3 and P4. A sealed config whose registration is a record the caller wrote (aimed at a registered
    pull, or at the real test id) is refused before anything is read or recorded; a rehearsal may read its own record, and
    then claims nothing in docs/sealed."""
    from es.sealed_run import run_sealed
    write, tmp = registered
    proto, reg, sha, kw = _bound(tmp, write, real_code=True)
    self_sha, self_kw = _self_made(tmp)
    _no_pull_check(monkeypatch)
    _no_read(monkeypatch)
    opened = _spy_parquet(monkeypatch)
    sealed = _sealed_cfg(tmp, {**kw, **self_kw})                        # sealed arms, pull and window; a record of its own
    with pytest.raises(ValueError, match="es.registration's protocol and registration record"):
        run_sealed(sealed, self_sha)
    with pytest.raises(ValueError, match="es.registration's protocol and registration record"):
        run_sealed(_sealed_cfg(tmp, {**kw, **self_kw}), sha)            # the registered sha does not help either
    assert opened == []
    _nothing_recorded(tmp)
    _stop_at_the_read(monkeypatch)
    rehearsal = _cfg_a(tmp, {**self_kw, "data_root": tmp}, out_dir=tmp / "rehearsal" / "A")
    with pytest.raises(RuntimeError, match="stop"):
        run_sealed(rehearsal, self_sha)                                 # reads the record it is given
    assert not list((tmp / "docs_sealed").iterdir())
    rec = json.loads((tmp / "rehearsal" / "RUNS.json").read_text())["A"]
    assert rec["protocol_sha256"] == self_sha and rec["registration_identifier"] == "10.17605/OSF.IO/QQQQQ"
    with pytest.raises(RuntimeError, match="stop"):
        run_sealed(_sealed_cfg(tmp, kw), sha)                           # the registered Test A is still free to run
    assert set(_sealed_runs(tmp)) == {"A"}


def test_a_sealed_run_needs_the_code_that_was_imported_as_its_code_root(registered, monkeypatch):
    """M8: the code manifest must describe the code that runs, so a sealed config's code_root is the folder es was imported
    from. (A rehearsal may hash another folder, as the departure tests do.)"""
    from es import sealed_run
    from es.sealed_run import run_sealed
    write, tmp = registered
    proto, reg, sha, kw = _bound(tmp, write, real_code=True)
    _no_pull_check(monkeypatch)
    _no_read(monkeypatch)
    assert sealed_run._imported_code_root() == DEV_ROOT == Path(__import__("es").__file__).resolve().parents[1]
    (tmp / "copy" / "es").mkdir(parents=True)                            # a copy of the code with the very same hashes
    for name in ("cutoffs.py", "sealed_run.py"):
        shutil.copy(DEV_ROOT / "es" / name, tmp / "copy" / "es" / name)
    with pytest.raises(ValueError, match="folder the es package was imported from"):
        run_sealed(_sealed_cfg(tmp, kw, code_root=tmp / "copy"), sha)
    _nothing_recorded(tmp)
    _stop_at_the_read(monkeypatch)
    with pytest.raises(RuntimeError, match="stop"):
        run_sealed(_sealed_cfg(tmp, kw, code_root=DEV_ROOT / "es" / ".."), sha)       # the same folder by another name


def test_the_sealed_panel_is_read_only_with_the_live_claim_of_the_run(registered, monkeypatch):
    """I1, the reviewer's P2 and P2b. The loader is private and needs the claim that only _claim_run makes (and _execute needs
    it too): called directly, after the registration and the pull, it read the sealed rows as often as it was asked, with no
    record of it. A missing, forged, foreign, altered or ended claim is refused, and nothing is read."""
    from es import sealed_run
    from es.sealed_run import _load_sealed_raw
    write, tmp = registered
    proto, reg, sha = write()
    dev, root = _fake_pull(tmp, proto, reg, sha)
    cfg = _cfg_pull(tmp, proto, reg, dev)
    reads = []
    real = sealed_run.load_dev_raw
    monkeypatch.setattr(sealed_run, "load_dev_raw", lambda *a, **k: reads.append(a) or real(*a, **k))
    opened = _spy_parquet(monkeypatch)
    assert not hasattr(sealed_run, "load_sealed_raw") and "needs the live claim" in _load_sealed_raw.__doc__

    class Lookalike:                                                    # every attribute of a claim, and not one
        live, test_id, nonce = True, "A", "x"
        out, path = Path(cfg.out_dir).resolve(), tmp / "docs_sealed" / "RUN_A.claim"

    with pytest.raises(TypeError):
        _load_sealed_raw(cfg, sha)                                      # no claim at all
    for fake in (None, "claim", object(), Lookalike()):
        with pytest.raises(PermissionError, match="live claim"):
            _load_sealed_raw(cfg, sha, fake)
    with pytest.raises(TypeError, match="made by _claim_run"):
        sealed_run._Claim(object(), "A", Path(cfg.out_dir).resolve(), tmp / "c", "n")
    Path(cfg.out_dir).mkdir()
    with pytest.raises(TypeError):                                      # _execute: the same, the reviewer's P2b
        sealed_run._execute(cfg, sha, 5, Path(cfg.out_dir), {}, 0.01, FAMILY, ["H-E"], None)
    with pytest.raises(PermissionError, match="live claim"):            # and it refuses before it writes a line
        sealed_run._execute(cfg, sha, 5, Path(cfg.out_dir), {}, 0.01, FAMILY, ["H-E"], None, None)
    assert reads == [] and opened == [] and not (tmp / "out" / "RUN_LOG.txt").exists() and not _sealed_runs(tmp)
    claim = _claimed(cfg, sha)
    other = _cfg_pull(tmp, proto, reg, dev)
    other.out_dir = tmp / "elsewhere"                                   # a claim is for the run that made it
    with pytest.raises(PermissionError, match="not for this config"):
        _load_sealed_raw(other, sha, claim)
    other = _cfg_pull(tmp, proto, reg, dev)
    other.test_id = "B"
    with pytest.raises(PermissionError, match="not for this config"):
        _load_sealed_raw(other, sha, claim)
    held = claim.path.read_text()
    claim.path.write_text(held.replace(claim.nonce, "0" * 32))           # the claim file is another claim
    with pytest.raises(PermissionError, match="does not hold this claim"):
        _load_sealed_raw(cfg, sha, claim)
    claim.path.write_text(held)
    assert reads == []
    assert len(_load_sealed_raw(cfg, sha, claim)) > 0 and len(reads) == 1      # the live claim of the run reads it
    claim.close()                                                       # the run has ended
    with pytest.raises(PermissionError, match="has ended"):
        _load_sealed_raw(cfg, sha, claim)
    assert len(reads) == 1


def test_the_claim_of_a_run_ends_with_it_whether_it_finishes_or_crashes(registered, monkeypatch):
    from es import sealed_run
    from es.sealed_run import run_sealed
    write, tmp = registered
    proto, reg, sha, kw = _bound(tmp, write)
    real, seen = sealed_run._load_sealed_raw, []

    def stop(cfg, protocol_sha256, claim):
        seen.append(claim)
        assert claim.live
        raise RuntimeError("stop")

    monkeypatch.setattr(sealed_run, "_load_sealed_raw", stop)
    cfg = _cfg_a(tmp, kw)
    with pytest.raises(RuntimeError, match="stop"):
        run_sealed(cfg, sha)
    assert len(seen) == 1 and seen[0].live is False                     # kept after the run: it is dead
    with pytest.raises(PermissionError, match="has ended"):
        real(cfg, sha, seen[0])


def test_finishing_a_run_adds_finished_utc_the_status_and_the_manifest_sha_to_its_record(registered):
    """I3: _finish writes finished_utc, the status and the SHA-256 of out_dir's manifest to the run's record in RUNS.json,
    which lives outside the folder (so the manifest still verifies), and nothing else of the record changes."""
    from es.sealed_pull import manifest_sha256, verify_manifest
    from es.sealed_run import _finish
    write, tmp = registered
    proto, reg, sha, kw = _bound(tmp, write)
    cfg = _cfg_a(tmp, kw)
    claim = _claimed(cfg, sha)
    before = _read_runs(tmp)["A"]
    out = Path(cfg.out_dir)
    res = _finish(out, {"test": "A", "started_utc": before["started_utc"], "status": "completed"}, claim)
    rec = _read_runs(tmp)["A"]
    assert rec == {**before, "finished_utc": res["finished_utc"], "status": "completed",
                   "manifest_sha256": manifest_sha256(out)}
    verify_manifest(out)
    assert (out / "RUN_LOG.txt").read_text().splitlines()[-1] == f"finished {res['finished_utc']}"


def test_a_record_that_cannot_be_updated_after_the_manifest_names_the_fields_and_leaves_the_output_alone(registered):
    from es.sealed_pull import manifest_sha256, verify_manifest
    from es.sealed_run import _finish
    write, tmp = registered
    proto, reg, sha, kw = _bound(tmp, write)
    cfg = _cfg_a(tmp, kw)
    claim = _claimed(cfg, sha)
    out = Path(cfg.out_dir)
    runs = _read_runs(tmp)
    runs["A"]["started_utc"] = "2026-10-23T05:00:00+00:00"              # the record is not this run's
    _runs_file(tmp).write_text(json.dumps(runs))
    with pytest.raises(RuntimeError, match=r"is not this run's.*add these fields to its record by hand") as err:
        _finish(out, {"test": "A", "started_utc": "2026-10-23T00:00:00+00:00", "status": "completed"}, claim)
    assert manifest_sha256(out) in str(err.value) and "finished_utc" in str(err.value)
    verify_manifest(out)                                                # the finished output is sealed and untouched
    assert "finished_utc" not in _read_runs(tmp)["A"]
    runs["A"]["started_utc"] = "2026-10-23T00:00:00+00:00"              # the record is this run's, and the disk is full
    _runs_file(tmp).write_text(json.dumps(runs))
    real = os.replace

    def killed(src, dst):
        if Path(dst).name == "RUNS.json":
            raise OSError("disk full")
        return real(src, dst)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(os, "replace", killed)
        with pytest.raises(RuntimeError, match=r"could not be updated \(OSError: disk full\).*by hand") as err:
            _finish(out, {"test": "A", "started_utc": "2026-10-23T00:00:00+00:00", "status": "completed"}, claim)
    assert "finished_utc" in str(err.value) and not list(tmp.glob("*.tmp"))


def _coverage_run(registered, monkeypatch, case="too_few_nodes", min_nodes=None, n_rep=5):
    """A run that fails the coverage rule on a tiny synthetic panel (so it is spent and finished in a second or two), as a
    rehearsal of Test A. Returns (results, cfg, sha, tmp)."""
    from es import sealed_run
    from es.analyses import Analysis
    from es.sealed_run import SealedConfig, run_sealed
    from es.windows import DateWindow as D
    write, tmp = registered
    dev = tmp / "dev"
    dev.mkdir()
    prices = _synthetic_prices("2025-01-01", "2025-04-30",
                               markets=tuple(m for m in NODE if not (case == "missing_operator" and m == "CAISO")))
    if case == "short_coverage":                      # a quarter of NYISO's March hours missing
        ny = prices["NYISO"]
        local = ny["timestamp_utc"].dt.tz_convert("America/New_York")
        prices["NYISO"] = ny[~((local.dt.date >= dt.date(2025, 3, 1)) & (local.dt.hour % 4 == 0))].reset_index(drop=True)
    for market, frame in prices.items():
        _write_prices(frame, dev / f"{market.lower()}_da_rt_syn.parquet")
    if min_nodes is not None:
        monkeypatch.setattr(sealed_run, "MIN_USABLE_NODES", min_nodes)
    fits = []
    monkeypatch.setattr(sealed_run, "score_analysis", lambda *a, **k: fits.append(a))
    train = D("2025-01-08", "2025-02-15")
    arm = Analysis("SMN", train, D("2025-02-16", "2025-02-28"), D("2025-03-01", "2025-03-31"), train, "rolling", "analog")
    proto, reg, sha, kw = _bound(tmp, write)
    cfg = SealedConfig("A", arm, arm, [], "2025-03-31", out_dir=tmp / "out", dev_dirs=(dev,), **kw)
    res = run_sealed(cfg, sha, n_rep=n_rep)
    assert fits == []
    return res, cfg, sha, tmp


def test_a_run_that_stopped_on_coverage_finishes_its_record_and_test_b_then_runs_alone(registered, monkeypatch):
    """I3 and I4 end to end, on a real (tiny) run: a NOT_RUN Test A has its finished record, the manifest of its folder is
    the number in it, and Test B, which finds Test A not run, goes on to its read on its own window."""
    from es.sealed_pull import manifest_sha256
    from es.sealed_run import run_sealed
    res, cfg, sha, tmp = _coverage_run(registered, monkeypatch)
    rec = _read_runs(tmp)["A"]
    assert res["status"] == "not run (coverage rule)" and rec["status"] == res["status"]
    assert rec["finished_utc"] == res["finished_utc"] and rec["manifest_sha256"] == manifest_sha256(tmp / "out")
    assert rec["started_utc"] == res["started_utc"] and rec["environment"] == res["environment"]
    _stop_at_the_read(monkeypatch)
    kw = {k: getattr(cfg, k) for k in ("registration", "cutoffs", "inputs", "code_manifest", "code_root", "data_root")}
    with pytest.raises(RuntimeError, match="stop"):
        run_sealed(_cfg_b(tmp, kw, tmp / "out"), sha)
    log = (tmp / "outB" / "RUN_LOG.txt").read_text()
    state = json.loads([ln for ln in log.splitlines() if ln.startswith("test_a ")][0].split(" ", 1)[1])
    assert state["state"] == "not run (coverage rule)" and state["manifest_sha256"] == rec["manifest_sha256"]
    assert "B" in _read_runs(tmp)


def test_a_failure_to_record_the_finish_does_not_touch_the_finished_output(registered, monkeypatch):
    """The crash handler appends to RUN_LOG.txt; after the manifest is written that would break the manifest of a complete
    output, so it does not (the exception still reaches the caller)."""
    from es import sealed_run
    from es.sealed_pull import verify_manifest
    monkeypatch.setattr(sealed_run, "_record_finished", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no record")))
    with pytest.raises(RuntimeError, match="no record"):
        _coverage_run(registered, monkeypatch)
    write, tmp = registered
    out = tmp / "out"
    verify_manifest(out)
    log = (out / "RUN_LOG.txt").read_text().splitlines()
    assert log[-1].startswith("finished ") and not any(ln.startswith("crashed ") for ln in log)
    assert "finished_utc" not in _read_runs(tmp)["A"]


def test_test_b_alone_pools_nothing_and_decides_nothing_but_runs_every_stage(tmp_path):
    """I4 through the stages. Test A's output is not available (no prior): E1-E4 are of Test B's window alone, H-E is the one-window rotation
    test and is not confirmatory, nothing replicates, and no step-down is run; the status says why."""
    from es.bootstrap import RowCube, rotation_test
    from es.sealed_run import _stage_decision, _stage_estimands, _stage_he
    run = _stage_run(tmp_path, test_id="B")
    assert run.alone and run.prior is None
    kept = _kept_windows(3)
    windows = _stage_estimands(run, kept)
    tables = tmp_path / "run_out" / "tables"
    assert windows == [kept] and run.errors == {} and not (tables / "estimands_this_quarter.csv").exists()
    assert run.results["estimands_scope"] == "Test B window alone (Test A's output is not available)"
    assert (tables / "estimands.csv").exists() and (tables / "estimands_ex_caiso.csv").exists()
    he = _stage_he(run, kept, windows)
    assert he["p"] == rotation_test([RowCube.from_rows(kept["P1"])])["p"] and he["n_null"] == 20 - 1
    hb = {"H-B1": {"t_hat": 0.4, "p": 0.001, "confirmatory": True}, "H-B2": {"t_hat": 0.3, "confirmatory": False}}
    _stage_decision(run, he, hb)
    entry = run.results["H-E"]
    assert entry["status"] == "not confirmatory (Test A's output is not available; the registered pooled test is unavailable)"
    assert entry["confirmatory"] is False and entry["p"] == he["p"] and run.results["holm"] == {
        "H-E": None, "H-B1": None, "H-B2": None}
    assert hb["H-B1"]["replicated"] is None and hb["H-B2"]["replicated"] is None and run.errors == {}


def test_test_b_pooled_with_test_a_keeps_the_scope_and_the_replication_verdict_for_a_confirmatory_h_b(tmp_path):
    from es.sealed_run import _stage_decision, _stage_estimands, _stage_he
    prior = tmp_path / "A_out"
    a, b = _kept_windows(7), _kept_windows(8, start=dt.date(2025, 10, 1))
    _write_prior_kept(prior, a)
    run = _stage_run(tmp_path, prior={"H-B1": {"p": 0.02, "t_hat": 0.2}, "H-B2": {"p": 0.3, "t_hat": 0.2}},
                     test_id="B", family={**FAMILY, "H-B2": {**FAMILY["H-B2"], "confirmatory": False, "reasons": ["power 0.4"]}},
                     confirmatory=["H-E", "H-B1"])
    run.cfg.prior_test_dir = prior
    windows = _stage_estimands(run, b)
    assert run.results["estimands_scope"] == "pooled A+B" and not run.alone
    he = _stage_he(run, b, windows)
    assert he["n_null"] == 20 * 20 - 1
    hb = {"H-B1": {"t_hat": 0.4, "p": 0.001, "confirmatory": True}, "H-B2": {"t_hat": 0.3, "confirmatory": False}}
    _stage_decision(run, he, hb)
    assert hb["H-B1"]["replicated"] is True and hb["H-B2"]["replicated"] is None and "p" not in hb["H-B2"]
    assert hb["H-B2"]["status"] == "estimand (registration rule): power 0.4" and run.results["H-E"]["status"].startswith("decided")


def test_h_b_estimands_have_no_p_in_their_decision_fields_and_the_percentile_interval_is_primary(tmp_path):
    """I5 and Ruling 110. A demoted H-B hypothesis is an estimand: no decision p (the bootstrap p is a diagnostic in a
    sub-dict of that name), no replication verdict, both intervals, and the record of which interval is primary. A
    confirmatory one is a test: its p, and the basic interval that goes with it."""
    from es.sealed_run import _stage_bootstraps, _stage_decision
    family = {**FAMILY, "H-B1": {"confirmatory": False, "reasons": ["size 0.19"], "approximate": True},
              "H-B2": {"confirmatory": True, "reasons": [], "approximate": False}}
    run = _stage_run(tmp_path, family=family, confirmatory=["H-E", "H-B2"])
    hb = _stage_bootstraps(run, _all_sets())
    est, test = hb["H-B1"], hb["H-B2"]
    assert est["confirmatory"] is False and "p" not in est and "replicated" not in est
    assert est["interval_registered"] == "percentile (primary); basic (diagnostic)"
    assert list(est["diagnostics"]) == ["bootstrap_p_one_sided"] and 0 < est["diagnostics"]["bootstrap_p_one_sided"] <= 1
    assert len(est["ci"]) == 2 and len(est["ci_basic"]) == 2 and est["ci_approximate"] is True
    assert test["confirmatory"] is True and 0 < test["p"] <= 1 and "diagnostics" not in test
    assert test["interval_registered"] == "basic" and len(test["ci"]) == 2 and len(test["ci_basic"]) == 2
    _stage_decision(run, {"t_obs": 1.0, "p": 0.2, "n_null": 5, "selc_obs": 0.0, "retained_share": []}, hb)       # Test A
    r = run.results
    assert r["H-B1"]["status"] == "estimand (registration rule): size 0.19" and "p" not in r["H-B1"]
    assert "replicated" not in r["H-B1"] and "replicated" not in r["H-B2"]               # Test A has no replication verdict
    assert r["H-B2"]["status"].startswith(("rejected in Test A", "not rejected at")) and "p" in r["H-B2"]
    assert "Traceback" not in json.dumps(r, default=str) and run.errors == {}


def test_e5_is_labelled_with_the_window_it_is_computed_on(tmp_path):
    """M5: E5 is the run's own window in both tests (the paper's pooled headline is E1-E4), and results.json says so."""
    from es.sealed_run import _stage_bootstraps
    for test_id, label in (("A", "Test A window"), ("B", "Test B window")):
        (tmp_path / test_id).mkdir()
        run = _stage_run(tmp_path / test_id, test_id=test_id)
        _stage_bootstraps(run, _all_sets())
        assert run.results["E5_scope"] == label and len(run.results["E5"]) == 10


def test_h_s_and_e5_report_the_aligned_interval_with_the_independent_one_beside_it_and_are_always_approximate(tmp_path):
    """Ruling 118, M2 (G1). The primary interval of H-S and E5 is the calendar-aligned draw; the registered independent draws'
    percentile interval is stored beside it as a diagnostic (H-S: ci_independent; E5, whose columns are flat: lo_independent,
    hi_independent, failures_lo_independent, failures_hi_independent); results say which scheme and which interval, and every
    one is approximate, whatever the NaN count. H-B and PC stay on the registered draws."""
    from es.bootstrap import block_contrast_bootstrap, block_level_bootstrap
    from es.sealed_run import ALPHA, HB_PAIRS, INTERVAL_E5_HS, _stage_bootstraps
    assert INTERVAL_E5_HS == "percentile, calendar-aligned (primary); independent draws (diagnostic)"
    sc = _all_sets()
    run = _stage_run(tmp_path)
    n = run.n_rep
    hb = _stage_bootstraps(run, sc)
    assert run.errors == {}
    for est in ("xgb", "logit"):
        frame = sc[("roll", "DART3", est, 42)]
        aligned = block_level_bootstrap(frame, alpha=ALPHA, n_rep=n, measure="failures", aligned=True)
        independent = block_level_bootstrap(frame, alpha=ALPHA, n_rep=n, measure="failures")
        hs = run.results["H-S"][est]
        assert (hs["failures"], hs["ci"], hs["n_nodes"]) == (aligned["t_hat"], list(aligned["ci"]), aligned["n_nodes"])
        assert hs["ci_independent"] == list(independent["ci"]) and hs["scheme"] == "aligned"
        assert hs["interval_registered"] == INTERVAL_E5_HS and hs["approximate"] is True
        assert hs["n_nan_replicates"] == 0 and hs["n_nan_replicates_independent"] == 0
    csv = pd.read_csv(tmp_path / "run_out" / "tables" / "E5_blocks.csv", float_precision="round_trip").set_index("block")
    assert len(run.results["E5"]) == 10
    for row in run.results["E5"]:
        frame = sc[("roll", row["block"], "xgb", 42)]
        median = {a: block_level_bootstrap(frame, alpha=ALPHA, n_rep=n, aligned=a) for a in (True, False)}
        failures = {a: block_level_bootstrap(frame, alpha=ALPHA, n_rep=n, measure="failures", aligned=a) for a in (True, False)}
        assert (row["median_log_error"], row["lo"], row["hi"]) == (median[True]["t_hat"], *median[True]["ci"])
        assert (row["lo_independent"], row["hi_independent"]) == median[False]["ci"]
        assert (row["failures"], row["failures_lo"], row["failures_hi"]) == (failures[True]["t_hat"], *failures[True]["ci"])
        assert (row["failures_lo_independent"], row["failures_hi_independent"]) == failures[False]["ci"]
        assert row["scheme"] == "aligned" and row["interval_registered"] == INTERVAL_E5_HS and row["approximate"] is True
        assert row["n_nan_replicates"] == row["n_nan_replicates_independent"] == 0
        assert row["n_nan_replicates_failures"] == row["n_nan_replicates_failures_independent"] == 0
        assert csv.loc[row["block"], "lo"] == row["lo"] and csv.loc[row["block"], "lo_independent"] == row["lo_independent"]
        assert csv.loc[row["block"], "scheme"] == "aligned" and bool(csv.loc[row["block"], "approximate"]) is True
    assert any(row["lo"] != row["lo_independent"] for row in run.results["E5"])          # the primary is not the old interval
    for hid, alt in HB_PAIRS:                                                               # H-B and PC are on the registered draws
        ref = block_contrast_bootstrap(sc[("roll", alt, "xgb", 42)], sc[("roll", "DART3", "xgb", 42)], alpha=ALPHA, n_rep=n)
        assert hb[hid]["ci"] == list(ref["ci"]) and hb[hid]["ci_basic"] == list(ref["ci_basic"]) and "scheme" not in hb[hid]
    assert all("scheme" not in v and "ci_independent" not in v for v in run.results["PC"].values())


def test_h_s_and_e5_with_nan_replicates_in_the_aligned_draw_are_computed_on_the_finite_ones(tmp_path):
    """Windows on different calendar days: the aligned draw is over the union of their (month, day) positions, so a replicate
    can draw none of one window's, and is NaN. H-S and E5 are still computed, on the finite replicates, and are approximate."""
    from es.sealed_run import _stage_bootstraps
    frame = _scores_in(("M", "N"), (("calib", dt.date(2025, 3, 1), 2), ("deploy", dt.date(2026, 7, 1), 10)), seed=5)
    run = _stage_run(tmp_path, n_rep=60)
    _stage_bootstraps(run, _all_sets(frame))
    hs = run.results["H-S"]["xgb"]
    assert "error" not in hs and hs["approximate"] is True and hs["scheme"] == "aligned"
    assert hs["n_nan_replicates"] > 0 and np.isfinite([*hs["ci"], *hs["ci_independent"]]).all()
    assert "nan_replicates H-S/xgb:" in (tmp_path / "run_out" / "RUN_LOG.txt").read_text()
    for row in run.results["E5"]:
        assert "error" not in row and row["approximate"] is True and row["n_nan_replicates"] > 0
        assert np.isfinite([row["lo"], row["hi"], row["failures_lo"], row["failures_hi"]]).all()


def test_the_nan_replicates_of_the_independent_draw_of_h_s_and_e5_are_counted_and_logged_too(tmp_path, monkeypatch):
    """The diagnostic draw has its own NaN count in H-S and in each E5 row (for the median and for the failures), its own line in
    the log, and its own 1% rule."""
    from es.bootstrap import percentile_ci
    from es.sealed_run import _stage_bootstraps
    aligned_star = np.r_[np.linspace(0.0, 1.0, 190), np.full(10, np.nan)]          # median: 10 of 200 (5%) ...
    independent_star = np.r_[np.linspace(0.2, 0.9, 194), np.full(6, np.nan)]        # ... and 6 of 200 (3%)
    failures = (np.r_[np.linspace(0.0, 5.0, 196), np.full(4, np.nan)],               # failures: 4 of 200 (2%) ...
                np.r_[np.linspace(0.0, 4.0, 199), np.nan])                           # ... and 1 of 200 (0.5%: not more than 1%)
    _stub_level_bootstrap(monkeypatch, aligned_star, independent_star, failures=failures)
    run = _stage_run(tmp_path, n_rep=200)
    _stage_bootstraps(run, _all_sets())
    hs = run.results["H-S"]["logit"]                                                 # H-S is the failures measure
    assert (hs["n_nan_replicates"], hs["n_nan_replicates_independent"]) == (4, 1) and hs["approximate"] is True
    assert hs["ci"] == list(percentile_ci(failures[0][:196])) and hs["ci_independent"] == list(percentile_ci(failures[1][:199]))
    log = (tmp_path / "run_out" / "RUN_LOG.txt").read_text()
    assert "nan_replicates H-S/logit: 4 of 200 (more than 1%: approximate)" in log
    assert "nan_replicates H-S/logit/independent: 1 of 200\n" in log               # (not more than 1%: no 'approximate' remark)
    for row in run.results["E5"]:
        assert (row["n_nan_replicates"], row["n_nan_replicates_independent"]) == (10, 6)
        assert (row["n_nan_replicates_failures"], row["n_nan_replicates_failures_independent"]) == (4, 1)
        assert (row["lo_independent"], row["hi_independent"]) == percentile_ci(independent_star[:194])
        assert (row["failures_lo_independent"], row["failures_hi_independent"]) == percentile_ci(failures[1][:199])
        assert (row["lo"], row["hi"]) == percentile_ci(aligned_star[:190]) and row["approximate"] is True


def _syn_economics():
    """Three nodes of synthetic combined-model and calendar-model score frames, as tests/es/test_run_economics.py builds them
    (45 training, 20 calibration and 28 deployment days in May-August 2025), and an analysis with those windows."""
    from es.analyses import Analysis
    from es.config import SETTLE_TZ
    from es.windows import DateWindow as D
    rng = np.random.default_rng(0)
    records = []
    for market, node in (("NYISO", "A"), ("ISONE", "B"), ("CAISO", "C")):
        for i in range(45 + 20 + 28):
            day = dt.date(2025, 5, 1) + dt.timedelta(days=i)
            split = "train" if i < 45 else "calib" if i < 65 else "deploy"
            spike_day = rng.random() < 0.1
            for h in range(24):
                hot = spike_day and 16 <= h <= 19
                dart = rng.normal(4.0, 2.0) - (14.0 if hot else 0.0)
                da = 40.0 + rng.normal(0.0, 3.0)
                ts = (pd.Timestamp(day) + pd.Timedelta(hours=h)).tz_localize(SETTLE_TZ[market]).tz_convert("UTC")
                records.append({"market": market, "node": node, "node_type": "zone", "timestamp_utc": ts, "settle_date": day,
                                "settle_hour": h, "weekday": day.weekday(), "wclass": int(day.weekday() >= 5), "da_lmp": da,
                                "rt_lmp": da - dart, "dart_spread": dart, "label": float(hot),
                                "abs_mean_168h": abs(rng.normal(3.0, 1.0)), "score": 2.0 * hot + rng.normal(0, 1), "split": split})
    primary = pd.DataFrame(records)
    g6 = primary.assign(score=0.5 * primary["settle_hour"].between(16, 19) + 0.01 * primary["weekday"])
    analysis = Analysis("SYN", D("2025-05-01", "2025-06-14"), D("2025-06-15", "2025-07-04"), D("2025-07-05", "2025-08-01"),
                        D("2025-05-01", "2025-06-14"), "rolling", "analog")
    return primary, g6, analysis


def test_h_e_still_runs_on_its_own_rows_when_economics_for_fails(tmp_path, monkeypatch):
    """I6. About 23 rotation tests and 250 summaries share economics_for's single guard; one comparator or exposure split that
    raises used to cost the run H-E, E1-E4 and the economic family together. Now H-E's rows are built on their own, with
    p1_rows_for, in a guard of their own; they are the rows economics_for keeps as kept['P1'], and H-E is the same test."""
    from es import sealed_run
    from es.bootstrap import RowCube, rotation_test
    from es.run_economics import economics_for
    from es.sealed_run import FAILED, _score_lookup, _stage_economics, _stage_he
    primary, g6, analysis = _syn_economics()
    sc = {("roll", "G1_24", "xgb", 42): primary, ("roll", "G6", "xgb", 42): g6}
    run = _stage_run(tmp_path)
    run.cfg.rolling = analysis
    _, kept = economics_for(None, analysis, None, None, 0.01, score_fn=_score_lookup(run.cfg, sc), keep_rows=True)
    monkeypatch.setattr(sealed_run, "economics_for", lambda *a, **k: (_ for _ in ()).throw(
        ValueError("summarize: one comparator has no loss")))
    econ, got = _stage_economics(run, None, sc)
    assert (econ, got) == (None, None) and run.errors == {"economics": "ValueError: summarize: one comparator has no loss"}
    assert run.results["economics"] == {"error": "ValueError: summarize: one comparator has no loss"}
    pd.testing.assert_frame_equal(run.p1_rows, kept["P1"])               # the very rows economics_for keeps
    he = _stage_he(run, got, None)
    assert he is not FAILED and he["p"] == rotation_test([RowCube.from_rows(kept["P1"])])["p"]
    assert "H-E" not in run.errors
    assert "economics_for did not complete" in (tmp_path / "run_out" / "RUN_LOG.txt").read_text()


def test_when_p1_rows_cannot_be_built_either_h_e_fails_alone_with_its_own_message(tmp_path, monkeypatch):
    from es import sealed_run
    from es.sealed_run import FAILED, _stage_economics, _stage_he
    primary, g6, analysis = _syn_economics()
    sc = {("roll", "G1_24", "xgb", 42): primary, ("roll", "G6", "xgb", 42): g6}
    run = _stage_run(tmp_path)
    run.cfg.rolling = analysis
    for name in ("economics_for", "p1_rows_for"):
        monkeypatch.setattr(sealed_run, name, lambda *a, **k: (_ for _ in ()).throw(ValueError("no node qualifies")))
    assert _stage_economics(run, None, sc) == (None, None) and run.p1_rows is None
    assert set(run.errors) == {"economics", "economics/p1_rows"}
    assert _stage_he(run, None, None) is FAILED and "needs the economic rows" in run.errors["H-E"]


def test_test_b_pooled_still_tests_h_e_from_test_a_s_p1_rows_when_economics_for_failed(tmp_path):
    """I6 in Test B: the estimands could not be built (no windows), H-E needs only the two P1 frames."""
    from es.bootstrap import RowCube, rotation_test
    from es.sealed_run import _stage_he
    prior = tmp_path / "A_out"
    a, b = _kept_windows(7), _kept_windows(8, start=dt.date(2025, 10, 1))
    _write_prior_kept(prior, a)
    run = _stage_run(tmp_path, prior={"H-B1": {"p": 0.1, "t_hat": 0.1}, "H-B2": {"p": 0.1, "t_hat": 0.1}})
    run.cfg.prior_test_dir = prior
    run.p1_rows = b["P1"]
    he = _stage_he(run, None, None)
    want = rotation_test([RowCube.from_rows(a["P1"]), RowCube.from_rows(b["P1"])])
    assert he["p"] == want["p"] and he["n_null"] == 20 * 20 - 1 and run.errors == {}


def test_the_audit_family_survives_nan_replicates_and_records_them(tmp_path, monkeypatch):
    """M1, the reviewer's S4a. audit_family called block_contrast_bootstrap directly, which refuses a NaN replicate: one of
    them lost all 18 audit tests. Now each of the 18 gets the finite-replicate fallback of the H-B contrasts, and the counts
    are recorded."""
    from es import sealed_run
    from es.bootstrap import block_contrast_bootstrap
    from es.sealed_run import _slim, _stage_secondary, score_keys
    alt, base = _contrast_pair(cal_alt=2, cal_base=2, dep=10)
    with pytest.raises(ValueError, match="NaN"):
        block_contrast_bootstrap(alt, base, alpha=0.05, n_rep=60)           # what audit_family used to call
    sc = {k: _slim(alt if k[1] in ("LEV6", "ROLL_DISP9") else base) for k in score_keys()}
    fake, _ = _fake_monitor_rows()
    monkeypatch.setattr(sealed_run, "monitor_rows", fake)
    run = _stage_run(tmp_path, n_rep=60)
    _stage_secondary(run, None, sc, _econ_rows())
    audit = run.results["secondary"]["audit robustness"]
    assert "error" not in audit and audit["tests"] == 18 and run.errors == {}
    assert len(audit["n_nan_replicates"]) == 18 and all(0 < n < 60 for n in audit["n_nan_replicates"].values())
    assert len(audit["nan_approximate"]) == 18                              # a large share of the replicates, so approximate
    table = pd.read_csv(tmp_path / "run_out" / "tables" / "secondary_bh.csv")
    audit_rows = table[table["family"] == "audit robustness"]
    assert len(audit_rows) == 18 and audit_rows["p"].between(0, 1).all() and audit_rows["included"].all()
    assert "nan_replicates audit/H-B1 logit:" in (tmp_path / "run_out" / "RUN_LOG.txt").read_text()


# --- I7: the environment (ENV_PINS.json is written and bound by Task 2A-4: {"executable": ..., "versions": {name: version}})
PINNED = ("python", "numpy", "pandas", "pyarrow", "xgboost", "scikit-learn", "scipy", "statsmodels")


def _bound_with_pins(tmp, write, executable=None, drop=(), text=None, bind=True, real_code=False, **versions):
    """_bound files plus docs/sealed/ENV_PINS.json: the live interpreter and library versions, some of them replaced (or dropped),
    or `text` as it is; `bind` says whether the protocol prints the file's hash."""
    from es.registration import sha256_file
    from es.sealed_run import live_environment
    live = live_environment()
    doc = {"executable": executable or live["executable"], "versions": {**live["versions"], **versions}}
    for name in drop:
        doc["versions"].pop(name)
    pins = tmp / "docs_sealed" / "ENV_PINS.json"
    pins.write_text(json.dumps(doc, indent=2) + "\n" if text is None else text)
    extra = f"ENV_PINS.json SHA-256: {sha256_file(pins)}\n" if bind else ""
    return pins, _bound(tmp, write, protocol_extra=extra, real_code=real_code)


def test_the_live_environment_is_the_interpreter_and_the_versions_the_pins_file_holds():
    from es.sealed_run import live_environment
    live = live_environment()
    assert live["executable"] == sys.executable and list(live["versions"]) == list(PINNED)
    assert live["versions"]["python"] == sys.version.split()[0] and live["versions"]["pandas"] == pd.__version__
    assert live["versions"]["numpy"] == np.__version__ and all(isinstance(v, str) for v in live["versions"].values())


@pytest.mark.parametrize("what", ["executable", *PINNED])
def test_the_run_is_refused_in_an_environment_other_than_the_pinned_one(registered, monkeypatch, what):
    """I7: a bound ENV_PINS.json that differs from the interpreter's path or from the version of python, numpy, pandas,
    pyarrow, xgboost, scikit-learn, scipy or statsmodels it runs on refuses the run, before the lock."""
    from es.registration import RegistrationError
    from es.sealed_run import run_sealed
    write, tmp = registered
    pins, (proto, reg, sha, kw) = _bound_with_pins(tmp, write, **({"executable": "/usr/bin/other-python"} if what == "executable"
                                                                   else {what: "0.0.0"}))
    _no_read(monkeypatch)
    with pytest.raises(RegistrationError, match=rf"differs from the pins the protocol binds.*{what}: pinned '.*', running"):
        run_sealed(_cfg_a(tmp, kw), sha)
    _nothing_recorded(tmp)


def test_a_sealed_config_checks_the_pins_too_and_matching_pins_are_recorded(registered, monkeypatch):
    from es.registration import RegistrationError, sha256_file
    from es.sealed_run import live_environment, run_sealed
    write, tmp = registered
    _no_pull_check(monkeypatch)
    _no_read(monkeypatch)
    pins, (proto, reg, sha, kw) = _bound_with_pins(tmp, write, real_code=True, numpy="0.0.0")
    with pytest.raises(RegistrationError, match="numpy: pinned '0.0.0'"):
        run_sealed(_sealed_cfg(tmp, kw), sha)
    assert not (tmp / "docs_sealed" / "RUNS.json").exists() and not list((tmp / "docs_sealed").glob("RUN_*.claim"))
    pins.write_text(json.dumps(live_environment(), indent=2) + "\n")       # the pins of this environment, bound afresh
    proto, reg, sha = write(_protocol_lines(kw["cutoffs"], kw["inputs"], kw["code_manifest"])
                            + f"ENV_PINS.json SHA-256: {sha256_file(pins)}\n")
    _stop_at_the_read(monkeypatch)
    with pytest.raises(RuntimeError, match="stop"):
        run_sealed(_sealed_cfg(tmp, kw), sha)
    env = _sealed_runs(tmp)["A"]["environment"]
    assert env == {**live_environment(), "pins": "matched", "pins_sha256": sha256_file(pins)}
    assert f"environment {json.dumps(env, sort_keys=True)}" in (_out(tmp) / "RUN_LOG.txt").read_text()


@pytest.mark.parametrize("what,match", [
    ("missing", "binds ENV_PINS.json, but .* does not exist"),
    ("changed", "not the registered file"),
    ("malformed", "not an environment pin file"),
    ("no_versions", "not an environment pin file"),
    ("a_library_not_pinned", "scipy: pinned None"),
])
def test_a_pin_file_the_protocol_binds_must_be_there_the_registered_one_and_complete(registered, monkeypatch, what, match):
    from es.registration import RegistrationError
    from es.sealed_run import run_sealed
    write, tmp = registered
    text = {"malformed": "{not json", "no_versions": json.dumps({"executable": sys.executable})}.get(what)
    pins, (proto, reg, sha, kw) = _bound_with_pins(tmp, write, text=text, drop=("scipy",) if what == "a_library_not_pinned" else ())
    if what == "missing":
        pins.unlink()
    elif what == "changed":
        pins.write_text(pins.read_text() + " ")                         # a pin file edited after the protocol bound it
    _no_read(monkeypatch)
    with pytest.raises(RegistrationError, match=match):
        run_sealed(_cfg_a(tmp, kw), sha)
    _nothing_recorded(tmp)


def test_a_pin_file_the_protocol_does_not_bind_pins_nothing_and_the_environment_is_recorded_either_way(registered, monkeypatch):
    from es.sealed_run import live_environment, run_sealed
    write, tmp = registered
    pins, (proto, reg, sha, kw) = _bound_with_pins(tmp, write, bind=False, numpy="0.0.0")   # there, unbound, and wrong
    _stop_at_the_read(monkeypatch)
    with pytest.raises(RuntimeError, match="stop"):
        run_sealed(_cfg_a(tmp, kw), sha)
    assert _read_runs(tmp)["A"]["environment"] == {**live_environment(), "pins": "not bound", "pins_sha256": None}


# --- I8: free disk, and a log that is best effort inside the guards
@pytest.mark.parametrize("where,label", [("out_dir", "out_dir"), ("sealed_dir", "the sealed records")])
def test_a_run_with_less_than_2_gb_free_is_refused_before_its_lock(registered, monkeypatch, where, label):
    from es import sealed_run
    from es.sealed_run import MIN_FREE_BYTES, run_sealed
    assert MIN_FREE_BYTES == 2 * 1024 ** 3
    write, tmp = registered
    proto, reg, sha, kw = _bound(tmp, write)
    _no_read(monkeypatch)
    target = tmp / "out" if where == "out_dir" else tmp / "docs_sealed"
    monkeypatch.setattr(sealed_run, "_free_bytes", lambda p: MIN_FREE_BYTES - 1 if Path(p) == target else 10 ** 12)
    with pytest.raises(RuntimeError, match=rf"only 2\.15 GB free where {label} lives"):
        run_sealed(_cfg_a(tmp, kw), sha)
    _nothing_recorded(tmp)
    _stop_at_the_read(monkeypatch)
    monkeypatch.setattr(sealed_run, "_free_bytes", lambda p: MIN_FREE_BYTES)        # exactly 2 GB is enough
    with pytest.raises(RuntimeError, match="stop"):
        run_sealed(_cfg_a(tmp, kw), sha)


def test_free_bytes_reads_the_nearest_existing_parent_of_a_folder_that_is_not_there_yet(tmp_path):
    from es.sealed_run import _free_bytes
    free = _free_bytes(tmp_path / "not" / "yet" / "there")
    assert isinstance(free, int) and free > 0 and abs(free - shutil.disk_usage(tmp_path).free) < 10 ** 9


def test_logging_inside_a_guard_is_best_effort_so_a_full_disk_does_not_end_the_run(tmp_path, monkeypatch):
    """I8, the reviewer's S7: the guards log the failure they catch, and an OSError there escaped both guards. The failure is
    now recorded in memory first (and so reaches results.json), the log write is best effort, and so are the other lines a
    stage logs after the lock."""
    from es import sealed_run
    from es.sealed_run import FAILED, _stage, _stage_call
    real = sealed_run._log
    wrote = []

    def full_disk(out, *lines):
        if lines and str(lines[0]).startswith(("error ", "nan_replicates", "stage ", "tau_check")):
            raise OSError(28, "No space left on device")
        wrote.append(lines)
        return real(out, *lines)

    monkeypatch.setattr(sealed_run, "_log", full_disk)
    run = _stage_run(tmp_path)
    assert _stage_call(run, "audit", lambda: run.guard("audit_activity", lambda: 1 / 0), None) is None
    assert run.errors == {"audit_activity": "ZeroDivisionError: division by zero"} and run.status() == "completed with errors"
    assert run.error_entry("audit_activity") == {"error": "ZeroDivisionError: division by zero"}
    assert run.guard("again", lambda: 1 / 0) is FAILED and "again" in run.errors
    run.note_nan("H-B1", {"n_nan_replicates": 3, "nan_approximate": True})          # best effort too
    _stage(tmp_path / "run_out", "bootstraps")
    assert wrote == []
    log = tmp_path / "run_out" / "RUN_LOG.txt"
    assert not log.exists() or "error audit_activity" not in log.read_text()


@pytest.mark.parametrize("what", ["no_fmm_folder", "no_nyiso_folder", "fmm_folder_without_a_table"])
def test_a_missing_development_settlement_table_refuses_the_run_before_its_lock(registered, monkeypatch, what):
    """M9: without caiso_fmm_dev or nyiso_rtint_dev the loaders return None and the premium direction is set from the
    5-minute settlement without a word. A listing of the folders is enough to see it: no row is read."""
    from es.sealed_run import run_sealed
    write, tmp = registered
    proto, reg, sha, kw = _bound(tmp, write)
    _no_read(monkeypatch)
    if what == "no_fmm_folder":
        shutil.rmtree(tmp / "caiso_fmm_dev")
    elif what == "no_nyiso_folder":
        shutil.rmtree(tmp / "nyiso_rtint_dev")
    else:
        (tmp / "caiso_fmm_dev" / "caiso_fmm_x.parquet").unlink()
    with pytest.raises(ValueError, match="development settlement tables are missing"):
        run_sealed(_cfg_a(tmp, kw), sha)
    _nothing_recorded(tmp)


def test_every_status_and_wording_string_is_a_module_constant_and_the_old_wording_is_gone():
    """The protocol quotes the vocabulary of the run, so every status string the module writes is one of its constants
    (Ruling 109); the wording of a confirmatory H-B result is 'larger operating-point error ...' (Ruling 84(d)), and the words
    it replaced appear nowhere in the module."""
    import ast
    from es import sealed_run
    src = Path(sealed_run.__file__).read_text()
    tree = ast.parse(src)
    docstrings = {id(n.body[0].value) for n in ast.walk(tree) if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef))
                  and n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    prefixes = ("completed", "not run", "interim (", "decided (", "not confirmatory", "estimand (", "rejected in Test",
                "not rejected at", "larger operating-point", "no difference in", "undetermined:", "timing skill",
                "single window", "pooled A+B", "Test B window", "Test A window")
    inside_functions = [c.value for fn in ast.walk(tree) if isinstance(fn, (ast.FunctionDef, ast.ClassDef))
                        for c in ast.walk(fn) if isinstance(c, ast.Constant) and isinstance(c.value, str)
                        and id(c) not in docstrings and c.value.startswith(prefixes)]
    assert inside_functions == []
    constants = {k: v for k, v in vars(sealed_run).items() if k.startswith(("STATUS_", "WORDING_", "SCOPE_", "INTERVAL_"))}
    assert {k: constants[k] for k in ("STATUS_COMPLETED", "STATUS_SECONDARY", "STATUS_ERRORS", "STATUS_NOT_RUN")} == {
        "STATUS_COMPLETED": "completed", "STATUS_SECONDARY": "completed with secondary errors",
        "STATUS_ERRORS": "completed with errors", "STATUS_NOT_RUN": "not run (coverage rule)"}
    assert len(constants) >= 20 and all(isinstance(v, (str, dict)) for v in constants.values())
    assert constants["STATUS_HB_BOTH"] == "larger operating-point error in both 2026 quarters"
    assert constants["STATUS_HB_Q3_ONLY"] == "larger operating-point error in Q3 2026; not replicated in Q4"
    assert constants["STATUS_HB_NONE"] == "no difference in operating-point error detected"
    for name in ("STATUS_HE_ALONE", "STATUS_HB_NO_TEST_A", "WORDING_HE_ALONE", "SCOPE_B_ALONE", "SCOPE_B_OWN_QUARTER"):
        assert "not available" in constants[name] and "did not run" not in constants[name], name        # Ruling 116, Mn6
    assert "less " + "port" + "able" not in src and "port" + "ab" + "ility" not in src          # (spelled apart: not here either)


# =====================================================================================================
# Controller Ruling 116 (fix round 2 of Task 2A-3, from the scoped re-review of round 1): a finished Test A without its
# stored rows (N1), a complete Test A whose record lacks its finish fields (N2), a failure between out_dir and the claim (N3),
# the development data of a sealed config (N4), a rehearsal nested in the sealed folders (Mn1), the interpreter pin (Mn2),
# the re-check of Test A's rows at the read (Mn3) and the wording of the states (Mn6). The reviewer's R-probes are tests here.
# =====================================================================================================
def test_test_a_s_fallback_keeps_the_p1_rows_it_tested_h_e_on_and_the_manifest_covers_them(tmp_path, monkeypatch):
    """N1(a). When economics_for fails and H-E is tested on p1_rows_for's rows (I6), those rows are written to kept/P1.parquet,
    so that the manifest of Test A's output holds them and Test B can pool H-E with them."""
    from es import sealed_run
    from es.bootstrap import RowCube, rotation_test
    from es.sealed_pull import verify_manifest, write_manifest
    from es.sealed_run import _prior_p1, _stage_economics
    primary, g6, analysis = _syn_economics()
    sc = {("roll", "G1_24", "xgb", 42): primary, ("roll", "G6", "xgb", 42): g6}
    run = _stage_run(tmp_path)
    run.cfg.rolling = analysis
    monkeypatch.setattr(sealed_run, "economics_for", lambda *a, **k: (_ for _ in ()).throw(ValueError("one comparator")))
    assert _stage_economics(run, None, sc) == (None, None)
    out = tmp_path / "run_out"
    assert [f.name for f in (out / "kept").glob("*")] == ["P1.parquet"] and list(run.errors) == ["economics"]
    write_manifest(out)                                              # what _finish does
    verify_manifest(out)
    assert "  kept/P1.parquet" in (out / "MANIFEST.sha256").read_text()
    back = _prior_p1(out)                                            # read back through the manifest check
    assert rotation_test([RowCube.from_rows(back)])["p"] == rotation_test([RowCube.from_rows(run.p1_rows)])["p"]


def test_a_p1_file_that_cannot_be_written_is_its_own_error_and_h_e_still_runs_in_memory(tmp_path, monkeypatch):
    from es import sealed_run
    from es.sealed_run import FAILED, _stage_economics, _stage_he
    primary, g6, analysis = _syn_economics()
    sc = {("roll", "G1_24", "xgb", 42): primary, ("roll", "G6", "xgb", 42): g6}
    run = _stage_run(tmp_path)
    run.cfg.rolling = analysis
    monkeypatch.setattr(sealed_run, "economics_for", lambda *a, **k: (_ for _ in ()).throw(ValueError("one comparator")))
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    assert _stage_economics(run, None, sc) == (None, None)
    assert set(run.errors) == {"economics", "economics/kept"} and run.errors["economics/kept"] == "OSError: disk full"
    assert _stage_he(run, None, None) is not FAILED and "H-E" not in run.errors


def test_test_b_estimands_cover_its_own_quarter_when_test_a_s_other_stored_rows_are_missing(tmp_path):
    """N1(b): E1-E4 are Test B's own quarter, with a scope label, and H-E is still pooled from the two P1 frames."""
    from es.bootstrap import RowCube, rotation_test
    from es.run_economics import KEPT
    from es.sealed_run import _stage_estimands, _stage_he
    prior = tmp_path / "A_out"
    a, b = _kept_windows(7), _kept_windows(8, start=dt.date(2025, 10, 1))
    missing = [n for n in KEPT if n != "P1"]
    _write_prior_kept(prior, a, skip=missing)
    run = _stage_run(tmp_path, prior={"H-B1": {"p": 0.1, "t_hat": 0.1}, "H-B2": {"p": 0.1, "t_hat": 0.1}})
    run.cfg.prior_test_dir = prior
    run.results["test_a"] = {"state": "pooled", "missing_kept": missing}
    windows = _stage_estimands(run, b)
    tables = tmp_path / "run_out" / "tables"
    assert run.errors == {} and windows == [b]
    est = pd.read_csv(tables / "estimands.csv").set_index("stat")
    assert (est["n_windows"] == 1).all() and not (tables / "estimands_this_quarter.csv").exists()
    assert (tables / "estimands_ex_caiso.csv").exists()
    assert run.results["estimands_scope"] == "Test B window only (Test A's stored rows for E1-E4 are not available)"
    he = _stage_he(run, b, windows)
    want = rotation_test([RowCube.from_rows(a["P1"]), RowCube.from_rows(b["P1"])])
    assert he["p"] == want["p"] and he["n_null"] == 20 * 20 - 1 and run.errors == {}


def test_test_a_s_rows_are_checked_against_its_manifest_when_test_b_reads_them(tmp_path):
    """Mn3 (the reviewer's R13). The pre-lock check passed, then a kept file changed: each file Test B reads after its lock is
    compared with the SHA-256 in Test A's MANIFEST.sha256, and a mismatch is that analysis' own recorded error."""
    from es.sealed_run import FAILED, _stage_estimands, _stage_he
    a, b = _kept_windows(7), _kept_windows(8, start=dt.date(2025, 10, 1))
    prior = tmp_path / "A_out"
    _write_prior_kept(prior, a)
    run = _stage_run(tmp_path / "ok", prior=prior) if (tmp_path / "ok").mkdir() is None else None
    windows = _stage_estimands(run, b)
    assert run.errors == {} and len(windows) == 2 and _stage_he(run, b, windows)["n_null"] == 399
    # one of the four other frames changes: the pooled estimands are lost (their own error), H-E (P1) is not
    a["P4"].assign(pnl_open=a["P4"]["pnl_open"] * 10).to_parquet(prior / "kept" / "P4.parquet", index=False)
    (tmp_path / "p4").mkdir()
    run = _stage_run(tmp_path / "p4", prior=prior)
    windows = _stage_estimands(run, b)
    assert windows is None and "changed since Test A's manifest was written" in run.errors["prior_kept"]
    assert {"prior_kept", "estimands", "estimands_ex_caiso"} <= set(run.errors) and "H-E" not in run.errors
    assert _stage_he(run, b, windows) is not FAILED and "H-E" not in run.errors
    # P1 changes (the reviewer's R13): H-E is that analysis' error, and the estimands too
    a["P1"].assign(pnl_open=a["P1"]["pnl_open"] * 10).to_parquet(prior / "kept" / "P1.parquet", index=False)
    (tmp_path / "p1").mkdir()
    run = _stage_run(tmp_path / "p1", prior=prior)
    windows = _stage_estimands(run, b)
    assert windows is None and _stage_he(run, b, windows) is FAILED
    assert "changed since Test A's manifest was written" in run.errors["H-E"] and "prior_kept" in run.errors


@pytest.mark.parametrize("manifest,message", [("none", "no MANIFEST.sha256 in .*cannot be checked"),
                                              ("without_the_file", "kept/P1.parquet is not listed in the MANIFEST.sha256")])
def test_a_kept_file_that_the_manifest_does_not_vouch_for_is_not_read(tmp_path, manifest, message):
    """Mn3: with no manifest there is nothing to check a stored file against, and a manifest that does not list the file (it
    was added after Test A finished) does not vouch for it. Either way the file is not used, and each says so in its own words."""
    from es.sealed_pull import write_manifest
    from es.sealed_run import _stage_he, FAILED
    prior = tmp_path / "A_out"
    (prior / "kept").mkdir(parents=True)
    if manifest == "without_the_file":
        (prior / "results.json").write_text("{}")
        write_manifest(prior)                                       # lists results.json only
    _kept_windows(7)["P1"].to_parquet(prior / "kept" / "P1.parquet", index=False)         # after the manifest, or with none
    run = _stage_run(tmp_path, prior={"H-B1": {"p": 0.1, "t_hat": 0.1}, "H-B2": {"p": 0.1, "t_hat": 0.1}})
    run.cfg.prior_test_dir = prior
    run.p1_rows = _kept_windows(8, start=dt.date(2025, 10, 1))["P1"]
    assert _stage_he(run, None, None) is FAILED
    assert re.search(message, run.errors["H-E"]), run.errors["H-E"]


# --- N3: a failure between out_dir and the claim
@pytest.mark.parametrize("where", ["records", "output_parent", "rehearsal_folder"])
def test_a_folder_the_run_cannot_write_is_refused_before_the_lock(registered, monkeypatch, where):
    """N3 (the reviewer's R7c). docs/sealed made read-only after the pull, to protect the records, failed the claim after out_dir
    had been made, and Test A was refused for good. The store and out_dir's parent are checked for write access before the lock:
    for a sealed test they are two folders (docs/sealed, and the output folder's parent: each is tested with the other writable);
    for a rehearsal they are one."""
    import stat
    from es.sealed_run import run_sealed
    write, tmp = registered
    docs = tmp / "docs_sealed"
    if where == "rehearsal_folder":
        proto, reg, sha, kw = _bound(tmp, write)
        target = tmp / "protected"
        target.mkdir()
        cfg = _cfg_a(tmp, kw, out_dir=target / "A")
    else:
        proto, reg, sha, kw = _bound(tmp, write, real_code=True)
        _no_pull_check(monkeypatch)
        cfg = _sealed_cfg(tmp, kw)
        if where == "records":
            target = docs
        else:
            target = _out(tmp).parent                              # the registered output folder's parent: outputs/es/sealed
            target.mkdir(parents=True)
    mode = target.stat().st_mode
    os.chmod(target, stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP)
    _no_read(monkeypatch)
    try:
        with pytest.raises(PermissionError, match=r"not writable"):
            run_sealed(cfg, sha)
    finally:
        os.chmod(target, mode)
    assert not list(target.iterdir()) and not (tmp / "out").exists() and not (tmp / "RUNS.json").exists()
    assert not (docs / "RUNS.json").exists() and not list(docs.glob("RUN_*.claim"))        # nothing was claimed anywhere
    assert not _out(tmp).exists()
    _stop_at_the_read(monkeypatch)
    with pytest.raises(RuntimeError, match="stop"):                       # permissions restored: nothing was spent
        run_sealed(cfg, sha)


def test_an_interrupt_before_the_lock_leaves_nothing_behind_and_the_run_can_be_made_again(registered, monkeypatch):
    """N3 (the reviewer's R7, Ctrl-C at the claim). The claim is the lock and it comes first: an interrupt there leaves no
    output folder, no claim and no record, so there is nothing to remove and the run is not spent."""
    from es.sealed_run import run_sealed
    write, tmp = registered
    proto, reg, sha, kw = _bound(tmp, write)
    _stop_at_the_read(monkeypatch)
    cfg = _cfg_a(tmp, kw)
    real = os.open

    def interrupt(path, flags, *a, **k):
        if str(path).endswith(".claim"):
            raise KeyboardInterrupt
        return real(path, flags, *a, **k)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(os, "open", interrupt)
        with pytest.raises(KeyboardInterrupt):
            run_sealed(cfg, sha)
    assert not (tmp / "out").exists() and not (tmp / "RUN_A.claim").exists() and not (tmp / "RUNS.json").exists()
    with pytest.raises(RuntimeError, match="stop"):
        run_sealed(cfg, sha)


def test_the_claim_comes_before_the_output_folder_and_a_failure_after_it_is_a_crash_that_counts(registered, monkeypatch):
    """N3. Order: the O_EXCL claim, then out_dir, then the 'started' line, which is inside the crash handler: a failure there
    is logged as a crash, the run counts (the claim is the lock), Test A is refused again and Test B runs alone."""
    from es import sealed_run
    from es.sealed_run import run_sealed
    write, tmp = registered
    proto, reg, sha, kw = _bound(tmp, write)
    cfg = _cfg_a(tmp, kw)
    seen, real_open, real_log = {}, os.open, sealed_run._log

    def open_(path, flags, *a, **k):
        if str(path).endswith(".claim"):
            seen["out_exists_at_the_claim"] = (tmp / "out").exists()
        return real_open(path, flags, *a, **k)

    def log(out, *lines):
        if lines and str(lines[0]).startswith("started "):
            seen["claim_exists_at_started"] = (tmp / "RUN_A.claim").exists()
            raise OSError(28, "No space left on device")
        return real_log(out, *lines)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(os, "open", open_)
        mp.setattr(sealed_run, "_log", log)
        with pytest.raises(OSError, match="No space left"):
            run_sealed(cfg, sha)
    assert seen == {"out_exists_at_the_claim": False, "claim_exists_at_started": True}
    lines = (tmp / "out" / "RUN_LOG.txt").read_text().splitlines()
    assert lines[0].startswith("crashed ") and "No space left" in lines[0]                  # the handler covered the started line
    assert not (tmp / "RUNS.json").exists() and (tmp / "RUN_A.claim").exists()
    with pytest.raises(FileExistsError, match=r"RUN_A.claim exists.*runs once"):
        run_sealed(cfg, sha)
    _stop_at_the_read(monkeypatch)
    with pytest.raises(RuntimeError, match="stop"):
        run_sealed(_cfg_b(tmp, kw, tmp / "out"), sha)
    assert _test_a_state(tmp)["state"] == "claimed, not recorded"


def test_an_output_folder_with_no_claim_is_refused_as_nothing_read_and_a_log_that_read_data_still_counts(registered, monkeypatch):
    """N3. An out_dir with no claim file for its test was never locked: the message says nothing was read and to remove it
    (it used to say that the run counts). A folder whose log shows the data load is another matter."""
    from es.sealed_run import run_sealed
    write, tmp = registered
    proto, reg, sha, kw = _bound(tmp, write)
    _stop_at_the_read(monkeypatch)
    (tmp / "out").mkdir()
    with pytest.raises(FileExistsError, match=r"nothing was read: remove .*out and run again") as err:
        run_sealed(_cfg_a(tmp, kw), sha)
    assert "counts" not in str(err.value)
    (tmp / "out").rmdir()
    with pytest.raises(RuntimeError, match="stop"):
        run_sealed(_cfg_a(tmp, kw), sha)
    (tmp / "elsewhere").mkdir()                                          # a log that shows the load, and no claim: tampering
    (tmp / "elsewhere" / "RUN_LOG.txt").write_text("started x\nstage load y\n")
    (tmp / "RUN_A.claim").unlink()
    (tmp / "RUNS.json").unlink()
    with pytest.raises(FileExistsError, match="counts"):
        run_sealed(_cfg_a(tmp, kw, out_dir=tmp / "elsewhere"), sha)


# --- N4: the development data of a sealed config
@pytest.mark.parametrize("what", ["empty", "missing_folder", "only_the_first", "reversed", "one_more"])
def test_a_sealed_config_must_read_the_registered_development_data(registered, monkeypatch, what):
    """N4 (the reviewer's R2). dev_dirs=() or a missing folder passed every pre-lock check, claimed, and crashed in the load,
    which spent Test A; a partial tuple ends NOT_RUN. A sealed config reads exactly DEV_DATA_DIRS."""
    from es.config import DEV_DATA_DIRS
    from es.sealed_run import run_sealed
    write, tmp = registered
    proto, reg, sha, kw = _bound(tmp, write, real_code=True)
    _no_pull_check(monkeypatch)
    _no_read(monkeypatch)
    dirs = {"empty": (), "missing_folder": (tmp / "nowhere",), "only_the_first": DEV_DATA_DIRS[:1],
            "reversed": DEV_DATA_DIRS[::-1], "one_more": (*DEV_DATA_DIRS, tmp)}[what]
    with pytest.raises(ValueError, match="development data"):
        run_sealed(_sealed_cfg(tmp, kw, dev_dirs=dirs), sha)
    _nothing_recorded(tmp)


def test_a_sealed_config_may_name_the_development_folders_by_another_path(registered, monkeypatch):
    """The comparison is of resolved paths, and the registered folders are a module constant that tests with fake development
    data patch, as they patch N_REP."""
    from es import sealed_run
    from es.sealed_run import run_sealed
    write, tmp = registered
    proto, reg, sha, kw = _bound(tmp, write, real_code=True)
    _no_pull_check(monkeypatch)
    _stop_at_the_read(monkeypatch)
    dev = tmp / "dev"
    dev.mkdir()
    link = tmp / "dev_link"
    link.symlink_to(dev, target_is_directory=True)
    monkeypatch.setattr(sealed_run, "DEV_DATA_DIRS", (dev,))
    with pytest.raises(RuntimeError, match="stop"):
        run_sealed(_sealed_cfg(tmp, kw, dev_dirs=(link,)), sha)


@pytest.mark.parametrize("test_id", ["A", "B"])
def test_the_registered_config_of_each_test_passes_the_config_check_with_the_real_development_folders(registered, test_id):
    """The tests that go past the config check patch DEV_DATA_DIRS to fake data; this one reads the real constant, so that
    binding dev_dirs to it (N4) cannot refuse the registered config itself."""
    from es.sealed_run import N_REP, _check_config, default_config
    cfg = default_config(test_id)
    _check_config(cfg, N_REP)


# --- N2b (Ruling 122): a sealed run's folders are the registered ones
@pytest.mark.parametrize("test_id,what", [
    ("A", "out_dir_elsewhere"), ("A", "out_dir_is_test_bs"), ("A", "out_dir_inside_its_own"),
    ("B", "out_dir_elsewhere"), ("B", "out_dir_is_test_as"), ("B", "prior_elsewhere"), ("B", "prior_is_test_bs")])
def test_a_sealed_run_writes_its_registered_output_folder_and_test_b_reads_test_a_s(registered, monkeypatch, test_id, what):
    """N2b (the reviewer's V2). A Test A run into another folder, whose record update then failed, left Test B (pointed at the
    registered folder) to run alone and lose the pooled H-E. A config with a sealed arm now names default_config(test_id)'s
    out_dir, and Test B's prior_test_dir is default_config('A')'s: each is refused before the lock, with nothing created."""
    from es.sealed_pull import manifest_sha256
    from es.sealed_run import run_sealed
    write, tmp = registered
    proto, reg, sha, kw = _bound(tmp, write, real_code=True)
    _no_pull_check(monkeypatch)
    _no_read(monkeypatch)
    if test_id == "B":                                       # Test A has run and finished, in its registered folder
        prior = _prior_dir(tmp, sha, record=False, folder=_out(tmp, "A"))
        _record_run(tmp / "docs_sealed", "A", prior, sha, STARTED_A, manifest_sha256(prior))
    over = {"out_dir_elsewhere": {"out_dir": tmp / "elsewhere"},
            "out_dir_is_test_bs": {"out_dir": _out(tmp, "B")},
            "out_dir_inside_its_own": {"out_dir": _out(tmp, "A") / "run"},
            "out_dir_is_test_as": {"out_dir": _out(tmp, "A")},
            "prior_elsewhere": {"prior_test_dir": tmp / "A_elsewhere"},
            "prior_is_test_bs": {"prior_test_dir": _out(tmp, "B")}}[what]
    with pytest.raises(ValueError, match="Test A's registered output folder" if what.startswith("prior") else "registered output folder"):
        run_sealed(_sealed_cfg(tmp, kw, test_id=test_id, **over), sha)
    assert not (tmp / "elsewhere").exists() and not (tmp / "A_elsewhere").exists() and not _out(tmp, "B").exists()
    assert not list((tmp / "docs_sealed").glob("RUN_B.claim")) and set(_sealed_runs(tmp)) == ({"A"} if test_id == "B" else set())
    if test_id == "A":
        _nothing_recorded(tmp)


def test_a_sealed_run_may_name_its_registered_output_folder_by_another_path(registered, monkeypatch):
    """The comparison is of resolved paths, as for dev_dirs and the registration paths."""
    from es.sealed_run import run_sealed
    write, tmp = registered
    proto, reg, sha, kw = _bound(tmp, write, real_code=True)
    _no_pull_check(monkeypatch)
    _stop_at_the_read(monkeypatch)
    (tmp / "outputs_es" / "sealed").mkdir(parents=True)
    (tmp / "link").symlink_to(tmp / "outputs_es" / "sealed", target_is_directory=True)
    with pytest.raises(RuntimeError, match="stop"):
        run_sealed(_sealed_cfg(tmp, kw, out_dir=tmp / "link" / "A"), sha)
    assert _sealed_runs(tmp)["A"]["out_dir"] == str(_out(tmp).resolve()) and (_out(tmp) / "RUN_LOG.txt").exists()


def test_a_sealed_test_b_may_name_the_registered_folders_by_another_path(registered, monkeypatch):
    """The same for Test B: its out_dir and its prior_test_dir are compared resolved, and the record of Test A names the
    resolved folder."""
    from es.sealed_pull import manifest_sha256
    from es.sealed_run import run_sealed
    write, tmp = registered
    proto, reg, sha, kw = _bound(tmp, write, real_code=True)
    _no_pull_check(monkeypatch)
    _stop_at_the_read(monkeypatch)
    prior = _prior_dir(tmp, sha, record=False, folder=_out(tmp, "A"))
    _record_run(tmp / "docs_sealed", "A", prior, sha, STARTED_A, manifest_sha256(prior))
    (tmp / "link").symlink_to(tmp / "outputs_es" / "sealed", target_is_directory=True)
    with pytest.raises(RuntimeError, match="stop"):
        run_sealed(_sealed_cfg(tmp, kw, test_id="B", prior_test_dir=tmp / "link" / "A", out_dir=tmp / "link" / "B"), sha)
    assert _sealed_runs(tmp)["B"]["out_dir"] == str(_out(tmp, "B").resolve())
    assert '"state": "pooled"' in (_out(tmp, "B") / "RUN_LOG.txt").read_text()


# --- Mn1: a rehearsal in, equal to or around the sealed folders
@pytest.mark.parametrize("where", ["a_default_out_dir", "inside_a_default_out_dir", "the_sealed_outputs_folder",
                                   "the_outputs_folder_above", "inside_docs_sealed", "docs_sealed_itself", "above_docs_sealed"])
def test_a_rehearsal_cannot_sit_in_equal_to_or_around_the_sealed_folders(registered, monkeypatch, where):
    """Mn1 (the reviewer's R8). A rehearsal at outputs/es/sealed/A/rehearsal made outputs/es/sealed/A and put its records in
    it; the registered Test A was then refused for finding the folder."""
    from es import sealed_run
    from es.sealed_run import run_sealed
    write, tmp = registered
    proto, reg, sha, kw = _bound(tmp, write, real_code=True)
    _no_read(monkeypatch)
    monkeypatch.setattr(sealed_run, "OUT_DIR", tmp / "outputs_es")
    real = tmp / "outputs_es" / "sealed" / "A"
    out = {"a_default_out_dir": real, "inside_a_default_out_dir": real / "rehearsal", "the_sealed_outputs_folder": real.parent,
           "the_outputs_folder_above": real.parent.parent, "inside_docs_sealed": tmp / "docs_sealed" / "x" / "y",
           "docs_sealed_itself": tmp / "docs_sealed", "above_docs_sealed": tmp}[where]
    with pytest.raises(ValueError, match="docs/sealed|sealed test"):
        run_sealed(_cfg_a(tmp, kw, out_dir=out), sha)
    assert not (tmp / "outputs_es").exists() and not [p for p in (tmp / "docs_sealed").rglob("*")] and not (tmp / "RUNS.json").exists()
    _stop_at_the_read(monkeypatch)                                       # a sibling is fine
    with pytest.raises(RuntimeError, match="stop"):
        run_sealed(_cfg_a(tmp, kw, out_dir=tmp / "outputs_es" / "rehearsal" / "A"), sha)


# --- Mn2: the interpreter pin compares binaries
def test_the_pinned_interpreter_is_compared_by_the_binary_it_names_not_by_its_spelling(registered, monkeypatch):
    """Mn2 (the reviewer's R9): /opt/anaconda3/bin/python, python3 and python3.12 are one binary and three strings. A pin that
    names the interpreter through a link to it matches (and the live path is recorded as it is); another binary does not."""
    from es.sealed_run import run_sealed
    write, tmp = registered
    link = tmp / "py_alias"
    link.symlink_to(sys.executable)
    pins, (proto, reg, sha, kw) = _bound_with_pins(tmp, write, executable=str(link))
    _stop_at_the_read(monkeypatch)
    with pytest.raises(RuntimeError, match="stop"):
        run_sealed(_cfg_a(tmp, kw), sha)
    env = _read_runs(tmp)["A"]["environment"]
    assert env["pins"] == "matched" and env["executable"] == sys.executable


def test_another_interpreter_that_exists_is_still_refused_by_the_pins(registered, monkeypatch):
    from es.registration import RegistrationError
    from es.sealed_run import run_sealed
    write, tmp = registered
    pins, (proto, reg, sha, kw) = _bound_with_pins(tmp, write, executable="/bin/sh")
    _no_read(monkeypatch)
    with pytest.raises(RegistrationError, match=r"executable: pinned '/bin/sh', running"):
        run_sealed(_cfg_a(tmp, kw), sha)
    _nothing_recorded(tmp)


# =====================================================================================================
# End to end. Two slow tests: a synthetic mini panel (this one runs in about a minute; it reaches every stage of
# run_sealed for Test A and Test B), and the brief's rehearsal on the development quarters (30-60 minutes on an idle
# machine: the controller runs it alone, with `-k rehearsal`).
# =====================================================================================================
def _synthetic_prices(start="2024-01-01", end="2025-10-31", seed=11, markets=tuple(NODE)):
    """Hourly prices for one node per market: a daily shape, a slowly moving volatility regime and fat-tailed real-time
    noise, so that spikes cluster and the models have something to learn. {market: frame in UTC}."""
    rng = np.random.default_rng(seed)
    hours = pd.date_range(start, f"{end} 23:00", freq="h", tz="UTC")
    day = ((hours - hours[0]) // pd.Timedelta(days=1)).to_numpy()
    out = {}
    for market in NODE:
        logv = np.zeros(day.max() + 1)
        for d in range(1, len(logv)):
            logv[d] = 0.9 * logv[d - 1] + 0.35 * rng.normal()
        da = 30 + 8 * np.sin(2 * np.pi * (hours.hour.to_numpy() - 6) / 24) + rng.normal(0, 2, len(hours))
        rt = da + 3 * np.exp(logv)[day] * rng.standard_t(4, len(hours))
        if market in markets:
            out[market] = pd.DataFrame({"market": market, "node": NODE[market], "node_type": "x", "timestamp_utc": hours,
                                        "da_lmp": da, "rt_lmp": rt})
    return out


def _write_prices(frame, path):
    """MISO files hold naive UTC, as the real ones do."""
    frame = frame.assign(timestamp_utc=frame["timestamp_utc"].dt.tz_localize(None)) if frame["market"].iloc[0] == "MISO" else frame
    frame.to_parquet(path, index=False)


def _synthetic_dev(dev, start="2024-01-01", end="2025-10-31", seed=11, markets=tuple(NODE)):
    for market, frame in _synthetic_prices(start, end, seed, markets).items():
        _write_prices(frame, dev / f"{market.lower()}_da_rt_syn.parquet")


def _synthetic_side_tables(root, prices):
    """The development settlement tables (CAISO's FMM, NYISO's integrated real-time LBMP) of the synthetic prices, before the
    seal, where a run looks for them (root/caiso_fmm_dev and root/nyiso_rtint_dev; Ruling 109, M9)."""
    from es.guard import first_sealed_utc
    for sub, name, market, column, shift, extra in (("caiso_fmm_dev", "caiso_fmm", "CAISO", "fmm_lmp", 0.5, {"n15": 4}),
                                                    ("nyiso_rtint_dev", "nyiso_rtint", "NYISO", "rt_int_lmp", -0.5, {})):
        f = prices[market]
        f = f[f["timestamp_utc"] < first_sealed_utc(market)]
        (root / sub).mkdir(exist_ok=True)
        pd.DataFrame({"node": NODE[market], "timestamp_utc": f["timestamp_utc"], column: f["rt_lmp"] + shift, **extra}
                     ).to_parquet(root / sub / f"{name}_syn.parquet", index=False)


def _light_models(monkeypatch):
    """Real score_analysis, real gates and scalers, a smaller XGBoost: the pipeline's code, not its accuracy, is on test."""
    from xgboost import XGBClassifier
    from es import scores

    def xgb(seed):
        def make(scale_pos_weight):
            return XGBClassifier(n_estimators=12, max_depth=2, learning_rate=0.3, subsample=0.8, colsample_bytree=0.8,
                                 scale_pos_weight=scale_pos_weight, eval_metric="aucpr", random_state=seed, n_jobs=1)
        return make

    monkeypatch.setitem(scores.FACTORIES, "xgb", xgb)


@pytest.mark.parametrize("case,min_nodes", [("too_few_nodes", None), ("missing_operator", 3), ("short_coverage", 4)])
def test_a_run_that_fails_the_coverage_rule_is_recorded_as_not_run(registered, monkeypatch, case, min_nodes):
    """Spec v3 §6.1: fewer than 28 usable nodes, a missing operator, or a node short of 90% of its deployment hours. The data
    were read, so the run is spent: it records NOT_RUN.md, the coverage table and results.json, and fits nothing."""
    from es.sealed_pull import manifest_sha256
    from es.sealed_run import run_sealed
    res, cfg, sha, tmp = _coverage_run(registered, monkeypatch, case, min_nodes)
    out = tmp / "out"
    assert res["coverage"]["ran"] is False
    assert res["status"] == "not run (coverage rule)" and res["errors"] == {} and "A" in _read_runs(tmp)
    assert (out / "NOT_RUN.md").exists() and (out / "tables" / "coverage.csv").exists()
    assert not (out / "scores").exists() and not (out / "kept").exists()
    assert json.loads((out / "results.json").read_text())["coverage"]["ran"] is False and "H-E" not in res
    log = (out / "RUN_LOG.txt").read_text().splitlines()
    assert log[0].startswith("started ") and log[-1].startswith("finished") and (out / "MANIFEST.sha256").exists()
    rec = _read_runs(tmp)["A"]                                        # the record is finished, with the manifest it names
    assert rec["finished_utc"] == res["finished_utc"] and rec["manifest_sha256"] == manifest_sha256(out)
    assert rec["status"] == "not run (coverage rule)" and res["rows_dropped"]["off_hour"] == 0
    dropped = res["rows_dropped"]                                     # M4: each kind of drop is counted and logged on its own
    assert dropped["beyond_window_end"] > 0 and dropped["node_map"] == 0 and dropped["duplicates"] == 0
    assert res["rows_read"] - res["rows_kept"] == sum(dropped.values()) and res["coverage"]["off_hour_dropped"] == {}
    assert {f"rows_read {res['rows_read']}", f"rows_kept {res['rows_kept']}", "node_map_dropped 0", "duplicates_dropped 0",
            "dev_duplicates_dropped 0", f"beyond_window_end {dropped['beyond_window_end']}", "off_hour_dropped {}",
            "nonfinite_dropped {}"} <= set(log)
    cov = pd.read_csv(out / "tables" / "coverage.csv").set_index("market")
    if case == "too_few_nodes":
        assert res["coverage"]["usable_nodes"] == 4 and cov["ok"].all()
    elif case == "missing_operator":
        assert res["coverage"]["usable_nodes"] == 3 and "CAISO" not in cov.index
    else:
        assert res["coverage"]["usable_nodes"] == 3 and not cov.loc["NYISO", "ok"] and cov.loc["NYISO", "share"] < 0.9
    with pytest.raises(FileExistsError, match="runs once"):          # a run that stopped on coverage was still the run
        run_sealed(cfg, sha, n_rep=5)


@pytest.mark.slow
def test_synthetic_mini_run_reaches_every_stage_of_test_a_and_test_b(registered, monkeypatch):
    from es import sealed_run
    from es.analyses import Analysis
    from es.cutoffs import cutoff_table
    from es.data import build_panel, load_dev_raw
    from es.registration import sha256_file
    from es.scores import CACHED_COLUMNS
    from es.sealed_run import SealedConfig, run_sealed, score_keys
    from es.windows import DateWindow as D
    write, tmp = registered
    dev = tmp / "dev"
    dev.mkdir()
    _synthetic_dev(dev)
    _synthetic_side_tables(tmp, _synthetic_prices())
    _light_models(monkeypatch)
    monkeypatch.setattr(sealed_run, "MIN_USABLE_NODES", 4)                 # four nodes stand in for the 33
    # Training spans the deployment seasons of the year before (May-June, September-October), as the real windows do:
    # P2's in-season cutoff needs training rows in those months. Calibration windows of two months keep the PC
    # bootstrap (which raises on a NaN resample, see the report) away from replicates that draw none of a frame's days.
    a_train, b_train = D("2024-01-08", "2025-02-28"), D("2024-01-08", "2025-06-30")
    a_cal, b_cal, a_dep, b_dep = (D("2025-03-01", "2025-04-30"), D("2025-07-01", "2025-08-31"),
                                  D("2025-05-01", "2025-06-30"), D("2025-09-01", "2025-10-30"))   # 61 and 60 days
    f_train = D("2024-01-08", "2024-12-31")
    arms = {"SM1": Analysis("SM1", a_train, a_cal, a_dep, a_train, "rolling", "analog"),
            "SM1F": Analysis("SM1F", f_train, a_cal, a_dep, f_train, "frozen", "analog"),
            "SM2": Analysis("SM2", b_train, b_cal, b_dep, b_train, "rolling", "analog"),
            "SM2F": Analysis("SM2F", f_train, b_cal, b_dep, f_train, "frozen", "analog")}
    pre = build_panel(load_dev_raw((dev,)))
    cut, inp, man = tmp / "cutoffs.csv", tmp / "inputs.json", tmp / "CODE_MANIFEST.sha256"
    cutoff_table(pre[pre["settle_date"] <= dt.date(2025, 8, 31)], list(arms.values())).to_csv(cut, index=False)
    inp.write_text(json.dumps({"eta": 0.01, "family": FAMILY}))
    man.write_text("".join(f"{sha256_file(DEV_ROOT / rel)}  {rel}\n" for rel in ("es/cutoffs.py", "es/sealed_run.py")))
    proto, reg, sha = write(_protocol_lines(cut, inp, man))

    def cfg(test_id, rolling, frozen, end, prior):
        return SealedConfig(test_id=test_id, rolling=arms[rolling], frozen=arms[frozen], sealed_dirs=[], window_end=end,
                            prior_test_dir=prior, registration=(proto, reg), out_dir=tmp / test_id, data_root=tmp,
                            cutoffs=cut, inputs=inp, code_manifest=man, dev_dirs=(dev,))

    res_a = run_sealed(cfg("A", "SM1", "SM1F", "2025-06-30", None), sha, n_rep=20)
    assert res_a["coverage"]["ran"] and res_a["coverage"]["usable_nodes"] == 4 and res_a["departures"] == []
    assert res_a["status"] == "completed" and res_a["errors"] == {} and "A" in _read_runs(tmp)       # Ruling 104
    from es.sealed_pull import manifest_sha256, verify_manifest
    rec_a = _read_runs(tmp)["A"]                                           # Ruling 109 I3: the finished record
    assert rec_a["finished_utc"] == res_a["finished_utc"] and rec_a["manifest_sha256"] == manifest_sha256(tmp / "A")
    assert rec_a["status"] == "completed" and rec_a["environment"] == res_a["environment"]
    assert (res_a["registration"], res_a["registration_kind"], rec_a["registration_kind"]) == (
        "10.17605/OSF.IO/ABCDE", "osf", "osf")                              # Ruling 140: the kind beside the identifier
    assert (tmp / "RUN_A.claim").exists() and not list((tmp / "docs_sealed").iterdir())      # a rehearsal keeps its own records
    verify_manifest(tmp / "A")
    assert res_a["environment"]["pins"] == "not bound" and res_a["E5_scope"] == "Test A window"
    assert all(res_a[h]["interval_registered"] == "basic" and "p" in res_a[h] for h in ("H-B1", "H-B2"))     # FAMILY: tests
    assert res_a["rows_dropped"]["off_hour"] == 0 and res_a["rows_dropped"]["duplicates"] == 0
    assert res_a["secondary"]["drift monitors"]["skipped_nodes"] == []
    assert res_a["H-B1"]["n_nan_replicates"] == 0 and res_a["H-S"]["xgb"]["n_nan_replicates"] == 0
    for est in ("xgb", "logit"):                         # Ruling 118, M2: H-S and E5 on the aligned draw, the old interval beside it
        hs = res_a["H-S"][est]
        assert hs["scheme"] == "aligned" and hs["approximate"] is True and len(hs["ci_independent"]) == 2 and "error" not in hs
    assert all(row["scheme"] == "aligned" and row["approximate"] is True and "lo_independent" in row for row in res_a["E5"])
    assert "scheme" not in res_a["H-B1"] and "scheme" not in res_a["PC"]["P2_all"]       # H-B and PC: the registered draws
    assert res_a["H-E"]["status"].startswith("interim") and res_a["estimands_scope"] == "single window (interim)"
    assert {"H-B1", "H-B2", "PC", "H-S", "estimands", "E5", "tau_check", "secondary"} <= set(res_a)
    assert res_a["tau_check"]["max_abs_diff"] == 0.0 and res_a["tau_check"]["nan_pattern_mismatches"] == 0
    assert res_a["tau_check"]["registered_only"] == 0 and res_a["tau_check"]["recomputed_only"] == 0
    assert res_a["tau_check"]["n_compared"] == 2 * 4 * 4                    # two arms, four blocks, four nodes
    assert set(res_a["secondary"]) == {"audit robustness", "economic robustness", "drift monitors"}
    assert res_a["secondary"]["audit robustness"]["tests"] == 18
    assert {"E4:dSELC|calendar@count", "E4:dSELC|VOL@count"} <= set(res_a["estimands"])      # Task 12 (Ruling 81)
    assert set(res_a["coverage"]["comparator_rows_dropped"]) >= {"P1", "P4", "P4@r*", "calendar@count", "VOL@count"}
    assert res_a["coverage"]["nonfinite_dropped"] == {}
    assert {"t_obs", "p", "n_null", "selc_obs", "retained_share"} <= set(res_a["H-E"])
    assert res_a["H-E"]["n_null"] == 61 - 1 and len(res_a["H-E"]["retained_share"]) == 1
    assert all(len(res_a[h]["ci_basic"]) == 2 and res_a[h]["status"] for h in ("H-B1", "H-B2"))
    for h in ("H-B1", "H-B2"):                                              # the status follows the registered threshold
        assert res_a[h]["status"].startswith("rejected in Test A at 0.05/3") == (res_a[h]["p"] <= 0.05 / 3)
    assert [e["block"] for e in res_a["E5"]] == list(__import__("es.blocks", fromlist=["BLOCKS"]).BLOCKS)
    assert set(res_a["PC"]) == {"P2_season", "P2_all"} and set(res_a["H-S"]) == {"xgb", "logit"}
    assert all(res_a["PC"][k]["ci_approximate"] for k in res_a["PC"])
    assert len(list((tmp / "A" / "scores").glob("*.parquet"))) == 34 == len(score_keys())
    slim = pd.read_parquet(tmp / "A" / "scores" / "roll__LEV6__xgb__s42.parquet")
    assert list(slim.columns) == CACHED_COLUMNS and slim["market"].dtype == object
    log = (tmp / "A" / "RUN_LOG.txt").read_text()
    assert log.startswith("started ") and "comparator_rows_dropped" in log and log.rstrip().splitlines()[-1].startswith("finished")
    assert (tmp / "A" / "MANIFEST.sha256").exists() and (tmp / "A" / "tables" / "secondary_bh.csv").exists()
    assert sorted(p.name for p in (tmp / "A" / "kept").glob("*.parquet")) == sorted(
        ["P1.parquet", "P4.parquet", "P4_at_rstar.parquet", "calendar_at_count.parquet", "VOL_at_count.parquet"])
    full, ex = (pd.read_csv(tmp / "A" / "tables" / f).set_index("stat") for f in ("estimands.csv", "estimands_ex_caiso.csv"))
    assert list(full.index) == list(ex.index) and not np.allclose(full.loc["NET", "estimate"], ex.loc["NET", "estimate"])
    # Test B with two of its secondary analyses broken: the run goes on, says so, and keeps every other result
    def broken(*_a, **_k):
        raise RuntimeError("injected")

    monkeypatch.setattr(sealed_run, "audit_family", broken)
    monkeypatch.setattr(sealed_run, "monitor_rows", broken)
    res_b = run_sealed(cfg("B", "SM2", "SM2F", "2025-10-30", tmp / "A"), sha, n_rep=20)
    assert res_b["status"] == "completed with secondary errors" and set(_read_runs(tmp)) == {"A", "B"}
    assert res_b["errors"] == {"audit robustness": "RuntimeError: injected", "drift monitors": "RuntimeError: injected"}
    assert res_b["secondary"]["audit robustness"] == {"error": "RuntimeError: injected"}
    assert res_b["secondary"]["drift monitors"] == {"error": "RuntimeError: injected"}
    assert res_b["secondary"]["economic robustness"]["tests"] == 23                  # adjusted without the two that failed
    assert list(pd.read_csv(tmp / "B" / "tables" / "secondary_bh.csv")["family"].unique()) == ["economic robustness"]
    log_b = (tmp / "B" / "RUN_LOG.txt").read_text()
    assert "error audit robustness: RuntimeError: injected" in log_b and "Traceback (most recent call last)" in log_b
    assert log_b.rstrip().splitlines()[-1].startswith("finished") and "status completed with secondary errors" in log_b
    assert res_b["H-E"]["status"].startswith("decided") and res_b["H-E"]["n_null"] == 61 * 60 - 1   # both windows' days
    assert res_b["test_a"]["state"] == "pooled" and res_b["test_a"]["manifest_sha256"] == rec_a["manifest_sha256"]
    assert res_b["E5_scope"] == "Test B window" and res_b["estimands_scope"] == "pooled A+B"
    assert res_b["H-E"]["approximate"] is FAMILY["H-E"]["approximate"]               # Ruling 104(b)
    assert len(res_b["H-E"]["retained_share"]) == 2 and "wording" in res_b["H-E"]
    assert set(res_b["holm"]) == {"H-E", "H-B1", "H-B2"}
    pooled, quarter = (pd.read_csv(tmp / "B" / "tables" / f).set_index("stat") for f in
                       ("estimands.csv", "estimands_this_quarter.csv"))
    assert (pooled["n_windows"] == 2).all() and (quarter["n_windows"] == 1).all()
    assert not np.allclose(pooled.loc["NET", "estimate"], quarter.loc["NET", "estimate"])    # pooled A+B is not B alone
    from es.run_inference import holm
    ps = [res_b["H-E"]["p"], res_a["H-B1"]["p"], res_a["H-B2"]["p"]]               # Holm on Test A's H-B p-values
    assert list(res_b["holm"].values()) == [bool(x) for x in holm(ps, 0.05)]
    assert res_b["estimands_scope"] == "pooled A+B" and "E1:J_0.5:timing" in res_b["estimands"]
    assert all("replicated" in res_b[h] for h in ("H-B1", "H-B2"))
    assert (tmp / "B" / "tables" / "estimands_this_quarter.csv").exists()
    assert json.loads((tmp / "B" / "results.json").read_text())["H-E"]["p"] == res_b["H-E"]["p"]
    shutil.rmtree(tmp / "B")                                                   # deleting the folder re-arms nothing
    with pytest.raises(FileExistsError, match="runs once"):
        run_sealed(cfg("B", "SM2", "SM2F", "2025-10-30", tmp / "A"), sha, n_rep=20)
    assert not (tmp / "B").exists()


@pytest.mark.slow
def test_synthetic_run_reads_a_sealed_pull_through_every_gate(registered, monkeypatch):
    """Test A as it will run: the registered analyses SA and SAF, one sealed pull made by pull_sealed (fake builders, as in
    the loader test) bound to the registration through PULLS.json, development files that stop at the seal, and the pull's
    FMM and integrated NYISO tables reaching economics_for. Synthetic prices; four nodes stand in for the 33."""
    from es import sealed_run
    from es.analyses import SEALED
    from es.cutoffs import cutoff_table
    from es.data import build_panel, load_dev_raw
    from es.guard import first_sealed_utc
    from es.registration import sha256_file
    from es.sealed_pull import manifest_sha256, pull_sealed, verify_manifest
    from es.sealed_run import default_config, run_sealed
    write, tmp = registered
    dev = tmp / "dev"
    dev.mkdir()
    prices = _synthetic_prices("2023-01-01", "2026-10-01")
    for market, frame in prices.items():                      # the development files stop at the seal, as the real ones do
        _write_prices(frame[frame["timestamp_utc"] < first_sealed_utc(market)], dev / f"{market.lower()}_da_rt_syn.parquet")
    _synthetic_side_tables(tmp, prices)
    _light_models(monkeypatch)
    monkeypatch.setattr(sealed_run, "MIN_USABLE_NODES", 4)
    monkeypatch.setattr(sealed_run, "N_REP", 20)                # the registered 2000, lowered for a test (Ruling 109, C2)
    monkeypatch.setattr(sealed_run, "DEV_DATA_DIRS", (dev,))    # and the registered development folders (Ruling 116, N4)
    pre = build_panel(load_dev_raw((dev,)))
    cut, inp, man = tmp / "cutoffs.csv", tmp / "inputs.json", tmp / "CODE_MANIFEST.sha256"
    cutoff_table(pre[pre["settle_date"] <= dt.date(2025, 9, 30)], [SEALED["SA"], SEALED["SAF"]]).to_csv(cut, index=False)
    inp.write_text(json.dumps({"eta": 0.01, "family": FAMILY}))
    man.write_text("".join(f"{sha256_file(DEV_ROOT / rel)}  {rel}\n" for rel in ("es/cutoffs.py", "es/sealed_run.py")))
    proto, reg, sha = write(_protocol_lines(cut, inp, man))

    def price(market):
        return lambda start, end, root: _write_prices(prices[market], root / f"{market.lower()}_da_rt_syn.parquet")

    def side(sub, name, market, column, shift, extra):
        def build(start, end, root):                       # the file names the settlement loaders glob for
            (root / sub).mkdir()
            f = prices[market]
            pd.DataFrame({"node": NODE[market], "timestamp_utc": f["timestamp_utc"], column: f["rt_lmp"] + shift, **extra}
                         ).to_parquet(root / sub / f"{name}_syn.parquet", index=False)
        return build

    builders = {m: price(m) for m in NODE}
    builders.update({"FMM": side("fmm", "caiso_fmm", "CAISO", "fmm_lmp", 0.5, {"n15": 4}),
                     "NYISO_INT": side("nyiso_rtint", "nyiso_rtint", "NYISO", "rt_int_lmp", -0.5, {})})
    root, manifest = pull_sealed("A", sha, data_root=tmp, builders=builders, reference_dirs=[], check_days=False)
    cfg = default_config("A")                                   # the registered arms, pull and window, on temporary files
    cfg.data_root, cfg.cutoffs, cfg.inputs, cfg.code_manifest, cfg.dev_dirs = tmp, cut, inp, man, (dev,)
    out = _out(tmp)                                             # out_dir stays the registered folder (Ruling 122, N2b)
    assert cfg.rolling is SEALED["SA"] and cfg.sealed_dirs == ["sealed_2026Q3"] and cfg.registration == (proto, reg)
    res = run_sealed(cfg, sha, n_rep=20)
    assert res["coverage"]["ran"] and res["coverage"]["usable_nodes"] == 4 and res["departures"] == [] and res["n_rep"] == 20
    assert res["status"] == "completed" and res["errors"] == {} and _sealed_runs(tmp)["A"]["out_dir"] == str(out.resolve())
    rec = _sealed_runs(tmp)["A"]                                # the record and the claim are in docs/sealed, and finished
    assert rec["finished_utc"] == res["finished_utc"] and rec["manifest_sha256"] == manifest_sha256(out)
    assert rec["environment"] == res["environment"] and (tmp / "docs_sealed" / "RUN_A.claim").exists()
    assert not (tmp / "RUNS.json").exists() and res["E5_scope"] == "Test A window"
    assert res["rows_dropped"]["duplicates"] > 0 and res["rows_read"] - res["rows_kept"] == sum(res["rows_dropped"].values())
    log = (out / "RUN_LOG.txt").read_text().splitlines()        # M4: the drops are logged one by one (the overlap days show)
    assert f"duplicates_dropped {res['rows_dropped']['duplicates']}" in log and "node_map_dropped 0" in log
    assert "beyond_window_end 0" in log and "off_hour_dropped {}" in log and "dev_duplicates_dropped 0" in log
    with pytest.raises(FileExistsError, match="runs once"):     # the one run is made, whatever its out_dir
        run_sealed(cfg, sha, n_rep=20)
    assert manifest == manifest_sha256(root) and res["sealed_manifests"] == {"sealed_2026Q3": manifest}
    assert f"sealed_manifest sealed_2026Q3 {manifest}" in (out / "RUN_LOG.txt").read_text()
    assert res["rows_read"] > res["rows_kept"] > 0                      # the pull repeats the last ten days of the development files
    assert res["tau_check"]["max_abs_diff"] == 0.0 and res["tau_check"]["n_compared"] == 2 * 4 * 4
    assert res["tau_check"]["registered_only"] == 0 and res["tau_check"]["recomputed_only"] == 0
    econ = pd.read_csv(out / "tables" / "economics.csv")
    assert (econ["settle_approx_share"] == 0).all()                     # CAISO's FMM and NYISO's integrated hours came from the pull
    cov = pd.read_csv(out / "tables" / "coverage.csv")
    assert (cov["hours"] == cov["expected_hours"]).all() and (cov["expected_hours"] == 2208).all()
    assert res["H-E"]["status"].startswith("interim") and res["H-E"]["n_null"] == 92 - 1
    assert "E4:dSELC|calendar@count" in res["estimands"] and "E4:dSELC|VOL@count" in res["estimands"]
    (out / "extra.txt").write_text("a file added to the finished run")
    with pytest.raises(RuntimeError, match="not in the manifest"):      # the finished run is sealed too
        verify_manifest(out)


def _chosen_with_k(request, word):
    """True when this run was started with `-k` naming `word`. The rehearsal asks to be chosen on purpose: a plain
    `pytest tests/es` (or `-m slow`) would otherwise run it, 30-60 minutes and several GB, beside other jobs."""
    return word in (request.config.getoption("keyword") or "")


def test_the_rehearsal_runs_only_when_asked_for_by_name():
    class Config:
        def __init__(self, keyword):
            self.keyword = keyword

        def getoption(self, name):
            assert name == "keyword"
            return self.keyword

    class Request:
        def __init__(self, keyword):
            self.config = Config(keyword)

    assert _chosen_with_k(Request("rehearsal"), "rehearsal") and _chosen_with_k(Request("rehearsal or x"), "rehearsal")
    assert not _chosen_with_k(Request(""), "rehearsal") and not _chosen_with_k(Request(None), "rehearsal")
    assert not _chosen_with_k(Request("synthetic"), "rehearsal")


@pytest.mark.slow
def test_rehearsal_on_development_quarters(registered, request):
    """End-to-end rehearsal under a temporary registration (select it with -k rehearsal; it skips itself otherwise):
    - AQ1 (deploy 2026-Q1) plays Test A and AQ2 (deploy 2026-Q2) plays Test B, on opened data;
    - the cutoff table is built on data up to 2025-12-31, i.e. before either pseudo-window;
    - the pooled H-E test, the pooled estimands and the Test B decision logic run on real score frames.
    Not exercised here (covered by the unit tests above): the sealed-dir branch of the private loader, the
    dev_only=False loaders, NOT_RUN, a frozen arm that differs from the rolling one, departures, and the
    non-confirmatory branches of the family rule."""
    if not _chosen_with_k(request, "rehearsal"):
        pytest.skip("the rehearsal takes 30-60 minutes and several GB and runs alone: select it with -k rehearsal")
    from es.analyses import ANALOGS
    from es.cutoffs import cutoff_table
    from es.data import build_panel, load_dev_raw
    from es.registration import sha256_file
    from es.scores import CACHED_COLUMNS
    from es.sealed_run import SealedConfig, run_sealed
    write, tmp = registered
    pre = build_panel(load_dev_raw())
    cut, inp, man = tmp / "cutoffs.csv", tmp / "inputs.json", tmp / "CODE_MANIFEST.sha256"
    cutoff_table(pre[pre["settle_date"] <= dt.date(2025, 12, 31)], [ANALOGS["AQ1"], ANALOGS["AQ2"]]).to_csv(
        cut, index=False)
    inp.write_text(json.dumps({"eta": 0.01, "family": FAMILY}))
    man.write_text("".join(f"{sha256_file(DEV_ROOT / rel)}  {rel}\n" for rel in ("es/cutoffs.py", "es/sealed_run.py")))
    proto, reg, sha = write(_protocol_lines(cut, inp, man))

    def cfg(test_id, analysis, end, prior):
        return SealedConfig(test_id=test_id, rolling=analysis, frozen=analysis, sealed_dirs=[], window_end=end,
                            prior_test_dir=prior, registration=(proto, reg), out_dir=tmp / test_id,
                            data_root=None, cutoffs=cut, inputs=inp, code_manifest=man)

    res_a = run_sealed(cfg("A", ANALOGS["AQ1"], "2026-03-31", None), sha, n_rep=50)
    assert res_a["coverage"]["ran"] and res_a["H-E"]["status"].startswith("interim")
    assert (tmp / "RUNS.json").exists() and (tmp / "RUN_A.claim").exists() and not list((tmp / "docs_sealed").iterdir())
    assert res_a["E5_scope"] == "Test A window" and res_a["environment"]["pins"] == "not bound"
    assert {"H-B1", "H-B2", "PC", "H-S", "estimands", "E5", "tau_check", "secondary"} <= set(res_a)
    assert res_a["departures"] == [] and (tmp / "A" / "RUN_LOG.txt").read_text().startswith("started ")
    assert res_a["errors"] == {} and res_a["status"] == "completed"           # a guarded failure would still be a finding
    assert res_a["secondary"]["drift monitors"]["skipped_nodes"] == []
    assert set(res_a["secondary"]) == {"audit robustness", "economic robustness", "drift monitors"}
    assert res_a["secondary"]["audit robustness"]["tests"] == 18
    assert res_a["estimands_scope"] == "single window (interim)"
    assert res_a["tau_check"]["max_abs_diff"] == 0.0 and res_a["tau_check"]["nan_pattern_mismatches"] == 0
    assert all(res_a[h]["status"] and len(res_a[h]["ci_basic"]) == 2 for h in ("H-B1", "H-B2"))
    assert {"t_obs", "p", "n_null", "selc_obs", "retained_share"} <= set(res_a["H-E"])
    assert len(res_a["H-E"]["retained_share"]) == 1 and 0 < res_a["H-E"]["retained_share"][0]["mean"] <= 1
    assert len(list((tmp / "A" / "scores").glob("*.parquet"))) == 34
    slim = pd.read_parquet(tmp / "A" / "scores" / "roll__LEV6__xgb__s42.parquet")
    assert list(slim.columns) == CACHED_COLUMNS and slim["market"].dtype == object
    res_b = run_sealed(cfg("B", ANALOGS["AQ2"], "2026-06-30", tmp / "A"), sha, n_rep=50)
    assert res_b["H-E"]["status"].startswith("decided") and res_b["H-E"]["n_null"] > 1000
    assert res_b["test_a"]["state"] == "pooled" and res_b["E5_scope"] == "Test B window"
    assert json.loads((tmp / "RUNS.json").read_text())["A"]["manifest_sha256"] == res_b["test_a"]["manifest_sha256"]
    assert res_b["errors"] == {} and res_b["status"] == "completed" and "approximate" in res_b["H-E"]
    assert len(res_b["H-E"]["retained_share"]) == 2
    assert set(res_b["holm"]) == {"H-E", "H-B1", "H-B2"} and "wording" in res_b["H-E"]
    assert res_b["estimands_scope"] == "pooled A+B" and "E1:J_0.5:timing" in res_b["estimands"]
    assert all("replicated" in res_b[h] for h in ("H-B1", "H-B2"))
    assert (tmp / "B" / "tables" / "estimands_this_quarter.csv").exists()
    with pytest.raises(FileExistsError):
        run_sealed(cfg("B", ANALOGS["AQ2"], "2026-06-30", tmp / "A"), sha, n_rep=50)
    assert json.loads((tmp / "B" / "results.json").read_text())["H-E"]["p"] == res_b["H-E"]["p"]


# =====================================================================================================
# Ruling 140 (26 Oct 2026): a sealed pull is bound to an OSF registration, a public release or a local record through the
# registration id of PULLS.json (osf_id is renamed registration_id), and the run records the kind beside the identifier.
# =====================================================================================================
RELEASE_URL = "https://github.com/ee2625/dart-spread/releases/tag/v1.0"
LOCAL_TOKEN = "author-statement-2026-10-26"
KINDS = {"osf": {}, "public_release": {"osf_doi": None, "public_record": RELEASE_URL},
         "local": {"osf_doi": None, "local_record": LOCAL_TOKEN}}
#: (registration id, the identifier to log) of each kind.
FACTS = {"osf": ("abcde", "10.17605/OSF.IO/ABCDE"),
         "public_release": ("github:ee2625/dart-spread/releases/tag/v1.0", RELEASE_URL),
         "local": ("local:author-statement-2026-10-26", LOCAL_TOKEN)}
PAIRS = [(a, b) for a in KINDS for b in KINDS if a != b]


@pytest.mark.parametrize("first, second", PAIRS)
def test_check_sealed_dirs_binds_a_pull_to_the_kind_of_registration_it_was_made_under(registered, first, second):
    """A pull made under one record is read under that record and under no other, as it was for OSF ids: the same protocol
    recorded as a public release, a local record or an OSF registration is another registration."""
    from es.registration import RegistrationError
    from es.sealed_run import check_sealed_dirs
    write, tmp = registered
    proto, reg, sha = write(reg_fields=KINDS[first])
    dev, root = _fake_pull(tmp, proto, reg, sha)
    entry = json.loads((tmp / "docs_sealed" / "PULLS.json").read_text())["A"]
    assert (entry["registration_id"], entry["registration_kind"]) == (FACTS[first][0], first) and "osf_id" not in entry
    assert check_sealed_dirs(_cfg_pull(tmp, proto, reg, dev), _record_of(sha, proto, reg))        # its own record
    proto, reg, sha2 = write(reg_fields=KINDS[second])
    assert sha2 == sha
    with pytest.raises(RegistrationError, match=r"pulled under a different registration \(registration id "
                                                 + re.escape(FACTS[first][0].casefold()) + r"\)"):
        check_sealed_dirs(_cfg_pull(tmp, proto, reg, dev), _record_of(sha, proto, reg))


@pytest.mark.parametrize("kind", ["public_release", "local"])
def test_the_pull_s_log_cannot_name_another_registration_than_its_entry_for_the_new_kinds_either(registered, kind):
    from es.registration import RegistrationError
    from es.sealed_run import _logged_registration_ids, check_sealed_dirs
    write, tmp = registered
    proto, reg, sha = write(reg_fields=KINDS[kind])
    dev, root = _fake_pull(tmp, proto, reg, sha)
    assert _logged_registration_ids(root) == {FACTS[kind][0].casefold()}
    other = "local" if kind == "public_release" else "public_release"
    proto, reg, sha = write(reg_fields=KINDS[other])                    # the record now names the other kind
    _edit_pulls(tmp, registration_id=FACTS[other][0])                    # and the entry was edited to match it; the log not
    with pytest.raises(RegistrationError, match=r"PULL_LOG.txt names a different registration \(registration id "
                                                 + re.escape(FACTS[kind][0].casefold()) + r"\)"):
        check_sealed_dirs(_cfg_pull(tmp, proto, reg, dev), _record_of(sha, proto, reg))


@pytest.mark.parametrize("kind", list(KINDS))
def test_an_entry_without_a_registration_id_is_not_a_pull_record_whatever_the_kind(registered, kind):
    """The old key, osf_id, is not read: an entry that carries only that is not the record of a pull_sealed pull."""
    from es.registration import RegistrationError
    from es.sealed_run import check_sealed_dirs
    write, tmp = registered
    proto, reg, sha = write(reg_fields=KINDS[kind])
    dev, root = _fake_pull(tmp, proto, reg, sha)
    _edit_pulls(tmp, registration_id=None, osf_id=FACTS[kind][0])
    with pytest.raises(RegistrationError, match="names no registration_id"):
        check_sealed_dirs(_cfg_pull(tmp, proto, reg, dev), _record_of(sha, proto, reg))


@pytest.mark.parametrize("first, second", PAIRS)
def test_the_loader_reads_a_pull_only_under_the_registration_it_was_made_under(registered, monkeypatch, first, second):
    """The loader (private, behind a live claim) checks the record again: a pull made under one kind of record is read under
    that record, and under another record of the same protocol it is refused before a price row is opened."""
    from es.registration import RegistrationError
    from es.sealed_run import _load_sealed_raw
    write, tmp = registered
    proto, reg, sha = write(reg_fields=KINDS[first])
    dev, root = _fake_pull(tmp, proto, reg, sha)
    cfg = _cfg_pull(tmp, proto, reg, dev)
    claim = _claimed(cfg, sha)
    assert _load_sealed_raw(cfg, sha, claim).attrs["rows_read"] == 16                    # its own record: the rows are read
    write(reg_fields=KINDS[second])
    opened = _spy_parquet(monkeypatch)
    with pytest.raises(RegistrationError, match="pulled under a different registration"):
        _load_sealed_raw(cfg, sha, claim)
    assert opened == []                                                                   # and no price file was opened


@pytest.mark.parametrize("kind", list(KINDS))
def test_the_run_records_the_kind_and_the_identifier_of_its_registration(registered, monkeypatch, kind):
    from es import sealed_run
    from es.sealed_run import run_sealed
    write, tmp = registered
    proto, reg, sha, kw = _bound(tmp, lambda text, fields=None: write(text, fields or KINDS[kind]))
    seen = {}

    def about_to_read(*_a, **_k):
        seen["log"] = (tmp / "out" / "RUN_LOG.txt").read_text()
        seen["runs"] = _read_runs(tmp)
        raise RuntimeError("stop")

    monkeypatch.setattr(sealed_run, "_load_sealed_raw", about_to_read)
    with pytest.raises(RuntimeError, match="stop"):
        run_sealed(_cfg_a(tmp, kw), sha)
    rec, identifier = seen["runs"]["A"], FACTS[kind][1]
    assert (rec["registration_identifier"], rec["registration_kind"]) == (identifier, kind)
    assert f"registration {identifier} registered_utc 2026-10-20T15:04:00Z" in seen["log"]
    assert f"\nregistration_kind {kind}\n" in seen["log"]


def test_the_run_reads_and_writes_no_osf_id_key():
    """Ruling 140: the key is registration_id in PULLS.json, .state.json and PULL_LOG.txt, and the helper that reads the
    log is _logged_registration_ids. A grep over the pull and the run."""
    for name in ("sealed_pull.py", "sealed_run.py"):
        assert "osf_id" not in (DEV_ROOT / "es" / name).read_text(), name
