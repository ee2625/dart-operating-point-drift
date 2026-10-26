"""Downloads for the sealed windows (spec v3 §6.1 and §6.5 steps 6 and 8). Runs once per window,
only after verify_registration. NYISO, ISO-NE and MISO use the July downloaders unchanged
(v2/ingest_*.build) with a new, explicit root; CAISO uses exact local-day UTC bounds; FMM and NYISO's
integrated real-time LBMP reuse es.settlement's parsers without its development-date check. Each
pull starts 10 days before deployment for the 216 h feature lookback; the pre-deployment rows are
checked against the files already opened; every price file is trimmed to the pull's own days.

The gate. pull_sealed reads the registration record from es.registration's own paths when it is called, never
from a record the caller hands it (fix round 1, C1). The pullers refuse dates after SEAL_LAST_DEV_DATE, and the
readers never load a row at or after the seal, unless pull_sealed passes the private _sealed=True.

Once only. A completed pull is recorded in docs/sealed/PULLS.json (test id, root, manifest SHA-256, times, protocol
and registration identifiers: the DOI and the OSF address, the release address or the local token, whichever the record
gives, with the kind of the registration and the registration id to compare), outside
any data root, and pull_sealed checks that record first, whatever data_root it is given. An exclusive lock
(docs/sealed/PULL_<test>.lock) is held for the whole pull. A resume continues the first attempt: the protocol and
registration it started under, and a fingerprint of the code, are kept in .state.json and PULL_LOG.txt, and a resume
must match both. A root that has files is resumed only from its .state.json; with the state missing or unreadable it is
refused, never started over (move the folder aside to start again). Test B needs Test A's completed pull, found in
PULLS.json, with a manifest that still verifies.

Nothing lost silently. Cache, state, manifest and trimmed-table writes go through a temp file and os.replace (and
fsync, with F_FULLFSYNC on macOS, then the directory). A response is cached only if it covers its whole interval grid
and lies inside its bounds, and its prices are finite numbers. Each step is trimmed to the pull's days and checked,
before it is recorded as done: every node-map node must have, on every local settlement day, as many rows on the hour
(minute and second 0 in UTC) with finite prices (for the FMM table, rows of four 15-minute intervals) as the day has
clock hours, and no more rows than that, for the price files and for the FMM and NYISO integrated tables, unless
allow_missing_days names the day. A row off the hour is not an hour; a day with more rows than hours (an hour twice)
fails like a short day (Ruling 116). A day has 24 hours, 23 or 25 when the clocks change (MISO keeps fixed EST); the
July NYISO and ISO-NE downloaders and the NYISO integrated parser drop the two rows of the repeated hour, so they keep
23 of the 25. COVERAGE.csv counts the hours on the hour and the non-finite rows of each series and node.
Each builder's stdout and stderr go to <root>/logs/<step>.log, part of the hashed record. The manifest lists every file
of the pull but itself and the transient files; verify_manifest compares the exact file set; manifest_sha256 gives the
manifest's own hash.

Chunks that never complete (Ruling 111a). A response that arrives whole but fails the validator on all 8 attempts stops
the pull with a ChunkError, and PULL_LOG.txt names the chunk (a chunk_failed line). An attempt with resume=True may then
name it in allow_incomplete_chunks, {series: [chunk key, ...]} with the keys as the log gives them: the whole
response with the fewest problems is accepted, short or with non-finite prices, and is never cached; every missing or
non-finite interval goes to
PULL_LOG.txt, and the chunk and the reason to .state.json and PULLS.json. A chunk that has not failed all 8 attempts in
an earlier attempt is requested and validated as usual. The day check still reports the days that the accepted chunk
leaves short, and those need allow_missing_days: two explicit, logged decisions. An answer that arrives whole but
holds no data (OASIS's no-data report, a CSV without rows, a zip without a CSV) counts with the refused ones, and a
chunk of such answers is accepted as zero rows; a network failure (a timeout, a connection error, an HTTP error
status) and an answer that cannot be read at all do not count (Ruling 121, F1). A chunk is accepted only if every
attempt of the accepting call was a whole answer that was refused, and the answer kept is the one with the fewest
problems (F2)."""
from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import hashlib
import inspect
import io
import json
import os
import socket
import time
import traceback
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from v2 import ingest_isone, ingest_miso, ingest_nyiso
from v2.run_decomposition import HERE as V2_DIR

from . import registration as _registration
from .config import DATA_ROOT, MARKETS, SEAL_LAST_DEV_DATE, SETTLE_TZ
from .data import _market_of, raw_files
from .guard import SealedDataError, first_sealed_utc, settlement_frame
from .registration import (registration_id, registration_identifier, registration_identifiers, registration_ids,
                           registration_kind, sha256_file, verify_registration)
from .settlement import NYISO_URL, _fetch, _parse_caiso, _parse_nyiso

SEALED_WINDOWS = {"A": ("2026-06-21", "2026-07-01", "2026-09-30", "sealed_2026Q3"),
                  "B": ("2026-09-21", "2026-10-01", "2026-12-31", "sealed_2026Q4")}
REFERENCE_DIRS = {"A": ("holdout_2026H1",), "B": ("sealed_2026Q3",)}
#: The development settlement tables (under the data root) that Test A's overlap days are compared with.
SIDE_REFERENCE_DIRS = {"A": {"fmm": ("caiso_fmm_dev",), "nyiso_rtint": ("nyiso_rtint_dev",)}}
# The first four steps, named for their markets, write the price files; FMM and NYISO_INT write the settlement tables.
STEPS = ("NYISO", "ISONE", "MISO", "CAISO", "FMM", "NYISO_INT")
SIDE_TABLES = (("fmm", "CAISO"), ("nyiso_rtint", "NYISO"))           # sub-directory and market of the settlement tables
SIDE_MARKET = dict(SIDE_TABLES)
STEP_KEY = {"NYISO": "NYISO", "ISONE": "ISONE", "MISO": "MISO", "CAISO": "CAISO",
            "FMM": "fmm", "NYISO_INT": "nyiso_rtint"}               # the market or series each step's rows belong to
OVERRIDE_KEYS = MARKETS + tuple(SIDE_MARKET)                         # what allow_missing_days may be keyed by
#: The price column(s) a row must hold as finite numbers to count as present, by series ('price' for every market).
VALUE_COLUMNS = {"price": ("da_lmp", "rt_lmp"), "fmm": ("fmm_lmp",), "nyiso_rtint": ("rt_int_lmp",)}
#: The series whose parsers localise with ambiguous="NaT" (the July NYISO and ISO-NE downloaders, the NYISO integrated
#: parser): both rows of the repeated hour of a fall-back day are dropped, so that day holds 23 hours, not 25.
REPEATED_HOUR_DROPPED = ("NYISO", "ISONE", "nyiso_rtint")
#: What the fingerprint of a pull hashes: this module and every module of the repository it imports, the downloaders it
#: runs and the node map it reads (the node map is a file of the protocol's code archive).
CODE_FILES = ("es/sealed_pull.py", "es/registration.py", "es/settlement.py", "es/guard.py", "es/config.py",
              "es/data.py", "v2/run_decomposition.py", "v2/ingest_nyiso.py", "v2/ingest_isone.py",
              "v2/ingest_miso.py", "v2/NODE_MAP.csv")
#: Attempts per request before a chunk is given up. A chunk "fails the validator" when every one of them brings a whole
#: response that the validator refuses.
CHUNK_ATTEMPTS = 8
#: The series whose steps download in validated chunks: the keys of allow_incomplete_chunks.
CHUNKED_SERIES = ("CAISO", "fmm", "nyiso_rtint")
#: Five-minute prices in one hour of CAISO's real-time series.
RT_INTERVALS_PER_HOUR = 12
CAISO_TZ = "America/Los_Angeles"
CAISO_NODE_TYPE = {"TH_SP15_GEN-APND": "trading_hub", "TH_NP15_GEN-APND": "trading_hub",
                   "TH_ZP26_GEN-APND": "trading_hub", "DLAP_PGAE-APND": "load_zone",
                   "DLAP_SCE-APND": "load_zone", "DLAP_SDGE-APND": "load_zone"}
OASIS_XML = ("https://oasis.caiso.com/oasisapi/SingleZip?queryname={qn}&startdatetime={s}-0000"
             "&enddatetime={e}-0000&market_run_id={mri}&node={node}&version=1")
OASIS_CSV = OASIS_XML + "&resultformat=6"
NS = {"o": "http://www.caiso.com/soa/OASISReport_v1.xsd"}
# Root-level names that are not part of the record: the state, the lock, the manifest and their temp files.
_UNRECORDED = (".state.json", ".state.json.tmp", ".lock", "MANIFEST.sha256", "MANIFEST.sha256.tmp")


class MissingDaysError(RuntimeError):
    """A step returned no rows for some (node, local settlement day); the step is not recorded as done."""


class ChunkError(RuntimeError):
    """One request (a chunk of a download) gave no usable response in `attempts` attempts. `key` names the chunk,
    `validator_failures` is how many of the answers arrived whole and were refused (rows the validator refused, or no
    rows at all; the others were network failures or answers that could not be read) and `reason` is the text for the
    last of them."""

    def __init__(self, message, key=None, attempts=0, validator_failures=0, reason=None):
        super().__init__(message)
        self.key, self.attempts, self.validator_failures, self.reason = key, attempts, validator_failures, reason


def _iso_now():
    return datetime.now(timezone.utc).isoformat()


def _stamp_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def local_day_bounds_utc(start, end, tz):
    s = pd.Timestamp(start).tz_localize(tz).tz_convert("UTC")
    e = (pd.Timestamp(end) + pd.Timedelta(days=1)).tz_localize(tz).tz_convert("UTC")
    return s, e


def expected_hours(market, start, end):
    """Clock hours in local settlement days start..end: 23 on a spring-forward day and 25 on a
    fall-back day (the July NYISO and ISO-NE downloaders keep 23 of those 25; spec §6.1)."""
    s, e = local_day_bounds_utc(start, end, SETTLE_TZ[market])
    return int((e - s) / pd.Timedelta(hours=1))


def _chunks(s, e, days=30):
    cur = s
    while cur < e:
        nxt = min(cur + pd.Timedelta(days=days), e)
        yield cur, nxt
        cur = nxt


def _fmt(ts):
    return ts.strftime("%Y%m%dT%H:%M")


def _chunk_key(query, node, cs, ce):
    """The key of one OASIS request, as PULL_LOG.txt prints it and allow_incomplete_chunks takes it: the query, the node
    and the UTC interval of the request, with no spaces, e.g.
    'PRC_RTPD_LMP/DLAP_SDGE-APND/20260621T07:00..20260721T07:00'."""
    return f"{query}/{node}/{_fmt(cs)}..{_fmt(ce)}"


def _chunk_keys(series, pull_start, deploy_end):
    """The keys of every chunk that the downloader of `series` ('CAISO', 'fmm' or 'nyiso_rtint') requests for a pull of
    the settlement days pull_start..deploy_end, in request order. For nyiso_rtint a chunk is a month, 'YYYY-MM'."""
    if series == "nyiso_rtint":
        return [str(p) for p in pd.period_range(pull_start[:7], deploy_end[:7], freq="M")]
    s, e = local_day_bounds_utc(pull_start, deploy_end, CAISO_TZ)
    queries = ("PRC_LMP", "PRC_INTVL_LMP") if series == "CAISO" else ("PRC_RTPD_LMP",)
    return [_chunk_key(q, node, cs, ce) for q in queries for node in CAISO_NODE_TYPE for cs, ce in _chunks(s, e)]


def _runs(stamps, step_minutes):
    """The UTC instants `stamps`, one interval of `step_minutes` each, as runs: 'first .. last (n)' for every stretch of
    consecutive intervals (a single one as its instant), joined by ', '."""
    step, runs, run = pd.Timedelta(minutes=step_minutes), [], []
    for t in sorted(stamps):
        if run and t - run[-1] != step:
            runs.append(run)
            run = []
        run.append(t)
    if run:
        runs.append(run)
    fmt = "%Y-%m-%dT%H:%MZ"
    return ", ".join(r[0].strftime(fmt) if len(r) == 1 else f"{r[0]:{fmt}} .. {r[-1]:{fmt}} ({len(r)})" for r in runs)


_UTC = "datetime64[ns, UTC]"
_OASIS_COLUMNS = (("timestamp_utc", _UTC), ("node", object), ("lmp", float))
_FMM_COLUMNS = (("node", object), ("timestamp_utc", _UTC), ("fmm15", float))
_NYISO_COLUMNS = (("node", object), ("timestamp_utc", _UTC), ("rt_int_lmp", float))


def _empty_table(columns, answer):
    """A table of `columns` ((name, dtype), ...) without rows, with what the server said in attrs["answer"]: the result
    of a parser for an answer that is whole and well formed and holds no data."""
    table = pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in columns})
    table.attrs["answer"] = answer
    return table


def _oasis_error_text(root):
    """The error that an OASIS report (an XML tree) holds, as 'OASIS error 1000: No data returned for the specified
    selection'; None if it holds none."""
    found = {}
    for element in root.iter():
        name = element.tag.split("}")[-1]
        if name in ("ERR_CODE", "ERR_DESC") and (element.text or "").strip():
            found[name] = " ".join(element.text.split())
    if not found:
        return None
    return "OASIS error" + (f" {found['ERR_CODE']}" if "ERR_CODE" in found else "") + (
        f": {found['ERR_DESC']}" if "ERR_DESC" in found else "")


def _zip_error_text(zf):
    """The error in the first XML report of an open zip that holds one; None if there is none."""
    for name in zf.namelist():
        if name.endswith(".xml"):
            try:
                text = _oasis_error_text(ET.fromstring(zf.read(name)))
            except ET.ParseError:
                continue
            if text:
                return text
    return None


def _parse_oasis_xml(blob):
    """The LMP rows of an OASIS XML report in a zip. A well-formed answer without data (no data items, an OASIS error,
    a zip with no file) is an empty table that says in attrs["answer"] what the report says; a blob that is no zip
    raises."""
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        names = zf.namelist()
        if not names:
            return _empty_table(_OASIS_COLUMNS, "the zip holds no files")
        root = ET.fromstring(zf.read(names[0]))
    rows = []
    for item in root.findall(".//o:REPORT_DATA", NS):
        d = {c.tag.split("}")[-1]: c.text for c in item}
        if d.get("DATA_ITEM") == "LMP_PRC":
            rows.append((d["INTERVAL_START_GMT"], d["RESOURCE_NAME"], float(d["VALUE"])))
    if not rows:
        items = root.findall(".//o:REPORT_DATA", NS)
        return _empty_table(_OASIS_COLUMNS, _oasis_error_text(root) or (
            "the report holds no LMP items" if items else "the report holds no data items"))
    df = pd.DataFrame(rows, columns=["timestamp_utc", "node", "lmp"])
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df


def _parse_fmm_answer(blob):
    """es.settlement's parser of an FMM answer (a zip with one CSV), except that a well-formed zip with no CSV in it,
    which is how OASIS says an error or no data, gives an empty table that says what the report says, not an exception.
    A blob that is no zip raises, as it did."""
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        if not any(n.endswith(".csv") for n in zf.namelist()):
            return _empty_table(_FMM_COLUMNS, _zip_error_text(zf) or "the zip holds no CSV file")
    try:
        return _parse_caiso(blob)
    except pd.errors.EmptyDataError:                       # a CSV of no bytes, not even a header
        return _empty_table(_FMM_COLUMNS, "the CSV in the zip is empty")


def _parse_nyiso_answer(blob):
    """es.settlement's parser of a NYISO monthly archive, except that a well-formed zip with no CSV file in it gives an
    empty table, not an exception. A blob that is no zip raises."""
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        if not any(n.endswith(".csv") for n in zf.namelist()):
            return _empty_table(_NYISO_COLUMNS, "the zip holds no CSV file")
    try:
        return _parse_nyiso(blob)
    except pd.errors.EmptyDataError:                       # a CSV of no bytes, not even a header
        return _empty_table(_NYISO_COLUMNS, "the CSV in the zip is empty")


# ----------------------------------------------------------------------------- durable, atomic writes
def _fsync(fd):
    """Flush a descriptor to the disk. macOS needs F_FULLFSYNC: plain fsync does not flush the drive's cache."""
    full = getattr(fcntl, "F_FULLFSYNC", None)
    if full is not None:
        try:
            fcntl.fcntl(fd, full)
            return
        except OSError:                                   # a file system that does not support it
            pass
    os.fsync(fd)


def _fsync_dir(directory):
    fd = os.open(directory, os.O_RDONLY)
    try:
        _fsync(fd)
    finally:
        os.close(fd)


def _atomic_write(path, data):
    """Write bytes through a temp file in the same directory and os.replace, so that a crash never leaves a
    half-written file under the final name; flush the file before the rename and the directory after it. A leftover
    temp file is overwritten by the next attempt."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        _fsync(fh.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def _atomic_to_parquet(df, path):
    """The same for a parquet file (a trimmed table)."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    with open(tmp, "r+b") as fh:
        _fsync(fh.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


# ----------------------------------------------------------------------------- one request
class IncompleteChunks:
    """The chunks of one series that a step may accept although their responses never came complete: the
    allow_incomplete_chunks of an attempt, matched by pull_sealed with what PULL_LOG.txt says failed earlier. `keys` are
    those chunks, `accepted` fills with {key: reason} as the step runs, and `log(*lines)` writes to PULL_LOG.txt."""

    def __init__(self, series, keys, log):
        self.series, self.keys, self.log, self.accepted = series, frozenset(keys), log, {}

    def take(self, key, part, validate):
        """Accept `part`, a whole response of chunk `key` (the one with the fewest problems, which _fetch_parse
        chooses), or return False. It is accepted when the chunk is one of `keys` and the only things wrong are
        intervals missing from the grid and prices that are not finite numbers; rows outside the interval that was asked
        for are never accepted. A table without rows is accepted as zero rows: every interval is missing, and the
        reason says what the server said. The reason, and every missing and non-finite interval, go to the log."""
        problems = getattr(validate, "problems", None)
        if key not in self.keys or problems is None:
            return False
        try:
            found = problems(part)
        except Exception:                                  # the facts cannot be read from this table: nothing to accept
            return False
        missing, bad = found["missing"], found["nonfinite"]
        if found["outside"] or not (len(missing) or len(bad)):
            return False
        reason = "; ".join(validate.texts(found))
        if not len(part):
            answer = getattr(part, "attrs", {}).get("answer")
            reason += "; no rows came" + (f" ({answer})" if answer else "")
        self.accepted[key] = reason
        lines = [f"chunk_accepted: {self.series} {key} | {reason}"]
        if len(missing):
            lines.append(f"chunk_accepted_detail: {self.series} {key} | {len(missing)} interval(s) of {validate.step} "
                         f"minutes missing: {_runs(missing, validate.step)}")
        if len(bad):
            where = "; ".join(f"{node}: {_runs(list(g['timestamp_utc']), validate.step)}"
                              for node, g in bad.groupby("node"))
            lines.append(f"chunk_accepted_detail: {self.series} {key} | {len(bad)} price(s) not finite: {where}")
        self.log(*lines)
        return True


def _trouble(validate, part):
    """How much is wrong with a refused answer, to choose the best of several: (1 if rows lie outside the interval
    that was asked for, else 0, the intervals missing plus the prices that are not finite numbers). Lower is better. An
    answer without rows is missing every interval. (0, 0) when the validator gives no facts."""
    problems = getattr(validate, "problems", None)
    if problems is None:
        return (0, 0)
    try:
        found = problems(part)
    except Exception:
        return (1, float("inf"))
    return (1 if found["outside"] else 0, len(found["missing"]) + len(found["nonfinite"]))


def _fetch_parse(url, parse, what, fetch, sleep, cache_dir=None, attempts=CHUNK_ATTEMPTS, validate=None, key=None,
                 incomplete=None):
    """One request; returns (rows, came_from_the_cache). A cached response is reused if it parses to rows that pass
    `validate` (which returns None, or the reason the rows are unusable: short of the interval grid, outside the
    requested bounds, prices that are not finite); otherwise (a crash can cut a cache file short) it counts as a miss
    and is fetched again and overwritten. Empty, unusable or error responses and network failures are retried with
    backoff (6 s doubling to 60 s, no pause after the last attempt), then raised as a ChunkError, a RuntimeError that
    names the chunk `key` and counts the answers that were refused (spec §6.5 step 6). A good response is cached,
    atomically, before it is returned, so an interrupted pull resumes (Ruling 33); an unusable one never is. An
    AssertionError (the tests' network block) or a SealedDataError is not a transient fault: it is raised at once.

    A whole answer is one the server gave (`fetch` returned) that can be read. It is refused if the validator refuses
    its rows, or if it holds no rows at all (OASIS's no-data report, a CSV without rows, a zip without a CSV: Ruling
    121, F1). Whatever `fetch` raises (a timeout, a connection error, an HTTP error status) is a network failure, and
    an answer that cannot be read (no zip, cut short) is counted with the network failures, not as a whole answer.

    The one exception to raising: `incomplete` (an IncompleteChunks) may accept chunk `key` after the last attempt, if
    every one of the attempts brought a whole answer that was refused (F2: no network failure in this call). The answer
    with the fewest problems (the last of equals) is returned, as (rows, False), not cached; an answer without rows is
    returned as zero rows."""
    path = Path(cache_dir) / f"{hashlib.sha256(url.encode()).hexdigest()}.bin" if cache_dir else None
    if path is not None and path.exists():
        try:
            part = parse(path.read_bytes())
            why = "holds no rows" if not len(part) else (validate(part) if validate else None)
        except (AssertionError, SealedDataError):
            raise
        except Exception as exc:
            why = f"does not parse ({exc!r})"
        if why is None:
            print(f"{_stamp_now()} {what}: cache hit {path.name}", flush=True)
            return part, True
        print(f"{_stamp_now()} {what}: cached response {path.name} {why}; fetching it again", flush=True)
    last, refused, last_why, best = None, 0, None, None       # best: (trouble, rows, reason) of the best refused answer
    for k in range(attempts):
        try:
            blob = fetch(url)                                  # the request: what it raises is a network failure
        except (AssertionError, SealedDataError):
            raise
        except Exception as exc:
            last = exc
        else:
            try:                                               # the server answered: can the answer be read?
                part = parse(blob)
                answer = getattr(part, "attrs", {}).get("answer") if not len(part) else None
                why = (("no rows" + (f" ({answer})" if answer else "")) if not len(part)
                       else (validate(part) if validate else None))
            except (AssertionError, SealedDataError):
                raise
            except Exception as exc:                           # an answer that cannot be read is no whole answer
                last = exc
            else:
                if why is None:
                    if path is not None:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        _atomic_write(path, blob)
                    print(f"{_stamp_now()} {what}: {len(blob):,} bytes on attempt {k + 1}", flush=True)
                    return part, False
                last = RuntimeError(f"no rows for {what}" + (f": {answer}" if answer else "") if not len(part)
                                    else f"{what}: {why}")
                refused, last_why = refused + 1, why
                trouble = _trouble(validate, part)
                if best is None or trouble <= best[0]:
                    best = (trouble, part, why)
        print(f"{_stamp_now()} {what}: attempt {k + 1}/{attempts} failed: {last!r}", flush=True)
        if k < attempts - 1:
            sleep(min(60, 6 * 2 ** k))
    if incomplete is not None and best is not None and refused == attempts and incomplete.take(key, best[1], validate):
        print(f"{_stamp_now()} {what}: accepted as incomplete, the answer with the fewest problems of {attempts} "
              f"({best[2]}); not cached", flush=True)
        return best[1], False
    raise ChunkError(f"{what}: no data after {attempts} attempts; the last problem: {last}", key=key, attempts=attempts,
                     validator_failures=refused, reason=last_why) from last


def _refuse_sealed(end, sealed):
    """The pullers fetch dates after SEAL_LAST_DEV_DATE only for pull_sealed, which has checked the registration."""
    if not sealed and pd.Timestamp(end) > pd.Timestamp(SEAL_LAST_DEV_DATE):
        raise SealedDataError(f"end {end} is after {SEAL_LAST_DEV_DATE}; sealed dates are pulled only by pull_sealed, "
                              "after the registration check")


def _finite(values):
    """Boolean array over `values`: True where the value is a finite number. NaN, infinity and text that is no number
    are not."""
    return np.isfinite(pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float, na_value=np.nan))


def _grid_validator(start, end, step_minutes, market, sealed, value):
    """The rows of one chunk must cover its interval grid, lie inside [start, end) and hold finite numbers in their
    price column `value`. A development pull that receives a row at or after the seal gets a SealedDataError at once.
    The function returned says why a chunk is unusable (None if it is fine). Its attribute problems(part) gives the
    facts {outside: rows outside the interval, missing: the intervals of the grid with no row, nonfinite: the rows whose
    price is not a finite number}, texts(facts) says them in words and step is the interval in minutes."""
    want = pd.date_range(start, end, freq=f"{step_minutes}min", inclusive="left")
    seal = first_sealed_utc(market)

    def problems(part):
        ts = part["timestamp_utc"]
        return {"outside": int(((ts < start) | (ts >= end)).sum()),
                "missing": want.difference(pd.DatetimeIndex(ts.unique())),
                "nonfinite": part.loc[~_finite(part[value]), ["node", "timestamp_utc"]]}

    def texts(found):
        out = []
        if found["outside"]:
            out.append(f"{found['outside']} row(s) outside the requested {start} .. {end}")
        if len(found["missing"]):
            out.append(f"short: {len(found['missing'])} of {len(want)} {step_minutes}-minute intervals missing, first "
                       f"{found['missing'][0]}")
        if len(found["nonfinite"]):
            out.append(f"{len(found['nonfinite'])} price(s) in {value} are not finite numbers")
        return out

    def check(part):
        ts = part["timestamp_utc"]
        if not sealed and bool((ts >= seal).any()):
            raise SealedDataError(f"a development pull received rows at or after {seal}; that window is sealed")
        found = texts(problems(part))
        return found[0] if found else None

    check.problems, check.texts, check.step = problems, texts, step_minutes
    return check


def _month_validator(period, sealed):
    """The rows of one NYISO monthly archive must lie inside that local month and hold finite prices. Like
    _grid_validator's function, it has the attributes problems, texts and step (an archive holds hourly prices)."""
    lo = period.start_time.tz_localize("America/New_York").tz_convert("UTC")
    hi = (period + 1).start_time.tz_localize("America/New_York").tz_convert("UTC")
    seal = first_sealed_utc("NYISO")

    def problems(part):
        """An archive has no grid to be short of, except when it holds no rows at all: then every hour of the month is
        missing (in every zone)."""
        ts = part["timestamp_utc"]
        every_hour = pd.date_range(lo, hi, freq="h", inclusive="left") if not len(part) else pd.DatetimeIndex([])
        return {"outside": int(((ts < lo) | (ts >= hi)).sum()), "missing": every_hour,
                "nonfinite": part.loc[~_finite(part["rt_int_lmp"]), ["node", "timestamp_utc"]]}

    def texts(found):
        out = []
        if found["outside"]:
            out.append(f"{found['outside']} row(s) outside the month {period} ({lo} .. {hi})")
        if len(found["missing"]):
            out.append(f"short: the archive of {period} holds no rows "
                       f"({len(found['missing'])} hours missing in every zone)")
        if len(found["nonfinite"]):
            out.append(f"{len(found['nonfinite'])} price(s) in rt_int_lmp are not finite numbers")
        return out

    def check(part):
        ts = part["timestamp_utc"]
        if not sealed and bool((ts >= seal).any()):
            raise SealedDataError(f"a development pull received rows at or after {seal}; that window is sealed")
        found = texts(problems(part))
        return found[0] if found else None

    check.problems, check.texts, check.step = problems, texts, 60
    return check


def _hourly_rt_mean(df):
    """Hourly mean of 5-minute prices (columns node, timestamp_utc, lmp). An hour in which any of the prices is not a
    finite number, or which has fewer than RT_INTERVALS_PER_HOUR of them, is NaN, instead of the mean of the others,
    which pandas' mean would give. (A validated chunk has every interval; a chunk accepted as incomplete may not.)"""
    df = df.assign(h=df["timestamp_utc"].dt.floor("h"), bad=~_finite(df["lmp"]))
    out = df.groupby(["node", "h"], as_index=False).agg(lmp=("lmp", "mean"), bad=("bad", "max"), n=("lmp", "size"))
    out.loc[out["bad"].astype(bool) | (out["n"] < RT_INTERVALS_PER_HOUR), "lmp"] = float("nan")
    return out.drop(columns=["bad", "n"]).rename(columns={"h": "timestamp_utc"})


def _hourly_fmm_mean(raw):
    """Hourly mean of 15-minute prices (columns node, timestamp_utc, fmm15) as fmm_lmp, and n15, the number of finite
    intervals in the hour. An hour in which any interval is not a finite number has a NaN price (and an n15 below 4)."""
    raw = raw.assign(h=raw["timestamp_utc"].dt.floor("h"), ok=_finite(raw["fmm15"]))
    out = raw.groupby(["node", "h"], as_index=False).agg(fmm_lmp=("fmm15", "mean"), n15=("ok", "sum"),
                                                          all_ok=("ok", "min"))
    out.loc[~out["all_ok"].astype(bool), "fmm_lmp"] = float("nan")
    return out.drop(columns="all_ok").rename(columns={"h": "timestamp_utc"})


def caiso_exact(start, end, root, fetch=None, pause=5.5, sleep=time.sleep, cache_dir=None, *, _sealed=False,
                incomplete=None):
    """CAISO DA (PRC_LMP/DAM) and RT (PRC_INTVL_LMP/RTM, 5-min -> hourly mean) over exact local days.
    Same schema and RT aggregation as v2/ingest_caiso.py. Every response must cover its interval grid (DAM: the hours
    of the chunk, RTM: 12 x hours) and stay inside it; a duplicate row inside a response is dropped. `incomplete`: an
    IncompleteChunks that may accept chunks which never come complete (see _fetch_parse)."""
    _refuse_sealed(end, _sealed)
    fetch = fetch or _fetch
    s, e = local_day_bounds_utc(start, end, CAISO_TZ)
    series = {}
    for qn, mri, col, hourly in (("PRC_LMP", "DAM", "da_lmp", False), ("PRC_INTVL_LMP", "RTM", "rt_lmp", True)):
        parts = []
        for node in CAISO_NODE_TYPE:
            for cs, ce in _chunks(s, e):
                url = OASIS_XML.format(qn=qn, mri=mri, node=node, s=_fmt(cs), e=_fmt(ce))
                part, cached = _fetch_parse(url, _parse_oasis_xml, f"{qn} {node} {cs}..{ce}", fetch, sleep, cache_dir,
                                            validate=_grid_validator(cs, ce, 5 if hourly else 60, "CAISO", _sealed,
                                                                     "lmp"),
                                            key=_chunk_key(qn, node, cs, ce), incomplete=incomplete)
                parts.append(part)
                if not cached:
                    sleep(pause)
        df = pd.concat(parts, ignore_index=True).drop_duplicates(["node", "timestamp_utc"])
        if hourly:
            df = _hourly_rt_mean(df)
        series[col] = df.rename(columns={"lmp": col})
    panel = series["da_lmp"].merge(series["rt_lmp"], on=["timestamp_utc", "node"], how="inner")
    panel["market"] = "CAISO"                     # every response was checked against its chunk: all rows lie in [s, e)
    panel["node_type"] = panel["node"].map(CAISO_NODE_TYPE)
    panel = (panel[["market", "node", "node_type", "timestamp_utc", "da_lmp", "rt_lmp"]]
             .sort_values(["node", "timestamp_utc"]).reset_index(drop=True))
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    out = root / f"caiso_da_rt_{start.replace('-', '')}_{end.replace('-', '')}.parquet"
    panel.to_parquet(out, index=False)
    return out


def caiso_fmm_exact(start, end, root, fetch=None, pause=5.5, sleep=time.sleep, cache_dir=None, *, _sealed=False,
                    incomplete=None):
    """CAISO's 15-minute FMM prices as hourly means with the number of intervals in each hour (n15). Every response
    must cover 4 x hours intervals and stay inside its chunk. `incomplete`: as for caiso_exact."""
    _refuse_sealed(end, _sealed)
    fetch = fetch or _fetch
    s, e = local_day_bounds_utc(start, end, CAISO_TZ)
    parts = []
    for node in CAISO_NODE_TYPE:
        for cs, ce in _chunks(s, e):
            url = OASIS_CSV.format(qn="PRC_RTPD_LMP", mri="RTPD", node=node, s=_fmt(cs), e=_fmt(ce))
            part, cached = _fetch_parse(url, _parse_fmm_answer, f"FMM {node} {cs}..{ce}", fetch, sleep, cache_dir,
                                        validate=_grid_validator(cs, ce, 15, "CAISO", _sealed, "fmm15"),
                                        key=_chunk_key("PRC_RTPD_LMP", node, cs, ce), incomplete=incomplete)
            parts.append(part)
            if not cached:
                sleep(pause)
    raw = pd.concat(parts, ignore_index=True).drop_duplicates(["node", "timestamp_utc"])   # all rows lie in [s, e)
    hourly = _hourly_fmm_mean(raw)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    out = root / f"caiso_fmm_{start.replace('-', '')}_{end.replace('-', '')}.parquet"
    hourly.to_parquet(out, index=False)
    return out


def nyiso_integrated_months(start_month, end_month, root, fetch=None, pause=2.0, sleep=time.sleep, cache_dir=None,
                            *, _sealed=False, log=None, incomplete=None):
    """NYISO's integrated hourly real-time LBMP, one monthly archive per month; each archive's rows must lie inside
    its local month. `log`, if given, is called with one line per month that says how many rows were kept and how
    many were dropped for an ambiguous or nonexistent local time (the repeated hour of a fall-back day, the missing
    hour of a spring-forward day). `incomplete`: as for caiso_exact; the key of a chunk is its month, 'YYYY-MM'."""
    _refuse_sealed(str(pd.Period(end_month, "M").end_time.date()), _sealed)
    fetch = fetch or _fetch
    frames = []
    for period in pd.period_range(start_month, end_month, freq="M"):
        url = NYISO_URL.format(ym=period.strftime("%Y%m"))
        part, cached = _fetch_parse(url, _parse_nyiso_answer, f"NYISO integrated {period}", fetch, sleep, cache_dir,
                                    validate=_month_validator(period, _sealed), key=str(period), incomplete=incomplete)
        frames.append(part)
        line = (f"NYISO integrated {period}: {len(part)} rows kept, {part.attrs.get('dropped_nat', 0)} dropped "
                "(ambiguous or nonexistent local time)")
        print(line, flush=True)
        if log is not None:
            log(line)
        if not cached:
            sleep(pause)
    out_df = pd.concat(frames, ignore_index=True).drop_duplicates(["node", "timestamp_utc"])
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    out = root / f"nyiso_rtint_{start_month.replace('-', '')}_{end_month.replace('-', '')}.parquet"
    out_df.to_parquet(out, index=False)
    return out


def _download_caiso(start, end, root, incomplete=None):
    return caiso_exact(start, end, root, cache_dir=root / "raw", _sealed=True, incomplete=incomplete)


def _download_fmm(start, end, root, incomplete=None):
    return caiso_fmm_exact(start, end, root / "fmm", cache_dir=root / "raw", _sealed=True, incomplete=incomplete)


def _download_nyiso_integrated(start, end, root, incomplete=None):
    return nyiso_integrated_months(start, end, root / "nyiso_rtint", cache_dir=root / "raw", _sealed=True,
                                   log=lambda line: _log(root, line), incomplete=incomplete)


def _default_builders():
    return {"NYISO": ingest_nyiso.build, "ISONE": ingest_isone.build, "MISO": ingest_miso.build,
            "CAISO": _download_caiso, "FMM": _download_fmm, "NYISO_INT": _download_nyiso_integrated}


#: The downloaders as they were when this module was imported (a test may replace a v2 module's build afterwards).
REAL_DOWNLOADERS = _default_builders()


# ----------------------------------------------------------------------------- reading the tables
def _days(df, market=None):
    """Local settlement date of each row. FMM and NYISO integrated tables have no market column: they pass theirs."""
    markets = df["market"] if market is None else pd.Series(market, index=df.index)
    return settlement_frame(pd.to_datetime(df["timestamp_utc"], utc=True), markets)["settle_date"]


def _trim(path, start, end, market=None):
    """Drop rows whose local settlement date lies outside start..end, atomically; return how many went."""
    df = pd.read_parquet(path)
    d = _days(df, market)
    keep = ((d >= dt.date.fromisoformat(start)) & (d <= dt.date.fromisoformat(end))).to_numpy()
    if not keep.all():
        _atomic_to_parquet(df[keep].reset_index(drop=True), path)
    return int((~keep).sum())


def _cut(market, before, sealed):
    """The UTC instant at which reading stops: the local midnight that starts `before` and, unless the caller is
    pull_sealed, the seal. None means no cut."""
    cuts = []
    if before is not None:
        cuts.append(pd.Timestamp(before).tz_localize(SETTLE_TZ[market]).tz_convert("UTC"))
    if not sealed:
        cuts.append(first_sealed_utc(market))
    return min(cuts) if cuts else None


def _read_before(path, cut):
    """A parquet file through a pyarrow predicate that stops at `cut` (the pattern of es.data.load_dev_raw), typed like
    the file's own column so tz-naive MISO files work: later rows never reach memory."""
    if cut is None:
        return pd.read_parquet(path)
    ts_type = pq.ParquetFile(path).schema_arrow.field("timestamp_utc").type
    below = pa.scalar(cut.to_pydatetime(), type=ts_type)
    return pq.read_table(path, filters=[("timestamp_utc", "<", below)]).to_pandas()


def _concat(frames):
    if not frames:
        return None
    nonempty = [f for f in frames if len(f)]
    df = pd.concat(nonempty, ignore_index=True) if nonempty else frames[0]
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df


def _read_prices(roots, before=None, markets=MARKETS, *, sealed=False):
    """Price rows of `markets` under `roots`, one per (market, node, timestamp). With `before`, a settlement date,
    each file stops at the local midnight starting that day; unless `sealed`, it stops at the seal too."""
    frames = [_read_before(path, _cut(_market_of(path), before, sealed))
              for r in roots for path in raw_files(dirs=[r], markets=markets)]
    df = _concat(frames)
    return None if df is None else df.drop_duplicates(["market", "node", "timestamp_utc"], keep="first")


def _read_side(directory, series, before=None, *, sealed=False):
    """The rows of an FMM or NYISO integrated table (the parquet files in `directory`), cut like _read_prices."""
    cut = _cut(SIDE_MARKET[series], before, sealed)
    df = _concat([_read_before(path, cut) for path in sorted(Path(directory).glob("*.parquet"))])
    return None if df is None else df.drop_duplicates(["node", "timestamp_utc"], keep="first")


def _price_overlap(new_root, reference_roots, lo, hi, end_exclusive, sealed):
    key = ["market", "node", "timestamp_utc"]
    new = _read_prices([new_root], end_exclusive, sealed=sealed)
    ref = _read_prices(reference_roots, end_exclusive, sealed=sealed)
    if new is None or ref is None:
        return [{"series": "price", "market": "all", "note": "no reference files" if ref is None else "no new files"}]
    new = new[((_days(new) >= lo) & (_days(new) < hi)).to_numpy()]
    ref = ref[((_days(ref) >= lo) & (_days(ref) < hi)).to_numpy()]
    m = new.merge(ref, on=key, how="outer", suffixes=("_new", "_ref"), indicator=True)
    rows = []
    for market, g in m.groupby("market"):
        both = g[g["_merge"] == "both"]
        dda = (both["da_lmp_new"] - both["da_lmp_ref"]).abs()
        drt = (both["rt_lmp_new"] - both["rt_lmp_ref"]).abs()
        rows.append({"series": "price", "market": market, "matched": int(len(both)),
                     "new_only": int((g["_merge"] == "left_only").sum()),
                     "reference_only": int((g["_merge"] == "right_only").sum()),
                     "rows_differing": int(((dda > 1e-9) | (drt > 1e-9)).sum()),
                     "max_abs_diff_da": float(dda.max()) if len(both) else float("nan"),
                     "max_abs_diff_rt": float(drt.max()) if len(both) else float("nan")})
    return rows


def _side_overlap(series, new_dir, reference_dirs, lo, hi, end_exclusive, sealed):
    """FMM (fmm_lmp and n15) or NYISO integrated (rt_int_lmp) rows of the overlap days against a reference table."""
    market = SIDE_MARKET[series]
    value = "fmm_lmp" if series == "fmm" else "rt_int_lmp"
    new = _read_side(new_dir, series, end_exclusive, sealed=sealed)
    refs = [_read_side(d, series, end_exclusive, sealed=sealed) for d in reference_dirs]
    ref = _concat([r for r in refs if r is not None])
    if new is None or ref is None:
        return [{"series": series, "market": market, "note": "no reference files" if ref is None else "no new files"}]
    needed = [value, "n15"] if series == "fmm" else [value]
    absent = [c for c in needed if c not in new.columns or c not in ref.columns]
    if absent:                                   # logged, not enforced: a table of another shape is noted, not a crash
        return [{"series": series, "market": market,
                 "note": f"cannot compare: column(s) {absent} missing from the new or the reference table"}]
    new = new[((_days(new, market) >= lo) & (_days(new, market) < hi)).to_numpy()]
    ref = ref[((_days(ref, market) >= lo) & (_days(ref, market) < hi)).to_numpy()]
    m = new.merge(ref, on=["node", "timestamp_utc"], how="outer", suffixes=("_new", "_ref"), indicator=True)
    both = m[m["_merge"] == "both"]
    diff = (both[f"{value}_new"] - both[f"{value}_ref"]).abs()
    differing = diff > 1e-9
    if series == "fmm":
        differing = differing | (both["n15_new"] != both["n15_ref"])
    return [{"series": series, "market": market, "matched": int(len(both)),
             "new_only": int((m["_merge"] == "left_only").sum()),
             "reference_only": int((m["_merge"] == "right_only").sum()),
             "rows_differing": int(differing.sum()), "max_abs_diff_da": float("nan"),
             "max_abs_diff_rt": float(diff.max()) if len(both) else float("nan")}]


def overlap_check(new_root, reference_roots, start, end_exclusive, *, _sealed=False, side_references=None):
    """Spec §6.5 step 6: pulled rows before deployment must match the files already opened.
    Logged, not enforced (ISOs revise settled prices). Neither side is read past the local midnight that starts
    end_exclusive, nor, unless pull_sealed passes _sealed=True, past the seal: the H1 CAISO file holds six rows of
    1 July 00:00 PDT that must stay out of memory. `side_references` is {series: [reference directories]} for the
    FMM and NYISO integrated tables in new_root/<series>."""
    lo, hi = dt.date.fromisoformat(start), dt.date.fromisoformat(end_exclusive)
    rows = _price_overlap(new_root, reference_roots, lo, hi, end_exclusive, _sealed)
    for series, directories in (side_references or {}).items():
        rows.extend(_side_overlap(series, Path(new_root) / series, directories, lo, hi, end_exclusive, _sealed))
    return pd.DataFrame(rows)


def _log_overlap(root, overlap):
    for r in overlap.to_dict("records"):
        if isinstance(r.get("note"), str):
            _log(root, f"overlap {r['series']} {r['market']}: {r['note']}")
        else:
            _log(root, f"overlap {r['series']} {r['market']}: matched {int(r['matched'])}, "
                       f"new_only {int(r['new_only'])}, reference_only {int(r['reference_only'])}, "
                       f"rows_differing {int(r['rows_differing'])}, max_abs_diff_da {r['max_abs_diff_da']}, "
                       f"max_abs_diff_rt {r['max_abs_diff_rt']}")


# ----------------------------------------------------------------------------- coverage
def _node_map_nodes():
    """(market, node) pairs of the frozen node map, v2/NODE_MAP.csv: the file es.data's restrict_to_node_map keeps,
    limited to the four sealed markets (it also lists SPP, which is not pulled). 33 nodes."""
    nm = pd.read_csv(V2_DIR / "NODE_MAP.csv")
    nm = nm[nm["market"].isin(MARKETS)]
    return sorted(zip(nm["market"], nm["node_id"]))


def _series_rows(root, key, *, sealed=False):
    """Every row of one series as stored: a market's price rows, the FMM table or the NYISO integrated table; None if
    there is no file."""
    if key in MARKETS:
        return _read_prices([root], markets=(key,), sealed=sealed)
    return _read_side(Path(root) / key, key, sealed=sealed)


def _row_flags(df, key, *, strict=False):
    """(finite, usable): boolean arrays over the rows of a series. A row is finite when every one of its price columns
    (VALUE_COLUMNS) holds a finite number: NaN, infinity, text that is no number and a column the table lacks are not
    finite. It is usable when it is finite and, for the FMM table, the hour holds all four 15-minute intervals
    (n15 == 4, as load_caiso_fmm demands). With `strict`, a table that lacks a column raises: the day check does not
    guess."""
    columns = VALUE_COLUMNS["price" if key in MARKETS else key]
    absent = [c for c in columns + (("n15",) if key == "fmm" else ()) if c not in df.columns]
    if absent and strict:
        raise RuntimeError(f"the rows of {key} have no {absent} column(s), so none can be shown to hold finite prices")
    finite = np.ones(len(df), dtype=bool)
    for c in columns:
        finite &= _finite(df[c]) if c in df.columns else False
    usable = finite.copy()
    if key == "fmm":
        usable &= (df["n15"] == 4).to_numpy() if "n15" in df.columns else False
    return finite, usable


def _on_the_hour(df):
    """Boolean array over the rows of a series: True where the instant is a whole hour in UTC (minute, second and
    fraction of a second 0). Only such a row can be one of the hours of a day; the markets' clocks are whole hours
    from UTC, so this is the hour on their clocks too."""
    stamps = pd.to_datetime(df["timestamp_utc"], utc=True)
    return (stamps == stamps.dt.floor("h")).to_numpy()


def coverage_manifest(root, start, end, *, _sealed=False):
    """Hours per node against expected, for every node of the node map and for each of three series: the price files
    ('price', 33 nodes), the FMM table ('fmm', the 6 CAISO nodes) and the NYISO integrated table ('nyiso_rtint', the
    10 zones). `hours` counts the rows on the hour (_on_the_hour) whose prices are finite numbers (for fmm also
    n15 == 4), the rows the day check counts; `nonfinite_rows` counts the rows in the same days that hold a NaN or an
    infinite price, which are not hours. A node with no rows at all is listed with 0 hours and share 0.0. Nodes in the
    price files but not in the node map (NYISO's other zones) are left out of the table and named in
    out.attrs["not_in_node_map"]; those of the settlement tables in out.attrs["side_not_in_node_map"]."""
    lo, hi = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    nodes = _node_map_nodes()
    frames, extra, side_extra = [], [], []
    for series, keys in (("price", MARKETS), ("fmm", ("fmm",)), ("nyiso_rtint", ("nyiso_rtint",))):
        for key in keys:
            market = key if key in MARKETS else SIDE_MARKET[key]
            df = _series_rows(root, key, sealed=_sealed)
            if df is None:
                hours, nonfinite, found = pd.Series(dtype="int64"), pd.Series(dtype="int64"), set()
            else:
                finite, usable = _row_flags(df, key)
                d = _days(df, market)
                inside = ((d >= lo) & (d <= hi)).to_numpy()
                hours = df[usable & _on_the_hour(df) & inside].groupby("node").size()
                nonfinite, found = df[~finite & inside].groupby("node").size(), set(df["node"])
            keep = [n for m, n in nodes if m == market]
            frames.append(pd.DataFrame([{"series": series, "market": market, "node": n, "hours": int(hours.get(n, 0)),
                                         "expected_hours": expected_hours(market, start, end),
                                         "nonfinite_rows": int(nonfinite.get(n, 0))} for n in keep]))
            (extra if series == "price" else side_extra).extend(
                (market, n) if series == "price" else (series, n) for n in sorted(found - set(keep)))
    out = pd.concat(frames, ignore_index=True)
    out["share"] = out["hours"] / out["expected_hours"]
    out.attrs["not_in_node_map"] = sorted(extra)
    out.attrs["side_not_in_node_map"] = sorted(side_extra)
    return out


# ----------------------------------------------------------------------------- the manifest
def _record_files(root):
    """The files of the sealed record: everything under root except the manifest, the state file, the lock and
    their temp files (_UNRECORDED)."""
    return [p for p in sorted(root.rglob("*")) if p.is_file() and p.relative_to(root).as_posix() not in _UNRECORDED]


def write_manifest(root):
    root = Path(root)
    lines = [f"{sha256_file(p)}  {p.relative_to(root).as_posix()}" for p in _record_files(root)]
    path = root / "MANIFEST.sha256"
    _atomic_write(path, ("\n".join(lines) + "\n").encode())
    return path


def verify_manifest(root):
    """Raise RuntimeError, naming every file, unless the files under root are exactly the ones the manifest lists,
    with the hashes it lists: a file changed, a listed file missing, or a file present but not listed."""
    root = Path(root)
    manifest = root / "MANIFEST.sha256"
    if not manifest.exists():
        raise RuntimeError(f"no MANIFEST.sha256 in {root}: the pull is not complete")
    listed = {}
    for line in manifest.read_text().splitlines():
        digest, sep, rel = line.partition("  ")
        if not sep or not rel:
            raise RuntimeError(f"malformed line in {manifest}: {line!r}")
        if rel in listed:
            raise RuntimeError(f"{manifest} lists {rel} twice")
        listed[rel] = digest
    actual = {p.relative_to(root).as_posix(): p for p in _record_files(root)}
    problems = [f"sealed file changed after it was written: {rel}"
                for rel in sorted(listed.keys() & actual.keys()) if sha256_file(actual[rel]) != listed[rel]]
    problems += [f"sealed file missing since the manifest was written: {rel}"
                 for rel in sorted(listed.keys() - actual.keys())]
    problems += [f"sealed file not in the manifest: {rel}" for rel in sorted(actual.keys() - listed.keys())]
    if problems:
        raise RuntimeError("; ".join(problems))


def manifest_sha256(root):
    """SHA-256 of MANIFEST.sha256 itself: the one number that pins the whole record, for run_sealed to record."""
    manifest = Path(root) / "MANIFEST.sha256"
    if not manifest.exists():
        raise RuntimeError(f"no MANIFEST.sha256 in {root}: the pull is not complete")
    return sha256_file(manifest)


def _log(root, *lines):
    with open(root / "PULL_LOG.txt", "a") as fh:
        fh.write("".join(f"{line}\n" for line in lines))


# ----------------------------------------------------------------------------- the day check
def _expected_day_hours(key, day):
    """The hours a complete (node, local settlement day) holds in a series: the clock hours of the day in the market's
    time zone (24; 23 on a spring-forward day; 25 on a fall-back day), except 23 on a fall-back day in the series that
    drop the repeated hour (REPEATED_HOUR_DROPPED)."""
    market = key if key in MARKETS else SIDE_MARKET[key]
    clock = expected_hours(market, day.isoformat(), day.isoformat())
    return clock - 2 if clock == 25 and key in REPEATED_HOUR_DROPPED else clock


def _day_facts(root, key, start, end, *, sealed=False):
    """(no_rows, short, over, off_hour): the (node, local settlement day) pairs of the node map's nodes of a market
    ('NYISO', ...) or a settlement series ('fmm', 'nyiso_rtint') that are not complete, over the days start to end
    inclusive in the files under root, and the rows that are not on the hour. A pair is complete when it holds exactly
    as many rows as the day has hours (_expected_day_hours): as many usable rows, that is rows on the hour whose prices
    are finite numbers (and, for fmm, n15 == 4), and no more rows of any kind. no_rows: a sorted list of the pairs with
    no usable row at all. short: {pair: (usable rows, expected hours, rows with a non-finite price, rows off the hour)}
    for the pairs with some usable rows but fewer than the day's hours, whether the ISO sent too few, the downloader
    dropped a row it could not read, a price is NaN or infinite, or rows are stamped off the hour. over: {pair: (rows,
    expected hours, rows off the hour)} for the pairs with enough usable rows but more rows than hours: an hour twice,
    a stray row, a fall-back day that kept the repeated hour. off_hour: {node: rows off the hour on those days}."""
    market = key if key in MARKETS else SIDE_MARKET[key]
    nodes = [n for m, n in _node_map_nodes() if m == market]
    days = [d.date() for d in pd.date_range(start, end, freq="D")]
    df = _series_rows(root, key, sealed=sealed)
    counted_n, bad_n, off_n, rows_n = {}, {}, {}, {}
    if df is not None:
        finite, usable = _row_flags(df, key, strict=True)
        aligned = _on_the_hour(df)
        tally = pd.DataFrame({"node": df["node"].to_numpy(), "day": _days(df, market).to_numpy(),
                              "counted": usable & aligned, "bad": ~finite, "off": ~aligned,
                              "rows": 1}).groupby(["node", "day"]).sum()
        counted_n, bad_n, off_n, rows_n = (tally[c].to_dict() for c in ("counted", "bad", "off", "rows"))
    need = {day: _expected_day_hours(key, day) for day in days}
    no_rows, short, over, off_hour = [], {}, {}, {}
    for node in nodes:
        for day in days:
            have, rows, off = (int(n.get((node, day), 0)) for n in (counted_n, rows_n, off_n))
            if off:
                off_hour[node] = off_hour.get(node, 0) + off
            if have == 0:
                no_rows.append((node, day))
            elif have < need[day]:
                short[(node, day)] = (have, need[day], int(bad_n.get((node, day), 0)), off)
            elif rows > need[day]:
                over[(node, day)] = (rows, need[day], off)
    return no_rows, short, over, off_hour


def _day_gaps(root, key, start, end, *, sealed=False):
    """(no_rows, short, over): the pairs that _day_facts finds incomplete, by kind."""
    return _day_facts(root, key, start, end, sealed=sealed)[:3]


def _missing_days(root, key, start, end, *, sealed=False):
    """Every (node, day) pair that _day_gaps reports, in one sorted list."""
    no_rows, short, over = _day_gaps(root, key, start, end, sealed=sealed)
    return sorted(no_rows + list(short) + list(over))


def _fmt_gaps(gaps):
    """'node: first .. last, day; node: ...' with runs of consecutive days joined."""
    by_node = {}
    for node, day in sorted(gaps):
        by_node.setdefault(node, []).append(day)
    parts = []
    for node, days in by_node.items():
        runs, first = [], days[0]
        for prev, day in zip(days, days[1:]):
            if (day - prev).days != 1:
                runs.append(f"{first} .. {prev}" if first != prev else f"{first}")
                first = day
        runs.append(f"{first} .. {days[-1]}" if first != days[-1] else f"{first}")
        parts.append(f"{node}: " + ", ".join(runs))
    return "; ".join(parts)


def _fmt_short(details):
    """'node: day (23 of 24 hours, 1 non-finite price(s), 2 off-hour row(s)), ...; node: ...' for {(node, day): (usable,
    hours, bad, off)}."""
    by_node = {}
    for (node, day), (have, need, bad, off) in sorted(details.items()):
        note = (f"{have} of {need} hours" + (f", {bad} non-finite price(s)" if bad else "")
                + (f", {off} off-hour row(s)" if off else ""))
        by_node.setdefault(node, []).append(f"{day} ({note})")
    return "; ".join(f"{node}: " + ", ".join(days) for node, days in by_node.items())


def _fmt_over(details):
    """'node: day (48 rows for 24 hours, 24 off-hour row(s)), ...; node: ...' for {(node, day): (rows, hours, off)}."""
    by_node = {}
    for (node, day), (rows, need, off) in sorted(details.items()):
        note = f"{rows} rows for {need} hours" + (f", {off} off-hour row(s)" if off else "")
        by_node.setdefault(node, []).append(f"{day} ({note})")
    return "; ".join(f"{node}: " + ", ".join(days) for node, days in by_node.items())


def _allowed_days(allow, pull_start, deploy_end):
    """allow_missing_days as {key: {ISO day}}, keyed by market ('MISO') or settlement series ('fmm', 'nyiso_rtint').
    It must be a dict or None. A typo is an error: an unknown key, a value that is not a list of 'YYYY-MM-DD' days, or
    a day outside the pull."""
    if allow is None:
        return {}
    if not isinstance(allow, dict):
        raise ValueError(f"allow_missing_days must be a dict {{market or series: ['YYYY-MM-DD', ...]}} or None, "
                         f"not {allow!r}")
    first, last = dt.date.fromisoformat(pull_start), dt.date.fromisoformat(deploy_end)
    out = {}
    for key, days in allow.items():
        if key not in OVERRIDE_KEYS:
            raise ValueError(f"allow_missing_days: unknown market or series {key!r}; "
                             f"the keys are {list(OVERRIDE_KEYS)}")
        if isinstance(days, str) or not hasattr(days, "__iter__"):
            raise ValueError(f"allow_missing_days: {key} needs a list of 'YYYY-MM-DD' days, not {days!r}")
        out[key] = set()
        for day in days:
            try:
                parsed = dt.date.fromisoformat(day)
            except (TypeError, ValueError):
                raise ValueError(f"allow_missing_days: {day!r} for {key} is not a 'YYYY-MM-DD' day") from None
            if not first <= parsed <= last:
                raise ValueError(f"allow_missing_days: {day} for {key} is outside the pull "
                                 f"{pull_start} .. {deploy_end}")
            out[key].add(parsed.isoformat())
    return out


def _allowed_chunks(allow, pull_start, deploy_end):
    """allow_incomplete_chunks as {series: {chunk key}}, the series being 'CAISO', 'fmm' or 'nyiso_rtint'. It must be a
    dict or None, of lists of keys, and every key a chunk that this pull requests (_chunk_keys): a typo is an error, not
    a chunk that is quietly never found. An empty list names nothing."""
    if allow is None:
        return {}
    if not isinstance(allow, dict):
        raise ValueError(
            f"allow_incomplete_chunks must be a dict {{series: ['chunk key', ...]}} or None, not {allow!r}")
    out = {}
    for series, keys in allow.items():
        if series not in CHUNKED_SERIES:
            raise ValueError(f"allow_incomplete_chunks: unknown series {series!r}; "
                             f"the series are {list(CHUNKED_SERIES)}")
        if not isinstance(keys, (list, tuple, set, frozenset)):
            raise ValueError(f"allow_incomplete_chunks: {series} needs a list of chunk keys, not {keys!r}")
        known = _chunk_keys(series, pull_start, deploy_end)
        for key in keys:
            if key not in known:
                raise ValueError(f"allow_incomplete_chunks: {key!r} is not a chunk of {series} in this pull "
                                 f"({pull_start} .. {deploy_end}); the keys are as PULL_LOG.txt prints them, e.g. "
                                 f"{known[0]!r}")
        if keys:
            out[series] = set(keys)
    return out


def _cached_archive_hint(root, key, pairs):
    """For the NYISO integrated table (F4): a sentence that names the cached monthly archives (raw/<sha>.bin of the pull
    root) behind the days in `pairs`, the (node, day) pairs, and says how to have one fetched again. An archive has no
    grid to be short of, so it is cached whatever hours it holds and reused as it is on every resume; if NYISO has since
    published the missing hours, deleting its file is how the next resume gets the new one. '' for any other series, and
    for months that are not cached."""
    if key != "nyiso_rtint":
        return ""
    files = []
    for ym in sorted({f"{day:%Y%m}" for _, day in pairs}):
        name = f"raw/{hashlib.sha256(NYISO_URL.format(ym=ym).encode()).hexdigest()}.bin"
        if (Path(root) / name).is_file():
            files.append(f"{name} (the archive of {ym[:4]}-{ym[4:]})")
    if not files:
        return ""
    return (" The archive of a month is kept in raw/ and reused as it is on every resume: if NYISO has since published "
            f"the missing hours, delete {' and '.join(files)} in the pull root, and the next resume fetches "
            f"{'the month' if len(files) == 1 else 'those months'} again.")


def _require_complete(root, key, start, end, allowed):
    """After a step: every node-map node of the market or series has, on every local day of the pull, exactly as many
    rows as the day has hours, all of them usable: on the hour, with finite prices (see _day_facts); except the days in
    `allowed`. The gaps, excused or not, and the rows that are not on the hour go to PULL_LOG.txt; an unexcused gap
    raises MissingDaysError, which the caller lets through before the step is recorded as done."""
    no_rows, short, over, off_hour = _day_facts(root, key, start, end, sealed=True)
    off_total = sum(off_hour.values())
    if off_total:
        _log(root, f"step {key}: {off_total} row(s) are not on the hour (minute or second not 0, UTC) and do not count "
                   "as hours: " + "; ".join(f"{node}: {n}" for node, n in sorted(off_hour.items())))
    if not (no_rows or short or over):
        _log(root, f"step {key}: every node-map node has the hours of every day from {start} to {end}, "
                   "with finite prices, on the hour, and no more rows than the day has hours")
    problems, unexcused = [], []
    for what, said, pairs, text in (
            ("without rows", "no rows for", dict.fromkeys(no_rows), lambda ps: _fmt_gaps(list(ps))),
            ("short of their hours", "fewer usable hours than the day has on", short, _fmt_short),
            ("over their hours", "more rows than the day has hours on", over, _fmt_over)):
        excused = {g: v for g, v in pairs.items() if g[1].isoformat() in allowed}
        missing = {g: v for g, v in pairs.items() if g[1].isoformat() not in allowed}
        if excused:
            _log(root, f"step {key}: {len(excused)} (node, day) pair(s) {what}, allowed by allow_missing_days: "
                       f"{text(excused)}")
        if missing:
            _log(root, f"step {key}: {len(missing)} (node, day) pair(s) {what}: {text(missing)}")
            problems.append(f"{said} {len(missing)} (node, day) pair(s): {text(missing)}")
            unexcused.extend(missing)
    if problems:
        off_note = (f" {off_total} row(s) are not on the hour and were not counted as hours (PULL_LOG.txt says where)."
                    if off_total else "")
        raise MissingDaysError(
            f"{key}: {'; '.join(problems)}.{off_note} The step is not recorded as done, so resume=True runs it again; "
            "name a day in allow_missing_days only if the ISO never published it with finite prices. Inspect the rows "
            "of a day before naming it: a series stamped off the hour, or with an hour twice, is a fault of the pull, "
            f"not a day the ISO skipped.{_cached_archive_hint(root, key, unexcused)}")


# ----------------------------------------------------------------------------- one step
def _takes_incomplete(builder):
    """True for a builder that declares an `incomplete` parameter: the real downloaders of the chunked series. An
    injected builder of the usual (start, end, root) form has no chunks and is not handed one."""
    try:
        return "incomplete" in inspect.signature(builder).parameters
    except (TypeError, ValueError):
        return False


def _run_step(name, builder, start, end, root, incomplete=None):
    """Run one download step with its stdout and stderr captured in <root>/logs/<name>.log, part of the hashed
    record. A failed attempt leaves its traceback there; a resumed attempt appends. A builder that takes `incomplete`
    gets the step's IncompleteChunks, if there is one."""
    extra = {"incomplete": incomplete} if incomplete is not None and _takes_incomplete(builder) else {}
    log_dir = root / "logs"
    log_dir.mkdir(exist_ok=True)
    with open(log_dir / f"{name}.log", "a", buffering=1) as fh, \
            contextlib.redirect_stdout(fh), contextlib.redirect_stderr(fh):
        print(f"=== {name}: attempt started {_iso_now()} for {start} .. {end} ===")
        try:
            builder(start, end, root, **extra)
        except BaseException:
            traceback.print_exc()
            raise


def _trim_step(root, name, pull_start, deploy_end):
    """Trim what the step wrote to the pull's own settlement days and log the counts."""
    key = STEP_KEY[name]
    if key in MARKETS:
        for f in raw_files(dirs=[root], markets=(key,)):
            _log(root, f"trimmed {_trim(f, pull_start, deploy_end)} row(s) outside {pull_start} .. {deploy_end} "
                       f"from {f.name}")
    else:
        for f in sorted((root / key).glob("*.parquet")):                  # the settlement tables have no market column
            _log(root, f"trimmed {_trim(f, pull_start, deploy_end, SIDE_MARKET[key])} row(s) outside {pull_start} .. "
                       f"{deploy_end} from {key}/{f.name}")


# ----------------------------------------------------------------------------- once only: record, lock, state
def completed_pulls():
    """docs/sealed/PULLS.json as a dict keyed by test id (empty if no pull is recorded). It lives outside any data
    root, so a pull is refused whatever data_root it is given."""
    path = _registration.SEALED_DIR / "PULLS.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except ValueError as exc:
        raise RuntimeError(f"{path} is not valid JSON; it records which sealed windows were pulled, so no pull may "
                           "start until it is repaired") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"{path} must hold a JSON object keyed by test id")
    return data


def _record_pull(test_id, entry):
    pulls = completed_pulls()
    pulls[test_id] = entry
    text = json.dumps(pulls, indent=2, sort_keys=True) + "\n"
    _atomic_write(_registration.SEALED_DIR / "PULLS.json", text.encode())


class _PullLock:
    """An exclusive lock on one test's pull, taken with O_CREAT | O_EXCL on docs/sealed/PULL_<test>.lock and removed on
    exit, however the pull ends. A hard kill leaves the file behind; the refusal says so."""

    def __init__(self, test_id):
        self.test_id = test_id
        self.path = _registration.SEALED_DIR / f"PULL_{test_id}.lock"

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            try:
                holder = self.path.read_text().strip() or "empty"
            except OSError:
                holder = "unreadable"
            raise FileExistsError(f"another pull of Test {self.test_id} holds {self.path} ({holder}); if no pull is "
                                  "running (a hard kill leaves the file behind), remove it") from None
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "test": self.test_id,
                                 "started_utc": _iso_now()}))
        return self

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)
        return False


def _require_test_a(pulls):
    """Test B needs Test A's completed pull: its record in PULLS.json, a manifest that still verifies and is the one
    recorded."""
    record = pulls.get("A")
    if record is None:
        raise RuntimeError("Test B needs Test A's completed pull: PULLS.json has no record of it")
    root = Path(record["root"])
    verify_manifest(root)
    if manifest_sha256(root) != record["manifest_sha256"]:
        raise RuntimeError("Test A's manifest differs from the one PULLS.json recorded "
                           f"({record['manifest_sha256'][:12]}...)")
    return record


def _code_fingerprint():
    """SHA-256 of each file of CODE_FILES: this module, every module of the repository it imports, the downloaders it
    runs and the node map it reads."""
    dev = Path(__file__).resolve().parents[1]
    return {rel: sha256_file(dev / rel) for rel in CODE_FILES}


def _load_state(root):
    """(state, why_not): the parsed .state.json of a root, or (None, 'missing' | 'unreadable')."""
    path = root / ".state.json"
    if not path.exists():
        return None, "missing"
    try:
        state = json.loads(path.read_text())
    except ValueError:
        return None, "unreadable"
    if not isinstance(state, dict) or not isinstance(state.get("done"), list):
        return None, "unreadable"
    return state, None


def _write_state(path, state):
    _atomic_write(path, json.dumps(state, sort_keys=True).encode())


def _registration_facts(record):
    """What identifies a verified registration: kept in .state.json, logged at every attempt, recorded in PULLS.json. The
    identifiers of its kind (a DOI and an OSF address, a release address or a local token; the others are None), the one
    to log, the registration id to compare and the kind: osf, public_release or local."""
    return {"protocol_sha256": record["protocol_sha256"], "osf_doi": record.get("osf_doi"),
            "osf_registration": record.get("osf_registration"), "public_record": record.get("public_record"),
            "local_record": record.get("local_record"), "registration_identifier": registration_identifier(record),
            "registration_id": registration_id(record), "registration_kind": registration_kind(record)}


def _same_registration(first, where, fields):
    """A resume continues the first attempt. `first` holds the protocol and the identifier(s) it started under (from
    .state.json, or from the first attempt's log lines): the registration now verified must have the same protocol and
    the same registration (the same registration id). A DOI that appears when an embargo ends, or an OSF address replaced
    by its own DOI, is the same registration; another registration, or the same protocol recorded as another kind of
    registration, is not."""
    protocol, then = first.get("protocol_sha256"), registration_ids(first)
    if protocol is None or not then:
        raise _registration.RegistrationError(f"{where} does not say which registration the pull started under")
    now = registration_ids(fields)
    if protocol != fields["protocol_sha256"]:
        raise _registration.RegistrationError(
            f"{where} was started under protocol {protocol[:12]}...; the registration now verified is protocol "
            f"{fields['protocol_sha256'][:12]}...")
    if then != now:                                  # one id each, as parse_registration guarantees: the same id
        raise _registration.RegistrationError(
            f"{where} was started under the registration {', '.join(registration_identifiers(first))}; the one now "
            f"verified, {', '.join(registration_identifiers(fields))}, is another registration")


def _log_values(root, key):
    path = root / "PULL_LOG.txt"
    lines = path.read_text().splitlines() if path.exists() else []
    return [line.split(": ", 1)[1] for line in lines if line.startswith(f"{key}: ")]


def _logged_registration(root):
    """What the first attempt logged in PULL_LOG.txt: when it started, its protocol and its registration identifiers (an
    identifier that was not given is logged as 'none'). All None for a root whose log has no such line yet."""
    def first(key):
        values = _log_values(root, key)
        return values[0] if values and values[0] != "none" else None

    return {"started_utc": first("pull_started_utc"), "protocol_sha256": first("protocol_sha256"),
            "osf_doi": first("osf_doi"), "osf_registration": first("osf_registration"),
            "public_record": first("public_record"), "local_record": first("local_record")}


def _log_chunk_failure(root, series, exc):
    """Write the one line that tells a later attempt which chunk gave up and why: how many of its attempts brought a
    whole response that the validator refused, and the last problem. allow_incomplete_chunks reads it back
    (_logged_chunk_failures); the key is the one to name there."""
    why = " ".join((exc.reason or str(exc)).split())
    _log(root, f"chunk_failed: {series} {exc.key or '?'} | {exc.validator_failures} of {exc.attempts} attempts failed "
               f"the validator | {why}")


def _logged_chunk_failures(root):
    """{series: {chunk key: reason}} for the chunks that PULL_LOG.txt says gave up after every one of their
    CHUNK_ATTEMPTS attempts brought a whole response that the validator refused. A chunk that gave up with a network
    failure among its attempts is not one of them."""
    found = {}
    for value in _log_values(root, "chunk_failed"):
        head, _, rest = value.partition(" | ")
        counts, _, reason = rest.partition(" | ")
        series, _, key = head.partition(" ")
        if counts == f"{CHUNK_ATTEMPTS} of {CHUNK_ATTEMPTS} attempts failed the validator":
            found.setdefault(series, {})[key] = reason
    return found


def _logged_accepted_chunks(root):
    """{series: {chunk key: reason}} for the chunks that PULL_LOG.txt says were accepted as incomplete, at any
    attempt."""
    found = {}
    for value in _log_values(root, "chunk_accepted"):
        head, _, reason = value.partition(" | ")
        series, _, key = head.partition(" ")
        found.setdefault(series, {})[key] = reason
    return found


def _pull_entry(root, manifest, started, finished, fields, incomplete):
    """The PULLS.json entry of a completed pull: where it is, its manifest hash, when it ran, under which
    registration (the protocol, the identifiers of its kind, whichever were given, its kind and the registration id to
    compare) and the chunks that were accepted although they never came complete, {series: {chunk key: reason}}."""
    return {"root": str(root), "manifest_sha256": manifest, "started_utc": started, "finished_utc": finished,
            **_registration_facts(fields), "registered_utc": fields["registered_utc"],
            "incomplete_chunks": incomplete}


def _adopt(test_id, root, fields):
    """A pull that sealed its record but was killed before PULLS.json was written: verify it, record it, and stop. No
    download happens, and a pull whose files no longer verify is not recorded."""
    verify_manifest(root)
    if not _log_values(root, "pull_finished_utc"):
        raise RuntimeError(f"{root} has a manifest but its PULL_LOG.txt does not say the pull finished; "
                           "it is not adopted")
    _same_registration(_logged_registration(root), f"{root} (PULL_LOG.txt)", fields)
    manifest = manifest_sha256(root)
    state, _ = _load_state(root)
    incomplete = state.get("incomplete_chunks") if state is not None else None
    if not isinstance(incomplete, dict):                      # no state to say: what the log says was accepted
        incomplete = _logged_accepted_chunks(root)
    _record_pull(test_id, _pull_entry(root, manifest, _log_values(root, "pull_started_utc")[0],
                                      _log_values(root, "pull_finished_utc")[-1], fields, incomplete))
    (root / ".state.json").unlink(missing_ok=True)
    return root, manifest


def _reference_arg(reference_dirs):
    if reference_dirs is None:
        return None
    if not isinstance(reference_dirs, (list, tuple)) or not all(isinstance(d, (str, Path)) for d in reference_dirs):
        raise ValueError(f"reference_dirs must be a list of paths, not {reference_dirs!r}")
    return list(reference_dirs)


def _references(test_id, data_root, explicit, prior):
    """(price reference directories, {series: settlement-table reference directories}) for the overlap check. Given
    reference_dirs replace the defaults and drop the settlement tables. By default Test A is compared with the H1 files
    and the development tables under the data root, Test B with Test A's pull as PULLS.json records it."""
    if explicit is not None:
        return [data_root / d for d in explicit], {}
    if test_id == "A":
        return ([data_root / d for d in REFERENCE_DIRS["A"]],
                {series: [data_root / d for d in dirs] for series, dirs in SIDE_REFERENCE_DIRS["A"].items()})
    a_root = Path(prior["root"])
    return [a_root], {series: [a_root / series] for series in SIDE_MARKET}


def pull_sealed(test_id, protocol_sha256, data_root=DATA_ROOT, builders=None, reference_dirs=None, resume=False,
                allow_missing_days=None, check_days=True, *, allow_incomplete_chunks=None):
    """Download one sealed window, once, after the registration check, and return (root, manifest_sha256).

    The registration record is read from es.registration's own paths: there is no way to hand over another. After
    each step the rows are trimmed to the pull's days and every node-map node must have, on every local day, as many
    rows on the hour with finite prices as the day has hours, and no more rows than that (price files, FMM and NYISO
    integrated tables); days an ISO never published, and days the pull gets wrong after inspection, are named in
    `allow_missing_days`, {market or series: ["YYYY-MM-DD", ...]}, and logged.
    `check_days=False` switches the day check off for injected test builders only (never for a real downloader), and
    is logged. A root that exists is resumed with resume=True, from its .state.json and PULL_LOG.txt, which must name
    the registration now verified; a root with files and no usable state is refused, never started over.
    `reference_dirs`: None for the defaults (H1 and the development tables for A, Test A's pull for B) or a list of
    paths under data_root or absolute.

    `allow_incomplete_chunks` (keyword only: a deliberate, named decision), {series: [chunk key, ...]} with the series
    'CAISO', 'fmm' or 'nyiso_rtint' and the keys as the chunk_failed line of PULL_LOG.txt prints them, is for a chunk
    that never comes complete. It needs resume=True and applies only to a chunk that failed the validator in all 8
    attempts in an earlier attempt, as PULL_LOG.txt says; any other chunk is requested and validated as usual. For a
    listed chunk the whole response with the fewest problems is accepted (zero rows if every answer held no data),
    short or with non-finite prices, and never cached, but only if every one of the attempts of this call brought a
    whole response that was refused (no network failure); the missing and non-finite intervals are logged, and the
    chunk and the reason go to .state.json and PULLS.json. The day check then reports the days the chunk leaves short,
    and those need `allow_missing_days`: the two decisions are separate, and both are logged."""
    fields = verify_registration(protocol_sha256)
    pull_start, deploy_start, deploy_end, dirname = SEALED_WINDOWS[test_id]
    allowed = _allowed_days(allow_missing_days, pull_start, deploy_end)
    chunks = _allowed_chunks(allow_incomplete_chunks, pull_start, deploy_end)
    if chunks and not resume:
        raise ValueError("allow_incomplete_chunks needs resume=True: it applies only to a chunk that failed the "
                         "validator in all of its attempts in an earlier attempt of this pull, as PULL_LOG.txt says")
    if not isinstance(check_days, bool):
        raise ValueError(f"check_days must be True or False, not {check_days!r}")
    if builders is None and not check_days:
        raise ValueError("check_days=False is for injected builders (tests, rehearsals); the real downloaders are "
                         "always checked")
    explicit = _reference_arg(reference_dirs)
    b = builders if builders is not None else _default_builders()
    lacking = [name for name in STEPS if name not in b]
    if lacking:
        raise ValueError(f"builders lacks the step(s) {lacking}; the steps are {list(STEPS)}")
    if not check_days:                              # an injected dict may still hold a real downloader (by identity)
        current = _default_builders()
        real = sorted(name for name in STEPS if b[name] is current[name] or b[name] is REAL_DOWNLOADERS[name])
        if real:
            raise ValueError("check_days=False is for injected builders (tests, rehearsals); the real downloaders "
                             f"of the step(s) {real} are always checked")
    data_root = Path(data_root).resolve()
    with _PullLock(test_id):
        pulls = completed_pulls()
        if test_id in pulls:
            done_before = pulls[test_id]
            raise FileExistsError(f"Test {test_id}: PULLS.json records a completed pull at {done_before['root']} "
                                  f"(manifest {done_before['manifest_sha256']}); a sealed window is pulled once")
        prior = _require_test_a(pulls) if test_id == "B" else None
        refs, side_refs = _references(test_id, data_root, explicit, prior)
        root = data_root / dirname
        state_path = root / ".state.json"
        if (root / "MANIFEST.sha256").exists():
            if not resume:
                raise FileExistsError(f"{root} holds a completed pull whose record in PULLS.json was not written; "
                                      "call again with resume=True, which verifies its manifest and adopts it "
                                      "without any download. A sealed window is pulled once")
            return _adopt(test_id, root, fields)
        if root.exists() and not resume:
            raise FileExistsError(f"{root} holds an interrupted pull; call again with resume=True")
        state, why = _load_state(root)
        if _log_values(root, "pull_finished_utc"):
            if state is None or set(state["done"]) != set(STEPS):
                raise FileExistsError(f"{root} holds a finished pull (PULL_LOG.txt has pull_finished_utc) whose "
                                      "manifest and state are gone; a sealed window is pulled once")
        if state is None and root.exists() and any(q.name != ".state.json.tmp" for q in root.iterdir()):
            raise FileExistsError(
                f"{root} holds files from an earlier attempt, but its .state.json is {why}, so the pull cannot be "
                "resumed, and it is not started over on top of them. If the pull is truly broken, move the whole "
                "folder aside (rename it): its PULL_LOG.txt stays there as evidence, PULLS.json is not touched, and "
                "pull_sealed can then start again")
        root.mkdir(parents=False, exist_ok=True)
        first_attempt = state is None
        if first_attempt:
            state = {"done": [], **_registration_facts(fields), "registered_utc": fields["registered_utc"],
                     "started_utc": _iso_now(), "code": _code_fingerprint(), "incomplete_chunks": {}}
            _write_state(state_path, state)
        else:
            logged = _logged_registration(root)           # the log is the durable record of the first attempt
            if logged["protocol_sha256"] is not None:
                _same_registration(logged, f"{root}/PULL_LOG.txt", fields)
                state["started_utc"] = logged["started_utc"] or state["started_utc"]
            _same_registration(state, f"{root}/.state.json", fields)
            state.setdefault("incomplete_chunks", {})
        failed = _logged_chunk_failures(root)           # what earlier attempts gave up on, after all their attempts
        eligible = {series: sorted(keys & set(failed.get(series, ()))) for series, keys in chunks.items()}
        refused = [(series, key) for series, keys in sorted(chunks.items()) for key in sorted(keys)
                   if key not in eligible[series]]
        leftovers = sorted(p.relative_to(root).as_posix() for p in root.rglob("*.tmp"))
        for rel in leftovers:                      # what a kill mid-write left; the lock means nobody else writes
            (root / rel).unlink()
        done = list(state["done"])
        override = json.dumps({k: sorted(v) for k, v in allowed.items()}, sort_keys=True) if allowed else "none"
        chunk_override = json.dumps({k: v for k, v in eligible.items() if v}, sort_keys=True) if any(
            eligible.values()) else "none"
        attempt_started = state["started_utc"] if first_attempt else _iso_now()      # PULLS.json keeps the first one
        _log(root, f"pull_started_utc: {attempt_started}", f"test: {test_id}",
             f"pull: {pull_start} .. {deploy_end} (deployment {deploy_start} .. {deploy_end})",
             f"protocol_sha256: {fields['protocol_sha256']}", f"osf_doi: {fields['osf_doi'] or 'none'}",
             f"osf_registration: {fields['osf_registration'] or 'none'}",
             f"public_record: {fields.get('public_record') or 'none'}",
             f"local_record: {fields.get('local_record') or 'none'}",
             f"registration_identifier: {registration_identifier(fields)}",
             f"registration_id: {registration_id(fields)}",
             f"registration_kind: {registration_kind(fields)}",
             f"registered_utc: {fields['registered_utc']}",
             f"resume: {resume}; steps already done: {done}",
             f"allow_missing_days: {override}",
             f"allow_incomplete_chunks: {chunk_override}",
             f"check_days: {check_days}")
        for series, key in refused:
            _log(root, f"allow_incomplete_chunks: {series} {key} did not fail the validator in all {CHUNK_ATTEMPTS} "
                       "attempts in an earlier attempt, so it is requested and validated as usual")
        code = _code_fingerprint()
        _log(root, *(f"code_sha256: {rel} {digest}" for rel, digest in code.items()))
        if not first_attempt:
            changed = sorted(rel for rel in set(state["code"]) | set(code) if state["code"].get(rel) != code.get(rel))
            _log(root, "code: CHANGED since the first attempt: " + ", ".join(changed) if changed
                 else "code: unchanged since the first attempt")
        if leftovers:
            _log(root, f"removed {len(leftovers)} leftover temp file(s): {', '.join(leftovers)}")
        months = (pull_start[:7], deploy_end[:7])
        days = (pull_start, deploy_end)
        for name, (s, e) in (("NYISO", months), ("ISONE", days), ("MISO", days), ("CAISO", days),
                             ("FMM", days), ("NYISO_INT", months)):
            if name in done:
                continue
            series = STEP_KEY[name]
            incomplete = IncompleteChunks(series, eligible[series], lambda *lines: _log(root, *lines)) \
                if eligible.get(series) else None
            try:
                _run_step(name, b[name], s, e, root, incomplete)
            except ChunkError as exc:
                _log_chunk_failure(root, series, exc)
                raise
            if incomplete is not None:
                for key in sorted(incomplete.keys - set(incomplete.accepted)):
                    _log(root, f"allow_incomplete_chunks: {series} {key} came complete this time; nothing is accepted "
                               "as incomplete for it")
            _trim_step(root, name, pull_start, deploy_end)
            if check_days:
                _require_complete(root, series, pull_start, deploy_end, allowed.get(series, ()))
            if incomplete is not None and incomplete.accepted:          # a series has one step, which is done once
                state["incomplete_chunks"][series] = dict(incomplete.accepted)
            done.append(name)
            state["done"] = done
            _write_state(state_path, state)
            _log(root, f"step {name} finished {_iso_now()}")
        overlap = overlap_check(root, refs, pull_start, deploy_start, _sealed=True, side_references=side_refs)
        overlap.to_csv(root / "OVERLAP_CHECK.csv", index=False)
        _log_overlap(root, overlap)
        coverage = coverage_manifest(root, deploy_start, deploy_end, _sealed=True)
        coverage.to_csv(root / "COVERAGE.csv", index=False)
        flagged = coverage[coverage["nonfinite_rows"] > 0]
        if len(flagged):                          # excused days, or a rehearsal without the day check: say how many
            _log(root, "rows with a non-finite price in the deployment days (series market node rows): "
                       + "; ".join(f"{r.series} {r.market} {r.node} {r.nonfinite_rows}" for r in flagged.itertuples()))
        if coverage.attrs["not_in_node_map"]:
            _log(root, "in the price files but not in the node map, so not in COVERAGE.csv: "
                       f"{coverage.attrs['not_in_node_map']}")
        if coverage.attrs["side_not_in_node_map"]:
            _log(root, "in the settlement tables but not in the node map, so not in COVERAGE.csv: "
                       f"{coverage.attrs['side_not_in_node_map']}")
        finished = _iso_now()
        _log(root, f"pull_finished_utc: {finished}")
        write_manifest(root)          # first: a kill before the next lines still leaves a finished, verifiable pull
        manifest = manifest_sha256(root)
        _record_pull(test_id, _pull_entry(root, manifest, state["started_utc"], finished, fields,
                                          state["incomplete_chunks"]))
        state_path.unlink()
        return root, manifest
