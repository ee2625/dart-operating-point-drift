"""One-shot sealed-test entry point (spec v3 §6.1-6.5; controller Rulings 15, 25-30, 55, 58, 60, 70, 80, 82, 104, 109 and 110).

Refuses to run without a registration record matching the protocol's SHA-256, with bound files
(pre-pull cutoffs, protocol inputs, code manifest, environment pins) that differ from the registered ones, or on code
that differs from the registered manifest unless the departure is recorded (spec §6.5 step 7);
refuses to run twice, and a run that started and crashed counts; never reads the score cache; never
reads data past its own window. Test A deploys 2026-07-01..09-30. Test B deploys 2026-10-01..12-31,
pools H-E and E1-E4 with Test A's stored rows, applies the full Holm step-down, and checks H-B
replication.

What a config can be (Ruling 109, C1, C2, I2, M8; Ruling 116, N4, Mn1; Ruling 122, N2b). The test id is "A" or "B", and
anything else is refused before anything else runs. A config with a sealed arm (status "sealed") must be the registered one:
its two arms, its sealed pulls and its window end equal default_config(test_id)'s, its n_rep is N_REP (2000), it reads
es.registration's protocol and registration paths (cfg.registration must be those paths), it reads the registered development
data (cfg.dev_dirs resolves to DEV_DATA_DIRS), it writes the registered output folder (cfg.out_dir resolves to
default_config(test_id).out_dir) and, for Test B, reads Test A's (cfg.prior_test_dir resolves to default_config("A").out_dir),
and cfg.code_root is the folder the es package was imported from, so that the code manifest describes the code that runs. A
config without a sealed arm is a rehearsal: it reads no sealed pull
(sealed_dirs == []), it reads the registration paths it is given, and its once-only records are written beside its out_dir
(the folder that holds it), never into docs/sealed; its out_dir may not be equal to, inside, or around docs/sealed or the
output folder of a sealed test.

The order of run_sealed. Everything before the lock reads no price row (Ruling 60) and creates nothing: the test id, the run
record and claim file (below), the config, out_dir.exists(), verify_registration, verify_bound (cutoffs, inputs, code
manifest), code_departures, the environment pins (ENV_PINS.json, if the protocol binds it), check_sealed_dirs (each sealed
pull's manifest, bound to the verified registration through docs/sealed/PULLS.json, Ruling 70), the registered inputs, Test
A's recorded output for Test B (which needs the family), the development settlement tables, the free disk space and the write
access to the folders the lock writes in. Then the lock: the claim file, then out_dir and the first line of RUN_LOG.txt, then
the run record; only then does the data load read anything. Everything after the claim is the one run: a crash leaves out_dir
behind, with the line that says where it died, and the next call is refused. An out_dir with no claim file for its test was
never locked: the refusal says that nothing was read and to remove it.

The records (Ruling 104(a), 109 I1, I3, I9; Ruling 116, N3). At the lock, before any folder is made, the run claims its test
id: it creates RUN_<test>.claim with O_CREAT|O_EXCL (so that of two runs started together only one succeeds). After out_dir
and the "started" line (inside the crash handler) it writes a record of the run (started_utc, protocol SHA-256, out_dir,
registration identifier and kind, a fingerprint of the code, the environment) atomically to RUNS.json, keyed by test id. Both are in
docs/sealed (beside out_dir for a rehearsal). run_sealed refuses a test id that has a record or a claim file before it does
anything else, so deleting out_dir cannot re-arm a test, and the claim file is never removed. The claim also yields the token
that the private loader and _execute require: nothing reads the sealed panel without a live claim. When the run finishes, its
record gets finished_utc, its status and the SHA-256 of out_dir's manifest.

Test B and Test A (Ruling 109 I3, I4; Ruling 116, N1, N2, Mn3; Ruling 122, N2b). Test B pools with Test A's output only if
RUNS.json["A"] names that folder, that manifest, that start and this protocol, and the manifest still verifies; each stored file
it reads after its own lock is compared again with the SHA-256 in Test A's manifest, and a mismatch is that analysis' recorded
error. If Test A was never started (no record and no claim file), Test B is refused, whatever folder it is pointed at: a folder
that looks like Test A's output without a record of the run that made it is not one. Test B must be pointed at the folder that
the record names, whatever state the record is in. If that folder is complete (it has its manifest) and the record lacks the
finish fields, Test B is refused with the repair. Test B runs on its own window, with H-E not confirmatory
(STATUS_HE_ALONE), nothing replicating and no step-down, when Test A's output is not available: its coverage rule failed, it was
claimed and left no manifest and no finished record, or it finished without stored P1 rows (results["test_a"]["state"] says
which). H-E is pooled from the two P1 frames alone; when Test A's other stored frames are missing, Test B's E1-E4 cover its own
quarter (SCOPE_B_OWN_QUARTER) and results["test_a"]["missing_kept"] lists what is missing. Test A's p and t_hat are needed only
for an H-B hypothesis that is confirmatory in the family.

After the lock (Ruling 104(c), (f)). Only a failure of the data load itself, before any analysis, aborts the run (and a crash
counts). Every analysis after it runs in its own guard: an exception is logged (type and message, then the traceback, in
RUN_LOG.txt only; the logging is best effort, so a full disk marks the analysis failed in memory and in results.json and does
not end the run), kept under the analysis' name in results["errors"] and in the analysis' own slot of results.json
({"error": "<type>: <message>"}, in results["secondary"][name] for a secondary analysis), and the others go on. An analysis
that needs one that failed records that as its own error. results["status"] is STATUS_COMPLETED, STATUS_SECONDARY or, when an
analysis outside the secondary families failed too, STATUS_ERRORS. A coverage-rule failure is not an error: it is the
registered NOT_RUN path (STATUS_NOT_RUN). There is no label control in the sealed run (the M3 control is a development
analysis, run_labels_pr), so there is none to guard.

NaN bootstrap replicates (Ruling 104(e)). The bootstrap functions stay strict; this module handles the replicates. A contrast
(H-B1, H-B2, PC, and the 18 tests of the audit family) is computed on the finite replicates when block_contrast_bootstrap
refuses a NaN one, and the level bootstraps (E5, H-S) get their intervals on the finite ones; each result records
n_nan_replicates, and is marked approximate when more than 1% are NaN (E5 and H-S are approximate in any case, below).
E1-E4 (estimand_cis) have no NaN replicates to handle: their sums are Python floats, so a degenerate replicate raises
ZeroDivisionError, which is that analysis' recorded error.

E5 and H-S (Ruling 118, M2 of the G1 review). The level bootstraps take the calendar-aligned draw of block_level_bootstrap
(aligned=True: one stationary draw over the quarter's (month, day) positions weights the calibration and the deployment days
alike) as the primary one: t_hat, ci, ci_basic and the NaN count are its, and the result records scheme "aligned"
(SCHEME_E5_HS) and interval_registered INTERVAL_E5_HS. The registered independent draws (aligned=False) are run beside it and
their percentile interval is stored as the diagnostic ci_independent (an E5 row, whose columns are flat: lo_independent,
hi_independent, failures_lo_independent, failures_hi_independent), with their own NaN count (n_nan_replicates_independent;
E5 also n_nan_replicates_failures_independent) and the same NaN handling. Every E5 and H-S result is approximate, whatever the
NaN count: the registered scheme is the independent one, and the aligned draw is not. H-B and PC (block_contrast_bootstrap)
stay on the registered draws.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib
import inspect
import io
import json
import numbers
import os
import secrets
import shutil
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from v2.run_decomposition import restrict_to_node_map

from . import registration as _registration
from .analyses import SEALED, Analysis
from .blocks import BLOCKS
from .bootstrap import (RowCube, _cal_dep_bootstrap, basic_ci, basic_p_one_sided, block_contrast_bootstrap,
                        block_level_bootstrap, percentile_ci, rotation_test)
from .config import ALPHA, DATA_ROOT, DEV_DATA_DIRS, DEV_ROOT, MARKETS, OUT_DIR
from .cutoffs import CUTOFF_BLOCKS, cutoffs_from_scores
from .data import build_panel, load_dev_raw, raw_files
from .guard import settlement_frame
from .policies import activity, p1_fixed, p2_insample, p4_rolling, p5_self_correcting, p6_exact
from .registration import (RegistrationError, _protocol_hashes, registration_id, registration_identifier,
                           registration_kind, sha256_file, verify_bound, verify_registration)
from .run_economics import KEPT, economics_for, estimand_cis, p1_rows_for
from .run_inference import holm
from .run_monitors import evaluate_monitors, monitor_rows
from .scores import CACHED_COLUMNS, feasibility, score_analysis
from .secondary import audit_family, bh_table, economic_family, monitor_family
from .sealed_pull import (SEALED_WINDOWS, completed_pulls, expected_hours, manifest_sha256, verify_manifest,
                          write_manifest)
from .settlement import load_caiso_fmm, load_nyiso_integrated

FAMILY_ORDER = ("H-E", "H-B1", "H-B2")
HB_PAIRS = (("H-B1", "LEV6"), ("H-B2", "ROLL_DISP9"))
LOGIT_BLOCKS = ("DART3", "LEV6", "ROLL_DISP9", "G1_24")
SEED_BLOCKS = CUTOFF_BLOCKS            # spec §6.3: seeds {0, 1, 7, 13, 42} for the primary pipelines
SEEDS = (0, 1, 7, 13)
#: Spec §6.1 coverage rule: at least this many usable nodes, and all four operators. A module constant, so that the one
#: test that runs the late stages on a small synthetic panel can lower it; nothing else changes it.
MIN_USABLE_NODES = 28
#: The two frames economics_for reads in full. Every other score set is slimmed to SLIM as it is built
#: (Ruling 60): 34 frames of 15 columns would not fit beside the panel on an 8 GB machine.
FULL = (("roll", "G1_24", "xgb", 42), ("roll", "G6", "xgb", 42))
SLIM = ["market", "node", "timestamp_utc", "settle_date", "split", "score", "label"]
KEYS = ("market", "node", "split")
#: The two sealed tests. A config with any other test id is refused before anything else runs (Ruling 109, C1).
TEST_IDS = ("A", "B")
#: The registered number of bootstrap replicates. A sealed run uses exactly this many (Ruling 109, C2). A module constant, so
#: that the tests that run the late stages on a small synthetic panel can lower it, as they lower MIN_USABLE_NODES; nothing
#: else does.
N_REP = 2000
#: A run needs at least this many free bytes where out_dir lives and where docs/sealed lives (Ruling 109, I8): it writes about
#: 236 MB, and a full disk would end it after its data were read.
MIN_FREE_BYTES = 2 * 1024 ** 3
#: The libraries whose versions ENV_PINS.json binds with the interpreter's path (Ruling 109, I7), as (name in the file,
#: module to import); "python" is sys.version's first word. The file is {"executable": ..., "versions": {name: version}}.
ENV_LIBRARIES = (("numpy", "numpy"), ("pandas", "pandas"), ("pyarrow", "pyarrow"), ("xgboost", "xgboost"),
                 ("scikit-learn", "sklearn"), ("scipy", "scipy"), ("statsmodels", "statsmodels"))
ENV_PINS_NAME = "ENV_PINS.json"
# The statuses, wordings and scopes that results.json carries (Ruling 109): each is a constant, so that the protocol quotes the
# vocabulary the code writes.
STATUS_COMPLETED = "completed"
STATUS_SECONDARY = "completed with secondary errors"
STATUS_ERRORS = "completed with errors"
STATUS_NOT_RUN = "not run (coverage rule)"
STATUS_HE_INTERIM = "interim (single window; decided after Test B)"
STATUS_HE_DECIDED = "decided (pooled A+B)"
STATUS_HE_ALONE = "not confirmatory (Test A's output is not available; the registered pooled test is unavailable)"
STATUS_HB_ESTIMAND = "estimand (registration rule)"                 # then ": " and the registration rule's reasons, if any
STATUS_HB_REJECTED_A = "rejected in Test A at {alpha}/{m}; replication pending"
STATUS_HB_NOT_REJECTED_A = "not rejected at {alpha}/{m} in Test A; full Holm after Test B"
STATUS_HB_BOTH = "larger operating-point error in both 2026 quarters"
STATUS_HB_Q3_ONLY = "larger operating-point error in Q3 2026; not replicated in Q4"
STATUS_HB_NONE = "no difference in operating-point error detected"
STATUS_HB_NEEDS_HE = "undetermined: the Holm step-down needs H-E, which did not complete"
STATUS_HB_NO_TEST_A = "undetermined: Test A's output is not available, so there is no replication and no step-down"
WORDING_HE_DETECTED = "timing skill within calendar cells detected"
WORDING_HE_NOT_DETECTED = "timing skill within calendar cells not detected"
WORDING_HE_ESTIMAND = "estimand (registration rule): pooled SELC with CI"
WORDING_HE_ALONE = "estimand (Test A's output is not available): SELC of the Test B window alone"
SCOPE_SINGLE = "single window (interim)"
SCOPE_POOLED = "pooled A+B"
SCOPE_B_ALONE = "Test B window alone (Test A's output is not available)"
SCOPE_B_OWN_QUARTER = "Test B window only (Test A's stored rows for E1-E4 are not available)"
SCOPE_E5 = {"A": "Test A window", "B": "Test B window"}             # E5 is this run's own window in both tests
#: The interval that is registered for H-B (Ruling 109 I5, Ruling 110): the basic interval goes with the basic p of a test
#: (Ruling 55(f)); for an estimand the percentile interval is the primary one and the basic one a diagnostic.
INTERVAL_HB_TEST = "basic"
INTERVAL_HB_ESTIMAND = "percentile (primary); basic (diagnostic)"
#: The interval of E5 and H-S (Ruling 118, M2): the percentile interval of the calendar-aligned draw is the primary one; the
#: percentile interval of the registered independent draws is stored beside it (ci_independent) as a diagnostic.
INTERVAL_E5_HS = "percentile, calendar-aligned (primary); independent draws (diagnostic)"
SCHEME_E5_HS = "aligned"
_ONCE = "{out} exists: a sealed test runs once, and a run that started and crashed counts"


@dataclass
class SealedConfig:
    """Where one sealed test reads and writes. The registration, the bound files and the code manifest default to
    es.registration's paths as they are when the config is made (docs/sealed in production; the `registered` fixture
    of the tests points them at a temporary folder). A config with a sealed arm must be default_config(test_id) in its arms,
    pulls and window end, with es.registration's paths and the imported code as its code root; one without is a rehearsal
    (see the module docstring and _check_config)."""
    test_id: str
    rolling: Analysis
    frozen: Analysis
    sealed_dirs: list = field(default_factory=list)
    window_end: str = ""
    prior_test_dir: Path | None = None
    registration: tuple = field(default_factory=lambda: (_registration.PROTOCOL_PATH, _registration.REGISTRATION_PATH))
    out_dir: Path | None = None
    data_root: Path | None = None
    cutoffs: Path = field(default_factory=lambda: _registration.CUTOFFS_PATH)
    inputs: Path = field(default_factory=lambda: _registration.INPUTS_PATH)
    # bound through the protocol like cutoffs and inputs
    code_manifest: Path = field(default_factory=lambda: _registration.SEALED_DIR / "CODE_MANIFEST.sha256")
    code_root: Path = DEV_ROOT                    # what the manifest's relative paths are resolved against
    dev_dirs: tuple = DEV_DATA_DIRS               # the development price directories (tests point this at fakes)


def default_config(test_id):
    """The registered configuration of a sealed test: the arms, pulls and window that run_sealed requires of any config with a
    sealed arm (Ruling 109, C1), and Test A's stored output as Test B's prior."""
    if test_id not in TEST_IDS:
        raise ValueError(f"test_id {test_id!r} is not one of the two sealed tests {list(TEST_IDS)}")
    end, dirname = SEALED_WINDOWS[test_id][2:]
    dirs = [SEALED_WINDOWS["A"][3]] + ([dirname] if test_id == "B" else [])
    roll, frozen = ("SA", "SAF") if test_id == "A" else ("SB", "SBF")
    return SealedConfig(test_id=test_id, rolling=SEALED[roll], frozen=SEALED[frozen], sealed_dirs=dirs, window_end=end,
                        prior_test_dir=OUT_DIR / "sealed" / "A" if test_id == "B" else None,
                        out_dir=OUT_DIR / "sealed" / test_id, data_root=DATA_ROOT)


# ---------------------------------------------------------------------------------------------------------------
# Checks that read no price row
# ---------------------------------------------------------------------------------------------------------------
def _check_test_id(cfg):
    """The two sealed tests are "A" and "B" (Ruling 109, C1). Any other id would name no window, and a once-only record
    keyed by it would not stop the registered test from being read again."""
    if not (isinstance(cfg.test_id, str) and cfg.test_id in TEST_IDS):
        raise ValueError(f"test_id {cfg.test_id!r} is not one of the two sealed tests {list(TEST_IDS)}")


def _is_sealed(cfg):
    """True for a config with a sealed arm: the registered run, which _check_config binds to default_config. False for a
    rehearsal (analog arms), which reads no sealed pull and keeps its records beside its own output."""
    return cfg.rolling.status == "sealed" or cfg.frozen.status == "sealed"


def _imported_code_root():
    """The folder that the es package was imported from (Ruling 109, M8): the code that runs, and so what a code manifest
    must be resolved against."""
    return Path(sys.modules[__package__].__file__).resolve().parents[1]


def _registration_paths(cfg):
    """(protocol, registration record) as this run reads them: es.registration's own paths as they are now for a sealed run,
    never a field of the config (Ruling 109, I2); cfg.registration for a rehearsal."""
    if _is_sealed(cfg):
        return Path(_registration.PROTOCOL_PATH), Path(_registration.REGISTRATION_PATH)
    return tuple(Path(x) for x in cfg.registration)


def _check_window(cfg):
    """The window a config reads: an ISO date no later than the test's own deployment end, and the sealed directories of
    its own pull and of earlier tests only. A later date or a later test's directory would read data past the test's own
    window (spec §6.1)."""
    _check_test_id(cfg)
    try:
        end = dt.date.fromisoformat(cfg.window_end)
    except (TypeError, ValueError):
        raise ValueError(f"window_end {cfg.window_end!r} is not an ISO date (YYYY-MM-DD)") from None
    window = SEALED_WINDOWS[cfg.test_id]
    if end > dt.date.fromisoformat(window[2]):
        raise ValueError(f"window_end {cfg.window_end} is after Test {cfg.test_id}'s own deployment end {window[2]}")
    own = {w[3] for test, w in SEALED_WINDOWS.items() if test <= cfg.test_id}
    stray = sorted(set(cfg.sealed_dirs) - own)
    if stray:
        raise ValueError(f"sealed_dirs {stray} do not belong to Test {cfg.test_id} or an earlier test "
                         f"(its own are {sorted(own)})")
    return end


def _check_sealed_config(cfg, n_rep):
    """A config with a sealed arm is the registered run and nothing else (Ruling 109, C1, C2, I2, M8): the arms, the sealed
    pulls and the window end of default_config(test_id), N_REP bootstrap replicates, es.registration's own paths, the
    development data, the code that was imported, and the folders (Ruling 122, N2b): out_dir is default_config(test_id)'s, and
    Test B's prior_test_dir is Test A's. Anything else (a window ending early, Test B's arms in Test A, no pull at all, 50
    replicates) would spend the run, or read a pull twice under another name."""
    want = default_config(cfg.test_id)

    def shown(x):
        return f"{x.id} (deploy {x.deploy.start}..{x.deploy.end})" if isinstance(x, Analysis) else repr(x)

    differs = [f"{name} {shown(got)} for {shown(expected)}" for name, got, expected in (
        ("rolling arm", cfg.rolling, want.rolling), ("frozen arm", cfg.frozen, want.frozen),
        ("sealed_dirs", list(cfg.sealed_dirs), want.sealed_dirs), ("window_end", cfg.window_end, want.window_end))
        if got != expected]
    if differs:
        raise ValueError(f"a config with a sealed arm must be the registered one, default_config({cfg.test_id!r}): "
                         + "; ".join(differs))
    if n_rep != N_REP:
        raise ValueError(f"a sealed run uses the registered n_rep, {N_REP}, not {n_rep!r}")
    if Path(cfg.out_dir).resolve() != Path(want.out_dir).resolve():
        raise ValueError(f"a sealed run writes the registered output folder, {want.out_dir}, not {cfg.out_dir}: Test B reads "
                         "Test A's output from the folder that Test A's record names, and a Test A that wrote anywhere else "
                         "could leave Test B unable to find it (Ruling 122, N2b)")
    if want.prior_test_dir is not None and Path(cfg.prior_test_dir).resolve() != Path(want.prior_test_dir).resolve():
        raise ValueError(f"a sealed Test B pools with Test A's registered output folder, {want.prior_test_dir}, not "
                         f"{cfg.prior_test_dir}: a Test A output anywhere else is not the one that was registered "
                         "(Ruling 122, N2b)")
    paths = (_registration.PROTOCOL_PATH, _registration.REGISTRATION_PATH)
    if tuple(Path(x).resolve() for x in cfg.registration) != tuple(Path(x).resolve() for x in paths):
        raise ValueError(f"a sealed run reads es.registration's protocol and registration record, {[str(x) for x in paths]}; "
                         f"cfg.registration is {[str(x) for x in cfg.registration]}")
    if tuple(Path(d).resolve() for d in cfg.dev_dirs) != tuple(Path(d).resolve() for d in DEV_DATA_DIRS):
        raise ValueError(f"a sealed run reads the registered development data, {[str(d) for d in DEV_DATA_DIRS]}; cfg.dev_dirs "
                         f"is {[str(d) for d in cfg.dev_dirs]}: other folders, or none, would end the run in its load or at "
                         "the coverage rule (Ruling 116, N4)")
    if Path(cfg.code_root).resolve() != _imported_code_root():
        raise ValueError(f"a sealed run's code_root must be the folder the es package was imported from, "
                         f"{_imported_code_root()}, not {Path(cfg.code_root).resolve()}: the code manifest must describe "
                         "the code that runs")


def _check_rehearsal_place(cfg):
    """A rehearsal's out_dir may not be equal to, inside, or around docs/sealed or the output folder of a sealed test (Ruling
    116, Mn1; the reviewer's R8). Its once-only records sit in the folder that holds its out_dir, and the sealed test refuses to
    run where its own out_dir exists: a rehearsal at outputs/es/sealed/A/rehearsal made outputs/es/sealed/A and blocked Test A."""
    out = Path(cfg.out_dir).resolve()

    def around(p):
        return out == p or p in out.parents or out in p.parents

    sealed_dir = Path(_registration.SEALED_DIR).resolve()
    if around(sealed_dir):
        raise ValueError(f"a rehearsal's out_dir {cfg.out_dir} is in docs/sealed or around it ({sealed_dir}): its records "
                         "would be, or would sit with, the real records of the sealed tests")
    for test_id in TEST_IDS:
        real = Path(default_config(test_id).out_dir).resolve()
        if around(real):
            raise ValueError(f"a rehearsal must not write in or around the output folder of a sealed test ({real}); its "
                             f"out_dir is {cfg.out_dir}: the sealed test would then be refused for finding the folder there")


def _check_config(cfg, n_rep):
    _check_test_id(cfg)
    if cfg.out_dir is None:
        raise ValueError("SealedConfig.out_dir is not set")
    _check_window(cfg)
    if cfg.test_id == "B" and cfg.prior_test_dir is None:
        raise ValueError("Test B pools with Test A: prior_test_dir must name Test A's output directory, or the run "
                         "would be read as a single-window Test A")
    if cfg.test_id == "A" and cfg.prior_test_dir is not None:
        raise ValueError("Test A is the first window and has no prior_test_dir")
    if isinstance(n_rep, bool) or not isinstance(n_rep, numbers.Integral) or n_rep < 1:
        raise ValueError(f"n_rep must be an integer of at least 1, not {n_rep!r}")
    if _is_sealed(cfg):
        _check_sealed_config(cfg, n_rep)
    elif list(cfg.sealed_dirs):
        raise ValueError(f"a rehearsal (no sealed arm) reads no sealed pull: sealed_dirs must be [], not {list(cfg.sealed_dirs)}")
    else:
        _check_rehearsal_place(cfg)


def _manifest_entries(manifest_path):
    """(registered SHA-256, relative path) for each line of a code manifest; a malformed line is refused."""
    out = []
    for line in Path(manifest_path).read_text().splitlines():
        if not line.strip():
            continue
        registered, sep, rel = line.partition("  ")
        if not (registered and sep and rel):
            raise RegistrationError(f"malformed line in the code manifest {manifest_path}: {line!r} "
                                    "(expected '<sha256>  <relative path>')")
        out.append((registered, rel))
    return out


def _current_sha(root, rel):
    p = Path(root) / rel
    return sha256_file(p) if p.is_file() else None


def code_departures(manifest_path, root):
    """Files whose SHA-256 differs from the registered code manifest (spec §6.5 step 7; §6.3 "Test B runs the
    registered code"), as [{"file", "registered", "current"}]; a missing file has current None. A file the
    manifest does not list is not a departure: the registered code is the listed set, and a new import shows up
    as a change in the file that imports it."""
    out = []
    for registered, rel in _manifest_entries(manifest_path):
        current = _current_sha(root, rel)
        if current != registered.lower():
            out.append({"file": rel, "registered": registered, "current": current})
    return out


def _code_fingerprint(manifest_path, root):
    """What identifies the code a run used: the code manifest's own SHA-256 and the current SHA-256 of each file it lists
    (None for a missing one)."""
    return {"code_manifest_sha256": sha256_file(manifest_path),
            "files": {rel: _current_sha(root, rel) for _, rel in _manifest_entries(manifest_path)}}


def _logged_registration_ids(root):
    """The registration ids (case folded) that a pull logged in its PULL_LOG.txt, at every attempt ('none' is no id). The
    log is part of the hashed record, so check_sealed_dirs reads it only after verify_manifest."""
    path = Path(root) / "PULL_LOG.txt"
    found = set()
    for line in (path.read_text().splitlines() if path.is_file() else []):
        key, sep, value = line.partition(": ")
        if sep and key == "registration_id" and value.strip() not in ("", "none"):
            found.add(value.strip().casefold())
    return found


def _utc(value, what):
    if not value:
        raise RegistrationError(f"{what} has no time")
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def check_sealed_dirs(cfg, record):
    """Verify each sealed pull's manifest (exact file set and hashes, Ruling 58) and bind it to the verified
    registration `record` (Ruling 70), without reading a price row. The completed-pull record that
    pull_sealed wrote to docs/sealed/PULLS.json (es.sealed_pull.completed_pulls) must name this root (both resolved) and
    this manifest SHA-256, the record's protocol_sha256 and its registration id (registration_id(record) against the
    entry's registration_id, whatever the case: for an OSF registration the id stays the same when a DOI is added after
    an embargo; a public release and a local record have ids of their own, so a pull made under one kind is not read under
    another), and a start time at or after its registered_utc. The pull's own PULL_LOG.txt, part of the hashed record,
    must not name another registration id than the entry. Returns {dirname: SHA-256 of its MANIFEST.sha256}."""
    manifests = {}
    if not cfg.sealed_dirs:
        return manifests
    pulls = completed_pulls()
    by_root = {Path(p["root"]).resolve(): p for p in pulls.values() if isinstance(p, dict) and "root" in p}
    now = registration_id(record).casefold()
    for d in cfg.sealed_dirs:
        root = (Path(cfg.data_root or DATA_ROOT) / d).resolve()
        verify_manifest(root)
        sha = manifest_sha256(root)
        p = by_root.get(root)
        if p is None:
            raise RegistrationError(f"{d}: PULLS.json has no completed pull record for {root}; only a pull that "
                                    "pull_sealed completed under the registration can be read")
        if p.get("manifest_sha256") != sha:
            raise RegistrationError(f"{d}: its manifest is not the one recorded when the pull completed")
        if str(p.get("protocol_sha256", "")).lower() != record["protocol_sha256"]:
            raise RegistrationError(f"{d}: pulled under a different protocol than the verified registration")
        then = str(p.get("registration_id") or "").strip().casefold()
        if not then:
            raise RegistrationError(f"{d}: its record in PULLS.json names no registration_id, so it is not the record of a "
                                    "pull_sealed pull under the verified registration")
        if then != now:
            raise RegistrationError(f"{d}: pulled under a different registration (registration id {then}) than the "
                                    f"verified one ({now})")
        logged = _logged_registration_ids(root) - {now}
        if logged:
            raise RegistrationError(f"{d}: its PULL_LOG.txt names a different registration (registration id "
                                    f"{', '.join(sorted(logged))}) than the verified one ({now})")
        if _utc(p.get("started_utc"), f"{d}: the pull record's started_utc") < _utc(record["registered_utc"],
                                                                                  "the registration"):
            raise RegistrationError(f"{d}: the pull started before the registration")
        manifests[d] = sha
    return manifests


def _safe(name):
    return name.replace("@", "_at_").replace("*", "star")


#: How Test A stands when Test B starts (results["test_a"]["state"]): its output is verified and pooled, it failed the
#: coverage rule, it finished without the stored P1 rows that H-E is pooled from, it was claimed and left no finished record
#: (and no manifest), or it was claimed and recorded nothing. The wording of the results says "output not available"; this
#: state says why.
PRIOR_POOLED, PRIOR_NOT_RUN = "pooled", STATUS_NOT_RUN
PRIOR_NO_P1 = "finished, without stored P1 rows"
PRIOR_UNFINISHED, PRIOR_UNRECORDED = "unfinished", "claimed, not recorded"


def _repair_message(prior, runs_path, rec):
    """The refusal of Test B for a complete Test A whose record lacks its finish fields (Ruling 116, N2): it names the fields
    that _record_finished writes and where each value comes from, with the values as they stand."""
    lacking = [k for k in ("finished_utc", "manifest_sha256") if not rec.get(k)]
    try:
        res = json.loads((Path(prior) / "results.json").read_text())
        finished, status = res.get("finished_utc"), res.get("status")
    except (OSError, ValueError, AttributeError):
        finished = status = "results.json cannot be read"
    return (f"{prior}: Test A's output is complete (it has a MANIFEST.sha256), but its record in {runs_path} lacks {lacking}: "
            "the update that _record_finished makes did not reach it. Test B would run alone and lose the pooled H-E, so it is "
            "refused before its lock. Repair the record by hand: in RUNS.json[\"A\"] set finished_utc to the finished_utc of "
            f"{Path(prior) / 'results.json'} ({finished}), status to the status there ({status}) and manifest_sha256 to the "
            f"SHA-256 of {Path(prior) / 'MANIFEST.sha256'} ({manifest_sha256(prior)}); then run Test B again.")


def _check_prior(cfg, record, confirmatory):
    """Test B pools with Test A's stored rows. Everything it needs from Test A's output is checked here, before the lock, so
    that a missing or altered file cannot spend Test B's run. Returns (Test A's results.json, a description of where Test A
    stands for results["test_a"]); (None, None) for Test A.

    Test A's record is RUNS.json["A"] (docs/sealed, or beside Test A's folder for a rehearsal, Ruling 109 I3). If Test A was
    never started (no record and no claim file), Test B is refused, whatever folder it is pointed at: a folder that looks like
    Test A's output without a record of the run is not one. Test B must be pointed at the folder that the record names, in
    every state of the record (Ruling 122, N2b). If Test A was claimed and left no manifest and no finished record (it crashed,
    or its record was never written), or its coverage rule failed (NOT_RUN), there is nothing to pool and Test B still runs, on
    its own window, and H-E is not confirmatory (Ruling 109, I4). If the folder that the record names is complete (it has its
    manifest) but the record lacks the finish fields, Test B is refused with the repair (Ruling 116, N2), whichever folder it
    was pointed at (N2b adds the pointer to the right one): running alone would lose the pooled H-E for a state that no data
    read caused. If Test A finished, its output must be the run that the record names: the same folder, the same manifest
    SHA-256 (and the manifest still verifies), the same start, the same protocol, and p and t_hat of each H-B hypothesis that
    is confirmatory in the family (an estimand needs none). H-E is pooled from the stored P1 rows alone (Ruling 116, N1):
    without kept/P1 Test B runs alone (PRIOR_NO_P1); without the frames E1-E4 are computed from (the other KEPT names) it
    pools H-E and its E1-E4 cover its own quarter, and the state lists what is
    missing ("missing_kept")."""
    if cfg.test_id != "B":
        return None, None
    store = _store(cfg, "A")
    prior = Path(cfg.prior_test_dir)
    rec = recorded_runs(store).get("A")
    claim = _claim_path(store, "A")
    if rec is None:
        if claim.exists():
            return None, {"state": PRIOR_UNRECORDED, "claim": str(claim)}
        orphan = f" ({prior} exists, but nothing records the run that made it)" if prior.exists() else ""
        raise ValueError(f"Test A has not been run: {_runs_path(store)} has no record of it and there is no {claim.name}; "
                         f"Test B comes after Test A{orphan}")
    if not isinstance(rec, dict):
        raise RuntimeError(f"{_runs_path(store)}: the record of Test A is not an object")
    recorded = Path(str(rec.get("out_dir")))                 # the folder that Test A's run wrote (Ruling 122, N2b)
    pointed_right = recorded.resolve() == prior.resolve()
    unfinished = not rec.get("finished_utc") or not rec.get("manifest_sha256")
    if unfinished and (recorded / "MANIFEST.sha256").exists():
        # the output the record names is complete, wherever Test B was pointed: running alone would lose the pooled H-E
        raise RuntimeError(_repair_message(recorded, _runs_path(store), rec)
                           + ("" if pointed_right else f" Point Test B's prior_test_dir at {recorded}, not {prior}."))
    if not pointed_right:
        raise RegistrationError(f"{prior}: Test A's record in {_runs_path(store)} names the folder {rec.get('out_dir')}")
    if unfinished:
        return None, {"state": PRIOR_UNFINISHED, "started_utc": rec.get("started_utc"), "out_dir": rec.get("out_dir")}
    if str(rec.get("protocol_sha256", "")).lower() != record["protocol_sha256"]:
        raise RegistrationError(f"{prior}: Test A ran under a different protocol than the verified registration")
    verify_manifest(prior)
    if manifest_sha256(prior) != rec["manifest_sha256"]:
        raise RegistrationError(f"{prior}: its manifest is not the one recorded when Test A finished (changed since, or "
                                "not the output of the recorded run)")
    try:
        res = json.loads((prior / "results.json").read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"{prior}: Test A's results.json cannot be read ({exc})") from None
    if not isinstance(res, dict) or res.get("test") != "A":
        raise ValueError(f"{prior}: its results.json is not Test A's")
    if res.get("started_utc") != rec.get("started_utc"):
        raise RegistrationError(f"{prior}: its results.json started {res.get('started_utc')}, the recorded run "
                                f"{rec.get('started_utc')}")
    if str(res.get("protocol_sha256", "")).lower() != record["protocol_sha256"]:
        raise RegistrationError(f"{prior}: Test A ran under a different protocol than the verified registration")
    state = {"started_utc": rec["started_utc"], "finished_utc": rec["finished_utc"], "manifest_sha256": rec["manifest_sha256"]}
    if not (res.get("coverage") or {}).get("ran"):
        return None, {"state": PRIOR_NOT_RUN, **state}
    missing = [name for name in KEPT if not (prior / "kept" / f"{_safe(name)}.parquet").exists()]
    if "P1" in missing:
        return None, {"state": PRIOR_NO_P1, **state, "missing_kept": missing}
    for hid, _ in HB_PAIRS:
        absent = [k for k in ("p", "t_hat") if hid in confirmatory and k not in (res.get(hid) or {})]
        if absent:
            raise ValueError(f"{prior}: in Test A's results.json {hid}, a confirmatory hypothesis, lacks {absent}")
    return res, {"state": PRIOR_POOLED, **state, **({"missing_kept": missing} if missing else {})}


def _read_inputs(path):
    """The registered inputs the run reads after its lock, checked before it: eta, and for each hypothesis of the family
    the keys that are read (H-E's interim_approximate only in Test A, but required in both: one file serves both)."""
    inputs = json.loads(Path(path).read_text())
    if not isinstance(inputs, dict):
        raise ValueError(f"{path}: protocol inputs must be a JSON object, not {type(inputs).__name__}")
    try:
        eta = float(inputs["eta"])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"{path}: 'eta' is missing or not a number ({inputs.get('eta')!r})") from None
    family = inputs.get("family")
    if not isinstance(family, dict):
        raise ValueError(f"{path}: there is no registered family (family is {family!r}); protocol_inputs.json must be "
                         "stage-complete")
    for hid in FAMILY_ORDER:
        entry = family.get(hid)
        needed = ("confirmatory", "reasons", "approximate") + (("interim_approximate",) if hid == "H-E" else ())
        absent = [k for k in needed if not isinstance(entry, dict) or k not in entry]
        if absent:
            raise ValueError(f"{path}: family[{hid!r}] lacks {absent}")
    return eta, family


def live_environment():
    """What the run is running on: the interpreter's path and the versions of python and of the libraries ENV_PINS.json pins
    (Ruling 109, I7), in the form of that file: {"executable": ..., "versions": {name: version}}."""
    versions = {"python": sys.version.split()[0]}
    for name, module in ENV_LIBRARIES:
        versions[name] = str(importlib.import_module(module).__version__)
    return {"executable": sys.executable, "versions": versions}


def _check_environment(protocol_path):
    """Compare the running environment with ENV_PINS.json (Ruling 109, I7), before the lock. The file is
    es.registration's SEALED_DIR / ENV_PINS.json, and it counts when the protocol binds it (prints a hash line for it): then it
    must exist, be the registered file (verify_bound), and agree with the live interpreter path and library versions on every
    one of them, or the run is refused. A protocol that does not bind it pins nothing. Returns the live environment, with
    "pins": "matched", or "not bound" (and "pins_sha256" when matched), for results.json and RUNS.json."""
    live = live_environment()
    pins_path = Path(_registration.SEALED_DIR) / ENV_PINS_NAME
    if not _protocol_hashes(Path(protocol_path).read_text(), ENV_PINS_NAME):
        return {**live, "pins": "not bound", "pins_sha256": None}
    if not pins_path.is_file():
        raise RegistrationError(f"the protocol binds {ENV_PINS_NAME}, but {pins_path} does not exist")
    bound = verify_bound(protocol_path, [pins_path])
    try:
        pins = json.loads(pins_path.read_text())
        pinned = {"executable": pins["executable"], **pins["versions"]}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RegistrationError(f"{pins_path} is not an environment pin file ({type(exc).__name__}: {exc}); expected "
                                '{"executable": ..., "versions": {name: version}}') from None
    now = {"executable": live["executable"], **live["versions"]}

    def same(name):
        if name != "executable":
            return pinned.get(name) == now[name]
        # the interpreter is one binary however it is spelled (python, python3, python3.12; a link): compare what they resolve
        # to (Ruling 116, Mn2)
        return isinstance(pinned.get(name), str) and os.path.realpath(pinned[name]) == os.path.realpath(now[name])

    differences = [f"{name}: pinned {pinned.get(name)!r}, running {now[name]!r}" for name in now if not same(name)]
    if differences:
        raise RegistrationError(f"the running environment differs from the pins the protocol binds ({pins_path}): "
                                + "; ".join(differences))
    return {**live, "pins": "matched", "pins_sha256": bound[str(pins_path)]}


def _check_side_tables(cfg):
    """The development settlement tables must be there (Ruling 109, M9). Without caiso_fmm_dev or nyiso_rtint_dev the loaders
    return None and the premium direction is set from the 5-minute settlement, silently; that must not be a run's first
    surprise after its data are read. A listing of the folders only: no row is read."""
    base = Path(cfg.data_root or DATA_ROOT)
    missing = [f"{base / sub}/{pattern}" for sub, pattern in (("caiso_fmm_dev", "caiso_fmm_*.parquet"),
                                                               ("nyiso_rtint_dev", "nyiso_rtint_*.parquet"))
               if not any((base / sub).glob(pattern))]
    if missing:
        raise ValueError(f"the development settlement tables are missing ({', '.join(missing)}): the run would set the "
                         "premium direction from the 5-minute settlement without a word")


def _nearest_existing(path):
    p = Path(path).resolve()
    while not p.exists() and p != p.parent:
        p = p.parent
    return p


def _free_bytes(path):
    """Free bytes on the filesystem that holds `path` (or its nearest existing parent)."""
    return shutil.disk_usage(_nearest_existing(path)).free


def _check_writable(cfg):
    """Refuse before the lock if the folder of the once-only records (docs/sealed, or a rehearsal's parent folder) or out_dir's
    parent cannot be written (Ruling 116, N3; the reviewer's R7c): the claim is taken at the lock, and a PermissionError
    there, after the lock decision, would have spent the test (docs/sealed locked read-only after the pull, to protect the
    records, was enough). Nothing has been created when this refuses."""
    for what, path in (("the once-only records", _store(cfg)), ("out_dir", Path(cfg.out_dir).parent)):
        d = _nearest_existing(path)
        if not (d.is_dir() and os.access(d, os.W_OK | os.X_OK)):
            raise PermissionError(f"{d} is not writable, and the run writes {what} there: it would fail after the lock was taken "
                                  "and the test would be spent without a row read. Make it writable and run again (nothing was "
                                  "created)")


def _check_free_space(cfg):
    """Refuse with less than MIN_FREE_BYTES free where out_dir lives or where docs/sealed lives (Ruling 109, I8): a run writes
    about 236 MB, and a full disk after the data were read would spend it."""
    for what, path in (("out_dir", cfg.out_dir), ("the sealed records", _registration.SEALED_DIR)):
        free = _free_bytes(path)
        if free < MIN_FREE_BYTES:
            raise RuntimeError(f"only {free / 1e9:.2f} GB free where {what} lives ({path}); a sealed run needs at least "
                               f"{MIN_FREE_BYTES / 1e9:.2f} GB, because a full disk after the data are read would spend it")


# ---------------------------------------------------------------------------------------------------------------
# The once-only records (controller Rulings 104(a) and 109 I1, I3, I9)
# ---------------------------------------------------------------------------------------------------------------
def _store(cfg, test_id=None):
    """The folder that holds the once-only records (RUNS.json and RUN_<test>.claim) of `test_id` (default: this run's own
    test). For a sealed run it is es.registration's SEALED_DIR as it is now (docs/sealed in production). For a rehearsal it is
    the folder that holds the test's output folder, so that a rehearsal can never write the real records (Ruling 109, C1);
    it is not out_dir itself because RUNS.json holds the SHA-256 of out_dir's own manifest, and a file inside out_dir would be
    listed by that manifest. None if a rehearsal has no folder to name yet."""
    if _is_sealed(cfg):
        return Path(_registration.SEALED_DIR)
    folder = cfg.out_dir if test_id in (None, cfg.test_id) else cfg.prior_test_dir
    return None if folder is None else Path(folder).resolve().parent


def _runs_path(store=None):
    return (Path(_registration.SEALED_DIR) if store is None else Path(store)) / "RUNS.json"


def _claim_path(store, test_id):
    return Path(store) / f"RUN_{test_id}.claim"


def recorded_runs(store=None):
    """RUNS.json as a dict keyed by test id (empty if no run is recorded): docs/sealed/RUNS.json, or the file in `store`.
    Like PULLS.json it lives outside every output folder, so deleting out_dir cannot re-arm a test."""
    path = _runs_path(store)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except ValueError as exc:
        raise RuntimeError(f"{path} is not valid JSON; it records which sealed tests have run, so no run may start until it "
                           "is repaired") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"{path} must hold a JSON object keyed by test id")
    return data


def _refuse_record(store, test_id):
    """The recorded runs, unless `test_id` already has one: a sealed test runs once, whatever its out_dir."""
    runs = recorded_runs(store)
    if test_id in runs:
        r = runs[test_id] if isinstance(runs[test_id], dict) else {}
        raise FileExistsError(f"Test {test_id}: {_runs_path(store)} records a run started {r.get('started_utc')} with out_dir "
                              f"{r.get('out_dir')}; a sealed test runs once, whatever its out_dir, and a run that started "
                              "and crashed counts")
    return runs


def _refuse_recorded(cfg):
    """Refuse a test id that has a record in RUNS.json or a claim file (Ruling 104(a), 109 I9): the first thing run_sealed does
    after it has checked the test id. Returns the recorded runs."""
    store = _store(cfg)
    if store is None:
        return {}
    runs = _refuse_record(store, cfg.test_id)
    claim = _claim_path(store, cfg.test_id)
    if claim.exists():
        raise FileExistsError(f"Test {cfg.test_id}: {claim} exists; a sealed test runs once, whatever its out_dir, and a run "
                              "that started and crashed counts (the claim is made at the lock and is never removed)")
    return runs


def _fsync_dir(path):
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass                                    # the file is in place; syncing the folder is best effort


def _atomic_write_json(path, obj):
    """`obj` as JSON in `path`, through a temporary file in the same folder, fsync'd, and os.replace: a reader or a crash
    never finds half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as fh:
            fh.write((json.dumps(obj, indent=2, sort_keys=True, default=_json_default) + "\n").encode())
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    _fsync_dir(path.parent)


_MINT = object()


class _Claim:
    """The proof that the run-once lock was taken (Ruling 109, I1): only _claim_run makes one, the private loader and _execute
    require one, and it ends with the run. It names the test, the output folder and the claim file, whose content must hold
    the same nonce, so a lookalike, a claim of another run or a token kept after the run is refused."""
    __slots__ = ("test_id", "out", "path", "nonce", "live")

    def __init__(self, mint, test_id, out, path, nonce):
        if mint is not _MINT:
            raise TypeError("a claim is made by _claim_run, at the lock, and by nothing else")
        self.test_id, self.out, self.path, self.nonce, self.live = test_id, out, path, nonce, True

    def close(self):
        self.live = False


def _require_claim(cfg, claim):
    """Reading the sealed panel is for the run that holds the lock, and for nothing else (Ruling 109, I1)."""
    what = "reading the sealed panel needs the live claim that run_sealed takes at its lock"
    if not isinstance(claim, _Claim):
        raise PermissionError(f"{what}; this is {type(claim).__name__}")
    if not claim.live:
        raise PermissionError(f"{what}; this claim belongs to a run that has ended")
    if claim.test_id != cfg.test_id or cfg.out_dir is None or claim.out != Path(cfg.out_dir).resolve():
        raise PermissionError(f"{what}; this claim is for Test {claim.test_id} in {claim.out}, not for this config")
    try:
        held = json.loads(claim.path.read_text()).get("nonce")
    except (OSError, ValueError, AttributeError):
        held = None
    if held != claim.nonce:
        raise PermissionError(f"{what}; {claim.path} does not hold this claim")


def _take_claim(cfg, started, out):
    """The lock on the test id (Ruling 104(a), 109 I9, 116 N3): RUN_<test>.claim is created with O_CREAT|O_EXCL, so that of two
    runs started together exactly one gets it; the file is never removed. It is the first thing the lock does, before out_dir
    is made: with no claim file nothing has happened (an out_dir that exists then is removable), and with one the run counts.
    Returns the claim, which the loader and _execute require (I1). An existing claim file refuses the run."""
    store = _store(cfg)
    store.mkdir(parents=True, exist_ok=True)
    path = _claim_path(store, cfg.test_id)
    nonce = secrets.token_hex(16)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        raise FileExistsError(f"Test {cfg.test_id}: {path} exists; a sealed test runs once, whatever its out_dir, and a run "
                              "that started and crashed counts (another run holds the claim)") from None
    with os.fdopen(fd, "wb") as fh:
        fh.write((json.dumps({"test": cfg.test_id, "started_utc": started, "out_dir": str(Path(out).resolve()),
                              "pid": os.getpid(), "nonce": nonce}, indent=2, sort_keys=True) + "\n").encode())
        fh.flush()
        os.fsync(fh.fileno())
    _fsync_dir(store)
    return _Claim(_MINT, cfg.test_id, Path(out).resolve(), path, nonce)


def _write_run_record(claim, cfg, record, started, out, fingerprint, n_rep, allow_departure, n_departures, environment=None):
    """The run record in RUNS.json, after the claim and before any price row is read; a record whose claim file was deleted
    refuses the run too."""
    store = claim.path.parent
    runs = _refuse_record(store, cfg.test_id)
    runs[cfg.test_id] = {"started_utc": started, "protocol_sha256": record["protocol_sha256"],
                         "out_dir": str(Path(out).resolve()), "registration_identifier": registration_identifier(record),
                         "registration_kind": registration_kind(record),
                         "registered_utc": record["registered_utc"], "code": fingerprint, "n_rep": n_rep,
                         "allow_departure": bool(allow_departure), "departures": n_departures,
                         "environment": environment}
    _atomic_write_json(_runs_path(store), runs)
    _log(out, f"claim {claim.path}", f"run recorded in {_runs_path(store)}")


def _claim_run(cfg, record, started, out, fingerprint, n_rep, allow_departure, n_departures, environment=None):
    """_take_claim and _write_run_record in one (run_sealed does them apart, with out_dir made between them)."""
    claim = _take_claim(cfg, started, out)
    _write_run_record(claim, cfg, record, started, out, fingerprint, n_rep, allow_departure, n_departures, environment)
    return claim


def _record_finished(claim, results):
    """Add finished_utc, the run's status and the SHA-256 of out_dir's manifest to the run's record (Ruling 109, I3): the
    number that ties Test A's output to the run that made it, which Test B checks. Called after the manifest is written, so a
    failure here does not touch the finished output; the message gives the fields, for a repair by hand."""
    store = claim.path.parent
    fields = {"finished_utc": results["finished_utc"], "status": results.get("status"),
              "manifest_sha256": manifest_sha256(claim.out)}
    try:
        runs = recorded_runs(store)
        rec = runs.get(claim.test_id)
        if not isinstance(rec, dict) or rec.get("started_utc") != results["started_utc"]:
            raise RuntimeError(f"the record of Test {claim.test_id} in {_runs_path(store)} is not this run's")
        rec.update(fields)
        _atomic_write_json(_runs_path(store), runs)
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(f"Test {claim.test_id} finished and its output in {claim.out} is complete, but {_runs_path(store)} "
                           f"could not be updated ({type(exc).__name__}: {exc}); add these fields to its record by hand: "
                           f"{json.dumps(fields)}") from exc


# ---------------------------------------------------------------------------------------------------------------
# The data
# ---------------------------------------------------------------------------------------------------------------
def _load_sealed_raw(cfg, protocol_sha256, claim):
    """The development files plus every pull in cfg.sealed_dirs, cut at cfg.window_end (local settlement days).

    Private to the run (Ruling 109, I1): it needs the live claim that run_sealed takes at its lock (_claim_run), the same one
    that _execute needs, and it is called by _execute and nothing else. It is also gated on its own, so the registration and
    intact, bound manifests are checked again here.

    Accounting (Ruling 82, Ruling 109 M2, M4): rows_read is the rows of all the files, the development files as the
    development loader hands them over (it restricts them to the node map and drops repeated hours itself:
    dev_duplicates_dropped is its count of the second). Then, each counted on its own as raw.attrs: node_map_dropped (nodes
    outside the node map, in the sealed pulls), duplicates_dropped (the first row of a market, node and hour wins, the
    development file: mostly the overlap days of a pull and the development files) and beyond_window_end. Then the rows
    inside the window whose timestamp is not on the hour are dropped and counted per market and node in
    raw.attrs["off_hour_dropped"] ({"MARKET/node": rows}), because the panel build refuses them; then every row whose da_lmp
    or rt_lmp is not finite, in raw.attrs["nonfinite_dropped"]. It never raises on any of them: a raise after the data are
    read would spend the one run (the development data has none)."""
    _require_claim(cfg, claim)
    end = _check_window(cfg)
    record = verify_registration(protocol_sha256, *_registration_paths(cfg))
    manifests = check_sealed_dirs(cfg, record)
    base = Path(cfg.data_root or DATA_ROOT)
    dev = load_dev_raw(cfg.dev_dirs)
    frames = [dev]
    for d in cfg.sealed_dirs:
        for f in raw_files(dirs=[base / d]):
            df = pd.read_parquet(f)
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)     # MISO files hold naive UTC
            frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    raw["timestamp_utc"] = pd.to_datetime(raw["timestamp_utc"], utc=True)
    n_read = len(raw)
    raw = restrict_to_node_map(raw)
    n_node_map = n_read - len(raw)
    n_mapped = len(raw)
    raw = raw.drop_duplicates(["market", "node", "timestamp_utc"], keep="first")
    n_duplicates = n_mapped - len(raw)
    s = settlement_frame(raw["timestamp_utc"], raw["market"])
    inside = (s["settle_date"] <= end).to_numpy()
    n_beyond = int((~inside).sum())
    raw = raw[inside]
    stamps = raw["timestamp_utc"]
    off_hour = (stamps != stamps.dt.floor("h")).to_numpy()
    off_dropped = raw[off_hour].groupby(["market", "node"]).size()
    raw = raw[~off_hour]
    finite = np.ones(len(raw), dtype=bool)
    for c in ("da_lmp", "rt_lmp"):
        finite &= np.isfinite(pd.to_numeric(raw[c], errors="coerce").to_numpy(dtype=float, na_value=np.nan))
    dropped = raw[~finite].groupby(["market", "node"]).size()
    raw = raw[finite].sort_values(["market", "node", "timestamp_utc"]).reset_index(drop=True)
    raw.attrs["rows_read"], raw.attrs["rows_kept"] = n_read, len(raw)
    raw.attrs["node_map_dropped"], raw.attrs["duplicates_dropped"] = n_node_map, n_duplicates
    raw.attrs["dev_duplicates_dropped"] = int(dev.attrs.get("duplicates_dropped", 0))
    raw.attrs["beyond_window_end"] = n_beyond
    raw.attrs["sealed_manifests"] = manifests
    raw.attrs["off_hour_dropped"] = {f"{m}/{n}": int(k) for (m, n), k in off_dropped.items()}
    raw.attrs["nonfinite_dropped"] = {f"{m}/{n}": int(k) for (m, n), k in dropped.items()}
    return raw


def coverage(panel, analysis):
    dep = panel[analysis.deploy.mask(panel).to_numpy()]
    rows = []
    for market in MARKETS:
        expected = expected_hours(market, analysis.deploy.start, analysis.deploy.end)
        counts = dep[dep["market"] == market].groupby("node").size()
        for node in sorted(panel.loc[panel["market"] == market, "node"].unique()):
            hours = int(counts.get(node, 0))
            rows.append({"market": market, "node": node, "expected_hours": expected, "hours": hours,
                         "share": hours / expected, "ok": hours >= 0.9 * expected})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------------------------------------------
def _slim(frame):
    """SLIM columns with categorical keys: about a fifth of a full score_analysis frame."""
    out = frame[SLIM].copy()
    for c in KEYS:
        out[c] = out[c].astype("category")
    return out


def _plain(frame, cols=("market", "node")):
    """Object-typed keys for es.policies and es.run_monitors, whose groupbys do not pass observed=True: on
    categorical keys they would add one row per unobserved (market, node) pair, with n = 0 (probe, 23 Oct).
    The bootstrap functions pass observed=True and take the categorical frames as they are."""
    out = frame.copy()
    for c in cols:
        if isinstance(out[c].dtype, pd.CategoricalDtype):
            out[c] = out[c].astype(object)
    return out


def score_keys():
    """The 34 score sets as (kind, block, estimator, seed): the 10 blocks under xgb, 4 under logit, 4 on the frozen arm
    and 16 seed variants, in the order they are fitted. They are every frame es.secondary.audit_family asks for, and the
    two frames economics_for reads."""
    keys = [("roll", b, "xgb", 42) for b in BLOCKS]
    keys += [("roll", b, "logit", 42) for b in LOGIT_BLOCKS]
    keys += [("frozen", b, "xgb", 42) for b in CUTOFF_BLOCKS]
    keys += [("roll", b, "xgb", sd) for b in SEED_BLOCKS for sd in SEEDS]
    return keys


def _fit(panel, cfg, key):
    """One of the 34 score sets. roll/G1_24 and roll/G6 stay full (economics_for needs their price and settlement columns);
    every other frame is slimmed as soon as it is fitted, so no more than one full frame is in flight at a time."""
    kind, block, est, seed = key
    s = score_analysis(panel, cfg.rolling if kind == "roll" else cfg.frozen, block, est, seed)
    return s if key in FULL else _slim(s)


def _need(sc, key):
    """sc[key], or a LookupError that says the score set is not there (its own fit failed and is logged under its name)."""
    try:
        return sc[key]
    except KeyError:
        raise LookupError(f"score set {key} is not available (its fit did not complete)") from None


def _score_lookup(cfg, sc):
    """The score_fn of economics_for: only the precomputed frames, so a KeyError (never a fit, never the cache) for any
    other (kind, block, estimator, seed)."""
    def lookup(_panel, analysis, block, estimator, seed, **_kw):
        kind = "roll" if analysis is cfg.rolling else "frozen"
        return sc[(kind, block, estimator, seed)]
    return lookup


def _audit(sc, cfg, eta, on_error=None):
    """The audit table (audit_activity.csv): P1 on every score set and the other six policies on the primary pipeline. With
    `on_error(name, exception)` a part that raises ('no node qualifies' in one policy, say) is reported to it and left out,
    so that the others still run (Ruling 104(f)); without it the first exception propagates."""
    def part(name, make):
        try:
            return make()
        except Exception as exc:
            if on_error is None:
                raise
            on_error(name, exc)
            return None

    parts = []
    for (k, b, e, sd), s in sc.items():
        got = part(f"P1/{k}/{b}/{e}/s{sd}", lambda: activity(p1_fixed(_plain(s), ALPHA), ALPHA).assign(
            maintenance=k, block=b, estimator=e, seed=sd, policy="P1"))
        if got is not None:
            parts.append(got)
    policies = (("P4_56", lambda p: p4_rolling(p, ALPHA, 56)), ("P4_28", lambda p: p4_rolling(p, ALPHA, 28)),
                ("P5", lambda p: p5_self_correcting(p, ALPHA, eta)), ("P6", lambda p: p6_exact(p, ALPHA)),
                ("P2_season", lambda p: p2_insample(p, ALPHA, cfg.rolling.deploy.months)),
                ("P2_all", lambda p: p2_insample(p, ALPHA, set(range(1, 13)))))
    for pol, make in policies:
        got = part(pol, lambda: activity(make(_need(sc, ("roll", "G1_24", "xgb", 42))), ALPHA).assign(
            maintenance="roll", block="G1_24", estimator="xgb", seed=42, policy=pol))
        if got is not None:
            parts.append(got)
    return pd.concat(parts, ignore_index=True)


def _as_calib(scores, months):
    """P2's cutoff rows (training rows in `months`) relabelled as calibration rows, so the
    calibration-and-deployment bootstrap re-places the P2 cutoff (PC interval, labelled approximate)."""
    tr = scores[(scores["split"] == "train") & scores["settle_date"].map(lambda d: d.month).isin(months)]
    return pd.concat([tr.assign(split="calib"), scores[scores["split"] == "deploy"]], ignore_index=True)


def _tau_check(sc, cfg):
    """The registered cutoffs (pre_pull_cutoffs.csv) against the ones this run's own fits give. Features look back only and
    the fits are deterministic, so the expected difference is exactly 0."""
    key = ["analysis", "block", "market", "node"]
    now = pd.DataFrame([r for kind, a in (("roll", cfg.rolling), ("frozen", cfg.frozen)) for b in CUTOFF_BLOCKS
                        for r in cutoffs_from_scores(_need(sc, (kind, b, "xgb", 42)), a, b)]).drop_duplicates(key)
    # round_trip: pandas' default float parser is off by 1 ulp on about half of the values (probe, 23 Oct), which
    # would turn the expected "exactly 0" into about 1e-16.
    reg = pd.read_csv(cfg.cutoffs, float_precision="round_trip")
    reg = reg[reg["analysis"].isin(now["analysis"].unique())]
    m = reg.merge(now, on=key, how="outer", suffixes=("_reg", "_now"), indicator=True)
    both = m[m["_merge"] == "both"]
    cols = ("tau_p1", "tau_p2_season", "tau_p2_all")
    d = np.abs(np.concatenate([(both[f"{c}_reg"] - both[f"{c}_now"]).to_numpy(float) for c in cols]))
    return {"max_abs_diff": float(np.nanmax(d)) if np.isfinite(d).any() else float("nan"),
            "nan_pattern_mismatches": int(sum((both[f"{c}_reg"].isna() != both[f"{c}_now"].isna()).sum()
                                              for c in cols)),
            "n_compared": int(len(both)), "registered_only": int((m["_merge"] == "left_only").sum()),
            "recomputed_only": int((m["_merge"] == "right_only").sum())}


def _manifest_hashes(root):
    """{relative path: SHA-256} of the MANIFEST.sha256 in `root`."""
    path = Path(root) / "MANIFEST.sha256"
    if not path.is_file():
        raise RuntimeError(f"no MANIFEST.sha256 in {root}: Test A's stored rows cannot be checked")
    out = {}
    for line in path.read_text().splitlines():
        digest, sep, rel = line.partition("  ")
        if sep and rel:
            out[rel] = digest.lower()
    return out


def _read_verified(root, rel):
    """The parquet file `rel` of Test A's output `root`, parsed from the very bytes whose SHA-256 is the one that Test A's
    MANIFEST.sha256 lists for it (Ruling 116, Mn3): Test B re-reads Test A's rows after its own lock, and the check made before
    the lock does not cover a file that changed in between. A mismatch is an error of the analysis that reads the file."""
    listed = _manifest_hashes(root).get(rel)
    if listed is None:
        raise RuntimeError(f"{rel} is not listed in the MANIFEST.sha256 of {root}")
    data = (Path(root) / rel).read_bytes()
    if hashlib.sha256(data).hexdigest() != listed:
        raise RuntimeError(f"{Path(root) / rel} changed since Test A's manifest was written (its SHA-256 is not the one the "
                           "manifest lists)")
    return pd.read_parquet(io.BytesIO(data))


def _load_kept(kept_dir, verify_against=None):
    """The five stored frames of a kept folder. With `verify_against` (Test A's output folder) each is read through that
    folder's manifest (_read_verified)."""
    if verify_against is not None:
        return {name: _read_verified(verify_against, f"kept/{_safe(name)}.parquet") for name in KEPT}
    return {name: pd.read_parquet(Path(kept_dir) / f"{_safe(name)}.parquet") for name in KEPT}


def _concat_opt(frames):
    frames = [f for f in frames if f is not None]
    return pd.concat(frames, ignore_index=True).drop_duplicates(["node", "timestamp_utc"]) if frames else None


# ---------------------------------------------------------------------------------------------------------------
# The log and the results
# ---------------------------------------------------------------------------------------------------------------
def _log(out, *lines):
    with open(Path(out) / "RUN_LOG.txt", "a") as fh:
        fh.write("".join(f"{line}\n" for line in lines))


def _log_quietly(out, *lines):
    """_log, best effort (Ruling 109, I8): the stages after the lock log what they do, and a write error there (a full disk)
    must not end a run whose data are read. The line is lost, and nothing else."""
    try:
        _log(out, *lines)
    except OSError:
        pass


def _stage(out, name):
    _log_quietly(out, f"stage {name} {datetime.now(timezone.utc).isoformat()}")


def _record_crash(out, exc):
    """Called inside the except block: the run counts as made, and its log says where it died. Not after the run's manifest is
    written: a finished output is sealed by that manifest, and a line appended to its log would break it."""
    if (Path(out) / "MANIFEST.sha256").exists():
        return
    try:
        _log(out, f"crashed {datetime.now(timezone.utc).isoformat()} {type(exc).__name__}: {exc}",
             *traceback.format_exc().splitlines())
    except OSError:
        pass                                    # the original error is the one to raise


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, (dt.date, pd.Timestamp)):
        return o.isoformat()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)


def _finish(out, results, claim=None):
    """Write results.json, the last log line and the manifest; with the run's `claim`, then add finished_utc, the status and
    the manifest's SHA-256 to its record in RUNS.json (Ruling 109, I3), which is what Test B checks Test A's output against."""
    results["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (out / "results.json").write_text(json.dumps(results, indent=2, default=_json_default))
    _log(out, f"finished {results['finished_utc']}")
    write_manifest(out)
    if claim is not None:
        _record_finished(claim, results)
    return results


# ---------------------------------------------------------------------------------------------------------------
# NaN bootstrap replicates (controller Ruling 104(e))
# ---------------------------------------------------------------------------------------------------------------
#: A statistic is marked approximate when more than this share of its bootstrap replicates is NaN.
NAN_APPROXIMATE_SHARE = 0.01
#: block_contrast_bootstrap's own defaults, read from it, so that the fallback below draws the very same replicates.
_CONTRAST_DEFAULTS = {name: p.default for name, p in inspect.signature(block_contrast_bootstrap).parameters.items()
                      if name in ("mean_block", "seed")}


def _split_nan(t_star):
    """(the finite replicates, how many were NaN, whether that is more than NAN_APPROXIMATE_SHARE of them)."""
    t_star = np.asarray(t_star, dtype=float)
    n_nan = int(np.isnan(t_star).sum())
    return t_star[~np.isnan(t_star)], n_nan, bool(n_nan > NAN_APPROXIMATE_SHARE * len(t_star))


def _contrast(alt, base, alpha, n_rep):
    """H-B1, H-B2 and PC: block_contrast_bootstrap, with NaN replicates handled instead of fatal.

    A replicate is NaN when it draws no calibration (or no deployment) day of one of the two frames. block_contrast_bootstrap
    then refuses (basic_p_one_sided raises: a NaN would give the smallest possible p), and a refusal after the data are read
    would spend the one run. When it refuses, the replicates are taken from the same core, with the same defaults, and p, the
    percentile interval and the basic interval are computed on the finite ones; n_nan_replicates is recorded, and a share
    above 1% marks the result approximate. With no NaN replicate the answer is block_contrast_bootstrap's own, number for
    number. A NaN point estimate, or no finite replicate, still raises: there is nothing to infer from. An input that
    block_contrast_bootstrap refuses for another reason is refused again by the core, so the error is the same."""
    try:
        r = block_contrast_bootstrap(alt, base, alpha=alpha, n_rep=n_rep)
    except ValueError:
        t_hat, t_star, n_nodes = _cal_dep_bootstrap([alt, base], lambda e: e[0] - e[1], alpha, n_rep,
                                                    _CONTRAST_DEFAULTS["mean_block"], _CONTRAST_DEFAULTS["seed"])
        finite, n_nan, approximate = _split_nan(t_star)
        if np.isnan(t_hat) or not finite.size:
            raise ValueError(f"the contrast has nothing to infer from: t_hat {t_hat}, {n_nan} of {len(t_star)} replicates "
                             "NaN") from None
        return {"t_hat": float(t_hat), "p": basic_p_one_sided(t_hat, finite), "ci": percentile_ci(finite),
                "ci_basic": basic_ci(t_hat, finite), "n_nodes": n_nodes, "n_nan_replicates": n_nan,
                "nan_approximate": approximate}
    return {**{k: r[k] for k in ("t_hat", "p", "ci", "ci_basic", "n_nodes")}, "n_nan_replicates": 0,
            "nan_approximate": False}


def _level_draw(scores, alpha, n_rep, measure, aligned):
    """One draw of block_level_bootstrap, which returns its replicates. Its intervals are NaN when one replicate is, so with NaN
    replicates they are recomputed on the finite ones; n_nan_replicates is recorded and a share above 1% marks the draw's
    result approximate (nan_approximate). With no NaN replicate the intervals are block_level_bootstrap's own."""
    r = block_level_bootstrap(scores, alpha=alpha, n_rep=n_rep, measure=measure, aligned=aligned)
    finite, n_nan, approximate = _split_nan(r["t_star"])
    out = {"t_hat": r["t_hat"], "ci": r["ci"], "ci_basic": r["ci_basic"], "n_nodes": r["n_nodes"],
           "n_nan_replicates": n_nan, "nan_approximate": approximate}
    if n_nan and finite.size and not np.isnan(r["t_hat"]):
        out["ci"], out["ci_basic"] = percentile_ci(finite), basic_ci(r["t_hat"], finite)
    return out


def _level(scores, alpha, n_rep, measure="median"):
    """E5 and H-S (Ruling 118, M2): the calendar-aligned draw of block_level_bootstrap is the primary one (t_hat, ci, ci_basic,
    n_nodes and the NaN count are its), and the registered independent draws are run beside it, their percentile interval
    stored as ci_independent, a diagnostic, with their own NaN count (n_nan_replicates_independent, nan_approximate_independent).
    Both draws get the NaN handling of Ruling 104(e). The result records scheme 'aligned' and is approximate whatever the NaN
    count: the registered scheme is the independent one, and the aligned draw is not."""
    aligned = _level_draw(scores, alpha, n_rep, measure, True)
    independent = _level_draw(scores, alpha, n_rep, measure, False)
    return {**aligned, "scheme": SCHEME_E5_HS, "ci_independent": independent["ci"],
            "n_nan_replicates_independent": independent["n_nan_replicates"],
            "nan_approximate_independent": independent["nan_approximate"], "approximate": True}


def _estimand_rows(windows, n_rep):
    """E1-E4: estimand_cis, as it is, with the NaN-replicate columns. A NaN replicate would make a percentile endpoint NaN, so
    finite endpoints prove there was none (n_nan_replicates 0); a non-finite endpoint marks the statistic approximate, with
    the count unknown (None), because estimand_cis does not return its replicates. It cannot produce one in practice: its
    inputs are checked finite, and a degenerate replicate (no loss at all) raises ZeroDivisionError, which the caller records
    as the estimands' error."""
    rows = estimand_cis(windows, n_rep=n_rep)
    out = []
    for r in rows:
        clean = bool(np.isfinite(r["lo"]) and np.isfinite(r["hi"]))
        out.append({**r, "n_nan_replicates": 0 if clean else None, "approximate": bool(r["approximate"]) or not clean})
    return out


# ---------------------------------------------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------------------------------------------
def _secondary_counts(sec):
    """Per BH family of secondary_bh.csv: `tests` is the number of rows BH adjusted (the `included` ones: a NaN p-value is no
    test that ran, Ruling 80), `rows` all of them, `bh_rejections` the rejected."""
    return {fam: {"tests": int(g["included"].sum()), "rows": int(len(g)), "bh_rejections": int(g["bh_reject"].sum())}
            for fam, g in sec.groupby("family")}


HB_IDS = tuple(h for h, _ in HB_PAIRS)


def _estimand_status(family, hid):
    """The registered label of a demoted H-B hypothesis, with the reasons of the registration rule when it has any."""
    reasons = "; ".join(family[hid]["reasons"])
    return f"{STATUS_HB_ESTIMAND}: {reasons}" if reasons else STATUS_HB_ESTIMAND


def _decide(he, hb, family, confirmatory, prior, test_id=None):
    """The registered decision rule's wording, given the numbers (spec v3 §6.3-6.4). Pure: no data, no files.

    `he` is rotation_test's result (one window in Test A, and in Test B when Test A did not run; both windows otherwise), or
    None if H-E did not complete; `hb` maps H-B1 and H-B2 to their bootstrap results (a test that did not complete is absent);
    `family` and `confirmatory` come from protocol_inputs.json, and `prior` is Test A's results.json (None in Test A, and in
    Test B when Test A left nothing to pool). `test_id` says which test it is ("B" with no `prior` is Test B alone); it defaults
    to "B" if there is a `prior` and "A" if not. Adds "status" (and, in Test B, "replicated") to each entry of `hb` in place, and
    returns (the H-E entry of results.json, or None if there is no `he`; the Holm rejections by hypothesis, None in Test A, or
    {"error": ...} if a confirmatory hypothesis is missing).

    Test A: H-E is interim (approximate when the registration rule says so) and never enters the Holm family; an H-B test
    is judged at alpha / m, m the number of confirmatory hypotheses, and a test the registration rule demoted is an
    estimand, with its reasons. Test B: H-E is decided on the pooled windows (and carries the pooled test's `approximate` flag,
    Ruling 104(b)); the full Holm step-down runs over the confirmatory hypotheses, with H-E's pooled p and Test A's p for each
    confirmatory H-B test; a confirmatory H-B test replicates when Test B's bootstrap has the sign of Test A's and p < alpha. An
    estimand has no replication verdict (`replicated` is None) and needs nothing from Test A (Ruling 109, I4, I5). Without H-E
    the step-down cannot be run as registered, and the H-B statuses say so. Test B alone (Test A did not run): H-E is tested on
    the one window and is not confirmatory, nothing is replicated and no step-down is run (Ruling 109, I4)."""
    test_b = (prior is not None) if test_id is None else test_id == "B"
    he_conf = "H-E" in confirmatory
    m = len(confirmatory)
    if not test_b:
        out = None if he is None else {"status": STATUS_HE_INTERIM, **he,
                                       "approximate": bool(family["H-E"]["interim_approximate"]), "confirmatory": he_conf}
        for hid in hb:
            if not hb[hid]["confirmatory"]:
                hb[hid]["status"] = _estimand_status(family, hid)
            elif hb[hid]["p"] <= ALPHA / m:
                hb[hid]["status"] = STATUS_HB_REJECTED_A.format(alpha=ALPHA, m=m)
            else:
                hb[hid]["status"] = STATUS_HB_NOT_REJECTED_A.format(alpha=ALPHA, m=m)
        return out, None
    if prior is None:
        out = None if he is None else {"status": STATUS_HE_ALONE, **he, "approximate": bool(family["H-E"]["interim_approximate"]),
                                       "confirmatory": False, "wording": WORDING_HE_ALONE}
        for hid in hb:
            hb[hid]["replicated"] = None
            hb[hid]["status"] = STATUS_HB_NO_TEST_A if hb[hid]["confirmatory"] else _estimand_status(family, hid)
        return out, {h: None for h in FAMILY_ORDER}
    out = None if he is None else {"status": STATUS_HE_DECIDED, **he,
                                   "approximate": bool(family["H-E"]["approximate"]), "confirmatory": he_conf}
    pvals = {h: prior[h]["p"] for h in confirmatory if h in HB_IDS}
    if he is not None:
        pvals["H-E"] = he["p"]
    runnable = all(h in pvals for h in confirmatory)
    reject = ((dict(zip(confirmatory, map(bool, holm([pvals[h] for h in confirmatory], ALPHA)))) if confirmatory else {})
              if runnable else None)
    for hid in hb:
        if not hb[hid]["confirmatory"]:
            hb[hid]["replicated"], hb[hid]["status"] = None, _estimand_status(family, hid)
            continue
        same_sign = hb[hid]["t_hat"] != 0 and np.sign(hb[hid]["t_hat"]) == np.sign(prior[hid]["t_hat"])
        hb[hid]["replicated"] = bool(same_sign and hb[hid]["p"] < ALPHA)
        if reject is None:
            hb[hid]["status"] = STATUS_HB_NEEDS_HE
        elif reject.get(hid) and hb[hid]["replicated"]:
            hb[hid]["status"] = STATUS_HB_BOTH
        elif reject.get(hid):
            hb[hid]["status"] = STATUS_HB_Q3_ONLY
        else:
            hb[hid]["status"] = STATUS_HB_NONE
    if out is not None:
        out["wording"] = (((WORDING_HE_DETECTED if reject.get("H-E") else WORDING_HE_NOT_DETECTED) if he_conf
                           else WORDING_HE_ESTIMAND))
    if reject is None:
        return out, {"error": "not computed: H-E, a confirmatory hypothesis, did not complete"}
    return out, {h: reject.get(h) for h in FAMILY_ORDER}


class _Failed:
    """What a guarded analysis returns when it raised."""
    def __repr__(self):
        return "FAILED"

    def __bool__(self):
        return False


FAILED = _Failed()


class _Run:
    """What the stages after the lock share: the config and the registered inputs, the results being built and the failures so
    far. A stage runs each analysis through `guard` (Ruling 104(c), (f)): an exception is logged (type and message, then the
    traceback, in RUN_LOG.txt only), kept in `errors` under the analysis' name, and the run goes on. `prior` is Test A's
    results.json when Test B pools with it; Test B with no `prior` is Test B alone (`alone`)."""

    def __init__(self, cfg, n_rep, out, results, eta, family, confirmatory, prior):
        self.cfg, self.n_rep, self.out, self.results = cfg, n_rep, out, results
        self.eta, self.family, self.confirmatory, self.prior = eta, family, confirmatory, prior
        self.errors, self.secondary_errors = {}, set()
        self.p1_rows = None                 # H-E's rows when economics_for failed and p1_rows_for did not (Ruling 109, I6)

    @property
    def alone(self):
        return self.cfg.test_id == "B" and self.prior is None

    def log(self, *lines):
        """RUN_LOG.txt, best effort (Ruling 109, I8): what a stage logs is not worth the run."""
        _log_quietly(self.out, *lines)

    def record(self, name, exc, secondary=False):
        message = f"{type(exc).__name__}: {exc}"
        self.errors[name] = message                      # in memory first: a write error below must not lose the failure
        if secondary:
            self.secondary_errors.add(name)
        self.log(f"error {name}: {message}", *"".join(traceback.format_exception(exc)).splitlines())
        return message

    def guard(self, name, fn, secondary=False):
        """fn(), or FAILED after recording its exception. (Exception, not BaseException: an interrupt still ends the run.)"""
        try:
            return fn()
        except Exception as exc:
            self.record(name, exc, secondary)
            return FAILED

    def skip(self, name, why, secondary=False):
        """An analysis that cannot run because one it needs did not complete: its own recorded error."""
        self.record(name, LookupError(why), secondary)

    def error_entry(self, name):
        return {"error": self.errors[name]}

    def note_nan(self, name, r):
        if r["n_nan_replicates"]:
            self.log(f"nan_replicates {name}: {r['n_nan_replicates']} of {self.n_rep}"
                     + (" (more than 1%: approximate)" if r["nan_approximate"] else ""))

    def status(self):
        if not self.errors:
            return STATUS_COMPLETED
        return STATUS_SECONDARY if set(self.errors) <= self.secondary_errors else STATUS_ERRORS


def _stage_call(run, name, fn, default):
    """A stage as a whole, as a last resort: a bug in the glue between analyses is recorded and the later stages still run."""
    r = run.guard(f"stage/{name}", fn)
    return default if r is FAILED else r


def _stage_scores(run, panel):
    """The 34 score sets, fitted one by one: a set that fails is recorded and left out (what needs it fails by itself)."""
    (run.out / "scores").mkdir()
    sc = {}
    for key in score_keys():
        kind, block, est, seed = key
        name = f"scores/{kind}/{block}/{est}/s{seed}"
        s = run.guard(name, lambda: _fit(panel, run.cfg, key))
        if s is FAILED:
            continue
        sc[key] = s               # object keys in the file, like the cache files
        run.guard(f"{name}/file", lambda: _plain(s[CACHED_COLUMNS], KEYS).to_parquet(
            run.out / "scores" / f"{kind}__{block}__{est}__s{seed}.parquet", index=False))
    return sc


def _stage_tau(run, sc):
    r = run.guard("tau_check", lambda: _tau_check(sc, run.cfg))
    run.results["tau_check"] = run.error_entry("tau_check") if r is FAILED else r
    if r is not FAILED:
        run.log(f"tau_check {json.dumps(r, sort_keys=True)}")


def _stage_audit(run, sc):
    table = run.guard("audit_activity", lambda: _audit(sc, run.cfg, run.eta,
                                                       on_error=lambda part, exc: run.record(f"audit_activity/{part}", exc)))
    if table is not FAILED:
        run.guard("audit_activity/file", lambda: table.to_csv(run.out / "tables" / "audit_activity.csv", index=False))


def _pc_contrast(sc, months, n_rep):
    prim = _need(sc, ("roll", "G1_24", "xgb", 42))
    return _contrast(_as_calib(prim, months), prim, ALPHA, n_rep)


def _e5_row(sc, block, n_rep):
    frame = _need(sc, ("roll", block, "xgb", 42))
    med, fail = _level(frame, ALPHA, n_rep), _level(frame, ALPHA, n_rep, measure="failures")
    return {"block": block, "median_log_error": med["t_hat"], "lo": med["ci"][0], "hi": med["ci"][1],
            "failures": fail["t_hat"], "failures_lo": fail["ci"][0], "failures_hi": fail["ci"][1],
            # Ruling 118, M2: lo, hi, failures_lo and failures_hi are the calendar-aligned draw's; the registered independent
            # draws' percentile intervals are the diagnostics beside them
            "lo_independent": med["ci_independent"][0], "hi_independent": med["ci_independent"][1],
            "failures_lo_independent": fail["ci_independent"][0], "failures_hi_independent": fail["ci_independent"][1],
            "n_nodes": med["n_nodes"], "n_nan_replicates": med["n_nan_replicates"],
            "n_nan_replicates_failures": fail["n_nan_replicates"],
            "n_nan_replicates_independent": med["n_nan_replicates_independent"],
            "n_nan_replicates_failures_independent": fail["n_nan_replicates_independent"],
            "scheme": med["scheme"], "interval_registered": INTERVAL_E5_HS, "approximate": True}


def _stage_bootstraps(run, sc):
    """H-B1 and H-B2, PC, H-S and E5, each in its own guard. Returns the H-B tests that completed, as results.json holds them."""
    cfg, n_rep, results, family = run.cfg, run.n_rep, run.results, run.family
    hb = {}
    for hid, alt in HB_PAIRS:
        r = run.guard(hid, lambda: _contrast(_need(sc, ("roll", alt, "xgb", 42)), _need(sc, ("roll", "DART3", "xgb", 42)),
                                             ALPHA, n_rep))
        if r is FAILED:
            results[hid] = run.error_entry(hid)
            continue
        run.note_nan(hid, r)
        confirmatory = hid in run.confirmatory
        hb[hid] = {"t_hat": r["t_hat"], "ci": list(r["ci"]), "ci_basic": list(r["ci_basic"]),
                   "n_nodes": r["n_nodes"], "n_nan_replicates": r["n_nan_replicates"],
                   "confirmatory": confirmatory,
                   # a NaN share above 1% marks the intervals approximate too (Ruling 104(e))
                   "ci_approximate": bool(family[hid]["approximate"]) or r["nan_approximate"],
                   # Ruling 109 I5, Ruling 110: the interval that goes with the hypothesis' registered status. A test: ci_basic,
                   # which goes with the basic p (Ruling 55(f)). An estimand: the percentile ci is the primary interval and
                   # ci_basic a diagnostic beside it, and the bootstrap p is a diagnostic, not a decision field.
                   "interval_registered": INTERVAL_HB_TEST if confirmatory else INTERVAL_HB_ESTIMAND}
        if confirmatory:
            hb[hid]["p"] = r["p"]
        else:
            hb[hid]["diagnostics"] = {"bootstrap_p_one_sided": r["p"]}
    results["PC"] = {}
    for name, months in (("P2_season", cfg.rolling.deploy.months), ("P2_all", set(range(1, 13)))):
        r = run.guard(f"PC/{name}", lambda: _pc_contrast(sc, months, n_rep))
        if r is FAILED:
            results["PC"][name] = run.error_entry(f"PC/{name}")
            continue
        run.note_nan(f"PC/{name}", r)
        results["PC"][name] = {"median_diff": r["t_hat"], "ci": list(r["ci"]), "n_nodes": r["n_nodes"],
                               "n_nan_replicates": r["n_nan_replicates"], "ci_approximate": True}
    results["H-S"] = {}
    for est in ("xgb", "logit"):
        r = run.guard(f"H-S/{est}", lambda: _level(_need(sc, ("roll", "DART3", est, 42)), ALPHA, n_rep, measure="failures"))
        if r is FAILED:
            results["H-S"][est] = run.error_entry(f"H-S/{est}")
            continue
        run.note_nan(f"H-S/{est}", r)
        run.note_nan(f"H-S/{est}/independent", {"n_nan_replicates": r["n_nan_replicates_independent"],
                                                "nan_approximate": r["nan_approximate_independent"]})
        results["H-S"][est] = {"failures": r["t_hat"], "ci": list(r["ci"]), "ci_independent": list(r["ci_independent"]),
                               "scheme": r["scheme"], "interval_registered": INTERVAL_E5_HS, "n_nodes": r["n_nodes"],
                               "n_nan_replicates": r["n_nan_replicates"],
                               "n_nan_replicates_independent": r["n_nan_replicates_independent"], "approximate": True}
    e5 = []
    for block in BLOCKS:
        r = run.guard(f"E5/{block}", lambda: _e5_row(sc, block, n_rep))
        e5.append({"block": block, **run.error_entry(f"E5/{block}")} if r is FAILED else r)
    run.guard("E5/file", lambda: pd.DataFrame(e5).to_csv(run.out / "tables" / "E5_blocks.csv", index=False))
    results["E5"] = e5
    results["E5_scope"] = SCOPE_E5[results["test"]]       # Ruling 109, M5: E5 is this run's own window, in Test B too
    return hb


def _stage_economics(run, panel, sc):
    """economics_for with the side tables of the development files and of the sealed pulls. Returns (econ, kept), or
    (None, None) if it did not complete: what needs the economic rows then records that itself. If it did not complete but the
    side tables loaded, H-E's rows are built on their own with p1_rows_for, in a guard of their own (run.p1_rows; Ruling 109,
    I6), so that one failing comparator or exposure split does not cost the run its one confirmatory test."""
    cfg, out = run.cfg, run.out
    tables = {}

    def compute():
        base = Path(cfg.data_root or DATA_ROOT)
        roots = [base / d for d in cfg.sealed_dirs]
        tables["fmm"] = _concat_opt([load_caiso_fmm(base / "caiso_fmm_dev")]
                                    + [load_caiso_fmm(r / "fmm", dev_only=False) for r in roots])      # Task 6 fix round, Minor 4
        tables["nyi"] = _concat_opt([load_nyiso_integrated(base / "nyiso_rtint_dev")]
                                    + [load_nyiso_integrated(r / "nyiso_rtint", dev_only=False) for r in roots])
        return economics_for(panel, cfg.rolling, tables["fmm"], tables["nyi"], run.eta, score_fn=_score_lookup(cfg, sc),
                             keep_rows=True)

    r = run.guard("economics", compute)
    if r is FAILED:
        run.results["economics"] = run.error_entry("economics")
        if "nyi" in tables:
            rows = run.guard("economics/p1_rows", lambda: p1_rows_for(panel, cfg.rolling, tables["fmm"], tables["nyi"],
                                                                      score_fn=_score_lookup(cfg, sc)))
            if rows is not FAILED:
                run.p1_rows = rows
                run.log("p1_rows_for: H-E runs on the P1 rows built on their own, because economics_for did not complete")

                def write_p1():                       # Ruling 116, N1(a): Test B pools H-E from these rows, so they are stored
                    (out / "kept").mkdir(exist_ok=True)
                    rows.to_parquet(out / "kept" / f"{_safe('P1')}.parquet", index=False)

                run.guard("economics/kept", write_p1)
        return None, None
    econ, kept = r
    # Task 12 N2: deployment rows that P1 does not have are dropped from the other policies' and comparators' frames;
    # economics_for warns, and the counts come back here so that the record keeps them.
    run.results["coverage"]["comparator_rows_dropped"] = dict(kept.attrs["dropped_rows"])
    run.log(f"comparator_rows_dropped {json.dumps(run.results['coverage']['comparator_rows_dropped'], sort_keys=True)}")
    run.guard("economics/table", lambda: pd.DataFrame(econ).to_csv(out / "tables" / "economics.csv", index=False))

    def write_kept():
        (out / "kept").mkdir()
        for name, frame in kept.items():
            frame.to_parquet(out / "kept" / f"{_safe(name)}.parquet", index=False)

    run.guard("economics/kept", write_kept)
    return econ, kept


def _stage_estimands(run, kept):
    """E1-E4: this quarter alone (Test B), pooled with Test A's rows (or alone, in Test A and in Test B when Test A did not
    run), and without CAISO. Returns the windows H-E pools, or None if they could not be built."""
    out, n_rep, prior, results = run.out, run.n_rep, run.prior, run.results
    results["estimands_scope"] = SCOPE_B_ALONE if run.alone else SCOPE_SINGLE if prior is None else SCOPE_POOLED
    windows = None
    if kept is None:
        for name in ("estimands", "estimands_ex_caiso") + (("estimands_this_quarter",) if prior is not None else ()):
            run.skip(name, "needs the economics rows (economics_for did not complete)")
    else:
        windows = [kept]
        if prior is not None and (results.get("test_a") or {}).get("missing_kept"):
            results["estimands_scope"] = SCOPE_B_OWN_QUARTER       # Ruling 116, N1(b): this quarter alone, and the state says why
        elif prior is not None:
            quarter = run.guard("estimands_this_quarter", lambda: _estimand_rows([kept], n_rep))
            if quarter is not FAILED:
                run.guard("estimands_this_quarter/file", lambda: pd.DataFrame(quarter).to_csv(
                    out / "tables" / "estimands_this_quarter.csv", index=False))
            earlier = run.guard("prior_kept", lambda: _load_kept(Path(run.cfg.prior_test_dir) / "kept",
                                                                 verify_against=run.cfg.prior_test_dir))
            windows = None if earlier is FAILED else [earlier, kept]
            if windows is None:
                run.skip("estimands", "needs Test A's stored rows (they did not load)")
                run.skip("estimands_ex_caiso", "needs Test A's stored rows (they did not load)")
    if windows is None:
        results["estimands"] = run.error_entry("estimands")
        return None
    cis = run.guard("estimands", lambda: _estimand_rows(windows, n_rep))
    if cis is FAILED:
        results["estimands"] = run.error_entry("estimands")
    else:
        run.guard("estimands/file", lambda: pd.DataFrame(cis).to_csv(out / "tables" / "estimands.csv", index=False))
        results["estimands"] = {r["stat"]: {k: r[k] for k in ("estimate", "lo", "hi", "approximate", "n_nan_replicates")}
                                for r in cis}
    # spec §4.1: without CAISO
    ex = run.guard("estimands_ex_caiso", lambda: _estimand_rows(
        [{k: v[v["market"] != "CAISO"] for k, v in w.items()} for w in windows], n_rep))
    if ex is not FAILED:
        run.guard("estimands_ex_caiso/file", lambda: pd.DataFrame(ex).to_csv(
            out / "tables" / "estimands_ex_caiso.csv", index=False))
    return windows


def _stage_secondary(run, panel, sc, econ):
    """The three BH families and their tables, each secondary analysis in its own guard (Ruling 104(c)): audit robustness,
    economic robustness, drift monitors, then the BH tables over the families that completed. A family that failed is
    recorded as results["secondary"][name] = {"error": ...} and is left out of the adjustment (a family is adjusted on its
    own, so the others are unaffected). There is no label control among them (see the module docstring)."""
    cfg, out, n_rep = run.cfg, run.out, run.n_rep
    sec = run.results.setdefault("secondary", {})
    families, skipped, audit_stats = {}, [], []

    def audit():
        """The 18 tests, each with the finite-replicate bootstrap of the H-B contrasts (Ruling 109, M1): a NaN replicate in one
        of them costs that test its NaN replicates, not the family its 18 tests. The NaN counts are kept in audit_stats."""
        audit_stats.clear()

        def contrast(alt, base, alpha, n_rep):
            audit_stats.append(_contrast(alt, base, alpha, n_rep))
            return audit_stats[-1]

        rows = audit_family(lambda kind, b, e, s: _need(sc, (kind, b, e, s)), n_rep=n_rep, contrast=contrast)
        for row, r in zip(rows, audit_stats):
            run.note_nan(f"audit/{row['test']}", r)
        return rows

    def economic():
        if econ is None:
            raise LookupError("needs the economic rows (economics_for did not complete)")
        return economic_family(econ)

    def monitors():
        parts = []
        for b in BLOCKS:
            if b == "G6":
                continue
            # one plain copy at a time; strict=False: a node with too few days is skipped and counted (Ruling 104(d))
            frame = monitor_rows(panel, cfg.rolling, {b: _plain(_need(sc, ("roll", b, "xgb", 42)))}, strict=False)
            skipped.extend(frame.attrs.get("skipped_nodes", []))
            parts.append(frame)
        mon = pd.concat(parts, ignore_index=True)
        mon.to_csv(out / "tables" / "monitor_scores.csv", index=False)
        return monitor_family(evaluate_monitors(mon).to_dict(orient="records"))

    _stage(out, "secondary")
    for name, build in (("audit robustness", audit), ("economic robustness", economic), ("drift monitors", monitors)):
        rows = run.guard(name, build, secondary=True)
        if rows is FAILED:
            sec[name] = run.error_entry(name)
        else:
            families[name] = rows
    for d in skipped:
        run.log(f"monitor_skipped {d['block']} {d['market']}/{d['node']}: {d['reason']}")
    if not families:
        return
    table = run.guard("BH tables", lambda: bh_table([row for rows in families.values() for row in rows]), secondary=True)
    if table is FAILED:
        sec["BH tables"] = run.error_entry("BH tables")
        return
    run.guard("BH tables/file", lambda: table.to_csv(out / "tables" / "secondary_bh.csv", index=False), secondary=True)
    counts = _secondary_counts(table)
    for name in families:
        counts.setdefault(name, {"tests": 0, "rows": 0, "bh_rejections": 0})        # a family with no row to adjust
    sec.update(counts)
    if "drift monitors" in families:
        sec["drift monitors"]["skipped_nodes"] = skipped
    if "audit robustness" in families:
        pairs = list(zip(families["audit robustness"], audit_stats))
        if any(r["n_nan_replicates"] for _, r in pairs):
            sec["audit robustness"]["n_nan_replicates"] = {row["test"]: r["n_nan_replicates"] for row, r in pairs
                                                           if r["n_nan_replicates"]}
            sec["audit robustness"]["nan_approximate"] = [row["test"] for row, r in pairs if r["nan_approximate"]]


def _prior_p1(prior_dir):
    """Test A's stored P1 rows alone (H-E needs no other frame), read through Test A's manifest (Ruling 116, Mn3)."""
    return _read_verified(prior_dir, f"kept/{_safe('P1')}.parquet")


def _stage_he(run, kept, windows):
    """H-E: the rotation test on this window (Test A, and Test B when Test A did not run) or pooled with Test A's (Test B).
    Its rows are kept["P1"], or, if economics_for did not complete, the P1 rows built on their own (run.p1_rows; Ruling 109,
    I6). FAILED if it did not complete."""
    def go():
        rows = kept["P1"] if kept is not None else run.p1_rows
        if rows is None:
            raise LookupError("needs the economic rows (economics_for did not complete)")
        cube = RowCube.from_rows(rows)
        if run.prior is None:
            return rotation_test([cube])
        if windows is not None and len(windows) == 2:
            earlier = windows[0]["P1"]
        else:
            earlier = _prior_p1(run.cfg.prior_test_dir)         # the pooled estimands were not built; H-E needs P1 only
        return rotation_test([RowCube.from_rows(earlier), cube])

    return run.guard("H-E", go)


def _stage_decision(run, he, hb):
    """The wording of the registered rule (`_decide`) from what completed; what did not stays an error entry."""
    results = run.results
    r = run.guard("decision", lambda: _decide(None if he is FAILED else he, hb, run.family, run.confirmatory, run.prior,
                                              run.cfg.test_id))
    if r is FAILED:
        results["H-E"] = run.error_entry("H-E") if he is FAILED else {"error": run.errors["decision"], **he}
    else:
        entry, holm_out = r
        results["H-E"] = run.error_entry("H-E") if he is FAILED else entry
        if holm_out is not None:
            results["holm"] = holm_out
    results.update(hb)


def _out_exists_message(out):
    """Why an existing out_dir refuses the run (Ruling 116, N3). By the time this is asked, the test has no claim file and no
    record (they are refused first), so the folder was never locked: nothing was read, and removing it is safe, unless its log
    shows that the run took the lock or got to the data load (then a claim file was deleted, and the run counts)."""
    log = Path(out) / "RUN_LOG.txt"
    try:
        lines = log.read_text().splitlines() if log.is_file() else []
    except OSError:
        lines = []
    if any(line.startswith(("claim ", "stage ")) for line in lines):
        return _ONCE.format(out=out)
    return f"{out} exists and no claim was made for this test; nothing was read: remove {out} and run again"


def run_sealed(cfg, protocol_sha256, n_rep=N_REP, allow_departure=False):
    """Run one sealed test, once, and return the results dict that is also written to cfg.out_dir/results.json.

    `protocol_sha256` is the SHA-256 of the registered protocol, the one docs/sealed/REGISTRATION.md names. `n_rep` is the
    number of bootstrap replicates: a sealed run (one with a sealed arm) uses exactly the registered N_REP, 2000, which is the
    default (read from this signature by es.protocol); a rehearsal may use another. The log and results.json record what was
    used. `allow_departure=True` lets the run go on although files of the registered code manifest differ, and records them
    (spec §6.5 step 7); without it the run is refused.

    Raises ValueError (the test id is not A or B, the config is not the registered one, an input is missing, or Test A has not
    been run), FileExistsError (the test id has a record or a claim file: the test already ran, or crashed; or out_dir exists
    with no claim, which says so), PermissionError (a folder the run writes is not writable), RegistrationError (including a
    different environment from the pinned one), RuntimeError (a sealed pull or Test A's output no longer matches its manifest,
    Test A's record lacks its finish fields, or too little disk is free), all before the lock, so before anything is created
    and before any price row is read. Once the lock is taken (the claim file; see the module docstring) the run counts,
    whatever happens next. Only a failure of the data load aborts it; a failure in an analysis is recorded and the others go on
    (results["status"] says whether any failed)."""
    _check_test_id(cfg)                                            # Ruling 109, C1: before anything else runs
    _refuse_recorded(cfg)                                          # Ruling 104(a), 109 I9: then the record and the claim file
    _check_config(cfg, n_rep)
    out = Path(cfg.out_dir)
    if out.exists():
        raise FileExistsError(_out_exists_message(out))
    protocol_path, registration_path = _registration_paths(cfg)
    record = verify_registration(protocol_sha256, protocol_path, registration_path)
    bound = verify_bound(protocol_path, [cfg.cutoffs, cfg.inputs, cfg.code_manifest])
    departures = code_departures(cfg.code_manifest, cfg.code_root)
    if departures and not allow_departure:
        names = ", ".join(d["file"] for d in departures[:5]) + (", ..." if len(departures) > 5 else "")
        raise RegistrationError(f"{len(departures)} file(s) differ from the registered code manifest ({names}); "
                                "spec §6.5 step 7: restore the registered code, or pass allow_departure=True to "
                                "run anyway and record the departure")
    environment = _check_environment(protocol_path)
    manifests = check_sealed_dirs(cfg, record)
    eta, family = _read_inputs(cfg.inputs)
    confirmatory = [h for h in FAMILY_ORDER if family[h]["confirmatory"]]
    prior, test_a = _check_prior(cfg, record, confirmatory)
    _check_side_tables(cfg)
    _check_free_space(cfg)
    _check_writable(cfg)
    fingerprint = _code_fingerprint(cfg.code_manifest, cfg.code_root)
    started = datetime.now(timezone.utc).isoformat()
    results = {"test": cfg.test_id, "started_utc": started, "protocol_sha256": record["protocol_sha256"],
               "registration": registration_identifier(record), "registration_kind": registration_kind(record),
               "registered_utc": record["registered_utc"], "n_rep": n_rep, "bound_files": bound,
               "departures": departures, "sealed_manifests": manifests,
               "environment": environment, "eta": eta, "family": confirmatory, **({"test_a": test_a} if test_a else {})}
    # The run-once lock. Every check above reads no price row and creates nothing; from here on this is the one run, whether or
    # not it finishes. The claim comes first (Ruling 116, N3): with no claim file nothing has happened, and an out_dir that
    # exists is removable; with one, a crash leaves out_dir with the line that says where it died, and the next call is refused.
    claim = _take_claim(cfg, started, out)
    try:
        try:
            try:
                out.mkdir(parents=True)
            except FileExistsError:
                raise FileExistsError(_ONCE.format(out=out)) from None
            (out / "tables").mkdir()
            _log(out, f"started {started}", f"test {cfg.test_id}", f"protocol {record['protocol_sha256']}",
                 f"registration {registration_identifier(record)} registered_utc {record['registered_utc']}",
                 f"registration_kind {registration_kind(record)}",
                 f"n_rep {n_rep}", *(f"bound {p} {s}" for p, s in bound.items()),
                 *(f"sealed_manifest {d} {s}" for d, s in manifests.items()), f"departures {len(departures)}",
                 f"environment {json.dumps(environment, sort_keys=True)}",
                 *([f"test_a {json.dumps(test_a, sort_keys=True)}"] if test_a else []))
            if departures:
                (out / "departures.txt").write_text("".join(f"{d['registered']}  {d['current']}  {d['file']}\n"
                                                            for d in departures))
            # Ruling 104(a), 109 I1, I9: the record of the run, after the log line and before any price row is read
            _write_run_record(claim, cfg, record, started, out, fingerprint, n_rep, allow_departure, len(departures),
                              environment)
            return _execute(cfg, protocol_sha256, n_rep, out, results, eta, family, confirmatory, prior, claim)
        finally:
            claim.close()
    except BaseException as exc:
        _record_crash(out, exc)
        raise


def _execute(cfg, protocol_sha256, n_rep, out, results, eta, family, confirmatory, prior, claim):
    """Everything after the lock: the one run. It needs the claim that _claim_run made (Ruling 109, I1), and passes it to the
    loader. The data load, the panel and the coverage rule come first and may abort it; every analysis after them is
    guarded (see the module docstring)."""
    _require_claim(cfg, claim)
    run = _Run(cfg, n_rep, out, results, eta, family, confirmatory, prior)
    _stage(out, "load")
    raw = _load_sealed_raw(cfg, protocol_sha256, claim)
    nonfinite, off_hour = raw.attrs["nonfinite_dropped"], raw.attrs["off_hour_dropped"]
    dropped = {"node_map": raw.attrs["node_map_dropped"], "duplicates": raw.attrs["duplicates_dropped"],
               "beyond_window_end": raw.attrs["beyond_window_end"], "off_hour": sum(off_hour.values()),
               "nonfinite": sum(nonfinite.values())}
    _log(out, f"rows_read {raw.attrs['rows_read']}", f"rows_kept {raw.attrs['rows_kept']}",
         f"node_map_dropped {dropped['node_map']}", f"duplicates_dropped {dropped['duplicates']}",
         f"dev_duplicates_dropped {raw.attrs['dev_duplicates_dropped']}",
         f"beyond_window_end {dropped['beyond_window_end']}",
         f"off_hour_dropped {json.dumps(off_hour, sort_keys=True)}",
         f"nonfinite_dropped {json.dumps(nonfinite, sort_keys=True)}")
    results.update({"rows_read": raw.attrs["rows_read"], "rows_kept": raw.attrs["rows_kept"], "rows_dropped": dropped})
    panel = build_panel(raw)
    del raw
    cov = coverage(panel, cfg.rolling)
    feas = feasibility(panel, cfg.rolling, "G1_24")
    ok = cov.merge(feas[["market", "node", "n_pos", "feasible", "reason"]], on=["market", "node"], how="left")
    ok["usable"] = ok["ok"] & ok["feasible"].fillna(False).astype(bool)
    ran = int(ok["usable"].sum()) >= MIN_USABLE_NODES and ok.loc[ok["usable"], "market"].nunique() == len(MARKETS)
    ok.to_csv(out / "tables" / "coverage.csv", index=False)
    results["coverage"] = {"ran": bool(ran), "usable_nodes": int(ok["usable"].sum()), "nonfinite_dropped": nonfinite,
                           "off_hour_dropped": off_hour}
    _log(out, f"coverage ran {bool(ran)} usable_nodes {int(ok['usable'].sum())}")
    if not ran:
        (out / "NOT_RUN.md").write_text("Coverage rule failed (spec v3 §6.1): fewer than 28 usable nodes "
                                         "or not all four operators. See tables/coverage.csv.\n")
        results["errors"], results["status"] = {}, STATUS_NOT_RUN
        return _finish(out, results, claim)
    usable = set(map(tuple, ok.loc[ok["usable"], ["market", "node"]].to_numpy()))
    panel = panel[[k in usable for k in zip(panel["market"], panel["node"])]].reset_index(drop=True)

    _stage(out, "scores")
    sc = _stage_call(run, "scores", lambda: _stage_scores(run, panel), {})
    _stage_call(run, "tau_check", lambda: _stage_tau(run, sc), None)
    _stage(out, "audit")
    _stage_call(run, "audit", lambda: _stage_audit(run, sc), None)
    _stage(out, "bootstraps")
    hb = _stage_call(run, "bootstraps", lambda: _stage_bootstraps(run, sc), {})
    _stage(out, "economics")
    econ, kept = _stage_call(run, "economics", lambda: _stage_economics(run, panel, sc), (None, None))
    windows = _stage_call(run, "estimands", lambda: _stage_estimands(run, kept), None)
    _stage_call(run, "secondary", lambda: _stage_secondary(run, panel, sc, econ), None)
    _stage(out, "H-E")
    he = _stage_he(run, kept, windows)
    _stage_call(run, "decision", lambda: _stage_decision(run, he, hb), None)

    results["errors"], results["status"] = dict(run.errors), run.status()
    _log_quietly(out, f"status {results['status']}")
    return _finish(out, results, claim)
