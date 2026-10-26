"""Development economic analysis with another feature block as the alert model (Task S1; an exploratory secondary
analysis, controller Ruling 148).

``es.run_economics`` values the alerts of the combined block (``G1_24``) against the calendar-only model and VOL at P1's
alert count. A review asked what the same analysis says when the alert model is a different block, above all the plain
DART-lag block (``DART3``: calendar + plain DART lags), the one block with a clear ranking gain in 2026-H1. This module
repeats the analysis on the development quarters with the chosen block as the alert model, and changes nothing else:
XGBoost, seed 42, rolling-origin, premium position, calibrated fees (``eta`` from outputs/es/analogs/protocol_inputs.json),
strata, P1 as the reference policy, the comparators ``calendar@count`` and ``VOL@count`` at P1's count, the day bootstrap,
the CAISO twin, the H-E rotation row and the pooled AQ3 + AQ4 rehearsal are all ``es.run_economics``'s own.

Mechanism, with no edit of any existing file. ``es.run_economics.run_all`` calls the module-level name
``economics_for(panel, a, fmm, nyi, eta, keep_rows=True)``, whose default ``score_fn`` is ``es.scores.cached_scores``, and
``economics_for`` asks that function for the primary frame as ``(panel, analysis, "G1_24", "xgb", 42)`` and for the
calendar-only frame as ``(panel, analysis, "G6", "xgb", 42)``. ``score_fn_for(block)`` is the score function that answers
the first request with the chosen block's scores and passes every other request through; ``alert_block(block)`` puts a
wrapper of ``economics_for`` that uses it in place of the module-level name for the duration of a run and restores the
original afterwards, also on an error. The VOL comparator is built from the unscaled ``abs_mean_168h`` column of the
primary frame, which the panel supplies whatever the block, and every comparator is matched to P1's rows and counts as
always.

Development data only: the loaders are those ``es.run_economics.main`` uses, with their sealed-window guard on, and the
analyses are keys of ``es.analyses.ANALOGS`` (all of which deploy on or before 2026-06-30).

Output: the five CSV files and ``run_meta.json`` of ``es.run_economics.run_all`` in a NEW folder (default
outputs/es/economics_block/<block>), and ``block_meta.json`` beside them, which says which block the files belong to
(the CSVs themselves do not). The primary folder outputs/es/economics is refused, with or without ``--overwrite``, and so is
a folder that already holds a summary.csv unless ``--overwrite`` is given.

Run: DART_PROJECT_ROOT=... /opt/anaconda3/bin/python3 -m es.run_economics_block --block DART3 --analyses AQ3,AQ4,AQ1,AQ2
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import sys
import time
from pathlib import Path

from . import run_economics as _econ
from .analyses import ANALOGS
from .blocks import BLOCKS, G6, PAPER_BLOCK_NAMES, features
from .config import DATA_ROOT, OUT_DIR, SEAL_LAST_DEV_DATE
from .data import build_panel, load_dev_raw
from .scores import cached_scores
from .settlement import load_caiso_fmm, load_nyiso_integrated

#: The block ``economics_for`` asks for as the alert model, and the one it asks for as the calendar-only comparator.
PRIMARY_BLOCK, CALENDAR_BLOCK = "G1_24", "G6"
#: The development quarters (AQ3F is the frozen-origin variant of AQ3, not a quarter of its own).
DEFAULT_ANALYSES = ("AQ3", "AQ4", "AQ1", "AQ2")
LABEL = ("exploratory secondary analysis on the development quarters (Ruling 148): the alert model is the chosen block "
         "wherever the primary analysis uses the combined block; everything else is as in es.run_economics")
TAG = "[economics_block]"


# ---------------------------------------------------------------------------------------------------------------
# The score function and the hook
# ---------------------------------------------------------------------------------------------------------------

def check_block(name):
    """``name`` if it is a key of ``es.blocks.BLOCKS``; otherwise a ValueError that names it and lists the valid blocks."""
    if not isinstance(name, str) or name not in BLOCKS:
        raise ValueError(f"unknown block {name!r}: the alert block must be one of {', '.join(BLOCKS)}")
    return name


def score_fn_for(block, base=None):
    """The ``score_fn`` of ``economics_for`` for ``block`` as the alert model.

    It is ``base`` (default ``cached_scores``, looked up in this module when the function is called) except that a request
    for the combined block, ``PRIMARY_BLOCK``, is answered with ``block``'s scores. The calendar block and every other
    request pass through untouched, with the estimator, the seed and any further argument."""
    check_block(block)

    def score_fn(panel, analysis, requested, *args, **kwargs):
        scorer = cached_scores if base is None else base
        return scorer(panel, analysis, block if requested == PRIMARY_BLOCK else requested, *args, **kwargs)
    return score_fn


@contextlib.contextmanager
def alert_block(block, base=None):
    """Within the ``with`` block ``es.run_economics.economics_for`` is a wrapper that calls the original with
    ``score_fn_for(block, ...)`` as its ``score_fn``, so ``run_all`` (which calls the module-level name) runs the economic
    analysis with ``block`` as the alert model. A ``score_fn`` the caller passes is wrapped, not replaced; ``base`` is the
    score function used when none is passed (default ``cached_scores``). The original is put back on leaving the block,
    also when it raises."""
    check_block(block)
    original = _econ.economics_for

    def economics_for(panel, analysis, fmm, nyi, eta, score_fn=None, keep_rows=False):
        """es.run_economics.economics_for with the chosen block as the alert model."""
        return original(panel, analysis, fmm, nyi, eta, score_fn=score_fn_for(block, base if score_fn is None else score_fn),
                        keep_rows=keep_rows)
    _econ.economics_for = economics_for
    try:
        yield economics_for
    finally:
        _econ.economics_for = original


# ---------------------------------------------------------------------------------------------------------------
# The command line's checks
# ---------------------------------------------------------------------------------------------------------------

def parse_analyses(text):
    """The analysis ids of a comma list, in the order given (``DEFAULT_ANALYSES`` for None). Each must be a key of
    ``es.analyses.ANALOGS`` and none may appear twice, since a repeated id would put its rows into the files twice."""
    if text is None:
        return list(DEFAULT_ANALYSES)
    ids = [part.strip() for part in text.split(",")]
    valid = ", ".join(ANALOGS)
    if not all(ids):
        raise ValueError(f"empty analysis name in {text!r}: give a comma list of analyses from {valid}")
    for i in ids:
        if i not in ANALOGS:
            raise ValueError(f"unknown analysis {i!r}: the analyses are {valid}")
        if ids.count(i) > 1:
            raise ValueError(f"analysis {i!r} is listed more than once in {text!r}")
    return ids


def default_out_dir(block):
    """A new folder per block beside, never inside, the primary one."""
    return OUT_DIR / "economics_block" / block


def _primary_dirs():
    """The folder(s) of the primary economic analysis: outputs/es/economics."""
    return [OUT_DIR / "economics", _econ.ECON_DIR]


def _is_inside(path, root):
    """True if ``path`` is ``root`` or lies below it: by real path (symlinks, '..', a trailing slash) and, where the two
    exist, by file identity, so that another spelling of the name (case, on a case-insensitive volume) does not get round it."""
    path, root = Path(path).resolve(), Path(root).resolve()
    if path == root or root in path.parents:
        return True
    if not root.exists():
        return False
    for candidate in (path, *path.parents):
        try:
            if candidate.exists() and candidate.samefile(root):
                return True
        except OSError:
            continue
    return False


def check_out_dir(out_dir, overwrite=False):
    """``out_dir`` resolved, after the checks below; creates nothing. Raises ValueError for
    - the primary economic analysis's folder, or anything inside it, with or without ``overwrite``;
    - a path that exists and is not a directory;
    - a folder that already holds a summary.csv (an earlier run's results), unless ``overwrite``."""
    out = Path(out_dir).expanduser()
    for primary in _primary_dirs():
        if _is_inside(out, primary):
            raise ValueError(f"refusing {out}: it is the folder of the primary economic analysis ({primary}), or inside it, "
                             "whose files stay as they are (--overwrite does not apply to it); choose another --out-dir")
    out = out.resolve()
    if out.exists() and not out.is_dir():
        raise ValueError(f"refusing {out}: it exists and is not a directory")
    if (out / "summary.csv").exists() and not overwrite:
        raise ValueError(f"refusing {out}: it already holds a summary.csv (the results of an earlier run); "
                         "choose another --out-dir, or pass --overwrite to replace that run's files")
    return out


# ---------------------------------------------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------------------------------------------

def describe_run(block, analyses, out_dir, n_rep):
    """The start-up message: which block, how many features (and which, when there are few), which analyses, and where
    the files go."""
    extra = BLOCKS[block]
    detail = f"{len(G6)} calendar + {len(extra)} price-history"
    if 0 < len(extra) <= 6:
        detail += ": " + ", ".join(extra)
    return "\n".join([
        f'{TAG} alert block {block} ("{PAPER_BLOCK_NAMES.get(block, block)}"): {len(features(block))} features ({detail}); '
        f'the primary analysis uses {PRIMARY_BLOCK} ("{PAPER_BLOCK_NAMES[PRIMARY_BLOCK]}"), {len(features(PRIMARY_BLOCK))} features',
        f"{TAG} analyses: {', '.join(analyses)} (development data only: nothing after {SEAL_LAST_DEV_DATE} is read)",
        f"{TAG} as in es.run_economics: XGBoost seed 42, rolling-origin, premium position, calibrated fees, P1 as reference "
        f"policy, calendar@count and VOL@count at P1's count, day bootstrap of {n_rep} resamples "
        f"(mean block {_econ.MEAN_BLOCK} days, seed {_econ.SEED})",
        f"{TAG} files go to: {out_dir}",
        f"{TAG} {LABEL}"])


def _now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_block_meta(out_dir, block, analyses, n_rep, argv):
    """block_meta.json: which block the CSVs beside it belong to (they carry only analysis ids), written before the run so
    that a failed run is labelled too."""
    out_dir.mkdir(parents=True, exist_ok=True)
    info = {"label": LABEL, "alert_block": block, "alert_block_name": PAPER_BLOCK_NAMES.get(block), "replaces": PRIMARY_BLOCK,
            "calendar_comparator_block": CALENDAR_BLOCK, "features": features(block), "n_features": len(features(block)),
            "primary_n_features": len(features(PRIMARY_BLOCK)), "analyses": list(analyses), "n_rep": n_rep,
            "out_dir": str(out_dir), "argv": list(argv), "started_utc": _now()}
    (out_dir / "block_meta.json").write_text(json.dumps(info, indent=2) + "\n")


def _load_inputs():
    """(panel, fmm, nyi, eta, eta_source) through the development loaders of ``es.run_economics.main``, called the same way:
    the sealed-window guard is on (no ``dev_only`` override)."""
    eta_path = OUT_DIR / "analogs" / "protocol_inputs.json"
    with open(eta_path) as f:
        eta = json.load(f)["eta"]
    fmm = load_caiso_fmm(DATA_ROOT / "caiso_fmm_dev")
    nyi = load_nyiso_integrated(DATA_ROOT / "nyiso_rtint_dev")
    panel = build_panel(load_dev_raw())
    return panel, fmm, nyi, eta, str(eta_path)


def run_block(block, analyses, out_dir, n_rep=_econ.N_REP, argv=()):
    """Print the start-up message, write block_meta.json, load the development inputs and run ``run_all`` on the analyses
    (ids) with ``block`` as the alert model. ``block``, ``analyses`` and ``out_dir`` must have been checked."""
    print(describe_run(block, analyses, out_dir, n_rep), flush=True)
    _write_block_meta(out_dir, block, analyses, n_rep, argv)
    t0 = time.time()
    panel, fmm, nyi, eta, eta_source = _load_inputs()
    print(f"{TAG} inputs loaded in {time.time() - t0:.0f} s", flush=True)
    with alert_block(block):
        out = _econ.run_all(panel, fmm, nyi, eta, [ANALOGS[i] for i in analyses], out_dir, n_rep=n_rep, eta_source=eta_source)
    print(f"{TAG} done in {(time.time() - t0) / 60:.1f} min; files in {out_dir}", flush=True)
    return out


def _parser():
    ap = argparse.ArgumentParser(
        prog="python -m es.run_economics_block", allow_abbrev=False,
        description="The development economic analysis with another feature block as the alert model "
                    "(exploratory secondary analysis; development quarters only).")
    ap.add_argument("--block", required=True,
                    help="the alert block, a key of es.blocks.BLOCKS (the primary analysis uses G1_24): " + ", ".join(BLOCKS))
    ap.add_argument("--analyses", default=None,
                    help="comma list of keys of es.analyses.ANALOGS (default " + ",".join(DEFAULT_ANALYSES) + ")")
    ap.add_argument("--out-dir", default=None,
                    help="where the files go (default outputs/es/economics_block/<block>); never the primary folder "
                         "outputs/es/economics, and a folder that already holds a summary.csv needs --overwrite")
    ap.add_argument("--n-rep", type=int, default=_econ.N_REP,
                    help=f"day-bootstrap resamples (default {_econ.N_REP}, as the primary analysis)")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace the files of an earlier run in --out-dir (never the primary folder)")
    return ap


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = _parser()
    args = ap.parse_args(argv)
    try:                                                    # every refusal comes before any data is read
        block = check_block(args.block)
        analyses = parse_analyses(args.analyses)
        if args.n_rep < 1:
            raise ValueError(f"--n-rep must be a positive number of resamples, not {args.n_rep}")
        out_dir = check_out_dir(default_out_dir(block) if args.out_dir is None else args.out_dir, args.overwrite)
    except ValueError as exc:
        ap.error(str(exc))
    run_block(block, analyses, out_dir, n_rep=args.n_rep, argv=argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
