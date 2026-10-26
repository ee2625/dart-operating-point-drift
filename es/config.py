"""Paths and constants for the Energy Systems revision (spec v2)."""
from __future__ import annotations

import os
from pathlib import Path

DEV_ROOT = Path(__file__).resolve().parents[1]          # .../EnergySystems_Revision/dart_dev
REVISION_ROOT = DEV_ROOT.parent                         # .../EnergySystems_Revision
DATA_ROOT = Path.home() / "dart-spread-data"
DEV_DATA_DIRS = (DATA_ROOT / "backfill_2023_2024", DATA_ROOT / "holdout_2026H1")
OUT_DIR = DEV_ROOT / "outputs" / "es"
CACHE_DIR = DEV_ROOT / "cache" / "es"

MARKETS = ("NYISO", "ISONE", "MISO", "CAISO")
#: Settlement clocks. MISO settles in fixed EST (see v2/panel_prep.py MARKET_TZ).
SETTLE_TZ = {"NYISO": "America/New_York", "ISONE": "America/New_York",
             "MISO": "Etc/GMT+5", "CAISO": "America/Los_Angeles"}

#: Last settlement date development code may read (spec invariant I1).
SEAL_LAST_DEV_DATE = "2026-06-30"
ALPHA = 0.05


def require_local_project_root() -> None:
    """Raise unless DART_PROJECT_ROOT points at the local revision workspace.

    Without it, src.paths resolves to the old Google Drive folder and writes could
    land in the collaborator's shared tree (FORENSICS §5b).
    """
    root = os.environ.get("DART_PROJECT_ROOT")
    if root is None or Path(root).resolve() != REVISION_ROOT.resolve():
        raise RuntimeError(
            f"Set DART_PROJECT_ROOT={REVISION_ROOT} before importing es "
            "(see EnergySystems_Revision/CLAUDE.md).")
    if not (REVISION_ROOT / "DART_Spike_v2_Clean").is_dir():
        raise RuntimeError(f"missing stub folder {REVISION_ROOT / 'DART_Spike_v2_Clean'}")


def ensure_dirs() -> None:
    for directory in (OUT_DIR, CACHE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
