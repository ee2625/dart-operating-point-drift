"""Where the manuscript lives (outside the dart_dev repo) and where generated files go.
The manuscript is EnergySystems_Revision/paper/; code writes only to paper/generated/ and to
outputs/es/paper/ (new files; nothing existing in outputs/ is touched)."""
from __future__ import annotations

from pathlib import Path

from .config import OUT_DIR, REVISION_ROOT

PAPER_DIR = REVISION_ROOT / "paper"
GENERATED_DIR = PAPER_DIR / "generated"
TABLES_DIR = GENERATED_DIR / "tables"
FIGURES_DIR = GENERATED_DIR / "figures"
SECTIONS_DIR = PAPER_DIR / "sections"
APPENDICES_DIR = PAPER_DIR / "appendices"
ESM_DIR = PAPER_DIR / "esm"
AUTHOR_INPUTS = PAPER_DIR / "author_inputs.json"
PAPER_OUT_DIR = OUT_DIR / "paper"                     # implementation.json, coverage.csv
SEALED_OUT = {"A": OUT_DIR / "sealed" / "A", "B": OUT_DIR / "sealed" / "B"}
DOCS_SEALED = REVISION_ROOT / "docs" / "sealed"


def ensure_paper_dirs() -> None:
    for d in (GENERATED_DIR, TABLES_DIR, FIGURES_DIR, PAPER_OUT_DIR):
        d.mkdir(parents=True, exist_ok=True)
