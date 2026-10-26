"""Generated tables for the manuscript (spec v3 sections 5, 7.2, 8). Each builder reads saved outputs
through es.paper_macros.Sources and returns the LaTeX body of one table (tabular or longtable); captions
and labels stay in the section files, except for the longtables, which carry their own (they are included
bare, outside any float). A builder whose sources are absent is skipped and listed in
generated/tables/INDEX.md; a present source with a missing column raises.

Layout. The text width of the sn-jnl class is 372 pt, so the headers are stacked (\\shortstack) or grouped
(\\multicolumn with a \\cmidrule), text columns are p{} columns, and the widest tables are set in a group with
\\footnotesize and a tighter column separation. Longtables carry their own caption and label and are included
bare; those of the Online Resource set their own group. LaTeX does not check the width of a longtable (one that is
too wide sticks out of the text block without a warning), so tab_econ_ci and tab_decomposition, whose widths depend
on their cells, estimate their width panel by panel and raise when one is wider than the text width
(_check_text_width). The one sideways table, tab_chronology, is placed in a sidewaystable by its section.
tests/es/test_paper_tables.py compiles every table with the real class: it fails on an overfull box over 1 pt, on a
float that is cut off, on a sideways float that leaves the text block, and when the width estimate is below the
typeset width or the text width is not that of the class.
Run: DART_PROJECT_ROOT=... /opt/anaconda3/bin/python3 -m es.paper_tables"""
from __future__ import annotations

import argparse
import calendar
import importlib
import math
import re
import sys
from collections.abc import Mapping
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .analyses import ANALOGS, LABEL_CONTROL, LEGACY, SEALED
from .blocks import BLOCKS, PAPER_BLOCK_NAMES
from .config import OUT_DIR
from .paper_macros import PENDING, Sources
from .paper_paths import DOCS_SEALED, PAPER_DIR
from .paper_pins import ALPHAS, BANDS, SEEDS, band_text
from .paper_registry import CODE_CONSTANTS, CODE_READERS, DESCRIPTIVE_BH_FAMILIES, STAT_FMT, STAT_LABEL, STAT_ORDER
from .paper_tex import FORMATS, MINUS, PAIR_FORMATS, fmt_g, longtable, tabular, tex_escape
from .windows import UtcWindow

BUILDERS: dict[str, tuple[Callable[[Sources], str], tuple[str, ...]]] = {}
#: One line per table for generated/tables/INDEX.md (a choice the reader of the index should know about), and the lines that a builder
#: writes while it runs (a measurement of this build: RUNTIME_NOTES is cleared at the start of build_all).
NOTES: dict[str, str] = {}
RUNTIME_NOTES: dict[str, str] = {}
MARKET_NAME = {"NYISO": "NYISO", "ISONE": "ISO-NE", "MISO": "MISO", "CAISO": "CAISO"}
MARKETS = ("NYISO", "ISONE", "MISO", "CAISO")
AUDIT_BLOCKS = ("G6", "LEV6", "MID3", "DART3", "MIDDART6", "ROLL_LOC6", "ROLL_DISP9", "DART18", "G1_24")
CORE_BLOCKS = ("G6", "DART3", "LEV6", "ROLL_DISP9", "DART18", "G1_24")
ECON_IDS = ("AQ3", "AQ4", "AQ1", "AQ2", "2024-2025", "24H2-25H2", "25H1-26H1")
LABEL_CONTROL_IDS = ("LC-full-q2023", "LC-full-q2324", "LC-h1-q2023", "LC-h1-q2324")
LABEL_CONTROL_BLOCKS = ("LEV6", "DART18", "G1_24", "DART3")
#: The first band is the primary one, whose failures the `failures` column holds; the others have a column of their own.
PRIMARY_BAND = next(iter(BANDS))
#: Table footnotes carry a letter mark only (journal: superscript letters), never a number.
MARK_A = "\\textsuperscript{a}"
APPROX_TEXT = "Interval labelled approximate (tail measures and tests whose size check exceeds nominal)."
APPROX_NOTE = "\\footnotetext{" + MARK_A + " " + APPROX_TEXT + "}\n"
APPROX_MARK = MARK_A
D2 = FORMATS["dec2"]
POSITION_ORDER = ("premium", "inc", "dec", "premium[short_spikes]", "premium[long_spikes]")
ECON_POLICY_ORDER = ("P1", "P4", "P5", "P6", "P4@r*", "comparator")


# ---------------------------------------------------------------- what the code holds: read when a table is built, never typed or kept
def _code(key: str):
    """A constant that the registry reads from the code: from CODE_CONSTANTS or from CODE_READERS, read now (never kept from the import,
    so that a source that changes or is replaced in a test is the one that a table uses, and a module that is heavy to import is
    imported only by the table that needs it)."""
    return CODE_CONSTANTS[key] if key in CODE_CONSTANTS else CODE_READERS[key]()


def _base_alpha() -> float:
    """The cutoff target of the audit's primary cells (es.config.ALPHA, the registry's alertAlpha); the sweep of tab_alpha_band varies it."""
    return _code("alertAlpha")


def _primary_seed() -> int:
    """The seed of the primary cells (the default of es.scores.score_analysis, the registry's seedPrimary)."""
    return _code("seedPrimary")


def _p4_windows() -> tuple[int, int]:
    """The window of the rolling P4 and the sensitivity window, in days (the registry's code constants)."""
    return _code("pFourWindow"), _code("pFourWindowAlt")


def _p4_ids() -> tuple[str, str]:
    """The ids that es.run_analogs gives the activity rows of the two P4 windows."""
    return tuple(f"P4_{w}" for w in _p4_windows())


def _analog_policy_order() -> tuple[str, ...]:
    return ("P1", *_p4_ids(), "P6")


def _fee_order() -> tuple[str, ...]:
    """The fee scenarios of the economics summary in the order of es.run_economics: the calibrated fees, every uniform fee of
    es.positions.UNIFORM_FEES, the stress scenario; the producer labels each with str(fee)."""
    positions = importlib.import_module(".positions", __package__)
    return ("calibrated", *(str(fee) for fee in positions.UNIFORM_FEES), "stress")


class _Computed(Mapping):
    """A mapping from ids to texts that are computed from the code each time one is read: no year and no window is typed in a text
    that the code can say, and a registry that changes is the registry that the tables show."""

    def __init__(self, ids: Callable[[], tuple[str, ...]], text: Callable[[str], str]):
        self._ids, self._text = ids, text

    def __getitem__(self, key):
        if key not in self._ids():
            raise KeyError(key)
        return self._text(key)

    def __iter__(self):
        return iter(self._ids())

    def __len__(self):
        return len(self._ids())


def _days(window) -> tuple[date, date]:
    """(first day, last day) of an analysis window: a half-open UTC window ends the day before its end, a window of local dates
    (the analogues and the sealed tests) is inclusive."""
    first, end = date.fromisoformat(window.start), date.fromisoformat(window.end)
    return (first, end - timedelta(days=1)) if isinstance(window, UtcWindow) else (first, end)


def _calendar(window) -> tuple[str, int, int]:
    """('years', first year, last year), ('half', year, 1 or 2) or ('quarter', year, 1 to 4): what a window is made of, when it is
    whole calendar years, one half year or one quarter. Any other window is an error: a label that names it has to say what it is."""
    first, last = _days(window)
    if (first.month, first.day) == (1, 1) and (last.month, last.day) == (12, 31):
        return "years", first.year, last.year
    if first.year == last.year and first.day == 1:
        if (first.month, last.month, last.day) == (1, 6, 30):
            return "half", first.year, 1
        if (first.month, last.month, last.day) == (7, 12, 31):
            return "half", first.year, 2
        quarter = (first.month - 1) // 3 + 1
        if first.month == 3 * quarter - 2 and last == date(first.year, 3 * quarter, calendar.monthrange(first.year, 3 * quarter)[1]):
            return "quarter", first.year, quarter
    raise ValueError(f"the window {window.start} to {window.end} is not whole calendar years, one half year or one quarter: "
                     "a row label cannot say what it is")


def _span(window) -> str:
    """'2025', '2023--24', '2026-H1' or '2025-Q3': the span of a window, as the paper writes it."""
    kind, a, b = _calendar(window)
    if kind == "years":
        return str(a) if a == b else f"{a}--{b % 100:02d}"
    return f"{a}-H{b}" if kind == "half" else f"{a}-Q{b}"


def _tag(window) -> str:
    """'24H1' for a half year, '2024' for a calendar year: how the ids of the legacy analyses name the windows they compare."""
    kind, a, b = _calendar(window)
    if kind == "half":
        return f"{a % 100:02d}H{b}"
    if kind == "years" and a == b:
        return str(a)
    raise ValueError(f"the window {window.start} to {window.end} is not a calendar year or a half year")


#: The words that follow the windows in the name of a legacy analysis that is not the plain one of its pair.
_LEGACY_SUFFIX = {"24H2-25H2-train2023": " (old fit)", "25H1-26H1": " (post-open)"}


def _analysis_name(aid: str) -> str:
    """The name of an analysis in a row label, from the windows of es.analyses."""
    if aid in ANALOGS:
        a = ANALOGS[aid]
        return f"{aid}: {_span(a.deploy)}" if a.maintenance == "rolling" else f"{aid.removesuffix('F')} frozen-origin"
    if aid in SEALED:
        a, test = SEALED[aid], "Test " + aid[1]                         # SA and SAF are Test A, SB and SBF Test B
        return f"{test}: {_span(a.deploy)}" if a.maintenance == "rolling" else f"{test} frozen-origin"
    if aid in LABEL_CONTROL:
        a = LABEL_CONTROL[aid]
        return f"calibrate {_span(a.calib)}, $q$ {_span(a.label_train)}"
    a = LEGACY[aid]
    if aid == "sealed26H1":
        return f"as-opened {_span(a.deploy)}"
    if aid == "frozen-26H1":
        return f"frozen-origin {_span(a.deploy)}"
    return f"{_tag(a.calib)}$\\to${_tag(a.deploy)}" + _LEGACY_SUFFIX.get(aid, "")


#: The analyses that have a shorter name for the tables whose columns leave little room (the status is in the chronology table), and the
#: id that es.run_economics writes for the pooled rehearsal, which wraps in a column of 4.7 cm.
SHORT_IDS = ("24H1-25H1", "24H2-25H2", "2024-2025", "25H1-26H1", "sealed26H1", "frozen-26H1")
REHEARSAL_ID = "AQ3+AQ4 (rehearsal of the pooled estimands)"


def _analysis_short(aid: str) -> str:
    if aid == REHEARSAL_ID:
        return "AQ3+AQ4 pooled (rehearsal)"
    a = LEGACY[aid]
    if aid in ("sealed26H1", "frozen-26H1"):
        return f"{'as-opened' if aid.startswith('sealed') else 'frozen'} {_tag(a.deploy)}"
    first, second = _tag(a.calib), _tag(a.deploy)
    return f"{first}$\\to${second[2:]}" if _calendar(a.calib)[0] == "years" else f"{first}$\\to${second}"     # 2024$\to$25, 24H1$\to$25H1


def _policy_names() -> dict[str, str]:
    """The display name of each cutoff policy. P4 carries its window from the code; P2 is the in-sample cutoff of es.policies.p2_insample
    (the quantile of the training scores of the season's months, or of every month), which the sealed audit reports as P2_season and
    P2_all (es.sealed_run)."""
    window, alt = _p4_windows()
    return {"P1": "P1 fixed, held-out", "P2_season": "P2 in-sample (season)", "P2_all": "P2 in-sample (all seasons)",
            "P4": f"P4 rolling ({window} d)", f"P4_{window}": f"P4 rolling ({window} d)",
            f"P4_{alt}": f"P4 rolling ({alt} d)", "P5": "P5 self-correcting", "P6": "P6 exact-$\\alpha$ (benchmark)",
            "P4@r*": "P4 at matched count (benchmark)"}


ANALYSIS_NAME = _Computed(lambda: (*LEGACY, *ANALOGS, *SEALED, *LABEL_CONTROL), _analysis_name)
ANALYSIS_SHORT = _Computed(lambda: (*SHORT_IDS, REHEARSAL_ID), _analysis_short)
POLICY_NAME = _Computed(lambda: tuple(_policy_names()), lambda code: _policy_names()[code])

#: How a table that prints the raw id of an analysis (the ledgers, the leave-one-operator-out and the monitor tables) prints it. The as-opened
#: 2026-H1 holdout of the earlier version is stored under the id "sealed26H1" (es.analyses, which the sealed code binds, and the saved outputs);
#: printed, the id must not read as one of the sealed tests.
PRINTED_ID = {"sealed26H1": "as-opened-26H1"}


def _aid(aid) -> str:
    """An analysis id as a table prints it, escaped for LaTeX."""
    return tex_escape(PRINTED_ID.get(str(aid), str(aid)))


# ---------------------------------------------------------------- layout helpers
def _stack(*lines: str, align: str = "c") -> str:
    """A column heading on several lines, so that a column of short numbers is as narrow as its numbers. A line that starts with
    '[' or '*' gets an empty group in front: after a line break (\\\\), LaTeX reads '[' as the optional length of the break."""
    safe = [("{}" + line) if line.startswith(("[", "*")) else line for line in lines]
    return "\\shortstack[" + align + "]{" + "\\\\".join(safe) + "}"


RAGGED = "\\raggedright\\hyphenpenalty=10000 "


def _rr(text: str) -> str:
    """The text of a paragraph-column (p{}) cell, set ragged right and without hyphenation: a justified narrow column leaves
    wide gaps, and a hyphenated 'het-eroskedastic' reads badly. The class's \\raggedright (sn-jnl) leaves \\\\ alone, so it is
    safe in the last column of a row as well."""
    return RAGGED + text


def _tab(colspec: str, header: list[str], rows: list[list[str]], midrules_before=(), groups=None) -> str:
    """paper_tex.tabular, with an optional row of group headings above the column headings. ``groups`` is a list of
    (text, span) that covers every column; an empty text leaves its columns blank, and a group of several columns has a
    \\cmidrule(lr) under it."""
    out = tabular(colspec, header, rows, midrules_before)
    if not groups:
        return out
    cells, rules, col = [], [], 1
    for text, span in groups:
        if not text:
            cells += [""] * span
        elif span == 1:
            cells.append(text)
        else:
            cells.append(f"\\multicolumn{{{span}}}{{c}}{{{text}}}")
            rules.append(f"\\cmidrule(lr){{{col}-{col + span - 1}}}")
        col += span
    lines = out.split("\n")                                                      # \begin{tabular}, \toprule, header, \midrule, ...
    return "\n".join(lines[:2] + [" & ".join(cells) + " \\\\"] + rules + lines[2:])


def _sized(tex: str, colsep: int = 3, size: str | None = "footnotesize") -> str:
    """The tabular(s) of one table in a group with a smaller font (``size``; None keeps the table font) and a tighter column
    separation (the wide economic tables); the footnote lines go after the group, at the size of the other footnotes."""
    return f"\\begingroup{'\\' + size if size else ''}\\setlength{{\\tabcolsep}}{{{colsep}pt}}\n{tex}\\endgroup\n"


def _breakable(tex: str, small: bool = False, colsep: float = 2.5) -> str:
    """A longtable whose underscores admit a line break after them (feature, policy and comparator names have no spaces);
    ``small`` also sets the group at 7 pt with a column separation of ``colsep`` pt, as the Online Resource ledgers need."""
    head = "\\begingroup" + (f"\\footnotesize\\setlength{{\\tabcolsep}}{{{colsep:g}pt}}" if small else "") + "\\renewcommand{\\_}{\\textunderscore\\allowbreak{}}\n"
    return head + tex + "\\endgroup\n"


def _runtime_note(name: str, text: str) -> None:
    RUNTIME_NOTES[name] = text


def _note_par(text: str) -> str:
    """A note under a longtable, which has no float to carry a \\footnotetext."""
    return f"\\par\\noindent {text}\\par\n"


def _sort(df: pd.DataFrame, *keys) -> pd.DataFrame:
    """Stable sort by columns; a key is a column name or (column, order), and values missing from an order come last."""
    out, cols = df.copy(), []
    for key in keys:
        col, order = (key, None) if isinstance(key, str) else key
        if order is not None:
            rank = {v: i for i, v in enumerate(order)}
            out[f"_rank_{col}"] = out[col].map(lambda v, rank=rank: rank.get(v, len(rank)))
            cols.append(f"_rank_{col}")
        cols.append(col)
    return out.sort_values(cols, kind="stable").drop(columns=[c for c in out.columns if c.startswith("_rank_")])


def _num(fmt: Callable, value) -> str:
    """A ledger cell: '--' where the statistic is undefined (NaN), the formatted number otherwise."""
    return "--" if pd.isna(value) else fmt(value)


def _range(lo, hi, fmt: Callable) -> str:
    """'lo--hi', or one value where the two end points print alike (a degenerate range)."""
    a, b = fmt(lo), fmt(hi)
    return a if a == b else f"{a}--{b}"


def _short(aid) -> str:
    return ANALYSIS_SHORT.get(aid, ANALYSIS_NAME.get(aid, tex_escape(aid)))


#: The analyses of bootstrap_ci.csv in reading order: the ECON_IDS, then the frozen-origin analogue that es.run_economics also runs.
#: Anything else (the pooled AQ3 + AQ4 rehearsal) follows.
ECON_ORDER = ECON_IDS + ("AQ3F",)


def _econ_order(ids) -> list[str]:
    """Economic analyses in ECON_ORDER, each full analysis followed by its ex-CAISO twin, then the rest."""
    def key(x):
        base, _, suffix = str(x).partition("|")
        return (ECON_ORDER.index(base) if base in ECON_ORDER else len(ECON_ORDER), str(x))
    return sorted(ids, key=key)


def _econ_label(aid) -> str:
    base, _, suffix = str(aid).partition("|")
    tail = ", without CAISO" if suffix == "ex-CAISO" else (", " + tex_escape(suffix) if suffix else "")
    return _short(base) + tail


def _starts(ids) -> set[int]:
    """Row indices where a midrule precedes a new full analysis (its ex-CAISO twin stays directly under it) when each id
    holds one row."""
    return {i for i, aid in enumerate(ids) if i > 0 and "|" not in str(aid)}


def ci_cell(point_fmt: str, ci_fmt: str, est, lo, hi, approx) -> str:
    return _approx(f"{FORMATS[point_fmt](est)} {PAIR_FORMATS[ci_fmt](lo, hi)}", approx)


def stat_cell(frame: pd.DataFrame, stat: str) -> str:
    """'estimate [lo, hi]' of one stat of a frame with columns stat, estimate, lo, hi, approximate; '--' where the frame has no
    row of it. Its number format is the registry's STAT_FMT."""
    g = frame[frame.stat == stat]
    pf, cf = STAT_FMT[stat]
    return "--" if g.empty else ci_cell(pf, cf, _one(g, stat).estimate, _one(g, stat).lo, _one(g, stat).hi, _one(g, stat).approximate)


def stat_rows(frame: pd.DataFrame, stats=STAT_ORDER, label_suffix: str = "") -> list[list[str]]:
    """One [label, 'estimate [lo, hi]'] row per stat of a frame with columns stat, estimate, lo, hi, approximate.
    Shared by the development, sealed, ex-CAISO and this-quarter estimand tables."""
    return [[STAT_LABEL[stat] + label_suffix, stat_cell(frame, stat)] for stat in stats]


def builder(name: str, *needs: str, note: str = ""):
    def wrap(fn):
        BUILDERS[name] = (fn, needs)
        if note:
            NOTES[name] = note
        return fn
    return wrap


#: 'file.py:123' and 'file.py:123-130' in the rule strings of implementation.json: line numbers go stale, so the table
#: cites the file and drops the line (a citation that is the whole value is left as the path).
_LINE_REFERENCE = re.compile(r"(\.py):\d+(?:\s*-\s*\d+)?")
SECTION_SIGN = "\u00a7"


def _mask_lines(text: str) -> str:
    return _LINE_REFERENCE.sub(r"\1", text)


#: The pointers of implementation.json and coverage.csv ("§3.3", "§6.1.1") name a section by the number that it had when the string was
#: written (outputs/es/paper/implementation.json is a saved output that no task changes). A typed number goes stale the day a section
#: moves, so the tables print none: each pointer is written as the title of the heading that it meant, through the cross-reference
#: macros of paper/crossref.tex. The paper's sections are \mainref{key} (the tables are in Online Resource 1, so a reference to the paper
#: names the heading by its title), and a section of Online Resource 1 is a \ref (the tables are printed there). Plan 2C task T2b.
#: The key of a heading is the line of the key table that holds its title; tests/es/test_paper_crossref.py checks that every key names a
#: heading, and tests/es/test_impl_table.py that each pointer means the heading that it names.
SECTION_POINTERS = {
    "3.3": r"\mainref{data-blocks}",                  # Feature blocks
    "3.4": r"\mainref{data-analyses}",                # Analyses and windows
    "3.7.1": r"\mainref{data-econ-positions}",        # Positions and fees
    "3.7.3": r"\mainref{data-econ-policies}",         # Cutoff policies and comparators
    "6.1.1": r"Section~\ref{sec:sealed-windows}",     # Windows, exclusions and daylight saving: a section of Online Resource 1 since plan 2C task T2a
}
_POINTER = re.compile(SECTION_SIGN + r"\s*(\d+(?:\.\d+)*)")


def _pointers(text: str) -> str:
    """The text with every typed section pointer ('§3.3') replaced by its SECTION_POINTERS entry. A pointer without an entry is an error: the
    table must not print a section number that nothing keeps right."""
    def name(m: re.Match) -> str:
        if m.group(1) not in SECTION_POINTERS:
            raise LookupError(f"the section pointer {SECTION_SIGN}{m.group(1)} has no entry in es.paper_tables.SECTION_POINTERS: add the key of the "
                              f"heading that it means (paper/crossref.tex), so that no typed section number is printed")
        return SECTION_POINTERS[m.group(1)]
    return _POINTER.sub(name, text)


def _esc(value) -> str:
    """Plain text in, LaTeX out, for the rule strings and notes of implementation.json: the line numbers of code citations
    are dropped, tex_escape runs on the rest, and a section pointer ('§3.3', the one non-ASCII character those strings carry) is
    written as the title of its heading (SECTION_POINTERS), so that a source file stays valid under any input encoding and no section
    number is typed."""
    return _pointers(tex_escape(_mask_lines(str(value)))).replace(SECTION_SIGN, "\\S{}")          # a bare section sign (no number) stays \S


def _need(df: pd.DataFrame, cols, where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise LookupError(f"{where}: missing columns {missing}; have {list(df.columns)}")


def _one(df: pd.DataFrame, where: str) -> pd.Series:
    if len(df) != 1:
        raise LookupError(f"{where}: expected one row, got {len(df)}")
    return df.iloc[0]


def _summary(src: Sources) -> pd.DataFrame:
    s = src.frame("legacy/summary.csv")
    _need(s, ["analysis", "block", "estimator", "seed", "policy", "alpha", "failures", "median_R", "median_log_error",
              "median_auprc", "prevalence"], "legacy/summary.csv")
    return s


def _cell(s: pd.DataFrame, analysis: str, block: str, est: str, col: str, fmt, seed=None, policy="P1", alpha=None) -> str:
    """The ``col`` of the one summary row of an analysis, block, estimator, seed (the primary one unless given), policy and cutoff target
    (the audit's unless given); the pending sentinel where there is none."""
    seed, alpha = _primary_seed() if seed is None else seed, _base_alpha() if alpha is None else alpha
    sel = s[(s.analysis == analysis) & (s.block == block) & (s.estimator == est) & (s.seed == seed)
            & (s.policy == policy) & (abs(s.alpha - alpha) < 1e-12)]
    if sel.empty:
        return PENDING["dev"]
    return fmt(_one(sel, f"summary[{analysis},{block},{est},{policy},{alpha}]")[col])


def _approx(value: str, flag) -> str:
    """The value with the letter mark of an approximate interval; a missing flag is an error, not a mark."""
    if pd.isna(flag):
        raise ValueError("the approximate flag is missing")
    return value + (APPROX_MARK if bool(flag) else "")


def _win_text(w) -> str:
    return f"[{w.start}, {w.end})" if isinstance(w, UtcWindow) else f"{w.start} -- {w.end}"


# ---------------------------------------------------------------- the page: how wide a table may be
#: Glyph widths in points at the size of the tables (the class's 8 bp body font, Computer Modern), measured one by one with the real
#: class. They only feed the width guard below; kerning and ligatures change a word by a fraction of a percent, so the sum is scaled
#: up by GLYPH_SAFETY, and tests/es/test_paper_tables.py compares the estimate with the tables as typeset.
_GLYPH_PT = {
    **dict(zip("abcdefghijklmnopqrstuvwxyz", (3.99, 4.42, 3.56, 4.42, 3.56, 2.48, 3.99, 4.42, 2.26, 2.48, 4.20, 2.26, 6.57, 4.42, 3.99, 4.42,
                                                 4.20, 3.13, 3.17, 3.13, 4.42, 4.20, 5.71, 4.20, 4.20, 3.56))),
    **dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", (5.90, 5.59, 5.71, 6.02, 5.38, 5.16, 6.19, 5.90, 2.89, 4.08, 6.12, 4.94, 7.19, 5.90, 6.14, 5.38,
                                                 6.14, 5.81, 4.42, 5.71, 5.90, 5.90, 8.06, 5.90, 5.90, 4.85))),
    **{digit: 3.99 for digit in "0123456789"},
    " ": 2.70, ",": 2.26, ".": 2.26, ";": 2.26, ":": 2.26, "!": 2.26, "?": 3.77, "'": 2.26, "(": 3.13, ")": 3.13, "[": 2.26, "]": 2.26, "/": 3.99,
    "=": 6.14, "+": 6.14, "-": 2.69, "*": 3.99, "@": 6.14, "|": 7.97, "%": 6.57, "&": 6.14, "#": 6.57,
    "\u2212": 6.25, "\u00aa": 3.90, "\u00a4": 3.99, "\u2017": 2.87, "\u00a7": 3.56, "\u2192": 7.97}   # minus, the letter mark, $, _, section sign, arrow
_GLYPH_UNKNOWN_PT = 6.2                                                  # any other character: as wide as a wide letter
GLYPH_SAFETY = 1.01
#: The text width of the sn-jnl class in points (tests/es/test_paper_tables.py compares it with \\textwidth of the class), and a
#: centimetre in TeX points. LaTeX does not check the width of a longtable: one that is too wide sticks out of the text block
#: without a warning, and the width of a column that is not a p-column is only known after a second run. So the tables whose width
#: depends on their cells are set in p-columns of known width, and _check_text_width raises when the columns do not fit the text.
TEXT_WIDTH_PT = 372.0
CM_PT = 72.27 / 2.54
WIDTH_MARGIN_PT = 1.0           # what the check leaves below the text width
#: The width in points of each panel that the last build of a guarded table set (tests/es/test_paper_tables.py compares it with the
#: sum of the columns in the LaTeX it wrote).
GEOMETRY: dict[str, tuple[float, ...]] = {}

#: What the cells of the generated tables write for a few glyphs, as the single character that _GLYPH_PT has the width of.
_TEX_GLYPHS = ((RAGGED, ""), ("\\textsuperscript{a}", "\u00aa"), ("\\textminus{}", "\u2212"), ("$-$", "\u2212"),
               ("\\$", "\u00a4"), ("\\%", "%"), ("\\_", "\u2017"), ("\\S{}", "\u00a7"), ("$\\to$", "\u2192"), ("{,}", ","))


def _tex_width_pt(cell: str) -> float:
    """The width in points that a generated cell takes in the tables' font (a generous estimate: see _GLYPH_PT)."""
    s = cell.removeprefix("\\midrule")
    stack = re.fullmatch(r"\\shortstack\[[lcr]\]\{(.*)\}", s)
    if stack:                                                # a stacked heading is a box: as wide as its widest line
        return max(_tex_width_pt(line) for line in stack.group(1).split("\\\\"))
    for old, new in _TEX_GLYPHS:
        s = s.replace(old, new)
    s = re.sub(r"\\[A-Za-z]+", "x", s)                     # any other command counts as one average glyph
    s = re.sub(r"[{}$^_]", "", s)                          # braces and the markers of math mode and of sub- and superscripts print nothing
    return GLYPH_SAFETY * sum(_GLYPH_PT.get(ch, _GLYPH_UNKNOWN_PT) for ch in s)


def _fit_widths(header: list[str], rows: list[list[str]], fixed: dict[int, float]) -> list[float]:
    """The width in points of each column of a table that is set in p-columns: the columns in ``fixed`` (index -> points) as given,
    the others as wide as their widest cell, rounded up to 0.1 pt, so that no cell of them wraps."""
    return [fixed[j] if j in fixed else math.ceil(max(_tex_width_pt(cell) for cell in [head] + [row[j] for row in rows]) * 10) / 10
            for j, head in enumerate(header)]


def _p_colspec(widths: list[float]) -> str:
    return "".join(f"p{{{w:.1f}pt}}" for w in widths)


#: Geometry of a longtable of p-columns at \footnotesize in the sn-jnl class, measured with it (a line of a cell 8.0 pt; the rules and the
#: heading row that every page repeats 19.2 pt; the caption of the first page with the skip above it about 30 pt) and \textheight.
TEXT_HEIGHT_PT = 552.7
LINE_PT, HEAD_PT, CAPTION_PT = 8.0, 19.2, 30.0


def _lines(cell: str, width_pt: float) -> int:
    """The lines that a ragged-right cell of a p-column of ``width_pt`` takes, by greedy filling (a word is never split)."""
    count, used = 1, 0.0
    for word in cell.removeprefix("\\midrule").replace(RAGGED, "").split(" "):
        w = _tex_width_pt(word)
        if used and used + _GLYPH_PT[" "] + w > width_pt:
            count, used = count + 1, w
        else:
            used += (_GLYPH_PT[" "] if used else 0.0) + w
    return count


def _longtable_pages(widths: list[float], rows: list[list[str]], colsep: float, caption: bool = True) -> int:
    """The pages that a longtable of p-columns of ``widths`` takes (an estimate from the widths of its cells): each row is as tall as
    its tallest cell, a row that does not fit what is left of a page starts the next one (longtable never splits a row), and every page
    repeats the heading. A row of one cell spans all the columns."""
    spanning = sum(widths) + 2 * colsep * (len(widths) - 1)
    room = TEXT_HEIGHT_PT - HEAD_PT
    pages, used = 1, CAPTION_PT if caption else 0.0
    for row in rows:
        height = LINE_PT * (max(_lines(c, w) for c, w in zip(row, widths)) if len(row) == len(widths) else _lines(row[0], spanning))
        if used + height > room:
            pages, used = pages + 1, 0.0
        used += height
    return pages


def _check_text_width(name: str, panels, colsep: float) -> tuple[float, ...]:
    """The width in points of each panel of the table ``name``, from the widths of its p-columns and the separations (two for each
    column, ``colsep`` pt each), recorded in GEOMETRY; a ValueError when a panel is wider than the text width of the class, or when
    an unbreakable word of a cell is wider than its column (it would overflow it). A panel is (header, rows, widths)."""
    totals = []
    for number, (header, rows, widths) in enumerate(panels, 1):
        for j, width in enumerate(widths):
            words = [(_tex_width_pt(w), w) for cell in [header[j]] + [row[j] for row in rows]
                     for w in ([cell] if cell.startswith("\\shortstack") else cell.removeprefix("\\midrule").replace(RAGGED, "").split(" ")) if w]
            longest, word = max(words, default=(0.0, ""))
            if longest > width:
                raise ValueError(f"{name}: the word {word!r} is about {longest:.0f} pt wide, more than the {width:.0f} pt of column {j + 1} "
                                 f"of panel {number}: widen the column")
        totals.append(sum(widths) + 2 * colsep * len(widths))
    GEOMETRY[name] = tuple(totals)
    wide = [(number, w) for number, w in enumerate(totals, 1) if w > TEXT_WIDTH_PT - WIDTH_MARGIN_PT]
    if wide:
        raise ValueError(f"{name}: panel {wide[0][0]} is {wide[0][1]:.0f} pt wide, more than the {TEXT_WIDTH_PT:.0f} pt text width of the "
                         "class: shorten the cells or move a column to another panel")
    return tuple(totals)


# ---------------------------------------------------------------- section 3: data and design
@builder("tab_panel", "legacy/table1_panel.csv")
def tab_panel(src: Sources) -> str:
    t = src.frame("legacy/table1_panel.csv")
    _need(t, ["market", "nodes", "hourly_rows", "hubs", "zones", "prevalence_24_q2023"], "table1_panel")
    t = t.set_index("market")
    rows = [[MARKET_NAME[m], f"{FORMATS['int'](t.loc[m, 'nodes'])} ({FORMATS['int'](t.loc[m, 'hubs'])} / {FORMATS['int'](t.loc[m, 'zones'])})",
             FORMATS["int"](t.loc[m, "hourly_rows"]), FORMATS["dec3"](t.loc[m, "prevalence_24_q2023"])] for m in MARKETS]
    rows.append(["all", FORMATS["int"](t["nodes"].sum()), FORMATS["int"](t["hourly_rows"].sum()), "--"])
    return tabular("lccc", ["Operator", "nodes (hub / zone)", "hourly rows", "prevalence 2024"], rows, midrules_before={4})


@builder("tab_chronology", note="set sideways: place it in a sidewaystable")
def tab_chronology(src: Sources) -> str:
    ids = ["24H1-25H1", "24H2-25H2", "2024-2025", "sealed26H1", "25H1-26H1", "frozen-26H1", "AQ3", "AQ4", "AQ1", "AQ2", "AQ3F", "SA", "SAF", "SB", "SBF"]
    reg = {**LEGACY, **ANALOGS, **SEALED}
    rows = [[ANALYSIS_NAME[i], reg[i].status, reg[i].maintenance, _win_text(reg[i].train), _win_text(reg[i].calib), _win_text(reg[i].deploy)]
            for i in ids]
    return tabular("llllll", ["analysis", "status", "maintenance", "train", "calibrate", "deploy"], rows, midrules_before={6, 11})


@builder("tab_volatility", "legacy/fig1_volatility.csv")
def tab_volatility(src: Sources) -> str:
    f = src.frame("legacy/fig1_volatility.csv")
    _need(f, ["kind", "da_std", "dart_std", "dart_p95", "dart_p95_pooled"], "fig1_volatility")
    label = f.columns[0]
    q = f[f["kind"] == "quarter"]                     # the CSV also holds "period" and "period_local" summary rows
    rows = [[tex_escape(r[label]), FORMATS["dec1"](r["da_std"]), FORMATS["dec1"](r["dart_std"]), FORMATS["dec1"](r["dart_p95"]),
             FORMATS["dec1"](r["dart_p95_pooled"])] for _, r in q.iterrows()]
    return tabular("lrrrr", ["quarter", _stack("DA price std", "(node median)"), _stack("DART std", "(node median)"),
                             _stack("p95 $|$DART$|$", "(node median)"), _stack("p95 $|$DART$|$", "(pooled)")], rows)


def _impl(src: Sources) -> dict:
    return src.json("paper/implementation.json")


def _kv(value, nested: bool = False) -> str:
    """A JSON value as table text: null is a pending cell, floats get six significant digits, strings are escaped, a dict is
    'key: value; key: value' and a dict inside a dict is parenthesised, so that the separators stay unambiguous."""
    if value is None:
        return PENDING["dev"]
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, dict):
        inner = ("; " if not nested else ", ").join(f"{_esc(k)}: {_kv(v, True)}" for k, v in value.items())
        return f"({inner})" if nested else inner
    if isinstance(value, (list, tuple)):
        return ", ".join(_kv(v, nested) for v in value)
    if isinstance(value, float):
        return fmt_g(value, 6)
    return _esc(value)


def _bands(value) -> str:
    """Failure bands as the paper writes them (the bandWide, bandMid and bandTight macros): [0.5, 2], [0.67, 1.5], ..."""
    if value is None:
        return PENDING["dev"]
    return ", ".join(band_text(v) for v in value.values())


#: The rows of tab_implementation read from rules, in reading order: (rules key, row label). A rule the table has no label
#: for is appended under its key, so that nothing in implementation.json is dropped silently.
IMPL_ROWS = (("calendar", "calendar features"), ("information_policy", "information policy"), ("missing_data", "missing data"),
             ("deduplication", "deduplication"), ("label", "label"), ("cutoff_P1", "cutoff"), ("alpha", "target alert rate"),
             ("alpha_sweep", "alpha sweep"), ("failure_bands", "failure bands"), ("zero_alert_floor", "zero-alert cells"),
             ("feasibility", "feasibility"), ("seeds", "seeds"), ("time_zones", "settlement clocks"),
             ("window_conventions", "window conventions"), ("dst", "daylight saving"),
             ("sealed_last_dev_date", "last development date"), ("P4_windows_days", "P4 windows (days)"),
             ("P5_eta_grid", "P5 step grid"), ("P5_clip", "P5 rate bounds"), ("P4_matched", "P4 at matched count"),
             ("bootstrap", "bootstrap"), ("es_fraction_pct", "expected-shortfall tail (\\%)"),
             ("fees_calibrated", "fees, calibrated (\\$/MWh)"), ("fees_uniform_grid", "fees, uniform grid"),
             ("stress_fee", "stress fee"), ("position_mw", "position"), ("lambda_grid", "value-function $\\lambda$ grid"),
             ("strata_primary", "SELC strata (primary)"), ("strata_sensitivity", "SELC strata (sensitivity)"),
             ("monitors", "drift monitors"), ("coverage_rule", "coverage rule"))
#: Labels of the spec-fixed entries (rules["spec_fixed"]); a key the table has no label for is shown under its key.
SPEC_LABELS = {"P4_windows_days": "P4 windows (days)", "P5_eta_grid": "P5 step grid", "position_mw": "position",
               "information_shift_h": "rolling-window shift (h)", "feature_lookback_h": "feature lookback (h)",
               "coverage_rule": "coverage rule"}
#: Rules folded into another row (the MISO note is part of the settlement clocks) or rendered elsewhere (spec_fixed).
IMPL_FOLDED = ("miso_note", "spec_fixed")
IMPLEMENTATION_CAPTION = ("Implementation settings. The first block is read from the code and the data; the second lists the values "
                          "that the design of the study fixes and that no code constant holds, each with the section of the paper that describes it.")


def _impl_value(key: str, rules: dict) -> str:
    if key == "failure_bands":
        return _bands(rules.get(key))
    if key == "position_mw":
        return f"{_kv(rules.get(key))} MW every hour"
    if key == "time_zones" and rules.get("miso_note"):
        return f"{_kv(rules.get(key))}; {_esc(rules['miso_note'])}"
    return _kv(rules.get(key))


def _sections(text: str) -> set[str]:
    """The section references ('§6.1', '§4.3') of a text."""
    return set(re.findall(SECTION_SIGN + r"\s*\d+(?:\.\d+)*", text))


def _spec_row_text(key: str, rules: dict, sentence: str) -> str:
    """The setting of a spec-fixed row: the value that the rules hold, then the sentence that says where the spec fixes it. A
    rule that is itself a text and already carries the sentence's section references is shown once (M4)."""
    if key not in rules:
        return _esc(sentence)
    shown = _impl_value(key, rules)
    if isinstance(rules[key], str) and _sections(sentence) <= _sections(rules[key]):
        return shown
    return f"{shown}; {_esc(sentence)}"


IMPLEMENTATION_CM, IMPLEMENTATION_TEXT_SHARE, IMPLEMENTATION_COLSEP = 2.9, 0.64, 2.5      # the p-columns: item in cm, setting in shares of \textwidth


@builder("tab_implementation", "paper/implementation.json",
         note="two blocks (read from the code; fixed by the design of the study, each with its section of the paper); a longtable with its own caption and label "
              "(tab:implementation) that sets its own font size: include it bare, not in a table float")
def tab_implementation(src: Sources) -> str:
    """Settings of the pipeline in two blocks (Ruling 65): the rules read from the code and the data, then the values that
    the design of the study fixes and that no code constant carries, each with the section of the paper that describes it (a
    section sign in its text). A longtable (Ruling 86): with the full rule texts it is longer than a float page."""
    d = _impl(src)
    r = d["rules"]
    spec = r.get("spec_fixed") or {}
    for key, text in spec.items():
        if SECTION_SIGN not in str(text):
            raise LookupError(f"implementation.json rules.spec_fixed[{key!r}] carries no reference to a section of the paper (a '{SECTION_SIGN}'): "
                              f"{text!r}")
    head = lambda text, rule="": [f"{rule}\\multicolumn{{2}}{{l}}{{\\textit{{{text}}}}}"]
    row = lambda label, value: [_rr(label), _rr(value)]
    rows = [head("Read from the code and the data"),
            row("estimator (primary)", "XGBoost: " + _kv(d["xgboost"])), row("estimator (contrast)", "logistic regression: " + _kv(d["logistic"])),
            row("scaler", _kv(d["scaler"]))]
    listed = {k for k, _ in IMPL_ROWS}
    rows += [row(label, _impl_value(key, r)) for key, label in IMPL_ROWS if key not in spec]
    rows += [row(_esc(k), _impl_value(k, r)) for k in r if k not in listed and k not in IMPL_FOLDED]
    rows.append(row("software", _kv(d["versions"])))
    rows.append(head("Fixed by the design of the study (no code constant holds the value)", rule="\\midrule"))
    for key, text in spec.items():
        rows.append(row(SPEC_LABELS.get(key, _esc(key)), _spec_row_text(key, r, text)))
    widths = [IMPLEMENTATION_CM * CM_PT, IMPLEMENTATION_TEXT_SHARE * TEXT_WIDTH_PT]
    _runtime_note("tab_implementation", f"about {_longtable_pages(widths, rows, IMPLEMENTATION_COLSEP)} pages long at footnotesize "
                                        "(an estimate from the width of the cells)")
    return _breakable(longtable(f"p{{{IMPLEMENTATION_CM:g}cm}}p{{{IMPLEMENTATION_TEXT_SHARE:g}\\textwidth}}", ["item", "setting"], rows,
                                IMPLEMENTATION_CAPTION, "tab:implementation"), small=True, colsep=IMPLEMENTATION_COLSEP)


@builder("tab_blocks_features", "paper/implementation.json")
def tab_blocks_features(src: Sources) -> str:
    b = _impl(src)["blocks"]
    rows = [[_rr("calendar (G6, in every model)"), FORMATS["int"](len(b["calendar_G6"])), _rr(", ".join("\\texttt{" + tex_escape(f) + "}" for f in b["calendar_G6"]))]]
    for key, v in b.items():
        if key == "calendar_G6":
            continue
        rows.append([_rr(f"{tex_escape(v['paper_name'])} ({tex_escape(key)})"), FORMATS["int"](v["n_price_features"]),
                     _rr(", ".join("\\texttt{" + tex_escape(f) + "}" for f in v["price_features"]) or "--")])
    return _breakable(longtable("p{3.1cm}rp{0.54\\textwidth}", ["block", "features", "feature names"], rows,
                                "Feature lists per block, read from \\texttt{es.blocks}", "tab:blocks-features"))


@builder("tab_coverage", "paper/implementation.json")
def tab_coverage(src: Sources) -> str:
    """Coverage of each series, with implementation.json["coverage_notes"] as a notes list under the table (Ruling 65)."""
    impl = _impl(src)
    cov = impl.get("coverage")
    notes = "".join(f"\\footnotetext{{{_esc(n)}}}\n" for n in impl.get("coverage_notes") or [])
    if not cov:
        return tabular("l", ["coverage"], [[PENDING["dev"] + " (run build.sh impl --with-data)"]]) + notes
    rows = [[_rr(tex_escape(MARKET_NAME.get(r["series"], r["series"]))), FORMATS["int"](r["nodes"]), FORMATS["int"](r["hourly_rows"]),
             tex_escape(r["first_local_date"]), tex_escape(r["last_local_date"])] for r in cov]
    return tabular("p{4.0cm}rrll", ["series", "nodes", "hourly rows", "first local date", "last local date"], rows) + notes


# ---------------------------------------------------------------- section 4: audit
def _xgb_logit_groups() -> list[tuple[str, int]]:
    """The headings above the XGB and logit column pairs of tab_block_audit and tab_estimators: the analyses of the two windows and the
    frozen origin, named from the windows of es.analyses."""
    return [("", 1), ("", 1), (ANALYSIS_NAME["2024-2025"], 2), (ANALYSIS_SHORT["25H1-26H1"], 2),
            (_stack("frozen", _span(LEGACY["frozen-26H1"].deploy)), 1)]


@builder("tab_block_audit", "legacy/summary.csv")
def tab_block_audit(src: Sources) -> str:
    s = _summary(src)
    I = FORMATS["int"]
    rows = []
    for b in AUDIT_BLOCKS:
        rows.append([PAPER_BLOCK_NAMES[b], I(len(BLOCKS[b])),
                     _cell(s, "2024-2025", b, "xgb", "failures", I), _cell(s, "2024-2025", b, "logit", "failures", I),
                     _cell(s, "25H1-26H1", b, "xgb", "failures", I), _cell(s, "25H1-26H1", b, "logit", "failures", I),
                     _cell(s, "frozen-26H1", b, "xgb", "failures", I)])
    return _tab("lrrrrrr", [_stack("block added", "to calendar", align="l"), "dim", "XGB", "logit", "XGB", "logit", "XGB"], rows,
                groups=[("", 1), ("", 1)] + _xgb_logit_groups()[2:])


@builder("tab_estimators", "legacy/summary.csv")
def tab_estimators(src: Sources) -> str:
    s = _summary(src)

    def fm(a, b, e):
        f, m = _cell(s, a, b, e, "failures", FORMATS["int"]), _cell(s, a, b, e, "median_log_error", D2)
        return f"{f} / {m}"
    rows = [[PAPER_BLOCK_NAMES[b], fm("2024-2025", b, "xgb"), fm("2024-2025", b, "logit"), fm("25H1-26H1", b, "xgb"),
             fm("25H1-26H1", b, "logit"), fm("frozen-26H1", b, "xgb")] for b in CORE_BLOCKS]
    return _tab("lccccc", ["block", "XGB", "logit", "XGB", "logit", "XGB"], rows,
                groups=[("", 1)] + _xgb_logit_groups()[2:])


@builder("tab_holdout", "legacy/summary.csv")
def tab_holdout(src: Sources) -> str:
    s = _summary(src)

    def fr(a, b):
        return f"{_cell(s, a, b, 'xgb', 'failures', FORMATS['int'])} ({_cell(s, a, b, 'xgb', 'median_R', D2)})"
    sealed, matched = LEGACY["sealed26H1"], LEGACY["25H1-26H1"]                # the windows in the labels are those of es.analyses
    rows = [[f"as opened: train {_span(sealed.train)}, calibrate {_span(sealed.calib)}, deploy {_span(sealed.deploy)}, "
             f"$q$ on {_span(sealed.label_train)}"] + [fr("sealed26H1", b) for b in ("LEV6", "DART18", "G1_24")],
            [f"season-matched (post-open): calibrate {_span(matched.calib)}, $q$ on {_span(matched.label_train)}"]
            + [fr("25H1-26H1", b) for b in ("LEV6", "DART18", "G1_24")]]
    return tabular("p{0.5\\textwidth}ccc", ["comparison", "level", "DART-relative", "combined"], rows)


@builder("tab_alpha_band", "legacy/summary.csv")
def tab_alpha_band(src: Sources) -> str:
    s = _summary(src)
    I = FORMATS["int"]
    band_cols = tuple((f"failures_band_{k}", k) for k in BANDS if k != PRIMARY_BAND)
    _need(s, [c for c, _ in band_cols], "legacy/summary.csv")
    rows = []
    for a in ("2024-2025", "25H1-26H1"):
        for b in ("LEV6", "DART18", "G1_24", "DART3"):
            cells = [_cell(s, a, b, "xgb", "failures", I, alpha=al) for al in ALPHAS]
            base = s[(s.analysis == a) & (s.block == b) & (s.estimator == "xgb") & (s.seed == _primary_seed()) & (s.policy == "P1")
                     & (abs(s.alpha - _base_alpha()) < 1e-12)]
            bands = [I(_one(base, "band")[c]) if not base.empty else PENDING["dev"] for c, _ in band_cols]
            rows.append([_short(a), PAPER_BLOCK_NAMES[b]] + cells + bands)

    def band_head(key):
        lo, hi = band_text(BANDS[key]).split(", ")
        return _stack(lo + ",", hi)
    header = ["analysis", "block"] + [f"{100 * al:g}\\%" for al in ALPHAS] + [band_head(k) for _, k in band_cols]
    return _tab("llrrrrrr", header, rows, midrules_before={4},
                groups=[("", 1), ("", 1), ("cutoff target $\\alpha$", len(ALPHAS)), (f"band at $\\alpha$={100 * _base_alpha():g}\\%", len(band_cols))])


@builder("tab_seeds", "legacy/summary.csv")
def tab_seeds(src: Sources) -> str:
    s = _summary(src)
    rows = []
    for a in ("2024-2025", "25H1-26H1"):
        for b in ("G6", "LEV6", "DART3", "ROLL_DISP9", "DART18"):
            sel = s[(s.analysis == a) & (s.block == b) & (s.estimator == "xgb") & (s.policy == "P1") & (abs(s.alpha - _base_alpha()) < 1e-12)]
            if sel["seed"].nunique() < len(SEEDS):
                rows.append([_short(a), PAPER_BLOCK_NAMES[b], PENDING["dev"], ""])
                continue
            rows.append([_short(a), PAPER_BLOCK_NAMES[b], _range(sel.failures.min(), sel.failures.max(), FORMATS["int"]),
                         _range(sel.median_log_error.min(), sel.median_log_error.max(), D2)])
    return tabular("llcc", ["analysis", "block", _stack("failures", "over seeds"), _stack("median $|\\log_{10} R|$", "over seeds")], rows,
                   midrules_before={5})


PR_COLS = ("precision_q", "recall_q", "precision_eval", "recall_eval", "precision_lift")
PR_GROUPS = [("training label $q$", 2), ("evaluation label", 2), ("", 1)]
PR_HEADS = ["precision", "recall", "precision", "recall", _stack("precision", "lift")]


@builder("tab_pr", "labels_pr/pr_all.csv")
def tab_pr(src: Sources) -> str:
    p = src.frame("labels_pr/pr_all.csv")
    _need(p, ["analysis", "block", "estimator", *PR_COLS], "pr_all")
    rows, starts = [], set()
    for a in ("2024-2025", "25H1-26H1", "AQ3", "AQ4"):
        starts.add(len(rows))
        for b in CORE_BLOCKS:
            g = p[(p.analysis == a) & (p.block == b) & (p.estimator == "xgb")]
            if g.empty:
                continue
            rows.append([_short(a), PAPER_BLOCK_NAMES[b]] + [D2(g[c].median()) for c in PR_COLS])
    return _tab("llrrrrr", ["analysis", "block"] + PR_HEADS, rows, midrules_before=starts - {0}, groups=[("", 1), ("", 1)] + PR_GROUPS)


@builder("tab_label_control", "labels_pr/label_control_2x2.csv",
         note="a cell reads failures/nodes when the cells have different node counts (n_nodes), failures alone otherwise")
def tab_label_control(src: Sources) -> str:
    c = src.frame("labels_pr/label_control_2x2.csv")
    _need(c, ["analysis", "block", "failures", "median_R"], "label_control_2x2")
    shown = c[c.analysis.isin(LABEL_CONTROL_IDS) & c.block.isin(LABEL_CONTROL_BLOCKS)]
    with_n = "n_nodes" in c.columns and shown["n_nodes"].nunique() > 1               # cells of different size must not be compared on failures alone
    rows = []
    for a in LABEL_CONTROL_IDS:
        cells = []
        for b in LABEL_CONTROL_BLOCKS:
            g = c[(c.analysis == a) & (c.block == b)]
            if g.empty:
                cells.append(PENDING["dev"])
                continue
            r = _one(g, a)
            count = FORMATS["int"](r["failures"]) + (f"/{FORMATS['int'](r['n_nodes'])}" if with_n else "")
            cells.append(f"{count} ({D2(r['median_R'])})")
        rows.append([ANALYSIS_NAME[a]] + cells)
    return tabular("lcccc", ["calibration window, label rule", "level", "DART-relative", "combined", "plain DART lags"], rows)


@builder("tab_monitors", "monitors/monitor_eval.csv")
def tab_monitors(src: Sources) -> str:
    m = src.frame("monitors/monitor_eval.csv")
    _need(m, ["analysis", "block", "monitor", "spearman", "auroc_fail"], "monitor_eval")
    names = {"W1_norm": "1-Wasserstein (normalised)", "PSI": "PSI on inputs", "domain_auc": "domain-classifier AUC",
             "early_rate": f"realised alert rate ({_code('earlyDays')} d)"}
    rows = []
    for mon, label in names.items():
        g = m[m.monitor == mon]
        rows.append([label, D2(g.spearman.median()), D2(g.auroc_fail.median()), FORMATS["int"](g.analysis.nunique()), FORMATS["int"](g.block.nunique())])
    return tabular("lrrrr", ["monitor", _stack("median Spearman", "with $|\\log_{10} R|$"), _stack("median AUROC", "for node failure"), "analyses", "blocks"], rows)


#: Calibration windows longer than this many days (a quarter is at most 92) are marked in the monitor tables: the two legacy
#: analyses calibrate on half a year and a year, where the domain classifier's AUC is optimistic, so only its rank is read.
LONG_CALIBRATION_DAYS = 100
LONG_CALIBRATION_NOTE = "Longer calibration window: the domain-AUC level is not interpretable, only its rank."


def _placebo_note() -> str:
    """What the no-drift reference column is (es.run_monitors, Ruling 83): the monitor's own value on a placebo split."""
    return (f"No-drift reference: the median over nodes of the monitor's own value (not of its correlation with the error) on a "
            f"placebo split inside the calibration window, its last {_code('earlyDays')} days against the earlier days. It is the level "
            "that the statistic takes without drift, descriptive only: it is not a monitor and enters no test. A dash: the alert rate "
            "has no reference.")


@builder("tab_monitors_full", "monitors/monitor_eval.csv")
def tab_monitors_full(src: Sources) -> str:
    return _monitors_longtable(src, "Drift monitors by analysis and block", "tab:monitors-full", ledger=False)


def _monitors_longtable(src: Sources, caption: str, label: str, ledger: bool) -> str:
    """Every analysis x block x monitor: the rank correlation with |log10 R| and its one-sided p-value (the monitors form one of
    the three exploratory BH families), the AUROC for node failure, the number of nodes behind them, and the no-drift reference
    (the placebo median, where the file has it). An analysis whose calibration window is longer than a quarter is marked. The
    ledger also gives the number of nodes behind the reference."""
    m = src.frame("monitors/monitor_eval.csv")
    _need(m, ["analysis", "block", "monitor", "spearman", "spearman_p_one_sided", "auroc_fail", "n_used", "calib_days"], "monitor_eval")
    placebo = "placebo_median" in m.columns
    if ledger and placebo:
        _need(m, ["placebo_n"], "monitor_eval")
    m = _sort(m, ("analysis", tuple(LEGACY) + tuple(ANALOGS)), ("block", tuple(BLOCKS)), "monitor")
    long_window = lambda r: pd.notna(r.calib_days) and r.calib_days > LONG_CALIBRATION_DAYS
    rows = []
    for r in m.itertuples():
        row = [_rr(_aid(r.analysis) + (MARK_A if long_window(r) else "")), _rr(PAPER_BLOCK_NAMES.get(r.block, tex_escape(r.block))),
               tex_escape(r.monitor), FORMATS["int"](r.n_used), _num(D2, r.spearman), _num(FORMATS["pval"], r.spearman_p_one_sided),
               _num(D2, r.auroc_fail)]
        if placebo:
            row.append(_num(D2, r.placebo_median))
            if ledger:
                row.append(_num(FORMATS["int"], r.placebo_n))
        rows.append(row)
    header = ["analysis", "block", "monitor", "nodes", "Spearman", _stack("$p$", "(one-sided)"), "AUROC"]
    colspec = "p{1.9cm}p{2.3cm}lrrrr"                        # analysis ids and block names wrap: the longest id is 19 characters
    if placebo:
        header.append(_stack("no-drift", "reference"))
        colspec += "r"
        if ledger:
            header.append(_stack("nodes", "(ref.)"))
            colspec += "r"
    notes = ""
    if any(long_window(r) for r in m.itertuples()):
        notes += _note_par(f"{MARK_A} {LONG_CALIBRATION_NOTE}")
    if placebo:
        notes += _note_par(_placebo_note())
    return _breakable(longtable(colspec, header, rows, caption, label) + notes, small=True, colsep=2)


@builder("tab_policies", "analogs/activity_all.csv", "analogs/eta_selection.csv", "analogs/protocol_inputs.json")
def tab_policies(src: Sources) -> str:
    a = src.frame("analogs/activity_all.csv")
    _need(a, ["analysis", "block", "estimator", "policy", "fail_0.5_2", "log_error", "R"], "activity_all")
    eta = src.frame("analogs/eta_selection.csv")
    _need(eta, ["analysis", "eta", "median_log_error"], "eta_selection")
    eta_star = float(src.json("analogs/protocol_inputs.json")["eta"])
    rows = []
    for aid in ("AQ3", "AQ4", "AQ1", "AQ2", "AQ3F"):
        cells = []
        for pol in _analog_policy_order():
            g = a[(a.analysis == aid) & (a.block == "G1_24") & (a.estimator == "xgb") & (a.policy == pol)]
            cells.append(PENDING["dev"] if g.empty else f"{FORMATS['int'](g['fail_0.5_2'].sum())} / {D2(g.log_error.median())}")
        e = eta[(eta.analysis == aid) & (abs(eta.eta - eta_star) < 1e-12)]
        cells.insert(3, "--" if e.empty else f"-- / {D2(_one(e, 'eta')['median_log_error'])}")
        rows.append([ANALYSIS_NAME[aid]] + cells)
    window, alt = _p4_windows()
    return tabular("lccccc", ["analogue", "P1", f"P4 ({window} d)", f"P4 ({alt} d)", f"P5 ($\\eta$={eta_star:g})", "P6"], rows)


@builder("tab_analog_blocks", "analogs/activity_all.csv")
def tab_analog_blocks(src: Sources) -> str:
    a = src.frame("analogs/activity_all.csv")
    _need(a, ["analysis", "block", "estimator", "policy", "fail_0.5_2"], "activity_all")
    rows = []
    for b in AUDIT_BLOCKS:
        cells = []
        for aid in ("AQ3", "AQ4", "AQ1", "AQ2"):
            for est in ("xgb", "logit"):
                g = a[(a.analysis == aid) & (a.block == b) & (a.estimator == est) & (a.policy == "P1")]
                cells.append(PENDING["dev"] if g.empty else FORMATS["int"](g["fail_0.5_2"].sum()))
        rows.append([PAPER_BLOCK_NAMES[b]] + cells)
    return _tab("l" + "rr" * 4, ["block"] + ["XGB", "logit"] * 4, rows, groups=[("", 1)] + [(q, 2) for q in ("AQ3", "AQ4", "AQ1", "AQ2")])


#: Words for the scenario and persistence columns of analogs/simulation.csv. H-E has homoskedastic and heteroskedastic profit-and-loss
#: noise; H-B has no drift and a common drift of both blocks by a number of score SDs (the id carries it: "common_drift_0.5"); the
#: persistence is the fitted one or the upper end of its interval.
SIM_SCENARIO = {"homoskedastic": "homoskedastic", "heteroskedastic": "heteroskedastic", "no_drift": "no drift"}
SIM_PERSISTENCE = {"fit": "fitted", "upper": "upper"}
#: The `test` values of simulation.csv that the registration rule reads (es.run_analogs.registration_family): H-E's size is that of
#: the pooled test, its power the single target row, and the curve is only reported.
HE_SIZE_TEST, HE_POWER_TEST, HE_CURVE_TEST, HB_SIZE_TEST = ("H-E pooled A+B, size", "H-E pooled A+B, power",
                                                           "H-E pooled A+B, power curve", "H-B size")
MARK_B = "\\textsuperscript{b}"
REGISTRATION_COLSEP = 3
REGISTRATION_CM = (1.5, 1.9, 2.9, 2.9)       # the p-columns: test, designated as, size at each level (the power column has its natural width)


def _registered_rows(sim: pd.DataFrame, test: str, laws=None) -> pd.DataFrame:
    """The simulation rows of ``test`` that the registration rule reads (es.run_analogs._test_rows): every scenario and persistence
    setting, under the registered (primary) strata, which is also what a row of the H-B tests has (no strata). Rows under the decile
    strata are the sensitivity: reported, never governing. With ``laws``, only the rows of those laws (a row with no law is read by
    every test): the rule record of protocol_inputs.json names H-B1's and H-B2's."""
    g = sim[sim["test"] == test]
    g = g[g["strata"].isna() | (g["strata"] == "primary")]
    if laws is not None:
        g = g[g["law"].isna() | g["law"].isin(list(laws))]
    return g


def _id_words(token: str) -> str:
    """A law, scenario or persistence id of simulation.csv in words (the producer's ids are not shown): a law is a block, whose paper
    name it takes; "common_drift_0.5" is a common drift of 0.5; an id that is none of them is escaped."""
    drift = re.fullmatch(r"common_drift_(.+)", token)
    if token in PAPER_BLOCK_NAMES:
        return PAPER_BLOCK_NAMES[token]
    return f"common drift {drift.group(1)}" if drift else SIM_SCENARIO.get(token, SIM_PERSISTENCE.get(token, _esc(token)))


def _where_words(row: pd.Series) -> str:
    """'law, scenario, persistence' of a simulation row in words, as far as it carries them."""
    return ", ".join(_id_words(str(row[c])) for c in ("law", "scenario", "persistence") if c in row.index and pd.notna(row[c]))


def _reason_words(reason: str) -> str:
    """A reason of the family (es.run_analogs: "simulated size 0.130 > 1.5 x 0.0167 (39 of 300 replications, worst of 12 rows: LEV6,
    common_drift_0.5, upper)") with the ids of the governing row in words, as the size cells give it."""
    return re.sub(r"(rows|replications): ([^()]*)\)$", lambda m: f"{m.group(1)}: " + ", ".join(_id_words(t) for t in m.group(2).split(", ")) + ")", reason)


def worst_case(sim: pd.DataFrame, test: str, col: str, laws=None):
    """(rate, 'law, scenario, persistence', n_sim) of the row of ``test`` with the highest rejection rate at the level ``col``
    (the worst-case size of the registration rule), or None when the simulation has no such row. A NaN rate raises: a size check
    cannot pass on a missing number (as in es.run_analogs)."""
    g = _registered_rows(sim, test, laws)
    if g.empty:
        return None
    rates = g[col].astype(float)
    if rates.isna().any():
        raise ValueError(f"the simulation row of '{test}' ({_where_words(g[rates.isna()].iloc[0]) or 'no label'}) has a NaN rejection "
                         f"rate in {col}: a size check cannot pass on a missing number")
    row = g.loc[rates.idxmax()]
    return float(row[col]), _where_words(row), int(row["n_sim"])


def registered_power(sim: pd.DataFrame, test: str, col: str):
    """(power, lower_bound_only) of a power test at the level ``col`` (es.run_analogs._power): the lowest over its rows, NaN when a
    row has no rate (a failed calibration: its `error` column says why); a row marked `power_lower_bound` is left out of that minimum,
    and a test whose rows are all such bounds has that bound. (None, False) when the simulation has no such row."""
    g = _registered_rows(sim, test)
    if g.empty:
        return None, False
    values = g[col].to_numpy(dtype=float)
    has_error = g["error"].notna().to_numpy() if "error" in g.columns else np.zeros(len(g), dtype=bool)
    if (np.isnan(values) & ~has_error).any():
        raise ValueError(f"a row of '{test}' has a NaN rejection rate in {col} and no error text: a power that failed must say why")
    lower = g["power_lower_bound"].fillna(False).astype(bool).to_numpy() if "power_lower_bound" in g.columns else np.zeros(len(g), dtype=bool)
    normal = values[~lower]
    if normal.size:
        return (float("nan") if np.isnan(normal).any() else float(normal.min())), False
    return float(values.min()), True


def lowest_power(sim: pd.DataFrame, test: str, col: str):
    """The power of ``test`` at the level ``col`` (registered_power), NaN when it failed, None when the simulation has no such row."""
    return registered_power(sim, test, col)[0]


def _worst_cell(found) -> str:
    return PENDING["dev"] if found is None else f"{FORMATS['dec3'](found[0])} ({found[1]})"


def _power_cell(power, lower_bound: bool = False) -> str:
    if power is None:
        return PENDING["dev"]
    return "--" if pd.isna(power) else ("$\\ge$ " if lower_bound else "") + FORMATS["dec3"](power)


def _ascii(text: str, where: str) -> str:
    if not text.isascii():
        raise ValueError(f"{where} has non-ASCII characters, which the tables cannot typeset under pdflatex's default encoding: {text!r}")
    return text


def _registration_inputs(src: Sources) -> tuple[dict, dict, dict]:
    """(inputs, family, rule record) of analogs/protocol_inputs.json, which must be the output of stage `sim` (Ruling 77)."""
    inputs = src.json("analogs/protocol_inputs.json")
    fam, rule = inputs.get("family"), inputs.get("rule")
    if inputs.get("stage") != "complete" or not isinstance(fam, dict):
        raise LookupError(f"analogs/protocol_inputs.json is at stage {inputs.get('stage')!r} and has no registered family (`family` is null: "
                          "the simulation stage has not run since the last analogues run): run `python -m es.run_analogs --stage sim`")
    if not isinstance(rule, dict) or not {"levels", "laws", "binomial_level"} <= rule.keys():
        raise LookupError(f"analogs/protocol_inputs.json has no complete rule record (it has {sorted(rule) if isinstance(rule, dict) else rule!r}; "
                          "levels, laws and binomial_level are needed): run `python -m es.run_analogs --stage sim`")
    missing = [h for h in ("H-E", *rule["laws"]) if not isinstance(fam.get(h), dict)]
    if missing:
        raise LookupError(f"analogs/protocol_inputs.json: the family has no entry for {missing} (it has {sorted(fam)})")
    return inputs, fam, rule


@builder("tab_registration", "analogs/protocol_inputs.json", "analogs/simulation.csv",
         note="the family and the rule record of protocol_inputs.json (stage complete) and the rows of simulation.csv that the rule reads: "
              "H-E's pooled size rows under the primary strata and the laws that the rule record names for H-B1 and H-B2; a second "
              "tabular gives the H-E power curve; the width of each tabular is checked against the text width")
def tab_registration(src: Sources) -> str:
    """The registered family (spec v3 section 6.2): per test, whether it is confirmatory or an estimand, the approximate flag and the
    reasons, as the registration rule recorded them in the family; the worst-case simulated size at the first Holm level and at the
    family-wise level with the row that governs it, and the power, from the rows of simulation.csv that the rule reads; and the
    H-E power curve, which the rule never reads."""
    inputs, fam, rule = _registration_inputs(src)
    first, whole = (float(x) for x in rule["levels"])
    holm_col, family_col = f"reject_{first:g}", f"reject_{whole:g}"
    holm_text = f"{whole:g}/{round(whole / first)}"                       # 0.05/3: the family-wise level over the three tests
    sim = src.frame("analogs/simulation.csv")
    _need(sim, ["test", "law", "scenario", "persistence", "strata", "n_sim", "rho", "mean_selc", holm_col, family_col], "simulation")
    plan = [("H-E", _rr("H-E (pooled A$\\cup$B)"), HE_SIZE_TEST, None, HE_POWER_TEST)]
    plan += [(hid, _rr(hid), HB_SIZE_TEST, laws, f"{hid} power") for hid, laws in rule["laws"].items()]
    colsep = REGISTRATION_COLSEP
    fixed = {j: round(cm * CM_PT, 1) for j, cm in enumerate(REGISTRATION_CM)}
    header = ["test", _stack("designated", "as", align="l"), _stack(f"size at {holm_text},", "worst case"),
              _stack(f"size at {whole:g},", "worst case"), _stack("power at", holm_text)]
    plain, reasons, n_sims, rows_read = [], {}, [], {}
    for hid, label, size_test, laws, power_test in plan:
        rows = _registered_rows(sim, size_test, laws)
        if len(rows) != fam[hid]["k_rows"]:
            raise ValueError(f"tab_registration: the simulation rows that the table reads for {hid} ({len(rows)}) are not the rows that the "
                             f"registration read (k_rows {fam[hid]['k_rows']}): the selection of the table and of the rule have drifted apart")
        rows_read[hid] = len(rows)
        small, large = worst_case(sim, size_test, holm_col, laws), worst_case(sim, size_test, family_col, laws)
        power, lower_bound = registered_power(sim, power_test, holm_col)
        if hid != "H-E":
            recorded = fam[hid].get("power")
            if not ((pd.isna(power) and pd.isna(recorded)) or (pd.notna(power) and pd.notna(recorded) and abs(power - recorded) < 1e-9)):
                raise ValueError(f"tab_registration: the power of {hid} in simulation.csv ({power}) is not the one of the family ({recorded}): "
                                 "the two files are not from the same run")
        n_sims += [c[2] for c in (small, large) if c is not None]
        decision = ("confirmatory" if fam[hid]["confirmatory"] else "estimand") + (MARK_A if fam[hid].get("approximate") else "")
        plain.append([label, _rr(decision), _rr(_worst_cell(small)), _rr(_worst_cell(large)),
                      _power_cell(power, lower_bound or bool(fam[hid].get("power_lower_bound"))) + (MARK_B if hid == "H-E" else "")])
        if fam[hid]["reasons"]:
            reasons[hid] = _ascii(_esc("; ".join(_reason_words(r) for r in fam[hid]["reasons"])), f"the reasons of {hid}")
    widths = _fit_widths(header, plain, fixed)
    inner = sum(widths) + 2 * colsep * (len(widths) - 1)                  # the width of a cell that spans all the columns
    rows_a, starts = [], set()
    for (hid, *_), row in zip(plan, plain):
        starts.add(len(rows_a))
        rows_a.append(row)
        if hid in reasons:
            rows_a.append([f"\\multicolumn{{{len(widths)}}}{{p{{{inner:.1f}pt}}}}{{{RAGGED}\\textit{{Reasons for {hid}:}} {reasons[hid]}}}"])
    colspec = "".join(f"p{{{w:.1f}pt}}" if j in fixed else "c" for j, w in enumerate(widths))
    table_a = _sized(tabular(colspec, header, rows_a, midrules_before=starts - {0}), colsep=colsep, size=None)

    # H-E's power curve, which the registration never reads, and the target row that is its power in the table above
    curve = _registered_rows(sim, HE_CURVE_TEST).sort_values("rho")
    target = _registered_rows(sim, HE_POWER_TEST)
    if curve.empty or len(target) != 1:
        raise LookupError(f"simulation.csv needs the rows '{HE_CURVE_TEST}' (the grid) and one row '{HE_POWER_TEST}' (the target); it has "
                          f"{len(curve)} and {len(target)}")
    t = target.iloc[0]
    if inputs.get("rho_star") is not None and abs(float(inputs["rho_star"]) - float(t["rho"])) > 1e-9:
        raise ValueError(f"tab_registration: rho* of protocol_inputs.json ({inputs['rho_star']}) is not the coupling of the target row ({t['rho']})")
    curve_rows = [[FORMATS["dec2"](r["rho"]), FORMATS["pct2"](r["mean_selc"]), FORMATS["dec3"](r[holm_col]), FORMATS["dec3"](r[family_col])]
                  for _, r in curve.iterrows()]
    curve_rows.append([FORMATS["dec2"](t["rho"]) + MARK_B, FORMATS["pct2"](t["mean_selc"]), FORMATS["dec3"](t[holm_col]), FORMATS["dec3"](t[family_col])])
    header_b = ["$\\rho$", _stack("mean", "SELC"), _stack("rate at", holm_text), _stack("rate at", f"{whole:g}")]
    n_curve = sorted({int(n) for n in curve["n_sim"]})
    title = f"H-E: simulated power by coupling $\\rho$ ({_range(n_curve[0], n_curve[-1], FORMATS['int'])} simulations per row)"
    table_b = (f"\\par\\noindent\\textit{{{title}}}\\par\\smallskip\n"
               + _sized(_tab("cccc", header_b, curve_rows, midrules_before={len(curve_rows) - 1}), colsep=colsep, size=None))
    _check_text_width("tab_registration", [(header, plain, widths), (header_b, curve_rows, _fit_widths(header_b, curve_rows, {}))], colsep)
    _check_text_width("tab_registration:reasons", [(["reasons"], [[r] for r in reasons.values()], [inner])], colsep)

    notes = []
    approx = [hid for hid, *_ in plan if fam[hid].get("approximate")]
    interim = bool(fam["H-E"].get("interim_approximate"))
    if approx or interim:
        factor = float(rule.get("approximate_factor", 1.0))
        above = "above nominal" if factor == 1.0 else f"above {fmt_g(factor)} times nominal"
        text = (f"Labelled approximate: the simulated size is significantly above nominal at some row that the test reads (a rate {above} and a "
                f"one-sided binomial test of the rejection count against nominal that rejects at {fmt_g(rule['binomial_level'])} divided by the "
                "number of rows).")
        if interim:
            text += " Test A's interim p-value, which rests on the single-quarter test of H-E, is labelled approximate."
        notes.append(f"{MARK_A} {text}")
    # Ruling 118, M1: the confirmatory family is {H-E}, decided at the family-wise level; the first Holm level is the draft family's
    notes.append(f"The level {holm_text} is the first Holm level of the three-member draft family. The confirmatory family is H-E alone, decided at "
                 f"{whole:g}, so the size at {whole:g} governs H-E.")
    reading = {"below": "The target lies below every grid point, so this is the power of the weakest coupling, not of the target.",
               "bracketed": "No grid point reaches the target, so this is the power of the strongest coupling."}
    note = str(t["note"]) if "note" in t.index and pd.notna(t["note"]) else ""
    why = next((v for k, v in reading.items() if k in note), _esc(note))
    target_selc = FORMATS["pct2"](t["target_selc"]) if "target_selc" in t.index and pd.notna(t["target_selc"]) else "--"
    notes.append(f"{MARK_B} H-E's power is that of the study's target row: {int(t['n_sim'])} simulations at the coupling {FORMATS['dec2'](t['rho'])}, "
                 f"the grid point whose mean SELC is nearest the effect target of {target_selc}. {why} H-E stays in the family whatever its power.")
    read = "; ".join(f"{hid}: {rows_read[hid]} rows" + (" from the simulated score laws of " + " and ".join(PAPER_BLOCK_NAMES.get(law, _esc(law)) for law in laws) if laws else " of the pooled test")
                     for hid, _, _, laws, _ in plan)
    notes.append(f"Each size is the worst case over the rows that the designation rule reads, under the primary strata ({read}); each rate is "
                 f"the rejection share of {_range(min(n_sims), max(n_sims), FORMATS['int'])} simulated replications.")
    decile = sim[(sim["test"] == HE_SIZE_TEST) & (sim["strata"] == "decile")]
    if not decile.empty:
        notes.append("Under the decile strata, the sensitivity (reported only; the designated status rests on the primary strata), the "
                     f"worst-case size of H-E is {FORMATS['dec3'](decile[holm_col].max())} at {holm_text} and "
                     f"{FORMATS['dec3'](decile[family_col].max())} at {whole:g}.")
    return table_a + "\\medskip\n" + table_b + "".join(f"\\footnotetext{{{n}}}\n" for n in notes)


# ---------------------------------------------------------------- section 5: economics
def _econ(src: Sources) -> pd.DataFrame:
    e = src.frame("economics/summary.csv")
    _need(e, ["analysis", "policy", "comparator", "position", "fees", "SELC", "SNET", "NET", "ESred", "CAP", "n_days"], "economics/summary.csv")
    return e


def _policy_legend(codes) -> str:
    """'P1: fixed, held-out; P4: rolling (56 d); ...': the code of each policy and its name from POLICY_NAME."""
    parts = []
    for code in codes:
        name = POLICY_NAME[code]
        parts.append(f"{code}: " + (name[len(code):].strip() if name.startswith(code + " ") else name))
    return "Cutoff policies: " + "; ".join(parts) + "."


@builder("tab_econ_primary", "economics/summary.csv",
         note="set in a group with a smaller font and a tighter column separation; the policy column shows the codes P1, P4, P5, P6, "
              "P4@r*, and a footnote gives their names")
def tab_econ_primary(src: Sources) -> str:
    e = _econ(src)
    opt = [c for c in ("LA", "PG") if c in e.columns]
    rows, starts = [], set()
    for aid in ECON_IDS:
        starts.add(len(rows))
        for pol in ("P1", "P4", "P5", "P6", "P4@r*"):
            g = e[(e.analysis == aid) & (e.policy == pol) & (e.position == "premium") & (e.fees == "calibrated") & (e.comparator == "none")]
            if g.empty:
                continue
            r = _one(g, f"econ[{aid},{pol}]")
            rows.append([_short(aid), pol] + [FORMATS["usd0"](r[c]) for c in opt]
                        + [FORMATS["usd0"](r.NET), FORMATS["pct1"](r.SELC), FORMATS["usd2"](r.SNET), FORMATS["usd0"](r.ESred), FORMATS["pct1"](r.CAP), FORMATS["int"](r.n_days)])
    table = _sized(tabular("ll" + "r" * (len(opt) + 6),
                           ["analysis", _stack("cutoff", "policy", align="l")] + opt + ["NET", "SELC", _stack("SNET", "(\\$/MW-day)"), "ESred", "CAP", "days"],
                           rows, midrules_before=starts - {0}))
    return table + f"\\footnotetext{{{_policy_legend(('P1', 'P4', 'P5', 'P6', 'P4@r*'))}}}\n"


#: What tab_econ_ci prints for each estimand: (short estimand label, comparison). The labels are local to the table; the registry's
#: STAT_LABEL is not changed and is printed in full in the notes under the table. The E4 block (Ruling 81) is the SELC on decile
#: strata, the SELC of each comparator at the matched alert count, and the paired difference P1 - comparator; Ruling 118 adds the
#: same for the tail measure (the comparator's own ESred and the paired difference in ESred), so the two paired differences carry
#: the measure in their label.
CI_LABEL = {"SELC": ("SELC", ""), "NET": ("NET", ""), "E2:SNET": ("SNET (\\$/MW-day)", ""), "E3:ESred": ("ESred", ""),
            "E4:SELC_decile": ("SELC, decile strata", ""),
            "E4:SELC|calendar@count": ("SELC at matched count", "calendar-only"),
            "E4:SELC|VOL@count": ("SELC at matched count", "VOL"),
            "E4:dSELC|calendar@count": ("paired difference, SELC", "P1 \\textminus{} calendar-only"),
            "E4:dSELC|VOL@count": ("paired difference, SELC", "P1 \\textminus{} VOL"),
            "E4:ESred|calendar@count": ("ESred at matched count", "calendar-only"),
            "E4:ESred|VOL@count": ("ESred at matched count", "VOL"),
            "E4:dESred|calendar@count": ("paired difference, ESred", "P1 \\textminus{} calendar-only"),
            "E4:dESred|VOL@count": ("paired difference, ESred", "P1 \\textminus{} VOL")}
#: The p-columns of tab_econ_ci in centimetres (analysis, estimand, estimate and interval, comparison), and the column separation of its group.
ECON_CI_CM = (2.1, 3.4, 4.1, 2.4)
ECON_CI_COLSEP = 2.5


@builder("tab_econ_ci", "economics/bootstrap_ci.csv",
         note="a longtable in long format (one row per analysis and estimand) with its own caption and label (tab:econ-ci): include it "
              "bare, outside a float; the layout with the analyses in rows and the estimands in columns is taller than a sideways "
              "page holds; its width is checked against the text width")
def tab_econ_ci(src: Sources) -> str:
    """The estimands E1-E4 with their intervals, by analysis (each full analysis followed by its ex-CAISO twin). A longtable
    with a row per analysis and estimand: its width does not depend on the number of estimands or analyses, and the labels
    that a column heading would carry are cells that wrap. Every cell is read from the rows of its own analysis, the twin's
    from the twin's."""
    c = src.frame("economics/bootstrap_ci.csv")
    _need(c, ["analysis", "stat", "estimate", "lo", "hi", "approximate"], "bootstrap_ci")
    missing = [s for s in STAT_ORDER if s not in CI_LABEL]
    if missing:
        raise LookupError(f"tab_econ_ci has no short label for the registry stat(s) {missing}: add them to CI_LABEL")
    rows = []
    for i, aid in enumerate(_econ_order(c.analysis.unique())):
        g = c[c.analysis == aid]
        for j, stat in enumerate(STAT_ORDER):
            label, comparator = CI_LABEL[stat]
            first = _rr(_econ_label(aid)) if j == 0 else ""
            if j == 0 and i > 0 and "|" not in str(aid):
                first = "\\midrule" + first                                  # a rule before each full analysis, never before its twin
            rows.append([first, _rr(label), _rr(stat_cell(g, stat)), comparator])
    header = ["analysis", "estimand", "estimate [interval]", "comparison"]
    widths = [round(cm * CM_PT, 1) for cm in ECON_CI_CM]
    _check_text_width("tab_econ_ci", [(header, rows, widths)], ECON_CI_COLSEP)
    level = fmt_g(_code("ciLevel"), 3)
    caption = (f"Economic estimands with {level}\\% bootstrap intervals, by analysis; each analysis is followed by the same "
               "analysis without the CAISO nodes.")
    notes = _note_par("Estimands as the text numbers them: " + "; ".join(STAT_LABEL[s] for s in STAT_ORDER) + ".")
    if any(APPROX_MARK in r[2] for r in rows):
        notes += _note_par(f"{APPROX_MARK} {APPROX_TEXT}")
    table = longtable(_p_colspec(widths), header, rows, caption, "tab:econ-ci")
    return _breakable(table + notes, small=True, colsep=ECON_CI_COLSEP)


#: The parts of the difference between the fixed cutoff and the rolling one (E1), the rows of each analysis in tab_decomposition.
DECOMPOSITION_PARTS = ("timing", "level", "total")
#: The measures of tab_decomposition in its two panels: (the measure's key in the E1 rows, its heading). A panel with all three
#: would be wider than the text (the intervals are up to 120 pt wide), so J_0.5 has a panel of its own.
DECOMPOSITION_PANELS = ((("NET", "NET"), ("ESred", "ESred")), (("J_0.5", "$J_{0.5}$"),))
DECOMPOSITION_ANALYSIS_CM = 2.2            # the column of the analysis; the other columns are as wide as their cells
DECOMPOSITION_COLSEP = 2
DECOMPOSITION_COMPONENTS = ("Components of the difference in the value V that the column measures: timing = V(P1) $-$ V(P4@r*), "
                            "level = V(P4@r*) $-$ V(P4), total = V(P1) $-$ V(P4).")


def _e1_cell(frame: pd.DataFrame, key: str, part: str) -> str:
    """'estimate [lo, hi]' (dollars) of the E1 row of a measure and a part of the difference; '--' where the frame has none."""
    g = frame[frame.stat == f"E1:{key}:{part}"]
    return "--" if g.empty else ci_cell("usd0", "ciusd0", _one(g, key).estimate, _one(g, key).lo, _one(g, key).hi, _one(g, key).approximate)


def decomposition_panels(src: Sources) -> list[tuple[str, list[float], list[str], list[list[str]]]]:
    """(title, widths, header, rows) of each panel of tab_decomposition: a row per analysis and part (timing, level, total), a column
    per measure, each full analysis followed by its ex-CAISO twin, every cell read from the rows of its own analysis. ``widths`` are
    those of its p-columns in points: the analysis column has a fixed one, the others are as wide as their widest cell."""
    c = src.frame("economics/bootstrap_ci.csv")
    _need(c, ["analysis", "stat", "estimate", "lo", "hi", "approximate"], "bootstrap_ci")
    ids = _econ_order(c.analysis.unique())
    panels = []
    for number, measures in enumerate(DECOMPOSITION_PANELS):
        rows = []
        for i, aid in enumerate(ids):
            g = c[c.analysis == aid]
            for j, part in enumerate(DECOMPOSITION_PARTS):
                first = _rr(_econ_label(aid)) if j == 0 else ""
                if j == 0 and i > 0 and "|" not in str(aid):
                    first = "\\midrule" + first                              # a rule before each full analysis, never before its twin
                rows.append([first, part] + [_e1_cell(g, key, part) for key, _ in measures])
        header = ["analysis", "component"] + [name for _, name in measures]
        panels.append((f"Panel {chr(ord('A') + number)}: " + " and ".join(name for _, name in measures),
                       _fit_widths(header, rows, {0: round(DECOMPOSITION_ANALYSIS_CM * CM_PT, 1)}), header, rows))
    return panels


def _panel_longtable(colspec: str, header: list[str], rows: list[list[str]], title: str, caption: str | None = None, label: str | None = None) -> str:
    """One panel of a table as a longtable: a heading row with the title, then the column headings (both repeat on every page).
    Only the first panel has the caption and label (paper_tex.longtable always has them)."""
    head = " & ".join(header) + " \\\\"
    title_row = f"\\multicolumn{{{len(header)}}}{{l}}{{\\textit{{{title}}}}} \\\\"
    lines = ["\\begin{longtable}{" + colspec + "}"]
    if caption is not None:
        lines.append("\\caption{" + caption + "}\\label{" + label + "} \\\\")
    lines += ["\\toprule", title_row, "\\midrule", head, "\\midrule", "\\endfirsthead", "\\toprule", title_row, "\\midrule", head, "\\midrule", "\\endhead",
              "\\bottomrule", "\\endfoot"]
    lines += [" & ".join(r) + " \\\\" for r in rows]
    lines.append("\\end{longtable}")
    return "\n".join(lines) + "\n"


@builder("tab_decomposition", "economics/bootstrap_ci.csv",
         note="two longtables (panel A: NET and ESred; panel B: J_0.5) with one caption and label (tab:decomposition): include it bare, "
              "outside a float; a row per analysis and part, a column per measure; the width of each panel is checked against the "
              "text width")
def tab_decomposition(src: Sources) -> str:
    """Timing, level and total of the difference between the fixed and the rolling cutoff (E1), by analysis, in a long format that
    fits the text width (Ruling 95)."""
    panels = decomposition_panels(src)
    _check_text_width("tab_decomposition", [(header, rows, widths) for _, widths, header, rows in panels], DECOMPOSITION_COLSEP)
    level = fmt_g(_code("ciLevel"), 3)
    caption = (f"Difference between the fixed cutoff (P1) and the rolling cutoff (P4), split into timing and level, with {level}\\% "
               "bootstrap intervals, by analysis; each analysis is followed by the same analysis without the CAISO nodes.")
    body = "".join(_panel_longtable(_p_colspec(widths), header, rows, title, *((caption, "tab:decomposition") if number == 0 else (None, None)))
                   for number, (title, widths, header, rows) in enumerate(panels))
    notes = _note_par(DECOMPOSITION_COMPONENTS)
    if any(MARK_A in cell for _, _, _, rows in panels for row in rows for cell in row):
        notes += _note_par(f"{MARK_A} {APPROX_TEXT}")
    return _breakable(body + notes, small=True, colsep=DECOMPOSITION_COLSEP)


BH_FAMILIES = ("audit robustness", "economic robustness", "drift monitors")
#: The note of a descriptive family (Ruling 118, M5; es.paper_registry.DESCRIPTIVE_BH_FAMILIES): the audit-robustness family re-runs the
#: portability contrasts of H-B1 and H-B2, which the registration rule made estimands because their bootstrap is oversized in the size
#: study. Its p-values and BH flags are therefore diagnostics. The tables count its tests and print nothing of its rejections.
DESCRIPTIVE_NOTE = ("The audit-robustness family is descriptive, as the portability contrasts it re-runs are estimands: its $p$-values come "
                    "from the calibration-and-deployment bootstrap whose simulated size exceeds nominal, so its Benjamini--Hochberg flags "
                    "are diagnostics and no count of its rejections is reported.")
DIAGNOSTIC_NOTE = (MARK_B + " Diagnostic only. " + DESCRIPTIVE_NOTE + " The flag of a test of that family is shown for completeness.")


def _bh(src: Sources) -> pd.DataFrame:
    """inference/bh.csv. Its `included` column (es.secondary.bh_table) is False where the p-value is NaN, a test that did not
    run; the number of tests of a family is the sum of `included`, not the number of rows. A file without the column (the
    earlier contract) has no such rows."""
    b = src.frame("inference/bh.csv")
    _need(b, ["family", "test", "estimate", "p", "bh_reject"], "inference/bh.csv")
    return b if "included" in b.columns else b.assign(included=True)


def _not_computed(family: str) -> str:
    return f"{family}: not computed (inference/bh.csv has no row of this family: the analysis that writes it has not run)."


@builder("tab_bh_families", "inference/bh.csv")
def tab_bh_families(src: Sources) -> str:
    """Benjamini-Hochberg within the three exploratory families (spec 5.2; Ruling 37). A family that bh.csv does not hold was
    not computed: its row is one pending sentinel over the three columns and a note says so (never 0 tests). A family that is
    there with no test that ran (every p-value NaN) has 0 tests. A descriptive family (Ruling 118, M5) shows the number of its tests
    and dashes for its rejections and its smallest p-value, with a note that says why."""
    b = _bh(src)
    rows, notes = [], []
    for fam in BH_FAMILIES:
        present = b[b.family == fam]
        if present.empty:
            rows.append([tex_escape(fam), f"\\multicolumn{{3}}{{c}}{{{PENDING['dev']}}}"])      # one sentinel over the three columns (its text is wider than one)
            notes.append(_not_computed(fam))
            continue
        g = present[present.included.astype(bool)]
        if fam in DESCRIPTIVE_BH_FAMILIES:
            rows.append([tex_escape(fam), FORMATS["int"](len(g)), "--", "--"])
            notes.append(DESCRIPTIVE_NOTE)
            continue
        rows.append([tex_escape(fam), FORMATS["int"](len(g)), FORMATS["int"](int(g.bh_reject.astype(bool).sum())),
                     "--" if g.empty else FORMATS["pval"](g.p.min())])
    q = fmt_g(_code("bhLevel"), 3)
    return tabular("lrrr", ["exploratory family", "tests", f"BH rejections ($q$ = {q})", "smallest $p$"], rows) \
        + "".join(f"\\footnotetext{{{n}}}\n" for n in notes)


@builder("tab_label_control_pr", "labels_pr/label_control_pr.csv")
def tab_label_control_pr(src: Sources) -> str:
    """Precision and recall of the label-control cells (spec 5.4 'item 3 also applies here'; Ruling 43). The file has
    no estimator column (xgb only)."""
    p = src.frame("labels_pr/label_control_pr.csv").assign(estimator="xgb")
    _need(p, ["analysis", "block", *PR_COLS], "label_control_pr")
    rows = []
    for a in LABEL_CONTROL_IDS:
        for b in LABEL_CONTROL_BLOCKS:
            g = p[(p.analysis == a) & (p.block == b)]
            if g.empty:
                continue
            rows.append([_rr(ANALYSIS_NAME[a]), PAPER_BLOCK_NAMES[b]] + [D2(g[c].median()) for c in PR_COLS])
    return _tab("p{2.7cm}lrrrrr", [_stack("calibration window,", "label rule", align="l"), "block"] + PR_HEADS, rows,
                groups=[("", 1), ("", 1)] + PR_GROUPS)


@builder("tab_comparators", "economics/summary.csv")
def tab_comparators(src: Sources) -> str:
    e = _econ(src)
    names = {"P1": "alerts (P1)", "calendar@count": "calendar-only model at matched count", "VOL@count": "VOL at matched count",
             "VOL_P1": "VOL through P1", "VOL_P4": "VOL through P4", "VOL_P6": "VOL through P6"}
    rows, starts = [], set()
    for aid in ECON_IDS:
        starts.add(len(rows))
        for key, label in names.items():
            g = e[(e.analysis == aid) & (e.position == "premium") & (e.fees == "calibrated") & ((e.policy == key) if key == "P1" else (e.comparator == key))]
            if g.empty:
                continue
            r = _one(g, key)
            rows.append([ANALYSIS_NAME[aid], _rr(label), FORMATS["pct1"](r.SELC), FORMATS["usd0"](r.NET), FORMATS["usd0"](r.ESred),
                         FORMATS["pct1"](r.CAP)])
    return _sized(tabular("lp{3.4cm}rrrr", ["analysis", "selection", "SELC", "NET", "ESred", "CAP"], rows, midrules_before=starts - {0}), colsep=6)


def _sign(v) -> str:
    if pd.isna(v):
        raise ValueError("the sign of a missing value")
    return "+" if v > 0 else MINUS if v < 0 else "0"


POSITION_HEAD = {"premium": "premium", "inc": "INC", "dec": "DEC", "premium[short_spikes]": _stack("premium,", "short spikes"),
                 "premium[long_spikes]": _stack("premium,", "long spikes")}


@builder("tab_fee_position_signs", "economics/summary.csv", "analogs/activity_all.csv")
def tab_fee_position_signs(src: Sources) -> str:
    """Sign consistency (spec 5.2): panel 1 over fees and positions from the economics rows; panel 2 over blocks, the
    sign of each block's node-median |log10 R| contrast against plain DART lags (Ruling 47, item 21)."""
    e = _econ(src)
    rows, starts = [], set()
    for aid in ("AQ3", "AQ4", "AQ1", "AQ2"):
        if aid not in set(e.analysis):
            continue
        starts.add(len(rows))
        for fee in _fee_order():
            cells = []
            for pos in POSITION_ORDER:
                g = e[(e.analysis == aid) & (e.policy == "P1") & (e.position == pos) & (e.fees.astype(str) == fee) & (e.comparator == "none")]
                cells.append("--" if g.empty else f"{_sign(_one(g, pos).NET)}/{_sign(_one(g, pos).SELC)}")
            rows.append([ANALYSIS_NAME[aid], tex_escape(fee)] + cells)
    panel1 = tabular("ll" + "c" * len(POSITION_ORDER), ["analysis", "fees"] + [POSITION_HEAD[p] for p in POSITION_ORDER], rows, midrules_before=starts - {0})
    a = src.frame("analogs/activity_all.csv")
    _need(a, ["analysis", "block", "estimator", "policy", "log_error"], "activity_all")
    a = a[(a.policy == "P1") & (a.estimator == "xgb")]
    analogs = [x for x in ("AQ3", "AQ4", "AQ1", "AQ2") if x in set(a.analysis)]
    block_rows = []
    for b in [x for x in AUDIT_BLOCKS if x != "DART3"]:
        cells = []
        for aid in analogs:
            g = a[(a.analysis == aid) & (a.block == b)].log_error
            base = a[(a.analysis == aid) & (a.block == "DART3")].log_error
            cells.append("--" if g.empty or base.empty else _sign(float(g.median() - base.median())))
        block_rows.append([PAPER_BLOCK_NAMES[b]] + cells)
    panel2 = tabular("l" + "c" * len(analogs), [_stack("block vs", "plain DART lags", align="l")] + analogs, block_rows)
    return panel1 + "\\medskip\n" + panel2 + \
        "\\footnotetext{Panel 1: sign(NET)/sign(SELC) of the P1 alerts for each fee scenario and position. Panel 2: the sign of each block's node-median $|\\log_{10} R|$ minus that of plain DART lags (+: the block drifts more). A sign that is the same in every fee row of an analysis does not depend on the fee scenario.}\n"


@builder("tab_alpha_curve", "economics/summary.csv")
def tab_alpha_curve(src: Sources) -> str:
    e = _econ(src)
    rows = []
    for aid in ("AQ3", "AQ4"):
        for al in ALPHAS:
            cells = []
            for pol in ("P1", "P6"):
                g = e[(e.analysis == aid) & (e.comparator == f"curve_{pol}_a{al}") & (e.position == "premium") & (e.fees == "calibrated")]
                cells += ["--", "--"] if g.empty else [FORMATS["usd0"](_one(g, pol).NET), FORMATS["usd0"](_one(g, pol).ESred)]
            rows.append([ANALYSIS_NAME[aid], f"{100 * al:g}\\%"] + cells)
    return tabular("llrrrr", ["analysis", "$\\alpha$", "P1 NET", "P1 ESred", "P6 NET", "P6 ESred"], rows, midrules_before={len(ALPHAS)})


@builder("tab_volmanagement", "economics/volmanagement_check.csv")
def tab_volmanagement(src: Sources) -> str:
    v = src.frame("economics/volmanagement_check.csv")
    _need(v, ["analysis", "sharpe_alerted", "sharpe_not_alerted"], "volmanagement_check")
    rows = [[ANALYSIS_NAME.get(r.analysis, tex_escape(r.analysis)), FORMATS["dec3"](r.sharpe_alerted), FORMATS["dec3"](r.sharpe_not_alerted)] for r in v.itertuples()]
    return tabular("lrr", ["analysis", _stack("mean/sd of hourly P\\&L,", "alerted hours"), "other hours"], rows)


@builder("tab_rotation_dev", "economics/rotation.csv", note="retained share printed as min / mean over the windows' alert paths, pct0")
def tab_rotation_dev(src: Sources) -> str:
    """Rotation test of timing on the development windows. ``t_obs`` is the full-grid studentized statistic T (unitless,
    Ruling 55) and ``selc_obs`` the SELC estimate: two columns, never one for the other. ``n_null`` is the product of the
    windows' days minus one (Ruling 69); ``retained_min`` and ``retained_mean`` are NaN for a window without alerts."""
    r = src.frame("economics/rotation.csv")
    _need(r, ["window", "t_obs", "p", "n_null", "selc_obs", "retained_min", "retained_mean"], "rotation")

    def retained(x) -> str:
        if pd.isna(x.retained_min) or pd.isna(x.retained_mean):
            return "--"
        return f"{FORMATS['pct0'](x.retained_min)} / {FORMATS['pct0'](x.retained_mean)}"
    rows = [[_rr(ANALYSIS_NAME.get(str(x.window), tex_escape(x.window))), FORMATS["pct1"](x.selc_obs), D2(x.t_obs), FORMATS["pval"](x.p),
             FORMATS["int"](x.n_null), retained(x)] for x in r.itertuples()]
    return tabular("p{3.0cm}rrrrr", ["window", "SELC", _stack("$T$", "(studentized)"), _stack("rotation $p$", "(exploratory)"), "rotations",
                                     _stack("retained share", "(min / mean)")], rows)


@builder("tab_looo", "inference/looo.csv")
def tab_looo(src: Sources) -> str:
    l = src.frame("inference/looo.csv")
    _need(l, ["dropped", "analysis", "block", "failures", "median_log_error"], "looo")
    keep = _sort(l[l.block.isin(["LEV6", "DART3", "ROLL_DISP9", "DART18"])], ("analysis", tuple(LEGACY) + tuple(ANALOGS)),
                 ("block", tuple(BLOCKS)), ("dropped", MARKETS))
    rows = [[_aid(r.analysis), PAPER_BLOCK_NAMES.get(r.block, tex_escape(r.block)), MARKET_NAME.get(r.dropped, tex_escape(r.dropped)),
             FORMATS["int"](r.failures), D2(r.median_log_error)] for r in keep.itertuples()]
    return longtable("llllr", ["analysis", "block", "operator dropped", "failures", "median $|\\log_{10} R|$"], rows, "Leave-one-operator-out checks", "tab:looo")


#: Longest line (characters) of a verbatim block at the two sizes: a typewriter character is 0.525 em, and the text width of
#: the class is 372 pt, so \small (9 pt) holds 78 and \footnotesize (7 pt) holds 101.
VERBATIM_MAX = (("small", 78), ("footnotesize", 101))


@builder("tab_mixed", "inference/mixed_model.txt")
def tab_mixed(src: Sources) -> str:
    text = src.path("inference/mixed_model.txt").read_text().rstrip()
    if not text.isascii():
        raise ValueError("inference/mixed_model.txt has non-ASCII characters, which verbatim cannot typeset under pdflatex's "
                         "default encoding: write the model summary in ASCII")
    longest = max((len(line) for line in text.splitlines()), default=0)
    for size, limit in VERBATIM_MAX:
        if longest <= limit:
            return f"\\begin{{{size}}}\\begin{{verbatim}}\n{text}\n\\end{{verbatim}}\\end{{{size}}}\n"
    raise ValueError(f"inference/mixed_model.txt has a line of {longest} characters, more than the {VERBATIM_MAX[-1][1]} that fit the "
                     "372 pt text width at 7 pt: reformat the model summary")


# ---------------------------------------------------------------- ESM ledgers (Online Resource 1)
#: The provenance note is its own part of ESM_1.pdf (esm_provenance), printed before the ledgers: it does not depend on the
#: ledger that happens to come first. ESM_1.tex includes it with \gentable{esm_provenance}.
ESM_PROVENANCE = ("\\noindent\\textbf{Provenance of the numbers.} Every number in the paper is traceable to its source row. "
                  "The build of the manuscript writes three files next to it. \\texttt{generated/registry.csv} lists every number that the text "
                  "may use, with the saved output, the row selector, the column and the format that it is read with. "
                  "\\texttt{generated/provenance.csv} gives, for each of them, the value that was printed, its status and a hash of the "
                  "source file. \\texttt{generated/BUILD\\_INFO.json} records when the build ran, which sealed tests had results and how many "
                  "numbers were still pending. The tables of the paper are built by the same step from the same saved outputs.\n\n")


@builder("esm_provenance", note="the provenance note of Online Resource 1: include it before the ledgers (\\gentable{esm_provenance})")
def esm_provenance(src: Sources) -> str:
    return ESM_PROVENANCE


#: The key columns of both legacy ledger tables: analysis and block in one cell, and the setting of the cell (both wrap).
LEGACY_KEYS = "p{2.3cm}p{2.4cm}"


def _setting(r: dict) -> str:
    """The estimator, seed, policy and cutoff target of one legacy cell, in one column."""
    return f"{r['estimator']}, s{int(r['seed'])}, {tex_escape(r['policy'])}, {100 * r['alpha']:g}\\%"


@builder("esm_ledger_legacy", "legacy/summary.csv", note="two longtables with the same rows: the first has the primary band, the second the other bands, "
                                                          "the failures by operator, the nodes, the lift, the prevalence and the zero-activity count")
def esm_ledger_legacy(src: Sources) -> str:
    """Every cell of legacy/summary.csv, in two longtables with the same rows (a row has more numbers than a page is wide)."""
    s = _summary_ledger(src)
    other_bands = [f"failures_band_{k}" for k in BANDS if k != PRIMARY_BAND]
    operators = [f"failures_{m}" for m in MARKETS]
    records = s.to_dict("records")                                 # not itertuples: the band columns are not identifiers
    key = lambda r: [_rr(_aid(r["analysis"]) + ", " + tex_escape(r["block"])), _rr(_setting(r))]
    first = [key(r) + [FORMATS["int"](r["failures"]), _num(D2, r["median_R"]), _num(D2, r["median_log_error"]), _num(D2, r["median_auprc"]),
                       _num(D2, r["auprc_lift"]), _num(FORMATS["dec3"], r["prevalence"]), FORMATS["int"](r["n_zero_activity"])] for r in records]
    second = [key(r) + [FORMATS["int"](r[c]) for c in other_bands] + [FORMATS["int"](r[c]) for c in operators] + [FORMATS["int"](r["n_nodes"])]
              for r in records]
    keys = [_stack("analysis, block", align="l"), _stack("estimator,", "seed, policy,", "$\\alpha$", align="l")]
    head_one = keys + ["failures", _stack("median", "$R$"), _stack("median", "$|\\log_{10}R|$"), _stack("median", "AUPRC"),
                       _stack("AUPRC", "lift"), "prevalence", _stack("zero", "alerts")]
    head_two = keys + [_stack("failures,", band_text(BANDS[k.removeprefix("failures_band_")]).replace(", ", ",")) for k in other_bands] \
        + [_stack("failures", MARKET_NAME[m]) for m in MARKETS] + ["nodes"]
    one = longtable(LEGACY_KEYS + "rrrrrrr", head_one, first, "Every regenerated audit cell of the published analyses (1 of 2): "
                    "failures at the primary band, median $R$ and $|\\log_{10}R|$, AUPRC and its lift, prevalence, nodes without an alert",
                    "tab:esm-legacy")
    two = longtable(LEGACY_KEYS + "r" * (len(other_bands) + len(operators) + 1), head_two, second,
                    "Every regenerated audit cell of the published analyses (2 of 2): the same rows, with the failures at the other bands "
                    "and by operator, and the number of nodes", "tab:esm-legacy-more")
    return _breakable(one + "\\clearpage\n" + two, small=True, colsep=2)


def _summary_ledger(src: Sources) -> pd.DataFrame:
    """legacy/summary.csv in the order of the ledger, with every column that the ledger prints."""
    s = _summary(src)
    _need(s, [f"failures_band_{k}" for k in BANDS if k != PRIMARY_BAND] + [f"failures_{m}" for m in MARKETS]
          + ["n_nodes", "auprc_lift", "prevalence", "n_zero_activity"], "legacy/summary.csv")
    return _sort(s, ("analysis", tuple(LEGACY) + LABEL_CONTROL_IDS), ("block", tuple(BLOCKS)), ("estimator", ("xgb", "logit")),
                 "seed", "policy", "alpha")


@builder("esm_ledger_analogs", "analogs/activity_all.csv")
def esm_ledger_analogs(src: Sources) -> str:
    a = src.frame("analogs/activity_all.csv")
    _need(a, ["analysis", "block", "estimator", "policy", "node", "log_error", "fail_0.5_2"], "activity_all")
    g = a.groupby(["analysis", "block", "estimator", "policy"]).agg(failures=("fail_0.5_2", "sum"), n=("node", "size"), mle=("log_error", "median")).reset_index()
    g = _sort(g, ("analysis", tuple(ANALOGS)), ("block", tuple(BLOCKS)), ("estimator", ("xgb", "logit")), ("policy", _analog_policy_order()))
    rows = [[tex_escape(r.analysis), tex_escape(r.block), r.estimator, tex_escape(r.policy), FORMATS["int"](r.failures), FORMATS["int"](r.n), _num(D2, r.mle)] for r in g.itertuples()]
    return _breakable(longtable("llllrrr", ["analogue", "block", "estimator", "policy", "failures", "nodes", _stack("median", "$|\\log_{10}R|$")], rows,
                                "Every analogue cell by block, estimator and cutoff policy", "tab:esm-analogs"), small=True)


@builder("esm_ledger_economics", "economics/summary.csv")
def esm_ledger_economics(src: Sources) -> str:
    e = _sort(_econ(src), ("analysis", ECON_IDS), ("policy", ECON_POLICY_ORDER), "comparator", ("position", POSITION_ORDER), ("fees", _fee_order()))
    brk = lambda s: tex_escape(s).replace("[", "[\\allowbreak{}").replace("@", "@\\allowbreak{}")   # premium[short_spikes], calendar@count
    rows = [[tex_escape(r.analysis), _rr(tex_escape(r.policy)), _rr(brk(r.comparator)), _rr(brk(r.position)), tex_escape(r.fees),
             FORMATS["pct1"](r.SELC), FORMATS["usd0"](r.NET), FORMATS["usd2"](r.SNET), FORMATS["usd0"](r.ESred)] for r in e.itertuples()]
    return _breakable(longtable("lp{1.5cm}p{1.4cm}p{1.5cm}lrrrr", ["analysis", "policy", "comparator", "position", "fees", "SELC", "NET", "SNET", "ESred"], rows,
                                "Every economic cell: policies, comparators, positions and fee scenarios", "tab:esm-econ"), small=True)


@builder("esm_ledger_pr", "labels_pr/pr_all.csv")
def esm_ledger_pr(src: Sources) -> str:
    p = src.frame("labels_pr/pr_all.csv")
    _need(p, ["analysis", "block", "estimator", *PR_COLS], "pr_all")
    g = p.groupby(["analysis", "block", "estimator"])[list(PR_COLS)].median().reset_index()
    g = _sort(g, ("analysis", tuple(LEGACY) + tuple(ANALOGS)), ("block", tuple(BLOCKS)), ("estimator", ("xgb", "logit")))
    rows = [[_aid(r.analysis), tex_escape(r.block), r.estimator] + [_num(D2, getattr(r, c)) for c in PR_COLS] for r in g.itertuples()]
    return _breakable(longtable("lllrrrrr", ["analysis", "block", "est.", _stack("prec.", "($q$)"), _stack("recall", "($q$)"), _stack("prec.", "(eval)"),
                                             _stack("recall", "(eval)"), "lift"], rows,
                                "Precision and recall at the deployed cutoff, node medians", "tab:esm-pr"), small=True)


@builder("esm_ledger_monitors", "monitors/monitor_eval.csv")
def esm_ledger_monitors(src: Sources) -> str:
    return _monitors_longtable(src, "Drift monitors by analysis and block", "tab:esm-monitors", ledger=True)


@builder("esm_ledger_bh", "inference/bh.csv")
def esm_ledger_bh(src: Sources) -> str:
    b = _sort(_bh(src), ("family", BH_FAMILIES), "p")
    ran = lambda r: bool(r.included)                                   # a test that did not run has no estimate, p or decision
    diagnostic = lambda r: r.family in DESCRIPTIVE_BH_FAMILIES         # Ruling 118, M5: a flag of the descriptive family is a diagnostic
    flag = lambda r: FORMATS["yesno"](r.bh_reject) + (MARK_B if diagnostic(r) else "")
    rows = [[_rr(tex_escape(r.family)), _rr(tex_escape(r.test)), _num(FORMATS["g3"], r.estimate) if ran(r) else "--",
             FORMATS["pval"](r.p) if ran(r) else "--", flag(r) if ran(r) else "--"] for r in b.itertuples()]
    notes = "".join(_note_par(_not_computed(f)) for f in BH_FAMILIES if f not in set(b.family))
    if any(diagnostic(r) and ran(r) for r in b.itertuples()):
        notes += _note_par(DIAGNOSTIC_NOTE)
    return _breakable(longtable(
        "p{2.2cm}p{0.5\\textwidth}rrl", ["family", "test", "estimate", "$p$", "BH reject"], rows,
        "Every exploratory test with its BH decision", "tab:esm-bh") + notes, small=True)


# ---------------------------------------------------------------- driver
GENERATED_HEADER = "%% GENERATED by es.paper_tables; do not edit.\n"


def _index(status: dict[str, str]) -> str:
    cell = lambda text: str(text).replace("|", "/").replace("\n", " ")
    note = lambda k: "; ".join(t for t in (NOTES.get(k, ""), f"this build: {RUNTIME_NOTES[k]}" if k in RUNTIME_NOTES else "") if t)
    return "| table | status | note |\n|---|---|---|\n" + "".join(f"| {k} | {cell(v)} | {cell(note(k))} |\n" for k, v in status.items())


def build_all(src: Sources, tables_dir: Path, raise_on_failure: bool = True) -> dict[str, str]:
    """Build every table whose sources exist and list all of them in INDEX.md. A table whose source file is absent is
    "skipped: missing <file>", and a table that is skipped or fails loses the file of an earlier run (a generated file that
    outlived its sources would print old numbers). A builder that raises (a present source without a column the contract
    names) is "FAILED": the other tables are still built, INDEX.md is written, and then a LookupError names every failed
    table (raise_on_failure=False returns the status instead, for main)."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    RUNTIME_NOTES.clear()
    status, errors = {}, {}
    for name, (fn, needs) in BUILDERS.items():
        path = tables_dir / f"{name}.tex"
        missing = [n for n in needs if not src.exists(n)]
        text = None
        if missing:
            status[name] = "skipped: missing " + ", ".join(missing)
        else:
            try:
                text = fn(src)
                if not isinstance(text, str):
                    raise TypeError(f"the builder returned {type(text).__name__}, not LaTeX text")
            except Exception as exc:           # reported below, with the table's name; the others are still built
                text = None
                errors[name] = exc
                status[name] = f"FAILED: {type(exc).__name__}: {exc}"
        if text is None:
            if path.exists() and path.read_text().startswith(GENERATED_HEADER):
                path.unlink()
            continue
        path.write_text(GENERATED_HEADER + text)
        status[name] = "built"
    (tables_dir / "INDEX.md").write_text(_index(status))
    if errors and raise_on_failure:
        first = next(iter(errors.values()))
        raise LookupError(f"{len(errors)} table(s) failed: " + "; ".join(f"{n}: {type(e).__name__}: {e}" for n, e in errors.items())) from first
    return status


def main(argv=None) -> int:
    from . import paper_tables_sealed  # noqa: F401  (Task 6 registers the sealed builders on import)
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--paper-dir", default=str(PAPER_DIR))
    a = ap.parse_args(argv)
    status = build_all(Sources(Path(a.out_dir), Path(a.paper_dir), DOCS_SEALED), Path(a.paper_dir) / "generated" / "tables",
                       raise_on_failure=False)
    built = sum(v == "built" for v in status.values())
    failed = {k: v for k, v in status.items() if v.startswith("FAILED")}
    print(f"{built} tables built, {len(status) - built} skipped or failed (see generated/tables/INDEX.md)")
    for name, v in status.items():
        if v.startswith("skipped"):
            print(f"  {name}: {v}")
    for name, note in RUNTIME_NOTES.items():                      # a measurement of this build that the reader of the console should see
        print(f"  note, {name}: {note}")
    for name, v in failed.items():
        print(f"  {name}: {v}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    # `python -m es.paper_tables` runs this file as __main__, and paper_tables_sealed imports it as es.paper_tables, a second
    # module with its own BUILDERS: hand over to that one, so that the builders Task 6 registers are the ones that get built.
    from es.paper_tables import main as _canonical_main
    sys.exit(_canonical_main())
