"""Figures 1-4 (spec v3 section 7; docs/energy_systems_requirements.md) and the three figures of plan 2C (FigFail, FigDrift,
FigContrast: one finding each). matplotlib, Arial lettering 8-9 pt (8 pt minimum), drawn at the sn-jnl text width (372 pt =
130.7 mm), Okabe-Ito colours with marker, hatch and line-style redundancy, no transparency (EPS has none).
Each figure is saved as PDF (what main.tex includes) and EPS (what the journal uploads). Each figure is tied
to the output files in SOURCES; a figure whose sources are absent is skipped (and its stale PDF and EPS removed) and
generated/figures/INDEX.md says why. Test A has no panel, row or word in any figure until it has run: while it has not,
Fig1 draws five panels, Fig2 and Fig3 four rows and Fig4 two panels, and INDEX.md gives their status as plain 'built' (a figure that
prints nothing of Test A has nothing to wait for); a Test A that has run is drawn, and one whose test failed its coverage rule says
'Test A: did not run' (their status is as it always was). The three new figures never have a Test A panel. INDEX.md also lists every
file a figure reads and what each figure shows (the caption notes: the whisker rule, the open markers, the approximate intervals).

FigDrift draws the alert rate over trailing windows of eight Monday-start weeks (es.run_driftpath.window_r, from drift_path.csv), not the rate of
one week: the weekly series were too noisy to show what a cutoff does.
Run: DART_PROJECT_ROOT=... /opt/anaconda3/bin/python3 -m es.paper_figures

What the 8 pt rule means in practice (checked by tests/es/test_paper_figures.py on the PDF and the EPS):
- No mathtext sub- or superscripts. mathtext sets them at 70 % of the size (6.3 pt at 9 pt) and Arial has no subscript
  digits, so "log10" is three text objects: "log" at 9 pt, a lowered "10" at 8 pt, and "R" (italic) at 9 pt. J_lambda is
  written "J, lambda = 0.5". Only upright and italic Arial at 8 or 9 pt is drawn.
- Width: savefig crops the margins (bbox_inches="tight"), so a file is about 5 % narrower than the text width; included at
  \\textwidth it is scaled by about 1.05 and prints at 8.4 and 9.4 pt. A test bounds the saved width above (never scaled
  down) and below (labels never print above 12 pt).
- Arial is required, not preferred: if matplotlib cannot find it the build stops (RuntimeError) rather than drawing DejaVu.
- STYLE is applied by each figure function and by save() themselves, because fonts are resolved when a figure is
  drawn and saved; a caller that does not go through build_all still gets embedded Arial (TrueType, fonttype 42).
- Output is reproducible: no creation date in the PDF (metadata) and a fixed one in the EPS (SOURCE_DATE_EPOCH)."""
from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from matplotlib.text import Annotation  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullLocator  # noqa: E402

from .analyses import ANALOGS, LEGACY  # noqa: E402
from .blocks import PAPER_BLOCK_NAMES  # noqa: E402
from .config import ALPHA, OUT_DIR  # noqa: E402
from .measures import LAMS, j_key  # noqa: E402
from .paper_macros import Sources  # noqa: E402
from .paper_paths import DOCS_SEALED, FIGURES_DIR, PAPER_DIR  # noqa: E402
from .paper_pins import BANDS, band_text, pct_of, signature_default  # noqa: E402
from .paper_tables import _cell, _summary  # noqa: E402
from .run_driftpath import COLUMNS as DRIFT_COLUMNS  # noqa: E402
from .run_driftpath import P4_WINDOW_DAYS, WINDOW_MIN_WEEKS, WINDOW_WEEKS, window_r  # noqa: E402
from .run_driftpath import SOURCE as DRIFT_SOURCE  # noqa: E402

MM = 1 / 25.4
#: The sn-jnl \textwidth in inches (372 pt = 130.7 mm, measured 23 Oct; under the journal's 174 mm). The figures are drawn
#: this wide, but savefig crops their margins, so the saved files are about 5 % narrower (see the module docstring).
WIDTH = 372 / 72.27
#: 8 pt is the smallest size anywhere (journal: 8-12 pt at final size).
STYLE = {"font.family": "sans-serif", "font.sans-serif": ["Arial"], "pdf.fonttype": 42, "ps.fonttype": 42,
         "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
         "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
         "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
         "lines.linewidth": 1.0, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "figure.dpi": 150}
TEXT_PT, SUB_PT, SUB_DROP = 9.0, 8.0, 2.0    # label size, subscript size, and how far the subscript is lowered (points)
GREY, BAND, BLACK = "#7F7F7F", "#E8E8E8", "#000000"
TEXT_GREY = "#595959"        # grey lettering (7:1 on white); GREY is for rules and bands only
BLUE, ORANGE, GREEN = "#0072B2", "#E69F00", "#009E73"
OPERATOR = {"NYISO": (BLUE, "o"), "ISONE": (ORANGE, "s"), "MISO": (GREEN, "^"), "CAISO": ("#CC79A7", "D")}
MARKER_AREA, JITTER = 20, 0.30   # Fig1 node markers: area in pt^2 (4.5 pt across, up from 3: the shapes read in greyscale) and x spread
BLOCK_ORDER = ["G6", "DART3", "LEV6", "ROLL_DISP9", "DART18", "G1_24"]
LABEL = {"2024-2025": "2024→2025", "25H1-26H1": "25H1→26H1 (post-open)", "AQ3": "AQ3: 2025-Q3", "AQ4": "AQ4: 2025-Q4",
         "AQ1": "AQ1: 2026-Q1", "AQ2": "AQ2: 2026-Q2", "SA": "Test A: 2026-Q3", "SB": "Test B: 2026-Q4", "POOL": "AQ3+AQ4 (rehearsal)"}
ACTIVITY, CI = "analogs/activity_all.csv", "economics/bootstrap_ci.csv"
#: The frame behind Table 3 (tab_block_audit): FigFail draws its cells.
SUMMARY = "legacy/summary.csv"
LEGACY_PANELS = ("2024-2025", "25H1-26H1")
ANALOG_PANELS = ("AQ3", "AQ4", "AQ2")
LEGACY_RUNS = tuple(f"legacy/{aid}__{b}__xgb__s42__P1__a0.05.csv" for aid in LEGACY_PANELS for b in BLOCK_ORDER)
SOURCES = {"Fig1": ("legacy/2024-2025__G1_24__xgb__s42__P1__a0.05.csv", ACTIVITY),
           "Fig2": (CI,), "Fig3": (CI,), "Fig4": (ACTIVITY,),
           "FigFail": (SUMMARY,), "FigDrift": (DRIFT_SOURCE,), "FigContrast": (CI,)}
#: Every file a figure reads (INDEX.md lists these; SOURCES holds only what must exist for the figure to be drawn at all).
READS = {**SOURCES, "Fig1": (*LEGACY_RUNS, ACTIVITY)}
#: The Test A table that each figure adds a panel or row from once Test A has run. None: the figure never has a Test A panel.
SEALED_A = {"Fig1": "sealed/A/tables/audit_activity.csv", "Fig2": "sealed/A/tables/estimands.csv",
            "Fig3": "sealed/A/tables/estimands.csv", "Fig4": "sealed/A/tables/E5_blocks.csv",
            "FigFail": None, "FigDrift": None, "FigContrast": None}
#: The word of the `pending` argument of the four older figure functions for a Test A that has not run: it has no panel then. The other word
#: (build_all passes "did not run") gives the panel that says so.
PENDING = "pending"
ECON_PANELS = ["AQ3", "AQ4", "AQ1", "AQ2", "POOL", "SA"]
BAND_KEY = "0.5_2"           # the failure band of R drawn in Fig1 and counted by fail_0.5_2: es.paper_pins.BANDS[BAND_KEY]
J_LAMBDA = 0.5               # the weight of J_lambda that Fig3 shows; one of es.measures.LAMS
J_KEY = j_key(J_LAMBDA)
if J_LAMBDA not in LAMS:
    raise ImportError(f"Fig3 shows J at lambda {J_LAMBDA}, which is not one of es.measures.LAMS {LAMS}")
#: Fig2, left panel. (statistic, legend label, marker, colour, line of the row). The four carry percentile intervals; the
#: two "P1 - comparator" rows are the paired contrasts of controller Ruling 81 (SELC of P1 minus SELC of the comparator at
#: P1's alert count, one paired interval); the comparator levels below them are open markers without intervals.
FIG2_INTERVALS = (("SELC", "alerts (P1)", "o", BLACK, 0),
                  ("E4:dSELC|calendar@count", "P1 − calendar-only (paired)", "s", BLUE, 1),
                  ("E4:dSELC|VOL@count", "P1 − VOL (paired)", "^", ORANGE, 2),
                  ("E4:SELC_decile", "alerts, decile strata", "D", GREEN, 3))
FIG2_LEVELS = (("E4:SELC|calendar@count", "calendar-only (level)", "s", BLUE, 1),
               ("E4:SELC|VOL@count", "VOL (level)", "^", ORANGE, 2))
#: Fig2, right panel (Ruling 118, G1 M6). The tail effect ESred of the alerts (E3) with its interval, the two paired contrasts
#: dESred|c = ESred(P1) - ESred(comparator) at P1's alert count (the tail protection of the alerts beyond a rule that alerts as often)
#: and the comparators' own ESred as open markers. Each series sits on the line, in the marker and in the colour of its SELC twin in
#: the left panel, so that a line of the figure is one estimate in both measures. (statistic, marker, colour, line of the row)
FIG2_ESRED_INTERVALS = (("E3:ESred", "o", BLACK, 0), ("E4:dESred|calendar@count", "s", BLUE, 1), ("E4:dESred|VOL@count", "^", ORANGE, 2))
FIG2_ESRED_LEVELS = (("E4:ESred|calendar@count", "s", BLUE, 1), ("E4:ESred|VOL@count", "^", ORANGE, 2))
E1_PARTS = ((-0.2, "timing", BLUE, None), (0.2, "level", ORANGE, "//"))
APPROX_LINE = "\napproximate interval"


# ---------------------------------------------------------------- style
def _require_arial() -> None:
    """The journal wants Arial. Without it matplotlib only logs a warning and draws DejaVu Sans, so check the three faces
    the figures use (regular; italic for the mathtext variables; bold for the mathtext bold) and refuse to go on."""
    for spec, style in (("Arial", "Regular"), ("Arial:italic", "Italic"), ("Arial:bold", "Bold")):
        try:
            face = font_manager.get_font(font_manager.findfont(spec, fallback_to_default=False))
            found = f"{face.family_name} {face.style_name}"
            ok = face.family_name == "Arial" and style in face.style_name
        except (ValueError, OSError) as exc:
            found, ok = f"nothing ({exc})", False
        if not ok:
            raise RuntimeError(f"matplotlib cannot find {spec!r}: it found {found}. The journal requires Arial lettering and "
                               "this module never substitutes another font; install Arial and rebuild matplotlib's font cache.")


@contextmanager
def _style():
    """STYLE for the duration of a block and nothing outside it: no rcParams are left changed behind the caller."""
    _require_arial()
    with plt.rc_context(STYLE):
        yield


@contextmanager
def _fixed_creation_date():
    """matplotlib's PS backend writes the time into %%CreationDate unless SOURCE_DATE_EPOCH is set (reproducible-builds.org);
    an epoch already in the environment is kept, otherwise 0. (The PDF backend takes metadata={'CreationDate': None}.)"""
    before = os.environ.get("SOURCE_DATE_EPOCH")
    os.environ["SOURCE_DATE_EPOCH"] = before if before is not None else "0"
    try:
        yield
    finally:
        if before is None:
            os.environ.pop("SOURCE_DATE_EPOCH", None)
        else:
            os.environ["SOURCE_DATE_EPOCH"] = before


def _with_test_a(sealed: bool, pending: str) -> bool:
    """Whether a figure has a Test A panel or row at all: when Test A has run (its results are drawn) and when it did not run (the panel says
    so). While it has not run (`pending` is the default word) there is none: a panel that only says 'pending' is not a figure."""
    return bool(sealed) or pending != PENDING


def _pending(ax, text):
    """A panel whose test did not run. In Fig1 and Fig4 its y axis is shared with the other panels, so the ticks are hidden
    with tick_params: set_yticks([]) would empty the shared locator and take the tick labels off every panel."""
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, color=TEXT_GREY)
    ax.tick_params(axis="both", which="both", length=0, labelleft=False, labelbottom=False)


def _legend_below(fig, handles, ncol, note: str | None = None) -> float:
    """A figure legend at the foot of the figure (inside it, not on any axes), with an optional one-line note under it.
    Returns the fraction of the figure height they need, for tight_layout(rect=(0, fraction, 1, 1))."""
    renderer, height, base = fig.canvas.get_renderer(), fig.bbox.height, 0.0
    if note:
        base = fig.text(0.5, 0.0, note, ha="center", va="bottom", fontsize=SUB_PT, color=TEXT_GREY).get_window_extent(renderer).height / height + 0.004
    for columns in range(ncol, 0, -1):                  # as many columns as fit: a legend wider than the figure would widen the saved file
        leg = fig.legend(handles=handles, loc="lower center", ncol=columns, frameon=False, bbox_to_anchor=(0.5, base), borderaxespad=0.1)
        if columns == 1 or leg.get_window_extent(renderer).width <= 0.98 * fig.bbox.width:
            break
        leg.remove()
    return base + leg.get_window_extent(renderer).height / height + 0.015


def _subscript_ylabel(fig, axes, head: str, tail: str, sub: str = "10") -> None:
    """The y label head + sub + tail with a real 8 pt subscript, as three text objects: head (9 pt), `sub` (8 pt, lowered by
    SUB_DROP points) and tail (9 pt). Each axes keeps its real y label as a hidden placeholder, so that tight_layout has made
    room for it; the pieces are put where the placeholder was drawn. The second and third piece are anchored to the top of the
    one before (an annotation relative to an artist), so they join whatever text metrics the PDF and PS backends use."""
    fig.canvas.draw()                                    # the placeholders now sit where they will be saved
    renderer = fig.canvas.get_renderer()
    for ax in axes:
        label = ax.yaxis.label
        box = label.get_window_extent(renderer)
        label.set_visible(False)
        piece = fig.text(box.x1 / fig.bbox.width, box.y0 / fig.bbox.height, head, rotation=90, ha="right", va="bottom", fontsize=TEXT_PT)
        for text, size, drop in ((sub, SUB_PT, SUB_DROP), (tail, TEXT_PT, -SUB_DROP)):      # 90 degrees: down is to the right
            nxt = Annotation(text, xy=(1, 1), xycoords=piece, xytext=(drop, 0), textcoords="offset points", rotation=90,
                             ha="right", va="bottom", fontsize=size)
            fig.add_artist(nxt)
            piece = nxt


def _interval(ax, est, lo, hi, pos, *, color, marker=None, ms=4.0, lw=0.8, cap=1.5, vertical=False, hollow=False):
    """[lo, hi] as a bar with caps, and the estimate as a marker (open, if `hollow`). The bar is drawn from its ends, not as
    est minus lo and hi minus est, so an estimate that falls outside its percentile interval (the ESred interval is only
    approximate) cannot make matplotlib raise on a negative error. A missing estimate or interval is not drawn."""
    est, lo, hi, pos = (np.asarray(v, float) for v in (est, lo, hi, pos))
    has = ~(np.isnan(lo) | np.isnan(hi))
    if has.any():
        mid, half = (lo + hi) / 2, np.abs(hi - lo) / 2
        kw = {"fmt": "none", "ecolor": color, "elinewidth": lw, "capsize": cap, "capthick": lw}
        if vertical:
            ax.errorbar(pos[has], mid[has], yerr=half[has], **kw)
        else:
            ax.errorbar(mid[has], pos[has], xerr=half[has], **kw)
    if marker is not None and (~np.isnan(est)).any():
        pt = ~np.isnan(est)
        x, y = (pos[pt], est[pt]) if vertical else (est[pt], pos[pt])
        look = {"mfc": "white", "mec": color, "mew": 1.0, "zorder": 3} if hollow else {"zorder": 4}   # white, not transparent: EPS has none; estimates on top
        ax.plot(x, y, marker, color=color, ms=ms, ls="", **look)


def _plain_money_axis(ax):
    """Dollars as plain numbers: no '1e6' multiplier or offset text at the end of the axis (it ran into the x label)."""
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)


def save(fig, name: str, figures_dir: Path = FIGURES_DIR):
    figures_dir.mkdir(parents=True, exist_ok=True)
    pdf, eps = figures_dir / f"{name}.pdf", figures_dir / f"{name}.eps"
    with _style(), _fixed_creation_date():
        fig.savefig(pdf, format="pdf", metadata={"CreationDate": None})
        fig.savefig(eps, format="eps")
    plt.close(fig)
    return pdf, eps


# ---------------------------------------------------------------- data
def _need(df: pd.DataFrame, cols, where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise LookupError(f"{where}: missing columns {missing}; have {list(df.columns)}")


def _drift_part(frame: pd.DataFrame, where: str, **fixed) -> pd.DataFrame:
    """The columns Fig1 draws from one per-node activity frame. log10 R is the log10 of R_floored, the floor that
    es.policies.activity puts under R (half an alert: 1/(2 n alpha)) and that log_error, Fig4 and E5 are built from, so a node
    that never alerts has a finite value that agrees with them. A frame without that column falls back to R, floored at 1e-3 so
    that log10 exists. zero_activity is the frame's own flag, or R = 0."""
    _need(frame, ["market", "node", "R"], where)
    r = frame["R"].astype(float)
    if "R_floored" in frame.columns:
        floored = frame["R_floored"].astype(float)
        floored = floored.where(floored.notna(), np.maximum(r, 1e-3))
    else:
        floored = np.maximum(r, 1e-3)
    zero = frame["zero_activity"].fillna(False).astype(bool) if "zero_activity" in frame.columns else r <= 0
    return frame.assign(**fixed, zero_activity=zero, log10R=np.log10(floored))[["analysis", "block", "market", "node", "R", "zero_activity", "log10R"]]


def r_drift_frame(src: Sources) -> pd.DataFrame:
    parts = []
    for aid in LEGACY_PANELS:
        for b in BLOCK_ORDER:
            p = f"legacy/{aid}__{b}__xgb__s42__P1__a0.05.csv"
            if src.exists(p):
                parts.append(_drift_part(src.frame(p), p, analysis=aid, block=b))
    if src.exists(ACTIVITY):
        a = src.frame(ACTIVITY)
        _need(a, ["analysis", "block", "estimator", "policy"], ACTIVITY)
        a = a[(a.policy == "P1") & (a.estimator == "xgb") & a.block.isin(BLOCK_ORDER) & a.analysis.isin(ANALOG_PANELS)]
        parts.append(_drift_part(a, ACTIVITY))
    if src.exists(SEALED_A["Fig1"]):
        s = src.frame(SEALED_A["Fig1"])
        _need(s, ["block", "estimator", "policy", "maintenance", "seed"], SEALED_A["Fig1"])
        s = s[(s.policy == "P1") & (s.estimator == "xgb") & (s.maintenance == "roll") & (s.seed == 42) & s.block.isin(BLOCK_ORDER)]
        parts.append(_drift_part(s, SEALED_A["Fig1"], analysis="SA"))
    parts = [p for p in parts if len(p)]                          # concat of an empty frame changes dtypes (pandas FutureWarning)
    cols = ["analysis", "block", "market", "node", "R", "zero_activity", "log10R"]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)


def e1_frame(src: Sources) -> pd.DataFrame:
    ci = src.frame(CI)
    _need(ci, ["analysis", "stat", "estimate", "lo", "hi"], CI)
    ci = ci[~ci.analysis.astype(str).str.contains("|ex-CAISO", regex=False)].copy()     # the CAISO-free twins are tabulated, not drawn
    ci.loc[ci.analysis.astype(str).str.startswith("AQ3+AQ4"), "analysis"] = "POOL"
    if src.exists(SEALED_A["Fig2"]):
        sealed = src.frame(SEALED_A["Fig2"])
        _need(sealed, ["stat", "estimate", "lo", "hi"], SEALED_A["Fig2"])
        ci = pd.concat([ci, sealed.assign(analysis="SA")], ignore_index=True)
    return ci


def blocks_frame(src: Sources):
    act = src.frame(ACTIVITY)
    _need(act, ["analysis", "block", "estimator", "policy", "log_error", "fail_0.5_2"], ACTIVITY)
    act = act[(act.policy == "P1") & (act.estimator == "xgb") & act.block.isin(BLOCK_ORDER)]
    e5 = src.frame(SEALED_A["Fig4"]) if src.exists(SEALED_A["Fig4"]) else None
    if e5 is not None:
        _need(e5, ["block", "median_log_error", "lo", "hi", "failures"], SEALED_A["Fig4"])
    return act, e5


def _ci_rows(ci: pd.DataFrame, analyses, stat: str, *, sealed: bool = False, interval: bool = True):
    """(estimate, lo, hi) of one statistic, one entry per analysis. Every analysis must have a finite estimate (and, if
    `interval`, a finite interval with lo <= hi): a missing statistic or a NaN would otherwise be left out of the drawing
    while its legend entry stays (a bootstrap_ci.csv from before Ruling 68 has E4:SELC_month, not E4:SELC_decile). Only the
    Test A row may be absent, and only while the test is pending (`sealed` False)."""
    est, lo, hi = [], [], []
    for a in analyses:
        g = ci[(ci.analysis == a) & (ci.stat == stat)]
        if len(g) > 1:
            raise LookupError(f"{CI}: {len(g)} rows for analysis {a!r}, stat {stat!r}; expected at most one")
        if not len(g):
            if a == "SA" and not sealed:
                est.append(np.nan)
                lo.append(np.nan)
                hi.append(np.nan)
                continue
            raise LookupError(f"{CI}: analysis {a!r} has no row for stat {stat!r}"
                              + (" (Test A has run, so its row is required)" if a == "SA" else "")
                              + f"; it has {sorted(map(str, ci[ci.analysis == a].stat))}")
        r = g.iloc[0]
        if pd.isna(r.estimate):
            raise LookupError(f"{CI}: analysis {a!r}, stat {stat!r}: the estimate is NaN")
        if interval and (pd.isna(r.lo) or pd.isna(r.hi)):
            raise LookupError(f"{CI}: analysis {a!r}, stat {stat!r}: the interval is NaN")
        if interval and r.lo > r.hi:
            raise LookupError(f"{CI}: analysis {a!r}, stat {stat!r}: lo {r.lo} is above hi {r.hi}")
        est.append(r.estimate)
        lo.append(r.lo)
        hi.append(r.hi)
    return np.array(est, float), np.array(lo, float), np.array(hi, float)


def _approximate(ci: pd.DataFrame, analyses, stats) -> bool:
    """True if the data flags a drawn interval of these statistics as approximate (the `approximate` column that
    es.run_economics writes: the ESred and J_lambda intervals), so the axis label says so only where the data does."""
    if "approximate" not in ci.columns:
        return False
    flags = ci[ci.analysis.isin(analyses) & ci.stat.isin(list(stats))]["approximate"]
    return bool(flags.fillna(False).astype(bool).any())


def _panel_labels(analyses, sealed, pending="pending"):
    return [LABEL[a] + ("" if (a != "SA" or sealed) else f": {pending}") for a in analyses]


# ---------------------------------------------------------------- figures
def fig1_r_drift(df: pd.DataFrame, sealed: bool, *, pending: str = "pending") -> Figure:
    with _style():
        with_a = _with_test_a(sealed, pending)
        panels = ["2024-2025", "25H1-26H1", "AQ3", "AQ4", "AQ2"] + (["SA"] if with_a else [])      # the sixth cell of the grid is Test A's
        band = BANDS[BAND_KEY]
        band_lo, band_hi = float(np.log10(band[0])), float(np.log10(band[1]))
        rows = {a: df[df.analysis == a] for a in (*panels, "SA")}
        if sealed and rows["SA"].empty:
            raise LookupError(f"Fig1: the Test A table exists ({SEALED_A['Fig1']}) but has no primary-pipeline row for the Test A panel")
        shown = [a for a in panels if len(rows[a]) and (sealed or a != "SA")]
        used = df[df.analysis.isin(shown)]
        body = used[~used.zero_activity.astype(bool)].log10R.dropna()
        lo, hi = min([band_lo, *body]), max([band_hi, *body])         # the range of the nodes that alerted, and the band
        pad = 0.06 * (hi - lo)
        ylo, yhi = lo - pad, hi + pad
        floor_y, has_zero = ylo, bool(used.zero_activity.astype(bool).any())
        if has_zero:                                                  # nodes with no alerts get a strip of their own under the data
            strip = 0.14 * (yhi - ylo)
            floor_y, ylo = ylo - strip / 2, ylo - strip
        fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 100 * MM), sharey=True)
        if not with_a:
            fig.delaxes(axes[1, 2])                                   # no Test A: five panels, not an empty sixth
        x = np.arange(len(BLOCK_ORDER))
        rng = np.random.default_rng(20261022)
        for ax, aid in zip(axes.ravel(), panels):
            ax.set_title(LABEL[aid], loc="left")
            if aid == "SA" and not sealed:
                _pending(ax, f"Test A: {pending}")
                continue
            d = rows[aid]
            if d.empty:
                _pending(ax, "not available")
                continue
            ax.axhspan(band_lo, band_hi, color=BAND, lw=0, zorder=0)
            ax.axhline(0, color=GREY, lw=0.6, ls=(0, (4, 3)), zorder=1)
            for i, b in enumerate(BLOCK_ORDER):
                g = d[d.block == b]
                for m, (col, mk) in OPERATOR.items():
                    gm = g[g.market == m]
                    xs = i + rng.uniform(-JITTER, JITTER, len(gm))
                    z = gm.zero_activity.to_numpy(bool)
                    ax.scatter(xs[~z], gm.log10R.to_numpy(float)[~z], s=MARKER_AREA, facecolors=col, edgecolors="white", linewidths=0.3,
                               marker=mk, zorder=2)
                    if z.any():                                       # open marker at the foot of the axis, not at its true (far lower) value
                        ax.scatter(xs[z], np.full(int(z.sum()), floor_y), s=MARKER_AREA, facecolors="white", edgecolors=col, linewidths=0.9,
                                   marker=mk, zorder=4)
                if len(g):
                    ax.hlines(g.log10R.median(), i - 0.34, i + 0.34, color=BLACK, lw=1.2, zorder=3)
            ax.set_xticks(x)
            ax.set_xticklabels([PAPER_BLOCK_NAMES[b] for b in BLOCK_ORDER], rotation=35, ha="right")
        axes[0, 0].set_ylim(ylo, yhi)                                 # shared by all six panels
        for ax in axes[:, 0]:
            ax.set_ylabel("log10 $R$")                                # placeholder: replaced by _subscript_ylabel below
        ms = float(np.sqrt(MARKER_AREA))
        handles = [Line2D([], [], color=c, marker=mk, ls="", ms=ms, mec="white", mew=0.3, label=m) for m, (c, mk) in OPERATOR.items()]
        handles += [Line2D([], [], color=BLACK, lw=1.2, label="node median"),
                    Patch(facecolor=BAND, edgecolor="none", label=f"R within {band_text(band)}"),
                    Line2D([], [], color=GREY, lw=0.6, ls=(0, (4, 3)), label="R = 1")]
        if has_zero:
            handles.append(Line2D([], [], color=BLACK, marker="o", mfc="white", mec=BLACK, mew=0.9, ls="", ms=ms,
                                  label="no alerts (at the foot of the axis)"))
        fig.tight_layout(rect=(0, _legend_below(fig, handles, 4), 1, 1), h_pad=1.6)
        _subscript_ylabel(fig, list(axes[:, 0]), "log", " $R$")
        fig.test_a_drawn = bool(sealed and len(rows["SA"]))
        return fig


def _econ_rows(ci: pd.DataFrame, sealed: bool, pending: str) -> list[str]:
    """The rows of Fig2 and Fig3: the analyses that the file has, and Test A's row only when Test A has run or did not run."""
    with_a = _with_test_a(sealed, pending)
    return [a for a in ECON_PANELS if (a != "SA" and a in set(ci.analysis)) or (a == "SA" and with_a)]


def fig3_decomposition(ci: pd.DataFrame, sealed: bool, *, pending: str = "pending") -> Figure:
    with _style():
        analyses = _econ_rows(ci, sealed, pending)
        keys = [("NET", "NET (USD)"), ("ESred", "ESred (USD)"), (J_KEY, f"$J$, $\\lambda$ = {J_KEY.split('_', 1)[1]} (USD)")]   # J with lambda, no subscript
        fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 70 * MM), sharey=True)
        y = np.arange(len(analyses))
        drawn = False
        for ax, (key, label) in zip(axes, keys):
            ax.axvline(0, color=GREY, lw=0.6)
            for off, part, col, hatch in E1_PARTS:
                est, lo, hi = _ci_rows(ci, analyses, f"E1:{key}:{part}", sealed=sealed)
                has = ~np.isnan(est)                       # no zero-length bar (its black edge is a stray tick on the zero line)
                ax.barh((y + off)[has], est[has], height=0.36, color=col, hatch=hatch, edgecolor=BLACK, linewidth=0.4)
                _interval(ax, est, lo, hi, y + off, color=BLACK, lw=0.6)
                drawn = drawn or bool("SA" in analyses and has[analyses.index("SA")])
            tot, _, _ = _ci_rows(ci, analyses, f"E1:{key}:total", sealed=sealed, interval=False)
            ax.plot(tot, y, "D", color=BLACK, ms=3.5)
            ax.set_yticks(y)
            ax.set_yticklabels(_panel_labels(analyses, sealed, pending))
            ax.set_xlabel(label + (APPROX_LINE if _approximate(ci, analyses, [f"E1:{key}:{p}" for _, p, _, _ in E1_PARTS] + [f"E1:{key}:total"]) else ""))
            _plain_money_axis(ax)
        axes[0].invert_yaxis()
        handles = [Patch(facecolor=col, edgecolor=BLACK, linewidth=0.4, hatch=hatch, label=part) for _, part, col, hatch in E1_PARTS]
        handles.append(Line2D([], [], color=BLACK, marker="D", ms=3.5, ls="", label="total"))
        # NET and J of the pooled rehearsal are sums over its two quarters; ESred is a day-weighted mean, SELC a ratio of sums
        note = f"{LABEL['POOL']} row: NET and J are each a sum of two quarters" if "POOL" in analyses else None
        fig.tight_layout(rect=(0, _legend_below(fig, handles, 3, note), 1, 1))
        fig.test_a_drawn = bool(sealed and drawn)
        return fig


def fig2_selc_esred(ci: pd.DataFrame, sealed: bool, *, pending: str = "pending") -> Figure:
    with _style():
        analyses = _econ_rows(ci, sealed, pending)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(WIDTH, 70 * MM), sharey=True, gridspec_kw={"width_ratios": [3, 2]})
        y = np.arange(len(analyses))
        for ax in (ax1, ax2):
            ax.axvline(0, color=GREY, lw=0.6)
        handles = []
        drawn = False
        for stat, label, mk, col, line in FIG2_INTERVALS:             # estimates with intervals (SELC in percent, contrasts in percentage points)
            est, lo, hi = _ci_rows(ci, analyses, stat, sealed=sealed)
            _interval(ax1, 100 * est, 100 * lo, 100 * hi, y + (line - 1.5) * 0.17, color=col, marker=mk)
            handles.append(Line2D([], [], color=col, marker=mk, ms=4, lw=0.8, label=label))
            drawn = drawn or bool("SA" in analyses and np.isfinite(est[analyses.index("SA")]))
        for stat, label, mk, col, line in FIG2_LEVELS:                # the comparators' own SELC, for context: open markers, no interval
            est, _, _ = _ci_rows(ci, analyses, stat, sealed=sealed, interval=False)
            _interval(ax1, 100 * est, np.full_like(est, np.nan), np.full_like(est, np.nan), y + (line - 1.5) * 0.17, color=col, marker=mk, hollow=True)
            handles.append(Line2D([], [], color=col, marker=mk, ms=4, ls="", mfc="white", mec=col, mew=1.0, label=label))
        for stat, mk, col, line in FIG2_ESRED_INTERVALS:             # the same lines as the SELC rows: the alerts, then the paired contrasts
            est, lo, hi = _ci_rows(ci, analyses, stat, sealed=sealed)
            _interval(ax2, est, lo, hi, y + (line - 1.5) * 0.17, color=col, marker=mk)
        for stat, mk, col, line in FIG2_ESRED_LEVELS:                # the comparators' own ESred, context: open markers, no interval
            est, _, _ = _ci_rows(ci, analyses, stat, sealed=sealed, interval=False)
            _interval(ax2, est, np.full_like(est, np.nan), np.full_like(est, np.nan), y + (line - 1.5) * 0.17, color=col, marker=mk, hollow=True)
        ax1.set_yticks(y)
        ax1.set_yticklabels(_panel_labels(analyses, sealed, pending))
        ax1.invert_yaxis()
        left = [s for s, *_ in FIG2_INTERVALS]
        ax1.set_xlabel("SELC (%); differences in pp" + (APPROX_LINE if _approximate(ci, analyses, left) else ""))   # plain text: matplotlib is not LaTeX here
        right = [s for s, *_ in FIG2_ESRED_INTERVALS + FIG2_ESRED_LEVELS]
        ax2.set_xlabel("ESred (USD);\ndifferences in USD" + (APPROX_LINE if _approximate(ci, analyses, right) else ""))   # two lines: tight bbox ignores
        _plain_money_axis(ax2)                                                                                       # the width of an x label wider than its axes
        fig.tight_layout(rect=(0, _legend_below(fig, handles, 2), 1, 1))
        fig.test_a_drawn = bool(sealed and drawn)
        return fig


def _headroom(ax, top: float, points: float) -> None:
    """Raise the upper y limit (of ax and the axes that share it) until `points` of the axes height are free above `top`:
    solves Y - top = points * (Y - y0) / height for the limit Y. Call after tight_layout, which fixes the axes height."""
    height = ax.get_position().height * ax.figure.get_figheight() * 72
    y0, y1 = ax.get_ylim()
    need = (top - points * y0 / height) / (1 - points / height)
    if need > y1:
        ax.set_ylim(y0, need)


def _count(ax, x, top, n_fail):
    """'N fail' above a block's whisker or interval, rotated: six horizontal labels would run into each other."""
    ax.annotate(f"{n_fail} fail", (x, top), textcoords="offset points", xytext=(0, 3), ha="center", va="bottom", rotation=90,
                fontsize=SUB_PT, color=TEXT_GREY)


def fig4_blocks(act: pd.DataFrame, e5, sealed: bool, *, pending: str = "pending") -> Figure:
    with _style():
        with_a = _with_test_a(sealed, pending)
        fig, axes = plt.subplots(1, 3 if with_a else 2, figsize=(WIDTH, 65 * MM), sharey=True)
        x = np.arange(len(BLOCK_ORDER))
        tops = []
        for ax, aid in zip(axes[:2], ("AQ3", "AQ4")):
            ax.set_title(LABEL[aid], loc="left")
            sub = act[act.analysis == aid]
            data = [sub[sub.block == b].log_error.to_numpy(float) for b in BLOCK_ORDER]
            if all(len(d) == 0 for d in data):
                _pending(ax, "not available")
                continue
            # whiskers at the minimum and maximum (whis=(0, 100)): the failing nodes are the tail, so none may be left out
            ax.boxplot(data, positions=x, widths=0.55, whis=(0, 100), showfliers=False, medianprops={"color": BLACK}, boxprops={"color": BLUE},
                       whiskerprops={"color": BLUE}, capprops={"color": BLUE})
            for i, b in enumerate(BLOCK_ORDER):
                if len(data[i]):
                    tops.append(float(np.max(data[i])))
                    _count(ax, i, tops[-1], int(sub[sub.block == b]["fail_0.5_2"].sum()))
            ax.set_xticks(x)
            ax.set_xticklabels([PAPER_BLOCK_NAMES[b] for b in BLOCK_ORDER], rotation=35, ha="right")
        drawn = False
        if with_a:
            ax = axes[2]
            ax.set_title(LABEL["SA"], loc="left")
            if not sealed or e5 is None:
                _pending(ax, f"Test A: {pending}")
            else:
                e = e5.set_index("block").reindex(BLOCK_ORDER)
                _interval(ax, e.median_log_error, e.lo, e.hi, x, color=BLACK, marker="o", cap=2.0, vertical=True)
                counts = e.dropna(subset=["failures", "hi"])
                if counts.empty:
                    raise LookupError(f"Fig4: the Test A table exists ({SEALED_A['Fig4']}) but has no row for any of {BLOCK_ORDER}")
                for b, row in counts.iterrows():
                    tops.append(float(row.hi))
                    _count(ax, BLOCK_ORDER.index(b), tops[-1], int(row.failures))
                ax.set_xticks(x)
                ax.set_xticklabels([PAPER_BLOCK_NAMES[b] for b in BLOCK_ORDER], rotation=35, ha="right")
                drawn = True
        axes[0].set_ylabel("|log10 $R$| per node")      # placeholder: replaced by _subscript_ylabel below
        fig.tight_layout()
        if tops:
            _headroom(axes[0], max(tops), 36.0)         # room for the rotated label above the highest whisker or interval
        _subscript_ylabel(fig, [axes[0]], "|log", " $R$| per node")
        fig.test_a_drawn = drawn
        return fig


# ---------------------------------------------------------------- the three figures of plan 2C
#: The rows of FigFail, top to bottom: calendar, plain DART lags, absolute level, rolling dispersion, DART-relative, combined.
FAIL_ROWS = BLOCK_ORDER
#: The panels of FigFail: (heading, groups); a group is (heading over its columns or None, columns); a column is (analysis, estimator). The left panel
#: is the headline (XGBoost, the rolling origin); the right one shows that the pattern of fragile blocks is not universal (another estimator, and
#: XGBoost with the origin frozen at the 2026-H1 deployment).
FAIL_PANELS = (("XGBoost, rolling origin", ((None, (("2024-2025", "xgb"), ("25H1-26H1", "xgb"))),)),
               ("robustness", (("logistic regression", (("2024-2025", "logit"), ("25H1-26H1", "logit"))),
                               ("XGBoost,\nfrozen origin", (("frozen-26H1", "xgb"),)))))
#: The label of a column: the period it is deployed in, after the calibration period.
FAIL_PERIOD = {"2024-2025": "2024 to\n2025", "25H1-26H1": "2025-H1 to\n2026-H1", "frozen-26H1": "2026-H1"}
#: A cell of this many failing nodes or more has bold type and a heavy outline, so that the figure reads in greyscale.
HEAVY_FROM = 5
CELL_LW, HEAVY_LW = 1.0, 1.5
FAIL_COLORS = ("#FFFFFF", BLUE)       # the fill runs from white (no node fails) to the blue of the audit (every node fails)
INK_SWITCH = 0.5                      # white type on a fill darker than this (luminance), black type on a lighter one


def _band_words(band) -> str:
    """The band as '0.5 to 2' (the text of band_text without its brackets)."""
    return band_text(band).strip("[]").replace(", ", " to ")


def fail_frame(src: Sources) -> pd.DataFrame:
    """The cells of FigFail, one row per panel, column and block: the number of nodes whose realised alert rate is outside the failure band
    (``failures``) of the ``n_nodes`` of the audit. Read from the frame that tab_block_audit (Table 3) is built from, with its own cell reader (the
    primary seed, the policy P1 and the 5 % cutoff of the audit), and never recomputed; a cell that the frame lacks is an error that names it."""
    summary = _summary(src)
    _need(summary, ["n_nodes"], SUMMARY)
    rows, missing = [], []
    for panel, (_, groups) in enumerate(FAIL_PANELS):
        for _, columns in groups:
            for analysis, estimator in columns:
                for block in FAIL_ROWS:
                    failures, nodes = (_cell(summary, analysis, block, estimator, column, int) for column in ("failures", "n_nodes"))
                    if isinstance(failures, str) or isinstance(nodes, str):         # _cell answers a row that is not there with a pending sentinel
                        missing.append(f"{analysis}/{estimator}/{block}")
                        continue
                    rows.append({"panel": panel, "analysis": analysis, "estimator": estimator, "block": block, "failures": failures, "n_nodes": nodes})
    if missing:
        raise LookupError(f"{SUMMARY}: no primary row (P1, the primary seed, the audit's cutoff target) for {', '.join(missing)}: the cells of Table 3")
    return pd.DataFrame(rows)


def fig_fail(cells: pd.DataFrame) -> Figure:
    """FigFail: the heatmap of operating-point failures (finding 1). Rows are the feature blocks, columns the analyses of Table 3; a cell prints the
    number of failing nodes, its fill grows with that number on one scale for both panels (white for none, up to all the nodes of the audit), and a
    cell of HEAVY_FROM or more has bold type and a heavy outline."""
    with _style():
        sizes = sorted(set(cells["n_nodes"]))
        uniform = len(sizes) == 1
        cmap = LinearSegmentedColormap.from_list("failures", list(FAIL_COLORS))
        fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.0), sharey=True,
                                 gridspec_kw={"width_ratios": [sum(len(g) for _, g in groups) for _, groups in FAIL_PANELS]})
        for panel, ax in enumerate(axes):
            heading, groups = FAIL_PANELS[panel]
            columns = [c for _, g in groups for c in g]
            for c, (analysis, estimator) in enumerate(columns):
                for r, block in enumerate(FAIL_ROWS):
                    cell = cells[(cells.panel == panel) & (cells.analysis == analysis) & (cells.estimator == estimator) & (cells.block == block)]
                    if len(cell) != 1:
                        raise LookupError(f"FigFail: {len(cell)} cells for {analysis}/{estimator}/{block}; expected one")
                    k, n = int(cell.failures.iloc[0]), int(cell.n_nodes.iloc[0])
                    fill = cmap(k / max(sizes))
                    heavy = k >= HEAVY_FROM
                    ax.add_patch(Rectangle((c, r), 1, 1, facecolor=fill, edgecolor=BLACK if heavy else "white",
                                           linewidth=HEAVY_LW if heavy else CELL_LW, zorder=3 if heavy else 2))
                    ax.text(c + 0.5, r + 0.5, str(k) if uniform else f"{k}/{n}", ha="center", va="center", fontsize=TEXT_PT,
                            fontweight="bold" if heavy else "normal", color="white" if _luminance(fill) < INK_SWITCH else BLACK, zorder=4)
            ax.set_xlim(0, len(columns))
            ax.set_ylim(len(FAIL_ROWS), 0)                               # the first row at the top
            ax.xaxis.tick_top()
            ax.set_xticks(np.arange(len(columns)) + 0.5)
            ax.set_xticklabels([FAIL_PERIOD[a] for a, _ in columns])
            ax.tick_params(length=0, pad=2)
            for side in ax.spines.values():
                side.set_visible(False)
            ax.set_title(heading, loc="left", pad=52)
            x0 = 0
            for group, members in groups:
                if group:
                    ax.annotate(group, xy=(x0 + len(members) / 2, 1), xycoords=("data", "axes fraction"), xytext=(0, 25), textcoords="offset points",
                                ha="center", va="bottom", fontsize=SUB_PT)
                x0 += len(members)
        axes[0].set_yticks(np.arange(len(FAIL_ROWS)) + 0.5)
        axes[0].set_yticklabels([PAPER_BLOCK_NAMES[b] for b in FAIL_ROWS])
        handles = [Patch(facecolor="white", edgecolor=BLACK, linewidth=HEAVY_LW, label=f"{HEAVY_FROM} or more failing nodes (bold)")]
        of = f"of {sizes[0]}" if uniform else "of the nodes of each analysis"
        note = (f"Cell: failing nodes {of}, those with a realised alert rate outside {_band_words(BANDS[BAND_KEY])} times the target;\n"
                "darker fill, more failing nodes")
        fig.tight_layout(rect=(0, _legend_below(fig, handles, 1, note) + 0.02, 1, 1), w_pad=0.6)
        fig.test_a_drawn = False
        return fig


def _luminance(rgba) -> float:
    return 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]


#: FigDrift: the panels (title, analysis) and, for each policy, (id, legend label, colour, line style). P1 is solid black and P4 is dashed blue, so that
#: they separate in black and white as well. The median over nodes is thick and the two quartiles thin, all of one style.
DRIFT_PANELS = (("(a) Deployment year 2025", "2024-2025"), ("(b) First half of 2026", "25H1-26H1"))
DRIFT_POLICIES = (("P1", "fixed cutoff (P1)", BLACK, "-"), ("P4", f"rolling {P4_WINDOW_DAYS}-day cutoff (P4)", BLUE, (0, (4, 2))))
DRIFT_THICK, DRIFT_THIN = 1.8, 0.6
#: The axis of R: log base 2 from linthresh up, with a linear stretch below it so that zero (a window without an alert) can be drawn. The ticks are the
#: five of the brief and the zero.
DRIFT_TICKS = (0, 0.25, 0.5, 1, 2, 4)
DRIFT_LINTHRESH, DRIFT_LINSCALE, DRIFT_FLOOR = 0.125, 0.45, -0.03
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
_NUMBER_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve")
_ORDINAL_WORDS = ("zeroth", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth", "eleventh", "twelfth")


def _number_word(n: int) -> str:
    """The English word of a small whole number (the window of eight weeks is 'eight' in the note under FigDrift), else its digits."""
    return _NUMBER_WORDS[n] if 0 <= n < len(_NUMBER_WORDS) else str(n)


def _ordinal_word(n: int) -> str:
    """The English ordinal of a small whole number ('fourth'), else its digits with 'th' (the first window ends with the fourth week)."""
    return _ORDINAL_WORDS[n] if 0 <= n < len(_ORDINAL_WORDS) else f"{n}th"


#: The note under the legend of FigDrift, in two lines (one line would be wider than the figure and widen the saved file).
DRIFT_NOTE = (f"thick line: median over nodes of the alert rate over the last {_number_word(WINDOW_WEEKS)} weeks\n"
              "(the weeks so far at the start); thin lines: its quartiles")


def drift_frame(src: Sources) -> pd.DataFrame:
    """The weekly path of es.run_driftpath (outputs/es/paper/drift_path.csv), with the week as a date."""
    d = src.frame(DRIFT_SOURCE)
    _need(d, DRIFT_COLUMNS, DRIFT_SOURCE)
    return d.assign(week_start=pd.to_datetime(d["week_start"]))


def drift_summary(frame: pd.DataFrame, analysis: str, policy: str) -> pd.DataFrame:
    """Per window end, over the nodes that have a window R: the quartiles and the median of R (``q1``, ``median``, ``q3``), the number of those nodes
    (``nodes``) and ``week``, the end of the window, where the point is drawn. The windows are those of es.run_driftpath.window_r on the weekly rows of
    this analysis and policy: the last eight Monday-start weeks (the weeks so far from the fourth week on at the start), the ratio of the alerts to the
    node-hours over the window, partial weeks included; a node whose window has too few hours is left out of it."""
    g = frame[(frame["analysis"] == analysis) & (frame["policy"] == policy)]
    if g.empty:
        raise LookupError(f"{DRIFT_SOURCE}: no rows for analysis {analysis!r}, policy {policy!r}")
    windows = window_r(g)
    if windows.empty:
        raise LookupError(f"{DRIFT_SOURCE}: analysis {analysis!r}, policy {policy!r} has no window (it needs {WINDOW_MIN_WEEKS} weeks with enough hours)")
    by_end = windows.groupby("week_end")["R"]
    q = by_end.quantile([0.25, 0.5, 0.75]).unstack()
    return pd.DataFrame({"week": q.index, "q1": q[0.25].to_numpy(), "median": q[0.5].to_numpy(), "q3": q[0.75].to_numpy(),
                         "nodes": by_end.size().to_numpy()}).reset_index(drop=True)


def fig_drift(frame: pd.DataFrame) -> Figure:
    """FigDrift: the alert rate over the target, P1 against P4, in trailing windows of eight weeks (finding 2). Two panels, one above the other: the
    deployment year 2025 and the first half of 2026. For each policy the median over nodes of the window's R is a thick line and the quartiles are
    two thin lines of the same style, each point at the end of its window (the first one after the fourth week). The axis is log base 2 with a place
    for zero, the band of failure is shaded and R = 1 is dashed."""
    with _style():
        series = {(analysis, policy): drift_summary(frame, analysis, policy) for _, analysis in DRIFT_PANELS for policy, *_ in DRIFT_POLICIES}
        top = max(DRIFT_TICKS[-1], *(s["q3"].max() for s in series.values())) * 1.05          # nothing is drawn above the axis
        band = BANDS[BAND_KEY]
        fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 4.0), sharey=True)
        for ax, (title, analysis) in zip(axes, DRIFT_PANELS):
            ax.set_yscale("symlog", base=2, linthresh=DRIFT_LINTHRESH, linscale=DRIFT_LINSCALE)
            ax.yaxis.set_minor_locator(NullLocator())
            ax.axhspan(band[0], band[1], color=BAND, lw=0, zorder=0)
            ax.axhline(1, color=GREY, lw=0.6, ls=(0, (4, 3)), zorder=1)
            for z, (policy, _, color, style) in zip((3, 2), DRIFT_POLICIES):
                s = series[(analysis, policy)]
                for column, lw in (("q1", DRIFT_THIN), ("q3", DRIFT_THIN), ("median", DRIFT_THICK)):
                    ax.plot(s["week"], s[column], color=color, ls=style, lw=lw, zorder=z)
            window = LEGACY[analysis].deploy                                  # the months of the deployment window, from the code
            ax.set_xlim(pd.Timestamp(window.start), pd.Timestamp(window.end))
            ax.set_ylim(DRIFT_FLOOR, top)
            ax.set_yticks(list(DRIFT_TICKS))
            ax.set_yticklabels([f"{v:g}" for v in DRIFT_TICKS])
            ax.set_ylabel("Alert rate / target")
            ax.xaxis.set_major_locator(mdates.MonthLocator())                 # a tick at the start of each month, a label in the middle of it
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: ""))
            ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=16))
            ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, pos: MONTHS[mdates.num2date(v).month - 1]))
            ax.tick_params(axis="x", which="minor", length=0)
            ax.set_title(title, loc="left")
        handles = [Line2D([], [], color=color, ls=style, lw=DRIFT_THICK, label=label) for _, label, color, style in DRIFT_POLICIES]
        handles += [Patch(facecolor=BAND, edgecolor="none", label=f"R within {band_text(band)}"),
                    Line2D([], [], color=GREY, lw=0.6, ls=(0, (4, 3)), label="R = 1")]
        fig.tight_layout(rect=(0, _legend_below(fig, handles, 4, DRIFT_NOTE), 1, 1), h_pad=1.2)
        fig.test_a_drawn = False
        return fig


#: FigContrast: the four quarters in chronological order (2025-Q3 to 2026-Q2), and for each panel (statistic, scale to the unit shown, panel title, x label).
#: There is no pooled row: the pooled AQ3+AQ4 estimate overlaps its two quarters and is not an independent observation.
CONTRAST_ROWS = ("AQ3", "AQ4", "AQ1", "AQ2")
CONTRAST_PANELS = (("E4:dSELC|calendar@count", 100, "(a) SELC contrast", "SELC difference,\npercentage points"),
                   ("E4:dESred|calendar@count", 1, "(b) ESred contrast", "ESred difference,\nUSD"))
#: The quarters whose deployment lies in the 2026-H1 window that the preprint opened: their rows say so.
OPENED_QUARTERS = tuple(a for a in CONTRAST_ROWS if ANALOGS[a].deploy.start >= LEGACY["sealed26H1"].deploy.start)
NOTE_LEFT, NOTE_RIGHT = "← calendar-only\nhigher", "alerts\nhigher →"
NOTE_GAP_PT = 3.0


def _contrast_label(aid: str) -> str:
    return LABEL[aid] + ("\n(opened data)" if aid in OPENED_QUARTERS else "")


def _room_for_notes(ax, left: Annotation, right: Annotation) -> None:
    """Widen the x limits of `ax`, if need be, until the note left of zero and the note right of zero each fit between zero and the edge of the axes
    on their own side (the notes sit under the axis, beside the zero line, and a ESred interval that runs far to the left of zero leaves little room
    on the right)."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width = ax.get_window_extent(renderer).width
    gap = NOTE_GAP_PT * fig.dpi / 72 + 1
    need_left, need_right = left.get_window_extent(renderer).width + gap, right.get_window_extent(renderer).width + gap
    if need_left + need_right >= width:
        raise ValueError(f"the two notes under a panel ({need_left + need_right:.0f} px) do not fit its width ({width:.0f} px)")
    lo, hi = ax.get_xlim()
    zero = -lo / (hi - lo)                                                    # where zero is, as a fraction of the axes
    if zero * width < need_left:
        zero = need_left / width
        lo = -zero * hi / (1 - zero)
    if (1 - zero) * width < need_right:
        zero = 1 - need_right / width
        hi = -lo * (1 - zero) / zero
    ax.set_xlim(lo, hi)


def fig_contrast(ci: pd.DataFrame) -> Figure:
    """FigContrast: paired contrasts with the calendar-only rule at the same alert count (finding 3). Two panels share four rows, the quarters AQ3,
    AQ4, AQ1 and AQ2: the SELC contrast of the alerts minus the calendar-only model in percentage points (left) and the ESred contrast in USD
    (right). One filled marker with its interval per row, a vertical zero line, and a note under each axis on which side is higher. The intervals are
    the day-bootstrap percentile intervals of bootstrap_ci.csv; the two statistics are read with _ci_rows, which refuses a missing row."""
    with _style():
        fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.0), sharey=True)
        y = np.arange(len(CONTRAST_ROWS))
        notes = []
        for ax, (stat, scale, title, xlabel) in zip(axes, CONTRAST_PANELS):
            est, lo, hi = _ci_rows(ci, CONTRAST_ROWS, stat)
            ax.axvline(0, color=TEXT_GREY, lw=0.8, zorder=1)
            _interval(ax, scale * est, scale * lo, scale * hi, y, color=BLACK, marker="o", ms=5, lw=0.9, cap=2.0)
            span = float(np.nanmax(scale * hi) - np.nanmin(scale * lo))
            ax.set_xlim(min(0, float(np.nanmin(scale * lo))) - 0.04 * span, max(0, float(np.nanmax(scale * hi))) + 0.04 * span)
            ax.set_ylim(len(CONTRAST_ROWS) - 0.4, -0.6)                       # AQ3 at the top
            ax.set_title(title, loc="left")
            ax.set_xlabel(xlabel, labelpad=28)                                # room for the two lines of notes between the tick labels and the label
            _plain_money_axis(ax)
            pair = [ax.annotate(text, xy=(0, 0), xycoords=("data", "axes fraction"), xytext=(sign * NOTE_GAP_PT, -22), textcoords="offset points",
                                ha="right" if sign < 0 else "left", va="top", fontsize=SUB_PT, color=TEXT_GREY, linespacing=1.15, annotation_clip=False)
                    for text, sign in ((NOTE_LEFT, -1), (NOTE_RIGHT, 1))]
            for note in pair:
                note.set_in_layout(False)       # the layout must not squeeze the axes for a note that sticks out sideways: _room_for_notes widens the limits instead
            notes.append((ax, *pair))
        axes[0].set_yticks(y)
        axes[0].set_yticklabels([_contrast_label(a) for a in CONTRAST_ROWS])
        level = pct_of(signature_default("bootstrap", "percentile_ci", "level"))
        handles = [Line2D([], [], color=BLACK, marker="o", ms=5, lw=0.9, label=f"estimate and its {level:g} % interval")]
        note = ("Contrast: the alerts minus a calendar-only rule with the same number of alerts at each node\n"
                "Intervals: day bootstrap, approximate, conditional on the realised quarter\n"
                "Exploratory analyses, no row is confirmatory; AQ1 and AQ2 use the opened 2026-H1 data")
        fig.tight_layout(rect=(0, _legend_below(fig, handles, 1, note), 1, 1), w_pad=1.2)
        for ax, left, right in notes:
            _room_for_notes(ax, left, right)
        fig.test_a_drawn = False
        return fig


# ---------------------------------------------------------------- driver
def _caption_notes() -> list[str]:
    """What each figure shows, for the captions in the section files. Numbers come from the code that makes them. The notes are
    written as captions are: the journal wants no punctuation at the end of a figure caption (Req L37), so no note ends with one."""
    band = band_text(BANDS[BAND_KEY])
    level = pct_of(signature_default("bootstrap", "percentile_ci", "level"))
    return [
        f"Fig1: one marker per node, at log10 of R_floored (R = realized alert rate over alpha, floored at half an alert as in the "
        f"audit tables); shaded band: R within {band}; dashed line: R = 1; black bar: median over nodes; an open marker is a node with "
        "no alerts, drawn at the foot of the axis because its floored value lies far below the others",
        f"Fig2: filled markers with bars are estimates with their {level:g} % percentile intervals; the two P1 minus comparator rows "
        "are paired contrasts (SELC of the alerts minus SELC of a comparator that alerts as often, in percentage points; in the right "
        "panel the ESred of the alerts minus the ESred of the comparator, in USD); open markers are the comparators' own SELC and ESred, "
        "without intervals. ESred in USD",
        "Fig3: timing is P1 minus P4 at P1's alert count, level is P4 at P1's alert count minus P4, total is their sum (diamond); bars "
        f"carry {level:g} % percentile intervals; the AQ3+AQ4 row sums two quarters for NET and J and is a day-weighted mean for ESred",
        "Fig2 and Fig3: an axis label says 'approximate interval' where the approximate column of bootstrap_ci.csv flags the statistic",
        f"Fig4: |log10 R_floored| per node; box = quartiles with the median line, whiskers run to the minimum and maximum over nodes "
        f"(no node is hidden); the number above each block counts nodes with R outside {band} (fail_0.5_2). The Test A panel shows "
        "the median over nodes with its bootstrap interval and the same count",
        "Fig1 to Fig4: a Test A that has not run has no panel, row or word in them; one that has run is drawn; one that failed its coverage rule "
        "says 'Test A: did not run'",
        f"FigFail: rows are the six feature blocks of Table 3 and columns its analyses (left, XGBoost with the rolling origin: the deployment years "
        f"2025 and 2026-H1; right, robustness: logistic regression for the same two periods, and XGBoost with the origin frozen at 2026-H1); a cell "
        f"prints the number of the audit's nodes that fail, a failure being a realised alert rate outside {band} times the target (fail_0.5_2); "
        f"the fill grows with the count on one scale for both panels, and a cell of {HEAVY_FROM} or more nodes has bold type and a heavy outline, "
        "so that the figure reads in greyscale. The numbers are the rows of Table 3 (tab_block_audit)",
        f"FigDrift: the realised alert rate over the target R (alerts over node-hours, over the {pct_of(ALPHA):g} % target) of the combined "
        f"block with XGBoost (seed 42), the fixed cutoff P1 against the rolling {P4_WINDOW_DAYS}-day cutoff P4, over trailing windows of "
        f"{_number_word(WINDOW_WEEKS)} Monday-start weeks (the weeks so far from the {_ordinal_word(WINDOW_MIN_WEEKS)} week on, at the start of the "
        "period; partial weeks and weeks with missing hours count with the hours they have, and a node's window with fewer than half its nominal "
        "hours is left out); each point is drawn at the end of its window; the thick line is the median over nodes and the thin lines are its "
        f"quartiles; the axis is log base 2, with a place for zero, the band R within {band} is shaded and R = 1 is dashed. The data are "
        "outputs/es/paper/drift_path.csv (es.run_driftpath), whose node totals are those of Table 3; the windows are summarised in "
        "outputs/es/paper/drift_summary.csv",
        f"FigContrast: for AQ3, AQ4, AQ1 and AQ2 (2025-Q3 to 2026-Q2; there is no pooled row), the paired contrast of the alerts minus a calendar-only "
        f"rule that alerts as often, for SELC in percentage points (left) and ESred in USD (right), with its {level:g} % day-bootstrap percentile "
        "interval (approximate, conditional on the realised quarter); AQ1 and AQ2 use the opened 2026-H1 data; every row is exploratory, and an "
        "interval that contains zero shows a lack of evidence, not equivalence",
    ]


def build_all(src: Sources, figures_dir: Path) -> dict[str, str]:
    _require_arial()
    figures_dir.mkdir(parents=True, exist_ok=True)
    status = {}
    sealed = src.exists("sealed/A/results.json")
    pending = "did not run" if src.did_not_run("A") else "pending"      # the macros say 'test did not run' for the same state
    for name, needs in SOURCES.items():
        missing = [n for n in needs if not src.exists(n)]
        if missing:
            status[name] = "skipped: missing " + ", ".join(missing)
            for ext in ("pdf", "eps"):
                (figures_dir / f"{name}.{ext}").unlink(missing_ok=True)     # an earlier build's file must not pass for this one's
            continue
        test_a = sealed and SEALED_A[name] is not None and src.exists(SEALED_A[name])
        if name == "Fig1":
            fig = fig1_r_drift(r_drift_frame(src), test_a, pending=pending)
        elif name == "Fig2":
            fig = fig2_selc_esred(e1_frame(src), test_a, pending=pending)
        elif name == "Fig3":
            fig = fig3_decomposition(e1_frame(src), test_a, pending=pending)
        elif name == "Fig4":
            act, e5 = blocks_frame(src)
            fig = fig4_blocks(act, e5, test_a and e5 is not None, pending=pending)
        elif name == "FigFail":
            fig = fig_fail(fail_frame(src))
        elif name == "FigDrift":
            fig = fig_drift(drift_frame(src))
        else:
            fig = fig_contrast(e1_frame(src))
        # from what was drawn, not from which files exist; a figure without a Test A panel has nothing to wait for, and while Test A has not run
        # (no results.json) the four older figures have none: they print nothing of Test A and their status is plain "built". A Test A that ran but
        # left this figure's table out, and one that did not run, keep "built (Test A ...)".
        status[name] = "built" if SEALED_A[name] is None or fig.test_a_drawn or not sealed else f"built (Test A {pending})"
        save(fig, name, figures_dir)
    (figures_dir / "INDEX.md").write_text(
        "| figure | sources | status |\n|---|---|---|\n"
        + "".join(f"| {k} | {', '.join(READS[k])} + {SEALED_A[k] or 'no Test A panel'} | {v} |\n" for k, v in status.items())
        + "\nCaption notes (what each figure draws):\n\n" + "".join(f"- {line}\n" for line in _caption_notes()))
    return status


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--paper-dir", default=str(PAPER_DIR))
    a = ap.parse_args(argv)
    status = build_all(Sources(Path(a.out_dir), Path(a.paper_dir), DOCS_SEALED), Path(a.paper_dir) / "generated" / "figures")
    for k, v in status.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
