"""Sealed-test tables (spec v3 section 6; plan 2A Task 2A-4 item 6, the registered table shells).

One builder per registered result and per test T in {A, B}: the family (H-E, H-B1, H-B2), PC and H-S, E1, E2 to E4 (with the
headline SELC and NET), E5; and the tables around them: coverage, the CAISO-free twin of E2 to E4, the activity audit, the
economics rows, the Benjamini-Hochberg families (tab_sealed_bh_T), and for Test B the Holm decision and the estimands of Test B
alone. Every name is tab_sealed_<table>_<T>. They read
what es.sealed_run writes: sealed/<T>/results.json (written last) and sealed/<T>/tables/*.csv.

Four states of a test, told by results.json (Sources.did_not_run is the one place that reads the coverage verdict):
- not run yet (no results.json): the registered results are shells, the cells of the registered protocol with a dash in each
  (the dashes of the protocol's Markdown shells); the other tables are skipped, because their sources are absent;
- run: the table, from the saved output;
- run, with an analysis that failed (controller Ruling 104: every analysis after the data load runs in its own guard, and a
  failure is kept in results["errors"] and as {"error": ...} in the analysis' own slot): the cells of what did not complete
  are dashes, as in a shell, and a footnote says what was not computed and why. Anything that the contract promises and the
  output lacks without such a record (a table file, a key, a column) raises a LookupError that names it;
- did not run (the coverage rule failed): every table prints "Test A did not run (coverage rule)" with the reason (the usable
  nodes against the rule, the operators with none), never a pending sentinel, so that a strict build passes. The coverage table
  also shows who was out.

The family is read, never typed: from results.json (each hypothesis' `confirmatory` flag, checked against the family the run
recorded) and, for a shell, from analogs/protocol_inputs.json. A hypothesis that is an estimand prints its estimate with its
interval and the mark of an approximate interval, no p-value, no Holm decision and no replication verdict (the registered
family rule of the Task 6 brief). Plain-text statuses and messages go through tex_escape; the module writes no number that the
saved outputs or the code can say."""
from __future__ import annotations

import math
from functools import partial

import numpy as np
import pandas as pd

from .blocks import BLOCKS, PAPER_BLOCK_NAMES
from .measures import E1_KEYS
from .paper_macros import PENDING, Sources
from .paper_registry import DESCRIPTIVE_BH_FAMILIES, E1_FMT, HB_LAWS, STAT_FMT
from .paper_tables import (APPROX_MARK, APPROX_NOTE, APPROX_TEXT, BH_FAMILIES, DESCRIPTIVE_NOTE, ECON_POLICY_ORDER, MARK_A, MARKET_NAME, MARKETS,
                           POLICY_NAME, TEXT_WIDTH_PT, WIDTH_MARGIN_PT, _approx, _check_text_width, _code, _fit_widths, _need, _one,
                           _policy_legend, _primary_seed, _range, _rr, _sized, _stack, _tab, builder, ci_cell, stat_rows)
from .paper_tex import FORMATS, PAIR_FORMATS, fmt_g, tabular, tex_escape

#: The tests of spec section 6.3, by name. Which of them is confirmatory is the registration's to say (the family).
HYPOTHESES = ("H-E", *HB_LAWS)
#: The registered results: a shell before the run (plan 2A Task 2A-4 item 6: H-E; H-B1 and H-B2; PC; H-S; E1; E2 to E4; E5).
RESULT_TABLES = ("family", "pc_hs", "e1", "e2e4", "e5")
COLSEP = 3                                                  # \tabcolsep of the sized tables, in points
I, P, D2, D3 = FORMATS["int"], FORMATS["pval"], FORMATS["dec2"], FORMATS["dec3"]
DASH = "--"
PARTS = ("timing", "level", "total")
ORIGIN = {"roll": "rolling", "frozen": "frozen"}
ESTIMATOR = {"xgb": "XGB", "logit": "logit"}
AUDIT_POLICY = {"P2_season": "P2 in-sample (season)", "P2_all": "P2 in-sample (all seasons)"}      # POLICY_NAME has no P2
PC_ROWS = (("P2_season", "PC: P2 (season) $-$ P1, median $|\\log_{10} R|$"), ("P2_all", "PC: P2 (all seasons) $-$ P1"))
E1_COMPONENTS = ("Components of the difference in the value V of the row: timing = V(P1) $-$ V(P4@r*), level = V(P4@r*) $-$ V(P4), "
                 "total = V(P1) $-$ V(P4).")
E4_NOTE = ("E4 estimands are the paired differences SELC(P1) $-$ SELC(comparator) and ESred(P1) $-$ ESred(comparator) at the same alert "
           "count, and the SELC under the decile strata, which is a sensitivity only; the SELC and the ESred of each comparator are context. "
           "ESred (E3) is the tail effect of the policy and not evidence of skill.")
ESTIMAND_COLUMNS = ["stat", "estimate", "lo", "hi", "approximate"]
E5_COLUMNS = ["block", "median_log_error", "lo", "hi", "failures", "failures_lo", "failures_hi", "n_nodes", "approximate"]
#: The names under which es.sealed_run records a failure (results["errors"]), by what a table needs. A name stands for itself and
#: for every analysis under it ("estimands" for "estimands/file"); "stage/..." is a stage that failed as a whole.
FAILS = {
    "estimands": ("estimands", "stage/estimands"), "ex_caiso": ("estimands_ex_caiso", "stage/estimands"),
    "this_quarter": ("estimands_this_quarter", "stage/estimands"), "economics": ("economics", "stage/economics"),
    "audit": ("audit_activity", "stage/audit"), "e5": ("E5", "stage/bootstraps"), "secondary": ("BH tables", "stage/secondary"),
    "decision": ("decision", "stage/decision"),
    "H-E": ("H-E", "decision", "stage/decision"), "H-B1": ("H-B1", "stage/bootstraps", "decision", "stage/decision"),
    "H-B2": ("H-B2", "stage/bootstraps", "decision", "stage/decision"),
    "PC": ("PC", "stage/bootstraps"), "H-S": ("H-S", "stage/bootstraps"),
}
#: The numbers of an entry of results.json that a table prints, by hypothesis.
NUMBERS = {"H-E": ("t_obs", "p", "n_null"), "H-B1": ("t_hat", "p", "ci", "ci_basic"), "H-B2": ("t_hat", "p", "ci", "ci_basic")}
_MISSING = object()


# ---------------------------------------------------------------- where things are, and in which state a test is
def _res(test: str) -> str:
    return f"sealed/{test}/results.json"


def _tbl(test: str, name: str) -> str:
    return f"sealed/{test}/tables/{name}"


def _level() -> str:
    return fmt_g(_code("ciLevel"), 3)


def _get(d, key, where: str):
    if not isinstance(d, dict) or key not in d:
        raise LookupError(f"{where}: missing {key!r}" + (f" (has {sorted(d)})" if isinstance(d, dict) else ""))
    return d[key]


def _finite(value) -> bool:
    """Whether every number in `value` (a number, or a list of them) is finite: a completed analysis can still carry a NaN (no node
    evaluable, say), which no format prints."""
    if isinstance(value, (list, tuple)):
        return all(_finite(v) for v in value)
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _flag(entry: dict, key: str, where: str) -> bool:
    """An `approximate` flag of an entry of results.json: the key must be there (a LookupError otherwise); a null (or NaN) value means
    not flagged."""
    value = _get(entry, key, where)
    return False if value is None or (isinstance(value, float) and math.isnan(value)) else bool(value)


def _short(message, limit: int = 160) -> str:
    """One line of plain ASCII (pdflatex's default encoding cannot typeset anything else), cut at `limit` characters."""
    text = " ".join(str(message).split()).encode("ascii", "replace").decode()
    return text if len(text) <= limit else text[:limit - 3] + "..."


class _Sealed:
    """One test's saved output as a table reads it: results.json and tables/*.csv in the state the run left them.

    A shell (no results.json) answers every question with None. So does a run with an analysis that failed, for that
    analysis only, and it keeps the reason for the footnote of the table (`footnote`). Anything the contract promises and the
    output lacks, with no failure recorded for it, raises a LookupError that names it."""

    def __init__(self, src: Sources, test: str):
        self.src, self.test, self.where = src, test, _res(test)
        self.state = "absent" if not src.exists(self.where) else ("notrun" if src.did_not_run(test) else "ran")
        self.results = src.json(self.where) if self.state != "absent" else {}
        errors = self.results.get("errors")
        self.errors = errors if isinstance(errors, dict) else {}
        self.reasons: dict[str, str] = {}

    def require_run(self) -> "_Sealed":
        """For a table that has no shell: a direct call before the run is an error."""
        if self.state == "absent":
            raise LookupError(f"{self.where} is missing: Test {self.test} has not run")
        return self

    def explained_by(self, names) -> dict[str, str]:
        return {n: m for n, m in self.errors.items() if any(n == x or n.startswith(x + "/") for x in names)}

    def not_computed(self, label: str, message) -> None:
        self.reasons.setdefault(label, _short(message))

    def note_failures(self, names) -> None:
        """The failures recorded under `names`, kept as reasons even though the table has what completed (a partial table)."""
        for name, message in self.explained_by(names).items():
            self.not_computed(name, message)

    def _explained(self, label: str, names, what: str) -> None:
        found = self.explained_by(names)
        if not found:
            raise LookupError(f"{what}, and no failure is recorded for it (results.json errors: {sorted(self.errors) or 'none'})")
        self.not_computed(label, "; ".join(dict.fromkeys(found.values())))

    def entry(self, path, label: str, names=(), numeric=()):
        """results.json at `path`: the entry, or None for a shell, for an error entry ({"error": ...}: the reason is kept) and for
        an entry that a recorded failure explains. An entry that has an "error" has not completed, whatever else it holds. An entry
        whose `numeric` keys hold a number that is not finite (NaN) is not computed either: no format prints it."""
        if self.state == "absent":
            return None
        node = self.results
        for key in path:
            node = node.get(key, _MISSING) if isinstance(node, dict) else _MISSING
            if node is _MISSING:
                break
        if node is _MISSING:
            self._explained(label, names, f"{self.where}: missing {'/'.join(path)!r}")
            return None
        if isinstance(node, dict) and "error" in node:
            self.not_computed(label, node["error"])
            return None
        bad = [k for k in numeric if isinstance(node, dict) and k in node and not _finite(node[k])]
        if bad:
            self.not_computed(label, f"{', '.join(bad)} not finite")
            return None
        return node

    def field(self, entry: dict, key: str, label: str, names=()):
        """entry[key], or None when a recorded failure explains why the entry lacks it (a decision that did not complete)."""
        if key in entry:
            return entry[key]
        self._explained(label, names, f"{self.where}: {label} has no {key!r}")
        return None

    def frame(self, name: str, cols, label: str, names) -> pd.DataFrame | None:
        """tables/<name>: the frame (with the columns the contract names), or None for a shell or for a file that a recorded failure
        explains."""
        if self.state == "absent":
            return None
        rel = _tbl(self.test, name)
        if self.src.exists(rel):
            df = self.src.frame(rel)
            _need(df, cols, rel)
            return df
        self._explained(label, names, f"{rel}: {self.where} is there, but this table is not")
        return None

    def tested(self) -> dict[str, bool]:
        """{hypothesis: confirmatory}: each completed entry's own flag, which must agree with the family the run recorded; an entry that
        did not complete has no flag, so the recorded family says."""
        listed = _get(self.results, "family", self.where)
        out = {}
        for hid in HYPOTHESES:
            slot = self.results.get(hid)
            if isinstance(slot, dict) and "error" not in slot:
                flag = bool(_get(slot, "confirmatory", f"{self.where} {hid}"))
                if flag != (hid in listed):
                    raise LookupError(f"{self.where}: {hid} has confirmatory = {flag}, but the family that the run recorded is {list(listed)}")
            out[hid] = hid in listed
        return out

    def text(self, *others: "_Sealed") -> str:
        reasons = {k: v for s in (self, *others) for k, v in s.reasons.items()}
        return "Not computed: " + "; ".join(f"{k} ({v})" for k, v in reasons.items()) + "." if reasons else ""

    def footnote(self, *others: "_Sealed") -> str:
        text = self.text(*others)
        return _footnote(tex_escape(text)) if text else ""


# ---------------------------------------------------------------- the pieces every table is made of
def _footnote(text: str) -> str:
    return f"\\footnotetext{{{text}}}\n"


def _marked(rows) -> bool:
    return any(APPROX_MARK in cell for row in rows for cell in row)


#: Ruling 118 (G1, M2; coordinator item D13): E5 and H-S are levels of one window, and their interval is the percentile interval of a
#: calendar-aligned calibration-and-deployment bootstrap (one stationary draw over the quarter's calendar weights both windows). The
#: size study evaluated the independent-draw scheme, so every one of these intervals is approximate whatever its NaN count, and the
#: coverage of the aligned scheme is not studied. The note of the mark says so, in the same footnote as the general reason.
ALIGNED_TEXT = ("The intervals of E5 and of H-S come from the calendar-aligned calibration-and-deployment bootstrap, in which one stationary "
                "draw over the quarter's calendar weights the calibration and the deployment days alike (Section~\\ref{esm:size-power}); the size "
                "study evaluated the independent-draw scheme, and the coverage of the aligned scheme is not studied.")
ALIGNED_NOTE = "\\footnotetext{" + MARK_A + " " + APPROX_TEXT + " " + ALIGNED_TEXT + "}\n"


def _approx_note(*row_groups, aligned: bool = False) -> str:
    """The footnote of the approximate mark of a table that has one: the general reason, and for the E5 and H-S rows (``aligned``) where
    their intervals come from as well."""
    if not any(_marked(rows) for rows in row_groups):
        return ""
    return ALIGNED_NOTE if aligned else APPROX_NOTE


def _width(header: list[str], rows: list[list[str]], fixed=None) -> float:
    """The width in points of the table with these cells in a sized group (the glyph widths of the guard are those of \\footnotesize)."""
    return sum(_fit_widths(header, rows, fixed or {})) + 2 * COLSEP * len(header)


def _fits(header: list[str], rows: list[list[str]], fixed=None) -> bool:
    return _width(header, rows, fixed) <= TEXT_WIDTH_PT - WIDTH_MARGIN_PT


def _fit(name: str, kinds: str, header: list[str], rows: list[list[str]], fixed=None, midrules_before=(), title: str | None = None) -> str:
    """The tabular in a footnotesize group with a tight column separation, after the guard of the other tables has checked that it fits
    the text width (a ValueError otherwise). ``kinds`` has a letter for each column: l, c or r (as wide as the widest cell) or p
    (a paragraph column of the width ``fixed`` gives for its index). ``title`` is a heading row inside the tabular, for a table of
    several panels: the class puts the caption above the first tabular, so a heading outside it would come before the caption."""
    fixed = fixed or {}
    if len(kinds) != len(header):
        raise ValueError(f"{name}: {len(kinds)} column kinds for {len(header)} headings")
    widths = _fit_widths(header, rows, fixed)
    _check_text_width(name, [(header, rows, widths)], COLSEP)
    colspec = "".join(f"p{{{widths[j]:.1f}pt}}" if kind == "p" else kind for j, kind in enumerate(kinds))
    groups = [(f"\\textit{{{title}}}", len(header))] if title else None
    return _sized(_tab(colspec, header, rows, midrules_before, groups), colsep=COLSEP)


def _fit_last_wrapping(name: str, kinds: str, header: list[str], rows: list[list[str]], title: str | None = None) -> str:
    """_fit with every column as wide as its widest cell, while that fits the text width (a wording on one line is never stretched by
    justification); when it does not (a longer wording than the pre-specified ones), the last column is a ragged paragraph column of the
    width that the others leave."""
    if _fits(header, rows):
        return _fit(name, kinds, header, rows, title=title)
    others = sum(_fit_widths(header[:-1], [r[:-1] for r in rows], {}))
    last = int((TEXT_WIDTH_PT - WIDTH_MARGIN_PT - 2 * COLSEP * len(header) - others) * 10) / 10 - 1.0
    return _fit(name, kinds[:-1] + "p", header, [r[:-1] + [_rr(r[-1])] for r in rows], {len(header) - 1: last}, title=title)


def _statement(text: str) -> str:
    """A table of one cell: a statement with no number and no sentinel (a test that did not run, an analysis that did not complete)."""
    return "\\begin{tabular}{@{}p{\\textwidth}@{}}\n\\toprule\n" + tex_escape(text) + " \\\\\n\\bottomrule\n\\end{tabular}\n"


def _not_run_text(s: _Sealed) -> str:
    """"Test A did not run (coverage rule): 20 usable nodes, 28 required; no usable node for CAISO." From results.json (the usable
    nodes) and coverage.csv (the operators with none), as far as they are there; the rule's minimum is the registry's covMinNodes."""
    coverage = s.results.get("coverage") or {}
    parts = []
    if "usable_nodes" in coverage:
        parts.append(f"{I(coverage['usable_nodes'])} usable nodes, {I(_code('covMinNodes'))} required")
    if s.src.exists(_tbl(s.test, "coverage.csv")):
        c = s.src.frame(_tbl(s.test, "coverage.csv"))
        if {"market", "usable"} <= set(c.columns):
            none = [m for m in MARKETS if not c[c.market == m].usable.astype(bool).any()]
            if none:
                parts.append("no usable node for " + ", ".join(MARKET_NAME[m] for m in none))
    return f"Test {s.test} did not run (coverage rule)" + (": " + "; ".join(parts) if parts else "") + "."


def _unavailable(s: _Sealed) -> str:
    """The statement of a table that has no shell and whose source did not complete."""
    return _statement(s.text())


# ---------------------------------------------------------------- the registered family
def _registered_family(src: Sources):
    """The family of analogs/protocol_inputs.json once the simulation stage has written it (stage complete), else None: the shell
    then waits for the designated status (a sentinel) instead of guessing."""
    rel = "analogs/protocol_inputs.json"
    if not src.exists(rel):
        return None
    inputs = src.json(rel)
    family = inputs.get("family")
    if inputs.get("stage") != "complete" or not isinstance(family, dict):
        return None
    bad = [h for h in HYPOTHESES if not isinstance(family.get(h), dict) or "confirmatory" not in family[h]]
    if bad:
        raise LookupError(f"{rel}: the family has no registered status for {bad} (it has {sorted(family)})")
    return family


def _hypothesis_label(hid: str) -> str:
    if hid == "H-E":
        return "H-E: timing within calendar cells"
    base, alt = HB_LAWS[hid]
    return f"{hid}: {PAPER_BLOCK_NAMES[alt]} vs {PAPER_BLOCK_NAMES[base]}"


def _hb_interval(h: dict, where: str) -> str:
    """The percentile bootstrap interval of an H-B entry: the primary interval of the estimands (Ruling 110). The basic interval reflects
    the bootstrap's shift about t-hat and is a diagnostic, kept in a note (_interval_note). Marked when it is approximate: the registered
    rule said so, or more than 1% of the replicates were NaN (es.sealed_run folds both into `ci_approximate`)."""
    return _approx(PAIR_FORMATS["ci3"](*_get(h, "ci", where)), _flag(h, "ci_approximate", where))


def _interval_note(ids: list[str], basic: list[str], approximate: bool, tested: bool) -> str:
    """The note of a table with H-B intervals (Ruling 110; Ruling 118, M7): they are percentile bootstrap intervals from independent
    draws of the calibration and the deployment days; the basic intervals (`basic`, one text each) are shown for comparison. The shift
    of the bootstrap distribution about the estimate that the basic interval reflects comes mainly from those independent draws, which
    re-randomize the calendar composition that the two windows share by design. Both intervals are approximate and may under-cover
    substantially under a common drift when the rule labelled them so; the p-value of a tested H-B hypothesis is the basic bootstrap's."""
    text = (f"The intervals of {' and '.join(ids)} are percentile bootstrap intervals from independent draws of the calibration and the "
            f"deployment days. The basic intervals, {', '.join(basic)}, reflect the shift of the bootstrap distribution about $\\hat t$, "
            "which comes mainly from re-randomizing the calendar composition that the two windows share by design.")
    if approximate:
        text += " Both intervals are approximate and may under-cover substantially under a common drift."
    if tested:
        text += " The p-value of a tested H-B hypothesis is the basic bootstrap p-value."
    return text


def _hb_cell(h: dict, where: str) -> str:
    """'estimate [lo, hi]' of an H-B entry."""
    return f"{D3(_get(h, 't_hat', where))} {_hb_interval(h, where)}"


#: Ruling 133: the words of the printed text. The sealed run (a registered file, not edited) writes "registration rule" and "registered
#: pooled test" into the statuses and wordings of results.json; the manuscript calls them the designation rule and the pre-specified
#: pooled test, so the tables print those words. The file keeps what the run wrote.
PRINTED_WORDS = (("registration rule", "designation rule"), ("registered pooled test", "pre-specified pooled test"))


def _printed_words(text):
    """``text`` of a status or a wording of results.json with the words of the manuscript (PRINTED_WORDS); None and a dash pass through."""
    if not isinstance(text, str):
        return text
    for written, printed in PRINTED_WORDS:
        text = text.replace(written, printed)
    return text


def _shown_status(status, confirmatory: bool):
    """The status of a hypothesis as a table prints it (Ruling 113): a confirmatory hypothesis keeps the whole status that the run
    wrote; an estimand shows its label, not the reasons of the designation rule that follow it after ": " (the section on the
    designation rule gives them). The words are the manuscript's (_printed_words). A status that is not there (None) or is a dash
    passes through."""
    return _printed_words(status if status is None or confirmatory else status.partition(": ")[0])


def _selc_parts(row: pd.Series) -> tuple[str, str]:
    pf, cf = STAT_FMT["SELC"]
    return FORMATS[pf](row.estimate), _approx(PAIR_FORMATS[cf](row.lo, row.hi), row.approximate)


def _selc_row(s: _Sealed) -> pd.Series | None:
    """The SELC row of the estimands, H-E's estimand (None where the estimands did not complete)."""
    e = s.frame("estimands.csv", ESTIMAND_COLUMNS, "estimands", FAILS["estimands"])
    if e is None:
        return None
    row = _one(e[e.stat == "SELC"], f"{_tbl(s.test, 'estimands.csv')} SELC")
    if not _finite([float(row.estimate), float(row.lo), float(row.hi)]):
        s.not_computed("SELC", "estimate or interval not finite")
        return None
    return row


def _retained(shares, test: str) -> str:
    """Share of alerts retained by the rotations (min / mean over the shifts), per window (Test B pools two)."""
    def one(share):
        lo, mean = share.get("min"), share.get("mean")
        return DASH if pd.isna(lo) or pd.isna(mean) else f"{FORMATS['pct0'](lo)} / {FORMATS['pct0'](mean)}"
    if not shares:
        return DASH
    if len(shares) == 1:
        return one(shares[0])
    names = ["Test A", "Test B"] if test == "B" and len(shares) == 2 else [f"window {i}" for i in range(1, len(shares) + 1)]
    return ", ".join(f"{n}: {one(x)}" for n, x in zip(names, shares))


def _family(src: Sources, test: str) -> str:
    s = _Sealed(src, test)
    if s.state == "notrun":
        return _statement(_not_run_text(s))
    plain = PAPER_BLOCK_NAMES[HB_LAWS["H-B1"][0]]
    header = ["hypothesis", "designated as", "estimate", f"{_level()}\\% CI", "$p$", "status"]
    rows, notes = [], [f"Estimates: the pooled SELC for H-E; for H-B1 and H-B2 the median over nodes of the paired difference in $|\\log_{{10}} R|$ "
                       f"between the block and {plain} (positive: the block has the larger operating-point error)."]
    if s.state == "ran":
        tested = s.tested()
        selc = _selc_row(s)
        entries = {hid: s.entry((hid,), hid, FAILS[hid], NUMBERS[hid]) for hid in HYPOTHESES}
        for hid in HYPOTHESES:
            h, where, conf = entries[hid], f"{s.where} {hid}", tested[hid]
            if hid == "H-E":
                estimate, ci = _selc_parts(selc) if selc is not None else (DASH, DASH)
                p = _approx(P(_get(h, "p", where)), _flag(h, "approximate", where)) if conf and h is not None else DASH
            else:
                estimate, ci = (D3(_get(h, "t_hat", where)), _hb_interval(h, where)) if h is not None else (DASH, DASH)
                p = P(_get(h, "p", where)) if conf and h is not None else DASH
            status = _shown_status(s.field(h, "status", hid, FAILS["decision"]) if h is not None else None, conf)
            rows.append([_rr(_hypothesis_label(hid)), FORMATS["registered"](conf), estimate, ci, p,
                         DASH if status is None else _rr(tex_escape(status))])
        he = entries["H-E"]
        if tested["H-E"] and he is not None:
            where = f"{s.where} H-E"
            notes.append(f"H-E: studentized statistic $T$ = {D2(_get(he, 't_obs', where))}; {I(_get(he, 'n_null', where))} non-identity "
                         f"rotations; share of alerts retained by the rotations (min / mean over the shifts): "
                         f"{_retained(_get(he, 'retained_share', where), test)}.")
        shown = [hid for hid in HB_LAWS if entries[hid] is not None]
        if shown:
            basic = [f"{hid} {PAIR_FORMATS['ci3'](*_get(entries[hid], 'ci_basic', f'{s.where} {hid}'))}" for hid in shown]
            notes.append(_interval_note(shown, basic, any(_flag(entries[hid], "ci_approximate", f"{s.where} {hid}") for hid in shown),
                                        any(tested[hid] for hid in shown)))
        if test == "B":
            notes.append("H-E is the test on the pooled windows of Tests A and B; the estimates of H-B1 and H-B2 are those of Test B alone.")
    else:
        family = _registered_family(src)
        for hid in HYPOTHESES:
            entry = family[hid] if family else None
            flag = bool(entry.get("interim_approximate" if test == "A" else "approximate")) if entry and hid == "H-E" else False
            ci_flag = bool(entry.get("approximate")) if entry and hid != "H-E" else False
            registered = FORMATS["registered"](entry["confirmatory"]) if entry else DASH
            rows.append([_rr(_hypothesis_label(hid)), registered, DASH, _approx(DASH, ci_flag), _approx(DASH, flag), DASH])
        if family is None:                                       # the designated status is not there: a sentinel (wider than a cell), which strict refuses
            notes.append(f"Designated status of each hypothesis: {PENDING['dev']}.")
    table = _fit(f"tab_sealed_family_{test}", "plrcrp", header, rows, {0: 86.0, 5: 88.0})
    return table + "".join(_footnote(n) for n in notes) + s.footnote() + _approx_note(rows)


# ---------------------------------------------------------------- PC and H-S
def _pc_hs(src: Sources, test: str) -> str:
    s = _Sealed(src, test)
    if s.state == "notrun":
        return _statement(_not_run_text(s))
    plain = PAPER_BLOCK_NAMES[HB_LAWS["H-B1"][0]]
    rows = []
    for key, label in PC_ROWS:
        d = s.entry(("PC", key), f"PC {key}", FAILS["PC"], ("median_diff", "ci"))
        where = f"{s.where} PC {key}"
        cells = [DASH, DASH, DASH] if d is None else [
            D3(_get(d, "median_diff", where)), _approx(PAIR_FORMATS["ci3"](*_get(d, "ci", where)), _flag(d, "ci_approximate", where)),
            I(_get(d, "n_nodes", where))]
        rows.append([_rr(label), *cells])
    for key, label in (("xgb", f"H-S: failures of {plain}, XGBoost"), ("logit", f"H-S: failures of {plain}, logistic")):
        d = s.entry(("H-S", key), f"H-S {key}", FAILS["H-S"], ("failures", "ci"))
        where = f"{s.where} H-S {key}"
        cells = [DASH, DASH, DASH] if d is None else [
            I(_get(d, "failures", where)), _approx(PAIR_FORMATS["ci1"](*_get(d, "ci", where)), _flag(d, "approximate", where)), I(_get(d, "n_nodes", where))]
        rows.append([_rr(label), *cells])
    header = ["descriptive quantity", "estimate", f"{_level()}\\% CI", "nodes"]
    aligned = _marked(rows[len(PC_ROWS):])                      # the H-S rows (after the PC rows) come from the calendar-aligned bootstrap
    return _fit(f"tab_sealed_pc_hs_{test}", "prcr", header, rows, {0: 150.0}) + s.footnote() + _approx_note(rows, aligned=aligned)


# ---------------------------------------------------------------- E1 to E5
def _e1_label(key: str) -> str:
    return key if not key.startswith("J_") else f"$J_{{{float(key[2:]):g}}}$"


def _scope_note(s: _Sealed) -> str:
    """What the estimands cover: results.json's estimands_scope ("single window (interim)" in Test A, "pooled A+B" in Test B)."""
    scope = s.results.get("estimands_scope") if s.state == "ran" else None
    return _footnote(f"Scope of the estimands: {tex_escape(scope)}.") if scope else ""


def _estimand_frame(s: _Sealed, name: str, label: str, names) -> pd.DataFrame:
    """The estimand rows of a file; a row whose estimate or end point is not finite (estimand_cis cannot give one in practice) is left
    out, so that it prints a dash, and named in the footnote."""
    e = s.frame(name, ESTIMAND_COLUMNS, label, names)
    if e is None:
        return pd.DataFrame(columns=ESTIMAND_COLUMNS)
    finite = np.isfinite(e[["estimate", "lo", "hi"]].astype(float)).all(axis=1)
    for stat in e.loc[~finite, "stat"]:
        s.not_computed(f"{label} {stat}", "estimate or interval not finite")
    return e[finite]


def _e1(src: Sources, test: str) -> str:
    """E1, the decomposition of the fixed against the rolling cutoff: a row per measure and a column per component while that fits
    the text width (a quarter's values do), otherwise a row per measure and component (the pooled values of Test B)."""
    s = _Sealed(src, test)
    if s.state == "notrun":
        return _statement(_not_run_text(s))
    e = _estimand_frame(s, "estimands.csv", "estimands", FAILS["estimands"])

    def cell(key: str, part: str) -> str:
        g = e[e.stat == f"E1:{key}:{part}"]
        if g.empty:
            return DASH                                                   # a stat the file does not hold prints a dash, as in stat_rows
        r = _one(g, f"E1:{key}:{part}")
        return ci_cell(*E1_FMT[key], r.estimate, r.lo, r.hi, r.approximate)
    name = f"tab_sealed_e1_{test}"
    wide = [[_e1_label(k), *(cell(k, p) for p in PARTS)] for k in E1_KEYS]
    header = ["E1 value", "timing: V(P1) $-$ V(P4@r*)", "level: V(P4@r*) $-$ V(P4)", "total"]
    if _fits(header, wide):
        table, rows, extra = _fit(name, "lccc", header, wide), wide, ""
    else:
        rows = [[_e1_label(k) if j == 0 else "", part, cell(k, part)] for k in E1_KEYS for j, part in enumerate(PARTS)]
        table = _fit(name, "llc", ["E1 value", "component", f"estimate [{_level()}\\% CI]"], rows,
                     midrules_before={len(PARTS) * i for i in range(1, len(E1_KEYS))})
        extra = _footnote(E1_COMPONENTS)
    return table + extra + _scope_note(s) + s.footnote() + _approx_note(rows)


def _stat_table(s: _Sealed, name: str, csv: str, heading: str, label: str, names, scope: str) -> str:
    """One row per stat of the registry (stat_rows): the estimate with its interval, or a dash where the file has no such row."""
    rows = stat_rows(_estimand_frame(s, csv, label, names))
    table = _fit(name, "lc", [heading, f"estimate [{_level()}\\% CI]"], rows)
    return table + _footnote(E4_NOTE) + scope + s.footnote() + _approx_note(rows)


def _e2e4(src: Sources, test: str) -> str:
    s = _Sealed(src, test)
    if s.state == "notrun":
        return _statement(_not_run_text(s))
    return _stat_table(s, f"tab_sealed_e2e4_{test}", "estimands.csv", "estimand", "estimands", FAILS["estimands"], _scope_note(s))


def _ex_caiso(src: Sources, test: str) -> str:
    """The CAISO-free twin of E2 to E4 (spec section 4.1: results are also reported without CAISO)."""
    s = _Sealed(src, test).require_run()
    if s.state == "notrun":
        return _statement(_not_run_text(s))
    return _stat_table(s, f"tab_sealed_ex_caiso_{test}", "estimands_ex_caiso.csv", "estimand, without CAISO", "estimands_ex_caiso",
                       FAILS["ex_caiso"], _scope_note(s))


def _this_quarter(src: Sources, test: str) -> str:
    s = _Sealed(src, test).require_run()
    if s.state == "notrun":
        return _statement(_not_run_text(s))
    return _stat_table(s, f"tab_sealed_this_quarter_{test}", "estimands_this_quarter.csv", "estimand, Test B alone", "estimands_this_quarter",
                       FAILS["this_quarter"], _footnote("Scope of the estimands: Test B alone."))


def _e5(src: Sources, test: str) -> str:
    """E5, the block table, from E5_blocks.csv: a block whose bootstrap failed has an `error` instead of its numbers (a dash row and a
    footnote), a block whose intervals rest on more than 1% NaN replicates is marked approximate."""
    s = _Sealed(src, test)
    if s.state == "notrun":
        return _statement(_not_run_text(s))
    e = s.frame("E5_blocks.csv", ["block"], "E5", FAILS["e5"])
    rows, shifted = [], []
    for b in BLOCKS:
        cells = [DASH, DASH, DASH]
        if e is not None:
            found = e[e.block == b]
            if found.empty:
                raise LookupError(f"{_tbl(test, 'E5_blocks.csv')}: no row for the block {b!r}")
            x = _one(found, f"E5_blocks.csv {b}")
            error = x["error"] if "error" in e.columns else None
            if error is not None and not pd.isna(error):
                s.not_computed(f"E5 {b}", error)
            elif not _finite([float(x[c]) for c in E5_COLUMNS[1:-1] if c in e.columns]):
                s.not_computed(f"E5 {b}", "an estimate or an interval end is not finite")
            else:
                _need(e, E5_COLUMNS, _tbl(test, "E5_blocks.csv"))
                flag = x["approximate"]
                cells = [_approx(f"{D2(x['median_log_error'])} {PAIR_FORMATS['ci2'](x['lo'], x['hi'])}", flag),
                         _approx(f"{I(x['failures'])} {PAIR_FORMATS['ci1'](x['failures_lo'], x['failures_hi'])}", flag), I(x["n_nodes"])]
                if not (x["lo"] <= x["median_log_error"] <= x["hi"] and x["failures_lo"] <= x["failures"] <= x["failures_hi"]):
                    shifted.append(PAPER_BLOCK_NAMES[b])                              # an interval that misses its own estimate
        rows.append([PAPER_BLOCK_NAMES[b], I(len(BLOCKS[b])), *cells])
    header = [_stack("block", "(added to calendar)", align="l"), "dim", _stack("median", "$|\\log_{10} R|$ [CI]"), _stack("failures", "[CI]"), "nodes"]
    names = shifted[0] if len(shifted) == 1 else ", ".join(shifted[:-1]) + " and " + shifted[-1] if shifted else ""
    note = (_footnote(f"For {names}, an interval does not contain its estimate: the bootstrap distribution is shifted relative to the "
                      "estimate, and the interval is approximate (see Section~\\ref{esm:size-power}). The intervals of the independent-draw "
                      "scheme are saved beside these as a diagnostic.") if shifted else "")
    return _fit(f"tab_sealed_e5_{test}", "lrccr", header, rows) + note + s.footnote() + _approx_note(rows, aligned=True)


# ---------------------------------------------------------------- coverage, the audit, the economics, the BH families
def _why_excluded(r) -> str:
    """Why a node is out: its deployment hours fell short of the rule ("coverage": `ok` is False) and/or es.scores.feasibility says no."""
    reasons = []
    if not bool(r.ok):
        reasons.append("coverage")
    if isinstance(r.reason, str) and r.reason not in ("ok", "coverage"):
        reasons.append(r.reason)
    return ", ".join(tex_escape(x) for x in reasons or [str(r.reason)])


def _coverage(src: Sources, test: str) -> str:
    s = _Sealed(src, test).require_run()
    c = s.frame("coverage.csv", ["market", "node", "share", "ok", "usable", "reason"], "coverage", ())      # written before the verdict: always there
    rows = []
    for m in MARKETS:
        g = c[c.market == m]
        usable = g.usable.astype(bool)
        out = ", ".join(f"{tex_escape(r.node)} ({FORMATS['pct0'](r.share)}, {_why_excluded(r)})" for r in g[~usable].itertuples())
        rows.append([MARKET_NAME[m], I(len(g)), I(int(usable.sum())), out or DASH])
    table = _fit(f"tab_sealed_coverage_{test}", "lrrp", ["operator", "nodes", "usable", "excluded (coverage, reason)"], rows, {3: 0.45 * TEXT_WIDTH_PT})
    return table + (_footnote(tex_escape(_not_run_text(s))) if s.state == "notrun" else "")


def _audit(src: Sources, test: str) -> str:
    """The activity audit as the run wrote it, in the run's order: per origin, block, estimator and policy the failures of the cutoff
    target (R outside 0.5 to 2) over nodes, the nodes and the medians; the seeds other than the primary give the range of failures.
    A part of the audit that failed is left out of the file and named in the footnote."""
    s = _Sealed(src, test).require_run()
    if s.state == "notrun":
        return _statement(_not_run_text(s))
    a = s.frame("audit_activity.csv", ["maintenance", "block", "estimator", "seed", "policy", "fail_0.5_2", "R", "log_error"],
                "activity audit", FAILS["audit"])
    if a is None:
        return _unavailable(s)
    s.note_failures(FAILS["audit"])
    seed = _primary_seed()
    keys = ["maintenance", "block", "estimator", "policy"]
    g = a[a.seed == seed].groupby(keys, sort=False).agg(failures=("fail_0.5_2", "sum"), n=("R", "size"), med_r=("R", "median"),
                                                        mle=("log_error", "median")).reset_index()
    block = lambda b: PAPER_BLOCK_NAMES.get(b, tex_escape(b))             # noqa: E731
    policy = lambda p: POLICY_NAME.get(p) or AUDIT_POLICY.get(p) or tex_escape(p)       # noqa: E731
    rows = [[ORIGIN.get(r.maintenance, tex_escape(r.maintenance)), _rr(block(r.block)), ESTIMATOR.get(r.estimator, tex_escape(r.estimator)),
             _rr(policy(r.policy)), I(r.failures), I(r.n), D2(r.med_r), D2(r.mle)] for r in g.itertuples()]
    per_seed = a[a.seed != seed].groupby(keys + ["seed"], sort=False)["fail_0.5_2"].sum().reset_index()
    spread = per_seed.groupby(keys, sort=False)["fail_0.5_2"].agg(["min", "max"]).reset_index()
    for r in spread.itertuples():
        rows.append([ORIGIN.get(r.maintenance, tex_escape(r.maintenance)), _rr(block(r.block)), ESTIMATOR.get(r.estimator, tex_escape(r.estimator)),
                     _rr(f"{r.policy}, other seeds (min--max)"), _range(r.min, r.max, I), DASH, DASH, DASH])
    header = ["origin", "block", "estimator", "policy", "failures", "nodes", _stack("median", "$R$"), _stack("median", "$|\\log_{10} R|$")]
    return _fit(f"tab_sealed_audit_{test}", "lplprrrr", header, rows, {1: 62.0, 3: 84.0}) + s.footnote()


def _econ(src: Sources, test: str) -> str:
    s = _Sealed(src, test).require_run()
    if s.state == "notrun":
        return _statement(_not_run_text(s))
    e = s.frame("economics.csv", ["policy", "comparator", "position", "fees", "SELC", "NET", "SNET", "ESred", "CAP", "n_days"],
                "economics", FAILS["economics"])
    if e is None:
        return _unavailable(s)
    policies = tuple(p for p in ECON_POLICY_ORDER if p != "comparator")
    rows = []
    for pol in policies:
        g = e[(e.policy == pol) & (e.position == "premium") & (e.fees == "calibrated") & (e.comparator == "none")]
        if g.empty:
            continue
        r = _one(g, f"economics.csv {pol}")
        rows.append([pol, FORMATS["usd0"](r.NET), FORMATS["pct1"](r.SELC), FORMATS["usd2"](r.SNET), FORMATS["usd0"](r.ESred), FORMATS["pct1"](r.CAP),
                     I(r.n_days)])
    header = [_stack("cutoff", "policy", align="l"), "NET", "SELC", _stack("SNET", "(\\$/MW-day)"), "ESred", "CAP", "days"]
    return _fit(f"tab_sealed_econ_{test}", "lrrrrrr", header, rows) + _footnote(_policy_legend(policies))


def _bh(src: Sources, test: str) -> str:
    """Benjamini-Hochberg within the three exploratory families (Ruling 37): tab_sealed_bh_T, the sealed twin of tab_bh_families (its name
    keeps the tab_sealed prefix that the name filters of the other tests rely on, Ruling 107). `included` is False
    where the p-value is NaN, a test that did not run (es.secondary.bh_table), so the tests of a family are the sum of `included`. A
    family that failed is results["secondary"][name] = {"error": ...} and has no row in the file: a dash row and a footnote."""
    s = _Sealed(src, test).require_run()
    if s.state == "notrun":
        return _statement(_not_run_text(s))
    families = {fam: s.entry(("secondary", fam), fam, FAILS["secondary"]) for fam in BH_FAMILIES}      # {"tests", "rows", "bh_rejections"} or None
    b = s.frame("secondary_bh.csv", ["family", "test", "estimate", "p", "bh_reject"], "BH tables", FAILS["secondary"]) \
        if any(f is not None for f in families.values()) else None
    if b is None:                                          # no family completed, or their table failed: nothing to show
        return _unavailable(s)
    b = b if "included" in b.columns else b.assign(included=True)
    rel, rows, descriptive = _tbl(test, "secondary_bh.csv"), [], False
    for fam in BH_FAMILIES:
        counts = families[fam]
        if counts is None:                                 # the family failed: it is not in the file and not in the adjustment
            rows.append([tex_escape(fam), DASH, DASH, DASH])
            continue
        g = b[(b.family == fam) & b.included.astype(bool)]
        tests, rejected = len(g), int(g.bh_reject.astype(bool).sum())
        where = f"{s.where} secondary {fam}"
        recorded = (int(_get(counts, "tests", where)), int(_get(counts, "bh_rejections", where)))
        if (tests, rejected) != recorded:                  # the table and the macros bhTests<Family>, bhRejections<Family> read the same run
            raise LookupError(f"{rel}: the family {fam!r} has {tests} tests and {rejected} rejections, but results.json records {recorded[0]} "
                              f"and {recorded[1]}")
        if fam in DESCRIPTIVE_BH_FAMILIES:                 # Ruling 118, M5: descriptive. The tests are counted, the rejections are not reported
            rows.append([tex_escape(fam), I(tests), DASH, DASH])
            descriptive = True
        else:
            rows.append([tex_escape(fam), I(tests), I(rejected), DASH if g.empty else P(g.p.min())])
    q = fmt_g(_code("bhLevel"), 3)
    return (tabular("lrrr", ["exploratory family", "tests", f"BH rejections ($q$ = {q})", "smallest $p$"], rows) + s.footnote()
            + (_footnote(DESCRIPTIVE_NOTE) if descriptive else ""))


# ---------------------------------------------------------------- Test B: the decision
def _same_sign(a: float, b: float) -> bool:
    return a != 0 and b != 0 and (a > 0) == (b > 0)


def _yn(value) -> str:
    """yes or no, or a dash for a verdict that was not computed (None) or is not one (a dash already)."""
    return DASH if value is None or (isinstance(value, str) and value == DASH) else FORMATS["yesno"](value)


def _decision(src: Sources, test: str = "B") -> str:
    """Test B against Test A. The hypotheses that the designation rule made tests are in the Holm family: their p-values, the Holm
    rejection, the replication verdict (H-B) and the pre-specified wording. The others are estimands: both quarters' estimates with their
    intervals, whether the signs agree, and the label, with no Holm decision and no replication verdict. What did not complete is a
    dash and a footnote."""
    sb, sa = _Sealed(src, "B").require_run(), _Sealed(src, "A").require_run()
    if sb.state == "notrun":
        return _statement(_not_run_text(sb))
    tested = sb.tested()
    holm = sb.entry(("holm",), "holm", FAILS["decision"])
    family_rows, estimand_rows, notes = [], [], []
    basic_ids, basic_texts, basic_approximate = [], [], False

    def verdict(hid: str) -> str:
        return DASH if holm is None else _yn(_get(holm, hid, f"{sb.where} holm"))

    def status_of(hid: str, hb):
        """The status cell of an H-B row: a dash where the entry did not complete, else the run's status as `_shown_status` has it."""
        return DASH if hb is None else _shown_status(sb.field(hb, "status", hid, FAILS["decision"]), tested[hid])

    for hid in HYPOTHESES:
        hb = sb.entry((hid,), hid, FAILS[hid], NUMBERS[hid])
        ha = sa.entry((hid,), f"Test A {hid}", FAILS[hid], NUMBERS[hid]) if hid != "H-E" else None        # H-E's Test A estimate is its SELC row, not its entry
        wb, wa = f"{sb.where} {hid}", f"{sa.where} {hid}"
        if tested[hid] and hid == "H-E":
            wording = DASH if hb is None else sb.field(hb, "wording", hid, FAILS["decision"])
            family_rows.append([hid, DASH if hb is None else _approx(P(_get(hb, "p", wb)), _flag(hb, "approximate", wb)), verdict(hid), DASH,
                                DASH if wording is None else tex_escape(_printed_words(wording))])
        elif tested[hid]:
            both = hb is not None and ha is not None
            replicated = DASH if hb is None else sb.field(hb, "replicated", hid, FAILS["decision"])
            status = status_of(hid, hb)
            p = f"A: {P(_get(ha, 'p', wa))}; B: {P(_get(hb, 'p', wb))}" if both else DASH
            family_rows.append([hid, _approx(p, _flag(hb, "ci_approximate", wb)) if both else DASH, verdict(hid), _yn(replicated),
                                DASH if status is None else tex_escape(status)])
        elif hid == "H-E":
            cells = []
            for t, reader in (("A", sa), ("B", sb)):
                selc = _selc_row(reader)
                cells.append(DASH if selc is None else ci_cell(*STAT_FMT["SELC"], selc.estimate, selc.lo, selc.hi, selc.approximate))
            wording = DASH if hb is None else sb.field(hb, "wording", hid, FAILS["decision"])
            estimand_rows.append([hid, *cells, DASH, DASH if wording is None else _rr(tex_escape(_printed_words(wording)))])
            notes.append("H-E: the Test B column is the pooled estimate of Tests A and B.")
        else:
            status = status_of(hid, hb)
            agree = DASH if hb is None or ha is None else _yn(_same_sign(_get(ha, "t_hat", wa), _get(hb, "t_hat", wb)))
            estimand_rows.append([hid, DASH if ha is None else _hb_cell(ha, wa), DASH if hb is None else _hb_cell(hb, wb), agree,
                                  DASH if status is None else _rr(tex_escape(status))])
            if ha is not None and hb is not None:                                 # the basic intervals of both quarters, for the note
                basic_ids.append(hid)
                basic_texts.append(f"{hid} {PAIR_FORMATS['ci3'](*_get(ha, 'ci_basic', wa))} in Test A and "
                                   f"{PAIR_FORMATS['ci3'](*_get(hb, 'ci_basic', wb))} in Test B")
                basic_approximate = basic_approximate or _flag(ha, "ci_approximate", wa) or _flag(hb, "ci_approximate", wb)
    level, two = _level(), bool(family_rows) and bool(estimand_rows)
    panels = []
    if family_rows:                    # the wording column is as wide as its widest wording: a wrapped, justified wording would be stretched
        header = ["hypothesis", "$p$", _stack("Holm", "rejects"), _stack("replicated", "in B"), "pre-specified wording"]
        panels.append(_fit_last_wrapping("tab_sealed_decision_B", "lllll", header, family_rows, title="Tests in the Holm family" if two else None))
    if estimand_rows:
        header = ["hypothesis", _stack("Test A:", f"estimate [{level}\\% CI]"), _stack("Test B:", f"estimate [{level}\\% CI]"), "same sign", "designated status"]
        panels.append(_fit("tab_sealed_decision_B:estimands", "lcccp", header, estimand_rows, {4: 96.0},
                           title="Estimands: an interval, no test" if two else None))
    if basic_ids:
        notes.append(_interval_note(basic_ids, basic_texts, basic_approximate, False))
    return ("\\medskip\n".join(panels) + "".join(_footnote(n) for n in dict.fromkeys(notes)) + sb.footnote(sa)
            + _approx_note(family_rows, estimand_rows))


# ---------------------------------------------------------------- registration
_NOTE = {
    "family": "H-E, H-B1, H-B2: confirmatory or estimand as results.json (a shell: as analogs/protocol_inputs.json) says; an estimand prints "
              "its interval and mark, no p and no decision; a shell or an analysis that did not complete has a dash in each cell",
    "pc_hs": "PC and H-S rows; a shell or an analysis that did not complete has a dash in each cell",
    "e1": "E1: a column per component while that fits the text width (a quarter), a row per component otherwise (pooled values); a shell "
          "or an analysis that did not complete has a dash in each cell",
    "e2e4": "the registry's stats (SELC, NET, E2, E3, E4); a shell or an analysis that did not complete has a dash in each cell",
    "e5": "the ten blocks; a shell or a block that did not complete has a dash in each cell",
    "coverage": "who was out of the test and why; a test that did not run adds the coverage verdict",
    "ex_caiso": "the CAISO-free twin of the E2 to E4 table",
    "audit": "failures, nodes and medians of the activity audit in the run's order",
    "econ": "NET, SELC, SNET, ESred, CAP of the cutoff policies (premium position, calibrated fees)",
    "bh": "the sealed twin of tab_bh_families: tests and Benjamini-Hochberg rejections in the three exploratory families; a family that did "
          "not complete is a row of dashes",
}


def _register(test: str) -> None:
    res = _res(test)
    for name, fn in (("family", _family), ("pc_hs", _pc_hs), ("e1", _e1), ("e2e4", _e2e4), ("e5", _e5)):
        builder(f"tab_sealed_{name}_{test}", note=_NOTE[name])(partial(fn, test=test))      # no needs: a shell before the run
    for name, fn in (("coverage", _coverage), ("ex_caiso", _ex_caiso), ("audit", _audit), ("econ", _econ), ("bh", _bh)):
        builder(f"tab_sealed_{name}_{test}", res, note=_NOTE[name])(partial(fn, test=test))
    if test == "B":
        builder("tab_sealed_decision_B", res, _res("A"),
                note="Holm panel for the tests of the confirmatory family, estimand panel for the others (an interval, the sign "
                     "agreement, no decision)")(partial(_decision, test="B"))
        builder("tab_sealed_this_quarter_B", res, note="the estimands of Test B alone")(partial(_this_quarter, test="B"))


for _test in ("A", "B"):
    _register(_test)
