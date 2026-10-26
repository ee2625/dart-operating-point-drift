"""Assemble the upload set locally (never uploaded by code): LaTeX source, class, style, bib, bbl, PDF, the PDF twins of the figures
(generated/figures/Fig*.pdf, under the names the LaTeX source uses), the EPS files of the figures that the main text cites, Online
Resource 1, cover letter and the corrections to the preprint (CORRECTIONS_TO_PREPRINT.md, the reader-facing list; the internal
revision log, REVISION_LOG.md, is not uploaded).

Process comments. The sources are uploaded too, and the files of the tree begin with comments about how they were made. Every .tex
file that goes into the zip (main.tex, crossref.tex, the sections, an appendix if there is one, the Online Resource and the sections it
inputs, the generated macros and tables) is copied without its full-line comments, those whose first non-blank character is a percent
sign; the files of the tree stay as they are. A comment line takes its own newline with it in LaTeX, so removing the line with its newline
compiles to the same text. A line with text before the percent sign (a trailing comment) is left alone, and so is the body of a
verbatim-type environment, where a percent sign is text.

Figure files. The journal wants the files named Fig1, Fig2, ... in the order of the text. The generator names its figures after what
they show: the main text cites FigFail (the failure counts), FigDrift (the drift of the alert rate) and FigContrast (the contrasts with the
calendar-only rule), and the four older figures Fig1 to Fig4 (the node-level figure, the effect figure, the decomposition and the error by block, which is also
in the Test A part of the forward-test section) are in Online Resource 1 only, since the forward-test design (plan 2C, task T2a) and the old figures of
Sections 4 and 5 (task T2c) moved there. So the EPS files in figures/ are named by position, and a generated name is not a position: the figures that the
main text cites, in the order of their first citation, are Fig1.eps (generated FigFail), Fig2.eps (FigDrift), Fig3.eps (FigContrast), ... (the
`\\genfigure{Name}` calls of main.tex and of the files it inputs, in `\\input` order; the figures of esm/ do not count; a branch that the switches of
generated/macros.tex skip is not read, so the mapping is the one of the state of the manuscript when the package is built).
figures/FIGURE_MAP.txt says which generated file each one is. A generated figure that the main text does not cite is not in figures/. The
LaTeX source and the PDF twins keep their generated names, so the compile of the zip is unchanged.
Run after ./build.sh strict, pdf and esm: DART_PROJECT_ROOT=... /opt/anaconda3/bin/python3 -m es.paper_package"""
from __future__ import annotations

import hashlib
import re
import sys
import zipfile
from datetime import date
from pathlib import Path
from typing import Iterator, NamedTuple

# The lint's own reading of the switches, so that the package and the lint agree on which branch of \ifTestA the main text prints.
from .paper_lint import SWITCHES, _dead_ranges, _strip_comment, switch_state
from .paper_paths import PAPER_DIR

REQUIRED = ("main.tex", "main.pdf", "main.bbl", "sn-jnl.cls", "sn-mathphys-num.bst", "references.bib", "crossref.tex", "generated/macros.tex",
            "esm/ESM_1.pdf", "cover_letter.md", "CORRECTIONS_TO_PREPRINT.md")
GLOBS = ("sections/*.tex", "appendices/*.tex", "esm/*.tex", "generated/tables/*.tex", "generated/figures/Fig*.pdf", "generated/provenance.csv")
FIGURE_MAP = "figures/FIGURE_MAP.txt"
_CALL = re.compile(r"\\(input|include|genfigure)\s*\{([^}]*)\}")


class Cited(NamedTuple):
    name: str       # the generated figure: generated/figures/<name>.pdf and .eps
    source: str     # the file of its first citation, relative to the paper folder
    label: str      # the \label of its figure environment ("" when it has none)


def switch_values(paper_dir: Path) -> dict[str, bool]:
    """The switches of generated/macros.tex (\\TestAtrue, \\TestAdidnotrunfalse, ...). A switch that the file does not set is false, as
    \\newif leaves it."""
    state = dict.fromkeys(SWITCHES, False)
    macros = Path(paper_dir) / "generated" / "macros.tex"
    if macros.is_file():
        state.update(switch_state(macros.read_text(encoding="utf-8")))
    return state


def _calls(paper_dir: Path, rel: str, state: dict[str, bool], chain: tuple[str, ...] = ()) -> Iterator[Cited]:
    """The \\genfigure calls of ``rel`` in text order, the files that it inputs read at the place of their \\input. Comments, the branches
    that ``state`` skips and the generated files (macros, tables) are not read; a file that does not exist is skipped (LaTeX would stop)."""
    path = Path(paper_dir) / rel
    if rel in chain or not path.is_file():
        return
    text = "\n".join(_strip_comment(line) for line in path.read_text(encoding="utf-8").splitlines())
    dead = _dead_ranges(text, state)
    for m in _CALL.finditer(text):
        if any(a <= m.start() < b for a, b in dead):
            continue
        target = m.group(2).strip()
        if m.group(1) == "genfigure":
            begin, end = text.find("\\begin{figure", m.end()), text.find("\\end{figure", m.end())
            inside = text[m.end():end] if end >= 0 and (begin < 0 or end < begin) else ""      # up to the end of its own environment
            label = re.search(r"\\label\{([^}]*)\}", inside)
            yield Cited(target, rel, label.group(1) if label else "")
        elif not target.startswith("generated/"):
            yield from _calls(paper_dir, target if target.endswith(".tex") else target + ".tex", state, chain + (rel,))


def main_text_figures(paper_dir: Path, state: dict[str, bool] | None = None) -> list[Cited]:
    """The generated figures that the main text cites, in the order of their first citation: main.tex and the files it inputs (in
    \\input order), not esm/. ``state`` defaults to the switches of the paper folder's generated/macros.tex."""
    paper_dir = Path(paper_dir)
    state = switch_values(paper_dir) if state is None else state
    seen: set[str] = set()
    out: list[Cited] = []
    for c in _calls(paper_dir, "main.tex", state):
        if c.name not in seen:
            seen.add(c.name)
            out.append(c)
    return out


def figure_map(paper_dir: Path, cited: list[Cited], state: dict[str, bool]) -> str:
    """The text of figures/FIGURE_MAP.txt: which generated file each EPS file of the zip is, and which generated figures are not in it."""
    paper_dir = Path(paper_dir)
    lines = ["Figure files of this upload set (written by es.paper_package).", "",
             "The journal wants the figure files named Fig1, Fig2, ... in the order in which the main text cites the figures. The names below are",
             "that order, read from main.tex and the files it inputs, for the state of the switches in generated/macros.tex:",
             " ".join(f"{k}={'true' if v else 'false'}" for k, v in state.items()), ""]
    lines += [f"Fig{i}.eps is generated/figures/{c.name}.eps (cited in {c.source}" + (f", label {c.label}" if c.label else "") + ")"
              for i, c in enumerate(cited, 1)] or ["The main text cites no figure."]
    figures = paper_dir / "generated" / "figures"
    others = sorted({p.stem for p in figures.glob("Fig*") if p.suffix in (".pdf", ".eps")} - {c.name for c in cited}) if figures.is_dir() else []
    if others:
        lines += ["", "Not in the main text, so not in figures/ (they are printed in Online Resource 1, or in the main text only after Test A has run):"]
        lines += [f"generated/figures/{name}.eps" for name in others]
    lines += ["", "The LaTeX source and the PDF twins (generated/figures/Fig*.pdf) keep the names that the generator gives, so the compile in this",
              "zip is unchanged."]
    return "\n".join(lines) + "\n"


#: Environments whose body is typeset or written out as it stands: a line that begins with a percent sign inside one of them is text.
VERBATIM_ENVIRONMENTS = ("verbatim", "verbatim*", "Verbatim", "Verbatim*", "BVerbatim", "LVerbatim", "lstlisting", "minted", "alltt",
                         "comment", "filecontents", "filecontents*")
_BEGIN_VERBATIM = re.compile(r"\\begin\s*\{(" + "|".join(re.escape(e) for e in VERBATIM_ENVIRONMENTS) + r")\}")
_VERB = re.compile(r"\\verb\*?([^A-Za-z*\s])(.*?)\1")


def _comment_start(line: str, start: int = 0) -> int:
    """The index of the first percent sign of ``line`` at or after ``start`` that begins a comment, or len(line) when there is none. A
    backslash takes the next character with it (\\% is a percent sign, \\\\ is a line break and a % after it is a comment), and the
    argument of an inline \\verb is text."""
    i, n = start, len(line)
    while i < n:
        c = line[i]
        if c == "\\":
            m = _VERB.match(line, i)
            i = m.end() if m else i + 2
        elif c == "%":
            return i
        else:
            i += 1
    return n


#: A trailing comment that is one of the markers this project leaves for its own use (to the author, to the controller, to the linter).
_MARKER_COMMENT = re.compile(r"%+[ \t]*(?:CONTROLLER|AUTHOR|lint:allow)\b")


def _drop_marker_comment(line: str) -> str:
    """``line`` without a trailing comment of ours (% CONTROLLER..., % AUTHOR..., % lint:allow ...) and the blanks before it. Only a comment
    that has a blank before its percent sign and text before the blank: a bare percent sign at the end of a line (it glues two lines
    together) and a comment of anyone else's are left alone."""
    start = _comment_start(line)
    if start in (0, len(line)) or line[start - 1] not in " \t" or not _MARKER_COMMENT.match(line, start):
        return line
    return line[:start].rstrip(" \t") + ("\r" if line.endswith("\r") else "")


def strip_comment_lines(text: str, markers: bool = False) -> str:
    """``text`` without its full-line comments: the lines whose first non-blank character is a percent sign, each removed with its newline.
    Not touched: a line with text before the percent sign, a line that begins with an escaped \\%, and everything between the \\begin and the
    \\end of a verbatim-type environment (VERBATIM_ENVIRONMENTS), where a percent sign is text. A \\begin{verbatim} inside a comment does
    not open one. Lines end at "\\n" only (a \\r stays with its line), so a file keeps the line ends it has. With ``markers`` a trailing
    comment of ours (_MARKER_COMMENT) goes too, outside the verbatim-type environments: TeX drops the blanks at the end of a line and
    turns its newline into one space, which is also what the line gave with the comment."""
    lines = text.split("\n")
    kept: list[str] = []
    closing: str | None = None                 # the \end{...} that closes the verbatim-type environment we are in
    for number, line in enumerate(lines):
        if closing is None and line.lstrip().startswith("%"):
            if number == len(lines) - 1:        # an unterminated last line: the newline before it stays
                kept.append("")
            continue
        kept.append(_drop_marker_comment(line) if markers and closing is None else line)
        pos = 0
        while True:
            if closing is not None:
                end = line.find(closing, pos)    # inside, a percent sign is text: the end marker counts wherever it stands
                if end < 0:
                    break
                pos, closing = end + len(closing), None
                continue
            opened = _BEGIN_VERBATIM.search(line, pos, _comment_start(line, pos))
            if opened is None:
                break
            pos, closing = opened.end(), "\\end{" + opened.group(1) + "}"
    return "\n".join(kept)


_HTML_COMMENT = re.compile(r"<!--.*?-->[ \t]*\n?", re.S)


def strip_bib_comments(text: str) -> str:
    """``text`` of a .bib file without the lines between its entries whose first non-blank character is a percent sign (BibTeX reads
    everything outside an entry as a comment). A line inside an entry, that is while braces are open, is never touched: there a percent
    sign is text. Braces preceded by a backslash do not count."""
    kept: list[str] = []
    depth = 0
    for line in text.split("\n"):
        if depth == 0 and line.lstrip().startswith("%"):
            continue
        kept.append(line)
        depth = max(0, depth + len(re.findall(r"(?<!\\)\{", line)) - len(re.findall(r"(?<!\\)\}", line)))
    return "\n".join(kept)


def _member_bytes(src: Path, arc: str) -> bytes:
    """What goes into the zip under ``arc``: the file as it is, except that a .tex file loses its full-line comments and its trailing
    comments of ours, a .md file (the cover letter) loses its HTML comments, which are notes to the author, and a .bib file loses the
    comment lines between its entries."""
    data = src.read_bytes()
    if arc.endswith(".tex"):
        data = strip_comment_lines(data.decode("utf-8", "surrogateescape"), markers=True).encode("utf-8", "surrogateescape")
    elif arc.endswith(".md"):
        data = (_HTML_COMMENT.sub("", data.decode("utf-8", "surrogateescape")).rstrip() + "\n").encode("utf-8", "surrogateescape")
    elif arc.endswith(".bib"):
        data = strip_bib_comments(data.decode("utf-8", "surrogateescape")).encode("utf-8", "surrogateescape")
    return data


def build_package(paper_dir: Path, out_dir: Path) -> Path:
    paper_dir, out_dir = Path(paper_dir), Path(out_dir)
    missing = [r for r in REQUIRED if not (paper_dir / r).exists()]
    if missing:
        raise FileNotFoundError(f"submission package incomplete: {missing}")
    state = switch_values(paper_dir)
    cited = main_text_figures(paper_dir, state)
    figures = paper_dir / "generated" / "figures"
    no_eps = [f"generated/figures/{c.name}.eps" for c in cited if not (figures / f"{c.name}.eps").is_file()]
    if no_eps:         # the names are places in the text: a missing file would shift every later name
        raise FileNotFoundError(f"submission package incomplete: the main text cites figures that have no EPS file: {no_eps}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"EnergySystems_submission_{date.today().isoformat()}.zip"
    members: list[tuple[Path, str]] = [(paper_dir / r, r) for r in REQUIRED]
    for g in GLOBS:
        members += [(p, str(p.relative_to(paper_dir))) for p in sorted(paper_dir.glob(g))]
    members += [(figures / f"{c.name}.eps", f"figures/Fig{i}.eps") for i, c in enumerate(cited, 1)]
    mapping = figure_map(paper_dir, cited, state).encode("utf-8")
    lines = []
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for src, arc in members:
            data = _member_bytes(src, arc)              # the manifest hashes what the zip holds, not the file of the tree
            info = zipfile.ZipInfo.from_file(src, arc)
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, data)
            lines.append(f"{hashlib.sha256(data).hexdigest()}  {arc}")
        z.writestr(FIGURE_MAP, mapping)
    lines.append(f"{hashlib.sha256(mapping).hexdigest()}  {FIGURE_MAP}")
    (out_dir / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    return out


def main(argv=None) -> int:
    out = build_package(PAPER_DIR, PAPER_DIR / "submission")
    print(out)
    with zipfile.ZipFile(out) as z:
        print(z.read(FIGURE_MAP).decode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
