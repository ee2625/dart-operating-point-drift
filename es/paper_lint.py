"""Manuscript lint (spec v3 sections 5 item 8, 7.3, 8, 10; docs/energy_systems_requirements.md).
Rules: numbers (no hand-typed digits; below), newcommand (no macro definitions in sections), reversal (the 2026
reversal stays out of title, abstract and contributions), contributions (the BEGIN/END CONTRIBUTIONS markers that the
reversal rule reads must exist; error in strict), headings (three levels at most), eps (never include EPS), bib (every
cited key exists), generated (generated/macros.tex exists, and every \\gentable/\\genfigure target exists; error in
strict), footnotes (no bibliographic-only footnotes), declarations (required \\bmhead headings), abstract (150-250
words in every state of the \\ifTestA, \\ifTestAdidnotrun, ... switches that it tests, so that each reader's abstract is counted and
not the sum of both branches, strict; conditionals that cannot be evaluated are an error in both modes), keywords (4-6, strict),
pending (no pending sentinel left: in the generated macros and table cells it is
an error in strict only; typed into the text it is an error in strict and info otherwise; Test B is info in both).

The numbers rule. Every digit run in the text is an error unless it is (a) an allowed pattern (ALLOWED_TOKENS: the years
2015-2030 and ISO dates in them, window ids such as 24H2-25H2 and 2026-H1, the design ids P1, H-B1, E5, ..., SHA-256,
UTF-8, ISO 8601, math subscripts), (b) an argument that holds a name, path or size (STRIP_COMMANDS: \\ref, \\label,
\\cite, \\input, \\hspace, ... with the number of brace groups each command takes; the text argument of \\multicolumn,
\\multirow and \\href is text and is linted), or (c) named in a trailing '% lint:allow <token> ...' comment of its line:
each listed token exempts that number on that line and nothing else.

The text rules and the typed-sentinel scan cover sections/, appendices/ (the manuscript has none since plan 2C task T2a) and every
esm/*.tex: the Online Resource documents (esm/ESM_1.tex) and the sections that they input (esm/forward_tests.tex,
esm/designation_rule_study.tex, ...). An ESM document is linted for numbers and the other text rules from \\begin{document} on (its
preamble is template boilerplate, like main.tex, which is not linted); a file without \\begin{document}, one that a document inputs, is
linted whole; the sentinel scan reads the whole file. The key of a \\esmref or \\mainref (paper/crossref.tex) is a name, not text.
Run: DART_PROJECT_ROOT=... /opt/anaconda3/bin/python3 -m es.paper_lint [--strict [--pending-ok A]] [--paper-dir DIR]
(--pending-ok A: strict for a submission made before Test A has run; the Test A placeholders of the generated macros and tables are then
info, while Test A is absent from generated/macros.tex. build.sh package-pending uses it.)"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from .paper_paths import PAPER_DIR

YEAR = r"20(?:1[5-9]|2\d|30)"          # the years this paper uses, 2015-2030; any other year is a hand-typed number
#: A year may end a sentence ("... to 2026."), so the lookahead refuses only a following digit or decimal part (2026.5).
ALLOWED_TOKENS = [                                                                 # order matters: dates first
    r"\b" + YEAR + r"-\d{2}-\d{2}\b",                                              # ISO dates
    r"(?<![\d.])" + YEAR + r"-(?:H[12]|Q[1-4])(?!\d|\.\d)",                        # 2026-H1, 2025-Q3
    r"(?<![\d.])" + YEAR + r"(?:\s*(?:--|-|–)\s*(?:20)?(?:1[5-9]|2\d|30))?(?!\d|\.\d)",   # years and ranges
    r"\b\d{2}H[12]\b", r"\bAQ[1-4]F?\b",                                           # window ids: 24H2-25H2, 25H1-26H1, AQ3
    r"\b(?:H[12]|Q[1-4])\b", r"\bP[1-6]\b", r"P4@r\*", r"\bH-(?:B[12]|E|S)\b", r"\bE[1-5]\b",
    r"\bD\d{1,2}\b", r"\bG[16]\b", r"\bR[1-3]\b", r"\bFig\d\b",
    r"ESM\\_\d", r"Online Resource[ ~]\d", r"\\log_\{10\}",                        # journal names and the log base
    r"\bSHA-256\b", r"\bUTF-8\b", r"\bISO[ ~]8601\b",                              # standards
    r"(?<!\\)_(?:\d+|\{[\w,+\-]*\})",                                              # math subscripts: x_1, y_{10}, R_{t-1}
]
_GROUP = r"\{(?:[^{}]|\{[^{}]*\})*\}"          # one brace group, with one level of nesting inside it
_OPT = r"(?:\[[^\]]*\]){0,2}"                   # up to two optional [..] arguments: \citep[see][p.~5]{key}


def _args(names: str, n: int) -> str:
    """``\\name[opt]{..}`` with exactly n brace groups: the arguments that are names, paths or sizes, never the text."""
    return r"\\(?:" + names + r")\*?" + _OPT + r"(?:" + _GROUP + r"){" + str(n) + r"}"


STRIP_COMMANDS = re.compile("|".join([
    _args(r"ref|autoref|cref|Cref|nameref|eqref|pageref|esmref|mainref|label|cite[a-z]*|input|include|gentable|bibliography|"
          r"bibliographystyle|includegraphics|hspace|vspace|url|href|doi|orcidlink|cline|begin|end|"
          r"scalebox|rotatebox|raisebox|parbox", 1),             # \href{url}{text}, \scalebox{0.9}{text}: the text stays linted
    _args(r"genfigure|setlength|multicolumn|multirow|resizebox|rule", 2),   # \multicolumn{2}{c}{text}: the text is a third group
    r"\\cmidrule(?:\[[^\]]*\])?(?:\([^)]*\))?" + _GROUP,          # booktabs: \cmidrule[width](trim){a-b}
    r"\\(?:makebox|framebox)(?:\[[^\]]*\]){1,2}",                 # the width is the bracket; the brace group is the text
    r"\\footnote(?:mark|text)\[[^\]]*\]",                         # the index of a table note; a \footnotetext keeps its text
    r"\\\\\*?\[[^\]]*\]",                                         # a row end with a size: \\[2pt]
]))
#: The size and column arguments of environments: (environment names, mandatory brace groups after the name and an
#: optional [..]). Longer alternatives first: tabular* takes two groups, tabular one.
_ENV_ARGS = [(r"tabularx|tabulary|tabular\*", 2),
             (r"tabular|longtable|minipage|subfigure|multicols\*?|thebibliography|adjustbox|array", 1),
             (r"table\*?|figure\*?", 0)]
#: wrapfig: \begin{wrapfigure}[lines]{placement}[overhang]{width}. The overhang is a third optional argument, between the
#: two groups, so the generic "optional, then n groups" form does not reach the width behind it.
_WRAP = (r"\\begin\{(?:wrapfigure|wraptable)\}(?:\[[^\]]*\])?(?:" + _GROUP + r")(?:\[[^\]]*\])?(?:" + _GROUP + r")")
STRIP_OPTIONS = re.compile("|".join([_WRAP] + [
    r"\\begin\{(?:" + names + r")\}(?:\[[^\]]*\])?(?:" + _GROUP + r"){" + str(n) + r"}" for names, n in _ENV_ARGS]))
REQUIRED_HEADINGS = ("Funding", "Competing interests", "Ethics approval", "Data availability",
                     "Code availability", "Author contribution", "Prior dissemination")
PENDING_TOKEN = re.compile(r"\\PENDING(CONST|DEV|A|B|AUTHOR)(?![A-Za-z])")     # the sentinels of es.paper_macros.PENDING
_NUMBER = re.compile(r"\d(?:[\d.,]*\d)?")                                       # no trailing punctuation: "13." is 13


@dataclass
class Finding:
    path: str
    line: int
    rule: str
    severity: str      # "error" | "warning" | "info"
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.rule} [{self.severity}]: {self.message}"


def _strip_comment(line: str) -> str:
    out, prev = [], ""
    for ch in line:
        if ch == "%" and prev != "\\":
            break
        out.append(ch)
        prev = ch
    return "".join(out)


def _comment(line: str) -> str:
    """The text after the first unescaped % of a line ("" when there is none)."""
    prev = ""
    for i, ch in enumerate(line):
        if ch == "%" and prev != "\\":
            return line[i + 1:]
        prev = ch
    return ""


def _allowed(line: str) -> set[str]:
    """The tokens a trailing '% lint:allow <token> ...' comment names (trailing punctuation dropped)."""
    m = re.search(r"lint:allow\b(.*)$", _comment(line))
    return {t.strip(".,;:") for t in m.group(1).split()} - {""} if m else set()


def lint_numbers(text: str, path: str) -> list[Finding]:
    out = []
    for i, raw in enumerate(text.splitlines(), 1):
        allowed = _allowed(raw)
        line = STRIP_COMMANDS.sub(" ", STRIP_OPTIONS.sub(" ", _strip_comment(raw)))   # environments first, then commands
        for pat in ALLOWED_TOKENS:
            line = re.sub(pat, " ", line)
        for m in _NUMBER.finditer(line):
            if m.group(0) in allowed:
                continue
            out.append(Finding(path, i, "numbers", "error",
                               f"hand-typed number {m.group(0)!r}: use a generated macro or add '% lint:allow {m.group(0)}'"))
    return out


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _lint_body(text: str) -> str:
    """The text the text rules apply to: the whole file, except that a standalone document (esm/ESM_1.tex) is linted
    from \\begin{document} on, because its preamble is template boilerplate (a \\newcommand with [1] and #1, an
    \\author*[1]), like main.tex's. The preamble lines are blanked, not dropped, so line numbers stay the file's own."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "\\begin{document}" in _strip_comment(line):
            return "\n".join([""] * (i + 1) + lines[i + 1:]) + "\n"
    return text


def lint_file(path: Path, strict: bool) -> list[Finding]:
    text = _lint_body(path.read_text(encoding="utf-8"))
    out = lint_numbers(text, str(path))
    for i, line in enumerate(text.splitlines(), 1):
        code = _strip_comment(line)
        if re.search(r"\\(?:re)?newcommand|\\def\\|\\providecommand", code):
            out.append(Finding(str(path), i, "newcommand", "error", "macros are defined only in generated/macros.tex"))
        if re.search(r"\\(?:sub)?paragraph\b", code):
            out.append(Finding(str(path), i, "headings", "error", "at most three heading levels: no \\paragraph"))
        if re.search(r"\\includegraphics(?:\[[^\]]*\])?\{[^}]*\.eps\}", code):
            out.append(Finding(str(path), i, "eps", "error", "never include EPS (Ghostscript is unavailable); include the PDF twin"))
        if re.search(r"\\footnote\{\s*\\cite[a-z]*\{[^}]*\}\s*\}", code):
            out.append(Finding(str(path), i, "footnotes", "error", "a footnote must not hold bibliographic details alone"))
    return out


def _contributions(text: str):
    """The text between the BEGIN and END CONTRIBUTIONS markers, or None when a marker is missing or out of order."""
    m = re.search(r"% BEGIN CONTRIBUTIONS(.*?)% END CONTRIBUTIONS", text, flags=re.S)
    return m.group(1) if m else None


def _braced_arg(text: str, command: str):
    """The balanced-brace argument of ``command{...}`` in ``text`` (one level of nesting or more), or None."""
    start = text.find(command + "{")
    if start < 0:
        return None
    depth, first = 0, start + len(command)
    for j in range(first, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[first + 1:j]
    return None


# ---------------------------------------------------------------- the abstract, as each reader sees it (Task 9)
#: The switches of the generated macros (es.paper_macros.render writes \newif\ifTestA and its three siblings) that a text may test.
#: A test that did not run has run: render() sets TestA and TestAdidnotrun both true.
SWITCHES = ("TestA", "TestAdidnotrun", "TestB", "TestBdidnotrun")
_SWITCH_TOKEN = re.compile(r"\\(?:if(" + "|".join(sorted(SWITCHES, key=len, reverse=True)) + r")|(else)|(fi))(?![A-Za-z])")
_WORD = re.compile(r"\\[A-Za-z]+|[A-Za-z][A-Za-z'\-]*")      # a word, or a macro (which prints a word or a number), as the count has always read it


@dataclass
class AbstractVariant:
    label: str      # the state of the switches in words ("Test A has run"); "" when the abstract tests none
    words: int
    text: str       # the abstract as this reader sees it: the branches that the state skips are gone, and the switches with them


def _resolve_switches(text: str, setting: dict[str, bool]) -> str | None:
    """``text`` with every \\ifTestA ... [\\else ...] \\fi (and the other switches) resolved under ``setting``; None when the conditionals
    are unbalanced (an \\else or \\fi without its \\if, an \\if never closed) or when the text tests anything but a switch."""
    if any(m.group(1) not in SWITCHES for m in re.finditer(r"\\if([A-Za-z]*)", text)):
        return None
    out, stack, pos = [], [], 0              # stack: [switch, in its \else part]

    def live() -> bool:
        return all(setting[switch] != in_else for switch, in_else in stack)
    for m in _SWITCH_TOKEN.finditer(text):
        if live():
            out.append(text[pos:m.start()])
        pos = m.end()
        if m.group(1):
            stack.append([m.group(1), False])
        elif m.group(2):
            if not stack or stack[-1][1]:
                return None
            stack[-1][1] = True
        else:
            if not stack:
                return None
            stack.pop()
    if stack:
        return None
    out.append(text[pos:])
    return "".join(out)


def _settings(text: str) -> list[dict[str, bool]]:
    """Every state of the switches that the text tests (the others stay false), without the impossible ones: a test that did not run
    has run. Two states that come out equal after that are one."""
    used = [s for s in SWITCHES if re.search(r"\\if" + s + r"(?![A-Za-z])", text)]
    seen, out = set(), []
    for values in itertools.product((False, True), repeat=len(used)):
        setting = dict.fromkeys(SWITCHES, False)
        setting.update(zip(used, values))
        for t in "AB":
            setting[f"Test{t}"] = setting[f"Test{t}"] or setting[f"Test{t}didnotrun"]
        key = tuple(setting[s] for s in used)
        if key not in seen:
            seen.add(key)
            out.append(setting)
    return out


def _state_label(setting: dict[str, bool], text: str) -> str:
    parts = []
    for t in "AB":
        if any(re.search(r"\\if" + s + r"(?![A-Za-z])", text) for s in (f"Test{t}", f"Test{t}didnotrun")):
            parts.append(f"Test {t} did not run" if setting[f"Test{t}didnotrun"]
                         else f"Test {t} has run" if setting[f"Test{t}"] else f"Test {t} has not run")
    return ", ".join(parts)


def abstract_variants(arg: str) -> list[AbstractVariant] | None:
    """The abstract (the argument of \\abstract, comments removed) once for each state of the switches that it tests, so that each
    reader's count is the count of the words that reader sees: the old count added the words of both branches of \\ifTestA, and the
    three switch tokens, to one number. An abstract that tests none is one variant with no label. None when the conditionals cannot
    be evaluated (unbalanced, or a conditional other than the switches)."""
    out = []
    for setting in _settings(arg):
        text = _resolve_switches(arg, setting)
        if text is None:
            return None
        out.append(AbstractVariant(_state_label(setting, arg), len(_WORD.findall(text)), text))
    return out


def abstract_word_counts(paper_dir: Path) -> dict[str, int]:
    """{state: words} of the abstract of a paper folder, as the lint counts it ({} when the folder has no abstract, or when its conditionals
    cannot be evaluated). The abstract of a paper that tests no switch has the one key "" ."""
    path = Path(paper_dir) / "sections" / "00_abstract.tex"
    if not path.exists():
        return {}
    body = " ".join(_strip_comment(line) for line in path.read_text(encoding="utf-8").splitlines())
    arg = _braced_arg(body, "\\abstract")
    variants = abstract_variants(arg) if arg is not None else None
    return {v.label: v.words for v in variants} if variants else {}


def switch_state(macros_text: str) -> dict[str, bool]:
    """The state of the switches that the generated macros.tex sets (\\TestAtrue, \\TestAdidnotrunfalse, ...); a switch that it does not set is
    absent from the result."""
    return {m.group(1): m.group(2) == "true"
            for m in re.finditer(r"\\(Test[AB](?:didnotrun)?)(true|false)(?![A-Za-z])", macros_text)}


def _dead_ranges(text: str, state: dict[str, bool]) -> list[tuple[int, int]]:
    """The character ranges of ``text`` that the switches skip under ``state`` (a branch of \\ifTestA that macros.tex turns off). A switch that
    ``state`` does not set keeps both its branches, and unbalanced conditionals skip nothing from the point where they go wrong."""
    def live(stack) -> bool:
        return all(state.get(switch) is None or state[switch] != in_else for switch, in_else in stack)
    ranges, stack, start = [], [], None
    for m in _SWITCH_TOKEN.finditer(text):
        was = live(stack)
        if m.group(1):
            stack.append([m.group(1), False])
        elif m.group(2):
            if stack:
                stack[-1][1] = True
        elif stack:
            stack.pop()
        now = live(stack)
        if was and not now:
            start = m.end()
        elif now and not was and start is not None:
            ranges.append((start, m.start()))
            start = None
    if start is not None:
        ranges.append((start, len(text)))
    return ranges


def lint_all(paper_dir: Path, strict: bool, pending_ok: tuple[str, ...] = ()) -> list[Finding]:
    """``pending_ok``: the sentinel stages that strict accepts in macro definitions and generated table cells; only "A", for a submission
    made before Test A has run. It holds only while generated/macros.tex says Test A is absent: once Test A is present nothing of it may
    be pending. A sentinel typed into the text is an error in strict whatever ``pending_ok`` says."""
    paper_dir = Path(paper_dir)
    out: list[Finding] = []
    sections = sorted((paper_dir / "sections").glob("*.tex")) + sorted((paper_dir / "appendices").glob("*.tex"))
    esm_docs = sorted((paper_dir / "esm").glob("*.tex"))    # Online Resource documents and the sections they input; main.tex is not linted
    for p in sections + esm_docs:
        out += lint_file(p, strict)
    # reversal: title, abstract, contributions
    inputs = json.loads((paper_dir / "author_inputs.json").read_text(encoding="utf-8")) if (paper_dir / "author_inputs.json").exists() else {}
    for key in ("title", "short_title"):
        if "revers" in str(inputs.get(key, "")).lower():
            out.append(Finding("author_inputs.json", 0, "reversal", "error", f"{key} mentions the reversal (spec 5 item 8)"))
    abstract = paper_dir / "sections" / "00_abstract.tex"
    if abstract.exists():
        atext = abstract.read_text(encoding="utf-8")
        body = " ".join(_strip_comment(l) for l in atext.splitlines())
        if "revers" in body.lower():
            out.append(Finding(str(abstract), 0, "reversal", "error", "the abstract mentions the reversal (spec 5 item 8)"))
        arg = _braced_arg(body, "\\abstract")                   # stops at the abstract's own closing brace
        variants = abstract_variants(arg) if arg is not None else [AbstractVariant("", 0, "")]
        kw = re.search(r"\\keywords\{([^}]*)\}", body)
        n_kw = len([k for k in kw.group(1).split(",") if k.strip()]) if kw else 0
        if variants is None:                                    # it would not compile either: an error whatever the mode
            switches = ", ".join("\\if" + s for s in SWITCHES)
            out.append(Finding(str(abstract), 0, "abstract", "error",
                               "the abstract's \\ifTestA ... \\fi conditionals are unbalanced, or test something other than "
                               f"{switches}: its words cannot be counted"))
        elif strict:
            for v in variants:                                  # every reader's abstract must fit, not the sum of their branches
                if not 150 <= v.words <= 250:
                    when = f" when {v.label}" if v.label else ""
                    out.append(Finding(str(abstract), 0, "abstract", "error", f"abstract has {v.words} words{when} (150-250 required)"))
        if strict and not 4 <= n_kw <= 6:
            out.append(Finding(str(abstract), 0, "keywords", "error", f"{n_kw} keywords (4-6 required)"))
    intro = paper_dir / "sections" / "01_introduction.tex"
    if intro.exists():
        block = _contributions(intro.read_text(encoding="utf-8"))
        if block is None:                                       # without the markers the reversal rule reads nothing
            out.append(Finding(str(intro), 0, "contributions", "error" if strict else "warning",
                               "the % BEGIN CONTRIBUTIONS and % END CONTRIBUTIONS markers are missing or out of order: "
                               "the reversal rule reads the list between them"))
        elif "revers" in block.lower():
            out.append(Finding(str(intro), 0, "reversal", "error", "the contributions list mentions the reversal (spec 5 item 8)"))
    # bib keys
    bib = paper_dir / "references.bib"
    keys = set(re.findall(r"^@[a-zA-Z]+\{([^,]+),", bib.read_text(encoding="utf-8"), flags=re.M)) if bib.exists() else set()
    for p in sections + esm_docs:
        for i, line in enumerate(_lines(p), 1):
            for m in re.finditer(r"\\cite[a-z]*\{([^}]*)\}", _strip_comment(line)):
                for k in m.group(1).split(","):
                    if k.strip() and k.strip() not in keys:
                        out.append(Finding(str(p), i, "bib", "error", f"cited key {k.strip()!r} is not in references.bib"))
    # generated targets
    macros = paper_dir / "generated" / "macros.tex"
    if not macros.exists():
        out.append(Finding(str(macros), 0, "generated", "error",
                           "generated/macros.tex is missing: run build.sh macros (nothing can be checked, and strict "
                           "cannot count pending macros, without it)"))
    state = switch_state(macros.read_text(encoding="utf-8")) if macros.exists() else {}
    for p in sections + esm_docs:
        stripped = "\n".join(_strip_comment(line) for line in _lines(p))
        dead = _dead_ranges(stripped, state)        # a table or figure of a branch that the generated switches skip is not printed: not required
        for pattern, folder, ext, what in ((r"\\gentable\{([^}]*)\}", "tables", "tex", "table"), (r"\\genfigure\{([^}]*)\}", "figures", "pdf", "figure")):
            for m in re.finditer(pattern, stripped):
                if any(a <= m.start() < b for a, b in dead):
                    continue
                if not (paper_dir / "generated" / folder / f"{m.group(1)}.{ext}").exists():
                    out.append(Finding(str(p), stripped.count("\n", 0, m.start()) + 1, "generated", "error" if strict else "warning",
                                       f"{what} {m.group(1)} not generated"))
    # declarations
    back = paper_dir / "sections" / "90_backmatter.tex"
    if back.exists():
        btext = back.read_text(encoding="utf-8")
        for h in REQUIRED_HEADINGS:
            if f"\\bmhead{{{h}" not in btext:
                out.append(Finding(str(back), 0, "declarations", "error", f"missing \\bmhead{{{h}...}}"))
    # pending sentinels (strict): macro definitions and generated table cells alike
    files = [macros] + sorted((paper_dir / "generated" / "tables").glob("*.tex"))
    for path in files:
        if not path.exists():
            continue
        for i, line in enumerate(_lines(path), 1):
            if line.startswith("\\newcommand{\\PENDING"):
                continue                                        # the sentinel definitions themselves
            for m in PENDING_TOKEN.finditer(line):
                named = re.search(r"\\newcommand\{\\([A-Za-z]+)\}", line)
                label = f"\\{named.group(1)}" if named else f"a cell of {path.name}"
                if m.group(1) == "B":
                    out.append(Finding(str(path), i, "pending-B", "info", f"{label} waits for Test B"))
                elif strict and m.group(1) in pending_ok and not state.get("TestA", False):
                    out.append(Finding(str(path), i, "pending-A", "info", f"{label} waits for Test A, which this submission does not include"))
                elif strict:
                    out.append(Finding(str(path), i, "pending", "error", f"{label} is pending ({m.group(1)})"))
    # pending sentinels typed into the text (sections, appendices, ESM documents; whole files, comments excluded):
    # an error in strict, info otherwise, except Test B, which may stay pending at submission (info)
    for p in sections + esm_docs:
        for i, line in enumerate(_lines(p), 1):
            for m in PENDING_TOKEN.finditer(_strip_comment(line)):
                if m.group(1) == "B":
                    out.append(Finding(str(p), i, "pending-B", "info", "\\PENDINGB is typed here and waits for Test B"))
                else:
                    out.append(Finding(str(p), i, "pending", "error" if strict else "info",
                                       f"\\PENDING{m.group(1)} is typed in the text: replace it with the final text "
                                       "or a generated macro"))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--paper-dir", default=str(PAPER_DIR))
    ap.add_argument("--pending-ok", default="", help="comma-separated sentinel stages that --strict accepts in the generated macros and "
                    "tables: only A, for a submission before Test A has run (refused for Test A once macros.tex says it is present)")
    a = ap.parse_args(argv)
    pending_ok = tuple(s for s in a.pending_ok.split(",") if s)
    if any(s != "A" for s in pending_ok):
        ap.error("--pending-ok accepts only A")
    findings = lint_all(Path(a.paper_dir), a.strict, pending_ok)
    for f in findings:
        print(f)
    errors = [f for f in findings if f.severity == "error"]
    print(f"{len(findings)} findings, {len(errors)} errors")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
