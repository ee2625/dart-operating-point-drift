"""Plan 2C, task T2a: the cross-reference mechanism between the paper and Online Resource 1 (paper/crossref.tex).

The paper (main.tex) and Online Resource 1 (esm/ESM_1.tex) are compiled on their own, so a \\ref cannot cross from one into the other, and a
typed section number would go stale each time a section moves. A reference across the boundary names its target by the title of the heading:
\\esmref{key} in the paper prints  Online Resource~1, section ``Title''  and \\mainref{key} in the ESM prints  section ``Title'' of the paper
(plan 2C task T2b: neither carries parentheses of its own, so that "(\\esmref{key})" prints one pair).
paper/crossref.tex holds the one key-to-title table. These tests keep it honest:

* every \\esmref{k} and \\mainref{k} used in a .tex file under paper/ has an entry (no key is typed that nothing defines);
* every entry's title is the text of one \\section, \\subsection or \\subsubsection heading of the target files (esm/*.tex for \\esmref,
  sections/*.tex for \\mainref), so that a renamed or removed heading fails here and not in a reader's hands;
* no \\ref crosses from one document into the other (a label of the paper is not referenced in the ESM, and the other way round);
* the macros print what they should and an unknown key stops the build; the ESM compiles with no undefined reference (slow test).

The readers of the table and of the headings are tested on small texts first, so that the checks on the real folder cannot pass because the
reader finds nothing. T2b and T2c add keys to the table: a new key needs one line there and nothing here."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

PAPER = Path(__file__).resolve().parents[3] / "paper"
needs_tree = pytest.mark.skipif(not (PAPER / "sections").exists(), reason="the manuscript folder is not next to dart_dev")
LATEXMK = Path("/Library/TeX/texbin/latexmk")
PDFLATEX = Path("/Library/TeX/texbin/pdflatex")
PDFTOTEXT = Path("/opt/homebrew/bin/pdftotext")
BUILD_SH = PAPER / "build.sh"
KEY = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")          # lower-case letters, digits and single hyphens: safe in a \csname and in a file name
SKIPPED_FOLDERS = ("submission", "build")


# ---------------------------------------------------------------------------------------------------- the readers
def live(text: str) -> str:
    """The text without its LaTeX comments (a commented use of a key is not a use)."""
    return re.sub(r"(?<!\\)%.*$", "", text, flags=re.M)


def group(text: str, start: int) -> str:
    """The content of the brace group that opens at text[start]; escaped characters (\\{, \\}, \\\\, \\%) do not count."""
    depth, i = 0, start
    while i < len(text):
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:i]
        i += 1
    raise ValueError("unbalanced braces after " + text[start:start + 60])


def squash(text: str) -> str:
    return " ".join(text.split())


def read_table(text: str) -> dict[str, list[tuple[str, str]]]:
    """{"esm": [(key, title), ...], "main": [...]} of the lines \\esmkey{key}{Title} and \\mainkey{key}{Title} of crossref.tex, in order
    (so that a key written twice shows)."""
    text = live(text)
    out: dict[str, list[tuple[str, str]]] = {"esm": [], "main": []}
    for m in re.finditer(r"\\(esm|main)key\s*\{([^{}]*)\}\s*\{", text):
        out[m.group(1)].append((m.group(2), squash(group(text, m.end() - 1))))
    return out


def read_uses(text: str) -> dict[str, list[str]]:
    """{"esm": [key, ...], "main": [...]}: the keys of the \\esmref{..} and \\mainref{..} of a text, comments excluded."""
    out: dict[str, list[str]] = {"esm": [], "main": []}
    for m in re.finditer(r"\\(esm|main)ref\s*\{([^{}]*)\}", live(text)):
        out[m.group(1)].append(m.group(2))
    return out


def read_headings(text: str) -> list[str]:
    """The titles (white space squashed) of the \\section, \\subsection and \\subsubsection headings of a text, starred or not, with or
    without a short title in brackets."""
    text = live(text)
    return [squash(group(text, m.end() - 1))
            for m in re.finditer(r"\\(?:sub){0,2}section\*?\s*(?:\[[^\]]*\])?\s*\{", text)]


def read_labels(text: str) -> set[str]:
    return set(re.findall(r"\\label\{([^}]*)\}", live(text)))


def read_refs(text: str) -> set[str]:
    return set(re.findall(r"\\(?:ref|autoref|cref|Cref|eqref|pageref|nameref)\*?\s*\{([^}]*)\}", live(text)))


def tex_files(*patterns: str) -> list[Path]:
    return [p for pattern in patterns for p in sorted(PAPER.glob(pattern))]


def all_tex_files() -> list[Path]:
    """Every .tex file under paper/ except crossref.tex itself (it holds the macros, with #1 where a key goes) and the build folders."""
    return [p for p in sorted(PAPER.rglob("*.tex"))
            if p.name != "crossref.tex" and not set(SKIPPED_FOLDERS) & set(p.relative_to(PAPER).parts)]


def gentables_of(*patterns: str) -> set[str]:
    """The names of the generated tables that the files of the patterns print with \\gentable{name}."""
    return {name for p in tex_files(*patterns) for name in re.findall(r"\\gentable\{([^{}]*)\}", live(p.read_text(encoding="utf-8")))}


def esm_only_tables() -> set[str]:
    """The generated tables that only Online Resource 1 prints (tab_implementation, tab_coverage, the ledger, ...): a file of generated/tables/ belongs to
    the document that inputs it, so a reference in such a table is a reference of the ESM (plan 2C task T2b: its section pointers are \\mainref)."""
    return gentables_of("esm/*.tex") - gentables_of("sections/*.tex", "main.tex")


def document_of(relative_path: str, esm_tables: set[str]) -> str:
    """"esm" for the files of esm/ and for the generated tables that only the ESM prints, "paper" for every other file."""
    if relative_path.startswith("esm/") or (relative_path.startswith("generated/tables/") and Path(relative_path).stem in esm_tables):
        return "esm"
    return "paper"


def paper_table() -> dict[str, list[tuple[str, str]]]:
    return read_table((PAPER / "crossref.tex").read_text(encoding="utf-8"))


def headings_of(*patterns: str) -> list[str]:
    return [title for p in tex_files(*patterns) for title in read_headings(p.read_text(encoding="utf-8"))]


# ---------------------------------------------------------------------------------------------------- the readers, on small texts
def test_the_readers_find_keys_titles_uses_and_headings_on_small_texts():
    table = ("% \\esmkey{in-a-comment}{Commented out}\n\\newcommand{\\esmkey}[2]{x}\n"
             "\\esmkey{one}{A title: with  a colon}\n\\esmkey{two}{Test A: July--September 2026}\n"
             "\\mainkey{three}{Statistics}\n\\mainkey{four}{A title with {braces} inside}\n")
    got = read_table(table)
    assert got["esm"] == [("one", "A title: with a colon"), ("two", "Test A: July--September 2026")]     # white space squashed
    assert got["main"] == [("three", "Statistics"), ("four", "A title with {braces} inside")]
    uses = read_uses("See \\esmref{one} and \\mainref{three}, \\esmref{one} again.\n% \\esmref{in-a-comment}\nAn escaped \\%\\esmref{two}.\n")
    assert uses == {"esm": ["one", "one", "two"], "main": ["three"]}
    text = ("\\section{Forward tests: full design}\\label{sec:a}\n\\subsection*{Drift monitors}\n"
            "\\subsubsection[Short]{Windows, exclusions\n  and daylight saving}\\label{sec:b}\n% \\section{Gone}\n\\paragraph{Not a heading}\n")
    assert read_headings(text) == ["Forward tests: full design", "Drift monitors", "Windows, exclusions and daylight saving"]
    assert read_labels(text) == {"sec:a", "sec:b"} and read_refs("See Section~\\ref{sec:a} and \\autoref{fig:x}, \\label{no}") == {"sec:a", "fig:x"}


# ---------------------------------------------------------------------------------------------------- the table
@needs_tree
def test_the_key_table_has_one_line_per_key_valid_keys_and_no_key_twice():
    raw = (PAPER / "crossref.tex").read_text(encoding="utf-8")
    table = read_table(raw)
    for kind in ("esm", "main"):
        keys = [k for k, _ in table[kind]]
        assert keys, f"the {kind} table is empty"
        assert len(keys) == len(set(keys)), f"a key is written twice in the {kind} table: {sorted(k for k in set(keys) if keys.count(k) > 1)}"
        assert [k for k in keys if not KEY.fullmatch(k)] == [], "keys are lower-case letters, digits and single hyphens"
        assert [k for k, title in table[kind] if not title] == [], "a title is empty"
    # one line per key, so that adding a key is one added line and a merge cannot garble two of them
    rows = [line for line in live(raw).splitlines() if re.match(r"\s*\\(?:esm|main)key\b", line)]
    assert len(rows) == len(table["esm"]) + len(table["main"])
    assert all(re.fullmatch(r"\\(?:esm|main)key\{[^{}]*\}\{.*\}\s*", line) for line in rows), "one complete \\esmkey{key}{Title} per line"
    # the explanation of how to add a key is in the file, as comments
    comments = "\n".join(line for line in raw.splitlines() if line.lstrip().startswith("%"))
    assert "To add a key" in comments and "\\esmkey{key}{Title}" in comments and "\\mainkey{key}{Title}" in comments


@needs_tree
def test_the_two_documents_input_the_table_and_the_moved_sections_are_where_the_keys_expect_them():
    """main.tex and the ESM input crossref.tex; the forward-test design and the size study are ESM sections (input by ESM_1.tex), not
    sections of the paper, and the paper has no appendix."""
    main, esm = live((PAPER / "main.tex").read_text(encoding="utf-8")), live((PAPER / "esm" / "ESM_1.tex").read_text(encoding="utf-8"))
    assert "\\input{crossref}" in main and "\\input{../crossref}" in esm
    assert "\\input{forward_tests}" in esm and "\\input{designation_rule_study}" in esm
    assert "sections/06_sealed" not in main and "appendices" not in main and "\\begin{appendices}" not in main
    assert not (PAPER / "sections" / "06_sealed.tex").exists() and not (PAPER / "appendices").exists()
    assert (PAPER / "esm" / "forward_tests.tex").is_file() and (PAPER / "esm" / "designation_rule_study.tex").is_file()
    assert "Forward tests: full design" in read_headings((PAPER / "esm" / "forward_tests.tex").read_text(encoding="utf-8"))
    assert "Size and power of the designation rule" in read_headings((PAPER / "esm" / "designation_rule_study.tex").read_text(encoding="utf-8"))


@needs_tree
def test_every_esmref_and_mainref_used_under_paper_has_an_entry():
    table = paper_table()
    known = {kind: {k for k, _ in table[kind]} for kind in table}
    used: dict[str, dict[str, list[str]]] = {"esm": {}, "main": {}}
    for path in all_tex_files():
        for kind, keys in read_uses(path.read_text(encoding="utf-8")).items():
            for key in keys:
                used[kind].setdefault(key, []).append(path.relative_to(PAPER).as_posix())
    assert used["esm"], "the paper does not use \\esmref: the check would pass vacuously"
    assert used["main"], "the ESM does not use \\mainref: the check would pass vacuously"
    for kind, macro in (("esm", "esmref"), ("main", "mainref")):
        missing = {k: sorted(set(files)) for k, files in used[kind].items() if k not in known[kind]}
        assert not missing, f"\\{macro} keys without a line in crossref.tex: {missing}"
    # a key is used in the document it is meant for: \esmref in the paper (sections/, main.tex), \mainref in the ESM (esm/ and the generated tables that only the ESM prints)
    esm_tables = esm_only_tables()
    assert {"tab_implementation", "tab_coverage"} <= esm_tables, esm_tables
    in_esm = {f for files in used["esm"].values() for f in files if document_of(f, esm_tables) == "esm"}
    in_paper = {f for files in used["main"].values() for f in files if document_of(f, esm_tables) == "paper"}
    assert not in_esm, f"\\esmref used inside Online Resource 1 (use \\ref there): {sorted(in_esm)}"
    assert not in_paper, f"\\mainref used inside the paper (use \\ref there): {sorted(in_paper)}"


@needs_tree
def test_every_entry_title_is_the_text_of_one_heading_of_its_target_files():
    table = paper_table()
    for kind, patterns, where in (("esm", ("esm/*.tex",), "esm/*.tex"), ("main", ("sections/*.tex",), "sections/*.tex")):
        headings = headings_of(*patterns)
        assert len(headings) > 10, f"the reader found only {len(headings)} headings in {where}"
        for key, title in table[kind]:
            n = headings.count(title)
            assert n >= 1, f"\\{kind}key{{{key}}}: no heading of {where} reads {title!r}"
            assert n == 1, f"\\{kind}key{{{key}}}: {n} headings of {where} read {title!r}; a reader cannot tell which one is meant"


@needs_tree
def test_no_label_reference_crosses_from_one_document_into_the_other():
    """A \\ref in the ESM names a label of the ESM, and a \\ref in the paper a label of the paper: across the boundary \\esmref and \\mainref
    are the only reference (a \\ref would print ?? in a document compiled on its own). Labels that the generated tables define are shared by both."""
    paper_files = tex_files("sections/*.tex") + [PAPER / "main.tex"]
    esm_files = tex_files("esm/*.tex")
    shared = {label for p in tex_files("generated/tables/*.tex") for label in read_labels(p.read_text(encoding="utf-8"))}
    paper_labels = {label for p in paper_files for label in read_labels(p.read_text(encoding="utf-8"))} - shared
    esm_labels = {label for p in esm_files for label in read_labels(p.read_text(encoding="utf-8"))} - shared
    assert paper_labels and esm_labels
    into_paper = {p.name: sorted(read_refs(p.read_text(encoding="utf-8")) & paper_labels - esm_labels) for p in esm_files}
    into_esm = {p.name: sorted(read_refs(p.read_text(encoding="utf-8")) & esm_labels - paper_labels) for p in paper_files}
    assert {k: v for k, v in into_paper.items() if v} == {}, "the ESM references labels of the paper; use \\mainref"
    assert {k: v for k, v in into_esm.items() if v} == {}, "the paper references labels of the ESM; use \\esmref"
    # the generated tables that only the ESM prints are part of it (plan 2C task T2b: a dead pointer to the old Section 6.1.1 became a \ref to a section of the ESM)
    esm_table_files = [PAPER / "generated" / "tables" / f"{name}.tex" for name in sorted(esm_only_tables())]
    esm_table_refs = {p.name: read_refs(p.read_text(encoding="utf-8")) for p in esm_table_files if p.is_file()}
    assert "sec:sealed-windows" in esm_table_refs["tab_implementation.tex"], "the implementation table should point at the ESM section by \\ref"
    assert {k: sorted(v & paper_labels - esm_labels) for k, v in esm_table_refs.items() if v & paper_labels - esm_labels} == {}, \
        "a table printed only in the ESM references a label of the paper; use \\mainref"
    assert {k: sorted(v - esm_labels - shared) for k, v in esm_table_refs.items() if v - esm_labels - shared} == {}, \
        "a table printed only in the ESM references a label that no file of the ESM defines"


# ---------------------------------------------------------------------------------------------------- the macros
def _stub(tmp_path: Path, body: str) -> subprocess.CompletedProcess:
    """Compile a minimal article that inputs the real crossref.tex and holds ``body``; returns the process (log in tmp_path/doc.log)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    shutil.copy(PAPER / "crossref.tex", tmp_path / "crossref.tex")
    (tmp_path / "doc.tex").write_text(                                  # no hyphenation: a title broken at the end of a line would not match
        "\\documentclass{article}\n\\hyphenpenalty=10000 \\exhyphenpenalty=10000 \\sloppy\n\\input{crossref}\n\\begin{document}\n" + body + "\n\\end{document}\n",
        encoding="utf-8")
    return subprocess.run([str(PDFLATEX), "-interaction=nonstopmode", "-halt-on-error", "doc.tex"], cwd=tmp_path, capture_output=True,
                          text=True, timeout=120)


def _text(tmp_path: Path) -> str:
    out = subprocess.run([str(PDFTOTEXT), "-layout", str(tmp_path / "doc.pdf"), "-"], capture_output=True, text=True, timeout=60).stdout
    return squash(out.replace("\u201c", '"').replace("\u201d", '"'))        # the typeset ``..'' pair, as plain quotation marks


needs_tex = pytest.mark.skipif(not (PDFLATEX.exists() and PDFTOTEXT.exists()), reason="pdflatex and pdftotext are not installed")


@needs_tex
@needs_tree
def test_the_macros_print_the_title_of_the_target_heading_and_no_section_number(tmp_path):
    table = paper_table()
    esm_key, esm_title = next((k, t) for k, t in table["esm"] if t == "Designation rule and designated status")
    main_key, main_title = next((k, t) for k, t in table["main"] if t == "Statistics")
    body = (f"A \\esmref{{{esm_key}}} B \\mainref{{{main_key}}} C \\emph{{\\esmref{{{esm_key}}}}}.\n\n"
            f"In parentheses (\\esmref{{{esm_key}}}) and (see \\mainref{{{main_key}}}).\n\n"
            f"\\section{{Moving}} \\mbox{{\\esmref{{{esm_key}}}}} in a \\footnote{{\\mainref{{{main_key}}}}}.\n\n"
            f"\\listoftables\n\\begin{{table}}[h]\\caption{{Moving argument, see \\esmref{{{esm_key}}}}}\\end{{table}}")
    for _ in range(2):                                                         # the second run reads the list of tables that the first wrote
        run = _stub(tmp_path, body)
        assert run.returncode == 0, run.stdout[-1500:]
    text = _text(tmp_path)
    assert f'A Online Resource 1, section "{esm_title}" B section "{main_title}" of the paper C Online Resource 1, section "{esm_title}".' in text
    assert f'In parentheses (Online Resource 1, section "{esm_title}") and (see section "{main_title}" of the paper).' in text   # one pair of parentheses, the writer's own
    assert f'section "{main_title}" of the paper' in text                                                      # also in a footnote
    assert text.count(f'Moving argument, see Online Resource 1, section "{esm_title}"') == 2                   # in the caption and in the list of tables
    assert "??" not in text and not re.search(r"Section\s+\d", text)                                           # no number is printed
    source = (PAPER / "crossref.tex").read_text(encoding="utf-8")
    assert "Online Resource~1, section ``" in source and "{section ``" in source and "'' of the paper" in source
    assert "Online Resource~1 (``" not in source and "the section ``" not in source                              # the forms of plan 2C task T2a are gone
    assert "\\DeclareRobustCommand{\\esmref}" in source and "\\DeclareRobustCommand{\\mainref}" in source      # safe in captions and headings


@needs_tex
@needs_tree
def test_an_unknown_key_stops_the_build_and_names_the_key_and_the_file_to_edit(tmp_path):
    run = _stub(tmp_path, "See \\esmref{no-such-key}.")
    log = (tmp_path / "doc.log").read_text(errors="replace")
    assert run.returncode != 0
    assert "Package crossref Error" in log and "no-such-key" in log and "crossref.tex" in log
    run = _stub(tmp_path / "main", "See \\mainref{no-such-key}.")
    assert run.returncode != 0 and "no-such-key" in (tmp_path / "main" / "doc.log").read_text(errors="replace")
    key = paper_table()["esm"][0][0]
    assert _stub(tmp_path / "ok", f"See \\esmref{{{key}}}.").returncode == 0                                       # and a known key builds


# ---------------------------------------------------------------------------------------------------- the ESM build
@pytest.mark.slow
@pytest.mark.skipif(not LATEXMK.exists() or not PDFTOTEXT.exists(), reason="TeX Live or poppler not installed")
@needs_tree
def test_the_esm_compiles_from_a_clean_copy_with_no_undefined_reference_and_prints_its_references_to_the_paper(tmp_path):
    """The ESM is compiled the way build.sh esm does it (latexmk in esm/ with ".." on the TeX, BibTeX and BST paths), in a copy without any build
    product, so that a stale .aux cannot hide an undefined label. The log goes through build.sh checklog, which refuses undefined references and
    citations, multiply defined labels and a log that asks for another run."""
    work = tmp_path / "paper"
    shutil.copytree(PAPER, work, ignore=shutil.ignore_patterns(*SKIPPED_FOLDERS, "*.pdf", "*.aux", "*.log", "*.out", "*.fls", "*.fdb_latexmk",
                                                               "*.bbl", "*.blg", "*.synctex.gz"))
    env = {**os.environ, "TEXINPUTS": "..:", "BSTINPUTS": "..:", "BIBINPUTS": "..:"}
    r = subprocess.run([str(LATEXMK), "-pdf", "-interaction=nonstopmode", "-halt-on-error", "ESM_1.tex"], cwd=work / "esm", env=env,
                       capture_output=True, text=True, timeout=1500)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-1000:]
    log_path = work / "esm" / "ESM_1.log"
    log = log_path.read_text(errors="replace")
    assert "Package crossref Error" not in log
    checked = subprocess.run([str(BUILD_SH), "checklog", str(log_path)], capture_output=True, text=True, timeout=60)
    assert checked.returncode == 0, checked.stderr[-2000:]
    text = squash(subprocess.run([str(PDFTOTEXT), "-layout", str(work / "esm" / "ESM_1.pdf"), "-"], capture_output=True, text=True,
                                 timeout=300).stdout.replace("\u201c", '"').replace("\u201d", '"'))
    assert re.search(r'section "[^"]+" of the paper', text), "the ESM prints no reference to the paper"
    assert "Forward tests: full design" in text and "Size and power of the designation rule" in text           # the two moved sections are in it
