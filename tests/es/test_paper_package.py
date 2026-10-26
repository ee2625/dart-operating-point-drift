from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest

SWITCHES_OFF = "\\newif\\ifTestA\n\\TestAfalse\n\\newif\\ifTestAdidnotrun\n\\TestAdidnotrunfalse\n"
SWITCHES_RAN = "\\newif\\ifTestA\n\\TestAtrue\n\\newif\\ifTestAdidnotrun\n\\TestAdidnotrunfalse\n"
SWITCHES_NOT_RUN = "\\newif\\ifTestA\n\\TestAtrue\n\\newif\\ifTestAdidnotrun\n\\TestAdidnotruntrue\n"


def _fig(name, label):
    return f"\\begin{{figure}}[t]\n\\centering\n\\genfigure{{{name}}}{{\\textwidth}}\n\\caption{{Caption of {name}}}\\label{{{label}}}\n\\end{{figure}}\n"


#: The generated figures that the main text cites since plan 2C (tasks T1, T1b, T2c), in the order of the text, and the four older ones, which are printed in Online Resource 1.
MAIN_FIGURES = ("FigFail", "FigDrift", "FigContrast")
OLD_FIGURES = ("Fig1", "Fig2", "Fig3", "Fig4")


def _paper(tmp_path: Path, switches: str = SWITCHES_OFF, sections: dict | None = None, main_inputs=("sections/04_audit", "sections/05_economic", "sections/06_sealed"),
           eps=MAIN_FIGURES + OLD_FIGURES) -> Path:
    """A paper folder shaped like the real one: the files the package needs, a main text that cites FigFail, FigDrift and FigContrast and, behind \\ifTestA,
    the block figure Fig4 (a stand-in: since plan 2C task T2a the real block figure is in esm/forward_tests.tex, not in the main text; the stand-in
    keeps the packager's rules for a figure behind a switch under test), and an Online Resource that cites the four older figures, which the two tasks T2a and T2c
    moved there (Fig1 and Fig2, the node-level figure and the effect figure of Sections 4 and 5, are Online Resource 1 only now). Each file holds its own
    name, so that a test can tell which generated file ended up under which name."""
    from es.paper_package import REQUIRED
    paper = tmp_path / "paper"
    for rel in REQUIRED:
        p = paper / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"content of {rel}")
    sections = sections or {
        "sections/04_audit.tex": _fig("FigFail", "fig:fail") + _fig("FigDrift", "fig:drift"),
        "sections/05_economic.tex": _fig("FigContrast", "fig:contrast"),
        "sections/06_sealed.tex": ("\\ifTestA\n\\ifTestAdidnotrun Test A did not run.\\else\n" + _fig("Fig4", "fig:sealed-blocks")
                                   + "\\fi\n\\else Test A is pending.\\fi\n"),
    }
    for rel, text in sections.items():
        (paper / rel).parent.mkdir(parents=True, exist_ok=True)
        (paper / rel).write_text(text)
    (paper / "main.tex").write_text(
        "\\documentclass{article}\n\\newcommand{\\genfigure}[2]{\\IfFileExists{generated/figures/#1.pdf}{#2}{}}\n\\input{generated/macros}\n"
        "\\begin{document}\n% \\genfigure{Fig3}{\\textwidth} a commented citation is not a citation\n"
        + "".join(f"\\input{{{rel}}}\n" for rel in main_inputs) + "\\end{document}\n")
    (paper / "generated" / "macros.tex").write_text(switches)
    (paper / "esm" / "ESM_1.tex").write_text("\\begin{document}\n" + _fig("Fig1", "fig:activity") + _fig("Fig2", "fig:selc-forest") + _fig("Fig4", "fig:analog-blocks")
                                              + _fig("Fig3", "fig:decomposition") + "\\end{document}\n")
    figures = paper / "generated" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for name in MAIN_FIGURES + OLD_FIGURES:
        (figures / f"{name}.pdf").write_text(f"pdf of {name}")
    for name in eps:
        (figures / f"{name}.eps").write_text(f"eps of {name}")
    (paper / "generated" / "tables").mkdir(parents=True, exist_ok=True)
    (paper / "generated" / "tables" / "tab_panel.tex").write_text("x")
    return paper


def _build(paper: Path):
    from es.paper_package import build_package
    out = build_package(paper, paper / "submission")
    return out, zipfile.ZipFile(out)


def _figures_folder(z: zipfile.ZipFile) -> dict[str, str]:
    """{file name: content} of figures/ in the zip, without the map."""
    return {Path(n).name: z.read(n).decode() for n in z.namelist() if n.startswith("figures/") and not n.endswith("FIGURE_MAP.txt")}


def test_package_lists_required_members(tmp_path):
    from es.paper_package import REQUIRED, build_package
    paper = _paper(tmp_path)
    out = build_package(paper, tmp_path / "paper" / "submission")
    names = set(zipfile.ZipFile(out).namelist())
    assert {"main.tex", "main.pdf", "main.bbl", "sn-jnl.cls", "sn-mathphys-num.bst", "references.bib", "figures/Fig1.eps",
            "generated/figures/Fig1.pdf", "generated/tables/tab_panel.tex", "esm/ESM_1.pdf", "cover_letter.md", "crossref.tex"} <= names
    assert set(REQUIRED) <= names
    assert (out.parent / "MANIFEST.sha256").exists()


def test_the_zip_holds_crossref_tex_and_every_tex_file_of_the_esm_and_stops_without_the_key_table(tmp_path):
    """Plan 2C task T2a: main.tex and the ESM both input crossref.tex (the \\esmref and \\mainref key table), so that a zip without it would not
    compile; the sections of the ESM (esm/forward_tests.tex, esm/designation_rule_study.tex) travel with ESM_1.tex through the glob."""
    from es.paper_package import GLOBS, REQUIRED, build_package
    assert "crossref.tex" in REQUIRED and "esm/*.tex" in GLOBS
    paper = _paper(tmp_path)
    (paper / "esm" / "forward_tests.tex").write_text("%% a note\n\\section{Forward tests: full design}\n")
    (paper / "esm" / "designation_rule_study.tex").write_text("\\section{Size and power of the designation rule}\n")
    (paper / "crossref.tex").write_text("%% how to add a key\n\\esmkey{sealed}{Forward tests: full design}\n")
    z = zipfile.ZipFile(build_package(paper, paper / "submission"))
    assert {"crossref.tex", "esm/ESM_1.tex", "esm/forward_tests.tex", "esm/designation_rule_study.tex"} <= set(z.namelist())
    assert z.read("crossref.tex").decode() == "\\esmkey{sealed}{Forward tests: full design}\n"      # a .tex file loses its full-line comments
    assert z.read("esm/forward_tests.tex").decode() == "\\section{Forward tests: full design}\n"
    (paper / "crossref.tex").unlink()
    with pytest.raises(FileNotFoundError, match="crossref.tex"):
        build_package(paper, paper / "submission")


# ------------------------------------------------------------------------------------------------------ figure file names (journal: Fig1, Fig2, ... in text order)
def test_without_test_a_the_main_text_cites_three_figures_and_the_zip_holds_three_eps_files(tmp_path):
    """The failure counts (generated as FigFail), the drift of the alert rate (FigDrift) and the contrasts with the calendar-only rule (FigContrast) are the
    figures of the main text, in that order; the four older figures (Fig1 to Fig4: the node-level figure, the effect figure, the decomposition and the block
    figure) are in Online Resource 1 only, and the block figure is in the main text only after Test A has run (a stand-in here). With Test A absent, figures/
    holds the three figures of the main text under their positions, Fig1.eps to Fig3.eps, and none of the older ones."""
    out, z = _build(_paper(tmp_path))
    assert _figures_folder(z) == {"Fig1.eps": "eps of FigFail", "Fig2.eps": "eps of FigDrift", "Fig3.eps": "eps of FigContrast"}
    names = set(z.namelist())
    assert {f"generated/figures/{n}.pdf" for n in MAIN_FIGURES + OLD_FIGURES} <= names        # the PDF twins keep their names: the compile is unchanged
    assert not any(n.endswith(".eps") and n not in {"figures/Fig1.eps", "figures/Fig2.eps", "figures/Fig3.eps"} for n in names)
    assert z.read("generated/figures/Fig4.pdf") == b"pdf of Fig4" and z.read("generated/figures/Fig1.pdf") == b"pdf of Fig1"      # the old Fig1 is the PDF twin of its own name


def test_after_test_a_the_block_figure_is_the_fourth_figure_and_is_named_fig4(tmp_path):
    """Test A has run: the main text prints the block figure (generated as Fig4) fourth, as Fig. 4. The upload file is Fig4.eps, the content of
    generated/figures/Fig4.eps; the decomposition (generated Fig3, Online Resource 1 only) is not in figures/, and the generated Fig3.eps is not the upload Fig3.eps
    (that one is FigContrast)."""
    out, z = _build(_paper(tmp_path, SWITCHES_RAN))
    assert _figures_folder(z) == {"Fig1.eps": "eps of FigFail", "Fig2.eps": "eps of FigDrift", "Fig3.eps": "eps of FigContrast", "Fig4.eps": "eps of Fig4"}
    assert z.read("generated/figures/Fig3.pdf") == b"pdf of Fig3" and z.read("generated/figures/Fig4.pdf") == b"pdf of Fig4"
    text = z.read("figures/FIGURE_MAP.txt").decode()
    assert "Fig4.eps is generated/figures/Fig4.eps" in text and "fig:sealed-blocks" in text and "sections/06_sealed.tex" in text
    assert "Fig1.eps is generated/figures/FigFail.eps" in text and "Fig2.eps is generated/figures/FigDrift.eps" in text
    assert "Fig3.eps is generated/figures/FigContrast.eps" in text
    assert "Fig5.eps is" not in text                                                          # nothing is named Fig5 in a main text of four figures


def test_a_test_that_did_not_run_leaves_the_block_figure_out_of_the_main_text(tmp_path):
    """macros.tex sets TestA and TestAdidnotrun both true when the coverage rule failed; the figure is in the \\else branch of \\ifTestAdidnotrun."""
    out, z = _build(_paper(tmp_path, SWITCHES_NOT_RUN))
    assert _figures_folder(z) == {"Fig1.eps": "eps of FigFail", "Fig2.eps": "eps of FigDrift", "Fig3.eps": "eps of FigContrast"}


def test_the_map_names_the_state_of_the_switches_and_the_figures_that_are_not_in_the_main_text(tmp_path):
    out, z = _build(_paper(tmp_path))
    text = z.read("figures/FIGURE_MAP.txt").decode()
    assert text.isascii() and "TestA=false" in text and "TestAdidnotrun=false" in text
    assert "Fig1.eps is generated/figures/FigFail.eps" in text and "Fig2.eps is generated/figures/FigDrift.eps" in text
    assert "Fig3.eps is generated/figures/FigContrast.eps" in text
    assert "Fig4.eps is" not in text
    tail = text.split("Not in the main text", 1)[1]
    assert all(f"generated/figures/{name}.eps" in tail for name in OLD_FIGURES)               # named, so that nobody wonders where they went
    assert not any(f"generated/figures/{name}.eps" in tail for name in MAIN_FIGURES)
    assert "Online Resource 1" in tail


def test_the_names_follow_the_order_of_the_first_citation_not_the_number_of_the_generated_file(tmp_path):
    """The first figure of the text is Fig2 here: it becomes Fig1.eps. A figure cited twice counts once, at its first place; a citation in a
    comment, in the Online Resource and in the generated macros counts for nothing; \\input is followed in the order of the text, with or
    without the extension, and an \\input of a file that does not exist is skipped."""
    sections = {
        "sections/01_first.tex": "Intro. \\input{sections/02_nested.tex}\n" + _fig("Fig2", "fig:b") + "\\input{sections/99_missing}\n",
        "sections/02_nested.tex": _fig("Fig4", "fig:c") + "% " + _fig("Fig1", "fig:no").replace("\n", " ") + "\n",
        "sections/03_last.tex": _fig("Fig1", "fig:a") + _fig("Fig2", "fig:b-again"),
    }
    paper = _paper(tmp_path, SWITCHES_OFF, sections, main_inputs=("sections/01_first", "sections/03_last"))
    out, z = _build(paper)
    assert _figures_folder(z) == {"Fig1.eps": "eps of Fig4", "Fig2.eps": "eps of Fig2", "Fig3.eps": "eps of Fig1"}
    text = z.read("figures/FIGURE_MAP.txt").decode()
    assert "Fig1.eps is generated/figures/Fig4.eps" in text and "Fig3.eps is generated/figures/Fig1.eps" in text
    assert "generated/figures/Fig3.eps" in text.split("Not in the main text", 1)[1]


def test_a_switch_that_macros_tex_does_not_set_is_false_and_a_main_text_without_figures_has_no_eps_file(tmp_path):
    """\\newif leaves a switch false, so the Test A branch is skipped when macros.tex sets nothing; a main text that cites no figure
    puts no EPS file into figures/, and the map says so."""
    out, z = _build(_paper(tmp_path / "a", "no switches here\n"))
    assert _figures_folder(z) == {"Fig1.eps": "eps of FigFail", "Fig2.eps": "eps of FigDrift", "Fig3.eps": "eps of FigContrast"}
    out, z = _build(_paper(tmp_path / "b", sections={"sections/04_audit.tex": "No figure here.\n"}, main_inputs=("sections/04_audit",)))
    assert _figures_folder(z) == {} and "The main text cites no figure." in z.read("figures/FIGURE_MAP.txt").decode()
    assert {"generated/figures/Fig1.pdf", "generated/figures/Fig4.pdf"} <= set(z.namelist())     # the PDF twins are still all there


def test_the_manifest_hashes_what_is_in_the_zip_under_the_names_it_has_there(tmp_path):
    out, z = _build(_paper(tmp_path, SWITCHES_RAN))
    manifest = {line.split("  ", 1)[1]: line.split("  ", 1)[0] for line in (out.parent / "MANIFEST.sha256").read_text().splitlines()}
    for arc in ("figures/Fig1.eps", "figures/Fig2.eps", "figures/Fig3.eps", "figures/Fig4.eps", "figures/FIGURE_MAP.txt", "main.tex", "generated/figures/Fig4.pdf"):
        assert manifest[arc] == hashlib.sha256(z.read(arc)).hexdigest(), arc
    assert manifest["figures/Fig3.eps"] == hashlib.sha256(b"eps of FigContrast").hexdigest()
    assert manifest["figures/Fig4.eps"] == hashlib.sha256(b"eps of Fig4").hexdigest()
    assert "figures/Fig5.eps" not in manifest and set(manifest) == set(z.namelist())          # the manifest lists exactly the members


def test_a_cited_figure_without_its_eps_file_stops_the_package(tmp_path):
    """The names are positions in the text, so a missing file would shift every later name: the package is refused instead."""
    paper = _paper(tmp_path, eps=("FigFail", "FigContrast") + OLD_FIGURES)                    # FigDrift.eps is missing
    from es.paper_package import build_package
    with pytest.raises(FileNotFoundError, match="FigDrift.eps"):
        build_package(paper, paper / "submission")
    assert not (paper / "submission").exists() or not list((paper / "submission").glob("*.zip"))
    paper = _paper(tmp_path / "ok", eps=MAIN_FIGURES)                                         # a missing EPS of a figure that is not cited is fine
    assert _figures_folder(_build(paper)[1]) == {"Fig1.eps": "eps of FigFail", "Fig2.eps": "eps of FigDrift", "Fig3.eps": "eps of FigContrast"}


def test_the_real_main_text_cites_its_figures_in_the_order_that_the_package_names_them():
    """The real paper/ folder, read only: the figures of the main text, in the order of their first citation: the failure counts, the drift of the alert rate and
    the contrasts with the calendar-only rule, which the package names Fig1.eps, Fig2.eps and Fig3.eps (plan 2C task T2c). The four older figures (Fig1 to Fig4) and the
    block figure of Test A are not among them whatever the switches say: they are in Online Resource 1 (the block figure in the forward-test section)."""
    from es.paper_package import main_text_figures
    paper = Path(__file__).resolve().parents[3] / "paper"
    if not (paper / "sections").exists():
        pytest.skip("the manuscript folder is not next to dart_dev")
    found = main_text_figures(paper)
    cited = [c.name for c in found]
    assert cited == ["FigFail", "FigDrift", "FigContrast"], cited         # the old Fig1 and Fig2 are Online Resource 1 only since task T2c, Fig4 (the block figure of Test A) since task T2a
    assert [c.label for c in found] == ["fig:fail", "fig:drift", "fig:contrast"]
    assert [c.source for c in found] == ["sections/04_audit.tex", "sections/04_audit.tex", "sections/05_economic.tex"]


# ------------------------------------------------------------------------------------------------------ process comments of the uploaded sources (G3, item 14)
def _strip(text):
    from es.paper_package import strip_comment_lines
    return strip_comment_lines(text)


def test_a_comment_line_between_paragraphs_and_in_the_middle_of_one_goes_with_its_newline():
    assert _strip("First paragraph.\n\n% a note on the next paragraph\n\nSecond paragraph.\n") == "First paragraph.\n\n\nSecond paragraph.\n"
    assert _strip("The line before\n% a comment inside a paragraph\nthe line after\n") == "The line before\nthe line after\n"
    assert _strip("a\n\t  % indented, with blanks and a tab before the percent sign\nb\n") == "a\nb\n"        # the first NON-BLANK character decides
    assert _strip("% one\n%% two\n%\nbody\n") == "body\n"                                                      # a bare % and a run of comment lines too


def test_a_comment_line_is_removed_with_its_newline_in_every_line_ending_and_at_the_end_of_the_file():
    assert _strip("a\r\n% c\r\nb\r\n") == "a\r\nb\r\n"                                                          # the line ends of the file stay
    assert _strip("a\n% last line, no newline") == "a\n"                                                         # the newline of "a" stays: it is a's own
    assert _strip("% only comments\n% still only comments\n") == ""
    assert _strip("plain\ntext\n") == "plain\ntext\n" and _strip("") == ""
    once = _strip("% x\nkeep\n% y\n")
    assert _strip(once) == once == "keep\n"                                                                      # idempotent


def test_an_escaped_percent_sign_and_a_trailing_comment_are_left_alone():
    assert _strip("\\% a line that begins with an escaped percent sign\nkeep\n") == "\\% a line that begins with an escaped percent sign\nkeep\n"
    assert _strip("It is 50\\% of the days.\n") == "It is 50\\% of the days.\n"
    assert _strip("Text before the comment % and the comment\nnext\n") == "Text before the comment % and the comment\nnext\n"
    assert _strip("1 & 2 \\\\ % a comment after a line break\n") == "1 & 2 \\\\ % a comment after a line break\n"
    assert _strip("x\\verb|%|y\n% gone\n") == "x\\verb|%|y\n"                                                    # the argument of an inline \verb is text


def test_the_body_of_a_verbatim_environment_is_untouched_and_a_begin_in_a_comment_opens_nothing():
    block = "\\begin{verbatim}\n% a line of the model summary\n   % and an indented one\n\\end{verbatim}\n"
    assert _strip("before\n" + block + "% dropped\nafter\n") == "before\n" + block + "after\n"
    listing = "\\begin{lstlisting}[language=Python]\n% a comment of the program text\n\\end{lstlisting}\n"
    assert _strip(listing + "% dropped\n") == listing
    one_line = "\\begin{verbatim}x\\end{verbatim}\n"
    assert _strip(one_line + "% dropped\ntail\n") == one_line + "tail\n"                                       # it opens and closes on one line
    wrapped = "\\begin{small}\\begin{verbatim}\n% kept\n\\end{verbatim}\\end{small}\n% dropped\n"
    assert _strip(wrapped) == "\\begin{small}\\begin{verbatim}\n% kept\n\\end{verbatim}\\end{small}\n"        # as es.paper_tables writes tab_mixed
    assert _strip("% \\begin{verbatim}\n% dropped too\nkeep\n") == "keep\n"                                      # a commented \begin opens no environment
    assert _strip("a % \\begin{verbatim}\n% dropped\nb\n") == "a % \\begin{verbatim}\nb\n"                       # nor does a trailing comment


def test_the_zip_holds_the_tex_files_without_their_full_line_comments_and_the_tree_keeps_them(tmp_path):
    paper = _paper(tmp_path)
    commented = "%% process note, first line\n\\section{Audit}\nText before % trailing note\n% a comment line\n\\% an escaped percent sign\n"
    (paper / "sections" / "04_audit.tex").write_text(commented + _fig("Fig1", "fig:activity"))
    (paper / "generated" / "tables" / "tab_panel.tex").write_text("%% GENERATED by es.paper_tables; do not edit.\n\\begin{tabular}{l}\nx\\\\\n\\end{tabular}\n")
    (paper / "cover_letter.md").write_text("<!-- notes -->\n% a percent sign in a markdown file is not a comment\n")
    before = {p: p.read_bytes() for p in paper.rglob("*") if p.is_file()}
    out, z = _build(paper)
    for name in z.namelist():
        if name.endswith(".tex"):
            assert not any(line.lstrip().startswith("%") for line in z.read(name).decode().splitlines()), name   # no full-line comment is uploaded
    section = z.read("sections/04_audit.tex").decode()
    assert section.startswith("\\section{Audit}\nText before % trailing note\n\\% an escaped percent sign\n") and "\\genfigure{Fig1}" in section
    assert z.read("generated/tables/tab_panel.tex").decode().startswith("\\begin{tabular}{l}")
    assert "\\input{sections/04_audit}" in z.read("main.tex").decode() and "a commented citation" not in z.read("main.tex").decode()
    assert z.read("cover_letter.md").decode() == "% a percent sign in a markdown file is not a comment\n"                    # the notes to the author go; a percent sign is text here
    after = {p: p.read_bytes() for p in paper.rglob("*") if p.is_file() and "submission" not in p.parts}
    assert {p: b for p, b in before.items() if "submission" not in p.parts} == after                                # the tree is read, not changed
    assert (paper / "sections" / "04_audit.tex").read_text().startswith("%% process note")


def test_with_markers_a_trailing_comment_of_ours_goes_and_every_other_comment_stays():
    from es.paper_package import strip_comment_lines
    s = lambda t: strip_comment_lines(t, markers=True)
    assert s("The text % CONTROLLER: add the sign here (Ruling 110)\nnext\n") == "The text\nnext\n"
    assert s("The text\t%  AUTHOR: confirm this\nnext\n") == "The text\nnext\n"
    assert s("The text (see below) % lint:allow 14 22 17\nnext\n") == "The text (see below)\nnext\n"
    assert s("The text %% CONTROLLER: two signs\n") == "The text\n"
    assert s("a\r\nThe text % CONTROLLER: x\r\nb\r\n") == "a\r\nThe text\r\nb\r\n"                                   # the line end of the file stays
    # left alone: a bare percent sign that glues two lines, a comment of anyone else's, an escaped one, the argument of \\verb
    assert s("glued%\nline\n") == "glued%\nline\n"
    assert s("The text % an explanation for the reader of the source\n") == "The text % an explanation for the reader of the source\n"
    assert s("It is 50\\% of the days. % AUTHOR: check\n") == "It is 50\\% of the days.\n"
    assert s("50\\% AUTHOR: not a comment\n") == "50\\% AUTHOR: not a comment\n"
    assert s("x\\verb|% AUTHOR|y\n") == "x\\verb|% AUTHOR|y\n"
    # not inside a verbatim block; full-line comments still go, and without the option nothing trailing goes
    block = "\\begin{verbatim}\nkept % CONTROLLER: inside the block\n\\end{verbatim}\n"
    assert s(block + "after % AUTHOR: x\n") == block + "after\n"
    assert strip_comment_lines("The text % CONTROLLER: x\n") == "The text % CONTROLLER: x\n"
    assert s(s("a % AUTHOR: x\n% y\nb\n")) == s("a % AUTHOR: x\n% y\nb\n") == "a\nb\n"                                   # idempotent


def test_the_cover_letter_in_the_zip_loses_its_html_comments_and_the_other_markdown_files_are_kept(tmp_path):
    from es.paper_package import build_package
    paper = _paper(tmp_path)
    letter = "Dear Editors,\n\nText of the letter.\n\nSincerely,\nE.E.\n\n<!-- AUTHOR NOTES (delete before pasting)\n1. A note.\n   Over two lines.\n-->\n"
    (paper / "cover_letter.md").write_text(letter)
    (paper / "CORRECTIONS_TO_PREPRINT.md").write_text("# Corrections\n\n| 1 | a | b |\n")
    z = zipfile.ZipFile(build_package(paper, paper / "submission"))
    assert z.read("cover_letter.md").decode() == "Dear Editors,\n\nText of the letter.\n\nSincerely,\nE.E.\n"
    assert z.read("CORRECTIONS_TO_PREPRINT.md").decode() == "# Corrections\n\n| 1 | a | b |\n"
    assert "AUTHOR NOTES" in (paper / "cover_letter.md").read_text()                                                  # the tree keeps its notes
    manifest = (paper / "submission" / "MANIFEST.sha256").read_text()
    import hashlib
    assert f"{hashlib.sha256(z.read('cover_letter.md')).hexdigest()}  cover_letter.md" in manifest                  # the manifest hashes what the zip holds


def test_a_bib_file_in_the_zip_loses_the_comment_lines_between_its_entries_and_nothing_inside_an_entry(tmp_path):
    from es.paper_package import build_package, strip_bib_comments
    bib = ("%% Bibliography for the ICAIF submission\n%% added by the controller\n\n@article{a2020,\n  title = {Fifty percent \\% of {Things}},\n"
           "% a line inside an entry stays\n  year = {2020}\n}\n% between entries\n  % indented, between entries\n@book{b2021,\n  title = {B}\n}\n")
    assert strip_bib_comments(bib) == ("\n@article{a2020,\n  title = {Fifty percent \\% of {Things}},\n% a line inside an entry stays\n  year = {2020}\n}\n"
                                       "@book{b2021,\n  title = {B}\n}\n")
    assert strip_bib_comments(strip_bib_comments(bib)) == strip_bib_comments(bib) and strip_bib_comments("") == ""
    assert strip_bib_comments("@a{k,\n  note = {an escaped \\{ brace},\n}\n% gone\n") == "@a{k,\n  note = {an escaped \\{ brace},\n}\n"
    paper = _paper(tmp_path)
    (paper / "references.bib").write_text(bib)
    z = zipfile.ZipFile(build_package(paper, paper / "submission"))
    assert z.read("references.bib").decode() == strip_bib_comments(bib) and "controller" not in z.read("references.bib").decode()
    assert "%% Bibliography" in (paper / "references.bib").read_text()                                                # the tree keeps them


def test_the_zip_requires_the_corrections_to_the_preprint_and_not_the_internal_revision_log(tmp_path):
    from es.paper_package import REQUIRED, build_package
    assert "CORRECTIONS_TO_PREPRINT.md" in REQUIRED and "REVISION_LOG.md" not in REQUIRED
    paper = _paper(tmp_path)
    (paper / "REVISION_LOG.md").write_text("the internal log: not uploaded")
    z = zipfile.ZipFile(build_package(paper, paper / "submission"))
    assert "CORRECTIONS_TO_PREPRINT.md" in z.namelist() and "REVISION_LOG.md" not in z.namelist()
    (paper / "CORRECTIONS_TO_PREPRINT.md").unlink()
    with pytest.raises(FileNotFoundError, match="CORRECTIONS_TO_PREPRINT.md"):
        build_package(paper, paper / "submission")


PDFLATEX = Path("/Library/TeX/texbin/pdflatex")
PDFTOTEXT = Path("/opt/homebrew/bin/pdftotext")


@pytest.mark.skipif(not (PDFLATEX.exists() and PDFTOTEXT.exists()), reason="pdflatex and pdftotext are not installed")
def test_a_document_without_its_comment_lines_compiles_to_the_same_text(tmp_path):
    """The claim behind the stripper: a comment line takes its newline with it in LaTeX, so the document is the same. Comment lines between
    paragraphs, inside one, after a control word, and a verbatim block that keeps its percent signs."""
    import subprocess
    body = ("%% a note before the preamble\n\\documentclass{article}\n% a note in the preamble\n\\newcommand{\\foo}{FOO}\n\\begin{document}\n"
            "First paragraph, with a line\n% a comment inside the paragraph\nthat goes on, and a macro \\foo\n% after a control word\nand text.\n\n"
            "% between paragraphs\n\n  % indented, between paragraphs\nSecond paragraph, 50\\% of it, % a trailing comment\nends here.\n"
            "\\begin{verbatim}\n% kept in the verbatim block\n\\end{verbatim}\n% last note\n\\end{document}\n")
    from es.paper_package import strip_comment_lines
    body = body.replace("First paragraph, with a line\n", "First paragraph, with a line % CONTROLLER: a note to us\nsecond line of it % lint:allow 7\n")
    texts = {}
    for name, source in (("with", body), ("without", strip_comment_lines(body, markers=True))):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "doc.tex").write_text(source)
        run = subprocess.run([str(PDFLATEX), "-interaction=nonstopmode", "-halt-on-error", "doc.tex"], cwd=folder, capture_output=True, text=True, timeout=120)
        assert run.returncode == 0, run.stdout[-1500:]
        texts[name] = subprocess.run([str(PDFTOTEXT), "-layout", str(folder / "doc.pdf"), "-"], capture_output=True, text=True, timeout=60).stdout
    assert texts["with"] == texts["without"] and "kept in the verbatim block" in texts["with"] and "a comment inside" not in texts["with"]
    assert "% a comment" not in _strip(body).replace("% a trailing comment", "") and "50% of it" in texts["with"]
