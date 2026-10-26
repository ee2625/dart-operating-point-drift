"""Polish of the submission files (journal rules that no earlier test reads).

Figure captions carry no punctuation at the end (docs/energy_systems_requirements.md line 37: "Captions begin Fig. N (bold), no
punctuation after the number or at the end"; paper/SUBMISSION_REQUIREMENTS_CHECK.md row R26). Table captions are unaffected. The check reads
the sources of the manuscript (the real paper/ folder), both branches of every \\ifTestA included, because the caption of a figure that is
printed only after Test A has run is in the source already. The first tests check the reader of the captions on small texts, so that the
check on the real folder cannot pass because the reader finds nothing."""
import re
from pathlib import Path

import pytest

PAPER = Path(__file__).resolve().parents[3] / "paper"
DEV = Path(__file__).resolve().parents[2]
needs_tree = pytest.mark.skipif(not (PAPER / "sections").exists(), reason="the manuscript folder is not next to dart_dev")

#: What a figure caption must not end with. The journal says "no punctuation at the end"; the full stop is the one that occurs.
END_PUNCTUATION = ".,;:!?"
#: The figures of the manuscript: the three in sections/ (the main text: the failure counts, the drift of the alert rate and the contrasts with the calendar-only rule,
#: plan 2C tasks T1, T1b and T2c), the block figure of the Test A part of the forward-test section (esm/forward_tests.tex, printed only after Test A has run; it was in
#: sections/06_sealed.tex before plan 2C task T2a) and the four other figures of Online Resource 1 (the two old figures of Sections 4 and 5 moved there in task T2c).
KNOWN_FIGURE_LABELS = {"fig:fail", "fig:drift", "fig:contrast", "fig:activity", "fig:selc-forest", "fig:sealed-blocks", "fig:analog-blocks", "fig:decomposition"}


# ------------------------------------------------------------------------------------------------------ the reader of captions
def _live(text: str) -> str:
    """The text without its LaTeX comments (a commented caption is not in the paper)."""
    return re.sub(r"(?<!\\)%.*$", "", text, flags=re.M)


def _group(text: str, start: int) -> str:
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


def figure_captions(text: str) -> list[tuple[str, str]]:
    """(label, caption) of every figure environment in ``text`` (also figure*; \\caption[short]{long} gives the long one). A figure
    environment without a caption gives an empty caption, which is not a violation but is visible to the caller."""
    out = []
    for env in re.finditer(r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}", _live(text), flags=re.S):
        body = env.group(1)
        cap = re.search(r"\\caption\s*(?:\[[^\]]*\])?\s*\{", body)
        caption = _group(body, cap.end() - 1) if cap else ""
        label = re.search(r"\\label\{([^}]*)\}", body)
        out.append((label.group(1) if label else "", caption))
    return out


def ends_with_punctuation(caption: str) -> bool:
    """Whether the caption ends with a mark that the journal forbids. The closing braces of inline groups are skipped, so that
    \\emph{x.} ends with a full stop and \\bandWide{} (a macro that prints an interval) does not."""
    s = caption.rstrip()
    while s and s[-1] in "} \t\n":
        s = s[:-1]
    return bool(s) and s[-1] in END_PUNCTUATION


def violations(paper: Path) -> tuple[list[str], set[str]]:
    """(the figures of the sources of ``paper`` whose caption ends with a mark, the labels of all figures found)."""
    files = [paper / "main.tex"]
    for folder in ("sections", "appendices", "esm", "generated/tables"):
        files += sorted((paper / folder).glob("*.tex"))
    bad, found = [], set()
    for path in files:
        if not path.is_file():
            continue
        for label, caption in figure_captions(path.read_text(encoding="utf-8")):
            found.add(label)
            if ends_with_punctuation(caption):
                bad.append(f"{path.relative_to(paper)}: {label or '(no label)'} ends with {caption.rstrip()[-40:]!r}")
    return bad, found


def test_the_reader_finds_the_caption_of_a_figure_and_judges_its_last_character():
    figure = ("\\begin{figure}[t]\n\\centering\n\\genfigure{Fig1}{\\textwidth}\n"
              "\\caption{Error $|\\log_{10} R|$ per node; shaded band: the band \\bandWide{}%s}\\label{fig:x}\n\\end{figure}\n")
    assert figure_captions(figure % "")[0][0] == "fig:x"
    assert not ends_with_punctuation(figure_captions(figure % "")[0][1])                  # ends with \bandWide{}: the macro prints the interval
    assert ends_with_punctuation(figure_captions(figure % ".")[0][1])                     # \bandWide{}.
    assert not ends_with_punctuation(figure_captions(figure % " \n")[0][1])               # white space at the end does not count
    for mark in (".", ";", ":", ",", "!", "?"):
        assert ends_with_punctuation("Activity ratio of every node by block" + mark)
        assert ends_with_punctuation("Activity ratio of every node by \\emph{block" + mark + "}")      # inside a group is still the end
        assert ends_with_punctuation("Activity ratio of every node by block" + mark + "  \n")
    for text in ("Activity ratio of every node by block", "Nodes outside the band (the median in parentheses)", "A ratio of 0.5 to 2",
                 "Target of 5\\%", "A caption that ends with a macro \\bandWide{}", "Dollars (USD)"):
        assert not ends_with_punctuation(text), text
    assert not ends_with_punctuation("")                                                  # no caption: nothing to judge


def test_the_reader_handles_nested_braces_optional_arguments_comments_and_other_environments():
    text = "\n".join([
        "\\begin{table}\\caption{A table caption ends with a full stop.}\\label{tab:t}\\end{table}",         # table captions are unaffected
        "% \\begin{figure}\\caption{Commented out.}\\label{fig:gone}\\end{figure}",
        "\\begin{figure*}[t]\\caption[Short.]{Long caption with \\textbf{nested {braces}} and \\{an escaped brace\\} and 5\\% done.}"
        "\\label{fig:wide}\\end{figure*}",
        "\\begin{figure}[t]\\caption{No label here}\\end{figure}",
        "\\begin{figure}[t]\\genfigure{Fig9}{3cm}\\end{figure}",
    ])
    got = figure_captions(text)
    assert [label for label, _ in got] == ["fig:wide", "", ""]                            # the table and the comment are not figures
    assert got[0][1] == "Long caption with \\textbf{nested {braces}} and \\{an escaped brace\\} and 5\\% done."      # the long form, whole
    assert got[1][1] == "No label here" and got[2][1] == ""
    assert ends_with_punctuation(got[0][1]) and not ends_with_punctuation(got[1][1]) and not ends_with_punctuation(got[2][1])
    assert ends_with_punctuation("Ends with a nested group \\textbf{nested {braces.}}")
    assert not ends_with_punctuation("Ends with an escaped brace \\{x\\}") and not ends_with_punctuation("Ends with 5\\%")


def test_the_check_fails_on_a_tree_with_a_full_stop_and_passes_when_it_is_removed(tmp_path):
    paper = tmp_path / "paper"
    (paper / "sections").mkdir(parents=True)
    (paper / "esm").mkdir()
    (paper / "main.tex").write_text("\\documentclass{article}\n")
    figure = "\\begin{figure}\\genfigure{Fig1}{3cm}\\caption{Activity ratio of every node%s}\\label{fig:%s}\\end{figure}\n"
    (paper / "sections" / "04_audit.tex").write_text(figure % (" by block.", "a"))
    (paper / "esm" / "ESM_1.tex").write_text(figure % (" by block", "b") + "\\begin{table}\\caption{Ends with a stop.}\\end{table}\n")
    bad, found = violations(paper)
    assert found == {"fig:a", "fig:b"} and len(bad) == 1 and bad[0].startswith("sections/04_audit.tex: fig:a ends with")
    (paper / "sections" / "04_audit.tex").write_text(figure % (" by block", "a"))
    assert violations(paper) == ([], {"fig:a", "fig:b"})


# ------------------------------------------------------------------------------------------------------ the manuscript
@needs_tree
def test_no_figure_caption_of_the_manuscript_ends_with_a_full_stop_or_another_mark():
    """R26: sections/04_audit.tex (twice), sections/05_economic.tex, esm/ESM_1.tex (four times) and the caption of the Test A figure in
    esm/forward_tests.tex (the former sections/06_sealed.tex). Table captions are not read."""
    bad, found = violations(PAPER)
    assert KNOWN_FIGURE_LABELS <= found, f"the reader did not find {sorted(KNOWN_FIGURE_LABELS - found)}; found {sorted(found)}"
    assert bad == [], "\n".join(["a figure caption ends with a mark that the journal forbids (Req L37):"] + bad)


def test_the_generators_write_no_figure_environment_and_no_caption_note_ends_with_a_full_stop():
    """The code that makes the figures writes no caption into any .tex file (the captions are in the section files, which the check
    above reads). The one text it writes for captions is the list of caption notes in generated/figures/INDEX.md, kept in the same form.
    A generator that starts to write a figure environment must be added to the check above: this test names it."""
    writers = [p.name for p in sorted((DEV / "es").glob("paper_*.py"))
               if re.search(r"begin\{\{?figure\*?\}\}?", p.read_text(encoding="utf-8"))]       # \\begin{figure}, also as an f-string
    assert writers == [], f"{writers} write a figure environment: check its captions with figure_captions()"
    from es.paper_figures import _caption_notes
    notes = _caption_notes()
    assert len(notes) >= 5 and [n for n in notes if ends_with_punctuation(n)] == []
