from pathlib import Path


def _paper(tmp_path: Path) -> Path:
    p = tmp_path / "paper"
    for d in ("sections", "appendices", "generated/tables", "generated/figures"):
        (p / d).mkdir(parents=True)
    (p / "author_inputs.json").write_text('{"title": "Operating-point drift", "short_title": "Drift"}')
    (p / "references.bib").write_text("@article{holm1979simple, title={x}}\n@article{unused2000, title={y}}\n")
    (p / "sections" / "00_abstract.tex").write_text("\\abstract{" + " ".join(["word"] * 160) + "}\n\\keywords{a, b, c, d}\n")
    (p / "sections" / "90_backmatter.tex").write_text("\n".join(
        "\\bmhead{" + h + "}" for h in ("Funding", "Competing interests", "Ethics approval", "Data availability",
                                        "Code availability", "Author contribution", "Prior dissemination")) + "\n")
    return p


def test_numbers_rule_allows_years_ids_and_macros_only(tmp_path):
    from es.paper_lint import lint_numbers
    ok = ("Across 2023--2026 and 2026-H1 the P1 cutoff failed at \\failLevYY\\ of \\nNodes\\ nodes "
          "(Table~\\ref{tab:x}; \\cite{holm1979simple}; D11, H-B1, E5, G6, Q3, 2025-07-01).")
    assert lint_numbers(ok, "s.tex") == []
    bad = "failed at 13 of 33 nodes, with $\\alpha = 5\\%$."
    msgs = [f.message for f in lint_numbers(bad, "s.tex")]
    assert len(msgs) == 3 and all("hand-typed number" in m for m in msgs)
    assert lint_numbers("1-Wasserstein distance % lint:allow 1", "s.tex") == []        # per token (Ruling 61): name the number
    assert lint_numbers("\\includegraphics[width=0.8\\textwidth]{Fig1.pdf}\\begin{tabular}{p{3cm}r}", "s.tex") == []
    assert lint_numbers("Online Resource 1 (ESM\\_1.pdf) lists $|\\log_{10} R|$ for the \\nNodes-node panel on 2025-07-01.", "s.tex") == []


def test_declarations_rule_asks_for_every_required_heading_and_names_the_one_that_is_missing(tmp_path):
    """The Statements and Declarations headings of the journal; the last, on prior dissemination (Ruling 133: it replaces the heading that
    named a preregistration), is asked for like the others."""
    from es.paper_lint import REQUIRED_HEADINGS, lint_all
    p = _paper(tmp_path)
    back = p / "sections" / "90_backmatter.tex"
    base = back.read_text()

    def declarations():
        return [(f.severity, f.message) for f in lint_all(p, strict=False) if f.rule == "declarations"]

    assert declarations() == [] and REQUIRED_HEADINGS[-1] == "Prior dissemination" and len(REQUIRED_HEADINGS) == 7
    for heading in REQUIRED_HEADINGS:
        back.write_text(base.replace("\\bmhead{" + heading + "}", ""))
        assert declarations() == [("error", f"missing \\bmhead{{{heading}...}}")], heading
    back.write_text(base.replace("\\bmhead{Prior dissemination}", "\\bmhead{Preregistration and prior dissemination}"))
    assert declarations() == [("error", "missing \\bmhead{Prior dissemination...}")]           # the old heading no longer does
    back.write_text(base.replace("\\bmhead{Ethics approval}", "\\bmhead{Ethics approval and consent to participate}"))
    assert declarations() == []                                                                 # a heading may continue after the required words


def test_reversal_rule_and_headings(tmp_path):
    from es.paper_lint import lint_all
    p = _paper(tmp_path)
    (p / "sections" / "01_introduction.tex").write_text(
        "% BEGIN CONTRIBUTIONS\n\\item The 2026 reversal is our headline.\n% END CONTRIBUTIONS\n\\paragraph{Deep.}\n")
    rules = {f.rule for f in lint_all(p, strict=False)}
    assert {"reversal", "headings"} <= rules


def test_strict_flags_pending_and_counts(tmp_path):
    from es.paper_lint import lint_all
    p = _paper(tmp_path)
    (p / "generated" / "macros.tex").write_text(
        "\\newcommand{\\PENDINGA}{\\textbf{[pending: Test A]}}\n\\newcommand{\\x}{\\PENDINGA}\n\\newcommand{\\y}{\\PENDINGB}\n")
    (p / "generated" / "tables" / "tab_x.tex").write_text("a & \\PENDINGDEV \\\\\n")
    (p / "sections" / "01_introduction.tex").write_text("Cited \\cite{nokey}. \\gentable{tab_missing}\n")
    findings = lint_all(p, strict=True)
    f = {(x.rule, x.severity) for x in findings}
    assert ("pending", "error") in f and ("bib", "error") in f and ("generated", "error") in f
    assert ("pending-B", "info") in f                       # Test B may stay pending at submission
    assert any(x.rule == "pending" and "tab_x.tex" in x.message for x in findings)   # table cells are scanned too
    assert not any(x.rule == "pending" and "\\PENDINGA is pending" in x.message for x in findings)  # definitions are not findings
    f2 = {x.rule for x in lint_all(p, strict=False)}
    assert "pending" not in f2


def test_abstract_count_stops_at_closing_brace(tmp_path):
    from es.paper_lint import lint_all
    p = _paper(tmp_path)
    (p / "sections" / "00_abstract.tex").write_text(
        "\\abstract{" + " ".join(["word"] * 140) + " \\text{\\macro} end}\n\\keywords{" + ", ".join(["k"] * 20) + "}\n")
    msgs = [x.message for x in lint_all(p, strict=True) if x.rule == "abstract"]
    assert msgs == ["abstract has 143 words (150-250 required)"]      # keywords are not counted; macros count as words
    kw = [x for x in lint_all(p, strict=True) if x.rule == "keywords"]
    assert kw and "20 keywords" in kw[0].message


def test_year_allowlist_is_2015_to_2030():
    from es.paper_lint import lint_numbers

    def flagged(text):
        return [f.message for f in lint_numbers(text, "s.tex")]

    for text, token in (("We use 2000 replicates.", "2000"), ("The panel has 1990 rows.", "1990"),
                        ("Data stop in 2014 here.", "2014"), ("Windows end in 2031 here.", "2031")):
        msgs = flagged(text)
        assert len(msgs) == 1 and f"hand-typed number '{token}" in msgs[0], (text, msgs)   # outside 2015-2030
    assert flagged("The 2026 sample starts in 2023 and Test B runs in 2027, 2015 and 2030 bound the window.") == []
    assert flagged("The panel starts on 2023-04-13 and covers 2023--2026, 2015-2030, 2026-H1 and 2025-Q3.") == []
    assert flagged("It ended in 2026.") == [] and flagged("Test B runs in January 2027.") == []   # a year may end a sentence
    assert flagged("Data run from 2023 to 2026.") == []
    assert flagged("A decimal like 2026.5 is a number.") != []                                   # but a decimal is not a year
    assert flagged("An ISO date outside the window, 1999-12-31, is a hand-typed number too.") != []


def test_typed_pending_sentinel_is_an_error_in_strict_and_info_otherwise(tmp_path):
    from es.paper_lint import lint_all, main
    p = _paper(tmp_path)
    (p / "sections" / "00_abstract.tex").write_text(
        "\\abstract{\\PENDINGDEV\\ " + " ".join(["word"] * 160) + "}\n\\keywords{a, b, c, d}\n")
    (p / "generated" / "macros.tex").write_text("\\newcommand{\\PENDINGDEV}{x}\n")           # a missing macros.tex is an error
    strict = [x for x in lint_all(p, strict=True) if "PENDINGDEV" in x.message]
    lax = [x for x in lint_all(p, strict=False) if "PENDINGDEV" in x.message]
    assert [(x.rule, x.severity, x.line) for x in strict] == [("pending", "error", 1)]
    assert [(x.rule, x.severity, x.line) for x in lax] == [("pending", "info", 1)]
    assert main(["--paper-dir", str(p)]) == 0 and main(["--paper-dir", str(p), "--strict"]) == 1


def test_typed_pending_scan_covers_appendices_and_esm_but_not_comments_or_main(tmp_path):
    from es.paper_lint import lint_all
    p = _paper(tmp_path)
    (p / "esm").mkdir()
    (p / "sections" / "04_audit.tex").write_text("% a comment may name \\PENDINGDEV\nPlain text.\n")
    (p / "sections" / "06_sealed.tex").write_text("Test B is pending: \\PENDINGB.\n")
    (p / "appendices" / "A_ledger.tex").write_text("\\section{Ledger}\nThe value is \\PENDINGA.\n")
    (p / "esm" / "forward_tests.tex").write_text("\\section{Forward tests}\nA section that the ESM inputs: \\PENDINGDEV.\n")      # no \\begin{document}
    (p / "esm" / "ESM_1.tex").write_text(
        "\\documentclass{sn-jnl}\n\\begin{document}\nA cell reads \\PENDINGAUTHOR.\n\\end{document}\n")
    (p / "main.tex").write_text(
        "\\documentclass{sn-jnl}\n\\begin{document}\nA \\PENDINGCONST is not scanned here.\n\\end{document}\n")

    def seen(strict):
        return {(Path(x.path).name, x.rule, x.severity) for x in lint_all(p, strict=strict) if "PENDING" in x.message}

    assert seen(True) == {("A_ledger.tex", "pending", "error"), ("ESM_1.tex", "pending", "error"), ("forward_tests.tex", "pending", "error"),
                          ("06_sealed.tex", "pending-B", "info")}       # Test B may stay pending at submission
    assert seen(False) == {("A_ledger.tex", "pending", "info"), ("ESM_1.tex", "pending", "info"), ("forward_tests.tex", "pending", "info"),
                           ("06_sealed.tex", "pending-B", "info")}


def test_esm_text_is_linted_from_begin_document_on_and_main_tex_is_skipped(tmp_path):
    from es.paper_lint import lint_all
    p = _paper(tmp_path)
    (p / "esm").mkdir()
    preamble = ("\\documentclass[pdflatex,sn-mathphys-num]{sn-jnl}\n"                               # line 1
                "\\newcommand{\\gentable}[1]{\\IfFileExists{../generated/tables/#1.tex}{}{}}\n"     # line 2: boilerplate
                "\\author*[1]{\\fnm{A} \\sur{B}}\n"                                                 # line 3: boilerplate
                "\\begin{document}\n")                                                              # line 4
    (p / "esm" / "ESM_1.tex").write_text(
        preamble + "\\maketitle\n\\gentable{esm_ledger_legacy}\nRuns use 2000 replicates in 2026.\n\\end{document}\n")
    (p / "main.tex").write_text("\\newcommand{\\z}[1]{#1}\nThe panel has 13 nodes.\n")

    def mine():
        return {(Path(x.path).name, x.line, x.rule) for x in lint_all(p, strict=False)
                if Path(x.path).name in ("ESM_1.tex", "main.tex") and x.rule in ("numbers", "newcommand")}

    assert mine() == {("ESM_1.tex", 7, "numbers")}      # the preamble and main.tex are template boilerplate: not linted
    (p / "esm" / "ESM_1.tex").write_text(preamble + "\\newcommand{\\myval}{13}\n\\end{document}\n")
    assert mine() == {("ESM_1.tex", 5, "newcommand"), ("ESM_1.tex", 5, "numbers")}   # but the text obeys the text rules


def test_the_sections_that_the_esm_inputs_are_linted_whole_and_the_key_of_a_cross_reference_is_a_name(tmp_path):
    """Plan 2C task T2a: the forward-test design is esm/forward_tests.tex, a file that ESM_1.tex inputs and that has no \\begin{document}. It obeys
    the text rules from its first line (the paper's sections did, before the move), and the key of \\esmref and \\mainref is a name like a label."""
    from es.paper_lint import lint_all, lint_numbers
    p = _paper(tmp_path)
    (p / "esm").mkdir()
    (p / "esm" / "forward_tests.tex").write_text(
        "\\section{Forward tests}\nThe design has 13 nodes, see \\mainref{data-stats-2} and \\esmref{sealed-3}.\n\\newcommand{\\x}{y}\n"
        "\\paragraph{Deep.}\nIt cites \\cite{nokey}.\n")
    found = {(x.line, x.rule) for x in lint_all(p, strict=False) if Path(x.path).name == "forward_tests.tex"}
    assert found == {(2, "numbers"), (3, "newcommand"), (4, "headings"), (5, "bib")}              # the file's own line numbers
    messages = [x.message for x in lint_all(p, strict=False) if x.rule == "numbers"]
    assert len(messages) == 1 and "'13'" in messages[0]                                             # the keys with digits are not hand-typed numbers
    only = lint_numbers("See \\esmref{sealed-3} and \\mainref{data-stats-2}, and \\esmref{a} for 13 nodes.", "s.tex")
    assert len(only) == 1 and "'13'" in only[0].message                                             # the text next to the keys is still text


# ---------------------------------------------------------------------------------------------- fix round 1 (Ruling 61)
def test_missing_macros_tex_is_an_error(tmp_path):
    from es.paper_lint import lint_all, main
    p = _paper(tmp_path)                                         # no generated/macros.tex: strict used to pass vacuously
    for strict in (True, False):
        found = [x for x in lint_all(p, strict=strict) if x.rule == "generated" and "macros.tex" in x.message]
        assert [x.severity for x in found] == ["error"], strict
    assert main(["--paper-dir", str(p), "--strict"]) == 1 and main(["--paper-dir", str(p)]) == 1
    (p / "generated" / "macros.tex").write_text("\\newcommand{\\PENDINGA}{x}\n")
    assert not [x for x in lint_all(p, strict=True) if x.rule == "generated" and "macros.tex" in x.message]


def test_generated_and_bib_rules_cover_the_esm(tmp_path):
    from es.paper_lint import lint_all
    p = _paper(tmp_path)
    (p / "esm").mkdir()
    (p / "esm" / "ESM_1.tex").write_text(
        "\\documentclass{sn-jnl}\n\\begin{document}\nCites \\cite{nokey}.\n\\gentable{esm_missing}\n"
        "\\genfigure{FigX}{5cm}\n\\end{document}\n")
    for strict, severity in ((True, "error"), (False, "warning")):
        found = {(x.rule, x.severity) for x in lint_all(p, strict=strict) if x.path.endswith("ESM_1.tex")}
        assert ("bib", "error") in found and ("generated", severity) in found
    messages = [x.message for x in lint_all(p, strict=True) if x.path.endswith("ESM_1.tex")]
    assert any("esm_missing" in m for m in messages) and any("FigX" in m for m in messages)
    assert any("nokey" in m for m in messages)
    (p / "generated" / "tables" / "esm_missing.tex").write_text("x\n")
    assert not any("esm_missing" in x.message for x in lint_all(p, strict=True))


def test_contribution_markers_are_required_in_strict_once_the_introduction_exists(tmp_path):
    from es.paper_lint import lint_all

    def hits(strict):
        return [x for x in lint_all(p, strict=strict) if x.rule == "contributions"]

    p = _paper(tmp_path)
    assert not hits(True)                                        # no introduction yet
    intro = p / "sections" / "01_introduction.tex"
    intro.write_text("\\section{Introduction}\nText only.\n")
    assert [x.severity for x in hits(True)] == ["error"] and [x.severity for x in hits(False)] == ["warning"]
    intro.write_text("% END CONTRIBUTIONS\n\\item x\n% BEGIN CONTRIBUTIONS\n")      # the wrong way round
    assert [x.severity for x in hits(True)] == ["error"]
    intro.write_text("% BEGIN CONTRIBUTIONS\n\\item A finding.\n% END CONTRIBUTIONS\n")
    assert not hits(True)


def test_sentinel_regex_catches_a_digit_after_the_name_but_not_a_longer_macro(tmp_path):
    from es.paper_lint import lint_all
    p = _paper(tmp_path)
    (p / "sections" / "04_audit.tex").write_text("The value is \\PENDINGA1 and \\PENDINGDEVX is another macro.\n")
    found = [x for x in lint_all(p, strict=True) if x.rule == "pending" and x.path.endswith("04_audit.tex")]
    assert len(found) == 1 and "\\PENDINGA is typed" in found[0].message


def test_brace_groups_are_swallowed_per_command():
    from es.paper_lint import lint_numbers

    def flagged(text):
        return [f.message for f in lint_numbers(text, "s.tex")]

    for text in ("\\multicolumn{2}{c}{13 nodes}", "\\multirow{2}{*}{13}", "\\href{https://x.y}{13 nodes}"):
        msgs = flagged(text)
        assert len(msgs) == 1 and "'13'" in msgs[0], (text, msgs)           # the text argument is text, not a name
    assert flagged("\\multicolumn{2}{p{3cm}}{nodes} \\multirow{2}{*}{nodes} \\href{https://x.y/2}{link}") == []
    assert flagged("\\genfigure{Fig1}{0.8\\textwidth} \\hspace{1em} \\setlength{\\tabcolsep}{4pt} "
                   "\\input{generated/tables/t1}") == []
    assert flagged("\\citep[see][p.~12]{k2020} and \\url{https://x.y/13}") == []


def test_numbers_allowed_by_pattern():
    from es.paper_lint import lint_numbers
    for text in ("the 24H2-25H2 and 25H1-26H1 pairs, 2026-H1 and 24H2$\\to$25H2",
                 "a SHA-256 hash, UTF-8 text and ISO 8601 dates, ISO~8601",
                 "see Section~\\ref{sec:a1}, \\autoref{fig:b2}, \\cref{tab:c3}, \\nameref{sec:d4} and \\label{x5}",
                 "a row end\\\\[2pt] and \\cline{2-4}",
                 "the node $x_1$, $y_{10}$ and $R_{t-1}$",
                 "the analog quarters AQ3 and AQ4"):
        assert lint_numbers(text, "s.tex") == [], text


def test_size_and_index_arguments_of_common_layout_commands_are_not_numbers():
    from es.paper_lint import lint_numbers

    def flagged(text):
        return [f.message for f in lint_numbers(text, "s.tex")]

    assert flagged("\\cmidrule(lr){2-3} \\cmidrule[0.5pt](lr){2-3} \\cmidrule{4-5}") == []      # booktabs: [width](trim){a-b}
    assert flagged("\\footnotemark[1] and \\footnotetext[2]{A note.}") == []                      # table-note letters
    assert flagged("\\scalebox{0.9}{\\gentable{tab_x}} \\resizebox{\\textwidth}{!}{x} \\parbox[t]{3cm}{x} "
                   "\\rotatebox{90}{x} \\raisebox{2pt}{x} \\makebox[2cm]{x} \\rule{1cm}{0.4pt}") == []
    for text in ("\\parbox{3cm}{13 nodes}", "\\footnotetext[1]{13 nodes}", "\\scalebox{0.9}{13 nodes}",
                 "\\resizebox{\\textwidth}{!}{13 nodes}"):
        msgs = flagged(text)
        assert len(msgs) == 1 and "'13'" in msgs[0], (text, msgs)           # the text argument is still text


def test_environment_size_and_column_arguments_are_not_numbers():
    from es.paper_lint import lint_numbers

    def flagged(text):
        return [f.message for f in lint_numbers(text, "s.tex")]

    for text in ("\\begin{subfigure}[t]{0.45\\textwidth}", "\\begin{multicols}{2}", "\\begin{thebibliography}{99}",
                 "\\begin{adjustbox}{max width=0.9\\textwidth}", "\\begin{wrapfigure}[10]{r}{0.4\\textwidth}",
                 "\\begin{tabularx}{\\textwidth}{lp{3cm}X}", "\\begin{wraptable}{r}{5cm}", "\\begin{tabular*}{\\textwidth}{p{3cm}r}",
                 "\\begin{minipage}[t]{0.5\\textwidth}", "\\begin{longtable}{p{3cm}rr}", "\\begin{table*}[!t]",
                 # wrapfig: \begin{wrapfigure}[lines]{placement}[overhang]{width}: the overhang sits between the two groups
                 "\\begin{wrapfigure}{r}[0pt]{0.4\\textwidth}", "\\begin{wrapfigure}[10]{r}[0pt]{0.4\\textwidth}",
                 "\\begin{wraptable}{r}[12pt]{5cm}", "\\begin{wrapfigure}{r}[0.5\\columnwidth]{0.4\\textwidth}"):
        assert flagged(text) == [], text
    for text in ("\\begin{subfigure}{0.45\\textwidth} 13 nodes", "\\begin{multicols}{2} 13 nodes",
                 "\\begin{wrapfigure}{r}{0.4\\textwidth} 13 nodes", "\\begin{wrapfigure}{r}[0pt]{0.4\\textwidth} 13 nodes",
                 "\\begin{wraptable}[10]{r}[12pt]{5cm} 13 nodes"):
        msgs = flagged(text)
        assert len(msgs) == 1 and "'13'" in msgs[0], (text, msgs)           # the text after the arguments is still linted
    # only the argument between the two groups is an overhang: a bracket after the width is text
    msgs = flagged("\\begin{wrapfigure}{r}{0.4\\textwidth}[13 nodes]")
    assert len(msgs) == 1 and "'13'" in msgs[0], msgs


def test_lint_allow_is_per_token():
    from es.paper_lint import lint_numbers
    assert lint_numbers("1-Wasserstein distance % lint:allow 1", "s.tex") == []
    msgs = [f.message for f in lint_numbers("13 nodes on 21 July % lint:allow 21", "s.tex")]
    assert len(msgs) == 1 and "'13'" in msgs[0]                             # only the named token is exempt
    assert len(lint_numbers("1-Wasserstein distance % lint:allow metric name", "s.tex")) == 1   # a reason is not a token
    assert lint_numbers("about 2,000 and 3.5 here % lint:allow 2,000 3.5", "s.tex") == []
    assert lint_numbers("on 21 and 22 July % lint:allow 21, 22.", "s.tex") == []
    assert len(lint_numbers("13 nodes % lint:allow 14", "s.tex")) == 1      # another number does not help


def test_numbers_are_reported_without_trailing_punctuation():
    from es.paper_lint import lint_numbers
    msgs = [f.message for f in lint_numbers("It has 13. Then 2,000, and 3.5.", "s.tex")]
    assert [m.split("'")[1] for m in msgs] == ["13", "2,000", "3.5"]


def test_lint_reads_utf8_whatever_the_locale(tmp_path):
    """The generated macros hold the multiplication sign (the strata texts) and the prose may hold accents: the lint must
    not depend on the locale's encoding (LC_ALL=C without UTF-8 mode reads ASCII)."""
    import os
    import subprocess
    import sys

    import es
    p = _paper(tmp_path)
    (p / "generated" / "macros.tex").write_text(
        "\\newcommand{\\strataPrimaryText}{node × month}\n", encoding="utf-8")
    (p / "sections" / "01_introduction.tex").write_text("Création of \\strataPrimaryText\\ strata.\n", encoding="utf-8")
    dart_dev = Path(es.__file__).resolve().parents[1]
    code = ("import sys\nfrom pathlib import Path\nfrom es.paper_lint import lint_all\n"
            "print(len(lint_all(Path(sys.argv[1]), strict=False)))\n")
    env = {**os.environ, "LC_ALL": "C", "LANG": "C", "PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0",
           "DART_PROJECT_ROOT": os.environ.get("DART_PROJECT_ROOT", str(dart_dev.parent))}
    r = subprocess.run([sys.executable, "-c", code, str(p)], cwd=dart_dev, env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-1500:]


# ---------------------------------------------------------------------------------------------- a submission before Test A (25 Oct)
def _pending_paper(tmp_path: Path, switches: str) -> Path:
    p = _paper(tmp_path)
    (p / "generated" / "macros.tex").write_text(
        "\\newcommand{\\PENDINGA}{\\textbf{[pending: Test A]}}\n\\newcommand{\\resA}{\\PENDINGA}\n"
        "\\newcommand{\\fieldByAuthor}{\\PENDINGAUTHOR}\n\\newcommand{\\resB}{\\PENDINGB}\n" + switches)
    (p / "generated" / "tables" / "tab_sealed_A.tex").write_text("a & \\PENDINGA \\\\\n")
    return p


def test_pending_ok_a_accepts_test_a_placeholders_only_while_test_a_is_absent(tmp_path):
    from es.paper_lint import lint_all
    p = _pending_paper(tmp_path, "\\TestAfalse\n\\TestBfalse\n")
    found = [(x.rule, x.severity, x.message) for x in lint_all(p, strict=True, pending_ok=("A",))
             if x.rule.startswith("pending")]
    assert [x[:2] for x in found if "resA" in x[2] or "tab_sealed_A" in x[2]] == [("pending-A", "info")] * 2   # the macro and the cell
    assert ("pending", "error") in {x[:2] for x in found if "fieldByAuthor" in x[2]}                          # the author's stay errors
    assert ("pending-B", "info") in {x[:2] for x in found}
    plain = {(x.rule, x.severity) for x in lint_all(p, strict=True) if "resA" in x.message}
    assert plain == {("pending", "error")}                                              # without the option the default is unchanged
    # Test A present, or Test A did not run: a pending Test A placeholder is an error whatever the option says
    for n, switches in enumerate(("\\TestAtrue\n", "\\TestAtrue\n\\TestAdidnotruntrue\n")):
        q = _pending_paper(tmp_path / f"present{n}", switches)
        assert any(x.rule == "pending" and x.severity == "error" and "resA" in x.message
                   for x in lint_all(q, strict=True, pending_ok=("A",)))
    # not strict: the option changes nothing
    assert not any(x.rule == "pending" for x in lint_all(p, strict=False, pending_ok=("A",)))


def test_pending_ok_a_does_not_excuse_a_placeholder_typed_into_the_text(tmp_path):
    from es.paper_lint import lint_all
    p = _pending_paper(tmp_path, "\\TestAfalse\n")
    (p / "sections" / "01_introduction.tex").write_text("The result is \\PENDINGA here.\n")
    typed = [x for x in lint_all(p, strict=True, pending_ok=("A",)) if "typed in the text" in x.message]
    assert [(x.rule, x.severity) for x in typed] == [("pending", "error")]


def test_pending_ok_command_line_takes_only_a_and_gives_exit_zero_when_nothing_else_is_pending(tmp_path):
    import pytest
    from es.paper_lint import main
    p = _paper(tmp_path)
    (p / "generated" / "macros.tex").write_text(
        "\\newcommand{\\PENDINGA}{x}\n\\newcommand{\\resA}{\\PENDINGA}\n\\TestAfalse\n")
    assert main(["--paper-dir", str(p), "--strict"]) == 1                                 # the default gate refuses a pending Test A
    assert main(["--paper-dir", str(p), "--strict", "--pending-ok", "A"]) == 0
    for bad in ("B", "AUTHOR", "A,B"):
        with pytest.raises(SystemExit):
            main(["--paper-dir", str(p), "--strict", "--pending-ok", bad])
