"""Plan 2C, task T2c: Sections 4 (Audit results) and 5 (Economic value) were cut from about 9 and 5 pages to about 5 and 4.5. The three new figures stand in them
(FigFail as fig:fail, FigDrift as fig:drift, FigContrast as fig:contrast), a hand-written table gives the headline economics (tab:headline), the old figures and the big
tables moved to Online Resource 1 with their numbers, and the paragraphs that were cut are in esm/audit_economic_details.tex and the supplementary tables of esm/ESM_1.tex.
These tests keep the move honest and say what it must not do:

* tab:headline has four rows, in the order AQ3, AQ4, AQ1, AQ2, and every number in it is a macro that exists in generated/macros.tex (no number is typed);
* each of fig:fail, fig:drift and fig:contrast is introduced by a \\ref before its float and its environment holds the figure it should, with the words that the
  caption must say; no caption ends with a mark (test_paper_polish.py reads that too);
* every macro that left Sections 4 and 5 and appears nowhere else in the main text is in Online Resource 1 (no number was lost with the cut text), and the moved
  paragraphs begin as they did in the paper;
* every table and figure that moved is defined in the ESM and in no file of the paper, and no \\ref of the paper points at it; the two old figures say nothing of Test A;
* the words that the wording rules forbid for the calendar-only rule and the rolling cutoff ("better", "worse", "equivalent", ...) are in no sentence of the two sections
  except in the sentences that this file names, one by one, because a ruling on them is the controller's;
* the caps of the brief (the ranking paragraph, the monitor paragraph, the third paragraph of 4.2, Section 5.5) hold; the failing-node counts are said to be
  descriptive once in Section 4 (and in the caption of FigFail) and never in Section 5; no inline correction tag stays in the paper; the sentences are short;
* the key of the section that received "Target rate and band" moved from the table of the paper to that of the ESM, and the headings "Audit results" and "Economic value"
  that the moved ESM text points at are still the headings of the paper.
They read the real paper/ folder, as test_paper_g1.py, test_paper_g3.py and test_paper_t2b.py do."""
import re
import statistics
from pathlib import Path

import pytest

PAPER = Path(__file__).resolve().parents[3] / "paper"
needs_tree = pytest.mark.skipif(not (PAPER / "sections").exists(), reason="the manuscript folder is not next to dart_dev")

#: The macros of the text that Sections 4 and 5 lost, and that no other file of the main text uses (the list of the report of task T2c). Each is in Online Resource 1.
MACROS_NOW_IN_THE_ESM_ONLY = (
    "bhEconomicDecRejections", "bhEconomicDecTests", "bhMonitorsEarlyRateRejections", "bhRejectionsDriftMonitors", "bhRejectionsEconomicRobustness", "bhTestsAuditRobustness",
    "bhTestsDriftMonitors", "bhTestsEconomicRobustness", "deltaAuprcNDartYY", "deltaAuprcNposDartYY", "eOneJHalfLevelAqThree", "eOneJHalfLevelHTwo", "eOneJHalfLevelYY",
    "failDartPostOpenAlphaOne", "failDartYYAlphaOne", "failLcCombLcFullQnew", "failLcCombLcFullQold", "failLcCombLcHalfQnew", "failLcCombLcHalfQold", "failLcLevLcFullQold",
    "failLcLevLcHalfQold", "failLcRelLcFullQold", "failLcRelLcHalfQold", "failLevPostOpenBandMid", "failLevYYAlphaOne", "failLevYYAlphaTwoHalf", "failLevYYBandMid",
    "failRelPostOpenBandMid", "failRelYYAlphaOne", "failRelYYAlphaTwoHalf", "failRelYYBandMid", "liftCalPostOpen", "liftCalYY", "medRCombPostOpen", "medRLcCombLcFullQold",
    "medRLcLevLcFullQold", "medRLcRelLcFullQold", "medRLevPostOpen", "medRRelPostOpen", "monDomainAucRhoMedian", "monDomainPlaceboMin", "monPsiRhoMedian", "monWoneRhoMedian",
    "opContrastPostOpenCaiso", "opContrastPostOpenIsone", "opContrastPostOpenMiso", "opContrastPostOpenNyiso", "opContrastYYCaiso", "opContrastYYIsone", "opContrastYYMiso",
    "opContrastYYNyiso", "pNinetyFiveDartTwentySixPooled", "pNinetyFiveDartYYpooled", "sharpeAlertedAqThree", "sharpeNotAlertedAqThree", "signPPostOpen", "signPYY",
    "snetAqFour", "snetAqFourCI", "snetAqOne", "snetAqOneCI", "snetAqThree", "snetAqThreeCI", "snetHTwo", "snetHTwoCI", "snetPostOpen", "snetPostOpenCI", "spearmanDriftTwentySix",
)

QUARTERS = (("AQ3", "AqThree"), ("AQ4", "AqFour"), ("AQ1", "AqOne"), ("AQ2", "AqTwo"))

#: Sentences of Sections 4 and 5 that carry a word of the wording rules and that stay, with the reason (task T2c, report section "Wording rules").
ALLOWED_EVALUATIVE = (
    # (Plan 2C task T3b: the first entry, the sentence on the timing and level values that said "did better" twice, is gone. The style pass reworded it to "gained more than" and
    # split it, as the report of task T2c proposed; the new sentences are asserted verbatim in test_the_timing_and_level_values_are_read_by_what_the_cutoffs_gained.)
    # "equivalence", negated: required rewording 3 and the caption of Figure contrast (task T1); as regular expressions since task T3b ("... a lack of evidence, not evidence of
    # equivalence" after the last style pass)
    r"An interval that contains zero (?:is|shows) a lack of evidence,? (?:and )?not (?:evidence of )?equivalence",
)
EVALUATIVE = re.compile(r"\b(?:better|worse|significantly|outperform\w*|superior|works|equivalen\w*|tracks|at least as)\b", re.I)


def _read(rel: str) -> str:
    return (PAPER / rel).read_text(encoding="utf-8")


def _live(text: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in text.splitlines())


def _flat(text: str) -> str:
    return " ".join(text.split())


def _group(text: str, start: int) -> str:
    """The content of the brace group that opens at text[start]."""
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
    raise ValueError("unbalanced braces")


def _macros(text: str) -> set[str]:
    return set(re.findall(r"\\([A-Za-z]+)(?=\{\}|\\|[^A-Za-z]|$)", text))


def _defined() -> set[str]:
    return set(re.findall(r"^\\newcommand\{\\([A-Za-z]+)\}", _read("generated/macros.tex"), flags=re.M))


def _environment(text: str, label: str, kind: str) -> str:
    """The live \\begin{kind} ... \\end{kind} that holds \\label{label}."""
    for m in re.finditer(r"\\begin\{%s\*?\}(.*?)\\end\{%s\*?\}" % (kind, kind), text, flags=re.S):
        if "\\label{%s}" % label in m.group(1):
            return m.group(1)
    raise AssertionError(f"no {kind} environment holds {label}")


def _caption(env: str) -> str:
    m = re.search(r"\\caption\s*(?:\[[^\]]*\])?\s*\{", env)
    assert m, "no caption"
    return _group(env, m.end() - 1)


def _typed_digits(text: str) -> list[str]:
    """The digits that a text types itself: what is left of it when the macros, the ids of the designs (P1 to P6, AQ1 to AQ4) and the years and window names of the
    lint (2025, 2024-H1, 2025-Q3) are taken out."""
    s = re.sub(r"\\(?:ref|esmref|mainref|label|cite)\{[^}]*\}", " ", text)
    s = re.sub(r"\\[A-Za-z]+(?:\{\})?", " ", s)
    s = re.sub(r"\bP[1-6]\b|\bAQ[1-4]F?\b|\b20(?:1[5-9]|2\d|30)(?:-(?:H[12]|Q[1-4]))?\b", " ", s)
    return re.findall(r"\d+", s)


def _paragraphs(text: str) -> list[str]:
    """The live paragraphs outside table and figure environments, one string each (white space squashed)."""
    text = re.sub(r"\\begin\{(table|figure)\*?\}.*?\\end\{\1\*?\}", "", _live(text), flags=re.S)
    return [_flat(b) for b in re.split(r"\n\s*\n", text) if _flat(b)]


def _paragraph(rel: str, start: str) -> str:
    hits = [p for p in _paragraphs(_read(rel)) if p.startswith(start)]
    assert len(hits) == 1, (rel, start, len(hits))
    return hits[0]


def _sentences(text: str) -> list[str]:
    flat = _flat(text)
    flat = re.sub(r"\b(?:e\.g|i\.e|et al|Fig|vs|No)\.", lambda m: m.group(0).replace(".", "<dot>"), flat)
    return [s.replace("<dot>", ".").strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\\(])", flat) if len(s.split()) >= 3]


def _words(sentence: str) -> int:
    """A macro counts as one word and an interval macro as two; a reference or a formula as one."""
    s = re.sub(r"\\[A-Za-z]*CI\{\}", " X X ", sentence)
    s = re.sub(r"\\(?:ref|esmref|mainref|cite|label)\{[^}]*\}", " X ", s)
    s = re.sub(r"\\[A-Za-z]+\{\}", " X ", s)
    s = re.sub(r"\$[^$]*\$", " X ", s)
    s = re.sub(r"\\[A-Za-z]+\*?", " ", s)
    return len(re.sub(r"[{}~\\]", " ", s).split())


def _body(rel: str) -> str:
    """The live text of a section file without its tables, figures and headings."""
    text = re.sub(r"\\begin\{(table|figure)\*?\}.*?\\end\{\1\*?\}", " ", _live(_read(rel)), flags=re.S)
    return re.sub(r"\\(?:sub)*section\{[^}]*\}(?:\\label\{[^}]*\})?", " ", text)


# ---------------------------------------------------------------------------------------------------- the headline table
@needs_tree
def test_the_headline_table_has_four_rows_in_the_order_of_the_quarters_and_no_typed_number():
    section = _live(_read("sections/05_economic.tex"))
    table = _environment(section, "tab:headline", "table")
    tabular = re.search(r"\\begin\{tabular\}\{([^}]*)\}(.*)\\end\{tabular\}", table, flags=re.S)
    assert tabular, "the table has no tabular"
    columns, body = tabular.groups()
    assert "|" not in columns and "\\hline" not in table and "\\vline" not in table, "booktabs rules only, no vertical rule"
    assert body.count("\\toprule") == 1 and body.count("\\midrule") == 1 and body.count("\\bottomrule") == 1
    # the two-line cells are nested tabulars: their own row end becomes a word, so that the rows of the table can be split at its row ends
    flat = re.sub(r"\\begin\{tabular\}\[t\]\{[^}]*\}(.*?)\\end\{tabular\}", lambda m: m.group(1).replace("\\\\", " NL "), body, flags=re.S)
    rows = [r for r in re.split(r"\\\\(?:\[[^\]]*\])?", flat.split("\\midrule", 1)[1].split("\\bottomrule", 1)[0]) if r.strip()]
    assert len(rows) == 4, rows
    defined = _defined()
    for row, (name, word) in zip(rows, QUARTERS):
        assert name in row.split("&")[0], (name, row)
        cells = [c.strip() for c in row.split("&")]
        assert len(cells) == 5, row
        used = _macros(row)
        assert used == {f"alertRatePOne{word}", f"net{word}", f"net{word}CI", f"selc{word}", f"selc{word}CI", f"esred{word}", f"esred{word}CI"}, (name, used)
        assert used <= defined, sorted(used - defined)
        # nothing else with a digit: the label is the only text of the row, and its digits are the quarter and the year
        rest = re.sub(r"\\[A-Za-z]+\{\}", " ", row)
        rest = re.sub(r"AQ[1-4]|20\d\d-Q[1-4]|NL|\\shortstack|\\textminus", " ", rest)
        assert not re.search(r"\d", rest), (name, rest)
    caption = _flat(_caption(table))
    for words in ("premium position, calibrated fees", "NET: loss avoided less profit given up", "not evidence of skill", "approximate", "AQ1 and AQ2 use the opened 2026-H1 data",
                  "\\alphaPct\\%", "\\ciLevel\\%"):
        assert words in caption, words
    assert "\\label{tab:headline}" in table


@needs_tree
def test_the_headline_table_reads_the_same_macros_as_the_old_primary_table_for_p1():
    """The macros of the headline table are the ones of the P1 rows of tab_econ_primary: the share of hours alerted and the NET, SELC and ESred of the four quarters."""
    defined = _defined()
    for _, word in QUARTERS:
        assert {f"alertRatePOne{word}", f"net{word}", f"net{word}CI", f"selc{word}", f"selc{word}CI", f"esred{word}", f"esred{word}CI"} <= defined
        assert _defined() >= {f"netPOne{word}"}
    macros = _read("generated/macros.tex")
    for _, word in QUARTERS:                                                          # the macro of the table is the macro of the text: the same value
        net = re.search(r"\\newcommand\{\\net%s\}\{(.*)\}" % word, macros).group(1)
        net_p1 = re.search(r"\\newcommand\{\\netPOne%s\}\{(.*)\}" % word, macros).group(1)
        assert net == net_p1, (word, net, net_p1)


# ---------------------------------------------------------------------------------------------------- the three figures
FIGURES = (("fig:fail", "FigFail", "sections/04_audit.tex"), ("fig:drift", "FigDrift", "sections/04_audit.tex"), ("fig:contrast", "FigContrast", "sections/05_economic.tex"))


@needs_tree
@pytest.mark.parametrize("label,name,rel", FIGURES)
def test_each_new_figure_is_introduced_by_a_reference_before_its_float_and_holds_its_own_file(label, name, rel):
    raw = _live(_read(rel))
    env = _environment(raw, label, "figure")
    assert f"\\genfigure{{{name}}}{{\\textwidth}}" in env, name
    refs = [m.start() for m in re.finditer(r"\\ref\{%s\}" % re.escape(label), raw)]
    assert refs, f"nothing in {rel} refers to {label}"
    assert min(refs) < raw.index("\\label{%s}" % label), f"{label} is referred to only after its float"
    # the generated file exists in the tree that the build reads
    assert (PAPER / "generated" / "figures" / f"{name}.pdf").is_file() and (PAPER / "generated" / "figures" / f"{name}.eps").is_file()


@needs_tree
def test_the_failure_figure_says_what_a_failure_is_the_two_panels_and_that_the_counts_carry_no_interval():
    caption = _flat(_caption(_environment(_live(_read("sections/04_audit.tex")), "fig:fail", "figure")))
    for words in ("A failure is a node whose realised alert rate over the deployment period lies outside \\bandWide{} times the target",
                  "failing nodes of the audit's \\nNodes{}", "The left panel is the headline", "The right panel is the robustness comparison",
                  "Every count is descriptive and carries no interval", "The counts are those of the audit table of \\esmref{supp-audit}"):
        assert words in caption, words
    assert "Table 3" not in caption and not re.search(r"Table\s*~?\s*\d", caption), "the audit table moved: the caption names it by the section of Online Resource 1"
    from es.paper_figures import HEAVY_FROM
    assert HEAVY_FROM == 5 and "a cell of five or more nodes" in caption, "the caption words the threshold of the heavy outline; the figure code holds the number"
    assert _typed_digits(caption) == [], "no number is typed in the caption"


@needs_tree
def test_the_drift_figure_caption_wires_every_value_as_a_macro_and_the_marker_for_the_reading_stands_after_it():
    section = _live(_read("sections/04_audit.tex"))
    caption = _flat(_caption(_environment(section, "fig:drift", "figure")))
    for macro in ("pFourWindow", "seedPrimary", "driftWindowWeeks", "driftWindowMinWeeks", "alphaPct", "nNodes", "bandWide"):
        assert f"\\{macro}" in caption, macro
    assert _typed_digits(caption) == [], "no number is typed in the caption"
    assert "Successive windows overlap in all but one of their weeks" in caption and "(a) the deployment year 2025 and (b) the first half of 2026" in caption
    # the introduction of the figure uses the words of the brief and nothing evaluative
    intro = _paragraph("sections/04_audit.tex", "Figure~\\ref{fig:drift} follows the combined price-history block")
    assert "the last \\driftWindowWeeks{} weeks (the weeks so far at the start)" in intro and "the window of P4 is \\pFourWindow{} days" in intro
    # the marker: the controller writes the reading there, with the drift macros; until then the comment line stands right after the figure
    raw = _read("sections/04_audit.tex")
    after = raw.split("\\label{fig:drift}", 1)[1].split("\\end{figure}", 1)[1]
    following = after.split("The quarterly analogues AQ3", 1)[0]
    assert "% T3: reading of Figure drift goes here" in following or "\\drift" in following, "neither the marker nor a reading stands between the figure and the quarterly analogues"


@needs_tree
def test_the_contrast_figure_caption_says_what_the_brief_requires_and_the_paper_introduces_it_in_section_5_4():
    caption = _flat(_caption(_environment(_live(_read("sections/05_economic.tex")), "fig:contrast", "figure")))
    for words in ("The four quarters are exploratory development analyses", "AQ1 and AQ2 use the opened 2026-H1 data", "Each contrast is paired at the same alert count at each node",
                  "approximate and conditional on the realised quarter", "no row is confirmatory", "\\ciLevel\\%", "day-bootstrap"):
        assert words in caption, words
    assert _typed_digits(caption) == [], "no number is typed in the caption"
    intro = _paragraph("sections/05_economic.tex", "The comparators are the calendar-only model and VOL")
    assert "Figure~\\ref{fig:contrast} draws the contrasts with the calendar-only model for the four quarters, the SELC contrast on the left and the ESred contrast on the right." in intro


@needs_tree
def test_no_caption_of_the_two_sections_or_of_the_two_old_figures_ends_with_a_mark_or_names_test_a():
    for rel, labels in (("sections/04_audit.tex", ("fig:fail", "fig:drift")), ("sections/05_economic.tex", ("fig:contrast",)), ("esm/ESM_1.tex", ("fig:activity", "fig:selc-forest"))):
        for label in labels:
            caption = _caption(_environment(_live(_read(rel)), label, "figure")).rstrip()
            assert caption[-1] not in ".,;:!?", (label, caption[-30:])
            if rel == "esm/ESM_1.tex":
                assert "Test A" not in caption, f"the caption of {label} was adapted: no mention of Test A (brief, 4.2 and 5.2)"


# ---------------------------------------------------------------------------------------------------- what moved
MOVED_LABELS = ("tab:block-audit", "tab:label-control", "tab:analog-blocks", "tab:monitors", "tab:alpha-band", "tab:holdout", "tab:econ-primary", "fig:activity", "fig:selc-forest")


@needs_tree
def test_every_table_and_figure_that_moved_is_in_the_esm_and_nowhere_in_the_paper_and_nothing_of_the_paper_refers_to_it():
    paper_files = sorted((PAPER / "sections").glob("*.tex")) + [PAPER / "main.tex"]
    esm_files = sorted((PAPER / "esm").glob("*.tex"))
    paper_text = "\n".join(_live(p.read_text(encoding="utf-8")) for p in paper_files)
    esm_text = "\n".join(_live(p.read_text(encoding="utf-8")) for p in esm_files)
    for label in MOVED_LABELS:
        assert f"\\label{{{label}}}" in esm_text, f"{label} is not defined in the ESM"
        assert f"\\label{{{label}}}" not in paper_text, f"{label} is still defined in the paper"
        assert not re.search(r"\\(?:ref|autoref|eqref)\{%s\}" % re.escape(label), paper_text), f"the paper refers to {label}"
    for table in ("tab_block_audit", "tab_label_control", "tab_analog_blocks", "tab_monitors", "tab_alpha_band", "tab_holdout", "tab_econ_primary"):
        assert f"\\gentable{{{table}}}" in esm_text and f"\\gentable{{{table}}}" not in paper_text, table
    for figure in ("Fig1", "Fig2"):                                              # the old node-level figure and the old effect figure
        assert f"\\genfigure{{{figure}}}" in esm_text and f"\\genfigure{{{figure}}}" not in paper_text, figure
    # the tables that stay: the policies table, the headline table and the paper's own panel and fee tables
    for label in ("tab:policies", "tab:headline", "tab:panel", "tab:fees"):
        assert f"\\label{{{label}}}" in paper_text, label
    # a label of the two sections that another file refers to is still defined
    defined = set(re.findall(r"\\label\{([^}]*)\}", paper_text))
    used = set()
    for p in paper_files:
        used |= set(re.findall(r"\\(?:ref|eqref|autoref)\{([^}]*)\}", _live(p.read_text(encoding="utf-8"))))
    used = {u for u in used if u.startswith(("sec:audit", "sec:econ", "fig:", "tab:"))}
    assert used <= defined | {"tab:fees"}, sorted(used - defined)
    assert {"sec:audit", "sec:audit-control", "sec:audit-blocks", "sec:audit-ranking", "sec:audit-adaptive", "sec:audit-monitors", "sec:audit-holdout",
            "sec:econ", "sec:econ-measures", "sec:econ-primary", "sec:econ-decomposition", "sec:econ-comparators", "sec:econ-robust"} <= defined
    assert "sec:audit-labels" not in defined and "sec:audit-target" not in defined          # the subsections that became the ranking paragraph and moved to the ESM


@needs_tree
def test_the_moved_paragraphs_begin_as_they_did_and_hold_every_macro_that_left_the_two_sections():
    details = _live(_read("esm/audit_economic_details.tex"))
    esm_main = _live(_read("esm/ESM_1.tex"))
    esm = _flat(details + "\n" + esm_main)
    starts = (
        "On the continuous error the 2024 to 2025 ordering is a soft tendency", "The audit measures the operating point and not accuracy",
        "The realised alert rate does not depend on the label", "The blocks order as on the longer analyses (Table~\\ref{tab:analog-blocks})",
        "The target of \\alphaPct\\% and the band \\bandWide{} are conventions of the audit", "Whether counts of these sizes could arise without any drift",
        "The preprint's 2026-H1 values", "Two further corrections concern the ranking of the earlier version",
        "Table~\\ref{tab:fee-position-signs} reports the signs of NET and SELC of P1", "Split by the position's spike exposure",
        "As a descriptive check of binary volatility management", "A drift monitor is useful if it flags the nodes that will fail without waiting for labels",
        "The AUROC of the domain classifier does not take the value one half without drift", "The stratified NET is negative, with an interval that excludes zero",
        "At the weight of one half the level effect on the value function is positive")
    for start in starts:
        assert start in esm, start
    for macro in MACROS_NOW_IN_THE_ESM_ONLY:
        assert f"\\{macro}{{}}" in esm, f"\\{macro} left Sections 4 and 5 and is in no text of the ESM: a number was lost with the cut text"
    # the moved text keeps the convention of the other moved sections: a reference into Online Resource 1 is a \ref, one into the paper a \mainref, never an \esmref
    assert "\\esmref" not in details and "\\esmref" not in esm_main
    assert "Online Resource 1" not in details
    ref_labels = set(re.findall(r"\\ref\{([^}]*)\}", details))
    esm_labels = set(re.findall(r"\\label\{([^}]*)\}", "".join(_live(p.read_text(encoding="utf-8")) for p in (PAPER / "esm").glob("*.tex"))))
    tables = set(re.findall(r"\\label\{([^}]*)\}", "".join(p.read_text(encoding="utf-8") for p in (PAPER / "generated" / "tables").glob("*.tex"))))
    assert ref_labels <= esm_labels | tables, sorted(ref_labels - esm_labels - tables)


@needs_tree
def test_the_new_esm_section_is_input_in_the_right_place_and_its_subsections_have_keys_and_labels():
    esm = _live(_read("esm/ESM_1.tex"))
    assert "\\input{audit_economic_details}" in esm
    assert esm.index("\\input{data_design_details}") < esm.index("\\input{audit_economic_details}") < esm.index("\\section{Supplementary tables of the data, audit and economic sections}")
    details = _live(_read("esm/audit_economic_details.tex"))
    table = _live(_read("crossref.tex"))
    wanted = (("audit-details", "Audit and economic results in detail", "esm:audit-details"), ("audit-order", "Ordering of the blocks, operators and volatility", "esm:audit-order"),
              ("audit-labels", "Precision, recall and the label control", "esm:audit-labels"), ("audit-analogs", "Blocks on the quarterly analogues", "esm:audit-analogs"),
              ("audit-target", "Target rate and band", "esm:audit-target"), ("audit-holdout", "The 2026-H1 holdout and the earlier version", "esm:audit-holdout"),
              ("econ-robust", "Economic robustness: fees, positions and targets", "esm:econ-robust"))
    for key, title, label in wanted:
        assert f"\\esmkey{{{key}}}{{{title}}}" in table, key
        assert re.search(r"\\(?:sub)?section\{%s\}\\label\{%s\}" % (re.escape(title), re.escape(label)), details), (title, label)
    for key, title in (("supp-audit", "Audit results"), ("supp-monitors", "Drift monitors"), ("supp-econ", "Economic value")):
        assert f"\\esmkey{{{key}}}{{{title}}}" in table and f"\\subsection*{{{title}}}" in esm, key
    # the key of "Target rate and band" moved from the table of the paper to the table of the ESM, and the size study points at the moved section with a \ref
    assert "\\mainkey{audit-target}" not in table and "\\esmkey{audit-target}" in table
    study = _live(_read("esm/designation_rule_study.tex"))
    assert "\\mainref{audit-target}" not in study and "Section~\\ref{esm:audit-target}" in study
    data = _live(_read("sections/03_data_design.tex"))
    assert "\\ref{sec:audit-target}" not in data and data.count("\\esmref{audit-target}") == 2
    # the headings that the moved ESM text points at are still the headings of the paper
    assert "\\mainkey{audit}{Audit results}" in table and "\\mainkey{econ}{Economic value}" in table
    assert "\\section{Audit results}\\label{sec:audit}" in _live(_read("sections/04_audit.tex"))
    assert "\\section{Economic value}\\label{sec:econ}" in _live(_read("sections/05_economic.tex"))
    # the discussion of the earlier text points at the table that moved with an \esmref
    assert "\\ref{tab:econ-primary}" not in _live(_read("sections/07_discussion.tex"))


# ---------------------------------------------------------------------------------------------------- the wording rules
@needs_tree
def test_the_two_sections_use_none_of_the_words_that_the_wording_rules_forbid_except_the_sentences_named_here():
    for rel in ("sections/04_audit.tex", "sections/05_economic.tex"):
        text = _flat(_live(_read(rel)))
        for sentence in re.split(r"(?<=[.;!?])\s+", text):
            if not EVALUATIVE.search(sentence):
                continue
            assert any(re.search(allowed, sentence) for allowed in ALLOWED_EVALUATIVE), (rel, sentence)
    # the allowed sentences are all still there, so that no exception is dead
    both = _flat(_live(_read("sections/04_audit.tex")) + _live(_read("sections/05_economic.tex")))
    for allowed in ALLOWED_EVALUATIVE:
        assert re.search(allowed, both), f"{allowed!r} is in no sentence: delete the exception"


@needs_tree
def test_the_timing_and_level_values_are_read_by_what_the_cutoffs_gained():
    """The sentence of Section 5.3 that says how to read a positive timing value and a positive level value compares the fixed and the rolling cutoff by their measured values
    ("gained more than"), with no evaluative word (it said "did better", which the wording rules forbid for the rolling cutoff and the calendar-only rule)."""
    econ = _flat(_live(_read("sections/05_economic.tex")))
    assert re.search(r"A positive timing value means that the fixed cutoff gained more than the floating cutoff at the same number of alerts", econ)
    assert re.search(r"A positive level value means that the rolling cutoff gained more at the alert count that P1 realised than at the target count", econ)
    assert "did better" not in econ and "did worse" not in econ


@needs_tree
def test_the_required_rewordings_are_in_the_text_verbatim():
    econ = _flat(_live(_read("sections/05_economic.tex")))
    assert ("The alerts are read for dominance in (NET, ESred): a policy that raises both dominates having no alerts for the position, and one that lowers NET and raises ESred "
            "trades forgone premium for a smaller tail loss, which the value function $J_\\lambda$ (\\esmref{econ-value}) prices.") in econ
    assert ("None of the paired intervals excludes zero: at the same number of alerts there is no demonstrated difference in the loss captured between the alerts and a "
            "calendar-only model or a trailing-volatility rule, and timing beyond the calendar is not found in development.") in econ
    # Required rewording 3, as its key phrases in their order (plan 2C task T3b: the controller inserted a sentence on the multiplicity of the contrasts between "Against VOL, no
    # interval excludes zero." and "An interval that contains zero is a lack of evidence ...", so the paragraph is no longer one string; each sentence of it is still in the text)
    keys = ("At matched activity the alerts therefore show no demonstrated improvement in tail protection over a calendar-only rule.",
            "The contrast is negative in AQ3, AQ4, AQ1, the 2024 to 2025 analysis, the second half-year analysis and the rehearsal of the pooled estimand "
            "(\\desredCalCountPooledRehearsal{}, \\desredCalCountPooledRehearsalCI{}).",
            "Its interval excludes zero in AQ4, where the tail effect of the calendar-only rule exceeds that of the alerts, and in the 2024 to 2025 analysis "
            "(\\desredCalCountYY{}, \\desredCalCountYYCI{}) and the second half-year analysis (\\desredCalCountHTwo{}, \\desredCalCountHTwoCI{}).",
            "It is positive only in AQ2 and the post-open 2026-H1 analysis (\\desredCalCountPostOpen{}), with intervals that contain zero.",
            "Against VOL, no interval excludes zero.")
    for key in keys:
        assert key in econ, key
    # the closing sentences were reworded by the last style pass ("... a lack of evidence, not evidence of equivalence. So the tail effect of Section 5.2 does not show that the alerts
    # protect beyond the calendar, and the paired SELC contrasts do not show that they add timing skill to it"): matched by their key phrases, in order
    closing = (r"An interval that contains zero is a lack of evidence,? (?:and )?not evidence of equivalence",
               r"the tail effect of Section~\\ref\{sec:econ-primary\} does not show that the alerts protect beyond the calendar",
               r"the paired SELC contrasts (?:give no evidence|do not show) that they add timing skill to it")
    spots = [re.search(c, econ).start() for c in closing]
    assert spots == sorted(spots) and spots[0] > econ.index(keys[-1])
    assert [econ.index(k) for k in keys] == sorted(econ.index(k) for k in keys)
    # and the facts of those sentences, against the macros
    def value(name):
        return float(re.search(r"^\\newcommand\{\\" + name + r"\}\{(.*)\}\s*$", _read("generated/macros.tex"), flags=re.M).group(1)
                     .replace("\\textminus{}", "-").replace("\\$", "").replace("{,}", ""))

    def interval(name):
        lo, hi = re.fullmatch(r"\[(.*), (.*)\]", re.search(r"^\\newcommand\{\\" + name + r"\}\{(.*)\}\s*$", _read("generated/macros.tex"), flags=re.M).group(1)).groups()
        return tuple(float(t.replace("\\textminus{}", "-").replace("\\$", "").replace("{,}", "")) for t in (lo, hi))

    analyses = ("AqThree", "AqFour", "AqOne", "AqTwo", "YY", "HTwo", "PooledRehearsal", "PostOpen")
    negative = [a for a in analyses if value(f"desredCalCount{a}") < 0]
    assert negative == ["AqThree", "AqFour", "AqOne", "YY", "HTwo", "PooledRehearsal"]               # "negative in AQ3, AQ4, AQ1, the 2024 to 2025 analysis, the second half-year analysis and the rehearsal"
    assert [a for a in analyses if a not in negative] == ["AqTwo", "PostOpen"]                       # "positive only in AQ2 and the post-open 2026-H1 analysis"
    excluding = [a for a in analyses if not interval(f"desredCalCount{a}CI")[0] < 0 < interval(f"desredCalCount{a}CI")[1]]
    assert excluding == ["AqFour", "YY", "HTwo"]                                                     # "Its interval excludes zero in AQ4 ... the 2024 to 2025 analysis ... the second half-year analysis"
    assert all(interval(f"desredVolCount{a}CI")[0] < 0 < interval(f"desredVolCount{a}CI")[1] for a in analyses)       # "Against VOL, no interval excludes zero"
    assert ("The alerts forgo more of the premium than the loss they avoid, which happens when the premium is positive on average, so NET alone is not a measure of skill") in econ
    for macro in ("desredCalCountPooledRehearsal", "desredCalCountPooledRehearsalCI", "desredCalCountYY", "desredCalCountYYCI", "desredCalCountHTwo", "desredCalCountHTwoCI",
                  "desredCalCountPostOpen"):
        assert macro in _defined(), macro
    for old in ("is better than none for the position", "at least as much tail protection", "The finding is plain", "do not capture more loss than a calendar-only model"):
        assert old not in econ, old


@needs_tree
def test_the_failing_node_counts_are_said_to_be_descriptive_once_in_section_4_and_never_in_section_5_and_no_correction_tag_stays():
    audit = _live(_read("sections/04_audit.tex"))
    econ = _live(_read("sections/05_economic.tex"))
    opening = _paragraph("sections/04_audit.tex", "This section reports the cutoff-drift audit")
    assert "The counts are descriptive: nodes within an operator co-move, and the development counts carry no interval (Section~\\ref{sec:data-stats})" in opening
    assert "listed in the file Corrections to the preprint that accompanies this manuscript" in opening and "none of it is a sealed result" not in opening
    assert "All of it is exploratory." in opening
    assert _flat(audit).count("carry no interval") == 1 and _flat(audit).count("carries no interval") == 1       # the opening paragraph and the caption of Figure fail
    assert "carry no interval" not in _flat(econ) and "carries no interval" not in _flat(econ) and "are descriptive" not in _flat(econ) and "is descriptive" not in _flat(econ)
    for text in (audit, econ):
        assert not re.search(r"correction[s]? D\d|\(D\d", text), "an inline correction tag stays in the paper; the accompanying file lists every correction with its section"
    assert "which is why the counts are reported as descriptive" not in audit
    # the ESM keeps the tags of the paragraphs that moved there
    assert "correction D7" in _live(_read("esm/audit_economic_details.tex")) and "(correction D5)" in _live(_read("esm/ESM_1.tex"))


@needs_tree
def test_no_comment_and_no_text_of_the_new_files_says_that_anything_is_registered_or_on_osf():
    for rel in ("sections/04_audit.tex", "sections/05_economic.tex", "esm/audit_economic_details.tex"):
        raw = _read(rel)                                                         # comments included: the brief asks for the carried-over ones to be corrected or deleted
        assert not re.search(r"\bOSF\b|Zenodo|regist(?:er|ered|ration)|preregist", raw, flags=re.I), (rel, re.search(r"\bOSF\b|Zenodo|regist\w*|preregist\w*", raw, flags=re.I).group(0))


# ---------------------------------------------------------------------------------------------------- the caps of the brief and the plain language
@needs_tree
def test_the_paragraphs_that_the_brief_caps_are_within_their_caps():
    ranking = _paragraph("sections/04_audit.tex", "The precision lift of the calendar block")
    assert len(ranking) <= 1200, len(ranking)
    for macro in ("auprcCombYY", "auprcCalYY", "auprcCombPostOpen", "auprcCalPostOpen", "precLiftCalYY", "precLiftCalPostOpen", "deltaAuprcDartYY", "deltaAuprcLevYY",
                  "deltaAuprcRelYY", "deltaAuprcDartPostOpen", "deltaAuprcLevPostOpen", "deltaAuprcRelPostOpen", "deltaAuprcDispPostOpen"):
        assert f"\\{macro}{{}}" in ranking, macro
    assert "give the only clear ranking gain in 2026-H1" in ranking and "\\esmref{audit-labels}" in ranking
    assert "The median AUPRC of the combined block" in ranking or "which feeds the alerts" in ranking
    monitors = _paragraph("sections/04_audit.tex", "A drift monitor is useful")
    assert len(monitors) <= 1000, len(monitors)
    assert re.search(r"only[^.]{0,80}monitor[^.]{0,60}positive median association|monitors[^.]{0,80}\bonly\b[^.]{0,80}positive median association", monitors)
    for macro in ("earlyDays", "monEarlyRateRhoMedian", "monEarlyRateAurocMedian", "monEarlyRateRhoQuarters", "monEarlyRateAurocQuarters"):
        assert f"\\{macro}{{}}" in monitors, macro
    third = _paragraph("sections/04_audit.tex", "On the continuous error")
    assert len(third) <= 600, len(third)
    robust = [p for p in _paragraphs(_read("sections/05_economic.tex"))]
    start = next(i for i, p in enumerate(robust) if p.startswith("The signs of NET and SELC of P1"))
    assert sum(len(p) for p in robust[start:start + 3]) <= 1600 and robust[start + 2].startswith("The NET of P1 is negative at every target"), robust[start:start + 3]
    assert any("its ESred is positive except in AQ4 (" in p for p in robust[start:start + 3])
    shortened = _paragraph("sections/04_audit.tex", "The fixed cutoff fails at")
    assert len(shortened) >= 0.75 * 1588 - 40, "this paragraph was cut by more than a quarter: the brief allows a quarter at most"


@needs_tree
def test_the_text_of_the_two_sections_is_in_short_sentences():
    """The author asked for a paper that is easy to follow (ruling 142). Before task T2c Sections 4 and 5 had a mean sentence length of about 30 and 34 words (source
    measure, a macro one word, an interval macro two) and 36 % and 47 % of their sentences were longer than 35 words. The caps leave a margin: the long sentences that
    stay are the ones the brief fixes word by word (the required rewordings and the sentence on the timing and level values)."""
    for rel, mean_cap, long_cap in (("sections/04_audit.tex", 22.0, 0.06), ("sections/05_economic.tex", 27.0, 0.10)):
        lengths = [_words(s) for s in _sentences(_body(rel))]
        assert len(lengths) > 60
        assert statistics.mean(lengths) < mean_cap, (rel, statistics.mean(lengths))
        assert sum(n > 35 for n in lengths) / len(lengths) <= long_cap, (rel, [n for n in lengths if n > 35])


@needs_tree
def test_no_sentence_of_the_two_sections_has_more_than_50_words():
    for rel in ("sections/04_audit.tex", "sections/05_economic.tex"):
        long = [(n, s[:80]) for s in _sentences(_body(rel)) if (n := _words(s)) > 50]
        assert not long, (rel, long)


# ---------------------------------------------------------------------------------------------------- the generated footnote that pointed into the design specification
@needs_tree
def test_the_footnote_of_the_fee_and_position_table_points_at_no_design_specification():
    tex = _read("generated/tables/tab_fee_position_signs.tex")
    note = tex[tex.index("\\footnotetext"):]
    assert "spec" not in note.lower() and "\\S" not in note and "§" not in note, note
    assert "A sign that is the same in every fee row of an analysis does not depend on the fee scenario." in note
    source = (Path(__file__).resolve().parents[2] / "es" / "paper_tables.py").read_text(encoding="utf-8")
    assert "sign-consistency report of spec" not in source
