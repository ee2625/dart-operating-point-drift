"""Plan 2C, task T2b: Section 3 of the paper (Data and design) was cut from about 8 to about 6 pages by moving mechanics and justifications to Online Resource 1
(esm/data_design_details.tex, input by esm/ESM_1.tex). The paper keeps one short paragraph or sentence for each moved piece and points to it with \\esmref{key};
the moved text keeps its numbers (macros) and its wording, apart from its references. These tests keep the move honest, and say what it must not do:

* every label that the other sections \\ref in Section 3 is still there (a label whose subject moved is re-pointed, not dropped);
* every moved subsection is in the ESM under its key and label, the ESM inputs the file, and Section 3 points to it with \\esmref{key};
* each moved paragraph begins as it did in the paper and still holds every macro that it held (no number is typed or lost: the old text is the baseline);
* the paragraphs that must stay as they were (the paragraph on the large-language-model assistants with its notes to the author, the short paragraph of
  3.5 on Online Resource 1, the sentences about the seal in 3.4) are unchanged;
* the sentence that failing-node counts are descriptive and carry no interval is said once in Section 3 (task T2c relies on it);
* the value function J_lambda is defined in the ESM only, and the sentences of the paper that use it point there.
They read the real paper/ folder, as test_paper_g1.py and test_paper_g3.py do."""
import re
from pathlib import Path

import pytest

PAPER = Path(__file__).resolve().parents[3] / "paper"
needs_tree = pytest.mark.skipif(not (PAPER / "sections").exists(), reason="the manuscript folder is not next to dart_dev")

#: The moved subsections of esm/data_design_details.tex: (key, title, label, the first words of the paragraph that opens it, as in the paper before the move).
MOVED = (
    ("data-settlement", "Settlement legs and duplicated rows", "esm:data-settlement",
     "Alerts and labels use the hourly panel series, but profit and loss uses the price at which a virtual position settles."),
    ("data-windows", "Windows and the label threshold", "esm:data-windows",
     "The threshold of the spike label is fitted on a label window"),
    ("stats-bootstrap", "Bootstrap intervals", "esm:stats-bootstrap",
     "Every rate and dollar measure of the economic analysis and every estimand of the sealed tests carries a"),
    ("stats-rotation", "Rotation test and designation rule", "esm:stats-rotation",
     "The designation rule of Section~\\ref{sec:sealed-rule} decides from simulated size and power which procedures are confirmatory tests."),
    ("stats-families", "Secondary test families", "esm:stats-families",
     "Three secondary families are run through the Benjamini--Hochberg procedure"),
    ("stats-mixed", "Mixed model and operator checks", "esm:stats-mixed",
     "Because nodes are correlated within operators, counts of failing nodes are descriptive."),
    ("econ-value", "Value function and profit per unit of risk", "esm:econ-value",
     "Alerts are read for dominance in (NET, ESred) and, as a scalar, through the value function"),
    ("econ-step", "Step size of the cutoff P5", "esm:econ-step",
     "The self-correcting cutoff P5 updates its target level each day"),
)

#: The macros of the paragraphs that moved, as the paper used them (the baseline of the move). Each must be in the ESM file now, so that no number was
#: lost, and none was typed in its place.
MOVED_MACROS = {
    "data-settlement": ("nyisoIntFirstDate", "fmmFirstDate", "placeboFallbackDays", "dupBoundaries", "dupDropped"),
    "data-windows": ("labelPct", "dTenLabelMissed", "dTenHoldoutExtra", "dTenHoldoutMissed", "dTenHoldoutRows"),
    "stats-bootstrap": ("ciLevel", "bootBlockDays", "bootB", "bootSeed"),
    "stats-rotation": ("famAlpha",),
    "stats-families": ("bhLevel", "alphaGrid", "labelPct", "seedList", "bhTestsAuditRobustness", "bhTestsEconomicRobustness", "bhTestsDriftMonitors",
                       "bhRejectionsEconomicRobustness", "bhRejectionsDriftMonitors"),
    "econ-value": ("lamGrid",),
    "econ-step": ("etaStar",),
}

#: The paragraph on the large-language-model assistants of 3.5 with the notes to the author above it: journal policy, kept exactly (task T2b brief).
AI_NOTES = """% AUTHOR: the next paragraph is a draft for your confirmation (Opus review I13 and re-review N2). It describes the AI assistants' role as it
% was (design proposals and review, code, data pulls and processing, analyses, tables and figures, drafting and editing) and states your review
% and responsibility. The Author contribution statement in the back matter puts every CRediT role that this list touches (methodology,
% validation, data curation, software, formal analysis, investigation, visualization, writing) into its AI-assisted clause, so the two
% statements and the cover letter must stay in step. Confirm that they are literally true before submission, or edit them together. The
% first draft said "Every analysis choice, number and claim was checked by the author against the saved outputs"; the review asks that
% such an attestation be true as written, so it is not repeated here.
"""
AI_PARAGRAPH = ("Large-language-model assistants, namely \\aiTools{}, were used under the author's direction to propose and review parts of the statistical design, "
                "to write, test and review code, to pull and process the data, to run the analyses, to build the tables and figures, and to draft and edit text. "
                "The author reviewed and approved the design, the code and the text and takes responsibility for the content. The generated-macro pipeline "
                "described in Online Resource 1 and the code archive record the provenance of each number.")
IMPLEMENTATION_PARAGRAPH = ("Online Resource 1 lists the settings of the pipeline in full: the hyperparameters of the gradient-boosted trees (XGBoost) that serve as "
                            "the primary estimator and of the logistic regression that serves as a contrast, the scaler, the label and cutoff rules, the seeds, "
                            "the treatment of missing data and duplicates, the time-zone and daylight-saving conventions, the bootstrap, the fees and the "
                            "software versions, together with the features of every block. A node is fitted only if it has at least \\minPositives{} positive "
                            "training labels and both classes. The primary seed is \\seedPrimary{}, and the seeds \\seedList{} serve the seed sensitivity of "
                            "Online Resource 1.")


def _read(rel: str) -> str:
    return (PAPER / rel).read_text(encoding="utf-8")


def _live(text: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in text.splitlines())


def _flat(text: str) -> str:
    return " ".join(text.split())


def _macros(text: str) -> set[str]:
    return set(re.findall(r"\\([A-Za-z]+)(?=\{\}|\\|[^A-Za-z]|$)", text))


def _subsection(text: str, label: str) -> str:
    """The live text of the \\subsection with the label, up to the next heading."""
    m = re.search(r"\\subsection\{[^{}]*\}\\label\{" + re.escape(label) + r"\}(.*?)(?=\\(?:sub)*section\b|\Z)", _live(text), flags=re.S)
    assert m, label
    return m.group(1)


# ---------------------------------------------------------------------------------------------------- labels and pointers
@needs_tree
def test_every_label_of_section_3_that_another_file_refs_is_still_there():
    section3 = _read("sections/03_data_design.tex")
    defined = set(re.findall(r"\\label\{([^}]*)\}", _live(section3)))
    wanted = {"sec:data", "sec:data-panel", "sec:data-labels", "sec:data-blocks", "sec:data-analyses", "sec:data-impl", "sec:data-stats", "sec:data-econ",
              "sec:data-econ-positions", "sec:data-econ-measures", "sec:data-econ-policies", "eq:dart", "eq:pnl", "eq:lapg", "eq:selc", "eq:decomp",
              "tab:panel", "tab:fees"}
    assert wanted <= defined, wanted - defined
    used = set()
    for path in sorted((PAPER / "sections").glob("*.tex")):
        used |= {r for r in re.findall(r"\\(?:ref|eqref|autoref)\{([^}]*)\}", _live(path.read_text(encoding="utf-8"))) if r.startswith(("sec:data", "eq:", "tab:panel", "tab:fees"))}
    assert used <= defined, f"a label of Section 3 that the sections use is no longer defined there: {sorted(used - defined)}"
    assert "eq:value" not in defined, "the value function moved to the ESM (esm:econ-value); its equation is not in Section 3"


@needs_tree
def test_each_moved_subsection_is_in_the_esm_under_its_key_and_the_paper_points_to_it():
    esm_main, details = _read("esm/ESM_1.tex"), _read("esm/data_design_details.tex")
    assert "\\input{data_design_details}" in _live(esm_main)
    assert re.search(r"\\section\{Data, statistics and economic design in detail\}\\label\{esm:design-details\}", _live(details))
    table = _live(_read("crossref.tex"))
    section3 = _live(_read("sections/03_data_design.tex"))
    assert "\\esmkey{design-details}{Data, statistics and economic design in detail}" in table
    for key, title, label, first in MOVED:
        assert f"\\esmkey{{{key}}}{{{title}}}" in table, key
        assert f"\\subsection{{{title}}}\\label{{{label}}}" in _flat(details), key
        assert _flat(_subsection(details, label)).startswith(first), (key, "the paragraph that opens the subsection")
        assert f"\\esmref{{{key}}}" in section3, f"Section 3 does not point to {key}"


@needs_tree
def test_the_moved_paragraphs_hold_every_macro_that_they_held_and_no_typed_number():
    details = _live(_read("esm/data_design_details.tex"))
    for key, title, label, first in MOVED:
        body = _subsection(_read("esm/data_design_details.tex"), label)
        missing = [m for m in MOVED_MACROS.get(key, ()) if m not in _macros(body)]
        assert not missing, f"{key}: macros of the paper's paragraph missing from the ESM subsection: {missing}"
    # the references of the moved text are references of the ESM: \ref inside it, \mainref to the paper, never \esmref
    assert "\\esmref" not in details
    refs = set(re.findall(r"\\ref\{([^}]*)\}", details))
    esm_labels = set(re.findall(r"\\label\{([^}]*)\}", "".join(_live(p.read_text(encoding="utf-8")) for p in (PAPER / "esm").glob("*.tex"))))
    tables = set(re.findall(r"\\label\{([^}]*)\}", "".join(_live(p.read_text(encoding="utf-8")) for p in (PAPER / "generated" / "tables").glob("*.tex"))))
    assert refs <= esm_labels | tables, f"the moved text refers to labels that the ESM does not define: {sorted(refs - esm_labels - tables)}"
    assert "Online Resource 1" not in details, "in the moved text 'Online Resource 1' means 'this Online Resource'"


@needs_tree
def test_the_paper_keeps_one_short_replacement_for_each_moved_piece():
    section3 = _flat(_live(_read("sections/03_data_design.tex")))
    # the sentence of the paper that replaces each moved piece (a few words that identify it)
    for words in ("The details by operator are in \\esmref{data-settlement}",
                  "(\\esmref{data-windows}). The label share is a parameter of its own",
                  "Two window conventions are in use and are never pooled",
                  "are described in \\esmref{stats-bootstrap}",
                  "(\\esmref{stats-rotation})",
                  "The three families are described in \\esmref{stats-families}",
                  "are in \\esmref{stats-mixed}",
                  "are in \\esmref{econ-value}",
                  "from a fixed grid (\\esmref{econ-step})"):
        assert words in section3, words
    for gone in ("0.005", "\\bhLevel{}, and the first of them is descriptive. Audit robustness", "Leave-one-operator-out checks show how far", "in each replicate",
                 "For the activity measures of the sealed tests"):
        assert gone not in section3, gone                                  # the grid, the family descriptions and the draws of each measure are in the ESM only
    # the value function is defined in the ESM; Section 3 and Section 5 only point to it
    assert "J_\\lambda = \\mathrm{NET}" not in section3 and "J_\\lambda = \\mathrm{NET}" in _flat(_live(_read("esm/data_design_details.tex")))
    assert "\\esmref{econ-value}" in _live(_read("sections/05_economic.tex"))


@needs_tree
def test_the_paragraphs_that_must_stay_as_they_were_are_unchanged():
    raw = _read("sections/03_data_design.tex")
    assert AI_NOTES + AI_PARAGRAPH in raw, "the paragraph on the large-language-model assistants, with the notes to the author above it, was changed"
    assert _flat(IMPLEMENTATION_PARAGRAPH) in _flat(_live(raw))
    flat = _flat(_live(raw))
    assert re.search(r"Development code reads no row dated after 2026-06-30, apart from one diagnostic count over six boundary rows[^.]*\(\\esmref\{sealed-record\}\)\.", flat)
    # plan 2C task T3b: the style pass wrote "the pull and the run of the forward tests" for "the sealed pull and the sealed run"; both are the same statement
    assert re.search(r"A loader guard rejects later rows, and only the (?:sealed )?pull and the (?:sealed )?run(?: of the forward tests)?, both of which require the SHA-256 hash of "
                     r"the protocol, may read them\.", flat)


@needs_tree
def test_the_sentence_that_failing_node_counts_are_descriptive_is_said_once_in_section_3():
    flat = _flat(_live(_read("sections/03_data_design.tex")))
    sentence = "counts of failing nodes (Section~\\ref{sec:audit}) are descriptive and carry no interval"
    assert flat.count(sentence) == 1
    assert flat.count("carry no interval") == 1 and flat.count("are descriptive") == 1
    assert "stationary bootstrap over settlement days" in flat and "the same days for all nodes" in flat
    assert "approximate and conditional on the realised quarters" in flat


@needs_tree
def test_section_3_prints_no_typed_section_number_and_no_registration_wording():
    live = _live(_read("sections/03_data_design.tex")) + _live(_read("esm/data_design_details.tex"))
    assert not re.search(r"(?:Section|Sections|Appendix)\s*~?\s*\d", live), "a typed section number"
    assert not re.search(r"\bregist(?:er|ered|ration)\w*|\bOSF\b|Zenodo|preregist", live, flags=re.I)
