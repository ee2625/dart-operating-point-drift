"""Gate G3 (independent read of the manuscript as submitted before Test A has run): the wording fixes of items 1 to 20, and the facts that
they state, checked against the generated macros that hold them. Every number in the manuscript was confirmed against its macro at G3; the
faults were in the wording, and these tests keep the corrected wording (and its internal vocabulary, its pending-state sentences and its
claims about Table 10) from drifting back. They read the real paper/ folder, as the last tests of test_paper_g1.py do.
Since plan 2C task T2a (26 October 2026) the former sections/06_sealed.tex is esm/forward_tests.tex and the former
appendices/A_ledger.tex is esm/designation_rule_study.tex; the tests that read them read the new files with the same assertions, apart from the
references (a reference to the paper from the ESM is \\mainref{key}, one to the ESM from the paper \\esmref{key}; paper/crossref.tex).
Since plan 2C task T2c (26 October 2026) the paragraphs of Sections 4 and 5 that were cut (the label control, the sign tests, the monitor paragraphs, the stratified NET,
the holdout corrections, the target and band) are in esm/audit_economic_details.tex and the supplementary tables of esm/ESM_1.tex; the assertions about them read those
files and say the same.
Since plan 2C task T3b (26 October 2026) the abstract, the introduction, the conclusion, the back matter and the cover letters are the controller's plain-language
rewrite, and the discussion and the limitations are one section (sections/07_discussion.tex: 6.1 What the evidence shows, 6.2 Governing fixed and adaptive thresholds,
6.3 Forward evidence, label sec:disc-forward, 6.4 Limitations, label sec:limitations); sections/08_limitations.tex is gone. The items below keep their intent and read the
new places; each docstring says what the item protected, where it is asserted now and, where an assertion went, why. In the main text and the cover letters the tests are
called "forward" and not "sealed" (the Online Resource keeps "sealed" for the protocol's own objects)."""
import re
from pathlib import Path

import pytest

PAPER = Path(__file__).resolve().parents[3] / "paper"
needs_tree = pytest.mark.skipif(not (PAPER / "sections").exists(), reason="the manuscript folder is not next to dart_dev")
PENDING = dict(TestA=False, TestAdidnotrun=False, TestB=False, TestBdidnotrun=False)       # the state of the manuscript as submitted
TEST_A_RAN = dict(PENDING, TestA=True)
INTERNAL = re.compile(r"Ruling|FORENSICS|Seat |Opus|controller|spec v")


def _read(rel):
    return (PAPER / rel).read_text(encoding="utf-8")


def _uploaded(rel):
    """What the packager puts into the zip for a markdown file: the file without its HTML comments, which are notes to the author."""
    from es.paper_package import _HTML_COMMENT
    return _HTML_COMMENT.sub("", _read(rel))


def _live(text):
    """The text without its LaTeX comments (full-line and trailing)."""
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in text.splitlines())


def _flat(text):
    return " ".join(text.split())


def _variant(rel, **setting):
    """The live text of a file as a reader sees it for one state of the switches, on one line."""
    from es.paper_lint import SWITCHES, _resolve_switches
    state = dict.fromkeys(SWITCHES, False)
    state.update(setting)
    text = _resolve_switches(_live(_read(rel)), state)
    assert text is not None, f"{rel}: the switches are unbalanced"
    return _flat(text)


def _macro(name):
    m = re.search(r"^\\newcommand\{\\" + name + r"\}\{(.*)\}\s*$", _read("generated/macros.tex"), flags=re.M)
    assert m, name
    return m.group(1)


def _number(text):
    """A dollar or percent macro value as a float: \\textminus{}\\$4{,}111 is -4111, 9.9\\% is 9.9."""
    s = text.replace("\\textminus{}", "-").replace("\\$", "").replace("{,}", "").replace("\\%", "").replace("\\", "")
    return float(s)


def _interval(text):
    lo, hi = re.fullmatch(r"\[(.*), (.*)\]", text).groups()
    return _number(lo), _number(hi)


SECTIONS = ("00_abstract", "01_introduction", "02_related", "03_data_design", "04_audit", "05_economic", "07_discussion",
            "09_conclusion", "90_backmatter")                                                      # 08_limitations is part of 07_discussion since plan 2C


#: The development quarters as the text names them, and the macro suffix of each (generated/macros.tex).
QUARTER = {"AqThree": "2025-Q3", "AqFour": "2025-Q4", "AqOne": "2026-Q1", "AqTwo": "2026-Q2"}
QUARTER_LIST = r"((?:20\d\d-Q[1-4](?:, | and )?)+)"                                                    # "2025-Q3, 2025-Q4 and 2026-Q1"


def _named(listing):
    """The quarters of a listing as the suffixes of the macros, in the order of the text."""
    by_name = {name: suffix for suffix, name in QUARTER.items()}
    return [by_name[n] for n in re.findall(r"20\d\d-Q[1-4]", listing)]


def _usd(name):
    return _number(_macro(name))


def _subsection(rel, label, **setting):
    """The live text of the subsection that carries \\label{label}, as a reader sees it for one state of the switches: from the label to the next
    \\subsection or \\section (or the end of the file)."""
    text = _variant(rel, **setting)
    start = text.index("\\label{%s}" % label)
    end = [text.find(h, start) for h in ("\\subsection{", "\\section{")]
    end = min([e for e in end if e >= 0], default=len(text))
    return text[start:end]


# ---------------------------------------------------------------------------------------------------- blockers
@needs_tree
def test_item_1_the_limitations_do_not_say_that_a_sealed_value_exists_before_either_test_has_run():
    """Item 1 protected: before either test has run, the text must not say that a value from the forward tests exists. Since plan 2C the limitations are Section 6.4
    (sec:limitations) and the status of the forward tests is Section 6.3 (sec:disc-forward) of sections/07_discussion.tex; sections/08_limitations.tex is gone. Both are read
    in the state in which Test A is absent (the manuscript as submitted before the run) and in the state in which it has run. The old sentences ("two sealed quarters follow",
    "no sealed value of the alerts exists") are in 6.3 as: the design has not been run, it is decided only on the pooled window, and no price dated after June 2026 has been
    used."""
    forward = _subsection("sections/07_discussion.tex", "sec:disc-forward", **PENDING)
    limitations = _subsection("sections/07_discussion.tex", "sec:limitations", **PENDING)
    assert re.search(r"(?:has|have|had) not been run", forward), "6.3 does not say that the design has not been run"
    assert re.search(r"one confirmatory test.{0,160}decided only on the pooled window", forward)
    assert re.search(r"no (?:price|data)[^.]{0,30}after June 2026", forward, flags=re.I)         # at submission nothing after June 2026 has been read
    for text in (forward, limitations):
        assert "interim" not in text.lower() and "TestA" not in text, text[:200]                  # no value of a forward test is reported or named
        assert "both registered" not in text and "registered and pending" not in text and "reported as interim" not in text    # Ruling 133
        assert not re.search(r"\bsealed (?:value|quarter|window)", text)                         # the main text says "forward"; "a fully sealed test" (6.4.2) is generic
    assert "\\ref{sec:disc-forward}" in limitations                                               # the limitations point at 6.3 for the forward design
    ran = _subsection("sections/07_discussion.tex", "sec:disc-forward", **TEST_A_RAN)            # the text for a Test A that has run
    assert re.search(r"interim estimate", ran) and re.search(r"decided only on the pooled window", ran) and re.search(r"Test B[^.]*(?:not been run|pending)", ran)
    assert "TestA" in ran and "TestA" not in forward


@needs_tree
def test_item_2_h_b1_and_h_b2_carry_no_p_value():
    for rel in ("esm/data_design_details.tex", "esm/forward_tests.tex"):          # the first was sections/03_data_design.tex (the paragraph moved to the ESM, plan 2C task T2b)
        text = _flat(_live(_read(rel)))
        assert "H-B1 and H-B2 carry no $p$-value" in text and "is a stored diagnostic, and only the descriptive audit-robustness family" in text, rel
        assert "uses $p$-values of this kind" in text, rel
        assert "their one-sided $p$-value is the basic-bootstrap $p$-value" not in text, rel
    assert "estimands with approximate intervals and no $p$-value" in _flat(_live(_read("sections/03_data_design.tex")))   # what the paper keeps of the moved paragraph
    sealed = _flat(_live(_read("esm/forward_tests.tex")))                                      # the definition of the diagnostic is kept
    assert "The $p$-value is the number of replicates whose centred statistic is at least the observed one, plus one, divided by the number of replicates plus one" in sealed


@needs_tree
def test_item_3_the_discussion_confines_the_stable_operating_point_claim_to_what_table_10_shows():
    """Item 3 protected: the claim about what a steady alert rate is worth in dollars is confined to what the primary-results table shows, quarter by quarter, and is not
    generalised. Since plan 2C it is the third paragraph of Section 6.2 (sec:disc-governance): "The dollar value of a steady rate depends on the quarter" and "Four quarters
    do not say which pattern is general"; the table is in Online Resource 1 (\\esmref{supp-econ}). The quarters that the text lists are checked against the macros of that
    table, for all four quarters (the old text named three quarters and the post-open 2026-H1 analysis; the new text names the four quarters and no other analysis)."""
    text = _subsection("sections/07_discussion.tex", "sec:disc-governance", **PENDING)
    assert "is therefore not worth more in dollars" not in text and "it loses on both where the fixed cutoff under-alerted" not in text
    assert re.search(r"dollar value of a steady rate depends on the quarter", text) and "\\esmref{supp-econ}" in text
    assert re.search(r"do(?:es)? not say which pattern is general", text) and "The pattern is not general." not in text
    for macro in ("esredPFourAqThree", "esredPOneAqThree", "netPFourAqOne", "netPOneAqOne", "alertRatePOneAqThree", "alertRatePFourAqThree", "alertRatePOneAqOne",
                  "alertRatePFourAqOne"):
        assert "\\" + macro + "{}" in text, macro
    # the claims themselves, against the macros of the primary-results table (tab_econ_primary)
    esred_lower = [q for q in QUARTER if _usd(f"esredPFour{q}") < _usd(f"esredPOne{q}")]
    esred_higher = [q for q in QUARTER if _usd(f"esredPFour{q}") > _usd(f"esredPOne{q}")]
    net_higher = [q for q in QUARTER if _usd(f"netPFour{q}") > _usd(f"netPOne{q}")]
    net_lower = [q for q in QUARTER if _usd(f"netPFour{q}") < _usd(f"netPOne{q}")]
    assert esred_lower and esred_higher and len(esred_lower + esred_higher) == len(QUARTER)         # no tie hides in the sentence
    assert net_higher and net_lower and len(net_higher + net_lower) == len(QUARTER)
    lower = re.search(r"lower tail effect in " + QUARTER_LIST + r"\s*and a higher one in " + QUARTER_LIST, text)       # the quarters that the text lists, by their key phrases
    assert lower and _named(lower.group(1)) == esred_lower and _named(lower.group(2)) == esred_higher, (esred_lower, esred_higher)
    net = re.search(r"net value of P4 is higher in " + QUARTER_LIST + r"\s*and lower in " + QUARTER_LIST, text)
    assert net and _named(net.group(1)) == net_higher and _named(net.group(2)) == net_lower, (net_higher, net_lower)
    target = _number(_macro("alphaPct"))
    assert _number(_macro("alertRatePOneAqThree")) > target                                          # "where P1 alerted in ... against a target": over-alerted
    assert _number(_macro("alertRatePOneAqOne")) < target                                            # "where P1 alerted in only ...": under-alerted
    assert _number(_macro("alertRatePFourAqOne")) > _number(_macro("alertRatePOneAqOne"))           # "and P4 raised the rate to ..."


# ---------------------------------------------------------------------------------------------------- should fix
@needs_tree
def test_item_4_the_pending_text_says_that_test_a_has_not_been_run_and_nothing_suggests_a_result():
    """Item 4 protected: before Test A has run, the abstract, the introduction and the conclusion say that the forward tests have not been run and nothing in them suggests a
    result. Since plan 2C the abstract says it in its last sentence, the introduction in its last paragraph before the contributions and the conclusion in its closing
    paragraph; the list of contributions has three items and no longer names the forward tests (the sentence that named Test A and Test B there is gone with the old list)."""
    for rel in ("sections/00_abstract.tex", "sections/01_introduction.tex", "sections/09_conclusion.tex"):
        text = _variant(rel, **PENDING)
        assert re.search(r"(?:has|have|had) not been run", text) and "forward" in text, rel   # the forward quarters are said not to have been run
        assert rel == "sections/00_abstract.tex" or "\\ref{sec:disc-forward}" in text, rel     # and the introduction and the conclusion point at Section 6.3
    contributions = re.search(r"% BEGIN CONTRIBUTIONS(.*?)% END CONTRIBUTIONS", _read("sections/01_introduction.tex"), flags=re.S).group(1)
    assert len(re.findall(r"\\item\[", contributions)) == 3 and not re.search(r"Test [AB]|forward|sealed", contributions)    # no result is suggested there
    for rel in ("sections/00_abstract.tex", "sections/01_introduction.tex", "sections/09_conclusion.tex"):
        assert "to be reported" not in _variant(rel, **PENDING), rel
        text = _variant(rel, **PENDING)                                                       # Ruling 133: nothing is registered or runs after a registration
        assert "registered and pending" not in text and "after registration" not in text, rel
        assert "TestA" not in text and "Test A" not in text, rel                              # nothing of Test A before the run
    ran = _variant("sections/01_introduction.tex", **TEST_A_RAN)                              # a Test A that has run is reported as an interim result
    assert re.search(r"has been run and is reported as an interim result", ran) and "\\ref{sec:disc-forward}" in ran
    from es.paper_lint import abstract_word_counts
    assert max(abstract_word_counts(PAPER).values()) <= 250, abstract_word_counts(PAPER)


@needs_tree
def test_item_5_both_windows_are_unseen_but_not_prospective_and_the_two_statements_agree():
    sealed = _flat(_live(_read("esm/forward_tests.tex")))
    assert ("Both windows are unseen but not prospective: Test A's had closed and Test B's had begun when the protocol was fixed, so that "
            "prices for all of Test A's days and for the first days of Test B's already existed.") in sealed
    assert "osfRegisteredDate" not in sealed
    assert "As Section~\\ref{sec:sealed-windows} says, its window is unseen but not prospective" in sealed
    assert "\\label{sec:sealed-windows}" in sealed


@needs_tree
def test_item_6_the_2026_h1_holdout_of_the_earlier_version_is_as_opened_and_never_sealed():
    """Item 6 protected: the 2026-H1 holdout of the earlier version is called as-opened, never "sealed". The limitations (it was sections/08_limitations.tex) are Section 6.4 of
    sections/07_discussion.tex since plan 2C, which the loop below already reads, so the old file left the list."""
    for rel in ("sections/03_data_design.tex", "sections/04_audit.tex"):
        text = _flat(_live(_read(rel)))
        assert "sealed holdout" not in text and "sealed 2026-H1" not in text, rel
    assert "the as-opened holdout of the earlier version" in _flat(_read("sections/03_data_design.tex"))
    assert "The as-opened 2026-H1 comparison of the earlier version" in _flat(_read("esm/audit_economic_details.tex"))      # the label-control paragraph moved to the ESM (plan 2C task T2c)
    assert "The season, and not the rule, accounts for the difference" in _flat(_live(_read("sections/04_audit.tex")))         # and the paper keeps its one sentence about it
    for rel in ("sections/05_economic.tex", "sections/07_discussion.tex", "esm/ESM_1.tex", "esm/designation_rule_study.tex",      # 07_discussion.tex holds the limitations (6.4)
                "esm/audit_economic_details.tex"):
        assert not re.search(r"sealed (?:2026-H1|holdout|26H1)|\(sealed\)", _live(_read(rel))), rel


@needs_tree
def test_item_7_e5_is_reported_per_window_and_is_not_pooled():
    text = _flat(_live(_read("esm/forward_tests.tex")))
    assert "The pooled estimands E1 to E4 are the headline numbers whenever they exist" in text
    assert "E5, the block table, is reported per window and is not pooled." in text
    assert "The pooled estimands, E1 to E4, add the per-window pieces" in text and "E5 is reported per window and is not pooled." in text
    assert "pooled estimands E1 to E5" not in text and "Pooled estimands add" not in text


@needs_tree
def test_item_8_the_seal_is_not_overstated_where_it_is_first_described():
    """Item 8 protected: where Section 3.4 first describes the seal, it does not claim more than holds (one diagnostic count over six boundary rows read the files whole;
    a loader guard rejects later rows; only the pull and the run, which require the hash of the protocol, may read them). Since the style pass of plan 2C the main text calls
    the windows "forward" and says "the pull and the run of the forward tests"; the Online Resource keeps "sealed pull" and "sealed run", the protocol's own objects."""
    data = _flat(_live(_read("sections/03_data_design.tex")))
    assert re.search(r"Development code reads no row dated after 2026-06-30, apart from one diagnostic count over six boundary rows[^.]*\(\\esmref\{sealed-record\}\)", data)
    assert re.search(r"only the (?:sealed )?pull and the (?:sealed )?run(?: of the forward tests)?, both of which require the SHA-256 hash of the protocol, may read them", data)
    sealed = _flat(_live(_read("esm/forward_tests.tex")))
    assert "downloaded by the sealed pull and read by the sealed run, and both require the SHA-256 hash of the protocol" in sealed
    assert "single entry point" not in sealed and "sealed-run entry point" not in data


@needs_tree
def test_item_9_the_tail_measure_intervals_are_called_approximate_and_no_h_b_estimand_excludes_zero():
    """Item 9 protected: an interval of the tail measure (ESred) is called approximate wherever the summary text says that it excludes zero, and no sentence about an H-B
    estimand says that an interval excludes zero. Since plan 2C the abstract says "with an approximate interval that excludes zero only in 2025-Q4" in every state of Test A and
    the discussion says it in 6.1; the summary texts (abstract, introduction, discussion, conclusion, the two cover letters) are scanned sentence by sentence, and a sentence that
    says "excludes zero" of the tail measure (or of no named quantity) has to carry the word "approximate". The conclusion no longer states an interval, so its assertion went."""
    from es.paper_lint import abstract_variants, _braced_arg
    body = _flat(_live(_read("sections/00_abstract.tex")))
    for v in abstract_variants(_braced_arg(body, "\\abstract")):
        assert re.search(r"approximate[^.]{0,30}interval[^.]{0,40}excludes zero", _flat(v.text)), v.label
    assert re.search(r"approximate[^.]{0,30}interval[^.]{0,80}excludes zero", _flat(_live(_read("sections/07_discussion.tex"))))
    summaries = {f"sections/{name}.tex": _live(_read(f"sections/{name}.tex")) for name in ("00_abstract", "01_introduction", "07_discussion", "09_conclusion")}
    summaries.update({name: _uploaded(name) for name in ("cover_letter.md", "cover_letter_TestA_ran.md")})
    seen = 0
    for name, text in summaries.items():
        for sentence in re.split(r"(?<=[.;?!])\s+", _flat(text)):
            if not re.search(r"exclud\w* zero|excluding zero", sentence):
                continue
            other_quantity = re.search(r"\b(?:NET|SELC|excess loss capture|net value)\b", sentence) and not re.search(r"ESred|expected.shortfall|tail", sentence)
            if not other_quantity:
                seen += 1
                assert "approximate" in sentence, (name, sentence)
    assert seen >= 3, seen                                                                          # the scan finds the abstract and the two sentences of the discussion
    for rel in [f"sections/{name}.tex" for name in SECTIONS] + ["esm/data_design_details.tex", "esm/audit_economic_details.tex"]:     # no sentence about an H-B estimand says that an interval excludes zero
        name = rel                                                                                    # (the details moved from Section 3 are checked too, plan 2C task T2b)
        for sentence in re.split(r"(?<=[.;?!])\s+", _flat(_live(_read(rel)))):
            if re.search(r"exclud\w* zero|excluding zero", sentence) and re.search(r"H-B[12]|portab", sentence):
                assert "do not say" in sentence or "never" in sentence, (name, sentence)


@needs_tree
def test_item_10_the_no_drift_reference_is_called_optimistic_because_it_draws_the_nodes_independently():
    text = _flat(_live(_read("sections/04_audit.tex")))
    assert "A count above it would be unusual without drift if the nodes moved independently, as the simulation assumes." in text
    assert "Nodes co-move, so the true no-drift range is wider and the reference is optimistic." in text
    assert "a count above it is unlikely without drift" not in text
    assert "The simulation draws the nodes independently" in _flat(_live(_read("esm/designation_rule_study.tex")))      # the source of the caveat


@needs_tree
def test_item_11_the_rules_that_ae_q4_breaks_are_stated_as_they_hold():
    """Item 11 protected: the two rules that AQ4 breaks (the expected shortfall falls whenever P1 alerts more often than its target; for the short-spike part of the premium position
    ESred is positive) are stated with their exception and agree with the macros. Both sentences are in Section 5.2 and pass unchanged. The third assertion, on a sentence of the
    discussion ("for the short-spike part of the premium position ESred is positive except in AQ4, at a cost in NET"), is gone: the style pass of plan 2C took that sentence out of
    Section 6, and Section 6.1 makes no claim about the short-spike part; the facts it stated stay asserted below, against the macros, for Section 5."""
    econ = _flat(_live(_read("sections/05_economic.tex")))
    assert "falls in four of the five analyses in which P1 alerted more often than its target, and not in AQ4" in econ
    assert "its ESred is positive except in AQ4 (" in econ
    # the counts, against the macros: five analyses over-alerted, four of them with a positive ESred whose interval lies above zero
    target = _number(_macro("alphaPct"))
    words = {"AQ3": "AqThree", "AQ4": "AqFour", "AQ1": "AqOne", "AQ2": "AqTwo", "2024-2025": "YY", "24H2-25H2": "HTwo", "post-open": "PostOpen"}
    over = [a for a, w in words.items() if _number(_macro(f"alertRatePOne{w}")) > target]
    assert sorted(over) == sorted(["AQ3", "AQ4", "AQ2", "2024-2025", "24H2-25H2"]), over            # "the five analyses in which P1 alerted more often than its target"
    falls = [a for a in over if _number(_macro(f"esred{words[a]}")) > 0 and _interval(_macro(f"esred{words[a]}CI"))[0] > 0]
    assert sorted(falls) == sorted(["AQ3", "AQ2", "2024-2025", "24H2-25H2"]), falls                # "four of the five ... and not in AQ4"
    assert _number(_macro("esredPOneAqFour")) < 0 and _interval(_macro("esredAqFourCI"))[0] < 0 < _interval(_macro("esredAqFourCI"))[1]
    short = {w: _number(_macro(f"esredShortPOneAq{w}")) for w in ("Three", "Four", "One", "Two")}
    assert [w for w, v in short.items() if v <= 0] == ["Four"], short                           # positive except in AQ4
    assert all(_number(_macro(f"netShortPOneAq{w}")) < 0 for w in short)                       # at a cost in NET


@needs_tree
def test_item_12_the_analysis_team_is_defined_once_and_nothing_in_the_summary_is_a_term_of_the_working_files():
    sealed = _flat(_live(_read("esm/forward_tests.tex")))
    assert sealed.count("by which we mean the author and the AI assistants") == 1
    assert "that a research agent saw in" in sealed and "a research agent opened eleven files" in sealed
    summary = sealed.split("states six facts, which are summarised here.", 1)[1].split("Apart from these facts", 1)[0]
    assert len(re.findall(r"\\item ", summary)) == 6
    assert not re.search(r"ruling|fee note|design specification|analysis log|first version", summary, flags=re.I)


@needs_tree
def test_item_13_the_cover_letters_and_the_data_availability_agree_on_the_operators_terms_and_say_nothing_of_an_osf_supplement():
    for name in ("cover_letter.md", "cover_letter_TestA_ran.md"):
        letter = _read(name)
        assert "not redistributed, pending a check of the operators' terms of use, and are available from me on request" in letter, name
        assert "because the operators' terms of use govern them" not in letter, name
    back = _flat(_live(_read("sections/90_backmatter.tex")))
    assert "The development outputs that set the registered protocol are posted with the OSF registration." not in back      # Ruling 133: no OSF
    assert "OSF" not in back and "archived at" not in back
    assert "The code repository (\\codeRepoURL) holds the pull scripts, which rebuild the NYISO, ISO-NE and MISO series." in back
    assert "subject to the operators' terms of use" in back


@needs_tree
def test_item_14_the_manuscript_prints_and_uploads_no_internal_vocabulary():
    for name in SECTIONS:
        live = _live(_read(f"sections/{name}.tex"))
        assert not INTERNAL.search(live), (name, INTERNAL.search(live).group(0))
    for rel in ("esm/forward_tests.tex", "esm/designation_rule_study.tex", "esm/ESM_1.tex", "esm/data_design_details.tex", "esm/audit_economic_details.tex", "main.tex", "crossref.tex"):
        assert not INTERNAL.search(_live(_read(rel))), rel
    for path in sorted((PAPER / "generated" / "tables").glob("*.tex")):                       # the tables of the PDFs (a build writes them)
        assert not INTERNAL.search(_live(path.read_text(encoding="utf-8"))), path.name
    for rel in ("cover_letter.md", "cover_letter_TestA_ran.md", "CORRECTIONS_TO_PREPRINT.md"):     # as uploaded: the notes to the author (HTML comments) go
        assert not INTERNAL.search(_uploaded(rel)), rel
        assert not re.search(r"Opus|\bruling\b|re-review|spec section", _uploaded(rel)), rel
    back = _read("sections/90_backmatter.tex")
    assert "% AUTHOR (Opus review C6" not in back and "% AUTHOR (Opus review minor 26)" not in back          # the two blocks of Data and Code availability are deleted
    assert "while the author was away" not in back and "ruling of 25 October" not in back
    assert "% AUTHOR (Opus review I13, re-review N2)" in back                                    # the author confirms the CRediT and AI-use statements: untouched
    assert "% AUTHOR: the next paragraph is a draft for your confirmation" in _read("sections/03_data_design.tex")
    text = _flat(_live(_read("esm/designation_rule_study.tex")))
    assert "at the methodology review" not in text and "A supplementary check made before the protocol was fixed" in text
    assert "every ruling in the analysis log" not in _flat(_live(_read("esm/forward_tests.tex")))


@needs_tree
def test_item_14_the_manuscript_points_to_the_corrections_file_and_not_to_the_internal_log():
    """Item 14 protected: the manuscript and the letters point the reader to the file Corrections to the preprint and never to the internal revision log, and the author's notes of the
    letter tell the author to upload that file. Both letters keep the sentence in their text; the notes of cover_letter.md keep note 5. The notes of cover_letter_TestA_ran.md
    (two notes since the letters were replaced) did not name the file until the controller added note 5 (26 Oct 2026, 05:51): asserted in its own test below."""
    assert "listed in the file Corrections to the preprint that accompanies this manuscript" in _flat(_live(_read("sections/04_audit.tex")))
    for name in ("cover_letter.md", "cover_letter_TestA_ran.md"):
        letter = _read(name)
        assert "each correction is listed in the accompanying file Corrections to the preprint" in letter, name
        assert "revision log that accompanies" not in letter and "upload a revision log" not in letter, name
    assert "CORRECTIONS_TO_PREPRINT.md" in _read("cover_letter.md").split("AUTHOR NOTES", 1)[1]


@needs_tree
def test_item_14_the_notes_of_the_test_a_letter_tell_the_author_to_upload_the_corrections_file():
    assert "CORRECTIONS_TO_PREPRINT.md" in _read("cover_letter_TestA_ran.md").split("AUTHOR NOTES", 1)[1]


@needs_tree
def test_item_14_the_corrections_file_has_thirteen_numbered_rows_with_the_numbers_of_the_log_and_no_others():
    lines = _read("CORRECTIONS_TO_PREPRINT.md").splitlines()
    assert lines[0] == "# Corrections to the preprint"
    header = [c.strip() for c in lines[[i for i, l in enumerate(lines) if l.startswith("| No.")][0]].strip("|").split("|")]
    assert header == ["No.", "What the preprint said", "What this manuscript says", "Section"]
    rows = {}
    for line in lines:
        if re.match(r"^\| \d+ \|", line):
            cells = [c.strip() for c in line.strip().strip("|").split(" | ")]
            assert len(cells) == 4, line
            rows[int(cells[0])] = cells
    assert sorted(rows) == list(range(1, 14))
    log = {}
    for line in _read("REVISION_LOG.md").splitlines():
        if line.startswith("| ") and not line.startswith(("| ID", "|---")):
            cells = [c.strip() for c in line.strip().strip("|").split(" | ")]
            log[cells[0]] = " ".join(cells[1:])
    number = re.compile(r"(?<![\w.])[+\-−]?\d[\d,]*(?:\.\d+)?%?|(?<![\w.])\.\d+")
    tokens = lambda s: {m.group(0).replace("−", "-").lstrip("+-").rstrip(",") for m in number.finditer(s.replace("Online Resource 1", ""))}   # noqa: E731
    for i, rid in enumerate(["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10", "D11", "5a", "5d"], 1):
        new = tokens(rows[i][1] + " " + rows[i][2]) - tokens(log[rid]) - {"2026", "25", "26"}      # years and the names of the windows are not results
        assert new == set(), (rid, sorted(new))                                                    # a number the log does not hold


@needs_tree
def test_item_15_the_demotion_rule_is_stated_with_the_level_divided_by_the_number_of_rows():
    text = _flat(_live(_read("esm/forward_tests.tex")))
    assert "rejects at the level \\ruleBinomLevel{} divided by the number of rows read; the same condition" in text


@needs_tree
def test_item_16_the_sealed_run_does_not_compute_failure_counts_at_other_targets_but_the_robustness_family_re_runs_h_b():
    text = _flat(_live(_read("esm/forward_tests.tex")))
    assert "and the failure counts at the other targets and in the narrower bands are development analyses that it does not compute" in text
    assert "other cutoff targets with the labels kept at" in text                                # the same paragraph re-runs H-B at other targets


@needs_tree
def test_item_17_the_sign_tests_carry_the_optimism_caveat():
    text = _flat(_live(_read("esm/audit_economic_details.tex")))                                    # the sign tests moved to the ESM (plan 2C task T2c)
    assert "The sign tests treat the nodes as independent although nodes within an operator co-move, so their $p$-values are optimistic." in text
    paper = _flat(_live(_read("sections/04_audit.tex")))                                           # the paper keeps the counts and no $p$-value of a sign test
    assert "\\signKYY{}" in paper and "\\signPYY{}" not in paper and "\\signPPostOpen{}" not in paper


@needs_tree
def test_item_18_the_input_monitors_are_uninformative_in_the_median_only_and_the_table_agrees():
    """Item 18 protected: the monitor claim is not overstated: the input monitors are uninformative in the median only, and the early alert rate is the only monitor with a positive
    median association. The ESM paragraph, Section 4.5 and the full monitor table are asserted as before. The discussion (6.1) says it in plain words since plan 2C ("of the
    label-free monitors we tested, only the share of hours flagged ... has a positive median association with the later drift, and the association is not the same in every
    quarter"); the test keys on those phrases and checks the claim against the macros of the medians and of the two quarterly correlations."""
    text = _flat(_live(_read("esm/ESM_1.tex")))                                                    # the monitor paragraphs moved to the ESM (plan 2C task T2c)
    assert "The input monitors are uninformative in the median" in text and "The input monitors are uninformative:" not in text
    assert "the only monitor with a positive median association" in text
    assert "In the 2024 to 2025 analysis the input monitors rank the nodes of the level block by their later error" in text
    assert re.search(r"only[^.]{0,80}monitor[^.]{0,60}positive median association|monitors[^.]{0,80}\bonly\b[^.]{0,80}positive median association",
                     _flat(_live(_read("sections/04_audit.tex"))))                                       # and the paper keeps the sentence
    # the claim, against the full monitor table of Online Resource 1: level block 2024-2025, PSI, W1 and domain classifier positive; early rate not
    rows = [line.replace("\\raggedright\\hyphenpenalty=10000 ", "") for line in _read("generated/tables/tab_monitors_full.tex").splitlines()]
    rows = [line for line in rows if line.startswith("2024-2025") and "absolute level" in line]
    assert len(rows) == 4
    by_monitor = {re.search(r"& (PSI|W1\\_norm|domain\\_auc|early\\_rate) &", r).group(1): r.split(" & ")[4] for r in rows}
    assert all(not by_monitor[m].startswith("\\textminus") for m in ("PSI", "W1\\_norm", "domain\\_auc"))
    assert by_monitor["early\\_rate"].startswith("\\textminus")
    # the discussion (6.1) says it in plain words since plan 2C ("the only monitor with a positive median association" is the sentence of Section 4.5); the claim is checked here
    # against the macros of Section 4.5: only the early alert rate has a positive median correlation with the later error, and the association differs between the quarters
    discussion = _flat(_live(_read("sections/07_discussion.tex")))
    assert re.search(r"label-free[^.]*\bonly\b[^.]{0,160}(?:median )?association|\bonly\b[^.]{0,160}label-free[^.]*association", discussion), \
        "6.1 no longer says that, of the label-free monitors, only the early alert rate is associated with the later drift"
    assert re.search(r"positive median association", discussion)
    assert re.search(r"not the same in every quarter|differs? (?:between|by|across) (?:the )?quarters|varies? (?:between|by|across) (?:the )?quarters", discussion)
    assert _number(_macro("monEarlyRateRhoMedian")) > 0 and all(_number(_macro(m)) < 0 for m in ("monPsiRhoMedian", "monWoneRhoMedian", "monDomainAucRhoMedian"))
    assert _number(_macro("monEarlyRateRhoAqThree")) > _number(_macro("monEarlyRateRhoAqFour")) >= 0


@needs_tree
def test_item_19_accuracy_of_five_statements():
    """Item 19 protected: five accurate statements (the ranking gain in 2026-H1, the factor of more than eight, the signs of the stratified NET, "every result" is generated, the
    definition of the expected shortfall). The fourth is asserted for the two cover letters here and for the introduction in its own test below (the introduction said "every number" until the controller fixed it, 26 Oct 2026, 05:51)."""
    audit = _flat(_live(_read("sections/04_audit.tex")))
    assert "give the only clear ranking gain in 2026-H1" in audit
    assert "a factor of more than eight" in audit and "\\medRRatioYY" not in audit
    assert 1.66 / 0.20 > 8 and float(_number(_macro("medRCombYY"))) / float(_number(_macro("medRCombYYInsample"))) > 8        # the printed values give 8.3
    econ = _flat(_live(_read("esm/ESM_1.tex")))                                                   # the stratified NET sentences moved to the ESM with the table that holds them (plan 2C task T2c)
    assert "negative, with an interval that excludes zero, in AQ3" in econ and "and the second half-year analysis (\\snetHTwo{}, \\snetHTwoCI{})" in econ
    assert "positive with an interval that contains zero in AQ1 (\\snetAqOne{}, \\snetAqOneCI{}) and the post-open 2026-H1 analysis (\\snetPostOpen{}, \\snetPostOpenCI{})" in econ
    sign = {w: _interval(_macro(f"snet{w}CI")) for w in ("AqThree", "AqFour", "HTwo", "AqOne", "PostOpen", "AqTwo", "YY")}
    assert all(hi < 0 for lo, hi in (sign[w] for w in ("AqThree", "AqFour", "HTwo")))             # negative, interval excludes zero
    assert all(lo < 0 < hi for lo, hi in (sign[w] for w in ("AqOne", "PostOpen", "AqTwo", "YY")))   # interval contains zero
    assert _number(_macro("snetAqOne")) > 0 and _number(_macro("snetPostOpen")) > 0 and _number(_macro("snetAqTwo")) < 0 and _number(_macro("snetYY")) < 0
    for name in ("cover_letter.md", "cover_letter_TestA_ran.md"):
        assert "Every result in the manuscript is generated from saved outputs" in _read(name) and "Every number in the manuscript" not in _read(name), name
    assert "the mean of the worst \\esFrac\\% of days as a positive loss" in _flat(_live(_read("sections/03_data_design.tex")))


@needs_tree
def test_item_19_the_introduction_says_every_result_and_not_every_number_is_generated():
    assert "every result in the paper is generated from saved outputs" in _flat(_live(_read("sections/01_introduction.tex")))


@needs_tree
def test_item_20_language():
    """Item 20 protected: plain language. "among it the day-of-year" stays out; the seasonality sentence stays in Sections 3 and 5 and in the Online Resource; the abstract never says
    "hours that alert" or "operators deploy" and defines the alert in plain words (its old positive phrases "the share of hours flagged" and "its users deploy a score" went with the
    old abstract; the new one says "a risk score per hour and a cutoff" and "its intended firing rate"); H-B1 and H-B2 are the portability contrasts of the operating-point error of
    two feature blocks (Section 3.6) and never "the ordering of two feature blocks"."""
    for name in SECTIONS:
        assert "among it the day-of-year" not in _flat(_live(_read(f"sections/{name}.tex"))), name
    assert "among it the day-of-year" not in _flat(_live(_read("esm/data_design_details.tex")))          # the details moved from Section 3 (plan 2C task T2b)
    for rel in ("sections/03_data_design.tex", "sections/05_economic.tex", "esm/forward_tests.tex"):          # the third was sections/06_sealed.tex
        assert "including the day-of-year seasonality that the model has learned from its training years" in _flat(_live(_read(rel))), rel
    from es.paper_lint import abstract_variants, _braced_arg
    body = _flat(_live(_read("sections/00_abstract.tex")))
    for v in abstract_variants(_braced_arg(body, "\\abstract")):
        # the old abstract said "the share of hours flagged" and "its users deploy a score"; the new one defines the alert and its rate in other plain words
        assert "hours that alert" not in v.text and re.search(r"A spike alert (?:warns of such hours|fires when a risk score exceeds a cutoff)", v.text) and re.search(r"risk score", v.text), v.label
        assert re.search(r"keeps its (?:intended )?(?:firing|alert) rate", v.text), v.label
        assert "operators deploy" not in v.text, v.label                                       # "operators" in the abstract are the four system operators
    sealed = _flat(_live(_read("esm/forward_tests.tex")))
    assert "at \\hePowerCurveSelcRhoThirty{} and above" in sealed and "from \\hePowerCurveSelcRhoThirty{}" not in sealed
    related = _flat(_live(_read("sections/02_related.tex")))
    assert "is expected to show" not in related and "its AUROC reaches \\monDomainPlaceboMax{} on the quarterly analogues" in related
    # H-B1 and H-B2 are described as the contrasts of the operating-point error of two blocks, never as "the ordering" (Section 3.6 holds the plain description since plan 2C;
    # the discussion has no sentence on them in the state before Test A, and the conclusion of the state after it says "estimate the portability contrasts")
    assert re.search(r"H-B1 and H-B2, the portability contrasts of the operating-point error of two feature blocks", _flat(_live(_read("sections/03_data_design.tex"))))
    for rel in ("sections/07_discussion.tex", "sections/09_conclusion.tex", "sections/03_data_design.tex"):
        assert "whether the ordering of two feature blocks" not in _flat(_live(_read(rel))), rel
    # (the ran conclusion no longer describes the block contrasts, the subject of its old assertion; 6.3 gives them as estimates for one quarter with approximate intervals)
    assert re.search(r"block contrasts of .{1,80} for H-B1 and .{1,80} for H-B2, estimates for one quarter with approximate intervals",
                     _subsection("sections/07_discussion.tex", "sec:disc-forward", **TEST_A_RAN))
