"""Plan 2C, task T3b: the wording rules of plan.md (section 2, W1 to W10) and Ruling 138, as tests on the text that the controller rewrote (abstract, introduction,
discussion and limitations, conclusion, back matter, the two cover letters).

What is read
* The main text: paper/sections/*.tex with the LaTeX comments removed (a note to the author is not text), the two cover letters (paper/cover_letter.md and
  paper/cover_letter_TestA_ran.md) with the HTML comments removed (the AUTHOR NOTES are not uploaded), and, when it exists and is not older than the files it was built from,
  the text of paper/main.pdf (pdftotext) up to its list of references. A stale PDF is skipped, not judged.
* The sources hold all the states of the Test A switches at once (\\ifTestA branches), so a rule on them holds for the manuscript before Test A, after it and when Test A did
  not run; the PDF holds the state that generated/macros.tex set when it was built. Nothing here depends on the state of the real tree, except the one test that compares the
  PDF with it (test_the_pdf_carries_the_state_that_macros_tex_sets).
* The Online Resource (paper/esm/) is read only where a test says so (W5 has its own, wider list of allowed phrases there).
* DART_PAPER_DIR_FOR_WORDING_TESTS may name a copy of paper/ (a dry run of these tests on a changed text or on the state in which Test A has run). It is not set in the suite.

The rules
* W2: no "better", "equivalent", "as good as", "at least as much/well", "outperform", "superior" (and, in the spirit of "as good as", no "no worse than") and no "give/provide no
  protection" in the main text or the letters. The hits that stay are the negations "a lack of evidence and not equivalence", listed one by one, with their lines, in ALLOWED_W2; a listed hit that is no
  longer in the text fails the test (no dead exception).
* W3: ESred is the tail effect of sitting out hours, not skill. (a) The places that define ESred say "tail effect" (Section 3.7.2, the note of Table 4, contribution (ii) of the
  introduction). (b) A sentence that reads the tail measure (it names ESred, the expected shortfall, the tail effect, tail protection or tail loss and says what it means, shows,
  protects or is evidence of) either anchors the reading in the tail effect of sitting out or denies it. (c) "skill" stands only in a negative or conditional sentence. A literal
  reading of "every sentence that defines or reads ESred" is not testable: the sentences that report values ("ESred is positive in AQ3") name no effect.
* W5: no "preregistered", "pre-registered" and no "registered" (nor any other form of "regist") in the main text or the letters except in the exact phrases "not registered with a
  third party", "no third party holds a registered" and "registered or time-stamped copy"; the Online Resource may also say "No third party registered the protocol", "No third
  party has registered the protocol" and name the file REGISTRATION.md. "pre-specified" stands in the abstract, the introduction, Section 6.3 and the conclusion in every state of
  Test A. No sentence says that a test of this paper "confirms" anything.
* W7: no sentence says that the alerts work(s), add(s) value or should be deployed. W8: the strict lint finds no error of its text rules (numbers, headings, ...) and the sign
  statements name the quarters and agree with the macros. Ruling 138: no "tracks"/"tracking". W9: a sentence that says that the alerts rank or identify spike hours says
  "calendar" too. The abstract has at most 250 words in every state of Test A (an interval macro counts as two words), and the introduction defines DART and spike alert before it
  uses them.
* The section numbers that the cover letters type (3.5, 3.1, 6.3) are the numbers of sec:data-impl, sec:data-panel and sec:disc-forward, in the sources and in main.aux; so are
  the numbers that the author's checklist and the corrections file (uploaded) type.
* The sentences that carry the three states of Test A exist in each state and compile to text without \\if, \\else or \\fi.

How the tests are written. The tests key on phrases and on the quarters that a sentence lists, not on whole sentences, so that the controller can reword around them. Each scanner is
tested on small texts first (what it finds and what it lets through), so that a scan of the real folder cannot pass because the scanner finds nothing. A text that breaks a rule fails
the test that names it and the sentence. (Four such texts were found on 26 October 2026 and fixed by the controller at 05:51: "Neither test is registered with a third party", the
word "pre-specified" missing from the introduction, 6.3 and the conclusion before the run, "every number" in the introduction, and a missing note in the Test A letter.) The report of
task T3b lists the mutations of the real text that were made on a copy of paper/ (DART_PAPER_DIR_FOR_WORDING_TESTS) and caught."""
import functools
import os
import re
import subprocess
from pathlib import Path
from typing import NamedTuple

import pytest

PAPER = Path(os.environ.get("DART_PAPER_DIR_FOR_WORDING_TESTS") or Path(__file__).resolve().parents[3] / "paper")
needs_tree = pytest.mark.skipif(not (PAPER / "sections").exists(), reason="the manuscript folder is not next to dart_dev")
PDFTOTEXT = Path("/opt/homebrew/bin/pdftotext")
LETTERS = ("cover_letter.md", "cover_letter_TestA_ran.md")
QUARTER = {"AqThree": "2025-Q3", "AqFour": "2025-Q4", "AqOne": "2026-Q1", "AqTwo": "2026-Q2"}       # in chronological order, as the text lists them
STATES = {"absent": dict(TestA=False, TestAdidnotrun=False), "ran": dict(TestA=True, TestAdidnotrun=False),
          "did_not_run": dict(TestA=True, TestAdidnotrun=True)}


# ------------------------------------------------------------------------------------------------------ reading the text
def _read(rel: str) -> str:
    return (PAPER / rel).read_text(encoding="utf-8")


def _live_tex(text: str) -> str:
    """The text without its LaTeX comments, full-line and trailing (a percent sign after a backslash is a percent sign)."""
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in text.splitlines())


def _live_md(text: str) -> str:
    """A letter as it is uploaded: without its HTML comments (the AUTHOR NOTES)."""
    return re.sub(r"<!--.*?-->", "", text, flags=re.S)


def _flat(text: str) -> str:
    return " ".join(text.split())


_KEYS = re.compile(r"\\(?:label|ref|eqref|cite[a-z]*|esmref|mainref|gentable|genfigure|input|includegraphics)\*?(?:\[[^\]]*\])?\{[^}]*\}")


def _printed(text: str) -> str:
    """The words of a LaTeX text as a reader would meet them: the keys of references and labels and the names of macros and commands are not text."""
    return re.sub(r"\\[A-Za-z]+\*?", " ", _KEYS.sub(" ", text))


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\\(\[`\u201c*])")


def _sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_END.split(_flat(text)) if s]


def sentence_hits(text: str, pattern: re.Pattern) -> list[tuple[str, str]]:
    """[(the matched words, the sentence)] of every match of ``pattern`` in the printed words of ``text``."""
    return [(m.group(0), s) for s in _sentences(_printed(text)) for m in pattern.finditer(s)]


@functools.lru_cache(maxsize=None)
def main_text_sources() -> dict[str, str]:
    """{name: live text} of paper/sections/*.tex and of the two cover letters as uploaded."""
    out = {f"sections/{p.name}": _live_tex(p.read_text(encoding="utf-8")) for p in sorted((PAPER / "sections").glob("*.tex"))}
    out.update({name: _live_md(_read(name)) for name in LETTERS})
    return out


def _pdf_inputs() -> list[Path]:
    return ([PAPER / "main.tex", PAPER / "crossref.tex", PAPER / "generated" / "macros.tex"] + sorted((PAPER / "sections").glob("*.tex"))
            + sorted((PAPER / "generated" / "tables").glob("*.tex")) + sorted((PAPER / "generated" / "figures").glob("*.pdf")))


@functools.lru_cache(maxsize=None)
def _pdf_text_cached(mtime: float) -> str:
    raw = subprocess.run([str(PDFTOTEXT), str(PAPER / "main.pdf"), "-"], capture_output=True, text=True, timeout=120).stdout
    parts = re.split(r"\n\x0c?References\n", raw)                                  # the heading may open a page: pdftotext then puts a form feed before it
    assert len(parts) == 2, "main.pdf has no heading 'References' on a line of its own: the text of the list of references cannot be cut off"
    return _flat(parts[0])


def pdf_text() -> str:
    """The text of main.pdf up to its references, or a skip when there is no PDF, no pdftotext, or the PDF is older than a file it was built from."""
    pdf = PAPER / "main.pdf"
    if not pdf.exists() or not PDFTOTEXT.exists():
        pytest.skip("no main.pdf or no pdftotext: nothing to scan")
    newest = max(p.stat().st_mtime for p in _pdf_inputs() if p.exists())
    if pdf.stat().st_mtime < newest:
        pytest.skip("main.pdf is older than the files it is built from: rebuild it (./build.sh pdf) and the scan of its text runs")
    return _pdf_text_cached(pdf.stat().st_mtime)


def _subsection(text: str, label: str) -> str:
    """The part of ``text`` from \\label{label} to the next \\subsection or \\section (or the end)."""
    start = text.index("\\label{%s}" % label)
    ends = [e for e in (text.find(h, start) for h in ("\\subsection{", "\\section{")) if e >= 0]
    return text[start:min(ends, default=len(text))]


def resolved(rel: str, state: str) -> str:
    """The live text of a file as a reader sees it in one state of Test A: the \\ifTestA ... \\fi branches that the state skips are gone, and the switches with them (the paper's own
    parser, es.paper_lint; the switch is set here, not read from generated/macros.tex)."""
    from es.paper_lint import SWITCHES, _resolve_switches
    setting = dict.fromkeys(SWITCHES, False)
    setting.update(STATES[state])
    text = _resolve_switches(_live_tex(_read(rel)), setting)
    assert text is not None, f"{rel}: the \\if...\\fi switches are unbalanced or test something other than a Test A or Test B switch"
    return text


def _macro(name: str) -> str:
    m = re.search(r"^\\newcommand\{\\" + name + r"\}\{(.*)\}\s*$", _read("generated/macros.tex"), flags=re.M)
    assert m, name
    return m.group(1)


def _number(name: str) -> float:
    """A dollar, percent or count macro as a float: \\textminus{}\\$4{,}111 is -4111, 9.9\\% is 9.9."""
    return _to_float(_macro(name))


def _to_float(text: str) -> float:
    return float(text.replace("\\textminus{}", "-").replace("\\$", "").replace("{,}", "").replace("\\%", "").replace("\\", ""))


def _interval(name: str) -> tuple[float, float]:
    lo, hi = re.fullmatch(r"\[(.*), (.*)\]", _macro(name)).groups()
    return _to_float(lo), _to_float(hi)


def _and(suffixes) -> str:
    """The quarters of ``suffixes`` as the text lists them: "2025-Q3, 2026-Q1 and 2026-Q2"."""
    names = [QUARTER[s] for s in suffixes]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


# ------------------------------------------------------------------------------------------------------ W2: no comparison words
W2_WORDS = re.compile(r"\b(?:better|equivalen(?:t|ts|ce|ces)|as good as|at least as (?:much|well)|outperform(?:s|ed|ing|ance)?|superior(?:ity)?|(?:no|not) worse than)\b"
                      r"|\b(?:give|gives|giving|provide|provides|providing)\s+no\s+protection\b", re.I)


class Allowed(NamedTuple):
    file: str
    line: int          # the line of the hit in the file when the list was made (26 October 2026); the test matches the words, not the line, so that an edit above does not break it
    pattern: str       # a regular expression for the words of the sentence that the test looks for (key words, not the whole sentence: the wording is the controller's to change)
    why: str

    def allows(self, sentence: str, name: str | None = None) -> bool:
        return name in (None, self.file) and re.search(self.pattern, sentence) is not None


#: Every hit of W2_WORDS in the main text and the letters that the test allows. Each is a negation: an interval that contains zero is a lack of evidence and not equivalence.
DENIES_EQUIVALENCE = r"\bnot (?:evidence of |as )?equivalen"                       # "not equivalence", "not evidence of equivalence", "not as equivalence"
ALLOWED_W2 = (
    Allowed("sections/05_economic.tex", 56, DENIES_EQUIVALENCE, "the end of the caption of Figure 3 (task T1): it denies equivalence"),
    Allowed("sections/05_economic.tex", 59, DENIES_EQUIVALENCE, "Section 5.4, required rewording 3 (task T2c): it denies equivalence"),
    Allowed("cover_letter.md", 11, DENIES_EQUIVALENCE, "'What the paper finds' of the letter: it denies equivalence"),
    Allowed("cover_letter_TestA_ran.md", 11, DENIES_EQUIVALENCE, "'What the paper finds' of the letter: it denies equivalence"),
)


def test_the_w2_scanner_finds_the_words_and_lets_the_wording_rules_pass():
    for bad in ("The alerts are better than a calendar-only rule.", "The alerts are significantly better than the rule.", "The two rules are equivalent.",
                "The alerts are as good as the calendar.", "The alerts provide at least as much protection.", "The alerts outperform the calendar-only rule.",
                "The alerts are superior.", "The alerts give no protection.", "The alerts provide no protection beyond the calendar.", "The alerts are no worse than the rule.",
                "A wide interval shows equivalence."):
        assert sentence_hits(bad, W2_WORDS), bad
    for good in ("The paired difference in ESred is negative in 2025-Q3.", "At the same number of alerts the alerts show no demonstrated improvement over a calendar-only rule.",
                 "The rolling cutoff raises the rate and the fixed cutoff lowers it.", "The data do not show protection beyond the calendar.",
                 "A wide interval is a lack of evidence.", "The best practice is described by Lago."):
        assert not sentence_hits(good, W2_WORDS), good


@needs_tree
def test_w2_the_main_text_and_the_letters_have_no_comparison_word_except_the_listed_negations():
    used = set()
    for name, text in main_text_sources().items():
        for word, sentence in sentence_hits(text, W2_WORDS):
            found = [a for a in ALLOWED_W2 if a.allows(sentence, name)]
            assert found, f"{name}: {word!r} in: {sentence}"
            used.update(found)
    assert used == set(ALLOWED_W2), f"allowed hits that are no longer in the text (delete them): {sorted(set(ALLOWED_W2) - used)}"


# ------------------------------------------------------------------------------------------------------ W3: ESred is the tail effect of sitting out, not skill
SKILL = re.compile(r"\bskill", re.I)
#: The one sentence that says "skill" without a negation: it defines skill for NET (Section 3.7.2) and makes no claim about ESred or the alerts.
ALLOWED_SKILL = (Allowed("sections/03_data_design.tex", 112, r"\bSkill is the part of NET beyond random selection", "the definition of skill for NET: the alerts against random selection within strata"),)
NEGATIVE_OR_CONDITIONAL = re.compile(r"\b(?:not|no|never|neither|nor|without|cannot|would|could|if|unless|whether)\b|n't", re.I)
TAIL_SUBJECT = re.compile(r"ESred|expected[- ]shortfall|tail effect|tail protection|tail[- ]risk|tail loss", re.I)
READING = re.compile(r"\bmeans\b|\bmean that\b|\b(?:shows?|showed|evidence|reflects?|indicates?)\b|\bprotect\w*", re.I)
TAIL_ANCHOR = re.compile(r"tail effect|sitting out|sits out|sit out", re.I)


def skill_violations(text: str, name: str | None = None) -> list[str]:
    """The sentences that say "skill" without a negation or a condition, except the listed definition (of the file ``name``, or of any file when it is None)."""
    return [s for s in _sentences(_printed(text)) if SKILL.search(s) and not NEGATIVE_OR_CONDITIONAL.search(s)
            and not any(a.allows(s, name) for a in ALLOWED_SKILL)]


def reading_violations(text: str) -> tuple[int, list[str]]:
    """(how many sentences read the tail measure, those of them that neither say 'tail effect' / 'sitting out' nor deny the reading)."""
    reading = [s for s in _sentences(_printed(text)) if TAIL_SUBJECT.search(s) and READING.search(s)]
    return len(reading), [s for s in reading if not (TAIL_ANCHOR.search(s) or NEGATIVE_OR_CONDITIONAL.search(s))]


def test_the_w3_scanners_find_a_claim_of_skill_or_protection_and_let_the_denials_pass():
    for bad in ("The alerts show skill in the tail.", "ESred measures the skill of the alerts.", "Their skill is clear."):
        assert skill_violations(bad), bad
    assert not skill_violations("Skill is the part of NET beyond random selection, which we isolate within strata.", "sections/03_data_design.tex")
    assert skill_violations("Skill is the part of NET beyond random selection, which we isolate within strata.", "sections/05_economic.tex")
    for good in ("ESred alone is not evidence of skill.", "No within-cell timing skill appears in development.", "A positive SELC would show skill beyond the calendar if it held.",
                 "What the alerts do not show is timing skill within calendar cells."):
        assert not skill_violations(good), good
    for bad in ("ESred shows that the alerts protect the portfolio.", "A positive ESred means that the alerts protect the tail.", "The expected shortfall is evidence of protection."):
        assert reading_violations(bad)[1], bad
    for good in ("A positive ESred means that sitting out the alerted hours reduced the tail loss.", "ESred is the tail effect of sitting out and is evidence of nothing else.",
                 "The tail effect does not show that the alerts protect beyond the calendar.", "ESred is positive in AQ3.",
                 "The tail-risk measure is the mean of the worst days of the expected shortfall."):
        assert not reading_violations(good)[1], good


@needs_tree
def test_w3_skill_stands_only_in_negative_or_conditional_sentences():
    seen = 0
    used = False
    for name, text in main_text_sources().items():
        assert skill_violations(text, name) == [], name
        seen += len([s for s in _sentences(_printed(text)) if SKILL.search(s)])
        used = used or any(a.allows(s) for a in ALLOWED_SKILL for s in _sentences(_printed(text)))
    assert seen >= 8 and used, (seen, used)                                                                 # the scan reads the sentences of Sections 3 and 5 and finds the definition
    assert any(SKILL.search(s) and re.search(r"ESred", s) for s in _sentences(_printed(main_text_sources()["sections/05_economic.tex"]))), "no sentence on ESred and skill"


@needs_tree
def test_w3_a_sentence_that_reads_the_tail_measure_names_the_tail_effect_of_sitting_out_or_denies_the_reading():
    seen = 0
    for name, text in main_text_sources().items():
        n, bad = reading_violations(text)
        assert bad == [], name
        seen += n
    assert seen >= 10, seen                                                                                 # the scan finds the definitions, the notes and the summaries


@needs_tree
def test_w3_the_places_that_define_esred_say_that_it_is_the_tail_effect_of_sitting_out():
    data = _flat(main_text_sources()["sections/03_data_design.tex"])
    definition = re.search(r"ESred is the expected shortfall without alerts minus the expected shortfall with alerts", data)
    assert definition, "Section 3.7.2 no longer defines ESred as the expected shortfall without alerts minus the one with alerts"
    around = data[definition.start():definition.start() + 700]
    assert re.search(r"positive ESred means that sitting out the alerted hours reduced the tail loss", around)       # the definition itself reads it as the effect of sitting out
    assert re.search(r"tail effect of (?:the policy|sitting out)", around) and re.search(r"not evidence of skill", around)   # and the next sentence calls it the tail effect, not skill
    econ = _flat(main_text_sources()["sections/05_economic.tex"])
    assert re.search(r"ESred: reduction of the expected shortfall of daily profit and loss, the tail effect of sitting out and not evidence of skill", econ)     # the note of Table 4
    intro = _flat(main_text_sources()["sections/01_introduction.tex"])
    assert re.search(r"tail effect of sitting out hours", intro)                                                 # contribution (ii)
    # every sentence of the sources that defines ESred with a copula or a colon (not "ESred is positive") says it or has the next sentence say it
    defined = 0
    for name, text in main_text_sources().items():
        sentences = _sentences(_printed(text))
        for i, s in enumerate(sentences):
            if re.search(r"\bESred\s+is\s+(?:the|defined)\b|\bESred:\s+reduction\b", s):                 # "the ESred: positive for the SELC" lists a result and defines nothing
                defined += 1
                assert TAIL_ANCHOR.search(" ".join(sentences[i:i + 2])), (name, s)
    assert defined >= 2, defined


# ------------------------------------------------------------------------------------------------------ W5: pre-specified, never registered
REGISTER_WORD = re.compile(r"\b(?:pre-?)?regist\w*|\bpreregist\w*", re.I)
W5_PHRASES = ("not registered with a third party", "no third party holds a registered", "registered or time-stamped copy")
W5_ESM_PHRASES = W5_PHRASES + ("no third party registered the protocol", "no third party has registered the protocol", "docs/sealed/registration.md")
CONFIRM = re.compile(r"\bconfirm(?:s|ed|ing)?\b|\bconfirmatory\s+(?:result|evidence|finding|support|outcome)", re.I)


def registered_hits(text: str, phrases=W5_PHRASES) -> list[str]:
    """The words around every form of "regist" that is not inside an allowed phrase (the phrases are matched ignoring case and white space)."""
    flat = _flat(_printed(text))
    for phrase in phrases:
        flat = re.sub(re.escape(phrase), " ", flat, flags=re.I)
    return [flat[max(0, m.start() - 40):m.end() + 40] for m in REGISTER_WORD.finditer(flat)]


def test_the_w5_scanner_finds_registration_words_and_lets_the_three_phrases_pass():
    for bad in ("The two tests are preregistered.", "The two tests are pre-registered.", "The protocol was registered on OSF.", "The registration is public.",
                "Neither was registered with a third party.", "The registered family is H-E alone.", "A preregistration of the design."):
        assert registered_hits(bad), bad
    for good in ("The forward tests are pre-specified, have not been run and are not registered with a third party.",
                 "No third party holds a registered or time-stamped copy of the protocol.", "No third party holds a registered copy of its protocol.",
                 "The design is pre-specified.", "The protocol record is dated \\registeredUtcTestA{} (Table~\\ref{tab:registration})."):
        assert not registered_hits(good), good
    assert registered_hits("No third party registered the protocol.") and not registered_hits("No third party registered the protocol.", W5_ESM_PHRASES)
    for bad in ("The test confirms the result.", "H-E was confirmed.", "A confirmatory result follows.", "It is confirming."):
        assert CONFIRM.search(bad), bad
    for good in ("The one confirmatory test is decided only on the pooled window.", "No row is confirmatory."):
        assert not CONFIRM.search(good), good


def _w5_cases():
    return sorted(main_text_sources()) if (PAPER / "sections").exists() else []


@needs_tree
@pytest.mark.parametrize("name", _w5_cases())
def test_w5_registered_stands_only_in_the_three_denials(name):
    assert registered_hits(main_text_sources()[name]) == [], name


@needs_tree
def test_w5_the_sources_and_the_letters_claim_no_confirmation():
    for name, text in main_text_sources().items():
        assert not CONFIRM.search(_printed(text)), (name, CONFIRM.search(_printed(text)).group(0))


@needs_tree
def test_w5_the_online_resource_has_registered_only_in_the_denials_and_the_name_of_the_local_record():
    """The ESM may also say that no third party registered the protocol and name docs/sealed/REGISTRATION.md, the local record. (Its keys and macro names are not text.)"""
    seen = 0
    for path in sorted((PAPER / "esm").glob("*.tex")):
        text = _live_tex(path.read_text(encoding="utf-8"))
        assert registered_hits(text, W5_ESM_PHRASES) == [], path.name
        seen += len(re.findall(r"registered", _printed(text), flags=re.I))
    assert seen >= 2, seen                                                                                  # the scan reaches the two denials of esm/forward_tests.tex


PLACES_OF_PRE_SPECIFIED = (("the abstract", "sections/00_abstract.tex", None), ("the introduction", "sections/01_introduction.tex", None),
                           ("Section 6.3", "sections/07_discussion.tex", "sec:disc-forward"), ("the conclusion", "sections/09_conclusion.tex", None))


def _w5_pre_specified_cases():
    if not (PAPER / "sections").exists():
        return []
    return [pytest.param(place, rel, label, state, id=f"{place}-{state}") for place, rel, label in PLACES_OF_PRE_SPECIFIED for state in STATES]


@needs_tree
@pytest.mark.parametrize("place,rel,label,state", _w5_pre_specified_cases())
def test_w5_the_forward_tests_are_called_pre_specified_where_the_rule_asks_for_it(place, rel, label, state):
    text = resolved(rel, state)
    text = _subsection(text, label) if label else text
    assert "pre-specified" in _flat(text), f"{place} does not say 'pre-specified' when Test A is in the state {state!r}"


# ------------------------------------------------------------------------------------------------------ "forward", not "sealed" (the style pass of plan 2C)
#: In the main text and the letters the tests are "forward tests" and "forward windows"; the Online Resource keeps "sealed" for the protocol's own objects (the sealed pull, the sealed
#: run), and the keys of \esmref (sealed, sealed-record, ...) are names, not text. The one generic use that stays says that the opened 2026-H1 holdout is weaker evidence than a test that
#: nobody had opened.
SEALED_OBJECT = re.compile(r"\bsealed (?:tests?|quarters?|windows?|values?|results?|runs?|pulls?|downloads?)\b", re.I)
ALLOWED_SEALED = (
    Allowed("sections/04_audit.tex", 75, r"\bfully sealed test\b", "Section 4.6: the opened holdout is weaker evidence than a test that nobody had opened"),
    Allowed("sections/07_discussion.tex", 59, r"\bfully sealed test\b", "Section 6.4.2: the same sentence"),
)


def test_the_sealed_scanner_finds_the_old_wording_and_lets_the_generic_use_pass():
    for bad in ("The two sealed quarters are pending.", "Test A is a sealed test.", "The sealed windows are not opened.", "The sealed pull needs the hash."):
        assert sentence_hits(bad, SEALED_OBJECT), bad
    for good in ("The forward tests are pending.", "Its 2026-H1 holdout was less sealed than described.", "The key is \\esmref{sealed}."):
        assert not sentence_hits(good, SEALED_OBJECT), good


@needs_tree
def test_the_main_text_and_the_letters_call_the_tests_forward_and_not_sealed():
    used = set()
    for name, text in main_text_sources().items():
        for word, sentence in sentence_hits(text, SEALED_OBJECT):
            found = [a for a in ALLOWED_SEALED if a.allows(sentence, name)]
            assert found, f"{name}: {word!r} in: {sentence}"
            used.update(found)
    assert used == set(ALLOWED_SEALED), f"allowed uses that are no longer in the text (delete them): {sorted(set(ALLOWED_SEALED) - used)}"


# ------------------------------------------------------------------------------------------------------ W7: the alerts do not "work", "add value", "get deployed"
_NOUN_WORK = re.compile(r"\b(?:this|the|our|their|its|related|prior|previous|recent|earlier|empirical|further|future|other|such|more|that|of|any|some|much)\s+(?:\d+\s+)?work\b", re.I)
W7_PATTERNS = (re.compile(r"\bworks\b|\bworked\b", re.I),
               re.compile(r"\b(?:alerts?|spike alerts?|they|it|this|these|cutoffs?|rules?|policy|policies|approach|method)\s+(?:(?:also|really|actually|do|does|did|can|could|will|would|still)\s+)?work\b", re.I),
               re.compile(r"\badds?\s+(?:\w+\s+){0,2}value\b|\badded value\b|\bvalue[- ]adding\b", re.I),
               re.compile(r"\bshould\s+(?:\w+\s+)?deploy\w*|\bworth\s+deploying\b", re.I))


def w7_hits(text: str) -> list[str]:
    out = []
    for s in _sentences(_printed(text)):
        s = _NOUN_WORK.sub(" ", s)
        out += [s for pattern in W7_PATTERNS if pattern.search(s)]
    return out


def test_the_w7_scanner_finds_the_claims_and_lets_the_nouns_pass():
    for bad in ("The alerts work.", "The alerts also work in 2025-Q3.", "This approach works.", "The alerts worked in the second half-year.", "The alerts add value.",
                "The alerts add real value to the position.", "They should be deployed.", "The alerts should deploy at 5%.", "The alerts are worth deploying.",
                "A spike alert adds some value."):
        assert w7_hits(bad), bad
    for good in ("Recent work in this journal evaluates forecasts.", "We are not aware of work that asks this question.", "An earlier version of this work opened a holdout.",
                 "The use of AI assistants in the work is described in Section 3.5.", "P4 and P5 are deployable, since they use only scores up to the day before.",
                 "Empirical 3 work on the Midwest finds frictions.", "The cutoff is deployed on a calendar basis."):
        assert not w7_hits(good), good


@needs_tree
def test_w7_no_sentence_says_that_the_alerts_work_add_value_or_should_be_deployed():
    for name, text in main_text_sources().items():
        assert w7_hits(text) == [], name


# ------------------------------------------------------------------------------------------------------ W9: only the calendar ranks or identifies
W9_PATTERN = re.compile(r"\balerts?\s+(?:also\s+|can\s+|do\s+|does\s+|will\s+|reliably\s+|correctly\s+)?rank\w*|\bidentif(?:y|ies|ying|ied)\s+(?:the\s+)?(?:spike\s+)?hours\b"
                        r"|\brank(?:s|ed|ing)?\s+(?:the\s+)?(?:spike\s+)?hours\b|\b(?:predict|forecast|anticipate|pick out)s?\s+(?:the\s+)?spike\s+hours\b", re.I)


def w9_violations(text: str) -> list[str]:
    return [s for s in _sentences(_printed(text)) if W9_PATTERN.search(s) and not re.search(r"calendar", s, re.I)]


def test_the_w9_scanner_finds_the_credit_and_lets_the_calendar_sentences_pass():
    for bad in ("The alerts rank spike hours.", "The alerts rank the hours well.", "The alerts can identify spike hours.", "The block identifies the hours of spikes.",
                "The model ranks the spike hours first.", "The alerts predict spike hours."):
        assert w9_violations(bad), bad
    for good in ("The alerts rank spike hours only as well as the calendar does.", "The calendar identifies spike hours.", "The price-history blocks mostly add no ranking beyond the calendar.",
                 "A spike alert warns that an hour looks like one.", "The alert that flags the exposed hours drifts."):
        assert not w9_violations(good), good


@needs_tree
def test_w9_no_sentence_credits_the_alerts_with_ranking_or_identifying_spike_hours_without_the_calendar():
    for name, text in main_text_sources().items():
        assert w9_violations(text) == [], name


# ------------------------------------------------------------------------------------------------------ no "the first is positive, as in" (controller, 26 Oct 2026, 06:16)
#: "The first is positive, as in 2026-Q1, and the second negative, as in ..." does not say which paired difference is meant; 6.3 names each by its statistic. No sentence of the main
#: text or of the letters may use the pair.
FIRST_IS_SIGNED = re.compile(r"\bthe first (?:is|was|are) (?:positive|negative)\b[^.;]{0,20}\bas in\b", re.I)


def test_the_first_is_positive_scanner_finds_the_pair_and_lets_the_named_statistics_pass():
    for bad in ("The first is positive, as in 2026-Q1 and 2026-Q2, and the second negative.", "The first was negative as in 2025-Q3.", "the first is positive, as in the other quarters"):
        assert sentence_hits(bad, FIRST_IS_SIGNED), bad
    for good in ("The paired difference in the excess loss capture was negative in 2025-Q3 and is positive in Test A.", "The first quarter is reported as an interim result.",
                 "The first half of 2026 was already used once, as in the earlier version."):
        assert not sentence_hits(good, FIRST_IS_SIGNED), good


@needs_tree
def test_no_sentence_of_the_main_text_says_the_first_is_positive_as_in():
    for name, text in main_text_sources().items():
        assert not sentence_hits(text, FIRST_IS_SIGNED), (name, sentence_hits(text, FIRST_IS_SIGNED))


# ------------------------------------------------------------------------------------------------------ Ruling 138: nothing "tracks"
TRACK = re.compile(r"\btrack\w*", re.I)


def test_the_ruling_138_scanner_finds_every_form():
    for bad in ("The rolling cutoff tracks the target.", "Tracking the rate is not enough.", "It tracked the market.", "To track drift, watch the rate."):
        assert TRACK.search(bad), bad
    assert not TRACK.search("The cutoff follows the target.")


@needs_tree
def test_ruling_138_the_main_text_has_neither_tracks_nor_tracking():
    for name, text in main_text_sources().items():
        assert not TRACK.search(_printed(text)), (name, TRACK.search(_printed(text)).group(0))


# ------------------------------------------------------------------------------------------------------ the text of the PDF
#: Each wording rule applied to the text of main.pdf (its references cut off). The PDF holds the generated tables, the values of the macros and the text of the figures, which the
#: sources do not show; the allowed hits of W2 are matched by their words in any file.
PDF_RULES = {
    "W2": lambda t: [f"{w!r} in: {s}" for w, s in sentence_hits(t, W2_WORDS) if not any(a.allows(s) for a in ALLOWED_W2)],
    "W3 skill": skill_violations,
    "W3 reading": lambda t: reading_violations(t)[1],
    "W5 registered": registered_hits,
    "W5 confirm": lambda t: [m.group(0) for m in CONFIRM.finditer(t)],
    "W7": w7_hits,
    "W9": w9_violations,
    "Ruling 138": lambda t: [m.group(0) for m in TRACK.finditer(t)],
    "the first is positive, as in": lambda t: [w for w, _ in sentence_hits(t, FIRST_IS_SIGNED)],
    "forward not sealed": lambda t: [f"{w!r} in: {s}" for w, s in sentence_hits(t, SEALED_OBJECT) if not any(a.allows(s) for a in ALLOWED_SEALED)],
}


def _pdf_rule_cases():
    return sorted(PDF_RULES)


@needs_tree
@pytest.mark.parametrize("rule", _pdf_rule_cases())
def test_the_text_of_the_pdf_passes_the_scanner_of_every_wording_rule(rule):
    assert PDF_RULES[rule](pdf_text()) == [], rule


@needs_tree
def test_the_pdf_scan_reads_the_text_it_is_meant_to_read():
    """The scan is not blind: the PDF text has the abstract, the headings of Sections 1 to 7, the tables' ESred note and the declarations, and stops before the references."""
    text = pdf_text()
    for words in ("Abstract", "Keywords:", "1 Introduction", "6 Discussion and limitations", "7 Conclusion", "Statements and Declarations", "Prior dissemination",
                  "ESred: reduction of the expected shortfall of daily profit and loss", "Fig. 3 Paired contrasts"):
        assert words in text, words
    assert "References" not in text.split("Prior dissemination", 1)[1] and "Isemonger" not in text


# ------------------------------------------------------------------------------------------------------ W8: no typed number, sign statements name the quarters
@needs_tree
def test_w8_the_strict_lint_finds_no_error_of_its_text_rules_and_only_the_authors_values_are_pending():
    """lint_all(strict=True) cannot be free of errors before the author has filled in three values (\\authorDept, \\authorOrcid, \\codeRepoURL) and before Test A has run (its macros
    are pending). The rules of the text (the numbers rule above all: no digit that is not a macro, a year, a quarter label, a reference or a '% lint:allow' token, and the rules on
    headings, citations, generated files, declarations, the abstract and the keywords) must have no error; a pending Test A macro is accepted while Test A is absent
    (pending_ok), and the only pending errors are the author's."""
    from es.paper_lint import lint_all
    errors = [f for f in lint_all(PAPER, strict=True, pending_ok=("A",)) if f.severity == "error"]
    rules = [f for f in errors if f.rule != "pending"]
    assert rules == [], "\n".join(str(f) for f in rules)
    pending = [f for f in errors if f.rule == "pending"]
    assert all(re.search(r"is pending \(AUTHOR\)", f.message) for f in pending), "\n".join(str(f) for f in pending if "(AUTHOR)" not in f.message)
    assert not [f for f in lint_all(PAPER, strict=False) if f.severity == "error"]


QUARTER_LIST = r"((?:20\d\d-Q[1-4](?:, | and )?)+)"                      # "2025-Q3, 2025-Q4 and 2026-Q1"


def _named(listing: str) -> list[str]:
    """The quarters of a listing as the suffixes of the macros, in the order of the text."""
    by_name = {name: suffix for suffix, name in QUARTER.items()}
    return [by_name[n] for n in re.findall(r"20\d\d-Q[1-4]", listing)]


def quarters_after(text: str, before: str, after: str = "") -> list[str]:
    """The quarters that ``text`` names between the regular expressions ``before`` and ``after``. The tests key on these phrases and not on whole sentences, so that the
    controller can reword around them; a phrase that is gone fails with the start of the text."""
    m = re.search(before + QUARTER_LIST + after, text)
    assert m, f"no text matching {before + '<quarters>' + after!r} in: {text[:400]}"
    return _named(m.group(1))


def test_the_quarter_lists_are_read_in_the_order_of_the_text():
    assert _named("2025-Q3, 2026-Q1 and 2026-Q2") == ["AqThree", "AqOne", "AqTwo"]
    assert quarters_after("it is negative in 2025-Q3 and 2025-Q4 and positive in 2026-Q2.", "negative in ", " and positive") == ["AqThree", "AqFour"]
    assert quarters_after("lower in 2026-Q1.", "lower in ") == ["AqOne"]
    with pytest.raises(AssertionError):
        quarters_after("nothing here", "negative in ")


@needs_tree
def test_w8_the_sign_statements_of_the_abstract_name_the_quarters_and_agree_with_the_macros():
    from es.paper_lint import abstract_variants, _braced_arg, _strip_comment
    body = " ".join(_strip_comment(line) for line in _read("sections/00_abstract.tex").splitlines())
    negative = [q for q in QUARTER if _number(f"desredCalCount{q}") < 0]
    excluding = [q for q in QUARTER if _interval(f"desredCalCount{q}CI")[0] > 0 or _interval(f"desredCalCount{q}CI")[1] < 0]
    net_negative = [q for q in QUARTER if _number(f"net{q}") < 0]
    assert negative == net_negative and len(negative) == 3 and excluding == ["AqFour"]                   # the abstract says "the same three quarters" and "only in 2025-Q4"
    for v in abstract_variants(_braced_arg(body, "\\abstract")):
        text = _flat(v.text)
        assert quarters_after(text, r"expected-shortfall reduction is negative in ") == negative, v.label
        assert quarters_after(text, r"excludes zero only in ") == excluding, v.label
        net = re.search(r"net value is negative in (?:the same three quarters|" + QUARTER_LIST + ")", text)      # (the abstract may leave NET out; if it says it, it must be right)
        if net:
            assert (_named(net.group(1)) if net.group(1) else negative) == net_negative, v.label       # "the same three quarters" are the ones named before
        assert re.search(r"\bat none\b|\bnone\b|\\driftOutPFourYY\{\}", text), v.label                       # a fixed cutoff fails at 12 nodes and a re-estimated cutoff at none
        assert re.search(r"eight-week window", text) and "\\driftWinOutMeanPFourYY{}" in text and "\\driftWinOutMeanPOneYY{}" in text, v.label
    assert _number("driftOutPFourYY") == 0 and _number("driftWindowWeeks") == 8
    assert 0 < _number("driftWinOutMeanPFourYY") < _number("driftWinOutMeanPOneYY")


@needs_tree
def test_w8_the_sign_statements_of_the_introduction_and_the_discussion_agree_with_the_macros():
    net_negative = [q for q in QUARTER if _number(f"net{q}") < 0]
    intro = _flat(_live_tex(_read("sections/01_introduction.tex")))
    assert quarters_after(intro, r"smaller than the profit they give up in ") == net_negative
    assert all(lo < 0 < hi for lo, hi in (_interval(f"net{q}CI") for q in QUARTER))                      # "these intervals include zero too": every NET interval contains zero
    assert _number("failRelPostOpen") > _number("failRelYY") and _number("failDispPostOpen") > _number("failDispYY")     # "two other feature sets fail at more nodes in that half-year"
    assert _number("failLevYY") > _number("failLevPostOpen") and _number("failCalYY") == _number("failCalPostOpen") == 0     # the level block improves; the calendar block never fails
    assert _number("driftWinOutMeanPFourYY") < _number("driftWinOutMeanPOneYY") and _number("driftOutPFourYY") < _number("driftOutPOneYY")
    discussion = _flat(_live_tex(_read("sections/07_discussion.tex")))
    assert re.search(r"calendar features do not fail in either main period", discussion)
    assert all(f"\\dselcCalCount{q}{{}}" in discussion for q in QUARTER)
    assert quarters_after(discussion, r"dselcCalCountAqTwo\{\} in ", r"[^.]{0,100}every interval contains zero") == list(QUARTER)
    assert all(lo < 0 < hi for lo, hi in (_interval(f"dselcCalCount{q}CI") for q in QUARTER))            # "every interval contains zero"
    esred = re.search(r"shortfall is negative in " + QUARTER_LIST + r"\s*and positive in " + QUARTER_LIST, discussion)
    assert esred, "the discussion no longer says in which quarters the paired difference in the expected shortfall is negative and positive"
    assert _named(esred.group(1)) == [q for q in QUARTER if _number(f"desredCalCount{q}") < 0]
    assert _named(esred.group(2)) == [q for q in QUARTER if _number(f"desredCalCount{q}") > 0]
    excluding = [q for q in QUARTER if not _interval(f"desredCalCount{q}CI")[0] < 0 < _interval(f"desredCalCount{q}CI")[1]]
    assert quarters_after(discussion, r"excludes zero only in ") == excluding
    own = [q for q in QUARTER if _interval(f"esred{q}CI")[0] > 0]                                         # the alerts' own ESred: an interval above zero
    assert quarters_after(discussion, r"reduce the expected shortfall in ", r",? with an approximate interval that excludes zero") == own
    net = re.search(r"value of the alerts is negative in " + QUARTER_LIST + r"\s*and about zero in " + QUARTER_LIST, discussion)
    assert net and _named(net.group(1)) == net_negative and _named(net.group(2)) == ["AqTwo"]
    assert re.search(r"about zero in 2026-Q2,? and every interval contains zero", discussion)
    assert abs(_number("netAqTwo")) < 0.01 * max(abs(_number(f"net{q}")) for q in QUARTER)                # "about zero"
    conclusion = _flat(_live_tex(_read("sections/09_conclusion.tex")))
    assert re.search(r"eight-week windows[^.;]{0,120}fewer[^.;]{0,60}than the fixed cutoff", conclusion)
    assert 0 < _number("driftWinOutMeanPFourPostOpen") < _number("driftWinOutMeanPOnePostOpen") and 0 < _number("driftWinOutMeanPFourYY") < _number("driftWinOutMeanPOneYY")


@needs_tree
def test_w8_the_sign_statements_and_the_typed_values_of_the_cover_letters_agree_with_the_macros():
    net_negative = [q for q in QUARTER if _number(f"net{q}") < 0]
    for name in LETTERS:
        letter = _flat(_live_md(_read(name)))
        assert quarters_after(letter, r"net value is negative in ") == net_negative, name
        nodes = re.search(r"(\d+) pricing nodes of four U\.S\. operators", letter)                           # author note 3: typed, and equal to the macros nNodes and ssrnId
        assert nodes and int(nodes.group(1)) == _number("nNodes"), name
        ssrn = re.search(r"SSRN preprint (\d+)", letter)
        assert ssrn and ssrn.group(1) == _macro("ssrnId"), name
        assert re.search(r"rolling cutoff leaves no node outside the band over a whole year or half-year", letter), name
        assert re.search(r"fewer than a fixed cutoff within eight-week windows", letter), name
    assert _number("driftOutPFourYY") == _number("driftOutPFourPostOpen") == 0
    assert 0 < _number("driftWinOutMeanPFourYY") < _number("driftWinOutMeanPOneYY") and 0 < _number("driftWinOutMeanPFourPostOpen") < _number("driftWinOutMeanPOnePostOpen")


def _sign_word(value: float) -> str:
    return "positive" if value > 0 else "negative"


_SIGNED_QUARTERS = re.compile(r"(positive|negative) in " + QUARTER_LIST)


def signed_quarters(clause: str) -> dict[str, str]:
    """{suffix: sign} for every "positive in LIST" and "negative in LIST" of ``clause`` ("was negative in 2025-Q3 and 2025-Q4 and positive in 2026-Q1 and 2026-Q2")."""
    out = {}
    for m in _SIGNED_QUARTERS.finditer(clause):
        out.update({q: m.group(1) for q in _named(m.group(2))})
    return out


def test_the_signed_quarters_are_read_from_a_clause():
    assert signed_quarters("was negative in 2025-Q3 and 2025-Q4 and positive in 2026-Q1 and 2026-Q2") == {
        "AqThree": "negative", "AqFour": "negative", "AqOne": "positive", "AqTwo": "positive"}
    assert signed_quarters("was negative in 2025-Q3, 2025-Q4 and 2026-Q1 and positive in 2026-Q2") == {
        "AqThree": "negative", "AqFour": "negative", "AqOne": "negative", "AqTwo": "positive"}
    assert signed_quarters("nothing signed here") == {}


@needs_tree
def test_w8_the_sign_statements_about_test_a_agree_with_the_macros_and_with_the_development_quarters():
    """The state after the run says, by sign, how the paired differences of Test A compare with the development quarters: 6.3 names the sign of each of the four quarters and the sign
    of Test A for the excess loss capture and for the expected shortfall ("was negative in 2025-Q3 and 2025-Q4 and positive in 2026-Q1 and 2026-Q2, and is positive in Test A; that in
    the expected shortfall was ..., and is negative in Test A. Both intervals of Test A contain zero"); the other places say that the intervals include zero and that the excess loss
    capture of the alerts is positive. The signs are the signs of the macros, and an interval that is said to contain zero contains it. Skipped while the macros hold no estimate of
    Test A."""
    if _macro("dselcCalCountTestA").startswith("\\PENDING"):
        pytest.skip("Test A has not run: generated/macros.tex holds no estimate of it")
    selc, esred = _number("dselcCalCountTestA"), _number("desredCalCountTestA")
    for name in ("dselcCalCountTestACI", "desredCalCountTestACI"):
        lo, hi = _interval(name)
        assert lo < 0 < hi, f"{name} no longer contains zero: the sentences that say so are wrong"
    forward = _flat(_subsection(resolved("sections/07_discussion.tex", "ran"), "sec:disc-forward"))
    m = re.search(r"paired difference in the excess loss capture (was .*?),? and is (positive|negative) in Test A; that in the expected shortfall (was .*?),? and is (positive|negative) "
                  r"in Test A\.\s*Both intervals of Test A contain zero", forward)
    assert m, "6.3 no longer names the signs of the paired differences by quarter and in Test A, and says that both intervals contain zero"
    assert signed_quarters(m.group(1)) == {q: _sign_word(_number(f"dselcCalCount{q}")) for q in QUARTER}
    assert m.group(2) == _sign_word(selc)
    assert signed_quarters(m.group(3)) == {q: _sign_word(_number(f"desredCalCount{q}")) for q in QUARTER}
    assert m.group(4) == _sign_word(esred)
    discussion = _flat(resolved("sections/07_discussion.tex", "ran"))
    assert re.search(r"Both intervals contain zero, so Test A, like the development quarters, does not demonstrate an improvement over the calendar-only rule", discussion)   # 6.1
    intro = _flat(resolved("sections/01_introduction.tex", "ran"))
    assert re.search(r"its paired differences with the calendar-only rule have intervals that include zero, and the excess loss capture of the alerts, which measures timing within "
                     r"calendar cells, is " + _sign_word(_number("selcTestA")), intro)
    conclusion = _flat(resolved("sections/09_conclusion.tex", "ran"))
    assert re.search(r"both intervals include zero, so Test A does not demonstrate an improvement either", conclusion)
    abstract = _flat(resolved("sections/00_abstract.tex", "ran"))
    assert re.search(r"both intervals include zero", abstract)


# ------------------------------------------------------------------------------------------------------ the abstract and the introduction
def _macro_or_none(name: str) -> str | None:
    m = re.search(r"^\\newcommand\{\\" + name + r"\}\{(.*)\}\s*$", _read("generated/macros.tex"), flags=re.M)
    return m.group(1) if m else None


def printed_words(text: str) -> int | None:
    """The words of a LaTeX text as a reader (or a word counter of the journal's submission system) counts them: every macro call ``\\name{}`` is replaced by the value that
    generated/macros.tex gives it (so that "[0.1%, 7.6%]" is two words and "2024 to 2025" is three), the rest of the markup goes, and the tokens that hold a letter or a digit are
    counted. None when a macro that the text uses is still a pending sentinel (Test A has not run): its printed length is not known."""
    pending = []

    def value(m):
        v = _macro_or_none(m.group(1))
        if v is None:
            return m.group(0)
        if "\\PENDING" in v:
            pending.append(m.group(1))
        return v

    out = re.sub(r"\\([A-Za-z]+)\{\}", value, text)
    if pending:
        return None
    out = out.replace("\\textminus{}", "-").replace("\\$", "$").replace("\\%", "%").replace("{,}", ",").replace("--", "-").replace("~", " ")
    out = re.sub(r"\\[A-Za-z]+\*?", " ", out)
    return len([t for t in out.split() if re.search(r"[A-Za-z0-9]", t)])


def test_the_printed_word_counter_counts_what_a_reader_counts():
    assert printed_words("In 2024 to 2025, fail at 13 nodes.") == 8
    assert printed_words("The band is [0.5, 2] and the ratio 9.3\\% is high.") == 11           # an interval is two words, a percentage one, a macro-free text as is
    assert printed_words("An \\emph{alert} fires --- often.") == 4                           # a dash alone is not a word


ABSTRACT_STATES = ("Test A has not run", "Test A has run", "Test A did not run")


@needs_tree
def test_the_abstract_has_150_to_250_words_in_every_state_by_the_count_of_the_lint():
    from es.paper_lint import abstract_variants, _braced_arg, _strip_comment
    body = " ".join(_strip_comment(line) for line in _read("sections/00_abstract.tex").splitlines())
    variants = abstract_variants(_braced_arg(body, "\\abstract"))
    assert variants is not None and {v.label for v in variants} == set(ABSTRACT_STATES)
    for v in variants:
        assert 150 <= v.words <= 250, (v.label, v.words)


def _abstract_printed_cases():
    # (The ran abstract had 255 printed words at 06:18, five over the limit; the controller cut it at 06:23, now 245: the xfail of case C6 is gone.)
    return list(ABSTRACT_STATES)


@needs_tree
@pytest.mark.parametrize("label", _abstract_printed_cases())
def test_the_abstract_has_at_most_250_printed_words_in_every_state(label):
    from es.paper_lint import abstract_variants, _braced_arg, _strip_comment
    body = " ".join(_strip_comment(line) for line in _read("sections/00_abstract.tex").splitlines())
    variant = next(v for v in abstract_variants(_braced_arg(body, "\\abstract")) if v.label == label)
    n = printed_words(variant.text)
    if n is None:
        pytest.skip(f"{label}: a macro of the abstract is still pending, its printed length is not known")
    assert n <= 250, f"the abstract of the state {label!r} has {n} printed words (the lint counts {variant.words})"


@needs_tree
def test_the_introduction_defines_dart_and_spike_alert_at_their_first_occurrence():
    intro = _live_tex(_read("sections/01_introduction.tex"))
    first = re.search(r"\bDART\b", intro)
    assert first and re.search(r"which we call\s*$", intro[:first.start()]), "the first DART is not 'which we call DART'"
    assert intro[first.end():first.end() + 1] in ",.", intro[first.start() - 40:first.end() + 40]
    alert = re.search(r"spike alerts?\b", intro, re.I)
    assert alert and intro[:alert.start()].rstrip().endswith("\\emph{"), "the first spike alert is not set in \\emph{...}"
    assert re.match(r"spike alert\}\s+warns that an hour looks like", intro[alert.start():]), intro[alert.start():alert.start() + 80]
    spike = re.search(r"\bspikes?\b", intro, re.I)                                                          # and the spike itself is defined before the alert
    assert spike.start() <= alert.start() and intro[:spike.start()].rstrip().endswith("\\emph{"), intro[spike.start() - 20:spike.start() + 40]


def test_the_definition_checks_find_a_use_before_the_definition():
    intro = "The DART is large. The gap, which we call DART, is small. A \\emph{spike alert} warns that an hour looks like one."
    first = re.search(r"\bDART\b", intro)
    assert not re.search(r"which we call\s*$", intro[:first.start()])
    intro = "Alerts matter. A spike alert is useful. A \\emph{spike alert} warns that an hour looks like one."
    alert = re.search(r"spike alerts?\b", intro, re.I)
    assert not intro[:alert.start()].rstrip().endswith("\\emph{")


# ------------------------------------------------------------------------------------------------------ the numbers of the sections that the letters type
def _headings_from_sources() -> dict[str, str]:
    """{label: number} of the headings of the sections that main.tex inputs, in the order of its \\input lines (starred headings are not numbered)."""
    out, counter = {}, [0, 0, 0]
    for name in re.findall(r"^\\input\{(sections/[^}]+)\}", _live_tex(_read("main.tex")), flags=re.M):
        text = _live_tex(_read(name + ".tex"))
        for m in re.finditer(r"\\(section|subsection|subsubsection)(\*?)\{[^}]*\}\s*(?:\\label\{([^}]*)\})?", text):
            if m.group(2):
                continue
            level = ("section", "subsection", "subsubsection").index(m.group(1))
            counter[level] += 1
            for deeper in range(level + 1, 3):
                counter[deeper] = 0
            if m.group(3):
                out[m.group(3)] = ".".join(str(n) for n in counter[:level + 1])
    return out


def _headings_from_aux() -> dict[str, str] | None:
    """{label: number} of main.aux, or None when there is no main.aux or it is older than the sections (a build is due)."""
    aux = PAPER / "main.aux"
    if not aux.exists() or aux.stat().st_mtime < max(p.stat().st_mtime for p in (PAPER / "sections").glob("*.tex")):
        return None
    return {m.group(1): m.group(2) for m in re.finditer(r"\\newlabel\{(sec:[^}]*)\}\{\{([0-9.]+)\}", aux.read_text(encoding="utf-8"))}


#: Every section number that the two cover letters (body and AUTHOR NOTES), the author's checklist and the corrections file (which is uploaded) type, and the label of the heading
#: that it means. A typed number that is not here fails the test below: it has to be added with its label, so that someone checks what it points at.
TYPED_SECTIONS = {"1": "sec:intro", "3.1": "sec:data-panel", "3.4": "sec:data-analyses", "3.5": "sec:data-impl", "4.1": "sec:audit-control", "4.2": "sec:audit-blocks",
                  "4.3": "sec:audit-ranking", "4.6": "sec:audit-holdout", "6.3": "sec:disc-forward", "6.4.2": "sec:lim-holdout"}
#: The three numbers that the cover letters type, with a phrase that must stand under the heading they mean: the use of AI assistants, the CAISO series, the forward evidence.
LETTER_SECTIONS = {"3.5": ("sec:data-impl", "sections/03_data_design.tex", "Large-language-model assistants"),
                   "3.1": ("sec:data-panel", "sections/03_data_design.tex", "CAISO"),
                   "6.3": ("sec:disc-forward", "sections/07_discussion.tex", "design for two forward quarters")}
OTHER_FILES_WITH_TYPED_SECTIONS = ("AUTHOR_CHECKLIST.md", "CORRECTIONS_TO_PREPRINT.md")


def typed_section_numbers(text: str) -> list[str]:
    """Every number that follows 'Section' or 'Sections' in ``text`` (also 'Sections 6.1.5 and 8.2' and 'Sections 4, 5 and 6')."""
    out = []
    for m in re.finditer(r"\bSections?\s+((?:\d+(?:\.\d+)*)(?:\s*(?:,|and)\s*\d+(?:\.\d+)*)*)", text):
        out += re.findall(r"\d+(?:\.\d+)*", m.group(1))
    return out


def test_the_section_number_finder_reads_lists():
    assert typed_section_numbers("(Sections 6.1.5 and 8.2) and Section 3.5, Sections 4, 5 and 6.") == ["6.1.5", "8.2", "3.5", "4", "5", "6"]
    assert typed_section_numbers("no number here, sections of the paper") == []


def _check_number_against_headings(number: str, label: str) -> None:
    from_sources, from_aux = _headings_from_sources(), _headings_from_aux()
    assert from_sources[label] == number, f"{label} is Section {from_sources[label]} in the sources, and a file types Section {number}"
    if from_aux is not None:
        assert from_aux[label] == number, f"main.aux numbers {label} {from_aux[label]}, and a file types Section {number}"


@needs_tree
def test_the_section_numbers_typed_in_the_letters_are_those_of_the_headings_they_mean():
    for number, (label, rel, phrase) in LETTER_SECTIONS.items():
        _check_number_against_headings(number, label)
        assert phrase in _subsection(_live_tex(_read(rel)), label), f"{label} no longer holds {phrase!r}: the number that the letters type may point at the wrong place"
    typed = {}
    for name in LETTERS:                                                                                    # the body and the AUTHOR NOTES alike
        for number in typed_section_numbers(_read(name)):
            typed.setdefault(number, []).append(name)
    assert set(typed) == set(LETTER_SECTIONS), f"a letter types a section number that this test does not know (add it with its label): {typed}"
    assert "Section 3.5 of the manuscript" in _read("cover_letter.md") and re.search(r"\(Section 6\.3 and Online Resource 1\)", _read("cover_letter_TestA_ran.md"))


@needs_tree
def test_every_section_number_typed_in_the_checklist_and_in_the_corrections_file_names_the_heading_it_means():
    """The author's checklist and the corrections file (uploaded with the manuscript, its last column names the sections that make each correction) type section numbers of the
    paper. Each is the number of the heading in TYPED_SECTIONS, in the sources and in main.aux; a heading that moves breaks the file here, not in front of a referee."""
    found = set()
    for name in OTHER_FILES_WITH_TYPED_SECTIONS:
        for number in typed_section_numbers(_live_md(_read(name))):
            assert number in TYPED_SECTIONS, f"{name} types Section {number}, which this test does not know (add it to TYPED_SECTIONS with its label)"
            _check_number_against_headings(number, TYPED_SECTIONS[number])
            found.add(number)
    assert {"3.5", "6.3", "4.6", "6.4.2"} <= found, found                                                # the scan reads both files


@needs_tree
def test_the_headings_of_the_sources_agree_with_main_aux_when_it_is_fresh():
    from_aux = _headings_from_aux()
    if from_aux is None:
        pytest.skip("main.aux is older than the sections or missing: rebuild the PDF and the comparison runs")
    from_sources = _headings_from_sources()
    assert set(from_sources) <= set(from_aux), sorted(set(from_sources) - set(from_aux))
    assert {k: v for k, v in from_aux.items() if k in from_sources} == from_sources


# ------------------------------------------------------------------------------------------------------ the three states of Test A
_RESIDUE = re.compile(r"\\(?:if[A-Za-z]*|else|fi)(?![A-Za-z])")
#: (place, file, label of the subsection or None, the markers of each state). A marker is a regular expression for a key phrase of the flat text of the place in that state, not a
#: sentence (the controller rewords around the key phrases); None says that the state has no Test A text at all (a place with one sentence for the run state only: the discussion
#: 6.1 and 6.2 and, in the other two states, the text of the place without the sentence).
STATE_PLACES = (
    ("abstract", "sections/00_abstract.tex", None,
     {"absent": r"not been run", "ran": r"\bIn Test A\b", "did_not_run": r"\bdid not run\b"}),
    ("introduction, last paragraph", "sections/01_introduction.tex", None,
     {"absent": r"not been run", "ran": r"\bhas been run and is reported as an interim result\b", "did_not_run": r"\bdid not run\b"}),
    ("discussion 6.1, the Test A sentence", "sections/07_discussion.tex", "sec:disc-audit",
     {"absent": None, "ran": r"\bIn Test A, the first forward quarter\b", "did_not_run": None}),
    ("discussion 6.2, the Test A sentence", "sections/07_discussion.tex", "sec:disc-governance",
     {"absent": None, "ran": r"\bIn Test A, an interim estimate\b", "did_not_run": None}),
    ("discussion 6.3", "sections/07_discussion.tex", "sec:disc-forward",
     {"absent": r"not been run", "ran": r"\bwas run once\b", "did_not_run": r"\bdid not run\b"}),
    ("conclusion", "sections/09_conclusion.tex", None,
     {"absent": r"not been run", "ran": r"\bhas been run\b", "did_not_run": r"\bdid not run\b"}),
    ("back matter, prior dissemination", "sections/90_backmatter.tex", None,
     {"absent": r"not been run", "ran": r"\bhas been run once\b", "did_not_run": r"\bdid not run\b"}),
    ("Online Resource, protocol record paragraph", "esm/forward_tests.tex", "sec:sealed-record",
     {"absent": r"\bto be run\b", "ran": r"Test A was run\b", "did_not_run": r"Test A was run\b"}),
)


def test_the_residue_finder_sees_a_switch_that_was_left_behind():
    for bad in ("text \\ifTestA more", "text \\fi more", "a \\else b", "\\ifTestAdidnotrun x"):
        assert _RESIDUE.search(bad), bad
    assert not _RESIDUE.search("a \\emph{fine} and \\selcTestA{} and the field")


@needs_tree
@pytest.mark.parametrize("place,rel,label,markers", STATE_PLACES, ids=[p[0] for p in STATE_PLACES])
def test_the_sentences_that_carry_the_three_states_of_test_a_exist_and_compile_without_a_switch(place, rel, label, markers):
    texts = {}
    for state in STATES:
        text = resolved(rel, state)
        text = _subsection(text, label) if label else text
        assert not _RESIDUE.search(text), (place, state, _RESIDUE.search(text).group(0))                  # no \if, \else or \fi left in the text of the state
        texts[state] = _flat(text)
        assert texts[state], (place, state)
        marker = markers[state]
        if marker is None:
            assert "In Test A" not in texts[state] and not re.search(r"\w+TestA\b", texts[state]), (place, state)
        else:
            assert re.search(marker, texts[state]), f"{place}, state {state}: no text matching {marker!r}"
    assert not re.search(r"\w+TestA(?:CI)?\b", texts["absent"]), f"{place}: a macro of Test A in the state before the run"       # nothing of Test A before the run
    assert texts["absent"] != texts["ran"], place
    same = markers["absent"] is None                                                                          # a place with one sentence for the run state only
    assert (texts["absent"] == texts["did_not_run"]) == same, place
    assert (texts["ran"] == texts["did_not_run"]) == (markers["ran"] == markers["did_not_run"]), place


_NOT_RUN = re.compile(r"\b(?:has|have|had|is|was) not (?:yet )?(?:been )?run\b|\bdid not run\b|\bis to be run\b|\bwill be run\b|\bis pending\b")


def says_test_a_has_not_run(sentence: str) -> bool:
    """Whether a clause of ``sentence`` says that Test A has not been run, did not run, is to be run or is pending: the test named last before the clause is Test A. ("... after Test B,
    which has not been run" is about Test B.)"""
    for m in _NOT_RUN.finditer(sentence):
        named = re.findall(r"Test [AB]\b", sentence[:m.start()])
        if named and named[-1] == "Test A":
            return True
    return False


def test_the_not_run_finder_reads_which_test_a_clause_is_about():
    assert says_test_a_has_not_run("Test A had not been run when this version was prepared.")
    assert says_test_a_has_not_run("Test A, the first quarter, did not run.") and says_test_a_has_not_run("Test A is pending.")
    assert not says_test_a_has_not_run("Test A has no confirmatory test of its own, and that test is decided only on the pooled window after Test B, which has not been run.")
    assert not says_test_a_has_not_run("In Test A the interim estimate is large; Test B is pending.")


#: The places whose text in the state after the run prints the estimates of Test A (as macros, so that the numbers are generated); the back matter and the Online Resource paragraph of the
#: protocol record say that Test A was run and print no estimate.
PLACES_WITH_TEST_A_ESTIMATES = ("abstract", "discussion 6.1, the Test A sentence", "discussion 6.2, the Test A sentence", "discussion 6.3", "conclusion")


@needs_tree
def test_the_state_after_the_run_prints_the_estimates_of_test_a_as_macros_and_never_says_that_test_a_has_not_been_run():
    """The manuscript as it is built now (generated/macros.tex says \\TestAtrue): the paired differences and the other estimates of Test A are macros, with their intervals where they
    are given, and no sentence of any place says that Test A has not been run, did not run or is to be run; Test B, which has not been run, is another matter."""
    for place, rel, label, markers in STATE_PLACES:
        text = _flat(resolved(rel, "ran"))
        text = _flat(_subsection(resolved(rel, "ran"), label)) if label else text
        if place in PLACES_WITH_TEST_A_ESTIMATES:
            assert re.search(r"\\\w+TestA\{\}", text), f"{place}: no estimate of Test A (a macro) in the state after the run"
        for sentence in _sentences(_printed(text)):
            assert not says_test_a_has_not_run(sentence), (place, sentence)


@needs_tree
def test_every_section_and_every_esm_section_resolves_in_all_three_states_without_a_residue():
    files = [f"sections/{p.name}" for p in sorted((PAPER / "sections").glob("*.tex"))]
    files += [f"esm/{name}" for name in ("forward_tests.tex", "designation_rule_study.tex", "data_design_details.tex", "audit_economic_details.tex")]
    for rel in files:
        for state in STATES:
            text = resolved(rel, state)
            assert not _RESIDUE.search(text), (rel, state, _RESIDUE.search(text).group(0))


@needs_tree
def test_the_pdf_carries_the_state_that_macros_tex_sets():
    """State-aware: the abstract of main.pdf has the key phrase of the state that generated/macros.tex set when it was built (\\TestAtrue / \\TestAdidnotruntrue), and not the key
    phrase of the other two. (The abstract only: "has not been run" and "did not run" stand elsewhere in the text of the state in which Test A has run.)"""
    from es.paper_lint import switch_state
    state = switch_state(_read("generated/macros.tex"))
    name = "did_not_run" if state.get("TestAdidnotrun") else "ran" if state.get("TestA") else "absent"
    text = pdf_text()
    assert "Abstract" in text and "Keywords:" in text
    abstract = text.split("Abstract", 1)[1].split("Keywords:", 1)[0]
    markers = {k: STATE_PLACES[0][3][k] for k in STATES}
    assert re.search(markers[name], abstract), f"macros.tex says the state is {name!r}, and the abstract of main.pdf has no text matching {markers[name]!r}"
    for other in set(STATES) - {name}:
        assert not re.search(markers[other], abstract), f"macros.tex says {name!r}, and the abstract of main.pdf has the key phrase of {other!r}"
