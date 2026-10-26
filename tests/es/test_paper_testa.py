"""Task 9, pre-drafted (the Test A text, written before Test A has run): what that text needs from the generators, and the rules that bind it.

* New macros of the sealed tests (stage sealedA/sealedB): the number of departures from the registered code manifest, whether H-E's
  p-value is labelled approximate, the start and the registration of the run to the minute (UTC). A test that did not run still
  prints its header (NOT_RUN_KEYS), and these keys are part of it.
* \\pullRecordTestA (Ruling 121): the sentence about the pull's own log, written by the controller into paper/author_inputs.json
  (key pull_record_test_a). It stays pending, with a visible sentinel, and fails the strict build of stage sealedA until it is filled.
* The abstract must have at most 250 words as each reader sees it: es.paper_lint counts every variant of the \\ifTestA switches
  (the old count added the words of both branches).
* The manuscript text itself (the real paper/ folder): the Test A variants of the abstract, and the wording rules of the Test A branch of
  the forward-test section (no p-value and no inferential reading of H-B1 and H-B2, no comparison with Test B). Since plan 2C task T2a that
  section is esm/forward_tests.tex, a section of Online Resource 1; it was sections/06_sealed.tex, Section 6.2 of the paper."""
import json
import re
from pathlib import Path

import pytest

PAPER = Path(__file__).resolve().parents[3] / "paper"
needs_tree = pytest.mark.skipif(not (PAPER / "sections").exists(), reason="the manuscript folder is not next to dart_dev")
SENTENCE = ("The pull accepted no chunk as incomplete and excused no day, as PULL_LOG and the entry of PULLS.json show.")


# ------------------------------------------------------------------------------------------------------ formats
def test_utc_minute_format_prints_the_stamp_in_utc_to_the_minute():
    from es.paper_tex import FORMATS
    f = FORMATS["utcmin"]
    assert f("2026-10-25T15:30:00Z") == "2026-10-25 15:30 UTC"
    assert f("2026-10-25T04:35:10.025739+00:00") == "2026-10-25 04:35 UTC"
    assert f("2026-10-25T11:30:00-04:00") == "2026-10-25 15:30 UTC"            # another offset is converted, never printed as it is
    for bad in ("2026-10-25", "2026-10-25T15:30:00", "yesterday", "", None, 7):   # a date, a naive time, text: never a silent guess
        with pytest.raises(ValueError):
            f(bad)


def test_approximate_label_format_says_what_the_flag_says():
    from es.paper_tex import FORMATS
    f = FORMATS["approxlabel"]
    assert f(True) == "labelled approximate" and f("true") == "labelled approximate"
    assert f(False) == "not labelled approximate" and f("false") == "not labelled approximate"
    with pytest.raises(ValueError):
        f("maybe")                                                              # not a boolean: an error, never a silent "no"


# ------------------------------------------------------------------------------------------------------ the registry
NEW_SEALED = {"departures": ("int", "departures"), "heApprox": ("approxlabel", "H-E/approximate"),
              "startedUtc": ("utcmin", "started_utc"), "registeredUtc": ("utcmin", "registered_utc")}


def test_the_new_sealed_macros_are_registered_for_both_tests_with_their_formats_and_sources():
    from es.paper_registry import entries_by_name
    from es.paper_tex import FORMATS
    by = entries_by_name()
    for test, stage in (("TestA", "sealedA"), ("TestB", "sealedB")):
        for word, (fmt, query) in NEW_SEALED.items():
            e = by[f"{word}{test}"]
            assert (e.fmt, e.stage, e.query) == (fmt, stage, query), e.name
            assert e.source == f"sealed/{test[-1]}/results.json" and fmt in FORMATS
    assert by["departuresTestA"].agg == "count"                                   # the number of entries of results.json["departures"]


def test_the_pull_record_is_a_pending_text_of_the_author_file_in_stage_sealed_a():
    from es.paper_registry import entries_by_name
    e = entries_by_name()["pullRecordTestA"]
    assert (e.source, e.fmt, e.stage) == ("author:pull_record_test_a", "text", "sealedA")
    assert "pullRecordTestB" not in entries_by_name()                              # Test B has its own task (Task 12)


def _results(ran=True, departures=(), approximate=False):
    res = {"test": "A", "started_utc": "2026-10-25T21:10:05.123456+00:00", "protocol_sha256": "ab" * 32,
           "registration": "10.17605/OSF.IO/ABCDE", "registered_utc": "2026-10-25T15:30:00Z", "n_rep": 2000,
           "departures": list(departures), "rows_read": 250000, "rows_kept": 240000, "eta": 0.05, "family": ["H-E"],
           "coverage": {"ran": ran, "usable_nodes": 33 if ran else 20}, "errors": {}, "status": "completed"}
    if ran:
        res["H-E"] = {"status": "interim (single window; decided after Test B)", "t_obs": 0.86, "p": 0.33, "n_null": 91,
                      "approximate": approximate, "confirmatory": True}
    else:
        res["status"] = "not run (coverage rule)"
    return res


def _src(tmp_path, results=None, author=None):
    from es.paper_macros import Sources
    out, paper = tmp_path / "out", tmp_path / "paper"
    paper.mkdir(parents=True, exist_ok=True)
    (paper / "author_inputs.json").write_text(json.dumps({"pull_record_test_a": ""} if author is None else author))
    if results is not None:
        d = out / "sealed" / "A"
        d.mkdir(parents=True, exist_ok=True)
        (d / "results.json").write_text(json.dumps(results))
    return Sources(out, paper, tmp_path / "docs_sealed")


def _resolve(name, src):
    from es.paper_macros import resolve
    from es.paper_registry import entries_by_name
    return resolve(entries_by_name()[name], src)


def test_the_new_macros_read_the_run_record(tmp_path):
    src = _src(tmp_path, _results())
    values = {n: _resolve(n, src) for n in ("departuresTestA", "heApproxTestA", "startedUtcTestA", "registeredUtcTestA")}
    assert {n: (r["status"], r["value"]) for n, r in values.items()} == {
        "departuresTestA": ("ok", "0"), "heApproxTestA": ("ok", "not labelled approximate"),
        "startedUtcTestA": ("ok", "2026-10-25 21:10 UTC"), "registeredUtcTestA": ("ok", "2026-10-25 15:30 UTC")}
    departed = _src(tmp_path / "d", _results(departures=[{"file": "es/a.py", "registered": "x", "current": "y"},
                                                          {"file": "es/b.py", "registered": "x", "current": "z"}], approximate=True))
    assert _resolve("departuresTestA", departed)["value"] == "2"                   # one per file of the code that differs
    assert _resolve("heApproxTestA", departed)["value"] == "labelled approximate"


def test_a_test_that_did_not_run_still_prints_its_departures_and_its_registration(tmp_path):
    """run_sealed writes departures and registered_utc in the header, before the coverage rule: they are part of NOT_RUN_KEYS."""
    from es.paper_macros import NOT_RUN_KEYS, PENDING
    assert {"departures", "registered_utc"} <= set(NOT_RUN_KEYS)
    src = _src(tmp_path, _results(ran=False))
    assert _resolve("departuresTestA", src)["value"] == "0"
    assert _resolve("registeredUtcTestA", src)["value"] == "2026-10-25 15:30 UTC"
    for name in ("heApproxTestA", "hePTestA"):                                   # the analysis results are not there
        r = _resolve(name, src)
        assert r["status"] == "pending" and r["value"] == PENDING["sealedA"] and r["detail"] == "test did not run"


# ------------------------------------------------------------------------------------------------ the pull record
def test_the_pull_record_is_pending_until_the_controller_fills_it(tmp_path):
    from es.paper_macros import PENDING
    for author in ({"pull_record_test_a": ""}, {"pull_record_test_a": "   "}, {"pull_record_test_a": None}):
        r = _resolve("pullRecordTestA", _src(tmp_path, author=author))
        assert (r["status"], r["value"]) == ("pending", PENDING["sealedA"]), author     # a visible sentinel, never an empty macro
    assert _resolve("pullRecordTestA", _src(tmp_path, results=_results(ran=False), author={"pull_record_test_a": ""}))["status"] == "pending"
    r = _resolve("pullRecordTestA", _src(tmp_path, author={"pull_record_test_a": SENTENCE}))
    assert (r["status"], r["value"]) == ("ok", SENTENCE.replace("_", "\\_"))            # plain text, escaped for LaTeX


def test_the_pull_record_must_be_a_finished_plain_sentence(tmp_path):
    for bad, why in (("The pull accepted no chunk as incomplete", "full stop"), ("TODO <fill in from PULL_LOG>.", "placeholder"),
                     ("The pull excused 2026-08-14 − one day.", "ASCII"), ("x" * 800 + ".", "long")):
        with pytest.raises(LookupError, match="pullRecordTestA"):
            _resolve("pullRecordTestA", _src(tmp_path, author={"pull_record_test_a": bad}))
    r = _resolve("pullRecordTestA", _src(tmp_path, author={"pull_record_test_a": "The pull accepted one chunk\nas incomplete.\n"}))
    assert r["value"] == "The pull accepted one chunk as incomplete."                 # one line: a blank line would end the paragraph
    with pytest.raises(LookupError, match="pull_record_test_a"):                       # an author file without the key is a registry bug
        _resolve("pullRecordTestA", _src(tmp_path, author={"title": "T"}))


def test_the_pull_record_fails_the_strict_build_of_stage_sealed_a_only(tmp_path):
    from es.paper_macros import main
    src = _src(tmp_path)
    args = ["--out-dir", str(src.out_dir), "--paper-dir", str(src.paper_dir), "--only", "pullRecordTestA", "--strict"]
    assert main(args + ["--stages", "sealedA"]) == 1                                   # the sentinel stays until the controller fills it
    assert main(args + ["--stages", "const,dev,author"]) == 0
    assert "\\newcommand{\\pullRecordTestA}{\\PENDINGA}" in (src.paper_dir / "generated" / "macros.tex").read_text()   # the sentinel
    (src.paper_dir / "author_inputs.json").write_text(json.dumps({"pull_record_test_a": SENTENCE}))
    assert main(args + ["--stages", "sealedA"]) == 0
    assert "\\newcommand{\\pullRecordTestA}{" + SENTENCE.replace("_", "\\_") + "}" in (src.paper_dir / "generated" / "macros.tex").read_text()


@needs_tree
def test_the_real_author_file_has_the_pull_record_key_and_says_where_to_fill_it():
    data = json.loads((PAPER / "author_inputs.json").read_text())
    assert isinstance(data.get("pull_record_test_a"), str)                              # empty until the controller fills it after the pull
    assert "pull_record_test_a" in data["_comment"] and "PULL_LOG" in data["_comment"]


# ------------------------------------------------------------------------------------------------ the abstract count
def _paper(tmp_path: Path) -> Path:
    p = tmp_path / "paper"
    for d in ("sections", "appendices", "generated/tables", "generated/figures"):
        (p / d).mkdir(parents=True)
    (p / "author_inputs.json").write_text('{"title": "Operating-point drift", "short_title": "Drift"}')
    (p / "references.bib").write_text("@article{k, title={x}}\n")
    (p / "generated" / "macros.tex").write_text("\\newcommand{\\PENDINGA}{x}\n")
    return p


def _abstract(p: Path, body: str) -> None:
    (p / "sections" / "00_abstract.tex").write_text("\\abstract{" + body + "}\n\\keywords{a, b, c, d}\n")


def _words(n: int) -> str:
    return " ".join(["word"] * n)


def test_the_abstract_is_counted_as_each_reader_sees_it():
    from es.paper_lint import abstract_variants
    one = abstract_variants("A b c. \\ifTestA yes yes yes \\else no \\fi d \\macro e.")
    counts = {v.label: v.words for v in one}
    assert counts == {"Test A has not run": 3 + 1 + 1 + 1 + 1, "Test A has run": 3 + 3 + 1 + 1 + 1}   # a b c, [no | yes yes yes], d, \macro, e
    plain = abstract_variants("One two three.")
    assert [(v.label, v.words) for v in plain] == [("", 3)]                            # no switch: one variant, no label


def test_the_abstract_variants_follow_the_switches_the_macros_set():
    from es.paper_lint import abstract_variants
    text = ("pre \\ifTestA \\ifTestAdidnotrun ran not \\else ran yes yes \\fi \\else pending \\fi post")
    got = {v.label: v.text.split() for v in abstract_variants(text)}
    assert got == {"Test A has not run": ["pre", "pending", "post"], "Test A has run": ["pre", "ran", "yes", "yes", "post"],
                   "Test A did not run": ["pre", "ran", "not", "post"]}                # a test that did not run has run: no fourth variant
    both = abstract_variants("\\ifTestA a \\fi \\ifTestB b \\else c \\fi")
    assert {v.label for v in both} == {"Test A has not run, Test B has not run", "Test A has not run, Test B has run",
                                       "Test A has run, Test B has not run", "Test A has run, Test B has run"}


def test_unbalanced_or_foreign_conditionals_cannot_be_counted():
    from es.paper_lint import abstract_variants
    for bad in ("\\ifTestA a", "a \\fi", "\\ifTestA a \\else b \\else c \\fi", "\\ifTestA a \\fi \\else", "\\ifdefined\\x a \\fi"):
        assert abstract_variants(bad) is None, bad


def test_strict_lint_checks_every_variant_of_the_abstract(tmp_path):
    from es.paper_lint import lint_all
    p = _paper(tmp_path)
    _abstract(p, _words(120) + " \\ifTestA " + _words(100) + " \\else " + _words(120) + " \\fi")         # 220 or 240 words, 340 by the old count
    assert [x for x in lint_all(p, strict=True) if x.rule == "abstract"] == []
    _abstract(p, _words(120) + " \\ifTestA " + _words(131) + " \\else " + _words(120) + " \\fi")         # 251 words when Test A has run
    msgs = [x.message for x in lint_all(p, strict=True) if x.rule == "abstract"]
    assert msgs == ["abstract has 251 words when Test A has run (150-250 required)"]
    assert [x for x in lint_all(p, strict=False) if x.rule == "abstract"] == []                          # the count is a strict rule
    _abstract(p, _words(140) + " \\ifTestA " + _words(10) + " \\fi")                                     # too short when Test A has not run
    msgs = [x.message for x in lint_all(p, strict=True) if x.rule == "abstract"]
    assert msgs == ["abstract has 140 words when Test A has not run (150-250 required)"]
    _abstract(p, _words(160) + " \\ifTestA " + _words(10))                                               # unbalanced: an error in both modes
    for strict in (True, False):
        found = [x for x in lint_all(p, strict=strict) if x.rule == "abstract"]
        assert [x.severity for x in found] == ["error"] and "\\if" in found[0].message, strict


def test_the_switches_that_macros_tex_sets_are_read_as_they_are():
    from es.paper_lint import switch_state
    assert switch_state("\\newif\\ifTestA\n\\TestAtrue\n\\newif\\ifTestAdidnotrun\n\\TestAdidnotrunfalse\n\\TestBfalse\n") == {
        "TestA": True, "TestAdidnotrun": False, "TestB": False}                        # a switch that is not set is not in the state
    assert switch_state("\\newcommand{\\TestAfalsey}{x}") == {}


def test_a_table_of_a_branch_that_the_generated_switches_skip_is_not_required(tmp_path):
    """The Test A tables sit inside \\ifTestA: while macros.tex says \\TestAfalse they are not printed, so that their absence is no finding
    (the lint warned of five tables before Test A, and strict would have refused a submission with Test A pending)."""
    from es.paper_lint import lint_all
    p = _paper(tmp_path)
    (p / "sections" / "06_sealed.tex").write_text(
        "\\ifTestA\n\\ifTestAdidnotrun\n\\gentable{tab_notrun}\n\\else\n\\gentable{tab_ran}\n\\genfigure{Fig9}{\\textwidth}\n\\fi\n\\else\n"
        "\\gentable{tab_pending}\n\\fi\n\\gentable{tab_always}\n% \\gentable{tab_in_a_comment}\n")

    def missing(switches, strict=True):
        (p / "generated" / "macros.tex").write_text("\\newcommand{\\PENDINGA}{x}\n" + switches)
        return sorted(f.message.split(" not generated")[0] for f in lint_all(p, strict=strict) if f.rule == "generated")

    pending = "\\TestAfalse\n\\TestAdidnotrunfalse\n"
    ran = "\\TestAtrue\n\\TestAdidnotrunfalse\n"
    did_not_run = "\\TestAtrue\n\\TestAdidnotruntrue\n"
    assert missing(pending) == ["table tab_always", "table tab_pending"]
    assert missing(ran) == ["figure Fig9", "table tab_always", "table tab_ran"]
    assert missing(did_not_run) == ["table tab_always", "table tab_notrun"]
    assert missing("") == ["figure Fig9", "table tab_always", "table tab_notrun", "table tab_pending", "table tab_ran"]   # no switch set: both branches live
    lax = [f for f in lint_all(p, strict=False) if f.rule == "generated" and "tab_pending" in f.message]
    assert [f.severity for f in lax] == ["warning"] and lax[0].line == 9                 # the line is the file's own


# ------------------------------------------------------------------------------------------------ the manuscript text
def _read(name):
    return (PAPER / name).read_text(encoding="utf-8")


def _live(text):
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in text.splitlines())


def _uploaded(name):
    """A markdown file as the packager puts it into the zip: without its HTML comments (the notes to the author)."""
    from es.paper_package import _HTML_COMMENT
    return _HTML_COMMENT.sub("", _read(name))


def _sentences(text):
    return [x for x in re.split(r"(?<=[.;?!])\s+", " ".join(text.split())) if x]


def _branch(text, switch="ifTestA"):
    """The text between \\ifTestA and its own \\fi in the live text of a section (nested conditionals counted), with the \\else part."""
    live = _live(text)
    m = re.search(r"\\" + switch + r"(?![A-Za-z])", live)
    assert m, f"no \\{switch} in the section"
    depth, pos = 0, m.start()
    for t in re.finditer(r"\\(if[A-Za-z]*|fi)(?![A-Za-z])", live[pos:]):
        depth += 1 if t.group(1).startswith("if") else -1
        if depth == 0:
            return live[pos + m.end() - m.start(): pos + t.start()]
    raise AssertionError("the \\ifTestA branch is not closed")


@needs_tree
def test_the_real_abstract_has_at_most_250_words_in_every_variant_and_the_test_a_sentence():
    from es.paper_lint import abstract_variants, _braced_arg, _strip_comment
    body = " ".join(_strip_comment(line) for line in _read("sections/00_abstract.tex").splitlines())
    variants = abstract_variants(_braced_arg(body, "\\abstract"))
    assert variants is not None
    assert {v.label for v in variants} == {"Test A has not run", "Test A has run", "Test A did not run"}
    assert all(150 <= v.words <= 250 for v in variants), {v.label: v.words for v in variants}
    ran = next(v for v in variants if v.label == "Test A has run").text
    assert "In Test A" in " ".join(ran.split())                                         # (the abstract of the last style pass says "In Test A, a pre-specified quarter,")
    # the interim estimate with its interval, as macros (the abstract of plan 2C gives the paired difference in the expected shortfall; it gave the SELC of the alerts before)
    assert re.search(r"\\\w+TestA\{\}", ran) and re.search(r"\\\w+TestACI\{\}", ran)
    assert re.search(r"pooled window", ran) and re.search(r"Test B is pending", " ".join(ran.split()))
    pending = next(v for v in variants if v.label == "Test A has not run").text
    assert "TestA" not in pending                                                      # nothing of Test A in the text before the run
    for v in variants:
        assert "pre-specified" in v.text and not re.search(r"regist", v.text, flags=re.I), v.label       # Ruling 133: pre-specified, nothing registered


TEXT_FILES = ("sections/00_abstract.tex", "esm/forward_tests.tex", "sections/07_discussion.tex", "sections/09_conclusion.tex")
TEST_A_TOKEN = re.compile(r"\\([A-Za-z]*TestA(?:CIBasic|CIPercentile|CI)?)(?![A-Za-z])")


def _variant(name, **setting):
    """The live text of a manuscript file as a reader sees it for one state of the switches (es.paper_lint resolves them)."""
    from es.paper_lint import SWITCHES, _resolve_switches
    state = dict.fromkeys(SWITCHES, False)
    state.update(setting)
    text = _resolve_switches(_live(_read(name)), state)
    assert text is not None, f"{name}: the \\if...\\fi switches are unbalanced"
    return text


def _sealed_a_macros(text):
    from es.paper_registry import entries_by_name
    by = entries_by_name()
    used = {m.group(1) for m in re.finditer(r"\\([A-Za-z]+)(?![A-Za-z])", text)} & set(by)
    return {n for n in used if by[n].stage == "sealedA"}


@needs_tree
def test_the_text_prints_nothing_of_test_a_before_the_run_and_only_what_a_test_that_did_not_run_still_has():
    """Three states of the switches: Test A not run yet (the real tree: no macro of Test A prints, and the sentinel of the pull record is not
    typeset), run (every macro that the text names of Test A is in the registry) and did not run (the macros that resolve to \\PENDINGA,
    \"test did not run\", are not used; the header macros of NOT_RUN_KEYS and the pull record are)."""
    from es.paper_macros import _in_not_run_header
    from es.paper_registry import entries_by_name
    by = entries_by_name()
    for name in TEXT_FILES:
        assert _sealed_a_macros(_variant(name)) == set(), name
        ran = _variant(name, TestA=True)
        assert _sealed_a_macros(ran), name                                              # each file has a Test A text
        assert {m.group(1) for m in TEST_A_TOKEN.finditer(ran)} <= set(by), name        # and names only macros of the registry
        unsafe = {n for n in _sealed_a_macros(_variant(name, TestA=True, TestAdidnotrun=True))
                  if not (_in_not_run_header(by[n]) or by[n].source.startswith("author:"))}
        assert unsafe == set(), (name, sorted(unsafe))
    assert "\\pullRecordTestA" in _variant("esm/forward_tests.tex", TestA=True, TestAdidnotrun=True)     # the pull is recorded either way
    assert "\\pullRecordTestA" in _variant("esm/forward_tests.tex", TestA=True)
    assert "\\pullRecordTestA" not in _variant("esm/forward_tests.tex")


@needs_tree
def test_the_test_a_branch_of_the_sealed_section_follows_the_binding_wording_rules():
    """The branch of \\ifTestA that holds the text of the subsection "Test A: July--September 2026" of esm/forward_tests.tex. Since plan 2C task T3b the file has an earlier,
    inline \\ifTestA in the paragraph of the subsection "Protocol record and knowledge statement" (two sentences, one for each state), so the branch is taken from the heading of
    the Test A subsection on; the first \\ifTestA of the file is no longer the one that this test is about."""
    section = _read("esm/forward_tests.tex").split("\\subsection{Test A: July--September 2026}", 1)[1]
    branch = " ".join(_branch(section).split())
    assert "\\ifTestAdidnotrun" in branch                                               # the coverage failure has its own text
    for macro in ("\\hbOnePTestA", "\\hbTwoPTestA", "\\hbOneStatusTestA", "\\hbTwoStatusTestA"):
        assert macro not in branch, macro                                              # H-B1 and H-B2 are estimands: no p-value
    for macro in ("\\hbOneCITestA", "\\hbOneCIBasicTestA", "\\hbTwoCITestA", "\\hbTwoCIBasicTestA", "\\hePTestA", "\\heStatusTestA",
                  "\\heApproxTestA", "\\selcTestA", "\\selcTestACI", "\\estimandsScopeTestA", "\\tauMaxDiffTestA", "\\startedUtcTestA",
                  "\\departuresTestA", "\\pullRecordTestA", "\\protocolShaTestA", "\\rowsReadTestA", "\\rowsKeptTestA", "\\usableNodesTestA"):
        assert macro in branch, macro
    for sentence in _sentences(branch):                                                 # H-B1 and H-B2 are estimands: no inferential reading
        if re.search(r"H-B[12]|\\hb(?:One|Two)|portab|block", sentence):
            for phrase in ("excludes zero", "excluding zero", "exclude zero", "less portable", "more portable"):
                assert phrase not in sentence, (phrase, sentence)
    for phrase in ("prospective", "reject H-E", "rejects H-E", "H-E is rejected"):
        assert phrase not in branch, phrase
    assert "not evidence that the features" in branch and "deterministic" in branch    # the tau check shows determinism and identical inputs, nothing more
    for sentence in re.split(r"(?<=[.;?!])\s+", branch):                               # no sentence compares Test A with Test B
        if "Test B" in sentence:
            assert re.search(r"pending|protocol records|pooled window|decided|January", sentence), sentence
            assert not re.search(r"\bagree|\bsame sign|compared with|than in Test B|versus", sentence), sentence
    for block in ("\\gentable{tab_sealed_coverage_A}", "\\gentable{tab_sealed_family_A}", "\\gentable{tab_sealed_e2e4_A}", "\\genfigure{Fig4}"):
        assert block in branch, block                                                  # the forward-test section keeps these (it was in the main text)
    for name in ("tab_sealed_pc_hs_A", "tab_sealed_e1_A", "tab_sealed_e5_A", "tab_sealed_audit_A", "tab_sealed_econ_A"):
        assert f"gentable{{{name}}}" not in branch, name                               # and the rest is in the Test A tables of ESM_1.tex
        assert f"gentable{{{name}}}" in _live(_read("esm/ESM_1.tex")), name


def _added_sentences(name, **setting):
    """The sentences that a state of the switches adds to the text of the same file before the run (the Test A text, not the development text
    around it, which says "excludes zero" about the development analyses)."""
    def sentences(text):
        return [x for x in re.split(r"(?<=[.;?!])\s+", " ".join(text.split())) if x]
    before = set(sentences(_variant(name)))
    return [x for x in sentences(_variant(name, **setting)) if x not in before]


@needs_tree
def test_the_conclusion_and_the_discussion_have_test_a_variants_with_macros_and_markers():
    for name in ("sections/09_conclusion.tex", "sections/07_discussion.tex"):
        assert "% CONTROLLER: choose wording by sign" in _read(name), name
        added = _added_sentences(name, TestA=True)
        assert added and any(_sealed_a_macros(x) for x in added), name                        # the Test A variant is written against macros
        for sentence in added:
            if re.search(r"H-B[12]|\\hb(?:One|Two)|portab|block", sentence):
                for phrase in ("excludes zero", "excluding zero", "less portable", "more portable"):
                    assert phrase not in sentence, (name, phrase, sentence)
            for phrase in ("prospective", "reject H-E", "rejects H-E", "H-E is rejected"):
                # Ruling 148: "neither test is prospective" and "not prospective" say the opposite of the claim that this guard stops
                assert phrase not in re.sub(r"(?:neither test is|not) prospective", "", sentence), (name, phrase)
            assert not (("Test A" in sentence or "TestA" in sentence) and "Test B" in sentence
                        and re.search(r"\bagree|\bsame sign|compared with|than in Test B|versus", sentence)), sentence
    conclusion = " ".join(_variant("sections/09_conclusion.tex", TestA=True).split())
    assert re.search(r"\\\w+TestA\{\}", conclusion) and re.search(r"\\\w+TestACI\{\}", conclusion)       # interim estimates with their intervals, as macros (the paired contrasts since plan 2C)
    assert "Test B is pending" in conclusion and "decided only on the pooled window" in conclusion
    assert "registered" not in conclusion
    # the interim p-value of H-E and the words "is not a decision" left the conclusion with the last style pass; Section 6.3 (the ran branch) keeps them
    forward = " ".join(_variant("sections/07_discussion.tex", TestA=True).split())
    forward = forward[forward.index("\\label{sec:disc-forward}"):forward.index("\\label{sec:limitations}")]
    assert "\\hePTestA" in forward and "is not a decision" in forward and "decided only on the pooled window" in forward


@needs_tree
def test_the_cover_letter_has_no_placeholder_for_what_the_sealed_run_resolves_and_the_title_candidates_name_no_reversal():
    for name in ("cover_letter.md", "cover_letter_TestA_ran.md"):                       # the letter for the submission as it stands, and the one for a Test A that has run
        letter = _read(name)
        for placeholder in ("[registration date", "[Test A", "[date of the registration", "[test a", "[OSF registration DOI]"):
            assert placeholder not in letter, (name, placeholder)                       # the repository address is the author's; nothing else is bracketed
        assert "the registration is public" not in letter, name                         # Ruling 133
    # Ruling 133, as the letters of plan 2C say it: before Test A, no third party holds a registered copy of the protocol; after it, the record is local and no third party held
    # the protocol before the pull (the words "not registered with a third party" are in the manuscript's back matter, which test_paper_wording_2c.py reads)
    assert re.search(r"no third party holds a registered copy of (?:its|the) protocol", _uploaded("cover_letter.md"))
    assert re.search(r"[Nn]o third party held (?:a copy of )?(?:its |the )?protocol before the (?:pull|data)", _uploaded("cover_letter_TestA_ran.md"))
    # the section of the manuscript that states the status of the forward tests and the record of the protocol is Section 6.3 since the discussion and the limitations became one
    # section (it was 6.1.5 and 8.2 before Test A, 6.2 after it); test_paper_wording_2c.py checks that the number typed in the letters is the number of sec:disc-forward in main.aux
    assert re.search(r"\(Section 6\.3(?: and Online Resource 1)?\)", _uploaded("cover_letter_TestA_ran.md"))
    assert re.search(r"\(Section 6\.3\)", _read("cover_letter.md").split("AUTHOR NOTES", 1)[1])
    titles = re.findall(r"^\s*(?:[-*]|\d+\.)\s*Title candidate \d: (.+)$", _read("AUTHOR_CHECKLIST.md"), flags=re.M)
    assert len(set(titles)) == len(titles)
    assert not any("revers" in t.lower() for t in titles)                               # the lint forbids "revers" in the title
