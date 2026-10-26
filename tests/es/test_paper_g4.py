"""Ruling 133 (Sunday 25 October 2026): the protocol of the sealed tests is not registered on OSF and the code is not archived at a DOI. The
printed text says "pre-specified", "designation rule", "designated status" and "confirmatory family", and nothing in it says that anything is
registered, public, on OSF or archived. These tests keep the macro registry, the author file, the lint, the generated tables and the upload
set in step with that text: the four macros that named an OSF or Zenodo identifier are gone, the author file has no key for them, the
printed text of the real tree has no OSF, Zenodo or registration claim (five sentences that say the opposite on purpose are the only
exceptions), the declarations lint asks for the heading "Prior dissemination", and the zip that the packager builds holds no OSF text.
The tests that read the real paper/ folder skip when it is not next to dart_dev, as test_paper_g1.py and test_paper_g3.py do."""
from __future__ import annotations

import csv
import json
import re
import shutil
import zipfile
from pathlib import Path

import pytest

PAPER = Path(__file__).resolve().parents[3] / "paper"
needs_tree = pytest.mark.skipif(not (PAPER / "sections").exists(), reason="the manuscript folder is not next to dart_dev")

REMOVED_MACROS = ("codeDOI", "osfDOI", "osfRegisteredDate", "registrationIdentifier")
REMOVED_AUTHOR_KEYS = ("zenodo_doi", "osf_doi", "osf_registered_utc")
#: The author-stage macros that remain: the six that the author's file feeds (title, short title, e-mail, department, ORCID, repository
#: address), the SSRN id and the statement on the use of AI tools. Nothing here names a registration or an archive.
AUTHOR_STAGE = {"paperTitle", "paperShortTitle", "authorEmail", "authorDept", "authorOrcid", "codeRepoURL", "ssrnId", "aiTools"}


# ---------------------------------------------------------------------------------------------------- the registry and the author file
def test_the_registry_has_none_of_the_four_macros_that_named_an_osf_or_zenodo_identifier():
    from es.paper_registry import ENTRIES, entries_by_name
    names = {e.name for e in ENTRIES}
    assert not names & set(REMOVED_MACROS), sorted(names & set(REMOVED_MACROS))
    assert not set(REMOVED_MACROS) & set(entries_by_name())
    # no entry reads the record of a registration (the source kind "docs:" is gone), and none reads an author key about OSF or Zenodo
    assert not [e.name for e in ENTRIES if e.source.startswith("docs:")]
    assert not [e.name for e in ENTRIES if re.search(r"\bosf|zenodo", e.source, re.I)]
    assert {e.name for e in ENTRIES if e.stage == "author"} == AUTHOR_STAGE
    # what stays: the time of the protocol record and the protocol's hash, both read from the sealed run's own output
    by = entries_by_name()
    for tw in ("TestA", "TestB"):
        assert (by[f"registeredUtc{tw}"].source, by[f"registeredUtc{tw}"].query) == (f"sealed/{tw[-1]}/results.json", "registered_utc")
        assert by[f"protocolSha{tw}"].source == f"sealed/{tw[-1]}/results.json"
    assert by["protocolSha"].source == "docsfile:PROTOCOL.sha256"


@needs_tree
def test_the_author_file_has_none_of_the_three_keys_of_the_removed_macros_and_every_author_source_has_its_key():
    data = json.loads((PAPER / "author_inputs.json").read_text(encoding="utf-8"))
    assert not set(REMOVED_AUTHOR_KEYS) & set(data), sorted(set(REMOVED_AUTHOR_KEYS) & set(data))
    assert {"title", "short_title", "email", "department", "orcid", "github_url", "ssrn_abstract_id", "ai_tools",
            "pull_record_test_a"} <= set(data)                                                   # every other key stays
    from es.paper_registry import ENTRIES
    wanted = {e.source.split(":", 1)[1] for e in ENTRIES if e.source.startswith("author:")}
    assert wanted <= set(data), sorted(wanted - set(data))                                       # a missing key is a LookupError of the build
    assert not [k for k in data if re.search(r"osf|zenodo", k, re.I)]


def test_the_notes_of_the_registry_use_the_words_of_the_manuscript():
    """The notes of the entries are written to generated/registry.csv, which reviewers read: the rule is the designation rule, the protocol is
    not registered, and the time of the run's protocol record is the protocol record's. (The path of REGISTRATION.md is a path.)"""
    from es.paper_registry import ENTRIES
    bad = {e.name: e.note for e in ENTRIES if re.search(r"regist(?:er|ered|ers|ration)", e.note.replace("docs/sealed/REGISTRATION.md", ""), flags=re.I)}
    assert not bad, bad
    notes = {e.name: e.note for e in ENTRIES}
    assert notes["ruleFactor"] == "a constant of the designation rule, as the run used it"
    assert notes["protocolSha"].startswith("SHA-256 of the protocol, from docs/sealed/PROTOCOL.sha256")
    assert notes["registeredUtcTestA"].startswith("the time of the protocol record that the run verified at its start")
    assert "differ from the manifest of the protocol" in notes["departuresTestA"] and notes["heApproxTestB"].startswith("whether the designation rule")
    assert all(notes[f"sizeCheck{w}"].endswith("not part of the designation rule") for w in ("PooledThird", "PooledFive", "SingleThird", "SingleFive", "Reps", "Phi"))


def test_the_test_b_sentinel_no_longer_says_registered():
    from es.paper_macros import PENDING_TEXT
    assert PENDING_TEXT["sealedB"] == "[pending: Test B]"
    assert PENDING_TEXT == {"const": "[pending: constant]", "dev": "[pending: development results]", "sealedA": "[pending: Test A]",
                            "sealedB": "[pending: Test B]", "author": "[pending: author input]"}


@needs_tree
def test_the_generated_files_of_the_real_tree_define_none_of_the_four_macros_and_use_the_new_sentinel():
    macros = (PAPER / "generated" / "macros.tex").read_text(encoding="utf-8")
    for name in REMOVED_MACROS:
        assert f"\\newcommand{{\\{name}}}" not in macros, name
    assert "\\newcommand{\\PENDINGB}{\\textbf{[pending: Test B]}}" in macros and "[pending: Test B, registered]" not in macros
    for csv_name in ("provenance.csv", "registry.csv"):
        with open(PAPER / "generated" / csv_name, encoding="utf-8", newline="") as fh:
            names = {row["name"] for row in csv.DictReader(fh)}
        assert not names & set(REMOVED_MACROS), (csv_name, sorted(names & set(REMOVED_MACROS)))
        assert "registeredUtcTestA" in names and "protocolShaTestA" in names                       # the sealed-stage macros stay


# ---------------------------------------------------------------------------------------------------- the printed text of the real tree
#: What the printed text may no longer say. Case matters except for the word "preregistration".
FORBIDDEN = tuple(re.compile(p) for p in (
    r"\bOSF\b", r"Open Science", r"Zenodo", r"preregistered", r"pre-registered", r"registered on", r"registration is public",
    r"\bregistered (?:copy|family|status|protocol|code|design|test|hypothesis|analysis|grid|rule|manifest|wording)\b",
    r"\bregistration (?:rule|record|identifier|DOI)\b"))
#: The five sentences that say the opposite on purpose: nothing is registered with a third party, and nobody holds a registered copy.
#: (Plan 2C task T3b: the sentence of the discussion, Section 6.3, begins with the phrase, so it is spelled with its capital; the cover letter says "no third party holds a
#: registered copy of its protocol", the Online Resource the two "No third party ... registered the protocol", and the back matter says in one sentence that neither test is registered
#: with a third party, which the controller words as it finds fit: the forms of that denial are one group, and one of them has to be in the printed text.)
ALLOWED_GROUPS = (("no third party holds a registered copy of it",),
                  ("No third party registered the protocol",),
                  ("No third party has registered the protocol",),
                  ("are not registered with a third party", "is not registered with a third party", "Neither test is registered with a third party"),
                  ("No third party holds a registered or time-stamped copy",))
ALLOWED_PHRASES = tuple(phrase for group in ALLOWED_GROUPS for phrase in group)
#: The word "preregistration" appeared once, in the sentence that says the protocol follows its logic without being one. The plain-language introduction of plan 2C has no such
#: sentence any more and no printed file uses the word; the exception stays in the guard (the guard test below lets the sentence through) and is not required to be alive.
PREREGISTRATION_PHRASE = "follows the logic of a preregistration"
_PREREGISTRATION = re.compile(r"\bpre-?registration\b", re.I)


def _live_tex(text: str) -> str:
    """The text without its LaTeX comments, full-line and trailing (a percent sign after a backslash is a percent sign)."""
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in text.splitlines())


def _live_md(text: str) -> str:
    return re.sub(r"<!--.*?-->", "", text, flags=re.S)


def _flat(text: str) -> str:
    return " ".join(text.split())


def violations(text: str) -> list[tuple[str, str]]:
    """[(pattern, the words around it)] of the forbidden phrases in ``text``, after the five allowed sentences and the one allowed use
    of the word "preregistration" are taken out."""
    flat = _flat(text)
    for phrase in (*ALLOWED_PHRASES, PREREGISTRATION_PHRASE):
        flat = flat.replace(phrase, " ")
    found = [(rx.pattern, flat[max(0, m.start() - 50):m.end() + 50]) for rx in FORBIDDEN for m in rx.finditer(flat)]
    return found + [(_PREREGISTRATION.pattern, flat[max(0, m.start() - 50):m.end() + 50]) for m in _PREREGISTRATION.finditer(flat)]


def _printed_text(paper: Path = PAPER) -> dict[str, str]:
    """{file: its live text}: the sections, the Online Resource documents and the sections they input (esm/*.tex: since plan 2C the forward-test
    design and the size study of the designation rule are there), main.tex, crossref.tex and the cover letter without its notes. (An appendix
    would be scanned too; the manuscript has none.)"""
    files = (sorted(paper.glob("sections/*.tex")) + sorted(paper.glob("appendices/*.tex")) + sorted(paper.glob("esm/*.tex"))
             + [paper / "main.tex", paper / "crossref.tex"])
    out = {f.relative_to(paper).as_posix(): _live_tex(f.read_text(encoding="utf-8")) for f in files}
    out["cover_letter.md"] = _live_md((paper / "cover_letter.md").read_text(encoding="utf-8"))
    return out


@needs_tree
def test_the_printed_text_of_the_real_tree_has_no_osf_zenodo_or_registration_claim():
    texts = _printed_text()
    assert {"main.tex", "crossref.tex", "cover_letter.md", "sections/00_abstract.tex", "sections/90_backmatter.tex", "esm/ESM_1.tex",
            "esm/forward_tests.tex", "esm/designation_rule_study.tex"} <= set(texts)          # the scan reaches every kind of file, the moved ones too
    bad = {name: violations(text) for name, text in texts.items() if violations(text)}
    assert not bad, "\n".join(f"{name}: {pattern!r} in ...{context}..." for name, hits in bad.items() for pattern, context in hits)


@needs_tree
@pytest.mark.parametrize("group", ALLOWED_GROUPS, ids=[g[0] for g in ALLOWED_GROUPS])
def test_each_allowed_sentence_is_in_the_printed_text_so_that_no_exception_is_dead(group):
    assert any(phrase in _flat(text) for phrase in group for text in _printed_text().values()), f"none of {group!r} is in a printed file: delete the exception"


@needs_tree
def test_the_word_preregistration_is_in_the_printed_text_only_in_the_sentence_that_denies_being_one():
    """Protected: the word "preregistration" is not used for the protocol. It stood once, in the introduction, in the sentence that says that the protocol follows the logic of a
    preregistration without being one. Plan 2C (task T3b) rewrote the introduction: no printed file uses the word now, which is the safeguard in its strongest form. The sentence
    stays an allowed exception, so a file that brings the word back is flagged unless it is that sentence."""
    hits = [name for name, text in _printed_text().items() if _PREREGISTRATION.search(_flat(text).replace(PREREGISTRATION_PHRASE, " "))]
    assert hits == [], hits


@pytest.mark.parametrize("sentence", [
    "The protocol was registered on OSF.", "We fixed it on the Open Science Framework.", "The code is archived at Zenodo.",
    "Two preregistered sealed tests.", "Two pre-registered sealed tests.", "The registration is public.",
    "The registered family is H-E alone.", "its registered status", "the registered protocol", "the registered code", "the registered design",
    "each registered test", "a registered hypothesis", "the registered analysis", "the registered grid", "the registered rule",
    "the registered manifest", "the registered wording", "The registration rule keeps one test.", "the registration record is dated",
    "the registration identifier is", "the registration DOI is", "Preregistration and prior dissemination", "a Preregistration of the design",
    "a pre-registration of the design", "The protocol follows the logic of a preregistration and is also a preregistration."])
def test_the_guard_flags_each_forbidden_phrase(sentence):
    assert violations(sentence), sentence


@pytest.mark.parametrize("sentence", [*[f"x {p} y." for p in ALLOWED_PHRASES], f"The protocol {PREREGISTRATION_PHRASE}~\\cite{{nosek2018preregistration}}.",
                                      "Ranking metrics do not register this.", "The designation rule keeps one confirmatory test.",
                                      "The protocol record is dated \\registeredUtcTestA{}.", "The primary family as designated is pre-specified.",
                                      "The sealed tests have not been run and are not registered with a third party."])
def test_the_guard_lets_through_the_allowed_sentences_and_the_new_vocabulary(sentence):
    assert violations(sentence) == [], sentence


def test_the_guard_ignores_comments_and_the_notes_of_the_cover_letter_and_nothing_else():
    assert violations(_live_tex("Plain text.\n% a full-line comment about OSF and Zenodo\nMore text. % OSF, a trailing comment\n")) == []
    assert violations(_live_tex("Fifty\\% of OSF counts.\n")) != []                            # an escaped percent sign is text, not a comment
    assert violations(_live_md("Letter text.\n\n<!-- AUTHOR NOTES: the OSF and Zenodo notes\nover two lines. -->\n")) == []
    assert violations(_live_md("Letter text about OSF.\n<!-- note -->\n")) != []


# ---------------------------------------------------------------------------------------------------- the generated tables of the real tree
@needs_tree
def test_the_generated_tables_of_the_real_tree_print_no_registered_status_registration_rule_osf_or_zenodo():
    """The headers, notes and footnotes that the generators write (es.paper_tables, es.paper_tables_sealed) and the index of the tables.
    The name of the table, tab_registration, is an identifier and is not printed text."""
    tables = sorted((PAPER / "generated" / "tables").glob("*.tex")) + [PAPER / "generated" / "tables" / "INDEX.md"]
    assert len(tables) > 40
    word = re.compile(r"\bregist(?:er|ers|ered|ering|ration)\b|\bOSF\b|Open Science|Zenodo|prereg", re.I)
    bad = {}
    for path in tables:
        text = path.read_text(encoding="utf-8")
        text = _live_tex(text) if path.suffix == ".tex" else text
        hits = word.findall(re.sub(r"tab[_:]registration", " ", text))
        if hits:
            bad[path.name] = hits
    assert not bad, bad


# ---------------------------------------------------------------------------------------------------- the abstract and the declarations
@needs_tree
def test_the_abstract_in_the_pending_state_has_at_most_250_words_and_the_keywords_are_four_to_six():
    from es.paper_lint import _strip_comment, abstract_word_counts
    counts = abstract_word_counts(PAPER)
    assert counts["Test A has not run"] <= 250, counts              # the state of the manuscript as submitted
    assert all(words <= 250 for words in counts.values()), counts   # and of the other readers' abstracts
    body = " ".join(_strip_comment(line) for line in (PAPER / "sections" / "00_abstract.tex").read_text(encoding="utf-8").splitlines())
    keywords = [k.strip() for k in re.search(r"\\keywords\{([^}]*)\}", body).group(1).split(",") if k.strip()]
    assert 4 <= len(keywords) <= 6, keywords
    assert not [k for k in keywords if re.search(r"regist", k, re.I)]


def _declarations(paper: Path) -> list:
    from es.paper_lint import lint_all
    return [f for f in lint_all(paper, strict=False) if f.rule == "declarations"]


def _backmatter_paper(tmp_path: Path, text: str) -> Path:
    paper = tmp_path / "paper"
    (paper / "sections").mkdir(parents=True)
    (paper / "sections" / "90_backmatter.tex").write_text(text, encoding="utf-8")
    return paper


@needs_tree
def test_the_declarations_lint_passes_with_the_new_heading_and_asks_for_it(tmp_path):
    from es.paper_lint import REQUIRED_HEADINGS
    assert "Prior dissemination" in REQUIRED_HEADINGS and not [h for h in REQUIRED_HEADINGS if "egistration" in h]
    real = (PAPER / "sections" / "90_backmatter.tex").read_text(encoding="utf-8")
    assert "\\bmhead{Prior dissemination}" in real
    assert _declarations(PAPER) == []                                                          # the real manuscript
    assert _declarations(_backmatter_paper(tmp_path / "ok", real)) == []
    missing = real.replace("\\bmhead{Prior dissemination}", "\\bmhead{Something else}")
    findings = _declarations(_backmatter_paper(tmp_path / "missing", missing))
    assert [(f.severity, f.message) for f in findings] == [("error", "missing \\bmhead{Prior dissemination...}")]
    old = real.replace("\\bmhead{Prior dissemination}", "\\bmhead{Preregistration and prior dissemination}")
    assert [f.message for f in _declarations(_backmatter_paper(tmp_path / "old", old))] == ["missing \\bmhead{Prior dissemination...}"]


# ---------------------------------------------------------------------------------------------------- the upload set
TEXT_SUFFIXES = (".tex", ".md", ".csv", ".txt", ".bib", ".bbl", ".bst", ".cls", ".json")
#: OSF, Open Science and Zenodo in any case, and the traces of the removed macros and author keys (osf_doi, \osfDOI, zenodo_doi, ...).
OSF_TEXT = re.compile(r"\bOSF\b|open science|zenodo|\bosf_|\\osf[A-Z]", re.I)


def _zip_text_hits(paper: Path, out: Path) -> dict[str, list[str]]:
    """{member: hits} of OSF, Open Science and Zenodo in the text members of the zip that the packager builds from ``paper``, except
    references.bib. The PDFs and the EPS files are binary and not read here (build.sh pdfcheck and the dry run read the text of the PDFs)."""
    from es.paper_package import build_package
    z = zipfile.ZipFile(build_package(paper, out))
    hits = {}
    for name in z.namelist():
        if not name.endswith(TEXT_SUFFIXES) or name == "references.bib":
            continue
        found = OSF_TEXT.findall(z.read(name).decode("utf-8", "replace"))
        if found:
            hits[name] = found
    assert z.namelist() and "main.tex" in z.namelist() and "cover_letter.md" in z.namelist()
    return hits


def _scratch_copy(tmp_path: Path) -> Path:
    scratch = tmp_path / "paper"
    shutil.copytree(PAPER, scratch, ignore=shutil.ignore_patterns("submission", "build"))
    return scratch


@needs_tree
def test_the_zip_of_a_scratch_copy_of_the_real_tree_has_no_osf_open_science_or_zenodo_text(tmp_path):
    scratch = _scratch_copy(tmp_path)
    assert _zip_text_hits(scratch, scratch / "submission") == {}
    z = zipfile.ZipFile(next((scratch / "submission").glob("*.zip")))
    names = set(z.namelist())
    assert len(names) > 60 and {"generated/macros.tex", "generated/provenance.csv", "CORRECTIONS_TO_PREPRINT.md"} <= names
    assert not names & {"author_inputs.json", "REVISION_LOG.md", "AUTHOR_CHECKLIST.md", "SUBMISSION_REQUIREMENTS_CHECK.md"}   # the files that name OSF stay home


@needs_tree
def test_the_zip_scan_sees_live_text_and_not_the_comments_that_the_packager_strips(tmp_path):
    """The scan is not blind: OSF in a printed line of a section and in the macros reaches the zip and is found; OSF in a full-line comment,
    in a trailing note of ours and in the notes of the cover letter does not reach it."""
    scratch = _scratch_copy(tmp_path)
    section = scratch / "sections" / "02_related.tex"
    section.write_text(section.read_text(encoding="utf-8") + "\n%% a comment line about OSF is stripped\nA line about Zenodo is live.\n"
                       "Another one % AUTHOR: OSF in a trailing note of ours is stripped\n", encoding="utf-8")
    letter = scratch / "cover_letter.md"
    letter.write_text(letter.read_text(encoding="utf-8") + "\n<!-- OSF notes to the author are stripped -->\nA letter line about the Open Science Framework.\n",
                      encoding="utf-8")
    hits = _zip_text_hits(scratch, scratch / "submission")
    assert hits == {"sections/02_related.tex": ["Zenodo"], "cover_letter.md": ["Open Science"]}, hits


# ---------------------------------------------------------------------------------------------------- no absolute path in the upload set
def test_a_pending_source_is_named_relative_to_the_project_in_the_provenance_detail(tmp_path):
    """generated/provenance.csv is uploaded: its detail column must not carry the author's absolute paths (user name, folder names)."""
    from es.paper_macros import PENDING, Sources, resolve
    from es.paper_registry import entries_by_name
    src = Sources(tmp_path / "out", tmp_path / "paper", tmp_path / "docs" / "sealed")
    by = entries_by_name()
    for name, prefix in (("selcTestA", "missing outputs/es/sealed/A/"), ("protocolSha", "missing docs/sealed/PROTOCOL.sha256")):
        assert name in by, name
        r = resolve(by[name], src)
        assert r["status"] == "pending", name
        assert r["detail"].startswith(prefix) and str(tmp_path) not in r["detail"] and "/Users/" not in r["detail"], (name, r["detail"])
    assert src.shown("author:github_url") == "paper/author_inputs.json"
    assert src.shown("sealed/B/results.json") == "outputs/es/sealed/B/results.json"
    assert src.shown("docsfile:PROTOCOL.sha256") == "docs/sealed/PROTOCOL.sha256"


@needs_tree
def test_the_provenance_file_of_the_real_tree_has_no_absolute_path_and_no_user_name():
    prov = PAPER / "generated" / "provenance.csv"
    if not prov.exists():
        pytest.skip("not built yet")
    text = prov.read_text(encoding="utf-8")
    assert "/Users/" not in text and "erceerez" not in text and "ICAIF" not in text

