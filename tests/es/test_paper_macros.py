import json
from pathlib import Path

import pandas as pd
import pytest


def _out(tmp_path: Path) -> Path:
    out = tmp_path / "out"
    (out / "legacy").mkdir(parents=True)
    pd.DataFrame([
        {"analysis": "2024-2025", "block": "LEV6", "estimator": "xgb", "metric": "failures", "regenerated": 13.0},
        {"analysis": "2024-2025", "block": "LEV6", "estimator": "xgb", "metric": "n_nodes", "regenerated": 33.0},
        {"analysis": "2024-2025", "block": "LEV6", "estimator": "xgb", "metric": "median_R", "regenerated": 1.3449},
        {"analysis": "panel", "block": "all", "estimator": "-", "metric": "n_nodes", "regenerated": 33.0},
    ]).to_csv(out / "legacy" / "regenerated_long.csv", index=False)
    return out


def _paper(tmp_path: Path) -> Path:
    paper = tmp_path / "paper"
    paper.mkdir()
    (paper / "author_inputs.json").write_text(json.dumps({"title": "T", "short_title": "S", "email": "e@x.y",
        "department": "", "orcid": "", "github_url": "", "ssrn_abstract_id": "7562678", "ai_tools": "LLM"}))
    return paper


def test_registry_names_are_unique_letters_and_fmts_exist():
    from es.paper_registry import ENTRIES
    from es.paper_tex import FORMATS, PAIR_FORMATS, ROW_FORMATS, macro_name_ok
    names = [e.name for e in ENTRIES]
    assert len(names) == len(set(names)) and all(macro_name_ok(n) for n in names)
    assert all(e.fmt in FORMATS or e.fmt in PAIR_FORMATS or e.fmt in ROW_FORMATS for e in ENTRIES)
    assert all(e.stage in ("const", "dev", "sealedA", "sealedB", "author") for e in ENTRIES)
    assert {"failLevYY", "medRLevYY", "nNodes", "alphaPct", "hbOnePTestA", "selcTestB", "codeRepoURL", "fmmFirstDate",
            "bhRejectionsDriftMonitors", "bhTestsAuditRobustnessTestA", "bootSeed", "verScipy"} <= set(names)
    from es.paper_registry import PINS
    assert PINS["bootB"] == 2000 and PINS["bootBlockDays"] == 7 and PINS["pFourWindow"] == 56   # read from the code


def test_resolve_reads_value_pending_and_errors(tmp_path):
    from es.paper_macros import PENDING, Sources, resolve
    from es.paper_registry import entries_by_name
    src = Sources(_out(tmp_path), _paper(tmp_path), tmp_path / "docs_sealed")
    by = entries_by_name()
    assert resolve(by["failLevYY"], src)["value"] == "13"
    assert resolve(by["medRLevYY"], src)["value"] == "1.34"
    r = resolve(by["hbOnePTestA"], src)                       # sealed/A absent -> pending, not an error
    assert r["status"] == "pending" and r["value"] == PENDING["sealedA"]
    assert resolve(by["authorDept"], src)["value"] == PENDING["author"]   # empty author field -> pending
    with pytest.raises(LookupError):                           # file present, row absent -> registry bug
        resolve(by["failDartYY"], src)
    from es.paper_registry import Entry
    bad = Entry("badInt", "legacy/regenerated_long.csv",
                'analysis == "2024-2025" and block == "LEV6" and estimator == "xgb" and metric == "median_R"',
                "regenerated", "int", "dev")
    with pytest.raises(LookupError, match="badInt"):           # a formatter refusal names the macro
        resolve(bad, src)


def test_render_and_strict(tmp_path):
    from es.paper_macros import Sources, main, render, resolve_all
    from es.paper_registry import ENTRIES
    out, paper = _out(tmp_path), _paper(tmp_path)
    src = Sources(out, paper, tmp_path / "docs_sealed")
    sub = [e for e in ENTRIES if e.name in ("failLevYY", "hbOnePTestA", "authorDept", "alphaPct")]
    tex = render(resolve_all(sub, src), {"A": False, "B": False})
    assert "\\newcommand{\\failLevYY}{13}" in tex and "\\newcommand{\\alphaPct}{5}" in tex
    assert "\\newif\\ifTestA" in tex and "\\TestAfalse" in tex
    assert "\\newcommand{\\PENDINGA}" in tex and "\\newcommand{\\hbOnePTestA}{\\PENDINGA}" in tex
    rc = main(["--out-dir", str(out), "--paper-dir", str(paper), "--only", "failLevYY,hbOnePTestA,authorDept",
               "--strict", "--stages", "dev"])
    assert rc == 0 and (paper / "generated" / "macros.tex").exists()
    assert (paper / "generated" / "provenance.csv").read_text().count("\n") >= 3
    rc = main(["--out-dir", str(out), "--paper-dir", str(paper), "--only", "failLevYY,hbOnePTestA,authorDept",
               "--strict", "--stages", "dev,sealedA"])
    assert rc == 1


# ---------------------------------------------------------------------------------------------- fix round 1 (Ruling 61)
def _write_results(out: Path, test_id: str, results: dict) -> None:
    d = out / "sealed" / test_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "results.json").write_text(json.dumps(results))


NOT_RUN = {"test": "A", "started_utc": "2026-10-28T01:02:03+00:00", "protocol_sha256": "ab" * 32,
           "registration": "10.17605/OSF.IO/ABCDE", "registered_utc": "2026-10-20T15:04:00+00:00", "n_rep": 2000, "bound_files": {},
           "departures": [], "sealed_manifests": {}, "eta": 0.02, "family": ["H-E", "H-B1", "H-B2"],
           "rows_read": 1000, "rows_kept": 900, "coverage": {"ran": False, "usable_nodes": 20, "nonfinite_dropped": {}},
           "errors": {}, "status": "not run (coverage rule)",
           "finished_utc": "2026-10-28T01:05:00+00:00"}               # what run_sealed writes when the coverage rule fails (_execute)


def test_sealed_he_formats_are_pinned_and_selc_stays_the_pooled_estimand():
    from es.paper_registry import entries_by_name
    by = entries_by_name()
    for tw in ("TestA", "TestB"):
        assert (by[f"heTobs{tw}"].fmt, by[f"heP{tw}"].fmt, by[f"heNnull{tw}"].fmt, by[f"heStatus{tw}"].fmt) == \
            ("dec2", "pval", "int", "text")                            # t_obs is the studentized statistic (Ruling 55): unitless
        assert by[f"heTobs{tw}"].query == "H-E/t_obs"
        selc = by[f"selc{tw}"]
        assert selc.source == f"sealed/{tw[-1]}/tables/estimands.csv" and selc.query == 'stat == "SELC"'
        assert (selc.column, selc.fmt) == ("estimate", "pct1") and by[f"selc{tw}CI"].fmt == "cipct1"
    assert by["heWording"].fmt == "text" and by["heWording"].stage == "sealedB"


def test_author_file_present_but_key_missing_raises(tmp_path):
    from es.paper_macros import Sources, resolve
    from es.paper_registry import entries_by_name
    paper = tmp_path / "paper"
    paper.mkdir()
    (paper / "author_inputs.json").write_text(json.dumps({"title": "T"}))            # no "email" key
    by = entries_by_name()
    with pytest.raises(LookupError, match="authorEmail"):
        resolve(by["authorEmail"], Sources(tmp_path / "out", paper, tmp_path / "docs_sealed"))
    absent = Sources(tmp_path / "out", tmp_path / "no_paper", tmp_path / "docs_sealed")  # no author file yet: pending
    assert resolve(by["authorEmail"], absent)["status"] == "pending"


def test_missing_selector_column_raises_lookup_error_not_a_pandas_error(tmp_path):
    from es.paper_macros import Sources, resolve
    from es.paper_registry import Entry
    src = Sources(_out(tmp_path), _paper(tmp_path), tmp_path / "docs_sealed")
    bad = Entry("badSel", "legacy/regenerated_long.csv", 'no_such_column == "x"', "regenerated", "int", "dev")
    with pytest.raises(LookupError, match="badSel") as exc:
        resolve(bad, src)
    assert "no_such_column" in str(exc.value) and "metric" in str(exc.value)         # it lists the columns that exist
    syntax = Entry("badSyntax", "legacy/regenerated_long.csv", "metric ==", "regenerated", "int", "dev")
    with pytest.raises(LookupError, match="badSyntax"):
        resolve(syntax, src)


# ---------------------------------------------------------------------------------------------- Ruling 133: no OSF, no Zenodo
#: The macros that printed the OSF DOI, the date and the identifier of the registration (read from docs/sealed/REGISTRATION.md, with
#: osf_doi and osf_registered_utc of the author file as the fallback) and the DOI of the code archive (zenodo_doi). The sealed protocol is
#: not registered with a third party and the code is not archived at a DOI, so none of them exists, and no macro reads the record.
REMOVED_MACROS = ("osfDOI", "osfRegisteredDate", "registrationIdentifier", "codeDOI")


def _record(tmp_path, protocol="protocol v1\n", **fields):
    """docs_sealed/PROTOCOL.md and a REGISTRATION.md next to it, valid unless a field is overridden (None leaves its line out)."""
    import hashlib
    docs = tmp_path / "docs_sealed"
    docs.mkdir(exist_ok=True)
    (docs / "PROTOCOL.md").write_text(protocol)
    values = {"protocol_sha256": hashlib.sha256(protocol.encode()).hexdigest(), "osf_doi": "https://doi.org/10.17605/OSF.IO/ABCDE",
              "registered_utc": "2026-10-20T15:04:00Z"}
    values.update(fields)
    (docs / "REGISTRATION.md").write_text("# OSF registration record\n" + "".join(f"{k}: {v}\n" for k, v in values.items() if v is not None))
    return docs


def _paper_dir(tmp_path):
    return tmp_path / "paper" if (tmp_path / "paper").exists() else _paper(tmp_path)


def _author_stage(tmp_path, docs):
    """{macro: (status, value, sha, detail)} of every author-stage macro, resolved against a docs/sealed folder."""
    from es.paper_macros import Sources, resolve
    from es.paper_registry import ENTRIES
    src = Sources(tmp_path / "out", _paper_dir(tmp_path), docs)
    return {e.name: (r["status"], r["value"], r["sha"], r["detail"]) for e in ENTRIES if e.stage == "author" for r in [resolve(e, src)]}


def test_the_osf_and_zenodo_macros_are_gone_and_the_author_file_resolves_without_their_keys(tmp_path):
    """Where the tests of the registration macros asserted that osfDOI, osfRegisteredDate and registrationIdentifier were read from the
    record through the gate (and codeDOI from the author file), this asserts that they are not in the registry and that the author file
    resolves every macro that remains without a key of theirs; a key it lacks is still a registry bug (test_author_file_present_but_key_missing_raises)."""
    from es.paper_macros import PENDING, VALIDATORS
    from es.paper_registry import ENTRIES, entries_by_name
    assert not set(REMOVED_MACROS) & {e.name for e in ENTRIES} and not set(REMOVED_MACROS) & set(entries_by_name())
    values = _author_stage(tmp_path, tmp_path / "no_docs")
    assert set(values) == {"paperTitle", "paperShortTitle", "authorEmail", "authorDept", "authorOrcid", "codeRepoURL", "ssrnId", "aiTools"}
    assert {n: v[:2] for n, v in values.items()} == {
        "paperTitle": ("ok", "T"), "paperShortTitle": ("ok", "S"), "authorEmail": ("ok", "e@x.y"), "ssrnId": ("ok", "7562678"), "aiTools": ("ok", "LLM"),
        "authorDept": ("pending", PENDING["author"]), "authorOrcid": ("pending", PENDING["author"]), "codeRepoURL": ("pending", PENDING["author"])}
    assert all(v[3] == "empty author field" for n, v in values.items() if v[0] == "pending")
    assert set(VALIDATORS) == {"pull_record_test_a"}                    # the form rules of osf_doi and osf_registered_utc went with their macros


def test_a_registration_record_on_disk_changes_nothing_in_the_paper_build_and_the_gate_is_never_called(tmp_path, monkeypatch):
    """The old tests read a valid record, an embargoed one with no DOI, a record the gate refuses (placeholders, a repeated key, a future date,
    another registry's DOI, a changed or missing protocol) and checked the gate's own functions; now no macro reads any of them. Every
    author-stage macro resolves the same with each of these folders as with none, and the public functions of es.registration are not called."""
    import es.registration as registration
    from es.paper_macros import Sources

    def never(*args, **kwargs):
        raise AssertionError("the paper build called the sealed gate")

    for name in ("parse_registration", "verify_registration", "registration_identifier", "registration_identifiers"):
        monkeypatch.setattr(registration, name, never)
    assert not hasattr(Sources, "registration_record")                  # the reader of the record is gone from the engine
    baseline = _author_stage(tmp_path, tmp_path / "no_docs")
    records = {
        "a valid record": dict(),
        "an embargoed registration with no DOI": dict(osf_doi=None, osf_registration="https://osf.io/abcde/"),
        "an XXXXX placeholder in the DOI": dict(osf_doi="10.17605/OSF.IO/XXXXX"),
        "an angle-bracket placeholder in the date": dict(registered_utc="<UTC-date>"),
        "a registration dated in the future": dict(registered_utc="2999-01-01T00:00:00Z"),
        "a date that is not ISO 8601 UTC": dict(registered_utc="yesterday"),
        "a DOI of another registry": dict(osf_doi="10.1234/not-osf"),
        "a DOI and an address of two registrations": dict(osf_registration="https://osf.io/zzzzz/"),
        "no identifier at all": dict(osf_doi=None),
        "a hash that is not the protocol's": dict(protocol_sha256="0" * 64)}
    for label, fields in records.items():
        assert _author_stage(tmp_path, _record(tmp_path, **fields)) == baseline, label
    docs = _record(tmp_path)
    (docs / "REGISTRATION.md").write_text((docs / "REGISTRATION.md").read_text() + "osf_doi: 10.17605/OSF.IO/ZZZZZ\n")          # a repeated key
    assert _author_stage(tmp_path, docs) == baseline
    (docs / "PROTOCOL.md").write_text("protocol v1, quietly edited\n")                                                          # a changed protocol
    assert _author_stage(tmp_path, docs) == baseline
    (docs / "PROTOCOL.md").unlink()                                                                                              # and none
    assert _author_stage(tmp_path, docs) == baseline


def test_formatter_failures_name_the_macro(tmp_path, monkeypatch):
    import decimal
    from es import paper_tex
    from es.paper_macros import Sources, resolve
    from es.paper_registry import entries_by_name
    out = _out(tmp_path)
    _write_results(out, "B", {"coverage": {"ran": True, "usable_nodes": 33}, "holm": {"H-E": "maybe", "H-B1": True, "H-B2": False}})
    src = Sources(out, _paper(tmp_path), tmp_path / "docs_sealed")
    by = entries_by_name()
    with pytest.raises(LookupError, match="holmHE"):                      # an unknown yes/no string is an error, not "no"
        resolve(by["holmHE"], src)
    for exc in (OverflowError("too big"), decimal.InvalidOperation("bad")):

        def boom(v, exc=exc):
            raise exc

        monkeypatch.setitem(paper_tex.FORMATS, "int", boom)
        with pytest.raises(LookupError, match="failLevYY"):
            resolve(by["failLevYY"], src)


def test_a_test_that_did_not_run_resolves_to_sentinels_and_sets_the_flag(tmp_path):
    from es.paper_macros import PENDING, Sources, render, resolve, resolve_all, sealed_flags
    from es.paper_registry import entries_by_name
    out = _out(tmp_path)
    _write_results(out, "A", NOT_RUN)
    src = Sources(out, _paper(tmp_path), tmp_path / "docs_sealed")
    by = entries_by_name()
    for name in ("heTobsTestA", "hbOnePTestA", "tauMaxDiffTestA", "selcTestA", "eFiveFailLevTestA",
                 "bhTestsDriftMonitorsTestA"):                            # keys and tables that a not-run results.json lacks
        r = resolve(by[name], src)
        assert r["status"] == "pending" and r["value"] == PENDING["sealedA"] and r["detail"] == "test did not run", name
    assert resolve(by["ranTestA"], src)["value"] == "no"                  # the header that results.json does carry
    assert resolve(by["usableNodesTestA"], src)["value"] == "20" and resolve(by["rowsReadTestA"], src)["value"] == "1{,}000"
    assert resolve(by["startedTestA"], src)["value"] == "2026-10-28"
    assert resolve(by["runStatusTestA"], src)["value"] == "not run (coverage rule)"      # the run's own status is part of that header
    assert resolve(by["nRunErrorsTestA"], src)["value"] == "0"
    flags = sealed_flags(src)
    assert flags == {"A": True, "B": False, "Adidnotrun": True, "Bdidnotrun": False}
    tex = render(resolve_all([by["ranTestA"], by["heTobsTestA"]], src), flags)
    assert "\\newif\\ifTestAdidnotrun" in tex and "\\TestAdidnotruntrue" in tex
    assert "\\newif\\ifTestBdidnotrun" in tex and "\\TestBdidnotrunfalse" in tex
    assert "\\newcommand{\\heTobsTestA}{\\PENDINGA}" in tex


def test_a_test_that_ran_resolves_and_holm_null_prints_na(tmp_path):
    from es.paper_macros import Sources, resolve
    from es.paper_registry import entries_by_name
    from es.paper_tex import MINUS
    out = _out(tmp_path)
    res = {"coverage": {"ran": True, "usable_nodes": 33},
           "H-E": {"t_obs": -1.2345, "p": 0.0166, "n_null": 4159, "status": "decided (pooled A+B)"},
           "holm": {"H-E": None, "H-B1": True, "H-B2": False}}
    _write_results(out, "B", res)
    by = entries_by_name()
    paper = _paper(tmp_path)
    src = Sources(out, paper, tmp_path / "docs_sealed")
    assert resolve(by["heTobsTestB"], src)["value"] == MINUS + "1.23" and resolve(by["hePTestB"], src)["value"] == "0.017"
    assert [resolve(by[n], src)["value"] for n in ("holmHE", "holmHBOne", "holmHBTwo")] == ["n/a", "yes", "no"]
    res["H-E"]["p"] = None                                                # a null that is not nullable stays an error
    _write_results(out, "B", res)
    with pytest.raises(LookupError, match="hePTestB"):
        resolve(by["hePTestB"], Sources(out, paper, tmp_path / "docs_sealed"))


def test_test_flags_is_renamed_so_pytest_never_collects_it():
    from es import paper_macros
    assert hasattr(paper_macros, "sealed_flags") and not hasattr(paper_macros, "test_flags")


def test_unknown_stage_names_are_rejected_and_nothing_is_written(tmp_path):
    from es.paper_macros import main
    out, paper = _out(tmp_path), _paper(tmp_path)
    with pytest.raises(SystemExit) as exc:
        main(["--out-dir", str(out), "--paper-dir", str(paper), "--only", "failLevYY", "--strict", "--stages", "dev,sealed"])
    assert exc.value.code == 2 and not (paper / "generated").exists()


def test_only_refuses_to_overwrite_the_real_generated_dir(tmp_path, monkeypatch):
    from es import paper_macros
    real = tmp_path / "real_paper"                                        # stands in for paper/: the real one is never written
    real.mkdir()
    monkeypatch.setattr(paper_macros, "PAPER_DIR", real)
    out = _out(tmp_path)
    with pytest.raises(SystemExit) as exc:
        paper_macros.main(["--out-dir", str(out), "--only", "failLevYY"])    # the default --paper-dir is the (patched) real one
    assert exc.value.code == 2 and not (real / "generated").exists()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    assert paper_macros.main(["--out-dir", str(out), "--paper-dir", str(scratch), "--only", "failLevYY"]) == 0
    assert (scratch / "generated" / "macros.tex").exists()
    assert paper_macros.main(["--out-dir", str(out), "--only", "failLevYY", "--overwrite-real"]) == 0
    assert (real / "generated" / "macros.tex").exists()


def test_seed_macros_check_the_legacy_filenames(tmp_path):
    from es.paper_macros import main
    from es.paper_pins import SEEDS
    out, paper = _out(tmp_path), _paper(tmp_path)
    for s in SEEDS:
        (out / "legacy" / f"2024-2025__G6__xgb__s{s}__P1__a0.05.csv").write_text("x\n")
    args = ["--out-dir", str(out), "--paper-dir", str(paper), "--only", "seedList,nSeeds"]
    assert main(args) == 0
    extra = out / "legacy" / "2024-2025__G6__xgb__s99__P1__a0.05.csv"
    extra.write_text("x\n")
    with pytest.raises(LookupError, match="seeds"):                       # a seed the constants do not know
        main(args)
    extra.unlink()
    (out / "legacy" / "2024-2025__G6__xgb__s7__P1__a0.05.csv").unlink()
    with pytest.raises(LookupError, match="seeds"):                       # or a known seed whose runs are absent
        main(args)


def test_code_constants_record_the_sha_of_their_source_module(tmp_path):
    import hashlib

    import es
    from es.paper_macros import Sources, resolve, resolve_all
    from es.paper_registry import ENTRIES, entries_by_name
    src = Sources(_out(tmp_path), _paper(tmp_path), tmp_path / "docs_sealed")
    by = entries_by_name()
    es_dir = Path(es.__file__).resolve().parent

    def sha(name):
        return hashlib.sha256((es_dir / f"{name}.py").read_bytes()).hexdigest()[:16]

    assert resolve(by["alphaPct"], src)["sha"] == sha("config")
    assert resolve(by["seedList"], src)["sha"] == sha("run_legacy")       # the source of truth, hashed without importing it
    assert resolve(by["bootB"], src)["sha"] == sha("bootstrap") and resolve(by["pFourWindow"], src)["sha"] == sha("policies")
    assert resolve(by["posMW"], src)["sha"] == sha("paper_registry")      # typed in the registry
    rows = resolve_all([e for e in ENTRIES if e.source.startswith("code:")], src)
    assert all(r["sha"] for r in rows if r["value"] != "not installed" and r["status"] == "ok")


def test_author_text_is_tex_escaped(tmp_path):
    from es.paper_macros import Sources, resolve
    from es.paper_registry import entries_by_name
    paper = tmp_path / "paper"
    paper.mkdir()
    (paper / "author_inputs.json").write_text(json.dumps(
        {"title": "Drift & Money: 5% of $x_1$", "short_title": "S_1", "email": "a_b@x.y"}))
    src = Sources(tmp_path / "out", paper, tmp_path / "docs_sealed")
    by = entries_by_name()
    assert resolve(by["paperTitle"], src)["value"] == "Drift \\& Money: 5\\% of \\$x\\_1\\$"
    assert resolve(by["paperShortTitle"], src)["value"] == "S\\_1" and resolve(by["authorEmail"], src)["value"] == "a\\_b@x.y"


def test_text_macros_escape_bar_less_and_greater(tmp_path):
    from es.paper_macros import Sources, render, resolve, resolve_all
    from es.paper_registry import entries_by_name
    paper = tmp_path / "paper"
    paper.mkdir()
    (paper / "author_inputs.json").write_text(json.dumps({"ai_tools": "spike = |DART| >= q, MAD <= 1e-6 & more"}))
    src = Sources(tmp_path / "out", paper, tmp_path / "docs_sealed")
    by = entries_by_name()
    value = resolve(by["aiTools"], src)["value"]
    assert value == "spike = \\textbar{}DART\\textbar{} \\textgreater{}= q, MAD \\textless{}= 1e-6 \\& more"
    assert "\\newcommand{\\aiTools}{" + value + "}" in render(resolve_all([by["aiTools"]], src), {"A": False, "B": False})


def test_unreadable_source_files_raise_lookup_error_naming_the_macro(tmp_path):
    from es.paper_macros import Sources, resolve
    from es.paper_registry import entries_by_name
    out, paper = _out(tmp_path), _paper(tmp_path)
    (out / "sealed" / "B").mkdir(parents=True)
    (out / "sealed" / "B" / "results.json").write_text("{not json")
    by = entries_by_name()
    with pytest.raises(LookupError, match=r"ranTestB.*not valid JSON"):
        resolve(by["ranTestB"], Sources(out, paper, tmp_path / "docs_sealed"))
    (out / "legacy" / "regenerated_long.csv").write_text("")                       # an empty CSV
    with pytest.raises(LookupError, match=r"failLevYY.*is empty"):
        resolve(by["failLevYY"], Sources(out, paper, tmp_path / "docs_sealed"))


# ---------------------------------------------------------------------------------------------- fix round 2 (Ruling 71)
def test_parser_decode_permission_and_json_null_failures_name_the_macro(tmp_path):
    import os

    from es.paper_macros import Sources, resolve
    from es.paper_registry import entries_by_name
    out, paper, docs = _out(tmp_path), _paper(tmp_path), tmp_path / "docs_sealed"
    by = entries_by_name()
    csv = out / "legacy" / "regenerated_long.csv"
    header = "analysis,block,estimator,metric,regenerated\n"

    def fails(name, pattern):
        with pytest.raises(LookupError, match=pattern):
            resolve(by[name], Sources(out, paper, docs))

    csv.write_text(header + "2024-2025,LEV6,xgb,failures,13\n2024-2025,LEV6,xgb,n_nodes,33,EXTRA,FIELDS\n")   # a ragged row
    fails("failLevYY", r"failLevYY.*ParserError")
    csv.write_bytes(header.encode() + b"\xff\xfe,LEV6,xgb,failures,13\n")                                     # not UTF-8
    fails("failLevYY", r"failLevYY.*UnicodeDecodeError")
    csv.write_text(header + "2024-2025,LEV6,xgb,failures,13\n")
    os.chmod(csv, 0)                                                                                           # unreadable
    try:
        fails("failLevYY", r"failLevYY.*PermissionError")
    finally:
        os.chmod(csv, 0o644)
    for content in ("null", "[1, 2]", "7"):                                           # the file must hold a JSON object
        (paper / "author_inputs.json").write_text(content)
        fails("authorEmail", r"authorEmail.*JSON object")
    (paper / "author_inputs.json").write_text('{"email": null}')                      # a null value is an empty field
    assert resolve(by["authorEmail"], Sources(out, paper, docs))["status"] == "pending"


def test_a_json_at_the_analogs_stage_leaves_the_later_stage_keys_pending_not_an_error(tmp_path):
    """Ruling 98: protocol_inputs.json written by the analogs stage holds eta and the placebo rows, and `family` is null
    until the sim stage runs. The macros that read rho_star and the registered family stay pending then. Once the file
    says it is past that stage, a missing key is a registry bug again."""
    from es.paper_macros import PENDING, Sources, resolve
    from es.paper_registry import entries_by_name
    out, paper, docs = _out(tmp_path), _paper(tmp_path), tmp_path / "docs_sealed"
    (out / "analogs").mkdir()
    path = out / "analogs" / "protocol_inputs.json"
    path.write_text(json.dumps({"eta": 0.05, "family": None, "stage": "analogs", "placebo": []}))
    by = entries_by_name()
    assert resolve(by["etaStar"], Sources(out, paper, docs))["value"] == "0.05"        # what the stage wrote is read
    for name in ("rhoStar", "heConfirmatory", "hbOneFamilyStatus", "hbOneFalseDemotion", "ruleFactor", "phiCorrectedDart"):
        r = resolve(by[name], Sources(out, paper, docs))                                # the sim stage writes these keys
        assert r["status"] == "pending" and r["value"] == PENDING["dev"] and "stage 'analogs'" in r["detail"]
    r = resolve(by["hbOnePower"], Sources(out, paper, docs))                            # a row of simulation.csv, not yet written
    assert r["status"] == "pending" and "simulation.csv" in r["detail"]
    path.write_text(json.dumps({"eta": 0.05, "family": None, "stage": "sim", "placebo": []}))
    with pytest.raises(LookupError, match="rhoStar"):
        resolve(by["rhoStar"], Sources(out, paper, docs))
    path.write_text(json.dumps({"eta": 0.05, "placebo": []}))                          # no stage field: strict, as before
    with pytest.raises(LookupError, match="rhoStar"):
        resolve(by["rhoStar"], Sources(out, paper, docs))


def test_lag_and_lookback_constants_are_bound_to_the_pipeline():
    from v2.panel_prep import G1_LAGS, G1_ROLL_WINDOWS
    from es.paper_registry import CODE_CONSTANTS, CODE_SOURCES
    assert all(CODE_SOURCES[k] == "v2.panel_prep" for k in ("lagMinH", "lagMaxH", "lookbackH"))    # read from it, not typed
    assert CODE_CONSTANTS["lagMinH"] == min(G1_LAGS) and CODE_CONSTANTS["lagMaxH"] == max(G1_LAGS)
    assert CODE_CONSTANTS["lookbackH"] == max(G1_ROLL_WINDOWS) + min(G1_LAGS)       # rolling windows end lagMinH before the row


def test_constants_that_only_the_spec_fixes_say_so_in_their_provenance_row(tmp_path):
    import hashlib

    import es
    from es.paper_macros import Sources, resolve
    from es.paper_registry import SPEC_FIXED, entries_by_name
    src = Sources(_out(tmp_path), _paper(tmp_path), tmp_path / "docs_sealed")
    by = entries_by_name()
    sections = {"posMW": "§4.2", "pFourWindowAlt": "§4.3", "covMinNodes": "§6.1", "covMinShare": "§6.1"}
    # no es/ constant carries these four (impl_table says so too); Ruling 74 adds the two P4 window candidates of spec §12
    assert set(SPEC_FIXED) == set(sections) | {"pFourWindowMin", "pFourWindowMax"}
    for name, section in sections.items():
        detail = resolve(by[name], src)["detail"]
        assert detail.startswith("fixed by spec v3 ") and section in detail, (name, detail)
    for name in ("lagMinH", "lagMaxH", "lookbackH", "alphaPct", "seedList", "bootB"):
        assert resolve(by[name], src)["detail"] == ""             # bound to the code: nothing to mark
    panel_prep = Path(es.__file__).resolve().parents[1] / "v2" / "panel_prep.py"     # the module behind the lags, hashed
    assert resolve(by["lagMinH"], src)["sha"] == hashlib.sha256(panel_prep.read_bytes()).hexdigest()[:16]


def test_the_decile_sensitivity_replaces_the_month_strata(tmp_path):
    """Ruling 68: the primary strata are node x hour x weekday class x month; the deciles are the sensitivity."""
    from es.measures import STRATA_DECILE, STRATA_PRIMARY
    from es.paper_macros import Sources, resolve
    from es.paper_registry import CODE_CONSTANTS, ENTRIES, STAT_FMT, STAT_LABEL, STAT_ORDER, STAT_WORD, entries_by_name
    times = " $\\times$ "
    assert "strataMonth" not in CODE_CONSTANTS
    assert CODE_CONSTANTS["strataPrimary"] == times.join(s.replace("_", "\\_") for s in STRATA_PRIMARY)
    assert CODE_CONSTANTS["strataDecile"] == times.join(s.replace("_", "\\_") for s in STRATA_DECILE)
    assert "settle\\_hour" in CODE_CONSTANTS["strataPrimary"] and "cal\\_decile" in CODE_CONSTANTS["strataDecile"]
    assert STAT_WORD["E4:SELC_decile"] == "selcDecile" and "E4:SELC_month" not in STAT_WORD
    assert STAT_FMT["E4:SELC_decile"] == ("pct1", "cipct1") and STAT_LABEL["E4:SELC_decile"] == "E4: SELC, decile strata"
    assert STAT_ORDER.index("E4:SELC_decile") == 4                # the row keeps its place in the tables
    names = {e.name for e in ENTRIES}
    assert {"selcDecileAqThree", "selcDecileAqThreeCI", "selcDecileTestA", "selcDecileTestBCI"} <= names
    assert not [n for n in names if n.startswith("selcMonth")]
    by = entries_by_name()
    assert by["selcDecileTestA"].query == 'stat == "E4:SELC_decile"' and by["selcDecileTestA"].source.endswith("estimands.csv")
    out = _out(tmp_path)
    (out / "economics").mkdir()
    pd.DataFrame([{"analysis": "AQ3", "stat": "E4:SELC_decile", "estimate": 0.1234, "lo": 0.05, "hi": 0.2,
                   "approximate": False, "n_windows": 1}]).to_csv(out / "economics" / "bootstrap_ci.csv", index=False)
    src = Sources(out, _paper(tmp_path), tmp_path / "docs_sealed")
    assert resolve(by["selcDecileAqThree"], src)["value"] == "12.3\\%"
    assert resolve(by["selcDecileAqThreeCI"], src)["value"] == "[5.0\\%, 20.0\\%]"


def test_macros_tex_is_written_as_utf8_whatever_the_locale(tmp_path):
    """The strata texts hold the multiplication sign. Path.write_text without an encoding writes the locale's, which is
    ASCII under LC_ALL=C (a build run from cron or a minimal shell): the build would stop on the sign, or write it wrongly."""
    import os
    import subprocess
    import sys

    import es
    dart_dev = Path(es.__file__).resolve().parents[1]
    code = ("import sys\nfrom pathlib import Path\nfrom es.paper_macros import write_outputs\n"
            "row = {'name': 'strataPrimaryText', 'stage': 'const', 'status': 'ok', 'value': 'node \\u00d7 month',\n"
            "       'source': 'code:strataPrimaryText', 'query': '', 'column': '', 'fmt': 'raw', 'sha': '', 'detail': 'node \\u00d7'}\n"
            "write_outputs([row], Path(sys.argv[1]), {'A': False, 'B': False, 'Adidnotrun': False, 'Bdidnotrun': False})\n")
    env = {**os.environ, "LC_ALL": "C", "LANG": "C", "PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0",
           "DART_PROJECT_ROOT": os.environ.get("DART_PROJECT_ROOT", str(dart_dev.parent))}
    env.pop("PYTHONIOENCODING", None)
    r = subprocess.run([sys.executable, "-c", code, str(tmp_path / "generated")], cwd=dart_dev, env=env, capture_output=True,
                       text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-1500:]
    raw = (tmp_path / "generated" / "macros.tex").read_bytes()
    assert b"\\newcommand{\\strataPrimaryText}{node \xc3\x97 month}" in raw          # U+00D7 as UTF-8, not "?" or an error
    assert "\u00d7" in (tmp_path / "generated" / "provenance.csv").read_text(encoding="utf-8")



# ---------------------------------------------------------------------------------------------- Ruling 101: the engine
def _entry_source(tmp_path, rows, name="probe.csv"):
    from es.paper_macros import Sources
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / name, index=False)
    return Sources(out, tmp_path / "paper", tmp_path / "docs")


def _probe(**kw):
    from es.paper_registry import Entry
    base = dict(name="probe", source="probe.csv", query="", column="x", fmt="int", stage="dev")
    base.update(kw)
    return Entry(**base)


def test_argmax_reads_the_row_of_the_maximum_first_on_a_tie_and_refuses_a_missing_number(tmp_path):
    from es.paper_macros import resolve
    rows = [{"g": "a", "rate": 0.5, "x": 1, "law": "L1", "sc": "s1"}, {"g": "a", "rate": 0.7, "x": 2, "law": None, "sc": "s2"},
            {"g": "a", "rate": 0.7, "x": 3, "law": "L3", "sc": "s3"}, {"g": "b", "rate": 0.9, "x": 4, "law": "L4", "sc": "s4"}]
    src = _entry_source(tmp_path, rows)
    assert resolve(_probe(query='g == "a"', agg="argmax", by="rate"), src)["value"] == "2"       # the first of the two maxima
    assert resolve(_probe(query='g == "b"', agg="argmax", by="rate"), src)["value"] == "4"
    assert resolve(_probe(query='g == "a"', column="rate", fmt="dec1", agg="argmax", by="rate"), src)["value"] == "0.7"
    where = resolve(_probe(query='g == "a"', column="law,sc", fmt="where", agg="argmax", by="rate"), src)
    assert where["value"] == "s2"                                                                # the row's missing law is skipped
    # nothing selected, a missing number in the rates, a `by` that does not exist or is not named: errors that name the macro
    for kw, pattern in ((dict(query='g == "c"', by="rate"), "matched no rows"), (dict(query='g == "a"', by="nope"), "nope"),
                        (dict(query='g == "a"', by=""), "needs a `by`")):
        with pytest.raises(LookupError, match=rf"probe.*{pattern}"):
            resolve(_probe(agg="argmax", **kw), src)
    rows[2]["rate"] = None
    src = _entry_source(tmp_path, rows)
    with pytest.raises(LookupError, match=r"probe.*rate.*missing"):
        resolve(_probe(query='g == "a"', agg="argmax", by="rate"), src)
    assert resolve(_probe(query='g == "b"', agg="argmax", by="rate"), src)["value"] == "4"      # rows the query leaves out do not count


def test_same_needs_one_value_over_all_the_selected_rows(tmp_path):
    from es.paper_macros import resolve
    src = _entry_source(tmp_path, [{"g": "a", "x": 300, "y": 1}, {"g": "a", "x": 300, "y": 2}, {"g": "b", "x": 200, "y": 3}])
    assert resolve(_probe(query='g == "a"', agg="same"), src)["value"] == "300"
    with pytest.raises(LookupError, match=r"probe.*differ in y"):
        resolve(_probe(query='g == "a"', column="y", agg="same"), src)
    with pytest.raises(LookupError, match=r"probe.*differ in x"):
        resolve(_probe(agg="same"), src)
    with pytest.raises(LookupError, match=r"probe.*matched no rows"):
        resolve(_probe(query='g == "c"', agg="same"), src)


def test_a_nullable_csv_value_prints_n_a_for_a_nan_and_nothing_else_does(tmp_path):
    from es.paper_macros import resolve
    src = _entry_source(tmp_path, [{"g": "a", "x": None, "lo": None, "hi": 2.0, "t": None}, {"g": "b", "x": 1.5, "lo": 1.0, "hi": 2.0, "t": "n"}])
    assert resolve(_probe(query='g == "a"', fmt="dec2", nullable=True), src)["value"] == "n/a"
    assert resolve(_probe(query='g == "b"', fmt="dec2", nullable=True), src)["value"] == "1.50"
    assert resolve(_probe(query='g == "a"', column="t", fmt="text", nullable=True), src)["value"] == "n/a"
    with pytest.raises(LookupError, match=r"probe.*missing"):
        resolve(_probe(query='g == "a"', fmt="dec2"), src)                                      # not nullable: an error, as before
    with pytest.raises(LookupError, match=r"probe.*missing"):
        resolve(_probe(query='g == "a"', column="lo,hi", fmt="ci2", nullable=True), src)         # half a pair is never n/a


def test_a_row_format_takes_all_the_columns_and_a_row_without_labels_is_an_error(tmp_path):
    from es.paper_macros import resolve
    src = _entry_source(tmp_path, [{"g": "a", "law": "L_1", "sc": "no_drift", "per": "fit"},
                                   {"g": "b", "law": None, "sc": "het", "per": "upper"}, {"g": "c", "law": None, "sc": None, "per": None}])
    assert resolve(_probe(query='g == "a"', column="law,sc,per", fmt="where"), src)["value"] == "L\\_1, no\\_drift, fit"
    assert resolve(_probe(query='g == "b"', column="law,sc,per", fmt="where"), src)["value"] == "het, upper"
    with pytest.raises(LookupError, match=r"probe.*cannot format"):
        resolve(_probe(query='g == "c"', column="law,sc,per", fmt="where"), src)
