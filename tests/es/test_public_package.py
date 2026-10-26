"""Tests of es.public_package (spec invariant I9: the public package holds only this paper's price-only pipeline).

Every test but the last group builds a fake dev tree under tmp_path: an included and an excluded file, the refusals, byte
identity of the copies, the four files the package gets, the optional protocol/ folder (a fake docs/sealed and a fake
outputs/es/sealed) and the three states of the README, and the scans. The last group reads the real tree
and writes nothing; no test of this file reads the real docs/sealed or the real sealed outputs. Sample secrets, addresses and
home paths are put together from pieces so that this file holds none of them in one piece for the scans to find."""
import hashlib
import importlib.metadata as metadata
import json
import re
from pathlib import Path

import pytest

AT = "@"
HOME = "/Users" + "/" + "someone"

#: The fake tree. Odd bytes on purpose: CRLF, a tab and trailing blanks, a byte that is not UTF-8, no final newline.
INCLUDED = {
    "es/__init__.py": b"",
    "es/sealed_run.py": b"from . import helper\r\nfrom v2 import x\r\n\r\nVALUE = 1  \t# trailing blanks\r\n",
    "es/helper.py": b"HELPER = 'caf\xc3\xa9'\n# an accented e, in UTF-8",
    "v2/__init__.py": b"",
    "v2/x.py": b"import json\n",
    "v2/NODE_MAP.csv": b"market,node_id\r\nNYISO,WEST\r\nISONE,.H.INTERNAL_HUB",
    "cache/common_panel_groups.json": b'{"a": "\xff"}',                          # a byte that is not UTF-8, in a data file
    "pytest.ini": b"[pytest]\naddopts = -q\n",
    "tests/es/conftest.py": b"",
    "tests/es/test_helper.py": b"def test_x():\n    assert True\n",
}
#: Present in the tree, reachable from no entry point and not under tests/es: must stay out.
UNRELATED = {
    "src/afd.py": b"X = 1\n",
    "CLAUDE.md": b"# rules\n",
    "notebook.ipynb": b"{}",
    "outputs/es/result.csv": b"a,b\n1,2\n",
    "cache/other.parquet": b"PAR1",
    "cache/other.json": b"{}",
    ".superpowers/notes.md": b"notes\n",
    "fundamentals.py": b"Y = 2\n",
    "tests/test_old.py": b"def test_old():\n    pass\n",
    "tests/es/__pycache__/test_helper.cpython-312.pyc": b"\x00\x01",
}
FAKE_DATA = ("v2/NODE_MAP.csv", "cache/common_panel_groups.json")
FAKE_ENTRY = ("es.sealed_run",)
NEW_FILES = {"README.md", "requirements.txt", "LICENSE", "MANIFEST.sha256"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_tree(root, extra=None, drop=()):
    root = Path(root)
    for rel, data in {**INCLUDED, **UNRELATED, **(extra or {})}.items():
        if rel in drop:
            continue
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return root


def snapshot(root):
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted(Path(root).rglob("*")) if p.is_file()}


def files_of(root):
    return {p.relative_to(root).as_posix() for p in Path(root).rglob("*") if p.is_file()}


def stage(tmp_path, **kwargs):
    from es.public_package import build_package
    dev = kwargs.pop("dev", None) or make_tree(tmp_path / "dev")
    out = tmp_path / "pkg"
    kwargs.setdefault("entry_points", FAKE_ENTRY)
    kwargs.setdefault("data", FAKE_DATA)
    return dev, out, build_package(dev, out, **kwargs)


# ---------------------------------------------------------------------------------------------------------------
# The fake docs/sealed and outputs/es/sealed. No test of this file reads the real ones: the folders
# are always passed in, or found beside the fake dev tree.
# ---------------------------------------------------------------------------------------------------------------
#: Odd bytes again: CRLF, a byte that is not UTF-8, no final newline.
PROTOCOL_MD = b"# Protocol\r\nThe bound files are hashed.\xff\r\nno final newline"
SEALED_FILES = {                                    # the other protocol files: copied whenever they exist
    "ENV_PINS.json": b'{"executable": "python", "versions": {}}\n',
    "KNOWLEDGE_STATEMENT.md": b"# Knowledge statement\r\n- [x] none\n",
    "protocol_inputs.json": b'{"stage": "complete"}',
    "pre_pull_cutoffs.csv": b"analysis,block\r\nSA,G1_24\r\n",
    "code_archive.tar.gz": b"\x1f\x8b\x08\x00 not an archive \xff\xfe",
}
SEALED_NOT_COPIED = {"KNOWLEDGE_STATEMENT_TEMPLATE.md": b"template\n", "REGISTRATION.template.md": b"template\n",
                     "PULL_A.lock": b"{}", "code.bundle": b"bundle", "notes.txt": b"notes\n"}
CODE_MANIFEST_FILES = ("es/sealed_run.py", "es/helper.py", "v2/x.py", "v2/NODE_MAP.csv", "cache/common_panel_groups.json")
COPIED_PROTOCOL = {"PROTOCOL.md", "PROTOCOL.sha256", "CODE_MANIFEST.sha256", *SEALED_FILES}
RECORD_NAMES = ("REGISTRATION.md", "PULLS.json", "RUNS.json", "RUN_A.claim", "RUN_B.claim")
RECORD_TIME = "2026-10-26T13:05:00Z"
RECORD_LINES = {"local": "local_record: author-statement-2026-10-26",
                "public": "public_record: https://github.com/someone/dart-spread/releases/tag/v1.0",
                "osf": "osf_doi: 10.17605/OSF.IO/ABCDE"}
RUN_A = {"started_utc": "2026-10-26T14:00:00+00:00", "finished_utc": "2026-10-26T20:30:00+00:00", "status": "completed",
         "out_dir": "/x/outputs/es/sealed/A"}
RUN_B = {"started_utc": "2026-11-02T09:00:00+00:00", "out_dir": "/x/outputs/es/sealed/B"}
OUTPUT_FILES = {
    "results.json": b'{"status": "completed", "t_hat": 0.123456}\n',
    "RUN_LOG.txt": b"started\r\nfinished\n",
    "MANIFEST.sha256": b"0" * 64 + b"  results.json\n",
    "tables/coverage.csv": b"market,node\r\nNYISO,WEST\r\n",
    "tables/economics.csv": b"a,b\n1,2\n",
    "kept/P1.parquet": b"PAR1",
    "scores/s.parquet": b"PAR1",
    ".DS_Store": b"\x00\x01",
    "stray.pyc": b"\x00\x01",                               # a compiled file outside any cache folder
    "__pycache__/notes.txt": b"anything in a cache folder\n",   # and a file of another kind inside one
}
COPIED_OUTPUTS = tuple(rel for rel in OUTPUT_FILES if not rel.endswith((".parquet", ".pyc")) and rel != ".DS_Store"
                       and not rel.startswith("__pycache__/"))
LOCAL_SENTENCE = ("The record is a local statement that an AI assistant wrote on the author's instruction; the author did not write or sign "
                  "it. No registry, repository or other third party held the protocol before the data of the window were pulled, so for the "
                  "order of events a reader has only that record and the logs of the pull and the run. The protocol is published here with "
                  "the code after the run.")
PULLS_CLAUSE = "The pull of each window is recorded in `PULLS.json` with its start and finish times."
PULL_LOGS_SENTENCE = ("The logs of the pulls are in `protocol/sealed_pulls/`: `PULL_LOG.txt` gives the start and finish times, the record the "
                      "pull ran under and the days that were excused; `COVERAGE.csv` and `OVERLAP_CHECK.csv` give the coverage of each node and the "
                      "check of the first days against the files already opened.")
#: The four small files of a pull folder that are copied (odd bytes again), and what else such a folder holds, which never is.
PULL_FILES = {
    "PULL_LOG.txt": b"pull_started_utc: 2026-10-26T13:40:00+00:00\r\nregistration_id: local:x\r\nstep done \xff\r\n",
    "COVERAGE.csv": b"series,market,node,hours\r\nNYISO,NYISO,WEST,2208\r\n",
    "OVERLAP_CHECK.csv": b"series,first_day,ok\nNYISO,2026-07-01,True",
    "MANIFEST.sha256": b"0" * 64 + b"  raw/NYISO/x.csv\n",
}
PULL_NEVER = {".state.json": b'{"done": []}', "raw/NYISO/x.csv": b"a,b\n1,2\n", "logs/y.txt": b"log\n", "fmm/z.parquet": b"PAR1",
              "nyiso_rtint/w.parquet": b"PAR1", "stray.parquet": b"PAR1", "notes.txt": b"notes\n", "extra.csv": b"a\n1\n"}


def make_pull(root, drop=(), extra=None):
    """A fake pull folder: the four files that are copied and the ones that never are."""
    root = Path(root)
    for rel, data in {**PULL_FILES, **PULL_NEVER, **(extra or {})}.items():
        if rel not in drop:
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    return root


def pulled_files(out):
    """The paths under protocol/sealed_pulls/ of a package folder."""
    return {p.relative_to(out).as_posix() for p in (out / "protocol" / "sealed_pulls").rglob("*") if p.is_file()} \
        if (out / "protocol" / "sealed_pulls").exists() else set()


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def code_manifest_text(rels=CODE_MANIFEST_FILES, changed=()):
    """CODE_MANIFEST.sha256 of the fake tree: the hash of each file's bytes (of other bytes for the `changed` ones)."""
    return "".join(f"{sha_bytes(b'other' if rel in changed else INCLUDED[rel])}  {rel}\n" for rel in rels)


def registration_text(sha, kind="local", when=RECORD_TIME):
    return f"# Record of the protocol\nprotocol_sha256: {sha}\nregistered_utc: {when}\n{RECORD_LINES[kind]}\n"


def make_sealed(root, *, record="local", runs=None, claims=(), pulls=False, manifest_changed=(), drop=(), extra=None):
    """A fake docs/sealed: the protocol with its bound files, the files that are never copied, and with ``record`` ("local", "public",
    "osf" or None) the record of the protocol, ``runs`` (RUNS.json), ``claims`` (RUN_<test>.claim) and ``pulls`` (PULLS.json: True for an entry
    without a root, or a dict of entries)."""
    root = Path(root)
    files = {"PROTOCOL.md": PROTOCOL_MD, "PROTOCOL.sha256": f"{sha_bytes(PROTOCOL_MD)}  PROTOCOL.md\n".encode(),
             "CODE_MANIFEST.sha256": code_manifest_text(changed=manifest_changed).encode(), **SEALED_FILES, **SEALED_NOT_COPIED}
    if record:
        files["REGISTRATION.md"] = registration_text(sha_bytes(PROTOCOL_MD), record).encode()
    if runs is not None:
        files["RUNS.json"] = json.dumps(runs, indent=2).encode()
    for test in claims:
        files[f"RUN_{test}.claim"] = json.dumps({"test": test, "started_utc": RUN_A["started_utc"], "out_dir": "/x/outputs",
                                                 "pid": 1, "nonce": "n"}).encode()
    if pulls:
        files["PULLS.json"] = (json.dumps(pulls, indent=2) + "\n").encode() if isinstance(pulls, dict) else b'{"A": {"complete": true}}\n'
    files.update(extra or {})
    root.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        if name not in drop:
            (root / name).write_bytes(data)
    return root


def make_outputs(root, tests=("A",), files=None):
    root = Path(root)
    for test in tests:
        for rel, data in (OUTPUT_FILES if files is None else files).items():
            path = root / test / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    return root


def stage_protocol(tmp_path, sealed, outs=None, **kwargs):
    """stage() with the sealed folders passed in (no sealed outputs unless ``outs`` is given)."""
    return stage(tmp_path, sealed_dir=sealed, sealed_outputs_dir=tmp_path / "no_outputs" if outs is None else outs, **kwargs)


def readme_for(tmp_path, record="local", outs=None, pull_dirs=None, **make):
    """(README text, package folder, Staged) of a package built from a fake docs/sealed."""
    sealed = make_sealed(tmp_path / "sealed", record=record, **make)
    _, out, staged = stage_protocol(tmp_path, sealed, outs, pull_dirs=pull_dirs)
    return (out / "README.md").read_text(), out, staged


# ---------------------------------------------------------------------------------------------------------------
# The file list
# ---------------------------------------------------------------------------------------------------------------
def test_file_list_is_the_closure_plus_tests_config_and_data(tmp_path):
    from es.public_package import check_plan, plan_files
    dev = make_tree(tmp_path / "dev")
    plan = plan_files(dev, entry_points=FAKE_ENTRY, data=FAKE_DATA)
    assert list(plan.files) == sorted(INCLUDED)
    assert plan.modules == {"es": "es/__init__.py", "es.helper": "es/helper.py", "es.sealed_run": "es/sealed_run.py",
                            "v2": "v2/__init__.py", "v2.x": "v2/x.py"}
    assert plan.tests == ("tests/es/conftest.py", "tests/es/test_helper.py")      # the pyc under __pycache__ is not a file
    assert plan.config == ("pytest.ini",)
    assert plan.data == tuple(sorted(FAKE_DATA))
    assert not set(UNRELATED) & set(plan.files)
    assert check_plan(plan) == []                                  # cache/common_panel_groups.json is the one JSON allowed from cache/


def test_the_list_is_computed_from_the_entry_points_not_from_the_folder(tmp_path):
    from es.public_package import plan_files
    dev = make_tree(tmp_path / "dev", extra={"es/extra.py": b"Z = 3\n"})
    assert "es/extra.py" not in plan_files(dev, entry_points=FAKE_ENTRY, data=FAKE_DATA).files
    dev2 = make_tree(tmp_path / "dev2", extra={"es/sealed_run.py": b"from . import helper, extra\n", "es/extra.py": b"Z = 3\n"})
    assert "es/extra.py" in plan_files(dev2, entry_points=FAKE_ENTRY, data=FAKE_DATA).files


# ---------------------------------------------------------------------------------------------------------------
# Refusals: nothing is written, and the message names the file and the rule
# ---------------------------------------------------------------------------------------------------------------
def _refusal_cases(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "linked.py").write_bytes(b"L = 1\n")
    (tmp_path / "outside.csv").write_bytes(b"a\n")
    return {
        "a module named like AFD, imported by the closure":
            dict(extra={"es/helper.py": b"from . import AFD_tools\n", "es/AFD_tools.py": b"A = 1\n"},
                 want=("es/AFD_tools.py", "afd")),
        "afd inside a module name":
            dict(extra={"es/helper.py": b"from . import my_afd_helpers\n", "es/my_afd_helpers.py": b"A = 1\n"},
                 want=("es/my_afd_helpers.py", "afd")),
        "the closure reaches src/afd.py":
            dict(extra={"es/helper.py": b"from src import afd\n", "src/__init__.py": b""}, want=("src/afd.py", "afd")),
        "a listed data file is a notebook":
            dict(extra={"notebooks/analysis.ipynb": b"{}"}, data=FAKE_DATA + ("notebooks/analysis.ipynb",),
                 want=("notebooks/analysis.ipynb", ".ipynb")),
        "a listed data file is CLAUDE.md":
            dict(data=FAKE_DATA + ("CLAUDE.md",), want=("CLAUDE.md",)),
        "a listed data file is under .superpowers":
            dict(data=FAKE_DATA + (".superpowers/notes.md",), want=(".superpowers/notes.md", ".superpowers")),
        "a listed data file is under outputs/":
            dict(data=FAKE_DATA + ("outputs/es/result.csv",), want=("outputs/es/result.csv", "outputs/")),
        "a listed data file is a parquet":
            dict(data=FAKE_DATA + ("cache/other.parquet",), want=("cache/other.parquet", ".parquet")),
        "a listed data file is cache/ but not the listed JSON":
            dict(data=FAKE_DATA + ("cache/other.json",), want=("cache/other.json", "cache/")),
        "a listed data file is outside the dev tree":
            dict(data=FAKE_DATA + ("../outside.csv",), want=("../outside.csv", "outside")),
        "a module of the closure is a link to a file outside the dev tree":
            dict(extra={"es/helper.py": b"from . import linked\n"}, links={"es/linked.py": outside / "linked.py"},
                 want=("es/linked.py", "outside")),
        "a listed data file is missing":
            dict(data=FAKE_DATA + ("v2/MISSING.csv",), want=("v2/MISSING.csv", "missing")),
        "a file of tests/es matches an exclusion (it is refused, not skipped)":
            dict(extra={"tests/es/test_afd_thing.py": b"def test_a():\n    pass\n"}, want=("tests/es/test_afd_thing.py", "afd")),
        "CLAUDE.md inside tests/es":
            dict(extra={"tests/es/CLAUDE.md": b"x\n"}, want=("tests/es/CLAUDE.md",)),
        "an entry point that is not a module":
            dict(entry=("es.sealed_run", "es.nope"), want=("es.nope",)),
    }


REFUSAL_CASES = (
    "a module named like AFD, imported by the closure", "afd inside a module name", "the closure reaches src/afd.py",
    "a listed data file is a notebook", "a listed data file is CLAUDE.md", "a listed data file is under .superpowers",
    "a listed data file is under outputs/", "a listed data file is a parquet",
    "a listed data file is cache/ but not the listed JSON", "a listed data file is outside the dev tree",
    "a module of the closure is a link to a file outside the dev tree", "a listed data file is missing",
    "a file of tests/es matches an exclusion (it is refused, not skipped)", "CLAUDE.md inside tests/es",
    "an entry point that is not a module")


def test_every_refusal_case_is_run(tmp_path):
    assert set(REFUSAL_CASES) == set(_refusal_cases(tmp_path))


@pytest.mark.parametrize("case", REFUSAL_CASES)
def test_the_build_refuses_and_writes_nothing(tmp_path, case):
    from es.public_package import PackageRefused, build_package
    spec = _refusal_cases(tmp_path)[case]
    dev = make_tree(tmp_path / "dev", extra=spec.get("extra"))
    for rel, target in spec.get("links", {}).items():
        (dev / rel).unlink(missing_ok=True)
        (dev / rel).symlink_to(target)
    before, out = snapshot(dev), tmp_path / "pkg"
    with pytest.raises(PackageRefused) as err:
        build_package(dev, out, entry_points=spec.get("entry", FAKE_ENTRY), data=spec.get("data", FAKE_DATA))
    message = str(err.value)
    for needle in spec["want"]:
        assert needle.lower() in message.lower(), (needle, message)
    assert not out.exists() and not out.with_name(out.name + ".partial").exists()
    assert snapshot(dev) == before


def test_every_problem_is_listed_not_only_the_first(tmp_path):
    from es.public_package import PackageRefused, build_package
    dev = make_tree(tmp_path / "dev")
    with pytest.raises(PackageRefused) as err:
        build_package(dev, tmp_path / "pkg", entry_points=FAKE_ENTRY,
                      data=FAKE_DATA + ("CLAUDE.md", "outputs/es/result.csv", "cache/other.parquet"))
    assert len(err.value.reasons) == 3
    assert all(r in str(err.value) for r in ("CLAUDE.md", "outputs/es/result.csv", "cache/other.parquet"))


def test_exclusion_rules_in_one_place():
    from es.public_package import exclusion_reason
    cache_json = {"cache/common_panel_groups.json"}
    cases = {
        "es/x.py": None, "tests/es/test_x.py": None, "src/cache.py": None, "es/outputs.py": None, "v2/NODE_MAP.csv": None,
        "cache/common_panel_groups.json": None,
        "a/b.ipynb": ".ipynb", "AFD_notes.md": "afd", "src/afd_features.py": "afd", "x/Afd/y.py": "afd", "a/CLAUDE.md": "CLAUDE.md",
        "a/.superpowers/b.md": ".superpowers", "outputs/es/a.csv": "outputs/", "a/outputs/b.csv": "outputs/",
        "x.parquet": ".parquet", "cache/y.json": "cache/", "cache/sub/common_panel_groups.json": "cache/",
    }
    for rel, expected in cases.items():
        got = exclusion_reason(rel, cache_json)
        assert (got is None) if expected is None else (got is not None and expected.lower() in got.lower()), (rel, got)


# ---------------------------------------------------------------------------------------------------------------
# Byte-identical copies, and nothing else
# ---------------------------------------------------------------------------------------------------------------
def test_copies_are_byte_identical_and_only_listed_files_are_copied(tmp_path):
    dev = make_tree(tmp_path / "dev")
    before = snapshot(dev)
    dev, out, staged = stage(tmp_path, dev=dev)
    for rel, data in INCLUDED.items():
        assert (out / rel).read_bytes() == data, rel               # CRLF, tabs, trailing blanks and the non-UTF-8 byte survive
    assert files_of(out) == set(INCLUDED) | NEW_FILES
    assert not [p for p in out.rglob("*") if "__pycache__" in p.parts or p.suffix == ".pyc"]
    assert snapshot(dev) == before                                  # the source tree is not touched
    assert staged.n_files == len(INCLUDED) + len(NEW_FILES)
    assert staged.out_dir == out


def test_a_source_that_changes_during_the_build_is_refused(tmp_path, monkeypatch):
    import es.public_package as pp
    dev = make_tree(tmp_path / "dev")
    real_copy = pp.shutil.copyfile
    calls = []

    def copy_then_edit(src, dst, *a, **k):
        out = real_copy(src, dst, *a, **k)
        if str(src).endswith("es/helper.py") and not calls:
            calls.append(1)
            Path(src).write_bytes(Path(src).read_bytes() + b"# edited by another process\n")
        return out

    monkeypatch.setattr(pp.shutil, "copyfile", copy_then_edit)
    with pytest.raises(RuntimeError, match="changed while"):
        pp.build_package(dev, tmp_path / "pkg", entry_points=FAKE_ENTRY, data=FAKE_DATA)
    assert not (tmp_path / "pkg").exists() and not (tmp_path / "pkg.partial").exists()


def test_the_manifest_lists_every_file_but_itself_and_detects_changes(tmp_path):
    from es.public_package import verify_manifest
    _, out, _ = stage(tmp_path)
    lines = (out / "MANIFEST.sha256").read_text().splitlines()
    rels = [ln.split("  ", 1)[1] for ln in lines]
    assert rels == sorted(rels) and set(rels) == set(INCLUDED) | (NEW_FILES - {"MANIFEST.sha256"})
    assert all(ln == f"{sha(out / ln.split('  ', 1)[1])}  {ln.split('  ', 1)[1]}" for ln in lines)
    assert verify_manifest(out) == []
    (out / "es" / "helper.py").write_bytes(b"changed")
    (out / "es" / "__pycache__").mkdir()
    (out / "es" / "__pycache__" / "x.pyc").write_bytes(b"\x00")
    (out / "v2" / "x.py").unlink()
    problems = " | ".join(verify_manifest(out))
    assert "es/helper.py" in problems and "differs" in problems
    assert "es/__pycache__/x.pyc" in problems and "not in the manifest" in problems
    assert "v2/x.py" in problems and "missing" in problems


def test_an_import_the_list_does_not_cover_is_found_in_the_package(tmp_path):
    from es.public_package import closure_problems, plan_files
    dev = make_tree(tmp_path / "dev")
    plan = plan_files(dev, entry_points=FAKE_ENTRY, data=FAKE_DATA)
    _, out, _ = stage(tmp_path, dev=dev)
    assert closure_problems(out, FAKE_ENTRY, plan) == []
    (out / "v2" / "x.py").unlink()
    assert any("v2/x.py" in p for p in closure_problems(out, FAKE_ENTRY, plan))


def test_the_package_dir_rules(tmp_path):
    from es.public_package import PackageRefused, build_package
    dev = make_tree(tmp_path / "dev")
    kw = dict(entry_points=FAKE_ENTRY, data=FAKE_DATA)
    with pytest.raises(PackageRefused, match="inside the dev tree"):
        build_package(dev, dev / "pkg", **kw)
    out = tmp_path / "pkg"
    build_package(dev, out, **kw)
    with pytest.raises(PackageRefused, match="already exists"):
        build_package(dev, out, **kw)
    (dev / "es" / "helper.py").write_bytes(b"HELPER = 2\n")
    build_package(dev, out, rebuild=True, **kw)
    assert (out / "es" / "helper.py").read_bytes() == b"HELPER = 2\n"
    assert not (tmp_path / "pkg.partial").exists()
    stranger = tmp_path / "stranger"                                # a folder that is not a previous build is never deleted
    stranger.mkdir()
    (stranger / "keep.txt").write_text("mine")
    with pytest.raises(PackageRefused, match="not a previous build"):
        build_package(dev, stranger, rebuild=True, **kw)
    assert (stranger / "keep.txt").read_text() == "mine"


def test_a_rebuild_never_deletes_a_git_repository_and_verification_ignores_one(tmp_path):
    from es.public_package import PackageRefused, build_package, verify_manifest
    dev = make_tree(tmp_path / "dev")
    out = tmp_path / "pkg"
    kw = dict(entry_points=FAKE_ENTRY, data=FAKE_DATA)
    build_package(dev, out, **kw)
    (out / ".git").mkdir()
    (out / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    assert verify_manifest(out) == []                                # a repository's own files are not package files
    with pytest.raises(PackageRefused, match=r"\.git"):
        build_package(dev, out, rebuild=True, **kw)
    assert (out / ".git" / "HEAD").exists()


# ---------------------------------------------------------------------------------------------------------------
# The four files the package gets
# ---------------------------------------------------------------------------------------------------------------
def test_license_is_mit_for_the_author_in_2026(tmp_path):
    from es.public_package import license_text
    text = license_text()
    assert text.startswith("MIT License\n\nCopyright (c) 2026 Erce Erez\n")
    assert "Permission is hereby granted, free of charge" in text and "THE SOFTWARE IS PROVIDED \"AS IS\"" in text
    _, out, _ = stage(tmp_path)
    assert (out / "LICENSE").read_text() == text


def test_environment_is_read_live():
    from es.public_package import CORE_LIBRARIES, environment
    env = environment()
    assert CORE_LIBRARIES == ("numpy", "pandas", "pyarrow", "xgboost", "scikit-learn", "scipy", "statsmodels", "matplotlib")
    assert env["libraries"] == {name: metadata.version(name) for name in CORE_LIBRARIES}
    import platform
    assert env["python"] == platform.python_version()


def test_requirements_pin_the_live_versions_and_list_what_else_is_imported(tmp_path):
    from es.public_package import CORE_LIBRARIES, environment
    env = environment()
    dev = make_tree(tmp_path / "dev", extra={
        "es/helper.py": b"import threadpoolctl\nimport numpy\n",                   # a runtime import beyond the core
        "tests/es/test_helper.py": b"import pytest\nimport numpy\n\ndef test_x():\n    pass\n"})   # a test-only import
    _, out, staged = stage(tmp_path, dev=dev)
    text = (out / "requirements.txt").read_text()
    pins = dict(re.findall(r"^([A-Za-z0-9_.\-]+)==(\S+)", text, flags=re.M))
    for name in CORE_LIBRARIES:
        assert pins[name] == env["libraries"][name] == metadata.version(name)
    assert pins["threadpoolctl"] == metadata.version("threadpoolctl")
    assert pins["pytest"] == metadata.version("pytest")
    assert f"Python {env['python']}" in text
    runtime, tests_only = text.split("only to run the tests")
    assert "threadpoolctl==" in runtime and "pytest==" in tests_only and "pytest==" not in runtime
    assert staged.requirements == pins


def test_requirements_without_an_installed_package_say_so_instead_of_inventing_a_version(tmp_path):
    from es.public_package import requirements_text
    text = requirements_text({"python": "3.12.7", "libraries": {"numpy": "1.0"}}, runtime={"notinstalledpkg": ["es/a.py"]},
                             tests_only={}, installed=lambda name: None)
    assert "numpy==1.0" in text and "notinstalledpkg  # NOT INSTALLED here: pin it yourself" in text


def test_readme_states_what_the_spec_asks_for(tmp_path):
    from es.public_package import CORE_LIBRARIES, environment
    env = environment()
    _, out, _ = stage(tmp_path, tests_note="TESTS NOTE SENTENCE")
    text = (out / "README.md").read_text()
    assert "Feature-Block Portability and Operating-Point Drift in DART-Spread Spike Prediction" in text
    assert "Erce Erez" in text and "sole author" in text
    assert "(DOI to be added)" in text                                          # the paper has no DOI yet; this line stays
    for name in CORE_LIBRARIES:
        assert env["libraries"][name] in text
    assert env["python"] in text
    assert "TESTS NOTE SENTENCE" in text
    assert "mis.nyiso.com" in text and "oasis.caiso.com" in text and "docs.misoenergy.org" in text and "iso-ne.com" in text
    assert "MANIFEST.sha256" in text and "CODE_MANIFEST.sha256" in text
    assert "price-only" in text
    assert "DART_PROJECT_ROOT" in text and "DART_Spike_v2_Clean" in text and "dart_dev" in text and "EnergySystems_Revision" in text


#: Ruling 133: no registration was made. The README is published as it is, so it must not say that a registration exists or will exist.
NO_REGISTRATION_PARAGRAPH = (
    "No registration was made. The sealed windows (Test A: 1 July to 30 September 2026; Test B: 1 October to 31 December 2026) have not been "
    "opened. The protocol files are not part of this repository. Because no registration record exists, the sealed pull and the sealed run "
    "refuse to run. The machinery is included as it stands, because the protocol fixes it.")


def claims_a_registration(text):
    """What a README must not hold: a line or word that says a registration exists, is pending or will be made, or that the copy is a staging
    copy that is to be rebuilt first. Returns the offending pieces; the technical names (es.registration, REGISTRATION.md, the
    registration record that does not exist) are not claims."""
    flat = " ".join(text.split())
    found = [m.group(0) for m in re.finditer(
        r"OSF registration:|\bOSF\b|[Pp]re-?regist\w*|\bregistered\b|\bregistration (?:DOI|identifier|is public|date)\b|[Ss]taging copy|<to be filled>"
        r"|rebuilt after registration|after the registration|only after the registration|registers `?PROTOCOL", flat)]
    stripped = flat.replace(NO_REGISTRATION_PARAGRAPH, " ")
    found += [m.group(0) for m in re.finditer(r"[Nn]o registration|registration was made", stripped)]    # said once, in the paragraph, and nowhere else
    return found


def test_readme_says_that_no_registration_was_made_and_claims_none(tmp_path):
    _, out, _ = stage(tmp_path, tests_note="TESTS NOTE SENTENCE")
    text = (out / "README.md").read_text()
    flat = " ".join(text.split())
    assert claims_a_registration(text) == [], claims_a_registration(text)
    section = flat.split("## How the protocol binds the code", 1)[1]
    assert section.startswith(" " + NO_REGISTRATION_PARAGRAPH + " The sealed tests are bound to the code by content:")      # at the top of that section
    assert flat.count("No registration was made.") == 1
    assert "preregist" not in text.lower() and "OSF" not in text and "Staging copy" not in text
    for line in text.splitlines():
        assert not line.startswith("OSF registration:"), line
    assert "pre-specified sealed tests" in flat and "Sealed tests (not run; see the next section):" in flat
    assert "How the preregistration binds the code" not in text and "## How the protocol binds the code" in text
    bullets = [line for line in text.splitlines() if line.startswith("- `es.registration.verify_registration`")]
    assert len(bullets) == 1 and "only when `W/docs/sealed/REGISTRATION.md` exists and names the SHA-256 of the unchanged `PROTOCOL.md`" in bullets[0]
    assert "OSF" not in bullets[0] and "registration" not in bullets[0].replace("verify_registration", "").replace("es.registration", "")
    departures = [line for line in text.splitlines() if line.startswith("- `es.sealed_run.code_departures(manifest, root)`")]
    assert len(departures) == 1 and "differ from the manifest of the protocol" in departures[0] and "registered" not in departures[0]
    assert "Cite the paper: Erce Erez," in text and text.rstrip().endswith("(DOI to be added).")                                  # this line stays
    assert [line for line in text.splitlines() if line.startswith("## ")] == [          # no section dropped or added; one renamed
        "## What this is", "## Environment", "## Layout and setup", "## Where the data come from", "## Run order",
        "## How the protocol binds the code", "## Tests", "## Verifying this package", "## Licence and citation"]
    assert "TESTS NOTE SENTENCE" in text and "<to be filled>" not in text


def test_the_readme_check_fails_when_the_readme_claims_a_registration(tmp_path):
    """The check above is not blind: put back each claim the old README made, or add the ones a registration would bring, and it finds them."""
    _, out, _ = stage(tmp_path)
    text = (out / "README.md").read_text()
    assert claims_a_registration(text) == []
    claims = [
        "OSF registration: <to be filled>\n", "OSF registration: 10.17605/OSF.IO/ABCDE\n",
        "**Staging copy, assembled 2026-10-25. It must be rebuilt after registration**, so that it carries the final version.\n",
        "the machinery of the two preregistered sealed tests\n", "the machinery of the two pre-registered sealed tests\n",
        "## How the preregistration binds the code\n", "Sealed tests (only after the registration):\n",
        "`PROTOCOL.md`, which is what is registered on OSF, prints the SHA-256\n",
        "The author registers `PROTOCOL.md` on OSF and writes `W/docs/sealed/REGISTRATION.md`.\n",
        "the files that differ from the registered manifest\n", "The registration DOI is 10.17605/OSF.IO/ABCDE.\n",
        "The registration is public.\n", "The registration identifier is abcde.\n", "The protocol was registered on 2026-10-27.\n"]
    for claim in claims:
        assert claims_a_registration(text + "\n" + claim), claim
    assert claims_a_registration(text.replace("No registration was made.", "The protocol is registered.")) != []
    assert claims_a_registration(text + "\nNo registration was made. We will register it later.\n") != []          # the sentence only once, in its place


# ---------------------------------------------------------------------------------------------------------------
# The protocol/ folder: the protocol, the records and the sealed outputs
# ---------------------------------------------------------------------------------------------------------------
def test_the_protocol_files_and_the_records_are_copied_byte_for_byte_and_listed_in_the_manifest(tmp_path):
    from es.public_package import closure_problems, plan_files, verify_manifest
    dev = make_tree(tmp_path / "dev")
    sealed = make_sealed(tmp_path / "sealed", record="local", runs={"A": RUN_A, "B": RUN_B}, claims=("A", "B"), pulls=True)
    before = (snapshot(dev), snapshot(sealed))
    _, out, staged = stage_protocol(tmp_path, sealed, dev=dev)
    names = COPIED_PROTOCOL | set(RECORD_NAMES)
    assert len(names) == 13
    assert files_of(out) == set(INCLUDED) | NEW_FILES | {f"protocol/{n}" for n in names}
    for name in names:
        assert (out / "protocol" / name).read_bytes() == (sealed / name).read_bytes(), name        # CRLF and the odd byte survive
    listed = {rel: digest for digest, rel in (ln.split("  ", 1) for ln in (out / "MANIFEST.sha256").read_text().splitlines())}
    assert all(listed[f"protocol/{n}"] == sha(sealed / n) for n in names)
    assert verify_manifest(out) == []
    plan = plan_files(dev, entry_points=FAKE_ENTRY, data=FAKE_DATA)
    assert not [f for f in plan.files if f.startswith("protocol/")]                      # data, never code of the closure
    assert closure_problems(out, FAKE_ENTRY, plan) == []
    assert (snapshot(dev), snapshot(sealed)) == before                                     # the sources are not touched
    assert staged.n_files == len(files_of(out))
    for name in SEALED_NOT_COPIED:                                                         # templates, locks, bundles: not copied
        assert not (out / "protocol" / name).exists(), name


def test_only_the_protocol_files_that_exist_are_copied(tmp_path):
    sealed = make_sealed(tmp_path / "sealed", record=None, drop=("ENV_PINS.json", "KNOWLEDGE_STATEMENT.md", "code_archive.tar.gz"))
    _, out, _ = stage_protocol(tmp_path, sealed)
    assert files_of(out) == set(INCLUDED) | NEW_FILES | {
        f"protocol/{n}" for n in ("PROTOCOL.md", "PROTOCOL.sha256", "CODE_MANIFEST.sha256", "protocol_inputs.json", "pre_pull_cutoffs.csv")}


def test_the_sealed_outputs_of_a_test_that_has_a_folder_are_copied_as_they_are_but_no_parquet(tmp_path):
    from es.public_package import verify_manifest
    sealed = make_sealed(tmp_path / "sealed", record="local", runs={"A": RUN_A}, claims=("A",))
    outs = make_outputs(tmp_path / "outs", tests=("A",))                                   # a folder for Test A, none for Test B
    before = snapshot(outs)
    _, out, _ = stage_protocol(tmp_path, sealed, outs)
    base = out / "protocol" / "sealed_outputs"
    assert {p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file()} == {f"A/{rel}" for rel in COPIED_OUTPUTS}
    for rel in COPIED_OUTPUTS:
        assert (base / "A" / rel).read_bytes() == OUTPUT_FILES[rel], rel
    assert not list(out.rglob("*.parquet")) and not list(out.rglob(".DS_Store")) and not list(out.rglob("*.pyc"))
    assert not list(out.rglob("__pycache__"))
    assert not (base / "B").exists()
    listed = {ln.split("  ", 1)[1] for ln in (out / "MANIFEST.sha256").read_text().splitlines()}
    assert {f"protocol/sealed_outputs/A/{rel}" for rel in COPIED_OUTPUTS} <= listed
    assert verify_manifest(out) == []
    assert snapshot(outs) == before


def test_the_sealed_outputs_of_both_tests_are_copied_when_both_folders_exist(tmp_path):
    sealed = make_sealed(tmp_path / "sealed", record="local", runs={"A": RUN_A, "B": RUN_B}, claims=("A", "B"))
    outs = make_outputs(tmp_path / "outs", tests=("A", "B"))
    _, out, _ = stage_protocol(tmp_path, sealed, outs)
    base = out / "protocol" / "sealed_outputs"
    assert {p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file()} == {
        f"{t}/{rel}" for t in "AB" for rel in COPIED_OUTPUTS}


def _protocol_refusals():
    other = "f" * 64
    sha_md = sha_bytes(PROTOCOL_MD)
    return {
        "PROTOCOL.md does not hash to the line of PROTOCOL.sha256":
            dict(sealed=dict(record=None, extra={"PROTOCOL.md": PROTOCOL_MD + b" edited"}), want=("PROTOCOL.md", "PROTOCOL.sha256")),
        "PROTOCOL.sha256 is missing":
            dict(sealed=dict(record=None, drop=("PROTOCOL.sha256",)), want=("PROTOCOL.sha256", "missing")),
        "PROTOCOL.sha256 holds no hash":
            dict(sealed=dict(record=None, extra={"PROTOCOL.sha256": b"not a hash  PROTOCOL.md\n"}), want=("PROTOCOL.sha256",)),
        "the record names another hash":
            dict(sealed=dict(extra={"REGISTRATION.md": registration_text(other).encode()}), want=("REGISTRATION.md", "protocol_sha256")),
        "the record names two kinds":
            dict(sealed=dict(extra={"REGISTRATION.md": (registration_text(sha_md) + RECORD_LINES["osf"] + "\n").encode()}),
                 want=("REGISTRATION.md", "more than one kind")),
        "the record still holds a placeholder":
            dict(sealed=dict(extra={"REGISTRATION.md": registration_text(sha_md).replace("author-statement-2026-10-26", "<date>").encode()}),
                 want=("REGISTRATION.md", "placeholder")),
        "the record has no time":
            dict(sealed=dict(extra={"REGISTRATION.md": f"protocol_sha256: {sha_md}\n{RECORD_LINES['local']}\n".encode()}),
                 want=("REGISTRATION.md", "registered_utc")),
        "RUN_A.claim without a record": dict(sealed=dict(record=None, claims=("A",)), want=("RUN_A.claim", "REGISTRATION.md")),
        "RUN_B.claim without a record": dict(sealed=dict(record=None, claims=("B",)), want=("RUN_B.claim", "REGISTRATION.md")),
        "RUNS.json without a record": dict(sealed=dict(record=None, runs={"A": RUN_A}), want=("RUNS.json", "REGISTRATION.md")),
        "PULLS.json without a record": dict(sealed=dict(record=None, pulls=True), want=("PULLS.json", "REGISTRATION.md")),
        "sealed outputs without a record": dict(sealed=dict(record=None), outputs=("A",), want=("outs", "REGISTRATION.md")),
        "sealed outputs of a test that has neither a record nor a claim":
            dict(sealed=dict(runs={}), outputs=("A",), want=("outs", "RUN_A.claim", "RUNS.json")),
        "RUNS.json is not JSON": dict(sealed=dict(extra={"RUNS.json": b"{not json"}), want=("RUNS.json", "JSON")),
        "RUNS.json is not an object": dict(sealed=dict(extra={"RUNS.json": b"[1, 2]"}), want=("RUNS.json", "object")),
        "CODE_MANIFEST.sha256 is missing":
            dict(sealed=dict(record=None, drop=("CODE_MANIFEST.sha256",)), want=("CODE_MANIFEST.sha256", "missing")),
        "CODE_MANIFEST.sha256 holds a line that is not sha256 and path":
            dict(sealed=dict(record=None, extra={"CODE_MANIFEST.sha256": b"es/x.py\n"}), want=("CODE_MANIFEST.sha256", "line 1")),
        "a record, and no PROTOCOL.md": dict(sealed=dict(drop=("PROTOCOL.md",)), want=("REGISTRATION.md", "PROTOCOL.md")),
    }


PROTOCOL_REFUSAL_CASES = (
    "PROTOCOL.md does not hash to the line of PROTOCOL.sha256", "PROTOCOL.sha256 is missing", "PROTOCOL.sha256 holds no hash",
    "the record names another hash", "the record names two kinds", "the record still holds a placeholder", "the record has no time",
    "RUN_A.claim without a record", "RUN_B.claim without a record", "RUNS.json without a record", "PULLS.json without a record",
    "sealed outputs without a record", "sealed outputs of a test that has neither a record nor a claim", "RUNS.json is not JSON",
    "RUNS.json is not an object", "CODE_MANIFEST.sha256 is missing", "CODE_MANIFEST.sha256 holds a line that is not sha256 and path",
    "a record, and no PROTOCOL.md")


def test_every_protocol_refusal_case_is_run():
    assert set(PROTOCOL_REFUSAL_CASES) == set(_protocol_refusals())


@pytest.mark.parametrize("case", PROTOCOL_REFUSAL_CASES)
def test_the_build_is_refused_for_a_protocol_or_a_record_that_does_not_hold_together(tmp_path, case):
    from es.public_package import PackageRefused, build_package
    spec = _protocol_refusals()[case]
    dev = make_tree(tmp_path / "dev")
    sealed = make_sealed(tmp_path / "sealed", **spec["sealed"])
    outs = make_outputs(tmp_path / "outs", tests=spec["outputs"]) if spec.get("outputs") else tmp_path / "no_outputs"
    before = (snapshot(dev), snapshot(sealed), snapshot(outs) if outs.exists() else {})
    out = tmp_path / "pkg"
    with pytest.raises(PackageRefused) as err:
        build_package(dev, out, entry_points=FAKE_ENTRY, data=FAKE_DATA, sealed_dir=sealed, sealed_outputs_dir=outs)
    for needle in spec["want"]:
        assert needle.lower() in str(err.value).lower(), (needle, str(err.value))
    assert not out.exists() and not out.with_name(out.name + ".partial").exists()
    assert (snapshot(dev), snapshot(sealed), snapshot(outs) if outs.exists() else {}) == before         # nothing was written anywhere


def test_every_protocol_problem_is_listed_not_only_the_first(tmp_path):
    from es.public_package import PackageRefused, build_package
    dev = make_tree(tmp_path / "dev")
    sealed = make_sealed(tmp_path / "sealed", record=None, claims=("A",), pulls=True, drop=("CODE_MANIFEST.sha256",),
                         extra={"PROTOCOL.md": PROTOCOL_MD + b" edited"})
    with pytest.raises(PackageRefused) as err:
        build_package(dev, tmp_path / "pkg", entry_points=FAKE_ENTRY, data=FAKE_DATA + ("CLAUDE.md",), sealed_dir=sealed,
                      sealed_outputs_dir=tmp_path / "no_outputs")
    reasons = err.value.reasons
    assert len(reasons) == 5                                          # the hash, the manifest, the claim, PULLS.json, CLAUDE.md
    for needle in ("PROTOCOL.sha256", "CODE_MANIFEST.sha256", "RUN_A.claim", "PULLS.json", "CLAUDE.md"):
        assert any(needle in r for r in reasons), (needle, reasons)


def test_a_link_among_the_protocol_files_or_the_sealed_outputs_is_refused(tmp_path):
    from es.public_package import PackageRefused
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"a file outside the sealed folders\n")
    for where in ("protocol", "outputs"):
        root = tmp_path / where
        sealed = make_sealed(root / "sealed", record="local", runs={"A": RUN_A}, claims=("A",))
        outs = make_outputs(root / "outs", tests=("A",))
        link = sealed / "ENV_PINS.json" if where == "protocol" else outs / "A" / "tables" / "economics.csv"
        link.unlink()
        link.symlink_to(outside)
        with pytest.raises(PackageRefused, match="symbolic link"):
            stage_protocol(root, sealed, outs)
        assert not (root / "pkg").exists() and not (root / "pkg.partial").exists()


def test_a_protocol_file_that_changes_during_the_build_is_refused(tmp_path, monkeypatch):
    import es.public_package as pp
    sealed = make_sealed(tmp_path / "sealed", record="local", runs={"A": RUN_A}, claims=("A",))
    real_copy = pp.shutil.copyfile
    calls = []

    def copy_then_edit(src, dst, *a, **k):
        out = real_copy(src, dst, *a, **k)
        if str(src).endswith("RUNS.json") and not calls:
            calls.append(1)
            Path(src).write_bytes(Path(src).read_bytes() + b"\n")                 # still valid JSON, but not the bytes that were copied
        return out

    monkeypatch.setattr(pp.shutil, "copyfile", copy_then_edit)
    with pytest.raises(RuntimeError, match="changed while"):
        stage_protocol(tmp_path, sealed)
    assert not (tmp_path / "pkg").exists() and not (tmp_path / "pkg.partial").exists()


def test_a_protocol_that_changes_between_the_check_and_the_copy_is_refused(tmp_path, monkeypatch):
    """The checks run on the sources and again on the copies: a PROTOCOL.md edited after the first and before the copy is not published."""
    import es.public_package as pp
    sealed = make_sealed(tmp_path / "sealed", record="local", runs={"A": RUN_A}, claims=("A",))
    real = pp.protocol_copies

    def edit_then_copy(*a, **k):
        (sealed / "PROTOCOL.md").write_bytes(PROTOCOL_MD + b" edited after the check")
        return real(*a, **k)

    monkeypatch.setattr(pp, "protocol_copies", edit_then_copy)
    with pytest.raises(RuntimeError, match="does not hold together"):
        stage_protocol(tmp_path, sealed)
    assert not (tmp_path / "pkg").exists() and not (tmp_path / "pkg.partial").exists()


def test_a_protocol_file_over_the_size_limit_is_named_in_the_warnings_and_still_copied(tmp_path, monkeypatch):
    import es.public_package as pp
    assert pp.LARGE_FILE_BYTES == 5 * 1024 * 1024                                       # the limit of the brief
    monkeypatch.setattr(pp, "LARGE_FILE_BYTES", 1000)
    sealed = make_sealed(tmp_path / "sealed", record="local", runs={"A": RUN_A}, claims=("A",),
                         extra={"code_archive.tar.gz": b"x" * 1001})
    outs = make_outputs(tmp_path / "outs", tests=("A",), files={"results.json": b"y" * 1001, "RUN_LOG.txt": b"short"})
    _, out, staged = stage_protocol(tmp_path, sealed, outs)
    assert (out / "protocol" / "code_archive.tar.gz").stat().st_size == 1001
    big = [w for w in staged.warnings if " over " in w]
    assert len(big) == 2 and any("protocol/code_archive.tar.gz" in w for w in big)
    assert any("protocol/sealed_outputs/A/results.json" in w for w in big) and not any("RUN_LOG" in w for w in big)


def test_the_sealed_folders_are_found_from_the_dev_tree_when_none_is_given(tmp_path):
    """The defaults are those of the real tree, found from the dev root: W/docs/sealed beside dart_dev, and dart_dev/outputs/es/sealed."""
    from es.public_package import build_package
    dev = make_tree(tmp_path / "W" / "dart_dev")
    make_sealed(tmp_path / "W" / "docs" / "sealed", record="local", runs={"A": RUN_A}, claims=("A",))
    make_outputs(dev / "outputs" / "es" / "sealed", tests=("A",))
    out = tmp_path / "pkg"
    staged = build_package(dev, out, entry_points=FAKE_ENTRY, data=FAKE_DATA)
    assert {"protocol/PROTOCOL.md", "protocol/REGISTRATION.md", "protocol/RUNS.json", "protocol/sealed_outputs/A/results.json"} <= files_of(out)
    assert staged.protocol is not None and staged.protocol.kind == "local"


def test_a_relative_dev_root_finds_the_sealed_folders_beside_it_and_a_folder_that_is_given_is_used_as_it_is(tmp_path, monkeypatch):
    from es.public_package import sealed_locations
    (tmp_path / "W" / "dart_dev").mkdir(parents=True)
    monkeypatch.chdir(tmp_path / "W" / "dart_dev")
    sealed, outs = sealed_locations(".")                                    # "." has no useful parent until it is resolved
    assert sealed == (tmp_path / "W" / "docs" / "sealed").resolve()
    assert outs == (tmp_path / "W" / "dart_dev" / "outputs" / "es" / "sealed").resolve()
    assert sealed_locations(".", "a", "b") == (Path("a"), Path("b")) and sealed_locations(".", "a")[1] == outs


def test_the_package_is_never_written_in_or_around_the_sealed_folders(tmp_path, capsys):
    from es.public_package import PackageRefused, build_package, main
    dev = make_tree(tmp_path / "dev")
    sealed = make_sealed(tmp_path / "sealed", record=None)
    outs = make_outputs(tmp_path / "outs", tests=("A",), files={"results.json": b"{}"})
    before = (snapshot(sealed), snapshot(outs))
    for out in (sealed / "pkg", sealed, outs / "pkg", outs, tmp_path):                  # inside, equal to and around each folder
        with pytest.raises(PackageRefused, match="in, or around"):
            build_package(dev, out, entry_points=FAKE_ENTRY, data=FAKE_DATA, sealed_dir=sealed, sealed_outputs_dir=outs)
    assert main(["--dev-root", str(dev), "--out", str(sealed / "pkg"), "--dry-run", "--entry", "es.sealed_run", "--data", "v2/NODE_MAP.csv",
                 "--data", "cache/common_panel_groups.json", "--sealed-dir", str(sealed), "--sealed-outputs-dir", str(outs)]) == 2
    assert "in, or around" in capsys.readouterr().out
    assert (snapshot(sealed), snapshot(outs)) == before and not (sealed / "pkg").exists() and not (sealed / "pkg.partial").exists()
    assert not (outs / "pkg").exists() and not (outs / "pkg.partial").exists()


def test_a_manifest_path_that_leaves_the_package_is_a_departure_and_is_not_read(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"a file beside the package folder\n")
    manifest = code_manifest_text() + f"{sha(outside)}  ../outside.txt\n{sha(outside)}  {outside}\n"
    text, _, staged = readme_for(tmp_path, record="local", extra={"CODE_MANIFEST.sha256": manifest.encode()})
    assert staged.protocol.departures == ("../outside.txt", str(outside))                 # matching hashes, but not files of the package
    assert "`../outside.txt`" in _binding_section(text)


@pytest.mark.parametrize("line, accepted", [
    ("{h}  PROTOCOL.md\n", True), ("{h} *PROTOCOL.md\n", True), ("{h}\n", True), ("{H}  PROTOCOL.md\n", True), ("\n{h}  PROTOCOL.md\n\n", True),
    ("{h}  OTHER.md\n", False), ("{h}  PROTOCOL.md\n{h}  PROTOCOL.md\n", False), ("\n", False), ("{h}x  PROTOCOL.md\n", False)])
def test_the_hash_line_of_protocol_sha256_is_read_as_shasum_and_the_protocol_writer_write_it(tmp_path, line, accepted):
    from es.public_package import protocol_problems
    h = sha_bytes(PROTOCOL_MD)
    sealed = make_sealed(tmp_path / "sealed", record=None, extra={"PROTOCOL.sha256": line.format(h=h, H=h.upper()).encode()})
    problems = protocol_problems(sealed, tmp_path / "no_outputs")
    assert (problems == []) if accepted else (len(problems) == 1 and "PROTOCOL.sha256" in problems[0]), problems


def test_a_package_without_a_protocol_is_the_package_of_today(tmp_path):
    """No PROTOCOL.md: no sealed folder at all, or one that holds only the templates and a lock, gives the same files and the same README."""
    a_root, b_root = tmp_path / "a", tmp_path / "b"
    sealed = make_sealed(b_root / "sealed", record=None, drop=tuple(COPIED_PROTOCOL))             # the templates, a lock and a note remain
    assert sorted(p.name for p in sealed.iterdir()) == sorted(SEALED_NOT_COPIED)
    _, out_a, staged_a = stage(a_root, sealed_dir=a_root / "no_sealed", sealed_outputs_dir=a_root / "no_outputs")
    _, out_b, staged_b = stage(b_root, sealed_dir=sealed, sealed_outputs_dir=b_root / "no_outputs")
    _, out_c, _ = stage(tmp_path / "c")                                                              # and with no argument at all
    assert files_of(out_a) == files_of(out_b) == files_of(out_c) == set(INCLUDED) | NEW_FILES
    texts = [(o / "README.md").read_text() for o in (out_a, out_b, out_c)]
    assert texts[0] == texts[1] == texts[2]
    assert [(o / "MANIFEST.sha256").read_text() for o in (out_a, out_b)] == [(out_c / "MANIFEST.sha256").read_text()] * 2
    assert NO_REGISTRATION_PARAGRAPH in " ".join(texts[0].split()) and claims_a_registration(texts[0]) == []
    assert "protocol/" not in texts[0] and LOCAL_SENTENCE not in texts[0]
    assert staged_a.protocol is None and staged_b.protocol is None


# ---------------------------------------------------------------------------------------------------------------
# The three states of the README: (a) no protocol/ folder, (b) a protocol and no record, (c) a record
# ---------------------------------------------------------------------------------------------------------------
SECTIONS = ["## What this is", "## Environment", "## Layout and setup", "## Where the data come from", "## Run order",
            "## How the protocol binds the code", "## Tests", "## Verifying this package", "## Licence and citation"]


def _flat(text):
    return " ".join(text.split())


def _binding_section(text):
    return _flat(text).split("## How the protocol binds the code", 1)[1].split("## Tests", 1)[0]


def test_readme_b_says_that_the_protocol_is_in_protocol_and_that_no_record_exists(tmp_path):
    text, out, staged = readme_for(tmp_path, record=None)
    flat, section = _flat(text), _binding_section(text)
    assert section.startswith(" The protocol is in `protocol/`, with its files: `PROTOCOL.md`, `PROTOCOL.sha256`, `CODE_MANIFEST.sha256`, ")
    assert "No record of it exists: there is no `protocol/REGISTRATION.md`." in section
    assert "The sealed windows (Test A: 1 July to 30 September 2026; Test B: 1 October to 31 December 2026) have not been opened." in section
    assert "Because no record exists, the sealed pull and the sealed run refuse to run." in section
    assert "The machinery is included as it stands, because the protocol fixes it." in section
    assert "The protocol files are not part of this repository" not in flat and "No registration was made" not in flat
    assert "A record of the protocol is" not in flat and "The record of the protocol is" not in flat and LOCAL_SENTENCE not in flat
    assert "Sealed tests (not run; see the next section):" in flat
    assert ("11. `python -m es.protocol` writes the protocol package into `W/docs/sealed/`" in flat
            and "The protocol and its files are copied to `protocol/` in this repository; the code reads `W/docs/sealed/`, not `protocol/`." in flat)
    assert "Steps 12 and 13 run only when a record, `W/docs/sealed/REGISTRATION.md`, names the SHA-256 of `PROTOCOL.md`; none exists." in flat
    assert "- Test A" not in text and "- Test B" not in text                                       # nothing has run, so no list of runs
    assert f"`shasum -a 256 protocol/PROTOCOL.md` prints `{sha_bytes(PROTOCOL_MD)}`. This is the content of `protocol/PROTOCOL.sha256`." in section
    assert "`shasum -a 256 -c protocol/CODE_MANIFEST.sha256`" in section and "protocol_sha256" not in section.split("To check")[1]
    assert "Apart from the folder `protocol/`" in flat
    assert [ln for ln in text.splitlines() if ln.startswith("## ")] == SECTIONS
    assert staged.protocol is not None and staged.protocol.record is None and staged.protocol.kind is None


@pytest.mark.parametrize("kind", ["local", "public", "osf"])
def test_readme_c_says_which_kind_of_record_it_is_and_its_utc_time(tmp_path, kind):
    text, out, staged = readme_for(tmp_path, record=kind, runs={"A": RUN_A}, claims=("A",))
    flat, section = _flat(text), _binding_section(text)
    assert section.startswith(" The protocol is in `protocol/`, with its files: `PROTOCOL.md`, ")
    assert "The record of the protocol is `protocol/REGISTRATION.md`." in section
    assert f"{RECORD_TIME} (UTC)" in section
    expected = {"local": ("It is a local record, `author-statement-2026-10-26`, recorded at",),
                "public": ("It is a public record:", "`https://github.com/someone/dart-spread/releases/tag/v1.0`",
                           "Nothing in this repository checks the record against GitHub."),
                "osf": ("It names an OSF registration, `10.17605/OSF.IO/ABCDE`,",
                        "Nothing in this repository checks the record against OSF.")}[kind]
    for needle in expected:
        assert needle in section, (needle, section)
    assert "No record of it exists" not in flat and "have not been opened" not in flat
    assert "The protocol files are not part of this repository" not in flat and "No registration was made" not in flat
    assert "Sealed tests (see the next section for the ones that have run):" in flat and "Sealed tests (not run;" not in flat
    assert ("The protocol and its files are copied to `protocol/` in this repository, and so are the records of the sealed tests; the code "
            "reads `W/docs/sealed/`, not `protocol/`." in flat)
    assert ("Steps 12 and 13 run only when a record, `W/docs/sealed/REGISTRATION.md`, names the SHA-256 of `PROTOCOL.md`. "
            "The record in `protocol/REGISTRATION.md` does." in flat)
    assert "The record of the protocol is `protocol/REGISTRATION.md`. It " in section              # the sentence after it says what kind of record it is
    assert "The records of the pulls and the runs are there too: `RUNS.json` and `RUN_A.claim`." in section
    assert section.index("The record of the protocol is") < section.index("The records of the pulls and the runs")
    assert "The pull of each window" not in section                                                  # no PULLS.json in this fixture
    assert (f"`shasum -a 256 protocol/PROTOCOL.md` prints `{sha_bytes(PROTOCOL_MD)}`. This is the content of `protocol/PROTOCOL.sha256` and "
            "the `protocol_sha256` of `protocol/REGISTRATION.md`." in section)
    assert "`shasum -a 256 -c protocol/CODE_MANIFEST.sha256`" in section
    assert [ln for ln in text.splitlines() if ln.startswith("## ")] == SECTIONS
    assert staged.protocol.kind == {"local": "local", "public": "public_release", "osf": "osf"}[kind]
    assert staged.protocol.record["registered_utc"] == RECORD_TIME and staged.protocol.protocol_sha256 == sha_bytes(PROTOCOL_MD)


@pytest.mark.parametrize("kind, count", [("local", 1), ("public", 0), ("osf", 0), (None, 0)])
def test_the_local_record_sentence_is_in_the_readme_only_for_a_local_record(tmp_path, kind, count):
    text, _, _ = readme_for(tmp_path, record=kind)
    assert text.count(LOCAL_SENTENCE) == count                    # literally, on one line, once
    assert "own statement" not in text                            # the earlier wording ("the author's own statement") is in no state
    assert text.count("an AI assistant wrote") == count and text.count("did not write or sign it") == count


def test_the_readme_says_where_the_pulls_are_recorded_only_when_pulls_json_is_copied(tmp_path):
    from dataclasses import replace

    from es.public_package import PROTOCOL_FILES, ProtocolState, RunState, readme_text
    text, _, staged = readme_for(tmp_path / "with", record="local", runs={"A": RUN_A}, claims=("A",), pulls=True)
    assert "PULLS.json" in staged.protocol.files and _flat(text).count(PULLS_CLAUSE) == 1
    assert ("The records of the pulls and the runs are there too: `PULLS.json`, `RUNS.json` and `RUN_A.claim`. " + PULLS_CLAUSE
            in _binding_section(text))                                                       # right after the sentence that names the file
    assert _binding_section(text).index(LOCAL_SENTENCE) < _binding_section(text).index(PULLS_CLAUSE)
    without, _, staged = readme_for(tmp_path / "without", record="local", runs={"A": RUN_A}, claims=("A",))
    assert "PULLS.json" not in staged.protocol.files and "PULLS.json" not in without and "The pull of each window" not in without
    # With a protocol and no record a build never has a PULLS.json (a pull needs a record, and the build refuses it); the text says it all the same.
    idle = RunState(False, False, None, None, False)
    state = ProtocolState(files=PROTOCOL_FILES[:3] + ("PULLS.json",), outputs=(), protocol_sha256="a" * 64, record=None, kind=None,
                          runs={"A": idle, "B": idle}, departures=())
    env = {"python": "3.12.0", "libraries": {"numpy": "1.0"}}
    flat_b = _flat(readme_text(None, env, protocol=state))
    assert flat_b.count(PULLS_CLAUSE) == 1 and "No record of it exists" in flat_b
    assert flat_b.index("because the protocol fixes it.") < flat_b.index(PULLS_CLAUSE)
    assert PULLS_CLAUSE not in _flat(readme_text(None, env, protocol=replace(state, files=PROTOCOL_FILES[:3])))


def _data_folder(tmp_path, monkeypatch):
    """A temporary stand-in for ~/dart-spread-data: the lookup of a pull folder from the root of its PULLS.json entry accepts a folder under it only."""
    import es.public_package as pp
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(pp, "DATA_ROOT", data)
    return data


def _pull_paths(test, names=PULL_FILES):
    return {f"protocol/sealed_pulls/{test}/{name}" for name in names}


def test_the_pull_logs_are_copied_byte_for_byte_and_listed_in_the_manifest(tmp_path):
    from es.public_package import PullLogState, read_protocol_state, verify_manifest
    pa, pb = make_pull(tmp_path / "pull_a"), make_pull(tmp_path / "pull_b")
    sealed = make_sealed(tmp_path / "sealed", record="local", runs={"A": RUN_A}, claims=("A",), pulls={"A": {"root": "unused"}, "B": {"root": "unused"}})
    before = (snapshot(pa), snapshot(pb), snapshot(sealed))
    _, out, staged = stage_protocol(tmp_path, sealed, pull_dirs={"A": pa, "B": pb})
    assert pulled_files(out) == _pull_paths("A") | _pull_paths("B")
    for test, folder in (("A", pa), ("B", pb)):
        for name, data in PULL_FILES.items():
            assert (out / "protocol" / "sealed_pulls" / test / name).read_bytes() == data == (folder / name).read_bytes(), (test, name)
    listed = {ln.split("  ", 1)[1] for ln in (out / "MANIFEST.sha256").read_text().splitlines()}
    assert _pull_paths("A") | _pull_paths("B") <= listed and verify_manifest(out) == []
    assert (snapshot(pa), snapshot(pb), snapshot(sealed)) == before                         # the pull folders are only read
    for p in out.rglob("*"):                                                                 # nothing else of a pull folder gets in
        assert p.name not in (".state.json", "extra.csv", "y.txt") and p.suffix != ".parquet", p
        assert not ({"raw", "logs", "fmm", "nyiso_rtint"} & set(p.relative_to(out).parts)), p
    assert not [w for w in staged.warnings if "pull log" in w]
    assert staged.protocol.pull_logs == {t: PullLogState(tuple(PULL_FILES), ()) for t in "AB"} == read_protocol_state(out).pull_logs
    flat = _flat((out / "README.md").read_text())
    assert flat.count(PULL_LOGS_SENTENCE) == 1 and "pull log of Test" not in flat and "are not included in this repository" not in flat
    assert "`MANIFEST.sha256` in each folder lists the files of that pull, which are not part of this repository." in flat


def test_a_pull_folder_without_the_four_files_gives_warnings_and_copies_nothing(tmp_path):
    pa = make_pull(tmp_path / "pull_a", drop=tuple(PULL_FILES))                              # only the files that never are copied
    sealed = make_sealed(tmp_path / "sealed", record="local", runs={"A": RUN_A}, claims=("A",), pulls={"A": {"root": "unused"}})
    _, out, staged = stage_protocol(tmp_path, sealed, pull_dirs={"A": pa})
    assert pulled_files(out) == set() and not (out / "protocol" / "sealed_pulls").exists()
    warns = [w for w in staged.warnings if "pull log" in w]
    assert len(warns) == 4 and all("Test A" in w and "does not exist" in w for w in warns)
    assert all(any(name in w for w in warns) for name in PULL_FILES)
    flat = _flat((out / "README.md").read_text())
    assert "The logs of the pulls are not included in this repository." in flat and PULL_LOGS_SENTENCE not in flat
    assert "protocol/sealed_pulls" not in flat


def test_a_missing_pull_folder_warns_and_the_readme_says_which_pull_log_is_not_included(tmp_path):
    pb = make_pull(tmp_path / "pull_b")
    sealed = make_sealed(tmp_path / "sealed", record="local", runs={"A": RUN_A}, claims=("A",), pulls={"A": {"root": "x"}, "B": {"root": "y"}})
    _, out, staged = stage_protocol(tmp_path, sealed, pull_dirs={"A": tmp_path / "nowhere", "B": pb})
    assert pulled_files(out) == _pull_paths("B")
    assert [w for w in staged.warnings if w.startswith("the pull log of Test A is not included:") and "is not a folder" in w]
    flat = _flat((out / "README.md").read_text())
    assert flat.count(PULL_LOGS_SENTENCE) == 1 and "The pull log of Test A is not included." in flat and "pull log of Test B" not in flat
    assert staged.protocol.pull_logs["A"].included == () and staged.protocol.pull_logs["B"].missing == ()


def test_a_missing_file_of_a_pull_log_warns_and_the_readme_names_it(tmp_path):
    one = make_pull(tmp_path / "one", drop=("COVERAGE.csv",))
    two = make_pull(tmp_path / "two", drop=("COVERAGE.csv", "OVERLAP_CHECK.csv"))
    sealed = make_sealed(tmp_path / "sealed", record="local", runs={"A": RUN_A}, claims=("A",), pulls={"A": {"root": "x"}, "B": {"root": "y"}})
    _, out, staged = stage_protocol(tmp_path, sealed, pull_dirs={"A": one, "B": two})
    assert pulled_files(out) == _pull_paths("A", [n for n in PULL_FILES if n != "COVERAGE.csv"]) | _pull_paths("B", ["PULL_LOG.txt", "MANIFEST.sha256"])
    warns = [w for w in staged.warnings if "pull log" in w]
    assert len(warns) == 3 and sum("Test A" in w for w in warns) == 1 and sum("Test B" in w for w in warns) == 2
    flat = _flat((out / "README.md").read_text())
    assert "Of the pull log of Test A, `COVERAGE.csv` is not included." in flat
    assert "Of the pull log of Test B, `COVERAGE.csv` and `OVERLAP_CHECK.csv` are not included." in flat
    assert flat.count(PULL_LOGS_SENTENCE) == 1


def test_a_pull_log_file_that_is_a_link_is_not_copied(tmp_path):
    pa = make_pull(tmp_path / "pull_a")
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"a file outside the pull folder\n")
    (pa / "PULL_LOG.txt").unlink()
    (pa / "PULL_LOG.txt").symlink_to(outside)
    sealed = make_sealed(tmp_path / "sealed", record="local", runs={"A": RUN_A}, claims=("A",), pulls={"A": {"root": "x"}})
    _, out, staged = stage_protocol(tmp_path, sealed, pull_dirs={"A": pa})
    assert pulled_files(out) == _pull_paths("A", [n for n in PULL_FILES if n != "PULL_LOG.txt"])
    assert [w for w in staged.warnings if "PULL_LOG.txt" in w and "symbolic link" in w]
    assert "Of the pull log of Test A, `PULL_LOG.txt` is not included." in _flat((out / "README.md").read_text())
    assert b"outside the pull folder" not in b"".join(p.read_bytes() for p in out.rglob("*") if p.is_file())


def test_the_pull_folder_is_found_from_the_root_of_the_pulls_json_entry_under_the_data_folder_only(tmp_path, monkeypatch):
    data = _data_folder(tmp_path, monkeypatch)
    inside, outside = make_pull(data / "sealed_a"), make_pull(tmp_path / "elsewhere")
    sealed = make_sealed(tmp_path / "sealed", record="local", runs={"A": RUN_A}, claims=("A",),
                         pulls={"A": {"root": str(inside)}, "B": {"root": str(outside)}})
    _, out, staged = stage_protocol(tmp_path, sealed)                                        # no pull_dirs: the lookup
    assert pulled_files(out) == _pull_paths("A")
    for name, content in PULL_FILES.items():
        assert (out / "protocol" / "sealed_pulls" / "A" / name).read_bytes() == content
    skipped = [w for w in staged.warnings if "Test B" in w]
    assert len(skipped) == 1 and "is not a folder under" in skipped[0]
    assert "The pull log of Test B is not included." in _flat((out / "README.md").read_text())


@pytest.mark.parametrize("case, why", [
    ("the data folder itself", "is not a folder under"), ("a relative path", "not an absolute path"),
    ("a folder that does not exist", "not a folder that exists"), ("a link that leads out of the data folder", "is not a folder under"),
    ("a folder outside the data folder", "is not a folder under"), ("no root", "has no root"), ("a root that is not a string", "has no root")])
def test_a_root_that_is_not_a_folder_under_the_data_folder_is_skipped_with_a_warning(tmp_path, monkeypatch, case, why):
    data = _data_folder(tmp_path, monkeypatch)
    outside = make_pull(tmp_path / "elsewhere")
    (data / "escape").symlink_to(outside)
    entry = {"the data folder itself": {"root": str(data)}, "a relative path": {"root": "sealed_a"},
             "a folder that does not exist": {"root": str(data / "nope")}, "a link that leads out of the data folder": {"root": str(data / "escape")},
             "a folder outside the data folder": {"root": str(outside)}, "no root": {}, "a root that is not a string": {"root": 7}}[case]
    sealed = make_sealed(tmp_path / "sealed", record="local", runs={"A": RUN_A}, claims=("A",), pulls={"A": entry})
    _, out, staged = stage_protocol(tmp_path, sealed)
    assert pulled_files(out) == set() and not (out / "protocol" / "sealed_pulls").exists()          # skipped, and the build goes on
    warns = [w for w in staged.warnings if "pull log of Test A" in w]
    assert len(warns) == 1 and why in warns[0], warns
    assert "The logs of the pulls are not included in this repository." in _flat((out / "README.md").read_text())


def test_with_pull_dirs_only_the_folders_it_gives_are_used(tmp_path, monkeypatch):
    data = _data_folder(tmp_path, monkeypatch)
    pa, pb = make_pull(tmp_path / "pull_a"), make_pull(data / "sealed_b")
    sealed = make_sealed(tmp_path / "sealed", record="local", runs={"A": RUN_A}, claims=("A",), pulls={"A": {"root": "x"}, "B": {"root": str(pb)}})
    _, out, staged = stage_protocol(tmp_path, sealed, pull_dirs={"A": pa})
    assert pulled_files(out) == _pull_paths("A")                                              # B's root is good, but it is not looked up
    assert [w for w in staged.warnings if "Test B" in w and "no folder was given" in w]
    other = tmp_path / "second"                                                                # a folder for a test with no entry is ignored
    sealed = make_sealed(other / "sealed", record="local", runs={"A": RUN_A}, claims=("A",), pulls={"B": {"root": "x"}})
    _, out, staged = stage_protocol(other, sealed, pull_dirs={"A": pa, "B": pb})
    assert pulled_files(out) == _pull_paths("B") and not [w for w in staged.warnings if "Test A" in w]


@pytest.mark.parametrize("text, why", [(b"{nope", "not valid JSON"), (b"[1, 2]", "not a JSON object")])
def test_a_pulls_json_that_is_not_a_json_object_warns_and_no_pull_log_is_included(tmp_path, text, why):
    pa = make_pull(tmp_path / "pull_a")
    sealed = make_sealed(tmp_path / "sealed", record="local", runs={"A": RUN_A}, claims=("A",), extra={"PULLS.json": text})
    _, out, staged = stage_protocol(tmp_path, sealed, pull_dirs={"A": pa})
    assert pulled_files(out) == set() and (out / "protocol" / "PULLS.json").read_bytes() == text   # the file is copied as it is
    assert [w for w in staged.warnings if "PULLS.json" in w and why in w]
    assert "The logs of the pulls" not in _flat((out / "README.md").read_text())


def test_no_pull_log_is_looked_for_without_an_entry_of_a_sealed_test_in_pulls_json(tmp_path):
    pa = make_pull(tmp_path / "pull_a")
    first = tmp_path / "none"                                                                  # no PULLS.json at all
    _, out, staged = stage_protocol(first, make_sealed(first / "sealed", record="local"), pull_dirs={"A": pa})
    assert pulled_files(out) == set() and not [w for w in staged.warnings if "pull log" in w]
    second = tmp_path / "other"                                                                # an entry of something that is not a sealed test
    sealed = make_sealed(second / "sealed", record="local", pulls={"C": {"root": "x"}})
    _, out, staged = stage_protocol(second, sealed, pull_dirs={"A": pa, "C": pa})
    assert pulled_files(out) == set() and not [w for w in staged.warnings if "pull log" in w]
    assert "The logs of the pulls" not in _flat((out / "README.md").read_text())


def test_the_package_is_never_written_in_or_around_a_pull_folder(tmp_path):
    from es.public_package import PackageRefused, build_package
    dev = make_tree(tmp_path / "dev")
    pa = make_pull(tmp_path / "pull_a")                                                        # a file added here would break its manifest check
    sealed = make_sealed(tmp_path / "sealed", record="local", pulls={"A": {"root": "x"}})
    before = snapshot(pa)
    for out in (pa / "pkg", pa, tmp_path):
        with pytest.raises(PackageRefused, match="pull of Test A"):
            build_package(dev, out, entry_points=FAKE_ENTRY, data=FAKE_DATA, sealed_dir=sealed, sealed_outputs_dir=tmp_path / "no_outputs",
                          pull_dirs={"A": pa})
    assert snapshot(pa) == before and not (pa / "pkg").exists() and not (pa / "pkg.partial").exists()


def test_cli_lists_the_pull_logs_and_prints_the_warnings_once(tmp_path, capsys, monkeypatch):
    from es.public_package import main
    data = _data_folder(tmp_path, monkeypatch)
    pa = make_pull(data / "sealed_a", drop=("COVERAGE.csv",))
    dev = make_tree(tmp_path / "dev")
    sealed = make_sealed(tmp_path / "sealed", record="local", runs={"A": RUN_A}, claims=("A",),
                         pulls={"A": {"root": str(pa)}, "B": {"root": str(data / "nope")}})
    out = tmp_path / "pkg"
    base = ["--dev-root", str(dev), "--out", str(out), "--entry", "es.sealed_run", "--data", "v2/NODE_MAP.csv",
            "--data", "cache/common_panel_groups.json", "--sealed-dir", str(sealed), "--sealed-outputs-dir", str(tmp_path / "no_outputs")]
    assert main(base + ["--dry-run"]) == 0
    text = capsys.readouterr().out
    assert not out.exists() and "protocol/sealed_pulls/A/PULL_LOG.txt" in text and "protocol/sealed_pulls/A/MANIFEST.sha256" in text
    assert "protocol/sealed_pulls/A/COVERAGE.csv" not in text and ".state.json" not in text and ".parquet" not in text
    assert "warning: COVERAGE.csv of the pull log of Test A is not included" in text and "warning: the pull log of Test B is not included" in text
    assert main(base) == 0
    printed = capsys.readouterr().out
    assert printed.count("warning: ") == 2                                                     # the build prints each of them once
    assert (out / "protocol" / "sealed_pulls" / "A" / "PULL_LOG.txt").read_bytes() == PULL_FILES["PULL_LOG.txt"]


@pytest.mark.parametrize("record", [None, "local", "public", "osf"])
def test_no_state_of_the_readme_claims_a_registration_or_a_confirmatory_result(tmp_path, record):
    outs = make_outputs(tmp_path / "outs", tests=("A",)) if record else None
    pa = make_pull(tmp_path / "pull_a")
    text, _, _ = readme_for(tmp_path, record=record, outs=outs, runs={"A": RUN_A} if record else None, claims=("A",) if record else (),
                            pulls={"A": {"root": "x"}} if record else None, pull_dirs={"A": pa})
    assert (PULL_LOGS_SENTENCE in _flat(text)) == bool(record)                      # a pull log is described when PULLS.json is copied
    low = _flat(text).lower()
    for word in ("preregist", "pre-regist", "registered on", "confirmatory", "staging copy", "<to be filled>", "osf registration:"):
        assert word not in low, word
    assert "No registration was made" not in text
    if record != "osf":
        bare = re.sub(r"es\.registration(\.\w+)?|verify_registration|registration\.md", "", low)
        assert "registration" not in bare, bare[bare.index("registration") - 80:bare.index("registration") + 80]
    else:
        assert "osf registration" in low                                                          # the one place where the word is the record's kind


def test_readme_c_says_which_tests_have_run_from_the_records_and_quotes_no_result(tmp_path):
    outs = make_outputs(tmp_path / "outs", tests=("A",))
    text, out, staged = readme_for(tmp_path, record="local", outs=outs, runs={"A": RUN_A}, claims=("A",))
    section = _binding_section(text)
    assert "This is what the records in `protocol/` show about them (no result is quoted here):" in section
    assert ("- Test A (1 July to 30 September 2026): started at 2026-10-26T14:00:00+00:00 (UTC). It finished at 2026-10-26T20:30:00+00:00 (UTC). "
            "Its record is in `protocol/RUNS.json`, and its claim file is `protocol/RUN_A.claim`." in section)
    assert ("- Test B (1 October to 31 December 2026): not started. There is no record of it in `protocol/RUNS.json`, "
            "and there is no `protocol/RUN_B.claim`." in section)
    assert "The files that the runs wrote are in `protocol/sealed_outputs/A/`." in section
    assert "The `.parquet` files of the runs are not part of this repository." in section
    assert "0.123456" not in text and "t_hat" not in text                       # a number of the results.json is nowhere in the README
    assert staged.protocol.runs["A"].finished_utc == RUN_A["finished_utc"] and not staged.protocol.runs["B"].started


def test_a_local_record_with_no_test_started_is_a_warning_because_the_readme_says_after_the_run(tmp_path):
    _, _, none = readme_for(tmp_path / "none", record="local")
    assert [w for w in none.warnings if "after the run" in w and "no sealed test has started" in w]
    _, _, ran = readme_for(tmp_path / "ran", record="local", runs={"A": RUN_A}, claims=("A",))
    assert not [w for w in ran.warnings if "after the run" in w]
    _, _, claimed = readme_for(tmp_path / "claimed", record="local", claims=("B",))                   # a claim file alone is a start
    assert not [w for w in claimed.warnings if "after the run" in w]
    _, _, public = readme_for(tmp_path / "public", record="public")
    assert not [w for w in public.warnings if "after the run" in w]                                   # only a local record's sentence says it


def test_readme_c_nothing_has_run_yet(tmp_path):
    text, _, staged = readme_for(tmp_path, record="local")
    section = _binding_section(text)
    for test, window in (("A", "1 July to 30 September 2026"), ("B", "1 October to 31 December 2026")):
        assert (f"- Test {test} ({window}): not started. There is no record of it in `protocol/RUNS.json`, "
                f"and there is no `protocol/RUN_{test}.claim`.") in section
    assert "sealed_outputs" not in section and "parquet" not in section and "are there too" not in section      # the record alone
    assert not staged.protocol.runs["A"].started


def test_readme_c_a_claim_without_a_run_record_and_a_record_without_a_finish_time_are_said_as_they_are(tmp_path):
    claim_only, _, _ = readme_for(tmp_path / "claim", record="local", claims=("A",))
    assert ("- Test A (1 July to 30 September 2026): started at 2026-10-26T14:00:00+00:00 (UTC). There is no record of it in `protocol/RUNS.json`, "
            "and its claim file is `protocol/RUN_A.claim`.") in _binding_section(claim_only)
    no_finish, _, _ = readme_for(tmp_path / "unfinished", record="local", runs={"A": {"started_utc": RUN_A["started_utc"]}}, claims=("A",))
    section = _binding_section(no_finish)
    assert "started at 2026-10-26T14:00:00+00:00 (UTC). `protocol/RUNS.json` gives no finish time." in section
    assert "It finished" not in section
    no_claim, _, _ = readme_for(tmp_path / "noclaim", record="local", runs={"A": RUN_A})
    assert "Its record is in `protocol/RUNS.json`, and there is no `protocol/RUN_A.claim`." in _binding_section(no_claim)


def test_readme_c_a_test_that_failed_its_coverage_rule_is_said_without_a_result(tmp_path):
    outs = make_outputs(tmp_path / "outs", tests=("A",), files={"NOT_RUN.md": b"Coverage rule failed.\n", "tables/coverage.csv": b"a\n1\n",
                                                                 "kept/x.parquet": b"PAR1"})
    run = dict(RUN_A, status="not run (coverage rule)")
    text, out, staged = readme_for(tmp_path, record="local", outs=outs, runs={"A": run}, claims=("A",))
    section = _binding_section(text)
    assert "The coverage rule failed: `protocol/sealed_outputs/A/NOT_RUN.md` is there, and the test gave no result." in section
    assert (out / "protocol" / "sealed_outputs" / "A" / "NOT_RUN.md").read_bytes() == b"Coverage rule failed.\n"
    assert staged.protocol.runs["A"].not_run and not staged.protocol.runs["B"].not_run
    other, _, _ = readme_for(tmp_path / "ok", record="local", outs=make_outputs(tmp_path / "ok" / "outs"), runs={"A": RUN_A}, claims=("A",))
    assert "coverage rule failed" not in _flat(other)


def test_readme_says_whether_the_code_matches_the_manifest_of_the_protocol_as_built(tmp_path):
    ok_text, _, ok = readme_for(tmp_path / "ok", record="local")
    section = _binding_section(ok_text)
    assert ("When this package was built, every line of `protocol/CODE_MANIFEST.sha256` matched the file of the same path in this package."
            in section)
    assert ok.protocol.departures == () and not [w for w in ok.warnings if "CODE_MANIFEST" in w]
    bad_text, _, bad = readme_for(tmp_path / "bad", record="local",
                                  extra={"CODE_MANIFEST.sha256": (code_manifest_text(changed=("es/helper.py",)) + "0" * 64 + "  es/gone.py\n").encode()})
    section = _binding_section(bad_text)
    assert ("these files listed in `protocol/CODE_MANIFEST.sha256` differed from the file of the same path in this package, or were not in it: "
            "`es/helper.py`, `es/gone.py`." in section)
    assert "every line of" not in section
    assert bad.protocol.departures == ("es/helper.py", "es/gone.py")
    assert [w for w in bad.warnings if "CODE_MANIFEST" in w and "es/helper.py" in w and "es/gone.py" in w]


def test_readme_b_and_c_name_only_protocol_files_that_are_in_the_package(tmp_path):
    b_text, b_out, _ = readme_for(tmp_path / "b", record=None)
    outs = make_outputs(tmp_path / "c" / "outs", tests=("A", "B"))
    c_text, c_out, _ = readme_for(tmp_path / "c", record="local", outs=outs, runs={"A": RUN_A, "B": RUN_B}, claims=("A", "B"))
    for text, out, said_missing in ((b_text, b_out, {"protocol/REGISTRATION.md"}), (c_text, c_out, set())):    # (b) says it is not there
        named = set(re.findall(r"`(protocol/[\w./-]+)`", text))
        assert named and all((out / p).exists() for p in named - said_missing), sorted(p for p in named if not (out / p).exists())
        assert said_missing == {p for p in named if not (out / p).exists()}
        assert {"protocol/PROTOCOL.sha256", "protocol/CODE_MANIFEST.sha256"} <= named
    assert {"protocol/REGISTRATION.md", "protocol/RUNS.json", "protocol/RUN_A.claim", "protocol/sealed_outputs/A/", "protocol/sealed_outputs/B/"} <= \
        set(re.findall(r"`(protocol/[\w./-]+)`", c_text))


def test_readme_b_and_c_name_only_modules_and_files_that_are_in_the_real_package(tmp_path):
    """The README of a package with a protocol names no module or file the real package lacks (the real plan, a fake protocol)."""
    from es.config import DEV_ROOT
    from es.public_package import environment, plan_files, readme_text
    plan = plan_files(DEV_ROOT)
    for record in (None, "local"):
        _, _, staged = readme_for(tmp_path / str(record), record=record)
        text = readme_text(plan, environment(), protocol=staged.protocol)
        modules = set(re.findall(r"-m (es\.\w+|v2\.\w+)", text)) | set(re.findall(r"`((?:es|v2|src)\.\w+)`", text))
        assert modules and modules <= set(plan.modules), sorted(modules - set(plan.modules))
        paths = {p for p in re.findall(r"`((?:es|v2|src|tests|cache)/[\w./-]+)`", text) if not p.endswith("/")}
        assert paths and paths <= set(plan.files) | NEW_FILES, sorted(paths - set(plan.files))


def test_the_state_of_a_finished_package_is_read_from_what_is_copied(tmp_path):
    from es.public_package import read_protocol_state
    _, out_none, _ = stage(tmp_path / "none")
    assert read_protocol_state(out_none) is None
    _, out, staged = stage_protocol(tmp_path / "with", make_sealed(tmp_path / "with" / "sealed", record="public", runs={"A": RUN_A}, claims=("A",)))
    state = read_protocol_state(out)
    assert state == staged.protocol and state.kind == "public_release"
    assert state.files == ("PROTOCOL.md", "PROTOCOL.sha256", "CODE_MANIFEST.sha256", "ENV_PINS.json", "KNOWLEDGE_STATEMENT.md",
                           "protocol_inputs.json", "pre_pull_cutoffs.csv", "code_archive.tar.gz", "REGISTRATION.md", "RUNS.json", "RUN_A.claim")


def test_cli_lists_the_protocol_files_in_a_dry_run_builds_them_in_and_refuses_a_bad_protocol(tmp_path, capsys):
    from es.public_package import main
    dev = make_tree(tmp_path / "dev")
    sealed = make_sealed(tmp_path / "sealed", record="local", runs={"A": RUN_A}, claims=("A",))
    outs = make_outputs(tmp_path / "outs", tests=("A",))
    out = tmp_path / "pkg"
    base = ["--dev-root", str(dev), "--out", str(out), "--entry", "es.sealed_run", "--data", "v2/NODE_MAP.csv",
            "--data", "cache/common_panel_groups.json", "--sealed-dir", str(sealed), "--sealed-outputs-dir", str(outs)]
    assert main(base + ["--dry-run"]) == 0
    text = capsys.readouterr().out
    assert not out.exists() and "dry run" in text.lower()
    for name in COPIED_PROTOCOL | {"REGISTRATION.md", "RUNS.json", "RUN_A.claim"}:
        assert f"protocol/{name}" in text, name
    assert "protocol/sealed_outputs/A/results.json" in text and "protocol/sealed_outputs/A/tables/coverage.csv" in text
    assert ".parquet" not in text and "PULL_A.lock" not in text and "KNOWLEDGE_STATEMENT_TEMPLATE" not in text
    assert main(base) == 0
    assert (out / "protocol" / "PROTOCOL.md").read_bytes() == PROTOCOL_MD and "protocol/PROTOCOL.md" in capsys.readouterr().out
    (sealed / "PROTOCOL.md").write_bytes(b"edited")
    assert main(base + ["--rebuild"]) == 2
    refused = capsys.readouterr().out
    assert "REFUSED" in refused and "PROTOCOL.md" in refused
    assert (out / "protocol" / "PROTOCOL.md").read_bytes() == PROTOCOL_MD                            # a refused rebuild leaves the old package


def test_cli_without_a_sealed_folder_prints_that_there_is_no_protocol(tmp_path, capsys):
    from es.public_package import main
    dev = make_tree(tmp_path / "dev")
    assert main(["--dev-root", str(dev), "--out", str(tmp_path / "pkg"), "--dry-run", "--entry", "es.sealed_run",
                 "--data", "v2/NODE_MAP.csv", "--data", "cache/common_panel_groups.json"]) == 0
    assert "protocol/: none" in capsys.readouterr().out


# ---------------------------------------------------------------------------------------------------------------
# The scans
# ---------------------------------------------------------------------------------------------------------------
def _cats(hits):
    return sorted({(h.scan, h.category, h.line) for h in hits})


def test_credential_scan_finds_keys_tokens_passwords_and_addresses():
    from es.public_package import scan_text
    text = "\n".join([
        "x = 1",                                                                              # 1 clean
        "mail = '" + "jane.roe" + AT + "example.org'",                                          # 2 e-mail
        "API_KEY = '" + "AKIA" + "ABCDEFGHIJKLMNOP' ",                                         # 3 key assignment and an AWS id
        "headers = {'Authorization': 'Bear" + "er abcdefgh12345678'}",                              # 4 bearer value
        "password = \"hunter2hunter2\"",                                                       # 5 password
        "url = 'https://user:pa55word" + AT + "host.example/path'",                             # 6 credentials in a URL
        "r = get('https://api.example/v1?api_" + "key=abcdef123456')",                              # 7 key in a query string
        "tok = '" + "gh" + "p_" + "a" * 36 + "'",                                              # 8 GitHub token
        "-----BEGIN " + "RSA PRIVATE KEY-----",                                                # 9 private key
        "token = None",                                                                         # 10 clean: no string value
        "password = ''",                                                                        # 11 clean: empty
        "sha = '" + "0" * 64 + "'",                                                            # 12 clean: a hash
    ])
    hits = scan_text("a.py", text)
    cred = [h for h in hits if h.scan == "credentials"]
    assert {h.line for h in cred} == {2, 3, 4, 5, 6, 7, 8, 9}, _cats(cred)
    assert {h.category for h in cred if h.line == 2} == {"email"}
    assert "aws access key id" in {h.category for h in cred if h.line == 3}
    assert all(h.file == "a.py" for h in cred)


def test_path_scan_finds_home_paths_the_author_name_and_google_drive_paths():
    from es.public_package import scan_text
    drive = "Google" + "Drive-" + "ee" + "2625" + AT + "columbia" + ".edu"
    text = "\n".join([
        "x = 1",                                                                              # 1
        "p = '/Users" + "/" + "erce" + "erez/Desktop/ICAIF'",                                  # 2 the author's home
        "q = '" + HOME + "/data'",                                                             # 3 another user's home
        "r = '/Users/" + "erce" + "erez/Library/Cloud" + "Storage/" + drive + "/x'",           # 4 Drive path, address in it
        "s = '/content/" + "drive/My" + "Drive/ICAIF26'",                                      # 5 Colab Drive path
        "t = Path.home() / 'dart-spread-data'",                                                # 6 clean: relative to home
        "u = '/private" + "/tmp/" + "claude-501/x'",                                                     # 7 scratch path
    ])
    hits = [h for h in scan_text("a.py", text) if h.scan == "paths"]
    assert {h.line for h in hits} == {2, 3, 4, 5, 7}, _cats(hits)
    assert "author username" in {h.category for h in hits if h.line == 2}
    assert "home path" in {h.category for h in hits if h.line == 3}
    assert {"cloud drive path", "institutional id"} <= {h.category for h in hits if h.line == 4}


def test_name_scan_finds_given_names_and_name_shaped_phrases_in_comments_and_strings_only():
    from es.public_package import scan_text
    text = "\n".join([
        "x = 1",                                                                              # 1 clean
        "# thanks " + "to Jane " + "Roe for the idea",                                         # 2 given name, in a comment
        "y = 'Roe et " + "al. (2019) show this'",                                              # 3 citation, in a string
        "z = 'D" + "r. Doe asked for it'",                                                     # 4 title
        "w = ['Janeroe', 'Roey']",                                                             # 5 clean: not the whole word
        "def jane_roe_fix():",                                                                 # 6 an identifier is not a comment or a string
        "    return 1  # co-authored" + "-by: Richard " + "Doe",                               # 7 attribution
        '"""Notes by Pat ' + 'Quill:',                                                         # 8 docstring line
        '   Tom ' + 'Hale checked this."""',                                                   # 9 docstring line, two capitalised words
    ])
    hits = [h for h in scan_text("a.py", text, names=("Jane " + "Roe",)) if h.scan == "names"]
    by_line = {line: {h.category for h in hits if h.line == line} for line in {h.line for h in hits}}
    assert 1 not in by_line and 5 not in by_line and 6 not in by_line
    assert "name given" in by_line[2]
    assert "citation" in by_line[3]
    assert "title" in by_line[4]
    assert "attribution" in by_line[7]
    assert 8 in by_line and 9 in by_line                           # candidates: two capitalised words that are not common words


def test_a_year_after_a_month_or_an_ordinary_word_is_not_a_citation_but_an_author_and_year_is():
    from es.public_package import scan_text
    text = "\n".join([
        "# the run of 22 October 2026 and the sample of the 2026 data",                        # 1 months and ordinary words
        "# The 2026 sample starts in July 2026 (Sep 2025) and May-August 2025, Jan-Apr 2024",  # 2 same, and ranges of months
        "# stationary bootstrap (Zorbis and Quill " + "1994)",                                  # 3 a citation
        "# see Quill (" + "1994)",                                                              # 4 a citation
    ])
    cites = sorted(h.line for h in scan_text("a.py", text) if h.category == "citation")
    assert cites == [3, 4]


def test_a_github_user_in_a_url_is_a_handle():
    from es.public_package import scan_text
    hits = scan_text("a.py", "# see https://github" + ".com/someone-else/repo\n# a decorator @property and a key @article{x}\n")
    assert [(h.line, h.category) for h in hits if h.scan == "names"] == [(1, "handle")]


def test_given_names_match_without_accents_case_or_dots():
    """A made-up name with the Turkish dotless i and s-cedilla (built from code points, so this file holds no such letter)."""
    from es.public_package import scan_text
    accented = "Zorb" + chr(0x131) + chr(0x15f)
    text = f"# {accented} asked\n# ZORBIS again\nx = 'zorbisi'\n"
    hits = [h for h in scan_text("a.py", text, names=("Zorbis " + "Example",)) if h.scan == "names"]
    assert {h.line for h in hits} == {1, 2}


def test_scan_files_reports_file_and_line_and_survives_binary_bytes(tmp_path):
    from es.public_package import scan_files
    root = make_tree(tmp_path / "dev", extra={"es/helper.py": b"# \xff\xfe\nq = '" + HOME.encode() + b"/x'\n"})
    hits = scan_files(root, ["es/helper.py", "es/sealed_run.py"])
    assert [(h.file, h.line, h.category) for h in hits if h.scan == "paths"] == [("es/helper.py", 2, "home path")]


def test_the_scan_report_is_text_with_file_and_line():
    from es.public_package import Hit, format_hits
    out = format_hits([Hit("paths", "home path", "es/a.py", 3, "q = 1"), Hit("names", "citation", "es/b.py", 9, "x")])
    assert "es/a.py:3" in out and "es/b.py:9" in out and "[home path]" in out and "[citation]" in out


def test_the_builder_and_its_test_hold_nothing_the_scans_flag():
    from es.config import DEV_ROOT
    from es.public_package import scan_files
    hits = [h for h in scan_files(DEV_ROOT, ["es/public_package.py", "tests/es/test_public_package.py"])
            if h.scan in ("credentials", "paths")]
    assert hits == [], "\n".join(h.format() for h in hits)


# ---------------------------------------------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------------------------------------------
def test_cli_dry_run_prints_the_list_and_writes_nothing(tmp_path, capsys):
    from es.public_package import main
    dev = make_tree(tmp_path / "dev")
    out = tmp_path / "pkg"
    code = main(["--dev-root", str(dev), "--out", str(out), "--dry-run", "--entry", "es.sealed_run",
                 "--data", "v2/NODE_MAP.csv", "--data", "cache/common_panel_groups.json"])
    text = capsys.readouterr().out
    assert code == 0 and not out.exists()
    for rel in INCLUDED:
        assert rel in text
    for rel in UNRELATED:
        assert rel not in text
    assert "dry run" in text.lower()


def test_cli_builds_and_verifies_and_refuses_with_exit_code_2(tmp_path, capsys):
    from es.public_package import main
    dev = make_tree(tmp_path / "dev")
    out = tmp_path / "pkg"
    base = ["--dev-root", str(dev), "--out", str(out), "--entry", "es.sealed_run", "--data", "v2/NODE_MAP.csv",
            "--data", "cache/common_panel_groups.json"]
    assert main(base) == 0
    assert (out / "MANIFEST.sha256").exists()
    capsys.readouterr()
    assert main(["--verify", str(out)]) == 0
    (out / "es" / "helper.py").write_bytes(b"x")
    assert main(["--verify", str(out)]) == 1
    capsys.readouterr()
    assert main(base[:-2] + ["--data", "CLAUDE.md", "--rebuild"]) == 2
    assert "CLAUDE.md" in capsys.readouterr().out
    assert (out / "README.md").exists()                              # a refused rebuild leaves the old package where it was


def test_cli_reads_the_tests_note_and_the_names_from_files(tmp_path, capsys):
    from es.public_package import main
    dev = make_tree(tmp_path / "dev", extra={"es/helper.py": b"# written " + b"by Jane " + b"Roe\n"})
    out = tmp_path / "pkg"
    note = tmp_path / "note.md"
    note.write_text("These tests need the raw data: test_a, test_b.\n")
    names = tmp_path / "names.txt"
    names.write_text("Jane " + "Roe\n")
    code = main(["--dev-root", str(dev), "--out", str(out), "--entry", "es.sealed_run", "--data", "v2/NODE_MAP.csv",
                 "--data", "cache/common_panel_groups.json", "--tests-note-file", str(note), "--names-file", str(names)])
    text = capsys.readouterr().out
    assert code == 0
    assert "These tests need the raw data: test_a, test_b." in (out / "README.md").read_text()
    assert "es/helper.py:1" in text and "name given" in text                 # the scan report is printed, not written to the package


# ---------------------------------------------------------------------------------------------------------------
# The real tree (read only)
# ---------------------------------------------------------------------------------------------------------------
def test_the_real_entry_points_are_the_ones_of_the_spec():
    from es import protocol
    from es.public_package import DEVELOPMENT_PIPELINE, DEVELOPMENT_PULLS, ENTRY_POINTS, PAPER_GENERATORS, SEALED_PATH
    assert SEALED_PATH == ("es.sealed_run", "es.sealed_pull", "es.registration", "es.protocol") == tuple(protocol.SEALED_ROOTS)
    assert DEVELOPMENT_PIPELINE == ("es.run_analogs", "es.simulate", "es.run_economics", "es.run_economics_block", "es.run_monitors", "es.run_inference",
                                    "es.run_labels_pr", "es.run_legacy", "es.run_digest")
    assert PAPER_GENERATORS == ("es.paper_macros", "es.paper_tables", "es.paper_tables_sealed", "es.paper_figures",
                                "es.impl_table", "es.paper_lint", "es.paper_package")
    assert {"es.settlement", "v2.ingest_nyiso", "v2.ingest_isone", "v2.ingest_miso", "v2.ingest_caiso"} <= set(DEVELOPMENT_PULLS)
    assert len(set(ENTRY_POINTS)) == len(ENTRY_POINTS)


def test_the_real_file_list_has_no_refusal_and_no_unrelated_project():
    from es import protocol
    from es.config import DEV_ROOT
    from es.public_package import check_plan, plan_files
    plan = plan_files(DEV_ROOT)
    assert check_plan(plan) == []
    files = set(plan.files)
    must = {"es/sealed_run.py", "es/sealed_pull.py", "es/registration.py", "es/protocol.py", "es/run_analogs.py", "es/simulate.py",
            "es/run_economics.py", "es/run_economics_block.py", "es/run_monitors.py", "es/run_inference.py", "es/run_labels_pr.py", "es/run_legacy.py",
            "es/run_digest.py", "es/paper_macros.py", "es/paper_tables.py", "es/paper_tables_sealed.py", "es/paper_figures.py",
            "es/impl_table.py", "es/paper_lint.py", "es/paper_package.py", "es/settlement.py", "es/public_package.py",
            "v2/ingest_nyiso.py", "v2/ingest_isone.py", "v2/ingest_miso.py", "v2/ingest_caiso.py", "src/paths.py",
            "tests/es/conftest.py", "tests/es/test_public_package.py", "tests/es/test_paper_package.py", "pytest.ini",
            "v2/NODE_MAP.csv", "cache/common_panel_groups.json", "es/published_v3.csv"}
    assert must <= files, sorted(must - files)
    assert set(protocol.OPENED_DATA_FILES) <= set(plan.data)
    assert not [f for f in files if "afd" in f.lower() or f.endswith((".ipynb", ".parquet")) or f.split("/")[0] in ("outputs", ".superpowers")]
    assert not [f for f in files if f.split("/")[0] == "cache" and f not in protocol.OPENED_DATA_FILES]
    assert not [f for f in files if "__pycache__" in f or f.endswith(".pyc")]
    assert "CLAUDE.md" not in {Path(f).name for f in files}
    assert all(f in files for f in plan.tests) and all(f.startswith("tests/es/") for f in plan.tests)
    # every test file of tests/es is in, whatever it is
    on_disk = {p.relative_to(DEV_ROOT).as_posix() for p in (DEV_ROOT / "tests" / "es").rglob("*")
               if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    assert on_disk == set(plan.tests)


def test_the_readme_names_only_modules_and_files_that_are_in_the_package():
    from es.config import DEV_ROOT
    from es.public_package import environment, plan_files, readme_text
    plan = plan_files(DEV_ROOT)
    text = readme_text(plan, environment())
    modules = set(re.findall(r"-m (es\.\w+|v2\.\w+)", text)) | set(re.findall(r"`((?:es|v2|src)\.\w+)`", text))
    assert modules and modules <= set(plan.modules), sorted(modules - set(plan.modules))
    paths = {p for p in re.findall(r"`((?:es|v2|src|tests|cache)/[\w./-]+)`", text) if not p.endswith("/")}
    assert paths and paths <= set(plan.files) | NEW_FILES, sorted(paths - set(plan.files))
    for must in ("v2/ingest_nyiso.py", "v2/ingest_isone.py", "v2/ingest_miso.py", "v2/ingest_caiso.py"):
        assert must in text


def test_the_readme_names_the_drift_path_step_with_the_other_upstream_steps_and_its_files_and_needs():
    """(T1b) es.run_driftpath is one more step of the run order, between the economics (step 4 and the others of the development pipeline) and the paper
    generators that read its files (build.sh has no target for it, as it has none for run_economics and run_analogs, and is not changed)."""
    from es.config import DEV_ROOT
    from es.public_package import environment, plan_files, readme_text
    plan = plan_files(DEV_ROOT)
    text = readme_text(plan, environment())
    order = text.split("## Run order", 1)[1].split("## How the protocol binds the code", 1)[0]
    steps = {int(m.group(1)): m.group(0) for m in re.finditer(r"^(\d+)\. .*$", order, flags=re.M)}
    drift = [n for n, line in steps.items() if "python -m es.run_driftpath" in line]
    analogs = [n for n, line in steps.items() if "python -m es.run_analogs" in line]
    economics = [n for n, line in steps.items() if "python -m es.run_economics" in line]
    generators = [n for n, line in steps.items() if "python -m es.paper_figures" in line]
    assert len(drift) == len(analogs) == len(economics) == len(generators) == 1                      # one step each, in the same numbered list
    assert max(analogs[0], economics[0]) < drift[0] < generators[0]                                   # after the steps it checks against, before the generators that read it
    line = steps[drift[0]]
    assert "outputs/es/paper/drift_path.csv" in line and "outputs/es/paper/drift_summary.csv" in line
    assert "cached scores" in line and "never refits a model" in line and "cold cache" in line         # it needs the score cache and stops without it
    assert "es.run_driftpath" in plan.modules                                                          # the package holds the module the README names
    assert [int(n) for n in steps] == list(range(1, len(steps) + 1))                                   # the numbered list is still one run of numbers
    assert f"Steps {len(steps) - 1} and {len(steps)} run only when a registration record" in order     # the sealed steps that the sentence names are the last two


def test_the_tests_import_no_project_module_that_the_package_lacks():
    from es.config import DEV_ROOT
    from es.public_package import import_gap, plan_files
    plan = plan_files(DEV_ROOT)
    assert import_gap(DEV_ROOT, plan) == {}
