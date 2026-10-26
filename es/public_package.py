"""Stage the public code package (spec invariant I9: "the public package contains only this paper's price-only pipeline").

Run: DART_PROJECT_ROOT=... /opt/anaconda3/bin/python3 -m es.public_package [--dry-run] [--rebuild] [--out DIR]
     [--names-file FILE] [--tests-note-file FILE] [--scan-report FILE] [--sealed-dir DIR] [--sealed-outputs-dir DIR]

Nothing here uploads, pushes or posts anything. It writes one folder (``code_package/dart-operating-point-drift`` beside the
paper by default), which the author inspects and publishes.

The file list is computed, never typed:

1. the import closure of ``ENTRY_POINTS`` (the sealed path, the development pipeline, the paper generators, the code that
   pulled the development prices, and this module), traced by ``es.protocol.import_closure``, the tracer that finds the
   registered code manifest;
2. all of ``tests/es/`` (without ``__pycache__``) and ``pytest.ini``;
3. the data files the closure reads by a fixed path: ``es.protocol.OPENED_DATA_FILES`` and ``EXTRA_DATA_FILES``.

The list is printed. The build is refused, and nothing is written, if a listed file is missing, is not inside the dev tree (or
resolves to something that is not), or matches an exclusion: ``*.ipynb``, ``AFD*`` and ``*afd*``, ``CLAUDE.md``,
``.superpowers``, ``outputs/``, ``*.parquet``, and ``cache/`` except the listed JSON. Every problem is reported, not the first.

The copies are byte for byte: the code of ``es/``, ``src/`` and ``v2/`` is registered by SHA-256, so even a comment would
break the verification. Each file is hashed before and after its copy, and a source that changes during the build stops it.
The package gets four files of its own, written into the package only: README.md, requirements.txt, LICENSE and
MANIFEST.sha256 (``sha256  path`` for every other file). The build happens in ``<out>.partial`` and moves into place once it
is complete and verified, so a failed build never leaves a half package that looks finished.

The package has one more, optional, part: a ``protocol/`` folder, present only when ``docs/sealed/PROTOCOL.md`` exists. It holds
byte-identical copies (hashed before and after, like the code) of the files of ``docs/sealed/`` that exist among PROTOCOL_FILES
and RECORD_FILES, and under ``protocol/sealed_outputs/<A|B>/`` the files of ``outputs/es/sealed/<test>/`` of each test whose
folder exists, without ``*.parquet``. They are data, never code of the closure, and the manifest lists them. Where the two
folders are is an input of ``build_package`` (default: ``W/docs/sealed`` beside the dev root, and ``<dev root>/outputs/es/sealed``).
The build is refused, and every problem listed, when the protocol does not hold together: ``PROTOCOL.md`` does not hash to the
line of ``PROTOCOL.sha256``, the record does not parse with ``es.registration.parse_registration`` or names another hash, or a
record of a pull or a run (or sealed outputs) exists without the record of the protocol. The README is written for the state read
from the copies: no ``protocol/`` folder, a protocol and no record, or a record.

Beside them, for each test that has an entry in ``PULLS.json``, the four small text files of PULL_LOG_FILES of its pull folder (the log, the
coverage of each node, the check of the first days, the manifest of the pull) go into ``protocol/sealed_pulls/<test>/``, byte for byte. The
folder is found from the ``root`` of the entry, and only if it exists and lies under ``DATA_ROOT`` once resolved (a link cannot lead out of
it); ``build_package(pull_dirs={test: folder})`` replaces that lookup, for tests. Nothing else of a pull folder is ever copied (not
``.state.json``, not ``*.parquet``, not the data folders). A missing folder or file is a warning, never a refusal: the build does not depend
on data outside the repository. The README then says which pull log is not included.

``scan_files`` reports, and never edits: credentials, absolute paths and cloud-drive paths, and the names of people."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import re
import shutil
import sys
import tokenize
import unicodedata
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path, PurePosixPath

from .config import DATA_ROOT, DEV_ROOT, REVISION_ROOT

PACKAGE_NAME = "dart-operating-point-drift"
DEFAULT_OUT = REVISION_ROOT / "code_package" / PACKAGE_NAME
PAPER_TITLE = "Feature-Block Portability and Operating-Point Drift in DART-Spread Spike Prediction"
AUTHOR = "Erce Erez"
LICENSE_YEAR = 2026

#: The entry points of the file list (spec I9), by group.
SEALED_PATH = ("es.sealed_run", "es.sealed_pull", "es.registration", "es.protocol")
DEVELOPMENT_PIPELINE = ("es.run_analogs", "es.simulate", "es.run_economics", "es.run_economics_block", "es.run_monitors", "es.run_inference",
                        "es.run_labels_pr", "es.run_legacy", "es.run_digest")
PAPER_GENERATORS = ("es.paper_macros", "es.paper_tables", "es.paper_tables_sealed", "es.paper_figures", "es.impl_table",
                    "es.paper_lint", "es.paper_package")
#: The code that pulled the development prices: ``es.settlement`` pulled ~/dart-spread-data/caiso_fmm_dev and nyiso_rtint_dev
#: (its drivers, run_caiso_pull.py and run_nyiso_pull.py, sit outside dart_dev and call it unmodified), and the four July
#: downloaders pulled the 33-node panel (backfill_2023_2024, holdout_2026H1). v2.ingest_caiso is imported by nothing else.
DEVELOPMENT_PULLS = ("es.settlement", "v2.ingest_nyiso", "v2.ingest_isone", "v2.ingest_miso", "v2.ingest_caiso")
#: This module: tests/es/ goes in whole, and test_public_package.py needs it.
PACKAGING = ("es.public_package",)
ENTRY_POINTS = SEALED_PATH + DEVELOPMENT_PIPELINE + PAPER_GENERATORS + DEVELOPMENT_PULLS + PACKAGING

TESTS_DIR = "tests/es"
CONFIG_FILES = ("pytest.ini",)
#: Read by a fixed path beyond ``es.protocol.OPENED_DATA_FILES`` (which lists those of the sealed closure only): the published
#: numbers of the earlier version, ``PUBLISHED_CSV = Path(__file__).with_name("published_v3.csv")`` in es/run_legacy.py, and
#: read by tests/es/test_run_legacy.py. Found by scanning the file-name constants of every module of the wider closure.
EXTRA_DATA_FILES = ("es/published_v3.csv",)
NEW_FILES = ("README.md", "requirements.txt", "LICENSE", "MANIFEST.sha256")
#: The optional protocol/ folder: the files of docs/sealed/ that exist (the protocol and what it binds, then the records of the
#: pull and of the run) and, under protocol/sealed_outputs/<test>/, the files of outputs/es/sealed/<test>/ except ``*.parquet``.
#: KNOWLEDGE_STATEMENT_TEMPLATE.md, REGISTRATION.template.md, PULL_*.lock and everything else in docs/sealed/ stay out.
PROTOCOL_DIR = "protocol"
PROTOCOL_FILES = ("PROTOCOL.md", "PROTOCOL.sha256", "CODE_MANIFEST.sha256", "ENV_PINS.json", "KNOWLEDGE_STATEMENT.md",
                  "protocol_inputs.json", "pre_pull_cutoffs.csv", "code_archive.tar.gz")
RECORD_FILES = ("REGISTRATION.md", "PULLS.json", "RUNS.json", "RUN_A.claim", "RUN_B.claim")
SEALED_OUTPUTS_DIR = "sealed_outputs"
TEST_IDS = ("A", "B")
TEST_WINDOWS = {"A": "1 July to 30 September 2026", "B": "1 October to 31 December 2026"}
#: A copied protocol file over this many bytes is named in the warnings of the build (it is copied all the same).
LARGE_FILE_BYTES = 5 * 1024 * 1024
#: The small text files of the folder of a pull (``~/dart-spread-data/<pull>/``) that go into protocol/sealed_pulls/<test>/ for each test that has
#: an entry in PULLS.json: its log, the coverage of each node, the check of its first days against the files already opened, and the manifest
#: of the pull. Nothing else: not .state.json, not ``*.parquet``, not raw/, logs/, fmm/ or nyiso_rtint/.
SEALED_PULLS_DIR = "sealed_pulls"
PULL_LOG_FILES = ("PULL_LOG.txt", "COVERAGE.csv", "OVERLAP_CHECK.csv", "MANIFEST.sha256")
CORE_LIBRARIES = ("numpy", "pandas", "pyarrow", "xgboost", "scikit-learn", "scipy", "statsmodels", "matplotlib")


class PackageRefused(RuntimeError):
    """The package is not built: ``reasons`` holds every problem found, as ``path: reason``."""

    def __init__(self, reasons):
        self.reasons = list(reasons)
        super().__init__("the public package is refused, nothing was written:\n  " + "\n  ".join(self.reasons))


def _protocol():
    from . import protocol
    return protocol


def default_data_files():
    return tuple(_protocol().OPENED_DATA_FILES) + EXTRA_DATA_FILES


def _allowed_cache():
    """The JSON files that may come from cache/: the listed ones of es.protocol."""
    return tuple(f for f in _protocol().OPENED_DATA_FILES if f.startswith("cache/") and f.endswith(".json"))


def _sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------------------------------------------
# The file list
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Plan:
    dev_root: Path
    entry_points: tuple
    modules: dict          # module name -> path relative to the dev root: the import closure of the entry points
    code: tuple            # the closure's files
    tests: tuple           # everything under tests/es/ but caches
    config: tuple
    data: tuple            # data files read by a fixed path

    @property
    def files(self):
        return tuple(sorted(set(self.code) | set(self.tests) | set(self.config) | set(self.data)))


def _tests_files(root):
    base = Path(root) / TESTS_DIR
    if not base.is_dir():
        return ()
    out = []
    for p in sorted(base.rglob("*")):
        rel = p.relative_to(root)
        if "__pycache__" in rel.parts or ".pytest_cache" in rel.parts or p.name == ".DS_Store" or p.suffix in (".pyc", ".pyo"):
            continue
        if p.is_dir() and not p.is_symlink():
            continue
        out.append(rel.as_posix())
    return tuple(out)


def plan_files(dev_root=DEV_ROOT, entry_points=ENTRY_POINTS, data=None, config=CONFIG_FILES) -> Plan:
    """The files of the package: the import closure of ``entry_points``, tests/es/, ``config`` and the data files."""
    root = Path(dev_root)
    missing = [f"{name}: an entry point that is not a module under {root}" for name in entry_points
               if not _is_project_module(root, name)]
    if missing:
        raise PackageRefused(missing)
    try:
        modules, _ = _protocol().import_closure(root, roots=tuple(entry_points))
    except (SyntaxError, UnicodeDecodeError) as exc:         # a module of the closure that Python itself could not read
        raise PackageRefused([f"a module of the import closure cannot be parsed: {exc}"]) from exc
    rel_modules = {name: Path(f).relative_to(root).as_posix() for name, f in modules.items()}
    data = default_data_files() if data is None else tuple(data)
    return Plan(root, tuple(entry_points), rel_modules, tuple(sorted(set(rel_modules.values()))), _tests_files(root),
                tuple(config), tuple(sorted(set(data))))


def exclusion_reason(rel, allowed_cache=()):
    """Why ``rel`` (a path relative to the dev root) may not go into the package, or None. All the rules that match are named."""
    path = PurePosixPath(rel)
    parts = path.parts
    why = []
    if path.name.endswith(".ipynb"):
        why.append("matches *.ipynb (a notebook)")
    if "afd" in rel.lower():
        why.append("matches AFD* / *afd* (the text-extraction project)")
    if path.name == "CLAUDE.md":
        why.append("matches CLAUDE.md (the repository's rules)")
    if ".superpowers" in parts:
        why.append("matches .superpowers (agent working files)")
    if "outputs" in parts:
        why.append("matches outputs/ (results are not code)")
    if path.name.endswith(".parquet"):
        why.append("matches *.parquet (data)")
    if "cache" in parts and rel not in set(allowed_cache):
        why.append("matches cache/ (only the listed JSON may come from cache/)")
    return "; ".join(why) or None


def _not_usable(plan_root, resolved_root, rel):
    target = Path(plan_root) / rel
    try:
        real = target.resolve(strict=True)
    except FileNotFoundError:
        return "missing: the listed file does not exist"
    except (OSError, RuntimeError) as exc:
        return f"cannot be resolved ({exc})"
    if not real.is_relative_to(resolved_root):
        return f"outside the dev tree: it resolves to {real}"
    if not real.is_file():
        return "not a regular file"
    return None


def check_plan(plan, allowed_cache=None):
    """Every problem of a plan, as ``path: reason`` strings (empty list: the plan may be built)."""
    resolved_root = Path(plan.dev_root).resolve()
    allowed = _allowed_cache() if allowed_cache is None else tuple(allowed_cache)
    problems = []
    for rel in plan.files:
        for why in (_not_usable(plan.dev_root, resolved_root, rel), exclusion_reason(rel, allowed)):
            if why:
                problems.append(f"{rel}: {why}")
    return problems


def format_plan(plan):
    lines = [f"entry points ({len(plan.entry_points)}): {', '.join(plan.entry_points)}", f"files ({len(plan.files)}):"]
    for title, files in (("code: import closure of the entry points", plan.code), ("tests/es/", plan.tests),
                         ("config", plan.config), ("data read by a fixed path", plan.data)):
        lines.append(f"-- {title} ({len(files)})")
        lines.extend(f"   {f}" for f in files)
    return "\n".join(lines)


def _is_project_module(root, name):
    base = Path(root).joinpath(*name.split("."))
    return base.with_suffix(".py").is_file() or (base / "__init__.py").is_file()


def import_gap(dev_root, plan):
    """{module: file} for the project modules that tests/es imports (and that they import in turn) and that the package does
    not carry: a test of the package would fail to import them."""
    root = Path(dev_root)
    wanted = set()
    for rel in plan.tests:
        if not rel.endswith(".py"):
            continue
        tree = ast.parse((root / rel).read_text(encoding="utf-8", errors="replace"), filename=rel)
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module] + [f"{node.module}.{a.name}" for a in node.names if a.name != "*"]
            wanted.update(n for n in names if _is_project_module(root, n))
    if not wanted:
        return {}
    modules, _ = _protocol().import_closure(root, roots=tuple(sorted(wanted)))
    have = set(plan.files)
    return {name: Path(f).relative_to(root).as_posix() for name, f in modules.items()
            if Path(f).relative_to(root).as_posix() not in have}


# ---------------------------------------------------------------------------------------------------------------
# The environment, requirements.txt and LICENSE
# ---------------------------------------------------------------------------------------------------------------
def environment():
    """The interpreter and the core libraries, read live from the running environment."""
    return {"python": platform.python_version(), "libraries": {name: metadata.version(name) for name in CORE_LIBRARIES}}


def _installed_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def third_party_imports(dev_root, rel_paths):
    """{import name: [files]} for the imports of the listed .py files that are neither standard library nor project modules."""
    root = Path(dev_root)
    found = {}
    for rel in rel_paths:
        if not rel.endswith(".py"):
            continue
        try:
            tree = ast.parse((root / rel).read_text(encoding="utf-8"), filename=rel)
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                top = name.split(".")[0]
                if top in sys.stdlib_module_names or top == "__future__" or _is_project_module(root, top) \
                        or (root / TESTS_DIR / f"{top}.py").is_file():
                    continue
                found.setdefault(top, set()).add(rel)
    return {k: sorted(v) for k, v in sorted(found.items())}


def _extra_requirements(dev_root, plan):
    """({distribution: files} imported by the package's code, {distribution: files} imported by the tests only), without the core."""
    third = third_party_imports(dev_root, [f for f in plan.files if f.endswith(".py")])
    dists = metadata.packages_distributions()
    core = {_norm(n) for n in CORE_LIBRARIES}
    runtime, tests_only = {}, {}
    for imp, files in third.items():
        dist = (dists.get(imp) or [imp])[0]
        if _norm(dist) in core:
            continue
        (runtime if any(not f.startswith("tests/") for f in files) else tests_only).setdefault(dist, []).extend(files)
    return runtime, tests_only


def requirements_text(env, runtime=None, tests_only=None, installed=_installed_version):
    def line(name, files):
        version = installed(name)
        by = f"imported by {files[0]}" + (f" and {len(files) - 1} more" if len(files) > 1 else "")
        return f"{name}=={version}  # {by}" if version else f"{name}  # NOT INSTALLED here: pin it yourself ({by})"

    lines = ["# Pinned to the environment that produced the results: the versions were read with importlib.metadata",
             f"# when this package was staged. Python {env['python']}."]
    lines += [f"{name}=={version}" for name, version in env["libraries"].items()]
    lines += [line(name, files) for name, files in sorted((runtime or {}).items())]
    if tests_only:
        lines += ["", "# needed only to run the tests"]
        lines += [line(name, files) for name, files in sorted(tests_only.items())]
    return "\n".join(lines) + "\n"


def license_text():
    return f"""MIT License

Copyright (c) {LICENSE_YEAR} {AUTHOR}

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


# ---------------------------------------------------------------------------------------------------------------
# The protocol/ folder: the protocol, the records and the sealed outputs
# ---------------------------------------------------------------------------------------------------------------
def _registration():
    from . import registration
    return registration


def _present(path):
    return path.exists() or path.is_symlink()


def _regular(path):
    return path.is_file() and not path.is_symlink()


def sealed_locations(dev_root, sealed_dir=None, sealed_outputs_dir=None):
    """(the folder of the protocol and its records, the folder of the sealed outputs). Not given, they are those of the real tree,
    found from the dev root: ``W/docs/sealed`` beside ``dart_dev``, and ``dart_dev/outputs/es/sealed``."""
    root = Path(dev_root).resolve()                      # a relative dev root such as "." has no useful parent until it is resolved
    return (Path(sealed_dir) if sealed_dir is not None else root.parent / "docs" / "sealed",
            Path(sealed_outputs_dir) if sealed_outputs_dir is not None else root / "outputs" / "es" / "sealed")


_CACHE_DIRS = ("__pycache__", ".pytest_cache")


def _walk_sealed_outputs(folder):
    """(files, number of parquet files, problems) of the output folder of one sealed test. ``files`` are the files that go into the
    package, as (path relative to the folder, source path): everything but ``*.parquet`` (counted, not copied), .DS_Store and caches.
    A link, or an entry that is not a regular file, is a problem: nothing is copied through a link."""
    folder = Path(folder)
    if folder.is_symlink():
        return [], 0, [f"{folder}: a symbolic link; the package copies files, not links"]
    if not folder.is_dir():
        return [], 0, []
    files, parquet, problems = [], 0, []
    for here, dirnames, filenames in os.walk(folder):
        here = Path(here)
        problems += [f"{here / d}: a symbolic link; the package copies files, not links" for d in sorted(dirnames) if (here / d).is_symlink()]
        dirnames[:] = sorted(d for d in dirnames if d not in _CACHE_DIRS and not (here / d).is_symlink())
        for name in sorted(filenames):
            path = here / name
            if name == ".DS_Store" or path.suffix in (".pyc", ".pyo"):
                continue
            if path.is_symlink():
                problems.append(f"{path}: a symbolic link; the package copies files, not links")
            elif not path.is_file():
                problems.append(f"{path}: not a regular file")
            elif path.suffix == ".parquet":
                parquet += 1
            else:
                files.append((path.relative_to(folder).as_posix(), path))
    return sorted(files), parquet, problems


def protocol_copies(sealed_dir, outputs_dir):
    """The files of the protocol/ folder, as (source, path in the package) sorted by that path: the files of ``sealed_dir`` among
    PROTOCOL_FILES and RECORD_FILES that exist, and the files of ``outputs_dir/<A|B>/`` of each test whose folder exists, without
    ``*.parquet``. Empty when ``sealed_dir`` has no PROTOCOL.md: there is then no protocol/ folder."""
    sealed, outs = Path(sealed_dir), Path(outputs_dir)
    if not _present(sealed / "PROTOCOL.md"):
        return []
    copies = [(sealed / name, f"{PROTOCOL_DIR}/{name}") for name in PROTOCOL_FILES + RECORD_FILES if _regular(sealed / name)]
    for test in TEST_IDS:
        files, _, _ = _walk_sealed_outputs(outs / test)
        copies += [(path, f"{PROTOCOL_DIR}/{SEALED_OUTPUTS_DIR}/{test}/{rel}") for rel, path in files]
    return sorted(copies, key=lambda c: c[1])


def _pull_folder(test, entry, pull_dirs, data_root):
    """(folder, None) or (None, why not): the folder of the pull of ``test``. With ``pull_dirs`` (a dict, for tests) it is the folder that
    the dict gives and nothing else. Otherwise it is the ``root`` of the PULLS.json ``entry``, if that is an absolute path to a folder that
    exists and lies under the data folder once resolved (so that a link cannot lead out of it), and is not the data folder itself."""
    if pull_dirs is not None:
        given = pull_dirs.get(test)
        if given is None:
            return None, "no folder was given for it"
        return (Path(given), None) if Path(given).is_dir() else (None, f"{given} is not a folder")
    root = entry.get("root") if isinstance(entry, dict) else None
    if not isinstance(root, str) or not root:
        return None, "its entry in PULLS.json has no root"
    path = Path(root)
    if not path.is_absolute():
        return None, f"its root {root} is not an absolute path"
    data, real = Path(DATA_ROOT if data_root is None else data_root).resolve(), path.resolve()
    if real == data or not real.is_relative_to(data):
        return None, f"its root {root} is not a folder under {data}"
    if not real.is_dir():
        return None, f"its root {root} is not a folder that exists"
    return real, None


def pull_log_copies(sealed_dir, pull_dirs=None, data_root=None):
    """([(source, path in the package)], [warnings], {test: pull folder}): for each sealed test that has an entry in ``sealed_dir``/PULLS.json,
    the files of PULL_LOG_FILES of its pull folder, as protocol/sealed_pulls/<test>/<name>. Only those four names are ever listed. A folder
    that cannot be found or is not accepted (``_pull_folder``), a file that is missing and a file that is a link give a warning and are left
    out: the build does not depend on data outside the repository. Empty without a PROTOCOL.md or a PULLS.json."""
    sealed = Path(sealed_dir)
    copies, warnings, folders = [], [], {}
    if not (_present(sealed / "PROTOCOL.md") and _regular(sealed / "PULLS.json")):
        return copies, warnings, folders
    try:
        doc = json.loads((sealed / "PULLS.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return copies, [f"{sealed / 'PULLS.json'} is not valid JSON ({exc}): no pull log is included"], folders
    if not isinstance(doc, dict):
        return copies, [f"{sealed / 'PULLS.json'} is not a JSON object: no pull log is included"], folders
    for test in TEST_IDS:
        if test not in doc:
            continue
        folder, why = _pull_folder(test, doc[test], pull_dirs, data_root)
        if folder is None:
            warnings.append(f"the pull log of Test {test} is not included: {why}")
            continue
        folders[test] = folder
        for name in PULL_LOG_FILES:
            path = folder / name
            if path.is_symlink():
                warnings.append(f"{name} of the pull log of Test {test} is not included: {path} is a symbolic link")
            elif not path.is_file():
                warnings.append(f"{name} of the pull log of Test {test} is not included: {path} does not exist")
            else:
                copies.append((path, f"{PROTOCOL_DIR}/{SEALED_PULLS_DIR}/{test}/{name}"))
    return copies, warnings, folders


def _manifest_entries(text):
    """([(sha256 in lower case, path)], [numbers of the lines that are not '<sha256>  <path>']) of a code manifest. Blank lines are skipped."""
    entries, bad = [], []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        digest, sep, rel = line.partition("  ")
        if sep and rel and re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            entries.append((digest.lower(), rel))
        else:
            bad.append(n)
    return entries, bad


def _protocol_line_hash(path):
    """(hash in lower case, None) of the one line of PROTOCOL.sha256, '<sha256>  PROTOCOL.md' as es.protocol and ``shasum`` write it,
    or (None, why not)."""
    try:
        lines = [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    except (OSError, UnicodeDecodeError) as exc:
        return None, f"cannot be read ({exc})"
    found = re.fullmatch(r"([0-9a-fA-F]{64})(?:[ \t]+\*?PROTOCOL\.md)?", lines[0]) if len(lines) == 1 else None
    if not found:
        return None, "is not one line '<sha256>  PROTOCOL.md'"
    return found.group(1).lower(), None


def protocol_problems(sealed_dir, outputs_dir):
    """Every reason why the protocol/ folder cannot be built from these two folders, as ``path: reason`` (empty: it can, or there is
    nothing to build).

    Without PROTOCOL.md there is no protocol/ folder, and a record or sealed outputs are the problem: the package would hold them and
    not the protocol. With it: PROTOCOL.md must hash to the line of PROTOCOL.sha256; CODE_MANIFEST.sha256 must be there and well
    formed (the README says whether the code matches it); REGISTRATION.md, if there is one, must parse
    (``es.registration.parse_registration``, read only) and name the hash of PROTOCOL.sha256; RUNS.json must be a JSON object; and
    PULLS.json, RUNS.json, a RUN_<test>.claim or the outputs of a test need REGISTRATION.md, and the outputs of a test need its record
    in RUNS.json or its claim file, so that the README can say what has run. A link, or a file that is not a regular file, among
    the files to copy is a problem too."""
    sealed, outs = Path(sealed_dir), Path(outputs_dir)
    walks = {test: _walk_sealed_outputs(outs / test) for test in TEST_IDS}
    has_outputs = {test: bool(files or n_parquet) for test, (files, n_parquet, _) in walks.items()}
    claims = sorted(p.name for p in sealed.glob("RUN_*.claim")) if sealed.is_dir() else []
    records = [n for n in ("REGISTRATION.md", "PULLS.json", "RUNS.json") if _present(sealed / n)] + claims
    protocol = sealed / "PROTOCOL.md"
    if not _present(protocol):
        return ([f"{sealed / n}: a record, but {protocol} does not exist: the package would hold the record and not the protocol"
                 for n in records]
                + [f"{outs / t}: sealed outputs, but {protocol} does not exist: the package would hold the outputs and not the protocol"
                   for t in TEST_IDS if has_outputs[t]])
    problems = []
    for name in PROTOCOL_FILES + RECORD_FILES:
        path = sealed / name
        if path.is_symlink():
            problems.append(f"{path}: a symbolic link; the package copies files, not links")
        elif path.exists() and not path.is_file():
            problems.append(f"{path}: not a regular file")
    problems += [why for _, _, bad in walks.values() for why in bad]

    line_hash = None
    line_path = sealed / "PROTOCOL.sha256"
    if not _present(line_path):
        problems.append(f"{line_path}: missing, so {protocol.name} cannot be checked against it")
    elif _regular(line_path) and _regular(protocol):
        line_hash, why = _protocol_line_hash(line_path)
        if why:
            problems.append(f"{line_path}: {why}")
        else:
            try:
                actual = _sha(protocol)
            except OSError as exc:
                problems.append(f"{protocol}: cannot be read ({exc})")
            else:
                if actual != line_hash:
                    problems.append(f"{protocol}: its SHA-256 {actual} is not the one in {line_path.name} ({line_hash})")

    manifest = sealed / "CODE_MANIFEST.sha256"
    if not _present(manifest):
        problems.append(f"{manifest}: missing; the protocol binds the code by this file")
    elif _regular(manifest):
        try:
            _, bad = _manifest_entries(manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError) as exc:
            problems.append(f"{manifest}: cannot be read ({exc})")
        else:
            if bad:
                problems.append(f"{manifest}: line {bad[0]} is not '<sha256>  <path>'" + (f" (and {len(bad) - 1} more)" if len(bad) > 1 else ""))

    registration = sealed / "REGISTRATION.md"
    if _regular(registration):
        try:
            record = _registration().parse_registration(registration)
        except Exception as exc:                   # RegistrationError for a record that is not well formed; OSError for an unreadable one
            problems.append(f"{registration}: {exc}")
        else:
            if line_hash is not None and record["protocol_sha256"] != line_hash:
                problems.append(f"{registration}: names protocol_sha256 {record['protocol_sha256']}, not {line_hash}, the hash in PROTOCOL.sha256")
    elif not _present(registration):
        problems += [f"{sealed / n}: exists, but there is no {registration.name}, the record of the protocol" for n in records]
        problems += [f"{outs / t}: sealed outputs, but there is no {registration}, the record of the protocol" for t in TEST_IDS if has_outputs[t]]

    runs, runs_known = {}, True
    runs_path = sealed / "RUNS.json"
    if _regular(runs_path):
        try:
            runs = json.loads(runs_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:                       # a JSON error and a byte that is not UTF-8 are both ValueErrors
            runs, runs_known = {}, False
            problems.append(f"{runs_path}: not valid JSON ({exc}); it records which sealed tests have run")
        else:
            if not isinstance(runs, dict):
                runs, runs_known = {}, False
                problems.append(f"{runs_path}: must hold a JSON object keyed by test id")
    if _regular(registration) and runs_known:
        for test in TEST_IDS:
            if has_outputs[test] and test not in runs and not _present(sealed / f"RUN_{test}.claim"):
                problems.append(f"{outs / test}: sealed outputs of Test {test}, but {runs_path.name} has no record of it and "
                                f"RUN_{test}.claim does not exist: the package would show outputs of a test that did not start")
    return problems


def sealed_overlap_problems(out_dir, sealed_dir, outputs_dir, pull_folders=None):
    """The package folder (and its ``.partial`` sibling) is never written in, or around, the folder of the protocol, of the sealed outputs
    or of a sealed pull (``pull_folders``: {test: folder}): those are read, and never written. A file added to a pull folder would break
    the manifest check that the sealed run makes of it."""
    out = Path(out_dir).resolve()
    problems = []
    places = [("the folder of the protocol", sealed_dir), ("the folder of the sealed outputs", outputs_dir)]
    places += [(f"the folder of the pull of Test {test}", folder) for test, folder in (pull_folders or {}).items()]
    for what, folder in places:
        real = Path(folder).resolve()
        if out == real or out.is_relative_to(real) or real.is_relative_to(out):
            problems.append(f"{out_dir}: the package folder is in, or around, {what} ({folder}); the package is never written there")
    return problems


def format_protocol(copies, sealed_dir):
    if not copies:
        return f"-- protocol/: none (no PROTOCOL.md in {sealed_dir})"
    return "\n".join([f"-- protocol/ ({len(copies)} files, copied byte for byte)"] + [f"   {rel}" for _, rel in copies])


@dataclass(frozen=True)
class RunState:
    """What protocol/ shows about one sealed test: whether and when it ran, never a result."""
    record: bool                 # RUNS.json holds a record of the test
    claim: bool                  # RUN_<test>.claim exists
    started_utc: str | None      # from the record, else from the claim file
    finished_utc: str | None     # from the record (None until the run finishes)
    not_run: bool                # NOT_RUN.md is among the outputs: the coverage rule failed, and the test gave no result

    @property
    def started(self):
        return self.record or self.claim


@dataclass(frozen=True)
class PullLogState:
    """The pull log of one sealed test in protocol/sealed_pulls/<test>/: which of PULL_LOG_FILES are there, and which are not."""
    included: tuple
    missing: tuple


@dataclass(frozen=True)
class ProtocolState:
    """The state of the protocol/ folder of a package, read from the files in it."""
    files: tuple                 # the names in protocol/ among PROTOCOL_FILES and RECORD_FILES, in that order
    outputs: tuple               # the tests that have a folder in protocol/sealed_outputs/
    protocol_sha256: str         # of protocol/PROTOCOL.md, in lower case
    record: dict | None          # es.registration.parse_registration of protocol/REGISTRATION.md (None: there is no record)
    kind: str | None             # es.registration.registration_kind of the record: "osf", "public_release" or "local"
    runs: dict                   # {"A": RunState, "B": RunState}
    departures: tuple            # paths listed in protocol/CODE_MANIFEST.sha256 whose file in the package differs or is not there
    pull_logs: dict = field(default_factory=dict)   # {test: PullLogState} for each sealed test that has an entry in protocol/PULLS.json


def _json_file(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _text_of(doc, key):
    value = doc.get(key) if isinstance(doc, dict) else None
    return value if isinstance(value, str) and value else None


def _departures(manifest, package_dir):
    """The paths of the code manifest whose file in the package has another SHA-256, or is not there (the meaning of
    ``es.sealed_run.code_departures`` against the package root), in the order of the manifest."""
    entries, bad = _manifest_entries(Path(manifest).read_text(encoding="utf-8"))
    if bad:
        raise ValueError(f"{manifest}: line {bad[0]} is not '<sha256>  <path>'")
    out = []
    for registered, rel in entries:
        posix, target = PurePosixPath(rel), Path(package_dir) / rel
        inside = not posix.is_absolute() and ".." not in posix.parts and target.is_file()
        if (_sha(target) if inside else None) != registered and rel not in out:
            out.append(rel)
    return tuple(out)


def read_protocol_state(package_dir):
    """The state of the protocol/ folder of a package folder, read from what is in it; None when there is no protocol/PROTOCOL.md.
    Raises what ``es.registration.parse_registration`` raises for a record that is not well formed."""
    root = Path(package_dir)
    folder = root / PROTOCOL_DIR
    if not (folder / "PROTOCOL.md").is_file():
        return None
    files = tuple(name for name in PROTOCOL_FILES + RECORD_FILES if (folder / name).is_file())
    reg = _registration()
    record = reg.parse_registration(folder / "REGISTRATION.md") if "REGISTRATION.md" in files else None
    recorded = _json_file(folder / "RUNS.json")
    recorded = recorded if isinstance(recorded, dict) else {}
    outputs = folder / SEALED_OUTPUTS_DIR
    runs = {}
    for test in TEST_IDS:
        entry = recorded.get(test)
        claim = (folder / f"RUN_{test}.claim").is_file()
        started = _text_of(entry, "started_utc") or (_text_of(_json_file(folder / f"RUN_{test}.claim"), "started_utc") if claim else None)
        runs[test] = RunState(test in recorded, claim, started, _text_of(entry, "finished_utc"), (outputs / test / "NOT_RUN.md").is_file())
    departures = _departures(folder / "CODE_MANIFEST.sha256", root) if "CODE_MANIFEST.sha256" in files else ()
    pulled, pull_logs = _json_file(folder / "PULLS.json"), {}
    for test in TEST_IDS if isinstance(pulled, dict) else ():
        if test in pulled:
            there = tuple(n for n in PULL_LOG_FILES if (folder / SEALED_PULLS_DIR / test / n).is_file())
            pull_logs[test] = PullLogState(there, tuple(n for n in PULL_LOG_FILES if n not in there))
    return ProtocolState(files, tuple(t for t in TEST_IDS if (outputs / t).is_dir()), _sha(folder / "PROTOCOL.md"), record,
                         reg.registration_kind(record) if record else None, runs, departures, pull_logs)


# ---------------------------------------------------------------------------------------------------------------
# README.md
# ---------------------------------------------------------------------------------------------------------------
#: The README sentence for a local record, written literally: who wrote the record, and what a reader can and cannot know of the order
#: of events.
LOCAL_RECORD_SENTENCE = ("The record is a local statement that an AI assistant wrote on the author's instruction; the author did not write or "
                         "sign it. No registry, repository or other third party held the protocol before the data of the window were pulled, "
                         "so for the order of events a reader has only that record and the logs of the pull and the run. The protocol is "
                         "published here with the code after the run.")

#: The pieces of the README that depend on the state of protocol/, for a package without a protocol/ folder (state a).
_A_HOLDS = "It holds"
_A_HEADING = "Sealed tests (not run; see the next section):"
_A_STEP11 = (" Steps 12 and 13 run only when a registration record, `W/docs/sealed/REGISTRATION.md`, names the SHA-256 of `PROTOCOL.md`; "
             "none exists.")
_A_LEAD = ("No registration was made. The sealed windows (Test A: 1 July to 30 September 2026; Test B: 1 October to 31 December 2026) have "
           "not been opened. The protocol files are not part of this repository. Because no registration record exists, the sealed pull "
           "and the sealed run refuse to run. The machinery is included as it stands, because the protocol fixes it.")
_A_TAIL = ("The files of this package are copied byte for byte, so the SHA-256 of a file here is that of the file the protocol lists. To "
           "check the code against the manifest of the protocol, from `dart_dev/`: `shasum -a 256 -c ../docs/sealed/CODE_MANIFEST.sha256` "
           "(or `sha256sum -c`), or in Python `es.sealed_run.code_departures(\"../docs/sealed/CODE_MANIFEST.sha256\", \".\")`, which "
           "returns `[]`. The `docs/sealed/` files of the protocol are not part of this repository.")


#: What the README says about the pull logs: where they are, and what each file gives.
PULL_LOGS_SENTENCE = ("The logs of the pulls are in `protocol/sealed_pulls/`: `PULL_LOG.txt` gives the start and finish times, the record the "
                      "pull ran under and the days that were excused; `COVERAGE.csv` and `OVERLAP_CHECK.csv` give the coverage of each node and the "
                      "check of the first days against the files already opened.")


def _names(items):
    """`a`, `b` and `c`."""
    quoted = [f"`{i}`" for i in items]
    return quoted[0] if len(quoted) == 1 else ", ".join(quoted[:-1]) + " and " + quoted[-1]


def _record_paragraph(state):
    """Which kind of record protocol/REGISTRATION.md is, and its UTC time (read with es.registration). Only a local record gets the
    sentence about the order of events; the others say that nothing here checks them against the outside."""
    reg, record = _registration(), state.record
    when = f"{record['registered_utc']} (UTC)"
    if state.kind == "local":
        return f"It is a local record, `{reg.registration_identifier(record)}`, recorded at {when}. {LOCAL_RECORD_SENTENCE}"
    if state.kind == "public_release":
        return (f"It is a public record: it names the tagged release `{record['public_record']}` of the code repository, with the time of "
                f"the release, {when}. Nothing in this repository checks the record against GitHub.")
    ids = ", ".join(f"`{i}`" for i in reg.registration_identifiers(record))
    return f"It names an OSF registration, {ids}, with the time of the record, {when}. Nothing in this repository checks the record against OSF."


def _pull_logs_paragraph(state):
    """What the README says about the logs of the pulls: where they are and what each file gives, and which are not included (the folder of a
    pull is outside the repository, so a log can be missing). Empty when PULLS.json has no entry of a sealed test."""
    if not state.pull_logs:
        return ""
    if not any(log.included for log in state.pull_logs.values()):
        return "The logs of the pulls are not included in this repository."
    text = PULL_LOGS_SENTENCE + " `MANIFEST.sha256` in each folder lists the files of that pull, which are not part of this repository."
    for test in TEST_IDS:
        log = state.pull_logs.get(test)
        if log is None or not log.missing:
            continue
        if not log.included:
            text += f" The pull log of Test {test} is not included."
        else:
            text += f" Of the pull log of Test {test}, {_names(log.missing)} {'is' if len(log.missing) == 1 else 'are'} not included."
    return text


def _run_line(test, run):
    """One line about a sealed test: whether and when it started and finished, from the records. No result is quoted."""
    head = f"- Test {test} ({TEST_WINDOWS[test]}): "
    if not run.started:
        return head + f"not started. There is no record of it in `protocol/RUNS.json`, and there is no `protocol/RUN_{test}.claim`."
    parts = ["started" + (f" at {run.started_utc} (UTC)." if run.started_utc else ".")]
    if run.finished_utc:
        parts.append(f"It finished at {run.finished_utc} (UTC).")
    elif run.record:
        parts.append("`protocol/RUNS.json` gives no finish time.")
    parts.append(("Its record is in `protocol/RUNS.json`" if run.record else "There is no record of it in `protocol/RUNS.json`")
                 + (f", and its claim file is `protocol/RUN_{test}.claim`." if run.claim else f", and there is no `protocol/RUN_{test}.claim`."))
    if run.not_run:
        parts.append(f"The coverage rule failed: `protocol/{SEALED_OUTPUTS_DIR}/{test}/NOT_RUN.md` is there, and the test gave no result.")
    return head + " ".join(parts)


def _readme_parts(state):
    """The pieces of the README that depend on the state of protocol/: ``holds`` (the sentence of "What this is" about what the package
    holds), ``heading`` and ``step11`` (the sealed steps of the run order), ``lead`` (the first paragraphs of "How the protocol binds the
    code") and ``tail`` (its closing paragraph). Three states: no protocol/ folder (``state`` is None: the text of a package without a
    protocol), a protocol and no record (b), and a protocol with its record (c)."""
    if state is None:
        return {"holds": _A_HOLDS, "heading": _A_HEADING, "step11": _A_STEP11, "lead": _A_LEAD, "tail": _A_TAIL}
    has_record = state.record is not None
    holds = 'Apart from the folder `protocol/` (see "How the protocol binds the code"), it holds'
    where = f"The protocol is in `protocol/`, with its files: {_names([n for n in state.files if n in PROTOCOL_FILES])}."
    pulls = " The pull of each window is recorded in `PULLS.json` with its start and finish times." if "PULLS.json" in state.files else ""
    if has_record:
        heading = "Sealed tests (see the next section for the ones that have run):"
        step11 = (" The protocol and its files are copied to `protocol/` in this repository, and so are the records of the sealed tests; the "
                  "code reads `W/docs/sealed/`, not `protocol/`. Steps 12 and 13 run only when a record, `W/docs/sealed/REGISTRATION.md`, names "
                  "the SHA-256 of `PROTOCOL.md`. The record in `protocol/REGISTRATION.md` does.")
        folders = [f"protocol/{SEALED_OUTPUTS_DIR}/{t}/" for t in state.outputs]
        outputs = ("\n\nThe files that the runs wrote are in " + _names(folders) + ". The `.parquet` files of the runs are not part of this "
                   "repository.") if folders else ""
        others = [n for n in state.files if n in RECORD_FILES and n != "REGISTRATION.md"]
        more = f" The records of the pulls and the runs are there too: {_names(others)}." if others else ""
        logs = _pull_logs_paragraph(state)
        lead = (f"{where} The record of the protocol is `protocol/REGISTRATION.md`. {_record_paragraph(state)}{more}{pulls}\n\n"
                f"The sealed windows are Test A ({TEST_WINDOWS['A']}) and Test B ({TEST_WINDOWS['B']}). This is what the records in "
                "`protocol/` show about them (no result is quoted here):\n\n"
                + "\n".join(_run_line(t, state.runs[t]) for t in TEST_IDS) + outputs + (f"\n\n{logs}" if logs else ""))
    else:
        heading = "Sealed tests (not run; see the next section):"
        step11 = (" The protocol and its files are copied to `protocol/` in this repository; the code reads `W/docs/sealed/`, not `protocol/`. "
                  "Steps 12 and 13 run only when a record, `W/docs/sealed/REGISTRATION.md`, names the SHA-256 of `PROTOCOL.md`; none exists.")
        lead = (f"{where} No record of it exists: there is no `protocol/REGISTRATION.md`. The sealed windows (Test A: {TEST_WINDOWS['A']}; "
                f"Test B: {TEST_WINDOWS['B']}) have not been opened. Because no record exists, the sealed pull and the sealed run refuse to "
                "run. The machinery is included as it stands, because the protocol fixes it." + pulls)
    if state.departures:
        listed = ", ".join(f"`{d}`" for d in state.departures)
        match = ("When this package was built, these files listed in `protocol/CODE_MANIFEST.sha256` differed from the file of the same path "
                 f"in this package, or were not in it: {listed}.")
    else:
        match = ("When this package was built, every line of `protocol/CODE_MANIFEST.sha256` matched the file of the same path in this "
                 "package.")
    check_hash = (f"- `shasum -a 256 protocol/PROTOCOL.md` prints `{state.protocol_sha256}`. This is the content of `protocol/PROTOCOL.sha256`"
                  + (" and the `protocol_sha256` of `protocol/REGISTRATION.md`." if has_record else "."))
    check_code = ("- `shasum -a 256 -c protocol/CODE_MANIFEST.sha256` (or `sha256sum -c`) checks the code that the protocol fixes, file by "
                  'file. In Python, `es.sealed_run.code_departures("protocol/CODE_MANIFEST.sha256", ".")` lists the files that differ, and '
                  "returns `[]` when none does.")
    tail = ("To check the protocol" + (", the record" if has_record else "") + " and the code, from the root of this repository "
            "(`dart_dev/` once it is cloned as above):\n\n" + check_hash + "\n" + check_code + "\n\n"
            "The files of this package are copied byte for byte, so the SHA-256 of a file here is that of the file the protocol lists. "
            + match)
    return {"holds": holds, "heading": heading, "step11": step11, "lead": lead, "tail": tail}


TESTS_NOTE_PLACEHOLDER = ("*Which tests need the raw prices, the paper folder or other files that are not in this package: "
                          "to be filled from the smoke check of this copy.*")


def readme_text(plan, env, *, tests_note=None, protocol=None):
    """The README of the package as it is published (Ruling 133: no registration was made, so it names none and has nothing left to fill in
    but the tests note). ``protocol`` is the ProtocolState read from the copies in protocol/ (None: the package has no protocol/ folder,
    and the text is that of a package without a protocol). With a protocol and no record, the README says that the protocol is in
    protocol/, that no record of it exists and that the sealed windows have not been opened. With a record, it says which kind of record
    it is and its UTC time, which tests have run (without a result), and how to check the protocol and the code. It carries no date of
    its own: the same inputs give the same text."""
    versions = "\n".join(f"| {name} | {version} |" for name, version in env["libraries"].items())
    parts = _readme_parts(protocol)
    return f"""# {PACKAGE_NAME}

Code for the paper **{PAPER_TITLE}**, by {AUTHOR} (sole author), written for *Energy Systems*.

## What this is

This is the price-only pipeline of the paper, and nothing else:

- the downloaders of the public day-ahead and real-time prices of four U.S. operators (NYISO, ISO-NE, MISO, CAISO) for a panel of 33 pricing nodes (`v2/ingest_nyiso.py`, `v2/ingest_isone.py`, `v2/ingest_miso.py`, `v2/ingest_caiso.py`) and the settlement legs of CAISO and NYISO (`es/settlement.py`);
- the development pipeline, on data up to 2026-06-30: scores, cutoff policies, economics, drift monitors, inference, and the regeneration of the numbers of the earlier version of the paper;
- the machinery of the two pre-specified sealed tests (Test A, July to September 2026; Test B, October to December 2026): the gate that checks the protocol record (`es.registration`), the sealed pull and the one-shot run;
- the generators of the paper's macros, tables and figures and of the submission package, which read the paper's folder (not part of this package);
- the tests (`tests/es/`).

{parts['holds']} no price data, raw or derived, no outputs, no manuscript source, no notebooks, and no script of the other projects that share the research repository (text extraction, fundamentals ingestion, ERCOT and PJM probes). `src/` is the shared library of that repository, and some of its comments and configuration entries still mention those projects: the copies are not edited.

## Environment

Python {env['python']}, with these exact versions (read with `importlib.metadata`; `requirements.txt` pins them). They are the versions of the environment that produced the results; other versions are untested.

| package | version |
|---|---|
{versions}

## Layout and setup

The code finds its folders from its own location and from `DART_PROJECT_ROOT` (`es/config.py`, `src/paths.py`). It expects a working folder `W`, called `EnergySystems_Revision` in the research repository (one test checks the name), that holds this package under the name `dart_dev` (`src/paths.py` fixes that name: `v2/cell_builder.py` reads `dart_dev/cache/common_panel_groups.json` through it, and the score cache is written to `dart_dev/cache/`), and an empty folder `DART_Spike_v2_Clean` that both modules still check for (a leftover of the research repository):

```
mkdir EnergySystems_Revision && cd EnergySystems_Revision
git clone <repository url> dart_dev
mkdir DART_Spike_v2_Clean
export DART_PROJECT_ROOT="$PWD"
cd dart_dev
python -m pip install -r requirements.txt
```

Raw prices are read from `~/dart-spread-data/`, outputs go to `dart_dev/outputs/es/`, and the protocol files are read from `W/docs/sealed/`. Run every command below from `dart_dev/` with `DART_PROJECT_ROOT` set.

## Where the data come from

All prices come from the public sites of the four operators, without a key or a login. This package holds none of them; its own code re-pulls them.

| series | source | code |
|---|---|---|
| NYISO day-ahead and real-time zonal prices | mis.nyiso.com/public/csv/ (monthly zips) | `v2/ingest_nyiso.py` |
| ISO-NE day-ahead and final real-time hourly prices | www.iso-ne.com/static-transform/csv/histRpts/ | `v2/ingest_isone.py` |
| MISO day-ahead ex-post and final real-time prices | docs.misoenergy.org/marketreports/ | `v2/ingest_miso.py` |
| CAISO day-ahead and real-time prices | oasis.caiso.com (OASIS) | `v2/ingest_caiso.py` |
| CAISO 15-minute FMM prices and NYISO integrated real-time prices (the settlement legs) | oasis.caiso.com, mis.nyiso.com | `es/settlement.py` |

The panel is the 33 nodes of `v2/NODE_MAP.csv`: 10 NYISO zones, 9 ISO-NE locations, 8 MISO hubs and 6 CAISO nodes; the SPP rows of that file are not part of the panel. CAISO's OASIS keeps a moving window of history, so its oldest requests may no longer be answered and a re-pull may start later than the paper's CAISO series. The development loaders refuse dates after 2026-06-30; the sealed windows open only through the gate that checks the protocol record (`es.registration`).

## Run order

Development period (data to 2026-06-30). Each module's docstring says what it reads and writes.

1. Prices. Run the four downloaders (`python v2/ingest_nyiso.py --start 2023-01 --end 2024-12 --root DIR`; the other three take `--start` and `--end` as dates, and `--help` lists the options) into `~/dart-spread-data/backfill_2023_2024/` for 2023 to 2025 and `~/dart-spread-data/holdout_2026H1/` for 2026-H1 (`DEV_DATA_DIRS` in `es/config.py`). Then the settlement legs, from Python: `es.settlement.pull_caiso_fmm(start, end, root)` with `root` = `~/dart-spread-data/caiso_fmm_dev/` and `es.settlement.pull_nyiso_integrated(start_month, end_month, root)` with `root` = `~/dart-spread-data/nyiso_rtint_dev/`; each needs a root that does not exist yet and refuses an end after 2026-06-30.
2. `python -m es.run_legacy`: regenerates the numbers of the earlier version of the paper and compares them with the printed ones (`es/published_v3.csv`).
3. `python -m es.run_analogs`: quarterly analogues, placebo pairings, the choice of eta, and the size-and-power simulation (`es/simulate.py`). `--stage analogs` and `--stage sim` run its two stages one at a time.
4. `python -m es.run_economics`: the development economic analysis. `python -m es.run_economics_block --block DART3` repeats it with the plain DART lags as the alert model (an exploratory secondary analysis on the development quarters; its files go to `outputs/es/economics_block/DART3/`).
5. `python -m es.run_labels_pr`: precision and recall, and the 2x2 label control.
6. `python -m es.run_monitors`: the label-free drift monitors.
7. `python -m es.run_inference`: mixed model, leave-one-operator-out, multiplicity (needs step 3).
8. `python -m es.run_digest`: one Markdown digest of all outputs.
9. `python -m es.run_driftpath`: the alert rate of the fixed and the rolling cutoff through the two main deployment windows, week by week and over trailing windows (`WINDOW_WEEKS` weeks, the length of the rolling cutoff's window), written to `outputs/es/paper/drift_path.csv` and `outputs/es/paper/drift_summary.csv`. It needs the cached scores (it reads the score cache and never refits a model: with a cold cache it stops) and checks its counts against the outputs of steps 2 and 4. The macros and `es.paper_figures` read its two files, so it runs before step 10.
10. The paper generators, which need the paper's folder (`paper/` beside `dart_dev/`): `python -m es.impl_table`, `python -m es.paper_macros`, `python -m es.paper_tables`, `python -m es.paper_figures`, `python -m es.paper_lint`, and `python -m es.paper_package` for the journal upload set.

{parts['heading']}

11. `python -m es.protocol` writes the protocol package into `W/docs/sealed/` (pre-pull cutoffs, protocol inputs, `CODE_MANIFEST.sha256`, `code_archive.tar.gz`, `ENV_PINS.json`, `PROTOCOL.md` and `PROTOCOL.sha256`). It renders the protocol from the design spec in `W/docs/superpowers/specs/`, which is not part of this package.{parts['step11']}
12. `es.sealed_pull.pull_sealed("A", "<protocol sha256>")` downloads the July to September 2026 window, once, and only when `es.registration.verify_registration` passes.
13. `es.sealed_run.run_sealed(es.sealed_run.default_config("A"), "<protocol sha256>")` runs Test A once. Test B (October to December 2026) is the same call with `"B"`; it pools Test A's stored rows and needs Test A's completed run.

## How the protocol binds the code

{parts['lead']}

The sealed tests are bound to the code by content:

- `python -m es.protocol` lists the code that the protocol fixes in `CODE_MANIFEST.sha256`, one `sha256  path` line per file, paths relative to `dart_dev/`: the import closure of `es.sealed_run`, `es.sealed_pull`, `es.registration` and `es.protocol`, traced statically, and the two data files that closure opens by a fixed path (`v2/NODE_MAP.csv` and `cache/common_panel_groups.json`). `PROTOCOL.md` prints the SHA-256 of the manifest, of `code_archive.tar.gz`, of `ENV_PINS.json`, of the pre-pull cutoffs and of the protocol inputs.
- `es.registration.verify_registration` opens a sealed window only when `W/docs/sealed/REGISTRATION.md` exists and names the SHA-256 of the unchanged `PROTOCOL.md`, and `es.registration.verify_bound` checks that every bound file matches the hash the protocol prints.
- `es.sealed_run.code_departures(manifest, root)` lists the files that differ from the manifest of the protocol. The sealed run is refused when there is one, unless the departure is recorded (`allow_departure=True`).

{parts['tail']}

## Tests

```
pytest tests/es -m "not slow" -q
```

The `slow` marker is for the tests that run the full 33-node pipeline. No test reaches a network (`tests/es/conftest.py` closes the sockets).

{tests_note.strip() if tests_note else TESTS_NOTE_PLACEHOLDER}

## Verifying this package

`MANIFEST.sha256` lists every file of the package except itself, one `sha256  path` line each: `shasum -a 256 -c MANIFEST.sha256` (or `sha256sum -c MANIFEST.sha256`), or `python -m es.public_package --verify .` from `dart_dev/`, which also reports files that the manifest does not list (for example `__pycache__` left by running the tests inside the package folder).

## Licence and citation

MIT (see `LICENSE`), copyright {LICENSE_YEAR} {AUTHOR}.

Cite the paper: {AUTHOR}, *{PAPER_TITLE}* (DOI to be added).
"""


# ---------------------------------------------------------------------------------------------------------------
# The manifest, its verification, and the closure check of the finished package
# ---------------------------------------------------------------------------------------------------------------
def _package_files(root):
    """The files of a package folder (a repository's own .git folder is not part of the package)."""
    root = Path(root)
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*")
                  if (p.is_file() or p.is_symlink()) and ".git" not in p.relative_to(root).parts)


def manifest_text(root, rels):
    return "".join(f"{_sha(Path(root) / rel)}  {rel}\n" for rel in sorted(rels))


def verify_manifest(package_dir):
    """Problems of a package folder against its MANIFEST.sha256: a missing or changed file, a file the manifest does not list."""
    root = Path(package_dir)
    manifest = root / "MANIFEST.sha256"
    if not manifest.is_file():
        return ["MANIFEST.sha256: missing"]
    problems, listed = [], {}
    for n, line in enumerate(manifest.read_text().splitlines(), 1):
        digest, sep, rel = line.partition("  ")
        if not sep or not rel or not re.fullmatch(r"[0-9a-f]{64}", digest):
            problems.append(f"MANIFEST.sha256:{n}: not a 'sha256  path' line")
        else:
            listed[rel] = digest
    for rel, digest in sorted(listed.items()):
        path = root / rel
        if not path.is_file():
            problems.append(f"{rel}: missing")
        elif _sha(path) != digest:
            problems.append(f"{rel}: differs from the manifest")
    problems += [f"{rel}: not in the manifest" for rel in sorted(set(_package_files(root)) - set(listed) - {"MANIFEST.sha256"})]
    return problems


def closure_problems(package_dir, entry_points, plan):
    """The import closure of the entry points, traced inside the finished package, must be the planned set of files."""
    root = Path(package_dir)
    absent = [f"{name}: an entry point that is not a module inside the package" for name in entry_points
              if not _is_project_module(root, name)]
    if absent:
        return absent
    try:
        modules, _ = _protocol().import_closure(root, roots=tuple(entry_points))
    except (SyntaxError, UnicodeDecodeError) as exc:
        return [f"a module inside the package cannot be parsed: {exc}"]
    found = {Path(f).relative_to(root).as_posix() for f in modules.values()}
    listed = set(plan.code)
    return ([f"{rel}: in the list, but not reached from the entry points inside the package" for rel in sorted(listed - found)]
            + [f"{rel}: reached from the entry points inside the package, but not in the list" for rel in sorted(found - listed)])


# ---------------------------------------------------------------------------------------------------------------
# The build
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class Staged:
    plan: Plan
    out_dir: Path
    n_files: int
    requirements: dict
    warnings: list
    protocol: ProtocolState | None = None      # the state of protocol/, read from the copies; None when the package has no protocol/ folder


def _out_dir_problems(root, out, rebuild):
    root_real, out_real = Path(root).resolve(), Path(out).resolve()
    if out_real == root_real or out_real.is_relative_to(root_real):
        return [f"{out}: the package folder is inside the dev tree"]
    if root_real.is_relative_to(out_real):
        return [f"{out}: the package folder contains the dev tree"]
    if out.is_symlink() or (out.exists() and not out.is_dir()):
        return [f"{out}: exists and is not a folder"]
    if out.is_dir() and any(out.iterdir()):
        if not rebuild:
            return [f"{out}: already exists and is not empty (--rebuild replaces a previous build)"]
        if not ((out / "MANIFEST.sha256").is_file() and (out / "README.md").is_file()):
            return [f"{out}: exists but is not a previous build (no MANIFEST.sha256 and README.md), so it is not deleted"]
        if (out / ".git").exists():
            return [f"{out}: holds a .git folder, which a rebuild would delete; publish from a copy of the package, or build "
                    "into another folder with --out"]
    return []


def build_package(dev_root, out_dir, *, entry_points=ENTRY_POINTS, data=None, rebuild=False, env=None, tests_note=None,
                  sealed_dir=None, sealed_outputs_dir=None, pull_dirs=None) -> Staged:
    """Stage the package. ``sealed_dir`` and ``sealed_outputs_dir`` are where the protocol and its records, and the outputs of the
    sealed tests, are (default: those of the real tree, found from ``dev_root`` by ``sealed_locations``; the tests pass temporary
    folders). With a PROTOCOL.md there, the package gets a protocol/ folder (see the module docstring); without one it has none.
    ``pull_dirs`` ({test id: folder}) gives the pull folders whose logs go into protocol/sealed_pulls/; without it they are found from
    the ``root`` of each entry of PULLS.json (see ``pull_log_copies``)."""
    root, out = Path(dev_root), Path(out_dir)
    plan = plan_files(root, entry_points=entry_points, data=data)
    sealed, sealed_out = sealed_locations(root, sealed_dir, sealed_outputs_dir)
    pull_copies, pull_warnings, pull_folders = pull_log_copies(sealed, pull_dirs)
    problems = (check_plan(plan) + _out_dir_problems(root, out, rebuild) + sealed_overlap_problems(out, sealed, sealed_out, pull_folders)
                + protocol_problems(sealed, sealed_out))
    partial = out.with_name(out.name + ".partial")
    if partial.exists():
        problems.append(f"{partial}: left by an interrupted build; delete it")
    if problems:
        raise PackageRefused(problems)
    env = env or environment()
    pairs = [(root / rel, rel) for rel in plan.files] + protocol_copies(sealed, sealed_out) + pull_copies   # (source, path in the package)
    source_sha = {rel: _sha(src) for src, rel in pairs}
    large = []
    try:
        partial.mkdir(parents=True)
        for src, rel in pairs:
            dst = partial / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            shutil.copymode(src, dst)
            if _sha(src) != source_sha[rel]:
                raise RuntimeError(f"{rel} changed while the package was being built; run the build again")
            if _sha(dst) != source_sha[rel]:
                raise RuntimeError(f"the copy of {rel} is not byte-identical to its source")
            if rel.startswith(f"{PROTOCOL_DIR}/") and dst.stat().st_size > LARGE_FILE_BYTES:
                large.append(f"{rel} is {dst.stat().st_size / 1024 ** 2:.1f} MB, over {LARGE_FILE_BYTES / 1024 ** 2:g} MB")
        if (partial / PROTOCOL_DIR).exists():              # the copies, not the sources, must hold together: nothing changed in between
            bad = protocol_problems(partial / PROTOCOL_DIR, partial / PROTOCOL_DIR / SEALED_OUTPUTS_DIR)
            if bad:
                raise RuntimeError("the protocol/ folder of the package does not hold together (did its sources change while the package "
                                   "was being built?): " + "; ".join(bad))
        state = read_protocol_state(partial)
        runtime, tests_only = _extra_requirements(root, plan)
        requirements = requirements_text(env, runtime, tests_only)
        (partial / "README.md").write_text(readme_text(plan, env, tests_note=tests_note, protocol=state))
        (partial / "requirements.txt").write_text(requirements)
        (partial / "LICENSE").write_text(license_text())
        bad = closure_problems(partial, plan.entry_points, plan)
        if bad:
            raise RuntimeError("the package is not closed under its imports: " + "; ".join(bad))
        (partial / "MANIFEST.sha256").write_text(manifest_text(partial, _package_files(partial)))
        bad = verify_manifest(partial)
        if bad:
            raise RuntimeError("the package does not verify against its manifest: " + "; ".join(bad))
        n_files = len(_package_files(partial))
        if out.exists():
            shutil.rmtree(out)
        partial.rename(out)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    warnings = [f"{name} ({rel}) is imported by tests/es and is not in the package" for name, rel in import_gap(root, plan).items()]
    warnings += large + pull_warnings
    if state is not None and state.kind == "local" and not any(run.started for run in state.runs.values()):
        warnings.append("the README of a local record says that the protocol is published with the code after the run, but no sealed test has "
                        "started (no record in RUNS.json and no RUN_<test>.claim)")
    if state is not None and state.departures:
        warnings.append("protocol/CODE_MANIFEST.sha256 lists files that differ from the file of the same path in this package, or are not in "
                        "it: " + ", ".join(state.departures))
    return Staged(plan, out, n_files, dict(re.findall(r"^([A-Za-z0-9_.\-]+)==(\S+)", requirements, flags=re.M)), warnings, state)


# ---------------------------------------------------------------------------------------------------------------
# The scans: they report and never edit
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Hit:
    scan: str          # credentials | paths | names | context
    category: str
    file: str
    line: int
    text: str

    def format(self):
        return f"{self.file}:{self.line}: [{self.category}] {self.text}"


def _clip(line, width=160):
    line = line.strip()
    return line if len(line) <= width else line[:width - 3] + "..."


#: Patterns that name a place, an account or a person are assembled from pieces, so that this file does not hold them in one
#: piece and the scans do not flag the scanner.
_CREDENTIALS = (
    ("email", re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}")),
    ("aws access key id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{20,})")),
    ("secret key (sk-)", re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}")),
    ("slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}")),
    ("google api key", re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}")),
    ("private key block", re.compile(r"-----BEGIN (?:[A-Z]+ )*PRIVATE KEY-----")),
    ("bearer token", re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{16,}")),
    ("credentials in a url", re.compile(r"://[^/\s:@'\"]+:[^/\s@'\"]+@")),
    ("secret in a query string", re.compile(
        r"[?&](?:api[_-]?key|apikey|key|token|access[_-]?token|password|passwd|pwd|secret)=[^&\s'\"{}]{6,}", re.I)),
    ("secret assignment", re.compile(
        r"""\b[\w.]*(?:api[_-]?key|apikey|secret|token|passw(?:or)?d|pwd|access[_-]?key|private[_-]?key)\w*["']?\s*[:=]\s*"""
        r"""[rbuf]{0,2}(["'])[^"'\n]{6,}\1""", re.I)),
)
_PATHS = (
    ("home path", re.compile(r"/Users/[A-Za-z0-9_.\-]+|(?<![\w.])/home/[a-z_][a-z0-9_\-]*/|[A-Za-z]:\\Users\\[^\\\s]+")),
    ("author username", re.compile("erce" + "erez", re.I)),
    ("cloud drive path", re.compile("|".join([
        "Cloud" + "Storage", "Google" + "Drive", "Google " + "Drive", "/content/" + "drive", "My" + "Drive",
        "shortcut-targets-" + "by-id", "drive" + r"\.google\.com", "docs" + r"\.google\.com"]), re.I)),
    ("scratch path", re.compile("/private" + "/tmp/|/var/" + "folders/|/tmp/" + "claude")),
    ("institutional id", re.compile(r"\bee" + r"2625\b|columbia" + r"\.edu", re.I)),
)
#: Informational: references that a reader of the public package cannot follow, and the other projects of the research repository.
_CONTEXT = (
    ("unrelated project", re.compile(r"\bA" + r"FD\b|\bafd_|Area Forecast|ERCOT|\bPJM\b|\bSPP\b|eia_?930|fundamentals|\.ipynb|Colab")),
    ("internal reference", re.compile(
        r"\b[Cc]ontroller\b|\bRulings?\s+\d|\bplan\s+2[AB]\b|FORENSICS|\breview round\b|\bgate G\d\b|superpowers|\bOpus\b|\bCodex\b"
        r"|\bClaude\b")),
)
_NAME_PATTERNS = (
    ("citation", re.compile(r"\b[A-Z][A-Za-z'\u2019\-]+ et al\b")),
    ("title", re.compile(r"\b(?:Dr|Prof|Professor|Mr|Mrs|Ms|Mx)\.? [A-Z][a-z]+")),
    ("attribution", re.compile(
        r"(?i:\b(?:thanks to|thank you|courtesy of|credit(?:ed)? to|suggested by|asked by|requested by|reported by|reviewed by"
        r"|written by|authored by|co-authored-by|signed-off-by|agreed with)\b[: ]+)[A-Z][a-z]+(?: [A-Z][a-z]+)?")),
    ("author field", re.compile(
        r"(?i:__author__|__maintainer__|@author|\bauthors?\s*[:=]|\bmaintainer\s*[:=]|\bcontact\s*[:=])\s*[\"']?[A-Z][a-z]+")),
    ("handle", re.compile(r"github\.com/[A-Za-z0-9\-]+")),
)
#: "Name 1994", "Name and Name (1994)": a citation unless the first word is a month or an ordinary word ("The 2026 sample").
_AUTHOR_YEAR = re.compile(r"\b([A-Z][A-Za-z'\u2019\-]+)(?: (?:and|&) [A-Z][A-Za-z'\u2019\-]+)? \(?(?:19|20)\d\d[a-z]?\b")
_CAPS_PAIR = re.compile(r"\b([A-Z][a-z]{2,}(?:['\u2019\-][A-Z][a-z]+)?)(?: [A-Z]\.)? ([A-Z][a-z]{2,}(?:['\u2019\-][A-Z][a-z]+)?)\b")
#: Capitalised words that are not part of a person's name.
_NOT_NAMES = frozenset("""January February March April May June July August September October November December Jan Feb Mar
Apr Jun Jul Aug Sep Sept Oct Nov Dec Monday Tuesday Wednesday Thursday Friday Saturday Sunday NYISO ISONE MISO CAISO""".split())


def _fold(text):
    """Lower case without accents; the dotless and the dotted i of Turkish are mapped to i."""
    text = text.replace("\u0131", "i").replace("\u0130", "I")
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c)).casefold()


def _name_matchers(names):
    out = []
    for name in names:
        tokens = _fold(name).split()
        if len(tokens) > 1:
            out.append(re.compile(r"(?<![a-z])" + r"[^a-z]+".join(map(re.escape, tokens)) + r"(?![a-z])"))
        out.extend(re.compile(r"(?<![a-z])" + re.escape(t) + r"(?![a-z])") for t in tokens if len(t) >= 4)
    return out


def _prose_lines(rel, text):
    """(line, text) of the comments and strings of a Python file, one entry per line; every line of any other file."""
    if not rel.endswith(".py"):
        return list(enumerate(text.splitlines(), 1))
    kinds = {tokenize.COMMENT, tokenize.STRING, getattr(tokenize, "FSTRING_MIDDLE", -1)}
    out = []
    try:
        for tok in tokenize.generate_tokens(StringIO(text).readline):
            if tok.type in kinds:
                out.extend((tok.start[0] + k, piece) for k, piece in enumerate(tok.string.splitlines() or [tok.string]))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return list(enumerate(text.splitlines(), 1))
    return out


def _words(text):
    return set(re.findall(r"[a-z]{3,}", text))


def _is_word(token, known):
    """A word of ordinary use: every part of it (split at hyphens and apostrophes) is a month or day name, or is seen in lower
    case in the scanned files."""
    return all(part in _NOT_NAMES or part.lower() in known for part in re.split(r"['\u2019\-]", token) if part)


def _candidate_pair(chunk, known):
    """Two capitalised words, at least one of which is never seen in lower case in the scanned files."""
    for m in _CAPS_PAIR.finditer(chunk):
        a, b = m.group(1), m.group(2)
        if a in _NOT_NAMES or b in _NOT_NAMES:
            continue
        if _is_word(a, known) and _is_word(b, known):
            continue
        return True
    return False


def _is_citation(chunk, known):
    return any(not _is_word(m.group(1), known) for m in _AUTHOR_YEAR.finditer(chunk))


def scan_text(rel, text, names=(), known_words=None):
    """The hits of one file's text. ``names``: people to look for in comments and strings (case, accents and the dotless i do
    not matter; a surname alone counts from four letters). ``known_words``: the lower-case words of the scanned corpus, which
    separate a name from two ordinary capitalised words (default: the words of this text)."""
    lines = text.splitlines()
    hits = []

    def add(scan, category, no):
        hits.append(Hit(scan, category, rel, no, _clip(lines[no - 1])))

    for no, line in enumerate(lines, 1):
        for scan, rules in (("credentials", _CREDENTIALS), ("paths", _PATHS), ("context", _CONTEXT)):
            for category, rx in rules:
                if rx.search(line):
                    add(scan, category, no)
    matchers = _name_matchers(names)
    known = _words(text) if known_words is None else known_words
    for no, chunk in _prose_lines(rel, text):
        folded = _fold(chunk)
        if any(rx.search(folded) for rx in matchers):
            add("names", "name given", no)
        for category, rx in _NAME_PATTERNS:
            if rx.search(chunk):
                add("names", category, no)
        if _is_citation(chunk, known):
            add("names", "citation", no)
        if _candidate_pair(chunk, known):
            add("names", "candidate", no)
    return sorted(set(hits), key=lambda h: (h.line, h.scan, h.category))


def scan_files(dev_root, rel_paths, names=()):
    """Scan the listed files (bytes decoded as UTF-8, undecodable bytes replaced); hits ordered by file and line."""
    root = Path(dev_root)
    texts = {rel: (root / rel).read_bytes().decode("utf-8", errors="replace") for rel in rel_paths}
    known = set()
    for text in texts.values():
        known |= _words(text)
    hits = []
    for rel in sorted(texts):
        hits.extend(scan_text(rel, texts[rel], names=names, known_words=known))
    return hits


SCAN_ORDER = ("credentials", "paths", "names", "context")


def format_hits(hits):
    if not hits:
        return "(none)"
    out = []
    for scan in SCAN_ORDER:
        mine = [h for h in hits if h.scan == scan]
        if mine:
            out.append(f"== {scan}: {len(mine)} hit(s)")
            out.extend(h.format() for h in mine)
    return "\n".join(out)


def summarize_hits(hits):
    counts = {}
    for h in hits:
        counts[(h.scan, h.category)] = counts.get((h.scan, h.category), 0) + 1
    return "\n".join(f"   {scan:12} {category:28} {n:5}" for (scan, category), n in sorted(
        counts.items(), key=lambda kv: (SCAN_ORDER.index(kv[0][0]), kv[0][1]))) or "   (no hits)"


# ---------------------------------------------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------------------------------------------
def _read_lines(path):
    return [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip() and not ln.lstrip().startswith("#")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stage the public code package (spec I9). Writes one folder; uploads nothing.")
    ap.add_argument("--dev-root", default=str(DEV_ROOT))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--entry", action="append", help="entry-point module, repeatable (default: the entry points of spec I9)")
    ap.add_argument("--data", action="append", help="data file read by a fixed path, repeatable (default: es.protocol's list "
                                                    "and es/published_v3.csv)")
    ap.add_argument("--dry-run", action="store_true", help="print the list, the refusals and the scans; write nothing")
    ap.add_argument("--rebuild", action="store_true", help="replace a previous build of the package folder")
    ap.add_argument("--tests-note-file", help="Markdown for the README's Tests section: which tests need data")
    ap.add_argument("--name", action="append", default=[], help="a person's name to look for in comments and strings")
    ap.add_argument("--names-file", help="one name per line")
    ap.add_argument("--scan-report", help="write the full scan report here (outside the package)")
    ap.add_argument("--verify", metavar="DIR", help="verify a package folder against its MANIFEST.sha256 and stop")
    ap.add_argument("--sealed-dir", help="the folder of the protocol and its records (default: docs/sealed beside the dev root)")
    ap.add_argument("--sealed-outputs-dir", help="the folder of the outputs of the sealed tests, one folder per test (default: "
                                                 "outputs/es/sealed in the dev root)")
    args = ap.parse_args(argv)

    if args.verify:
        problems = verify_manifest(args.verify)
        print("\n".join(problems) if problems else f"{args.verify}: every file matches MANIFEST.sha256, and there is no other file")
        return 1 if problems else 0

    root, out = Path(args.dev_root), Path(args.out)
    try:
        plan = plan_files(root, entry_points=tuple(args.entry) if args.entry else ENTRY_POINTS, data=args.data)
    except PackageRefused as exc:
        print(f"REFUSED: {exc}")
        return 2
    print(format_plan(plan))
    sealed, sealed_out = sealed_locations(root, args.sealed_dir, args.sealed_outputs_dir)
    pull_copies, pull_warnings, pull_folders = pull_log_copies(sealed)
    print(format_protocol(sorted(protocol_copies(sealed, sealed_out) + pull_copies, key=lambda c: c[1]), sealed))
    problems = (check_plan(plan) + _out_dir_problems(root, out, args.rebuild) + sealed_overlap_problems(out, sealed, sealed_out, pull_folders)
                + protocol_problems(sealed, sealed_out))
    if problems:
        print("REFUSED, nothing was written:\n  " + "\n  ".join(problems))
        return 2

    names = list(args.name) + (_read_lines(args.names_file) if args.names_file else [])
    hits = scan_files(root, plan.files, names=names)
    shown = [h for h in hits if h.scan in ("credentials", "paths") or (h.scan == "names" and h.category != "candidate")]
    print("\nscans of the listed files (reported, never edited):\n" + summarize_hits(hits))
    print(format_hits(shown))
    if args.scan_report:
        Path(args.scan_report).write_text(format_hits(hits) + "\n")
        print(f"full scan report: {args.scan_report}")
    gap = import_gap(root, plan)
    if gap:
        print("tests import project modules that the package would lack: " + ", ".join(f"{n} ({f})" for n, f in gap.items()))

    if args.dry_run:
        for warning in pull_warnings:                        # a build prints them with its other warnings
            print(f"warning: {warning}")
        print("\ndry run: nothing written")
        return 0
    note = Path(args.tests_note_file).read_text() if args.tests_note_file else None
    staged = build_package(root, out, entry_points=plan.entry_points, data=plan.data, rebuild=args.rebuild, tests_note=note,
                           sealed_dir=sealed, sealed_outputs_dir=sealed_out)
    print(f"\nwrote {staged.out_dir}: {staged.n_files} files (byte-identical copies, README.md, requirements.txt, LICENSE, "
          "MANIFEST.sha256)")
    print("pins: " + ", ".join(f"{k}=={v}" for k, v in staged.requirements.items()))
    for warning in staged.warnings:
        print(f"warning: {warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
