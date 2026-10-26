"""The protocol package of the sealed tests (plan 2A, Task 2A-4; spec v3 section 6.5 steps 3 and 4; controller Rulings
28, 29, 30, 60, 77, 82, 94, 104, 105, 106, 107, 109, 110, 111, 112, 116, 118, 119, 120, 122, 140 and 141).

``main()`` writes, into docs/sealed/ (nothing else, and nothing once REGISTRATION.md exists):

- ``pre_pull_cutoffs.csv``: the per-node cutoffs of the confirmatory pipelines on the sealed analyses, fitted on
  development data (``es.cutoffs.cutoff_table``, Ruling 30);
- ``protocol_inputs.json``: a byte-for-byte copy of outputs/es/analogs/protocol_inputs.json, refused unless its stage is
  "complete" and its family holds H-E, H-B1 and H-B2 (Ruling 77);
- ``CODE_MANIFEST.sha256`` (one ``sha256  relpath`` line per file of ``code_files``) and ``code_archive.tar.gz`` (a
  deterministic archive of the same files): the code is identified by content (Ruling 28); if HEAD is clean for those
  files it also records the commit and writes ``code.bundle``. The files are the import closure of the sealed path
  (controller Ruling 107(2)): ``es.sealed_run``, ``es.sealed_pull``, ``es.registration`` and this module, and what they
  import, traced statically; and, as roots of their own (Ruling 118, Seat C1, C1-M1), ``es.run_analogs`` and
  ``es.simulate``, which computed the registered family and the simulation rows, so that the archive can regenerate
  ``protocol_inputs.json``; plus the data files they open by a fixed path. Tests, the paper and reporting modules,
  the other development runners and outputs are not in it, so editing them is not a departure from the registered code;
- ``ENV_PINS.json``: the interpreter's path and the versions of python, numpy, pandas, pyarrow, xgboost, scikit-learn,
  scipy and statsmodels (Ruling 109(1)), which the sealed run compares with its own environment before its lock;
- ``PROTOCOL.md`` and ``PROTOCOL.sha256``, rendered by ``render_protocol`` from the spec, the code, the inputs and
  the knowledge statement;
- ``REGISTRATION.template.md``, which is turned into REGISTRATION.md when the protocol is recorded: as an OSF registration,
  a public release of the code repository or the author's local record (Ruling 140).

``render_protocol(ctx)`` reads no data file (it reads the source of ``es.sealed_run`` once, for the vocabulary of the
run statuses, which that module keeps as STATUS_* constants or as literals, and quotes its other constants by name: the
words of the decision, the limits of a run and the claim file; a constant that is gone stops the protocol). ``ctx``
holds: ``inputs`` (the parsed protocol_inputs.json), ``versions``,
``sha`` (``manifest``, ``archive``, ``cutoffs``, ``inputs`` and ``env``: SHA-256 of the bound files; ``env``, the
ENV_PINS.json file, is optional), ``git`` (None, or
``{"commit": ..., "bundle_sha256": ..., "not_in_commit": [...]}``), ``knowledge`` (the knowledge statement's text, or
None), ``spec`` (the design spec's text) and ``forensics`` (accepted and not used: FORENSICS is cited by section, and
its facts reach the protocol through the knowledge statement). Optional: ``author``, ``date``, ``cutoff_summary``
(per analysis and block: nodes and the two cutoff orderings), ``code`` (``code_summary``: the closure's files and their
counts), ``knowledge_source``.

The sealed run binds PROTOCOL.md to four files through ``es.registration.verify_bound`` (the pre-pull cutoffs, the
protocol inputs, the code manifest and ENV_PINS.json, Ruling 109(1)): the protocol must print a line
``<file name> SHA-256: <hash>`` for each (Part F). Nothing here is typed that the code holds: fees, windows, seeds,
bootstrap settings, monitor settings, hyperparameters and the registration rule are read from their modules and from
the inputs when the protocol is rendered."""
from __future__ import annotations

import ast
import datetime as dt
import functools
import gzip
import importlib
import inspect
import json
import math
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path

from . import registration as _registration
from . import sealed_run as _sealed_run
from . import secondary as _secondary
from .analyses import ANALOGS, SEALED
from .blocks import BLOCKS, PAPER_BLOCK_NAMES
from .config import ALPHA, DEV_ROOT, OUT_DIR, REVISION_ROOT
from .cutoffs import CUTOFF_BLOCKS, cutoff_table
from .measures import E1_KEYS, LAMS, STRATA_DECILE, STRATA_PRIMARY
from .policies import p4_rolling, p5_self_correcting
from .positions import CALIBRATED_FEES, UNIFORM_FEES
from .registration import sha256_file
from .run_economics import E4_COMPARATORS, MEAN_BLOCK, N_REP, SEED
from .run_monitors import MONITOR_DAYS, MONITORS, N_FOLDS
from .scores import MIN_POSITIVES, logit_factory, score_analysis, xgb_factory
from .sealed_pull import CHUNK_ATTEMPTS, RT_INTERVALS_PER_HOUR, SEALED_WINDOWS, expected_hours
from .sealed_run import (FAMILY_ORDER, HB_PAIRS, LOGIT_BLOCKS, MIN_USABLE_NODES, SEEDS, default_config, run_sealed,
                         score_keys)

SPEC_REL = "docs/superpowers/specs/2026-10-22-economic-value-sealed-q3-design.md"
FORENSICS_REL = "docs/forensics/FORENSICS.md"
#: Where the ``[AUTHOR: ...]`` marker of a missing knowledge statement points.
MISSING_KNOWLEDGE = ("[AUTHOR: fill docs/sealed/KNOWLEDGE_STATEMENT.md from "
                     "docs/sealed/KNOWLEDGE_STATEMENT_TEMPLATE.md]")
#: The data files that the closure's modules open by a fixed path, read off those modules (not guessed; a test scans the
#: closure for fixed file names and fails on one that is neither listed here nor in tests' NOT_CODE): ``v2/NODE_MAP.csv``
#: (``v2.run_decomposition`` line 36 and ``es.sealed_pull`` ``_node_map_nodes``) and ``cache/common_panel_groups.json``
#: (``v2.cell_builder`` ``g6_features``, line 38). Parquet caches, run records and outputs are data, not code.
OPENED_DATA_FILES = ("v2/NODE_MAP.csv", "cache/common_panel_groups.json")
#: Versions the protocol records, as (label, importable module). python comes from sys.version. The libraries of
#: ENV_PINS.json are es.sealed_run's own (``ENV_LIBRARIES``), so that the file written here and the comparison the run
#: makes against it cannot drift apart; matplotlib is recorded in the protocol (Part I) only.
VERSION_MODULES = (*_sealed_run.ENV_LIBRARIES, ("matplotlib", "matplotlib"))
#: The libraries whose versions ENV_PINS.json binds, with the interpreter's path (Ruling 109(1)); matplotlib is recorded
#: in the protocol (Part I) but does not enter the fitted numbers of the sealed run, so it is not pinned.
ENV_PIN_LIBRARIES = ("python", *(name for name, _ in _sealed_run.ENV_LIBRARIES))
#: The keys of protocol_inputs.json that ``main`` insists on (those the protocol prints in full).
REQUIRED_INPUT_KEYS = ("eta", "family", "rule", "phi", "simulation", "placebo", "analog_selc", "rho_star")


# ---------------------------------------------------------------------------------------------------------------
# The verified pieces (plan 2A, Task 2A-4 Step 3, executed against the real spec, FORENSICS.md and es/*.py on 23 Oct)
# ---------------------------------------------------------------------------------------------------------------
def section(markdown, heading):
    """The text from the line that is ``heading`` or starts with ``heading`` + " ", up to the next heading of
    the same or a higher level (so "## 5c." in FORENSICS.md ends at "## 5b.", and "### 6.1" at "### 6.2")."""
    lines = markdown.splitlines(keepends=True)
    level = len(heading) - len(heading.lstrip("#"))
    start = next(i for i, line in enumerate(lines) if line.rstrip("\n") == heading or line.startswith(heading + " "))
    stop = next((j for j in range(start + 1, len(lines)) if re.match(r"^#{1,%d} " % level, lines[j])), len(lines))
    return "".join(lines[start:stop])


_ITEM = re.compile(r"^(\d+)\. ")


def list_item(markdown, heading, n):
    """Item ``n`` of the top-level numbered list under ``heading``: its first line and every following line that
    is blank or indented, up to the next top-level item or heading. Spec section 5's items 1 (inference) and 2
    (multiplicity) are list items under "## 5.", not headings, so section() cannot reach them."""
    body = section(markdown, heading).splitlines(keepends=True)
    start = next(i for i, line in enumerate(body) if (m := _ITEM.match(line)) and int(m.group(1)) == n)
    out = [body[start]]
    for line in body[start + 1:]:
        if line.strip() and not line.startswith((" ", "\t")):
            break
        out.append(line)
    return "".join(out).rstrip("\n") + "\n"


def build_archive(dev_root, files, out_path):
    """A deterministic tar.gz: members sorted by path, mtime 0, uid and gid 0, empty uname and gname, mode 0o644,
    GNU format; gzip with an empty FNAME and mtime 0, so the header carries no file name (gzip.open stores the
    output's basename, which made a.tar.gz and b.tar.gz differ: probe, 23 Oct). Returns the file's SHA-256."""
    dev_root, out_path = Path(dev_root), Path(out_path)
    with open(out_path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz, \
            tarfile.open(fileobj=gz, mode="w", format=tarfile.GNU_FORMAT) as tar:
        for p in sorted(Path(f) for f in files):
            info = tar.gettarinfo(p, arcname=p.relative_to(dev_root).as_posix())
            info.mtime, info.uid, info.gid, info.uname, info.gname, info.mode = 0, 0, 0, "", "", 0o644
            with open(p, "rb") as fh:
                tar.addfile(info, fh)
    return sha256_file(out_path)


AUTHOR_MARKER = re.compile(r"\[AUTHOR:[^\]]*\]")
KNOWLEDGE_BEGIN, KNOWLEDGE_END = "<!-- begin knowledge statement -->", "<!-- end knowledge statement -->"
#: What counts as a tick in a box of the knowledge statement's item-1 table (Seat M, M15; Ruling 119): a ballot box with an
#: X or with a check, the two check marks, a letter x in either case (optionally followed by a detail). A box that is still
#: empty is "\u2610".
_TICK = re.compile(r"(?:\u2612|\u2611|\u2713|\u2714|x|X)(?:\s.*)?", re.S)
#: A paragraph that is one italic span: the guidance of the template, which the author deletes.
_ITALIC_PARAGRAPH = re.compile(r"\*(?![\s*])[^*]+(?<![\s*])\*")
_GUIDANCE_STARTS = ("*Fill this in", "*This covers")


def author_markers(text):
    """The ``[AUTHOR: ...]`` markers of ``text``, in order: what the author still has to fill in."""
    return AUTHOR_MARKER.findall(text)


def _knowledge_block(text):
    """The knowledge statement inside ``text``: between its two comment lines in a rendered protocol; None for a protocol
    that has no statement (its Part H holds the one marker); ``text`` itself when it is the statement."""
    found = re.search(re.escape(KNOWLEDGE_BEGIN) + r"\n(.*?)\n" + re.escape(KNOWLEDGE_END), text, flags=re.S)
    if found:
        return found.group(1)
    if "## Part H. Knowledge statement" in text or text.lstrip().startswith(
            ("# Pre-specified protocol", "# Pre-registration protocol")):
        return None
    return text


def _heading(line):
    """(level, title) of a markdown heading line, else None."""
    found = re.match(r"^(#{1,6})\s+(.*\S)\s*$", line)
    return (len(found.group(1)), found.group(2)) if found else None


def _item_one_rows(block):
    """(item text, option cells) for each data row of the table under the heading of item 1 of the statement, at any
    heading level: the rows between that heading and the next heading of the same or a higher level that begin with a
    pipe, without the header row (its first cell is "Item") and the separator row."""
    lines = block.splitlines()
    start = next((i for i, ln in enumerate(lines) if (h := _heading(ln)) and re.match(r"1\.\s", h[1])), None)
    if start is None:
        return []
    level, rows = _heading(lines[start])[0], []
    for ln in lines[start + 1:]:
        h = _heading(ln)
        if h and h[0] <= level:
            break
        if ln.strip().startswith("|"):
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if cells[0] != "Item" and not all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                rows.append((cells[0], cells[1:]))
    return rows


def knowledge_problems(text):
    """What is still wrong with the knowledge statement in ``text`` (a rendered protocol or the statement itself), as
    marker-like strings (Seat M, M15; controller Ruling 119): every row of the item-1 table that does not have exactly one
    ticked box, and every template guidance paragraph or line that is left (one italic span, or one that starts with
    "*Fill this in" or "*This covers"). Nothing for a protocol without a statement, and no complaint about a statement that
    has no item-1 table."""
    block = _knowledge_block(text)
    if not block:
        return []
    problems = []
    for item, options in _item_one_rows(block):
        ticked = sum(bool(_TICK.fullmatch(c)) for c in options)
        if ticked != 1:
            problems.append(f'[ITEM 1: {"no box is" if ticked == 0 else f"{ticked} boxes are"} ticked in the row "{item}"]')
    seen = set()
    for chunk in re.split(r"\n\s*\n", block) + block.splitlines():           # a paragraph, or a line inside a longer one
        p = chunk.strip()
        if p and p not in seen and (p.startswith(_GUIDANCE_STARTS) or _ITALIC_PARAGRAPH.fullmatch(p)):
            seen.add(p)
            problems.append(f"[GUIDANCE: {p[:60]}{chr(0x2026) if len(p) > 60 else ''}]")
    return problems


def ready_to_register(text):
    """Everything that still blocks registration in ``text``: the ``[AUTHOR: ...]`` markers, then the problems of the
    knowledge statement (``knowledge_problems``). Empty when the protocol can be registered."""
    return author_markers(text) + knowledge_problems(text)


# ---------------------------------------------------------------------------------------------------------------
# The code set: manifest and archive (controller Ruling 107(2): the import closure of the sealed path)
# ---------------------------------------------------------------------------------------------------------------
#: The modules the sealed path starts from: the run, the pull, the registration gate and this module.
SEALED_ROOTS = ("es.sealed_run", "es.sealed_pull", "es.registration", "es.protocol")
#: The two modules that computed the registered family and the simulation rows of protocol_inputs.json: the registration
#: rule, the placebo pairings and the choice of eta (es.run_analogs), and the size and power study (es.simulate). Nothing on
#: the sealed path imports them, but they are roots of the registered code set, so that the archive can regenerate
#: protocol_inputs.json (controller Ruling 118, Seat C1, C1-M1). They have no reason to change after registration.
PROVENANCE_ROOTS = ("es.run_analogs", "es.simulate")
#: Everything the code manifest and the archive start from.
CODE_ROOTS = SEALED_ROOTS + PROVENANCE_ROOTS
#: A module whose path matches is development, reporting or test code. The sealed path must not import one: it would make
#: every later edit of the paper or of a test a "departure" from the registered code (Ruling 107(2)).
NOT_SEALED_CODE = re.compile(r"^(tests/|outputs/|es/(paper_\w+|run_digest|run_legacy|impl_table)\.py$)")


def _module_file(root, name):
    """The file of the project module ``name`` (dotted) under ``root``: ``a/b.py`` or ``a/b/__init__.py``; None if
    ``root`` holds no such module (a standard-library or third-party module, or an attribute of a module)."""
    base = Path(root).joinpath(*name.split("."))
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    return None


def _is_main_guard(test):
    """``__name__ == "__main__"``."""
    return (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name) and test.left.id == "__name__"
            and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq) and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == "__main__")


def _import_statements(tree):
    """Every import statement of a syntax tree, in functions and ``try`` blocks too, except those under an
    ``if __name__ == "__main__":`` guard: they run only when the file is run as a script, never when it is imported."""
    stack = [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, ast.If) and _is_main_guard(node.test):
            stack.extend(node.orelse)                      # the else branch does run on import
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node
        stack.extend(ast.iter_child_nodes(node))


def _project_imports(root, name, file):
    """The project modules that module ``name`` (in ``file``) imports, by absolute or relative name. ``from a import b``
    names ``a``, and ``a.b`` as well if that is a module (a submodule imported through its package)."""
    tree = ast.parse(Path(file).read_text(encoding="utf-8"), filename=str(file))
    package = name if Path(file).name == "__init__.py" else name.rpartition(".")[0]
    found = set()
    for node in _import_statements(tree):
        if isinstance(node, ast.Import):
            targets = [a.name for a in node.names]
        elif node.level:
            parts = package.split(".") if package else []
            keep = len(parts) - (node.level - 1)
            if keep < 1:
                continue                                    # a relative import above the top-level package: invalid
            base = ".".join(parts[:keep] + ([node.module] if node.module else []))
            targets = [base] + [f"{base}.{a.name}" for a in node.names if a.name != "*"]
        else:
            targets = [node.module] + [f"{node.module}.{a.name}" for a in node.names if a.name != "*"]
        found.update(t for t in targets if t and _module_file(root, t))
    return found


def import_closure(dev_root, roots=CODE_ROOTS):
    """The transitive import closure of ``roots`` within the project, found by tracing the imports of each file's
    syntax tree (``_project_imports``) recursively, with the ``__init__.py`` of every package above a module (it runs
    when the module is imported). Returns ({module name: file}, {module name: the modules that import it}), limited to
    the project's own modules (``es/``, ``v2/``, ``src/`` and the top-level scripts they import). Static: a lazy
    import inside a function counts, an import under ``if __name__ == "__main__"`` does not. Raises ValueError for a
    root that is not a project module."""
    root = Path(dev_root)
    modules, importers, todo = {}, {}, []
    for name in roots:
        if _module_file(root, name) is None:
            raise ValueError(f"{name} is not a module under {root}")
        todo.append((name, None))
    while todo:
        name, by = todo.pop()
        if by is not None:
            importers.setdefault(name, set()).add(by)
        if name in modules:
            continue
        modules[name] = _module_file(root, name)
        parts = name.split(".")
        for i in range(1, len(parts)):
            if _module_file(root, ".".join(parts[:i])):
                todo.append((".".join(parts[:i]), name))
        todo.extend((dep, name) for dep in sorted(_project_imports(root, name, modules[name])))
    return dict(sorted(modules.items())), {k: sorted(v) for k, v in sorted(importers.items())}


def code_files(dev_root):
    """The code the registration identifies, sorted (controller Rulings 107(2) and 118): the files of the import closure
    of the sealed path and of the two provenance roots (``import_closure`` of ``CODE_ROOTS``: ``SEALED_ROOTS`` and
    ``PROVENANCE_ROOTS``), and the data files those modules open by a fixed path (``OPENED_DATA_FILES``). The tests, the
    paper and reporting modules, the other development runners and ``outputs/`` are not in it. Raises RuntimeError if
    one of the roots imports a module of ``NOT_SEALED_CODE``
    (naming who imports it: report it, do not register it silently), and FileNotFoundError for a listed data file that
    is missing: a registered set must be complete."""
    root = Path(dev_root)
    modules, importers = import_closure(root)
    rel = {name: f.relative_to(root).as_posix() for name, f in modules.items()}
    bad = {name: r for name, r in rel.items() if NOT_SEALED_CODE.match(r)}
    if bad:
        raise RuntimeError("the registered code set (the sealed path and the provenance roots) imports code that is not "
                           "sealed code, so registering it would make every later "
                           "edit of it a departure (controller Ruling 107(2)); report it: "
                           + "; ".join(f"{r} (imported by {', '.join(importers.get(n, ['a root']))})"
                                       for n, r in sorted(bad.items())))
    found = {root / r for r in rel.values()} | {root / d for d in OPENED_DATA_FILES}
    missing = sorted(p.relative_to(root).as_posix() for p in found if not p.is_file())
    if missing:
        raise FileNotFoundError(f"code files missing under {root}: {missing}")
    return sorted(found)


def manifest_text(dev_root, files):
    """CODE_MANIFEST.sha256: one ``sha256  relpath`` line per file (two spaces; relpath relative to the dev root, POSIX
    separators), the convention of ``es.sealed_pull.write_manifest`` that ``es.sealed_run.code_departures`` reads."""
    root = Path(dev_root)
    return "".join(f"{sha256_file(p)}  {p.relative_to(root).as_posix()}\n" for p in sorted(Path(f) for f in files))


def code_summary(dev_root, files):
    """What the protocol prints about the code set: the relative paths (modules and data files) in order, the number of
    files per top-level folder, and the modules of ``NOT_SEALED_CODE`` among them (none, or code_files would have raised)."""
    root = Path(dev_root)
    rel = [Path(p).relative_to(root).as_posix() for p in sorted(Path(f) for f in files)]
    groups = {}
    for r in rel:
        key = r.split("/")[0] if "/" in r else "(top level)"
        groups[key] = groups.get(key, 0) + 1
    return {"files": rel, "n_files": len(rel), "groups": sorted(groups.items()), "data": list(OPENED_DATA_FILES),
            "not_sealed": [r for r in rel if NOT_SEALED_CODE.match(r)]}


# ---------------------------------------------------------------------------------------------------------------
# The vocabulary of the sealed run's status (Ruling 104), read from es.sealed_run's own strings
# ---------------------------------------------------------------------------------------------------------------
STATUS_KEYS = ("completed", "secondary", "errors", "not_run")


def _status_strings(values):
    """The strings among ``values`` (a string, or a mapping, list, tuple or set of them), flattened."""
    out = []
    for v in values:
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, dict):
            out.extend(x for x in v.values() if isinstance(x, str))
        elif isinstance(v, (list, tuple, set, frozenset)):
            out.extend(x for x in v if isinstance(x, str))
    return out


def statuses_from_source(source, named=()):
    """The run statuses written by the module whose source text is ``source``: the string constants (not docstrings) that
    read "completed", "completed with secondary errors", "completed with errors" and "not run ..." (the status of a run
    that failed the coverage rule), together with ``named`` (strings from the module's STATUS* attributes, if it has them:
    see ``statuses_from_module``), as {"completed", "secondary", "errors", "not_run"}. es.sealed_run keeps them as literals
    in ``_Run.status`` and ``_execute``; named constants (a string, or a mapping or sequence of strings) are read as well,
    in whatever form the module takes. RuntimeError unless each of the four is found, as exactly one string, so that a
    renamed status stops the protocol instead of being misdescribed."""
    tree = ast.parse(source)
    docstrings = {id(n.body[0].value) for n in ast.walk(tree)
                  if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.body
                  and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    candidates = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
                  and id(n) not in docstrings]
    found = {k: set() for k in STATUS_KEYS}
    for v in [*candidates, *named]:
        if v == "completed":
            found["completed"].add(v)
        elif v.startswith("completed with secondary"):
            found["secondary"].add(v)
        elif v.startswith("completed with"):
            found["errors"].add(v)
        elif v.startswith("not run"):
            found["not_run"].add(v)
    bad = {k: sorted(v) for k, v in found.items() if len(v) != 1}
    if bad:
        raise RuntimeError(f"cannot read the run statuses from es.sealed_run: each of {list(STATUS_KEYS)} must be written "
                           f"as exactly one string; found {bad}")
    return {k: next(iter(v)) for k, v in found.items()}


def statuses_from_module(module):
    """``statuses_from_source`` of a loaded module's source, plus the values of its attributes named STATUS*: the
    vocabulary is the same whether the module writes the statuses as literals or as named constants."""
    source = Path(module.__file__).read_text(encoding="utf-8")
    return statuses_from_source(source, named=_status_strings(v for k, v in vars(module).items() if k.startswith("STATUS")))


@functools.lru_cache(maxsize=None)
def sealed_statuses():
    """The statuses of the sealed run, from the es.sealed_run module that is loaded (its source and any STATUS*
    constants; read once)."""
    return statuses_from_module(_sealed_run)


def _sealed(name):
    """A constant of es.sealed_run, by name: the words the run writes (STATUS_*, WORDING_*, SCOPE_*, PRIOR_*, INTERVAL_*)
    and its limits (TEST_IDS, N_REP, MIN_FREE_BYTES). RuntimeError if the module no longer has it, so that a renamed
    constant stops the protocol instead of being misdescribed (the rule of ``sealed_statuses``)."""
    try:
        return getattr(_sealed_run, name)
    except AttributeError:
        raise RuntimeError(f"es.sealed_run has no {name}, which the protocol quotes") from None


def _gib(n_bytes):
    return f"{n_bytes / 1024 ** 3:g} GiB"


def _percent(level):
    """0.025 -> '2.5%', 0.1 -> '10%'."""
    return f"{level * 100:g}%"


def _and_list(items):
    """['1%', '2.5%', '10%'] -> '1%, 2.5% and 10%'."""
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _repo_path(path):
    """A folder as the protocol prints it: relative to the repository root (``dart_dev/outputs/es/sealed/A``), or as it
    is if it lies elsewhere."""
    path = Path(path)
    try:
        return path.resolve().relative_to(Path(REVISION_ROOT).resolve()).as_posix()
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------------------------------------------
# Versions, pins and git
# ---------------------------------------------------------------------------------------------------------------
def capture_versions():
    """The live versions of python (``sys.version``) and of the libraries in ``VERSION_MODULES`` (``__version__``), in the
    order of the protocol. scipy enters the fitted simulation numbers and statsmodels the mixed model (Task 17
    re-review N-3)."""
    out = {"python": sys.version.split()[0]}
    for label, module in VERSION_MODULES:
        out[label] = str(importlib.import_module(module).__version__)
    return out


def pinned_versions(spec):
    """The ``name -> version`` pins of spec section 6.4's "Versions" bullet ("numpy (version recorded at freeze)" is not
    a pin)."""
    line = next((ln for ln in _need(spec, "### 6.4", "pins").splitlines() if "**Versions:**" in ln), "")
    return {name.lower(): version for name, version in re.findall(r"([A-Za-z][A-Za-z-]*) (\d+(?:\.\d+)+)", line)}


def check_pins(spec, versions):
    """The pins of spec section 6.4 that the live versions do not meet, as messages (empty if they all do)."""
    out = []
    for name, wanted in pinned_versions(spec).items():
        have = versions.get(name)
        if have is not None and have != wanted:
            out.append(f"{name} {have} is running, but spec section 6.4 pins {wanted}")
    return out


def env_pins(versions, executable=None):
    """The content of ENV_PINS.json: ``{"executable": <sys.executable>, "versions": {library: version}}`` for the libraries
    of ``ENV_PIN_LIBRARIES``, in that order (Ruling 109(1)). ``es.sealed_run`` compares the running environment with it
    before its lock. ValueError if ``versions`` lacks one of the libraries."""
    missing = [k for k in ENV_PIN_LIBRARIES if k not in versions]
    if missing:
        raise ValueError(f"cannot pin the environment: no version of {missing}")
    return {"executable": str(sys.executable if executable is None else executable),
            "versions": {k: str(versions[k]) for k in ENV_PIN_LIBRARIES}}


def env_pins_text(versions, executable=None):
    """ENV_PINS.json as written: JSON with two-space indentation, keys in the order of ``env_pins``, one final newline."""
    return json.dumps(env_pins(versions, executable), indent=2) + "\n"


def _git(dev_root, *args):
    """The finished ``git -C dev_root <args>``, or None if git is missing or the command failed."""
    try:
        done = subprocess.run(["git", "-C", str(dev_root), *args], capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.SubprocessError):
        return None
    return done if done.returncode == 0 else None


def git_state(dev_root, files, bundle_path):
    """None unless ``git -C dev_root rev-parse HEAD`` succeeds and ``git status --porcelain -- <files>`` is empty. Then
    ``{"commit": HEAD, "bundle": <name>, "bundle_sha256": ..., "not_in_commit": [files git does not track]}``, after
    ``git bundle create <bundle_path> HEAD`` (the absolute path: a relative one would land inside the dev root). A
    file that git ignores shows nothing in ``status`` yet is not in the commit, so it is listed in ``not_in_commit``
    (the manifest and the archive identify it)."""
    head = _git(dev_root, "rev-parse", "HEAD")
    if head is None:
        return None
    rel = [Path(f).relative_to(dev_root).as_posix() for f in files]
    status = _git(dev_root, "status", "--porcelain", "--", *rel)
    if status is None or status.stdout.strip():
        return None
    tracked = _git(dev_root, "ls-files", "--", *rel)
    tracked = set(tracked.stdout.split("\n")) if tracked is not None else set()
    bundle_path = Path(bundle_path).resolve()
    made = _git(dev_root, "bundle", "create", str(bundle_path), "HEAD")
    if made is None:
        raise RuntimeError(f"git bundle create {bundle_path} failed although HEAD is clean for the code files")
    return {"commit": head.stdout.strip(), "bundle": bundle_path.name, "bundle_sha256": sha256_file(bundle_path),
            "not_in_commit": sorted(set(rel) - tracked)}


# ---------------------------------------------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------------------------------------------
DASH = "\u2014"


def _g(x):
    """A number as the shortest faithful text: 0.05 -> '0.05', 1.5 -> '1.5', 2000 -> '2000'."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return DASH
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return format(x, "g")
    return str(x)


def _n(x, digits=4):
    """A number with ``digits`` decimals (DASH for None or NaN); integers and text as they are."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return DASH
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def _val(x):
    """Any JSON value of the inputs as one line."""
    if isinstance(x, dict):
        return "; ".join(f"{k} = {_val(v)}" for k, v in x.items())
    if isinstance(x, (list, tuple)):
        return ", ".join(_val(v) for v in x)
    return _g(x)


def _cell(x):
    return str(DASH if x is None else x).replace("|", "\\|").replace("\n", " ")


def _table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _bullets(items):
    return "\n".join(f"- {item}" for item in items)


def _window(w):
    return f"{w.start} to {w.end}"


def _days(w):
    return (dt.date.fromisoformat(w.end) - dt.date.fromisoformat(w.start)).days + 1


def _default(fn, name):
    """The default of parameter ``name`` of ``fn``: a number that the code holds, read from the code."""
    return inspect.signature(fn).parameters[name].default


def _need(text, heading, what):
    try:
        return section(text, heading)
    except StopIteration:
        raise ValueError(f"the spec has no section headed {heading!r} ({what})") from None


def _need_item(text, heading, n, what):
    try:
        return list_item(text, heading, n)
    except StopIteration:
        raise ValueError(f"the spec has no item {n} under {heading!r} ({what})") from None


# ---------------------------------------------------------------------------------------------------------------
# Rendering: one function per part
# ---------------------------------------------------------------------------------------------------------------
def _part_title(ctx):
    author = ctx.get("author") or "Erce Erez"
    date = ctx.get("date") or dt.date.today().isoformat()
    windows = "; ".join(f"Test {tid}: deployment {w[1]} to {w[2]}" for tid, w in SEALED_WINDOWS.items())
    return "\n\n".join([
        "# Pre-specified protocol: sealed out-of-sample tests, July to December 2026",
        f"Paper (working title): *Feature-Block Portability and Operating-Point Drift in DART-Spread Spike Prediction*  \n"
        f"Author: {author} (sole author)  \nDate of this version: {date}  \n"
        f"Design: spec v3, 22 October 2026 (`{SPEC_REL}`), with the controller rulings listed in Part E  \n"
        f"Tests: {windows} (local settlement days).",
        "## How to read this protocol",
        "This file is generated by `es.protocol.main()`. It fixes, before either sealed window is opened, the hypotheses, "
        "the decision logic, the pins, the thresholds, the table shells and the code of two sealed tests. The sealed "
        "windows are read only by `es.sealed_pull.pull_sealed` and `es.sealed_run.run_sealed`, and both refuse to start "
        "unless `docs/sealed/REGISTRATION.md` names the SHA-256 of this file. In this file, \"registered\" and "
        "\"registration\" describe a design that is fixed here and recorded, with the SHA-256 of this file, in "
        "`docs/sealed/REGISTRATION.md`; they do not mean an entry in a third-party registry unless that record is an "
        "OSF registration (Rulings 140 and 141).",
        _bullets([
            "**Part A** copies the registered design (spec v3, with amendments) word for word: the economic analysis "
            "(section 4), the statistics (section 5) and the sealed tests (sections 6.1 to 6.5), with the inline "
            "amendment notes the spec itself carries. Its headings and numbers are the spec's own. Three sentences of "
            "sections 6.1 and 6.5 that named OSF as the record of the design are rewritten (Ruling 140).",
            "**Parts B to D** state what the code does, rendered from the code and from `protocol_inputs.json`: the "
            "windows, the development inputs, the registered family and the way each analysis is run.",
            "**Part E** lists the controller rulings that changed or pinned an analysis after spec v3, one line each, "
            "with the ledger number as the evidence pointer.",
            "**Part F** binds the files and the code to this protocol by SHA-256. **Part G** gives the table shells. "
            "**Part H** is the knowledge statement. **Part I** lists the library versions.",
        ]),
        "**Order of authority.** Parts B to D and Part E describe the code that will run, and the code is bound to this "
        "protocol by hash. Where one of them and Part A differ, Parts B to E govern; each such difference is a ruling "
        "listed in Part E (for example Ruling 69 for the rotation set, Ruling 84(d) for the wording of an H-B "
        "rejection, Ruling 28 for the identity of the code).",
    ])


def _fee_levels_in_words():
    """The calibrated fees as 'NYISO 0.20/0.15, ...' (INC/DEC), read from the code when the protocol is rendered."""
    return ", ".join(f"{m} {side[1]:.2f}/{side[-1]:.2f}" for m, side in CALIBRATED_FEES.items())


#: The three sentences of the spec that make an OSF registration the record of the design (the table of section 6.1, and
#: steps 3 and 5 of section 6.5), each with what Part A prints in its place (Ruling 140). The protocol had not been registered
#: or published when it was rewritten, so the record of the design is now one of three: an OSF registration, a public release
#: of the code repository or the author's local record. Each spec text must occur exactly once in Part A, or the protocol
#: stops: the spec has changed under it.
SPEC_RECORD_SENTENCES = (
    ("right after OSF registration",
     "right after the protocol is recorded (6.5 step 5)"),
    ("*(Amended 24 Oct, Ruling 119, G1 M14: the development outputs, outputs/es/analogs/ and DIGEST.md, are uploaded to the "
     "OSF registration as supplementary files that are not bound by hash.",
     "*(Amended 24 Oct, Ruling 119, G1 M14, and 26 Oct, Ruling 140: the development outputs, outputs/es/analogs/ and "
     "DIGEST.md, are published with the code repository as supplementary files that are not bound by hash, instead of "
     "being uploaded to an OSF registration."),
    ("5. **The author registers on OSF** and records the DOI and date in `docs/sealed/REGISTRATION.md`. *(Amended 23 Oct, "
     "Ruling 66c: public or embargoed is the author's choice; public is recommended because it gives a DOI at once. An "
     "embargoed registration may record its OSF registration URL instead of a DOI.)*",
     "5. **The protocol is recorded** in `docs/sealed/REGISTRATION.md`, with its SHA-256 and the UTC time of the record, as "
     "one of three records: an OSF registration, a public release of the code repository or the author's local record. "
     "*(Amended 23 Oct, Ruling 66c, for an OSF registration: public or embargoed is the author's choice; public is "
     "recommended because it gives a DOI at once. An embargoed registration may record its OSF registration URL instead "
     "of a DOI. Amended 26 Oct, Rulings 140 and 141: the record may also be a public release of the code repository or "
     "the author's local record, and Test A is run on a local record.)*"),
)


def rewrite_record_sentences(text):
    """``text`` (Part A) with the three sentences of ``SPEC_RECORD_SENTENCES`` rewritten. ValueError if one of them is not
    in ``text`` exactly once."""
    for old, new in SPEC_RECORD_SENTENCES:
        if text.count(old) != 1:
            raise ValueError(f"the spec no longer holds, exactly once, the sentence that Ruling 140 rewrites: {old[:70]!r}")
        text = text.replace(old, new)
    return text


def _part_a(spec):
    """Part A: the registered design, copied from the spec word for word, inline amendment notes included: section 4
    (economic analysis, 4.1 to 4.6, with its sign-convention line), section 5 (statistics, items 1 to 9) and sections 6.1
    to 6.5 (sealed tests), except for the three sentences that named OSF as the record of the design (Ruling 140,
    ``SPEC_RECORD_SENTENCES``). A protocol note follows the fee table of 4.2, which the spec itself marks as superseded."""
    s4 = _need(spec, "## 4.", "economic analysis")
    first = s4.find("### 4.1")
    if first < 0:
        raise ValueError("the spec has no section headed '### 4.1' (settlement prices)")
    parts = [
        "## Part A. Registered design (spec v3 with amendments)",
        f"Source: `{SPEC_REL}` (v3, 22 October 2026). Section 4 (economic analysis, 4.1 to 4.6), section 5 (statistics, "
        "items 1 to 9) and sections 6.1 to 6.5 (sealed tests) are copied unchanged, with the amendment notes that the spec "
        "itself carries, except for three sentences that named OSF as the record of the design (the table of section 6.1, "
        "and steps 3 and 5 of section 6.5), which are rewritten under Ruling 140. Their headings and numbers are the "
        "spec's own (`## 4.`, `## 5.`, `### 6.1` and so on). The rest "
        "of the spec (context, objectives, invariants, section 6.6, and sections 7 to 13) is not copied: "
        "cross-references in the copied text to those sections (for example to sections 3, 7 and 13) point to the "
        "spec. Parts B to E state what the code does where it matters for the sealed tests.",
        s4[:first].rstrip("\n"),
    ]
    for k in range(1, 7):
        parts.append(_need(spec, f"### 4.{k}", "economic analysis").rstrip("\n"))
        if k == 2:
            parts.append("> **Protocol note on the fee table of section 4.2.** The table above is superseded, as the "
                         "spec's own note under it says (Rulings 78 and 90). The binding levels are those of "
                         "`es.positions.CALIBRATED_FEES`, rendered from the code in Part D (Fees) and repeated here "
                         f"(INC/DEC): {_fee_levels_in_words()}. Where the two differ, the rendered levels govern.")
    parts.append(_need(spec, "## 5.", "statistics").rstrip("\n"))
    for heading in ("### 6.1", "### 6.2", "### 6.3", "### 6.4", "### 6.5"):
        parts.append(_need(spec, heading, "sealed tests").rstrip("\n"))
    return rewrite_record_sentences("\n\n".join(parts))


def _part_b():
    rows = []
    for tid, (pull_start, deploy_start, deploy_end, dirname) in SEALED_WINDOWS.items():
        lookback = (dt.date.fromisoformat(deploy_start) - dt.date.fromisoformat(pull_start)).days
        rows.append([f"Test {tid}", pull_start, deploy_start, deploy_end, f"{lookback} days", f"`{dirname}`"])
    windows = _table(["Test", "Pull starts", "Deployment starts", "Deployment ends", "Pull starts before deployment",
                      "Pull directory under the data root"], rows)
    analyses = _table(
        ["Analysis", "Maintenance", "Train", "Calibrate", "Deploy"],
        [[f"{a.id}", a.maintenance, _window(a.train), _window(a.calib), _window(a.deploy)] for a in SEALED.values()])
    return "\n\n".join([
        "## Part B. Pull windows and sealed analyses",
        "Each sealed pull starts before its deployment window, so that the feature lookback of spec section 6.1 is "
        "covered. The rows before deployment come from data that were already open (the 2026-H1 file for Test A, Test "
        "A's pull for Test B), and the new pull is compared with them; the comparison is logged in `OVERLAP_CHECK.csv` "
        "and is not enforced, because ISOs revise settled prices. A pull is written to a new "
        "folder under `~/dart-spread-data/` with an explicit root, and runs once.",
        windows,
        "The four sealed analyses (`es.analyses.SEALED`), in inclusive local settlement days. SA and SB refit the "
        "model at each origin on all rows before the calibration quarter (rolling-origin); SAF and SBF train on 2023, "
        "calibrate on the same quarter of 2024 and apply the system unchanged (frozen-origin).",
        analyses,
    ])


def _analogue_table():
    return _table(["Analogue", "Maintenance", "Train", "Calibrate", "Deploy"],
                  [[a.id, a.maintenance, _window(a.train), _window(a.calib), _window(a.deploy)]
                   for a in ANALOGS.values()])


def _simulation_table(rows):
    cols = [("test", "Test"), ("law", "Law"), ("scenario", "Scenario"), ("persistence", "Persistence"),
            ("strata", "Strata"), ("rho", "rho"), ("n_sim", "Simulations"), ("seed", "Seed"),
            (f"reject_{ALPHA / 3:g}", f"Rejects at {ALPHA}/3"), (f"reject_{ALPHA:g}", f"Rejects at {ALPHA:g}"),
            ("note", "Note")]
    cols = [(k, h) for k, h in cols if any(k in r for r in rows)]
    body = []
    for r in rows:
        body.append([(_n(r.get(k), 4) if k.startswith("reject_") else _g(r.get(k)) if k in r else DASH)
                     for k, _ in cols])
    return _table([h for _, h in cols], body)


_BANDS = (("0.5_2", "[0.5, 2]"), ("0.67_1.5", "[0.67, 1.5]"), ("0.8_1.25", "[0.8, 1.25]"))


def _nodrift_table(rows):
    """The no-drift context: the 95th percentile, over the simulations of an H-B size row without drift, of the number of
    nodes whose activity ratio R lies outside each failure band. None if no row carries it."""
    keep = [r for r in rows if any(r.get(f"nodrift_fail_q95_{k}") is not None for k, _ in _BANDS)]
    if not keep:
        return None
    return _table(["Law (base block)", "Persistence", "Simulations"] + [f"Nodes outside {b}" for _, b in _BANDS],
                  [[r.get("law"), r.get("persistence"), _g(r.get("n_sim"))]
                   + [_g(r.get(f"nodrift_fail_q95_{k}")) for k, _ in _BANDS] for r in keep])


def _placebo_table(rows):
    cols = [("pair", "Pairing"), ("counted", "Counted by the rule"), ("p", "p"), ("t_obs", "T"), ("n_null", "Null size"),
            ("selc_obs", "Pooled SELC"), ("n_rows", "Rows"), ("n_settle_approx", "Rows on the approximate leg"),
            ("note", "Note")]
    cols = [(k, h) for k, h in cols if any(k in r for r in rows)]
    body = [[(_n(r.get(k), 4) if k in ("p", "t_obs", "selc_obs") else _g(r.get(k)) if k in r else DASH)
             for k, _ in cols] for r in rows]
    return _table([h for _, h in cols], body)


def _spec_numbers(spec, pattern):
    """The numbers of the first match of ``pattern`` (one group of comma-separated numbers) in the spec, or None."""
    m = re.search(pattern, spec)
    return [float(x) for x in m.group(1).split(",")] if m else None


def _eta_text(inputs, spec):
    """The line on eta. The registered grid is read from the spec (section 4.3, P5), not from the study's module: this
    module must not import the development runner, or the code identity would include it (Ruling 107(2))."""
    eta = inputs.get("eta")
    if eta is None:
        return []
    grid = _spec_numbers(spec, "\u03b7 \u2208 \\{([0-9., ]+)\\}")
    line = (f"eta = {_g(eta)}: the step of the self-correcting cutoff P5 (alpha_(d+1) = clip(alpha_d + eta (alpha - r_d), "
            f"{_g(_default(p5_self_correcting, 'lo'))}, {_g(_default(p5_self_correcting, 'hi'))})), chosen "
            + (f"from the registered grid {{{', '.join(_g(e) for e in grid)}}} " if grid else "")
            + "by the least panel-median |log10 R| on the quarterly analogues, ties to the smaller eta (spec section 4.3).")
    if grid and float(eta) == max(grid):
        line += " The choice is the largest value of the grid; the grid is not extended."
    return [line]


_PHI_COMPONENTS = {"score": "score", "pnl": "P&L"}
_PHI_STATUSES = ("corrected", "clipped_high", "clipped_low", "no_signal", "uncorrected")


def _phi_status_rows(status):
    """One row per block and daily component, from ``phi_status`` of protocol_inputs.json (Seat M, M27). ``by_component``
    counts each component on its own; the block-level counts give a node once for each status it has in any component, so
    for a block with two components they add up to more than its nodes and are not shown. A block without ``by_component``
    (an older file) gets one row of its block-level counts, marked as such."""
    rows = []
    for block, v in status.items():
        by = v.get("by_component")
        if not by:
            rows.append([block, "all (a node counts once per status)", _g(v.get("phi_nodes")),
                         *[_g(v.get(f"phi_{s}")) for s in _PHI_STATUSES], v.get("note") or DASH])
            continue
        for component in v.get("components") or list(by):
            if component not in by:
                raise ValueError(f"protocol_inputs.json: phi_status of {block} lists the component {component!r}, which "
                                 "by_component does not count")
            rows.append([block, _PHI_COMPONENTS.get(component, component), _g(v.get("phi_nodes")),
                         *[_g(by[component].get(s)) for s in _PHI_STATUSES], v.get("note") or DASH])
    return rows


def _part_c(ctx, inputs):
    out = ["## Part C. Development inputs",
           "Everything in this Part was computed on development data (up to 2026-06-30), before registration, by "
           "`es.run_analogs` (stage `analogs`, then stage `sim`). It is the content of `protocol_inputs.json`, a "
           "byte-identical copy of which is bound in Part F."]
    out.append("### Quarterly analogues, the self-correcting step and the effect coupling")
    out.append("The quarterly analogues (`es.analyses.ANALOGS`) have the structure of the sealed tests, on development "
               "data. AQ3 stands for Test A and AQ4 for Test B; AQ1 and AQ2 add breadth; AQ3F is the frozen-origin "
               "analogue of AQ3 (spec section 6.2).")
    out.append(_analogue_table())
    lines = _eta_text(inputs, ctx["spec"])
    if "rho_star" in inputs:
        couplings = sorted({r["rho"] for r in inputs.get("simulation") or []
                            if r.get("test") == "H-E pooled A+B, power curve" and r.get("rho") is not None})
        lines.append(f"rho_star = {_g(inputs['rho_star'])}: the coupling of the H-E power simulation"
                     + (f", from the grid {{{', '.join(_g(r) for r in couplings)}}}," if couplings else "")
                     + " at which the simulated pooled SELC is nearest the effect target, half of the loss-weighted "
                     "pooled SELC of the analogues AQ3 and AQ4 (the power rows of the simulation table below say "
                     "whether the target lies inside the grid).")
    sel = inputs.get("analog_selc")
    if sel:
        names = ("AQ3", "AQ4") if len(sel) == 2 else tuple(f"#{i + 1}" for i in range(len(sel)))
        text = ", ".join(f"{n} = {_n(v, 4)}" for n, v in zip(names, sel))
        if inputs.get("analog_selc_pooled") is not None:
            text += f"; loss-weighted pooled = {_n(inputs['analog_selc_pooled'], 4)}"
        lines.append(f"SELC of the fixed cutoff P1 on the analogues (premium position, calibrated fees, primary "
                     f"strata): {text}.")
    out.append(_bullets(lines))
    phi = inputs.get("phi")
    if phi:
        out.append("### Fitted persistence of the block scores")
        out.append("The daily lag-one persistence of each block's score component, median over nodes: the corrected "
                   "value (Ruling 84(a), indirect inference) and the raw one. The corrected value is reported beside "
                   "every H-B estimate.")
        out.append(_table(["Block", "Corrected phi (median)", "Raw phi (median)"],
                          [[b, _n(v.get("phi_score_median"), 3), _n(v.get("phi_score_raw_median"), 3)]
                           for b, v in phi.items()]))
    status = inputs.get("phi_status")
    if status:
        out.append("How the persistence corrections ended, by block and by daily component (nodes with each status in "
                   "that component; Task 17 re-review N-2, Seat M, M27). H-E draws two daily components, the score and "
                   "the profit and loss (P&L), and each is counted on its own: a node can be corrected in one and clipped "
                   "in the other, so the counts of the components of a block are not added. "
                   "\"Clipped high\" means that the correction hit the top of its search range, phi = 0.95 (the observed "
                   "lag-one statistic lay above what the map reaches there, Seat M, M22); \"clipped low\" means the bottom of "
                   "the range, phi = -0.5; \"no signal\" means that the daily means hold no variance beyond the hourly noise, "
                   "and \"uncorrected\" that there were too few day pairs:")
        out.append(_table(["Block", "Component", "Nodes", "Corrected", "Clipped high", "Clipped low", "No signal",
                           "Uncorrected", "Note"], _phi_status_rows(status)))
    fit = re.search(r"Fit this to the primary pipeline's out-of-sample scores and P&L for (\d{4}-\d{2}) to (\d{4}-\d{2})",
                    ctx["spec"])
    out.append("### Size and power simulation")
    out.append("Each node's hourly score and profit and loss is modelled as a seasonal mean (hour of day x weekday class "
               "x month, Ruling 64), a daily AR(1) component and hourly noise (heteroskedastic by hour and weekday "
               "class, Ruling 55(g)), fitted to the primary pipeline's out-of-sample scores and P&L"
               + (f" from {fit.group(1)} to {fit.group(2)}" if fit else "")
               + " (spec section 6.2(b); `SIMFIT` in `es.run_analogs`); two independent quarters "
               "are simulated, under no skill (H-E) and under equal portability (H-B). The table gives the rejection "
               "rates of the H-E rotation test and the H-B bootstrap, by scenario, at the two Holm levels. Rows under "
               "the decile strata are the sensitivity of Ruling 68 and are never read by the rule.")
    out.append(_simulation_table(inputs["simulation"]) if inputs.get("simulation") else "(no simulation rows)")
    nodrift = _nodrift_table(inputs.get("simulation") or [])
    if nodrift:
        out.append("**No-drift context** (spec section 6.2). In the H-B size rows without drift, the 95th percentile "
                   "over the simulations of the number of nodes whose activity ratio R of the base block lies outside "
                   "each failure band. It is reported descriptively next to the observed counts and is not a decision "
                   "threshold (review N2); the simulation has no logistic-regression no-drift context (Task 11 "
                   "review M9).")
        out.append(nodrift)
    out.append("### Placebo pairings")
    out.append("Analogue alerts paired with the profit and loss of the same calendar dates one and two years earlier "
               "(spec section 6.2(a), as amended by Ruling 94). Only the pairings counted by the rule enter it.")
    out.append(_placebo_table(inputs["placebo"]) if inputs.get("placebo") else "(no placebo rows)")
    summary = ctx.get("cutoff_summary")
    out.append("### Pre-pull cutoffs")
    out.append(
        f"`pre_pull_cutoffs.csv` holds, for each sealed analysis ({', '.join(SEALED)}), each confirmatory pipeline "
        f"({', '.join(CUTOFF_BLOCKS)}; XGBoost, seed {_default(cutoff_table, 'seed')}, alpha = "
        f"{_g(_default(cutoff_table, 'alpha'))}) and each node: the number of calibration rows, the fixed cutoff "
        "tau_p1 (the (1 - alpha) quantile of the calibration scores), the in-sample cutoffs tau_p2_season (training "
        "rows of the deployment months) and tau_p2_all (all training rows), and the orderings p2_season_above_p1 and "
        "p2_all_above_p1. They use training and calibration rows only (Ruling 30), so they were fixed on development "
        "data before the pull; the per-node ordering of tau_P1 and tau_P2 is thereby recorded before the pull (spec "
        "section 6.3, PC). The sealed run recomputes them from its own fits and reports the largest absolute "
        "difference, expected to be exactly 0 (`tau_check`).")
    if summary:
        out.append(_table(["Analysis", "Pipeline", "Nodes", "P2 season-matched above P1", "P2 all-season above P1",
                           "Nodes without a P1 cutoff"],
                          [[r["analysis"], r["block"], r["nodes"], r["p2_season_above_p1"], r["p2_all_above_p1"],
                            r["nodes_without_tau_p1"]] for r in summary]))
    return "\n\n".join(out)


_THIRD = f"{ALPHA:g}/3"
_RULE_MEANING = {
    "factor": "a size is oversized only above this multiple of the nominal level",
    "binomial_level": "level of the one-sided binomial test of a rejection count against the nominal level (divided "
                      "by the number of rows read when the worst of several rows is taken)",
    "approximate_factor": "multiple of nominal for the label \"approximate\" (the same combined condition)",
    "min_power": f"least power at {_THIRD} that an H-B test needs (H-E stays in whatever its power)",
    "levels": f"the two nominal levels read: {_THIRD} and {ALPHA:g}",
    "clipped_high_share": "share of persistence corrections clipped at the top, beyond which each row says it is "
                          "not a bound",
    "k_rows": "number of simulation rows each test's size is read over",
    "k_rows_interim": "rows read for the single-window H-E label",
    "laws": "the score laws whose rows each H-B test reads",
    "n_rep": f"bootstrap replicates inside each H-B simulation (so that {_THIRD} and {ALPHA:g} are exactly attainable)",
    "n_sim": "simulations per row, by study",
    "seeds": "seeds of the study (row i of the size study takes size_base + i)",
    "calibration_grid": "grid of effect sizes of the H-B power calibration",
    "counted_pairings": "the placebo pairings that the rule counts",
}


def _names(items):
    """'H-B1', 'H-B1 and H-B2', 'H-E, H-B1 and H-B2'."""
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _holm_members(fam):
    """The hypotheses of the Holm family: those the registration rule left confirmatory, in the order of the sealed run
    (``confirmatory`` in ``es.sealed_run.run_sealed``, which ``_decide`` reads)."""
    return [h for h in FAMILY_ORDER if (fam.get(h) or {}).get("confirmatory")]


def _decision_level_text(fam):
    """The level at which the registered family is decided, generated from the family the registration rule computed
    (its size m) and ALPHA (controller Ruling 118; Seat M, M1; Seat C1, C1-I1). With the registered family, {H-E}, Holm
    over one hypothesis is the test at 0.05; Part A's three-member step-down and its 0.05/3 rule do not apply. With more
    members the step-down is stated instead."""
    members = _holm_members(fam)
    replaced = [h for h in FAMILY_ORDER if h not in members]
    m, a = len(members), f"{ALPHA:g}"
    lead = (f"The registration rule replaced {_names(replaced)} by {'an estimand' if len(replaced) == 1 else 'estimands'}, "
            "so the confirmatory family is " if replaced
            else "The registration rule left every hypothesis confirmatory, so the confirmatory family is ")
    lead += "{" + ", ".join(members) + "}" if members else "empty"
    if m == 0:
        return lead + ". No hypothesis is tested and there is no Holm step-down."
    detected, not_detected = _sealed("WORDING_HE_DETECTED"), _sealed("WORDING_HE_NOT_DETECTED")
    interim, decided = _sealed("STATUS_HE_INTERIM"), _sealed("STATUS_HE_DECIDED")
    if m == 1 and members == ["H-E"]:
        return (f"{lead}. Holm over one hypothesis is a one-sided test at {a}: H-E reads \"{detected}\" if and only if its "
                f"pooled rotation p \u2264 {a}, and \"{not_detected}\" otherwise. In Test A its entry carries the status "
                f"\"{interim}\" and is never confirmatory; in Test B, pooled, its status is \"{decided}\". The "
                f"three-member step-down and the {a}/3 rule of Part A do not apply.")
    if m == 1:
        return (f"{lead}. Holm over one hypothesis is a one-sided test at {a}: {members[0]} is rejected in Test A if and only "
                f"if its p-value is at most {a}, and it is confirmed only if Test B replicates it (the same sign and a "
                f"one-sided p below {a}). H-E, replaced by its estimand, is not tested. The three-member step-down and the "
                f"{a}/3 rule of Part A do not apply.")
    thresholds = ", ".join(f"{a}/{m - i}" if m - i > 1 else a for i in range(m))
    text = (f"{lead}, m = {m}. The Holm step-down runs over these {m}: their p-values are ordered from the smallest and "
            f"compared in turn with {thresholds}, and the procedure stops at the first p-value that misses its threshold.")
    hb = [h for h in members if h != "H-E"]
    if hb:
        text += f" In Test A an H-B test of the family is rejected only if its p-value is at most {a}/{m}."
    text += (" In Test B the step-down takes " + ("H-E's pooled rotation p" if "H-E" in members else "")
             + (" and " if "H-E" in members and hb else "")
             + (f"Test A's p-value{'s' if len(hb) > 1 else ''} of {_names(hb)}" if hb else "") + ".")
    if "H-E" in members:
        text += (f" H-E reads \"{detected}\" if the step-down rejects it and \"{not_detected}\" otherwise; its status is "
                 f"\"{interim}\" in Test A and \"{decided}\" in Test B.")
    text += (" This is the three-member family of Part A." if m == 3 else
             f" Part A's three-member step-down and its {a}/3 rule do not apply as written; the thresholds above are those "
             f"of the {m} members.")
    return text


def _holm_clause(fam):
    """The same decision, in one clause for the description of the run."""
    members = _holm_members(fam)
    a = f"{ALPHA:g}"
    where = "here " + ("{" + ", ".join(members) + "}" if members else "empty, so no step-down is run")
    note = f" (Holm over one hypothesis is the test at {a}; see Part D)" if len(members) == 1 else " (see Part D)"
    return f"the Holm step-down is applied over the confirmatory family of the registered inputs only, {where}{note}"


def _family_text(inputs):
    fam, rule = inputs["family"], inputs.get("rule") or {}
    third = _THIRD
    rows = []
    for h in FAMILY_ORDER:
        f = fam.get(h, {})
        rows.append([h, "confirmatory test" if f.get("confirmatory") else "estimand (replaced by the rule)",
                     _g(f.get("approximate")), _g(f.get("interim_approximate")) if h == "H-E" else DASH,
                     _g(f.get("k_rows")), _n(f.get("false_demotion_bound"), 3),
                     _n(f["power"], 2) if "power" in f else "not read",
                     "; ".join(f.get("reasons") or []) or "none"])
    out = [_table(["Hypothesis", "Registered status", "Approximate", "Interim approximate", "Rows read (k)",
                   "False-demotion bound", f"Power at {third}", "Reasons from the rule"], rows)]
    bullets = []
    he = fam.get("H-E", {})
    if he.get("confirmatory"):
        text = ("**H-E is the one confirmatory test.** It is decided on the pooled Test A + Test B window, after Test B. "
                "Test A's single-window rotation p-value is interim and is never confirmatory. The pooled test needs "
                "Test A's stored P1 rows. If Test A's output is not available (it is NOT_RUN because its coverage rule "
                "failed, it was claimed and left no manifest and no finished record, or it finished without stored P1 "
                "rows), Test B runs alone and tests H-E on its own window, reported as not confirmatory. Test B is "
                "refused, before anything is spent, if Test A was never started, and if Test A's output is complete but "
                "its record lacks the finish fields, which a repair by hand restores (Rulings 112 and 116; see Failure "
                "handling).")
        text += (" The interim p-value is labelled approximate." if he.get("interim_approximate")
                 else " The interim p-value is not labelled approximate: no row of the size study of the "
                      "single-window test is significantly above nominal.")
        if he.get("approximate"):
            text += " The pooled test's size is significantly above nominal in some row, so its result is labelled approximate."
        bullets.append(text)
    else:
        bullets.append("**H-E is replaced by its estimand** (pooled SELC with an interval), for the reasons in the "
                       "table; its rotation p-value is reported with the estimand and is not a test. "
                       + ("The interval is labelled approximate." if he.get("approximate") else ""))
    bullets.append("**Decision level (Ruling 118).** " + _decision_level_text(fam))
    m = len(_holm_members(fam))                                                   # the Holm family the run counts
    for h, _ in HB_PAIRS:
        f = fam.get(h, {})
        if f.get("confirmatory"):
            bullets.append(f"**{h} is a confirmatory test**, decided by the registered logic of spec section 6.3: "
                           f"rejected in Test A at {ALPHA:g}/{m} (Holm; {m} is the number of confirmatory hypotheses "
                           "of the registered family), with replication in Test B"
                           + (", and its interval is labelled approximate" if f.get("approximate") else "") + ".")
        else:
            bullets.append(
                f"**{h} is an estimand, not a test**, for the reasons in the table. It is reported as the median over "
                "nodes of the paired |log10 R| difference, with the intervals described below. Its intervals are "
                + ("labelled approximate and " if f.get("approximate") else "")
                + "may under-cover (see the note under the interpretation table in Part A).")
    if not any(fam.get(h, {}).get("confirmatory") for h, _ in HB_PAIRS):
        bullets.append("**Test A has no confirmatory test of its own.** It reports the estimands, the estimates of "
                       "H-B1 and H-B2 with their fitted persistence, and the interim H-E p-value. H-E is decided on the "
                       "pooled window once Test B is in.")
    demoted = [h for h, _ in HB_PAIRS if fam.get(h) and not fam[h].get("confirmatory")]
    confirmed = [h for h, _ in HB_PAIRS if fam.get(h, {}).get("confirmatory")]
    if demoted:
        names = " and ".join(demoted)
        bullets.append(
            f"**Intervals of {names} (Ruling 110).** The primary reported interval is the percentile bootstrap "
            "interval (the `ci` field of the results). The basic interval (2t \u2212 q\u2080.\u2089\u2087\u2085, "
            "2t \u2212 q\u2080.\u2080\u2082\u2085), the `ci_basic` field, is shown beside it as a diagnostic: it "
            "inherits the over-rejection mechanisms that the size study found. The bootstrap distribution sits below "
            "the estimate t-hat mainly because the calibration days and the deployment days are drawn independently, "
            "which re-randomizes the matched calendar composition of the two windows (the effect that widens the E5 "
            "intervals; see \"E5 and H-S\" below); the median over nodes and the kink of |log10 R| add to it, and the "
            "basic interval reflects the whole shift about t-hat. Both intervals are approximate and may under-cover: "
            "the simulated bootstrap SD ratio is 0.68 to 0.83 under the fitted persistence in the synthetic no-drift "
            "study, and the intervals may under-cover substantially under a common drift (Ruling 118). Neither is "
            "read inferentially: the protocol makes no \"excludes zero\" claim for "
            f"{names}. No `p` field is stored for them: the one-sided bootstrap p-value is kept as a diagnostic, under "
            "`diagnostics.bootstrap_p_one_sided`, and is not a test. The `interval_registered` field reads "
            f"\"{_sealed('INTERVAL_HB_ESTIMAND')}\".")
    if confirmed:
        bullets.append(
            f"**Interval of {' and '.join(confirmed)}.** A confirmatory H-B test is reported with its one-sided "
            "basic bootstrap p-value (the `p` field) and the basic interval (2t \u2212 q\u2080.\u2089\u2087\u2085, "
            "2t \u2212 q\u2080.\u2080\u2082\u2085), the `ci_basic` field (spec section 5 item 1, Ruling 55(f)); "
            "the percentile interval (`ci`) is stored as well, and the `interval_registered` field reads "
            f"\"{_sealed('INTERVAL_HB_TEST')}\".")
    bullets.append("**Wording.** The H-B estimands have no inferential wording: neither interval is read as excluding "
                   "or containing zero (Ruling 110). If an H-B test were confirmatory, a rejection would read "
                   "\"larger operating-point error\", and never \"less portable\" (Ruling 84(d)): day-to-day "
                   "persistence alone produces gaps in |log10 R| of the tested size. H-E reads \"timing skill within "
                   "calendar cells detected\" or \"not detected\" (when it is confirmatory).")
    phi = inputs.get("phi")
    if phi:
        bullets.append("**Persistence.** Each block's fitted persistence (phi, corrected) is reported next to H-B: "
                       + "; ".join(f"{b} {_n(v.get('phi_score_median'), 3)} (raw {_n(v.get('phi_score_raw_median'), 3)})"
                                   for b, v in phi.items()) + ".")
    bounds = [f"{h} {_n(fam[h]['false_demotion_bound'], 3)}" for h in FAMILY_ORDER
              if fam.get(h, {}).get("false_demotion_bound") is not None]
    if bounds:
        bullets.append("**False-demotion bounds.** The probability, under independent rows, that the rule would replace "
                       "a correctly sized test by its estimand: " + ", ".join(bounds) + ".")
    bullets.append("**The status is computed, not decided.** `es.run_analogs.registration_family` applies the rule "
                   "below to the simulation and placebo rows (Rulings 29, 63, 84, 92 and 94); the result is the "
                   "`family` entry of `protocol_inputs.json`, which the sealed run reads.")
    out.append(_bullets(bullets))
    if fam.get("notes"):
        out.append("Notes recorded with the family:\n\n" + _bullets(fam["notes"]))
    if rule:
        out.append("**Rule constants and seeds** (`rule` in `protocol_inputs.json`):")
        out.append(_table(["Constant", "Value", "Meaning"],
                          [[k, _val(v), _RULE_MEANING.get(k, DASH)] for k, v in rule.items()]))
    return "\n\n".join(out)


def _he_text(inputs):
    sa, sb = SEALED["SA"].deploy, SEALED["SB"].deploy
    d_a, d_b = _days(sa), _days(sb)
    return "\n\n".join([
        "### H-E as run (Rulings 55, 68 and 69)",
        _bullets([
            f"**Strata.** The primary SELC strata are node x local hour x weekday class x month "
            f"(`{', '.join(STRATA_PRIMARY)}`), for the test statistic and for the headline pooled SELC. The sensitivity "
            f"strata are node x decile of the calendar-only model's score within the window (`{', '.join(STRATA_DECILE)}`); "
            "E4 reports SELC under them (Ruling 68).",
            "**Statistic.** Within a window and a stratum s, let N_s be the rows, A_s the alerted rows, L_s their loss "
            "and Q_s the sum of squared losses. The excess is the sum over strata of the captured loss minus "
            "(A_s / N_s) L_s. The test statistic is T = (sum of the windows' excesses) / sqrt(sum of the windows' "
            "variances), where a window's variance is the sum over strata of A_s (N_s - A_s) / (N_s (N_s - 1)) x "
            "(Q_s - L_s^2 / N_s): the variance of the captured loss when A_s of the N_s rows are chosen at random "
            "within the stratum, at the path's own alert counts. T = 0 when that variance is 0. The reported estimand "
            "stays pooled SELC = (excess_A + excess_B) / (L_A + L_B) (Ruling 55).",
            f"**Reference set.** In each window every node-hour's alert series is rotated circularly by the same whole "
            f"number of days s in the full cyclic group {{0, 1, ..., D - 1}}, with s = 0 the observed state (D = {d_a} days "
            f"for Test A, {d_b} for Test B). The two windows are rotated independently, and the p-value is "
            f"p = (1 + the number of non-identity grid cells with T at least the observed T) / (the number of cells): "
            f"{d_a} cells for Test A alone and {d_a * d_b} for the pooled test, so the smallest attainable p is "
            f"1/{d_a} for a single window and 1/{d_a * d_b} pooled (Rulings 55 and 69). If Test A's output is not "
            f"available, Test B tests H-E on its own window alone ({d_b} cells), and the result is not confirmatory "
            "(Rulings 112 and 116).",
            "**Validity.** The p-value is exact (a randomization p-value) under exchangeability of each window's "
            "observed alert path with its rotations, on a complete grid of days and node-hours. Where hours are "
            "missing, a rotation drops an alert whose target day and hour has no row, the rotations no longer form a "
            "group acting on the paths, and the guarantee is approximate; the share of alerts retained is reported "
            "for every window (`retained_share`, Ruling 55(e)).",
            "**What the null is not.** \"Timing within calendar cells\" includes any timing inside a month, among it "
            "the day-of-year seasonality that the model has learned from its training years; a rejection does not by "
            "itself show skill beyond the calendar. The paired E4 contrast of the alerts with the calendar-only model "
            "at the same count is the quantity that isolates timing beyond the calendar (Task 11 review M13).",
            "**What H-E measures (Ruling 118, Seat M, M12).** H-E tests the gross timing of losses within calendar cells, "
            "including timing that the model has learned from calendar features. It is not net value: the net quantities "
            "are the estimands NET and SNET (E1 and E2), and a rejection of H-E is not described as net value.",
        ]),
    ])


def _pooled_rates(rows, keys):
    """{key: (rejections, simulations)} summed over ``rows``, from the rates and n_sim of simulation.csv (a rate is the
    count over n_sim, so the count is recovered by rounding); None unless every row has n_sim and every key."""
    if not rows or any(not r.get("n_sim") or any(r.get(k) is None for k in keys) for r in rows):
        return None
    return {k: (sum(round(r[k] * r["n_sim"]) for r in rows), sum(int(r["n_sim"]) for r in rows)) for k in keys}


def _binom_tail(k, n, p):
    """P(X >= k) for X ~ Binomial(n, p). The same sum as ``es.run_analogs._binom_upper`` (a test holds the two equal)."""
    return float(sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(max(k, 0), n + 1)))


def _least_demoting_count(n, a, factor, rows, level):
    """The least number of rejections among ``n`` simulations at nominal level ``a`` that the registration rule demotes
    (Ruling 63, 84(b)): the rate exceeds ``factor * a``, and a one-sided binomial test of the count against ``a`` rejects at
    ``level / rows``. None if no count does. The same rule as ``es.run_analogs._smallest_demoting_count`` (a test holds the
    two equal), written out here so that the sealed path does not import the development runner."""
    return next((k for k in range(1, n + 1) if k / n > factor * a and _binom_tail(k, n, a) < level / rows), None)


def _operating_characteristic(rows, rule):
    """What the registration rule can detect at the replication count of the study (Seat M, M13): the least number of
    rejections that demotes a row at each nominal level, and the probability that a row whose true size is ``factor`` times
    nominal reaches it. Empty unless the rule's constants are in the inputs."""
    factor, levels, binomial = rule.get("factor"), rule.get("levels"), rule.get("binomial_level")
    n_sim = min((int(r["n_sim"]) for r in rows if r.get("n_sim")), default=0)
    if factor is None or not levels or binomial is None or not n_sim:
        return ""
    k_rows = (rule.get("k_rows") or {}).get("H-E") or len(rows)            # the rows the rule read for H-E
    cells = []
    for a in levels:
        k = _least_demoting_count(n_sim, a, factor, k_rows, binomial)
        cells.append((a, k, _binom_tail(k, n_sim, factor * a) if k else 0.0))

    def level(a):
        return f"{ALPHA:g}/3" if abs(a - ALPHA / 3) < 1e-9 else f"{a:g}"

    reads = " and ".join(f"{k} of {n_sim} ({_n(k / n_sim, 4)}) at {level(a)}" for a, k, _ in cells if k)
    chance = " and ".join(f"{_n(p, 2)} at {level(a)}" for a, k, p in cells if k)
    return (f"**What the registration rule can detect (Ruling 118, Seat M, M13).** The rule demotes a row only when its "
            f"rejection rate exceeds {_g(factor)} times nominal and a one-sided binomial test of its count against nominal "
            f"rejects at {_g(rule.get('binomial_level'))}/{k_rows}, the level divided by the {k_rows} rows read. With "
            f"{n_sim} simulations per row that takes at least {reads}. A row whose true size is {_g(factor)} times "
            f"nominal is therefore demoted with probability {chance} (the exact binomial tail). The rule is built "
            "against a false demotion rather than to catch a modest oversize. For H-E that costs nothing, because its "
            "pooled rows are at nominal.")


def _size_text(inputs):
    sim = inputs.get("simulation") or []
    third, five = f"reject_{ALPHA / 3:g}", f"reject_{ALPHA:g}"
    rows = [r for r in sim if r.get("test") == "H-E pooled A+B, size" and r.get("strata") in (None, "primary")]
    bullets = []
    note = next((r.get("note") for r in sim if r.get("test") == "H-E pooled A+B, power" and r.get("note")), None)
    power_note = {"target below the grid": " The effect target (half of the pooled analogue SELC, which is negative in "
                                           "development) lies below the grid of couplings, so the power read at the "
                                           "lowest coupling does not describe a realistic effect.",
                  "target not bracketed": " The effect target lies above the grid of couplings, so the power read at "
                                          "the top coupling is a lower bound."}.get(note, "")
    if rows:
        def worst(key):
            return max(rows, key=lambda r: r.get(key) if r.get(key) is not None else -1.0)
        w3, w5 = worst(third), worst(five)
        where = ", ".join(str(w3[c]) for c in ("scenario", "persistence") if w3.get(c))
        count = (f" ({round(w3[third] * w3['n_sim'])} of {w3['n_sim']} replications; {where})"
                 if w3.get("n_sim") and w3.get(third) is not None and where else "")
        pooled = _pooled_rates(rows, (third, five))
        bullets.append(f"**H-E's size is at or near nominal.** The worst of the {len(rows)} rows that the rule reads for "
                       f"the pooled test (primary strata) rejects {_n(w3.get(third), 4)} at {ALPHA:g}/3{count} and "
                       f"{_n(w5.get(five), 4)} at {ALPHA:g}. " + (
                           f"Pooled over those {len(rows)} rows, the rejection rate is "
                           f"{pooled[third][0]}/{pooled[third][1]} = {_n(pooled[third][0] / pooled[third][1], 4)} at "
                           f"{ALPHA:g}/3 and {pooled[five][0]}/{pooled[five][1]} = "
                           f"{_n(pooled[five][0] / pooled[five][1], 4)} at {ALPHA:g}. " if pooled else "")
                       + "The protocol makes no claim that the test is conservative.")
        operating = _operating_characteristic(rows, inputs.get("rule") or {})
        if operating:
            bullets.append(operating)
    fam, laws = inputs.get("family") or {}, (inputs.get("rule") or {}).get("laws")
    demoted = [h for h, _ in HB_PAIRS if fam.get(h) and not fam[h].get("confirmatory")]
    if any("simulated size" in r for h in demoted for r in fam[h].get("reasons") or []):
        bullets.append(f"**Why {' and '.join(demoted)} {'is an estimand' if len(demoted) == 1 else 'are estimands'}.** The "
                       "size reasons in the table come from the calibration-and-deployment bootstrap under persistence, "
                       "with and without a common deployment shift of both blocks (the common-drift rows of the size "
                       "study, Rulings 63(b) and 84(b)). They are a property of the bootstrap, not a finding about the "
                       "blocks.")
    if laws:
        bullets.append("**Rows each H-B test reads** (Ruling 94, M3): " + "; ".join(f"{h} reads the laws {', '.join(ls)}"
                                                                                    for h, ls in laws.items()) + ".")
    bullets += [
        "**Persistence correction.** The daily persistence of the simulated score and profit and loss is corrected by "
        "indirect inference (Ruling 84(a)): the simulation is run on the fit calendar, the same de-meaning and noise "
        "share are applied, and the lag-one autocorrelation is matched. The persistence \"upper end\" is "
        "phi + 1.96 sqrt((1 - phi^2) / n) per node, capped at 0.99, with the stationary SD held fixed (Ruling 63(a)). "
        "That upper end covers the true persistence in about 80 to 90 per cent of fits, because the corrected "
        "estimator's SD exceeds the formula's standard error (Task 17 re-review N-1); the size at the upper end is a "
        "sensitivity, not a bound.",
        "**Assumptions of the size study (Ruling 118, Seat M, M8).** The simulation draws the nodes independently, with "
        "Gaussian shocks and AR(1) days. A supplementary check by the G1 methodology review, which is not part of the "
        "registration rule and is not in the inputs, ran the registered rotation test (full cyclic group, studentized "
        "statistic, primary strata) on synthetic data of 12 nodes in 3 markets and two 92-day windows, 500 replications "
        "per scenario, under departures that the study does not contain: a market-wide common daily factor (0.8 of the "
        "daily variance) in the score and in the profit and loss, market-wide spike days, daily persistence 0.95, and "
        "independent trends within each window. The pooled rejection rates were at most 0.026 at 0.05/3 and 0.068 at "
        "0.05, and the single-window rates at most 0.016 and 0.054; none is significantly above nominal on the "
        "registered binomial check (one-sided, at 0.05/4). The exact guarantee of the rotation test needs each window's "
        "alert path to be exchangeable with its rotations, so a drift within a window is the departure it is least "
        "protected against: the two trend scenarios gave the highest pooled rates at 0.05. The probes and their outputs "
        "are archived in `docs/reviews/g1_probes/`.",
        "**Placebo pairings.** Only the 1-year pairings count (Ruling 94): their profit and loss comes from the "
        "calibration year, outside the alert model's training window. The pooled 1-year check demotes H-E if and "
        f"only if its p is at most {ALPHA:g}/3. The 2-year pairings are descriptive: their P&L quarter lies inside the "
        "training window and the day-of-year features let the alerts reproduce that year's spike days, so a 2-year "
        "pairing is not a no-skill null. Alignment is by calendar date shifted one year, which equals day-of-quarter "
        "alignment except for one March day in leap-year pairings (Rulings 94 and M2).",
        "**Power.** H-E stays in the family whatever its power, and the rule never reads it; its power is reported as "
        "a curve over the coupling grid." + power_note + " H-B power is read at the first Holm level at half the AQ3 "
        "median gap (Rulings 29, 84 and 94).",
    ]
    return "\n\n".join(["### Size study, power and placebo checks", _bullets(bullets)])


def _estimands_text():
    """What E1 to E4 measure (Rulings 81 and 118): the E4 contrasts, the ESred rows that go with them, and the readings that
    the paper must not exceed (Seat M, M6 and M12)."""
    comp = " and ".join(f"`{c}`" for c in E4_COMPARATORS)
    return "\n\n".join([
        "### E1 to E4: what each estimand measures (Rulings 81 and 118)",
        _bullets([
            "**E4 (Ruling 81).** E4's registered quantities are the paired contrasts dSELC = SELC(P1) - SELC(comparator) "
            f"for the comparators {comp}: the calendar-only model's cutoff and VOL (the trailing 168-hour mean |DART|, "
            "lagged 48 hours), each set to P1's alert count per node. The contrast is computed inside every bootstrap "
            "replicate, so its interval is paired and uses the same resampled days. The comparators' own SELC levels are "
            "context.",
            "**E4's ESred rows (Ruling 118, Seat M, M6).** For each of the same comparators c, E4 also reports ESred(c) "
            "(`E4:ESred|c`, context) and the paired contrast dESred|c = ESred(P1) - ESred(c) (`E4:dESred|c`), computed "
            "inside every replicate like dSELC. dESred|c is the downside protection of the alerts beyond a comparator at "
            "matched activity: it separates what the alerts add from what a rule with the same number of alerts per node "
            "(the calendar-only model, or VOL) already removes. Both are tail measures, so their intervals are labelled "
            "approximate.",
            "**E4's sensitivity.** SELC of P1 under the decile strata (`E4:SELC_decile`) is a sensitivity only. It is "
            "biased upward under calendar-only nulls (the decile rows of the size study show the bias, Ruling 68), so no "
            "decision reads it (Ruling 118, M12).",
            "**E3 (ESred).** E3 is the tail effect of the policy, including plain exposure reduction and calendar "
            "allocation. It is not evidence of skill. The matched comparisons are SELC, SNET, E1's timing part and E4 "
            "(Ruling 118, M6).",
            "**E1's level effect.** E1 splits V(P1) - V(P4) into a timing part (P1 against P4@r*, which has P1's alert "
            "count) and a level part (P4@r* against P4). The level effect is a decomposition relative to P4, not a causal "
            "cost of a drifting cutoff (Ruling 118, M12).",
        ]),
    ])


def _levels_text():
    """The interval of a level, E5 and H-S (controller Ruling 118; Seat M, M2): the calendar-aligned percentile interval is
    the primary one, the independent-draw interval of H-B a diagnostic beside it."""
    return "\n\n".join([
        "### E5 and H-S: the interval of a level (Ruling 118)",
        _bullets([
            "E5 (each block's median |log10 R| and failure count) and H-S (the failure counts of the recalibrated plain DART "
            "lags) are levels of one window, not contrasts. Their bootstrap is a calibration-and-deployment bootstrap, as "
            "H-B's is, with one difference: a single circular stationary draw over the quarter's calendar, indexed by "
            "(month, day), weights the calibration days and the deployment days alike. The two windows are the same "
            "calendar quarter a year apart, so every replicate keeps their matched calendar composition. The calendar "
            "index is the sorted union of the (month, day) positions of the two windows' days, so a day that only one "
            "window has (a leap day) takes its count from the same draw.",
            "Drawing the two windows independently, as the H-B bootstrap does, re-randomizes that composition. For any "
            "score with a calendar component, which every block has, the mismatch alone moves the cutoff against the "
            "deployment rate and inflates |log10 R| in the replicates, so the independent-draw interval measures the "
            "re-randomization of the calendar and not only the sampling variation of the quarter; for a score that is "
            "deterministic in the calendar it sits above a realised error that is near zero by construction. The "
            "comparison on development windows is in the probes archived in `docs/reviews/g1_probes/`.",
            "The primary interval of E5 and H-S is the aligned percentile interval: the `ci` field of H-S and, in an E5 "
            "row, `lo` and `hi` (the median |log10 R|) and `failures_lo` and `failures_hi` (the failures). The interval of "
            "the independent-draw scheme is stored beside it as a diagnostic: `ci_independent` for H-S, and "
            "`lo_independent`, `hi_independent`, `failures_lo_independent` and `failures_hi_independent` in an E5 row.",
            "Every E5 and H-S interval is labelled approximate (`approximate` is true whatever the NaN count): the size "
            "study evaluated the independent scheme, and the coverage of the aligned scheme is not studied. They are "
            "resampling ranges for the realised quarter, not intervals of studied coverage.",
            "H-B1, H-B2 and PC keep the independent scheme, the one the size study evaluated.",
        ]),
    ])


def _monitors_text():
    return "\n\n".join([
        "### Drift monitors (Rulings 66(b), 73, 83, 93, 104 and 109)",
        _bullets([
            f"Monitors: {', '.join(MONITORS)}. The inputs of the monitors come from deployment days 1 to "
            f"{MONITOR_DAYS} (counted from each node's first deployment day), and the outcome, |log10 R| and node "
            f"failure, from day {MONITOR_DAYS + 1} to the end of the window, with the calibration cutoff unchanged.",
            f"The domain classifier is a balanced logistic regression with a class-stratified {N_FOLDS}-fold "
            "cross-validation blocked by local settlement day (Ruling 73), which amends the \"blocked by week\" of "
            "spec section 5 item 5.",
            "The 1-Wasserstein monitor is the July pipeline's statistic (Ruling 44). With no drift the day-blocked AUC "
            "is somewhat above 0.5 for persistent inputs: about 0.55 for AR(0.9) features in simulation, on every "
            "window length tried, and up to about 0.9 for 168-hour rolling means (Ruling 93).",
            "The placebo reference columns (the same statistic on a split inside the calibration window) are "
            "descriptive: they are not monitors, have no p-value and enter no BH family (Ruling 83). Only ranks across "
            "nodes are interpreted.",
            "In the sealed run a node whose windows cannot carry the monitors (no calibration or deployment rows with "
            f"complete features, fewer than {MONITOR_DAYS} early days, or no later day), or that the block's score frame "
            "does not cover, is skipped and counted, not raised (Rulings 104(d) and 109, M3); the skipped nodes and "
            "their reasons are listed under `results[\"secondary\"][\"drift monitors\"][\"skipped_nodes\"]`.",
            "The Spearman p-values of the drift-monitor family treat the nodes as independent, although nodes within an "
            "operator co-move, so they, and the BH decisions that use them, are optimistic (Ruling 40's caveat; spec "
            "section 5 item 2, Ruling 118, Seat M, M9).",
        ]),
    ])


def _bh_text():
    doc = _secondary.__doc__ or ""
    if "```" in doc:
        raise ValueError("es.secondary's docstring contains a code fence; it cannot be quoted in a fenced block")
    return "\n\n".join([
        "### Secondary families (Benjamini-Hochberg; Rulings 37, 80, 105(c) and 118)",
        "The three families are defined by `es.secondary`; its module docstring follows, word for word.",
        "```text\n" + doc + "\n```",
        _bullets([
            "BH's m is the number of rows with a p-value, the `included` rows of `secondary_bh.csv`: a NaN p-value is "
            "no test that ran, so it is left out of its family's adjustment and never rejected (Ruling 80).",
            "**The audit-robustness family is descriptive, as H-B is (Rulings 110 and 118, Seat M, M5).** This text "
            "governs the `es.secondary` docstring quoted above, which describes the family as adjusted by BH: the flags "
            "are computed, as the docstring says, but they are diagnostics (Seat M, M16; the seed variants that spec "
            "section 6.3 lists under Secondary belong to this family). Its p-values come "
            "from the H-B calibration-and-deployment bootstrap, whose simulated size exceeds nominal, so its BH flags are "
            "diagnostics: they are not read as control of the false-discovery rate (Ruling 105(c)), and no count of its "
            "rejections is reported or interpreted. The family is computed and stored as registered (`secondary_bh.csv`, "
            "and the counts that `results.json` keeps for every family are a record of what the run computed, not a "
            "result).",
            "**The economic-robustness family counts a repeated result once (Ruling 118, Seat M, M9).** Rejections that "
            "repeat one result across fee levels, for example the same position at every fee level, are reported as one "
            "finding. Its \"P1 premium fees=calibrated\" row is H-E's own single-window statistic, not a second H-E result "
            "(see Part G).",
        ]),
    ])


def _fees_text():
    rows = [[m, _n(CALIBRATED_FEES[m][1], 2), _n(CALIBRATED_FEES[m][-1], 2)] for m in CALIBRATED_FEES]
    return "\n\n".join([
        "### Fees (Rulings 78 and 90)",
        "Every fee level comes from a source dated on or before 30 June 2026 (the workspace rule; spec invariant I4 "
        "was reworded to match). Each component takes its latest value published on or before that date, rounded to "
        "0.05. MISO omits the Schedule 17 and Schedule 24 administrative adders, because no source dated on or before "
        "June 2026 could be retrieved; the uniform grid covers higher fee levels. The final adopted dict is given in "
        "the controller note at the top of `docs/fee_rederivation_2026-10-23.md`; the adopted values differ from that "
        "record's section 3 recommendation for MISO.",
        "**The binding levels** are those of `es.positions.CALIBRATED_FEES`, read from the code when this protocol is "
        "rendered. They are the levels the sealed run uses. The fee table of spec section 4.2, copied in Part A, is "
        "superseded by them (its own note says so); where the two differ, this table governs.",
        _table(["Market", "INC ($/MWh)", "DEC ($/MWh)"], rows),
        f"Uniform sensitivity grid (`UNIFORM_FEES`): {', '.join(_g(c) for c in UNIFORM_FEES)} $/MWh. Stress "
        "sensitivity: the fee doubles on a node-day whose lagged 168-hour mean |DART| at its first hour is at or above "
        "the node's training 90th percentile.",
    ])


def _run_text(fam, env_bound=True):
    n_sets = len(score_keys())
    seeds = ", ".join(str(s) for s in SEEDS)
    n_rep = _default(run_sealed, "n_rep")
    ids = " or ".join(f"`{t}`" for t in _sealed("TEST_IDS"))
    registered_reps = _sealed("N_REP")
    free = _gib(_sealed("MIN_FREE_BYTES"))
    scope_a, scope_pooled, scope_alone, scope_own = (
        _sealed(n) for n in ("SCOPE_SINGLE", "SCOPE_POOLED", "SCOPE_B_ALONE", "SCOPE_B_OWN_QUARTER"))
    e5_scope = _sealed("SCOPE_E5")
    scheme, interval_e5 = _sealed("SCHEME_E5_HS"), _sealed("INTERVAL_E5_HS")
    _, b_first, b_last, _ = SEALED_WINDOWS["B"]
    cover_expected = expected_hours("NYISO", b_first, b_last)          # the clock hours of Test B's deployment days
    cover_kept = cover_expected - 2                                      # a fall-back day that keeps 23 of its 25 hours
    folder = {t: _repo_path(default_config(t).out_dir) for t in ("A", "B")}      # the registered output folders (M24)
    prior_folder = _repo_path(default_config("B").prior_test_dir)                  # where Test B reads Test A's output
    return "\n\n".join([
        "### The sealed pull and the sealed run (Rulings 15, 28, 33, 58, 60, 70, 82, 104, 106, 109, 111, 112, 116, 118, 121 and 122)",
        "**Pull** (`es.sealed_pull.pull_sealed`). It runs once per window and only after `verify_registration` has read "
        "`docs/sealed/REGISTRATION.md`, which it takes from `es.registration`'s own paths and never from a record handed "
        "to it. `verify_registration` also refuses a protocol that is not ready to register (Ruling 122, C2-1), so the "
        "pull refuses an unfilled knowledge statement before any download. NYISO, ISO-NE and MISO use the July "
        "downloaders unchanged with a new, explicit root; CAISO uses exact "
        "local-day UTC bounds, retries and an empty-result check, and pulls the FMM prices; NYISO's integrated "
        "real-time LBMP is pulled too. Each step is trimmed to the pull's own days and checked before it is recorded as "
        "done: on every local settlement day, every node of the node map must have exactly as many rows as the day has "
        "clock hours (24; 23 or 25 when the clocks change, MISO keeping fixed EST), every one of them on the hour "
        "(minute and second 0, in UTC) and with a finite price, in the price files and in the FMM and NYISO integrated "
        "tables (an FMM hour counts only with all four of its 15-minute intervals). A row off the hour is not an hour, and "
        "is logged. A short day, and a day with more rows than hours (an hour twice), both fail, and either is excused "
        "only by naming the day in `allow_missing_days`, which is logged. On a fall-back day NYISO, ISO-NE and the NYISO "
        "integrated table expect 23 hours, because the July NYISO and ISO-NE downloaders and the NYISO integrated "
        "parser drop both rows of the repeated hour; CAISO and its FMM table expect 25. CAISO's real-time hour is the "
        f"mean of its {RT_INTERVALS_PER_HOUR} five-minute prices, and an hour with fewer than that, or with a price that is "
        "not finite, is NaN, so it is not an hour for this check (Ruling 116, Mn4). The pre-deployment rows are compared "
        "with the files already opened; the comparison is logged (`OVERLAP_CHECK.csv`, `PULL_LOG.txt`) and is not "
        "enforced, because ISOs revise settled prices.",
        "The pull writes `PULL_LOG.txt`, `COVERAGE.csv` (`hours` per node, counting the rows on the hour that have "
        "finite prices, against `expected_hours`, with the count of non-finite rows), `OVERLAP_CHECK.csv`, the builders' "
        "logs under `logs/` and a `MANIFEST.sha256` over its folder. In `COVERAGE.csv`, `expected_hours` is the number of "
        "clock hours of the deployment window's local settlement days (23 on a spring-forward day, 25 on a fall-back "
        "day), which is not what the day check expects on a fall-back day. Test B holds one, 1 November 2026, so there "
        f"a NYISO, ISO-NE or integrated node with the 23 hours of that day shows {cover_kept:,} of {cover_expected:,} "
        f"hours, a share of {cover_kept / cover_expected:.5f} and not 1; the coverage rule of the run (at least 90%) "
        "does not notice the difference. The pull is resumable but still once: raw responses are cached "
        "under `<root>/raw/`, finished steps in `.state.json`, and every attempt is logged. The NYISO integrated "
        "archive of a month has no interval grid to be short of, so it is cached whatever hours it holds and reused as "
        "it is on every resume; when the day check refuses the NYISO integrated table, its message names the cached "
        "archive of each month concerned, and deleting that file makes the next resume fetch the month again "
        "(Ruling 121, F4). A root that has files is "
        "resumed only from its `.state.json`; without it the pull is refused and never started over. A finished pull "
        "whose record in `PULLS.json` was not written (it was killed after its manifest) is adopted by a call with "
        "`resume=True`, which verifies its manifest and its log and downloads nothing. A completed pull "
        "is recorded in `docs/sealed/PULLS.json` (root, manifest SHA-256, times, protocol SHA-256 and the id and kind of the "
        "registration it ran under) and refuses any further call; Test B's pull needs Test A's completed pull, with a "
        "manifest that still verifies (Rulings 33, 58 and 106).",
        f"**A chunk that never comes complete (Rulings 111 and 121).** An answer that the server gave and that can be "
        f"read is whole; it is refused if the validator refuses its rows (short of the interval grid, outside the "
        "bounds asked for, a price that is not finite) or if it holds no rows at all (OASIS's no-data report, a CSV "
        "without rows, a zip without a CSV). A network failure (a timeout, a connection error, an HTTP error status) "
        f"and an answer that cannot be read are no whole answer. A chunk that gives no good answer in "
        f"{CHUNK_ATTEMPTS} attempts stops the pull, and `PULL_LOG.txt` names the chunk and says how many of its "
        "attempts were refused answers. A later attempt (`resume=True`) may name in `allow_incomplete_chunks` a chunk "
        f"for which the log says that all {CHUNK_ATTEMPTS} attempts were refused answers, and the chunk is then "
        "accepted only if every one of the attempts of that call brought a whole answer that was refused: the answer "
        "with the fewest problems (the last of equals) is accepted, short or with non-finite prices, and never cached, "
        "and a chunk whose answers held no rows is accepted as zero rows; if any attempt of the call was a network "
        "failure or an unreadable answer, the pull stops. Every missing or non-finite interval is logged, and the chunk "
        f"and the reason go to `.state.json` and `PULLS.json`. A chunk that did not have all {CHUNK_ATTEMPTS} attempts "
        "refused in an earlier attempt is requested and validated as usual, and the days that an accepted chunk "
        "leaves short still have to be named in `allow_missing_days`: two separate decisions, both logged.",
        "**Run** (`es.sealed_run.run_sealed`). Before the lock, and reading no price row (nothing is created either):",
        _bullets([
            f"the test id must be {ids}, and the run is refused if `docs/sealed/RUNS.json` already holds a record of the "
            "test or the claim file `RUN_<test>.claim` exists, whatever the output folder (Rulings 104(a) and 109, C1 "
            "and I9);",
            "the configuration is checked: the window end and the pull folders named are the test's own or earlier "
            "tests'. A configuration with a sealed arm must be the registered one (the rolling and frozen arms, the "
            f"sealed pull folders and the window end of `default_config(test_id)`), must use the registered "
            f"{registered_reps} bootstrap replicates, must read `es.registration`'s own protocol and registration paths, "
            "must read the registered development data (its `dev_dirs` resolve to the registered folders, "
            "`DEV_DATA_DIRS`; other folders, or none, would end the run in its load or at the coverage rule), must write "
            f"the registered output folder (`{folder['A']}` for Test A and `{folder['B']}` for Test B, the `out_dir` of "
            "`default_config(test_id)`) and, for Test B, read Test A's registered folder "
            f"(`{prior_folder}`, the `out_dir` of `default_config(\"A\")`), both compared as resolved paths, because "
            "Test B finds Test A's output through the folder that `RUNS.json[\"A\"]` records and must be pointed at it "
            "(Ruling 122, N2b), and must take its code from the folder "
            "that `es` was imported from, so that the code manifest describes the code that runs (Ruling 109, C1, C2, "
            "I2 and M8; Ruling 116, N4). The run is refused too if the output folder exists: with no claim file for its "
            "test it was never locked, and the refusal says that nothing was read and to remove the folder and run "
            "again; if its log shows that the run took its claim or began its data load, the refusal says that the test "
            "counts (Ruling 116, N3);",
            "`verify_registration` reads the registration record and this protocol as they are at that moment: the "
            "record must be well formed (it names exactly one registration, which is an OSF registration, a public "
            "release of the code repository or the author's local record; a record that gives both a DOI and an OSF "
            "address must give the same registration in both), not dated in the future, and name this protocol's "
            "SHA-256, which the file must still have;",
            "`verify_registration` also refuses a protocol that is not ready to register: `es.protocol.ready_to_register` "
            "must find nothing in the text of `PROTOCOL.md`, that is, no author marker (an `AUTHOR` placeholder of the "
            "template), exactly one tick in every item-1 row of the knowledge statement and no template guidance left "
            "in it. The refusal says that `PROTOCOL.md` is not ready to register, how many items are still open and the "
            "first two, and to fill in `docs/sealed/KNOWLEDGE_STATEMENT.md`, re-run `es.protocol.main()` and register "
            "that `PROTOCOL.md` (the SHA-256 in the `PROTOCOL.sha256` it writes) (Ruling 122, C2-1);",
            ("`verify_bound` requires the pre-pull cutoffs, the protocol inputs, the code manifest and `ENV_PINS.json` "
             "to have the SHA-256 printed in Part F;" if env_bound else
             "`verify_bound` requires the pre-pull cutoffs, the protocol inputs and the code manifest to have the "
             "SHA-256 printed in Part F;"),
            "the SHA-256 of every file of `CODE_MANIFEST.sha256` is recomputed: any difference refuses the run, "
            "unless `allow_departure=True`, which records the departing files (spec section 6.5 step 7; section 6.3, "
            "\"Test B runs the registered code\");",
            *(["the running environment is compared with `ENV_PINS.json`: a missing file refuses the run, and so does a "
               "different interpreter or a different version of python or of a pinned library; the interpreter is "
               "compared by the path it resolves to, so that python, python3 and a link to the same binary are one "
               "interpreter (Ruling 109(1), I7; Ruling 116, Mn2); the live environment is recorded in `results.json` and "
               "`RUNS.json`. `ENV_PINS.json` binds the interpreter path and the library versions of the environment in "
               "which this protocol was generated, the Anaconda installation at `/opt/anaconda3`: if it is updated, moved "
               "or reinstalled before Test B has run, in January 2027, so that the interpreter's path or the version of "
               "python or of a pinned library differs, Test B is refused before its lock until the environment is "
               "restored, so it must not be updated or moved until Test B has run (Ruling 122, C2-3);"]
              if env_bound else []),
            "each sealed pull folder's manifest (the exact file set and hashes) is verified and bound to the verified "
            "registration through `PULLS.json` (Ruling 70): the record of the pull must name that folder, its manifest "
            "SHA-256, this protocol and the registration id of the verified registration (for an OSF registration the id "
            "stays the same when a DOI is added after an embargo; a public release and a local record have ids of their "
            "own, so a pull made under one record is not read under another), the pull's own `PULL_LOG.txt` must not "
            "name another registration id, and the pull must have started at or after `registered_utc`;",
            "the registered inputs are read from the bound `protocol_inputs.json`: eta, and for each of H-E, H-B1 and H-B2 "
            "its `confirmatory` flag, its `reasons` and its `approximate` flag (and H-E's `interim_approximate`);",
            "for Test B, the record of Test A is read. Test B is refused if Test A was never started, that is, if "
            "`docs/sealed/RUNS.json` has no record of Test A and the claim file `RUN_A.claim` does not exist "
            "(Ruling 112); Test B must be pointed at the folder that the record names, in every state of the record, "
            "and is refused otherwise; and it is refused if the output folder that Test A's record names is complete "
            "(it has its `MANIFEST.sha256`) but the record lacks `finished_utc` or `manifest_sha256`, with the repair to "
            "make by hand and a pointer to the recorded folder (Ruling 116, N2; Ruling 122, N2b). If Test A's output is "
            "available, its record, manifest, `results.json` and stored P1 rows are verified too (see Failure handling);",
            "the development settlement tables (`caiso_fmm_dev` and `nyiso_rtint_dev` under the data root) must be "
            "there, because without them the premium direction would be set from the 5-minute settlement without a "
            "word; the folders are listed and no row is read (Ruling 109, M9);",
            f"at least {free} must be free where the output folder lives and where `docs/sealed` lives, because a full "
            "disk after the data are read would spend the run (Ruling 109, I8);",
            "the folder that holds the once-only records (`docs/sealed`) and the parent of the output folder must be "
            "writable, because a refusal to write after the claim would spend the test without a row read; nothing has "
            "been created when this refuses (Ruling 116, N3).",
        ]),
        "**The lock.** The first thing the lock does is to claim the test id: `docs/sealed/RUN_<test>.claim` is created "
        "exclusively (`O_CREAT|O_EXCL`), so that of two runs started together only one gets it, and it is never removed. "
        "Then the output folder is created and the first lines of `RUN_LOG.txt` are written (the start time, the test, "
        "the protocol, the registration, the number of replicates, the hashes of the bound files and of the sealed "
        "pulls' manifests, the departures, the environment and, for Test B, how Test A stands), and the run is recorded "
        "in `docs/sealed/RUNS.json` (test, start time, protocol SHA-256, output folder, registration identifier, its kind "
        "and time, a fingerprint of the code, the number of replicates, the departures and the environment). All of this happens "
        "before any price row is read, and the claim is what the loader of the sealed panel requires: nothing reads the "
        "sealed panel without it. From the claim on this is the one run: a crash leaves the claim (and the folder, and "
        "the record once it is written) behind, and a second call is refused, whatever the output folder (Ruling 104(a), "
        "Ruling 109, I1 and I9; Ruling 116, N3). Because the claim comes first, an output folder that exists with no "
        "claim file for its test was never locked. When the run finishes, its record in `RUNS.json` gets "
        "`finished_utc`, its status and the SHA-256 of the output folder's manifest, which is what Test B checks Test "
        "A's output against (Ruling 109, I3).",
        "**After the lock.**",
        _bullets([
            "The data are the development files and the pull folders, restricted to the frozen node map, deduplicated by "
            "market, node and timestamp (first row kept) and cut at the window end in local settlement days. Rows whose "
            "timestamp is not on the hour are dropped and counted per market and node, because the panel build refuses "
            "them (Ruling 109, M2); so are rows whose day-ahead or real-time price is not finite (Ruling 82). The rows "
            "read and each kind of row dropped (outside the node map, duplicate, beyond the window end, off the hour, "
            "non-finite) are counted in `results.json` and `RUN_LOG.txt`, and none of them raises.",
            f"Coverage rule (spec section 6.1): a node is usable with at least 90% of its expected deployment hours "
            f"(`expected_hours` in `tables/coverage.csv`: the clock hours of the deployment window's local settlement "
            f"days, as in the pull's `COVERAGE.csv`) "
            f"and a feasible fit (at least {MIN_POSITIVES} positive training labels). The test runs only if at least "
            f"{MIN_USABLE_NODES} nodes are usable and all four operators have one; otherwise the run writes "
            f"`NOT_RUN.md` with the reasons and its status reads \"{sealed_statuses()['not_run']}\".",
            f"The {n_sets} score sets are fitted directly and never read from the score cache: the blocks "
            f"{', '.join(BLOCKS)} under XGBoost (rolling-origin, seed 42); {', '.join(LOGIT_BLOCKS)} under logistic "
            f"regression; {', '.join(CUTOFF_BLOCKS)} on the frozen-origin arm; and those {len(CUTOFF_BLOCKS)} pipelines "
            f"again for the seeds {seeds} (seed 42 is primary). The registered cutoffs are then compared with the "
            "cutoffs of these fits (`tau_check`).",
            f"The bootstraps use {n_rep} replicates, the default of `run_sealed(n_rep=...)` and the registered "
            f"value, which a sealed run must use (mean block {MEAN_BLOCK} days, seed {SEED}; `es.run_economics` "
            f"N_REP = {N_REP}).",
            "H-B1 and H-B2 are stored with their estimate (`t_hat`), their percentile interval (`ci`), their basic "
            "interval (`ci_basic`), the numbers of nodes and of NaN replicates and the field `interval_registered`. A "
            "hypothesis that is confirmatory in the registered family also has the one-sided basic bootstrap p-value "
            f"(`p`), and its `interval_registered` reads \"{_sealed('INTERVAL_HB_TEST')}\". An estimand has no `p`: its "
            "bootstrap p-value is kept as a diagnostic, under `diagnostics.bootstrap_p_one_sided`, its "
            f"`interval_registered` reads \"{_sealed('INTERVAL_HB_ESTIMAND')}\", and its status reads "
            f"\"{_sealed('STATUS_HB_ESTIMAND')}\", followed by \": \" and the reasons of the registration rule, in both "
            "tests (Ruling 109, I5 and M11; Ruling 110).",
            "E5 and H-S are stored with the calendar-aligned percentile interval as their primary interval and the "
            "interval of the independent-draw scheme beside it as a diagnostic (Ruling 118; see \"E5 and H-S\" above). "
            "H-S stores `ci` and `ci_independent` as lists. An E5 row stores flat columns: `lo` and `hi` (the median "
            "|log10 R|) and `failures_lo` and `failures_hi` (the failures) for the aligned interval, and `lo_independent`, "
            "`hi_independent`, `failures_lo_independent` and `failures_hi_independent` for the independent draws, with the "
            "NaN counts of each draw. Both carry `scheme` (\"" + scheme + "\") and `interval_registered` (\""
            + interval_e5 + "\"), and `approximate` is true in every case, whatever the NaN count. H-B1, H-B2 and PC "
            "keep the independent scheme.",
            "The economic analysis (`economics_for`) is run with the development settlement tables and those of the "
            "sealed pulls. If it fails as a whole, H-E's rows are built on their own (`p1_rows_for`), in a guard of their "
            "own, so that one failing comparator or exposure split does not cost the run its one confirmatory test, and "
            "they are stored as `kept/P1.parquet`, which Test B pools H-E from (Ruling 109, I6; Ruling 116, N1).",
            "Comparator and policy rows are restricted to P1's rows and the dropped rows are counted (Ruling 82). "
            "Estimands are also computed without CAISO (Ruling 41).",
            f"In Test A, E1 to E4 and H-E are computed on the quarter alone (`estimands_scope`: \"{scope_a}\"). In Test B "
            f"they pool with Test A's stored rows when Test A's output is complete (\"{scope_pooled}\"); E1 to E4 are "
            f"then also reported for the quarter alone, and {_holm_clause(fam)} (spec section 6.3). H-E is pooled from "
            "Test A's stored P1 rows alone: when Test A's other stored frames, the ones E1 to E4 are computed from, are "
            f"missing, H-E is still pooled, E1 to E4 cover Test B's own quarter (\"{scope_own}\") and "
            "`results[\"test_a\"][\"missing_kept\"]` lists what is missing (Ruling 116, N1). When Test A's output is not "
            f"available they are Test B's window alone (\"{scope_alone}\"). E5, H-B, PC and H-S are always this run's "
            "own window, and `results.json` labels E5's scope (`E5_scope`: "
            f"\"{e5_scope['A']}\" or \"{e5_scope['B']}\"; Ruling 109, M5).",
            "After its own lock, Test B reads each stored file of Test A through Test A's `MANIFEST.sha256`: the file "
            "must be listed there with the SHA-256 listed, or the analysis that reads it records the error (Ruling 116, "
            "Mn3).",
        ]),
    ])


def _failure_text(env_bound=True):
    """Failure handling (Rulings 104, 109, 112, 116 and 122), with the status vocabulary of es.sealed_run."""
    st = sealed_statuses()
    alone = {k: _sealed(k) for k in ("STATUS_HE_ALONE", "WORDING_HE_ALONE", "SCOPE_B_ALONE", "SCOPE_B_OWN_QUARTER",
                                     "STATUS_HB_NO_TEST_A", "STATUS_HB_NEEDS_HE")}
    states = [_sealed(k) for k in ("PRIOR_POOLED", "PRIOR_NOT_RUN", "PRIOR_NO_P1", "PRIOR_UNFINISHED", "PRIOR_UNRECORDED")]
    free = _gib(_sealed("MIN_FREE_BYTES"))
    table = _table(["Status (`results[\"status\"]` in `results.json`)", "When"], [
        [f"\"{st['completed']}\"", "no analysis failed."],
        [f"\"{st['secondary']}\"", "only secondary analyses failed: the three Benjamini-Hochberg families (audit "
                                    "robustness, economic robustness, drift monitors) or their tables. The others are "
                                    "reported as usual."],
        [f"\"{st['errors']}\"", "an analysis outside the secondary families failed too (a score set, the cutoff check, "
                                 "a bootstrap, the economics, the estimands, H-E or the decision). The paper reports what "
                                 "completed and states what failed and why (`results[\"errors\"]`, `RUN_LOG.txt`)."],
        [f"\"{st['not_run']}\"", "the coverage rule failed: `NOT_RUN.md` gives the reasons and no analysis ran, as "
                                  "registered (spec section 6.1)."],
    ])
    return "\n\n".join([
        "### Failure handling (Rulings 104, 109, 112, 116 and 122)",
        "An exception in the data load, in the panel build or in the computation of the coverage table aborts the run "
        "(and a crash counts, as below); a coverage rule that is not met is no exception, but the registered NOT_RUN. "
        "Every analysis after the coverage rule runs in its own guard: an exception is logged in `RUN_LOG.txt`, recorded "
        "under the analysis' name in `results.json` (in `results[\"secondary\"]` for a secondary analysis), and the "
        "others go on. The run ends with one of these statuses:",
        table,
        _bullets([
            "**Coverage failure.** It means NOT_RUN, as registered; it is not an error. An analysis in which no node "
            "qualifies raises, and that is recorded as the analysis' own error without stopping the others "
            "(Ruling 104(f)).",
            "**Test B before Test A (Rulings 112 and 116, N2).** Test B is refused before its lock, and nothing is spent, "
            "unless Test A has been started: `docs/sealed/RUNS.json` holds a record of Test A, or Test A's claim file "
            "`RUN_A.claim` exists. A Test B that ran before Test A, by mistake, would lose the pooled confirmatory test of "
            "H-E, which a Test A run later could still have given. The folder that Test B is pointed at does not change "
            "this: a folder that looks like Test A's output, without a record of the run that made it, is not Test A's "
            "output. Test B is refused too when the output folder that Test A's record names is complete (it has its "
            "`MANIFEST.sha256`) but the record in `RUNS.json` lacks `finished_utc` or `manifest_sha256`: running alone "
            "would lose the pooled H-E for a state that no data read caused (Ruling 122, N2b). The refusal gives the repair, which is to set `finished_utc` and `status` in "
            "the record to the values in Test A's `results.json` and `manifest_sha256` to the SHA-256 of Test A's "
            "`MANIFEST.sha256`, and to run Test B again.",
            "**Test B without Test A's output (Rulings 109, 112 and 116).** Once Test A has been started, Test B runs even "
            "if Test A's output is not available: if Test A is NOT_RUN (its coverage rule failed), or was claimed and left "
            "no manifest and no finished record (it crashed, or its record was never completed), or finished without "
            "stored P1 rows, Test B runs alone. H-E is then tested on Test B's window alone and reported as not "
            "confirmatory, because the registered pooled test is unavailable: "
            f"its status reads \"{alone['STATUS_HE_ALONE']}\" and its wording \"{alone['WORDING_HE_ALONE']}\". E1 to E4 "
            f"are Test B's window alone (\"{alone['SCOPE_B_ALONE']}\"), nothing is replicated and no Holm step-down is "
            f"run, and a confirmatory H-B test reads \"{alone['STATUS_HB_NO_TEST_A']}\". `results.json` says how Test A "
            "stood (`test_a`, with the state " + ", ".join(f"\"{s}\"" for s in states[:-1]) + f" or \"{states[-1]}\").",
            "**Test A's other stored frames (Ruling 116, N1).** H-E is pooled from Test A's stored P1 rows alone. If P1 is "
            "there but the other stored frames that E1 to E4 are computed from are not, Test B still pools H-E, its E1 to "
            f"E4 cover its own quarter (\"{alone['SCOPE_B_OWN_QUARTER']}\"), and `results[\"test_a\"][\"missing_kept\"]` "
            "lists what is missing. Test A's fallback for a failed economic analysis stores its P1 rows "
            "(`kept/P1.parquet`), so that this pooling is possible.",
            "**Test B and the H-B estimands (Ruling 109).** Test B never depends on Test A's demoted (estimand) H-B "
            "results, and a failed Test A estimand does not block Test B: Test A's `p` and `t_hat` are needed only for an "
            "H-B hypothesis that is confirmatory in the registered family.",
            "**H-E fails in Test B.** The registered step-down needs H-E's p-value. If H-E did not complete, H-E is "
            "recorded as an error and no Holm decision is made: `results[\"holm\"]` says so, and a confirmatory H-B test "
            f"reads \"{alone['STATUS_HB_NEEDS_HE']}\" (Ruling 104(c)).",
            "**Test A's stored output.** Where Test A's output is complete and its run happened, Test B pools H-E with "
            "Test A's stored P1 rows and, when the other stored frames are there, E1 to E4 with them (`kept/`), and checks "
            "them before its own lock: the record of Test A in `docs/sealed/RUNS.json` must name Test A's output folder, "
            "the SHA-256 of its manifest (which must still verify), its start time and this protocol; Test A's "
            "`results.json` must be Test A's own, of that start and under this protocol; and the stored P1 rows must be "
            "there (without them Test B runs alone). Stored rows that fail the check stop Test B before the lock, so "
            "that nothing is spent, and the paper then reports Test B as not run, with the reason. After its own lock "
            "Test B reads each stored file again through Test A's manifest (Ruling 116, Mn3). The last write of Test A's "
            "run puts `finished_utc`, its status and its manifest hash into its record; if that write fails, the output "
            "is complete, the error gives the fields to add to the record by hand, and until they are added Test B is "
            "refused (it does not run alone).",
            "**Interruption.** An interruption after the claim (Ctrl-C or a crash) spends the run, as registered: a run "
            "that started counts, its claim file `RUN_<test>.claim` (never removed) stays in `docs/sealed`, with its record "
            "once the record was written, and a second call is refused whatever the output folder (Ruling 104(a), "
            "Ruling 109, I9; Ruling 116, N3). A crash between the claim and the record leaves Test A as "
            f"\"{states[-1]}\" for Test B.",
            "**NaN replicates.** NaN bootstrap replicates are counted per statistic (`n_nan_replicates`), the p-value "
            "and the intervals are computed on the finite ones, and a result with more than 1% NaN replicates is "
            "marked approximate (Ruling 104(e)). Test B's decided H-E carries the approximate flag of the registered "
            "family (Ruling 104(b)).",
            "**Refused before any download and before the lock: the protocol is not ready to register (unfilled knowledge "
            "statement) (Ruling 122, C2-1).** `verify_registration` refuses a `PROTOCOL.md` that "
            "`es.protocol.ready_to_register` flags: an author marker (an `AUTHOR` placeholder of the template), an "
            "item-1 row of the knowledge statement without exactly one tick, or template guidance left in. Both sealed "
            "pulls and both sealed runs call it first, so none of them downloads or reads anything under such a "
            "protocol. The refusal says how many items are still open and names the first two, and says to fill in "
            "`docs/sealed/KNOWLEDGE_STATEMENT.md`, re-run `es.protocol.main()` and register that `PROTOCOL.md`; a draft "
            "that was registered with its statement unfilled cannot be used, because a registered protocol is immutable.",
            f"**Disk space and write access (Rulings 109, I8, and 116, N3).** The run is refused before its lock, with "
            f"nothing created, if less than {free} is free where its output folder lives or where `docs/sealed` lives, or "
            "if `docs/sealed` or the parent of the output folder is not writable. The log lines of the analyses after the "
            "coverage rule are best effort: a write error there, such as a full disk, loses that line and does not end "
            "the run.",
            *(["**The environment (Ruling 122, C2-3).** `ENV_PINS.json` binds the interpreter path and the library "
               "versions of the environment in which this protocol was generated, the Anaconda installation at "
               "`/opt/anaconda3`. If that installation is updated, moved or reinstalled before Test B has run, in January "
               "2027, so that the interpreter's path or the version of python or of a pinned library differs, Test B is "
               "refused before its lock, with nothing spent, until the environment is restored: do not update or move it "
               "until Test B has run."] if env_bound else []),
            "**Runbook.** Never open the sealed pull folders or the outputs in Finder: a `.DS_Store` file makes "
            "`verify_manifest` refuse.",
        ]),
    ])


@functools.lru_cache(maxsize=None)
def alert_conventions():
    """The three conventions of an alert, checked against ``es.policies`` when the protocol is rendered (Seat M, M19; read
    once): an alert fires iff score >= tau, tau is ``np.quantile`` of the scores at 1 - alpha with numpy's default (linear)
    interpolation, and ``top_count`` breaks ties by time order, earliest first (a stable sort of minus the score over a
    node's rows, which ``_per_node`` has put in time order). RuntimeError, naming what differs, if the code no longer does
    what the protocol says, so that a changed convention stops the protocol instead of being misdescribed. Returns the
    checks, by what they say."""
    import numpy as np

    from . import policies
    source = {n: inspect.getsource(getattr(policies, n)) for n in ("_with_tau", "_tau", "top_count", "_per_node")}
    checks = {
        "an alert fires iff score >= tau": ">= tau_arr" in source["_with_tau"] and 'dep["score"]' in source["_with_tau"],
        "tau is np.quantile of the scores at 1 - alpha": "np.quantile(" in source["_tau"] and "1 - alpha" in source["_tau"],
        "the quantile interpolates linearly": "method=" not in source["_tau"] and "interpolation=" not in source["_tau"]
        and inspect.signature(np.quantile).parameters["method"].default == "linear",
        "top_count ranks by a stable sort of minus the score": 'np.argsort(-dep["score"]' in source["top_count"]
        and 'kind="stable"' in source["top_count"],
        "the rows of a node are in time order": 'sort_values("timestamp_utc")' in source["_per_node"],
    }
    wrong = [what for what, holds in checks.items() if not holds]
    if wrong:
        raise RuntimeError("es.policies no longer does what the protocol says about an alert: " + "; ".join(wrong))
    return checks


def _constants_text():
    alert_conventions()
    xgb = xgb_factory(42)(1.0).get_params()
    keep = ("n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "reg_lambda", "eval_metric",
            "random_state")
    logit = logit_factory(42)(1.0).get_params()
    return "\n\n".join([
        "### Pins read from the code (spec section 6.4)",
        _bullets([
            f"Cutoff target alpha = {_g(ALPHA)} (`es.config.ALPHA`); label quantile: the node's "
            f"{_g(1 - _default(score_analysis, 'label_alpha'))} quantile of |DART| on the analysis's own training rows "
            f"(`label_alpha = {_g(_default(score_analysis, 'label_alpha'))}`).",
            "**The alert rule (Seat M, M19; read from `es.policies` when this protocol is rendered).** An alert fires if "
            "and only if score >= tau (`es.policies._with_tau`): a score equal to the cutoff alerts. tau is the "
            "(1 - alpha) quantile of the calibration scores by `np.quantile` with numpy's default linear interpolation "
            "(`es.policies._tau`; `es.cutoffs` repeats the rule for `pre_pull_cutoffs.csv`). `top_count` "
            "(`es.policies.top_count`), which places `calendar@count` and `VOL@count`, ranks a node's deployment rows by a "
            "stable sort of minus the score over rows in time order (`_per_node`), so a tie is broken by time, earliest "
            "row first. On the development rows the G1 review found the tie rule immaterial (Seat M, M19; Seat C1).",
            "XGBoost (`es.scores.xgb_factory`): " + ", ".join(f"{k} = {_g(xgb[k])}" for k in keep)
            + f"; scale_pos_weight = negatives / positives on the training rows; no fit with fewer than {MIN_POSITIVES} "
            "positive training labels.",
            "Logistic regression (`es.scores.logit_factory`): " + ", ".join(f"{k} = {_g(logit[k])}"
                                                                           for k in ("max_iter", "C", "class_weight")) + ".",
            f"Rolling cutoff P4: window {_g(_default(p4_rolling, 'window_days'))} days (primary); the sensitivity "
            "window of spec section 4.3 is run beside it. "
            f"Value function lambda in {{{', '.join(_g(x) for x in LAMS)}}}; E1 keys: {', '.join(E1_KEYS)}.",
            f"Day bootstrap: mean block {MEAN_BLOCK} days, B = {N_REP}, seed {SEED}.",
            "H-B pairs: " + ", ".join(f"{h} = {b} against DART3" for h, b in HB_PAIRS) + ".",
        ]),
    ])


def _not_computed_text():
    """What Part A registers that the sealed run does not compute (controller Ruling 119; Seat M, M14, corrected by M23):
    development-only quantities, quantities that come from the reporting code, and the development outputs that are published
    unbound. The cutoff levels of the audit-robustness family are read from ``es.secondary.AUDIT_VARIANTS``."""
    audit_levels = [level for name, _, _, _, level in _secondary.AUDIT_VARIANTS if name.startswith("alpha=")]
    if not audit_levels:
        raise RuntimeError("es.secondary.AUDIT_VARIANTS has no variant at another cutoff level of alpha, which the protocol "
                           "says the audit-robustness family computes")
    levels = _and_list(_percent(level) for level in audit_levels)
    return "\n\n".join([
        "### What Part A registers that the sealed run does not compute (Ruling 119)",
        "Part A copies the design of spec v3 for the whole project, and the order of authority above applies: what this "
        "Part D says the sealed run computes is what it computes. Spec v3 carries an amendment note at each of the places "
        "below (Ruling 119, Seat M, M14).",
        _bullets([
            "**Development-only; the sealed tests neither compute nor report them:** ES10 and maximum drawdown (spec section "
            "4.4); precision and recall at the deployed cutoff against both labels (section 5 item 3; `es.run_labels_pr`); "
            "the mixed model and the leave-one-operator-out checks (section 5 item 1); the failure counts at the levels "
            "of alpha other than 5%, and the failure bands [0.8, 1.25] and [0.67, 1.5] (section 5 item 6). The sealed run "
            "counts failures only for the band [0.5, 2] at alpha = 5%.",
            f"**Other levels of alpha that the sealed run does compute (Seat M, M23):** the audit-robustness family "
            f"re-runs the H-B1 and H-B2 contrasts at the cutoff levels {levels}, with the labels fixed at "
            f"{_percent(ALPHA)}, as descriptive diagnostics (the `es.secondary` docstring quoted under the secondary "
            "families above), and the economic analysis computes the economic curve over alpha of section 5 item 6, for "
            "P6 and P1.",
            "**From the reporting code, not registered computations:** figures, which are drawn after the run from the "
            "stored tables, and sign consistency across fees, positions and blocks, which is read from the stored "
            "estimates (section 5 item 2). The registered shells are the table shells of Part G.",
            "**Supplements, not bound by hash:** the development outputs, `outputs/es/analogs/` and `outputs/es/DIGEST.md`, "
            "are published with the code repository as supplementary files, instead of being uploaded to an OSF "
            "registration (section 6.5 step 3, Rulings 119 and 140). The code is identified by "
            "content, not by a git bundle (Ruling 28).",
        ]),
    ])


def _part_d(ctx, inputs):
    out = ["## Part D. The registered family and how each analysis is run",
           "### Registered family and decision rule (Rulings 29, 63, 84, 92, 94, 105, 110 and 118)",
           "Copied from `protocol_inputs.json` (`family` and `rule`), which the registration rule computed from the "
           "simulation and placebo rows of Part C. The text states each status as the file records it.",
           _family_text(inputs), _constants_text(), _he_text(inputs), _size_text(inputs), _estimands_text(),
           _levels_text(), _monitors_text(), _bh_text(), _fees_text(),
           _run_text(inputs["family"], bool(ctx["sha"].get("env"))), _not_computed_text(),
           _failure_text(bool(ctx["sha"].get("env")))]
    return "\n\n".join(out)


def _refinements(inputs):
    rule = inputs.get("rule") or {}
    factor = _g(rule.get("factor")) if rule.get("factor") is not None else "the rule's factor"
    binom = _g(rule.get("binomial_level")) if rule.get("binomial_level") is not None else "the rule's level"
    n_rep = _g(rule.get("n_rep")) if rule.get("n_rep") is not None else "the study's"
    third = f"{ALPHA:g}/3"
    dev, a2, b2, c2 = "dev", "2A", "2B", "2C"
    members = _holm_members(inputs.get("family") or {})
    items = [
        (12, dev, "Precision lift divides by the node's realized evaluation-label prevalence, not by alpha (they differ by "
                  "at most 1/n); spec section 5 item 3 is amended accordingly."),
        (15, dev, "The sealed entry point calls score_analysis directly and never reads the score cache."),
        (26, dev, "Calendar-propensity deciles are taken by value per node, so that tied scores share a stratum and the "
                  "strata do not depend on row order; every public measure raises on an empty frame or a missing or NaN "
                  "input instead of dropping rows silently."),
        (27, dev, "E1 to E4 carry day-bootstrap percentile intervals, and quarters pool by summing their additive pieces, "
                  "each quarter resampled independently over days with the alerts fixed on the real path; E1 also covers "
                  "J_lambda for every registered lambda; E2 is SNET."),
        (28, dev, "The code is identified by content (the SHA-256 of a deterministic archive and a per-file SHA-256 "
                  "manifest), not by a git commit; if the author commits before registering, the commit and a bundle "
                  "are recorded as well (spec section 6.5 step 3)."),
        (29, dev, "The registration family is computed from the size and power study by a registered rule, never decided "
                  f"by hand: the pooled size governs the confirmatory H-E, the single-window size only labels Test A's "
                  f"interim p-value, H-B power is read at {third}, and an interval is labelled approximate when the "
                  "simulated size exceeds nominal (refined by Rulings 63, 92 and 94)."),
        (30, dev, "Every cutoff of the confirmatory pipelines uses training and calibration rows only, so the cutoffs are "
                  "computed before the pull and registered in `pre_pull_cutoffs.csv`; the sealed run recomputes them and "
                  "reports any difference."),
        (31, dev, "The tie-safe deciles of Ruling 26 are pd.qcut of rank(method=\"min\"): tied scores share a stratum, the "
                  "result does not depend on row order, and distinct scores keep exactly the rank-based deciles."),
        (32, dev, "The 13,104 CAISO rows before 2023-07-13, for which OASIS no longer serves FMM prices, keep the flagged "
                  "panel real-time fallback (all are training-period rows); economics rows carry the share of "
                  "approximate settlement."),
        (33, dev, "The sealed pull is resumable but still once: raw responses are cached under `<root>/raw/`, finished "
                  "steps in `.state.json`, and a completed pull refuses any further call."),
        (37, dev, "The three Benjamini-Hochberg families (audit robustness, economic robustness, drift monitors) are "
                  "defined in `es.secondary`, each adjusted on its own at q = 0.05."),
        (39, dev, "A row whose DART or label is NaN is neither hit nor miss and leaves that metric's denominator; the "
                  "count is reported."),
        (40, dev, "The domain classifier is class-balanced (class_weight = \"balanced\"); Ruling 73 changed its blocking "
                  "from weeks to days."),
        (41, dev, "Estimands are also reported without CAISO (spec section 4.1), by a row filter after the per-node "
                  "models."),
        (43, dev, "Precision and recall against both labels are also computed for the label-control cells (spec section 5 "
                  "item 4); that control is a development analysis and is not part of the sealed run."),
        (44, dev, "The 1-Wasserstein monitor is the July pipeline's statistic (scaled by the calibration IQR, NaN with "
                  "fewer than 50 observations on a side), aggregated by the median over the block's features."),
        (55, dev, "H-E: the rotation test runs on the full grid (each window's shift set includes the observed state 0, so "
                  "p is a randomization p-value) with a studentized statistic T; pooled SELC stays the reported "
                  "estimand. A confirmatory H-B test is reported with its basic p-value and the basic interval "
                  "(2t - q.975, 2t - q.025); the estimands keep percentile intervals (Ruling 110 for the demoted H-B)."),
        (63, dev, f"The registration rule demotes a test only when its estimated size exceeds {factor} x nominal and a "
                  f"one-sided binomial test of the rejection count against nominal rejects at {binom}; the persistence "
                  "upper end is phi + 1.96 sqrt((1 - phi^2)/n) with the stationary SD fixed; H-B's size study includes "
                  "an equal-portability-under-drift null; the H-B power effect is calibrated to the target gap."),
        (58, a2, "The pull's cache, state, manifest and trimmed-table writes are atomic; `verify_manifest` compares the "
                 "exact file set; `COVERAGE.csv` lists every node of the node map, so an absent node shows 0 hours; the "
                 "registration check lowercases the SHA-256 and accepts a DOI written as a doi.org link."),
        (60, a2, "A per-node, per-day completeness check runs after each step of a pull (a day an ISO never published "
                 "must be named and is logged; Ruling 106 counts the rows of a day against its clock hours); reference "
                 "reads are cut before the first sealed row; the three July downloaders are imported at module level, so "
                 "they are in the code manifest."),
        (64, dev, "The simulation's seasonal means carry a weekday-class dimension and its strata mirror the calendar "
                  "propensity (with the heteroskedastic noise of Ruling 55(g)), so that the size study can see the bias "
                  "of decile strata; the size study decides H-E's status."),
        (66, b2, "(b) Every monitor reads deployment days 1 to 14 and is scored against |log10 R| and failure over day 15 "
                 "to the end; (c) if the protocol is recorded as an OSF registration, that registration is public or "
                 "embargoed at the author's choice (public recommended; Ruling 140)."),
        (68, dev, "The primary SELC strata are node x local hour x weekday class x month, for the H-E statistic and the "
                  "headline SELC; node x calendar-propensity decile is the sensitivity (E4)."),
        (69, dev, "H-E's reference set is the full cyclic group of day rotations {0, ..., D - 1} per window, not "
                  "{14, ..., D - 14}, which is not a group and over-rejects under day-to-day persistence."),
        (70, a2, "Each sealed pull folder is bound to the verified registration through `docs/sealed/PULLS.json` (root, "
                 "manifest SHA-256, protocol SHA-256, the registration id, which the pull's own log must not "
                 "contradict, a start time at or after `registered_utc`); no record handed in by a caller opens a sealed "
                 "window."),
        (73, dev, "The domain classifier's cross-validation is class-stratified 5-fold, blocked by local settlement day; "
                  "it amends the \"blocked by week\" of spec section 5 item 5."),
        (77, dev, "The runner of the size and power study is staged; `es.protocol` refuses a `protocol_inputs.json` whose "
                  "stage is not \"complete\" or whose family lacks H-E, H-B1 or H-B2."),
        (78, dev, "Every fee level comes from a source published on or before 2026-06-30 (the workspace rule); spec "
                  "invariant I4 is reworded accordingly."),
        (80, dev, "The mixed model includes the (1|operator) intercept; BH's m is the number of rows with a p-value "
                  "(`included`); a repeated (family, test) pair raises."),
        (81, dev, "E4's registered quantities are the paired contrasts dSELC against the calendar-only model and against "
                  "VOL at matched counts; the comparators' own SELC are context."),
        (82, dev, "Sealed rows with a non-finite day-ahead or real-time price are dropped and counted per market and node "
                  "before any analysis; every policy and comparator frame is restricted to P1's rows, and the dropped "
                  "rows are counted."),
        (83, dev, "Each input monitor has a descriptive placebo reference column (the same statistic on a split inside "
                  "the calibration window); the columns are not monitors and enter no BH family."),
        (84, dev, f"(a) The daily persistence of score and P&L in the size study is corrected by indirect inference; (b) "
                  f"the H-B size rows draw both blocks from the DART3, LEV6 and ROLL_DISP9 laws, add common-drift rows "
                  f"and use {n_rep} bootstrap replicates, and the binomial check is made at {binom} divided by the "
                  "number of rows read; (c) H-B's status is decided by the rule on the corrected study; (d) a rejection "
                  "reads \"larger operating-point error\", never \"less portable\"."),
        (89, dev, "BH and Holm decisions carry a 1e-12 relative tolerance at exact thresholds; `bh.csv` is written "
                  "atomically; a NaN in a family raises."),
        (90, dev, "Each fee component takes its latest value published on or before 2026-06-30, rounded to 0.05; MISO "
                  "omits Schedule 17 and 24 because no compliant source exists, and the uniform grid covers higher fees."),
        (92, dev, f"A fast path of the H-B bootstrap, bit-identical to `block_contrast_bootstrap`, serves the size study; "
                  f"\"approximate\" means a size significantly above nominal (a rate above the level and a binomial tail "
                  f"below {binom} divided by the number of rows); H-E's power is reported as a curve and never read by "
                  "the rule."),
        (93, dev, "With no drift the day-blocked domain AUC is somewhat above 0.5 for persistent inputs (about 0.55 for "
                  "AR(0.9) features in simulation, up to about 0.9 for 168-hour rolling means); the placebo columns give "
                  "each block's empirical no-drift reference, and only ranks across nodes are interpreted."),
        (94, dev, f"The placebo rule counts only the 1-year pairings (their P&L comes from the calibration year, outside "
                  f"the alert model's training window); the pooled check demotes H-E if and only if its p is at most "
                  f"{third}; each size row has its own seed; H-B1 and H-B2 read different laws."),
        (104, a2, "One-shot robustness of the sealed run: (a) the run claims its test id first, with an exclusive claim "
                  "file `RUN_<test>.claim` that is never removed (Ruling 109, I9; Ruling 116, N3), and `docs/sealed/RUNS.json` "
                  "records the run at the lock, whatever the output folder; (b) Test B's decided H-E carries the approximate flag of the registered family; (c) "
                  "each analysis after the coverage rule runs in its own guard; (d) a monitor node whose windows cannot carry "
                  "the monitors, or that a block's score frame does not cover, is skipped and counted; (e) NaN "
                  "bootstrap replicates are counted and the result is marked approximate "
                  "above 1%; (f) a coverage-rule failure is the registered NOT_RUN path."),
        (106, a2, "A pull's root that has files is resumed only from its `.state.json` and the facts of its first attempt, "
                  "and is never started over; the day check counts the rows of a day against its clock hours (23 or 25 "
                  "when the clocks change) for every series, and a NaN propagates through the FMM hour; a registration "
                  "record that names both a DOI and an OSF address must name the same registration, and the registration "
                  "id is logged and recorded in `docs/sealed/PULLS.json`."),
        (107, a2, "The code manifest and the archive cover the import closure of the sealed path (es.sealed_run, "
                  "es.sealed_pull, es.registration and es.protocol, and what they import, traced statically) and the "
                  "data files those modules open by a fixed path; tests, the paper and reporting modules and outputs "
                  "are not in it (Part F). Ruling 118 adds es.run_analogs and es.simulate as roots, so that the archive "
                  "can regenerate `protocol_inputs.json`."),
        (109, a2, "(1) `ENV_PINS.json` binds the interpreter path and the versions of python, numpy, pandas, pyarrow, "
                  "xgboost, scikit-learn, scipy and statsmodels, and the sealed run refuses before its lock if the file "
                  "is missing or the environment differs. (2) If Test A's output is not available (NOT_RUN; claimed and "
                  "left no manifest and no finished record; finished without stored P1 rows), Test B runs alone, with H-E "
                  "on its own window and not confirmatory (Rulings 112 and 116 say when Test B is refused instead); Test B never depends on Test A's demoted H-B results, and a failed Test A "
                  "estimand does not block it. (3) The test id must be A or B, and a configuration with a sealed arm must "
                  f"be the registered one, with {_sealed('N_REP')} bootstrap replicates, `es.registration`'s own paths and "
                  "the imported code as its code root. (4) The run's record in `docs/sealed/RUNS.json` gets "
                  "`finished_utc`, its status and its manifest hash, which Test B checks Test A's output against. (5) A "
                  "demoted H-B has no `p` and no replication, and its status keeps the rule's reasons in Test B too; H-E "
                  "falls back to rows built on their own if the economic analysis fails; E5 is labelled as the run's own "
                  "window; off-hour rows are dropped and counted. (6) Before the lock, at least "
                  f"{_gib(_sealed('MIN_FREE_BYTES'))} must be free and the development settlement tables must be present."),
        (110, a2, "Once H-B1 and H-B2 were demoted to estimands, the percentile interval is their primary reported "
                  "interval and the basic interval is a diagnostic beside it; neither is read inferentially (no "
                  "\"excludes zero\" claim), and no `p` is stored for an estimand (its bootstrap p-value is a diagnostic). "
                  "Ruling 118 corrects the mechanism: the shift of the bootstrap below t-hat comes mainly from "
                  "re-randomizing the calendar composition of the two windows, and the intervals may under-cover "
                  "substantially under a common drift."),
        (111, a2, "A pull may accept a chunk that never comes complete (a response that fails the validator on all "
                  f"{CHUNK_ATTEMPTS} attempts) only on a resume, by naming it in `allow_incomplete_chunks`; the accepted "
                  "chunk, its missing or non-finite intervals and the reason are logged and recorded in `.state.json` and "
                  "`docs/sealed/PULLS.json`, and the days it leaves short still need `allow_missing_days`."),
        (121, a2, "(F1) An answer that holds no rows (OASIS's no-data report, a CSV without rows, a zip without a CSV) "
                  "counts with the refused whole answers, and a chunk of such answers can be accepted as zero rows; a "
                  "network failure or an unreadable answer is no whole answer. (F2) A chunk is accepted as incomplete "
                  "only if every attempt of that call brought a whole answer that was refused, and the answer kept is the "
                  "one with the fewest problems. (F4) The day check's refusal for the NYISO integrated table names the "
                  "cached archive of each month concerned, which can be deleted to fetch the month again."),
        (112, a2, "Test B is refused, before anything is spent, unless Test A has been started (`docs/sealed/RUNS.json` "
                  "has a record of Test A, or Test A's claim file `RUN_A.claim` exists), so that a Test B run before Test A "
                  "by mistake cannot lose the pooled confirmatory test of H-E. Once Test A has been started, Test B runs "
                  "alone (H-E on its own window, not confirmatory) if Test A's output is not available, and is refused with "
                  "a repair instruction if Test A's folder is complete but its record lacks the finish fields (Ruling 116)."),
        (116, a2, "Final reconciliation of the sealed path. (N1) H-E is pooled from Test A's stored P1 rows alone; the "
                  "fallback for a failed economic analysis stores them; without P1 Test B runs alone, and without the other "
                  "stored frames its E1 to E4 cover its own quarter. (N2) A complete Test A folder whose record lacks the "
                  "finish fields refuses Test B, with the repair; Test B runs alone only when Test A left no manifest. (N3) "
                  "The claim is taken before the output folder is made, an output folder with no claim is refused (nothing "
                  "was read), and the write access of `docs/sealed` and of the output folder's parent is checked before the "
                  "lock. (N4) A sealed arm reads the registered development folders. (Mn2) The interpreter is compared by "
                  "its resolved path. (Mn3) Test B reads each stored file of Test A through Test A's manifest. (Mn4) The "
                  "pull's day check wants exactly as many rows on the hour as the day has hours, a day with more rows than "
                  "hours fails like a short one, `COVERAGE.csv` counts rows on the hour (its `expected_hours` stays at the "
                  "clock hours, so a fall-back day shows a share just under 1), and a CAISO hour with fewer than "
                  "twelve five-minute prices is NaN. (Mn6) The wordings of the Test-B-alone statuses now say that Test A's "
                  "output is not available, because Test A may have run."),
        (118, a2, f"After gate G1. (M1) The confirmatory family is {{{', '.join(members)}}} (m = {len(members)}), "
                  + (f"so H-E is decided at {ALPHA:g}; " if members == ["H-E"] else "") + "the protocol generates the "
                  "decision-level sentence from the family. (M2) The E5 and H-S intervals use one circular stationary draw "
                  "over the quarter's (month, day) calendar for both windows: `ci` is that aligned percentile interval, "
                  "`ci_independent` (the independent draws of H-B) a diagnostic beside it, all approximate; H-B1, H-B2 and "
                  "PC keep the independent scheme. (M5) The audit-robustness BH family is descriptive: its flags are "
                  "diagnostics and no count of its rejections is reported. (M6) E3 is the tail effect of the policy, not "
                  "evidence of skill; E4 gains `E4:ESred|c` and `E4:dESred|c` for both comparators. (M7) The H-B bootstrap "
                  "shift is mostly calendar re-randomization. (M8, M13) The size study's assumptions, a supplementary "
                  "check and the registration rule's operating characteristic are stated. (M9) The monitor Spearman "
                  "p-values are optimistic, and a result repeated across fee levels is one finding. (M12) H-E is gross "
                  "loss timing, not net value; E4 under the decile strata is a sensitivity; E1's level effect is a "
                  "decomposition. (C1-M1) `es.run_analogs` and `es.simulate` are roots of the registered code set."),
        (119, a2, "Part A registers some quantities that the sealed run does not compute, and says so in an amendment note "
                  "at each: ES10 and maximum drawdown, precision and recall at the deployed cutoff, the mixed model and the "
                  "leave-one-operator-out checks, the failure counts at the levels of alpha other than 5% and the failure "
                  "bands other than [0.5, 2] are development-only (the audit-robustness family and the economic curve do "
                  "use other levels of alpha, Seat M, M23); figures are drawn after the run from the stored tables and sign "
                  "consistency is read from the stored estimates, neither a registered computation; `outputs/es/analogs/` "
                  "and `outputs/es/DIGEST.md` are published with the code repository as unbound supplements (Ruling 140 "
                  "replaced the upload to OSF). The knowledge statement's item-1 boxes "
                  "and its template guidance now block registration, and its headings sit two levels deeper in Part H."),
        (122, a2, "After the sealed-path integration walk. (C2-1) `verify_registration` refuses a PROTOCOL.md that "
                  "`ready_to_register` flags, so both pulls and both runs refuse a protocol whose knowledge statement is "
                  "not filled in. (N2b) A sealed run writes the registered output folder and "
                  "Test B pools with Test A's registered output folder, and the repair for a complete Test A whose record "
                  "lacks the finish fields is looked for in the folder that the record names. (C2-2) In `REGISTRATION.template.md` the `embargo_until` "
                  "line is pre-filled as `none` and is informational: the sealed code does not use it. (C2-3) "
                  "`ENV_PINS.json` ties Test B to the environment of this protocol: if `/opt/anaconda3` is updated or moved "
                  "before Test B has run, Test B is refused before its lock until it is restored."),
        (105, b2, "(c) The audit-robustness BH family is computed as registered, but Ruling 118 (M5) supersedes this "
                  "ruling's reading of its rejections as approximate: the family is descriptive. Its p-values come from the "
                  "H-B bootstrap, whose simulated size exceeds nominal, so its BH flags are diagnostics; they are not read "
                  "as control of the false-discovery rate, and no count of its rejections is reported or interpreted. (e) "
                  "This protocol states the registered outcome in the terms of sections 6.1.2 and 6.1.3 of the paper."),
        (140, c2, "26 Oct 2026: the registration record may be an OSF registration, a public release of the code "
                  "repository or the author's local record; `osf_id` is renamed `registration_id`; the protocol had not "
                  "been registered or published before this amendment."),
        (141, c2, "26 Oct 2026: the author authorised the run of Test A on a local record, before any publication of the "
                  "protocol; the manuscript states that no third party held the protocol before the pull."),
    ]
    return sorted(items, key=lambda item: item[0])


def _part_e(inputs):
    legend = {"dev": "2026-10-22-development-phase/progress.md", "2A": "2026-10-22-plan-2a-sealed-machinery/progress.md",
              "2B": "2026-10-23-plan-2b-manuscript/progress.md", "2C": "2026-10-25-plan-2c-focus/progress.md"}
    lines = [f"- **Ruling {n}** ({tag}): {text}" for n, tag, text in _refinements(inputs)]
    return "\n\n".join([
        "## Part E. Refinements of the registered content",
        "These controller rulings changed or pinned an analysis after spec v3. Each line gives the ruling's number, "
        "which is also the evidence pointer into the ledger named in brackets ("
        + "; ".join(f"{k}: `dart_dev/.superpowers/sdd/{v}`" for k, v in legend.items()) + "). The sections of "
        "Part D state the resulting behaviour in full.",
        "\n".join(lines),
    ])


def _part_f(ctx):
    sha = ctx["sha"]
    code = ctx.get("code") or {}
    git = ctx.get("git")
    lines = [
        f"- `pre_pull_cutoffs.csv` SHA-256: `{sha['cutoffs']}`",
        f"- `protocol_inputs.json` SHA-256: `{sha['inputs']}`",
        f"- `CODE_MANIFEST.sha256` SHA-256: `{sha['manifest']}`",
        *([f"- `ENV_PINS.json` SHA-256: `{sha['env']}`"] if sha.get("env") else []),
        f"- `code_archive.tar.gz` SHA-256: `{sha['archive']}`",
    ]
    out = [
        "## Part F. Bound files and code identity",
        "The sealed run refuses to start unless the bound files below are exactly the ones whose SHA-256 this "
        "protocol prints (`es.registration.verify_bound`); the sealed pull needs the registration record. The protocol "
        "itself is identified by the SHA-256 in `PROTOCOL.sha256`, which the registration record must name.",
        "\n".join(lines),
        "`pre_pull_cutoffs.csv` holds the per-node cutoffs of Part C. `protocol_inputs.json` is a byte-for-byte copy of "
        "`outputs/es/analogs/protocol_inputs.json` (stage \"complete\"), with the family, rule, simulation and "
        "placebo rows of Part C. `CODE_MANIFEST.sha256` has one line `<sha256>  <path relative to dart_dev>` for each "
        "code file; `es.sealed_run` recomputes every line before it runs and refuses to run on departing code unless the "
        "departure is recorded. "
        + ("`ENV_PINS.json` holds the path of the interpreter (`sys.executable`) and the versions of "
           + ", ".join(ENV_PIN_LIBRARIES) + " that this protocol was generated with; `es.sealed_run` refuses before its "
           "lock if the running environment differs (Ruling 109(1)), and if the file is missing. " if sha.get("env") else "")
        + "`code_archive.tar.gz` is a deterministic archive of the same files (members sorted by "
        "path, modification time 0, owner 0, mode 644, GNU format; gzip with an empty file name and time 0).",
        "**Which files (Rulings 107(2) and 118).** The manifest and the archive cover the transitive import closure of the "
        "sealed path, not the whole repository: the modules " + ", ".join(f"`{m}`" for m in SEALED_ROOTS) + " and, as "
        "roots of their own, " + " and ".join(f"`{m}`" for m in PROVENANCE_ROOTS) + ", and every module they import, "
        "together with the data files these modules open by a fixed path ("
        + ", ".join(f"`{d}`" for d in OPENED_DATA_FILES) + "). The two provenance roots computed the registered family "
        "and the simulation rows of `protocol_inputs.json`: nothing on the sealed path imports them, but with them the "
        "archive can regenerate `protocol_inputs.json` from the development data (Seat C1, C1-M1). They have no reason "
        "to change after registration.",
        "**Method.** The closure is found by tracing the imports in each file's syntax tree (`es.protocol.import_closure`): "
        "every `import` and `from ... import` statement in the file, in functions and `try` blocks too, but not under "
        "`if __name__ == \"__main__\":`, which runs only when a file is run as a script; recursively, limited to the "
        "project's own modules (`es/`, `v2/`, `src/` and the top-level scripts they import); adding the `__init__.py` of "
        "each package above a module, because it runs when the module is imported. The data files are those that the "
        "modules of the closure open by a fixed file name, found by reading them; parquet caches, the run records and "
        "outputs are data, not code.",
        "**Not in the set.** The tests, the paper and reporting modules (`paper_*`, `run_digest`, `run_legacy`, "
        "`impl_table`), the other development runners and tools that neither the sealed path nor the two provenance "
        "roots import (for example `run_labels_pr` and the packaging code), and `outputs/`. Editing them is therefore "
        "not a departure from the registered code. None of the roots imports a paper, reporting or test module: "
        "`code_files` refuses to produce a set that includes one, and names who imports it.",
    ]
    if code.get("files"):
        out.append(f"**The closure.** The manifest lists {code['n_files']} files ("
                   + ", ".join(f"{n} in `{g}`" for g, n in code["groups"]) + "):\n\n"
                   + _bullets(f"`{f}`" for f in code["files"]))
    elif code.get("n_files"):
        out.append(f"The manifest lists {code['n_files']} files.")
    if git:
        out.append("**Git (spec section 6.5 step 3).** The code files were clean at the time of writing, at commit "
                   f"`{git['commit']}`. `{git.get('bundle', 'code.bundle')}` is a bundle of HEAD"
                   + (f", SHA-256 `{git['bundle_sha256']}`" if git.get("bundle_sha256") else "") + "."
                   + (" Files of the manifest that git does not track, and which the commit therefore lacks: "
                      + ", ".join(f"`{f}`" for f in git["not_in_commit"]) + "; they are identified by the manifest "
                      "and the archive." if git.get("not_in_commit") else ""))
    else:
        out.append("**No git commit identifies this code** (Ruling 28): commits need the author, so the code is "
                   "identified by content, through the manifest and the archive above, which are stronger than a commit "
                   "identifier. Spec section 6.5 step 3 asks for a commit and a bundle; they are recorded here if the "
                   "author commits before registering and the code files are clean.")
    out.append("**The record of the protocol (Rulings 140 and 141).** `docs/sealed/REGISTRATION.md`, written from "
               "`REGISTRATION.template.md`, names the SHA-256 of this protocol (exactly the file whose SHA-256 is in "
               "`PROTOCOL.sha256`), the UTC time of the record and exactly one registration of it, which is one of three: "
               "an OSF registration (`osf_doi` and/or `osf_registration`), a public release of the code repository "
               "(`public_record`, the address of a tagged GitHub release, whose publication time GitHub records) or the "
               "author's local record (`local_record`, `author-statement-` and the date of the author's authorisation). A "
               "record that names more than one kind is refused. The gate reads no network, so it cannot check any of "
               "them against the outside world: what protects the sealed windows is that nothing opens without a "
               "deliberate, well-formed record of the exact hash of this protocol, and what a document says about the "
               "record must be exactly what the record is.")
    out.append("For the run of Test A the record is the **local record**: `docs/sealed/REGISTRATION.md` holds the SHA-256 "
               "of this protocol and the UTC time at which the author's authorisation was recorded; no registry, "
               "repository or other third party holds the protocol before the pull; and this protocol and the code are "
               "published with the code repository after the run. The author authorised this on 26 October 2026. The OSF "
               "route stays accepted by the gate and is not used. For Test B the author intends a public release of the "
               "code repository before the pull; that is an intention, not a rule of the gate.")
    out.append("**Immutability.** Once `docs/sealed/REGISTRATION.md` exists, `PROTOCOL.md`, `pre_pull_cutoffs.csv` and "
               "`protocol_inputs.json` are never rewritten: `es.protocol.main()` refuses. `pre_pull_cutoffs.csv`, "
               "`protocol_inputs.json`, `CODE_MANIFEST.sha256`, "
               + ("`ENV_PINS.json` " if sha.get("env") else "") + "and `code_archive.tar.gz` are bound to the protocol "
               "by the SHA-256 lines above. The development outputs behind the inputs (`outputs/es/analogs/`, "
               "`outputs/es/DIGEST.md`) stay in the repository and are published with the code repository as "
               "supplementary files that are not part of the bound files (spec section 6.5 step 3, Rulings 119 and 140).")
    return "\n\n".join(out)


def _shell(rows):
    return _table(["Quantity", "Estimate", "95% CI", "Status"], [[r, DASH, DASH, DASH] for r in rows])


def _part_g():
    e1 = [f"{part}: {key}" for part in ("timing", "level", "total") for key in E1_KEYS]
    blocks = [f"{b} ({PAPER_BLOCK_NAMES.get(b, b)}): {m}" for b in BLOCKS
              for m in ("median |log10 R|", "nodes outside [0.5, 2]")]
    shells = [
        ("H-E: timing within calendar cells (pooled SELC; rotation test)",
         ["Test A, single window (interim)", "Pooled A + B (decision)"]),
        ("H-B1 and H-B2: median over nodes of the paired |log10 R| difference (95% CI: percentile, primary; basic "
         "interval beside it, diagnostic)",
         ["H-B1: calendar + level against calendar + plain DART lags",
          "H-B2: calendar + rolling dispersion against calendar + plain DART lags"]),
        ("PC: positive control (median over nodes of |log10 R|(P2) - |log10 R|(P1))",
         ["P2 season-matched", "P2 all-season"]),
        ("H-S: failure counts of recalibrated plain DART lags (95% CI: calendar-aligned percentile, primary; "
         "independent-draw interval beside it, diagnostic; approximate)", ["XGBoost", "Logistic regression"]),
        (f"E1: decomposition of V(P1) - V(P4) into timing and level ({', '.join(E1_KEYS)})", e1),
        ("E2: SNET ($ per MW-day)", ["SNET"]),
        ("E3: ESred (the tail effect of the policy, not evidence of skill; approximate)", ["ESred"]),
        ("E4: SELC and ESred against the comparators at matched counts",
         [f"dSELC: P1 minus {c}" for c in E4_COMPARATORS]
         + [f"SELC of {c} (context)" for c in E4_COMPARATORS]
         + [f"dESred: P1 minus {c} (approximate)" for c in E4_COMPARATORS]
         + [f"ESred of {c} (context, approximate)" for c in E4_COMPARATORS]
         + ["SELC of P1 under the decile strata (sensitivity only)"]),
        ("E5: the full block table (95% CI: calendar-aligned percentile, primary; independent-draw interval beside it, "
         "diagnostic; approximate)", blocks),
    ]
    out = ["## Part G. Table shells",
           "The registered result tables. Every cell reads \"\u2014\" until the run fills it. The columns are the "
           "estimate, the 95% interval (a percentile interval; for the demoted H-B estimands the percentile interval is "
           "the primary one and the basic interval is shown beside it as a diagnostic, Ruling 110; for E5 and H-S it is "
           "the calendar-aligned percentile interval, with the independent-draw interval beside it as a diagnostic, and "
           "every such interval is approximate, Ruling 118) and the status "
           "(for example interim, approximate, estimand, or the registered wording). The rows that involve ESred (E1's "
           "ESred part and its J rows with lambda above 0, E3, and E4's ESred and dESred) are tail measures and are "
           "labelled approximate. Each table is reported for Test A "
           "alone (interim). E1 to E4 are given for Test B alone and pooled. H-E is pooled only, and is Test B alone only "
           "when Test A's output is not available; the other tables are reported per test. If Test A's output is not "
           "available (Ruling 112), the pooled rows are Test B's window alone, and H-E there is labelled not "
           "confirmatory; if only Test A's stored P1 rows are available, H-E is pooled and E1 to E4 are Test B's own "
           "quarter (Ruling 116). The economic-robustness family's \"P1 premium fees=calibrated\" row is H-E's own "
           "single-window statistic, not a separate finding: in Test A it is the interim p, and in Test B the same "
           "statistic on Test B's window alone (Ruling 119, Seat M, M17)."]
    for title, rows in shells:
        out.append(f"### {title}")
        out.append(_shell(rows))
    return "\n\n".join(out)


def demote_headings(markdown, levels=2):
    """``markdown`` with every heading ``levels`` deeper (at most six hashes), outside fenced code: a statement that
    brings its own "#" and "##" headings then sits under the Part that embeds it (Seat M, M20)."""
    out, fenced = [], False
    for line in markdown.splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
        found = None if fenced else re.match(r"^(#{1,6})(\s.*)$", line)
        out.append("#" * min(len(found.group(1)) + levels, 6) + found.group(2) if found else line)
    return "\n".join(out) + ("\n" if markdown.endswith("\n") else "")


def _part_h(ctx):
    knowledge = ctx.get("knowledge")
    source = ctx.get("knowledge_source")
    if isinstance(knowledge, str):
        intro = ("The author's statement of what the author knows about market conditions after 30 June 2026, "
                 "together with the facts stated by the analysis team"
                 + (f" (`{source}`)" if source else "") + ", follows word for word, its headings two levels deeper so "
                 "that they sit under this Part (Ruling 119)."
                 + (" This is the unfilled template: its author markers must be replaced before registration."
                    if source and source.endswith("TEMPLATE.md") else ""))
        body = KNOWLEDGE_BEGIN + "\n" + demote_headings(knowledge).rstrip("\n") + "\n" + KNOWLEDGE_END
    else:
        intro = "The knowledge statement has not been written yet."
        body = MISSING_KNOWLEDGE
    return "\n\n".join(["## Part H. Knowledge statement", intro, body])


def _part_i(ctx):
    versions = ctx.get("versions") or {}
    return "\n\n".join([
        "## Part I. Library versions",
        "Captured when this protocol was generated, from the interpreter that runs the code. Spec section 6.4 pins "
        "python, pandas, xgboost, scikit-learn and pyarrow, and records numpy at freeze; scipy enters the fitted "
        "simulation and statsmodels the mixed model.",
        "\n".join(f"- {name} {version}" for name, version in versions.items()),
    ])


def render_protocol(ctx):
    """The protocol as Markdown, from ``ctx`` (see the module docstring). Reads no file. Raises ValueError if the
    generated text, outside the knowledge statement, holds an ``[AUTHOR: ...]`` marker (only the author's statement may
    carry one)."""
    inputs = ctx["inputs"]
    spec = ctx["spec"]
    fixed = [_part_title(ctx), _part_a(spec), _part_b(), _part_c(ctx, inputs), _part_d(ctx, inputs),
             _part_e(inputs), _part_f(ctx), _part_g()]
    stray = author_markers("\n\n".join(fixed))
    if stray:
        raise ValueError(f"the generated text holds an author marker outside the knowledge statement: {stray}")
    text = "\n\n".join(fixed + [_part_h(ctx), _part_i(ctx)])
    return text.rstrip("\n") + "\n"


# ---------------------------------------------------------------------------------------------------------------
# REGISTRATION.template.md
# ---------------------------------------------------------------------------------------------------------------
def registration_template():
    """The template of docs/sealed/REGISTRATION.md, for the three records of Ruling 140: an OSF registration (``osf_doi``
    and/or ``osf_registration``), a public release of the code repository (``public_record``) or the author's local record
    (``local_record``). Its gate lines (``protocol_sha256``, the two OSF lines, ``public_record``, ``local_record`` and
    ``registered_utc``) are ``<paste ...>`` placeholders with their instructions, which
    ``es.registration.parse_registration`` refuses (a line naming a gate key must hold one token with no angle bracket), so
    a gate line left unfilled blocks the sealed windows, and so does a record that keeps the lines of more than one kind.
    The author replaces the whole value of each gate line they keep and deletes the whole line of every identifier they do
    not have and of the two kinds they do not use. The last line, ``embargo_until``, is informational (Ruling 122, C2-2): it
    is not a gate key, nothing in ``es`` uses it, and it is pre-filled as ``none``. A multi-word placeholder there would
    match no ``key: value`` pair, so it would block nothing either."""
    return "\n".join([
        "# Registration record: template",
        "",
        "Copy this file to docs/sealed/REGISTRATION.md when the protocol is recorded (Ruling 140). The record is one of "
        "three kinds, and it may name exactly one: an OSF registration (the lines `osf_doi` and/or `osf_registration`), a "
        "public release of the code repository (the line `public_record`) or the author's local record (the line "
        "`local_record`). Keep `protocol_sha256` and `registered_utc`, keep the lines of the one kind you use, and delete "
        "the whole line of every identifier you do not have and of the two kinds you do not use (a line left as a "
        "placeholder is refused, and so is a record that names more than one kind). Replace the whole value of each "
        "`<paste ...>` line you keep, the whole placeholder included, with the real value, with no spaces in it. For an "
        "OSF registration, if you keep both identifiers, they must name the same OSF registration: the id inside the DOI "
        "is the id of the OSF address (a record that names two registrations is refused). The last line, "
        "`embargo_until: none`, is already filled in and is informational (the sealed code does not use it): it matters "
        "only for an OSF registration, and you change it only if you embargo that registration. Then the sealed windows "
        "can be pulled.",
        "",
        "> Guidance, which the gate ignores: for an OSF registration, fill this in only after OSF shows the registration "
        "as registered (approved), not as pending approval, and copy registered_utc from the approved registration, the "
        "timestamp that OSF shows for it (Ruling 120).",
        "> Guidance, which the gate ignores: for a public release, publish the tagged release of the code repository first, "
        "then copy `url` and `publishedAt` from `gh release view <tag> --json url,publishedAt` into `public_record` and "
        "`registered_utc`: registered_utc is the publication time that GitHub records for the release.",
        "> Guidance, which the gate ignores: for the author's local record, `local_record` is `author-statement-` and the "
        "date of the author's authorisation (YYYY-MM-DD), and registered_utc is the UTC time at which that authorisation "
        "was recorded; no registry, repository or other third party holds the protocol before the pull, and the protocol "
        "and the code are published with the code repository after the run (Rulings 140 and 141).",
        "",
        "protocol_sha256: <paste the 64 hexadecimal characters of docs/sealed/PROTOCOL.sha256>",
        "osf_doi: <paste the DOI of the OSF registration, 10.17605/OSF.IO/ followed by its OSF id; delete this line if the record is not an OSF registration or OSF minted no DOI; if you keep both identifiers they must name the same registration>",
        "osf_registration: <paste the OSF address of the registration, https://osf.io/ followed by its OSF id and a slash; delete this line if the record is not an OSF registration or you have only the DOI. For an OSF registration at least one of osf_doi and osf_registration is required, and if you keep both they must name the same registration (the same OSF id)>",
        "public_record: <paste the url of the tagged release, https://github.com/ followed by the owner, the repository, /releases/tag/ and the tag, with nothing after the tag (no slash, query or fragment); delete this line unless the record is a public release>",
        "local_record: <paste author-statement- followed by the date of the author's authorisation, YYYY-MM-DD; delete this line unless the record is the author's local record>",
        "registered_utc: <paste the time of the record in UTC, ISO 8601 with a trailing Z and no spaces: OSF's registration timestamp, the publishedAt of the release, or the time at which the author's authorisation was recorded>",
        "embargo_until: none",
        "",
    ])


# ---------------------------------------------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------------------------------------------
class _Out:
    """The files main() writes, as paths. By default es.registration's own (the ones the gate and the sealed run read,
    taken when main() is called); with a ``sealed_dir`` every file sits in that folder under the same name."""

    def __init__(self, sealed_dir=None):
        reg = _registration
        if sealed_dir is None:
            self.dir = Path(reg.SEALED_DIR)
            self.protocol, self.registration = Path(reg.PROTOCOL_PATH), Path(reg.REGISTRATION_PATH)
            self.cutoffs, self.inputs = Path(reg.CUTOFFS_PATH), Path(reg.INPUTS_PATH)
        else:
            self.dir = Path(sealed_dir)
            self.protocol, self.registration = self.dir / reg.PROTOCOL_PATH.name, self.dir / reg.REGISTRATION_PATH.name
            self.cutoffs, self.inputs = self.dir / reg.CUTOFFS_PATH.name, self.dir / reg.INPUTS_PATH.name
        self.protocol_sha = self.dir / "PROTOCOL.sha256"
        self.manifest = self.dir / "CODE_MANIFEST.sha256"
        self.env_pins = self.dir / "ENV_PINS.json"
        self.archive = self.dir / "code_archive.tar.gz"
        self.bundle = self.dir / "code.bundle"
        self.template = self.dir / "REGISTRATION.template.md"
        self.knowledge = self.dir / "KNOWLEDGE_STATEMENT.md"
        self.knowledge_template = self.dir / "KNOWLEDGE_STATEMENT_TEMPLATE.md"


def load_inputs(path):
    """(bytes, parsed) of protocol_inputs.json. RuntimeError unless the file parses, is a JSON object, has stage
    "complete" (Ruling 77: Task 11's staged runner writes "analogs" before the simulation has run), holds H-E, H-B1 and
    H-B2 in its family, and has the keys the protocol prints."""
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"{path} does not exist; run es.run_analogs (stage sim) first")
    raw = path.read_bytes()
    try:
        doc = json.loads(raw)
    except ValueError as exc:
        raise RuntimeError(f"{path} is not valid JSON: {exc}") from None
    if not isinstance(doc, dict):
        raise RuntimeError(f"{path} must hold a JSON object, not {type(doc).__name__}")
    if doc.get("stage") != "complete":
        raise RuntimeError(f"{path} is at stage {doc.get('stage')!r}, not 'complete': run `python -m es.run_analogs "
                           "--stage sim` before the protocol is generated (Ruling 77)")
    family = doc.get("family")
    absent = [h for h in FAMILY_ORDER if not isinstance(family, dict) or not isinstance(family.get(h), dict)]
    if absent:
        raise RuntimeError(f"{path}: the registered family lacks {absent}")
    missing = [k for k in REQUIRED_INPUT_KEYS if k not in doc]
    if missing:
        raise RuntimeError(f"{path} lacks {missing}, which the protocol prints")
    return raw, doc


def _write(path, data):
    """Bytes (or text, as UTF-8 with LF) into ``path`` through a temporary file in the same folder and os.replace."""
    path = Path(path)
    if isinstance(data, str):
        data = data.encode("utf-8")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _cutoff_csv(table):
    return table.to_csv(index=False, lineterminator="\n")


def cutoff_summary(table):
    """Per analysis and pipeline: nodes, the two orderings of the cutoffs and the nodes without a P1 cutoff."""
    rows = []
    for (analysis, block), g in table.groupby(["analysis", "block"], sort=False):
        rows.append({"analysis": analysis, "block": block, "nodes": int(len(g)),
                     "p2_season_above_p1": int(g["p2_season_above_p1"].sum()),
                     "p2_all_above_p1": int(g["p2_all_above_p1"].sum()),
                     "nodes_without_tau_p1": int(g["tau_p1"].isna().sum())})
    return rows


def dev_cutoff_table():
    """The cutoff table of the sealed analyses, fitted on development data (heavy: run it alone, at low priority)."""
    from .data import build_panel, load_dev_raw
    panel = build_panel(load_dev_raw())
    return cutoff_table(panel, [SEALED[k] for k in ("SA", "SAF", "SB", "SBF")])


def _changed(before, after):
    """The paths whose line differs between two manifests (text), or that only one of them lists."""
    one = {ln.partition("  ")[2]: ln.partition("  ")[0] for ln in before.splitlines()}
    two = {ln.partition("  ")[2]: ln.partition("  ")[0] for ln in after.splitlines()}
    return sorted(k for k in one.keys() | two.keys() if one.get(k) != two.get(k))


def _require_unchanged(dev_root, files, manifest_before, files_given):
    """RuntimeError if the code changed while the package was being generated: the fit of the cutoff table takes minutes,
    other work on the tree goes on, and a manifest or archive of files that differ from the ones the fit ran on would
    register code that never produced the cutoffs. The manifest is taken again, and unless the caller named the files, the
    file set too."""
    changed = _changed(manifest_before, manifest_text(dev_root, files))
    if not files_given:
        now = [Path(p).relative_to(dev_root).as_posix() for p in code_files(dev_root)]
        changed += [r for r in sorted(set(now) ^ {Path(p).relative_to(dev_root).as_posix() for p in files})
                    if r not in changed]
    if changed:
        raise RuntimeError("the code changed while the protocol was being generated, so nothing was written: "
                           + ", ".join(changed[:10]) + (", ..." if len(changed) > 10 else "")
                           + "; run es.protocol.main() again once the tree is quiet")


def main(sealed_dir=None, dev_root=None, inputs_source=None, cutoffs=None, files=None, spec_path=None,
         forensics_path=None, versions=None, date=None):
    """Write the protocol package. Every argument defaults to the real tree; the tests pass temporary ones.

    ``sealed_dir``: where the files go (default: es.registration's paths). ``dev_root``: what the code manifest is
    relative to. ``inputs_source``: protocol_inputs.json (default outputs/es/analogs/protocol_inputs.json). ``cutoffs``: a
    callable returning the cutoff table (default ``dev_cutoff_table``). ``files``: the code files (default
    ``code_files(dev_root)``). Returns {"protocol_sha256", "sha", "markers", "git", "files"}.

    In order: refuse if REGISTRATION.md exists (registered files are immutable); validate the inputs and the pins, and
    render the protocol once with placeholder hashes, so that a problem with the spec or the inputs stops the run before
    the heavy step; fit the cutoff table; build the archive and look at git; refuse if the code changed meanwhile; then
    write the bound files (cutoffs, inputs copy, manifest, archive, the bundle if git is clean), read the knowledge
    statement (KNOWLEDGE_STATEMENT.md, else the template), render and write PROTOCOL.md and PROTOCOL.sha256, write
    REGISTRATION.template.md and print what still blocks registration: the author markers, and the knowledge statement's
    item-1 rows without exactly one ticked box and its left-over template guidance (``ready_to_register``, Ruling 119)."""
    out = _Out(sealed_dir)
    dev_root = Path(DEV_ROOT if dev_root is None else dev_root)
    if out.registration.exists():
        raise RuntimeError(f"{out.registration} exists: the protocol is registered, and its files are immutable "
                           "(spec section 6.5); es.protocol.main() will not rewrite them")
    spec_path = Path(REVISION_ROOT / SPEC_REL if spec_path is None else spec_path)
    forensics_path = Path(REVISION_ROOT / FORENSICS_REL if forensics_path is None else forensics_path)
    inputs_raw, inputs = load_inputs(OUT_DIR / "analogs" / "protocol_inputs.json" if inputs_source is None
                                     else inputs_source)
    spec = spec_path.read_text(encoding="utf-8")
    versions = capture_versions() if versions is None else versions
    mismatched = check_pins(spec, versions)
    if mismatched:
        raise RuntimeError("the interpreter does not meet the pins of spec section 6.4: " + "; ".join(mismatched))
    files_given = files is not None
    files = sorted(Path(f) for f in files) if files_given else code_files(dev_root)
    manifest_before = manifest_text(dev_root, files)                   # the code the fit below runs on

    if out.knowledge.is_file():
        knowledge_path = out.knowledge
    elif out.knowledge_template.is_file():
        knowledge_path = out.knowledge_template
    else:
        knowledge_path = None
    ctx = {"inputs": inputs, "versions": versions, "git": None,
           "knowledge": knowledge_path.read_text(encoding="utf-8") if knowledge_path else None,
           "knowledge_source": f"docs/sealed/{knowledge_path.name}" if knowledge_path else None,
           "spec": spec, "forensics": forensics_path.read_text(encoding="utf-8") if forensics_path.is_file() else "",
           "date": date, "code": code_summary(dev_root, files)}
    render_protocol({**ctx, "sha": dict.fromkeys(("cutoffs", "inputs", "manifest", "env", "archive"), "0" * 64)})   # dry run

    out.dir.mkdir(parents=True, exist_ok=True)
    archive_tmp = out.archive.with_name(out.archive.name + ".tmp")
    bundle_tmp = out.bundle.with_name(out.bundle.name + ".tmp")
    try:
        table = (dev_cutoff_table if cutoffs is None else cutoffs)()
        cutoff_bytes = _cutoff_csv(table).encode("utf-8")
        archive_sha = build_archive(dev_root, files, archive_tmp)
        git = git_state(dev_root, files, bundle_tmp)
        _require_unchanged(dev_root, files, manifest_before, files_given)
        if git:
            os.replace(bundle_tmp, out.bundle)
            git["bundle"] = out.bundle.name
        else:
            out.bundle.unlink(missing_ok=True)                  # a bundle of an earlier, clean run no longer matches

        _write(out.cutoffs, cutoff_bytes)
        _write(out.inputs, inputs_raw)
        _write(out.manifest, manifest_before)
        _write(out.env_pins, env_pins_text(versions))
        os.replace(archive_tmp, out.archive)
    finally:
        archive_tmp.unlink(missing_ok=True)
        bundle_tmp.unlink(missing_ok=True)
    sha = {"cutoffs": sha256_file(out.cutoffs), "inputs": sha256_file(out.inputs),
           "manifest": sha256_file(out.manifest), "env": sha256_file(out.env_pins), "archive": archive_sha}
    if sha["archive"] != sha256_file(out.archive):
        raise RuntimeError("the archive changed while it was being moved into place")

    text = render_protocol({**ctx, "sha": sha, "git": git, "cutoff_summary": cutoff_summary(table)})
    _write(out.protocol, text)
    protocol_sha = sha256_file(out.protocol)
    _write(out.protocol_sha, f"{protocol_sha}  {out.protocol.name}\n")
    _write(out.template, registration_template())

    markers = ready_to_register(text)
    n_author = len(author_markers(text))
    print(f"wrote {out.dir}:")
    for label, path in (("pre_pull_cutoffs.csv", out.cutoffs), ("protocol_inputs.json", out.inputs),
                        ("CODE_MANIFEST.sha256", out.manifest), ("ENV_PINS.json", out.env_pins),
                        ("code_archive.tar.gz", out.archive),
                        ("PROTOCOL.md", out.protocol)):
        print(f"  {label}  {sha256_file(path)}")
    print(f"  git: {git['commit'] if git else 'none (the code is identified by content)'}")
    print(f"ready_to_register: {n_author} author marker(s) remain"
          + (f" and {len(markers) - n_author} other problem(s) in the knowledge statement" if len(markers) > n_author else "")
          + (f": {markers}" if markers else ""))
    return {"protocol_sha256": protocol_sha, "sha": sha, "markers": markers, "git": git, "files": len(files)}


if __name__ == "__main__":
    main()
