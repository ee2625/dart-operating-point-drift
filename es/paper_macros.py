"""Resolve the registry against the saved outputs; write paper/generated/macros.tex, provenance.csv
and registry.csv. A macro whose source *file* is absent becomes a visible pending sentinel for its
stage. A present file with no matching row (or several) raises LookupError: that is a registry bug, not a stage.
Every failure to read or format a value is a LookupError that names the macro: a missing or ambiguous row, a bad
query, an empty, ragged, non-UTF-8 or unreadable file, a JSON file that is not an object, a value the formatter refuses.

A number that no single file or constant holds is *derived* (source "derived:<name>", DERIVED below): computed when the
macro is resolved from named output files and from the code's own definitions, with the digests of both in provenance
and a detail that says what was counted. Its output files missing is pending, like any other.

A derived number may also read the archived output of a review under docs/reviews (source "docsreviews:<path under docs/reviews>"; Ruling
118, M8: the supplementary size check of gate G1, docs/reviews/g1_probes): Sources.docs_reviews, docs/reviews next to docs/sealed unless a
test gives another folder. A missing file is pending like any source; a file that is not what the reader expects is a LookupError.

The author's identifiers (title, e-mail, department, ORCID, the address of the code repository, the SSRN id, the statement on the AI tools)
are read from paper/author_inputs.json (source "author:<key>"): an empty value is pending, a key that the file lacks is a LookupError.
Ruling 133 (25 October 2026): the protocol of the sealed tests is not registered with a third party and the code is not archived at a DOI,
so no macro names an OSF or Zenodo identifier and no macro reads docs/sealed/REGISTRATION.md. The time of the protocol record that a sealed
run verified at its start is the run's own header (results.json, registered_utc; macro registeredUtc<Test>).

The simulation stage (Ruling 101): check_simulation_rule compares what the registry types about the rule (the laws each H-B test
reads, the rows each size selection reads, the levels, the coupling grid) with what the stage recorded, at every build.

Sealed tests (plan 2A): when results.json says coverage.ran is false the test did not run, and every sealed macro of
that test except the header that run_sealed still writes (NOT_RUN_KEYS) resolves to the stage sentinel with the detail
"test did not run"; \\TestAdidnotrun (or B) is set. Holm decisions may be null: they print "n/a".

The pull record of Test A (Task 9, Ruling 121): \\pullRecordTestA is the one macro of a sealed test that a person writes, because sealed_run
does not read the pull's own log (PULL_LOG, the entry of PULLS.json). The controller fills paper/author_inputs.json, key "pull_record_test_a",
with one plain sentence (VALIDATORS says what is refused); while it is empty the macro is pending in stage sealedA and prints \\PENDINGA.

Run: DART_PROJECT_ROOT=... /opt/anaconda3/bin/python3 -m es.paper_macros [--strict] [--stages const,dev,sealedA,author]"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import importlib
import math
import re
import sys
from functools import lru_cache
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import registration
from .config import OUT_DIR
from .paper_paths import DOCS_SEALED, PAPER_DIR
from .paper_pins import SEEDS
from .paper_registry import (CODE_CONSTANTS, CODE_READERS, CODE_SOURCES, CURVE_TEST, ENTRIES, HB_LAWS, PROTOCOL, RHO_WORD, SIM,
                             SIM_LEVELS, SIZE_TESTS, SPEC_FIXED, CodeUnavailable, Entry, date_window, entries_by_name)
from .paper_tex import FORMATS, PAIR_FORMATS, ROW_FORMATS, is_missing, macro_name_ok

PENDING = {"const": "\\PENDINGCONST", "dev": "\\PENDINGDEV", "sealedA": "\\PENDINGA",
           "sealedB": "\\PENDINGB", "author": "\\PENDINGAUTHOR"}
PENDING_TEXT = {"const": "[pending: constant]", "dev": "[pending: development results]",
                "sealedA": "[pending: Test A]", "sealedB": "[pending: Test B]",
                "author": "[pending: author input]"}
DEFAULT_STRICT_STAGES = ("const", "dev", "sealedA", "author")
#: `stage` values of a JSON output (analogs/protocol_inputs.json) that is written in steps: while the file says it is at one of
#: these, the keys that only a later stage writes (rho_star, the registered family) are pending, not a registry error.
EARLY_JSON_STAGES = ("analogs",)
ES_DIR = Path(__file__).resolve().parent
#: The top-level keys of results.json that macros read and that run_sealed writes also when the coverage rule fails: a test
#: that did not run has these (and the registration, the family and the bound files, which no macro reads) and nothing else
#: (no tau_check, H-E, H-B1, ..., and no tables). status is then "not run (coverage rule)" and errors is empty.
#: Task 9 adds departures and registered_utc: run_sealed writes both in the header (results = {...} before the lock), so the execution
#: record of a test that did not run can state them (macros departures<Test>, registeredUtc<Test>).
NOT_RUN_KEYS = ("coverage", "rows_read", "rows_kept", "started_utc", "protocol_sha256", "eta", "status", "errors",
                "departures", "registered_utc")
ESTIMAND_TEXT = "n/a (estimand)"
_SEALED = re.compile(r"^sealed/([AB])/")
_RUN_SEED = re.compile(r"__s(\d+)__")             # the seed in the name of a legacy run file: <...>__xgb__s42__P1__a0.05.csv


class SourceError(LookupError):
    """A source file that cannot be read as what it should be (empty, malformed, not UTF-8, unreadable, not a JSON
    object). It names the file; resolve() re-raises it as a LookupError that also names the macro."""


def _one_line(exc: BaseException) -> str:
    return " ".join(str(exc).split())


class Sources:
    def __init__(self, out_dir: Path = OUT_DIR, paper_dir: Path = PAPER_DIR, docs_sealed: Path = DOCS_SEALED,
                 docs_reviews: Path | None = None):
        self.out_dir, self.paper_dir, self.docs_sealed = Path(out_dir), Path(paper_dir), Path(docs_sealed)
        self.docs_reviews = Path(docs_reviews) if docs_reviews is not None else self.docs_sealed.parent / "reviews"
        self._frames: dict[str, pd.DataFrame] = {}
        self._json: dict[str, dict] = {}

    def path(self, source: str) -> Path | None:
        if source.startswith(("code:", "derived:")):
            return None
        if source.startswith("author:"):
            return self.paper_dir / "author_inputs.json"
        if source.startswith("docsfile:"):
            return self.docs_sealed / source[9:]
        if source.startswith("docsreviews:"):
            return self.docs_reviews / source[12:]
        return self.out_dir / source

    def shown(self, source: str) -> str:
        """The source as the provenance file names it: relative to the project, never an absolute path. generated/provenance.csv is a
        member of the upload set, and an absolute path would carry the author's user name and folder names into it."""
        if source.startswith("author:"):
            return "paper/author_inputs.json"
        if source.startswith("docsfile:"):
            return "docs/sealed/" + source[9:]
        if source.startswith("docsreviews:"):
            return "docs/reviews/" + source[12:]
        return "outputs/es/" + source

    def exists(self, source: str) -> bool:
        p = self.path(source)
        return p is None or p.exists()

    def frame(self, source: str) -> pd.DataFrame:
        if source not in self._frames:
            path = self.path(source)
            try:
                self._frames[source] = pd.read_csv(path)
            except pd.errors.EmptyDataError as exc:
                raise SourceError(f"{source}: {path} is empty") from exc
            except (pd.errors.ParserError, UnicodeDecodeError, OSError, ValueError) as exc:    # ragged, not UTF-8, unreadable
                raise SourceError(f"{source}: {path} cannot be read as a CSV ({type(exc).__name__}: {_one_line(exc)})") from exc
        return self._frames[source]

    def json(self, source: str) -> dict:
        if source not in self._json:
            path = self.path(source)
            try:
                data = json.loads(path.read_text())
            except json.JSONDecodeError as exc:
                raise SourceError(f"{source}: {path} is not valid JSON: {exc}") from exc
            except (UnicodeDecodeError, OSError) as exc:
                raise SourceError(f"{source}: {path} cannot be read ({type(exc).__name__}: {_one_line(exc)})") from exc
            if not isinstance(data, dict):                                  # null, a list, a number: not what the registry reads
                raise SourceError(f"{source}: {path} must hold a JSON object, not {type(data).__name__}")
            self._json[source] = data
        return self._json[source]

    def text(self, source: str) -> str:
        path = self.path(source)
        try:
            return path.read_text()
        except (UnicodeDecodeError, OSError) as exc:
            raise SourceError(f"{source}: {path} cannot be read ({type(exc).__name__}: {_one_line(exc)})") from exc

    def sha(self, source: str) -> str:
        p = self.path(source)
        if p is None or not p.exists():
            return ""
        try:
            return hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        except OSError as exc:                                          # unreadable, or a directory
            raise SourceError(f"{source}: {p} cannot be read ({type(exc).__name__}: {_one_line(exc)})") from exc

    def did_not_run(self, test_id: str) -> bool:
        """True when sealed/<test_id>/results.json exists and records that the coverage rule failed (coverage.ran false)."""
        source = f"sealed/{test_id}/results.json"
        if not self.exists(source):
            return False
        coverage = self.json(source).get("coverage")                # json() returns a dict, or raises
        return isinstance(coverage, dict) and coverage.get("ran") is False

    def legacy_seeds(self) -> set[int]:
        """The seeds in the names of the legacy run files under out_dir/legacy (empty when there are none)."""
        return {int(m.group(1)) for f in (self.out_dir / "legacy").glob("*.csv") if (m := _RUN_SEED.search(f.name))}


def _source_digest(module: str) -> str:
    """Full SHA-256 of es/<module>.py, or of the pipeline's v2/<name>.py for "v2.<name>"; "" when the file is not there."""
    path = ES_DIR.parent / "v2" / f"{module[3:]}.py" if module.startswith("v2.") else ES_DIR / f"{module}.py"
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def code_sha(key: str) -> str:
    """SHA-256 (first 16 hex digits) of the file that defines the code constant ``key`` (es/<module>.py or the pipeline's
    v2/<module>.py, read without importing it), of the installed package's METADATA, or of the interpreter's version.
    A source of several modules ("analyses+bootstrap") gives one digest over their own digests; a source file that is
    not there (yet) gives ""."""
    source = CODE_SOURCES.get(key, "paper_registry")
    if source.startswith("pkg:"):
        try:
            text = md.distribution(source[4:]).read_text("METADATA") or ""
        except md.PackageNotFoundError:
            return ""                                                       # "not installed": there is nothing to hash
        return hashlib.sha256(text.encode()).hexdigest()[:16]
    if source == "python":
        return hashlib.sha256(sys.version.encode()).hexdigest()[:16]
    digests = [_source_digest(m) for m in source.split("+")]
    if not all(digests):
        return ""
    return digests[0][:16] if len(digests) == 1 else hashlib.sha256("".join(digests).encode()).hexdigest()[:16]


def _pointer(obj, pointer: str):
    for part in pointer.split("/"):
        if isinstance(obj, list):
            obj = obj[int(part)]
        else:
            obj = obj[part]
    return obj


def _is_null(v) -> bool:
    """A JSON null or an empty CSV cell: what a nullable macro prints as n/a."""
    return v is None or (isinstance(v, float) and math.isnan(v))


def _format(entry: Entry, values: list):
    if entry.fmt in ROW_FORMATS:                                    # the labels of one row, some of which may be missing
        return ROW_FORMATS[entry.fmt](values)
    if entry.fmt in PAIR_FORMATS:
        if len(values) != 2:
            raise LookupError(f"{entry.name}: pair format {entry.fmt} needs two values, got {values}")
        return PAIR_FORMATS[entry.fmt](values[0], values[1])
    if len(values) != 1:
        raise LookupError(f"{entry.name}: expected one value, got {values}")
    return FORMATS[entry.fmt](values[0])


def _query_of(entry: Entry, df: pd.DataFrame) -> str:
    """The entry's query, and-ed with its when_query when the file has the when_column (a file written before that column
    existed is read as it always was)."""
    if entry.when_column and entry.when_column in df.columns:
        return f"({entry.query}) and ({entry.when_query})" if entry.query else entry.when_query
    return entry.query


def _from_frame(entry: Entry, df: pd.DataFrame) -> list:
    query = _query_of(entry, df)
    try:
        sel = df.query(query, engine="python") if query else df
    except (NameError, SyntaxError, TypeError, AttributeError, ValueError, KeyError) as exc:   # UndefinedVariableError is a NameError
        raise LookupError(f"{entry.name}: query {query!r} failed on {entry.source} (columns {list(df.columns)}): "
                          f"{type(exc).__name__}: {exc}") from exc
    cols = [c for c in entry.column.split(",") if c]
    for c in cols:
        if c not in sel.columns:
            raise LookupError(f"{entry.name}: column {c!r} not in {entry.source} (columns {list(df.columns)})")
    if entry.agg == "count":
        return [int(len(sel))]
    if entry.agg == "one":
        if len(sel) != 1:
            raise LookupError(f"{entry.name}: query {query!r} on {entry.source} matched {len(sel)} rows, expected 1")
        return [sel.iloc[0][c] for c in cols]
    if entry.agg == "range" and len(sel) != 2:                      # a gap needs two ends: never a silent 0 or a wrong gap
        raise LookupError(f"{entry.name}: query {query!r} on {entry.source} matched {len(sel)} rows, but agg=range is the "
                          f"gap between exactly 2 rows")
    if sel.empty:
        raise LookupError(f"{entry.name}: query {query!r} on {entry.source} matched no rows")
    if entry.agg == "argmax":                                       # the row where `by` is largest, the first on a tie (Ruling 101)
        if not entry.by:
            raise LookupError(f"{entry.name}: agg=argmax needs a `by` column (the one whose largest value picks the row)")
        if entry.by not in sel.columns:
            raise LookupError(f"{entry.name}: column {entry.by!r} (`by`) not in {entry.source} (columns {list(df.columns)})")
        rates = pd.to_numeric(sel[entry.by], errors="coerce")
        if rates.isna().any():                                      # the rule refuses a NaN rate too: a worst case never skips one
            raise LookupError(f"{entry.name}: {int(rates.isna().sum())} of the {len(sel)} rows that query selects have no "
                              f"{entry.by} in {entry.source}: a worst case must not pass over a missing number")
        row = sel.iloc[int(np.argmax(rates.to_numpy()))]
        return [row[c] for c in cols]
    if entry.agg == "same":                                         # one value for all the selected rows (the replications)
        for c in cols:
            distinct = sel[c].drop_duplicates()
            if len(distinct) != 1:
                raise LookupError(f"{entry.name}: the {len(sel)} rows that query selects on {entry.source} differ in {c}: "
                                  f"{sorted(map(str, distinct))}")
        return [sel.iloc[0][c] for c in cols]
    try:
        if entry.agg == "range":                                    # the gap between the two selected rows: max - min
            return [sel[c].max() - sel[c].min() for c in cols]
        return [getattr(sel[c], entry.agg)() for c in cols]
    except (AttributeError, TypeError, ValueError) as exc:
        raise LookupError(f"{entry.name}: cannot take the {entry.agg} of {cols} on {entry.source}: {exc}") from exc


def _pull_record(value) -> str:
    """The controller's sentence on the pull of Test A (Ruling 121; macro pullRecordTestA): chunks accepted incomplete and days excused,
    from the pull's own log, or that there were none. Plain text of one line (a line break or a run of spaces becomes one space, so that
    no blank line ends the paragraph), ASCII only (the journal class cannot typeset every Unicode sign), at most 700 characters, with no
    template text left in it (an angle bracket, XXXXX), and whole sentences: it ends with a full stop."""
    text = " ".join(str(value).split())
    if registration.has_placeholder(text):
        raise ValueError(f"{text[:60]!r} still holds a placeholder (an angle bracket or XXXXX)")
    if not text.isascii():
        raise ValueError(f"{text[:60]!r} has a character that is not ASCII: write plain text")
    if len(text) > 700:
        raise ValueError(f"the sentence has {len(text)} characters, 700 at most: say what the log says and no more")
    if not text.endswith("."):
        raise ValueError(f"{text[-60:]!r} does not end with a full stop: write whole sentences")
    return text


#: Validators of the author_inputs.json values: the one sentence that only the controller can write (pull_record_test_a). The identifiers
#: of the author are text, as they stand.
VALIDATORS = {"pull_record_test_a": _pull_record}


def _validated(entry: Entry, key: str, value):
    check = VALIDATORS.get(key)
    if check is None:
        return value
    try:
        return check(value)
    except ValueError as exc:
        raise LookupError(f"{entry.name}: {exc}") from exc


def _author_input(entry: Entry, sources: Sources):
    """'author:<key>': the value of that key of paper/author_inputs.json. Returns (value, source used, why it is missing). No author file
    yet gives "" (pending); an author file without the key is a LookupError (a registry bug); an empty value is pending; a value that
    fails its form rule (VALIDATORS) is a LookupError."""
    used, why = "author:", "empty author field"
    key = entry.source.split(":", 1)[1]
    if not sources.exists("author:"):
        return "", used, why
    data = sources.json("author:")
    if key not in data:
        raise LookupError(f"{entry.name}: author_inputs.json has no key {key!r} (it has {sorted(data)})")
    v = data[key]
    return ("", used, why) if is_missing(v) else (_validated(entry, key, v), used, "")


def _extract(entry: Entry, value):
    """The one group of ``entry.pattern`` in the strings of ``value`` (a string, or a list of strings): the pattern must
    match exactly once in all of them together. Two matches, in two strings or in one, are never resolved by picking one."""
    if re.compile(entry.pattern).groups < 1:
        raise LookupError(f"{entry.name}: the pattern {entry.pattern!r} needs one group")
    texts = [value] if isinstance(value, str) else [t for t in value if isinstance(t, str)] if isinstance(value, list) else []
    hits = [m.group(1) for t in texts for m in re.finditer(entry.pattern, t)]
    if len(hits) != 1:
        raise LookupError(f"{entry.name}: the pattern {entry.pattern!r} matched {len(hits)} times in the {len(texts)} strings "
                          f"at {entry.query!r} in {entry.source}, expected exactly 1")
    return hits[0]


def _hash_from_file(entry: Entry, text: str) -> str:
    """The SHA-256 a hash file starts with (the first token of its first line: a bare hash or shasum's own format)."""
    lines = [line for line in text.splitlines() if line.strip()]
    token = lines[0].split()[0] if lines else ""
    if not re.fullmatch(r"[0-9a-fA-F]{64}", token):
        raise LookupError(f"{entry.name}: {entry.source} must start with a SHA-256 (64 hexadecimal characters), "
                          f"got {token[:24]!r}")
    return token.lower()


def _shift_back(day: date, years: int) -> date:
    """The same calendar day ``years`` earlier; 29 February in a year without one is 28 February."""
    try:
        return day.replace(year=day.year - years)
    except ValueError:
        return day.replace(year=day.year - years, day=28)


def _raw(by: dict, name: str, sources: Sources):
    """The unformatted value of another entry (one registry row, one column): the same query as that macro's own."""
    entry = by[name]
    return _from_frame(entry, sources.frame(entry.source))[0]


def _placebo_fallback_days(entry: Entry, sources: Sources) -> tuple[int, str]:
    """Days on which the placebo pairing of spec 6.2(a) falls back to CAISO's real-time mean. The analogue alerts of
    AQ1-AQ4 (es.analyses.ANALOGS, deployment windows) are paired with P&L from the same quarter one and two years earlier.
    CAISO has a P&L from its panel start (firstDateCaiso) and a settlement-leg (FMM) P&L only from fmmFirstDate on, so
    the days of those shifted windows from the panel start to the day before the FMM first date use the fallback.
    Each day counts once, however many analogues and shifts reach it."""
    by = entries_by_name()
    panel_first = date.fromisoformat(str(_raw(by, "firstDateCaiso", sources))[:10])
    fmm_first = date.fromisoformat(str(_raw(by, "fmmFirstDate", sources))[:10])
    analyses = importlib.import_module(".analyses", __package__)
    days: set[int] = set()
    used = []
    for aid, analysis in analyses.ANALOGS.items():
        if not re.fullmatch(r"AQ[1-4]", aid):
            continue
        window = date_window(analysis.deploy, f"es.analyses.ANALOGS[{aid!r}].deploy")
        start, end = date.fromisoformat(window.start), date.fromisoformat(window.end)
        used.append(aid)
        for years in (1, 2):
            lo = max(panel_first, _shift_back(start, years))
            hi = min(_shift_back(end, years), fmm_first - timedelta(days=1))
            days.update(range(lo.toordinal(), hi.toordinal() + 1))
    if not used:
        raise LookupError("es.analyses.ANALOGS has none of AQ1-AQ4")
    first, last = (date.fromordinal(min(days)), date.fromordinal(max(days))) if days else (None, None)
    return len(days), (f"days of the AQ1-AQ4 deployment windows ({', '.join(sorted(used))}) shifted back 1 and 2 years that lie "
                       f"from the CAISO panel start {panel_first} to the day before the FMM first date {fmm_first}"
                       + (f" ({first} to {last})" if days else " (none)"))


ETA_TABLE = "analogs/eta_selection.csv"


def _eta_at_grid_top(entry: Entry, sources: Sources) -> tuple[bool, str]:
    """Whether eta*, the step size the analogues chose (protocol_inputs.json), is the largest value of the grid it was chosen
    from (the etas of eta_selection.csv): its optimum is then at the edge of the grid. eta* must be a point of the grid."""
    eta = float(sources.json(PROTOCOL)["eta"])
    grid = sorted({float(e) for e in sources.frame(ETA_TABLE)["eta"]})
    if not any(math.isclose(eta, g, rel_tol=1e-9) for g in grid):
        raise LookupError(f"eta* = {eta:g} is not a point of the grid {grid} in {ETA_TABLE}")
    return math.isclose(eta, grid[-1], rel_tol=1e-9), f"eta* = {eta:g}; the grid of {ETA_TABLE} is {', '.join(f'{g:g}' for g in grid)}"


# The false-demotion bound of the registration rule (es.run_analogs._size_bound and _placebo_bound, Ruling 94 M7), recomputed from the
# rule's constants: the probability, under independent rows, that a correctly sized test is demoted. The engine never imports
# run_analogs (a heavy module that another task edits); the recomputation must agree with the stored bound or the macro is an error.
def _binom_upper(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p)."""
    return float(sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(max(k, 0), n + 1)))


def _oversized(rate: float, n: int, a: float, factor: float, level: float, rows: int = 1) -> bool:
    """The demotion condition: the rate exceeds factor * a and a one-sided binomial test of the count against a rejects at
    level / rows (Bonferroni over the rows that enter the maximum)."""
    return bool(n > 0 and rate > factor * a and _binom_upper(round(rate * n), n, a) < level / rows)


@lru_cache(maxsize=None)
def _smallest_demoting_count(n: int, a: float, factor: float, level: float, rows: int = 1):
    return next((k for k in range(1, n + 1) if _oversized(k / n, n, a, factor, level, rows)), None)


def _size_bound(ns: list[int], levels: list[float], factor: float, level: float) -> float:
    """Per level 1 - the product over the rows of (1 - the tail of the smallest demoting count), summed over the levels."""
    total = 0.0
    for a in levels:
        none = 1.0
        for n in ns:
            k = _smallest_demoting_count(n, a, factor, level, len(ns))
            if k is not None:
                none *= 1.0 - _binom_upper(k, n, a)
        total += 1.0 - none
    return min(total, 1.0)


def _placebo_bound(n: int, levels: list[float], factor: float, level: float) -> float:
    """The same for the pooled placebo check on n counted pairings: the sum over the levels of the tail of the smallest demoting count."""
    total = 0.0
    for a in levels:
        k = _smallest_demoting_count(n, a, factor, level) if n else None
        if k is not None:
            total += _binom_upper(k, n, a)
    return min(total, 1.0)


def _he_false_demotion(sources: Sources) -> tuple[float, float, str]:
    """(simulation part, placebo part, detail) of H-E's false-demotion bound, which must add up to the stored one."""
    doc = sources.json(PROTOCOL)
    if doc.get("stage") in EARLY_JSON_STAGES:
        raise CodeUnavailable(f"{PROTOCOL} is at stage {doc['stage']!r}: the pre-specified family is written by a later stage")
    rule, he = doc["rule"], doc["family"]["H-E"]
    factor, level, levels = float(rule["factor"]), float(rule["binomial_level"]), [float(a) for a in rule["levels"]]
    rows = sources.frame(SIM).query(entries_by_name()["heSizePooledFive"].query, engine="python")      # the rows the rule reads
    ns, pooled, stored = [int(n) for n in rows["n_sim"]], int(he["placebo_counted"]["pooled"]), float(he["false_demotion_bound"])
    sim_part, placebo_part = _size_bound(ns, levels, factor, level), _placebo_bound(pooled, levels, factor, level)
    if not math.isclose(min(1.0, sim_part + placebo_part), stored, rel_tol=1e-9, abs_tol=1e-12):
        raise LookupError(f"the bound recomputed from the rule ({sim_part:.6f} from {len(ns)} simulation rows + {placebo_part:.6f} from "
                          f"{pooled} pooled placebo pairings) is not the stored family/H-E/false_demotion_bound = {stored:.6f}")
    return sim_part, placebo_part, (f"recomputed from the rule: {len(ns)} rows {sim_part:.4f} + {pooled} pooled placebo "
                                    f"pairing(s) {placebo_part:.4f} = {stored:.4f}, the stored false_demotion_bound")


def _he_false_demotion_sim(entry: Entry, sources: Sources) -> tuple[float, str]:
    sim_part, _, detail = _he_false_demotion(sources)
    return sim_part, detail


def _he_false_demotion_placebo(entry: Entry, sources: Sources) -> tuple[float, str]:
    _, placebo_part, detail = _he_false_demotion(sources)
    return placebo_part, detail


# The supplementary size check of gate G1 (Ruling 118, Seat M, M8): the registered rotation test of H-E on synthetic data with a market-wide
# factor, market-wide spike days, daily persistence 0.95 and independent trends within each window, which the size study (independent
# nodes, Gaussian shocks, AR(1) days) does not contain. The probes and their outputs are archived in docs/reviews/g1_probes and are not
# part of the registration rule; the macros are the worst pooled and single-window rejection rates of those outputs.
SIZE_CHECK_FILES = ("docsreviews:g1_probes/he_common_500.txt", "docsreviews:g1_probes/he_trend_500.txt")
_SCENARIO = re.compile(r"^(?P<label>S\d+\s.*?)\s+n=(?P<n>\d+)\s+single A:\s*(?P<s3>\d*\.\d+)\s*/\s*(?P<s5>\d*\.\d+)"
                       r"\s+pooled A\+B:\s*(?P<p3>\d*\.\d+)\s*/\s*(?P<p5>\d*\.\d+)")
_PHI = re.compile(r"\bphi\s+(\d*\.\d+)")


def _size_check_scenarios(name: str, sources: Sources) -> list[dict]:
    """The scenario lines of the two probe outputs: label, replications, and the single-window and pooled rejection rates at the first
    Holm level (alpha / 3) and at the family-wise level. Each file must hold at least one; every line must be one; the replications of
    all of them must agree, or the maximum of the rates would compare unlike runs."""
    scenarios = []
    for rel in SIZE_CHECK_FILES:
        found = 0
        for line in sources.text(rel).splitlines():
            if not line.strip():
                continue
            m = _SCENARIO.match(line)
            if m is None:
                raise LookupError(f"{name}: {rel}: the line {line.strip()[:80]!r} is not a scenario line "
                                  "(S<k> <label> n=<replications> single A: a / b pooled A+B: a / b)")
            scenarios.append({"file": Path(rel).name, "label": m["label"].strip(), "n": int(m["n"]),
                              **{k: float(m[k]) for k in ("s3", "s5", "p3", "p5")}})
            found += 1
        if not found:
            raise LookupError(f"{name}: {rel} holds no scenario line")
    if len({r["n"] for r in scenarios}) != 1:
        raise LookupError(f"{name}: the probe outputs differ in their replications per scenario "
                          f"({sorted({r['n'] for r in scenarios})}): the worst rates would not be comparable")
    return scenarios


def _size_check(kind: str):
    """The reader of one size-check macro: the worst rate of the scenarios (kind "p3": pooled test at alpha / 3, "p5": pooled at the
    family-wise level, "s3" and "s5" the same for one window), the replications ("n") or the largest daily persistence ("phi")."""
    def read(entry: Entry, sources: Sources) -> tuple[float, str]:
        scenarios = _size_check_scenarios(entry.name, sources)
        files = ", ".join(dict.fromkeys(r["file"] for r in scenarios))
        if kind == "n":
            return scenarios[0]["n"], f"replications per scenario in {files}"
        if kind == "phi":
            phis = [float(m[1]) for r in scenarios for m in _PHI.finditer(r["label"])]
            if not phis:
                raise LookupError(f"{entry.name}: no scenario label of {files} names a persistence (phi .<digits>)")
            return max(phis), f"the largest persistence named by the {len(scenarios)} scenario labels of {files}"
        worst = max(scenarios, key=lambda r: r[kind])
        return worst[kind], f"the largest of {len(scenarios)} scenarios of {files}: {worst['label']!r} ({worst['file']})"
    return read


#: Derived numbers by name: (reader(entry, sources) -> (value, detail), the output files it reads, the es modules it reads).
DERIVED = {"placeboFallbackDays": (_placebo_fallback_days, ("paper/coverage.csv",), ("analyses",)),
           "etaAtGridTop": (_eta_at_grid_top, (PROTOCOL, ETA_TABLE), ()),
           "heFalseDemotionSim": (_he_false_demotion_sim, (PROTOCOL, SIM), ()),
           "heFalseDemotionPlacebo": (_he_false_demotion_placebo, (PROTOCOL, SIM), ()),
           **{f"sizeCheck{word}": (_size_check(kind), SIZE_CHECK_FILES, ())
              for word, kind in (("PooledThird", "p3"), ("PooledFive", "p5"), ("SingleThird", "s3"), ("SingleFive", "s5"),
                                 ("Reps", "n"), ("Phi", "phi"))}}


def _derived_sha(name: str, sources: Sources) -> str:
    """One digest over the digests of the files and modules a derived number is computed from."""
    _, files, modules = DERIVED[name]
    return hashlib.sha256(("".join(sources.sha(f) for f in files) + "".join(_source_digest(m) for m in modules)).encode()).hexdigest()[:16]


def _sealed_test(entry: Entry) -> str | None:
    m = _SEALED.match(entry.source)
    return m.group(1) if m else None


def _marks_estimand(entry: Entry, doc: dict) -> tuple[bool, str]:
    """Whether the family of the run marks ``entry.estimand_of`` as an estimand, and why. results.json["family"] is the list of the
    confirmatory hypotheses of the run (es.sealed_run), so a hypothesis that is not in it is an estimand; a registered family
    (protocol_inputs.json: {hypothesis: {"confirmatory": ...}}) is read the same way. A results.json without a family says it in
    the confirmatory flag of the hypothesis' own entry. With none of them the macro cannot tell, and refuses."""
    hid, family = entry.estimand_of, doc.get("family")
    if isinstance(family, (list, tuple)):
        return hid not in family, f"{hid} is an estimand: it is not among the confirmatory hypotheses of the run's family {list(family)}"
    flag = None
    if isinstance(family, dict) and isinstance(family.get(hid), dict):
        flag = family[hid].get("confirmatory")
    elif isinstance(doc.get(hid), dict):
        flag = doc[hid].get("confirmatory")
    if flag is None:
        raise LookupError(f"{entry.name}: cannot tell whether {hid} is an estimand: {entry.source} has no family that lists it and "
                          f"no {hid}/confirmatory")
    return FORMATS["yesno"](flag) != "yes", f"{hid} is an estimand: the family of {entry.source} marks it as not confirmatory"


def _in_not_run_header(entry: Entry) -> bool:
    return entry.source.endswith("results.json") and entry.query.split("/")[0] in NOT_RUN_KEYS


def resolve(entry: Entry, sources: Sources) -> dict:
    if not macro_name_ok(entry.name):
        raise ValueError(f"illegal macro name {entry.name!r}")
    out = {"name": entry.name, "stage": entry.stage, "source": entry.source, "query": entry.query,
           "column": entry.column, "fmt": entry.fmt, "sha": "", "status": "ok", "detail": ""}
    try:
        test_id = _sealed_test(entry)
        if test_id and not _in_not_run_header(entry) and sources.did_not_run(test_id):
            out.update(status="pending", value=PENDING[entry.stage], detail="test did not run",
                       sha=sources.sha(f"sealed/{test_id}/results.json"))
            return out
        if entry.source.startswith("code:"):
            key = entry.source[5:]
            out["sha"] = code_sha(key)
            if key in SPEC_FIXED:                                       # no code carries it: the spec fixes it (provenance row)
                out["detail"] = f"fixed by {SPEC_FIXED[key]}"
            if key in CODE_READERS:                                     # read now, from the source, not at import
                try:
                    values = [CODE_READERS[key]()]
                except CodeUnavailable as exc:                          # the source is not written yet: pending, not an error
                    out.update(status="pending", value=PENDING[entry.stage], detail=str(exc))
                    return out
                except (ImportError, AttributeError, LookupError, ValueError, TypeError, OSError) as exc:
                    raise LookupError(f"{entry.name}: cannot read {key} from the code: {type(exc).__name__}: {exc}") from exc
            else:
                values = [CODE_CONSTANTS[key]]
        elif entry.source.startswith("derived:"):
            name = entry.source[8:]
            if name not in DERIVED:
                raise LookupError(f"{entry.name}: there is no derived number {name!r} (DERIVED has {sorted(DERIVED)})")
            reader, files, _ = DERIVED[name]
            absent = [f for f in files if not sources.exists(f)]
            if absent:
                out.update(status="pending", value=PENDING[entry.stage], detail=f"missing {sources.shown(absent[0])}")
                return out
            out["sha"] = _derived_sha(name, sources)
            try:
                value, out["detail"] = reader(entry, sources)
            except CodeUnavailable as exc:                          # an input that a later stage writes: pending, not an error
                out.update(status="pending", value=PENDING[entry.stage], detail=str(exc))
                return out
            except (ImportError, AttributeError, LookupError, ValueError, TypeError, OSError) as exc:
                if isinstance(exc, SourceError):
                    raise
                raise LookupError(f"{entry.name}: cannot derive it: {type(exc).__name__}: {exc}") from exc
            values = [value]
        elif entry.source.startswith("author:"):
            v, used, why = _author_input(entry, sources)
            out["sha"] = sources.sha(used)
            if is_missing(v):
                out.update(status="pending", value=PENDING[entry.stage], detail=why)
                return out
            values = [v]
        elif not sources.exists(entry.source):
            out.update(status="pending", value=PENDING[entry.stage], detail=f"missing {sources.shown(entry.source)}")
            return out
        elif entry.source.startswith("docsfile:"):                      # a file of docs/sealed that holds one hash
            out["sha"] = sources.sha(entry.source)
            values = [_hash_from_file(entry, sources.text(entry.source))]
        elif entry.source.endswith(".json"):
            out["sha"] = sources.sha(entry.source)
            doc = sources.json(entry.source)
            if entry.estimand_of:                                       # Ruling 107: an estimand has no confirmatory p-value
                estimand, why = _marks_estimand(entry, doc)
                if estimand:
                    out.update(value=ESTIMAND_TEXT, detail=why)
                    return out
            try:
                v = _pointer(doc, entry.query)
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                if doc.get("stage") in EARLY_JSON_STAGES:               # Ruling 98: a later stage writes this key, so it is pending
                    out.update(status="pending", value=PENDING[entry.stage],
                               detail=f"{entry.source} is at stage {doc['stage']!r}: {entry.query!r} is written by a later stage")
                    return out
                raise LookupError(f"{entry.name}: {entry.query!r} not found in {entry.source}: {exc!r}") from exc
            if entry.pattern:
                values = [_extract(entry, v)]
            elif entry.agg == "count":                                  # the number of entries (the failed analyses of results.json)
                if not isinstance(v, (dict, list)):
                    raise LookupError(f"{entry.name}: {entry.query!r} in {entry.source} is neither an object nor a list, so it has "
                                      f"no count")
                values = [len(v)]
            else:
                values = list(v) if entry.fmt in PAIR_FORMATS and isinstance(v, (list, tuple)) else [v]
        else:
            out["sha"] = sources.sha(entry.source)
            frame = sources.frame(entry.source)
            values = _from_frame(entry, frame)
            if _query_of(entry, frame) != entry.query:                  # the file has the column: say which rows were read
                out["detail"] = f"rows also filtered by {entry.when_query} (the file has {entry.when_column})"
        if entry.nullable and all(_is_null(v) for v in values):
            out["value"] = "n/a"                                        # a null or NaN the contract allows (Holm, a failed power)
            return out
        if entry.fmt not in ROW_FORMATS and any(is_missing(v) for v in values):
            raise LookupError(f"{entry.name}: value is missing/NaN in {entry.source}")
        try:
            out["value"] = _format(entry, values)
        except (ValueError, ArithmeticError) as exc:    # a formatter refusal (ArithmeticError: OverflowError, InvalidOperation)
            raise LookupError(f"{entry.name}: cannot format {values!r} as {entry.fmt!r}: {exc!r}") from exc
    except SourceError as exc:
        raise LookupError(f"{entry.name}: {exc}") from exc
    except (KeyError, IndexError) as exc:
        raise LookupError(f"{entry.name}: {entry.query!r} not found in {entry.source}: {exc}") from exc
    return out


def resolve_all(entries: list[Entry], sources: Sources) -> list[dict]:
    return [resolve(e, sources) for e in entries]


def sealed_flags(sources: Sources) -> dict[str, bool]:
    """{"A": results.json present, "B": ..., "Adidnotrun": coverage rule failed, "Bdidnotrun": ...}."""
    flags = {t: sources.exists(f"sealed/{t}/results.json") for t in ("A", "B")}
    flags.update({f"{t}didnotrun": sources.did_not_run(t) for t in ("A", "B")})
    return flags


def check_legacy_seeds(sources: Sources) -> None:
    """The seeds in the legacy output file names must be exactly SEEDS, or the seed macros describe another run."""
    found = sources.legacy_seeds()
    if found and found != set(SEEDS):
        raise LookupError(f"the legacy outputs under {sources.out_dir / 'legacy'} were run with seeds {sorted(found)} but "
                          f"SEEDS (es.run_legacy.SEEDS) is {sorted(SEEDS)}: the seed macros would describe another run")


def check_simulation_rule(sources: Sources) -> None:
    """The registry types what it reads of the simulation stage (the laws of each H-B test, the levels the columns are named
    after, the couplings of the power curve) and selects rows by queries; the stage records the rule it used. They must agree,
    or the size macros would read other rows than the rule did (Ruling 101). Nothing to compare until the stage is complete."""
    if not (sources.exists(PROTOCOL) and sources.exists(SIM)):
        return
    doc = sources.json(PROTOCOL)
    rule = doc.get("rule")
    if doc.get("stage") != "complete" or not isinstance(rule, dict):
        return
    sim, problems, by = sources.frame(SIM), [], entries_by_name()
    for hid, laws in HB_LAWS.items():
        read = (rule.get("laws") or {}).get(hid)
        if read is None or set(read) != set(laws):
            problems.append(f"{hid}: the rule reads the laws {read}, the registry {list(laws)} (HB_LAWS)")
    levels = [f"reject_{float(a):g}" for a in rule.get("levels") or []]
    if levels != [col for _, col in SIM_LEVELS]:
        problems.append(f"the rule's levels name the columns {levels}, the registry reads {[col for _, col in SIM_LEVELS]} (SIM_LEVELS)")
    for prefix, (_, _, _, key) in SIZE_TESTS.items():
        try:
            expected = _pointer(rule, key)
        except (KeyError, IndexError, TypeError):
            problems.append(f"{prefix}: the rule has no {key}")
            continue
        read = len(sim.query(by[f"{prefix}Five"].query, engine="python"))
        if read != expected:
            problems.append(f"{prefix}: the registry's selection reads {read} rows of {SIM}, the rule's {key} is {expected}")
    grid = sorted(float(r) for r in sim.loc[sim["test"] == CURVE_TEST, "rho"])
    named = sorted(RHO_WORD)
    if len(grid) != len(named) or not all(math.isclose(a, b, rel_tol=1e-9) for a, b in zip(grid, named)):
        problems.append(f"the power curve of {SIM} has the couplings rho = {grid}, the registry names rho = {named} (RHO_WORD)")
    if problems:
        raise LookupError("the simulation stage's rule is not the one the registry reads: " + "; ".join(problems))


def render(resolved: list[dict], flags: dict[str, bool]) -> str:
    lines = ["%% GENERATED by dart_dev/es/paper_macros.py. Do not edit; rerun paper/build.sh macros.",
             f"%% built {datetime.now(timezone.utc).isoformat(timespec='seconds')}", ""]
    for stage, cmd in PENDING.items():
        lines.append(f"\\newcommand{{{cmd}}}{{\\textbf{{{PENDING_TEXT[stage]}}}}}")
    for t in ("A", "B"):
        lines += [f"\\newif\\ifTest{t}", f"\\Test{t}{'true' if flags[t] else 'false'}",
                  f"\\newif\\ifTest{t}didnotrun", f"\\Test{t}didnotrun{'true' if flags.get(t + 'didnotrun') else 'false'}"]
    lines.append("")
    for r in sorted(resolved, key=lambda r: r["name"]):
        lines.append(f"\\newcommand{{\\{r['name']}}}{{{r['value']}}}")
    return "\n".join(lines) + "\n"


def write_outputs(resolved: list[dict], generated_dir: Path, flags: dict[str, bool]) -> None:
    generated_dir.mkdir(parents=True, exist_ok=True)
    (generated_dir / "macros.tex").write_text(render(resolved, flags), encoding="utf-8")   # \u00d7 is in the strata texts
    pd.DataFrame(resolved)[["name", "stage", "status", "value", "source", "query", "column", "fmt", "sha", "detail"]] \
        .to_csv(generated_dir / "provenance.csv", index=False)
    pd.DataFrame([e.__dict__ for e in ENTRIES]).to_csv(generated_dir / "registry.csv", index=False)
    info = {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "test_flags": flags,
            "n_macros": len(resolved), "n_pending": sum(r["status"] == "pending" for r in resolved),
            "pending_by_stage": {s: sum(r["status"] == "pending" and r["stage"] == s for r in resolved) for s in PENDING}}
    (generated_dir / "BUILD_INFO.json").write_text(json.dumps(info, indent=2))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="exit 1 if any macro of --stages is pending")
    ap.add_argument("--stages", default=",".join(DEFAULT_STRICT_STAGES),
                    help="comma-separated stages that --strict checks, from " + ", ".join(PENDING))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--paper-dir", default=str(PAPER_DIR))
    ap.add_argument("--docs-sealed", default=str(DOCS_SEALED))
    ap.add_argument("--only", default="", help="comma-separated macro names (tests); the partial macros.tex it writes "
                                               "belongs in a scratch --paper-dir, not in paper/generated/")
    ap.add_argument("--overwrite-real", action="store_true",
                    help="let --only replace the real paper/generated/ with a partial macros.tex")
    a = ap.parse_args(argv)
    stages = [s for s in a.stages.split(",") if s]
    unknown = [s for s in stages if s not in PENDING]
    if unknown:
        ap.error(f"unknown stage(s) {unknown}; choose from {list(PENDING)}")
    if a.only and not a.overwrite_real and Path(a.paper_dir).resolve() == Path(PAPER_DIR).resolve():
        ap.error("--only writes a partial macros.tex: give a scratch --paper-dir, or --overwrite-real to replace "
                 "the real paper/generated/")
    sources = Sources(Path(a.out_dir), Path(a.paper_dir), Path(a.docs_sealed))
    check_legacy_seeds(sources)
    check_simulation_rule(sources)
    by = entries_by_name()
    entries = [by[n] for n in a.only.split(",") if n] if a.only else ENTRIES
    resolved = resolve_all(entries, sources)
    flags = sealed_flags(sources)
    write_outputs(resolved, Path(a.paper_dir) / "generated", flags)
    pending = [r for r in resolved if r["status"] == "pending"]

    def state(t):
        return ("present, did not run" if flags[f"{t}didnotrun"] else "present") if flags[t] else "absent"

    print(f"{len(resolved)} macros, {len(pending)} pending; Test A {state('A')}, Test B {state('B')}")
    strict_pending = [r for r in pending if r["stage"] in set(stages)]
    if a.strict and strict_pending:
        for r in strict_pending:
            print(f"PENDING {r['stage']:8s} \\{r['name']}  ({r['detail']})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
