"""Registration gate for the sealed tests (spec v3 §6.5, invariant I1). The sealed windows open only
when docs/sealed/REGISTRATION.md names the SHA-256 of the unchanged protocol, a time, and exactly one registration of
it (Ruling 140): an OSF registration (a DOI, or for an embargoed registration, which may have no DOI yet, its OSF
address), a public release of the code repository (the address of a tagged GitHub release, whose publication time
GitHub records) or the author's local record (the date of the author's authorisation, with the time at which it was
recorded). The gate reads no network, so it cannot check any of them against the outside world: what protects the
sealed windows is that nothing opens without a deliberate, well-formed record of the exact hash of the protocol, and
what a document says about the record must be exactly what the record is. Files the protocol binds (pre-pull cutoffs,
protocol inputs) must match the SHA-256 the protocol prints for them. The protocol must also be ready to register:
es.protocol.ready_to_register finds nothing open in its text (no [AUTHOR: ...] marker, and the knowledge statement
filled in), so a draft that was recorded by mistake opens nothing.

The gate reads this module's own record paths (PROTOCOL_PATH, REGISTRATION_PATH) when it is called, never a
record a caller hands it. The record is read as `key: value` lines; a value is one token with no spaces. Required:
protocol_sha256, registered_utc and exactly one registration: osf_doi and/or osf_registration (which name one OSF
registration), or public_record, or local_record. A record that names more than one kind is refused. No key may
appear twice.

    protocol_sha256: <64 hexadecimal characters, in either case>
    registered_utc: 2026-01-15T09:30:00Z      (ISO 8601, in UTC: a trailing Z or +00:00, and not in the future)
    osf_doi: 10.17605/OSF.IO/XXXXX            (or https://doi.org/10.17605/OSF.IO/XXXXX; http also works)
    osf_registration: https://osf.io/xxxxx/   (or the bare 5 or 6 character OSF id; for a registration without a DOI)
    public_record: https://github.com/<owner>/<repository>/releases/tag/<tag>
                                              (https only; a release-tag address with nothing after the tag; the
                                              registered_utc above is the release's publication time)
    local_record: author-statement-2026-10-26 (the date of the author's authorisation; the registered_utc above is the
                                              UTC time at which the authorisation was recorded)

A line that names one of these six keys must hold exactly one token: a line left in the template with its
instructions, or a blank one, is refused (delete the line of an identifier or of a kind you do not use). A value that
still holds a placeholder (XXXXX, or an angle bracket) is refused, whichever key it belongs to. If both OSF identifiers
are given, both must be valid and must name the same registration: the id inside the DOI (10.17605/OSF.IO/<ID>) is the
id of the OSF address, whatever the case.

The identity of a registration, which binds a sealed pull to it: an OSF registration is its OSF id (lower case), a
public release is github:<owner>/<repository>/releases/tag/<tag> (owner and repository in lower case, the tag as
written) and a local record is local:<token>. registration_kind(record) is "osf", "public_release" or "local".
registration_identifier(record) is the one to log: the DOI if there is one, else the OSF address, else the release
address, else the local token; registration_id(record) is the id that all the identifiers of one registration stand
for, which does not change when an embargo lifts and the DOI joins the address (or replaces it).
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

from .config import REVISION_ROOT

SEALED_DIR = REVISION_ROOT / "docs" / "sealed"
PROTOCOL_PATH = SEALED_DIR / "PROTOCOL.md"
REGISTRATION_PATH = SEALED_DIR / "REGISTRATION.md"
CUTOFFS_PATH = SEALED_DIR / "pre_pull_cutoffs.csv"
INPUTS_PATH = SEALED_DIR / "protocol_inputs.json"
OSF_DOI_RE = re.compile(r"10\.17605/OSF\.IO/[A-Z0-9]+", flags=re.I)
DOI_LINK_RE = re.compile(r"^https?://doi\.org/", flags=re.I)
OSF_ID_RE = re.compile(r"[a-z0-9]{5,6}", flags=re.I)
OSF_URL_RE = re.compile(r"https?://osf\.io/([a-z0-9]{5,6})/?", flags=re.I)
UTC_STAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]00:?00)")
PLACEHOLDER_RE = re.compile(r"XXXXX|[<>]", flags=re.I)
#: A public release record (Ruling 140): https://github.com/<owner>/<repository>/releases/tag/<tag>, https only, with
#: nothing after the tag (no query, no fragment, no trailing slash, no path). The scheme and the host are matched in any
#: case, as the OSF address is; the pattern is not bound to one repository name.
PUBLIC_RECORD_RE = re.compile(r"(?i:https://github\.com)/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)/releases/tag/([A-Za-z0-9_.-]+)")
#: The token of the author's local record (Ruling 140): author-statement- and the date of the author's authorisation.
LOCAL_RECORD_RE = re.compile(r"author-statement-([0-9]{4}-[0-9]{2}-[0-9]{2})")
# The names es.paper_macros imports; they mean what they always did.
_OSF_DOI, _DOI_LINK, _UTC_STAMP = OSF_DOI_RE, DOI_LINK_RE, UTC_STAMP_RE
_GATE_KEYS = ("protocol_sha256", "osf_doi", "osf_registration", "public_record", "local_record", "registered_utc")
#: The kinds of registration a record may hold (Ruling 140), by the lines that name them. A record holds exactly one.
_KINDS = (("osf", ("osf_doi", "osf_registration")), ("public_release", ("public_record",)), ("local", ("local_record",)))
_RECORD_ADVICE = ("record the protocol in docs/sealed/REGISTRATION.md (an OSF registration, a public release or the "
                  "author's local record)")


class RegistrationError(PermissionError):
    """No valid registration record matches this protocol, so the sealed windows stay closed."""


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def is_iso_utc(value) -> bool:
    """True for an ISO 8601 date and time written in UTC (`Z` or an offset of zero), with real calendar values.
    A format check only: whether the time lies in the future is verify_registration's business."""
    if not UTC_STAMP_RE.fullmatch(value):
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


_is_iso_utc = is_iso_utc


def has_placeholder(value) -> bool:
    """True if a record value still holds template text: XXXXX in any case, or an angle bracket."""
    return bool(PLACEHOLDER_RE.search(value))


def normalise_osf_registration(value):
    """The OSF address https://osf.io/<id>/ (id in lower case) for an OSF registration URL, with or without the final
    slash and with http or https, or for a bare OSF id of 5 or 6 letters and digits; None for anything else."""
    url = OSF_URL_RE.fullmatch(value)
    if url:
        return f"https://osf.io/{url.group(1).lower()}/"
    if OSF_ID_RE.fullmatch(value):
        return f"https://osf.io/{value.lower()}/"
    return None


def normalise_public_record(value):
    """The release address https://github.com/<owner>/<repository>/releases/tag/<tag> with the scheme and the host in
    lower case, the owner and the repository in lower case and the tag as written, for the address of a tagged release
    of a GitHub repository (https only, nothing after the tag); None for anything else."""
    found = PUBLIC_RECORD_RE.fullmatch(value)
    if not found:
        return None
    owner, repository, tag = found.groups()
    return f"https://github.com/{owner.lower()}/{repository.lower()}/releases/tag/{tag}"


def is_local_record(value) -> bool:
    """True for author-statement-YYYY-MM-DD with a real calendar date: the token of the author's local record. A format
    check only, like is_iso_utc."""
    found = LOCAL_RECORD_RE.fullmatch(value)
    if not found:
        return False
    try:
        datetime.strptime(found.group(1), "%Y-%m-%d")
    except ValueError:
        return False
    return True


def _osf_id_of_doi(doi):
    """The OSF id (lower case) of a bare OSF DOI, 10.17605/OSF.IO/<ID>."""
    return doi.rsplit("/", 1)[1].lower()


def _osf_id_of_address(address):
    """The OSF id (lower case) of a normalised OSF address, https://osf.io/<id>/."""
    return address.rstrip("/").rsplit("/", 1)[1].lower()


def registration_ids(record) -> frozenset:
    """The ids a record names (Ruling 140): the OSF id (lower case) through its DOI and its OSF address,
    github:<owner>/<repository>/releases/tag/<tag> through its public record (owner and repository in lower case, the tag
    as written) and local:<token> through its local record. A record that parse_registration accepted has one: it
    refuses a DOI and an address that name different registrations, and a record that names more than one kind. A value
    that does not have the form of its kind names nothing."""
    ids = set()
    doi = record.get("osf_doi")
    if doi and OSF_DOI_RE.fullmatch(DOI_LINK_RE.sub("", doi)):
        ids.add(_osf_id_of_doi(DOI_LINK_RE.sub("", doi)))
    address = normalise_osf_registration(record["osf_registration"]) if record.get("osf_registration") else None
    if address:
        ids.add(_osf_id_of_address(address))
    release = PUBLIC_RECORD_RE.fullmatch(record["public_record"]) if record.get("public_record") else None
    if release:
        owner, repository, tag = release.groups()
        ids.add(f"github:{owner.lower()}/{repository.lower()}/releases/tag/{tag}")
    if record.get("local_record") and is_local_record(record["local_record"]):
        ids.add(f"local:{record['local_record']}")
    return frozenset(ids)


def registration_id(record) -> str:
    """The one id that a record's identifiers stand for: the registration itself, whichever of its identifiers the record
    carries (the OSF id of a DOI and of an OSF address, github:... for a public release, local:... for a local record).
    Raises RegistrationError for a record with none, or with two ids."""
    ids = registration_ids(record)
    if len(ids) != 1:
        raise RegistrationError(f"the record names {len(ids)} registrations ({sorted(ids)}), not one")
    return next(iter(ids))


def registration_identifiers(record) -> tuple:
    """Every identifier a record names: its DOI (bare form) first, then its OSF address, its release address and its
    local token; empty ones are left out."""
    return tuple(v for v in (record.get("osf_doi"), record.get("osf_registration"), record.get("public_record"),
                             record.get("local_record")) if v)


def registration_identifier(record) -> str:
    """The one identifier of a record to log and to compare: its DOI if it has one, else its OSF address, else the
    address of its release, else its local token. `record` is what parse_registration or verify_registration returns."""
    found = registration_identifiers(record)
    if not found:
        raise RegistrationError("the record names no registration (osf_doi, osf_registration, public_record or "
                                "local_record)")
    return found[0]


def registration_kind(record) -> str:
    """"osf", "public_release" or "local": the kind of registration that a record names. Raises RegistrationError for a
    record with none, or with more than one kind."""
    kinds = [kind for kind, keys in _KINDS if any(record.get(k) for k in keys)]
    if len(kinds) != 1:
        raise RegistrationError(f"the record names {len(kinds)} kinds of registration ({kinds}), not one")
    return kinds[0]


def parse_registration(registration_path=None) -> dict:
    """Read the record and check its form, not the clock and not the protocol: the file exists, no key repeats, each of
    the six keys of the gate that appears is on a line of the form `key: value` with one token, the required ones are
    there (protocol_sha256, registered_utc, and exactly one registration: osf_doi and/or osf_registration, or
    public_record, or local_record), no value is a placeholder, a DOI is an OSF DOI, an OSF address is
    https://osf.io/<id>/ or a bare id, a public record is the address of a tagged GitHub release, a local record is
    author-statement- and a real date, and registered_utc is an ISO 8601 UTC timestamp. Returns the fields with the DOI
    in bare form (None if not given), the OSF address as https://osf.io/<id>/ (None if not given), the release address
    with its host, owner and repository in lower case (None if not given), the local token (None if not given) and the
    hash in lower case, or raises RegistrationError."""
    registration_path = Path(REGISTRATION_PATH if registration_path is None else registration_path)
    if not registration_path.exists():
        raise RegistrationError(f"no registration record at {registration_path}; {_RECORD_ADVICE}")
    text = registration_path.read_text()
    pairs = re.findall(r"^(\w+):[ \t]*(\S+)[ \t]*$", text, flags=re.M)               # read_text() has made CRLF into LF
    keys = [k for k, _ in pairs]
    named = {k: re.findall(rf"^{k}:(.*)$", text, flags=re.M) for k in _GATE_KEYS}     # every line naming a gate key
    repeated = sorted({k for k in keys if keys.count(k) > 1} | {k for k, lines in named.items() if len(lines) > 1})
    if repeated:
        raise RegistrationError(f"registration record repeats {repeated}; each key may appear once")
    fields = dict(pairs)
    unparsed = [k for k in _GATE_KEYS if named[k] and k not in fields]
    if unparsed:
        raise RegistrationError(f"the line for {unparsed} is not of the form 'key: value' with one token and no spaces "
                                "(is a placeholder or a comment still there? delete the line of an identifier or of a "
                                "kind you do not use)")
    missing = [k for k in ("protocol_sha256", "registered_utc") if k not in fields]
    kinds = [kind for kind, lines in _KINDS if any(k in fields for k in lines)]          # the kinds that the record names
    if not kinds:
        missing.append("osf_doi or osf_registration, public_record or local_record")
    if missing:
        raise RegistrationError(f"registration record lacks {missing}")
    if len(kinds) > 1:
        raise RegistrationError(f"registration record names more than one kind of registration ({', '.join(kinds)}); "
                                "name exactly one: an OSF registration (osf_doi and/or osf_registration), a "
                                "public_record or a local_record")
    placeholders = sorted(k for k, v in fields.items() if has_placeholder(v))
    if placeholders:
        raise RegistrationError(f"registration record still holds a placeholder value for {placeholders}")
    doi = osf_registration = public_record = local_record = None
    if "osf_doi" in fields:
        doi = DOI_LINK_RE.sub("", fields["osf_doi"])                    # the DOI may be written as a doi.org link
        if not OSF_DOI_RE.fullmatch(doi):
            raise RegistrationError(f"osf_doi {fields['osf_doi']!r} is not an OSF DOI (10.17605/OSF.IO/...)")
    if "osf_registration" in fields:
        osf_registration = normalise_osf_registration(fields["osf_registration"])
        if osf_registration is None:
            raise RegistrationError(f"osf_registration {fields['osf_registration']!r} is not an OSF registration "
                                    "(https://osf.io/<id>/, or the bare 5 or 6 character id)")
    if doi and osf_registration and _osf_id_of_doi(doi) != _osf_id_of_address(osf_registration):
        raise RegistrationError(f"osf_doi {doi} and osf_registration {osf_registration} name different OSF "
                                f"registrations ({_osf_id_of_doi(doi)} and {_osf_id_of_address(osf_registration)})")
    if "public_record" in fields:
        public_record = normalise_public_record(fields["public_record"])
        if public_record is None:
            raise RegistrationError(f"public_record {fields['public_record']!r} is not the address of a tagged release "
                                    "of a GitHub repository (https://github.com/<owner>/<repository>/releases/tag/<tag>, "
                                    "https only, with nothing after the tag)")
    if "local_record" in fields:
        local_record = fields["local_record"]
        if not is_local_record(local_record):
            raise RegistrationError(f"local_record {local_record!r} is not a local record (author-statement- and the "
                                    "date of the author's authorisation, YYYY-MM-DD, for example "
                                    "author-statement-2026-10-26)")
    if not is_iso_utc(fields["registered_utc"]):
        raise RegistrationError(f"registered_utc {fields['registered_utc']!r} is not an ISO 8601 UTC timestamp "
                                "(for example 2026-01-15T09:30:00Z)")
    return {**fields, "protocol_sha256": fields["protocol_sha256"].lower(), "osf_doi": doi,
            "osf_registration": osf_registration, "public_record": public_record, "local_record": local_record}


def verify_registration(protocol_sha256, protocol_path=None, registration_path=None):
    """Return the record's fields (see parse_registration: the DOI as the bare 10.17605/... form, the OSF address as
    https://osf.io/<id>/, the release address with its host, owner and repository in lower case, the local token, each
    None if the record does not give it, and the hash in lower case), or raise RegistrationError. Called without paths
    it reads PROTOCOL_PATH and REGISTRATION_PATH as they are at that moment. The record must be well formed
    (parse_registration), registered at or before the current UTC time, and name the supplied hash; the protocol file is
    hashed again here and must still have it. Hashes compare in lower case. Last, the protocol must be ready to register:
    es.protocol.ready_to_register must find no open item in its text (an [AUTHOR: ...] marker, a row of the knowledge
    statement without exactly one box ticked, guidance left in), or this raises RegistrationError, which says to fill
    docs/sealed/KNOWLEDGE_STATEMENT.md, run es.protocol.main() again and register that PROTOCOL.md. pull_sealed and
    run_sealed both call this before they spend anything (a download, the lock), so an unfilled statement opens nothing."""
    protocol_path = Path(PROTOCOL_PATH if protocol_path is None else protocol_path)
    registration_path = Path(REGISTRATION_PATH if registration_path is None else registration_path)
    if not registration_path.exists():
        raise RegistrationError(f"no registration record at {registration_path}; {_RECORD_ADVICE}")
    if not protocol_path.exists():
        raise RegistrationError(f"no protocol at {protocol_path}; the registered protocol must be in place")
    fields = parse_registration(registration_path)
    registered = datetime.fromisoformat(fields["registered_utc"].replace("Z", "+00:00"))
    now = _utc_now()
    if registered > now:
        raise RegistrationError(f"registered_utc {fields['registered_utc']} is in the future (now "
                                f"{now:%Y-%m-%dT%H:%M:%SZ}); a registration cannot be dated ahead of the clock")
    recorded, supplied = fields["protocol_sha256"], protocol_sha256.lower()
    if recorded != supplied:
        raise RegistrationError("the registered protocol SHA-256 differs from the one supplied")
    if sha256_file(protocol_path) != supplied:
        raise RegistrationError("the protocol file changed since registration")
    from .protocol import ready_to_register          # here, not at the top: es.protocol imports this module
    open_items = ready_to_register(protocol_path.read_text(encoding="utf-8"))
    if open_items:
        n = len(open_items)
        shown = " and ".join(open_items[:2])
        raise RegistrationError(
            f"{protocol_path.name} is not ready to register: {n} item{'s' if n != 1 else ''} still open "
            f"({'the first two: ' if n > 1 else ''}{shown}). Fill docs/sealed/KNOWLEDGE_STATEMENT.md, re-run "
            "es.protocol.main(), and register that PROTOCOL.md (the SHA-256 in the PROTOCOL.sha256 it writes)")
    return fields


def _protocol_hashes(text, name):
    """The SHA-256s (lower case) that `text` prints on lines of the form '<name> SHA-256: <hash>'. The name may be
    wrapped in backticks or asterisks and the line may start a list item."""
    pattern = (rf"^[ \t>*+-]*[`*]*{re.escape(name)}[`*]*[ \t]+SHA-256:[ \t]*[`*]*([0-9a-fA-F]{{64}})(?![0-9a-fA-F])")
    return [h.lower() for h in re.findall(pattern, text, flags=re.M)]


def verify_bound(protocol_path, paths):
    """Each bound file must have its own '<file name> SHA-256: <hash>' line in the protocol, with its hash. A hash
    that appears elsewhere in the protocol, or on another file's line, does not bind this file."""
    text = Path(protocol_path).read_text()
    bound = {}
    for p in paths:
        sha = sha256_file(p)
        name = Path(p).name
        listed = set(_protocol_hashes(text, name))
        if not listed:
            raise RegistrationError(f"{p} is not bound: the protocol has no '{name} SHA-256: <hash>' line")
        if len(listed) > 1:
            raise RegistrationError(f"the protocol prints two different hashes for {name}")
        if sha not in listed:
            raise RegistrationError(f"{p} is not the registered file (SHA-256 {sha[:12]}... is not the one the "
                                    f"protocol prints for {name})")
        bound[str(p)] = sha
    return bound
