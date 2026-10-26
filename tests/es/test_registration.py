import hashlib
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


def test_valid_registration_passes(registered):
    from es.registration import verify_registration
    write, _ = registered
    proto, reg, sha = write()
    assert verify_registration(sha, proto, reg)["osf_doi"] == "10.17605/OSF.IO/ABCDE"


@pytest.mark.parametrize("override", [{"osf_doi": "10.1234/xyz"}, {"osf_doi": None},
                                      {"registered_utc": None}, {"protocol_sha256": "0" * 64}])
def test_bad_records_are_refused(registered, override):
    from es.registration import RegistrationError, verify_registration
    write, _ = registered
    proto, reg, sha = write(reg_fields=override)
    with pytest.raises(RegistrationError):
        verify_registration(sha, proto, reg)


def test_protocol_edited_after_registration_is_refused(registered):
    from es.registration import RegistrationError, verify_registration
    write, _ = registered
    proto, reg, sha = write()
    proto.write_text("protocol v1, quietly edited\n")
    with pytest.raises(RegistrationError, match="changed"):
        verify_registration(sha, proto, reg)


def test_missing_record_is_refused(tmp_path):
    from es.registration import RegistrationError, verify_registration
    with pytest.raises(RegistrationError):
        verify_registration("0" * 64, tmp_path / "PROTOCOL.md", tmp_path / "REGISTRATION.md")


def test_bound_files_must_be_named_in_the_protocol(registered):
    from es.registration import RegistrationError, sha256_file, verify_bound
    write, tmp = registered
    f = tmp / "inputs.json"
    f.write_text('{"eta": 0.01}')
    proto, _, _ = write(f"inputs.json SHA-256: {sha256_file(f)}\n")
    assert verify_bound(proto, [f]) == {str(f): sha256_file(f)}
    f.write_text('{"eta": 0.02}')
    with pytest.raises(RegistrationError, match="not the registered file"):
        verify_bound(proto, [f])


# --- pre-review round (controller Ruling 58, item 7): the record may be written in upper case or with a doi.org link
def test_sha256_is_compared_case_insensitively(registered):
    from es.registration import RegistrationError, verify_registration
    write, _ = registered
    text = "protocol v1\n"
    sha = hashlib.sha256(text.encode()).hexdigest()
    proto, reg, _ = write(text, reg_fields={"protocol_sha256": sha.upper()})          # the record is in upper case
    assert verify_registration(sha, proto, reg)["protocol_sha256"] == sha            # returned in lower case
    assert verify_registration(sha.upper(), proto, reg)["osf_doi"] == "10.17605/OSF.IO/ABCDE"   # supplied in upper case
    with pytest.raises(RegistrationError, match="differs"):
        verify_registration("0" * 64, proto, reg)                                    # a different hash is still refused
    proto.write_text("protocol v1, quietly edited\n")
    with pytest.raises(RegistrationError, match="changed"):
        verify_registration(sha.upper(), proto, reg)                        # and an edited protocol is still caught


def test_an_upper_case_record_with_another_protocols_hash_is_still_refused(registered):
    from es.registration import RegistrationError, verify_registration
    write, _ = registered
    text = "protocol v1\n"
    proto, reg, sha = write(text, reg_fields={"protocol_sha256": "F" * 64})
    with pytest.raises(RegistrationError, match="differs"):
        verify_registration(sha, proto, reg)
    with pytest.raises(RegistrationError, match="changed"):
        verify_registration("f" * 64, proto, reg)       # record and argument agree (in another case), the file does not


@pytest.mark.parametrize("given", ["https://doi.org/10.17605/OSF.IO/ABCDE", "http://doi.org/10.17605/OSF.IO/ABCDE"])
def test_osf_doi_may_be_given_as_a_doi_org_link(registered, given):
    from es.registration import verify_registration
    write, _ = registered
    proto, reg, sha = write(reg_fields={"osf_doi": given})
    assert verify_registration(sha, proto, reg)["osf_doi"] == "10.17605/OSF.IO/ABCDE"       # returned as the bare DOI


# not an OSF DOI | an OSF-looking path elsewhere | another scheme | no DOI | one prefix comes off, not two
@pytest.mark.parametrize("given", ["https://doi.org/10.1234/xyz",
                                   "https://example.org/10.17605/OSF.IO/ABCDE",
                                   "ftp://doi.org/10.17605/OSF.IO/ABCDE",
                                   "https://doi.org/",
                                   "https://doi.org/https://doi.org/10.17605/OSF.IO/ABCDE"])
def test_other_links_are_still_refused(registered, given):
    from es.registration import RegistrationError, verify_registration
    write, _ = registered
    proto, reg, sha = write(reg_fields={"osf_doi": given})
    with pytest.raises(RegistrationError, match="not an OSF DOI"):
        verify_registration(sha, proto, reg)


# --- pre-review round (controller Ruling 60, item E): registered_utc must be an ISO 8601 UTC timestamp
@pytest.mark.parametrize("stamp", ["2026-10-20T15:04:00Z", "2026-10-20T15:04Z", "2026-10-20T15:04:00.250Z",
                                   "2026-10-20T15:04:00+00:00"])
def test_registered_utc_accepts_iso_8601_utc(registered, stamp):
    from es.registration import verify_registration
    write, _ = registered
    proto, reg, sha = write(reg_fields={"registered_utc": stamp})
    assert verify_registration(sha, proto, reg)["registered_utc"] == stamp


@pytest.mark.parametrize("stamp", ["yesterday", "2026-10-20", "2026-10-20T15:04:00", "2026-10-20T15:04:00+02:00",
                                   "2026-10-20T15:04:00-05:00", "2026-13-45T25:61:00Z", "20/10/2026T15:04Z",
                                   "2026-10-20T15:04:00ZZ"])
def test_registered_utc_that_is_not_an_iso_8601_utc_timestamp_is_refused(registered, stamp):
    """In the past, so that the future check cannot be what refuses them."""
    from es.registration import RegistrationError, verify_registration
    write, _ = registered
    proto, reg, sha = write(reg_fields={"registered_utc": stamp})
    with pytest.raises(RegistrationError, match="not an ISO 8601 UTC timestamp"):
        verify_registration(sha, proto, reg)


# =====================================================================================================
# Fix round 1 (controller Ruling 70). Each test below fails on the code that preceded the round.
# =====================================================================================================
NOW = datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc)


# ----------------------------------------------------------------------------- I2: a record from the future
def test_a_registered_utc_in_the_future_is_refused(registered, monkeypatch):
    import es.registration as registration
    from es.registration import RegistrationError, verify_registration
    monkeypatch.setattr(registration, "_utc_now", lambda: NOW)
    write, _ = registered
    for stamp, accepted in (("2026-10-23T11:59:59Z", True), ("2026-10-23T12:00:00Z", True),
                            ("2026-10-23T12:00:01Z", False), ("2026-10-27T15:04:00Z", False)):
        proto, reg, sha = write(reg_fields={"registered_utc": stamp})
        if accepted:
            assert verify_registration(sha, proto, reg)["registered_utc"] == stamp
        else:
            with pytest.raises(RegistrationError, match="in the future"):
                verify_registration(sha, proto, reg)


def test_the_real_clock_refuses_a_record_from_2099(registered):
    from es.registration import RegistrationError, verify_registration
    write, _ = registered
    proto, reg, sha = write(reg_fields={"registered_utc": "2099-01-01T00:00:00Z"})
    with pytest.raises(RegistrationError, match="in the future"):
        verify_registration(sha, proto, reg)


def test_the_clock_is_the_current_utc_time():
    from es.registration import _utc_now
    now = _utc_now()
    assert now.utcoffset() == timedelta(0) and abs((now - datetime.now(timezone.utc)).total_seconds()) < 5


def test_the_real_clock_accepts_yesterday_and_refuses_tomorrow(registered):
    from es.registration import RegistrationError, verify_registration
    write, _ = registered
    stamp = lambda days: (datetime.now(timezone.utc) + timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    proto, reg, sha = write(reg_fields={"registered_utc": stamp(-1)})
    assert verify_registration(sha, proto, reg)["registered_utc"] == stamp(-1)
    proto, reg, sha = write(reg_fields={"registered_utc": stamp(1)})
    with pytest.raises(RegistrationError, match="in the future"):
        verify_registration(sha, proto, reg)


def test_an_absent_protocol_is_refused_with_its_own_message(registered):
    from es.registration import RegistrationError, verify_registration
    write, _ = registered
    proto, reg, sha = write()
    proto.unlink()
    with pytest.raises(RegistrationError, match="no protocol at"):
        verify_registration(sha, proto, reg)


def test_parse_registration_of_an_absent_file_is_a_registration_error(tmp_path):
    from es.registration import RegistrationError, parse_registration
    with pytest.raises(RegistrationError, match="no registration record at"):
        parse_registration(tmp_path / "REGISTRATION.md")


# ----------------------------------------------------------------------------- I2: repeated keys
def test_a_repeated_key_is_refused_whatever_its_value(registered):
    from es.registration import RegistrationError, verify_registration
    write, _ = registered
    proto, reg, sha = write()
    good = reg.read_text()
    for extra in ("osf_doi: 10.17605/OSF.IO/ZZZZZ\n",                        # a leftover example with another value
                  "osf_doi: 10.17605/OSF.IO/ABCDE\n",                        # the same value twice
                  "embargo_until: 2027-02-15\nembargo_until: 2027-03-01\n"):  # any key, not only the required ones
        reg.write_text(good + "\n## example\n" + extra)
        with pytest.raises(RegistrationError, match="repeats"):
            verify_registration(sha, proto, reg)
    reg.write_text(good + "embargo_until: 2027-02-15\n")                      # a different, informational key is fine
    assert verify_registration(sha, proto, reg)["embargo_until"] == "2027-02-15"


# ----------------------------------------------------------------------------- I2: placeholders
@pytest.mark.parametrize("key, value", [("osf_doi", "10.17605/OSF.IO/XXXXX"),
                                        ("osf_doi", "https://doi.org/10.17605/OSF.IO/xxxxx"),
                                        ("osf_doi", "<DOI>"),
                                        ("protocol_sha256", "<64-hex>"),
                                        ("registered_utc", "<utc>"),
                                        ("embargo_until", "<date>"),
                                        ("embargo_until", "XXXXX")])
def test_placeholder_values_are_refused(registered, key, value):
    from es.registration import RegistrationError, verify_registration
    write, _ = registered
    proto, reg, sha = write(reg_fields={key: value})
    with pytest.raises(RegistrationError, match="placeholder"):
        verify_registration(sha, proto, reg)


def test_a_template_line_with_spaces_in_its_value_names_the_key(registered):
    from es.registration import RegistrationError, verify_registration
    write, _ = registered
    proto, reg, sha = write()
    reg.write_text(reg.read_text().replace("osf_doi: 10.17605/OSF.IO/ABCDE", "osf_doi: <paste the OSF DOI>"))
    with pytest.raises(RegistrationError, match="osf_doi") as err:
        verify_registration(sha, proto, reg)
    assert "key: value" in str(err.value)                               # it says what shape the line must have


# ----------------------------------------------------------------------------- C1: the gate reads its own paths
def test_the_gate_reads_the_module_paths_at_call_time(registered, monkeypatch):
    import es.registration as registration
    from es.registration import RegistrationError, verify_registration
    write, tmp = registered
    proto, reg, sha = write()
    assert verify_registration(sha)["osf_doi"] == "10.17605/OSF.IO/ABCDE"       # no paths given: the module's, patched
    nowhere = tmp / "nowhere"
    monkeypatch.setattr(registration, "PROTOCOL_PATH", nowhere / "PROTOCOL.md")
    monkeypatch.setattr(registration, "REGISTRATION_PATH", nowhere / "REGISTRATION.md")
    with pytest.raises(RegistrationError, match="no registration record"):
        verify_registration(sha)                          # a valid record elsewhere is not consulted


def test_default_paths_are_pinned():
    from es import registration
    from es.config import REVISION_ROOT
    d = REVISION_ROOT / "docs" / "sealed"
    assert (registration.SEALED_DIR, registration.PROTOCOL_PATH, registration.REGISTRATION_PATH,
            registration.CUTOFFS_PATH, registration.INPUTS_PATH) == (
        d, d / "PROTOCOL.md", d / "REGISTRATION.md", d / "pre_pull_cutoffs.csv", d / "protocol_inputs.json")


# ----------------------------------------------------------------------------- the addendum: names paper_macros imports
def test_public_names_and_the_private_aliases_that_es_paper_macros_imports():
    from es import registration as r
    assert r._DOI_LINK is r.DOI_LINK_RE and r._OSF_DOI is r.OSF_DOI_RE and r._is_iso_utc is r.is_iso_utc
    assert r._DOI_LINK.sub("", "https://doi.org/10.17605/OSF.IO/ABCDE") == "10.17605/OSF.IO/ABCDE"
    assert r._OSF_DOI.fullmatch("10.17605/OSF.IO/ABCDE") and not r._OSF_DOI.fullmatch("10.1234/xyz")
    assert r._is_iso_utc("2026-10-27T15:04:00Z") and not r._is_iso_utc("2026-10-27")
    assert r.is_iso_utc("2099-01-01T00:00:00Z")  # format only: the clock is verify_registration's business


def test_parse_registration_checks_the_format_and_not_the_clock(registered):
    from es.registration import RegistrationError, parse_registration, verify_registration
    write, _ = registered
    proto, reg, sha = write(reg_fields={"registered_utc": "2099-01-01T00:00:00Z", "protocol_sha256": sha_upper()})
    fields = parse_registration()                                  # the module's record: format, repeats, placeholders
    assert fields["osf_doi"] == "10.17605/OSF.IO/ABCDE" and fields["protocol_sha256"] == sha_upper().lower()
    assert fields["registered_utc"] == "2099-01-01T00:00:00Z"
    with pytest.raises(RegistrationError, match="in the future"):
        verify_registration(sha_upper().lower(), proto, reg)
    reg.write_text(reg.read_text() + "osf_doi: 10.17605/OSF.IO/ZZZZZ\n")
    with pytest.raises(RegistrationError, match="repeats"):
        parse_registration(reg)


def sha_upper():
    return hashlib.sha256(b"protocol v1\n").hexdigest().upper()


# ----------------------------------------------------------------------------- M7: bound files by name and hash
def _bound(tmp):
    from es.registration import sha256_file
    inputs, cutoffs = tmp / "protocol_inputs.json", tmp / "pre_pull_cutoffs.csv"
    inputs.write_text('{"eta": 0.01}')
    cutoffs.write_text("tau\n0.5\n")
    return inputs, cutoffs, sha256_file(inputs), sha256_file(cutoffs)


def test_a_bound_file_is_matched_by_its_name_not_by_a_hash_anywhere_in_the_protocol(registered):
    from es.registration import RegistrationError, verify_bound
    write, tmp = registered
    inputs, cutoffs, h_inputs, h_cutoffs = _bound(tmp)
    proto, _, _ = write(f"protocol_inputs.json SHA-256: {h_inputs}\npre_pull_cutoffs.csv SHA-256: {h_cutoffs}\n")
    assert verify_bound(proto, [inputs, cutoffs]) == {str(inputs): h_inputs, str(cutoffs): h_cutoffs}
    cutoffs.write_text(inputs.read_text())                  # the cutoffs file now holds the inputs file's bytes
    with pytest.raises(RegistrationError, match="not the registered file"):
        verify_bound(proto, [cutoffs])                      # its hash IS in the protocol, but on the other file's line


def test_a_bound_file_with_no_line_of_its_own_is_refused(registered):
    from es.registration import RegistrationError, verify_bound
    write, tmp = registered
    inputs, cutoffs, h_inputs, h_cutoffs = _bound(tmp)
    proto, _, _ = write(f"protocol_inputs.json SHA-256: {h_inputs}\nthe cutoffs hash is {h_cutoffs}\n")
    with pytest.raises(RegistrationError, match="no 'pre_pull_cutoffs.csv SHA-256"):
        verify_bound(proto, [cutoffs])                      # the hash appears, on no line that names the file


def test_a_protocol_that_gives_two_hashes_for_one_file_is_refused(registered):
    from es.registration import RegistrationError, verify_bound
    write, tmp = registered
    inputs, cutoffs, h_inputs, h_cutoffs = _bound(tmp)
    proto, _, _ = write(f"protocol_inputs.json SHA-256: {h_inputs}\nprotocol_inputs.json SHA-256: {h_cutoffs}\n")
    with pytest.raises(RegistrationError, match="two different hashes"):
        verify_bound(proto, [inputs])


def test_markdown_decoration_and_upper_case_hashes_are_accepted_on_the_named_line(registered):
    from es.registration import verify_bound
    write, tmp = registered
    inputs, cutoffs, h_inputs, h_cutoffs = _bound(tmp)
    proto, _, _ = write(f"- `protocol_inputs.json` SHA-256: `{h_inputs.upper()}`\n"
                        f"* **pre_pull_cutoffs.csv** SHA-256: {h_cutoffs}\n")
    assert verify_bound(proto, [inputs, cutoffs]) == {str(inputs): h_inputs, str(cutoffs): h_cutoffs}


# =====================================================================================================
# Addendum 2 to fix round 1: an embargoed registration has no DOI yet, so the record names its OSF address instead
# (or as well). osf_doi OR osf_registration: at least one, each valid when present.
# =====================================================================================================
DOI = "10.17605/OSF.IO/ABCDE"
URL = "https://osf.io/abcde/"


def test_a_record_with_a_doi_only_has_no_osf_address(registered):
    from es.registration import registration_identifier, registration_identifiers, verify_registration
    write, _ = registered
    proto, reg, sha = write()
    record = verify_registration(sha, proto, reg)
    assert (record["osf_doi"], record["osf_registration"]) == (DOI, None)
    assert registration_identifier(record) == DOI and registration_identifiers(record) == (DOI,)


@pytest.mark.parametrize("given", ["abcde", "ABCDE", "ab1c2d", "https://osf.io/abcde/", "https://osf.io/abcde",
                                   "http://osf.io/abcde/", "HTTPS://OSF.IO/ABCDE/"])
def test_a_record_with_an_osf_registration_and_no_doi_opens_the_gate(registered, given):
    from es.registration import registration_identifier, registration_identifiers, verify_registration
    write, _ = registered
    proto, reg, sha = write(reg_fields={"osf_doi": None, "osf_registration": given})
    record = verify_registration(sha, proto, reg)
    expected = f"https://osf.io/{given.lower().rstrip('/').rsplit('/', 1)[-1]}/"
    assert record["osf_doi"] is None and record["osf_registration"] == expected         # lower-case id, whole URL
    assert registration_identifier(record) == expected and registration_identifiers(record) == (expected,)
    assert verify_registration(sha)["osf_registration"] == expected                 # and through the module's own paths


def test_a_record_with_both_identifiers_returns_both_and_prefers_the_doi(registered):
    from es.registration import registration_identifier, registration_identifiers, verify_registration
    write, _ = registered
    proto, reg, sha = write(reg_fields={"osf_registration": URL})
    record = verify_registration(sha, proto, reg)
    assert (record["osf_doi"], record["osf_registration"]) == (DOI, URL)
    assert registration_identifier(record) == DOI and registration_identifiers(record) == (DOI, URL)


def test_a_record_with_neither_identifier_is_refused(registered):
    from es.registration import RegistrationError, parse_registration, verify_registration
    write, _ = registered
    proto, reg, sha = write(reg_fields={"osf_doi": None})
    with pytest.raises(RegistrationError, match="osf_doi or osf_registration"):
        verify_registration(sha, proto, reg)
    with pytest.raises(RegistrationError, match="osf_doi or osf_registration"):
        parse_registration(reg)


def test_registration_identifier_of_a_record_with_neither_raises():
    from es.registration import RegistrationError, registration_identifier, registration_identifiers
    for record in ({}, {"osf_doi": None, "osf_registration": None}, {"osf_doi": "", "osf_registration": ""}):
        assert registration_identifiers(record) == ()
        with pytest.raises(RegistrationError, match="names no registration"):
            registration_identifier(record)
    assert registration_identifier({"osf_doi": None, "osf_registration": URL}) == URL


@pytest.mark.parametrize("given", ["XXXXX", "xxxxx", "https://osf.io/XXXXX/", "https://osf.io/xxxxx", "<id>",
                                   "https://osf.io/<id>/"])
def test_a_placeholder_in_osf_registration_is_refused(registered, given):
    from es.registration import RegistrationError, verify_registration
    write, _ = registered
    for fields in ({"osf_doi": None, "osf_registration": given}, {"osf_registration": given}):  # alone, beside a DOI
        proto, reg, sha = write(reg_fields=fields)
        with pytest.raises(RegistrationError, match="placeholder"):
            verify_registration(sha, proto, reg)


def test_a_placeholder_in_the_doi_is_refused_even_beside_a_valid_osf_registration(registered):
    from es.registration import RegistrationError, verify_registration
    write, _ = registered
    for doi in ("10.17605/OSF.IO/XXXXX", "<DOI>"):
        proto, reg, sha = write(reg_fields={"osf_doi": doi, "osf_registration": URL})
        with pytest.raises(RegistrationError, match="placeholder"):
            verify_registration(sha, proto, reg)


@pytest.mark.parametrize("given", ["osf.io/abcde", "https://example.org/abcde/", "https://osf.io/abcde/files/",
                                   "ftp://osf.io/abcde/", "https://osf.io/", "https://osf.io/abcd/", "abcd", "abcdefg",
                                   "abc_e", "https://osf.io/abcdefg/", "https://www.osf.io/abcde/"])
def test_other_forms_of_osf_registration_are_refused(registered, given):
    from es.registration import RegistrationError, normalise_osf_registration, verify_registration
    write, _ = registered
    assert normalise_osf_registration(given) is None
    proto, reg, sha = write(reg_fields={"osf_doi": None, "osf_registration": given})
    with pytest.raises(RegistrationError, match="not an OSF registration"):
        verify_registration(sha, proto, reg)


def test_each_identifier_that_is_present_must_be_valid_whatever_the_other_one_is(registered):
    from es.registration import RegistrationError, verify_registration
    write, _ = registered
    proto, reg, sha = write(reg_fields={"osf_doi": DOI, "osf_registration": "not-an-id!"})
    with pytest.raises(RegistrationError, match="not an OSF registration"):
        verify_registration(sha, proto, reg)
    proto, reg, sha = write(reg_fields={"osf_doi": "10.1234/xyz", "osf_registration": URL})
    with pytest.raises(RegistrationError, match="not an OSF DOI"):
        verify_registration(sha, proto, reg)


def test_normalise_osf_registration():
    from es.registration import normalise_osf_registration as n
    assert n("abcde") == n("ABCDE") == n("https://osf.io/abcde") == n("http://osf.io/ABCDE/") == URL
    assert n("a1b2c3") == "https://osf.io/a1b2c3/" and n("") is None and n("https://osf.io/abcde//") is None


def test_the_other_checks_do_not_depend_on_which_identifier_the_record_has(registered, monkeypatch):
    import es.registration as registration
    from es.registration import RegistrationError, verify_registration
    write, _ = registered
    only_url = {"osf_doi": None, "osf_registration": URL}
    proto, reg, sha = write(reg_fields=only_url)
    with pytest.raises(RegistrationError, match="differs"):
        verify_registration("0" * 64, proto, reg)
    proto.write_text("protocol v1, quietly edited\n")
    with pytest.raises(RegistrationError, match="changed"):
        verify_registration(sha, proto, reg)
    proto, reg, sha = write(reg_fields={**only_url, "registered_utc": "2099-01-01T00:00:00Z"})
    with pytest.raises(RegistrationError, match="in the future"):
        verify_registration(sha, proto, reg)
    monkeypatch.setattr(registration, "_utc_now", lambda: NOW)
    proto, reg, sha = write(reg_fields={**only_url, "registered_utc": "2026-10-23T12:00:01Z"})
    with pytest.raises(RegistrationError, match="in the future"):
        verify_registration(sha, proto, reg)


def test_a_repeated_osf_registration_is_refused_even_when_the_second_line_is_a_template_line(registered):
    from es.registration import RegistrationError, verify_registration
    write, _ = registered
    proto, reg, sha = write(reg_fields={"osf_registration": URL})
    good = reg.read_text()
    for extra in ("osf_registration: https://osf.io/zzzzz/\n", f"osf_registration: {URL}\n",
                  "osf_registration: <paste the OSF address>\n", "osf_registration:\n"):
        reg.write_text(good + "\n## example\n" + extra)
        with pytest.raises(RegistrationError, match="repeats"):
            verify_registration(sha, proto, reg)


@pytest.mark.parametrize("line", ["osf_registration: <paste the OSF address>",
                                  "osf_registration: https://osf.io/abcde/ # mine",
                                  "osf_registration:", "osf_registration:   ",
                                  "osf_doi: <paste the DOI or delete this line>", "osf_doi:"])
def test_a_line_that_names_an_identifier_must_hold_exactly_one_token(registered, line):
    """The identifier that is not known has to be left out, not left as a template line or a blank."""
    from es.registration import RegistrationError, verify_registration
    write, _ = registered
    key = line.split(":")[0]
    other = {"osf_doi": {"osf_doi": None, "osf_registration": URL}, "osf_registration": {}}[key]  # a valid one beside
    proto, reg, sha = write(reg_fields=other)
    reg.write_text(reg.read_text() + line + "\n")
    with pytest.raises(RegistrationError, match=key) as err:
        verify_registration(sha, proto, reg)
    assert "key: value" in str(err.value)


def test_a_value_on_the_line_after_its_key_is_refused(registered):
    """The old parser let \\s* run over the newline and read the next line as the value."""
    from es.registration import RegistrationError, verify_registration
    write, _ = registered
    proto, reg, sha = write()
    reg.write_text(reg.read_text().replace(f"protocol_sha256: {sha}", f"protocol_sha256:\n{sha}"))
    with pytest.raises(RegistrationError, match="protocol_sha256"):
        verify_registration(sha, proto, reg)


def test_a_record_with_windows_line_endings_is_read(registered):
    from es.registration import verify_registration
    write, _ = registered
    proto, reg, sha = write(reg_fields={"osf_registration": URL})
    reg.write_bytes(reg.read_text().replace("\n", "\r\n").encode())
    record = verify_registration(sha, proto, reg)
    assert (record["osf_doi"], record["osf_registration"]) == (DOI, URL)


def test_the_public_names_for_the_osf_address():
    from es import registration as r
    assert r.OSF_URL_RE.fullmatch("https://osf.io/abcde/") and not r.OSF_URL_RE.fullmatch("https://osf.io/abc/")
    assert r.OSF_ID_RE.fullmatch("abcde") and r.OSF_ID_RE.fullmatch("abcdef") and not r.OSF_ID_RE.fullmatch("abcdefg")
    assert r._OSF_DOI is r.OSF_DOI_RE and r._DOI_LINK is r.DOI_LINK_RE         # the names es.paper_macros imports


# =====================================================================================================
# Fix round 2 (controller Ruling 106): the DOI and the OSF address name one registration
# =====================================================================================================
def test_a_doi_and_an_osf_address_that_name_different_registrations_are_refused(registered):
    from es.registration import RegistrationError, parse_registration, verify_registration
    write, _ = registered
    proto, reg, sha = write(reg_fields={"osf_registration": "https://osf.io/zzzzz/"})       # the DOI is ABCDE
    with pytest.raises(RegistrationError, match=r"name different OSF registrations \(abcde and zzzzz\)") as err:
        verify_registration(sha, proto, reg)
    assert DOI in str(err.value) and "https://osf.io/zzzzz/" in str(err.value)
    with pytest.raises(RegistrationError, match="name different OSF registrations"):
        parse_registration(reg)


@pytest.mark.parametrize("doi, address", [("10.17605/OSF.IO/ABCDE", "abcde"), ("10.17605/osf.io/abcde", "ABCDE"),
                                          ("https://doi.org/10.17605/OSF.IO/ABCDE", "https://osf.io/abcde/"),
                                          ("10.17605/OSF.IO/AB1C2D", "HTTPS://OSF.IO/ab1c2d")])
def test_a_doi_and_an_osf_address_of_the_same_id_agree_whatever_their_case_and_form(registered, doi, address):
    from es.registration import registration_id, verify_registration
    write, _ = registered
    proto, reg, sha = write(reg_fields={"osf_doi": doi, "osf_registration": address})
    record = verify_registration(sha, proto, reg)
    assert registration_id(record) == address.rstrip("/").rsplit("/", 1)[-1].lower()


def test_registration_ids_and_the_one_id_of_a_record(registered):
    from es.registration import RegistrationError, registration_id, registration_ids
    assert registration_ids({"osf_doi": DOI}) == {"abcde"} and registration_ids({"osf_registration": URL}) == {"abcde"}
    assert registration_ids({"osf_doi": DOI, "osf_registration": URL}) == {"abcde"}
    assert registration_ids({"osf_doi": "https://doi.org/" + DOI.lower()}) == {"abcde"}
    assert registration_ids({"osf_doi": None, "osf_registration": None}) == frozenset() == registration_ids({})
    assert registration_ids({"osf_doi": "10.1234/xyz", "osf_registration": "not an id"}) == frozenset()   # not OSF ids
    assert registration_ids({"osf_doi": DOI, "osf_registration": "https://osf.io/zzzzz/"}) == {"abcde", "zzzzz"}
    assert registration_id({"osf_doi": None, "osf_registration": "ABCDE"}) == "abcde"
    for record in ({}, {"osf_doi": DOI, "osf_registration": "https://osf.io/zzzzz/"}):
        with pytest.raises(RegistrationError, match=r"names \d registrations"):
            registration_id(record)


def test_the_id_taken_from_a_doi_or_from_an_address_is_in_lower_case():
    from es import registration as r
    assert r._osf_id_of_doi("10.17605/OSF.IO/AB1C2D") == "ab1c2d"
    assert r._osf_id_of_address("https://osf.io/AB1C2D/") == "ab1c2d" == r._osf_id_of_address("https://osf.io/ab1c2d")


def test_the_docstring_example_of_registered_utc_is_a_clearly_past_time():
    import re

    from es import registration
    example = re.search(r"registered_utc: (\S+)", registration.__doc__).group(1)
    assert registration.is_iso_utc(example) and example != "2026-10-27T15:04:00Z"   # that one is accepted after 27 Oct
    assert datetime.fromisoformat(example.replace("Z", "+00:00")) < datetime(2026, 10, 1, tzinfo=timezone.utc)


# =====================================================================================================
# Ruling 122, C2-1: a protocol that is not ready to register opens nothing. verify_registration, which pull_sealed and
# run_sealed both call before they spend anything, refuses a PROTOCOL.md whose knowledge statement is still unfilled.
# =====================================================================================================
TEMPLATE = Path(__file__).resolve().parents[3] / "docs" / "sealed" / "KNOWLEDGE_STATEMENT_TEMPLATE.md"
#: A statement in the shape of the template: guidance in italics, an item-1 table with boxes, three markers.
SHORT_STATEMENT = (
    "## 1. What I know\n\n*This covers July to December. Tick one option for each item.*\n\n"
    "| Item | A. Nothing | B. General | C. Specific |\n|---|---|---|---|\n| Prices | ☐ | ☐ | ☐ |\n"
    "| Reports | ☐ | ☐ | ☐ |\n\nDetails: [AUTHOR: one or two sentences, or \"none\"]\n\n"
    "## 3. Statement\n\n[AUTHOR: confirm or edit] \"I have not examined prices after 30 June 2026.\"\n\n"
    "Name and date: [AUTHOR: Erce Erez, date]\n")


def _statements():
    out = {"a short statement": SHORT_STATEMENT}
    if TEMPLATE.is_file():
        out["the real template"] = TEMPLATE.read_text()
    return out


def _filled(statement):
    """The statement as the author leaves it: every marker replaced, one box ticked in each row of the item-1 table, the
    paragraphs that are one italic span (the guidance) deleted."""
    text = re.sub(r"\[AUTHOR:[^\]]*\]", "filled in by the author", statement)
    rows = [line.replace("☐", "☒", 1) if line.startswith("|") and "☐" in line else line
            for line in text.splitlines()]
    italic = re.compile(r"\*(?![\s*])[^*]+(?<![\s*])\*")
    kept = [p for p in re.split(r"\n\s*\n", "\n".join(rows)) if not italic.fullmatch(p.strip())]
    return "\n\n".join(kept) + "\n"


def _protocol_with(statement):
    """A protocol text that holds `statement` between the two comment lines that es.protocol writes around it."""
    from es.protocol import KNOWLEDGE_BEGIN, KNOWLEDGE_END
    return f"# Pre-registration protocol\n\nprotocol v1\n\n{KNOWLEDGE_BEGIN}\n{statement}\n{KNOWLEDGE_END}\n"


@pytest.mark.parametrize("which", ["a short statement", "the real template"])
def test_c2_a_protocol_whose_statement_is_not_filled_in_is_refused_by_the_gate(registered, which):
    from es.protocol import ready_to_register
    from es.registration import RegistrationError, verify_registration
    statements = _statements()
    if which not in statements:
        pytest.skip("docs/sealed/KNOWLEDGE_STATEMENT_TEMPLATE.md is not there")
    text = _protocol_with(statements[which])
    items = ready_to_register(text)
    assert len(items) >= 4                                         # markers, empty rows, guidance
    write, _ = registered
    proto, reg, sha = write(text)                                  # a valid record that names this very protocol
    with pytest.raises(RegistrationError) as err:
        verify_registration(sha, proto, reg)
    message = str(err.value)
    assert f"PROTOCOL.md is not ready to register: {len(items)} items still open (the first two: {items[0]} and " \
           f"{items[1]}). " in message
    assert items[2] not in message                                 # the first two only
    assert message.endswith("Fill docs/sealed/KNOWLEDGE_STATEMENT.md, re-run es.protocol.main(), and register that "
                            "PROTOCOL.md (the SHA-256 in the PROTOCOL.sha256 it writes)")


def test_c2_one_open_item_is_named_alone(registered):
    from es.registration import RegistrationError, verify_registration
    write, _ = registered
    proto, reg, sha = write("protocol v1\n\nName and date: [AUTHOR: Erce Erez, date]\n")
    with pytest.raises(RegistrationError, match=r"is not ready to register: 1 item still open \(\[AUTHOR: Erce Erez, "
                                                r"date\]\)\. Fill docs/sealed/KNOWLEDGE_STATEMENT\.md"):
        verify_registration(sha, proto, reg)


@pytest.mark.parametrize("which", ["a short statement", "the real template"])
@pytest.mark.parametrize("defect", ["a marker", "a row with no box ticked", "a row with two boxes ticked",
                                    "a guidance paragraph"])
def test_c2_each_kind_of_open_item_is_refused_alone_and_a_filled_statement_passes(registered, which, defect):
    from es.registration import RegistrationError, verify_registration
    statements = _statements()
    if which not in statements:
        pytest.skip("docs/sealed/KNOWLEDGE_STATEMENT_TEMPLATE.md is not there")
    ready = _filled(statements[which])
    write, _ = registered
    proto, reg, sha = write(_protocol_with(ready))
    assert verify_registration(sha, proto, reg)["osf_doi"] == "10.17605/OSF.IO/ABCDE"          # a ready protocol passes
    broken = {"a marker": lambda t: t.replace("filled in by the author", "[AUTHOR: sign here]", 1),
              "a row with no box ticked": lambda t: t.replace("☒", "☐", 1),
              "a row with two boxes ticked": lambda t: re.sub(r"(☒)( \| )☐", r"\1\2☒", t, count=1),
              "a guidance paragraph": lambda t: t + "\n*Delete this paragraph before you register.*\n"}[defect]
    proto, reg, sha = write(_protocol_with(broken(ready)))
    with pytest.raises(RegistrationError, match=r"is not ready to register: 1 item still open"):
        verify_registration(sha, proto, reg)


@pytest.mark.parametrize("text", ["protocol v1\n", "protocol v2, edited after the first attempt\n", "",
                                  "# Pre-registration protocol\n\nPart A.\n",
                                  "protocol\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\nsome *emphasis* in a sentence.\n"])
def test_c2_a_protocol_with_no_marker_and_no_statement_passes(registered, text):
    from es.registration import verify_registration
    write, _ = registered
    proto, reg, sha = write(text)
    assert verify_registration(sha, proto, reg)["registered_utc"] == "2026-10-20T15:04:00Z"


def test_c2_the_other_refusals_come_first(registered):
    """An unready protocol that is also the wrong one, edited, missing, dated ahead or without a good record is refused
    for that reason: the readiness check comes last."""
    from es.registration import RegistrationError, verify_registration
    write, tmp = registered
    unready = "protocol v1\n\n[AUTHOR: x]\n"
    proto, reg, sha = write(unready)
    with pytest.raises(RegistrationError, match="differs"):
        verify_registration("0" * 64, proto, reg)
    proto.write_text("protocol v1, edited\n\n[AUTHOR: x]\n")
    with pytest.raises(RegistrationError, match="changed since registration"):
        verify_registration(sha, proto, reg)
    proto, reg, sha = write(unready, {"registered_utc": "2999-01-01T00:00:00Z"})
    with pytest.raises(RegistrationError, match="in the future"):
        verify_registration(sha, proto, reg)
    proto, reg, sha = write(unready, {"osf_doi": None})                      # a record with no OSF identifier at all
    with pytest.raises(RegistrationError, match="registration record lacks"):
        verify_registration(sha, proto, reg)
    proto, reg, sha = write(unready)
    proto.unlink()
    with pytest.raises(RegistrationError, match="no protocol at"):
        verify_registration(sha, proto, reg)
    proto, reg, sha = write(unready)
    reg.unlink()
    with pytest.raises(RegistrationError, match="no registration record"):
        verify_registration(sha, proto, reg)


def test_c2_run_sealed_refuses_a_protocol_that_is_not_ready_before_its_lock(registered, monkeypatch):
    from es import sealed_run
    from es.protocol import ready_to_register
    from es.registration import RegistrationError
    write, tmp = registered
    text = _protocol_with(SHORT_STATEMENT)
    proto, reg, sha = write(text)
    monkeypatch.setattr(sealed_run, "OUT_DIR", tmp / "outputs" / "es")
    with pytest.raises(RegistrationError, match=f"is not ready to register: {len(ready_to_register(text))} items"):
        sealed_run.run_sealed(sealed_run.default_config("A"), sha)
    assert not (tmp / "outputs").exists()                                # no output folder
    assert sorted(p.name for p in (tmp / "docs_sealed").iterdir()) == []   # no claim file, no RUNS.json
    proto, reg, sha = write(_protocol_with(_filled(SHORT_STATEMENT)))     # filled in, the protocol passes the gate
    with pytest.raises(Exception) as err:                           # and the run goes on to its next check, the
        sealed_run.run_sealed(sealed_run.default_config("A"), sha)  # bound files, which this protocol does not name
    assert "not ready to register" not in str(err.value)
    assert not (tmp / "outputs").exists() and sorted(p.name for p in (tmp / "docs_sealed").iterdir()) == []


# =====================================================================================================
# Ruling 140 (26 Oct 2026): in place of an OSF registration the record may be a public release of the code repository
# (public_record) or the author's local record (local_record). Exactly one kind; the gate reads no network, so it checks
# the form of the record, the hash, the clock and the readiness of the protocol, as it always did.
# =====================================================================================================
LOCAL = "author-statement-2026-10-26"
RELEASE = "https://github.com/ee2625/dart-spread/releases/tag/v1.0"
RELEASE_ID = "github:ee2625/dart-spread/releases/tag/v1.0"


def _local(**over):
    """The fixture's record with its DOI taken out and a local record put in."""
    return {"osf_doi": None, "local_record": LOCAL, **over}


def _public(**over):
    return {"osf_doi": None, "public_record": RELEASE, **over}


# ----------------------------------------------------------------------------- the local record
def test_a_local_record_opens_the_gate_and_has_its_own_identity(registered):
    from es.registration import (registration_id, registration_identifier, registration_identifiers, registration_ids,
                                 registration_kind, verify_registration)
    write, _ = registered
    proto, reg, sha = write(reg_fields=_local())
    record = verify_registration(sha, proto, reg)
    assert record["local_record"] == LOCAL
    assert (record["osf_doi"], record["osf_registration"], record["public_record"]) == (None, None, None)
    assert registration_id(record) == "local:author-statement-2026-10-26"
    assert registration_ids(record) == {"local:author-statement-2026-10-26"}
    assert registration_identifier(record) == LOCAL and registration_identifiers(record) == (LOCAL,)
    assert registration_kind(record) == "local"
    assert verify_registration(sha)["local_record"] == LOCAL                    # and through the module's own paths


@pytest.mark.parametrize("given", ["author-statement-26-10-2026", "author-statement-2026-10-26T01:40",
                                   "author-statement-2026-1-5", "author-statement-20261026", "author-statement-2026-10-26-",
                                   "author-statement-2026-10-26x", "statement-2026-10-26", "Author-Statement-2026-10-26",
                                   "author_statement-2026-10-26", "author-statement-", "author-statement-2026-13-01",
                                   "author-statement-2026-02-30", "author-statement-\uff12\uff10\uff12\uff16-10-26",
                                   "2026-10-26", "yes"])
def test_a_local_record_that_is_not_author_statement_and_a_real_date_is_refused(registered, given):
    from es.registration import RegistrationError, is_local_record, verify_registration
    write, _ = registered
    assert not is_local_record(given)
    proto, reg, sha = write(reg_fields=_local(local_record=given))
    with pytest.raises(RegistrationError, match="not a local record"):
        verify_registration(sha, proto, reg)


def test_the_token_of_a_local_record():
    from es.registration import is_local_record
    assert is_local_record("author-statement-2026-10-26") and is_local_record("author-statement-2024-02-29")
    assert not is_local_record("author-statement-2025-02-29")                         # not a leap year


@pytest.mark.parametrize("given", ["XXXXX", "author-statement-XXXXX", "author-statement-xxxxx", "<local_record>",
                                   "author-statement-<date>"])
def test_a_placeholder_in_a_local_record_is_refused(registered, given):
    from es.registration import RegistrationError, verify_registration
    write, _ = registered
    proto, reg, sha = write(reg_fields=_local(local_record=given))
    with pytest.raises(RegistrationError, match="placeholder"):
        verify_registration(sha, proto, reg)


def test_a_placeholder_in_any_value_is_refused_whatever_the_kind(registered):
    from es.registration import RegistrationError, verify_registration
    write, _ = registered
    for fields in (_local(registered_utc="<utc>"), _local(protocol_sha256="<64-hex>"), _local(embargo_until="<date>"),
                   _public(registered_utc="<utc>"), _public(embargo_until="XXXXX"), _public(protocol_sha256="XXXXX")):
        proto, reg, sha = write(reg_fields=fields)
        with pytest.raises(RegistrationError, match="placeholder"):
            verify_registration(sha, proto, reg)


@pytest.mark.parametrize("stamp, accepted", [("2026-10-23T11:59:59Z", True), ("2026-10-23T12:00:00Z", True),
                                             ("2026-10-23T12:00:01Z", False), ("2099-01-01T00:00:00Z", False)])
@pytest.mark.parametrize("kind", ["local", "public"])
def test_a_time_in_the_future_is_refused_for_the_new_kinds_too(registered, monkeypatch, kind, stamp, accepted):
    import es.registration as registration
    from es.registration import RegistrationError, verify_registration
    monkeypatch.setattr(registration, "_utc_now", lambda: NOW)
    write, _ = registered
    proto, reg, sha = write(reg_fields={"local": _local, "public": _public}[kind](registered_utc=stamp))
    if accepted:
        assert verify_registration(sha, proto, reg)["registered_utc"] == stamp
    else:
        with pytest.raises(RegistrationError, match="in the future"):
            verify_registration(sha, proto, reg)


@pytest.mark.parametrize("kind", ["local", "public"])
def test_a_wrong_hash_or_an_edited_protocol_is_refused_for_the_new_kinds_too(registered, kind):
    from es.registration import RegistrationError, verify_registration
    write, _ = registered
    make = {"local": _local, "public": _public}[kind]
    proto, reg, sha = write(reg_fields=make(protocol_sha256="0" * 64))                  # the record names another protocol
    with pytest.raises(RegistrationError, match="differs"):
        verify_registration(sha, proto, reg)
    proto, reg, sha = write(reg_fields=make())
    with pytest.raises(RegistrationError, match="differs"):
        verify_registration("0" * 64, proto, reg)                                       # or another one is supplied
    proto.write_text("protocol v1, quietly edited\n")
    with pytest.raises(RegistrationError, match="changed"):
        verify_registration(sha, proto, reg)                                            # or the file changed
    proto, reg, sha = write(reg_fields=make(protocol_sha256=sha_upper()))
    assert verify_registration(sha_upper().lower(), proto, reg)["protocol_sha256"] == sha_upper().lower()   # case is free


@pytest.mark.parametrize("kind", ["local", "public"])
def test_the_protocol_must_still_be_ready_to_register_for_the_new_kinds_too(registered, kind):
    from es.registration import RegistrationError, verify_registration
    write, _ = registered
    proto, reg, sha = write("protocol v1\n\n[AUTHOR: x]\n", {"local": _local, "public": _public}[kind]())
    with pytest.raises(RegistrationError, match="is not ready to register: 1 item still open"):
        verify_registration(sha, proto, reg)


# ----------------------------------------------------------------------------- the public release record
def test_a_public_record_opens_the_gate_and_has_its_own_identity(registered):
    from es.registration import (registration_id, registration_identifier, registration_identifiers, registration_ids,
                                 registration_kind, verify_registration)
    write, _ = registered
    proto, reg, sha = write(reg_fields=_public())
    record = verify_registration(sha, proto, reg)
    assert record["public_record"] == RELEASE
    assert (record["osf_doi"], record["osf_registration"], record["local_record"]) == (None, None, None)
    assert registration_id(record) == RELEASE_ID and registration_ids(record) == {RELEASE_ID}
    assert registration_identifier(record) == RELEASE and registration_identifiers(record) == (RELEASE,)
    assert registration_kind(record) == "public_release"
    assert verify_registration(sha)["public_record"] == RELEASE


@pytest.mark.parametrize("given", [
    "http://github.com/o/r/releases/tag/v1",                          # not https
    "ftp://github.com/o/r/releases/tag/v1", "github.com/o/r/releases/tag/v1", "//github.com/o/r/releases/tag/v1",
    "https://gitlab.com/o/r/releases/tag/v1",                         # another host
    "https://www.github.com/o/r/releases/tag/v1", "https://github.com.example.org/o/r/releases/tag/v1",
    "https://example.org/github.com/o/r/releases/tag/v1", "https://github.com:443/o/r/releases/tag/v1",
    "https://user@github.com/o/r/releases/tag/v1", "https://api.github.com/o/r/releases/tag/v1",
    "https://github.com/o/r/releases/latest",                         # not a tagged release
    "https://github.com/o/r/releases/tag/", "https://github.com/o/r/releases", "https://github.com/o/r/releases/",
    "https://github.com/o/r/releases/download/v1/x.tar.gz", "https://github.com/o/r/tree/x", "https://github.com/o/r/tree/v1",
    "https://github.com/o/r/commit/abc123", "https://github.com/o/r", "https://github.com/o/releases/tag/v1",
    "https://github.com/o/r/Releases/tag/v1", "https://github.com/o/r/releases/TAG/v1",
    "https://github.com/o/r/releases/tag/v1/",                        # a trailing slash
    "https://github.com/o/r/releases/tag/v1?x=1", "https://github.com/o/r/releases/tag/v1#notes",   # query, fragment
    "https://github.com/o/r/releases/tag/v1/assets", "https://github.com/o/r/releases/tag/a/b",   # a path after the tag
    "https://github.com//r/releases/tag/v1", "https://github.com/o//releases/tag/v1",
    "https://github.com/o/r/releases/tag/v1%20x", "https://github.com/o/r/releases/tag/v1\u00e9",   # not name characters
    "not-a-url", "v1.0",
])
def test_other_forms_of_a_public_record_are_refused(registered, given):
    from es.registration import RegistrationError, normalise_public_record, verify_registration
    write, _ = registered
    assert normalise_public_record(given) is None
    proto, reg, sha = write(reg_fields=_public(public_record=given))
    with pytest.raises(RegistrationError, match="not the address of a tagged release"):
        verify_registration(sha, proto, reg)


def test_the_host_the_owner_and_the_repository_of_a_public_record_are_lower_cased_and_the_tag_is_kept(registered):
    from es.registration import (normalise_public_record, registration_id, registration_ids, registration_identifier,
                                 verify_registration)
    write, _ = registered
    given = "HTTPS://GitHub.COM/EE2625/Dart-Spread/releases/tag/V1.0-RC1"
    proto, reg, sha = write(reg_fields=_public(public_record=given))
    record = verify_registration(sha, proto, reg)
    assert record["public_record"] == "https://github.com/ee2625/dart-spread/releases/tag/V1.0-RC1"      # normalised
    assert registration_identifier(record) == record["public_record"]
    assert registration_id(record) == "github:ee2625/dart-spread/releases/tag/V1.0-RC1"               # the tag as written
    assert normalise_public_record(given) == record["public_record"]
    assert registration_ids({"public_record": record["public_record"]}) == registration_ids({"public_record": given}) \
        == {"github:ee2625/dart-spread/releases/tag/V1.0-RC1"}
    assert registration_id({"public_record": "https://github.com/ee2625/dart-spread/releases/tag/v1.0-rc1"}) != \
        registration_id(record)                                                          # a tag is not lower-cased


@pytest.mark.parametrize("tag", ["v1", "v1.0.0", "2026.10.27", "release_2026-10", "V1.0-RC1", "a", "protocol-A"])
def test_any_tag_of_letters_digits_dot_dash_and_underscore_is_a_release_tag(registered, tag):
    from es.registration import registration_id, verify_registration
    write, _ = registered
    proto, reg, sha = write(reg_fields=_public(public_record=f"https://github.com/o/r/releases/tag/{tag}"))
    assert registration_id(verify_registration(sha, proto, reg)) == f"github:o/r/releases/tag/{tag}"


def test_a_public_record_is_not_bound_to_one_repository_name(registered):
    from es.registration import registration_id, verify_registration
    write, _ = registered
    seen = set()
    for url in ("https://github.com/ee2625/dart-spread/releases/tag/v1", "https://github.com/ee2625/dart-spread-es/releases/tag/v1",
                "https://github.com/someone-else/anything.at-all_2/releases/tag/v1"):
        proto, reg, sha = write(reg_fields=_public(public_record=url))
        seen.add(registration_id(verify_registration(sha, proto, reg)))
    assert len(seen) == 3 and all(i.startswith("github:") for i in seen)


@pytest.mark.parametrize("given", ["https://github.com/XXXXX/dart-spread/releases/tag/v1",
                                   "https://github.com/ee2625/XXXXX/releases/tag/v1",
                                   "https://github.com/ee2625/dart-spread/releases/tag/xxxxx",
                                   "https://github.com/<owner>/<repository>/releases/tag/<tag>", "<url>"])
def test_a_placeholder_in_a_public_record_is_refused(registered, given):
    from es.registration import RegistrationError, verify_registration
    write, _ = registered
    proto, reg, sha = write(reg_fields=_public(public_record=given))
    with pytest.raises(RegistrationError, match="placeholder"):
        verify_registration(sha, proto, reg)


# ----------------------------------------------------------------------------- exactly one kind, no key twice, one token
@pytest.mark.parametrize("fields", [
    {"local_record": LOCAL},                                                              # beside the fixture's DOI
    {"public_record": RELEASE},
    {"osf_doi": None, "public_record": RELEASE, "local_record": LOCAL},
    {"osf_doi": None, "osf_registration": URL, "public_record": RELEASE},
    {"osf_doi": None, "osf_registration": URL, "local_record": LOCAL},
    {"osf_doi": DOI, "osf_registration": URL, "public_record": RELEASE, "local_record": LOCAL},
], ids=["doi+local", "doi+public", "public+local", "address+public", "address+local", "all four"])
def test_a_record_that_names_more_than_one_kind_is_refused(registered, fields):
    from es.registration import RegistrationError, parse_registration, verify_registration
    write, _ = registered
    proto, reg, sha = write(reg_fields=fields)
    for call in (lambda: verify_registration(sha, proto, reg), lambda: parse_registration(reg)):
        with pytest.raises(RegistrationError, match="more than one kind") as err:
            call()
        assert "exactly one" in str(err.value)


def test_a_record_with_no_registration_at_all_is_refused_and_says_what_it_may_hold(registered):
    from es.registration import RegistrationError, parse_registration, verify_registration
    write, _ = registered
    proto, reg, sha = write(reg_fields={"osf_doi": None})
    for call in (lambda: verify_registration(sha, proto, reg), lambda: parse_registration(reg)):
        with pytest.raises(RegistrationError, match="registration record lacks") as err:
            call()
        assert "osf_doi or osf_registration, public_record or local_record" in str(err.value)


def test_the_osf_pair_is_one_kind(registered):
    """A DOI and an OSF address that name the same registration are one record, not two kinds."""
    from es.registration import registration_kind, verify_registration
    write, _ = registered
    proto, reg, sha = write(reg_fields={"osf_registration": URL})
    assert registration_kind(verify_registration(sha, proto, reg)) == "osf"


@pytest.mark.parametrize("key, value", [("public_record", RELEASE), ("local_record", LOCAL)])
def test_a_repeated_new_key_is_refused_whatever_its_value(registered, key, value):
    from es.registration import RegistrationError, verify_registration
    write, _ = registered
    proto, reg, sha = write(reg_fields=_local() if key == "local_record" else _public())
    good = reg.read_text()
    for extra in (f"{key}: {value}\n", f"{key}: zzzzz\n", f"{key}: <paste the value>\n", f"{key}:\n"):
        reg.write_text(good + "\n## example\n" + extra)
        with pytest.raises(RegistrationError, match="repeats"):
            verify_registration(sha, proto, reg)


@pytest.mark.parametrize("line", ["public_record: <paste the address of the release>", "public_record:",
                                  "public_record: https://github.com/o/r/releases/tag/v1 # mine",
                                  "local_record: <paste author-statement- and the date>", "local_record:   ",
                                  "local_record: author-statement-2026-10-26 (mine)"])
def test_a_line_that_names_a_new_key_must_hold_exactly_one_token(registered, line):
    """The kind that is not used has to be left out, not left as a template line or a blank."""
    from es.registration import RegistrationError, verify_registration
    write, _ = registered
    proto, reg, sha = write()                                                             # a valid OSF record beside it
    reg.write_text(reg.read_text() + line + "\n")
    key = line.split(":")[0]
    with pytest.raises(RegistrationError, match=key) as err:
        verify_registration(sha, proto, reg)
    assert "key: value" in str(err.value)


def test_the_gate_keys_are_the_six_lines_of_the_record():
    from es import registration as r
    assert set(r._GATE_KEYS) == {"protocol_sha256", "osf_doi", "osf_registration", "public_record", "local_record",
                                 "registered_utc"} and len(r._GATE_KEYS) == 6


# ----------------------------------------------------------------------------- the identity of every kind
def test_the_three_kinds_have_three_disjoint_kinds_of_id_and_a_kind_each():
    from es.registration import RegistrationError, registration_id, registration_ids, registration_kind
    osf, public, local = {"osf_doi": DOI}, {"public_record": RELEASE}, {"local_record": LOCAL}
    assert [registration_kind(r) for r in (osf, {"osf_registration": URL}, public, local)] == \
        ["osf", "osf", "public_release", "local"]
    assert [registration_id(r) for r in (osf, public, local)] == ["abcde", RELEASE_ID, "local:author-statement-2026-10-26"]
    assert len({registration_id(r) for r in (osf, public, local)}) == 3
    both = {**osf, **local}                                                               # parse_registration never lets this out
    assert registration_ids(both) == {"abcde", "local:author-statement-2026-10-26"}
    for record in ({}, both, {**public, **local}):
        with pytest.raises(RegistrationError):
            registration_kind(record)
        with pytest.raises(RegistrationError, match=r"names \d registrations" if record else r"names 0 registrations"):
            registration_id(record)
    # a value that does not have the form of its kind names nothing
    assert registration_ids({"public_record": "https://example.org/x", "local_record": "nonsense"}) == frozenset()
    assert registration_ids({"public_record": None, "local_record": ""}) == frozenset()


# ----------------------------------------------------------------------------- the messages
def test_the_messages_do_not_tell_the_author_to_register_on_osf(tmp_path):
    import es.registration as registration
    from es.registration import RegistrationError, parse_registration, verify_registration
    for call in (lambda: parse_registration(tmp_path / "REGISTRATION.md"),
                 lambda: verify_registration("0" * 64, tmp_path / "PROTOCOL.md", tmp_path / "REGISTRATION.md")):
        with pytest.raises(RegistrationError) as err:
            call()
        assert "record the protocol in docs/sealed/REGISTRATION.md (an OSF registration, a public release or the " \
               "author's local record)" in str(err.value)
        assert "register on OSF" not in str(err.value)
    source = Path(registration.__file__).read_text()
    assert "register on OSF" not in source and "registering on OSF" not in source


def test_the_module_docstring_names_the_three_kinds_and_keeps_its_past_example():
    from es import registration
    doc = registration.__doc__
    for needle in ("osf_doi", "osf_registration", "public_record", "local_record", "exactly one", "reads no network",
                   "author-statement-", "releases/tag/"):
        assert needle in doc, needle
    example = re.search(r"registered_utc: (\S+)", doc).group(1)
    assert registration.is_iso_utc(example) and datetime.fromisoformat(example.replace("Z", "+00:00")) < datetime(
        2026, 10, 1, tzinfo=timezone.utc)
