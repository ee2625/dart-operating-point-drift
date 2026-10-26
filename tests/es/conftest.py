import os
import socket
from pathlib import Path

import pytest

# tests/es/conftest.py -> parents: es, tests, dart_dev, EnergySystems_Revision
REVISION_ROOT = Path(__file__).resolve().parents[3]
os.environ["DART_PROJECT_ROOT"] = str(REVISION_ROOT)

#: Every socket-level network attempt of the session, as (test, call, address). It must stay empty.
SOCKET_ATTEMPTS = []


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: runs the full 33-node pipeline (about 1 minute)")


def pytest_terminal_summary(terminalreporter):
    terminalreporter.write_line(f"[netguard] socket-level network attempts during the session: {len(SOCKET_ATTEMPTS)}")
    for attempt in SOCKET_ATTEMPTS[:10]:
        terminalreporter.write_line(f"[netguard]   {attempt}")


def _trap(call):
    def trap(*args, **kwargs):
        SOCKET_ATTEMPTS.append((os.environ.get("PYTEST_CURRENT_TEST", "?"), call, repr(args[1:2] or args[:1])))
        raise AssertionError(f"network access attempted in a unit test: socket {call}")
    return trap


@pytest.fixture(autouse=True)
def _network_closed_at_the_socket_layer(monkeypatch):
    """No test reaches a network. Connecting, resolving a name and create_connection raise and are counted in
    SOCKET_ATTEMPTS, so a downloader that bound its own urlopen at import (the v2 ingesters do) is stopped too."""
    monkeypatch.setattr(socket.socket, "connect", _trap("connect"))
    monkeypatch.setattr(socket.socket, "connect_ex", _trap("connect_ex"))
    monkeypatch.setattr(socket, "create_connection", _trap("create_connection"))
    monkeypatch.setattr(socket, "getaddrinfo", _trap("getaddrinfo"))


@pytest.fixture
def socket_attempts():
    """The session's list of socket-level network attempts (empty unless a test is broken)."""
    return SOCKET_ATTEMPTS


@pytest.fixture
def registered(tmp_path, monkeypatch):
    """A temporary protocol plus a valid registration record, with the gate pointed at them: es.registration's
    PROTOCOL_PATH and REGISTRATION_PATH are the two files and SEALED_DIR (where PULLS.json and the pull locks live)
    is a temporary directory, so no test reads or writes the real docs/sealed. Returns (write, tmp_path), where
    write(protocol_text) returns (protocol_path, registration_path, protocol_sha256)."""
    import es.registration as registration
    from es.registration import sha256_file

    sealed_dir = tmp_path / "docs_sealed"
    sealed_dir.mkdir()
    monkeypatch.setattr(registration, "PROTOCOL_PATH", tmp_path / "PROTOCOL.md")
    monkeypatch.setattr(registration, "REGISTRATION_PATH", tmp_path / "REGISTRATION.md")
    monkeypatch.setattr(registration, "SEALED_DIR", sealed_dir)

    def write(protocol_text="protocol v1\n", reg_fields=None):
        proto = tmp_path / "PROTOCOL.md"
        proto.write_text(protocol_text)
        sha = sha256_file(proto)
        fields = {"protocol_sha256": sha, "osf_doi": "10.17605/OSF.IO/ABCDE",
                  "registered_utc": "2026-10-20T15:04:00Z"}
        fields.update(reg_fields or {})
        reg = tmp_path / "REGISTRATION.md"
        reg.write_text("# OSF registration record\n" + "".join(f"{k}: {v}\n" for k, v in fields.items() if v))
        return proto, reg, sha

    return write, tmp_path
