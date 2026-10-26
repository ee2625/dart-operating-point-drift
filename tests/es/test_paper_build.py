import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from es.paper_paths import PAPER_DIR

LATEXMK = Path("/Library/TeX/texbin/latexmk")
PDFTOTEXT = Path("/opt/homebrew/bin/pdftotext")
BUILD_SH = PAPER_DIR / "build.sh"


@pytest.mark.slow
@pytest.mark.skipif(not LATEXMK.exists(), reason="TeX Live not installed")
def test_manuscript_compiles_with_pending_macros(tmp_path):
    work = tmp_path / "paper"
    shutil.copytree(PAPER_DIR, work, ignore=shutil.ignore_patterns("build", "*.pdf", "submission"))
    r = subprocess.run([str(LATEXMK), "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                       cwd=work, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-1000:]
    log = (work / "main.log").read_text(errors="replace")
    assert "LaTeX Warning: Reference" not in log
    assert "There were undefined citations" not in log
    assert (work / "main.pdf").exists()


# ---------------------------------------------------------------------------------------------- fix round 1 (Ruling 61)
def _checklog(tmp_path, text):
    log = tmp_path / "x.log"
    log.write_text(text)
    return subprocess.run([str(BUILD_SH), "checklog", str(log)], capture_output=True, text=True, timeout=60)


def test_checklog_catches_wrapped_undefined_reference_and_citation_lines(tmp_path):
    name = "sec:" + "a_very_long_label_name_" * 3
    wrapped = f"LaTeX Warning: Reference `{name[:48]}\n{name[48:]}' on page 1 undefined on input line 12.\n"
    assert "undefined" not in wrapped.splitlines()[0]            # the reviewer's case: "undefined" lands on the next line
    r = _checklog(tmp_path, "some log\n" + wrapped + "more\n")
    assert r.returncode == 1 and "undefined" in r.stderr
    cite = ("Package natbib Warning: Citation `a_very_long_key_that_wraps_the_warning_line_in_the_log_f\n"
            "ile' on page 1 undefined on input line 3.\n")
    assert _checklog(tmp_path, cite).returncode == 1
    plain = "LaTeX Warning: Citation `nokey' on page 1 undefined on input line 3.\n"      # the BibTeX-without-natbib form
    assert _checklog(tmp_path, plain).returncode == 1


def test_checklog_catches_the_summary_lines_and_multiply_defined_labels(tmp_path):
    for bad in ("LaTeX Warning: There were undefined references.\n",
                "Package natbib Warning: There were undefined citations.\n",
                "LaTeX Warning: There were multiply-defined labels.\n",
                "LaTeX Warning: Label `sec:a_very_long_label_name_that_wraps_the_line_in_the_log_fi\nle' multiply defined.\n",
                "LaTeX Warning: Label(s) may have changed. Rerun to get cross-references right.\n"):
        assert _checklog(tmp_path, bad).returncode == 1, bad


def test_checklog_accepts_a_clean_log(tmp_path):
    clean = ("Package natbib Warning: Empty `thebibliography' environment on input line 46.\n"
             "Package hyperref Warning: Difference (4) between bookmark levels is greater \n"
             "Underfull \\vbox (badness 10000) has occurred while \\output is active []\n")
    assert _checklog(tmp_path, clean).returncode == 0


def test_build_sh_reinvokes_itself_by_absolute_path(tmp_path):
    """`all` and `package` call build.sh again; a relative $0 broke them from any directory but paper/."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    log = tmp_path / "ok.log"
    log.write_text("clean\n")
    r = subprocess.run(["bash", os.path.relpath(BUILD_SH, elsewhere), "selfcheck", str(log)], cwd=elsewhere,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    text = BUILD_SH.read_text()
    assert text.count('"$0"') == 1 and '== "$0" ]]' in text           # the only $0 left is the source guard's comparison


def _stub_paper(work):
    """The real main.tex, class, style, bibliography, key table (crossref.tex) and generated/ macros, with stub sections: Task 7's text is not
    under test here. The manuscript has no appendix since plan 2C task T2a."""
    work.mkdir()
    for name in ("main.tex", "sn-jnl.cls", "sn-mathphys-num.bst", "references.bib", "crossref.tex"):
        shutil.copy(PAPER_DIR / name, work / name)
    shutil.copytree(PAPER_DIR / "generated", work / "generated")
    (work / "sections").mkdir()
    for name in re.findall(r"\\input\{(sections/[^}]+)\}", (PAPER_DIR / "main.tex").read_text()):
        body = "\\abstract{Stub.}\n\\keywords{a, b, c, d}\n" if name.endswith("00_abstract") else "\\section{Stub}\n"
        (work / f"{name}.tex").write_text(body)


@pytest.mark.slow
@pytest.mark.skipif(not LATEXMK.exists(), reason="TeX Live not installed")
def test_long_undefined_ref_name_wraps_in_the_real_log_and_is_caught(tmp_path):
    work = tmp_path / "paper"
    _stub_paper(work)
    label = "sec:" + "a_very_long_missing_label_name_" * 3                        # longer than the 79-column log line
    (work / "sections" / "01_introduction.tex").write_text(
        "\\section{Introduction}\\label{sec:intro}\nSee Section~\\ref{" + label + "}.\n")
    r = subprocess.run([str(LATEXMK), "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=work,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-1000:]
    log = (work / "main.log").read_text(errors="replace")
    ref_lines = [line for line in log.splitlines() if line.startswith("LaTeX Warning: Reference")]
    assert ref_lines and all("undefined" not in line for line in ref_lines)         # the warning wrapped before "undefined"
    assert not re.search(r"LaTeX Warning: (Reference|Citation) .* undefined", log)  # so the old one-line check passed this log
    caught = subprocess.run([str(BUILD_SH), "checklog", str(work / "main.log")], capture_output=True, text=True, timeout=60)
    assert caught.returncode == 1 and "Reference" in caught.stderr


@pytest.mark.slow
@pytest.mark.skipif(not LATEXMK.exists() or not PDFTOTEXT.exists(), reason="TeX Live or poppler not installed")
def test_escaped_bar_less_and_greater_render_as_themselves_in_the_pdf(tmp_path):
    """Ruling 65: under OT1, |DART| >= q prints as an em dash, an inverted ! and an inverted ? unless tex_escape maps them."""
    from es.paper_tex import tex_escape
    work = tmp_path / "paper"
    _stub_paper(work)
    plain = ["spike = |DART| >= q", "MAD <= 1e-6"]
    (work / "sections" / "01_introduction.tex").write_text(
        "\\section{Introduction}\\label{sec:intro}\n"
        "RAW " + " and ".join(plain) + ".\n\n"                                       # the control: what the bug printed
        "ESC " + " and ".join(tex_escape(s) for s in plain) + ".\n\n"
        "MATH $\\text{" + tex_escape(plain[0]) + "}$.\n")
    r = subprocess.run([str(LATEXMK), "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=work,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-1000:]
    text = subprocess.run([str(PDFTOTEXT), "-layout", str(work / "main.pdf"), "-"], capture_output=True, text=True,
                          timeout=60).stdout
    text = " ".join(text.split())
    assert "ESC spike = |DART| >= q and MAD <= 1e-6." in text
    assert "MATH spike = |DART| >= q." in text                                         # also inside \\text{} in math mode
    assert "RAW spike = |DART| >= q" not in text and "—DART—" in text                 # the unescaped text is garbled


# ---------------------------------------------------------------------------------------------- fix round 2 (Ruling 71)
def _run_bash(script):
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=120)


def test_checklog_fails_on_a_missing_unreadable_or_empty_log(tmp_path):
    """grep exits 2 on a file it cannot read, which `if grep -q` treated as "no problem found"."""
    def check(path):
        return subprocess.run([str(BUILD_SH), "checklog", str(path)], capture_output=True, text=True, timeout=60)

    missing = check(tmp_path / "nope.log")
    assert missing.returncode == 1 and "no readable LaTeX log" in missing.stderr
    empty = tmp_path / "empty.log"
    empty.write_text("")
    assert check(empty).returncode == 1                       # a finished LaTeX run always writes a banner
    assert check(tmp_path).returncode == 1                    # a directory is not a log
    locked = tmp_path / "locked.log"
    locked.write_text("This is pdfTeX, Version 3.14\n")
    os.chmod(locked, 0)
    try:
        refused = check(locked)
        assert refused.returncode == 1 and "no readable LaTeX log" in refused.stderr
    finally:
        os.chmod(locked, 0o644)
    ok = tmp_path / "ok.log"
    ok.write_text("This is pdfTeX, Version 3.14\n")
    assert check(ok).returncode == 0


def test_compile_fails_without_a_log_and_passes_with_a_clean_one(tmp_path):
    """compile calls check_log on the log it expects; a latexmk that leaves none must not be a good build. The script is
    sourced with a bogus target, so that a build.sh without the source guard stops at "unknown target", not at `all`."""
    work = tmp_path / "work"
    work.mkdir()

    def fake(name, body):
        path = tmp_path / name
        path.write_text("#!/bin/bash\n" + body + "\nexit 0\n")
        path.chmod(0o755)
        return path

    def compile_with(latexmk):
        return _run_bash(f'source "{BUILD_SH}" __sourced__; LATEXMK="{latexmk}"; compile "{work}" main')

    silent = compile_with(fake("silent_latexmk", ":"))
    assert silent.returncode == 1 and "no readable LaTeX log" in silent.stderr
    assert compile_with(fake("clean_latexmk", 'echo "This is pdfTeX" > main.log')).returncode == 0
    bad = compile_with(fake("bad_latexmk", 'echo "LaTeX Warning: There were undefined references." > main.log'))
    assert bad.returncode == 1 and "undefined" in bad.stderr


# ---------------------------------------------------------------------------------------------- Ruling 74
def test_all_and_package_run_impl_with_data_first_and_a_failing_impl_stops_the_build(tmp_path):
    """`all` and `package` never ran impl, so implementation.json could be stale relative to the code. The script is sourced
    with a fake SELF that logs its calls, so the real paper is never built and `all` is never run for real."""
    calls = tmp_path / "calls.log"
    fake_self = tmp_path / "fake_self"
    fake_self.write_text('#!/bin/bash\necho "$*" >> "$CALLS"\n[ "${FAIL_ON:-}" = "$1" ] && exit 1\nexit 0\n')
    fake_self.chmod(0o755)

    def run(target, fail_on=""):
        calls.write_text("")
        script = (f'source "{BUILD_SH}" __sourced__; SELF="{fake_self}"; '
                  'run_py() { echo "run_py $*" >> "$CALLS"; }; main ' + target)
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60,
                           env={**os.environ, "CALLS": str(calls), "FAIL_ON": fail_on})
        return r.returncode, calls.read_text().splitlines()

    assert run("all") == (0, ["impl --with-data", "macros", "tables", "figures", "pdf", "lint"])
    assert run("package") == (0, ["impl --with-data", "strict", "pdf", "esm", "run_py es.paper_package"])
    for target in ("all", "package"):
        rc, seen = run(target, fail_on="impl")
        assert rc != 0 and seen == ["impl --with-data"], (target, rc, seen)      # nothing runs after a failed impl
    rc, seen = run("all", fail_on="macros")
    assert rc != 0 and seen == ["impl --with-data", "macros"]                  # the chain still stops at any other failure


# ---------------------------------------------------------------------------------------------- hardening round (Ruling 76)
@pytest.mark.slow
@pytest.mark.skipif(not LATEXMK.exists() or not PDFTOTEXT.exists(), reason="TeX Live or poppler not installed")
def test_strata_texts_and_stat_labels_compile_and_read_in_the_pdf(tmp_path):
    """The strata texts hold the multiplication sign (U+00D7) and the E4 contrast labels a minus. This class stops on the
    Unicode minus sign (U+2212: "Unicode character - not set up for use with LaTeX"), so the labels use \\textminus{}."""
    from es.paper_macros import Sources, render, resolve
    from es.paper_registry import STAT_LABEL, entries_by_name
    by = entries_by_name()
    src = Sources(tmp_path / "out", tmp_path / "paper_inputs", tmp_path / "docs_sealed")
    resolved = [resolve(by[n], src) for n in ("strataPrimaryText", "strataDecileText")]
    assert all(r["status"] == "ok" for r in resolved)
    macros = render(resolved, {"A": False, "B": False, "Adidnotrun": False, "Bdidnotrun": False})
    assert "\u2212" not in macros and not any("\u2212" in label for label in STAT_LABEL.values())
    work = tmp_path / "paper"
    _stub_paper(work)
    # the engine's own macro lines, under other names so that they do not meet the real macros.tex in the copy
    defs = [line.replace("\\newcommand{\\strata", "\\newcommand{\\probeStrata")
            for line in macros.splitlines() if line.startswith("\\newcommand{\\strata")]
    assert len(defs) == 2
    (work / "sections" / "01_introduction.tex").write_text(
        "\\section{Introduction}\\label{sec:intro}\n" + "\n".join(defs) + "\n"
        "PRIMARY \\probeStrataPrimaryText.\n\nDECILE \\probeStrataDecileText.\n\n"          # one paragraph each: no line break inside
        + "".join(f"LABEL {label}.\n\n" for label in STAT_LABEL.values()), encoding="utf-8")
    r = subprocess.run([str(LATEXMK), "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=work,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-1000:]
    assert subprocess.run([str(BUILD_SH), "checklog", str(work / "main.log")], capture_output=True, text=True,
                          timeout=60).returncode == 0
    text = subprocess.run([str(PDFTOTEXT), "-layout", str(work / "main.pdf"), "-"], capture_output=True, text=True,
                          timeout=60).stdout
    text = " ".join(text.split()).replace("\u2212", "-").replace("\u2013", "-")        # the minus glyph may extract as either
    text = text.replace("\u2019", "'")                                                  # the typeset apostrophe
    assert "PRIMARY node \u00d7 local hour \u00d7 weekday class \u00d7 month." in text
    assert "DECILE node \u00d7 decile of the calendar-only model's score." in text
    assert "LABEL E4: SELC(P1) - SELC(calendar-only), matched count." in text
    assert "LABEL E4: SELC(P1) - SELC(VOL), matched count." in text
    assert "LABEL E4: SELC of the calendar-only model, matched count." in text


# ---------------------------------------------------------------------------------------------- a submission before Test A (25 Oct)
def _main_with_fake_self(tmp_path, target, fail_on=""):
    calls = tmp_path / "calls.log"
    calls.write_text("")
    fake_self = tmp_path / "fake_self"
    fake_self.write_text('#!/bin/bash\necho "$*" >> "$CALLS"\n[ "${FAIL_ON:-}" = "$1" ] && exit 1\nexit 0\n')
    fake_self.chmod(0o755)
    script = (f'source "{BUILD_SH}" __sourced__; SELF="{fake_self}"; '
              'run_py() { echo "run_py $*" >> "$CALLS"; }; main ' + target)
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60,
                       env={**os.environ, "CALLS": str(calls), "FAIL_ON": fail_on})
    return r.returncode, calls.read_text().splitlines()


def test_package_pending_gates_with_the_pending_stages_and_checks_both_pdfs_before_it_packages(tmp_path):
    rc, seen = _main_with_fake_self(tmp_path, "package-pending")
    assert rc == 0 and len(seen) == 6
    assert seen[:4] == ["impl --with-data", "strict-pending", "pdf", "esm"]
    assert seen[4].split()[0] == "pdfcheck" and seen[4].split()[1].endswith("paper/main.pdf") and seen[4].split()[2].endswith("paper/esm/ESM_1.pdf")
    assert seen[5] == "run_py es.paper_package"
    for stage in ("impl", "strict-pending", "pdf", "esm", "pdfcheck"):
        rc, seen = _main_with_fake_self(tmp_path, "package-pending", fail_on=stage)
        assert rc != 0 and "run_py es.paper_package" not in seen, stage              # nothing is packaged after a failed step
    # the plain package is unchanged: the strict gate with Test A in it
    assert _main_with_fake_self(tmp_path, "package") == (0, ["impl --with-data", "strict", "pdf", "esm", "run_py es.paper_package"])


def test_strict_pending_checks_every_stage_but_test_a_and_lets_the_lint_accept_a_pending_test_a(tmp_path):
    assert _main_with_fake_self(tmp_path, "strict-pending") == (
        0, ["run_py es.paper_macros --strict --stages const,dev,author", "run_py es.paper_lint --strict --pending-ok A"])
    rc, seen = _main_with_fake_self(tmp_path, "strict")                           # the default gate still checks Test A (its default stages)
    assert (rc, seen) == (0, ["run_py es.paper_macros --strict", "run_py es.paper_lint --strict"])


def test_pdfcheck_refuses_printed_placeholders_but_not_the_word_pending_in_running_text(tmp_path):
    """pdftotext is replaced by `cat`: the "PDFs" here are text files."""
    fake = tmp_path / "fake_pdftotext"
    fake.write_text('#!/bin/bash\ncat "$1"\n')
    fake.chmod(0o755)

    def check(*files):
        script = f'source "{BUILD_SH}" __sourced__; PDFTOTEXT="{fake}"; pdfcheck ' + " ".join(f'"{f}"' for f in files)
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)

    clean = tmp_path / "clean.pdf"
    clean.write_text("Test B (October-December 2026) is pending.\nTest A: pending\n")
    placeholder = tmp_path / "placeholder.pdf"
    placeholder.write_text("Fine line.\ncode at [pending: author input], archived at x\n")
    assert check(clean).returncode == 0 and check(clean, clean).returncode == 0
    r = check(clean, placeholder)
    assert r.returncode == 1 and "[pending: author input]" in r.stderr and "placeholder.pdf" in r.stderr
    from es.paper_macros import PENDING_TEXT
    sentinels = ("[pending: Test A]", "[pending: Test B]", "[pending: constant]", "[pending: development results]", "[pending: author input]")
    assert set(sentinels) == set(PENDING_TEXT.values())                         # every sentinel that the macros can print is refused
    for sentinel in sentinels:
        bad = tmp_path / "bad.pdf"
        bad.write_text(f"value {sentinel} here\n")
        assert check(bad).returncode == 1, sentinel
    assert check(tmp_path / "missing.pdf").returncode == 1                      # a missing PDF is a failed check, not a clean one
    empty = tmp_path / "empty.pdf"
    empty.write_text("")
    assert check(empty).returncode == 1
    assert subprocess.run(["bash", "-c", f'source "{BUILD_SH}" __sourced__; pdfcheck'], capture_output=True, text=True).returncode == 2
