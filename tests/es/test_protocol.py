import gzip
import io
import tarfile
from pathlib import Path

DEV_ROOT = Path(__file__).resolve().parents[2]
SPEC = DEV_ROOT.parent / "docs/superpowers/specs/2026-10-22-economic-value-sealed-q3-design.md"
FORENSICS = DEV_ROOT.parent / "docs/forensics/FORENSICS.md"


def _ctx():
    return {"inputs": {"eta": 0.01, "rho_star": 0.1, "analog_selc": [0.05, 0.04],
                       "simulation": [{"test": "H-E pooled A+B, size", "reject_0.0166667": 0.012, "reject_0.05": 0.04}],
                       "placebo": [{"pair": "AQ3-1y", "p": 0.4}],
                       "family": {h: {"confirmatory": True, "reasons": [], "approximate": False,
                                      "interim_approximate": False} for h in ("H-E", "H-B1", "H-B2")}},
            "versions": {"python": "3.12.7", "xgboost": "3.0.2"},
            "sha": {"manifest": "a" * 64, "archive": "b" * 64, "cutoffs": "c" * 64, "inputs": "d" * 64},
            "git": None,
            "knowledge": None,                     # no statement yet: the section renders one [AUTHOR: ...] marker
            "spec": SPEC.read_text(),
            "forensics": FORENSICS.read_text()}


def test_protocol_contains_registered_content_and_pins():
    from es.protocol import render_protocol
    text = render_protocol(_ctx())
    for needle in ("H-E", "H-B1", "H-B2", "rotation", "Holm", "0.05/3", "2026-06-21", "2026-07-01",
                   "2026-12-31", "28 nodes", "xgboost 3.0.2", "a" * 64, "b" * 64, "c" * 64, "d" * 64,
                   "eta = 0.01", "Knowledge statement", "pre_pull_cutoffs.csv", "Table shells",
                   "day-rotation test", "Benjamini", "Secondary families"):
        assert needle in text, needle


def test_author_markers_block_registration():
    from es.protocol import ready_to_register, render_protocol
    ctx = _ctx()
    assert ready_to_register(render_protocol(ctx))                                   # knowledge None
    with_marker = "## 3. Statement\n\n[AUTHOR: confirm or edit] \"I have not examined ...\"\n"
    assert ready_to_register(render_protocol({**ctx, "knowledge": with_marker})) == ["[AUTHOR: confirm or edit]"]
    filled = "## 3. Statement\n\n\"I have not examined electricity prices after 30 June 2026.\"\n\nErce Erez, 2026-10-27\n"
    text = render_protocol({**ctx, "knowledge": filled})
    assert ready_to_register(text) == [] and filled.replace("## 3. Statement", "#### 3. Statement") in text   # Ruling 119, M20


def test_section_extracts_one_heading():
    from es.protocol import section
    md = "## 6\n### 6.1 A\nalpha\n### 6.2 B\nbeta\n## 7\n"
    assert section(md, "### 6.1") == "### 6.1 A\nalpha\n"
    assert section(md, "## 6") == "## 6\n### 6.1 A\nalpha\n### 6.2 B\nbeta\n"


def test_section_and_list_item_on_the_real_spec_and_forensics():
    from es.protocol import list_item, section
    spec, forensics = SPEC.read_text(), FORENSICS.read_text()
    s63 = section(spec, "### 6.3")
    assert s63.startswith("### 6.3 Hypotheses") and "H-B2" in s63 and "### 6.4" not in s63
    assert section(spec, "### 6.5").startswith("### 6.5 Freeze") and "### 6.6" not in section(spec, "### 6.5")
    one, two = list_item(spec, "## 5.", 1), list_item(spec, "## 5.", 2)
    assert one.startswith("1. **Inference**") and "day-rotation test" in one and "Mixed model" in one
    assert "Multiplicity" not in one
    assert two.startswith("2. **Multiplicity.**") and "Benjamini" in two and "Table and figure shells" in two
    assert "Precision and recall" not in two
    assert "Precision lift" in list_item(spec, "## 5.", 3)            # a blank line then an indented paragraph
    s5c = section(forensics, "## 5c.")
    assert s5c.startswith("## 5c.") and "## 5b." not in s5c and section(forensics, "## 5a.").startswith("## 5a.")


def _members(path):
    with gzip.open(path, "rb") as gz, tarfile.open(fileobj=io.BytesIO(gz.read())) as tar:
        return [(m.name, m.mtime, m.uid, m.gid, m.uname, m.gname, m.mode) for m in tar.getmembers()]


def test_archive_bytes_are_deterministic_for_a_fixed_file_list(tmp_path):
    from es.protocol import build_archive
    files = sorted(DEV_ROOT.glob("es/*.py")) + [DEV_ROOT / "pytest.ini"]
    a = build_archive(DEV_ROOT, files, tmp_path / "a.tar.gz")
    b = build_archive(DEV_ROOT, list(reversed(files)), tmp_path / "b.tar.gz")
    assert a == b and (tmp_path / "a.tar.gz").read_bytes() == (tmp_path / "b.tar.gz").read_bytes()
    header = (tmp_path / "a.tar.gz").read_bytes()[:10]
    assert header[3] == 0 and header[4:8] == b"\x00\x00\x00\x00"       # no FNAME flag, mtime 0
    members = _members(tmp_path / "a.tar.gz")
    assert [m[0] for m in members] == sorted(p.relative_to(DEV_ROOT).as_posix() for p in files)
    assert all(m[1:] == (0, 0, 0, "", "", 0o644) for m in members)


def test_archive_is_deterministic(tmp_path):
    from es.protocol import build_archive, code_files
    files = code_files(DEV_ROOT)
    for name in ("sealed_run.py", "panel_prep.py", "ingest_nyiso.py", "ingest_isone.py", "ingest_miso.py", "NODE_MAP.csv"):
        assert any(p.name == name for p in files), name
    assert build_archive(DEV_ROOT, files, tmp_path / "a.tar.gz") == build_archive(DEV_ROOT, files, tmp_path / "b.tar.gz")


# --------------------------------------------------------------------------------------------------------------------
# Beyond the brief's six tests: the rest of the module (renderer, bound files, manifest, git, template, main()).
# --------------------------------------------------------------------------------------------------------------------
import ast
import copy
import json
import re
import subprocess
import sys

import pandas as pd
import pytest

MARKER = "[AUTHOR:"
REQUIRED_RULINGS = (12, 26, 27, 29, 30, 31, 32, 33, 37, 39, 40, 41, 43, 44, 55, 63, 64, 66, 68, 69, 70, 73, 77, 78, 80,
                    81, 82, 83, 84, 89, 90, 92, 93, 94, 104, 106, 107, 109, 110, 111, 112)


def _inputs():
    """protocol_inputs.json in its real structure (stage complete), small."""
    def size(test, **kw):
        return {"test": test, "scenario": "homoskedastic", "persistence": "fit", "n_sim": 300, "seed": 1, "rho": 0.0,
                "reject_0.0166667": 0.01, "reject_0.05": 0.04, **kw}

    sim = [size("H-E single A, size", strata="primary"), size("H-E single A, size", strata="decile"),
           size("H-E pooled A+B, size", strata="primary", scenario="heteroskedastic", **{"reject_0.0166667": 0.0233,
                                                                                         "reject_0.05": 0.07}),
           size("H-E pooled A+B, size", strata="primary"), size("H-E pooled A+B, size", strata="decile"),
           *[size("H-E pooled A+B, power curve", strata="primary", rho=r, n_sim=100, seed=7)
             for r in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)],
           size("H-E pooled A+B, power", strata="primary", rho=0.05, note="target below the grid"),
           {"test": "H-B size", "law": "LEV6", "scenario": "common_drift_0.5", "persistence": "upper", "n_sim": 300,
            "seed": 3, "reject_0.0166667": 0.13, "reject_0.05": 0.19},
           {"test": "H-B1 power", "law": "DART3", "scenario": "no_drift", "persistence": "fit", "n_sim": 300, "seed": 13,
            "reject_0.0166667": 0.28, "reject_0.05": 0.43}]
    fam_h = {"approximate": True, "k_rows": 12, "false_demotion_bound": 0.052, "power": 0.28, "power_lower_bound": False,
             "confirmatory": False, "reasons": ["simulated size 0.130 > 1.5 x 0.0167 (39 of 300 replications)"]}
    return {"eta": 0.05, "stage": "complete", "rho_star": 0.05, "analog_selc": [-0.0063, -0.0043],
            "analog_selc_pooled": -0.0053,
            "family": {"H-E": {"confirmatory": True, "reasons": [], "approximate": False, "interim_approximate": False,
                               "k_rows": 4, "false_demotion_bound": 0.074},
                       "H-B1": copy.deepcopy(fam_h), "H-B2": {**copy.deepcopy(fam_h), "power": 0.05},
                       "notes": ["Pooled placebo check (1 counted 1-year pairing): H-E is replaced when its p is at "
                                 "most 0.0167."]},
            "rule": {"factor": 1.5, "binomial_level": 0.05, "approximate_factor": 1.0, "min_power": 0.5,
                     "levels": [0.016666666666666666, 0.05], "n_rep": 299, "k_rows": {"H-E": 4, "H-B1": 12, "H-B2": 12},
                     "seeds": {"size_base": 20261022, "rho_search": 7}},
            "phi": {"DART3": {"phi_score_median": 0.79, "phi_score_raw_median": 0.69},
                    "LEV6": {"phi_score_median": 0.77, "phi_score_raw_median": 0.68}},
            "phi_status": {"H-E": {"components": ["score", "pnl"], "phi_nodes": 33, "phi_corrected": 33,
                                   "phi_clipped_high": 3, "phi_clipped_low": 0, "phi_no_signal": 1, "phi_uncorrected": 0,
                                   "by_component": {
                                       "score": {"corrected": 33, "clipped_high": 0, "clipped_low": 0, "no_signal": 0,
                                                 "uncorrected": 0},
                                       "pnl": {"corrected": 29, "clipped_high": 3, "clipped_low": 0, "no_signal": 1,
                                               "uncorrected": 0}}, "note": None}},
            "simulation": sim,
            "placebo": [{"pair": "AQ3-1y", "p": 0.27, "t_obs": 1.37, "n_null": 91, "selc_obs": 0.012, "counted": True},
                        {"pair": "AQ3-2y", "p": 0.59, "t_obs": -1.57, "n_null": 91, "selc_obs": -0.011, "counted": False,
                         "note": "P&L year inside the alert model's training window"}]}


def _full_ctx(**over):
    ctx = {**_ctx(), "inputs": _inputs(), "date": "2026-10-23"}
    ctx["versions"] = {"python": "3.12.7", "numpy": "1.26.4", "pandas": "2.2.2", "xgboost": "3.0.2"}
    ctx["sha"] = {**ctx["sha"], "env": "e" * 64}                       # ENV_PINS.json (Ruling 109); the brief's ctx has none
    ctx.update(over)
    return ctx


@pytest.fixture(scope="module")
def real_code_files():
    from es.protocol import code_files
    return code_files(DEV_ROOT)


def test_part_a_is_the_registered_design_verbatim():
    """Controller decision on item 7: sections 4.1-4.6, 5 (items 1-9), 6.2 and 6.3 (and 6.1, 6.4, 6.5) word for word."""
    from es.protocol import SPEC_RECORD_SENTENCES, list_item, render_protocol, section
    text = render_protocol(_full_ctx())
    spec = SPEC.read_text()
    assert "## Part A. Registered design (spec v3 with amendments)" in text
    s4 = section(spec, "## 4.")
    assert s4[:s4.index("### 4.1")].rstrip("\n") in text                     # the heading and the sign convention
    for k in range(1, 7):
        assert section(spec, f"### 4.{k}").rstrip("\n") in text, k
    s5 = section(spec, "## 5.")
    assert s5.rstrip("\n") in text                                           # items 1 to 9, as one block
    for n in range(1, 10):
        assert re.search(rf"^{n}\. ", s5, flags=re.M), n
    assert list_item(spec, "## 5.", 1) in text and list_item(spec, "## 5.", 2) in text
    for h in ("### 6.1", "### 6.2", "### 6.3", "### 6.4", "### 6.5"):
        expected = section(spec, h).rstrip("\n")
        for old, new in SPEC_RECORD_SENTENCES:          # Ruling 140: the three sentences that named OSF as the record are
            expected = expected.replace(old, new)       # rewritten, in sections 6.1 and 6.5 only; everything else is verbatim
        assert expected in text, h
    assert all(spec.count(old) == 1 and old not in text and new in text for old, new in SPEC_RECORD_SENTENCES)
    # the parts of the spec that are not registered design are not copied
    assert "### 6.6" not in text and "## 3. Invariants" not in text and "## 7. Paper and package" not in text
    # the order is the spec's: 4 before 5 before 6.1 ... 6.5, all inside Part A
    part_a = text[text.index("## Part A."):text.index("## Part B.")]
    marks = [part_a.index(m) for m in ("## 4. Economic analysis", "### 4.2 Positions and fees", "### 4.6 Windows",
                                       "## 5. Statistics", "### 6.1 Windows", "### 6.2 Development analogues",
                                       "### 6.3 Hypotheses", "### 6.4 Pins", "### 6.5 Freeze")]
    assert marks == sorted(marks)


def test_both_fee_tables_appear_and_the_rendered_one_is_declared_binding():
    """The spec's section 4.2 table (superseded, as the spec's own note says) and the table rendered from
    CALIBRATED_FEES both appear, and the protocol says in both places which one governs."""
    import es.positions as positions
    from es.protocol import render_protocol
    text = render_protocol(_full_ctx())
    part_a = text[text.index("## Part A."):text.index("## Part B.")]
    assert "| NYISO | 0.20 | 0.20 |" in part_a and "| ISO-NE | 0.75 | 0.75 |" in part_a      # the superseded table, verbatim
    note = part_a.index("Superseded 23 Oct by controller Rulings 78 and 90")
    mine = part_a.index("> **Protocol note on the fee table of section 4.2.**")
    assert note < mine < part_a.index("### 4.3 Maintenance protocols")              # right after the table and its note
    words = ", ".join(f"{m} {side[1]:.2f}/{side[-1]:.2f}" for m, side in positions.CALIBRATED_FEES.items())
    assert f"(INC/DEC): {words}. Where the two differ, the rendered levels govern." in part_a
    fees = text[text.index("### Fees (Rulings 78 and 90)"):text.index("### The sealed pull and the sealed run")]
    assert "**The binding levels** are those of `es.positions.CALIBRATED_FEES`" in fees
    assert "superseded by them (its own note says so); where the two differ, this table governs" in fees
    for market, side in positions.CALIBRATED_FEES.items():
        assert f"| {market} | {side[1]:.2f} | {side[-1]:.2f} |" in fees
    # the final dict is cited from the controller note, not from the record's section 3 headline
    assert "controller note at the top of `docs/fee_rederivation_2026-10-23.md`" in fees
    assert "the adopted values differ from that record's section 3 recommendation for MISO" in fees


def test_the_spec_and_the_generated_text_carry_no_author_marker_but_the_knowledge_statement():
    from es.protocol import author_markers, ready_to_register, render_protocol
    assert MARKER not in SPEC.read_text()
    text = render_protocol(_full_ctx(knowledge="Name and date: Erce Erez, 2026-10-27\n"))
    assert ready_to_register(text) == []
    template = (DEV_ROOT.parent / "docs/sealed/KNOWLEDGE_STATEMENT_TEMPLATE.md").read_text()
    rendered = render_protocol(_full_ctx(knowledge=template))
    markers = author_markers(rendered)
    assert len(markers) == 4 and sum(m.startswith("[AUTHOR: one or two") for m in markers) == 1
    # Ruling 119, M15: the unfilled template blocks registration with its markers, an unticked row for each question of
    # item 1, and its two guidance paragraphs: 4 + 4 + 2 = 10 problems, the markers first
    items = ready_to_register(rendered)
    rows = [ln for ln in template.splitlines() if ln.startswith("|") and "\u2610" in ln]
    guidance = [p for p in re.split(r"\n\s*\n", template) if p.strip().startswith(("*Fill this in", "*This covers"))]
    kinds = [re.match(r"\[(AUTHOR|ITEM 1|GUIDANCE)", i).group(1) for i in items]
    assert (len(markers), len(rows), len(guidance)) == (4, 4, 2) and len(items) == 10
    assert items[:4] == markers and [kinds.count(k) for k in ("AUTHOR", "ITEM 1", "GUIDANCE")] == [4, 4, 2]


def test_a_marker_in_the_generated_text_outside_the_knowledge_statement_is_refused():
    from es.protocol import render_protocol
    ctx = _full_ctx(spec=SPEC.read_text().replace("### 6.5 Freeze", "### 6.5 [AUTHOR: x] Freeze"))
    with pytest.raises(ValueError, match="author marker"):
        render_protocol(ctx)


def test_the_bound_files_are_bound_through_verify_bound(tmp_path):
    from es.protocol import render_protocol
    from es.registration import RegistrationError, sha256_file, verify_bound
    files = {name: tmp_path / name for name in ("pre_pull_cutoffs.csv", "protocol_inputs.json", "CODE_MANIFEST.sha256",
                                                "ENV_PINS.json")}
    for name, p in files.items():
        p.write_text(f"content of {name}\n")
    sha = {"cutoffs": sha256_file(files["pre_pull_cutoffs.csv"]), "inputs": sha256_file(files["protocol_inputs.json"]),
           "manifest": sha256_file(files["CODE_MANIFEST.sha256"]), "env": sha256_file(files["ENV_PINS.json"]),
           "archive": "e" * 64}
    protocol = tmp_path / "PROTOCOL.md"
    protocol.write_text(render_protocol(_full_ctx(sha=sha)))
    bound = verify_bound(protocol, list(files.values()))
    assert set(bound.values()) == {sha["cutoffs"], sha["inputs"], sha["manifest"], sha["env"]}
    files["ENV_PINS.json"].write_text("another interpreter\n")              # the environment pins are bound too
    with pytest.raises(RegistrationError, match="ENV_PINS.json"):
        verify_bound(protocol, list(files.values()))
    files["ENV_PINS.json"].write_text("content of ENV_PINS.json\n")
    files["protocol_inputs.json"].write_text("changed\n")                  # a changed bound file is refused
    with pytest.raises(RegistrationError, match="not the registered file"):
        verify_bound(protocol, list(files.values()))


def test_each_bound_name_has_exactly_one_hash_line():
    from es.protocol import render_protocol
    from es.registration import _protocol_hashes
    text = render_protocol(_full_ctx())
    for name, digit in (("pre_pull_cutoffs.csv", "c"), ("protocol_inputs.json", "d"), ("CODE_MANIFEST.sha256", "a"),
                        ("ENV_PINS.json", "e"), ("code_archive.tar.gz", "b")):
        assert _protocol_hashes(text, name) == [digit * 64], name


def test_the_family_text_follows_the_inputs():
    from es.protocol import render_protocol
    text = render_protocol(_full_ctx())
    assert "**H-E is the one confirmatory test.**" in text
    assert "**H-B1 is an estimand, not a test**" in text and "**H-B2 is an estimand, not a test**" in text
    assert "**Test A has no confirmatory test of its own.**" in text
    assert "never \"less portable\"" in text and "larger operating-point error" in text
    assert ("If Test A's output is not available (it is NOT_RUN because its coverage rule failed, it was claimed and left "
            "no manifest and no finished record, or it finished without stored P1 rows), Test B runs alone and tests H-E on "
            "its own window, reported as not confirmatory.") in text
    assert "if Test A's output is complete but its record lacks the finish fields, which a repair by hand restores " \
           "(Rulings 112 and 116; see Failure handling)" in text
    assert "Test B is refused, before anything is spent, if Test A was never started, and if Test A's output is complete " \
           "but its record lacks the finish fields" in text
    assert "has no recorded run" not in text                     # the earlier wording, which the code does not do
    assert "DART3 0.790 (raw 0.690)" in text and "H-E 0.074, H-B1 0.052, H-B2 0.052" in text
    assert "simulated size 0.130 > 1.5 x 0.0167" in text                         # the reasons, as the file records them
    # all three confirmatory: no demotion text, and Test A can confirm H-B
    allc = _full_ctx()
    for h in ("H-B1", "H-B2"):
        allc["inputs"]["family"][h].update(confirmatory=True, reasons=[], approximate=False)
    t2 = render_protocol(allc)
    assert "**H-B1 is a confirmatory test**" in t2 and "has no confirmatory test of its own" not in t2
    # H-E demoted: stated as an estimand
    demoted = _full_ctx()
    demoted["inputs"]["family"]["H-E"].update(confirmatory=False, reasons=["simulated size too large"])
    t3 = render_protocol(demoted)
    assert "**H-E is replaced by its estimand**" in t3 and "H-E is the one confirmatory test" not in t3


def test_h_e_text_is_at_or_near_nominal_and_never_conservative():
    from es.protocol import render_protocol
    text = render_protocol(_full_ctx())
    assert "H-E's size is at or near nominal" in text
    assert "0.0233 at 0.05/3" in text and "0.0700 at 0.05" in text                # the worst row, from the simulation
    assert "conservative" in text and "no claim that the test is conservative" in text
    assert "above the grid" not in text and "lies below the grid of couplings" in text      # the note of the power row
    no_note = _full_ctx()
    for r in no_note["inputs"]["simulation"]:
        r.pop("note", None)
    assert "lies below the grid of couplings" not in render_protocol(no_note)


def test_numbers_that_the_code_holds_are_read_from_the_code(monkeypatch):
    import es.positions as positions
    from es.analyses import SEALED
    from es.protocol import render_protocol
    text = render_protocol(_full_ctx())
    for market, side in positions.CALIBRATED_FEES.items():
        assert f"| {market} | {side[1]:.2f} | {side[-1]:.2f} |" in text
    assert "MISO omits the Schedule 17 and Schedule 24" in text and "docs/fee_rederivation_2026-10-23.md" in text
    assert "They are the levels the sealed run uses." in text
    assert "mean block 7 days, B = 2000, seed 20261022" in text
    assert "(D = 92 days for Test A, 92 for Test B)" in text and "8464 for the pooled test" in text
    assert "1/92 for a single window and 1/8464 pooled" in text
    for key in ("SA", "SAF", "SB", "SBF"):
        a = SEALED[key]
        assert f"| {a.id} | {a.maintenance} | {a.train.start} to {a.train.end} | {a.calib.start} to {a.calib.end} |" in text
    assert "Monitors: W1_norm, PSI, domain_auc, early_rate" in text and "days 1 to 14" in text
    assert "class-stratified 5-fold" in text and "34 score sets" in text and "at least 28 nodes" in text
    # a changed fee shows up: the protocol states what the code holds at the time it is rendered
    monkeypatch.setitem(positions.CALIBRATED_FEES["NYISO"], 1, 0.35)
    changed = render_protocol(_full_ctx())
    assert "| NYISO | 0.35 | 0.15 |" in changed and "(INC/DEC): NYISO 0.35/0.15, ISONE" in changed       # Part D and Part A's note


def test_the_secondary_families_docstring_is_quoted_word_for_word():
    import es.secondary as secondary
    from es.protocol import render_protocol
    text = render_protocol(_full_ctx())
    assert "### Secondary families" in text and secondary.__doc__ in text
    assert "BH's m is the number of rows with a p-value" in text and "not read as control of the false-discovery rate" in text


def test_the_refinements_list_has_every_required_ruling_once():
    from es.protocol import render_protocol
    text = render_protocol(_full_ctx())
    part = text[text.index("## Part E."):text.index("## Part F.")]
    numbers = [int(n) for n in re.findall(r"^- \*\*Ruling (\d+)\*\*", part, flags=re.M)]
    assert numbers == sorted(numbers) and len(numbers) == len(set(numbers))
    assert set(REQUIRED_RULINGS) <= set(numbers), sorted(set(REQUIRED_RULINGS) - set(numbers))
    assert 28 in numbers and 105 in numbers
    assert "1.5 x nominal" in part                                              # the rule constants come from the inputs


def test_the_table_shells_are_empty_tables_for_every_registered_result():
    from es.measures import E1_KEYS
    from es.protocol import render_protocol
    from es.blocks import BLOCKS
    text = render_protocol(_full_ctx())
    shells = text[text.index("## Part G."):text.index("## Part H.")]
    titles = re.findall(r"^### (.*)$", shells, flags=re.M)
    assert [t.split(":")[0] for t in titles] == ["H-E", "H-B1 and H-B2", "PC", "H-S", "E1", "E2", "E3", "E4", "E5"]
    rows = [ln for ln in shells.splitlines() if ln.startswith("| ") and not ln.startswith("| Quantity")]
    assert all(ln.endswith("| — | — | — |") for ln in rows)
    e1 = [ln for ln in rows if re.match(r"\| (timing|level|total): ", ln)]
    assert len(e1) == 3 * len(E1_KEYS) == 18
    assert {ln.split("|")[1].strip().split(": ")[1] for ln in e1} == {"NET", "ESred", "J_0.0", "J_0.25", "J_0.5", "J_1.0"}
    assert len([ln for ln in rows if "nodes outside" in ln]) == len(BLOCKS)


def test_the_missing_knowledge_statement_is_one_marker_and_a_given_one_is_verbatim():
    from es.protocol import MISSING_KNOWLEDGE, ready_to_register, render_protocol
    assert ready_to_register(render_protocol(_full_ctx())) == [MISSING_KNOWLEDGE]
    text = "free text\n"
    assert "free text\n<!-- end knowledge statement -->" in render_protocol(_full_ctx(knowledge=text))


def test_render_reads_no_file_and_is_deterministic(monkeypatch):
    import builtins
    import pathlib

    from es.protocol import render_protocol, sealed_statuses
    sealed_statuses()                                                       # reads es.sealed_run's source, once, here
    ctx = _full_ctx()                                                       # the text of the spec is read here, not below
    first = render_protocol(ctx)

    def refuse(*a, **k):
        raise AssertionError("render_protocol read a file")
    monkeypatch.setattr(builtins, "open", refuse)
    monkeypatch.setattr(pathlib.Path, "read_text", refuse)
    monkeypatch.setattr(pathlib.Path, "read_bytes", refuse)
    assert render_protocol(ctx) == first


def test_the_renderer_accepts_a_git_record_and_says_none_otherwise():
    from es.protocol import render_protocol
    none = render_protocol(_full_ctx())
    assert "**No git commit identifies this code**" in none
    git = render_protocol(_full_ctx(git={"commit": "f" * 40, "bundle": "code.bundle", "bundle_sha256": "9" * 64,
                                         "not_in_commit": ["cache/common_panel_groups.json"]}))
    assert "f" * 40 in git and "9" * 64 in git and "cache/common_panel_groups.json" in git
    assert "No git commit identifies" not in git
    assert "f" * 40 in render_protocol(_full_ctx(git={"commit": "f" * 40}))      # the brief's minimal form


def test_the_real_inputs_render_when_the_stage_is_complete():
    from es.config import OUT_DIR
    from es.protocol import ready_to_register, render_protocol
    path = OUT_DIR / "analogs" / "protocol_inputs.json"
    doc = json.loads(path.read_text())
    if doc.get("stage") != "complete":
        pytest.skip("protocol_inputs.json is not at stage complete")
    text = render_protocol(_full_ctx(inputs=doc, knowledge="Name and date: Erce Erez\n"))
    assert ready_to_register(text) == []
    for h, f in doc["family"].items():
        if h != "notes":
            assert ("**H-E is the one confirmatory test.**" in text) == doc["family"]["H-E"]["confirmatory"]
            assert all(r in text for r in f["reasons"])
    assert f"eta = {doc['eta']:g}" in text


# ------------------------------------------------------------------------------------------------- code set and archive
def test_code_files_are_the_import_closure_of_the_sealed_path_plus_its_data_files(real_code_files):
    """Rulings 107(2) and 118: es.sealed_run, es.sealed_pull, es.registration and es.protocol and what they import, with
    es.run_analogs and es.simulate as roots of their own, and the data files those modules open: not the tests, not the
    paper or reporting modules, not outputs."""
    from es.protocol import OPENED_DATA_FILES
    files = real_code_files
    rel = [p.relative_to(DEV_ROOT).as_posix() for p in files]
    assert rel == sorted(rel) and len(rel) == len(set(rel)) and all(p.is_file() for p in files)
    assert not any("__pycache__" in r or r.endswith(".pyc") for r in rel)
    for needed in ("es/sealed_run.py", "es/sealed_pull.py", "es/registration.py", "es/protocol.py", "es/__init__.py",
                   "v2/ingest_nyiso.py", "v2/ingest_isone.py", "v2/ingest_miso.py", "v2/panel_prep.py",
                   "v2/run_decomposition.py", "src/transfer.py", "src/__init__.py", "v2/__init__.py", *OPENED_DATA_FILES):
        assert needed in rel, needed
    assert OPENED_DATA_FILES == ("v2/NODE_MAP.csv", "cache/common_panel_groups.json")
    assert not [r for r in rel if r.startswith(("tests/", "outputs/")) or r == "pytest.ini"]
    assert not [r for r in rel if re.match(r"es/(paper_\w+|run_digest|run_legacy|impl_table)\.py$", r)]
    assert "es/paper_tables.py" not in rel and "es/impl_table.py" not in rel
    assert {"es/run_analogs.py", "es/simulate.py"} <= set(rel)           # Ruling 118, C1-M1: the provenance roots are in
    assert {"es/run_labels_pr.py", "es/public_package.py"}.isdisjoint(rel)   # tools and runners that no root imports are not
    assert set(rel) <= {p.relative_to(DEV_ROOT).as_posix() for p in DEV_ROOT.glob("**/*") if p.is_file()}


def _top_level_project_imports(path):
    """Independent of the tracer: the project files named by the top-level import statements of one es module."""
    out = set()
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.ImportFrom) and node.level == 1:
            out |= {f"es/{node.module}.py"} if node.module else {f"es/{a.name}.py" for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] in ("v2", "src"):
            parts = node.module.split(".")
            if len(parts) > 1:
                out.add("/".join(parts) + ".py")
            else:
                out |= {f"{parts[0]}/{a.name}.py" for a in node.names if (DEV_ROOT / parts[0] / f"{a.name}.py").is_file()}
    return {r for r in out if (DEV_ROOT / r).is_file()}


def test_the_closure_contains_everything_the_sealed_modules_import_and_excludes_paper_tables(real_code_files):
    rel = {p.relative_to(DEV_ROOT).as_posix() for p in real_code_files}
    wanted = set()
    for name in ("sealed_run", "sealed_pull", "registration", "protocol"):
        wanted |= _top_level_project_imports(DEV_ROOT / "es" / f"{name}.py")
    assert {"es/run_economics.py", "es/bootstrap.py", "es/secondary.py", "es/settlement.py", "es/run_monitors.py",
            "es/run_inference.py", "es/cutoffs.py", "es/scores.py", "es/data.py", "v2/run_decomposition.py",
            "v2/ingest_nyiso.py"} <= wanted                         # the independent reading finds the known imports
    assert wanted <= rel, sorted(wanted - rel)
    assert "es/paper_tables.py" not in rel and not [r for r in rel if "paper_" in r]


def test_the_closure_is_closed_under_the_imports_of_its_own_files(real_code_files):
    """Every project module that any file of the set imports (at module level or in a function) is in the set."""
    from es.protocol import _project_imports
    rel = {p.relative_to(DEV_ROOT).as_posix() for p in real_code_files}
    for p in real_code_files:
        if p.suffix != ".py":
            continue
        name = p.relative_to(DEV_ROOT).with_suffix("").as_posix().replace("/", ".")
        name = name[:-len(".__init__")] if name.endswith(".__init__") else name
        for dep in _project_imports(DEV_ROOT, name, p):
            assert (DEV_ROOT / (dep.replace(".", "/") + ".py")).relative_to(DEV_ROOT).as_posix() in rel or \
                (DEV_ROOT / dep.replace(".", "/") / "__init__.py").relative_to(DEV_ROOT).as_posix() in rel, (name, dep)


def _tree_for_the_tracer(tmp_path):
    root = tmp_path / "dev"
    files = {
        "es/__init__.py": "",
        "es/sealed_run.py": ("from . import registration\nfrom .data import load\nimport v2.run\nimport numpy as np\n"
                             "import json\nfrom ..outside import nothing\n\n\ndef f():\n    from .lazy import g\n\n\n"
                             "if __name__ == '__main__':\n    from .demo import x\nelse:\n    from .other import y\n"
                             "try:\n    import src.fast\nexcept ImportError:\n    import src.slow\n"),
        "es/sealed_pull.py": "from v2 import ingest\nfrom v2.ingest import thing\nfrom v2.ingest import *\n",
        "es/registration.py": "",
        "es/protocol.py": "",
        "es/data.py": "def load():\n    pass\n",
        "es/lazy.py": "", "es/other.py": "", "es/demo.py": "",
        "es/run_analogs.py": "from . import provenance_only\nfrom .simulate import sim\n",   # a provenance root (Ruling 118)
        "es/simulate.py": "sim = 1\n", "es/provenance_only.py": "",
        "v2/__init__.py": "from . import ingest\n",
        "v2/run.py": "from .sub.deep import z\n",
        "v2/ingest.py": "thing = 1\n",
        "v2/sub/deep.py": "from .. import ingest\nfrom ... import nowhere\nz = 1\n",
        "src/__init__.py": "", "src/fast.py": "", "src/slow.py": "",
        "nowhere.py": "",                                            # a top-level module that no valid import reaches
        "v2/NODE_MAP.csv": "market,node_id\n", "cache/common_panel_groups.json": "{}\n",
    }
    for rel, text in files.items():
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
    return root


def test_the_tracer_follows_imports_the_way_python_does(tmp_path):
    from es.protocol import code_files, import_closure
    root = _tree_for_the_tracer(tmp_path)
    modules, importers = import_closure(root)
    got = {n: f.relative_to(root).as_posix() for n, f in modules.items()}
    assert set(got.values()) == {
        "es/__init__.py", "es/sealed_run.py", "es/sealed_pull.py", "es/registration.py", "es/protocol.py", "es/data.py",
        "es/lazy.py", "es/other.py",                              # a function-level import and an else branch: traced
        "es/run_analogs.py", "es/simulate.py", "es/provenance_only.py",   # the provenance roots and what only they import
        "v2/__init__.py", "v2/run.py", "v2/ingest.py", "v2/sub/deep.py",   # package __init__, a relative import of a package
        "src/__init__.py", "src/fast.py", "src/slow.py"}           # both arms of a try; numpy and json are not project modules
    assert "es.demo" not in got                                    # under if __name__ == "__main__": never run on import
    assert "nowhere" not in got                                    # "from ... import nowhere" climbs above the top package
    assert importers["es.lazy"] == ["es.sealed_run"] and "es.sealed_run" in importers["es.data"]
    assert importers["v2.ingest"] == ["es.sealed_pull", "v2", "v2.sub.deep"]
    assert importers["es.provenance_only"] == ["es.run_analogs"] and importers["es.simulate"] == ["es.run_analogs"]
    from es.protocol import SEALED_ROOTS
    sealed_only, _ = import_closure(root, roots=SEALED_ROOTS)          # without the provenance roots they are not there
    assert not {"es.run_analogs", "es.simulate", "es.provenance_only"} & set(sealed_only)
    # the files of the set: the closure and the two data files
    rel = [p.relative_to(root).as_posix() for p in code_files(root)]
    assert rel == sorted(set(got.values()) | {"v2/NODE_MAP.csv", "cache/common_panel_groups.json"})
    (root / "v2" / "NODE_MAP.csv").unlink()
    with pytest.raises(FileNotFoundError, match="v2/NODE_MAP.csv"):
        code_files(root)


def test_a_sealed_path_that_imports_a_paper_module_is_refused_and_reported(tmp_path):
    from es.protocol import code_files
    root = _tree_for_the_tracer(tmp_path)
    (root / "es" / "paper_tables.py").write_text("")
    (root / "es" / "data.py").write_text("def load():\n    from .paper_tables import table\n    return table\n")
    with pytest.raises(RuntimeError, match=r"es/paper_tables\.py \(imported by es\.data\)") as exc:
        code_files(root)
    assert "Ruling 107(2)" in str(exc.value) and "report it" in str(exc.value)
    for module in ("run_digest", "run_legacy", "impl_table"):
        (root / "es" / "data.py").write_text(f"import es.{module}\n")
        (root / "es" / f"{module}.py").write_text("")
        with pytest.raises(RuntimeError, match=f"es/{module}.py"):
            code_files(root)
    (root / "es" / "data.py").write_text("")
    (root / "tests").mkdir(exist_ok=True)
    (root / "tests" / "helper.py").write_text("")
    (root / "es" / "lazy.py").write_text("import tests.helper\n")
    with pytest.raises(RuntimeError, match="tests/helper.py"):
        code_files(root)
    (root / "es" / "lazy.py").write_text("")
    (root / "es" / "provenance_only.py").write_text("from . import paper_tables\n")        # a provenance root's import too
    with pytest.raises(RuntimeError, match=r"es/paper_tables\.py \(imported by es\.provenance_only\)"):
        code_files(root)


def test_a_root_that_is_not_a_project_module_is_refused(tmp_path):
    from es.protocol import import_closure
    root = _tree_for_the_tracer(tmp_path)
    with pytest.raises(ValueError, match="es.nothing is not a module"):
        import_closure(root, roots=("es.nothing",))


def test_every_module_the_sealed_modules_load_is_in_the_static_closure(real_code_files):
    """What importing the roots actually loads, in a clean interpreter, is a subset of the static closure: the closure
    never misses a module that python imports. (It may hold more: a lazy import inside a function is traced, not run.)"""
    import subprocess
    import sys
    probe = (
        "import json, sys\n"
        "from pathlib import Path\n"
        "import es.sealed_run, es.sealed_pull, es.registration, es.protocol\n"
        "root = Path.cwd().resolve()\n"
        "print('LOADED ' + json.dumps(sorted({Path(m.__file__).resolve().relative_to(root).as_posix() "
        "for m in list(sys.modules.values()) if getattr(m, '__file__', None) "
        "and Path(m.__file__).resolve().is_relative_to(root)})))\n")
    done = subprocess.run([sys.executable, "-c", probe], cwd=DEV_ROOT, capture_output=True, text=True,
                          env={**__import__("os").environ, "DART_PROJECT_ROOT": str(DEV_ROOT.parent),
                               "PYTHONPATH": str(DEV_ROOT)})
    assert done.returncode == 0, done.stderr[-1500:]
    loaded = set(json.loads(next(ln for ln in done.stdout.splitlines() if ln.startswith("LOADED "))[len("LOADED "):]))
    listed = {p.relative_to(DEV_ROOT).as_posix() for p in real_code_files}
    assert {"es/sealed_run.py", "es/protocol.py", "v2/run_decomposition.py"} <= loaded
    assert loaded <= listed, sorted(loaded - listed)
    assert not [r for r in loaded if re.match(r"es/(paper_\w+|run_digest|run_legacy|impl_table)\.py$", r)]
    provenance = {"es/run_analogs.py", "es/simulate.py"}                  # roots of the registered set, never loaded by the path
    assert provenance <= listed and not provenance & loaded


_READERS = {"open", "read_csv", "read_json", "read_parquet", "read_text", "read_bytes", "load", "read_table",
            "ParquetFile", "read_excel", "loadtxt", "genfromtxt", "urlopen", "glob"}
_FILE_NAME = re.compile(r"[\w.-]+\.(csv|json|ini|txt|parquet|yaml|yml|toml|cfg)")
#: Fixed file names that the closure's modules pass to a reader and that are not code or configuration of the sealed path.
NOT_CODE = {"results.json": "a record written by run_sealed, read back by Test B",
            "PULL_LOG.txt": "the log that pull_sealed writes", "RUN_LOG.txt": "the log that run_sealed writes"}


def _fixed_names_read(path):
    """(line, name) for each string constant that looks like a file name in the arguments of a call that reads a file,
    outside `if __name__ == "__main__":` blocks and outside functions named main (script entry points never run on the
    sealed path). Names built from variables are not seen: they are paths given at run time (data roots, pull folders)."""
    from es.protocol import _is_main_guard
    out = []

    def visit(node):
        if isinstance(node, ast.If) and _is_main_guard(node.test):
            for n in node.orelse:
                visit(n)
            return
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            return
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else fn.id if isinstance(fn, ast.Name) else None
            if name in _READERS:
                for arg in [*node.args, *[k.value for k in node.keywords]]:
                    out.extend((node.lineno, c.value) for c in ast.walk(arg)
                               if isinstance(c, ast.Constant) and isinstance(c.value, str) and _FILE_NAME.fullmatch(c.value))
        for child in ast.iter_child_nodes(node):
            visit(child)
    visit(ast.parse(Path(path).read_text()))
    return out


def test_the_data_files_the_sealed_modules_open_by_a_fixed_name_are_listed_or_explained(real_code_files):
    """Read off the closure's modules (Ruling 107(2): "grep for open and read_csv"): every fixed file name that a module
    of the closure passes to a reader is a listed data file or a documented non-code file."""
    from es.protocol import OPENED_DATA_FILES
    listed = {Path(f).name for f in OPENED_DATA_FILES}
    seen = {}
    for p in real_code_files:
        if p.suffix == ".py":
            for line, name in _fixed_names_read(p):
                seen.setdefault(name, []).append(f"{p.relative_to(DEV_ROOT).as_posix()}:{line}")
    assert {"NODE_MAP.csv", "common_panel_groups.json"} <= set(seen), seen          # the scan finds what the list holds
    unexplained = {n: where for n, where in seen.items() if n not in listed and n not in NOT_CODE}
    assert not unexplained, f"a file the closure opens by name is neither listed nor explained: {unexplained}"
    assert "v2/run_decomposition.py:36" in seen["NODE_MAP.csv"] or any("run_decomposition" in w for w in seen["NODE_MAP.csv"])
    assert any("cell_builder" in w for w in seen["common_panel_groups.json"])


def test_the_manifest_lists_every_file_and_code_departures_reads_it(tmp_path):
    from es.protocol import manifest_text
    from es.registration import sha256_file
    from es.sealed_run import code_departures
    root = tmp_path / "dev"
    (root / "es").mkdir(parents=True)
    a, b = root / "es" / "a.py", root / "pytest.ini"
    a.write_text("x = 1\n")
    b.write_text("[pytest]\n")
    text = manifest_text(root, [b, a])
    assert text == f"{sha256_file(a)}  es/a.py\n{sha256_file(b)}  pytest.ini\n"
    manifest = tmp_path / "CODE_MANIFEST.sha256"
    manifest.write_text(text)
    assert code_departures(manifest, root) == []
    a.write_text("x = 2\n")
    assert [d["file"] for d in code_departures(manifest, root)] == ["es/a.py"]
    a.unlink()
    assert code_departures(manifest, root)[0]["current"] is None


def test_the_archive_holds_exactly_the_manifest_files_with_their_bytes(tmp_path):
    from es.protocol import build_archive, manifest_text
    root = tmp_path / "dev"
    (root / "es").mkdir(parents=True)
    files = [root / "es" / "a.py", root / "es" / "b.py", root / "pytest.ini"]
    for i, f in enumerate(files):
        f.write_text(f"content {i}\n")
    build_archive(root, files, tmp_path / "x.tar.gz")
    with gzip.open(tmp_path / "x.tar.gz", "rb") as gz, tarfile.open(fileobj=io.BytesIO(gz.read())) as tar:
        got = {m.name: tar.extractfile(m).read() for m in tar.getmembers()}
    assert got == {f.relative_to(root).as_posix(): f.read_bytes() for f in files}
    assert [ln.split("  ")[1] for ln in manifest_text(root, files).splitlines()] == sorted(got)


def test_code_summary_counts_the_files_by_folder_and_lists_them_in_order(tmp_path):
    from es.protocol import code_summary
    root = tmp_path
    names = ["es/b.py", "es/a.py", "v2/x.py", "v2/NODE_MAP.csv", "cache/common_panel_groups.json", "run_a.py"]
    summary = code_summary(root, [root / n for n in names])
    assert summary["files"] == sorted(names) and summary["n_files"] == 6
    assert summary["groups"] == [("(top level)", 1), ("cache", 1), ("es", 2), ("v2", 2)]
    assert summary["data"] == ["v2/NODE_MAP.csv", "cache/common_panel_groups.json"] and summary["not_sealed"] == []
    assert code_summary(root, [root / "es/paper_tables.py", root / "tests/es/t.py"])["not_sealed"] == [
        "es/paper_tables.py", "tests/es/t.py"]


def test_the_protocol_lists_the_closure_with_the_method_and_what_is_left_out(real_code_files):
    from es.protocol import code_summary, render_protocol
    code = code_summary(DEV_ROOT, real_code_files)
    text = render_protocol(_full_ctx(code=code))
    part = text[text.index("## Part F."):text.index("## Part G.")]
    assert "Rulings 107(2) and 118" in part and "transitive import closure of the sealed path" in part
    for root in ("es.sealed_run", "es.sealed_pull", "es.registration", "es.protocol", "es.run_analogs", "es.simulate"):
        assert f"`{root}`" in part
    assert "can regenerate `protocol_inputs.json` from the development data" in part and "Seat C1, C1-M1" in part
    assert "for example `run_labels_pr` and the packaging code" in part and "for example `run_analogs`" not in part
    assert "syntax tree" in part and "`if __name__ == \"__main__\":`" in part and "`__init__.py`" in part
    assert "`v2/NODE_MAP.csv`" in part and "`cache/common_panel_groups.json`" in part
    assert f"The manifest lists {code['n_files']} files" in part
    for f in code["files"]:
        assert f"- `{f}`" in part, f
    assert "- `es/paper_tables.py`" not in part and "- `tests/" not in part and "- `pytest.ini`" not in part
    assert "`paper_*`" in part and "`run_digest`" in part and "`run_legacy`" in part and "`impl_table`" in part
    assert "Editing them is therefore not a departure" in part


# ------------------------------------------------------------------------------------------------------ versions, git
def test_versions_are_captured_live_and_meet_the_pins_of_the_spec():
    from es.protocol import capture_versions, check_pins, pinned_versions
    v = capture_versions()
    assert list(v) == ["python", "numpy", "pandas", "pyarrow", "xgboost", "scikit-learn", "scipy", "statsmodels",
                       "matplotlib"]
    assert all(re.fullmatch(r"\d+(\.\d+)+.*", x) for x in v.values())
    assert pinned_versions(SPEC.read_text()) == {"python": "3.12.7", "pandas": "2.2.2", "xgboost": "3.0.2",
                                                 "scikit-learn": "1.5.1", "pyarrow": "16.1.0"}
    assert check_pins(SPEC.read_text(), v) == []
    assert check_pins(SPEC.read_text(), {**v, "pandas": "2.3.0"}) == ["pandas 2.3.0 is running, but spec section 6.4 pins 2.2.2"]


def _git(path, *args):
    subprocess.run(["git", "-C", str(path), "-c", "user.name=t", "-c", "user.email=t@example.com",
                    "-c", "commit.gpgsign=false", *args], check=True, capture_output=True)


def test_git_state_is_none_without_a_repository_or_with_a_dirty_tree(tmp_path):
    from es.protocol import git_state
    root = tmp_path / "dev"
    root.mkdir()
    (root / "a.py").write_text("x = 1\n")
    assert git_state(root, [root / "a.py"], tmp_path / "code.bundle") is None          # not a repository
    _git(root, "init", "-q")
    assert git_state(root, [root / "a.py"], tmp_path / "code.bundle") is None          # untracked file
    _git(root, "add", "a.py")
    assert git_state(root, [root / "a.py"], tmp_path / "code.bundle") is None          # staged, not committed
    _git(root, "commit", "-q", "-m", "c")
    (root / "a.py").write_text("x = 2\n")
    assert git_state(root, [root / "a.py"], tmp_path / "code.bundle") is None          # modified
    assert not (tmp_path / "code.bundle").exists()


def test_git_state_records_the_commit_and_writes_a_bundle_when_clean(tmp_path):
    from es.protocol import git_state
    from es.registration import sha256_file
    root = tmp_path / "dev"
    (root / "cache").mkdir(parents=True)
    (root / "a.py").write_text("x = 1\n")
    (root / "cache" / "g.json").write_text("{}\n")
    (root / ".gitignore").write_text("cache/\n")
    _git(root, "init", "-q")
    _git(root, "add", "a.py", ".gitignore")
    _git(root, "commit", "-q", "-m", "c")
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    state = git_state(root, [root / "a.py", root / "cache" / "g.json"], tmp_path / "out" / "code.bundle"
                      if (tmp_path / "out").mkdir() is None else None)
    assert state["commit"] == head and state["bundle"] == "code.bundle"
    assert state["not_in_commit"] == ["cache/g.json"]                      # ignored by git, so absent from the commit
    bundle = tmp_path / "out" / "code.bundle"
    assert bundle.is_file() and state["bundle_sha256"] == sha256_file(bundle)       # an absolute path outside the repo
    assert subprocess.run(["git", "bundle", "verify", str(bundle)], cwd=root, capture_output=True).returncode == 0


# ------------------------------------------------------------------------------------------------------ the template
RELEASE = "https://github.com/ee2625/dart-spread/releases/tag/v1.0"
#: The lines a record keeps in each of the three kinds (Ruling 140); the OSF record may keep one identifier or both.
THREE_RECORDS = {"osf (both identifiers)": {"osf_doi": "10.17605/OSF.IO/ABCDE", "osf_registration": "https://osf.io/abcde/"},
                 "osf (address only)": {"osf_registration": "https://osf.io/abcde/"},
                 "public release": {"public_record": RELEASE},
                 "local record": {"local_record": "author-statement-2026-10-26"}}


def _filled_template(template, kind, sha="ab" * 32, utc="2026-10-24T18:30:00Z"):
    """The template as the author leaves it for one kind of record: protocol_sha256, registered_utc and the lines of that
    kind filled in, the whole line of every other identifier and kind deleted, the prose, the guidance and the embargo line
    as they ship."""
    from es.registration import _GATE_KEYS
    values = {"protocol_sha256": sha, "registered_utc": utc, **THREE_RECORDS[kind]}
    lines = []
    for ln in template.splitlines():
        key = ln.split(":")[0]
        if key not in _GATE_KEYS:
            lines.append(ln)
        elif key in values:
            lines.append(f"{key}: {values[key]}")                 # a line of the gate that this kind keeps
    return "\n".join(lines) + "\n"


def test_the_registration_template_is_refused_until_filled(tmp_path):
    from es.protocol import registration_template
    from es.registration import _GATE_KEYS, RegistrationError, parse_registration
    template = registration_template()
    reg = tmp_path / "REGISTRATION.md"
    reg.write_text(template)
    with pytest.raises(RegistrationError):
        parse_registration(reg)                                           # every value is a <paste ...> placeholder
    keys = [ln.split(":")[0] for ln in template.splitlines() if re.match(r"^(protocol_sha256|osf_doi|osf_registration|"
                                                                          r"public_record|local_record|registered_utc|"
                                                                          r"embargo_until):", ln)]
    assert keys == ["protocol_sha256", "osf_doi", "osf_registration", "public_record", "local_record", "registered_utc",
                    "embargo_until"]
    pairs = [ln for ln in template.splitlines() if re.match(r"^[a-z0-9_]+: ", ln)]          # protocol_sha256 has digits
    assert [ln.split(":")[0] for ln in pairs if "<paste" in ln] == ["protocol_sha256", "osf_doi", "osf_registration",
                                                                     "public_record", "local_record", "registered_utc"]
    assert set(ln.split(":")[0] for ln in pairs if "<paste" in ln) == set(_GATE_KEYS)       # the six lines of the gate
    assert [ln for ln in pairs if "<paste" not in ln] == ["embargo_until: none"]            # Ruling 122, C2-2: pre-filled
    assert template.splitlines()[0] == "# Registration record: template" and "OSF registration record" not in template
    # a filled copy of each of the three kinds is accepted, with the other kinds' fields None
    for kind, kept in THREE_RECORDS.items():
        reg.write_text(_filled_template(template, kind))
        rec = parse_registration(reg)
        assert rec["protocol_sha256"] == "ab" * 32 and rec["registered_utc"] == "2026-10-24T18:30:00Z", kind
        assert rec["embargo_until"] == "none", kind                            # it passes through as any other key
        for key in ("osf_doi", "osf_registration", "public_record", "local_record"):
            assert (rec[key] is not None) == (key in kept), (kind, key)
        reg.write_text(_filled_template(template, kind).replace("\nembargo_until: none", ""))
        assert parse_registration(reg)["registered_utc"] == "2026-10-24T18:30:00Z"            # and it needs no embargo line
    # a line of the template left in beside a filled kind is refused, whichever it is: the template's placeholder, or a
    # second kind filled in
    filled = _filled_template(template, "local record")
    for left in ("osf_doi", "osf_registration", "public_record"):
        line = next(ln for ln in template.splitlines() if ln.startswith(f"{left}:"))
        reg.write_text(filled + line + "\n")
        with pytest.raises(RegistrationError, match=left):
            parse_registration(reg)
    reg.write_text(filled + f"public_record: {RELEASE}\n")
    with pytest.raises(RegistrationError, match="more than one kind"):
        parse_registration(reg)
    reg.write_text(filled + "osf_registration: https://osf.io/abcde/\n")
    with pytest.raises(RegistrationError, match="more than one kind"):
        parse_registration(reg)
    # a gate line of the kind in use left as a placeholder is refused
    reg.write_text(filled.replace("local_record: author-statement-2026-10-26",
                                  next(ln for ln in template.splitlines() if ln.startswith("local_record:"))))
    with pytest.raises(RegistrationError, match="local_record"):
        parse_registration(reg)


# ------------------------------------------------------------------------------------------------------- the inputs
def test_load_inputs_refuses_anything_but_a_complete_stage_with_the_family(tmp_path):
    from es.protocol import load_inputs
    good = tmp_path / "ok.json"
    good.write_text(json.dumps(_inputs(), indent=2) + "\n")
    raw, doc = load_inputs(good)
    assert raw == good.read_bytes() and doc["stage"] == "complete"

    def bad(change, match):
        d = _inputs()
        change(d)
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(d))
        with pytest.raises(RuntimeError, match=match):
            load_inputs(p)

    bad(lambda d: d.update(stage="analogs"), "stage 'analogs'")
    bad(lambda d: d.pop("stage"), "not 'complete'")
    bad(lambda d: d.update(family=None), "family lacks")
    bad(lambda d: d["family"].pop("H-B2"), r"family lacks \['H-B2'\]")
    bad(lambda d: d.pop("rule"), r"lacks \['rule'\]")
    (tmp_path / "bad.json").write_text("{not json")
    with pytest.raises(RuntimeError, match="not valid JSON"):
        load_inputs(tmp_path / "bad.json")
    (tmp_path / "bad.json").write_text("[1, 2]")
    with pytest.raises(RuntimeError, match="JSON object"):
        load_inputs(tmp_path / "bad.json")
    with pytest.raises(RuntimeError, match="does not exist"):
        load_inputs(tmp_path / "missing.json")


# ------------------------------------------------------------------------------------------------------------ main()
def _cutoffs():
    rows = [{"analysis": a, "block": b, "market": "NYISO", "node": n, "n_calib": 2208, "tau_p1": 0.5 + i,
             "tau_p2_season": 0.4, "tau_p2_all": 0.6, "p2_season_above_p1": i % 2 == 0, "p2_all_above_p1": True}
            for a in ("SA", "SAF", "SB", "SBF") for b in ("G1_24", "DART3") for i, n in enumerate(("N.Y.C.", "CAPITL"))]
    rows[0]["tau_p1"] = float("nan")
    return pd.DataFrame(rows)


def _tree(tmp_path):
    """A small dev root, an empty sealed folder and a complete inputs file; calling it again changes nothing."""
    root = tmp_path / "dev"
    (root / "es").mkdir(parents=True, exist_ok=True)
    for path, text in ((root / "es" / "a.py", "x = 1\n"), (root / "es" / "b.py", "y = 2\n"),
                       (root / "pytest.ini", "[pytest]\n"),
                       (tmp_path / "protocol_inputs.json", json.dumps(_inputs(), indent=2) + "\n")):
        if not path.exists():
            path.write_text(text)
    sealed = tmp_path / "sealed"
    sealed.mkdir(exist_ok=True)
    return root, sealed, tmp_path / "protocol_inputs.json"


_PINNED = {"python": "3.12.7", "numpy": "1.26.4", "pandas": "2.2.2", "pyarrow": "16.1.0", "xgboost": "3.0.2",
           "scikit-learn": "1.5.1", "scipy": "1.13.1", "statsmodels": "0.14.2", "matplotlib": "3.9.2"}


def _run_main(tmp_path, **kw):
    from es.protocol import main
    root, sealed, inputs = _tree(tmp_path)
    files = [root / "es" / "a.py", root / "es" / "b.py", root / "pytest.ini"]
    args = dict(sealed_dir=sealed, dev_root=root, inputs_source=inputs, cutoffs=_cutoffs, files=files,
                spec_path=SPEC, forensics_path=FORENSICS, date="2026-10-23",
                versions={"python": "3.12.7", "numpy": "1.26.4", "pandas": "2.2.2", "pyarrow": "16.1.0",
                          "xgboost": "3.0.2", "scikit-learn": "1.5.1", "scipy": "1.13.1", "statsmodels": "0.14.2",
                          "matplotlib": "3.9.2"})
    args.update(kw)
    return main(**args), root, sealed


def test_main_writes_the_package_in_the_given_folder_and_it_verifies(tmp_path, capsys):
    from es.registration import sha256_file, verify_bound
    from es.protocol import ready_to_register
    from es.sealed_run import code_departures
    result, root, sealed = _run_main(tmp_path)
    names = sorted(p.name for p in sealed.iterdir())
    assert names == ["CODE_MANIFEST.sha256", "ENV_PINS.json", "PROTOCOL.md", "PROTOCOL.sha256", "REGISTRATION.template.md",
                     "code_archive.tar.gz", "pre_pull_cutoffs.csv", "protocol_inputs.json"]
    assert (sealed / "protocol_inputs.json").read_bytes() == (tmp_path / "protocol_inputs.json").read_bytes()
    assert (sealed / "PROTOCOL.sha256").read_text() == f"{sha256_file(sealed / 'PROTOCOL.md')}  PROTOCOL.md\n"
    assert result["protocol_sha256"] == sha256_file(sealed / "PROTOCOL.md")
    bound = verify_bound(sealed / "PROTOCOL.md", [sealed / "pre_pull_cutoffs.csv", sealed / "protocol_inputs.json",
                                                  sealed / "CODE_MANIFEST.sha256", sealed / "ENV_PINS.json"])
    assert len(bound) == 4
    pins = json.loads((sealed / "ENV_PINS.json").read_text())
    assert pins == {"executable": sys.executable, "versions": {k: _PINNED[k] for k in _PINNED if k != "matplotlib"}}
    assert (sealed / "ENV_PINS.json").read_text().endswith("}\n")
    assert code_departures(sealed / "CODE_MANIFEST.sha256", root) == []
    text = (sealed / "PROTOCOL.md").read_text()
    assert f"`code_archive.tar.gz` SHA-256: `{sha256_file(sealed / 'code_archive.tar.gz')}`" in text
    assert f"`ENV_PINS.json` SHA-256: `{sha256_file(sealed / 'ENV_PINS.json')}`" in text
    assert "`CODE_MANIFEST.sha256`, `ENV_PINS.json` and `code_archive.tar.gz` are bound to the protocol by the SHA-256 " \
           "lines above" in text
    assert "The manifest lists 3 files" in text and "No git commit identifies this code" in text
    assert "| SA | G1_24 | 2 | 1 | 2 | 1 |" in text                         # the cutoff summary: one P1 cutoff is NaN
    assert not (sealed / "code.bundle").exists() and not any(p.suffix == ".tmp" for p in sealed.iterdir())
    # no knowledge file in the folder: the statement is the one marker; the printed line says so
    assert result["markers"] == ready_to_register(text) and len(result["markers"]) == 1
    assert "ready_to_register: 1 author marker(s) remain" in capsys.readouterr().out
    csv = pd.read_csv(sealed / "pre_pull_cutoffs.csv")
    assert list(csv.columns) == list(_cutoffs().columns) and len(csv) == 16 and "\r" not in (sealed / "pre_pull_cutoffs.csv").read_text()


def test_main_reads_the_filled_knowledge_statement_else_the_template(tmp_path):
    root, sealed, inputs = _tree(tmp_path)
    template = (DEV_ROOT.parent / "docs/sealed/KNOWLEDGE_STATEMENT_TEMPLATE.md").read_text()
    (sealed / "KNOWLEDGE_STATEMENT_TEMPLATE.md").write_text(template)
    result, _, _ = _run_main(tmp_path)
    from es.protocol import author_markers
    assert len(author_markers((sealed / "PROTOCOL.md").read_text())) == 4   # the template's own markers block registration
    assert result["markers"][:4] == author_markers((sealed / "PROTOCOL.md").read_text()) and len(result["markers"]) > 4
    assert "`docs/sealed/KNOWLEDGE_STATEMENT_TEMPLATE.md`" in (sealed / "PROTOCOL.md").read_text()
    filled = "# Knowledge statement\n\nI know nothing after 30 June 2026.\n\nErce Erez, 2026-10-24\n"
    (sealed / "KNOWLEDGE_STATEMENT.md").write_text(filled)
    result, _, _ = _run_main(tmp_path)
    text = (sealed / "PROTOCOL.md").read_text()
    assert result["markers"] == [] and filled.replace("# Knowledge statement", "### Knowledge statement").rstrip("\n") in text
    assert "KNOWLEDGE_STATEMENT.md`" in text


def test_main_is_deterministic_for_the_four_bound_files_and_the_protocol(tmp_path):
    first, _, sealed = _run_main(tmp_path)
    hashes = {p.name: p.read_bytes() for p in sealed.iterdir()}
    second, _, _ = _run_main(tmp_path)
    assert first["sha"] == second["sha"] and first["protocol_sha256"] == second["protocol_sha256"]
    assert {p.name: p.read_bytes() for p in sealed.iterdir()} == hashes


def test_main_refuses_once_the_registration_exists_and_touches_nothing(tmp_path):
    from es.protocol import main
    root, sealed, inputs = _tree(tmp_path)
    (sealed / "REGISTRATION.md").write_text("protocol_sha256: " + "a" * 64 + "\n")
    (sealed / "PROTOCOL.md").write_text("registered text\n")
    before = {p.name: p.read_bytes() for p in sealed.iterdir()}

    def must_not_run():
        raise AssertionError("the cutoff fit started after the registration")
    with pytest.raises(RuntimeError, match="immutable"):
        main(sealed_dir=sealed, dev_root=root, inputs_source=inputs, cutoffs=must_not_run, files=[root / "pytest.ini"],
             spec_path=SPEC, forensics_path=FORENSICS)
    assert {p.name: p.read_bytes() for p in sealed.iterdir()} == before


def test_main_refuses_an_incomplete_inputs_file_and_a_wrong_interpreter_before_it_fits_anything(tmp_path):
    from es.protocol import main
    root, sealed, inputs = _tree(tmp_path)
    doc = _inputs()
    doc["stage"] = "analogs"
    inputs.write_text(json.dumps(doc))

    def must_not_run():
        raise AssertionError("the cutoff fit started")
    kw = dict(sealed_dir=sealed, dev_root=root, cutoffs=must_not_run, files=[root / "pytest.ini"], spec_path=SPEC,
              forensics_path=FORENSICS)
    with pytest.raises(RuntimeError, match="not 'complete'"):
        main(inputs_source=inputs, **kw)
    inputs.write_text(json.dumps(_inputs()))
    with pytest.raises(RuntimeError, match="pins of spec section 6.4"):
        main(inputs_source=inputs, versions={"python": "3.12.7", "pandas": "9.9.9"}, **kw)
    assert list(sealed.iterdir()) == []


def test_main_removes_a_stale_bundle_when_git_is_not_clean(tmp_path):
    root, sealed, inputs = _tree(tmp_path)
    (sealed / "code.bundle").write_bytes(b"stale")
    _run_main(tmp_path)
    assert not (sealed / "code.bundle").exists()


def test_main_with_a_clean_repository_records_the_commit_and_the_bundle(tmp_path):
    root, sealed, inputs = _tree(tmp_path)
    _git(root, "init", "-q")
    _git(root, "add", "es/a.py", "es/b.py", "pytest.ini")
    _git(root, "commit", "-q", "-m", "c")
    result, _, _ = _run_main(tmp_path)
    assert result["git"]["commit"] and (sealed / "code.bundle").is_file()
    text = (sealed / "PROTOCOL.md").read_text()
    assert result["git"]["commit"] in text and "No git commit identifies" not in text
    assert not any(p.suffix == ".tmp" for p in sealed.iterdir())
    assert re.search(r"SHA-256 `[0-9a-f]{64}`", text)


def test_main_defaults_point_at_the_files_the_gate_and_the_sealed_run_read(monkeypatch, tmp_path):
    """Without a sealed_dir, main() writes to es.registration's own paths (as they are when it is called)."""
    import es.registration as registration
    from es.protocol import _Out
    sealed = tmp_path / "docs_sealed"
    monkeypatch.setattr(registration, "SEALED_DIR", sealed)
    monkeypatch.setattr(registration, "PROTOCOL_PATH", sealed / "PROTOCOL.md")
    monkeypatch.setattr(registration, "REGISTRATION_PATH", sealed / "REGISTRATION.md")
    monkeypatch.setattr(registration, "CUTOFFS_PATH", sealed / "pre_pull_cutoffs.csv")
    monkeypatch.setattr(registration, "INPUTS_PATH", sealed / "protocol_inputs.json")
    out = _Out()
    assert (out.protocol, out.registration, out.cutoffs, out.inputs) == (
        registration.PROTOCOL_PATH, registration.REGISTRATION_PATH, registration.CUTOFFS_PATH, registration.INPUTS_PATH)
    from es.sealed_run import SealedConfig
    assert out.manifest == SealedConfig(test_id="A", rolling=None, frozen=None).code_manifest      # the run's default
    real = _Out()
    assert real.dir == sealed                                                                      # patched, as the tests do


def test_the_eta_line_gives_the_grid_and_whether_the_choice_is_its_top():
    from es.protocol import render_protocol
    text = render_protocol(_full_ctx())
    assert "eta = 0.05: the step of the self-correcting cutoff P5" in text and "{0.005, 0.01, 0.02, 0.05}" in text
    assert "The choice is the largest value of the grid" in text
    mid = _full_ctx()
    mid["inputs"]["eta"] = 0.02
    assert "eta = 0.02" in render_protocol(mid) and "largest value of the grid" not in render_protocol(mid)
    assert "rho_star = 0.05" in text and "{0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9}" in text


def test_the_unfilled_template_is_called_unfilled_in_the_protocol():
    from es.protocol import ready_to_register, render_protocol
    template = (DEV_ROOT.parent / "docs/sealed/KNOWLEDGE_STATEMENT_TEMPLATE.md").read_text()
    text = render_protocol(_full_ctx(knowledge=template, knowledge_source="docs/sealed/KNOWLEDGE_STATEMENT_TEMPLATE.md"))
    from es.protocol import author_markers
    items = ready_to_register(text)
    assert "This is the unfilled template" in text and len(author_markers(text)) == 4
    assert len(items) == 10 and [i.split(":")[0] for i in items] == ["[AUTHOR"] * 4 + ["[ITEM 1"] * 4 + ["[GUIDANCE"] * 2
    filled = render_protocol(_full_ctx(knowledge="Name and date: Erce Erez\n",
                                       knowledge_source="docs/sealed/KNOWLEDGE_STATEMENT.md"))
    assert "unfilled template" not in filled and "`docs/sealed/KNOWLEDGE_STATEMENT.md`" in filled


def test_the_no_drift_context_is_a_descriptive_table_of_the_simulation_rows():
    from es.protocol import render_protocol
    ctx = _full_ctx()
    ctx["inputs"]["simulation"].append({"test": "H-B size", "law": "DART3", "scenario": "no_drift", "persistence": "fit",
                                        "n_sim": 300, "seed": 5, "reject_0.0166667": 0.06, "reject_0.05": 0.1,
                                        "nodrift_fail_q95_0.5_2": 9.0, "nodrift_fail_q95_0.67_1.5": 17.0,
                                        "nodrift_fail_q95_0.8_1.25": 25.0})
    text = render_protocol(ctx)
    assert "| DART3 | fit | 300 | 9 | 17 | 25 |" in text and "Nodes outside [0.67, 1.5]" in text
    assert "not a decision threshold" in text and "from 2025-07 to 2026-06 (spec section 6.2(b)" in text
    assert "the simulation has no logistic-regression no-drift context" not in render_protocol(_full_ctx())   # no row: no table


def test_the_size_study_text_names_the_demoted_tests_and_the_laws_each_reads():
    from es.protocol import render_protocol
    ctx = _full_ctx()
    ctx["inputs"]["rule"]["laws"] = {"H-B1": ["DART3", "LEV6"], "H-B2": ["DART3", "ROLL_DISP9"]}
    text = render_protocol(ctx)
    assert "**Why H-B1 and H-B2 are estimands.**" in text
    assert "H-B1 reads the laws DART3, LEV6; H-B2 reads the laws DART3, ROLL_DISP9" in text
    ctx["inputs"]["family"]["H-B2"].update(confirmatory=True, reasons=[])
    assert "**Why H-B1 is an estimand.**" in render_protocol(ctx)
    for h in ("H-B1", "H-B2"):
        ctx["inputs"]["family"][h].update(confirmatory=True, reasons=[])
    assert "Why H-B1" not in render_protocol(ctx)


def test_main_refuses_to_publish_when_the_code_changed_during_the_fit(tmp_path):
    from es.protocol import main
    root, sealed, inputs = _tree(tmp_path)
    files = [root / "es" / "a.py", root / "es" / "b.py", root / "pytest.ini"]

    def fit_while_somebody_edits():
        (root / "es" / "a.py").write_text("x = 99\n")                  # another agent edits a registered file mid-run
        return _cutoffs()
    with pytest.raises(RuntimeError, match=r"code changed while.*es/a\.py"):
        main(sealed_dir=sealed, dev_root=root, inputs_source=inputs, cutoffs=fit_while_somebody_edits, files=files,
             spec_path=SPEC, forensics_path=FORENSICS, versions=_PINNED)
    assert list(sealed.iterdir()) == []                                # nothing written, no temporary file left


def test_main_refuses_when_a_new_code_file_appears_during_the_fit(tmp_path, monkeypatch):
    import es.protocol as protocol
    root, sealed, inputs = _tree(tmp_path)
    listed = [root / "es" / "a.py", root / "pytest.ini"]
    monkeypatch.setattr(protocol, "code_files", lambda dev_root: sorted(listed))
    def fit_then_a_new_file():
        listed.append(root / "es" / "b.py")
        return _cutoffs()
    with pytest.raises(RuntimeError, match=r"code changed while.*es/b\.py"):
        protocol.main(sealed_dir=sealed, dev_root=root, inputs_source=inputs, cutoffs=fit_then_a_new_file,
                      spec_path=SPEC, forensics_path=FORENSICS, versions=_PINNED)
    assert list(sealed.iterdir()) == []


def test_main_renders_once_before_the_fit_so_that_a_bad_spec_stops_the_run_early(tmp_path):
    from es.protocol import main
    root, sealed, inputs = _tree(tmp_path)
    bad_spec = tmp_path / "spec.md"
    bad_spec.write_text(SPEC.read_text().replace("### 6.4 Pins", "### 6.4x Pins").replace("### 6.5 Freeze", "### 6.5x Freeze"))

    def must_not_run():
        raise AssertionError("the cutoff fit started although the spec cannot be rendered")
    with pytest.raises(ValueError, match="no section headed"):
        main(sealed_dir=sealed, dev_root=root, inputs_source=inputs, cutoffs=must_not_run, files=[root / "pytest.ini"],
             spec_path=bad_spec, forensics_path=FORENSICS, versions=_PINNED)
    assert list(sealed.iterdir()) == []


# ---------------------------------------------------------------------------------------- failure handling (Ruling 104)
_VOCABULARY = {"completed": "completed", "secondary": "completed with secondary errors",
               "errors": "completed with errors", "not_run": "not run (coverage rule)"}


def test_the_run_statuses_are_read_from_the_sealed_run_module():
    from es.protocol import sealed_statuses, statuses_from_source
    assert sealed_statuses() == _VOCABULARY
    source = (DEV_ROOT / "es/sealed_run.py").read_text()
    for status in _VOCABULARY.values():
        assert f'"{status}"' in source                                      # the strings the module writes
    ok = ('"""not run yet: a docstring is not a status."""\n'
          'class R:\n    def status(self):\n        if not self.errors:\n            return "completed"\n'
          '        return "completed with secondary errors" if self.s else "completed with errors"\n'
          'results["status"] = "not run (coverage rule)"\n')
    assert statuses_from_source(ok) == _VOCABULARY
    with pytest.raises(RuntimeError, match="cannot read the run statuses"):
        statuses_from_source(ok.replace('return "completed"\n', 'return "done"\n'))              # one is missing
    with pytest.raises(RuntimeError, match="cannot read the run statuses"):
        statuses_from_source(ok + 'x = "completed with problems"\n')                           # one is ambiguous


def test_the_failure_handling_section_states_the_registered_outcomes_with_the_modules_own_words(monkeypatch):
    import es.protocol as protocol
    text = protocol.render_protocol(_full_ctx())
    part = text[text.index("### Failure handling (Rulings 104, 109, 112, 116 and 122)"):text.index("## Part E.")]
    rows = [ln for ln in part.splitlines() if ln.startswith('| "')]
    assert [ln.split("|")[1].strip() for ln in rows] == [f'"{v}"' for v in _VOCABULARY.values()]
    assert 'Status (`results["status"]` in `results.json`)' in part and "last line" not in part
    assert "An exception in the data load, in the panel build or in the computation of the coverage table aborts the run" \
           in part and "a coverage rule that is not met is no exception, but the registered NOT_RUN" in part
    assert "only secondary analyses failed" in part and "The others are reported as usual" in part
    assert "The paper reports what completed and states what failed and why" in part
    assert "`NOT_RUN.md` gives the reasons" in part and "**Coverage failure.** It means NOT_RUN, as registered" in part
    assert "**Test B before Test A (Rulings 112 and 116, N2).** Test B is refused before its lock, and nothing is spent, " \
           "unless Test A has been started: `docs/sealed/RUNS.json` holds a record of Test A, or Test A's claim file " \
           "`RUN_A.claim` exists." in part
    assert "Test B is refused too when the output folder that Test A's record names is complete (it has its " \
           "`MANIFEST.sha256`) but the record in `RUNS.json` lacks `finished_utc` or `manifest_sha256`" in part
    assert "**Test B without Test A's output (Rulings 109, 112 and 116).** Once Test A has been started, Test B runs even " \
           "if Test A's output is not available: if Test A is NOT_RUN (its coverage rule failed), or was claimed and left " \
           "no manifest and no finished record" in part
    assert "or finished without stored P1 rows, Test B runs alone. H-E is then tested on Test B's window alone and reported " \
           "as not confirmatory, because the registered pooled test is unavailable" in part
    assert "has no recorded run" not in part
    assert "Test B never depends on Test A's demoted (estimand) H-B results, and a failed Test A estimand does not block " \
           "Test B" in part
    assert "Where Test A's output is complete and its run happened, Test B pools H-E with Test A's stored P1 rows" in part
    assert "until they are added Test B is refused (it does not run alone)" in part
    assert "reads Test A as unfinished and runs alone" not in part                         # the earlier, wrong consequence
    assert "stop Test B before the lock, so that nothing is spent" in part and "reports Test B as not run, with the reason" in part
    assert "H-B estimates and p-values" not in part and "that replication needs" not in part     # Test B needs no H-B result
    assert "An interruption after the claim (Ctrl-C or a crash) spends the run, as registered: a run that started counts" in part
    assert "docs/sealed/RUNS.json" in part and "Never open the sealed pull folders" in part
    # the words follow the module: if it renames a status, so does the protocol
    monkeypatch.setattr(protocol, "sealed_statuses", lambda: {"completed": "ok", "secondary": "ok, secondary failed",
                                                              "errors": "ok, primary failed", "not_run": "not run"})
    changed = protocol.render_protocol(_full_ctx())
    assert '| "ok, secondary failed" |' in changed and "completed with secondary errors" not in changed
    assert 'its status reads "not run"' in changed


def test_the_failure_handling_matches_what_the_sealed_run_does_before_its_lock():
    """The structural facts the text leans on (Ruling 116, N3): the checks of Test A's output and of the write access come
    before the lock, the claim is taken before the output folder is made, and the run record follows the log's first lines."""
    source = (DEV_ROOT / "es/sealed_run.py").read_text()
    run = source[source.index("def run_sealed("):source.index("def _execute(")]
    run = run[run.index("    _check_test_id(cfg)"):]
    assert run.index("_check_prior(") < run.index("_check_writable(") < run.index("_take_claim(") < run.index("out.mkdir(") \
        < run.index("_log(out, f\"started") < run.index("_write_run_record(")


def test_the_run_statuses_are_accepted_as_literals_or_as_named_constants(tmp_path):
    """The controller may give sealed_run STATUS_* constants after the rehearsal: the vocabulary is read either way."""
    import importlib.util

    from es.protocol import statuses_from_module

    def load(name, text):
        path = tmp_path / f"{name}.py"
        path.write_text(text)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    literals = ('def status(e):\n    if not e:\n        return "completed"\n'
                '    return "completed with secondary errors" if e == 1 else "completed with errors"\n'
                'def run(r):\n    r["status"] = "not run (coverage rule)"\n')
    constants = ('STATUS_COMPLETED = "completed"\nSTATUS_SECONDARY = f"{STATUS_COMPLETED} with secondary errors"\n'
                 'STATUS_ERRORS = STATUS_COMPLETED + " with errors"\nSTATUS_NOT_RUN = "not run (coverage rule)"\n\n'
                 'def status(e):\n    return STATUS_COMPLETED\n')
    mapping = ('STATUSES = {"completed": "completed", "secondary": "completed with secondary errors", '
               '"errors": "completed with errors", "not_run": "not run (coverage rule)"}\n'
               'def status(e):\n    return STATUSES["completed"]\n')
    sequence = ('STATUS_ORDER = ("completed", "completed with secondary errors", "completed with errors", '
                '"not run (coverage rule)")\n')
    for name, text in (("literals", literals), ("constants", constants), ("mapping", mapping), ("sequence", sequence)):
        assert statuses_from_module(load(name, text)) == _VOCABULARY, name
    with pytest.raises(RuntimeError, match="cannot read the run statuses"):          # a constant that clashes with a literal
        statuses_from_module(load("clash", constants + 'x = "completed with errors!"\n'))
    with pytest.raises(RuntimeError, match="cannot read the run statuses"):          # a status that is not there at all
        statuses_from_module(load("missing", constants.replace('STATUS_NOT_RUN = "not run (coverage rule)"\n', "")))


def test_the_demoted_h_b_estimands_report_the_percentile_interval_with_the_basic_one_as_a_diagnostic():
    """Ruling 110: percentile primary, basic (2t - q.975, 2t - q.025) beside it as a diagnostic, neither read inferentially."""
    from es.protocol import render_protocol
    text = render_protocol(_full_ctx())
    family = text[text.index("### Registered family and decision rule"):text.index("### Pins read from the code")]
    assert "**Intervals of H-B1 and H-B2 (Ruling 110).**" in family
    assert "The primary reported interval is the percentile bootstrap interval (the `ci` field of the results)" in family
    assert "The basic interval (2t \u2212 q\u2080.\u2089\u2087\u2085, 2t \u2212 q\u2080.\u2080\u2082\u2085), the `ci_basic` field, " \
           "is shown beside it as a diagnostic" in family
    assert "The bootstrap distribution sits below the estimate t-hat mainly because the calibration days and the deployment " \
           "days are drawn independently, which re-randomizes the matched calendar composition of the two windows" in family
    assert "the median over nodes and the kink of |log10 R| add to it" in family
    assert "the median-centre pull" not in family                                  # the earlier, incomplete mechanism
    assert "Both intervals are approximate and may under-cover: the simulated bootstrap SD ratio is 0.68 to 0.83 under the " \
           "fitted persistence in the synthetic no-drift study, and the intervals may under-cover substantially under a " \
           "common drift (Ruling 118)" in family
    assert "Neither is read inferentially: the protocol makes no \"excludes zero\" claim for H-B1 and H-B2." in family
    assert "No `p` field is stored for them: the one-sided bootstrap p-value is kept as a diagnostic, under " \
           "`diagnostics.bootstrap_p_one_sided`, and is not a test." in family
    assert "The `interval_registered` field reads \"percentile (primary); basic (diagnostic)\"." in family
    assert "The basic p-value (`p`) is stored with them" not in family                        # the earlier, wrong statement
    assert "interval excludes zero" not in family and "no difference detected" not in family     # the earlier wording is gone
    assert "The H-B estimands have no inferential wording" in family
    assert "Intervals of H-B1 and H-B2" in text and "a percentile interval; for the demoted H-B estimands" in text
    # the spec's own note (copied in Part A) says the same
    assert "they report the percentile interval as primary and the basic interval as a diagnostic" in text
    # one H-B test demoted, one confirmatory: each gets its own statement
    mixed = _full_ctx()
    mixed["inputs"]["family"]["H-B2"].update(confirmatory=True, reasons=[], approximate=False)
    t2 = render_protocol(mixed)
    assert "**Intervals of H-B1 (Ruling 110).**" in t2 and "**Interval of H-B2.** A confirmatory H-B test is reported" in t2
    assert "(spec section 5 item 1, Ruling 55(f))" in t2 and "the `ci_basic` field" in t2
    # none demoted: no Ruling 110 text at all
    allc = _full_ctx()
    for h in ("H-B1", "H-B2"):
        allc["inputs"]["family"][h].update(confirmatory=True, reasons=[], approximate=False)
    assert "**Intervals of" not in render_protocol(allc)


def test_env_pins_hold_the_interpreter_and_the_versions_of_eight_libraries():
    """Ruling 109(1): sys.executable and python, numpy, pandas, pyarrow, xgboost, scikit-learn, scipy, statsmodels."""
    from es.protocol import ENV_PIN_LIBRARIES, capture_versions, env_pins, env_pins_text
    live = capture_versions()
    pins = env_pins(live)
    assert list(pins) == ["executable", "versions"] and pins["executable"] == sys.executable
    assert list(pins["versions"]) == ["python", "numpy", "pandas", "pyarrow", "xgboost", "scikit-learn", "scipy",
                                      "statsmodels"] == list(ENV_PIN_LIBRARIES)
    assert pins["versions"] == {k: live[k] for k in ENV_PIN_LIBRARIES} and "matplotlib" not in pins["versions"]
    text = env_pins_text(live)
    assert text.endswith("}\n") and json.loads(text) == pins and text == env_pins_text(dict(reversed(list(live.items()))))
    assert env_pins({k: "1" for k in ENV_PIN_LIBRARIES}, executable="/x/python")["executable"] == "/x/python"
    with pytest.raises(ValueError, match="scipy"):
        env_pins({k: "1" for k in ENV_PIN_LIBRARIES if k != "scipy"})


def test_the_protocol_binds_and_describes_the_environment_pins():
    from es.protocol import ENV_PIN_LIBRARIES, render_protocol
    text = render_protocol(_full_ctx())
    part = text[text.index("## Part F."):text.index("## Part G.")]
    assert "- `ENV_PINS.json` SHA-256: `" + "e" * 64 + "`" in part
    order = [part.index(f"- `{n}` SHA-256:") for n in ("pre_pull_cutoffs.csv", "protocol_inputs.json", "CODE_MANIFEST.sha256",
                                                      "ENV_PINS.json", "code_archive.tar.gz")]
    assert order == sorted(order)                                        # next to the other three bound files
    assert "`ENV_PINS.json` holds the path of the interpreter (`sys.executable`) and the versions of " \
           + ", ".join(ENV_PIN_LIBRARIES) in part
    assert "refuses before its lock if the running environment differs (Ruling 109(1))" in part
    assert "`CODE_MANIFEST.sha256`, `ENV_PINS.json` and `code_archive.tar.gz` are bound to the protocol by the SHA-256 " \
           "lines above" in part
    assert "the sealed pull needs the registration record" in part       # only the run verifies the bound files
    run = text[text.index("### The sealed pull and the sealed run"):text.index("### Failure handling")]
    assert "the code manifest and `ENV_PINS.json` to have the SHA-256 printed in Part F" in run
    assert "the running environment is compared with `ENV_PINS.json`" in run
    # a protocol rendered without the env hash (the brief's ctx) has no such line
    bare = render_protocol({**_full_ctx(), "sha": {k: "a" * 64 for k in ("manifest", "archive", "cutoffs", "inputs")}})
    assert "ENV_PINS.json" not in bare[bare.index("## Part F."):bare.index("## Part G.")]
    assert "ENV_PINS.json" not in bare[bare.index("### The sealed pull and the sealed run"):bare.index("### Failure handling")]


def test_the_refinements_name_the_closure_the_environment_pins_and_the_interval_ruling():
    from es.protocol import render_protocol
    text = render_protocol(_full_ctx())
    part = text[text.index("## Part E."):text.index("## Part F.")]
    assert "- **Ruling 107** (2A): The code manifest and the archive cover the import closure of the sealed path" in part
    assert "- **Ruling 109** (2A): (1) `ENV_PINS.json` binds the interpreter path" in part
    assert "Test B runs alone, with H-E on its own window and not confirmatory (Rulings 112 and 116 say when Test B is " \
           "refused instead)" in part and "has no recorded run" not in part
    assert "- **Ruling 110** (2A): Once H-B1 and H-B2 were demoted to estimands, the percentile interval is their primary" in part


# ---------------------------------------------------- Ruling 112, and the protocol against the final es.sealed_run
def test_ruling_112_is_stated_in_the_three_places_and_has_its_own_entry():
    """Test B is refused unless Test A has been started, and runs alone when Test A's output is not available (NOT_RUN, claimed
    with no manifest and no finished record, finished without stored P1 rows: Ruling 116). The family text, Failure handling and
    Ruling 109's entry say so (the earlier "NOT_RUN or has no recorded run" is gone), and Ruling 112 has an entry beside those
    of 109 and 110, with Ruling 116's after them."""
    import es.sealed_run as sealed_run
    from es.protocol import render_protocol
    text = render_protocol(_full_ctx())
    family = text[text.index("### Registered family and decision rule"):text.index("### Pins read from the code")]
    failure = text[text.index("### Failure handling"):text.index("## Part E.")]
    part_e = text[text.index("## Part E."):text.index("## Part F.")]
    # the H-E paragraph of the family text
    assert "Test B is refused, before anything is spent, if Test A was never started, and if Test A's output is complete " \
           "but its record lacks the finish fields, which a repair by hand restores (Rulings 112 and 116" in family
    assert "or it finished without stored P1 rows), Test B runs alone and tests H-E on its own window" in family
    # Failure handling: the refusal, the claim file, the reason, then the way to run alone, in the code's own words
    assert "`docs/sealed/RUNS.json` holds a record of Test A, or Test A's claim file `RUN_A.claim` exists" in failure
    assert "would lose the pooled confirmatory test of H-E" in failure
    for name in ("STATUS_HE_ALONE", "WORDING_HE_ALONE", "SCOPE_B_ALONE", "SCOPE_B_OWN_QUARTER", "STATUS_HB_NO_TEST_A",
                 "PRIOR_POOLED", "PRIOR_NOT_RUN", "PRIOR_NO_P1", "PRIOR_UNFINISHED", "PRIOR_UNRECORDED"):
        assert f'"{getattr(sealed_run, name)}"' in failure, name
    assert "Test A's output is not available" in sealed_run.STATUS_HE_ALONE                  # Ruling 116, Mn6: quoted, not typed
    # Part E: Ruling 109's second point, and the entry of Ruling 112 after those of 109, 110 and 111
    entries = {int(n): body for n, body in re.findall(r"^- \*\*Ruling (\d+)\*\* \((?:dev|2A|2B)\): (.*)$", part_e, flags=re.M)}
    assert sorted(entries) == list(entries) and 109 < 110 < 111 < 112 < 116 < 118
    assert "(2) If Test A's output is not available (NOT_RUN; claimed and left no manifest and no finished record; finished " \
           "without stored P1 rows), Test B runs alone" in entries[109]
    assert entries[112].startswith("Test B is refused, before anything is spent, unless Test A has been started")
    assert "if Test A's output is not available" in entries[112] and "(Ruling 116)" in entries[112]
    assert "`RUN_A.claim`" in entries[112] and "cannot lose the pooled confirmatory test of H-E" in entries[112]
    assert "(`docs/sealed/RUNS.json` has a record of Test A, or Test A's claim file" in entries[112]
    assert "has no recorded run" not in text


def test_the_sealed_run_does_what_ruling_112_says(tmp_path):
    """The three states of Test A that the text describes, against the real functions of es.sealed_run: never started (Test B
    refused), claimed and not recorded, and recorded without a finish (both: Test B runs alone); and the decision of a Test B
    that runs alone. A rehearsal config, so that the once-only records sit in the temporary folder."""
    import es.sealed_run as sealed_run
    from es.analyses import ANALOGS
    from es.sealed_run import SealedConfig
    arm = ANALOGS["AQ2"]
    prior = tmp_path / "A_out"
    cfg = SealedConfig("B", arm, arm, [], "2026-06-30", prior_test_dir=prior, out_dir=tmp_path / "outB")
    record = {"protocol_sha256": "0" * 64}
    # Test A was never started: no record, no claim file. Refused, whatever folder Test B is pointed at.
    with pytest.raises(ValueError, match=r"Test A has not been run: .*has no record of it and there is no RUN_A\.claim"):
        sealed_run._check_prior(cfg, record, [])
    prior.mkdir()
    (prior / "results.json").write_text(json.dumps({"test": "A", "coverage": {"ran": True}}))
    with pytest.raises(ValueError, match="nothing records the run that made it"):
        sealed_run._check_prior(cfg, record, [])
    # Test A was claimed and left no record: Test B runs alone
    (tmp_path / "RUN_A.claim").write_text("{}")
    none, state = sealed_run._check_prior(cfg, record, [])
    assert none is None and state["state"] == sealed_run.PRIOR_UNRECORDED == "claimed, not recorded"
    # Test A has a record without a finish: Test B runs alone
    (tmp_path / "RUNS.json").write_text(json.dumps({"A": {"started_utc": "2026-10-01T00:00:00+00:00", "out_dir": str(prior)}}))
    none, state = sealed_run._check_prior(cfg, record, [])
    assert none is None and state["state"] == sealed_run.PRIOR_UNFINISHED == "unfinished"
    # the decision of a Test B that runs alone (no prior): H-E not confirmatory, nothing replicated, no step-down
    family = {"H-E": {"confirmatory": True, "reasons": [], "approximate": False, "interim_approximate": True},
              "H-B1": {"confirmatory": True, "reasons": [], "approximate": False},
              "H-B2": {"confirmatory": False, "reasons": ["simulated size too large"], "approximate": True}}
    hb = {"H-B1": {"confirmatory": True, "t_hat": 0.1, "p": 0.01}, "H-B2": {"confirmatory": False, "t_hat": 0.2}}
    entry, holm_out = sealed_run._decide({"p": 0.2}, hb, family, ["H-E", "H-B1"], None, "B")
    assert entry["status"] == sealed_run.STATUS_HE_ALONE and entry["confirmatory"] is False
    assert entry["wording"] == sealed_run.WORDING_HE_ALONE
    assert holm_out == {"H-E": None, "H-B1": None, "H-B2": None}
    assert hb["H-B1"]["status"] == sealed_run.STATUS_HB_NO_TEST_A and hb["H-B1"]["replicated"] is None
    assert hb["H-B2"]["status"] == "estimand (registration rule): simulated size too large"
    assert hb["H-B2"]["replicated"] is None
    # a Test B that pools, with H-E failed: no step-down, and a confirmatory H-B test says why
    hb2 = {"H-B1": {"confirmatory": True, "t_hat": 0.1, "p": 0.01}}
    entry, holm_out = sealed_run._decide(None, hb2, family, ["H-E", "H-B1"], {"H-B1": {"p": 0.01, "t_hat": 0.1}}, "B")
    assert entry is None and "error" in holm_out and hb2["H-B1"]["status"] == sealed_run.STATUS_HB_NEEDS_HE
    from es.protocol import render_protocol
    failure = render_protocol(_full_ctx())
    failure = failure[failure.index("### Failure handling"):failure.index("## Part E.")]
    assert "**H-E fails in Test B.**" in failure and f'"{sealed_run.STATUS_HB_NEEDS_HE}"' in failure


def test_the_pre_lock_list_follows_run_sealed_and_the_claim_comes_before_the_record():
    """The order of the checks in the text is the order in run_sealed; the claim file is created exclusively, first, before the
    output folder is made and before the record is written, and the record gets its finish at the end (Ruling 116, N3; the lock
    paragraph says so)."""
    from es.protocol import render_protocol
    source = (DEV_ROOT / "es/sealed_run.py").read_text()
    run = source[source.index("def run_sealed("):source.index("def _execute(")]
    run = run[run.index("    _check_test_id(cfg)"):]                          # past the docstring
    calls = ["_check_test_id(", "_refuse_recorded(", "_check_config(", "out.exists()", "verify_registration(",
             "verify_bound(", "code_departures(", "_check_environment(", "check_sealed_dirs(", "_read_inputs(",
             "_check_prior(", "_check_side_tables(", "_check_free_space(", "_check_writable(", "_take_claim(", "out.mkdir(",
             "_write_run_record("]
    at = [run.index(c) for c in calls]
    assert at == sorted(at), dict(zip(calls, at))
    text = render_protocol(_full_ctx())
    run_text = text[text.index("**Run** (`es.sealed_run.run_sealed`)"):text.index("**After the lock.**")]
    phrases = ["the test id must be", "the configuration is checked", "`verify_registration` reads", "`verify_bound` requires",
               "is recomputed", "the running environment is compared", "each sealed pull folder's manifest",
               "the registered inputs are read", "for Test B, the record of Test A is read",
               "the development settlement tables", "must be free where the output folder lives",
               "the folder that holds the once-only records", "**The lock.**"]
    in_text = [run_text.index(p) for p in phrases]
    assert in_text == sorted(in_text), dict(zip(phrases, in_text))
    take = source[source.index("def _take_claim("):source.index("def _write_run_record(")]
    record = source[source.index("def _write_run_record("):source.index("def _claim_run(")]
    assert "os.O_CREAT | os.O_EXCL" in take and "_atomic_write_json(" in record and "_atomic_write_json(" not in take
    assert "mkdir" not in take.split("store.mkdir")[1]                                         # the claim makes no output folder
    finish = source[source.index("def _record_finished("):source.index("def _load_sealed_raw(")]
    for field in ("finished_utc", "status", "manifest_sha256"):
        assert f'"{field}"' in finish
    lock = run_text[run_text.index("**The lock.**"):]
    record_fields = {"started_utc": "start time", "protocol_sha256": "protocol SHA-256", "out_dir": "output folder",
                     "registration_identifier": "registration identifier", "registration_kind": "its kind",
                     "registered_utc": "kind and time",
                     "code": "a fingerprint of the code", "n_rep": "the number of replicates",
                     "departures": "the departures", "environment": "the environment"}
    for field, words in record_fields.items():                       # what the record holds, and what the paragraph lists
        assert f'"{field}"' in record and words in lock, field


def test_the_names_and_limits_the_protocol_quotes_are_the_codes():
    from pathlib import Path

    import es.sealed_run as sealed_run
    from es.protocol import render_protocol
    text = render_protocol(_full_ctx())
    assert sealed_run._claim_path(Path("x"), "A").name == "RUN_A.claim"
    assert sealed_run._claim_path(Path("x"), "<test>").name == "RUN_<test>.claim"
    assert sealed_run._runs_path(Path("x")).name == "RUNS.json" and sealed_run.ENV_PINS_NAME == "ENV_PINS.json"
    assert sealed_run.TEST_IDS == ("A", "B") and sealed_run.N_REP == 2000 and sealed_run.MIN_FREE_BYTES == 2 * 1024 ** 3
    assert "the test id must be `A` or `B`" in text and "must use the registered 2000 bootstrap replicates" in text
    assert "whatever the output folder (Rulings 104(a) and 109, C1 and I9)" in text
    assert "at least 2 GiB must be free where the output folder lives and where `docs/sealed` lives" in text
    assert "is refused before its lock, with nothing created, if less than 2 GiB is free" in text
    assert "`docs/sealed/RUN_<test>.claim` is created exclusively (`O_CREAT|O_EXCL`)" in text and "it is never removed" in text
    assert "its record in `RUNS.json` gets `finished_utc`, its status and the SHA-256 of the output folder's manifest" in text
    assert "a missing file refuses the run, and so does a different interpreter or a different version of python or of a " \
           "pinned library; the interpreter is compared by the path it resolves to" in text
    assert "the live environment is recorded in `results.json` and `RUNS.json`" in text
    assert "the development settlement tables (`caiso_fmm_dev` and `nyiso_rtint_dev` under the data root) must be there" in text


def test_the_protocol_quotes_the_modules_constants_and_stops_when_one_is_gone(monkeypatch):
    import es.sealed_run as sealed_run
    from es.protocol import render_protocol
    monkeypatch.setattr(sealed_run, "MIN_FREE_BYTES", 4 * 1024 ** 3)
    monkeypatch.setattr(sealed_run, "N_REP", 1000)
    monkeypatch.setattr(sealed_run, "INTERVAL_HB_ESTIMAND", "percentile only")
    monkeypatch.setattr(sealed_run, "INTERVAL_HB_TEST", "basic only")
    monkeypatch.setattr(sealed_run, "STATUS_HE_ALONE", "H-E alone")
    monkeypatch.setattr(sealed_run, "SCOPE_E5", {"A": "window A", "B": "window B"})
    changed = render_protocol(_full_ctx())
    assert "at least 4 GiB must be free" in changed and "less than 4 GiB is free" in changed
    assert "must use the registered 1000 bootstrap replicates" in changed
    assert "must be the registered one, with 1000 bootstrap replicates, `es.registration`'s own paths" in changed
    assert "`interval_registered` reads \"percentile only\"" in changed and "`interval_registered` reads \"basic only\"" in changed
    assert "its status reads \"H-E alone\"" in changed and "(`E5_scope`: \"window A\" or \"window B\"" in changed
    monkeypatch.delattr(sealed_run, "STATUS_HB_NO_TEST_A")
    with pytest.raises(RuntimeError, match="es.sealed_run has no STATUS_HB_NO_TEST_A"):
        render_protocol(_full_ctx())


def test_the_environment_pins_are_what_the_sealed_run_compares():
    import es.sealed_run as sealed_run
    from es.protocol import ENV_PIN_LIBRARIES, VERSION_MODULES, capture_versions, env_pins
    assert env_pins(capture_versions()) == sealed_run.live_environment()          # the file written is the one compared
    assert ENV_PIN_LIBRARIES == ("python", *(name for name, _ in sealed_run.ENV_LIBRARIES))
    assert tuple(VERSION_MODULES[:len(sealed_run.ENV_LIBRARIES)]) == tuple(sealed_run.ENV_LIBRARIES)
    assert set(sealed_run.live_environment()["versions"]) <= set(capture_versions())


def test_the_pull_text_states_the_day_check_the_overlap_the_resume_rule_and_the_incomplete_chunks():
    import inspect

    import es.sealed_pull as sealed_pull
    from es.protocol import render_protocol
    text = render_protocol(_full_ctx())
    run = text[text.index("### The sealed pull and the sealed run"):text.index("### Failure handling")]
    partb = text[text.index("## Part B."):text.index("## Part C.")]
    assert "every node of the node map must have exactly as many rows as the day has clock hours (24; 23 or 25 when the " \
           "clocks change, MISO keeping fixed EST), every one of them on the hour (minute and second 0, in UTC) and with a " \
           "finite price" in run
    assert "in the price files and in the FMM and NYISO integrated tables (an FMM hour counts only with all four of its " \
           "15-minute intervals)" in run
    assert "A row off the hour is not an hour, and is logged. A short day, and a day with more rows than hours (an hour " \
           "twice), both fail, and either is excused only by naming the day in `allow_missing_days`, which is logged." in run
    assert "On a fall-back day NYISO, ISO-NE and the NYISO integrated table expect 23 hours, because the July NYISO and " \
           "ISO-NE downloaders and the NYISO integrated parser drop both rows of the repeated hour; CAISO and its FMM table " \
           "expect 25." in run
    assert f"CAISO's real-time hour is the mean of its {sealed_pull.RT_INTERVALS_PER_HOUR} five-minute prices, and an hour " \
           "with fewer than that, or with a price that is not finite, is NaN" in run and sealed_pull.RT_INTERVALS_PER_HOUR == 12
    assert "`COVERAGE.csv` (`hours` per node, counting the rows on the hour that have finite prices, against " \
           "`expected_hours`, with the count of non-finite rows)" in run
    assert "In `COVERAGE.csv`, `expected_hours` is the number of clock hours of the deployment window's local settlement " \
           "days (23 on a spring-forward day, 25 on a fall-back day), which is not what the day check expects on a " \
           "fall-back day." in run
    assert "as many rows with finite prices as the day has clock hours" not in text      # the earlier, weaker rule
    assert sealed_pull.REPEATED_HOUR_DROPPED == ("NYISO", "ISONE", "nyiso_rtint")      # the series the sentence names
    assert "Logged, not enforced" in inspect.getdoc(sealed_pull.overlap_check)
    assert "is not enforced, because ISOs revise settled prices" in run and "is not enforced, because ISOs revise" in partb
    assert "must reproduce" not in text
    assert f"A chunk that gives no good answer in {sealed_pull.CHUNK_ATTEMPTS} attempts stops the pull, and `PULL_LOG.txt` " \
           "names the chunk and says how many of its attempts were refused answers." in run
    assert f"may name in `allow_incomplete_chunks` a chunk for which the log says that all {sealed_pull.CHUNK_ATTEMPTS} " \
           "attempts were refused answers, and the chunk is then accepted only if every one of the attempts of that call " in run
    assert sealed_pull.CHUNK_ATTEMPTS == 8
    assert "**A chunk that never comes complete (Rulings 111 and 121).** An answer that the server gave and that can be " \
           "read is whole; it is refused if the validator refuses its rows (short of the interval grid, outside the bounds " \
           "asked for, a price that is not finite) or if it holds no rows at all (OASIS's no-data report, a CSV without " \
           "rows, a zip without a CSV). A network failure (a timeout, a connection error, an HTTP error status) and an " \
           "answer that cannot be read are no whole answer." in run
    assert "the chunk is then accepted only if every one of the attempts of that call brought a whole answer that was " \
           "refused: the answer with the fewest problems (the last of equals) is accepted, short or with non-finite " \
           "prices, and never cached, and a chunk whose answers held no rows is accepted as zero rows; if any attempt of " \
           "the call was a network failure or an unreadable answer, the pull stops." in run
    assert "the last whole response" not in text                                          # the rule before Ruling 121
    assert "(`resume=True`) may name in `allow_incomplete_chunks` a chunk" in run
    assert "still have to be named in `allow_missing_days`" in run
    assert "A root that has files is resumed only from its `.state.json`; without it the pull is refused and never started " \
           "over" in run
    assert "the id and kind of the registration it ran under" in run and "Test B's pull needs Test A's completed pull" in run
    assert "A finished pull whose record in `PULLS.json` was not written (it was killed after its manifest) is adopted by a " \
           "call with `resume=True`, which verifies its manifest and its log and downloads nothing." in run
    assert "return _adopt(test_id, root, fields)" in inspect.getsource(sealed_pull.pull_sealed)    # the code does that: it adopts
    adopt = inspect.getsource(sealed_pull._adopt)
    assert "No download happens" in " ".join(inspect.getdoc(sealed_pull._adopt).split())
    assert "verify_manifest(root)" in adopt and "pull_finished_utc" in adopt
    assert "the registration id of the verified registration (for an OSF registration the id stays the same when a DOI is " \
           "added after an embargo; a public release and a local record have ids of their own, so a pull made under one " \
           "record is not read under another)" in run
    assert "must not name another registration id" in run and "OSF id" not in run         # Ruling 140: osf_id is registration_id


def test_the_after_the_lock_text_names_the_fields_and_the_counts_the_sealed_run_writes():
    from es.protocol import render_protocol
    source = (DEV_ROOT / "es/sealed_run.py").read_text()
    text = render_protocol(_full_ctx())
    after = text[text.index("**After the lock.**"):text.index("### Failure handling")]
    stage = source[source.index("def _stage_bootstraps("):source.index("def _stage_economics(")]
    for token in ("bootstrap_p_one_sided", "interval_registered", "E5_scope"):                  # the writer has them
        assert token in stage, token
    assert "An estimand has no `p`: its bootstrap p-value is kept as a diagnostic, under `diagnostics.bootstrap_p_one_sided`" \
           in after
    assert "its `interval_registered` reads \"percentile (primary); basic (diagnostic)\"" in after
    assert "also has the one-sided basic bootstrap p-value (`p`), and its `interval_registered` reads \"basic\"" in after
    assert "its status reads \"estimand (registration rule)\", followed by \": \" and the reasons of the registration rule" in after
    load = source[source.index("def _load_sealed_raw("):source.index("def coverage(")]
    for token in ("node_map_dropped", "duplicates_dropped", "beyond_window_end", "off_hour_dropped", "nonfinite_dropped"):
        assert token in load, token
    load = load[load.index("    _require_claim(cfg, claim)"):]                         # past the docstring
    assert "Rows whose timestamp is not on the hour are dropped and counted per market and node" in after
    assert "(outside the node map, duplicate, beyond the window end, off the hour, non-finite)" in after
    assert "If it fails as a whole, H-E's rows are built on their own (`p1_rows_for`)" in after and "p1_rows_for(" in source
    assert "(`estimands_scope`: \"single window (interim)\")" in after and "\"pooled A+B\"" in after
    assert "(`E5_scope`: \"Test A window\" or \"Test B window\"; Ruling 109, M5)" in after
    # the order of the cuts is the writer's: node map, then duplicates, then the window end, then off-hour, then non-finite
    cuts = [load.index(t) for t in ("restrict_to_node_map(", "drop_duplicates(", "settlement_frame(", "off_hour = ", "np.isfinite")]
    assert cuts == sorted(cuts)
    assert after.index("restricted to the frozen node map") < after.index("deduplicated by market") < after.index("cut at the window end")


def test_the_registration_template_says_that_two_identifiers_must_name_the_same_registration(tmp_path):
    from es.protocol import registration_template
    from es.registration import RegistrationError, parse_registration
    template = registration_template()
    doi_line = next(ln for ln in template.splitlines() if ln.startswith("osf_doi:"))
    address_line = next(ln for ln in template.splitlines() if ln.startswith("osf_registration:"))
    assert "same registration" in doi_line and "same registration (the same OSF id)" in address_line
    assert "they must name the same OSF registration" in template and "a record that names two registrations is refused" in template
    reg = tmp_path / "REGISTRATION.md"

    def record(doi_id, address_id):
        reg.write_text(f"protocol_sha256: {'ab' * 32}\nosf_doi: 10.17605/OSF.IO/{doi_id}\n"
                       f"osf_registration: https://osf.io/{address_id}/\nregistered_utc: 2026-10-24T18:30:00Z\nembargo_until: none\n")
        return parse_registration(reg)

    assert record("ABCDE", "abcde")["osf_registration"] == "https://osf.io/abcde/"          # the same id, whatever the case
    with pytest.raises(RegistrationError, match="different OSF registrations"):
        record("ABCDE", "fghij")


# ------------------------------------------------------------------------------------------------ Ruling 118 (gate G1)
def _section_of(text, start, stop):
    return text[text.index(start):text.index(stop)]


def _with_family(**confirmatory):
    """_full_ctx with the confirmatory flags of the family set (keys HE, HB1, HB2); the others stay as they are."""
    ctx = _full_ctx()
    fam = ctx["inputs"]["family"]
    for key, name in (("HE", "H-E"), ("HB1", "H-B1"), ("HB2", "H-B2")):
        if key in confirmatory:
            fam[name].update(confirmatory=confirmatory[key], reasons=[] if confirmatory[key] else ["simulated size too large"],
                             approximate=not confirmatory[key])
    return ctx


def test_part_a_carries_the_ruling_118_amendments_of_the_spec():
    """The four amendments (section 5 item 2, section 6.3's decision logic, E3 and E4, and the E5 note) are in the spec and, as
    Part A copies those sections word for word, in the render."""
    from es.protocol import render_protocol
    spec = SPEC.read_text()
    text = render_protocol(_full_ctx())
    part_a = _section_of(text, "## Part A.", "## Part B.")
    phrases = [
        "the audit-robustness family is descriptive, as H-B is (Ruling 110). Its p-values come from the calibration-and-deployment "
        "bootstrap that the size study found oversized, so its BH flags are diagnostics. No count of its rejections is reported "
        "or interpreted.",
        "the drift-monitor Spearman p-values treat the nodes as independent, although nodes within an operator co-move, so they "
        "are optimistic. Rejections in the economic-robustness family that repeat one result across fee levels are reported as "
        "one finding.",
        "so the registered primary family is {H-E} alone.",
        "Holm over one hypothesis is a one-sided test at 0.05: H-E reads \"timing skill within calendar cells detected\" if and "
        "only if its pooled rotation p ≤ 0.05.",
        "E3 is the tail effect of the policy, including plain exposure reduction and calendar allocation. It is not evidence of "
        "skill.",
        "E4 also reports, per comparator c, ESred(c) and the paired contrast dESred|c = ESred(P1) − ESred(c)",
        "the E5 and H-S level intervals come from a calendar-aligned calibration-and-deployment bootstrap.",
        "One circular stationary draw over the quarter's calendar, indexed by (month, day), weights the calibration and the "
        "deployment days alike.",
        "H-B1, H-B2 and PC keep the independent scheme the size study evaluated.",
    ]
    for phrase in phrases:
        assert phrase in spec, phrase                                  # the spec says it (it may have been edited since)
        assert phrase in part_a, phrase                                # and Part A copied it
    assert part_a.count("Ruling 118") >= 6                              # the six notes of the four amendments


def test_the_decision_level_is_generated_from_the_family_and_alpha(monkeypatch):
    """Ruling 118, M1 and C1-I1: with the registered family, {H-E}, Holm over one hypothesis is the test at 0.05. The sentence
    is in Part D and in the description of the run, and is generated from the size of the confirmatory family."""
    from es.protocol import render_protocol
    text = render_protocol(_full_ctx())
    family = _section_of(text, "### Registered family and decision rule", "### Pins read from the code")
    run = _section_of(text, "### The sealed pull and the sealed run", "### Failure handling")
    import es.sealed_run as sealed_run
    assert ("**Decision level (Ruling 118).** The registration rule replaced H-B1 and H-B2 by estimands, so the confirmatory "
            "family is {H-E}. Holm over one hypothesis is a one-sided test at 0.05: H-E reads \"timing skill within calendar "
            "cells detected\" if and only if its pooled rotation p ≤ 0.05, and \"timing skill within calendar cells not "
            "detected\" otherwise. In Test A its entry carries the status \"interim (single window; decided after Test B)\" "
            "and is never confirmatory; in Test B, pooled, its status is \"decided (pooled A+B)\". The three-member "
            "step-down and the 0.05/3 rule of Part A do not apply.") in family
    for constant in ("STATUS_HE_INTERIM", "STATUS_HE_DECIDED", "WORDING_HE_DETECTED", "WORDING_HE_NOT_DETECTED"):
        assert f'"{getattr(sealed_run, constant)}"' in family, constant           # Seat M, M18: quoted from the module
    assert ("the Holm step-down is applied over the confirmatory family of the registered inputs only, here {H-E} (Holm over "
            "one hypothesis is the test at 0.05; see Part D) (spec section 6.3)") in run
    assert "the full Holm step-down is applied" not in text
    # three members: the step-down of Part A, with its thresholds
    t3 = render_protocol(_with_family(HE=True, HB1=True, HB2=True))
    f3 = _section_of(t3, "### Registered family and decision rule", "### Pins read from the code")
    assert "the confirmatory family is {H-E, H-B1, H-B2}, m = 3. The Holm step-down runs over these 3" in f3
    assert "compared in turn with 0.05/3, 0.05/2, 0.05, and the procedure stops at the first p-value that misses its threshold" in f3
    assert "In Test A an H-B test of the family is rejected only if its p-value is at most 0.05/3." in f3
    assert "In Test B the step-down takes H-E's pooled rotation p and Test A's p-values of H-B1 and H-B2." in f3
    assert "This is the three-member family of Part A." in f3 and "do not apply" not in f3.split("Decision level")[1].split("\n")[0]
    assert "its status is \"interim (single window; decided after Test B)\" in Test A and \"decided (pooled A+B)\" in Test B" in f3
    # two members: the thresholds of two
    t2 = render_protocol(_with_family(HE=True, HB1=True, HB2=False))
    f2 = _section_of(t2, "### Registered family and decision rule", "### Pins read from the code")
    assert "replaced H-B2 by an estimand, so the confirmatory family is {H-E, H-B1}, m = 2" in f2
    assert "compared in turn with 0.05/2, 0.05," in f2 and "In Test A an H-B test of the family is rejected only if its p-value " \
        "is at most 0.05/2." in f2
    assert "Part A's three-member step-down and its 0.05/3 rule do not apply as written" in f2
    assert "here {H-E, H-B1} (see Part D)" in _section_of(t2, "### The sealed pull and the sealed run", "### Failure handling")
    # an H-B test alone (H-E replaced by its estimand), and an empty family
    t1 = render_protocol(_with_family(HE=False, HB1=True, HB2=False))
    assert "the confirmatory family is {H-B1}. Holm over one hypothesis is a one-sided test at 0.05: H-B1 is rejected in " \
        "Test A if and only if its p-value is at most 0.05" in t1 and "H-E, replaced by its estimand, is not tested." in t1
    t0 = render_protocol(_with_family(HE=False, HB1=False, HB2=False))
    assert "the confirmatory family is empty. No hypothesis is tested and there is no Holm step-down." in t0
    # the level follows ALPHA
    import es.protocol as protocol
    monkeypatch.setattr(protocol, "ALPHA", 0.1)
    assert "a one-sided test at 0.1: H-E reads" in protocol._decision_level_text(_full_ctx()["inputs"]["family"])
    assert "the 0.1/3 rule of Part A do not apply" in protocol._decision_level_text(_full_ctx()["inputs"]["family"])


def test_the_decision_level_is_what_the_sealed_run_decides():
    """The sentence against the real _decide and holm (Seat C1's probe): with the registered family a pooled p of 0.03 is
    "detected", 0.05 is, and 0.0500001 is not; the literal three-member Holm would not reject at 0.03."""
    import es.sealed_run as sealed_run
    from es.run_inference import holm
    family = _inputs()["family"]
    confirmatory = [h for h in sealed_run.FAMILY_ORDER if family[h]["confirmatory"]]
    assert confirmatory == ["H-E"]
    wording = {}
    for p in (0.01, 0.0167, 0.03, 0.05, 0.0500001):
        entry, holm_out = sealed_run._decide({"p": p}, {}, family, confirmatory, {}, "B")
        wording[p] = entry["wording"]
        assert holm_out["H-E"] is (p <= 0.05)
    assert [w == sealed_run.WORDING_HE_DETECTED for w in wording.values()] == [True, True, True, True, False]
    assert wording[0.0500001] == sealed_run.WORDING_HE_NOT_DETECTED
    assert not holm([0.03, 1.0, 1.0], 0.05)[0]                         # what Part A's three-member wording would give


def test_the_e5_and_h_s_intervals_are_the_calendar_aligned_ones(monkeypatch):
    """Ruling 118, M2: one circular stationary draw over the quarter's (month, day) calendar for both windows; `ci` is the
    aligned percentile interval, `ci_independent` a diagnostic; all approximate; H-B1, H-B2 and PC keep the independent scheme."""
    from es.protocol import render_protocol
    text = render_protocol(_full_ctx())
    levels = _section_of(text, "### E5 and H-S: the interval of a level (Ruling 118)", "### Drift monitors")
    for phrase in ("a single circular stationary draw over the quarter's calendar, indexed by (month, day), weights the "
                   "calibration days and the deployment days alike",
                   "The two windows are the same calendar quarter a year apart, so every replicate keeps their matched calendar "
                   "composition. The calendar index is the sorted union of the (month, day) positions of the two windows' days",
                   "Drawing the two windows independently, as the H-B bootstrap does, re-randomizes that composition.",
                   "inflates |log10 R| in the replicates",
                   "The primary interval of E5 and H-S is the aligned percentile interval: the `ci` field of H-S and, in an E5 "
                   "row, `lo` and `hi` (the median |log10 R|) and `failures_lo` and `failures_hi` (the failures). The interval "
                   "of the independent-draw scheme is stored beside it as a diagnostic: `ci_independent` for H-S, and "
                   "`lo_independent`, `hi_independent`, `failures_lo_independent` and `failures_hi_independent` in an E5 row.",
                   "Every E5 and H-S interval is labelled approximate (`approximate` is true whatever the NaN count): the size "
                   "study evaluated the independent scheme, and the coverage of the aligned scheme is not studied.",
                   "H-B1, H-B2 and PC keep the independent scheme, the one the size study evaluated.",
                   "`docs/reviews/g1_probes/`"):
        assert phrase in levels, phrase
    run = _section_of(text, "### The sealed pull and the sealed run", "### Failure handling")
    import inspect

    import es.sealed_run as sealed_run
    assert "E5 and H-S are stored with the calendar-aligned percentile interval as their primary interval and the interval " \
           "of the independent-draw scheme beside it as a diagnostic (Ruling 118; see \"E5 and H-S\" above). H-S stores `ci` " \
           "and `ci_independent` as lists. An E5 row stores flat columns: `lo` and `hi` (the median |log10 R|) and " \
           "`failures_lo` and `failures_hi` (the failures) for the aligned interval, and `lo_independent`, `hi_independent`, " \
           "`failures_lo_independent` and `failures_hi_independent` for the independent draws, with the NaN counts of each " \
           "draw." in run
    # the scheme and the label are the module's (Ruling 116 stores them), and `approximate` is true in every case
    assert f"Both carry `scheme` (\"{sealed_run.SCHEME_E5_HS}\") and `interval_registered` (\"{sealed_run.INTERVAL_E5_HS}\"), " \
           "and `approximate` is true in every case, whatever the NaN count." in run
    assert sealed_run.SCHEME_E5_HS == "aligned" and sealed_run.INTERVAL_E5_HS.startswith("percentile, calendar-aligned (primary)")
    # the E5 row of the code has the columns the text names (the bootstrap itself replaced by a canned level)
    drawn = {"t_hat": 0.1, "ci": [0.0, 0.2], "ci_basic": [0.0, 0.2], "n_nodes": 3, "n_nan_replicates": 0,
             "nan_approximate": False, "scheme": sealed_run.SCHEME_E5_HS, "ci_independent": [-0.1, 0.4],
             "n_nan_replicates_independent": 0, "nan_approximate_independent": False, "approximate": True}
    monkeypatch.setattr(sealed_run, "_level", lambda *a, **k: dict(drawn))
    row = sealed_run._e5_row({("roll", "G1_24", "xgb", 42): object()}, "G1_24", 5)
    named = ["lo", "hi", "failures_lo", "failures_hi", "lo_independent", "hi_independent", "failures_lo_independent",
             "failures_hi_independent", "scheme", "interval_registered", "approximate"]
    assert set(named) <= set(row) and (row["lo"], row["hi"], row["lo_independent"], row["hi_independent"]) == (0.0, 0.2, -0.1, 0.4)
    assert row["scheme"] == sealed_run.SCHEME_E5_HS and row["interval_registered"] == sealed_run.INTERVAL_E5_HS
    assert row["approximate"] is True
    stage = inspect.getsource(sealed_run._stage_bootstraps)                              # and the H-S result has its two intervals
    assert '"ci_independent": list(r["ci_independent"])' in stage and '"approximate": True' in stage
    part_g = _section_of(text, "## Part G.", "## Part H.")
    assert part_g.count("calendar-aligned percentile, primary; independent-draw interval beside it, diagnostic; approximate") == 2
    assert "for E5 and H-S it is the calendar-aligned percentile interval, with the independent-draw interval beside it as a " \
           "diagnostic, and every such interval is approximate, Ruling 118" in part_g
    entries = dict(re.findall(r"^- \*\*Ruling (\d+)\*\* \((?:dev|2A|2B)\): (.*)$", _section_of(text, "## Part E.", "## Part F."),
                              flags=re.M))
    assert "`ci_independent` (the independent draws of H-B) a diagnostic beside it, all approximate" in entries["118"]
    assert "re-randomizing the calendar composition of the two windows" in entries["110"]
    assert "substantially under a common drift" in entries["110"]


def test_the_audit_family_is_descriptive_and_the_monitors_and_fee_levels_carry_their_caveats():
    """Ruling 118, M5 and M9."""
    from es.protocol import render_protocol
    text = render_protocol(_full_ctx())
    bh = _section_of(text, "### Secondary families", "### Fees")
    assert "**The audit-robustness family is descriptive, as H-B is (Rulings 110 and 118, Seat M, M5).**" in bh
    assert "so its BH flags are diagnostics: they are not read as control of the false-discovery rate (Ruling 105(c)), and no " \
           "count of its rejections is reported or interpreted." in bh
    assert "are a record of what the run computed, not a result" in bh and "The family stays as registered" not in bh
    assert "**The economic-robustness family counts a repeated result once (Ruling 118, Seat M, M9).** Rejections that repeat " \
           "one result across fee levels, for example the same position at every fee level, are reported as one finding." in bh
    monitors = _section_of(text, "### Drift monitors", "### Secondary families")
    assert "The Spearman p-values of the drift-monitor family treat the nodes as independent, although nodes within an " \
           "operator co-move, so they, and the BH decisions that use them, are optimistic (Ruling 40's caveat" in monitors
    entries = dict(re.findall(r"^- \*\*Ruling (\d+)\*\* \((?:dev|2A|2B)\): (.*)$", _section_of(text, "## Part E.", "## Part F."),
                              flags=re.M))
    assert "(c) The audit-robustness BH family is computed as registered, but Ruling 118 (M5) supersedes this ruling's reading " \
           "of its rejections as approximate: the family is descriptive. Its p-values come from the H-B bootstrap, whose " \
           "simulated size exceeds nominal, so its BH flags are diagnostics; they are not read as control of the " \
           "false-discovery rate, and no count of its rejections is reported or interpreted." in entries["105"]
    assert "BH rejections are approximate" not in entries["105"] and "stays as registered" not in entries["105"]   # Seat M, M23


def test_e3_is_the_tail_effect_and_e4_gains_the_esred_rows():
    """Ruling 118, M6: E3 is not evidence of skill; E4 reports ESred(c) and dESred|c for both comparators, as approximate
    tail measures, in the text and in the shells."""
    from es.protocol import E4_COMPARATORS, render_protocol
    from es.run_economics import _approximate
    text = render_protocol(_full_ctx())
    est = _section_of(text, "### E1 to E4: what each estimand measures (Rulings 81 and 118)", "### E5 and H-S")
    assert "**E3 (ESred).** E3 is the tail effect of the policy, including plain exposure reduction and calendar allocation. " \
           "It is not evidence of skill. The matched comparisons are SELC, SNET, E1's timing part and E4 (Ruling 118, M6)." in est
    assert "For each of the same comparators c, E4 also reports ESred(c) (`E4:ESred|c`, context) and the paired contrast " \
           "dESred|c = ESred(P1) - ESred(c) (`E4:dESred|c`), computed inside every replicate like dSELC." in est
    assert "Both are tail measures, so their intervals are labelled approximate." in est
    assert "(`E4:SELC_decile`) is a sensitivity only. It is biased upward under calendar-only nulls" in est
    shells = _section_of(text, "## Part G.", "## Part H.")
    assert "### E3: ESred (the tail effect of the policy, not evidence of skill; approximate)" in shells
    assert "### E4: SELC and ESred against the comparators at matched counts" in shells
    for c in E4_COMPARATORS:
        for row in (f"dSELC: P1 minus {c}", f"SELC of {c} (context)", f"dESred: P1 minus {c} (approximate)",
                    f"ESred of {c} (context, approximate)"):
            assert f"| {row} | — | — | — |" in shells, row
        assert _approximate(f"E4:dESred|{c}") and _approximate(f"E4:ESred|{c}")        # the code labels them approximate
    assert "| SELC of P1 under the decile strata (sensitivity only) | — | — | — |" in shells
    import inspect

    from es import run_economics                                   # the statistics the shells name are the code's
    stats = inspect.getsource(run_economics._stats)
    assert 'out[f"E4:ESred|{c}"]' in stats and 'out[f"E4:dESred|{c}"] = out["E3:ESred"] - out[f"E4:ESred|{c}"]' in stats
    assert "The rows that involve ESred (E1's ESred part and its J rows with lambda above 0, E3, and E4's ESred and dESred) " \
           "are tail measures and are labelled approximate." in shells


def test_the_readings_the_paper_must_not_exceed_are_stated():
    """Ruling 118, M12."""
    from es.protocol import render_protocol
    text = render_protocol(_full_ctx())
    he = _section_of(text, "### H-E as run", "### Size study")
    assert "**What H-E measures (Ruling 118, Seat M, M12).** H-E tests the gross timing of losses within calendar cells, " \
           "including timing that the model has learned from calendar features. It is not net value: the net quantities are " \
           "the estimands NET and SNET (E1 and E2)" in he
    est = _section_of(text, "### E1 to E4: what each estimand measures", "### E5 and H-S")
    assert "The level effect is a decomposition relative to P4, not a causal cost of a drifting cutoff (Ruling 118, M12)." in est
    assert "E1 splits V(P1) - V(P4) into a timing part (P1 against P4@r*, which has P1's alert count) and a level part " \
           "(P4@r* against P4)." in est
    assert "so no decision reads it (Ruling 118, M12)" in est


def test_the_size_study_assumptions_and_the_supplementary_check_are_disclosed_with_their_evidence():
    """Ruling 118, M8: the assumptions, the check and its figures, which are typed from the archived probes and checked here
    against their outputs."""
    from scipy.stats import binom

    from es.protocol import render_protocol
    text = render_protocol(_full_ctx())
    size = _section_of(text, "### Size study, power and placebo checks", "### E1 to E4")
    assert "**Assumptions of the size study (Ruling 118, Seat M, M8).** The simulation draws the nodes independently, with " \
           "Gaussian shocks and AR(1) days." in size
    for phrase in ("12 nodes in 3 markets and two 92-day windows, 500 replications per scenario",
                   "a market-wide common daily factor (0.8 of the daily variance)", "market-wide spike days",
                   "daily persistence 0.95", "independent trends within each window",
                   "The pooled rejection rates were at most 0.026 at 0.05/3 and 0.068 at 0.05, and the single-window rates at "
                   "most 0.016 and 0.054",
                   "none is significantly above nominal on the registered binomial check (one-sided, at 0.05/4)",
                   "a drift within a window is the departure it is least protected against",
                   "archived in `docs/reviews/g1_probes/`"):
        assert phrase in size, phrase
    probes = DEV_ROOT.parent / "docs/reviews/g1_probes"
    files = [probes / "he_common_500.txt", probes / "he_trend_500.txt"]
    if not all(f.is_file() for f in files):
        pytest.skip("the archived probe outputs are not there")
    pattern = re.compile(r"n=(\d+)\s+single A: ([\d.]+) / ([\d.]+)\s+pooled A\+B: ([\d.]+) / ([\d.]+)")
    rows = [tuple(float(x) for x in m.groups()) for f in files for m in pattern.finditer(f.read_text())]
    assert len(rows) == 7 and {r[0] for r in rows} == {500.0}
    assert (max(r[3] for r in rows), max(r[4] for r in rows), max(r[1] for r in rows), max(r[2] for r in rows)) == \
        (0.026, 0.068, 0.016, 0.054)                                     # pooled at 0.05/3 and 0.05, single at 0.05/3 and 0.05
    third = 0.05 / 3
    tails = [binom.sf(round(rate * 500) - 1, 500, a) for r in rows for rate, a in zip(r[1:], (third, 0.05, third, 0.05))]
    assert min(tails) > 0.05 / 4                                        # no rate is significant at the rule's level


def test_the_pooled_rates_and_the_rules_operating_characteristic_are_computed_from_the_inputs():
    """Ruling 118, M13: the pooled rejection rates over H-E's primary pooled rows come from simulation.csv, and the demoting
    counts and probabilities from the rule's constants; nothing is typed."""
    from scipy.stats import binom

    from es.config import OUT_DIR
    from es.protocol import render_protocol
    text = render_protocol(_full_ctx())
    size = _section_of(text, "### Size study, power and placebo checks", "### E1 to E4")
    # the fixture's two primary pooled rows: 7 + 3 of 600 at 0.05/3, 21 + 12 of 600 at 0.05
    rows = [r for r in _inputs()["simulation"] if r["test"] == "H-E pooled A+B, size" and r.get("strata") == "primary"]
    assert len(rows) == 2
    hits3 = sum(round(r["reject_0.0166667"] * r["n_sim"]) for r in rows)
    hits5 = sum(round(r["reject_0.05"] * r["n_sim"]) for r in rows)
    total = sum(r["n_sim"] for r in rows)
    assert (hits3, hits5, total) == (10, 33, 600)
    assert f"Pooled over those 2 rows, the rejection rate is {hits3}/{total} = 0.0167 at 0.05/3 and {hits5}/{total} = 0.0550 " \
           "at 0.05." in size
    # the operating characteristic, by brute force from the rule's description (factor, level / rows, binomial tail)
    n, k_rows, factor = 300, 4, 1.5                                      # the fixture's rule: k_rows["H-E"] = 4, factor 1.5

    def least(a):
        return next(k for k in range(1, n + 1) if k / n > factor * a and binom.sf(k - 1, n, a) < 0.05 / k_rows)

    k3, k5 = least(0.05 / 3), least(0.05)
    p3, p5 = binom.sf(k3 - 1, n, factor * 0.05 / 3), binom.sf(k5 - 1, n, factor * 0.05)
    assert (k3, k5) == (12, 25) and round(p5, 2) == 0.32
    assert (f"With {n} simulations per row that takes at least {k3} of {n} ({k3 / n:.4f}) at 0.05/3 and {k5} of {n} "
            f"({k5 / n:.4f}) at 0.05. A row whose true size is 1.5 times nominal is therefore demoted with probability "
            f"{p3:.2f} at 0.05/3 and {p5:.2f} at 0.05 (the exact binomial tail).") in size
    assert "rejects at 0.05/4, the level divided by the 4 rows read" in size and "For H-E that costs nothing" in size
    # the real inputs: the figures of the G1 review, from the simulation rows
    doc = json.loads((OUT_DIR / "analogs" / "protocol_inputs.json").read_text())
    if doc.get("stage") == "complete":
        real = _section_of(render_protocol(_full_ctx(inputs=doc)), "### Size study, power and placebo checks", "### E1 to E4")
        assert "Pooled over those 4 rows, the rejection rate is 20/1200 = 0.0167 at 0.05/3 and 59/1200 = 0.0492 at 0.05." in real
        assert "demoted with probability 0.08 at 0.05/3 and 0.32 at 0.05" in real


def test_the_provenance_roots_join_the_code_set_and_the_guard_still_holds(real_code_files):
    """Ruling 118, C1-M1: es.run_analogs and es.simulate are roots of the registered code set; the sealed roots are as they
    were (es.public_package reads them), and the guard against paper, reporting and test modules still applies."""
    from es.protocol import CODE_ROOTS, PROVENANCE_ROOTS, SEALED_ROOTS, code_files, import_closure
    assert SEALED_ROOTS == ("es.sealed_run", "es.sealed_pull", "es.registration", "es.protocol")
    assert PROVENANCE_ROOTS == ("es.run_analogs", "es.simulate") and CODE_ROOTS == SEALED_ROOTS + PROVENANCE_ROOTS
    sealed_only, _ = import_closure(DEV_ROOT, roots=SEALED_ROOTS)
    with_provenance, _ = import_closure(DEV_ROOT)
    assert "es.run_analogs" not in sealed_only and "es.simulate" not in sealed_only
    assert set(with_provenance) - set(sealed_only) == {"es.run_analogs", "es.simulate"}   # nothing else comes with them
    from es.protocol import OPENED_DATA_FILES
    rel = {p.relative_to(DEV_ROOT).as_posix() for p in code_files(DEV_ROOT)}
    assert rel == {p.relative_to(DEV_ROOT).as_posix() for p in real_code_files}
    assert {"es/run_analogs.py", "es/simulate.py"} <= rel
    assert rel == {f.relative_to(DEV_ROOT).as_posix() for f in with_provenance.values()} | set(OPENED_DATA_FILES)
    assert len(rel) == len(real_code_files) == 45                                       # 43 before, plus the two roots
    assert not [r for r in rel if re.match(r"es/(paper_\w+|run_digest|run_legacy|impl_table)\.py$", r) or r.startswith("tests/")]


# ------------------------------------------------------------------------------------------------ Ruling 119 (gate G1)
def test_part_a_carries_the_ruling_119_amendments_of_the_spec():
    """The amendments of Ruling 119 (sections 4.4, 5 items 1, 2, 3 and 6, 6.5 step 3) and the M16 note on the seed bullet of
    section 6.3 are in the spec and, as Part A copies those sections word for word, in the render."""
    from es.protocol import render_protocol
    spec = SPEC.read_text()
    part_a = _section_of(render_protocol(_full_ctx()), "## Part A.", "## Part B.")
    phrases = [
        "development-only. The sealed tests do not compute or report them.",
        "the mixed model and the leave-one-operator-out checks are development-only; the sealed run does not fit them.",
        "the registered shells are the table shells of the protocol's Part G. Figures are drawn after the run by the reporting "
        "code from the stored tables, and sign consistency is read from the stored estimates. Neither is a registered "
        "computation.",
        "development-only, es.run_labels_pr; the sealed run does not compute them",
        "the sealed run computes failure counts only for the band [0.5, 2] at α = 5%, plus the economic curve over α. The "
        "failure counts at the other α levels and the two narrower bands are development-only. The audit-robustness family, "
        "which is descriptive, also computes its H-B contrasts at the cutoff levels 1%, 2.5% and 10%, with labels fixed at 5%.",
        "E1–E4 pool Test A and Test B; E5, the block table, is computed per window and is not pooled.",
        "these seed variants belong to the audit-robustness family, which is descriptive; its BH flags are diagnostics.",
    ]
    for phrase in phrases:
        assert phrase in spec, phrase
        assert phrase in part_a, phrase
    # the supplements note: the spec says the outputs go to the OSF registration; Part A says what Ruling 140 made of it
    old = ("the development outputs, outputs/es/analogs/ and DIGEST.md, are uploaded to the OSF registration as supplementary "
           "files that are not bound by hash.")
    new = ("the development outputs, outputs/es/analogs/ and DIGEST.md, are published with the code repository as "
           "supplementary files that are not bound by hash, instead of being uploaded to an OSF registration.")
    assert old in spec and old not in part_a and new in part_a
    assert "*(Amended 24 Oct, Ruling 119, G1 M14, and 26 Oct, Ruling 140: " + new in part_a
    assert part_a.count("Ruling 119") >= 6


def test_what_part_a_registers_and_the_sealed_run_does_not_compute_is_listed():
    """Ruling 119, M14: one paragraph in Part D and one line in Part E."""
    from es.protocol import render_protocol
    text = render_protocol(_full_ctx())
    part = _section_of(text, "### What Part A registers that the sealed run does not compute (Ruling 119)", "### Failure handling")
    assert "what this Part D says the sealed run computes is what it computes" in part
    for phrase in ("ES10 and maximum drawdown (spec section 4.4)",
                   "precision and recall at the deployed cutoff against both labels (section 5 item 3; `es.run_labels_pr`)",
                   "the mixed model and the leave-one-operator-out checks (section 5 item 1)",
                   "the failure counts at the levels of alpha other than 5%, and the failure bands [0.8, 1.25] and [0.67, 1.5] "
                   "(section 5 item 6)",
                   "The sealed run counts failures only for the band [0.5, 2] at alpha = 5%.",
                   "**Other levels of alpha that the sealed run does compute (Seat M, M23):** the audit-robustness family "
                   "re-runs the H-B1 and H-B2 contrasts at the cutoff levels 1%, 2.5% and 10%, with the labels fixed at 5%, "
                   "as descriptive diagnostics (the `es.secondary` docstring quoted under the secondary families above), and "
                   "the economic analysis computes the economic curve over alpha of section 5 item 6, for P6 and P1.",
                   "**From the reporting code, not registered computations:** figures, which are drawn after the run from the "
                   "stored tables, and sign consistency across fees, positions and blocks, which is read from the stored "
                   "estimates (section 5 item 2). The registered shells are the table shells of Part G.",
                   "**Supplements, not bound by hash:** the development outputs, `outputs/es/analogs/` and "
                   "`outputs/es/DIGEST.md`, are published with the code repository as supplementary files, instead of being "
                   "uploaded to an OSF registration (section 6.5 step 3, Rulings 119 and 140)"):
        assert phrase in part, phrase
    entry = dict(re.findall(r"^- \*\*Ruling (\d+)\*\* \((?:dev|2A|2B)\): (.*)$", _section_of(text, "## Part E.", "## Part F."),
                            flags=re.M))["119"]
    assert entry.startswith("Part A registers some quantities that the sealed run does not compute") and "unbound supplements" in entry
    assert "are published with the code repository as unbound supplements (Ruling 140 replaced the upload to OSF)" in entry
    f = _section_of(text, "## Part F.", "## Part G.")
    assert "published with the code repository as supplementary files that are not part of the bound files (spec section 6.5 " \
           "step 3, Rulings 119 and 140)" in f
    # the sealed run really has none of them: no fit of the mixed model, no precision and recall, no drawdown (the names it
    # imports, defines or calls, not its docstrings, which mention the label control)
    import ast
    tree = ast.parse((DEV_ROOT / "es/sealed_run.py").read_text())
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names |= {a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
    names |= {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    names |= {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    assert not [n for n in names if "fit_mixed" in n or "run_labels_pr" in n or "drawdown" in n.lower()]


def test_the_audit_text_governs_the_quoted_docstring_and_the_premium_row_is_h_es_own():
    """Ruling 119, M16 and M17."""
    import es.secondary as secondary
    from es.protocol import render_protocol
    text = render_protocol(_full_ctx())
    bh = _section_of(text, "### Secondary families", "### Fees")
    assert secondary.__doc__ in bh                                                    # the docstring is quoted, unchanged
    assert "This text governs the `es.secondary` docstring quoted above, which describes the family as adjusted by BH: the " \
           "flags are computed, as the docstring says, but they are diagnostics (Seat M, M16" in bh
    part_g = _section_of(text, "## Part G.", "## Part H.")
    assert "E1 to E4 are given for Test B alone and pooled. H-E is pooled only, and is Test B alone only when Test A's output " \
           "is not available" in part_g
    assert "Each table is reported for Test A alone (interim)." in part_g and "(Ruling 112)" in part_g
    assert "The economic-robustness family's \"P1 premium fees=calibrated\" row is H-E's own single-window statistic, not a " \
           "separate finding: in Test A it is the interim p, and in Test B the same statistic on Test B's window alone" in part_g
    assert "row is H-E's own single-window statistic, not a second H-E result (see Part G)" in bh
    # the label and the statistic are the code's
    import inspect

    from es.run_economics import economics_for
    row = {"policy": "P1", "comparator": "none", "position": "premium", "fees": "calibrated", "SELC": 0.1, "rotation_p": 0.3}
    assert secondary.economic_family([row])[0]["test"] == "P1 premium fees=calibrated"
    source = inspect.getsource(economics_for)
    assert 'if name == "P1"' in source and 'rotation_test([RowCube.from_rows(rows)])["p"]' in source
    assert 'position == "premium" and fees == "calibrated" and name in KEPT' in source      # and kept["P1"] is that row set


def test_the_alert_rule_is_read_from_the_code_and_cited():
    """Ruling 119, M19: score >= tau, np.quantile with linear interpolation, top_count's ties by time order. The text is
    checked against es.policies when it is rendered, and here against what the code does."""
    import numpy as np
    import pandas as pd

    import es.protocol as protocol
    from es import policies
    text = protocol.render_protocol(_full_ctx())
    pins = _section_of(text, "### Pins read from the code", "### H-E as run")
    for phrase in ("**The alert rule (Seat M, M19; read from `es.policies` when this protocol is rendered).**",
                   "An alert fires if and only if score >= tau (`es.policies._with_tau`): a score equal to the cutoff alerts.",
                   "tau is the (1 - alpha) quantile of the calibration scores by `np.quantile` with numpy's default linear "
                   "interpolation (`es.policies._tau`; `es.cutoffs` repeats the rule for `pre_pull_cutoffs.csv`).",
                   "`top_count` (`es.policies.top_count`), which places `calendar@count` and `VOL@count`, ranks a node's "
                   "deployment rows by a stable sort of minus the score over rows in time order (`_per_node`), so a tie is "
                   "broken by time, earliest row first."):
        assert phrase in pins, phrase
    assert set(protocol.alert_conventions().values()) == {True}
    # what the code does: equality alerts; linear interpolation; ties by time order, whatever the order of the input
    stamps = pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC")
    calib = pd.DataFrame({"market": "M", "node": "n", "timestamp_utc": stamps, "split": "calib", "score": [0.0, 1.0, 2.0, 3.0, 4.0]})
    deploy = pd.DataFrame({"market": "M", "node": "n", "timestamp_utc": stamps + pd.Timedelta(days=1), "split": "deploy",
                           "score": [2.0, 1.9, 2.1, 5.0, 0.0]})
    out = policies.p1_fixed(pd.concat([calib, deploy], ignore_index=True), 0.5)
    assert out["tau"].iloc[0] == 2.0 and out["alert"].tolist() == [True, False, True, True, False]       # score == tau alerts
    assert policies._tau([0.0, 1.0, 2.0, 10.0], 0.5) == float(np.quantile([0.0, 1.0, 2.0, 10.0], 0.5, method="linear")) == 1.5
    tied = pd.DataFrame({"market": "M", "node": "n", "timestamp_utc": stamps, "split": "deploy", "score": [1.0, 2.0, 2.0, 2.0, 3.0]})
    for frame in (tied, tied.iloc[::-1], tied.sample(frac=1.0, random_state=3)):
        got = policies.top_count(frame, {("M", "n"): 2}).sort_values("timestamp_utc")
        assert got["alert"].tolist() == [False, True, False, False, True]                              # the 3.0, then the earliest 2.0


def test_a_changed_alert_convention_stops_the_protocol(monkeypatch):
    import es.protocol as protocol
    from es import policies

    def _with_tau(dep, tau):
        return dep.assign(tau=tau, alert=dep["score"].to_numpy(dtype=float) > tau)             # strictly greater: not the registered rule

    protocol.alert_conventions.cache_clear()
    monkeypatch.setattr(policies, "_with_tau", _with_tau)
    try:
        with pytest.raises(RuntimeError, match="es.policies no longer does what the protocol says about an alert: an alert fires "
                                               "iff score >= tau"):
            protocol.alert_conventions()
    finally:
        monkeypatch.undo()
        protocol.alert_conventions.cache_clear()
    assert set(protocol.alert_conventions().values()) == {True}


def test_clipped_high_is_said_to_be_the_top_of_the_search_range():
    """Ruling 119, M22."""
    from es import simulate
    from es.protocol import render_protocol
    text = render_protocol(_full_ctx())
    part_c = _section_of(text, "## Part C.", "## Part D.")
    assert "\"Clipped high\" means that the correction hit the top of its search range, phi = 0.95" in part_c
    assert "\"clipped low\" means the bottom of the range, phi = -0.5" in part_c
    assert simulate._PHI_SEARCH == (-0.5, 0.95)                                    # the range the text names


def test_the_rule_written_out_in_the_protocol_equals_the_rule_in_run_analogs():
    """The demotion counts and probabilities printed in Part D use a local copy of the rule (so that the sealed path does not
    import the development runner); it must be the rule's."""
    from es import run_analogs
    from es.protocol import _binom_tail, _least_demoting_count
    for n in (100, 300, 1000):
        for a in run_analogs.NOMINAL:
            for rows in (1, 4, 12):
                assert _least_demoting_count(n, a, run_analogs.FACTOR, rows, run_analogs.BINOMIAL_LEVEL) == \
                    run_analogs._smallest_demoting_count(n, a, run_analogs.FACTOR, rows)
        for k in (0, 1, n // 20, n // 2, n):
            assert _binom_tail(k, n, 0.075) == run_analogs._binom_upper(k, n, 0.075)


def test_the_embedded_statement_has_its_headings_demoted_by_two_levels():
    """Ruling 119, M20."""
    from es.protocol import demote_headings, render_protocol
    assert demote_headings("# A\n\n## B\ntext # not a heading\n#### C\n##### D\n###### E\n") == \
        "### A\n\n#### B\ntext # not a heading\n###### C\n###### D\n###### E\n"                       # capped at six hashes
    assert demote_headings("```\n# code\n```\n## B") == "```\n# code\n```\n#### B"                     # fenced code is left alone
    assert demote_headings("#nospace\n##\tTab\n") == "#nospace\n####\tTab\n" and demote_headings("a\n") == "a\n"
    template = (DEV_ROOT.parent / "docs/sealed/KNOWLEDGE_STATEMENT_TEMPLATE.md").read_text()
    part_h = _section_of(render_protocol(_full_ctx(knowledge=template)), "## Part H.", "## Part I.")
    assert re.findall(r"^#{1,2} ", part_h, flags=re.M) == ["## "]                                      # only Part H's own heading
    titles = [ln for ln in re.findall(r"^#+ .*$", template, flags=re.M)]
    assert titles and all("##" + t in part_h for t in titles), titles                                   # each one, two levels deeper
    assert "follows word for word, its headings two levels deeper so that they sit under this Part (Ruling 119)." in part_h


# ---- ready_to_register on the knowledge statement (Ruling 119, M15). The template is read at test time, never copied here.
_TEMPLATE = DEV_ROOT.parent / "docs/sealed/KNOWLEDGE_STATEMENT_TEMPLATE.md"


def _template_text():
    return _TEMPLATE.read_text()


def _fill_markers(text):
    return re.sub(r"\[AUTHOR:[^\]]*\]", "filled in by the author", text)


def _tick_rows(text, glyph="☒", per_row=1):
    """Tick ``per_row`` empty boxes in each table row that has them: the first ones of the row."""
    out = []
    for line in text.splitlines():
        if line.startswith("|") and "☐" in line:
            for _ in range(per_row):
                line = line.replace("☐", glyph, 1)
        out.append(line)
    return "\n".join(out) + "\n"


def _drop_guidance(text):
    """Without the paragraphs that are one italic span (the guidance the template tells the author to delete)."""
    keep = [p for p in re.split(r"\n\s*\n", text) if not re.fullmatch(r"\*(?![\s*])[^*]+(?<![\s*])\*", p.strip())]
    return "\n\n".join(keep) + "\n"


def _ready(knowledge):
    from es.protocol import ready_to_register, render_protocol
    return ready_to_register(render_protocol(_full_ctx(knowledge=knowledge)))


def test_the_unfilled_template_is_flagged_for_its_boxes_and_its_guidance():
    """Ruling 119, M15: markers, then every item-1 row without a tick, then each guidance paragraph."""
    from es.protocol import author_markers, knowledge_problems, ready_to_register
    template = _template_text()
    rows = [ln for ln in template.splitlines() if ln.startswith("|") and "☐" in ln]
    assert rows                                                                           # the template asks its questions
    problems = knowledge_problems(template)                                               # the bare statement is checked too
    boxes = [p for p in problems if p.startswith("[ITEM 1:")]
    guidance = [p for p in problems if p.startswith("[GUIDANCE:")]
    assert len(boxes) == len(rows) and all("no box is ticked in the row" in p for p in boxes)
    assert any(p.startswith("[GUIDANCE: *Fill this in") for p in guidance) and any(p.startswith("[GUIDANCE: *This covers") for p in guidance)
    assert ready_to_register(template) == author_markers(template) + problems
    rendered = _ready(template)
    assert rendered == ready_to_register(template) and len(rendered) == len(author_markers(template)) + len(rows) + len(guidance)


def test_a_filled_statement_passes_whichever_tick_the_author_uses():
    """Accept ☒, ✓, ✔, x and X as a tick (and ☑, a box with a check) and ☐ as empty; anything else is not a tick."""
    base = _drop_guidance(_fill_markers(_template_text()))
    assert _ready(_tick_rows(base)) == []
    for glyph in ("☒", "✓", "✔", "x", "X", "☑"):
        assert _ready(_tick_rows(base, glyph)) == [], glyph
    assert _ready(_tick_rows(base, "x (news headlines)")) == []                            # a tick with its detail
    n_rows = len([ln for ln in base.splitlines() if ln.startswith("|") and "☐" in ln])
    for not_a_tick in ("yes", "xylophone", "A", "-", "☐x"):
        assert len(_ready(_tick_rows(base, not_a_tick))) == n_rows, not_a_tick


def test_a_half_filled_statement_is_flagged_for_what_is_left():
    """Ruling 119, M15: boxes not ticked, two ticked, the guidance left in, a marker left in."""
    filled = _drop_guidance(_fill_markers(_template_text()))
    # (a) no box ticked in any row
    unticked = _ready(filled)
    rows = [ln for ln in _template_text().splitlines() if ln.startswith("|") and "\u2610" in ln]
    assert len(unticked) == len(rows) and all(p.startswith("[ITEM 1: no box is ticked in the row \"") for p in unticked)
    # exactly one row with two ticks, the others with one
    lines = _tick_rows(filled).splitlines()
    first = next(i for i, ln in enumerate(lines) if ln.startswith("|") and "☒" in ln)
    lines[first] = lines[first].replace("☐", "☒", 1)
    double = _ready("\n".join(lines) + "\n")
    assert len(double) == 1 and double[0].startswith("[ITEM 1: 2 boxes are ticked in the row \"")
    # (b) the guidance paragraphs left in, boxes ticked, markers filled
    with_guidance = _tick_rows(_fill_markers(_template_text()))
    left = _ready(with_guidance)
    assert len(left) >= 2 and all(p.startswith("[GUIDANCE: ") for p in left)
    assert any("*Fill this in" in p for p in left) and any("*This covers" in p for p in left)
    # the two named guidance starts are flagged even when the paragraph is no longer one italic span
    broken = _tick_rows(_drop_guidance(_fill_markers(_template_text()))) + "\n*Fill this in before you register.\n\n*This covers July.\n"
    flagged = _ready(broken)
    assert len(flagged) == 2 and flagged[0].startswith("[GUIDANCE: *Fill this in") and flagged[1].startswith("[GUIDANCE: *This covers")
    # an italic guidance line inside a longer paragraph is flagged too
    inline = _tick_rows(_drop_guidance(_fill_markers(_template_text()))) + "\n#### 4. Another\n*Tick one option for each item.*\n"
    flagged_inline = _ready(inline)
    assert len(flagged_inline) == 1 and flagged_inline[0].startswith("[GUIDANCE: *Tick one option for each item.*")
    # a marker left in: the marker, and only the marker
    marked = _tick_rows(_drop_guidance(_fill_markers(_template_text()))) + "\nName and date: [AUTHOR: Erce Erez, date]\n"
    assert _ready(marked) == ["[AUTHOR: Erce Erez, date]"]
    # a marker and a missing tick: both are reported, markers first
    from es.protocol import author_markers
    unfilled = _drop_guidance(_template_text())
    both = _ready(unfilled)
    n = len(author_markers(unfilled))
    assert n >= 1 and both[:n] == author_markers(unfilled) and len(both) == n + len(rows)
    assert all(p.startswith("[ITEM 1:") for p in both[n:])


def test_the_generated_parts_are_not_checked_for_guidance_or_boxes():
    """The spec in Part A has italic paragraphs and the protocol has tables with boxes of its own kind: only the statement is
    checked, and only when there is one."""
    from es.protocol import knowledge_problems, render_protocol
    text = render_protocol(_full_ctx())                                                    # no statement: its marker only
    assert knowledge_problems(text) == []
    assert any(re.fullmatch(r"\*(?![\s*])[^*]+(?<![\s*])\*", p.strip()) for p in re.split(r"\n\s*\n", text))   # italics exist
    filled = _tick_rows(_drop_guidance(_fill_markers(_template_text())))
    assert _ready(filled) == []
    # a statement without an item-1 table is not complained about (the older, free-text statements)
    assert _ready("Name and date: Erce Erez\n") == [] and _ready("## 1. Nothing\n\nI know nothing.\n") == []


# --------------------------------------------------------------------- Ruling 116 (final reconciliation) and Ruling 120
def _isone_rows(nodes, first, last):
    """Every hour of the local days first..last ('YYYY-MM-DD'), on the hour, price 1.0, for the ISO-NE nodes (synthetic)."""
    from es.config import SETTLE_TZ
    from es.sealed_pull import local_day_bounds_utc
    start, end = local_day_bounds_utc(first, last, SETTLE_TZ["ISONE"])
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    return pd.DataFrame([("ISONE", n, "hub", t, 1.0, 1.0) for n in nodes for t in hours],
                        columns=["market", "node", "node_type", "timestamp_utc", "da_lmp", "rt_lmp"])


def test_the_day_check_is_the_one_the_pull_text_describes(tmp_path):
    """Ruling 116, Mn4: exactly the day's hours, on the hour, with finite prices; a short day and a day with more rows than
    hours both fail and only allow_missing_days excuses them; COVERAGE.csv counts rows on the hour; the fall-back day expects
    23 hours where the downloaders drop the repeated hour; a CAISO hour needs all of its twelve five-minute prices. June days
    only, synthetic rows."""
    import datetime as dt

    from es import sealed_pull
    nodes = [n for m, n in sealed_pull._node_map_nodes() if m == "ISONE"]
    a, b = nodes[:2]
    first, last = "2026-06-21", "2026-06-23"
    root = tmp_path / "pull"
    root.mkdir()
    full = _isone_rows(nodes, first, last)
    day = dt.date(2026, 6, 22)
    hour = pd.Timestamp("2026-06-22 09:00", tz="America/New_York").tz_convert("UTC")

    def write(frame):
        frame.to_parquet(root / "isone_da_rt_x.parquet", index=False)

    write(full)
    assert sealed_pull._day_gaps(root, "ISONE", first, last) == ([], {}, {})                  # exactly the hours: complete
    sealed_pull._require_complete(root, "ISONE", first, last, set())                           # nothing to excuse
    # a short day: one hour of one node is missing
    write(full[(full["node"] != a) | (full["timestamp_utc"] != hour)])
    no_rows, short, over = sealed_pull._day_gaps(root, "ISONE", first, last)
    assert (no_rows, over) == ([], {}) and short == {(a, day): (23, 24, 0, 0)}
    with pytest.raises(sealed_pull.MissingDaysError):
        sealed_pull._require_complete(root, "ISONE", first, last, set())
    sealed_pull._require_complete(root, "ISONE", first, last, {"2026-06-22"})                  # only the named day excuses it
    # a day with more rows than hours: a row off the hour is no hour, so the day is over its hours, not short
    stray = full[(full["node"] == b) & (full["timestamp_utc"] == hour)].assign(timestamp_utc=hour + pd.Timedelta(minutes=30))
    write(pd.concat([full, stray], ignore_index=True))
    no_rows, short, over = sealed_pull._day_gaps(root, "ISONE", first, last)
    assert (no_rows, short) == ([], {}) and over == {(b, day): (25, 24, 1)}
    with pytest.raises(sealed_pull.MissingDaysError, match="more rows than the day has hours"):
        sealed_pull._require_complete(root, "ISONE", first, last, set())
    sealed_pull._require_complete(root, "ISONE", first, last, {"2026-06-22"})
    cov = sealed_pull.coverage_manifest(root, first, last, _sealed=True)
    assert int(cov[(cov["series"] == "price") & (cov["node"] == b)]["hours"].iloc[0]) == 72     # the half-hour row is no hour
    # a series stamped off the hour has no hours at all
    write(full.assign(timestamp_utc=full["timestamp_utc"] + pd.Timedelta(minutes=30)))
    no_rows, short, over = sealed_pull._day_gaps(root, "ISONE", first, last)
    assert len(no_rows) == 3 * len(nodes) and short == {} and over == {}
    # the hours a day must hold: 23 on the fall-back day where the downloaders drop the repeated hour, 25 in CAISO
    for key, hours in (("NYISO", 23), ("ISONE", 23), ("nyiso_rtint", 23), ("CAISO", 25), ("fmm", 25), ("MISO", 24)):
        assert sealed_pull._expected_day_hours(key, dt.date(2026, 11, 1)) == hours, key
    # a CAISO hour needs all twelve five-minute prices, and finite ones
    stamps = list(pd.date_range("2026-06-22 10:00", periods=12, freq="5min", tz="UTC"))
    eleven = list(pd.date_range("2026-06-22 11:00", periods=11, freq="5min", tz="UTC"))
    nan_in = list(pd.date_range("2026-06-22 12:00", periods=12, freq="5min", tz="UTC"))
    frame = pd.DataFrame({"node": "n", "timestamp_utc": stamps + eleven + nan_in,
                          "lmp": [2.0] * 12 + [2.0] * 11 + [2.0] * 11 + [float("nan")]})
    hourly = sealed_pull._hourly_rt_mean(frame).sort_values("timestamp_utc")
    assert hourly["lmp"].isna().tolist() == [False, True, True] and hourly["lmp"].iloc[0] == 2.0
    assert sealed_pull.RT_INTERVALS_PER_HOUR == 12


def test_the_states_of_test_a_that_test_b_meets_are_the_ones_the_text_lists(tmp_path):
    """Ruling 116, N1 and N2, against the real _check_prior: never started (refused), claimed with no record, a record without
    its finish and no manifest (both: Test B runs alone), the same with a manifest (refused, with the repair), NOT_RUN, no P1
    (alone), P1 without the other frames (pooled, missing_kept), everything (pooled)."""
    import es.sealed_run as sealed_run
    from es.analyses import ANALOGS
    from es.run_economics import KEPT
    from es.sealed_pull import manifest_sha256, write_manifest
    from es.sealed_run import SealedConfig, _safe
    arm = ANALOGS["AQ2"]
    prior = tmp_path / "A_out"
    cfg = SealedConfig("B", arm, arm, [], "2026-06-30", prior_test_dir=prior, out_dir=tmp_path / "outB")
    sha, started = "0" * 64, "2026-10-01T00:00:00+00:00"
    record = {"protocol_sha256": sha}
    runs = tmp_path / "RUNS.json"
    check = lambda: sealed_run._check_prior(cfg, record, [])                                          # noqa: E731
    elsewhere = tmp_path / "elsewhere"                                  # another folder, with the same store of records
    elsewhere.mkdir()
    pointed_wrong = SealedConfig("B", arm, arm, [], "2026-06-30", prior_test_dir=elsewhere, out_dir=tmp_path / "outB")
    with pytest.raises(ValueError, match="Test A has not been run"):
        check()
    (tmp_path / "RUN_A.claim").write_text("{}")
    none, state = check()
    assert none is None and state["state"] == sealed_run.PRIOR_UNRECORDED == "claimed, not recorded"
    # a record without its finish: no manifest, Test B runs alone; with the manifest, Test B is refused with the repair
    prior.mkdir()
    results = {"test": "A", "started_utc": started, "protocol_sha256": sha, "coverage": {"ran": True},
               "finished_utc": "2026-10-01T01:00:00+00:00", "status": "completed"}
    (prior / "results.json").write_text(json.dumps(results))
    unfinished = {"started_utc": started, "protocol_sha256": sha, "out_dir": str(prior)}
    runs.write_text(json.dumps({"A": unfinished}))
    none, state = check()
    assert none is None and state["state"] == sealed_run.PRIOR_UNFINISHED == "unfinished"
    # Ruling 122, N2b: Test B must be pointed at the folder that the record names, in every state of the record
    with pytest.raises(sealed_run.RegistrationError, match="names the folder"):
        sealed_run._check_prior(pointed_wrong, record, [])
    write_manifest(prior)
    with pytest.raises(RuntimeError, match="Repair the record by hand") as refused:
        check()
    with pytest.raises(RuntimeError, match=f"Point Test B's prior_test_dir at {prior}, not {elsewhere}") as pointed:
        sealed_run._check_prior(pointed_wrong, record, [])                       # the repair, and where to point Test B
    assert "Repair the record by hand" in str(pointed.value)
    message = str(refused.value)
    assert "Test A's output is complete (it has a MANIFEST.sha256)" in message and "refused before its lock" in message
    for word in ("finished_utc", "manifest_sha256", "status", manifest_sha256(prior)):
        assert word in message, word

    def finish(coverage_ran=True):
        results["coverage"] = {"ran": coverage_ran}
        (prior / "results.json").write_text(json.dumps(results))
        (prior / "MANIFEST.sha256").unlink()
        write_manifest(prior)
        runs.write_text(json.dumps({"A": {**unfinished, "finished_utc": results["finished_utc"], "status": "x",
                                          "manifest_sha256": manifest_sha256(prior)}}))
        return check()

    none, state = finish(coverage_ran=False)
    assert none is None and state["state"] == sealed_run.PRIOR_NOT_RUN == "not run (coverage rule)"
    none, state = finish()
    assert none is None and state["state"] == sealed_run.PRIOR_NO_P1 == "finished, without stored P1 rows"
    assert state["missing_kept"] == list(KEPT)
    (prior / "kept").mkdir()
    pd.DataFrame({"x": [1]}).to_parquet(prior / "kept" / f"{_safe('P1')}.parquet")
    res, state = finish()
    assert res["test"] == "A" and state["state"] == sealed_run.PRIOR_POOLED == "pooled"
    with pytest.raises(sealed_run.RegistrationError, match="names the folder"):
        sealed_run._check_prior(pointed_wrong, record, [])                       # a finished record too (N2b)
    assert state["missing_kept"] == [k for k in KEPT if k != "P1"]                                    # E1 to E4 on its own quarter
    for name in KEPT:
        pd.DataFrame({"x": [1]}).to_parquet(prior / "kept" / f"{_safe(name)}.parquet")
    res, state = finish()
    assert state["state"] == "pooled" and "missing_kept" not in state


def test_ruling_116_is_in_the_lock_paragraph_the_pre_lock_list_and_part_e():
    """N1 to N4, Mn2, Mn3, Mn4 and Mn6, in the text, with the module's words."""
    import es.sealed_run as sealed_run
    from es.protocol import render_protocol
    text = render_protocol(_full_ctx())
    run = _section_of(text, "### The sealed pull and the sealed run", "### What Part A registers")
    failure = _section_of(text, "### Failure handling", "## Part E.")
    part_e = _section_of(text, "## Part E.", "## Part F.")
    # N3: the claim first, an output folder without a claim, the write access
    assert run.index("**The lock.** The first thing the lock does is to claim the test id: `docs/sealed/RUN_<test>.claim` "
                     "is created exclusively") < run.index("Then the output folder is created and the first lines of `RUN_LOG.txt`")
    assert "Because the claim comes first, an output folder that exists with no claim file for its test was never locked." in run
    assert "with no claim file for its test it was never locked, and the refusal says that nothing was read and to remove " \
           "the folder and run again" in run
    assert "the folder that holds the once-only records (`docs/sealed`) and the parent of the output folder must be writable" \
           in run and "(Ruling 116, N3)" in run
    # N4: the registered development data
    assert "must read the registered development data (its `dev_dirs` resolve to the registered folders" in run
    # N1, N2: the states, the refusal, the other frames, the fallback that stores P1
    assert "Test B must be pointed at the folder that the record names, in every state of the record, and is refused " \
           "otherwise; and it is refused if the output folder that Test A's record names is complete (it has its " \
           "`MANIFEST.sha256`) but the record lacks `finished_utc` or `manifest_sha256`, with the repair to make by hand " \
           "and a pointer to the recorded folder (Ruling 116, N2; Ruling 122, N2b)" in run
    # N2b: the registered folders
    out_a, out_b = (sealed_run.default_config(t).out_dir.resolve().relative_to(DEV_ROOT.parent.resolve()).as_posix()
                    for t in ("A", "B"))
    assert (out_a, out_b) == ("dart_dev/outputs/es/sealed/A", "dart_dev/outputs/es/sealed/B")
    assert sealed_run.default_config("B").prior_test_dir == sealed_run.default_config("A").out_dir
    assert f"must write the registered output folder (`{out_a}` for Test A and `{out_b}` for Test B, the `out_dir` of " \
           f"`default_config(test_id)`) and, for Test B, read Test A's registered folder (`{out_a}`, the `out_dir` of " \
           "`default_config(\"A\")`), both compared as resolved paths, because Test B finds Test A's output through the " \
           "folder that `RUNS.json[\"A\"]` records and must be pointed at it (Ruling 122, N2b)" in run
    assert "its `dev_dirs` resolve to the registered folders, `DEV_DATA_DIRS`" in run
    assert "they are stored as `kept/P1.parquet`, which Test B pools H-E from (Ruling 109, I6; Ruling 116, N1)" in run
    assert f'("{sealed_run.SCOPE_B_OWN_QUARTER}")' in run and "`results[\"test_a\"][\"missing_kept\"]` lists what is missing" in run
    assert f'"{sealed_run.SCOPE_B_OWN_QUARTER}"' in failure and "**Test A's other stored frames (Ruling 116, N1).**" in failure
    # Mn3: the stored files are read through Test A's manifest
    assert "After its own lock, Test B reads each stored file of Test A through Test A's `MANIFEST.sha256`" in run
    # Mn6: the module's words
    for constant in ("STATUS_HE_ALONE", "WORDING_HE_ALONE", "SCOPE_B_ALONE", "STATUS_HB_NO_TEST_A"):
        assert "Test A's output is not available" in getattr(sealed_run, constant) and getattr(sealed_run, constant) in failure
    assert "did not run" not in " ".join(getattr(sealed_run, c) for c in ("STATUS_HE_ALONE", "WORDING_HE_ALONE", "SCOPE_B_ALONE",
                                                                            "STATUS_HB_NO_TEST_A"))
    # Part E: the line of Ruling 116
    entry = dict(re.findall(r"^- \*\*Ruling (\d+)\*\* \((?:dev|2A|2B)\): (.*)$", part_e, flags=re.M))["116"]
    for word in ("(N1)", "(N2)", "(N3)", "(N4)", "(Mn2)", "(Mn3)", "(Mn4)", "(Mn6)", "stored P1 rows", "refuses Test B",
                 "resolved path", "through Test A's manifest", "rows on the hour", "twelve five-minute prices",
                 "because Test A may have run"):
        assert word in entry, word
    assert "did not run" not in text.split("## Part E.")[1].split("## Part F.")[0]                    # the old wording is gone


def test_the_new_guards_of_the_lock_do_what_the_text_says(tmp_path, monkeypatch):
    """Ruling 116, N3, N4, Mn2 and Mn3, against the real functions."""
    import os

    import es.sealed_run as sealed_run
    from es.analyses import ANALOGS
    from es.sealed_pull import write_manifest
    from es.sealed_run import SealedConfig
    arm = ANALOGS["AQ1"]
    # N3: an output folder that exists with no claim file was never locked: the refusal says that nothing was read
    out = tmp_path / "folder" / "out"
    out.mkdir(parents=True)
    message = sealed_run._out_exists_message(out)
    assert "no claim was made for this test" in message and "nothing was read" in message and "remove" in message
    (out / "RUN_LOG.txt").write_text("started x\nclaim /x/RUN_A.claim\n")                          # but one that took its claim counts
    assert "a sealed test runs once" in sealed_run._out_exists_message(out)
    # N3: the folders the lock writes in must be writable, before the lock
    cfg = SealedConfig("A", arm, arm, [], "2026-03-31", out_dir=tmp_path / "locked" / "out")
    locked = tmp_path / "locked"
    locked.mkdir()
    os.chmod(locked, 0o555)
    try:
        with pytest.raises(PermissionError, match="is not writable"):
            sealed_run._check_writable(cfg)
    finally:
        os.chmod(locked, 0o755)
    sealed_run._check_writable(cfg)
    # N4: a sealed arm reads the registered development folders only
    sealed = sealed_run.default_config("A")
    sealed.dev_dirs = (tmp_path / "elsewhere",)
    with pytest.raises(ValueError, match="registered development data"):
        sealed_run._check_sealed_config(sealed, sealed_run.N_REP)
    # Ruling 122, N2b: a sealed run writes the registered output folder, and Test B pools with Test A's registered folder
    sealed = sealed_run.default_config("A")
    sealed.out_dir = tmp_path / "elsewhere" / "A"
    with pytest.raises(ValueError, match="writes the registered output folder"):
        sealed_run._check_sealed_config(sealed, sealed_run.N_REP)
    sealed = sealed_run.default_config("B")
    sealed.prior_test_dir = tmp_path / "elsewhere" / "A"
    with pytest.raises(ValueError, match="pools with Test A's registered output folder"):
        sealed_run._check_sealed_config(sealed, sealed_run.N_REP)
    sealed_run._check_sealed_config(sealed_run.default_config("B"), sealed_run.N_REP)               # the registered one passes
    # Mn3: a stored file is read through Test A's manifest
    prior = tmp_path / "A_out"
    (prior / "kept").mkdir(parents=True)
    pd.DataFrame({"x": [1, 2]}).to_parquet(prior / "kept" / "P1.parquet")
    pd.DataFrame({"x": [3]}).to_parquet(prior / "kept" / "other.parquet")
    write_manifest(prior)
    assert sealed_run._read_verified(prior, "kept/P1.parquet")["x"].tolist() == [1, 2]
    pd.DataFrame({"x": [9]}).to_parquet(prior / "kept" / "P1.parquet")                          # changed after the manifest
    with pytest.raises(RuntimeError, match="changed since Test A's manifest was written"):
        sealed_run._read_verified(prior, "kept/P1.parquet")
    with pytest.raises(RuntimeError, match="is not listed in the MANIFEST.sha256"):
        sealed_run._read_verified(prior, "kept/unlisted.parquet")
    # Mn2: the interpreter is compared by the path it resolves to (python, python3 and a link are one interpreter)
    import hashlib

    sealed_dir = tmp_path / "sealed"
    sealed_dir.mkdir()
    monkeypatch.setattr(sealed_run._registration, "SEALED_DIR", sealed_dir)
    link = tmp_path / "python-link"
    link.symlink_to(sys.executable)
    live = sealed_run.live_environment()

    def bound(executable):
        pins = sealed_dir / "ENV_PINS.json"
        pins.write_text(json.dumps({"executable": str(executable), "versions": live["versions"]}, indent=2) + "\n")
        protocol = tmp_path / "PROTOCOL.md"
        protocol.write_text(f"- `ENV_PINS.json` SHA-256: `{hashlib.sha256(pins.read_bytes()).hexdigest()}`\n")
        return protocol

    assert sealed_run._check_environment(bound(link))["pins"] == "matched"                       # a link to the same binary
    other = tmp_path / "another-python"
    other.write_text("#!/bin/sh\n")
    with pytest.raises(sealed_run.RegistrationError, match="executable"):
        sealed_run._check_environment(bound(other))                                              # another file: another interpreter


def test_the_template_carries_the_guidance_of_ruling_120_and_the_gate_ignores_it(tmp_path):
    """Ruling 120, M21, and Ruling 140: one comment line for each kind of record, ignored by the parser, and the template
    still parses once filled."""
    from es.protocol import registration_template
    from es.registration import _GATE_KEYS, parse_registration
    template = registration_template()
    guidance = [ln for ln in template.splitlines() if ln.startswith(">")]
    assert len(guidance) == 3                                                    # an OSF registration, a release, a local record
    osf, public, local = guidance
    assert "for an OSF registration, fill this in only after OSF shows the registration as registered (approved), not as " \
           "pending approval" in osf
    assert "copy registered_utc from the approved registration" in osf and "Ruling 120" in osf
    assert "copy `url` and `publishedAt` from `gh release view <tag> --json url,publishedAt` into `public_record` and " \
           "`registered_utc`" in public
    assert "`local_record` is `author-statement-` and the date of the author's authorisation (YYYY-MM-DD)" in local
    assert "registered_utc is the UTC time at which that authorisation was recorded" in local and "Rulings 140 and 141" in local
    for line in guidance:
        assert line.startswith("> Guidance, which the gate ignores: ")
        assert not re.match(r"^(\w+):[ \t]*(\S+)[ \t]*$", line)                         # not a `key: value` line
        assert not any(line.startswith(f"{k}:") for k in _GATE_KEYS)                  # nor a line naming a gate key
    reg = tmp_path / "REGISTRATION.md"
    for kind in THREE_RECORDS:
        filled = _filled_template(template, kind, utc="2026-10-25T18:30:00Z")        # embargo_until ships filled (Ruling 122)
        assert sum(ln.startswith(">") for ln in filled.splitlines()) == 3
        reg.write_text(filled)
        with_guidance = parse_registration(reg)
        assert with_guidance["registered_utc"] == "2026-10-25T18:30:00Z"
        reg.write_text("\n".join(ln for ln in filled.splitlines() if not ln.startswith(">")) + "\n")   # the record without them
        assert parse_registration(reg) == with_guidance


# ------------------------------------------------------------------ Ruling 122 (the sealed-path integration walk) and COVERAGE
def _run_and_failure_sections(text):
    return (text[text.index("### The sealed pull and the sealed run"):text.index("### Failure handling")],
            text[text.index("### Failure handling"):text.index("## Part E.")])


def test_the_expected_hours_of_the_coverage_tables_are_clock_hours_and_the_text_says_so(tmp_path):
    """The pull's COVERAGE.csv and the run's tables/coverage.csv count against expected_hours, the clock hours of the local
    days (25 on a fall-back day), while the day check expects 23 where the downloaders drop the repeated hour: the share is
    just under 1 on that day, and the 90% rule does not notice. Synthetic ISO-NE rows for the fall-back day of 2025."""
    import datetime as dt
    import types

    import es.sealed_run as sealed_run
    from es import sealed_pull
    from es.config import MARKETS
    from es.protocol import render_protocol
    run, _ = _run_and_failure_sections(render_protocol(_full_ctx()))
    nodes = [n for m, n in sealed_pull._node_map_nodes() if m == "ISONE"]
    first, last = "2025-11-01", "2025-11-03"                                  # 2 November 2025 is a fall-back day
    full = _isone_rows(nodes, first, last)                                     # every clock hour: 24 + 25 + 24
    repeated = [pd.Timestamp("2025-11-02 05:00", tz="UTC"), pd.Timestamp("2025-11-02 06:00", tz="UTC")]
    written = full[~full["timestamp_utc"].isin(repeated)]                      # what the ISO-NE downloader writes: 23 hours
    root = tmp_path / "pull"
    root.mkdir()
    written.to_parquet(root / "isone_da_rt_x.parquet", index=False)
    assert sealed_pull._day_gaps(root, "ISONE", first, last) == ([], {}, {})            # the day check wants 23 and has 23
    cov = sealed_pull.coverage_manifest(root, first, last, _sealed=True)
    row = cov[(cov["series"] == "price") & (cov["node"] == nodes[0])].iloc[0]
    assert (row["hours"], row["expected_hours"]) == (24 + 23 + 24, 24 + 25 + 24)         # COVERAGE.csv counts clock hours
    assert row["share"] == pytest.approx(71 / 73)
    full.to_parquet(root / "isone_da_rt_x.parquet", index=False)                        # a day that kept the repeated hour
    _, short, over = sealed_pull._day_gaps(root, "ISONE", first, last)
    assert short == {} and over == {(n, dt.date(2025, 11, 2)): (25, 23, 0) for n in nodes}
    # Test B's window, by the module's own calendar: one fall-back day, 1 November 2026; Test A's has none
    b_first, b_last = sealed_pull.SEALED_WINDOWS["B"][1:3]
    a_first, a_last = sealed_pull.SEALED_WINDOWS["A"][1:3]
    days_b = [d.date() for d in pd.date_range(b_first, b_last, freq="D")]
    assert [d.isoformat() for d in days_b if sealed_pull.expected_hours("NYISO", d.isoformat(), d.isoformat()) == 25] == [
        "2026-11-01"]
    assert [sealed_pull.expected_hours(m, a_first, a_last) for m in MARKETS] == [92 * 24] * len(MARKETS)
    clock = sealed_pull.expected_hours("NYISO", b_first, b_last)
    kept = sum(sealed_pull._expected_day_hours("NYISO", d) for d in days_b)
    assert (clock, kept, sealed_pull.expected_hours("MISO", b_first, b_last)) == (2209, 2207, 2208)     # MISO: fixed EST
    assert "Test B holds one, 1 November 2026, so there a NYISO, ISO-NE or integrated node with the 23 hours of that day " \
           f"shows {kept:,} of {clock:,} hours, a share of {kept / clock:.5f} and not 1; the coverage rule of the run (at " \
           "least 90%) does not notice the difference." in run
    assert "2,207 of 2,209 hours, a share of 0.99909" in run
    # the coverage rule of the run: against the same expected_hours, at 90% (1,988.1 of 2,209 hours: 1,989 are needed)
    assert "a node is usable with at least 90% of its expected deployment hours (`expected_hours` in `tables/coverage.csv`: " \
           "the clock hours of the deployment window's local settlement days, as in the pull's `COVERAGE.csv`)" in run
    assert 'ok.to_csv(out / "tables" / "coverage.csv", index=False)' in Path(sealed_run.__file__).read_text()
    analysis = types.SimpleNamespace(deploy=types.SimpleNamespace(
        start=b_first, end=b_last, mask=lambda panel: pd.Series(True, index=panel.index)))
    for hours, ok in ((1988, False), (1989, True)):
        table = sealed_run.coverage(pd.DataFrame({"market": "NYISO", "node": "X", "n": range(hours)}), analysis)
        assert (table.loc[0, "expected_hours"], table.loc[0, "hours"], bool(table.loc[0, "ok"])) == (2209, hours, ok)


def test_the_embargo_line_ships_filled_and_informational_and_the_gate_does_not_use_it(tmp_path):
    """Ruling 122, C2-2: the docstring no longer says that all five lines block; embargo_until is `none`, it is not a gate key
    and nothing else in es/ uses it; a multi-word placeholder there was ignored by the parser all along. Ruling 140: the
    template has three kinds of record, so it says that the embargo matters only for an OSF registration."""
    import inspect

    from es.protocol import registration_template
    from es.registration import _GATE_KEYS, RegistrationError, parse_registration
    template = registration_template()
    assert "embargo_until: none" in template.splitlines() and "embargo_until" not in _GATE_KEYS
    doc = " ".join(inspect.getdoc(registration_template).split())
    assert "Each of the five lines" not in doc and "Four of its five lines" not in doc
    assert "are ``<paste ...>`` placeholders with their instructions" in doc and "so a gate line left unfilled blocks the " \
           "sealed windows" in doc
    assert "The last line, ``embargo_until``, is informational (Ruling 122, C2-2): it is not a gate key, nothing in ``es`` " \
           "uses it, and it is pre-filled as ``none``." in doc
    assert "Replace the whole value of each `<paste ...>` line you keep" in template
    assert "The last line, `embargo_until: none`, is already filled in and is informational (the sealed code does not use " \
           "it): it matters only for an OSF registration, and you change it only if you embargo that registration." in template
    for path in sorted((DEV_ROOT / "es").glob("*.py")):                          # nothing but the template names the field
        assert path.name == "protocol.py" or "embargo_until" not in path.read_text(), path.name
    filled = [ln for ln in _filled_template(template, "osf (address only)").splitlines() if not ln.startswith("embargo_until:")]
    reg = tmp_path / "REGISTRATION.md"
    reg.write_text("\n".join(filled + ["embargo_until: none", ""]))
    shipped = parse_registration(reg)
    assert shipped["embargo_until"] == "none" and shipped["osf_doi"] is None             # as the template ships the line
    reg.write_text("\n".join(filled + ["embargo_until: <paste the date the embargo ends, or none for a public registration>",
                                       ""]))
    assert "embargo_until" not in parse_registration(reg)                                # the old line: no pair, ignored
    reg.write_text("\n".join(filled + ["embargo_until: <date>", ""]))
    with pytest.raises(RegistrationError, match="placeholder value for \\['embargo_until'\\]"):
        parse_registration(reg)                                                          # a one-word placeholder is refused


def test_the_environment_sentence_of_ruling_122_is_in_part_d_and_in_the_failure_handling(tmp_path, monkeypatch):
    """C2-3: ENV_PINS.json binds the interpreter path and the library versions; if /opt/anaconda3 is updated or moved before
    Test B has run, Test B is refused before its lock. The sentence is only there when the protocol binds ENV_PINS.json."""
    import hashlib
    import inspect

    import es.sealed_run as sealed_run
    from es.protocol import render_protocol
    text = render_protocol(_full_ctx())
    run, failure = _run_and_failure_sections(text)
    assert "`ENV_PINS.json` binds the interpreter path and the library versions of the environment in which this protocol " \
           "was generated, the Anaconda installation at `/opt/anaconda3`: if it is updated, moved or reinstalled before " \
           "Test B has run, in January 2027, so that the interpreter's path or the version of python or of a pinned library " \
           "differs, Test B is refused before its lock until the environment is restored, so it must not be updated or " \
           "moved until Test B has run (Ruling 122, C2-3);" in run
    assert "**The environment (Ruling 122, C2-3).** `ENV_PINS.json` binds the interpreter path and the library versions of " \
           "the environment in which this protocol was generated, the Anaconda installation at `/opt/anaconda3`. If that " \
           "installation is updated, moved or reinstalled before Test B has run, in January 2027, so that the " \
           "interpreter's path or the version of python or of a pinned library differs, Test B is refused before its " \
           "lock, with nothing spent, until the environment is restored: do not update or move it until Test B has run." \
           in failure
    assert failure.index("**Disk space and write access") < failure.index("**The environment (Ruling 122") < failure.index(
        "**Runbook.**")
    assert "January 2027, once final prices are posted" in text                          # Part A dates Test B the same way
    unbound = _full_ctx()
    unbound["sha"] = {k: v for k, v in unbound["sha"].items() if k != "env"}
    run_u, failure_u = _run_and_failure_sections(render_protocol(unbound))
    assert "Ruling 122, C2-3" not in run_u and "The environment (Ruling 122" not in failure_u      # nothing is pinned then
    # Part E
    part_e = text[text.index("## Part E."):text.index("## Part F.")]
    entry = next(ln for ln in part_e.splitlines() if ln.startswith("- **Ruling 122**"))
    assert entry.startswith("- **Ruling 122** (2A): After the sealed-path integration walk.")
    assert all(word in entry for word in ("(N2b) A sealed run writes the registered output folder", "(C2-2)", "(C2-3)"))
    assert "(C2-3) `ENV_PINS.json` ties Test B to the environment of this protocol" in entry
    numbers = [int(n) for n in re.findall(r"^- \*\*Ruling (\d+)\*\*", part_e, flags=re.M)]
    assert numbers == sorted(numbers) and numbers[-3:] == [122, 140, 141]               # Rulings 140 and 141 follow
    # what the sentence says is what the code does: the check is before the claim, and a moved or updated environment fails it
    source = inspect.getsource(sealed_run.run_sealed)
    assert source.index("_check_environment(") < source.index("_take_claim(") < source.index("out.mkdir(")
    sealed_dir = tmp_path / "sealed"
    sealed_dir.mkdir()
    monkeypatch.setattr(sealed_run._registration, "SEALED_DIR", sealed_dir)
    live = sealed_run.live_environment()

    def check(executable=None, **versions):
        pins = sealed_dir / "ENV_PINS.json"
        pins.write_text(json.dumps({"executable": str(executable or live["executable"]),
                                    "versions": {**live["versions"], **versions}}, indent=2) + "\n")
        protocol = tmp_path / "PROTOCOL.md"
        protocol.write_text(f"- `ENV_PINS.json` SHA-256: `{hashlib.sha256(pins.read_bytes()).hexdigest()}`\n")
        return sealed_run._check_environment(protocol)

    assert check()["pins"] == "matched"
    with pytest.raises(sealed_run.RegistrationError, match="pandas: pinned '0.0.1'"):
        check(pandas="0.0.1")                                                           # a library updated
    with pytest.raises(sealed_run.RegistrationError, match="python: pinned '2.7.18'"):
        check(python="2.7.18")                                                          # python updated
    with pytest.raises(sealed_run.RegistrationError, match="executable: pinned"):
        check(executable=tmp_path / "moved" / "bin" / "python3")                        # the installation moved


# ------------------------------------------------------------------------------------------------ Ruling 121 (the pull)
def test_a_chunk_is_accepted_as_incomplete_only_as_the_text_of_ruling_121_says(tmp_path):
    """Rulings 111 and 121 against the real _fetch_parse and IncompleteChunks: only whole answers that were refused count, in
    every attempt of the call; the one with the fewest problems (the last of equals) is kept; no rows at all is zero rows; a
    network failure or an unreadable answer stops the pull; so does a chunk that was not named. Synthetic answers."""
    from es import sealed_pull
    start, end = pd.Timestamp("2026-06-21 00:00", tz="UTC"), pd.Timestamp("2026-06-21 02:00", tz="UTC")
    validate = sealed_pull._grid_validator(start, end, 15, "CAISO", False, "fmm15")
    grid = pd.date_range(start, end, freq="15min", inclusive="left")                       # eight intervals

    def answer(rows, node="n"):
        return pd.DataFrame({"node": node, "timestamp_utc": grid[:rows], "fmm15": 1.0})

    empty = sealed_pull._empty_table(sealed_pull._FMM_COLUMNS, "OASIS error 1000: No data returned for the specified selection")

    def fetch_chunk(answers, keys=("k",)):
        """answers: one per attempt, a table (a whole answer), "net" (the request raised) or "bad" (an unreadable answer)."""
        log = []
        incomplete = sealed_pull.IncompleteChunks("fmm", keys, lambda *lines: log.extend(lines))
        calls = iter(range(len(answers)))

        def fetch(url):
            i = next(calls)
            if isinstance(answers[i], str) and answers[i] == "net":
                raise OSError("timed out")
            return str(i).encode()

        def parse(blob):
            got = answers[int(blob)]
            if isinstance(got, str):
                raise ValueError("not a zip file")
            return got

        def call():
            return sealed_pull._fetch_parse("http://x", parse, "chunk", fetch, lambda s: None, None, attempts=len(answers),
                                            validate=validate, key="k", incomplete=incomplete)

        return call, incomplete, log

    # F2: the answer with the fewest problems, the last of equals; nothing is cached
    call, incomplete, log = fetch_chunk([answer(6), answer(7, "a"), answer(7, "b"), answer(5)])
    part, cached = call()
    assert cached is False and len(part) == 7 and set(part["node"]) == {"b"}
    assert "short: 1 of 8 15-minute intervals missing" in incomplete.accepted["k"]
    assert any(line.startswith("chunk_accepted: fmm k | ") for line in log)
    # F1: an answer with no rows counts with the refused ones (here beside two short answers), and all of them being
    # empty gives zero rows, with what the server said
    call, incomplete, _ = fetch_chunk([answer(6), empty, answer(7)])
    part, _ = call()
    assert len(part) == 7
    call, incomplete, _ = fetch_chunk([empty, empty, empty])
    part, cached = call()
    assert len(part) == 0 and cached is False
    assert "no rows came (OASIS error 1000: No data returned for the specified selection)" in incomplete.accepted["k"]
    # F2: one network failure, or one unreadable answer, in the call, and the chunk is not accepted
    for odd in ("net", "bad"):
        call, incomplete, _ = fetch_chunk([answer(6), odd, answer(7)])
        with pytest.raises(sealed_pull.ChunkError) as stopped:
            call()
        assert incomplete.accepted == {} and stopped.value.validator_failures == 2 and stopped.value.attempts == 3
    # a chunk that was not named stops the pull, whatever its answers
    call, incomplete, _ = fetch_chunk([answer(6), answer(7), answer(5)], keys=())
    with pytest.raises(sealed_pull.ChunkError, match="no data after 3 attempts"):
        call()
    # rows outside the interval that was asked for are never accepted
    outside = answer(6).assign(timestamp_utc=lambda f: f["timestamp_utc"] + pd.Timedelta(days=1))
    call, incomplete, _ = fetch_chunk([outside, outside, outside])
    with pytest.raises(sealed_pull.ChunkError):
        call()
    # the log names every chunk that gave up, but only one whose attempts were all refused answers can be named later
    root = tmp_path / "pull"
    root.mkdir()
    for key, refused in (("all", sealed_pull.CHUNK_ATTEMPTS), ("some", sealed_pull.CHUNK_ATTEMPTS - 1)):
        sealed_pull._log_chunk_failure(root, "fmm", sealed_pull.ChunkError(
            "no data", key=key, attempts=sealed_pull.CHUNK_ATTEMPTS, validator_failures=refused, reason="short"))
    log = (root / "PULL_LOG.txt").read_text()
    assert f"{sealed_pull.CHUNK_ATTEMPTS - 1} of {sealed_pull.CHUNK_ATTEMPTS} attempts failed the validator" in log
    assert sealed_pull._logged_chunk_failures(root) == {"fmm": {"all": "short"}}


def test_the_day_checks_message_names_the_cached_nyiso_archive_and_the_text_says_so(tmp_path):
    """Ruling 121, F4: the NYISO integrated archive of a month is cached whatever hours it holds; the refusal of the day check
    names the cached file whose deletion makes the next resume fetch the month again."""
    import datetime as dt
    import hashlib

    from es import sealed_pull
    from es.protocol import render_protocol
    root = tmp_path / "pull"
    (root / "raw").mkdir(parents=True)
    name = f"raw/{hashlib.sha256(sealed_pull.NYISO_URL.format(ym='202606').encode()).hexdigest()}.bin"
    pairs = [("N.Y.C.", dt.date(2026, 6, 22))]
    assert sealed_pull._cached_archive_hint(root, "nyiso_rtint", pairs) == ""                    # nothing cached: no hint
    (root / name).write_bytes(b"zip")
    hint = sealed_pull._cached_archive_hint(root, "nyiso_rtint", pairs)
    assert f"delete {name} (the archive of 2026-06) in the pull root, and the next resume fetches the month again" in hint
    assert "reused as it is on every resume" in hint
    assert sealed_pull._cached_archive_hint(root, "NYISO", pairs) == ""                          # only the integrated table
    text = render_protocol(_full_ctx())
    run, _ = _run_and_failure_sections(text)
    assert "The NYISO integrated archive of a month has no interval grid to be short of, so it is cached whatever hours " \
           "it holds and reused as it is on every resume; when the day check refuses the NYISO integrated table, its " \
           "message names the cached archive of each month concerned, and deleting that file makes the next resume fetch " \
           "the month again (Ruling 121, F4)." in run
    part_e = text[text.index("## Part E."):text.index("## Part F.")]
    entry = next(ln for ln in part_e.splitlines() if ln.startswith("- **Ruling 121**"))
    assert entry.startswith("- **Ruling 121** (2A): (F1) An answer that holds no rows") and "(F2)" in entry and "(F4)" in entry


# ---------------------------------------------------------------------- Ruling 122, C2-1 (the gate refuses a draft)
def test_verify_registration_refuses_a_protocol_that_is_not_ready_to_register_and_the_text_says_so(tmp_path):
    """Ruling 122, C2-1, against the real verify_registration, pull_sealed and run_sealed: a PROTOCOL.md that ready_to_register
    flags (no statement, the unfilled template, markers filled but no box ticked, boxes ticked but guidance left) is refused
    even though its hash is registered; a filled statement is accepted."""
    import inspect

    import es.sealed_pull as sealed_pull
    import es.sealed_run as sealed_run
    from es.protocol import ready_to_register, render_protocol
    from es.registration import RegistrationError, sha256_file, verify_registration
    cases = []

    def registered(knowledge):
        """A PROTOCOL.md rendered with this statement, and a REGISTRATION.md that names its hash: the paths of both."""
        folder = tmp_path / f"case{len(cases)}"
        cases.append(folder)
        folder.mkdir()
        protocol, registration = folder / "PROTOCOL.md", folder / "REGISTRATION.md"
        protocol.write_text(render_protocol(_full_ctx(knowledge=knowledge)))
        registration.write_text(f"protocol_sha256: {sha256_file(protocol)}\nosf_registration: https://osf.io/abcde/\n"
                                "registered_utc: 2026-10-24T18:30:00Z\n")
        return protocol, registration

    template = _template_text()
    drafts = (None, template, _fill_markers(template), _tick_rows(_fill_markers(template)), _drop_guidance(_fill_markers(template)))
    for knowledge in drafts:
        protocol, registration = registered(knowledge)
        assert ready_to_register(protocol.read_text())
        with pytest.raises(RegistrationError, match="is not ready to register") as refused:
            verify_registration(sha256_file(protocol), protocol, registration)
        message, n = str(refused.value), len(ready_to_register(protocol.read_text()))
        assert message.startswith("PROTOCOL.md is not ready to register: ") and f"{n} item{'s' if n != 1 else ''} still open" in message
        assert ("(the first two: " in message) == (n > 1)                                    # it names the first two
        assert "Fill docs/sealed/KNOWLEDGE_STATEMENT.md, re-run es.protocol.main(), and register that PROTOCOL.md" in message
    protocol, registration = registered(_tick_rows(_drop_guidance(_fill_markers(template))))
    assert ready_to_register(protocol.read_text()) == []
    assert verify_registration(sha256_file(protocol), protocol, registration)["osf_registration"] == "https://osf.io/abcde/"
    # both pulls and both runs call it first (pull_sealed before anything is downloaded, run_sealed before the lock)
    pull_source, run_source = inspect.getsource(sealed_pull.pull_sealed), inspect.getsource(sealed_run.run_sealed)
    assert "verify_registration(protocol_sha256)" in pull_source
    assert run_source.index("verify_registration(") < run_source.index("_take_claim(")
    text = render_protocol(_full_ctx())
    run, failure = _run_and_failure_sections(text)
    assert "`verify_registration` also refuses a protocol that is not ready to register (Ruling 122, C2-1), so the pull " \
           "refuses an unfilled knowledge statement before any download." in run
    assert "`verify_registration` also refuses a protocol that is not ready to register: `es.protocol.ready_to_register` " \
           "must find nothing in the text of `PROTOCOL.md`, that is, no author marker (an `AUTHOR` placeholder of the " \
           "template), exactly one tick in every item-1 row of the knowledge statement and no template guidance left in " \
           "it. The refusal says that `PROTOCOL.md` is not ready to register, how many items are still open and the first " \
           "two, and to fill in `docs/sealed/KNOWLEDGE_STATEMENT.md`, re-run `es.protocol.main()` and register that " \
           "`PROTOCOL.md` (the SHA-256 in the `PROTOCOL.sha256` it writes) (Ruling 122, C2-1);" in run
    assert "**Refused before any download and before the lock: the protocol is not ready to register (unfilled knowledge " \
           "statement) (Ruling 122, C2-1).** `verify_registration` refuses a `PROTOCOL.md` that " \
           "`es.protocol.ready_to_register` flags: an author marker (an `AUTHOR` placeholder of the template), an item-1 " \
           "row of the knowledge statement without exactly one tick, or template guidance left in. Both sealed pulls and " \
           "both sealed runs call it first, so none of them downloads or reads anything under such a protocol. The " \
           "refusal says how many items are still open and names the first two, and says to fill in " \
           "`docs/sealed/KNOWLEDGE_STATEMENT.md`, re-run `es.protocol.main()` and register that `PROTOCOL.md`" in failure
    assert failure.index("**Refused before any download and before the lock") < failure.index("**Disk space and write access")
    part_e = text[text.index("## Part E."):text.index("## Part F.")]
    entry = next(ln for ln in part_e.splitlines() if ln.startswith("- **Ruling 122**"))
    assert "(C2-1) `verify_registration` refuses a PROTOCOL.md that `ready_to_register` flags" in entry


# -------------------------------------------------------------------------------- final wording (M23, M24 and M27)
def test_the_other_levels_of_alpha_are_said_as_the_sealed_run_computes_them():
    """Seat M, M23: the audit-robustness family does compute its H-B contrasts at the cutoff levels 1%, 2.5% and 10% (the labels
    stay at 5%) and the economic analysis the curve over alpha; only the failure counts at the other levels and the two narrower
    bands are development-only. Against the real es.secondary.audit_family, the economic code and the sealed run."""
    import inspect

    import es.run_economics as run_economics
    import es.secondary as secondary
    import es.sealed_run as sealed_run
    from es.protocol import render_protocol
    text = render_protocol(_full_ctx())
    part = _section_of(text, "### What Part A registers that the sealed run does not compute (Ruling 119)", "### Failure handling")
    assert "the failure counts at the levels of alpha other than 5%, and the failure bands" in part
    assert "(section 5 item 1); the levels of alpha other than 5%," not in part                  # the earlier, wider claim
    assert "neither compute nor report them" in part and "and computes the economic curve over alpha" not in part
    assert "- **Other levels of alpha that the sealed run does compute (Seat M, M23):** the audit-robustness family re-runs the " \
           "H-B1 and H-B2 contrasts at the cutoff levels 1%, 2.5% and 10%, with the labels fixed at 5%, as descriptive " \
           "diagnostics (the `es.secondary` docstring quoted under the secondary families above), and the economic analysis " \
           "computes the economic curve over alpha of section 5 item 6, for P6 and P1." in part
    # the audit family, run for real with a contrast that records its cutoff levels and a score that is given none
    levels, score_calls = [], []

    def score(kind, block, estimator, seed):
        score_calls.append((kind, block, estimator, seed))

    def contrast(alt, base, alpha, n_rep):
        levels.append(alpha)
        return {"t_hat": 0.0, "p": 0.5}

    rows = secondary.audit_family(score, n_rep=1, contrast=contrast)
    assert len(rows) == 18 and sorted(set(levels)) == [0.01, 0.025, 0.05, 0.1]
    assert sorted({a for a in levels if a != 0.05}) == [0.01, 0.025, 0.1] and all(len(c) == 4 for c in score_calls)
    assert "(labels stay at 5%)" in secondary.__doc__ and "cutoff alpha 1%, 2.5% and 10%" in secondary.__doc__
    assert "audit_family(" in inspect.getsource(sealed_run)                                       # the sealed run calls it
    source = inspect.getsource(run_economics)                                                     # and computes the curve
    assert "for alpha_c in (0.01, 0.025, 0.05, 0.10):" in source
    assert 'comparators[f"curve_P1_a{alpha_c}"]' in source and 'comparators[f"curve_P6_a{alpha_c}"]' in source
    # Part E: the Ruling 119 line says what is development-only, and the Ruling 105(c) line no longer calls the flags approximate
    entries = dict(re.findall(r"^- \*\*Ruling (\d+)\*\* \((?:dev|2A|2B)\): (.*)$", _section_of(text, "## Part E.", "## Part F."),
                              flags=re.M))
    assert "the failure counts at the levels of alpha other than 5% and the failure bands other than [0.5, 2] are " \
           "development-only (the audit-robustness family and the economic curve do use other levels of alpha, Seat M, " \
           "M23)" in entries["119"]
    assert "development-only; figures are drawn" not in entries["119"]                           # the earlier, wider claim


def test_the_registered_output_folders_are_read_from_default_config(monkeypatch, tmp_path):
    """Seat M, M24: the config bullet prints the folders that default_config gives, relative to the repository, not a typed
    path: moving OUT_DIR moves the text, and a folder outside the repository is printed as it is."""
    import es.sealed_run as sealed_run
    from es.protocol import render_protocol

    def bullet():
        run = _section_of(render_protocol(_full_ctx()), "### The sealed pull and the sealed run", "### What Part A registers")
        return run[run.index("must write the registered output folder"):run.index("and must take its code from")]

    assert bullet().startswith("must write the registered output folder (`dart_dev/outputs/es/sealed/A` for Test A and "
                               "`dart_dev/outputs/es/sealed/B` for Test B, the `out_dir` of `default_config(test_id)`) and, for "
                               "Test B, read Test A's registered folder (`dart_dev/outputs/es/sealed/A`, the `out_dir` of "
                               "`default_config(\"A\")`)")
    monkeypatch.setattr(sealed_run, "OUT_DIR", DEV_ROOT.parent / "elsewhere" / "es")
    assert "(`elsewhere/es/sealed/A` for Test A and `elsewhere/es/sealed/B` for Test B" in bullet()
    assert "read Test A's registered folder (`elsewhere/es/sealed/A`," in bullet()
    monkeypatch.setattr(sealed_run, "OUT_DIR", tmp_path / "es")
    assert f"(`{tmp_path / 'es' / 'sealed' / 'A'}` for Test A and `{tmp_path / 'es' / 'sealed' / 'B'}` for Test B" in bullet()


def test_the_persistence_status_table_counts_each_component_on_its_own():
    """Seat M, M27: H-E's score and P&L are two rows built from by_component of the registered protocol_inputs.json, not one
    row that counts a node once for each status it has in either component (33 corrected + 3 clipped high + 1 no signal for 33
    nodes). The numbers are computed from the inputs."""
    import copy

    from es.protocol import load_inputs, render_protocol
    raw, doc = load_inputs(DEV_ROOT / "outputs/es/analogs/protocol_inputs.json")
    header = "| Block | Component | Nodes | Corrected | Clipped high | Clipped low | No signal | Uncorrected | Note |"

    def table(inputs):
        part_c = _section_of(render_protocol(_full_ctx(inputs=inputs)), "## Part C.", "## Part D.")
        lines = part_c.splitlines()
        first = lines.index(header)
        rows = []
        for line in lines[first + 2:]:
            if not line.startswith("|"):
                break
            rows.append([cell.strip() for cell in line.strip("|").split("|")])
        return part_c, rows

    part_c, rows = table(doc)
    expected = [[block, {"score": "score", "pnl": "P&L"}[component], str(v["phi_nodes"]),
                 *[str(v["by_component"][component][s]) for s in ("corrected", "clipped_high", "clipped_low", "no_signal",
                                                                  "uncorrected")], "—"]
                for block, v in doc["phi_status"].items() for component in v["components"]]
    assert rows == expected and len(rows) == 5
    he = {row[1]: row[2:] for row in rows if row[0] == "H-E"}
    assert he["score"] == ["33", "33", "0", "0", "0", "0", "—"] and he["P&L"] == ["33", "29", "3", "0", "1", "0", "—"]
    assert all(sum(int(c) for c in row[3:8]) == int(row[2]) for row in rows)           # each component adds up to its nodes
    assert ["H-E", "all (a node counts once per status)"] not in [row[:2] for row in rows]
    assert "by block and by daily component (nodes with each status in that component; Task 17 re-review N-2, Seat M, M27)" in part_c
    assert "a node can be corrected in one and clipped in the other, so the counts of the components of a block are not added" \
           in part_c
    # the numbers follow the inputs: change one count and the row changes
    changed = copy.deepcopy(doc)
    changed["phi_status"]["H-E"]["by_component"]["pnl"].update(corrected=28, no_signal=2)
    assert {r[1]: r[2:] for r in table(changed)[1] if r[0] == "H-E"}["P&L"] == ["33", "28", "3", "0", "2", "0", "—"]
    # an older file without by_component gets one marked row of its block-level counts; a component that is not counted raises
    legacy = copy.deepcopy(doc)
    for v in legacy["phi_status"].values():
        v.pop("by_component")
    assert [r[:3] for r in table(legacy)[1] if r[0] == "H-E"] == [["H-E", "all (a node counts once per status)", "33"]]
    broken = copy.deepcopy(doc)
    del broken["phi_status"]["H-E"]["by_component"]["pnl"]
    with pytest.raises(ValueError, match="lists the component 'pnl', which by_component does not count"):
        render_protocol(_full_ctx(inputs=broken))


# ---------------------------------------------------------------------------------------------------- Rulings 140 and 141
def test_no_sentence_of_the_protocol_says_that_the_design_is_registered_on_osf():
    """Ruling 140: the protocol had not been registered or published anywhere, and its record is one of three (an OSF
    registration, a public release, the author's local record). OSF is named as an accepted route that is not used, and in
    the notes that say what was rewritten; no sentence makes it the place where the design is registered, uploaded or
    recorded. The spec itself, which the renderer does not edit, still holds the old sentences."""
    from es.protocol import registration_template, render_protocol
    spec = SPEC.read_text()
    text = render_protocol(_full_ctx(knowledge="Name and date: Erce Erez, 2026-10-27\n"))
    forbidden = [r"registers? [^.]{0,40}on OSF", r"registered on OSF", r"registering on OSF", r"right after OSF registration",
                 r"(uploaded|posted) to the OSF", r"\bgoes? to OSF", r"\bon OSF\b", r"the OSF registration\b", r"OSF id\b",
                 r"OSF protocol"]
    for pattern in forbidden:
        found = re.search(pattern, text)
        assert not found, (pattern, text[max(0, found.start() - 150):found.end() + 100] if found else "")
    assert re.search(r"registers on OSF", spec) and "right after OSF registration" in spec and "OSF registration as" in spec
    assert text.startswith("# Pre-specified protocol: sealed out-of-sample tests") and "Pre-registration" not in text.split("## How to read this protocol")[0]
    assert "OSF" not in text.split("## How to read this protocol")[0]   # the title block
    for needle in ("The OSF route stays accepted by the gate and is not used.", "an OSF registration (`osf_doi` and/or "
                   "`osf_registration`)", "the author's local record", "**local record**", "a public release of the code repository"):
        assert needle in text, needle
    assert "OSF" in registration_template()                                  # the template keeps the OSF route as one of three


def test_part_f_says_what_the_record_of_the_protocol_is_and_that_test_a_runs_on_a_local_record():
    from es.protocol import render_protocol
    from es.registration import _GATE_KEYS
    text = render_protocol(_full_ctx())
    f = _section_of(text, "## Part F.", "## Part G.")
    record = f[f.index("**The record of the protocol (Rulings 140 and 141).**"):f.index("For the run of Test A")]
    for needle in (
            "names the SHA-256 of this protocol (exactly the file whose SHA-256 is in `PROTOCOL.sha256`), the UTC time of "
            "the record and exactly one registration of it, which is one of three: an OSF registration (`osf_doi` and/or "
            "`osf_registration`), a public release of the code repository (`public_record`, the address of a tagged GitHub "
            "release, whose publication time GitHub records) or the author's local record (`local_record`, "
            "`author-statement-` and the date of the author's authorisation). A record that names more than one kind is "
            "refused.",
            "The gate reads no network, so it cannot check any of them against the outside world: what protects the sealed "
            "windows is that nothing opens without a deliberate, well-formed record of the exact hash of this protocol, and "
            "what a document says about the record must be exactly what the record is."):
        assert needle in record, needle
    # the keys the text names are the keys the gate reads
    assert {k for k in re.findall(r"`(\w+)`", record) if k in _GATE_KEYS} == set(_GATE_KEYS) - {"protocol_sha256", "registered_utc"}
    test_a = f[f.index("For the run of Test A"):f.index("**Immutability.**")]
    assert test_a == (
        "For the run of Test A the record is the **local record**: `docs/sealed/REGISTRATION.md` holds the SHA-256 of this "
        "protocol and the UTC time at which the author's authorisation was recorded; no registry, repository or other third "
        "party holds the protocol before the pull; and this protocol and the code are published with the code repository "
        "after the run. The author authorised this on 26 October 2026. The OSF route stays accepted by the gate and is not "
        "used. For Test B the author intends a public release of the code repository before the pull; that is an "
        "intention, not a rule of the gate.\n\n")
    assert f.index("**The record of the protocol") < f.index("For the run of Test A") < f.index("**Immutability.**")
    assert "are bound to the protocol by the SHA-256 lines above" in f and " on OSF" not in f


def test_rulings_140_and_141_are_in_part_e_with_their_ledger():
    from es.protocol import render_protocol
    text = render_protocol(_full_ctx())
    part_e = _section_of(text, "## Part E.", "## Part F.")
    entries = dict(re.findall(r"^- \*\*Ruling (\d+)\*\* \((?:dev|2A|2B|2C)\): (.*)$", part_e, flags=re.M))
    assert entries["140"] == (
        "26 Oct 2026: the registration record may be an OSF registration, a public release of the code repository or the "
        "author's local record; `osf_id` is renamed `registration_id`; the protocol had not been registered or published "
        "before this amendment.")
    assert entries["141"] == (
        "26 Oct 2026: the author authorised the run of Test A on a local record, before any publication of the protocol; "
        "the manuscript states that no third party held the protocol before the pull.")
    numbers = [int(n) for n in re.findall(r"^- \*\*Ruling (\d+)\*\*", part_e, flags=re.M)]
    assert numbers[-2:] == [140, 141] and numbers == sorted(numbers) and len(numbers) == len(set(numbers))
    assert "2C: `dart_dev/.superpowers/sdd/2026-10-25-plan-2c-focus/progress.md`" in part_e
    # the older rulings that named OSF as the record now say what is true
    assert "(c) if the protocol is recorded as an OSF registration, that registration is public or embargoed at the author's " \
           "choice (public recommended; Ruling 140)." in entries["66"]
    assert "the registration id, which the pull's own log must not contradict" in entries["70"] and "OSF id" not in entries["70"]
    assert "and the registration id is logged and recorded in `docs/sealed/PULLS.json`" in entries["106"]
    assert "OSF id" not in entries["106"] and "go to OSF" not in entries["119"]


def test_the_three_sentences_of_the_spec_that_named_osf_are_rewritten_and_a_changed_spec_stops_the_protocol():
    from es.protocol import SPEC_RECORD_SENTENCES, render_protocol, rewrite_record_sentences
    spec = SPEC.read_text()
    assert len(SPEC_RECORD_SENTENCES) == 3 and all(spec.count(old) == 1 for old, _ in SPEC_RECORD_SENTENCES)
    part_a = _section_of(render_protocol(_full_ctx()), "## Part A.", "## Part B.")
    assert all(old not in part_a and new in part_a for old, new in SPEC_RECORD_SENTENCES)
    assert "right after the protocol is recorded (6.5 step 5)" in part_a
    assert "5. **The protocol is recorded** in `docs/sealed/REGISTRATION.md`, with its SHA-256 and the UTC time of the " \
           "record, as one of three records: an OSF registration, a public release of the code repository or the author's " \
           "local record." in part_a
    assert "Amended 26 Oct, Rulings 140 and 141: the record may also be a public release of the code repository or the " \
           "author's local record, and Test A is run on a local record.)*" in part_a
    assert "which are rewritten under Ruling 140" in part_a                    # Part A says that it does not copy them
    sample = " | ".join(old for old, _ in SPEC_RECORD_SENTENCES)
    assert rewrite_record_sentences(sample) == " | ".join(new for _, new in SPEC_RECORD_SENTENCES)
    for old, _ in SPEC_RECORD_SENTENCES:
        with pytest.raises(ValueError, match="exactly once, the sentence that Ruling 140 rewrites"):
            render_protocol(_full_ctx(spec=spec.replace(old, "")))             # a sentence that is gone stops the protocol
        with pytest.raises(ValueError, match="exactly once, the sentence that Ruling 140 rewrites"):
            render_protocol(_full_ctx(spec=spec.replace(old, old + " " + old)))            # and so does one that is there twice


def test_the_run_text_names_the_kind_of_the_registration_beside_its_identifier():
    from es.protocol import render_protocol
    text = render_protocol(_full_ctx())
    run = text[text.index("### The sealed pull and the sealed run"):text.index("### Failure handling")]
    assert "registration identifier, its kind and time, a fingerprint of the code" in run
    assert "the id and kind of the registration it ran under" in run
    assert "it names exactly one registration, which is an OSF registration, a public release of the code repository or the " \
           "author's local record" in run


def test_the_module_no_longer_calls_itself_the_osf_package_and_main_writes_the_template_it_renders(tmp_path):
    import es.protocol as protocol
    assert "OSF protocol package" not in protocol.__doc__ and "after registering on OSF" not in protocol.__doc__
    assert "Ruling 140" in protocol.__doc__ and "140 and 141" in protocol.__doc__
    result, root, sealed = _run_main(tmp_path)
    assert (sealed / "REGISTRATION.template.md").read_text() == protocol.registration_template()
    assert not (sealed / "REGISTRATION.md").exists()                                  # main() never writes the record
