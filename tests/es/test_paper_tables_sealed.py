"""Tests of es.paper_tables_sealed: the readers of the sealed outputs and the section 6 table shells.

The fixtures follow what es.sealed_run writes (its results.json keys and its tables/*.csv columns), not the plan's first
draft: H-E has t_obs, selc_obs and retained_share, and in both tests its `approximate` flag; H-B has ci and ci_basic and the
NaN-replicate count; secondary_bh.csv has `included`; the `reason` of coverage.csv is the feasibility reason; an analysis that
failed (Ruling 104) is an {"error": ...} slot, a line of results["errors"] and, for a table, a missing file or an `error` cell. The registered family (outputs/es/analogs/protocol_inputs.json) makes H-E the one
confirmatory test and H-B1 and H-B2 estimands with approximate intervals (REGISTERED). ALL_TESTED is the first draft's
family, where every hypothesis is a test; it keeps the cells that the brief pinned pinned. SWAPPED is a registration that
went the other way, to show that nothing here knows the family."""
import json
import re
from pathlib import Path

import pandas as pd
import pytest

HYPOTHESES = ("H-E", "H-B1", "H-B2")
#: family[hid] = (confirmatory, approximate)
ALL_TESTED = {"H-E": (True, False), "H-B1": (True, False), "H-B2": (True, False)}
REGISTERED = {"H-E": (True, False), "H-B1": (False, True), "H-B2": (False, True)}
SWAPPED = {"H-E": (False, False), "H-B1": (True, False), "H-B2": (True, True)}
REASONS = ["simulated size 0.130 > 1.5 x 0.0167 (39 of 300 replications, worst of 12 rows: LEV6, common_drift_0.5, upper)",
           "simulated power 0.28 at 0.05/3 is below 0.5"]
RESULT_TABLES = ("family", "pc_hs", "e1", "e2e4", "e5")          # the registered results: a shell before the run
DATA_TABLES = ("coverage", "ex_caiso", "audit", "econ", "bh")      # no shell: skipped until the run has happened
RAGGED = "\\raggedright\\hyphenpenalty=10000 "
BLOCK_ORDER = ("G6", "LEV6", "MID3", "DART3", "MIDDART6", "ROLL15", "ROLL_DISP9", "ROLL_LOC6", "DART18", "G1_24")


# ---------------------------------------------------------------------------------------------- fixtures
def _protocol_inputs(family):
    return {"eta": 0.05, "stage": "complete",
            "family": {h: {"confirmatory": c, "reasons": [] if c else REASONS, "approximate": a,
                           **({"interim_approximate": False} if h == "H-E" else {})} for h, (c, a) in family.items()}}


def _hb(test, hid, family, p, t_hat, rejected=False, replicated=False):
    """An H-B entry of results.json, with the wording that es.sealed_run._decide gives it."""
    conf, approx = family[hid]
    m = sum(c for c, _ in family.values())
    if test == "A":
        status = ("estimand (registration rule): " + "; ".join(REASONS) if not conf
                  else f"rejected in Test A at 0.05/{m}; replication pending" if p <= 0.05 / m
                  else f"not rejected at 0.05/{m} in Test A; full Holm after Test B")
    else:
        status = ("estimand (registration rule)" if not conf else "less portable in both 2026 quarters" if rejected and replicated
                  else "less portable in Q3 2026; not replicated in Q4" if rejected else "no portability difference detected")
    d = {"t_hat": t_hat, "p": p, "ci": [round(t_hat - 0.09, 3), round(t_hat + 0.08, 3)],
         "ci_basic": [round(t_hat - 0.08, 3), round(t_hat + 0.09, 3)], "n_nodes": 31, "n_nan_replicates": 0, "confirmatory": conf,
         "ci_approximate": approx, "status": status}
    if test == "B":
        d["replicated"] = replicated
    return d


def _he(test, family):
    conf = family["H-E"][0]
    if test == "A":
        return {"status": "interim (single window; decided after Test B)", "t_obs": 0.08, "p": 0.11, "n_null": 91, "selc_obs": 1.0,
                "retained_share": [{"min": 0.95, "mean": 0.98}], "approximate": False, "confirmatory": conf}
    return {"status": "decided (pooled A+B)", "t_obs": 0.09, "p": 0.01, "n_null": 8189, "selc_obs": 1.5,
            "retained_share": [{"min": 0.95, "mean": 0.98}, {"min": 0.93, "mean": 0.97}], "approximate": False, "confirmatory": conf,
            "wording": "timing skill within calendar cells detected" if conf else "estimand (registration rule): pooled SELC with CI"}


def _results(test="A", family=ALL_TESTED, ran=True, usable=31):
    r = {"test": test, "started_utc": "2026-10-28T14:00:00+00:00", "protocol_sha256": "ab" * 32, "registration": "10.17605/OSF.IO/ABCDE",
         "registered_utc": "2026-10-24T18:30:00Z", "n_rep": 2000, "bound_files": {}, "departures": [], "sealed_manifests": {},
         "eta": 0.05, "family": [h for h in HYPOTHESES if family[h][0]], "rows_read": 1000, "rows_kept": 990,
         "coverage": {"ran": ran, "usable_nodes": usable, "nonfinite_dropped": {}}, "errors": {}, "status": "completed",
         "finished_utc": "2026-10-28T15:00:00+00:00"}
    if not ran:
        r["status"] = "not run (coverage rule)"
        return r
    ps = {"A": {"H-B1": (0.004, 0.12), "H-B2": (0.2, 0.05)}, "B": {"H-B1": (0.004, 0.12), "H-B2": (0.4, 0.01)}}[test]
    r.update({"tau_check": {"max_abs_diff": 0.0, "nan_pattern_mismatches": 0, "n_compared": 264, "registered_only": 0, "recomputed_only": 0},
              "PC": {"P2_season": {"median_diff": 0.2, "ci": [0.1, 0.3], "n_nodes": 31, "n_nan_replicates": 0, "ci_approximate": True},
                     "P2_all": {"median_diff": 0.5, "ci": [0.3, 0.7], "n_nodes": 31, "n_nan_replicates": 0, "ci_approximate": True}},
              "H-S": {"xgb": {"failures": 2, "ci": [0, 5], "ci_independent": [0, 30], "scheme": "aligned", "n_nodes": 31, "n_nan_replicates": 0,
                              "n_nan_replicates_independent": 0, "approximate": False},
                      "logit": {"failures": 1, "ci": [0, 4], "ci_independent": [0, 20], "scheme": "aligned", "n_nodes": 31, "n_nan_replicates": 0,
                                "n_nan_replicates_independent": 0, "approximate": False}},
              "E5": [], "estimands": {}, "estimands_scope": "single window (interim)" if test == "A" else "pooled A+B",
              "secondary": {f: {"tests": 2, "rows": 2, "bh_rejections": 1} for f in ("audit robustness", "economic robustness", "drift monitors")},
              "H-E": _he(test, family)})
    for hid, (p, t_hat) in ps.items():
        r[hid] = _hb(test, hid, family, p, t_hat, rejected=(hid == "H-B1"), replicated=(hid == "H-B1"))
    if test == "B":
        r["holm"] = {h: ((h != "H-B2") if family[h][0] else None) for h in HYPOTHESES}
    return r


def _stats():
    from es.paper_registry import STAT_ORDER
    e1 = [f"E1:{k}:{p}" for k in ("NET", "ESred", "J_0.0", "J_0.25", "J_0.5", "J_1.0") for p in ("timing", "level", "total")]
    return list(STAT_ORDER) + e1


def _estimands(estimate, lo, hi):
    return pd.DataFrame([{"stat": s, "estimate": estimate, "lo": lo, "hi": hi, "approximate": "ESred" in s, "n_windows": 1, "n_nan_replicates": 0}
                         for s in _stats()])


def _write_tables(d: Path, test: str, not_run: bool):
    """tables/*.csv as es.sealed_run writes them (a test that did not run writes coverage.csv only)."""
    cov = pd.DataFrame({"market": ["NYISO"] * 2 + ["ISONE", "MISO", "CAISO"], "node": ["a", "b", "c", "d", "e"], "expected_hours": 2208,
                        "hours": [2208, 1000, 2208, 2208, 2208], "share": [1, 0.45, 1, 1, 1], "ok": [True, False, True, True, True],
                        "n_pos": 50, "feasible": True, "reason": "ok", "usable": [True, False, True, True, True]})
    if not_run:                                  # every CAISO node is out: the coverage rule fails on the operator
        cov.loc[cov.market == "CAISO", ["ok", "usable", "hours", "share"]] = [False, False, 0, 0.0]
    cov.to_csv(d / "coverage.csv", index=False)
    if not_run:
        return
    _estimands(1.0, 0.5, 1.5).to_csv(d / "estimands.csv", index=False)
    _estimands(3.0, 2.0, 4.0).to_csv(d / "estimands_ex_caiso.csv", index=False)
    pd.DataFrame([{"family": f, "test": f"{f} {i}", "estimate": 0.1, "p": p, "bh_reject": p < 0.01, "included": True}
                  for f in ("audit robustness", "economic robustness", "drift monitors") for i, p in enumerate((0.005, 0.3), 1)]
                 ).to_csv(d / "secondary_bh.csv", index=False)
    if test == "B":
        _estimands(2.0, 1.0, 3.0).to_csv(d / "estimands_this_quarter.csv", index=False)
    pd.DataFrame({"block": list(BLOCK_ORDER), "median_log_error": 0.1, "lo": 0.05, "hi": 0.2, "failures": 3, "failures_lo": 1, "failures_hi": 6,
                  "n_nodes": 31, "n_nan_replicates": 0, "n_nan_replicates_failures": 0,
                  # Ruling 118, M2: the flat diagnostic columns of the independent-draw scheme, and the scheme that the primary interval is of.
                  # The shared fixture keeps approximate False, so that the unmarked cells stay tested; the real run writes True for every row
                  # (tests/es/test_paper_g1.py builds that state).
                  "lo_independent": 0.07, "hi_independent": 1.1, "failures_lo_independent": 0, "failures_hi_independent": 30,
                  "scheme": "aligned", "interval_registered": "percentile, calendar-aligned (primary); independent draws (diagnostic)",
                  "approximate": False}).to_csv(d / "E5_blocks.csv", index=False)
    pd.DataFrame([{"market": "NYISO", "node": n, "n": 2208, "alerts": 110, "rate": 0.05, "R": 1.0, "zero_activity": False, "R_floored": 1.0,
                   "log_error": 0.0, "fail_0.5_2": False, "maintenance": m, "block": b, "estimator": e, "seed": s, "policy": p}
                  for n in ("a", "c") for m, b, e, s, p in (("roll", "G1_24", "xgb", 42, "P1"), ("roll", "G1_24", "xgb", 42, "P4_56"),
                                                           ("frozen", "LEV6", "xgb", 42, "P1"), ("roll", "DART3", "logit", 42, "P1"),
                                                           ("roll", "DART3", "xgb", 0, "P1"))]).to_csv(d / "audit_activity.csv", index=False)
    pd.DataFrame([{"analysis": "SA", "policy": pol, "comparator": "none", "position": "premium", "fees": "calibrated", "SELC": 0.1,
                   "NET": 100.0, "SNET": 0.5, "ESred": 20.0, "CAP": 0.2, "n_days": 92}
                  for pol in ("P1", "P4", "P5", "P6", "P4@r*")]).to_csv(d / "economics.csv", index=False)


def _sources(tmp_path: Path, tests=("A",), family=ALL_TESTED, not_run=(), registered=None):
    """Sealed outputs of the given tests under tmp_path/out; analogs/protocol_inputs.json holds the registered family when
    `registered` is a family (as the real tree has it)."""
    from es.paper_macros import Sources
    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    for t in tests:
        d = out / "sealed" / t / "tables"
        d.mkdir(parents=True)
        (out / "sealed" / t / "results.json").write_text(json.dumps(_results(t, family, ran=t not in not_run, usable=20 if t in not_run else 31)))
        _write_tables(d, t, t in not_run)
    if registered is not None:
        (out / "analogs").mkdir(exist_ok=True)
        (out / "analogs" / "protocol_inputs.json").write_text(json.dumps(_protocol_inputs(registered)))
    return Sources(out, tmp_path / "paper", tmp_path / "docs")


def _fresh(src):
    from es.paper_macros import Sources
    return Sources(src.out_dir, src.paper_dir, src.docs_sealed)


def _build(name, src):
    import es.paper_tables_sealed  # noqa: F401
    from es.paper_tables import BUILDERS
    return BUILDERS[name][0](src)


def _row(tex: str, start: str) -> str:
    """The one line of a table that begins with `start`, without the ragged-right prefix of a paragraph cell."""
    found = [line.replace(RAGGED, "") for line in tex.splitlines() if line.replace(RAGGED, "").startswith(start)]
    assert len(found) == 1, (start, found)
    return found[0]


def _body_rows(tex: str) -> list[str]:
    """The lines of a table that end a row (the heading row first), without the ragged-right prefixes."""
    return [line.replace(RAGGED, "") for line in tex.splitlines() if line.endswith("\\\\")]


def _tables_dir(tmp_path):
    return tmp_path / "paper" / "generated" / "tables"


def _clean(tex: str):
    assert "\\PENDING" not in tex and not re.search(r"\bnan\b|\bNone\b|\binf\b", tex), tex


def _edit_json(path: Path, edit):
    r = json.loads(path.read_text())
    edit(r)
    path.write_text(json.dumps(r))


# ---------------------------------------------------------------------------------------------- the brief's tests
def test_sealed_builders_exist_and_render_test_a(tmp_path):
    import es.paper_tables_sealed  # noqa: F401
    from es.paper_tables import BUILDERS, build_all
    src = _sources(tmp_path)
    status = build_all(src, _tables_dir(tmp_path))
    for name in RESULT_TABLES + DATA_TABLES:
        assert status[f"tab_sealed_{name}_A"] == "built", status
    for name in RESULT_TABLES:                            # Test B has not run: the registered results are shells, the rest is skipped
        assert status[f"tab_sealed_{name}_B"] == "built", status
    for name in (*(f"tab_sealed_{n}_B" for n in DATA_TABLES), "tab_sealed_decision_B", "tab_sealed_this_quarter_B"):
        assert status[name] == "skipped: missing sealed/B/results.json", (name, status[name])
    fam = BUILDERS["tab_sealed_family_A"][0](src)
    assert "interim" in fam and "0.004" in fam and "replication pending" in fam
    e1 = BUILDERS["tab_sealed_e1_A"][0](src)
    assert e1.count("\\\\") == 1 + 6 and r"$J_{0.25}$ & \$1 [\$1, \$2] & \$1 [\$1, \$2] & \$1 [\$1, \$2] \\" in e1     # 0.5 rounds half-up
    cov = BUILDERS["tab_sealed_coverage_A"][0](src)
    assert r"NYISO & 2 & 1 & b (45\%, coverage)" in cov                  # the excluded node, its coverage and reason
    assert "ISO-NE & 1 & 1 & --" in cov


def test_sealed_bh_and_ex_caiso_tables(tmp_path):
    src = _sources(tmp_path)
    bh = _build("tab_sealed_bh_A", src)
    assert "audit robustness & 2 & -- & --" in bh and "audit robustness & 2 & 1" not in bh and bh.count("\\\\") == 4     # Ruling 118, M5
    assert "economic robustness & 2 & 1 & 0.005" in bh and "no count of its rejections" in bh
    ex = _build("tab_sealed_ex_caiso_A", src)
    assert r"SELC & 300.0\% [200.0\%, 400.0\%] \\" in ex and r"E3: ESred & \$3 [\$2, \$4]\textsuperscript{a} \\" in ex
    full = _build("tab_sealed_e2e4_A", src)
    assert r"SELC & 100.0\% [50.0\%, 150.0\%] \\" in full


def test_test_b_decision_table(tmp_path):
    src = _sources(tmp_path, tests=("A", "B"))
    dec = _build("tab_sealed_decision_B", src)
    assert r"H-B1 & A: 0.004; B: 0.004 & yes & yes & less portable in both 2026 quarters \\" in dec
    assert r"H-B2 & A: 0.200; B: 0.400 & no & no & no portability difference detected \\" in dec
    assert r"H-E & 0.010 & yes & -- & timing skill within calendar cells detected \\" in dec
    assert r"SELC & 200.0\% [100.0\%, 300.0\%] \\" in _build("tab_sealed_this_quarter_B", src)


# ---------------------------------------------------------------------------------------------- an estimand prints no decision
def test_the_registered_family_prints_the_estimands_with_interval_and_mark_and_no_decision(tmp_path):
    src = _sources(tmp_path, family=REGISTERED, registered=REGISTERED)
    fam = _build("tab_sealed_family_A", src)
    he, b1, b2 = _row(fam, "H-E"), _row(fam, "H-B1"), _row(fam, "H-B2")
    assert "& confirmatory & 100.0\\% & [50.0\\%, 150.0\\%] & 0.110 & interim (single window; decided after Test B) \\\\" in he
    assert b1.endswith(r"& estimand & 0.120 & [0.030, 0.200]\textsuperscript{a} & -- & estimand (designation rule) \\")
    assert b2.endswith(r"& estimand & 0.050 & [\textminus{}0.040, 0.130]\textsuperscript{a} & -- & estimand (designation rule) \\")
    for word in ("reject", "replication", "Holm"):                                      # no decision
        assert word not in b1 and word not in b2, word
    assert "0.004" not in b1 and "0.200" not in b2                                      # and not the p-value of an estimand
    assert "simulated size" not in fam                                                  # the reasons belong to the registration table
    assert ("The intervals of H-B1 and H-B2 are percentile bootstrap intervals from independent draws of the calibration and the deployment "
            "days. The basic intervals, H-B1 [0.040, 0.210], "
            r"H-B2 [\textminus{}0.030, 0.140], reflect the shift of the bootstrap distribution about $\hat t$, which comes mainly from "
            "re-randomizing the calendar composition that the two windows share by design. "
            "Both intervals are approximate and may under-cover substantially under a common drift.") in fam      # Ruling 110: the basic interval is a diagnostic; Ruling 118, M7
    assert "exclud" not in fam and "tested H-B" not in fam                              # no "excludes zero" wording; no tested H-B hypothesis here
    assert "\\footnotetext{\\textsuperscript{a} Interval labelled approximate" in fam
    _clean(fam)


def test_the_family_is_read_from_the_results_and_never_hard_coded(tmp_path):
    src = _sources(tmp_path, family=SWAPPED)
    fam = _build("tab_sealed_family_A", src)
    he, b1, b2 = _row(fam, "H-E"), _row(fam, "H-B1"), _row(fam, "H-B2")
    assert "& estimand & 100.0\\% & [50.0\\%, 150.0\\%] & -- &" in he and "0.110" not in he     # H-E demoted: its SELC with an interval, no p-value
    assert "& confirmatory & 0.120 & [0.030, 0.200] & 0.004 & rejected in Test A at 0.05/2; replication pending \\\\" in b1
    assert "& confirmatory & 0.050 & [\\textminus{}0.040, 0.130]\\textsuperscript{a} & 0.200 &" in b2   # the approximate interval is marked, the p-value is a test's
    assert "The p-value of a tested H-B hypothesis is the basic bootstrap p-value." in fam
    src_b = _sources(tmp_path / "b", tests=("A", "B"), family=SWAPPED)
    dec = _build("tab_sealed_decision_B", src_b)
    assert r"H-B1 & A: 0.004; B: 0.004 & yes & yes & less portable in both 2026 quarters \\" in dec     # the Holm panel holds the tests ...
    assert r"H-B2 & A: 0.200; B: 0.400\textsuperscript{a} & no & no & no portability difference detected \\" in dec
    assert _row(dec, "H-E").endswith(r"& -- & estimand (designation rule): pooled SELC with CI \\")     # ... and H-E, an estimand, its own panel


def test_the_decision_table_gives_the_estimands_their_own_panel(tmp_path):
    src = _sources(tmp_path, tests=("A", "B"), family=REGISTERED, registered=REGISTERED)
    dec = _build("tab_sealed_decision_B", src)
    assert r"H-E & 0.010 & yes & -- & timing skill within calendar cells detected \\" in dec
    assert _row(dec, "H-B1") == (r"H-B1 & 0.120 [0.030, 0.200]\textsuperscript{a} & 0.120 [0.030, 0.200]\textsuperscript{a} & yes & "
                                 r"estimand (designation rule) \\")
    assert _row(dec, "H-B2") == (r"H-B2 & 0.050 [\textminus{}0.040, 0.130]\textsuperscript{a} & 0.010 [\textminus{}0.080, 0.090]\textsuperscript{a} & yes & "
                                 r"estimand (designation rule) \\")
    assert ("The intervals of H-B1 and H-B2 are percentile bootstrap intervals from independent draws of the calibration and the deployment "
            "days. The basic intervals, H-B1 [0.040, 0.210] in Test A and "
            r"[0.040, 0.210] in Test B, H-B2 [\textminus{}0.030, 0.140] in Test A and [\textminus{}0.070, 0.100] in Test B, reflect the shift of the "
            "bootstrap distribution about $\\hat t$"
            ) in dec and "Both intervals are approximate and may under-cover substantially under a common drift." in dec and "exclud" not in dec
    for word in ("less portable", "no portability", "Holm rejects"):                  # an estimand has no Holm decision and no replication verdict
        assert word not in _row(dec, "H-B1") + _row(dec, "H-B2")
    assert "Estimands" in dec and "Holm" in dec
    _clean(dec)


def test_the_pooled_h_e_flag_is_the_runs_own(tmp_path):
    """Test B's H-E carries the pooled test's `approximate` flag (Ruling 104(b)), Test A's the interim one; the p-value is marked with it."""
    src = _sources(tmp_path, tests=("A", "B"))
    assert "textsuperscript" not in _build("tab_sealed_decision_B", src)
    path = src.out_dir / "sealed" / "B" / "results.json"
    _edit_json(path, lambda r: r["H-E"].__setitem__("approximate", True))
    assert r"H-E & 0.010\textsuperscript{a} & yes" in _build("tab_sealed_decision_B", _fresh(src))
    assert "& 0.010\\textsuperscript{a} &" in _row(_build("tab_sealed_family_B", _fresh(src)), "H-E")
    _edit_json(path, lambda r: r["H-E"].pop("approximate"))
    with pytest.raises(LookupError, match="approximate"):
        _build("tab_sealed_decision_B", _fresh(src))


def test_an_inconsistent_family_stops_the_build(tmp_path):
    src = _sources(tmp_path)
    _edit_json(src.out_dir / "sealed" / "A" / "results.json", lambda r: r.__setitem__("family", ["H-E", "H-B2"]))   # H-B1 says confirmatory
    with pytest.raises(LookupError, match="H-B1"):
        _build("tab_sealed_family_A", _fresh(src))


# ---------------------------------------------------------------------------------------------- the family table reads what es.sealed_run writes
def test_the_family_table_reports_the_rotation_test_and_the_scope(tmp_path):
    src = _sources(tmp_path, tests=("A", "B"), family=REGISTERED, registered=REGISTERED)
    a = _build("tab_sealed_family_A", src)
    assert r"studentized statistic $T$ = 0.08" in a and "91 non-identity rotations" in a and r"95\% / 98\%" in a
    assert "Test B alone" not in a
    b = _build("tab_sealed_family_B", src)
    assert "decided (pooled A+B)" in _row(b, "H-E") and "& 0.010 &" in _row(b, "H-E")
    assert "8,189 non-identity rotations" in b.replace("{,}", ",")
    assert r"Test A: 95\% / 98\%" in b and r"Test B: 93\% / 97\%" in b
    assert "the estimates of H-B1 and H-B2 are those of Test B alone" in b


def test_the_h_b_interval_is_the_percentile_one_and_the_basic_one_is_a_diagnostic_in_the_note(tmp_path):
    """Ruling 110: the primary interval of the demoted estimands is the percentile `ci`; `ci_basic` is shown in the note."""
    src = _sources(tmp_path, family=REGISTERED, registered=REGISTERED)
    fam = _build("tab_sealed_family_A", src)
    b1 = _row(fam, "H-B1")
    assert "[0.030, 0.200]" in b1 and "[0.040, 0.210]" not in b1                          # ci in the table
    assert "The basic intervals, H-B1 [0.040, 0.210]" in fam                              # ci_basic in the note
    path = src.out_dir / "sealed" / "A" / "results.json"
    _edit_json(path, lambda r: r["H-B1"].pop("ci"))
    with pytest.raises(LookupError, match="missing 'ci'"):
        _build("tab_sealed_family_A", _fresh(src))
    _edit_json(path, lambda r: r["H-B1"].update(ci=[0.03, 0.2]))
    _edit_json(path, lambda r: r["H-B1"].pop("ci_basic"))
    with pytest.raises(LookupError, match="missing 'ci_basic'"):                          # the note needs it
        _build("tab_sealed_family_A", _fresh(src))


def test_an_e5_interval_that_misses_its_estimate_gets_a_note_computed_from_the_data(tmp_path):
    src = _sources(tmp_path)
    assert "does not contain its estimate" not in _build("tab_sealed_e5_A", src)             # the fixture's estimates are all inside
    path = src.out_dir / "sealed" / "A" / "tables" / "E5_blocks.csv"
    e5 = pd.read_csv(path)
    e5.loc[e5.block == "G6", "median_log_error"] = 0.30                                      # above its upper end 0.2
    e5.loc[e5.block == "MID3", "failures"] = 9                                               # above its upper end 6
    e5.loc[e5.block == "DART3", "median_log_error"] = 0.05                                   # equal to its lower end: contained
    e5.loc[e5.block == "LEV6", ["median_log_error", "lo"]] = [float("nan"), 0.5]             # not finite: not computed, not counted
    e5.to_csv(path, index=False)
    table = _build("tab_sealed_e5_A", _fresh(src))
    assert ("\\footnotetext{For calendar and midpoint, an interval does not contain its estimate: the bootstrap distribution is shifted relative "
            "to the estimate, and the interval is approximate (see Section~\\ref{esm:size-power}). The intervals of the independent-draw scheme are "
            "saved beside these as a diagnostic.}") in table
    assert table.count("does not contain its estimate") == 1
    e5.loc[e5.block == "ROLL15", "hi"] = 0.01                                               # a third block
    e5.to_csv(path, index=False)
    assert "For calendar, midpoint and all rolling, an interval does not contain" in _build("tab_sealed_e5_A", _fresh(src))
    e5.loc[e5.block.isin(["MID3", "ROLL15"]), ["median_log_error", "hi", "failures"]] = [0.1, 0.2, 3]
    e5.to_csv(path, index=False)
    assert "For calendar, an interval does not contain" in _build("tab_sealed_e5_A", _fresh(src))                      # a single block


# ---------------------------------------------------------------------------------------------- shells before the run
def test_shells_before_the_run_show_the_registered_cells(tmp_path):
    import es.paper_tables_sealed  # noqa: F401
    from es.blocks import BLOCKS, PAPER_BLOCK_NAMES
    from es.paper_registry import STAT_LABEL
    from es.paper_tables import BUILDERS, build_all
    src = _sources(tmp_path, tests=(), registered=REGISTERED)
    status = build_all(src, _tables_dir(tmp_path))
    for t in "AB":
        for name in RESULT_TABLES:
            assert status[f"tab_sealed_{name}_{t}"] == "built", status
            _clean(BUILDERS[f"tab_sealed_{name}_{t}"][0](src))
        for name in DATA_TABLES:
            assert status[f"tab_sealed_{name}_{t}"] == f"skipped: missing sealed/{t}/results.json"
    fam = _build("tab_sealed_family_A", src)
    assert "& confirmatory & -- & -- & -- & -- \\\\" in _row(fam, "H-E")
    assert "& estimand & -- & --\\textsuperscript{a} & -- & -- \\\\" in _row(fam, "H-B1")
    e1 = _build("tab_sealed_e1_A", src)                          # E1: timing, level and total of NET, ESred and the four J
    assert e1.count("\\\\") == 7 and r"NET & -- & -- & -- \\" in e1 and r"$J_{0}$ & -- & -- & -- \\" in e1 and r"$J_{1}$ & -- & -- & -- \\" in e1
    e2e4 = _build("tab_sealed_e2e4_B", src)                      # E2, E3, E4 and the headline SELC and NET: the registry's stats
    for label in STAT_LABEL.values():
        assert f"{label} & -- \\\\" in e2e4, label
    e5 = _build("tab_sealed_e5_A", src)                          # E5: the ten blocks
    for b in BLOCKS:
        assert f"{PAPER_BLOCK_NAMES[b]} & {len(BLOCKS[b])} & -- & -- & -- \\\\" in e5
    assert _build("tab_sealed_pc_hs_A", src).count("-- & -- & --") == 4     # PC twice, H-S twice


def test_the_family_shell_waits_for_the_designated_status_not_for_a_default(tmp_path):
    """No sealed output and no family of designated statuses: the shell still builds (the empty tree must not fail), with the development
    sentinel where the designated status belongs, which the strict lint refuses."""
    src = _sources(tmp_path, tests=())
    fam = _build("tab_sealed_family_A", src)
    assert fam.count("\\PENDINGDEV") == 1 and "\\footnotetext{Designated status of each hypothesis: \\PENDINGDEV.}" in fam
    assert "Registered" not in fam and "registered" not in fam
    assert "H-E" in fam and "H-B2" in fam and "& confirmatory" not in fam and "& estimand" not in fam      # no status is guessed
    (src.out_dir / "analogs").mkdir()
    (src.out_dir / "analogs" / "protocol_inputs.json").write_text(json.dumps({"eta": 0.05, "stage": "analogs", "family": None}))
    assert _build("tab_sealed_family_B", _fresh(src)).count("\\PENDINGDEV") == 1          # the early stage has no family yet


def test_the_module_run_builds_the_shells_on_an_empty_tree_without_failing(tmp_path):
    from es.paper_tables import main
    out, paper = tmp_path / "out", tmp_path / "paper"
    out.mkdir()
    paper.mkdir()
    assert main(["--out-dir", str(out), "--paper-dir", str(paper)]) == 0
    index = (paper / "generated" / "tables" / "INDEX.md").read_text()
    assert "| tab_sealed_family_A | built |" in index and "| tab_sealed_e5_B | built |" in index
    assert "| tab_sealed_audit_A | skipped: missing sealed/A/results.json |" in index


# ---------------------------------------------------------------------------------------------- a test that did not run
def test_a_test_that_did_not_run_prints_the_statement_and_no_sentinel(tmp_path):
    import es.paper_tables_sealed  # noqa: F401
    from es.paper_lint import PENDING_TOKEN
    from es.paper_tables import BUILDERS, build_all
    src = _sources(tmp_path, not_run=("A",), registered=REGISTERED)
    status = build_all(src, _tables_dir(tmp_path))
    names = [f"tab_sealed_{n}_A" for n in RESULT_TABLES + DATA_TABLES]
    assert all(status[n] == "built" for n in names), status                      # every table that a section may include exists
    for n in names:
        tex = BUILDERS[n][0](src)
        assert "Test A did not run (coverage rule)" in tex, n
        assert "20 usable nodes, 28 required" in tex and "no usable node for CAISO" in tex, n
        assert not PENDING_TOKEN.search(tex), n
        _clean(tex)
        if n != "tab_sealed_coverage_A":
            assert tex.count("\\\\") == 1 and "\\midrule" not in tex, n          # a statement, not a table of dashes
    cov = _build("tab_sealed_coverage_A", src)                                   # the coverage table itself still shows who was out
    assert r"CAISO & 1 & 0 & e (0\%, coverage)" in cov and r"NYISO & 2 & 1 & b (45\%, coverage)" in cov
    for n in ("tab_sealed_e1_B", "tab_sealed_family_B"):                         # the other test is unaffected: a shell
        assert "did not run" not in BUILDERS[n][0](src)


def test_every_operator_without_a_usable_node_is_named(tmp_path):
    src = _sources(tmp_path, not_run=("A",))
    path = src.out_dir / "sealed" / "A" / "tables" / "coverage.csv"
    cov = pd.read_csv(path)
    cov.loc[cov.market == "MISO", ["ok", "usable"]] = False
    cov.to_csv(path, index=False)
    assert "no usable node for MISO, CAISO" in _build("tab_sealed_family_A", _fresh(src))


# ---------------------------------------------------------------------------------------------- the data tables
def test_coverage_names_the_cause_of_each_exclusion(tmp_path):
    src = _sources(tmp_path)
    path = src.out_dir / "sealed" / "A" / "tables" / "coverage.csv"
    cov = pd.read_csv(path)
    extra = pd.DataFrame([
        {"market": "CAISO", "node": "f_x", "expected_hours": 2208, "hours": 2208, "share": 1.0, "ok": True, "n_pos": 3, "feasible": False,
         "reason": "lt_20_positives", "usable": False},
        {"market": "CAISO", "node": "g", "expected_hours": 2208, "hours": 700, "share": 0.317, "ok": False, "n_pos": 0, "feasible": False,
         "reason": "no_train_rows", "usable": False}])
    pd.concat([cov, extra], ignore_index=True).to_csv(path, index=False)
    row = _row(_build("tab_sealed_coverage_A", _fresh(src)), "CAISO")
    assert row.startswith("CAISO & 3 & 1 & ")
    assert r"f\_x (100\%, lt\_20\_positives)" in row and r"g (32\%, coverage, no\_train\_rows)" in row


def test_the_bh_table_counts_the_tests_that_ran(tmp_path):
    src = _sources(tmp_path)
    path = src.out_dir / "sealed" / "A" / "tables" / "secondary_bh.csv"
    b = pd.read_csv(path)
    extra = pd.DataFrame([{"family": "audit robustness", "test": "H-B2 frozen", "estimate": float("nan"), "p": float("nan"),
                           "bh_reject": False, "included": False}])
    pd.concat([b, extra], ignore_index=True).to_csv(path, index=False)
    bh = _build("tab_sealed_bh_A", _fresh(src))
    assert "audit robustness & 2 & -- & --" in bh and "BH rejections ($q$ = 0.05)" in bh        # the NaN test is no test that ran; Ruling 118: no rejections
    b[b.family != "drift monitors"].to_csv(path, index=False)                    # the file lost a family that results.json counts
    with pytest.raises(LookupError, match="drift monitors.*0 tests.*records 2"):
        _build("tab_sealed_bh_A", _fresh(src))
    _edit_json(src.out_dir / "sealed" / "A" / "results.json", lambda r: r["secondary"]["drift monitors"].update(tests=0, rows=0, bh_rejections=0))
    assert r"drift monitors & 0 & 0 & -- \\" in _build("tab_sealed_bh_A", _fresh(src))       # a family with no row to adjust has no test


def test_the_audit_table_follows_the_runs_order_and_aggregates_the_other_seeds(tmp_path):
    audit = _build("tab_sealed_audit_A", _sources(tmp_path))
    rows = _body_rows(audit)[1:]                                                 # without the heading
    assert len(rows) == 5
    assert rows[0].startswith(r"rolling & combined & XGB & P1 fixed, held-out & 0 & 2 & 1.00 & 0.00 \\")
    assert rows[1].startswith(r"rolling & combined & XGB & P4 rolling (56 d) & 0 & 2 &")
    assert rows[2].startswith(r"frozen & absolute level & XGB & P1 fixed, held-out & 0 & 2 &")
    assert rows[3].startswith(r"rolling & plain DART lags & logit & P1 fixed, held-out & 0 & 2 &")
    assert rows[4] == r"rolling & plain DART lags & XGB & P1, other seeds (min--max) & 0 & -- & -- & -- \\"
    _clean(audit)


def test_the_economics_table_lists_the_policies_with_their_codes(tmp_path):
    econ = _build("tab_sealed_econ_A", _sources(tmp_path))
    rows = _body_rows(econ)[1:]
    assert [r.split(" & ")[0] for r in rows] == ["P1", "P4", "P5", "P6", "P4@r*"]
    assert rows[0] == r"P1 & \$100 & 10.0\% & \$0.50 & \$20 & 20.0\% & 92 \\"
    assert "Cutoff policies: P1: fixed, held-out" in econ
    _clean(econ)


def test_the_pc_and_hs_rows(tmp_path):
    pc = _build("tab_sealed_pc_hs_A", _sources(tmp_path))
    assert r"0.200 & [0.100, 0.300]\textsuperscript{a} & 31" in pc and r"0.500 & [0.300, 0.700]\textsuperscript{a} & 31" in pc
    assert "& 2 & [0.0, 5.0] & 31" in pc and "& 1 & [0.0, 4.0] & 31" in pc
    assert "\\footnotetext{\\textsuperscript{a} Interval labelled approximate" in pc


def test_the_e5_rows_follow_the_blocks(tmp_path):
    src = _sources(tmp_path)
    e5 = _build("tab_sealed_e5_A", src)
    assert r"absolute level & 6 & 0.10 [0.05, 0.20] & 3 [1.0, 6.0] & 31 \\" in e5
    assert r"calendar & 0 & " in e5 and len(_body_rows(e5)) == 11
    path = src.out_dir / "sealed" / "A" / "tables" / "E5_blocks.csv"
    pd.read_csv(path).iloc[:-1].to_csv(path, index=False)                        # a block of the table is missing from the file
    with pytest.raises(LookupError, match="G1_24"):
        _build("tab_sealed_e5_A", _fresh(src))


def test_the_scope_and_the_e4_estimands_are_stated_under_the_estimand_tables(tmp_path):
    src = _sources(tmp_path, tests=("A", "B"))
    a, b = _build("tab_sealed_e2e4_A", src), _build("tab_sealed_e2e4_B", src)
    assert "single window (interim)" in a and "pooled A+B" in b and "Test B alone" in _build("tab_sealed_this_quarter_B", src)
    assert "paired differences" in a and "context" in a
    assert "single window (interim)" in _build("tab_sealed_e1_A", src) and "single window (interim)" in _build("tab_sealed_ex_caiso_A", src)


def test_the_e1_table_changes_to_a_long_format_when_the_wide_one_would_not_fit(tmp_path):
    src = _sources(tmp_path)
    path = src.out_dir / "sealed" / "A" / "tables" / "estimands.csv"
    e = pd.read_csv(path)
    e.loc[e.stat.str.startswith("E1:"), ["estimate", "lo", "hi"]] = [1234567.0, -2345678.0, 3456789.0]
    e.to_csv(path, index=False)
    long_ = _build("tab_sealed_e1_A", _fresh(src))
    assert "component" in long_ and long_.count("\\\\") == 1 + 18 and long_.count("\\midrule") == 1 + 5
    assert r"NET & timing & \$1{,}234{,}567 [\textminus{}\$2{,}345{,}678, \$3{,}456{,}789] \\" in long_
    assert r"$J_{0.25}$ & total & \$1{,}234{,}567 [\textminus{}\$2{,}345{,}678, \$3{,}456{,}789]\textsuperscript{a}" not in long_    # fixture: J rows unmarked
    assert "component" not in _build("tab_sealed_e1_A", _sources(tmp_path / "w"))


def test_the_e1_cells_carry_the_approximate_mark(tmp_path):
    src = _sources(tmp_path)
    path = src.out_dir / "sealed" / "A" / "tables" / "estimands.csv"
    e = pd.read_csv(path)
    e.loc[e.stat.str.startswith("E1:J_0.25"), "approximate"] = True
    e.to_csv(path, index=False)
    e1 = _build("tab_sealed_e1_A", _fresh(src))
    assert r"$J_{0.25}$ & \$1 [\$1, \$2]\textsuperscript{a} & \$1 [\$1, \$2]\textsuperscript{a} & \$1 [\$1, \$2]\textsuperscript{a} \\" in e1
    assert r"ESred & \$1 [\$1, \$2]\textsuperscript{a} &" in e1 and "\\footnotetext{\\textsuperscript{a} Interval labelled approximate" in e1


def test_an_unreadable_ran_test_fails_with_the_name_of_what_is_missing(tmp_path):
    src = _sources(tmp_path)
    tables = src.out_dir / "sealed" / "A" / "tables"
    (tables / "economics.csv").unlink()
    with pytest.raises(LookupError, match="economics.csv"):
        _build("tab_sealed_econ_A", _fresh(src))
    pd.read_csv(tables / "estimands.csv").drop(columns=["hi"]).to_csv(tables / "estimands.csv", index=False)
    with pytest.raises(LookupError, match="hi"):
        _build("tab_sealed_e2e4_A", _fresh(src))
    _edit_json(src.out_dir / "sealed" / "A" / "results.json", lambda r: r.pop("PC"))
    with pytest.raises(LookupError, match="PC"):
        _build("tab_sealed_pc_hs_A", _fresh(src))


# ---------------------------------------------------------------------------------------------- an analysis that failed (Ruling 104)
FAILURES = {"H-B1": "ValueError: the contrast has nothing to infer from", "PC/P2_season": "LookupError: score set ('roll', 'G1_24', 'xgb', 42) is not available",
            "H-S/logit": "ValueError: boom", "E5/LEV6": "ValueError: no finite replicate", "economics": "RuntimeError: the settlement frame is empty",
            "estimands": "LookupError: needs the economic rows (economics_for did not complete)",
            "estimands_ex_caiso": "LookupError: needs the economic rows (economics_for did not complete)",
            "drift monitors": "ValueError: no node has enough days", "audit_activity/P6": "ValueError: no node qualifies"}


def _fail_a_run(src, test="A"):
    """The output of a run in which several analyses failed, as es.sealed_run leaves it: an {"error": ...} slot, a line of
    results["errors"], and no file for what the analysis writes (an E5 block that failed has an `error` cell instead of numbers)."""
    d = src.out_dir / "sealed" / test

    def edit(r):
        r["H-B1"] = {"error": FAILURES["H-B1"]}
        r["PC"]["P2_season"] = {"error": FAILURES["PC/P2_season"]}
        r["H-S"]["logit"] = {"error": FAILURES["H-S/logit"]}
        r["economics"] = {"error": FAILURES["economics"]}
        r["estimands"] = {"error": FAILURES["estimands"]}
        r["secondary"]["drift monitors"] = {"error": FAILURES["drift monitors"]}
        r["errors"], r["status"] = dict(FAILURES), "completed with errors"
    _edit_json(d / "results.json", edit)
    tables = d / "tables"
    for name in ("economics.csv", "estimands.csv", "estimands_ex_caiso.csv"):
        (tables / name).unlink()
    e5 = pd.read_csv(tables / "E5_blocks.csv")
    e5["error"] = None
    e5["approximate"] = e5["approximate"].astype(object)                             # as read back from a file that has a NaN in the column
    numbers = ["median_log_error", "lo", "hi", "failures", "failures_lo", "failures_hi", "n_nodes", "n_nan_replicates", "n_nan_replicates_failures", "approximate"]
    e5.loc[e5.block == "LEV6", numbers] = float("nan")
    e5.loc[e5.block == "LEV6", "error"] = FAILURES["E5/LEV6"]
    e5.to_csv(tables / "E5_blocks.csv", index=False)
    bh = pd.read_csv(tables / "secondary_bh.csv")
    bh[bh.family != "drift monitors"].to_csv(tables / "secondary_bh.csv", index=False)


def test_an_analysis_that_failed_is_dashes_and_a_reason_not_a_failed_build(tmp_path):
    import es.paper_tables_sealed  # noqa: F401
    from es.paper_lint import PENDING_TOKEN
    from es.paper_tables import BUILDERS, build_all
    src = _sources(tmp_path, family=REGISTERED, registered=REGISTERED)
    _fail_a_run(src)
    status = build_all(src, _tables_dir(tmp_path))                              # raises if any builder fails
    names = [f"tab_sealed_{n}_A" for n in RESULT_TABLES + DATA_TABLES]
    assert all(status[n] == "built" for n in names), status
    out = {n: BUILDERS[n][0](_fresh(src)) for n in names}
    for tex in out.values():
        _clean(tex)
        assert not PENDING_TOKEN.search(tex)
    fam = out["tab_sealed_family_A"]                                           # H-B1 did not complete; H-E has no SELC (the estimands failed)
    assert _row(fam, "H-B1").endswith(r"& estimand & -- & -- & -- & -- \\")
    assert _row(fam, "H-E").endswith(r"& confirmatory & -- & -- & 0.110 & interim (single window; decided after Test B) \\")
    assert "Not computed: " in fam and "H-B1 (ValueError: the contrast has nothing to infer from)" in fam
    assert r"estimands (LookupError: needs the economic rows (economics\_for did not complete))" in fam
    assert r"H-B2 [\textminus{}0.030, 0.140]" in fam and "H-B1 [" not in fam                # the note keeps the basic interval of what completed
    pc = out["tab_sealed_pc_hs_A"]
    assert _row(pc, "PC: P2 (season)").endswith("& -- & -- & -- \\\\") and "& 0.500 & [0.300, 0.700]" in _row(pc, "PC: P2 (all")
    assert _row(pc, "H-S: failures of plain DART lags, logistic").endswith("& -- & -- & -- \\\\") and "& 2 & [0.0, 5.0] & 31" in pc
    assert r"PC P2\_season (LookupError: score set ('roll', 'G1\_24', 'xgb', 42) is not available)" in pc and r"H-S logit (ValueError: boom)" in pc
    for name in ("e1", "e2e4", "ex_caiso"):                                    # every cell of the estimand tables is a dash
        rows = _body_rows(out[f"tab_sealed_{name}_A"])[1:]
        assert rows and all(row.endswith("& -- \\\\") for row in rows), name
        assert "Not computed:" in out[f"tab_sealed_{name}_A"]
    assert all(row.endswith("-- & -- & -- \\\\") for row in _body_rows(out["tab_sealed_e1_A"])[1:])
    assert r"estimands\_ex\_caiso (LookupError" in out["tab_sealed_ex_caiso_A"]
    e5 = out["tab_sealed_e5_A"]
    assert _row(e5, "absolute level").endswith("& 6 & -- & -- & -- \\\\") and r"midpoint & 3 & 0.10 [0.05, 0.20] & 3 [1.0, 6.0] & 31 \\" in e5
    assert r"E5 LEV6 (ValueError: no finite replicate)" in e5
    econ = out["tab_sealed_econ_A"]                                            # no shell, no rows: a statement
    assert econ.count("\\\\") == 1 and r"Not computed: economics (RuntimeError: the settlement frame is empty)." in econ
    audit = out["tab_sealed_audit_A"]                                          # what completed, and the part that did not
    assert len(_body_rows(audit)) == 1 + 5 and r"audit\_activity/P6 (ValueError: no node qualifies)" in audit
    bh = out["tab_sealed_bh_A"]
    assert r"audit robustness & 2 & -- & -- \\" in bh and r"drift monitors & -- & -- & -- \\" in bh and r"drift monitors (ValueError: no node has enough days)" in bh


def test_a_stage_that_failed_as_a_whole_leaves_no_slot_and_the_tables_still_build(tmp_path):
    """results["errors"] holds "stage/bootstraps" and the run has no PC, H-S, E5 slot and no H-B slot at all."""
    from es.paper_tables import build_all
    src = _sources(tmp_path, family=REGISTERED, registered=REGISTERED)
    path = src.out_dir / "sealed" / "A" / "results.json"
    _edit_json(path, lambda r: ([r.pop(k) for k in ("PC", "H-S", "H-B1", "H-B2")], r.update(errors={"stage/bootstraps": "RuntimeError: glue"}, status="completed with errors")))
    status = build_all(src, _tables_dir(tmp_path))
    assert all(status[f"tab_sealed_{n}_A"] == "built" for n in RESULT_TABLES), status
    fam = _build("tab_sealed_family_A", _fresh(src))
    assert _row(fam, "H-B1").endswith("& estimand & -- & -- & -- & -- \\\\") and "Not computed: H-B1 (RuntimeError: glue); H-B2 (RuntimeError: glue)." in fam
    assert _row(_build("tab_sealed_pc_hs_A", _fresh(src)), "H-S: failures of plain DART lags, XGBoost").endswith("& -- & -- & -- \\\\")
    _edit_json(path, lambda r: r.update(errors={}))                              # the same absence with no failure recorded is an error
    with pytest.raises(LookupError, match="no failure is recorded"):
        _build("tab_sealed_pc_hs_A", _fresh(src))


def _fail_the_decision(src):
    """_stage_decision: a decision that raised in Test A leaves the H-B entries without a status and H-E an error entry that keeps its
    numbers; Test B's H-E did not complete, so the step-down is {"error": ...} and its H-B statuses say so."""
    def a_edit(r):
        r["H-E"] = {"error": "ValueError: wording failed", **{k: v for k, v in r["H-E"].items() if k not in ("status", "confirmatory", "approximate")}}
        for hid in ("H-B1", "H-B2"):
            r[hid].pop("status")
        r["errors"], r["status"] = {"decision": "ValueError: wording failed"}, "completed with errors"

    def b_edit(r):
        r["H-E"] = {"error": "ValueError: rotation failed"}
        r["holm"] = {"error": "not computed: H-E, a confirmatory hypothesis, did not complete"}
        for hid in ("H-B1", "H-B2"):
            r[hid]["status"] = "undetermined: the Holm step-down needs H-E, which did not complete"
        r["errors"], r["status"] = {"H-E": "ValueError: rotation failed"}, "completed with errors"
    _edit_json(src.out_dir / "sealed" / "A" / "results.json", a_edit)
    _edit_json(src.out_dir / "sealed" / "B" / "results.json", b_edit)


def test_a_decision_that_did_not_complete_leaves_no_status_and_no_holm(tmp_path):
    src = _sources(tmp_path, tests=("A", "B"), family=ALL_TESTED)
    _fail_the_decision(src)
    src = _fresh(src)
    a = _build("tab_sealed_family_A", src)
    assert _row(a, "H-E").endswith("& confirmatory & 100.0\\% & [50.0\\%, 150.0\\%] & -- & -- \\\\")      # no p-value and no status of an entry that failed
    assert _row(a, "H-B1").endswith("& confirmatory & 0.120 & [0.030, 0.200] & 0.004 & -- \\\\")
    assert "Not computed: H-E (ValueError: wording failed); H-B1 (ValueError: wording failed); H-B2 (ValueError: wording failed)." in a
    dec = _build("tab_sealed_decision_B", src)
    assert _row(dec, "H-E") == "H-E & -- & -- & -- & -- \\\\"
    assert _row(dec, "H-B1") == r"H-B1 & A: 0.004; B: 0.004 & -- & yes & undetermined: the Holm step-down needs H-E, which did not complete \\"
    assert "holm (not computed: H-E, a confirmatory hypothesis, did not complete)" in dec and "Test A H-E" not in dec
    _clean(dec)
    _edit_json(src.out_dir / "sealed" / "A" / "results.json", lambda r: r.update(errors={}))
    with pytest.raises(LookupError, match="status"):                             # no status and no recorded failure: the contract is broken
        _build("tab_sealed_family_A", _fresh(src))


def test_a_number_that_is_not_finite_is_a_dash_and_a_reason_not_a_crash(tmp_path):
    """A completed analysis can still carry a NaN; no format prints one, so it is not computed, like a failed analysis."""
    import es.paper_tables_sealed  # noqa: F401
    from es.paper_tables import build_all
    src = _sources(tmp_path, family=REGISTERED, registered=REGISTERED)

    def edit(r):
        r["H-B2"]["ci_basic"] = [float("nan"), 0.1]
        r["PC"]["P2_all"]["median_diff"] = float("nan")
        r["H-S"]["xgb"]["failures"] = float("nan")
    _edit_json(src.out_dir / "sealed" / "A" / "results.json", edit)
    tables = src.out_dir / "sealed" / "A" / "tables"
    e5 = pd.read_csv(tables / "E5_blocks.csv")
    e5.loc[e5.block == "LEV6", "median_log_error"] = float("nan")
    e5.to_csv(tables / "E5_blocks.csv", index=False)
    e = pd.read_csv(tables / "estimands.csv")
    e.loc[e.stat.isin(["SELC", "E3:ESred", "E1:NET:timing"]), "lo"] = float("nan")
    e.to_csv(tables / "estimands.csv", index=False)
    status = build_all(src, _tables_dir(tmp_path))                              # raises if a builder fails
    assert all(status[f"tab_sealed_{n}_A"] == "built" for n in RESULT_TABLES), status
    fam = _build("tab_sealed_family_A", _fresh(src))
    assert _row(fam, "H-B2").endswith("& estimand & -- & -- & -- & -- \\\\") and r"H-B2 (ci\_basic not finite)" in fam
    assert _row(fam, "H-E").endswith("& confirmatory & -- & -- & 0.110 & interim (single window; decided after Test B) \\\\") and "SELC (estimate or interval not finite)" in fam
    pc = _build("tab_sealed_pc_hs_A", _fresh(src))
    assert _row(pc, "PC: P2 (all").endswith("& -- & -- & -- \\\\") and _row(pc, "H-S: failures of plain DART lags, XGBoost").endswith("& -- & -- & -- \\\\")
    assert _row(_build("tab_sealed_e5_A", _fresh(src)), "absolute level").endswith("& 6 & -- & -- & -- \\\\")
    e2e4 = _build("tab_sealed_e2e4_A", _fresh(src))
    assert r"E3: ESred & -- \\" in e2e4 and "estimands E3:ESred (estimate or interval not finite)" in e2e4
    assert r"NET & -- & -- & -- \\" not in _build("tab_sealed_e1_A", _fresh(src))     # only the one component is a dash
    assert _row(_build("tab_sealed_e1_A", _fresh(src)), "NET").startswith("NET & -- & ")


def test_a_null_flag_means_not_flagged_and_a_missing_key_is_an_error(tmp_path):
    """The approximate flags of results.json may be null (not flagged); the key itself must be there."""
    src = _sources(tmp_path, tests=("A", "B"), family=ALL_TESTED)
    paths = {t: src.out_dir / "sealed" / t / "results.json" for t in "AB"}

    def nulls(r):
        r["H-B1"]["ci_approximate"] = None
        r["H-E"]["approximate"] = None
        r["PC"]["P2_season"]["ci_approximate"] = None
        r["H-S"]["xgb"]["approximate"] = None
    for path in paths.values():
        _edit_json(path, nulls)
    src = _fresh(src)
    assert "textsuperscript" not in _row(_build("tab_sealed_family_A", src), "H-B1") and "textsuperscript" not in _row(_build("tab_sealed_family_B", src), "H-E")
    pc = _build("tab_sealed_pc_hs_A", src)
    assert r"0.200 & [0.100, 0.300] & 31" in pc and "& 2 & [0.0, 5.0] & 31" in pc                  # P2_season and H-S xgb: null, so unmarked
    assert r"0.500 & [0.300, 0.700]\textsuperscript{a} & 31" in pc                                   # P2_all still carries its True
    assert r"H-B1 & A: 0.004; B: 0.004 & yes & yes &" in _build("tab_sealed_decision_B", src)
    _edit_json(paths["A"], lambda r: r["H-B1"].pop("ci_approximate"))
    with pytest.raises(LookupError, match="ci_approximate"):
        _build("tab_sealed_family_A", _fresh(src))


def test_the_nan_replicate_flags_mark_the_intervals(tmp_path):
    """H-S and E5 carry `approximate` (more than 1% NaN replicates, Ruling 104(e)); H-B folds it into ci_approximate."""
    src = _sources(tmp_path)
    _edit_json(src.out_dir / "sealed" / "A" / "results.json", lambda r: r["H-S"]["xgb"].update(approximate=True, n_nan_replicates=60))
    path = src.out_dir / "sealed" / "A" / "tables" / "E5_blocks.csv"
    e5 = pd.read_csv(path)
    e5.loc[e5.block == "MID3", "approximate"] = True
    e5.to_csv(path, index=False)
    src = _fresh(src)
    pc = _build("tab_sealed_pc_hs_A", src)
    assert r"& 2 & [0.0, 5.0]\textsuperscript{a} & 31" in pc and r"& 1 & [0.0, 4.0] & 31" in pc
    table = _build("tab_sealed_e5_A", src)
    assert r"midpoint & 3 & 0.10 [0.05, 0.20]\textsuperscript{a} & 3 [1.0, 6.0]\textsuperscript{a} & 31 \\" in table
    assert r"absolute level & 6 & 0.10 [0.05, 0.20] & 3 [1.0, 6.0] & 31 \\" in table and "Interval labelled approximate" in table


# ---------------------------------------------------------------------------------------------- the readers follow es.sealed_run
def test_the_readers_follow_what_the_decision_rule_of_sealed_run_writes(tmp_path):
    """results.json parts made by es.sealed_run._decide itself (the pure decision rule), for the registered family."""
    import numpy as np
    from es.sealed_run import _decide
    fam = _protocol_inputs(REGISTERED)["family"]
    confirmatory = [h for h in HYPOTHESES if fam[h]["confirmatory"]]

    def hb(hid, t_hat, p):
        return {"t_hat": t_hat, "p": p, "ci": [t_hat - 0.09, t_hat + 0.08], "ci_basic": [t_hat - 0.08, t_hat + 0.09], "n_nodes": 31,
                "confirmatory": hid in confirmatory, "ci_approximate": bool(fam[hid]["approximate"])}
    he_a = {"t_obs": 0.08, "p": 0.11, "n_null": 91, "selc_obs": 1.0, "retained_share": [{"min": 0.95, "mean": 0.98}]}
    hb_a = {"H-B1": hb("H-B1", 0.12, 0.004), "H-B2": hb("H-B2", 0.05, 0.2)}
    out_a, holm_a = _decide(he_a, hb_a, fam, confirmatory, None)
    assert holm_a is None
    res_a = {**_results("A", REGISTERED), "H-E": out_a, **hb_a}
    he_b = {"t_obs": 0.09, "p": 0.01, "n_null": 8189, "selc_obs": 1.5, "retained_share": [{"min": 0.95, "mean": 0.98}, {"min": 0.93, "mean": 0.97}]}
    hb_b = {"H-B1": hb("H-B1", 0.10, 0.03), "H-B2": hb("H-B2", -0.02, 0.6)}
    out_b, holm_b = _decide(he_b, hb_b, fam, confirmatory, res_a)
    res_b = {**_results("B", REGISTERED), "H-E": out_b, "holm": holm_b, **hb_b}
    assert holm_b == {"H-E": True, "H-B1": None, "H-B2": None}                    # the estimands are no part of the Holm family
    src = _sources(tmp_path, tests=("A", "B"), family=REGISTERED, registered=REGISTERED)
    for t, res in (("A", res_a), ("B", res_b)):
        (src.out_dir / "sealed" / t / "results.json").write_text(json.dumps(res, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)))
    src = _fresh(src)
    assert "estimand (designation rule)" in _row(_build("tab_sealed_family_B", src), "H-B1")
    dec = _build("tab_sealed_decision_B", src)
    assert "timing skill within calendar cells detected" in dec
    assert "& no & estimand (designation rule) \\\\" in _row(dec, "H-B2")          # the sign of H-B2 changed between the quarters
    assert "& yes & estimand (designation rule) \\\\" in _row(dec, "H-B1")
    _clean(dec)


# ---------------------------------------------------------------------------------------------- widths in the real class
PDFLATEX = Path("/Library/TeX/texbin/pdflatex")
PDFTOTEXT = Path("/opt/homebrew/bin/pdftotext")
# ---------------------------------------------------------------------------------------------- Ruling 133: the words of the printed text
def test_the_family_and_decision_tables_say_designated_and_pre_specified_not_registered(tmp_path):
    """The column "registered as" is "designated as", in the shells and in the tables of a run; the panels of the Test B decision say
    "pre-specified wording" and "designated status" where they said "registered wording" and "registered status"."""
    ran = _sources(tmp_path / "ran", tests=("A", "B"), family=REGISTERED, registered=REGISTERED)
    shell = _sources(tmp_path / "shell", tests=(), registered=REGISTERED)
    bare = _sources(tmp_path / "bare", tests=())
    for label, src in (("ran", ran), ("shell", shell), ("bare", bare)):
        for test in ("A", "B"):
            fam = _build(f"tab_sealed_family_{test}", src)
            assert _body_rows(fam)[0] == r"hypothesis & designated as & estimate & 95\% CI & $p$ & status \\", (label, test)
            assert "registered" not in fam.lower(), (label, test)
    dec = _build("tab_sealed_decision_B", ran)                                         # H-E is the test: the Holm panel; H-B1 and H-B2 are estimands
    assert r"& pre-specified wording \\" in dec and r"& same sign & designated status \\" in dec
    assert "registered" not in dec.lower()
    both = _build("tab_sealed_decision_B", _sources(tmp_path / "all", tests=("A", "B")))   # every hypothesis a test: only the Holm panel
    assert r"& pre-specified wording \\" in both and "designated status" not in both


def test_what_the_sealed_run_wrote_is_printed_in_the_words_of_the_manuscript():
    """es.sealed_run (a registered file, not edited) writes "estimand (registration rule)" and "the registered pooled test" into the statuses
    and wordings of results.json. The tables print the designation rule and the pre-specified pooled test; the file keeps what the run wrote."""
    import es.sealed_run as sealed_run
    from es.paper_tables_sealed import PRINTED_WORDS, _printed_words, _shown_status
    constants = {n: v for n, v in vars(sealed_run).items() if n.startswith(("STATUS_", "WORDING_")) and isinstance(v, str)}
    assert {"STATUS_HB_ESTIMAND", "STATUS_HE_ALONE", "WORDING_HE_ESTIMAND", "WORDING_HE_DETECTED"} <= set(constants)
    assert "registration rule" in sealed_run.STATUS_HB_ESTIMAND and "registered pooled test" in sealed_run.STATUS_HE_ALONE      # what it writes
    for name, value in constants.items():                                              # every status and wording the run can write prints without it
        assert not re.search(r"regist", _printed_words(value), flags=re.I), (name, _printed_words(value))
    assert _printed_words(sealed_run.STATUS_HB_ESTIMAND) == "estimand (designation rule)"
    assert _printed_words(sealed_run.WORDING_HE_ESTIMAND) == "estimand (designation rule): pooled SELC with CI"
    assert _printed_words(sealed_run.STATUS_HE_ALONE) == ("not confirmatory (Test A's output is not available; "
                                                          "the pre-specified pooled test is unavailable)")
    # an estimand shows its label without the reasons of the rule, a test keeps its whole status, a missing status and a dash pass through
    assert _shown_status(sealed_run.STATUS_HB_ESTIMAND + ": simulated size 0.130 > 1.5 x 0.0167 (39 of 300 replications)", False) == "estimand (designation rule)"
    assert _shown_status(sealed_run.STATUS_HB_BOTH, True) == sealed_run.STATUS_HB_BOTH
    assert _shown_status(sealed_run.STATUS_HE_ALONE, True) == _printed_words(sealed_run.STATUS_HE_ALONE)
    assert _shown_status(None, False) is None and _shown_status("--", False) == "--" and _printed_words(None) is None
    assert ("registration rule", "designation rule") in PRINTED_WORDS


def test_the_notes_that_index_md_prints_beside_the_tables_use_the_words_of_the_manuscript():
    """tables/INDEX.md prints es.paper_tables.NOTES beside each table: the Test B decision is the Holm panel for the tests of the confirmatory
    family, and no note of any table says "registered" or "registration" (the name tab_registration is not a note)."""
    import es.paper_tables_sealed  # noqa: F401
    from es.paper_tables import NOTES
    assert NOTES["tab_sealed_decision_B"].startswith("Holm panel for the tests of the confirmatory family, estimand panel for the others")
    assert "tab_registration" in NOTES and len(NOTES) >= 30                               # the notes of the development and the sealed tables
    bad = {name: note for name, note in NOTES.items() if re.search(r"regist(?:er|ered|ers|ration)", note, flags=re.I)}
    assert not bad, bad


def _all_sealed_tables(src, folder):
    import es.paper_tables_sealed  # noqa: F401
    from es.paper_tables import build_all
    status = build_all(src, folder, raise_on_failure=False)
    assert not [n for n, s in status.items() if s.startswith("FAILED")], status
    return {n: (folder / f"{n}.tex").read_text(encoding="utf-8") for n, s in status.items() if s == "built" and n.startswith("tab_sealed")}


def test_no_sealed_table_in_any_state_prints_registered_or_registration(tmp_path):
    """A test that ran (the family as designated, a swapped one), a shell with the family and without it, a run with analyses that failed,
    an undetermined decision and a test that did not run: no table of any of them says "registered" or "registration"."""
    failed = _sources(tmp_path / "failed", tests=("A", "B"), family=REGISTERED, registered=REGISTERED)
    _fail_a_run(failed)
    undetermined = _sources(tmp_path / "undetermined", tests=("A", "B"), family=ALL_TESTED)
    _fail_the_decision(undetermined)
    states = {"ran": _sources(tmp_path / "ran", tests=("A", "B"), family=REGISTERED, registered=REGISTERED),
              "swapped": _sources(tmp_path / "swapped", tests=("A", "B"), family=SWAPPED),
              "shell": _sources(tmp_path / "shell", tests=(), registered=REGISTERED), "bare": _sources(tmp_path / "bare", tests=()),
              "notrun": _sources(tmp_path / "notrun", tests=("A",), not_run=("A",), registered=REGISTERED),
              "failed": failed, "undetermined": undetermined}
    seen = 0
    for label, src in states.items():
        for name, tex in _all_sealed_tables(src, tmp_path / "tables" / label).items():
            seen += 1
            assert not re.search(r"regist(?:er|ered|ers|ration)", tex, flags=re.I), (label, name)
    assert seen > 70, seen
    assert "estimand (designation rule)" in _all_sealed_tables(states["ran"], tmp_path / "tables" / "again")["tab_sealed_family_A"]


LATEX_PREAMBLE = r"""\documentclass[pdflatex,sn-mathphys-num]{sn-jnl}
\usepackage{graphicx}
\usepackage{multirow}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{amsthm}
\usepackage[title]{appendix}
\usepackage{xcolor}
\usepackage{textcomp}
\usepackage{manyfoot}
\usepackage{booktabs}
\usepackage{longtable}
%%PENDING%%
\newcommand{\gentable}[1]{\IfFileExists{generated/tables/#1.tex}{\input{generated/tables/#1.tex}}{\begin{tabular}{l}\textbf{#1 missing}\end{tabular}}}
\title[probe]{probe}
\author*[1]{\fnm{A} \sur{B}}\email{a@b.c}
\affil*[1]{\orgdiv{D}, \orgname{U}, \orgaddress{\city{C}, \state{S}, \country{USA}}}
\begin{document}
\maketitle
\section{Size and power of the designation rule}\label{esm:size-power}
"""


def _width_problems(log: str) -> dict:
    current, problems = None, {}
    for line in log.splitlines():
        marker = re.match(r"=== (\S+) ===", line)
        if marker:
            current = marker.group(1)
            continue
        over = re.match(r"Overfull \\hbox \(([\d.]+)pt too wide\)", line)
        if over and float(over.group(1)) > 1.0 and current:
            problems.setdefault(current, []).append(f"overfull hbox by {over.group(1)} pt")
        if "Float too large for page" in line and current:
            problems.setdefault(current, []).append(line.strip())
    return problems


@pytest.mark.slow
@pytest.mark.skipif(not (PDFLATEX.exists() and PDFTOTEXT.exists()), reason="TeX Live or poppler not installed")
def test_every_sealed_table_fits_the_page_in_the_real_class(tmp_path):
    """Every sealed table, as a shell, as a result and as the statement of a test that did not run, set in a `table` float of
    the real class: no overfull box over 1 pt and no float cut off. The results carry realistic magnitudes (the quarter's J
    values of the development analogue AQ1; Test B's pooled estimands are twice as large, which makes the E1 table change its
    layout), excluded nodes with long names and a full audit."""
    import os
    import subprocess
    import es.paper_tables_sealed  # noqa: F401
    from es.paper_macros import PENDING, PENDING_TEXT
    from es.paper_paths import PAPER_DIR
    from es.paper_tables import build_all
    from es.sealed_run import score_keys
    if not (PAPER_DIR / "sn-jnl.cls").exists():
        pytest.skip("sn-jnl.cls is not in the paper directory")
    ran = _sources(tmp_path / "ran", tests=("A", "B"), family=REGISTERED, registered=REGISTERED)
    for t, (est, lo, hi) in {"A": (74631.6, -7871.1, 198246.1), "B": (149263.2, -15742.2, 396492.2)}.items():
        d = ran.out_dir / "sealed" / t / "tables"
        e = pd.read_csv(d / "estimands.csv")
        e.loc[e.stat.str.startswith("E1:"), ["estimate", "lo", "hi"]] = [est, lo, hi]
        e.loc[e.stat == "NET", ["estimate", "lo", "hi"]] = [-4600.3, -13662.7, 2924.3]
        e.to_csv(d / "estimands.csv", index=False)
        cov = pd.read_csv(d / "coverage.csv")
        extra = pd.DataFrame([{"market": "CAISO", "node": f"TH_SP15_GEN-APND_{i}", "expected_hours": 2208, "hours": 700, "share": 0.317,
                               "ok": False, "n_pos": 0, "feasible": False, "reason": "lt_20_positives", "usable": False} for i in range(3)])
        pd.concat([cov, extra], ignore_index=True).to_csv(d / "coverage.csv", index=False)
        e5 = pd.read_csv(d / "E5_blocks.csv")
        e5.loc[e5.block.isin(["G6", "MID3"]), "median_log_error"] = 0.30                    # an interval that misses its estimate: the note is typeset
        e5.to_csv(d / "E5_blocks.csv", index=False)
        a = pd.read_csv(d / "audit_activity.csv").iloc[:2]                 # two nodes; the rows of the run's audit: P1 on each score set, six policies on the primary
        parts = [a.assign(maintenance=kind, block=block, estimator=est_, seed=seed, policy="P1") for kind, block, est_, seed in score_keys()]
        parts += [a.assign(maintenance="roll", block="G1_24", estimator="xgb", seed=42, policy=p)
                  for p in ("P4_56", "P4_28", "P5", "P6", "P2_season", "P2_all")]
        pd.concat(parts, ignore_index=True).to_csv(d / "audit_activity.csv", index=False)
    failed = _sources(tmp_path / "failed", tests=("A", "B"), family=REGISTERED, registered=REGISTERED)
    _fail_a_run(failed)                                                              # Test A with six analyses that did not complete
    undetermined = _sources(tmp_path / "undetermined", tests=("A", "B"), family=ALL_TESTED)
    _fail_the_decision(undetermined)                                                 # a long status in the Holm panel
    docs = {"ran": ran, "shell": _sources(tmp_path / "shell", tests=(), registered=REGISTERED), "failed": failed, "undetermined": undetermined,
            "notrun": _sources(tmp_path / "notrun", tests=("A",), not_run=("A",), registered=REGISTERED)}
    doc = tmp_path / "doc"
    (doc / "generated" / "tables").mkdir(parents=True)
    names = []
    for label, src in docs.items():
        status = build_all(src, doc / "generated" / label, raise_on_failure=False)
        assert not [n for n, s in status.items() if s.startswith("FAILED")], status
        for n, s in status.items():
            if s == "built" and n.startswith("tab_sealed"):
                (doc / "generated" / "tables" / f"{label}_{n}.tex").write_text((doc / "generated" / label / f"{n}.tex").read_text())
                names.append(f"{label}_{n}")
    assert {"ran_tab_sealed_decision_B", "ran_tab_sealed_audit_B", "shell_tab_sealed_family_A", "notrun_tab_sealed_audit_A",
            "failed_tab_sealed_econ_A", "failed_tab_sealed_family_A", "undetermined_tab_sealed_decision_B"} <= set(names)
    for n in names:                                                                  # Ruling 133: none of the typeset tables says registered or registration
        assert not re.search(r"regist(?:er|ered|ers|ration)", (doc / "generated" / "tables" / f"{n}.tex").read_text(), flags=re.I), n
    pending = "\n".join(f"\\newcommand{{{command}}}{{\\textbf{{{PENDING_TEXT[stage]}}}}}" for stage, command in PENDING.items())
    body = []
    for n in names:
        mark = n.replace("_", "")
        body.append(f"\\typeout{{=== {n} ===}}\\begin{{table}}[p]\\caption{{STARTMARK{mark} A caption of realistic length that wraps over two lines.}}"
                    f"\\gentable{{{n}}}\\par ENDMARK{mark}\\end{{table}}\\clearpage")
    (doc / "probe.tex").write_text(LATEX_PREAMBLE.replace("%%PENDING%%", pending) + "\n".join(body) + "\n\\end{document}\n")
    env = dict(os.environ, TEXINPUTS=f".:{PAPER_DIR}:", BSTINPUTS=f".:{PAPER_DIR}:")
    run = subprocess.run([str(PDFLATEX), "-interaction=nonstopmode", "-halt-on-error", "probe.tex"], cwd=doc, env=env, capture_output=True, timeout=900)
    log = (doc / "probe.log").read_bytes().decode("utf-8", errors="replace")
    assert run.returncode == 0, "\n".join(l for l in log.splitlines() if l.startswith("!") or l.startswith("l.")) + "\n" + log[-1500:]
    assert not _width_problems(log), _width_problems(log)
    text = subprocess.run([str(PDFTOTEXT), "probe.pdf", "-"], cwd=doc, capture_output=True, timeout=300).stdout.decode("utf-8", errors="replace")
    clipped = [n for n in names if f"STARTMARK{n.replace('_', '')}" not in text or f"ENDMARK{n.replace('_', '')}" not in text]
    assert not clipped, f"floats cut off by the page: {clipped}"
    assert "component" in (doc / "generated" / "tables" / "ran_tab_sealed_e1_B.tex").read_text()                  # pooled magnitudes: the long layout
    assert "component" not in (doc / "generated" / "tables" / "ran_tab_sealed_e1_A.tex").read_text()              # a quarter's: the wide one
