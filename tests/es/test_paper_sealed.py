"""Ruling 107 (3, 5): the sealed H-B macros and the run's own status.

* hbOneCI* and hbTwoCI* print the percentile bootstrap interval (results.json "ci"; Ruling 110: for a demoted estimand the basic
  interval inherits the bootstrap shift that drove the over-rejection, so the percentile one is primary); hbOneCIBasic* and
  hbTwoCIBasic* print the basic interval ("ci_basic"); hbOneCIPercentile* and hbTwoCIPercentile* are aliases of the canonical ones.
* hbOneP* and hbTwoP* print "n/a (estimand)" when the family marks the hypothesis as an estimand: results.json["family"] is the
  list of the confirmatory hypotheses the run used, so a hypothesis that is not in it is an estimand.
* hbOneReplicated and hbTwoReplicated print n/a for the null that Test B's _decide writes for an estimand (it has no replication
  verdict), as holmHBOne and holmHBTwo do.
* runStatus* is results["status"] and nRunErrors* the number of entries of results["errors"].

The fixtures follow what es/sealed_run.py writes (_stage_bootstraps, _decide, _Run.status, _execute); one test builds a results.json
with those writers themselves."""
import json
from pathlib import Path

import pytest

TESTS = ("TestA", "TestB")
CI_MACROS = tuple(f"{h}CI{kind}{t}" for h in ("hbOne", "hbTwo") for kind in ("", "Basic", "Percentile") for t in TESTS)
P_MACROS = tuple(f"{h}P{t}" for h in ("hbOne", "hbTwo") for t in TESTS)
STATUS_MACROS = tuple(f"{name}{t}" for name in ("runStatus", "nRunErrors") for t in TESTS)
NEW_MACROS = tuple(f"hb{n}CI{kind}{t}" for n in ("One", "Two") for kind in ("Basic", "Percentile") for t in TESTS) + STATUS_MACROS
ESTIMAND = "n/a (estimand)"
#: The interval that es.sealed_run registers for a test and for an estimand (INTERVAL_HB_TEST and INTERVAL_HB_ESTIMAND): the writer
#: test compares them. The status of a demoted H-B carries the reasons of the registration rule in both tests (the writer's M11).
INTERVAL_TEST, INTERVAL_ESTIMAND = "basic", "percentile (primary); basic (diagnostic)"
#: The interval of H-S and E5 (INTERVAL_E5_HS, Ruling 118, M2): the calendar-aligned draw's percentile interval is `ci`, the registered
#: independent draws' is `ci_independent`, a diagnostic. The writer test compares the string.
INTERVAL_LEVEL = "percentile, calendar-aligned (primary); independent draws (diagnostic)"
ESTIMAND_STATUS = "estimand (registration rule): simulated size 0.127 > ..."
ABSENT = object()           # for `replicated`: no such key in the entry (Test A has no replication verdict)


def hb_entry(t_hat, p, ci, ci_basic, confirmatory=True, status="rejected in Test A at 0.05/2; replication pending", replicated=ABSENT):
    """One H-B hypothesis as _stage_bootstraps writes it, with the status (and, in Test B, the verdict) that _decide adds. A test has
    its basic-bootstrap p and registers the basic interval; an estimand has no p (its bootstrap p is a diagnostic) and registers the
    percentile interval (Ruling 109 I5, Ruling 110, Ruling 113). `replicated` is a key of Test B's entries only, and None there for
    an estimand."""
    entry = {"t_hat": t_hat, "ci": list(ci), "ci_basic": list(ci_basic), "n_nodes": 33, "n_nan_replicates": 0,
             "confirmatory": confirmatory, "ci_approximate": True,
             "interval_registered": INTERVAL_TEST if confirmatory else INTERVAL_ESTIMAND}
    if confirmatory:
        entry["p"] = p
    else:
        entry["diagnostics"] = {"bootstrap_p_one_sided": p}
    entry["status"] = status
    if replicated is not ABSENT:
        entry["replicated"] = replicated
    return entry


def sealed_results(test_id="A", confirmatory=("H-E", "H-B1"), errors=None, status=None, ran=True, drop=()):
    """The results.json of a sealed test, with the keys run_sealed writes (a fixture; values are planted so that the percentile and
    the basic interval differ, and H-S's primary interval and its independent diagnostic). `confirmatory` is the run's family: an H-B hypothesis that it lists is a test and one that it leaves
    out an estimand, and each entry has the keys of its kind in this test (hb_entry). H-B2 is not in the default family: the
    registration rule demoted it to an estimand."""
    errors = {} if errors is None else errors
    b = test_id == "B"

    def hb(hid, t_hat, p, ci, ci_basic, replicated):
        tested = hid in confirmatory
        text = ("less portable in both 2026 quarters" if b else "rejected in Test A at 0.05/2; replication pending") if tested else ESTIMAND_STATUS
        verdict = ABSENT if not b else replicated if tested else None        # Test A has none; in Test B an estimand's is null
        return hb_entry(t_hat, p, ci, ci_basic, tested, text, replicated=verdict)

    res = {"test": test_id, "started_utc": "2026-10-28T01:02:03+00:00", "protocol_sha256": "ab" * 32,
           "registration": "10.17605/OSF.IO/ABCDE", "registered_utc": "2026-10-20T15:04:00+00:00", "n_rep": 2000,
           "bound_files": {}, "departures": [], "sealed_manifests": {}, "eta": 0.05, "family": list(confirmatory),
           "rows_read": 250000, "rows_kept": 240000,
           "coverage": {"ran": ran, "usable_nodes": 33 if ran else 20, "nonfinite_dropped": {}},
           "errors": errors, "status": status or ("completed" if not errors else "completed with secondary errors"),
           "finished_utc": "2026-10-28T03:00:00+00:00"}
    if not ran:
        res["errors"], res["status"] = {}, "not run (coverage rule)"
        return res
    he = {"status": "decided (pooled A+B)" if b else "interim (single window; decided after Test B)", "t_obs": 2.1, "p": 0.02,
          "n_null": 4159, "selc_obs": 0.004, "retained_share": [{"min": 0.95, "mean": 0.98} for _ in range(2 if b else 1)],
          "approximate": False, "confirmatory": "H-E" in confirmatory}
    if b:                                                    # _decide words the decision on the pooled windows only
        he["wording"] = "timing skill within calendar cells detected" if he["confirmatory"] else "estimand (registration rule): pooled SELC with CI"
    res.update({
        "tau_check": {"max_abs_diff": 0.0},
        "H-B1": hb("H-B1", 0.052, 0.011, (0.0121, 0.0934), (0.0147, 0.0958), replicated=True),
        "H-B2": hb("H-B2", -0.018, 0.640, (-0.0412, 0.0033), (-0.0391, 0.0052), replicated=False),
        "H-E": he,
        "PC": {"P2_season": {"median_diff": 0.004, "ci": [-0.01, 0.02], "n_nodes": 33, "n_nan_replicates": 0, "ci_approximate": True},
               "P2_all": {"median_diff": 0.001, "ci": [-0.02, 0.02], "n_nodes": 33, "n_nan_replicates": 0, "ci_approximate": True}},
        "H-S": {e: {"failures": 12.0, "ci": [9.0, 15.0], "ci_independent": [7.0, 17.0], "scheme": "aligned",
                    "interval_registered": INTERVAL_LEVEL, "n_nodes": 33, "n_nan_replicates": 0, "n_nan_replicates_independent": 0,
                    "approximate": True} for e in ("xgb", "logit")},
        "estimands_scope": "pooled A+B" if b else "single window (interim)", "secondary": {}})
    if b:
        res["holm"] = {"H-E": True, "H-B1": True, "H-B2": None}
    for key in drop:
        res.pop(key, None)
    return res


def write(tmp_path, results, test_id="A") -> Path:
    out = tmp_path / "out"
    (out / "sealed" / test_id).mkdir(parents=True, exist_ok=True)
    (out / "sealed" / test_id / "results.json").write_text(json.dumps(results))
    return out


def resolved(name, tmp_path):
    from es.paper_macros import Sources, resolve
    from es.paper_registry import entries_by_name
    return resolve(entries_by_name()[name], Sources(tmp_path / "out", tmp_path / "paper", tmp_path / "docs_sealed"))


def value(name, tmp_path):
    r = resolved(name, tmp_path)
    assert r["status"] == "ok", (name, r)
    return r["value"]


# ------------------------------------------------------------------------------------------------ the registry
def test_the_registry_reads_the_percentile_interval_as_the_canonical_one_and_names_the_basic_one():
    from es.paper_registry import entries_by_name
    from es.paper_tex import FORMATS, PAIR_FORMATS
    by = entries_by_name()
    for t in TESTS:
        res = f"sealed/{t[-1]}/results.json"
        for h, hid in (("hbOne", "H-B1"), ("hbTwo", "H-B2")):
            canonical, basic, alias = by[f"{h}CI{t}"], by[f"{h}CIBasic{t}"], by[f"{h}CIPercentile{t}"]
            assert (canonical.source, canonical.query, canonical.fmt) == (res, f"{hid}/ci", "ci3")
            assert (basic.source, basic.query, basic.fmt) == (res, f"{hid}/ci_basic", "ci3")
            # the Percentile name is an alias: the same entry as the canonical one under another name
            assert (alias.source, alias.query, alias.column, alias.fmt, alias.agg, alias.by, alias.nullable, alias.stage) == \
                (canonical.source, canonical.query, canonical.column, canonical.fmt, canonical.agg, canonical.by, canonical.nullable,
                 canonical.stage)
            assert canonical.name in alias.note and "alias" in alias.note
            assert by[f"{h}P{t}"].estimand_of == hid and by[f"{h}P{t}"].query == f"{hid}/p"
            assert by[f"{h}T{t}"].estimand_of == ""                              # only the p-value is replaced by the estimand text
        assert (by[f"runStatus{t}"].query, by[f"runStatus{t}"].fmt) == ("status", "text")
        assert (by[f"nRunErrors{t}"].query, by[f"nRunErrors{t}"].agg, by[f"nRunErrors{t}"].fmt) == ("errors", "count", "int")
        assert by[f"pcSeasonCI{t}"].query == "PC/P2_season/ci" and by[f"hsFailCI{t}"].query == "H-S/xgb/ci"      # the others stay percentile
        assert by[f"runStatus{t}"].stage == by[f"nRunErrors{t}"].stage == f"sealed{t[-1]}"
    for name in NEW_MACROS:
        assert name in by and by[name].fmt in FORMATS | PAIR_FORMATS


# ------------------------------------------------------------------------------------------------ the intervals
def test_hb_ci_is_the_percentile_interval_and_the_basic_one_has_its_own_macro(tmp_path):
    write(tmp_path, sealed_results("A"), "A")
    write(tmp_path, sealed_results("B"), "B")
    from es.paper_tex import MINUS
    got = {n: value(n, tmp_path) for n in CI_MACROS}
    for t in TESTS:
        assert got[f"hbOneCI{t}"] == got[f"hbOneCIPercentile{t}"] == "[0.012, 0.093]"       # the percentile interval, and its alias
        assert got[f"hbOneCIBasic{t}"] == "[0.015, 0.096]"
        assert got[f"hbTwoCI{t}"] == got[f"hbTwoCIPercentile{t}"] == f"[{MINUS}0.041, 0.003]"
        assert got[f"hbTwoCIBasic{t}"] == f"[{MINUS}0.039, 0.005]"
    res = sealed_results("A")                                                   # a changed source changes the macros that read it
    res["H-B1"]["ci_basic"], res["H-B2"]["ci"] = [0.2, 0.4], [0.5, 0.7]
    write(tmp_path, res, "A")
    assert value("hbOneCIBasicTestA", tmp_path) == "[0.200, 0.400]"
    assert value("hbOneCITestA", tmp_path) == value("hbOneCIPercentileTestA", tmp_path) == "[0.012, 0.093]"      # H-B1's ci did not change
    assert value("hbTwoCITestA", tmp_path) == value("hbTwoCIPercentileTestA", tmp_path) == "[0.500, 0.700]"
    assert value("hbTwoCIBasicTestA", tmp_path) == f"[{MINUS}0.039, 0.005]"
    res["H-B1"].pop("ci_basic")                                                 # no basic interval: only the basic macro is an error
    write(tmp_path, res, "A")
    with pytest.raises(LookupError, match="hbOneCIBasicTestA"):
        resolved("hbOneCIBasicTestA", tmp_path)
    assert value("hbOneCITestA", tmp_path) == "[0.012, 0.093]"
    res["H-B1"].pop("ci")                                                       # no percentile interval: the canonical macro and its alias
    write(tmp_path, res, "A")
    for name in ("hbOneCITestA", "hbOneCIPercentileTestA"):
        with pytest.raises(LookupError, match=name):
            resolved(name, tmp_path)


# ------------------------------------------------------------------------------------------------ estimands
def test_hb_p_is_n_a_estimand_when_the_family_marks_the_hypothesis_as_one(tmp_path):
    for test_id in "AB":
        write(tmp_path, sealed_results(test_id), test_id)
    assert value("hbOnePTestA", tmp_path) == "0.011" and value("hbOnePTestB", tmp_path) == "0.011"       # confirmatory: its p
    for t in TESTS:
        r = resolved(f"hbTwoP{t}", tmp_path)
        assert r["status"] == "ok" and r["value"] == ESTIMAND and "H-B2" in r["detail"] and "estimand" in r["detail"]
        assert value(f"hbTwoT{t}", tmp_path) == "\\textminus{}0.018"            # only the p-value is replaced
        assert value(f"hbTwoStatus{t}", tmp_path).startswith("estimand (registration rule)")
    # the family is the list of the run's confirmatory hypotheses: all three confirmatory, nothing is an estimand
    write(tmp_path, sealed_results("A", confirmatory=("H-E", "H-B1", "H-B2")), "A")
    assert value("hbTwoPTestA", tmp_path) == "0.640" and value("hbOnePTestA", tmp_path) == "0.011"
    # H-B1 the estimand instead
    write(tmp_path, sealed_results("A", confirmatory=("H-E", "H-B2")), "A")
    assert (value("hbOnePTestA", tmp_path), value("hbTwoPTestA", tmp_path)) == (ESTIMAND, "0.640")
    write(tmp_path, sealed_results("A", confirmatory=()), "A")
    assert (value("hbOnePTestA", tmp_path), value("hbTwoPTestA", tmp_path)) == (ESTIMAND, ESTIMAND)


def test_the_family_can_also_be_the_registered_one_or_the_entrys_own_flag(tmp_path):
    """results.json["family"] is a list of ids; a protocol_inputs-shaped dict is read the same way, and a results.json without a
    family falls back to the confirmatory flag of the H-B entry. With none of them the macro cannot tell, and refuses."""
    registered = {"H-E": {"confirmatory": True}, "H-B1": {"confirmatory": True}, "H-B2": {"confirmatory": False}}
    res = sealed_results("A")
    res["family"] = registered
    write(tmp_path, res, "A")
    assert (value("hbOnePTestA", tmp_path), value("hbTwoPTestA", tmp_path)) == ("0.011", ESTIMAND)
    res["family"]["H-B1"]["confirmatory"] = False
    write(tmp_path, res, "A")
    assert (value("hbOnePTestA", tmp_path), value("hbTwoPTestA", tmp_path)) == (ESTIMAND, ESTIMAND)
    write(tmp_path, sealed_results("A", drop=("family",)), "A")                  # the entries say it: H-B1 true, H-B2 false
    assert (value("hbOnePTestA", tmp_path), value("hbTwoPTestA", tmp_path)) == ("0.011", ESTIMAND)
    res = sealed_results("A", drop=("family",))
    res["H-B1"].pop("confirmatory")
    write(tmp_path, res, "A")
    with pytest.raises(LookupError, match=r"hbOnePTestA.*cannot tell whether H-B1 is an estimand"):
        resolved("hbOnePTestA", tmp_path)
    assert value("hbTwoPTestA", tmp_path) == ESTIMAND


def test_an_h_b_test_that_failed_is_an_error_for_a_confirmatory_hypothesis_and_still_an_estimand_otherwise(tmp_path):
    res = sealed_results("A", confirmatory=("H-E", "H-B1", "H-B2"))
    res["H-B2"] = {"error": "ValueError: the contrast has nothing to infer from"}       # run.error_entry("H-B2")
    write(tmp_path, res, "A")
    with pytest.raises(LookupError, match="hbTwoPTestA"):                                # a missing p of a confirmatory test is not n/a
        resolved("hbTwoPTestA", tmp_path)
    res["family"] = ["H-E", "H-B1"]
    write(tmp_path, res, "A")
    assert value("hbTwoPTestA", tmp_path) == ESTIMAND                                   # an estimand has no p to read


def test_the_replication_verdict_of_an_estimand_is_n_a_and_one_that_is_no_key_is_an_error(tmp_path):
    """Test B's _decide writes `replicated: None` for a demoted H-B (an estimand has no replication verdict) and none in Test A: the
    macros print n/a for the null, as for the Holm decisions, and a test keeps its yes or no. A verdict that is no key stays an error."""
    write(tmp_path, sealed_results("B"), "B")
    assert (value("hbOneReplicated", tmp_path), value("hbTwoReplicated", tmp_path)) == ("yes", "n/a")      # H-B1 a test, H-B2 an estimand
    write(tmp_path, sealed_results("B", confirmatory=("H-E", "H-B1", "H-B2")), "B")                          # both are tests: both have one
    assert (value("hbOneReplicated", tmp_path), value("hbTwoReplicated", tmp_path)) == ("yes", "no")
    write(tmp_path, sealed_results("B", confirmatory=()), "B")                                                # both are estimands
    assert (value("hbOneReplicated", tmp_path), value("hbTwoReplicated", tmp_path)) == ("n/a", "n/a")
    res = sealed_results("B")
    res["H-B2"].pop("replicated")
    write(tmp_path, res, "B")
    with pytest.raises(LookupError, match="hbTwoReplicated"):
        resolved("hbTwoReplicated", tmp_path)
    assert value("hbOneReplicated", tmp_path) == "yes"


def test_a_test_that_did_not_run_is_pending_not_an_estimand(tmp_path):
    from es.paper_macros import PENDING
    write(tmp_path, sealed_results("A", ran=False), "A")
    for name in ("hbOnePTestA", "hbTwoPTestA", "hbOneCITestA", "hbOneCIBasicTestA", "hbOneCIPercentileTestA"):
        r = resolved(name, tmp_path)
        assert r["status"] == "pending" and r["value"] == PENDING["sealedA"] and r["detail"] == "test did not run", name


# ------------------------------------------------------------------------------------------------ the run's status
def test_run_status_and_error_count_macros(tmp_path):
    errors = {"audit robustness": "ValueError: boom", "BH tables": "LookupError: no family"}
    write(tmp_path, sealed_results("A", errors=errors), "A")
    write(tmp_path, sealed_results("B"), "B")
    assert {n: value(n, tmp_path) for n in STATUS_MACROS} == {
        "runStatusTestA": "completed with secondary errors", "nRunErrorsTestA": "2",
        "runStatusTestB": "completed", "nRunErrorsTestB": "0"}
    write(tmp_path, sealed_results("B", errors={"H-E": "LookupError: needs the economic rows"}, status="completed with errors"), "B")
    assert (value("runStatusTestB", tmp_path), value("nRunErrorsTestB", tmp_path)) == ("completed with errors", "1")
    # a test that did not run still has its status and an empty error list: the header that run_sealed writes before the coverage rule
    write(tmp_path, sealed_results("A", ran=False), "A")
    assert (value("runStatusTestA", tmp_path), value("nRunErrorsTestA", tmp_path)) == ("not run (coverage rule)", "0")
    # no errors key, or one that is neither an object nor a list: the count is an error, not a zero
    res = sealed_results("A", drop=("errors",))
    write(tmp_path, res, "A")
    with pytest.raises(LookupError, match="nRunErrorsTestA"):
        resolved("nRunErrorsTestA", tmp_path)
    res["errors"] = 3
    write(tmp_path, res, "A")
    with pytest.raises(LookupError, match=r"nRunErrorsTestA.*no count"):
        resolved("nRunErrorsTestA", tmp_path)
    assert value("runStatusTestA", tmp_path) == "completed"


def test_the_status_text_is_escaped(tmp_path):
    write(tmp_path, sealed_results("A", status="completed with errors in A&B_50%"), "A")
    assert value("runStatusTestA", tmp_path) == "completed with errors in A\\&B\\_50\\%"


def test_the_h_s_macros_print_the_primary_interval_and_not_the_independent_diagnostic(tmp_path):
    """Ruling 118, M2: results.json H-S[est]["ci"] is the calendar-aligned draw's percentile interval, the primary one; the
    registered independent draws' interval sits beside it as "ci_independent" (a diagnostic), and no macro prints it."""
    write(tmp_path, sealed_results("A"), "A")
    assert value("hsFailCITestA", tmp_path) == value("hsFailLogitCITestA", tmp_path) == "[9.0, 15.0]"
    assert value("hsFailTestA", tmp_path) == value("hsFailLogitTestA", tmp_path) == "12"


# ------------------------------------------------------------------------------------------------ the writers themselves
def test_the_macros_read_a_results_json_that_the_writers_of_sealed_run_wrote(tmp_path, monkeypatch):
    """_stage_bootstraps, _decide and _Run.status of es.sealed_run build the results (the statistical kernels are replaced by
    results of their shape); the macros read what those writers put in results.json, and the hand fixture has the writers' keys."""
    ss = pytest.importorskip("es.sealed_run")
    from types import SimpleNamespace

    def contrast(t, p, ci, basic):
        return {"t_hat": t, "p": p, "ci": ci, "ci_basic": basic, "n_nodes": 33, "n_nan_replicates": 0, "nan_approximate": False}

    by_block = {"LEV6": contrast(0.052, 0.011, (0.0121, 0.0934), (0.0147, 0.0958)),
                "ROLL_DISP9": contrast(-0.018, 0.640, (-0.0412, 0.0033), (-0.0391, 0.0052))}
    level = {"t_hat": 12.0, "ci": (9.0, 15.0), "ci_basic": (9.5, 15.5), "n_nodes": 33, "n_nan_replicates": 0, "nan_approximate": False,
             "scheme": "aligned", "ci_independent": (7.0, 17.0), "n_nan_replicates_independent": 0,           # _level's keys since
             "nan_approximate_independent": False, "approximate": True}                                       # Ruling 118, M2
    e5_row = {"median_log_error": 0.1, "lo": 0.05, "hi": 0.15, "failures": 12.0, "failures_lo": 9.0, "failures_hi": 15.0,
              "lo_independent": 0.03, "hi_independent": 0.17, "failures_lo_independent": 7.0, "failures_hi_independent": 17.0,
              "n_nodes": 33, "n_nan_replicates": 0, "n_nan_replicates_failures": 0, "n_nan_replicates_independent": 0,
              "n_nan_replicates_failures_independent": 0, "scheme": "aligned", "interval_registered": INTERVAL_LEVEL,
              "approximate": True}                                                                           # an E5 row, as written
    real_e5_row = ss._e5_row
    monkeypatch.setattr(ss, "_need", lambda sc, key: key)                                       # a "score set" is its key
    monkeypatch.setattr(ss, "_contrast", lambda alt, base, alpha, n_rep: by_block[alt[1]])
    monkeypatch.setattr(ss, "_pc_contrast", lambda sc, months, n_rep: contrast(0.004, 0.2, (-0.01, 0.02), (-0.011, 0.019)))
    monkeypatch.setattr(ss, "_level", lambda scores, alpha, n_rep, measure="median": level)
    assert set(real_e5_row({}, "LEV6", 2000)) == {"block", *e5_row}                  # the hand E5 row has the writer's keys
    monkeypatch.setattr(ss, "_e5_row", lambda sc, block, n_rep: {"block": block, **e5_row})
    family = {"H-E": {"confirmatory": True, "reasons": [], "approximate": False, "interim_approximate": False},
              "H-B1": {"confirmatory": True, "reasons": [], "approximate": False},
              "H-B2": {"confirmatory": False, "reasons": ["simulated size 0.127 > 1.5 x 0.0167 (38 of 300 replications)"],
                       "approximate": True}}
    confirmatory = [h for h in ss.FAMILY_ORDER if family[h]["confirmatory"]]              # what run_sealed derives
    he = {"t_obs": 2.1, "p": 0.02, "n_null": 4159, "selc_obs": 0.004, "retained_share": [{"min": 0.95, "mean": 0.98}]}     # rotation_test's keys

    def run_one(test_id, prior):
        out = tmp_path / f"run_{test_id}"
        (out / "tables").mkdir(parents=True)
        cfg = SimpleNamespace(rolling=SimpleNamespace(deploy=SimpleNamespace(months={7, 8, 9})))
        results = {"test": test_id, "eta": 0.05, "family": confirmatory,
                   "coverage": {"ran": True, "usable_nodes": 33, "nonfinite_dropped": {}}, "rows_read": 1, "rows_kept": 1,
                   "started_utc": "2026-10-28T01:02:03+00:00", "protocol_sha256": "ab" * 32}
        run = ss._Run(cfg, 2000, out, results, 0.05, family, confirmatory, prior)
        hb = run.guard("bootstraps", lambda: ss._stage_bootstraps(run, {}))
        entry, holm_out = ss._decide(he, hb, family, confirmatory, prior)
        results["H-E"] = entry
        if holm_out is not None:
            results["holm"] = holm_out
        results.update(hb)
        if test_id == "A":
            run.record("audit robustness", ValueError("boom"), secondary=True)
        results["errors"], results["status"] = dict(run.errors), run.status()
        return results

    res_a = run_one("A", None)
    res_b = run_one("B", res_a)
    for hid in ("H-E", "H-B1", "H-B2"):                         # the hand fixture has the writers' keys: the test's and the estimand's
        for test_id, res in (("A", res_a), ("B", res_b)):
            assert set(res[hid]) == set(sealed_results(test_id)[hid]), (test_id, hid)
    assert res_a["H-B1"]["interval_registered"] == sealed_results("A")["H-B1"]["interval_registered"] == ss.INTERVAL_HB_TEST
    assert res_b["H-B2"]["interval_registered"] == sealed_results("B")["H-B2"]["interval_registered"] == ss.INTERVAL_HB_ESTIMAND
    assert res_b["H-B2"]["replicated"] is None is sealed_results("B")["H-B2"]["replicated"]       # the null of an estimand in Test B
    assert set(res_a["PC"]["P2_season"]) == set(sealed_results("A")["PC"]["P2_season"])
    for test_id, res in (("A", res_a), ("B", res_b)):                      # H-S: the aligned interval is primary, the other beside it
        for est in ("xgb", "logit"):
            assert set(res["H-S"][est]) == set(sealed_results(test_id)["H-S"][est]), (test_id, est)
            assert res["H-S"][est]["scheme"] == "aligned" and res["H-S"][est]["approximate"] is True
            assert res["H-S"][est]["ci_independent"] == [7.0, 17.0] and res["H-S"][est]["ci"] == [9.0, 15.0]
    assert res_a["H-S"]["xgb"]["interval_registered"] == sealed_results("A")["H-S"]["xgb"]["interval_registered"] == ss.INTERVAL_E5_HS
    assert INTERVAL_LEVEL == ss.INTERVAL_E5_HS
    assert [row["block"] for row in res_a["E5"]] == list(ss.BLOCKS) and all(set(row) == {"block", *e5_row} for row in res_a["E5"])
    for test_id, res in (("A", res_a), ("B", res_b)):
        write(tmp_path, json.loads(json.dumps(res, default=ss._json_default)), test_id)
    t = {name: value(name, tmp_path) for name in CI_MACROS + P_MACROS + STATUS_MACROS}
    assert t["hbOneCITestA"] == t["hbOneCIPercentileTestB"] == "[0.012, 0.093]"             # the percentile interval, as written
    assert t["hbOneCIBasicTestA"] == "[0.015, 0.096]" and t["hbTwoCIBasicTestB"] == "[\\textminus{}0.039, 0.005]"     # and the basic one
    assert t["hbOnePTestA"] == t["hbOnePTestB"] == "0.011"
    assert t["hbTwoPTestA"] == t["hbTwoPTestB"] == ESTIMAND                                  # the writer's family list leaves H-B2 out
    assert (value("hbOneReplicated", tmp_path), value("hbTwoReplicated", tmp_path)) == ("yes", "n/a")      # H-B2's verdict is null in Test B
    assert (t["runStatusTestA"], t["nRunErrorsTestA"]) == ("completed with secondary errors", "1")
    assert (t["runStatusTestB"], t["nRunErrorsTestB"]) == ("completed", "0")
    assert value("hbOneStatusTestA", tmp_path).startswith("rejected in Test A at 0.05/2") and "estimand" in value("hbTwoStatusTestB", tmp_path)
