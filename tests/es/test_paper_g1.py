"""Gate G1 (controller Rulings 118 and 119): what the manuscript generators do with the findings of the methodology review.

M6: the four new E4 statistics (the comparators' own ESred and the paired contrast dESred|c = ESred(P1) - ESred(c)) are named
wherever the E4 statistics are named, with macros for every development analysis the text cites. M5: the audit-robustness
family of Benjamini-Hochberg tests is descriptive, so no table and no macro counts its rejections. M8: the numbers of the
supplementary size check come from the archived probe outputs (docs/reviews/g1_probes), never from a typed digit. M2 and D13:
the footnote of an approximate mark says where the intervals of E5 and H-S come from. M1, M5 and the knowledge statement (which was to be
public with an OSF registration until Ruling 133; see test_paper_g4.py) also bind the text of the manuscript, which the last tests read
(the real paper/ folder).
Since plan 2C task T2a (26 October 2026) the former sections/06_sealed.tex is esm/forward_tests.tex and the former
appendices/A_ledger.tex is esm/designation_rule_study.tex; the tests that read them read the new files with the same assertions, apart from the
references (a reference to the paper from the ESM is \\mainref{key}, one to the ESM from the paper \\esmref{key}; paper/crossref.tex)."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

NEW_STATS = {"E4:ESred|calendar@count": "esredCalCount", "E4:ESred|VOL@count": "esredVolCount",
             "E4:dESred|calendar@count": "desredCalCount", "E4:dESred|VOL@count": "desredVolCount"}
DEV_SUFFIX = {"AQ3": "AqThree", "AQ4": "AqFour", "AQ1": "AqOne", "AQ2": "AqTwo", "2024-2025": "YY", "24H2-25H2": "HTwo",
              "25H1-26H1": "PostOpen"}
PAPER = Path(__file__).resolve().parents[3] / "paper"
needs_tree = pytest.mark.skipif(not (PAPER / "sections").exists(), reason="the manuscript folder is not next to dart_dev")
DOCS = PAPER.parent / "docs"
STAGE_A = "\\PENDINGA"
RAGGED = "\\raggedright\\hyphenpenalty=10000 "


def _resolve(name, src):
    from es.paper_macros import resolve
    from es.paper_registry import entries_by_name
    return resolve(entries_by_name()[name], src)


def _sources(tmp_path, rows=(), probes=None):
    """Sources over tmp_path/out with a bootstrap_ci.csv of `rows` (analysis, stat, estimate, lo, hi) and, when `probes` is a
    {file name: text}, a docs/reviews/g1_probes folder holding them."""
    from es.paper_macros import Sources
    out = tmp_path / "out"
    (out / "economics").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"analysis": a, "stat": s, "estimate": e, "lo": lo, "hi": hi, "approximate": True, "n_windows": 1}
                  for a, s, e, lo, hi in rows]).to_csv(out / "economics" / "bootstrap_ci.csv", index=False)
    docs = tmp_path / "docs"
    (docs / "sealed").mkdir(parents=True, exist_ok=True)
    if probes is not None:
        folder = docs / "reviews" / "g1_probes"
        folder.mkdir(parents=True, exist_ok=True)
        for name, text in probes.items():
            (folder / name).write_text(text)
    return Sources(out, tmp_path / "paper", docs / "sealed")


# ---------------------------------------------------------------------------------------------------- M6: the registry
def test_the_four_esred_statistics_are_registered_next_to_the_dselc_contrasts():
    from es.paper_registry import STAT_FMT, STAT_LABEL, STAT_ORDER, STAT_WORD
    assert {s: STAT_WORD[s] for s in NEW_STATS} == NEW_STATS
    for stat in NEW_STATS:
        assert STAT_FMT[stat] == ("usd0", "ciusd0"), stat                       # dollars, as E3's ESred
    # the labels are LaTeX: the class cannot typeset U+2212 (see paper_registry)
    assert STAT_LABEL["E4:ESred|calendar@count"] == "E4: ESred of the calendar-only model, matched count"
    assert STAT_LABEL["E4:ESred|VOL@count"] == "E4: ESred of VOL, matched count"
    assert STAT_LABEL["E4:dESred|calendar@count"] == "E4: ESred(P1) \\textminus{} ESred(calendar-only), matched count"
    assert STAT_LABEL["E4:dESred|VOL@count"] == "E4: ESred(P1) \\textminus{} ESred(VOL), matched count"
    assert not any("\u2212" in label for label in STAT_LABEL.values())
    # levels first, then the contrasts, as for SELC; the first nine rows keep their places
    assert STAT_ORDER[:9] == ("SELC", "NET", "E2:SNET", "E3:ESred", "E4:SELC_decile", "E4:SELC|calendar@count", "E4:SELC|VOL@count",
                              "E4:dSELC|calendar@count", "E4:dSELC|VOL@count")
    assert STAT_ORDER[9:] == ("E4:ESred|calendar@count", "E4:ESred|VOL@count", "E4:dESred|calendar@count", "E4:dESred|VOL@count")


def test_every_development_analysis_the_text_cites_has_macros_for_the_new_statistics():
    from es.paper_registry import entries_by_name
    by = entries_by_name()
    for stat, word in NEW_STATS.items():
        for aid, suffix in DEV_SUFFIX.items():
            for name in (f"{word}{suffix}", f"{word}{suffix}CI"):
                assert name in by, name
                assert by[name].stage == "dev" and by[name].source == "economics/bootstrap_ci.csv", name
            assert by[f"{word}{suffix}"].query == f'analysis == "{aid}" and stat == "{stat}"'
        for name in (f"{word}PooledRehearsal", f"{word}PooledRehearsalCI"):
            assert by[name].stage == "dev" and 'analysis.str.startswith("AQ3+AQ4")' in by[name].query and stat in by[name].query, name
        for test, stage in (("TestA", "sealedA"), ("TestB", "sealedB")):         # the sealed side may stay pending until the run
            assert by[f"{word}{test}"].stage == stage and by[f"{word}{test}CI"].stage == stage


def test_the_esred_macros_print_dollars_with_a_real_minus_and_an_interval(tmp_path):
    rows = [("AQ4", "E4:dESred|calendar@count", -482.4309779999985, -1136.9497339999998, -35.921580000001995),
            ("AQ3", "E4:dESred|calendar@count", -2537.9407675000007, -9934.921015274991, 2328.082265999994),
            ("AQ4", "E4:ESred|VOL@count", 13.904000000000451, -51.92884049999977, 42.69441234999826),
            ("AQ3+AQ4 (rehearsal of the pooled estimands)", "E4:dESred|VOL@count", -913.79, -6376.26, 2484.66)]
    src = _sources(tmp_path, rows)
    assert _resolve("desredCalCountAqFour", src)["value"] == "\\textminus{}\\$482"
    assert _resolve("desredCalCountAqFourCI", src)["value"] == "[\\textminus{}\\$1{,}137, \\textminus{}\\$36]"
    assert _resolve("desredCalCountAqThreeCI", src)["value"] == "[\\textminus{}\\$9{,}935, \\$2{,}328]"
    assert _resolve("esredVolCountAqFour", src)["value"] == "\\$14"
    assert _resolve("desredVolCountPooledRehearsal", src)["value"] == "\\textminus{}\\$914"
    assert _resolve("desredVolCountPooledRehearsalCI", src)["value"] == "[\\textminus{}\\$6{,}376, \\$2{,}485]"


def test_the_sealed_side_of_the_new_statistics_is_pending_until_the_test_has_run(tmp_path):
    src = _sources(tmp_path)
    for word in NEW_STATS.values():
        r = _resolve(f"{word}TestA", src)
        assert r["status"] == "pending" and r["value"] == STAGE_A, (word, r)


# ---------------------------------------------------------------------------------------------------- M8: the size check
HE_COMMON = """\
S0 independent nodes, phi .8             n=500  single A: 0.0080 / 0.0480   pooled A+B: 0.0120 / 0.0440   (49s)
S1 common factor .8, phi .8              n=500  single A: 0.0140 / 0.0460   pooled A+B: 0.0160 / 0.0360   (50s)
S2 S1 + market-wide spike days           n=500  single A: 0.0140 / 0.0480   pooled A+B: 0.0260 / 0.0580   (52s)
S3 common .8, phi .95 both               n=500  single A: 0.0140 / 0.0340   pooled A+B: 0.0080 / 0.0320   (48s)
S4 common .8, phi .95 + spikes           n=500  single A: 0.0080 / 0.0320   pooled A+B: 0.0120 / 0.0320   (49s)
"""
HE_TREND = """\
S5 common .8, phi .8, trend sd 1.5       n=500  single A: 0.0120 / 0.0540   pooled A+B: 0.0180 / 0.0620   (46s)
S6 common .8, phi .8, trend sd 3         n=500  single A: 0.0160 / 0.0420   pooled A+B: 0.0220 / 0.0680   (46s)
"""
PROBES = {"he_common_500.txt": HE_COMMON, "he_trend_500.txt": HE_TREND}
SIZE_CHECK = ("sizeCheckPooledThird", "sizeCheckPooledFive", "sizeCheckSingleThird", "sizeCheckSingleFive", "sizeCheckReps",
              "sizeCheckPhi")


def test_the_size_check_macros_are_the_worst_rates_of_the_archived_probe_outputs(tmp_path):
    src = _sources(tmp_path, probes=PROBES)
    got = {n: _resolve(n, src) for n in SIZE_CHECK}
    assert all(r["status"] == "ok" and r["stage"] == "dev" for r in got.values()), got
    assert {n: r["value"] for n, r in got.items()} == {
        "sizeCheckPooledThird": "0.026", "sizeCheckPooledFive": "0.068",               # the worst scenario of the pooled test, S2 and S6
        "sizeCheckSingleThird": "0.016", "sizeCheckSingleFive": "0.054",               # the worst scenario of one window, S6 and S5
        "sizeCheckReps": "500", "sizeCheckPhi": "0.95"}                                # replications per scenario, the largest persistence
    assert all(r["sha"] for r in got.values())                                         # provenance names the files that were read


def test_the_size_check_macros_follow_their_files_and_wait_for_them(tmp_path):
    changed = {"he_common_500.txt": HE_COMMON.replace("0.0260 / 0.0580", "0.0300 / 0.0580"), "he_trend_500.txt": HE_TREND}
    assert _resolve("sizeCheckPooledThird", _sources(tmp_path / "a", probes=changed))["value"] == "0.030"
    r = _resolve("sizeCheckPooledThird", _sources(tmp_path / "b"))               # no docs/reviews folder at all
    assert r["status"] == "pending" and r["value"] == "\\PENDINGDEV" and "g1_probes" in r["detail"]
    r = _resolve("sizeCheckPooledThird", _sources(tmp_path / "c", probes={"he_common_500.txt": HE_COMMON}))     # one of the two files
    assert r["status"] == "pending"


def test_a_probe_file_that_is_not_what_the_macros_read_is_an_error_never_a_guess(tmp_path):
    for bad in ("", "S1 something else entirely\n", HE_COMMON.replace("n=500", "n=300", 1)):      # empty, no match, replications differ
        src = _sources(tmp_path / str(abs(hash(bad))), probes={"he_common_500.txt": bad, "he_trend_500.txt": HE_TREND})
        with pytest.raises(LookupError, match="sizeCheck"):
            _resolve("sizeCheckPooledFive", src)


@needs_tree
def test_the_real_probe_files_give_the_values_the_review_reported():
    """The review (M8) states at most 0.026 at 0.05/3 and 0.068 at 0.05 for the pooled test, from these files."""
    from es.paper_macros import Sources
    src = Sources(PAPER.parent / "dart_dev" / "outputs" / "es", PAPER, DOCS / "sealed")
    assert [_resolve(n, src)["value"] for n in ("sizeCheckPooledThird", "sizeCheckPooledFive")] == ["0.026", "0.068"]


def test_no_scenario_of_the_size_check_is_significantly_above_nominal_on_the_rules_binomial_check():
    """The text says that none of the rates is significantly above nominal; this is the check that says so (a one-sided binomial
    test against the nominal level at the rule's 0.05 divided by the four rows it reads for H-E)."""
    import math
    rule = {"third": 0.05 / 3, "five": 0.05}
    worst = {"third": (13, 500), "five": (34, 500)}                                 # 0.026 and 0.068 of 500 replications
    for key, (k, n) in worst.items():
        tail = sum(math.comb(n, j) * rule[key] ** j * (1 - rule[key]) ** (n - j) for j in range(k, n + 1))
        assert tail > 0.05 / 4, (key, tail)


# ---------------------------------------------------------------------------------------------------- M5: the audit family
def test_no_macro_counts_the_rejections_of_the_audit_robustness_family():
    from es.paper_registry import entries_by_name
    names = set(entries_by_name())
    assert not [n for n in names if n.startswith("bhRejectionsAuditRobustness")]
    assert {"bhTestsAuditRobustness", "bhTestsAuditRobustnessTestA", "bhTestsAuditRobustnessTestB"} <= names          # the tests are counted
    assert {"bhRejectionsEconomicRobustness", "bhRejectionsDriftMonitors", "bhRejectionsDriftMonitorsTestA"} <= names


def _bh(tmp_path, rows):
    from es.paper_macros import Sources
    out = tmp_path / "out"
    (out / "inference").mkdir(parents=True)
    pd.DataFrame(rows).to_csv(out / "inference" / "bh.csv", index=False)
    return Sources(out, tmp_path / "paper", tmp_path / "docs")


BH_ROWS = [{"family": "audit robustness", "test": "H-B1 logit", "estimate": 0.4, "p": 0.004, "bh_reject": True, "included": True},
           {"family": "audit robustness", "test": "H-B1 seed=0", "estimate": 0.1, "p": 0.3, "bh_reject": False, "included": True},
           {"family": "audit robustness", "test": "H-B2 frozen", "estimate": float("nan"), "p": float("nan"), "bh_reject": False,
            "included": False},
           {"family": "economic robustness", "test": "P1 dec fees=0.1", "estimate": 0.12, "p": 0.02, "bh_reject": True, "included": True},
           {"family": "economic robustness", "test": "P1 inc fees=0.1", "estimate": 0.02, "p": 0.6, "bh_reject": False, "included": True},
           {"family": "drift monitors", "test": "LEV6 W1_norm", "estimate": 0.2, "p": 0.6, "bh_reject": False, "included": True},
           {"family": "drift monitors", "test": "LEV6 early_rate", "estimate": 0.5, "p": 0.001, "bh_reject": True, "included": True}]


def test_the_bh_summary_table_counts_the_tests_of_the_audit_family_and_not_its_rejections(tmp_path):
    from es.paper_tables import BUILDERS
    src = _bh(tmp_path, BH_ROWS)
    tex = BUILDERS["tab_bh_families"][0](src)
    audit = next(line for line in tex.splitlines() if line.startswith("audit robustness"))
    assert audit.startswith("audit robustness & 2 & ") and audit.count("--") == 2          # the tests that ran; no count, no smallest p
    assert not re.search(r"audit robustness & 2 & [0-9]", audit)
    assert "economic robustness & 2 & 1 & 0.020" in tex and "drift monitors & 2 & 1 & 0.001" in tex       # the other families as before
    assert "descriptive" in tex and "diagnostic" in tex and "no count of its rejections" in tex          # the note says why


def test_the_ledger_marks_every_flag_of_the_audit_family_as_a_diagnostic_and_no_other(tmp_path):
    from es.paper_tables import BUILDERS
    tex = BUILDERS["esm_ledger_bh"][0](_bh(tmp_path, BH_ROWS))
    rows = {line.split(" & ")[1].replace(RAGGED, ""): line for line in tex.splitlines() if " & " in line and line.endswith("\\\\")}
    mark = "\\textsuperscript{b}"
    for test in ("H-B1 logit", "H-B1 seed=0"):
        assert rows[test].rstrip(" \\\\").endswith(mark), rows[test]               # yes/no, then the mark of the note
    assert mark not in rows["P1 dec fees=0.1"] and mark not in rows["LEV6 early\\_rate"]
    assert "H-B2 frozen" in rows and mark not in rows["H-B2 frozen"]               # a test that did not run has no flag to mark
    assert re.search(r"\{?b\}? Diagnostic only", tex) or "Diagnostic only" in tex
    assert "no count of its rejections" in tex


def _sealed(tmp_path):
    import json
    from es.paper_macros import Sources
    out = tmp_path / "out"
    d = out / "sealed" / "A" / "tables"
    d.mkdir(parents=True)
    (out / "sealed" / "A" / "results.json").write_text(json.dumps({
        "test": "A", "errors": {}, "coverage": {"ran": True, "usable_nodes": 31}, "family": ["H-E"],
        "secondary": {f: {"tests": 2, "rows": 2, "bh_rejections": 1} for f in ("audit robustness", "economic robustness", "drift monitors")}}))
    pd.DataFrame([{"family": f, "test": f"{f} {i}", "estimate": 0.1, "p": p, "bh_reject": p < 0.01, "included": True}
                  for f in ("audit robustness", "economic robustness", "drift monitors") for i, p in enumerate((0.005, 0.3), 1)]
                 ).to_csv(d / "secondary_bh.csv", index=False)
    return Sources(out, tmp_path / "paper", tmp_path / "docs")


def test_the_sealed_bh_table_does_not_count_the_audit_family_either(tmp_path):
    import es.paper_tables_sealed  # noqa: F401
    from es.paper_tables import BUILDERS
    tex = BUILDERS["tab_sealed_bh_A"][0](_sealed(tmp_path))
    assert "audit robustness & 2 & -- & --" in tex
    assert "economic robustness & 2 & 1 & 0.005" in tex and "drift monitors & 2 & 1 & 0.005" in tex
    assert "no count of its rejections" in tex


def test_the_sealed_bh_table_still_checks_the_audit_counts_against_the_run(tmp_path):
    import json
    import es.paper_tables_sealed  # noqa: F401
    from es.paper_tables import BUILDERS
    src = _sealed(tmp_path)
    path = src.out_dir / "sealed" / "A" / "results.json"
    r = json.loads(path.read_text())
    r["secondary"]["audit robustness"]["bh_rejections"] = 2                            # the run says 2; the file has 1
    path.write_text(json.dumps(r))
    src._json.clear()
    with pytest.raises(LookupError, match="audit robustness"):
        BUILDERS["tab_sealed_bh_A"][0](src)


def test_the_digest_and_the_paper_agree_on_which_bh_family_is_descriptive():
    """The digest (an OSF supplement) prints no rejection count for the family that the paper calls descriptive (Ruling 118, M5); it
    imports nothing heavy, so its tuple is typed there and compared here."""
    from es import run_digest
    from es.paper_registry import DESCRIPTIVE_BH_FAMILIES
    assert run_digest.DESCRIPTIVE_BH_FAMILIES == DESCRIPTIVE_BH_FAMILIES == ("audit robustness",)


# ---------------------------------------------------------------------------------------------------- M2, D13: the aligned intervals
def _e5_rows(**more):
    base = dict(median_log_error=0.1, lo=0.05, hi=0.2, failures=3, failures_lo=1, failures_hi=6, n_nodes=31, n_nan_replicates=0,
                n_nan_replicates_failures=0, lo_independent=0.07, hi_independent=1.1, failures_lo_independent=0, failures_hi_independent=30,
                scheme="aligned", interval_registered="percentile, calendar-aligned (primary); independent draws (diagnostic)",
                approximate=True)
    base.update(more)
    blocks = ("G6", "LEV6", "MID3", "DART3", "MIDDART6", "ROLL15", "ROLL_DISP9", "ROLL_LOC6", "DART18", "G1_24")
    return pd.DataFrame([{"block": b, **base} for b in blocks])


def _e5_sources(tmp_path, **more):
    import json
    from es.paper_macros import Sources
    out = tmp_path / "out"
    d = out / "sealed" / "A" / "tables"
    d.mkdir(parents=True)
    hs = {"failures": 2, "ci": [0, 5], "ci_independent": [0, 31], "scheme": "aligned", "n_nodes": 31, "n_nan_replicates": 0,
          "n_nan_replicates_independent": 0, "approximate": True}
    pc = {"median_diff": 0.2, "ci": [0.1, 0.3], "n_nodes": 31, "n_nan_replicates": 0, "ci_approximate": True}
    (out / "sealed" / "A" / "results.json").write_text(json.dumps({
        "test": "A", "errors": {}, "coverage": {"ran": True, "usable_nodes": 31}, "family": ["H-E"],
        "PC": {"P2_season": pc, "P2_all": pc}, "H-S": {"xgb": hs, "logit": hs}}))
    _e5_rows(**more).to_csv(d / "E5_blocks.csv", index=False)
    return Sources(out, tmp_path / "paper", tmp_path / "docs")


def test_the_approximate_footnote_of_e5_and_h_s_says_where_the_intervals_come_from(tmp_path):
    import es.paper_tables_sealed  # noqa: F401
    from es.paper_tables import APPROX_TEXT, BUILDERS
    src = _e5_sources(tmp_path)
    for name in ("tab_sealed_e5_A", "tab_sealed_pc_hs_A"):
        tex = BUILDERS[name][0](src)
        note = re.search(r"\\footnotetext\{\\textsuperscript\{a\} (.*?)\}\n", tex, flags=re.S)
        assert note, (name, tex)
        assert APPROX_TEXT in note.group(1)                                              # the old reason stays
        assert "calendar-aligned" in note.group(1) and "coverage" in note.group(1) and "not studied" in note.group(1), note.group(1)


def test_the_aligned_note_is_not_added_to_a_table_without_e5_or_h_s_rows(tmp_path):
    import es.paper_tables_sealed  # noqa: F401
    from es.paper_registry import STAT_ORDER
    from es.paper_tables import APPROX_NOTE, BUILDERS
    src = _e5_sources(tmp_path)
    pd.DataFrame([{"stat": s, "estimate": 1.0, "lo": 0.5, "hi": 1.5, "approximate": "ESred" in s, "n_windows": 1} for s in STAT_ORDER]
                 ).to_csv(src.out_dir / "sealed" / "A" / "tables" / "estimands.csv", index=False)
    tex = BUILDERS["tab_sealed_e2e4_A"][0](src)
    assert APPROX_NOTE in tex and "calendar-aligned" not in tex                              # E2 to E4 keep the plain reason


def test_the_e5_table_builds_with_the_new_flat_columns_and_does_not_show_them(tmp_path):
    import es.paper_tables_sealed  # noqa: F401
    from es.paper_tables import BUILDERS
    tex = BUILDERS["tab_sealed_e5_A"][0](_e5_sources(tmp_path))
    assert "0.10 [0.05, 0.20]\\textsuperscript{a}" in tex and "3 [1.0, 6.0]\\textsuperscript{a}" in tex
    assert "1.10" not in tex and "30.0" not in tex and "independent" not in tex.split("\\footnotetext", 1)[0]    # the diagnostics are not columns


def test_an_e5_interval_that_misses_its_estimate_is_still_explained_without_the_old_mechanism(tmp_path):
    import es.paper_tables_sealed  # noqa: F401
    from es.paper_tables import BUILDERS
    tex = BUILDERS["tab_sealed_e5_A"][0](_e5_sources(tmp_path, lo=0.12, hi=0.2))
    assert "does not contain its estimate" in tex
    assert "pulled toward" not in tex and "median" not in tex.split("does not contain its estimate", 1)[1].split("}", 1)[0]


def test_the_h_b_interval_note_names_the_calendar_composition_as_the_main_source_of_the_shift():
    from es.paper_tables_sealed import _interval_note
    note = _interval_note(["H-B1", "H-B2"], ["H-B1 [0.100, 0.200]", "H-B2 [0.100, 0.300]"], True, False)
    assert "re-randomiz" in note and "calendar" in note
    assert "inherit the over-rejection mechanisms" not in note                      # the earlier explanation was incomplete (M7)
    assert "may under-cover" in note


# ---------------------------------------------------------------------------------------------------- M1: the registration table
def test_the_registration_table_says_that_the_first_holm_level_is_the_drafts_and_that_h_e_is_decided_at_the_family_level(tmp_path):
    """Ruling 118, M1: the table reads every test at 0.05/3 and at 0.05, as the rule does; a reader must not take 0.05/3 for H-E's level."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import test_paper_tables as tt
    from es.paper_tables import BUILDERS
    src, _ = tt._sources(tmp_path)
    tt._write_registration(src)
    tex = BUILDERS["tab_registration"][0](src)
    note = re.search(r"\\footnotetext\{(The level 0\.05/3 is[^}]*)\}", tex)
    assert note, tex[-1500:]
    assert "first Holm level of the three-member draft family" in note.group(1)
    assert "confirmatory family is H-E alone, decided at 0.05" in note.group(1) and "governs H-E" in note.group(1)
    assert "registered" not in note.group(1)                                                     # Ruling 133: the words of the printed text


# ---------------------------------------------------------------------------------------------------- the manuscript
def _text(*names):
    return "\n".join((PAPER / n).read_text(encoding="utf-8") for n in names)


def _live(text):
    """The text without its LaTeX comments (a comment line is not in the paper, but it must not carry the old alternative either)."""
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in text.splitlines())


SECTIONS = sorted(p.name for p in (PAPER / "sections").glob("*.tex")) if (PAPER / "sections").exists() else []


@needs_tree
def test_no_section_counts_the_rejections_of_the_audit_robustness_family():
    for name in SECTIONS:
        text = _live(_text(f"sections/{name}"))
        assert "bhRejectionsAuditRobustness" not in text, name
    for name in ("esm/forward_tests.tex", "esm/designation_rule_study.tex", "esm/ESM_1.tex",         # the former Section 6, the former Appendix A, the ESM,
                 "esm/data_design_details.tex",                                                       # the details moved from Section 3 (plan 2C task T2b: the secondary families)
                 "esm/audit_economic_details.tex"):                                                   # and the text of Sections 4 and 5 that was cut (plan 2C task T2c: the target and band)
        assert "bhRejectionsAuditRobustness" not in _live(_text(name)), name


@needs_tree
def test_no_commented_embargo_alternative_remains_and_no_file_that_had_one_says_that_a_registration_is_public():
    """Ruling 118 queue, item 8 removed the commented alternatives for an embargoed OSF registration; Ruling 133 removed the registration, so
    the public wording ("the registration is public") that stayed then is gone too, from the three files and from their notes."""
    for name in ("esm/forward_tests.tex", "sections/90_backmatter.tex", "cover_letter.md"):          # esm/forward_tests.tex was sections/06_sealed.tex
        text = _text(name)
        assert not re.search(r"embargo|option \(b\)|view-only", text, flags=re.I), name
        printed = " ".join(_live(text).split())                                                 # a comment line may still remember the old wording
        assert "registration is public" not in printed and "public registration" not in printed, name


@needs_tree
def test_the_pre_specified_family_is_h_e_alone_one_sided_at_the_family_level_on_the_pooled_window():
    text = _live(_text("esm/forward_tests.tex"))                                                 # was sections/06_sealed.tex: the section moved to the ESM
    flat = " ".join(text.split())
    assert re.search(r"(?:pre-specified|confirmatory) family (?:is \{?H-E\}? alone|has one member)", flat), "the family is not said to be H-E alone"
    assert "registered family" not in flat                                                       # Ruling 133: pre-specified, not registered
    assert "\\famAlpha" in flat and "pooled" in flat
    assert "step-down" not in flat.replace("three-member step-down", "")                      # the step-down appears only to be set aside
    for bad in ("0.05/3", "\\famAlpha{}/3", "one third of \\famAlpha{} in Test A"):
        assert bad not in flat, bad

@needs_tree
def test_the_knowledge_statement_summary_follows_the_template_and_no_longer_points_to_a_public_registration():
    """Ruling 118 (M3, M4): items 2 and 6 of the facts the analysis team states are the protocol's statement, summarised; and the author's own
    answers (item 1) are left to the author. Ruling 133: the statement is part of the protocol, which is held by the author and not registered,
    so the summary no longer says that its full text is public with a registration."""
    raw = _text("esm/forward_tests.tex")                                                          # was sections/06_sealed.tex: the section moved to the ESM
    text = " ".join(_live(raw).split())
    assert "may not be able to open" not in text
    assert "The protocol contains a knowledge statement. In it the author records" in text
    assert "public registration" not in text and "\\osfDOI" not in text
    assert "five documents" not in text and "None of the five" not in text and "probably also the 2025 annual report" not in text
    assert "seven documents dated, or now served, after 2026-06-30" in text
    assert "was created on 2026-06-25, is compliant and is used" in text                          # the CAISO report
    assert "\\$14--22 per MWh and \\$17 per MWh" in text and "read them for their dates (for one, also its first page); no number from them" in text
    assert "the second and third drafts and every decision logged during the analysis came after it" in text
    assert "It could not have informed them" in text and "every hypothesis status is computed by the designation rule" in text
    assert "% AUTHOR: once you have filled in the template" not in raw                            # the author answered item 1 on 25 October: nothing, in all four rows
    assert "records that the author knows nothing about market conditions after June 2026" in text
    assert "the category ``nothing'' is ticked for each of four kinds of knowledge" in text and "the detail field says ``none''" in text
    assert "% AUTHOR: confirm this sentence when you sign the template" not in raw
    # G3, item 12: a referee tests "nothing" against this passage, so every term resolves for a reader and the facts are Part H's
    assert "the analysis team, by which we mean the author and the AI assistants of \\mainref{data-impl}, states six facts" in text
    assert "that a research agent saw in ISO-NE's October 2026 virtual-bid financial-assurance proxy file" in text
    assert "anyone on the team" not in text                                                       # it left open whether the author saw the proxies
    assert "In the re-derivation a research agent opened eleven files, three of them among the seven" in text
    assert "the note on virtual-bidding fees, written during the fee research, drew on seven documents" in text
    assert "behind the first draft of the design;" in text
    for old in ("ruling", "Ruling", "fee note", "design specification", "first version", "second and third versions"):
        assert old not in text, old                                                               # terms of the working files, not of a reader
