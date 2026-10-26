"""es/run_digest.py: the Markdown digest of the development outputs (Task 16).

Every test builds its inputs from small fixture files in tmp_path, in the column layout of the real outputs (taken from
the producers: run_legacy, run_analogs and simulate, run_economics, run_monitors, run_labels_pr, run_inference and
secondary), and points the module at them. Nothing here reads a price or writes the real outputs/: the digest reads
output files and two code constants only, and `sandbox` fails any write under the real outputs/ or cache/."""
import ast
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

DEV_ROOT = Path(__file__).resolve().parents[2]
NAN = float("nan")


# ---------------------------------------------------------------------------------------------------------------
# Reading a digest back: sections, sub-sections and Markdown tables
# ---------------------------------------------------------------------------------------------------------------

def _cells(line):
    return [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", line.strip())[1:-1]]


def tables(text):
    """Every Markdown pipe table in ``text`` as a list of rows ({header: cell}); an escaped pipe is a pipe in a cell."""
    found, lines, i = [], text.splitlines(), 0
    while i < len(lines):
        if lines[i].startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|$", lines[i + 1].strip()):
            header, rows = _cells(lines[i]), []
            i += 2
            while i < len(lines) and lines[i].startswith("|"):
                cells = _cells(lines[i])
                assert len(cells) == len(header), f"a row with {len(cells)} cells under {len(header)} headers: {lines[i]}"
                rows.append(dict(zip(header, cells)))
                i += 1
            found.append(rows)
        else:
            i += 1
    return found


def table_with(text, column):
    for rows in tables(text):
        if rows and column in rows[0]:
            return rows
    raise AssertionError(f"no table with a column {column!r} in:\n{text[:3000]}")


def section(text, prefix):
    m = re.search(r"^## " + re.escape(prefix) + r"[^\n]*\n(.*?)(?=^## |\Z)", text, re.S | re.M)
    assert m, f"no section '## {prefix}' in:\n{text[:1500]}"
    return m.group(1)


def sub(text, prefix):
    m = re.search(r"^#{3,6} " + re.escape(prefix) + r"[^\n]*\n(.*?)(?=^#{1,6} |\Z)", text, re.S | re.M)
    assert m, f"no sub-section '{prefix}' in:\n{text[:3000]}"
    return m.group(1)


# ---------------------------------------------------------------------------------------------------------------
# Fixture outputs, in the layout of the real ones
# ---------------------------------------------------------------------------------------------------------------

def put(out, rel, content):
    path = out / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, pd.DataFrame):
        content.to_csv(path, index=False)
    elif isinstance(content, (dict, list)):
        path.write_text(json.dumps(content, indent=2))
    else:
        path.write_text(content)
    return path


def comparison():
    cols = ["item", "analysis", "block", "estimator", "metric", "value", "source_line", "decimals", "note",
            "stated_analysis", "regenerated", "as_stated_value", "abs_diff", "status", "match"]
    rows = [
        ["T1_panel", "panel", "NYISO", "-", "n_nodes", 10.0, "MR:36", 0, NAN, NAN, 10.0, NAN, 0.0, "match", True],
        ["T1_panel", "panel", "NYISO", "-", "rows", 329000.0, "MR:36", -3, "D6: printed in thousands | see D6", NAN,
         306390.0, NAN, 22610.0, "mismatch", False],
        ["S42_calendar_auprc", "heldout-2024", "G6", "xgb", "median_auprc", 0.075, "MR:201", 3, "D3: no counterpart", NAN,
         NAN, NAN, NAN, "no_counterpart", False],
        ["T4_flip", "24H2-25H2-train2023", "LEV6", "xgb", "median_log_error", 0.23, "MR:275", 2, "D1: other fit",
         "24H2-25H2", 0.228497, 0.319702, 0.001503, "match_other_design", False],
    ]
    return pd.DataFrame(rows, columns=cols)


CHANGES = """# Regenerating the 31 Jul 2026 paper: what changed (Task 10)

## 1. Headline

Some text.

## 2. The known corrections, one by one

### D1. The 24H2->25H2 row mixes two model fits
The published row is reproduced only by combining two analyses.

### D6. Table 1
Node counts reproduce.

### Not a numbered correction
Text.
"""


def family(h_b1_power=0.41):
    return {"H-E": {"confirmatory": True, "reasons": [], "approximate": False, "interim_approximate": True},
            "H-B1": {"confirmatory": h_b1_power >= 0.5,
                      "reasons": [] if h_b1_power >= 0.5 else [f"simulated power {h_b1_power:.2f} at 0.05/3 is below 0.5"],
                      "power": h_b1_power, "approximate": False},
            "H-B2": {"confirmatory": True, "reasons": [], "power": 0.62, "approximate": False},
            "notes": ["Pooled placebo check (2 pairings): H-E is replaced by its estimand when 1 or more reject at "
                      "0.0167. The check is retained as registered."]}


def protocol(complete=True):
    if not complete:                           # what stage `analogs` writes
        return {"eta": 0.05, "family": None, "stage": "analogs", "placebo": [{"pair": "AQ3-1y", "t_obs": 1.0}]}
    return {"eta": 0.05, "family": family(), "stage": "complete", "analog_selc": [0.0121, -0.0046],
            "analog_selc_pooled": 0.0038, "rho_star": 0.5,
            "phi": {"DART3": {"phi_score_median": 0.61, "phi_score_raw_median": 0.58},
                    "LEV6": {"phi_score_median": 0.7, "phi_score_raw_median": None},
                    "ROLL_DISP9": {"phi_score_median": 0.9, "phi_score_raw_median": 0.88}},
            "rule": {"factor": 1.5, "binomial_level": 0.05, "approximate_factor": 1.0, "min_power": 0.5,
                     "levels": [0.016666666666666666, 0.05], "k_rows": {"H-E": 4, "H-B1": 12, "H-B2": 12},
                     "n_rep": 299, "n_sim": {"he_size": 300, "hb_power": 150},
                     "seeds": {"size_base": 20261022, "rho_search": 7, "he_power": 11, "hb_power": 13},
                     "calibration_grid": [0.05, 0.1], "counted_pairings": ["AQ3-1y", "AQ4-1y"]},
            "simulation": [{"test": "DECOY simulation row inside the json"}],
            "placebo": [{"pair": "DECOY placebo row inside the json"}]}


def eta_selection():
    rows = []
    for aid, (a, b) in {"AQ3": (0.4, 0.04), "AQ4": (0.3, 0.08), "AQ1": (0.2, 0.12), "AQ2": (0.5, 0.16),
                        "AQ3F": (0.9, 0.9)}.items():
        rows += [{"analysis": aid, "eta": 0.01, "median_log_error": a}, {"analysis": aid, "eta": 0.05, "median_log_error": b}]
    return pd.DataFrame(rows)


def placebo():
    return pd.DataFrame([
        {"pair": "AQ3-1y", "t_obs": 1.5, "p": 0.25, "n_null": 91, "selc_obs": 0.012, "counted": True, "note": NAN},
        {"pair": "AQ3-2y", "t_obs": -1.5, "p": 0.5, "n_null": 91, "selc_obs": -0.011, "counted": False,
         "note": "not counted: the 2-year pairing (Ruling 94)"}])


def activity():
    """AQ3 and AQ4 x G6 and LEV6 x xgb and logit x P1 and P6, three nodes each. xgb: node k fails when k < fails."""
    rows = []
    for aid, fails_xgb in (("AQ3", 2), ("AQ4", 0)):
        for block in ("G6", "LEV6"):
            for est in ("xgb", "logit"):
                for pol in ("P1", "P6"):
                    for k in range(3):
                        fail = k < (fails_xgb if est == "xgb" else 3)
                        rows.append({"market": "M", "node": f"N{k}", "n": 2000, "alerts": 100, "rate": 0.05, "R": 1.0,
                                     "zero_activity": False, "R_floored": 1.0, "log_error": 0.1 * (k + 1),
                                     "fail_0.5_2": fail, "analysis": aid, "block": block, "estimator": est,
                                     "policy": pol})
    return pd.DataFrame(rows)


def _sim_row(test, a3, p05, **kw):
    row = {"test": test, "law": NAN, "scenario": NAN, "persistence": NAN, "strata": NAN, "n_sim": 300,
           "reject_0.0166667": a3, "reject_0.05": p05}
    row.update(kw)
    return row


def simulation(b1_power=0.41):
    return pd.DataFrame([
        _sim_row("H-E single A, size", 0.02, 0.06, scenario="homoskedastic", persistence="fit", strata="primary"),
        _sim_row("H-E pooled A+B, size", 0.017, 0.049, scenario="heteroskedastic", persistence="upper", strata="primary"),
        _sim_row("H-E single A, size", 0.0555, 0.1111, scenario="homoskedastic", persistence="fit", strata="decile"),
        _sim_row("H-E pooled A+B, power curve", 0.02, 0.08, scenario="homoskedastic", persistence="fit", strata="primary",
                 n_sim=100, rho=0.05, mean_selc=0.0007),
        _sim_row("H-E pooled A+B, power curve", 0.7, 0.9, scenario="homoskedastic", persistence="fit", strata="primary",
                 n_sim=100, rho=0.9, mean_selc=0.0044),
        _sim_row("H-E pooled A+B, power", 0.55, 0.8, scenario="homoskedastic", persistence="fit", strata="primary",
                 rho=0.5, mean_selc=0.0021, target_selc=0.0019, note="target not bracketed"),
        _sim_row("H-B size", 0.015, 0.05, law="DART3", scenario="no_drift", persistence="fit",
                 **{"nodrift_fail_q95_0.5_2": 9.0, "nodrift_fail_q95_0.67_1.5": 14.0, "nodrift_fail_q95_0.8_1.25": 20.0}),
        _sim_row("H-B size", 0.03, 0.09, law="LEV6", scenario="common_drift_0.5", persistence="upper",
                 **{"nodrift_fail_q95_0.5_2": NAN, "nodrift_fail_q95_0.67_1.5": NAN, "nodrift_fail_q95_0.8_1.25": NAN}),
        _sim_row("H-B1 power", b1_power, 0.62, law="DART3", scenario="no_drift", persistence="fit", n_sim=150,
                 target_log_error_gap=0.04, calibrated_delta=0.3, mean_t_hat=0.039),
        _sim_row("H-B2 power", 0.62, 0.8, law="DART3", scenario="no_drift", persistence="fit", n_sim=150,
                 target_log_error_gap=0.05, calibrated_delta=0.35, mean_t_hat=0.051),
    ])


ECON_COLUMNS = ["LA", "PG", "NET", "SELC", "SELC_decile", "SNET", "ESred", "ES5_without", "CAP", "excess", "n_days",
                "alert_rate", "J_0.0", "J_0.25", "J_0.5", "J_1.0", "analysis", "policy", "comparator", "position", "fees",
                "settle_approx_share", "rotation_p"]


def _econ(analysis, policy, comparator, position, fees, selc, rotation_p=NAN):
    return {"LA": 1.0, "PG": 2.0, "NET": -58696.598, "SELC": selc, "SELC_decile": 0.0391, "SNET": -14.85, "ESred": 3210.9,
            "ES5_without": 5.0, "CAP": 0.2721, "excess": 0.1809, "n_days": 92, "alert_rate": 0.0912, "J_0.0": -58696.598,
            "J_0.25": 15154.27, "J_0.5": 89005.14, "J_1.0": 236706.9, "analysis": analysis, "policy": policy,
            "comparator": comparator, "position": position, "fees": fees, "settle_approx_share": 0.0,
            "rotation_p": rotation_p}


def summary():
    rows = [_econ("AQ3", "P1", "none", "premium", "calibrated", 0.011, 0.576),
            _econ("AQ3", "P4", "none", "premium", "calibrated", 0.012),
            _econ("AQ4", "P1", "none", "premium", "calibrated", 0.013, 0.663),
            _econ("AQ3", "comparator", "calendar@count", "premium", "calibrated", 0.014),
            _econ("AQ3", "P1", "none", "premium[long_spikes]", "calibrated", 0.015, 0.033),
            _econ("AQ3", "P1", "none", "premium[short_spikes]", "calibrated", 0.016, 0.664),
            # decoys: another fee setting, another position
            _econ("AQ3", "P1", "none", "premium", "0.25", 0.987, 0.5),
            _econ("AQ3", "P1", "none", "inc", "calibrated", 0.988, 0.5),
            _econ("AQ3", "P1", "none", "premium[long_spikes]", "0.25", 0.989, 0.5)]
    return pd.DataFrame(rows, columns=ECON_COLUMNS)


def bootstrap_ci():
    rows = [("AQ3", "SELC", 0.011, -0.03, 0.05, False, 1), ("AQ3", "NET", -58696.6, -100000.0, 4540.6, False, 1),
            ("AQ3", "E4:dSELC|calendar@count", 0.021, -0.01, 0.06, False, 1),
            ("AQ3|ex-CAISO", "E4:dSELC|VOL@count", 0.031, -0.02, 0.07, False, 1),
            ("AQ3", "E1:NET:timing", -6126.4, -16941.0, 3828.0, False, 1),
            ("AQ3", "E1:ESred:total", 1355.8, 100.0, 2000.0, True, 1),
            ("AQ3+AQ4 (rehearsal of the pooled estimands)", "SELC", 0.005, -0.02, 0.03, False, 2)]
    return pd.DataFrame(rows, columns=["analysis", "stat", "estimate", "lo", "hi", "approximate", "n_windows"])


def decomposition():
    return pd.DataFrame([{"analysis": "AQ3", "NET_timing": -6126.4, "NET_level": -2840.1, "NET_total": -8966.6},
                         {"analysis": "AQ4", "NET_timing": -8427.0, "NET_level": 1551.7, "NET_total": -6875.4}])


def rotation():
    return pd.DataFrame([{"window": "AQ3", "t_obs": -0.56, "p": 0.576, "n_null": 91, "selc_obs": -0.0063,
                          "retained_min": 1.0, "retained_mean": 1.0},
                         {"window": "AQ3+AQ4 (rehearsal of the pooled H-E test)", "t_obs": -0.93, "p": 0.635,
                          "n_null": 8463, "selc_obs": -0.0053, "retained_min": 0.99, "retained_mean": 0.999}])


def run_meta(status="complete"):
    meta = {"status": status, "analyses": ["AQ3", "AQ4", "AQ1"], "analyses_done": ["AQ3", "AQ4", "AQ1"], "n_rep": 2000,
            "seed": 20261022, "mean_block": 7, "eta": 0.05, "alpha": 0.05, "W": 56,
            "strata_primary": ["market", "node", "settle_hour", "wclass", "month"],
            "rotation_test_defaults": {"min_shift": 1}, "started_utc": "2026-10-23T09:51:10Z",
            "finished_utc": "2026-10-23T10:12:33Z"}
    if status == "running":
        meta.update(analyses_done=["AQ3"], finished_utc=None)
    if status == "failed":
        meta.update(analyses_done=["AQ3"], failed_on="AQ4", error="ValueError: boom")
    return meta


def volmanagement():
    return pd.DataFrame([{"analysis": "AQ3", "sharpe_alerted": 0.092, "sharpe_not_alerted": 0.019}])


def monitor_eval():
    cols = ["analysis", "block", "monitor", "spearman", "spearman_p_one_sided", "auroc_fail", "n_used", "calib_days",
            "placebo_median", "placebo_n"]
    rows = [["AQ3", "LEV6", "W1_norm", 0.31, 0.04, 0.7, 33, 92, 0.4, 33],
            ["AQ3", "LEV6", "early_rate", 0.2, 0.1, 0.6, 33, 92, NAN, NAN],
            ["EDGE100", "LEV6", "PSI", 0.1, 0.3, 0.5, 33, 100, 0.2, 33],
            ["EDGE101", "LEV6", "PSI", 0.1, 0.3, 0.5, 33, 101, 0.2, 33],
            ["2024-2025", "LEV6", "domain_auc", 0.5, 0.01, 0.8, 33, 366, 0.9, 33]]
    return pd.DataFrame(rows, columns=cols)


def label_2x2():
    rows = []
    for aid, fails in (("LC-full-q2023", (7, 7)), ("LC-full-q2324", (8, 7)), ("LC-h1-q2023", (1, 12)), ("LC-h1-q2324", (2, 13))):
        for block, f in zip(("LEV6", "DART18"), fails):
            rows.append({"analysis": aid, "block": block, "failures": f, "n_nodes": 33, "median_R": 0.97})
    return pd.DataFrame(rows)


def label_pr():
    rows = []
    for aid in ("LC-full-q2023", "LC-h1-q2023"):
        for k, (pq, rq) in enumerate(((0.1, 0.3), (0.2, 0.2), (0.6, 0.1))):
            rows.append({"market": "M", "node": f"N{k}", "precision_q": pq, "recall_q": rq, "precision_eval": pq + 0.01,
                         "recall_eval": rq + 0.01, "prevalence_eval": 0.05, "n_nan_outcome": k, "precision_lift": 2.0 + k,
                         "analysis": aid, "block": "LEV6"})
    return pd.DataFrame(rows)


SETTLEMENT = """# Settlement real-time legs: pull status (Task 6)

## 1. CAISO earliest available FMM date

**Earliest FMM date: 2023-07-13** (first full day returned by OASIS).

```
# a comment inside a fence stays a comment
```

### Bisection

| probe date | result |
|---|---|
| 2023-04-15 | no data |
"""

ADDENDUM = "# Addendum to settlement_status.md (23 Oct 2026, controller)\n\nTask 6 fix round 1 changed the module.\n"


def bh():
    rows = [{"family": "audit robustness", "test": "H-B1 logit", "estimate": 0.03, "p": 0.001, "bh_reject": True,
             "included": True},
            {"family": "audit robustness", "test": "H-B1 frozen", "estimate": 0.01, "p": 0.4, "bh_reject": False,
             "included": True},
            {"family": "audit robustness", "test": "H-B2 logit", "estimate": NAN, "p": NAN, "bh_reject": False,
             "included": False},
            {"family": "economic robustness", "test": "P1 premium fees=calibrated", "estimate": 0.01, "p": 0.5,
             "bh_reject": False, "included": True},
            {"family": "economic robustness", "test": "P1 inc fees=0.25", "estimate": 0.02, "p": 0.6, "bh_reject": False,
             "included": True}]
    return pd.DataFrame(rows)


def looo():
    return pd.DataFrame([{"dropped": "CAISO", "analysis": "AQ3", "block": "LEV6", "failures": 5, "median_log_error": 0.31},
                         {"dropped": "MISO", "analysis": "AQ4", "block": "G6", "failures": 6, "median_log_error": 0.29}])


MIXED = "".join(f"mixed line {i:02d}\n" for i in range(1, 61))

#: Every output the digest reads, with its fixture. The addendum is optional and not listed in the missing markers.
FILES = {
    "legacy/comparison.csv": comparison, "legacy/CHANGES.md": lambda: CHANGES,
    "analogs/protocol_inputs.json": protocol, "analogs/eta_selection.csv": eta_selection,
    "analogs/placebo_he.csv": placebo, "analogs/activity_all.csv": activity, "analogs/simulation.csv": simulation,
    "economics/summary.csv": summary, "economics/bootstrap_ci.csv": bootstrap_ci,
    "economics/decomposition.csv": decomposition, "economics/rotation.csv": rotation,
    "economics/run_meta.json": run_meta, "economics/volmanagement_check.csv": volmanagement,
    "monitors/monitor_eval.csv": monitor_eval,
    "labels_pr/label_control_2x2.csv": label_2x2, "labels_pr/label_control_pr.csv": label_pr,
    "settlement_status.md": lambda: SETTLEMENT, "settlement_status_addendum.md": lambda: ADDENDUM,
    "inference/bh.csv": bh, "inference/looo.csv": looo, "inference/mixed_model.txt": lambda: MIXED,
}


def write_all(out):
    for rel, make in FILES.items():
        put(out, rel, make())
    return out


@pytest.fixture
def rd():
    import es.run_digest as module
    return module


@pytest.fixture
def full(tmp_path):
    return write_all(tmp_path / "es")


def build(rd, out, tmp_path):
    return rd.build_digest(out, dev_root=tmp_path / "no_repo")


# ---------------------------------------------------------------------------------------------------------------
# Missing and unreadable files
# ---------------------------------------------------------------------------------------------------------------

def test_every_missing_output_prints_its_marker_and_the_header_and_all_sections_stay(rd, tmp_path):
    out = tmp_path / "es"
    out.mkdir()
    text = build(rd, out, tmp_path)
    for name in ("comparison.csv", "CHANGES.md", "protocol_inputs.json", "eta_selection.csv", "placebo_he.csv",
                 "activity_all.csv", "simulation.csv", "summary.csv", "bootstrap_ci.csv", "decomposition.csv",
                 "rotation.csv", "run_meta.json", "volmanagement_check.csv", "monitor_eval.csv",
                 "label_control_2x2.csv", "label_control_pr.csv", "settlement_status.md", "bh.csv", "looo.csv",
                 "mixed_model.txt"):
        assert f"_missing: {name}_" in text, name
    for prefix in ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"):
        section(text, prefix)
    assert "No sealed data has been read." in text and "ERROR" not in text


def test_the_out_directory_itself_may_be_absent(rd, tmp_path):
    text = build(rd, tmp_path / "does_not_exist", tmp_path)
    assert "_missing: comparison.csv_" in text


def test_an_unreadable_file_is_marked_and_the_rest_of_the_digest_is_written(rd, full, tmp_path):
    (full / "monitors" / "monitor_eval.csv").write_text("")                  # pandas: EmptyDataError
    (full / "analogs" / "protocol_inputs.json").write_text("{not json")
    text = build(rd, full, tmp_path)
    assert re.search(r"_unreadable: monitor_eval\.csv: \w+: ", text)
    assert re.search(r"_unreadable: protocol_inputs\.json: \w+: ", text)
    assert table_with(section(text, "(a)"), "metric")                         # other sections are intact
    assert table_with(section(text, "(h)"), "family")


def test_a_section_that_cannot_be_built_says_so_and_the_others_are_still_written(rd, full, tmp_path):
    frame = pd.read_csv(full / "economics" / "summary.csv").drop(columns=["position"])
    frame.to_csv(full / "economics" / "summary.csv", index=False)
    text = build(rd, full, tmp_path)
    c = section(text, "(c)")
    assert "ERROR" in c and "KeyError" in c
    assert table_with(section(text, "(d)"), "n_sim")


# ---------------------------------------------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------------------------------------------

def test_the_header_states_the_cutoff_the_sealed_sentence_the_sha_and_each_outputs_mtime(rd, full, tmp_path):
    ts = 1_700_000_000
    os.utime(full / "legacy" / "comparison.csv", (ts, ts))
    (full / "monitors" / "monitor_eval.csv").unlink()
    text = build(rd, full, tmp_path)
    assert text.startswith("# Development digest (data up to 2026-06-30 only)\n")
    head = text.split("\n## (a)")[0]
    assert "Data cut-off: 2026-06-30" in head and "No sealed data has been read." in head
    assert "dart_dev HEAD: not available" in head                              # tmp_path/no_repo is no repository
    stamps = {r["output"]: r["modified (UTC)"] for r in table_with(head, "modified (UTC)")}
    assert stamps["legacy/comparison.csv"] == "2023-11-14 22:13:20Z"
    assert stamps["monitors/monitor_eval.csv"] == "missing"
    for name in ("legacy/CHANGES.md", "analogs/protocol_inputs.json", "analogs/simulation.csv", "economics/summary.csv",
                 "economics/run_meta.json", "labels_pr/label_control_2x2.csv", "settlement_status.md",
                 "inference/bh.csv", "inference/mixed_model.txt"):
        assert re.fullmatch(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\dZ", stamps[name]), name


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_the_git_short_sha_is_that_of_the_repository_head(rd, tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)                      # no signing, hooks or aliases of the user's
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    repo = tmp_path / "repo"
    repo.mkdir()
    for cmd in (["init", "-q"], ["-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "--allow-empty",
                                 "--no-verify", "-m", "x"]):
        subprocess.run(["git", "-C", str(repo), *cmd], check=True, capture_output=True)
    expected = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], check=True, capture_output=True,
                              text=True).stdout.strip()
    assert rd._git_short_sha(repo) == expected and len(expected) >= 7
    assert rd._git_short_sha(tmp_path / "no_repo") is None
    out = write_all(tmp_path / "es")
    assert f"dart_dev HEAD: `{expected}`" in rd.build_digest(out, dev_root=repo)


def test_a_missing_git_binary_is_no_sha_and_no_crash(rd, tmp_path, monkeypatch):
    def no_git(*a, **k):
        raise FileNotFoundError("git")
    monkeypatch.setattr(rd.subprocess, "run", no_git)
    assert rd._git_short_sha(tmp_path) is None


# ---------------------------------------------------------------------------------------------------------------
# (a) regenerated vs published
# ---------------------------------------------------------------------------------------------------------------

def test_a_shows_the_status_counts_the_rows_that_do_not_match_and_the_changes_list(rd, full, tmp_path):
    a = section(build(rd, full, tmp_path), "(a)")
    counts = {r["status"]: r["rows"] for r in table_with(a, "rows")}
    assert counts == {"match": "1", "mismatch": "1", "no_counterpart": "1", "match_other_design": "1"}
    assert "3 of 4 published numbers do not reproduce as stated" in a
    bad = table_with(a, "metric")
    assert [r["metric"] for r in bad] == ["rows", "median_auprc", "median_log_error"]       # n_nodes matched: not listed
    assert all(r["status"] != "match" for r in bad)
    assert bad[0]["note"] == "D6: printed in thousands | see D6"                            # the pipe survives, escaped
    assert (bad[0]["value"], bad[0]["regenerated"]) == ("329000", "306390")
    assert (bad[2]["regenerated"], bad[2]["as_stated_value"], bad[2]["stated_analysis"]) == ("0.2285", "0.3197", "24H2-25H2")
    assert "- D1. The 24H2->25H2 row mixes two model fits" in a and "- D6. Table 1" in a
    assert "Not a numbered correction" not in a and "legacy/CHANGES.md" in a


def test_a_with_no_mismatch_says_so(rd, full, tmp_path):
    frame = comparison()
    frame = frame[frame["match"]]
    put(full, "legacy/comparison.csv", frame)
    a = section(build(rd, full, tmp_path), "(a)")
    assert "All 1 published numbers reproduce as stated" in a


def test_a_reads_a_match_column_that_pandas_could_not_parse_as_bool(rd, full, tmp_path):
    frame = comparison().astype({"match": object})
    frame.loc[0, "match"] = NAN                                              # not True: counts as not matching
    put(full, "legacy/comparison.csv", frame)
    bad = table_with(section(build(rd, full, tmp_path), "(a)"), "metric")
    assert [r["metric"] for r in bad] == ["n_nodes", "rows", "median_auprc", "median_log_error"]


# ---------------------------------------------------------------------------------------------------------------
# (b) analogues
# ---------------------------------------------------------------------------------------------------------------

def test_b_shows_eta_the_registered_family_phi_the_rule_constants_and_seeds(rd, full, tmp_path):
    b = section(build(rd, full, tmp_path), "(b)")
    assert "- stage: `complete`" in b and "- eta: `0.05`" in b and "- rho_star: `0.5`" in b
    assert "- analog_selc: `[0.0121, -0.0046]`" in b and "WARNING" not in b
    fam = {r["hypothesis"]: r for r in table_with(sub(b, "Registered family"), "hypothesis")}
    assert list(fam) == ["H-E", "H-B1", "H-B2"]
    assert fam["H-E"]["decision"] == "confirmatory" and fam["H-E"]["reasons"] == "none"
    assert (fam["H-E"]["approximate"], fam["H-E"]["interim_approximate"]) == ("False", "True")
    assert fam["H-B1"]["decision"] == "replaced by its estimand"
    assert fam["H-B1"]["reasons"] == "simulated power 0.41 at 0.05/3 is below 0.5" and fam["H-B1"]["power"] == "0.41"
    assert fam["H-B2"]["power"] == "0.62" and fam["H-B2"]["interim_approximate"] == ""
    assert "The check is retained as registered." in sub(b, "Registered family")
    phi = {r["block"]: r for r in table_with(sub(b, "Score persistence"), "phi_score_median")}
    assert phi["DART3"]["phi_score_median"] == "0.61" and phi["LEV6"]["phi_score_raw_median"] == ""
    rule = sub(b, "Rule constants and seeds")
    for line in ("- factor: `1.5`", "- binomial_level: `0.05`", "- min_power: `0.5`", "- levels: `[0.016666666666666666, 0.05]`",
                 '- seeds: `{"size_base": 20261022, "rho_search": 7, "he_power": 11, "hb_power": 13}`',
                 '- counted_pairings: `["AQ3-1y", "AQ4-1y"]`'):
        assert line in rule, line
    assert "DECOY" not in b                                                   # the json's own copies of the CSVs stay out


def test_b_with_the_interim_stage_warns_that_the_family_has_not_been_computed(rd, full, tmp_path):
    put(full, "analogs/protocol_inputs.json", protocol(complete=False))
    b = section(build(rd, full, tmp_path), "(b)")
    assert "- stage: `analogs`" in b and "WARNING" in b and "--stage sim" in b
    assert "Registered family" not in b and "no `rule` block" in b


def test_b_eta_selection_is_one_row_per_analogue_with_the_pooled_mean(rd, full, tmp_path):
    rows = table_with(sub(section(build(rd, full, tmp_path), "(b)"), "Eta selection"), "0.05")
    assert [r["analysis"] for r in rows[:5]] == ["AQ3", "AQ4", "AQ1", "AQ2", "AQ3F"]        # the file's order
    assert (rows[0]["0.01"], rows[0]["0.05"]) == ("0.4", "0.04")
    pooled = rows[5]
    assert pooled["analysis"].startswith("mean of AQ1-AQ4") and "AQ3F excluded" in pooled["analysis"]
    assert (pooled["0.01"], pooled["0.05"]) == ("0.35", "0.1")                  # AQ3F (0.9) is not pooled


def test_b_placebo_pairings_carry_counted_and_note_and_cite_ruling_94(rd, full, tmp_path):
    p = sub(section(build(rd, full, tmp_path), "(b)"), "H-E placebo pairings")
    rows = table_with(p, "pair")
    assert [(r["pair"], r["counted"]) for r in rows] == [("AQ3-1y", "True"), ("AQ3-2y", "False")]
    assert rows[1]["note"] == "not counted: the 2-year pairing (Ruling 94)" and rows[0]["note"] == ""
    assert "Ruling 94" in p.split("|")[0]
    stage_one = pd.read_csv(full / "analogs" / "placebo_he.csv").drop(columns=["counted", "note"])
    put(full, "analogs/placebo_he.csv", stage_one)                            # the stage-one file has neither column
    p = sub(section(build(rd, full, tmp_path), "(b)"), "H-E placebo pairings")
    assert "counted" not in table_with(p, "pair")[0] and "Ruling 94" not in p


def test_b_activity_is_failures_over_nodes_and_median_error_by_block_and_policy_for_xgb(rd, full, tmp_path):
    a = sub(section(build(rd, full, tmp_path), "(b)"), "Activity of the analogues")
    rows = table_with(a, "P1")
    assert [(r["analysis"], r["block"]) for r in rows] == [("AQ3", "G6"), ("AQ3", "LEV6"), ("AQ4", "G6"), ("AQ4", "LEV6")]
    assert (rows[0]["P1"], rows[0]["P6"]) == ("2/3 (0.20)", "2/3 (0.20)")       # 2 of 3 nodes fail; median of .1 .2 .3
    assert rows[2]["P1"] == "0/3 (0.20)"
    assert "3/3" not in a                                                      # the logit rows (all failing) are left out


# ---------------------------------------------------------------------------------------------------------------
# (c) economics
# ---------------------------------------------------------------------------------------------------------------

def test_c_primary_table_is_premium_at_calibrated_fees_with_rotation_p_on_p1_rows_only(rd, full, tmp_path):
    c = section(build(rd, full, tmp_path), "(c)")
    rows = table_with(sub(c, "Premium position, calibrated fees: the policies"), "rotation_p")
    assert [(r["analysis"], r["policy"], r["SELC"]) for r in rows] == [("AQ3", "P1", "0.011"), ("AQ3", "P4", "0.012"),
                                                                       ("AQ4", "P1", "0.013")]
    assert [r["rotation_p"] for r in rows] == ["0.576", "", "0.663"]
    for col in ("alert_rate", "SELC_decile", "SNET", "NET", "ESred", "CAP", "J_0.0", "J_0.25", "J_0.5", "J_1.0"):
        assert col in rows[0]
    assert rows[0]["NET"] == "-58697" and rows[0]["J_1.0"] == "236707" and rows[0]["SELC_decile"] == "0.0391"
    assert "0.987" not in c and "0.988" not in c and "0.989" not in c         # another fee setting or position: left out


def test_c_comparators_and_the_spike_exposure_split_are_their_own_tables(rd, full, tmp_path):
    c = section(build(rd, full, tmp_path), "(c)")
    legend = sub(c, "Comparators").split("|")[0]
    assert "`calendar@count`: after-the-fact top-k" in legend and "`VOL@count`" in legend and "`curve_P6_a<alpha>`" in legend
    comp = table_with(sub(c, "Comparators"), "comparator")
    assert [(r["analysis"], r["comparator"], r["SELC"]) for r in comp] == [("AQ3", "calendar@count", "0.014")]
    split = table_with(sub(c, "Spike-exposure split"), "position")
    assert [(r["position"], r["SELC"], r["rotation_p"]) for r in split] == [
        ("premium[long_spikes]", "0.015", "0.033"), ("premium[short_spikes]", "0.016", "0.664")]


def test_c_bootstrap_intervals_keep_the_pipe_rows_ex_caiso_and_the_pooled_rehearsal(rd, full, tmp_path):
    c = section(build(rd, full, tmp_path), "(c)")
    main_rows = table_with(sub(c, "Bootstrap intervals: SELC, NET"), "stat")
    stats = [(r["analysis"], r["stat"]) for r in main_rows]
    assert ("AQ3", "E4:dSELC|calendar@count") in stats and ("AQ3|ex-CAISO", "E4:dSELC|VOL@count") in stats
    assert ("AQ3+AQ4 (rehearsal of the pooled estimands)", "SELC") in stats
    assert not any(s.startswith("E1:") for _, s in stats) and len(stats) == 5
    first = main_rows[0]
    assert (first["estimate"], first["lo"], first["hi"], first["approximate"], first["n_windows"]) == (
        "0.011", "-0.03", "0.05", "False", "1")
    e1 = table_with(sub(c, "Bootstrap intervals: E1"), "stat")
    assert [r["stat"] for r in e1] == ["E1:NET:timing", "E1:ESred:total"] and e1[1]["approximate"] == "True"


def test_c_decomposition_is_one_column_per_analysis(rd, full, tmp_path):
    d = sub(section(build(rd, full, tmp_path), "(c)"), "Decomposition (E1)")
    rows = table_with(d, "AQ3")
    assert [(r["measure_part"], r["AQ3"], r["AQ4"]) for r in rows] == [
        ("NET_timing", "-6126", "-8427"), ("NET_level", "-2840", "1552"), ("NET_total", "-8967", "-6875")]
    assert "`timing = P1 - P4@r*`" in d and "`level = P4@r* - P4`" in d and "`total = P1 - P4`" in d


def test_c_rotation_and_volatility_management_tables(rd, full, tmp_path):
    c = section(build(rd, full, tmp_path), "(c)")
    r = sub(c, "Rotation test")
    rows = table_with(r, "window")
    assert [x["window"] for x in rows] == ["AQ3", "AQ3+AQ4 (rehearsal of the pooled H-E test)"]
    assert (rows[0]["t_obs"], rows[0]["p"], rows[1]["n_null"]) == ("-0.56", "0.576", "8463")
    assert "studentized T" in r and "selc_obs" in r
    vm = table_with(sub(c, "Volatility-management check"), "sharpe_alerted")
    assert (vm[0]["analysis"], vm[0]["sharpe_alerted"], vm[0]["sharpe_not_alerted"]) == ("AQ3", "0.092", "0.019")


def test_c_shows_the_run_meta_and_flags_a_run_that_is_not_complete(rd, full, tmp_path):
    run = sub(section(build(rd, full, tmp_path), "(c)"), "Run")
    assert "- status: `complete`" in run and "- n_rep: `2000`" in run and "- seed: `20261022`" in run
    assert '- strata_primary: `["market", "node", "settle_hour", "wclass", "month"]`' in run and "WARNING" not in run
    put(full, "economics/run_meta.json", run_meta("running"))
    run = sub(section(build(rd, full, tmp_path), "(c)"), "Run")
    assert "WARNING" in run and "`running`" in run and "1 of 3 analyses" in run
    put(full, "economics/run_meta.json", run_meta("failed"))
    run = sub(section(build(rd, full, tmp_path), "(c)"), "Run")
    assert "WARNING" in run and "`failed`" in run and "AQ4" in run and "ValueError: boom" in run


# ---------------------------------------------------------------------------------------------------------------
# (d) size and power
# ---------------------------------------------------------------------------------------------------------------

A3, P05 = "rate at alpha/3 (0.0167)", "rate at 0.05"


def test_d_h_e_size_rows_are_split_into_primary_and_decile_strata_with_the_decile_ones_labelled(rd, full, tmp_path):
    d = section(build(rd, full, tmp_path), "(d)")
    prim = table_with(sub(d, "H-E size, primary strata"), "scenario")
    assert [(r["test"], r["scenario"], r["persistence"], r["n_sim"], r[A3], r[P05]) for r in prim] == [
        ("H-E single A, size", "homoskedastic", "fit", "300", "0.02", "0.06"),
        ("H-E pooled A+B, size", "heteroskedastic", "upper", "300", "0.017", "0.049")]
    assert "to be read against 0.0167 and 0.05" in sub(d, "H-E size, primary strata").split("|")[0]
    assert "sensitivity, not used by the rule" in re.search(r"^#{3,6} H-E size, decile[^\n]*$", d, re.M).group(0)
    assert "never governing the rule" in sub(d, "H-E size, decile strata").split("|")[0]
    dec = table_with(sub(d, "H-E size, decile strata"), "scenario")
    assert [(r["strata"], r[A3], r[P05]) for r in dec] == [("decile", "0.0555", "0.1111")]


def test_d_h_b_size_rows_are_by_law_persistence_and_scenario_with_the_no_drift_failure_context(rd, full, tmp_path):
    hb = sub(section(build(rd, full, tmp_path), "(d)"), "H-B size by law")
    rows = table_with(hb, "law")
    assert [(r["law"], r["persistence"], r["scenario"], r["n_sim"], r[A3], r[P05]) for r in rows] == [
        ("DART3", "fit", "no_drift", "300", "0.015", "0.05"), ("LEV6", "upper", "common_drift_0.5", "300", "0.03", "0.09")]
    assert [r["nodrift_fail_q95_0.5_2"] for r in rows] == ["9", ""]            # defined for the no-drift rows only
    assert (rows[0]["nodrift_fail_q95_0.67_1.5"], rows[0]["nodrift_fail_q95_0.8_1.25"]) == ("14", "20")
    assert "95th percentile" in hb.split("|")[0]
    sim = pd.read_csv(full / "analogs" / "simulation.csv").drop(columns=[c for c in pd.read_csv(
        full / "analogs" / "simulation.csv").columns if c.startswith("nodrift_")])
    put(full, "analogs/simulation.csv", sim)                                  # a file without them: no column, no caption
    hb = sub(section(build(rd, full, tmp_path), "(d)"), "H-B size by law")
    assert "nodrift" not in hb and "95th percentile" not in hb


def test_d_power_curve_and_h_b_power_rows(rd, full, tmp_path):
    d = section(build(rd, full, tmp_path), "(d)")
    curve = table_with(sub(d, "H-E power curve"), "rho")
    assert [(r["test"], r["rho"], r["n_sim"], r[A3], r[P05]) for r in curve] == [
        ("H-E pooled A+B, power curve", "0.05", "100", "0.02", "0.08"),
        ("H-E pooled A+B, power curve", "0.9", "100", "0.7", "0.9"),
        ("H-E pooled A+B, power", "0.5", "300", "0.55", "0.8")]
    assert curve[2]["target_selc"] == "0.0019" and curve[2]["note"] == "target not bracketed"
    hb = table_with(sub(d, "H-B power"), "calibrated_delta")
    assert [(r["test"], r["calibrated_delta"], r["n_sim"], r[A3], r[P05]) for r in hb] == [
        ("H-B1 power", "0.3", "150", "0.41", "0.62"), ("H-B2 power", "0.35", "150", "0.62", "0.8")]


def test_d_flags_every_power_below_the_floor_and_takes_the_decision_from_the_protocol_inputs(rd, full, tmp_path):
    d = section(build(rd, full, tmp_path), "(d)")
    power = {r["hypothesis"]: r for r in table_with(sub(d, "Power per hypothesis"), "floor check")}
    assert list(power) == ["H-E", "H-B1", "H-B2"]
    assert power["H-B1"]["floor check"] == "below 0.5" and power["H-B1"][A3] == "0.41"
    assert power["H-B2"]["floor check"] == "ok" and power["H-E"]["floor check"] == "ok"
    assert "**Below the floor of 0.5: H-B1 (0.41).**" in sub(d, "Power per hypothesis")
    decision = sub(d, "Family decision")
    assert "- **H-E**: confirmatory" in decision
    assert "- **H-B1**: replaced by its estimand (simulated power 0.41 at 0.05/3 is below 0.5)" in decision
    assert "- **H-B2**: confirmatory" in decision


def test_d_a_power_that_is_unavailable_counts_as_below_the_floor(rd, full, tmp_path):
    sim = simulation()
    sim.loc[sim["test"] == "H-B2 power", ["reject_0.0166667", "reject_0.05"]] = NAN
    sim.loc[sim["test"] == "H-B2 power", "error"] = "target outside the grid"
    put(full, "analogs/simulation.csv", sim)
    power = {r["hypothesis"]: r for r in table_with(sub(section(build(rd, full, tmp_path), "(d)"), "Power per hypothesis"),
                                                    "floor check")}
    assert power["H-B2"]["floor check"] == "below 0.5 (unavailable)" and power["H-B2"][A3] == ""
    assert "**Below the floor of 0.5: H-B1 (0.41), H-B2 (unavailable).**" in sub(
        section(build(rd, full, tmp_path), "(d)"), "Power per hypothesis")


def test_d_without_a_simulation_or_with_the_interim_protocol_it_says_what_is_missing(rd, full, tmp_path):
    (full / "analogs" / "simulation.csv").unlink()
    put(full, "analogs/protocol_inputs.json", protocol(complete=False))
    d = section(build(rd, full, tmp_path), "(d)")
    assert "_missing: simulation.csv_" in d
    assert "Not computed" in sub(d, "Family decision") and "stage `analogs`" in sub(d, "Family decision")


def test_d_reads_the_floor_from_the_rule_when_the_protocol_has_one(rd, full, tmp_path):
    meta = protocol()
    meta["rule"]["min_power"] = 0.7
    put(full, "analogs/protocol_inputs.json", meta)
    power = {r["hypothesis"]: r for r in table_with(sub(section(build(rd, full, tmp_path), "(d)"), "Power per hypothesis"),
                                                    "floor check")}
    assert power["H-B2"]["floor check"] == "below 0.7" and power["H-E"]["floor check"] == "below 0.7"


# ---------------------------------------------------------------------------------------------------------------
# (e) monitors, (f) label control
# ---------------------------------------------------------------------------------------------------------------

def test_e_flags_calibration_windows_longer_than_100_days_and_keeps_the_placebo_columns(rd, full, tmp_path):
    e = section(build(rd, full, tmp_path), "(e)")
    rows = table_with(e, "placebo_median")
    note = "longer calibration window: only ranks interpreted"
    assert [(r["analysis"], r["monitor"], r["calib_days"], r["note"]) for r in rows] == [
        ("AQ3", "W1_norm", "92", ""), ("AQ3", "early_rate", "92", ""), ("EDGE100", "PSI", "100", ""),
        ("EDGE101", "PSI", "101", note), ("2024-2025", "domain_auc", "366", note)]
    assert (rows[0]["placebo_median"], rows[0]["placebo_n"], rows[1]["placebo_median"]) == ("0.4", "33", "")
    for col in ("spearman", "spearman_p_one_sided", "auroc_fail", "n_used"):
        assert col in rows[0]


def test_f_the_2x2_control_is_grouped_by_block_and_the_pr_table_is_a_median_over_nodes(rd, full, tmp_path):
    f = section(build(rd, full, tmp_path), "(f)")
    rows = table_with(sub(f, "Failures and median R"), "failures")
    assert [(r["block"], r["analysis"], r["failures"]) for r in rows] == [
        ("LEV6", "LC-full-q2023", "7"), ("LEV6", "LC-full-q2324", "8"), ("LEV6", "LC-h1-q2023", "1"),
        ("LEV6", "LC-h1-q2324", "2"),
        ("DART18", "LC-full-q2023", "7"), ("DART18", "LC-full-q2324", "7"), ("DART18", "LC-h1-q2023", "12"),
        ("DART18", "LC-h1-q2324", "13")]
    assert rows[0]["n_nodes"] == "33" and rows[0]["median_R"] == "0.97"
    pr = table_with(sub(f, "Precision and recall"), "precision_q")
    assert [(r["analysis"], r["block"], r["nodes"], r["precision_q"], r["recall_q"], r["n_nan_outcome"]) for r in pr] == [
        ("LC-full-q2023", "LEV6", "3", "0.2", "0.2", "3"), ("LC-h1-q2023", "LEV6", "3", "0.2", "0.2", "3")]
    assert pr[0]["precision_lift"] == "3"                                      # median of 2, 3, 4


# ---------------------------------------------------------------------------------------------------------------
# (g) settlement, (h) inference
# ---------------------------------------------------------------------------------------------------------------

def test_g_embeds_the_status_note_two_levels_down_outside_code_and_reads_the_fees_from_the_code(
        rd, full, tmp_path, monkeypatch):
    monkeypatch.setattr(rd, "CALIBRATED_FEES", {"NYISO": {1: 0.21, -1: 0.16}, "CAISO": {1: 0.56, -1: 0.17}})
    text = build(rd, full, tmp_path)
    g = section(text, "(g)")
    assert "### Settlement real-time legs: pull status (Task 6)" in g and "#### 1. CAISO earliest available FMM date" in g
    assert "##### Bisection" in g
    assert "\n# a comment inside a fence stays a comment\n" in g                             # a fence is left alone
    assert "**Earliest FMM date: 2023-07-13**" in g and "| 2023-04-15 | no data |" in g
    assert "### Addendum to settlement_status.md (23 Oct 2026, controller)" in g and "Task 6 fix round 1" in g
    assert not re.search(r"^# Settlement|^# Addendum|^## 1\. CAISO", text, re.M)
    assert len(re.findall(r"^# Development digest", text, re.M)) == 1
    fees = {r["market"]: r for r in table_with(sub(g, "Calibrated fees"), "market")}
    assert (fees["NYISO"]["INC (d=+1)"], fees["NYISO"]["DEC (d=-1)"]) == ("0.21", "0.16")
    assert (fees["CAISO"]["INC (d=+1)"], fees["CAISO"]["DEC (d=-1)"]) == ("0.56", "0.17")
    assert "docs/fee_rederivation_2026-10-23.md" in sub(g, "Calibrated fees")


def test_g_without_the_addendum_prints_no_marker_and_the_real_fees_are_the_code_constant(rd, full, tmp_path):
    from es.positions import CALIBRATED_FEES
    assert rd.CALIBRATED_FEES is CALIBRATED_FEES
    (full / "settlement_status_addendum.md").unlink()
    g = section(build(rd, full, tmp_path), "(g)")
    assert "Addendum" not in g and "addendum" not in g
    fees = {r["market"]: r for r in table_with(sub(g, "Calibrated fees"), "market")}
    assert set(fees) == set(CALIBRATED_FEES)
    assert fees["ISONE"]["INC (d=+1)"] == "0.70"


def test_h_bh_families_looo_and_the_first_40_lines_of_the_mixed_model(rd, full, tmp_path):
    h = section(build(rd, full, tmp_path), "(h)")
    fam = {r["family"]: r for r in table_with(sub(h, "Benjamini-Hochberg families"), "BH rejections")}
    assert (fam["audit robustness"]["tests"], fam["audit robustness"]["included"], fam["audit robustness"]["BH rejections"]) == (
        "3", "2", "n/a (descriptive)")                                  # Ruling 118, M5: the audit family is descriptive, no rejection count
    assert "no count of its rejections is reported" in sub(h, "Benjamini-Hochberg families")
    assert "diagnostic" in sub(h, "Tests")                            # its flags are diagnostics
    assert (fam["economic robustness"]["tests"], fam["economic robustness"]["included"],
            fam["economic robustness"]["BH rejections"]) == ("2", "2", "0")
    tests = table_with(sub(h, "Tests"), "bh_reject")
    assert len(tests) == 5 and tests[2]["p"] == "" and tests[2]["included"] == "False" and tests[0]["bh_reject"] == "True"
    lo = table_with(sub(h, "Leave-one-operator-out"), "dropped")
    assert [(r["dropped"], r["failures"], r["median_log_error"]) for r in lo] == [("CAISO", "5", "0.31"), ("MISO", "6", "0.29")]
    mixed = sub(h, "Mixed model")
    assert "mixed line 40" in mixed and "mixed line 41" not in mixed and "mixed line 01" in mixed
    assert "first 40 of 60 lines" in mixed and "inference/mixed_model.txt" in mixed


def test_h_a_short_mixed_model_is_shown_whole_without_a_truncation_note(rd, full, tmp_path):
    put(full, "inference/mixed_model.txt", "only\ntwo lines\n")
    mixed = sub(section(build(rd, full, tmp_path), "(h)"), "Mixed model")
    assert "only\ntwo lines" in mixed and "first" not in mixed


# ---------------------------------------------------------------------------------------------------------------
# Cells, tables and the document
# ---------------------------------------------------------------------------------------------------------------

def test_cells_are_formatted_to_four_significant_digits_with_no_exponent_for_large_numbers(rd):
    fmt = rd._fmt
    assert fmt(0.5760869565217391) == "0.5761" and fmt(-0.006297) == "-0.006297" and fmt(0.05) == "0.05"
    assert fmt(58696.598) == "58697" and fmt(-236706.886) == "-236707" and fmt(1234567.8) == "1234568"
    assert fmt(7.0) == "7" and fmt(1e-7) == "1e-07" and fmt(0.1 + 0.2) == "0.3"
    assert fmt(33) == "33" and fmt(np.int64(8463)) == "8463" and fmt(np.float64(0.25)) == "0.25"
    assert fmt(True) == "True" and fmt(np.bool_(False)) == "False"
    assert fmt(NAN) == "" and fmt(None) == "" and fmt(pd.NA) == ""
    assert fmt(float("inf")) == "inf" and fmt(float("-inf")) == "-inf"
    assert fmt("a|b") == "a\\|b" and fmt("two\nlines") == "two lines"


def test_a_table_cannot_be_broken_by_a_pipe_or_a_newline_in_a_cell_or_a_header(rd):
    frame = pd.DataFrame({"E4:dSELC|x": ["a|b", "line\nbreak"], "n": [1.5, 2000.0]})
    rows = tables(rd._md(frame))[0]
    assert rows == [{"E4:dSELC|x": "a|b", "n": "1.5"}, {"E4:dSELC|x": "line break", "n": "2000"}]
    assert "---:" in rd._md(frame).splitlines()[1]                              # a numeric column is right-aligned


def test_an_empty_table_is_a_note_not_a_crash(rd):
    assert rd._md(pd.DataFrame({"a": []})) == "_(no rows)_"
    assert rd._md(pd.DataFrame()) == "_(no rows)_"


def test_a_table_longer_than_the_limit_is_cut_and_says_so(rd, monkeypatch):
    monkeypatch.setattr(rd, "MAX_ROWS", 3)
    text = rd._md(pd.DataFrame({"a": range(5)}))
    assert [r["a"] for r in tables(text)[0]] == ["0", "1", "2"]
    assert "first 3 of 5 rows" in text


# ---------------------------------------------------------------------------------------------------------------
# main(): the file, the print, determinism and the real tree
# ---------------------------------------------------------------------------------------------------------------

@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    """es.run_digest pointed at tmp_path/es (the fixture outputs), with any write under the real outputs/ or cache/
    failing the test, so a main() that ignored OUT_DIR would fail here instead of writing there."""
    import es.config as config
    import es.run_digest as module

    guarded = [config.DEV_ROOT / "outputs", config.DEV_ROOT / "cache"]

    def check(target):
        resolved = pathlib.Path(target).resolve()
        assert not any(g.resolve() in (resolved, *resolved.parents) for g in guarded), (
            f"a test wrote to {resolved}, under the real outputs/ or cache/")

    original = {"write_text": pathlib.Path.write_text, "mkdir": pathlib.Path.mkdir, "replace": os.replace}

    def write_text(self, *a, **k):
        check(self)
        return original["write_text"](self, *a, **k)

    def mkdir(self, *a, **k):
        check(self)
        return original["mkdir"](self, *a, **k)

    def replace(src, dst, *a, **k):
        check(src)
        check(dst)
        return original["replace"](src, dst, *a, **k)

    monkeypatch.setattr(pathlib.Path, "write_text", write_text)
    monkeypatch.setattr(pathlib.Path, "mkdir", mkdir)
    monkeypatch.setattr(os, "replace", replace)
    out = write_all(tmp_path / "es")
    monkeypatch.setattr(module, "OUT_DIR", out)
    monkeypatch.setattr(module, "DEV_ROOT", tmp_path / "no_repo")
    return module, out


def test_main_writes_digest_md_beside_the_outputs_and_prints_its_path(sandbox, capsys):
    rd, out = sandbox
    rd.main()
    printed = capsys.readouterr().out.strip().splitlines()
    assert printed == [str(out / "DIGEST.md")]
    text = (out / "DIGEST.md").read_text(encoding="utf-8")
    assert text == rd.build_digest(out, dev_root=out.parent / "no_repo")
    assert text.startswith("# Development digest (data up to 2026-06-30 only)\n") and text.endswith("\n")
    assert not [p for p in out.rglob("*") if p.name.endswith(".tmp")]


def test_main_creates_the_out_directory_when_it_is_absent(sandbox, tmp_path, monkeypatch, capsys):
    rd, _ = sandbox
    fresh = tmp_path / "fresh" / "es"
    monkeypatch.setattr(rd, "OUT_DIR", fresh)
    rd.main()
    assert (fresh / "DIGEST.md").exists() and "_missing: comparison.csv_" in (fresh / "DIGEST.md").read_text(encoding="utf-8")


def test_main_is_deterministic_and_does_not_read_its_own_previous_digest(sandbox, capsys):
    rd, out = sandbox
    rd.main()
    first = (out / "DIGEST.md").read_bytes()
    rd.main()
    assert (out / "DIGEST.md").read_bytes() == first


def test_main_warns_on_standard_output_when_a_section_could_not_be_built(sandbox, capsys):
    rd, out = sandbox
    pd.read_csv(out / "economics" / "summary.csv").drop(columns=["policy"]).to_csv(out / "economics" / "summary.csv", index=False)
    rd.main()
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[-1] == str(out / "DIGEST.md") and any("WARNING" in line and "(c)" in line for line in lines[:-1])


def test_the_digest_of_the_fixture_outputs_never_touches_the_real_outputs(sandbox):
    rd, out = sandbox
    real = DEV_ROOT / "outputs" / "es" / "DIGEST.md"
    before = real.stat().st_mtime_ns if real.exists() else None
    rd.main()
    after = real.stat().st_mtime_ns if real.exists() else None
    assert before == after


def test_the_module_imports_no_loader_of_prices_scores_or_settlements():
    """The digest reads output files and two code constants only: a fresh interpreter that imports it must not have
    loaded any es module that could read a price (data, scores, settlement, the runners, ...)."""
    code = "import sys, es.run_digest; print(sorted(m for m in sys.modules if m == 'es' or m.startswith('es.')))"
    env = dict(os.environ, DART_PROJECT_ROOT=str(DEV_ROOT.parent))
    res = subprocess.run([sys.executable, "-c", code], cwd=DEV_ROOT, env=env, capture_output=True, text=True, check=True)
    loaded = set(ast.literal_eval(res.stdout.strip().splitlines()[-1]))
    assert loaded <= {"es", "es.config", "es.positions", "es.run_digest"}, loaded


def test_d_names_no_hypothesis_when_every_power_clears_the_floor(rd, full, tmp_path):
    put(full, "analogs/simulation.csv", simulation(b1_power=0.9))
    p = sub(section(build(rd, full, tmp_path), "(d)"), "Power per hypothesis")
    assert "Below the floor" not in p and "Every power is at or above the floor of 0.5." in p


def test_the_document_has_no_run_of_blank_lines_and_no_trailing_spaces_on_a_heading(rd, full, tmp_path):
    text = build(rd, full, tmp_path)
    assert "\n\n\n" not in text and text.endswith("\n") and not text.endswith("\n\n")
    assert not [line for line in text.splitlines() if line.startswith("#") and line != line.rstrip()]


def test_d_a_hypothesis_without_a_power_row_is_named_in_the_floor_line(rd, full, tmp_path):
    sim = simulation()
    put(full, "analogs/simulation.csv", sim[sim["test"] != "H-B2 power"])
    p = sub(section(build(rd, full, tmp_path), "(d)"), "Power per hypothesis")
    assert "**Below the floor of 0.5: H-B1 (0.41), H-B2 (no row).**" in p
    rows = {r["hypothesis"]: r for r in table_with(p, "floor check")}
    assert rows["H-B2"]["floor check"] == "no row in simulation.csv"


def test_a_family_entry_without_a_verdict_is_unknown_not_an_estimand(rd, full, tmp_path):
    meta = protocol()
    del meta["family"]["H-B2"]["confirmatory"]
    put(full, "analogs/protocol_inputs.json", meta)
    text = build(rd, full, tmp_path)
    fam = {r["hypothesis"]: r for r in table_with(sub(section(text, "(b)"), "Registered family"), "hypothesis")}
    assert fam["H-B2"]["decision"] == "unknown (no `confirmatory` entry)"
    assert "- **H-B2**: unknown (no `confirmatory` entry)" in sub(section(text, "(d)"), "Family decision")
    assert fam["H-B1"]["decision"] == "replaced by its estimand" and fam["H-E"]["decision"] == "confirmatory"


def test_a_filesystem_error_while_looking_for_a_file_is_unreadable_not_a_crash(rd, full, tmp_path, monkeypatch):
    real_exists = pathlib.Path.exists

    def exists(self, *a, **k):
        if self.name in ("comparison.csv", "settlement_status_addendum.md"):
            raise TimeoutError("read timed out")
        return real_exists(self, *a, **k)

    monkeypatch.setattr(pathlib.Path, "exists", exists)
    text = build(rd, full, tmp_path)
    assert "_unreadable: comparison.csv: TimeoutError: read timed out_" in text
    assert "_unreadable: settlement_status_addendum.md: TimeoutError: read timed out_" in text
    stamps = {r["output"]: r["modified (UTC)"] for r in table_with(text.split("\n## (a)")[0], "modified (UTC)")}
    assert stamps["legacy/comparison.csv"] == "unreadable" and stamps["legacy/CHANGES.md"] != "unreadable"
    assert table_with(section(text, "(h)"), "family")                       # the rest is intact
