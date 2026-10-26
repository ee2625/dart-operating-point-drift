import json
from pathlib import Path

import pandas as pd


def _sources(tmp_path: Path):
    from es.paper_macros import Sources
    out, paper = tmp_path / "out", tmp_path / "paper"
    (out / "legacy").mkdir(parents=True)
    (out / "economics").mkdir()
    (out / "paper").mkdir()
    pd.DataFrame({"market": ["NYISO", "ISONE", "MISO", "CAISO"], "nodes": [10, 9, 8, 6], "hourly_rows": [306390, 276000, 245000, 169200],
                  "hubs": [0, 1, 8, 3], "zones": [10, 8, 0, 3], "prevalence_24_q2023": [0.05, 0.04, 0.08, 0.05],
                  "prevalence_24_q2023H1": [0.053, 0.043, 0.082, 0.048]}).to_csv(out / "legacy" / "table1_panel.csv", index=False)
    rows = []
    for a in ("2024-2025", "25H1-26H1", "frozen-26H1"):
        for b in ("G6", "LEV6", "MID3", "DART3", "MIDDART6", "ROLL_LOC6", "ROLL_DISP9", "DART18", "G1_24"):
            for e in ("xgb", "logit"):
                rows.append({"analysis": a, "block": b, "estimator": e, "seed": 42, "policy": "P1", "alpha": 0.05,
                             "failures": 3, "n_nodes": 33, "median_R": 0.9, "median_log_error": 0.1, "median_auprc": 0.2,
                             "auprc_lift": 1.5, "prevalence": 0.1, "n_zero_activity": 0,
                             "failures_band_0.67_1.5": 5, "failures_band_0.8_1.25": 9,
                             "failures_NYISO": 1, "failures_ISONE": 1, "failures_MISO": 1, "failures_CAISO": 0})
    pd.DataFrame(rows).to_csv(out / "legacy" / "summary.csv", index=False)
    stats = ("SELC", "NET", "E2:SNET", "E3:ESred", "E4:SELC_decile", "E4:SELC|calendar@count", "E4:SELC|VOL@count",
             "E4:ESred|calendar@count", "E4:ESred|VOL@count", "E4:dESred|calendar@count", "E4:dESred|VOL@count",
             "E1:NET:timing", "E1:NET:level", "E1:NET:total", "E1:ESred:timing", "E1:ESred:level", "E1:ESred:total",
             "E1:J_0.5:timing", "E1:J_0.5:level", "E1:J_0.5:total")

    def ci_row(analysis, s, scale):
        est, lo, hi = (1234.5, 100.0, 2000.0) if s == "E3:ESred" else (10.0 * scale, 5.0 * scale, 15.0 * scale)
        return {"analysis": analysis, "stat": s, "estimate": est, "lo": lo, "hi": hi, "approximate": s.startswith("E3"), "n_windows": 1}
    pd.DataFrame([ci_row("AQ3", s, 1) for s in stats] + [ci_row("AQ3|ex-CAISO", s, 2) for s in stats]
                 + [ci_row("AQ4", s, 3) for s in stats]).to_csv(out / "economics" / "bootstrap_ci.csv", index=False)
    (out / "inference").mkdir()
    pd.DataFrame([{"family": f, "test": f"{f} test {i}", "estimate": 0.1 * i, "p": p, "bh_reject": p < 0.01}
                  for f in ("audit robustness", "economic robustness", "drift monitors") for i, p in enumerate((0.001, 0.2, 0.5), 1)]) \
        .to_csv(out / "inference" / "bh.csv", index=False)
    (out / "labels_pr").mkdir()
    pd.DataFrame([{"analysis": a, "block": b, "market": "M", "node": n, "precision_q": 0.2, "recall_q": 0.1, "precision_eval": 0.3,
                   "recall_eval": 0.15, "precision_lift": 3.0, "prevalence_eval": 0.1, "n_nan_outcome": 0}
                  for a in ("LC-full-q2023", "LC-h1-q2324") for b in ("LEV6", "DART3") for n in ("a", "b")]) \
        .to_csv(out / "labels_pr" / "label_control_pr.csv", index=False)
    (out / "analogs").mkdir()
    pd.DataFrame([{"analysis": "AQ3", "block": b, "estimator": "xgb", "policy": "P1", "market": "M", "node": n,
                   "R": 1.0, "log_error": le, "fail_0.5_2": False}
                  for b, le in (("DART3", 0.10), ("LEV6", 0.30), ("ROLL_DISP9", 0.05), ("DART18", 0.20), ("G1_24", 0.25), ("G6", 0.01))
                  for n in ("a", "b")]).to_csv(out / "analogs" / "activity_all.csv", index=False)
    pd.DataFrame([{"analysis": "AQ3", "policy": "P1", "comparator": "none", "position": pos, "fees": fee, "SELC": selc, "SNET": 0.1,
                   "NET": net, "ESred": 1.0, "CAP": 0.2, "n_days": 92}
                  for pos in ("premium", "inc", "dec", "premium[short_spikes]", "premium[long_spikes]")
                  for fee, net, selc in (("calibrated", 5.0, 0.1), ("0.0", 6.0, 0.1), ("0.1", 4.0, 0.1), ("0.25", 3.0, 0.1),
                                         ("0.5", 1.0, 0.1), ("1.0", -2.0, -0.1), ("stress", -1.0, 0.05))]) \
        .to_csv(out / "economics" / "summary.csv", index=False)
    (out / "paper" / "implementation.json").write_text(json.dumps(
        {"xgboost": {"n_estimators": 300}, "logistic": {"C": 1.0}, "scaler": {"centre": "median"}, "windows": [],
         "blocks": {"calendar_G6": ["a"], "LEV6": {"paper_name": "absolute level", "n_price_features": 6, "price_features": ["da_lmp_lag_48h"], "n_model_features": 16}},
         "rules": {"alpha": 0.05, "bootstrap": {"replicates": None}}, "versions": {"python": "3.12.7"},
         "coverage": [{"series": "CAISO", "nodes": 6, "hourly_rows": 169200, "first_local_date": "2023-04-13", "last_local_date": "2026-06-30"}]}))
    paper.mkdir()
    return Sources(out, paper, tmp_path / "docs_sealed"), paper


def test_panel_table_rows_and_formats(tmp_path):
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    tex = BUILDERS["tab_panel"][0](src)
    assert tex.count("\\\\") == 6 and "ISO-NE & 9 (1 / 8) & 276{,}000 & 0.040" in tex      # header + 4 operators + all


def test_block_audit_table_orders_blocks_and_reads_both_estimators(tmp_path):
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    tex = BUILDERS["tab_block_audit"][0](src)
    body = tex.split("\\midrule")[1]
    assert body.index("calendar") < body.index("absolute level") < body.index("plain DART lags") < body.index("combined")
    assert "absolute level & 6 & 3 & 3 & 3 & 3 & 3" in tex


def test_decomposition_and_ci_tables_flag_approximate(tmp_path):
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    ci = BUILDERS["tab_econ_ci"][0](src)
    assert "\\$1{,}235 [\\$100, \\$2{,}000]\\textsuperscript{a}" in ci          # the ESred cell carries the mark
    assert "1000.0\\% [500.0\\%, 1500.0\\%] &" in ci                           # the SELC cell does not
    assert ci.index("AQ3: 2025-Q3 &") < ci.index("AQ3: 2025-Q3, without CAISO &") < ci.index("AQ4: 2025-Q4 &")
    dec = BUILDERS["tab_decomposition"][0](src)
    # The two assertions below replace the brief's (a sideways tabular with `1 + 3 * 3` line ends): controller Ruling 95 made the table
    # long-format longtables that fit the text width, a row per analysis and component and a column per measure.
    assert "AQ3: 2025-Q3 & timing & \\$10 [\\$5, \\$15] & \\$10 [\\$5, \\$15] \\\\" in dec
    assert dec.count(" & timing & ") == 3 * 2 and "\\begin{tabular}" not in dec           # 3 analyses in each of the two panels


def test_bh_families_and_label_control_pr(tmp_path):
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    bh = BUILDERS["tab_bh_families"][0](src)
    assert "drift monitors & 3 & 1 & 0.001" in bh and bh.count("\\\\") == 4
    ledger = BUILDERS["esm_ledger_bh"][0](src)
    assert ledger.count("\\\\") >= 9 + 2 and "yes" in ledger and "no" in ledger
    pr = BUILDERS["tab_label_control_pr"][0](src)
    assert "calibrate 2025, $q$ 2023 & absolute level & 0.20 & 0.10 & 0.30 & 0.15 & 3.00" in pr


def test_block_sign_panel_reads_activity(tmp_path):
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    tex = BUILDERS["tab_fee_position_signs"][0](src)
    assert tex.count("\\begin{tabular}") == 2                                   # fee x position panel, then block panel
    block_panel = tex.split("\\begin{tabular}")[2]
    assert "absolute level & +" in block_panel and "rolling dispersion & \\textminus{}" in block_panel
    assert "blocks are not covered" not in tex and "\\footnotetext{" in tex


def test_build_all_skips_missing_sources_and_writes_index(tmp_path):
    from es.paper_tables import build_all
    src, paper = _sources(tmp_path)
    status = build_all(src, paper / "generated" / "tables")
    assert status["tab_panel"] == "built" and status["tab_pr"].startswith("skipped")
    assert (paper / "generated" / "tables" / "tab_panel.tex").exists()
    assert "tab_pr" in (paper / "generated" / "tables" / "INDEX.md").read_text()


def test_implementation_tables_render_null_as_pending(tmp_path):
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    assert "\\PENDINGDEV" in BUILDERS["tab_implementation"][0](src)
    assert "da\\_lmp\\_lag\\_48h" in BUILDERS["tab_blocks_features"][0](src)
    assert "2023-04-13" in BUILDERS["tab_coverage"][0](src)


# =====================================================================================================================
# Added by the implementer (Task 4 rulings 65, 68, 69, 72 and the coordinator's rotation correction). Everything below
# the seven specified tests above. The "full" fixture has every source file of the contract, with real market and block
# names and realistic magnitudes (six-digit dollar values, two-digit failure counts), so that each builder runs and
# the widths of the real table shapes can be checked.
# =====================================================================================================================
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from es.paper_macros import PENDING, PENDING_TEXT  # noqa: E402

LEGACY_IDS = ("24H1-25H1", "24H2-25H2", "24H2-25H2-train2023", "2024-2025", "25H1-26H1", "sealed26H1", "frozen-26H1")
ANALOG_IDS = ("AQ3", "AQ4", "AQ1", "AQ2", "AQ3F")
LC_IDS = ("LC-full-q2023", "LC-full-q2324", "LC-h1-q2023", "LC-h1-q2324")
ALL_BLOCKS = ("G6", "LEV6", "MID3", "DART3", "MIDDART6", "ROLL15", "ROLL_DISP9", "ROLL_LOC6", "DART18", "G1_24")
ECON_ANALYSES = ("AQ3", "AQ4", "AQ1", "AQ2", "2024-2025", "24H2-25H2", "25H1-26H1", "AQ3F")      # es.run_economics runs ANALOGS (with AQ3F) and 3 legacy
NODES = [(m, f"{m}_node_{i}") for m, k in (("NYISO", 10), ("ISONE", 9), ("MISO", 8), ("CAISO", 6)) for i in range(k)]
FEES = ("calibrated", "0.0", "0.1", "0.25", "0.5", "1.0", "stress")
POSITIONS = ("premium", "inc", "dec", "premium[short_spikes]", "premium[long_spikes]")
IMPL_RULES = {
    "label": "spike = |DART| >= q, with q the node's 95th percentile of |DART| on the analysis's label-training rows (linear interpolation); NaN |DART| stays unlabelled (es.labels.label_training_q)",
    "cutoff_P1": "tau = numpy.quantile(calibration scores, 1 - alpha), linear interpolation; alert when score >= tau (es.policies.p1_fixed)",
    "alpha": 0.05, "alpha_sweep": [0.01, 0.025, 0.05, 0.1],
    "failure_bands": {"0.5_2": [0.5, 2.0], "0.67_1.5": [0.6666666666666666, 1.5], "0.8_1.25": [0.8, 1.25]},
    "zero_alert_floor": "R floored at 1/(2 n alpha) for log statistics; zero-activity cells counted as failures (es.policies.activity)",
    "feasibility": "a node is fitted only with at least 20 positive training labels and two classes (es.scores._gate)",
    "missing_data": "nothing is imputed. Each ingester inner-merges the day-ahead and real-time series on timestamp and node, so an hour with a price on one side only is absent from the panel (v2/ingest_nyiso.py:104, v2/ingest_isone.py:97, v2/ingest_miso.py:101, v2/ingest_caiso.py:79). A lag feature (lags of 48, 72 and 168 h) takes the observation at exactly that hour before the row on a complete hourly grid, so it is missing when that hour is absent or its price is missing; a rolling statistic (windows of 24, 72 and 168 h, ending 48 h before the row) is computed when at least max(2, w // 2) of its w hours are observed (24 h: 12, 72 h: 36, 168 h: 84) and is missing otherwise; a row with any missing feature is then dropped for that block (es.scores._prepare)",
    "deduplication": "drop_duplicates on (market, node, timestamp_utc), keep first (es.data.load_dev_raw); the count is in coverage.csv and its split by market in coverage_notes",
    "information_policy": "realized lags 48, 72 and 168 h (v2.panel_prep.G1_LAGS); rolling windows of 24, 72 and 168 h (G1_ROLL_WINDOWS) end at target minus 48 h (set by the information policy, which keeps every input known before the day-ahead bidding deadline, §3.3); feature lookback 216 h = 168 h + 48 h (the longest rolling window plus the shift, §3.3; v2.panel_prep.add_g1_features)",
    "calendar": "the calendar flags and encodings use the market-local clock, which is the settlement clock; peak hour = local hours 7 to 22; weekend = Saturday and Sunday; each cyclic feature is the sin and cos of 2 pi x / period, with periods hour 24, day of week 7, day of year 365.25, month 12 (v2.panel_prep.add_features, v2/panel_prep.py:43)",
    "seeds": {"primary": 42, "all": [0, 1, 7, 13, 42], "extra_seeds_scope": "the 4 extra seeds (0, 1, 7, 13) run only for the seed-sensitivity blocks (G6, LEV6, DART3, ROLL_DISP9, DART18) of 2 analyses (2024-2025, 25H1-26H1), XGBoost, P1 at alpha 0.05; every other run uses the primary seed 42 (es.run_legacy.run_plan, es/run_legacy.py:530)"},
    "time_zones": {"NYISO": "America/New_York", "ISONE": "America/New_York", "MISO": "Etc/GMT+5", "CAISO": "America/Los_Angeles"},
    "miso_note": "MISO settles in fixed EST (Etc/GMT+5)",
    "window_conventions": "UTC half-open [start, end): legacy (7 analyses), label_control (4 analyses); local settlement dates [start, end]: analog (5 analyses), sealed (4 analyses); the two conventions are never pooled (§3.4)",
    "dst": "spring-forward days have 23 hours; on fall-back days CAISO keeps both 01:00 rows, NYISO and ISO-NE drop both (§6.1.1)",
    "sealed_last_dev_date": "2026-06-30", "P4_windows_days": [56, 28], "P5_eta_grid": [0.005, 0.01, 0.02, 0.05],
    "P5_clip": "after each deployment day the alert rate is clipped to [0.001, 0.5]: alpha_(d+1) = min(0.5, max(0.001, alpha_d + eta (alpha - r_d))), with alpha_1 = alpha and r_d the day's realized alert rate (es.policies.p5_self_correcting, es/policies.py:156)",
    "P4_matched": "P4@r*: per node, a bisection of the P4 alert rate on [0.0001, 0.5] with 40 iterations, to the target alert count (that of P1, §3.7.3); es.policies.p4_matched at es/policies.py:174, the bounds at es/policies.py:17",
    "bootstrap": {"replicates": 2000, "mean_block_days": 7, "seed": 20261022, "rotation_min_shift_days": 1},
    "es_fraction_pct": 5.0,
    "fees_calibrated": {"NYISO": {"INC": 0.2, "DEC": 0.15}, "ISONE": {"INC": 0.7, "DEC": 0.7}, "MISO": {"INC": 0.15, "DEC": 0.05}, "CAISO": {"INC": 0.55, "DEC": 0.15}},
    "fees_uniform_grid": [0.0, 0.1, 0.25, 0.5, 1.0], "position_mw": 1,
    "stress_fee": "fees double on stressed days: a node-day is stressed when abs_mean_168h at its first hour is at or above the node's 90th percentile over the training rows, and the fee is then multiplied by 2 in every hour of that day (es.positions.stress_threshold, es/positions.py:73; _stressed_days, es/positions.py:79; attach_pnl, es/positions.py:92)",
    "lambda_grid": [0.0, 0.25, 0.5, 1.0], "strata_primary": ["market", "node", "settle_hour", "wclass", "month"], "strata_sensitivity": ["market", "node", "cal_decile"],
    "monitors": "the monitors are W1_norm, PSI, domain_auc and early_rate; every monitor sees only deployment days 1 to 14: W1_norm, PSI and domain_auc compare the calibration rows with the deployment rows of those days, and early_rate is the alert rate over those days; the outcome they are scored against, |log10 R| and the failure flag, comes from deployment days 15 to the end. The domain classifier is a logistic regression (max_iter = 2000, class_weight = balanced) scored by out-of-fold AUROC over 5-fold GroupKFold blocked by ISO week (es.run_monitors: MONITOR_DAYS, es/run_monitors.py:41; N_FOLDS, es/run_monitors.py:38; domain_auc, es/run_monitors.py:120)",
    "coverage_rule": "a sealed test runs only with at least 28 usable nodes and all four operators; a node needs at least 90% of expected deployment hours (set before the sealed windows were opened, §6.1.1)",
    "spec_fixed": {
        "P4_windows_days": "the 28-day sensitivity window is set by the design of the study (§3.7.3); the 56-day primary is the default of es.policies.p4_rolling",
        "P5_eta_grid": "the grid from which the step size is chosen on the quarterly analogues (§3.7.3); es.policies.p5_self_correcting takes eta as an argument and no code constant holds the grid yet",
        "position_mw": "the standing position of the economic design (§3.7.1); es/positions.py states it in its docstring, not as a constant",
        "information_shift_h": "48 h rolling shift, set by the information policy (§3.3); checked against min(v2.panel_prep.G1_LAGS)",
        "feature_lookback_h": "216 h = 168 h + 48 h, the longest rolling window plus the shift (§3.3); checked against v2.panel_prep.G1_ROLL_WINDOWS",
        "coverage_rule": "28 usable nodes and 90% of expected deployment hours, set before the sealed windows were opened (§6.1.1)"}}
COVERAGE = [
    {"series": "NYISO", "nodes": 10, "hourly_rows": 306390, "first_local_date": "2023-01-01", "last_local_date": "2026-06-30"},
    {"series": "ISONE", "nodes": 9, "hourly_rows": 275769, "first_local_date": "2023-01-01", "last_local_date": "2026-06-30"},
    {"series": "MISO", "nodes": 8, "hourly_rows": 245184, "first_local_date": "2023-01-01", "last_local_date": "2026-06-30"},
    {"series": "CAISO", "nodes": 6, "hourly_rows": 169200, "first_local_date": "2023-04-13", "last_local_date": "2026-06-30"},
    {"series": "CAISO FMM (settlement leg)", "nodes": 6, "hourly_rows": 156096, "first_local_date": "2023-07-13", "last_local_date": "2026-06-30"},
    {"series": "NYISO integrated RT LBMP (settlement leg)", "nodes": 15, "hourly_rows": 459615, "first_local_date": "2023-01-01", "last_local_date": "2026-06-30"}]
COVERAGE_NOTES = [
    "Duplicates: 390 duplicate rows, all at NYISO month boundaries (each at 00:00 local on the 1st of a month: 39 month starts, 10 nodes); per-market counts from es.data.load_dev_raw: NYISO 390, ISONE 0, MISO 0, CAISO 0. The first copy is kept.",
    "NYISO panel: both rows of the repeated 01:00 local hour are absent on each of the 3 fall-back days (2023-11-05, 2024-11-03, 2025-11-02), 6 hours per node. The July downloaders drop them (§6.1.1).",
    "NYISO integrated RT LBMP (settlement leg): 15 names, 5 of them not panel nodes (H Q, MILLWD, NPX, O H, PJM)."]
MIXED_MODEL = """         Mixed Linear Model Regression Results
===========================================================================
Model:            MixedLM  Dependent Variable:  log_error
No. Observations: 1320     Method:              REML
No. Groups:       4        Scale:               0.0112
Min. group size:  264      Log-Likelihood:      1033.5821
Max. group size:  396      Converged:           Yes
Mean group size:  330.0
---------------------------------------------------------------------------
                        Coef.   Std.Err.     z     P>|z|   [0.025   0.975]
---------------------------------------------------------------------------
Intercept               0.121    0.034     3.560   0.000    0.054    0.188
block[T.LEV6]           0.087    0.011     7.910   0.000    0.065    0.109
block[T.ROLL_DISP9]     0.012    0.011     1.090   0.276   -0.010    0.034
block[T.DART18]         0.044    0.011     4.000   0.000    0.022    0.066
Group Var               0.003    0.021
===========================================================================
"""


def _impl_json(rules=None, coverage=COVERAGE, notes=COVERAGE_NOTES) -> dict:
    from es.blocks import BLOCKS, G6, PAPER_BLOCK_NAMES
    blocks = {"calendar_G6": list(G6)}
    for key, feats in BLOCKS.items():
        if key != "G6":
            blocks[key] = {"paper_name": PAPER_BLOCK_NAMES[key], "n_price_features": len(feats), "price_features": list(feats),
                           "n_model_features": len(G6) + len(feats)}
    return {"xgboost": {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
                        "reg_lambda": 2.0, "eval_metric": "aucpr", "random_state": 42,
                        "scale_pos_weight": "n_negative / n_positive of the node's feature-complete training rows (src/transfer.py:205); every other parameter at the xgboost 3.0.2 default except n_jobs = 4 (the thread count)"},
            "logistic": {"penalty": "l2", "C": 1.0, "class_weight": "balanced", "max_iter": 2000, "solver": "lbfgs"},
            "scaler": {"centre": "median of the training rows", "spread": "1.4826 x median absolute deviation of the training rows",
                       "fallback": "training standard deviation (n - 1) when 1.4826 x MAD <= 1e-6, then 1.0 where that is undefined or zero",
                       "fit_on": "training rows only, never calibration or deployment rows", "applied_to": "every row of the node"},
            "windows": [], "blocks": blocks, "rules": IMPL_RULES if rules is None else rules,
            "versions": {"python": "3.12.7", "numpy": "1.26.4", "pandas": "2.2.2", "xgboost": "3.0.2", "scikit-learn": "1.5.1",
                         "pyarrow": "16.1.0", "scipy": "1.13.1", "statsmodels": "0.14.2", "matplotlib": "3.9.2"},
            "coverage": coverage, "coverage_notes": notes}


def _full_sources(tmp_path: Path):
    """The brief's fixture plus every other source file of the contract, with realistic magnitudes."""
    src, paper = _sources(tmp_path)
    out = src.out_dir
    rng = np.random.default_rng(7)
    (out / "paper" / "implementation.json").write_text(json.dumps(_impl_json()))
    pd.DataFrame({"market": ["CAISO", "ISONE", "MISO", "NYISO"], "nodes": [6, 9, 8, 10], "hourly_rows": [169200, 275769, 245184, 306390],
                  "hubs": [3, 1, 8, 0], "zones": [3, 8, 0, 10], "prevalence_24_q2023": [0.0316, 0.0485, 0.0700, 0.0574],
                  "prevalence_24_q2023H1": [0.048, 0.0432, 0.0818, 0.0532]}).to_csv(out / "legacy" / "table1_panel.csv", index=False)
    quarters = [f"{y}Q{q}" for y in (2023, 2024, 2025) for q in (1, 2, 3, 4)] + ["2026Q1", "2026Q2"]
    vol = [{"period": qq, "kind": "quarter", "da_std": float(rng.uniform(10, 45)), "dart_std": float(rng.uniform(15, 40)),
            "dart_p95": float(rng.uniform(19, 66)), "da_std_pooled": 30.1, "dart_std_pooled": 33.3, "dart_p95_pooled": float(rng.uniform(22, 66))}
           for qq in quarters]
    vol += [{"period": "2023-2026", "kind": "period", "da_std": 25.0, "dart_std": 25.0, "dart_p95": 40.0, "da_std_pooled": 30.0,
             "dart_std_pooled": 30.0, "dart_p95_pooled": 40.0}]
    pd.DataFrame(vol).to_csv(out / "legacy" / "fig1_volatility.csv", index=False)

    rows = []

    def add(a, b, e, seed, policy, alpha):
        rows.append({"analysis": a, "block": b, "estimator": e, "seed": seed, "policy": policy, "alpha": alpha,
                     "failures": int(rng.integers(0, 34)), "n_nodes": 33, "median_R": float(rng.uniform(0.4, 1.6)),
                     "median_log_error": float(rng.uniform(0, 0.5)), "median_auprc": float(rng.uniform(0.05, 0.4)),
                     "auprc_lift": 1.5, "prevalence": 0.07, "n_zero_activity": 0,
                     "failures_band_0.67_1.5": int(rng.integers(0, 34)), "failures_band_0.8_1.25": int(rng.integers(0, 34)),
                     "failures_NYISO": 1, "failures_ISONE": 1, "failures_MISO": 1, "failures_CAISO": 0})
    for a in LEGACY_IDS:
        for b in ALL_BLOCKS:
            for e in ("xgb", "logit"):
                add(a, b, e, 42, "P1", 0.05)
    for a in ("2024-2025", "25H1-26H1"):
        for b in ("G6", "LEV6", "DART3", "ROLL_DISP9", "DART18"):
            for sd in (0, 1, 7, 13):
                add(a, b, "xgb", sd, "P1", 0.05)
        for b in ("LEV6", "DART18", "G1_24", "DART3"):
            for al in (0.01, 0.025, 0.1):
                add(a, b, "xgb", 42, "P1", al)
        add(a, "G1_24", "xgb", 42, "P2_all_train", 0.05)
    pd.DataFrame(rows).to_csv(out / "legacy" / "summary.csv", index=False)

    # economics
    econ = []
    for aid in ECON_ANALYSES:
        def row(policy, comparator, position, fee):
            net = float(rng.normal(60000, 90000))
            return {"analysis": aid, "policy": policy, "comparator": comparator, "position": position, "fees": fee,
                    "LA": abs(net) * 2.1, "PG": abs(net) * 1.1, "NET": net, "SELC": float(rng.uniform(-0.02, 0.15)),
                    "SELC_decile": 0.1, "SNET": float(rng.normal(8, 20)), "ESred": float(rng.uniform(500, 9000)),
                    "CAP": float(rng.uniform(0.05, 0.3)), "n_days": 92, "rotation_p": 0.04, "settle_approx_share": 0.0}
        for pol in ("P1", "P4", "P5", "P6", "P4@r*"):
            for pos in POSITIONS:
                for fee in FEES:
                    econ.append(row(pol, "none", pos, fee))
        for comp in ("calendar@count", "VOL@count", "VOL_P1", "VOL_P4", "VOL_P6") + tuple(f"curve_{p}_a{al}" for p in ("P1", "P6") for al in (0.01, 0.025, 0.05, 0.1)):
            for pos in ("premium", "inc", "dec"):
                for fee in FEES:
                    econ.append(row("comparator", comp, pos, fee))
    pd.DataFrame(econ).to_csv(out / "economics" / "summary.csv", index=False)
    stats = list(("SELC", "NET", "E2:SNET", "E3:ESred", "E4:SELC_decile", "E4:SELC|calendar@count", "E4:SELC|VOL@count",
                  "E4:dSELC|calendar@count", "E4:dSELC|VOL@count",
                  "E4:ESred|calendar@count", "E4:ESred|VOL@count", "E4:dESred|calendar@count", "E4:dESred|VOL@count"))
    stats += [f"E1:{k}:{p}" for k in ("NET", "ESred", "J_0.0", "J_0.25", "J_0.5", "J_1.0") for p in ("timing", "level", "total")]
    ci = []
    for aid in ECON_ANALYSES + tuple(f"{x}|ex-CAISO" for x in ECON_ANALYSES) + ("AQ3+AQ4 (rehearsal of the pooled estimands)",):
        for s in stats:
            money = s in ("NET", "E3:ESred") or s.startswith(("E1:", "E4:ESred", "E4:dESred"))
            est = float(rng.normal(60000, 80000)) if money else float(rng.normal(4, 3)) if s == "E2:SNET" else float(rng.uniform(0.0, 0.2))
            width = abs(est) * 0.8 + (1000 if money else 1)
            ci.append({"analysis": aid, "stat": s, "estimate": est, "lo": est - width, "hi": est + width,
                       "approximate": "ESred" in s or s.startswith("E1:J_0.5"), "n_windows": 1})
    pd.DataFrame(ci).to_csv(out / "economics" / "bootstrap_ci.csv", index=False)
    windows = {"AQ3": 92, "AQ4": 92, "AQ1": 90, "AQ2": 91, "2024-2025": 365, "24H2-25H2": 184, "25H1-26H1": 181}
    rot = [{"window": w, "t_obs": float(rng.uniform(-1, 4)), "p": 1.0 / d * float(rng.integers(1, 20)), "n_null": d - 1,
            "selc_obs": float(rng.uniform(0, 0.15)), "retained_min": 0.4, "retained_mean": 0.62} for w, d in windows.items()]
    rot.append({"window": "AQ3+AQ4 (rehearsal of the pooled H-E test)", "t_obs": 2.9, "p": 0.0123, "n_null": 92 * 92 - 1,
                "selc_obs": 0.081, "retained_min": 0.35, "retained_mean": 0.6})
    pd.DataFrame(rot).to_csv(out / "economics" / "rotation.csv", index=False)
    pd.DataFrame([{"analysis": a, "sharpe_alerted": float(rng.uniform(-0.2, 0.2)), "sharpe_not_alerted": float(rng.uniform(-0.05, 0.1))}
                  for a in ECON_ANALYSES]).to_csv(out / "economics" / "volmanagement_check.csv", index=False)

    # inference
    fam = {"audit robustness": 20, "economic robustness": 31, "drift monitors": 24}
    pd.DataFrame([{"family": f, "test": f"{f}: contrast {i} of the {f} family", "estimate": float(rng.normal(0.1, 0.2)),
                   "p": float(rng.uniform(0, 1) ** 2), "bh_reject": bool(rng.uniform() < 0.2)}
                  for f, k in fam.items() for i in range(1, k + 1)]).to_csv(out / "inference" / "bh.csv", index=False)
    pd.DataFrame([{"dropped": m, "analysis": a, "block": b, "failures": int(rng.integers(0, 30)), "median_log_error": float(rng.uniform(0, 0.5))}
                  for m in MARKETS_ORDER for a in ("2024-2025", "25H1-26H1", "AQ3", "AQ4") for b in ("LEV6", "DART3", "ROLL_DISP9", "DART18")]) \
        .to_csv(out / "inference" / "looo.csv", index=False)
    (out / "inference" / "mixed_model.txt").write_text(MIXED_MODEL)

    # labels and precision/recall
    pr = []
    for a in LEGACY_IDS + ANALOG_IDS:
        for b in ALL_BLOCKS:
            for e in ("xgb", "logit"):
                for m, n in NODES[::5]:
                    pr.append({"analysis": a, "block": b, "estimator": e, "market": m, "node": n,
                               "precision_q": float(rng.uniform(0.05, 0.5)), "recall_q": float(rng.uniform(0.02, 0.3)),
                               "precision_eval": float(rng.uniform(0.05, 0.5)), "recall_eval": float(rng.uniform(0.02, 0.3)),
                               "precision_lift": float(rng.uniform(0.8, 6)), "prevalence_eval": 0.05, "n_nan_outcome": 0})
    pd.DataFrame(pr).to_csv(out / "labels_pr" / "pr_all.csv", index=False)
    pd.DataFrame([{"analysis": a, "block": b, "failures": int(rng.integers(0, 34)), "n_nodes": 33, "median_R": float(rng.uniform(0.4, 1.6))}
                  for a in LC_IDS for b in ("LEV6", "DART18", "G1_24", "DART3")]).to_csv(out / "labels_pr" / "label_control_2x2.csv", index=False)
    pd.DataFrame([{"analysis": a, "block": b, "market": m, "node": n, "precision_q": float(rng.uniform(0.05, 0.5)),
                   "recall_q": float(rng.uniform(0.02, 0.3)), "precision_eval": float(rng.uniform(0.05, 0.5)),
                   "recall_eval": float(rng.uniform(0.02, 0.3)), "precision_lift": float(rng.uniform(0.8, 6)),
                   "prevalence_eval": 0.05, "n_nan_outcome": 0}
                  for a in LC_IDS for b in ("LEV6", "DART18", "G1_24", "DART3") for m, n in NODES[::6]]) \
        .to_csv(out / "labels_pr" / "label_control_pr.csv", index=False)

    # monitors
    (out / "monitors").mkdir(exist_ok=True)
    mon = [{"analysis": a, "block": b, "monitor": m, "spearman": float(rng.uniform(-0.3, 0.7)),
            "spearman_p_one_sided": float(rng.uniform(0, 1) ** 2), "auroc_fail": float(rng.uniform(0.3, 0.9)), "n_used": 33,
            "calib_days": days, "placebo_median": float("nan") if m == "early_rate" else float(rng.uniform(0.1, 0.9)),
            "placebo_n": float("nan") if m == "early_rate" else 33}                       # es.run_monitors.EVAL_COLUMNS (Rulings 73, 83)
           for a, days in (("25H1-26H1", 181), ("AQ3", 92), ("AQ4", 92)) for b in ALL_BLOCKS[1:] for m in ("W1_norm", "PSI", "domain_auc", "early_rate")]
    pd.DataFrame(mon).to_csv(out / "monitors" / "monitor_eval.csv", index=False)

    # analogues
    act = []
    for a in ANALOG_IDS:
        for b in ALL_BLOCKS:
            for e in ("xgb", "logit"):
                for pol in ("P1", "P4_56", "P4_28", "P6"):
                    for m, n in NODES[::5]:
                        r = float(rng.uniform(0.2, 3))
                        act.append({"analysis": a, "block": b, "estimator": e, "policy": pol, "market": m, "node": n, "R": r,
                                    "log_error": abs(float(np.log10(r))), "fail_0.5_2": bool(r < 0.5 or r > 2)})
    pd.DataFrame(act).to_csv(out / "analogs" / "activity_all.csv", index=False)
    pd.DataFrame([{"analysis": a, "eta": eta, "median_log_error": float(rng.uniform(0, 0.4))} for a in ANALOG_IDS for eta in (0.005, 0.01, 0.02, 0.05)]) \
        .to_csv(out / "analogs" / "eta_selection.csv", index=False)
    (out / "analogs" / "protocol_inputs.json").write_text(json.dumps(REAL_PROTOCOL_INPUTS))          # the real outputs of the analogue study, copied
    (out / "analogs" / "simulation.csv").write_text(REAL_SIMULATION_CSV)
    return src, paper


MARKETS_ORDER = ("NYISO", "ISONE", "MISO", "CAISO")


@pytest.fixture
def full(tmp_path):
    return _full_sources(tmp_path)


# ---------------------------------------------------------------------------------------------- Ruling 65: coverage notes
def _write_impl(src, **kw):
    (src.out_dir / "paper" / "implementation.json").write_text(json.dumps(_impl_json(**kw)))


def _fresh(src):
    """A new Sources over the same directories (Sources caches every file it has read)."""
    from es.paper_macros import Sources
    return Sources(src.out_dir, src.paper_dir, src.docs_sealed)


def test_coverage_table_lists_the_coverage_notes_under_the_table(tmp_path):
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    _write_impl(src, notes=["Duplicates: 390 duplicate rows (the NYISO month boundaries); per-market counts from es.data.load_dev_raw.",
                            "NYISO panel: 2 hours absent (§6.1.1)."])
    tex = BUILDERS["tab_coverage"][0](src)
    assert tex.count("\\footnotetext{") == 2
    assert tex.index("\\end{tabular}") < tex.index("the NYISO month boundaries")             # the notes follow the table
    assert "es.data.load\\_dev\\_raw" in tex                                              # plain text escaped
    assert "Section~\\ref{sec:sealed-windows}" in tex and "6.1.1" not in tex and "\\S{}" not in tex   # plan 2C T2b: the typed pointer names its section (now of Online Resource 1)
    _write_impl(src, notes=[])
    assert "\\footnotetext" not in BUILDERS["tab_coverage"][0](_fresh(src))


def test_coverage_table_masks_line_numbers_in_notes(tmp_path):
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    _write_impl(src, notes=["counted in es/data.py:88 and v2/ingest_nyiso.py:104-110"])
    tex = BUILDERS["tab_coverage"][0](src)
    assert "es/data.py" in tex and "v2/ingest\\_nyiso.py" in tex and not __import__("re").search(r"\.py:\d", tex)


# ---------------------------------------------------------------------------------------------- Ruling 65 and the coordinator: tab_implementation
IMPL_RULES_SMALL = {
    "alpha": 0.05, "P4_windows_days": [56, 28], "position_mw": 1,
    "failure_bands": {"0.5_2": [0.5, 2.0], "0.67_1.5": [0.6666666666666666, 1.5]},
    "calendar": "flags use the market-local clock (v2.panel_prep.add_features, v2/panel_prep.py:43)",
    "P5_clip": "the alert rate is clipped to [0.001, 0.5] (es.policies.p5_self_correcting, es/policies.py:156)",
    "P4_matched": "bisection on [0.0001, 0.5], 40 iterations; es.policies.p4_matched at es/policies.py:174, the bounds at es/policies.py:17",
    "stress_fee": "fees double on stressed days (es.positions.stress_threshold, es/positions.py:73; attach_pnl, es/positions.py:92-99)",
    "monitors": "the monitors are W1_norm and PSI (es.run_monitors: MONITOR_DAYS, es/run_monitors.py:41)",
    "spec_fixed": {"P4_windows_days": "the 28-day sensitivity window is set by the design of the study (§3.7.3); the 56-day primary is the default",
                   "position_mw": "the standing position of the economic design (§3.7.1)"}}


def test_implementation_table_separates_code_rules_from_spec_fixed_values(tmp_path):
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    _write_impl(src, rules=IMPL_RULES_SMALL)
    tex = BUILDERS["tab_implementation"][0](src)
    code_header, spec_header = tex.index("Read from the code"), tex.index("Fixed by the design of the study")
    assert code_header < spec_header
    for label in ("calendar features", "P5 rate bounds", "P4 at matched count", "stress fee", "drift monitors"):
        assert code_header < tex.index(label) < spec_header, label                         # the new rules sit under the code header
    spec_part = tex[spec_header:]
    assert "\\mainref{data-econ-policies}" in spec_part and "\\mainref{data-econ-positions}" in spec_part      # each spec-fixed entry carries its section of the paper, by title (plan 2C T2b)
    assert "\\S{}" not in tex and "\u00a7" not in tex                                       # and no typed section pointer
    assert tex.count("P4 windows") == 1 and "1 MW" in spec_part                             # a spec-fixed value is listed once, in the spec part
    assert "positions.stress\\_threshold" in tex                                           # plain text went through tex_escape


def test_implementation_table_masks_line_numbers_but_keeps_file_paths(tmp_path):
    import re
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    _write_impl(src, rules=IMPL_RULES_SMALL)
    tex = BUILDERS["tab_implementation"][0](src)
    assert not re.search(r"\.py:\d", tex)
    for path in ("v2/panel\\_prep.py", "es/policies.py", "es/positions.py", "es/run\\_monitors.py"):
        assert path in tex, path
    assert "es/policies.py, the bounds at es/policies.py)" not in tex                       # (the p4_matched sentence has no closing paren)
    assert "es/policies.py, the bounds at es/policies.py" in tex                            # both citations of one rule survive without their lines


def test_implementation_table_renders_a_citation_that_is_the_whole_value_as_the_path(tmp_path):
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    rules = dict(IMPL_RULES_SMALL, calendar="es/policies.py:156")
    _write_impl(src, rules=rules)
    row = next(line for line in BUILDERS["tab_implementation"][0](src).splitlines() if "calendar features" in line)
    assert row.endswith("& \\raggedright\\hyphenpenalty=10000 es/policies.py \\\\")


def test_implementation_table_formats_numbers_and_keeps_unlisted_rules(tmp_path):
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    _write_impl(src, rules=dict(IMPL_RULES_SMALL, some_new_rule="a new rule text", miso_note="MISO settles in fixed EST",
                                some_ratio=1 / 3, time_zones={"MISO": "Etc/GMT+5"}))
    tex = BUILDERS["tab_implementation"][0](src)
    assert "0.333333" in tex and "0.3333333333333333" not in tex                           # no raw 16-digit floats
    assert "[0.5, 2], [0.67, 1.5]" in tex and "0.6666666666666666" not in tex               # the bands as the paper writes them
    assert "some\\_new\\_rule" in tex and "a new rule text" in tex                          # a rule the table has no label for is not dropped
    assert "miso\\_note" not in tex and "MISO settles in fixed EST" in tex                  # the MISO note joins the settlement clocks row


def test_implementation_table_requires_a_section_reference_in_every_spec_fixed_entry(tmp_path):
    import pytest as _pytest
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    _write_impl(src, rules=dict(IMPL_RULES_SMALL, spec_fixed={"position_mw": "fixed somewhere"}))
    with _pytest.raises(LookupError, match="position_mw"):
        BUILDERS["tab_implementation"][0](src)


# ---------------------------------------------------------------------------------------------- coordinator: rotation columns
def test_rotation_table_reads_t_and_selc_from_separate_columns(tmp_path):
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    pd.DataFrame([{"window": "AQ3", "t_obs": 2.4567, "p": 1 / 92, "n_null": 91, "selc_obs": 0.0345, "retained_min": 0.41, "retained_mean": 0.62},
                  {"window": "AQ3+AQ4 (rehearsal of the pooled H-E test)", "t_obs": -0.3, "p": 0.0123, "n_null": 92 * 92 - 1,
                   "selc_obs": 0.081, "retained_min": float("nan"), "retained_mean": float("nan")}]) \
        .to_csv(src.out_dir / "economics" / "rotation.csv", index=False)
    tex = BUILDERS["tab_rotation_dev"][0](src)
    assert "AQ3: 2025-Q3 & 3.5\\% & 2.46 & 0.011 & 91 & 41\\% / 62\\% \\\\" in tex          # T is dec2, SELC is pct1, never the other way round
    assert "8{,}463" in tex and "\\textminus{}0.30" in tex and "8.1\\%" in tex
    assert tex.count(" -- \\\\") == 1                                                    # no alerts, no retained share
    header = tex.split("\\midrule")[0]
    for word in ("SELC", "$T$", "$p$", "rotations", "retained"):
        assert word in header, word


def test_rotation_table_needs_the_columns_of_the_contract(tmp_path):
    import pytest as _pytest
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    pd.DataFrame([{"window": "AQ3", "t_obs": 2.0, "p": 0.1, "n_null": 91}]).to_csv(src.out_dir / "economics" / "rotation.csv", index=False)
    with _pytest.raises(LookupError, match="selc_obs"):
        BUILDERS["tab_rotation_dev"][0](src)


# ---------------------------------------------------------------------------------------------- tab_registration on the real outputs (controller addendum)
import copy  # noqa: E402
import io  # noqa: E402

MARK_B = "\\textsuperscript{b}"
HOLM, WHOLE = "reject_0.0166667", "reject_0.05"


def _real_simulation() -> pd.DataFrame:
    return pd.read_csv(io.StringIO(REAL_SIMULATION_CSV))


def _write_registration(src, sim=None, protocol=None):
    """The real analogue outputs (copied above) in the output tree of ``src``; ``sim`` (a DataFrame) and ``protocol`` (a dict) replace them."""
    folder = src.out_dir / "analogs"
    folder.mkdir(exist_ok=True)
    (_real_simulation() if sim is None else sim).to_csv(folder / "simulation.csv", index=False)
    (folder / "protocol_inputs.json").write_text(json.dumps(copy.deepcopy(REAL_PROTOCOL_INPUTS) if protocol is None else protocol))


def _protocol(**changes):
    p = copy.deepcopy(REAL_PROTOCOL_INPUTS)
    p.update(changes)
    return p


def _hypothesis_rows(tex):
    """{first cell: the five cells} of the rows of the registration table (the sub-rows of reasons have one cell)."""
    return {r[0]: r for r in _rows(tex.split("\\medskip")[0]) if len(r) == 5 and r[0] not in ("test",)}


def _top(sim, test, col, laws=None, strata="primary"):
    """The row of ``test`` with the highest ``col`` among the rows of the primary strata (or of none) and of ``laws``, from the copied file."""
    rows = sim[sim["test"] == test]
    rows = rows[rows["strata"].isna() | (rows["strata"] == strata)]
    if laws is not None:
        rows = rows[rows["law"].isin(laws)]
    return rows.loc[rows[col].idxmax()]


def test_the_registration_cells_are_re_derived_from_the_real_file(tmp_path):
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    _write_registration(src)
    rows = _hypothesis_rows(BUILDERS["tab_registration"][0](src))
    he, hb1, hb2 = rows["H-E (pooled A$\\cup$B)"], rows["H-B1"], rows["H-B2"]
    assert he[1:] == ["confirmatory", "0.023 (heteroskedastic, fitted)", "0.070 (heteroskedastic, fitted)", "0.093" + MARK_B]
    assert hb1[1:] == ["estimand\\textsuperscript{a}", "0.130 (absolute level, common drift 0.5, upper)", "0.193 (plain DART lags, common drift 0.5, fitted)", "0.280"]
    assert hb2[1:] == ["estimand\\textsuperscript{a}", "0.127 (plain DART lags, common drift 0.5, fitted)", "0.193 (plain DART lags, common drift 0.5, fitted)", "0.050"]
    # the same numbers from the rows of the copied file, without the module: the rows that the registration rule reads
    sim = _real_simulation()
    pooled = [_top(sim, "H-E pooled A+B, size", col) for col in (HOLM, WHOLE)]
    assert [f"{r[c]:.3f}" for r, c in zip(pooled, (HOLM, WHOLE))] == ["0.023", "0.070"] and {(r["scenario"], r["persistence"]) for r in pooled} == {("heteroskedastic", "fit")}
    for cells, laws in ((hb1, ["DART3", "LEV6"]), (hb2, ["DART3", "ROLL_DISP9"])):
        worst = [_top(sim, "H-B size", col, laws) for col in (HOLM, WHOLE)]
        assert [f"{r[c]:.3f}" for r, c in zip(worst, (HOLM, WHOLE))] == [cells[2][:5], cells[3][:5]]
        assert len(sim[(sim["test"] == "H-B size") & sim["law"].isin(laws)]) == 12                    # k = 12 rows each
    assert hb1[2].startswith("0.130") and _top(sim, "H-B size", HOLM, ["DART3", "LEV6"])["law"] == "LEV6"     # the governing row is named
    assert [round(float(_top(sim, f"{h} power", HOLM)[HOLM]), 3) for h in ("H-B1", "H-B2")] == [0.28, 0.05]
    assert round(float(sim[sim["test"] == "H-E pooled A+B, power"][HOLM].iloc[0]), 3) == 0.093


def test_the_stored_family_is_what_the_rule_gives_for_the_stored_simulation():
    """The family that the table shows is the rule's own: applying es.run_analogs.registration_family to the copied rows and placebo
    pairings gives the copied family."""
    import math
    from es.run_analogs import registration_family

    def same(a, b):
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
        if isinstance(a, list):
            return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
        if isinstance(a, float):
            return (math.isnan(a) and math.isnan(b)) or abs(a - b) < 1e-12
        return a == b
    assert same(registration_family(_real_simulation(), REAL_PROTOCOL_INPUTS["placebo"]), REAL_PROTOCOL_INPUTS["family"])


def test_the_decision_the_approximate_flag_and_the_reasons_come_from_the_family(tmp_path):
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    _write_registration(src)
    tex = BUILDERS["tab_registration"][0](src)
    table = tex.split("\\medskip")[0]
    assert "\\textit{Reasons for H-B1:} simulated size 0.130 \\textgreater{} 1.5 x 0.0167 (39 of 300 replications, worst of 12 rows: absolute level, common drift 0.5, upper); " in table
    assert "simulated size 0.193 \\textgreater{} 1.5 x 0.0500 (58 of 300 replications, worst of 12 rows: plain DART lags, common drift 0.5, fitted); simulated power 0.28 at 0.05/3 is below 0.5}" in table
    assert "\\textit{Reasons for H-B2:} simulated size 0.127 \\textgreater{} 1.5 x 0.0167 (38 of 300 replications" in table and "simulated power 0.05 at 0.05/3 is below 0.5}" in table
    assert "Reasons for H-E" not in table                                                         # confirmatory with no reason
    assert "x 0.0167 (39 of 300 replications, worst of 12 rows: LEV6" not in tex                 # the ids of the producer are in words
    # flip the family: the table follows it, whatever the rows say
    flipped = _protocol()
    flipped["family"]["H-E"].update(confirmatory=False, reasons=["a reason of the family"], approximate=True)
    flipped["family"]["H-B1"].update(approximate=False)
    _write_registration(src, protocol=flipped)
    rows = _hypothesis_rows(BUILDERS["tab_registration"][0](_fresh(src)))
    assert rows["H-E (pooled A$\\cup$B)"][1] == "estimand\\textsuperscript{a}" and rows["H-B1"][1] == "estimand"
    assert "\\textit{Reasons for H-E:} a reason of the family}" in BUILDERS["tab_registration"][0](_fresh(src))
    quiet = _protocol()
    for hid in ("H-E", "H-B1", "H-B2"):
        quiet["family"][hid]["approximate"] = False
    _write_registration(src, protocol=quiet)
    tex = BUILDERS["tab_registration"][0](_fresh(src))
    assert "Labelled approximate" not in tex and tex.count("textsuperscript{a}") == 0              # no mark, no note for it
    interim = _protocol()
    interim["family"]["H-E"]["interim_approximate"] = True
    _write_registration(src, protocol=interim)
    assert "interim p-value" in BUILDERS["tab_registration"][0](_fresh(src))


def test_the_registration_table_uses_the_words_of_the_manuscript(tmp_path):
    """Ruling 133: the column says "designated as"; the notes speak of the confirmatory family, the designation rule and the designated status;
    nothing the table prints says "registered" or "registration" (its name, tab_registration, is an identifier and is not printed)."""
    import re
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    _write_registration(src)
    interim = _protocol()
    interim["family"]["H-E"]["interim_approximate"] = True                                 # also the note on Test A's interim p-value
    _write_registration(src, protocol=interim)
    tex = BUILDERS["tab_registration"][0](_fresh(src))
    header = next(r for r in _rows(tex.split("\\medskip")[0]) if r[0] == "test")
    assert header[1] == "\\shortstack[l]{designated\\\\as}" and len(header) == 5
    notes = [n.rstrip("\n").removesuffix("}") for n in tex.split("\\footnotetext{")[1:]]
    assert any(n.startswith("The level 0.05/3 is the first Holm level of the three-member draft family. "
                            "The confirmatory family is H-E alone, decided at 0.05, so the size at 0.05 governs H-E.") for n in notes)
    assert any(n.startswith("Each size is the worst case over the rows that the designation rule reads, under the primary strata (") for n in notes)
    assert any(n.startswith("Under the decile strata, the sensitivity (reported only; the designated status rests on the primary strata), the ") for n in notes)
    assert not re.search(r"regist", tex, flags=re.I), re.findall(r".{30}regist.{30}", tex, flags=re.I)


def test_the_he_power_curve_lists_the_grid_and_the_target_row_in_a_table_of_its_own(tmp_path):
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    _write_registration(src)
    tex = BUILDERS["tab_registration"][0](src)
    assert tex.count("\\begin{tabular}") == 2
    curve = tex.split("\\medskip")[1]
    assert "simulated power by coupling $\\rho$ (100 simulations per row)" in curve
    cells = [r for r in _rows(curve) if len(r) == 4]
    assert cells[0] == ["$\\rho$", "\\shortstack[c]{mean\\\\SELC}", "\\shortstack[c]{rate at\\\\0.05/3}", "\\shortstack[c]{rate at\\\\0.05}"]
    assert cells[1:] == [["0.05", "0.11\\%", "0.130", "0.220"], ["0.10", "0.20\\%", "0.360", "0.530"], ["0.20", "0.39\\%", "0.870", "0.960"],
                         ["0.30", "0.59\\%", "1.000", "1.000"], ["0.50", "1.00\\%", "1.000", "1.000"], ["0.70", "1.43\\%", "1.000", "1.000"],
                         ["0.90", "1.88\\%", "1.000", "1.000"], ["0.05" + MARK_B, "0.10\\%", "0.093", "0.210"]]         # the grid, then the target row
    sim = _real_simulation()
    grid = sim[sim["test"] == "H-E pooled A+B, power curve"].sort_values("rho")
    assert [f"{r:.3f}" for r in grid[HOLM]] == [c[2] for c in cells[1:8]] and [f"{r:.3f}" for r in grid[WHOLE]] == [c[3] for c in cells[1:8]]
    note = next(n for n in tex.split("\\footnotetext{") if n.startswith(MARK_B))
    assert "300 simulations at the coupling 0.05" in note and "\\textminus{}0.26\\%" in note and "weakest coupling, not of the target" in note
    assert "stays in the family whatever its power" in note


def test_the_table_reads_the_rows_that_the_rule_reads_and_only_those(tmp_path):
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    sim = _real_simulation()

    def build(frame, protocol=None):
        _write_registration(src, sim=frame, protocol=protocol)
        return _hypothesis_rows(BUILDERS["tab_registration"][0](_fresh(src)))

    def loud(frame, **where):
        frame = frame.copy()
        mask = pd.Series(True, index=frame.index)
        for column, value in where.items():
            mask &= frame[column] == value
        frame.loc[mask, [HOLM, WHOLE]] = 0.9
        return frame
    base = build(sim)
    pooled = "H-E pooled A+B, size"
    assert build(loud(sim, test=pooled, strata="decile"))["H-E (pooled A$\\cup$B)"] == base["H-E (pooled A$\\cup$B)"]          # the decile strata never govern
    assert build(loud(sim, test="H-E single A, size", strata="primary"))["H-E (pooled A$\\cup$B)"] == base["H-E (pooled A$\\cup$B)"]   # H-E's size is the pooled test's
    assert build(loud(sim, test="H-B size", law="ROLL_DISP9", scenario="no_drift", persistence="fit"))["H-B1"] == base["H-B1"]        # not a law of H-B1
    assert build(loud(sim, test="H-B size", law="LEV6", scenario="no_drift", persistence="fit"))["H-B2"] == base["H-B2"]              # not a law of H-B2
    assert build(loud(sim, test="H-B size", law="ROLL_DISP9", scenario="no_drift", persistence="fit"))["H-B2"][2] == "0.900 (rolling dispersion, no drift, fitted)"
    assert build(loud(sim, test="H-B size", law="LEV6", scenario="no_drift", persistence="fit"))["H-B1"][3] == "0.900 (absolute level, no drift, fitted)"
    assert build(loud(sim, test="H-B size", law="DART3", scenario="no_drift", persistence="fit"))["H-B2"][2].startswith("0.900 (plain DART lags")      # the base law is in both
    # the laws are the rule record's: say that H-B1 reads ROLL_DISP9 and the table does
    record = _protocol()
    record["rule"]["laws"]["H-B1"] = ["DART3", "ROLL_DISP9"]
    assert build(sim, record)["H-B1"][2] == "0.127 (plain DART lags, common drift 0.5, fitted)"


def test_the_table_reads_the_levels_from_the_rule_record(tmp_path):
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    protocol = _protocol()
    protocol["rule"]["levels"] = [0.02, 0.06]
    _write_registration(src, sim=_real_simulation().rename(columns={HOLM: "reject_0.02", WHOLE: "reject_0.06"}), protocol=protocol)
    tex = BUILDERS["tab_registration"][0](src)
    header = _rows(tex.split("\\medskip")[0])[0]
    assert "size at 0.06/3," in header[2] and "size at 0.06," in header[3] and "power at\\\\0.06/3" in header[4]
    assert "0.05/3" not in "".join(header)
    assert next(r for r in _rows(tex) if r[0] == "H-B1")[2].startswith("0.130")


def test_a_power_that_failed_is_a_dash_and_a_lower_bound_says_so(tmp_path):
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    sim = _real_simulation()
    sim["error"] = None
    sim["power_lower_bound"] = False
    protocol = _protocol()
    failed = sim["test"] == "H-B2 power"
    sim.loc[failed, [HOLM, WHOLE]] = float("nan")
    sim.loc[failed, "error"] = "target out of range"
    protocol["family"]["H-B2"].update(power=float("nan"), reasons=["simulated power is unavailable (target out of range), counted as below 0.5"])
    sim.loc[sim["test"] == "H-B1 power", "power_lower_bound"] = True
    protocol["family"]["H-B1"]["power_lower_bound"] = True
    _write_registration(src, sim=sim, protocol=protocol)
    tex = BUILDERS["tab_registration"][0](src)
    rows = _hypothesis_rows(tex)
    assert rows["H-B2"][4] == "--" and rows["H-B1"][4] == "$\\ge$ 0.280"
    assert "simulated power is unavailable (target out of range), counted as below 0.5}" in tex and "NaN" not in tex and "nan" not in tex
    sim.loc[failed, "error"] = None                                                                 # a power that failed must say why
    _write_registration(src, sim=sim, protocol=protocol)
    with pytest.raises(ValueError, match="no error text"):
        BUILDERS["tab_registration"][0](_fresh(src))


def test_the_table_stops_when_its_selection_or_its_files_are_not_the_registered_ones(tmp_path):
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    build = lambda: BUILDERS["tab_registration"][0](_fresh(src))               # noqa: E731
    _write_registration(src, protocol={"eta": 0.01, "family": None})                                  # after `--stage analogs`, before `--stage sim`
    with pytest.raises(LookupError, match="--stage sim"):
        build()
    _write_registration(src, protocol=_protocol(stage="analogs"))                                      # a family left over from an earlier run
    with pytest.raises(LookupError, match="--stage sim"):
        build()
    no_rule = _protocol()
    del no_rule["rule"]
    _write_registration(src, protocol=no_rule)
    with pytest.raises(LookupError, match="no complete rule record"):
        build()
    drifted = _protocol()
    drifted["family"]["H-B1"]["k_rows"] = 10
    _write_registration(src, protocol=drifted)
    with pytest.raises(ValueError, match="drifted apart"):
        build()
    other_run = _protocol()
    other_run["family"]["H-B1"]["power"] = 0.5
    _write_registration(src, protocol=other_run)
    with pytest.raises(ValueError, match="not from the same run"):
        build()
    nan_size = _real_simulation()
    nan_size.loc[(nan_size["test"] == "H-B size") & (nan_size["law"] == "LEV6"), HOLM] = float("nan")
    _write_registration(src, sim=nan_size)
    with pytest.raises(ValueError, match="cannot pass on a missing number"):
        build()
    _write_registration(src, sim=_real_simulation().drop(columns=["law"]))
    with pytest.raises(LookupError, match="law"):
        build()
    _write_registration(src, sim=_real_simulation()[lambda f: f["test"] != "H-E pooled A+B, power"])
    with pytest.raises(LookupError, match="target"):
        build()
    unicode_reason = _protocol()
    unicode_reason["family"]["H-B1"]["reasons"] = ["size ≤ nominal"]
    _write_registration(src, protocol=unicode_reason)
    with pytest.raises(ValueError, match="non-ASCII"):
        build()


def test_the_registration_table_is_no_longer_provisional(tmp_path):
    from es.paper_tables import build_all
    src, paper = _sources(tmp_path)
    _write_registration(src)
    status = build_all(src, paper / "generated" / "tables")
    assert status["tab_registration"] == "built"
    line = next(l for l in (paper / "generated" / "tables" / "INDEX.md").read_text().splitlines() if l.startswith("| tab_registration |"))
    assert "provisional" not in line and "rule record" in line and "power curve" in line


def test_the_width_of_the_registration_tables_is_checked_against_the_text(tmp_path, monkeypatch):
    import es.paper_tables as pt
    src, _ = _sources(tmp_path)
    _write_registration(src)
    pt.BUILDERS["tab_registration"][0](src)
    table, curve = pt.GEOMETRY["tab_registration"]
    assert table < pt.TEXT_WIDTH_PT - pt.WIDTH_MARGIN_PT and curve < table
    assert pt.GEOMETRY["tab_registration:reasons"] == pytest.approx((table,))                          # the reasons span the whole table
    monkeypatch.setattr(pt, "REGISTRATION_CM", (1.5, 1.9, 4.2, 4.2))
    with pytest.raises(ValueError, match="more than the 372 pt text width"):
        pt.BUILDERS["tab_registration"][0](_fresh(src))
    monkeypatch.setattr(pt, "REGISTRATION_CM", (0.6, 1.9, 2.9, 2.9))
    with pytest.raises(ValueError, match="widen the column"):
        pt.BUILDERS["tab_registration"][0](_fresh(src))


def test_the_ids_of_the_simulation_and_of_its_reasons_are_put_into_words():
    from es.paper_tables import _id_words, _reason_words
    assert [_id_words(t) for t in ("DART3", "LEV6", "ROLL_DISP9", "no_drift", "common_drift_0.25", "homoskedastic", "fit", "upper", "other_id")] == [
        "plain DART lags", "absolute level", "rolling dispersion", "no drift", "common drift 0.25", "homoskedastic", "fitted", "upper", "other\\_id"]
    assert _reason_words("simulated size 0.130 > 1.5 x 0.0167 (39 of 300 replications, worst of 12 rows: LEV6, common_drift_0.5, upper)") == \
        "simulated size 0.130 > 1.5 x 0.0167 (39 of 300 replications, worst of 12 rows: absolute level, common drift 0.5, upper)"
    assert _reason_words("simulated size 0.050 > 1.5 x 0.0167 (4 of 100 replications: heteroskedastic, fit)") == \
        "simulated size 0.050 > 1.5 x 0.0167 (4 of 100 replications: heteroskedastic, fitted)"
    assert _reason_words("2/2 placebo rejections at 0.0167") == "2/2 placebo rejections at 0.0167"       # a reason without a governing row is as it is


# ---------------------------------------------------------------------------------------------- ESM provenance note (M9)
def test_the_provenance_note_is_a_part_of_its_own_and_needs_no_source(full):
    from es.paper_tables import BUILDERS
    src, _ = full
    fn, needs = BUILDERS["esm_provenance"]
    assert needs == ()
    note = fn(src)
    for word in ("registry.csv", "provenance.csv", "BUILD\\_INFO.json", "source row"):
        assert word in note, word
    assert "\\begin{longtable}" not in note
    assert all("registry.csv" not in BUILDERS[n][0](src)
               for n in ("esm_ledger_legacy", "esm_ledger_analogs", "esm_ledger_economics", "esm_ledger_pr", "esm_ledger_monitors", "esm_ledger_bh"))


def test_the_provenance_note_survives_a_missing_legacy_summary(tmp_path):
    from es.paper_tables import build_all
    src, paper = _sources(tmp_path)
    (src.out_dir / "legacy" / "summary.csv").unlink()                                       # the ledger that used to carry the note
    status = build_all(src, paper / "generated" / "tables")
    assert status["esm_ledger_legacy"].startswith("skipped: missing legacy/summary.csv")
    assert status["esm_provenance"] == "built" and (paper / "generated" / "tables" / "esm_provenance.tex").exists()


# ---------------------------------------------------------------------------------------------- the driver
def test_build_all_removes_the_file_of_a_table_that_is_now_skipped(tmp_path):
    from es.paper_tables import build_all
    src, paper = _sources(tmp_path)
    tables = paper / "generated" / "tables"
    tables.mkdir(parents=True)
    (tables / "tab_pr.tex").write_text("%% GENERATED by es.paper_tables; do not edit.\nold numbers\n")
    (tables / "tab_volatility.tex").write_text("written by hand\n")                       # not a generated file: left alone
    status = build_all(src, tables)
    assert status["tab_pr"].startswith("skipped: missing labels_pr/pr_all.csv") and not (tables / "tab_pr.tex").exists()
    assert (tables / "tab_volatility.tex").read_text() == "written by hand\n"


def test_build_all_builds_the_others_and_names_the_table_that_failed(tmp_path):
    import pytest as _pytest
    from es.paper_tables import build_all
    src, paper = _sources(tmp_path)
    pd.DataFrame([{"window": "AQ3", "t_obs": 2.0, "p": 0.1, "n_null": 91}]).to_csv(src.out_dir / "economics" / "rotation.csv", index=False)
    tables = paper / "generated" / "tables"
    with _pytest.raises(LookupError, match=r"tab_rotation_dev.*selc_obs"):
        build_all(src, tables)
    assert (tables / "tab_panel.tex").exists()                                             # the failure did not stop the run
    index = (tables / "INDEX.md").read_text()
    assert "| tab_rotation_dev | FAILED: LookupError" in index and "| tab_panel | built |" in index
    status = build_all(_fresh(src), tables, raise_on_failure=False)
    assert status["tab_rotation_dev"].startswith("FAILED") and not (tables / "tab_rotation_dev.tex").exists()


def test_main_prints_one_line_per_skipped_table_and_exits_nonzero_on_failure(tmp_path, capsys):
    from es.paper_tables import main
    src, paper = _sources(tmp_path)
    assert main(["--out-dir", str(src.out_dir), "--paper-dir", str(paper)]) == 0
    out = capsys.readouterr().out
    assert "tab_pr: skipped: missing labels_pr/pr_all.csv" in out and "tables built" in out
    pd.DataFrame([{"window": "AQ3", "t_obs": 2.0, "p": 0.1, "n_null": 91}]).to_csv(src.out_dir / "economics" / "rotation.csv", index=False)
    assert main(["--out-dir", str(src.out_dir), "--paper-dir", str(paper)]) == 1
    assert "tab_rotation_dev" in capsys.readouterr().err


# ---------------------------------------------------------------------------------------------- every builder on the full fixture
def _balanced(tex: str) -> bool:
    import re
    opened = re.findall(r"\\begin\{([a-z]+\*?)\}", tex)
    closed = re.findall(r"\\end\{([a-z]+\*?)\}", tex)
    return sorted(opened) == sorted(closed)


def test_every_development_builder_runs_on_the_full_fixture(full):
    import re
    from es.paper_tables import BUILDERS
    src, _ = full
    names = [n for n in BUILDERS if not n.startswith("tab_sealed")]
    assert len(names) == 36 + 1                                                            # the 36 names of the brief and esm_provenance
    for name in names:
        fn, needs = BUILDERS[name]
        assert all(src.exists(n) for n in needs), (name, needs)
        tex = fn(src)
        assert isinstance(tex, str) and tex.strip(), name
        assert _balanced(tex), name
        assert "\\PENDING" not in tex, name                                                 # the full fixture has every source
        assert not re.search(r"\bnan\b|\bNone\b|\binf\b", tex), name


# ---------------------------------------------------------------------------------------------- more behaviours
BRIEF_NAMES = (
    "tab_panel", "tab_chronology", "tab_volatility", "tab_implementation", "tab_blocks_features", "tab_coverage", "tab_block_audit",
    "tab_estimators", "tab_holdout", "tab_alpha_band", "tab_seeds", "tab_pr", "tab_label_control", "tab_label_control_pr", "tab_monitors",
    "tab_monitors_full", "tab_policies", "tab_analog_blocks", "tab_registration", "tab_bh_families", "tab_econ_primary", "tab_econ_ci",
    "tab_decomposition", "tab_comparators", "tab_fee_position_signs", "tab_alpha_curve", "tab_volmanagement", "tab_rotation_dev", "tab_looo",
    "tab_mixed", "esm_ledger_legacy", "esm_ledger_analogs", "esm_ledger_economics", "esm_ledger_pr", "esm_ledger_monitors", "esm_ledger_bh")


def test_the_registered_builders_are_the_names_of_the_brief_and_the_provenance_part():
    from es.paper_tables import BUILDERS
    assert {n for n in BUILDERS if not n.startswith("tab_sealed")} == set(BRIEF_NAMES) | {"esm_provenance"}
    assert all(isinstance(needs, tuple) for _, needs in BUILDERS.values())


def test_implementation_table_renders_nested_values_without_ambiguity(tmp_path):
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    _write_impl(src, rules=dict(IMPL_RULES_SMALL, fees_calibrated={"NYISO": {"INC": 0.2, "DEC": 0.15}, "MISO": {"INC": 0.25, "DEC": 0.15}}))
    tex = BUILDERS["tab_implementation"][0](src)
    assert "NYISO: (INC: 0.2, DEC: 0.15); MISO: (INC: 0.25, DEC: 0.15)" in tex


def test_label_control_cells_show_the_node_count_only_when_it_differs(tmp_path):
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    (src.out_dir / "labels_pr").mkdir(exist_ok=True)
    rows = [{"analysis": a, "block": b, "failures": 7, "n_nodes": 33, "median_R": 1.03}
            for a in ("LC-full-q2023", "LC-full-q2324", "LC-h1-q2023", "LC-h1-q2324") for b in ("LEV6", "DART18", "G1_24", "DART3")]
    pd.DataFrame(rows).to_csv(src.out_dir / "labels_pr" / "label_control_2x2.csv", index=False)
    tex = BUILDERS["tab_label_control"][0](src)
    assert "7 (1.03)" in tex and "/33" not in tex
    rows[3]["n_nodes"] = 32                                                                  # a cell with a node that was not scored
    pd.DataFrame(rows).to_csv(src.out_dir / "labels_pr" / "label_control_2x2.csv", index=False)
    tex = BUILDERS["tab_label_control"][0](_fresh(src))
    assert "7/32 (1.03)" in tex and "7/33 (1.03)" in tex


def test_monitors_ledger_carries_the_one_sided_p_and_the_node_count(full):
    from es.paper_tables import BUILDERS
    src, _ = full
    tex = BUILDERS["esm_ledger_monitors"][0](src)
    head = tex.split("\\midrule")[0]
    assert "one-sided" in head and "nodes" in head and "Spearman" in head and "AUROC" in head
    first_row = tex.split("\\endfoot")[1].strip().splitlines()[0]
    assert first_row.count("&") == 8 and first_row.rstrip().endswith("\\\\")                 # analysis, block, monitor, nodes, rho, p, AUROC, reference, nodes behind it
    assert BUILDERS["tab_monitors_full"][0](src).count("tab:monitors-full") == 1 and "tab:esm-monitors" in tex


def test_ledgers_print_a_dash_for_an_undefined_cell(tmp_path):
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    s = pd.read_csv(src.out_dir / "legacy" / "summary.csv")
    s.loc[0, "median_auprc"] = float("nan")                                                    # a cell without positives has no AUPRC
    s.to_csv(src.out_dir / "legacy" / "summary.csv", index=False)
    tex = BUILDERS["esm_ledger_legacy"][0](_fresh(src))
    assert " & -- & " in tex


def test_mixed_model_text_is_set_by_its_longest_line(tmp_path):
    import pytest as _pytest
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    (src.out_dir / "inference").mkdir(exist_ok=True)
    (src.out_dir / "inference" / "mixed_model.txt").write_text("x" * 78 + "\nshort\n")
    assert BUILDERS["tab_mixed"][0](src).startswith("\\begin{small}\\begin{verbatim}\n" + "x" * 78)
    (src.out_dir / "inference" / "mixed_model.txt").write_text("x" * 90 + "\n")
    assert BUILDERS["tab_mixed"][0](_fresh(src)).startswith("\\begin{footnotesize}")
    (src.out_dir / "inference" / "mixed_model.txt").write_text("x" * 120 + "\n")
    with _pytest.raises(ValueError, match="120 characters"):
        BUILDERS["tab_mixed"][0](_fresh(src))


def test_economic_tables_keep_each_ex_caiso_row_directly_under_its_analysis(full):
    from es.paper_tables import BUILDERS
    src, _ = full
    ci = BUILDERS["tab_econ_ci"][0](src)
    body = ci.split("\\endhead")[1].split("\\end{longtable}")[0]
    labels = [c for c in (line.split(" & ")[0].removeprefix("\\midrule").replace("\\raggedright\\hyphenpenalty=10000 ", "")
                          for line in body.splitlines() if line.endswith("\\\\")) if c]
    assert len(labels) == 17                                                                    # 8 analyses, their 8 ex-CAISO twins, the pooled rehearsal
    for i, label in enumerate(labels):
        if label.endswith(", without CAISO"):
            assert labels[i - 1] == label[:-len(", without CAISO")]
    assert labels[-1].startswith("AQ3+AQ4") and labels[-3] == "AQ3 frozen-origin"              # the frozen-origin analogue after the seven, the rehearsal last
    assert body.count("\\midrule") == 8                                                         # one before each of 8 further full groups, never before a twin
    dec = BUILDERS["tab_decomposition"][0](src)
    for panel in dec.split("\\begin{longtable}")[1:]:
        assert panel.split("\\endhead")[1].count("\\midrule") == 8                           # before each of the 8 further full groups, in both panels


# ---------------------------------------------------------------------------------------------- widths in the real class
PDFLATEX = Path("/Library/TeX/texbin/pdflatex")
PDFTOTEXT = Path("/opt/homebrew/bin/pdftotext")
#: The environment each table is placed in, as the INDEX notes ask the sections to place it. A table not listed is in a `table`
#: float. Longtables of the main text sit in \footnotesize groups, those of the Online Resource set their own group, and the
#: longtables that carry their own caption and label (tab_econ_ci, tab_decomposition, tab_implementation) are included bare
#: (Rulings 86 and 95). The chronology is the one sideways table, in a sidewaystable with a long caption.
SIDEWAYS_TABLES = ("tab_chronology",)
SECTION_FOOTNOTESIZE = ("tab_blocks_features", "tab_monitors_full", "tab_looo")
BARE_TABLES = ("tab_econ_ci", "tab_decomposition", "tab_implementation", "tab_mixed", "esm_provenance", "esm_ledger_legacy", "esm_ledger_analogs",
               "esm_ledger_economics", "esm_ledger_pr", "esm_ledger_monitors", "esm_ledger_bh")
SIDEWAYS_CAPTION = ("A sideways caption of four lines, as long as the sections write them: it names the quantity, the analyses, the "
                    "interval level, the meaning of the mark and the comparison that the table makes, so that the table can be read "
                    "without the text and the caption width follows the width of the table that it belongs to. " * 2)
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
\input{crossref}%% the section pointers of tab_implementation and tab_coverage are \mainref{key}, a macro of paper/crossref.tex (plan 2C task T2b)
%%PENDING%%
\newcommand{\gentable}[1]{\IfFileExists{generated/tables/#1.tex}{\input{generated/tables/#1.tex}}{\begin{tabular}{l}\textbf{#1 missing}\end{tabular}}}
\newsavebox{\probebox}
\newcommand{\probegeometry}[1]{\sbox{\probebox}{\tablebodyfont\input{generated/tables/#1.tex}}\typeout{GEOM #1 \the\wd\probebox\space\the\ht\probebox\space\the\dp\probebox}}
\newcommand{\probedim}[2]{\typeout{DIM #1 \the\dimexpr #2\relax}}
\title[probe]{probe}
\author*[1]{\fnm{A} \sur{B}}\email{a@b.c}
\affil*[1]{\orgdiv{D}, \orgname{U}, \orgaddress{\city{C}, \state{S}, \country{USA}}}
\begin{document}
\maketitle
\typeout{TEXTWIDTH \the\textwidth}
\typeout{TEXTHEIGHT \the\textheight}
"""
#: The pending sentinels with the text that es.paper_macros gives them in generated/macros.tex, not a shorter stand-in: a sentinel is wider
#: than a number, and a table that holds one has to hold it.
LATEX_PREAMBLE = LATEX_PREAMBLE.replace("%%PENDING%%", "\n".join(
    f"\\newcommand{{{command}}}{{\\textbf{{{PENDING_TEXT[stage]}}}}}" for stage, command in PENDING.items()))


def _width_problems(log: str) -> dict[str, list[str]]:
    """Overfull boxes over 1 pt and floats too large for a page, by the table that was being typeset (the \\typeout markers)."""
    import re
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


def _float_ends_on_its_page(text: str, name: str) -> bool:
    """The caption's start marker and the end marker that follows the table are both in the extracted text of the PDF."""
    return f"STARTMARK{name.replace('_', '')}" in text and f"ENDMARK{name.replace('_', '')}" in text


@pytest.mark.slow
@pytest.mark.skipif(not (PDFLATEX.exists() and PDFTOTEXT.exists()), reason="TeX Live or poppler not installed")
def test_every_table_of_the_full_fixture_fits_the_page_in_the_real_class(full, tmp_path):
    """Each development table, built from realistic data and set in the environment of its section, compiles with sn-jnl
    without an overfull box over 1 pt, and every float ends on its page. LaTeX does not check the width of a longtable, so the
    guard of tab_decomposition and tab_econ_ci is checked against the class: the text width of the guard is the class's
    \\textwidth, each panel of the decomposition (typeset as a tabular with the same columns and cells, because LaTeX cannot box a
    longtable) is never wider than the guard's estimate, not far narrower, and not wider than \\textwidth, and the width of
    tab_econ_ci is the TeX sum of its columns. The sideways chronology stays inside the text block."""
    import os
    import re
    import subprocess
    from es.paper_paths import PAPER_DIR
    from es.paper_tex import tabular
    import es.paper_tables as pt
    if not (PAPER_DIR / "sn-jnl.cls").exists():
        pytest.skip("sn-jnl.cls is not in the paper directory")
    src, _ = full
    doc = tmp_path / "doc"
    status = pt.build_all(src, doc / "generated" / "tables")
    names = [n for n, s in status.items() if s == "built" and not n.startswith("tab_sealed")]
    assert len(names) == len(BRIEF_NAMES) + 1
    # a tabular with the cells of each panel of the decomposition and columns of their natural width, to measure what the cells need
    twins = []
    for i, (_, widths, header, rows) in enumerate(pt.decomposition_panels(src)):
        twins.append(f"probe_decomposition_{i}")
        natural = f"p{{{widths[0]:.1f}pt}}l" + "c" * (len(header) - 2)
        (doc / "generated" / "tables" / f"{twins[-1]}.tex").write_text(      # no stray space around the tabular: it is measured in a box, not set in vertical mode
            f"\\begingroup\\footnotesize\\setlength{{\\tabcolsep}}{{{pt.DECOMPOSITION_COLSEP}pt}}%\n{tabular(natural, header, rows).rstrip()}%\n\\endgroup%\n")
    # the p-columns of each longtable as written: their sum plus two separations for each column, as TeX adds them
    sums = {}
    for name, colsep in (("tab_decomposition", pt.DECOMPOSITION_COLSEP), ("tab_econ_ci", pt.ECON_CI_COLSEP)):
        specs = re.findall(r"\\begin\{longtable\}\{((?:p\{[\d.]+pt\})+)\}", (doc / "generated" / "tables" / f"{name}.tex").read_text())
        sums[name] = ["+".join(re.findall(r"p\{([\d.]+pt)\}", spec)) + f"+{2 * colsep * spec.count('p{'):g}pt" for spec in specs]
    # the BH table of a run that did not compute one family: its row is one \PENDINGDEV, whose text is wider than a number
    bh = pd.read_csv(src.out_dir / "inference" / "bh.csv")
    bh[bh["family"] != "economic robustness"].to_csv(src.out_dir / "inference" / "bh.csv", index=False)
    (doc / "generated" / "tables" / "tab_bh_absent.tex").write_text(pt.GENERATED_HEADER + pt.BUILDERS["tab_bh_families"][0](_fresh(src)))
    floats = ["tab_bh_absent"]
    body = [f"\\probegeometry{{{t}}}" for t in twins]
    body += [f"\\probedim{{{name}{i}}}{{{expression}}}" for name, expressions in sums.items() for i, expression in enumerate(expressions)]
    for n in names:
        mark = n.replace("_", "")
        body.append(f"\\typeout{{=== {n} ===}}")
        if n in SIDEWAYS_TABLES:
            floats.append(n)
            body.append(f"\\begin{{sidewaystable}}[p]\\caption{{STARTMARK{mark} {SIDEWAYS_CAPTION}}}\\gentable{{{n}}}\\par ENDMARK{mark}\\end{{sidewaystable}}\\clearpage")
        elif n == "tab_implementation":                                                               # its pages are counted from where two labels land
            body.append("\\label{probe:impl-start}\\gentable{tab_implementation}\\label{probe:impl-end}\\clearpage")
        elif n in BARE_TABLES:
            body.append(f"\\gentable{{{n}}}\\clearpage")
        elif n in SECTION_FOOTNOTESIZE:
            body.append(f"\\begingroup\\footnotesize\\gentable{{{n}}}\\endgroup\\clearpage")
        else:
            floats.append(n)
            body.append(f"\\begin{{table}}[p]\\caption{{STARTMARK{mark} A caption of realistic length that wraps over two lines.}}\\gentable{{{n}}}\\par ENDMARK{mark}\\end{{table}}\\clearpage")
        if n == "tab_bh_families":                                                                     # the variant with the pending row, in the same kind of float
            body.append("\\typeout{=== tab_bh_absent ===}\\begin{table}[p]\\caption{STARTMARKtabbhabsent Benjamini--Hochberg families, one of them not computed.}"
                        "\\gentable{tab_bh_absent}\\par ENDMARKtabbhabsent\\end{table}\\clearpage")
    (doc / "probe.tex").write_text(LATEX_PREAMBLE + "\n".join(body) + "\n\\end{document}\n")
    env = dict(os.environ, TEXINPUTS=f".:{PAPER_DIR}:", BSTINPUTS=f".:{PAPER_DIR}:")             # the working directory first: the paper's own generated/tables must not shadow the fixture's
    run = subprocess.run([str(PDFLATEX), "-interaction=nonstopmode", "-halt-on-error", "probe.tex"], cwd=doc, env=env, capture_output=True, timeout=900)
    log = (doc / "probe.log").read_bytes().decode("utf-8", errors="replace")
    assert run.returncode == 0, "\n".join(l for l in log.splitlines() if l.startswith("!") or l.startswith("l.")) + "\n" + log[-1500:]
    assert str(PAPER_DIR / "generated") not in log.replace("\n", ""), "a table of the paper's own generated/ was typeset instead of the fixture's"
    problems = _width_problems(log)
    assert not problems, problems
    text = subprocess.run([str(PDFTOTEXT), "probe.pdf", "-"], cwd=doc, capture_output=True, timeout=300).stdout.decode("utf-8", errors="replace")
    clipped = [n for n in floats if not _float_ends_on_its_page(text, n)]
    assert not clipped, f"floats cut off by the page: {clipped}"

    # the estimated length of the implementation table against its pages in the PDF, and the text height that the estimate uses
    assert abs(float(re.search(r"TEXTHEIGHT ([\d.]+)pt", log).group(1)) - pt.TEXT_HEIGHT_PT) < 0.01
    aux = (doc / "probe.aux").read_text()
    first, last = (int(re.search(r"\\newlabel\{probe:impl-" + w + r"\}\{\{[^}]*\}\{(\d+)\}", aux).group(1)) for w in ("start", "end"))
    estimated = int(re.search(r"about (\d+) pages", pt.RUNTIME_NOTES["tab_implementation"]).group(1))
    assert abs(estimated - (last - first + 1)) <= 1, (estimated, first, last)                           # within a page of the table as the class sets it
    assert last - first + 1 >= 2                                                                         # (the fixture's rule texts are as long as the real ones)

    # the width guard against the class
    text_width = float(re.search(r"TEXTWIDTH ([\d.]+)pt", log).group(1))
    assert text_width == pt.TEXT_WIDTH_PT                                                      # the constant of the guard is the \textwidth of the class
    geom = {m.group(1): float(m.group(2)) for m in re.finditer(r"GEOM (\S+) ([\d.]+)pt", log)}
    dims = {m.group(1): float(m.group(2)) for m in re.finditer(r"DIM (\S+) ([\d.]+)pt", log)}
    for name in ("tab_decomposition", "tab_econ_ci"):
        for i, estimate in enumerate(pt.GEOMETRY[name]):
            assert abs(estimate - dims[f"{name}{i}"]) < 0.01, (name, i, estimate, dims[f"{name}{i}"])       # the guard's width is the TeX sum of the columns written
            assert dims[f"{name}{i}"] <= text_width, f"panel {i + 1} of {name} is {dims[f'{name}{i}']:.1f} pt wide, more than \\textwidth ({text_width:.1f} pt)"
    for i, estimate in enumerate(pt.GEOMETRY["tab_decomposition"]):
        natural = geom[f"probe_decomposition_{i}"]                                            # what the cells need, in columns of their natural width
        assert natural <= estimate <= 1.05 * natural, (i, natural, estimate)                  # no cell wraps, and no column is far too wide

    # the sideways chronology: LaTeX checks the width of a sideways float only; its height is its extent across the page
    try:
        import pdfplumber
    except ImportError:
        pdfplumber = None
    if pdfplumber is not None:
        pages = text.split("\f")
        with pdfplumber.open(doc / "probe.pdf") as pdf:
            for n in SIDEWAYS_TABLES:
                index = next(i for i, t in enumerate(pages) if f"STARTMARK{n.replace('_', '')}" in t)
                page = pdf.pages[index]
                rotated = [c for c in page.chars if not c["upright"]]
                assert rotated, f"{n}: no rotated text on page {index + 1}"
                left, right = min(c["x0"] for c in rotated), max(c["x1"] for c in rotated)
                assert 0 <= left and right <= page.width, f"{n} runs off the paper: x {left:.0f}..{right:.0f} of {page.width:.0f}"
                assert right - left <= text_width + 1, f"{n} is {right - left:.0f} pt across its page, more than the text block"


# ---------------------------------------------------------------------------------------------- a missing value never prints as a sign or a flag
def test_a_missing_sign_or_flag_is_an_error_not_a_zero_or_a_mark():
    import pytest as _pytest
    from es.paper_tables import _approx, _sign
    assert _sign(2.0) == "+" and _sign(-2.0) == "\\textminus{}" and _sign(0.0) == "0"
    with _pytest.raises(ValueError, match="missing"):
        _sign(float("nan"))
    assert _approx("x", True) == "x\\textsuperscript{a}" and _approx("x", False) == "x" and _approx("x", 0) == "x"
    with _pytest.raises(ValueError, match="approximate"):
        _approx("x", float("nan"))


def test_fee_position_signs_stop_on_a_missing_net(tmp_path):
    import pytest as _pytest
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    e = pd.read_csv(src.out_dir / "economics" / "summary.csv")
    e.loc[0, "NET"] = float("nan")
    e.to_csv(src.out_dir / "economics" / "summary.csv", index=False)
    with _pytest.raises(ValueError, match="missing"):
        BUILDERS["tab_fee_position_signs"][0](_fresh(src))


# ---------------------------------------------------------------------------------------------- python -m es.paper_tables
@pytest.mark.filterwarnings("ignore:'es.paper_tables' found in sys.modules:RuntimeWarning")
def test_running_the_module_builds_the_builders_that_paper_tables_sealed_registers(tmp_path, monkeypatch):
    """`python -m es.paper_tables` runs this file as __main__, while paper_tables_sealed imports es.paper_tables as a second
    module: the sealed builders must still be built (the guard hands over to the canonical module)."""
    import runpy
    import sys
    import es.paper_tables as canonical
    src, paper = _sources(tmp_path)
    monkeypatch.setitem(canonical.BUILDERS, "tab_sealed_probe", (lambda s: "\\begin{tabular}{l}\nprobe \\\\\n\\end{tabular}\n", ()))
    monkeypatch.setattr(sys, "argv", ["es.paper_tables", "--out-dir", str(src.out_dir), "--paper-dir", str(paper)])
    with pytest.raises(SystemExit) as stop:
        runpy.run_module("es.paper_tables", run_name="__main__")
    assert stop.value.code == 0
    assert (paper / "generated" / "tables" / "tab_sealed_probe.tex").exists()
    assert (paper / "generated" / "tables" / "tab_panel.tex").exists()


def test_alpha_curve_rows_follow_the_registered_alpha_grid(full):
    from es.paper_pins import ALPHAS
    from es.paper_tables import BUILDERS
    src, _ = full
    tex = BUILDERS["tab_alpha_curve"][0](src)
    body = tex.split("\\midrule", 1)[1].split("\\bottomrule")[0]
    assert [f"{100 * a:g}\\%" for a in ALPHAS] == ["1\\%", "2.5\\%", "5\\%", "10\\%"]
    rows = [line for line in body.splitlines() if line.endswith("\\\\")]
    assert len(rows) == 2 * len(ALPHAS) and all(f" & {100 * a:g}\\% & " in rows[i] for i, a in enumerate(ALPHAS))
    assert body.count("\\midrule") == 1 and "--" not in body                                       # a rule between the two analyses; every curve row found


# ---------------------------------------------------------------------------------------------- the producers' contracts (es.secondary, es.run_analogs)
def _bh_with_included(src):
    rows = [{"family": "audit robustness", "test": "H-B1 logit", "estimate": 0.4, "p": 0.004, "bh_reject": True, "included": True},
            {"family": "audit robustness", "test": "H-B1 seed=0", "estimate": 0.1, "p": 0.3, "bh_reject": False, "included": True},
            {"family": "audit robustness", "test": "H-B2 frozen", "estimate": float("nan"), "p": float("nan"), "bh_reject": False, "included": False},
            {"family": "economic robustness", "test": "P1 premium fees=calibrated", "estimate": 0.12, "p": 0.02, "bh_reject": False, "included": True},
            {"family": "drift monitors", "test": "LEV6 W1_norm", "estimate": 0.2, "p": 0.6, "bh_reject": False, "included": True}]
    pd.DataFrame(rows).to_csv(src.out_dir / "inference" / "bh.csv", index=False)


def test_bh_tables_count_the_tests_that_ran_and_dash_the_ones_that_did_not(tmp_path):
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    _bh_with_included(src)
    bh = BUILDERS["tab_bh_families"][0](src)
    assert "audit robustness & 2 & -- & --" in bh                                              # three rows, two tests: `included` is the m BH used; Ruling 118: no rejections
    assert "drift monitors & 1 & 0 & 0.600" in bh                                              # the other families are counted as before
    ledger = BUILDERS["esm_ledger_bh"][0](src)
    row = next(line for line in ledger.splitlines() if "H-B2 frozen" in line)
    assert row.count("--") == 3 and "not run" not in row                                       # estimate, p and decision are dashes
    assert "H-B1 seed=0" in ledger and "P1 premium fees=calibrated" in ledger                  # '=' and spaces survive tex_escape
    assert next(line for line in ledger.splitlines() if "H-B1 logit" in line).rstrip().endswith("yes\\textsuperscript{b} \\\\")      # a diagnostic flag (Ruling 118)
    assert next(line for line in ledger.splitlines() if "P1 premium fees=calibrated" in line).rstrip().endswith("no \\\\")      # the other families keep theirs


# ---------------------------------------------------------------------------------------------- two guards
def test_mixed_model_text_must_be_ascii_for_verbatim(tmp_path):
    import pytest as _pytest
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    (src.out_dir / "inference").mkdir(exist_ok=True)
    (src.out_dir / "inference" / "mixed_model.txt").write_text("coef ± se\n")
    with _pytest.raises(ValueError, match="non-ASCII"):
        BUILDERS["tab_mixed"][0](src)


def test_a_builder_that_returns_no_text_is_reported_as_failed_not_dropped(tmp_path, monkeypatch):
    import pytest as _pytest
    import es.paper_tables as pt
    src, paper = _sources(tmp_path)
    monkeypatch.setitem(pt.BUILDERS, "tab_probe_none", (lambda s: None, ()))
    tables = paper / "generated" / "tables"
    status = pt.build_all(src, tables, raise_on_failure=False)
    assert status["tab_probe_none"].startswith("FAILED: TypeError") and "| tab_probe_none | FAILED" in (tables / "INDEX.md").read_text()
    assert not (tables / "tab_probe_none.tex").exists()
    with _pytest.raises(LookupError, match="tab_probe_none"):
        pt.build_all(_fresh(src), tables)


# =====================================================================================================================
# Fix round 1 (controller Rulings 86 and 91; review of Task 4: C1, I1, I2, I3 and M1-M10). Each test below builds its own
# small input with distinct numbers, so that a builder that reads the wrong row, column, estimator, policy or analysis fails.
# =====================================================================================================================
import re  # noqa: E402

RR = "\\raggedright\\hyphenpenalty=10000 "
SUMMARY_DEFAULTS = {"failures": 3, "n_nodes": 33, "median_R": 0.9, "median_log_error": 0.1, "median_auprc": 0.2, "auprc_lift": 1.5,
                    "prevalence": 0.1, "n_zero_activity": 0, "failures_band_0.67_1.5": 5, "failures_band_0.8_1.25": 9,
                    "failures_NYISO": 1, "failures_ISONE": 1, "failures_MISO": 1, "failures_CAISO": 0}


def _csv(src, rel, rows):
    path = src.out_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _srow(analysis, block, estimator="xgb", seed=42, policy="P1", alpha=0.05, **kw):
    """One row of legacy/summary.csv, with the numbers of SUMMARY_DEFAULTS unless given."""
    return {"analysis": analysis, "block": block, "estimator": estimator, "seed": seed, "policy": policy, "alpha": alpha,
            **{**SUMMARY_DEFAULTS, **kw}}


def _rows(tex):
    """The rows of a generated table as lists of cells, without a \\midrule in front or the ragged-right prefix of a text cell."""
    return [[c.replace(RR, "") for c in line.removeprefix("\\midrule").removesuffix(" \\\\").split(" & ")]
            for line in tex.splitlines() if line.endswith("\\\\")]


def _row(tex, *lead):
    """The first row whose leading cells are ``lead``."""
    return next(r for r in _rows(tex) if r[:len(lead)] == list(lead))


def _ci(analysis, stat, estimate, lo, hi, approximate=False):
    return {"analysis": analysis, "stat": stat, "estimate": estimate, "lo": lo, "hi": hi, "approximate": approximate, "n_windows": 1}


def _econ_row(aid, policy, comparator="none", position="premium", fees="calibrated", **kw):
    return {"analysis": aid, "policy": policy, "comparator": comparator, "position": position, "fees": fees,
            **{**{"SELC": 0.0412, "SNET": 7.891, "NET": 12345.6, "ESred": 4321.4, "CAP": 0.2278, "n_days": 92}, **kw}}


# ---------------------------------------------------------------------------------------------- I2: values of the builders that only the smoke loop touched
def test_chronology_prints_each_analysis_with_its_status_and_its_three_windows(tmp_path):
    from es.analyses import ANALOGS, LEGACY
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    tex = BUILDERS["tab_chronology"][0](src)
    w = LEGACY["24H2-25H2"]
    assert _row(tex, "24H2$\\to$25H2") == ["24H2$\\to$25H2", w.status, w.maintenance, f"[{w.train.start}, {w.train.end})",
                                          f"[{w.calib.start}, {w.calib.end})", f"[{w.deploy.start}, {w.deploy.end})"]      # UTC, half open
    q = ANALOGS["AQ3"]
    assert _row(tex, "AQ3: 2025-Q3")[1:] == [q.status, q.maintenance] + [f"{x.start} -- {x.end}" for x in (q.train, q.calib, q.deploy)]   # local days, closed
    assert _row(tex, "AQ3 frozen-origin")[2] == ANALOGS["AQ3F"].maintenance == "frozen"
    assert tex.count("\\midrule") == 1 + 2                                                  # under the header, before the analogues, before the sealed tests
    assert tex.index("24H2$\\to$25H2") < tex.index("AQ3: 2025-Q3") < tex.index("Test A: 2026-Q3")


def test_volatility_table_keeps_the_node_median_and_the_pooled_p95_apart(tmp_path):
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    pooled = {"da_std_pooled": 1.0, "dart_std_pooled": 2.0}
    _csv(src, "legacy/fig1_volatility.csv", [
        {"period": "2024Q1", "kind": "quarter", "da_std": 11.11, "dart_std": 22.22, "dart_p95": 33.33, "dart_p95_pooled": 44.44, **pooled},
        {"period": "2024Q2", "kind": "quarter", "da_std": 15.51, "dart_std": 26.66, "dart_p95": 37.77, "dart_p95_pooled": 48.88, **pooled},
        {"period": "2024", "kind": "period", "da_std": 99.9, "dart_std": 99.9, "dart_p95": 99.9, "dart_p95_pooled": 99.9, **pooled}])
    tex = BUILDERS["tab_volatility"][0](src)
    assert _row(tex, "2024Q1") == ["2024Q1", "11.1", "22.2", "33.3", "44.4"]               # DA std, DART std, p95 (node median), p95 (pooled)
    assert _row(tex, "2024Q2") == ["2024Q2", "15.5", "26.7", "37.8", "48.9"]
    assert "99.9" not in tex                                                                 # the period summary row is not a quarter
    header = tex.split("\\midrule")[0]
    assert header.index("(node median)") < header.rindex("(pooled)")                       # the pooled p95 is the last column


def test_estimator_table_puts_each_analysis_and_estimator_in_its_own_column(tmp_path):
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    cells = {("2024-2025", "xgb"): (11, 0.11), ("2024-2025", "logit"): (12, 0.12), ("25H1-26H1", "xgb"): (13, 0.13),
             ("25H1-26H1", "logit"): (14, 0.14), ("frozen-26H1", "xgb"): (15, 0.15)}
    rows = [_srow(a, "G6", e, failures=f, median_log_error=m) for (a, e), (f, m) in cells.items()]
    rows += [_srow(a, "LEV6", e, failures=f + 20, median_log_error=m + 0.2) for (a, e), (f, m) in cells.items()]
    _csv(src, "legacy/summary.csv", rows)
    tex = BUILDERS["tab_estimators"][0](src)
    assert _row(tex, "calendar")[1:] == ["11 / 0.11", "12 / 0.12", "13 / 0.13", "14 / 0.14", "15 / 0.15"]
    assert _row(tex, "absolute level")[1:] == ["31 / 0.31", "32 / 0.32", "33 / 0.33", "34 / 0.34", "35 / 0.35"]
    assert _row(tex, "plain DART lags")[1] == "\\PENDINGDEV / \\PENDINGDEV"                 # no row: pending, not a number
    assert tex.index("2024$\\to$2025") < tex.index("25H1$\\to$26H1") < tex.index("frozen")


def test_holdout_table_reads_the_sealed_and_the_season_matched_comparison(tmp_path):
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    rows = [_srow(a, b, failures=base + k, median_R=1.0 + (base + k) / 100)
            for a, base in (("sealed26H1", 20), ("25H1-26H1", 30)) for k, b in enumerate(("LEV6", "DART18", "G1_24"))]
    _csv(src, "legacy/summary.csv", rows)
    tex = BUILDERS["tab_holdout"][0](src)
    as_opened = next(r for r in _rows(tex) if r[0].startswith("as opened"))
    matched = next(r for r in _rows(tex) if r[0].startswith("season-matched"))
    assert as_opened[1:] == ["20 (1.20)", "21 (1.21)", "22 (1.22)"]                         # level, DART-relative, combined
    assert matched[1:] == ["30 (1.30)", "31 (1.31)", "32 (1.32)"]
    assert tex.index("level") < tex.index("DART-relative") < tex.index("combined")
    assert "sealed" not in tex                                                              # G3 item 6: the holdout of the earlier version is as-opened, not sealed


def test_the_ledgers_print_the_as_opened_holdout_under_an_id_that_does_not_say_sealed(tmp_path):
    """G3 item 6: the holdout of the earlier version is stored under the id sealed26H1 (es.analyses, which the sealed code binds). The ledger
    prints the id of the analysis in its first column, and a reader must not take it for one of the sealed tests A and B."""
    import es.paper_tables as pt
    src, _ = _sources(tmp_path)
    _csv(src, "legacy/summary.csv", [_srow(a, "G6") for a in ("24H1-25H1", "sealed26H1", "frozen-26H1", "LC-full-q2023")])
    tex = pt.BUILDERS["esm_ledger_legacy"][0](_fresh(src))
    assert {"24H1-25H1, G6", "as-opened-26H1, G6", "frozen-26H1, G6", "LC-full-q2023, G6"} <= {r[0] for r in _rows(tex.split("\\clearpage")[0])}
    assert "sealed" not in tex                                                                    # parallel to frozen-26H1; the other ids are as they were
    assert pt._aid("sealed26H1") == "as-opened-26H1" and pt._aid("24H1-25H1") == "24H1-25H1" and pt._aid("LC-h1-q2324") == "LC-h1-q2324"
    assert pt.LEGACY["sealed26H1"].id == "sealed26H1"                                           # the registered id itself is not renamed


def test_alpha_band_table_reads_the_sweep_and_the_two_bands_in_their_own_columns(tmp_path):
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    rows = []
    for a, base in (("2024-2025", 0), ("25H1-26H1", 100)):
        for k, b in enumerate(("LEV6", "DART18", "G1_24", "DART3")):
            n = base + 10 * k
            for alpha, add in ((0.01, 1), (0.025, 2), (0.05, 3), (0.1, 4)):
                at_base = alpha == 0.05                                                      # the band counts are those of the base cutoff target
                rows.append(_srow(a, b, alpha=alpha, failures=n + add, **{"failures_band_0.67_1.5": n + 5 if at_base else 999,
                                                                          "failures_band_0.8_1.25": n + 6 if at_base else 998}))
    _csv(src, "legacy/summary.csv", rows)
    tex = BUILDERS["tab_alpha_band"][0](src)
    assert _row(tex, "2024$\\to$25", "absolute level") == ["2024$\\to$25", "absolute level", "1", "2", "3", "4", "5", "6"]
    assert _row(tex, "25H1$\\to$26H1", "combined")[2:] == ["121", "122", "123", "124", "125", "126"]
    assert "999" not in tex and "998" not in tex
    heading = next(r for r in _rows(tex) if r[:2] == ["analysis", "block"])
    assert heading[2:6] == ["1\\%", "2.5\\%", "5\\%", "10\\%"]                               # the sweep, in the order of the registered grid
    assert "[0.67" in heading[6] and "[0.8" in heading[7]                                    # the wider band's column comes first


def test_seed_table_prints_the_range_over_seeds_and_one_value_when_it_does_not_vary(tmp_path):
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    rows = []
    for seed, f, m in ((0, 3, 0.10), (1, 5, 0.30), (7, 4, 0.20), (13, 9, 0.25), (42, 6, 0.15)):
        rows.append(_srow("2024-2025", "G6", seed=seed, failures=f, median_log_error=m))
        rows.append(_srow("2024-2025", "LEV6", seed=seed, failures=7, median_log_error=0.2))   # does not vary at all
        if seed != 13:
            rows.append(_srow("2024-2025", "DART3", seed=seed))                              # one seed is missing
    _csv(src, "legacy/summary.csv", rows)
    tex = BUILDERS["tab_seeds"][0](src)
    assert _row(tex, "2024$\\to$25", "calendar") == ["2024$\\to$25", "calendar", "3--9", "0.10--0.30"]
    assert _row(tex, "2024$\\to$25", "absolute level")[2:] == ["7", "0.20"]                  # M10: a degenerate range is one value
    assert "7--7" not in tex and "0.20--0.20" not in tex
    assert _row(tex, "2024$\\to$25", "plain DART lags")[2] == "\\PENDINGDEV"                 # four of the five seeds: pending, not a range


def test_a_stacked_heading_never_starts_a_line_with_a_bracket():
    from es.paper_tables import _stack
    assert _stack("failures,", "[0.67,1.5]") == "\\shortstack[c]{failures,\\\\{}[0.67,1.5]}"              # '\\\\[' would be read as the optional length of the break
    assert _stack("[0.67", "1.5]") == "\\shortstack[c]{{}[0.67\\\\1.5]}" and _stack("a", "b", align="l") == "\\shortstack[l]{a\\\\b}"


def test_the_range_of_equal_values_is_one_value():
    from es.paper_tables import _range
    from es.paper_tex import FORMATS
    assert _range(0, 0, FORMATS["int"]) == "0" and _range(1, 1, FORMATS["int"]) == "1" and _range(3, 9, FORMATS["int"]) == "3--9"
    assert _range(0.201, 0.204, FORMATS["dec2"]) == "0.20"                                  # equal where they print alike


def test_monitor_summary_table_takes_the_median_over_analyses_and_blocks(tmp_path):
    from es.paper_registry import CODE_READERS
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    values = {"W1_norm": ([0.1, 0.2, 0.4, 0.9], [0.5, 0.6, 0.8, 0.9]), "PSI": ([0.0, 0.1, 0.3, 0.4], [0.52, 0.56, 0.64, 0.68]),
              "domain_auc": ([-0.2, 0.0, 0.2, 0.6], [0.6, 0.7, 0.8, 0.9]), "early_rate": ([-0.4, -0.2, 0.2, 0.4], [0.45, 0.47, 0.49, 0.51])}
    rows = []
    for mon, (rho, auroc) in values.items():
        for i, (analysis, block) in enumerate((("AQ3", "LEV6"), ("AQ3", "DART3"), ("AQ4", "LEV6"), ("AQ4", "DART3"))):
            rows.append({"analysis": analysis, "block": block, "monitor": mon, "spearman": rho[i], "spearman_p_one_sided": 0.5, "auroc_fail": auroc[i],
                         "n_used": 33, "calib_days": 92})
    _csv(src, "monitors/monitor_eval.csv", rows)
    tex = BUILDERS["tab_monitors"][0](src)
    assert _row(tex, "1-Wasserstein (normalised)") == ["1-Wasserstein (normalised)", "0.30", "0.70", "2", "2"]
    assert _row(tex, "PSI on inputs") == ["PSI on inputs", "0.20", "0.60", "2", "2"]
    assert _row(tex, "domain-classifier AUC") == ["domain-classifier AUC", "0.10", "0.75", "2", "2"]
    assert _row(tex, f"realised alert rate ({CODE_READERS['earlyDays']()} d)")[1:] == ["0.00", "0.48", "2", "2"]


def _activity(aid, block, est, policy, fails, logs):
    return [{"analysis": aid, "block": block, "estimator": est, "policy": policy, "market": "M", "node": f"n{i}", "R": 1.0,
             "log_error": le, "fail_0.5_2": bool(f)} for i, (f, le) in enumerate(zip(fails, logs))]


def test_policy_table_keeps_the_two_p4_windows_apart(tmp_path):
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    rows = (_activity("AQ3", "G1_24", "xgb", "P1", [1, 0], [0.10, 0.20]) + _activity("AQ3", "G1_24", "xgb", "P4_56", [1, 1], [0.30, 0.50])
            + _activity("AQ3", "G1_24", "xgb", "P4_28", [0, 0], [0.05, 0.07]) + _activity("AQ3", "G1_24", "xgb", "P6", [1, 0], [0.11, 0.13])
            + _activity("AQ3", "G1_24", "logit", "P1", [1, 1], [0.9, 0.9]) + _activity("AQ3", "LEV6", "xgb", "P1", [1, 1], [0.9, 0.9]))   # not the cells shown
    _csv(src, "analogs/activity_all.csv", rows)
    _csv(src, "analogs/eta_selection.csv", [{"analysis": "AQ3", "eta": 0.01, "median_log_error": 0.222}, {"analysis": "AQ3", "eta": 0.05, "median_log_error": 0.9}])
    (src.out_dir / "analogs" / "protocol_inputs.json").write_text(json.dumps({"eta": 0.01}))
    tex = BUILDERS["tab_policies"][0](src)
    assert _row(tex, "AQ3: 2025-Q3")[1:] == ["1 / 0.15", "2 / 0.40", "0 / 0.06", "-- / 0.22", "1 / 0.12"]    # P1, P4 (56 d), P4 (28 d), P5, P6
    assert _row(tex, "AQ4: 2025-Q4")[1] == "\\PENDINGDEV"
    header = tex.split("\\midrule")[0]
    assert header.index("P1") < header.index("P4 (56 d)") < header.index("P4 (28 d)") < header.index("P5") < header.index("P6")


def test_analogue_block_table_counts_the_p1_failures_per_quarter_and_estimator(tmp_path):
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    rows, k = [], 0
    for aid in ("AQ3", "AQ4", "AQ1", "AQ2"):
        for est in ("xgb", "logit"):
            k += 1
            rows += _activity(aid, "LEV6", est, "P1", [i < k for i in range(10)], [0.1] * 10)       # k failures of 10 nodes
            rows += _activity(aid, "LEV6", est, "P4_56", [1] * 10, [0.1] * 10)                       # another policy: not counted
    _csv(src, "analogs/activity_all.csv", rows)
    tex = BUILDERS["tab_analog_blocks"][0](src)
    assert _row(tex, "absolute level")[1:] == ["1", "2", "3", "4", "5", "6", "7", "8"]        # AQ3 xgb, logit, AQ4 xgb, logit, AQ1, AQ2
    assert _row(tex, "calendar")[1] == "\\PENDINGDEV"
    groups = tex.split("\\midrule")[0]
    assert groups.index("AQ3") < groups.index("AQ4") < groups.index("AQ1") < groups.index("AQ2")


def test_primary_economics_table_prints_each_measure_in_its_own_column_and_names_the_policies(tmp_path):
    from es.paper_registry import CODE_CONSTANTS
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    _csv(src, "economics/summary.csv", [
        _econ_row("AQ3", "P1"),
        _econ_row("AQ3", "P4", NET=-500.0, SELC=0.1111, SNET=-1.5, ESred=100.0, CAP=0.2222, n_days=91),
        _econ_row("AQ4", "P1", NET=22222.0, SELC=0.0777, SNET=9.0, ESred=800.0, CAP=0.3333),
        _econ_row("AQ3", "P1", position="inc", NET=1.0, SELC=0.9, CAP=0.9)])                   # not the premium position: not shown
    tex = BUILDERS["tab_econ_primary"][0](src)
    assert _row(tex, "AQ3: 2025-Q3", "P1") == ["AQ3: 2025-Q3", "P1", "\\$12{,}346", "4.1\\%", "\\$7.89", "\\$4{,}321", "22.8\\%", "92"]   # NET, SELC, SNET, ESred, CAP, days
    assert _row(tex, "AQ3: 2025-Q3", "P4")[2:] == ["\\textminus{}\\$500", "11.1\\%", "\\textminus{}\\$1.50", "\\$100", "22.2\\%", "91"]
    assert _row(tex, "AQ4: 2025-Q4", "P1")[2:] == ["\\$22{,}222", "7.8\\%", "\\$9.00", "\\$800", "33.3\\%", "92"]
    header = tex.split("\\midrule")[0]
    assert header.index("NET") < header.index("SELC") < header.index("SNET") < header.index("ESred") < header.index("CAP")
    legend = tex[tex.index("\\footnotetext"):]                                              # the codes of the column are named under the table
    for text in ("P1: fixed, held-out", f"P4: rolling ({CODE_CONSTANTS['pFourWindow']} d)", "P5: self-correcting", "P6: exact-$\\alpha$ (benchmark)", "P4@r*:"):
        assert text in legend, text


def test_comparator_table_reads_each_comparator_from_its_own_rows(tmp_path):
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    rows = [_econ_row("AQ3", "P1", SELC=0.1111, NET=1000.0, ESred=111.0, CAP=0.5)]          # Ruling 118: ESred sits beside NET
    for i, comparator in enumerate(("calendar@count", "VOL@count", "VOL_P1", "VOL_P4", "VOL_P6"), 2):
        rows.append(_econ_row("AQ3", "comparator", comparator, SELC=0.1111 * i, NET=1000.0 * i, ESred=111.0 * i * i, CAP=0.1 * (i + 4)))
    rows.append(_econ_row("AQ3", "comparator", "VOL@count", fees="0.5", SELC=0.9, NET=9.0, ESred=9.0, CAP=0.9))        # another fee scenario: not shown
    _csv(src, "economics/summary.csv", rows)
    tex = BUILDERS["tab_comparators"][0](src)
    assert [c.strip() for c in tex.split("\\midrule")[0].split("\\\\")[0].split("&")][-4:] == ["SELC", "NET", "ESred", "CAP"]
    shown = {r[1]: r[2:] for r in _rows(tex) if r[0] == "AQ3: 2025-Q3"}
    assert shown["alerts (P1)"] == ["11.1\\%", "\\$1{,}000", "\\$111", "50.0\\%"]
    assert shown["calendar-only model at matched count"] == ["22.2\\%", "\\$2{,}000", "\\$444", "60.0\\%"]
    assert shown["VOL at matched count"] == ["33.3\\%", "\\$3{,}000", "\\$999", "70.0\\%"]          # not the calendar-only row's numbers
    assert shown["VOL through P1"] == ["44.4\\%", "\\$4{,}000", "\\$1{,}776", "80.0\\%"] and shown["VOL through P4"][1] == "\\$5{,}000"
    assert shown["VOL through P6"][1] == "\\$6{,}000" and shown["VOL through P6"][2] == "\\$3{,}996"


def test_volatility_management_table_prints_the_two_sharpe_ratios_in_order(tmp_path):
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    _csv(src, "economics/volmanagement_check.csv", [{"analysis": "AQ3", "sharpe_alerted": 0.123, "sharpe_not_alerted": -0.045},
                                                    {"analysis": "AQ4", "sharpe_alerted": 0.456, "sharpe_not_alerted": 0.078}])
    tex = BUILDERS["tab_volmanagement"][0](src)
    assert _row(tex, "AQ3: 2025-Q3") == ["AQ3: 2025-Q3", "0.123", "\\textminus{}0.045"]
    assert _row(tex, "AQ4: 2025-Q4") == ["AQ4: 2025-Q4", "0.456", "0.078"]
    assert tex.index("alerted hours") < tex.index("other hours")


def test_leave_one_operator_out_table_keeps_the_four_blocks_and_orders_the_rows(tmp_path):
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    _csv(src, "inference/looo.csv", [
        {"dropped": "CAISO", "analysis": "AQ3", "block": "DART3", "failures": 12, "median_log_error": 0.34},
        {"dropped": "NYISO", "analysis": "AQ3", "block": "LEV6", "failures": 5, "median_log_error": 0.21},
        {"dropped": "ISONE", "analysis": "2024-2025", "block": "LEV6", "failures": 7, "median_log_error": 0.17},
        {"dropped": "NYISO", "analysis": "2024-2025", "block": "G6", "failures": 9, "median_log_error": 0.5}])         # G6 is not one of the four
    data = [r for r in _rows(BUILDERS["tab_looo"][0](src)) if r[0] in ("2024-2025", "AQ3")]
    assert data == [["2024-2025", "absolute level", "ISO-NE", "7", "0.17"], ["AQ3", "absolute level", "NYISO", "5", "0.21"],
                    ["AQ3", "plain DART lags", "CAISO", "12", "0.34"]]


def test_analogue_ledger_sums_the_failures_and_takes_the_median_per_cell_in_policy_order(tmp_path):
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    rows = (_activity("AQ3", "DART3", "xgb", "P6", [0, 0, 0], [0.9, 0.9, 0.9]) + _activity("AQ3", "DART3", "xgb", "P1", [1, 0, 1], [0.1, 0.4, 0.2])
            + _activity("AQ3", "DART3", "xgb", "P4_28", [1, 1, 1], [0.3, 0.3, 0.6]) + _activity("AQ3", "DART3", "logit", "P1", [0, 0, 1], [0.5, 0.6, 0.7]))
    _csv(src, "analogs/activity_all.csv", rows)
    data = [r for r in _rows(BUILDERS["esm_ledger_analogs"][0](src)) if r[0] == "AQ3"]
    assert data == [["AQ3", "DART3", "xgb", "P1", "2", "3", "0.20"], ["AQ3", "DART3", "xgb", "P4\\_28", "3", "3", "0.30"],
                    ["AQ3", "DART3", "xgb", "P6", "0", "3", "0.90"], ["AQ3", "DART3", "logit", "P1", "1", "3", "0.60"]]


def test_economics_ledger_prints_every_row_with_its_own_numbers_in_order(tmp_path):
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    _csv(src, "economics/summary.csv", [
        _econ_row("AQ4", "P1", SELC=0.0777, NET=22222.0, SNET=9.0, ESred=800.0),
        _econ_row("AQ3", "comparator", "VOL@count", SELC=0.3333, NET=3000.0, SNET=3.33, ESred=333.0),
        _econ_row("AQ3", "P6", SELC=0.0666, NET=666.0, SNET=6.66, ESred=66.0),
        _econ_row("AQ3", "P1")])
    data = [r for r in _rows(BUILDERS["esm_ledger_economics"][0](src)) if r[0] in ("AQ3", "AQ4")]
    assert [r[:3] for r in data] == [["AQ3", "P1", "none"], ["AQ3", "P6", "none"], ["AQ3", "comparator", "VOL@\\allowbreak{}count"], ["AQ4", "P1", "none"]]
    assert data[0][3:] == ["premium", "calibrated", "4.1\\%", "\\$12{,}346", "\\$7.89", "\\$4{,}321"]
    assert data[1][5:] == ["6.7\\%", "\\$666", "\\$6.66", "\\$66"]
    assert data[2][5:] == ["33.3\\%", "\\$3{,}000", "\\$3.33", "\\$333"]
    assert data[3][5:] == ["7.8\\%", "\\$22{,}222", "\\$9.00", "\\$800"]


def test_precision_recall_ledger_takes_the_node_median_of_each_column(tmp_path):
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    nodes = [(0.1, 0.01, 0.3, 0.05, 1.0), (0.2, 0.02, 0.4, 0.15, 2.0), (0.6, 0.09, 0.5, 0.25, 9.0)]     # the mean of the first column would be 0.30
    _csv(src, "labels_pr/pr_all.csv", [{"analysis": "2024-2025", "block": "G6", "estimator": "xgb", "market": "M", "node": f"n{i}", "precision_q": a, "recall_q": b,
                                        "precision_eval": c, "recall_eval": d, "precision_lift": e} for i, (a, b, c, d, e) in enumerate(nodes)])
    assert _row(BUILDERS["esm_ledger_pr"][0](src), "2024-2025", "G6", "xgb") == ["2024-2025", "G6", "xgb", "0.20", "0.02", "0.40", "0.15", "2.00"]


# ---------------------------------------------------------------------------------------------- I2: twin cells and the approximate flag follow the data
def _two_analyses(**flags):
    """AQ3 and its ex-CAISO twin with different numbers; ``flags`` maps a stat to its approximate flag (default False)."""
    rows = []
    for aid, k in (("AQ3", 1), ("AQ3|ex-CAISO", 2)):
        rows += [_ci(aid, "NET", 1000.0 * k, 1000.0 * k - 100, 1000.0 * k + 100, flags.get("NET", False)),
                 _ci(aid, "SELC", 0.03 * k, 0.01 * k, 0.05 * k, flags.get("SELC", False)),
                 _ci(aid, "E3:ESred", 300.0 * k, 200.0 * k, 400.0 * k, flags.get("E3:ESred", False))]
        for part, scale in (("timing", 1), ("level", 2), ("total", 3)):
            rows.append(_ci(aid, f"E1:NET:{part}", 10.0 * scale * k, 5.0 * scale * k, 15.0 * scale * k, flags.get(f"E1:NET:{part}", False)))
    return rows


def _econ_ci_cells(tex):
    """{(analysis label, estimand label, comparator): cell} of tab_econ_ci; an empty first cell continues the analysis above."""
    body = tex.split("\\endhead")[1].split("\\end{longtable}")[0]
    out, analysis = {}, None
    for cells in _rows(body):
        analysis = cells[0] or analysis
        out[(analysis, cells[1], cells[3])] = cells[2]
    return out


def _dec_cells(tex):
    """{(panel, analysis label, part): [cell of each measure]} of tab_decomposition; an empty first cell continues the analysis above."""
    out = {}
    for number, panel in enumerate(tex.split("\\begin{longtable}")[1:]):
        analysis = None
        for cells in _rows(panel.split("\\endhead")[1].split("\\end{longtable}")[0]):
            analysis = cells[0] or analysis
            out[(number, analysis, cells[1])] = cells[2:]
    return out


def test_ex_caiso_cells_come_from_the_rows_of_the_twin(tmp_path):
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    _csv(src, "economics/bootstrap_ci.csv", _two_analyses())
    cells = _econ_ci_cells(BUILDERS["tab_econ_ci"][0](src))
    assert cells[("AQ3: 2025-Q3", "NET", "")] == "\\$1{,}000 [\\$900, \\$1{,}100]"
    assert cells[("AQ3: 2025-Q3, without CAISO", "NET", "")] == "\\$2{,}000 [\\$1{,}900, \\$2{,}100]"
    assert cells[("AQ3: 2025-Q3", "SELC", "")] == "3.0\\% [1.0\\%, 5.0\\%]"
    assert cells[("AQ3: 2025-Q3, without CAISO", "SELC", "")] == "6.0\\% [2.0\\%, 10.0\\%]"
    dec = _dec_cells(BUILDERS["tab_decomposition"][0](_fresh(src)))
    parts = ("timing", "level", "total")
    assert [dec[(0, "AQ3: 2025-Q3", p)][0] for p in parts] == ["\\$10 [\\$5, \\$15]", "\\$20 [\\$10, \\$30]", "\\$30 [\\$15, \\$45]"]      # NET, by part
    assert [dec[(0, "AQ3: 2025-Q3, without CAISO", p)][0] for p in parts] == ["\\$20 [\\$10, \\$30]", "\\$40 [\\$20, \\$60]", "\\$60 [\\$30, \\$90]"]


def test_the_approximate_mark_follows_the_flag_of_the_data(tmp_path):
    from es.paper_tables import BUILDERS
    mark = "\\textsuperscript{a}"
    src, _ = _sources(tmp_path)
    _csv(src, "economics/bootstrap_ci.csv", _two_analyses(SELC=True, **{"E1:NET:level": True}))             # the flags of the brief's fixture, moved
    ci = _econ_ci_cells(BUILDERS["tab_econ_ci"][0](src))
    assert ci[("AQ3: 2025-Q3", "SELC", "")].endswith(mark) and ci[("AQ3: 2025-Q3, without CAISO", "SELC", "")].endswith(mark)
    assert not ci[("AQ3: 2025-Q3", "NET", "")].endswith(mark) and mark not in ci[("AQ3: 2025-Q3", "ESred", "")]      # E3:ESred is not flagged now
    dec_tex = BUILDERS["tab_decomposition"][0](_fresh(src))
    dec = _dec_cells(dec_tex)
    assert [dec[(0, "AQ3: 2025-Q3", p)][0].endswith(mark) for p in ("timing", "level", "total")] == [False, True, False]
    ci_tex = BUILDERS["tab_econ_ci"][0](_fresh(src))
    assert "labelled approximate" in ci_tex and "labelled approximate" in dec_tex


def test_a_table_without_an_approximate_interval_has_no_footnote_for_the_mark(tmp_path):
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    _csv(src, "economics/bootstrap_ci.csv", _two_analyses())                                            # every flag False
    for name in ("tab_econ_ci", "tab_decomposition"):
        tex = BUILDERS[name][0](_fresh(src))
        assert "textsuperscript{a}" not in tex and "labelled approximate" not in tex, name


# ---------------------------------------------------------------------------------------------- C1 and Ruling 86: tab_econ_ci, tab_decomposition, tab_implementation
def test_econ_ci_is_a_longtable_with_its_own_caption_and_label(full):
    from es.paper_registry import CODE_READERS
    from es.paper_tables import BUILDERS
    src, _ = full
    tex = BUILDERS["tab_econ_ci"][0](src)
    assert tex.count("\\begin{longtable}") == 1 and "\\begin{tabular}" not in tex
    caption = re.search(r"\\caption\{([^}]*)\}\\label\{tab:econ-ci\}", tex)
    assert caption and f"{CODE_READERS['ciLevel']():g}\\%" in caption.group(1)               # the interval level is read from the code
    assert tex.count("\\toprule") == 2                                                          # first head and continuation head


def test_econ_ci_has_a_row_per_analysis_and_estimand_with_the_comparison_in_its_own_column(full):
    from es.paper_registry import STAT_ORDER
    from es.paper_tables import BUILDERS
    src, _ = full
    cells = _econ_ci_cells(BUILDERS["tab_econ_ci"][0](src))
    assert len(cells) == 17 * len(STAT_ORDER)                                                   # 17 groups x every registry stat, none dropped
    assert {(e, c) for (a, e, c) in cells if a == "AQ4: 2025-Q4"} == {
        ("SELC", ""), ("NET", ""), ("SNET (\\$/MW-day)", ""), ("ESred", ""), ("SELC, decile strata", ""),
        ("SELC at matched count", "calendar-only"), ("SELC at matched count", "VOL"),
        ("paired difference, SELC", "P1 \\textminus{} calendar-only"), ("paired difference, SELC", "P1 \\textminus{} VOL"),
        ("ESred at matched count", "calendar-only"), ("ESred at matched count", "VOL"),
        ("paired difference, ESred", "P1 \\textminus{} calendar-only"), ("paired difference, ESred", "P1 \\textminus{} VOL")}
    assert all("--" not in cell for cell in cells.values())                                     # the full fixture has every stat for every group
    header = BUILDERS["tab_econ_ci"][0](src).split("\\midrule")[0]
    for word in ("analysis", "estimand", "estimate [interval]", "comparison"):
        assert word in header


def test_econ_ci_prints_the_registry_wording_in_full_in_the_notes_and_leaves_the_registry_alone(full):
    from es.paper_registry import STAT_LABEL, STAT_ORDER
    from es.paper_tables import BUILDERS, CI_LABEL
    src, _ = full
    tex = BUILDERS["tab_econ_ci"][0](src)
    body, notes = tex.split("\\end{longtable}")
    for stat in STAT_ORDER:
        assert STAT_LABEL[stat] in notes, stat                                                  # E4: SELC of the calendar-only model, matched count, ...
        if CI_LABEL[stat][0] != STAT_LABEL[stat]:
            assert STAT_LABEL[stat] not in body, stat                                           # the rows carry the table's own short labels
    assert all(len(label) <= 48 for label, _ in CI_LABEL.values())
    assert STAT_LABEL["E4:SELC|calendar@count"] == "E4: SELC of the calendar-only model, matched count"         # the registry is as it was


def test_econ_ci_asks_for_a_short_label_when_the_registry_gains_a_stat(full, monkeypatch):
    import es.paper_tables as pt
    src, _ = full
    monkeypatch.setattr(pt, "STAT_ORDER", pt.STAT_ORDER + ("E5:new",))
    with pytest.raises(LookupError, match="E5:new"):
        pt.BUILDERS["tab_econ_ci"][0](src)


def test_implementation_is_a_longtable_with_its_own_caption_and_label(tmp_path):
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    _write_impl(src, rules=IMPL_RULES_SMALL)
    tex = BUILDERS["tab_implementation"][0](src)
    assert tex.count("\\begin{longtable}") == 1 and "\\begin{tabular}" not in tex
    assert re.search(r"\\caption\{[^}]+\}\\label\{tab:implementation\} \\\\", tex)
    assert "\\midrule\\multicolumn{2}{l}{\\textit{Fixed by the design of the study" in tex             # the second block opens with a rule and a one-cell heading
    assert tex.index("Read from the code") < tex.index("Fixed by the design of the study")
    assert "spec v" not in tex and "specification" not in tex                                       # the reader has no design specification


def test_a_spec_fixed_rule_that_the_rules_state_with_its_section_is_printed_once(tmp_path):
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    _write_impl(src)                                                                              # the real rule texts: coverage_rule is in rules and in spec_fixed
    tex = BUILDERS["tab_implementation"][0](src)
    row = next(r for r in _rows(tex) if r[0] == "coverage rule")
    assert len(row) == 2 and row[1].count("Section~\\ref{sec:sealed-windows}") == 1 and "28 usable nodes" in row[1]    # M4: the pointer once, not "(set ..., 6.1.1); 28 usable nodes ..., set ..., 6.1.1"
    assert sum(1 for r in _rows(tex) if r[0] == "coverage rule") == 1
    other = next(r for r in _rows(tex) if r[0] == "P5 step grid")                                # the rules have no text with the section: value and sentence
    assert "\\mainref{data-econ-policies}" in other[1] and "0.005" in other[1]


# ---------------------------------------------------------------------------------------------- Ruling 95 (M8): the decomposition is long-format longtables that fit the text width
def _decomposition_ci(groups, estimate=123456.0):
    rows = []
    for aid in groups:
        for key in ("NET", "ESred", "J_0.5"):
            for part in ("timing", "level", "total"):
                rows.append(_ci(aid, f"E1:{key}:{part}", -estimate, -2 * estimate, estimate))
    return rows


def test_the_decomposition_is_two_longtables_with_one_caption_and_a_row_per_analysis_and_component(full):
    from es.paper_tables import BUILDERS
    src, _ = full
    tex = BUILDERS["tab_decomposition"][0](src)
    assert tex.count("\\begin{longtable}") == 2 and "\\begin{tabular}" not in tex
    assert tex.count("\\caption{") == 1 and re.search(r"\\caption\{[^}]*\}\\label\{tab:decomposition\} \\\\", tex)      # the first panel carries them
    panel_a, panel_b = tex.split("\\begin{longtable}")[1:]
    assert "Panel A: NET and ESred" in panel_a and "Panel B: $J_{0.5}$" in panel_b
    assert "analysis & component & NET & ESred \\\\" in panel_a and "analysis & component & $J_{0.5}$ \\\\" in panel_b
    cells = _dec_cells(tex)
    assert len(cells) == 2 * 17 * 3                                                             # two panels x 17 groups x timing, level, total
    assert {len(v) for (panel, _, _), v in cells.items() if panel == 0} == {2} and {len(v) for (panel, _, _), v in cells.items() if panel == 1} == {1}
    assert "timing = V(P1) $-$ V(P4@r*), level = V(P4@r*) $-$ V(P4), total = V(P1) $-$ V(P4)" in tex      # the parts are defined under the table


def test_each_decomposition_cell_is_the_measure_and_the_part_of_its_column_and_row(tmp_path):
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    rows = []
    for aid, k in (("AQ3", 1), ("AQ3|ex-CAISO", 2)):
        for m, key in enumerate(("NET", "ESred", "J_0.5"), 1):
            for q, part in enumerate(("timing", "level", "total"), 1):
                estimate = 1000.0 * k * m + 100.0 * q                                           # distinct for every analysis, measure and part
                rows.append(_ci(aid, f"E1:{key}:{part}", estimate, estimate - 10, estimate + 10))
    _csv(src, "economics/bootstrap_ci.csv", rows)
    cells = _dec_cells(BUILDERS["tab_decomposition"][0](src))
    assert cells[(0, "AQ3: 2025-Q3", "timing")] == ["\\$1{,}100 [\\$1{,}090, \\$1{,}110]", "\\$2{,}100 [\\$2{,}090, \\$2{,}110]"]      # NET, ESred
    assert cells[(0, "AQ3: 2025-Q3", "total")] == ["\\$1{,}300 [\\$1{,}290, \\$1{,}310]", "\\$2{,}300 [\\$2{,}290, \\$2{,}310]"]
    assert cells[(1, "AQ3: 2025-Q3", "level")] == ["\\$3{,}200 [\\$3{,}190, \\$3{,}210]"]                                                       # J_0.5
    assert cells[(1, "AQ3: 2025-Q3, without CAISO", "level")] == ["\\$6{,}200 [\\$6{,}190, \\$6{,}210]"]


def test_the_decomposition_width_guard_lets_the_producers_size_pass_and_records_the_panel_widths(full):
    import es.paper_tables as pt
    src, paper = full
    status = pt.build_all(src, paper / "generated" / "tables")
    assert status["tab_decomposition"] == "built"
    widths = pt.GEOMETRY["tab_decomposition"]
    assert len(widths) == 2 and widths[0] > widths[1]                                           # panel A has two measures, panel B one
    assert all(w < pt.TEXT_WIDTH_PT - pt.WIDTH_MARGIN_PT for w in widths)


def test_the_width_of_the_decomposition_does_not_depend_on_the_number_of_analyses(tmp_path):
    import es.paper_tables as pt
    src, _ = _sources(tmp_path)
    _csv(src, "economics/bootstrap_ci.csv", _decomposition_ci(["AQ3", "AQ4"]))
    pt.BUILDERS["tab_decomposition"][0](src)
    few = pt.GEOMETRY["tab_decomposition"]
    _csv(src, "economics/bootstrap_ci.csv", _decomposition_ci([f"X{i:02d}" for i in range(40)]))      # 120 rows: a longtable runs on to another page
    pt.BUILDERS["tab_decomposition"][0](_fresh(src))
    assert pt.GEOMETRY["tab_decomposition"] == few


def test_the_decomposition_width_guard_raises_for_a_panel_wider_than_the_text(tmp_path):
    import es.paper_tables as pt
    src, _ = _sources(tmp_path)
    _csv(src, "economics/bootstrap_ci.csv", _decomposition_ci(["AQ3", "AQ4"], estimate=1234567.0))      # seven-digit values in every cell
    with pytest.raises(ValueError, match=r"panel 1 is \d+ pt wide, more than the 372 pt text width"):
        pt.BUILDERS["tab_decomposition"][0](src)
    _csv(src, "economics/bootstrap_ci.csv", _decomposition_ci(["AQ3", "AQ4"], estimate=123456.0))
    assert pt.BUILDERS["tab_decomposition"][0](_fresh(src))                                          # six digits are fine


def test_the_width_guard_adds_the_columns_and_the_separations_and_checks_the_words_of_each_column():
    from es.paper_tables import TEXT_WIDTH_PT, _check_text_width, _fit_widths, _p_colspec
    (width,) = _check_text_width("probe", [(["a", "b"], [["x", "0123456789"]], [50.0, 45.0])], colsep=2)
    assert width == pytest.approx(50.0 + 45.0 + 2 * 2 * 2)                                       # two columns, two separations each
    with pytest.raises(ValueError, match="more than the 372 pt text width"):
        _check_text_width("probe", [(["a"], [["x"]], [380.0])], colsep=2)
    with pytest.raises(ValueError, match="widen the column"):                                    # an unbreakable word wider than its column overflows it
        _check_text_width("probe", [(["a"], [[RR + "unbreakable" * 6]], [40.0])], colsep=2)
    assert TEXT_WIDTH_PT == 372.0
    widths = _fit_widths(["a", "b"], [["x", "0123456789"]], {0: 50.0})
    assert widths[0] == 50.0 and widths[1] >= 39.9 and _p_colspec(widths) == f"p{{50.0pt}}p{{{widths[1]:.1f}pt}}"     # a column is as wide as its widest cell


def test_the_econ_ci_width_guard_checks_its_columns_against_the_text_width(full, monkeypatch):
    import es.paper_tables as pt
    src, _ = full
    pt.BUILDERS["tab_econ_ci"][0](src)
    (width,) = pt.GEOMETRY["tab_econ_ci"]
    assert width == pytest.approx(sum(pt.ECON_CI_CM) * pt.CM_PT + 2 * pt.ECON_CI_COLSEP * 4, abs=0.3)      # four p-columns of fixed widths (rounded to 0.1 pt)
    monkeypatch.setattr(pt, "ECON_CI_CM", (2.1, 3.4, 5.0, 2.4))
    with pytest.raises(ValueError, match="more than the 372 pt text width"):
        pt.BUILDERS["tab_econ_ci"][0](_fresh(src))
    monkeypatch.setattr(pt, "ECON_CI_CM", (0.5, 3.4, 4.1, 2.4))
    with pytest.raises(ValueError, match="widen the column"):
        pt.BUILDERS["tab_econ_ci"][0](_fresh(src))


def test_the_columns_of_the_decomposition_are_as_wide_as_their_cells_and_never_wrap_a_cell(full):
    import es.paper_tables as pt
    src, _ = full
    for _, widths, header, rows in pt.decomposition_panels(src):
        assert widths[0] == round(pt.DECOMPOSITION_ANALYSIS_CM * pt.CM_PT, 1)                   # the analysis column is the fixed one
        for j in range(1, len(header)):
            needed = max(pt._tex_width_pt(c) for c in [header[j]] + [r[j] for r in rows])
            assert needed <= widths[j] <= needed + 0.1                                           # wide enough for its widest cell, and no wider


# ---------------------------------------------------------------------------------------------- I1: a BH family that bh.csv does not hold
def _bh_rows(families, included=True):
    return [{"family": f, "test": f"{f} test {i}", "estimate": 0.1 * i, "p": p, "bh_reject": p < 0.01, "included": included}
            for f in families for i, p in enumerate((0.001, 0.2, 0.5), 1)]


def test_a_bh_family_that_bh_csv_does_not_hold_is_pending_never_zero(tmp_path):
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    _csv(src, "inference/bh.csv", _bh_rows(["audit robustness", "drift monitors"]))                  # run_inference wrote no economic family
    tex = BUILDERS["tab_bh_families"][0](src)
    assert _row(tex, "economic robustness") == ["economic robustness", "\\multicolumn{3}{c}{\\PENDINGDEV}"]
    assert _row(tex, "audit robustness")[1:] == ["3", "--", "--"] and _row(tex, "drift monitors")[1:] == ["3", "1", "0.001"]      # Ruling 118, M5
    assert "economic robustness: not computed" in tex                                              # and the table says why
    ledger = BUILDERS["esm_ledger_bh"][0](_fresh(src))
    assert "economic robustness: not computed" in ledger and not any(r[0] == "economic robustness" for r in _rows(ledger))


def test_a_bh_family_that_is_there_without_a_test_that_ran_prints_zero(tmp_path):
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    rows = _bh_rows(["audit robustness", "drift monitors"]) + [dict(r, p=float("nan"), estimate=float("nan"), bh_reject=False, included=False)
                                                               for r in _bh_rows(["economic robustness"])]
    _csv(src, "inference/bh.csv", rows)
    tex = BUILDERS["tab_bh_families"][0](src)
    assert _row(tex, "economic robustness") == ["economic robustness", "0", "0", "--"]               # three rows, none ran: 0 tests, no smallest p
    assert "not computed" not in tex and "PENDING" not in tex


# ---------------------------------------------------------------------------------------------- I3: calib_days and the no-drift reference of the monitor tables
def _monitor_rows(calib_days=(181, 92), placebo=True, monitors=("domain_auc", "early_rate"), **extra):
    rows = []
    for analysis, days in zip(("25H1-26H1", "AQ3"), calib_days):
        for monitor in monitors:
            row = {"analysis": analysis, "block": "LEV6", "monitor": monitor, "spearman": 0.2, "spearman_p_one_sided": 0.05, "auroc_fail": 0.6,
                   "n_used": 33, "calib_days": days, **extra}
            if placebo:
                row.update(placebo_median=float("nan") if monitor == "early_rate" else 0.83, placebo_n=float("nan") if monitor == "early_rate" else 31)
            rows.append(row)
    return rows


def test_monitor_tables_mark_a_calibration_window_longer_than_a_quarter_and_say_why(tmp_path):
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    _csv(src, "monitors/monitor_eval.csv", _monitor_rows())
    for name in ("tab_monitors_full", "esm_ledger_monitors"):
        tex = BUILDERS[name][0](_fresh(src))
        mark = "\\textsuperscript{a}"
        assert _row(tex, "25H1-26H1" + mark, "absolute level", "domain\\_auc")                      # the 181-day window is marked on every one of its rows
        assert _row(tex, "25H1-26H1" + mark, "absolute level", "early\\_rate")
        assert _row(tex, "AQ3", "absolute level", "domain\\_auc")                                   # a quarter is not
        assert f"{mark} Longer calibration window: the domain-AUC level is not interpretable, only its rank." in tex, name


def test_a_calibration_window_of_exactly_the_limit_is_not_marked(tmp_path):
    from es.paper_tables import BUILDERS, LONG_CALIBRATION_DAYS
    src, _ = _sources(tmp_path)
    _csv(src, "monitors/monitor_eval.csv", _monitor_rows(calib_days=(LONG_CALIBRATION_DAYS, LONG_CALIBRATION_DAYS - 8)))
    assert "textsuperscript{a}" not in BUILDERS["tab_monitors_full"][0](src)
    _csv(src, "monitors/monitor_eval.csv", _monitor_rows(calib_days=(LONG_CALIBRATION_DAYS + 1, 92)))
    assert "25H1-26H1\\textsuperscript{a}" in BUILDERS["tab_monitors_full"][0](_fresh(src))


def test_monitor_tables_show_the_placebo_median_as_the_no_drift_reference_where_the_file_has_it(tmp_path):
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    _csv(src, "monitors/monitor_eval.csv", _monitor_rows())
    full_tex = BUILDERS["tab_monitors_full"][0](src)
    ledger = BUILDERS["esm_ledger_monitors"][0](_fresh(src))
    assert "no-drift" in full_tex and "reference" in full_tex.split("\\midrule")[0]
    domain, early = _row(full_tex, "AQ3", "absolute level", "domain\\_auc"), _row(full_tex, "AQ3", "absolute level", "early\\_rate")
    assert domain[3:] == ["33", "0.20", "0.050", "0.60", "0.83"] and early[-1] == "--"          # nodes, rho, p, AUROC, reference; the alert rate has none
    assert _row(ledger, "AQ3", "absolute level", "domain\\_auc")[-2:] == ["0.83", "31"]         # the ledger adds the nodes behind the reference
    assert _row(ledger, "AQ3", "absolute level", "early\\_rate")[-2:] == ["--", "--"]
    assert "not a monitor and enters no test" in full_tex


def test_monitor_tables_without_placebo_columns_have_no_reference_column(tmp_path):
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    _csv(src, "monitors/monitor_eval.csv", _monitor_rows(placebo=False))
    tex = BUILDERS["esm_ledger_monitors"][0](src)
    assert "no-drift" not in tex and len(_row(tex, "AQ3", "absolute level", "domain\\_auc")) == 7


def test_monitor_tables_need_calib_days(tmp_path):
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    _csv(src, "monitors/monitor_eval.csv", [{k: v for k, v in r.items() if k != "calib_days"} for r in _monitor_rows()])
    for name in ("tab_monitors_full", "esm_ledger_monitors"):
        with pytest.raises(LookupError, match="calib_days"):
            BUILDERS[name][0](_fresh(src))


# ---------------------------------------------------------------------------------------------- M1 and M2: a missing family, and constants that the code holds
# ---------------------------------------------------------------------------------------------- Ruling 100a: constants that the code holds are read when a table is built
import dataclasses  # noqa: E402


def test_the_audit_target_is_read_from_the_code_when_a_table_is_built(tmp_path, monkeypatch):
    import es.config
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    rows = []
    for a, alpha in enumerate((0.01, 0.025, 0.05, 0.1)):
        for k, seed in enumerate((0, 1, 7, 13, 42)):
            rows.append(_srow("2024-2025", "LEV6", seed=seed, alpha=alpha, failures=100 * a + k, median_log_error=0.10 + 0.01 * a,
                              **{"failures_band_0.67_1.5": 10 * a + 1, "failures_band_0.8_1.25": 10 * a + 2}))
    _csv(src, "legacy/summary.csv", rows)
    band = lambda: _rows(BUILDERS["tab_alpha_band"][0](_fresh(src)))                        # noqa: E731
    seeds = lambda: _row(BUILDERS["tab_seeds"][0](_fresh(src)), "2024$\\to$25", "absolute level")[2:]      # noqa: E731
    row = lambda: next(r for r in band() if r[:2] == ["2024$\\to$25", "absolute level"])    # noqa: E731
    assert row()[2:] == ["4", "104", "204", "304", "21", "22"]                              # the sweep at the primary seed; the bands at alpha 0.05
    assert seeds() == ["200--204", "0.12"]                                                  # over the five seeds, at alpha 0.05
    monkeypatch.setattr(es.config, "ALPHA", 0.025)                                          # the audit's target is read when the table is built
    assert row()[2:] == ["4", "104", "204", "304", "11", "12"]
    assert seeds() == ["100--104", "0.11"]
    assert "band at $\\alpha$=2.5\\%" in BUILDERS["tab_alpha_band"][0](_fresh(src))


def test_the_primary_seed_is_read_from_the_code_when_a_table_is_built(tmp_path, monkeypatch):
    import es.paper_tables as pt
    src, _ = _sources(tmp_path)
    rows = [_srow(a, "G6", e, seed=seed, failures=f, median_log_error=m)
            for a, e in (("2024-2025", "xgb"), ("2024-2025", "logit"), ("25H1-26H1", "xgb"), ("25H1-26H1", "logit"), ("frozen-26H1", "xgb"))
            for seed, f, m in ((42, 11, 0.11), (7, 77, 0.77))]
    _csv(src, "legacy/summary.csv", rows)
    assert _row(pt.BUILDERS["tab_estimators"][0](_fresh(src)), "calendar")[1:] == ["11 / 0.11"] * 2 + ["11 / 0.11"] * 2 + ["11 / 0.11"]
    assert _row(pt.BUILDERS["tab_block_audit"][0](_fresh(src)), "calendar")[2:] == ["11"] * 5
    monkeypatch.setitem(pt.CODE_CONSTANTS, "seedPrimary", 7)
    assert _row(pt.BUILDERS["tab_estimators"][0](_fresh(src)), "calendar")[1:] == ["77 / 0.77"] * 5
    assert _row(pt.BUILDERS["tab_block_audit"][0](_fresh(src)), "calendar")[2:] == ["77"] * 5


def test_the_p4_windows_are_read_from_the_code_when_a_table_is_built(tmp_path, monkeypatch):
    import es.paper_tables as pt
    src, _ = _sources(tmp_path)
    for window, alt in ((56, 28), (40, 20)):
        monkeypatch.setitem(pt.CODE_CONSTANTS, "pFourWindow", window)
        monkeypatch.setitem(pt.CODE_CONSTANTS, "pFourWindowAlt", alt)
        rows = (_activity("AQ3", "G1_24", "xgb", "P1", [1, 0], [0.10, 0.20]) + _activity("AQ3", "G1_24", "xgb", f"P4_{window}", [1, 1], [0.30, 0.50])
                + _activity("AQ3", "G1_24", "xgb", f"P4_{alt}", [0, 0], [0.05, 0.07]) + _activity("AQ3", "G1_24", "xgb", "P6", [1, 0], [0.11, 0.13]))
        _csv(src, "analogs/activity_all.csv", rows)
        _csv(src, "analogs/eta_selection.csv", [{"analysis": "AQ3", "eta": 0.01, "median_log_error": 0.222}])
        (src.out_dir / "analogs" / "protocol_inputs.json").write_text(json.dumps({"eta": 0.01}))
        policies = pt.BUILDERS["tab_policies"][0](_fresh(src))
        assert f"P4 ({window} d)" in policies and f"P4 ({alt} d)" in policies and policies.index(f"P4 ({window} d)") < policies.index(f"P4 ({alt} d)")
        assert _row(policies, "AQ3: 2025-Q3")[1:] == ["1 / 0.15", "2 / 0.40", "0 / 0.06", "-- / 0.22", "1 / 0.12"]     # P1, the primary window, the alternative, P5, P6
        ledger = [r for r in _rows(pt.BUILDERS["esm_ledger_analogs"][0](_fresh(src))) if r[0] == "AQ3"]
        assert [r[3] for r in ledger] == ["P1", f"P4\\_{window}", f"P4\\_{alt}", "P6"]
        assert pt.POLICY_NAME["P4"] == f"P4 rolling ({window} d)" and pt.POLICY_NAME[f"P4_{alt}"] == f"P4 rolling ({alt} d)"
        assert f"P4_{window}" in pt.POLICY_NAME and ("P4_56" in pt.POLICY_NAME) == (window == 56)
        _csv(src, "economics/summary.csv", [_econ_row("AQ3", "P1"), _econ_row("AQ3", "P4")])
        assert f"P4: rolling ({window} d)" in pt.BUILDERS["tab_econ_primary"][0](_fresh(src))


def test_the_policy_names_include_the_two_in_sample_policies_of_the_sealed_audit(monkeypatch):
    import es.paper_tables as pt
    ids = {"P2_season", "P2_all"}                                                                      # what es.sealed_run names them
    assert pt.POLICY_NAME["P2_season"] == "P2 in-sample (season)" and pt.POLICY_NAME["P2_all"] == "P2 in-sample (all seasons)"
    assert ids <= set(pt.POLICY_NAME) and {"P1", "P4", "P5", "P6", "P4@r*"} <= set(pt.POLICY_NAME)    # alongside the others, which are as they were
    others = {name for code, name in pt.POLICY_NAME.items() if code not in ids}
    assert pt.POLICY_NAME["P2_season"] != pt.POLICY_NAME["P2_all"] and not ({pt.POLICY_NAME[i] for i in ids} & others)
    assert pt.tex_escape(pt.POLICY_NAME["P2_season"]) == pt.POLICY_NAME["P2_season"]                  # plain text: no character that LaTeX would read
    monkeypatch.setitem(pt.CODE_CONSTANTS, "pFourWindow", 40)                                          # the P4 names still follow the code
    assert pt.POLICY_NAME["P4"] == "P4 rolling (40 d)" and pt.POLICY_NAME["P2_all"] == "P2 in-sample (all seasons)"


def test_the_fee_scenarios_are_those_of_the_economics_producer(tmp_path, monkeypatch):
    import es.paper_tables as pt
    import es.positions as positions
    from es.run_economics import UNIFORM_FEES
    assert pt._fee_order() == ("calibrated", *(str(f) for f in UNIFORM_FEES), "stress")      # the producer loops over ("calibrated",) + UNIFORM_FEES + ("stress",)
    monkeypatch.setattr(positions, "UNIFORM_FEES", (0.0, 0.3))
    assert pt._fee_order() == ("calibrated", "0.0", "0.3", "stress")
    src, _ = _sources(tmp_path)
    rows = [_econ_row("AQ3", "P1", fees=fee, NET=net, SELC=selc) for fee, net, selc in (("stress", -1.0, 0.05), ("0.3", 1.0, -0.1), ("calibrated", 5.0, 0.1), ("0.0", 6.0, 0.1))]
    _csv(src, "economics/summary.csv", rows)
    panel = pt.BUILDERS["tab_fee_position_signs"][0](src).split("\\begin{tabular}")[1]
    assert [r[1] for r in _rows(panel) if r[0] == "AQ3: 2025-Q3"] == ["calibrated", "0.0", "0.3", "stress"]           # in the order of the producer
    ledger = [r for r in _rows(pt.BUILDERS["esm_ledger_economics"][0](_fresh(src))) if r[0] == "AQ3"]
    assert [r[4] for r in ledger] == ["calibrated", "0.0", "0.3", "stress"]


def test_the_years_of_the_row_labels_come_from_the_windows_of_the_analyses(tmp_path, monkeypatch):
    import es.paper_tables as pt
    from es.windows import DateWindow, UtcWindow
    src, _ = _sources(tmp_path)
    _csv(src, "legacy/summary.csv", [_srow("sealed26H1", "LEV6"), _srow("25H1-26H1", "LEV6")])
    labels = lambda: [r[0] for r in _rows(pt.BUILDERS["tab_holdout"][0](_fresh(src)))[1:]]          # noqa: E731
    assert labels() == ["as opened: train 2023--24, calibrate 2025, deploy 2026-H1, $q$ on 2023--24",
                        "season-matched (post-open): calibrate 2025-H1, $q$ on 2023"]
    # the registry says something else, and the labels say it
    monkeypatch.setitem(pt.LEGACY, "sealed26H1", dataclasses.replace(pt.LEGACY["sealed26H1"], calib=UtcWindow("2024-07-01", "2025-01-01"),
                                                                     deploy=UtcWindow("2025-07-01", "2026-01-01")))
    monkeypatch.setitem(pt.LEGACY, "25H1-26H1", dataclasses.replace(pt.LEGACY["25H1-26H1"], calib=UtcWindow("2025-07-01", "2026-01-01"),
                                                                    label_train=UtcWindow("2024-01-01", "2025-01-01")))
    assert labels() == ["as opened: train 2023--24, calibrate 2024-H2, deploy 2025-H2, $q$ on 2023--24",
                        "season-matched (post-open): calibrate 2025-H2, $q$ on 2024"]
    monkeypatch.setitem(pt.ANALOGS, "AQ3", dataclasses.replace(pt.ANALOGS["AQ3"], deploy=DateWindow("2026-01-01", "2026-03-31")))
    monkeypatch.setitem(pt.SEALED, "SA", dataclasses.replace(pt.SEALED["SA"], deploy=DateWindow("2026-10-01", "2026-12-31")))
    monkeypatch.setitem(pt.LEGACY, "24H1-25H1", dataclasses.replace(pt.LEGACY["24H1-25H1"], deploy=UtcWindow("2025-07-01", "2026-01-01")))
    assert pt.ANALYSIS_NAME["AQ3"] == "AQ3: 2026-Q1" and pt.ANALYSIS_NAME["SA"] == "Test A: 2026-Q4"
    assert pt.ANALYSIS_NAME["24H1-25H1"] == pt.ANALYSIS_SHORT["24H1-25H1"] == "24H1$\\to$25H2"
    assert pt._xgb_logit_groups()[2:4] == [("2024$\\to$2025", 2), ("25H2$\\to$26H1", 2)] and "2026-H1" in pt._xgb_logit_groups()[4][0]
    monkeypatch.setitem(pt.LEGACY, "frozen-26H1", dataclasses.replace(pt.LEGACY["frozen-26H1"], deploy=UtcWindow("2026-07-01", "2027-01-01")))
    assert "2026-H2" in pt._xgb_logit_groups()[4][0]                                               # the frozen column's heading follows its window too
    monkeypatch.setitem(pt.ANALOGS, "AQ3", dataclasses.replace(pt.ANALOGS["AQ3"], deploy=DateWindow("2026-01-01", "2026-02-15")))
    with pytest.raises(ValueError, match="not whole calendar years, one half year or one quarter"):
        pt.ANALYSIS_NAME["AQ3"]


def test_the_names_of_the_analyses_are_those_the_tables_always_printed():
    import es.paper_tables as pt
    assert dict(pt.ANALYSIS_NAME) == {
        "24H1-25H1": "24H1$\\to$25H1", "24H2-25H2": "24H2$\\to$25H2", "24H2-25H2-train2023": "24H2$\\to$25H2 (old fit)", "2024-2025": "2024$\\to$2025",
        "25H1-26H1": "25H1$\\to$26H1 (post-open)", "sealed26H1": "as-opened 2026-H1", "frozen-26H1": "frozen-origin 2026-H1",
        "AQ3": "AQ3: 2025-Q3", "AQ4": "AQ4: 2025-Q4", "AQ1": "AQ1: 2026-Q1", "AQ2": "AQ2: 2026-Q2", "AQ3F": "AQ3 frozen-origin",
        "SA": "Test A: 2026-Q3", "SAF": "Test A frozen-origin", "SB": "Test B: 2026-Q4", "SBF": "Test B frozen-origin",
        "LC-full-q2023": "calibrate 2025, $q$ 2023", "LC-full-q2324": "calibrate 2025, $q$ 2023--24",
        "LC-h1-q2023": "calibrate 2025-H1, $q$ 2023", "LC-h1-q2324": "calibrate 2025-H1, $q$ 2023--24"}
    assert dict(pt.ANALYSIS_SHORT) == {"24H1-25H1": "24H1$\\to$25H1", "24H2-25H2": "24H2$\\to$25H2", "2024-2025": "2024$\\to$25", "25H1-26H1": "25H1$\\to$26H1",
                                      "sealed26H1": "as-opened 26H1", "frozen-26H1": "frozen 26H1", "AQ3+AQ4 (rehearsal of the pooled estimands)": "AQ3+AQ4 pooled (rehearsal)"}
    assert "nothing" not in pt.ANALYSIS_NAME and pt.ANALYSIS_NAME.get("nothing", "none") == "none"            # an id that is none of them is a missing key, not an error


def test_the_interval_level_and_the_early_days_are_read_from_the_code_when_a_table_is_built(tmp_path, monkeypatch):
    import es.paper_tables as pt
    for key, value in (("earlyDays", 10), ("bhLevel", 0.1), ("ciLevel", 90.0)):
        monkeypatch.setitem(pt.CODE_READERS, key, lambda value=value: value)
    src, _ = _sources(tmp_path)
    _csv(src, "monitors/monitor_eval.csv", _monitor_rows(monitors=("W1_norm", "PSI", "domain_auc", "early_rate")))
    assert "realised alert rate (10 d)" in pt.BUILDERS["tab_monitors"][0](src)
    for name in ("tab_monitors_full", "esm_ledger_monitors"):
        assert "its last 10 days against the earlier days" in pt.BUILDERS[name][0](_fresh(src))                # the placebo note says it too
    assert "$q$ = 0.1" in pt.BUILDERS["tab_bh_families"][0](src)
    _csv(src, "economics/bootstrap_ci.csv", _two_analyses())
    assert "with 90\\% bootstrap intervals" in pt.BUILDERS["tab_econ_ci"][0](src)
    assert "with 90\\% bootstrap intervals" in pt.BUILDERS["tab_decomposition"][0](_fresh(src))                # the decomposition caption too


# ---------------------------------------------------------------------------------------------- Ruling 100a: the labels of tab_econ_ci, the column orders, tab_implementation
CI_LABEL_EXPECTED = {
    "SELC": ("SELC", ""), "NET": ("NET", ""), "E2:SNET": ("SNET (\\$/MW-day)", ""), "E3:ESred": ("ESred", ""),
    "E4:SELC_decile": ("SELC, decile strata", ""),
    "E4:SELC|calendar@count": ("SELC at matched count", "calendar-only"), "E4:SELC|VOL@count": ("SELC at matched count", "VOL"),
    "E4:dSELC|calendar@count": ("paired difference, SELC", "P1 \\textminus{} calendar-only"),
    "E4:dSELC|VOL@count": ("paired difference, SELC", "P1 \\textminus{} VOL"),
    "E4:ESred|calendar@count": ("ESred at matched count", "calendar-only"), "E4:ESred|VOL@count": ("ESred at matched count", "VOL"),
    "E4:dESred|calendar@count": ("paired difference, ESred", "P1 \\textminus{} calendar-only"),
    "E4:dESred|VOL@count": ("paired difference, ESred", "P1 \\textminus{} VOL")}


def test_every_estimand_of_the_registry_is_printed_under_its_own_label_and_comparison(tmp_path):
    from es.paper_registry import STAT_FMT, STAT_ORDER
    from es.paper_tables import BUILDERS, CI_LABEL
    from es.paper_tex import FORMATS, PAIR_FORMATS
    src, _ = _sources(tmp_path)
    assert dict(CI_LABEL) == CI_LABEL_EXPECTED and set(STAT_ORDER) == set(CI_LABEL_EXPECTED)             # the mapping is pinned, and complete
    rows, expected = [], {}
    for i, stat in enumerate(STAT_ORDER, 1):                                                              # a different number for every estimand
        point_format, interval_format = STAT_FMT[stat]
        estimate = 1000.0 * i if point_format.startswith("usd") else 0.011 * i
        rows.append(_ci("AQ3", stat, estimate, 0.5 * estimate, 1.5 * estimate))
        expected[("AQ3: 2025-Q3",) + CI_LABEL_EXPECTED[stat]] = f"{FORMATS[point_format](estimate)} {PAIR_FORMATS[interval_format](0.5 * estimate, 1.5 * estimate)}"
    _csv(src, "economics/bootstrap_ci.csv", rows)
    assert len(set(expected.values())) == len(STAT_ORDER)
    assert _econ_ci_cells(BUILDERS["tab_econ_ci"][0](src)) == expected                                    # a swapped label would put a number under another name


def test_the_alpha_curve_prints_p1_before_p6_and_net_before_expected_shortfall(tmp_path):
    from es.paper_pins import ALPHAS
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    rows = []
    for aid, base in (("AQ3", 0), ("AQ4", 100000)):
        for i, alpha in enumerate(ALPHAS, 1):
            for policy, shift_net, shift_es in (("P1", 0, 0), ("P6", 5000, 7000)):
                rows.append(_econ_row(aid, "comparator", f"curve_{policy}_a{alpha}", NET=base + 1000 * i + shift_net, ESred=base + 100 * i + 1 + shift_es))
    _csv(src, "economics/summary.csv", rows)
    tex = BUILDERS["tab_alpha_curve"][0](src)
    assert _rows(tex)[0] == ["analysis", "$\\alpha$", "P1 NET", "P1 ESred", "P6 NET", "P6 ESred"]
    aq3 = [r for r in _rows(tex) if r[0] == "AQ3: 2025-Q3"]
    assert aq3[0][1:] == ["1\\%", "\\$1{,}000", "\\$101", "\\$6{,}000", "\\$7{,}101"]                      # P1 NET, P1 ESred, P6 NET, P6 ESred
    assert aq3[3][1:] == ["10\\%", "\\$4{,}000", "\\$401", "\\$9{,}000", "\\$7{,}401"]
    assert [r for r in _rows(tex) if r[0] == "AQ4: 2025-Q4"][1][1:] == ["2.5\\%", "\\$102{,}000", "\\$100{,}201", "\\$107{,}000", "\\$107{,}201"]


def test_the_block_audit_reads_xgb_in_the_frozen_column_and_both_estimators_in_the_others(tmp_path):
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    rows = []
    for analysis, xgb, logit in (("2024-2025", 11, 12), ("25H1-26H1", 13, 14), ("frozen-26H1", 15, 16)):
        rows += [_srow(analysis, "G6", "xgb", failures=xgb), _srow(analysis, "G6", "logit", failures=logit)]
    _csv(src, "legacy/summary.csv", rows)
    tex = BUILDERS["tab_block_audit"][0](src)
    assert _row(tex, "calendar")[2:] == ["11", "12", "13", "14", "15"]                                     # XGB, logit, XGB, logit, and XGB (not the logit 16) of the frozen origin
    assert _rows(tex)[0] == ["", "", "\\multicolumn{2}{c}{2024$\\to$2025}", "\\multicolumn{2}{c}{25H1$\\to$26H1}", "\\shortstack[c]{frozen\\\\2026-H1}"]


def test_the_fee_and_position_table_prints_net_before_selc_and_the_positions_in_order(tmp_path):
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    cases = (("premium", 5.0, -0.1), ("inc", -5.0, 0.1), ("dec", 5.0, 0.1), ("premium[short_spikes]", -5.0, -0.1), ("premium[long_spikes]", 0.0, 0.1))
    _csv(src, "economics/summary.csv", [_econ_row("AQ3", "P1", position=position, NET=net, SELC=selc) for position, net, selc in cases])
    tex = BUILDERS["tab_fee_position_signs"][0](src)
    row = next(r for r in _rows(tex.split("\\begin{tabular}")[1]) if r[:2] == ["AQ3: 2025-Q3", "calibrated"])
    assert row[2:] == ["+/\\textminus{}", "\\textminus{}/+", "+/+", "\\textminus{}/\\textminus{}", "0/+"]     # sign(NET)/sign(SELC), premium, INC, DEC, short spikes, long spikes
    head = _rows(tex.split("\\begin{tabular}")[1])[0]
    assert head[0:2] == ["analysis", "fees"] and "INC" in head[3] and "DEC" in head[4]


def test_the_label_control_table_prints_the_four_blocks_in_the_order_of_its_headings(tmp_path):
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    blocks = {"LEV6": (1, 1.01), "DART18": (2, 1.02), "G1_24": (3, 1.03), "DART3": (4, 1.04)}
    _csv(src, "labels_pr/label_control_2x2.csv", [{"analysis": "LC-full-q2023", "block": b, "failures": f, "n_nodes": 33, "median_R": r} for b, (f, r) in blocks.items()])
    tex = BUILDERS["tab_label_control"][0](src)
    assert _rows(tex)[0] == ["calibration window, label rule", "level", "DART-relative", "combined", "plain DART lags"]
    assert _row(tex, "calibrate 2025, $q$ 2023")[1:] == ["1 (1.01)", "2 (1.02)", "3 (1.03)", "4 (1.04)"]     # absolute level, DART-relative, combined, plain DART lags


def test_the_implementation_table_sets_its_own_font_size_and_reports_its_length_of_this_build(tmp_path):
    from es.paper_tables import NOTES, build_all
    src, paper = _sources(tmp_path)
    _write_impl(src)
    status = build_all(src, paper / "generated" / "tables")
    assert status["tab_implementation"] == "built"
    tex = (paper / "generated" / "tables" / "tab_implementation.tex").read_text()
    assert tex.splitlines()[1].startswith("\\begingroup\\footnotesize\\setlength{\\tabcolsep}{2.5pt}") and tex.rstrip().endswith("\\endgroup")
    line = next(l for l in (paper / "generated" / "tables" / "INDEX.md").read_text().splitlines() if l.startswith("| tab_implementation |"))
    assert re.search(r"this build: about \d+ pages long at footnotesize", line)                              # a measurement of this build...
    assert not re.search(r"\d+ pages|pages long", NOTES["tab_implementation"])                                # ...never typed into the note


def test_the_length_of_a_longtable_is_estimated_from_its_rows_and_never_splits_a_row():
    from es.paper_tables import HEAD_PT, LINE_PT, TEXT_HEIGHT_PT, _lines, _longtable_pages
    assert _lines(RR + "one two three", 200.0) == 1 and _lines(RR + "word " * 60, 100.0) > 10
    assert _lines("\\midrule" + RR + "short", 100.0) == 1
    capacity = int((TEXT_HEIGHT_PT - HEAD_PT) // LINE_PT)
    rows = [["x", "y"]] * (capacity - 4)
    assert _longtable_pages([60.0, 60.0], rows, 2.5, caption=False) == 1
    assert _longtable_pages([60.0, 60.0], rows + [["x", "y"]] * 6, 2.5, caption=False) == 2              # more rows than a page holds: the next page
    tall = [[RR + "word " * 60, "y"]]                                                                       # a row of many lines does not fit what is left: it starts a page
    assert _longtable_pages([60.0, 60.0], rows + tall, 2.5, caption=False) == 2
    assert _longtable_pages([60.0, 60.0], [["\\multicolumn{2}{l}{" + "word " * 30 + "}"]], 2.5, caption=True) == 1


def test_the_bh_row_of_a_family_that_was_not_computed_is_one_pending_cell(tmp_path):
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    _csv(src, "inference/bh.csv", _bh_rows(["audit robustness", "drift monitors"]))
    tex = BUILDERS["tab_bh_families"][0](src)
    assert _row(tex, "economic robustness") == ["economic robustness", "\\multicolumn{3}{c}{\\PENDINGDEV}"]       # one cell over the three columns, not three of them
    assert tex.count("\\PENDINGDEV") == 1


# ---------------------------------------------------------------------------------------------- M3, M5, M7
def test_index_notes_carry_no_typed_widths_or_page_counts():
    import es.paper_tables as pt
    for name, note in pt.NOTES.items():
        assert not re.search(r"\b\d+(\.\d+)?\s*(pt|bp|cm|mm)\b", note), (name, note)               # a width in a note is a measurement of one build, not a constant
        assert not re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(printed\s+)?pages?\b|\bpages? long\b", note, re.I), (name, note)    # nor is a length


@pytest.mark.parametrize("name, rel, dropped", [
    ("tab_analog_blocks", "analogs/activity_all.csv", "fail_0.5_2"), ("esm_ledger_analogs", "analogs/activity_all.csv", "node"),
    ("tab_policies", "analogs/eta_selection.csv", "median_log_error"), ("esm_ledger_pr", "labels_pr/pr_all.csv", "recall_eval"),
    ("tab_decomposition", "economics/bootstrap_ci.csv", "approximate")])
def test_a_renamed_column_is_a_lookup_error_that_names_it(full, name, rel, dropped):
    from es.paper_tables import BUILDERS
    src, _ = full
    frame = pd.read_csv(src.out_dir / rel)
    frame.drop(columns=[dropped]).to_csv(src.out_dir / rel, index=False)
    with pytest.raises(LookupError, match=dropped):
        BUILDERS[name][0](_fresh(src))


def test_the_unused_ex_caiso_constant_is_gone():
    import es.paper_tables as pt
    assert not hasattr(pt, "EX_CAISO")


def test_legacy_ledger_has_every_column_of_the_summary_in_two_longtables_with_the_same_rows(tmp_path):
    from es.paper_tables import BUILDERS
    src, _ = _sources(tmp_path)
    _csv(src, "legacy/summary.csv", [
        _srow("2024-2025", "G6", failures=7, median_R=0.91, median_log_error=0.12, median_auprc=0.21, auprc_lift=2.5, prevalence=0.071,
              n_zero_activity=2, n_nodes=34, **{"failures_band_0.67_1.5": 11, "failures_band_0.8_1.25": 13, "failures_NYISO": 3,
                                                 "failures_ISONE": 4, "failures_MISO": 5, "failures_CAISO": 6}),
        _srow("2024-2025", "G6", estimator="logit", seed=7, policy="P2_all_train", alpha=0.025, failures=8, n_nodes=33)])
    tex = BUILDERS["esm_ledger_legacy"][0](src)
    one, two = tex.split("\\clearpage")
    assert tex.count("\\begin{longtable}") == 2 and "tab:esm-legacy}" in one and "tab:esm-legacy-more}" in two
    assert _row(one, "2024-2025, G6")[1:] == ["xgb, s42, P1, 5\\%", "7", "0.91", "0.12", "0.21", "2.50", "0.071", "2"]   # setting, failures, R, log error, AUPRC, lift, prevalence, zero alerts
    assert _row(two, "2024-2025, G6")[1:] == ["xgb, s42, P1, 5\\%", "11", "13", "3", "4", "5", "6", "34"]              # bands, NYISO ISONE MISO CAISO, nodes
    second = [r for r in _rows(one) if r[0] == "2024-2025, G6"][1]
    assert second[1] == "logit, s7, P2\\_all\\_train, 2.5\\%"
    head = two.split("\\midrule")[0]
    assert "[0.67,1.5]" in head and "[0.8,1.25]" in head and "ISO-NE" in head and "nodes" in head
    assert "lift" in one.split("\\midrule")[0] and "prevalence" in one.split("\\midrule")[0]


# =====================================================================================================================
# The real outputs of the analogue study (controller addendum): copied here, never read from outputs/
# =====================================================================================================================
#: outputs/es/analogs/simulation.csv as the simulation stage wrote it (23 Oct 2026, 07:28), verbatim: the real schema, with the `law`,
#: `rho` and `mean_selc` columns and the other columns that the table does not read.
REAL_SIMULATION_CSV = """test,law,scenario,persistence,strata,n_sim,seed,reject_0.0166667,reject_0.05,mean_selc,mean_t_obs,rho,phi_scale,hetero,phi_nodes,phi_corrected,phi_clipped_high,phi_clipped_low,phi_no_signal,phi_uncorrected,target_selc,note,nodrift_fail_q95_0.5_2,nodrift_fail_q95_0.67_1.5,nodrift_fail_q95_0.8_1.25,mean_t_hat,mean_node_gap,delta,common_shift,phi_score_median,phi_score_raw_median,target_log_error_gap,calibration_seed,calibrated_delta
"H-E single A, size",,homoskedastic,fit,primary,300,20261022,0.0033333333333333335,0.023333333333333334,-7.820633278018685e-06,-0.012656927493947846,0.0,1.0,False,33,33,3,0,1,0,,,,,,,,,,,,,,
"H-E single A, size",,homoskedastic,fit,decile,300,20261023,0.0,0.02666666666666667,-0.0008996455185198235,-0.7674308792665178,0.0,1.0,False,33,33,3,0,1,0,,,,,,,,,,,,,,
"H-E single A, size",,heteroskedastic,fit,primary,300,20261024,0.01,0.043333333333333335,-0.00013440581787169185,-0.0808956962756673,0.0,1.0,True,33,33,3,0,1,0,,,,,,,,,,,,,,
"H-E single A, size",,heteroskedastic,fit,decile,300,20261025,0.0,0.0,-0.0002868051086151908,-0.14391956709545686,0.0,1.0,True,33,33,3,0,1,0,,,,,,,,,,,,,,
"H-E single A, size",,homoskedastic,upper,primary,300,20261026,0.01,0.05333333333333334,4.153018005731545e-05,0.04206252662473471,0.0,1.0,False,33,33,3,0,1,0,,,,,,,,,,,,,,
"H-E single A, size",,homoskedastic,upper,decile,300,20261027,0.0033333333333333335,0.016666666666666666,-0.0009138948376058143,-0.7947554075583451,0.0,1.0,False,33,33,3,0,1,0,,,,,,,,,,,,,,
"H-E single A, size",,heteroskedastic,upper,primary,300,20261028,0.01,0.043333333333333335,5.7400967129108534e-05,0.03946218219651848,0.0,1.0,True,33,33,3,0,1,0,,,,,,,,,,,,,,
"H-E single A, size",,heteroskedastic,upper,decile,300,20261029,0.0,0.0,-0.00019917990453333637,-0.0981049341098319,0.0,1.0,True,33,33,3,0,1,0,,,,,,,,,,,,,,
"H-E pooled A+B, size",,homoskedastic,fit,primary,300,20261030,0.01,0.043333333333333335,-3.581361875057553e-05,-0.045281586111041244,0.0,1.0,False,33,33,3,0,1,0,,,,,,,,,,,,,,
"H-E pooled A+B, size",,homoskedastic,fit,decile,300,20261031,0.0033333333333333335,0.023333333333333334,-0.0005849650610989026,-0.7050218004455953,0.0,1.0,False,33,33,3,0,1,0,,,,,,,,,,,,,,
"H-E pooled A+B, size",,heteroskedastic,fit,primary,300,20261032,0.023333333333333334,0.07,0.00012861748667637263,0.10307102437530813,0.0,1.0,True,33,33,3,0,1,0,,,,,,,,,,,,,,
"H-E pooled A+B, size",,heteroskedastic,fit,decile,300,20261033,0.0,0.0,0.0019000133237413897,1.3783050223885391,0.0,1.0,True,33,33,3,0,1,0,,,,,,,,,,,,,,
"H-E pooled A+B, size",,homoskedastic,upper,primary,300,20261034,0.02,0.04,-1.6448212874195108e-05,-0.023432575161730362,0.0,1.0,False,33,33,3,0,1,0,,,,,,,,,,,,,,
"H-E pooled A+B, size",,homoskedastic,upper,decile,300,20261035,0.0,0.006666666666666667,-0.0005457630982662939,-0.6590615802597988,0.0,1.0,False,33,33,3,0,1,0,,,,,,,,,,,,,,
"H-E pooled A+B, size",,heteroskedastic,upper,primary,300,20261036,0.013333333333333334,0.043333333333333335,0.00016432952898559424,0.13565773823950286,0.0,1.0,True,33,33,3,0,1,0,,,,,,,,,,,,,,
"H-E pooled A+B, size",,heteroskedastic,upper,decile,300,20261037,0.0,0.0,0.0016548089848126798,1.2133409882238764,0.0,1.0,True,33,33,3,0,1,0,,,,,,,,,,,,,,
"H-E pooled A+B, power curve",,homoskedastic,fit,primary,100,7,0.13,0.22,0.0010790136542288323,1.4177525473434465,0.05,1.0,False,33,33,3,0,1,0,,,,,,,,,,,,,,
"H-E pooled A+B, power curve",,homoskedastic,fit,primary,100,7,0.36,0.53,0.0020195368351771303,2.654045483605921,0.1,1.0,False,33,33,3,0,1,0,,,,,,,,,,,,,,
"H-E pooled A+B, power curve",,homoskedastic,fit,primary,100,7,0.87,0.96,0.003938325402913672,5.166281855582066,0.2,1.0,False,33,33,3,0,1,0,,,,,,,,,,,,,,
"H-E pooled A+B, power curve",,homoskedastic,fit,primary,100,7,1.0,1.0,0.0059078593618100165,7.730407460194031,0.3,1.0,False,33,33,3,0,1,0,,,,,,,,,,,,,,
"H-E pooled A+B, power curve",,homoskedastic,fit,primary,100,7,1.0,1.0,0.009999317705203738,13.007495819320939,0.5,1.0,False,33,33,3,0,1,0,,,,,,,,,,,,,,
"H-E pooled A+B, power curve",,homoskedastic,fit,primary,100,7,1.0,1.0,0.014303443316146015,18.48258656964568,0.7,1.0,False,33,33,3,0,1,0,,,,,,,,,,,,,,
"H-E pooled A+B, power curve",,homoskedastic,fit,primary,100,7,1.0,1.0,0.018843228298752977,24.167181563169073,0.9,1.0,False,33,33,3,0,1,0,,,,,,,,,,,,,,
"H-E pooled A+B, power",,homoskedastic,fit,primary,300,11,0.09333333333333334,0.21,0.0010482008073017294,1.372373432034457,0.05,1.0,False,33,33,3,0,1,0,-0.002641386800581041,target below the grid,,,,,,,,,,,,
H-B size,DART3,no_drift,fit,,300,20261038,0.06,0.10333333333333333,,,,1.0,,33,33,0,0,0,0,,,9.0,17.0,25.0,0.0014351502844212915,-0.0016363757523990634,0.0,0.0,0.7876892089843749,0.6869921969527949,,,
H-B size,DART3,common_drift_0.25,fit,,300,20261039,0.08666666666666667,0.15666666666666668,,,,1.0,,33,33,0,0,0,0,,,,,,0.0007935719174274986,-0.001177429165757476,0.0,0.25,0.7876892089843749,0.6869921969527949,,,
H-B size,DART3,common_drift_0.5,fit,,300,20261040,0.12666666666666668,0.19333333333333333,,,,1.0,,33,33,0,0,0,0,,,,,,0.0016163148213851196,-0.0014340594671884595,0.0,0.5,0.7876892089843749,0.6869921969527949,,,
H-B size,DART3,no_drift,upper,,300,20261041,0.043333333333333335,0.08333333333333333,,,,1.0,,33,33,0,0,0,0,,,12.0,20.0,27.0,-0.0005666395027756278,-0.002223251486899279,0.0,0.0,0.7876892089843749,0.6869921969527949,,,
H-B size,DART3,common_drift_0.25,upper,,300,20261042,0.09333333333333334,0.13666666666666666,,,,1.0,,33,33,0,0,0,0,,,,,,-0.0008132306400446763,0.0010962101627610263,0.0,0.25,0.7876892089843749,0.6869921969527949,,,
H-B size,DART3,common_drift_0.5,upper,,300,20261043,0.10666666666666667,0.18666666666666668,,,,1.0,,33,33,0,0,0,0,,,,,,0.0017347161294869007,-0.0001603482571806968,0.0,0.5,0.7876892089843749,0.6869921969527949,,,
H-B size,LEV6,no_drift,fit,,300,20261044,0.05,0.12,,,,1.0,,33,33,0,0,0,0,,,12.0,20.0,27.0,-0.0006655205980951932,0.0008804196731778672,0.0,0.0,0.772821044921875,0.6788219867335992,,,
H-B size,LEV6,common_drift_0.25,fit,,300,20261045,0.07,0.14,,,,1.0,,33,33,0,0,0,0,,,,,,-0.003660569531541996,-0.0008737200018733926,0.0,0.25,0.772821044921875,0.6788219867335992,,,
H-B size,LEV6,common_drift_0.5,fit,,300,20261046,0.12,0.18666666666666668,,,,1.0,,33,33,0,0,0,0,,,,,,0.001039239130184237,-0.0027874619278973004,0.0,0.5,0.772821044921875,0.6788219867335992,,,
H-B size,LEV6,no_drift,upper,,300,20261047,0.07,0.11666666666666667,,,,1.0,,33,33,0,0,0,0,,,14.050000000000011,22.0,28.0,0.0006669372509114204,0.0004414226261008432,0.0,0.0,0.772821044921875,0.6788219867335992,,,
H-B size,LEV6,common_drift_0.25,upper,,300,20261048,0.1,0.16333333333333333,,,,1.0,,33,33,0,0,0,0,,,,,,-0.0006293107159457516,-0.000709153699490949,0.0,0.25,0.772821044921875,0.6788219867335992,,,
H-B size,LEV6,common_drift_0.5,upper,,300,20261049,0.13,0.18666666666666668,,,,1.0,,33,33,0,0,0,0,,,,,,0.0011970207483622366,-0.0024348409536967025,0.0,0.5,0.772821044921875,0.6788219867335992,,,
H-B size,ROLL_DISP9,no_drift,fit,,300,20261050,0.043333333333333335,0.08,,,,1.0,,33,33,0,0,0,0,,,9.0,17.0,25.0,0.0010750754485944084,-0.00047239218827498114,0.0,0.0,0.7696350097656249,0.6762744927855636,,,
H-B size,ROLL_DISP9,common_drift_0.25,fit,,300,20261051,0.09666666666666666,0.16333333333333333,,,,1.0,,33,33,0,0,0,0,,,,,,0.0030643804214887924,0.0032238933352889323,0.0,0.25,0.7696350097656249,0.6762744927855636,,,
H-B size,ROLL_DISP9,common_drift_0.5,fit,,300,20261052,0.09,0.15333333333333332,,,,1.0,,33,33,0,0,0,0,,,,,,-0.0005967069047801821,-0.0021726872718667426,0.0,0.5,0.7696350097656249,0.6762744927855636,,,
H-B size,ROLL_DISP9,no_drift,upper,,300,20261053,0.05333333333333334,0.11,,,,1.0,,33,33,0,0,0,0,,,13.0,21.0,27.0,-0.0018084744061961477,-0.0005285119087608979,0.0,0.0,0.7696350097656249,0.6762744927855636,,,
H-B size,ROLL_DISP9,common_drift_0.25,upper,,300,20261054,0.08666666666666667,0.14,,,,1.0,,33,33,0,0,0,0,,,,,,0.0013444121587584809,0.0027412677676285236,0.0,0.25,0.7696350097656249,0.6762744927855636,,,
H-B size,ROLL_DISP9,common_drift_0.5,upper,,300,20261055,0.07333333333333333,0.12666666666666668,,,,1.0,,33,33,0,0,0,0,,,,,,-0.006512241701467006,-0.0034212919392673236,0.0,0.5,0.7696350097656249,0.6762744927855636,,,
H-B1 power,DART3,no_drift,fit,,300,13,0.28,0.4266666666666667,,,,1.0,,33,33,0,0,0,0,,,,,,0.08452835402815723,0.0707573762252972,0.3194993448765063,0.0,0.7876892089843749,0.6869921969527949,0.08544503268493674,20261022.0,0.3194993448765063
H-B2 power,DART3,no_drift,fit,,300,13,0.05,0.13,,,,1.0,,33,33,0,0,0,0,,,,,,0.01093085424501083,0.005101797223577497,0.13928203394486055,0.0,0.7876892089843749,0.6869921969527949,0.015085734079089468,20261022.0,0.13928203394486055
"""
#: outputs/es/analogs/protocol_inputs.json of the same run (stage "complete"), the keys that the tables and the rule read: eta, family,
#: stage, rule (the record of the registered rule: levels, laws, k_rows, ...), rho_star and the placebo pairings. The list `simulation`,
#: which repeats the CSV, and the fit records (phi, phi_status, analog_selc) are left out.
REAL_PROTOCOL_INPUTS = {'eta': 0.05,
 'family': {'H-E': {'confirmatory': True,
                    'reasons': [],
                    'approximate': False,
                    'interim_approximate': False,
                    'k_rows': 4,
                    'k_rows_interim': 4,
                    'placebo_counted': {'single': 4, 'pooled': 1},
                    'false_demotion_bound': 0.07357713220329704},
            'H-B1': {'confirmatory': False,
                     'reasons': ['simulated size 0.130 > 1.5 x 0.0167 (39 of 300 replications, worst of 12 rows: LEV6, common_drift_0.5, upper)',
                                 'simulated size 0.193 > 1.5 x 0.0500 (58 of 300 replications, worst of 12 rows: DART3, common_drift_0.5, fit)',
                                 'simulated power 0.28 at 0.05/3 is below 0.5'],
                     'power': 0.28,
                     'power_lower_bound': False,
                     'approximate': True,
                     'k_rows': 12,
                     'false_demotion_bound': 0.0520643927912553},
            'H-B2': {'confirmatory': False,
                     'reasons': ['simulated size 0.127 > 1.5 x 0.0167 (38 of 300 replications, worst of 12 rows: DART3, common_drift_0.5, fit)',
                                 'simulated size 0.193 > 1.5 x 0.0500 (58 of 300 replications, worst of 12 rows: DART3, common_drift_0.5, fit)',
                                 'simulated power 0.05 at 0.05/3 is below 0.5'],
                     'power': 0.05,
                     'power_lower_bound': False,
                     'approximate': True,
                     'k_rows': 12,
                     'false_demotion_bound': 0.0520643927912553},
            'notes': ['Pooled placebo check (1 counted 1-year pairing): H-E is replaced by its estimand when its p is at most 0.0167 (false-demotion '
                      'probability under a correctly sized test ≤ 0.0167). It cannot demote at 0.0500. The 2-year pairings (their P&L quarter lies '
                      "inside the alert model's training window) are reported with counted = false and do not enter the rule. The check is retained "
                      'as registered.',
                      'False-demotion bounds under independent rows, for a correctly sized test (union over the two levels): H-E 0.074 (simulation, '
                      '4 rows, 0.057; placebo 0.0167), H-B1 0.052 (12 rows), H-B2 0.052 (12 rows).']},
 'stage': 'complete',
 'rule': {'factor': 1.5,
          'binomial_level': 0.05,
          'approximate_factor': 1.0,
          'min_power': 0.5,
          'levels': [0.016666666666666666, 0.05],
          'clipped_high_share': 0.1,
          'k_rows': {'H-E': 4, 'H-B1': 12, 'H-B2': 12},
          'k_rows_interim': 4,
          'laws': {'H-B1': ['DART3', 'LEV6'], 'H-B2': ['DART3', 'ROLL_DISP9']},
          'n_rep': 299,
          'n_sim': {'he_size': 300, 'he_rho': 100, 'he_power': 300, 'hb_size': 300, 'hb_power': 300, 'hb_calibrate': 100},
          'seeds': {'size_base': 20261022,
                    'he_size_first_index': 0,
                    'hb_size_first_index': 16,
                    'rho_search': 7,
                    'he_power': 11,
                    'hb_power': 13,
                    'calibration': 20261022},
          'calibration_grid': [0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8],
          'counted_pairings': ['AQ3-1y', 'AQ4-1y', 'AQ1-1y', 'AQ2-1y', 'AQ3+AQ4-1y']},
 'rho_star': 0.05,
 'placebo': [{'pair': 'AQ3-1y',
              't_obs': 1.3657861655062646,
              'p': 0.2717391304347826,
              'n_null': 91,
              'selc_obs': 0.01212047145389822,
              'n_rows': 72864,
              'n_settle_approx': 0,
              'counted': True},
             {'pair': 'AQ3-2y',
              't_obs': -1.567419389543626,
              'p': 0.5869565217391305,
              'n_null': 91,
              'selc_obs': -0.011558964995859207,
              'n_rows': 72864,
              'n_settle_approx': 1728,
              'counted': False,
              'note': "P&L year inside the alert model's training window"},
             {'pair': 'AQ4-1y',
              't_obs': -1.4560689931318311,
              'p': 0.717391304347826,
              'n_null': 91,
              'selc_obs': -0.004634573478025674,
              'n_rows': 72636,
              'n_settle_approx': 0,
              'counted': True},
             {'pair': 'AQ4-2y',
              't_obs': 3.5497042934237144,
              'p': 0.11956521739130435,
              'n_null': 91,
              'selc_obs': 0.012571950114412507,
              'n_rows': 72617,
              'n_settle_approx': 0,
              'counted': False,
              'note': "P&L year inside the alert model's training window"},
             {'pair': 'AQ1-1y',
              't_obs': -4.332786144508482,
              'p': 0.9333333333333333,
              'n_null': 89,
              'selc_obs': -0.010316075151596977,
              'n_rows': 71230,
              'n_settle_approx': 0,
              'counted': True},
             {'pair': 'AQ1-2y',
              't_obs': -0.4455371688582598,
              'p': 0.5,
              'n_null': 89,
              'selc_obs': -0.001255577069441954,
              'n_rows': 71230,
              'n_settle_approx': 0,
              'counted': False,
              'note': "P&L year inside the alert model's training window"},
             {'pair': 'AQ2-1y',
              't_obs': 4.42433290664691,
              'p': 0.26373626373626374,
              'n_null': 90,
              'selc_obs': 0.043635420162238925,
              'n_rows': 71992,
              'n_settle_approx': 0,
              'counted': True},
             {'pair': 'AQ2-2y',
              't_obs': 3.959902280651898,
              'p': 0.06593406593406594,
              'n_null': 90,
              'selc_obs': 0.024780055507711292,
              'n_rows': 72072,
              'n_settle_approx': 0,
              'counted': False,
              'note': "P&L year inside the alert model's training window"},
             {'pair': 'AQ3+AQ4-1y',
              't_obs': 0.8388761633726948,
              'p': 0.3405009451795841,
              'n_null': 8463,
              'selc_obs': 0.004077616869944558,
              'n_rows': 145500,
              'n_settle_approx': 0,
              'counted': True},
             {'pair': 'AQ3+AQ4-2y',
              't_obs': -0.32371866154862,
              'p': 0.46242911153119093,
              'n_null': 8463,
              'selc_obs': -0.0014695075340209773,
              'n_rows': 145481,
              'n_settle_approx': 1728,
              'counted': False,
              'note': "P&L year inside the alert model's training window"}]}
