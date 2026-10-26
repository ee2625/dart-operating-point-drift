"""Ruling 101 (the Ruling 66d re-keying): the size, power, family and rule macros of the simulation stage.

analogs/simulation.csv has one row per simulated setting, so a macro that names no setting matches several rows. Each macro
here reads what the registered rule reads (es.run_analogs._test_rows): the rows under the primary strata and, for an H-B test,
the rows of its own laws. The worst case is the maximum and the row that governs it is the first one with that maximum.
Every macro is tested against a fixture with the real file's columns and row structure (with values planted so that a
wrong selection gives another number), and then against the real file."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REAL_OUT = Path(__file__).resolve().parents[2] / "outputs" / "es"

SIM_COLUMNS = [
    "test", "law", "scenario", "persistence", "strata", "n_sim", "seed", "reject_0.0166667", "reject_0.05", "mean_selc",
    "mean_t_obs", "rho", "phi_scale", "hetero", "phi_nodes", "phi_corrected", "phi_clipped_high", "phi_clipped_low",
    "phi_no_signal", "phi_uncorrected", "target_selc", "note", "nodrift_fail_q95_0.5_2", "nodrift_fail_q95_0.67_1.5",
    "nodrift_fail_q95_0.8_1.25", "mean_t_hat", "mean_node_gap", "delta", "common_shift", "phi_score_median",
    "phi_score_raw_median", "target_log_error_gap", "calibration_seed", "calibrated_delta"]
THIRD, FIVE = "reject_0.0166667", "reject_0.05"

SIZE_PREFIXES = ("heSizePooled", "heSizeSingle", "hbOneSize", "hbTwoSize")
LEVELS = ("Third", "Five")
SIZE_MACROS = tuple(f"{p}{lv}{part}" for p in SIZE_PREFIXES for lv in LEVELS
                    for part in ("", "WorstRow", "Law", "Scenario", "Persistence")) + tuple(f"{p}NSim" for p in SIZE_PREFIXES)
RHO_WORDS = ("RhoFive", "RhoTen", "RhoTwenty", "RhoThirty", "RhoFifty", "RhoSeventy", "RhoNinety")
RHOS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
CURVE_MACROS = tuple(f"hePowerCurve{kind}{rw}" for rw in RHO_WORDS for kind in ("Third", "Five", "Selc")) + ("hePowerCurveNSim",)
TARGET_MACROS = ("hePowerTargetThird", "hePowerTargetFive", "hePowerTargetRho", "hePowerTargetSelc", "hePowerTargetMeanSelc",
                 "hePowerTargetNSim", "hePowerTargetNote")
HB_POWER_MACROS = ("hbOnePowerThird", "hbOnePowerFive", "hbTwoPowerThird", "hbTwoPowerFive")
FAMILY_MACROS = tuple(f"{h}{k}" for h in ("he", "hbOne", "hbTwo")
                      for k in ("FamilyStatus", "Approximate", "Reasons", "FalseDemotion")) + (
    "heInterimApproximate", "hbOnePowerLowerBound", "hbTwoPowerLowerBound")
SPLIT_MACROS = ("heFalseDemotionSim", "heFalseDemotionPlacebo")
RULE_MACROS = ("ruleFactor", "ruleApproxFactor", "ruleBinomLevel", "ruleMinPower", "ruleLevelThird", "ruleLevelFive", "ruleKHe",
               "ruleKHbOne", "ruleKHbTwo", "ruleKHeInterim", "nSimHeSize", "nSimHeRho", "nSimHePower", "nSimHbSize",
               "nSimHbPower", "nSimHbCalibrate", "ruleNRep")
PHI_MACROS = tuple(f"phi{kind}{b}" for b in ("Dart", "Lev", "Disp") for kind in ("Corrected", "Raw"))
ETA_MACROS = ("etaAtGridTop",)
PLACEBO_MACROS = ("nPlaceboPairsDescriptive", "nPlaceboRejectFiveDescriptive", "placeboPooledP", "placeboPooledSelc",
                  "placeboSinglesPMin")
NEW_MACROS = (SIZE_MACROS + CURVE_MACROS + TARGET_MACROS + HB_POWER_MACROS + FAMILY_MACROS + SPLIT_MACROS + RULE_MACROS
              + PHI_MACROS + ETA_MACROS + PLACEBO_MACROS)
#: Read from the same file under Ruling 100 (the no-drift context of the audit counts): the extremes over the laws and persistences.
NODRIFT_MACROS = tuple(f"noDriftFailNinetyFive{bw}{w}" for bw in ("Wide", "Mid", "Tight") for w in ("Min", "Max"))
#: The old names stay, as aliases of the new primary entries, so that the sections that use them compile.
ALIASES = {"heSizePooled": "heSizePooledFive", "heSizeSingle": "heSizeSingleFive", "hePowerPooled": "hePowerTargetThird",
           "hbOnePower": "hbOnePowerThird", "hbTwoPower": "hbTwoPowerThird"}
#: Entries that read these files and are not new: unchanged, or the modified placebo counts.
OLD_KEPT = ("etaStar", "rhoStar", "heConfirmatory", "hbOneConfirmatory", "hbTwoConfirmatory", "nPlaceboPairs",
            "nPlaceboRejectFive", "nPlaceboOneYear", "nPlaceboOneYearReject", "placeboOneYearPMin", "nPlaceboTwoYear",
            "nPlaceboTwoYearReject", "placeboTwoYearPMin")

#: Task 8 (Ruling 100, Section 4.6): the no-drift context of the audit counts, read from the size study of the portability bootstrap.
NO_DRIFT_MACROS = tuple(f"noDriftFailNinetyFive{band}{end}" for band in ("Wide", "Mid", "Tight") for end in ("Min", "Max"))


# ------------------------------------------------------------------------------------------------ fixtures
HE_SETTINGS = [("homoskedastic", "fit", "primary"), ("homoskedastic", "fit", "decile"),
               ("heteroskedastic", "fit", "primary"), ("heteroskedastic", "fit", "decile"),
               ("homoskedastic", "upper", "primary"), ("homoskedastic", "upper", "decile"),
               ("heteroskedastic", "upper", "primary"), ("heteroskedastic", "upper", "decile")]
HB_SETTINGS = [(sc, per) for per in ("fit", "upper") for sc in ("no_drift", "common_drift_0.25", "common_drift_0.5")]
#: Planted rejection rates, in file order. The decile rows are decoys (higher than the primary ones); H-B1 does not read
#: ROLL_DISP9 and H-B2 does not read LEV6 (their decoys are the 0.140 and the 0.200 of ROLL_DISP9, and LEV6's 0.187 ties).
HE_SINGLE = {"third": [0.003, 0.090, 0.010, 0.080, 0.010, 0.070, 0.010, 0.060],           # the primary maximum is a three-way tie
             "five": [0.020, 0.200, 0.043, 0.150, 0.053, 0.140, 0.043, 0.130]}
HE_POOLED = {"third": [0.010, 0.050, 0.023, 0.040, 0.020, 0.030, 0.013, 0.020],
             "five": [0.043, 0.150, 0.070, 0.120, 0.040, 0.110, 0.043, 0.100]}
HB_SIZE = {"DART3": {"third": [0.060, 0.087, 0.113, 0.043, 0.093, 0.107], "five": [0.103, 0.157, 0.193, 0.083, 0.137, 0.187]},
           "LEV6": {"third": [0.050, 0.070, 0.120, 0.070, 0.100, 0.130], "five": [0.120, 0.140, 0.187, 0.117, 0.163, 0.187]},
           "ROLL_DISP9": {"third": [0.043, 0.097, 0.090, 0.053, 0.140, 0.073], "five": [0.080, 0.163, 0.200, 0.110, 0.140, 0.127]}}
#: nodrift_fail_q95_<band>: the 95th percentile over replications of the nodes outside the band without drift, (fit, upper) by law
NODRIFT_FAIL = {"0.5_2": {"DART3": (8.0, 11.0), "LEV6": (12.0, 15.5), "ROLL_DISP9": (10.0, 13.0)},
                "0.67_1.5": {"DART3": (16.0, 18.0), "LEV6": (17.0, 24.5), "ROLL_DISP9": (19.0, 21.0)},
                "0.8_1.25": {"DART3": (24.0, 27.0), "LEV6": (25.0, 30.0), "ROLL_DISP9": (26.0, 29.0)}}
CURVE = {"third": [0.11, 0.34, 0.85, 0.99, 1.0, 1.0, 1.0], "five": [0.21, 0.51, 0.95, 1.0, 1.0, 1.0, 1.0],
         "selc": [0.00112, 0.00208, 0.00391, 0.00587, 0.01003, 0.01431, 0.01884]}


def _row(test, **kw):
    row = {c: np.nan for c in SIM_COLUMNS}
    row.update(test=test, phi_scale=1.0)
    row.update(kw)
    return row


def sim_fixture() -> pd.DataFrame:
    """The 44 rows of the real simulation.csv, with its columns, its row order and planted values."""
    rows, seed = [], 20261022
    for test, table in (("H-E single A, size", HE_SINGLE), ("H-E pooled A+B, size", HE_POOLED)):
        for i, (scenario, persistence, strata) in enumerate(HE_SETTINGS):
            rows.append(_row(test, scenario=scenario, persistence=persistence, strata=strata, n_sim=300, seed=seed, rho=0.0,
                             hetero=(scenario == "heteroskedastic"), mean_selc=0.0001 * i, mean_t_obs=0.1 * i,
                             phi_nodes=33, phi_corrected=33, phi_clipped_high=3, phi_clipped_low=0, phi_no_signal=1,
                             phi_uncorrected=0, **{THIRD: table["third"][i], FIVE: table["five"][i]}))
            seed += 1
    for i, rho in enumerate(RHOS):
        rows.append(_row("H-E pooled A+B, power curve", scenario="homoskedastic", persistence="fit", strata="primary",
                         n_sim=100, seed=7, rho=rho, hetero=False, mean_selc=CURVE["selc"][i], mean_t_obs=1.5 * (i + 1),
                         phi_nodes=33, phi_corrected=33, phi_clipped_high=3, phi_clipped_low=0, phi_no_signal=1,
                         phi_uncorrected=0, **{THIRD: CURVE["third"][i], FIVE: CURVE["five"][i]}))
    rows.append(_row("H-E pooled A+B, power", scenario="homoskedastic", persistence="fit", strata="primary", n_sim=300,
                     seed=11, rho=0.05, hetero=False, mean_selc=0.001048, mean_t_obs=1.37, target_selc=-0.002641,
                     note="target below the grid", phi_nodes=33, phi_corrected=33, phi_clipped_high=3, phi_clipped_low=0,
                     phi_no_signal=1, phi_uncorrected=0, **{THIRD: 0.093333, FIVE: 0.21}))
    seed = 20261038
    for law, table in HB_SIZE.items():
        for i, (scenario, persistence) in enumerate(HB_SETTINGS):
            rows.append(_row("H-B size", law=law, scenario=scenario, persistence=persistence, n_sim=300, seed=seed,
                             mean_t_hat=0.001 * i, mean_node_gap=-0.001, delta=0.0, common_shift=(0.0, 0.25, 0.5)[i % 3],
                             phi_score_median=0.78, phi_score_raw_median=0.68,
                             **({f"nodrift_fail_q95_{band}": NODRIFT_FAIL[band][law][persistence == "upper"] for band in NODRIFT_FAIL}
                                if scenario == "no_drift" else {}),
                             **{THIRD: table["third"][i], FIVE: table["five"][i]}))
            seed += 1
    for hid, third, five in (("H-B1 power", 0.31, 0.45), ("H-B2 power", 0.04, 0.12)):
        rows.append(_row(hid, law="DART3", scenario="no_drift", persistence="fit", n_sim=300, seed=13, mean_t_hat=0.08,
                         mean_node_gap=0.07, delta=0.32, common_shift=0.0, phi_score_median=0.78, phi_score_raw_median=0.68,
                         target_log_error_gap=0.085, calibration_seed=20261022.0, calibrated_delta=0.32,
                         phi_nodes=33, phi_corrected=33, phi_clipped_high=0, phi_clipped_low=0, phi_no_signal=0,
                         phi_uncorrected=0, **{THIRD: third, FIVE: five}))
    return pd.DataFrame(rows, columns=SIM_COLUMNS)


PLACEBO_COLUMNS = ["pair", "t_obs", "p", "n_null", "selc_obs", "n_rows", "n_settle_approx", "counted", "note"]


def placebo_fixture() -> pd.DataFrame:
    note = "P&L year inside the alert model's training window"
    rows = [("AQ3-1y", 1.37, 0.272, 91, 0.0121, True, ""), ("AQ3-2y", -1.57, 0.587, 91, -0.0116, False, note),
            ("AQ4-1y", -1.46, 0.717, 91, -0.0046, True, ""), ("AQ4-2y", 3.55, 0.040, 91, 0.0126, False, note),
            ("AQ1-1y", -4.33, 0.933, 89, -0.0103, True, ""), ("AQ1-2y", -0.45, 0.500, 89, -0.0013, False, note),
            ("AQ2-1y", 4.42, 0.231, 90, 0.0436, True, ""), ("AQ2-2y", 3.96, 0.066, 90, 0.0248, False, note),
            ("AQ3+AQ4-1y", 0.84, 0.341, 8463, 0.00408, True, ""), ("AQ3+AQ4-2y", -0.32, 0.462, 8463, -0.0015, False, note)]
    return pd.DataFrame([dict(zip(PLACEBO_COLUMNS, (r[0], r[1], r[2], r[3], r[4], 72000, 0, r[5], r[6] or np.nan)))
                         for r in rows], columns=PLACEBO_COLUMNS)


REASONS = {"H-B1": ["simulated size 0.130 > 1.5 x 0.0167 (39 of 300 replications, worst of 12 rows: LEV6, common_drift_0.5, upper)",
                    "simulated size 0.193 > 1.5 x 0.0500 (58 of 300 replications, worst of 12 rows: DART3, common_drift_0.5, fit)",
                    "simulated power 0.28 at 0.05/3 is below 0.5"],
           "H-B2": ["simulated size 0.127 > 1.5 x 0.0167 (38 of 300 replications, worst of 12 rows: DART3, common_drift_0.5, fit)"]}
HE_BOUND, HB_BOUND = 0.07357713220329704, 0.0520643927912553      # what run_analogs stored for these rule constants and row counts


def protocol_fixture(**over) -> dict:
    """analogs/protocol_inputs.json at stage "complete", with the real file's keys."""
    doc = {
        "eta": 0.05, "stage": "complete",
        "family": {
            "H-E": {"confirmatory": True, "reasons": [], "approximate": False, "interim_approximate": False, "k_rows": 4,
                    "k_rows_interim": 4, "placebo_counted": {"single": 4, "pooled": 1}, "false_demotion_bound": HE_BOUND},
            "H-B1": {"confirmatory": False, "reasons": REASONS["H-B1"], "power": 0.31, "power_lower_bound": False,
                     "approximate": True, "k_rows": 12, "false_demotion_bound": HB_BOUND},
            "H-B2": {"confirmatory": True, "reasons": [], "power": 0.04, "power_lower_bound": True,
                     "approximate": False, "k_rows": 12, "false_demotion_bound": HB_BOUND},
            "notes": ["Pooled placebo check (1 counted 1-year pairing): ...", "False-demotion bounds ..."]},
        "rule": {"factor": 1.5, "binomial_level": 0.05, "approximate_factor": 1.0, "min_power": 0.5,
                 "levels": [0.05 / 3, 0.05], "clipped_high_share": 0.1,
                 "k_rows": {"H-E": 4, "H-B1": 12, "H-B2": 12}, "k_rows_interim": 4,
                 "laws": {"H-B1": ["DART3", "LEV6"], "H-B2": ["DART3", "ROLL_DISP9"]}, "n_rep": 299,
                 "n_sim": {"he_size": 300, "he_rho": 100, "he_power": 300, "hb_size": 300, "hb_power": 300, "hb_calibrate": 100},
                 "seeds": {"size_base": 20261022, "he_size_first_index": 0, "hb_size_first_index": 16, "rho_search": 7,
                           "he_power": 11, "hb_power": 13, "calibration": 20261022},
                 "calibration_grid": [0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8],
                 "counted_pairings": ["AQ3-1y", "AQ4-1y", "AQ1-1y", "AQ2-1y", "AQ3+AQ4-1y"]},
        "analog_selc": [-0.0063, -0.0043], "analog_selc_pooled": -0.00528, "rho_star": 0.05,
        "phi": {"DART3": {"phi_score_median": 0.7877, "phi_score_raw_median": 0.6870},
                "LEV6": {"phi_score_median": 0.7728, "phi_score_raw_median": 0.6788},
                "ROLL_DISP9": {"phi_score_median": 0.7696, "phi_score_raw_median": 0.6763}},
        "phi_status": {"H-E": {"phi_nodes": 33}}, "simulation": [], "placebo": []}
    doc.update(over)
    return doc


def eta_fixture(grid=(0.005, 0.01, 0.02, 0.05)) -> pd.DataFrame:
    return pd.DataFrame([{"analysis": a, "eta": e, "median_log_error": 0.1 + e} for a in ("AQ3", "AQ4") for e in grid])


def write_inputs(tmp_path, sim=None, doc=None, placebo=None, eta=None) -> Path:
    out = tmp_path / "out"
    (out / "analogs").mkdir(parents=True, exist_ok=True)
    (sim if sim is not None else sim_fixture()).to_csv(out / "analogs" / "simulation.csv", index=False)
    (out / "analogs" / "protocol_inputs.json").write_text(json.dumps(doc if doc is not None else protocol_fixture(), indent=1))
    (placebo if placebo is not None else placebo_fixture()).to_csv(out / "analogs" / "placebo_he.csv", index=False)
    (eta if eta is not None else eta_fixture()).to_csv(out / "analogs" / "eta_selection.csv", index=False)
    return out


def sources(tmp_path):
    from es.paper_macros import Sources
    return Sources(tmp_path / "out", tmp_path / "paper", tmp_path / "docs_sealed")


def resolved(name, tmp_path):
    from es.paper_macros import resolve
    from es.paper_registry import entries_by_name
    return resolve(entries_by_name()[name], sources(tmp_path))


def value(name, tmp_path):
    r = resolved(name, tmp_path)
    assert r["status"] == "ok", (name, r)
    return r["value"]


def values(names, tmp_path):
    return {n: value(n, tmp_path) for n in names}


# ------------------------------------------------------------------------------------------------ the registry
def test_the_registry_has_every_ruling_101_name_and_no_selector_on_the_noise_columns():
    from es.paper_registry import ENTRIES, entries_by_name
    from es.paper_tex import FORMATS, PAIR_FORMATS, ROW_FORMATS
    by = entries_by_name()
    assert len(set(NEW_MACROS)) == len(NEW_MACROS) and set(NEW_MACROS) <= set(by)
    reading = {e.name for e in ENTRIES if e.source in ("analogs/simulation.csv", "analogs/protocol_inputs.json",
                                                      "analogs/placebo_he.csv") or e.source.startswith("derived:")}
    # nothing reads these files that this module does not test: the new names, the aliases and the unchanged ones
    size_check = {"sizeCheckPooledThird", "sizeCheckPooledFive", "sizeCheckSingleThird", "sizeCheckSingleFive", "sizeCheckReps", "sizeCheckPhi"}
    expected = (set(NEW_MACROS) | set(ALIASES) | {"hbSize", "placeboFallbackDays"} | set(OLD_KEPT) | set(NODRIFT_MACROS) | set(NO_DRIFT_MACROS)
                | size_check)                                           # Ruling 118, M8: the size check of the review, tested in test_paper_g1.py
    assert reading == expected, sorted(reading ^ expected)
    for e in ENTRIES:                                                    # hetero and phi_scale are noise, not settings
        assert "hetero" not in e.query and "phi_scale" not in e.query, e.name
    assert all(e.fmt in FORMATS or e.fmt in PAIR_FORMATS or e.fmt in ROW_FORMATS for e in ENTRIES)
    assert all(by[n].stage == "dev" for n in NEW_MACROS)


def test_the_old_names_are_aliases_of_the_primary_entries_and_nothing_else_is_ambiguous():
    from es.paper_registry import entries_by_name
    by = entries_by_name()
    for old, new in ALIASES.items():
        a, p = by[old], by[new]
        assert (a.source, a.query, a.column, a.fmt, a.agg, a.by, a.nullable) == \
            (p.source, p.query, p.column, p.fmt, p.agg, p.by, p.nullable), old
        assert new in a.note and "alias" in a.note
    # hbSize, "the worst-case size of the portability bootstrap", is the larger of the two H-B tests' worst cases at the
    # family-wise level: the maximum over the rows of both tests' laws
    h = by["hbSize"]
    assert (h.agg, h.by, h.column, h.fmt) == ("argmax", FIVE, FIVE, "dec3") and "hbOneSizeFive" in h.note and "hbTwoSizeFive" in h.note
    for law in ("DART3", "LEV6", "ROLL_DISP9"):
        assert law in h.query


# ------------------------------------------------------------------------------------------------ size
SIZE_EXPECTED = {
    "heSizePooled": {"Third": ("0.023", "heteroskedastic, fit", "n/a", "heteroskedastic", "fit"),
                     "Five": ("0.070", "heteroskedastic, fit", "n/a", "heteroskedastic", "fit")},
    "heSizeSingle": {"Third": ("0.010", "heteroskedastic, fit", "n/a", "heteroskedastic", "fit"),         # a tie: the first row
                     "Five": ("0.053", "homoskedastic, upper", "n/a", "homoskedastic", "upper")},
    "hbOneSize": {"Third": ("0.130", "LEV6, common\\_drift\\_0.5, upper", "LEV6", "common\\_drift\\_0.5", "upper"),
                  "Five": ("0.193", "DART3, common\\_drift\\_0.5, fit", "DART3", "common\\_drift\\_0.5", "fit")},
    "hbTwoSize": {"Third": ("0.140", "ROLL\\_DISP9, common\\_drift\\_0.25, upper", "ROLL\\_DISP9", "common\\_drift\\_0.25", "upper"),
                  "Five": ("0.200", "ROLL\\_DISP9, common\\_drift\\_0.5, fit", "ROLL\\_DISP9", "common\\_drift\\_0.5", "fit")}}


def test_size_macros_read_the_rows_the_rule_reads_and_name_the_row_that_governs(tmp_path):
    write_inputs(tmp_path)
    for prefix, levels in SIZE_EXPECTED.items():
        for level, (rate, row, law, scenario, persistence) in levels.items():
            got = values([f"{prefix}{level}", f"{prefix}{level}WorstRow", f"{prefix}{level}Law", f"{prefix}{level}Scenario",
                          f"{prefix}{level}Persistence"], tmp_path)
            assert list(got.values()) == [rate, row, law, scenario, persistence], (prefix, level, got)
        assert value(f"{prefix}NSim", tmp_path) == "300"
    # the old names are aliases: the family-wise level, and hbSize the larger of the two H-B tests
    assert values(["heSizePooled", "heSizeSingle", "hbSize"], tmp_path) == {"heSizePooled": "0.070", "heSizeSingle": "0.053",
                                                                            "hbSize": "0.200"}


def test_a_changed_simulation_file_changes_every_size_macro_it_feeds(tmp_path):
    sim = sim_fixture()
    write_inputs(tmp_path, sim=sim)
    before = values(SIZE_MACROS, tmp_path)
    sim.loc[(sim["test"] == "H-E pooled A+B, size") & (sim["strata"] == "primary") & (sim["scenario"] == "heteroskedastic")
            & (sim["persistence"] == "upper"), THIRD] = 0.031                         # a new governing row at the first level
    sim.loc[(sim["test"] == "H-B size") & (sim["law"] == "LEV6") & (sim["scenario"] == "no_drift")
            & (sim["persistence"] == "upper"), FIVE] = 0.250                          # H-B1's new worst case; H-B2 does not read LEV6
    sim.loc[sim["test"].isin(["H-E single A, size", "H-E pooled A+B, size", "H-B size"]), "n_sim"] = 500
    write_inputs(tmp_path, sim=sim)
    after = values(SIZE_MACROS, tmp_path)
    assert after["heSizePooledThird"] == "0.031" and after["heSizePooledThirdWorstRow"] == "heteroskedastic, upper"
    assert after["hbOneSizeFive"] == "0.250" and after["hbOneSizeFiveWorstRow"] == "LEV6, no\\_drift, upper"
    assert after["hbTwoSizeFive"] == before["hbTwoSizeFive"] == "0.200"                # H-B2 never reads LEV6
    assert {after[f"{p}NSim"] for p in SIZE_PREFIXES} == {"500"}
    assert values(["hbSize"], tmp_path) == {"hbSize": "0.250"}


def test_decile_rows_and_the_other_tests_laws_never_govern(tmp_path):
    """Decile strata are the sensitivity of Ruling 68; H-B1 reads DART3 and LEV6 only, H-B2 DART3 and ROLL_DISP9 only."""
    sim = sim_fixture()
    sim.loc[sim["strata"] == "decile", [THIRD, FIVE]] = 0.9                          # every decile row of both H-E tests
    write_inputs(tmp_path, sim=sim)
    assert values(["heSizePooledThird", "heSizePooledFive", "heSizeSingleThird", "heSizeSingleFive"], tmp_path) == {
        "heSizePooledThird": "0.023", "heSizePooledFive": "0.070", "heSizeSingleThird": "0.010", "heSizeSingleFive": "0.053"}
    sim = sim_fixture()
    sim.loc[(sim["test"] == "H-B size") & (sim["law"] == "LEV6"), [THIRD, FIVE]] = 0.5      # a decoy for H-B2, the worst of H-B1
    write_inputs(tmp_path, sim=sim)
    assert values(["hbOneSizeThird", "hbOneSizeFive", "hbTwoSizeThird", "hbTwoSizeFive"], tmp_path) == {
        "hbOneSizeThird": "0.500", "hbOneSizeFive": "0.500", "hbTwoSizeThird": "0.140", "hbTwoSizeFive": "0.200"}
    sim = sim_fixture()
    sim.loc[(sim["test"] == "H-B size") & (sim["law"] == "ROLL_DISP9"), [THIRD, FIVE]] = 0.6   # a decoy for H-B1
    write_inputs(tmp_path, sim=sim)
    assert values(["hbOneSizeThird", "hbOneSizeFive", "hbTwoSizeThird", "hbTwoSizeFive"], tmp_path) == {
        "hbOneSizeThird": "0.130", "hbOneSizeFive": "0.193", "hbTwoSizeThird": "0.600", "hbTwoSizeFive": "0.600"}


def test_a_missing_rate_among_the_registered_rows_is_an_error_not_a_smaller_maximum(tmp_path):
    """The rule refuses a NaN rejection rate (a size check must not pass on a missing number); so do the macros."""
    sim = sim_fixture()
    nan_row = (sim["test"] == "H-E pooled A+B, size") & (sim["strata"] == "primary") & (sim["scenario"] == "heteroskedastic") \
        & (sim["persistence"] == "fit")
    sim.loc[nan_row, FIVE] = np.nan
    write_inputs(tmp_path, sim=sim)
    from es.paper_macros import resolve
    from es.paper_registry import entries_by_name
    with pytest.raises(LookupError, match=r"heSizePooledFive.*reject_0.05"):
        resolve(entries_by_name()["heSizePooledFive"], sources(tmp_path))
    assert value("heSizePooledThird", tmp_path) == "0.023"                 # the other level has all its rates
    sim.loc[nan_row, FIVE] = 0.07
    sim.loc[(sim["test"] == "H-E pooled A+B, size") & (sim["strata"] == "decile"), FIVE] = np.nan   # rows the rule does not read
    write_inputs(tmp_path, sim=sim)
    assert value("heSizePooledFive", tmp_path) == "0.070"


def test_n_sim_is_one_number_for_the_registered_rows_or_an_error(tmp_path):
    sim = sim_fixture()
    sim.loc[(sim["test"] == "H-B size") & (sim["law"] == "LEV6") & (sim["scenario"] == "no_drift"), "n_sim"] = 200
    write_inputs(tmp_path, sim=sim)
    assert value("hbTwoSizeNSim", tmp_path) == "300"                       # H-B2 does not read LEV6
    with pytest.raises(LookupError, match=r"hbOneSizeNSim.*differ.*n_sim"):
        resolved("hbOneSizeNSim", tmp_path)


def test_a_missing_or_old_simulation_file_is_pending_or_names_the_column(tmp_path):
    from es.paper_macros import PENDING
    (tmp_path / "out").mkdir()
    for name in SIZE_MACROS + CURVE_MACROS + TARGET_MACROS + HB_POWER_MACROS:
        r = resolved(name, tmp_path)
        assert r["status"] == "pending" and r["value"] == PENDING["dev"] and "simulation.csv" in r["detail"], name
    old = sim_fixture().drop(columns=["strata"])                          # a file from before Ruling 68
    write_inputs(tmp_path, sim=old)
    with pytest.raises(LookupError, match=r"heSizePooledFive.*strata"):
        resolved("heSizePooledFive", tmp_path)


def test_no_drift_context_macros_are_the_extremes_over_the_no_drift_rows_of_all_the_laws(tmp_path):
    """Ruling 100: descriptive context, not a test: the smallest and the largest, over the three laws and the two persistence
    settings, of the 95th percentile of the number of nodes outside the band without drift."""
    sim = sim_fixture()
    write_inputs(tmp_path, sim=sim)
    assert values(NODRIFT_MACROS, tmp_path) == {
        "noDriftFailNinetyFiveWideMin": "8", "noDriftFailNinetyFiveWideMax": "15.5", "noDriftFailNinetyFiveMidMin": "16",
        "noDriftFailNinetyFiveMidMax": "24.5", "noDriftFailNinetyFiveTightMin": "24", "noDriftFailNinetyFiveTightMax": "30"}
    drift = (sim["test"] == "H-B size") & (sim["scenario"] != "no_drift")
    sim.loc[drift, [f"nodrift_fail_q95_{band}" for band in NODRIFT_FAIL]] = 99.0      # rows with drift are not the no-drift context
    sim.loc[(sim["test"] == "H-B size") & (sim["law"] == "ROLL_DISP9") & (sim["scenario"] == "no_drift")
            & (sim["persistence"] == "upper"), "nodrift_fail_q95_0.5_2"] = 20.0        # a changed source changes the extreme
    sim.loc[(sim["test"] == "H-B size") & (sim["law"] == "DART3") & (sim["scenario"] == "no_drift")
            & (sim["persistence"] == "fit"), "nodrift_fail_q95_0.8_1.25"] = 22.5
    write_inputs(tmp_path, sim=sim)
    got = values(NODRIFT_MACROS, tmp_path)
    assert (got["noDriftFailNinetyFiveWideMax"], got["noDriftFailNinetyFiveWideMin"]) == ("20", "8")
    assert (got["noDriftFailNinetyFiveTightMin"], got["noDriftFailNinetyFiveTightMax"]) == ("22.5", "30")
    assert (got["noDriftFailNinetyFiveMidMin"], got["noDriftFailNinetyFiveMidMax"]) == ("16", "24.5")


# ------------------------------------------------------------------------------------------------ power
def test_power_curve_target_and_h_b_power_macros(tmp_path):
    write_inputs(tmp_path)
    got = values(CURVE_MACROS + TARGET_MACROS + HB_POWER_MACROS[:3], tmp_path)
    for i, rw in enumerate(RHO_WORDS):
        assert got[f"hePowerCurveThird{rw}"] == ["0.11", "0.34", "0.85", "0.99", "1.00", "1.00", "1.00"][i]
        assert got[f"hePowerCurveFive{rw}"] == ["0.21", "0.51", "0.95", "1.00", "1.00", "1.00", "1.00"][i]
        assert got[f"hePowerCurveSelc{rw}"] == ["0.11\\%", "0.21\\%", "0.39\\%", "0.59\\%", "1.00\\%", "1.43\\%", "1.88\\%"][i]
    assert got["hePowerCurveNSim"] == "100"
    assert (got["hePowerTargetThird"], got["hePowerTargetFive"], got["hePowerTargetRho"], got["hePowerTargetNSim"]) == (
        "0.09", "0.21", "0.05", "300")
    assert (got["hePowerTargetSelc"], got["hePowerTargetMeanSelc"]) == ("\\textminus{}0.26\\%", "0.10\\%")
    assert got["hePowerTargetNote"] == "target below the grid"
    assert (got["hbOnePowerThird"], got["hbOnePowerFive"], got["hbTwoPowerThird"]) == ("0.31", "0.45", "0.04")
    assert values(["hePowerPooled", "hbOnePower", "hbTwoPower"], tmp_path) == {
        "hePowerPooled": "0.09", "hbOnePower": "0.31", "hbTwoPower": "0.04"}                  # the old names: the rates at alpha / 3


def test_h_b_power_prints_n_a_for_a_nan_and_the_target_note_for_a_target_inside_the_grid(tmp_path):
    sim = sim_fixture()
    sim.loc[sim["test"] == "H-B2 power", FIVE] = np.nan
    sim.loc[sim["test"] == "H-B1 power", [THIRD, FIVE]] = np.nan                        # a failed calibration
    sim.loc[sim["test"] == "H-E pooled A+B, power", "note"] = np.nan
    write_inputs(tmp_path, sim=sim)
    assert values(HB_POWER_MACROS, tmp_path) == {"hbOnePowerThird": "n/a", "hbOnePowerFive": "n/a",
                                                  "hbTwoPowerThird": "0.04", "hbTwoPowerFive": "n/a"}
    assert value("hePowerTargetNote", tmp_path) == "n/a"
    sim.loc[sim["test"] == "H-E pooled A+B, power", THIRD] = np.nan               # but the H-E rates are never optional
    write_inputs(tmp_path, sim=sim)
    with pytest.raises(LookupError, match="hePowerTargetThird"):
        resolved("hePowerTargetThird", tmp_path)


def test_power_rows_that_match_twice_or_not_at_all_are_errors(tmp_path):
    sim = sim_fixture()
    write_inputs(tmp_path, sim=pd.concat([sim, sim[sim["test"] == "H-B1 power"]], ignore_index=True))
    with pytest.raises(LookupError, match=r"hbOnePowerThird.*matched 2 rows"):
        resolved("hbOnePowerThird", tmp_path)
    write_inputs(tmp_path, sim=sim[sim["test"] != "H-E pooled A+B, power"])
    with pytest.raises(LookupError, match=r"hePowerTargetThird.*matched 0 rows"):
        resolved("hePowerTargetThird", tmp_path)
    write_inputs(tmp_path, sim=sim[~((sim["test"] == "H-E pooled A+B, power curve") & (sim["rho"] == 0.3))])
    with pytest.raises(LookupError, match=r"hePowerCurveFiveRhoThirty.*matched 0 rows"):
        resolved("hePowerCurveFiveRhoThirty", tmp_path)


# ------------------------------------------------------------------------------------------------ the registered family
def test_family_macros_read_the_structured_fields(tmp_path):
    write_inputs(tmp_path)
    got = values(FAMILY_MACROS, tmp_path)
    assert (got["heFamilyStatus"], got["hbOneFamilyStatus"], got["hbTwoFamilyStatus"]) == ("confirmatory", "estimand", "confirmatory")
    assert (got["heApproximate"], got["heInterimApproximate"], got["hbOneApproximate"], got["hbTwoApproximate"]) == (
        "no", "no", "yes", "no")
    assert (got["hbOnePowerLowerBound"], got["hbTwoPowerLowerBound"]) == ("no", "yes")
    assert (got["heFalseDemotion"], got["hbOneFalseDemotion"], got["hbTwoFalseDemotion"]) == ("0.074", "0.052", "0.052")
    assert got["heReasons"] == "none" and got["hbTwoReasons"] == "none"                  # no reason is not an empty macro
    assert got["hbOneReasons"] == ("simulated size 0.130 \\textgreater{} 1.5 x 0.0167 (39 of 300 replications, worst of 12 rows: "
                                   "LEV6, common\\_drift\\_0.5, upper); simulated size 0.193 \\textgreater{} 1.5 x 0.0500 (58 of "
                                   "300 replications, worst of 12 rows: DART3, common\\_drift\\_0.5, fit); simulated power 0.28 at "
                                   "0.05/3 is below 0.5")
    assert values(["heConfirmatory", "hbOneConfirmatory"], tmp_path) == {"heConfirmatory": "yes", "hbOneConfirmatory": "no"}
    doc = protocol_fixture()                                              # a changed source changes the values
    doc["family"]["H-E"].update(confirmatory=False, reasons=["1/1 placebo rejections at 0.0167"], approximate=True,
                                interim_approximate=True, false_demotion_bound=0.1)
    doc["family"]["H-B2"].update(power_lower_bound=False, false_demotion_bound=0.0123)
    write_inputs(tmp_path, doc=doc)
    changed = values(FAMILY_MACROS, tmp_path)
    assert (changed["heFamilyStatus"], changed["heApproximate"], changed["heInterimApproximate"]) == ("estimand", "yes", "yes")
    assert changed["heReasons"] == "1/1 placebo rejections at 0.0167" and changed["heFalseDemotion"] == "0.100"
    assert (changed["hbTwoPowerLowerBound"], changed["hbTwoFalseDemotion"]) == ("no", "0.012")


def test_a_family_status_that_is_not_a_boolean_is_an_error(tmp_path):
    doc = protocol_fixture()
    doc["family"]["H-B1"]["confirmatory"] = "perhaps"
    write_inputs(tmp_path, doc=doc)
    with pytest.raises(LookupError, match="hbOneFamilyStatus"):
        resolved("hbOneFamilyStatus", tmp_path)


def test_the_false_demotion_split_is_recomputed_from_the_rule_and_agrees_with_the_stored_bound(tmp_path):
    """H-E's stored bound is the simulation rows' bound plus the pooled placebo check's. The split is recomputed from the rule
    constants (factor, levels, binomial level), the registered rows' n_sim and the number of counted pooled pairings, and it
    must add up to the structured total."""
    write_inputs(tmp_path)
    assert values(SPLIT_MACROS, tmp_path) == {"heFalseDemotionSim": "0.057", "heFalseDemotionPlacebo": "0.0167"}
    r = resolved("heFalseDemotionSim", tmp_path)
    assert "4 rows" in r["detail"] and "0.0736" in r["detail"]
    doc = protocol_fixture()
    doc["family"]["H-E"]["false_demotion_bound"] = 0.2                        # a stored total that the recomputation cannot match
    write_inputs(tmp_path, doc=doc)
    for name in SPLIT_MACROS:
        with pytest.raises(LookupError, match=rf"{name}.*false_demotion_bound"):
            resolved(name, tmp_path)
    # two counted pooled pairings: the placebo bound is the sum over the two levels of the tail of the smallest demoting
    # count. At alpha / 3 one rejection of two (0.5 > 1.5 alpha / 3) has probability 1 - (1 - alpha / 3) ** 2; at alpha one
    # rejection has 1 - 0.95 ** 2 = 0.0975, which is not below 0.05, and two have 0.05 ** 2.
    by_hand = 1 - (1 - 0.05 / 3) ** 2 + 0.05 ** 2
    doc = protocol_fixture()
    doc["family"]["H-E"].update(placebo_counted={"single": 4, "pooled": 2}, false_demotion_bound=HE_BOUND - 0.05 / 3 + by_hand)
    write_inputs(tmp_path, doc=doc)
    assert value("heFalseDemotionPlacebo", tmp_path) == "0.0356" and value("heFalseDemotionSim", tmp_path) == "0.057"
    # a rule with a different factor changes the simulation part, so the stored total no longer matches
    doc = protocol_fixture()
    doc["rule"]["factor"] = 2.0
    write_inputs(tmp_path, doc=doc)
    with pytest.raises(LookupError, match="heFalseDemotionSim"):
        resolved("heFalseDemotionSim", tmp_path)


def test_json_macros_are_pending_while_the_file_is_at_the_analogs_stage(tmp_path):
    """Ruling 98 for the new keys: the sim stage writes the family, the rule and phi; until then they are pending."""
    from es.paper_macros import PENDING
    out = tmp_path / "out"
    (out / "analogs").mkdir(parents=True)
    (out / "analogs" / "protocol_inputs.json").write_text(json.dumps({"eta": 0.05, "family": None, "stage": "analogs", "placebo": []}))
    eta_fixture().to_csv(out / "analogs" / "eta_selection.csv", index=False)
    assert value("etaAtGridTop", tmp_path) == "yes"                       # stage one has both of its inputs
    for name in FAMILY_MACROS + SPLIT_MACROS + RULE_MACROS + PHI_MACROS:
        r = resolved(name, tmp_path)
        assert r["status"] == "pending" and r["value"] == PENDING["dev"], (name, r)
    assert "stage 'analogs'" in resolved("hbOneFamilyStatus", tmp_path)["detail"]
    (out / "analogs" / "protocol_inputs.json").write_text(json.dumps({"eta": 0.05, "stage": "complete", "placebo": []}))
    with pytest.raises(LookupError, match="ruleFactor"):                     # complete without a rule: a registry bug again
        resolved("ruleFactor", tmp_path)


# ------------------------------------------------------------------------------------------------ rule, phi and eta
def test_rule_constants_phi_medians_and_eta(tmp_path):
    write_inputs(tmp_path)
    got = values(RULE_MACROS + PHI_MACROS + ETA_MACROS, tmp_path)
    assert (got["ruleFactor"], got["ruleApproxFactor"], got["ruleBinomLevel"], got["ruleMinPower"]) == ("1.5", "1", "0.05", "0.5")
    assert (got["ruleLevelThird"], got["ruleLevelFive"]) == ("0.0167", "0.05")
    assert (got["ruleKHe"], got["ruleKHbOne"], got["ruleKHbTwo"], got["ruleKHeInterim"]) == ("4", "12", "12", "4")
    assert [got[n] for n in ("nSimHeSize", "nSimHeRho", "nSimHePower", "nSimHbSize", "nSimHbPower", "nSimHbCalibrate")] == [
        "300", "100", "300", "300", "300", "100"]
    assert got["ruleNRep"] == "299"
    assert [got[n] for n in PHI_MACROS] == ["0.788", "0.687", "0.773", "0.679", "0.770", "0.676"]
    assert got["etaAtGridTop"] == "yes" and value("etaStar", tmp_path) == "0.05"
    doc = protocol_fixture()                                              # a changed source changes the values
    doc["rule"].update(factor=2.0, binomial_level=0.1, min_power=0.8, n_rep=199)
    doc["rule"]["n_sim"]["hb_calibrate"] = 50
    doc["rule"]["k_rows"]["H-B2"] = 9
    doc["phi"]["LEV6"]["phi_score_raw_median"] = 0.5
    write_inputs(tmp_path, doc=doc)
    changed = values(RULE_MACROS + PHI_MACROS, tmp_path)
    assert (changed["ruleFactor"], changed["ruleBinomLevel"], changed["ruleMinPower"], changed["ruleNRep"]) == ("2", "0.1", "0.8", "199")
    assert (changed["nSimHbCalibrate"], changed["ruleKHbTwo"], changed["phiRawLev"]) == ("50", "9", "0.500")


def test_eta_at_the_top_of_the_grid_is_read_from_the_selection_table(tmp_path):
    write_inputs(tmp_path)
    assert value("etaAtGridTop", tmp_path) == "yes"
    doc = protocol_fixture(eta=0.02)                                       # chosen inside the grid
    write_inputs(tmp_path, doc=doc)
    assert value("etaAtGridTop", tmp_path) == "no" and value("etaStar", tmp_path) == "0.02"
    write_inputs(tmp_path, eta=eta_fixture(grid=(0.005, 0.01, 0.02, 0.05, 0.1)))     # a grid that goes further up than eta*
    assert value("etaAtGridTop", tmp_path) == "no"
    write_inputs(tmp_path, doc=protocol_fixture(eta=0.03))                 # eta* is not a point of the grid at all
    with pytest.raises(LookupError, match=r"etaAtGridTop.*not a point of the grid"):
        resolved("etaAtGridTop", tmp_path)
    (tmp_path / "out" / "analogs" / "eta_selection.csv").unlink()
    assert resolved("etaAtGridTop", tmp_path)["status"] == "pending"


# ------------------------------------------------------------------------------------------------ the build-time cross-check
def test_the_registry_laws_row_counts_levels_and_rho_grid_are_checked_against_the_rule(tmp_path):
    from es.paper_macros import check_simulation_rule
    out = write_inputs(tmp_path)
    check_simulation_rule(sources(tmp_path))                              # consistent: silent
    doc = protocol_fixture()
    doc["rule"]["laws"]["H-B2"] = ["DART3", "LEV6"]                        # a rule that reads other laws
    write_inputs(tmp_path, doc=doc)
    with pytest.raises(LookupError, match=r"H-B2.*laws"):
        check_simulation_rule(sources(tmp_path))
    doc = protocol_fixture()
    doc["rule"]["k_rows"]["H-B1"] = 6                                      # the registry reads 12 rows for H-B1
    doc["rule"]["k_rows_interim"] = 8
    write_inputs(tmp_path, doc=doc)
    with pytest.raises(LookupError) as exc:
        check_simulation_rule(sources(tmp_path))
    assert "hbOneSize" in str(exc.value) and "heSizeSingle" in str(exc.value)
    doc = protocol_fixture()
    doc["rule"]["levels"] = [0.02, 0.05]                                   # the columns are named after the levels
    write_inputs(tmp_path, doc=doc)
    with pytest.raises(LookupError, match=r"levels"):
        check_simulation_rule(sources(tmp_path))
    sim = sim_fixture()
    write_inputs(tmp_path, sim=sim[~((sim["test"] == "H-E pooled A+B, power curve") & (sim["rho"] == 0.9))])
    with pytest.raises(LookupError, match=r"rho"):
        check_simulation_rule(sources(tmp_path))
    sim = sim_fixture()
    sim.loc[(sim["test"] == "H-E pooled A+B, power curve") & (sim["rho"] == 0.9), "rho"] = 0.95
    write_inputs(tmp_path, sim=sim)
    with pytest.raises(LookupError, match=r"rho"):
        check_simulation_rule(sources(tmp_path))
    # nothing to compare: a missing file, or a file that has not reached the end of the sim stage
    (out / "analogs" / "simulation.csv").unlink()
    check_simulation_rule(sources(tmp_path))
    write_inputs(tmp_path, doc={"eta": 0.05, "family": None, "stage": "analogs", "placebo": []})
    check_simulation_rule(sources(tmp_path))


def test_main_runs_the_rule_check_before_it_writes(tmp_path):
    from es.paper_macros import main
    out = write_inputs(tmp_path)
    doc = protocol_fixture()
    doc["rule"]["laws"]["H-B1"] = ["LEV6"]
    write_inputs(tmp_path, doc=doc)
    paper = tmp_path / "paper"
    paper.mkdir()
    with pytest.raises(LookupError, match="laws"):
        main(["--out-dir", str(out), "--paper-dir", str(paper), "--docs-sealed", str(tmp_path / "docs"),
              "--only", "etaStar"])
    assert not (paper / "generated" / "macros.tex").exists()


# ------------------------------------------------------------------------------------------------ the placebo counts (O4)
def test_placebo_counts_read_the_counted_rows_and_the_descriptive_ones_the_rest(tmp_path):
    write_inputs(tmp_path)
    got = values(("nPlaceboPairs", "nPlaceboRejectFive", "nPlaceboPairsDescriptive", "nPlaceboRejectFiveDescriptive",
                  "placeboPooledP", "placeboPooledSelc", "placeboSinglesPMin"), tmp_path)
    # the counted rows are the 1-year pairings: four single quarters and the pooled AQ3+AQ4-1y; the 2-year rows are descriptive
    assert got == {"nPlaceboPairs": "5", "nPlaceboRejectFive": "0", "nPlaceboPairsDescriptive": "5",
                   "nPlaceboRejectFiveDescriptive": "1", "placeboPooledP": "0.341", "placeboPooledSelc": "0.4\\%",
                   "placeboSinglesPMin": "0.231"}
    placebo = placebo_fixture()
    placebo.loc[placebo["pair"] == "AQ3+AQ4-1y", ["p", "selc_obs"]] = [0.012, -0.0123]
    placebo.loc[placebo["pair"] == "AQ1-1y", "p"] = 0.049
    placebo.loc[placebo["pair"] == "AQ3+AQ4-2y", "p"] = 0.001
    write_inputs(tmp_path, placebo=placebo)
    got = values(("nPlaceboPairs", "nPlaceboRejectFive", "nPlaceboRejectFiveDescriptive", "placeboPooledP", "placeboPooledSelc",
                  "placeboSinglesPMin"), tmp_path)
    assert got == {"nPlaceboPairs": "5", "nPlaceboRejectFive": "2", "nPlaceboRejectFiveDescriptive": "2", "placeboPooledP": "0.012",
                   "placeboPooledSelc": "\\textminus{}1.2\\%", "placeboSinglesPMin": "0.049"}
    # a 2-year row with a very small p is not counted; an uncounted row never moves a counted macro
    placebo.loc[placebo["pair"] == "AQ4-2y", "p"] = 0.0001
    write_inputs(tmp_path, placebo=placebo)
    assert values(("nPlaceboPairs", "nPlaceboRejectFive", "placeboSinglesPMin"), tmp_path) == {
        "nPlaceboPairs": "5", "nPlaceboRejectFive": "2", "placeboSinglesPMin": "0.049"}
    # a pairing that failed has no p-value and counts as nothing, as in the rule
    failed = placebo_fixture()
    failed.loc[failed["pair"].isin(["AQ2-1y", "AQ4-2y"]), "p"] = np.nan
    write_inputs(tmp_path, placebo=failed)
    assert values(("nPlaceboPairs", "nPlaceboPairsDescriptive", "placeboSinglesPMin"), tmp_path) == {
        "nPlaceboPairs": "4", "nPlaceboPairsDescriptive": "4", "placeboSinglesPMin": "0.272"}
    # the pooled row is a single row: two counted pooled rows are an error
    extra = pd.concat([placebo, placebo[placebo["pair"] == "AQ3+AQ4-1y"]], ignore_index=True)
    write_inputs(tmp_path, placebo=extra)
    with pytest.raises(LookupError, match=r"placeboPooledP.*matched 2 rows"):
        resolved("placeboPooledP", tmp_path)
    # no placebo file yet (stage one of the runner has not run): pending
    (tmp_path / "out" / "analogs" / "placebo_he.csv").unlink()
    for name in PLACEBO_MACROS + ("nPlaceboPairs", "nPlaceboRejectFive"):
        r = resolved(name, tmp_path)
        assert r["status"] == "pending" and "placebo_he.csv" in r["detail"], name


# ------------------------------------------------------------------------------------------------ shape of the fixtures
@pytest.mark.skipif(not (REAL_OUT / "analogs" / "simulation.csv").exists(), reason="the simulation stage has not run")
def test_the_fixtures_have_the_real_files_shape():
    real = pd.read_csv(REAL_OUT / "analogs" / "simulation.csv")
    fixture = sim_fixture()
    assert list(fixture.columns) == list(real.columns) and len(fixture) == len(real) == 44
    assert fixture["test"].value_counts().to_dict() == real["test"].value_counts().to_dict()
    for column in ("test", "law", "scenario", "persistence", "strata", "n_sim", "rho"):          # the same layout, row by row
        same = (fixture[column].fillna("-").astype(str) == real[column].fillna("-").astype(str))
        assert same.all(), (column, list(real.loc[~same, column]))
    placebo_real = pd.read_csv(REAL_OUT / "analogs" / "placebo_he.csv")
    assert list(placebo_fixture().columns) == list(placebo_real.columns)
    assert placebo_fixture()[["pair", "counted"]].equals(placebo_real[["pair", "counted"]])
    doc = json.loads((REAL_OUT / "analogs" / "protocol_inputs.json").read_text())

    def shape(d):
        return {k: shape(v) if isinstance(v, dict) else None for k, v in d.items()}

    fam = protocol_fixture()["family"]
    assert set(protocol_fixture()) == set(doc)
    assert {h: set(fam[h]) for h in ("H-E", "H-B1", "H-B2")} == {h: set(doc["family"][h]) for h in ("H-E", "H-B1", "H-B2")}
    assert set(protocol_fixture()["rule"]) == set(doc["rule"]) and set(protocol_fixture()["phi"]) == set(doc["phi"])
    assert shape(protocol_fixture()["rule"]) == shape(doc["rule"])


# ------------------------------------------------------------------------------------------------ the real files
real_files = pytest.mark.skipif(
    not all((REAL_OUT / "analogs" / f).exists() for f in ("simulation.csv", "protocol_inputs.json", "placebo_he.csv", "eta_selection.csv")),
    reason="the simulation stage has not run")


def real_value(name):
    from es.paper_macros import Sources, resolve
    from es.paper_registry import entries_by_name
    r = resolve(entries_by_name()[name], Sources())
    assert r["status"] == "ok", (name, r)
    return r["value"]


@real_files
def test_the_real_files_resolve_every_ruling_101_macro_and_every_alias():
    from es.paper_macros import Sources, resolve
    from es.paper_registry import entries_by_name
    by, src = entries_by_name(), Sources()
    for name in NEW_MACROS + tuple(ALIASES) + ("hbSize",) + OLD_KEPT + NODRIFT_MACROS:
        r = resolve(by[name], src)                                           # none matches several rows: LookupError otherwise
        assert r["status"] == "ok" and r["value"] and "PENDING" not in r["value"], (name, r)
    for old, new in ALIASES.items():
        assert real_value(old) == real_value(new), old


@real_files
def test_the_real_size_power_and_family_values():
    got = {n: real_value(n) for n in NEW_MACROS}
    assert [got[f"{p}{lv}"] for p in SIZE_PREFIXES for lv in LEVELS] == ["0.023", "0.070", "0.010", "0.053", "0.130", "0.193",
                                                                         "0.127", "0.193"]
    assert got["heSizePooledFiveWorstRow"] == "heteroskedastic, fit" and got["heSizeSingleFiveWorstRow"] == "homoskedastic, upper"
    assert got["hbOneSizeThirdWorstRow"] == "LEV6, common\\_drift\\_0.5, upper"
    assert got["hbOneSizeFiveWorstRow"] == "DART3, common\\_drift\\_0.5, fit"
    assert got["hbTwoSizeThirdWorstRow"] == got["hbTwoSizeFiveWorstRow"] == "DART3, common\\_drift\\_0.5, fit"
    assert {got[f"{p}NSim"] for p in SIZE_PREFIXES} == {"300"}
    assert real_value("hbSize") == "0.193"
    assert [got[f"hePowerCurveThird{rw}"] for rw in RHO_WORDS] == ["0.13", "0.36", "0.87", "1.00", "1.00", "1.00", "1.00"]
    assert [got[f"hePowerCurveFive{rw}"] for rw in RHO_WORDS] == ["0.22", "0.53", "0.96", "1.00", "1.00", "1.00", "1.00"]
    assert (got["hePowerTargetThird"], got["hePowerTargetFive"], got["hePowerTargetRho"], got["hePowerTargetNote"]) == (
        "0.09", "0.21", "0.05", "target below the grid")
    assert (got["hbOnePowerThird"], got["hbOnePowerFive"], got["hbTwoPowerThird"], got["hbTwoPowerFive"]) == (
        "0.28", "0.43", "0.05", "0.13")
    assert [got[n] for n in ("heFamilyStatus", "hbOneFamilyStatus", "hbTwoFamilyStatus")] == ["confirmatory", "estimand", "estimand"]
    assert [got[n] for n in ("heApproximate", "hbOneApproximate", "hbTwoApproximate")] == ["no", "yes", "yes"]
    assert [got[n] for n in ("heFalseDemotion", "hbOneFalseDemotion", "hbTwoFalseDemotion")] == ["0.074", "0.052", "0.052"]
    assert (got["heFalseDemotionSim"], got["heFalseDemotionPlacebo"]) == ("0.057", "0.0167")       # the note: 0.057; placebo 0.0167
    assert got["etaAtGridTop"] == "yes" and real_value("etaStar") == "0.05"
    assert [got[n] for n in PHI_MACROS] == ["0.788", "0.687", "0.773", "0.679", "0.770", "0.676"]
    assert got["ruleFactor"] == "1.5" and got["ruleNRep"] == "299" and (got["ruleKHe"], got["ruleKHbOne"]) == ("4", "12")
    assert [real_value(n) for n in ("nPlaceboPairs", "nPlaceboRejectFive", "nPlaceboPairsDescriptive", "nPlaceboRejectFiveDescriptive")] == [
        "5", "0", "5", "0"]                                              # the counted rows are the 1-year pairings, 4 singles and 1 pooled
    assert (got["placeboPooledP"], got["placeboPooledSelc"], got["placeboSinglesPMin"]) == ("0.341", "0.4\\%", "0.264")
    assert [real_value(n) for n in NODRIFT_MACROS] == ["9", "14.1", "17", "22", "25", "28"]


@real_files
def test_the_real_macros_agree_with_the_reasons_the_rule_wrote():
    """The size reasons name the worst row and its count; the macros must name the same row and the same rate."""
    doc = json.loads((REAL_OUT / "analogs" / "protocol_inputs.json").read_text())
    n_sim = int(real_value("hbOneSizeNSim"))
    level_text = {"Third": "0.0167", "Five": "0.0500"}
    checked = 0
    for prefix, hid in (("hbOneSize", "H-B1"), ("hbTwoSize", "H-B2")):
        for reason in doc["family"][hid]["reasons"]:
            for level, text in level_text.items():
                if f"> 1.5 x {text} (" not in reason or not reason.startswith("simulated size"):
                    continue
                rate = real_value(f"{prefix}{level}")
                assert reason.startswith(f"simulated size {rate} ")
                where = reason.rsplit(": ", 1)[1].rstrip(")")
                assert real_value(f"{prefix}{level}WorstRow").replace("\\_", "_") == where
                assert f"({round(float(rate) * n_sim)} of {n_sim} replications, worst of {doc['rule']['k_rows'][hid]} rows" in reason
                checked += 1
    assert checked == 4                                                  # both levels of both H-B tests are oversized
    assert real_value("hbOnePowerThird") == f"{doc['family']['H-B1']['power']:.2f}"
    assert real_value("hbTwoPowerThird") == f"{doc['family']['H-B2']['power']:.2f}"
    assert real_value("hePowerTargetRho") == real_value("rhoStar")
    assert real_value("ruleKHbOne") == str(doc["family"]["H-B1"]["k_rows"])
    assert real_value("heFalseDemotion") == f"{doc['family']['H-E']['false_demotion_bound']:.3f}"


@real_files
def test_the_real_file_passes_the_build_time_rule_check():
    from es.paper_macros import Sources, check_simulation_rule
    check_simulation_rule(Sources())


# ------------------------------------------------------------------------------------------------ the rule's own code as the oracle
def test_the_recomputed_false_demotion_bounds_equal_the_rules_own_functions():
    """paper_macros recomputes the rule's bound (it never imports the heavy run_analogs); run_analogs is the oracle here."""
    ra = pytest.importorskip("es.run_analogs")
    from es.paper_macros import _placebo_bound, _size_bound
    levels = list(ra.NOMINAL)
    for n in range(0, 9):
        for factor in (1.0, 1.5, 2.0):
            assert _placebo_bound(n, levels, factor, ra.BINOMIAL_LEVEL) == pytest.approx(ra._placebo_bound(n, factor), abs=1e-15)
    for ns in ([300] * 4, [100] * 7, [300, 100, 50], [20], [300] * 12):
        sim = pd.DataFrame({"test": "T", "strata": "primary", "n_sim": ns})
        for factor in (1.0, 1.5):
            assert _size_bound(ns, levels, factor, ra.BINOMIAL_LEVEL) == pytest.approx(ra._size_bound(sim, "T", factor), abs=1e-15)


@real_files
def test_the_real_selections_are_the_rules_own_rows():
    """The registry's queries, against es.run_analogs._test_rows, _where and _worst on the real file."""
    ra = pytest.importorskip("es.run_analogs")
    from es.paper_tex import FORMATS
    sim = pd.read_csv(REAL_OUT / "analogs" / "simulation.csv")
    cases = {"heSizePooled": ("H-E pooled A+B, size", None), "heSizeSingle": ("H-E single A, size", None),
             "hbOneSize": ("H-B size", ra._hb_laws("LEV6")), "hbTwoSize": ("H-B size", ra._hb_laws("ROLL_DISP9"))}
    for prefix, (test, laws) in cases.items():
        rows = ra._test_rows(sim, test, laws)
        for level, column in (("Third", THIRD), ("Five", FIVE)):
            worst = rows.loc[rows[column].astype(float).idxmax()]               # the first row with the maximum
            assert real_value(f"{prefix}{level}") == FORMATS["dec3"](worst[column]), (prefix, level)
            assert real_value(f"{prefix}{level}WorstRow").replace("\\_", "_") == ra._where(worst), (prefix, level)
            for attr in ("scenario", "persistence"):
                assert real_value(f"{prefix}{level}{attr.capitalize()}").replace("\\_", "_") == worst[attr]
        assert real_value(f"{prefix}NSim") == str(int(rows["n_sim"].iloc[0])) and rows["n_sim"].nunique() == 1
    for test, prefix in (("H-E pooled A+B, size", "heSizePooled"), ("H-E single A, size", "heSizeSingle")):
        worst = ra._worst(sim, test)                                               # the rule's reporting maximum
        assert real_value(f"{prefix}Third") == FORMATS["dec3"](worst[ra.NOMINAL[0]])
        assert real_value(f"{prefix}Five") == FORMATS["dec3"](worst[ra.NOMINAL[1]])
    # the constants of the rule, as the code has them
    assert real_value("ruleFactor") == FORMATS["g3"](ra.FACTOR) and real_value("ruleBinomLevel") == FORMATS["g3"](ra.BINOMIAL_LEVEL)
    assert real_value("ruleMinPower") == FORMATS["g3"](ra.MIN_POWER) and real_value("ruleApproxFactor") == FORMATS["g3"](ra.APPROXIMATE_FACTOR)
    assert (real_value("ruleLevelThird"), real_value("ruleLevelFive")) == tuple(FORMATS["g3"](a) for a in ra.NOMINAL)
    # and the two parts of H-E's bound are the rule's own
    assert real_value("heFalseDemotionSim") == FORMATS["dec3"](ra._size_bound(sim, "H-E pooled A+B, size", ra.FACTOR))
    assert real_value("heFalseDemotionPlacebo") == FORMATS["dec4"](ra._placebo_bound(1, ra.FACTOR))


# ------------------------------------------------------------------------------------------------ the no-drift context (Task 8)
def test_the_no_drift_context_is_the_range_over_the_no_drift_rows_of_the_95th_percentile(tmp_path):
    """Section 4.6 sets the audit counts next to what the size study of the portability bootstrap produces without drift: the
    95th percentile of the number of nodes outside the band, as the smallest and the largest over the study's laws and
    persistences. Only the no-drift rows of the H-B size study carry the column; a row with drift is a decoy."""
    sim = sim_fixture()
    no_drift = (sim["test"] == "H-B size") & (sim["scenario"] == "no_drift")
    assert no_drift.sum() == 6                                              # three laws, two persistences
    sim.loc[no_drift, "nodrift_fail_q95_0.5_2"] = [9.0, 12.0, 12.0, 14.05, 9.0, 13.0]
    sim.loc[no_drift, "nodrift_fail_q95_0.67_1.5"] = [17.0, 20.0, 20.0, 22.0, 17.0, 21.0]
    sim.loc[no_drift, "nodrift_fail_q95_0.8_1.25"] = [25.0, 27.0, 27.0, 28.0, 25.0, 27.0]
    sim.loc[(sim["test"] == "H-B size") & (sim["scenario"] == "common_drift_0.5"), "nodrift_fail_q95_0.5_2"] = 99.0   # a decoy
    write_inputs(tmp_path, sim=sim)
    assert values(NO_DRIFT_MACROS, tmp_path) == {
        "noDriftFailNinetyFiveWideMin": "9", "noDriftFailNinetyFiveWideMax": "14.1",
        "noDriftFailNinetyFiveMidMin": "17", "noDriftFailNinetyFiveMidMax": "22",
        "noDriftFailNinetyFiveTightMin": "25", "noDriftFailNinetyFiveTightMax": "28"}


def test_the_no_drift_context_is_pending_before_the_simulation_file_exists(tmp_path):
    from es.paper_macros import PENDING
    (tmp_path / "out").mkdir()
    for name in NO_DRIFT_MACROS:
        r = resolved(name, tmp_path)
        assert r["status"] == "pending" and r["value"] == PENDING["dev"], (name, r)


@real_files
def test_the_real_no_drift_context():
    sim = pd.read_csv(REAL_OUT / "analogs" / "simulation.csv")
    rows = sim[(sim["test"] == "H-B size") & (sim["scenario"] == "no_drift")]
    from es.paper_tex import FORMATS
    for band, column in (("Wide", "nodrift_fail_q95_0.5_2"), ("Mid", "nodrift_fail_q95_0.67_1.5"), ("Tight", "nodrift_fail_q95_0.8_1.25")):
        assert real_value(f"noDriftFailNinetyFive{band}Min") == FORMATS["g3"](rows[column].min())
        assert real_value(f"noDriftFailNinetyFive{band}Max") == FORMATS["g3"](rows[column].max())
    assert (real_value("noDriftFailNinetyFiveWideMin"), real_value("noDriftFailNinetyFiveWideMax")) == ("9", "14.1")
