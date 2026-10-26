"""Tests of es/run_analogs.py (Task 11): quarterly analogues, placebo pairings, eta, size and power.

Layers, in order:
- the brief's Step 1 tests;
- the registration rule (Rulings 29, 63, 68): a pure function of simulation rows and placebo rows;
- placebo_cube and the runner's pieces, on a small synthetic score frame injected through ``cached_scores``;
- the staged runner (Ruling 77), end to end on stubbed inputs with a handful of replications.
No test reads real data or the score cache.
"""


def test_simfit_windows_are_development_only():
    from es.run_analogs import SIMFIT
    assert SIMFIT.deploy.end == "2026-06-30" and SIMFIT.train.end == "2025-06-30"
    assert SIMFIT.calib.start == "2025-07-01"


def test_shift_year_keeps_month_day():
    import datetime as dt
    from es.run_analogs import shift_years
    assert shift_years(dt.date(2025, 7, 15), 1) == dt.date(2024, 7, 15)
    assert shift_years(dt.date(2024, 2, 29), 1) == dt.date(2023, 2, 28)


# ---------------------------------------------------------------------------------------------------------------
# The registration rule (spec v3 §6.2; Rulings 29, 63, 68). A pure function of simulation rows and placebo rows.
# ---------------------------------------------------------------------------------------------------------------

ALPHA3 = 0.05 / 3


def _sim(he_pooled=0.05, he_single=0.05, hb=0.05, p1=0.8, p2=0.8, n_sim=300):
    """The rows registration_family reads, in the order of the brief (index 3 is H-B1's power). H-E rows carry their
    strata ("primary"), H-B rows none; every size row carries its n_sim (Ruling 63)."""
    rows = []
    for test, r05, strata in (("H-E pooled A+B, size", he_pooled, "primary"),
                              ("H-E single A, size", he_single, "primary"), ("H-B size", hb, None)):
        row = {"test": test, "n_sim": n_sim, "reject_0.0166667": r05 / 3, "reject_0.05": r05}
        if strata:
            row["strata"] = strata
        rows.append(row)
    rows += [{"test": "H-B1 power", "n_sim": 150, "reject_0.0166667": p1, "reject_0.05": p1},
             {"test": "H-B2 power", "n_sim": 150, "reject_0.0166667": p2, "reject_0.05": p2}]
    return rows


QUARTERS = ("AQ3", "AQ4", "AQ1", "AQ2")
UNCOUNTED = "P&L year inside the alert model's training window"


def _placebo(p_single=(0.5,) * 4, p_pooled=0.5, p_single_2y=(0.5,) * 4, p_pooled_2y=0.5):
    """The ten pairings as placebo_he.csv holds them (Ruling 94, C1): the four 1-year single quarters and the pooled
    AQ3+AQ4-1y count; the five 2-year pairings are descriptive (counted False, with the note). Index 0 to 3 are the
    1-year singles, 4 to 7 the 2-year singles, 8 the pooled 1-year pairing and 9 the pooled 2-year one."""
    rows = [{"pair": f"{q}-1y", "p": p, "counted": True} for q, p in zip(QUARTERS, p_single)]
    rows += [{"pair": f"{q}-2y", "p": p, "counted": False, "note": UNCOUNTED} for q, p in zip(QUARTERS, p_single_2y)]
    rows += [{"pair": "AQ3+AQ4-1y", "p": p_pooled, "counted": True},
             {"pair": "AQ3+AQ4-2y", "p": p_pooled_2y, "counted": False, "note": UNCOUNTED}]
    return rows


def test_registration_family_keeps_well_sized_powered_tests():
    from es.run_analogs import registration_family
    fam = registration_family(_sim(), _placebo())
    hids = ("H-E", "H-B1", "H-B2")
    assert all(fam[h]["confirmatory"] for h in hids)
    assert all(fam[h]["reasons"] == [] for h in hids)
    assert not any(fam[h]["approximate"] for h in hids) and not fam["H-E"]["interim_approximate"]   # Ruling 92b
    assert not any(fam[h]["power_lower_bound"] for h in ("H-B1", "H-B2"))


def test_registration_family_demotes_oversized_and_underpowered():
    from es.run_analogs import registration_family
    fam = registration_family(_sim(he_pooled=0.09, p2=0.3), _placebo())
    assert not fam["H-E"]["confirmatory"] and fam["H-B1"]["confirmatory"]
    assert not fam["H-B2"]["confirmatory"] and "power" in fam["H-B2"]["reasons"][0]
    assert "simulated size" in fam["H-E"]["reasons"][0]


def test_only_the_one_year_pairings_count_for_h_e():
    """Ruling 94, C1: a 2-year P&L quarter lies inside the alert model's training window, so those pairings are
    reported and never counted. A 2-year rejection, however strong, changes nothing; the pooled 1-year pairing (n = 1)
    demotes H-E if and only if its p is at most alpha/3 (a single rejection has tail alpha/3 = 0.0167 < 0.05 there, and
    0.05 at 0.05, which is not below 0.05)."""
    from es.run_analogs import registration_family
    base = registration_family(_sim(), _placebo())
    assert base["H-E"]["confirmatory"] and not base["H-E"]["interim_approximate"] and not base["H-E"]["approximate"]
    assert registration_family(_sim(), _placebo(p_single_2y=(0.0001,) * 4, p_pooled_2y=0.0001)) == base
    for p in (0.0, 0.01, 0.0166):                                              # at or below alpha/3 = 0.016667
        fam = registration_family(_sim(), _placebo(p_pooled=p))
        assert not fam["H-E"]["confirmatory"] and "placebo" in fam["H-E"]["reasons"][0], p
    for p in (0.0168, 0.04, 0.05, 0.9):                                        # above alpha/3: nothing can demote
        assert registration_family(_sim(), _placebo(p_pooled=p))["H-E"]["confirmatory"], p
    assert registration_family(_sim(), _placebo(p_single=(0.0001,) * 4))["H-E"]["confirmatory"]   # singles never demote


def test_h_e_is_approximate_when_the_placebo_check_demotes_it():
    """Ruling 94 (M8): spec §5.1 labels a CI approximate on any size check, so a placebo demotion labels it too."""
    from es.run_analogs import registration_family
    assert not registration_family(_sim(), _placebo())["H-E"]["approximate"]
    assert registration_family(_sim(), _placebo(p_pooled=0.01))["H-E"]["approximate"]
    assert not registration_family(_sim(), _placebo(p_pooled_2y=0.0001))["H-E"]["approximate"]   # not a counted one


def test_power_floor_is_read_at_alpha_over_3():
    from es.run_analogs import registration_family
    sim = _sim()
    sim[3]["reject_0.0166667"], sim[3]["reject_0.05"] = 0.45, 0.70
    assert not registration_family(sim, _placebo())["H-B1"]["confirmatory"]


def test_binomial_upper_tail_matches_the_hand_checks():
    from es.run_analogs import _binom_upper
    assert _binom_upper(1, 8, 0.075) == __import__("pytest").approx(0.4640, abs=5e-5)    # the brief's "0.46"
    assert _binom_upper(2, 2, 0.025) == __import__("pytest").approx(0.000625)
    assert _binom_upper(0, 10, 0.3) == __import__("pytest").approx(1.0)
    assert _binom_upper(11, 10, 0.3) == 0.0


def test_a_correctly_sized_test_is_not_demoted_at_300_replications():
    """Ruling 63: the count at its expected value, one SD above it, and the noisy 1.8x at alpha/3 (9 of 300, whose
    binomial tail is 0.066) all stay confirmatory."""
    from es.run_analogs import registration_family
    for r05, r3 in ((0.05, ALPHA3), (19 / 300, 7 / 300), (0.07, 9 / 300)):
        sim = _sim()
        for row in sim[:3]:
            row["reject_0.05"], row["reject_0.0166667"] = r05, r3
        fam = registration_family(sim, _placebo())
        assert all(fam[h]["confirmatory"] for h in ("H-E", "H-B1", "H-B2")), (r05, r3, fam)
        assert not any(fam[h]["approximate"] for h in ("H-E", "H-B1", "H-B2")), (r05, r3, fam)   # nor "approximate"


def test_a_test_at_twice_nominal_is_demoted_at_300_replications():
    from es.run_analogs import registration_family
    fam = registration_family(_sim(he_pooled=0.10, hb=0.10), _placebo())
    assert not fam["H-E"]["confirmatory"] and not fam["H-B1"]["confirmatory"] and not fam["H-B2"]["confirmatory"]
    assert len(fam["H-E"]["reasons"]) == 2                              # one per nominal level
    sim = _sim()
    sim[0]["reject_0.05"], sim[0]["reject_0.0166667"] = 0.05, 2 * ALPHA3   # 2x at the first Holm level only
    assert not registration_family(sim, _placebo())["H-E"]["confirmatory"]


def test_the_binomial_condition_needs_the_replications_behind_the_rate():
    """0.09 is 1.8x nominal, but with 20 replications it is 2 rejections (tail 0.26): not demoted. With 300 it is."""
    from es.run_analogs import registration_family
    assert registration_family(_sim(he_pooled=0.09, n_sim=20), _placebo())["H-E"]["confirmatory"]
    assert not registration_family(_sim(he_pooled=0.09, n_sim=300), _placebo())["H-E"]["confirmatory"]


def test_h_e_status_reads_the_primary_strata_only():
    """Appendix B: a decile-strata row at 3x nominal is reported and never demotes H-E."""
    from es.run_analogs import registration_family
    sim = _sim()
    sim.append({"test": "H-E pooled A+B, size", "strata": "decile", "n_sim": 300,
                "reject_0.0166667": 0.05, "reject_0.05": 0.15})
    sim.append({"test": "H-E single A, size", "strata": "decile", "n_sim": 300,
                "reject_0.0166667": 0.05, "reject_0.05": 0.15})
    fam = registration_family(sim, _placebo())
    assert fam["H-E"]["confirmatory"] and not fam["H-E"]["approximate"] and not fam["H-E"]["interim_approximate"]
    sim[-2]["strata"] = "primary"                                       # the same row, now a registered one
    assert not registration_family(sim, _placebo())["H-E"]["confirmatory"]


def test_h_b_size_is_the_worst_over_laws_persistence_and_drift():
    from es.run_analogs import registration_family
    sim = _sim()
    sim += [{"test": "H-B size", "law": law, "persistence": p, "scenario": sc, "n_sim": 300, "reject_0.0166667": r / 3,
             "reject_0.05": r}
            for law, p, sc, r in (("DART3", "fit", "no_drift", 0.05), ("DART3", "upper", "no_drift", 0.06),
                                  ("DART3", "fit", "common_drift_0.25", 0.05),
                                  ("DART3", "upper", "common_drift_0.5", 0.14))]
    fam = registration_family(sim, _placebo())
    assert fam["H-E"]["confirmatory"]
    assert not fam["H-B1"]["confirmatory"] and not fam["H-B2"]["confirmatory"]     # a DART3 row is a null of both
    said = " ".join(fam["H-B1"]["reasons"])                      # one reason per nominal level, the worst row each
    assert "0.140" in said and "common_drift_0.5" in said and "upper" in said and "DART3" in said


def _hb_size_rows_at(rates, n_sim=300):
    """The rows registration_family reads, with one H-B size row per rate in ``rates`` (at alpha / 3 a third of it)."""
    sim = [r for r in _sim() if r["test"] != "H-B size"]
    return sim + [{"test": "H-B size", "law": "DART3", "persistence": "fit", "scenario": f"row{i}", "n_sim": n_sim,
                   "reject_0.0166667": r / 3, "reject_0.05": r} for i, r in enumerate(rates)]


def test_the_worst_of_k_rows_must_clear_a_binomial_check_at_005_over_k():
    """Ruling 84b item 4 (M-6): the maximum over k rows is tested against 0.05 / k. A row at 23 of 300 (1.53x nominal,
    tail 0.029) is demoted alone and not as the worst of twelve; one at 3x nominal is demoted among twelve."""
    from es.run_analogs import registration_family
    nominal, worst = 15 / 300, 23 / 300
    assert not registration_family(_hb_size_rows_at([worst]), _placebo())["H-B1"]["confirmatory"]
    twelve = registration_family(_hb_size_rows_at([nominal] * 11 + [worst]), _placebo())
    assert twelve["H-B1"]["confirmatory"] and twelve["H-B2"]["confirmatory"]
    thrice = registration_family(_hb_size_rows_at([nominal] * 11 + [45 / 300]), _placebo())
    assert not thrice["H-B1"]["confirmatory"] and "worst of 12 rows" in " ".join(thrice["H-B1"]["reasons"])
    assert registration_family(_hb_size_rows_at([nominal] * 12), _placebo())["H-B1"]["confirmatory"]


def test_twelve_rows_at_the_true_size_are_rarely_demoted_with_the_correction():
    """The point of the correction, by simulation: the worst of twelve rows exactly at nominal, 300 replications each.
    Uncorrected, about half of such studies would demote a correctly sized test; corrected, about 5% or fewer."""
    from es.run_analogs import NOMINAL, _oversized
    rng = np.random.default_rng(84)
    n, rows, draws = 300, 12, 600
    corrected = uncorrected = 0
    for _ in range(draws):
        hit = hit_old = False
        for a in NOMINAL:
            k = int(rng.binomial(n, a, size=rows).max())
            hit |= _oversized(k / n, n, a, 1.5, rows)
            hit_old |= _oversized(k / n, n, a, 1.5)
        corrected += hit
        uncorrected += hit_old
    assert corrected / draws < 0.10 and uncorrected / draws > 0.20


def test_the_correction_counts_only_the_rows_that_enter_the_maximum():
    """H-E's status reads the 4 primary rows of a test, so k = 4, not 8: a primary row at 25 of 300 (tail 0.0093 <
    0.0125) is demoted even with four decile rows beside it."""
    from es.run_analogs import registration_family
    sim = [r for r in _sim() if r["test"] != "H-E pooled A+B, size"]
    for strata in ("primary", "decile"):
        for i in range(4):
            r = 25 / 300 if (strata == "primary" and i == 0) else 15 / 300
            sim.append({"test": "H-E pooled A+B, size", "strata": strata, "persistence": "fit", "scenario": f"row{i}",
                        "n_sim": 300, "reject_0.0166667": r / 3, "reject_0.05": r})
    fam = registration_family(sim, _placebo())
    assert not fam["H-E"]["confirmatory"] and "worst of 4 rows" in " ".join(fam["H-E"]["reasons"])


def test_approximate_means_significantly_above_nominal():
    """Ruling 92b: the combined condition with factor 1.0: rate > a AND the one-sided binomial tail below 0.05 / rows.
    With one row per test and n = 300: exactly nominal is not approximate, noise is not, a row at 1.6x (24 of 300, tail
    0.017) is."""
    from es.run_analogs import registration_family

    def labels(**kw):
        fam = registration_family(_sim(**kw), _placebo())
        return fam["H-E"]["approximate"], fam["H-B1"]["approximate"], fam["H-E"]["interim_approximate"]

    assert labels() == (False, False, False)                                    # exactly at nominal
    assert labels(he_pooled=0.06, he_single=0.06, hb=0.06) == (False, False, False)   # 18 of 300: tail 0.25, noise
    assert labels(he_pooled=0.08, he_single=0.08, hb=0.08) == (True, True, True)       # 1.6x nominal
    assert labels(he_pooled=0.05, he_single=0.05, hb=0.08) == (False, True, False)      # each label reads its own rows
    assert labels(he_pooled=0.05, he_single=0.08, hb=0.05) == (False, False, True)
    assert labels(he_single=22 / 300) == (False, False, True)              # the interim label has the label's factor
    # the label asks for less than the demotion does: 22 of 300 is 1.47x (below 1.5x) with tail 0.0486
    fam = registration_family(_sim(he_pooled=22 / 300), _placebo())
    assert fam["H-E"]["approximate"] and fam["H-E"]["confirmatory"]
    assert registration_family(_sim(hb=22 / 300), _placebo())["H-B1"]["approximate"]


def _rows_at(test, rates, strata=None, n_sim=300):
    """One simulation row of ``test`` per rate (at alpha / 3 a third of it), the rest of the rows of ``_sim()`` kept."""
    return [{"test": test, **({"strata": strata} if strata else {}), "persistence": "fit", "scenario": f"row{i}",
             "n_sim": n_sim, "reject_0.0166667": r / 3, "reject_0.05": r} for i, r in enumerate(rates)]


def test_approximate_is_bonferroni_corrected_over_the_rows_of_the_test():
    """The same 0.05 / k as the demotion: 4 primary H-E rows need a tail below 0.0125 (24 of 300 has 0.0168, 25 has
    0.0093), the 18 H-B rows 0.0028 (26 of 300 has 0.0050, 27 has 0.0026); a single row needs 0.05."""
    from es.run_analogs import registration_family
    nominal = 15 / 300

    def label(test, worst, rows, hid):
        sim = [r for r in _sim() if r["test"] != test] + _rows_at(test, [worst] + [nominal] * (rows - 1),
                                                                    "primary" if test.startswith("H-E") else None)
        fam = registration_family(sim, _placebo())
        return fam[hid]["approximate"]

    assert not label("H-E pooled A+B, size", 24 / 300, 4, "H-E") and label("H-E pooled A+B, size", 25 / 300, 4, "H-E")
    assert label("H-E pooled A+B, size", 24 / 300, 1, "H-E")
    assert not label("H-B size", 26 / 300, 18, "H-B1") and label("H-B size", 27 / 300, 18, "H-B1")
    assert label("H-B size", 27 / 300, 18, "H-B2")


def test_two_of_the_four_one_year_singles_set_the_interim_label():
    """Ruling 94, C1: the singles have n = 4. The combined rule's arithmetic gives 2 of 4 at either level: at 0.05 the
    tail of 2 is 0.014 (1 has 0.186), at alpha/3 it is 0.002 (1 has 0.065). They only label the interim p-value."""
    from es.run_analogs import registration_family
    one = registration_family(_sim(), _placebo(p_single=(0.04, 0.5, 0.5, 0.5)))
    two = registration_family(_sim(), _placebo(p_single=(0.04, 0.04, 0.5, 0.5)))
    assert not one["H-E"]["interim_approximate"] and two["H-E"]["interim_approximate"]
    assert one["H-E"]["confirmatory"] and two["H-E"]["confirmatory"]            # singles never demote
    one3 = registration_family(_sim(), _placebo(p_single=(0.01, 0.5, 0.5, 0.5)))
    two3 = registration_family(_sim(), _placebo(p_single=(0.01, 0.01, 0.5, 0.5)))
    assert not one3["H-E"]["interim_approximate"] and two3["H-E"]["interim_approximate"]
    mixed = registration_family(_sim(), _placebo(p_single=(0.01, 0.04, 0.5, 0.5)))   # 2 of 4 reject at 0.05
    assert mixed["H-E"]["interim_approximate"]
    assert not registration_family(_sim(), _placebo(p_single_2y=(0.0001,) * 4))["H-E"]["interim_approximate"]


def test_placebo_rows_without_a_p_are_ignored_by_the_rule():
    """A pairing that failed (error row), or whose p came back from the CSV as NaN, is neither a rejection nor not.
    (run_sim_stage refuses to run unless the counted ones all have a p: see the stage tests.)"""
    from es.run_analogs import registration_family
    rows = _placebo()
    rows[0] = {"pair": "AQ3-1y", "error": "window too short", "counted": True}
    rows[1] = {"pair": "AQ4-1y", "p": float("nan"), "counted": True}
    rows[8] = {"pair": "AQ3+AQ4-1y", "p": None, "counted": True}
    fam = registration_family(_sim(), rows)
    assert fam["H-E"]["confirmatory"] and not fam["H-E"]["interim_approximate"]
    assert "did not run" in " ".join(fam["notes"])
    assert registration_family(_sim(), [{"pair": "AQ3-1y", "error": "x"}])["H-E"]["confirmatory"]


def test_a_row_without_the_counted_flag_counts_if_its_pairing_is_one_year():
    """Placebo rows written before the flag existed: the pairing's name decides."""
    from es.run_analogs import registration_family
    rows = [{k: v for k, v in r.items() if k not in ("counted", "note")} for r in _placebo(p_pooled=0.01)]
    assert not registration_family(_sim(), rows)["H-E"]["confirmatory"]
    rows = [{k: v for k, v in r.items() if k not in ("counted", "note")} for r in _placebo(p_pooled_2y=0.0001)]
    assert registration_family(_sim(), rows)["H-E"]["confirmatory"]
    # when the flag is there it decides, whatever the name says
    rows = _placebo(p_pooled=0.01)
    rows[8]["counted"] = False
    assert registration_family(_sim(), rows)["H-E"]["confirmatory"]
    rows = _placebo(p_pooled_2y=0.01)
    rows[9]["counted"] = True
    assert not registration_family(_sim(), rows)["H-E"]["confirmatory"]


def test_missing_simulation_rows_or_replication_counts_raise():
    import pytest
    from es.run_analogs import registration_family
    with pytest.raises(ValueError, match="no 'H-B size' row"):
        registration_family([r for r in _sim() if r["test"] != "H-B size"], _placebo())
    with pytest.raises(ValueError, match="no 'H-B2 power' row"):
        registration_family([r for r in _sim() if r["test"] != "H-B2 power"], _placebo())
    rows = _sim()
    del rows[0]["n_sim"]
    with pytest.raises(ValueError, match="n_sim"):
        registration_family(rows, _placebo())
    only_decile = [dict(r, strata="decile") if r["test"].startswith("H-E") else r for r in _sim()]
    with pytest.raises(ValueError, match="no 'H-E pooled A\\+B, size' row"):
        registration_family(only_decile, _placebo())


def test_a_power_calibration_that_failed_demotes_the_test():
    """Ruling 63: calibrate_delta raising leaves a row with an error and no rejection rates; power counts as < 0.5."""
    from es.run_analogs import registration_family
    sim = _sim()
    sim[3] = {"test": "H-B1 power", "scenario": "no_drift", "target_log_error_gap": -0.1,
              "error": "calibrate_delta: target_gap -0.1 is outside the gap range"}
    fam = registration_family(sim, _placebo())
    assert not fam["H-B1"]["confirmatory"] and fam["H-B2"]["confirmatory"]
    assert "power" in fam["H-B1"]["reasons"][0] and "outside the gap range" in fam["H-B1"]["reasons"][0]
    assert fam["H-B1"]["power"] != fam["H-B1"]["power"]                  # NaN


def test_power_is_read_at_the_lowest_row_and_h_e_ignores_its_power():
    from es.run_analogs import registration_family
    sim = _sim() + [{"test": "H-B1 power", "scenario": "common_drift", "n_sim": 150,
                     "reject_0.0166667": 0.3, "reject_0.05": 0.6},
                    {"test": "H-E pooled A+B, power", "n_sim": 300, "reject_0.0166667": 0.01, "reject_0.05": 0.02}]
    fam = registration_family(sim, _placebo())
    assert not fam["H-B1"]["confirmatory"] and fam["H-B1"]["power"] == 0.3
    assert fam["H-E"]["confirmatory"]                                    # H-E stays in the family whatever its power


def test_nothing_in_the_registration_reads_h_e_power():
    """Ruling 92b item 1: H-E stays in whatever its power, so neither its power row nor its power curve is read: the
    family is the same with none, with a curve of 0, 1 or NaN, and with a power row that failed."""
    from es.run_analogs import RHO_GRID, registration_family
    base = registration_family(_sim(), _placebo())

    def curve(p):
        return [{"test": "H-E pooled A+B, power curve", "scenario": "homoskedastic", "strata": "primary", "rho": r,
                 "n_sim": 100, "reject_0.0166667": p, "reject_0.05": p, "mean_selc": 0.1 * r} for r in RHO_GRID]

    for power in (0.0, 1.0, float("nan")):
        row = {"test": "H-E pooled A+B, power", "n_sim": 300, "reject_0.0166667": power, "reject_0.05": power}
        assert registration_family(_sim() + curve(power) + [row], _placebo()) == base
    failed = {"test": "H-E pooled A+B, power", "error": "no coupling reaches the target"}
    assert registration_family(_sim() + curve(0.0) + [failed], _placebo()) == base


def test_notes_state_the_pooled_one_year_false_demotion_bound():
    """Ruling 94 (M7): the pooled 1-year check has n = 1, so H-E is demoted when its p is at most alpha/3, with
    false-demotion probability at most 0.0167, and cannot be demoted at 0.05. The check is kept as registered, and the
    2-year pairings are named as descriptive."""
    from es.run_analogs import registration_family
    fam = registration_family(_sim(), _placebo())
    assert isinstance(fam["notes"], list) and set(fam) == {"H-E", "H-B1", "H-B2", "notes"}
    text = " ".join(fam["notes"])
    assert "≤ 0.0167" in text and "cannot" in text and "retained as registered" in text
    assert "2-year" in text and "training window" in text
    gone = registration_family(_sim(), [r for r in _placebo() if r["pair"] != "AQ3+AQ4-1y"])
    assert "did not run" in " ".join(gone["notes"])


def _hb_matrix(overrides=None, n_sim=300, nominal=15 / 300):
    """The 18 H-B size rows (3 laws x 2 persistence settings x 3 drifts), every one at ``nominal`` except the
    ``overrides`` {(law, persistence, scenario): rate}, beside the other rows of ``_sim()``."""
    overrides = overrides or {}
    sim = [r for r in _sim() if r["test"] != "H-B size"]
    for law in HB_LAWS:
        for p in ("fit", "upper"):
            for sc in HB_SCENARIOS:
                r = overrides.get((law, p, sc), nominal)
                sim.append({"test": "H-B size", "law": law, "persistence": p, "scenario": sc, "n_sim": n_sim,
                            "reject_0.0166667": r / 3, "reject_0.05": r})
    return sim


def test_each_h_b_test_reads_its_own_two_laws():
    """Ruling 94 (M3): H-B1 reads the DART3 and LEV6 rows, H-B2 the DART3 and ROLL_DISP9 rows: k = 12 for each. A LEV6
    row is not a null of H-B2 (and the reverse); a DART3 row is a null of both."""
    from es.run_analogs import registration_family
    big = 45 / 300                                                              # 3x nominal
    lev = registration_family(_hb_matrix({("LEV6", "upper", "common_drift_0.5"): big}), _placebo())
    assert not lev["H-B1"]["confirmatory"] and lev["H-B2"]["confirmatory"]
    assert lev["H-B1"]["k_rows"] == lev["H-B2"]["k_rows"] == 12
    said = " ".join(lev["H-B1"]["reasons"])
    assert "LEV6" in said and "worst of 12 rows" in said
    assert lev["H-B1"]["approximate"] and not lev["H-B2"]["approximate"]
    roll = registration_family(_hb_matrix({("ROLL_DISP9", "fit", "no_drift"): big}), _placebo())
    assert roll["H-B1"]["confirmatory"] and not roll["H-B2"]["confirmatory"]
    assert roll["H-B2"]["approximate"] and not roll["H-B1"]["approximate"]
    dart = registration_family(_hb_matrix({("DART3", "fit", "common_drift_0.25"): big}), _placebo())
    assert not dart["H-B1"]["confirmatory"] and not dart["H-B2"]["confirmatory"]
    assert dart["H-E"]["confirmatory"]
    none = registration_family(_hb_matrix(), _placebo())
    assert none["H-B1"]["confirmatory"] and none["H-B2"]["confirmatory"]


def test_a_nan_rejection_rate_raises():
    """Ruling 94 (M5): a NaN size rate must not pass as 'not oversized'; nor may a NaN power without an error text."""
    from es.run_analogs import registration_family
    for column in ("reject_0.05", "reject_0.0166667"):
        sim = _sim()
        sim[2][column] = float("nan")                                           # the H-B size row
        with pytest.raises(ValueError, match="NaN"):
            registration_family(sim, _placebo())
        sim = _sim()
        sim[0][column] = float("nan")                                           # the H-E pooled size row
        with pytest.raises(ValueError, match="NaN"):
            registration_family(sim, _placebo())
    sim = _sim()
    sim[3]["reject_0.0166667"] = float("nan")                                   # a power row with no error text
    with pytest.raises(ValueError, match="NaN"):
        registration_family(sim, _placebo())


def test_a_power_lower_bound_never_demotes():
    """Ruling 94 (M4): a target above the grid's top gap leaves the power at delta = 0.8, a lower bound on the power at
    the target. It is reported as such and never demotes, whatever its value; a normal row below the floor does."""
    from es.run_analogs import registration_family
    sim = _sim()
    sim[3] = {"test": "H-B1 power", "scenario": "no_drift", "n_sim": 300, "reject_0.0166667": 0.30,
              "reject_0.05": 0.5, "power_lower_bound": True}
    fam = registration_family(sim, _placebo())
    assert fam["H-B1"]["confirmatory"] and fam["H-B1"]["power_lower_bound"] and fam["H-B1"]["power"] == 0.30
    assert not fam["H-B2"]["power_lower_bound"]
    sim.append({"test": "H-B1 power", "scenario": "common_drift", "n_sim": 300, "reject_0.0166667": 0.2,
                "reject_0.05": 0.4})
    assert not registration_family(sim, _placebo())["H-B1"]["confirmatory"]     # a normal row below 0.5 demotes


def test_reading_the_lower_bound_flag_raises_no_warning():
    """NB2: the flag is a column that is True on a few rows and missing on the rest (object dtype), as the rows come out
    of the runner. Reading it must not go through a downcasting ``fillna``, which pandas warns about."""
    import warnings
    from es.run_analogs import registration_family
    sim = _sim()
    sim[3] = {**sim[3], "power_lower_bound": True}                               # only H-B1's power row has the flag
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fam = registration_family(sim, _placebo())
    assert fam["H-B1"]["power_lower_bound"] and not fam["H-B2"]["power_lower_bound"]
    sim[3]["power_lower_bound"] = False                                          # an explicit False is not a bound
    sim[4] = {**sim[4], "power_lower_bound": float("nan")}                       # nor is a NaN (a CSV's empty cell)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fam = registration_family(sim, _placebo())
    assert not fam["H-B1"]["power_lower_bound"] and not fam["H-B2"]["power_lower_bound"]


def test_false_demotion_bounds_are_recorded_for_every_test():
    """Ruling 94 (M7): each test's overall false-demotion bound under independent rows, as the union over the two levels
    of 1 - (1 - tail of the smallest oversized count)^k, plus the placebo's alpha/3 for H-E. Thresholds at 300
    replications: 12 and 25 for 4 rows, 13 and 27 for 12 rows."""
    from scipy.stats import binom
    from es.run_analogs import NOMINAL, registration_family

    def union(counts, rows):
        return sum(1 - (1 - binom.sf(k - 1, 300, a)) ** rows for k, a in zip(counts, NOMINAL))

    sim = [r for r in _hb_matrix() if r["test"] != "H-E pooled A+B, size"]
    sim += _rows_at("H-E pooled A+B, size", [15 / 300] * 4, "primary")
    fam = registration_family(sim, _placebo())
    assert fam["H-E"]["k_rows"] == 4
    assert fam["H-E"]["false_demotion_bound"] == pytest.approx(union((12, 25), 4) + NOMINAL[0], rel=1e-9)
    for hid in ("H-B1", "H-B2"):
        assert fam[hid]["k_rows"] == 12
        assert fam[hid]["false_demotion_bound"] == pytest.approx(union((13, 27), 12), rel=1e-9)
    text = " ".join(fam["notes"])
    assert f"{fam['H-E']['false_demotion_bound']:.3f}" in text and f"{fam['H-B1']['false_demotion_bound']:.3f}" in text


def test_worst_is_the_highest_rate_over_the_rows_the_rule_reads():
    from es.run_analogs import NOMINAL, _worst
    sim = pd.DataFrame(_sim(he_pooled=0.06) + _rows_at("H-E pooled A+B, size", [0.2], "decile"))
    got = _worst(sim, "H-E pooled A+B, size")                                  # the decile row (0.2) never counts
    assert got[NOMINAL[0]] == pytest.approx(0.02) and got[NOMINAL[1]] == pytest.approx(0.06)


# ---------------------------------------------------------------------------------------------------------------
# A synthetic score frame in the shape of es.scores.cached_scores, injected in place of the cache.
# ---------------------------------------------------------------------------------------------------------------

import datetime as dt           # noqa: E402
import functools                # noqa: E402
import zlib                     # noqa: E402

import numpy as np              # noqa: E402
import pandas as pd             # noqa: E402
import pytest                   # noqa: E402

NODES = (("MISO", "M1"), ("NYISO", "N1"))


@functools.lru_cache(maxsize=1)
def _base():
    """Hourly rows of two nodes, 2023-01-01 to 2026-06-30 local settlement days (the development window), with the
    settlement clock of each market (so NYISO has its 23- and 25-hour days), prices, and the columns of es.scores."""
    from es.guard import settlement_frame
    rng = np.random.default_rng(2026)
    ts = pd.date_range("2023-01-01", "2026-07-01", freq="h", tz="UTC", inclusive="left")
    parts = []
    for market, node in NODES:
        f = pd.DataFrame({"market": market, "node": node, "node_type": "hub", "timestamp_utc": ts})
        f = pd.concat([f, settlement_frame(f["timestamp_utc"], f["market"])], axis=1)
        f = f[(f["settle_date"] >= dt.date(2023, 1, 1)) & (f["settle_date"] <= dt.date(2026, 6, 30))].copy()
        n = len(f)
        da = 30 + 4 * np.sin(2 * np.pi * f["settle_hour"].to_numpy() / 24) + rng.normal(0, 3, n)
        f["da_lmp"], f["rt_lmp"] = da, da - rng.normal(0.8, 6.0, n)       # DART = da - rt, a premium of 0.8 on average
        f["dart_spread"] = f["da_lmp"] - f["rt_lmp"]
        f["wclass"] = (f["weekday"] >= 5).astype(int)
        f["label"] = 0.0
        f["abs_mean_168h"] = rng.uniform(0.5, 5.0, n)
        parts.append(f)
    out = pd.concat(parts, ignore_index=True)
    out["ordinal"] = np.array([d.toordinal() for d in out["settle_date"]])
    out["hour_effect"] = 0.5 * np.sin(2 * np.pi * out["settle_hour"].to_numpy() / 24)
    return out


def stub_scores(panel, analysis, block, estimator="xgb", seed=42, label_alpha=0.05, version=None):
    """What ``cached_scores`` returns, from the synthetic frame: scores independent of the P&L (no timing skill),
    different for every (block, estimator, seed), and ``split`` set by the analysis's own windows."""
    from es.scores import OUTPUT_COLUMNS
    base = _base()
    rng = np.random.default_rng(zlib.crc32(f"{block}|{estimator}|{seed}".encode()))
    frame = base.copy()
    frame["score"] = frame["hour_effect"] + rng.normal(0.0, 1.0, len(frame))
    ordinal = frame["ordinal"].to_numpy()

    def inside(window):
        return (ordinal >= dt.date.fromisoformat(window.start).toordinal()) & \
               (ordinal <= dt.date.fromisoformat(window.end).toordinal())

    frame["split"] = np.select([inside(analysis.train), inside(analysis.calib), inside(analysis.deploy)],
                               ["train", "calib", "deploy"], default="other")
    return frame[OUTPUT_COLUMNS]


def _analysis(aid, train, calib, deploy):
    from es.analyses import Analysis
    from es.windows import DateWindow as D
    return Analysis(aid, D(*train), D(*calib), D(*deploy), D(*train), "rolling", "analog")


#: A 20-day deployment window in July 2025, so its placebo windows are July 2024 and July 2023 (no DST, no leap day).
TINY = _analysis("TINY", ("2024-01-01", "2024-06-30"), ("2024-07-01", "2024-09-30"), ("2025-07-01", "2025-07-20"))
#: Its placebo window for one year back, 20 Feb to 10 Mar 2024, holds 29 Feb, which 2025 does not have.
LEAP = _analysis("LEAP", ("2023-07-01", "2023-12-31"), ("2024-07-01", "2024-09-30"), ("2025-02-20", "2025-03-10"))


@pytest.fixture
def stubbed(monkeypatch):
    """es.run_analogs with ``cached_scores`` replaced by the synthetic frame."""
    import es.run_analogs as ra
    monkeypatch.setattr(ra, "cached_scores", stub_scores)
    return ra


# ---------------------------------------------------------------------------------------------------------------
# placebo_cube and _settled_rows
# ---------------------------------------------------------------------------------------------------------------

def test_settled_rows_give_the_primary_scores_and_the_training_direction(stubbed):
    primary, direction = stubbed._settled_rows(None, TINY, None, None)
    assert set(primary["split"]) >= {"train", "calib", "deploy"}
    assert direction.index.names == ["market", "node", "settle_hour"] and set(direction) <= {1, -1}
    assert len(direction) == len(NODES) * 24
    train = primary[primary["split"] == "train"]
    mean = (train["da_lmp"] - train["rt_lmp"]).groupby([train["market"], train["node"], train["settle_hour"]]).mean()
    assert ((mean >= 0).astype(int) * 2 - 1).tolist() == direction.tolist()


@pytest.mark.parametrize("years_back", [1, 2])
def test_placebo_cube_pairs_the_alerts_with_the_pnl_of_the_same_dates_earlier(stubbed, years_back):
    """Every row of the cube is a (node, date, hour) of the placebo window carrying that hour's P&L (premium position,
    calibrated fee) and the alert that P1 raised on the same calendar date years_back years later."""
    from es.policies import p1_fixed
    from es.positions import CALIBRATED_FEES
    cube = stubbed.placebo_cube(None, TINY, years_back, None, None)
    sc = stub_scores(None, TINY, "G1_24")
    a = p1_fixed(sc, 0.05)[["market", "node", "settle_date", "settle_hour", "alert"]].copy()
    a["settle_date"] = a["settle_date"].map(lambda d: d.replace(year=d.year - years_back))
    expect = sc.drop(columns="split").merge(a, on=["market", "node", "settle_date", "settle_hour"])
    train = sc[sc["split"] == "train"]
    keys = [train["market"], train["node"], train["settle_hour"]]
    d = ((train["da_lmp"] - train["rt_lmp"]).groupby(keys).mean() >= 0)
    expect = expect.join(d.map({True: 1, False: -1}).rename("d"), on=["market", "node", "settle_hour"])
    fee = [CALIBRATED_FEES[m][int(x)] for m, x in zip(expect["market"], expect["d"])]
    expect["pnl"] = expect["d"] * (expect["da_lmp"] - expect["rt_lmp"]) - fee
    expect["rank"] = expect["node"].map({"M1": 0, "N1": 1})
    expect = expect.sort_values(["rank", "settle_date", "settle_hour"])
    got = pd.DataFrame({"rank": cube.cell_idx // 24, "settle_date": cube.days[cube.day_idx],
                        "settle_hour": cube.cell_idx % 24, "pnl": cube.pnl, "alert": cube.alert.astype(bool)})
    got = got.sort_values(["rank", "settle_date", "settle_hour"])
    assert len(got) == len(expect) == 20 * 24 * len(NODES)
    assert got["settle_date"].min() == dt.date(2025 - years_back, 7, 1)    # the placebo dates, not the alert dates
    assert (got["rank"].to_numpy() == expect["rank"].to_numpy()).all()
    assert (got["settle_date"].to_numpy() == expect["settle_date"].to_numpy()).all()
    assert (got["settle_hour"].to_numpy() == expect["settle_hour"].to_numpy()).all()
    np.testing.assert_allclose(got["pnl"].to_numpy(), expect["pnl"].to_numpy())
    assert (got["alert"].to_numpy() == expect["alert"].to_numpy()).all()
    assert got["alert"].sum() > 0                                              # the pairing carries alerts


def test_placebo_cube_is_stratified_by_hour_weekday_class_and_month_of_the_pnl_dates(stubbed):
    from es.bootstrap import rotation_test
    cube = stubbed.placebo_cube(None, TINY, 1, None, None)
    assert cube.n_strata == len(NODES) * 24 * 2 * 1                            # node x hour x weekday class x one month
    res = rotation_test([cube])
    assert res["n_null"] == 20 - 1                                             # the full cyclic group (Ruling 69)
    assert set(res) >= {"t_obs", "p", "n_null", "selc_obs"}


def test_placebo_cube_counts_the_rows_whose_settlement_leg_is_approximate(stubbed):
    """Ruling 94 (M1): where no settlement table covers a row (here NYISO, whose table is None) its P&L rests on the
    5-minute RT mean, and the cube says how many of its rows do."""
    cube = stubbed.placebo_cube(None, TINY, 1, None, None)
    assert cube.n_rows == 20 * 24 * len(NODES) and cube.n_settle_approx == 20 * 24   # NYISO's rows; MISO settles on RT


def test_placebo_cube_raises_a_value_error_for_a_window_without_rows(stubbed):
    with pytest.raises(ValueError, match="no rows"):
        stubbed.placebo_cube(None, TINY, 10, None, None)


def test_placebo_cube_says_what_it_leaves_unpaired(stubbed, recwarn):
    stubbed.placebo_cube(None, TINY, 1, None, None)
    assert not [w for w in recwarn if "placebo_cube" in str(w.message)]         # nothing unpaired, nothing said
    with pytest.warns(UserWarning, match=r"placebo_cube.*without a paired alert"):
        cube = stubbed.placebo_cube(None, LEAP, 1, None, None)                  # 29 Feb 2024 has no 2025 counterpart
    assert cube.n_days == 19                                                    # 20 Feb - 10 Mar 2025 has 19 days
    assert dt.date(2024, 2, 29) not in set(cube.days)


# ---------------------------------------------------------------------------------------------------------------
# eta, the activity table and the placebo table
# ---------------------------------------------------------------------------------------------------------------

def test_select_eta_pools_the_four_quarters_and_breaks_ties_to_the_smaller_eta():
    from es.run_analogs import ETAS, select_eta
    errors = {"AQ3": (0.5, 0.25, 0.25, 0.75), "AQ4": (0.5, 0.25, 0.25, 0.75),
              "AQ1": (0.5, 0.25, 0.25, 0.75), "AQ2": (0.5, 0.25, 0.25, 0.75),
              "AQ3F": (0.9, 0.9, 0.9, 0.0)}                     # the frozen arm alone would pick 0.05; it is not pooled
    rows = [{"analysis": a, "eta": eta, "median_log_error": e} for a, es in errors.items() for eta, e in zip(ETAS, es)]
    assert select_eta(pd.DataFrame(rows)) == 0.01              # 0.01 and 0.02 tie at 0.25: the smaller one


def test_select_eta_is_the_minimum_of_the_pooled_panel_median():
    from es.run_analogs import ETAS, select_eta
    rows = [{"analysis": a, "eta": eta, "median_log_error": e}
            for a, pattern in (("AQ3", (0.4, 0.3, 0.2, 0.5)), ("AQ4", (0.4, 0.3, 0.1, 0.5)),
                               ("AQ1", (0.4, 0.3, 0.2, 0.5)), ("AQ2", (0.4, 0.3, 0.2, 0.5)))
            for eta, e in zip(ETAS, pattern)]
    assert select_eta(pd.DataFrame(rows)) == 0.02


def test_eta_table_has_a_row_per_analysis_and_eta(stubbed):
    from es.analyses import ANALOGS
    tab = stubbed.eta_table(None)
    assert list(tab.columns) == ["analysis", "eta", "median_log_error"]
    assert len(tab) == len(ANALOGS) * len(stubbed.ETAS) and set(tab["analysis"]) == set(ANALOGS)
    assert set(tab["eta"]) == set(stubbed.ETAS) and tab["median_log_error"].notna().all()
    # one cell recomputed by hand: AQ3, eta 0.02
    from es.policies import activity, p5_self_correcting
    want = float(activity(p5_self_correcting(stub_scores(None, ANALOGS["AQ3"], "G1_24"), 0.05, 0.02), 0.05)
                 ["log_error"].median())
    got = tab[(tab["analysis"] == "AQ3") & (tab["eta"] == 0.02)]["median_log_error"].iloc[0]
    assert got == pytest.approx(want)


def test_activity_table_covers_every_analysis_block_estimator_and_policy(stubbed, monkeypatch):
    from es.analyses import ANALOGS
    from es.policies import activity, p4_rolling
    monkeypatch.setattr(stubbed, "BLOCKS", {"G6": [], "LEV6": []})
    tab = stubbed.activity_table(None)
    assert set(tab["analysis"]) == set(ANALOGS) and set(tab["block"]) == {"G6", "LEV6"}
    assert set(tab["estimator"]) == {"xgb", "logit"} and set(tab["policy"]) == {"P1", "P4_56", "P4_28", "P6"}
    assert len(tab) == len(ANALOGS) * 2 * 2 * 4 * len(NODES)
    assert {"market", "node", "n", "alerts", "rate", "R", "log_error", "fail_0.5_2"} <= set(tab.columns)
    want = activity(p4_rolling(stub_scores(None, ANALOGS["AQ1"], "LEV6", "logit"), 0.05, 28), 0.05)
    got = tab[(tab["analysis"] == "AQ1") & (tab["block"] == "LEV6") & (tab["estimator"] == "logit")
              & (tab["policy"] == "P4_28")]
    assert got["alerts"].tolist() == want["alerts"].tolist() and got["log_error"].tolist() == want["log_error"].tolist()


def test_placebo_table_has_the_eight_single_and_two_pooled_pairings(stubbed):
    rows = stubbed.placebo_table(None, None, None)
    singles = [f"{a}-{k}y" for a in ("AQ3", "AQ4", "AQ1", "AQ2") for k in (1, 2)]
    pairs = singles + ["AQ3+AQ4-1y", "AQ3+AQ4-2y"]
    assert [r["pair"] for r in rows] == pairs
    base = {"pair", "t_obs", "p", "n_null", "selc_obs", "n_rows", "n_settle_approx", "counted"}      # Rulings 64, 94
    assert all(set(r) == base | (set() if r["counted"] else {"note"}) for r in rows)
    assert [r["n_null"] for r in rows] == [91, 91, 91, 91, 89, 89, 90, 90, 92 * 92 - 1, 92 * 92 - 1]   # Ruling 69
    assert all(0 < r["p"] <= 1 for r in rows)
    # Ruling 94, C1: the 1-year pairings count; the 2-year ones are reported, with the reason
    assert {r["pair"]: r["counted"] for r in rows} == {p: p.endswith("-1y") for p in pairs}
    assert all(r["note"] == UNCOUNTED for r in rows if not r["counted"])
    # M1: the rows resting on an approximate settlement leg, per pairing; a pooled pairing adds its two quarters
    by = {r["pair"]: r for r in rows}
    for k in (1, 2):
        pooled, a, b = by[f"AQ3+AQ4-{k}y"], by[f"AQ3-{k}y"], by[f"AQ4-{k}y"]
        assert pooled["n_rows"] == a["n_rows"] + b["n_rows"]
        assert pooled["n_settle_approx"] == a["n_settle_approx"] + b["n_settle_approx"]
    assert all(0 < r["n_settle_approx"] < r["n_rows"] for r in rows)           # NYISO's rows only, in the stub


def test_placebo_table_records_a_pairing_that_cannot_be_built_and_still_runs_the_rest(stubbed, monkeypatch):
    real = stubbed.placebo_cube
    broken = {("AQ4", 2), ("AQ1", 1)}

    def flaky(panel, analysis, years_back, fmm, nyi):
        if (analysis.id, years_back) in broken:
            raise ValueError("no rows in the placebo window")
        return real(panel, analysis, years_back, fmm, nyi)

    monkeypatch.setattr(stubbed, "placebo_cube", flaky)
    rows = {r["pair"]: r for r in stubbed.placebo_table(None, None, None)}
    assert rows["AQ4-2y"] == {"pair": "AQ4-2y", "error": "no rows in the placebo window", "counted": False,
                              "note": UNCOUNTED}
    assert rows["AQ1-1y"] == {"pair": "AQ1-1y", "error": "no rows in the placebo window", "counted": True}
    assert "p" in rows["AQ4-1y"] and "p" in rows["AQ3+AQ4-1y"] and "p" in rows["AQ3-2y"]
    assert "p" not in rows["AQ3+AQ4-2y"] and "AQ4-2y" in rows["AQ3+AQ4-2y"]["error"]       # never absent in silence
    assert rows["AQ3+AQ4-2y"]["counted"] is False and rows["AQ3+AQ4-2y"]["note"] == UNCOUNTED


def test_placebo_table_records_a_window_too_short_for_the_rotation_test(stubbed, monkeypatch):
    """Ruling 49: both rotation_test calls, the single-quarter one and the pooled one, sit inside the try."""
    real = stubbed.rotation_test

    def picky(cubes):
        if len(cubes) == 2:
            raise ValueError("rotation_test: window(s) of 1 days leave no admissible rotation")
        if cubes[0].n_days == 90:
            raise ValueError("rotation_test: window(s) of 90 days are too short for this test")
        return real(cubes)

    monkeypatch.setattr(stubbed, "rotation_test", picky)
    rows = {r["pair"]: r for r in stubbed.placebo_table(None, None, None)}
    assert rows["AQ1-1y"] == {"pair": "AQ1-1y", "counted": True,
                              "error": "rotation_test: window(s) of 90 days are too short for this test"}
    assert all(rows[f"AQ3+AQ4-{k}y"]["error"].startswith("rotation_test") for k in (1, 2))
    assert rows["AQ3+AQ4-1y"]["counted"] is True and rows["AQ3+AQ4-2y"]["counted"] is False
    assert "p" in rows["AQ3-1y"] and "p" in rows["AQ2-2y"]


# ---------------------------------------------------------------------------------------------------------------
# Stage "sim": the pieces of the simulation study, with he_size_power / hb_size / calibrate_delta replaced by recorders
# ---------------------------------------------------------------------------------------------------------------

EXPECTED_HE_SIZE = [(t, p, s, st) for t in ("H-E single A, size", "H-E pooled A+B, size") for p in ("fit", "upper")
                    for s in ("homoskedastic", "heteroskedastic") for st in ("primary", "decile")]
HB_LAWS = ("DART3", "LEV6", "ROLL_DISP9")
HB_SCENARIOS = ("no_drift", "common_drift_0.25", "common_drift_0.5")
#: (test, law, persistence, scenario) of the 18 H-B size rows, in the order they are made (Ruling 84b, items 1 and 2).
EXPECTED_HB_SIZE = [("H-B size", law, p, sc) for law in HB_LAWS for p in ("fit", "upper") for sc in HB_SCENARIOS]


def _fake_he(calls, slope=0.1, means=None):
    """A stand-in for he_size_power: mean SELC slope * rho (or means[rho] when a curve is given), rejection rates
    0.02 + rho and 0.05 + rho (so they equal 0.02 and 0.05 at rho = 0, as under no skill)."""
    def fake(params, cal, dep, rho, n_sim, **kw):
        calls.append({"cal": cal, "dep": dep, "rho": rho, "n_sim": n_sim, **kw})
        return {"reject_0.0166667": 0.02 + rho, "reject_0.05": 0.05 + rho,
                "mean_selc": means[rho] if means else slope * rho,
                "mean_t_obs": 0.0,
                "n_sim": n_sim, "rho": rho, "phi_scale": 1.0, "hetero": kw.get("hetero", False),
                "persistence": kw.get("persistence", "fit"), "strata": kw.get("strata", "primary")}
    return fake


def _fake_hb(calls, mean_gap=0.1, t_hat=0.1):
    """A stand-in for hb_size. ``mean_node_gap`` is the mean over nodes of the paired gap; ``mean_t_hat`` is the mean
    over simulations of its median over nodes, the scale of the median-based effect target (Ruling 84b item 5)."""
    def fake(params, cal, dep, n_sim, n_rep=299, **kw):
        calls.append({"params": params, "cal": cal, "dep": dep, "n_sim": n_sim, "n_rep": n_rep, **kw})
        return {"reject_0.0166667": 0.6, "reject_0.05": 0.8, "nodrift_fail_q95_0.5_2": 1.0, "mean_t_hat": t_hat,
                "mean_node_gap": mean_gap, "n_sim": n_sim, "delta": kw.get("delta", 0.0), "phi_scale": 1.0,
                "persistence": kw.get("persistence", "fit"), "common_shift": kw.get("common_shift", 0.0)}
    return fake


def _laws():
    """{law: {node: fit}}, a recognisable persistence per law: phi_s 0.1 * i and phi_s_raw 0.05 * i for the i-th."""
    return {law: {("M", "n"): SimpleNamespace(phi_s=0.1 * i, phi_s_raw=0.05 * i)} for i, law in enumerate(HB_LAWS, 1)}


def test_the_registered_replication_counts_and_windows():
    from es.run_analogs import SimCounts, SimWindows
    c = SimCounts()
    # H-B size: 300 simulations of 299 bootstrap replicates, so alpha/3 and 0.05 are exactly attainable (Ruling 84b)
    # H-B power: 300 simulations too (Ruling 94, M11: at 150 its standard error near 0.5 is 0.041)
    assert (c.he_size, c.he_rho, c.he_power, c.hb_size, c.hb_power, c.hb_rep, c.hb_calibrate) == \
        (300, 100, 300, 300, 300, 299, 100)
    w = SimWindows.default()
    assert (w.cal_a[0], w.cal_a[-1], len(w.cal_a)) == (dt.date(2025, 7, 1), dt.date(2025, 9, 30), 92)
    assert (w.cal_b[0], w.cal_b[-1], len(w.cal_b)) == (dt.date(2025, 10, 1), dt.date(2025, 12, 31), 92)
    assert (w.dep_a[0], w.dep_a[-1], len(w.dep_a)) == (dt.date(2026, 7, 1), dt.date(2026, 9, 30), 92)
    assert (w.dep_b[0], w.dep_b[-1], len(w.dep_b)) == (dt.date(2026, 10, 1), dt.date(2026, 12, 31), 92)


def test_he_size_rows_are_the_registered_sixteen_row_matrix(monkeypatch):
    import es.run_analogs as ra
    calls = []
    monkeypatch.setattr(ra, "he_size_power", _fake_he(calls))
    w, c = ra.SimWindows.default(), ra.SimCounts()
    rows = ra.he_size_rows({("M", "n"): object()}, w, c)
    assert [(r["test"], r["persistence"], r["scenario"], r["strata"]) for r in rows] == EXPECTED_HE_SIZE
    assert len(rows) == len(calls) == 16
    assert all("phi_scale" not in k for k in calls)                      # deprecated: never passed (appendix A)
    for i, (r, k) in enumerate(zip(rows, calls)):
        assert k["seed"] == r["seed"] == 20261022 + i                         # Ruling 94, I2: 20261022 + row index
        pooled = r["test"].startswith("H-E pooled")
        assert k["rho"] == 0.0 and k["n_sim"] == 300
        assert k["hetero"] == (r["scenario"] == "heteroskedastic")
        assert k["persistence"] == r["persistence"] and k["strata"] == r["strata"]
        assert k["cal"] == ([w.cal_a, w.cal_b] if pooled else [w.cal_a])
        assert k["dep"] == ([w.dep_a, w.dep_b] if pooled else [w.dep_a])
    assert all({"reject_0.0166667", "reject_0.05", "n_sim"} <= set(r) for r in rows)


def test_he_power_rows_are_a_curve_over_the_rho_grid_and_then_the_target_row(monkeypatch):
    """Ruling 92b item 1: the rho search is reported as a curve, one row per grid point with its rejection rates and
    mean SELC, and the single target row follows with its own run (300 simulations at the chosen rho)."""
    import es.run_analogs as ra
    calls = []
    monkeypatch.setattr(ra, "he_size_power", _fake_he(calls, slope=0.1))        # mean_selc = 0.1 * rho
    w, c = ra.SimWindows.default(), ra.SimCounts()
    rows, rho_star = ra.he_power_rows({("M", "n"): object()}, w, c, target=0.045)
    curve, row = rows[:-1], rows[-1]
    assert len(rows) == 8 and [r["test"] for r in curve] == ["H-E pooled A+B, power curve"] * 7
    assert [r["rho"] for r in curve] == [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]       # Ruling 63's grid
    for r in curve:
        assert r["n_sim"] == 100 and r["scenario"] == "homoskedastic" and "note" not in r
        assert r["reject_0.0166667"] == pytest.approx(0.02 + r["rho"])
        assert r["reject_0.05"] == pytest.approx(0.05 + r["rho"])
        assert r["mean_selc"] == pytest.approx(0.1 * r["rho"])
    assert rho_star == 0.5 and row["rho"] == 0.5 and row["target_selc"] == 0.045 and "note" not in row
    assert row["test"] == "H-E pooled A+B, power" and row["scenario"] == "homoskedastic" and row["n_sim"] == 300
    assert len(calls) == 8                                                   # the curve IS the search: no extra runs
    assert all(k["n_sim"] == 100 and k["seed"] == 7 for k in calls[:-1])
    assert calls[-1]["rho"] == 0.5 and calls[-1]["n_sim"] == 300 and calls[-1]["seed"] == 11
    assert calls[-1]["cal"] == [w.cal_a, w.cal_b] and calls[-1]["dep"] == [w.dep_a, w.dep_b]


def test_he_power_rows_without_a_bracketing_grid_point_use_rho_0_9_and_say_so(monkeypatch):
    import es.run_analogs as ra
    peaked = dict(zip(ra.RHO_GRID, (0.01, 0.02, 0.03, 0.04, 0.05, 0.09, 0.08)))   # tops out at rho = 0.7, not 0.9
    monkeypatch.setattr(ra, "he_size_power", _fake_he([], means=peaked))
    w, c = ra.SimWindows.default(), ra.SimCounts()
    rows, rho_star = ra.he_power_rows({("M", "n"): object()}, w, c, target=0.5)
    assert rho_star == 0.9 and rows[-1]["rho"] == 0.9 and rows[-1]["note"] == "target not bracketed"   # not the nearest
    assert all("note" not in r for r in rows[:-1])                          # the note belongs to the target row only
    below, rho_low = ra.he_power_rows({("M", "n"): object()}, w, c, target=-0.01)   # below every grid point
    assert rho_low == 0.05 and below[-1]["note"] == "target below the grid"
    assert [r["rho"] for r in below[:-1]] == list(ra.RHO_GRID)              # the whole curve is reported all the same


def test_hb_size_rows_cross_the_three_laws_with_persistence_and_drift(monkeypatch):
    """Ruling 84b items 1 to 3: both blocks drawn from DART3's, LEV6's or ROLL_DISP9's SIMFIT fit, at both persistence
    settings, with no drift and common drifts of 0.25 and 0.5 score SDs: 18 rows of 300 simulations, 299 replicates."""
    import es.run_analogs as ra
    calls = []
    monkeypatch.setattr(ra, "hb_size", _fake_hb(calls))
    w, c = ra.SimWindows.default(), ra.SimCounts()
    laws = _laws()
    rows = ra.hb_size_rows(laws, w, c)
    assert [(r["test"], r["law"], r["persistence"], r["scenario"]) for r in rows] == EXPECTED_HB_SIZE
    assert len(rows) == len(calls) == 18
    assert all("strata" not in r for r in rows)                                # H-B rows carry no strata
    assert [k["common_shift"] for k in calls] == [0.0, 0.25, 0.5] * 6
    assert all(k["n_sim"] == 300 and k["n_rep"] == 299 and "phi_scale" not in k and k["cal"] == w.cal_a
               and k["dep"] == w.dep_a for k in calls)
    for i, (r, k) in enumerate(zip(rows, calls)):
        assert k["seed"] == r["seed"] == 20261022 + 16 + i                    # the 16 H-E size rows come first (I2)
        assert k["params"] is laws[r["law"]]                                  # each law draws from its own fit
        i = HB_LAWS.index(r["law"]) + 1                                       # and reports that fit's persistence (I-3)
        assert r["phi_score_median"] == pytest.approx(0.1 * i) and r["phi_score_raw_median"] == pytest.approx(0.05 * i)


def test_every_size_row_has_a_seed_of_its_own(monkeypatch):
    """Ruling 94, I2: the Bonferroni correction assumes independent rows, so no two size rows share a seed (common
    random numbers made them copies of one experiment). H-E size rows take 20261022 + 0..15 in the order test (single,
    pooled), persistence, scenario, strata; H-B size rows 20261022 + 16..33 in the order law, persistence, drift."""
    import es.run_analogs as ra
    monkeypatch.setattr(ra, "he_size_power", _fake_he([]))
    monkeypatch.setattr(ra, "hb_size", _fake_hb([]))
    w, c = ra.SimWindows.default(), ra.SimCounts()
    seeds = [r["seed"] for r in ra.he_size_rows({("M", "n"): object()}, w, c)]
    seeds += [r["seed"] for r in ra.hb_size_rows(_laws(), w, c)]
    assert seeds == list(range(20261022, 20261022 + 34)) and len(set(seeds)) == 34


def test_the_other_rows_record_their_documented_seeds(stubbed, monkeypatch):
    """The power curve (seed 7), the H-E target row (11) and the H-B power runs (13, with the calibration at 20261022)
    keep their own seeds, and say so in the row."""
    monkeypatch.setattr(stubbed, "he_size_power", _fake_he([]))
    rows, _ = stubbed.he_power_rows({("M", "n"): object()}, stubbed.SimWindows.default(), stubbed.SimCounts(), 0.045)
    assert [r["seed"] for r in rows] == [7] * 7 + [11]
    monkeypatch.setattr(stubbed, "calibrate_delta", lambda p, c, d, target, **kw: {"delta": 0.2, "target_gap": target})
    monkeypatch.setattr(stubbed, "hb_size", _fake_hb([]))
    power = stubbed.hb_power_rows(None, _laws(), stubbed.SimWindows.default(), stubbed.SimCounts())
    assert [(r["seed"], r["calibration_seed"]) for r in power] == [(13, 20261022)] * 2


def test_phi_medians_read_the_raw_persistence_only_where_the_fit_has_it():
    """Ruling 84b item 6: the median over nodes of the score's fitted persistence (``phi_s``, as simulated) and of its
    plain AR(1) value before the indirect-inference correction (``phi_s_raw``, the name the landed NodeParams uses; the
    ruling called it phi_raw, which is read when phi_s_raw is absent); NaN when a fit has neither."""
    from es.run_analogs import _phi_medians
    plain = {("M", str(i)): SimpleNamespace(phi_s=v) for i, v in enumerate((0.1, 0.2, 0.9))}     # median 0.2, mean 0.4
    got = _phi_medians(plain)
    assert got["phi_score_median"] == pytest.approx(0.2) and math.isnan(got["phi_score_raw_median"])
    raw = {k: SimpleNamespace(phi_s=v.phi_s, phi_s_raw=v.phi_s - 0.1) for k, v in plain.items()}
    assert _phi_medians(raw)["phi_score_raw_median"] == pytest.approx(0.1)             # median of 0.0, 0.1, 0.8
    assert _phi_medians(raw)["phi_score_median"] == pytest.approx(0.2)
    named_in_the_ruling = {k: SimpleNamespace(phi_s=v.phi_s, phi_raw=v.phi_s - 0.05) for k, v in plain.items()}
    assert _phi_medians(named_in_the_ruling)["phi_score_raw_median"] == pytest.approx(0.15)
    both = {k: SimpleNamespace(phi_s=v.phi_s, phi_s_raw=0.3, phi_raw=0.7) for k, v in plain.items()}
    assert _phi_medians(both)["phi_score_raw_median"] == pytest.approx(0.3)            # the landed name wins
    none_yet = {k: SimpleNamespace(phi_s=v.phi_s, phi_s_raw=None) for k, v in plain.items()}
    assert math.isnan(_phi_medians(none_yet)["phi_score_raw_median"])


def _aq3_gap(block):
    """Half the median over nodes of the AQ3 |log10 R| gap of ``block`` to DART3 under P1, on the synthetic frame."""
    from es.analyses import ANALOGS
    from es.policies import activity, p1_fixed
    alt = activity(p1_fixed(stub_scores(None, ANALOGS["AQ3"], block), 0.05), 0.05)
    dar = activity(p1_fixed(stub_scores(None, ANALOGS["AQ3"], "DART3"), 0.05), 0.05)
    both = alt.merge(dar, on=["market", "node"], suffixes=("_alt", "_dart"))
    return 0.5 * float((both["log_error_alt"] - both["log_error_dart"]).median())


def test_aq3_level_gap_is_half_the_median_over_nodes_not_the_mean(monkeypatch):
    """Ruling 84b item 5 (M-4): 0.5 x the median over nodes of |log10 R|(block) - |log10 R|(DART3) on AQ3."""
    import es.run_analogs as ra
    frames = iter([pd.DataFrame({"market": "M", "node": list("abc"), "log_error": [0.9, 0.5, 0.4]}),      # the block
                   pd.DataFrame({"market": "M", "node": list("abc"), "log_error": [0.1, 0.4, 0.3]})])     # DART3
    monkeypatch.setattr(ra, "cached_scores", lambda *a, **k: None)
    monkeypatch.setattr(ra, "p1_fixed", lambda scores, alpha: None)
    monkeypatch.setattr(ra, "activity", lambda alerts, alpha: next(frames))
    assert ra._aq3_level_gap(None, "LEV6") == pytest.approx(0.5 * 0.1)         # differences 0.8, 0.1, 0.1: median 0.1


def test_hb_power_rows_calibrate_delta_to_half_the_aq3_gap(stubbed, monkeypatch):
    hb_calls, cal_calls = [], []

    def fake_calibrate(params, cal, dep, target, **kw):
        cal_calls.append({"params": params, "cal": cal, "dep": dep, "target": target, **kw})
        return {"delta": 0.3, "grid": [0.1, 0.3], "grid_gaps": [0.0, target], "target_gap": target}

    monkeypatch.setattr(stubbed, "calibrate_delta", fake_calibrate)
    monkeypatch.setattr(stubbed, "hb_size", _fake_hb(hb_calls, mean_gap=0.07, t_hat=0.06))
    w, c = stubbed.SimWindows.default(), stubbed.SimCounts()
    laws = _laws()
    rows = stubbed.hb_power_rows(None, laws, w, c)
    assert [r["test"] for r in rows] == ["H-B1 power", "H-B2 power"]
    assert len(cal_calls) == len(hb_calls) == 2                          # one calibration and one simulation each
    for r, block, call, hb in zip(rows, ("LEV6", "ROLL_DISP9"), cal_calls, hb_calls):
        gap = _aq3_gap(block)
        assert r["target_log_error_gap"] == pytest.approx(gap) and call["target"] == pytest.approx(gap)
        assert r["calibrated_delta"] == 0.3 and r["scenario"] == "no_drift"
        assert r["mean_node_gap"] == 0.07 and r["mean_t_hat"] == 0.06        # both simulated gaps are in the row
        assert r["law"] == "DART3" and r["phi_score_median"] == pytest.approx(0.1)     # both blocks draw from DART3
        assert call["n_sim"] == 100 and call["cal"] == w.cal_a and call["dep"] == w.dep_a   # Task 17: +-0.015 at 30
        assert tuple(call["grid"]) == stubbed.CALIBRATION_GRID and call["seed"] == 20261022
        assert call["n_rep"] == 299 and call["params"] is laws["DART3"] and hb["params"] is laws["DART3"]
        assert hb["delta"] == 0.3 and hb["seed"] == 13 and hb["n_sim"] == 300 and hb["n_rep"] == 299   # Ruling 94, M11
        assert "phi_scale" not in hb and "phi_scale" not in call
        assert "reject_0.0166667" in r and "error" not in r


def test_hb_power_row_records_a_failed_calibration_below_the_grid_as_an_error_row(stubbed, monkeypatch):
    """Ruling 63, as amended by 94 (M4, NB1): a calibration that raises with the simulated gap at the grid's top already
    above the target is a target below the grid: an error row, no power, and the one run made for it (at delta 0.8, with
    the calibration's own seed and simulation count, to tell below from above) is not kept."""
    hb_calls = []

    def failing(params, cal, dep, target, **kw):
        if abs(target - _aq3_gap("LEV6")) < 1e-12:
            raise ValueError("calibrate_delta: target_gap is outside the gap range of the grid")
        return {"delta": 0.2, "grid": [], "grid_gaps": [], "target_gap": target}

    monkeypatch.setattr(stubbed, "calibrate_delta", failing)
    monkeypatch.setattr(stubbed, "hb_size", _fake_hb(hb_calls, t_hat=5.0))        # the gap at delta 0.8 is far above it
    rows = stubbed.hb_power_rows(None, _laws(), stubbed.SimWindows.default(), stubbed.SimCounts())
    assert rows[0]["test"] == "H-B1 power" and "outside the gap range" in rows[0]["error"]
    assert rows[0]["law"] == "DART3" and rows[0]["persistence"] == "fit"
    assert "reject_0.0166667" not in rows[0] and rows[0]["target_log_error_gap"] == pytest.approx(_aq3_gap("LEV6"))
    assert "power_lower_bound" not in rows[0]
    assert rows[1]["test"] == "H-B2 power" and "error" not in rows[1]
    assert [k["delta"] for k in hb_calls] == [0.8, 0.2]                       # the check at the top, then H-B2's power
    assert (hb_calls[0]["seed"], hb_calls[0]["n_sim"]) == (20261022, 100)   # the check is the calibration's draw (NB1)
    assert (hb_calls[1]["seed"], hb_calls[1]["n_sim"]) == (13, 300)         # the power run is not


def _calibration_out_of_range(params, cal, dep, target, **kw):
    raise ValueError(f"calibrate_delta: target_gap {target:.4g} is outside the gap range of the grid [0.01, 0.2]")


def test_a_target_above_the_grid_records_the_power_at_its_top_as_a_lower_bound(stubbed, monkeypatch):
    """Ruling 94 (M4, NB1): a target beyond the largest gap the grid reaches (the calibration's own simulated gap at
    delta = 0.8 stays below it) is a very large effect. The power at delta = 0.8, from the usual power run, is recorded
    as a lower bound, with the calibration's message, and the test is not demoted."""
    from es.run_analogs import registration_family
    hb_calls = []
    monkeypatch.setattr(stubbed, "calibrate_delta", _calibration_out_of_range)
    monkeypatch.setattr(stubbed, "_aq3_level_gap", lambda panel, block: 0.5)
    monkeypatch.setattr(stubbed, "hb_size", _fake_hb(hb_calls, t_hat=0.2))          # the gap at delta 0.8 is 0.2 < 0.5
    rows = stubbed.hb_power_rows(None, _laws(), stubbed.SimWindows.default(), stubbed.SimCounts())
    assert len(hb_calls) == 4 and all(k["delta"] == 0.8 for k in hb_calls)     # per test: the check, then the power run
    assert [(k["seed"], k["n_sim"]) for k in hb_calls] == [(20261022, 100), (13, 300)] * 2
    for r in rows:
        assert r["power_lower_bound"] is True and r["delta"] == 0.8 and r["seed"] == 13
        assert "outside the gap range" in r["calibration_error"] and "lower bound" in r["note"]
        assert "error" not in r and "calibrated_delta" not in r and "reject_0.0166667" in r
    sim = [r for r in _sim() if "power" not in r["test"]] + rows
    low = {**rows[0], "reject_0.0166667": 0.2}                                      # even a low bound does not demote
    fam = registration_family([r for r in sim if r is not rows[0]] + [low], _placebo())
    assert fam["H-B1"]["confirmatory"] and fam["H-B1"]["power_lower_bound"] and fam["H-B1"]["power"] == 0.2


def test_a_target_below_the_grid_or_not_finite_still_demotes(stubbed, monkeypatch):
    """Below the grid (the gap at delta = 0.8 already exceeds the target, so the grid's bottom is what is above it)
    and a non-finite target remain failed calibrations: an error row, power counted as below the floor."""
    from es.run_analogs import registration_family
    hb_calls = []
    monkeypatch.setattr(stubbed, "calibrate_delta", _calibration_out_of_range)
    monkeypatch.setattr(stubbed, "hb_size", _fake_hb(hb_calls, t_hat=0.9))
    for target in (0.5, float("nan")):
        monkeypatch.setattr(stubbed, "_aq3_level_gap", lambda panel, block, target=target: target)
        rows = stubbed.hb_power_rows(None, _laws(), stubbed.SimWindows.default(), stubbed.SimCounts())
        assert all("outside the gap range" in r["error"] and "power_lower_bound" not in r for r in rows)
        assert all("reject_0.0166667" not in r for r in rows)
        sim = [r for r in _sim() if "power" not in r["test"]] + rows
        fam = registration_family(sim, _placebo())
        assert not fam["H-B1"]["confirmatory"] and not fam["H-B2"]["confirmatory"]
        assert not fam["H-B1"]["power_lower_bound"]
        assert len(hb_calls) == 2                   # a check for each test at 0.5; a target that is NaN checks nothing
    assert not any(k["delta"] != 0.8 or k["seed"] != 20261022 or k["n_sim"] != 100 for k in hb_calls)


def _fake_hb_by_seed(calls, calibration_gap, power_gap):
    """hb_size whose simulated gap depends on the seed: the calibration's own (20261022) reads ``calibration_gap``, any
    other seed (the power run's 13) ``power_gap``. Two Monte Carlo draws that can read opposite sides of one target."""
    base = _fake_hb(calls)

    def fake(params, cal, dep, n_sim, n_rep=299, **kw):
        out = base(params, cal, dep, n_sim, n_rep, **kw)
        return {**out, "mean_t_hat": calibration_gap if kw.get("seed") == 20261022 else power_gap}

    return fake


def test_the_direction_of_a_failed_calibration_is_read_from_the_calibrations_own_draw(stubbed, monkeypatch):
    """NB1: calibrate_delta raised because its own gaps (seed 20261022, 100 simulations) leave the target outside the
    grid. Whether the target is above the grid's top or below its bottom is read from the delta = 0.8 gap of that same
    draw, not from the 300-simulation power run (seed 13), which near the edge can read the other side of the target."""
    monkeypatch.setattr(stubbed, "calibrate_delta", _calibration_out_of_range)
    monkeypatch.setattr(stubbed, "_aq3_level_gap", lambda panel, block: 0.5)
    w, c, laws, calls = stubbed.SimWindows.default(), stubbed.SimCounts(), _laws(), []
    # the calibration's draw ends below the target (above the grid) although the power run reads above it
    monkeypatch.setattr(stubbed, "hb_size", _fake_hb_by_seed(calls, calibration_gap=0.45, power_gap=0.55))
    rows = stubbed.hb_power_rows(None, laws, w, c)
    assert all(r["power_lower_bound"] is True and "error" not in r and r["seed"] == 13 for r in rows)
    assert all(r["reject_0.0166667"] == 0.6 for r in rows)                     # the power is the power run's
    check, power = calls[0], calls[1]
    assert (check["delta"], check["seed"], check["n_sim"], check["n_rep"]) == (0.8, 20261022, 100, 299)
    assert check["params"] is laws["DART3"] and check["cal"] == w.cal_a and check["dep"] == w.dep_a
    assert (power["delta"], power["seed"], power["n_sim"]) == (0.8, 13, 300) and len(calls) == 4
    # the calibration's draw ends above the target (below the grid) although the power run reads below it
    calls.clear()
    monkeypatch.setattr(stubbed, "hb_size", _fake_hb_by_seed(calls, calibration_gap=0.55, power_gap=0.45))
    rows = stubbed.hb_power_rows(None, laws, w, c)
    assert all("error" in r and "power_lower_bound" not in r and "reject_0.0166667" not in r for r in rows)
    assert [(k["delta"], k["seed"], k["n_sim"]) for k in calls] == [(0.8, 20261022, 100)] * 2   # no power run at all


def test_hb_power_row_notes_a_simulated_gap_more_than_twice_off_the_target(stubbed, monkeypatch):
    """The target is a median over nodes, so it is compared with the simulated median gap, ``mean_t_hat`` (what the
    fixed calibrate_delta matches), not with the mean over nodes."""
    monkeypatch.setattr(stubbed, "calibrate_delta", lambda p, c, d, target, **kw: {"delta": 0.2, "target_gap": target})
    gap = _aq3_gap("LEV6")

    def note(**gaps):
        monkeypatch.setattr(stubbed, "hb_size", _fake_hb([], **gaps))
        return stubbed.hb_power_rows(None, _laws(), stubbed.SimWindows.default(), stubbed.SimCounts())[0] \
            .get("note", "")

    for t_hat, noted in ((gap * 1.5, False), (gap * 2.5, True), (gap * 0.4, True)):
        assert ("factor of 2" in note(t_hat=t_hat, mean_gap=gap)) == noted, (t_hat, gap)
    assert "factor of 2" not in note(t_hat=gap * 1.2, mean_gap=gap * 9)     # mean over nodes: not the target's scale
    assert "factor of 2" in note(t_hat=gap * 3, mean_gap=gap)


def test_fit_params_use_the_out_of_sample_year_only(stubbed, monkeypatch):
    seen = []

    def spy(rows):
        seen.append(rows.copy())
        return len(seen)

    monkeypatch.setattr(stubbed, "fit_node", spy)
    params = stubbed.fit_he_params(None, None, None)
    assert sorted(params) == sorted(NODES) and len(seen) == len(NODES)
    for frame in seen:
        assert frame["settle_date"].min() == dt.date(2025, 7, 1) and frame["settle_date"].max() == dt.date(2026, 6, 30)
        assert {"score", "pnl_open", "settle_hour"} <= set(frame.columns) and frame["pnl_open"].abs().sum() > 0
    seen.clear()
    hb = stubbed.fit_hb_params(None)
    assert sorted(hb) == sorted(NODES)
    assert all((f["pnl_open"] == 0.0).all() and f["settle_date"].min() == dt.date(2025, 7, 1) for f in seen)


def test_fit_hb_laws_draw_each_block_from_its_own_simfit_scores(stubbed, monkeypatch):
    """Ruling 84b item 1: each law's params come from cached_scores(SIMFIT, <block>, "xgb", 42), out-of-sample rows."""
    asked, fitted = [], []
    real = stubbed.cached_scores

    def spy(panel, analysis, block, estimator="xgb", seed=42, **kw):
        asked.append((analysis.id, block, estimator, seed))
        return real(panel, analysis, block, estimator, seed)

    def fit(rows):
        fitted.append(rows)
        return float(rows["score"].mean())

    monkeypatch.setattr(stubbed, "cached_scores", spy)
    monkeypatch.setattr(stubbed, "fit_node", fit)
    laws = stubbed.fit_hb_laws(None)
    assert list(laws) == list(HB_LAWS) and stubbed.HB_LAWS == HB_LAWS
    assert asked == [("SIMFIT", b, "xgb", 42) for b in HB_LAWS]
    assert all(sorted(v) == sorted(NODES) for v in laws.values())
    assert len({laws[law][NODES[0]] for law in HB_LAWS}) == 3                   # three blocks, three different fits
    assert all(f["settle_date"].min() == dt.date(2025, 7, 1) and (f["pnl_open"] == 0.0).all() for f in fitted)


def test_the_h_e_target_is_half_the_loss_weighted_pooled_selc_of_aq3_and_aq4(stubbed):
    """Ruling 94 (M12): spec §6.3's pooled estimand, (excess_A + excess_B) / (L_A + L_B), not the mean of the SELCs."""
    from es.run_analogs import pooled_selc
    parts = stubbed.analog_selc_parts(None, None, None)
    assert len(parts) == 2 and all(len(p) == 2 and p[1] > 0 for p in parts)
    (e3, l3), (e4, l4) = parts
    assert pooled_selc(parts) == pytest.approx((e3 + e4) / (l3 + l4))
    assert pooled_selc(parts) != pytest.approx((e3 / l3 + e4 / l4) / 2)           # the two weights differ on this frame
    per_quarter = stubbed.analog_selc(None, None, None)
    assert per_quarter == pytest.approx([e3 / l3, e4 / l4])
    assert pooled_selc([(1.0, 0.0), (2.0, 0.0)]) == 0.0                           # no loss at all: 0 by convention


def test_analog_selc_is_the_p1_selc_of_aq3_and_aq4(stubbed):
    from es.analyses import ANALOGS
    from es.measures import add_calendar_deciles, selc
    from es.policies import p1_fixed
    from es.positions import attach_pnl
    from es.settlement import attach_settlement
    got = stubbed.analog_selc(None, None, None)
    assert len(got) == 2
    for aid, value in zip(("AQ3", "AQ4"), got):
        pri, drc = stubbed._settled_rows(None, ANALOGS[aid], None, None)
        rows = add_calendar_deciles(attach_pnl(attach_settlement(p1_fixed(pri, 0.05), None, None), "premium", drc),
                                    stub_scores(None, ANALOGS[aid], "G6"))
        assert value == pytest.approx(selc(rows)) and isinstance(value, float)


# ---------------------------------------------------------------------------------------------------------------
# Ruling 94, N-2 of the Task 17 re-review: how each node's persistence correction ended (NodeParams.phi_status), next to
# every size and power row. A clipped-high law's size is not a bound in a known direction.
# ---------------------------------------------------------------------------------------------------------------

PHI_COLUMNS = ["phi_nodes", "phi_corrected", "phi_clipped_high", "phi_clipped_low", "phi_no_signal", "phi_uncorrected"]


def _fit(score="corrected", pnl="corrected"):
    """A NodeParams with the status ``fit_node`` would have recorded for its two components (the counts read nothing
    else)."""
    from es.simulate import NodeParams
    return NodeParams(mu={}, nu={}, phi_s=0.5, sd_s=1.0, sd_eps=1.0, phi_p=0.5, sd_p=1.0, sd_eta=1.0,
                      phi_status={"score": score, "pnl": pnl})


def _block(*statuses):
    """{(market, node): NodeParams}, one node for each (score status, P&L status) given."""
    return {("M", f"n{i}"): _fit(*s) for i, s in enumerate(statuses)}


def _phi_of(row):
    return {c: row[c] for c in PHI_COLUMNS}


def _phi_laws():
    """Three H-B laws of ten nodes each, told apart by their statuses. The H-B fits' P&L is a constant zero, whose
    correction ends in no_signal. DART3 sits at exactly 10% clipped high, LEV6 at 30%, and ROLL_DISP9's P&L is clipped
    high at every node, which H-B never draws."""
    ok, odd = ("corrected", "no_signal"), ("corrected", "clipped_high")
    return {"DART3": _block(*[ok] * 9, ("clipped_high", "no_signal")),
            "LEV6": _block(*[ok] * 6, *[("clipped_high", "no_signal")] * 3, ("clipped_low", "no_signal")),
            "ROLL_DISP9": _block(*[odd] * 9, ("no_signal", "clipped_high"))}


def _phi_he():
    """An H-E fit of ten nodes: one clipped high in its score and another in its P&L, so 20% of the nodes and 10% of
    each component."""
    return _block(("clipped_high", "corrected"), ("corrected", "clipped_high"), *[("corrected", "corrected")] * 7,
                  ("clipped_low", "no_signal"))


EXPECTED_PHI = {"H-E": dict(zip(PHI_COLUMNS, (10, 9, 2, 1, 1, 0))),
                "DART3": dict(zip(PHI_COLUMNS, (10, 9, 1, 0, 0, 0))),
                "LEV6": dict(zip(PHI_COLUMNS, (10, 6, 3, 1, 0, 0))),
                "ROLL_DISP9": dict(zip(PHI_COLUMNS, (10, 9, 0, 0, 1, 0)))}


def _noted(row):
    return "not a bound in a known direction" in str(row.get("note", ""))


def test_phi_status_counts_a_node_under_each_status_of_the_components_its_block_draws():
    from es.run_analogs import HB_COMPONENTS, HE_COMPONENTS, phi_status_counts
    block = _block(("corrected", "corrected"), ("clipped_high", "corrected"), ("corrected", "clipped_high"),
                   ("clipped_low", "no_signal"), ("no_signal", "uncorrected"), ("clipped_high", "clipped_high"))
    both = phi_status_counts(block, HE_COMPONENTS)
    assert _phi_of(both) == dict(zip(PHI_COLUMNS, (6, 3, 3, 1, 2, 1)))     # a node is in every status it has, in either
    assert both["components"] == ["score", "pnl"]
    assert both["by_component"] == {
        "score": dict(corrected=2, clipped_high=2, clipped_low=1, no_signal=1, uncorrected=0),
        "pnl": dict(corrected=2, clipped_high=2, clipped_low=0, no_signal=1, uncorrected=1)}
    score = phi_status_counts(block, HB_COMPONENTS)                            # H-B draws no P&L: only the score counts
    assert _phi_of(score) == dict(zip(PHI_COLUMNS, (6, 2, 2, 1, 1, 0))) and score["components"] == ["score"]
    assert score["by_component"] == {"score": both["by_component"]["score"]}


def test_the_constant_pnl_column_of_an_h_b_fit_is_not_counted_as_a_status():
    """H-B fits are made with pnl_open = 0, so their P&L correction always ends in no_signal: counting it would put
    every H-B node under no_signal."""
    from es.run_analogs import HB_COMPONENTS, HE_COMPONENTS, phi_status_counts
    block = _block(*[("corrected", "no_signal")] * 4)
    assert _phi_of(phi_status_counts(block, HB_COMPONENTS)) == dict(zip(PHI_COLUMNS, (4, 4, 0, 0, 0, 0)))
    assert _phi_of(phi_status_counts(block, HE_COMPONENTS)) == dict(zip(PHI_COLUMNS, (4, 4, 0, 0, 4, 0)))


def test_a_node_without_a_recorded_status_is_uncorrected_and_an_unknown_status_raises():
    from es.run_analogs import HE_COMPONENTS, clipped_high_note, phi_status_counts
    by_hand = {("M", "a"): SimpleNamespace(), ("M", "b"): object(), ("M", "c"): _fit(),
               ("M", "d"): SimpleNamespace(phi_status={})}                      # no correction ran on a hand-built fit
    assert _phi_of(phi_status_counts(by_hand, HE_COMPONENTS)) == dict(zip(PHI_COLUMNS, (4, 1, 0, 0, 0, 3)))
    with pytest.raises(ValueError, match="clipped_sideways"):
        phi_status_counts(_block(("corrected", "clipped_sideways")), HE_COMPONENTS)
    with pytest.raises(TypeError, match="phi_status"):
        phi_status_counts({("M", "a"): SimpleNamespace(phi_status="corrected")}, HE_COMPONENTS)
    empty = phi_status_counts({}, HE_COMPONENTS)
    assert empty["phi_nodes"] == 0 and clipped_high_note(empty) is None        # no node, no share, no note


def test_the_share_that_flags_a_block_is_more_than_a_tenth_of_its_nodes():
    from es.run_analogs import CLIPPED_HIGH_SHARE, HB_COMPONENTS, clipped_high_note, phi_status_counts
    assert CLIPPED_HIGH_SHARE == 0.10
    ok, high = ("corrected", "no_signal"), ("clipped_high", "no_signal")
    for n_high, n_ok, flagged in ((1, 9, False), (2, 18, False), (11, 99, False), (1, 10, False), (0, 5, False),
                                  (2, 9, True), (2, 8, True), (12, 98, True), (3, 0, True)):
        counts = phi_status_counts(_block(*[high] * n_high, *[ok] * n_ok), HB_COMPONENTS)
        note = clipped_high_note(counts)
        assert (note is not None) == flagged, (n_high, n_ok)
        if flagged:
            assert f"{n_high} of {n_high + n_ok} nodes" in note and "not a bound in a known direction" in note


def test_every_size_and_power_row_carries_the_status_counts_of_its_own_law(stubbed, monkeypatch):
    monkeypatch.setattr(stubbed, "he_size_power", _fake_he([]))
    monkeypatch.setattr(stubbed, "hb_size", _fake_hb([]))
    monkeypatch.setattr(stubbed, "calibrate_delta", lambda p, c, d, target, **kw: {"delta": 0.2, "target_gap": target})
    w, c = stubbed.SimWindows.default(), stubbed.SimCounts()
    he_size = stubbed.he_size_rows(_phi_he(), w, c)
    he_power, _ = stubbed.he_power_rows(_phi_he(), w, c, target=0.045)
    hb_size = stubbed.hb_size_rows(_phi_laws(), w, c)
    hb_power = stubbed.hb_power_rows(None, _phi_laws(), w, c)
    assert (len(he_size), len(he_power), len(hb_size), len(hb_power)) == (16, 8, 18, 2)
    for r in he_size + he_power:                                               # H-E: both components of its own fit
        assert _phi_of(r) == EXPECTED_PHI["H-E"], r["test"]
    for r in hb_size:                                                          # H-B size: the score of the row's law
        assert _phi_of(r) == EXPECTED_PHI[r["law"]], (r["law"], r["scenario"], r["persistence"])
    for r in hb_power:                                                         # H-B power: both blocks drawn from DART3
        assert r["law"] == "DART3" and _phi_of(r) == EXPECTED_PHI["DART3"], r["test"]


def test_rows_of_a_block_with_over_a_tenth_of_its_nodes_clipped_high_say_they_are_no_bound(stubbed, monkeypatch):
    monkeypatch.setattr(stubbed, "he_size_power", _fake_he([]))
    monkeypatch.setattr(stubbed, "hb_size", _fake_hb([]))
    w, c = stubbed.SimWindows.default(), stubbed.SimCounts()
    he_size = stubbed.he_size_rows(_phi_he(), w, c)
    he_power, _ = stubbed.he_power_rows(_phi_he(), w, c, target=0.5)           # beyond every grid point: not bracketed
    for r in he_size + he_power:                                               # 2 of 10 nodes: either component counts
        assert "2 of 10 nodes (20%) are clipped high" in r["note"] and "score or pnl" in r["note"], r["test"]
        assert r["note"].count("not a bound in a known direction") == 1
    assert he_power[-1]["note"].startswith("target not bracketed; ")           # appended to the note it already had
    assert all(r["note"].startswith("2 of 10") for r in he_size + he_power[:-1])
    hb_size = stubbed.hb_size_rows(_phi_laws(), w, c)
    for r in hb_size:                                                          # DART3 sits at exactly 10%, not more
        assert _noted(r) == (r["law"] == "LEV6"), (r["law"], r["scenario"], r["persistence"])
    lev6 = [r for r in hb_size if r["law"] == "LEV6"]
    assert len(lev6) == 6 and all("3 of 10 nodes (30%) are clipped high" in r["note"] and "pnl" not in r["note"]
                                  for r in lev6)
    assert not any("note" in r for r in hb_size if r["law"] != "LEV6")


def test_the_power_rows_of_a_flagged_law_say_so_after_the_notes_they_already_have(stubbed, monkeypatch):
    """H-B power draws both blocks from DART3: a note on its rows when DART3 is flagged, appended to the gap note, kept
    beside the lower-bound note, and on an error row (which has no other note) as well."""
    w, c = stubbed.SimWindows.default(), stubbed.SimCounts()
    monkeypatch.setattr(stubbed, "calibrate_delta",
                        lambda p, cal, dep, target, **kw: {"delta": 0.2, "target_gap": target})
    monkeypatch.setattr(stubbed, "hb_size", _fake_hb([]))
    assert not any(_noted(r) for r in stubbed.hb_power_rows(None, _phi_laws(), w, c))       # DART3 at exactly 10%
    flagged = {**_phi_laws(), "DART3": _block(*[("corrected", "no_signal")] * 7, *[("clipped_high", "no_signal")] * 3)}
    expected = dict(zip(PHI_COLUMNS, (10, 7, 3, 0, 0, 0)))
    monkeypatch.setattr(stubbed, "hb_size", _fake_hb([], t_hat=9.0))           # a simulated gap far from the target
    for r in stubbed.hb_power_rows(None, flagged, w, c):
        assert r["note"].startswith("simulated median gap") and "3 of 10 nodes (30%) are clipped high" in r["note"]
        assert _phi_of(r) == expected
    monkeypatch.setattr(stubbed, "calibrate_delta", _calibration_out_of_range)
    monkeypatch.setattr(stubbed, "_aq3_level_gap", lambda panel, block: 0.5)
    monkeypatch.setattr(stubbed, "hb_size", _fake_hb([], t_hat=0.2))           # below the target: a lower bound
    for r in stubbed.hb_power_rows(None, flagged, w, c):
        assert r["power_lower_bound"] is True and _phi_of(r) == expected
        assert "lower bound" in r["note"] and "3 of 10 nodes (30%) are clipped high" in r["note"]
    monkeypatch.setattr(stubbed, "hb_size", _fake_hb([], t_hat=0.9))           # above it: the calibration failed
    for r in stubbed.hb_power_rows(None, flagged, w, c):
        assert "error" in r and "reject_0.0166667" not in r and _phi_of(r) == expected
        assert r["note"].startswith("3 of 10 nodes (30%) are clipped high")


def test_the_status_columns_and_notes_do_not_enter_the_registration():
    """The counts and the note are information next to a row: the rule reads rejection rates and nothing else."""
    from es.run_analogs import registration_family
    plain = _sim()
    marked = [{**r, **dict(zip(PHI_COLUMNS, (10, 0, 10, 0, 0, 0))), "note": "10 of 10 nodes are clipped high"}
              for r in plain]
    assert registration_family(marked, _placebo()) == registration_family(plain, _placebo())


# ---------------------------------------------------------------------------------------------------------------
# The staged runner (Ruling 77): main() and its stages, end to end on the synthetic frame with a few replications
# ---------------------------------------------------------------------------------------------------------------

import json                     # noqa: E402
import math                     # noqa: E402
import os                       # noqa: E402
import shutil                   # noqa: E402
from types import SimpleNamespace   # noqa: E402

FILES = ("activity_all.csv", "eta_selection.csv", "placebo_he.csv", "simulation.csv", "protocol_inputs.json")
TINY_COUNTS = dict(he_size=3, he_rho=2, he_power=3, hb_size=3, hb_power=2, hb_rep=5, hb_calibrate=2)


def _days(start, n):
    return [start + dt.timedelta(days=i) for i in range(n)]


_FITS = {}


def _memo_fit_node(real):
    """``fit_node`` made once per distinct input for the whole module: stage tests refit the same synthetic nodes."""
    def fit(rows):
        key = (len(rows), float(rows["score"].sum()), float(rows["pnl_open"].sum()))
        if key not in _FITS:
            _FITS[key] = real(rows)
        return _FITS[key]
    return fit


def _point(mp, ra, out, seen=None):
    """Point everything main() touches at ``out`` and the synthetic frame, with a few replications on short windows,
    and a quarter of the real number of blocks and estimators (the blocks the runner names must stay: G6, G1_24)."""
    def scores(panel, analysis, block, estimator="xgb", seed=42, **kw):
        if seen is not None:
            seen.append(analysis)
        return stub_scores(panel, analysis, block, estimator, seed)

    import es.simulate as simulate
    # fit_node corrects the fitted persistence by simulation (Ruling 84a): seconds per node, and a stage fits eight.
    # These tests are about plumbing, so the correction runs on a few paths and steps (raising=False: a renamed
    # constant in es.simulate only costs the speed-up), and each distinct fit is made once.
    mp.setattr(simulate, "_CORRECTION_PATHS", 400, raising=False)
    mp.setattr(simulate, "_CORRECTION_STEPS", 2, raising=False)
    mp.setattr(ra, "fit_node", _memo_fit_node(simulate.fit_node))
    mp.setattr(ra, "ANALOG_DIR", out)
    mp.setattr(ra, "cached_scores", scores)
    mp.setattr(ra, "ensure_dirs", lambda: None)
    mp.setattr(ra, "_load_inputs", lambda: (None, None, None))
    mp.setattr(ra, "BLOCKS", {"G6": [], "G1_24": []})
    mp.setattr(ra, "ESTIMATORS", ("xgb",))
    mp.setattr(ra, "SIM_COUNTS", ra.SimCounts(**TINY_COUNTS))
    mp.setattr(ra, "SIM_WINDOWS", ra.SimWindows(_days(dt.date(2025, 7, 1), 14), _days(dt.date(2025, 10, 1), 14),
                                                _days(dt.date(2026, 7, 1), 14), _days(dt.date(2026, 10, 1), 14)))


@pytest.fixture(scope="module")
def stage_one(tmp_path_factory):
    """The output directory of one real stage 'analogs' run on the synthetic frame, made once for the module."""
    import es.run_analogs as ra
    out = tmp_path_factory.mktemp("stage_one") / "analogs"
    mp = pytest.MonkeyPatch()
    try:
        _point(mp, ra, out)
        ra.main(["--stage", "analogs"])
    finally:
        mp.undo()
    return out


@pytest.fixture
def env(tmp_path, monkeypatch):
    """(es.run_analogs, an empty output directory, the analyses the runner asked the score cache for), patched in."""
    import es.run_analogs as ra
    seen = []
    out = tmp_path / "analogs"
    _point(monkeypatch, ra, out, seen)
    return ra, out, seen


def _bytes(directory):
    return {f.name: f.read_bytes() for f in sorted(directory.iterdir())}


def test_stage_analogs_writes_its_files_and_a_provisional_protocol_inputs(stage_one):
    assert sorted(f.name for f in stage_one.iterdir()) == sorted(f for f in FILES if f != "simulation.csv")
    meta = json.loads((stage_one / "protocol_inputs.json").read_text())
    assert meta["stage"] == "analogs" and meta["family"] is None and meta["eta"] in (0.005, 0.01, 0.02, 0.05)
    from es.run_analogs import select_eta
    assert meta["eta"] == select_eta(pd.read_csv(stage_one / "eta_selection.csv"))
    act = pd.read_csv(stage_one / "activity_all.csv")
    assert set(act["policy"]) == {"P1", "P4_56", "P4_28", "P6"} and set(act["analysis"]) >= {"AQ3", "AQ3F"}
    assert len(act) == 5 * 2 * 1 * 4 * len(NODES)
    placebo = pd.read_csv(stage_one / "placebo_he.csv")
    assert list(placebo.columns) == ["pair", "t_obs", "p", "n_null", "selc_obs", "n_rows", "n_settle_approx", "counted",
                                     "note"] and len(placebo) == 10                       # Rulings 64 and 94
    assert placebo["counted"].tolist() == [p.endswith("-1y") for p in placebo["pair"]]
    assert placebo.loc[~placebo["counted"], "note"].eq(UNCOUNTED).all()
    assert placebo.loc[placebo["counted"], "note"].isna().all()
    assert [r["pair"] for r in meta["placebo"]] == placebo["pair"].tolist()


def test_stage_sim_stops_before_loading_anything_when_stage_one_has_not_run(env, monkeypatch):
    ra, out, _ = env
    monkeypatch.setattr(ra, "_load_inputs", lambda: pytest.fail("the inputs were loaded"))
    with pytest.raises(FileNotFoundError, match="--stage analogs"):
        ra.main(["--stage", "sim"])
    assert not out.exists() or list(out.iterdir()) == []


def test_stage_sim_refuses_a_protocol_inputs_it_does_not_know(env):
    ra, out, _ = env
    out.mkdir()
    (out / "placebo_he.csv").write_text("pair,p\nAQ3-1y,0.5\n")
    (out / "protocol_inputs.json").write_text(json.dumps({"eta": 0.01, "stage": "half-way"}))
    with pytest.raises(ValueError, match="half-way"):
        ra.main(["--stage", "sim"])


def test_stage_one_files_read_back_exactly(env):
    """Placebo rows come back from the CSV as the dicts they were: errors without NaN keys, n_null int, p to the bit."""
    ra, out, _ = env
    out.mkdir()
    rows = [{"pair": "AQ3-1y", "t_obs": 0.1 + 0.2, "p": 1 / 3, "n_null": 91, "selc_obs": -0.0045, "n_rows": 4416,
             "n_settle_approx": 2208, "counted": True},
            {"pair": "AQ1-2y", "error": 'rotation_test: window(s) of 1 days, "too short"\nsecond line',
             "counted": False, "note": UNCOUNTED}]
    ra._publish({out / "placebo_he.csv": ra._csv(pd.DataFrame(rows)),
                 out / "protocol_inputs.json": ra._json({"eta": 0.02, "family": None, "stage": "analogs"})})
    eta, placebo = ra._read_stage_one()
    assert eta == 0.02 and placebo == rows
    assert isinstance(placebo[0]["n_null"], int) and isinstance(placebo[0]["p"], float)
    assert isinstance(placebo[0]["n_settle_approx"], int) and isinstance(placebo[0]["n_rows"], int)
    assert placebo[0]["counted"] is True and placebo[1]["counted"] is False


def test_stage_sim_completes_the_protocol_inputs_and_writes_the_size_matrix(env, stage_one):
    ra, out, _ = env
    shutil.copytree(stage_one, out)
    before = _bytes(out)
    ra.main(["--stage", "sim"])
    after = _bytes(out)
    assert sorted(after) == sorted(FILES)
    assert all(after[f] == before[f] for f in ("activity_all.csv", "eta_selection.csv", "placebo_he.csv"))
    sim = pd.read_csv(out / "simulation.csv")
    got = sim[sim["test"].str.endswith("size")]
    he = [(t, p, s, st) for t, p, s, st in zip(got["test"], got["persistence"], got["scenario"], got["strata"])
          if t.startswith("H-E")]
    assert he == EXPECTED_HE_SIZE
    hb = [(t, law, p, sc) for t, law, p, sc in zip(got["test"], got["law"], got["persistence"], got["scenario"])
          if t.startswith("H-B")]
    assert hb == EXPECTED_HB_SIZE and got[got["test"] == "H-B size"]["strata"].isna().all()
    assert sim["test"].tolist()[16:24] == ["H-E pooled A+B, power curve"] * 7 + ["H-E pooled A+B, power"]
    assert set(sim["test"]) >= {"H-B1 power", "H-B2 power"} and len(sim) == 16 + 7 + 1 + 18 + 2
    curve = sim[sim["test"] == "H-E pooled A+B, power curve"]
    assert curve["rho"].tolist() == list(ra.RHO_GRID) and (curve["n_sim"] == 2).all()    # TINY he_rho
    assert curve[["reject_0.0166667", "reject_0.05", "mean_selc"]].notna().all().all()
    assert (sim[sim["test"].str.startswith("H-E") & sim["test"].str.endswith("size")]["n_sim"] == 3).all()
    assert (sim[sim["test"] == "H-B size"]["n_sim"] == 3).all()
    for col in ("reject_0.0166667", "reject_0.05", "persistence", "scenario", "n_sim"):
        assert col in sim.columns
    # the protocol inputs
    text = (out / "protocol_inputs.json").read_text()
    meta = json.loads(text, parse_constant=lambda c: pytest.fail(f"{c} in protocol_inputs.json (not strict JSON)"))
    assert meta["stage"] == "complete" and set(meta["family"]) == {"H-E", "H-B1", "H-B2", "notes"}
    assert "retained as registered" in " ".join(meta["family"]["notes"]) and "≤ 0.0167" in meta["family"]["notes"][0]
    assert meta["eta"] == json.loads(before["protocol_inputs.json"])["eta"]
    assert meta["rho_star"] in ra.RHO_GRID and len(meta["analog_selc"]) == 2
    # M12: the target is half the loss-weighted pooled SELC of AQ3 and AQ4
    assert meta["analog_selc_pooled"] == pytest.approx(ra.pooled_selc(ra.analog_selc_parts(None, None, None)))
    assert sim[sim["test"] == "H-E pooled A+B, power"]["target_selc"].iloc[0] == pytest.approx(
        0.5 * meta["analog_selc_pooled"])
    # I2: a seed of its own for each size row; the other rows record their documented seeds
    assert sim[sim["test"].str.endswith("size")]["seed"].tolist() == list(range(20261022, 20261022 + 34))
    assert sim[sim["test"] == "H-E pooled A+B, power curve"]["seed"].eq(7).all()
    assert sim[sim["test"] == "H-E pooled A+B, power"]["seed"].eq(11).all()
    assert sim[sim["test"].isin(["H-B1 power", "H-B2 power"])]["seed"].eq(13).all()
    # M10: every rule constant and seed, as used in this run
    rule = meta["rule"]
    assert (rule["factor"], rule["binomial_level"], rule["approximate_factor"], rule["min_power"]) == \
        (1.5, 0.05, 1.0, 0.5)
    assert rule["levels"] == pytest.approx([0.05 / 3, 0.05]) and rule["k_rows"] == {"H-E": 4, "H-B1": 12, "H-B2": 12}
    assert rule["n_rep"] == 5 and rule["n_sim"] == {k: v for k, v in TINY_COUNTS.items() if k != "hb_rep"}
    assert rule["seeds"] == {"size_base": 20261022, "he_size_first_index": 0, "hb_size_first_index": 16,
                             "rho_search": 7, "he_power": 11, "hb_power": 13, "calibration": 20261022}
    assert rule["calibration_grid"] == list(ra.CALIBRATION_GRID)
    assert rule["counted_pairings"] == list(ra.COUNTED_PAIRINGS)
    assert len(meta["simulation"]) == len(sim) and len(meta["placebo"]) == 10
    for hid in ("H-E", "H-B1", "H-B2"):
        fam = meta["family"][hid]
        assert isinstance(fam["confirmatory"], bool) and isinstance(fam["reasons"], list)
    assert meta["family"]["H-E"]["interim_approximate"] in (True, False)
    # I-3: each block's fitted persistence, in the rows of its law and once per block in the protocol inputs
    from dataclasses import fields
    from es.simulate import NodeParams
    has_raw = bool({"phi_s_raw", "phi_raw"} & {f.name for f in fields(NodeParams)})
    assert set(meta["phi"]) == set(HB_LAWS)
    for law in HB_LAWS:
        rows = sim[sim["law"] == law]
        assert len(rows) >= 6 and rows["phi_score_median"].nunique() == 1
        assert rows["phi_score_median"].iloc[0] == pytest.approx(meta["phi"][law]["phi_score_median"])
        assert -1 < meta["phi"][law]["phi_score_median"] < 1
        raw = meta["phi"][law]["phi_score_raw_median"]
        assert raw is None or -1 < raw < 1
        if has_raw:                                                              # reported next to the corrected phi
            assert raw is not None and rows["phi_score_raw_median"].iloc[0] == pytest.approx(raw)
        else:                                                                    # NodeParams has no raw phi (yet): NaN
            assert raw is None and rows["phi_score_raw_median"].isna().all()
    assert sim[sim["test"].isin(["H-B1 power", "H-B2 power"])]["law"].tolist() == ["DART3", "DART3"]
    # N-2 (Task 17 re-review): how each block's persistence corrections ended, in its rows and once per block here
    assert set(meta["phi_status"]) == {"H-E", *HB_LAWS} and meta["rule"]["clipped_high_share"] == 0.10
    blocks = {"H-E": sim[sim["test"].str.startswith("H-E")], **{law: sim[sim["law"] == law] for law in HB_LAWS}}
    assert [len(rows) for rows in blocks.values()] == [24, 8, 6, 6]            # DART3's include the two power rows
    for block, rows in blocks.items():
        record = meta["phi_status"][block]
        assert record["components"] == (["score", "pnl"] if block == "H-E" else ["score"])
        assert record["phi_nodes"] == len(NODES)
        for column in PHI_COLUMNS:
            assert rows[column].eq(record[column]).all(), (block, column)
        assert all(sum(v.values()) == len(NODES) for v in record["by_component"].values())
        assert record["note"] is None or "not a bound in a known direction" in record["note"]
        said = (rows["note"].fillna("").str.contains("not a bound in a known direction") if "note" in rows
                else pd.Series(False, index=rows.index))
        assert said.eq(record["note"] is not None).all(), block              # every row of a flagged block, none else
    power = sim[sim["test"] == "H-E pooled A+B, power"].iloc[0]
    assert power["rho"] == meta["rho_star"] and power["scenario"] == "homoskedastic"
    # the family is registration_family of exactly these rows, as the JSON holds them
    assert ra._jsonable(ra.registration_family(meta["simulation"], meta["placebo"])) == meta["family"]
    assert [f for f in out.iterdir() if f.name.endswith(".tmp")] == []


def test_a_run_notes_every_flagged_block_in_its_rows_and_in_the_protocol_inputs(env, stage_one, monkeypatch):
    """Every fit marked clipped high in its score (and corrected in its P&L): all 44 rows say they are no bound, and the
    four blocks' records in protocol_inputs.json carry the note and the counts, which are numbers the rows repeat."""
    import dataclasses
    ra, out, _ = env
    shutil.copytree(stage_one, out)
    real = ra.fit_node

    def clipped(rows):
        return dataclasses.replace(real(rows), phi_status={"score": "clipped_high", "pnl": "corrected"})

    monkeypatch.setattr(ra, "fit_node", clipped)
    ra.main(["--stage", "sim"])
    sim = pd.read_csv(out / "simulation.csv")
    meta = json.loads((out / "protocol_inputs.json").read_text())
    assert len(sim) == 44 and sim["phi_clipped_high"].eq(len(NODES)).all() and sim["phi_nodes"].eq(len(NODES)).all()
    assert sim["note"].str.contains(f"{len(NODES)} of {len(NODES)} nodes (100%) are clipped high", regex=False).all()
    assert sim["note"].str.contains("not a bound in a known direction").all()
    he = sim["test"].str.startswith("H-E")
    assert sim.loc[he, "phi_corrected"].eq(len(NODES)).all()                   # the P&L of H-E is corrected: counted
    assert sim.loc[~he, "phi_corrected"].eq(0).all()                           # H-B draws the score alone: not counted
    assert set(meta["phi_status"]) == {"H-E", *HB_LAWS}
    for block, record in meta["phi_status"].items():
        assert record["phi_clipped_high"] == len(NODES) and "not a bound in a known direction" in record["note"]
        assert record["by_component"]["score"]["clipped_high"] == len(NODES)
    assert meta["phi_status"]["H-E"]["by_component"]["pnl"]["corrected"] == len(NODES)
    assert "pnl" not in meta["phi_status"]["LEV6"]["by_component"]               # H-B draws the score alone
    # the note is information: the registered family is the one the rows alone give
    assert ra._jsonable(ra.registration_family(meta["simulation"], meta["placebo"])) == meta["family"]


def test_two_stages_equal_the_one_shot_run(env, stage_one, monkeypatch, tmp_path):
    """Ruling 77: --stage analogs then --stage sim leaves the same five files as no --stage, byte for byte (the files
    hold no timing field), and no analysis past the seal reaches the score cache."""
    ra, one_shot, seen = env
    ra.main([])
    assert sorted(f.name for f in one_shot.iterdir()) == sorted(FILES)
    two_stage = tmp_path / "two_stage"
    shutil.copytree(stage_one, two_stage)
    monkeypatch.setattr(ra, "ANALOG_DIR", two_stage)
    ra.main(["--stage", "sim"])
    a, b = _bytes(one_shot), _bytes(two_stage)
    assert a.keys() == b.keys()
    for name in FILES:
        assert a[name] == b[name], name
    assert seen and max(x.deploy.end for x in seen) <= "2026-06-30"             # development data only
    assert {x.id for x in seen} >= {"AQ3", "AQ4", "AQ1", "AQ2", "AQ3F", "SIMFIT"}


def test_a_stage_that_fails_writes_nothing(env, stage_one, monkeypatch):
    ra, out, _ = env
    monkeypatch.setattr(ra, "activity_table", lambda panel: pd.DataFrame({"analysis": ["AQ3"]}))
    monkeypatch.setattr(ra, "eta_table", lambda panel: pd.DataFrame({"analysis": ["AQ3"], "eta": [0.01],
                                                                     "median_log_error": [0.2]}))

    def boom(panel, fmm, nyi):
        raise RuntimeError("placebo exploded")

    monkeypatch.setattr(ra, "placebo_table", boom)
    with pytest.raises(RuntimeError, match="placebo exploded"):
        ra.main(["--stage", "analogs"])
    assert not out.exists() or list(out.iterdir()) == []                        # two tables were computed; none written


def test_a_failed_sim_stage_leaves_the_files_of_stage_one_untouched(env, stage_one, monkeypatch):
    ra, out, _ = env
    shutil.copytree(stage_one, out)
    before = _bytes(out)
    real, calls = ra.he_size_power, []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 5:
            raise RuntimeError("simulation exploded")
        return real(*args, **kwargs)

    monkeypatch.setattr(ra, "he_size_power", flaky)
    with pytest.raises(RuntimeError, match="simulation exploded"):
        ra.main(["--stage", "sim"])
    assert _bytes(out) == before                                                # no simulation.csv, same JSON, no .tmp


# ---------------------------------------------------------------------------------------------------------------
# Ruling 94: every counted pairing needs a p (I1), and no state may pair new placebo rows or a new simulation with an
# old complete family (M6)
# ---------------------------------------------------------------------------------------------------------------

def test_the_counted_pairings_all_need_a_finite_p():
    """I1: the registered check rests on the four 1-year quarters and the pooled AQ3+AQ4-1y; one of them without a p
    would silently remove a check (or drop the pooled one to nothing), so the stage refuses. 2-year ones may fail."""
    from es.run_analogs import COUNTED_PAIRINGS, _require_counted_placebo
    assert COUNTED_PAIRINGS == ("AQ3-1y", "AQ4-1y", "AQ1-1y", "AQ2-1y", "AQ3+AQ4-1y")
    _require_counted_placebo(_placebo())
    two_year_failed = _placebo()
    two_year_failed[5] = {"pair": "AQ4-2y", "error": "x", "counted": False, "note": UNCOUNTED}
    two_year_failed[9] = {"pair": "AQ3+AQ4-2y", "error": "y", "counted": False, "note": UNCOUNTED}
    _require_counted_placebo(two_year_failed)
    broken = _placebo()
    broken[1] = {"pair": "AQ4-1y", "error": "every row needs a calendar-model score", "counted": True}
    with pytest.raises(ValueError, match=r"AQ4-1y.*calendar-model score"):
        _require_counted_placebo(broken)
    not_a_number = _placebo()
    not_a_number[8] = {"pair": "AQ3+AQ4-1y", "p": float("nan"), "counted": True}
    with pytest.raises(ValueError, match=r"AQ3\+AQ4-1y"):
        _require_counted_placebo(not_a_number)
    with pytest.raises(ValueError, match="AQ2-1y"):
        _require_counted_placebo([r for r in _placebo() if r["pair"] != "AQ2-1y"])


def _stage_one_files(ra, out, rows=None, stage="analogs", family=None, simulation=None):
    """A stage-1 directory (and, if given, a simulation.csv), written without running anything."""
    out.mkdir(parents=True, exist_ok=True)
    rows = _placebo() if rows is None else rows
    files = {out / "activity_all.csv": "analysis\nAQ3\n",
             out / "eta_selection.csv": "analysis,eta,median_log_error\nAQ3,0.01,0.2\n",
             out / "placebo_he.csv": ra._csv(pd.DataFrame(rows))}
    if simulation is not None:
        files[out / "simulation.csv"] = simulation
    files[out / "protocol_inputs.json"] = ra._json({"eta": 0.01, "family": family, "stage": stage, "placebo": rows})
    ra._publish(files)


def _complete_dir(ra, out):
    """The files of an earlier complete run, with a family of its own."""
    _stage_one_files(ra, out, stage="complete", family={"H-E": {"confirmatory": True}}, simulation="test\nold\n")


def _fake_stage_one(monkeypatch, ra):
    monkeypatch.setattr(ra, "activity_table", lambda panel: pd.DataFrame({"analysis": ["AQ3"]}))
    monkeypatch.setattr(ra, "eta_table", lambda panel: pd.DataFrame({"analysis": ["AQ3", "AQ3"], "eta": [0.01, 0.02],
                                                                     "median_log_error": [0.2, 0.3]}))
    monkeypatch.setattr(ra, "placebo_table", lambda panel, fmm, nyi: _placebo())


def _fake_stage_two(monkeypatch, ra):
    """The pieces of stage 2 replaced by canned rows, so that its publishing can be tested in a moment."""
    sim = _sim()
    curve = [{"test": "H-E pooled A+B, power curve", "rho": 0.1, "n_sim": 2, "reject_0.0166667": 0.1,
              "reject_0.05": 0.2, "mean_selc": 0.01, "seed": 7}]
    target = {"test": "H-E pooled A+B, power", "rho": 0.1, "n_sim": 2, "reject_0.0166667": 0.1, "reject_0.05": 0.2,
              "target_selc": 0.0, "seed": 11}
    monkeypatch.setattr(ra, "analog_selc_parts", lambda panel, fmm, nyi: [(0.01, 5.0), (0.02, 4.0)])
    monkeypatch.setattr(ra, "fit_he_params", lambda panel, fmm, nyi: {("M", "n"): SimpleNamespace(phi_s=0.5)})
    monkeypatch.setattr(ra, "fit_hb_laws", lambda panel: _laws())
    monkeypatch.setattr(ra, "he_size_rows", lambda params, windows, counts: [
        r for r in sim if r["test"].endswith("size") and r["test"].startswith("H-E")])
    monkeypatch.setattr(ra, "he_power_rows", lambda params, windows, counts, target_selc: (curve + [target], 0.1))
    monkeypatch.setattr(ra, "hb_size_rows", lambda laws, windows, counts: [r for r in sim if r["test"] == "H-B size"])
    monkeypatch.setattr(ra, "hb_power_rows", lambda panel, laws, windows, counts: [r for r in sim if "H-B" in r["test"]
                                                                                    and "power" in r["test"]])


def _kill_on_replace(monkeypatch, ra, kill_on):
    """os.replace fails on its ``kill_on``-th call, as a process killed between two of the stage's replaces."""
    real, calls = os.replace, []

    def killed(src, dst):
        calls.append(str(dst))
        if len(calls) == kill_on:
            raise OSError("killed between two replaces")
        real(src, dst)

    monkeypatch.setattr(ra.os, "replace", killed)
    return calls


def test_stage_sim_stops_before_loading_when_a_counted_pairing_has_no_p(env, monkeypatch):
    ra, out, _ = env
    rows = _placebo()
    rows[3] = {"pair": "AQ2-1y", "error": "RowCube.from_rows: NaN in ['pnl_open']", "counted": True}
    _stage_one_files(ra, out, rows)
    monkeypatch.setattr(ra, "_load_inputs", lambda: pytest.fail("the inputs were loaded"))
    with pytest.raises(ValueError, match=r"AQ2-1y.*pnl_open"):
        ra.main(["--stage", "sim"])
    assert not (out / "simulation.csv").exists()


def test_run_sim_stage_itself_refuses_a_counted_pairing_without_a_p(env, monkeypatch):
    """I1 again, one level down: run_sim_stage (not only main's early check) raises before any simulation is made."""
    ra, out, _ = env
    rows = _placebo()
    rows[8] = {"pair": "AQ3+AQ4-1y", "error": "no cube for AQ4-1y", "counted": True}
    _stage_one_files(ra, out, rows)
    monkeypatch.setattr(ra, "analog_selc_parts", lambda *a: pytest.fail("the study started"))
    with pytest.raises(ValueError, match=r"AQ3\+AQ4-1y.*no cube for AQ4-1y"):
        ra.run_sim_stage(None, None, None)


def test_stage_one_removes_the_stale_simulation_and_the_old_complete_marker(env, monkeypatch):
    """M6: re-running stage analogs leaves no simulation.csv from an earlier run beside new placebo rows, and the marker
    says 'analogs' again."""
    ra, out, _ = env
    _complete_dir(ra, out)
    _fake_stage_one(monkeypatch, ra)
    ra.main(["--stage", "analogs"])
    assert not (out / "simulation.csv").exists()
    meta = json.loads((out / "protocol_inputs.json").read_text())
    assert meta["stage"] == "analogs" and meta["family"] is None and not list(out.glob("*.tmp"))


@pytest.mark.parametrize("kill_on", [1, 2, 3, 4])
def test_a_kill_during_stage_one_never_pairs_new_placebo_rows_with_an_old_family(env, monkeypatch, kill_on):
    """M6: whichever replace the process dies at, protocol_inputs.json is either absent (stage sim then refuses to
    start) or not 'complete', and the old simulation.csv is gone."""
    ra, out, _ = env
    _complete_dir(ra, out)
    _fake_stage_one(monkeypatch, ra)
    _kill_on_replace(monkeypatch, ra, kill_on)
    with pytest.raises(OSError, match="killed"):
        ra.main(["--stage", "analogs"])
    marker = out / "protocol_inputs.json"
    assert not marker.exists() or json.loads(marker.read_text())["stage"] != "complete"
    assert not (out / "simulation.csv").exists()


@pytest.mark.parametrize("kill_on", [2, 3])
def test_a_kill_during_the_sim_publish_never_pairs_a_new_simulation_with_an_old_family(env, monkeypatch, kill_on):
    """M6: the old family is withdrawn (the marker goes back to stage 1's form, keeping eta and the placebo rows) before
    the new simulation.csv goes in, so a kill leaves a state stage sim can simply be run again from."""
    ra, out, _ = env
    _complete_dir(ra, out)
    _fake_stage_two(monkeypatch, ra)
    _kill_on_replace(monkeypatch, ra, kill_on)
    with pytest.raises(OSError, match="killed"):
        ra.main(["--stage", "sim"])
    meta = json.loads((out / "protocol_inputs.json").read_text())
    assert meta["stage"] == "analogs" and meta["family"] is None
    assert meta["eta"] == 0.01 and len(meta["placebo"]) == 10
    assert (out / "simulation.csv").exists() == (kill_on == 3)                   # only the third replace is after it
    if kill_on == 3:
        assert (out / "simulation.csv").read_text() != "test\nold\n"


def test_a_sim_stage_without_a_kill_leaves_one_complete_state(env, monkeypatch):
    ra, out, _ = env
    _complete_dir(ra, out)
    _fake_stage_two(monkeypatch, ra)
    ra.main(["--stage", "sim"])
    meta = json.loads((out / "protocol_inputs.json").read_text())
    assert meta["stage"] == "complete" and set(meta["family"]) == {"H-E", "H-B1", "H-B2", "notes"}
    assert pd.read_csv(out / "simulation.csv")["test"].tolist()[:2] == ["H-E pooled A+B, size", "H-E single A, size"]
    assert not list(out.glob("*.tmp"))


def test_publish_removes_stale_files_only_after_everything_is_staged(tmp_path):
    import es.run_analogs as ra
    (tmp_path / "old.csv").write_text("old")
    with pytest.raises(OSError):
        ra._publish({tmp_path / "new.csv": "x", tmp_path / "missing_dir" / "b.json": "y"},
                    stale=(tmp_path / "old.csv",))
    assert (tmp_path / "old.csv").read_text() == "old" and not (tmp_path / "new.csv").exists()   # nothing changed
    ra._publish({tmp_path / "new.csv": "x\n"}, stale=(tmp_path / "old.csv", tmp_path / "never_existed.csv"))
    assert not (tmp_path / "old.csv").exists() and (tmp_path / "new.csv").read_text() == "x\n"


def test_publish_moves_files_into_place_only_once_all_are_staged(tmp_path):
    import es.run_analogs as ra
    (tmp_path / "a.csv").write_text("old")
    with pytest.raises(OSError):
        ra._publish({tmp_path / "a.csv": "new", tmp_path / "missing_dir" / "b.json": "{}"})
    assert (tmp_path / "a.csv").read_text() == "old" and not list(tmp_path.glob("*.tmp"))
    ra._publish({tmp_path / "a.csv": "new\n", tmp_path / "b.json": "{}\n"})
    assert (tmp_path / "a.csv").read_text() == "new\n" and (tmp_path / "b.json").read_text() == "{}\n"
    assert not list(tmp_path.glob("*.tmp"))


def test_json_is_strict_and_free_of_numpy_types():
    import es.run_analogs as ra
    text = ra._json({"a": np.float64(1.5), "b": np.int64(3), "c": float("nan"), "d": [np.bool_(True), float("inf")],
                     "e": dt.date(2026, 6, 30), "f": {"x": None}})
    assert json.loads(text) == {"a": 1.5, "b": 3, "c": None, "d": [True, None], "e": "2026-06-30", "f": {"x": None}}
    assert "NaN" not in text and "Infinity" not in text


def test_main_takes_the_two_stage_names_and_nothing_else():
    import es.run_analogs as ra
    with pytest.raises(SystemExit):
        ra.main(["--stage", "everything"])
    assert ra.STAGES == ("analogs", "sim")


def test_the_inputs_come_from_the_development_loaders_only(monkeypatch):
    import es.run_analogs as ra
    calls = []
    monkeypatch.setattr(ra, "load_caiso_fmm", lambda *a, **k: calls.append(("fmm", a, k)) or "FMM")
    monkeypatch.setattr(ra, "load_nyiso_integrated", lambda *a, **k: calls.append(("nyi", a, k)) or "NYI")
    monkeypatch.setattr(ra, "load_dev_raw", lambda *a, **k: calls.append(("raw", a, k)) or "RAW")
    monkeypatch.setattr(ra, "build_panel", lambda raw: calls.append(("panel", (raw,), {})) or "PANEL")
    assert ra._load_inputs() == ("PANEL", "FMM", "NYI")
    kinds = {c[0]: c for c in calls}
    assert kinds["fmm"][1][0].name == "caiso_fmm_dev" and kinds["nyi"][1][0].name == "nyiso_rtint_dev"
    assert all(not k for _, _, k in calls)                                      # no dev_only=False, no dirs overridden
    assert kinds["raw"][1] == () and kinds["panel"][1] == ("RAW",)
