import datetime as dt
import zlib

import numpy as np
import pandas as pd
import pytest


def test_audit_variants_are_the_registered_eighteen():
    from es.secondary import AUDIT_VARIANTS, HB_PAIRS
    assert len(HB_PAIRS) * len(AUDIT_VARIANTS) == 18
    assert {v[0] for v in AUDIT_VARIANTS} == {"logit", "frozen", "alpha=0.01", "alpha=0.025", "alpha=0.1",
                                              "seed=0", "seed=1", "seed=7", "seed=13"}


def test_economic_family_takes_p1_rows_only():
    from es.secondary import economic_family
    rows = [{"policy": "P1", "comparator": "none", "position": "inc", "fees": "0.25", "SELC": 0.1, "rotation_p": 0.2},
            {"policy": "P4", "comparator": "none", "position": "inc", "fees": "0.25", "SELC": 0.1, "rotation_p": 0.2},
            {"policy": "comparator", "comparator": "VOL@count", "position": "inc", "fees": "0.25", "SELC": 0.0,
             "rotation_p": 0.9}]
    out = economic_family(rows)
    assert len(out) == 1 and out[0]["test"] == "P1 inc fees=0.25"


def test_bh_is_applied_within_each_family():
    from es.secondary import bh_table
    rows = ([{"family": "a", "test": str(i), "estimate": 0.0, "p": p} for i, p in enumerate([0.001, 0.02, 0.04, 0.3])]
            + [{"family": "b", "test": "x", "estimate": 0.0, "p": 0.03}, {"family": "b", "test": "y", "estimate": 0.0,
                                                                        "p": np.nan}])
    t = bh_table(rows)
    assert t.loc[t.family == "a", "bh_reject"].tolist() == [True, True, False, False]
    assert t.loc[t.family == "b", "bh_reject"].tolist() == [True, False]


# --------------------------------------------------------------------------------------------------
# Above: the brief's three tests, verbatim. Below: the additions of controller decisions 1 and 3. All data
# are synthetic; nothing is read from outputs/ and no score cache is touched.
# --------------------------------------------------------------------------------------------------

#: The 18 registered audit tests, spelled out (spec v3 §5.2 and the module docstring), not built from AUDIT_VARIANTS.
_REGISTERED_AUDIT_TESTS = [f"{hid} {name}" for hid in ("H-B1", "H-B2")
                           for name in ("logit", "frozen", "alpha=0.01", "alpha=0.025", "alpha=0.1",
                                        "seed=0", "seed=1", "seed=7", "seed=13")]

#: Deployment-score shift of each block in the synthetic scores: LEV6 drifts, DART3 and ROLL_DISP9 do not.
_SHIFT = {"LEV6": 2.0, "ROLL_DISP9": 0.0, "DART3": 0.0}


def _synthetic_score(kind, block, estimator, seed):
    """A score frame of the shape es.bootstrap reads (market, node, split, settle_date, score): 3 nodes, 20
    calibration and 20 deployment days of 12 hours. Deterministic in its arguments and different for different ones."""
    rng = np.random.default_rng(zlib.crc32(repr((kind, block, estimator, seed)).encode()))
    rows = []
    for node in ("A", "B", "C"):
        for split, start, mu in (("calib", dt.date(2024, 7, 1), 0.0), ("deploy", dt.date(2025, 7, 1), _SHIFT[block])):
            for d in range(20):
                for _ in range(12):
                    rows.append({"market": "M", "node": node, "split": split,
                                 "settle_date": start + dt.timedelta(days=d), "score": rng.normal(mu, 1.0)})
    return pd.DataFrame(rows)


def test_the_registered_audit_settings_are_pinned():
    from es.secondary import AUDIT_VARIANTS, HB_PAIRS
    assert HB_PAIRS == (("H-B1", "LEV6"), ("H-B2", "ROLL_DISP9"))
    # (name, kind, estimator, score seed, cutoff alpha); the labels stay at 5% in every variant
    assert AUDIT_VARIANTS == (("logit", "roll", "logit", 42, 0.05), ("frozen", "frozen", "xgb", 42, 0.05),
                              ("alpha=0.01", "roll", "xgb", 42, 0.01), ("alpha=0.025", "roll", "xgb", 42, 0.025),
                              ("alpha=0.1", "roll", "xgb", 42, 0.10), ("seed=0", "roll", "xgb", 0, 0.05),
                              ("seed=1", "roll", "xgb", 1, 0.05), ("seed=7", "roll", "xgb", 7, 0.05),
                              ("seed=13", "roll", "xgb", 13, 0.05))


def test_the_docstring_that_is_registered_states_what_the_constants_do():
    import es.secondary as sec
    doc = sec.__doc__
    for phrase in ("Audit robustness", "Economic robustness", "Drift monitors", "18 one-sided",
                   "cutoff alpha 1%, 2.5% and 10% (labels stay at 5%)", "seeds 0, 1, 7 and 13",
                   "the logistic estimator", "the frozen-origin arm", "H-B1 (LEV6 vs DART3)",
                   "H-B2 (ROLL_DISP9 vs DART3)", "Each family is adjusted on its own, at q = 0.05"):
        assert phrase in doc, phrase
    assert sorted({v[4] for v in sec.AUDIT_VARIANTS} - {0.05}) == [0.01, 0.025, 0.10]
    assert sorted({v[3] for v in sec.AUDIT_VARIANTS} - {42}) == [0, 1, 7, 13]
    assert sec.bh_table.__defaults__ == (0.05,)


def test_audit_family_returns_the_eighteen_registered_tests():
    from es.secondary import audit_family
    asked = []

    def score(kind, block, estimator, seed):
        asked.append((kind, block, estimator, seed))
        return _synthetic_score(kind, block, estimator, seed)

    rows = audit_family(score, n_rep=50)
    assert len(rows) == 18
    assert [r["test"] for r in rows] == _REGISTERED_AUDIT_TESTS
    assert all(set(r) == {"family", "test", "estimate", "p"} for r in rows)
    assert {r["family"] for r in rows} == {"audit robustness"}
    assert all(0.0 <= r["p"] <= 1.0 and np.isfinite(r["estimate"]) for r in rows)
    assert all(abs(r["p"] * 51 - round(r["p"] * 51)) < 1e-9 for r in rows)             # (1 + k) / (n_rep + 1)
    # the frames asked for: LEV6 / ROLL_DISP9 and DART3 under each pipeline, 2 per test and 3 x 7 distinct
    pipelines = {("roll", "logit", 42), ("frozen", "xgb", 42), ("roll", "xgb", 42), ("roll", "xgb", 0),
                 ("roll", "xgb", 1), ("roll", "xgb", 7), ("roll", "xgb", 13)}
    assert len(asked) == 36
    assert set(asked) == {(k, b, e, s) for (k, e, s) in pipelines for b in ("LEV6", "ROLL_DISP9", "DART3")}


def test_audit_family_estimates_alt_minus_base_for_the_right_hypothesis():
    from es.secondary import audit_family
    rows = audit_family(_synthetic_score, n_rep=50)
    h_b1 = [r for r in rows if r["test"].startswith("H-B1")]
    h_b2 = [r for r in rows if r["test"].startswith("H-B2")]
    # LEV6 drifts and DART3 does not: alt - base is clearly positive and, with 50 replicates, at its smallest p of 1 / 51
    assert all(r["estimate"] > 0.5 and r["p"] <= 0.05 for r in h_b1)
    # ROLL_DISP9 behaves like DART3: no consistent gap
    assert max(r["estimate"] for r in h_b2) < min(r["estimate"] for r in h_b1)
    assert np.mean([r["p"] for r in h_b2]) > 0.2


def test_audit_family_hands_the_bootstrap_alt_then_base_with_alpha_and_n_rep_only(monkeypatch):
    import es.secondary as sec
    seen = []

    def fake_bootstrap(alt, base, **kwargs):
        seen.append((alt, base, kwargs))
        return {"t_hat": 0.25 + len(seen), "p": 0.5}

    monkeypatch.setattr(sec, "block_contrast_bootstrap", fake_bootstrap)
    rows = sec.audit_family(lambda kind, block, estimator, seed: (kind, block, estimator, seed), n_rep=7)
    assert len(seen) == len(rows) == 18
    alphas = [0.05, 0.05, 0.01, 0.025, 0.10, 0.05, 0.05, 0.05, 0.05]
    pipelines = [("roll", "logit", 42), ("frozen", "xgb", 42), ("roll", "xgb", 42), ("roll", "xgb", 42),
                 ("roll", "xgb", 42), ("roll", "xgb", 0), ("roll", "xgb", 1), ("roll", "xgb", 7), ("roll", "xgb", 13)]
    for i, (alt, base, kwargs) in enumerate(seen):
        alt_block = ("LEV6", "ROLL_DISP9")[i // 9]
        kind, est, seed = pipelines[i % 9]
        assert alt == (kind, alt_block, est, seed) and base == (kind, "DART3", est, seed)     # alt first, base second
        # the bootstrap's own seed and mean block length stay at its registered defaults (20261022 and 7)
        assert kwargs == {"alpha": alphas[i % 9], "n_rep": 7}
        assert rows[i]["estimate"] == 0.25 + (i + 1) and rows[i]["p"] == 0.5
    seen.clear()
    sec.audit_family(lambda kind, block, estimator, seed: (kind, block, estimator, seed))
    assert {kw["n_rep"] for _, _, kw in seen} == {2000}                                         # the registered B


def test_audit_family_takes_the_bootstrap_it_is_given(monkeypatch):
    """Controller Ruling 109, M1: the sealed run hands in its finite-replicate bootstrap. It is called as the default is
    (alt, base, alpha=, n_rep=), once per test and in the same order; its t_hat and p are the row's, whatever else its
    result holds; and the default is not called at all."""
    import es.secondary as sec
    seen = []

    def mine(alt, base, **kwargs):
        seen.append((alt, base, kwargs))
        return {"t_hat": float(len(seen)), "p": 0.25, "n_nan_replicates": 3, "ci": (0.0, 1.0)}

    def default(*_a, **_k):
        raise AssertionError("the default bootstrap was called although another one was given")

    monkeypatch.setattr(sec, "block_contrast_bootstrap", default)
    rows = sec.audit_family(lambda kind, block, estimator, seed: (kind, block, estimator, seed), n_rep=9, contrast=mine)
    assert len(rows) == len(seen) == 18 and [r["test"] for r in rows] == _REGISTERED_AUDIT_TESTS
    assert all(set(r) == {"family", "test", "estimate", "p"} for r in rows)          # the rows keep their four keys
    assert [r["estimate"] for r in rows] == [float(i) for i in range(1, 19)] and {r["p"] for r in rows} == {0.25}
    alphas = [0.05, 0.05, 0.01, 0.025, 0.10, 0.05, 0.05, 0.05, 0.05] * 2
    assert [kw for _, _, kw in seen] == [{"alpha": a, "n_rep": 9} for a in alphas]
    assert seen[0][0] == ("roll", "LEV6", "logit", 42) and seen[0][1] == ("roll", "DART3", "logit", 42)


def test_audit_family_propagates_a_missing_score_frame():
    from es.secondary import audit_family

    def score(kind, block, estimator, seed):
        if (kind, estimator, seed) == ("frozen", "xgb", 42):
            raise KeyError((kind, block, estimator, seed))      # as the sealed run's lookup does for an unknown key
        return _synthetic_score(kind, block, estimator, seed)

    with pytest.raises(KeyError):
        audit_family(score, n_rep=5)


def _econ_row(policy="P1", comparator="none", position="premium", fees="calibrated", selc=0.1, p=0.2, **extra):
    return {"analysis": "AQ3", "policy": policy, "comparator": comparator, "position": position, "fees": fees,
            "SELC": selc, "rotation_p": p, **extra}


def test_economic_family_covers_every_position_fee_setting_and_exposure_split_of_p1():
    from es.secondary import economic_family
    fees = ("calibrated", "0.0", "0.1", "0.25", "0.5", "1.0", "stress")
    p1 = [_econ_row(position=pos, fees=f, selc=0.01 * i, p=0.01 * (i + 1))
          for i, (pos, f) in enumerate((pos, f) for pos in ("premium", "inc", "dec") for f in fees)]
    splits = [_econ_row(position=f"premium[{ex}]", selc=-0.5, p=0.7) for ex in ("high", "low")]
    other = ([_econ_row(policy=pol, p=0.001) for pol in ("P4", "P5", "P6", "P4@r*")]
             + [_econ_row(policy="comparator", comparator=c, p=0.001) for c in ("VOL@count", "calendar@count", "VOL_P1")]
             + [_econ_row(policy="P1", comparator="VOL@count", p=0.001)])
    out = economic_family(other[:3] + p1 + other[3:6] + splits + other[6:])
    assert [r["test"] for r in out] == ([f"P1 {pos} fees={f}" for pos in ("premium", "inc", "dec") for f in fees]
                                        + ["P1 premium[high] fees=calibrated", "P1 premium[low] fees=calibrated"])
    assert len(out) == 23 and {r["family"] for r in out} == {"economic robustness"}
    assert all(set(r) == {"family", "test", "estimate", "p"} for r in out)
    assert [r["estimate"] for r in out[:21]] == [0.01 * i for i in range(21)]                   # SELC
    assert [r["p"] for r in out[:21]] == [0.01 * (i + 1) for i in range(21)]                    # rotation_p
    assert [(r["estimate"], r["p"]) for r in out[21:]] == [(-0.5, 0.7)] * 2


def test_economic_family_keeps_a_nan_p_for_bh_table_to_leave_out_and_refuses_a_missing_column():
    from es.secondary import economic_family
    out = economic_family([_econ_row(p=float("nan"))])
    assert len(out) == 1 and np.isnan(out[0]["p"])
    assert economic_family([]) == []
    row = _econ_row()
    del row["rotation_p"]
    with pytest.raises(KeyError, match="rotation_p"):                  # a schema change upstream is an error, not an empty family
        economic_family([row])


def _monitor_frame(n=12):
    return pd.DataFrame({"analysis": "A", "block": "LEV6", "market": "M", "node": [f"n{i}" for i in range(n)],
                         "log_error": np.linspace(0, 1, n), "fail": [False] * 6 + [True] * 6,
                         "W1_norm": np.linspace(0, 1, n), "PSI": np.linspace(1, 0, n),
                         "domain_auc": np.linspace(0.5, 1, n), "early_rate": np.linspace(0.05, 0.2, n)})


def test_monitor_family_reads_the_records_of_evaluate_monitors():
    from es.run_monitors import EVAL_COLUMNS, evaluate_monitors
    from es.secondary import monitor_family
    ev = evaluate_monitors(_monitor_frame())
    assert {"block", "monitor", "spearman", "spearman_p_one_sided"} <= set(EVAL_COLUMNS)
    out = monitor_family(ev.to_dict(orient="records"))
    assert [r["test"] for r in out] == ["LEV6 W1_norm", "LEV6 PSI", "LEV6 domain_auc", "LEV6 early_rate"]
    assert {r["family"] for r in out} == {"drift monitors"} and all(set(r) == {"family", "test", "estimate", "p"} for r in out)
    by_monitor = ev.set_index("monitor")
    for r in out:
        monitor = r["test"].split()[1]
        assert r["estimate"] == by_monitor.loc[monitor, "spearman"]
        assert r["p"] == by_monitor.loc[monitor, "spearman_p_one_sided"]
    assert out[0]["p"] < 0.01 and out[1]["p"] > 0.99                     # W1 rises with the error, PSI falls with it
    assert monitor_family([]) == []


def test_monitor_family_keeps_a_nan_p_when_a_monitor_is_constant():
    from es.run_monitors import evaluate_monitors
    from es.secondary import bh_table, monitor_family
    from scipy.stats import ConstantInputWarning
    frame = _monitor_frame().assign(PSI=0.3)
    with pytest.warns(ConstantInputWarning, match="constant"):          # scipy says so; the rho and p are NaN
        ev = evaluate_monitors(frame)
    out = monitor_family(ev.to_dict(orient="records"))
    psi = next(r for r in out if r["test"] == "LEV6 PSI")
    assert np.isnan(psi["estimate"]) and np.isnan(psi["p"])
    t = bh_table(out)
    assert t.loc[t["test"] == "LEV6 PSI", "bh_reject"].tolist() == [False]
    assert t["bh_reject"].sum() >= 1                                        # the other monitors are still adjusted


def _rows(family, ps):
    return [{"family": family, "test": f"{family}{i}", "estimate": 0.0, "p": p} for i, p in enumerate(ps)]


def test_bh_table_decisions_land_on_their_own_rows_when_the_families_are_interleaved():
    from es.secondary import bh_table
    a, b = _rows("a", [0.001, 0.02, 0.04, 0.3]), _rows("b", [0.2, 0.012, 0.9])
    rows = [a[3], b[0], a[0], b[1], a[2], b[2], a[1]]
    t = bh_table(rows)
    assert t["test"].tolist() == [r["test"] for r in rows]                                  # row order is kept
    assert t["bh_reject"].tolist() == [False, False, True, True, False, False, True]
    assert list(t.columns) == ["family", "test", "estimate", "p", "bh_reject", "included"]
    assert t["bh_reject"].dtype == bool and t["included"].dtype == bool


def test_bh_table_adjusts_each_family_on_its_own_not_pooled():
    from es.secondary import bh_table
    # alone, 0.04 is rejected at 0.05; pooled with the three other p-values of family a it would not be (0.04 * 5 / 4 > 0.05)
    t = bh_table(_rows("a", [0.001, 0.02, 0.5]) + _rows("b", [0.04]))
    assert t.loc[t.family == "b", "bh_reject"].tolist() == [True]
    assert t.loc[t.family == "a", "bh_reject"].tolist() == [True, True, False]
    q = bh_table(_rows("a", [0.001, 0.02, 0.5]), q=0.01)
    assert q["bh_reject"].tolist() == [True, False, False]


def test_bh_table_leaves_a_nan_p_out_of_the_adjustment_and_never_rejects_it():
    from es.secondary import bh_table
    # with the NaN counted as a test, 0.03 would face 0.05 * 1 / 2 and fail
    t = bh_table(_rows("b", [0.03, np.nan]))
    assert t["bh_reject"].tolist() == [True, False]
    assert bh_table(_rows("c", [np.nan, np.nan]))["bh_reject"].tolist() == [False, False]
    assert bh_table(_rows("d", [None, 0.01]))["bh_reject"].tolist() == [False, True]


def test_bh_table_refuses_a_p_value_outside_zero_one():
    from es.secondary import bh_table
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        bh_table(_rows("a", [0.01, 1.2]))


def test_bh_table_of_no_rows_is_an_empty_table_with_its_columns():
    from es.secondary import bh_table
    t = bh_table([])
    assert len(t) == 0 and list(t.columns) == ["family", "test", "estimate", "p", "bh_reject", "included"]


def test_bh_table_does_not_depend_on_the_index_of_a_frame_it_is_given():
    from es.secondary import bh_table
    frame = pd.DataFrame(_rows("a", [0.001, 0.02, 0.04, 0.3]), index=[5, 5, 2, 2])      # duplicate labels
    t = bh_table(frame)
    assert t["bh_reject"].tolist() == [True, True, False, False]
    assert frame.columns.tolist() == ["family", "test", "estimate", "p"]                 # the argument is not modified


def test_bh_table_matches_statsmodels_fdr_bh_within_each_family():
    from statsmodels.stats.multitest import multipletests
    from es.secondary import bh_table
    rng = np.random.default_rng(5)
    rows = []
    for family, m in (("audit robustness", 18), ("economic robustness", 23), ("drift monitors", 20)):
        rows += _rows(family, rng.uniform(size=m) ** 3)
    t = bh_table(rows)
    for family, g in t.groupby("family"):
        assert g["bh_reject"].tolist() == multipletests(g["p"], alpha=0.05, method="fdr_bh")[0].tolist()
    assert t["bh_reject"].sum() > 5


def test_bh_table_columns_are_the_documented_ones():
    from es.secondary import TABLE_COLUMNS, bh_table
    assert TABLE_COLUMNS == ["family", "test", "estimate", "p", "bh_reject", "included"]
    assert list(bh_table(_rows("a", [0.01])).columns) == TABLE_COLUMNS


def test_bh_table_marks_the_rows_that_entered_the_adjustment():
    from es.run_inference import bh
    from es.secondary import bh_table
    rows = _rows("a", [0.012, np.nan, 0.03, 0.9]) + _rows("b", [np.nan, np.nan]) + _rows("c", [0.03, np.nan])
    t = bh_table(rows)
    assert t["included"].tolist() == [True, False, True, True, False, False, True, False]
    assert t["included"].tolist() == t["p"].notna().tolist() and t["included"].dtype == bool
    assert not t.loc[~t["included"], "bh_reject"].any()                        # an excluded row is never rejected
    # `included` summed over a family is the m that BH used: re-running bh on the included p-values gives the decisions,
    # and counting the NaN row as a test would have changed them (0.03 faces 0.05 / 2 in family c, and fails)
    for family, g in t.groupby("family"):
        used = g[g["included"]]
        assert used["bh_reject"].tolist() == bh(used["p"].to_numpy(), 0.05).tolist()
    assert t.groupby("family")["included"].sum().to_dict() == {"a": 3, "b": 0, "c": 1}
    assert t.loc[t.family == "c", "bh_reject"].tolist() == [True, False]
    assert bh([0.03, 0.5]).tolist() == [False, False]                          # what a count of 2 would have decided


def test_bh_table_refuses_a_test_that_appears_twice_in_a_family():
    from es.secondary import bh_table
    rows = _rows("a", [0.01, 0.02, 0.5]) + [{"family": "a", "test": "a1", "estimate": 9.0, "p": 0.04}]
    with pytest.raises(ValueError, match=r"2 rows repeat a \(family, test\) pair.*a1"):      # both copies are counted
        bh_table(rows)
    with pytest.raises(ValueError, match="repeat"):                              # an exact copy of a row is no better
        bh_table(_rows("a", [0.01]) * 2)
    with pytest.raises(ValueError, match="repeat"):                              # nor is a repeated NaN-p test
        bh_table(_rows("a", [np.nan]) * 2)


def test_bh_table_allows_the_same_test_name_in_different_families():
    from es.secondary import bh_table
    rows = [{"family": f, "test": "same name", "estimate": 0.0, "p": p} for f, p in (("a", 0.01), ("b", 0.02))]
    assert bh_table(rows)["bh_reject"].tolist() == [True, True]


def test_bh_table_names_every_repeated_pair_it_reports():
    from es.secondary import bh_table
    rows = (_rows("a", [0.1, 0.2]) + _rows("a", [0.3])[:1] + _rows("b", [0.4, 0.5]) + _rows("b", [0.6, 0.7]))
    with pytest.raises(ValueError) as caught:
        bh_table(rows)
    message = str(caught.value)
    assert "a0" in message and "b0" in message and "b1" in message and "a1" not in message


def test_bh_table_refuses_a_row_without_a_family():
    from es.secondary import bh_table
    for missing in (np.nan, None):                       # groupby would skip the row: neither adjusted nor counted
        rows = _rows("a", [0.01, 0.2]) + [{"family": missing, "test": "orphan", "estimate": 0.0, "p": 0.001}]
        with pytest.raises(ValueError, match=r"1 row\(s\) have no family \(tests \['orphan'\]\)"):
            bh_table(rows)
    rows = (_rows("a", [0.01, 0.2]) + [{"family": np.nan, "test": "orphan1", "estimate": 0.0, "p": 0.001},
                                       {"family": None, "test": "orphan2", "estimate": 0.0, "p": 0.002}])
    with pytest.raises(ValueError, match=r"2 row\(s\) have no family \(tests \['orphan1', 'orphan2'\]\)"):
        bh_table(rows)


def test_the_registered_docstring_is_pinned_by_its_sha256():
    import hashlib
    import es.secondary as sec
    # The protocol quotes this docstring verbatim (Ruling 37). Any edit, even of one word, changes the hash, which the 11
    # phrases of the test above would not notice ("one-sided" to "two-sided", "SELC" to "NET").
    digest = hashlib.sha256(sec.__doc__.encode("utf-8")).hexdigest()
    assert digest == "f763c38f9cb5660a04d039603e2e8dd28749142e82b50c33fff676244ee33b40"
