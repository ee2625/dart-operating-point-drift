import datetime as dt

import numpy as np
import pandas as pd


def _scores():
    rng = np.random.default_rng(0)
    rows = [{"market": "M", "node": "A", "split": split, "settle_date": start + dt.timedelta(days=d),
             "score": rng.normal()}
            for split, start, n in (("train", dt.date(2024, 7, 1), 120), ("calib", dt.date(2025, 7, 1), 92),
                                    ("deploy", dt.date(2026, 7, 1), 92))
            for d in range(n)]
    return pd.DataFrame(rows)


def test_cutoffs_use_only_pre_deployment_rows():
    from es.analyses import SEALED
    from es.cutoffs import cutoffs_from_scores
    s = _scores()
    a = SEALED["SA"]
    r1 = cutoffs_from_scores(s, a, "G1_24")[0]
    r2 = cutoffs_from_scores(s.assign(score=np.where(s["split"] == "deploy", 99.0, s["score"])), a, "G1_24")[0]
    assert r1 == r2
    assert r1["tau_p1"] == float(np.quantile(s.loc[s["split"] == "calib", "score"], 0.95))
    season = s[(s["split"] == "train") & s["settle_date"].map(lambda d: d.month).isin({7, 8, 9})]["score"]
    assert r1["tau_p2_season"] == float(np.quantile(season, 0.95))
    assert r1["analysis"] == "SA" and r1["n_calib"] == 92


def test_cutoffs_ignore_unobserved_categorical_pairs():
    """The sealed run passes slim frames with categorical keys (Ruling 60). Two markets with one node each
    must give two rows, not the four of the category product."""
    from es.analyses import SEALED
    from es.cutoffs import cutoffs_from_scores
    s = pd.concat([_scores(), _scores().assign(market="N", node="B")], ignore_index=True)
    c = s.copy()
    for col in ("market", "node", "split"):
        c[col] = c[col].astype("category")
    plain, cat = cutoffs_from_scores(s, SEALED["SA"], "G1_24"), cutoffs_from_scores(c, SEALED["SA"], "G1_24")
    assert len(cat) == 2 and cat == plain


def test_cutoff_table_round_trips_through_csv(tmp_path):
    """_tau_check reads the registered table with float_precision="round_trip"; pandas' default parser does
    not round-trip to_csv's shortest-repr floats (probe, 23 Oct: 99,544 of 200,000 values off by 1 ulp)."""
    rng = np.random.default_rng(1)
    v = rng.normal(size=50_000) * 10.0 ** rng.integers(-3, 4, size=50_000)
    t = pd.DataFrame({"tau_p1": v, "tau_p2_season": np.where(rng.random(50_000) < 0.01, np.nan, v)})
    t.to_csv(tmp_path / "c.csv", index=False)
    back = pd.read_csv(tmp_path / "c.csv", float_precision="round_trip")
    assert (back["tau_p1"].to_numpy() == v).all()
    assert (back["tau_p2_season"].isna() == t["tau_p2_season"].isna()).all()


# --------------------------------------------------------------------------------------------------------------------
# Beyond the brief's three tests: the rule itself, the columns, and the edge cases the sealed run meets.
# --------------------------------------------------------------------------------------------------------------------
def test_the_cutoff_rule_is_the_policies_rule():
    """cutoffs.py and es.policies must place tau with the same quantile rule, or the sealed run's tau check and the
    registered table would disagree with the cutoffs P1 and P2 actually use."""
    from es.analyses import SEALED
    from es.cutoffs import cutoffs_from_scores
    from es.policies import p1_fixed, p2_insample
    s = _scores()
    s["timestamp_utc"] = pd.to_datetime(s["settle_date"]).dt.tz_localize("UTC")      # es.policies sorts on it
    row = cutoffs_from_scores(s, SEALED["SA"], "G1_24", alpha=0.10)[0]
    p1 = p1_fixed(s, 0.10)
    p2 = p2_insample(s, 0.10, SEALED["SA"].deploy.months)
    assert row["tau_p1"] == p1["tau"].iloc[0] and row["tau_p2_season"] == p2["tau"].iloc[0]


def test_a_node_without_calibration_rows_gets_a_nan_cutoff_and_n_calib_zero():
    from es.analyses import SEALED
    from es.cutoffs import cutoffs_from_scores
    s = _scores()
    s = s[s["split"] != "calib"]
    row = cutoffs_from_scores(s, SEALED["SA"], "G1_24")[0]
    assert row["n_calib"] == 0 and np.isnan(row["tau_p1"])
    assert np.isfinite(row["tau_p2_all"]) and np.isfinite(row["tau_p2_season"])


def test_a_node_without_training_rows_in_the_season_gets_a_nan_season_cutoff():
    from es.analyses import SEALED
    from es.cutoffs import cutoffs_from_scores
    s = _scores()
    # SA deploys in July-September; keep only training rows dated in October
    s = s[(s["split"] != "train") | s["settle_date"].map(lambda d: d.month == 10)]
    row = cutoffs_from_scores(s, SEALED["SA"], "G1_24")[0]
    assert np.isnan(row["tau_p2_season"]) and np.isfinite(row["tau_p2_all"]) and np.isfinite(row["tau_p1"])


def test_cutoff_table_columns_ordering_flags_and_arguments(monkeypatch):
    """cutoff_table fits one xgb model per (analysis, block) through score_analysis (faked here: the fit is the
    scores module's business), and flags where each P2 cutoff lies above P1's."""
    import es.cutoffs as cutoffs
    from es.analyses import SEALED
    calls = []

    def fake_score_analysis(panel, analysis, block, estimator, seed):
        calls.append((analysis.id, block, estimator, seed))
        return _scores().assign(score=lambda f: f["score"] + (0.5 if block == "DART3" else 0.0))

    monkeypatch.setattr(cutoffs, "score_analysis", fake_score_analysis)
    t = cutoffs.cutoff_table("panel", [SEALED["SA"], SEALED["SAF"]], blocks=("G1_24", "DART3"), seed=7, alpha=0.05)
    assert calls == [("SA", "G1_24", "xgb", 7), ("SA", "DART3", "xgb", 7),
                     ("SAF", "G1_24", "xgb", 7), ("SAF", "DART3", "xgb", 7)]
    assert list(t.columns) == ["analysis", "block", "market", "node", "n_calib", "tau_p1", "tau_p2_season",
                               "tau_p2_all", "p2_season_above_p1", "p2_all_above_p1"]
    assert list(t["analysis"]) == ["SA", "SA", "SAF", "SAF"] and list(t["block"]) == ["G1_24", "DART3"] * 2
    assert (t["p2_season_above_p1"] == (t["tau_p2_season"] > t["tau_p1"])).all()
    assert (t["p2_all_above_p1"] == (t["tau_p2_all"] > t["tau_p1"])).all()


def test_cutoff_blocks_are_the_four_registered_primary_pipelines():
    from es.blocks import BLOCKS
    from es.cutoffs import CUTOFF_BLOCKS
    assert CUTOFF_BLOCKS == ("G1_24", "DART3", "LEV6", "ROLL_DISP9") and set(CUTOFF_BLOCKS) <= set(BLOCKS)
