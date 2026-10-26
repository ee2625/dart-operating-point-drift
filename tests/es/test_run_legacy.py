from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.mark.slow
def test_sealed_row_reproduces_published():
    from es.data import build_panel, load_dev_raw
    from es.analyses import LEGACY
    from es.run_legacy import legacy_summary
    panel = build_panel(load_dev_raw())
    s = legacy_summary(panel, LEGACY["sealed26H1"], "G1_24", "xgb", 42, "P1")
    assert int(s["fail_0.5_2"].sum()) == 14
    assert round(float(s["R"].median()), 2) == 0.59


def test_published_csv_is_well_formed():
    p = pd.read_csv(Path(__file__).resolve().parents[2] / "es" / "published_v3.csv")
    assert {"item", "analysis", "block", "estimator", "metric", "value", "source_line"} <= set(p.columns)
    assert len(p) >= 60


# ---------------------------------------------------------------------------------------------
# Helpers added beyond the brief. Synthetic data only, so these run in a few seconds.
# ---------------------------------------------------------------------------------------------

OPERATORS = {"NYISO": 10, "ISONE": 9, "MISO": 8, "CAISO": 6}      # 33 nodes, as in the panel


def _fake_frame(rng, r_scale=0.6):
    """A per-node frame shaped like ``es.run_legacy._per_node``'s output (33 nodes)."""
    rows = []
    for market, n in OPERATORS.items():
        for i in range(n):
            r = float(rng.lognormal(0.0, r_scale))
            rows.append({"market": market, "node": f"{market}{i}", "n": 1000, "alerts": int(50 * r), "rate": 0.05 * r,
                         "R": r, "zero_activity": False, "R_floored": r, "log_error": abs(np.log10(r)),
                         "fail_0.5_2": r < 0.5 or r > 2, "fail_0.67_1.5": r < 2 / 3 or r > 1.5,
                         "fail_0.8_1.25": r < 0.8 or r > 1.25,
                         "auprc": float(rng.uniform(0.05, 0.3)), "prevalence": float(rng.uniform(0.05, 0.15))})
    return pd.DataFrame(rows)


def _fake_scores(rng):
    rows = []
    for node in ("a", "b"):
        for split, n in (("train", 300), ("calib", 400), ("deploy", 500)):
            rows.append(pd.DataFrame({"market": "NYISO", "node": node, "split": split,
                                      "score": rng.uniform(0, 1, n)}))
    return pd.concat(rows, ignore_index=True)


def test_comparison_matches_at_the_printed_precision():
    from es.run_legacy import build_comparison
    published = pd.DataFrame({
        "item": list("abcdefghi"), "analysis": "x", "block": "B", "estimator": "xgb",
        "metric": ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9"],
        "value": [12.0, 1.7, 0.20, 329000.0, 0.99, 5.0, 7.0, -0.003, -0.003],
        "decimals": [0, 1, 2, -3, 2, 0, 0, 3, 3]})
    regenerated = pd.DataFrame({
        "analysis": "x", "block": "B", "estimator": "xgb",
        "metric": ["m1", "m2", "m3", "m4", "m5", "m6", "m8", "m9"],
        "regenerated": [12.0, 1.662, 0.1965, 306390.0, 1.0051, 6.0, -0.0027, 0.003]})
    comp = build_comparison(published, regenerated)
    assert comp["status"].tolist() == ["match", "match", "match", "mismatch", "mismatch", "mismatch",
                                       "no_counterpart", "match", "mismatch"]         # m9: the sign differs
    assert comp["match"].tolist() == [True, True, True, False, False, False, False, True, False]


def test_comparison_rejects_duplicate_regenerated_keys():
    from es.run_legacy import build_comparison
    published = pd.DataFrame({"item": ["a"], "analysis": "x", "block": "B", "estimator": "xgb", "metric": "m",
                              "value": [1.0], "decimals": [0]})
    regenerated = pd.DataFrame({"analysis": "x", "block": "B", "estimator": "xgb", "metric": "m",
                                "regenerated": [1.0, 2.0]})
    with pytest.raises(ValueError, match="not unique"):
        build_comparison(published, regenerated)


def test_sign_test_operator_contrast_and_delta_auprc():
    from es.run_legacy import delta_auprc, operator_contrasts, sign_test
    keys = pd.DataFrame({"market": ["NYISO"] * 4 + ["MISO"] * 4, "node": list("abcdefgh")})
    lev = keys.assign(log_error=[0.5, 0.4, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1], auprc=[0.2, 0.2, 0.2, 0.2, 0.3, 0.3, np.nan, 0.3])
    dart = keys.assign(log_error=[0.1, 0.1, 0.1, 0.3, 0.2, 0.3, 0.2, 0.1], auprc=[0.3, 0.1, 0.4, 0.2, 0.2, 0.4, 0.5, 0.3])
    k, n, p = sign_test(lev, dart)
    assert (k, n) == (3, 8)                               # level worse at a, b, c; DART-relative worse at d, e, f, g
    assert p == pytest.approx(0.7265625)                  # two-sided binomial(8, 0.5), k = 3
    op = operator_contrasts(lev, dart)
    assert op["NYISO"] == pytest.approx(np.median([-0.4, -0.3, -0.2, 0.1]))   # DART minus level
    assert op["MISO"] == pytest.approx(np.median([0.1, 0.2, 0.1, 0.0]))
    med, npos, n_cmp = delta_auprc(dart, lev)             # node g has no level AUPRC and is not compared
    assert n_cmp == 7 and npos == 3                       # a, c and f improve; d and h tie
    assert med == pytest.approx(np.median([0.1, -0.1, 0.2, 0.0, -0.1, 0.1, 0.0]))


def test_score_dispersion_and_source_activity():
    from es.run_legacy import score_dispersion, source_activity
    rng = np.random.default_rng(1)
    scores = []
    for node in ("a", "b", "c"):
        scores += [pd.DataFrame({"market": "NYISO", "node": node, "split": "calib", "score": rng.normal(0, 1, 5000)}),
                   pd.DataFrame({"market": "NYISO", "node": node, "split": "deploy", "score": rng.normal(0, 2, 5000)})]
    scores = pd.concat(scores, ignore_index=True)
    sd, iq = score_dispersion(scores)
    assert 1.9 < sd < 2.1 and 1.9 < iq < 2.1              # deployment scores are twice as spread out
    lo, hi = source_activity(scores, 0.05)                # P1 on the calibration rows: just above 5%
    assert 0.05 <= lo <= hi < 0.0505
    tie = pd.DataFrame({"market": "NYISO", "node": "a", "split": "calib", "score": np.arange(101.0)})
    assert source_activity(tie, 0.05) == (6 / 101, 6 / 101)   # tau is the 96th score and "score >= tau" keeps it


def test_nwd_is_the_open_holdout_formula():
    from es.run_legacy import _nwd
    a = np.arange(100, dtype=float)
    iqr = np.subtract(*np.percentile(a, [75, 25]))
    assert _nwd(a, a) == 0.0
    assert _nwd(a, a + 10) == pytest.approx(10 / iqr)     # W1 of a pure shift, divided by the calibration IQR
    assert np.isnan(_nwd(a[:49], a)) and np.isnan(_nwd(a, a[:49]))   # fewer than 50 finite values
    assert _nwd(np.r_[a, np.nan], a) == 0.0               # non-finite values are dropped


def _fake_panel(rng):
    """Two nodes of one operator and one of another, hourly over 2023-2024 (UTC), through the real build."""
    from es.data import build_panel
    end = pd.Timestamp("2025-01-01", tz="UTC")
    frames = []
    for market, node, ntype, scale in (("NYISO", "Z1", "load_zone", 1.0), ("NYISO", "Z2", "load_zone", 3.0),
                                       ("MISO", "H1", "trading_hub", 2.0)):
        ts = pd.date_range(pd.Timestamp("2023-01-01", tz="UTC"), end, freq="h", inclusive="left")
        da = 30 + rng.normal(0, 3, len(ts))
        frames.append(pd.DataFrame({"timestamp_utc": ts, "market": market, "node": node, "node_type": ntype,
                                    "da_lmp": da, "rt_lmp": da + rng.normal(0, 4 * scale, len(ts))}))
    return build_panel(pd.concat(frames, ignore_index=True))


def _prevalence_2024(panel, q_window):
    """Per operator, the share of 2024 node-hours (UTC) with |DART| >= the node's 95th percentile on ``q_window``."""
    def utc(day):
        return pd.Timestamp(day, tz="UTC")
    out = {}
    for market, g in panel.groupby("market"):
        hits = total = 0
        for _, h in g.groupby("node"):
            t = pd.to_datetime(h["timestamp_utc"], utc=True)
            a = h["dart_spread"].abs().to_numpy()
            q = np.percentile(a[((t >= utc(q_window[0])) & (t < utc(q_window[1]))).to_numpy()], 95)
            in24 = ((t >= utc("2024-01-01")) & (t < utc("2025-01-01"))).to_numpy()
            hits += int((a[in24] >= q).sum())
            total += int(in24.sum())
        out[market] = hits / total
    return pd.Series(out, dtype=float)


def test_table1_and_figure1_on_a_synthetic_panel():
    from es.run_legacy import fig1_volatility, table1_panel
    panel = _fake_panel(np.random.default_rng(2))
    t1 = table1_panel(panel)
    assert t1.loc["NYISO", ["nodes", "hubs", "zones"]].tolist() == [2, 0, 2]
    assert t1.loc["MISO", ["nodes", "hubs", "zones"]].tolist() == [1, 1, 0]
    assert t1.loc["NYISO", "hourly_rows"] == 2 * len(panel[panel["node"] == "Z1"])
    # D6: the 2024 event rate under q on the 2023 training year (the caption) and under q on 2023-H1 (what the
    # printed column reproduces), each recomputed here from the panel with numpy only
    for column, window in (("prevalence_24_q2023", ("2023-01-01", "2024-01-01")),
                           ("prevalence_24_q2023H1", ("2023-01-01", "2023-07-01"))):
        pd.testing.assert_series_equal(t1[column], _prevalence_2024(panel, window), check_names=False)
    assert (t1["prevalence_24_q2023H1"] != t1["prevalence_24_q2023"]).all()          # the two rules do differ
    f1 = fig1_volatility(panel)
    assert {"quarter", "period", "period_local"} == set(f1["kind"])
    q = f1[f1["kind"] == "quarter"]
    # quarters are by local settlement date, so the first New York evening hours of 2023-01-01 UTC are 2022Q4
    assert list(q.index) == ["2022Q4"] + [f"{y}Q{i}" for y in (2023, 2024) for i in (1, 2, 3, 4)]
    row = q.loc["2023Q1"]
    in_q1 = panel[pd.PeriodIndex(pd.to_datetime(panel["settle_date"]), freq="Q").astype(str) == "2023Q1"]
    assert row["dart_std"] == pytest.approx(in_q1.groupby(["market", "node"])["dart_spread"].std().median())
    assert row["dart_std_pooled"] == pytest.approx(in_q1["dart_spread"].std())      # all node-hours at once
    assert row["dart_std"] != pytest.approx(row["dart_std_pooled"])
    p24 = f1[(f1["kind"] == "period") & (f1.index == "2024")].iloc[0]
    sel = panel[(panel["timestamp_utc"] >= pd.Timestamp("2024-01-01", tz="UTC"))]
    assert p24["dart_p95_pooled"] == pytest.approx(sel["dart_spread"].abs().quantile(0.95))


def _fake_run(rng):
    """Frames, summary and extras for every run main() makes, without fitting a model."""
    from es import run_legacy as rl
    frames, summ, extras = {}, {}, []
    for analysis, block, est, seed, variants in rl.run_plan():
        for policy, alpha in variants:
            key = (analysis.id, block, est, seed, policy, alpha)
            frames[key] = _fake_frame(rng)
            summ[key] = rl._summ(frames[key])
        extras += rl._extras_for(_fake_scores(rng), analysis, block, est, seed)
    return frames, rl.summary_frame(summ), extras


def test_run_plan_covers_the_runs_the_brief_asks_for():
    from es import run_legacy as rl
    from es.analyses import LABEL_CONTROL, LEGACY
    plan = list(rl.run_plan())
    ids = {a.id for a, *_ in plan}
    assert ids == set(LEGACY) | set(LABEL_CONTROL)
    runs = [(a.id, b, e, s, pol, al) for a, b, e, s, variants in plan for pol, al in variants]
    assert len(runs) == len(set(runs))                    # no run is made twice
    assert ("sealed26H1", "G1_24", "xgb", 42, "P1", 0.05) in runs
    assert ("2024-2025", "G1_24", "xgb", 42, "P2_all_train", 0.05) in runs      # Fig. 2 provenance
    assert ("25H1-26H1", "DART18", "xgb", 42, "P1", 0.025) in runs              # alpha sweep
    assert ("25H1-26H1", "ROLL_DISP9", "xgb", 13, "P1", 0.05) in runs           # seeds
    assert ("25H1-26H1", "DART18", "logit", 42, "P1", 0.05) in runs             # estimator robustness
    assert not any(a == "24H1-25H1" and s != 42 for a, _, _, s, _, _ in runs)  # seeds only for the two key pairs


def test_every_published_number_has_a_regenerated_counterpart():
    """The key plumbing: after a run, only the number with no matched-design counterpart is unmatched (D3)."""
    from es import run_legacy as rl
    rng = np.random.default_rng(3)
    frames, summary, extras = _fake_run(rng)
    table1 = pd.DataFrame({"nodes": [6, 9, 8, 10], "hourly_rows": [1, 2, 3, 4], "hubs": [3, 1, 8, 0],
                           "zones": [3, 8, 0, 10], "prevalence_24_q2023": 0.06, "prevalence_24_q2023H1": 0.05},
                          index=pd.Index(["CAISO", "ISONE", "MISO", "NYISO"], name="market"))
    fig1 = pd.DataFrame({"kind": ["period", "period"], "dart_p95_pooled": [30.0, 62.0]},
                        index=pd.Index(["2024", "2026H1"], name="period"))
    published = rl.read_published()
    regenerated = rl.regenerated_numbers(summary, frames, extras, table1, fig1, rho=0.047, n_nodes=33,
                                         published=published)
    comp = rl.build_comparison(published, regenerated)
    orphans = comp[comp["status"] == "no_counterpart"]
    assert set(orphans["analysis"]) == {"heldout-2024"}, orphans[["item", "analysis", "block", "estimator", "metric"]]
    assert len(orphans) == 1
    # D1 and D6: the six rows tagged with a stated design find their as-stated value (the call raises otherwise)
    flagged = comp[comp["stated_analysis"] != ""]
    assert len(flagged) == 6 and flagged["as_stated_value"].notna().all()
    assert set(flagged["stated_analysis"]) == {"24H2-25H2", "panel-q2023"}
    for market in ("CAISO", "ISONE", "MISO", "NYISO"):                           # D6: each label rule reaches its own key
        pick = (regenerated["block"] == market) & (regenerated["metric"] == "prevalence")
        assert regenerated[pick & (regenerated["analysis"] == "panel-q2023")]["regenerated"].tolist() == [0.06]
        assert regenerated[pick & (regenerated["analysis"] == "panel-q2023H1")]["regenerated"].tolist() == [0.05]


def test_regenerated_names_follow_the_published_vocabulary():
    from es import run_legacy as rl
    frames, summary, extras = _fake_run(np.random.default_rng(4))
    long = rl.base_metrics(summary)
    names = set(long["metric"])
    assert {"failures", "median_R", "median_log_error", "median_auprc", "auprc_lift", "prevalence", "n_nodes",
            "failures@band1.5", "failures@a0.025", "failures@a0.1", "failures@P2_all_train",
            "median_R@P2_all_train", "failures_seed_min", "failures_seed_max", "failures_NYISO"} <= names
    one = long[(long["analysis"] == "2024-2025") & (long["block"] == "LEV6") & (long["estimator"] == "xgb")]
    seeds = summary.reset_index().query("analysis == '2024-2025' and block == 'LEV6' and policy == 'P1' "
                                        "and estimator == 'xgb' and alpha == 0.05")["failures"]
    assert len(seeds) == 5
    got = one.set_index("metric")["regenerated"]
    assert got["failures_seed_min"] == seeds.min() and got["failures_seed_max"] == seeds.max()


def test_derived_counts_follow_the_definitions():
    from es import run_legacy as rl
    frames, summary, extras = _fake_run(np.random.default_rng(5))
    # make Fig. 2 deterministic: every node's R falls under the in-sample cutoff
    held = frames[("2024-2025", "G1_24", "xgb", 42, "P1", 0.05)]
    frames[("2024-2025", "G1_24", "xgb", 42, "P2_all_train", 0.05)] = held.assign(R=held["R"] / 2)
    d = rl.derived_metrics(frames).set_index(["analysis", "block", "estimator", "metric"])["regenerated"]
    assert d[("2024-2025", "G1_24", "xgb", "nodes_down")] == 33
    assert d[("2024-2025", "G1_24", "xgb", "median_R_ratio")] == pytest.approx(2.0)
    f = [int(frames[(a, "LEV6", "xgb", 42, "P1", 0.05)]["fail_0.5_2"].sum()) for a in rl.PAIRS_2024_25]
    assert d[("agg:2024-25", "LEV6", "xgb", "failures_min")] == min(f)
    assert d[("agg:2024-25", "LEV6", "xgb", "failures_max")] == max(f)
    failing = above = 0
    for a in rl.PAIRS_2:
        for b in rl.RANKING_BLOCKS:
            x = frames[(a, b, "xgb", 42, "P1", 0.05)]
            x = x[x["fail_0.5_2"]]
            failing += len(x)
            above += int((x["auprc"] / x["prevalence"] > 1).sum())
    assert d[("agg:2pairs", "5blocks", "xgb", "n_failing_nodeblocks")] == failing
    assert d[("agg:2pairs", "5blocks", "xgb", "n_failing_above_prev")] == above
    assert d[("all", "G6", "-", "dim")] == 0 and d[("all", "G1_24", "-", "dim")] == 24
    assert d[("all", "LEV6", "-", "dim")] == 6 and d[("all", "DART18", "-", "dim")] == 18


def test_published_csv_vocabulary():
    p = pd.read_csv(Path(__file__).resolve().parents[2] / "es" / "published_v3.csv", dtype=str, keep_default_na=False)
    assert {"decimals", "note", "stated_analysis"} <= set(p.columns)
    flagged = p[p["stated_analysis"] != ""]
    assert set(zip(flagged["analysis"], flagged["block"], flagged["metric"], flagged["stated_analysis"])) == {
        ("24H2-25H2-train2023", "LEV6", "median_log_error", "24H2-25H2"),       # D1: the m-tilde column
        ("24H2-25H2-train2023", "DART18", "median_log_error", "24H2-25H2"),
        ("panel-q2023H1", "NYISO", "prevalence", "panel-q2023"),                # D6: the Table 1 prevalence column
        ("panel-q2023H1", "ISONE", "prevalence", "panel-q2023"),
        ("panel-q2023H1", "MISO", "prevalence", "panel-q2023"),
        ("panel-q2023H1", "CAISO", "prevalence", "panel-q2023")}
    assert flagged["note"].str.match(r"^D[16]:").all()
    assert p["value"].map(lambda v: float(v) == float(v)).all()           # numeric, no NaN
    assert p["decimals"].map(lambda v: int(v) == float(v)).all()
    assert p["source_line"].str.match(r"^(MR|MAIN|IR|DS|PM):\d+(-\d+)?$").all()
    assert not p.duplicated(["item", "analysis", "block", "estimator", "metric", "source_line"]).any()


def test_label_control_twins_are_compared_run_by_run():
    """LC-full-q2324 is the sealed design and LC-h1-q2023 is the 25H1->26H1 design under other ids; their
    runs must agree to the last digit, which is a determinism check on the whole scoring path."""
    from es import run_legacy as rl
    rng = np.random.default_rng(6)
    frames = {}
    for twin, original in rl.LC_TWINS.items():
        for block, est in (("LEV6", "xgb"), ("G6", "logit")):
            f = _fake_frame(rng)
            frames[(original, block, est, 42, "P1", 0.05)] = f
            frames[(twin, block, est, 42, "P1", 0.05)] = f.copy()
    assert rl.twin_check(frames) == {twin: 0.0 for twin in rl.LC_TWINS}
    key = ("LC-h1-q2023", "LEV6", "xgb", 42, "P1", 0.05)
    frames[key] = frames[key].assign(R=frames[key]["R"] + 0.25)
    assert rl.twin_check(frames)["LC-h1-q2023"] == pytest.approx(0.25)


def _scored_nodes(rng):
    """Two nodes of a score frame as ``es.scores.score_analysis`` returns it (split, score, label, dates)."""
    parts = []
    for node, shift in (("a", 1.0), ("b", 0.4)):
        for split, start, n in (("train", "2023-01-01", 400), ("calib", "2024-01-01", 1000), ("deploy", "2025-01-01", 1000)):
            ts = pd.date_range(start, periods=n, freq="h", tz="UTC")
            raw = rng.uniform(0, 1, n)
            label = (raw + rng.normal(0, 0.3, n) > 1.0).astype(float)      # labels follow the score, noisily
            parts.append(pd.DataFrame({
                "market": "NYISO", "node": node, "timestamp_utc": ts, "settle_date": ts.date, "split": split,
                "score": raw * (shift if split == "deploy" else 1.0), "label": label}))
    return pd.concat(parts, ignore_index=True)


def test_per_node_run_reports_activity_ranking_and_the_cutoff(tmp_path, monkeypatch):
    from es import run_legacy as rl
    from es.analyses import LEGACY
    monkeypatch.setattr(rl, "LEGACY_DIR", tmp_path)
    scores = _scored_nodes(np.random.default_rng(7))
    out = rl._per_node(scores, LEGACY["2024-2025"], "LEV6", "xgb", 42, "P1", 0.05)
    assert out["node"].tolist() == ["a", "b"] and set(out["analysis"]) == {"2024-2025"}
    for node, g in scores.groupby("node"):
        tau = np.quantile(g.loc[g["split"] == "calib", "score"], 0.95)
        dep = g[g["split"] == "deploy"]
        row = out[out["node"] == node].iloc[0]
        assert row["tau"] == pytest.approx(tau)
        assert row["R"] == pytest.approx((dep["score"] >= tau).mean() / 0.05)
        assert row["prevalence"] == pytest.approx(dep["label"].mean())
        assert row["fail_0.5_2"] == (row["R"] < 0.5 or row["R"] > 2)
    assert out.loc[out["node"] == "b", "R"].iloc[0] < out.loc[out["node"] == "a", "R"].iloc[0]   # lower deployment scores
    assert (out["auprc"] > out["prevalence"]).all()
    written = pd.read_csv(tmp_path / "2024-2025__LEV6__xgb__s42__P1__a0.05.csv")
    assert len(written) == 2 and "tau" in written.columns
    # the in-sample policy puts the cutoff on the training rows of every month
    ins = rl._per_node(scores, LEGACY["2024-2025"], "G1_24", "xgb", 42, "P2_all_train", 0.05)
    for node, g in scores.groupby("node"):
        assert ins.loc[ins["node"] == node, "tau"].iloc[0] == pytest.approx(
            np.quantile(g.loc[g["split"] == "train", "score"], 0.95))


def test_zero_activity_is_counted_over_the_runs_behind_the_published_numbers():
    """"No cell has r = 0" (MR:50-51) covers every run behind a published number: the default runs, the alpha = 2.5%
    and 10% cells, the Fig. 2 in-sample cell and the seed runs. A zero cell in a run the paper does not report (here
    the sealed-design calendar logit) is counted in ``agg:legacy`` only, so it cannot contradict the sentence."""
    from es import run_legacy as rl
    runs = rl.paper_runs(rl.read_published())
    assert ("2024-2025", "LEV6", "xgb", 42, "P1", 0.05) in runs and ("frozen-26H1", "DART3", "xgb", 42, "P1", 0.05) in runs
    assert ("2024-2025", "DART3", "logit", 42, "P1", 0.05) in runs                       # Fig. 3, rolling logit
    assert ("24H2-25H2", "G6", "xgb", 42, "P1", 0.05) in runs                            # only inside agg:4pairs
    assert ("2024-2025", "DART18", "xgb", 42, "P1", 0.05) in runs and ("25H1-26H1", "LEV6", "xgb", 42, "P1", 0.05) in runs
    assert ("25H1-26H1", "LEV6", "xgb", 42, "P1", 0.025) in runs and ("25H1-26H1", "DART18", "xgb", 42, "P1", 0.1) in runs
    assert ("2024-2025", "LEV6", "xgb", 42, "P1", 0.025) in runs                         # alpha = 2.5%, MR:307
    assert ("2024-2025", "G1_24", "xgb", 42, "P2_all_train", 0.05) in runs               # Fig. 2
    assert ("2024-2025", "LEV6", "xgb", 13, "P1", 0.05) in runs and ("25H1-26H1", "ROLL_DISP9", "xgb", 0, "P1", 0.05) in runs
    assert ("2024-2025", "LEV6", "xgb", 42, "P1", 0.1) not in runs                       # 10% is reported for 2026 only
    assert ("2024-2025", "LEV6", "xgb", 42, "P1", 0.01) not in runs and ("24H1-25H1", "LEV6", "xgb", 42, "P1", 0.025) not in runs
    assert ("2024-2025", "DART18", "xgb", 13, "P1", 0.05) not in runs                    # no DART18 seed range is printed
    assert ("sealed26H1", "G6", "logit", 42, "P1", 0.05) not in runs
    assert ("2024-2025", "ROLL15", "xgb", 42, "P1", 0.05) not in runs
    assert not any(k[0] not in rl.LEGACY for k in runs)

    frames, _, _ = _fake_run(np.random.default_rng(8))

    def zero_rows(frames_, runs_=runs):
        d = rl.derived_metrics(frames_, runs_).set_index(["analysis", "block", "estimator", "metric"])["regenerated"]
        return d[("agg:paper", "all", "all", "n_zero_activity")], d[("agg:legacy", "all", "all", "n_zero_activity")]

    def put(key, n):
        frames[key] = frames[key].assign(zero_activity=[True] * n + [False] * (33 - n))

    assert zero_rows(frames) == (0, 0)
    put(("sealed26H1", "G6", "logit", 42, "P1", 0.05), 1)                                # outside the paper
    assert zero_rows(frames) == (0, 1)
    put(("2024-2025", "LEV6", "xgb", 42, "P1", 0.025), 1)                                # reported alpha = 2.5%
    assert zero_rows(frames) == (1, 1)
    put(("25H1-26H1", "DART18", "xgb", 42, "P1", 0.1), 2)                                # reported alpha = 10%
    assert zero_rows(frames) == (3, 1)
    put(("2024-2025", "G1_24", "xgb", 42, "P2_all_train", 0.05), 3)                      # the Fig. 2 in-sample cell
    assert zero_rows(frames) == (6, 1)
    put(("2024-2025", "LEV6", "xgb", 7, "P1", 0.05), 4)                                  # a reported seed run
    assert zero_rows(frames) == (10, 1)
    for key in (("2024-2025", "LEV6", "xgb", 42, "P1", 0.01), ("24H1-25H1", "LEV6", "xgb", 42, "P1", 0.025),
                ("2024-2025", "DART18", "xgb", 13, "P1", 0.05), ("LC-h1-q2324", "G6", "xgb", 42, "P1", 0.05)):
        put(key, 5)                                                                      # none of these is reported
    assert zero_rows(frames) == (10, 1)
    put(("2024-2025", "LEV6", "xgb", 42, "P1", 0.05), 2)                                 # a default cell of the paper
    assert zero_rows(frames) == (12, 3)
    with pytest.raises(ValueError, match="not made"):                                    # a reported run that was never run
        rl.derived_metrics(frames, runs | {("2024-2025", "LEV6", "xgb", 99, "P1", 0.05)})


def test_comparison_separates_numbers_that_reproduce_only_under_another_design():
    """D1 and D6: a published number can be reproduced by the analysis that produced it and still be wrong under the
    design the paper states or the revision adopts. Those rows get status match_other_design and match False."""
    from es.run_legacy import build_comparison
    published = pd.DataFrame({
        "item": list("abcdefg"), "analysis": ["producer"] * 5 + ["gone", "producer"], "block": "B", "estimator": "xgb",
        "metric": ["m1", "m2", "m3", "m4", "m5", "m6", "m7"], "value": [0.23, 0.16, 0.30, 0.20, 0.50, 0.90, 0.40],
        "decimals": 2, "stated_analysis": ["stated", "stated", "stated", "", "stated", "stated", "stated"]})
    regenerated = pd.DataFrame([
        ("producer", "B", "xgb", "m1", 0.2301), ("stated", "B", "xgb", "m1", 0.32),
        ("producer", "B", "xgb", "m2", 0.159), ("stated", "B", "xgb", "m2", 0.2113),
        ("producer", "B", "xgb", "m3", 0.301), ("stated", "B", "xgb", "m3", 0.299),
        ("producer", "B", "xgb", "m4", 0.2004),
        ("producer", "B", "xgb", "m5", 0.60), ("stated", "B", "xgb", "m5", 0.50),
        ("stated", "B", "xgb", "m6", 0.90),
        ("producer", "B", "xgb", "m7", 0.70), ("stated", "B", "xgb", "m7", 0.10)],
        columns=["analysis", "block", "estimator", "metric", "regenerated"])
    comp = build_comparison(published, regenerated)
    assert comp["status"].tolist() == ["match_other_design", "match_other_design", "match", "match", "mismatch",
                                       "no_counterpart", "mismatch"]            # m7: neither design reproduces it
    assert comp["match"].tolist() == [False, False, True, True, False, False, False]
    assert comp["as_stated_value"].iloc[:3].round(4).tolist() == [0.32, 0.2113, 0.299]
    assert comp["as_stated_value"].iloc[3:].isna().sum() == 1                      # m4 has no stated design
    assert comp["regenerated"].iloc[:2].round(4).tolist() == [0.2301, 0.159]       # the producer's value stays in place
    # the columns the digest and the review filters use keep their names
    assert {"item", "analysis", "block", "estimator", "metric", "value", "source_line", "decimals", "note",
            "regenerated", "abs_diff", "status", "match"} <= set(build_comparison(
                published.assign(source_line="", note=""), regenerated).columns)
    # a stated design with nothing regenerated under it is an error, not a silent pass
    with pytest.raises(ValueError, match="stated"):
        build_comparison(published.assign(stated_analysis="nowhere"), regenerated)
    # a published frame without the column compares as before
    plain = build_comparison(published.drop(columns="stated_analysis"), regenerated)
    assert plain["status"].tolist() == ["match", "match", "match", "match", "mismatch", "no_counterpart", "mismatch"]


def test_d10_boundaries_count_what_half_open_utc_windows_do_to_local_days():
    from es.run_legacy import window_boundaries
    b = window_boundaries(_fake_panel(np.random.default_rng(9))).set_index(["window_start", "window_end", "market"])

    def get(start, end, market):
        r = b.loc[(start, end, market)]
        return (int(r["rows_utc"]), int(r["rows_local"]), int(r["extra_other_dates"]), int(r["missed_local_dates"]))

    # the synthetic panel runs hourly from 2023-01-01 00:00 UTC to 2024-12-31 23:00 UTC for 2 NYISO nodes and 1 MISO node
    # 2023: the UTC window opens at 19:00 on 31 Dec 2022 in New York (5 rows per node on the wrong date) and closes at
    # 19:00 on 31 Dec 2023, so the last 5 local hours of 2023 fall in the next window
    assert get("2023-01-01", "2024-01-01", "NYISO") == (17520, 17520, 10, 10)
    assert get("2023-01-01", "2024-01-01", "MISO") == (8760, 8760, 5, 5)             # fixed EST, UTC-5
    assert get("2024-01-01", "2025-01-01", "NYISO") == (17568, 17558, 10, 0)         # the data stop before the missed rows
    # 2024-H1 ends 20:00 EDT on 30 Jun in New York (4 hours per node missed) and 19:00 EST in MISO (5 hours)
    assert get("2024-01-01", "2024-07-01", "NYISO") == (8736, 8734, 10, 8)
    assert get("2024-01-01", "2024-07-01", "MISO") == (4368, 4368, 5, 5)
    assert get("2023-01-01", "2024-01-01", "CAISO") == (0, 0, 0, 0)                  # no CAISO rows in this panel
    used = b.loc[("2026-01-01", "2026-07-01", "NYISO"), "used_by"]
    assert "deploy:sealed26H1" in used and "deploy:25H1-26H1" in used and "deploy:frozen-26H1" in used
    assert "label_train:2024-2025" in b.loc[("2023-01-01", "2024-01-01", "NYISO"), "used_by"]


@pytest.mark.slow
def test_paper_reported_alpha_and_in_sample_cells_have_no_zero_activity(tmp_path, monkeypatch):
    """The alpha = 2.5% and 10% cells and the Fig. 2 in-sample cell that the paper reports, on the real panel."""
    from es import run_legacy as rl
    from es.analyses import LEGACY
    from es.data import build_panel, load_dev_raw
    monkeypatch.setattr(rl, "LEGACY_DIR", tmp_path)
    panel = build_panel(load_dev_raw())
    runs = sorted(k for k in rl.paper_runs(rl.read_published()) if k[3] == 42 and (k[4], k[5]) != ("P1", 0.05))
    assert [k[:2] + k[4:] for k in runs] == [
        ("2024-2025", "DART18", "P1", 0.025), ("2024-2025", "G1_24", "P2_all_train", 0.05),
        ("2024-2025", "LEV6", "P1", 0.025), ("25H1-26H1", "DART18", "P1", 0.025), ("25H1-26H1", "DART18", "P1", 0.1),
        ("25H1-26H1", "LEV6", "P1", 0.025), ("25H1-26H1", "LEV6", "P1", 0.1)]
    for analysis, block, estimator, seed, policy, alpha in runs:
        out = rl.legacy_summary(panel, LEGACY[analysis], block, estimator, seed, policy, alpha)
        assert len(out) == 33 and not out["zero_activity"].any(), (analysis, block, policy, alpha)


@pytest.mark.slow
def test_d10_boundary_counts_on_the_real_panel():
    """The counts quoted in CHANGES.md for D10 come from this function."""
    from es.data import build_panel, load_dev_raw
    from es.run_legacy import window_boundaries
    b = window_boundaries(build_panel(load_dev_raw())).set_index(["window_start", "window_end", "market"])

    def col(start, end, name):
        return {m: int(b.loc[(start, end, m), name]) for m in ("CAISO", "ISONE", "MISO", "NYISO")}

    assert col("2023-01-01", "2024-01-01", "missed_local_dates") == {"CAISO": 48, "ISONE": 45, "MISO": 40, "NYISO": 50}
    assert col("2023-01-01", "2024-01-01", "extra_other_dates") == {"CAISO": 0, "ISONE": 0, "MISO": 0, "NYISO": 0}
    assert col("2026-01-01", "2026-07-01", "extra_other_dates") == {"CAISO": 48, "ISONE": 45, "MISO": 40, "NYISO": 50}
    missed = col("2026-01-01", "2026-07-01", "missed_local_dates")
    assert missed == {"CAISO": 42, "ISONE": 36, "MISO": 40, "NYISO": 40} and sum(missed.values()) == 158
    assert sum(col("2026-01-01", "2026-07-01", "rows_utc").values()) == 143352      # the sealed analysis's deployment rows
