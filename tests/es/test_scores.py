import dataclasses
import inspect
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def panel():
    from es.data import build_panel, load_dev_raw
    return build_panel(load_dev_raw())


@pytest.mark.slow
def test_reproduces_as_opened_sealed_level_family_exactly(panel):
    """The 2026-07-27 as-opened output must be reproduced to float precision (FORENSICS §5b)."""
    from es.analyses import LEGACY
    from es.scores import score_analysis
    sc = score_analysis(panel, LEGACY["sealed26H1"], "LEV6")
    rows = []
    for (m, n), g in sc.groupby(["market", "node"]):
        tau = float(np.quantile(g.loc[g.split == "calib", "score"], 0.95))
        dep = g.loc[g.split == "deploy", "score"].to_numpy()
        rows.append((m, n, (dep >= tau).mean() / 0.05))
    ours = pd.DataFrame(rows, columns=["target_market", "target_node", "R"])
    saved = pd.read_csv(Path.home() / "dart-spread-data/holdout_2026H1/holdout_level.csv")
    saved = saved[saved.contrast == "temporal"][["target_market", "target_node", "R"]]
    j = ours.merge(saved, on=["target_market", "target_node"], suffixes=("", "_saved"))
    assert len(j) == 33
    assert np.max(np.abs(j.R - j.R_saved)) < 1e-12


def test_split_labels_and_unscaled_column(panel):
    from es.analyses import LEGACY
    from es.scores import score_analysis
    one = panel[panel["node"] == "N.Y.C."]
    # G1_24 contains the rolling column, so it is scaled inside the model: only a G1_24 run can
    # catch abs_mean_168h being read from the scaled frame.
    sc = score_analysis(one, LEGACY["sealed26H1"], "G1_24")
    assert set(sc["split"]) <= {"train", "calib", "deploy", "other"}
    assert {"train", "calib", "deploy"} <= set(sc["split"])
    raw_col = one.set_index("timestamp_utc").loc[sc["timestamp_utc"], "dart_spread_roll_abs_mean_168h_shift_48h"]
    assert np.allclose(sc["abs_mean_168h"].to_numpy(), raw_col.to_numpy(), equal_nan=True)


# ---------------------------------------------------------------------------------------------
# Synthetic panel for infeasible-node behaviour (real build_panel, no real data involved)
# ---------------------------------------------------------------------------------------------

def _synthetic_analysis(analysis_id: str = "syn"):
    from es.analyses import Analysis
    from es.windows import UtcWindow as U
    train = U("2023-01-01", "2023-02-15")
    return Analysis(analysis_id, train, U("2023-02-15", "2023-03-15"), U("2023-03-15", "2023-04-01"),
                    train, "rolling", "legacy")


@pytest.fixture(scope="module")
def syn_panel():
    """Four NYISO nodes, hourly, built through the real ``build_panel``.

    SYN_OK:    2023-01-01 to 2023-04-01; plenty of training hours (1080 in the train window).
    SYN_SHORT: starts 100 h before the train window ends, so it has exactly 100 training hours and
               the top-5% label gives it exactly 5 positives.
    SYN_FLAT:  as SYN_OK but da_lmp == rt_lmp, so |DART| is constant, every row is a positive and
               there is a single class.
    SYN_TINY:  only 120 h in total (inside the calib window): under a 168 h-lag block none of its
               rows has complete features, and in any block it has no training rows.
    """
    from es.data import build_panel
    rng = np.random.default_rng(7)
    end = pd.Timestamp("2023-04-01", tz="UTC")

    def node(name, start, flat=False, hours=None):
        ts = (pd.date_range(start, end, freq="h", inclusive="left") if hours is None
              else pd.date_range(start, periods=hours, freq="h"))
        da = 30 + rng.normal(0, 3, len(ts))
        rt = da.copy() if flat else da + rng.normal(0, 4, len(ts))
        return pd.DataFrame({"market": "NYISO", "node": name, "node_type": "zone",
                             "timestamp_utc": ts, "da_lmp": da, "rt_lmp": rt})

    jan1 = pd.Timestamp("2023-01-01", tz="UTC")
    short_start = pd.Timestamp("2023-02-15", tz="UTC") - pd.Timedelta(hours=100)
    raw = pd.concat([node("SYN_OK", jan1), node("SYN_SHORT", short_start),
                     node("SYN_FLAT", jan1, flat=True),
                     node("SYN_TINY", pd.Timestamp("2023-03-01", tz="UTC"), hours=120)], ignore_index=True)
    return build_panel(raw)


def test_feasibility_reports_each_gate_reason(syn_panel):
    from es.scores import feasibility
    out = feasibility(syn_panel, _synthetic_analysis(), "G6")
    assert list(out.columns) == ["market", "node", "n_train", "n_pos", "feasible", "reason"]
    f = out.set_index("node")
    assert f.loc["SYN_OK", "reason"] == "ok" and bool(f.loc["SYN_OK", "feasible"])
    assert f.loc["SYN_OK", "n_pos"] >= 20
    assert (f.loc["SYN_SHORT", "n_train"], f.loc["SYN_SHORT", "n_pos"]) == (100, 5)
    assert f.loc["SYN_SHORT", "reason"] == "lt_20_positives" and not f.loc["SYN_SHORT", "feasible"]
    # one class wins over the positives test, as in fit_model: 1080 positives, still infeasible
    assert f.loc["SYN_FLAT", "reason"] == "one_class" and not f.loc["SYN_FLAT", "feasible"]
    assert f.loc["SYN_FLAT", "n_pos"] == f.loc["SYN_FLAT", "n_train"] >= 20


@pytest.mark.parametrize("n_zero, n_one, reason", [
    (0, 0, "no_train_rows"),
    (30, 0, "one_class"),            # all negative
    (0, 10, "one_class"),            # all positive and under 20: the class check comes first
    (0, 50, "one_class"),            # all positive and over 20: still a single class
    (30, 5, "lt_20_positives"),
    (30, 19, "lt_20_positives"),
    (30, 20, "ok"),
])
def test_gate_mirrors_src_transfer_fit_model(n_zero, n_one, reason):
    from es.scores import _gate, xgb_factory
    from src import transfer
    rng = np.random.default_rng(0)
    n = n_zero + n_one
    train = pd.DataFrame({"x": rng.normal(size=n), "label": np.array([0.0] * n_zero + [1.0] * n_one)})
    assert _gate(train) == (n, n_one, reason)
    fitted = transfer.fit_model(xgb_factory(0), train, ["x"], "label")
    assert (fitted is not None) == (reason == "ok")


def test_score_analysis_warns_and_omits_infeasible_nodes(syn_panel):
    from es.scores import feasibility, score_analysis
    an = _synthetic_analysis()
    with pytest.warns(UserWarning) as rec:
        sc = score_analysis(syn_panel, an, "G6")
    msgs = [str(w.message) for w in rec if "not scored" in str(w.message)]
    assert len(msgs) == 1
    assert "NYISO/SYN_SHORT (lt_20_positives: n_train=100, n_pos=5)" in msgs[0]
    assert "NYISO/SYN_FLAT (one_class" in msgs[0]
    assert "SYN_OK" not in msgs[0]
    assert set(sc["node"]) == {"SYN_OK"}
    # the feasibility report predicts exactly which nodes the pipeline scores
    f = feasibility(syn_panel, an, "G6")
    assert set(f.loc[f["feasible"], "node"]) == set(sc["node"])


def test_score_analysis_raises_descriptive_error_when_no_node_is_feasible(syn_panel):
    from es.scores import score_analysis
    only_infeasible = syn_panel[syn_panel["node"].isin(["SYN_SHORT", "SYN_FLAT"])]
    with pytest.raises(ValueError, match="no node is feasible") as exc:
        score_analysis(only_infeasible, _synthetic_analysis(), "G6")
    msg = str(exc.value)
    assert "NYISO/SYN_SHORT (lt_20_positives" in msg and "NYISO/SYN_FLAT (one_class" in msg


def test_node_without_complete_feature_rows_is_reported_not_lost(syn_panel):
    from es.scores import feasibility, score_analysis
    an = _synthetic_analysis()
    # DART3 needs 168 h of history. SYN_TINY has 120 h in total, so none of its rows has complete
    # features and it never reaches the model loop; SYN_SHORT has later rows with complete features,
    # but none inside the train window. Both must still be reported.
    f = feasibility(syn_panel, an, "DART3").set_index("node")
    for name in ("SYN_TINY", "SYN_SHORT"):
        assert f.loc[name, ["n_train", "n_pos", "reason"]].tolist() == [0, 0, "no_train_rows"]
    assert f.loc["SYN_OK", "reason"] == "ok" and f.loc["SYN_FLAT", "reason"] == "one_class"
    with pytest.warns(UserWarning) as rec:
        sc = score_analysis(syn_panel, an, "DART3")
    msgs = [str(w.message) for w in rec if "not scored" in str(w.message)]
    assert len(msgs) == 1
    assert "NYISO/SYN_TINY (no_train_rows: n_train=0, n_pos=0)" in msgs[0]
    assert "NYISO/SYN_SHORT (no_train_rows: n_train=0, n_pos=0)" in msgs[0]
    assert set(sc["node"]) == {"SYN_OK"}


def test_overlapping_windows_are_rejected(syn_panel):
    from es.analyses import Analysis
    from es.scores import score_analysis
    from es.windows import UtcWindow as U
    bad = Analysis("syn_overlap", U("2023-01-01", "2023-02-20"), U("2023-02-15", "2023-03-15"),
                   U("2023-03-15", "2023-04-01"), U("2023-01-01", "2023-02-15"), "rolling", "legacy")
    with pytest.raises(ValueError, match="train and calib windows overlap"):
        score_analysis(syn_panel, bad, "G6")


def test_split_labels_partition_the_synthetic_windows(syn_panel):
    from es.scores import score_analysis
    with pytest.warns(UserWarning):
        sc = score_analysis(syn_panel, _synthetic_analysis(), "G6")
    ts = sc["timestamp_utc"]
    train_end, calib_end = pd.Timestamp("2023-02-15", tz="UTC"), pd.Timestamp("2023-03-15", tz="UTC")
    expected = np.select([ts < train_end, ts < calib_end], ["train", "calib"], default="deploy")
    assert (sc["split"].to_numpy() == expected).all()


def test_label_alpha_is_the_label_quantile_and_positional_order_is_kept(syn_panel):
    from es.scores import SCORES_VERSION, cached_scores, feasibility, score_analysis
    assert list(inspect.signature(score_analysis).parameters) == [
        "panel", "analysis", "block", "estimator", "seed", "label_alpha"]
    assert list(inspect.signature(feasibility).parameters) == ["panel", "analysis", "block", "label_alpha"]
    assert list(inspect.signature(cached_scores).parameters) == [
        "panel", "analysis", "block", "estimator", "seed", "label_alpha", "version"]
    assert inspect.signature(cached_scores).parameters["version"].default == SCORES_VERSION == "v2"
    # a 25% label quantile gives the 100-hour node 25 positives, which clears the 20-positive gate
    f = feasibility(syn_panel, _synthetic_analysis(), "G6", 0.25).set_index("node")
    assert f.loc["SYN_SHORT", ["n_pos", "feasible", "reason"]].tolist() == [25, True, "ok"]
    assert f.loc["SYN_FLAT", "reason"] == "one_class"


# ---------------------------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------------------------

@pytest.fixture
def throwaway_id():
    """A unique analysis id; every cache entry created under it is deleted afterwards."""
    from src.paths import DEV
    analysis_id = f"tmp_test_scores_{uuid.uuid4().hex[:10]}"
    yield analysis_id
    for entry in DEV.cache.glob(f"es_scores__{analysis_id}__*.parquet"):
        entry.unlink()


def test_cache_name_depends_on_panel_and_analysis_definition(panel):
    from es.analyses import LEGACY
    from es.scores import cache_name
    from es.windows import UtcWindow
    an = LEGACY["sealed26H1"]
    one = panel[panel["node"] == "N.Y.C."]
    two = panel[panel["node"].isin(["N.Y.C.", ".H.INTERNAL_HUB"])]
    name = cache_name(one, an, "DART3")
    assert cache_name(one.copy(), an, "DART3") == name                       # stable
    assert cache_name(two, an, "DART3") != name                              # one node vs two nodes
    edited = dataclasses.replace(an, calib=UtcWindow("2025-01-01", "2025-07-01"))
    assert edited.id == an.id and cache_name(one, edited, "DART3") != name   # windows edited, same id
    moved = one.copy()
    moved.iloc[10, moved.columns.get_loc("dart_spread")] += 1.0
    assert cache_name(moved, an, "DART3") != name                            # a spread changed
    assert cache_name(one, an, "LEV6") != name                               # another block
    assert cache_name(one, an, "DART3", label_alpha=0.1) != name
    assert cache_name(one, an, "DART3", "logit") != name and cache_name(one, an, "DART3", seed=7) != name


def test_cached_scores_equals_score_analysis_after_cold_read(panel, throwaway_id, capsys):
    from es.analyses import LEGACY
    from es.scores import CACHED_COLUMNS, cached_scores, score_analysis
    from src.paths import DEV
    an = dataclasses.replace(LEGACY["sealed26H1"], id=throwaway_id)
    one = panel[panel["node"] == "N.Y.C."]
    two = panel[panel["node"].isin(["N.Y.C.", ".H.INTERNAL_HUB"])]
    expected = score_analysis(one, an, "DART3")
    first = cached_scores(one, an, "DART3")        # miss: builds and writes
    capsys.readouterr()
    cold = cached_scores(one, an, "DART3")         # hit: read back from the parquet file
    assert "[cache] hit" in capsys.readouterr().out
    pd.testing.assert_frame_equal(first, expected)
    pd.testing.assert_frame_equal(cold, expected)
    entries = list(DEV.cache.glob(f"es_scores__{throwaway_id}__*.parquet"))
    assert len(entries) == 1 and pd.read_parquet(entries[0]).columns.tolist() == CACHED_COLUMNS
    # a different panel under the same arguments gets its own entry, not the one-node one
    both = cached_scores(two, an, "DART3")
    assert set(both["node"]) == {"N.Y.C.", ".H.INTERNAL_HUB"}
    assert len(list(DEV.cache.glob(f"es_scores__{throwaway_id}__*.parquet"))) == 2
    pd.testing.assert_frame_equal(both, score_analysis(two, an, "DART3"))


def test_cached_scores_repeats_the_exclusion_warning_on_a_cache_hit(syn_panel, throwaway_id):
    from es.scores import cached_scores
    an = _synthetic_analysis(throwaway_id)
    with pytest.warns(UserWarning, match=r"NYISO/SYN_SHORT \(lt_20_positives"):   # miss: score_analysis warns
        first = cached_scores(syn_panel, an, "G6")
    with pytest.warns(UserWarning, match=r"cached entry.*NYISO/SYN_SHORT \(lt_20_positives"):  # hit: replayed
        again = cached_scores(syn_panel, an, "G6")
    pd.testing.assert_frame_equal(first, again)
    assert set(again["node"]) == {"SYN_OK"}


def test_rejoin_refuses_a_panel_that_does_not_match_the_cached_rows(syn_panel):
    from es.scores import CACHED_COLUMNS, _rejoin, score_analysis
    with pytest.warns(UserWarning):
        slim = score_analysis(syn_panel, _synthetic_analysis(), "G6")[CACHED_COLUMNS]
    assert len(_rejoin(slim, syn_panel)) == len(slim)       # sanity: the matching panel round-trips
    # cached rows the panel does not have: refuse rather than fill with NaN
    shorter = syn_panel[syn_panel["timestamp_utc"] >= pd.Timestamp("2023-01-02", tz="UTC")]
    with pytest.raises(RuntimeError, match="not in the panel"):
        _rejoin(slim, shorter)
    # duplicate keys in the panel would multiply rows in the join: refuse
    duplicated = pd.concat([syn_panel, syn_panel.iloc[:5]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        _rejoin(slim, duplicated)


def test_cached_scores_with_no_feasible_node_raises_and_writes_nothing(syn_panel, throwaway_id):
    from es.scores import cached_scores
    from src.paths import DEV
    only_infeasible = syn_panel[syn_panel["node"].isin(["SYN_SHORT", "SYN_FLAT"])]
    with pytest.raises(ValueError, match="no node is feasible"):
        cached_scores(only_infeasible, _synthetic_analysis(throwaway_id), "G6")
    assert list(DEV.cache.glob(f"es_scores__{throwaway_id}__*.parquet")) == []
