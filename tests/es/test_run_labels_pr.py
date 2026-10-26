from pathlib import Path

import numpy as np
import pandas as pd


def test_precision_recall_definitions():
    from es.run_labels_pr import precision_recall
    a = pd.DataFrame({"market": "M", "node": "A", "alert": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                      "label": [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                      "dart_spread": [9, 1, 8, 0, 0, 0, 0, 0, 0, 0]}).astype({"alert": bool})
    pr = precision_recall(a, alpha=0.2).iloc[0]
    assert np.isclose(pr.precision_q, 0.5) and np.isclose(pr.recall_q, 0.5)
    assert np.isclose(pr.precision_eval, 0.5) and np.isclose(pr.precision_lift, 2.5)


# ---------------------------------------------------------------------------------------------
# Tests added beyond the brief. Synthetic data only: nothing here reads prices or fits a model.
# ---------------------------------------------------------------------------------------------

def test_output_columns_are_the_brief_columns_plus_prevalence_and_the_nan_count():
    from es.run_labels_pr import precision_recall
    a = pd.DataFrame({"market": "M", "node": "A", "alert": [True, False, False, False],
                      "label": [1, 0, 0, 0], "dart_spread": [4.0, 3.0, 2.0, 1.0]})
    assert list(precision_recall(a, alpha=0.25).columns) == [
        "market", "node", "precision_q", "recall_q", "precision_eval", "recall_eval",
        "prevalence_eval", "n_nan_outcome", "precision_lift"]


def test_precision_lift_divides_by_realized_prevalence_not_alpha():
    """Ruling 12. Seven rows at alpha 0.2: the 0.8 quantile of |DART| 1..7 is 5 + 0.8 * (6 - 5) = 5.8,
    so only |DART| 6 and 7 are evaluation positives, a prevalence of 2/7 and not 0.2."""
    from es.run_labels_pr import precision_recall
    a = pd.DataFrame({"market": "M", "node": "A",
                      "dart_spread": [-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0],
                      "label": [0, 0, 0, 0, 1, 1, 1],
                      "alert": [False, False, False, True, False, True, True]})
    pr = precision_recall(a, alpha=0.2).iloc[0]
    assert np.isclose(pr.prevalence_eval, 2 / 7)
    assert np.isclose(pr.precision_eval, 2 / 3)          # alerts on |DART| 4, 6, 7; two are positive
    assert np.isclose(pr.recall_eval, 1.0)               # both positives are alerted
    assert np.isclose(pr.precision_lift, 7 / 3)          # (2/3) / (2/7); dividing by alpha gives 10/3
    assert np.isclose(pr.precision_q, 2 / 3) and np.isclose(pr.recall_q, 2 / 3)
    assert pr.n_nan_outcome == 0                         # nothing is NaN here


def test_rows_with_an_unknown_outcome_leave_every_metric():
    """A NaN DART has a NaN label: neither a hit nor a miss, so it is not a false positive when it alerts and
    it is not in the prevalence denominator. (da_lmp - rt_lmp is NaN wherever one price is missing.)"""
    from es.run_labels_pr import precision_recall
    nan = float("nan")
    a = pd.DataFrame({"market": "M", "node": "A",
                      "dart_spread": [10.0, -9.0, 8.0, 7.0, -6.0, 5.0, 4.0, -3.0, nan, nan],
                      "label": [1, 0, 1, 1, 0, 0, 0, 0, nan, nan],
                      "alert": [True, False, True, False, False, False, False, False, True, False]})
    # eight known |DART| 3..10: the 0.75 quantile is 8 + 0.25 * (9 - 8) = 8.25, so 9 and 10 are positive
    pr = precision_recall(a, alpha=0.25).iloc[0]
    assert np.isclose(pr.prevalence_eval, 2 / 8)         # 2/10 if the two unknown rows were counted
    assert np.isclose(pr.precision_eval, 1 / 2)          # 1/3 if the unknown alert were a false positive
    assert np.isclose(pr.recall_eval, 1 / 2)
    assert np.isclose(pr.precision_lift, 2.0)
    assert np.isclose(pr.precision_q, 1.0)               # 2/3 if the unknown alert were a false positive
    assert np.isclose(pr.recall_q, 2 / 3)
    assert pr.n_nan_outcome == 2                         # the two unknown rows, each counted once


def test_a_row_whose_label_alone_is_unknown_is_dropped_from_the_q_metrics_only():
    """Each metric drops only the rows its own label cannot score. Rows 0 and 1 have a DART but no training-q
    label: the q metrics skip them, the evaluation metrics keep them. (The pipeline loses a row's DART and
    label together; this pins the two masks themselves.)"""
    from es.run_labels_pr import precision_recall
    nan = float("nan")
    a = pd.DataFrame({"market": "M", "node": "A",
                      "dart_spread": [10.0, -9.0, 8.0, 7.0, -6.0, 5.0, 4.0, -3.0],
                      "label": [nan, nan, 1, 0, 1, 1, 0, 0],
                      "alert": [True, False, True, True, False, False, False, False]})
    # |DART| 3..10: the 0.75 quantile is 8.25, so rows 0 and 1 (10 and 9) are the evaluation positives
    pr = precision_recall(a, alpha=0.25).iloc[0]
    assert np.isclose(pr.prevalence_eval, 2 / 8)         # 0/6 if rows 0 and 1 were dropped: both positives
    assert np.isclose(pr.precision_eval, 1 / 3)          # alerts on rows 0, 2, 3; only row 0 is positive
    assert np.isclose(pr.recall_eval, 1 / 2)             # row 0 of the positives 0 and 1
    assert np.isclose(pr.precision_lift, 4 / 3)          # (1/3) / (1/4)
    assert np.isclose(pr.precision_q, 1 / 2)             # alerts on rows 2, 3; 1/3 if the alert on row 0 counted
    assert np.isclose(pr.recall_q, 1 / 3)                # row 2 of the positives 2, 4, 5
    assert pr.n_nan_outcome == 2


def test_a_row_whose_dart_alone_is_unknown_is_dropped_from_the_evaluation_metrics_only():
    """The mirror image. Rows 8 and 9 have a training-q label but no DART: the evaluation metrics skip them
    (and the top-alpha threshold ignores them), the q metrics keep them."""
    from es.run_labels_pr import precision_recall
    nan = float("nan")
    a = pd.DataFrame({"market": "M", "node": "A",
                      "dart_spread": [10.0, -9.0, 8.0, 7.0, -6.0, 5.0, 4.0, -3.0, nan, nan],
                      "label": [1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                      "alert": [True, True, True, False, False, False, False, False, True, False]})
    # the eight known |DART| 3..10 give the 0.75 quantile 8.25, so rows 0 and 1 are the evaluation positives
    pr = precision_recall(a, alpha=0.25).iloc[0]
    assert np.isclose(pr.prevalence_eval, 2 / 8)         # 2/10 if the two unknown rows were counted
    assert np.isclose(pr.precision_eval, 2 / 3)          # alerts on rows 0, 1, 2; rows 0 and 1 are positive
    assert np.isclose(pr.recall_eval, 1.0)
    assert np.isclose(pr.precision_lift, 8 / 3)          # (2/3) / (1/4)
    assert np.isclose(pr.precision_q, 1 / 2)             # alerts on rows 0, 1, 2, 8; 2/3 with row 8 gone
    assert np.isclose(pr.recall_q, 2 / 3)                # rows 0 and 2 of the positives 0, 2, 9; 1.0 with row 9 gone
    assert pr.n_nan_outcome == 2


def test_n_nan_outcome_counts_each_row_with_a_nan_dart_or_label_once_per_node():
    from es.run_labels_pr import precision_recall
    nan = float("nan")
    a = pd.DataFrame({"market": "M", "node": "A",
                      "dart_spread": [5.0, nan, 3.0, nan, 2.0, 1.0],      # rows 1 and 3 have no DART
                      "label": [1, 0, nan, nan, 0, 0],                    # rows 2 and 3 have no label
                      "alert": [True, False, True, False, False, False]})
    b = a.assign(node="B", dart_spread=[5.0, 4.0, 3.0, 2.0, 1.0, 0.5], label=[1, 0, 0, 0, 0, 0])
    pr = precision_recall(pd.concat([a, b]), alpha=0.25).set_index("node")
    assert pr.loc["A", "n_nan_outcome"] == 3        # DART only, label only, and both: three rows, not four
    assert pr.loc["B", "n_nan_outcome"] == 0


def test_undefined_ratios_are_nan_and_each_node_has_its_own_threshold():
    from es.run_labels_pr import precision_recall
    base = pd.DataFrame({"alert": [False] * 10, "label": [1, 0, 1] + [0] * 7,
                         "dart_spread": [9.0, 1.0, 8.0] + [0.0] * 7})
    quiet = base.assign(market="M", node="A")                       # never alerts
    no_pos = base.assign(market="M", node="B", label=0, alert=[True] + [False] * 9,
                         dart_spread=base["dart_spread"] * 100)     # no training-q positive; 100x the scale
    stacked = pd.concat([no_pos, quiet])                            # duplicate row labels, as stacked frames have
    pr = precision_recall(stacked, alpha=0.2).set_index("node")
    assert list(pr.index) == ["A", "B"]
    a, b = pr.loc["A"], pr.loc["B"]
    assert np.isnan(a.precision_q) and np.isnan(a.precision_eval) and np.isnan(a.precision_lift)
    assert a.recall_q == 0.0 and a.recall_eval == 0.0
    assert b.precision_q == 0.0 and np.isnan(b.recall_q)
    # Node A's top-alpha threshold is its own (2.4), not the 240 of node B, so A keeps its two positives.
    assert np.isclose(a.prevalence_eval, 0.2) and np.isclose(b.prevalence_eval, 0.2)
    assert np.isclose(b.precision_eval, 1.0) and np.isclose(b.recall_eval, 0.5)
    assert np.isclose(b.precision_lift, 5.0)
    assert a.n_nan_outcome == 0 and b.n_nan_outcome == 0


def test_lift_is_nan_when_no_row_is_an_evaluation_positive(monkeypatch):
    """Not reachable with label_eval_top_alpha (its maximum is always at or above its own quantile), but a
    zero prevalence must give a NaN lift and not a ZeroDivisionError that aborts a long run."""
    from es import run_labels_pr as mod
    monkeypatch.setattr(mod, "label_eval_top_alpha", lambda rows, alpha: pd.Series(0, index=rows.index))
    a = pd.DataFrame({"market": "M", "node": "A", "alert": [True, False], "label": [1, 0],
                      "dart_spread": [2.0, 1.0]})
    pr = mod.precision_recall(a, alpha=0.5).iloc[0]
    assert pr.prevalence_eval == 0.0 and pr.precision_eval == 0.0
    assert np.isnan(pr.recall_eval) and np.isnan(pr.precision_lift)


def test_default_output_directory():
    from es import run_labels_pr
    from es.config import DEV_ROOT
    assert run_labels_pr.PR_DIR.relative_to(DEV_ROOT) == Path("outputs") / "es" / "labels_pr"


# ---------------------------------------------------------------------------------------------
# main(): wired to fake scores, so the full grid runs in seconds and no price data is read.
# ---------------------------------------------------------------------------------------------

CONTROL_BLOCKS = ("LEV6", "DART18", "G1_24", "DART3")
#: Shifts of a node's deployment scores. At shift d a node fires 5 + d alerts, so R = 1 + d / 5: -4 breaks the
#: failure band from below, 0 and 3 stay inside [0.5, 2], and 6 and above break it from above.
DRIFTS = (-4, 0, 3, 6, 9, 12, 15)


def _fake_scores(drifts):
    """One node per entry of ``drifts`` (node -> shift), each with 100 calibration and 100 deployment rows.
    The score and |DART| both run 0..99, so the calibration cutoff is 94.05 and the label (the top 5%) is
    |DART| >= 95. Shifting a node's deployment scores by d fires 5 + d alerts, 5 + min(d, 0) of them labelled."""
    t0 = pd.Timestamp("2025-01-01", tz="UTC")
    base = np.arange(100.0)
    parts = []
    for node, d in drifts.items():
        for split, offset in (("calib", 0), ("deploy", 100)):
            parts.append(pd.DataFrame({
                "market": "M", "node": node, "split": split,
                "timestamp_utc": t0 + pd.to_timedelta(offset + np.arange(100), unit="h"),
                "score": base + (d if split == "deploy" else 0.0),
                "label": (base >= 95).astype(float), "dart_spread": base}))
    return pd.concat(parts, ignore_index=True)


def _assert_closed_form_pr(frame, drift_of):
    """Check every row of a per-node precision/recall frame against the fake world. A node shifted by d
    fires 5 + d alerts and 5 + min(d, 0) of them are hits, out of 5 positives; ``drift_of(row)`` gives d."""
    d = np.array([drift_of(r) for r in frame.itertuples()])
    alerts, hits = 5 + d, 5 + np.minimum(d, 0)
    assert np.allclose(frame["precision_q"], hits / alerts) and np.allclose(frame["precision_eval"], hits / alerts)
    assert np.allclose(frame["recall_q"], hits / 5) and np.allclose(frame["recall_eval"], hits / 5)
    assert np.allclose(frame["prevalence_eval"], 0.05)
    assert np.allclose(frame["precision_lift"], hits / alerts / 0.05)
    assert (frame["n_nan_outcome"] == 0).all()                      # the fake scores have no NaN outcome


def test_main_runs_the_brief_grid_and_writes_all_three_files(monkeypatch, tmp_path):
    from es import run_labels_pr as mod
    from es.analyses import ANALOGS, LABEL_CONTROL, LEGACY
    from es.blocks import BLOCKS

    grid = [(a.id, b, e) for a in list(LEGACY.values()) + list(ANALOGS.values())
            for b in BLOCKS for e in ("xgb", "logit")]
    control = [(a.id, b, "xgb") for a in LABEL_CONTROL.values() for b in CONTROL_BLOCKS]
    assert len(grid) == 12 * 10 * 2 and len(control) == 4 * 4
    # Node N1 never drifts; N2, N3 and N4 take a different shift in every cell, so a result filed under the
    # wrong analysis, block or estimator shows up as a wrong number. A cell has two, three or four nodes (as
    # when cached_scores drops infeasible nodes), so the node count differs from cell to cell.
    def cell_drifts(i):
        shifts = {"N1": 0, "N2": DRIFTS[i % 7], "N3": DRIFTS[(i + 3) % 7], "N4": DRIFTS[(i + 5) % 7]}
        return dict(list(shifts.items())[:2 + i % 3])

    drifts = {cell: cell_drifts(i) for i, cell in enumerate(grid + control)}
    calls = []

    def fake_cached_scores(panel, analysis, block, estimator, seed, label_alpha=0.05):
        assert panel == ("panel", "raw")
        calls.append((analysis.id, block, estimator, seed, label_alpha))
        return _fake_scores(drifts[(analysis.id, block, estimator)])

    monkeypatch.setattr(mod, "PR_DIR", tmp_path / "labels_pr")
    monkeypatch.setattr(mod, "ensure_dirs", lambda: None)
    monkeypatch.setattr(mod, "load_dev_raw", lambda: "raw")
    monkeypatch.setattr(mod, "build_panel", lambda raw: ("panel", raw))
    monkeypatch.setattr(mod, "cached_scores", fake_cached_scores)
    mod.main()

    assert sorted(c[:3] for c in calls) == sorted(grid + control)         # each cell exactly once
    assert {c[3:] for c in calls} == {(42, 0.05)}                         # seed 42, the default 5% label

    pr = pd.read_csv(tmp_path / "labels_pr" / "pr_all.csv")
    assert list(pr.columns) == ["market", "node", "precision_q", "recall_q", "precision_eval",
                                "recall_eval", "prevalence_eval", "n_nan_outcome", "precision_lift",
                                "analysis", "block", "estimator"]
    assert len(pr) == sum(len(drifts[cell]) for cell in grid)
    assert not pr.duplicated(["analysis", "block", "estimator", "node"]).any()
    assert set(zip(pr["analysis"], pr["block"], pr["estimator"])) == set(grid)
    _assert_closed_form_pr(pr, lambda r: drifts[(r.analysis, r.block, r.estimator)][r.node])

    ctl = pd.read_csv(tmp_path / "labels_pr" / "label_control_2x2.csv")
    assert list(ctl.columns) == ["analysis", "block", "failures", "n_nodes", "median_R"]
    assert sorted(zip(ctl["analysis"], ctl["block"])) == sorted((a, b) for a, b, _ in control)
    rs = [np.array([(5 + d) / 5 for d in drifts[(a, b, "xgb")].values()])    # R of each node in the cell
          for a, b in zip(ctl["analysis"], ctl["block"])]
    assert list(ctl["n_nodes"]) == [len(r) for r in rs]
    assert list(ctl["failures"]) == [int(((r < 0.5) | (r > 2)).sum()) for r in rs]
    assert np.allclose(ctl["median_R"], [np.median(r) for r in rs])
    # the fake cells have 2, 3 and 4 nodes, reach both tails of the band, and the median is not always the mean
    assert set(ctl["n_nodes"]) == {2, 3, 4}
    assert any((r < 0.5).any() for r in rs) and any((r > 2).any() for r in rs)
    assert any(np.median(r) != r.mean() for r in rs)

    # Ruling 43: spec 5.4 ("item 3 also applies here") wants precision and recall for the control cells too,
    # from the alerts already computed (the cells were requested once each, above), in a file of their own.
    cpr = pd.read_csv(tmp_path / "labels_pr" / "label_control_pr.csv")
    assert list(cpr.columns) == ["market", "node", "precision_q", "recall_q", "precision_eval",
                                 "recall_eval", "prevalence_eval", "n_nan_outcome", "precision_lift",
                                 "analysis", "block"]
    assert len(cpr) == sum(len(drifts[cell]) for cell in control)
    assert not cpr.duplicated(["analysis", "block", "node"]).any()
    assert set(zip(cpr["analysis"], cpr["block"])) == {(a, b) for a, b, _ in control}
    _assert_closed_form_pr(cpr, lambda r: drifts[(r.analysis, r.block, "xgb")][r.node])
    sizes = cpr.groupby(["analysis", "block"]).size()                # the same nodes as the 2x2 file counts
    assert [sizes[(a, b)] for a, b in zip(ctl["analysis"], ctl["block"])] == list(ctl["n_nodes"])
    assert sorted(f.name for f in (tmp_path / "labels_pr").iterdir()) == [
        "label_control_2x2.csv", "label_control_pr.csv", "pr_all.csv"]
