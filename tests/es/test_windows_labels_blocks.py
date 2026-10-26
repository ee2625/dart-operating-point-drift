import datetime as dt

import numpy as np
import pandas as pd


def _frame():
    ts = pd.date_range("2025-06-30 00:00", periods=72, freq="h", tz="UTC")
    return pd.DataFrame({"market": "NYISO", "node": "A", "timestamp_utc": ts,
                         "settle_date": ts.tz_convert("America/New_York").date,
                         "dart_spread": np.arange(72, dtype=float) - 36})


def test_date_window_is_inclusive_local():
    from es.windows import DateWindow
    f = _frame()
    m = DateWindow("2025-07-01", "2025-07-01").mask(f)
    assert set(f.loc[m, "settle_date"]) == {dt.date(2025, 7, 1)}
    assert m.sum() == 24


def test_utc_window_is_half_open():
    from es.windows import UtcWindow
    f = _frame()
    m = UtcWindow("2025-07-01", "2025-07-02").mask(f)
    assert m.sum() == 24
    assert f.loc[m, "timestamp_utc"].min() == pd.Timestamp("2025-07-01", tz="UTC")


def test_label_training_q_matches_v2_add_label_rule():
    from es.labels import label_training_q
    f = _frame()
    train = f["timestamp_utc"] < pd.Timestamp("2025-07-01", tz="UTC")
    label, q = label_training_q(f, train, alpha=0.05)
    expected_q = np.quantile(np.abs(f.loc[train, "dart_spread"]), 0.95)
    assert np.isclose(q[("NYISO", "A")], expected_q)
    assert (label == (f["dart_spread"].abs() >= expected_q).astype(float)).all()


def test_eval_label_has_alpha_prevalence():
    from es.labels import label_eval_top_alpha
    rng = np.random.default_rng(0)
    rows = pd.DataFrame({"market": "M", "node": "A", "dart_spread": rng.normal(size=2000)})
    lab = label_eval_top_alpha(rows, alpha=0.05)
    assert abs(lab.mean() - 0.05) < 0.002


def test_blocks_match_regime_ablation_definitions():
    from es.blocks import BLOCKS, G1, G6, features
    sizes = {k: len(v) for k, v in BLOCKS.items()}
    assert sizes == {"G6": 0, "LEV6": 6, "MID3": 3, "DART3": 3, "MIDDART6": 6, "ROLL15": 15,
                     "ROLL_DISP9": 9, "ROLL_LOC6": 6, "DART18": 18, "G1_24": 24}
    assert len(G6) == 10 and features("G1_24") == G6 + G1


def test_g6_order_matches_cell_builder():
    from es.blocks import G6
    from v2.cell_builder import g6_features
    assert G6 == g6_features()


def test_analyses_are_disjoint_and_ordered():
    from es.analyses import ANALOGS, SEALED
    for a in list(ANALOGS.values()) + list(SEALED.values()):
        assert a.train.end < a.calib.start <= a.calib.end < a.deploy.start
    assert max(a.deploy.end for a in ANALOGS.values()) == "2026-06-30"
