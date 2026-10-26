"""Spike labels. q is the node's (1-alpha) quantile of |DART| on training rows only
(linear interpolation), the rule of v2/panel_prep.py:add_label; NaN |DART| stays NaN."""
from __future__ import annotations

import numpy as np
import pandas as pd


def label_training_q(panel: pd.DataFrame, label_mask: pd.Series, alpha: float = 0.05):
    absd = panel["dart_spread"].abs()
    train = panel[label_mask.to_numpy()]
    q = train.assign(_a=absd[label_mask.to_numpy()]).groupby(["market", "node"])["_a"].quantile(1 - alpha)
    keys = pd.MultiIndex.from_arrays([panel["market"], panel["node"]])
    qv = q.reindex(keys).to_numpy(dtype=float)
    a = absd.to_numpy(dtype=float)
    label = np.where(np.isnan(a) | np.isnan(qv), np.nan, (a >= qv).astype(float))
    return pd.Series(label, index=panel.index, name="label"), q


def label_eval_top_alpha(rows: pd.DataFrame, alpha: float = 0.05) -> pd.Series:
    """Evaluation-only label: top-alpha |DART| within ``rows`` for each node (spec §5.3)."""
    absd = rows["dart_spread"].abs()
    thr = absd.groupby([rows["market"], rows["node"]]).transform(lambda s: s.quantile(1 - alpha))
    return (absd >= thr).astype(int)
