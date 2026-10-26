"""Per-node cutoffs fixed before a sealed pull (spec v3 §6.3, PC; controller Ruling 30). They use
training and calibration rows only, so the sealed run recomputes them exactly and reports any
difference."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .scores import score_analysis

CUTOFF_BLOCKS = ("G1_24", "DART3", "LEV6", "ROLL_DISP9")


def _tau(values, alpha):
    values = np.asarray(values, dtype=float)
    return float(np.quantile(values, 1 - alpha)) if values.size else float("nan")   # es.policies' rule


def cutoffs_from_scores(scores, analysis, block, alpha=0.05):
    months = analysis.deploy.months
    rows = []
    # observed=True: the sealed run passes slim frames with categorical keys (Ruling 60); without it pandas
    # would add one row per unobserved (market, node) pair, with NaN cutoffs.
    for (market, node), g in scores.groupby(["market", "node"], sort=True, observed=True):
        cal = g.loc[g["split"] == "calib", "score"]
        tr = g[g["split"] == "train"]
        season = tr.loc[tr["settle_date"].map(lambda d: d.month).isin(months), "score"]
        rows.append({"analysis": analysis.id, "block": block, "market": market, "node": node,
                     "n_calib": int(len(cal)), "tau_p1": _tau(cal, alpha),
                     "tau_p2_season": _tau(season, alpha), "tau_p2_all": _tau(tr["score"], alpha)})
    return rows


def cutoff_table(panel, analyses, blocks=CUTOFF_BLOCKS, seed=42, alpha=0.05):
    t = pd.DataFrame([r for a in analyses for b in blocks
                      for r in cutoffs_from_scores(score_analysis(panel, a, b, "xgb", seed), a, b, alpha)])
    t["p2_season_above_p1"] = t["tau_p2_season"] > t["tau_p1"]
    t["p2_all_above_p1"] = t["tau_p2_all"] > t["tau_p1"]
    return t
