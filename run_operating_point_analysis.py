"""Corrected operating-point analysis: symmetric error, primary headline, gradient.

Replaces the pooled-over-specifications headline, which was dominated by
secondary specs that go inert. Three corrections, all requested in review:

1. **Symmetric error.** Under- and over-activation are both failures, so the
   primary metric is |log10 R|, which scores R=0.5 and R=2 equally. R=0 (zero
   activity) is floored at a documented value and the zero-activity rate is
   reported separately.
2. **Primary headline = G6, raw, 5% budget, single-source.** The pooled 192-cell
   number is reported only as a clearly labelled specification average.
3. **Complexity gradient tested formally**, by a paired bootstrap of
   |log R_richer| - |log R_G6| on matched cells, rather than comparing four
   separate failure rates.

Reads the sealed ``four_arm_transfer.csv``. Holdout stays sealed.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

from src.paths import DEV, assert_writable

warnings.filterwarnings("ignore")
pd.set_option("display.width", 240)

#: R below this is treated as this value for the log metric. The finest
#: resolvable nonzero rate on ~13k target hours at a 5% budget is ~0.0015, so a
#: floor of 0.01 is conservatively above the resolution limit while keeping
#: |log10 R| bounded at 2. Zero-activity cells are also reported separately.
R_FLOOR = 0.01
BUDGET_PRIMARY = 0.05
SEED = 20260720
REPLICATES = 2000
BLOCK_PAIRS = True   # resample (pair) clusters, the coarsest dependence unit


def log_error(ratio: np.ndarray) -> np.ndarray:
    return np.abs(np.log10(np.clip(ratio, R_FLOOR, 100.0)))


def main() -> None:
    fa = pd.read_csv(DEV.tables / "four_arm_transfer.csv")
    a = fa[fa["arm"] == "A_strict_data_free"].copy()
    a["log_error"] = log_error(a["alert_rate_ratio"].to_numpy())
    a["zero_activity"] = a["alert_rate_ratio"] == 0

    # ---- 1. primary headline: G6, raw, 5% -------------------------
    prim = a[(a["specification"] == "G6") & (a["target_definition"] == "raw")
             & (np.abs(a["alert_budget"] - BUDGET_PRIMARY) < 1e-6)]
    print("=" * 78)
    print("PRIMARY: G6, raw, 5% budget, strict data-free  (6 pairs x 2 folds)")
    print("=" * 78)
    print(f"  cells                    {len(prim)}")
    print(f"  median R                 {prim['alert_rate_ratio'].median():.3f}")
    print(f"  R range                  [{prim['alert_rate_ratio'].min():.2f}, "
          f"{prim['alert_rate_ratio'].max():.2f}]")
    print(f"  median |log10 R|         {prim['log_error'].median():.3f}  "
          f"(0 = perfect; 0.30 = half or double)")
    print(f"  frac |log10 R| > 0.30    {(prim['log_error'] > 0.30).mean():.3f}  "
          "(fires <0.5x or >2x intended)")
    print(f"  under-fire (R<1)         {(prim['alert_rate_ratio'] < 1).mean():.3f}")
    print(f"  over-fire  (R>1)         {(prim['alert_rate_ratio'] > 1).mean():.3f}")
    print(f"  zero-activity            {prim['zero_activity'].mean():.3f}")
    print(f"  median ranking retention {prim['auprc_retention'].median():.3f}")
    print(f"  median threshold pctile  {prim['threshold_target_percentile'].median():.1f}")

    piv = prim.pivot_table(index="pair", columns="fold", values="alert_rate_ratio")
    print("\n  R by pair and fold (two-sided instability, not a fixed direction):")
    print(piv.round(3).to_string().replace("\n", "\n  "))

    # ---- 2. specification average (labelled, NOT the headline) ----
    print("\n" + "=" * 78)
    print("SPECIFICATION AVERAGE (robustness only, not the headline)")
    print("=" * 78)
    spec = a.groupby("specification").agg(
        cells=("log_error", "size"),
        median_R=("alert_rate_ratio", "median"),
        median_log_error=("log_error", "median"),
        zero_activity=("zero_activity", "mean"),
        retention=("auprc_retention", "median"))
    print(spec.round(3).to_string())

    # ---- 3. complexity gradient, paired bootstrap -----------------
    print("\n" + "=" * 78)
    print("COMPLEXITY GRADIENT: paired |log R| difference vs G6, cluster bootstrap")
    print("EXPLORATORY -- the across-specification comparison was not preregistered")
    print("=" * 78)

    key = ["pair", "target_definition", "fold", "alert_budget"]
    g6 = a[a["specification"] == "G6"].set_index(key)["log_error"]
    rng = np.random.default_rng(SEED)
    rows = []
    for richer in ("G6+G3-R", "G6+G4", "all_shared"):
        rich = a[a["specification"] == richer].set_index(key)["log_error"]
        common = g6.index.intersection(rich.index)
        diff = (rich.loc[common] - g6.loc[common]).to_frame("d")
        diff["pair"] = [i[0] for i in diff.index]

        # Cluster bootstrap on ordered pair, the coarsest shared unit.
        pairs = diff["pair"].unique()
        draws = []
        for _ in range(REPLICATES):
            chosen = rng.choice(pairs, size=len(pairs), replace=True)
            draws.append(np.mean(np.concatenate(
                [diff.loc[diff["pair"] == p, "d"].to_numpy() for p in chosen])))
        draws = np.array(draws)
        lo, hi = np.quantile(draws, [0.025, 0.975])
        rows.append({"comparison": f"{richer} - G6", "cells": len(common),
                     "mean_delta_log_error": float(diff["d"].mean()),
                     "ci_lower": float(lo), "ci_upper": float(hi),
                     "excludes_zero": bool(lo > 0 or hi < 0)})
    gradient = pd.DataFrame(rows)
    print(gradient.round(4).to_string(index=False))
    print("\n  positive delta => richer spec has WORSE operating-point portability")

    out = assert_writable(DEV.tables / "operating_point_primary.csv")
    prim.to_csv(out, index=False)
    gradient.to_csv(assert_writable(DEV.tables / "complexity_gradient.csv"), index=False)
    print(f"\nwrote {out}")
    print(f"wrote {DEV.tables / 'complexity_gradient.csv'}")


if __name__ == "__main__":
    main()
