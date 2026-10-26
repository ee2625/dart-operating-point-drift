"""Secondary (Benjamini-Hochberg) families of spec v3 §5.2 and §6.3, fixed by controller Ruling 37.

- Audit robustness. The H-B1 (LEV6 vs DART3) and H-B2 (ROLL_DISP9 vs DART3) contrasts, re-run under:
  - the logistic estimator;
  - the frozen-origin arm;
  - cutoff alpha 1%, 2.5% and 10% (labels stay at 5%);
  - seeds 0, 1, 7 and 13.
  This gives 18 one-sided calibration-and-deployment bootstrap p-values.
- Economic robustness. The P1 rotation-test p-value of SELC for every position x fee setting, plus each
  spike-exposure split of the premium position at calibrated fees (es.run_economics rows, key "rotation_p").
- Drift monitors. The one-sided Spearman p-value of every (block, monitor) pair against |log10 R| across nodes
  (es.run_monitors.evaluate_monitors).

Each family is adjusted on its own, at q = 0.05."""
from __future__ import annotations

import pandas as pd

from .bootstrap import block_contrast_bootstrap
from .run_inference import bh

TABLE_COLUMNS = ["family", "test", "estimate", "p", "bh_reject", "included"]      # outputs/es/inference/bh.csv
HB_PAIRS = (("H-B1", "LEV6"), ("H-B2", "ROLL_DISP9"))
AUDIT_VARIANTS = ((("logit", "roll", "logit", 42, 0.05), ("frozen", "frozen", "xgb", 42, 0.05))
                  + tuple((f"alpha={a:g}", "roll", "xgb", 42, a) for a in (0.01, 0.025, 0.10))
                  + tuple((f"seed={s}", "roll", "xgb", s, 0.05) for s in (0, 1, 7, 13)))


def audit_family(score, n_rep=2000, contrast=None):
    """`score(kind, block, estimator, seed)` returns a score frame (kind is "roll" or "frozen").

    `contrast(alt, base, alpha=..., n_rep=...)` is the bootstrap of each of the 18 tests: block_contrast_bootstrap, looked
    up when this function is called, unless the caller passes another with the same "t_hat" and "p" in its result. The sealed
    run passes its finite-replicate version (controller Ruling 109, M1), so that one NaN replicate does not cost it all 18."""
    contrast = block_contrast_bootstrap if contrast is None else contrast
    rows = []
    for hid, alt in HB_PAIRS:
        for name, kind, est, seed, alpha in AUDIT_VARIANTS:
            r = contrast(score(kind, alt, est, seed), score(kind, "DART3", est, seed), alpha=alpha, n_rep=n_rep)
            rows.append({"family": "audit robustness", "test": f"{hid} {name}", "estimate": r["t_hat"],
                         "p": r["p"]})
    return rows


def economic_family(econ_rows):
    return [{"family": "economic robustness", "test": f"P1 {r['position']} fees={r['fees']}",
             "estimate": r["SELC"], "p": r["rotation_p"]}
            for r in econ_rows if r["policy"] == "P1" and r["comparator"] == "none"]


def monitor_family(eval_rows):
    return [{"family": "drift monitors", "test": f"{r['block']} {r['monitor']}", "estimate": r["spearman"],
             "p": r["spearman_p_one_sided"]} for r in eval_rows]


def bh_table(rows, q=0.05):
    """``rows`` (the families' dicts, or a frame of them) with two more columns, ``bh_reject`` and ``included``.

    ``bh_reject`` is BH at ``q`` within each family. ``included`` is True where the p-value is not NaN: a NaN is no test
    that ran, so it is left out of its family's adjustment and never rejected, and ``included`` summed over a family
    is the m that BH used (count with it, not with the rows). A (family, test) pair that occurs twice would be counted
    as two tests, so it raises ValueError, and so does a row without a family. No rows give an empty table with the
    columns ``TABLE_COLUMNS``."""
    df = pd.DataFrame(rows).reset_index(drop=True)      # a copy, with unique labels whatever the caller's index
    if df.empty:
        return pd.DataFrame(columns=TABLE_COLUMNS)
    no_family = df["family"].isna()
    if no_family.any():
        orphans = df.loc[no_family, "test"].tolist()[:5]
        raise ValueError(f"bh_table: {int(no_family.sum())} row(s) have no family (tests {orphans}); groupby would "
                         "skip them, so they would be neither adjusted nor counted")
    repeated = df.duplicated(["family", "test"], keep=False)
    if repeated.any():
        pairs = [tuple(p) for p in df.loc[repeated, ["family", "test"]].drop_duplicates().to_numpy().tolist()]
        raise ValueError(f"bh_table: {int(repeated.sum())} rows repeat a (family, test) pair, which BH would count as "
                         f"separate tests: {pairs[:5]}{' ...' if len(pairs) > 5 else ''}")
    df["bh_reject"] = False
    df["included"] = df["p"].notna()
    for _, g in df.groupby("family"):
        used = g[g["included"]]
        if len(used):
            df.loc[used.index, "bh_reject"] = bh(used["p"].to_numpy(), q)
    return df
