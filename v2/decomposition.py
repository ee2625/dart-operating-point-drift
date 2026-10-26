"""Three-way operating-point-error decomposition (plan W2).

v1 could only ask ``does crossing a market add error beyond carrying a cutoff
forward in time?`` and answered *no* with three clusters. A multi-node panel lets
us split the deployment gap into three matched contrasts and ask which one, if
any, dominates:

    temporal       same market, same node, calibrate period P1 -> deploy P2
    spatial        same market, same period, node A -> node B   (within operator)
    institutional  same period, matched node_type, operator X -> operator Y

Each contrast is measured by the operating-point error ``|log10 R|`` of applying
the source cutoff to the target. The temporal contrast is the baseline (it is the
error that any redeployment incurs just from time passing); ``spatial`` and
``institutional`` are reported as **excess over the temporal baseline**, paired by
target (market, node, period). This generalizes v1's cross-minus-within excess.

Inference is clustered on ``market`` (never node: nodes within a market are
heavily cross-correlated). With ~6 markets the cluster count is small, so we use
a Rademacher sign-flip (wild cluster) test for ``H0: mean excess = 0`` and a
cluster resample for the interval, rather than trusting a normal approximation.

Input is a tidy long frame, one row per deployment, with at least:

    target_market, target_node, period, contrast, log_error

``contrast`` in {'temporal','spatial','institutional'}. Multiple source imports
for one target cell (e.g. several other nodes) are averaged to one component error
per (target_market, target_node, period), matching v1's "mean error from the other
sources". Nothing here fits a model or reads prices; it consumes the panel's
per-cell errors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CONTRASTS = ("temporal", "spatial", "institutional")
CELL_KEYS = ("target_market", "target_node", "period")
CLUSTER = "target_market"


def _component_error(cells: pd.DataFrame, contrast: str) -> pd.Series:
    """Mean ``log_error`` per target cell for one contrast (averages sources)."""
    sub = cells[cells["contrast"] == contrast]
    if sub.empty:
        return pd.Series(dtype=float)
    return sub.groupby(list(CELL_KEYS))["log_error"].mean()


def paired_excess(cells: pd.DataFrame, component: str) -> pd.DataFrame:
    """Per target cell: temporal baseline, component error, and their difference.

    Returns columns ``target_market, target_node, period, temporal, component,
    excess`` for every target cell that has both a temporal baseline and a
    ``component`` import.
    """
    if component not in ("spatial", "institutional"):
        raise ValueError(f"component must be spatial or institutional, got {component!r}")
    temporal = _component_error(cells, "temporal").rename("temporal")
    comp = _component_error(cells, component).rename("component")
    out = pd.concat([temporal, comp], axis=1, join="inner").reset_index()
    out["excess"] = out["component"] - out["temporal"]
    return out


def _cluster_means(frame: pd.DataFrame, value: str, cluster: str = CLUSTER) -> np.ndarray:
    """One value per cluster (market), equally weighting clusters downstream."""
    return frame.groupby(cluster)[value].mean().to_numpy(float)


def wild_sign_flip_p(cluster_vals: np.ndarray, reps: int = 10000, seed: int = 20260725) -> float:
    """Two-sided Rademacher sign-flip p-value for H0: mean of cluster means = 0.

    The analysis unit is the cluster (market) mean; under the null its sign is
    exchangeable, so flipping cluster signs generates the reference distribution.
    Appropriate when the number of clusters is small. Returns the finite-sample
    p-value ``(#|M*|>=|M| + 1)/(reps + 1)``.
    """
    cluster_vals = np.asarray(cluster_vals, float)
    g = len(cluster_vals)
    if g == 0:
        return float("nan")
    observed = abs(cluster_vals.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(reps, g))
    stars = np.abs((signs * cluster_vals).mean(axis=1))
    return float((np.sum(stars >= observed) + 1) / (reps + 1))


def cluster_bootstrap_ci(cluster_vals: np.ndarray, reps: int = 10000,
                         seed: int = 20260725, alpha: float = 0.05) -> tuple[float, float, float]:
    """Point estimate and percentile CI of the mean, resampling clusters w/ repl."""
    cluster_vals = np.asarray(cluster_vals, float)
    g = len(cluster_vals)
    point = float(cluster_vals.mean()) if g else float("nan")
    if g < 2:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = cluster_vals[rng.integers(0, g, size=(reps, g))].mean(axis=1)
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def decompose(cells: pd.DataFrame, reps: int = 10000, seed: int = 20260725) -> pd.DataFrame:
    """Full three-way summary: baseline temporal error and the excess of the
    spatial and institutional contrasts over it, with market-clustered inference.

    Returns one row per contrast with the mean error, and for spatial and
    institutional the mean paired excess over temporal, its cluster-bootstrap CI,
    the wild sign-flip p-value, and the number of clusters and cells.
    """
    _validate(cells)
    rows = []

    temp_cell = _component_error(cells, "temporal").reset_index(name="log_error")
    temp_cm = _cluster_means(temp_cell, "log_error")
    rows.append({
        "contrast": "temporal", "role": "baseline",
        "mean_error": float(temp_cm.mean()) if len(temp_cm) else np.nan,
        "mean_excess": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
        "sign_flip_p": np.nan, "n_clusters": int(len(temp_cm)),
        "n_cells": int(len(temp_cell)),
    })

    for component in ("spatial", "institutional"):
        ex = paired_excess(cells, component)
        if ex.empty:
            rows.append({"contrast": component, "role": "excess over temporal",
                         "mean_error": np.nan, "mean_excess": np.nan,
                         "ci_lo": np.nan, "ci_hi": np.nan, "sign_flip_p": np.nan,
                         "n_clusters": 0, "n_cells": 0})
            continue
        comp_cm = _cluster_means(ex, "component")
        excess_cm = _cluster_means(ex, "excess")
        point, lo, hi = cluster_bootstrap_ci(excess_cm, reps, seed)
        rows.append({
            "contrast": component, "role": "excess over temporal",
            "mean_error": float(comp_cm.mean()),
            "mean_excess": point, "ci_lo": lo, "ci_hi": hi,
            "sign_flip_p": wild_sign_flip_p(excess_cm, reps, seed),
            "n_clusters": int(len(excess_cm)), "n_cells": int(len(ex)),
        })

    return pd.DataFrame(rows)


def _validate(cells: pd.DataFrame) -> None:
    missing = {*CELL_KEYS, "contrast", "log_error"} - set(cells.columns)
    if missing:
        raise ValueError(f"cells missing required columns: {sorted(missing)}")
    bad = set(cells["contrast"].unique()) - set(CONTRASTS)
    if bad:
        raise ValueError(f"unexpected contrast labels: {sorted(bad)}")
    if "temporal" not in set(cells["contrast"].unique()):
        raise ValueError("no temporal-baseline rows; excess is undefined")


if __name__ == "__main__":  # tiny smoke demo on synthetic cells
    from v2.synthetic import make_cells
    print(decompose(make_cells(seed=0)).round(3).to_string(index=False))
