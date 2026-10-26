"""Regenerate every result of the 31 Jul paper from saved outputs (FORENSICS section 6, D1-D10).

Run: DART_PROJECT_ROOT=... /opt/anaconda3/bin/python3 -m es.run_legacy

Outputs go to ``outputs/es/legacy/`` only:

* one CSV per (analysis, block, estimator, seed, policy, alpha) with the per-node activity, AUPRC and
  prevalence (``legacy_summary`` writes it);
* ``summary.csv`` (one row per run), ``regenerated_long.csv`` (every regenerated number in the long
  form the comparison joins on) and ``comparison.csv`` (``es/published_v3.csv`` against it);
* ``table1_panel.csv``, ``fig1_volatility.csv``, ``drift_2026.csv``, ``d10_boundaries.csv`` and
  ``d5_d9.json``.

A published number is tagged with the analysis that actually produced it, so the comparison shows
which numbers each variant reproduces (D1: ``24H2-25H2`` trains on 2023-01..2024-06, and the
``24H2-25H2-train2023`` variant trains on 2023 only). A number reproduces when the regenerated value
rounds to the printed one, i.e. it lies within half a unit of the last printed digit (``decimals``).

When the producing design is not the one the paper states or the revision adopts, the row also names
that design in ``stated_analysis`` (D1: the median abs log-ratio of 24H2->25H2 comes from a different fit
than the rest of its row; D6: the Table 1 prevalence uses q from 2023-H1, the caption says 2023). If the
producing analysis reproduces the printed number but the stated design does not, the status is
``match_other_design``, and ``comparison.csv`` keeps the stated design's value in ``as_stated_value``.
The statuses are ``match``, ``match_other_design``, ``mismatch`` and ``no_counterpart``; ``match`` is True
for the first only, so every row with ``match == False`` needs the author's attention.

Vocabulary of ``es/published_v3.csv`` (one row per number printed in the paper):

* ``analysis``: an id of ``es.analyses.LEGACY``, or a pseudo-analysis: ``panel`` (Table 1 counts),
  ``panel-q2023H1`` and ``panel-q2023`` (the 2024 event rate under the 2023-H1 label rule that produced
  the printed prevalence, and under the 2023 rule the caption states), ``fig1``, ``drift2026``, ``all`` (a
  property of a block), ``agg:2024-25``, ``agg:4pairs`` and ``agg:2pairs`` (a range or a count over
  several analyses), ``agg:paper`` (the runs behind a published number), ``agg:legacy`` (every default
  legacy run) and ``heldout-2024`` (the July design behind the calendar AUPRC of 0.075, which has no
  counterpart in the matched designs).
* ``block``: a key of ``es.blocks.BLOCKS``, an operator (Table 1), a period (Figure 1) or the label
  of a comparison (``DART18-LEV6``, ``5blocks``).
* ``metric``: a column of ``summary.csv`` (failures, median_R, ...), with ``@P2_all_train``, ``@a<alpha>``
  and ``@band1.5`` for another policy, cutoff alpha or failure band, ``failures_<operator>``, and the
  derived numbers of ``derived_metrics``.
* ``decimals``: digits printed after the decimal point (negative: rounded to thousands).
* ``source_line``: ``<file>:<line>`` or ``<file>:<first>-<last>``, with MR = ``paper_v3/v3_method_results.tex``,
  MAIN = ``paper_v3/v3_main.tex`` (the abstract), IR = ``paper_v3/v3_intro_related.tex``,
  DS = ``paper_v3/v3_discussion.tex`` and PM = ``v2/scripts/make_pipeline_map.py`` (the Figure 3 values).
* ``stated_analysis``: the design the paper states or the revision adopts, when it differs from ``analysis``.
"""
from __future__ import annotations

import json
import re
import time
import warnings
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr, wasserstein_distance
from sklearn.metrics import average_precision_score

from .analyses import LABEL_CONTROL, LEGACY
from .blocks import BLOCKS
from .config import DATA_ROOT, MARKETS, OUT_DIR, ensure_dirs
from .data import build_panel, load_dev_raw
from .labels import label_training_q
from .policies import activity, p1_fixed, p2_insample
from .scores import cached_scores
from .windows import UtcWindow

LEGACY_DIR = OUT_DIR / "legacy"
PUBLISHED_CSV = Path(__file__).with_name("published_v3.csv")
SEEDS = (0, 1, 7, 13, 42)
ALPHAS = (0.01, 0.025, 0.05, 0.10)
BANDS = {"0.5_2": (0.5, 2.0), "0.67_1.5": (2 / 3, 1.5), "0.8_1.25": (0.8, 1.25)}
#: Blocks of the alpha / band sweep (D8) and of the seed runs (D8).
SWEEP_BLOCKS = ("LEV6", "DART18", "G1_24", "DART3")
SEED_BLOCKS = ("G6", "LEV6", "DART3", "ROLL_DISP9", "DART18")
#: The five blocks of the matched block audit behind "49 of 53" (v3_ranking_cutoff_nodes.csv).
RANKING_BLOCKS = ("G6", "LEV6", "DART3", "ROLL_DISP9", "DART18")
#: Label-control analyses that repeat a legacy design under another id (same windows and label rule).
LC_TWINS = {"LC-full-q2324": "sealed26H1", "LC-h1-q2023": "25H1-26H1"}
PAIRS_2024_25 = ("24H1-25H1", "24H2-25H2", "2024-2025")
PAIRS_ALL = PAIRS_2024_25 + ("25H1-26H1",)
PAIRS_2 = ("2024-2025", "25H1-26H1")
#: Pseudo-analyses that aggregate a published range or count over several analyses.
AGGREGATES = {"agg:2024-25": PAIRS_2024_25, "agg:4pairs": PAIRS_ALL, "agg:2pairs": PAIRS_2}
#: Published metric names for the non-default band, as used in published_v3.csv.
BAND_METRIC = {"0.67_1.5": "failures@band1.5", "0.8_1.25": "failures@band1.25"}
NOT_APPLICABLE = "-"

KEY = ["analysis", "block", "estimator", "metric"]


# --------------------------------------------------------------------------------------------
# One run: scores -> cutoff policy -> per-node activity, AUPRC, prevalence
# --------------------------------------------------------------------------------------------

def _auprc(dep: pd.DataFrame) -> float:
    d = dep.dropna(subset=["label"])
    y, s = d["label"].to_numpy(int), d["score"].to_numpy(float)
    return float(average_precision_score(y, s)) if len(y) and y.min() != y.max() else float("nan")


def _node_extras(alerts: pd.DataFrame) -> pd.DataFrame:
    """Per node: deployment AUPRC and prevalence, and the node's cutoff (one tau per node under P1 and P2)."""
    rows = []
    for (market, node), d in alerts.groupby(["market", "node"], sort=True):
        label = d["label"].dropna()
        rows.append({"market": market, "node": node, "auprc": _auprc(d),
                     "prevalence": float(label.mean()) if len(label) else float("nan"),
                     "tau": float(d["tau"].iloc[0])})
    return pd.DataFrame(rows, columns=["market", "node", "auprc", "prevalence", "tau"])


def _alerts(scores: pd.DataFrame, policy: str, alpha: float) -> pd.DataFrame:
    if policy == "P1":
        return p1_fixed(scores, alpha)
    if policy == "P2_all_train":            # Fig. 2: the cutoff on all training scores
        return p2_insample(scores, alpha, months=set(range(1, 13)))
    raise ValueError(policy)


def _per_node(scores, analysis, block, estimator, seed, policy, alpha=0.05) -> pd.DataFrame:
    """Per-node activity of ``policy`` on ``scores`` and the CSV of it (one file per run)."""
    alerts = _alerts(scores, policy, alpha)
    act = activity(alerts, alpha)
    out = act.merge(_node_extras(alerts), on=["market", "node"], validate="one_to_one")
    for name, (lo, hi) in BANDS.items():
        out[f"fail_{name}"] = (out["R"] < lo) | (out["R"] > hi)
    out = out.assign(analysis=analysis.id, block=block, estimator=estimator, seed=seed,
                     policy=policy, alpha=alpha)
    LEGACY_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(LEGACY_DIR / f"{analysis.id}__{block}__{estimator}__s{seed}__{policy}__a{alpha}.csv",
               index=False)
    return out


def legacy_summary(panel, analysis, block, estimator, seed, policy, alpha=0.05):
    """One row per node: activity (R, log error, failure flags), AUPRC and prevalence.

    The scores come from ``cached_scores`` with the default 5% label, so an alpha sweep varies only the
    cutoff alpha, as the July code did (``v2/scripts/regime_extra.py:17,48``)."""
    scores = cached_scores(panel, analysis, block, estimator, seed)
    return _per_node(scores, analysis, block, estimator, seed, policy, alpha)


def _summ(df: pd.DataFrame) -> pd.Series:
    out = {"failures": int(df["fail_0.5_2"].sum()), "n_nodes": len(df),
           "median_R": float(df["R"].median()),
           "median_log_error": float(df["log_error"].median()),
           "median_auprc": float(df["auprc"].median()),
           "auprc_lift": float((df["auprc"] / df["prevalence"]).median()),
           "prevalence": float(df["prevalence"].median()),
           "n_zero_activity": int(df["zero_activity"].sum())}
    for name in BANDS:
        if name != "0.5_2":
            out[f"failures_band_{name}"] = int(df[f"fail_{name}"].sum())
    for market in MARKETS:
        out[f"failures_{market}"] = int(df.loc[df["market"] == market, "fail_0.5_2"].sum())
    return pd.Series(out)


# --------------------------------------------------------------------------------------------
# Derived numbers (the published ones that are not columns of the per-run summary)
# --------------------------------------------------------------------------------------------

def sign_test(level: pd.DataFrame, dart: pd.DataFrame) -> tuple[int, int, float]:
    """Nodes whose level error exceeds their DART-relative error, of n, and the two-sided sign test p
    (regime_rigor_fast.py: ``k = (eL > eD).sum()``, ``binomtest(k, n, 0.5)``)."""
    m = level.merge(dart, on=["market", "node"], suffixes=("_l", "_d"))
    k, n = int((m["log_error_l"] > m["log_error_d"]).sum()), len(m)
    return k, n, float(binomtest(k, n, 0.5).pvalue)


def operator_contrasts(level: pd.DataFrame, dart: pd.DataFrame) -> pd.Series:
    """Per operator, the median over its nodes of ``log_error`` (DART-relative) minus ``log_error``
    (level): positive when the DART-relative cutoff is worse (regime_ablation.py)."""
    m = level.merge(dart, on=["market", "node"], suffixes=("_l", "_d"))
    m["d"] = m["log_error_d"] - m["log_error_l"]
    return m.groupby("market")["d"].median()


def delta_auprc(block: pd.DataFrame, base: pd.DataFrame) -> tuple[float, int, int]:
    """Median per-node change in AUPRC over the calendar baseline, nodes with a positive change, and
    nodes compared (incremental_auprc.py: nodes where both AUPRCs exist)."""
    m = block.merge(base, on=["market", "node"], suffixes=("", "_base"))
    d = (m["auprc"] - m["auprc_base"]).dropna()
    return float(d.median()), int((d > 0).sum()), len(d)


def score_dispersion(scores: pd.DataFrame) -> tuple[float, float]:
    """Deployment-to-calibration ratio of the score SD and of the score IQR, median over nodes
    (v2/scripts/score_dispersion.py)."""
    def iqr(x):
        return np.subtract(*np.percentile(x, [75, 25]))
    sd, iq = [], []
    for _, g in scores.groupby(["market", "node"], sort=True):
        c = g.loc[g["split"] == "calib", "score"].to_numpy(float)
        d = g.loc[g["split"] == "deploy", "score"].to_numpy(float)
        if len(c) and len(d):
            sd.append(d.std() / (c.std() or 1e-9))
            iq.append(iqr(d) / (iqr(c) or 1e-9))
    return float(np.nanmedian(sd)), float(np.nanmedian(iq))


def source_activity(scores, alpha):
    """D9: realized alert rate of P1 on the calibration rows themselves (min, max over nodes)."""
    rates = []
    for _, g in scores.groupby(["market", "node"]):
        cal = g.loc[g["split"] == "calib", "score"].to_numpy(float)
        rates.append(float((cal >= np.quantile(cal, 1 - alpha)).mean()))
    return min(rates), max(rates)


def _row(analysis, block, estimator, metric, value):
    return {"analysis": analysis, "block": block, "estimator": estimator, "metric": metric,
            "regenerated": float(value)}


def base_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    """The per-run summary in the long form the comparison joins on.

    Default runs (seed 42, P1, alpha 5%) keep the metric name. Other policies and alphas add
    ``@<policy>`` and ``@a<alpha>``, and the 1.5x band is ``failures@band1.5``."""
    rows = []
    s = summary.reset_index()
    s42 = s[s["seed"] == 42]
    for _, r in s42.iterrows():
        a, b, e = r["analysis"], r["block"], r["estimator"]
        if r["policy"] == "P1" and r["alpha"] == 0.05:
            for m in ("failures", "n_nodes", "median_R", "median_log_error", "median_auprc",
                      "auprc_lift", "prevalence", "n_zero_activity", *[f"failures_{k}" for k in MARKETS]):
                rows.append(_row(a, b, e, m, r[m]))
            for band, name in BAND_METRIC.items():
                rows.append(_row(a, b, e, name, r[f"failures_band_{band}"]))
        elif r["policy"] == "P1":
            rows.append(_row(a, b, e, f"failures@a{r['alpha']:g}", r["failures"]))
        elif r["policy"] == "P2_all_train" and r["alpha"] == 0.05:
            for m in ("failures", "median_R", "median_log_error", "n_nodes"):
                rows.append(_row(a, b, e, f"{m}@{r['policy']}", r[m]))
    # seed ranges (D8): the 5-seed min and max of the failure count
    seeded = s[(s["policy"] == "P1") & (s["alpha"] == 0.05) & (s["estimator"] == "xgb")]
    for (a, b), g in seeded.groupby(["analysis", "block"]):
        if g["seed"].nunique() == len(SEEDS):
            rows.append(_row(a, b, "xgb", "failures_seed_min", g["failures"].min()))
            rows.append(_row(a, b, "xgb", "failures_seed_max", g["failures"].max()))
    return pd.DataFrame(rows)


def paper_runs(published: pd.DataFrame) -> set:
    """Keys (analysis, block, estimator, seed, policy, alpha) of the legacy runs behind a published number.

    The metric says which run: ``@a<alpha>`` is a cutoff alpha, ``@P2_all_train`` the in-sample cutoff,
    ``failures_seed_*`` the five seeds; any other metric is the default run (seed 42, P1, alpha 5%)."""
    runs = set()
    for r in published.itertuples(index=False):
        if r.estimator not in ("xgb", "logit"):
            continue
        policy, alpha = "P1", 0.05
        if "@a" in r.metric:
            alpha = float(r.metric.split("@a")[1])
        elif r.metric.endswith("@P2_all_train"):
            policy = "P2_all_train"
        seeds = SEEDS if r.metric.startswith("failures_seed_") else (42,)
        blocks = RANKING_BLOCKS if r.block == "5blocks" else tuple(re.split(r"[-+]", r.block))
        for name in AGGREGATES.get(r.analysis, (r.analysis,)):
            if name in LEGACY:
                runs.update((name, b, r.estimator, seed, policy, alpha) for b in blocks if b in BLOCKS for seed in seeds)
    return runs


def derived_metrics(frames: dict, runs: set | None = None) -> pd.DataFrame:
    """Published numbers computed from the per-node frames of several runs.

    ``frames`` maps (analysis id, block, estimator, seed, policy, alpha) to a per-node frame. Needs the
    default runs (seed 42, P1, alpha 5%) of the legacy analyses and the Fig. 2 in-sample run. ``runs``
    (see ``paper_runs``) are the runs behind a published number; "no cell has r = 0" is counted over them
    (``agg:paper``), and ``None`` counts the default runs instead. Every run in ``runs`` must have been made."""
    rows = []

    def base(a, b, e="xgb"):
        return frames[(a, b, e, 42, "P1", 0.05)]

    def has(a, b, e):
        return (a, b, e, 42, "P1", 0.05) in frames

    signs = {}
    for e in ("xgb", "logit"):
        for a in sorted({k[0] for k in frames}):
            if has(a, "LEV6", e) and has(a, "DART18", e):
                lev, dart = base(a, "LEV6", e), base(a, "DART18", e)
                k, _, p = sign_test(lev, dart)
                signs[(a, e)] = (k, p)
                rows += [_row(a, "LEV6", e, "signtest_k", k), _row(a, "LEV6", e, "signtest_p", p)]
                rows += [_row(a, "DART18-LEV6", e, f"opcontrast_{op}", v)
                         for op, v in operator_contrasts(lev, dart).items()]
            if has(a, "G6", e):
                for b in BLOCKS:
                    if b != "G6" and has(a, b, e):
                        med, npos, n = delta_auprc(base(a, b, e), base(a, "G6", e))
                        rows += [_row(a, b, e, "delta_auprc_median", med), _row(a, b, e, "delta_auprc_npos", npos),
                                 _row(a, b, e, "delta_auprc_n", n)]
    # ranges over several analyses
    for agg, names in AGGREGATES.items():
        for blk in ("LEV6", "DART18", "G6"):
            f = [int(base(a, blk)["fail_0.5_2"].sum()) for a in names]
            rows += [_row(agg, blk, "xgb", "failures_min", min(f)), _row(agg, blk, "xgb", "failures_max", max(f))]
        prev = [float(base(a, "LEV6")["prevalence"].median()) for a in names]
        rows += [_row(agg, "LEV6", "xgb", "prevalence_min", min(prev)),
                 _row(agg, "LEV6", "xgb", "prevalence_max", max(prev))]
    ks = [signs[(a, "xgb")][0] for a in PAIRS_2024_25]
    ps = [signs[(a, "xgb")][1] for a in PAIRS_2024_25]
    rows += [_row("agg:2024-25", "LEV6", "xgb", "signtest_k_min", min(ks)),
             _row("agg:2024-25", "LEV6", "xgb", "signtest_k_max", max(ks)),
             _row("agg:2024-25", "LEV6", "xgb", "signtest_p_min", min(ps))]
    # "49 of 53": failing node-block results that still rank above the prevalence baseline
    failing = above = 0
    for a in PAIRS_2:
        for b in RANKING_BLOCKS:
            d = base(a, b)
            f = d[d["fail_0.5_2"]]
            failing += len(f)
            above += int((f["auprc"] / f["prevalence"] > 1.0).sum())
    rows += [_row("agg:2pairs", "5blocks", "xgb", "n_failing_nodeblocks", failing),
             _row("agg:2pairs", "5blocks", "xgb", "n_failing_above_prev", above)]
    # "no cell has r = 0": over the runs behind a published number, and over every default legacy run
    default = {k: f for k, f in frames.items() if k[0] in LEGACY and k[3] == 42 and k[4] == "P1" and k[5] == 0.05}
    if runs is not None:
        missing = sorted(k for k in runs if k not in frames)
        if missing:
            raise ValueError(f"the paper reports {len(missing)} run(s) that were not made, e.g. {missing[0]}")
    behind = default if runs is None else {k: frames[k] for k in runs}
    rows.append(_row("agg:paper", "all", "all", "n_zero_activity", sum(int(f["zero_activity"].sum()) for f in behind.values())))
    rows.append(_row("agg:legacy", "all", "all", "n_zero_activity", sum(int(f["zero_activity"].sum()) for f in default.values())))
    # Fig. 2: every node moves down when the cutoff moves to the training scores
    held = base("2024-2025", "G1_24").set_index(["market", "node"])["R"]
    insample = frames[("2024-2025", "G1_24", "xgb", 42, "P2_all_train", 0.05)].set_index(["market", "node"])["R"]
    rows.append(_row("2024-2025", "G1_24", "xgb", "nodes_down", int((insample.reindex(held.index) < held).sum())))
    rows.append(_row("2024-2025", "G1_24", "xgb", "median_R_ratio", held.median() / insample.median()))
    # feature counts of Table 5 (calendar-only has no price-history features)
    rows += [_row("all", b, NOT_APPLICABLE, "dim", len(feats)) for b, feats in BLOCKS.items()]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# Comparison with the published numbers
# --------------------------------------------------------------------------------------------

def read_published(path: Path = PUBLISHED_CSV) -> pd.DataFrame:
    p = pd.read_csv(path, dtype=str, keep_default_na=False)
    p["value"] = p["value"].astype(float)
    p["decimals"] = p["decimals"].astype(int)
    if "stated_analysis" not in p.columns:
        p["stated_analysis"] = ""
    return p


def build_comparison(published: pd.DataFrame, regenerated: pd.DataFrame) -> pd.DataFrame:
    """Left-join the published numbers to the regenerated ones.

    A number reproduces when the regenerated value lies within half a unit of the last printed digit
    (``decimals``): it then rounds to the published number. ``status`` is
    * ``no_counterpart``: nothing regenerated under that key;
    * ``mismatch``: the producing analysis does not reproduce the printed number;
    * ``match_other_design``: it does, but the row names the design the paper states or the revision adopts
      (``stated_analysis``) and that design does not (``as_stated_value`` holds its value);
    * ``match``: the producing analysis reproduces it and no stated design disagrees.
    ``match`` is True for ``match`` only."""
    if regenerated.duplicated(KEY).any():
        dup = regenerated[regenerated.duplicated(KEY, keep=False)]
        raise ValueError(f"regenerated numbers are not unique on {KEY}:\n{dup.head(10)}")
    pub = published.assign(stated_analysis=(published["stated_analysis"].fillna("") if "stated_analysis" in published
                                            else ""))
    comp = pub.merge(regenerated, on=KEY, how="left")
    stated = regenerated.rename(columns={"analysis": "stated_analysis", "regenerated": "as_stated_value"})
    comp = comp.merge(stated, on=["stated_analysis", "block", "estimator", "metric"], how="left")
    comp.insert(comp.columns.get_loc("regenerated") + 1, "as_stated_value", comp.pop("as_stated_value"))
    flagged = comp["stated_analysis"] != ""
    unresolved = comp[flagged & comp["as_stated_value"].isna()]
    if len(unresolved):
        raise ValueError("stated design with nothing regenerated under it: "
                         + "; ".join(f"{r.stated_analysis}/{r.block}/{r.estimator}/{r.metric}"
                                     for r in unresolved.head(5).itertuples()))
    comp["abs_diff"] = (comp["regenerated"] - comp["value"]).abs()
    tol = 0.5 * 10.0 ** (-comp["decimals"].astype(float)) + 1e-9
    reproduces = comp["abs_diff"] <= tol
    stated_reproduces = (comp["as_stated_value"] - comp["value"]).abs() <= tol
    comp["status"] = np.select([comp["regenerated"].isna(), ~reproduces, flagged & ~stated_reproduces],
                               ["no_counterpart", "mismatch", "match_other_design"], default="match")
    comp["match"] = comp["status"] == "match"
    return comp


# --------------------------------------------------------------------------------------------
# Table 1, Figure 1 and the 2026 drift (D5-D7)
# --------------------------------------------------------------------------------------------

def table1_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """D6: node and row counts per operator, and the 2024 event rate under two label rules.

    ``prevalence_24_q2023`` is what the caption says (q on the 2023 training year). The published values
    are reproduced by ``prevalence_24_q2023H1`` (q on 2023-H1, the deepening.py design)."""
    in24 = UtcWindow("2024-01-01", "2025-01-01").mask(panel).to_numpy()
    out = panel.groupby("market").agg(nodes=("node", "nunique"), hourly_rows=("node", "size"))
    out["hubs"] = panel[panel["node_type"] == "trading_hub"].groupby("market")["node"].nunique()
    out["zones"] = panel[panel["node_type"] == "load_zone"].groupby("market")["node"].nunique()
    out[["hubs", "zones"]] = out[["hubs", "zones"]].fillna(0).astype(int)
    for name, window in (("q2023", UtcWindow("2023-01-01", "2024-01-01")),
                         ("q2023H1", UtcWindow("2023-01-01", "2023-07-01"))):
        label, _ = label_training_q(panel, window.mask(panel), 0.05)
        out[f"prevalence_24_{name}"] = label[in24].groupby(panel.loc[in24, "market"]).mean()
    return out


def fig1_volatility(panel):
    """D7: what the Figure 1 caption says. Panel median over nodes of the quarterly std of the
    DA price and of DART, plus quarterly p95 of |DART| (node median and pooled).

    ``da_std_pooled`` and ``dart_std_pooled`` are what the July script computed (a pooled std over all
    node-hours). The ``kind == "period"`` rows give the same columns for whole periods on the legacy UTC
    half-open windows (``period_local``: inclusive settlement dates): the caption's "p95 |DART| 30 -> 62"
    is the pooled p95 of 2024 against 2026-H1 (``dart_p95_pooled``), not a panel median."""
    date = pd.to_datetime(panel["settle_date"])
    work = panel.assign(_abs=panel["dart_spread"].abs())

    def table(groups: pd.Series, kind: str) -> pd.DataFrame:
        w = work.assign(_g=groups)
        w = w[w["_g"].notna()]
        per_node = w.groupby(["_g", "market", "node"]).agg(
            da_std=("da_lmp", "std"), dart_std=("dart_spread", "std"),
            dart_p95=("_abs", lambda s: s.quantile(0.95)))
        out = per_node.groupby("_g").median()
        pooled = w.groupby("_g").agg(da_std_pooled=("da_lmp", "std"), dart_std_pooled=("dart_spread", "std"),
                                     dart_p95_pooled=("_abs", lambda s: s.quantile(0.95)))
        out = out.join(pooled)
        out.index.name = "period"
        out.insert(0, "kind", kind)
        return out

    parts = [table(pd.Series(pd.PeriodIndex(date, freq="Q").astype(str), index=panel.index), "quarter")]
    for name, (lo, hi) in {"2023": ("2023-01-01", "2024-01-01"), "2024": ("2024-01-01", "2025-01-01"),
                           "2025": ("2025-01-01", "2026-01-01"), "2026H1": ("2026-01-01", "2026-07-01"),
                           "2023-24": ("2023-01-01", "2025-01-01")}.items():
        utc = UtcWindow(lo, hi).mask(panel)
        local = (date >= pd.Timestamp(lo)) & (date < pd.Timestamp(hi))
        for kind, sel in (("period", utc), ("period_local", local)):
            parts.append(table(pd.Series(np.where(sel, name, None), index=panel.index), kind))
    return pd.concat(parts)


def window_boundaries(panel: pd.DataFrame) -> pd.DataFrame:
    """D10: what the half-open UTC windows of the legacy analyses do to local operating days.

    For every distinct window of ``LEGACY`` and every operator: the rows inside the UTC window (``rows_utc``), the
    rows of the same window read as inclusive local settlement dates (``rows_local``), the rows inside the UTC
    window that fall on another local date (``extra_other_dates``), and the rows on the window's local dates that
    the UTC window misses (``missed_local_dates``). ``used_by`` lists ``role:analysis`` for each use."""
    dates = pd.to_datetime(panel["settle_date"])
    used: dict = {}
    for analysis in LEGACY.values():
        for role in ("label_train", "train", "calib", "deploy"):
            window = getattr(analysis, role)
            used.setdefault((window.start, window.end), []).append(f"{role}:{analysis.id}")
    rows = []
    for (start, end), roles in sorted(used.items()):
        utc = UtcWindow(start, end).mask(panel).to_numpy()
        local = ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy()
        for market in MARKETS:
            here = (panel["market"] == market).to_numpy()
            rows.append({"window_start": start, "window_end": end, "market": market,
                         "rows_utc": int((utc & here).sum()), "rows_local": int((local & here).sum()),
                         "extra_other_dates": int((utc & ~local & here).sum()),
                         "missed_local_dates": int((~utc & local & here).sum()), "used_by": " ".join(roles)})
    return pd.DataFrame(rows)


def _nwd(a, d):
    """``nwd`` of v2/scripts/open_holdout.py, verbatim: W1 normalised by the IQR of the calibration values."""
    a, d = a[np.isfinite(a)], d[np.isfinite(d)]
    if len(a) < 50 or len(d) < 50:
        return np.nan
    iqr = np.subtract(*np.percentile(a, [75, 25])) or 1e-9
    return wasserstein_distance(a, d) / iqr


def drift_2026(panel) -> float:
    """D5: 2025 -> 2026-H1 feature drift against the as-opened |log10 R| of the sealed design.

    Logic of ``v2/scripts/open_holdout.py:36-58`` on the saved ``holdout_*.csv`` cells: per node and family
    the median over the family's features of the IQR-normalised 1-Wasserstein distance (calibration 2025,
    deployment 2026-H1), against the node's mean temporal ``log_error``. The returned Spearman is pooled
    over the level and DART-relative families, as printed there; ``drift_2026.csv`` also lists the
    combined family (G1)."""
    cal, dep = UtcWindow("2025-01-01", "2026-01-01"), UtcWindow("2026-01-01", "2026-07-01")
    families = (("level", "LEV6"), ("DARTrel", "DART18"), ("G1", "G1_24"))
    errors = {fam: pd.read_csv(DATA_ROOT / "holdout_2026H1" / f"holdout_{fam}.csv").query("contrast == 'temporal'")
              .groupby(["target_market", "target_node"])["log_error"].mean() for fam, _ in families}
    rows = []
    for (m, n), g in panel.groupby(["market", "node"]):
        c, d = g[cal.mask(g).to_numpy()], g[dep.mask(g).to_numpy()]
        for fam, block in families:
            w1 = float(np.nanmedian([_nwd(c[f].to_numpy(float), d[f].to_numpy(float)) for f in BLOCKS[block]]))
            rows.append({"family": fam, "market": m, "node": n, "W1_norm": w1, "log_error": float(errors[fam][(m, n)])})
    df = pd.DataFrame(rows)
    LEGACY_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(LEGACY_DIR / "drift_2026.csv", index=False)
    pooled = df[df["family"].isin(["level", "DARTrel"])]
    return float(spearmanr(pooled["W1_norm"], pooled["log_error"], nan_policy="omit").correlation)


# --------------------------------------------------------------------------------------------
# The full run
# --------------------------------------------------------------------------------------------

def run_plan():
    """Every scored run of ``main``: (analysis, block, estimator, seed, [(policy, alpha), ...]).

    Each run uses one scores frame, so its policy variants share a single read of the score cache."""
    for a in list(LEGACY.values()) + list(LABEL_CONTROL.values()):
        for block in BLOCKS:
            for est in ("xgb", "logit"):
                variants = [("P1", 0.05)]
                if est == "xgb":
                    if block in SWEEP_BLOCKS:                      # D8: alpha / band sweep
                        variants += [("P1", alpha) for alpha in ALPHAS if alpha != 0.05]
                    if block == "G1_24":                           # Fig. 2 provenance: in-sample placement
                        variants.append(("P2_all_train", 0.05))
                yield a, block, est, 42, variants
    for a in (LEGACY["2024-2025"], LEGACY["25H1-26H1"]):           # D8: seeds
        for block in SEED_BLOCKS:
            for seed in SEEDS:
                if seed != 42:
                    yield a, block, "xgb", seed, [("P1", 0.05)]


def _extras_for(scores, analysis, block, estimator, seed) -> list[dict]:
    """The published numbers that need the scores themselves: dispersion ratios and source activity."""
    rows = []
    if estimator == "xgb" and seed == 42:
        if analysis.id in PAIRS_2 and block in ("LEV6", "DART18"):
            sd, iq = score_dispersion(scores)                      # D8: printed only
            rows += [_row(analysis.id, block, estimator, "sd_ratio", sd),
                     _row(analysis.id, block, estimator, "iqr_ratio", iq)]
    if seed == 42 and analysis.id in LEGACY:
        lo, hi = source_activity(scores, 0.05)                     # D9
        rows += [_row(analysis.id, block, estimator, "source_activity_min", lo),
                 _row(analysis.id, block, estimator, "source_activity_max", hi)]
    return rows


def twin_check(frames: dict) -> dict:
    """Largest absolute difference in R between each label-control twin and the legacy run it repeats."""
    worst = {twin: 0.0 for twin in LC_TWINS}
    for twin, original in LC_TWINS.items():
        for key, f in frames.items():
            if key[0] == twin:
                other = frames[(original, *key[1:])]
                m = f.merge(other, on=["market", "node"], suffixes=("", "_o"), validate="one_to_one")
                worst[twin] = max(worst[twin], float((m["R"] - m["R_o"]).abs().max()))
    return worst


def summary_frame(summ: dict) -> pd.DataFrame:
    """The per-run summaries (key -> Series) as one frame indexed by (analysis, block, estimator, seed, policy, alpha)."""
    summary = pd.DataFrame(list(summ.values()), index=pd.MultiIndex.from_tuples(
        list(summ), names=["analysis", "block", "estimator", "seed", "policy", "alpha"]))
    return summary.astype({c: int for c in summary.columns
                           if c.startswith("failures") or c in ("n_nodes", "n_zero_activity")})


def regenerated_numbers(summary, frames, extras, table1, fig1, rho, n_nodes, published=None) -> pd.DataFrame:
    """Every regenerated number in the long form the comparison joins on (analysis, block, estimator, metric).

    ``published`` (the transcribed paper numbers) defines the runs that "no cell has r = 0" refers to."""
    rows = [_row("panel", "all", NOT_APPLICABLE, "n_nodes", n_nodes)]
    for market, r in table1.iterrows():
        rows += [_row("panel", market, NOT_APPLICABLE, "n_nodes", r["nodes"]),
                 _row("panel", market, NOT_APPLICABLE, "n_hubs", r["hubs"]),
                 _row("panel", market, NOT_APPLICABLE, "n_zones", r["zones"]),
                 _row("panel", market, NOT_APPLICABLE, "rows", r["hourly_rows"]),
                 _row("panel-q2023", market, NOT_APPLICABLE, "prevalence", r["prevalence_24_q2023"]),
                 _row("panel-q2023H1", market, NOT_APPLICABLE, "prevalence", r["prevalence_24_q2023H1"])]
    for period in ("2024", "2026H1"):
        sel = fig1[(fig1["kind"] == "period") & (fig1.index == period)]
        rows.append(_row("fig1", period, NOT_APPLICABLE, "p95_abs_dart_pooled", sel["dart_p95_pooled"].iloc[0]))
    rows.append(_row("drift2026", "LEV6+DART18", NOT_APPLICABLE, "spearman_2026", rho))
    activity_rows = [r for r in extras if r["metric"].startswith("source_activity")]
    for metric, fn in (("source_activity_min", min), ("source_activity_max", max)):
        values = [r["regenerated"] for r in activity_rows if r["metric"] == metric]
        if values:                                                  # D9: over every default legacy run
            rows.append(_row("agg:legacy", "all", "all", metric, fn(values)))
    runs = None if published is None else paper_runs(published)
    return pd.concat([base_metrics(summary), derived_metrics(frames, runs), pd.DataFrame(extras), pd.DataFrame(rows)],
                     ignore_index=True)


@contextmanager
def _warning_log():
    """Count every warning, print each distinct one once, and write the lot to ``warnings.txt``."""
    counts: Counter = Counter()
    original = warnings.showwarning

    def hook(message, category, filename, lineno, file=None, line=None):
        text = f"{category.__name__}: {message}"
        if counts[text] == 0:
            print(f"[warning] {text}", flush=True)
        counts[text] += 1

    warnings.showwarning = hook
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            yield counts
    finally:
        warnings.showwarning = original
        LEGACY_DIR.mkdir(parents=True, exist_ok=True)
        with open(LEGACY_DIR / "warnings.txt", "w") as fh:
            for text, n in counts.most_common():
                fh.write(f"{n}x {text}\n")


def _log(msg: str, t0: float) -> None:
    print(f"[{time.strftime('%H:%M:%S')} +{(time.time() - t0) / 60:5.1f} min] {msg}", flush=True)


def main():
    ensure_dirs()
    LEGACY_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with _warning_log():
        panel = build_panel(load_dev_raw())
        frames: dict = {}
        summ: dict = {}
        extras: list[dict] = []
        last = None
        for a, block, est, seed, variants in run_plan():
            scores = cached_scores(panel, a, block, est, seed)
            for policy, alpha in variants:
                key = (a.id, block, est, seed, policy, alpha)
                frames[key] = _per_node(scores, a, block, est, seed, policy, alpha)
                summ[key] = _summ(frames[key])
            extras += _extras_for(scores, a, block, est, seed)
            phase = (a.id, "seeds" if seed != 42 else "main")
            if phase != last:
                last = phase
                _log(f"{a.id} {phase[1]}: started", t0)
        summary = summary_frame(summ)
        summary.to_csv(LEGACY_DIR / "summary.csv")
        bad = summary[summary["n_nodes"] != 33]
        if len(bad):
            print(f"[check] {len(bad)} run(s) did not score all 33 nodes:\n{bad[['n_nodes']].to_string()}", flush=True)
        twins = twin_check(frames)
        (LEGACY_DIR / "checks.json").write_text(json.dumps(
            {"lc_twins_max_abs_diff_R": twins, "runs_with_fewer_than_33_nodes": len(bad)}, indent=2))
        _log(f"summary.csv written; label-control twins differ by at most {max(twins.values()):.3g} in R", t0)

        table1 = table1_panel(panel)                                  # D6
        table1.to_csv(LEGACY_DIR / "table1_panel.csv")
        fig1 = fig1_volatility(panel)                                 # D7
        fig1.to_csv(LEGACY_DIR / "fig1_volatility.csv")
        window_boundaries(panel).to_csv(LEGACY_DIR / "d10_boundaries.csv", index=False)       # D10
        rho = drift_2026(panel)                                       # D5
        d9 = {r["metric"]: r["regenerated"] for r in extras if r["metric"].startswith("source_activity")
              and (r["analysis"], r["block"], r["estimator"]) == ("2024-2025", "G1_24", "xgb")}
        wide = [r for r in extras if r["metric"].startswith("source_activity")]
        (LEGACY_DIR / "d5_d9.json").write_text(json.dumps(
            {"spearman_2026_W1_vs_log_error": rho, **d9,
             "source_activity_all_designs_min": min(r["regenerated"] for r in wide if r["metric"].endswith("min")),
             "source_activity_all_designs_max": max(r["regenerated"] for r in wide if r["metric"].endswith("max"))},
            indent=2))
        _log("table1, fig1, d10 boundaries, drift and D9 written", t0)

        published = read_published()
        regenerated = regenerated_numbers(summary, frames, extras, table1, fig1, rho,
                                          n_nodes=panel[["market", "node"]].drop_duplicates().shape[0],
                                          published=published)
        regenerated.to_csv(LEGACY_DIR / "regenerated_long.csv", index=False)
        comp = build_comparison(published, regenerated)
        comp.to_csv(LEGACY_DIR / "comparison.csv", index=False)
        counts = comp["status"].value_counts().to_dict()
        _log(f"comparison.csv written: of {len(comp)} published numbers, {counts.get('match', 0)} reproduce as stated, "
             f"{counts.get('match_other_design', 0)} only under another design, {counts.get('mismatch', 0)} do not, "
             f"{counts.get('no_counterpart', 0)} have no counterpart", t0)
        print(comp[~comp["match"]].drop(columns=["note"]).to_string(), flush=True)


if __name__ == "__main__":
    main()
