"""Per-node fit and per-node-hour scores (the July model core, src.transfer), cached so
every cutoff policy for a pipeline reads identical scores (spec invariant I5).

``score_analysis`` returns one row per node-hour that has complete features, for every node that
could be fitted: the ``BASE`` columns (identity, prices and the training-quantile ``label``),
``abs_mean_168h`` (the *unscaled* ``dart_spread_roll_abs_mean_168h_shift_48h``), ``score`` and
``split``.

``split`` is "train", "calib" or "deploy" for rows inside that window of the analysis and
"other" for every row outside all three: pre-train rows, gaps between windows, post-deploy rows,
and the local late hours of 2026-06-30 that fall on UTC 1 July under the half-open UTC windows.
Consumers must select "calib" or "deploy" explicitly and never use ``!= "train"``.
Exception: P4's trailing window (``es/policies.py``) deliberately uses every non-training row.
The three windows are asserted pairwise disjoint, so every row carries exactly one label.

``label_alpha`` is the label quantile (a spike is a top-``label_alpha`` share of |DART| on the
analysis's label window). It is not the cutoff's target alert rate.

A node that cannot be fitted is never dropped silently: ``feasibility`` reports every node and
why, ``score_analysis`` warns naming each skipped node, and ``cached_scores`` repeats that warning
when it serves a cached entry.
"""
from __future__ import annotations

import hashlib
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src import transfer
from src.cache import cached_frame

from .blocks import features as block_features
from .data import raw_files
from .labels import label_training_q

#: Default ``version`` of ``cached_scores``. Bump it whenever anything that changes cached scores
#: changes (model settings, scoring code, panel construction): one bump reaches every caller.
SCORES_VERSION = "v2"

BASE = ["market", "node", "node_type", "timestamp_utc", "settle_date", "settle_hour", "weekday",
        "wclass", "da_lmp", "rt_lmp", "dart_spread", "label"]
OUTPUT_COLUMNS = BASE + ["abs_mean_168h", "score", "split"]
KEY = ["market", "node", "timestamp_utc"]
#: All the cache stores. The other OUTPUT_COLUMNS are re-joined from the panel on (KEY) at read.
CACHED_COLUMNS = ["market", "node", "timestamp_utc", "label", "score", "split"]
_ABS_MEAN_SOURCE = "dart_spread_roll_abs_mean_168h_shift_48h"

FEASIBILITY_COLUMNS = ["market", "node", "n_train", "n_pos", "feasible", "reason"]
#: src.transfer.fit_model declines to fit below this many positive training labels.
MIN_POSITIVES = 20


def xgb_factory(seed: int):
    def make(scale_pos_weight: float):
        return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, reg_lambda=2.0, scale_pos_weight=scale_pos_weight,
                             eval_metric="aucpr", random_state=seed, n_jobs=4)
    return make


def logit_factory(seed: int):
    def make(scale_pos_weight: float):
        return LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    return make


FACTORIES = {"xgb": xgb_factory, "logit": logit_factory}


# --------------------------------------------------------------------------------------------
# Shared steps. score_analysis and feasibility both go through these, so the feasibility report
# cannot drift from what the scoring pipeline actually does.
# --------------------------------------------------------------------------------------------

def _prepare(panel: pd.DataFrame, analysis, block: str, label_alpha: float):
    """Feature list and the rows with complete features, carrying the training-quantile label."""
    feats = block_features(block)
    label, _ = label_training_q(panel, analysis.label_train.mask(panel), label_alpha)
    return feats, panel.assign(label=label).dropna(subset=feats)


def _train_rows(g: pd.DataFrame, analysis) -> pd.DataFrame:
    """A node's training rows: in the train window (``g`` is already feature-complete), label set."""
    return g[analysis.train.mask(g).to_numpy()].dropna(subset=["label"])


def _gate(train: pd.DataFrame) -> tuple[int, int, str]:
    """(n_train, n_pos, reason): src.transfer.fit_model's gate, in the order it applies it.

    ``train`` is feature-complete with a non-NaN label. fit_model declines when there are no
    rows or fewer than 2 label classes (checked first), then when there are under 20 positives.
    score_analysis cross-checks this against the real fit_model on every node it handles.
    """
    n_train = len(train)
    if n_train == 0:
        return 0, 0, "no_train_rows"
    n_pos = int(train["label"].sum())
    if train["label"].nunique() < 2:
        return n_train, n_pos, "one_class"
    if n_pos < MIN_POSITIVES:
        return n_train, n_pos, "lt_20_positives"
    return n_train, n_pos, "ok"


def _panel_nodes(frame: pd.DataFrame) -> list[tuple[str, str]]:
    return sorted(map(tuple, frame[["market", "node"]].drop_duplicates().to_numpy().tolist()))


def _assert_disjoint_windows(frame: pd.DataFrame, analysis) -> None:
    """Raise unless the train, calib and deploy masks are pairwise disjoint on ``frame``."""
    masks = {name: getattr(analysis, name).mask(frame).to_numpy()
             for name in ("train", "calib", "deploy")}
    for a, b in (("train", "calib"), ("train", "deploy"), ("calib", "deploy")):
        n = int((masks[a] & masks[b]).sum())
        if n:
            raise ValueError(f"analysis {analysis.id!r}: the {a} and {b} windows overlap on {n} rows, "
                             "so those rows would carry two split labels")


def _describe(skipped) -> str:
    return "; ".join(f"{m}/{n} ({reason}: n_train={nt}, n_pos={npos})"
                     for m, n, nt, npos, reason in skipped)


def _warn_skipped(skipped, where: str) -> None:
    warnings.warn(f"{where}: {len(skipped)} node(s) not scored, no model was fitted: "
                  f"{_describe(skipped)}", UserWarning, stacklevel=3)


# --------------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------------

def feasibility(panel: pd.DataFrame, analysis, block: str, label_alpha: float = 0.05) -> pd.DataFrame:
    """Which nodes ``score_analysis`` can fit, and why not for the rest.

    One row per (market, node) in ``panel``. ``n_train`` counts the node's training-window rows
    with complete features and a label; ``n_pos`` the positives among them. ``reason`` is "ok",
    "no_train_rows", "one_class" or "lt_20_positives", mirroring src.transfer.fit_model's gate.
    """
    _, p = _prepare(panel, analysis, block, label_alpha)
    found = {key: _gate(_train_rows(g, analysis)) for key, g in p.groupby(["market", "node"], sort=True)}
    rows = []
    for key in _panel_nodes(panel):
        n_train, n_pos, reason = found.get(key, (0, 0, "no_train_rows"))   # no complete-feature rows
        rows.append((*key, n_train, n_pos, reason == "ok", reason))
    return pd.DataFrame(rows, columns=FEASIBILITY_COLUMNS)


def score_analysis(panel: pd.DataFrame, analysis, block: str, estimator: str = "xgb",
                   seed: int = 42, label_alpha: float = 0.05) -> pd.DataFrame:
    """Fit one model per node on its training window and score every feature-complete node-hour.

    Columns: ``OUTPUT_COLUMNS``. ``abs_mean_168h`` is the unscaled rolling feature, whether or not
    the block uses it. ``split`` is "train", "calib" or "deploy", or "other" for rows outside all
    three windows (pre-train rows, gaps, post-deploy rows, and local late hours of 2026-06-30 that
    fall on UTC 1 July): select "calib" or "deploy" explicitly, never ``!= "train"``.
    Exception: P4's trailing window (``es/policies.py``) deliberately uses every non-training row.

    Nodes ``src.transfer.fit_model`` declines to fit are omitted, with one UserWarning naming each
    node and its reason (see ``feasibility``). If no node can be fitted, raises ValueError.
    """
    feats, p = _prepare(panel, analysis, block, label_alpha)
    _assert_disjoint_windows(p, analysis)
    factory = FACTORIES[estimator](seed)
    parts, skipped, seen = [], [], set()
    for (market, node), g in p.groupby(["market", "node"], sort=True):
        seen.add((market, node))
        train = _train_rows(g, analysis)
        n_train, n_pos, reason = _gate(train)
        centre, spread = transfer.fit_scaler(train, feats)
        model = transfer.fit_model(factory, transfer.apply_scaler(train, feats, centre, spread),
                                   feats, "label")
        if (model is None) != (reason != "ok"):
            raise RuntimeError(
                f"feasibility gate out of sync with src.transfer.fit_model for {market}/{node}: "
                f"gate says {reason!r}, fit_model returned {'None' if model is None else 'a model'}")
        if model is None:
            skipped.append((market, node, n_train, n_pos, reason))
            continue
        scored = transfer.score_with_model(model, transfer.apply_scaler(g, feats, centre, spread), feats)
        if scored is None:   # not reachable (g is feature-complete and non-empty); never skip silently
            skipped.append((market, node, n_train, n_pos, "no_scorable_rows"))
            continue
        base = g.loc[scored.index, BASE].copy()
        base["abs_mean_168h"] = g.loc[scored.index, _ABS_MEAN_SOURCE].to_numpy()
        base["score"] = scored["score"].to_numpy()
        base["split"] = np.select(
            [analysis.train.mask(base).to_numpy(), analysis.calib.mask(base).to_numpy(),
             analysis.deploy.mask(base).to_numpy()], ["train", "calib", "deploy"], default="other")
        parts.append(base)
    # Nodes with no complete-feature row at all never reach the loop; they are skipped too.
    skipped += [(m, n, 0, 0, "no_train_rows") for m, n in sorted(set(_panel_nodes(panel)) - seen)]
    skipped.sort(key=lambda r: (r[0], r[1]))
    where = f"score_analysis({analysis.id!r}, {block!r})"
    if not parts:
        raise ValueError(f"{where}: no node is feasible, so there is nothing to score "
                         f"({_describe(skipped) or 'the panel has no nodes'})")
    if skipped:
        _warn_skipped(skipped, where)
    return pd.concat(parts, ignore_index=True)


def _digest(panel: pd.DataFrame, analysis, block: str, label_alpha: float) -> str:
    """12 hex chars identifying everything the scores depend on besides the model settings in the
    cache name: the analysis definition, the feature list, the label quantile and the panel."""
    h = hashlib.sha256()
    for part in (repr(analysis), repr(block_features(block)), repr(float(label_alpha)), str(len(panel))):
        h.update(part.encode())
        h.update(b"\x1f")
    h.update(pd.util.hash_pandas_object(panel[KEY + ["dart_spread"]], index=False).to_numpy().tobytes())
    return h.hexdigest()[:12]


def cache_name(panel: pd.DataFrame, analysis, block: str, estimator: str = "xgb", seed: int = 42,
               label_alpha: float = 0.05) -> str:
    """Cache entry name. The digest makes it depend on the panel (rows, order, spreads) and on the
    analysis definition, not just its id, so a subset panel or edited windows get their own entry."""
    return (f"es_scores__{analysis.id}__{block}__{estimator}__s{seed}__a{label_alpha}"
            f"__{_digest(panel, analysis, block, label_alpha)}")


def _rejoin(slim: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Rebuild score_analysis's frame from the slim cached columns plus the panel's own columns."""
    extra = [c for c in OUTPUT_COLUMNS if c not in CACHED_COLUMNS and c != "abs_mean_168h"]
    side = panel[KEY + extra].assign(abs_mean_168h=panel[_ABS_MEAN_SOURCE].to_numpy())
    try:
        out = slim.merge(side, on=KEY, how="left", validate="many_to_one", indicator=True)
    except pd.errors.MergeError as exc:
        raise ValueError("the panel has duplicate (market, node, timestamp_utc) rows, so cached "
                         "scores cannot be re-joined to it") from exc
    if len(out) != len(slim):
        raise RuntimeError(f"re-joining panel columns changed the row count ({len(slim)} -> {len(out)})")
    if not (out["_merge"] == "both").all():
        raise RuntimeError("cached scores contain rows that are not in the panel")
    return out[OUTPUT_COLUMNS]


def _replay_exclusions(scored: pd.DataFrame, panel: pd.DataFrame, analysis, block: str,
                       label_alpha: float) -> None:
    """On a cache hit score_analysis did not run, so repeat its exclusion warning."""
    missing = sorted(set(_panel_nodes(panel)) - set(_panel_nodes(scored)))
    if not missing:
        return
    keys = pd.MultiIndex.from_arrays([panel["market"], panel["node"]])
    feas = feasibility(panel[keys.isin(missing)], analysis, block, label_alpha)
    if feas["feasible"].any():
        raise RuntimeError("cached scores lack feasible node(s) "
                           f"{feas.loc[feas['feasible'], ['market', 'node']].values.tolist()}; the cache "
                           "entry is out of date with the code, so bump SCORES_VERSION")
    skipped = list(feas[["market", "node", "n_train", "n_pos", "reason"]].itertuples(index=False, name=None))
    _warn_skipped(skipped, f"cached_scores({analysis.id!r}, {block!r}) [cached entry]")


def cached_scores(panel: pd.DataFrame, analysis, block: str, estimator: str = "xgb", seed: int = 42,
                  label_alpha: float = 0.05, version: str = SCORES_VERSION) -> pd.DataFrame:
    """``score_analysis``, cached. Returns exactly its frame (same columns, same order).

    Only ``CACHED_COLUMNS`` are stored; the other columns are re-joined from ``panel``, on a miss
    as well as on a hit. The entry name carries a digest of the panel and the analysis definition
    (see ``cache_name``), and ``version`` defaults to ``SCORES_VERSION``.
    """
    built = []

    def build():
        built.append(True)
        return score_analysis(panel, analysis, block, estimator, seed, label_alpha)[CACHED_COLUMNS]

    slim = cached_frame(name=cache_name(panel, analysis, block, estimator, seed, label_alpha),
                        builder=build, dependencies=raw_files(), version=version)
    out = _rejoin(slim, panel)
    if not built:
        _replay_exclusions(out, panel, analysis, block, label_alpha)
    return out
