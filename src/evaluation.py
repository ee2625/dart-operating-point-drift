"""Evaluation metrics for the magnitude and direction tasks.

High-spread hours are rare — roughly 5% of the sample by construction — so plain
accuracy is uninformative and is deliberately absent from the headline metrics.
The reported quantities are those in the revised plan's Panels A–D: ranking
quality (AUPRC and its lift over the base rate), operating-point behaviour
(precision and recall at fixed alert budgets), event-level coverage
(episode-start recall), and economic concentration (how much of the total
absolute spread the alerts capture).

All confidence intervals come from a *moving-block* bootstrap over time rather
than an i.i.d. bootstrap, because DART spreads are strongly autocorrelated and
resampling hours independently would understate the intervals substantially.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

__all__ = [
    "DEFAULT_ALERT_FRACTIONS",
    "block_bootstrap_interval",
    "common_rows",
    "direction_metrics",
    "magnitude_metrics",
    "metrics_by_group",
    "precision_at_k",
    "recall_at_k",
]

#: Alert budgets reported at every operating point, as a share of all hours.
DEFAULT_ALERT_FRACTIONS = (0.01, 0.05)

#: Block length for the temporal bootstrap. One week of hours, matching the
#: master notebook's ``bootstrap_block_hours``.
DEFAULT_BLOCK_HOURS = 168


# ---------------------------------------------------------------------
# Operating-point helpers
# ---------------------------------------------------------------------

def _top_k_mask(scores: np.ndarray, fraction: float) -> np.ndarray:
    """Flag the highest-scoring ``fraction`` of rows.

    Ranks descending and takes at least one row, so a small evaluation window
    still produces a defined operating point.
    """
    count = max(1, int(np.floor(len(scores) * fraction)))
    # argpartition is O(n) and sufficient — the exact order within the top-k
    # does not matter, only membership.
    cutoff_index = np.argpartition(-scores, count - 1)[:count]
    mask = np.zeros(len(scores), dtype=bool)
    mask[cutoff_index] = True
    return mask


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, fraction: float) -> float:
    """Share of the top ``fraction`` of scored hours that are true events."""
    mask = _top_k_mask(scores, fraction)
    return float(y_true[mask].mean()) if mask.any() else np.nan


def recall_at_k(y_true: np.ndarray, scores: np.ndarray, fraction: float) -> float:
    """Share of all true events falling in the top ``fraction`` of scored hours."""
    positives = y_true.sum()
    if positives == 0:
        return np.nan
    mask = _top_k_mask(scores, fraction)
    return float(y_true[mask].sum() / positives)


# ---------------------------------------------------------------------
# Panel A — magnitude detection
# ---------------------------------------------------------------------

def magnitude_metrics(
    frame: pd.DataFrame,
    score_column: str = "score",
    label_column: str = "label_high_spread",
    spread_column: str = "abs_dart_spread",
    threshold: float | None = None,
    alert_fractions: Sequence[float] = DEFAULT_ALERT_FRACTIONS,
) -> dict[str, float]:
    """Compute the Panel A metrics for one evaluation slice.

    Args:
        frame: Scored evaluation rows.
        score_column: Predicted probability or score.
        label_column: Binary high-spread target.
        spread_column: Absolute spread, used for the economic concentration
            metrics. Optional — pass a missing column name to skip them.
        threshold: A frozen decision threshold selected on development data. When
            given, precision, recall, and F1 are reported at that threshold.
        alert_fractions: Alert budgets for the precision@k and recall@k metrics.

    Returns:
        Metric name to value. ``nan`` where a metric is undefined, for example
        AUPRC on a slice containing no events.
    """
    y_true = frame[label_column].to_numpy(dtype=int)
    scores = frame[score_column].to_numpy(dtype=float)

    positives = int(y_true.sum())
    base_rate = float(y_true.mean()) if len(y_true) else np.nan

    results: dict[str, float] = {
        "row_count": int(len(y_true)),
        "event_count": positives,
        "base_rate": base_rate,
    }

    # Ranking metrics are undefined when the slice is single-class.
    if positives == 0 or positives == len(y_true):
        results.update({"auprc": np.nan, "auprc_lift": np.nan, "roc_auc": np.nan})
    else:
        auprc = float(average_precision_score(y_true, scores))
        results["auprc"] = auprc
        # Lift over a random ranker, whose expected AUPRC is the base rate.
        results["auprc_lift"] = auprc / base_rate if base_rate > 0 else np.nan
        results["roc_auc"] = float(roc_auc_score(y_true, scores))

    for fraction in alert_fractions:
        tag = f"{fraction:.0%}".replace("%", "pct")
        results[f"precision_at_{tag}"] = precision_at_k(y_true, scores, fraction)
        results[f"recall_at_{tag}"] = recall_at_k(y_true, scores, fraction)

    if threshold is not None:
        predicted = (scores >= threshold).astype(int)
        results["frozen_threshold"] = float(threshold)
        results["frozen_alert_rate"] = float(predicted.mean())
        results["frozen_precision"] = float(
            precision_score(y_true, predicted, zero_division=0)
        )
        results["frozen_recall"] = float(recall_score(y_true, predicted, zero_division=0))
        precision = results["frozen_precision"]
        recall = results["frozen_recall"]
        results["frozen_f1"] = (
            float(2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )

    # Three complementary coverage views of the same alerts.
    #
    #   hour recall            — share of event hours alerted (already above)
    #   episode detection      — share of episodes with at least one alerted hour
    #   episode-start recall   — share of episodes alerted at their first hour
    #
    # Detection recall is the primary operational number: every delivery hour is
    # forecast before the auction, so catching hour three of a four-hour episode
    # is still actionable even when hour one was missed. Start recall is the
    # stricter secondary number, and the gap between the two says how much of the
    # model's coverage comes from reacting late to an episode already underway.
    if "episode_id" in frame.columns:
        episode_ids = frame["episode_id"]
        total_episodes = episode_ids.nunique(dropna=True)

        for fraction in alert_fractions:
            tag = f"{fraction:.0%}".replace("%", "pct")

            if total_episodes == 0:
                results[f"episode_detection_recall_at_{tag}"] = np.nan
                results[f"episode_count"] = 0
                continue

            mask = _top_k_mask(scores, fraction)
            detected = episode_ids[mask].nunique(dropna=True)
            results[f"episode_detection_recall_at_{tag}"] = float(detected / total_episodes)

        results["episode_count"] = int(total_episodes)

    if "is_episode_start" in frame.columns:
        starts = frame["is_episode_start"].fillna(False).to_numpy(dtype=bool)
        for fraction in alert_fractions:
            tag = f"{fraction:.0%}".replace("%", "pct")
            if starts.sum() == 0:
                results[f"episode_start_recall_at_{tag}"] = np.nan
            else:
                mask = _top_k_mask(scores, fraction)
                results[f"episode_start_recall_at_{tag}"] = float(
                    (mask & starts).sum() / starts.sum()
                )

    # Economic concentration: how severe are the alerted hours, and how much of
    # the period's total absolute spread do they account for?
    if spread_column in frame.columns:
        absolute = frame[spread_column].to_numpy(dtype=float)
        total_absolute = np.nansum(absolute)
        for fraction in alert_fractions:
            tag = f"{fraction:.0%}".replace("%", "pct")
            mask = _top_k_mask(scores, fraction)
            alerted = absolute[mask]
            results[f"mean_abs_dart_alerts_at_{tag}"] = float(np.nanmean(alerted))
            results[f"median_abs_dart_alerts_at_{tag}"] = float(np.nanmedian(alerted))
            results[f"captured_abs_dart_share_at_{tag}"] = (
                float(np.nansum(alerted) / total_absolute) if total_absolute > 0 else np.nan
            )

    return results


# ---------------------------------------------------------------------
# Panel C — conditional direction
# ---------------------------------------------------------------------

def direction_metrics(
    frame: pd.DataFrame,
    score_column: str = "direction_score",
    label_column: str = "label_direction_binary",
    positive_class: str = "INC",
) -> dict[str, float]:
    """Compute the Panel C metrics on rows eligible for the direction task.

    ``label_column`` is 1 for INC and 0 for DEC. The majority-direction baseline
    is reported alongside accuracy, because a directional model that fails to
    beat it has found nothing beyond the structural class imbalance.
    """
    eligible = frame[frame[label_column].notna()]
    if eligible.empty:
        return {"row_count": 0, "directional_accuracy": np.nan}

    y_true = eligible[label_column].to_numpy(dtype=int)
    scores = eligible[score_column].to_numpy(dtype=float)
    predicted = (scores >= 0.5).astype(int)

    inc_share = float(y_true.mean())
    # NOTE: this is computed on the evaluation slice itself, so it is an ORACLE
    # majority rule — it knows which sign dominated the period being scored. It
    # is a descriptive upper bound on what any constant-sign rule could achieve,
    # not something a forecaster could implement. The implementable baselines
    # are fitted on training rows by labels.build_direction_baselines.
    oracle_majority_baseline = max(inc_share, 1.0 - inc_share)

    results: dict[str, float] = {
        "row_count": int(len(y_true)),
        "inc_count": int(y_true.sum()),
        "dec_count": int(len(y_true) - y_true.sum()),
        "inc_share": inc_share,
        "dec_share": 1.0 - inc_share,
        "majority_direction": positive_class if inc_share >= 0.5 else "DEC",
        "oracle_majority_baseline": oracle_majority_baseline,
        "directional_accuracy": float((predicted == y_true).mean()),
    }

    # Gap to the oracle constant-sign rule. Negative values are expected and do
    # not by themselves mean the model is useless: a model can carry real
    # directional information (balanced accuracy and MCC well above chance) while
    # still losing on raw accuracy to a lopsided constant rule.
    results["accuracy_over_oracle_majority"] = (
        results["directional_accuracy"] - oracle_majority_baseline
    )

    if len(np.unique(y_true)) < 2:
        results.update(
            {
                "balanced_accuracy": np.nan,
                "matthews_corrcoef": np.nan,
                "auprc_inc": np.nan,
            }
        )
    else:
        results["balanced_accuracy"] = float(balanced_accuracy_score(y_true, predicted))
        results["matthews_corrcoef"] = float(matthews_corrcoef(y_true, predicted))
        results["auprc_inc"] = float(average_precision_score(y_true, scores))

    results["inc_precision"] = float(precision_score(y_true, predicted, pos_label=1, zero_division=0))
    results["inc_recall"] = float(recall_score(y_true, predicted, pos_label=1, zero_division=0))
    results["dec_precision"] = float(precision_score(y_true, predicted, pos_label=0, zero_division=0))
    results["dec_recall"] = float(recall_score(y_true, predicted, pos_label=0, zero_division=0))

    matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
    results["confusion_dec_dec"] = int(matrix[0, 0])
    results["confusion_dec_inc"] = int(matrix[0, 1])
    results["confusion_inc_dec"] = int(matrix[1, 0])
    results["confusion_inc_inc"] = int(matrix[1, 1])

    return results


# ---------------------------------------------------------------------
# Grouped reporting
# ---------------------------------------------------------------------

def metrics_by_group(
    frame: pd.DataFrame,
    by: Sequence[str],
    metric_function=magnitude_metrics,
    **kwargs,
) -> pd.DataFrame:
    """Apply a metric function to each group and stack the results.

    Used for the seasonal and per-fold breakdowns the plan asks for, e.g.
    ``metrics_by_group(scored, by=["iso", "season"])``.
    """
    records = []
    for keys, group in frame.groupby(list(by), sort=True):
        record = dict(zip(by, keys if isinstance(keys, tuple) else (keys,)))
        record.update(metric_function(group, **kwargs))
        records.append(record)

    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------
# Temporal uncertainty
# ---------------------------------------------------------------------

def block_bootstrap_interval(
    frame: pd.DataFrame,
    metric_function=magnitude_metrics,
    metric_name: str = "auprc",
    block_hours: int = DEFAULT_BLOCK_HOURS,
    repetitions: int = 1000,
    confidence_level: float = 0.95,
    time_column: str = "timestamp_utc",
    random_seed: int = 42,
    **kwargs,
) -> dict[str, float]:
    """Moving-block bootstrap confidence interval for a single metric.

    Contiguous blocks of ``block_hours`` are resampled with replacement, which
    preserves the short-range dependence in both the spreads and the model
    scores. An i.i.d. bootstrap over hours would produce intervals that are far
    too narrow here.

    Returns the point estimate, the percentile interval bounds, and the number of
    replicates that produced a defined value.
    """
    rng = np.random.default_rng(random_seed)

    ordered = frame.sort_values(time_column).reset_index(drop=True)
    row_count = len(ordered)

    point_estimate = metric_function(ordered, **kwargs).get(metric_name, np.nan)

    if row_count < block_hours * 2:
        # Too short to resample meaningfully; report the point estimate only
        # rather than a falsely precise interval.
        return {
            "metric": metric_name,
            "point_estimate": point_estimate,
            "lower_bound": np.nan,
            "upper_bound": np.nan,
            "valid_replicates": 0,
            "note": "series too short for block bootstrap",
        }

    block_count = int(np.ceil(row_count / block_hours))
    maximum_start = row_count - block_hours

    replicates: list[float] = []
    for _ in range(repetitions):
        starts = rng.integers(0, maximum_start + 1, size=block_count)
        indices = np.concatenate([np.arange(start, start + block_hours) for start in starts])
        sample = ordered.iloc[indices[:row_count]]

        value = metric_function(sample, **kwargs).get(metric_name, np.nan)
        if np.isfinite(value):
            replicates.append(value)

    if not replicates:
        return {
            "metric": metric_name,
            "point_estimate": point_estimate,
            "lower_bound": np.nan,
            "upper_bound": np.nan,
            "valid_replicates": 0,
            "note": "no replicate produced a defined metric",
        }

    alpha = 1.0 - confidence_level
    return {
        "metric": metric_name,
        "point_estimate": point_estimate,
        "lower_bound": float(np.quantile(replicates, alpha / 2)),
        "upper_bound": float(np.quantile(replicates, 1 - alpha / 2)),
        "valid_replicates": len(replicates),
        "block_hours": block_hours,
        "repetitions": repetitions,
    }


def paired_block_bootstrap_difference(
    scored_a: pd.DataFrame,
    scored_b: pd.DataFrame,
    metric_function=None,
    label_column: str = "label_high_spread",
    score_column: str = "score",
    key_columns: Sequence[str] = ("timestamp_utc",),
    block_hours: int = DEFAULT_BLOCK_HOURS,
    repetitions: int = 500,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> dict:
    """Interval for the difference ``B − A`` between two models on identical rows.

    Comparing two independently computed marginal intervals is the wrong test.
    Both models score the same evaluation period and the same events, so their
    errors are strongly correlated — overlapping marginal intervals do not imply
    the difference is insignificant, and non-overlapping ones overstate it.

    This resamples the same blocks for both models and takes the difference
    within each replicate, which cancels the shared period effect and yields a
    much tighter, correctly specified interval.

    Args:
        scored_a: Baseline model's scored rows.
        scored_b: Comparison model's scored rows.
        metric_function: Callable taking ``(y_true, scores)``. Defaults to
            average precision.
        label_column: Binary target, taken from ``scored_a``.
        score_column: Score column present in both frames.
        key_columns: Columns identifying a row for the inner join.
        block_hours: Contiguous block length, preserving autocorrelation.

    Returns:
        Point estimate, interval bounds, replicate count, and ``excludes_zero``
        — the significance verdict.
    """
    if metric_function is None:
        metric_function = average_precision_score

    keys = list(key_columns)
    merged = scored_a[keys + [label_column, score_column]].merge(
        scored_b[keys + [score_column]], on=keys, suffixes=("_a", "_b")
    )

    undefined = {
        "point_estimate": np.nan,
        "lower_bound": np.nan,
        "upper_bound": np.nan,
        "valid_replicates": 0,
        "excludes_zero": False,
    }
    if merged.empty:
        return {**undefined, "note": "no common rows"}

    merged = merged.sort_values(keys).reset_index(drop=True)
    y_true = merged[label_column].to_numpy(dtype=int)
    scores_a = merged[f"{score_column}_a"].to_numpy(dtype=float)
    scores_b = merged[f"{score_column}_b"].to_numpy(dtype=float)

    def difference(indices: np.ndarray) -> float:
        sample = y_true[indices]
        if sample.sum() == 0 or sample.sum() == len(sample):
            return np.nan
        return float(
            metric_function(sample, scores_b[indices])
            - metric_function(sample, scores_a[indices])
        )

    point_estimate = difference(np.arange(len(merged)))

    row_count = len(merged)
    if row_count < block_hours * 2:
        return {**undefined, "point_estimate": point_estimate, "note": "series too short"}

    rng = np.random.default_rng(random_seed)
    block_count = int(np.ceil(row_count / block_hours))
    maximum_start = row_count - block_hours

    replicates: list[float] = []
    for _ in range(repetitions):
        starts = rng.integers(0, maximum_start + 1, size=block_count)
        indices = np.concatenate([np.arange(s, s + block_hours) for s in starts])[:row_count]
        value = difference(indices)
        if np.isfinite(value):
            replicates.append(value)

    if not replicates:
        return {**undefined, "point_estimate": point_estimate}

    alpha = 1.0 - confidence_level
    lower = float(np.quantile(replicates, alpha / 2))
    upper = float(np.quantile(replicates, 1 - alpha / 2))

    return {
        "point_estimate": point_estimate,
        "lower_bound": lower,
        "upper_bound": upper,
        "valid_replicates": len(replicates),
        "excludes_zero": bool(lower > 0 or upper < 0),
        "block_hours": block_hours,
    }


# ---------------------------------------------------------------------
# Common-row comparison
# ---------------------------------------------------------------------

def common_rows(
    frames: dict[str, pd.DataFrame],
    key_columns: Sequence[str] = ("fold_id", "iso", "node", "timestamp_utc"),
) -> dict[str, pd.DataFrame]:
    """Restrict every frame to the rows all of them share.

    Feature sets differ in coverage: a model using load forecasts can only score
    hours where those forecasts exist. Comparing it against a history-only model
    on its own larger sample confounds coverage with skill. Every model
    comparison therefore reports both the model's full deployable sample and this
    common intersection.

    Raises:
        ValueError: if the intersection is empty.
    """
    if not frames:
        return {}

    keys = list(key_columns)
    index_sets = [set(map(tuple, frame[keys].to_numpy())) for frame in frames.values()]
    shared = set.intersection(*index_sets)

    if not shared:
        raise ValueError(
            "No rows are common to all feature sets. Check that the key columns "
            f"{keys} are populated consistently across frames."
        )

    result = {}
    for name, frame in frames.items():
        mask = frame[keys].apply(tuple, axis=1).isin(shared)
        result[name] = frame[mask].sort_values(keys).reset_index(drop=True)

    return result
