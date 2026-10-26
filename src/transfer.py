"""Cross-market transfer experiments (Panel D).

Implements the four arms the plan calls for, for magnitude and direction
separately:

* **within**    — train and evaluate inside one market. The reference point.
* **pooled**    — train on every market, evaluate on each.
* **zero-shot** — train on the source market, evaluate on the target unseen.
* **few-shot**  — train on the source plus a small slice of target *training*
  rows, evaluate on the target.

Transfer degradation is reported relative to the within-market arm.

## Why normalization is the whole problem

A model trained on ISO-NE and applied to NYISO sees the same feature *names*
carrying different units of meaning: a $40 rolling spread mean is ordinary in one
market and extreme in another. Raw-dollar features let a tree learn thresholds
that are simply wrong elsewhere.

:func:`normalize_within_market` therefore rescales every feature using statistics
fitted on **that market's own training rows**. This is not leakage — a
forecaster standing at the information cutoff knows their own market's price
history — and it is what makes the transfer question about *mechanisms* rather
than about price levels.

The un-normalized arm is retained as an ablation, because the size of the gap
between them is itself a result.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from . import evaluation

__all__ = [
    "ARMS",
    "normalize_within_market",
    "run_magnitude_transfer",
    "run_direction_transfer",
    "summarize_degradation",
    "fit_model",
    "fit_scaler",
    "apply_scaler",
    "score_with_model",
]

#: The experiment arms, in reporting order.
ARMS = ("within", "pooled", "zero_shot", "few_shot")

#: Fraction of target training rows revealed to the few-shot arm.
DEFAULT_FEW_SHOT_FRACTION = 0.10


# ---------------------------------------------------------------------
# Cross-market normalization
# ---------------------------------------------------------------------

def normalize_within_market(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    market_column: str = "iso",
    split_column: str = "split_role",
    training_role: str = "train",
    minimum_scale: float = 1e-6,
) -> pd.DataFrame:
    """Rescale features using each market's own training-row statistics.

    Uses median and a MAD-based robust scale rather than mean and standard
    deviation, because the price-history features inherit the heavy tails of the
    spreads themselves and a handful of scarcity hours would otherwise set the
    scale.

    Statistics come only from rows where ``split_column == training_role``, so
    evaluation rows never inform their own normalization.
    """
    result = frame.copy()
    names = list(feature_names)

    for market, group in result.groupby(market_column, sort=False):
        training = group[group[split_column] == training_role]
        if training.empty:
            raise ValueError(f"Market {market!r} has no training rows to fit scales on.")

        centre = training[names].median()
        # 1.4826 makes the MAD comparable to a standard deviation under normality.
        spread = (training[names] - centre).abs().median() * 1.4826
        spread = spread.where(spread > minimum_scale, other=training[names].std())
        spread = spread.fillna(1.0).replace(0.0, 1.0)

        result.loc[group.index, names] = (group[names] - centre) / spread

    return result


# ---------------------------------------------------------------------
# Arm construction
# ---------------------------------------------------------------------

def _training_rows_for_arm(
    frame: pd.DataFrame,
    arm: str,
    source: str,
    target: str,
    few_shot_fraction: float,
    random_seed: int,
) -> pd.DataFrame:
    """Select the training rows each arm is entitled to see.

    Every arm draws exclusively from ``split_role == "train"``, so no arm ever
    touches an evaluation row of any market.
    """
    training = frame[frame["split_role"] == "train"]

    if arm == "within":
        return training[training["iso"] == target]

    if arm == "pooled":
        return training

    if arm == "zero_shot":
        return training[training["iso"] == source]

    if arm == "few_shot":
        source_rows = training[training["iso"] == source]
        target_pool = training[training["iso"] == target]

        # Take the most *recent* slice of target training rows rather than a
        # random sample: a practitioner adapting to a new market would have its
        # recent history, not a random scatter across four years.
        count = max(1, int(len(target_pool) * few_shot_fraction))
        target_slice = target_pool.sort_values("timestamp_utc").tail(count)

        return pd.concat([source_rows, target_slice], ignore_index=False)

    raise ValueError(f"Unknown arm {arm!r}. Known arms: {ARMS}")


def _fit_and_score(
    model_factory,
    train: pd.DataFrame,
    evaluate: pd.DataFrame,
    feature_names: Sequence[str],
    label_column: str,
    score_column: str,
) -> pd.DataFrame | None:
    """Fit one model and attach its scores. Returns None when infeasible."""
    names = list(feature_names)

    train_complete = train.dropna(subset=names)
    evaluate_complete = evaluate.dropna(subset=names)

    if evaluate_complete.empty or train_complete.empty:
        return None
    if train_complete[label_column].nunique() < 2:
        return None

    positives = int(train_complete[label_column].sum())
    negatives = len(train_complete) - positives
    if positives < 20:
        return None

    model = model_factory(scale_pos_weight=negatives / max(positives, 1))
    model.fit(
        train_complete[names].to_numpy(dtype=float),
        train_complete[label_column].to_numpy(dtype=int),
    )

    scored = evaluate_complete.copy()
    scored[score_column] = model.predict_proba(
        evaluate_complete[names].to_numpy(dtype=float)
    )[:, 1]
    return scored


def fit_model(model_factory, train: pd.DataFrame, feature_names: Sequence[str],
              label_column: str):
    """Fit one model on ``train`` and return it, or ``None`` when infeasible.

    Split out of :func:`_fit_and_score` so a single fit can be scored against
    several frames — the mechanism Analysis A's transfer arms rely on to
    isolate threshold-origin as the only difference between them. Two arms
    that call :func:`score_with_model` on the same returned model are, by
    construction, comparing the same fitted model under two calibration
    policies, never two independently fitted models.

    Feasibility gate matches :func:`_fit_and_score`: at least 20 positive
    labels in the training rows with complete features.
    """
    names = list(feature_names)
    train_complete = train.dropna(subset=names)

    if train_complete.empty or train_complete[label_column].nunique() < 2:
        return None

    positives = int(train_complete[label_column].sum())
    negatives = len(train_complete) - positives
    if positives < 20:
        return None

    model = model_factory(scale_pos_weight=negatives / max(positives, 1))
    model.fit(
        train_complete[names].to_numpy(dtype=float),
        train_complete[label_column].to_numpy(dtype=int),
    )
    return model


def score_with_model(model, frame: pd.DataFrame, feature_names: Sequence[str],
                     score_column: str = "score") -> pd.DataFrame | None:
    """Score a previously fitted ``model`` on ``frame``. ``None`` if empty."""
    names = list(feature_names)
    complete = frame.dropna(subset=names)
    if complete.empty:
        return None

    scored = complete.copy()
    scored[score_column] = model.predict_proba(
        complete[names].to_numpy(dtype=float)
    )[:, 1]
    return scored


# ---------------------------------------------------------------------
# Magnitude transfer
# ---------------------------------------------------------------------

def run_magnitude_transfer(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    model_factory,
    markets: Sequence[str] | None = None,
    normalize: bool = True,
    few_shot_fraction: float = DEFAULT_FEW_SHOT_FRACTION,
    alert_fractions: Sequence[float] = (0.01, 0.05),
    random_seed: int = 42,
) -> pd.DataFrame:
    """Run every arm, fold, and market pair for the magnitude task.

    Args:
        frame: Fold-expanded labelled rows with features joined.
        feature_names: The shared-core feature set to transfer.
        model_factory: Callable taking ``scale_pos_weight`` and returning an
            unfitted estimator with ``fit`` and ``predict_proba``.
        markets: Markets in scope. Defaults to every market present.
        normalize: Apply per-market robust normalization before fitting.
        few_shot_fraction: Share of target training rows revealed to few-shot.
        alert_fractions: Alert budgets for the operating-point metrics.

    Returns:
        One row per fold, target market, and arm.
    """
    markets = list(markets) if markets else sorted(frame["iso"].unique())
    if len(markets) < 2:
        raise ValueError(f"Transfer needs at least two markets, got {markets}")

    records: list[dict] = []

    for fold_id, fold_frame in frame.groupby("fold_id", sort=True):
        working = (
            normalize_within_market(fold_frame, feature_names)
            if normalize
            else fold_frame.copy()
        )

        for target in markets:
            sources = [market for market in markets if market != target]
            evaluate = working[
                (working["iso"] == target) & (working["split_role"] == "evaluation")
            ]
            if evaluate.empty:
                continue

            for arm in ARMS:
                # With more than two markets, zero-shot and few-shot are run once
                # per source so the pairwise matrix stays legible.
                arm_sources = sources if arm in ("zero_shot", "few_shot") else [None]

                for source in arm_sources:
                    train = _training_rows_for_arm(
                        working,
                        arm=arm,
                        source=source or target,
                        target=target,
                        few_shot_fraction=few_shot_fraction,
                        random_seed=random_seed,
                    )

                    scored = _fit_and_score(
                        model_factory,
                        train,
                        evaluate,
                        feature_names,
                        label_column="label_high_spread",
                        score_column="score",
                    )
                    if scored is None:
                        continue

                    metrics = evaluation.magnitude_metrics(
                        scored, alert_fractions=alert_fractions
                    )
                    records.append(
                        {
                            "fold_id": fold_id,
                            "target_market": target,
                            "source_market": source or target,
                            "arm": arm,
                            "normalized": normalize,
                            "train_rows": int(len(train)),
                            **metrics,
                        }
                    )

    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------
# Direction transfer
# ---------------------------------------------------------------------

def run_direction_transfer(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    model_factory,
    markets: Sequence[str] | None = None,
    normalize: bool = True,
    few_shot_fraction: float = DEFAULT_FEW_SHOT_FRACTION,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Run every arm for conditional INC/DEC direction on realized events.

    Restricted to ``direction_eligible`` rows — high-spread events with a
    non-zero spread. This is the harder secondary result; given the fold-level
    sign instability between markets, degradation here is expected to be severe
    and that is itself the finding.
    """
    markets = list(markets) if markets else sorted(frame["iso"].unique())
    eligible = frame[frame["direction_eligible"]].copy()

    records: list[dict] = []

    for fold_id, fold_frame in eligible.groupby("fold_id", sort=True):
        working = (
            normalize_within_market(fold_frame, feature_names)
            if normalize
            else fold_frame.copy()
        )

        for target in markets:
            sources = [market for market in markets if market != target]
            evaluate = working[
                (working["iso"] == target) & (working["split_role"] == "evaluation")
            ]
            if len(evaluate) < 20:
                continue

            for arm in ARMS:
                arm_sources = sources if arm in ("zero_shot", "few_shot") else [None]

                for source in arm_sources:
                    train = _training_rows_for_arm(
                        working,
                        arm=arm,
                        source=source or target,
                        target=target,
                        few_shot_fraction=few_shot_fraction,
                        random_seed=random_seed,
                    )
                    if len(train) < 40:
                        continue

                    scored = _fit_and_score(
                        model_factory,
                        train,
                        evaluate,
                        feature_names,
                        label_column="label_direction_binary",
                        score_column="direction_score",
                    )
                    if scored is None:
                        continue

                    metrics = evaluation.direction_metrics(scored)
                    records.append(
                        {
                            "fold_id": fold_id,
                            "target_market": target,
                            "source_market": source or target,
                            "arm": arm,
                            "normalized": normalize,
                            "train_rows": int(len(train)),
                            **metrics,
                        }
                    )

    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------
# Degradation reporting
# ---------------------------------------------------------------------

def summarize_degradation(
    results: pd.DataFrame,
    metric: str = "auprc",
    reference_arm: str = "within",
) -> pd.DataFrame:
    """Express every arm's performance relative to the within-market reference.

    Folds are averaged with equal weight before the ratio is taken, since event
    prevalence varies more than threefold across folds and a row-weighted pool
    would be dominated by the most dislocated half-year.

    Returns absolute and relative degradation. ``retention`` is the share of
    within-market performance the arm preserves — the headline transfer number.
    """
    if metric not in results.columns:
        raise KeyError(f"Metric {metric!r} not in results. Available: {sorted(results.columns)}")

    macro = (
        results.groupby(["target_market", "source_market", "arm"], sort=True)[metric]
        .mean()
        .reset_index()
    )

    reference = (
        macro[macro["arm"] == reference_arm]
        .set_index("target_market")[metric]
        .rename("reference")
    )

    summary = macro.merge(reference, left_on="target_market", right_index=True, how="left")
    summary["degradation_absolute"] = summary[metric] - summary["reference"]
    summary["retention"] = summary[metric] / summary["reference"].replace(0.0, np.nan)

    order = {arm: index for index, arm in enumerate(ARMS)}
    summary["_order"] = summary["arm"].map(order)

    return (
        summary.sort_values(["target_market", "_order", "source_market"])
        .drop(columns="_order")
        .reset_index(drop=True)
    )


def fit_scaler(train: pd.DataFrame, feature_names: Sequence[str],
               minimum_scale: float = 1e-6) -> tuple[pd.Series, pd.Series]:
    """Fit a median/MAD scaler on ``train`` rows. Returns (centre, spread).

    Split out so a scaler fitted on one market's training rows can be applied
    to another market's rows. That separation is what makes the four-arm
    design of PROTOCOL_AMENDMENT_2026-07-20_scaler_orthogonalization.md
    expressible: feature scaling and threshold placement become independent
    choices instead of both being tied to the target market by
    :func:`normalize_within_market`.
    """
    names = list(feature_names)
    centre = train[names].median()
    spread = (train[names] - centre).abs().median() * 1.4826
    spread = spread.where(spread > minimum_scale, other=train[names].std())
    spread = spread.fillna(1.0).replace(0.0, 1.0)
    return centre, spread


def apply_scaler(frame: pd.DataFrame, feature_names: Sequence[str],
                 centre: pd.Series, spread: pd.Series) -> pd.DataFrame:
    """Apply a fitted scaler to ``frame``, returning a copy."""
    names = list(feature_names)
    result = frame.copy()
    result[names] = (frame[names] - centre) / spread
    return result
