"""Threshold calibration with explicit provenance, for Analysis A.

Exists because the v1 implementation silently calibrated its "zero-shot" arm on
target-market scores — see ``DEFECT_2026-07-20_zero_shot_threshold_mislabelled.md``.
The defect was invisible in the result table: the arm was named ``zero_shot`` and
nothing recorded where its threshold had actually come from.

Every threshold produced here therefore carries a :class:`ThresholdProvenance`
stating which market's scores it came from, which folds, how many observations,
and which fitted model. Provenance travels into the result table, so the same
mistake cannot be made silently again.

Three arms, and the distinction between the last two is the point:

``within``
    Target market's own model, threshold from the target's F1/F2 scores.

``zero_shot_source_threshold``
    Model trained on the source markets, threshold from the **source markets'**
    F1/F2 score distribution, applied unchanged to the held-out target. Pure
    zero-shot deployment: no target-market row is touched at any stage.

``zero_label_target_calibrated``
    The **same fitted model** as above, but the threshold comes from the held-out
    target's own *unlabeled* F1/F2 scores. Not zero-shot — it needs target
    feature history — but it needs no target labels.

Comparing the last two isolates probability-scale transfer from ranking
transfer, which is the decomposition the corrected experiment exists to make.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

import pandas as pd

__all__ = [
    "ThresholdProvenance",
    "WITHIN",
    "ZERO_SHOT_SOURCE",
    "ZERO_LABEL_TARGET",
    "ARMS",
    "OUTCOME_COLUMNS",
    "calibrate_threshold",
    "assert_no_target_rows",
    "assert_no_outcome_columns",
    "transfer_model_identifier",
]

WITHIN = "within"
ZERO_SHOT_SOURCE = "zero_shot_source_threshold"
ZERO_LABEL_TARGET = "zero_label_target_calibrated"

#: Fixed reporting order. The two transfer arms sit adjacent deliberately — they
#: share a fitted model and differ only in threshold origin.
ARMS = (WITHIN, ZERO_SHOT_SOURCE, ZERO_LABEL_TARGET)

#: Columns that encode an outcome. None of these may inform a threshold.
OUTCOME_COLUMNS = (
    "label_high_spread",
    "label_high_spread_conditional",
    "label_direction_class",
    "label_direction_binary",
    "label_upper_tail",
    "label_lower_tail",
    "label_any_tail",
    "label_direction",
    "dart_spread",
    "abs_dart_spread",
    "conditional_magnitude_score",
    "da_lmp",
    "rt_lmp",
)


@dataclass(frozen=True)
class ThresholdProvenance:
    """A frozen threshold together with a record of where it came from."""

    value: float
    origin_arm: str
    origin_market: str
    calibration_folds: str
    calibration_sample_size: int
    nominal_budget: float
    model_identifier: str

    def as_record(self) -> dict:
        """Flatten to ``threshold_*`` result-table columns."""
        payload = asdict(self)
        return {
            "threshold_value": payload["value"],
            "threshold_origin_arm": payload["origin_arm"],
            "threshold_origin_market": payload["origin_market"],
            "threshold_calibration_folds": payload["calibration_folds"],
            "threshold_calibration_sample_size": payload["calibration_sample_size"],
            "nominal_budget": payload["nominal_budget"],
            "model_identifier": payload["model_identifier"],
        }


def transfer_model_identifier(target: str, specification: str,
                              held_out_market: str, fold: str) -> str:
    """Identify the model trained with ``held_out_market`` excluded.

    Both transfer arms must receive the identical string: they are the same
    fitted model evaluated under two threshold policies. A test asserts this,
    because refitting for the second arm would silently reintroduce a difference
    that has nothing to do with calibration.
    """
    return f"{target}|{specification}|transfer|excl={held_out_market}|{fold}"


def assert_no_target_rows(frame: pd.DataFrame, target_market: str,
                          context: str, market_column: str = "iso") -> None:
    """Raise if ``frame`` contains any row from the held-out target market.

    Guards the pure zero-shot arm. This is the specific check that would have
    caught the v1 defect: there, the calibration frame was *entirely* target
    rows while the arm was labelled zero-shot.
    """
    if market_column not in frame.columns:
        raise KeyError(
            f"{context}: cannot verify market isolation, no '{market_column}' column"
        )
    contaminating = frame[market_column] == target_market
    if bool(contaminating.any()):
        raise ValueError(
            f"{context}: pure zero-shot calibration touched {int(contaminating.sum())} "
            f"row(s) from the held-out target market {target_market!r}. The "
            f"threshold must come from source-market scores only."
        )


def assert_no_outcome_columns(frame: pd.DataFrame, context: str,
                              columns: Sequence[str] = OUTCOME_COLUMNS) -> None:
    """Raise if any outcome column is present on a calibration frame.

    Guards the zero-label arm. Target *scores* are permitted; target labels,
    DART values, and prices are not. The columns are dropped before the call
    rather than merely left unused, so their absence is verifiable.
    """
    present = [column for column in columns if column in frame.columns]
    if present:
        raise ValueError(
            f"{context}: zero-label calibration frame carries outcome column(s) "
            f"{present}. Drop them before calibrating — unlabeled target scores "
            f"are admissible, outcomes are not."
        )


def calibrate_threshold(scores: pd.Series, budget: float, *, origin_arm: str,
                        origin_market: str, calibration_folds: str,
                        model_identifier: str) -> ThresholdProvenance:
    """Take the ``1 - budget`` quantile of ``scores`` and record its provenance.

    Args:
        scores: Model scores on the calibration slice. Which slice that is
            differs by arm and is the substance of the distinction between them.
        budget: Nominal alert budget, e.g. ``0.05`` for a 5% alert rate.
        origin_arm: One of :data:`ARMS`.
        origin_market: Market(s) whose scores produced this threshold. For the
            pure zero-shot arm this is the source markets, never the target.
    """
    if origin_arm not in ARMS:
        raise ValueError(f"unknown arm {origin_arm!r}; expected one of {ARMS}")
    if not 0.0 < budget < 1.0:
        raise ValueError(f"budget must lie in (0, 1), got {budget}")

    clean = pd.Series(scores).dropna()
    if clean.empty:
        raise ValueError(f"{origin_arm}: no scores to calibrate on")

    return ThresholdProvenance(
        value=float(clean.quantile(1.0 - budget)),
        origin_arm=origin_arm,
        origin_market=origin_market,
        calibration_folds=calibration_folds,
        calibration_sample_size=int(len(clean)),
        nominal_budget=float(budget),
        model_identifier=model_identifier,
    )
