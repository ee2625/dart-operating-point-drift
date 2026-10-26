"""Target construction for the magnitude-then-direction task hierarchy.

The collaborator's pipeline targets *signed* tails: separate upper and lower
2.5% quantiles of the DART spread. The revised plan reframes the problem as

1. **magnitude** — is ``|DA - RT|`` unusually large? and
2. **direction** — conditional on a large spread, is it INC or DEC?

This module builds both targets alongside the signed-tail labels, never in place
of them. Every threshold is estimated on fold training rows only and frozen
before evaluation, matching the existing information policy.

Sign convention, inherited from the master notebook
(``spread_definition = "da_lmp_minus_rt_lmp"``):

* ``dart_spread = da_lmp - rt_lmp``
* **INC** — positive spread, day-ahead above real-time
* **DEC** — negative spread, real-time above day-ahead
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from . import cache

__all__ = [
    "INC",
    "DEC",
    "EPISODE_MAXIMUM_GAP_HOURS",
    "MARKET_TIMEZONES",
    "add_episode_structure",
    "add_peak_indicator",
    "add_season",
    "assign_season",
    "audit_direction_balance",
    "audit_target_structure",
    "build_direction_baselines",
    "build_direction_labels",
    "build_magnitude_labels",
    "build_magnitude_thresholds",
]

INC = "INC"
DEC = "DEC"

#: Grouping used for every train-only estimate.
THRESHOLD_SCOPE = ("fold_id", "iso", "node")

#: Rows the thresholds may be estimated from.
TRAINING_SPLIT_ROLE = "train"


# ---------------------------------------------------------------------
# Season handling
# ---------------------------------------------------------------------

def assign_season(month: pd.Series) -> pd.Series:
    """Map delivery months onto winter, summer, and shoulder.

    Winter is December through February, summer is June through August, and the
    remaining months are shoulder. These are the seasons the revised evaluation
    plan reports against.
    """
    season = pd.Series("shoulder", index=month.index, dtype="object")
    season[month.isin([12, 1, 2])] = "winter"
    season[month.isin([6, 7, 8])] = "summer"
    return season


#: Delivery time zones, mirroring ``CONFIG["data"]["market_timezones"]``.
MARKET_TIMEZONES = {
    "NYISO": "America/New_York",
    "ISONE": "America/New_York",
    "PJM": "America/New_York",
    "CAISO": "America/Los_Angeles",
    "ERCOT": "America/Chicago",
}


def add_season(
    frame: pd.DataFrame,
    timestamp_column: str = "timestamp_utc",
    iso_column: str = "iso",
) -> pd.DataFrame:
    """Attach ``season``, ``delivery_month_local``, and ``delivery_hour_local``.

    Seasons and hours are derived from *market-local* delivery time, so a summer
    evening means the same thing in ISO-NE and NYISO despite the markets sitting
    in different zones.

    Local time is reconstructed from the UTC timestamp and the market's zone
    rather than read from the panel's ``delivery_timestamp_local`` column. That
    column mixes UTC offsets across the daylight-saving boundary, so pandas
    parses it to ``object`` dtype and the ``.dt`` accessors fail on it.
    """
    result = frame.copy()
    utc = pd.to_datetime(result[timestamp_column], utc=True)

    month = pd.Series(pd.NA, index=result.index, dtype="Int64")
    hour = pd.Series(pd.NA, index=result.index, dtype="Int64")

    for iso, group_index in result.groupby(iso_column, sort=False).groups.items():
        zone = MARKET_TIMEZONES.get(iso)
        if zone is None:
            raise KeyError(
                f"No delivery time zone configured for market {iso!r}. "
                f"Known markets: {sorted(MARKET_TIMEZONES)}"
            )
        local = utc.loc[group_index].dt.tz_convert(zone)
        month.loc[group_index] = local.dt.month.astype("Int64")
        hour.loc[group_index] = local.dt.hour.astype("Int64")

    result["delivery_month_local"] = month
    result["delivery_hour_local"] = hour
    result["season"] = assign_season(month)
    return result


# ---------------------------------------------------------------------
# Task 1 — magnitude thresholds and labels
# ---------------------------------------------------------------------

def build_magnitude_thresholds(
    labelled: pd.DataFrame,
    tail_probability: float = 0.05,
    scope: Sequence[str] = THRESHOLD_SCOPE,
    spread_column: str = "dart_spread",
) -> pd.DataFrame:
    """Estimate the ``|DART|`` threshold for each fold, market, and node.

    The threshold is the ``1 - tail_probability`` quantile of ``|DART|`` over
    **training rows only**. It is returned as a table rather than applied
    directly so that the estimate can be inspected, frozen, and audited before
    any evaluation row sees it.

    Args:
        labelled: Fold-expanded rows carrying ``split_role`` and the spread.
        tail_probability: Target event rate, e.g. ``0.05`` for the top 5%.
        scope: Columns defining an independent threshold.
        spread_column: Signed DART spread column.

    Returns:
        One row per scope group with the threshold and its training diagnostics.
    """
    if not 0.0 < tail_probability < 1.0:
        raise ValueError(f"tail_probability must be in (0, 1), got {tail_probability}")

    training = labelled[labelled["split_role"] == TRAINING_SPLIT_ROLE]
    if training.empty:
        raise ValueError(
            "No training rows found. Expected split_role == "
            f"{TRAINING_SPLIT_ROLE!r} in the input frame."
        )

    records: list[dict] = []

    for keys, group in training.groupby(list(scope), sort=True):
        absolute = group[spread_column].abs().dropna()
        threshold = float(absolute.quantile(1.0 - tail_probability, interpolation="linear"))
        exceeding = int((absolute >= threshold).sum())

        record = dict(zip(scope, keys if isinstance(keys, tuple) else (keys,)))
        record.update(
            {
                "tail_probability": tail_probability,
                "magnitude_threshold": threshold,
                "quantile_method": "linear",
                "threshold_source": "fold_training_rows_only",
                "training_row_count": int(len(absolute)),
                "training_event_count": exceeding,
                "training_event_rate": exceeding / len(absolute) if len(absolute) else np.nan,
                # A large tie count means the threshold sits on a repeated value
                # and the realised rate will overshoot the target.
                "threshold_tie_count": int((absolute == threshold).sum()),
                "training_minimum_timestamp_utc": group["timestamp_utc"].min(),
                "training_maximum_timestamp_utc": group["timestamp_utc"].max(),
            }
        )
        records.append(record)

    return pd.DataFrame.from_records(records)


def build_magnitude_labels(
    labelled: pd.DataFrame | None = None,
    tail_probability: float = 0.05,
    scope: Sequence[str] = THRESHOLD_SCOPE,
    spread_column: str = "dart_spread",
    standardize: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach the Task 1 magnitude target to fold-expanded rows.

    The label is ``1{|DART| >= threshold}`` where the threshold comes from
    :func:`build_magnitude_thresholds` — training rows only, frozen before it
    touches any evaluation row.

    Args:
        labelled: Fold-expanded rows. Defaults to the collaborator's signed-tail
            label table, which already carries folds, splits, and spreads.
        tail_probability: Target event rate.
        scope: Columns defining an independent threshold.
        spread_column: Signed DART spread column.
        standardize: Threshold the *standardized* spread instead of the raw one.
            The scale is a train-only robust spread per market × season × hour,
            which makes the event definition comparable across markets and
            regimes rather than dominated by absolute price level.

    Returns:
        ``(labelled_rows, thresholds)``. The rows gain ``abs_dart_spread``,
        ``magnitude_threshold``, and ``label_high_spread``.
    """
    if labelled is None:
        labelled = cache.load_signed_tail_labels()

    frame = add_season(labelled)
    frame["abs_dart_spread"] = frame[spread_column].abs()

    target_column = spread_column

    if standardize:
        frame, scales = _apply_robust_scale(frame, spread_column=spread_column)
        target_column = "dart_spread_standardized"
        frame["abs_dart_spread_standardized"] = frame[target_column].abs()
    else:
        scales = None

    thresholds = build_magnitude_thresholds(
        frame,
        tail_probability=tail_probability,
        scope=scope,
        spread_column=target_column,
    )

    frame = frame.merge(
        thresholds[list(scope) + ["magnitude_threshold", "tail_probability"]],
        on=list(scope),
        how="left",
        suffixes=("", "_threshold"),
    )

    if frame["magnitude_threshold"].isna().any():
        missing = frame.loc[frame["magnitude_threshold"].isna(), list(scope)].drop_duplicates()
        raise ValueError(
            "Some rows have no training-fitted threshold. Unmatched groups:\n"
            f"{missing.to_string(index=False)}"
        )

    frame["label_high_spread"] = (
        frame[target_column].abs() >= frame["magnitude_threshold"]
    ).astype(int)

    if scales is not None:
        frame.attrs["robust_scales"] = scales

    return frame, thresholds


def _apply_robust_scale(
    frame: pd.DataFrame,
    spread_column: str = "dart_spread",
    scope: Sequence[str] = ("fold_id", "iso", "node", "season", "delivery_hour_local"),
    minimum_scale: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide the spread by a train-only robust scale per market, season, hour.

    Uses the median absolute deviation rather than the standard deviation,
    because DART spreads have heavy tails and a handful of scarcity events would
    otherwise set the scale for an entire cell.

    Cells whose scale is degenerate — too few observations, or a MAD at or near
    zero — fall back to the group-wide scale rather than producing exploding
    standardized values. ``minimum_scale`` floors the divisor in dollars.
    """
    training = frame[frame["split_role"] == TRAINING_SPLIT_ROLE]

    def robust_scale(values: pd.Series) -> float:
        clean = values.dropna()
        if len(clean) < 30:
            return np.nan
        # 1.4826 rescales the MAD to be consistent with the standard deviation
        # under normality, keeping the units interpretable.
        mad = float(np.median(np.abs(clean - clean.median())) * 1.4826)
        return mad if mad > 0 else np.nan

    scales = (
        training.groupby(list(scope), sort=True)[spread_column]
        .apply(robust_scale)
        .rename("robust_scale")
        .reset_index()
    )

    # Fold-and-market-level fallback for degenerate cells.
    fallback = (
        training.groupby(["fold_id", "iso", "node"], sort=True)[spread_column]
        .apply(robust_scale)
        .rename("fallback_scale")
        .reset_index()
    )

    scales = scales.merge(fallback, on=["fold_id", "iso", "node"], how="left")
    scales["robust_scale"] = (
        scales["robust_scale"].fillna(scales["fallback_scale"]).clip(lower=minimum_scale)
    )

    result = frame.merge(scales[list(scope) + ["robust_scale"]], on=list(scope), how="left")
    result["robust_scale"] = result["robust_scale"].fillna(
        result.groupby(["fold_id", "iso", "node"])["robust_scale"].transform("median")
    )
    result["dart_spread_standardized"] = result[spread_column] / result["robust_scale"]

    return result, scales


# ---------------------------------------------------------------------
# Task 2 — conditional direction
# ---------------------------------------------------------------------

def build_direction_labels(
    frame: pd.DataFrame,
    spread_column: str = "dart_spread",
    condition_column: str = "label_high_spread",
) -> pd.DataFrame:
    """Attach the INC/DEC direction target.

    ``label_direction_binary`` is 1 for INC (``DA > RT``) and 0 for DEC. Exactly
    zero spreads are left null rather than assigned arbitrarily — they carry no
    directional information and should not inflate either class.

    ``direction_eligible`` marks the rows the conditional model actually trains
    and evaluates on: high-spread events with a non-zero spread.
    """
    result = frame.copy()

    result["label_direction_class"] = np.where(
        result[spread_column] > 0,
        INC,
        np.where(result[spread_column] < 0, DEC, None),
    )
    result["label_direction_binary"] = np.where(
        result[spread_column] > 0,
        1.0,
        np.where(result[spread_column] < 0, 0.0, np.nan),
    )

    if condition_column in result.columns:
        result["direction_eligible"] = (
            (result[condition_column] == 1) & result["label_direction_binary"].notna()
        )
    else:
        result["direction_eligible"] = result["label_direction_binary"].notna()

    return result


# ---------------------------------------------------------------------
# Episode structure
# ---------------------------------------------------------------------

#: Frozen episode definition, fixed 2026-07-19 before any model was scored.
#:
#: Contiguous high-spread hours constitute one episode. A new episode begins
#: after at least one non-event hour. Equivalently: two event hours join the
#: same episode only when they are exactly one hour apart.
#:
#: This is frozen deliberately. Episode boundaries determine both episode
#: detection recall and episode-start recall, so a definition tuned after seeing
#: model scores would let the coverage metrics be selected rather than measured.
EPISODE_MAXIMUM_GAP_HOURS = 1


def add_episode_structure(
    frame: pd.DataFrame,
    label_column: str = "label_high_spread",
    scope: Sequence[str] = ("fold_id", "iso", "node"),
    time_column: str = "timestamp_utc",
    maximum_gap_hours: int = EPISODE_MAXIMUM_GAP_HOURS,
) -> pd.DataFrame:
    """Group consecutive event hours into episodes.

    High-spread hours cluster: a single scarcity evening can produce five
    flagged hours. Scoring those as five independent successes overstates how
    many *distinct* dislocations the model found, so the evaluation reports
    episode-level coverage alongside hour-level recall.

    Uses the frozen definition in :data:`EPISODE_MAXIMUM_GAP_HOURS`.

    Adds ``episode_id`` (null for non-events) and ``is_episode_start``.
    """
    result = frame.sort_values(list(scope) + [time_column]).copy()
    result["episode_id"] = pd.Series(pd.NA, index=result.index, dtype="object")
    result["is_episode_start"] = False

    for keys, group in result.groupby(list(scope), sort=False):
        events = group[group[label_column] == 1]
        if events.empty:
            continue

        times = pd.to_datetime(events[time_column])
        gap_hours = times.diff().dt.total_seconds() / 3600.0

        # A new episode starts at the first event and after any gap wider than
        # the tolerance.
        starts = gap_hours.isna() | (gap_hours > maximum_gap_hours)
        episode_number = starts.cumsum()

        prefix = "_".join(str(key) for key in (keys if isinstance(keys, tuple) else (keys,)))
        result.loc[events.index, "episode_id"] = [
            f"{prefix}_E{number:04d}" for number in episode_number
        ]
        result.loc[events.index, "is_episode_start"] = starts.to_numpy()

    return result


# ---------------------------------------------------------------------
# Directional baselines
# ---------------------------------------------------------------------

#: Minimum training events required before a regime-conditional baseline is
#: allowed to depart from its fold's overall majority. Below this the cell is a
#: lookup table fitted to noise.
MINIMUM_REGIME_TRAINING_EVENTS = 50


def add_peak_indicator(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach ``is_peak_hour``: weekday hours ending 8 through 23, local time.

    This is the conventional on-peak block for eastern ISOs and matches the
    master pipeline's ``calendar_is_peak_hour``.
    """
    result = frame if "delivery_hour_local" in frame.columns else add_season(frame)
    result = result.copy()

    utc = pd.to_datetime(result["timestamp_utc"], utc=True)
    weekday = pd.Series(pd.NA, index=result.index, dtype="Int64")
    for iso, group_index in result.groupby("iso", sort=False).groups.items():
        local = utc.loc[group_index].dt.tz_convert(MARKET_TIMEZONES[iso])
        weekday.loc[group_index] = local.dt.dayofweek.astype("Int64")

    hour = result["delivery_hour_local"]
    result["is_peak_hour"] = ((weekday < 5) & (hour >= 7) & (hour <= 22)).fillna(False)
    return result


def build_direction_baselines(
    frame: pd.DataFrame,
    spread_column: str = "dart_spread",
    condition_column: str = "label_high_spread",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the three directional baselines on training rows and score evaluation.

    All three are fitted on fold **training** high-spread events only and then
    applied unchanged to evaluation events, exactly like the magnitude
    threshold. A baseline fitted on the evaluation period would be an oracle,
    and the fold-to-fold sign instability makes that difference large.

    The baselines, in increasing order of flexibility:

    1. ``always_inc`` — a fixed global rule. Tests the unconditional risk-premium
       story directly.
    2. ``fold_majority`` — the majority sign among each fold and market's own
       training events. The primary baseline.
    3. ``regime_majority`` — the majority sign per training season and
       peak/off-peak block. Reported as **secondary**: it has many more degrees
       of freedom and degrades into a lookup table on thin cells, so it falls
       back to the fold majority below
       :data:`MINIMUM_REGIME_TRAINING_EVENTS` training events.

    Returns:
        ``(scored, fitted)``. ``scored`` gains one prediction column per
        baseline; ``fitted`` records what each baseline learned from training.
    """
    working = add_peak_indicator(frame)

    events = working[
        (working[condition_column] == 1) & (working[spread_column] != 0)
    ].copy()
    events["is_inc"] = (events[spread_column] > 0).astype(int)

    training = events[events["split_role"] == TRAINING_SPLIT_ROLE]
    if training.empty:
        raise ValueError("No training high-spread events available to fit baselines.")

    # Baseline 1 — always INC.
    working["baseline_always_inc"] = 1.0

    # Baseline 2 — per fold and market training majority.
    fold_majority = (
        training.groupby(["fold_id", "iso"])["is_inc"]
        .agg(["mean", "size"])
        .rename(columns={"mean": "training_inc_share", "size": "training_event_count"})
        .reset_index()
    )
    fold_majority["fold_majority_is_inc"] = (
        fold_majority["training_inc_share"] >= 0.5
    ).astype(float)

    working = working.merge(
        fold_majority[["fold_id", "iso", "fold_majority_is_inc", "training_inc_share"]],
        on=["fold_id", "iso"],
        how="left",
    )
    working["baseline_fold_majority"] = working["fold_majority_is_inc"]

    # Baseline 3 — per training season and peak block, with a fallback.
    regime_keys = ["fold_id", "iso", "season", "is_peak_hour"]
    regime_majority = (
        training.groupby(regime_keys)["is_inc"]
        .agg(["mean", "size"])
        .rename(columns={"mean": "regime_inc_share", "size": "regime_event_count"})
        .reset_index()
    )
    regime_majority["regime_is_inc"] = (regime_majority["regime_inc_share"] >= 0.5).astype(
        float
    )
    # Thin cells defer to the fold majority rather than inventing a regime.
    regime_majority = regime_majority.merge(
        fold_majority[["fold_id", "iso", "fold_majority_is_inc"]],
        on=["fold_id", "iso"],
        how="left",
    )
    thin = regime_majority["regime_event_count"] < MINIMUM_REGIME_TRAINING_EVENTS
    regime_majority.loc[thin, "regime_is_inc"] = regime_majority.loc[
        thin, "fold_majority_is_inc"
    ]
    regime_majority["used_fallback"] = thin

    working = working.merge(
        regime_majority[regime_keys + ["regime_is_inc", "regime_event_count", "used_fallback"]],
        on=regime_keys,
        how="left",
    )
    # Regimes absent from training fall back to the fold majority too.
    working["baseline_regime_majority"] = working["regime_is_inc"].fillna(
        working["fold_majority_is_inc"]
    )

    fitted = fold_majority.merge(
        regime_majority[
            regime_keys + ["regime_inc_share", "regime_event_count", "regime_is_inc", "used_fallback"]
        ],
        on=["fold_id", "iso"],
        how="left",
    )

    return working, fitted


# ---------------------------------------------------------------------
# Priority 1 — target structure audit
# ---------------------------------------------------------------------

def audit_target_structure(
    frame: pd.DataFrame,
    spread_column: str = "dart_spread",
    split_role: str | None = None,
) -> pd.DataFrame:
    """Summarise the unconditional and conditional shape of the target.

    Answers the Priority 1 questions directly: how often is ``DA > RT``, does
    that hold across markets and seasons, and does it survive conditioning on a
    large absolute spread?

    Args:
        frame: Rows carrying spreads, seasons, and ``label_high_spread``.
        spread_column: Signed DART spread column.
        split_role: Restrict to one split, e.g. ``"train"``. ``None`` uses all
            rows in the frame.

    Returns:
        One row per market × season with unconditional and conditional shares.
    """
    working = frame if split_role is None else frame[frame["split_role"] == split_role]
    if "season" not in working.columns:
        working = add_season(working)

    def summarise(group: pd.DataFrame) -> pd.Series:
        spread = group[spread_column].dropna()
        events = group[group.get("label_high_spread", 0) == 1]
        event_spread = events[spread_column].dropna()

        return pd.Series(
            {
                "row_count": int(len(spread)),
                "mean_dart": float(spread.mean()),
                "median_dart": float(spread.median()),
                "mean_abs_dart": float(spread.abs().mean()),
                "inc_share_unconditional": float((spread > 0).mean()),
                "dec_share_unconditional": float((spread < 0).mean()),
                "zero_share": float((spread == 0).mean()),
                "event_count": int(len(event_spread)),
                "event_rate": float(len(event_spread) / len(spread)) if len(spread) else np.nan,
                "inc_share_conditional": (
                    float((event_spread > 0).mean()) if len(event_spread) else np.nan
                ),
                "dec_share_conditional": (
                    float((event_spread < 0).mean()) if len(event_spread) else np.nan
                ),
                # The score an always-predict-the-majority-direction rule earns
                # on high-spread events. Any directional model must beat this.
                "majority_direction_baseline": (
                    float(max((event_spread > 0).mean(), (event_spread < 0).mean()))
                    if len(event_spread)
                    else np.nan
                ),
                "mean_abs_dart_events": (
                    float(event_spread.abs().mean()) if len(event_spread) else np.nan
                ),
            }
        )

    by_market_season = (
        working.groupby(["iso", "season"], sort=True).apply(summarise, include_groups=False).reset_index()
    )
    by_market = working.groupby(["iso"], sort=True).apply(summarise, include_groups=False).reset_index()
    by_market["season"] = "all"

    return pd.concat([by_market, by_market_season], ignore_index=True).sort_values(
        ["iso", "season"]
    )


def audit_direction_balance(
    frame: pd.DataFrame,
    spread_column: str = "dart_spread",
    by: Sequence[str] = ("iso", "fold_id"),
) -> pd.DataFrame:
    """Track whether the INC/DEC balance is stable across folds and markets.

    A directional signal that is really just a persistent class imbalance will
    show a stable majority share here and little else. Instability across folds
    is itself informative: it means a majority-direction rule fitted on early
    data would not transfer forward.
    """
    if "season" not in frame.columns:
        frame = add_season(frame)

    def summarise(group: pd.DataFrame) -> pd.Series:
        spread = group[spread_column].dropna()
        events = group[group.get("label_high_spread", 0) == 1][spread_column].dropna()

        return pd.Series(
            {
                "row_count": int(len(spread)),
                "inc_share_unconditional": float((spread > 0).mean()),
                "event_count": int(len(events)),
                "inc_share_conditional": (
                    float((events > 0).mean()) if len(events) else np.nan
                ),
                "majority_direction": (
                    (INC if (events > 0).mean() >= 0.5 else DEC) if len(events) else None
                ),
            }
        )

    return frame.groupby(list(by), sort=True).apply(summarise, include_groups=False).reset_index()


# ---------------------------------------------------------------------
# Season/hour-conditional magnitude target
# ---------------------------------------------------------------------

#: Cells with fewer training observations than this fall back to the market-wide
#: location and scale rather than trusting their own order statistics.
MINIMUM_CONDITIONAL_CELL_ROWS = 60


def build_conditional_magnitude_labels(
    labelled: pd.DataFrame,
    tail_probability: float = 0.05,
    conditioning: Sequence[str] = ("season", "delivery_hour_local"),
    scope: Sequence[str] = THRESHOLD_SCOPE,
    spread_column: str = "dart_spread",
    method: str = "standardized",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Label spreads that are unusual **for that kind of hour**.

    The raw magnitude target rewards knowing which hours are typically risky. An
    August evening can exceed a market-wide quantile while being entirely
    ordinary for an August evening, so calendar features can score well without
    detecting any actual dislocation. That makes the raw target a poor instrument
    for testing whether operating mechanisms add information beyond seasonality.

    This target removes the seasonal baseline first:

    ``standardized`` — threshold the top ``tail_probability`` of

    .. math::
        z_t = \\frac{|DART_t| - \\mathrm{median}_{m,s,h}(|DART|)}
                    {\\mathrm{MAD}_{m,s,h}(|DART|)}

    ``conditional_quantile`` — label ``|DART_t|`` above the
    ``1 - tail_probability`` quantile estimated *within* each cell.

    Both estimate every statistic on fold training rows only. The standardized
    form is the primary specification: it produces a single global threshold on a
    comparable scale, whereas the conditional-quantile form fixes the event rate
    within every cell by construction and so cannot express that some hour types
    are genuinely more prone to surprises than others.

    Args:
        labelled: Fold-expanded rows with ``split_role``.
        tail_probability: Target event rate.
        conditioning: Cell definition. Season and local hour by default.
        scope: Columns defining an independent market/fold estimate.
        spread_column: Signed DART spread.
        method: ``"standardized"`` or ``"conditional_quantile"``.

    Returns:
        ``(rows, cells)``. Rows gain ``conditional_magnitude_score`` and
        ``label_high_spread_conditional``.
    """
    if method not in ("standardized", "conditional_quantile"):
        raise ValueError(f"Unknown method {method!r}")

    frame = add_season(labelled)
    frame["abs_dart_spread"] = frame[spread_column].abs()

    cell_keys = list(scope) + list(conditioning)
    training = frame[frame["split_role"] == TRAINING_SPLIT_ROLE]
    if training.empty:
        raise ValueError("No training rows available to fit conditional statistics.")

    def robust_stats(values: pd.Series) -> pd.Series:
        clean = values.dropna()
        median = clean.median()
        # 1.4826 rescales MAD to be comparable with a standard deviation.
        mad = float(np.median(np.abs(clean - median)) * 1.4826) if len(clean) else np.nan
        return pd.Series({
            "cell_median": median,
            "cell_scale": mad,
            "cell_rows": len(clean),
            "cell_quantile": clean.quantile(1.0 - tail_probability) if len(clean) else np.nan,
        })

    cells = (
        training.groupby(cell_keys, sort=True)["abs_dart_spread"]
        .apply(robust_stats).unstack().reset_index()
    )

    # Market-wide fallback for thin or degenerate cells.
    fallback = (
        training.groupby(list(scope), sort=True)["abs_dart_spread"]
        .apply(robust_stats).unstack().reset_index()
        .rename(columns={
            "cell_median": "fallback_median", "cell_scale": "fallback_scale",
            "cell_quantile": "fallback_quantile", "cell_rows": "fallback_rows",
        })
    )
    cells = cells.merge(fallback, on=list(scope), how="left")

    thin = (cells["cell_rows"] < MINIMUM_CONDITIONAL_CELL_ROWS) | (cells["cell_scale"] <= 0)
    cells["used_fallback"] = thin
    cells.loc[thin, "cell_median"] = cells.loc[thin, "fallback_median"]
    cells.loc[thin, "cell_scale"] = cells.loc[thin, "fallback_scale"]
    cells.loc[thin, "cell_quantile"] = cells.loc[thin, "fallback_quantile"]

    frame = frame.merge(
        cells[cell_keys + ["cell_median", "cell_scale", "cell_quantile", "used_fallback"]],
        on=cell_keys, how="left",
    )

    if method == "standardized":
        frame["conditional_magnitude_score"] = (
            (frame["abs_dart_spread"] - frame["cell_median"])
            / frame["cell_scale"].replace(0.0, np.nan)
        )
        # One global threshold per market and fold, on the standardized scale,
        # fitted on training rows only.
        thresholds = (
            frame[frame["split_role"] == TRAINING_SPLIT_ROLE]
            .groupby(list(scope), sort=True)["conditional_magnitude_score"]
            .quantile(1.0 - tail_probability)
            .rename("conditional_threshold").reset_index()
        )
        frame = frame.merge(thresholds, on=list(scope), how="left")
        frame["label_high_spread_conditional"] = (
            frame["conditional_magnitude_score"] >= frame["conditional_threshold"]
        ).astype(int)
    else:
        frame["conditional_magnitude_score"] = (
            frame["abs_dart_spread"] - frame["cell_quantile"]
        )
        frame["conditional_threshold"] = 0.0
        frame["label_high_spread_conditional"] = (
            frame["abs_dart_spread"] >= frame["cell_quantile"]
        ).astype(int)

    return frame, cells
