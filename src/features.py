"""Versioned feature sets and leakage-safe feature construction.

The master notebook froze a permanent 37-feature G1/G6 specification. The revised
plan replaces that with versioned sets so feature families can be added and
ablated:

* **F0** — history and calendar. The collaborator's current specification,
  preserved unchanged as the baseline every other set is measured against.
* **F1** — F0 plus load, renewables, and net load.
* **F2** — F0 plus load, renewables, congestion, scarcity, and weather.

Only F0 and the congestion part of F2 are buildable from cached data today. See
:func:`feature_set_availability` for what each set currently costs.

Every feature here obeys the same information policy as the master pipeline:
realized quantities are lagged at least 48 hours, rolling windows end at target
minus 48 hours, and nothing is imputed.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from . import cache
from .paths import MASTER

__all__ = [
    "FEATURE_SETS",
    "MARKET_FORECAST_SUPPORT",
    "MARKET_PRICE_SUPPORT",
    "SAFE_LAGS_HOURS",
    "ROLLING_WINDOWS_HOURS",
    "ROLLING_SHIFT_HOURS",
    "assemble_design_matrix",
    "build_congestion_features",
    "feature_set_availability",
    "load_baseline_feature_names",
    "shared_core_groups",
]

#: Realized lags permitted by the information policy. 1h and 24h are prohibited.
SAFE_LAGS_HOURS = (48, 72, 168)

#: Rolling window lengths, matching the master pipeline.
ROLLING_WINDOWS_HOURS = (24, 72, 168)

#: Every rolling window ends at target minus this many hours.
ROLLING_SHIFT_HOURS = 48

#: Feature-set definitions, as the feature groups each one draws on.
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "F0": ("G1", "G6"),
    "F1": ("G1", "G6", "G2", "G3"),
    "F2": ("G1", "G6", "G2", "G3", "G4", "G5"),
}

#: Feature groups buildable from data already in the cache.
BUILDABLE_GROUPS = frozenset({"G1", "G6", "G4"})


# ---------------------------------------------------------------------
# Cross-market specifications
# ---------------------------------------------------------------------
#
# Markets do not support the same variables, so forcing one dictionary on all of
# them either drops markets or silently compares different information sets.
# Two specifications are maintained instead:
#
#   shared-core       — only variables with equivalent definitions in every
#                       market in scope. Used for the transfer claim.
#   market-augmented  — each market's best available safe features. Used for
#                       within-market performance, and to price the cost of
#                       requiring a transportable information set.
#
# The gap between them is itself a result: how much accuracy is lost by
# restricting to features that travel across heterogeneous data environments?

#: What each market's price endpoint actually supplies.
#:
#: NYISO and ISO-NE use ``get_lmp``, which decomposes into energy, congestion,
#: and loss. ERCOT uses ``get_spp`` at a Trading Hub — settlement point prices,
#: which are a single undecomposed number. G4 therefore **cannot** be part of a
#: shared core that includes ERCOT.
MARKET_PRICE_SUPPORT = {
    "NYISO": {"method": "get_lmp", "lmp_decomposition": True, "rt_interval_minutes": 5},
    "ISONE": {"method": "get_lmp", "lmp_decomposition": True, "rt_interval_minutes": 5},
    # PJM replaces ERCOT as the third market (decision 2026-07-19). It uses
    # get_lmp with a decomposed schema, so unlike ERCOT it preserves G4 in the
    # shared core. Historical depth comes from Data Miner rather than a rolling
    # 30-45 day document listing. Eastern time, matching the other two markets.
    "PJM": {"method": "get_lmp", "lmp_decomposition": True, "rt_interval_minutes": 5},
    # CAISO is the third market as of 2026-07-20 — credential-free historical
    # access and a fully decomposed LMP, so it preserves G4 in the shared core.
    # RT15 rather than RT5: convergence bids settle against the 15-minute FMM
    # price, making it the economically correct real-time reference.
    "CAISO": {"method": "get_lmp", "lmp_decomposition": True, "rt_interval_minutes": 15},
    # Retained for the record. ERCOT's get_spp returns a single undecomposed
    # settlement point price, and its historical window is unreachable without
    # Public API credentials. See ERCOT_PROBE_VERDICT.md.
    "ERCOT": {"method": "get_spp", "lmp_decomposition": False, "rt_interval_minutes": 15},
}

#: Pre-cutoff forecast families each market supports, from the master pipeline's
#: publication-timing audit. NYISO and ISO-NE support *disjoint* families.
MARKET_FORECAST_SUPPORT = {
    "NYISO": {"load_forecast": True, "wind_forecast": False, "solar_forecast": False},
    "ISONE": {"load_forecast": False, "wind_forecast": True, "solar_forecast": True},
    # PJM publishes historical load forecasts with vintages, pending the same
    # publication-timing audit every other forecast family had to pass.
    "PJM": {"load_forecast": None, "wind_forecast": None, "solar_forecast": None},
    # CAISO supports all three with Publish Time vintages, but they are
    # market-specific: no forecast family exists in all three study markets, so
    # the shared spec uses EIA-930 lagged realized state instead.
    "CAISO": {"load_forecast": True, "wind_forecast": True, "solar_forecast": True},
    "ERCOT": {"load_forecast": None, "wind_forecast": None, "solar_forecast": None},
}


def shared_core_groups(markets: Sequence[str]) -> tuple[str, ...]:
    """Return the feature groups with equivalent definitions in every market.

    G1 and G6 are always available: price history and calendar need nothing but
    the price panel. G4 survives only when every market decomposes its LMP.

    NYISO, ISO-NE, and PJM all use ``get_lmp`` with a decomposed schema, so the
    three-market shared core retains G4. ERCOT would have collapsed it to G1 and
    G6 — that is, exactly the F0 baseline — which is one reason PJM replaced it
    as the third market.
    """
    unknown = [market for market in markets if market not in MARKET_PRICE_SUPPORT]
    if unknown:
        raise KeyError(f"No price support recorded for markets: {unknown}")

    groups = ["G1", "G6"]

    if all(MARKET_PRICE_SUPPORT[market]["lmp_decomposition"] for market in markets):
        groups.append("G4")

    return tuple(groups)


# ---------------------------------------------------------------------
# Baseline feature set
# ---------------------------------------------------------------------

def load_baseline_feature_names(groups: Iterable[str] = ("G1", "G6")) -> list[str]:
    """Return the master pipeline's approved model-input features.

    Reads the collaborator's ``feature_definition_registry.csv`` rather than
    hard-coding names, so the F0 baseline stays identical to theirs even if they
    revise it. Only rows flagged ``allowed`` and ``model_input`` are returned;
    the registry also lists prohibited variables such as 1h and 24h lags.
    """
    registry = cache.read_master_csv(MASTER.feature_definitions)

    selected = registry[
        registry["feature_group"].isin(list(groups))
        & registry["allowed"].astype(str).str.lower().eq("true")
        & registry["model_input"].astype(str).str.lower().eq("true")
    ]

    return sorted(selected["feature_name"].tolist())


# ---------------------------------------------------------------------
# Leakage-safe transformations
# ---------------------------------------------------------------------

def _exact_timestamp_lags(
    series: pd.Series,
    lags_hours: Sequence[int],
    name: str,
) -> pd.DataFrame:
    """Lag a series by exact wall-clock hours, not by row position.

    The series must be indexed by a gap-free hourly ``DatetimeIndex``. Positional
    shifting would silently produce a shorter lag wherever an hour is missing
    from the panel, which is exactly the kind of quiet leakage the master
    pipeline's exact-timestamp policy exists to prevent.
    """
    return pd.DataFrame(
        {f"{name}_lag_{lag}h": series.shift(lag, freq="h") for lag in lags_hours}
    )


def _shifted_rolling(
    series: pd.Series,
    windows_hours: Sequence[int],
    shift_hours: int,
    name: str,
) -> pd.DataFrame:
    """Rolling statistics over windows that end at target minus ``shift_hours``.

    Shifting before rolling — rather than after — guarantees the window closes at
    the information cutoff regardless of window length.
    """
    shifted = series.shift(shift_hours, freq="h")

    columns: dict[str, pd.Series] = {}
    for window in windows_hours:
        rolling = shifted.rolling(f"{window}h", min_periods=max(2, window // 4))
        suffix = f"{window}h_shift_{shift_hours}h"
        columns[f"{name}_roll_mean_{suffix}"] = rolling.mean()
        columns[f"{name}_roll_std_{suffix}"] = rolling.std()
        columns[f"{name}_roll_abs_mean_{suffix}"] = shifted.abs().rolling(
            f"{window}h", min_periods=max(2, window // 4)
        ).mean()

    return pd.DataFrame(columns)


# ---------------------------------------------------------------------
# G4 — congestion state
# ---------------------------------------------------------------------

def build_congestion_features(
    panel: pd.DataFrame | None = None,
    lags_hours: Sequence[int] = SAFE_LAGS_HOURS,
    windows_hours: Sequence[int] = ROLLING_WINDOWS_HOURS,
    shift_hours: int = ROLLING_SHIFT_HOURS,
) -> pd.DataFrame:
    """Build the G4 congestion feature family from the certified price panel.

    This is the only new economic feature family that needs no additional data
    ingestion: the enriched panel already carries the day-ahead and real-time LMP
    decompositions.

    Features are of three kinds:

    * lagged congestion and loss components, at the approved 48/72/168h lags;
    * rolling congestion level, volatility, and absolute level, all ending at
      target minus 48 hours;
    * congestion as a *share* of total LMP magnitude, which is the form that
      transfers across markets — a $20 congestion component means something very
      different in a $30 market than in a $300 one.

    Args:
        panel: Enriched hourly panel. Loaded from cache when omitted.
        lags_hours: Exact-timestamp lags to apply.
        windows_hours: Rolling window lengths.
        shift_hours: Information cutoff, in hours before the target.

    Returns:
        One row per ``iso``, ``node``, ``timestamp_utc`` with the G4 features.
    """
    if panel is None:
        panel = cache.load_enriched_panel(
            columns=[
                "iso",
                "node",
                "timestamp_utc",
                "da_lmp",
                "rt_lmp",
                "da_congestion",
                "da_loss",
                "rt_congestion",
                "rt_loss",
            ]
        )

    source_columns = ("da_congestion", "da_loss", "rt_congestion", "rt_loss")
    missing = [column for column in source_columns if column not in panel.columns]
    if missing:
        raise KeyError(f"Enriched panel is missing congestion columns: {missing}")

    outputs: list[pd.DataFrame] = []

    for (iso, node), group in panel.groupby(["iso", "node"], sort=True):
        ordered = group.sort_values("timestamp_utc").copy()
        ordered["timestamp_utc"] = pd.to_datetime(ordered["timestamp_utc"], utc=True)
        ordered = ordered.set_index("timestamp_utc")

        # Reindex onto a gap-free hourly grid so that every lag below is an exact
        # wall-clock lag. Missing hours stay missing — they are never imputed.
        full_index = pd.date_range(ordered.index.min(), ordered.index.max(), freq="h", tz="UTC")
        ordered = ordered.reindex(full_index)
        ordered.index.name = "timestamp_utc"

        pieces: list[pd.DataFrame] = []

        for column in source_columns:
            pieces.append(_exact_timestamp_lags(ordered[column], lags_hours, column))

        for column in ("da_congestion", "rt_congestion"):
            pieces.append(_shifted_rolling(ordered[column], windows_hours, shift_hours, column))

        # Normalised congestion share, then lagged. Computing the ratio first and
        # lagging second keeps numerator and denominator from the same hour.
        total_magnitude = ordered["da_lmp"].abs() + ordered["rt_lmp"].abs()
        for column in ("da_congestion", "rt_congestion"):
            share = ordered[column] / total_magnitude.replace(0.0, np.nan)
            pieces.append(_exact_timestamp_lags(share, lags_hours, f"{column}_share"))

        # Persistence: how often congestion held the same sign recently. A market
        # that has been congested in the same direction all week is in a
        # structurally different state from one flipping hour to hour.
        congestion_sign = np.sign(ordered["da_congestion"]).shift(shift_hours, freq="h")
        persistence = pd.DataFrame(
            {
                f"da_congestion_sign_persistence_{window}h_shift_{shift_hours}h": (
                    congestion_sign.rolling(f"{window}h", min_periods=max(2, window // 4)).mean()
                )
                for window in windows_hours
            }
        )
        pieces.append(persistence)

        combined = pd.concat(pieces, axis=1)
        combined = combined.reindex(ordered.index)
        combined.insert(0, "node", node)
        combined.insert(0, "iso", iso)

        outputs.append(combined.reset_index())

    result = pd.concat(outputs, ignore_index=True)

    # Drop grid hours that carry no features at all, so the reindexing does not
    # inflate the panel with rows the source never had.
    feature_columns = [
        column for column in result.columns if column not in ("iso", "node", "timestamp_utc")
    ]
    result = result.dropna(subset=feature_columns, how="all").reset_index(drop=True)

    return result


def build_congestion_features_cached() -> pd.DataFrame:
    """Cached wrapper around :func:`build_congestion_features`."""
    return cache.cached_frame(
        name="g4_congestion_features",
        builder=build_congestion_features,
        dependencies=[MASTER.enriched_panel],
        version="v1",
    )


# ---------------------------------------------------------------------
# Design matrix assembly
# ---------------------------------------------------------------------

def assemble_design_matrix(
    labelled: pd.DataFrame,
    feature_set: str = "F0",
    key_columns: Sequence[str] = ("iso", "node", "timestamp_utc"),
) -> tuple[pd.DataFrame, list[str]]:
    """Join the requested feature set onto labelled rows.

    Args:
        labelled: Fold-expanded rows carrying targets and fold membership.
        feature_set: One of the keys of :data:`FEATURE_SETS`.
        key_columns: Join keys shared by the label and feature panels.

    Returns:
        ``(frame, feature_names)``. Rows with an incomplete feature vector are
        retained and flagged via ``feature_complete`` rather than dropped, so the
        caller can decide between the maximum deployable sample and the common
        evaluation rows.

    Raises:
        NotImplementedError: if the set requires feature families that are not
            yet in the cache.
    """
    if feature_set not in FEATURE_SETS:
        raise KeyError(f"Unknown feature set {feature_set!r}. Known: {sorted(FEATURE_SETS)}")

    groups = FEATURE_SETS[feature_set]
    unbuildable = [group for group in groups if group not in BUILDABLE_GROUPS]
    if unbuildable:
        raise NotImplementedError(
            f"Feature set {feature_set} requires {unbuildable}, which need the "
            "fundamentals backfill to be executed first. See "
            "feature_set_availability() for what is currently buildable."
        )

    keys = list(key_columns)
    frame = labelled.copy()
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)

    feature_names: list[str] = []

    if "G1" in groups or "G6" in groups:
        baseline_groups = tuple(group for group in ("G1", "G6") if group in groups)
        baseline_names = load_baseline_feature_names(baseline_groups)

        panel = cache.load_feature_panel(columns=keys + baseline_names)
        panel["timestamp_utc"] = pd.to_datetime(panel["timestamp_utc"], utc=True)

        frame = frame.merge(panel, on=keys, how="left")
        feature_names.extend(baseline_names)

    if "G4" in groups:
        congestion = build_congestion_features_cached()
        congestion["timestamp_utc"] = pd.to_datetime(congestion["timestamp_utc"], utc=True)

        congestion_names = [column for column in congestion.columns if column not in keys]
        frame = frame.merge(congestion, on=keys, how="left")
        feature_names.extend(congestion_names)

    frame["feature_complete"] = frame[feature_names].notna().all(axis=1)
    frame["feature_nonnull_count"] = frame[feature_names].notna().sum(axis=1)

    return frame, feature_names


# ---------------------------------------------------------------------
# Availability reporting
# ---------------------------------------------------------------------

def feature_set_availability() -> pd.DataFrame:
    """Report what each feature set costs to build right now.

    Distinguishes families that are a join away from families that require the
    540 outstanding fundamentals backfill chunks or an entirely new data source.
    This is the table to consult before planning an experiment.
    """
    group_status = {
        "G1": ("cached", "Leakage-certified price history panel exists."),
        "G6": ("cached", "Calendar features exist in the same panel."),
        "G4": (
            "cached",
            "LMP congestion and loss components are already in the enriched panel.",
        ),
        "G2": (
            "requires_backfill",
            "Load forecasts need the fundamentals backfill. ISONE load_forecast "
            "failed the publication-timing audit and is excluded entirely.",
        ),
        "G3": (
            "requires_backfill",
            "Wind and solar forecasts need the fundamentals backfill. Supported "
            "for ISONE only; NYISO has btm_solar_forecast excluded.",
        ),
        "G5": (
            "requires_new_source",
            "No outage, reserve-margin, offer-stack, or gas-price source exists "
            "in the pipeline. Fuel mix is available but is realized, not forecast.",
        ),
    }

    records = []
    for name, groups in FEATURE_SETS.items():
        statuses = [group_status[group][0] for group in groups]
        if all(status == "cached" for status in statuses):
            overall = "buildable_now"
        elif "requires_new_source" in statuses:
            overall = "blocked_on_new_data_source"
        else:
            overall = "blocked_on_backfill"

        records.append(
            {
                "feature_set": name,
                "groups": ", ".join(groups),
                "status": overall,
                "blocking_groups": ", ".join(
                    group for group in groups if group_status[group][0] != "cached"
                ),
            }
        )

    sets_frame = pd.DataFrame.from_records(records)

    groups_frame = pd.DataFrame(
        [
            {"feature_group": group, "status": status, "note": note}
            for group, (status, note) in group_status.items()
        ]
    ).sort_values("feature_group")

    return sets_frame, groups_frame
